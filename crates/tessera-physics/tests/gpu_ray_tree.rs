//! GPU scene-tree ray casting agrees with linear search and analytic sphere geometry.
#![cfg(feature = "gpu-contact")]
use tessera_physics::{
    gpu_broad_phase::GpuAabb,
    gpu_contact_pipeline::GpuContactDevice,
    gpu_lbvh::GpuLbvh,
    gpu_ray_query::{GpuRigidRay, GpuRigidRayQueries},
    gpu_rigid_sphere_contact::{GpuRigidCollisionGroups, GpuRigidSphereContacts},
    gpu_rigid_state::{GpuRigidBodyState, GpuRigidStateSession},
};
use wgpu::util::DeviceExt;
#[test]
fn scene_tree_preserves_ray_hits_filters_and_ties() -> Result<(), Box<dyn core::error::Error>> {
    let states = (0..67)
        .map(|i| GpuRigidBodyState {
            position_inverse_mass: [((i * 23) % 66) as f32 * 4.0, 0.0, 3.0, 1.0],
            orientation: [0.0, 0.0, 0.0, 1.0],
            linear_velocity: [0.0; 4],
            angular_velocity: [0.0; 4],
            inverse_inertia_sleep: [1.0, 1.0, 1.0, 0.0],
        })
        .collect::<Vec<_>>();
    let bounds = states
        .iter()
        .map(|s| {
            let p = s.position_inverse_mass;
            GpuAabb::new([p[0] - 0.5, -0.5, 2.5], [p[0] + 0.5, 0.5, 3.5])
        })
        .collect::<Result<Vec<_>, _>>()?;
    let template = GpuRigidRay {
        origin: [2.0, 0.0, 3.0],
        max_t: 0.75,
        direction: [-2.0, 0.0, 0.0],
        groups: GpuRigidCollisionGroups::default(),
        excluded_body: None,
        body_range: None,
        solid: false,
    };
    let mut points = states
        .iter()
        .map(|s| GpuRigidRay {
            origin: [s.position_inverse_mass[0] + 0.9, 0.0, 3.0],
            ..template
        })
        .collect::<Vec<_>>();
    points.extend([
        template,
        GpuRigidRay {
            max_t: 0.74,
            ..template
        },
        GpuRigidRay {
            body_range: Some([66, 67]),
            excluded_body: Some(0),
            ..template
        },
        GpuRigidRay {
            groups: GpuRigidCollisionGroups {
                memberships: 0,
                filter: u32::MAX,
            },
            ..template
        },
        GpuRigidRay {
            origin: [0.0, 0.0, 3.0],
            solid: true,
            max_t: 0.0,
            ..template
        },
        GpuRigidRay {
            body_range: Some([0, 0]),
            ..template
        },
    ]);
    points.extend([
        GpuRigidRay {
            direction: [2.0, 0.0, 0.0],
            ..template
        },
        GpuRigidRay {
            origin: [0.0, 0.5, 3.0],
            direction: [2.0, 0.0, 0.0],
            ..template
        },
        GpuRigidRay {
            origin: [0.0, 0.5001, 3.0],
            direction: [2.0, 0.0, 0.0],
            max_t: 200.0,
            ..template
        },
        GpuRigidRay {
            origin: [1.0, 1.0, 3.0],
            direction: [-1.0, -1.0, 0.0],
            max_t: 1.0,
            ..template
        },
    ]);
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        tested += 1;
        let session = GpuRigidStateSession::new(context.device(), context.queue(), &states)?;
        let mut contacts =
            GpuRigidSphereContacts::new(context.device(), &session, &[0.5; 67], &[], None)?;
        let builder = GpuLbvh::new(context.device());
        let input = context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&bounds),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let mut encoder = context.device().create_command_encoder(&Default::default());
        let tree = builder.encode_tree_resident(context.device(), &mut encoder, &input, 67)?;
        let accelerated =
            GpuRigidRayQueries::with_scene_tree(context.device(), &contacts, &points, &tree)?;
        let linear = GpuRigidRayQueries::new(context.device(), &contacts, &points)?;
        accelerated.encode(&mut encoder);
        linear.encode(&mut encoder);
        let _ = context.queue().submit(Some(encoder.finish()));
        let hits = accelerated.readback(context.device(), context.queue())?;
        let reference = linear.readback(context.device(), context.queue())?;
        for (i, (hit, expected)) in hits.iter().zip(&reference).enumerate() {
            assert_eq!(
                hit.ids, expected.ids,
                "{backend:?}: {i}: {hit:?}, linear={expected:?}"
            );
            assert_eq!(hit.point_toi, expected.point_toi);
            assert_eq!(hit.normal, expected.normal);
        }
        for (i, hit) in hits[..67].iter().enumerate() {
            assert_eq!(hit.ids[0], if i == 66 { 0 } else { i as u32 });
            assert_eq!(hit.ids[2], 1);
            assert!((hit.point_toi[3] - 0.2).abs() < 2e-5);
            assert!((hit.point_toi[0] - states[i].position_inverse_mass[0] - 0.5).abs() < 2e-5);
        }
        assert_eq!(hits[67].ids, [0, 0, 1, 0]);
        assert_eq!(hits[67].point_toi, [0.5, 0.0, 3.0, 0.75]);
        assert_eq!(hits[68].ids[2], 0);
        assert_eq!(hits[69].ids, [66, 0, 1, 0]);
        assert_eq!(hits[70].ids[2], 0);
        assert_eq!(hits[71].ids, [0, 0, 1, 1]);
        assert_eq!(hits[72].ids[2], 0);
        assert_eq!(hits[73].ids, [23, 0, 1, 0]);
        assert_eq!(hits[73].point_toi, [3.5, 0.0, 3.0, 0.75]);
        assert_eq!(hits[73].normal, [-1.0, 0.0, 0.0, 0.0]);
        assert_eq!(hits[74].ids, [0, 0, 1, 0]);
        assert_eq!(hits[74].point_toi, [0.0, 0.5, 3.0, 0.0]);
        assert_eq!(hits[74].normal, [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(hits[75].ids[2], 0);
        assert_eq!(hits[76].ids, [0, 0, 1, 0]);
        assert!((hits[76].point_toi[3] - (1.0 - 0.5 / 2.0_f32.sqrt())).abs() < 2e-5);
        let mut moved = states[1];
        moved.position_inverse_mass[0] = 2.0;
        session.write_body(context.queue(), 1, moved)?;
        let mut moved_bounds = bounds.clone();
        moved_bounds[1] = GpuAabb::new([1.5, -0.5, 2.5], [2.5, 0.5, 3.5])?;
        let input = context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&moved_bounds),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let mut encoder = context.device().create_command_encoder(&Default::default());
        let rebuilt = builder.encode_tree_resident(context.device(), &mut encoder, &input, 67)?;
        let updated = GpuRigidRayQueries::with_scene_tree(
            context.device(),
            &contacts,
            &[template],
            &rebuilt,
        )?;
        updated.encode(&mut encoder);
        let _ = context.queue().submit(Some(encoder.finish()));
        let updated_hits = updated.readback(context.device(), context.queue())?;
        assert_eq!(updated_hits[0].ids, [1, 0, 1, 0]);
        assert_eq!(updated_hits[0].point_toi, [1.5, 0.0, 3.0, 0.25]);
        contacts.set_collision_groups(
            context.queue(),
            1,
            GpuRigidCollisionGroups {
                memberships: 0,
                filter: u32::MAX,
            },
        )?;
        let mut encoder = context.device().create_command_encoder(&Default::default());
        updated.encode(&mut encoder);
        let _ = context.queue().submit(Some(encoder.finish()));
        let filtered = updated.readback(context.device(), context.queue())?;
        assert_eq!(filtered[0].ids, [0, 0, 1, 0]);
        assert_eq!(filtered[0].point_toi, [0.5, 0.0, 3.0, 0.75]);
        let mut wrong = tree;
        wrong.collider_count = 66;
        assert!(
            GpuRigidRayQueries::with_scene_tree(context.device(), &contacts, &points, &wrong)
                .is_err()
        );
    }
    assert!(tested > 0);
    Ok(())
}
