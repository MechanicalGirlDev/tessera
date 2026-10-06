//! GPU ray batches against every resident shape, filtering and live poses.
#![cfg(feature = "gpu-contact")]
use tessera_physics::{
    gpu_contact_pipeline::GpuContactDevice,
    gpu_ray_query::{GpuRigidRay, GpuRigidRayQueries},
    gpu_rigid_shape::GpuRigidShape,
    gpu_rigid_sphere_contact::{GpuRigidCollisionGroups, GpuRigidSphereContacts},
    gpu_rigid_state::{GpuRigidBodyState, GpuRigidStateSession},
};
fn ray(origin: [f32; 3], direction: [f32; 3], max_t: f32) -> GpuRigidRay {
    GpuRigidRay {
        origin,
        direction,
        max_t,
        groups: GpuRigidCollisionGroups::default(),
        excluded_body: None,
        body_range: None,
        solid: false,
    }
}
#[test]
fn all_shapes_match_analytic_rays_and_observe_live_state() -> Result<(), Box<dyn core::error::Error>>
{
    let hull = (0..8)
        .map(|i| {
            [
                if i & 1 == 0 { -0.5 } else { 0.5 },
                if i & 2 == 0 { -0.5 } else { 0.5 },
                if i & 4 == 0 { -0.5 } else { 0.5 },
            ]
        })
        .collect();
    let shapes = vec![
        GpuRigidShape::Sphere { radius: 0.5 },
        GpuRigidShape::Box {
            half_extents: [0.5; 3],
        },
        GpuRigidShape::Capsule {
            radius: 0.5,
            half_length: 0.5,
        },
        GpuRigidShape::Cylinder {
            radius: 0.5,
            half_length: 0.5,
        },
        GpuRigidShape::Cone {
            radius: 0.5,
            half_length: 0.5,
        },
        GpuRigidShape::Convex { vertices: hull },
        GpuRigidShape::TriangleMesh {
            vertices: vec![[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]],
            triangles: vec![[0, 1, 2]],
        },
        GpuRigidShape::Polyline {
            vertices: vec![[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            segments: vec![[0, 1]],
        },
    ];
    let bodies = shapes
        .iter()
        .enumerate()
        .map(|(i, _)| GpuRigidBodyState {
            position_inverse_mass: [i as f32 * 4.0, 0.0, 3.0, 1.0],
            orientation: [0.0, 0.0, 0.0, 1.0],
            linear_velocity: [0.0; 4],
            angular_velocity: [0.0; 4],
            inverse_inertia_sleep: [1.0, 1.0, 1.0, 0.0],
        })
        .collect::<Vec<_>>();
    let mut rays = Vec::new();
    let mut expected = Vec::new();
    for i in 0..8 {
        let x = i as f32 * 4.0;
        let r = if i >= 6 {
            ray([x, 0.0, 5.0], [0.0, 0.0, -2.0], 2.0)
        } else {
            ray([x + 2.0, 0.0, 3.0], [-2.0, 0.0, 0.0], 2.0)
        };
        let t = if i >= 6 {
            1.0
        } else if i == 4 {
            0.875
        } else {
            0.75
        };
        rays.push(r);
        expected.push(Some((i as u32, t)));
        let mut limited = r;
        limited.max_t = t - 0.01;
        rays.push(limited);
        expected.push(None);
        if i < 6 {
            let mut inside = ray([x, 0.0, 3.0], [2.0, 0.0, 0.0], 1.0);
            inside.solid = true;
            rays.push(inside);
            expected.push(Some((i as u32, 0.0)));
        }
    }
    rays.push(ray([0.0, 2.0, 1.0], [0.0, 0.0, -1.0], 2.0));
    expected.push(Some((u32::MAX, 1.0)));
    let mut filtered = ray([2.0, 0.0, 3.0], [-2.0, 0.0, 0.0], 2.0);
    filtered.groups.memberships = 0;
    rays.push(filtered);
    expected.push(None);
    let mut excluded = ray([2.0, 0.0, 3.0], [-2.0, 0.0, 0.0], 2.0);
    excluded.excluded_body = Some(0);
    rays.push(excluded);
    expected.push(None);
    let mut ranged = ray([2.0, 0.0, 3.0], [-2.0, 0.0, 0.0], 2.0);
    ranged.body_range = Some([1, 8]);
    rays.push(ranged);
    expected.push(None);
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        tested += 1;
        eprintln!("ray queries: {backend:?}");
        let session = GpuRigidStateSession::new(context.device(), context.queue(), &bodies)?;
        let mut contacts = GpuRigidSphereContacts::new_with_shapes(
            context.device(),
            &session,
            &shapes,
            &[],
            Some(50.0),
        )?;
        let queries = GpuRigidRayQueries::new(context.device(), &contacts, &rays)?;
        let mut encoder = context.device().create_command_encoder(&Default::default());
        queries.encode(&mut encoder);
        let _ = context.queue().submit(Some(encoder.finish()));
        let hits = queries.readback(context.device(), context.queue())?;
        for (index, (hit, reference)) in hits.iter().zip(&expected).enumerate() {
            if let Some((body, t)) = reference {
                assert_eq!(hit.ids[2], 1, "{backend:?}: ray={index}: {hit:?}");
                assert_eq!(hit.ids[0], *body, "{backend:?}: ray={index}: {hit:?}");
                assert!(
                    (hit.point_toi[3] - t).abs() < 2e-5,
                    "{backend:?}: ray={index}: {hit:?}"
                );
            } else {
                assert_eq!(hit.ids[2], 0, "{backend:?}: ray={index}: {hit:?}");
            }
        }
        contacts.set_collision_groups(
            context.queue(),
            0,
            GpuRigidCollisionGroups {
                memberships: 0,
                filter: 0,
            },
        )?;
        let mut moved = bodies[1];
        moved.position_inverse_mass[0] += 1.0;
        session.write_body(context.queue(), 1, moved)?;
        let mut encoder = context.device().create_command_encoder(&Default::default());
        queries.encode(&mut encoder);
        let _ = context.queue().submit(Some(encoder.finish()));
        let live = queries.readback(context.device(), context.queue())?;
        assert_eq!(live[0].ids[2], 0);
        assert!(
            (live[3].point_toi[3] - 0.25).abs() < 2e-5,
            "{backend:?}: {:?}",
            live[3]
        );
        let empty = GpuRigidRayQueries::new(context.device(), &contacts, &[])?;
        assert!(
            empty
                .readback(context.device(), context.queue())?
                .is_empty()
        );
        let invalid = ray([0.0; 3], [0.0; 3], 1.0);
        assert!(GpuRigidRayQueries::new(context.device(), &contacts, &[invalid]).is_err());
    }
    assert!(tested > 0);
    Ok(())
}
