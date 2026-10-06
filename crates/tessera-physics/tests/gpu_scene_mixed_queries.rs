//! Automatic resident scene queries across mixed shapes and updated arbitrary poses.
#![cfg(feature = "gpu-contact")]
use nalgebra::{UnitQuaternion, Vector3};
use tessera_physics::{
    gpu_contact_pipeline::GpuContactDevice,
    gpu_point_query::GpuRigidPoint,
    gpu_ray_query::GpuRigidRay,
    gpu_rigid_shape::GpuRigidShape,
    gpu_rigid_sphere_contact::GpuRigidCollisionGroups,
    gpu_rigid_sphere_world::{GpuRigidSphereWorld, GpuRigidSphereWorldConfig},
    gpu_rigid_state::GpuRigidBodyState,
};
#[test]
fn mixed_scene_queries_match_analytic_geometry_after_pose_updates()
-> Result<(), Box<dyn core::error::Error>> {
    let hull = (0..8)
        .map(|i| {
            [
                if i & 1 == 0 { -0.5 } else { 0.5 },
                if i & 2 == 0 { -0.5 } else { 0.5 },
                if i & 4 == 0 { -0.5 } else { 0.5 },
            ]
        })
        .collect();
    let mut shapes = vec![
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
            vertices: vec![
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [0.0, 1.0, 0.0],
                [10.0, 0.0, 0.0],
                [11.0, 0.0, 0.0],
                [10.0, 1.0, 0.0],
            ],
            triangles: vec![[3, 4, 5], [0, 1, 2]],
        },
        GpuRigidShape::Polyline {
            vertices: vec![
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [11.0, 0.0, 0.0],
            ],
            segments: vec![[2, 3], [0, 1]],
        },
    ];

    shapes.extend((0..9).map(|_| GpuRigidShape::Sphere { radius: 0.5 }));
    let rotations = [
        UnitQuaternion::from_scaled_axis(Vector3::new(0.3_f32, -0.7, 0.2)),
        UnitQuaternion::from_scaled_axis(Vector3::new(0.2_f32, 0.4, -0.5)),
    ];
    let cases = [
        ([2.0, 0.0, 0.0], [0.5, 0.0, 0.0]),
        ([2.0, 1.0, 1.0], [0.5, 0.5, 0.5]),
        ([0.0, 0.0, 2.0], [0.0, 0.0, 1.0]),
        ([1.0, 0.0, 1.0], [0.5, 0.0, 0.5]),
        ([1.0, 0.0, 0.0], [0.4, 0.0, -0.3]),
        ([1.0, 1.0, 1.0], [0.5, 0.5, 0.5]),
        ([0.0, 0.0, 2.0], [0.0, 0.0, 0.0]),
        ([0.2, 1.0, 1.0], [0.2, 0.0, 0.0]),
    ];
    let tops = [0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.0, 0.0];
    let bodies = (0..17)
        .map(|i| GpuRigidBodyState {
            position_inverse_mass: if i < 8 {
                [i as f32 * 20.0, 0.0, 3.0, 0.0]
            } else {
                [-100.0 - i as f32 * 10.0, 100.0, 0.0, 0.0]
            },
            orientation: [0.0, 0.0, 0.0, 1.0],
            linear_velocity: [0.0; 4],
            angular_velocity: [0.0; 4],
            inverse_inertia_sleep: [0.0; 4],
        })
        .collect::<Vec<_>>();
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        tested += 1;
        eprintln!("mixed scene queries: {backend:?}");
        let mut world = GpuRigidSphereWorld::new_primitives(
            context.device(),
            context.queue(),
            &bodies,
            &shapes,
            GpuRigidSphereWorldConfig {
                gravity: [0.0; 3],
                ground_half_extent: None,
                ..Default::default()
            },
        )?;
        for (frame, rotation) in rotations.iter().enumerate() {
            let q = rotation.quaternion();
            let shift = Vector3::new(frame as f32, -2.0 * frame as f32, 0.5 * frame as f32);
            let mut rays = Vec::new();
            let mut points = Vec::new();
            let mut targets = Vec::new();
            for (i, (input, expected)) in cases.iter().enumerate() {
                let center = Vector3::new(i as f32 * 20.0, 0.0, 3.0) + shift;
                let mut state = bodies[i];
                state.position_inverse_mass = [center.x, center.y, center.z, 0.0];
                state.orientation = [q.i, q.j, q.k, q.w];
                world.write_body(i, state)?;
                rays.push(GpuRigidRay {
                    origin: (rotation * Vector3::new(0.0, 0.0, 2.0) + center).into(),
                    direction: (rotation * Vector3::new(0.0, 0.0, -2.0)).into(),
                    max_t: 1.5,
                    groups: GpuRigidCollisionGroups::default(),
                    excluded_body: None,
                    body_range: None,
                    solid: false,
                });
                points.push(GpuRigidPoint {
                    point: (rotation * Vector3::from(*input) + center).into(),
                    max_distance: 3.0,
                    groups: GpuRigidCollisionGroups::default(),
                    excluded_body: None,
                    body_range: None,
                    solid: false,
                });
                targets.push(rotation * Vector3::from(*expected) + center);
            }
            let mut encoder = context.device().create_command_encoder(&Default::default());
            let tree_rays = world.encode_ray_queries(&mut encoder, &rays)?;
            let tree_points = world.encode_point_queries(&mut encoder, &points)?;
            let linear_rays = world.prepare_ray_queries(&rays)?;
            let linear_points = world.prepare_point_queries(&points)?;
            linear_rays.encode(&mut encoder);
            linear_points.encode(&mut encoder);
            let _ = context.queue().submit(Some(encoder.finish()));
            let ray_hits = tree_rays.readback(context.device(), context.queue())?;
            let point_hits = tree_points.readback(context.device(), context.queue())?;
            let reference_rays = linear_rays.readback(context.device(), context.queue())?;
            let reference_points = linear_points.readback(context.device(), context.queue())?;
            for i in 0..8 {
                let r = ray_hits[i];
                let p = point_hits[i];
                assert_eq!(r.ids, reference_rays[i].ids);
                assert_eq!(p.ids, reference_points[i].ids);
                assert_eq!(r.point_toi, reference_rays[i].point_toi);
                assert_eq!(p.point_distance, reference_points[i].point_distance);
                assert_eq!(
                    r.ids,
                    [i as u32, u32::from(i >= 6), 1, 0],
                    "{backend:?}: frame={frame} ray={i}: {r:?}"
                );
                assert_eq!(
                    p.ids,
                    [i as u32, u32::from(i >= 6), 1, 0],
                    "{backend:?}: frame={frame} point={i}: {p:?}"
                );
                assert!(
                    (r.point_toi[3] - (2.0 - tops[i]) / 2.0).abs() < 2e-4,
                    "{backend:?}: frame={frame} ray={i}: {r:?}"
                );
                assert!(
                    (Vector3::new(
                        p.point_distance[0],
                        p.point_distance[1],
                        p.point_distance[2]
                    ) - targets[i])
                        .norm()
                        < 2e-4,
                    "{backend:?}: frame={frame} point={i}: {p:?}, expected={:?}",
                    targets[i]
                );
                assert!(
                    (p.point_distance[3]
                        - (Vector3::from(cases[i].0) - Vector3::from(cases[i].1)).norm())
                    .abs()
                        < 2e-4
                );
            }
        }
    }
    assert!(tested > 0);
    Ok(())
}
