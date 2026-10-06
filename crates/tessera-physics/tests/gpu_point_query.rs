//! Resident GPU point projection against every shape.
#![cfg(feature = "gpu-contact")]
use tessera_physics::{
    gpu_contact_pipeline::GpuContactDevice,
    gpu_point_query::{GpuRigidPoint, GpuRigidPointQueries},
    gpu_rigid_shape::GpuRigidShape,
    gpu_rigid_sphere_contact::{GpuRigidCollisionGroups, GpuRigidSphereContacts},
    gpu_rigid_state::{GpuRigidBodyState, GpuRigidStateSession},
};
#[test]
fn projections_match_analytic_boundaries() -> Result<(), Box<dyn core::error::Error>> {
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

    let mut points = Vec::new();
    let distances = [1.5, 1.5, 1.0, 1.5, 1.5, 1.5, 2.0, 2.0];
    for (i, distance) in distances.iter().copied().enumerate() {
        let query = GpuRigidPoint {
            point: [i as f32 * 4.0, 0.0, 5.0],
            max_distance: 3.0,
            groups: GpuRigidCollisionGroups::default(),
            excluded_body: None,
            body_range: Some([i as u32, i as u32 + 1]),
            solid: false,
        };
        points.push(query);
        points.push(GpuRigidPoint {
            max_distance: distance - 0.01,
            ..query
        });
    }
    points.push(GpuRigidPoint {
        point: [0.0, 0.0, 3.0],
        max_distance: 0.0,
        solid: true,
        ..points[0]
    });
    points.push(GpuRigidPoint {
        point: [0.0, 2.0, 1.0],
        max_distance: 1.0,
        body_range: Some([0, 0]),
        ..points[0]
    });
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        tested += 1;
        eprintln!("point queries: {backend:?}");
        let session = GpuRigidStateSession::new(context.device(), context.queue(), &bodies)?;
        let contacts = GpuRigidSphereContacts::new_with_shapes(
            context.device(),
            &session,
            &shapes,
            &[],
            Some(50.0),
        )?;
        let queries = GpuRigidPointQueries::new(context.device(), &contacts, &points)?;
        let mut encoder = context.device().create_command_encoder(&Default::default());
        queries.encode(&mut encoder);
        let _ = context.queue().submit(Some(encoder.finish()));
        let hits = queries.readback(context.device(), context.queue())?;
        for (i, distance) in distances.iter().copied().enumerate() {
            let hit = hits[i * 2];
            assert_eq!(hit.ids[0], i as u32, "{backend:?}: {hit:?}");
            assert_eq!(hit.ids[2], 1);
            assert_eq!(hit.ids[1], u32::from(i >= 6));
            assert!(
                (hit.point_distance[3] - distance).abs() < 2e-5,
                "{backend:?}: {hit:?}"
            );
            assert!((hit.point_distance[2] - (5.0 - distance)).abs() < 2e-5);
            assert!((hit.normal[2] - 1.0).abs() < 2e-5, "{backend:?}: {hit:?}");
            assert!(hit.normal[0].abs() < 2e-5 && hit.normal[1].abs() < 2e-5);
            assert_eq!(hits[i * 2 + 1].ids[2], 0);
        }
        assert_eq!(hits[16].ids, [0, 0, 1, 1]);
        assert_eq!(hits[16].point_distance, [0.0, 0.0, 3.0, 0.0]);
        assert_eq!(hits[16].normal, [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(hits[17].ids, [u32::MAX, 0, 1, 0]);
        assert_eq!(hits[17].point_distance, [0.0, 2.0, 0.0, 1.0]);
        assert_eq!(hits[17].normal, [0.0, 0.0, 1.0, 0.0]);
        let rotation =
            nalgebra::UnitQuaternion::from_scaled_axis(nalgebra::Vector3::new(0.3_f32, -0.7, 0.2));
        let quat = rotation.quaternion();
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
        let mut rotated_points = Vec::new();
        let mut expected_points = Vec::new();
        for (i, (input, expected)) in cases.iter().enumerate() {
            let mut state = bodies[i];
            state.orientation = [quat.i, quat.j, quat.k, quat.w];
            session.write_body(context.queue(), i, state)?;
            let center = nalgebra::Vector3::new(i as f32 * 4.0, 0.0, 3.0);
            let local = nalgebra::Vector3::from(*input);
            let boundary = nalgebra::Vector3::from(*expected);
            rotated_points.push(GpuRigidPoint {
                point: (rotation * local + center).into(),
                ..points[i * 2]
            });
            expected_points.push((rotation * boundary + center, (local - boundary).norm()));
        }
        // An inside hollow query must keep containment while returning the boundary.
        rotated_points.push(GpuRigidPoint {
            point: (rotation * nalgebra::Vector3::new(0.1, 0.0, 0.0)
                + nalgebra::Vector3::new(20.0, 0.0, 3.0))
            .into(),
            ..points[10]
        });
        expected_points.push((
            rotation * nalgebra::Vector3::new(0.5, 0.0, 0.0)
                + nalgebra::Vector3::new(20.0, 0.0, 3.0),
            0.4,
        ));
        let posed = GpuRigidPointQueries::new(context.device(), &contacts, &rotated_points)?;
        let mut encoder = context.device().create_command_encoder(&Default::default());
        posed.encode(&mut encoder);
        let _ = context.queue().submit(Some(encoder.finish()));
        for (index, (hit, (expected, distance))) in posed
            .readback(context.device(), context.queue())?
            .iter()
            .zip(&expected_points)
            .enumerate()
        {
            assert_eq!(hit.ids[2], 1, "{backend:?}: posed={index}: {hit:?}");
            assert_eq!(hit.ids[0], if index == 8 { 5 } else { index as u32 });
            assert_eq!(hit.ids[3], u32::from(index == 8));
            assert_eq!(hit.ids[1], u32::from(index == 6 || index == 7));
            assert!(
                (nalgebra::Vector3::new(
                    hit.point_distance[0],
                    hit.point_distance[1],
                    hit.point_distance[2]
                ) - expected)
                    .norm()
                    < 5e-5,
                "{backend:?}: posed={index}: {hit:?}, expected={expected:?}"
            );
            assert!((hit.point_distance[3] - distance).abs() < 5e-5);
            let query = nalgebra::Vector3::from(rotated_points[index].point);
            let displacement = if index == 8 {
                expected - query
            } else {
                query - expected
            };
            let expected_normal = displacement.normalize();
            let actual_normal = nalgebra::Vector3::new(hit.normal[0], hit.normal[1], hit.normal[2]);
            assert!(
                (actual_normal - expected_normal).norm() < 5e-5,
                "{backend:?}: posed={index}: {hit:?}, expected_normal={expected_normal:?}"
            );
        }
        let empty = GpuRigidPointQueries::new(context.device(), &contacts, &[])?;
        assert!(
            empty
                .readback(context.device(), context.queue())?
                .is_empty()
        );
        assert!(
            GpuRigidPointQueries::new(
                context.device(),
                &contacts,
                &[GpuRigidPoint {
                    max_distance: -1.0,
                    ..points[0]
                }]
            )
            .is_err()
        );
    }
    assert!(tested > 0);
    Ok(())
}
