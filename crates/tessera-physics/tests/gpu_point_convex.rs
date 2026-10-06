//! Analytic face, edge, corner and inside projection after hull optimization.
#![cfg(feature = "gpu-contact")]
use nalgebra::{UnitQuaternion, Vector3};
use tessera_physics::{
    gpu_contact_pipeline::GpuContactDevice,
    gpu_point_query::{GpuRigidPoint, GpuRigidPointQueries},
    gpu_rigid_shape::GpuRigidShape,
    gpu_rigid_sphere_contact::{GpuRigidCollisionGroups, GpuRigidSphereContacts},
    gpu_rigid_state::{GpuRigidBodyState, GpuRigidStateSession},
};
#[test]
fn hull_support_faces_and_edges_match_transformed_cube() -> Result<(), Box<dyn core::error::Error>>
{
    let mut vertices = (0..8)
        .map(|i| {
            [
                if i & 1 == 0 { -0.5 } else { 0.5 },
                if i & 2 == 0 { -0.5 } else { 0.5 },
                if i & 4 == 0 { -0.5 } else { 0.5 },
            ]
        })
        .collect::<Vec<_>>();
    vertices.extend([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]);
    let local_center = Vector3::new(2.0, -1.0, 0.25);
    for vertex in &mut vertices {
        *vertex = (Vector3::from(*vertex) + local_center).into();
    }
    let rotation = UnitQuaternion::from_scaled_axis(Vector3::new(0.3_f32, -0.7, 0.2));
    let q = rotation.quaternion();
    let center = Vector3::new(3.0, -4.0, 7.0);
    let state = GpuRigidBodyState {
        position_inverse_mass: [3.0, -4.0, 7.0, 1.0],
        orientation: [q.i, q.j, q.k, q.w],
        linear_velocity: [0.0; 4],
        angular_velocity: [0.0; 4],
        inverse_inertia_sleep: [1.0, 1.0, 1.0, 0.0],
    };
    let cases = [
        ([1.0, 0.1, 0.2], [0.5, 0.1, 0.2], false, false),
        ([1.0, 1.0, 0.2], [0.5, 0.5, 0.2], false, false),
        ([1.0, 1.0, 1.0], [0.5, 0.5, 0.5], false, false),
        ([0.1, 0.0, 0.0], [0.5, 0.0, 0.0], true, false),
        ([0.1, 0.0, 0.0], [0.1, 0.0, 0.0], true, true),
    ];
    let points = cases
        .iter()
        .map(|(input, _, _, solid)| GpuRigidPoint {
            point: (rotation * (Vector3::from(*input) + local_center) + center).into(),
            max_distance: 3.0,
            groups: GpuRigidCollisionGroups::default(),
            excluded_body: None,
            body_range: None,
            solid: *solid,
        })
        .collect::<Vec<_>>();
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        tested += 1;
        eprintln!("hull projection: {backend:?}");
        let session = GpuRigidStateSession::new(context.device(), context.queue(), &[state])?;
        let contacts = GpuRigidSphereContacts::new_with_shapes(
            context.device(),
            &session,
            &[GpuRigidShape::Convex {
                vertices: vertices.clone(),
            }],
            &[],
            None,
        )?;
        let queries = GpuRigidPointQueries::new(context.device(), &contacts, &points)?;
        let mut encoder = context.device().create_command_encoder(&Default::default());
        queries.encode(&mut encoder);
        let _ = context.queue().submit(Some(encoder.finish()));
        for (index, (hit, (input, expected, inside, _))) in queries
            .readback(context.device(), context.queue())?
            .iter()
            .zip(&cases)
            .enumerate()
        {
            assert_eq!(
                hit.ids,
                [0, 0, 1, u32::from(*inside)],
                "{backend:?}: {index}: {hit:?}"
            );
            let target = rotation * (Vector3::from(*expected) + local_center) + center;
            assert!(
                (Vector3::new(
                    hit.point_distance[0],
                    hit.point_distance[1],
                    hit.point_distance[2]
                ) - target)
                    .norm()
                    < 5e-5,
                "{backend:?}: {index}: {hit:?}, expected={target:?}"
            );
            assert!(
                (hit.point_distance[3] - (Vector3::from(*input) - Vector3::from(*expected)).norm())
                    .abs()
                    < 5e-5
            );
        }
    }
    assert!(tested > 0);
    Ok(())
}
