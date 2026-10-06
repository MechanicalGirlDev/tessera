//! Signed speculative ground manifolds for every GPU shape kind.
#![cfg(feature = "gpu-contact")]
use tessera_physics::{
    gpu_contact_pipeline::GpuContactDevice,
    gpu_rigid_shape::GpuRigidShape,
    gpu_rigid_sphere_contact::GpuRigidSphereContacts,
    gpu_rigid_state::{GpuRigidBodyState, GpuRigidStateSession},
};
#[test]
fn all_shape_ground_rows_preserve_gap_and_margin_boundary()
-> Result<(), Box<dyn core::error::Error>> {
    let cube = (0..8)
        .map(|i| {
            [
                if i & 1 == 0 { -0.5 } else { 0.5 },
                if i & 2 == 0 { -0.5 } else { 0.5 },
                if i & 4 == 0 { -0.5 } else { 0.5 },
            ]
        })
        .collect();
    let shapes = [
        GpuRigidShape::Sphere { radius: 0.5 },
        GpuRigidShape::Box {
            half_extents: [0.5; 3],
        },
        GpuRigidShape::Capsule {
            radius: 0.25,
            half_length: 0.25,
        },
        GpuRigidShape::Cylinder {
            radius: 0.5,
            half_length: 0.5,
        },
        GpuRigidShape::Cone {
            radius: 0.5,
            half_length: 0.5,
        },
        GpuRigidShape::Convex { vertices: cube },
        GpuRigidShape::TriangleMesh {
            vertices: vec![[-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.0, 0.5, -0.5]],
            triangles: vec![[0, 1, 2]],
        },
        GpuRigidShape::Polyline {
            vertices: vec![[-0.5, 0.0, -0.5], [0.5, 0.0, -0.5]],
            segments: vec![[0, 1]],
        },
    ];
    let bodies = (0..shapes.len())
        .map(|i| GpuRigidBodyState {
            position_inverse_mass: [i as f32 * 2.0, 0.0, 0.55, 1.0],
            orientation: [0.0, 0.0, 0.0, 1.0],
            linear_velocity: [0.0; 4],
            angular_velocity: [0.0; 4],
            inverse_inertia_sleep: [1.0, 1.0, 1.0, 0.0],
        })
        .collect::<Vec<_>>();
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        tested += 1;
        eprintln!("speculative ground: {backend:?}");
        let session = GpuRigidStateSession::new(context.device(), context.queue(), &bodies)?;
        let contacts = GpuRigidSphereContacts::new_with_shapes(
            context.device(),
            &session,
            &shapes,
            &[],
            Some(30.0),
        )?;
        for (margin, expected_hit) in [(0.1, true), (0.04, false), (0.0, false)] {
            let mut encoder = context.device().create_command_encoder(&Default::default());
            contacts.encode_speculative_ground(context.device(), &mut encoder, margin)?;
            let _ = context.queue().submit(Some(encoder.finish()));
            let result = contacts.readback(context.device(), context.queue())?;
            for (i, contact) in result.ground.iter().enumerate() {
                assert_eq!(
                    contact.is_contact(),
                    expected_hit,
                    "{backend:?}: shape={i}: {contact:?}"
                );
                if expected_hit {
                    assert!(
                        (contact.depth_hit[0] + 0.05).abs() < 1e-6,
                        "shape={i}: {contact:?}"
                    );
                    assert!((contact.point[2] - 0.025).abs() < 1e-6);
                    assert_eq!(contact.normal, [0.0, 0.0, 1.0, 0.0]);
                }
            }
            for contacts in &result.ground_extra {
                for contact in contacts {
                    if contact.is_contact() {
                        assert!((contact.depth_hit[0] + 0.05).abs() < 1e-6);
                    }
                }
            }
            if !expected_hit {
                assert!(
                    result
                        .ground_extra
                        .iter()
                        .flatten()
                        .all(|c| !c.is_contact())
                );
            }
        }
    }
    assert!(tested > 0, "no GPU backend available");
    Ok(())
}
