//! Signed speculative distances for triangle and segment BVH leaves.
#![cfg(feature = "gpu-contact")]
use tessera_physics::{
    gpu_broad_phase::GpuPair,
    gpu_contact_pipeline::GpuContactDevice,
    gpu_rigid_shape::GpuRigidShape,
    gpu_rigid_sphere_contact::GpuRigidSphereContacts,
    gpu_rigid_state::{GpuRigidBodyState, GpuRigidStateSession},
};
fn mesh() -> GpuRigidShape {
    let mut vertices = vec![[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]];
    let mut triangles = vec![[0, 1, 2]];
    for i in 0..32 {
        let first = vertices.len() as u32;
        let x = 10.0 + 3.0 * i as f32;
        vertices.extend([[x, -1.0, 0.0], [x + 1.0, -1.0, 0.0], [x, 1.0, 0.0]]);
        triangles.push([first, first + 1, first + 2]);
    }
    GpuRigidShape::TriangleMesh {
        vertices,
        triangles,
    }
}
fn line() -> GpuRigidShape {
    let mut vertices = vec![[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let mut segments = vec![[0, 1]];
    for i in 0..32 {
        let first = vertices.len() as u32;
        let x = 10.0 + 3.0 * i as f32;
        vertices.extend([[x, 0.0, 0.0], [x + 1.0, 0.0, 0.0]]);
        segments.push([first, first + 1]);
    }
    GpuRigidShape::Polyline { vertices, segments }
}
fn body(x: f32, z: f32) -> GpuRigidBodyState {
    GpuRigidBodyState {
        position_inverse_mass: [x, 0.0, z, 1.0],
        orientation: [0.0, 0.0, 0.0, 1.0],
        linear_velocity: [0.0; 4],
        angular_velocity: [0.0; 4],
        inverse_inertia_sleep: [1.0, 1.0, 1.0, 0.0],
    }
}
#[test]
fn surface_bvh_distances_match_gaps_and_skip_distant_leaves()
-> Result<(), Box<dyn core::error::Error>> {
    let sphere = GpuRigidShape::Sphere { radius: 0.5 };
    let capsule = GpuRigidShape::Capsule {
        radius: 0.5,
        half_length: 0.3,
    };
    let cases = [
        (mesh(), sphere.clone(), 0.0, 0.55, 0.025),
        (line(), sphere.clone(), 0.0, 0.55, 0.025),
        (mesh(), mesh(), 0.0, 0.05, 0.025),
        (line(), line(), 0.0, 0.05, 0.025),
        (mesh(), line(), 0.0, 0.05, 0.025),
        (sphere, mesh(), -0.55, 0.0, -0.025),
        (mesh(), capsule.clone(), 0.0, 0.85, 0.025),
        (line(), capsule.clone(), 0.0, 0.85, 0.025),
        (capsule, mesh(), -0.85, 0.0, -0.025),
    ];
    let mut shapes = Vec::new();
    let mut bodies = Vec::new();
    let mut pairs = Vec::new();
    for (i, (a, b, za, zb, _)) in cases.iter().enumerate() {
        shapes.extend([a.clone(), b.clone()]);
        bodies.extend([body(i as f32 * 5.0, *za), body(i as f32 * 5.0, *zb)]);
        pairs.push(GpuPair {
            a: 2 * i as u32,
            b: 2 * i as u32 + 1,
        });
    }
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        tested += 1;
        eprintln!("speculative surface BVH: {backend:?}");
        let session = GpuRigidStateSession::new(context.device(), context.queue(), &bodies)?;
        let contacts = GpuRigidSphereContacts::new_with_shapes(
            context.device(),
            &session,
            &shapes,
            &pairs,
            None,
        )?;
        for rotated in [false, true] {
            if rotated {
                for (index, initial) in bodies.iter().enumerate() {
                    let [x, y, z, mass] = initial.position_inverse_mass;
                    session.write_body(
                        context.queue(),
                        index,
                        GpuRigidBodyState {
                            position_inverse_mass: [z + 2.0, y + 3.0, -x + 4.0, mass],
                            orientation: [
                                0.0,
                                core::f32::consts::FRAC_1_SQRT_2,
                                0.0,
                                core::f32::consts::FRAC_1_SQRT_2,
                            ],
                            ..*initial
                        },
                    )?;
                }
            }
            for (margin, hit) in [(0.1, true), (0.04, false)] {
                let mut encoder = context.device().create_command_encoder(&Default::default());
                let status =
                    contacts.encode_speculative_checked(context.device(), &mut encoder, margin)?;
                let _ = context.queue().submit(Some(encoder.finish()));
                let output = contacts.readback(context.device(), context.queue())?;
                if let Err(error) = status.readback(context.device(), context.queue()) {
                    panic!("{backend:?}: margin={margin}: {error}: {:?}", output.pairs);
                }
                for (i, (_, contact)) in output.pairs.iter().enumerate() {
                    assert_eq!(
                        contact.is_contact(),
                        hit,
                        "{backend:?}: case={i}: {contact:?}"
                    );
                    if hit {
                        assert!(
                            (contact.depth_hit[0] + 0.05).abs() < 3e-5,
                            "{backend:?}: case={i}: {contact:?}"
                        );
                        let axis = if rotated { 0 } else { 2 };
                        let expected_point = cases[i].4 + if rotated { 2.0 } else { 0.0 };
                        assert!(
                            (contact.point[axis] - expected_point).abs() < 3e-5,
                            "{contact:?}"
                        );
                        for component in 0..3 {
                            let expected = if component == axis { 1.0 } else { 0.0 };
                            assert!(
                                (contact.normal[component] - expected).abs() < 3e-4,
                                "{contact:?}"
                            );
                        }
                    }
                }
            }
        }
    }
    assert!(tested > 0, "no GPU backend available");
    Ok(())
}
