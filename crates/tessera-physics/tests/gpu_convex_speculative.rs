//! Separated convex pair distance witnesses on both GPU backends.
#![cfg(feature = "gpu-contact")]
use tessera_physics::{
    gpu_broad_phase::GpuPair,
    gpu_contact_pipeline::GpuContactDevice,
    gpu_rigid_shape::GpuRigidShape,
    gpu_rigid_sphere_contact::GpuRigidSphereContacts,
    gpu_rigid_state::{GpuRigidBodyState, GpuRigidStateSession},
};
fn body(p: [f32; 3]) -> GpuRigidBodyState {
    GpuRigidBodyState {
        position_inverse_mass: [p[0], p[1], p[2], 1.0],
        orientation: [0.0, 0.0, 0.0, 1.0],
        linear_velocity: [0.0; 4],
        angular_velocity: [0.0; 4],
        inverse_inertia_sleep: [1.0, 1.0, 1.0, 0.0],
    }
}
#[test]
fn convex_gap_witnesses_match_analytic_distances() -> Result<(), Box<dyn core::error::Error>> {
    let cube = (0..8)
        .map(|i| {
            [
                if i & 1 == 0 { -0.5 } else { 0.5 },
                if i & 2 == 0 { -0.5 } else { 0.5 },
                if i & 4 == 0 { -0.5 } else { 0.5 },
            ]
        })
        .collect();
    let kinds = [
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
    ];
    let mut shapes = Vec::new();
    let mut bodies = Vec::new();
    let mut pairs = Vec::new();
    for (i, kind) in kinds.iter().enumerate() {
        shapes.extend([kind.clone(), kind.clone()]);
        bodies.extend([
            body([i as f32 * 3.0, 0.0, 0.0]),
            body([i as f32 * 3.0, 0.0, 1.05]),
        ]);
        pairs.push(GpuPair {
            a: 2 * i as u32,
            b: 2 * i as u32 + 1,
        });
    }
    shapes.extend([kinds[1].clone(), kinds[1].clone()]);
    bodies.extend([body([20.0, 0.0, 0.0]), body([21.03, 1.04, 0.0])]);
    pairs.push(GpuPair { a: 12, b: 13 });
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        tested += 1;
        eprintln!("convex distance: {backend:?}");
        let session = GpuRigidStateSession::new(context.device(), context.queue(), &bodies)?;
        let contacts = GpuRigidSphereContacts::new_with_shapes(
            context.device(),
            &session,
            &shapes,
            &pairs,
            None,
        )?;
        for (reversed, rotated) in [(false, false), (true, false), (false, true), (true, true)] {
            for index in 0..bodies.len() {
                let source = if reversed { index ^ 1 } else { index };
                let mut state = bodies[source];
                if rotated {
                    let p = state.position_inverse_mass;
                    state.position_inverse_mass = [-p[1], p[0], p[2], p[3]];
                    state.orientation = [
                        0.0,
                        0.0,
                        core::f32::consts::FRAC_1_SQRT_2,
                        core::f32::consts::FRAC_1_SQRT_2,
                    ];
                }
                session.write_body(context.queue(), index, state)?;
            }
            for (margin, hit) in [(0.1, true), (0.04, false)] {
                let mut encoder = context.device().create_command_encoder(&Default::default());
                let status =
                    contacts.encode_speculative_checked(context.device(), &mut encoder, margin)?;
                let _ = context.queue().submit(Some(encoder.finish()));
                status.readback(context.device(), context.queue())?;
                let result = contacts.readback(context.device(), context.queue())?;
                for (i, (_, contact)) in result.pairs.iter().enumerate() {
                    assert_eq!(
                        contact.is_contact(),
                        hit,
                        "{backend:?}: kind={i}: {contact:?}"
                    );
                    if hit {
                        assert!(
                            (contact.depth_hit[0] + 0.05).abs() < 2e-5,
                            "{backend:?}: kind={i}: {contact:?}"
                        );
                        let normal = if i == 6 {
                            [0.6, 0.8, 0.0]
                        } else {
                            [0.0, 0.0, 1.0]
                        };
                        let normal = if rotated {
                            [-normal[1], normal[0], normal[2]]
                        } else {
                            normal
                        };
                        let normal = if reversed { normal.map(|v| -v) } else { normal };
                        for (axis, expected) in normal.iter().enumerate() {
                            assert!(
                                (contact.normal[axis] - expected).abs() < 2e-4,
                                "{contact:?}"
                            );
                        }
                        if i == 6 {
                            let point = if rotated {
                                [-0.52, 20.515]
                            } else {
                                [20.515, 0.52]
                            };
                            assert!((contact.point[0] - point[0]).abs() < 2e-5);
                            assert!((contact.point[1] - point[1]).abs() < 2e-5);
                        } else {
                            assert!((contact.point[2] - 0.525).abs() < 2e-5);
                        }
                        if i == 1 || i == 5 {
                            for extra in &result.pair_extra[i] {
                                assert!(
                                    extra.is_contact(),
                                    "missing speculative face row: {extra:?}"
                                );
                                assert!((extra.depth_hit[0] + 0.05).abs() < 2e-5);
                                assert!((extra.point[2] - 0.525).abs() < 2e-5);
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(tested > 0, "no GPU backend available");
    Ok(())
}
