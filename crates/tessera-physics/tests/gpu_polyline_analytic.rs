//! Zero-thickness segment contacts against analytic curved shapes.

#![cfg(feature = "gpu-contact")]

use tessera_physics::{
    gpu_contact_pipeline::GpuContactDevice,
    gpu_rigid_shape::GpuRigidShape,
    gpu_rigid_sphere_world::{GpuRigidPrimitiveWorld, GpuRigidSphereWorldConfig},
    gpu_rigid_state::GpuRigidBodyState,
};

#[test]
fn polyline_intersects_cylinder_and_cone() -> Result<(), Box<dyn core::error::Error>> {
    let state = GpuRigidBodyState {
        position_inverse_mass: [0.0; 4],
        orientation: [0.0, 0.0, 0.0, 1.0],
        linear_velocity: [0.0; 4],
        angular_velocity: [0.0; 4],
        inverse_inertia_sleep: [0.0; 4],
    };
    let line = GpuRigidShape::Polyline {
        vertices: vec![[-2.0, 0.0, 0.0], [12.0, 0.0, 0.0]],
        segments: vec![[0, 1]],
    };
    let shapes = [
        line.clone(),
        GpuRigidShape::Cylinder {
            radius: 0.5,
            half_length: 0.5,
        },
        GpuRigidShape::Cone {
            radius: 0.5,
            half_length: 0.5,
        },
        line,
    ];
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        tested += 1;
        eprintln!("polyline analytic backend: {backend:?}");
        let mut states = [state; 4];
        states[2].position_inverse_mass[0] = 10.0;
        let mut world = GpuRigidPrimitiveWorld::new_primitives(
            context.device(),
            context.queue(),
            &states,
            &shapes,
            GpuRigidSphereWorldConfig {
                gravity: [0.0; 3],
                ground_half_extent: None,
                ..Default::default()
            },
        )?;
        for rotated in [false, true] {
            let translation = if rotated { [-3.0, 5.0, 2.0] } else { [0.0; 3] };
            let to_world = |p: [f32; 3]| {
                let p = if rotated { [p[2], p[1], -p[0]] } else { p };
                [
                    p[0] + translation[0],
                    p[1] + translation[1],
                    p[2] + translation[2],
                    0.0,
                ]
            };
            for active_line in [0, 3] {
                for (positions, expected, endpoint) in [
                    (
                        [[0.0, 0.0, 0.4], [10.0, 0.0, 0.4]],
                        [Some(0.1), Some(0.1)],
                        false,
                    ),
                    (
                        [[0.0, 0.0, -0.4], [10.0, 0.0, -0.4]],
                        [Some(0.1), Some(0.1 / 5.0_f32.sqrt())],
                        false,
                    ),
                    ([[0.0, 0.0, 1.2], [10.0, 0.0, 1.2]], [None, None], false),
                    (
                        [[-2.4, 0.0, 0.0], [12.2, 0.0, 0.0]],
                        [Some(0.1), Some(0.1 / 5.0_f32.sqrt())],
                        true,
                    ),
                    (
                        [[-2.5, 0.0, 0.0], [12.25, 0.0, 0.0]],
                        [Some(0.0), Some(0.0)],
                        true,
                    ),
                    ([[-2.6, 0.0, 0.0], [12.3, 0.0, 0.0]], [None, None], true),
                ] {
                    for (index, body) in states.iter_mut().enumerate() {
                        let local = match index {
                            1 => positions[0],
                            2 => positions[1],
                            _ => [0.0, if index == active_line { 0.0 } else { 100.0 }, 0.0],
                        };
                        body.position_inverse_mass = to_world(local);
                        body.orientation = if rotated {
                            let q = core::f32::consts::FRAC_1_SQRT_2;
                            [0.0, q, 0.0, q]
                        } else {
                            [0.0, 0.0, 0.0, 1.0]
                        };
                    }
                    world.reset(&states)?;
                    let _pairs = world.step(0.01)?;
                    let contacts = world.readback_contacts()?;
                    for other in [1, 2] {
                        let (first, second) = if active_line < other {
                            (active_line, other)
                        } else {
                            (other, active_line)
                        };
                        let (pair_index, (_, point)) = contacts
                            .pairs
                            .iter()
                            .enumerate()
                            .find(|(_, (pair, _))| {
                                pair.a as usize == first && pair.b as usize == second
                            })
                            .ok_or("missing exhaustive candidate")?;
                        let expected = expected[other - 1];
                        assert_eq!(
                            point.is_contact(),
                            expected.is_some(),
                            "{backend:?}: {rotated}: {active_line}: {positions:?}: {point:?}"
                        );
                        if let Some(depth) = expected {
                            assert!((point.depth_hit[0] - depth).abs() < 2e-3, "{point:?}");
                            assert!(point.point.iter().all(|value| value.is_finite()));
                            let extra = &contacts.pair_extra[pair_index];
                            let cap = !endpoint && (other == 1 || positions[1][2] > 0.0);
                            if cap {
                                assert!(
                                    extra[0].is_contact(),
                                    "missing cap interval: {point:?}: {extra:?}"
                                );
                                let span = point
                                    .point
                                    .iter()
                                    .zip(extra[0].point.iter())
                                    .take(3)
                                    .map(|(a, b)| (a - b).powi(2))
                                    .sum::<f32>()
                                    .sqrt();
                                let expected_span = if other == 1 { 1.0 } else { 0.9 };
                                assert!((span - expected_span).abs() < 2e-3, "{span}: {extra:?}");
                                assert!((extra[0].depth_hit[0] - depth).abs() < 2e-3);
                                assert!(
                                    point
                                        .normal
                                        .iter()
                                        .zip(extra[0].normal.iter())
                                        .all(|(a, b)| (a - b).abs() < 2e-3)
                                );
                            } else {
                                assert!(!extra[0].is_contact(), "unexpected interval: {extra:?}");
                            }
                            assert!(extra[1..].iter().all(|p| !p.is_contact()));
                            let sign = if active_line < other { 1.0 } else { -1.0 };
                            let normal = [
                                point.normal[0] * sign,
                                point.normal[1] * sign,
                                point.normal[2] * sign,
                            ];
                            let witness = [
                                point.point[0] + normal[0] * point.depth_hit[0] * 0.5
                                    - translation[0],
                                point.point[1] + normal[1] * point.depth_hit[0] * 0.5
                                    - translation[1],
                                point.point[2] + normal[2] * point.depth_hit[0] * 0.5
                                    - translation[2],
                            ];
                            let witness = if rotated {
                                [-witness[2], witness[1], witness[0]]
                            } else {
                                witness
                            };
                            assert!(
                                witness[1].abs() < 2e-3 && witness[2].abs() < 2e-3,
                                "{point:?}"
                            );
                            if endpoint {
                                assert!(
                                    (witness[0] - if other == 1 { -2.0 } else { 12.0 }).abs()
                                        < 2e-3,
                                    "{point:?}"
                                );
                            } else {
                                let axial_normal = if rotated { normal[0] } else { normal[2] };
                                assert!(
                                    axial_normal * positions[other - 1][2].signum() > 0.4,
                                    "{point:?}"
                                );
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
