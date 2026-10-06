//! Curved-body support by zero-thickness polyline segments on real GPU backends.
#![cfg(feature = "gpu-contact")]

use tessera_physics::{
    gpu_contact_pipeline::GpuContactDevice,
    gpu_rigid_shape::GpuRigidShape,
    gpu_rigid_sphere_solver::GpuRigidSphereSolveParams,
    gpu_rigid_sphere_world::{GpuRigidPrimitiveWorld, GpuRigidSphereWorldConfig},
    gpu_rigid_state::GpuRigidBodyState,
    sleep::SleepSettings,
};

#[test]
fn segmented_polyline_supports_caps_and_parallel_cylinder_side()
-> Result<(), Box<dyn core::error::Error>> {
    let base = GpuRigidBodyState {
        position_inverse_mass: [0.0; 4],
        orientation: [0.0, 0.0, 0.0, 1.0],
        linear_velocity: [0.0; 4],
        angular_velocity: [0.0; 4],
        inverse_inertia_sleep: [0.0; 4],
    };
    let mut vertices = vec![
        [-2.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [8.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [12.0, 0.0, 0.0],
    ];
    let mut segments = vec![[0, 1], [1, 2], [3, 4], [4, 5]];
    for index in 0..60 {
        let first = vertices.len() as u32;
        let x = 100.0 + index as f32 * 2.0;
        vertices.extend([[x, 100.0, 0.0], [x + 1.0, 100.0, 0.0]]);
        segments.push([first, first + 1]);
    }
    let shapes = [
        GpuRigidShape::Polyline { vertices, segments },
        GpuRigidShape::Cylinder {
            radius: 0.5,
            half_length: 0.5,
        },
        GpuRigidShape::Cone {
            radius: 0.5,
            half_length: 0.5,
        },
    ];
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        tested += 1;
        eprintln!("polyline support backend: {backend:?}");
        let mut states = [base; 3];
        states[1].position_inverse_mass = [0.0, 0.0, 0.4, 0.0];
        states[2].position_inverse_mass = [10.0, 0.0, 0.4, 0.0];
        let mut world = GpuRigidPrimitiveWorld::new_primitives(
            context.device(),
            context.queue(),
            &states,
            &shapes,
            GpuRigidSphereWorldConfig {
                ground_half_extent: None,
                solve: GpuRigidSphereSolveParams {
                    friction: 0.4,
                    restitution: 0.0,
                    bias_factor: 0.15,
                    iterations: 16,
                },
                sleep: SleepSettings {
                    enabled: false,
                    ..Default::default()
                },
                ..Default::default()
            },
        )?;
        for side in [false, true] {
            states = [base; 3];
            states[1].position_inverse_mass = [0.0, 0.0, 0.4, 0.0];
            states[2].position_inverse_mass = [10.0, 0.0, 0.4, 0.0];
            if side {
                let q = core::f32::consts::FRAC_1_SQRT_2;
                states[1].orientation = [0.0, q, 0.0, q];
            }
            world.reset(&states)?;
            let _ = world.step(1.0 / 120.0)?;
            let contacts = world.readback_contacts()?;
            for other in [1, 2] {
                let pair_index = contacts
                    .pairs
                    .iter()
                    .position(|(pair, _)| pair.a == 0 && pair.b == other)
                    .ok_or("missing contact candidate")?;
                let first = contacts.pairs[pair_index].1;
                let extras = contacts.pair_extra[pair_index];
                assert!(
                    first.is_contact() && extras[0].is_contact(),
                    "{backend:?}: side={side}: {first:?}: {extras:?}"
                );
                assert!(extras[1..].iter().all(|contact| !contact.is_contact()));
                for point in [first, extras[0]] {
                    assert!((point.depth_hit[0] - 0.1).abs() < 2e-3, "{point:?}");
                    assert!(point.normal[2] > 0.99, "{point:?}");
                }
                let expected_span = if other == 1 { 1.0 } else { 0.9 };
                assert!(((first.point[0] - extras[0].point[0]).abs() - expected_span).abs() < 2e-3);
            }
            for body in &mut states[1..] {
                body.position_inverse_mass[3] = 1.0;
                body.inverse_inertia_sleep = [4.0, 4.0, 4.0, 0.0];
                body.linear_velocity[2] = -1.0;
            }
            world.reset(&states)?;
            for step in 0..240 {
                let _ = world.step(1.0 / 120.0)?;
                let actual = world.readback()?;
                for body in &actual[1..] {
                    assert!(
                        body.position_inverse_mass[2] > 0.35,
                        "{backend:?}: side={side}: step={step}: {body:?}"
                    );
                    assert!(body.orientation.iter().all(|v| v.is_finite()));
                }
            }
            let actual = world.readback()?;
            for body in &actual[1..] {
                assert!(
                    (body.position_inverse_mass[2] - 0.5).abs() < 0.015,
                    "{backend:?}: side={side}: {body:?}"
                );
                for velocity in [body.linear_velocity, body.angular_velocity] {
                    let speed = velocity[..3].iter().map(|v| v * v).sum::<f32>().sqrt();
                    assert!(speed < 0.15, "{backend:?}: side={side}: {body:?}");
                }
                assert_eq!(body.inverse_inertia_sleep[3], 0.0);
            }
        }
        let mut large_states = vec![base; 3];
        large_states[1].position_inverse_mass = [0.0, 0.0, 0.4, 0.0];
        large_states[2].position_inverse_mass = [10.0, 0.0, 0.4, 0.0];
        world.reset(&large_states)?;
        for index in 0..14 {
            let mut distant = base;
            distant.position_inverse_mass = [300.0 + index as f32 * 10.0, -100.0, 0.0, 0.0];
            let added = world.append_primitive(distant, GpuRigidShape::Sphere { radius: 0.1 })?;
            assert_eq!(added, large_states.len());
            large_states.push(distant);
        }
        assert!(!world.uses_exhaustive_pairs());
        let _ = world.step(1.0 / 120.0)?;
        let contacts = world.readback_contacts()?;
        assert_eq!(contacts.pairs.len(), 2, "{backend:?}: {contacts:?}");
        for other in [1, 2] {
            let index = contacts
                .pairs
                .iter()
                .position(|(pair, _)| pair.a == 0 && pair.b == other)
                .ok_or("missing LBVH curved candidate")?;
            for point in [contacts.pairs[index].1, contacts.pair_extra[index][0]] {
                assert!(point.is_contact(), "{backend:?}: {point:?}");
                assert!((point.depth_hit[0] - 0.1).abs() < 2e-3);
                assert!(point.normal[2] > 0.99);
            }
        }
    }
    assert!(tested > 0, "no GPU backend available");
    Ok(())
}
