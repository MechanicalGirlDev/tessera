//! Separated velocity and pose integration for temporal contact solving.
#![cfg(feature = "gpu-contact")]
use tessera_physics::{
    gpu_contact_pipeline::GpuContactDevice,
    gpu_rigid_state::{GpuRigidBodyForces, GpuRigidBodyState, GpuRigidStateSession},
};
fn body() -> GpuRigidBodyState {
    GpuRigidBodyState {
        position_inverse_mass: [0.0, 0.0, 1.0, 0.5],
        orientation: [0.0, 0.0, 0.0, 1.0],
        linear_velocity: [0.0; 4],
        angular_velocity: [0.0; 4],
        inverse_inertia_sleep: [2.0, 2.0, 2.0, 0.0],
    }
}
#[test]
fn split_single_step_matches_combined_gyro_static_sleep_and_force_wake()
-> Result<(), Box<dyn core::error::Error>> {
    let mut initial = [body(); 4];
    let q = (core::f32::consts::FRAC_PI_8).sin();
    initial[0].orientation = [0.0, q, 0.0, (core::f32::consts::FRAC_PI_8).cos()];
    initial[0].inverse_inertia_sleep = [1.0, 0.5, 0.25, 0.0];
    initial[0].angular_velocity = [1.0, 2.0, 3.0, 0.0];
    initial[1].inverse_inertia_sleep[3] = 1.0;
    initial[2].inverse_inertia_sleep[3] = 1.0;
    initial[3].position_inverse_mass[3] = 0.0;
    initial[3].inverse_inertia_sleep = [0.0; 4];
    let forces = [
        GpuRigidBodyForces {
            force: [2.0, 1.0, 0.0, 0.0],
            torque: [1.0, 0.5, 2.0, 0.0],
        },
        GpuRigidBodyForces {
            force: [2.0, 0.0, 0.0, 0.0],
            torque: [0.0; 4],
        },
        GpuRigidBodyForces::default(),
        GpuRigidBodyForces {
            force: [5.0; 4],
            torque: [3.0; 4],
        },
    ];
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        tested += 1;
        eprintln!("split integration backend: {backend:?}");
        let combined = GpuRigidStateSession::new(context.device(), context.queue(), &initial)?;
        let split = GpuRigidStateSession::new(context.device(), context.queue(), &initial)?;
        for (index, force) in forces.into_iter().enumerate() {
            combined.write_forces(context.queue(), index, force)?;
            split.write_forces(context.queue(), index, force)?;
        }
        for dt in [0.01, 0.02] {
            combined.step(context.device(), context.queue(), dt, [0.0, 0.0, -9.81])?;
            let before = split.readback(context.device(), context.queue())?;
            let mut encoder = context.device().create_command_encoder(&Default::default());
            split.encode_capture_forces(context.device(), &mut encoder);
            split.encode_velocity_step(context.device(), &mut encoder, dt, [0.0, 0.0, -9.81])?;
            let _ = context.queue().submit(Some(encoder.finish()));
            let velocities = split.readback(context.device(), context.queue())?;
            for (old, current) in before.iter().zip(&velocities) {
                assert_eq!(old.position_inverse_mass, current.position_inverse_mass);
                assert_eq!(old.orientation, current.orientation);
            }
            assert_eq!(velocities[1].inverse_inertia_sleep[3], 0.0);
            assert_eq!(velocities[2].inverse_inertia_sleep[3], 1.0);
            let mut encoder = context.device().create_command_encoder(&Default::default());
            split.encode_position_step(context.device(), &mut encoder, dt)?;
            let _ = context.queue().submit(Some(encoder.finish()));
            let actual = split.readback(context.device(), context.queue())?;
            let expected = combined.readback(context.device(), context.queue())?;
            for (actual, expected) in actual.iter().zip(&expected) {
                let components = |s: &GpuRigidBodyState| {
                    [
                        s.position_inverse_mass,
                        s.orientation,
                        s.linear_velocity,
                        s.angular_velocity,
                        s.inverse_inertia_sleep,
                    ]
                };
                for (a, b) in components(actual)
                    .into_iter()
                    .flatten()
                    .zip(components(expected).into_iter().flatten())
                {
                    assert!(
                        (a - b).abs() < 2e-6,
                        "{backend:?}: {actual:?}: {expected:?}"
                    );
                }
            }
            assert_eq!(
                actual[2].position_inverse_mass,
                initial[2].position_inverse_mass
            );
            assert_eq!(
                actual[3].position_inverse_mass,
                initial[3].position_inverse_mass
            );
        }
    }
    assert!(tested > 0, "no GPU backend available");
    Ok(())
}
#[test]
fn frame_force_snapshot_spans_substeps_and_defers_new_force_until_next_capture()
-> Result<(), Box<dyn core::error::Error>> {
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        tested += 1;
        let state = GpuRigidStateSession::new(context.device(), context.queue(), &[body()])?;
        state.write_forces(
            context.queue(),
            0,
            GpuRigidBodyForces {
                force: [2.0, 0.0, 0.0, 0.0],
                torque: [0.0, 0.0, 2.0, 0.0],
            },
        )?;
        let mut capture = context.device().create_command_encoder(&Default::default());
        state.encode_capture_forces(context.device(), &mut capture);
        let _ = context.queue().submit(Some(capture.finish()));
        state.write_forces(
            context.queue(),
            0,
            GpuRigidBodyForces {
                force: [4.0, 0.0, 0.0, 0.0],
                torque: [0.0; 4],
            },
        )?;
        for frame in 0..3 {
            let mut encoder = context.device().create_command_encoder(&Default::default());
            if frame > 0 {
                state.encode_capture_forces(context.device(), &mut encoder);
            }
            for _ in 0..4 {
                state.encode_velocity_step(
                    context.device(),
                    &mut encoder,
                    0.01,
                    [0.0, 0.0, -2.0],
                )?;
                state.encode_position_step(context.device(), &mut encoder, 0.01)?;
            }
            let _ = context.queue().submit(Some(encoder.finish()));
            let current = state.readback(context.device(), context.queue())?[0];
            let expected_x = [0.001, 0.0046, 0.0094][frame];
            let expected_vx = [0.04, 0.12, 0.12][frame];
            let expected_z = [0.998, 0.9928, 0.9844][frame];
            let expected_angle = [0.004, 0.0104, 0.0168][frame];
            assert!(
                (current.position_inverse_mass[0] - expected_x).abs() < 2e-6,
                "{current:?}"
            );
            assert!((current.position_inverse_mass[2] - expected_z).abs() < 2e-6);
            assert!((current.linear_velocity[0] - expected_vx).abs() < 2e-6);
            assert!((current.linear_velocity[2] + 0.08 * (frame + 1) as f32).abs() < 2e-6);
            assert!((current.angular_velocity[2] - 0.16).abs() < 2e-6);
            assert!(
                (2.0 * current.orientation[2].atan2(current.orientation[3]) - expected_angle).abs()
                    < 2e-6
            );
        }
        let mut encoder = context.device().create_command_encoder(&Default::default());
        assert!(
            state
                .encode_velocity_step(context.device(), &mut encoder, f32::NAN, [0.0; 3])
                .is_err()
        );
        assert!(
            state
                .encode_position_step(context.device(), &mut encoder, 0.0)
                .is_err()
        );
    }
    assert!(tested > 0, "no GPU backend available");
    Ok(())
}

#[test]
fn position_integration_uses_the_constraint_corrected_velocity()
-> Result<(), Box<dyn core::error::Error>> {
    use tessera_physics::{
        gpu_rigid_contact_transport::GpuRigidContactTransport,
        gpu_rigid_sphere_contact::GpuRigidSphereContacts,
        gpu_rigid_sphere_solver::{GpuRigidSphereSolveParams, GpuRigidSphereSolver},
    };
    let mut initial = body();
    initial.linear_velocity[2] = -1.0;
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        tested += 1;
        let session = GpuRigidStateSession::new(context.device(), context.queue(), &[initial])?;
        let free = GpuRigidStateSession::new(context.device(), context.queue(), &[initial])?;
        let contacts =
            GpuRigidSphereContacts::new(context.device(), &session, &[1.0], &[], Some(10.0))?;
        let anchors = GpuRigidContactTransport::new(context.device(), &contacts)?;
        let solver = GpuRigidSphereSolver::new(context.device());
        let mut encoder = context.device().create_command_encoder(&Default::default());
        contacts.encode(&mut encoder);
        anchors.encode_capture(&mut encoder);
        session.encode_capture_forces(context.device(), &mut encoder);
        free.encode_capture_forces(context.device(), &mut encoder);
        for _ in 0..4 {
            session.encode_velocity_step(
                context.device(),
                &mut encoder,
                0.01,
                [0.0, 0.0, -9.81],
            )?;
            anchors.encode_refresh(&mut encoder);
            solver.encode(
                context.device(),
                &mut encoder,
                &contacts,
                0.01,
                GpuRigidSphereSolveParams {
                    friction: 0.0,
                    restitution: 0.0,
                    bias_factor: 0.2,
                    iterations: 4,
                },
            )?;
            session.encode_position_step(context.device(), &mut encoder, 0.01)?;
            free.encode_velocity_step(context.device(), &mut encoder, 0.01, [0.0, 0.0, -9.81])?;
            free.encode_position_step(context.device(), &mut encoder, 0.01)?;
        }
        let _ = context.queue().submit(Some(encoder.finish()));
        let supported = session.readback(context.device(), context.queue())?[0];
        let falling = free.readback(context.device(), context.queue())?[0];
        assert!(
            (supported.position_inverse_mass[2] - 1.0).abs() < 2e-6,
            "{supported:?}"
        );
        assert!(supported.linear_velocity[2].abs() < 2e-6);
        assert!(
            (falling.position_inverse_mass[2] - 0.95019).abs() < 2e-6,
            "{falling:?}"
        );
        assert!((falling.linear_velocity[2] + 1.3924).abs() < 2e-6);
    }
    assert!(tested > 0, "no GPU backend available");
    Ok(())
}
