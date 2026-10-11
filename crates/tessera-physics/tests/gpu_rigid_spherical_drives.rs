//! Physical drive budgets and quaternion-tangent spherical rows on real devices.
#![cfg(feature = "gpu-contact")]

use nalgebra::{UnitQuaternion, Vector3};
use tessera_physics::{
    articulated_world::MotorModel,
    gpu_contact_pipeline::GpuContactDevice,
    gpu_rigid_ball_joint::{
        GpuRigidAxisMotor, GpuRigidAxisServo, GpuRigidBallJoint, GpuRigidBallJointSolver,
        GpuRigidPrismaticJoint, GpuRigidRevoluteJoint, GpuRigidSphericalLimit,
        GpuRigidTemporalJointParams,
    },
    gpu_rigid_state::{GpuRigidBodyState, GpuRigidStateSession},
};

fn body(inverse: f32) -> GpuRigidBodyState {
    GpuRigidBodyState {
        position_inverse_mass: [0.0, 0.0, 0.0, inverse],
        orientation: [0.0, 0.0, 0.0, 1.0],
        linear_velocity: [0.0; 4],
        angular_velocity: [0.0; 4],
        inverse_inertia_sleep: [inverse, inverse, inverse, 0.0],
    }
}

fn ball() -> GpuRigidBallJoint {
    GpuRigidBallJoint {
        body_a: 0,
        body_b: 1,
        local_anchor_a: [0.0; 3],
        local_anchor_b: [0.0; 3],
    }
}

fn solve(
    context: &GpuContactDevice,
    solver: &mut GpuRigidBallJointSolver,
    temporal: bool,
    iterations: u32,
) -> Result<(), Box<dyn core::error::Error>> {
    let mut encoder = context.device().create_command_encoder(&Default::default());
    if temporal {
        let step = solver.prepare_temporal(
            context.device(),
            &mut encoder,
            0.01,
            GpuRigidTemporalJointParams {
                iterations,
                ..Default::default()
            },
        )?;
        step.encode_capture_angles(&mut encoder);
        step.encode_capture_drives(&mut encoder);
        step.encode_warm(&mut encoder);
        step.encode_bias(&mut encoder);
    } else {
        solver.encode(context.queue(), &mut encoder, 0.01, iterations, 0.2)?;
    }
    let _ = context.queue().submit(Some(encoder.finish()));
    Ok(())
}

#[test]
fn rigid_axis_models_preserve_acceleration_and_physical_caps()
-> Result<(), Box<dyn core::error::Error>> {
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        tested += 1;
        eprintln!("rigid axis models: {backend:?}");
        for kind in 0..3 {
            let session = GpuRigidStateSession::new(
                context.device(),
                context.queue(),
                &[body(0.0), body(1.0)],
            )?;
            let balls = if kind == 0 { vec![ball()] } else { vec![] };
            let hinges = if kind == 1 {
                vec![GpuRigidRevoluteJoint {
                    body_a: 0,
                    body_b: 1,
                    local_anchor_a: [0.0; 3],
                    local_anchor_b: [0.0; 3],
                    local_axis_a: [0.0, 0.0, 1.0],
                    local_axis_b: [0.0, 0.0, 1.0],
                }]
            } else {
                vec![]
            };
            let sliders = if kind == 2 {
                vec![GpuRigidPrismaticJoint {
                    body_a: 0,
                    body_b: 1,
                    local_anchor_a: [0.0; 3],
                    local_anchor_b: [0.0; 3],
                    local_rotation_a: [0.0, 0.0, 0.0, 1.0],
                    local_rotation_b: [0.0, 0.0, 0.0, 1.0],
                }]
            } else {
                vec![]
            };
            let mut solver = GpuRigidBallJointSolver::new_with_prismatic(
                context.device(),
                session.state_buffer(),
                2,
                &[0; 2],
                &balls,
                &[],
                &hinges,
                &sliders,
            )?;
            for temporal in [false, true] {
                for inverse in [0.25, 1.0, 4.0] {
                    for model in [MotorModel::ForceBased, MotorModel::AccelerationBased] {
                        for cap in [0.0, 2.0, f32::INFINITY] {
                            session.write_body(context.queue(), 0, body(0.0))?;
                            session.write_body(context.queue(), 1, body(inverse))?;
                            solver.clear_cache(context.queue());
                            let servo = GpuRigidAxisServo {
                                model,
                                position_target: 1.0,
                                velocity_target: 0.0,
                                stiffness: 10.0,
                                damping: if temporal { 20.0 } else { 0.0 },
                                max_force: cap,
                            };
                            match kind {
                                0 => solver.set_spherical_servo(
                                    context.queue(),
                                    0,
                                    2,
                                    Some(servo),
                                )?,
                                1 => solver.set_revolute_servo(context.queue(), 0, Some(servo))?,
                                _ => solver.set_prismatic_servo(context.queue(), 0, Some(servo))?,
                            }
                            solve(&context, &mut solver, temporal, 8)?;
                            let state = session.readback(context.device(), context.queue())?[1];
                            let gain_mass = if model == MotorModel::AccelerationBased {
                                1.0
                            } else {
                                inverse
                            };
                            let expected = (0.1 * gain_mass
                                / (1.0 + 0.01 * servo.damping * gain_mass))
                                .min(cap * 0.01 * inverse);
                            let actual = if kind == 2 {
                                state.linear_velocity[2]
                            } else {
                                state.angular_velocity[2]
                            };
                            assert!(
                                (actual - expected).abs() < 2e-5,
                                "{backend:?}: kind={kind} temporal={temporal} inverse={inverse} model={model:?} cap={cap}: {actual} != {expected}"
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

#[test]
fn spherical_axes_use_quaternion_tangents_and_physical_torque_limits()
-> Result<(), Box<dyn core::error::Error>> {
    let frame = UnitQuaternion::<f32>::from_scaled_axis(Vector3::new(-0.3, 0.4, 0.2));
    let rotation = Vector3::new(0.7, -0.5, 0.9);
    let relative = UnitQuaternion::from_scaled_axis(rotation);
    let orientation = frame * relative;
    let inertia = Vector3::new(0.3, 1.7, 2.5);
    let dt = 0.01;
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        tested += 1;
        eprintln!("spherical tangent rows: {backend:?}");
        let mut fixed = body(0.0);
        fixed.orientation = frame.coords.into();
        let mut moving = body(1.0);
        moving.orientation = orientation.coords.into();
        moving.inverse_inertia_sleep = [inertia.x, inertia.y, inertia.z, 0.0];
        let session =
            GpuRigidStateSession::new(context.device(), context.queue(), &[fixed, moving])?;
        let mut solver = GpuRigidBallJointSolver::new(
            context.device(),
            session.state_buffer(),
            2,
            &[0; 2],
            &[ball()],
        )?;
        assert_eq!(solver.spherical_motor(0, 3), None);
        assert_eq!(solver.spherical_servo(1, 0), None);
        assert!(
            solver
                .set_spherical_motor(context.queue(), 0, 3, None)
                .is_err()
        );
        for invalid in [f32::NAN, f32::NEG_INFINITY, -1.0] {
            assert!(
                solver
                    .set_spherical_motor(
                        context.queue(),
                        0,
                        0,
                        Some(GpuRigidAxisMotor {
                            target_velocity: 1.0,
                            max_force: invalid
                        })
                    )
                    .is_err()
            );
        }
        for axis in 0..3 {
            // Differentiate quaternion motion independently of the shader's
            // analytic Jacobian, including the noncommuting mixed-axis pose.
            let mut gradient = Vector3::zeros();
            let epsilon = 1e-3;
            for direction in 0..3 {
                let mut delta = Vector3::zeros();
                delta[direction] = epsilon;
                let plus =
                    (frame.inverse() * UnitQuaternion::from_scaled_axis(delta) * orientation)
                        .scaled_axis();
                let minus =
                    (frame.inverse() * UnitQuaternion::from_scaled_axis(-delta) * orientation)
                        .scaled_axis();
                gradient[direction] = (plus[axis] - minus[axis]) / (2.0 * epsilon);
            }
            let response =
                orientation * ((orientation.inverse() * gradient).component_mul(&inertia));
            let projected_inverse = gradient.dot(&response);
            for sign in [-1.0, 1.0] {
                let mut initial = moving;
                for value in &mut initial.orientation {
                    *value *= sign;
                }
                for temporal in [false, true] {
                    for cap in [0.0, 2.0, f32::INFINITY] {
                        session.write_body(context.queue(), 1, initial)?;
                        solver.clear_cache(context.queue());
                        let servo = GpuRigidAxisServo {
                            model: MotorModel::AccelerationBased,
                            position_target: rotation[axis] + 1.0,
                            velocity_target: 0.0,
                            stiffness: 10.0,
                            damping: 0.0,
                            max_force: cap,
                        };
                        solver.set_spherical_servo(context.queue(), 0, axis, Some(servo))?;
                        assert_eq!(solver.spherical_servo(0, axis), Some(Some(servo)));
                        solve(&context, &mut solver, temporal, 8)?;
                        let state = session.readback(context.device(), context.queue())?[1];
                        let actual = Vector3::new(
                            state.angular_velocity[0],
                            state.angular_velocity[1],
                            state.angular_velocity[2],
                        );
                        let impulse = (0.1 / projected_inverse).min(cap * dt / gradient.norm());
                        assert!(
                            (actual - response * impulse).norm() < 2e-4,
                            "{backend:?}: axis={axis} sign={sign} temporal={temporal} cap={cap}: {actual:?}"
                        );
                    }
                    solver.set_spherical_servo(context.queue(), 0, axis, None)?;
                    for cap in [0.0, f32::INFINITY] {
                        session.write_body(context.queue(), 1, initial)?;
                        solver.clear_cache(context.queue());
                        let motor = GpuRigidAxisMotor {
                            target_velocity: 0.4,
                            max_force: cap,
                        };
                        solver.set_spherical_motor(context.queue(), 0, axis, Some(motor))?;
                        assert_eq!(solver.spherical_motor(0, axis), Some(Some(motor)));
                        solve(&context, &mut solver, temporal, 8)?;
                        let state = session.readback(context.device(), context.queue())?[1];
                        let actual = Vector3::new(
                            state.angular_velocity[0],
                            state.angular_velocity[1],
                            state.angular_velocity[2],
                        );
                        let expected = if cap == 0.0 { 0.0 } else { 0.4 };
                        assert!((gradient.dot(&actual) - expected).abs() < 2e-4);
                    }
                    solver.set_spherical_motor(context.queue(), 0, axis, None)?;
                }
            }
            for upper in [false, true] {
                session.write_body(context.queue(), 1, moving)?;
                solver.clear_cache(context.queue());
                let limit = if upper {
                    GpuRigidSphericalLimit {
                        min: rotation[axis] - 1.0,
                        max: rotation[axis] - 0.1,
                    }
                } else {
                    GpuRigidSphericalLimit {
                        min: rotation[axis] + 0.1,
                        max: rotation[axis] + 1.0,
                    }
                };
                solver.set_spherical_limit(context.queue(), 0, axis, Some(limit))?;
                assert_eq!(solver.spherical_limit(0, axis), Some(Some(limit)));
                solve(&context, &mut solver, false, 1)?;
                let state = session.readback(context.device(), context.queue())?[1];
                let actual = Vector3::new(
                    state.angular_velocity[0],
                    state.angular_velocity[1],
                    state.angular_velocity[2],
                );
                let expected_rate = if upper { -2.0 } else { 2.0 };
                assert!(
                    (actual - response * (expected_rate / projected_inverse)).norm() < 8e-4,
                    "{backend:?}: axis={axis} upper={upper}: {actual:?}"
                );
                solver.set_spherical_limit(context.queue(), 0, axis, None)?;
            }
        }
    }
    assert!(tested > 0, "no GPU backend available");
    Ok(())
}
