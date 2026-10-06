//! Temporal motor budgets, implicit servo damping and unilateral joint limits.
#![cfg(feature = "gpu-contact")]
use tessera_physics::{
    gpu_contact_pipeline::GpuContactDevice,
    gpu_rigid_ball_joint::{
        GpuRigidAxisMotor, GpuRigidAxisServo, GpuRigidPrismaticJoint, GpuRigidPrismaticLimit,
        GpuRigidRevoluteJoint, GpuRigidRevoluteLimit,
    },
    gpu_rigid_sphere_solver::GpuRigidTemporalSolveParams,
    gpu_rigid_sphere_world::{GpuRigidSphereWorld, GpuRigidSphereWorldConfig},
    gpu_rigid_state::GpuRigidBodyState,
    sleep::SleepSettings,
};
fn body(z: f32, mass: f32) -> GpuRigidBodyState {
    GpuRigidBodyState {
        position_inverse_mass: [0.0, 0.0, z, mass],
        orientation: [0.0, 0.0, 0.0, 1.0],
        linear_velocity: [0.0; 4],
        angular_velocity: [0.0; 4],
        inverse_inertia_sleep: [mass, mass, mass, 0.0],
    }
}
#[test]
fn drive_impulses_and_soft_limits_match_expected_motion_on_free_axes()
-> Result<(), Box<dyn core::error::Error>> {
    let settings = GpuRigidTemporalSolveParams {
        iterations: 8,
        friction: 0.0,
        ..GpuRigidTemporalSolveParams::default()
    };
    let config = GpuRigidSphereWorldConfig {
        gravity: [0.0; 3],
        ground_half_extent: None,
        sleep: SleepSettings {
            enabled: false,
            ..SleepSettings::default()
        },
        ..GpuRigidSphereWorldConfig::default()
    };
    let fixed = body(0.0, 0.0);
    let moving = body(5.0, 1.0);
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        tested += 1;
        eprintln!("temporal drives: {backend:?}");
        for revolute in [false, true] {
            let mut world = GpuRigidSphereWorld::new(
                context.device(),
                context.queue(),
                &[fixed, moving],
                &[0.1; 2],
                config,
            )?;
            if revolute {
                world.set_revolute_joints(&[GpuRigidRevoluteJoint {
                    body_a: 0,
                    body_b: 1,
                    local_anchor_a: [0.0, 0.0, 5.0],
                    local_anchor_b: [0.0; 3],
                    local_axis_a: [0.0, 0.0, 1.0],
                    local_axis_b: [0.0, 0.0, 1.0],
                }])?;
            } else {
                world.set_prismatic_joints(&[GpuRigidPrismaticJoint {
                    body_a: 0,
                    body_b: 1,
                    local_anchor_a: [0.0; 3],
                    local_anchor_b: [0.0; 3],
                    local_rotation_a: [0.0, 0.0, 0.0, 1.0],
                    local_rotation_b: [0.0, 0.0, 0.0, 1.0],
                }])?;
            }
            let motor = Some(GpuRigidAxisMotor {
                target_velocity: 5.0,
                max_force: 2.0,
            });
            if revolute {
                world.set_revolute_motor(0, motor)?;
            } else {
                world.set_prismatic_motor(0, motor)?;
            }
            for (coordinate, velocity) in [(0.0125, 0.2), (0.045, 0.4)] {
                let _ = world.step_temporal(0.1, 4, settings)?;
                let state = world.readback()?[1];
                let actual_coordinate = if revolute {
                    world.readback_revolute_angle(0)?
                } else {
                    state.position_inverse_mass[2] - 5.0
                };
                let actual_velocity = if revolute {
                    state.angular_velocity[2]
                } else {
                    state.linear_velocity[2]
                };
                assert!(
                    (actual_coordinate - coordinate).abs() < 2e-5,
                    "{backend:?}: hinge={revolute}: {state:?}"
                );
                assert!(
                    (actual_velocity - velocity).abs() < 2e-5,
                    "{backend:?}: hinge={revolute}: {state:?}"
                );
            }
            world.reset(&[fixed, moving])?;
            let origin = if revolute { 0.0 } else { 5.0 };
            let servo = GpuRigidAxisServo {
                position_target: origin + 1.0,
                velocity_target: 0.0,
                stiffness: 100.0,
                damping: 400.0,
                max_force: 100.0,
            };
            if revolute {
                world.set_revolute_servo(0, Some(servo))?;
            } else {
                world.set_prismatic_servo(0, Some(servo))?;
            }
            let mut expected_coordinate = 0.0;
            let mut expected_velocity = 0.0;
            for _ in 0..4 {
                expected_velocity = (expected_velocity
                    + 0.025 * 100.0 * (1.0 - expected_coordinate))
                    / (1.0 + 0.025 * 400.0);
                expected_coordinate += 0.025 * expected_velocity;
            }
            let _ = world.step_temporal(0.1, 4, settings)?;
            let state = world.readback()?[1];
            let coordinate = if revolute {
                world.readback_revolute_angle(0)?
            } else {
                state.position_inverse_mass[2] - origin
            };
            let velocity = if revolute {
                state.angular_velocity[2]
            } else {
                state.linear_velocity[2]
            };
            assert!(
                (coordinate - expected_coordinate).abs() < 2e-5,
                "{backend:?}: hinge={revolute}: {state:?}"
            );
            assert!(
                (velocity - expected_velocity).abs() < 2e-5,
                "{backend:?}: hinge={revolute}: {state:?}"
            );
            world.reset(&[fixed, moving])?;
            let servo = Some(GpuRigidAxisServo {
                max_force: 2.0,
                ..servo
            });
            if revolute {
                world.set_revolute_servo(0, servo)?;
            } else {
                world.set_prismatic_servo(0, servo)?;
            }
            let _ = world.step_temporal(0.1, 4, settings)?;
            let state = world.readback()?[1];
            let velocity = if revolute {
                state.angular_velocity[2]
            } else {
                state.linear_velocity[2]
            };
            assert!((velocity - 0.2).abs() < 2e-5, "{state:?}");
            world.reset(&[fixed, moving])?;
            if revolute {
                world.set_revolute_motor(0, motor)?;
                world.set_revolute_limit(0, Some(GpuRigidRevoluteLimit { min: 0.0, max: 0.1 }))?;
            } else {
                world.set_prismatic_motor(0, motor)?;
                world
                    .set_prismatic_limit(0, Some(GpuRigidPrismaticLimit { min: 5.0, max: 5.1 }))?;
            }
            for reverse in [false, true] {
                if reverse {
                    let motor = Some(GpuRigidAxisMotor {
                        target_velocity: -5.0,
                        max_force: 2.0,
                    });
                    if revolute {
                        world.set_revolute_motor(0, motor)?;
                    } else {
                        world.set_prismatic_motor(0, motor)?;
                    }
                }
                for frame in 0..120 {
                    let _ = world.step_temporal(1.0 / 120.0, 4, settings)?;
                    let state = world.readback()?[1];
                    let coordinate = if revolute {
                        world.readback_revolute_angle(0)?
                    } else {
                        state.position_inverse_mass[2] - origin
                    };
                    assert!(
                        (-0.003..=0.103).contains(&coordinate),
                        "{backend:?}: hinge={revolute}: reverse={reverse}: frame={frame}: coordinate={coordinate}: {state:?}"
                    );
                }
                let state = world.readback()?[1];
                let coordinate = if revolute {
                    world.readback_revolute_angle(0)?
                } else {
                    state.position_inverse_mass[2] - origin
                };
                let expected = if reverse { 0.0 } else { 0.1 };
                assert!(
                    (coordinate - expected).abs() < 0.003,
                    "{backend:?}: hinge={revolute}: reverse={reverse}: {state:?}"
                );
                let velocity = if revolute {
                    state.angular_velocity[2]
                } else {
                    state.linear_velocity[2]
                };
                assert!(velocity.abs() < 0.05, "{state:?}");
            }
        }
    }
    assert!(tested > 0, "no GPU backend available");
    Ok(())
}
