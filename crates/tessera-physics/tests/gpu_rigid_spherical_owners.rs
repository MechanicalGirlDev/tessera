//! Independent rigid owner indexing, transactions, and spherical drive retention on real GPUs.
#![cfg(feature = "gpu-contact")]

use tessera_physics::{
    articulated_world::MotorModel,
    gpu_contact_pipeline::GpuContactDevice,
    gpu_rigid_ball_joint::{
        GpuRigidAxisMotor, GpuRigidAxisServo, GpuRigidBallJoint, GpuRigidFixedJoint,
        GpuRigidPrismaticJoint, GpuRigidRevoluteJoint, GpuRigidSphericalLimit,
        GpuRigidTemporalJointParams,
    },
    gpu_rigid_shape::GpuRigidShape,
    gpu_rigid_sphere_contact::GpuRigidCollisionGroups,
    gpu_rigid_sphere_solver::GpuRigidTemporalSolveParams,
    gpu_rigid_sphere_world::{
        GpuRigidPrimitiveBatch, GpuRigidPrimitiveEnvironment, GpuRigidSphereBatch,
        GpuRigidSphereEnvironment, GpuRigidSphereWorld, GpuRigidSphereWorldConfig,
    },
    gpu_rigid_state::GpuRigidBodyState,
};

type TestResult = Result<(), Box<dyn core::error::Error>>;

fn body(inverse: f32) -> GpuRigidBodyState {
    GpuRigidBodyState {
        position_inverse_mass: [0.0, 0.0, 0.0, inverse],
        orientation: [0.0, 0.0, 0.0, 1.0],
        linear_velocity: [0.0; 4],
        angular_velocity: [0.0; 4],
        inverse_inertia_sleep: [inverse, inverse, inverse, 0.0],
    }
}

fn ball(body_a: u32, body_b: u32) -> GpuRigidBallJoint {
    GpuRigidBallJoint {
        body_a,
        body_b,
        local_anchor_a: [0.0; 3],
        local_anchor_b: [0.0; 3],
    }
}

fn config() -> GpuRigidSphereWorldConfig {
    let mut config = GpuRigidSphereWorldConfig {
        gravity: [0.0; 3],
        ground_half_extent: None,
        ..Default::default()
    };
    config.sleep.enabled = false;
    config.solve.iterations = 8;
    config
}

fn servo(model: MotorModel, cap: f32) -> GpuRigidAxisServo {
    GpuRigidAxisServo {
        model,
        position_target: 1.0,
        velocity_target: 0.0,
        stiffness: 10.0,
        damping: 0.0,
        max_force: cap,
    }
}

fn check_settings(world: &GpuRigidSphereWorld, index: usize) -> TestResult {
    assert_eq!(
        world.spherical_motor(index, 0)?,
        Some(GpuRigidAxisMotor {
            target_velocity: 0.5,
            max_force: f32::INFINITY,
        })
    );
    assert_eq!(
        world.spherical_servo(index, 1)?,
        Some(servo(MotorModel::AccelerationBased, 2.0))
    );
    assert_eq!(
        world.spherical_limit(index, 2)?,
        Some(GpuRigidSphericalLimit {
            min: -0.2,
            max: 0.3
        })
    );
    assert_eq!(world.spherical_servo(index, 0)?, None);
    assert_eq!(world.spherical_motor(index, 1)?, None);
    assert_eq!(world.spherical_motor(index, 2)?, None);
    Ok(())
}

#[test]
fn spherical_owner_transactions_and_rebuild_retention() -> TestResult {
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        tested += 1;
        eprintln!("spherical owner retention: {backend:?}");
        let initial = [body(0.0), body(1.0), body(1.0)];
        let mut world = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &initial,
            &[0.1; 3],
            config(),
        )?;
        // Duplicate joint descriptors must retain independently configured rows.
        world.set_ball_joints(&[ball(0, 1), ball(0, 1)])?;
        assert_eq!(world.spherical_joint_count(), 2);
        assert_eq!(world.spherical_axis_count(), 6);
        world.set_spherical_motor(
            0,
            0,
            Some(GpuRigidAxisMotor {
                target_velocity: 0.5,
                max_force: f32::INFINITY,
            }),
        )?;
        world.set_spherical_servo(0, 1, Some(servo(MotorModel::AccelerationBased, 2.0)))?;
        world.set_spherical_limit(
            0,
            2,
            Some(GpuRigidSphericalLimit {
                min: -0.2,
                max: 0.3,
            }),
        )?;
        let zero_motor = GpuRigidAxisMotor {
            target_velocity: -1.0,
            max_force: 0.0,
        };
        world.set_spherical_motor(1, 2, Some(zero_motor))?;
        for (index, axis) in [(2, 0), (0, 3), (usize::MAX, usize::MAX)] {
            assert!(world.set_spherical_motor(index, axis, None).is_err());
            assert!(world.set_spherical_servo(index, axis, None).is_err());
            assert!(world.set_spherical_limit(index, axis, None).is_err());
            assert!(world.spherical_motor(index, axis).is_err());
            assert!(world.spherical_servo(index, axis).is_err());
            assert!(world.spherical_limit(index, axis).is_err());
        }
        assert!(
            world
                .set_spherical_motor(
                    0,
                    0,
                    Some(GpuRigidAxisMotor {
                        target_velocity: f32::NAN,
                        max_force: 1.0,
                    })
                )
                .is_err()
        );
        assert!(
            world
                .set_spherical_servo(0, 1, Some(servo(MotorModel::ForceBased, -1.0)))
                .is_err()
        );
        assert!(
            world
                .set_spherical_limit(
                    0,
                    2,
                    Some(GpuRigidSphericalLimit {
                        min: 1.0,
                        max: -1.0
                    })
                )
                .is_err()
        );
        check_settings(&world, 0)?;
        assert_eq!(world.spherical_motor(1, 2)?, Some(zero_motor));
        world.set_fixed_joints(&[GpuRigidFixedJoint {
            body_a: 0,
            body_b: 2,
            local_anchor_a: [0.0; 3],
            local_anchor_b: [0.0; 3],
            local_rotation_a: [0.0, 0.0, 0.0, 1.0],
            local_rotation_b: [0.0, 0.0, 0.0, 1.0],
        }])?;
        check_settings(&world, 0)?;
        world.set_revolute_joints(&[GpuRigidRevoluteJoint {
            body_a: 0,
            body_b: 2,
            local_anchor_a: [0.0; 3],
            local_anchor_b: [0.0; 3],
            local_axis_a: [0.0, 0.0, 1.0],
            local_axis_b: [0.0, 0.0, 1.0],
        }])?;
        check_settings(&world, 0)?;
        world.set_prismatic_joints(&[GpuRigidPrismaticJoint {
            body_a: 0,
            body_b: 2,
            local_anchor_a: [0.0; 3],
            local_anchor_b: [0.0; 3],
            local_rotation_a: [0.0, 0.0, 0.0, 1.0],
            local_rotation_b: [0.0, 0.0, 0.0, 1.0],
        }])?;
        check_settings(&world, 0)?;
        assert_eq!(world.spherical_motor(1, 2)?, Some(zero_motor));
        assert!(world.set_ball_joints(&[ball(0, 99)]).is_err());
        check_settings(&world, 0)?;
        world.reset(&initial)?;
        check_settings(&world, 0)?;
        let _added = world.append_body(body(1.0), 0.1)?;
        check_settings(&world, 0)?;
        let _removed = world.remove_body(2)?;
        check_settings(&world, 0)?;
        assert_eq!(world.spherical_motor(1, 2)?, Some(zero_motor));
        // Motor and servo exclusion is confined to the requested axis.
        world.set_spherical_servo(0, 0, Some(servo(MotorModel::ForceBased, 0.0)))?;
        assert_eq!(world.spherical_motor(0, 0)?, None);
        assert_eq!(
            world.spherical_servo(0, 1)?,
            Some(servo(MotorModel::AccelerationBased, 2.0))
        );
        world.set_spherical_motor(0, 0, Some(zero_motor))?;
        assert_eq!(world.spherical_servo(0, 0)?, None);
    }
    assert!(tested > 0, "no GPU backend available");
    Ok(())
}

#[test]
fn spherical_batch_environment_mapping_reset_and_remap() -> TestResult {
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        tested += 1;
        eprintln!("spherical batch mapping: {backend:?}");
        let states = [body(0.0), body(1.0)];
        let radii = [0.1; 2];
        let shapes = [
            GpuRigidShape::Sphere { radius: 0.1 },
            GpuRigidShape::Box {
                half_extents: [0.1; 3],
            },
        ];
        for primitive in [false, true] {
            let mut batch = if primitive {
                GpuRigidPrimitiveBatch::new_primitives(
                    context.device(),
                    context.queue(),
                    &[GpuRigidPrimitiveEnvironment {
                        states: &states,
                        shapes: &shapes,
                    }; 3],
                    config(),
                )?
            } else {
                GpuRigidSphereBatch::new(
                    context.device(),
                    context.queue(),
                    &[GpuRigidSphereEnvironment {
                        states: &states,
                        radii: &radii,
                    }; 3],
                    config(),
                )?
            };
            // Deliberately make packed joint order differ from environment order.
            for environment in [2, 0, 1] {
                batch.set_ball_joints_environment(environment, &[ball(0, 1)])?;
            }
            assert_eq!(batch.spherical_joint_index_environment(2, 0)?, 0);
            assert_eq!(batch.spherical_joint_index_environment(0, 0)?, 1);
            assert_eq!(batch.spherical_joint_index_environment(1, 0)?, 2);
            for environment in 0..3 {
                assert_eq!(batch.spherical_joint_count_environment(environment)?, 1);
                assert_eq!(batch.spherical_axis_count_environment(environment)?, 3);
            }
            let motor = GpuRigidAxisMotor {
                target_velocity: 0.5,
                max_force: f32::INFINITY,
            };
            let drive = servo(MotorModel::AccelerationBased, 2.0);
            let limit = GpuRigidSphericalLimit {
                min: -0.2,
                max: 0.3,
            };
            batch.set_spherical_motor_environment(2, 0, 0, Some(motor))?;
            batch.set_spherical_servo_environment(2, 0, 1, Some(drive))?;
            batch.set_spherical_limit_environment(2, 0, 2, Some(limit))?;
            for (environment, index, axis) in [(3, 0, 0), (2, 1, 0), (2, 0, 3)] {
                assert!(
                    batch
                        .set_spherical_motor_environment(environment, index, axis, None)
                        .is_err()
                );
                assert!(
                    batch
                        .set_spherical_servo_environment(environment, index, axis, None)
                        .is_err()
                );
                assert!(
                    batch
                        .set_spherical_limit_environment(environment, index, axis, None)
                        .is_err()
                );
            }
            assert!(batch.spherical_joint_count_environment(3).is_err());
            assert!(batch.spherical_axis_count_environment(3).is_err());
            batch.reset_environment(2, &states)?;
            batch.reset_all(&states.repeat(3))?;
            check_settings(
                batch.world(),
                batch.spherical_joint_index_environment(2, 0)?,
            )?;
            for environment in [0, 1] {
                assert_eq!(batch.spherical_motor_environment(environment, 0, 0)?, None);
                assert_eq!(batch.spherical_servo_environment(environment, 0, 1)?, None);
                assert_eq!(batch.spherical_limit_environment(environment, 0, 2)?, None);
            }
            // Remove an earlier environment, shifting body IDs and local environment IDs.
            if primitive {
                let _removed = batch.remove_environment_primitives(0)?;
            } else {
                let _removed = batch.remove_environment(0)?;
            }
            assert_eq!(batch.spherical_joint_count_environment(1)?, 1);
            check_settings(
                batch.world(),
                batch.spherical_joint_index_environment(1, 0)?,
            )?;
            assert_eq!(batch.spherical_motor_environment(0, 0, 0)?, None);
            batch.set_ball_joints_environment(0, &[])?;
            check_settings(
                batch.world(),
                batch.spherical_joint_index_environment(1, 0)?,
            )?;
            assert_eq!(batch.spherical_joint_count_environment(0)?, 0);
            assert!(
                batch
                    .set_spherical_motor_environment(0, 0, 0, Some(motor))
                    .is_err()
            );
            // Submit and read back the actual mixed-axis owner dispatch.
            for index in 0..batch.world().len() {
                batch.world_mut().set_body_collision_groups(
                    index,
                    GpuRigidCollisionGroups {
                        memberships: 0,
                        filter: 0,
                    },
                )?;
            }
            let _pairs = batch.step(0.01)?;
            let driven = batch.readback_environment(1)?[1];
            let untouched = batch.readback_environment(0)?[1];
            assert!(
                driven.angular_velocity[0] > 0.0,
                "{backend:?} primitive={primitive}: {driven:?}"
            );
            assert!(
                driven.angular_velocity[1] > 0.0,
                "{backend:?} primitive={primitive}: {driven:?}"
            );
            assert!(
                untouched.angular_velocity[..3]
                    .iter()
                    .all(|value| value.abs() < 1e-6),
                "{untouched:?}"
            );
        }
    }
    assert!(tested > 0, "no GPU backend available");
    Ok(())
}

#[test]
fn spherical_world_dispatch_preserves_acceleration_and_physical_caps() -> TestResult {
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        tested += 1;
        eprintln!("spherical world drive dispatch: {backend:?}");
        let mut world = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &[body(0.0), body(1.0)],
            &[0.1; 2],
            config(),
        )?;
        for index in 0..2 {
            world.set_body_collision_groups(
                index,
                GpuRigidCollisionGroups {
                    memberships: 0,
                    filter: 0,
                },
            )?;
        }
        world.set_ball_joints(&[ball(0, 1)])?;
        for temporal in [false, true] {
            for inverse in [0.25, 4.0] {
                for model in [MotorModel::ForceBased, MotorModel::AccelerationBased] {
                    for cap in [0.0, 2.0, f32::INFINITY] {
                        world.reset(&[body(0.0), body(inverse)])?;
                        world.set_spherical_servo(0, 2, Some(servo(model, cap)))?;
                        if temporal {
                            let _pairs = world.step_temporal_with_joints(
                                0.01,
                                1,
                                GpuRigidTemporalSolveParams::default(),
                                GpuRigidTemporalJointParams::default(),
                                0.0,
                            )?;
                        } else {
                            let _pairs = world.step(0.01)?;
                        }
                        let state = world.readback()?[1];
                        let expected = (0.1
                            * if model == MotorModel::AccelerationBased {
                                1.0
                            } else {
                                inverse
                            })
                        .min(cap * 0.01 * inverse);
                        assert!(
                            (state.angular_velocity[2] - expected).abs() < 2e-5,
                            "{backend:?} temporal={temporal} inverse={inverse} model={model:?} cap={cap}: {state:?} expected={expected}"
                        );
                    }
                }
            }
        }
    }
    assert!(tested > 0, "no GPU backend available");
    Ok(())
}
