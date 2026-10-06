//! Temporal contact/joint coupling, force lifetime and continuous hinge angles.
#![cfg(feature = "gpu-contact")]
use tessera_physics::{
    gpu_contact_pipeline::GpuContactDevice,
    gpu_rigid_ball_joint::{
        GpuRigidAxisMotor, GpuRigidFixedJoint, GpuRigidPrismaticJoint, GpuRigidRevoluteJoint,
        GpuRigidTemporalJointParams,
    },
    gpu_rigid_sphere_solver::GpuRigidTemporalSolveParams,
    gpu_rigid_sphere_world::{GpuRigidSphereWorld, GpuRigidSphereWorldConfig},
    gpu_rigid_state::{GpuRigidBodyForces, GpuRigidBodyState},
    sleep::SleepSettings,
};
fn body(x: f32, z: f32, mass: f32) -> GpuRigidBodyState {
    GpuRigidBodyState {
        position_inverse_mass: [x, 0.0, z, mass],
        orientation: [0.0, 0.0, 0.0, 1.0],
        linear_velocity: [0.0; 4],
        angular_velocity: [0.0; 4],
        inverse_inertia_sleep: [mass, mass, mass, 0.0],
    }
}
#[test]
fn coupled_ground_support_preserves_momentum_and_multiturn_hinge_angles()
-> Result<(), Box<dyn core::error::Error>> {
    let contacts = GpuRigidTemporalSolveParams {
        friction: 0.0,
        iterations: 8,
        ..GpuRigidTemporalSolveParams::default()
    };
    let joints = GpuRigidTemporalJointParams {
        frequency: 40.0,
        iterations: 8,
        ..GpuRigidTemporalJointParams::default()
    };
    let config = GpuRigidSphereWorldConfig {
        gravity: [0.0, 0.0, -9.81],
        ground_half_extent: Some(10.0),
        sleep: SleepSettings {
            enabled: false,
            ..SleepSettings::default()
        },
        ..GpuRigidSphereWorldConfig::default()
    };
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        tested += 1;
        eprintln!("temporal joint world: {backend:?}");
        let mut initial = [body(-1.0, 0.55, 1.0), body(1.0, 0.55, 1.0)];
        for state in &mut initial {
            state.linear_velocity[2] = -1.0;
        }
        let mut world = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &initial,
            &[0.5; 2],
            config,
        )?;
        world.set_fixed_joints(&[GpuRigidFixedJoint {
            body_a: 0,
            body_b: 1,
            local_anchor_a: [1.0, 0.0, 0.0],
            local_anchor_b: [-1.0, 0.0, 0.0],
            local_rotation_a: [0.0, 0.0, 0.0, 1.0],
            local_rotation_b: [0.0, 0.0, 0.0, 1.0],
        }])?;
        world.write_forces(
            0,
            GpuRigidBodyForces {
                force: [2.0, 0.0, 0.0, 0.0],
                torque: [0.0; 4],
            },
        )?;
        assert!(
            world
                .step_temporal_with_joints(
                    0.1,
                    8,
                    contacts,
                    GpuRigidTemporalJointParams {
                        frequency: 0.0,
                        ..joints
                    },
                    0.1
                )
                .is_err()
        );
        assert!(!world.is_faulted());
        let _ = world.step_temporal_with_joints(0.1, 8, contacts, joints, 0.1)?;
        let states = world.readback()?;
        assert!(
            ((states[0].linear_velocity[0] + states[1].linear_velocity[0]) * 0.5 - 0.1).abs()
                < 2e-5,
            "{backend:?}: {states:?}"
        );
        assert!(
            ((states[0].position_inverse_mass[0] + states[1].position_inverse_mass[0]) * 0.5
                - 0.005625)
                .abs()
                < 2e-5,
            "{backend:?}: {states:?}"
        );
        for frame in 0..12 {
            let _ = world.step_temporal_with_joints(1.0 / 60.0, 4, contacts, joints, 0.1)?;
            let states = world.readback()?;
            for state in &states {
                assert!(
                    state.position_inverse_mass[2] > 0.49,
                    "{backend:?}: frame={frame}: {states:?}"
                );
            }
            assert!(
                (states[1].position_inverse_mass[0] - states[0].position_inverse_mass[0] - 2.0)
                    .abs()
                    < 0.01,
                "{states:?}"
            );
            assert!(
                (states[1].position_inverse_mass[2] - states[0].position_inverse_mass[2]).abs()
                    < 0.01,
                "{states:?}"
            );
            assert!(
                ((states[0].linear_velocity[0] + states[1].linear_velocity[0]) * 0.5 - 0.1).abs()
                    < 2e-5,
                "{states:?}"
            );
        }
        let _ = world.step(0.001)?;
        let _ = world.step_temporal(0.001, 2, contacts)?;

        let mut spinning = body(0.0, 1.0, 1.0);
        spinning.angular_velocity[2] = 1000.0;
        let mut hinge = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &[body(0.0, 0.0, 0.0), spinning],
            &[0.1; 2],
            GpuRigidSphereWorldConfig {
                gravity: [0.0; 3],
                ground_half_extent: None,
                ..config
            },
        )?;
        hinge.set_revolute_joints(&[GpuRigidRevoluteJoint {
            body_a: 0,
            body_b: 1,
            local_anchor_a: [0.0, 0.0, 1.0],
            local_anchor_b: [0.0; 3],
            local_axis_a: [0.0, 0.0, 1.0],
            local_axis_b: [0.0, 0.0, 1.0],
        }])?;
        let _ = hinge.step_temporal(0.02, 2, contacts)?;
        assert!(
            (hinge.readback_revolute_angle(0)? - 20.0).abs() < 1e-3,
            "{backend:?}"
        );
        hinge.set_revolute_motor(
            0,
            Some(GpuRigidAxisMotor {
                target_velocity: 0.0,
                max_force: 1.0,
            }),
        )?;
        hinge.write_forces(
            1,
            GpuRigidBodyForces {
                force: [0.0; 4],
                torque: [0.0, 0.0, 1.0, 0.0],
            },
        )?;
        assert!(
            hinge
                .step_temporal_with_joints(
                    0.01,
                    1,
                    contacts,
                    GpuRigidTemporalJointParams {
                        frequency: 0.0,
                        ..joints
                    },
                    0.0
                )
                .is_err()
        );
        assert!(!hinge.is_faulted());
        assert!((hinge.readback_revolute_angle(0)? - 20.0).abs() < 1e-3);
        hinge.set_revolute_motor(0, None)?;
        let _ = hinge.step_temporal(0.01, 1, contacts)?;
        assert!((hinge.readback()?[1].angular_velocity[2] - 1000.01).abs() < 2e-4);
        assert!((hinge.readback_revolute_angle(0)? - 30.0001).abs() < 1e-3);
        let mut sliding = body(0.1, 5.0, 1.0);
        sliding.linear_velocity = [-1.0, 0.0, 1.5, 0.0];
        sliding.orientation = [0.0, 0.0, (0.1_f32).sin(), (0.1_f32).cos()];
        sliding.angular_velocity[2] = 1.0;
        let mut slider = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &[body(0.0, 0.0, 0.0), sliding],
            &[0.1; 2],
            GpuRigidSphereWorldConfig {
                gravity: [0.0; 3],
                ground_half_extent: None,
                ..config
            },
        )?;
        slider.set_prismatic_joints(&[GpuRigidPrismaticJoint {
            body_a: 0,
            body_b: 1,
            local_anchor_a: [0.0; 3],
            local_anchor_b: [0.0; 3],
            local_rotation_a: [0.0, 0.0, 0.0, 1.0],
            local_rotation_b: [0.0, 0.0, 0.0, 1.0],
        }])?;
        slider.write_forces(
            1,
            GpuRigidBodyForces {
                force: [0.0, 0.0, 2.0, 0.0],
                torque: [0.0; 4],
            },
        )?;
        let _ = slider.step_temporal_with_joints(0.1, 4, contacts, joints, 0.0)?;
        let state = slider.readback()?[1];
        assert!((state.linear_velocity[2] - 1.7).abs() < 2e-5, "{state:?}");
        assert!(
            (state.position_inverse_mass[2] - 5.1625).abs() < 2e-5,
            "{state:?}"
        );
        let _ = slider.step_temporal_with_joints(0.1, 4, contacts, joints, 0.0)?;
        let state = slider.readback()?[1];
        assert!((state.linear_velocity[2] - 1.7).abs() < 2e-5, "{state:?}");
        assert!(
            (state.position_inverse_mass[2] - 5.3325).abs() < 2e-5,
            "{state:?}"
        );
        assert!(state.position_inverse_mass[0].abs() < 0.01, "{state:?}");
        assert!(state.orientation[2].abs() < 0.01, "{state:?}");
    }
    assert!(tested > 0, "no GPU backend available");
    Ok(())
}
