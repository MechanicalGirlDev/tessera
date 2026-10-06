//! Temporal world scheduling, force lifetime and initial LBVH contact topology.
#![cfg(feature = "gpu-contact")]
use tessera_physics::material::ColliderMaterial;
use tessera_physics::{
    gpu_contact_pipeline::GpuContactDevice,
    gpu_rigid_sphere_solver::GpuRigidTemporalSolveParams,
    gpu_rigid_sphere_world::{GpuRigidSphereWorld, GpuRigidSphereWorldConfig},
    gpu_rigid_state::{GpuRigidBodyForces, GpuRigidBodyState},
    sleep::SleepSettings,
};
fn body(position: [f32; 3], inverse_mass: f32) -> GpuRigidBodyState {
    GpuRigidBodyState {
        position_inverse_mass: [position[0], position[1], position[2], inverse_mass],
        orientation: [0.0, 0.0, 0.0, 1.0],
        linear_velocity: [0.0; 4],
        angular_velocity: [0.0; 4],
        inverse_inertia_sleep: [inverse_mass, inverse_mass, inverse_mass, 0.0],
    }
}
#[test]
fn temporal_world_preserves_frame_forces_and_ground_support()
-> Result<(), Box<dyn core::error::Error>> {
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        tested += 1;
        let settings = GpuRigidTemporalSolveParams::default();
        let config = GpuRigidSphereWorldConfig {
            gravity: [0.0; 3],
            ground_half_extent: None,
            sleep: SleepSettings {
                enabled: false,
                ..SleepSettings::default()
            },
            ..GpuRigidSphereWorldConfig::default()
        };
        let initial = body([0.0; 3], 1.0);
        let mut world = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &[initial],
            &[1.0],
            config,
        )?;
        world.write_forces(
            0,
            GpuRigidBodyForces {
                force: [2.0, 0.0, 0.0, 0.0],
                torque: [0.0; 4],
            },
        )?;
        assert!(world.step_temporal(0.1, 0, settings).is_err());
        assert!(!world.is_faulted());
        assert!(
            world
                .step_temporal(
                    0.1,
                    4,
                    GpuRigidTemporalSolveParams {
                        friction: f32::NAN,
                        ..settings
                    }
                )
                .is_err()
        );
        assert!(!world.is_faulted());
        let _ = world.step_temporal(0.1, 4, settings)?;
        let state = world.readback()?[0];
        assert!((state.linear_velocity[0] - 0.2).abs() < 1e-6);
        assert!((state.position_inverse_mass[0] - 0.0125).abs() < 1e-6);
        let _ = world.step_temporal(0.1, 4, settings)?;
        let state = world.readback()?[0];
        assert!((state.linear_velocity[0] - 0.2).abs() < 1e-6);
        assert!((state.position_inverse_mass[0] - 0.0325).abs() < 1e-6);
        // Switching to PGS and back must discard incompatible impulse history.
        let _ = world.step(0.01)?;
        let _ = world.step_temporal(0.01, 2, settings)?;
        assert!((world.readback()?[0].linear_velocity[0] - 0.2).abs() < 1e-6);
        let mut supported = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &[body([0.0, 0.0, 1.0], 1.0)],
            &[1.0],
            GpuRigidSphereWorldConfig {
                gravity: [0.0, 0.0, -9.81],
                ground_half_extent: Some(10.0),
                ..config
            },
        )?;
        for _ in 0..120 {
            let _ = supported.step_temporal(1.0 / 120.0, 4, settings)?;
        }
        let state = supported.readback()?[0];
        assert!(
            (state.position_inverse_mass[2] - 1.0).abs() < 0.005,
            "{backend:?}: {state:?}"
        );
        assert!(
            state.linear_velocity[2].abs() < 0.1,
            "{backend:?}: {state:?}"
        );
    }
    assert!(tested > 0, "no GPU backend available");
    Ok(())
}

#[test]
fn temporal_lbvh_frame_matches_exhaustive_pair_solve() -> Result<(), Box<dyn core::error::Error>> {
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        tested += 1;
        let config = GpuRigidSphereWorldConfig {
            gravity: [0.0; 3],
            ground_half_extent: None,
            sleep: SleepSettings {
                enabled: false,
                ..SleepSettings::default()
            },
            ..GpuRigidSphereWorldConfig::default()
        };
        let mut moving = body([2.0, 0.0, 0.0], 1.0);
        moving.linear_velocity[0] = -1.0;
        let pair = [body([0.0; 3], 0.0), moving];
        let mut all = pair.to_vec();
        for i in 0..15 {
            all.push(body([20.0 + 5.0 * i as f32, 0.0, 0.0], 0.0));
        }
        let mut small =
            GpuRigidSphereWorld::new(context.device(), context.queue(), &pair, &[1.0; 2], config)?;
        let mut large =
            GpuRigidSphereWorld::new(context.device(), context.queue(), &all, &[1.0; 17], config)?;
        assert!(!large.uses_exhaustive_pairs());
        for _ in 0..3 {
            let _ = small.step_temporal(0.01, 4, GpuRigidTemporalSolveParams::default())?;
            let _ = large.step_temporal(0.01, 4, GpuRigidTemporalSolveParams::default())?;
            let a = small.readback()?[1];
            let b = large.readback()?[1];
            for axis in 0..3 {
                assert!(
                    (a.position_inverse_mass[axis] - b.position_inverse_mass[axis]).abs() < 1e-6
                );
                assert!((a.linear_velocity[axis] - b.linear_velocity[axis]).abs() < 1e-6);
            }
        }
    }
    assert!(tested > 0, "no GPU backend available");
    Ok(())
}

#[test]
fn initial_speculative_rows_cover_separated_pair_and_ground()
-> Result<(), Box<dyn core::error::Error>> {
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        tested += 1;
        let config = GpuRigidSphereWorldConfig {
            gravity: [0.0; 3],
            ground_half_extent: None,
            sleep: SleepSettings {
                enabled: false,
                ..SleepSettings::default()
            },
            ..GpuRigidSphereWorldConfig::default()
        };
        let mut moving = body([2.05, 0.0, 0.0], 1.0);
        moving.linear_velocity[0] = -1.0;
        let pair = [body([0.0; 3], 0.0), moving];
        let mut all = pair.to_vec();
        for i in 0..15 {
            all.push(body([20.0 + 5.0 * i as f32, 0.0, 0.0], 0.0));
        }
        let mut small =
            GpuRigidSphereWorld::new(context.device(), context.queue(), &pair, &[1.0; 2], config)?;
        let mut large =
            GpuRigidSphereWorld::new(context.device(), context.queue(), &all, &[1.0; 17], config)?;
        let mut control =
            GpuRigidSphereWorld::new(context.device(), context.queue(), &pair, &[1.0; 2], config)?;
        let settings = GpuRigidTemporalSolveParams::default();
        let _ = control.step_temporal(0.1, 4, settings)?;
        let _ = small.step_temporal_speculative(0.1, 4, settings, 0.1)?;
        let count = large.step_temporal_speculative(0.1, 4, settings, 0.1)?;
        assert_eq!(count, 1, "expanded LBVH must retain the separated pair");
        let a = small.readback()?[1];
        let b = large.readback()?[1];
        assert!((a.position_inverse_mass[0] - b.position_inverse_mass[0]).abs() < 1e-6);
        assert!((a.linear_velocity[0] - b.linear_velocity[0]).abs() < 1e-6);
        assert!(a.position_inverse_mass[0] > 1.99, "{backend:?}: {a:?}");
        assert!(control.readback()?[1].position_inverse_mass[0] < 1.96);
        assert!(small.readback_contacts()?.pairs[0].1.is_contact());
        let mut falling = body([0.0, 0.0, 1.05], 1.0);
        falling.linear_velocity[2] = -1.0;
        let mut ground = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &[falling],
            &[1.0],
            GpuRigidSphereWorldConfig {
                ground_half_extent: Some(10.0),
                ..config
            },
        )?;
        let _ = ground.step_temporal_speculative(0.1, 4, settings, 0.1)?;
        let state = ground.readback()?[0];
        assert!(
            state.position_inverse_mass[2] > 0.99,
            "{backend:?}: {state:?}"
        );
        // A distant speculative row must permit the full closing velocity.
        moving.position_inverse_mass[0] = 2.5;
        small.write_body(1, moving)?;
        let _ = small.step_temporal_speculative(0.01, 4, settings, 0.6)?;
        let state = small.readback()?[1];
        assert!((state.linear_velocity[0] + 1.0).abs() < 1e-6);
        assert!(small.readback_contacts()?.pairs[0].1.depth_hit[0] < -0.48);
    }
    assert!(tested > 0, "no GPU backend available");
    Ok(())
}

#[test]
fn speed_bounded_margin_covers_fast_sphere_pair_in_exhaustive_and_lbvh_worlds()
-> Result<(), Box<dyn core::error::Error>> {
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        tested += 1;
        let config = GpuRigidSphereWorldConfig {
            gravity: [0.0; 3],
            ground_half_extent: None,
            sleep: SleepSettings {
                enabled: false,
                ..SleepSettings::default()
            },
            ..GpuRigidSphereWorldConfig::default()
        };
        let mut moving = body([2.5, 0.0, 0.0], 1.0);
        moving.linear_velocity[0] = -10.0;
        let pair = [body([0.0; 3], 0.0), moving];
        let mut all = pair.to_vec();
        for index in 0..15 {
            all.push(body([20.0 + index as f32 * 5.0, 0.0, 0.0], 0.0));
        }
        let mut small =
            GpuRigidSphereWorld::new(context.device(), context.queue(), &pair, &[1.0; 2], config)?;
        let mut large =
            GpuRigidSphereWorld::new(context.device(), context.queue(), &all, &[1.0; 17], config)?;
        let mut control =
            GpuRigidSphereWorld::new(context.device(), context.queue(), &pair, &[1.0; 2], config)?;
        let settings = GpuRigidTemporalSolveParams::default();
        assert!(small.speed_bounded_speculative_margin(0.1).is_err());
        for world in [&mut small, &mut large, &mut control] {
            world.set_max_linear_speed(Some(10.0))?;
        }
        let margin = small.speed_bounded_speculative_margin(0.1)?;
        assert!(margin > 2.0 && margin < 2.001);
        assert!(small.speed_bounded_speculative_margin(f32::NAN).is_err());
        let _ = control.step_temporal(0.1, 4, settings)?;
        let _ = small.step_temporal_speed_bounded(0.1, 4, settings)?;
        let count = large.step_temporal_speed_bounded(0.1, 4, settings)?;
        assert_eq!(count, 1, "{backend:?}");
        let a = small.readback()?[1];
        let b = large.readback()?[1];
        assert!((a.position_inverse_mass[0] - b.position_inverse_mass[0]).abs() < 1e-5);
        assert!((a.linear_velocity[0] - b.linear_velocity[0]).abs() < 1e-5);
        assert!(a.position_inverse_mass[0] > 1.99, "{backend:?}: {a:?}");
        assert!(control.readback()?[1].position_inverse_mass[0] < 1.6);
    }
    assert!(tested > 0, "no GPU backend available");
    Ok(())
}

#[test]
fn speculative_ground_contact_uses_incoming_speed_for_restitution()
-> Result<(), Box<dyn core::error::Error>> {
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        tested += 1;
        let mut initial = body([0.0, 0.0, 1.05], 1.0);
        initial.linear_velocity[2] = -10.0;
        let config = GpuRigidSphereWorldConfig {
            gravity: [0.0; 3],
            ground_half_extent: Some(10.0),
            sleep: SleepSettings {
                enabled: false,
                ..SleepSettings::default()
            },
            ..GpuRigidSphereWorldConfig::default()
        };
        let mut world = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &[initial],
            &[1.0],
            config,
        )?;
        let material = ColliderMaterial::new(0.0, 0.5);
        world.set_body_material(0, material)?;
        world.set_ground_material(material)?;
        let _ = world.step_temporal_speculative(
            0.01,
            1,
            GpuRigidTemporalSolveParams::default(),
            0.1,
        )?;
        let state = world.readback()?[0];
        assert!(state.linear_velocity[2] > 4.5, "{backend:?}: {state:?}");
        let mut far = body([0.0, 0.0, 1.5], 1.0);
        far.linear_velocity[2] = -10.0;
        let mut separated =
            GpuRigidSphereWorld::new(context.device(), context.queue(), &[far], &[1.0], config)?;
        separated.set_body_material(0, material)?;
        separated.set_ground_material(material)?;
        let _ = separated.step_temporal_speculative(
            0.01,
            1,
            GpuRigidTemporalSolveParams::default(),
            0.6,
        )?;
        let state = separated.readback()?[0];
        assert!(
            (state.linear_velocity[2] + 10.0).abs() < 1e-5,
            "{backend:?}: {state:?}"
        );

        let fixed = body([0.0; 3], 0.0);
        let mut moving = body([2.05, 0.0, 0.0], 1.0);
        moving.linear_velocity[0] = -10.0;
        let mut pair = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &[fixed, moving],
            &[1.0; 2],
            GpuRigidSphereWorldConfig {
                ground_half_extent: None,
                ..config
            },
        )?;
        pair.set_body_material(0, material)?;
        pair.set_body_material(1, material)?;
        let _ =
            pair.step_temporal_speculative(0.01, 1, GpuRigidTemporalSolveParams::default(), 0.1)?;
        let state = pair.readback()?[1];
        assert!(state.linear_velocity[0] > 4.5, "{backend:?}: {state:?}");
    }
    assert!(tested > 0, "no GPU backend available");
    Ok(())
}

#[test]
fn temporal_box_uses_initial_speculative_ground_manifold() -> Result<(), Box<dyn core::error::Error>>
{
    use tessera_physics::gpu_rigid_shape::GpuRigidShape;
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        tested += 1;
        let mut initial = body([0.0, 0.0, 0.55], 1.0);
        initial.linear_velocity[2] = -1.0;
        let mut world = GpuRigidSphereWorld::new_primitives(
            context.device(),
            context.queue(),
            &[initial],
            &[GpuRigidShape::Box {
                half_extents: [0.5; 3],
            }],
            GpuRigidSphereWorldConfig {
                gravity: [0.0; 3],
                sleep: SleepSettings {
                    enabled: false,
                    ..SleepSettings::default()
                },
                ..GpuRigidSphereWorldConfig::default()
            },
        )?;
        let _ = world.step_temporal_speculative_ground(
            0.1,
            8,
            GpuRigidTemporalSolveParams::default(),
            0.1,
        )?;
        let state = world.readback()?[0];
        assert!(
            state.position_inverse_mass[2] > 0.49,
            "{backend:?}: {state:?}"
        );
        let contacts = world.readback_contacts()?;
        assert!(contacts.ground[0].is_contact());
        assert!(contacts.ground_extra[0].iter().all(|c| c.is_contact()));
    }
    assert!(tested > 0, "no GPU backend available");
    Ok(())
}
