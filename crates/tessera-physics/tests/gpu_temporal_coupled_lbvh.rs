//! Shared-base drives against contact and limits in exhaustive and LBVH worlds.
#![cfg(feature = "gpu-contact")]
use tessera_physics::{
    gpu_contact_pipeline::GpuContactDevice,
    gpu_rigid_ball_joint::{GpuRigidAxisMotor, GpuRigidPrismaticJoint, GpuRigidPrismaticLimit},
    gpu_rigid_sphere_solver::GpuRigidTemporalSolveParams,
    gpu_rigid_sphere_world::{GpuRigidSphereWorld, GpuRigidSphereWorldConfig},
    gpu_rigid_state::GpuRigidBodyState,
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
fn configure(world: &mut GpuRigidSphereWorld) -> Result<(), Box<dyn core::error::Error>> {
    let joints = [-1.0, 1.0]
        .into_iter()
        .enumerate()
        .map(|(index, x)| GpuRigidPrismaticJoint {
            body_a: 0,
            body_b: index as u32 + 1,
            local_anchor_a: [x, 0.0, 0.0],
            local_anchor_b: [0.0; 3],
            local_rotation_a: [0.0, 0.0, 0.0, 1.0],
            local_rotation_b: [0.0, 0.0, 0.0, 1.0],
        })
        .collect::<Vec<_>>();
    world.set_prismatic_joints(&joints)?;
    for index in 0..2 {
        world.set_prismatic_limit(index, Some(GpuRigidPrismaticLimit { min: 0.5, max: 0.8 }))?;
    }
    Ok(())
}
#[test]
fn shared_base_actuators_release_ground_and_stop_at_both_limits()
-> Result<(), Box<dyn core::error::Error>> {
    let mut initial = [
        body(0.0, 0.0, 0.0),
        body(-1.0, 0.55, 1.0),
        body(1.0, 0.55, 0.5),
    ];
    initial[1].inverse_inertia_sleep = [5.0, 4.0, 3.0, 0.0];
    initial[2].inverse_inertia_sleep = [2.0, 3.0, 4.0, 0.0];
    for state in &mut initial[1..] {
        state.orientation = [0.0, (0.05_f32).sin(), 0.0, (0.05_f32).cos()];
        state.angular_velocity = [0.2, -0.1, 0.3, 0.0];
    }
    let mut large_bodies = initial.to_vec();
    let mut large_radii = vec![0.1, 0.5, 0.5];
    for index in 0..14 {
        large_bodies.push(body(50.0 + index as f32 * 4.0, 3.0, 0.0));
        large_radii.push(0.1);
    }
    let config = GpuRigidSphereWorldConfig {
        gravity: [0.0, 0.0, -9.81],
        ground_half_extent: Some(10.0),
        sleep: SleepSettings {
            enabled: false,
            ..SleepSettings::default()
        },
        ..GpuRigidSphereWorldConfig::default()
    };
    let settings = GpuRigidTemporalSolveParams {
        friction: 0.0,
        iterations: 8,
        ..GpuRigidTemporalSolveParams::default()
    };
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        tested += 1;
        eprintln!("temporal coupled LBVH: {backend:?}");
        let mut small = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &initial,
            &[0.1, 0.5, 0.5],
            config,
        )?;
        let mut large = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &large_bodies,
            &large_radii,
            config,
        )?;
        assert!(small.uses_exhaustive_pairs());
        assert!(!large.uses_exhaustive_pairs());
        configure(&mut small)?;
        configure(&mut large)?;
        for (phase, (velocity, frames, expected)) in
            [(-1.0, 60, 0.5), (0.5, 120, 0.8), (-1.0, 120, 0.5)]
                .into_iter()
                .enumerate()
        {
            for world in [&mut small, &mut large] {
                for index in 0..2 {
                    world.set_prismatic_motor(
                        index,
                        Some(GpuRigidAxisMotor {
                            target_velocity: velocity,
                            max_force: 40.0,
                        }),
                    )?;
                }
            }
            for frame in 0..frames {
                let _ = small.step_temporal_speculative(1.0 / 120.0, 4, settings, 0.1)?;
                let _ = large.step_temporal_speculative(1.0 / 120.0, 4, settings, 0.1)?;
                let a = small.readback()?;
                let b = large.readback()?;
                assert_eq!(b[0].position_inverse_mass, initial[0].position_inverse_mass);
                for index in 1..3 {
                    assert!(
                        (0.49..=0.81).contains(&b[index].position_inverse_mass[2]),
                        "{backend:?}: phase={phase}: frame={frame}: body={index}: {:?}",
                        b[index]
                    );
                    for axis in 0..3 {
                        assert!(
                            (a[index].position_inverse_mass[axis]
                                - b[index].position_inverse_mass[axis])
                                .abs()
                                < 1e-4,
                            "{backend:?}: phase={phase}: frame={frame}: small={:?}: LBVH={:?}",
                            a[index],
                            b[index]
                        );
                        assert!(
                            (a[index].linear_velocity[axis] - b[index].linear_velocity[axis]).abs()
                                < 1e-4,
                            "{backend:?}: phase={phase}: frame={frame}: small={:?}: LBVH={:?}",
                            a[index],
                            b[index]
                        );
                    }
                }
            }
            let states = large.readback()?;
            for (offset, state) in states[1..3].iter().enumerate() {
                assert!(
                    (state.position_inverse_mass[2] - expected).abs() < 0.005,
                    "{backend:?}: phase={phase}: {state:?}"
                );
                assert!(
                    state.linear_velocity[..3].iter().all(|v| v.abs() < 0.05),
                    "{state:?}"
                );
                assert!(
                    (state.position_inverse_mass[0] - initial[offset + 1].position_inverse_mass[0])
                        .abs()
                        < 0.01,
                    "{state:?}"
                );
                assert!(
                    state.orientation[..3].iter().all(|v| v.abs() < 0.005),
                    "{state:?}"
                );
                assert!(
                    state.angular_velocity[..3].iter().all(|v| v.abs() < 0.02),
                    "{state:?}"
                );
            }
            let contacts = large.readback_contacts()?;
            for index in 1..3 {
                assert_eq!(contacts.ground[index].is_contact(), phase != 1);
            }
        }
    }
    assert!(tested > 0, "no GPU backend available");
    Ok(())
}
