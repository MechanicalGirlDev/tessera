//! Multi-point box stack stability with friction and a live LBVH broad phase.
#![cfg(feature = "gpu-contact")]
use tessera_physics::{
    gpu_contact_pipeline::GpuContactDevice,
    gpu_rigid_shape::GpuRigidShape,
    gpu_rigid_sphere_solver::GpuRigidTemporalSolveParams,
    gpu_rigid_sphere_world::{GpuRigidSphereWorld, GpuRigidSphereWorldConfig},
    gpu_rigid_state::GpuRigidBodyState,
    sleep::SleepSettings,
};

#[test]
fn five_box_stack_dissipates_tangent_motion_without_sleep()
-> Result<(), Box<dyn core::error::Error>> {
    let mut bodies = Vec::new();
    for index in 0..17 {
        let dynamic = index < 5;
        bodies.push(GpuRigidBodyState {
            position_inverse_mass: if dynamic {
                [0.0, 0.0, 0.5 + index as f32 * 1.01, 1.0]
            } else {
                [50.0 + index as f32 * 3.0, 0.0, 1.0, 0.0]
            },
            orientation: [0.0, 0.0, 0.0, 1.0],
            linear_velocity: if index == 4 {
                [0.1, 0.0, 0.0, 0.0]
            } else {
                [0.0; 4]
            },
            angular_velocity: [0.0; 4],
            inverse_inertia_sleep: if dynamic {
                [6.0, 6.0, 6.0, 0.0]
            } else {
                [0.0; 4]
            },
        });
    }
    let shapes = vec![
        GpuRigidShape::Box {
            half_extents: [0.5; 3]
        };
        bodies.len()
    ];
    let settings = GpuRigidTemporalSolveParams {
        friction: 0.7,
        iterations: 8,
        ..Default::default()
    };
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        tested += 1;
        eprintln!("temporal five-box stack: {backend:?}");
        let mut world = GpuRigidSphereWorld::new_primitives(
            context.device(),
            context.queue(),
            &bodies,
            &shapes,
            GpuRigidSphereWorldConfig {
                gravity: [0.0, 0.0, -9.81],
                ground_half_extent: Some(10.0),
                sleep: SleepSettings {
                    enabled: false,
                    ..Default::default()
                },
                ..Default::default()
            },
        )?;
        for frame in 0..240 {
            if let Err(error) = world.step_temporal_speculative(1.0 / 120.0, 4, settings, 0.05) {
                eprintln!(
                    "{backend:?}: failed frame={frame}: states={:?}",
                    world.readback()?
                );
                return Err(error.into());
            }
            let states = world.readback()?;
            for (index, state) in states[..5].iter().enumerate() {
                let expected = 0.5 + index as f32;
                assert!(
                    (state.position_inverse_mass[2] - expected).abs() < 0.04,
                    "{backend:?}: frame={frame}: body={index}: {state:?}"
                );
                assert!(
                    state.position_inverse_mass[..2]
                        .iter()
                        .all(|v| v.abs() < 0.1),
                    "{backend:?}: frame={frame}: body={index}: {state:?}"
                );
            }
            for (state, initial) in states[5..].iter().zip(&bodies[5..]) {
                assert_eq!(state.position_inverse_mass, initial.position_inverse_mass);
                assert_eq!(state.linear_velocity, initial.linear_velocity);
            }
        }
        let states = world.readback()?;
        for (index, state) in states[..5].iter().enumerate() {
            assert!(
                state.linear_velocity[..3].iter().all(|v| v.abs() < 0.03),
                "{backend:?}: body={index}: {state:?}"
            );
            assert!(
                state.angular_velocity[..3].iter().all(|v| v.abs() < 0.05),
                "{backend:?}: body={index}: {state:?}"
            );
            assert!(
                state.orientation[..3].iter().all(|v| v.abs() < 0.02),
                "{backend:?}: body={index}: {state:?}"
            );
        }
    }
    assert!(tested > 0, "no GPU backend available");
    Ok(())
}
