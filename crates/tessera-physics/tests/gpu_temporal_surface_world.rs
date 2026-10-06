//! Surface speculative contacts through LBVH, anchor refresh and temporal solve.
#![cfg(feature = "gpu-contact")]
use tessera_physics::{
    gpu_contact_pipeline::GpuContactDevice,
    gpu_rigid_shape::GpuRigidShape,
    gpu_rigid_sphere_solver::GpuRigidTemporalSolveParams,
    gpu_rigid_sphere_world::{GpuRigidSphereWorld, GpuRigidSphereWorldConfig},
    gpu_rigid_state::GpuRigidBodyState,
    sleep::SleepSettings,
};

fn body(x: f32, z: f32, moving: bool) -> GpuRigidBodyState {
    let inverse_mass = if moving { 1.0 } else { 0.0 };
    GpuRigidBodyState {
        position_inverse_mass: [x, 0.0, z, inverse_mass],
        orientation: [0.0, 0.0, 0.0, 1.0],
        linear_velocity: [0.0, 0.0, if moving { -1.0 } else { 0.0 }, 0.0],
        angular_velocity: [0.0; 4],
        inverse_inertia_sleep: [inverse_mass, inverse_mass, inverse_mass, 0.0],
    }
}

#[test]
fn surface_gaps_resist_crossing_in_both_registration_orders()
-> Result<(), Box<dyn core::error::Error>> {
    let mesh = GpuRigidShape::TriangleMesh {
        vertices: vec![
            [-2.0, -2.0, 0.0],
            [2.0, -2.0, 0.0],
            [2.0, 2.0, 0.0],
            [-2.0, 2.0, 0.0],
        ],
        triangles: vec![[0, 1, 2], [0, 2, 3]],
    };
    let line = GpuRigidShape::Polyline {
        vertices: vec![[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        segments: vec![[0, 1]],
    };
    let sphere = GpuRigidShape::Sphere { radius: 0.5 };
    let mut bodies = Vec::new();
    let mut shapes = Vec::new();
    let mut moving_indices = Vec::new();
    for (index, surface) in [mesh.clone(), line.clone(), mesh, line]
        .into_iter()
        .enumerate()
    {
        let x = index as f32 * 8.0;
        if index < 2 {
            moving_indices.push(bodies.len() + 1);
            bodies.extend([body(x, 0.0, false), body(x, 0.55, true)]);
            shapes.extend([surface, sphere.clone()]);
        } else {
            moving_indices.push(bodies.len());
            bodies.extend([body(x, 0.55, true), body(x, 0.0, false)]);
            shapes.extend([sphere.clone(), surface]);
        }
    }
    // Force the owned world to use expanded GPU bounds and compact LBVH candidates.
    for index in 0..9 {
        bodies.push(body(100.0 + index as f32 * 8.0, 0.0, false));
        shapes.push(sphere.clone());
    }
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        tested += 1;
        eprintln!("temporal surface world: {backend:?}");
        let mut world = GpuRigidSphereWorld::new_primitives(
            context.device(),
            context.queue(),
            &bodies,
            &shapes,
            GpuRigidSphereWorldConfig {
                gravity: [0.0; 3],
                ground_half_extent: None,
                sleep: SleepSettings {
                    enabled: false,
                    ..SleepSettings::default()
                },
                ..GpuRigidSphereWorldConfig::default()
            },
        )?;
        assert!(!world.uses_exhaustive_pairs());
        let settings = GpuRigidTemporalSolveParams {
            friction: 0.0,
            iterations: 8,
            ..GpuRigidTemporalSolveParams::default()
        };
        for frame in 0..3 {
            let count = world
                .step_temporal_speculative(0.1, 16, settings, 0.1)
                .unwrap_or_else(|error| {
                    panic!(
                        "{backend:?}: frame={frame}: {error}: contacts={:?}: states={:?}",
                        world.readback_contacts(),
                        world.readback()
                    )
                });
            assert_eq!(count, 4);
            assert!(!world.is_faulted());
            let states = world.readback()?;
            for index in &moving_indices {
                let state = states[*index];
                assert!(
                    state.position_inverse_mass[2] > 0.495,
                    "{backend:?}: frame={frame}: body={index}: {state:?}"
                );
                assert!(
                    state.angular_velocity[..3]
                        .iter()
                        .all(|value| value.abs() < 0.02),
                    "{backend:?}: frame={frame}: body={index}: {state:?}"
                );
            }
            for (index, initial) in bodies.iter().enumerate() {
                if initial.position_inverse_mass[3] == 0.0 {
                    assert_eq!(
                        states[index].position_inverse_mass,
                        initial.position_inverse_mass
                    );
                    assert_eq!(states[index].linear_velocity, initial.linear_velocity);
                }
            }
        }
    }
    assert!(tested > 0, "no GPU backend available");
    Ok(())
}

#[test]
fn triangle_meshes_resist_crossing_with_speculative_contacts()
-> Result<(), Box<dyn core::error::Error>> {
    let mesh = GpuRigidShape::TriangleMesh {
        vertices: vec![[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [0.0, 2.0, 0.0]],
        triangles: vec![[0, 1, 2]],
    };
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        tested += 1;
        for moving_first in [false, true] {
            let mut fixed = body(0.0, 0.0, false);
            fixed.inverse_inertia_sleep = [0.0; 4];
            let mut moving = body(0.0, 0.05, true);
            moving.inverse_inertia_sleep = [0.0; 4];
            let bodies = if moving_first {
                [moving, fixed]
            } else {
                [fixed, moving]
            };
            let mut world = GpuRigidSphereWorld::new_primitives(
                context.device(),
                context.queue(),
                &bodies,
                &[mesh.clone(), mesh.clone()],
                GpuRigidSphereWorldConfig {
                    gravity: [0.0; 3],
                    ground_half_extent: None,
                    sleep: SleepSettings {
                        enabled: false,
                        ..SleepSettings::default()
                    },
                    ..GpuRigidSphereWorldConfig::default()
                },
            )?;
            let settings = GpuRigidTemporalSolveParams {
                friction: 0.0,
                iterations: 8,
                ..GpuRigidTemporalSolveParams::default()
            };
            for frame in 0..3 {
                let count = world.step_temporal_speculative(0.1, 16, settings, 0.1)?;
                assert_eq!(count, 1, "{backend:?}: frame={frame}");
                let states = world.readback()?;
                let moving = states[usize::from(!moving_first)];
                assert!(
                    moving.position_inverse_mass[2] > -0.005,
                    "{backend:?}: moving_first={moving_first}: frame={frame}: {moving:?}"
                );
            }
        }
    }
    assert!(tested > 0, "no GPU backend available");
    Ok(())
}
