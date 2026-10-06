//! Run a small GPU-resident 3D sphere scene without CPU state transfers per step.

use tessera_physics::gpu_contact_pipeline::GpuContactDevice;
use tessera_physics::gpu_rigid_sphere_world::{GpuRigidSphereWorld, GpuRigidSphereWorldConfig};
use tessera_physics::gpu_rigid_state::GpuRigidBodyState;

fn sphere(x: f32, velocity_x: f32) -> GpuRigidBodyState {
    GpuRigidBodyState {
        position_inverse_mass: [x, 0.0, 3.0, 1.0],
        orientation: [0.0, 0.0, 0.0, 1.0],
        linear_velocity: [velocity_x, 0.0, 0.0, 0.0],
        angular_velocity: [0.0; 4],
        inverse_inertia_sleep: [2.5, 2.5, 2.5, 0.0],
    }
}

fn main() -> Result<(), Box<dyn core::error::Error>> {
    let gpu = GpuContactDevice::new()?;
    let mut config = GpuRigidSphereWorldConfig {
        gravity: [0.0; 3],
        ground_half_extent: None,
        ..Default::default()
    };
    config.solve.bias_factor = 0.0;
    let mut world = GpuRigidSphereWorld::new(
        gpu.device(),
        gpu.queue(),
        &[sphere(-1.0, 1.0), sphere(1.0, -1.0)],
        &[1.0, 1.0],
        config,
    )?;
    let _candidate_count = world.step_substeps(1.0 / 120.0, 120)?;
    for (index, state) in world.readback()?.iter().enumerate() {
        println!(
            "sphere {index}: x={:.4}, vx={:.4}, sleeping={}",
            state.position_inverse_mass[0],
            state.linear_velocity[0],
            state.inverse_inertia_sleep[3] != 0.0,
        );
    }
    Ok(())
}
