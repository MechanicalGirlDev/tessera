//! Run overlapping sphere environments in one GPU-resident batch.

use tessera_physics::gpu_contact_pipeline::GpuContactDevice;
use tessera_physics::gpu_rigid_sphere_world::{
    GpuRigidSphereBatch, GpuRigidSphereEnvironment, GpuRigidSphereWorldConfig,
};
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
    println!("adapter: {:?}", gpu.adapter_info());
    let initial = [sphere(-1.0, 1.0), sphere(1.0, -1.0)];
    let radii = [1.0; 2];
    let environments = vec![
        GpuRigidSphereEnvironment {
            states: &initial,
            radii: &radii,
        };
        64
    ];
    let mut config = GpuRigidSphereWorldConfig {
        gravity: [0.0; 3],
        ground_half_extent: None,
        ..Default::default()
    };
    config.solve.bias_factor = 0.0;
    let mut batch = GpuRigidSphereBatch::new(gpu.device(), gpu.queue(), &environments, config)?;
    let _candidate_count = batch.step_substeps(1.0 / 120.0, 120)?;
    for environment in [0, batch.len() - 1] {
        let states = batch.readback_environment(environment)?;
        println!(
            "environment {environment}: x=({:.4}, {:.4}), sleeping=({}, {})",
            states[0].position_inverse_mass[0],
            states[1].position_inverse_mass[0],
            states[0].inverse_inertia_sleep[3] != 0.0,
            states[1].inverse_inertia_sleep[3] != 0.0,
        );
    }
    Ok(())
}
