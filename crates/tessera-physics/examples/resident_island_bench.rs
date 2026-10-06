//! Measure end-to-end GPU-resident contact steps for sparse and connected scenes.

use std::time::Instant;

use tessera_physics::gpu_contact_pipeline::GpuContactDevice;
use tessera_physics::gpu_rigid_sphere_world::{GpuRigidSphereWorld, GpuRigidSphereWorldConfig};
use tessera_physics::gpu_rigid_state::GpuRigidBodyState;

fn sphere(x: f32) -> GpuRigidBodyState {
    GpuRigidBodyState {
        position_inverse_mass: [x, 0.0, 3.0, 1.0],
        orientation: [0.0, 0.0, 0.0, 1.0],
        linear_velocity: [0.0; 4],
        angular_velocity: [0.0; 4],
        inverse_inertia_sleep: [1.0, 1.0, 1.0, 0.0],
    }
}

fn main() -> Result<(), Box<dyn core::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let body_count = args
        .first()
        .map_or(Ok(70), |value| value.parse::<usize>())?;
    let samples = args.get(1).map_or(Ok(50), |value| value.parse::<usize>())?;
    let mode = args.get(2).map_or("independent", String::as_str);
    if body_count < 17 || samples == 0 || !matches!(mode, "independent" | "chain") {
        return Err(
            "usage: resident_island_bench [bodies>=17] [samples>0] [independent|chain] [default|vulkan|dx12|metal]".into(),
        );
    }
    if mode == "independent" && body_count % 2 != 0 {
        return Err("independent mode requires an even body count".into());
    }
    let states = (0..body_count)
        .map(|index| {
            let x = if mode == "chain" {
                index as f32 * 1.5
            } else {
                (index / 2) as f32 * 10.0 + (index % 2) as f32 * 1.5
            };
            sphere(x)
        })
        .collect::<Vec<_>>();
    let gpu = match args.get(3).map(String::as_str) {
        None | Some("default") => GpuContactDevice::new()?,
        Some("vulkan") => GpuContactDevice::new_with_backends(wgpu::Backends::VULKAN)?,
        Some("dx12") => GpuContactDevice::new_with_backends(wgpu::Backends::DX12)?,
        Some("metal") => GpuContactDevice::new_with_backends(wgpu::Backends::METAL)?,
        _ => return Err("backend must be default, vulkan, dx12, or metal".into()),
    };
    let adapter = gpu.adapter_info();
    let mut config = GpuRigidSphereWorldConfig {
        gravity: [0.0; 3],
        ground_half_extent: None,
        ..Default::default()
    };
    config.solve.bias_factor = 0.0;
    let mut world = GpuRigidSphereWorld::new(
        gpu.device(),
        gpu.queue(),
        &states,
        &vec![1.0; body_count],
        config,
    )?;
    let expected_pairs = if mode == "chain" {
        body_count - 1
    } else {
        body_count / 2
    };
    for _ in 0..5 {
        let _buffers = world.step_gpu_resident(1.0 / 120.0)?;
    }
    let mut times_us = Vec::with_capacity(samples);
    for _ in 0..samples {
        let start = Instant::now();
        let _buffers = world.step_gpu_resident(1.0 / 120.0)?;
        times_us.push(start.elapsed().as_secs_f64() * 1_000_000.0);
        if world.candidate_pair_count() != expected_pairs {
            return Err("candidate count changed during the benchmark".into());
        }
    }
    times_us.sort_by(f64::total_cmp);
    let percentile = |percent: usize| times_us[(samples - 1) * percent / 100];
    let mean = times_us.iter().sum::<f64>() / samples as f64;
    println!(
        "adapter={:?} backend={:?} driver={} bodies={body_count} mode={mode} candidates={expected_pairs} samples={samples} mean_us={mean:.1} p50_us={:.1} p99_us={:.1}",
        adapter.name,
        adapter.backend,
        adapter.driver,
        percentile(50),
        percentile(99),
    );
    Ok(())
}
