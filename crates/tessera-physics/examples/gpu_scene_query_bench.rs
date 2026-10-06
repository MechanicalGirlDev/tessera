//! End-to-end scene-query timings, including bounds/tree build and result readback.
//! Run with `--features gpu-contact --example gpu_scene_query_bench -- [body_count] [samples]`.
use std::time::Instant;
use tessera_physics::{
    gpu_contact_pipeline::GpuContactDevice,
    gpu_point_query::GpuRigidPoint,
    gpu_rigid_sphere_contact::GpuRigidCollisionGroups,
    gpu_rigid_sphere_world::{GpuRigidSphereWorld, GpuRigidSphereWorldConfig},
    gpu_rigid_state::GpuRigidBodyState,
};
fn main() -> Result<(), Box<dyn core::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let count = args.first().map_or(Ok(4096), |s| s.parse::<usize>())?;
    let samples = args.get(1).map_or(Ok(3), |s| s.parse::<usize>())?;
    if count < 17 || samples == 0 || count > u32::MAX as usize {
        return Err("body_count >= 17 and samples > 0 required".into());
    }
    let context = GpuContactDevice::new_with_backends(wgpu::Backends::VULKAN)?;
    println!(
        "adapter={:?}, backend={:?}",
        context.adapter_info().name,
        context.adapter_info().backend
    );
    let states = (0..count)
        .map(|i| GpuRigidBodyState {
            position_inverse_mass: [
                (i % 16) as f32 * 4.0,
                ((i / 16) % 16) as f32 * 4.0,
                (i / 256) as f32 * 4.0 + 3.0,
                0.0,
            ],
            orientation: [0.0, 0.0, 0.0, 1.0],
            linear_velocity: [0.0; 4],
            angular_velocity: [0.0; 4],
            inverse_inertia_sleep: [0.0; 4],
        })
        .collect::<Vec<_>>();
    let world = GpuRigidSphereWorld::new(
        context.device(),
        context.queue(),
        &states,
        &vec![0.5; count],
        GpuRigidSphereWorldConfig {
            gravity: [0.0; 3],
            ground_half_extent: None,
            ..Default::default()
        },
    )?;
    let points = states
        .iter()
        .map(|s| GpuRigidPoint {
            point: [
                s.position_inverse_mass[0] + 0.9,
                s.position_inverse_mass[1],
                s.position_inverse_mass[2],
            ],
            max_distance: 1.0,
            groups: GpuRigidCollisionGroups::default(),
            excluded_body: None,
            body_range: None,
            solid: false,
        })
        .collect::<Vec<_>>();
    let mut times = [Vec::new(), Vec::new()];
    for sample in 0..=samples {
        // Alternate order to avoid consistently giving one method a warmer device.
        for offset in 0..2 {
            let mode = (sample + offset) % 2;
            let begin = Instant::now();
            let mut encoder = context.device().create_command_encoder(&Default::default());
            let queries = if mode == 0 {
                let queries = world.prepare_point_queries(&points)?;
                queries.encode(&mut encoder);
                queries
            } else {
                world.encode_point_queries(&mut encoder, &points)?
            };
            let _ = context.queue().submit(Some(encoder.finish()));
            let hits = queries.readback(context.device(), context.queue())?;
            let elapsed = begin.elapsed().as_secs_f64() * 1000.0;
            for (index, hit) in hits.iter().enumerate() {
                if hit.ids != [index as u32, 0, 1, 0] || (hit.point_distance[3] - 0.4).abs() > 2e-5
                {
                    return Err(format!("query {index}: {hit:?}").into());
                }
            }
            if sample > 0 {
                times[mode].push(elapsed);
            }
        }
    }
    for (name, values) in ["linear", "lbvh"].into_iter().zip(&mut times) {
        values.sort_by(f64::total_cmp);
        println!(
            "{name}: bodies={count}, queries={}, samples={samples}, median_ms={:.3}, min_ms={:.3}, max_ms={:.3}",
            points.len(),
            values[values.len() / 2],
            values[0],
            values[values.len() - 1]
        );
    }
    println!(
        "Includes CPU allocation/encoding, GPU execution and result mapping; excludes body-state readback. Every result checked against analytic sphere geometry."
    );
    Ok(())
}
