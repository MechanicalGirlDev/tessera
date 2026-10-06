//! Compare one-world latency and independent-world throughput for contact solves.

use core::time::Duration;
use std::time::Instant;

use nalgebra::{DMatrix, DVector};
use tessera_physics::contact_reference::{
    ContactConstraint, ContactProblem, ContactSolution, SolveParams, solve_contacts,
};
use tessera_physics::gpu_contact_solver::{
    GpuContactReadback, GpuContactSolveRequest, GpuContactSolver,
};

fn problem(environments: usize) -> ContactProblem {
    let width = environments * 3;
    let mut velocity = DVector::zeros(width);
    let mut contacts = Vec::with_capacity(environments);
    for environment in 0..environments {
        let base = environment * 3;
        velocity[base] = 0.5;
        velocity[base + 2] = -1.0;
        let mut normal = DVector::zeros(width);
        let mut tangent_x = DVector::zeros(width);
        let mut tangent_y = DVector::zeros(width);
        normal[base + 2] = 1.0;
        tangent_x[base] = 1.0;
        tangent_y[base + 1] = 1.0;
        contacts.push(ContactConstraint {
            scalar: None,
            normal,
            tangents: [tangent_x, tangent_y],
            penetration: 0.01,
            friction: 0.5,
            restitution: 0.0,
        });
    }
    ContactProblem {
        inverse_mass: DMatrix::identity(width, width),
        velocity,
        contacts,
    }
}

fn connected_branches(branches: usize) -> ContactProblem {
    let width = branches + 1;
    let constraint = |normal: DVector<f64>| ContactConstraint {
        scalar: None,
        normal,
        tangents: [DVector::zeros(width), DVector::zeros(width)],
        penetration: 0.0,
        friction: 0.0,
        restitution: 0.0,
    };
    let mut contacts = vec![constraint(DVector::from_element(width, 1.0))];
    for coordinate in 1..width {
        let mut normal = DVector::zeros(width);
        normal[coordinate] = 1.0;
        contacts.push(constraint(normal));
    }
    ContactProblem {
        inverse_mass: DMatrix::identity(width, width),
        velocity: DVector::from_iterator(
            width,
            (0..width).map(|index| if index % 2 == 0 { -2.0 } else { 0.0 }),
        ),
        contacts,
    }
}

fn average(total: Duration, samples: u32) -> f64 {
    total.as_secs_f64() * 1_000_000.0 / f64::from(samples)
}

fn solve_encoded_many(
    solver: &GpuContactSolver,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    problems: &[ContactProblem],
    params: SolveParams,
) -> Result<Vec<ContactSolution>, Box<dyn core::error::Error>> {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Tessera independent contact batch"),
    });
    let outputs = problems
        .iter()
        .map(|problem| {
            solver
                .encode(device, &mut encoder, problem, params, None)?
                .ok_or_else(|| "benchmark problem has no contacts".into())
        })
        .collect::<Result<Vec<_>, Box<dyn core::error::Error>>>()?;
    let readbacks = outputs
        .iter()
        .map(|output| output.encode_readback(device, &mut encoder))
        .collect();
    let _submission = queue.submit(Some(encoder.finish()));
    Ok(GpuContactReadback::finish_many(readbacks, device)?)
}

fn main() -> Result<(), Box<dyn core::error::Error>> {
    let samples = std::env::args()
        .nth(1)
        .map(|value| value.parse::<u32>())
        .transpose()?
        .unwrap_or(10);
    if samples == 0 {
        return Err("sample count must be positive".into());
    }
    let instance = wgpu::Instance::default();
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
    let info = adapter.get_info();
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))?;
    let solver = GpuContactSolver::new(&device);
    let params = SolveParams {
        dt: 0.01,
        iterations: 12,
        position_gain: 0.2,
        max_correction_speed: 2.0,
    };
    println!(
        "adapter={} backend={:?} profile={} samples={samples}",
        info.name,
        info.backend,
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
    );
    println!("environments,cpu_us,gpu_dense_us,gpu_encoded_many_us,gpu_local_packed_us");
    for environments in [1, 16, 64] {
        let input = problem(environments);
        let separate = (0..environments).map(|_| problem(1)).collect::<Vec<_>>();
        let requests = separate
            .iter()
            .map(|problem| GpuContactSolveRequest {
                problem,
                warm_start: None,
            })
            .collect::<Vec<_>>();
        let reference = solve_contacts(&input, params, None)?;
        let gpu = solver.solve(&device, &queue, &input, params, None)?;
        for (actual, expected) in gpu.velocity.iter().zip(reference.velocity.iter()) {
            if (actual - expected).abs() > 1e-4 {
                return Err("CPU/GPU solve mismatch".into());
            }
        }
        let cpu_start = Instant::now();
        for _ in 0..samples {
            let _result = solve_contacts(&input, params, None)?;
        }
        let cpu_duration = cpu_start.elapsed();
        let gpu_start = Instant::now();
        for _ in 0..samples {
            let _result = solver.solve(&device, &queue, &input, params, None)?;
        }
        let gpu_duration = gpu_start.elapsed();
        let gpu_us = average(gpu_duration, samples);
        let _warmup = solve_encoded_many(&solver, &device, &queue, &separate, params)?;
        let encoded_start = Instant::now();
        for _ in 0..samples {
            let _result = solve_encoded_many(&solver, &device, &queue, &separate, params)?;
        }
        let encoded_duration = encoded_start.elapsed();
        let packed = solver.solve_packed(&device, &queue, &requests, params)?;
        for solution in &packed {
            for (actual, expected) in solution
                .velocity
                .iter()
                .zip(reference.velocity.iter().take(3))
            {
                if (actual - expected).abs() > 1e-4 {
                    return Err("packed CPU/GPU solve mismatch".into());
                }
            }
        }
        let packed_start = Instant::now();
        for _ in 0..samples {
            let _result = solver.solve_packed(&device, &queue, &requests, params)?;
        }
        let packed_duration = packed_start.elapsed();
        println!(
            "{environments},{:.1},{:.1},{:.1},{:.1}",
            average(cpu_duration, samples),
            gpu_us,
            average(encoded_duration, samples),
            average(packed_duration, samples)
        );
    }
    println!("connected_branches,cpu_us,gpu_colored_us");
    for branches in [8, 32, 64] {
        let input = connected_branches(branches);
        let reference = solve_contacts(&input, params, None)?;
        let gpu = solver.solve(&device, &queue, &input, params, None)?;
        for (actual, expected) in gpu.velocity.iter().zip(reference.velocity.iter()) {
            if (actual - expected).abs() > 1e-4 {
                return Err("connected CPU/GPU solve mismatch".into());
            }
        }
        let cpu_start = Instant::now();
        for _ in 0..samples {
            let _result = solve_contacts(&input, params, None)?;
        }
        let cpu_duration = cpu_start.elapsed();
        let gpu_start = Instant::now();
        for _ in 0..samples {
            let _result = solver.solve(&device, &queue, &input, params, None)?;
        }
        println!(
            "{branches},{:.1},{:.1}",
            average(cpu_duration, samples),
            average(gpu_start.elapsed(), samples)
        );
    }
    Ok(())
}
