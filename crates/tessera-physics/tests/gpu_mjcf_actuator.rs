//! Real GPU tests of imported stateless actuator feedback.
#![cfg(feature = "gpu-contact")]

extern crate alloc;

use alloc::collections::BTreeMap;
use core::time::Duration;
use tessera_physics::{
    gpu_contact_pipeline::GpuContactDevice,
    gpu_mjcf_actuator::{GpuMjcfActuatorBatch, GpuMjcfActuatorInput},
    mjcf::{MjcfActuatorInfo, MjcfActuatorKind},
    wgpu,
};
use wgpu::util::DeviceExt;

type TestResult<T = ()> = Result<T, Box<dyn core::error::Error>>;

fn entry(coordinate: Option<usize>, kind: MjcfActuatorKind) -> MjcfActuatorInfo {
    MjcfActuatorInfo {
        name: "test".into(),
        joint: "joint".into(),
        coordinate,
        kind,
        source_kind: "test".into(),
        attributes: BTreeMap::new(),
        unsupported_reason: None,
        gear: 1.0,
        control_range: None,
        force_range: None,
    }
}

fn storage(device: &wgpu::Device, values: &[f32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("actuator test scalar storage"),
        contents: bytemuck::cast_slice(values),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    })
}

fn readback(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    submission: wgpu::SubmissionIndex,
) -> TestResult<Vec<f32>> {
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _delivery = sender.send(result);
        });
    let _poll_status = device.poll(wgpu::PollType::Wait {
        submission_index: Some(submission),
        timeout: Some(Duration::from_secs(30)),
    })?;
    receiver.recv_timeout(Duration::from_secs(30))??;
    let view = buffer.slice(..).get_mapped_range();
    let result = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
    drop(view);
    buffer.unmap();
    Ok(result)
}

fn staging(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("actuator test output"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}

#[test]
fn feedback_reads_each_resident_microstep_and_aggregates_heterogeneous_layouts() -> TestResult {
    let context = GpuContactDevice::new_with_backends(wgpu::Backends::VULKAN)?;
    eprintln!("MJCF feedback GPU adapter: {:?}", context.adapter_info());
    let device = context.device();
    let queue = context.queue();
    let mut motor = entry(Some(0), MjcfActuatorKind::Motor);
    motor.gear = -2.0;
    motor.control_range = Some([-1.0, 2.0]);
    motor.force_range = Some([-3.0, 1.0]);
    let mut position = entry(Some(0), MjcfActuatorKind::Position { kp: 4.0, kv: 2.0 });
    position.gear = 99.0; // Servos do not apply gear.
    position.force_range = Some([-1.0, 2.0]);
    let mut velocity = entry(Some(1), MjcfActuatorKind::Velocity { kv: 3.0 });
    velocity.gear = -99.0;
    velocity.force_range = Some([-2.0, 4.0]);
    let first = [
        motor,
        position,
        velocity,
        entry(Some(1), MjcfActuatorKind::Damper { gain: 2.0 }),
        entry(None, MjcfActuatorKind::Unsupported),
    ];
    let mut affine = entry(
        Some(0),
        MjcfActuatorKind::General {
            gain: 2.0,
            bias: [1.0, -3.0, -4.0],
            affine: true,
        },
    );
    affine.gear = 101.0;
    let mut fixed = entry(
        Some(1),
        MjcfActuatorKind::General {
            gain: 3.0,
            bias: [999.0; 3],
            affine: false,
        },
    );
    fixed.gear = -2.0;
    fixed.force_range = Some([-5.0, -1.0]);
    let second = [affine, fixed, entry(Some(2), MjcfActuatorKind::Unsupported)];
    let mut q1 = [0.0; 14];
    let mut v1 = [0.0; 14];
    q1[6] = 1.0;
    v1[6] = 0.5;
    v1[7] = 2.0;
    q1[10] = 2.0;
    v1[10] = -1.0;
    let mut q2 = q1;
    let mut v2 = v1;
    q2[6] = 3.0;
    v2[6] = -1.0;
    v2[7] = -2.0;
    q2[10] = -1.0;
    v2[10] = 0.25;
    let q = storage(device, &q1);
    let v = storage(device, &v1);
    let force = storage(device, &[10.0; 14]);
    let q_next = storage(device, &q2);
    let v_next = storage(device, &v2);
    let baseline = storage(device, &[10.0; 14]);
    let inputs = [
        GpuMjcfActuatorInput {
            coordinates: 0..8,
            joint_prefix: 6,
            entries: &first,
            controls: &[3.0, 2.0, 1.0, -0.5, 999.0],
        },
        GpuMjcfActuatorInput {
            coordinates: 10..14,
            joint_prefix: 0,
            entries: &second,
            controls: &[0.5, 2.0, 999.0],
        },
    ];
    let owner = GpuMjcfActuatorBatch::new(device, queue, &q, &v, &force, &inputs)?;
    let bytes = force.size();
    let result = staging(device, bytes * 2);
    let mut encoder = device.create_command_encoder(&Default::default());
    owner.encode(&mut encoder);
    encoder.copy_buffer_to_buffer(&force, 0, &result, 0, bytes);
    // Change the resident state without any host state readback between feedback passes.
    encoder.copy_buffer_to_buffer(&q_next, 0, &q, 0, bytes);
    encoder.copy_buffer_to_buffer(&v_next, 0, &v, 0, bytes);
    encoder.copy_buffer_to_buffer(&baseline, 0, &force, 0, bytes);
    owner.encode(&mut encoder);
    encoder.copy_buffer_to_buffer(&force, 0, &result, bytes, bytes);
    let submission = queue.submit(Some(encoder.finish()));
    let actual = readback(device, &result, submission)?;
    let mut expected1 = [10.0f32; 14];
    expected1[6] = 9.0;
    expected1[7] = 6.0;
    expected1[10] = 10.0;
    expected1[11] = 5.0;
    let mut expected2 = expected1;
    expected2[6] = 6.0;
    expected2[7] = 16.0;
    expected2[10] = 14.0;
    for (actual, expected) in actual.iter().zip(expected1.iter().chain(&expected2)) {
        assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
    }
    // Failed updates must leave all held controls unchanged.
    assert!(owner.update_controls(0, &[0.0]).is_err());
    assert!(
        owner
            .update_controls(0, &[0.0, 0.0, 0.0, 0.0, f64::NAN])
            .is_err()
    );
    assert!(owner.update_controls(2, &[]).is_err());
    owner.update_controls(1, &[1.5, 2.0, 999.0])?;
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(&baseline, 0, &force, 0, bytes);
    owner.encode(&mut encoder);
    encoder.copy_buffer_to_buffer(&force, 0, &result, 0, bytes);
    let submission = queue.submit(Some(encoder.finish()));
    expected2[10] = 16.0;
    for (actual, expected) in readback(device, &result, submission)?
        .iter()
        .take(14)
        .zip(expected2)
    {
        assert!((actual - expected).abs() < 1e-5);
    }
    Ok(())
}

#[test]
fn rejects_invalid_layout_metadata_and_unrepresentable_controls() -> TestResult {
    let context = GpuContactDevice::new_with_backends(wgpu::Backends::VULKAN)?;
    let device = context.device();
    let q = storage(device, &[0.0; 8]);
    let v = storage(device, &[0.0; 8]);
    let force = storage(device, &[0.0; 8]);
    let definitions = [entry(Some(0), MjcfActuatorKind::Motor)];
    let valid = GpuMjcfActuatorInput {
        coordinates: 0..8,
        joint_prefix: 6,
        entries: &definitions,
        controls: &[0.0],
    };
    let build = |inputs: &[GpuMjcfActuatorInput<'_>]| {
        GpuMjcfActuatorBatch::new(device, context.queue(), &q, &v, &force, inputs)
    };
    assert!(build(&[valid.clone(), valid.clone()]).is_err());
    assert!(
        build(&[GpuMjcfActuatorInput {
            joint_prefix: 9,
            ..valid.clone()
        }])
        .is_err()
    );
    assert!(
        build(&[GpuMjcfActuatorInput {
            coordinates: 0..9,
            ..valid.clone()
        }])
        .is_err()
    );
    assert!(
        build(&[GpuMjcfActuatorInput {
            controls: &[f64::MAX],
            ..valid.clone()
        }])
        .is_err()
    );
    assert!(
        build(&[GpuMjcfActuatorInput {
            controls: &[],
            ..valid.clone()
        }])
        .is_err()
    );
    for coordinate in [None, Some(2)] {
        let entries = [entry(coordinate, MjcfActuatorKind::Motor)];
        assert!(
            build(&[GpuMjcfActuatorInput {
                entries: &entries,
                ..valid.clone()
            }])
            .is_err()
        );
    }
    let mut invalid_range = definitions.clone();
    invalid_range[0].force_range = Some([1.0, -1.0]);
    assert!(
        build(&[GpuMjcfActuatorInput {
            entries: &invalid_range,
            ..valid
        }])
        .is_err()
    );
    Ok(())
}

#[test]
fn unsupported_only_and_empty_batches_leave_forces_untouched() -> TestResult {
    let context = GpuContactDevice::new_with_backends(wgpu::Backends::VULKAN)?;
    let device = context.device();
    let q = storage(device, &[0.0; 2]);
    let v = storage(device, &[0.0; 2]);
    let force = storage(device, &[3.0, -7.0]);
    let mut unsupported = entry(Some(usize::MAX), MjcfActuatorKind::Unsupported);
    unsupported.gear = f64::NAN; // Undriven metadata does not participate.
    let entries = [unsupported];
    let inputs = [GpuMjcfActuatorInput {
        coordinates: 0..2,
        joint_prefix: 0,
        entries: &entries,
        controls: &[42.0],
    }];
    let owner = GpuMjcfActuatorBatch::new(device, context.queue(), &q, &v, &force, &inputs)?;
    let empty = GpuMjcfActuatorBatch::new(device, context.queue(), &q, &v, &force, &[])?;
    owner.update_controls(0, &[0.0])?;
    let result = staging(device, force.size());
    let mut encoder = device.create_command_encoder(&Default::default());
    owner.encode(&mut encoder);
    empty.encode(&mut encoder);
    encoder.copy_buffer_to_buffer(&force, 0, &result, 0, force.size());
    let submission = context.queue().submit(Some(encoder.finish()));
    for (actual, expected) in readback(device, &result, submission)?
        .iter()
        .zip([3.0f32, -7.0])
    {
        assert!((actual - expected).abs() < 1e-6);
    }
    Ok(())
}
