use super::GpuContactImpulseSensor;
use crate::gpu_articulated_ground_contact::GpuArticulatedGroundContactError;
use crate::gpu_contact_pipeline::GpuContactDevice;
use wgpu::util::DeviceExt;

#[test]
fn normal_impulse_sensor_adds_opposing_contacts_without_friction_or_cross_environment_leaks() {
    // Given: native packed contact words for two environments and distinct owners.
    let context = GpuContactDevice::new().unwrap();
    let device = context.device();
    let queue = context.queue();
    // Minimal row layout: owners, normal impulse, friction, and diagnostic signs.
    let row = |first, second, normal: f32, friction: f32, signs: [f32; 2]| {
        [
            first,
            0,
            second,
            normal.to_bits(),
            friction.to_bits(),
            signs[0].to_bits(),
            signs[1].to_bits(),
        ]
    };
    let rows = [
        row(1, 0, 2.0, 100.0, [1.0, 0.0]),
        row(0, 1, 3.0, -100.0, [0.0, -1.0]),
        row(0, 0, 40.0, 0.0, [0.0, 0.0]), // not a geometric contact
        row(3, 0, 7.0, 200.0, [1.0, 0.0]),
    ];
    let buffer = |label: &str, contents: &[u8]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        })
    };
    let contacts = buffer("sensor contact fixtures", bytemuck::cast_slice(&rows));
    let status = buffer("sensor source state", bytemuck::cast_slice(&[0_u32; 2]));
    let selections = [[0, 3, 1, 0], [0, 3, 0, 0], [3, 1, 3, 1], [3, 1, 2, 1]];
    let sensor = GpuContactImpulseSensor::new(
        device,
        &contacts,
        &status,
        &status,
        &selections,
        [7, 3, 5, 6],
        &[1, 0],
    )
    .unwrap();

    // When: the contact observation is computed on the actual required GPU.
    let mut encoder = device.create_command_encoder(&Default::default());
    sensor.encode(&mut encoder);
    let _submission = queue.submit(Some(encoder.finish()));
    let observed = sensor.readback(queue).unwrap();

    // Then: opposing normals add scalar magnitudes, not friction or other environments.
    assert_eq!(observed, [vec![5.0, 0.0], vec![7.0, 0.0]]);
    assert_eq!(sensor.selected_links(), [1, 0]);
}

#[test]
fn normal_impulse_sensor_overwrites_previous_observation_and_propagates_source_faults() {
    // Given: one selected native link with a completed previous observation.
    let context = GpuContactDevice::new().unwrap();
    let device = context.device();
    let queue = context.queue();
    let contacts = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("normal impulse source"),
        contents: bytemuck::cast_slice(&[0_u32, 0, 0, 2_f32.to_bits(), 1_f32.to_bits(), 0]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let status = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("normal impulse status"),
        contents: bytemuck::cast_slice(&[0_u32]),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    });
    let sensor = GpuContactImpulseSensor::new(
        device,
        &contacts,
        &status,
        &status,
        &[[0, 1, 0, 0]],
        [6, 3, 4, 5],
        &[0],
    )
    .unwrap();
    let observe = || {
        let mut encoder = device.create_command_encoder(&Default::default());
        sensor.encode(&mut encoder);
        let _submission = queue.submit(Some(encoder.finish()));
    };
    observe();
    assert_eq!(sensor.readback(queue).unwrap(), [vec![2.0]]);

    // When: the latest solve changes, then its source becomes invalid.
    queue.write_buffer(&contacts, 12, bytemuck::cast_slice(&[3_f32]));
    observe();

    // Then: this is last-substep output, not cumulative output.
    assert_eq!(sensor.readback(queue).unwrap(), [vec![3.0]]);
    queue.write_buffer(&status, 0, bytemuck::cast_slice(&[1_u32]));
    observe();
    assert!(matches!(
        sensor.readback(queue),
        Err(GpuArticulatedGroundContactError::SourceFault(0))
    ));
    let bytes =
        crate::gpu_articulated_mass::read_buffer(device, queue, sensor.output_buffer()).unwrap();
    assert!(f32::from_ne_bytes(bytes[..4].try_into().unwrap()).is_nan());
}

#[test]
fn normal_impulse_sensor_propagates_mass_fault_before_state_integration() {
    // Given: valid coordinate status but a failed upstream mass solve.
    let context = GpuContactDevice::new().unwrap();
    let device = context.device();
    let queue = context.queue();
    let buffer = |label: &str, contents: &[u8]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        })
    };
    let contacts = buffer(
        "mass fault contact fixture",
        bytemuck::cast_slice(&[0_u32, 0, 0, 2_f32.to_bits(), 1_f32.to_bits(), 0]),
    );
    let status = buffer("healthy coordinate status", bytemuck::cast_slice(&[0_u32]));
    let mass_status = buffer("failed mass source", bytemuck::cast_slice(&[1_u32]));
    let sensor = GpuContactImpulseSensor::new(
        device,
        &contacts,
        &status,
        &mass_status,
        &[[0, 1, 0, 0]],
        [6, 3, 4, 5],
        &[0],
    )
    .unwrap();

    // When: a policy samples contacts before the state integrator propagates the fault.
    let mut encoder = device.create_command_encoder(&Default::default());
    sensor.encode(&mut encoder);
    let _submission = queue.submit(Some(encoder.finish()));

    // Then: neither host nor resident consumers receive a plausible healthy observation.
    assert!(matches!(
        sensor.readback(queue),
        Err(GpuArticulatedGroundContactError::SourceFault(0))
    ));
    let bytes =
        crate::gpu_articulated_mass::read_buffer(device, queue, sensor.output_buffer()).unwrap();
    assert!(f32::from_ne_bytes(bytes[..4].try_into().unwrap()).is_nan());
}
