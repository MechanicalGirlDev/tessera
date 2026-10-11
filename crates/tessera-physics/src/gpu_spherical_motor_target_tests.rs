use super::*;
use crate::articulated_world::MotorModel;
use crate::gpu_articulated_mass::read_buffer;
use crate::gpu_contact_pipeline::GpuContactDevice;
use nalgebra::{Quaternion, UnitQuaternion, Vector3};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TestDrive {
    indices: [u32; 4],
    target: [f32; 4],
    velocity: [f32; 4],
    stiffness: [f32; 4],
    damping: [f32; 4],
    cap: [f32; 4],
}

fn drive() -> SphericalJointDrive {
    SphericalJointDrive {
        model: MotorModel::ForceBased,
        orientation_target: UnitQuaternion::identity(),
        velocity_target: Vector3::zeros(),
        stiffness: Vector3::repeat(2.0),
        damping: Vector3::repeat(3.0),
        max_torque: Vector3::repeat(4.0),
    }
}
fn mapping(joint: usize, axis: usize, mode: GpuMotorTargetMode) -> GpuSphericalMotorTargetMapping {
    GpuSphericalMotorTargetMapping { joint, axis, mode }
}
fn read_drives(context: &GpuContactDevice, buffer: &wgpu::Buffer) -> Vec<TestDrive> {
    read_buffer(context.device(), context.queue(), buffer)
        .unwrap()
        .chunks_exact(96)
        .map(bytemuck::pod_read_unaligned)
        .collect()
}
fn rotation(drive: &TestDrive) -> Vector3<f32> {
    let [x, y, z, w] = drive.target;
    UnitQuaternion::new_normalize(Quaternion::new(w, x, y, z)).scaled_axis()
}
fn close(actual: Vector3<f32>, expected: Vector3<f32>) {
    assert!(
        (actual - expected).norm() < 2e-5,
        "{actual:?} != {expected:?}"
    );
}

#[test]
fn resident_spherical_scatter_gpu() {
    use GpuMotorTargetMode::{Position, Velocity};
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        tested += 1;
        let device = context.device();
        let queue = context.queue();
        let q = UnitQuaternion::from_scaled_axis(Vector3::new(0.2f32, -0.3, 0.4));
        let target: [f32; 4] = q.quaternion().coords.into();
        let first = TestDrive {
            indices: [0, 0, 0, 1],
            target,
            velocity: [1.0, 2.0, 3.0, 0.0],
            stiffness: [2.0; 4],
            damping: [3.0; 4],
            cap: [4.0; 4],
        };
        let mut negative = first;
        negative.target = target.map(|v| -v);
        negative.indices[1] = 1;
        let initial = [first, first, negative, negative];
        let parameters = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Spherical scatter test drives"),
            contents: bytemuck::cast_slice(&initial),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let drives = [vec![Some(drive()); 2], vec![Some(drive()); 2]];
        // Mixed modes on one joint and velocity scatter to another joint.
        let entries = vec![
            mapping(0, 0, Position),
            mapping(0, 1, Position),
            mapping(0, 0, Velocity),
            mapping(1, 2, Velocity),
        ];
        let control = GpuSphericalMotorTargetControl::new(
            device,
            &parameters,
            &drives,
            &[entries.clone(), entries.clone()],
            &[0, 0],
        )
        .unwrap();
        assert_eq!(control.action_bytes(), 32);
        assert!(control.accepts_inputs(&drives));
        assert!(!control.accepts_inputs(&[vec![None; 2], drives[1].clone()]));
        let actions = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU-produced actions"),
            size: 32,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Actual GPU producer, not a queue upload: consumer follows in the same encoder.
        let producer_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Spherical scatter action producer"),
            source: wgpu::ShaderSource::Wgsl(
                "@group(0) @binding(0) var<storage, read_write> a: array<f32>;
                 @compute @workgroup_size(1) fn produce() {
                    a[0]=0.6; a[1]=0.6; a[2]=-0.7; a[3]=-0.7;
                    a[4]=8.0; a[5]=8.0; a[6]=9.0; a[7]=9.0;
                 }"
                .into(),
            ),
        });
        let producer = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Action producer"),
            layout: None,
            module: &producer_shader,
            entry_point: Some("produce"),
            compilation_options: Default::default(),
            cache: None,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &producer.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: actions.as_entire_binding(),
            }],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&producer);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        control.encode_latch(device, &mut encoder, &actions);
        control.encode_apply(device, &mut encoder);
        let _submission = queue.submit(Some(encoder.finish()));
        let actual = read_drives(&context, &parameters);
        for i in [0, 2] {
            close(rotation(&actual[i]), Vector3::new(0.6, -0.7, 0.4));
            close(
                Vector3::from_column_slice(&actual[i].velocity[..3]),
                Vector3::new(8.0, 2.0, 3.0),
            );
            assert!(actual[i].target[3] >= 0.0);
            assert_eq!(actual[i].stiffness, first.stiffness);
            assert_eq!(actual[i].damping, first.damping);
            assert_eq!(actual[i].cap, first.cap);
        }
        for i in [1, 3] {
            assert_eq!(actual[i].target, initial[i].target); // velocity does not canonicalize position
            close(
                Vector3::from_column_slice(&actual[i].velocity[..3]),
                Vector3::new(1.0, 2.0, 9.0),
            );
        }
        assert_eq!(actual[0].target, actual[2].target);

        // Delayed latch skips precisely d applies; replacement restarts the delay.
        let delay = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[2u32, 2]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        queue.write_buffer(
            &actions,
            0,
            bytemuck::cast_slice(&[0.1f32, 0.1, 0.2, 0.2, 4.0, 4.0, 5.0, 5.0]),
        );
        let mut encoder = device.create_command_encoder(&Default::default());
        control.encode_delays(device, &mut encoder, &delay);
        control.encode_latch(device, &mut encoder, &actions);
        control.encode_apply(device, &mut encoder);
        let _submission = queue.submit(Some(encoder.finish()));
        assert_eq!(
            read_drives(&context, &parameters)[0].target,
            actual[0].target
        );
        queue.write_buffer(
            &actions,
            0,
            bytemuck::cast_slice(&[0.9f32, 0.9, 0.8, 0.8, 6.0, 6.0, 7.0, 7.0]),
        );
        let mut encoder = device.create_command_encoder(&Default::default());
        control.encode_latch(device, &mut encoder, &actions);
        control.encode_apply(device, &mut encoder);
        control.encode_apply(device, &mut encoder);
        let _submission = queue.submit(Some(encoder.finish()));
        let delayed = read_drives(&context, &parameters);
        assert_eq!(delayed[0].target, actual[0].target);
        assert_eq!(delayed[2].target, actual[2].target);

        // Selected environment cancellation and owner snapshot restoration are ordered.
        let snapshot = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&initial),
            usage: wgpu::BufferUsages::COPY_SRC,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        control.encode_cancel_environment(&mut encoder, 0).unwrap();
        encoder.copy_buffer_to_buffer(&snapshot, 0, &parameters, 0, 192);
        control.encode_apply(device, &mut encoder);
        let _submission = queue.submit(Some(encoder.finish()));
        let reset = read_drives(&context, &parameters);
        assert_eq!(reset[0].target, initial[0].target);
        assert_eq!(reset[0].velocity, initial[0].velocity);
        close(rotation(&reset[2]), Vector3::new(0.9, 0.8, 0.4));
        let mut encoder = device.create_command_encoder(&Default::default());
        assert!(control.encode_cancel_environment(&mut encoder, 2).is_err());

        // Delay replacement of already queued actions is also resident.
        let mut encoder = device.create_command_encoder(&Default::default());
        control.encode_latch(device, &mut encoder, &actions);
        let _submission = queue.submit(Some(encoder.finish()));
        queue.write_buffer(&delay, 0, bytemuck::cast_slice(&[0u32, 0]));
        let mut encoder = device.create_command_encoder(&Default::default());
        control.encode_delays(device, &mut encoder, &delay);
        control.encode_apply(device, &mut encoder);
        let _submission = queue.submit(Some(encoder.finish()));
        close(
            rotation(&read_drives(&context, &parameters)[0]),
            Vector3::new(0.9, 0.8, 0.4),
        );
        let before_cancel = read_drives(&context, &parameters);
        queue.write_buffer(
            &actions,
            0,
            bytemuck::cast_slice(&[0.1f32, 0.1, 0.2, 0.2, 4.0, 4.0, 5.0, 5.0]),
        );
        let mut encoder = device.create_command_encoder(&Default::default());
        control.encode_latch(device, &mut encoder, &actions);
        let _submission = queue.submit(Some(encoder.finish()));
        control.cancel(queue).unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        control.encode_apply(device, &mut encoder);
        let _submission = queue.submit(Some(encoder.finish()));
        let after_cancel = read_drives(&context, &parameters);
        for i in 0..4 {
            assert_eq!(after_cancel[i].target, before_cancel[i].target);
            assert_eq!(after_cancel[i].velocity, before_cancel[i].velocity);
        }

        // Exactly-pi antipodes use the same lexicographic axis.
        let mut pi = initial;
        pi[0].target = [-1.0, 0.0, 0.0, 0.0];
        pi[2].target = [1.0, 0.0, 0.0, 0.0];
        queue.write_buffer(&parameters, 0, bytemuck::cast_slice(&pi));
        let pi_entries = vec![mapping(0, 1, Position)];
        let pi_control = GpuSphericalMotorTargetControl::new(
            device,
            &parameters,
            &drives,
            &[pi_entries.clone(), pi_entries],
            &[0, 0],
        )
        .unwrap();
        queue.write_buffer(&actions, 0, bytemuck::cast_slice(&[0.2f32, 0.2]));
        let mut encoder = device.create_command_encoder(&Default::default());
        pi_control.encode_latch(device, &mut encoder, &actions);
        pi_control.encode_apply(device, &mut encoder);
        let _submission = queue.submit(Some(encoder.finish()));
        let pi_actual = read_drives(&context, &parameters);
        assert_eq!(pi_actual[0].target, pi_actual[2].target);
        let expected =
            UnitQuaternion::from_scaled_axis(Vector3::new(core::f32::consts::PI, 0.2, 0.0))
                .scaled_axis();
        close(rotation(&pi_actual[0]), expected);

        for invalid in [
            vec![],
            vec![mapping(2, 0, Position)],
            vec![mapping(0, 3, Position)],
            vec![mapping(0, 0, Position), mapping(0, 0, Position)],
        ] {
            assert!(
                GpuSphericalMotorTargetControl::new(
                    device,
                    &parameters,
                    &drives,
                    &[invalid.clone(), invalid],
                    &[0, 0]
                )
                .is_err()
            );
        }
        assert!(
            GpuSphericalMotorTargetControl::new(
                device,
                &parameters,
                &[vec![None; 2], drives[1].clone()],
                &[entries.clone(), entries.clone()],
                &[0, 0]
            )
            .is_err()
        );
        assert!(
            GpuSphericalMotorTargetControl::new(
                device,
                &parameters,
                &drives,
                &[entries.clone(), vec![]],
                &[0, 0]
            )
            .is_err()
        );
        assert!(
            GpuSphericalMotorTargetControl::new(
                device,
                &parameters,
                &drives,
                &[entries.clone(), entries],
                &[0]
            )
            .is_err()
        );
    }
    assert!(tested > 0, "no actual GPU backend available");
}
