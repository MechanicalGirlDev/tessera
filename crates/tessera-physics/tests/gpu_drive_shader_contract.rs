//! Direct actual-WGSL coverage for spherical drive projection on Vulkan and DX12.
#![cfg(feature = "gpu-contact")]

use core::time::Duration;
use std::sync::mpsc;
use tessera_physics::gpu_contact_pipeline::GpuContactDevice;
use wgpu::util::DeviceExt;

fn bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

type TestResult = Result<(), Box<dyn core::error::Error>>;

#[test]
fn spherical_shader_projects_inertia_and_caps_on_each_backend() -> TestResult {
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        let device = context.device();
        let queue = context.queue();
        tested += 1;
        eprintln!("spherical shader contract: {backend:?}");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera spherical drive contract"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../src/gpu_articulated_spherical_drive.wgsl").into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tessera spherical drive contract"),
            layout: None,
            module: &shader,
            entry_point: Some("assemble_spherical_drives"),
            compilation_options: Default::default(),
            cache: None,
        });
        let storage = |words: &[u32]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: &bytes(words),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            })
        };
        let buffers = [
            storage(&[0; 24]),
            storage(&[0, 0, 0, 1.0f32.to_bits()]),
            storage(&[0; 3]),
            storage(&[0]),
            storage(&[0]),
            storage(&[0; 3]),
            storage(&[0.5f32.to_bits(), 0.25f32.to_bits(), 2.0f32.to_bits()]),
            storage(&[0, 1]),
        ];
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &buffers
                .iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 12,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        for acceleration in [false, true] {
            for cap in [[-1.0f32; 3], [0.0; 3], [-1.0, 2.0, 0.0]] {
                for inverse_z in [0.0f32, 2.0] {
                    let mut parameters = vec![0, 0, 0, if acceleration { 2 } else { 1 }];
                    for vector in [
                        [0.1f32.sin(), 0.0, 0.0, 0.1f32.cos()],
                        [1.0, 2.0, 3.0, 0.0],
                        [10.0, 0.0, 0.0, 0.0],
                        [2.0, 2.0, 2.0, 0.0],
                        [cap[0], cap[1], cap[2], 0.0],
                    ] {
                        parameters.extend(vector.map(f32::to_bits));
                    }
                    queue.write_buffer(&buffers[0], 0, &bytes(&parameters));
                    queue.write_buffer(&buffers[5], 0, &bytes(&[0; 3]));
                    queue.write_buffer(&buffers[6], 8, &inverse_z.to_le_bytes());
                    let mut encoder = device.create_command_encoder(&Default::default());
                    {
                        let mut pass =
                            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                        pass.set_pipeline(&pipeline);
                        pass.set_bind_group(0, &group, &[]);
                        pass.dispatch_workgroups(1, 1, 1);
                    }
                    encoder.copy_buffer_to_buffer(&buffers[5], 0, &readback, 0, 12);
                    let _ = queue.submit(Some(encoder.finish()));
                    let (sender, receiver) = mpsc::channel();
                    readback
                        .slice(..)
                        .map_async(wgpu::MapMode::Read, move |result| {
                            let _delivery = sender.send(result);
                        });
                    let _ = device.poll(wgpu::PollType::Wait {
                        submission_index: None,
                        timeout: Some(Duration::from_secs(5)),
                    })?;
                    receiver.recv_timeout(Duration::from_secs(5))??;
                    let mapped = readback.slice(..).get_mapped_range();
                    for axis in 0..3 {
                        let actual = f32::from_le_bytes(mapped[axis * 4..axis * 4 + 4].try_into()?);
                        let mut expected = [4.0f32, 4.0, 6.0][axis];
                        let response = [0.5, 0.25, inverse_z][axis];
                        if acceleration {
                            expected = if response == 0.0 {
                                0.0
                            } else {
                                expected / response
                            };
                        }
                        if cap[axis] >= 0.0 {
                            expected = expected.clamp(-cap[axis], cap[axis]);
                        }
                        assert!(
                            (actual - expected).abs() < 2e-5,
                            "{backend:?}: acceleration={acceleration}, cap={cap:?}, axis={axis}: {actual} != {expected}"
                        );
                    }
                    drop(mapped);
                    readback.unmap();
                }
            }
        }
    }
    assert!(tested > 0, "no GPU backend available");
    Ok(())
}
