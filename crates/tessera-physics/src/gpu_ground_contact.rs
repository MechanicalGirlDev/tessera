//! GPU sphere-ground collision detection for the reference world.

use core::mem::{size_of, size_of_val};
use core::time::Duration;
use std::sync::mpsc;

use wgpu::util::DeviceExt;

use crate::gpu_contact_pipeline::ContactPipelineError;
use crate::gpu_sphere_contact::{GpuSphere, GpuSphereContact};

/// Reusable compute pipeline for a finite, horizontal ground plane.
#[derive(Debug)]
pub struct GpuGroundContacts {
    pipeline: wgpu::ComputePipeline,
}

impl GpuGroundContacts {
    /// Compile the ground contact shader.
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera sphere ground contact"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_ground_contact.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tessera ground contact pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { pipeline }
    }

    /// Detect one ground contact per sphere or zero-radius point.
    pub fn detect(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        spheres: &[GpuSphere],
        ground_half_extent: f32,
    ) -> Result<Vec<GpuSphereContact>, ContactPipelineError> {
        if !ground_half_extent.is_finite()
            || ground_half_extent <= 0.0
            || spheres.iter().any(|sphere| {
                sphere.center_radius.iter().any(|x| !x.is_finite()) || sphere.center_radius[3] < 0.0
            })
        {
            return Err(ContactPipelineError::InvalidInput);
        }
        if spheres.is_empty() {
            return Ok(Vec::new());
        }
        let count = u32::try_from(spheres.len()).map_err(|_| ContactPipelineError::Capacity)?;
        let output_size = u64::from(count) * size_of::<GpuSphereContact>() as u64;
        let limits = device.limits();
        if count.div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || size_of_val(spheres) as u64 > u64::from(limits.max_storage_buffer_binding_size)
            || output_size > u64::from(limits.max_storage_buffer_binding_size)
            || output_size > limits.max_buffer_size
        {
            return Err(ContactPipelineError::Capacity);
        }
        let spheres_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera ground spheres"),
            contents: bytemuck::cast_slice(spheres),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let ground_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera ground half extent"),
            contents: bytemuck::cast_slice(&[[ground_half_extent, 0.0, 0.0, 0.0]]),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera ground contacts"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera ground contact readback"),
            size: output_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera ground contact inputs"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: spheres_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: ground_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera ground contact encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Tessera ground contact pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output_size);
        let _submission = queue.submit(Some(encoder.finish()));
        let (sender, receiver) = mpsc::channel();
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        let _status = device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(5)),
            })
            .map_err(|error| ContactPipelineError::Readback(error.to_string()))?;
        receiver
            .recv_timeout(Duration::from_secs(5))
            .map_err(|error| ContactPipelineError::Readback(error.to_string()))?
            .map_err(|error| ContactPipelineError::Readback(error.to_string()))?;
        let view = readback.slice(..).get_mapped_range();
        let contacts = view
            .chunks_exact(size_of::<GpuSphereContact>())
            .map(bytemuck::pod_read_unaligned)
            .collect();
        drop(view);
        readback.unmap();
        Ok(contacts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn check_ground(instance: wgpu::Instance) {
        let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
        else {
            eprintln!("ground contact adapter unavailable; skipping backend test");
            return;
        };
        let info = adapter.get_info();
        eprintln!("Tessera ground adapter: {} ({:?})", info.name, info.backend);
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let spheres = [
            GpuSphere::new([0.0, 0.0, 0.4], 0.5).unwrap(),
            GpuSphere::new([0.0, 0.0, 0.5], 0.5).unwrap(),
            GpuSphere::new([0.0, 0.0, 0.6], 0.5).unwrap(),
            GpuSphere::new([3.0, 0.0, 0.4], 0.5).unwrap(),
        ];
        let contacts = GpuGroundContacts::new(&device)
            .detect(&device, &queue, &spheres, 1.0)
            .unwrap();
        assert_eq!(contacts.len(), 4);
        assert!(contacts[0].is_contact());
        assert!((contacts[0].depth_hit[0] - 0.1).abs() < 1e-6);
        assert!(contacts[1].is_contact());
        assert_eq!(contacts[1].depth_hit[0], 0.0);
        assert!(!contacts[2].is_contact());
        assert!(!contacts[3].is_contact());
    }

    #[tokio::test]
    async fn gpu_ground_contact_matches_expected_depth() {
        check_ground(wgpu::Instance::default()).await;
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn dx12_ground_contact_matches_expected_depth() {
        check_ground(wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::DX12,
            ..Default::default()
        }))
        .await;
    }
}
