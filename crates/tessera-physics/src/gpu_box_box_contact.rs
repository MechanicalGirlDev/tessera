//! GPU oriented-box pair overlap and minimum-separation-axis contacts.

use core::mem::{size_of, size_of_val};
use core::time::Duration;
use std::sync::mpsc;

use wgpu::util::DeviceExt;

use crate::gpu_contact_pipeline::ContactPipelineError;
use crate::gpu_sphere_box_contact::GpuBox;
use crate::gpu_sphere_contact::GpuSphereContact;

/// Reusable GPU box-box SAT narrow phase.
#[derive(Debug)]
pub struct GpuBoxBoxContacts {
    pipeline: wgpu::ComputePipeline,
}

impl GpuBoxBoxContacts {
    /// Compile the 15-axis SAT kernel.
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera box box contacts"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_box_box_contact.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tessera box box contact pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { pipeline }
    }

    /// Detect every unordered box pair; result index is `a * box_count + b`.
    pub fn detect(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        boxes: &[GpuBox],
    ) -> Result<Vec<GpuSphereContact>, ContactPipelineError> {
        if boxes.is_empty() {
            return Ok(Vec::new());
        }
        let count = u32::try_from(boxes.len()).map_err(|_| ContactPipelineError::Capacity)?;
        let pair_count = count
            .checked_mul(count)
            .ok_or(ContactPipelineError::Capacity)?;
        let output_bytes = u64::from(pair_count) * size_of::<GpuSphereContact>() as u64;
        let limits = device.limits();
        if boxes.iter().any(|shape| !shape.valid()) {
            return Err(ContactPipelineError::InvalidInput);
        }
        if pair_count.div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || size_of_val(boxes) as u64 > u64::from(limits.max_storage_buffer_binding_size)
            || output_bytes > u64::from(limits.max_storage_buffer_binding_size)
            || output_bytes > limits.max_buffer_size
        {
            return Err(ContactPipelineError::Capacity);
        }
        let boxes_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera SAT boxes"),
            contents: bytemuck::cast_slice(boxes),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera SAT contacts"),
            size: output_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera SAT readback"),
            size: output_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera SAT bindings"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: boxes_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera SAT encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Tessera SAT pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(pair_count.div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, output_bytes);
        let _submission = queue.submit(Some(encoder.finish()));
        let (sender, receiver) = mpsc::channel();
        staging
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
        let view = staging.slice(..).get_mapped_range();
        let result = view
            .chunks_exact(size_of::<GpuSphereContact>())
            .map(bytemuck::pod_read_unaligned)
            .collect();
        drop(view);
        staging.unmap();
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_contact_pipeline::GpuContactDevice;

    #[test]
    fn gpu_sat_reports_overlap_separation_and_rotation() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let boxes = [
            GpuBox::new([0.0; 3], identity, [0.5; 3]).unwrap(),
            GpuBox::new([0.8, 0.0, 0.0], identity, [0.5; 3]).unwrap(),
            GpuBox::new([3.0, 0.0, 0.0], identity, [0.5; 3]).unwrap(),
            GpuBox::new(
                [0.0, 0.8, 0.0],
                [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                [0.5; 3],
            )
            .unwrap(),
        ];
        let result = context
            .box_box()
            .detect(context.device(), context.queue(), &boxes)
            .unwrap();
        assert!(result[1].is_contact());
        assert!((result[1].depth_hit[0] - 0.2).abs() < 1e-5);
        assert!(result[1].normal[0] > 0.99);
        assert!(!result[2].is_contact());
        assert!(result[3].is_contact());
        assert!((result[3].depth_hit[0] - 0.2).abs() < 1e-5);
    }
}
