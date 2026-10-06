//! GPU sphere against oriented box contact detection.

use core::mem::{size_of, size_of_val};
use core::time::Duration;
use std::sync::mpsc;

use wgpu::util::DeviceExt;

use crate::gpu_contact_pipeline::ContactPipelineError;
use crate::gpu_sphere_contact::{GpuSphere, GpuSphereContact};

/// World-space oriented box, with orthonormal axes and positive half extents.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuBox {
    /// World center.
    pub center: [f32; 4],
    /// First world-space box axis.
    pub axis_x: [f32; 4],
    /// Second world-space box axis.
    pub axis_y: [f32; 4],
    /// Third world-space box axis.
    pub axis_z: [f32; 4],
    /// Positive half extents in axis order.
    pub half_extents: [f32; 4],
}

impl GpuBox {
    /// Validate and pack one oriented box.
    pub fn new(
        center: [f32; 3],
        axes: [[f32; 3]; 3],
        half_extents: [f32; 3],
    ) -> Result<Self, ContactPipelineError> {
        if center.iter().any(|value| !value.is_finite())
            || half_extents
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
            || axes
                .iter()
                .flat_map(|axis| axis.iter())
                .any(|value| !value.is_finite())
        {
            return Err(ContactPipelineError::InvalidInput);
        }
        for a in 0..3 {
            let length = axes[a].iter().map(|value| value * value).sum::<f32>();
            if (length - 1.0).abs() > 1e-3 {
                return Err(ContactPipelineError::InvalidInput);
            }
            for b in 0..a {
                let dot = (0..3).map(|i| axes[a][i] * axes[b][i]).sum::<f32>();
                if dot.abs() > 1e-3 {
                    return Err(ContactPipelineError::InvalidInput);
                }
            }
        }
        for row in 0..3 {
            let reach = (0..3)
                .map(|axis| axes[axis][row].abs() * half_extents[axis])
                .sum::<f32>();
            if !(center[row] - reach).is_finite() || !(center[row] + reach).is_finite() {
                return Err(ContactPipelineError::InvalidInput);
            }
        }
        let pack = |xyz: [f32; 3]| [xyz[0], xyz[1], xyz[2], 0.0];
        Ok(Self {
            center: pack(center),
            axis_x: pack(axes[0]),
            axis_y: pack(axes[1]),
            axis_z: pack(axes[2]),
            half_extents: pack(half_extents),
        })
    }

    pub(crate) fn valid(&self) -> bool {
        Self::new(
            [self.center[0], self.center[1], self.center[2]],
            [
                [self.axis_x[0], self.axis_x[1], self.axis_x[2]],
                [self.axis_y[0], self.axis_y[1], self.axis_y[2]],
                [self.axis_z[0], self.axis_z[1], self.axis_z[2]],
            ],
            [
                self.half_extents[0],
                self.half_extents[1],
                self.half_extents[2],
            ],
        )
        .is_ok()
    }
}

/// Reusable GPU sphere-box narrow phase with one output per input pair.
#[derive(Debug)]
pub struct GpuSphereBoxContacts {
    pipeline: wgpu::ComputePipeline,
}

impl GpuSphereBoxContacts {
    /// Compile the sphere-box contact kernel.
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera sphere box contacts"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_sphere_box_contact.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tessera sphere box contact pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { pipeline }
    }

    /// Detect all sphere-box pairs, in sphere-major order, then read contacts.
    pub fn detect(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        spheres: &[GpuSphere],
        boxes: &[GpuBox],
    ) -> Result<Vec<GpuSphereContact>, ContactPipelineError> {
        if spheres.is_empty() || boxes.is_empty() {
            return Ok(Vec::new());
        }
        let pair_count = spheres
            .len()
            .checked_mul(boxes.len())
            .ok_or(ContactPipelineError::Capacity)?;
        let pair_count = u32::try_from(pair_count).map_err(|_| ContactPipelineError::Capacity)?;
        let output_bytes = u64::from(pair_count) * size_of::<GpuSphereContact>() as u64;
        let limits = device.limits();
        if spheres.iter().any(|sphere| {
            let [x, y, z, radius] = sphere.center_radius;
            radius <= 0.0
                || !radius.is_finite()
                || [x, y, z].iter().any(|center| {
                    !center.is_finite()
                        || !(*center - radius).is_finite()
                        || !(*center + radius).is_finite()
                })
        }) {
            return Err(ContactPipelineError::InvalidInput);
        }
        if boxes.iter().any(|shape| !shape.valid()) {
            return Err(ContactPipelineError::InvalidInput);
        }
        if pair_count.div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || size_of_val(spheres) as u64 > u64::from(limits.max_storage_buffer_binding_size)
            || size_of_val(boxes) as u64 > u64::from(limits.max_storage_buffer_binding_size)
            || output_bytes > u64::from(limits.max_storage_buffer_binding_size)
            || output_bytes > limits.max_buffer_size
        {
            return Err(ContactPipelineError::Capacity);
        }
        let spheres_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera sphere box spheres"),
            contents: bytemuck::cast_slice(spheres),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let boxes_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera sphere box boxes"),
            contents: bytemuck::cast_slice(boxes),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera sphere box contacts"),
            size: output_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera sphere box readback"),
            size: output_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera sphere box bindings"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: spheres_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: boxes_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera sphere box encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Tessera sphere box pass"),
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
    fn gpu_sphere_box_reports_outside_inside_and_rotation() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let boxes = [
            GpuBox::new(
                [0.0; 3],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [1.0; 3],
            )
            .unwrap(),
            GpuBox::new(
                [0.0; 3],
                [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                [1.0, 0.2, 0.2],
            )
            .unwrap(),
        ];
        let spheres = [
            GpuSphere::new([1.4, 0.0, 0.0], 0.5).unwrap(),
            GpuSphere::new([2.0, 0.0, 0.0], 0.5).unwrap(),
            GpuSphere::new([0.0; 3], 0.5).unwrap(),
            GpuSphere::new([0.0, 1.4, 0.0], 0.5).unwrap(),
        ];
        let contacts = context
            .sphere_box()
            .detect(context.device(), context.queue(), &spheres, &boxes)
            .unwrap();
        assert_eq!(contacts.len(), spheres.len() * boxes.len());
        assert!(contacts[0].is_contact());
        assert!((contacts[0].depth_hit[0] - 0.1).abs() < 1e-5);
        assert!((contacts[0].normal[0] - 1.0).abs() < 1e-5);
        assert!(!contacts[2].is_contact());
        assert!(contacts[4].is_contact());
        assert!((contacts[4].depth_hit[0] - 1.5).abs() < 1e-5);
        assert!(contacts[7].is_contact());
        assert!((contacts[7].depth_hit[0] - 0.1).abs() < 1e-5);
        assert!((contacts[7].normal[1] - 1.0).abs() < 1e-5);
    }
}
