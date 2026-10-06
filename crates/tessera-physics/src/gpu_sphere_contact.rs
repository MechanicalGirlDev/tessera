//! GPU sphere-sphere narrow phase for validating the first contact path.

use core::mem::{size_of, size_of_val};
use core::time::Duration;
use std::sync::mpsc;

use wgpu::util::DeviceExt;

use crate::gpu_broad_phase::GpuPair;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
/// Sphere center in XYZ and radius in W, in metres.
pub struct GpuSphere {
    /// GPU packed center and radius.
    pub center_radius: [f32; 4],
}

impl GpuSphere {
    /// Validate and pack a sphere for the narrow-phase shader.
    pub fn new(center: [f32; 3], radius: f32) -> Result<Self, SphereContactError> {
        if center.iter().any(|x| !x.is_finite()) || !radius.is_finite() || radius <= 0.0 {
            return Err(SphereContactError::InvalidInput);
        }
        Ok(Self {
            center_radius: [center[0], center[1], center[2], radius],
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
/// Contact result for one candidate sphere pair.
pub struct GpuSphereContact {
    /// Midpoint between surface witnesses; W holds edge A's index for Z = 5,
    /// or the selected face normal's index for Z = 6 or 7.
    pub point: [f32; 4],
    /// Unit normal from shape A to B; W holds edge B's index for Z = 5.
    pub normal: [f32; 4],
    /// X is penetration depth, Y is 1 for a contact and 0 otherwise. The
    /// convex kernel sets Z to 1 for a mesh-prism sphere-face contact and 2
    /// for an edge or corner contact. Z is 3 for a rounded segment pair, 4
    /// for an analytic cylinder or cone against a sphere, and 5 for a convex
    /// edge-axis witness (edge indices are in the W components above). Z is 6
    /// for a face axis from A and 7 for a face axis from B.
    pub depth_hit: [f32; 4],
}

impl GpuSphereContact {
    /// Whether the input spheres overlap or touch.
    pub fn is_contact(&self) -> bool {
        self.depth_hit[1] != 0.0
    }
}

/// Invalid sphere inputs or GPU execution failure.
#[derive(Debug, thiserror::Error)]
pub enum SphereContactError {
    /// An invalid sphere or pair index was supplied.
    #[error("invalid sphere or pair index")]
    InvalidInput,
    /// The input exceeds the device's buffer or dispatch capacity.
    #[error("sphere contact input exceeds GPU limits")]
    Capacity,
    /// The diagnostic GPU readback did not complete.
    #[error("sphere contact GPU readback failed: {0}")]
    Readback(String),
}

/// Reusable sphere contact compute pipeline.
#[derive(Debug)]
pub struct GpuSphereContacts {
    pipeline: wgpu::ComputePipeline,
}

impl GpuSphereContacts {
    /// Compile the WGSL narrow-phase kernel.
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sphere contact narrow phase"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_sphere_contact.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sphere contact pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { pipeline }
    }

    /// Generate contacts for candidate pairs. Readback is for reference tests;
    /// the integrated solver will consume the output buffer on the GPU.
    pub fn contacts(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        spheres: &[GpuSphere],
        pairs: &[GpuPair],
    ) -> Result<Vec<GpuSphereContact>, SphereContactError> {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }
        if spheres.is_empty()
            || spheres.iter().any(|sphere| {
                sphere.center_radius.iter().any(|x| !x.is_finite())
                    || sphere.center_radius[3] <= 0.0
            })
            || pairs.iter().any(|pair| {
                pair.a == pair.b
                    || pair.a as usize >= spheres.len()
                    || pair.b as usize >= spheres.len()
            })
        {
            return Err(SphereContactError::InvalidInput);
        }
        let count = u32::try_from(pairs.len()).map_err(|_| SphereContactError::Capacity)?;
        let output_size = u64::from(count) * size_of::<GpuSphereContact>() as u64;
        let limits = device.limits();
        if count.div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || [
                size_of_val(spheres) as u64,
                size_of_val(pairs) as u64,
                output_size,
            ]
            .iter()
            .any(|size| *size > u64::from(limits.max_storage_buffer_binding_size))
        {
            return Err(SphereContactError::Capacity);
        }
        let spheres_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("contact spheres"),
            contents: bytemuck::cast_slice(spheres),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let pairs_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sphere contact pairs"),
            contents: bytemuck::cast_slice(pairs),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sphere contacts"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sphere contact readback"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = self.pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sphere contact inputs"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: spheres_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pairs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sphere contact encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sphere contact pass"),
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
            .map_err(|error| SphereContactError::Readback(error.to_string()))?;
        receiver
            .recv_timeout(Duration::from_secs(5))
            .map_err(|error| SphereContactError::Readback(error.to_string()))?
            .map_err(|error| SphereContactError::Readback(error.to_string()))?;
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

    #[tokio::test]
    async fn gpu_spheres_report_overlap_tangency_and_separation() {
        let instance = wgpu::Instance::default();
        let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
        else {
            return;
        };
        let info = adapter.get_info();
        eprintln!(
            "GPU sphere contact test adapter: {} ({:?}, {:?})",
            info.name, info.backend, info.device_type
        );
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let spheres = [
            GpuSphere::new([0.0, 0.0, 0.0], 1.0).unwrap(),
            GpuSphere::new([1.5, 0.0, 0.0], 1.0).unwrap(),
            GpuSphere::new([2.0, 0.0, 0.0], 1.0).unwrap(),
            GpuSphere::new([4.0, 0.0, 0.0], 1.0).unwrap(),
            GpuSphere::new([0.0, 0.0, 0.0], 0.5).unwrap(),
        ];
        let pairs = [
            GpuPair { a: 0, b: 1 },
            GpuPair { a: 0, b: 2 },
            GpuPair { a: 0, b: 3 },
            GpuPair { a: 0, b: 4 },
        ];
        let contacts = GpuSphereContacts::new(&device)
            .contacts(&device, &queue, &spheres, &pairs)
            .unwrap();
        assert_eq!(contacts.len(), 4);
        assert!(contacts[0].is_contact());
        assert!((contacts[0].depth_hit[0] - 0.5).abs() < 1e-6);
        assert!((contacts[0].point[0] - 0.75).abs() < 1e-6);
        assert!((contacts[0].normal[0] - 1.0).abs() < 1e-6);
        assert!(contacts[1].is_contact());
        assert_eq!(contacts[1].depth_hit[0], 0.0);
        assert!(!contacts[2].is_contact());
        assert!(contacts[3].is_contact());
        assert!((contacts[3].depth_hit[0] - 1.5).abs() < 1e-6);
        assert_eq!(contacts[3].normal[0], 1.0);
    }
}
