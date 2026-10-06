//! Vendor-neutral GPU AABB pair filtering prototype.
//!
//! This is a broad-phase building block, not a selectable physics engine. The
//! final pipeline will keep its overlap flags on the GPU for narrow phase.

use core::mem::size_of_val;
use core::time::Duration;
use std::sync::mpsc;

use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
/// An AABB with 16-byte aligned GPU vector fields.
pub struct GpuAabb {
    /// Minimum XYZ; W is padding.
    pub lower: [f32; 4],
    /// Maximum XYZ; W is padding.
    pub upper: [f32; 4],
}

impl GpuAabb {
    /// Validate finite, ordered bounds and pack them for WGSL.
    pub fn new(lower: [f32; 3], upper: [f32; 3]) -> Result<Self, BroadPhaseError> {
        if (0..3).any(|axis| {
            !lower[axis].is_finite() || !upper[axis].is_finite() || lower[axis] > upper[axis]
        }) {
            return Err(BroadPhaseError::InvalidInput);
        }
        Ok(Self {
            lower: [lower[0], lower[1], lower[2], 0.0],
            upper: [upper[0], upper[1], upper[2], 0.0],
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
/// Indices of two collision shapes to compare.
pub struct GpuPair {
    /// First shape index.
    pub a: u32,
    /// Second shape index.
    pub b: u32,
}

/// Invalid broad-phase input or GPU readback failure.
#[derive(Debug, thiserror::Error)]
pub enum BroadPhaseError {
    /// Invalid bounds or a pair refers to an absent shape.
    #[error("invalid AABB or pair index")]
    InvalidInput,
    /// The requested inputs exceed device limits.
    #[error("pair count exceeds the GPU buffer or dispatch limit")]
    Capacity,
    /// The GPU did not complete the diagnostic result transfer.
    #[error("GPU readback failed: {0}")]
    Readback(String),
}

/// Reusable compute pipeline for AABB overlap flags.
#[derive(Debug)]
pub struct GpuBroadPhase {
    pipeline: wgpu::ComputePipeline,
}

impl GpuBroadPhase {
    /// Compile the WGSL overlap kernel on the selected device.
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("contact AABB overlap"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_broad_phase.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("contact AABB broad phase"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { pipeline }
    }

    /// Return one overlap flag per input pair. This readback is for validation;
    /// the integrated physics pipeline must consume flags on the GPU instead.
    pub fn overlaps(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bounds: &[GpuAabb],
        pairs: &[GpuPair],
    ) -> Result<Vec<bool>, BroadPhaseError> {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }
        if bounds.is_empty()
            || pairs.iter().any(|pair| {
                pair.a == pair.b
                    || pair.a as usize >= bounds.len()
                    || pair.b as usize >= bounds.len()
            })
            || bounds.iter().any(|bound| {
                (0..3).any(|axis| {
                    !bound.lower[axis].is_finite()
                        || !bound.upper[axis].is_finite()
                        || bound.lower[axis] > bound.upper[axis]
                })
            })
        {
            return Err(BroadPhaseError::InvalidInput);
        }
        let count = u32::try_from(pairs.len()).map_err(|_| BroadPhaseError::Capacity)?;
        let dispatch = count.div_ceil(64);
        let limits = device.limits();
        let sizes = [
            size_of_val(bounds) as u64,
            size_of_val(pairs) as u64,
            u64::from(count) * 4,
        ];
        if dispatch > limits.max_compute_workgroups_per_dimension
            || sizes
                .iter()
                .any(|size| *size > u64::from(limits.max_storage_buffer_binding_size))
        {
            return Err(BroadPhaseError::Capacity);
        }

        let bounds_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("contact AABBs"),
            contents: bytemuck::cast_slice(bounds),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let pairs_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("contact candidate pairs"),
            contents: bytemuck::cast_slice(pairs),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let flags_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("contact overlap flags"),
            size: sizes[2],
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("contact overlap readback"),
            size: sizes[2],
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = self.pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("contact broad phase inputs"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bounds_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pairs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: flags_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("contact broad phase encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("contact broad phase pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&flags_buffer, 0, &readback, 0, sizes[2]);
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
            .map_err(|error| BroadPhaseError::Readback(error.to_string()))?;
        receiver
            .recv_timeout(Duration::from_secs(5))
            .map_err(|error| BroadPhaseError::Readback(error.to_string()))?
            .map_err(|error| BroadPhaseError::Readback(error.to_string()))?;
        let view = readback.slice(..).get_mapped_range();
        let flags = view
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) != 0)
            .collect();
        drop(view);
        readback.unmap();
        Ok(flags)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn gpu_pair_flags_match_cpu_aabb_overlap() {
        let instance = wgpu::Instance::default();
        let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
        else {
            return;
        };
        let info = adapter.get_info();
        eprintln!(
            "GPU broad-phase test adapter: {} ({:?}, {:?})",
            info.name, info.backend, info.device_type
        );
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let bounds = [
            GpuAabb::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]).unwrap(),
            GpuAabb::new([1.0, 0.5, 0.5], [2.0, 1.5, 1.5]).unwrap(),
            GpuAabb::new([2.1, 0.0, 0.0], [3.0, 1.0, 1.0]).unwrap(),
        ];
        let pair_pattern = [
            GpuPair { a: 0, b: 1 },
            GpuPair { a: 0, b: 2 },
            GpuPair { a: 1, b: 2 },
        ];
        let pairs: Vec<_> = (0..130).map(|i| pair_pattern[i % 3]).collect();
        let gpu = GpuBroadPhase::new(&device)
            .overlaps(&device, &queue, &bounds, &pairs)
            .unwrap();
        let cpu: Vec<bool> = pairs
            .iter()
            .map(|pair| {
                let a = &bounds[pair.a as usize];
                let b = &bounds[pair.b as usize];
                (0..3).all(|axis| a.lower[axis] <= b.upper[axis] && b.lower[axis] <= a.upper[axis])
            })
            .collect();
        assert_eq!(gpu, cpu);
        assert_eq!(&gpu[..3], &[true, false, false]);
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn dx12_backend_runs_same_overlap_kernel() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::DX12,
            ..Default::default()
        });
        let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
        else {
            eprintln!("DX12 adapter unavailable; skipping DX12 backend test");
            return;
        };
        let info = adapter.get_info();
        eprintln!(
            "DX12 broad-phase test adapter: {} ({:?})",
            info.name, info.device_type
        );
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let bounds = [
            GpuAabb::new([0.0; 3], [1.0; 3]).unwrap(),
            GpuAabb::new([0.5; 3], [1.5; 3]).unwrap(),
        ];
        let pairs = [GpuPair { a: 0, b: 1 }];
        assert_eq!(
            GpuBroadPhase::new(&device)
                .overlaps(&device, &queue, &bounds, &pairs)
                .unwrap(),
            [true]
        );
    }
}
