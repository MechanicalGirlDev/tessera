//! Selective native-layout reset. Only selection metadata is uploaded at reset.

use super::{GpuGeneralizedStateBatch, GpuGeneralizedStateError};
use crate::gpu_articulated_pose::GpuArticulatedPoseBatch;
use crate::gpu_articulated_spherical::GpuArticulatedSphericalBatch;
use wgpu::util::DeviceExt;

/// One destination environment and the snapshot environment used as its template.
#[derive(Debug, Clone, Copy)]
pub struct GpuArticulatedResetSelection {
    /// Destination environment index.
    pub environment: usize,
    /// Environment index in the immutable snapshot.
    pub template: usize,
    /// World translation added to a floating root; rejected for fixed roots.
    pub root_translation: Option<[f64; 3]>,
    /// World linear XYZ and angular XYZ velocities; rejected for fixed roots.
    pub root_velocity: Option<[f64; 6]>,
}

#[cfg(test)]
#[path = "gpu_articulated_reset_tests.rs"]
mod tests;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Request {
    destination: [u32; 4],
    source: [u32; 4],
    counts: [u32; 4],
    translation: [f32; 4],
    velocity: [f32; 8],
}

#[derive(Debug)]
struct Layout {
    start: usize,
    dimension: usize,
    spherical_start: usize,
    slots: Vec<usize>,
    floating: bool,
}

/// Immutable snapshot bound to its source resident batch.
///
/// Snapshot and reset are encoded, not submitted. The caller serializes submission
/// with stepping. Fault flags are captured and restored: only a healthy snapshot
/// clears a fault. Root poses and native XYZW spherical quaternions are preserved;
/// scalar workspace is not converted to Euler coordinates. Mapping is allowed
/// only between equal coordinate counts, root types and spherical velocity slots.
#[derive(Debug)]
pub struct GpuArticulatedResetTemplates {
    data: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    positions: wgpu::Buffer,
    velocities: wgpu::Buffer,
    roots: wgpu::Buffer,
    orientations: wgpu::Buffer,
    state_status: wgpu::Buffer,
    mass_status: wgpu::Buffer,
    layouts: Vec<Layout>,
    velocity_offset: usize,
    root_offset: usize,
    spherical_offset: usize,
    status_offset: usize,
}

impl GpuGeneralizedStateBatch {
    /// Capture every current environment on the device, without readback.
    ///
    /// Commands earlier in this encoder complete before capture. The returned
    /// templates are immutable and may reset only this source batch.
    pub fn snapshot_reset_templates(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        poses: &GpuArticulatedPoseBatch,
        spherical: Option<&GpuArticulatedSphericalBatch>,
    ) -> Result<GpuArticulatedResetTemplates, GpuGeneralizedStateError> {
        if poses.coordinate_buffer() != self.position_buffer()
            || poses.floating_roots().len() != self.ranges.len()
            || poses.uses_spherical_quaternions() != spherical.is_some()
            || spherical.is_some_and(|s| {
                !s.matches_source(self)
                    || poses.spherical_orientation_buffer() != s.orientation_buffer()
            })
        {
            return Err(GpuGeneralizedStateError::InvalidInput);
        }
        let velocity_offset = self.positions.size() as usize / 4;
        let root_offset = velocity_offset * 2;
        let spherical_offset = root_offset + self.ranges.len() * 8;
        let spherical_size = spherical.map_or(0, |s| s.orientation_buffer().size() as usize / 4);
        let status_offset = spherical_offset + spherical_size;
        let size = (status_offset + self.ranges.len() * 2) as u64 * 4;
        let limits = self.device.limits();
        if size > limits.max_buffer_size
            || size > u64::from(limits.max_storage_buffer_binding_size)
            || limits.max_storage_buffers_per_shader_stage < 8
        {
            return Err(GpuGeneralizedStateError::Capacity);
        }
        let mut spherical_start = 0;
        let layouts = self
            .ranges
            .iter()
            .enumerate()
            .map(|(env, range)| {
                let slots = spherical.map_or_else(Vec::new, |s| s.velocity_slots()[env].clone());
                let layout = Layout {
                    start: range.start,
                    dimension: range.len(),
                    spherical_start,
                    floating: poses.floating_roots()[env],
                    slots,
                };
                spherical_start += layout.slots.len();
                layout
            })
            .collect();
        let data = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera immutable articulated reset templates"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let orientations = spherical.map_or_else(
            || {
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Tessera unused reset quaternion"),
                    size: 16,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                })
            },
            |s| s.orientation_buffer().clone(),
        );
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Tessera selective articulated reset"),
                source: wgpu::ShaderSource::Wgsl(include_str!("gpu_articulated_reset.wgsl").into()),
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Tessera selective articulated reset"),
                layout: None,
                module: &shader,
                entry_point: Some("reset"),
                compilation_options: Default::default(),
                cache: None,
            });
        for (buffer, offset) in [
            (&self.positions, 0),
            (&self.velocities, velocity_offset),
            (poses.root_pose_buffer(), root_offset),
            (&self.state_status, status_offset),
            (&self.mass_status, status_offset + self.ranges.len()),
        ] {
            encoder.copy_buffer_to_buffer(buffer, 0, &data, offset as u64 * 4, buffer.size());
        }
        if spherical.is_some() {
            encoder.copy_buffer_to_buffer(
                &orientations,
                0,
                &data,
                spherical_offset as u64 * 4,
                orientations.size(),
            );
        }
        Ok(GpuArticulatedResetTemplates {
            data,
            pipeline,
            positions: self.positions.clone(),
            velocities: self.velocities.clone(),
            roots: poses.root_pose_buffer().clone(),
            orientations,
            state_status: self.state_status.clone(),
            mass_status: self.mass_status.clone(),
            layouts,
            velocity_offset,
            root_offset,
            spherical_offset,
            status_offset,
        })
    }
}

#[path = "gpu_articulated_reset_encode.rs"]
mod encode;
