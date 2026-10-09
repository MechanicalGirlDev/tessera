//! Device-resident selected-link normal impulses from the last contact solve.

use wgpu::util::DeviceExt;

use crate::gpu_articulated_ground_contact::GpuArticulatedGroundContactError;

#[cfg(test)]
#[path = "gpu_contact_sensor_tests.rs"]
mod tests;

/// A GPU observation of up to four selected links, shared by every environment.
///
/// Values are sums of normal impulse magnitudes in newton seconds, not resultant
/// forces. Friction and bilateral constraints are excluded. Each encoding
/// replaces the observation with the latest solved substep; it does not sum a
/// control frame's substeps. Encode after the owner's contact solve on its device.
#[derive(Debug)]
pub struct GpuContactImpulseSensor {
    device: wgpu::Device,
    pipeline: wgpu::ComputePipeline,
    bindings: wgpu::BindGroup,
    output: wgpu::Buffer,
    source_status: wgpu::Buffer,
    mass_status: wgpu::Buffer,
    links: Vec<usize>,
    environments: usize,
    count: u32,
}

impl GpuContactImpulseSensor {
    pub(crate) fn new(
        device: &wgpu::Device,
        contacts: &wgpu::Buffer,
        source_status: &wgpu::Buffer,
        mass_status: &wgpu::Buffer,
        rows: &[[u32; 4]],
        layout: [u32; 4],
        links: &[usize],
    ) -> Result<Self, GpuArticulatedGroundContactError> {
        let count =
            u32::try_from(rows.len()).map_err(|_| GpuArticulatedGroundContactError::Capacity)?;
        if count.div_ceil(64) > device.limits().max_compute_workgroups_per_dimension {
            return Err(GpuArticulatedGroundContactError::Capacity);
        }
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera contact normal impulses"),
            size: u64::from(count) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let metadata = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera contact sensor selections"),
            contents: bytemuck::cast_slice(rows),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let parameters = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera contact sensor row layout"),
            contents: bytemuck::cast_slice(&layout),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera normal impulse observation"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_contact_sensor.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tessera normal impulse observation"),
            layout: None,
            module: &shader,
            entry_point: Some("observe"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let buffers = [
            contacts,
            &metadata,
            &output,
            source_status,
            &parameters,
            mass_status,
        ];
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera contact sensor bindings"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &buffers
                .iter()
                .zip(0_u32..)
                .map(|(buffer, binding)| wgpu::BindGroupEntry {
                    binding,
                    resource: buffer.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        Ok(Self {
            device: device.clone(),
            pipeline,
            bindings,
            output,
            source_status: source_status.clone(),
            mass_status: mass_status.clone(),
            links: links.to_vec(),
            environments: rows.len() / links.len(),
            count,
        })
    }

    /// Environment-local link indices in observation order.
    pub fn selected_links(&self) -> &[usize] {
        &self.links
    }

    /// Packed `f32` normal impulses in `[environment, selected_link]` order.
    ///
    /// Initially zero. A source numerical fault produces NaN for that
    /// environment, so device consumers must not treat a failed solve as zero.
    pub fn output_buffer(&self) -> &wgpu::Buffer {
        &self.output
    }

    /// Sample the most recent contact solve, without CPU readback or submission.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Tessera contact impulse observation"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bindings, &[]);
        pass.dispatch_workgroups(self.count.div_ceil(64), 1, 1);
    }

    /// Download the most recently encoded observation from the owning queue.
    ///
    /// This does not encode an observation. It propagates source numerical faults.
    pub fn readback(
        &self,
        queue: &wgpu::Queue,
    ) -> Result<Vec<Vec<f32>>, GpuArticulatedGroundContactError> {
        let read = |buffer| {
            crate::gpu_articulated_mass::read_buffer(&self.device, queue, buffer)
                .map_err(|error| GpuArticulatedGroundContactError::Readback(error.to_string()))
        };
        for buffer in [&self.source_status, &self.mass_status] {
            let status = read(buffer)?;
            for (environment, bytes) in status.chunks_exact(4).enumerate() {
                if u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) != 0 {
                    return Err(GpuArticulatedGroundContactError::SourceFault(environment));
                }
            }
        }
        let bytes = read(&self.output)?;
        let values: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            .collect();
        if values.len() != self.environments * self.links.len()
            || values
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(GpuArticulatedGroundContactError::InvalidInput);
        }
        Ok(values
            .chunks_exact(self.links.len())
            .map(<[f32]>::to_vec)
            .collect())
    }
}
