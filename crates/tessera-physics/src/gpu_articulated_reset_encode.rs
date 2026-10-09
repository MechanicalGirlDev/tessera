//! Request validation and encoding for selected native-layout resets.

use super::*;
use crate::gpu_articulated_state::finite_f32;

impl GpuGeneralizedStateBatch {
    /// Encode selected resets; validate the entire request before recording any mutation.
    ///
    /// Unselected coordinates, poses, quaternions and fault flags are untouched.
    /// Controls, delayed actions, contact caches and sleep state belong to the
    /// dynamics owner and must be coordinated there. Encode forward kinematics
    /// after reset to refresh derived link poses before their next use.
    pub fn encode_reset_envs_from_templates(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        templates: &GpuArticulatedResetTemplates,
        selections: &[GpuArticulatedResetSelection],
    ) -> Result<(), GpuGeneralizedStateError> {
        if templates.positions != self.positions {
            return Err(GpuGeneralizedStateError::InvalidInput);
        }
        let mut selected = vec![false; self.ranges.len()];
        let mut requests = Vec::with_capacity(selections.len());
        for selection in selections {
            let destination = templates
                .layouts
                .get(selection.environment)
                .ok_or(GpuGeneralizedStateError::InvalidInput)?;
            let source = templates
                .layouts
                .get(selection.template)
                .ok_or(GpuGeneralizedStateError::InvalidInput)?;
            if selected[selection.environment]
                || destination.dimension != source.dimension
                || destination.floating != source.floating
                || destination.slots != source.slots
                || (!destination.floating
                    && (selection.root_translation.is_some() || selection.root_velocity.is_some()))
            {
                return Err(GpuGeneralizedStateError::InvalidInput);
            }
            selected[selection.environment] = true;
            let mut translation = [0.0; 4];
            if let Some(values) = selection.root_translation {
                for (out, value) in translation.iter_mut().zip(values) {
                    *out = finite_f32(value)?;
                }
            }
            let mut velocity = [0.0; 8];
            if let Some(values) = selection.root_velocity {
                for (out, value) in velocity.iter_mut().zip(values) {
                    *out = finite_f32(value)?;
                }
            }
            let indices = [
                destination.start,
                selection.environment,
                destination.spherical_start,
                destination.dimension,
                source.start,
                templates.velocity_offset + source.start,
                templates.root_offset + selection.template * 8,
                templates.spherical_offset + source.spherical_start * 4,
                source.slots.len(),
                usize::from(selection.root_velocity.is_some()),
                templates.status_offset + selection.template,
                templates.status_offset + templates.layouts.len() + selection.template,
            ]
            .map(u32::try_from);
            let mut packed = [0; 12];
            for (out, index) in packed.iter_mut().zip(indices) {
                *out = index.map_err(|_| GpuGeneralizedStateError::Capacity)?;
            }
            requests.push(Request {
                destination: [packed[0], packed[1], packed[2], packed[3]],
                source: [packed[4], packed[5], packed[6], packed[7]],
                counts: [packed[8], packed[9], packed[10], packed[11]],
                translation,
                velocity,
            });
        }
        if requests.is_empty() {
            return Ok(());
        }
        let limits = self.device.limits();
        let bytes = requests.len() as u64 * size_of::<Request>() as u64;
        if bytes > limits.max_buffer_size
            || bytes > u64::from(limits.max_storage_buffer_binding_size)
            || requests.len().div_ceil(64) > limits.max_compute_workgroups_per_dimension as usize
        {
            return Err(GpuGeneralizedStateError::Capacity);
        }
        let requests = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tessera selected reset metadata"),
                contents: bytemuck::cast_slice(&requests),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let buffers = [
            &templates.data,
            &requests,
            &templates.positions,
            &templates.velocities,
            &templates.roots,
            &templates.orientations,
            &templates.state_status,
            &templates.mass_status,
        ];
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera selected reset bindings"),
            layout: &templates.pipeline.get_bind_group_layout(0),
            entries: &buffers
                .iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Tessera selected reset"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&templates.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(selections.len().div_ceil(64) as u32, 1, 1);
        Ok(())
    }
}
