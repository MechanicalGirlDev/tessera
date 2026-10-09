//! Owner-coordinated snapshots and selected environment resets.

use super::{GpuArticulatedDynamicsBatch, GpuArticulatedDynamicsError, GpuJointForceInput};
use crate::gpu_articulated_state::reset::{
    GpuArticulatedResetSelection, GpuArticulatedResetTemplates,
};
use crate::spherical_drive::SphericalJointDrive;

#[cfg(test)]
#[path = "gpu_dynamics_reset_tests.rs"]
mod tests;

/// Immutable resident state and drive templates for one dynamics owner.
///
/// Coordinates, floating roots, spherical orientations, fault flags, and scalar
/// and quaternion motor/passive parameters are captured on the GPU. Resetting
/// selected environments also clears only their pending actions, contact history,
/// and sleep diagnostics. Unselected environments retain their state and controls.
#[derive(Debug)]
pub struct GpuArticulatedDynamicsResetTemplates {
    state: GpuArticulatedResetTemplates,
    joints: wgpu::Buffer,
    spherical: Option<wgpu::Buffer>,
    joint_inputs: Vec<Vec<GpuJointForceInput>>,
    spherical_inputs: Vec<Vec<Option<SphericalJointDrive>>>,
    joint_ranges: Vec<core::ops::Range<u64>>,
    spherical_ranges: Vec<core::ops::Range<u64>>,
}

impl GpuArticulatedDynamicsBatch {
    /// Capture current state and native drives after prior commands in this encoder.
    ///
    /// The caller submits this encoder before using the immutable templates on
    /// another encoder. Templates are tied to this owner and its native layouts.
    pub fn snapshot_reset_templates(
        &self,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<GpuArticulatedDynamicsResetTemplates, GpuArticulatedDynamicsError> {
        let joint_inputs = self
            .accepted_joints
            .lock()
            .map_err(|_| GpuArticulatedDynamicsError::InvalidInput)?
            .clone();
        let spherical_inputs = self
            .accepted_spherical_drives
            .lock()
            .map_err(|_| GpuArticulatedDynamicsError::InvalidInput)?
            .clone();
        let copy = |source: &wgpu::Buffer, encoder: &mut wgpu::CommandEncoder| {
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Tessera immutable drive reset template"),
                size: source.size(),
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            encoder.copy_buffer_to_buffer(source, 0, &buffer, 0, source.size());
            buffer
        };
        let state =
            self.state
                .snapshot_reset_templates(encoder, &self.poses, self.spherical.as_ref())?;
        let joints = copy(self.joint_forces.parameter_buffer(), encoder);
        let spherical = self
            .spherical_drives
            .as_ref()
            .map(|drives| copy(drives.parameter_buffer(), encoder));
        let ranges = |counts: Vec<usize>, bytes: u64| {
            let stride = bytes / counts.iter().sum::<usize>().max(1) as u64;
            let mut offset = 0;
            counts
                .into_iter()
                .map(|count| {
                    let start = offset;
                    offset += count as u64 * stride;
                    start..offset
                })
                .collect::<Vec<_>>()
        };
        let joint_ranges = ranges(self.dimensions.clone(), joints.size());
        let spherical_ranges = spherical.as_ref().map_or_else(Vec::new, |buffer| {
            ranges(
                spherical_inputs.iter().map(Vec::len).collect(),
                buffer.size(),
            )
        });
        Ok(GpuArticulatedDynamicsResetTemplates {
            state,
            joints,
            spherical,
            joint_inputs,
            spherical_inputs,
            joint_ranges,
            spherical_ranges,
        })
    }

    /// Reset selected environments from resident templates without batch readback.
    ///
    /// Selections must have compatible native coordinate/quaternion layouts.
    /// The whole request is validated before recording mutation. Submitting and
    /// stepping must be serialized; derived link poses are refreshed on the GPU.
    pub fn encode_reset_envs_from_templates(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        templates: &GpuArticulatedDynamicsResetTemplates,
        selections: &[GpuArticulatedResetSelection],
    ) -> Result<(), GpuArticulatedDynamicsError> {
        let mut accepted = self
            .accepted_joints
            .lock()
            .map_err(|_| GpuArticulatedDynamicsError::InvalidInput)?;
        let mut spherical_inputs = self
            .accepted_spherical_drives
            .lock()
            .map_err(|_| GpuArticulatedDynamicsError::InvalidInput)?;
        let mut restored = accepted.clone();
        let mut restored_spherical = spherical_inputs.clone();
        for selection in selections {
            let source = templates
                .joint_inputs
                .get(selection.template)
                .ok_or(GpuArticulatedDynamicsError::InvalidInput)?;
            let target = restored
                .get_mut(selection.environment)
                .ok_or(GpuArticulatedDynamicsError::InvalidInput)?;
            if source.len() != target.len() {
                return Err(GpuArticulatedDynamicsError::InvalidInput);
            }
            target.clone_from(source);
            if self.spherical_drives.is_some() {
                let source = templates
                    .spherical_inputs
                    .get(selection.template)
                    .ok_or(GpuArticulatedDynamicsError::InvalidInput)?;
                restored_spherical[selection.environment].clone_from(source);
            }
        }
        if self
            .motor_targets
            .as_ref()
            .is_some_and(|targets| !targets.accepts_inputs(&restored))
        {
            return Err(GpuArticulatedDynamicsError::InvalidInput);
        }
        self.state
            .encode_reset_envs_from_templates(encoder, &templates.state, selections)?;
        for selection in selections {
            let environment = selection.environment;
            let source = &templates.joint_ranges[selection.template];
            let destination = &templates.joint_ranges[environment];
            encoder.copy_buffer_to_buffer(
                &templates.joints,
                source.start,
                self.joint_forces.parameter_buffer(),
                destination.start,
                source.end - source.start,
            );
            if let (Some(drives), Some(parameters)) = (&self.spherical_drives, &templates.spherical)
            {
                let source = &templates.spherical_ranges[selection.template];
                let destination = &templates.spherical_ranges[environment];
                let count = restored_spherical[environment].len();
                let stride = (source.end - source.start) / count.max(1) as u64;
                for joint in 0..count {
                    // Preserve destination coordinate/environment/orientation indices.
                    let payload = joint as u64 * stride + 16;
                    encoder.copy_buffer_to_buffer(
                        parameters,
                        source.start + payload,
                        drives.parameter_buffer(),
                        destination.start + payload,
                        stride - 16,
                    );
                }
            }
            if let Some(targets) = &self.motor_targets {
                targets.encode_cancel_environment(encoder, environment, self.dimensions.len());
            }
            if let Some(contact) = &self.ground_contact {
                contact.encode_clear_environment(encoder, environment);
            }
            if let Some(freeze) = &self.sleep_freeze {
                let range = &self.state.ranges()[environment];
                encoder.clear_buffer(
                    freeze.frozen_buffer(),
                    range.start as u64 * 4,
                    Some(range.len() as u64 * 4),
                );
            }
        }
        *accepted = restored;
        *spherical_inputs = restored_spherical;
        self.poses.encode(encoder);
        Ok(())
    }
}
