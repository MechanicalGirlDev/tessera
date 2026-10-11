//! Owner-coordinated snapshots and selected environment resets.

use super::{GpuArticulatedDynamicsBatch, GpuArticulatedDynamicsError, GpuJointForceInput};
use crate::gpu_articulated_joint_limit::PackedLimit;
use crate::gpu_articulated_state::reset::{
    GpuArticulatedResetSelection, GpuArticulatedResetTemplates,
};
use crate::spherical_drive::SphericalJointDrive;
use core::ops::Range;

/// Snapshot a resident storage buffer even when its owner does not expose COPY_SRC.
pub(crate) fn snapshot_buffer(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    source: &wgpu::Buffer,
) -> wgpu::Buffer {
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Tessera immutable physical reset template"),
        size: source.size(),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if source.usage().contains(wgpu::BufferUsages::COPY_SRC) {
        encoder.copy_buffer_to_buffer(source, 0, &output, 0, source.size());
    } else {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera storage snapshot"),
            source: wgpu::ShaderSource::Wgsl(
                "@group(0) @binding(0) var<storage, read> source: array<u32>;
                 @group(0) @binding(1) var<storage, read_write> destination: array<u32>;
                 @compute @workgroup_size(64)
                 fn snapshot(@builtin(global_invocation_id) id: vec3<u32>) {
                     var index = id.x;
                     while index < arrayLength(&source) {
                         destination[index] = source[index];
                         index += 64u;
                     }
                 }"
                .into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tessera storage snapshot"),
            layout: None,
            module: &shader,
            entry_point: Some("snapshot"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera storage snapshot"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: source.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bindings, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    output
}

fn copy_payloads(
    encoder: &mut wgpu::CommandEncoder,
    source: &wgpu::Buffer,
    destination: &wgpu::Buffer,
    source_range: Range<usize>,
    destination_range: Range<usize>,
    stride: u64,
    payload: Range<u64>,
) -> Result<(), GpuArticulatedDynamicsError> {
    for (source_index, destination_index) in source_range.zip(destination_range) {
        let source_index =
            u64::try_from(source_index).map_err(|_| GpuArticulatedDynamicsError::InvalidInput)?;
        let destination_index = u64::try_from(destination_index)
            .map_err(|_| GpuArticulatedDynamicsError::InvalidInput)?;
        encoder.copy_buffer_to_buffer(
            source,
            source_index * stride + payload.start,
            destination,
            destination_index * stride + payload.start,
            payload.end - payload.start,
        );
    }
    Ok(())
}

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
    joint_ranges: Vec<Range<u64>>,
    spherical_ranges: Vec<Range<u64>>,
    link_metadata: wgpu::Buffer,
    pose_joints: wgpu::Buffer,
    limits: wgpu::Buffer,
    limit_parameters: Vec<PackedLimit>,
    link_loads: wgpu::Buffer,
    gravities: wgpu::Buffer,
    vectors: wgpu::Buffer,
    prescribed: wgpu::Buffer,
    prescribed_coordinates: Vec<Vec<bool>>,
    limit_compliance: Vec<Vec<crate::prescribed_dof::ConstraintCompliance>>,
    joint_bounds: Vec<Vec<Option<(f64, f64)>>>,
    armature: Option<Vec<f32>>,
    mjcf_controls: Option<(wgpu::Buffer, wgpu::Buffer, Vec<Range<usize>>)>,
}

impl GpuArticulatedDynamicsBatch {
    pub(crate) fn state(&self) -> &crate::gpu_articulated_state::GpuGeneralizedStateBatch {
        &self.state
    }

    pub(crate) fn matches_reset_templates(
        &self,
        templates: &GpuArticulatedDynamicsResetTemplates,
    ) -> bool {
        templates.state.matches_source(&self.state)
    }

    pub(crate) fn snapshot_scene_contacts(
        &self,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Option<crate::gpu_articulated_ground_contact::scene_reset::SceneContactResetTemplates>
    {
        self.ground_contact
            .as_ref()
            .map(|contacts| contacts.snapshot_scene_contacts(encoder))
    }

    pub(crate) fn matches_scene_contacts(
        &self,
        templates: &Option<
            crate::gpu_articulated_ground_contact::scene_reset::SceneContactResetTemplates,
        >,
    ) -> bool {
        match (&self.ground_contact, templates) {
            (Some(contacts), Some(templates)) => contacts.matches_scene_contacts(templates),
            (None, None) => true,
            _ => false,
        }
    }

    pub(crate) fn encode_reset_scene_contacts(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        templates: &Option<
            crate::gpu_articulated_ground_contact::scene_reset::SceneContactResetTemplates,
        >,
        environments: &[usize],
    ) {
        if let (Some(contacts), Some(templates)) = (&mut self.ground_contact, templates) {
            contacts.encode_reset_scene_contacts(encoder, templates, environments);
        }
    }

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
            link_metadata: snapshot_buffer(
                &self.device,
                encoder,
                self.link_terms.metadata_buffer(),
            ),
            pose_joints: snapshot_buffer(&self.device, encoder, self.poses.joint_buffer()),
            limits: snapshot_buffer(&self.device, encoder, self.joint_limits.parameter_buffer()),
            limit_parameters: self.joint_limits.reset_parameters().to_vec(),
            link_loads: snapshot_buffer(&self.device, encoder, self.forces.link_load_buffer()),
            gravities: snapshot_buffer(&self.device, encoder, self.forces.gravity_buffer()),
            vectors: snapshot_buffer(&self.device, encoder, self.mass.vectors_buffer()),
            prescribed: snapshot_buffer(&self.device, encoder, self.mass.prescribed_buffer()),
            prescribed_coordinates: self.prescribed_coordinates.clone(),
            limit_compliance: self.limit_compliance.clone(),
            joint_bounds: self.joint_bounds.clone(),
            armature: self.joint_forces.reset_armature().map(<[f32]>::to_vec),
            mjcf_controls: self.mjcf_actuators.as_ref().map(|actuators| {
                (
                    snapshot_buffer(&self.device, encoder, actuators.controls_buffer()),
                    actuators.controls_buffer().clone(),
                    actuators.control_ranges().to_vec(),
                )
            }),
        })
    }

    /// Reset selected environments from resident templates without batch readback.
    ///
    /// Selections must have compatible native coordinate/quaternion layouts.
    /// The whole request is validated before recording mutation. Submitting and
    /// stepping must be serialized; derived link poses are refreshed on the GPU.
    pub fn encode_reset_envs_from_templates(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        templates: &GpuArticulatedDynamicsResetTemplates,
        selections: &[GpuArticulatedResetSelection],
    ) -> Result<(), GpuArticulatedDynamicsError> {
        if !templates.state.matches_source(&self.state) {
            return Err(GpuArticulatedDynamicsError::InvalidInput);
        }
        match (&templates.mjcf_controls, &self.mjcf_actuators) {
            (Some((_, original, ranges)), Some(actuators))
                if original == actuators.controls_buffer()
                    && ranges == actuators.control_ranges() => {}
            (None, None) => {}
            _ => return Err(GpuArticulatedDynamicsError::InvalidInput),
        }
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
        let mut armature = templates.armature.clone();
        let mut prescribed = self.prescribed_coordinates.clone();
        for selection in selections {
            let source = self
                .poses
                .reset_topology()
                .get(selection.template)
                .ok_or(GpuArticulatedDynamicsError::InvalidInput)?;
            let target = self
                .poses
                .reset_topology()
                .get(selection.environment)
                .ok_or(GpuArticulatedDynamicsError::InvalidInput)?;
            if source != target
                || self.poses.link_ranges()[selection.template].len()
                    != self.poses.link_ranges()[selection.environment].len()
            {
                return Err(GpuArticulatedDynamicsError::InvalidInput);
            }
            if let Some((_, _, ranges)) = &templates.mjcf_controls
                && ranges[selection.template].len() != ranges[selection.environment].len()
            {
                return Err(GpuArticulatedDynamicsError::InvalidInput);
            }
            prescribed[selection.environment]
                .clone_from(&templates.prescribed_coordinates[selection.template]);
            if let (Some(original), Some(restored)) = (&templates.armature, &mut armature) {
                restored[self.state.ranges()[selection.environment].clone()]
                    .copy_from_slice(&original[self.state.ranges()[selection.template].clone()]);
            }
        }
        let replacement_freeze =
            if self.sleep_freeze.is_some() && prescribed != self.prescribed_coordinates {
                Some(self.build_contact_sleep_freeze_with_selection(&prescribed)?)
            } else {
                None
            };
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
            || self
                .spherical_targets
                .as_ref()
                .is_some_and(|targets| !targets.accepts_inputs(&restored_spherical))
        {
            return Err(GpuArticulatedDynamicsError::InvalidInput);
        }
        self.state
            .encode_reset_envs_from_templates(encoder, &templates.state, selections)?;
        if let (Some(previous), Some(replacement)) = (&self.sleep_freeze, replacement_freeze) {
            encoder.copy_buffer_to_buffer(
                previous.frozen_buffer(),
                0,
                replacement.frozen_buffer(),
                0,
                previous.frozen_buffer().size(),
            );
            self.sleep_freeze = Some(replacement);
        }
        for selection in selections {
            let environment = selection.environment;
            if let (Some((controls, _, ranges)), Some(actuators)) =
                (&templates.mjcf_controls, &self.mjcf_actuators)
            {
                copy_payloads(
                    encoder,
                    controls,
                    actuators.controls_buffer(),
                    ranges[selection.template].clone(),
                    ranges[environment].clone(),
                    4,
                    0..4,
                )?;
            }
            let source = &templates.joint_ranges[selection.template];
            let destination = &templates.joint_ranges[environment];
            // The implicit force row contains an absolute destination vector index
            // at byte 68. It must never be imported from another environment.
            for coordinate in 0..self.dimensions[environment] {
                let offset = u64::try_from(coordinate)
                    .map_err(|_| GpuArticulatedDynamicsError::InvalidInput)?
                    * 80;
                for payload in [0..68, 72..80] {
                    encoder.copy_buffer_to_buffer(
                        &templates.joints,
                        source.start + offset + payload.start,
                        self.joint_forces.parameter_buffer(),
                        destination.start + offset + payload.start,
                        payload.end - payload.start,
                    );
                }
            }
            let source_coordinates = self.state.ranges()[selection.template].clone();
            let destination_coordinates = self.state.ranges()[environment].clone();
            let source_links = self.poses.link_ranges()[selection.template].clone();
            let destination_links = self.poses.link_ranges()[environment].clone();
            let link_stride = u64::try_from(super::GpuArticulatedLinkTermsBatch::metadata_stride())
                .map_err(|_| GpuArticulatedDynamicsError::InvalidInput)?;
            copy_payloads(
                encoder,
                &templates.link_metadata,
                self.link_terms.metadata_buffer(),
                source_links.clone(),
                destination_links.clone(),
                link_stride,
                16..link_stride,
            )?;
            let pose_stride = u64::try_from(super::GpuArticulatedPoseBatch::joint_stride())
                .map_err(|_| GpuArticulatedDynamicsError::InvalidInput)?;
            copy_payloads(
                encoder,
                &templates.pose_joints,
                self.poses.joint_buffer(),
                self.poses.joint_ranges()[selection.template].clone(),
                self.poses.joint_ranges()[environment].clone(),
                pose_stride,
                16..80,
            )?;
            copy_payloads(
                encoder,
                &templates.link_loads,
                self.forces.link_load_buffer(),
                source_links,
                destination_links,
                32,
                0..32,
            )?;
            copy_payloads(
                encoder,
                &templates.limits,
                self.joint_limits.parameter_buffer(),
                source_coordinates.clone(),
                destination_coordinates.clone(),
                32,
                0..32,
            )?;
            copy_payloads(
                encoder,
                &templates.prescribed,
                self.mass.prescribed_buffer(),
                source_coordinates.clone(),
                destination_coordinates.clone(),
                4,
                0..4,
            )?;
            encoder.copy_buffer_to_buffer(
                &templates.gravities,
                u64::try_from(selection.template)
                    .map_err(|_| GpuArticulatedDynamicsError::InvalidInput)?
                    * 16,
                self.forces.gravity_buffer(),
                u64::try_from(environment)
                    .map_err(|_| GpuArticulatedDynamicsError::InvalidInput)?
                    * 16,
                16,
            );
            let source_armature = source_coordinates.start * 2
                ..source_coordinates.start * 2 + source_coordinates.len();
            let destination_armature = destination_coordinates.start * 2
                ..destination_coordinates.start * 2 + destination_coordinates.len();
            copy_payloads(
                encoder,
                &templates.vectors,
                self.mass.vectors_buffer(),
                source_armature,
                destination_armature,
                4,
                0..4,
            )?;
            self.joint_limits.restore_reset_parameters(
                destination_coordinates,
                &templates.limit_parameters[source_coordinates],
            );
            self.joint_bounds[environment].clone_from(&templates.joint_bounds[selection.template]);
            self.limit_compliance[environment]
                .clone_from(&templates.limit_compliance[selection.template]);
            if let (Some(drives), Some(parameters)) = (&self.spherical_drives, &templates.spherical)
            {
                let source = &templates.spherical_ranges[selection.template];
                let destination = &templates.spherical_ranges[environment];
                let count = restored_spherical[environment].len();
                let stride = (source.end - source.start) / count.max(1) as u64;
                for joint in 0..count {
                    // Preserve the three destination indices, but restore the
                    // fourth word: disabled, force-based, or acceleration-based.
                    let payload = joint as u64 * stride + 12;
                    encoder.copy_buffer_to_buffer(
                        parameters,
                        source.start + payload,
                        drives.parameter_buffer(),
                        destination.start + payload,
                        stride - 12,
                    );
                }
            }
            if let Some(targets) = &self.motor_targets {
                targets.encode_cancel_environment(encoder, environment, self.dimensions.len());
            }
            if let Some(targets) = &self.spherical_targets {
                targets.encode_cancel_environment(encoder, environment)?;
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
        if let Some(armature) = &armature {
            let environments = selections
                .iter()
                .map(|selection| selection.environment)
                .collect::<Vec<_>>();
            self.joint_forces
                .restore_reset_armature(armature, &environments);
        }
        self.prescribed_coordinates = prescribed;
        self.poses.encode(encoder);
        Ok(())
    }
}
