//! Device-resident velocity bias for articulated trees.

use core::mem::size_of;

use nalgebra::{DMatrix, DVector, Isometry3, UnitQuaternion};
use wgpu::util::DeviceExt;

use crate::articulation::Articulation;
use crate::gpu_articulated_force::GpuArticulatedForceBatch;
use crate::gpu_articulated_link_terms::{
    GpuArticulatedLinkTermsBatch, GpuArticulatedLinkTermsError,
};
use crate::gpu_articulated_mass_assembly::{
    GpuArticulatedMassAssemblyBatch, GpuMassAssemblySystem, GpuMassLink,
};
use crate::gpu_articulated_pose::{GpuArticulatedPoseBatch, GpuArticulatedPoseError};
use crate::gpu_articulated_spherical::{
    GpuArticulatedSphericalBatch, GpuArticulatedSphericalError, GpuSphericalJointState,
};
use crate::gpu_articulated_state::{
    GpuGeneralizedState, GpuGeneralizedStateBatch, GpuGeneralizedStateError,
};

/// Invalid layout, exhausted GPU capacity, or a dependent batch error.
#[derive(Debug, thiserror::Error)]
pub enum GpuArticulatedVelocityBiasError {
    /// Source state, articulation, force, and mass layouts differ.
    #[error("invalid articulated velocity bias layout")]
    InvalidInput,
    /// Packed velocity bias exceeds GPU limits.
    #[error("articulated velocity bias exceeds GPU capacity")]
    Capacity,
    /// Scratch mass assembly could not be created.
    #[error(transparent)]
    Mass(#[from] crate::gpu_articulated_mass::GpuArticulatedMassError),
    /// Scratch generalized state could not be created.
    #[error(transparent)]
    State(#[from] GpuGeneralizedStateError),
    /// Shifted pose batch could not be created.
    #[error(transparent)]
    Pose(#[from] GpuArticulatedPoseError),
    /// Shifted link Jacobians could not be created.
    #[error(transparent)]
    LinkTerms(#[from] GpuArticulatedLinkTermsError),
    /// Scratch quaternion state could not be created.
    #[error(transparent)]
    Spherical(#[from] GpuArticulatedSphericalError),
}

#[derive(Debug)]
struct ShiftedTerms {
    _mass: GpuArticulatedMassAssemblyBatch,
    state: GpuGeneralizedStateBatch,
    poses: GpuArticulatedPoseBatch,
    terms: GpuArticulatedLinkTermsBatch,
    spherical: Option<GpuArticulatedSphericalBatch>,
}

#[derive(Debug)]
struct RootShift {
    pipeline: wgpu::ComputePipeline,
    roots: wgpu::Buffer,
    layouts: wgpu::Buffer,
}

#[derive(Debug)]
struct SphericalShift {
    pipeline: wgpu::ComputePipeline,
    orientations: wgpu::Buffer,
    mass_status: wgpu::Buffer,
    layouts: wgpu::Buffer,
    count: usize,
}

/// Subtracts velocity-dependent generalized force from resident joint drives.
///
/// Encode the current pose and link terms, then this pass, then gravity/link
/// loads and mass assembly. The caller's base drive must exclude velocity bias.
/// Two shifted GPU kinematics evaluations approximate the convective term;
/// the current world inertia supplies the gyroscopic term. The implementation
/// intentionally uses a larger finite difference than the
/// f64 CPU reference to retain precision in f32 coordinates.
#[derive(Debug)]
pub struct GpuArticulatedVelocityBiasBatch {
    device: wgpu::Device,
    shift_pipeline: wgpu::ComputePipeline,
    bias_pipeline: wgpu::ComputePipeline,
    current_positions: wgpu::Buffer,
    current_velocities: wgpu::Buffer,
    current_status: wgpu::Buffer,
    owners: wgpu::Buffer,
    metadata: wgpu::Buffer,
    current_links: wgpu::Buffer,
    base_forces: wgpu::Buffer,
    plus: ShiftedTerms,
    minus: ShiftedTerms,
    coordinate_count: usize,
    environment_count: usize,
    root_shift: Option<RootShift>,
    spherical_shift: Option<SphericalShift>,
}

impl GpuArticulatedVelocityBiasBatch {
    /// Allocate shifted GPU kinematics for matching fixed-root articulations.
    /// Spherical states must use legacy intrinsic XYZ coordinates and their derivatives.
    pub fn new(
        state: &GpuGeneralizedStateBatch,
        mass: &GpuArticulatedMassAssemblyBatch,
        forces: &GpuArticulatedForceBatch,
        articulations: &[&Articulation],
        roots: &[Isometry3<f64>],
    ) -> Result<Self, GpuArticulatedVelocityBiasError> {
        Self::new_internal(state, mass, forces, articulations, roots, None, None)
    }

    /// Evaluate fixed/floating bias from the same resident roots used by FK.
    ///
    /// Floating velocities are world-frame linear/angular twists followed by
    /// joint velocities. Each pass copies current roots and shifts them in both
    /// time directions on the GPU; no host root update is needed between steps.
    pub fn new_with_floating_roots(
        state: &GpuGeneralizedStateBatch,
        mass: &GpuArticulatedMassAssemblyBatch,
        forces: &GpuArticulatedForceBatch,
        articulations: &[&Articulation],
        poses: &GpuArticulatedPoseBatch,
    ) -> Result<Self, GpuArticulatedVelocityBiasError> {
        if poses.coordinate_buffer() != state.position_buffer()
            || poses.floating_roots().len() != articulations.len()
            || poses.uses_spherical_quaternions()
        {
            return Err(GpuArticulatedVelocityBiasError::InvalidInput);
        }
        Self::new_internal(
            state,
            mass,
            forces,
            articulations,
            &vec![Isometry3::identity(); articulations.len()],
            Some(poses),
            None,
        )
    }

    /// Evaluate quaternion spherical bias with joint-frame angular velocity slots.
    /// Roots, scalar coordinates, and spherical orientations are perturbed in
    /// both time directions. The source orientation buffer is never modified.
    pub fn new_with_spherical_state(
        state: &GpuGeneralizedStateBatch,
        mass: &GpuArticulatedMassAssemblyBatch,
        forces: &GpuArticulatedForceBatch,
        articulations: &[&Articulation],
        poses: &GpuArticulatedPoseBatch,
        spherical: &GpuArticulatedSphericalBatch,
    ) -> Result<Self, GpuArticulatedVelocityBiasError> {
        if poses.coordinate_buffer() != state.position_buffer()
            || !poses.uses_spherical_quaternions()
            || poses.floating_roots().len() != articulations.len()
            || !spherical.matches_source(state)
            || poses.spherical_orientation_buffer() != spherical.orientation_buffer()
        {
            return Err(GpuArticulatedVelocityBiasError::InvalidInput);
        }
        Self::new_internal(
            state,
            mass,
            forces,
            articulations,
            &vec![Isometry3::identity(); articulations.len()],
            Some(poses),
            Some(spherical),
        )
    }

    fn new_internal(
        state: &GpuGeneralizedStateBatch,
        mass: &GpuArticulatedMassAssemblyBatch,
        forces: &GpuArticulatedForceBatch,
        articulations: &[&Articulation],
        roots: &[Isometry3<f64>],
        source_poses: Option<&GpuArticulatedPoseBatch>,
        source_spherical: Option<&GpuArticulatedSphericalBatch>,
    ) -> Result<Self, GpuArticulatedVelocityBiasError> {
        let floating = source_poses.map_or_else(
            || vec![false; articulations.len()],
            |poses| poses.floating_roots().to_vec(),
        );
        if articulations.is_empty()
            || articulations.len() != roots.len()
            || articulations.len() != state.ranges().len()
            || articulations.len() != mass.dimensions().len()
            || forces.dimensions() != mass.dimensions()
        {
            return Err(GpuArticulatedVelocityBiasError::InvalidInput);
        }
        let mut systems = Vec::with_capacity(articulations.len());
        let mut scratch_states = Vec::with_capacity(articulations.len());
        let mut owners = Vec::new();
        for (index, ((articulation, range), (&dimension, &link_count))) in articulations
            .iter()
            .zip(state.ranges())
            .zip(mass.dimensions().iter().zip(mass.link_counts()))
            .enumerate()
        {
            let root_dofs = if floating[index] { 6 } else { 0 };
            if articulation.dof().checked_add(root_dofs) != Some(dimension)
                || range.len() != dimension
            {
                return Err(GpuArticulatedVelocityBiasError::InvalidInput);
            }
            let positive_count = (0..articulation.link_count())
                .filter(|&link| articulation.link(link).is_some_and(|item| item.mass > 0.0))
                .count();
            let include_all = link_count == articulation.link_count();
            if !include_all && link_count != positive_count {
                return Err(GpuArticulatedVelocityBiasError::InvalidInput);
            }
            let links = (0..articulation.link_count())
                .filter_map(|link| {
                    let item = articulation.link(link)?;
                    (include_all || item.mass > 0.0).then(|| GpuMassLink {
                        mass: item.mass,
                        inertia_world: item.inertia,
                        linear_jacobian: DMatrix::zeros(3, dimension),
                        angular_jacobian: DMatrix::zeros(3, dimension),
                    })
                })
                .collect();
            systems.push(GpuMassAssemblySystem {
                links,
                armature: DVector::zeros(dimension),
                force: DVector::zeros(dimension),
            });
            scratch_states.push(GpuGeneralizedState {
                positions: DVector::zeros(dimension),
                velocities: DVector::zeros(dimension),
            });
            owners.extend(core::iter::repeat_n(
                u32::try_from(index).map_err(|_| GpuArticulatedVelocityBiasError::Capacity)?,
                dimension,
            ));
        }
        let device = state.device();
        let limits = device.limits();
        let owner_bytes = owners
            .len()
            .checked_mul(size_of::<u32>())
            .ok_or(GpuArticulatedVelocityBiasError::Capacity)? as u64;
        let root_layout_bytes = articulations
            .len()
            .checked_mul(16)
            .ok_or(GpuArticulatedVelocityBiasError::Capacity)?
            as u64;
        if owner_bytes > limits.max_buffer_size
            || owner_bytes > u64::from(limits.max_storage_buffer_binding_size)
            || owners.len().div_ceil(64) > limits.max_compute_workgroups_per_dimension as usize
            || articulations.len() > limits.max_compute_workgroups_per_dimension as usize
            || limits.max_storage_buffers_per_shader_stage < 7
            || (source_poses.is_some()
                && (root_layout_bytes > limits.max_buffer_size
                    || root_layout_bytes > u64::from(limits.max_storage_buffer_binding_size)))
        {
            return Err(GpuArticulatedVelocityBiasError::Capacity);
        }
        let plus = Self::shifted(
            state,
            &systems,
            &scratch_states,
            articulations,
            roots,
            &floating,
            source_spherical,
        )?;
        let minus = Self::shifted(
            state,
            &systems,
            &scratch_states,
            articulations,
            roots,
            &floating,
            source_spherical,
        )?;
        let spherical_shift = source_spherical
            .map(|source| {
                let mut packed = Vec::<[u32; 4]>::new();
                for (environment, (slots, range)) in source
                    .velocity_slots()
                    .iter()
                    .zip(state.ranges())
                    .enumerate()
                {
                    for slot in slots {
                        packed.push([
                            u32::try_from(range.start + slot)
                                .map_err(|_| GpuArticulatedVelocityBiasError::Capacity)?,
                            u32::try_from(environment)
                                .map_err(|_| GpuArticulatedVelocityBiasError::Capacity)?,
                            u32::try_from(
                                source
                                    .orientation_index(environment, *slot)
                                    .ok_or(GpuArticulatedVelocityBiasError::InvalidInput)?,
                            )
                            .map_err(|_| GpuArticulatedVelocityBiasError::Capacity)?,
                            0,
                        ]);
                    }
                }
                let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Tessera velocity bias shifted spherical orientations"),
                    source: wgpu::ShaderSource::Wgsl(
                        include_str!("gpu_articulated_spherical_shift.wgsl").into(),
                    ),
                });
                Ok::<_, GpuArticulatedVelocityBiasError>(SphericalShift {
                    pipeline: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("Tessera velocity bias shifted spherical orientations"),
                        layout: None,
                        module: &shader,
                        entry_point: Some("shift_spherical"),
                        compilation_options: Default::default(),
                        cache: None,
                    }),
                    orientations: source.orientation_buffer().clone(),
                    mass_status: state.mass_status_buffer().clone(),
                    layouts: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Tessera velocity bias spherical layouts"),
                        contents: bytemuck::cast_slice(&packed),
                        usage: wgpu::BufferUsages::STORAGE,
                    }),
                    count: packed.len(),
                })
            })
            .transpose()?;
        let root_shift = source_poses
            .map(|poses| {
                let layouts = state
                    .ranges()
                    .iter()
                    .zip(&floating)
                    .enumerate()
                    .map(|(i, (range, &enabled))| {
                        Ok([
                            u32::try_from(range.start)
                                .map_err(|_| GpuArticulatedVelocityBiasError::Capacity)?,
                            u32::from(enabled),
                            u32::try_from(i)
                                .map_err(|_| GpuArticulatedVelocityBiasError::Capacity)?,
                            0,
                        ])
                    })
                    .collect::<Result<Vec<_>, GpuArticulatedVelocityBiasError>>()?;
                let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Tessera velocity bias shifted roots"),
                    source: wgpu::ShaderSource::Wgsl(
                        include_str!("gpu_articulated_root_shift.wgsl").into(),
                    ),
                });
                Ok::<_, GpuArticulatedVelocityBiasError>(RootShift {
                    pipeline: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("Tessera velocity bias shifted roots"),
                        layout: None,
                        module: &shader,
                        entry_point: Some("shift_roots"),
                        compilation_options: Default::default(),
                        cache: None,
                    }),
                    roots: poses.root_pose_buffer().clone(),
                    layouts: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Tessera velocity bias root layouts"),
                        contents: bytemuck::cast_slice(&layouts),
                        usage: wgpu::BufferUsages::STORAGE,
                    }),
                })
            })
            .transpose()?;
        let owners = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera velocity bias coordinate owners"),
            contents: bytemuck::cast_slice(&owners),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let shift_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera velocity bias shifted coordinates"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("gpu_articulated_velocity_shift.wgsl").into(),
            ),
        });
        let shift_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tessera velocity bias shifted coordinates"),
            layout: None,
            module: &shift_shader,
            entry_point: Some("shift_coordinates"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bias_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera articulated velocity bias"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("gpu_articulated_velocity_bias.wgsl").into(),
            ),
        });
        let bias_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tessera articulated velocity bias"),
            layout: None,
            module: &bias_shader,
            entry_point: Some("subtract_velocity_bias"),
            compilation_options: Default::default(),
            cache: None,
        });
        Ok(Self {
            device: device.clone(),
            shift_pipeline,
            bias_pipeline,
            current_positions: state.position_buffer().clone(),
            current_velocities: state.velocity_buffer().clone(),
            current_status: state.status_buffer().clone(),
            owners,
            metadata: mass.metadata_buffer().clone(),
            current_links: mass.link_terms_buffer().clone(),
            base_forces: forces.base_force_buffer().clone(),
            plus,
            minus,
            coordinate_count: state.ranges().last().map_or(0, |range| range.end),
            environment_count: articulations.len(),
            root_shift,
            spherical_shift,
        })
    }

    fn shifted(
        state: &GpuGeneralizedStateBatch,
        systems: &[GpuMassAssemblySystem],
        scratch_states: &[GpuGeneralizedState],
        articulations: &[&Articulation],
        roots: &[Isometry3<f64>],
        floating: &[bool],
        source_spherical: Option<&GpuArticulatedSphericalBatch>,
    ) -> Result<ShiftedTerms, GpuArticulatedVelocityBiasError> {
        let mass = GpuArticulatedMassAssemblyBatch::new(state.device(), state.queue(), systems)?;
        let shifted_state = GpuGeneralizedStateBatch::from_assembly_batch(&mass, scratch_states)?;
        let spherical = source_spherical
            .map(|source| {
                let joints = source
                    .velocity_slots()
                    .iter()
                    .map(|slots| {
                        slots
                            .iter()
                            .map(|&slot| GpuSphericalJointState {
                                velocity_slot: slot,
                                orientation: UnitQuaternion::identity(),
                            })
                            .collect()
                    })
                    .collect::<Vec<_>>();
                GpuArticulatedSphericalBatch::new(&shifted_state, &joints, 0.001)
            })
            .transpose()?;
        let poses = match &spherical {
            Some(spherical) => GpuArticulatedPoseBatch::new_with_spherical_state(
                &shifted_state,
                articulations,
                roots,
                floating,
                spherical,
            )?,
            None => GpuArticulatedPoseBatch::new_with_floating_roots(
                &shifted_state,
                articulations,
                roots,
                floating,
            )?,
        };
        let terms = GpuArticulatedLinkTermsBatch::new(&poses, &mass, articulations)?;
        Ok(ShiftedTerms {
            _mass: mass,
            state: shifted_state,
            poses,
            terms,
            spherical,
        })
    }

    /// Keep legacy fixed roots aligned; resident-root batches copy roots on encode.
    pub fn set_root_pose(
        &self,
        environment: usize,
        root: Isometry3<f64>,
    ) -> Result<(), GpuArticulatedVelocityBiasError> {
        self.plus.poses.set_root_pose(environment, root)?;
        self.minus.poses.set_root_pose(environment, root)?;
        Ok(())
    }

    /// Encode central-difference Jacobians and subtract the resulting bias.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let buffers = [
            &self.current_positions,
            &self.current_velocities,
            self.plus.state.position_buffer(),
            self.minus.state.position_buffer(),
            &self.owners,
            &self.current_status,
        ];
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera velocity bias shift bindings"),
            layout: &self.shift_pipeline.get_bind_group_layout(0),
            entries: &buffers
                .iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Tessera velocity bias shifted coordinates"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.shift_pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(self.coordinate_count.div_ceil(64) as u32, 1, 1);
        }
        if let Some(shift) = &self.root_shift {
            let buffers = [
                &shift.roots,
                &self.current_velocities,
                &shift.layouts,
                self.plus.poses.root_pose_buffer(),
                self.minus.poses.root_pose_buffer(),
                &self.current_status,
            ];
            let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Tessera velocity bias root shift bindings"),
                layout: &shift.pipeline.get_bind_group_layout(0),
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
                label: Some("Tessera velocity bias root shift"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&shift.pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(self.environment_count.div_ceil(64) as u32, 1, 1);
        }
        if let (Some(shift), Some(plus), Some(minus)) = (
            &self.spherical_shift,
            &self.plus.spherical,
            &self.minus.spherical,
        ) {
            let buffers = [
                &shift.orientations,
                &self.current_velocities,
                &shift.layouts,
                plus.orientation_buffer(),
                minus.orientation_buffer(),
                &self.current_status,
                &shift.mass_status,
            ];
            let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Tessera velocity bias spherical shift bindings"),
                layout: &shift.pipeline.get_bind_group_layout(0),
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
                label: Some("Tessera velocity bias spherical shift"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&shift.pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(shift.count.div_ceil(64) as u32, 1, 1);
        }
        self.plus.poses.encode(encoder);
        self.plus.terms.encode(encoder);
        self.minus.poses.encode(encoder);
        self.minus.terms.encode(encoder);
        let buffers = [
            &self.metadata,
            &self.current_links,
            self.plus._mass.link_terms_buffer(),
            self.minus._mass.link_terms_buffer(),
            &self.current_velocities,
            &self.base_forces,
            &self.current_status,
        ];
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera articulated velocity bias bindings"),
            layout: &self.bias_pipeline.get_bind_group_layout(0),
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
            label: Some("Tessera articulated velocity bias"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.bias_pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(self.environment_count as u32, 1, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::articulation::{JointKind, JointSpec, LinkSpec};
    use crate::gpu_articulated_joint_force::{GpuArticulatedJointForceBatch, GpuJointForceInput};
    use crate::gpu_articulated_mass::read_buffer;
    use crate::gpu_contact_pipeline::GpuContactDevice;
    use nalgebra::{Matrix3, Vector3};

    fn system(articulation: &Articulation) -> GpuMassAssemblySystem {
        let n = articulation.dof();
        GpuMassAssemblySystem {
            links: (0..articulation.link_count())
                .map(|index| {
                    let link = articulation.link(index).unwrap();
                    GpuMassLink {
                        mass: link.mass,
                        inertia_world: link.inertia,
                        linear_jacobian: DMatrix::zeros(3, n),
                        angular_jacobian: DMatrix::zeros(3, n),
                    }
                })
                .collect(),
            armature: DVector::from_element(n, 1.0),
            force: DVector::zeros(n),
        }
    }

    #[test]
    fn resident_floating_velocity_bias_and_free_steps_match_cpu() {
        use crate::gpu_articulated_root::GpuArticulatedRootBatch;
        use nalgebra::UnitQuaternion;
        let mut body = link(2.0);
        body.center_of_mass = Vector3::new(0.2, -0.1, 0.3);
        body.inertia = Matrix3::from_diagonal(&Vector3::new(0.4, 0.6, 0.8));
        let mut jointed = Articulation::new(
            vec![body.clone(), body.clone(), body.clone()],
            vec![
                JointSpec {
                    parent: 1,
                    child: 0,
                    kind: JointKind::Revolute,
                    origin: Isometry3::translation(0.8, 0.1, -0.2),
                    axis: Vector3::z(),
                    limits: None,
                },
                JointSpec {
                    parent: 0,
                    child: 2,
                    kind: JointKind::Revolute,
                    origin: Isometry3::translation(0.2, 0.7, 0.1),
                    axis: Vector3::y(),
                    limits: None,
                },
            ],
            1,
        )
        .unwrap();
        jointed.set_mimics(&[(1, 0, -1.5, 0.2)]).unwrap();
        let spherical = Articulation::new(
            vec![body.clone(), body.clone()],
            vec![JointSpec {
                parent: 0,
                child: 1,
                kind: JointKind::Spherical,
                origin: Isometry3::translation(0.3, -0.4, 0.7),
                axis: Vector3::z(),
                limits: None,
            }],
            0,
        )
        .unwrap();
        let pure = Articulation::new(vec![body], vec![], 0).unwrap();
        let arts = [&jointed, &jointed, &pure, &spherical];
        let flags = [true, false, true, true];
        let roots = [
            Isometry3::translation(1.0, -2.0, 0.5)
                * Isometry3::rotation(Vector3::new(0.3, -0.2, 0.1)),
            Isometry3::rotation(Vector3::new(-0.1, 0.2, -0.3)),
            Isometry3::translation(-0.5, 0.2, 2.0)
                * Isometry3::rotation(Vector3::new(0.2, 0.4, -0.3)),
            Isometry3::translation(0.2, 0.8, 1.0)
                * Isometry3::rotation(Vector3::new(-0.4, 0.1, 0.2)),
        ];
        let initial = vec![
            GpuGeneralizedState {
                positions: DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3]),
                velocities: DVector::from_vec(vec![0.1, -0.2, 0.3, 0.4, -0.6, 0.8, 0.7]),
            },
            GpuGeneralizedState {
                positions: DVector::from_vec(vec![-0.2]),
                velocities: DVector::from_vec(vec![-0.4]),
            },
            GpuGeneralizedState {
                positions: DVector::zeros(6),
                velocities: DVector::from_vec(vec![-0.2, 0.4, -0.1, -0.7, 0.3, 0.5]),
            },
            GpuGeneralizedState {
                positions: DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, -0.2, 0.3]),
                velocities: DVector::from_vec(vec![
                    0.2, -0.3, 0.1, -0.2, 0.4, -0.5, 0.6, -0.7, 0.8,
                ]),
            },
        ];
        let systems = arts
            .iter()
            .zip(&initial)
            .map(|(art, state)| {
                let n = state.positions.len();
                GpuMassAssemblySystem {
                    links: (0..art.link_count())
                        .map(|i| {
                            let link = art.link(i).unwrap();
                            GpuMassLink {
                                mass: link.mass,
                                inertia_world: link.inertia,
                                linear_jacobian: DMatrix::zeros(3, n),
                                angular_jacobian: DMatrix::zeros(3, n),
                            }
                        })
                        .collect(),
                    armature: DVector::from_element(n, 0.2),
                    force: DVector::zeros(n),
                }
            })
            .collect::<Vec<_>>();
        let gravity = Vector3::new(0.1, -0.2, -9.81);
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            eprintln!("floating bias and 32 free steps: {backend:?}");
            let mass =
                GpuArticulatedMassAssemblyBatch::new(context.device(), context.queue(), &systems)
                    .unwrap();
            let state = GpuGeneralizedStateBatch::from_assembly_batch(&mass, &initial).unwrap();
            let poses =
                GpuArticulatedPoseBatch::new_with_floating_roots(&state, &arts, &roots, &flags)
                    .unwrap();
            let terms = GpuArticulatedLinkTermsBatch::new(&poses, &mass, &arts).unwrap();
            let forces = GpuArticulatedForceBatch::new(
                &mass,
                &systems.iter().map(|s| s.force.clone()).collect::<Vec<_>>(),
                &[gravity; 4],
            )
            .unwrap();
            let bias = GpuArticulatedVelocityBiasBatch::new_with_floating_roots(
                &state, &mass, &forces, &arts, &poses,
            )
            .unwrap();
            let advance = GpuArticulatedRootBatch::new(&poses, &state, 0.001).unwrap();
            // Verify the individual bias before exercising the full free-step path.
            let mut encoder = context.device().create_command_encoder(&Default::default());
            poses.encode(&mut encoder);
            terms.encode(&mut encoder);
            bias.encode(&mut encoder);
            let _ = context.queue().submit(Some(encoder.finish()));
            let bytes = read_buffer(
                context.device(),
                context.queue(),
                forces.base_force_buffer(),
            )
            .unwrap();
            let values = bytemuck::cast_slice::<u8, f32>(&bytes);
            let mut offset = 0;
            for i in 0..arts.len() {
                let start = if flags[i] { 6 } else { 0 };
                let dynamics = arts[i]
                    .generalized_dynamics(
                        roots[i],
                        &initial[i].positions.as_slice()[start..],
                        &initial[i].velocities,
                        flags[i],
                        gravity,
                    )
                    .unwrap();
                let actual = DVector::from_iterator(
                    initial[i].positions.len(),
                    values[offset..offset + initial[i].positions.len()]
                        .iter()
                        .map(|&v| f64::from(v)),
                );
                assert!((&actual + dynamics.velocity_bias).norm() < 2e-3);
                offset += initial[i].positions.len();
            }
            let mut expected = initial.clone();
            let mut expected_roots = roots;
            for _ in 0..32 {
                for i in 0..arts.len() {
                    let start = if flags[i] { 6 } else { 0 };
                    let dynamics = arts[i]
                        .generalized_dynamics(
                            expected_roots[i],
                            &expected[i].positions.as_slice()[start..],
                            &expected[i].velocities,
                            flags[i],
                            gravity,
                        )
                        .unwrap();
                    let n = expected[i].positions.len();
                    let acceleration = (dynamics.mass + DMatrix::identity(n, n) * 0.2)
                        .lu()
                        .solve(&(dynamics.gravity_force - dynamics.velocity_bias))
                        .unwrap();
                    expected[i].velocities += acceleration * 0.001;
                    let velocity = expected[i].velocities.clone();
                    expected[i].positions += &velocity * 0.001;
                    if flags[i] {
                        expected_roots[i].translation.vector +=
                            Vector3::new(velocity[0], velocity[1], velocity[2]) * 0.001;
                        expected_roots[i].rotation = UnitQuaternion::from_scaled_axis(
                            Vector3::new(velocity[3], velocity[4], velocity[5]) * 0.001,
                        ) * expected_roots[i].rotation;
                    }
                }
            }
            let mut encoder = context.device().create_command_encoder(&Default::default());
            for _ in 0..32 {
                encoder.clear_buffer(forces.base_force_buffer(), 0, None);
                poses.encode(&mut encoder);
                terms.encode(&mut encoder);
                bias.encode(&mut encoder);
                forces.encode(&mut encoder);
                mass.encode(&mut encoder);
                state.encode_step(&mut encoder, 0.001).unwrap();
                advance.encode(&mut encoder);
            }
            poses.encode(&mut encoder);
            let _ = context.queue().submit(Some(encoder.finish()));
            let actual = state.readback().unwrap();
            let actual_poses = poses.readback().unwrap();
            for i in 0..arts.len() {
                assert!((&actual[i].velocities - &expected[i].velocities).norm() < 2e-3);
                assert!((&actual[i].positions - &expected[i].positions).norm() < 2e-4);
                let start = if flags[i] { 6 } else { 0 };
                let cpu = arts[i]
                    .pose(
                        expected_roots[i],
                        &expected[i].positions.as_slice()[start..],
                    )
                    .unwrap();
                for (actual, reference) in actual_poses[i].iter().zip(cpu.links) {
                    assert!(
                        (actual.translation.vector - reference.translation.vector).norm() < 2e-4
                    );
                    assert!((actual.rotation.inverse() * reference.rotation).angle() < 2e-4);
                }
            }
        }
        assert!(tested > 0, "no GPU backend available");
    }

    fn link(mass: f64) -> LinkSpec {
        LinkSpec {
            mass,
            center_of_mass: Vector3::new(0.35, 0.12, -0.08),
            inertia: Matrix3::from_diagonal(&Vector3::new(0.2, 0.3, 0.5)),
        }
    }

    #[test]
    fn quaternion_velocity_bias_matches_cpu_without_mutating_source() {
        let articulation = Articulation::new(
            vec![link(2.0), link(1.8)],
            vec![JointSpec {
                parent: 0,
                child: 1,
                kind: JointKind::Spherical,
                origin: Isometry3::translation(0.2, 0.3, -0.1)
                    * Isometry3::rotation(Vector3::new(0.2, -0.1, 0.4)),
                axis: Vector3::z(),
                limits: None,
            }],
            0,
        )
        .unwrap();
        let roots = [Isometry3::translation(0.4, -0.2, 0.3)
            * Isometry3::rotation(Vector3::new(0.1, -0.2, 0.3)); 2];
        let flags = [false, true];
        let orientations = [
            UnitQuaternion::from_euler_angles(0.3, core::f64::consts::FRAC_PI_2, -0.4),
            UnitQuaternion::from_euler_angles(-0.5, 1.8, 0.2),
        ];
        let states = [
            GpuGeneralizedState {
                positions: DVector::from_element(3, 123.0),
                velocities: DVector::from_column_slice(&[0.8, -0.5, 0.6]),
            },
            GpuGeneralizedState {
                positions: DVector::from_element(9, 123.0),
                velocities: DVector::from_column_slice(&[
                    0.2, -0.3, 0.4, -0.7, 0.9, 0.3, 0.8, -0.5, 0.6,
                ]),
            },
        ];
        let systems = flags.map(|floating| {
            let mut s = system(&articulation);
            if floating {
                s.armature = DVector::from_element(9, 1.0);
                s.force = DVector::zeros(9);
                for l in &mut s.links {
                    l.linear_jacobian = DMatrix::zeros(3, 9);
                    l.angular_jacobian = DMatrix::zeros(3, 9);
                }
            }
            s
        });
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let mass =
                GpuArticulatedMassAssemblyBatch::new(context.device(), context.queue(), &systems)
                    .unwrap();
            let state = GpuGeneralizedStateBatch::from_assembly_batch(&mass, &states).unwrap();
            let spherical = GpuArticulatedSphericalBatch::new(
                &state,
                &[
                    vec![GpuSphericalJointState {
                        velocity_slot: 0,
                        orientation: orientations[0],
                    }],
                    vec![GpuSphericalJointState {
                        velocity_slot: 6,
                        orientation: orientations[1],
                    }],
                ],
                0.001,
            )
            .unwrap();
            let arts = [&articulation; 2];
            let poses = GpuArticulatedPoseBatch::new_with_spherical_state(
                &state, &arts, &roots, &flags, &spherical,
            )
            .unwrap();
            let terms = GpuArticulatedLinkTermsBatch::new(&poses, &mass, &arts).unwrap();
            let forces = GpuArticulatedForceBatch::new(
                &mass,
                &[DVector::zeros(3), DVector::zeros(9)],
                &[Vector3::zeros(); 2],
            )
            .unwrap();
            let bias = GpuArticulatedVelocityBiasBatch::new_with_spherical_state(
                &state, &mass, &forces, &arts, &poses, &spherical,
            )
            .unwrap();
            let before = read_buffer(
                context.device(),
                context.queue(),
                spherical.orientation_buffer(),
            )
            .unwrap();
            let mut encoder = context.device().create_command_encoder(&Default::default());
            poses.encode(&mut encoder);
            terms.encode(&mut encoder);
            bias.encode(&mut encoder);
            let _ = context.queue().submit(Some(encoder.finish()));
            let after = read_buffer(
                context.device(),
                context.queue(),
                spherical.orientation_buffer(),
            )
            .unwrap();
            assert_eq!(before, after);
            let actual_state = state.readback().unwrap();
            for (a, b) in actual_state.iter().zip(&states) {
                assert_eq!(a.positions, b.positions);
                assert!((&a.velocities - &b.velocities).norm() < 1e-6);
            }
            let bytes = read_buffer(
                context.device(),
                context.queue(),
                forces.base_force_buffer(),
            )
            .unwrap();
            let actual = bytes
                .chunks_exact(4)
                .map(|c| f64::from(f32::from_le_bytes(c.try_into().unwrap())))
                .collect::<Vec<_>>();
            let mut offset = 0;
            for i in 0..2 {
                let reference = articulation
                    .generalized_dynamics_with_spherical_orientations(
                        roots[i],
                        &[123.0; 3],
                        &[Some(orientations[i])],
                        &states[i].velocities,
                        flags[i],
                        Vector3::zeros(),
                    )
                    .unwrap()
                    .velocity_bias;
                assert!(reference.norm() > 0.01);
                for (j, expected) in reference.iter().enumerate() {
                    assert!(
                        (actual[offset + j] + expected).abs() < 0.002,
                        "backend {backend:?}, environment {i}, slot {j}: {} vs {}",
                        actual[offset + j],
                        -expected
                    );
                }
                offset += reference.len();
            }
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn gpu_velocity_bias_matches_cpu_for_packed_serial_and_spherical_trees() {
        let joint = |parent, child, kind, origin| JointSpec {
            parent,
            child,
            kind,
            origin,
            axis: Vector3::z(),
            limits: None,
        };
        let serial = Articulation::new(
            vec![link(0.0), link(2.0), link(1.5)],
            vec![
                joint(
                    0,
                    1,
                    JointKind::Revolute,
                    Isometry3::translation(0.7, 0.0, 0.0),
                ),
                joint(
                    1,
                    2,
                    JointKind::Revolute,
                    Isometry3::translation(0.8, 0.1, 0.0),
                ),
            ],
            0,
        )
        .unwrap();
        let spherical = Articulation::new(
            vec![link(0.0), link(1.8)],
            vec![joint(
                0,
                1,
                JointKind::Spherical,
                Isometry3::translation(0.2, 0.3, -0.1),
            )],
            0,
        )
        .unwrap();
        let articulations = [&serial, &spherical];
        let systems = articulations.map(system);
        let roots = [
            Isometry3::identity(),
            Isometry3::translation(0.4, -0.2, 0.3)
                * Isometry3::rotation(Vector3::new(0.1, -0.2, 0.3)),
        ];
        let states = [
            GpuGeneralizedState {
                positions: DVector::from_column_slice(&[0.35, -0.45]),
                velocities: DVector::from_column_slice(&[1.2, -0.7]),
            },
            GpuGeneralizedState {
                positions: DVector::from_column_slice(&[0.2, -0.3, 0.4]),
                velocities: DVector::from_column_slice(&[0.8, -0.5, 0.6]),
            },
        ];
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let mass =
                GpuArticulatedMassAssemblyBatch::new(context.device(), context.queue(), &systems)
                    .unwrap();
            let state = GpuGeneralizedStateBatch::from_assembly_batch(&mass, &states).unwrap();
            let pose = GpuArticulatedPoseBatch::new(&state, &articulations, &roots).unwrap();
            let terms = GpuArticulatedLinkTermsBatch::new(&pose, &mass, &articulations).unwrap();
            let forces = GpuArticulatedForceBatch::new(
                &mass,
                &[DVector::zeros(2), DVector::zeros(3)],
                &[Vector3::zeros(); 2],
            )
            .unwrap();
            let bias = GpuArticulatedVelocityBiasBatch::new(
                &state,
                &mass,
                &forces,
                &articulations,
                &roots,
            )
            .unwrap();
            let mut encoder = context.device().create_command_encoder(&Default::default());
            pose.encode(&mut encoder);
            terms.encode(&mut encoder);
            bias.encode(&mut encoder);
            let _ = context.queue().submit(Some(encoder.finish()));
            let bytes = read_buffer(
                context.device(),
                context.queue(),
                forces.base_force_buffer(),
            )
            .unwrap();
            let actual = bytes
                .chunks_exact(4)
                .map(|chunk| f64::from(f32::from_le_bytes(chunk.try_into().unwrap())))
                .collect::<Vec<_>>();
            let mut offset = 0;
            for (index, articulation) in articulations.iter().enumerate() {
                let reference = articulation
                    .generalized_dynamics(
                        roots[index],
                        states[index].positions.as_slice(),
                        &states[index].velocities,
                        false,
                        Vector3::zeros(),
                    )
                    .unwrap()
                    .velocity_bias;
                assert!(reference.norm() > 0.01);
                for (column, expected) in reference.iter().enumerate() {
                    assert!(
                        (actual[offset + column] + expected).abs() < 0.02,
                        "system {index}, coordinate {column}: {} vs {}",
                        actual[offset + column],
                        -expected,
                    );
                }
                offset += reference.len();
            }
            let drives = [
                vec![GpuJointForceInput::default(); 2],
                vec![GpuJointForceInput::default(); 3],
            ];
            let joint = GpuArticulatedJointForceBatch::new(&state, &forces, &drives).unwrap();
            let dt = 0.01;
            let mut expected_states = states.clone();
            for _ in 0..2 {
                for (index, articulation) in articulations.iter().enumerate() {
                    let dynamics = articulation
                        .generalized_dynamics(
                            roots[index],
                            expected_states[index].positions.as_slice(),
                            &expected_states[index].velocities,
                            false,
                            Vector3::zeros(),
                        )
                        .unwrap();
                    let effective_mass =
                        dynamics.mass + DMatrix::identity(articulation.dof(), articulation.dof());
                    let acceleration = effective_mass.lu().solve(&-dynamics.velocity_bias).unwrap();
                    expected_states[index].velocities += acceleration * dt;
                    expected_states[index].positions +=
                        expected_states[index].velocities.clone() * dt;
                }
            }
            let mut encoder = context.device().create_command_encoder(&Default::default());
            for _ in 0..2 {
                pose.encode(&mut encoder);
                terms.encode(&mut encoder);
                joint.encode(&mut encoder);
                bias.encode(&mut encoder);
                forces.encode(&mut encoder);
                mass.encode(&mut encoder);
                state.encode_step(&mut encoder, dt).unwrap();
            }
            let _ = context.queue().submit(Some(encoder.finish()));
            let actual_states = state.readback().unwrap();
            for (actual, expected) in actual_states.iter().zip(&expected_states) {
                assert!((&actual.positions - &expected.positions).norm() < 2e-3);
                assert!((&actual.velocities - &expected.velocities).norm() < 2e-3);
            }
        }
        assert!(tested > 0, "no GPU backend available");
    }
}
