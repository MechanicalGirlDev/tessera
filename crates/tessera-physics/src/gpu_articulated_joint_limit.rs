//! Device-resident joint bounds during generalized integration.

use core::mem::size_of;

use wgpu::util::DeviceExt;

use crate::articulation::Articulation;
use crate::gpu_articulated_mass_assembly::GpuArticulatedMassAssemblyBatch;
use crate::gpu_articulated_state::GpuGeneralizedStateBatch;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PackedLimit {
    lower: f32,
    upper: f32,
    max_speed: f32,
    flags: f32,
}

/// Invalid joint-bound layout or exhausted GPU capacity.
#[derive(Debug, thiserror::Error)]
pub enum GpuArticulatedJointLimitError {
    /// Articulation, speed-limit, or timestep input is invalid.
    #[error("invalid articulated joint limits")]
    InvalidInput,
    /// Packed joint limits exceed the GPU buffer or dispatch limits.
    #[error("articulated joint limits exceed GPU capacity")]
    Capacity,
}

/// Integrates generalized state and enforces CPU-compatible joint bounds.
///
/// The velocity cap applies before the position update. A position that crosses
/// a hard bound is clamped and its velocity is set to zero. This matches the
/// fixed-root post-solve rule of `ArticulatedWorld::finish_contact_step`.
/// The position integration mask is captured at construction. Disabled slots
/// retain their position and bypass position bounds, while speed caps still apply.
/// Set the state's mask before constructing this batch; reset retains that mask.
/// Encode after the mass solve instead of `GpuGeneralizedStateBatch::encode_step`.
#[derive(Debug)]
pub struct GpuArticulatedJointLimitBatch {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    positions: wgpu::Buffer,
    position_integration: wgpu::Buffer,
    velocities: wgpu::Buffer,
    accelerations: wgpu::Buffer,
    mass_status: wgpu::Buffer,
    owners: wgpu::Buffer,
    state_status: wgpu::Buffer,
    limits: wgpu::Buffer,
    timestep: wgpu::Buffer,
    packed: Vec<PackedLimit>,
    dimensions: Vec<usize>,
    root_dofs: Vec<usize>,
}

impl GpuArticulatedJointLimitBatch {
    /// Bind static coordinate bounds and optional speed caps to matching batches.
    pub fn new(
        state: &GpuGeneralizedStateBatch,
        mass: &GpuArticulatedMassAssemblyBatch,
        articulations: &[&Articulation],
        velocity_limits: &[Option<f64>],
        timestep: f64,
    ) -> Result<Self, GpuArticulatedJointLimitError> {
        Self::new_with_floating_roots(
            state,
            mass,
            articulations,
            velocity_limits,
            timestep,
            &vec![false; articulations.len()],
        )
    }

    /// Bind joint-only bounds after optional six-slot floating root workspaces.
    ///
    /// Root velocities have neither joint speed caps nor position bounds. Encode
    /// root pose integration after this pass; workspace positions are not poses.
    pub fn new_with_floating_roots(
        state: &GpuGeneralizedStateBatch,
        mass: &GpuArticulatedMassAssemblyBatch,
        articulations: &[&Articulation],
        velocity_limits: &[Option<f64>],
        timestep: f64,
        floating_roots: &[bool],
    ) -> Result<Self, GpuArticulatedJointLimitError> {
        if articulations.is_empty()
            || floating_roots.len() != articulations.len()
            || articulations.len() != state.ranges().len()
            || articulations.len() != mass.dimensions().len()
            || !timestep.is_finite()
            || timestep <= 0.0
        {
            return Err(GpuArticulatedJointLimitError::InvalidInput);
        }
        let dt = finite_f32(timestep)?;
        let dimensions = mass.dimensions().to_vec();
        let root_dofs = floating_roots
            .iter()
            .map(|&flag| if flag { 6 } else { 0 })
            .collect::<Vec<_>>();
        for (((articulation, range), &dimension), &root_dof) in articulations
            .iter()
            .zip(state.ranges())
            .zip(&dimensions)
            .zip(&root_dofs)
        {
            if articulation.dof().checked_add(root_dof) != Some(dimension)
                || range.len() != dimension
            {
                return Err(GpuArticulatedJointLimitError::InvalidInput);
            }
        }
        let joint_limits = pack_limits(articulations, velocity_limits)?;
        let mut packed = Vec::new();
        let mut offset = 0;
        for (articulation, &root_dof) in articulations.iter().zip(&root_dofs) {
            packed.extend(core::iter::repeat_n(
                PackedLimit {
                    lower: 0.0,
                    upper: 0.0,
                    max_speed: 0.0,
                    flags: 0.0,
                },
                root_dof,
            ));
            packed.extend_from_slice(&joint_limits[offset..offset + articulation.dof()]);
            offset += articulation.dof();
        }
        let count = packed.len();
        let bytes = count
            .checked_mul(size_of::<PackedLimit>())
            .ok_or(GpuArticulatedJointLimitError::Capacity)? as u64;
        let device = state.device();
        let limits = device.limits();
        if bytes > limits.max_buffer_size
            || bytes > u64::from(limits.max_storage_buffer_binding_size)
            || count.div_ceil(64) > limits.max_compute_workgroups_per_dimension as usize
            || limits.max_storage_buffers_per_shader_stage < 8
            || state.position_buffer().size() != mass.solution_buffer().size()
        {
            return Err(GpuArticulatedJointLimitError::Capacity);
        }
        let limits_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera articulated joint limits"),
            contents: bytemuck::cast_slice(&packed),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let timestep = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera joint limit timestep"),
            contents: bytemuck::cast_slice(&[dt, 0.0f32, 0.0, 0.0]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera articulated limited state update"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("gpu_articulated_joint_limit.wgsl").into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tessera articulated limited state update"),
            layout: None,
            module: &shader,
            entry_point: Some("advance_limited"),
            compilation_options: Default::default(),
            cache: None,
        });
        Ok(Self {
            device: device.clone(),
            queue: state.queue().clone(),
            pipeline,
            positions: state.position_buffer().clone(),
            position_integration: state.position_integration_buffer().clone(),
            velocities: state.velocity_buffer().clone(),
            accelerations: mass.solution_buffer().clone(),
            mass_status: mass.status_buffer().clone(),
            owners: state.system_indices_buffer().clone(),
            state_status: state.status_buffer().clone(),
            limits: limits_buffer,
            timestep,
            packed,
            dimensions,
            root_dofs,
        })
    }

    /// Replace per-environment speed caps without changing hard position bounds.
    pub fn update_velocity_limits(
        &mut self,
        velocity_limits: &[Option<f64>],
    ) -> Result<(), GpuArticulatedJointLimitError> {
        if velocity_limits.len() != self.dimensions.len() {
            return Err(GpuArticulatedJointLimitError::InvalidInput);
        }
        let mut updated = self.packed.clone();
        let mut offset = 0;
        for ((&dimension, limit), &root_dof) in self
            .dimensions
            .iter()
            .zip(velocity_limits)
            .zip(&self.root_dofs)
        {
            let cap = limit.map(finite_f32).transpose()?;
            if cap.is_some_and(|value| value <= 0.0) {
                return Err(GpuArticulatedJointLimitError::InvalidInput);
            }
            for entry in &mut updated[offset + root_dof..offset + dimension] {
                entry.max_speed = cap.unwrap_or(0.0);
                entry.flags = if entry.flags == 1.0 || entry.flags == 3.0 {
                    1.0
                } else {
                    0.0
                } + if cap.is_some() { 2.0 } else { 0.0 };
            }
            offset += dimension;
        }
        self.queue
            .write_buffer(&self.limits, 0, bytemuck::cast_slice(&updated));
        self.packed = updated;
        Ok(())
    }

    /// Replace caps per generalized coordinate, preserving hard position bounds.
    /// Floating-root prefix entries must be None. Every environment is validated
    /// before any device upload, so a rejected update retains all previous caps.
    pub fn update_coordinate_velocity_limits(
        &mut self,
        limits: &[Vec<Option<f64>>],
    ) -> Result<(), GpuArticulatedJointLimitError> {
        if limits.len() != self.dimensions.len() {
            return Err(GpuArticulatedJointLimitError::InvalidInput);
        }
        let mut updated = self.packed.clone();
        let mut offset = 0;
        for ((caps, &dimension), &root_dofs) in
            limits.iter().zip(&self.dimensions).zip(&self.root_dofs)
        {
            if caps.len() != dimension || caps[..root_dofs].iter().any(Option::is_some) {
                return Err(GpuArticulatedJointLimitError::InvalidInput);
            }
            for (entry, cap) in updated[offset..offset + dimension].iter_mut().zip(caps) {
                let cap = cap.map(finite_f32).transpose()?;
                if cap.is_some_and(|value| value <= 0.0) {
                    return Err(GpuArticulatedJointLimitError::InvalidInput);
                }
                entry.max_speed = cap.unwrap_or(0.0);
                entry.flags = if entry.flags == 1.0 || entry.flags == 3.0 {
                    1.0
                } else {
                    0.0
                } + if cap.is_some() { 2.0 } else { 0.0 };
            }
            offset += dimension;
        }
        self.queue
            .write_buffer(&self.limits, 0, bytemuck::cast_slice(&updated));
        self.packed = updated;
        Ok(())
    }

    /// Encode bounded state integration after mass assembly and solve.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let buffers = [
            &self.positions,
            &self.velocities,
            &self.accelerations,
            &self.mass_status,
            &self.owners,
            &self.state_status,
            &self.limits,
            &self.timestep,
            &self.position_integration,
        ];
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera articulated joint limit bindings"),
            layout: &self.pipeline.get_bind_group_layout(0),
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
            label: Some("Tessera articulated limited state update"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(self.packed.len().div_ceil(64) as u32, 1, 1);
    }
}

fn pack_limits(
    articulations: &[&Articulation],
    velocity_limits: &[Option<f64>],
) -> Result<Vec<PackedLimit>, GpuArticulatedJointLimitError> {
    if articulations.len() != velocity_limits.len() {
        return Err(GpuArticulatedJointLimitError::InvalidInput);
    }
    let mut packed = Vec::new();
    for (articulation, speed_limit) in articulations.iter().zip(velocity_limits) {
        let cap = speed_limit.map(finite_f32).transpose()?;
        if cap.is_some_and(|value| value <= 0.0) {
            return Err(GpuArticulatedJointLimitError::InvalidInput);
        }
        for slot in 0..articulation.dof() {
            let bounds = articulation.joint_limit(slot);
            let (lower, upper) = bounds
                .map(|(lower, upper)| Ok((finite_f32(lower)?, finite_f32(upper)?)))
                .transpose()?
                .unwrap_or((0.0, 0.0));
            packed.push(PackedLimit {
                lower,
                upper,
                max_speed: cap.unwrap_or(0.0),
                flags: if bounds.is_some() { 1.0 } else { 0.0 }
                    + if cap.is_some() { 2.0 } else { 0.0 },
            });
        }
    }
    Ok(packed)
}

fn finite_f32(value: f64) -> Result<f32, GpuArticulatedJointLimitError> {
    let narrowed = value as f32;
    if !value.is_finite() || !narrowed.is_finite() || (value != 0.0 && narrowed == 0.0) {
        return Err(GpuArticulatedJointLimitError::InvalidInput);
    }
    Ok(narrowed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::articulation::{JointKind, JointSpec, LinkSpec};
    use crate::gpu_articulated_mass_assembly::GpuMassAssemblySystem;
    use crate::gpu_articulated_state::GpuGeneralizedState;
    use crate::gpu_contact_pipeline::GpuContactDevice;
    use nalgebra::{DVector, Isometry3, Matrix3, Vector3};

    fn articulation(bounds: Option<(f64, f64)>) -> Articulation {
        Articulation::new(
            vec![
                LinkSpec {
                    mass: 0.0,
                    center_of_mass: Vector3::zeros(),
                    inertia: Matrix3::zeros(),
                };
                2
            ],
            vec![JointSpec {
                parent: 0,
                child: 1,
                kind: JointKind::Prismatic,
                origin: Isometry3::identity(),
                axis: Vector3::z(),
                limits: bounds,
            }],
            0,
        )
        .unwrap()
    }

    #[test]
    fn coordinate_speed_caps_preserve_free_slots_and_reject_partial_updates() {
        let art = Articulation::new(
            vec![
                LinkSpec {
                    mass: 0.0,
                    center_of_mass: Vector3::zeros(),
                    inertia: Matrix3::zeros()
                };
                3
            ],
            vec![
                JointSpec {
                    parent: 0,
                    child: 1,
                    kind: JointKind::Prismatic,
                    origin: Isometry3::identity(),
                    axis: Vector3::x(),
                    limits: Some((0.0, 1.0)),
                },
                JointSpec {
                    parent: 1,
                    child: 2,
                    kind: JointKind::Prismatic,
                    origin: Isometry3::identity(),
                    axis: Vector3::y(),
                    limits: None,
                },
            ],
            0,
        )
        .unwrap();
        let systems = [2, 8].map(|n| GpuMassAssemblySystem {
            links: Vec::new(),
            armature: DVector::from_element(n, 1.0),
            force: DVector::zeros(n),
        });
        let initial = [
            GpuGeneralizedState {
                positions: DVector::from_vec(vec![0.99, 0.0]),
                velocities: DVector::from_vec(vec![3.0, -5.0]),
            },
            GpuGeneralizedState {
                positions: DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.99, 0.0]),
                velocities: DVector::from_vec(vec![7.0; 8]),
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
            let state = GpuGeneralizedStateBatch::from_assembly_batch(&mass, &initial).unwrap();
            let mut limits = GpuArticulatedJointLimitBatch::new_with_floating_roots(
                &state,
                &mass,
                &[&art, &art],
                &[Some(0.2); 2],
                0.1,
                &[false, true],
            )
            .unwrap();
            let caps = vec![
                vec![Some(0.5), None],
                vec![None, None, None, None, None, None, Some(0.5), None],
            ];
            limits.update_coordinate_velocity_limits(&caps).unwrap();
            let saved = bytemuck::cast_slice::<PackedLimit, u8>(&limits.packed).to_vec();
            for bad in [Some(-1.0), Some(f64::NAN), Some(1e-100), Some(f64::MAX)] {
                let mut rejected = caps.clone();
                rejected[0][0] = Some(0.1);
                rejected[1][7] = bad;
                assert!(limits.update_coordinate_velocity_limits(&rejected).is_err());
                assert_eq!(
                    bytemuck::cast_slice::<PackedLimit, u8>(&limits.packed),
                    saved
                );
            }
            let mut rejected = caps.clone();
            rejected[1][0] = Some(1.0);
            assert!(limits.update_coordinate_velocity_limits(&rejected).is_err());
            rejected = caps.clone();
            let _ = rejected[1].pop();
            assert!(limits.update_coordinate_velocity_limits(&rejected).is_err());
            let mut encoder = context.device().create_command_encoder(&Default::default());
            mass.encode(&mut encoder);
            limits.encode(&mut encoder);
            let _ = context.queue().submit(Some(encoder.finish()));
            let output = state.readback().unwrap();
            assert_eq!(output[0].positions[0], 1.0);
            assert_eq!(output[0].velocities[0], 0.0);
            assert_eq!(output[0].velocities[1], -5.0);
            assert!((output[0].positions[1] + 0.5).abs() < 1e-6);
            assert_eq!(&output[1].velocities.as_slice()[..6], &[7.0; 6]);
            assert_eq!(output[1].velocities[6], 0.0);
            assert_eq!(output[1].velocities[7], 7.0);
            eprintln!("coordinate speed caps passed on {backend:?}");
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn limited_scalar_and_quaternion_workspace_share_acceleration_update() {
        use crate::gpu_articulated_spherical::{
            GpuArticulatedSphericalBatch, GpuSphericalJointState,
        };
        use nalgebra::UnitQuaternion;
        let art = Articulation::new(
            vec![
                LinkSpec {
                    mass: 0.0,
                    center_of_mass: Vector3::zeros(),
                    inertia: Matrix3::zeros()
                };
                3
            ],
            vec![
                JointSpec {
                    parent: 0,
                    child: 1,
                    kind: JointKind::Prismatic,
                    origin: Isometry3::identity(),
                    axis: Vector3::z(),
                    limits: Some((0.0, 1.0)),
                },
                JointSpec {
                    parent: 1,
                    child: 2,
                    kind: JointKind::Spherical,
                    origin: Isometry3::identity(),
                    axis: Vector3::z(),
                    limits: None,
                },
            ],
            0,
        )
        .unwrap();
        let systems = [GpuMassAssemblySystem {
            links: Vec::new(),
            armature: DVector::from_element(4, 1.0),
            force: DVector::from_vec(vec![1.0, 0.0, 2.0, 0.0]),
        }];
        let initial = [GpuGeneralizedState {
            positions: DVector::from_vec(vec![0.99, 7.0, 8.0, 9.0]),
            velocities: DVector::from_vec(vec![2.0, 0.0, 4.0, 0.0]),
        }];
        let initial_orientation = UnitQuaternion::from_euler_angles(0.2, 1.4, -0.3);
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let mass =
                GpuArticulatedMassAssemblyBatch::new(context.device(), context.queue(), &systems)
                    .unwrap();
            let mut state = GpuGeneralizedStateBatch::from_assembly_batch(&mass, &initial).unwrap();
            state
                .set_position_integration(&[vec![true, false, false, false]])
                .unwrap();
            let mut limits =
                GpuArticulatedJointLimitBatch::new(&state, &mass, &[&art], &[Some(4.1)], 0.001)
                    .unwrap();
            let spherical = GpuArticulatedSphericalBatch::new(
                &state,
                &[vec![GpuSphericalJointState {
                    velocity_slot: 1,
                    orientation: initial_orientation,
                }]],
                0.001,
            )
            .unwrap();
            let mut expected = initial_orientation;
            let mut omega: f64 = 4.0;
            let mut encoder = context.device().create_command_encoder(&Default::default());
            mass.encode(&mut encoder);
            for _ in 0..1000 {
                omega = (omega + 0.002).min(4.1);
                expected =
                    UnitQuaternion::from_scaled_axis(Vector3::y() * omega * 0.001) * expected;
                limits.encode(&mut encoder);
                spherical.encode(&mut encoder);
            }
            let _ = context.queue().submit(Some(encoder.finish()));
            let result = state.readback().unwrap();
            assert_eq!(result[0].positions[0], 1.0);
            assert_eq!(result[0].velocities[0], 0.0);
            assert_eq!(&result[0].positions.as_slice()[1..], &[7.0, 8.0, 9.0]);
            assert!((result[0].velocities[2] - 4.1).abs() < 1e-6);
            assert!((spherical.readback().unwrap()[0][0].inverse() * expected).angle() < 3e-4);
            state.reset(&initial).unwrap();
            limits.update_velocity_limits(&[Some(3.0)]).unwrap();
            let mut encoder = context.device().create_command_encoder(&Default::default());
            limits.encode(&mut encoder);
            let _ = context.queue().submit(Some(encoder.finish()));
            let result = state.readback().unwrap();
            assert_eq!(&result[0].positions.as_slice()[1..], &[7.0, 8.0, 9.0]);
            assert_eq!(result[0].velocities[2], 3.0);
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn floating_root_efforts_bypass_joint_caps_and_implicit_drives_use_joint_slots() {
        use crate::articulated_world::{JointMotor, JointPassive};
        use crate::gpu_articulated_force::GpuArticulatedForceBatch;
        use crate::gpu_articulated_joint_force::{
            GpuArticulatedJointForceBatch, GpuJointForceInput,
        };
        let bounded = articulation(Some((0.0, 1.0)));
        let pure = Articulation::new(
            vec![LinkSpec {
                mass: 0.0,
                center_of_mass: Vector3::zeros(),
                inertia: Matrix3::zeros(),
            }],
            vec![],
            0,
        )
        .unwrap();
        let arts = [&bounded, &bounded, &pure];
        let flags = [true, false, true];
        let dimensions = [7, 1, 6];
        let systems = dimensions
            .iter()
            .map(|&n| GpuMassAssemblySystem {
                links: Vec::new(),
                armature: DVector::from_element(n, 1.0),
                force: DVector::zeros(n),
            })
            .collect::<Vec<_>>();
        let initial = vec![
            GpuGeneralizedState {
                positions: DVector::from_vec(vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 0.99]),
                velocities: DVector::from_vec(vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 2.0]),
            },
            GpuGeneralizedState {
                positions: DVector::from_element(1, 0.01),
                velocities: DVector::from_element(1, -2.0),
            },
            GpuGeneralizedState {
                positions: DVector::from_vec(vec![-10.0, -11.0, -12.0, -13.0, -14.0, -15.0]),
                velocities: DVector::from_vec(vec![-2.0, -3.0, -4.0, -5.0, -6.0, -7.0]),
            },
        ];
        let drive = GpuJointForceInput {
            passive: JointPassive {
                stiffness: 10.0,
                damping: 1.0,
                rest_position: 0.5,
            },
            motor: Some(JointMotor {
                position_target: Some(1.0),
                velocity_target: 0.0,
                stiffness: 20.0,
                damping: 0.0,
                max_force: 1.0,
            }),
            ..Default::default()
        };
        let mut inputs = dimensions
            .iter()
            .map(|&n| {
                vec![
                    GpuJointForceInput {
                        base_force: 0.1,
                        ..Default::default()
                    };
                    n
                ]
            })
            .collect::<Vec<_>>();
        inputs[0][6] = drive;
        inputs[1][0] = drive;
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            eprintln!("floating joint drive/limit offsets: {backend:?}");
            let mass =
                GpuArticulatedMassAssemblyBatch::new(context.device(), context.queue(), &systems)
                    .unwrap();
            let state = GpuGeneralizedStateBatch::from_assembly_batch(&mass, &initial).unwrap();
            let forces = GpuArticulatedForceBatch::new(
                &mass,
                &dimensions
                    .iter()
                    .map(|&n| DVector::zeros(n))
                    .collect::<Vec<_>>(),
                &[Vector3::zeros(); 3],
            )
            .unwrap();
            let drives = GpuArticulatedJointForceBatch::new_implicit(
                &state,
                &forces,
                &inputs,
                &systems
                    .iter()
                    .map(|s| s.armature.clone())
                    .collect::<Vec<_>>(),
                0.1,
            )
            .unwrap();
            assert!(
                GpuArticulatedJointLimitBatch::new(&state, &mass, &arts, &[Some(0.5); 3], 0.1)
                    .is_err()
            );
            assert!(
                GpuArticulatedJointLimitBatch::new_with_floating_roots(
                    &state,
                    &mass,
                    &arts,
                    &[Some(0.5); 3],
                    0.1,
                    &[true]
                )
                .is_err()
            );
            let mut limits = GpuArticulatedJointLimitBatch::new_with_floating_roots(
                &state,
                &mass,
                &arts,
                &[Some(0.5); 3],
                0.1,
                &flags,
            )
            .unwrap();
            for cap in [0.5, 0.25] {
                state.reset(&initial).unwrap();
                limits.update_velocity_limits(&[Some(cap); 3]).unwrap();
                let mut expected = initial.clone();
                for _ in 0..10 {
                    for i in 0..3 {
                        for slot in 0..dimensions[i] {
                            let root_slot = flags[i] && slot < 6;
                            let q = expected[i].positions[slot];
                            let v = expected[i].velocities[slot];
                            let acceleration = if root_slot {
                                0.1
                            } else {
                                let force = 10.0 * (0.5 - q) - v
                                    + (20.0 * (1.0 - q)).clamp(-1.0, 1.0)
                                    - 0.1 * 10.0 * v;
                                force / (1.0 + 0.1 + 0.01 * 10.0)
                            };
                            let next_v = v + 0.1 * acceleration;
                            let next_v = if root_slot {
                                next_v
                            } else {
                                next_v.clamp(-cap, cap)
                            };
                            let next_q = q + 0.1 * next_v;
                            expected[i].positions[slot] = if root_slot {
                                next_q
                            } else {
                                next_q.clamp(0.0, 1.0)
                            };
                            expected[i].velocities[slot] =
                                if !root_slot && !(0.0..=1.0).contains(&next_q) {
                                    0.0
                                } else {
                                    next_v
                                };
                        }
                    }
                }
                let mut encoder = context.device().create_command_encoder(&Default::default());
                for _ in 0..10 {
                    drives.encode(&mut encoder);
                    forces.encode(&mut encoder);
                    mass.encode(&mut encoder);
                    limits.encode(&mut encoder);
                }
                let _ = context.queue().submit(Some(encoder.finish()));
                for (actual, reference) in state.readback().unwrap().iter().zip(&expected) {
                    assert!((&actual.positions - &reference.positions).norm() < 2e-5);
                    assert!((&actual.velocities - &reference.velocities).norm() < 2e-5);
                }
            }
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn gpu_limits_stop_at_both_bounds_and_cap_unbounded_velocity() {
        let models = [
            articulation(Some((0.0, 1.0))),
            articulation(Some((0.0, 1.0))),
            articulation(None),
        ];
        let systems = vec![
            GpuMassAssemblySystem {
                links: Vec::new(),
                armature: DVector::from_element(1, 1.0),
                force: DVector::zeros(1),
            };
            3
        ];
        let states = [
            GpuGeneralizedState {
                positions: DVector::from_element(1, 0.01),
                velocities: DVector::from_element(1, -2.0),
            },
            GpuGeneralizedState {
                positions: DVector::from_element(1, 0.99),
                velocities: DVector::from_element(1, 2.0),
            },
            GpuGeneralizedState {
                positions: DVector::zeros(1),
                velocities: DVector::from_element(1, 2.0),
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
            let mut limits = GpuArticulatedJointLimitBatch::new(
                &state,
                &mass,
                &[&models[0], &models[1], &models[2]],
                &[None, Some(0.5), Some(0.5)],
                0.1,
            )
            .unwrap();
            let mut encoder = context.device().create_command_encoder(&Default::default());
            mass.encode(&mut encoder);
            limits.encode(&mut encoder);
            let _ = context.queue().submit(Some(encoder.finish()));
            let actual = state.readback().unwrap();
            assert!(actual[0].positions[0].abs() < 1e-6);
            assert!(actual[0].velocities[0].abs() < 1e-6);
            assert!((actual[1].positions[0] - 1.0).abs() < 1e-6);
            assert!(actual[1].velocities[0].abs() < 1e-6);
            assert!((actual[2].positions[0] - 0.05).abs() < 1e-6);
            assert!((actual[2].velocities[0] - 0.5).abs() < 1e-6);
            assert!(
                limits
                    .update_velocity_limits(&[None, Some(-1.0), None])
                    .is_err()
            );
            state.reset(&states).unwrap();
            limits
                .update_velocity_limits(&[None, None, Some(0.25)])
                .unwrap();
            let mut encoder = context.device().create_command_encoder(&Default::default());
            mass.encode(&mut encoder);
            limits.encode(&mut encoder);
            let _ = context.queue().submit(Some(encoder.finish()));
            let updated = state.readback().unwrap();
            assert!((updated[2].positions[0] - 0.025).abs() < 1e-6);
            assert!((updated[2].velocities[0] - 0.25).abs() < 1e-6);
        }
        assert!(tested > 0, "no GPU backend available");
    }
}
