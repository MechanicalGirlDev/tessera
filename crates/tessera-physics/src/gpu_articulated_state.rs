//! Device-resident generalized coordinates driven by a batched mass solve.
//!
//! Coordinates are Euclidean scalar joint coordinates. Articulated link poses,
//! spherical-joint orientation, contacts, and force assembly remain separate.

use core::ops::Range;

use nalgebra::DVector;
use wgpu::util::DeviceExt;

use crate::gpu_articulated_mass::{GpuArticulatedMassBatch, GpuArticulatedMassError, read_buffer};
use crate::gpu_articulated_mass_assembly::GpuArticulatedMassAssemblyBatch;

/// One packed generalized-coordinate state.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuGeneralizedState {
    /// Scalar generalized coordinates in the mass system's DOF order.
    pub positions: DVector<f64>,
    /// Generalized velocities in the same DOF order.
    pub velocities: DVector<f64>,
}

/// Invalid state, GPU capacity, or a failed mass/state update.
#[derive(Debug, thiserror::Error)]
pub enum GpuGeneralizedStateError {
    /// A state layout, value, or timestep is invalid.
    #[error("invalid generalized state or timestep")]
    InvalidInput,
    /// A packed buffer or dispatch exceeds device limits.
    #[error("generalized state exceeds GPU capacity")]
    Capacity,
    /// The indexed system overflowed during state integration.
    #[error("generalized state {0} became non-finite")]
    NonFinite(usize),
    /// The source mass solve failed or its readback failed.
    #[error(transparent)]
    Mass(#[from] GpuArticulatedMassError),
}

/// Packed GPU position and velocity buffers for independent mass systems.
///
/// Construct this after the mass batch and retain that batch while this session
/// is used. Call the mass batch's `encode` before each `encode_step` in the same
/// command encoder. A failed mass solve leaves that system's state unchanged.
/// An integration overflow faults the system until `reset`; its individual
/// coordinates may already have advanced in the failing dispatch.
#[derive(Debug)]
pub struct GpuGeneralizedStateBatch {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    positions: wgpu::Buffer,
    velocities: wgpu::Buffer,
    accelerations: wgpu::Buffer,
    mass_status: wgpu::Buffer,
    state_status: wgpu::Buffer,
    system_indices: wgpu::Buffer,
    position_integration: wgpu::Buffer,
    ranges: Vec<Range<usize>>,
}

impl GpuGeneralizedStateBatch {
    /// Bind a direct GPU mass solver without downloading its acceleration.
    pub fn from_mass_batch(
        mass: &GpuArticulatedMassBatch,
        states: &[GpuGeneralizedState],
    ) -> Result<Self, GpuGeneralizedStateError> {
        Self::new(
            mass.device(),
            mass.queue(),
            mass.solution_buffer(),
            mass.status_buffer(),
            mass.solution_ranges(),
            states,
        )
    }

    /// Bind a GPU-assembled mass solver without downloading its acceleration.
    pub fn from_assembly_batch(
        mass: &GpuArticulatedMassAssemblyBatch,
        states: &[GpuGeneralizedState],
    ) -> Result<Self, GpuGeneralizedStateError> {
        Self::new(
            mass.device(),
            mass.queue(),
            mass.solution_buffer(),
            mass.status_buffer(),
            mass.solution_ranges(),
            states,
        )
    }

    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        accelerations: &wgpu::Buffer,
        mass_status: &wgpu::Buffer,
        ranges: &[Range<usize>],
        states: &[GpuGeneralizedState],
    ) -> Result<Self, GpuGeneralizedStateError> {
        let (positions, velocities) = pack_states(ranges, states)?;
        let count = positions.len();
        let mut system_indices = Vec::with_capacity(count);
        for (system, range) in ranges.iter().enumerate() {
            system_indices.extend(core::iter::repeat_n(
                u32::try_from(system).map_err(|_| GpuGeneralizedStateError::Capacity)?,
                range.len(),
            ));
        }
        let bytes = (count as u64)
            .checked_mul(4)
            .ok_or(GpuGeneralizedStateError::Capacity)?;
        let status_bytes = (ranges.len() as u64)
            .checked_mul(4)
            .ok_or(GpuGeneralizedStateError::Capacity)?;
        let limits = device.limits();
        if count > u32::MAX as usize
            || bytes > limits.max_buffer_size
            || bytes > u64::from(limits.max_storage_buffer_binding_size)
            || status_bytes > u64::from(limits.max_storage_buffer_binding_size)
            || count.div_ceil(64) > limits.max_compute_workgroups_per_dimension as usize
            || limits.max_storage_buffers_per_shader_stage < 7
        {
            return Err(GpuGeneralizedStateError::Capacity);
        }
        let positions = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera generalized positions"),
            contents: bytemuck::cast_slice(&positions),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let velocities = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera generalized velocities"),
            contents: bytemuck::cast_slice(&velocities),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let system_indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera generalized state owners"),
            contents: bytemuck::cast_slice(&system_indices),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let position_integration = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera generalized position integration mask"),
            contents: bytemuck::cast_slice(&vec![1u32; count]),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let state_status = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera generalized state status"),
            contents: bytemuck::cast_slice(&vec![0u32; ranges.len()]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera generalized state integration"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_articulated_state.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tessera generalized state integration"),
            layout: None,
            module: &shader,
            entry_point: Some("advance"),
            compilation_options: Default::default(),
            cache: None,
        });
        Ok(Self {
            device: device.clone(),
            queue: queue.clone(),
            pipeline,
            positions,
            velocities,
            accelerations: accelerations.clone(),
            mass_status: mass_status.clone(),
            state_status,
            system_indices,
            position_integration,
            ranges: ranges.to_vec(),
        })
    }

    pub(crate) fn position_integration_buffer(&self) -> &wgpu::Buffer {
        &self.position_integration
    }

    /// Select Euclidean position slots while retaining velocity updates for every slot.
    /// Use false for quaternion spherical workspace and independently integrated root poses.
    /// The mask survives reset. Already encoded commands retain their previous mask.
    pub fn set_position_integration(
        &mut self,
        masks: &[Vec<bool>],
    ) -> Result<(), GpuGeneralizedStateError> {
        if masks.len() != self.ranges.len()
            || masks
                .iter()
                .zip(&self.ranges)
                .any(|(mask, range)| mask.len() != range.len())
        {
            return Err(GpuGeneralizedStateError::InvalidInput);
        }
        let packed = masks
            .iter()
            .flatten()
            .map(|&enabled| u32::from(enabled))
            .collect::<Vec<_>>();
        self.position_integration =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Tessera generalized position integration mask"),
                    contents: bytemuck::cast_slice(&packed),
                    usage: wgpu::BufferUsages::STORAGE,
                });
        Ok(())
    }

    /// Encode a semi-implicit velocity and selected scalar-coordinate update.
    pub fn encode_step(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        timestep: f64,
    ) -> Result<(), GpuGeneralizedStateError> {
        let dt = finite_f32(timestep)?;
        if dt <= 0.0 {
            return Err(GpuGeneralizedStateError::InvalidInput);
        }
        let params = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tessera generalized timestep"),
                contents: bytemuck::cast_slice(&[dt, 0.0, 0.0, 0.0]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let buffers = [
            &self.positions,
            &self.velocities,
            &self.accelerations,
            &self.mass_status,
            &self.system_indices,
            &self.state_status,
            &params,
            &self.position_integration,
        ];
        let entries = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>();
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera generalized state bindings"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Tessera generalized state advance"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(
            (self.positions.size() as usize / 4).div_ceil(64) as u32,
            1,
            1,
        );
        Ok(())
    }

    /// Replace the state and clear persistent numerical-failure flags.
    pub(crate) fn validate_reset(
        &self,
        states: &[GpuGeneralizedState],
    ) -> Result<(), GpuGeneralizedStateError> {
        let _ = pack_states(&self.ranges, states)?;
        Ok(())
    }

    /// Validate and replace coordinates and velocities, clearing state fault flags.
    pub fn reset(&self, states: &[GpuGeneralizedState]) -> Result<(), GpuGeneralizedStateError> {
        let (positions, velocities) = pack_states(&self.ranges, states)?;
        self.queue
            .write_buffer(&self.positions, 0, bytemuck::cast_slice(&positions));
        self.queue
            .write_buffer(&self.velocities, 0, bytemuck::cast_slice(&velocities));
        self.queue.write_buffer(
            &self.state_status,
            0,
            bytemuck::cast_slice(&vec![0u32; self.ranges.len()]),
        );
        Ok(())
    }

    /// Read coordinates and reject failed mass solves or numerical overflow.
    pub fn readback(&self) -> Result<Vec<GpuGeneralizedState>, GpuGeneralizedStateError> {
        let mass = read_buffer(&self.device, &self.queue, &self.mass_status)?;
        for (index, flag) in mass.chunks_exact(4).enumerate() {
            let flag = u32::from_le_bytes(
                flag.try_into()
                    .map_err(|_| GpuGeneralizedStateError::Capacity)?,
            );
            match flag {
                0 => {}
                1 => return Err(GpuArticulatedMassError::Singular(index).into()),
                _ => return Err(GpuArticulatedMassError::NonFinite(index).into()),
            }
        }
        let status = read_buffer(&self.device, &self.queue, &self.state_status)?;
        for (index, flag) in status.chunks_exact(4).enumerate() {
            if u32::from_le_bytes(
                flag.try_into()
                    .map_err(|_| GpuGeneralizedStateError::Capacity)?,
            ) != 0
            {
                return Err(GpuGeneralizedStateError::NonFinite(index));
            }
        }
        let positions = read_buffer(&self.device, &self.queue, &self.positions)?;
        let velocities = read_buffer(&self.device, &self.queue, &self.velocities)?;
        self.ranges
            .iter()
            .enumerate()
            .map(|(system, range)| {
                let positions = unpack_range(&positions, range, system)?;
                let velocities = unpack_range(&velocities, range, system)?;
                Ok(GpuGeneralizedState {
                    positions,
                    velocities,
                })
            })
            .collect()
    }

    /// Packed device-resident generalized coordinates.
    pub fn position_buffer(&self) -> &wgpu::Buffer {
        &self.positions
    }

    /// Packed device-resident generalized velocities.
    pub fn velocity_buffer(&self) -> &wgpu::Buffer {
        &self.velocities
    }

    pub(crate) fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub(crate) fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub(crate) fn ranges(&self) -> &[Range<usize>] {
        &self.ranges
    }

    pub(crate) fn status_buffer(&self) -> &wgpu::Buffer {
        &self.state_status
    }

    pub(crate) fn mass_status_buffer(&self) -> &wgpu::Buffer {
        &self.mass_status
    }

    pub(crate) fn system_indices_buffer(&self) -> &wgpu::Buffer {
        &self.system_indices
    }
}

fn finite_f32(value: f64) -> Result<f32, GpuGeneralizedStateError> {
    let result = value as f32;
    if !value.is_finite() || !result.is_finite() || (value != 0.0 && result == 0.0) {
        return Err(GpuGeneralizedStateError::InvalidInput);
    }
    Ok(result)
}

fn pack_states(
    ranges: &[Range<usize>],
    states: &[GpuGeneralizedState],
) -> Result<(Vec<f32>, Vec<f32>), GpuGeneralizedStateError> {
    if ranges.is_empty() || states.len() != ranges.len() {
        return Err(GpuGeneralizedStateError::InvalidInput);
    }
    let mut positions = Vec::with_capacity(ranges.last().map_or(0, |range| range.end));
    let mut velocities = Vec::with_capacity(positions.capacity());
    for (range, state) in ranges.iter().zip(states) {
        if range.start != positions.len()
            || range.is_empty()
            || state.positions.len() != range.len()
            || state.velocities.len() != range.len()
        {
            return Err(GpuGeneralizedStateError::InvalidInput);
        }
        for value in state.positions.iter() {
            positions.push(finite_f32(*value)?);
        }
        for value in state.velocities.iter() {
            velocities.push(finite_f32(*value)?);
        }
    }
    Ok((positions, velocities))
}

fn unpack_range(
    bytes: &[u8],
    range: &Range<usize>,
    system: usize,
) -> Result<DVector<f64>, GpuGeneralizedStateError> {
    range
        .clone()
        .map(|index| {
            let start = index * 4;
            let value = f32::from_le_bytes(
                bytes[start..start + 4]
                    .try_into()
                    .map_err(|_| GpuGeneralizedStateError::Capacity)?,
            );
            if !value.is_finite() {
                return Err(GpuGeneralizedStateError::NonFinite(system));
            }
            Ok(f64::from(value))
        })
        .collect::<Result<Vec<_>, _>>()
        .map(DVector::from_vec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_articulated_mass::GpuArticulatedMassSystem;
    use crate::gpu_articulated_mass_assembly::{GpuMassAssemblySystem, GpuMassLink};
    use crate::gpu_contact_pipeline::GpuContactDevice;
    use nalgebra::{DMatrix, Matrix3};

    fn state(positions: &[f64], velocities: &[f64]) -> GpuGeneralizedState {
        GpuGeneralizedState {
            positions: DVector::from_column_slice(positions),
            velocities: DVector::from_column_slice(velocities),
        }
    }

    #[test]
    fn mass_solve_advances_packed_states_without_intermediate_readback() {
        let systems = [
            GpuArticulatedMassSystem {
                mass: DMatrix::from_diagonal(&DVector::from_vec(vec![2.0, 4.0])),
                force: DVector::from_vec(vec![2.0, 8.0]),
            },
            GpuArticulatedMassSystem {
                mass: DMatrix::from_element(1, 1, 5.0),
                force: DVector::from_element(1, 10.0),
            },
        ];
        let initial = [state(&[0.0, 1.0], &[0.5, -1.0]), state(&[3.0], &[1.0])];
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let mass =
                GpuArticulatedMassBatch::new(context.device(), context.queue(), &systems).unwrap();
            let state = GpuGeneralizedStateBatch::from_mass_batch(&mass, &initial).unwrap();
            assert!(
                state
                    .position_buffer()
                    .usage()
                    .contains(wgpu::BufferUsages::STORAGE)
            );
            let mut encoder = context.device().create_command_encoder(&Default::default());
            mass.encode(&mut encoder);
            state.encode_step(&mut encoder, 0.25).unwrap();
            let _ = context.queue().submit(Some(encoder.finish()));
            let results = state.readback().unwrap();
            for (result, (positions, velocities)) in results.iter().zip([
                ([0.1875, 0.875].as_slice(), [0.75, -0.5].as_slice()),
                ([3.375].as_slice(), [1.5].as_slice()),
            ]) {
                for (value, expected) in result.positions.iter().zip(positions) {
                    assert!((value - expected).abs() < 1e-6, "{backend:?}: {results:?}");
                }
                for (value, expected) in result.velocities.iter().zip(velocities) {
                    assert!((value - expected).abs() < 1e-6, "{backend:?}: {results:?}");
                }
            }
            state.reset(&initial).unwrap();
            let reset = state.readback().unwrap();
            assert_eq!(reset, initial);
            let mut encoder = context.device().create_command_encoder(&Default::default());
            mass.encode(&mut encoder);
            state.encode_step(&mut encoder, 0.25).unwrap();
            state.encode_step(&mut encoder, 0.5).unwrap();
            let _ = context.queue().submit(Some(encoder.finish()));
            let chained = state.readback().unwrap();
            assert!((chained[0].velocities[0] - 1.25).abs() < 1e-6);
            assert!((chained[0].positions[0] - 0.8125).abs() < 1e-6);
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn quaternion_workspace_stays_fixed_while_angular_velocity_advances() {
        use crate::gpu_articulated_spherical::{
            GpuArticulatedSphericalBatch, GpuSphericalJointState,
        };
        use nalgebra::{UnitQuaternion, Vector3};
        let initial = [state(&[0.0, 7.0, 8.0, 9.0], &[0.5, 0.0, 4.0, 0.0])];
        let systems = [GpuArticulatedMassSystem {
            mass: DMatrix::identity(4, 4),
            force: DVector::from_column_slice(&[1.0, 0.0, 0.2, 0.0]),
        }];
        let initial_orientation = UnitQuaternion::from_euler_angles(0.2, 1.4, -0.3);
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let mass =
                GpuArticulatedMassBatch::new(context.device(), context.queue(), &systems).unwrap();
            let mut session = GpuGeneralizedStateBatch::from_mass_batch(&mass, &initial).unwrap();
            session
                .set_position_integration(&[vec![true, false, false, false]])
                .unwrap();
            assert!(session.set_position_integration(&[vec![true]]).is_err());
            let spherical = GpuArticulatedSphericalBatch::new(
                &session,
                &[vec![GpuSphericalJointState {
                    velocity_slot: 1,
                    orientation: initial_orientation,
                }]],
                0.001,
            )
            .unwrap();
            let mut expected = initial_orientation;
            let mut velocity = 4.0;
            let mut encoder = context.device().create_command_encoder(&Default::default());
            mass.encode(&mut encoder);
            for _ in 0..1000 {
                velocity += 0.2 * 0.001;
                expected =
                    UnitQuaternion::from_scaled_axis(Vector3::y() * velocity * 0.001) * expected;
                session.encode_step(&mut encoder, 0.001).unwrap();
                spherical.encode(&mut encoder);
            }
            let _ = context.queue().submit(Some(encoder.finish()));
            let actual = session.readback().unwrap();
            assert_eq!(&actual[0].positions.as_slice()[1..], &[7.0, 8.0, 9.0]);
            assert!((actual[0].positions[0] - 1.0005).abs() < 2e-4);
            assert!((actual[0].velocities[2] - velocity).abs() < 3e-4);
            let orientation = spherical.readback().unwrap();
            assert!((orientation[0][0].inverse() * expected).angle() < 3e-4);
            session.reset(&initial).unwrap();
            let mut encoder = context.device().create_command_encoder(&Default::default());
            session.encode_step(&mut encoder, 0.001).unwrap();
            let _ = context.queue().submit(Some(encoder.finish()));
            let actual = session.readback().unwrap();
            assert_eq!(&actual[0].positions.as_slice()[1..], &[7.0, 8.0, 9.0]);
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn failed_mass_solve_does_not_advance_state() {
        let invalid = GpuArticulatedMassSystem {
            mass: DMatrix::from_element(1, 1, 0.0),
            force: DVector::from_element(1, 1.0),
        };
        let valid = GpuArticulatedMassSystem {
            mass: DMatrix::from_element(1, 1, 2.0),
            force: DVector::from_element(1, 2.0),
        };
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mass =
            GpuArticulatedMassBatch::new(context.device(), context.queue(), &[invalid]).unwrap();
        let state =
            GpuGeneralizedStateBatch::from_mass_batch(&mass, &[state(&[3.0], &[4.0])]).unwrap();
        let mut encoder = context.device().create_command_encoder(&Default::default());
        mass.encode(&mut encoder);
        state.encode_step(&mut encoder, 0.5).unwrap();
        let _ = context.queue().submit(Some(encoder.finish()));
        assert!(matches!(
            state.readback(),
            Err(GpuGeneralizedStateError::Mass(
                GpuArticulatedMassError::Singular(0)
            ))
        ));
        mass.update(&[valid]).unwrap();
        let mut encoder = context.device().create_command_encoder(&Default::default());
        mass.encode(&mut encoder);
        state.encode_step(&mut encoder, 0.5).unwrap();
        let _ = context.queue().submit(Some(encoder.finish()));
        let result = state.readback().unwrap();
        assert!((result[0].positions[0] - 5.25).abs() < 1e-6);
        assert!((result[0].velocities[0] - 4.5).abs() < 1e-6);
    }

    #[test]
    fn assembled_mass_solution_advances_state() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let system = GpuMassAssemblySystem {
            links: vec![GpuMassLink {
                mass: 2.0,
                inertia_world: Matrix3::identity(),
                linear_jacobian: DMatrix::from_column_slice(3, 1, &[1.0, 0.0, 0.0]),
                angular_jacobian: DMatrix::zeros(3, 1),
            }],
            armature: DVector::zeros(1),
            force: DVector::from_element(1, 4.0),
        };
        let mass =
            GpuArticulatedMassAssemblyBatch::new(context.device(), context.queue(), &[system])
                .unwrap();
        let state =
            GpuGeneralizedStateBatch::from_assembly_batch(&mass, &[state(&[1.0], &[0.0])]).unwrap();
        let mut encoder = context.device().create_command_encoder(&Default::default());
        mass.encode(&mut encoder);
        state.encode_step(&mut encoder, 0.25).unwrap();
        let _ = context.queue().submit(Some(encoder.finish()));
        let result = state.readback().unwrap();
        assert!((result[0].velocities[0] - 0.5).abs() < 1e-6);
        assert!((result[0].positions[0] - 1.125).abs() < 1e-6);
    }

    #[test]
    fn integration_overflow_faults_until_reset() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mass = GpuArticulatedMassBatch::new(
            context.device(),
            context.queue(),
            &[GpuArticulatedMassSystem {
                mass: DMatrix::from_element(1, 1, 1.0),
                force: DVector::from_element(1, 1e29),
            }],
        )
        .unwrap();
        let session =
            GpuGeneralizedStateBatch::from_mass_batch(&mass, &[state(&[0.0], &[0.0])]).unwrap();
        let mut encoder = context.device().create_command_encoder(&Default::default());
        mass.encode(&mut encoder);
        session.encode_step(&mut encoder, 100.0).unwrap();
        let _ = context.queue().submit(Some(encoder.finish()));
        assert!(matches!(
            session.readback(),
            Err(GpuGeneralizedStateError::NonFinite(0))
        ));
        mass.update(&[GpuArticulatedMassSystem {
            mass: DMatrix::from_element(1, 1, 1.0),
            force: DVector::from_element(1, 1.0),
        }])
        .unwrap();
        let mut encoder = context.device().create_command_encoder(&Default::default());
        mass.encode(&mut encoder);
        session.encode_step(&mut encoder, 0.5).unwrap();
        let _ = context.queue().submit(Some(encoder.finish()));
        assert!(matches!(
            session.readback(),
            Err(GpuGeneralizedStateError::NonFinite(0))
        ));
        session.reset(&[state(&[0.0], &[0.0])]).unwrap();
        let mut encoder = context.device().create_command_encoder(&Default::default());
        mass.encode(&mut encoder);
        session.encode_step(&mut encoder, 0.5).unwrap();
        let _ = context.queue().submit(Some(encoder.finish()));
        let recovered = session.readback().unwrap();
        assert!((recovered[0].velocities[0] - 0.5).abs() < 1e-6);
        assert!((recovered[0].positions[0] - 0.25).abs() < 1e-6);
    }
}
