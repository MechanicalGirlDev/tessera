//! Quaternion spherical-joint state integrated from joint-frame angular velocities.

use crate::gpu_articulated_mass::read_buffer;
use crate::gpu_articulated_state::GpuGeneralizedStateBatch;
use nalgebra::{Quaternion, UnitQuaternion};
use wgpu::util::DeviceExt;

/// Initial orientation and the first of three angular velocity slots in an environment.
#[derive(Debug, Clone, Copy)]
pub struct GpuSphericalJointState {
    /// Local generalized slot containing joint-frame angular velocity X, Y, Z.
    /// These slots must use a tangent-space mass/Jacobian layout, not Euler rates.
    pub velocity_slot: usize,
    /// Child orientation relative to the parent-side joint frame.
    pub orientation: UnitQuaternion<f64>,
}

/// Invalid layout, device capacity, or a faulted source environment.
#[derive(Debug, thiserror::Error)]
pub enum GpuArticulatedSphericalError {
    /// Invalid timestep, overlapping slots, dimensions, or quaternion.
    #[error("invalid spherical quaternion state input")]
    InvalidInput,
    /// Requested buffers exceed device capacity.
    #[error("spherical quaternion state exceeds GPU capacity")]
    Capacity,
    /// A source environment faulted; reset both source state and orientations.
    #[error("spherical quaternion source environment {0} faulted")]
    SourceFault(usize),
    /// Readback failed or returned invalid state.
    #[error("spherical quaternion readback failed: {0}")]
    Readback(String),
}

/// Device-resident spherical orientations with a left exponential quaternion update.
///
/// This integrates prescribed joint-frame angular velocities. It does not convert
/// the legacy intrinsic-XYZ mass/Jacobian layout into a tangent-space layout.
/// Encode it after the matching velocity solve and before forward kinematics.
/// Submit and read back under the same external serialization as the source state.
#[derive(Debug)]
pub struct GpuArticulatedSphericalBatch {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    orientations: wgpu::Buffer,
    velocities: wgpu::Buffer,
    layouts: wgpu::Buffer,
    status: wgpu::Buffer,
    mass_status: wgpu::Buffer,
    timestep: wgpu::Buffer,
    slots: Vec<Vec<usize>>,
    count: usize,
}

impl GpuArticulatedSphericalBatch {
    /// Construct quaternion state in environment order and then supplied joint order.
    pub fn new(
        state: &GpuGeneralizedStateBatch,
        joints: &[Vec<GpuSphericalJointState>],
        timestep: f64,
    ) -> Result<Self, GpuArticulatedSphericalError> {
        let dt = timestep as f32;
        if !timestep.is_finite()
            || !dt.is_finite()
            || dt <= 0.0
            || joints.len() != state.ranges().len()
        {
            return Err(GpuArticulatedSphericalError::InvalidInput);
        }
        let mut layouts = Vec::<[u32; 4]>::new();
        let mut slots = Vec::new();
        for (environment, (items, range)) in joints.iter().zip(state.ranges()).enumerate() {
            let mut occupied = vec![false; range.len()];
            let mut environment_slots = Vec::new();
            for joint in items {
                let end = joint
                    .velocity_slot
                    .checked_add(3)
                    .ok_or(GpuArticulatedSphericalError::InvalidInput)?;
                if end > range.len() || occupied[joint.velocity_slot..end].iter().any(|v| *v) {
                    return Err(GpuArticulatedSphericalError::InvalidInput);
                }
                occupied[joint.velocity_slot..end].fill(true);
                environment_slots.push(joint.velocity_slot);
                layouts.push([
                    u32::try_from(range.start + joint.velocity_slot)
                        .map_err(|_| GpuArticulatedSphericalError::Capacity)?,
                    u32::try_from(environment)
                        .map_err(|_| GpuArticulatedSphericalError::Capacity)?,
                    0,
                    0,
                ]);
            }
            slots.push(environment_slots);
        }
        if layouts.is_empty() {
            return Err(GpuArticulatedSphericalError::InvalidInput);
        }
        let packed = pack_orientations(joints)?;
        let device = state.device();
        let limits = device.limits();
        let bytes = layouts
            .len()
            .checked_mul(16)
            .ok_or(GpuArticulatedSphericalError::Capacity)? as u64;
        if bytes > limits.max_buffer_size
            || bytes > u64::from(limits.max_storage_buffer_binding_size)
            || layouts.len().div_ceil(64) > limits.max_compute_workgroups_per_dimension as usize
            || limits.max_storage_buffers_per_shader_stage < 5
        {
            return Err(GpuArticulatedSphericalError::Capacity);
        }
        let orientations = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera spherical orientations"),
            contents: bytemuck::cast_slice(&packed),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let layout_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera spherical layouts"),
            contents: bytemuck::cast_slice(&layouts),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let timestep = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera spherical timestep"),
            contents: bytemuck::cast_slice(&[dt, 0.0f32, 0.0, 0.0]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera spherical integration"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_articulated_spherical.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tessera spherical integration"),
            layout: None,
            module: &shader,
            entry_point: Some("advance_spherical"),
            compilation_options: Default::default(),
            cache: None,
        });
        Ok(Self {
            device: device.clone(),
            queue: state.queue().clone(),
            pipeline,
            orientations,
            velocities: state.velocity_buffer().clone(),
            layouts: layout_buffer,
            status: state.status_buffer().clone(),
            mass_status: state.mass_status_buffer().clone(),
            timestep,
            slots,
            count: packed.len(),
        })
    }

    /// Packed XYZW unit quaternions, in constructor environment/joint order.
    pub fn orientation_buffer(&self) -> &wgpu::Buffer {
        &self.orientations
    }

    pub(crate) fn velocity_slots(&self) -> &[Vec<usize>] {
        &self.slots
    }

    pub(crate) fn matches_source(&self, state: &GpuGeneralizedStateBatch) -> bool {
        self.velocities == *state.velocity_buffer()
            && self.status == *state.status_buffer()
            && self.mass_status == *state.mass_status_buffer()
    }

    pub(crate) fn orientation_index(&self, environment: usize, slot: usize) -> Option<usize> {
        let local = self
            .slots
            .get(environment)?
            .iter()
            .position(|value| *value == slot)?;
        Some(
            self.slots[..environment]
                .iter()
                .map(Vec::len)
                .sum::<usize>()
                + local,
        )
    }

    /// Record an orientation update using the current generalized angular velocities.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let buffers = [
            &self.orientations,
            &self.velocities,
            &self.layouts,
            &self.status,
            &self.timestep,
            &self.mass_status,
        ];
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera spherical bindings"),
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
            label: Some("Tessera spherical integration"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(self.count.div_ceil(64) as u32, 1, 1);
    }

    /// Replace orientations without changing the layout. Reset source state separately.
    /// Invalid input causes no buffer write. This does not clear source faults.
    pub(crate) fn validate_reset(
        &self,
        joints: &[Vec<GpuSphericalJointState>],
    ) -> Result<(), GpuArticulatedSphericalError> {
        if joints.len() != self.slots.len()
            || joints.iter().zip(&self.slots).any(|(items, slots)| {
                items.len() != slots.len()
                    || items
                        .iter()
                        .zip(slots)
                        .any(|(item, slot)| item.velocity_slot != *slot)
            })
        {
            return Err(GpuArticulatedSphericalError::InvalidInput);
        }
        let _ = pack_orientations(joints)?;
        Ok(())
    }

    /// Validate and replace orientations with matching registration slots.
    pub fn reset(
        &self,
        joints: &[Vec<GpuSphericalJointState>],
    ) -> Result<(), GpuArticulatedSphericalError> {
        self.validate_reset(joints)?;
        let packed = pack_orientations(joints)?;
        self.queue
            .write_buffer(&self.orientations, 0, bytemuck::cast_slice(&packed));
        Ok(())
    }

    /// Read quaternions, rejecting faulted environments and malformed output.
    pub fn readback(&self) -> Result<Vec<Vec<UnitQuaternion<f64>>>, GpuArticulatedSphericalError> {
        let mass_status = read_buffer(&self.device, &self.queue, &self.mass_status)
            .map_err(|e| GpuArticulatedSphericalError::Readback(e.to_string()))?;
        for (environment, bytes) in mass_status.chunks_exact(4).enumerate() {
            if u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) != 0 {
                return Err(GpuArticulatedSphericalError::SourceFault(environment));
            }
        }
        let status = read_buffer(&self.device, &self.queue, &self.status)
            .map_err(|e| GpuArticulatedSphericalError::Readback(e.to_string()))?;
        for (environment, bytes) in status.chunks_exact(4).enumerate() {
            if u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) != 0 {
                return Err(GpuArticulatedSphericalError::SourceFault(environment));
            }
        }
        let bytes = read_buffer(&self.device, &self.queue, &self.orientations)
            .map_err(|e| GpuArticulatedSphericalError::Readback(e.to_string()))?;
        let mut values = bytes.chunks_exact(16);
        self.slots
            .iter()
            .map(|slots| {
                slots
                    .iter()
                    .map(|_| {
                        let bytes = values.next().ok_or_else(|| {
                            GpuArticulatedSphericalError::Readback("missing quaternion".into())
                        })?;
                        let mut c = [0.0; 4];
                        for (value, bytes) in c.iter_mut().zip(bytes.chunks_exact(4)) {
                            *value =
                                f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64;
                        }
                        let q = Quaternion::new(c[3], c[0], c[1], c[2]);
                        if !q.coords.iter().all(|v| v.is_finite())
                            || (q.norm_squared() - 1.0).abs() > 1e-4
                        {
                            return Err(GpuArticulatedSphericalError::Readback(
                                "invalid quaternion".into(),
                            ));
                        }
                        Ok(UnitQuaternion::new_normalize(q))
                    })
                    .collect()
            })
            .collect()
    }
}

fn pack_orientations(
    joints: &[Vec<GpuSphericalJointState>],
) -> Result<Vec<[f32; 4]>, GpuArticulatedSphericalError> {
    joints
        .iter()
        .flatten()
        .map(|joint| {
            let q = joint.orientation.quaternion();
            if !q.coords.iter().all(|v| v.is_finite()) || (q.norm_squared() - 1.0).abs() > 1e-8 {
                return Err(GpuArticulatedSphericalError::InvalidInput);
            }
            Ok([q.i as f32, q.j as f32, q.k as f32, q.w as f32])
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_articulated_mass::{GpuArticulatedMassBatch, GpuArticulatedMassSystem};
    use crate::gpu_articulated_state::GpuGeneralizedState;
    use crate::gpu_contact_pipeline::GpuContactDevice;
    use nalgebra::{DMatrix, DVector, Vector3};

    #[test]
    fn quaternion_joint_state_crosses_euler_singularity_and_resets_faults() {
        let initial = vec![
            GpuGeneralizedState {
                positions: DVector::zeros(7),
                velocities: DVector::from_vec(vec![0.0, 4.0, 0.0, 0.2, -0.3, 0.5, 0.0]),
            },
            GpuGeneralizedState {
                positions: DVector::zeros(1),
                velocities: DVector::zeros(1),
            },
            GpuGeneralizedState {
                positions: DVector::zeros(6),
                velocities: DVector::from_vec(vec![0.0, 0.0, 0.0, 8000.0, 0.0, 0.0]),
            },
        ];
        let joints = vec![
            vec![
                GpuSphericalJointState {
                    velocity_slot: 0,
                    orientation: UnitQuaternion::identity(),
                },
                GpuSphericalJointState {
                    velocity_slot: 3,
                    orientation: UnitQuaternion::from_scaled_axis(Vector3::new(0.3, 0.4, -0.2)),
                },
            ],
            vec![],
            vec![GpuSphericalJointState {
                velocity_slot: 3,
                orientation: UnitQuaternion::from_scaled_axis(Vector3::new(0.0, 0.2, 0.1)),
            }],
        ];
        let systems = initial
            .iter()
            .map(|state| GpuArticulatedMassSystem {
                mass: DMatrix::identity(state.positions.len(), state.positions.len()),
                force: DVector::zeros(state.positions.len()),
            })
            .collect::<Vec<_>>();
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            eprintln!("spherical quaternion state: {backend:?}");
            let mass =
                GpuArticulatedMassBatch::new(context.device(), context.queue(), &systems).unwrap();
            let state = GpuGeneralizedStateBatch::from_mass_batch(&mass, &initial).unwrap();
            assert!(GpuArticulatedSphericalBatch::new(&state, &joints, 0.0).is_err());
            let mut bad = joints.clone();
            bad[0][1].velocity_slot = 2;
            assert!(GpuArticulatedSphericalBatch::new(&state, &bad, 0.001).is_err());
            bad[0][1].velocity_slot = usize::MAX;
            assert!(GpuArticulatedSphericalBatch::new(&state, &bad, 0.001).is_err());
            let batch = GpuArticulatedSphericalBatch::new(&state, &joints, 0.001).unwrap();
            for chunk in 0..10 {
                let mut encoder = context.device().create_command_encoder(&Default::default());
                for _ in 0..100 {
                    batch.encode(&mut encoder);
                }
                let _submission = context.queue().submit([encoder.finish()]);
                let output = batch.readback().unwrap();
                assert!(output[1].is_empty());
                let time = f64::from(chunk + 1) * 0.1;
                for (environment, items) in joints.iter().enumerate() {
                    for (joint, item) in items.iter().enumerate() {
                        let start = item.velocity_slot;
                        let angular = Vector3::new(
                            initial[environment].velocities[start],
                            initial[environment].velocities[start + 1],
                            initial[environment].velocities[start + 2],
                        );
                        let expected =
                            UnitQuaternion::from_scaled_axis(angular * time) * item.orientation;
                        assert!(
                            output[environment][joint].angle_to(&expected) < 2e-3,
                            "{backend:?}, env {environment}, joint {joint}, time {time}"
                        );
                    }
                }
            }
            // Orientation reset must preserve topology and reject all invalid input before writing.
            let saved = batch.readback().unwrap();
            assert!(batch.reset(&bad).is_err());
            assert_eq!(batch.readback().unwrap(), saved);
            bad = joints.clone();
            bad[0][1].orientation =
                UnitQuaternion::new_unchecked(Quaternion::new(f64::NAN, 0.0, 0.0, 0.0));
            assert!(batch.reset(&bad).is_err());
            assert_eq!(batch.readback().unwrap(), saved);
            // A bad angular step faults its source environment until an explicit source reset.
            let mut overflow = initial.clone();
            overflow[0].velocities[0] = 1e30;
            state.reset(&overflow).unwrap();
            let mut encoder = context.device().create_command_encoder(&Default::default());
            batch.encode(&mut encoder);
            let _submission = context.queue().submit([encoder.finish()]);
            assert!(matches!(
                batch.readback(),
                Err(GpuArticulatedSphericalError::SourceFault(0))
            ));
            batch.reset(&joints).unwrap();
            assert!(matches!(
                batch.readback(),
                Err(GpuArticulatedSphericalError::SourceFault(0))
            ));
            state.reset(&initial).unwrap();
            let reset = batch.readback().unwrap();
            for (values, items) in reset.iter().zip(&joints) {
                for (value, item) in values.iter().zip(items) {
                    assert!(value.angle_to(&item.orientation) < 1e-6);
                }
            }
            let mut tiny = initial.clone();
            tiny[0].velocities.fill(0.0);
            tiny[0].velocities[0] = 1e-6;
            state.reset(&tiny).unwrap();
            let mut encoder = context.device().create_command_encoder(&Default::default());
            batch.encode(&mut encoder);
            let _submission = context.queue().submit([encoder.finish()]);
            assert!(
                batch.readback().unwrap()[0][0].angle_to(&UnitQuaternion::from_scaled_axis(
                    Vector3::new(1e-9, 0.0, 0.0)
                )) < 1e-12
            );
            let mut singular = systems.clone();
            singular[0].mass.fill(0.0);
            mass.update(&singular).unwrap();
            mass.submit();
            // Readback must reject a failed mass solve even before its state pass runs.
            assert!(matches!(
                batch.readback(),
                Err(GpuArticulatedSphericalError::SourceFault(0))
            ));
            let mut encoder = context.device().create_command_encoder(&Default::default());
            batch.encode(&mut encoder);
            let _submission = context.queue().submit([encoder.finish()]);
            assert!(matches!(
                batch.readback(),
                Err(GpuArticulatedSphericalError::SourceFault(0))
            ));
            mass.update(&systems).unwrap();
            mass.submit();
            // The orientation pass propagated the source fault into persistent state status.
            assert!(matches!(
                batch.readback(),
                Err(GpuArticulatedSphericalError::SourceFault(0))
            ));
            state.reset(&initial).unwrap();
            batch.reset(&joints).unwrap();
            assert!(batch.readback().is_ok());
        }
        assert!(tested > 0, "no Vulkan or DX12 adapter was available");
    }
}
