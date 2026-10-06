//! Quaternion spherical drives assembled directly from resident state.

use crate::gpu_articulated_force::GpuArticulatedForceBatch;
use crate::gpu_articulated_spherical::GpuArticulatedSphericalBatch;
use crate::gpu_articulated_state::GpuGeneralizedStateBatch;
use crate::spherical_drive::SphericalJointDrive;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PackedDrive {
    indices: [u32; 4],
    target: [f32; 4],
    velocity: [f32; 4],
    stiffness: [f32; 4],
    damping: [f32; 4],
    cap: [f32; 4],
}

/// Invalid spherical drive layout or exhausted device capacity.
#[derive(Debug, thiserror::Error)]
pub enum GpuSphericalDriveError {
    /// State, orientation, force, or drive parameters do not match.
    #[error("invalid GPU spherical drive input")]
    InvalidInput,
    /// The packed drive exceeds device limits.
    #[error("GPU spherical drive exceeds device capacity")]
    Capacity,
}

/// Adds quaternion PD torque to the three angular velocity force slots.
///
/// Encode after scalar/base joint force assembly and before velocity bias and
/// gravity projection. Scalar drive inputs on these spherical slots must contain
/// baseline efforts only: Euler position springs cannot be used here. This pass
/// is explicit; it does not add implicit stiffness to the mass diagonal.
#[derive(Debug)]
pub struct GpuArticulatedSphericalDriveBatch {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    parameters: wgpu::Buffer,
    orientations: wgpu::Buffer,
    velocities: wgpu::Buffer,
    status: wgpu::Buffer,
    mass_status: wgpu::Buffer,
    output: wgpu::Buffer,
    layouts: Vec<Vec<[u32; 4]>>,
    count: usize,
}

impl GpuArticulatedSphericalDriveBatch {
    /// Bind one optional drive per quaternion in constructor environment/joint order.
    pub fn new(
        state: &GpuGeneralizedStateBatch,
        spherical: &GpuArticulatedSphericalBatch,
        forces: &GpuArticulatedForceBatch,
        drives: &[Vec<Option<SphericalJointDrive>>],
    ) -> Result<Self, GpuSphericalDriveError> {
        if !spherical.matches_source(state)
            || forces.dimensions().len() != state.ranges().len()
            || forces
                .dimensions()
                .iter()
                .zip(state.ranges())
                .any(|(&n, r)| n != r.len())
        {
            return Err(GpuSphericalDriveError::InvalidInput);
        }
        let layouts = spherical
            .velocity_slots()
            .iter()
            .zip(state.ranges())
            .enumerate()
            .map(|(environment, (slots, range))| {
                slots
                    .iter()
                    .map(|&slot| {
                        Ok([
                            u32::try_from(range.start + slot)
                                .map_err(|_| GpuSphericalDriveError::Capacity)?,
                            u32::try_from(environment)
                                .map_err(|_| GpuSphericalDriveError::Capacity)?,
                            u32::try_from(
                                spherical
                                    .orientation_index(environment, slot)
                                    .ok_or(GpuSphericalDriveError::InvalidInput)?,
                            )
                            .map_err(|_| GpuSphericalDriveError::Capacity)?,
                            0,
                        ])
                    })
                    .collect::<Result<Vec<_>, GpuSphericalDriveError>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let packed = pack(&layouts, drives)?;
        let bytes = size_of_val(packed.as_slice()) as u64;
        let limits = state.device().limits();
        if packed.is_empty()
            || bytes > limits.max_buffer_size
            || bytes > u64::from(limits.max_storage_buffer_binding_size)
            || packed.len().div_ceil(64) > limits.max_compute_workgroups_per_dimension as usize
            || limits.max_storage_buffers_per_shader_stage < 6
        {
            return Err(GpuSphericalDriveError::Capacity);
        }
        let parameters = state
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tessera spherical drive parameters"),
                contents: bytemuck::cast_slice(&packed),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        let shader = state
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Tessera spherical drive"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("gpu_articulated_spherical_drive.wgsl").into(),
                ),
            });
        let pipeline = state
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Tessera spherical drive"),
                layout: None,
                module: &shader,
                entry_point: Some("assemble_spherical_drives"),
                compilation_options: Default::default(),
                cache: None,
            });
        Ok(Self {
            device: state.device().clone(),
            queue: state.queue().clone(),
            pipeline,
            parameters,
            orientations: spherical.orientation_buffer().clone(),
            velocities: state.velocity_buffer().clone(),
            status: state.status_buffer().clone(),
            mass_status: state.mass_status_buffer().clone(),
            output: forces.base_force_buffer().clone(),
            layouts,
            count: packed.len(),
        })
    }

    /// Replace targets and gains only after validating every environment.
    pub fn update(
        &self,
        drives: &[Vec<Option<SphericalJointDrive>>],
    ) -> Result<(), GpuSphericalDriveError> {
        let packed = pack(&self.layouts, drives)?;
        self.queue
            .write_buffer(&self.parameters, 0, bytemuck::cast_slice(&packed));
        Ok(())
    }

    /// Add drive torque without downloading the resident orientation or velocity.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let buffers = [
            &self.parameters,
            &self.orientations,
            &self.velocities,
            &self.status,
            &self.mass_status,
            &self.output,
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
            label: Some("Tessera spherical drive bindings"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Tessera spherical drive assembly"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(self.count.div_ceil(64) as u32, 1, 1);
    }
}

fn narrow(value: f64) -> Result<f32, GpuSphericalDriveError> {
    let result = value as f32;
    if !value.is_finite() || !result.is_finite() || (value != 0.0 && result == 0.0) {
        return Err(GpuSphericalDriveError::InvalidInput);
    }
    Ok(result)
}

fn pack(
    layouts: &[Vec<[u32; 4]>],
    drives: &[Vec<Option<SphericalJointDrive>>],
) -> Result<Vec<PackedDrive>, GpuSphericalDriveError> {
    if layouts.len() != drives.len() || layouts.iter().zip(drives).any(|(l, d)| l.len() != d.len())
    {
        return Err(GpuSphericalDriveError::InvalidInput);
    }
    let mut result = Vec::new();
    for (layout, drive) in layouts.iter().flatten().zip(drives.iter().flatten()) {
        let mut entry = PackedDrive {
            indices: *layout,
            target: [0.0, 0.0, 0.0, 1.0],
            velocity: [0.0; 4],
            stiffness: [0.0; 4],
            damping: [0.0; 4],
            cap: [0.0; 4],
        };
        if let Some(drive) = drive {
            drive
                .validate()
                .map_err(|_| GpuSphericalDriveError::InvalidInput)?;
            for i in 0..4 {
                entry.target[i] = narrow(drive.orientation_target.quaternion().coords[i])?;
            }
            for i in 0..3 {
                entry.velocity[i] = narrow(drive.velocity_target[i])?;
                entry.stiffness[i] = narrow(drive.stiffness[i])?;
                entry.damping[i] = narrow(drive.damping[i])?;
                entry.cap[i] = narrow(drive.max_torque[i])?;
            }
            entry.indices[3] = 1;
        }
        result.push(entry);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_articulated_mass::read_buffer;
    use crate::gpu_articulated_mass_assembly::{
        GpuArticulatedMassAssemblyBatch, GpuMassAssemblySystem,
    };
    use crate::gpu_articulated_spherical::GpuSphericalJointState;
    use crate::gpu_articulated_state::GpuGeneralizedState;
    use crate::gpu_contact_pipeline::GpuContactDevice;
    use nalgebra::{DVector, UnitQuaternion, Vector3};

    #[test]
    fn packed_quaternion_drives_match_cpu_and_validate_updates() {
        let q = UnitQuaternion::from_euler_angles(0.3, core::f64::consts::FRAC_PI_2, -0.5);
        let drive = SphericalJointDrive {
            orientation_target: UnitQuaternion::from_scaled_axis(Vector3::new(0.2, -0.3, 0.1)) * q,
            velocity_target: Vector3::new(-0.2, 0.1, 0.3),
            stiffness: Vector3::new(10.0, 20.0, 30.0),
            damping: Vector3::new(2.0, 3.0, 4.0),
            max_torque: Vector3::new(100.0, 4.0, 100.0),
        };
        let omega = Vector3::new(0.4, -0.2, 0.7);
        let states = [
            GpuGeneralizedState {
                positions: DVector::from_element(7, 123.0),
                velocities: DVector::from_column_slice(&[0.0, 0.4, -0.2, 0.7, 0.4, -0.2, 0.7]),
            },
            GpuGeneralizedState {
                positions: DVector::from_element(9, 456.0),
                velocities: DVector::from_column_slice(&[
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, -0.2, 0.7,
                ]),
            },
        ];
        let systems = [7, 9].map(|n| GpuMassAssemblySystem {
            links: Vec::new(),
            armature: DVector::from_element(n, 1.0),
            force: DVector::zeros(n),
        });
        let baseline = [
            DVector::from_element(7, 0.25),
            DVector::from_element(9, -0.5),
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
            let spherical = GpuArticulatedSphericalBatch::new(
                &state,
                &[
                    vec![
                        GpuSphericalJointState {
                            velocity_slot: 4,
                            orientation: q,
                        },
                        GpuSphericalJointState {
                            velocity_slot: 1,
                            orientation: q,
                        },
                    ],
                    vec![GpuSphericalJointState {
                        velocity_slot: 6,
                        orientation: UnitQuaternion::new_unchecked(-q.into_inner()),
                    }],
                ],
                0.001,
            )
            .unwrap();
            let forces =
                GpuArticulatedForceBatch::new(&mass, &baseline, &[Vector3::zeros(); 2]).unwrap();
            let drives = [vec![None, Some(drive)], vec![Some(drive)]];
            let batch =
                GpuArticulatedSphericalDriveBatch::new(&state, &spherical, &forces, &drives)
                    .unwrap();
            let mut invalid = drive;
            invalid.damping.x = f64::NAN;
            assert!(
                batch
                    .update(&[vec![None, Some(drive)], vec![Some(invalid)]])
                    .is_err()
            );
            assert!(
                batch
                    .update(&[vec![Some(drive)], vec![Some(drive)]])
                    .is_err()
            );
            let before = read_buffer(
                context.device(),
                context.queue(),
                spherical.orientation_buffer(),
            )
            .unwrap();
            let mut encoder = context.device().create_command_encoder(&Default::default());
            batch.encode(&mut encoder);
            let _ = context.queue().submit(Some(encoder.finish()));
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
            let torque = drive.torque(q, omega).unwrap();
            for i in 0..16 {
                let expected = if (1..4).contains(&i) {
                    0.25 + torque[i - 1]
                } else if (13..16).contains(&i) {
                    -0.5 + torque[i - 13]
                } else if i < 7 {
                    0.25
                } else {
                    -0.5
                };
                assert!(
                    (actual[i] - expected).abs() < 2e-5,
                    "{backend:?} slot {i}: {} vs {expected}",
                    actual[i]
                );
            }
            assert_eq!(
                before,
                read_buffer(
                    context.device(),
                    context.queue(),
                    spherical.orientation_buffer()
                )
                .unwrap()
            );
            let actual_state = state.readback().unwrap();
            for (a, b) in actual_state.iter().zip(&states) {
                assert_eq!(a.positions, b.positions);
                assert!((&a.velocities - &b.velocities).norm() < 1e-6);
            }
            batch.update(&[vec![None, None], vec![None]]).unwrap();
            forces
                .update_inputs(&baseline, &[Vector3::zeros(); 2])
                .unwrap();
            let mut encoder = context.device().create_command_encoder(&Default::default());
            batch.encode(&mut encoder);
            let _ = context.queue().submit(Some(encoder.finish()));
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
            assert_eq!(
                actual,
                baseline
                    .iter()
                    .flat_map(|b| b.iter().copied())
                    .collect::<Vec<_>>()
            );
        }
        assert!(tested > 0, "no GPU backend available");
    }
}
