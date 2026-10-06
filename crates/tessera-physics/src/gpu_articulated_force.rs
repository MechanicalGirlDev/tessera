//! GPU projection of world gravity through resident link Jacobians.

use core::mem::size_of;

use nalgebra::{DVector, Vector3};
use wgpu::util::DeviceExt;

use crate::gpu_articulated_mass_assembly::GpuArticulatedMassAssemblyBatch;

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct PackedLinkLoad {
    force_scale: [f32; 4],
    torque: [f32; 4],
}

/// A world-frame load about an assembled link's center of mass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuMassLinkLoad {
    /// Force in newtons, in world coordinates.
    pub force: Vector3<f64>,
    /// Torque in newton-metres about the center of mass, in world coordinates.
    pub torque: Vector3<f64>,
    /// Multiplier applied to the link's gravitational force.
    pub gravity_scale: f64,
}

impl Default for GpuMassLinkLoad {
    fn default() -> Self {
        Self {
            force: Vector3::zeros(),
            torque: Vector3::zeros(),
            gravity_scale: 1.0,
        }
    }
}

/// Invalid force input or a GPU capacity limit.
#[derive(Debug, thiserror::Error)]
pub enum GpuArticulatedForceError {
    /// Force or gravity input does not match the bound mass batch.
    #[error("invalid articulated force input")]
    InvalidInput,
    /// Packed force buffers exceed the device limits.
    #[error("articulated forces exceed GPU capacity")]
    Capacity,
}

/// Adds gravity and link wrenches to generalized forces without downloading Jacobians.
///
/// Base forces may contain motor, passive, and velocity-dependent terms.
/// Link loads follow the link order of each mass assembly system. Include every
/// articulation link in that system to project loads on zero-mass frames.
/// Encode link Jacobian generation first, this pass second, then mass
/// assembly. The force result is written directly into the mass assembly's
/// packed vector buffer; its armature half is preserved.
#[derive(Debug)]
pub struct GpuArticulatedForceBatch {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    systems: wgpu::Buffer,
    links: wgpu::Buffer,
    base_forces: wgpu::Buffer,
    gravities: wgpu::Buffer,
    link_loads: wgpu::Buffer,
    vectors: wgpu::Buffer,
    dimensions: Vec<usize>,
    link_counts: Vec<usize>,
}

impl GpuArticulatedForceBatch {
    pub(crate) fn link_load_buffer(&self) -> &wgpu::Buffer {
        &self.link_loads
    }

    pub(crate) fn gravity_buffer(&self) -> &wgpu::Buffer {
        &self.gravities
    }

    /// Bind one base force vector and gravity vector per mass assembly system.
    pub fn new(
        mass: &GpuArticulatedMassAssemblyBatch,
        base_forces: &[DVector<f64>],
        gravities: &[Vector3<f64>],
    ) -> Result<Self, GpuArticulatedForceError> {
        let dimensions = mass.dimensions().to_vec();
        let link_counts = mass.link_counts().to_vec();
        let (forces, gravity) = pack_inputs(&dimensions, base_forces, gravities)?;
        let mut loads = vec![
            PackedLinkLoad {
                force_scale: [0.0, 0.0, 0.0, 1.0],
                torque: [0.0; 4],
            };
            link_counts.iter().sum()
        ];
        if loads.is_empty() {
            loads.push(PackedLinkLoad::default());
        }
        let device = mass.device();
        let limits = device.limits();
        let force_bytes = forces
            .len()
            .checked_mul(4)
            .ok_or(GpuArticulatedForceError::Capacity)? as u64;
        let gravity_bytes = gravity
            .len()
            .checked_mul(16)
            .ok_or(GpuArticulatedForceError::Capacity)? as u64;
        let load_bytes = loads
            .len()
            .checked_mul(size_of::<PackedLinkLoad>())
            .ok_or(GpuArticulatedForceError::Capacity)? as u64;
        if force_bytes
            .checked_mul(2)
            .ok_or(GpuArticulatedForceError::Capacity)?
            != mass.vectors_buffer().size()
            || [force_bytes, gravity_bytes, load_bytes]
                .into_iter()
                .any(|bytes| {
                    bytes > limits.max_buffer_size
                        || bytes > u64::from(limits.max_storage_buffer_binding_size)
                })
            || dimensions.len() > limits.max_compute_workgroups_per_dimension as usize
            || limits.max_storage_buffers_per_shader_stage < 6
        {
            return Err(GpuArticulatedForceError::Capacity);
        }
        let base_forces = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera articulated base forces"),
            contents: bytemuck::cast_slice(&forces),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let gravities = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera articulated world gravities"),
            contents: bytemuck::cast_slice(&gravity),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let link_loads = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera articulated world link loads"),
            contents: bytemuck::cast_slice(&loads),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera articulated gravity projection"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_articulated_force.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tessera articulated gravity projection"),
            layout: None,
            module: &module,
            entry_point: Some("assemble_forces"),
            compilation_options: Default::default(),
            cache: None,
        });
        Ok(Self {
            device: device.clone(),
            queue: mass.queue().clone(),
            pipeline,
            systems: mass.metadata_buffer().clone(),
            links: mass.link_terms_buffer().clone(),
            base_forces,
            gravities,
            link_loads,
            vectors: mass.vectors_buffer().clone(),
            dimensions,
            link_counts,
        })
    }

    /// Replace all base forces and gravities before the next submission.
    pub fn update_inputs(
        &self,
        base_forces: &[DVector<f64>],
        gravities: &[Vector3<f64>],
    ) -> Result<(), GpuArticulatedForceError> {
        let (forces, gravity) = pack_inputs(&self.dimensions, base_forces, gravities)?;
        self.queue
            .write_buffer(&self.base_forces, 0, bytemuck::cast_slice(&forces));
        self.queue
            .write_buffer(&self.gravities, 0, bytemuck::cast_slice(&gravity));
        Ok(())
    }

    /// Replace world-frame loads in assembly link order for every system.
    pub fn set_link_loads(
        &self,
        loads: &[Vec<GpuMassLinkLoad>],
    ) -> Result<(), GpuArticulatedForceError> {
        let packed = pack_link_loads(&self.link_counts, loads)?;
        self.queue
            .write_buffer(&self.link_loads, 0, bytemuck::cast_slice(&packed));
        Ok(())
    }

    /// Encode gravity projection into the bound mass assembly vector buffer.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let buffers = [
            &self.systems,
            &self.links,
            &self.base_forces,
            &self.gravities,
            &self.vectors,
            &self.link_loads,
        ];
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera articulated gravity bindings"),
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
            label: Some("Tessera articulated gravity projection"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(self.dimensions.len() as u32, 1, 1);
    }

    pub(crate) fn dimensions(&self) -> &[usize] {
        &self.dimensions
    }

    pub(crate) fn base_force_buffer(&self) -> &wgpu::Buffer {
        &self.base_forces
    }

    pub(crate) fn vectors_buffer(&self) -> &wgpu::Buffer {
        &self.vectors
    }
}

fn pack_inputs(
    dimensions: &[usize],
    base_forces: &[DVector<f64>],
    gravities: &[Vector3<f64>],
) -> Result<(Vec<f32>, Vec<[f32; 4]>), GpuArticulatedForceError> {
    if dimensions.is_empty()
        || dimensions.len() != base_forces.len()
        || dimensions.len() != gravities.len()
    {
        return Err(GpuArticulatedForceError::InvalidInput);
    }
    let mut forces = Vec::new();
    let mut gravity = Vec::with_capacity(dimensions.len());
    for ((&dimension, force), world_gravity) in dimensions.iter().zip(base_forces).zip(gravities) {
        if dimension != force.len() {
            return Err(GpuArticulatedForceError::InvalidInput);
        }
        for &value in force.iter() {
            forces.push(finite_f32(value)?);
        }
        gravity.push([
            finite_f32(world_gravity.x)?,
            finite_f32(world_gravity.y)?,
            finite_f32(world_gravity.z)?,
            0.0,
        ]);
    }
    Ok((forces, gravity))
}

fn finite_f32(value: f64) -> Result<f32, GpuArticulatedForceError> {
    let narrowed = value as f32;
    if !value.is_finite() || !narrowed.is_finite() || (value != 0.0 && narrowed == 0.0) {
        return Err(GpuArticulatedForceError::InvalidInput);
    }
    Ok(narrowed)
}

fn pack_link_loads(
    link_counts: &[usize],
    loads: &[Vec<GpuMassLinkLoad>],
) -> Result<Vec<PackedLinkLoad>, GpuArticulatedForceError> {
    if link_counts.len() != loads.len() {
        return Err(GpuArticulatedForceError::InvalidInput);
    }
    let mut packed = Vec::new();
    for (&count, system_loads) in link_counts.iter().zip(loads) {
        if count != system_loads.len() {
            return Err(GpuArticulatedForceError::InvalidInput);
        }
        for load in system_loads {
            packed.push(PackedLinkLoad {
                force_scale: [
                    finite_f32(load.force.x)?,
                    finite_f32(load.force.y)?,
                    finite_f32(load.force.z)?,
                    finite_f32(load.gravity_scale)?,
                ],
                torque: [
                    finite_f32(load.torque.x)?,
                    finite_f32(load.torque.y)?,
                    finite_f32(load.torque.z)?,
                    0.0,
                ],
            });
        }
    }
    if packed.is_empty() {
        packed.push(PackedLinkLoad::default());
    }
    Ok(packed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::articulation::{Articulation, JointKind, JointSpec, LinkSpec};
    use crate::gpu_articulated_link_terms::GpuArticulatedLinkTermsBatch;
    use crate::gpu_articulated_mass::read_buffer;
    use crate::gpu_articulated_mass_assembly::{GpuMassAssemblySystem, GpuMassLink};
    use crate::gpu_articulated_pose::GpuArticulatedPoseBatch;
    use crate::gpu_articulated_state::{GpuGeneralizedState, GpuGeneralizedStateBatch};
    use crate::gpu_contact_pipeline::GpuContactDevice;
    use nalgebra::{DMatrix, Isometry3, Matrix3};

    #[test]
    fn packed_gpu_gravity_projects_through_link_jacobians_and_updates() {
        let make_link = |mass, linear_jacobian: DMatrix<f64>| GpuMassLink {
            mass,
            inertia_world: Matrix3::identity(),
            angular_jacobian: DMatrix::zeros(3, linear_jacobian.ncols()),
            linear_jacobian,
        };
        let mut systems = [
            GpuMassAssemblySystem {
                links: vec![make_link(
                    2.0,
                    DMatrix::from_row_slice(3, 1, &[0.0, 0.0, 1.0]),
                )],
                armature: DVector::from_element(1, 1.0),
                force: DVector::zeros(1),
            },
            GpuMassAssemblySystem {
                links: vec![make_link(
                    3.0,
                    DMatrix::from_row_slice(3, 2, &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
                )],
                armature: DVector::from_column_slice(&[0.5, 0.7]),
                force: DVector::zeros(2),
            },
        ];
        systems[1].links[0].angular_jacobian[(2, 1)] = 1.0;
        let bases = [
            DVector::from_element(1, 3.0),
            DVector::from_column_slice(&[1.0, 2.0]),
        ];
        let gravities = [Vector3::new(0.0, 0.0, -9.8), Vector3::new(0.5, -2.0, 0.0)];
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let mass =
                GpuArticulatedMassAssemblyBatch::new(context.device(), context.queue(), &systems)
                    .unwrap();
            let forces = GpuArticulatedForceBatch::new(&mass, &bases, &gravities).unwrap();
            let mut encoder = context.device().create_command_encoder(&Default::default());
            forces.encode(&mut encoder);
            mass.encode(&mut encoder);
            let _ = context.queue().submit(Some(encoder.finish()));
            let bytes =
                read_buffer(context.device(), context.queue(), mass.vectors_buffer()).unwrap();
            let vectors = bytemuck::cast_slice::<u8, f32>(&bytes);
            for (actual, expected) in vectors.iter().zip([1.0, -16.6, 0.5, 0.7, 2.5, -4.0]) {
                assert!(
                    (*actual - expected).abs() < 2e-5,
                    "{backend:?}: {vectors:?}"
                );
            }
            let acceleration = mass.readback().unwrap();
            assert!((acceleration[0][0] + 16.6 / 3.0).abs() < 1e-5);
            assert!((acceleration[1][0] - 2.5 / 3.5).abs() < 1e-5);
            assert!((acceleration[1][1] + 4.0 / 4.7).abs() < 1e-5);

            let changed_bases = [
                DVector::from_element(1, -1.0),
                DVector::from_column_slice(&[2.0, -3.0]),
            ];
            let changed_gravities = [Vector3::new(0.0, 0.0, 2.0), Vector3::new(-1.0, 1.0, 0.0)];
            assert!(
                forces
                    .update_inputs(
                        &changed_bases,
                        &[Vector3::new(f64::NAN, 0.0, 0.0), changed_gravities[1],]
                    )
                    .is_err()
            );
            forces
                .update_inputs(&changed_bases, &changed_gravities)
                .unwrap();
            let mut encoder = context.device().create_command_encoder(&Default::default());
            forces.encode(&mut encoder);
            mass.encode(&mut encoder);
            let _ = context.queue().submit(Some(encoder.finish()));
            let bytes =
                read_buffer(context.device(), context.queue(), mass.vectors_buffer()).unwrap();
            let vectors = bytemuck::cast_slice::<u8, f32>(&bytes);
            for (actual, expected) in vectors.iter().zip([1.0, 3.0, 0.5, 0.7, -1.0, 0.0]) {
                assert!(
                    (*actual - expected).abs() < 2e-5,
                    "{backend:?}: {vectors:?}"
                );
            }

            let loads = [
                vec![GpuMassLinkLoad {
                    force: Vector3::z(),
                    torque: Vector3::zeros(),
                    gravity_scale: 0.5,
                }],
                vec![GpuMassLinkLoad {
                    force: Vector3::x(),
                    torque: Vector3::new(0.0, 0.0, 2.0),
                    gravity_scale: 1.0,
                }],
            ];
            assert!(
                forces
                    .set_link_loads(&[
                        vec![GpuMassLinkLoad {
                            force: Vector3::repeat(f64::NAN),
                            ..loads[0][0]
                        }],
                        loads[1].clone(),
                    ])
                    .is_err()
            );
            forces.set_link_loads(&loads).unwrap();
            let mut encoder = context.device().create_command_encoder(&Default::default());
            forces.encode(&mut encoder);
            mass.encode(&mut encoder);
            let _ = context.queue().submit(Some(encoder.finish()));
            let bytes =
                read_buffer(context.device(), context.queue(), mass.vectors_buffer()).unwrap();
            let vectors = bytemuck::cast_slice::<u8, f32>(&bytes);
            for (actual, expected) in vectors.iter().zip([1.0, 2.0, 0.5, 0.7, 0.0, 2.0]) {
                assert!(
                    (*actual - expected).abs() < 2e-5,
                    "{backend:?}: {vectors:?}"
                );
            }
            let acceleration = mass.readback().unwrap();
            assert!((acceleration[0][0] - 2.0 / 3.0).abs() < 1e-5);
            assert!(acceleration[1][0].abs() < 1e-5);
            assert!((acceleration[1][1] - 2.0 / 4.7).abs() < 1e-5);
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn massless_link_torque_acts_without_adding_inertia() {
        let system = GpuMassAssemblySystem {
            links: vec![GpuMassLink {
                mass: 0.0,
                inertia_world: Matrix3::identity() * 10.0,
                linear_jacobian: DMatrix::zeros(3, 1),
                angular_jacobian: DMatrix::from_row_slice(3, 1, &[0.0, 0.0, 1.0]),
            }],
            armature: DVector::from_element(1, 2.0),
            force: DVector::zeros(1),
        };
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let mass = GpuArticulatedMassAssemblyBatch::new(
                context.device(),
                context.queue(),
                core::slice::from_ref(&system),
            )
            .unwrap();
            let forces =
                GpuArticulatedForceBatch::new(&mass, &[DVector::zeros(1)], &[Vector3::zeros()])
                    .unwrap();
            forces
                .set_link_loads(&[vec![GpuMassLinkLoad {
                    force: Vector3::zeros(),
                    torque: Vector3::new(0.0, 0.0, 3.0),
                    gravity_scale: 1.0,
                }]])
                .unwrap();
            let mut encoder = context.device().create_command_encoder(&Default::default());
            forces.encode(&mut encoder);
            mass.encode(&mut encoder);
            let _ = context.queue().submit(Some(encoder.finish()));
            assert!((mass.readback().unwrap()[0][0] - 1.5).abs() < 1e-5);
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn resident_massless_link_load_step_rebuilds_terms_without_readback() {
        let articulation = Articulation::new(
            vec![
                LinkSpec {
                    mass: 0.0,
                    center_of_mass: Vector3::zeros(),
                    inertia: Matrix3::zeros(),
                },
                LinkSpec {
                    mass: 0.0,
                    center_of_mass: Vector3::zeros(),
                    inertia: Matrix3::zeros(),
                },
                LinkSpec {
                    mass: 2.0,
                    center_of_mass: Vector3::new(0.1, 0.0, 0.0),
                    inertia: Matrix3::identity() * 0.2,
                },
            ],
            vec![
                JointSpec {
                    parent: 0,
                    child: 1,
                    origin: Isometry3::identity(),
                    kind: JointKind::Prismatic,
                    axis: Vector3::z(),
                    limits: None,
                },
                JointSpec {
                    parent: 1,
                    child: 2,
                    origin: Isometry3::identity(),
                    kind: JointKind::Fixed,
                    axis: Vector3::z(),
                    limits: None,
                },
            ],
            0,
        )
        .unwrap();
        let root = Isometry3::identity();
        let initial = GpuGeneralizedState {
            positions: DVector::from_element(1, 1.0),
            velocities: DVector::zeros(1),
        };
        let reference_pose = articulation.pose(root, &[1.0]).unwrap();
        let system = GpuMassAssemblySystem {
            links: (0..articulation.link_count())
                .map(|index| {
                    let link = articulation.link(index).unwrap();
                    let (linear_jacobian, angular_jacobian) = articulation
                        .point_jacobians(&reference_pose, index, link.center_of_mass)
                        .unwrap();
                    GpuMassLink {
                        mass: link.mass,
                        inertia_world: link.inertia,
                        linear_jacobian,
                        angular_jacobian,
                    }
                })
                .collect(),
            armature: DVector::from_element(1, 1.0),
            force: DVector::zeros(1),
        };
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let mass = GpuArticulatedMassAssemblyBatch::new(
                context.device(),
                context.queue(),
                core::slice::from_ref(&system),
            )
            .unwrap();
            let state = GpuGeneralizedStateBatch::from_assembly_batch(
                &mass,
                core::slice::from_ref(&initial),
            )
            .unwrap();
            let pose = GpuArticulatedPoseBatch::new(&state, &[&articulation], &[root]).unwrap();
            let terms = GpuArticulatedLinkTermsBatch::new(&pose, &mass, &[&articulation]).unwrap();
            let forces = GpuArticulatedForceBatch::new(
                &mass,
                &[DVector::zeros(1)],
                &[Vector3::new(0.0, 0.0, -9.0)],
            )
            .unwrap();
            forces
                .set_link_loads(&[vec![
                    GpuMassLinkLoad::default(),
                    GpuMassLinkLoad {
                        force: Vector3::new(0.0, 0.0, 3.0),
                        ..GpuMassLinkLoad::default()
                    },
                    GpuMassLinkLoad {
                        gravity_scale: 0.5,
                        ..GpuMassLinkLoad::default()
                    },
                ]])
                .unwrap();
            let mut encoder = context.device().create_command_encoder(&Default::default());
            for _ in 0..2 {
                pose.encode(&mut encoder);
                terms.encode(&mut encoder);
                forces.encode(&mut encoder);
                mass.encode(&mut encoder);
                state.encode_step(&mut encoder, 0.1).unwrap();
            }
            pose.encode(&mut encoder);
            let _ = context.queue().submit(Some(encoder.finish()));
            let result = state.readback().unwrap();
            assert!((result[0].velocities[0] + 0.4).abs() < 1e-5);
            assert!((result[0].positions[0] - 0.94).abs() < 1e-5);
            let poses = pose.readback().unwrap();
            assert!((poses[0][2].translation.vector.z - 0.94).abs() < 1e-5);
        }
        assert!(tested > 0, "no GPU backend available");
    }
}
