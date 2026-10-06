//! Device-resident link inertia and Jacobians for articulated trees.

use core::mem::size_of;

use wgpu::util::DeviceExt;

use crate::articulation::Articulation;
use crate::gpu_articulated_mass_assembly::GpuArticulatedMassAssemblyBatch;
use crate::gpu_articulated_pose::GpuArticulatedPoseBatch;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LinkMeta {
    indices: [u32; 4],
    center_of_mass: [f32; 4],
    inertia_rows: [[f32; 4]; 3],
    mass: [f32; 4],
}

/// Invalid topology, incompatible mass layout, or GPU capacity exhaustion.
#[derive(Debug, thiserror::Error)]
pub enum GpuArticulatedLinkTermsError {
    /// The pose and mass batches do not describe the same articulated trees.
    #[error("incompatible articulated link term layout")]
    InvalidInput,
    /// The packed link terms exceed GPU capacity.
    #[error("articulated link terms exceed GPU capacity")]
    Capacity,
}

/// Rebuilds world inertia and center-of-mass Jacobians from resident link poses.
///
/// The mass assembly batch must contain either all links or all positive-mass
/// links in articulation order, with the same fixed/floating layout as the poses. Include massless
/// links when their Jacobians are needed for external load projection.
/// Encode poses first, link terms second,
/// and mass assembly last in the same command encoder. Recreate this batch when
/// topology, link inertial properties, or mass assembly layout changes.
#[derive(Debug)]
pub struct GpuArticulatedLinkTermsBatch {
    device: wgpu::Device,
    pipeline: wgpu::ComputePipeline,
    metadata: wgpu::Buffer,
    poses: wgpu::Buffer,
    joints: wgpu::Buffer,
    environments: wgpu::Buffer,
    coordinates: wgpu::Buffer,
    status: wgpu::Buffer,
    output: wgpu::Buffer,
    link_count: usize,
}

impl GpuArticulatedLinkTermsBatch {
    /// Bind packed articulations to an existing mass assembly batch.
    pub fn new(
        poses: &GpuArticulatedPoseBatch,
        mass: &GpuArticulatedMassAssemblyBatch,
        articulations: &[&Articulation],
    ) -> Result<Self, GpuArticulatedLinkTermsError> {
        if articulations.is_empty()
            || articulations.len() != poses.link_ranges().len()
            || articulations.len() != mass.dimensions().len()
        {
            return Err(GpuArticulatedLinkTermsError::InvalidInput);
        }
        let mut metadata = Vec::new();
        let mut joint_offset = 0usize;
        let mut output_offset = 0usize;
        for (environment, ((articulation, link_range), (&dimension, &mass_link_count))) in
            articulations
                .iter()
                .zip(poses.link_ranges())
                .zip(mass.dimensions().iter().zip(mass.link_counts()))
                .enumerate()
        {
            let root_dofs = if poses.floating_roots()[environment] {
                6
            } else {
                0
            };
            if articulation.dof().checked_add(root_dofs) != Some(dimension)
                || link_range.len() != articulation.link_count()
            {
                return Err(GpuArticulatedLinkTermsError::InvalidInput);
            }
            let (_, traversal, joints) = articulation.gpu_pose_topology();
            let positive_count = (0..articulation.link_count())
                .filter(|&index| articulation.link(index).is_some_and(|link| link.mass > 0.0))
                .count();
            let include_all_links = mass_link_count == articulation.link_count();
            if !include_all_links && mass_link_count != positive_count {
                return Err(GpuArticulatedLinkTermsError::InvalidInput);
            }
            let mut incoming = vec![u32::MAX; articulation.link_count()];
            for (index, &edge) in traversal.iter().enumerate() {
                incoming[joints[edge].child] = checked_u32(joint_offset + index)?;
            }
            let stride = 10usize
                .checked_add(
                    dimension
                        .checked_mul(6)
                        .ok_or(GpuArticulatedLinkTermsError::Capacity)?,
                )
                .ok_or(GpuArticulatedLinkTermsError::Capacity)?;
            let mut counted = 0usize;
            for (link_index, &incoming_joint) in incoming.iter().enumerate() {
                let link = articulation
                    .link(link_index)
                    .ok_or(GpuArticulatedLinkTermsError::InvalidInput)?;
                let output = if include_all_links || link.mass > 0.0 {
                    counted += 1;
                    let offset = checked_u32(output_offset)?;
                    output_offset = output_offset
                        .checked_add(stride)
                        .ok_or(GpuArticulatedLinkTermsError::Capacity)?;
                    offset
                } else {
                    u32::MAX
                };
                let mut inertia_rows = [[0.0; 4]; 3];
                for (row, entries) in inertia_rows.iter_mut().enumerate() {
                    for (col, entry) in entries.iter_mut().take(3).enumerate() {
                        *entry = finite_f32(link.inertia[(row, col)])?;
                    }
                }
                metadata.push(LinkMeta {
                    indices: [
                        checked_u32(environment)?,
                        checked_u32(link_index)?,
                        incoming_joint,
                        output,
                    ],
                    center_of_mass: [
                        finite_f32(link.center_of_mass.x)?,
                        finite_f32(link.center_of_mass.y)?,
                        finite_f32(link.center_of_mass.z)?,
                        0.0,
                    ],
                    inertia_rows,
                    mass: [finite_f32(link.mass)?, 0.0, 0.0, 0.0],
                });
            }
            if counted != mass_link_count {
                return Err(GpuArticulatedLinkTermsError::InvalidInput);
            }
            joint_offset = joint_offset
                .checked_add(traversal.len())
                .ok_or(GpuArticulatedLinkTermsError::Capacity)?;
        }
        let device = poses.device();
        let bytes = metadata
            .len()
            .checked_mul(size_of::<LinkMeta>())
            .ok_or(GpuArticulatedLinkTermsError::Capacity)? as u64;
        let limits = device.limits();
        let output_bytes = output_offset
            .checked_mul(4)
            .ok_or(GpuArticulatedLinkTermsError::Capacity)? as u64;
        if output_bytes.max(4) != mass.link_terms_buffer().size()
            || bytes > limits.max_buffer_size
            || bytes > u64::from(limits.max_storage_buffer_binding_size)
            || metadata.len().div_ceil(64) > limits.max_compute_workgroups_per_dimension as usize
            || limits.max_storage_buffers_per_shader_stage < 7
        {
            return Err(GpuArticulatedLinkTermsError::Capacity);
        }
        let metadata = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera articulated link inertial metadata"),
            contents: bytemuck::cast_slice(&metadata),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera articulated link Jacobians"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("gpu_articulated_link_terms.wgsl").into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tessera articulated link Jacobians"),
            layout: None,
            module: &module,
            entry_point: Some("build_link_terms"),
            compilation_options: Default::default(),
            cache: None,
        });
        Ok(Self {
            device: device.clone(),
            pipeline,
            metadata,
            poses: poses.link_pose_buffer().clone(),
            joints: poses.joint_buffer().clone(),
            environments: poses.environment_buffer().clone(),
            coordinates: poses.coordinate_buffer().clone(),
            status: poses.status_buffer().clone(),
            output: mass.link_terms_buffer().clone(),
            link_count: poses.link_ranges().last().map_or(0, |range| range.end),
        })
    }

    /// Write current link terms directly into the bound mass assembly buffer.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let buffers = [
            &self.metadata,
            &self.poses,
            &self.joints,
            &self.environments,
            &self.coordinates,
            &self.status,
            &self.output,
        ];
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera articulated link Jacobian bindings"),
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
            label: Some("Tessera articulated link Jacobians"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(self.link_count.div_ceil(64) as u32, 1, 1);
    }
}

fn checked_u32(value: usize) -> Result<u32, GpuArticulatedLinkTermsError> {
    u32::try_from(value).map_err(|_| GpuArticulatedLinkTermsError::Capacity)
}

fn finite_f32(value: f64) -> Result<f32, GpuArticulatedLinkTermsError> {
    let narrowed = value as f32;
    if !value.is_finite() || !narrowed.is_finite() || (value != 0.0 && narrowed == 0.0) {
        return Err(GpuArticulatedLinkTermsError::InvalidInput);
    }
    Ok(narrowed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::articulation::{JointKind, JointSpec, LinkSpec};
    use crate::gpu_articulated_mass::{
        GpuArticulatedMassBatch, GpuArticulatedMassSystem, read_buffer,
    };
    use crate::gpu_articulated_mass_assembly::{GpuMassAssemblySystem, GpuMassLink};
    use crate::gpu_articulated_state::{GpuGeneralizedState, GpuGeneralizedStateBatch};
    use crate::gpu_contact_pipeline::GpuContactDevice;
    use nalgebra::{DMatrix, DVector, Isometry3, Matrix3, Vector3};

    fn reference_system(
        articulation: &Articulation,
        root: Isometry3<f64>,
        q: &[f64],
    ) -> GpuMassAssemblySystem {
        let pose = articulation.pose(root, q).unwrap();
        let links = (0..articulation.link_count())
            .filter_map(|index| {
                let link = articulation.link(index).unwrap();
                if link.mass == 0.0 {
                    return None;
                }
                let (linear_jacobian, angular_jacobian) = articulation
                    .point_jacobians(&pose, index, link.center_of_mass)
                    .unwrap();
                let rotation = pose.links[index].rotation.to_rotation_matrix();
                Some(GpuMassLink {
                    mass: link.mass,
                    inertia_world: rotation.matrix() * link.inertia * rotation.matrix().transpose(),
                    linear_jacobian,
                    angular_jacobian,
                })
            })
            .collect();
        GpuMassAssemblySystem {
            links,
            armature: DVector::from_element(articulation.dof(), 0.2),
            force: DVector::from_element(articulation.dof(), 1.0),
        }
    }

    fn check_terms(bytes: &[u8], reference: &GpuMassAssemblySystem) {
        let floats = bytemuck::cast_slice::<u8, f32>(bytes);
        let n = reference.force.len();
        for (index, link) in reference.links.iter().enumerate() {
            let base = index * (10 + 6 * n);
            assert!((f64::from(floats[base]) - link.mass).abs() < 1e-5);
            for row in 0..3 {
                for col in 0..3 {
                    assert!(
                        (f64::from(floats[base + 1 + row * 3 + col])
                            - link.inertia_world[(row, col)])
                            .abs()
                            < 2e-5
                    );
                }
                for col in 0..n {
                    assert!(
                        (f64::from(floats[base + 10 + row * n + col])
                            - link.linear_jacobian[(row, col)])
                            .abs()
                            < 2e-5
                    );
                    assert!(
                        (f64::from(floats[base + 10 + 3 * n + row * n + col])
                            - link.angular_jacobian[(row, col)])
                            .abs()
                            < 2e-5
                    );
                }
            }
        }
    }

    #[test]
    fn quaternion_fk_tangent_terms_and_mass_inverse_match_cpu_at_gimbal_lock() {
        use crate::gpu_articulated_force::GpuArticulatedForceBatch;
        use crate::gpu_articulated_spherical::{
            GpuArticulatedSphericalBatch, GpuSphericalJointState,
        };
        use crate::gpu_articulated_velocity_bias::GpuArticulatedVelocityBiasBatch;
        use nalgebra::UnitQuaternion;
        let links = (0..5)
            .map(|index| LinkSpec {
                mass: 1.0 + f64::from(index),
                center_of_mass: Vector3::new(0.1, -0.05, 0.15),
                inertia: Matrix3::from_diagonal(&Vector3::new(0.2, 0.4, 0.6)),
            })
            .collect();
        let joint = |parent, child, kind, origin| JointSpec {
            parent,
            child,
            kind,
            origin,
            axis: Vector3::z(),
            limits: None,
        };
        let articulation = Articulation::new(
            links,
            vec![
                joint(
                    3,
                    1,
                    JointKind::Spherical,
                    Isometry3::translation(0.3, 0.1, 0.2)
                        * Isometry3::rotation(Vector3::new(0.2, -0.1, 0.3)),
                ),
                joint(
                    1,
                    0,
                    JointKind::Revolute,
                    Isometry3::translation(0.4, 0.2, 0.1),
                ),
                joint(
                    3,
                    2,
                    JointKind::Spherical,
                    Isometry3::translation(-0.5, 0.1, 0.3),
                ),
                joint(
                    2,
                    4,
                    JointKind::Fixed,
                    Isometry3::translation(0.1, -0.2, 0.4),
                ),
            ],
            3,
        )
        .unwrap();
        let roots = [
            Isometry3::translation(1.0, 2.0, 3.0)
                * Isometry3::rotation(Vector3::new(0.3, -0.2, 0.1)),
            Isometry3::translation(-1.0, 0.5, 2.0),
        ];
        let flags = [false, true];
        let initial_rotations = [
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), core::f64::consts::FRAC_PI_2),
            UnitQuaternion::from_scaled_axis(Vector3::new(0.2, -0.3, 0.4)),
        ];
        let rates = [Vector3::new(0.0, 4.0, 0.0), Vector3::new(0.3, -0.2, 0.4)];
        let initial = flags
            .iter()
            .map(|&floating| {
                let offset = if floating { 6 } else { 0 };
                let mut positions = DVector::from_element(offset + 7, 123.0);
                positions[offset + 3] = 0.4;
                let mut velocities = DVector::zeros(offset + 7);
                for (start, rate) in [(offset, rates[0]), (offset + 4, rates[1])] {
                    for axis in 0..3 {
                        velocities[start + axis] = rate[axis];
                    }
                }
                GpuGeneralizedState {
                    positions,
                    velocities,
                }
            })
            .collect::<Vec<_>>();
        let reference = |environment: usize, time: f64| {
            let offset = if flags[environment] { 6 } else { 0 };
            let rotations = [
                Some(UnitQuaternion::from_scaled_axis(rates[0] * time) * initial_rotations[0]),
                None,
                Some(UnitQuaternion::from_scaled_axis(rates[1] * time) * initial_rotations[1]),
                None,
            ];
            let pose = articulation
                .pose_with_spherical_orientations(
                    roots[environment],
                    &initial[environment].positions.as_slice()[offset..],
                    &rotations,
                )
                .unwrap();
            let links = (0..articulation.link_count())
                .map(|index| {
                    let link = articulation.link(index).unwrap();
                    let (linear_jacobian, angular_jacobian) = articulation
                        .generalized_point_jacobians(
                            &pose,
                            index,
                            link.center_of_mass,
                            flags[environment],
                        )
                        .unwrap();
                    let rotation = pose.links[index].rotation.to_rotation_matrix();
                    GpuMassLink {
                        mass: link.mass,
                        inertia_world: rotation.matrix()
                            * link.inertia
                            * rotation.matrix().transpose(),
                        linear_jacobian,
                        angular_jacobian,
                    }
                })
                .collect();
            (
                pose,
                GpuMassAssemblySystem {
                    links,
                    armature: DVector::zeros(offset + 7),
                    force: DVector::from_element(offset + 7, 0.2),
                },
            )
        };
        let systems = (0..2)
            .map(|environment| reference(environment, 0.0).1)
            .collect::<Vec<_>>();
        let joint_states = vec![
            vec![
                GpuSphericalJointState {
                    velocity_slot: 4,
                    orientation: initial_rotations[1],
                },
                GpuSphericalJointState {
                    velocity_slot: 0,
                    orientation: initial_rotations[0],
                },
            ],
            vec![
                GpuSphericalJointState {
                    velocity_slot: 6,
                    orientation: initial_rotations[0],
                },
                GpuSphericalJointState {
                    velocity_slot: 10,
                    orientation: initial_rotations[1],
                },
            ],
        ];
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            eprintln!("quaternion FK and tangent mass: {backend:?}");
            let mass =
                GpuArticulatedMassAssemblyBatch::new(context.device(), context.queue(), &systems)
                    .unwrap();
            let state = GpuGeneralizedStateBatch::from_assembly_batch(&mass, &initial).unwrap();
            let quaternion =
                GpuArticulatedSphericalBatch::new(&state, &joint_states, 0.001).unwrap();
            let arts = [&articulation, &articulation];
            let pose = GpuArticulatedPoseBatch::new_with_spherical_state(
                &state,
                &arts,
                &roots,
                &flags,
                &quaternion,
            )
            .unwrap();
            assert!(pose.uses_spherical_quaternions());
            let forces = GpuArticulatedForceBatch::new(
                &mass,
                &systems
                    .iter()
                    .map(|system| system.force.clone())
                    .collect::<Vec<_>>(),
                &[Vector3::zeros(); 2],
            )
            .unwrap();
            assert!(
                GpuArticulatedVelocityBiasBatch::new_with_floating_roots(
                    &state, &mass, &forces, &arts, &pose
                )
                .is_err()
            );
            let other_state =
                GpuGeneralizedStateBatch::from_assembly_batch(&mass, &initial).unwrap();
            assert!(
                GpuArticulatedPoseBatch::new_with_spherical_state(
                    &other_state,
                    &arts,
                    &roots,
                    &flags,
                    &quaternion
                )
                .is_err()
            );
            let mut incomplete = joint_states.clone();
            let _removed = incomplete[0].pop();
            let incomplete = GpuArticulatedSphericalBatch::new(&state, &incomplete, 0.001).unwrap();
            assert!(
                GpuArticulatedPoseBatch::new_with_spherical_state(
                    &state,
                    &arts,
                    &roots,
                    &flags,
                    &incomplete
                )
                .is_err()
            );
            let terms = GpuArticulatedLinkTermsBatch::new(&pose, &mass, &arts).unwrap();
            for steps in [0, 100] {
                let mut encoder = context.device().create_command_encoder(&Default::default());
                for _ in 0..steps {
                    quaternion.encode(&mut encoder);
                }
                pose.encode(&mut encoder);
                terms.encode(&mut encoder);
                mass.encode_with_inverse(&mut encoder);
                let _submission = context.queue().submit([encoder.finish()]);
                let actual_poses = pose.readback().unwrap();
                let actual_acceleration = mass.readback().unwrap();
                let actual_inverse = mass.readback_inverse().unwrap();
                let bytes =
                    read_buffer(context.device(), context.queue(), mass.link_terms_buffer())
                        .unwrap();
                let mut byte_offset = 0;
                for environment in 0..2 {
                    let (expected_pose, expected) =
                        reference(environment, f64::from(steps) * 0.001);
                    for (link, (actual, expected)) in actual_poses[environment]
                        .iter()
                        .zip(&expected_pose.links)
                        .enumerate()
                    {
                        assert!(
                            (actual.translation.vector - expected.translation.vector).norm() < 1e-5
                        );
                        assert!(
                            actual.rotation.angle_to(&expected.rotation) < 1e-5,
                            "{backend:?}, steps {steps}, env {environment}, link {link}, angle {}, actual {:?}, expected {:?}",
                            actual.rotation.angle_to(&expected.rotation),
                            actual.rotation,
                            expected.rotation
                        );
                    }
                    let n = expected.force.len();
                    let length = expected.links.len() * (10 + 6 * n) * 4;
                    check_terms(&bytes[byte_offset..byte_offset + length], &expected);
                    byte_offset += length;
                    let mut matrix = DMatrix::zeros(n, n);
                    for link in &expected.links {
                        matrix +=
                            link.linear_jacobian.transpose() * &link.linear_jacobian * link.mass
                                + link.angular_jacobian.transpose()
                                    * link.inertia_world
                                    * &link.angular_jacobian;
                    }
                    let cholesky = matrix.cholesky().unwrap();
                    assert!(
                        (&actual_acceleration[environment] - cholesky.solve(&expected.force))
                            .norm()
                            < 2e-4
                    );
                    assert!((&actual_inverse[environment] - cholesky.inverse()).norm() < 2e-3);
                }
            }
        }
        assert!(tested > 0, "no Vulkan or DX12 adapter was available");
    }

    #[test]
    fn resident_link_terms_match_cpu_after_coordinate_change() {
        let links = (0..5)
            .map(|index| LinkSpec {
                mass: if index == 2 { 0.0 } else { 1.0 + index as f64 },
                center_of_mass: Vector3::new(0.1 * index as f64, -0.05, 0.15),
                inertia: Matrix3::from_diagonal(&Vector3::new(0.2 + index as f64 * 0.1, 0.4, 0.6)),
            })
            .collect();
        let joint = |parent, child, kind, origin| JointSpec {
            parent,
            child,
            kind,
            origin,
            axis: Vector3::z(),
            limits: None,
        };
        let mut articulation = Articulation::new(
            links,
            vec![
                joint(
                    0,
                    1,
                    JointKind::Revolute,
                    Isometry3::translation(1.0, 0.0, 0.0),
                ),
                joint(
                    1,
                    2,
                    JointKind::Prismatic,
                    Isometry3::translation(0.0, 1.0, 0.0),
                ),
                joint(
                    2,
                    3,
                    JointKind::Spherical,
                    Isometry3::translation(0.0, 0.0, 1.0),
                ),
                joint(
                    0,
                    4,
                    JointKind::Revolute,
                    Isometry3::translation(-1.0, 0.0, 0.0),
                ),
            ],
            0,
        )
        .unwrap();
        articulation.set_mimics(&[(3, 0, -1.5, 0.2)]).unwrap();
        let root = Isometry3::translation(1.0, 2.0, -0.5)
            * Isometry3::rotation(Vector3::new(0.1, -0.2, 0.3));
        let q = [0.3, 0.2, -0.1, 0.4, -0.2];
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let direct = GpuArticulatedMassBatch::new(
                context.device(),
                context.queue(),
                &[GpuArticulatedMassSystem {
                    mass: DMatrix::identity(q.len(), q.len()),
                    force: DVector::zeros(q.len()),
                }],
            )
            .unwrap();
            let state = GpuGeneralizedStateBatch::from_mass_batch(
                &direct,
                &[GpuGeneralizedState {
                    positions: DVector::from_column_slice(&q),
                    velocities: DVector::zeros(q.len()),
                }],
            )
            .unwrap();
            let pose = GpuArticulatedPoseBatch::new(&state, &[&articulation], &[root]).unwrap();
            let reference = reference_system(&articulation, root, &q);
            let mass = GpuArticulatedMassAssemblyBatch::new(
                context.device(),
                context.queue(),
                core::slice::from_ref(&reference),
            )
            .unwrap();
            let terms = GpuArticulatedLinkTermsBatch::new(&pose, &mass, &[&articulation]).unwrap();
            let mut encoder = context.device().create_command_encoder(&Default::default());
            pose.encode(&mut encoder);
            terms.encode(&mut encoder);
            mass.encode(&mut encoder);
            let _ = context.queue().submit(Some(encoder.finish()));
            let bytes =
                read_buffer(context.device(), context.queue(), mass.link_terms_buffer()).unwrap();
            check_terms(&bytes, &reference);
            let expected = {
                let pose = articulation.pose(root, &q).unwrap();
                let dynamics = articulation.dynamics(&pose, Vector3::zeros()).unwrap();
                (dynamics.mass + DMatrix::identity(q.len(), q.len()) * 0.2)
                    .lu()
                    .solve(&reference.force)
                    .unwrap()
            };
            let actual = mass.readback().unwrap();
            assert!((&actual[0] - expected).norm() < 1e-4);

            let changed_q = [-0.4, 0.35, 0.25, -0.3, 0.15];
            let changed_root = Isometry3::translation(-2.0, 0.5, 1.0)
                * Isometry3::rotation(Vector3::new(-0.2, 0.1, -0.4));
            state
                .reset(&[GpuGeneralizedState {
                    positions: DVector::from_column_slice(&changed_q),
                    velocities: DVector::zeros(changed_q.len()),
                }])
                .unwrap();
            pose.set_root_pose(0, changed_root).unwrap();
            let changed_reference = reference_system(&articulation, changed_root, &changed_q);
            let mut encoder = context.device().create_command_encoder(&Default::default());
            pose.encode(&mut encoder);
            terms.encode(&mut encoder);
            mass.encode(&mut encoder);
            let _ = context.queue().submit(Some(encoder.finish()));
            let bytes =
                read_buffer(context.device(), context.queue(), mass.link_terms_buffer()).unwrap();
            check_terms(&bytes, &changed_reference);
            let changed_pose = articulation.pose(changed_root, &changed_q).unwrap();
            let changed_dynamics = articulation
                .dynamics(&changed_pose, Vector3::zeros())
                .unwrap();
            let expected = (changed_dynamics.mass
                + DMatrix::identity(changed_q.len(), changed_q.len()) * 0.2)
                .lu()
                .solve(&changed_reference.force)
                .unwrap();
            let actual = mass.readback().unwrap();
            assert!((&actual[0] - expected).norm() < 1e-4);
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn packed_environments_keep_distinct_link_and_joint_offsets() {
        let make_link = |mass| LinkSpec {
            mass,
            center_of_mass: Vector3::new(0.2, -0.1, 0.15),
            inertia: Matrix3::from_diagonal(&Vector3::new(0.2, 0.3, 0.5)),
        };
        let joint = |parent, child, kind| JointSpec {
            parent,
            child,
            kind,
            origin: Isometry3::translation(0.8, 0.1, 0.0),
            axis: Vector3::z(),
            limits: None,
        };
        let first = Articulation::new(
            vec![make_link(1.0), make_link(2.0)],
            vec![joint(0, 1, JointKind::Revolute)],
            0,
        )
        .unwrap();
        let second = Articulation::new(
            vec![make_link(1.5), make_link(0.0), make_link(3.0)],
            vec![
                joint(0, 1, JointKind::Prismatic),
                joint(1, 2, JointKind::Revolute),
            ],
            0,
        )
        .unwrap();
        let roots = [
            Isometry3::identity(),
            Isometry3::translation(2.0, -1.0, 0.5),
        ];
        let coordinates = [vec![0.3], vec![0.2, -0.4]];
        let references = [
            reference_system(&first, roots[0], &coordinates[0]),
            reference_system(&second, roots[1], &coordinates[1]),
        ];
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let direct = GpuArticulatedMassBatch::new(
                context.device(),
                context.queue(),
                &coordinates
                    .iter()
                    .map(|q| GpuArticulatedMassSystem {
                        mass: DMatrix::identity(q.len(), q.len()),
                        force: DVector::zeros(q.len()),
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            let state = GpuGeneralizedStateBatch::from_mass_batch(
                &direct,
                &coordinates
                    .iter()
                    .map(|q| GpuGeneralizedState {
                        positions: DVector::from_column_slice(q),
                        velocities: DVector::zeros(q.len()),
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            let pose = GpuArticulatedPoseBatch::new(&state, &[&first, &second], &roots).unwrap();
            let mass = GpuArticulatedMassAssemblyBatch::new(
                context.device(),
                context.queue(),
                &references,
            )
            .unwrap();
            let terms =
                GpuArticulatedLinkTermsBatch::new(&pose, &mass, &[&first, &second]).unwrap();
            let mut encoder = context.device().create_command_encoder(&Default::default());
            pose.encode(&mut encoder);
            terms.encode(&mut encoder);
            mass.encode(&mut encoder);
            let _ = context.queue().submit(Some(encoder.finish()));
            let bytes =
                read_buffer(context.device(), context.queue(), mass.link_terms_buffer()).unwrap();
            let first_bytes = references[0].links.len() * (10 + 6 * coordinates[0].len()) * 4;
            check_terms(&bytes[..first_bytes], &references[0]);
            check_terms(&bytes[first_bytes..], &references[1]);
            assert_eq!(mass.readback().unwrap().len(), 2);
        }
        assert!(tested > 0, "no GPU backend available");
    }
}
