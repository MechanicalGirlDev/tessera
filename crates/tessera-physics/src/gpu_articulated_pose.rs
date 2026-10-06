//! Device-resident forward kinematics for packed articulated trees.

use core::{mem::size_of, ops::Range};

use nalgebra::{Isometry3, Quaternion, Translation3, UnitQuaternion};
use wgpu::util::DeviceExt;

use crate::articulation::{Articulation, JointKind};
use crate::gpu_articulated_mass::read_buffer;
use crate::gpu_articulated_spherical::GpuArticulatedSphericalBatch;
use crate::gpu_articulated_state::GpuGeneralizedStateBatch;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuPose {
    position: [f32; 4],
    orientation: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuJoint {
    indices: [u32; 4],
    origin_position: [f32; 4],
    origin_orientation: [f32; 4],
    axis_scale: [f32; 4],
    offset: [f32; 4],
    spherical: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuEnvironment {
    indices: [u32; 4],
    counts: [u32; 4],
}

/// Invalid model, exceeded device limits, or failed GPU pose readback.
#[derive(Debug, thiserror::Error)]
pub enum GpuArticulatedPoseError {
    /// The number or layout of articulations and generalized states differ.
    #[error("invalid articulated pose input")]
    InvalidInput,
    /// The packed topology exceeds GPU buffer or dispatch limits.
    #[error("articulated pose batch exceeds GPU capacity")]
    Capacity,
    /// A GPU output contains non-finite or invalid pose components.
    #[error("articulated pose {0} is invalid")]
    InvalidPose(usize),
    /// The source generalized state has a persistent numerical fault.
    #[error("generalized state {0} is faulted")]
    SourceFault(usize),
    /// A GPU readback failed.
    #[error("articulated pose GPU readback failed: {0}")]
    Readback(String),
}

/// GPU link poses derived from the coordinates of a generalized state batch.
///
/// Joint coordinates follow each articulation's independent DOF order, optionally
/// after a six-slot floating root workspace. Roots are supplied separately. Topology is immutable
/// for this batch; rebuild it after changing a model or environment layout.
#[derive(Debug)]
pub struct GpuArticulatedPoseBatch {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    coordinates: wgpu::Buffer,
    joints: wgpu::Buffer,
    environments: wgpu::Buffer,
    roots: wgpu::Buffer,
    links: wgpu::Buffer,
    state_status: wgpu::Buffer,
    spherical_orientations: wgpu::Buffer,
    link_ranges: Vec<Range<usize>>,
    floating_roots: Vec<bool>,
    quaternion_spherical: bool,
}

impl GpuArticulatedPoseBatch {
    /// Upload fixed-root tree topology and bind device-resident coordinates.
    pub fn new(
        states: &GpuGeneralizedStateBatch,
        articulations: &[&Articulation],
        root_poses: &[Isometry3<f64>],
    ) -> Result<Self, GpuArticulatedPoseError> {
        Self::new_with_floating_roots(
            states,
            articulations,
            root_poses,
            &vec![false; articulations.len()],
        )
    }

    /// Bind joint coordinates after an optional six-slot root workspace.
    ///
    /// Floating environments reserve six leading generalized slots. Their
    /// velocities are world-frame linear and angular velocities; root poses
    /// remain in the separate pose buffer, updated by the root integration pass.
    /// The reserved positions are not interpreted as Euler root coordinates.
    pub fn new_with_floating_roots(
        states: &GpuGeneralizedStateBatch,
        articulations: &[&Articulation],
        root_poses: &[Isometry3<f64>],
        floating_roots: &[bool],
    ) -> Result<Self, GpuArticulatedPoseError> {
        Self::new_impl(states, articulations, root_poses, floating_roots, None)
    }

    /// Bind quaternion spherical orientations and tangent angular velocity layouts.
    ///
    /// Every spherical edge must have exactly one matching source velocity slot,
    /// including a six-slot prefix in floating environments. FK ignores the three
    /// spherical coordinate slots and reads quaternions from `spherical` instead.
    /// Link-term assembly uses joint-frame angular velocity Jacobian columns.
    pub fn new_with_spherical_state(
        states: &GpuGeneralizedStateBatch,
        articulations: &[&Articulation],
        root_poses: &[Isometry3<f64>],
        floating_roots: &[bool],
        spherical: &GpuArticulatedSphericalBatch,
    ) -> Result<Self, GpuArticulatedPoseError> {
        Self::new_impl(
            states,
            articulations,
            root_poses,
            floating_roots,
            Some(spherical),
        )
    }

    fn new_impl(
        states: &GpuGeneralizedStateBatch,
        articulations: &[&Articulation],
        root_poses: &[Isometry3<f64>],
        floating_roots: &[bool],
        spherical: Option<&GpuArticulatedSphericalBatch>,
    ) -> Result<Self, GpuArticulatedPoseError> {
        let device = states.device();
        let queue = states.queue();
        if articulations.is_empty()
            || articulations.len() != states.ranges().len()
            || articulations.len() != root_poses.len()
            || articulations.len() != floating_roots.len()
            || spherical.is_some_and(|batch| !batch.matches_source(states))
        {
            return Err(GpuArticulatedPoseError::InvalidInput);
        }
        let mut joints = Vec::new();
        let mut environments = Vec::with_capacity(articulations.len());
        let mut roots = Vec::with_capacity(articulations.len());
        let mut link_ranges = Vec::with_capacity(articulations.len());
        let mut link_count = 0usize;
        for (environment, (((articulation, root_pose), coordinates), &floating)) in articulations
            .iter()
            .zip(root_poses)
            .zip(states.ranges())
            .zip(floating_roots)
            .enumerate()
        {
            let root_dofs = if floating { 6 } else { 0 };
            if articulation.dof().checked_add(root_dofs) != Some(coordinates.len()) {
                return Err(GpuArticulatedPoseError::InvalidInput);
            }
            let (root, traversal, specs) = articulation.gpu_pose_topology();
            if spherical.is_some_and(|batch| {
                batch.velocity_slots()[environment].len()
                    != specs
                        .iter()
                        .filter(|joint| joint.kind == JointKind::Spherical)
                        .count()
            }) {
                return Err(GpuArticulatedPoseError::InvalidInput);
            }
            let link_start = link_count;
            link_count = link_count
                .checked_add(articulation.link_count())
                .ok_or(GpuArticulatedPoseError::Capacity)?;
            let joint_start = joints.len();
            for &edge in traversal {
                let spec = &specs[edge];
                let kind = match spec.kind {
                    JointKind::Fixed => 0,
                    JointKind::Revolute => 1,
                    JointKind::Prismatic => 2,
                    JointKind::Spherical => {
                        if spherical.is_some() {
                            4
                        } else {
                            3
                        }
                    }
                };
                let slot = if spec.kind == JointKind::Fixed {
                    u32::MAX
                } else {
                    checked_u32(
                        articulation
                            .joint_coordinate_range(edge)
                            .ok_or(GpuArticulatedPoseError::InvalidInput)?
                            .start
                            .checked_add(root_dofs)
                            .ok_or(GpuArticulatedPoseError::Capacity)?,
                    )?
                };
                let origin = gpu_pose(spec.origin)?;
                let scale = articulation
                    .joint_coordinate_scale(edge)
                    .ok_or(GpuArticulatedPoseError::InvalidInput)?;
                let offset = articulation
                    .joint_coordinate_offset(edge)
                    .ok_or(GpuArticulatedPoseError::InvalidInput)?;
                let spherical_index = if kind == 4 {
                    checked_u32(
                        spherical
                            .and_then(|batch| batch.orientation_index(environment, slot as usize))
                            .ok_or(GpuArticulatedPoseError::InvalidInput)?,
                    )?
                } else {
                    u32::MAX
                };
                joints.push(GpuJoint {
                    indices: [
                        kind,
                        checked_u32(spec.parent)?,
                        checked_u32(spec.child)?,
                        slot,
                    ],
                    origin_position: origin.position,
                    origin_orientation: origin.orientation,
                    axis_scale: [
                        finite_f32(spec.axis.x)?,
                        finite_f32(spec.axis.y)?,
                        finite_f32(spec.axis.z)?,
                        finite_f32(scale)?,
                    ],
                    offset: [finite_f32(offset)?, 0.0, 0.0, 0.0],
                    spherical: [spherical_index, 0, 0, 0],
                });
            }
            environments.push(GpuEnvironment {
                indices: [
                    checked_u32(link_start)?,
                    checked_u32(joint_start)?,
                    checked_u32(coordinates.start)?,
                    checked_u32(root)?,
                ],
                counts: [
                    checked_u32(traversal.len())?,
                    checked_u32(coordinates.len())?,
                    checked_u32(root_dofs)?,
                    0,
                ],
            });
            link_ranges.push(link_start..link_count);
            roots.push(gpu_pose(*root_pose)?);
        }
        if joints.is_empty() {
            joints.push(<GpuJoint as bytemuck::Zeroable>::zeroed());
        }
        let limits = device.limits();
        let joint_bytes = byte_size::<GpuJoint>(joints.len())?;
        let environment_bytes = byte_size::<GpuEnvironment>(environments.len())?;
        let root_bytes = byte_size::<GpuPose>(roots.len())?;
        let link_bytes = byte_size::<GpuPose>(link_count)?;
        if [joint_bytes, environment_bytes, root_bytes, link_bytes]
            .into_iter()
            .any(|bytes| {
                bytes > limits.max_buffer_size
                    || bytes > u64::from(limits.max_storage_buffer_binding_size)
            })
            || articulations.len().div_ceil(64)
                > limits.max_compute_workgroups_per_dimension as usize
            || limits.max_storage_buffers_per_shader_stage < 7
        {
            return Err(GpuArticulatedPoseError::Capacity);
        }
        let joints = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera GPU articulation joints"),
            contents: bytemuck::cast_slice(&joints),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let environments = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera GPU articulation layouts"),
            contents: bytemuck::cast_slice(&environments),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let roots = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera GPU articulation roots"),
            contents: bytemuck::cast_slice(&roots),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let links = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera GPU articulation link poses"),
            size: link_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let spherical_orientations = if let Some(batch) = spherical {
            batch.orientation_buffer().clone()
        } else {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tessera unused spherical orientation"),
                contents: bytemuck::cast_slice(&[0.0f32, 0.0, 0.0, 1.0]),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera GPU articulation forward kinematics"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_articulated_pose.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tessera GPU articulation forward kinematics"),
            layout: None,
            module: &module,
            entry_point: Some("forward_kinematics"),
            compilation_options: Default::default(),
            cache: None,
        });
        Ok(Self {
            device: device.clone(),
            queue: queue.clone(),
            pipeline,
            coordinates: states.position_buffer().clone(),
            joints,
            environments,
            roots,
            links,
            state_status: states.status_buffer().clone(),
            spherical_orientations,
            link_ranges,
            floating_roots: floating_roots.to_vec(),
            quaternion_spherical: spherical.is_some(),
        })
    }

    /// Encode forward kinematics after any queued coordinate updates.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let inputs = [
            &self.coordinates,
            &self.joints,
            &self.environments,
            &self.roots,
            &self.links,
            &self.state_status,
            &self.spherical_orientations,
        ];
        let entries = inputs
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>();
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera GPU articulation pose bindings"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Tessera GPU articulation forward kinematics"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(self.link_ranges.len().div_ceil(64) as u32, 1, 1);
    }

    /// Change one environment's fixed root pose before its next encode.
    pub fn set_root_pose(
        &self,
        environment: usize,
        pose: Isometry3<f64>,
    ) -> Result<(), GpuArticulatedPoseError> {
        if environment >= self.link_ranges.len() {
            return Err(GpuArticulatedPoseError::InvalidInput);
        }
        self.queue.write_buffer(
            &self.roots,
            byte_size::<GpuPose>(environment)?,
            bytemuck::bytes_of(&gpu_pose(pose)?),
        );
        Ok(())
    }

    /// Link transforms in stable input-link order for each environment.
    pub fn readback(&self) -> Result<Vec<Vec<Isometry3<f64>>>, GpuArticulatedPoseError> {
        self.check_source_status()?;
        let bytes = read_buffer(&self.device, &self.queue, &self.links)
            .map_err(|error| GpuArticulatedPoseError::Readback(error.to_string()))?;
        self.link_ranges
            .iter()
            .map(|range| {
                range
                    .clone()
                    .map(|index| {
                        let start = index * size_of::<GpuPose>();
                        let pose = bytemuck::pod_read_unaligned::<GpuPose>(
                            &bytes[start..start + size_of::<GpuPose>()],
                        );
                        pose_to_f64(pose, index)
                    })
                    .collect()
            })
            .collect()
    }

    /// Root poses in environment order, independent of each model's root link index.
    pub fn readback_roots(&self) -> Result<Vec<Isometry3<f64>>, GpuArticulatedPoseError> {
        self.check_source_status()?;
        let bytes = read_buffer(&self.device, &self.queue, &self.roots)
            .map_err(|error| GpuArticulatedPoseError::Readback(error.to_string()))?;
        bytes
            .chunks_exact(size_of::<GpuPose>())
            .enumerate()
            .map(|(index, bytes)| {
                pose_to_f64(bytemuck::pod_read_unaligned::<GpuPose>(bytes), index)
            })
            .collect()
    }

    fn check_source_status(&self) -> Result<(), GpuArticulatedPoseError> {
        let status = read_buffer(&self.device, &self.queue, &self.state_status)
            .map_err(|error| GpuArticulatedPoseError::Readback(error.to_string()))?;
        for (index, flag) in status.chunks_exact(4).enumerate() {
            let flag = u32::from_le_bytes(
                flag.try_into()
                    .map_err(|_| GpuArticulatedPoseError::Capacity)?,
            );
            if flag != 0 {
                return Err(GpuArticulatedPoseError::SourceFault(index));
            }
        }
        Ok(())
    }

    /// Device-resident packed link pose buffer for later dynamics and contact passes.
    pub fn link_pose_buffer(&self) -> &wgpu::Buffer {
        &self.links
    }

    /// Link index ranges for each environment in the packed pose buffer.
    pub fn link_ranges(&self) -> &[Range<usize>] {
        &self.link_ranges
    }

    /// Whether spherical slots denote joint-frame angular velocities and use quaternion FK.
    pub fn uses_spherical_quaternions(&self) -> bool {
        self.quaternion_spherical
    }

    pub(crate) fn spherical_orientation_buffer(&self) -> &wgpu::Buffer {
        &self.spherical_orientations
    }

    pub(crate) fn root_pose_buffer(&self) -> &wgpu::Buffer {
        &self.roots
    }

    pub(crate) fn floating_roots(&self) -> &[bool] {
        &self.floating_roots
    }

    pub(crate) fn joint_buffer(&self) -> &wgpu::Buffer {
        &self.joints
    }
    pub(crate) fn environment_buffer(&self) -> &wgpu::Buffer {
        &self.environments
    }
    pub(crate) fn coordinate_buffer(&self) -> &wgpu::Buffer {
        &self.coordinates
    }
    pub(crate) fn status_buffer(&self) -> &wgpu::Buffer {
        &self.state_status
    }
    pub(crate) fn device(&self) -> &wgpu::Device {
        &self.device
    }
}

fn checked_u32(value: usize) -> Result<u32, GpuArticulatedPoseError> {
    u32::try_from(value).map_err(|_| GpuArticulatedPoseError::Capacity)
}

fn byte_size<T>(count: usize) -> Result<u64, GpuArticulatedPoseError> {
    u64::try_from(count)
        .ok()
        .and_then(|count| count.checked_mul(size_of::<T>() as u64))
        .ok_or(GpuArticulatedPoseError::Capacity)
}

fn finite_f32(value: f64) -> Result<f32, GpuArticulatedPoseError> {
    let narrowed = value as f32;
    if !value.is_finite() || !narrowed.is_finite() || (value != 0.0 && narrowed == 0.0) {
        return Err(GpuArticulatedPoseError::InvalidInput);
    }
    Ok(narrowed)
}

fn gpu_pose(pose: Isometry3<f64>) -> Result<GpuPose, GpuArticulatedPoseError> {
    let point = pose.translation.vector;
    let rotation = pose.rotation.quaternion();
    Ok(GpuPose {
        position: [
            finite_f32(point.x)?,
            finite_f32(point.y)?,
            finite_f32(point.z)?,
            0.0,
        ],
        orientation: [
            finite_f32(rotation.i)?,
            finite_f32(rotation.j)?,
            finite_f32(rotation.k)?,
            finite_f32(rotation.w)?,
        ],
    })
}

fn pose_to_f64(pose: GpuPose, index: usize) -> Result<Isometry3<f64>, GpuArticulatedPoseError> {
    let [x, y, z, _] = pose.position;
    let [i, j, k, w] = pose.orientation;
    if [x, y, z, i, j, k, w].iter().any(|value| !value.is_finite()) {
        return Err(GpuArticulatedPoseError::InvalidPose(index));
    }
    let rotation = Quaternion::new(f64::from(w), f64::from(i), f64::from(j), f64::from(k));
    if rotation.norm_squared() < 1e-12 {
        return Err(GpuArticulatedPoseError::InvalidPose(index));
    }
    Ok(Isometry3::from_parts(
        Translation3::new(f64::from(x), f64::from(y), f64::from(z)),
        UnitQuaternion::new_normalize(rotation),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::articulation::{JointSpec, LinkSpec};
    use crate::gpu_articulated_mass::{GpuArticulatedMassBatch, GpuArticulatedMassSystem};
    use crate::gpu_articulated_state::GpuGeneralizedState;
    use crate::gpu_contact_pipeline::GpuContactDevice;
    use nalgebra::{DMatrix, DVector, Matrix3, Vector3};

    fn links(count: usize) -> Vec<LinkSpec> {
        (0..count)
            .map(|_| LinkSpec {
                mass: 1.0,
                center_of_mass: Vector3::zeros(),
                inertia: Matrix3::identity(),
            })
            .collect()
    }

    fn joint(parent: usize, child: usize, kind: JointKind, origin: Isometry3<f64>) -> JointSpec {
        JointSpec {
            parent,
            child,
            origin,
            kind,
            axis: Vector3::z(),
            limits: None,
        }
    }

    fn check_pose(actual: &[Isometry3<f64>], expected: &[Isometry3<f64>]) {
        assert_eq!(actual.len(), expected.len());
        for (result, reference) in actual.iter().zip(expected) {
            assert!((result.translation.vector - reference.translation.vector).norm() < 1e-5);
            assert!((result.rotation.inverse() * reference.rotation).angle() < 1e-5);
        }
    }

    #[test]
    fn packed_gpu_tree_poses_follow_state_update_and_mimic_offsets() {
        let first = Articulation::new(
            links(5),
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
                    Isometry3::translation(0.0, 1.0, 0.0)
                        * Isometry3::rotation(Vector3::new(0.2, 0.1, 0.0)),
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
                    JointKind::Fixed,
                    Isometry3::translation(-1.0, 0.5, 0.0),
                ),
            ],
            0,
        )
        .unwrap();
        let mut second = Articulation::new(
            links(3),
            vec![
                joint(
                    0,
                    1,
                    JointKind::Revolute,
                    Isometry3::translation(0.0, 1.0, 0.0),
                ),
                joint(
                    1,
                    2,
                    JointKind::Revolute,
                    Isometry3::translation(1.0, 0.0, 0.0),
                ),
            ],
            0,
        )
        .unwrap();
        second.set_mimics(&[(1, 0, 2.0, 0.1)]).unwrap();
        let third = Articulation::new(
            links(2),
            vec![joint(
                1,
                0,
                JointKind::Prismatic,
                Isometry3::translation(0.0, 0.0, 1.0),
            )],
            1,
        )
        .unwrap();
        let roots = [
            Isometry3::translation(1.0, 2.0, 3.0)
                * Isometry3::rotation(Vector3::new(0.0, 0.0, 0.3)),
            Isometry3::translation(-2.0, 1.0, 0.0)
                * Isometry3::rotation(Vector3::new(0.0, -0.2, 0.0)),
            Isometry3::translation(0.0, 2.0, -1.0),
        ];
        let q = [vec![0.3, 0.2, 0.1, -0.2, 0.4], vec![-0.4], vec![0.7]];
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let systems = q
                .iter()
                .map(|coordinates| GpuArticulatedMassSystem {
                    mass: DMatrix::identity(coordinates.len(), coordinates.len()),
                    force: DVector::from_element(coordinates.len(), 1.0),
                })
                .collect::<Vec<_>>();
            let mass =
                GpuArticulatedMassBatch::new(context.device(), context.queue(), &systems).unwrap();
            let states = q
                .iter()
                .map(|coordinates| GpuGeneralizedState {
                    positions: DVector::from_column_slice(coordinates),
                    velocities: DVector::zeros(coordinates.len()),
                })
                .collect::<Vec<_>>();
            let state = GpuGeneralizedStateBatch::from_mass_batch(&mass, &states).unwrap();
            let poses =
                GpuArticulatedPoseBatch::new(&state, &[&first, &second, &third], &roots).unwrap();
            let mut encoder = context.device().create_command_encoder(&Default::default());
            poses.encode(&mut encoder);
            let _ = context.queue().submit(Some(encoder.finish()));
            let result = poses.readback().unwrap();
            check_pose(&result[0], &first.pose(roots[0], &q[0]).unwrap().links);
            check_pose(&result[1], &second.pose(roots[1], &q[1]).unwrap().links);
            check_pose(&result[2], &third.pose(roots[2], &q[2]).unwrap().links);

            let changed_root = Isometry3::translation(3.0, -1.0, 2.0);
            poses.set_root_pose(1, changed_root).unwrap();
            let mut encoder = context.device().create_command_encoder(&Default::default());
            mass.encode(&mut encoder);
            state.encode_step(&mut encoder, 0.1).unwrap();
            poses.encode(&mut encoder);
            let _ = context.queue().submit(Some(encoder.finish()));
            let result = poses.readback().unwrap();
            let advanced_first = q[0].iter().map(|value| value + 0.01).collect::<Vec<_>>();
            let advanced_second = [q[1][0] + 0.01];
            check_pose(
                &result[0],
                &first.pose(roots[0], &advanced_first).unwrap().links,
            );
            check_pose(
                &result[1],
                &second.pose(changed_root, &advanced_second).unwrap().links,
            );
            check_pose(
                &result[2],
                &third.pose(roots[2], &[q[2][0] + 0.01]).unwrap().links,
            );
            assert!(
                poses
                    .link_pose_buffer()
                    .usage()
                    .contains(wgpu::BufferUsages::STORAGE)
            );

            let mut overflowing = systems.clone();
            overflowing[0].force.fill(1e29);
            mass.update(&overflowing).unwrap();
            let mut encoder = context.device().create_command_encoder(&Default::default());
            mass.encode(&mut encoder);
            state.encode_step(&mut encoder, 100.0).unwrap();
            poses.encode(&mut encoder);
            let _ = context.queue().submit(Some(encoder.finish()));
            assert!(matches!(
                poses.readback(),
                Err(GpuArticulatedPoseError::SourceFault(0))
            ));
            state.reset(&states).unwrap();
            let mut encoder = context.device().create_command_encoder(&Default::default());
            poses.encode(&mut encoder);
            let _ = context.queue().submit(Some(encoder.finish()));
            let result = poses.readback().unwrap();
            check_pose(&result[0], &first.pose(roots[0], &q[0]).unwrap().links);
        }
        assert!(tested > 0, "no GPU backend available");
    }
}
