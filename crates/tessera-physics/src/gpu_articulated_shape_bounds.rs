//! Device-resident world bounds for articulated sphere, capsule, and box shapes.

use core::mem::size_of;

use nalgebra::{Isometry3, Vector3};
use wgpu::util::DeviceExt;

use crate::articulation::Articulation;
use crate::gpu_articulated_mass_assembly::GpuArticulatedMassAssemblyBatch;
use crate::gpu_articulated_pose::GpuArticulatedPoseBatch;
use crate::gpu_articulated_state::GpuGeneralizedStateBatch;
use crate::gpu_broad_phase::{GpuAabb, GpuPair};
use crate::gpu_lbvh::{GpuLbvh, GpuLbvhError, GpuLbvhResidentPairs};

/// A collision shape fixed to one link in its environment.
#[derive(Debug, Clone, Copy)]
pub enum GpuArticulatedBoundsShape {
    /// Sphere with a link-local center and positive radius.
    Sphere {
        /// Link index in the environment.
        link: usize,
        /// Center in link coordinates.
        center: Vector3<f64>,
        /// Sphere radius.
        radius: f64,
    },
    /// Capsule with link-local segment endpoints and positive radius.
    Capsule {
        /// Link index in the environment.
        link: usize,
        /// First segment endpoint in link coordinates.
        a: Vector3<f64>,
        /// Second segment endpoint in link coordinates.
        b: Vector3<f64>,
        /// Capsule radius.
        radius: f64,
    },
    /// Oriented box with link-local pose and positive half extents.
    Box {
        /// Link index in the environment.
        link: usize,
        /// Box pose in link coordinates.
        pose: Isometry3<f64>,
        /// Positive box half extents.
        half_extents: Vector3<f64>,
    },
}

impl GpuArticulatedBoundsShape {
    fn link(self) -> usize {
        match self {
            Self::Sphere { link, .. } | Self::Capsule { link, .. } | Self::Box { link, .. } => link,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PackedShape {
    index_kind: [u32; 4],
    a: [f32; 4],
    b: [f32; 4],
    orientation: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PackedSweepMeta {
    indices: [u32; 4],
    center_radius: [f32; 4],
    center_of_mass_dt: [f32; 4],
}

#[derive(Debug)]
struct SweepState {
    pipeline: wgpu::ComputePipeline,
    velocities: wgpu::Buffer,
    link_terms: wgpu::Buffer,
    accelerations: wgpu::Buffer,
    metadata: wgpu::Buffer,
}

/// Invalid shape geometry or exceeded GPU storage/dispatch capacity.
#[derive(Debug, thiserror::Error)]
pub enum GpuArticulatedShapeBoundsError {
    /// A link index, shape dimension, or coordinate is invalid.
    #[error("invalid articulated collision shape")]
    InvalidInput,
    /// The shape buffers or dispatch exceed device limits.
    #[error("articulated shape bounds exceed GPU capacity")]
    Capacity,
    /// The resident LBVH could not allocate its candidate buffer.
    #[error(transparent)]
    Lbvh(#[from] GpuLbvhError),
}

/// Compute world AABBs from packed link poses without a CPU pose readback.
///
/// Shapes are packed in environment order, then in each supplied slice's order.
/// Their indices match the collider indices returned by `encode_candidates`.
#[derive(Debug)]
pub struct GpuArticulatedShapeBoundsBatch {
    pipeline: wgpu::ComputePipeline,
    filter_pipeline: wgpu::ComputePipeline,
    poses: wgpu::Buffer,
    shapes: wgpu::Buffer,
    filters: wgpu::Buffer,
    exclusions: wgpu::Buffer,
    bounds: wgpu::Buffer,
    count: u32,
    sweep: Option<SweepState>,
}

impl GpuArticulatedShapeBoundsBatch {
    /// Pack stable link-local shapes and articulation collision exclusions.
    /// Subsequent encodes use current GPU poses; rebuild after topology changes.
    pub fn new(
        device: &wgpu::Device,
        poses: &GpuArticulatedPoseBatch,
        articulations: &[&Articulation],
        environments: &[&[GpuArticulatedBoundsShape]],
    ) -> Result<Self, GpuArticulatedShapeBoundsError> {
        if environments.len() != poses.link_ranges().len()
            || articulations.len() != environments.len()
            || articulations
                .iter()
                .zip(poses.link_ranges())
                .any(|(articulation, range)| articulation.link_count() != range.len())
        {
            return Err(GpuArticulatedShapeBoundsError::InvalidInput);
        }
        let count = environments.iter().try_fold(0usize, |total, shapes| {
            total
                .checked_add(shapes.len())
                .ok_or(GpuArticulatedShapeBoundsError::Capacity)
        })?;
        if count == 0 {
            return Err(GpuArticulatedShapeBoundsError::InvalidInput);
        }
        let count_u32 =
            u32::try_from(count).map_err(|_| GpuArticulatedShapeBoundsError::Capacity)?;
        let limits = device.limits();
        let shape_bytes = size_of::<PackedShape>() as u64 * count as u64;
        let bounds_bytes = size_of::<GpuAabb>() as u64 * count as u64;
        let filter_bytes = size_of::<[u32; 4]>() as u64 * count as u64;
        if count_u32.div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || [shape_bytes, bounds_bytes, filter_bytes]
                .iter()
                .any(|&bytes| {
                    bytes > limits.max_buffer_size
                        || bytes > u64::from(limits.max_storage_buffer_binding_size)
                })
            || limits.max_storage_buffers_per_shader_stage < 3
        {
            return Err(GpuArticulatedShapeBoundsError::Capacity);
        }
        let mut packed = Vec::with_capacity(count);
        let mut filters = Vec::with_capacity(count);
        let mut exclusions = Vec::<[u32; 2]>::new();
        for (articulation, range) in articulations.iter().zip(poses.link_ranges()) {
            for first in 0..range.len() {
                for second in first + 1..range.len() {
                    if articulation.adjacent(first, second) {
                        exclusions.push([
                            checked_link(range.start, first)?,
                            checked_link(range.start, second)?,
                        ]);
                    }
                }
            }
        }
        if exclusions.is_empty() {
            exclusions.push([u32::MAX, u32::MAX]);
        }
        let exclusion_bytes = exclusions.len() as u64 * size_of::<[u32; 2]>() as u64;
        if exclusion_bytes > limits.max_buffer_size
            || exclusion_bytes > u64::from(limits.max_storage_buffer_binding_size)
            || limits.max_storage_buffers_per_shader_stage < 6
        {
            return Err(GpuArticulatedShapeBoundsError::Capacity);
        }
        for (environment, (range, shapes)) in
            poses.link_ranges().iter().zip(environments).enumerate()
        {
            let environment =
                u32::try_from(environment).map_err(|_| GpuArticulatedShapeBoundsError::Capacity)?;
            for shape in *shapes {
                if shape.link() >= range.len() {
                    return Err(GpuArticulatedShapeBoundsError::InvalidInput);
                }
                let link = checked_link(range.start, shape.link())?;
                let mut entry = PackedShape {
                    index_kind: [link, 0, 0, 0],
                    a: [0.0; 4],
                    b: [0.0; 4],
                    orientation: [0.0, 0.0, 0.0, 1.0],
                };
                match *shape {
                    GpuArticulatedBoundsShape::Sphere { center, radius, .. } => {
                        if radius <= 0.0 {
                            return Err(GpuArticulatedShapeBoundsError::InvalidInput);
                        }
                        entry.a = vector_radius(center, radius)?;
                    }
                    GpuArticulatedBoundsShape::Capsule { a, b, radius, .. } => {
                        if radius <= 0.0 {
                            return Err(GpuArticulatedShapeBoundsError::InvalidInput);
                        }
                        entry.index_kind[1] = 1;
                        entry.a = vector_radius(a, radius)?;
                        entry.b = vector_radius(b, 0.0)?;
                    }
                    GpuArticulatedBoundsShape::Box {
                        pose, half_extents, ..
                    } => {
                        entry.index_kind[1] = 2;
                        entry.a = vector_radius(pose.translation.vector, 0.0)?;
                        entry.b = vector_radius(half_extents, 0.0)?;
                        if entry.b[..3].iter().any(|&extent| extent <= 0.0) {
                            return Err(GpuArticulatedShapeBoundsError::InvalidInput);
                        }
                        let q = pose.rotation.quaternion();
                        entry.orientation = [
                            finite_f32(q.i)?,
                            finite_f32(q.j)?,
                            finite_f32(q.k)?,
                            finite_f32(q.w)?,
                        ];
                    }
                }
                packed.push(entry);
                filters.push([environment, u32::MAX, u32::MAX, 0]);
            }
        }
        let shapes = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera articulated collision shapes"),
            contents: bytemuck::cast_slice(&packed),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let filters = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera articulated shape environment filters"),
            contents: bytemuck::cast_slice(&filters),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let exclusions = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera articulated excluded link pairs"),
            contents: bytemuck::cast_slice(&exclusions),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let bounds = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera articulated world AABBs"),
            size: bounds_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera articulated shape bounds"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("gpu_articulated_shape_bounds.wgsl").into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tessera articulated shape bounds"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let filter_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera articulated candidate filter"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("gpu_articulated_candidate_filter.wgsl").into(),
            ),
        });
        let filter_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tessera articulated candidate filter"),
            layout: None,
            module: &filter_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Ok(Self {
            pipeline,
            filter_pipeline,
            poses: poses.link_pose_buffer().clone(),
            shapes,
            filters,
            exclusions,
            bounds,
            count: count_u32,
            sweep: None,
        })
    }

    /// Expand bounds over one step using generalized motion and link Jacobians.
    ///
    /// Encode link terms and mass acceleration before these bounds. This
    /// includes approaching, off-center, and rotating shapes in the candidates.
    pub fn with_motion(
        mut self,
        poses: &GpuArticulatedPoseBatch,
        state: &GpuGeneralizedStateBatch,
        mass: &GpuArticulatedMassAssemblyBatch,
        articulations: &[&Articulation],
        environments: &[&[GpuArticulatedBoundsShape]],
        timestep: f64,
    ) -> Result<Self, GpuArticulatedShapeBoundsError> {
        let device = state.device();
        let total_shapes = environments.iter().try_fold(0usize, |total, shapes| {
            total
                .checked_add(shapes.len())
                .ok_or(GpuArticulatedShapeBoundsError::Capacity)
        })?;
        if !timestep.is_finite()
            || timestep <= 0.0
            || environments.len() != articulations.len()
            || environments.len() != poses.link_ranges().len()
            || environments.len() != state.ranges().len()
            || environments.len() != mass.dimensions().len()
            || environments.len() != mass.link_counts().len()
            || total_shapes != self.count as usize
        {
            return Err(GpuArticulatedShapeBoundsError::InvalidInput);
        }
        let dt = finite_f32(timestep)?;
        let mut metadata = Vec::with_capacity(self.count as usize);
        let mut link_offset = 0usize;
        for (environment, shapes) in environments.iter().enumerate() {
            let articulation = articulations[environment];
            let n = mass.dimensions()[environment];
            let count = mass.link_counts()[environment];
            let root_dofs = if poses.floating_roots()[environment] {
                6
            } else {
                0
            };
            if articulation.dof().checked_add(root_dofs) != Some(n)
                || n != state.ranges()[environment].len()
                || count != articulation.link_count()
                || count != poses.link_ranges()[environment].len()
            {
                return Err(GpuArticulatedShapeBoundsError::InvalidInput);
            }
            let stride = 10usize
                .checked_add(
                    n.checked_mul(6)
                        .ok_or(GpuArticulatedShapeBoundsError::Capacity)?,
                )
                .ok_or(GpuArticulatedShapeBoundsError::Capacity)?;
            for shape in *shapes {
                let link = articulation
                    .link(shape.link())
                    .ok_or(GpuArticulatedShapeBoundsError::InvalidInput)?;
                let (center, radius) = match *shape {
                    GpuArticulatedBoundsShape::Sphere { center, radius, .. } => (center, radius),
                    GpuArticulatedBoundsShape::Capsule { a, b, radius, .. } => {
                        ((a + b) * 0.5, (b - a).norm() * 0.5 + radius)
                    }
                    GpuArticulatedBoundsShape::Box {
                        pose, half_extents, ..
                    } => (pose.translation.vector, half_extents.norm()),
                };
                let term_offset = link_offset
                    .checked_add(
                        shape
                            .link()
                            .checked_mul(stride)
                            .ok_or(GpuArticulatedShapeBoundsError::Capacity)?,
                    )
                    .ok_or(GpuArticulatedShapeBoundsError::Capacity)?;
                let mut com = vector_radius(link.center_of_mass, 0.0)?;
                com[3] = dt;
                metadata.push(PackedSweepMeta {
                    indices: [
                        checked_link(term_offset, 0)?,
                        checked_link(state.ranges()[environment].start, 0)?,
                        checked_link(n, 0)?,
                        0,
                    ],
                    center_radius: vector_radius(center, radius)?,
                    center_of_mass_dt: com,
                });
            }
            link_offset = link_offset
                .checked_add(
                    count
                        .checked_mul(stride)
                        .ok_or(GpuArticulatedShapeBoundsError::Capacity)?,
                )
                .ok_or(GpuArticulatedShapeBoundsError::Capacity)?;
        }
        let bytes = metadata.len() as u64 * size_of::<PackedSweepMeta>() as u64;
        let limits = device.limits();
        if bytes > limits.max_buffer_size
            || bytes > u64::from(limits.max_storage_buffer_binding_size)
            || limits.max_storage_buffers_per_shader_stage < 7
        {
            return Err(GpuArticulatedShapeBoundsError::Capacity);
        }
        let metadata = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera articulated shape sweep metadata"),
            contents: bytemuck::cast_slice(&metadata),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera articulated swept shape bounds"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("gpu_articulated_shape_bounds.wgsl").into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tessera articulated swept shape bounds"),
            layout: None,
            module: &shader,
            entry_point: Some("swept"),
            compilation_options: Default::default(),
            cache: None,
        });
        self.sweep = Some(SweepState {
            pipeline,
            velocities: state.velocity_buffer().clone(),
            link_terms: mass.link_terms_buffer().clone(),
            accelerations: mass.solution_buffer().clone(),
            metadata,
        });
        Ok(self)
    }

    /// Encode world AABB updates after forward kinematics in the same encoder.
    pub fn encode(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        let pipeline = self
            .sweep
            .as_ref()
            .map_or(&self.pipeline, |sweep| &sweep.pipeline);
        let mut inputs = vec![&self.poses, &self.shapes, &self.bounds];
        if let Some(sweep) = &self.sweep {
            inputs.extend([
                &sweep.velocities,
                &sweep.link_terms,
                &sweep.metadata,
                &sweep.accelerations,
            ]);
        }
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera articulated shape bounds bindings"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &inputs
                .iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Tessera articulated shape bounds"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(self.count.div_ceil(64), 1, 1);
    }

    /// Update bounds and build GPU-resident overlap candidates in one encoder.
    ///
    /// The caller must encode forward kinematics first. The returned pairs
    /// exclude different environments, same-link shapes, and excluded link
    /// pairs. Pair order may change between frames.
    pub fn encode_candidates(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        lbvh: &GpuLbvh,
    ) -> Result<GpuLbvhResidentPairs, GpuArticulatedShapeBoundsError> {
        self.encode(device, encoder);
        let raw = lbvh.encode_buffer_resident_filtered(
            device,
            encoder,
            &self.bounds,
            &self.filters,
            self.count,
        )?;
        let pair_bytes = u64::from(raw.pair_capacity) * size_of::<GpuPair>() as u64;
        let pairs = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera articulated eligible candidates"),
            size: pair_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let counter = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera articulated eligible candidate counter"),
            size: 8,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.clear_buffer(&counter, 0, None);
        encoder.copy_buffer_to_buffer(&raw.counter, 4, &counter, 4, 4);
        let dispatch = lbvh.encode_candidate_dispatch_args(device, encoder, &raw, 64)?;
        let inputs = [
            &raw.pairs,
            &raw.counter,
            &self.shapes,
            &self.exclusions,
            &pairs,
            &counter,
        ];
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera articulated candidate filter bindings"),
            layout: &self.filter_pipeline.get_bind_group_layout(0),
            entries: &inputs
                .iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Tessera articulated candidate filter"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.filter_pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups_indirect(&dispatch, 0);
        drop(pass);
        Ok(GpuLbvhResidentPairs {
            pairs,
            counter,
            pair_capacity: raw.pair_capacity,
            collider_count: self.count,
        })
    }

    /// GPU buffer of AABBs in stable input shape order.
    pub fn bounds_buffer(&self) -> &wgpu::Buffer {
        &self.bounds
    }

    /// Number of packed collision shapes.
    pub fn shape_count(&self) -> u32 {
        self.count
    }
}

fn checked_link(base: usize, local: usize) -> Result<u32, GpuArticulatedShapeBoundsError> {
    u32::try_from(
        base.checked_add(local)
            .ok_or(GpuArticulatedShapeBoundsError::Capacity)?,
    )
    .map_err(|_| GpuArticulatedShapeBoundsError::Capacity)
}

fn finite_f32(value: f64) -> Result<f32, GpuArticulatedShapeBoundsError> {
    let converted = value as f32;
    if !converted.is_finite() || (value != 0.0 && converted == 0.0) {
        return Err(GpuArticulatedShapeBoundsError::InvalidInput);
    }
    Ok(converted)
}

fn vector_radius(
    vector: Vector3<f64>,
    radius: f64,
) -> Result<[f32; 4], GpuArticulatedShapeBoundsError> {
    let out = [
        finite_f32(vector.x)?,
        finite_f32(vector.y)?,
        finite_f32(vector.z)?,
        finite_f32(radius)?,
    ];
    if radius < 0.0 {
        return Err(GpuArticulatedShapeBoundsError::InvalidInput);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::articulation::{Articulation, JointKind, JointSpec, LinkSpec};
    use crate::gpu_articulated_mass::{
        GpuArticulatedMassBatch, GpuArticulatedMassSystem, read_buffer,
    };
    use crate::gpu_articulated_state::{GpuGeneralizedState, GpuGeneralizedStateBatch};
    use crate::gpu_broad_phase::GpuPair;
    use crate::gpu_contact_pipeline::GpuContactDevice;
    use nalgebra::{DMatrix, DVector, Matrix3};

    fn model() -> Articulation {
        let links = (0..3)
            .map(|_| LinkSpec {
                mass: 1.0,
                center_of_mass: Vector3::zeros(),
                inertia: Matrix3::identity(),
            })
            .collect();
        Articulation::new(
            links,
            vec![
                JointSpec {
                    parent: 0,
                    child: 1,
                    origin: Isometry3::translation(0.5, 0.0, 0.0),
                    kind: JointKind::Revolute,
                    axis: Vector3::z(),
                    limits: None,
                },
                JointSpec {
                    parent: 1,
                    child: 2,
                    origin: Isometry3::translation(0.5, 0.0, 0.0),
                    kind: JointKind::Revolute,
                    axis: Vector3::z(),
                    limits: None,
                },
            ],
            0,
        )
        .unwrap()
    }

    #[test]
    fn moving_link_updates_shape_bounds_and_lbvh_candidates_without_pose_readback() {
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let model = model();
            let mut excluded_model = model.clone();
            excluded_model.exclude_collision_pair(0, 2).unwrap();
            let mass = GpuArticulatedMassBatch::new(
                context.device(),
                context.queue(),
                &(0..2)
                    .map(|_| GpuArticulatedMassSystem {
                        mass: DMatrix::identity(2, 2),
                        force: DVector::zeros(2),
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            let state_at = |q| GpuGeneralizedState {
                positions: DVector::from_column_slice(&[0.0, q]),
                velocities: DVector::zeros(2),
            };
            let states =
                GpuGeneralizedStateBatch::from_mass_batch(&mass, &[state_at(0.0), state_at(0.0)])
                    .unwrap();
            let poses = GpuArticulatedPoseBatch::new(
                &states,
                &[&model, &excluded_model],
                &[Isometry3::identity(), Isometry3::identity()],
            )
            .unwrap();
            let shapes = [
                GpuArticulatedBoundsShape::Sphere {
                    link: 0,
                    center: Vector3::zeros(),
                    radius: 0.5,
                },
                GpuArticulatedBoundsShape::Capsule {
                    link: 2,
                    a: Vector3::new(-0.8, 0.0, -0.3),
                    b: Vector3::new(-0.8, 0.0, 0.3),
                    radius: 0.2,
                },
                GpuArticulatedBoundsShape::Box {
                    link: 2,
                    pose: Isometry3::translation(0.0, 1.0, 0.0),
                    half_extents: Vector3::new(0.4, 0.2, 0.1),
                },
                GpuArticulatedBoundsShape::Sphere {
                    link: 1,
                    center: Vector3::new(-0.5, 0.0, 0.0),
                    radius: 0.3,
                },
            ];
            let second_environment = [shapes[0], shapes[1]];
            let bounds = GpuArticulatedShapeBoundsBatch::new(
                context.device(),
                &poses,
                &[&model, &excluded_model],
                &[&shapes, &second_environment],
            )
            .unwrap();
            let lbvh = GpuLbvh::new(context.device());
            let run = |angle| {
                states.reset(&[state_at(angle), state_at(0.0)]).unwrap();
                let mut encoder = context.device().create_command_encoder(&Default::default());
                poses.encode(&mut encoder);
                let candidates = bounds
                    .encode_candidates(context.device(), &mut encoder, &lbvh)
                    .unwrap();
                let _submission = context.queue().submit(Some(encoder.finish()));
                let bytes =
                    read_buffer(context.device(), context.queue(), bounds.bounds_buffer()).unwrap();
                let aabbs = bytes
                    .chunks_exact(size_of::<GpuAabb>())
                    .map(bytemuck::pod_read_unaligned::<GpuAabb>)
                    .collect::<Vec<_>>();
                let pairs = candidates
                    .readback(context.device(), context.queue())
                    .unwrap();
                (aabbs, pairs)
            };
            let (initial, initial_pairs) = run(0.0);
            assert_eq!(bounds.shape_count(), 6);
            assert!((initial[0].lower[0] + 0.5).abs() < 1e-5);
            assert!((initial[1].lower[0] - 0.0).abs() < 1e-5);
            assert!((initial[1].upper[2] - 0.5).abs() < 1e-5);
            assert!((initial[2].lower[0] - 0.6).abs() < 1e-5);
            assert!((initial[2].upper[1] - 1.2).abs() < 1e-5);
            assert!(
                initial_pairs
                    .iter()
                    .any(|GpuPair { a, b }| (*a, *b) == (0, 1) || (*a, *b) == (1, 0))
            );
            assert!(initial_pairs.iter().all(|pair| pair.a < 4 && pair.b < 4));
            assert!(
                !initial_pairs
                    .iter()
                    .any(|pair| { (pair.a == 1 && pair.b == 2) || (pair.a == 2 && pair.b == 1) })
            );
            assert!(initial_pairs.iter().all(|pair| pair.a != 3 && pair.b != 3));
            let (moved, moved_pairs) = run(core::f64::consts::PI);
            assert!((moved[1].lower[0] - 1.6).abs() < 1e-5);
            assert!((moved[2].upper[1] + 0.8).abs() < 1e-5);
            assert!(
                !moved_pairs
                    .iter()
                    .any(|GpuPair { a, b }| (*a, *b) == (0, 1) || (*a, *b) == (1, 0))
            );
        }
        assert!(tested > 0, "no Vulkan or DX12 GPU backend available");
    }

    #[test]
    fn lbvh_filters_many_articulated_shapes_after_joint_motion() {
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let model = model();
            let mass = GpuArticulatedMassBatch::new(
                context.device(),
                context.queue(),
                &[GpuArticulatedMassSystem {
                    mass: DMatrix::identity(2, 2),
                    force: DVector::zeros(2),
                }],
            )
            .unwrap();
            let state_at = |angle| GpuGeneralizedState {
                positions: DVector::from_column_slice(&[0.0, angle]),
                velocities: DVector::zeros(2),
            };
            let states =
                GpuGeneralizedStateBatch::from_mass_batch(&mass, &[state_at(0.0)]).unwrap();
            let poses =
                GpuArticulatedPoseBatch::new(&states, &[&model], &[Isometry3::identity()]).unwrap();
            let mut shapes = Vec::new();
            for index in 0..35 {
                shapes.push(GpuArticulatedBoundsShape::Sphere {
                    link: 0,
                    center: Vector3::new(f64::from(index) * 10.0, 0.0, 0.0),
                    radius: 0.25,
                });
            }
            for index in 0..35 {
                shapes.push(GpuArticulatedBoundsShape::Sphere {
                    link: 2,
                    center: Vector3::new(f64::from(index) * 10.0 - 1.0, 0.0, 0.0),
                    radius: 0.25,
                });
            }
            let bounds = GpuArticulatedShapeBoundsBatch::new(
                context.device(),
                &poses,
                &[&model],
                &[&shapes],
            )
            .unwrap();
            let lbvh = GpuLbvh::new(context.device());
            for (angle, expected) in [(0.0, 35), (core::f64::consts::PI, 0)] {
                states.reset(&[state_at(angle)]).unwrap();
                let mut encoder = context.device().create_command_encoder(&Default::default());
                poses.encode(&mut encoder);
                let candidates = bounds
                    .encode_candidates(context.device(), &mut encoder, &lbvh)
                    .unwrap();
                let _submission = context.queue().submit(Some(encoder.finish()));
                let pairs = candidates
                    .readback(context.device(), context.queue())
                    .unwrap();
                assert_eq!(pairs.len(), expected);
                if angle == 0.0 {
                    for index in 0..35 {
                        assert!(pairs.iter().any(|pair| {
                            (pair.a == index && pair.b == index + 35)
                                || (pair.b == index && pair.a == index + 35)
                        }));
                    }
                }
            }
        }
        assert!(tested > 0, "no Vulkan or DX12 GPU backend available");
    }

    #[test]
    fn rejects_nonpositive_shape_dimensions() {
        let Ok(context) = GpuContactDevice::new_with_backends(wgpu::Backends::VULKAN) else {
            return;
        };
        let model = model();
        let mass = GpuArticulatedMassBatch::new(
            context.device(),
            context.queue(),
            &[GpuArticulatedMassSystem {
                mass: DMatrix::identity(2, 2),
                force: DVector::zeros(2),
            }],
        )
        .unwrap();
        let state = GpuGeneralizedStateBatch::from_mass_batch(
            &mass,
            &[GpuGeneralizedState {
                positions: DVector::zeros(2),
                velocities: DVector::zeros(2),
            }],
        )
        .unwrap();
        let poses =
            GpuArticulatedPoseBatch::new(&state, &[&model], &[Isometry3::identity()]).unwrap();
        let shapes = [GpuArticulatedBoundsShape::Sphere {
            link: 1,
            center: Vector3::zeros(),
            radius: 0.0,
        }];
        assert!(matches!(
            GpuArticulatedShapeBoundsBatch::new(context.device(), &poses, &[&model], &[&shapes]),
            Err(GpuArticulatedShapeBoundsError::InvalidInput)
        ));
    }
}
