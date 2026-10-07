//! Sphere contacts computed directly from GPU-resident rigid-body positions.
//!
//! Pair and ground contacts remain in storage buffers for later constraint
//! kernels. Readback is only needed for diagnostics or CPU reference checks.

use core::mem::{size_of, size_of_val};
use core::ops::Range;
use core::time::Duration;
use std::borrow::Cow;
use std::sync::OnceLock;
use std::sync::mpsc;

use wgpu::util::DeviceExt;

use crate::gpu_broad_phase::{GpuAabb, GpuPair};
use crate::gpu_lbvh::{GpuLbvh, GpuLbvhError, GpuLbvhResidentPairs};
use crate::gpu_rigid_ball_joint::GpuRigidBallJointSolver;
use crate::gpu_rigid_shape::{
    GpuRigidShape, GpuRigidShapeData, convex_edges, convex_face_normals, mesh_bvh_nodes,
};
use crate::gpu_rigid_state::GpuRigidStateSession;
use crate::gpu_sphere_contact::GpuSphereContact;
use crate::material::ColliderMaterial;
use crate::sleep::SleepSettings;

fn reuse_pipeline_cache(
    previous: Option<&OnceLock<wgpu::ComputePipeline>>,
) -> OnceLock<wgpu::ComputePipeline> {
    let cache = OnceLock::new();
    if let Some(pipeline) = previous.and_then(OnceLock::get) {
        let _ = cache.set(pipeline.clone());
    }
    cache
}

/// Reciprocal collision membership and filter masks for one resident collider.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuRigidCollisionGroups {
    /// Collision layers occupied by this collider.
    pub memberships: u32,
    /// Layers with which this collider may interact.
    pub filter: u32,
}

impl Default for GpuRigidCollisionGroups {
    fn default() -> Self {
        Self {
            memberships: u32::MAX,
            filter: u32::MAX,
        }
    }
}

impl GpuRigidCollisionGroups {
    /// Whether both colliders permit the pair.
    pub fn allows(self, other: Self) -> bool {
        self.memberships & other.filter != 0 && other.memberships & self.filter != 0
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ContactParams {
    body_count: u32,
    pair_count: u32,
    ground_half_extent: f32,
    ground_enabled: u32,
    ground_memberships: u32,
    ground_filter: u32,
    padding: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuSleepParams {
    thresholds: [f32; 4],
    counts: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuSleepState {
    previous_position_idle: [f32; 4],
    flags: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuColliderMaterial {
    coefficients: [f32; 4],
    rules_enabled: [u32; 4],
}

impl GpuColliderMaterial {
    fn from_material(material: ColliderMaterial) -> Option<Self> {
        let friction = material.friction as f32;
        let restitution = material.restitution as f32;
        let product_safe_limit = f32::MAX.sqrt();
        (material.is_valid()
            && friction.is_finite()
            && restitution.is_finite()
            && friction <= product_safe_limit
            && restitution <= product_safe_limit)
            .then_some(Self {
                coefficients: [friction, restitution, 0.0, 0.0],
                rules_enabled: [
                    material.friction_combine_rule as u32,
                    material.restitution_combine_rule as u32,
                    1,
                    0,
                ],
            })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct IslandRange {
    body_offset: u32,
    body_count: u32,
    pair_offset: u32,
    pair_count: u32,
}

struct IslandTopology {
    ranges: Vec<IslandRange>,
    indices: Vec<u32>,
}

fn max_island_contact_slots(
    topology: &IslandTopology,
    ground_enabled: bool,
    pair_stride: u32,
    ground_stride: u32,
) -> Result<u32, GpuRigidSphereContactError> {
    topology
        .ranges
        .iter()
        .try_fold(0, |best, range| {
            let ground = if ground_enabled {
                range.body_count.checked_mul(ground_stride)?
            } else {
                0
            };
            Some(
                best.max(
                    range
                        .pair_count
                        .checked_mul(pair_stride)?
                        .checked_add(ground)?,
                ),
            )
        })
        .ok_or(GpuRigidSphereContactError::Capacity)
}

fn build_islands(
    body_count: usize,
    pairs: &[GpuPair],
    ground_enabled: bool,
) -> Result<IslandTopology, GpuRigidSphereContactError> {
    fn root(parents: &mut [usize], mut index: usize) -> usize {
        while parents[index] != index {
            parents[index] = parents[parents[index]];
            index = parents[index];
        }
        index
    }

    let mut parents = (0..body_count).collect::<Vec<_>>();
    for pair in pairs {
        let a = root(&mut parents, pair.a as usize);
        let b = root(&mut parents, pair.b as usize);
        if a != b {
            parents[a.max(b)] = a.min(b);
        }
    }
    let mut by_root = vec![usize::MAX; body_count];
    let mut body_groups: Vec<Vec<u32>> = Vec::new();
    let mut pair_groups: Vec<Vec<u32>> = Vec::new();
    for index in 0..body_count {
        let component = root(&mut parents, index);
        let island = if by_root[component] == usize::MAX {
            let next = body_groups.len();
            by_root[component] = next;
            body_groups.push(Vec::new());
            pair_groups.push(Vec::new());
            next
        } else {
            by_root[component]
        };
        body_groups[island].push(index as u32);
    }
    for (index, pair) in pairs.iter().enumerate() {
        let component = root(&mut parents, pair.a as usize);
        pair_groups[by_root[component]].push(index as u32);
    }
    let mut ranges = Vec::new();
    let mut body_indices = Vec::new();
    let mut pair_indices = Vec::new();
    for (bodies, pair_ids) in body_groups.into_iter().zip(pair_groups) {
        if !ground_enabled && pair_ids.is_empty() {
            continue;
        }
        let body_offset =
            u32::try_from(body_indices.len()).map_err(|_| GpuRigidSphereContactError::Capacity)?;
        let pair_offset =
            u32::try_from(pair_indices.len()).map_err(|_| GpuRigidSphereContactError::Capacity)?;
        let body_len =
            u32::try_from(bodies.len()).map_err(|_| GpuRigidSphereContactError::Capacity)?;
        let pair_len =
            u32::try_from(pair_ids.len()).map_err(|_| GpuRigidSphereContactError::Capacity)?;
        ranges.push(IslandRange {
            body_offset,
            body_count: body_len,
            pair_offset,
            pair_count: pair_len,
        });
        body_indices.extend(bodies);
        pair_indices.extend(pair_ids);
    }
    let pair_base =
        u32::try_from(body_indices.len()).map_err(|_| GpuRigidSphereContactError::Capacity)?;
    for range in &mut ranges {
        range.pair_offset = range
            .pair_offset
            .checked_add(pair_base)
            .ok_or(GpuRigidSphereContactError::Capacity)?;
    }
    let total_indices = body_indices
        .len()
        .checked_add(pair_indices.len())
        .ok_or(GpuRigidSphereContactError::Capacity)?;
    let _ = u32::try_from(total_indices).map_err(|_| GpuRigidSphereContactError::Capacity)?;
    body_indices.extend(pair_indices);
    Ok(IslandTopology {
        ranges,
        indices: body_indices,
    })
}

fn build_body_pair_adjacency(
    body_count: usize,
    pairs: &[GpuPair],
) -> Result<(Vec<u32>, Vec<u32>), GpuRigidSphereContactError> {
    let mut offsets = vec![0u32; body_count + 1];
    for pair in pairs {
        for body in [pair.a, pair.b] {
            offsets[body as usize + 1] = offsets[body as usize + 1]
                .checked_add(1)
                .ok_or(GpuRigidSphereContactError::Capacity)?;
        }
    }
    for index in 1..offsets.len() {
        offsets[index] = offsets[index]
            .checked_add(offsets[index - 1])
            .ok_or(GpuRigidSphereContactError::Capacity)?;
    }
    let mut indices = vec![0u32; offsets[body_count] as usize];
    let mut next = offsets[..body_count].to_vec();
    for (pair_index, pair) in pairs.iter().enumerate() {
        let pair_index =
            u32::try_from(pair_index).map_err(|_| GpuRigidSphereContactError::Capacity)?;
        for body in [pair.a, pair.b] {
            let slot = &mut next[body as usize];
            indices[*slot as usize] = pair_index;
            *slot += 1;
        }
    }
    Ok((offsets, indices))
}

/// Invalid sphere geometry, device capacity, or a failed diagnostic readback.
#[derive(Debug, thiserror::Error)]
pub enum GpuRigidSphereContactError {
    /// A radius, pair index, or finite ground extent was invalid.
    #[error("invalid GPU rigid sphere contact input")]
    InvalidInput,
    /// The input exceeds device buffer or dispatch capacity.
    #[error("GPU rigid sphere contacts exceed device capacity")]
    Capacity,
    /// GPU broad-phase generation or candidate transfer failed.
    #[error("GPU rigid sphere broad phase failed: {0}")]
    BroadPhase(#[from] GpuLbvhError),
    /// Mapping the contact result failed.
    #[error("GPU rigid sphere contact readback failed: {0}")]
    Readback(String),
    /// A speculative convex distance query did not converge.
    #[error("GPU speculative convex distance query did not converge")]
    SpeculativeUnconverged,
}

/// Four-byte GPU summary of speculative distance query failures.
#[derive(Debug)]
pub struct GpuRigidSpeculativeStatus {
    buffer: wgpu::Buffer,
}
impl GpuRigidSpeculativeStatus {
    fn encode(
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        contacts: &wgpu::Buffer,
        count: u32,
    ) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera speculative query status"),
            contents: bytemuck::bytes_of(&0u32),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        if count > 0 {
            let counts = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tessera speculative status counts"),
                contents: bytemuck::cast_slice(&[count, 0u32, 0, 0]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let buffers = [contacts, &buffer, &counts];
            let entries = buffers
                .iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect::<Vec<_>>();
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Tessera speculative status inputs"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            });
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
        }
        Self { buffer }
    }
    /// Read only the summary; contacts and body states remain resident.
    pub fn readback(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<(), GpuRigidSphereContactError> {
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera speculative status readback"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, 4);
        let _ = queue.submit(Some(encoder.finish()));
        map_readback(device, &staging)?;
        let view = staging.slice(..).get_mapped_range();
        let status = bytemuck::pod_read_unaligned::<u32>(&view);
        drop(view);
        staging.unmap();
        if status == 0 {
            Ok(())
        } else {
            Err(GpuRigidSphereContactError::SpeculativeUnconverged)
        }
    }
}

/// Diagnostic contact values in the same stable order as the supplied pairs.
#[derive(Debug)]
pub struct GpuRigidSphereContactReadback {
    /// Candidate pair and its contact or zeroed miss slot.
    pub pairs: Vec<(GpuPair, GpuSphereContact)>,
    /// Three additional pair slots per candidate in primitive worlds.
    pub pair_extra: Vec<[GpuSphereContact; 3]>,
    /// One ground contact or zeroed miss slot per body, when ground is enabled.
    pub ground: Vec<GpuSphereContact>,
    /// Additional ground manifold contacts, three slots per primitive body.
    pub ground_extra: Vec<[GpuSphereContact; 3]>,
}

/// Contact buffers produced directly from GPU-resident LBVH candidates.
///
/// Pair contacts use the candidate capacity as the primary-contact stride:
/// primary contacts occupy `[0, pair_capacity)`, followed by three optional
/// manifold points per pair. Only the first GPU counter word's worth of pair
/// slots is valid after the encoded pass completes.
#[derive(Clone, Debug)]
pub struct GpuRigidCandidateContacts {
    /// Pair contact storage, including optional primitive manifold points.
    pub pairs: wgpu::Buffer,
    /// Ground contact storage owned by the source contact session.
    pub ground: wgpu::Buffer,
    /// Capacity of the candidate pair buffer.
    pub pair_capacity: u32,
    /// One sphere point or four primitive manifold points per pair.
    pub pair_stride: u32,
    /// One sphere point or four primitive manifold points per body.
    pub ground_stride: u32,
}

impl GpuRigidCandidateContacts {
    /// Read only valid pair contacts after the encoded work is submitted.
    pub fn readback_pairs(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        candidates: &GpuLbvhResidentPairs,
    ) -> Result<GpuRigidSphereContactReadback, GpuRigidSphereContactError> {
        if candidates.pair_capacity != self.pair_capacity {
            return Err(GpuRigidSphereContactError::InvalidInput);
        }
        let pairs = candidates.readback(device, queue)?;
        if pairs.is_empty() {
            return Ok(GpuRigidSphereContactReadback {
                pairs: Vec::new(),
                pair_extra: Vec::new(),
                ground: Vec::new(),
                ground_extra: Vec::new(),
            });
        }
        let contact_bytes = size_of::<GpuSphereContact>() as u64;
        let primary_bytes = pairs.len() as u64 * contact_bytes;
        let extra_bytes = if self.pair_stride == 4 {
            primary_bytes * 3
        } else {
            0
        };
        let total = primary_bytes + extra_bytes;
        if total > device.limits().max_buffer_size {
            return Err(GpuRigidSphereContactError::Capacity);
        }
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera resident LBVH contact readback"),
            size: total,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera resident LBVH contact readback encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.pairs, 0, &staging, 0, primary_bytes);
        if extra_bytes > 0 {
            encoder.copy_buffer_to_buffer(
                &self.pairs,
                u64::from(self.pair_capacity) * contact_bytes,
                &staging,
                primary_bytes,
                extra_bytes,
            );
        }
        let _submission = queue.submit(Some(encoder.finish()));
        map_readback(device, &staging)?;
        let view = staging.slice(..).get_mapped_range();
        let contacts = view[..primary_bytes as usize]
            .chunks_exact(size_of::<GpuSphereContact>())
            .map(bytemuck::pod_read_unaligned);
        let pair_extra = view[primary_bytes as usize..]
            .chunks_exact(size_of::<[GpuSphereContact; 3]>())
            .map(bytemuck::pod_read_unaligned)
            .collect();
        let pairs = pairs.into_iter().zip(contacts).collect();
        drop(view);
        staging.unmap();
        Ok(GpuRigidSphereContactReadback {
            pairs,
            pair_extra,
            ground: Vec::new(),
            ground_extra: Vec::new(),
        })
    }
}

/// Persistent contact buffers bound to one `GpuRigidStateSession`.
#[derive(Debug)]
pub struct GpuRigidSphereContacts {
    speculative_status_pipeline: OnceLock<wgpu::ComputePipeline>,
    ray_query_pipeline: OnceLock<wgpu::ComputePipeline>,
    point_query_pipeline: OnceLock<wgpu::ComputePipeline>,
    state_buffer: wgpu::Buffer,
    radii: Vec<f32>,
    radii_buffer: wgpu::Buffer,
    collision_groups: Vec<GpuRigidCollisionGroups>,
    collision_groups_buffer: wgpu::Buffer,
    ground_collision_groups: GpuRigidCollisionGroups,
    params: wgpu::Buffer,
    pairs: Vec<GpuPair>,
    pair_buffer: wgpu::Buffer,
    pair_contacts: wgpu::Buffer,
    ground_contacts: wgpu::Buffer,
    materials: wgpu::Buffer,
    sleep_timers: wgpu::Buffer,
    moving_kinematic: wgpu::Buffer,
    kinematic_activity_pipeline: wgpu::ComputePipeline,
    island_ranges: wgpu::Buffer,
    island_indices: wgpu::Buffer,
    body_pair_offsets: wgpu::Buffer,
    body_pair_indices: wgpu::Buffer,
    island_count: u32,
    max_island_contacts: u32,
    pair_stride: u32,
    ground_stride: u32,
    ground_enabled: bool,
    ground_half_extent: f32,
    body_count: usize,
    pair_pipeline: wgpu::ComputePipeline,
    ground_pipeline: wgpu::ComputePipeline,
    aabb_pipeline: wgpu::ComputePipeline,
    bounds_margin_pipeline: wgpu::ComputePipeline,
    sleep_pipeline: wgpu::ComputePipeline,
    candidate_sleep_pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    primitive: Option<GpuRigidPrimitivePipelines>,
}

#[derive(Debug)]
struct GpuRigidPrimitivePipelines {
    speculative_pipeline: OnceLock<wgpu::ComputePipeline>,
    shape_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    edge_buffer: wgpu::Buffer,
    aabb_pipeline: wgpu::ComputePipeline,
    pair_pipeline: wgpu::ComputePipeline,
    specialized_pair_pipelines: Vec<(u32, wgpu::ComputePipeline)>,
    ground_pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
}

struct GpuRigidPrimitiveData<'a> {
    shapes: &'a [GpuRigidShapeData],
    vertices: &'a [[f32; 4]],
    edges: &'a [[u32; 4]],
}

impl GpuRigidSphereContacts {
    /// Bind sphere radii and initial candidate pairs to a live rigid-state buffer.
    pub fn new(
        device: &wgpu::Device,
        session: &GpuRigidStateSession,
        radii: &[f32],
        pairs: &[GpuPair],
        ground_half_extent: Option<f32>,
    ) -> Result<Self, GpuRigidSphereContactError> {
        Self::new_inner(
            device,
            session,
            radii,
            pairs,
            ground_half_extent,
            None,
            None,
        )
    }

    /// Bind primitive colliders to live rigid-body states.
    pub fn new_with_shapes(
        device: &wgpu::Device,
        session: &GpuRigidStateSession,
        shapes: &[GpuRigidShape],
        pairs: &[GpuPair],
        ground_half_extent: Option<f32>,
    ) -> Result<Self, GpuRigidSphereContactError> {
        Self::new_with_shapes_reusing(device, session, shapes, pairs, ground_half_extent, None)
    }

    /// Rebuild shape buffers while reusing primitive kernels from the same device.
    pub(crate) fn new_with_shapes_reusing(
        device: &wgpu::Device,
        session: &GpuRigidStateSession,
        shapes: &[GpuRigidShape],
        pairs: &[GpuPair],
        ground_half_extent: Option<f32>,
        previous: Option<&Self>,
    ) -> Result<Self, GpuRigidSphereContactError> {
        let radii = shapes
            .iter()
            .map(|shape| {
                shape
                    .bounding_radius()
                    .ok_or(GpuRigidSphereContactError::InvalidInput)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut vertices = Vec::<[f32; 4]>::new();
        let mut edges = Vec::<[u32; 4]>::new();
        let mut packed = Vec::with_capacity(shapes.len());
        for shape in shapes {
            let first =
                u32::try_from(vertices.len()).map_err(|_| GpuRigidSphereContactError::Capacity)?;
            // Segment bounds share the triangle BVH layout; repeated endpoints are
            // used only for bounds construction, never as contact triangles.
            let primitives = match shape {
                GpuRigidShape::TriangleMesh {
                    vertices,
                    triangles,
                } => Some((vertices, Cow::Borrowed(triangles.as_slice()))),
                GpuRigidShape::Polyline { vertices, segments } => Some((
                    vertices,
                    Cow::Owned(segments.iter().map(|&[a, b]| [a, b, b]).collect::<Vec<_>>()),
                )),
                _ => None,
            };
            if let GpuRigidShape::Convex { vertices: hull } = shape {
                vertices.extend(hull.iter().map(|point| [point[0], point[1], point[2], 0.0]));
                let first_normal = u32::try_from(vertices.len())
                    .map_err(|_| GpuRigidSphereContactError::Capacity)?;
                let normals = convex_face_normals(hull);
                let normal_count = u32::try_from(normals.len())
                    .map_err(|_| GpuRigidSphereContactError::Capacity)?;
                if normal_count < 4 {
                    return Err(GpuRigidSphereContactError::InvalidInput);
                }
                for normal in &normals {
                    // Preserve the normal XYZ contract and cache the supporting
                    // plane offset in the previously unused fourth component.
                    let offset = hull
                        .iter()
                        .map(|point| normal.iter().zip(point).map(|(n, v)| n * v).sum::<f32>())
                        .fold(f32::NEG_INFINITY, f32::max);
                    if !offset.is_finite() {
                        return Err(GpuRigidSphereContactError::InvalidInput);
                    }
                    vertices.push([normal[0], normal[1], normal[2], offset]);
                }
                let first_edge =
                    u32::try_from(edges.len()).map_err(|_| GpuRigidSphereContactError::Capacity)?;
                let hull_edges = convex_edges(hull, &normals);
                let edge_count = u32::try_from(hull_edges.len())
                    .map_err(|_| GpuRigidSphereContactError::Capacity)?;
                if edge_count < 3 {
                    return Err(GpuRigidSphereContactError::InvalidInput);
                }
                for [a, b] in hull_edges {
                    edges.push([
                        first
                            .checked_add(a)
                            .ok_or(GpuRigidSphereContactError::Capacity)?,
                        first
                            .checked_add(b)
                            .ok_or(GpuRigidSphereContactError::Capacity)?,
                        0,
                        0,
                    ]);
                }
                packed.push(shape.packed(
                    first,
                    first_normal,
                    normal_count,
                    first_edge,
                    edge_count,
                ));
            } else if let Some((mesh_vertices, triangles)) = primitives {
                let node_count = triangles
                    .len()
                    .checked_mul(2)
                    .and_then(|count| count.checked_sub(1))
                    .ok_or(GpuRigidSphereContactError::Capacity)?;
                let planned_edges = edges
                    .len()
                    .checked_add(triangles.len())
                    .and_then(|count| count.checked_add(node_count.checked_mul(3)?))
                    .ok_or(GpuRigidSphereContactError::Capacity)?;
                let planned_bytes = planned_edges
                    .checked_mul(size_of::<[u32; 4]>())
                    .and_then(|bytes| u64::try_from(bytes).ok())
                    .ok_or(GpuRigidSphereContactError::Capacity)?;
                let limits = device.limits();
                if planned_edges > u32::MAX as usize
                    || planned_bytes > u64::from(limits.max_storage_buffer_binding_size)
                    || planned_bytes > limits.max_buffer_size
                {
                    return Err(GpuRigidSphereContactError::Capacity);
                }
                vertices.extend(
                    mesh_vertices
                        .iter()
                        .map(|point| [point[0], point[1], point[2], 0.0]),
                );
                let first_triangle =
                    u32::try_from(edges.len()).map_err(|_| GpuRigidSphereContactError::Capacity)?;
                for triangle in triangles.iter() {
                    edges.push([
                        first
                            .checked_add(triangle[0])
                            .ok_or(GpuRigidSphereContactError::Capacity)?,
                        first
                            .checked_add(triangle[1])
                            .ok_or(GpuRigidSphereContactError::Capacity)?,
                        first
                            .checked_add(triangle[2])
                            .ok_or(GpuRigidSphereContactError::Capacity)?,
                        0,
                    ]);
                }
                let first_node =
                    u32::try_from(edges.len()).map_err(|_| GpuRigidSphereContactError::Capacity)?;
                let nodes = mesh_bvh_nodes(mesh_vertices, &triangles);
                let node_count =
                    u32::try_from(nodes.len()).map_err(|_| GpuRigidSphereContactError::Capacity)?;
                for node in nodes {
                    edges.push([
                        node.lower[0].to_bits(),
                        node.lower[1].to_bits(),
                        node.lower[2].to_bits(),
                        0,
                    ]);
                    edges.push([
                        node.upper[0].to_bits(),
                        node.upper[1].to_bits(),
                        node.upper[2].to_bits(),
                        0,
                    ]);
                    edges.push([
                        node.triangle
                            .map(|triangle| {
                                let triangle = u32::try_from(triangle)
                                    .map_err(|_| GpuRigidSphereContactError::Capacity)?;
                                first_triangle
                                    .checked_add(triangle)
                                    .ok_or(GpuRigidSphereContactError::Capacity)
                            })
                            .transpose()?
                            .unwrap_or(u32::MAX),
                        u32::try_from(node.escape)
                            .map_err(|_| GpuRigidSphereContactError::Capacity)?,
                        0,
                        0,
                    ]);
                }
                packed.push(shape.packed(first, first_node, node_count, first_triangle, 0));
            } else {
                packed.push(shape.packed(first, 0, 0, 0, 0));
            }
        }
        let _vertex_count =
            u32::try_from(vertices.len()).map_err(|_| GpuRigidSphereContactError::Capacity)?;
        let _edge_count =
            u32::try_from(edges.len()).map_err(|_| GpuRigidSphereContactError::Capacity)?;
        Self::new_inner(
            device,
            session,
            &radii,
            pairs,
            ground_half_extent,
            Some(GpuRigidPrimitiveData {
                shapes: &packed,
                vertices: &vertices,
                edges: &edges,
            }),
            previous,
        )
    }

    fn new_inner(
        device: &wgpu::Device,
        session: &GpuRigidStateSession,
        radii: &[f32],
        pairs: &[GpuPair],
        ground_half_extent: Option<f32>,
        primitive_data: Option<GpuRigidPrimitiveData<'_>>,
        previous: Option<&Self>,
    ) -> Result<Self, GpuRigidSphereContactError> {
        if radii.len() != session.len()
            || radii
                .iter()
                .any(|radius| !radius.is_finite() || *radius <= 0.0)
            || ground_half_extent.is_some_and(|half| !half.is_finite() || half <= 0.0)
            || pairs.iter().any(|pair| {
                pair.a == pair.b
                    || pair.a as usize >= radii.len()
                    || pair.b as usize >= radii.len()
                    || !(radii[pair.a as usize] + radii[pair.b as usize]).is_finite()
            })
        {
            return Err(GpuRigidSphereContactError::InvalidInput);
        }
        let body_count =
            u32::try_from(radii.len()).map_err(|_| GpuRigidSphereContactError::Capacity)?;
        let pair_count =
            u32::try_from(pairs.len()).map_err(|_| GpuRigidSphereContactError::Capacity)?;
        let limits = device.limits();
        let pair_bytes = size_of_val(pairs) as u64;
        let radius_bytes = size_of_val(radii) as u64;
        let collision_group_bytes =
            radii.len() as u64 * size_of::<GpuRigidCollisionGroups>() as u64;
        let shape_bytes = primitive_data
            .as_ref()
            .map_or(0, |data| size_of_val(data.shapes) as u64);
        let vertex_bytes = primitive_data
            .as_ref()
            .map_or(0, |data| size_of_val(data.vertices) as u64);
        let edge_bytes = primitive_data
            .as_ref()
            .map_or(0, |data| size_of_val(data.edges) as u64);
        let pair_contact_bytes = pairs.len() as u64 * size_of::<GpuSphereContact>() as u64;
        let pair_stride = if primitive_data.is_some() { 4 } else { 1 };
        let pair_extra_bytes = if pair_stride == 4 {
            pair_contact_bytes
                .checked_mul(3)
                .ok_or(GpuRigidSphereContactError::Capacity)?
        } else {
            0
        };
        let pair_storage_bytes = pair_contact_bytes
            .checked_add(pair_extra_bytes)
            .ok_or(GpuRigidSphereContactError::Capacity)?;
        let ground_contact_bytes = radii.len() as u64 * size_of::<GpuSphereContact>() as u64;
        let ground_stride = if primitive_data.is_some() { 4 } else { 1 };
        let ground_extra_bytes = if ground_stride == 4 && ground_half_extent.is_some() {
            ground_contact_bytes
                .checked_mul(3)
                .ok_or(GpuRigidSphereContactError::Capacity)?
        } else {
            0
        };
        let ground_storage_bytes = ground_contact_bytes
            .checked_add(ground_extra_bytes)
            .ok_or(GpuRigidSphereContactError::Capacity)?;
        if pair_count.div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || body_count.div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || [
                pair_bytes,
                radius_bytes,
                collision_group_bytes,
                shape_bytes,
                vertex_bytes,
                edge_bytes,
                pair_storage_bytes,
                ground_storage_bytes,
            ]
            .iter()
            .any(|size| {
                *size > u64::from(limits.max_storage_buffer_binding_size)
                    || *size > limits.max_buffer_size
            })
        {
            return Err(GpuRigidSphereContactError::Capacity);
        }
        let topology = build_islands(radii.len(), pairs, ground_half_extent.is_some())?;
        let max_island_contacts = max_island_contact_slots(
            &topology,
            ground_half_extent.is_some(),
            pair_stride,
            ground_stride,
        )?;
        let (body_pair_offsets, body_pair_indices) = build_body_pair_adjacency(radii.len(), pairs)?;
        let island_count = u32::try_from(topology.ranges.len())
            .map_err(|_| GpuRigidSphereContactError::Capacity)?;
        let island_bytes = size_of_val(topology.ranges.as_slice()) as u64;
        let index_bytes = size_of_val(topology.indices.as_slice()) as u64;
        let adjacency_bytes = [
            size_of_val(body_pair_offsets.as_slice()) as u64,
            size_of_val(body_pair_indices.as_slice()) as u64,
        ];
        if island_count.div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || [
                island_bytes,
                index_bytes,
                adjacency_bytes[0],
                adjacency_bytes[1],
            ]
            .iter()
            .any(|size| {
                *size > u64::from(limits.max_storage_buffer_binding_size)
                    || *size > limits.max_buffer_size
            })
        {
            return Err(GpuRigidSphereContactError::Capacity);
        }
        let island_ranges = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera resident sphere island ranges"),
            contents: if topology.ranges.is_empty() {
                bytemuck::bytes_of(&IslandRange {
                    body_offset: 0,
                    body_count: 0,
                    pair_offset: 0,
                    pair_count: 0,
                })
            } else {
                bytemuck::cast_slice(&topology.ranges)
            },
            usage: wgpu::BufferUsages::STORAGE,
        });
        let island_indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera resident sphere island indices"),
            contents: if topology.indices.is_empty() {
                bytemuck::bytes_of(&0u32)
            } else {
                bytemuck::cast_slice(&topology.indices)
            },
            usage: wgpu::BufferUsages::STORAGE,
        });
        let body_pair_offsets = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera resident sphere body pair offsets"),
            contents: bytemuck::cast_slice(&body_pair_offsets),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let body_pair_indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera resident sphere body pair indices"),
            contents: if body_pair_indices.is_empty() {
                bytemuck::bytes_of(&0u32)
            } else {
                bytemuck::cast_slice(&body_pair_indices)
            },
            usage: wgpu::BufferUsages::STORAGE,
        });
        let pair_buffer = if pairs.is_empty() {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Tessera empty rigid sphere pairs"),
                size: size_of::<GpuPair>() as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        } else {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tessera rigid sphere pairs"),
                contents: bytemuck::cast_slice(pairs),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let radii_buffer = if radii.is_empty() {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Tessera empty rigid sphere radii"),
                size: size_of::<f32>() as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        } else {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tessera rigid sphere radii"),
                contents: bytemuck::cast_slice(radii),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let collision_groups = vec![GpuRigidCollisionGroups::default(); radii.len()];
        let default_groups = [GpuRigidCollisionGroups::default()];
        let collision_groups_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tessera resident collision groups"),
                contents: bytemuck::cast_slice(if collision_groups.is_empty() {
                    &default_groups[..]
                } else {
                    &collision_groups[..]
                }),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        let pair_contacts = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera rigid sphere pair contacts"),
            size: pair_storage_bytes.max(size_of::<GpuSphereContact>() as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let ground_contacts = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera rigid sphere ground contacts"),
            size: ground_storage_bytes.max(size_of::<GpuSphereContact>() as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let material_slots = radii
            .len()
            .checked_add(1)
            .ok_or(GpuRigidSphereContactError::Capacity)?;
        let material_bytes = material_slots as u64 * size_of::<GpuColliderMaterial>() as u64;
        let sleep_bytes = radii.len() as u64 * size_of::<GpuSleepState>() as u64;
        if material_bytes > u64::from(limits.max_storage_buffer_binding_size)
            || material_bytes > limits.max_buffer_size
            || sleep_bytes > u64::from(limits.max_storage_buffer_binding_size)
            || sleep_bytes > limits.max_buffer_size
        {
            return Err(GpuRigidSphereContactError::Capacity);
        }
        let materials = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera resident sphere materials"),
            contents: bytemuck::cast_slice(&vec![GpuColliderMaterial::default(); material_slots]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let sleep_timers = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera resident sphere sleep timers"),
            contents: bytemuck::cast_slice(&vec![GpuSleepState::default(); radii.len().max(1)]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera rigid sphere contact params"),
            contents: bytemuck::bytes_of(&ContactParams {
                body_count,
                pair_count,
                ground_half_extent: ground_half_extent.unwrap_or(0.0),
                ground_enabled: u32::from(ground_half_extent.is_some()),
                ground_memberships: u32::MAX,
                ground_filter: u32::MAX,
                padding: [0; 2],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera rigid sphere contacts"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_rigid_sphere_contact.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Tessera rigid sphere contact layout"),
            entries: &[0, 1, 2, 3, 4, 5, 6].map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 5 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: binding != 3 && binding != 4,
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Tessera rigid sphere contact pipeline layout"),
            bind_group_layouts: &[&layout],
            immediate_size: 0,
        });
        let make_pipeline = |entry_point| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Tessera rigid sphere contact pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pair_pipeline = make_pipeline("pair_contacts_main");
        let ground_pipeline = make_pipeline("ground_contacts_main");
        let aabb_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera resident sphere AABBs"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_rigid_sphere_aabb.wgsl").into()),
        });
        let aabb_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tessera resident sphere AABB pipeline"),
            layout: None,
            module: &aabb_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bounds_margin_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera speculative bounds expansion"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_rigid_bounds_margin.wgsl").into()),
        });
        let bounds_margin_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Tessera speculative bounds expansion"),
                layout: None,
                module: &bounds_margin_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let sleep_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera resident sphere sleep"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_rigid_sphere_sleep.wgsl").into()),
        });
        let activity_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera kinematic activity snapshot"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("gpu_rigid_kinematic_activity.wgsl").into(),
            ),
        });
        let kinematic_activity_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Tessera kinematic activity snapshot"),
                layout: None,
                module: &activity_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let moving_kinematic = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera moving kinematic flags"),
            size: (radii.len() as u64 * 4).max(4),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let sleep_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tessera resident sphere sleep pipeline"),
            layout: None,
            module: &sleep_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let candidate_sleep_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera resident candidate sleep"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_rigid_candidate_sleep.wgsl").into()),
        });
        let candidate_sleep_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Tessera resident candidate sleep pipeline"),
                layout: None,
                module: &candidate_sleep_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera rigid sphere contact inputs"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: session.state_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: radii_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: pair_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: pair_contacts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: ground_contacts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: collision_groups_buffer.as_entire_binding(),
                },
            ],
        });
        let primitive = primitive_data.map(|data| {
            let shapes = data.shapes;
            let mesh_hull_needed = shapes.iter().any(|shape| matches!(shape.kind(), 6 | 7))
                && shapes.iter().any(|shape| shape.kind() == 5);
            let polyline_analytic_needed = shapes.iter().any(|shape| shape.kind() == 7)
                && shapes.iter().any(|shape| matches!(shape.kind(), 3 | 4));
            let thin_surface_needed = shapes
                .iter()
                .filter(|shape| matches!(shape.kind(), 6 | 7))
                .count()
                >= 2;
            let shape_buffer = if shapes.is_empty() {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Tessera empty resident primitive shapes"),
                    size: size_of::<GpuRigidShapeData>() as u64,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                })
            } else {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Tessera resident primitive shapes"),
                    contents: bytemuck::cast_slice(shapes),
                    usage: wgpu::BufferUsages::STORAGE,
                })
            };
            let vertices = data.vertices;
            let vertex_buffer = if vertices.is_empty() {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Tessera empty resident convex vertices"),
                    size: size_of::<[f32; 4]>() as u64,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                })
            } else {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Tessera resident convex vertices"),
                    contents: bytemuck::cast_slice(vertices),
                    usage: wgpu::BufferUsages::STORAGE,
                })
            };
            let edges = data.edges;
            let edge_buffer = if edges.is_empty() {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Tessera empty resident convex edges"),
                    size: size_of::<[u32; 4]>() as u64,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                })
            } else {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Tessera resident convex edges"),
                    contents: bytemuck::cast_slice(edges),
                    usage: wgpu::BufferUsages::STORAGE,
                })
            };
            let (
                aabb_pipeline,
                pair_pipeline,
                specialized_pair_pipelines,
                ground_pipeline,
                primitive_layout,
            ) = if let Some(previous) = previous.and_then(|contacts| contacts.primitive.as_ref()) {
                let primitive_layout = previous.pair_pipeline.get_bind_group_layout(0);
                let specialized_pair_pipelines = [
                    (1u32, mesh_hull_needed),
                    (2, polyline_analytic_needed),
                    (3, thin_surface_needed),
                ]
                .into_iter()
                .filter(|(_, needed)| *needed)
                .map(|(mode, _)| {
                    let cached = previous
                        .specialized_pair_pipelines
                        .iter()
                        .find(|(cached_mode, _)| *cached_mode == mode)
                        .map(|(_, pipeline)| pipeline.clone());
                    let pipeline = cached.unwrap_or_else(|| {
                        let layout =
                            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                                label: Some("Tessera added contact pipeline layout"),
                                bind_group_layouts: &[&primitive_layout],
                                immediate_size: 0,
                            });
                        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                            label: Some("Tessera added contact shader"),
                            source: wgpu::ShaderSource::Wgsl(
                                include_str!("gpu_rigid_primitive_contact.wgsl").into(),
                            ),
                        });
                        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                            label: Some("Tessera added contact pipeline"),
                            layout: Some(&layout),
                            module: &shader,
                            entry_point: Some("pair_contacts_main"),
                            compilation_options: wgpu::PipelineCompilationOptions {
                                constants: &[("MESH_HULL_MODE", f64::from(mode))],
                                ..Default::default()
                            },
                            cache: None,
                        })
                    });
                    (mode, pipeline)
                })
                .collect();
                (
                    previous.aabb_pipeline.clone(),
                    previous.pair_pipeline.clone(),
                    specialized_pair_pipelines,
                    previous.ground_pipeline.clone(),
                    primitive_layout,
                )
            } else {
                let aabb_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Tessera resident primitive AABBs"),
                    source: wgpu::ShaderSource::Wgsl(
                        include_str!("gpu_rigid_primitive_aabb.wgsl").into(),
                    ),
                });
                let aabb_pipeline =
                    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("Tessera resident primitive AABB pipeline"),
                        layout: None,
                        module: &aabb_shader,
                        entry_point: Some("main"),
                        compilation_options: Default::default(),
                        cache: None,
                    });
                let primitive_layout =
                    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("Tessera resident primitive contact layout"),
                        entries: &[0, 1, 2, 3, 4, 5, 6, 7, 8].map(|binding| {
                            wgpu::BindGroupLayoutEntry {
                                binding,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: if binding == 5 {
                                        wgpu::BufferBindingType::Uniform
                                    } else {
                                        wgpu::BufferBindingType::Storage {
                                            read_only: binding != 3 && binding != 4,
                                        }
                                    },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            }
                        }),
                    });
                let primitive_pipeline_layout =
                    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("Tessera resident primitive contact pipeline layout"),
                        bind_group_layouts: &[&primitive_layout],
                        immediate_size: 0,
                    });
                let primitive_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Tessera resident primitive contacts"),
                    source: wgpu::ShaderSource::Wgsl(
                        include_str!("gpu_rigid_primitive_contact.wgsl").into(),
                    ),
                });
                let pipeline = |entry_point, mode: u32| {
                    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("Tessera resident primitive contact pipeline"),
                        layout: Some(&primitive_pipeline_layout),
                        module: &primitive_shader,
                        entry_point: Some(entry_point),
                        compilation_options: wgpu::PipelineCompilationOptions {
                            constants: &[("MESH_HULL_MODE", f64::from(mode))],
                            ..Default::default()
                        },
                        cache: None,
                    })
                };
                let pair_pipeline = pipeline("pair_contacts_main", 0);
                let specialized_pair_pipelines = [
                    (1u32, mesh_hull_needed),
                    (2, polyline_analytic_needed),
                    (3, thin_surface_needed),
                ]
                .into_iter()
                .filter(|(_, needed)| *needed)
                .map(|(mode, _)| (mode, pipeline("pair_contacts_main", mode)))
                .collect();
                let ground_pipeline = pipeline("ground_contacts_main", 0);
                (
                    aabb_pipeline,
                    pair_pipeline,
                    specialized_pair_pipelines,
                    ground_pipeline,
                    primitive_layout,
                )
            };
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Tessera resident primitive contact inputs"),
                layout: &primitive_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: session.state_buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: shape_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: pair_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: pair_contacts.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: ground_contacts.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: vertex_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: edge_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: collision_groups_buffer.as_entire_binding(),
                    },
                ],
            });
            GpuRigidPrimitivePipelines {
                speculative_pipeline: reuse_pipeline_cache(
                    previous
                        .and_then(|p| p.primitive.as_ref())
                        .map(|p| &p.speculative_pipeline),
                ),
                shape_buffer,
                vertex_buffer,
                edge_buffer,
                aabb_pipeline,
                pair_pipeline,
                specialized_pair_pipelines,
                ground_pipeline,
                bind_group,
            }
        });
        Ok(Self {
            ray_query_pipeline: reuse_pipeline_cache(previous.map(|p| &p.ray_query_pipeline)),
            point_query_pipeline: reuse_pipeline_cache(previous.map(|p| &p.point_query_pipeline)),
            speculative_status_pipeline: reuse_pipeline_cache(
                previous.map(|p| &p.speculative_status_pipeline),
            ),
            state_buffer: session.state_buffer().clone(),
            radii: radii.to_vec(),
            radii_buffer,
            collision_groups,
            collision_groups_buffer,
            ground_collision_groups: GpuRigidCollisionGroups::default(),
            params,
            pairs: pairs.to_vec(),
            pair_buffer,
            pair_contacts,
            ground_contacts,
            materials,
            sleep_timers,
            moving_kinematic,
            kinematic_activity_pipeline,
            island_ranges,
            island_indices,
            body_pair_offsets,
            body_pair_indices,
            island_count,
            max_island_contacts,
            pair_stride,
            ground_stride,
            ground_enabled: ground_half_extent.is_some(),
            ground_half_extent: ground_half_extent.unwrap_or(0.0),
            body_count: radii.len(),
            pair_pipeline,
            ground_pipeline,
            aabb_pipeline,
            bounds_margin_pipeline,
            sleep_pipeline,
            candidate_sleep_pipeline,
            layout,
            bind_group,
            primitive,
        })
    }

    /// Refresh candidate pairs from GPU-resident body poses and collider shapes.
    ///
    /// The previous state update must already be submitted. AABBs and LBVH
    /// traversal run on the GPU; the compact pair list is read back so the
    /// current CPU-built solver islands can be updated. Rigid state and
    /// materials remain on the GPU.
    pub fn refresh_candidate_pairs_from_state(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        broad_phase: &GpuLbvh,
    ) -> Result<(), GpuRigidSphereContactError> {
        self.refresh_candidate_pairs_from_state_impl(device, queue, broad_phase, None, false, 0.0)
    }

    /// Refresh candidates while excluding pairs from different environments.
    pub fn refresh_candidate_pairs_from_state_grouped(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        broad_phase: &GpuLbvh,
        environment_ids: &[u32],
    ) -> Result<(), GpuRigidSphereContactError> {
        if environment_ids.len() != self.body_count {
            return Err(GpuRigidSphereContactError::InvalidInput);
        }
        self.refresh_candidate_pairs_from_state_impl(
            device,
            queue,
            broad_phase,
            Some(environment_ids),
            false,
            0.0,
        )
    }

    /// Refresh candidates with environment isolation and reciprocal masks on the GPU.
    pub fn refresh_candidate_pairs_from_state_filtered(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        broad_phase: &GpuLbvh,
        environment_ids: &[u32],
    ) -> Result<(), GpuRigidSphereContactError> {
        if environment_ids.len() != self.body_count {
            return Err(GpuRigidSphereContactError::InvalidInput);
        }
        self.refresh_candidate_pairs_from_state_impl(
            device,
            queue,
            broad_phase,
            Some(environment_ids),
            true,
            0.0,
        )
    }

    /// Refresh filtered LBVH pairs with a finite speculative contact distance.
    /// Each AABB expands by half the pair margin; masks and environments still apply.
    pub fn refresh_speculative_pairs_from_state_filtered(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        broad_phase: &GpuLbvh,
        environment_ids: &[u32],
        margin: f32,
    ) -> Result<(), GpuRigidSphereContactError> {
        if environment_ids.len() != self.body_count || !margin.is_finite() || margin < 0.0 {
            return Err(GpuRigidSphereContactError::InvalidInput);
        }
        self.refresh_candidate_pairs_from_state_impl(
            device,
            queue,
            broad_phase,
            Some(environment_ids),
            true,
            margin,
        )
    }

    pub(crate) fn encode_state_bounds(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<wgpu::Buffer, GpuRigidSphereContactError> {
        let count =
            u32::try_from(self.body_count).map_err(|_| GpuRigidSphereContactError::Capacity)?;
        let bounds_bytes = self.body_count as u64 * size_of::<GpuAabb>() as u64;
        let limits = device.limits();
        if count == 0
            || count.div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || bounds_bytes > u64::from(limits.max_storage_buffer_binding_size)
            || bounds_bytes > limits.max_buffer_size
        {
            return Err(GpuRigidSphereContactError::Capacity);
        }
        let bounds = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera resident rigid AABBs"),
            size: bounds_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let (pipeline, entries) = if let Some(primitive) = &self.primitive {
            (
                &primitive.aabb_pipeline,
                vec![
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.state_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: primitive.shape_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: primitive.vertex_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: bounds.as_entire_binding(),
                    },
                ],
            )
        } else {
            (
                &self.aabb_pipeline,
                vec![
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.state_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.radii_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: bounds.as_entire_binding(),
                    },
                ],
            )
        };
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera resident rigid AABB inputs"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Tessera resident rigid AABB generation"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
        drop(pass);
        Ok(bounds)
    }

    fn refresh_candidate_pairs_from_state_impl(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        broad_phase: &GpuLbvh,
        environment_ids: Option<&[u32]>,
        filter_collision_groups: bool,
        margin: f32,
    ) -> Result<(), GpuRigidSphereContactError> {
        if self.body_count < 2 {
            return self.set_candidate_pairs(device, &[]);
        }
        let count =
            u32::try_from(self.body_count).map_err(|_| GpuRigidSphereContactError::Capacity)?;
        let limits = device.limits();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera resident sphere AABB encoder"),
        });
        let bounds = self.encode_state_bounds(device, &mut encoder)?;
        if margin > 0.0 {
            let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tessera speculative margin"),
                contents: bytemuck::cast_slice(&[margin * 0.5, 0.0, 0.0, 0.0]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Tessera speculative bounds"),
                layout: &self.bounds_margin_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: bounds.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: uniform.as_entire_binding(),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.bounds_margin_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
        }
        let _submission = queue.submit(Some(encoder.finish()));
        let candidates = if filter_collision_groups {
            let environment_ids =
                environment_ids.ok_or(GpuRigidSphereContactError::InvalidInput)?;
            let packed = environment_ids
                .iter()
                .copied()
                .zip(self.collision_groups.iter().copied())
                .map(|(environment, groups)| [environment, groups.memberships, groups.filter, 0])
                .collect::<Vec<_>>();
            let filter_bytes = bytemuck::cast_slice(&packed);
            if filter_bytes.len() as u64 > u64::from(limits.max_storage_buffer_binding_size)
                || filter_bytes.len() as u64 > limits.max_buffer_size
            {
                return Err(GpuRigidSphereContactError::Capacity);
            }
            let filter_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tessera resident environment and collision masks"),
                contents: filter_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });
            broad_phase.dispatch_buffer_filtered(device, queue, &bounds, &filter_buffer, count)?
        } else if let Some(environment_ids) = environment_ids {
            let group_bytes = bytemuck::cast_slice(environment_ids);
            if group_bytes.len() as u64 > u64::from(limits.max_storage_buffer_binding_size)
                || group_bytes.len() as u64 > limits.max_buffer_size
            {
                return Err(GpuRigidSphereContactError::Capacity);
            }
            let group_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tessera resident sphere environment IDs"),
                contents: group_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });
            broad_phase.dispatch_buffer_grouped(device, queue, &bounds, &group_buffer, count)?
        } else {
            broad_phase.dispatch_buffer(device, queue, &bounds, count)?
        };
        let mut pairs = candidates.readback(device, queue)?;
        for pair in &mut pairs {
            if pair.a > pair.b {
                core::mem::swap(&mut pair.a, &mut pair.b);
            }
        }
        pairs.sort_unstable_by_key(|pair| (pair.a, pair.b));
        pairs.dedup_by(|left, right| left.a == right.a && left.b == right.b);
        if pairs
            .iter()
            .any(|pair| pair.a as usize >= self.body_count || pair.b as usize >= self.body_count)
        {
            return Err(GpuRigidSphereContactError::InvalidInput);
        }
        if let Some(environment_ids) = environment_ids {
            pairs.retain(|pair| {
                environment_ids[pair.a as usize] == environment_ids[pair.b as usize]
                    && (!filter_collision_groups
                        || self.collision_groups[pair.a as usize]
                            .allows(self.collision_groups[pair.b as usize]))
            });
        }
        self.set_candidate_pairs(device, &pairs)
    }

    /// Replace broad-phase candidate pairs while retaining body state and materials.
    ///
    /// Call this before encoding the next contact and solver passes. Pair order
    /// defines contact readback order. The update rebuilds contact and island
    /// buffers, but does not upload or read back rigid-body state.
    pub fn set_candidate_pairs(
        &mut self,
        device: &wgpu::Device,
        pairs: &[GpuPair],
    ) -> Result<(), GpuRigidSphereContactError> {
        if pairs.len() == self.pairs.len()
            && pairs
                .iter()
                .zip(&self.pairs)
                .all(|(left, right)| left.a == right.a && left.b == right.b)
        {
            return Ok(());
        }
        if pairs.iter().any(|pair| {
            pair.a == pair.b
                || pair.a as usize >= self.body_count
                || pair.b as usize >= self.body_count
                || !(self.radii[pair.a as usize] + self.radii[pair.b as usize]).is_finite()
        }) {
            return Err(GpuRigidSphereContactError::InvalidInput);
        }
        let pair_count =
            u32::try_from(pairs.len()).map_err(|_| GpuRigidSphereContactError::Capacity)?;
        let limits = device.limits();
        let pair_bytes = size_of_val(pairs) as u64;
        let contact_bytes = pairs.len() as u64 * size_of::<GpuSphereContact>() as u64;
        let contact_storage_bytes = contact_bytes
            .checked_mul(u64::from(self.pair_stride))
            .ok_or(GpuRigidSphereContactError::Capacity)?;
        if pair_count.div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || [pair_bytes, contact_storage_bytes].iter().any(|size| {
                *size > u64::from(limits.max_storage_buffer_binding_size)
                    || *size > limits.max_buffer_size
            })
        {
            return Err(GpuRigidSphereContactError::Capacity);
        }
        let topology = build_islands(self.body_count, pairs, self.ground_enabled)?;
        let max_island_contacts = max_island_contact_slots(
            &topology,
            self.ground_enabled,
            self.pair_stride,
            self.ground_stride,
        )?;
        let (body_pair_offsets, body_pair_indices) =
            build_body_pair_adjacency(self.body_count, pairs)?;
        let island_count = u32::try_from(topology.ranges.len())
            .map_err(|_| GpuRigidSphereContactError::Capacity)?;
        let island_bytes = size_of_val(topology.ranges.as_slice()) as u64;
        let index_bytes = size_of_val(topology.indices.as_slice()) as u64;
        let adjacency_bytes = [
            size_of_val(body_pair_offsets.as_slice()) as u64,
            size_of_val(body_pair_indices.as_slice()) as u64,
        ];
        if island_count.div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || [
                island_bytes,
                index_bytes,
                adjacency_bytes[0],
                adjacency_bytes[1],
            ]
            .iter()
            .any(|size| {
                *size > u64::from(limits.max_storage_buffer_binding_size)
                    || *size > limits.max_buffer_size
            })
        {
            return Err(GpuRigidSphereContactError::Capacity);
        }
        let pair_buffer = if pairs.is_empty() {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Tessera empty rigid sphere pairs"),
                size: size_of::<GpuPair>() as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        } else {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tessera rigid sphere pairs"),
                contents: bytemuck::cast_slice(pairs),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let pair_contacts = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera rigid sphere pair contacts"),
            size: contact_storage_bytes.max(size_of::<GpuSphereContact>() as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let island_ranges = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera resident sphere island ranges"),
            contents: if topology.ranges.is_empty() {
                bytemuck::bytes_of(&IslandRange {
                    body_offset: 0,
                    body_count: 0,
                    pair_offset: 0,
                    pair_count: 0,
                })
            } else {
                bytemuck::cast_slice(&topology.ranges)
            },
            usage: wgpu::BufferUsages::STORAGE,
        });
        let island_indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera resident sphere island indices"),
            contents: if topology.indices.is_empty() {
                bytemuck::bytes_of(&0u32)
            } else {
                bytemuck::cast_slice(&topology.indices)
            },
            usage: wgpu::BufferUsages::STORAGE,
        });
        let body_pair_offsets = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera resident sphere body pair offsets"),
            contents: bytemuck::cast_slice(&body_pair_offsets),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let body_pair_indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera resident sphere body pair indices"),
            contents: if body_pair_indices.is_empty() {
                bytemuck::bytes_of(&0u32)
            } else {
                bytemuck::cast_slice(&body_pair_indices)
            },
            usage: wgpu::BufferUsages::STORAGE,
        });
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera rigid sphere contact params"),
            contents: bytemuck::bytes_of(&ContactParams {
                body_count: self.body_count as u32,
                pair_count,
                ground_half_extent: self.ground_half_extent,
                ground_enabled: u32::from(self.ground_enabled),
                ground_memberships: self.ground_collision_groups.memberships,
                ground_filter: self.ground_collision_groups.filter,
                padding: [0; 2],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera rigid sphere contact inputs"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.state_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.radii_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: pair_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: pair_contacts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.ground_contacts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.collision_groups_buffer.as_entire_binding(),
                },
            ],
        });
        self.pairs = pairs.to_vec();
        self.pair_buffer = pair_buffer;
        self.pair_contacts = pair_contacts;
        self.island_ranges = island_ranges;
        self.island_indices = island_indices;
        self.body_pair_offsets = body_pair_offsets;
        self.body_pair_indices = body_pair_indices;
        self.island_count = island_count;
        self.max_island_contacts = max_island_contacts;
        self.params = params.clone();
        self.bind_group = bind_group;
        if let Some(primitive) = &mut self.primitive {
            primitive.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Tessera resident primitive contact inputs"),
                layout: &primitive.pair_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.state_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: primitive.shape_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.pair_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.pair_contacts.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.ground_contacts.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: primitive.vertex_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: primitive.edge_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: self.collision_groups_buffer.as_entire_binding(),
                    },
                ],
            });
        }
        Ok(())
    }

    /// Candidate pair indices in the order used by the GPU contact buffer.
    pub fn pairs(&self) -> &[GpuPair] {
        &self.pairs
    }

    /// Current collision masks for one body.
    pub fn collision_groups(&self, index: usize) -> Option<GpuRigidCollisionGroups> {
        self.collision_groups.get(index).copied()
    }

    /// Change one body's masks without transferring its rigid state.
    pub fn set_collision_groups(
        &mut self,
        queue: &wgpu::Queue,
        index: usize,
        groups: GpuRigidCollisionGroups,
    ) -> Result<(), GpuRigidSphereContactError> {
        let current = self
            .collision_groups
            .get_mut(index)
            .ok_or(GpuRigidSphereContactError::InvalidInput)?;
        queue.write_buffer(
            &self.collision_groups_buffer,
            (index * size_of::<GpuRigidCollisionGroups>()) as u64,
            bytemuck::bytes_of(&groups),
        );
        *current = groups;
        Ok(())
    }

    /// Current collision masks for the finite ground.
    pub fn ground_collision_groups(&self) -> Option<GpuRigidCollisionGroups> {
        self.ground_enabled.then_some(self.ground_collision_groups)
    }

    /// Change ground masks used by both sphere and primitive contact kernels.
    pub fn set_ground_collision_groups(
        &mut self,
        queue: &wgpu::Queue,
        groups: GpuRigidCollisionGroups,
    ) -> Result<(), GpuRigidSphereContactError> {
        if !self.ground_enabled {
            return Err(GpuRigidSphereContactError::InvalidInput);
        }
        queue.write_buffer(
            &self.params,
            0,
            bytemuck::bytes_of(&ContactParams {
                body_count: self.body_count as u32,
                pair_count: self.pairs.len() as u32,
                ground_half_extent: self.ground_half_extent,
                ground_enabled: 1,
                ground_memberships: groups.memberships,
                ground_filter: groups.filter,
                padding: [0; 2],
            }),
        );
        self.ground_collision_groups = groups;
        Ok(())
    }

    /// Number of rigid bodies bound to this contact pipeline.
    pub fn body_count(&self) -> usize {
        self.body_count
    }

    /// Whether finite-ground contact slots are generated.
    pub fn has_ground(&self) -> bool {
        self.ground_enabled
    }

    pub(crate) fn ground_contact_stride(&self) -> u32 {
        self.ground_stride
    }

    pub(crate) fn pair_contact_stride(&self) -> u32 {
        self.pair_stride
    }

    fn encode_kinematic_activity(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera kinematic activity snapshot"),
            layout: &self.kinematic_activity_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.state_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.moving_kinematic.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Tessera capture moving kinematic bodies"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.kinematic_activity_pipeline);
        pass.set_bind_group(0, &bindings, &[]);
        pass.dispatch_workgroups((self.body_count as u32).div_ceil(64), 1, 1);
    }

    /// Encode automatic sleep after contact generation and impulse solving.
    ///
    /// Bodies sleep only after supported low-speed motion persists for the
    /// configured duration. Call this after the solver in each substep.
    pub fn encode_sleep(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        dt: f32,
        settings: SleepSettings,
    ) -> Result<(), GpuRigidSphereContactError> {
        self.encode_sleep_with_joints(device, encoder, dt, settings, None)
    }

    pub(crate) fn encode_sleep_with_joints(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        dt: f32,
        settings: SleepSettings,
        joints: Option<&GpuRigidBallJointSolver>,
    ) -> Result<(), GpuRigidSphereContactError> {
        let linear = settings.linear_velocity_threshold as f32;
        let angular = settings.angular_velocity_threshold as f32;
        let duration = settings.time_threshold as f32;
        if !settings.is_valid()
            || !dt.is_finite()
            || dt <= 0.0
            || !linear.is_finite()
            || !angular.is_finite()
            || !linear.mul_add(linear, 0.0).is_finite()
            || !angular.mul_add(angular, 0.0).is_finite()
            || !duration.is_finite()
            || duration <= 0.0
        {
            return Err(GpuRigidSphereContactError::InvalidInput);
        }
        if self.body_count == 0 {
            return Ok(());
        }
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera resident sphere sleep params"),
            contents: bytemuck::bytes_of(&GpuSleepParams {
                thresholds: [dt, linear * linear, angular * angular, duration],
                counts: [
                    self.body_count as u32,
                    u32::from(self.ground_enabled),
                    u32::from(settings.enabled),
                    self.pair_stride,
                ],
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera resident sphere sleep inputs"),
            layout: &self.sleep_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.state_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.pair_contacts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.ground_contacts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.body_pair_offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.body_pair_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.sleep_timers.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.pair_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: self.moving_kinematic.as_entire_binding(),
                },
            ],
        });
        self.encode_kinematic_activity(device, encoder);
        if let Some(joints) = joints {
            joints.encode_kinematic_activity(device, encoder, &self.moving_kinematic);
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Tessera resident sphere automatic sleep"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.sleep_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((self.body_count as u32).div_ceil(64), 1, 1);
        Ok(())
    }

    /// Update sleep from GPU-resident LBVH candidates after a candidate solve.
    ///
    /// Each body scans the valid candidate pairs for supporting contacts. A
    /// failed solve leaves sleep state unchanged through the status buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_candidate_sleep(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        candidates: &GpuLbvhResidentPairs,
        output: &GpuRigidCandidateContacts,
        solve_status: &wgpu::Buffer,
        dt: f32,
        settings: SleepSettings,
    ) -> Result<(), GpuRigidSphereContactError> {
        let linear = settings.linear_velocity_threshold as f32;
        let angular = settings.angular_velocity_threshold as f32;
        let duration = settings.time_threshold as f32;
        if candidates.collider_count as usize != self.body_count
            || candidates.pair_capacity != output.pair_capacity
            || !settings.is_valid()
            || !dt.is_finite()
            || dt <= 0.0
            || !linear.is_finite()
            || !angular.is_finite()
            || !linear.mul_add(linear, 0.0).is_finite()
            || !angular.mul_add(angular, 0.0).is_finite()
            || !duration.is_finite()
            || duration <= 0.0
        {
            return Err(GpuRigidSphereContactError::InvalidInput);
        }
        if self.body_count == 0 {
            return Ok(());
        }
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera resident candidate sleep params"),
            contents: bytemuck::bytes_of(&GpuSleepParams {
                thresholds: [dt, linear * linear, angular * angular, duration],
                counts: [
                    self.body_count as u32,
                    u32::from(self.ground_enabled),
                    u32::from(settings.enabled),
                    0,
                ],
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera resident candidate sleep inputs"),
            layout: &self.candidate_sleep_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.state_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: candidates.pairs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.pairs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output.ground.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: candidates.counter.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.sleep_timers.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: solve_status.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: self.moving_kinematic.as_entire_binding(),
                },
            ],
        });
        self.encode_kinematic_activity(device, encoder);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Tessera resident candidate sleep"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.candidate_sleep_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((self.body_count as u32).div_ceil(64), 1, 1);
        Ok(())
    }

    /// Sleep history and its packed stride for native GPU topology transfers.
    pub(crate) fn sleep_state_buffer(&self) -> (&wgpu::Buffer, u64) {
        (&self.sleep_timers, size_of::<GpuSleepState>() as u64)
    }

    /// Clear one body's idle timer after replacing its state or geometry.
    pub fn reset_sleep_timer(
        &self,
        queue: &wgpu::Queue,
        index: usize,
    ) -> Result<(), GpuRigidSphereContactError> {
        if index >= self.body_count {
            return Err(GpuRigidSphereContactError::InvalidInput);
        }
        queue.write_buffer(
            &self.sleep_timers,
            (index * size_of::<GpuSleepState>()) as u64,
            bytemuck::bytes_of(&GpuSleepState::default()),
        );
        Ok(())
    }

    /// Override one sphere's material without rebuilding contact buffers.
    pub fn set_body_material(
        &self,
        queue: &wgpu::Queue,
        index: usize,
        material: ColliderMaterial,
    ) -> Result<(), GpuRigidSphereContactError> {
        if index >= self.body_count {
            return Err(GpuRigidSphereContactError::InvalidInput);
        }
        let packed = GpuColliderMaterial::from_material(material)
            .ok_or(GpuRigidSphereContactError::InvalidInput)?;
        queue.write_buffer(
            &self.materials,
            (index * size_of::<GpuColliderMaterial>()) as u64,
            bytemuck::bytes_of(&packed),
        );
        Ok(())
    }

    /// Restore the solver's default material for one sphere.
    pub fn clear_body_material(
        &self,
        queue: &wgpu::Queue,
        index: usize,
    ) -> Result<(), GpuRigidSphereContactError> {
        if index >= self.body_count {
            return Err(GpuRigidSphereContactError::InvalidInput);
        }
        queue.write_buffer(
            &self.materials,
            (index * size_of::<GpuColliderMaterial>()) as u64,
            bytemuck::bytes_of(&GpuColliderMaterial::default()),
        );
        Ok(())
    }

    /// Override the finite ground plane's material.
    pub fn set_ground_material(
        &self,
        queue: &wgpu::Queue,
        material: ColliderMaterial,
    ) -> Result<(), GpuRigidSphereContactError> {
        if !self.ground_enabled {
            return Err(GpuRigidSphereContactError::InvalidInput);
        }
        let packed = GpuColliderMaterial::from_material(material)
            .ok_or(GpuRigidSphereContactError::InvalidInput)?;
        queue.write_buffer(
            &self.materials,
            (self.body_count * size_of::<GpuColliderMaterial>()) as u64,
            bytemuck::bytes_of(&packed),
        );
        Ok(())
    }

    /// Restore the solver's default material for the finite ground.
    pub fn clear_ground_material(
        &self,
        queue: &wgpu::Queue,
    ) -> Result<(), GpuRigidSphereContactError> {
        if !self.ground_enabled {
            return Err(GpuRigidSphereContactError::InvalidInput);
        }
        queue.write_buffer(
            &self.materials,
            (self.body_count * size_of::<GpuColliderMaterial>()) as u64,
            bytemuck::bytes_of(&GpuColliderMaterial::default()),
        );
        Ok(())
    }

    /// GPU pair index buffer for a later constraint pass.
    pub fn pair_buffer(&self) -> &wgpu::Buffer {
        &self.pair_buffer
    }

    /// GPU pair contacts, with primary slots followed by primitive manifold slots.
    pub fn pair_contact_buffer(&self) -> &wgpu::Buffer {
        &self.pair_contacts
    }

    /// GPU ground contacts, with primary slots followed by primitive manifold slots.
    pub fn ground_contact_buffer(&self) -> Option<&wgpu::Buffer> {
        self.ground_enabled.then_some(&self.ground_contacts)
    }

    pub(crate) fn ground_buffer_raw(&self) -> &wgpu::Buffer {
        &self.ground_contacts
    }

    pub(crate) fn ray_query_pipeline(&self, device: &wgpu::Device) -> wgpu::ComputePipeline {
        self.ray_query_pipeline
            .get_or_init(|| crate::gpu_ray_query::create_pipeline(device))
            .clone()
    }

    pub(crate) fn point_query_pipeline(&self, device: &wgpu::Device) -> wgpu::ComputePipeline {
        self.point_query_pipeline
            .get_or_init(|| crate::gpu_point_query::create_pipeline(device))
            .clone()
    }

    pub(crate) fn ray_query_buffers(&self, device: &wgpu::Device) -> [wgpu::Buffer; 5] {
        let (shapes, vertices, edges) = if let Some(primitive) = &self.primitive {
            (
                primitive.shape_buffer.clone(),
                primitive.vertex_buffer.clone(),
                primitive.edge_buffer.clone(),
            )
        } else {
            let shapes = self
                .radii
                .iter()
                .map(|radius| GpuRigidShape::Sphere { radius: *radius }.packed(0, 0, 0, 0, 0))
                .collect::<Vec<_>>();
            let dummy = [0u32; 12];
            let shapes = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tessera ray sphere descriptors"),
                contents: if shapes.is_empty() {
                    bytemuck::cast_slice(&dummy)
                } else {
                    bytemuck::cast_slice(&shapes)
                },
                usage: wgpu::BufferUsages::STORAGE,
            });
            let dummy = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tessera ray empty geometry"),
                contents: bytemuck::cast_slice(&dummy),
                usage: wgpu::BufferUsages::STORAGE,
            });
            (shapes, dummy.clone(), dummy)
        };
        [
            self.state_buffer.clone(),
            shapes,
            vertices,
            edges,
            self.collision_groups_buffer.clone(),
        ]
    }

    pub(crate) fn ray_query_ground(&self) -> (bool, f32, GpuRigidCollisionGroups) {
        (
            self.ground_enabled,
            self.ground_half_extent,
            self.ground_collision_groups,
        )
    }

    pub(crate) fn state_buffer(&self) -> &wgpu::Buffer {
        &self.state_buffer
    }

    pub(crate) fn material_buffer(&self) -> &wgpu::Buffer {
        &self.materials
    }

    pub(crate) fn island_range_buffer(&self) -> &wgpu::Buffer {
        &self.island_ranges
    }

    pub(crate) fn island_index_buffer(&self) -> &wgpu::Buffer {
        &self.island_indices
    }

    pub(crate) fn island_count(&self) -> u32 {
        self.island_count
    }

    pub(crate) fn max_island_contacts(&self) -> u32 {
        self.max_island_contacts
    }

    /// Generate GPU AABBs, filtered LBVH candidates, and contacts in one encoder.
    ///
    /// A missing environment list places every body in one environment. This
    /// path requires at least two bodies and worst-case pair capacity on the
    /// selected device. No candidate count or pair list is read by the CPU.
    pub fn encode_state_lbvh_contacts(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        lbvh: &GpuLbvh,
        environment_ids: Option<&[u32]>,
    ) -> Result<(GpuLbvhResidentPairs, GpuRigidCandidateContacts), GpuRigidSphereContactError> {
        if self.body_count < 2 || environment_ids.is_some_and(|ids| ids.len() != self.body_count) {
            return Err(GpuRigidSphereContactError::InvalidInput);
        }
        let count =
            u32::try_from(self.body_count).map_err(|_| GpuRigidSphereContactError::Capacity)?;
        let filter_data = self
            .collision_groups
            .iter()
            .enumerate()
            .map(|(index, groups)| {
                [
                    environment_ids.map_or(0, |ids| ids[index]),
                    groups.memberships,
                    groups.filter,
                    0,
                ]
            })
            .collect::<Vec<_>>();
        let filter_bytes = bytemuck::cast_slice(&filter_data);
        let limits = device.limits();
        if filter_bytes.len() as u64 > u64::from(limits.max_storage_buffer_binding_size)
            || filter_bytes.len() as u64 > limits.max_buffer_size
        {
            return Err(GpuRigidSphereContactError::Capacity);
        }
        let filter_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera resident LBVH environment and collision masks"),
            contents: filter_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
        let bounds = self.encode_state_bounds(device, encoder)?;
        let candidates =
            lbvh.encode_buffer_resident_filtered(device, encoder, &bounds, &filter_buffer, count)?;
        let contacts = self.encode_lbvh_candidates(device, encoder, lbvh, &candidates)?;
        Ok((candidates, contacts))
    }

    /// Generate contacts from GPU-resident LBVH pairs without a candidate readback.
    ///
    /// Encode the LBVH build before this call in the same command encoder. The
    /// pair buffer is sized to its capacity; the indirect dispatch visits only
    /// valid candidate slots. Pair and ground contacts remain on the GPU for
    /// downstream constraint passes.
    pub fn encode_lbvh_candidates(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        lbvh: &GpuLbvh,
        candidates: &GpuLbvhResidentPairs,
    ) -> Result<GpuRigidCandidateContacts, GpuRigidSphereContactError> {
        if candidates.collider_count as usize != self.body_count {
            return Err(GpuRigidSphereContactError::InvalidInput);
        }
        let contact_bytes = u64::from(candidates.pair_capacity)
            .checked_mul(u64::from(self.pair_stride))
            .and_then(|slots| slots.checked_mul(size_of::<GpuSphereContact>() as u64))
            .ok_or(GpuRigidSphereContactError::Capacity)?;
        let limits = device.limits();
        if contact_bytes > u64::from(limits.max_storage_buffer_binding_size)
            || contact_bytes > limits.max_buffer_size
        {
            return Err(GpuRigidSphereContactError::Capacity);
        }
        let contacts = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera resident LBVH pair contacts"),
            size: contact_bytes.max(size_of::<GpuSphereContact>() as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera resident LBVH contact parameters"),
            contents: bytemuck::bytes_of(&ContactParams {
                body_count: self.body_count as u32,
                pair_count: candidates.pair_capacity,
                ground_half_extent: self.ground_half_extent,
                ground_enabled: u32::from(self.ground_enabled),
                ground_memberships: self.ground_collision_groups.memberships,
                ground_filter: self.ground_collision_groups.filter,
                padding: [0; 2],
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let (pair_pipeline, specialized_pair_pipelines, ground_pipeline, bind_group) =
            if let Some(primitive) = &self.primitive {
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Tessera resident LBVH primitive contacts"),
                    layout: &primitive.pair_pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.state_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: primitive.shape_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: candidates.pairs.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: contacts.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: self.ground_contacts.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: params.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: primitive.vertex_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: primitive.edge_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 8,
                            resource: self.collision_groups_buffer.as_entire_binding(),
                        },
                    ],
                });
                (
                    &primitive.pair_pipeline,
                    primitive.specialized_pair_pipelines.as_slice(),
                    &primitive.ground_pipeline,
                    bind_group,
                )
            } else {
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Tessera resident LBVH sphere contacts"),
                    layout: &self.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.state_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.radii_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: candidates.pairs.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: contacts.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: self.ground_contacts.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: params.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: self.collision_groups_buffer.as_entire_binding(),
                        },
                    ],
                });
                (
                    &self.pair_pipeline,
                    &[] as &[(u32, wgpu::ComputePipeline)],
                    &self.ground_pipeline,
                    bind_group,
                )
            };
        let dispatch_args = lbvh.encode_candidate_dispatch_args(device, encoder, candidates, 64)?;
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Tessera resident LBVH pair contacts"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pair_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups_indirect(&dispatch_args, 0);
            for (_, specialized_pipeline) in specialized_pair_pipelines {
                pass.set_pipeline(specialized_pipeline);
                pass.dispatch_workgroups_indirect(&dispatch_args, 0);
            }
        }
        if self.ground_enabled && self.body_count > 0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Tessera resident LBVH ground contacts"),
                timestamp_writes: None,
            });
            pass.set_pipeline(ground_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((self.body_count as u32).div_ceil(64), 1, 1);
        }
        Ok(GpuRigidCandidateContacts {
            pairs: contacts,
            ground: self.ground_contacts.clone(),
            pair_capacity: candidates.pair_capacity,
            pair_stride: self.pair_stride,
            ground_stride: self.ground_stride,
        })
    }

    /// Encode speculative rows and capture a failure summary before transport refresh.
    pub fn encode_speculative_checked(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        margin: f32,
    ) -> Result<GpuRigidSpeculativeStatus, GpuRigidSphereContactError> {
        self.encode_speculative(device, encoder, margin)?;
        let pipeline = self.speculative_status_pipeline.get_or_init(|| {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Tessera speculative query status reduction"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("gpu_rigid_speculative_status.wgsl").into(),
                ),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Tessera speculative status pipeline"),
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        Ok(GpuRigidSpeculativeStatus::encode(
            device,
            encoder,
            pipeline,
            &self.pair_contacts,
            self.pairs.len() as u32,
        ))
    }

    /// Encode signed convex contact rows within a finite speculative distance.
    /// Unconverged queries set `depth_hit.z` on an inactive primary row. Use
    /// `encode_speculative_checked` to capture a summary before later row refresh.
    /// Surface meshes and polylines traverse their triangle or segment BVHs.
    pub fn encode_speculative(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        margin: f32,
    ) -> Result<(), GpuRigidSphereContactError> {
        if !margin.is_finite() || margin < 0.0 {
            return Err(GpuRigidSphereContactError::InvalidInput);
        }
        if let Some(primitive) = &self.primitive {
            self.encode_speculative_ground(device, encoder, margin)?;
            if self.pairs.is_empty() || margin == 0.0 {
                return Ok(());
            }
            let pipeline = primitive.speculative_pipeline.get_or_init(|| {
                let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Tessera convex distance witnesses"),
                    source: wgpu::ShaderSource::Wgsl(
                        format!(
                            "{}\n{}",
                            include_str!("gpu_rigid_primitive_contact.wgsl"),
                            include_str!("gpu_rigid_convex_distance.wgsl")
                        )
                        .into(),
                    ),
                });
                let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Tessera convex distance layout"),
                    bind_group_layouts: &[&primitive.pair_pipeline.get_bind_group_layout(0)],
                    immediate_size: 0,
                });
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Tessera convex distance pipeline"),
                    layout: Some(&layout),
                    module: &shader,
                    entry_point: Some("convex_speculative_main"),
                    compilation_options: Default::default(),
                    cache: None,
                })
            });
            let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tessera convex speculative parameters"),
                contents: bytemuck::bytes_of(&ContactParams {
                    body_count: self.body_count as u32,
                    pair_count: self.pairs.len() as u32,
                    ground_half_extent: self.ground_half_extent,
                    ground_enabled: u32::from(self.ground_enabled),
                    ground_memberships: self.ground_collision_groups.memberships,
                    ground_filter: self.ground_collision_groups.filter,
                    padding: [margin.to_bits(), margin.to_bits()],
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let buffers = [
                &self.state_buffer,
                &primitive.shape_buffer,
                &self.pair_buffer,
                &self.pair_contacts,
                &self.ground_contacts,
                &params,
                &primitive.vertex_buffer,
                &primitive.edge_buffer,
                &self.collision_groups_buffer,
            ];
            let entries = buffers
                .iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect::<Vec<_>>();
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Tessera convex speculative inputs"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            });
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((self.pairs.len() as u32).div_ceil(64), 1, 1);
            return Ok(());
        }
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera speculative sphere contact parameters"),
            contents: bytemuck::bytes_of(&ContactParams {
                body_count: self.body_count as u32,
                pair_count: self.pairs.len() as u32,
                ground_half_extent: self.ground_half_extent,
                ground_enabled: u32::from(self.ground_enabled),
                ground_memberships: self.ground_collision_groups.memberships,
                ground_filter: self.ground_collision_groups.filter,
                padding: [margin.to_bits(), margin.to_bits()],
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let buffers = [
            &self.state_buffer,
            &self.radii_buffer,
            &self.pair_buffer,
            &self.pair_contacts,
            &self.ground_contacts,
            &params,
            &self.collision_groups_buffer,
        ];
        let entries = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera speculative sphere inputs"),
            layout: &self.layout,
            entries: &entries,
        });
        for (pipeline, count) in [
            (&self.pair_pipeline, self.pairs.len()),
            (
                &self.ground_pipeline,
                if self.ground_enabled {
                    self.body_count
                } else {
                    0
                },
            ),
        ] {
            if count == 0 {
                continue;
            }
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((count as u32).div_ceil(64), 1, 1);
        }
        Ok(())
    }

    /// Encode all ordinary pairs and speculative ground manifolds for every shape.
    /// Pair contacts retain their ordinary overlap conditions.
    pub fn encode_speculative_ground(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        margin: f32,
    ) -> Result<(), GpuRigidSphereContactError> {
        if !margin.is_finite() || margin < 0.0 {
            return Err(GpuRigidSphereContactError::InvalidInput);
        }
        self.encode_narrow_phase(encoder, false);
        if !self.ground_enabled || self.body_count == 0 {
            return Ok(());
        }
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera speculative ground parameters"),
            contents: bytemuck::bytes_of(&ContactParams {
                body_count: self.body_count as u32,
                pair_count: self.pairs.len() as u32,
                ground_half_extent: self.ground_half_extent,
                ground_enabled: 1,
                ground_memberships: self.ground_collision_groups.memberships,
                ground_filter: self.ground_collision_groups.filter,
                padding: [0, margin.to_bits()],
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let (pipeline, buffers) = if let Some(primitive) = &self.primitive {
            (
                &primitive.ground_pipeline,
                vec![
                    &self.state_buffer,
                    &primitive.shape_buffer,
                    &self.pair_buffer,
                    &self.pair_contacts,
                    &self.ground_contacts,
                    &params,
                    &primitive.vertex_buffer,
                    &primitive.edge_buffer,
                    &self.collision_groups_buffer,
                ],
            )
        } else {
            (
                &self.ground_pipeline,
                vec![
                    &self.state_buffer,
                    &self.radii_buffer,
                    &self.pair_buffer,
                    &self.pair_contacts,
                    &self.ground_contacts,
                    &params,
                    &self.collision_groups_buffer,
                ],
            )
        };
        let entries = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera speculative ground inputs"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((self.body_count as u32).div_ceil(64), 1, 1);
        Ok(())
    }

    /// Encode narrow-phase passes after state updates on the same encoder.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        self.encode_narrow_phase(encoder, true);
    }

    fn encode_narrow_phase(&self, encoder: &mut wgpu::CommandEncoder, include_ground: bool) {
        let (pair_pipeline, specialized_pair_pipelines, ground_pipeline, bind_group) =
            if let Some(primitive) = &self.primitive {
                (
                    &primitive.pair_pipeline,
                    primitive.specialized_pair_pipelines.as_slice(),
                    &primitive.ground_pipeline,
                    &primitive.bind_group,
                )
            } else {
                (
                    &self.pair_pipeline,
                    &[] as &[(u32, wgpu::ComputePipeline)],
                    &self.ground_pipeline,
                    &self.bind_group,
                )
            };
        if !self.pairs.is_empty() {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Tessera rigid sphere pair contacts"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pair_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups((self.pairs.len() as u32).div_ceil(64), 1, 1);
            for (_, specialized_pipeline) in specialized_pair_pipelines {
                pass.set_pipeline(specialized_pipeline);
                pass.dispatch_workgroups((self.pairs.len() as u32).div_ceil(64), 1, 1);
            }
        }
        if include_ground && self.ground_enabled && self.body_count > 0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Tessera rigid sphere ground contacts"),
                timestamp_writes: None,
            });
            pass.set_pipeline(ground_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups((self.body_count as u32).div_ceil(64), 1, 1);
        }
    }

    /// Synchronize contact slots for diagnostics or a CPU reference solver.
    pub fn readback(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<GpuRigidSphereContactReadback, GpuRigidSphereContactError> {
        let pair_indices: Vec<_> = (0..self.pairs.len()).collect();
        self.readback_selected(device, queue, &pair_indices, 0..self.body_count)
    }

    pub(crate) fn readback_ground(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<GpuRigidSphereContactReadback, GpuRigidSphereContactError> {
        self.readback_selected(device, queue, &[], 0..self.body_count)
    }

    /// Transfer only pair and ground slots belonging to a contiguous body range.
    pub fn readback_body_range(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        body_range: Range<usize>,
    ) -> Result<GpuRigidSphereContactReadback, GpuRigidSphereContactError> {
        if body_range.start > body_range.end || body_range.end > self.body_count {
            return Err(GpuRigidSphereContactError::InvalidInput);
        }
        let pair_indices: Vec<_> = self
            .pairs
            .iter()
            .enumerate()
            .filter_map(|(index, pair)| {
                (body_range.contains(&(pair.a as usize)) && body_range.contains(&(pair.b as usize)))
                    .then_some(index)
            })
            .collect();
        self.readback_selected(device, queue, &pair_indices, body_range)
    }

    fn readback_selected(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pair_indices: &[usize],
        body_range: Range<usize>,
    ) -> Result<GpuRigidSphereContactReadback, GpuRigidSphereContactError> {
        let contact_bytes = size_of::<GpuSphereContact>() as u64;
        let pair_bytes = pair_indices.len() as u64 * contact_bytes;
        let pair_extra_bytes = if self.pair_stride == 4 {
            pair_bytes * 3
        } else {
            0
        };
        let pair_storage_bytes = pair_bytes + pair_extra_bytes;
        let ground_bytes = if self.ground_enabled {
            body_range.len() as u64 * contact_bytes
        } else {
            0
        };
        let extra_bytes = if self.ground_stride == 4 {
            ground_bytes * 3
        } else {
            0
        };
        let total = pair_storage_bytes + ground_bytes + extra_bytes;
        if total == 0 {
            return Ok(GpuRigidSphereContactReadback {
                pairs: Vec::new(),
                pair_extra: Vec::new(),
                ground: Vec::new(),
                ground_extra: Vec::new(),
            });
        }
        if total > device.limits().max_buffer_size {
            return Err(GpuRigidSphereContactError::Capacity);
        }
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera rigid sphere contact readback"),
            size: total,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera rigid sphere contact readback encoder"),
        });
        let mut run_start = 0;
        while run_start < pair_indices.len() {
            let mut run_end = run_start + 1;
            while run_end < pair_indices.len()
                && pair_indices[run_end] == pair_indices[run_end - 1] + 1
            {
                run_end += 1;
            }
            let source = pair_indices[run_start] as u64;
            let count = (run_end - run_start) as u64;
            encoder.copy_buffer_to_buffer(
                &self.pair_contacts,
                source * contact_bytes,
                &staging,
                run_start as u64 * contact_bytes,
                count * contact_bytes,
            );
            if pair_extra_bytes > 0 {
                encoder.copy_buffer_to_buffer(
                    &self.pair_contacts,
                    (self.pairs.len() as u64 + source * 3) * contact_bytes,
                    &staging,
                    pair_bytes + run_start as u64 * 3 * contact_bytes,
                    count * 3 * contact_bytes,
                );
            }
            run_start = run_end;
        }
        if ground_bytes > 0 {
            encoder.copy_buffer_to_buffer(
                &self.ground_contacts,
                body_range.start as u64 * contact_bytes,
                &staging,
                pair_storage_bytes,
                ground_bytes,
            );
        }
        if extra_bytes > 0 {
            encoder.copy_buffer_to_buffer(
                &self.ground_contacts,
                (self.body_count as u64 + body_range.start as u64 * 3) * contact_bytes,
                &staging,
                pair_storage_bytes + ground_bytes,
                extra_bytes,
            );
        }
        let _submission = queue.submit(Some(encoder.finish()));
        map_readback(device, &staging)?;
        let view = staging.slice(..).get_mapped_range();
        let pair_contacts = view[..pair_bytes as usize]
            .chunks_exact(size_of::<GpuSphereContact>())
            .map(bytemuck::pod_read_unaligned);
        let pair_extra = view[pair_bytes as usize..pair_storage_bytes as usize]
            .chunks_exact(size_of::<[GpuSphereContact; 3]>())
            .map(bytemuck::pod_read_unaligned)
            .collect();
        let ground_end = (pair_storage_bytes + ground_bytes) as usize;
        let ground = view[pair_storage_bytes as usize..ground_end]
            .chunks_exact(size_of::<GpuSphereContact>())
            .map(bytemuck::pod_read_unaligned)
            .collect();
        let ground_extra = view[ground_end..]
            .chunks_exact(size_of::<[GpuSphereContact; 3]>())
            .map(bytemuck::pod_read_unaligned)
            .collect();
        let pairs = pair_indices
            .iter()
            .map(|&index| self.pairs[index])
            .zip(pair_contacts)
            .collect();
        drop(view);
        staging.unmap();
        Ok(GpuRigidSphereContactReadback {
            pairs,
            pair_extra,
            ground,
            ground_extra,
        })
    }
}

pub(crate) fn map_readback(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
) -> Result<(), GpuRigidSphereContactError> {
    let (sender, receiver) = mpsc::channel();
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
    let _status = device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(5)),
        })
        .map_err(|error| GpuRigidSphereContactError::Readback(error.to_string()))?;
    receiver
        .recv_timeout(Duration::from_secs(5))
        .map_err(|error| GpuRigidSphereContactError::Readback(error.to_string()))?
        .map_err(|error| GpuRigidSphereContactError::Readback(error.to_string()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::gpu_contact_pipeline::GpuContactDevice;
    use crate::gpu_rigid_state::{GpuRigidBodyForces, GpuRigidBodyState};

    #[test]
    fn speculative_status_reduction_preserves_failures_after_contact_refresh() {
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let device = context.device();
            let queue = context.queue();
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Speculative status test shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("gpu_rigid_speculative_status.wgsl").into(),
                ),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Speculative status test pipeline"),
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            let zero = GpuSphereContact {
                point: [0.0; 4],
                normal: [0.0; 4],
                depth_hit: [0.0; 4],
            };
            let mut values = vec![zero; 130];
            values[0].depth_hit = [0.1, 1.0, 0.0, 0.0];
            values[129].depth_hit = [0.0, 0.0, 1.0, 0.0];
            let contacts = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Synthetic speculative status rows"),
                contents: bytemuck::cast_slice(&values),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            for (count, success) in [(0, true), (129, true), (130, false)] {
                queue.write_buffer(&contacts, 0, bytemuck::cast_slice(&values));
                let mut encoder = device.create_command_encoder(&Default::default());
                let status = GpuRigidSpeculativeStatus::encode(
                    device,
                    &mut encoder,
                    &pipeline,
                    &contacts,
                    count,
                );
                let _ = queue.submit(Some(encoder.finish()));
                // Later anchor refresh can erase inactive failure rows, but not this summary.
                queue.write_buffer(&contacts, 0, bytemuck::cast_slice(&vec![zero; 130]));
                let result = status.readback(device, queue);
                assert_eq!(result.is_ok(), success, "{backend:?}: count={count}");
                if !success {
                    assert!(matches!(
                        result,
                        Err(GpuRigidSphereContactError::SpeculativeUnconverged)
                    ));
                }
            }
        }
        assert!(tested > 0, "no GPU backend available");
    }

    fn body(position: [f32; 3], velocity: [f32; 3]) -> GpuRigidBodyState {
        GpuRigidBodyState {
            position_inverse_mass: [position[0], position[1], position[2], 1.0],
            orientation: [0.0, 0.0, 0.0, 1.0],
            linear_velocity: [velocity[0], velocity[1], velocity[2], 0.0],
            angular_velocity: [0.0; 4],
            inverse_inertia_sleep: [1.0, 1.0, 1.0, 0.0],
        }
    }

    #[test]
    fn resident_lbvh_candidates_feed_sphere_contacts_without_pair_readback() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let device = context.device();
        let queue = context.queue();
        let states = (0..70)
            .map(|index| body([index as f32, 0.0, 0.5], [0.0; 3]))
            .collect::<Vec<_>>();
        let session = GpuRigidStateSession::new(device, queue, &states).unwrap();
        let contacts =
            GpuRigidSphereContacts::new(device, &session, &[0.51; 70], &[], Some(100.0)).unwrap();
        let bounds = (0..70)
            .map(|index| {
                GpuAabb::new(
                    [index as f32 - 0.51, -0.51, -0.01],
                    [index as f32 + 0.51, 0.51, 1.01],
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let bounds_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Resident sphere LBVH contact test bounds"),
            contents: bytemuck::cast_slice(&bounds),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let lbvh = GpuLbvh::new(device);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Resident sphere LBVH contact test encoder"),
        });
        let candidates = lbvh
            .encode_buffer_resident(device, &mut encoder, &bounds_buffer, 70)
            .unwrap();
        let output = contacts
            .encode_lbvh_candidates(device, &mut encoder, &lbvh, &candidates)
            .unwrap();
        assert_eq!(output.pair_capacity, 70 * 69 / 2);
        assert_eq!(output.pair_stride, 1);
        let _submission = queue.submit(Some(encoder.finish()));
        let result = output.readback_pairs(device, queue, &candidates).unwrap();
        let actual = result
            .pairs
            .iter()
            .map(|(pair, _)| (pair.a, pair.b))
            .collect::<BTreeSet<_>>();
        assert_eq!(actual, (0..69).map(|index| (index, index + 1)).collect());
        assert!(
            result
                .pairs
                .iter()
                .all(|(_, contact)| contact.is_contact()
                    && (contact.depth_hit[0] - 0.02).abs() < 1e-4)
        );
        assert!(result.pair_extra.is_empty());
        let ground = contacts.readback(device, queue).unwrap();
        assert_eq!(ground.ground.len(), 70);
        assert!(ground.ground.iter().all(GpuSphereContact::is_contact));
    }

    #[test]
    fn resident_lbvh_candidates_feed_box_manifolds_without_pair_readback() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let device = context.device();
        let queue = context.queue();
        let states = (0..70)
            .map(|index| body([index as f32, 0.0, 2.0], [0.0; 3]))
            .collect::<Vec<_>>();
        let session = GpuRigidStateSession::new(device, queue, &states).unwrap();
        let shapes = vec![
            GpuRigidShape::Box {
                half_extents: [0.51, 0.5, 0.5],
            };
            70
        ];
        let contacts =
            GpuRigidSphereContacts::new_with_shapes(device, &session, &shapes, &[], None).unwrap();
        let lbvh = GpuLbvh::new(device);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Resident box LBVH contact test encoder"),
        });
        let (candidates, output) = contacts
            .encode_state_lbvh_contacts(device, &mut encoder, &lbvh, None)
            .unwrap();
        assert_eq!(output.pair_stride, 4);
        let _submission = queue.submit(Some(encoder.finish()));
        let result = output.readback_pairs(device, queue, &candidates).unwrap();
        assert_eq!(result.pairs.len(), 69);
        assert_eq!(result.pair_extra.len(), 69);
        assert!(result.pairs.iter().all(|(_, contact)| contact.is_contact()));
        assert!(
            result
                .pair_extra
                .iter()
                .all(|extra| extra.iter().all(GpuSphereContact::is_contact))
        );
    }

    #[test]
    fn resident_state_to_contact_path_tracks_moved_bodies_and_environments() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let device = context.device();
        let queue = context.queue();
        let states = (0..70)
            .map(|index| body([index as f32 * 10.0, 0.0, 2.0], [0.0; 3]))
            .collect::<Vec<_>>();
        let session = GpuRigidStateSession::new(device, queue, &states).unwrap();
        let contacts =
            GpuRigidSphereContacts::new(device, &session, &[1.0; 70], &[], None).unwrap();
        let lbvh = GpuLbvh::new(device);
        let run = |environment_ids| {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Resident state LBVH contact test"),
            });
            let (candidates, output) = contacts
                .encode_state_lbvh_contacts(device, &mut encoder, &lbvh, environment_ids)
                .unwrap();
            let _submission = queue.submit(Some(encoder.finish()));
            output.readback_pairs(device, queue, &candidates).unwrap()
        };
        assert!(run(None).pairs.is_empty());
        session
            .write_body(queue, 1, body([1.5, 0.0, 2.0], [0.0; 3]))
            .unwrap();
        let mut environment_ids = vec![0; 70];
        environment_ids[1] = 1;
        assert!(run(Some(&environment_ids)).pairs.is_empty());
        let result = run(None);
        assert_eq!(result.pairs.len(), 1);
        assert_eq!((result.pairs[0].0.a, result.pairs[0].0.b), (0, 1));
        assert!(result.pairs[0].1.is_contact());
        assert!((result.pairs[0].1.depth_hit[0] - 0.5).abs() < 1e-4);
    }

    #[test]
    fn contact_islands_partition_shared_and_independent_bodies() {
        let pairs = [
            GpuPair { a: 0, b: 1 },
            GpuPair { a: 1, b: 2 },
            GpuPair { a: 4, b: 5 },
        ];
        let topology = build_islands(6, &pairs, false).unwrap();
        assert_eq!(topology.ranges.len(), 2);
        assert_eq!(max_island_contact_slots(&topology, false, 1, 1).unwrap(), 2);
        assert_eq!(topology.indices, [0, 1, 2, 4, 5, 0, 1, 2]);
        assert_eq!(topology.ranges[0].body_count, 3);
        assert_eq!(topology.ranges[0].pair_count, 2);
        assert_eq!(topology.ranges[1].body_count, 2);
        assert_eq!(topology.ranges[1].pair_count, 1);

        let with_ground = build_islands(6, &pairs, true).unwrap();
        assert_eq!(with_ground.ranges.len(), 3);
        assert_eq!(
            max_island_contact_slots(&with_ground, true, 1, 1).unwrap(),
            5
        );
        assert_eq!(
            max_island_contact_slots(&with_ground, true, 1, 4).unwrap(),
            14
        );
        assert_eq!(
            max_island_contact_slots(&with_ground, true, 4, 4).unwrap(),
            20
        );
        assert_eq!(with_ground.ranges[1].body_count, 1);
        assert_eq!(with_ground.ranges[1].pair_count, 0);
    }

    #[test]
    fn contact_range_readback_selects_disjoint_pairs_and_ground_slots() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let device = context.device();
        let queue = context.queue();
        let states: Vec<_> = (0..5)
            .map(|index| body([index as f32 * 0.5, 0.0, 0.5], [0.0; 3]))
            .collect();
        let state = GpuRigidStateSession::new(device, queue, &states).unwrap();
        let pairs = [
            GpuPair { a: 0, b: 1 },
            GpuPair { a: 1, b: 2 },
            GpuPair { a: 3, b: 4 },
            GpuPair { a: 2, b: 3 },
        ];
        let contacts =
            GpuRigidSphereContacts::new(device, &state, &[1.0; 5], &pairs, Some(10.0)).unwrap();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera contact range readback test"),
        });
        contacts.encode(&mut encoder);
        let _submission = queue.submit(Some(encoder.finish()));

        let full = contacts.readback(device, queue).unwrap();
        let selected = contacts.readback_body_range(device, queue, 1..4).unwrap();
        assert_eq!(selected.pairs.len(), 2);
        for (selected_pair, full_index) in selected.pairs.iter().zip([1, 3]) {
            assert_eq!(selected_pair.0.a, full.pairs[full_index].0.a);
            assert_eq!(selected_pair.0.b, full.pairs[full_index].0.b);
            assert_eq!(
                bytemuck::bytes_of(&selected_pair.1),
                bytemuck::bytes_of(&full.pairs[full_index].1)
            );
        }
        assert_eq!(selected.ground.len(), 3);
        for (selected_ground, full_ground) in selected.ground.iter().zip(&full.ground[1..4]) {
            assert_eq!(
                bytemuck::bytes_of(selected_ground),
                bytemuck::bytes_of(full_ground)
            );
        }
        let empty = contacts.readback_body_range(device, queue, 3..3).unwrap();
        assert!(empty.pairs.is_empty());
        assert!(empty.ground.is_empty());
        assert!(matches!(
            contacts.readback_body_range(device, queue, 4..6),
            Err(GpuRigidSphereContactError::InvalidInput)
        ));
    }

    #[test]
    fn resident_state_feeds_pair_and_ground_contacts_without_state_readback() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let device = context.device();
        let queue = context.queue();
        let state = GpuRigidStateSession::new(
            device,
            queue,
            &[
                body([0.0, 0.0, 0.8], [2.0, 0.0, 0.0]),
                body([2.4, 0.0, 0.8], [0.0; 3]),
                body([5.0, 0.0, 3.0], [0.0; 3]),
            ],
        )
        .unwrap();
        let contacts = GpuRigidSphereContacts::new(
            device,
            &state,
            &[1.0, 1.0, 0.5],
            &[GpuPair { a: 0, b: 1 }, GpuPair { a: 0, b: 2 }],
            Some(2.0),
        )
        .unwrap();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera resident sphere contact test"),
        });
        state
            .encode_step(device, &mut encoder, 0.25, [0.0; 3])
            .unwrap();
        contacts.encode(&mut encoder);
        let _submission = queue.submit(Some(encoder.finish()));
        let first = contacts.readback(device, queue).unwrap();
        assert_eq!(first.pairs.len(), 2);
        assert!(first.pairs[0].1.is_contact());
        assert!((first.pairs[0].1.depth_hit[0] - 0.1).abs() < 1e-5);
        assert!((first.pairs[0].1.point[0] - 1.45).abs() < 1e-5);
        assert!(first.pairs[0].1.normal[0] > 0.99);
        assert!(!first.pairs[1].1.is_contact());
        assert_eq!(first.ground.len(), 3);
        assert!((first.ground[0].depth_hit[0] - 0.2).abs() < 1e-5);
        assert!((first.ground[1].depth_hit[0] - 0.2).abs() < 1e-5);
        assert!(!first.ground[2].is_contact());

        let mut next = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera resident sphere contact second step"),
        });
        state
            .encode_step(device, &mut next, 0.25, [0.0; 3])
            .unwrap();
        contacts.encode(&mut next);
        let _submission = queue.submit(Some(next.finish()));
        let second = contacts.readback(device, queue).unwrap();
        assert!((second.pairs[0].1.depth_hit[0] - 0.6).abs() < 1e-5);
    }

    #[test]
    fn empty_pairs_and_invalid_geometry_are_handled() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let device = context.device();
        let queue = context.queue();
        let state = GpuRigidStateSession::new(device, queue, &[body([0.0; 3], [0.0; 3])]).unwrap();
        assert!(matches!(
            GpuRigidSphereContacts::new(device, &state, &[f32::NAN], &[], None),
            Err(GpuRigidSphereContactError::InvalidInput)
        ));
        assert!(matches!(
            GpuRigidSphereContacts::new(device, &state, &[1.0], &[GpuPair { a: 0, b: 1 }], None,),
            Err(GpuRigidSphereContactError::InvalidInput)
        ));
        let contacts = GpuRigidSphereContacts::new(device, &state, &[1.0], &[], None).unwrap();
        assert!(matches!(
            contacts.set_body_material(queue, 0, ColliderMaterial::new(-1.0, 0.0)),
            Err(GpuRigidSphereContactError::InvalidInput)
        ));
        assert!(matches!(
            contacts.set_body_material(queue, 0, ColliderMaterial::new(f64::MAX, 0.0)),
            Err(GpuRigidSphereContactError::InvalidInput)
        ));
        assert!(matches!(
            contacts.set_body_material(queue, 1, ColliderMaterial::default()),
            Err(GpuRigidSphereContactError::InvalidInput)
        ));
        assert!(matches!(
            contacts.set_ground_material(queue, ColliderMaterial::default()),
            Err(GpuRigidSphereContactError::InvalidInput)
        ));
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera empty resident sphere contacts"),
        });
        contacts.encode(&mut encoder);
        let _submission = queue.submit(Some(encoder.finish()));
        let result = contacts.readback(device, queue).unwrap();
        assert!(result.pairs.is_empty());
        assert!(result.ground.is_empty());
        assert!(contacts.ground_contact_buffer().is_none());
    }

    #[test]
    fn candidate_pairs_change_without_replacing_resident_state() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let device = context.device();
        let queue = context.queue();
        let state = GpuRigidStateSession::new(
            device,
            queue,
            &[
                body([0.0, 0.0, 1.0], [0.0; 3]),
                body([1.5, 0.0, 1.0], [0.0; 3]),
                body([4.0, 0.0, 1.0], [0.0; 3]),
            ],
        )
        .unwrap();
        let mut contacts =
            GpuRigidSphereContacts::new(device, &state, &[1.0; 3], &[], None).unwrap();
        assert_eq!(contacts.island_count(), 0);
        contacts
            .set_candidate_pairs(device, &[GpuPair { a: 0, b: 1 }])
            .unwrap();
        assert_eq!(contacts.island_count(), 1);
        assert_eq!(contacts.max_island_contacts(), 1);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera updated resident pair contacts"),
        });
        contacts.encode(&mut encoder);
        let _submission = queue.submit(Some(encoder.finish()));
        assert!(
            contacts.readback(device, queue).unwrap().pairs[0]
                .1
                .is_contact()
        );

        assert!(matches!(
            contacts.set_candidate_pairs(device, &[GpuPair { a: 1, b: 3 }]),
            Err(GpuRigidSphereContactError::InvalidInput)
        ));
        assert_eq!(contacts.pairs()[0].a, 0);
        contacts
            .set_candidate_pairs(device, &[GpuPair { a: 1, b: 2 }])
            .unwrap();
        state
            .write_body(queue, 2, body([2.5, 0.0, 1.0], [0.0; 3]))
            .unwrap();
        let mut next = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera changed resident pair"),
        });
        contacts.encode(&mut next);
        let _submission = queue.submit(Some(next.finish()));
        let result = contacts.readback(device, queue).unwrap();
        assert_eq!(result.pairs[0].0.a, 1);
        assert!(result.pairs[0].1.is_contact());

        contacts.set_candidate_pairs(device, &[]).unwrap();
        assert_eq!(contacts.island_count(), 0);
        assert!(contacts.pairs().is_empty());
    }

    #[test]
    fn candidate_update_keeps_ground_contacts() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let device = context.device();
        let queue = context.queue();
        let state = GpuRigidStateSession::new(
            device,
            queue,
            &[
                body([0.0, 0.0, 0.8], [0.0; 3]),
                body([1.5, 0.0, 0.8], [0.0; 3]),
            ],
        )
        .unwrap();
        let mut contacts =
            GpuRigidSphereContacts::new(device, &state, &[1.0; 2], &[], Some(3.0)).unwrap();
        contacts
            .set_candidate_pairs(device, &[GpuPair { a: 0, b: 1 }])
            .unwrap();
        assert_eq!(contacts.island_count(), 1);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera updated pairs with ground"),
        });
        contacts.encode(&mut encoder);
        let _submission = queue.submit(Some(encoder.finish()));
        let result = contacts.readback(device, queue).unwrap();
        assert!(result.pairs[0].1.is_contact());
        assert_eq!(result.ground.len(), 2);
        assert!(result.ground.iter().all(GpuSphereContact::is_contact));
    }

    #[test]
    fn resident_broad_phase_tracks_moving_spheres_without_state_readback() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let device = context.device();
        let queue = context.queue();
        let state = GpuRigidStateSession::new(
            device,
            queue,
            &[
                body([0.0, 0.0, 2.0], [0.0; 3]),
                body([1.5, 0.0, 2.0], [0.0; 3]),
                body([5.0, 0.0, 2.0], [0.0; 3]),
            ],
        )
        .unwrap();
        let mut contacts =
            GpuRigidSphereContacts::new(device, &state, &[1.0; 3], &[], None).unwrap();
        let broad_phase = GpuLbvh::new(device);
        contacts
            .refresh_candidate_pairs_from_state(device, queue, &broad_phase)
            .unwrap();
        assert_eq!(contacts.pairs().len(), 1);
        assert_eq!((contacts.pairs()[0].a, contacts.pairs()[0].b), (0, 1));
        state
            .write_body(queue, 1, body([4.2, 0.0, 2.0], [0.0; 3]))
            .unwrap();
        contacts
            .refresh_candidate_pairs_from_state(device, queue, &broad_phase)
            .unwrap();
        assert_eq!(contacts.pairs().len(), 1);
        assert_eq!((contacts.pairs()[0].a, contacts.pairs()[0].b), (1, 2));
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera resident broad-phase contacts"),
        });
        contacts.encode(&mut encoder);
        let _submission = queue.submit(Some(encoder.finish()));
        assert!(
            contacts.readback(device, queue).unwrap().pairs[0]
                .1
                .is_contact()
        );
    }

    #[test]
    fn resident_broad_phase_keeps_sparse_large_scene_compact() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let device = context.device();
        let queue = context.queue();
        let initial = (0..70)
            .map(|index| body([index as f32, 0.0, 2.0], [0.0; 3]))
            .collect::<Vec<_>>();
        let state = GpuRigidStateSession::new(device, queue, &initial).unwrap();
        let mut contacts =
            GpuRigidSphereContacts::new(device, &state, &[0.51; 70], &[], None).unwrap();
        contacts
            .refresh_candidate_pairs_from_state(device, queue, &GpuLbvh::new(device))
            .unwrap();
        assert_eq!(contacts.pairs().len(), 69);
        for (index, pair) in contacts.pairs().iter().enumerate() {
            assert_eq!((pair.a, pair.b), (index as u32, index as u32 + 1));
        }
        assert_eq!(contacts.island_count(), 1);
        assert_eq!(contacts.max_island_contacts(), 69);
    }

    #[test]
    fn resident_two_sphere_broad_phase_handles_empty_and_overlap() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let device = context.device();
        let queue = context.queue();
        let state = GpuRigidStateSession::new(
            device,
            queue,
            &[
                body([0.0, 0.0, 2.0], [0.0; 3]),
                body([5.0, 0.0, 2.0], [0.0; 3]),
            ],
        )
        .unwrap();
        let mut contacts =
            GpuRigidSphereContacts::new(device, &state, &[1.0; 2], &[], None).unwrap();
        let broad_phase = GpuLbvh::new(device);
        contacts
            .refresh_candidate_pairs_from_state(device, queue, &broad_phase)
            .unwrap();
        assert!(contacts.pairs().is_empty());
        state
            .write_body(queue, 1, body([1.5, 0.0, 2.0], [0.0; 3]))
            .unwrap();
        contacts
            .refresh_candidate_pairs_from_state(device, queue, &broad_phase)
            .unwrap();
        assert_eq!(contacts.pairs().len(), 1);
        assert_eq!((contacts.pairs()[0].a, contacts.pairs()[0].b), (0, 1));
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn dx12_resident_broad_phase_tracks_gpu_positions() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::DX12,
            ..Default::default()
        });
        let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
        else {
            eprintln!("DX12 adapter unavailable; skipping resident broad-phase test");
            return;
        };
        eprintln!(
            "Tessera resident broad phase adapter: {:?}",
            adapter.get_info().backend
        );
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let state = GpuRigidStateSession::new(
            &device,
            &queue,
            &[
                body([0.0, 0.0, 2.0], [0.0; 3]),
                body([1.5, 0.0, 2.0], [0.0; 3]),
            ],
        )
        .unwrap();
        let mut contacts =
            GpuRigidSphereContacts::new(&device, &state, &[1.0; 2], &[], None).unwrap();
        contacts
            .refresh_candidate_pairs_from_state(&device, &queue, &GpuLbvh::new(&device))
            .unwrap();
        assert_eq!(contacts.pairs().len(), 1);
        assert_eq!((contacts.pairs()[0].a, contacts.pairs()[0].b), (0, 1));
    }

    #[test]
    fn automatic_sleep_requires_supported_idle_time_and_force_wakes() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let device = context.device();
        let queue = context.queue();
        let state =
            GpuRigidStateSession::new(device, queue, &[body([0.0, 0.0, 1.0], [0.0; 3])]).unwrap();
        let contacts = GpuRigidSphereContacts::new(device, &state, &[1.0], &[], Some(5.0)).unwrap();
        let settings = SleepSettings {
            time_threshold: 0.025,
            ..Default::default()
        };
        for iteration in 0..3 {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Tessera automatic sleep step"),
            });
            contacts.encode(&mut encoder);
            contacts
                .encode_sleep(device, &mut encoder, 0.01, settings)
                .unwrap();
            let _submission = queue.submit(Some(encoder.finish()));
            let actual = state.readback(device, queue).unwrap()[0];
            assert_eq!(
                actual.inverse_inertia_sleep[3],
                if iteration == 2 { 1.0 } else { 0.0 }
            );
        }
        state
            .write_forces(
                queue,
                0,
                GpuRigidBodyForces {
                    force: [10.0, 0.0, 0.0, 0.0],
                    torque: [0.0; 4],
                },
            )
            .unwrap();
        state.step(device, queue, 0.01, [0.0; 3]).unwrap();
        let awake = state.readback(device, queue).unwrap()[0];
        assert_eq!(awake.inverse_inertia_sleep[3], 0.0);
        assert!(awake.linear_velocity[0] > 0.09);
    }

    #[test]
    fn automatic_sleep_follows_pair_updates_and_disable_setting() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let device = context.device();
        let queue = context.queue();
        let state = GpuRigidStateSession::new(
            device,
            queue,
            &[
                body([-1.0, 0.0, 3.0], [0.0; 3]),
                body([1.0, 0.0, 3.0], [0.0; 3]),
            ],
        )
        .unwrap();
        let mut contacts =
            GpuRigidSphereContacts::new(device, &state, &[1.0; 2], &[], None).unwrap();
        let settings = SleepSettings {
            time_threshold: 0.015,
            ..Default::default()
        };
        let encode = |contacts: &GpuRigidSphereContacts| {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Tessera pair support sleep step"),
            });
            contacts.encode(&mut encoder);
            contacts
                .encode_sleep(device, &mut encoder, 0.01, settings)
                .unwrap();
            let _submission = queue.submit(Some(encoder.finish()));
        };
        for _ in 0..3 {
            encode(&contacts);
        }
        assert!(
            state
                .readback(device, queue)
                .unwrap()
                .iter()
                .all(|body| body.inverse_inertia_sleep[3] == 0.0)
        );
        contacts
            .set_candidate_pairs(device, &[GpuPair { a: 0, b: 1 }])
            .unwrap();
        for _ in 0..2 {
            encode(&contacts);
        }
        assert!(
            state
                .readback(device, queue)
                .unwrap()
                .iter()
                .all(|body| body.inverse_inertia_sleep[3] == 1.0)
        );
        let mut disabled = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera disabled automatic sleep"),
        });
        contacts.encode(&mut disabled);
        contacts
            .encode_sleep(
                device,
                &mut disabled,
                0.01,
                SleepSettings {
                    enabled: false,
                    ..settings
                },
            )
            .unwrap();
        let _submission = queue.submit(Some(disabled.finish()));
        assert!(
            state
                .readback(device, queue)
                .unwrap()
                .iter()
                .all(|body| body.inverse_inertia_sleep[3] == 0.0)
        );
        for _ in 0..2 {
            encode(&contacts);
        }
        assert!(
            state
                .readback(device, queue)
                .unwrap()
                .iter()
                .all(|body| body.inverse_inertia_sleep[3] == 1.0)
        );
        contacts.set_candidate_pairs(device, &[]).unwrap();
        encode(&contacts);
        assert!(
            state
                .readback(device, queue)
                .unwrap()
                .iter()
                .all(|body| body.inverse_inertia_sleep[3] == 0.0)
        );

        assert!(matches!(
            contacts.reset_sleep_timer(queue, 2),
            Err(GpuRigidSphereContactError::InvalidInput)
        ));
        let mut invalid = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera invalid sleep settings"),
        });
        assert!(matches!(
            contacts.encode_sleep(
                device,
                &mut invalid,
                0.01,
                SleepSettings {
                    time_threshold: 0.0,
                    ..settings
                },
            ),
            Err(GpuRigidSphereContactError::InvalidInput)
        ));
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn dx12_resident_ground_sleep_uses_gpu_timer() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::DX12,
            ..Default::default()
        });
        let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
        else {
            eprintln!("DX12 adapter unavailable; skipping resident sleep test");
            return;
        };
        eprintln!(
            "Tessera resident sleep adapter: {:?}",
            adapter.get_info().backend
        );
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let state =
            GpuRigidStateSession::new(&device, &queue, &[body([0.0, 0.0, 1.0], [0.0; 3])]).unwrap();
        let contacts =
            GpuRigidSphereContacts::new(&device, &state, &[1.0], &[], Some(5.0)).unwrap();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera DX12 resident sleep"),
        });
        for _ in 0..2 {
            contacts.encode(&mut encoder);
            contacts
                .encode_sleep(
                    &device,
                    &mut encoder,
                    0.01,
                    SleepSettings {
                        time_threshold: 0.015,
                        ..Default::default()
                    },
                )
                .unwrap();
        }
        let _submission = queue.submit(Some(encoder.finish()));
        assert_eq!(
            state.readback(&device, &queue).unwrap()[0].inverse_inertia_sleep[3],
            1.0
        );
    }

    #[test]
    fn sliding_body_stays_awake_and_reset_clears_idle_history() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let device = context.device();
        let queue = context.queue();
        let state =
            GpuRigidStateSession::new(device, queue, &[body([0.0, 0.0, 1.0], [1.0, 0.0, 0.0])])
                .unwrap();
        let contacts = GpuRigidSphereContacts::new(device, &state, &[1.0], &[], Some(5.0)).unwrap();
        let settings = SleepSettings {
            time_threshold: 0.025,
            ..Default::default()
        };
        for _ in 0..5 {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Tessera sliding sphere sleep check"),
            });
            state
                .encode_step(device, &mut encoder, 0.01, [0.0; 3])
                .unwrap();
            contacts.encode(&mut encoder);
            contacts
                .encode_sleep(device, &mut encoder, 0.01, settings)
                .unwrap();
            let _submission = queue.submit(Some(encoder.finish()));
        }
        let moving = state.readback(device, queue).unwrap()[0];
        assert_eq!(moving.inverse_inertia_sleep[3], 0.0);
        assert!((moving.position_inverse_mass[0] - 0.05).abs() < 1e-4);
        state
            .write_body(
                queue,
                0,
                body([moving.position_inverse_mass[0], 0.0, 1.0], [0.0; 3]),
            )
            .unwrap();
        contacts.reset_sleep_timer(queue, 0).unwrap();
        for iteration in 0..3 {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Tessera reset idle timer check"),
            });
            contacts.encode(&mut encoder);
            contacts
                .encode_sleep(device, &mut encoder, 0.01, settings)
                .unwrap();
            let _submission = queue.submit(Some(encoder.finish()));
            let expected = if iteration == 2 { 1.0 } else { 0.0 };
            assert_eq!(
                state.readback(device, queue).unwrap()[0].inverse_inertia_sleep[3],
                expected
            );
        }
    }
}
