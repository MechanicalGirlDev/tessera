//! Vendor-neutral GPU linear bounding-volume hierarchy for arbitrary AABBs.
//!
//! The pipeline follows the same domain, Morton, radix, Karras build, refit,
//! and traversal phases used by Nexus. Candidate storage grows and traversal
//! is retried when the first capacity estimate is too small.

use core::mem::{size_of, size_of_val};
use core::time::Duration;
use std::sync::{Mutex, mpsc};

use wgpu::util::DeviceExt;

use crate::gpu_broad_phase::{GpuAabb, GpuPair};

const WORKGROUP_SIZE: u32 = 64;
const NODE_SIZE: u64 = 64;
const BRUTE_FORCE_MAX_COLLIDERS: u32 = 64;

#[derive(Clone, Copy)]
enum PairFilter<'a> {
    EnvironmentIds(&'a wgpu::Buffer),
    CollisionGroups(&'a wgpu::Buffer),
}

impl<'a> PairFilter<'a> {
    fn buffer(self) -> &'a wgpu::Buffer {
        match self {
            Self::EnvironmentIds(buffer) | Self::CollisionGroups(buffer) => buffer,
        }
    }

    fn mode(self) -> u32 {
        match self {
            Self::EnvironmentIds(_) => 1,
            Self::CollisionGroups(_) => 2,
        }
    }

    fn bytes_per_collider(self) -> u64 {
        match self {
            Self::EnvironmentIds(_) => size_of::<u32>() as u64,
            Self::CollisionGroups(_) => size_of::<[u32; 4]>() as u64,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct LbvhParams {
    count: u32,
    pair_capacity: u32,
    radix_shift: u32,
    filter_groups: u32,
}

/// GPU LBVH input, capacity, or readback failure.
#[derive(Debug, thiserror::Error)]
pub enum GpuLbvhError {
    /// The broad phase received too few or invalid AABBs.
    #[error("invalid LBVH AABB input")]
    InvalidInput,
    /// Tree or candidate buffers cannot fit the selected GPU.
    #[error("LBVH buffers exceed GPU limits")]
    Capacity,
    /// The candidate counter could not be transferred from the GPU.
    #[error("LBVH counter readback failed: {0}")]
    Readback(String),
    /// Internal reusable state could not be locked.
    #[error("LBVH reusable state is unavailable")]
    State,
}

/// GPU-resident candidate pair buffer produced by LBVH traversal.
#[derive(Debug)]
pub struct GpuLbvhPairs {
    /// Compact `GpuPair` storage containing `pair_count` valid elements.
    pub pairs: wgpu::Buffer,
    /// Number of valid candidate pairs.
    pub pair_count: u32,
    /// Capacity used by the final traversal.
    pub pair_capacity: u32,
}

/// Independently owned resident tree for scene-query consumers.
/// Root is node zero; leaves start at `collider_count - 1`.
#[derive(Debug)]
pub struct GpuLbvhTree {
    /// 64-byte nodes: lower vec4, upper vec4, then left, right, parent,
    /// first/last sorted range, and three padding words. A leaf's left is its
    /// original collider ID; upper.w is one for leaves and zero for internal nodes.
    pub nodes: wgpu::Buffer,
    /// Number of indexed colliders, at least two.
    pub collider_count: u32,
}

impl GpuLbvhPairs {
    /// Transfer the compact candidate list to a CPU contact builder.
    pub fn readback(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<GpuPair>, GpuLbvhError> {
        if self.pair_count == 0 {
            return Ok(Vec::new());
        }
        let size = u64::from(self.pair_count) * size_of::<GpuPair>() as u64;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera LBVH pair readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera LBVH pair readback encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.pairs, 0, &readback, 0, size);
        let _submission = queue.submit(Some(encoder.finish()));
        map_buffer(device, &readback)?;
        let view = readback.slice(..).get_mapped_range();
        let pairs = view
            .chunks_exact(size_of::<GpuPair>())
            .map(bytemuck::pod_read_unaligned)
            .collect();
        drop(view);
        readback.unmap();
        Ok(pairs)
    }
}

/// Candidate pairs and their counter remain on the GPU after an encoded LBVH build.
///
/// The caller must submit the encoder before reading these buffers. The first
/// counter word is the valid pair count; the second is the overflow flag.
#[derive(Clone, Debug)]
pub struct GpuLbvhResidentPairs {
    /// Pair storage for up to `pair_capacity` elements.
    pub pairs: wgpu::Buffer,
    /// Two `u32` words containing the pair count and overflow flag.
    pub counter: wgpu::Buffer,
    /// Maximum number of candidate pairs that can be stored.
    pub pair_capacity: u32,
    /// Number of input colliders used for this build.
    pub collider_count: u32,
}

impl GpuLbvhResidentPairs {
    /// Read only the valid pair count and overflow flag after submission.
    pub fn readback_count(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<u32, GpuLbvhError> {
        let [count, overflow] = read_counter(device, queue, &self.counter)?;
        if overflow != 0 || count > self.pair_capacity {
            return Err(GpuLbvhError::Capacity);
        }
        Ok(count)
    }

    /// Read the count and pairs explicitly for diagnostics or a CPU fallback.
    pub fn readback(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<GpuPair>, GpuLbvhError> {
        let count = self.readback_count(device, queue)?;
        GpuLbvhPairs {
            pairs: self.pairs.clone(),
            pair_count: count,
            pair_capacity: self.pair_capacity,
        }
        .readback(device, queue)
    }
}

/// Reusable LBVH compute pipelines.
#[derive(Debug)]
pub struct GpuLbvh {
    compute_domain: wgpu::ComputePipeline,
    compute_morton: wgpu::ComputePipeline,
    radix_histogram: wgpu::ComputePipeline,
    radix_scan: wgpu::ComputePipeline,
    radix_scatter: wgpu::ComputePipeline,
    build_tree: wgpu::ComputePipeline,
    refit_leaves: wgpu::ComputePipeline,
    refit_internal: wgpu::ComputePipeline,
    find_pairs: wgpu::ComputePipeline,
    dispatch_args: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    scratch: Mutex<LbvhScratch>,
}

#[derive(Debug)]
struct LbvhScratch {
    collider_capacity: u32,
    allocation_generation: u64,
    domain: wgpu::Buffer,
    morton_a: wgpu::Buffer,
    morton_b: wgpu::Buffer,
    tree: wgpu::Buffer,
    radix_offsets: wgpu::Buffer,
    counter: wgpu::Buffer,
}

impl LbvhScratch {
    fn new(device: &wgpu::Device, collider_capacity: u32, allocation_generation: u64) -> Self {
        let morton_bytes = u64::from(collider_capacity) * 8;
        let node_count = collider_capacity.saturating_mul(2).saturating_sub(1);
        let tree_bytes = u64::from(node_count) * NODE_SIZE;
        let radix_bytes = u64::from(collider_capacity.div_ceil(256)) * 16 * size_of::<u32>() as u64;
        Self {
            collider_capacity,
            allocation_generation,
            domain: create_storage(device, "Tessera LBVH domain", 32),
            morton_a: create_storage(device, "Tessera LBVH Morton A", morton_bytes),
            morton_b: create_storage(device, "Tessera LBVH Morton B", morton_bytes),
            tree: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Tessera LBVH tree"),
                size: tree_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            radix_offsets: create_storage(device, "Tessera LBVH radix offsets", radix_bytes),
            counter: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Tessera LBVH pair counter"),
                size: 8,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        }
    }

    fn ensure_capacity(&mut self, device: &wgpu::Device, collider_count: u32) {
        if collider_count > self.collider_capacity {
            *self = Self::new(device, collider_count, self.allocation_generation + 1);
        }
    }
}

impl GpuLbvh {
    /// Compile the LBVH phases for one wgpu device.
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera GPU LBVH"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_lbvh.wgsl").into()),
        });
        let indirect_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera LBVH indirect dispatch args"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_lbvh_indirect.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Tessera GPU LBVH layout"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, false),
                storage_entry(2, false),
                storage_entry(3, false),
                storage_entry(4, false),
                storage_entry(5, false),
                storage_entry(6, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage_entry(8, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Tessera GPU LBVH pipeline layout"),
            bind_group_layouts: &[&layout],
            immediate_size: 0,
        });
        let pipeline = |entry_point, label| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Self {
            compute_domain: pipeline("compute_domain", "Tessera LBVH domain"),
            compute_morton: pipeline("compute_morton", "Tessera LBVH Morton codes"),
            radix_histogram: pipeline("radix_histogram", "Tessera LBVH radix histogram"),
            radix_scan: pipeline("radix_scan", "Tessera LBVH radix scan"),
            radix_scatter: pipeline("radix_scatter", "Tessera LBVH radix scatter"),
            build_tree: pipeline("build_tree", "Tessera LBVH Karras build"),
            refit_leaves: pipeline("refit_leaves", "Tessera LBVH leaf refit"),
            refit_internal: pipeline("refit_internal", "Tessera LBVH internal refit"),
            find_pairs: pipeline("find_pairs", "Tessera LBVH traversal"),
            dispatch_args: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Tessera LBVH candidate dispatch args"),
                layout: None,
                module: &indirect_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            }),
            layout,
            scratch: Mutex::new(LbvhScratch::new(device, 1, 0)),
        }
    }

    /// Build and traverse an LBVH, returning compact overlapping AABB pairs.
    ///
    /// This path is intended for more than 64 colliders. Smaller scenes use the
    /// contact pipeline's brute-force path, matching Nexus' crossover policy.
    pub fn dispatch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bounds: &[GpuAabb],
    ) -> Result<GpuLbvhPairs, GpuLbvhError> {
        let count = u32::try_from(bounds.len()).map_err(|_| GpuLbvhError::Capacity)?;
        if count <= BRUTE_FORCE_MAX_COLLIDERS || bounds.iter().any(|bound| !valid_aabb(bound)) {
            return Err(GpuLbvhError::InvalidInput);
        }
        if !buffer_fits(size_of_val(bounds) as u64, &device.limits()) {
            return Err(GpuLbvhError::Capacity);
        }
        let bounds_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera LBVH AABBs"),
            contents: bytemuck::cast_slice(bounds),
            usage: wgpu::BufferUsages::STORAGE,
        });
        self.dispatch_buffer(device, queue, &bounds_buffer, count)
    }

    /// Build candidates from AABBs already stored on the GPU.
    ///
    /// The caller must have written finite, ordered AABBs before this call.
    /// Small resident scenes also use this path to avoid a state readback.
    pub(crate) fn dispatch_buffer(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bounds_buffer: &wgpu::Buffer,
        count: u32,
    ) -> Result<GpuLbvhPairs, GpuLbvhError> {
        self.dispatch_buffer_impl(device, queue, bounds_buffer, count, None)
    }

    /// Filter overlapping candidates by GPU-resident collider group IDs.
    pub(crate) fn dispatch_buffer_grouped(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bounds_buffer: &wgpu::Buffer,
        group_ids: &wgpu::Buffer,
        count: u32,
    ) -> Result<GpuLbvhPairs, GpuLbvhError> {
        self.dispatch_buffer_impl(
            device,
            queue,
            bounds_buffer,
            count,
            Some(PairFilter::EnvironmentIds(group_ids)),
        )
    }

    /// Filter candidates by environment and reciprocal collision masks on the GPU.
    pub(crate) fn dispatch_buffer_filtered(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bounds_buffer: &wgpu::Buffer,
        filters: &wgpu::Buffer,
        count: u32,
    ) -> Result<GpuLbvhPairs, GpuLbvhError> {
        self.dispatch_buffer_impl(
            device,
            queue,
            bounds_buffer,
            count,
            Some(PairFilter::CollisionGroups(filters)),
        )
    }

    /// Encode an LBVH build without submitting or reading back its candidate count.
    ///
    /// Candidate storage is sized for the worst case, so this path is suitable
    /// only when that allocation fits the device. The returned pair and counter
    /// buffers can be consumed by later passes in the same command encoder.
    /// Bounds must contain finite, ordered AABBs at execution time.
    pub fn encode_buffer_resident(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        bounds_buffer: &wgpu::Buffer,
        count: u32,
    ) -> Result<GpuLbvhResidentPairs, GpuLbvhError> {
        self.encode_buffer_resident_impl(device, encoder, bounds_buffer, count, None)
    }

    /// Encode resident candidates with environment IDs and reciprocal masks.
    ///
    /// Each filter element contains `[environment_id, memberships, filter, 0]`.
    pub fn encode_buffer_resident_filtered(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        bounds_buffer: &wgpu::Buffer,
        filters: &wgpu::Buffer,
        count: u32,
    ) -> Result<GpuLbvhResidentPairs, GpuLbvhError> {
        self.encode_buffer_resident_impl(
            device,
            encoder,
            bounds_buffer,
            count,
            Some(PairFilter::CollisionGroups(filters)),
        )
    }

    /// Build an independently owned tree without finding pairs or allocating
    /// quadratic candidate storage. Bounds must be finite and ordered at execution.
    /// The input must have STORAGE usage and contain at least `count` AABBs.
    /// No submission or CPU readback is performed; later passes may use the tree.
    pub fn encode_tree_resident(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        bounds: &wgpu::Buffer,
        count: u32,
    ) -> Result<GpuLbvhTree, GpuLbvhError> {
        if count < 2 {
            return Err(GpuLbvhError::InvalidInput);
        }
        let nodes = count
            .checked_mul(2)
            .and_then(|n| n.checked_sub(1))
            .ok_or(GpuLbvhError::Capacity)?;
        let limits = device.limits();
        let bounds_bytes = u64::from(count) * size_of::<GpuAabb>() as u64;
        let radix_groups = count.div_ceil(256);
        if bounds.size() < bounds_bytes
            || !bounds.usage().contains(wgpu::BufferUsages::STORAGE)
            || count.div_ceil(WORKGROUP_SIZE) > limits.max_compute_workgroups_per_dimension
            || [
                bounds_bytes,
                u64::from(count) * 8,
                u64::from(nodes) * NODE_SIZE,
                u64::from(radix_groups) * 64,
            ]
            .iter()
            .any(|&bytes| !buffer_fits(bytes, &limits))
        {
            return Err(GpuLbvhError::Capacity);
        }
        // Dedicated allocations prevent a later build from replacing a prepared
        // query's nodes, unlike the scratch storage used by contact candidates.
        let scratch = LbvhScratch::new(device, count, 0);
        let (_, bindings, _) = self.resources(
            device,
            bounds,
            &scratch.domain,
            &scratch.morton_a,
            &scratch.morton_b,
            &scratch.tree,
            &scratch.radix_offsets,
            &scratch.counter,
            count,
            1,
            None,
        )?;
        self.encode_tree_phases(encoder, &bindings, count, radix_groups);
        Ok(GpuLbvhTree {
            nodes: scratch.tree,
            collider_count: count,
        })
    }

    /// Encode an indirect compute dispatch sized to the resident candidate count.
    ///
    /// The returned buffer contains three `u32` dispatch dimensions at offset
    /// zero. Encode this after the LBVH build and pass it to
    /// `ComputePass::dispatch_workgroups_indirect` in the same encoder.
    pub fn encode_candidate_dispatch_args(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        candidates: &GpuLbvhResidentPairs,
        workgroup_size: u32,
    ) -> Result<wgpu::Buffer, GpuLbvhError> {
        if workgroup_size == 0
            || candidates.pair_capacity.div_ceil(workgroup_size)
                > device.limits().max_compute_workgroups_per_dimension
        {
            return Err(GpuLbvhError::Capacity);
        }
        let args = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera LBVH indirect candidate dispatch"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera LBVH indirect dispatch parameters"),
            contents: bytemuck::cast_slice(&[workgroup_size, candidates.pair_capacity, 0, 0]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera LBVH indirect dispatch inputs"),
            layout: &self.dispatch_args.get_bind_group_layout(0),
            entries: &[
                binding(0, &candidates.counter),
                binding(1, &args),
                binding(2, &params),
            ],
        });
        encode_pass(encoder, &self.dispatch_args, &bind_group, 1);
        Ok(args)
    }

    fn encode_buffer_resident_impl(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        bounds_buffer: &wgpu::Buffer,
        count: u32,
        pair_filter: Option<PairFilter<'_>>,
    ) -> Result<GpuLbvhResidentPairs, GpuLbvhError> {
        if count < 2 {
            return Err(GpuLbvhError::InvalidInput);
        }
        let pair_capacity = count
            .checked_mul(count - 1)
            .map(|twice| twice / 2)
            .ok_or(GpuLbvhError::Capacity)?;
        let node_count = count
            .checked_mul(2)
            .and_then(|value| value.checked_sub(1))
            .ok_or(GpuLbvhError::Capacity)?;
        let limits = device.limits();
        let bounds_bytes = u64::from(count) * size_of::<GpuAabb>() as u64;
        let morton_bytes = u64::from(count) * 8;
        let tree_bytes = u64::from(node_count) * NODE_SIZE;
        let radix_groups = count.div_ceil(256);
        let radix_bytes = u64::from(radix_groups) * 16 * size_of::<u32>() as u64;
        let pair_bytes = u64::from(pair_capacity) * size_of::<GpuPair>() as u64;
        if bounds_buffer.size() < bounds_bytes
            || pair_filter.is_some_and(|filter| {
                filter.buffer().size() < u64::from(count) * filter.bytes_per_collider()
            })
            || count.div_ceil(WORKGROUP_SIZE) > limits.max_compute_workgroups_per_dimension
            || [
                bounds_bytes,
                morton_bytes,
                tree_bytes,
                radix_bytes,
                pair_bytes,
            ]
            .iter()
            .any(|&size| !buffer_fits(size, &limits))
        {
            return Err(GpuLbvhError::Capacity);
        }

        let mut scratch = self.scratch.lock().map_err(|_| GpuLbvhError::State)?;
        scratch.ensure_capacity(device, count);
        let counter = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera resident LBVH pair counter"),
            size: 8,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.clear_buffer(&counter, 0, None);
        let (pairs, bind_groups, traversal) = self.resources(
            device,
            bounds_buffer,
            &scratch.domain,
            &scratch.morton_a,
            &scratch.morton_b,
            &scratch.tree,
            &scratch.radix_offsets,
            &counter,
            count,
            pair_capacity,
            pair_filter,
        )?;
        self.encode_build(
            encoder,
            &bind_groups,
            traversal.as_ref(),
            count,
            radix_groups,
        );
        Ok(GpuLbvhResidentPairs {
            pairs,
            counter,
            pair_capacity,
            collider_count: count,
        })
    }

    fn dispatch_buffer_impl(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bounds_buffer: &wgpu::Buffer,
        count: u32,
        pair_filter: Option<PairFilter<'_>>,
    ) -> Result<GpuLbvhPairs, GpuLbvhError> {
        if count < 2 {
            return Err(GpuLbvhError::InvalidInput);
        }
        let maximum_pairs = count
            .checked_mul(count - 1)
            .map(|twice| twice / 2)
            .ok_or(GpuLbvhError::Capacity)?;
        let node_count = count
            .checked_mul(2)
            .and_then(|value| value.checked_sub(1))
            .ok_or(GpuLbvhError::Capacity)?;
        let limits = device.limits();
        let bounds_bytes = u64::from(count) * size_of::<GpuAabb>() as u64;
        let morton_bytes = u64::from(count) * 8;
        let tree_bytes = u64::from(node_count) * NODE_SIZE;
        let radix_groups = count.div_ceil(256);
        let radix_bytes = u64::from(radix_groups) * 16 * size_of::<u32>() as u64;
        if bounds_buffer.size() < bounds_bytes
            || pair_filter.is_some_and(|filter| {
                filter.buffer().size() < u64::from(count) * filter.bytes_per_collider()
            })
            || count.div_ceil(WORKGROUP_SIZE) > limits.max_compute_workgroups_per_dimension
            || [bounds_bytes, morton_bytes, tree_bytes, radix_bytes]
                .iter()
                .any(|&size| !buffer_fits(size, &limits))
        {
            return Err(GpuLbvhError::Capacity);
        }

        let mut scratch = self.scratch.lock().map_err(|_| GpuLbvhError::State)?;
        scratch.ensure_capacity(device, count);
        let initial_capacity = maximum_pairs.min(count.saturating_mul(8).max(1));
        reset_counter(queue, &scratch.counter);
        let (mut pairs, first_bind_groups, first_traversal) = self.resources(
            device,
            bounds_buffer,
            &scratch.domain,
            &scratch.morton_a,
            &scratch.morton_b,
            &scratch.tree,
            &scratch.radix_offsets,
            &scratch.counter,
            count,
            initial_capacity,
            pair_filter,
        )?;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera LBVH build encoder"),
        });
        self.encode_build(
            &mut encoder,
            &first_bind_groups,
            first_traversal.as_ref(),
            count,
            radix_groups,
        );
        let _submission = queue.submit(Some(encoder.finish()));
        let [pair_count, overflow] = read_counter(device, queue, &scratch.counter)?;
        if overflow == 0 && pair_count <= initial_capacity {
            return Ok(GpuLbvhPairs {
                pairs,
                pair_count,
                pair_capacity: initial_capacity,
            });
        }
        if pair_count > maximum_pairs {
            return Err(GpuLbvhError::Capacity);
        }

        let grown_capacity = pair_count
            .max(1)
            .checked_next_power_of_two()
            .unwrap_or(maximum_pairs)
            .min(maximum_pairs);
        reset_counter(queue, &scratch.counter);
        let grown = self.resources(
            device,
            bounds_buffer,
            &scratch.domain,
            &scratch.morton_a,
            &scratch.morton_b,
            &scratch.tree,
            &scratch.radix_offsets,
            &scratch.counter,
            count,
            grown_capacity,
            pair_filter,
        )?;
        pairs = grown.0;
        let bind_groups = grown.1;
        let traversal = grown.2;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera LBVH resized traversal encoder"),
        });
        encode_pass(
            &mut encoder,
            &self.find_pairs,
            traversal.as_ref().unwrap_or(&bind_groups[0]),
            count.div_ceil(WORKGROUP_SIZE),
        );
        let _submission = queue.submit(Some(encoder.finish()));
        let [resized_count, resized_overflow] = read_counter(device, queue, &scratch.counter)?;
        if resized_overflow != 0 || resized_count > grown_capacity {
            return Err(GpuLbvhError::Capacity);
        }
        Ok(GpuLbvhPairs {
            pairs,
            pair_count: resized_count,
            pair_capacity: grown_capacity,
        })
    }

    fn encode_build(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_groups: &[wgpu::BindGroup],
        traversal: Option<&wgpu::BindGroup>,
        count: u32,
        radix_groups: u32,
    ) {
        self.encode_tree_phases(encoder, bind_groups, count, radix_groups);
        encode_pass(
            encoder,
            &self.find_pairs,
            traversal.unwrap_or(&bind_groups[0]),
            count.div_ceil(WORKGROUP_SIZE),
        );
    }

    fn encode_tree_phases(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_groups: &[wgpu::BindGroup],
        count: u32,
        radix_groups: u32,
    ) {
        encode_pass(encoder, &self.compute_domain, &bind_groups[0], 1);
        encode_pass(
            encoder,
            &self.compute_morton,
            &bind_groups[0],
            count.div_ceil(WORKGROUP_SIZE),
        );
        for bind_group in bind_groups {
            encode_pass(encoder, &self.radix_histogram, bind_group, radix_groups);
            encode_pass(encoder, &self.radix_scan, bind_group, 1);
            encode_pass(encoder, &self.radix_scatter, bind_group, radix_groups);
        }
        encode_pass(
            encoder,
            &self.build_tree,
            &bind_groups[0],
            (count - 1).div_ceil(WORKGROUP_SIZE),
        );
        encode_pass(
            encoder,
            &self.refit_leaves,
            &bind_groups[0],
            count.div_ceil(WORKGROUP_SIZE),
        );
        encode_pass(
            encoder,
            &self.refit_internal,
            &bind_groups[0],
            (count - 1).div_ceil(WORKGROUP_SIZE),
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn resources(
        &self,
        device: &wgpu::Device,
        bounds: &wgpu::Buffer,
        domain: &wgpu::Buffer,
        morton_a: &wgpu::Buffer,
        morton_b: &wgpu::Buffer,
        tree: &wgpu::Buffer,
        radix_offsets: &wgpu::Buffer,
        counter: &wgpu::Buffer,
        count: u32,
        pair_capacity: u32,
        pair_filter: Option<PairFilter<'_>>,
    ) -> Result<(wgpu::Buffer, Vec<wgpu::BindGroup>, Option<wgpu::BindGroup>), GpuLbvhError> {
        let pair_bytes = u64::from(pair_capacity) * size_of::<GpuPair>() as u64;
        if !buffer_fits(pair_bytes, &device.limits()) {
            return Err(GpuLbvhError::Capacity);
        }
        let pairs = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera LBVH candidate pairs"),
            size: pair_bytes.max(size_of::<GpuPair>() as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut bind_groups = Vec::with_capacity(8);
        for pass in 0..8u32 {
            let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tessera LBVH parameters"),
                contents: bytemuck::bytes_of(&LbvhParams {
                    count,
                    pair_capacity,
                    radix_shift: pass * 4,
                    filter_groups: 0,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let (input, output) = if pass % 2 == 0 {
                (morton_a, morton_b)
            } else {
                (morton_b, morton_a)
            };
            bind_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Tessera LBVH resources"),
                layout: &self.layout,
                entries: &[
                    binding(0, bounds),
                    binding(1, domain),
                    binding(2, input),
                    binding(3, output),
                    binding(4, tree),
                    binding(5, &pairs),
                    binding(6, counter),
                    binding(7, &params),
                    binding(8, radix_offsets),
                ],
            }));
        }
        let traversal = pair_filter.map(|filter| {
            let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tessera grouped LBVH parameters"),
                contents: bytemuck::bytes_of(&LbvhParams {
                    count,
                    pair_capacity,
                    radix_shift: 0,
                    filter_groups: filter.mode(),
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Tessera grouped LBVH traversal"),
                layout: &self.layout,
                entries: &[
                    binding(0, bounds),
                    binding(1, domain),
                    binding(2, morton_a),
                    binding(3, morton_b),
                    binding(4, tree),
                    binding(5, &pairs),
                    binding(6, counter),
                    binding(7, &params),
                    binding(8, filter.buffer()),
                ],
            })
        });
        Ok((pairs, bind_groups, traversal))
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn binding<'a>(binding: u32, buffer: &'a wgpu::Buffer) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn create_storage(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    })
}

fn buffer_fits(size: u64, limits: &wgpu::Limits) -> bool {
    size <= u64::from(limits.max_storage_buffer_binding_size) && size <= limits.max_buffer_size
}

fn valid_aabb(bound: &GpuAabb) -> bool {
    (0..3).all(|axis| {
        bound.lower[axis].is_finite()
            && bound.upper[axis].is_finite()
            && bound.lower[axis] <= bound.upper[axis]
    })
}

fn encode_pass(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    workgroups: u32,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("Tessera LBVH phase"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(workgroups, 1, 1);
}

fn reset_counter(queue: &wgpu::Queue, counter: &wgpu::Buffer) {
    queue.write_buffer(counter, 0, bytemuck::cast_slice(&[0u32, 0u32]));
}

fn read_counter(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    counter: &wgpu::Buffer,
) -> Result<[u32; 2], GpuLbvhError> {
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Tessera LBVH counter readback"),
        size: 8,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Tessera LBVH counter readback encoder"),
    });
    encoder.copy_buffer_to_buffer(counter, 0, &readback, 0, 8);
    let _submission = queue.submit(Some(encoder.finish()));
    map_buffer(device, &readback)?;
    let view = readback.slice(..).get_mapped_range();
    let result = [
        u32::from_le_bytes([view[0], view[1], view[2], view[3]]),
        u32::from_le_bytes([view[4], view[5], view[6], view[7]]),
    ];
    drop(view);
    readback.unmap();
    Ok(result)
}

fn map_buffer(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Result<(), GpuLbvhError> {
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
        .map_err(|error| GpuLbvhError::Readback(error.to_string()))?;
    receiver
        .recv_timeout(Duration::from_secs(5))
        .map_err(|error| GpuLbvhError::Readback(error.to_string()))?
        .map_err(|error| GpuLbvhError::Readback(error.to_string()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    fn read_pairs(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        result: &GpuLbvhPairs,
    ) -> Vec<GpuPair> {
        result.readback(device, queue).unwrap()
    }

    async fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .ok()?;
        adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .ok()
    }

    fn sphere_aabb(center: [f32; 3], radius: f32) -> GpuAabb {
        GpuAabb::new(
            [center[0] - radius, center[1] - radius, center[2] - radius],
            [center[0] + radius, center[1] + radius, center[2] + radius],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn encoded_resident_pairs_keep_independent_counts_and_filter_on_gpu() {
        let Some((device, queue)) = device().await else {
            return;
        };
        let bounds = (0..70)
            .map(|index| sphere_aabb([index as f32, 0.0, 0.0], 0.51))
            .collect::<Vec<_>>();
        let bounds_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Resident LBVH test bounds"),
            contents: bytemuck::cast_slice(&bounds),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let filters = (0..70)
            .map(|index| [u32::from(index >= 35), 1, 1, 0])
            .collect::<Vec<_>>();
        let filter_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Resident LBVH test filters"),
            contents: bytemuck::cast_slice(&filters),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let lbvh = GpuLbvh::new(&device);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Resident LBVH test encoder"),
        });
        let all = lbvh
            .encode_buffer_resident(&device, &mut encoder, &bounds_buffer, 70)
            .unwrap();
        let grouped = lbvh
            .encode_buffer_resident_filtered(
                &device,
                &mut encoder,
                &bounds_buffer,
                &filter_buffer,
                70,
            )
            .unwrap();
        let indirect = lbvh
            .encode_candidate_dispatch_args(&device, &mut encoder, &grouped, 64)
            .unwrap();
        let consumer_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Resident LBVH indirect test consumer"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
struct Pair { a: u32, b: u32 }
@group(0) @binding(0) var<storage, read> pairs: array<Pair>;
@group(0) @binding(1) var<storage, read> counter: array<u32>;
@group(0) @binding(2) var<storage, read_write> visited: array<atomic<u32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < counter[0] && pairs[id.x].a != pairs[id.x].b) {
        atomicAdd(&visited[0], 1u);
    }
}
"#
                .into(),
            ),
        });
        let consumer = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Resident LBVH indirect test consumer"),
            layout: None,
            module: &consumer_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let visited = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Resident LBVH visited count"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let visited_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Resident LBVH visited readback"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.clear_buffer(&visited, 0, None);
        let consumer_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Resident LBVH indirect test inputs"),
            layout: &consumer.get_bind_group_layout(0),
            entries: &[
                binding(0, &grouped.pairs),
                binding(1, &grouped.counter),
                binding(2, &visited),
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Resident LBVH indirect test pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&consumer);
            pass.set_bind_group(0, &consumer_group, &[]);
            pass.dispatch_workgroups_indirect(&indirect, 0);
        }
        encoder.copy_buffer_to_buffer(&visited, 0, &visited_readback, 0, 4);
        assert_eq!(all.pair_capacity, 70 * 69 / 2);
        assert_eq!(grouped.pair_capacity, all.pair_capacity);
        let _submission = queue.submit(Some(encoder.finish()));
        map_buffer(&device, &visited_readback).unwrap();
        let view = visited_readback.slice(..).get_mapped_range();
        assert_eq!(u32::from_le_bytes(view[..4].try_into().unwrap()), 68);
        drop(view);
        visited_readback.unmap();
        let all_pairs: BTreeSet<_> = all
            .readback(&device, &queue)
            .unwrap()
            .into_iter()
            .map(|pair| (pair.a, pair.b))
            .collect();
        let grouped_pairs: BTreeSet<_> = grouped
            .readback(&device, &queue)
            .unwrap()
            .into_iter()
            .map(|pair| (pair.a, pair.b))
            .collect();
        assert_eq!(all_pairs, (0..69).map(|index| (index, index + 1)).collect());
        assert_eq!(grouped_pairs.len(), 68);
        assert!(!grouped_pairs.contains(&(34, 35)));
        assert!(grouped_pairs.is_subset(&all_pairs));
    }

    #[tokio::test]
    async fn sparse_scene_emits_only_overlapping_aabbs() {
        let Some((device, queue)) = device().await else {
            return;
        };
        let bounds: Vec<_> = (0..130)
            .map(|index| sphere_aabb([index as f32, 0.0, 0.0], 0.51))
            .collect();
        let result = GpuLbvh::new(&device)
            .dispatch(&device, &queue, &bounds)
            .unwrap();
        let actual: BTreeSet<_> = read_pairs(&device, &queue, &result)
            .into_iter()
            .map(|pair| (pair.a, pair.b))
            .collect();
        let expected: BTreeSet<_> = (0..129).map(|index| (index, index + 1)).collect();
        assert_eq!(actual, expected);
        assert_eq!(result.pair_count, 129);
        assert!(result.pair_count < 130 * 129 / 2);
    }

    #[tokio::test]
    async fn dense_duplicate_morton_codes_grow_candidate_storage() {
        let Some((device, queue)) = device().await else {
            return;
        };
        let bounds = vec![sphere_aabb([0.0; 3], 1.0); 70];
        let result = GpuLbvh::new(&device)
            .dispatch(&device, &queue, &bounds)
            .unwrap();
        let pairs = read_pairs(&device, &queue, &result);
        let unique: BTreeSet<_> = pairs.iter().map(|pair| (pair.a, pair.b)).collect();
        assert_eq!(result.pair_count, 70 * 69 / 2);
        assert!(result.pair_capacity >= result.pair_count);
        assert_eq!(unique.len(), result.pair_count as usize);
    }

    #[tokio::test]
    async fn grouped_traversal_filters_before_counting_candidates() {
        let Some((device, queue)) = device().await else {
            return;
        };
        let bounds = vec![sphere_aabb([0.0; 3], 1.0); 100];
        let bounds_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grouped LBVH test AABBs"),
            contents: bytemuck::cast_slice(&bounds),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let group_ids: Vec<u32> = (0..100).collect();
        let group_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grouped LBVH test group IDs"),
            contents: bytemuck::cast_slice(&group_ids),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let lbvh = GpuLbvh::new(&device);
        let distinct = lbvh
            .dispatch_buffer_grouped(&device, &queue, &bounds_buffer, &group_buffer, 100)
            .unwrap();
        assert_eq!(distinct.pair_count, 0);
        assert_eq!(distinct.pair_capacity, 800);
        assert!(read_pairs(&device, &queue, &distinct).is_empty());

        let group_ids: Vec<u32> = (0..100).map(|index| index / 50).collect();
        let group_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grouped LBVH test two group IDs"),
            contents: bytemuck::cast_slice(&group_ids),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let grouped = lbvh
            .dispatch_buffer_grouped(&device, &queue, &bounds_buffer, &group_buffer, 100)
            .unwrap();
        assert_eq!(grouped.pair_count, 2 * 50 * 49 / 2);
        let pairs = read_pairs(&device, &queue, &grouped);
        assert!(
            pairs
                .iter()
                .all(|pair| group_ids[pair.a as usize] == group_ids[pair.b as usize])
        );
    }

    #[tokio::test]
    async fn varied_scene_matches_cpu_aabb_pairs() {
        let Some((device, queue)) = device().await else {
            return;
        };
        let mut state = 0x1234_5678u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state as f32 / u32::MAX as f32
        };
        let bounds: Vec<_> = (0..257)
            .map(|_| {
                let center = [
                    next() * 20.0 - 10.0,
                    next() * 12.0 - 6.0,
                    next() * 8.0 - 4.0,
                ];
                sphere_aabb(center, 0.15 + next() * 0.85)
            })
            .collect();
        let result = GpuLbvh::new(&device)
            .dispatch(&device, &queue, &bounds)
            .unwrap();
        let actual: BTreeSet<_> = read_pairs(&device, &queue, &result)
            .into_iter()
            .map(|pair| (pair.a, pair.b))
            .collect();
        let expected: BTreeSet<_> = (0..bounds.len())
            .flat_map(|a| {
                let bounds = &bounds;
                (a + 1..bounds.len()).filter_map(move |b| {
                    let first = &bounds[a];
                    let second = &bounds[b];
                    (0..3)
                        .all(|axis| {
                            first.lower[axis] <= second.upper[axis]
                                && second.lower[axis] <= first.upper[axis]
                        })
                        .then_some((a as u32, b as u32))
                })
            })
            .collect();
        assert_eq!(actual, expected);
        assert_eq!(actual.len(), result.pair_count as usize);
    }

    #[tokio::test]
    async fn many_radix_workgroups_preserve_stable_candidates() {
        let Some((device, queue)) = device().await else {
            return;
        };
        const COUNT: u32 = 4_096;
        const MULTIPLIER: u32 = 4_051;
        let bounds: Vec<_> = (0..COUNT)
            .map(|index| {
                let position = index.wrapping_mul(MULTIPLIER) % COUNT;
                sphere_aabb([position as f32, 0.0, 0.0], 0.51)
            })
            .collect();
        let mut collider_at = vec![0u32; COUNT as usize];
        for collider in 0..COUNT {
            let position = collider.wrapping_mul(MULTIPLIER) % COUNT;
            collider_at[position as usize] = collider;
        }
        let expected: BTreeSet<_> = (0..COUNT - 1)
            .map(|position| {
                let first = collider_at[position as usize];
                let second = collider_at[position as usize + 1];
                (first.min(second), first.max(second))
            })
            .collect();
        let result = GpuLbvh::new(&device)
            .dispatch(&device, &queue, &bounds)
            .unwrap();
        let actual: BTreeSet<_> = read_pairs(&device, &queue, &result)
            .into_iter()
            .map(|pair| (pair.a, pair.b))
            .collect();
        assert_eq!(actual, expected);
        assert_eq!(result.pair_count, COUNT - 1);
    }

    #[tokio::test]
    async fn scratch_buffers_reuse_capacity_and_grow_monotonically() {
        let Some((device, queue)) = device().await else {
            return;
        };
        let lbvh = GpuLbvh::new(&device);
        let bounds = |count: usize| {
            (0..count)
                .map(|index| sphere_aabb([index as f32 * 2.0, 0.0, 0.0], 0.5))
                .collect::<Vec<_>>()
        };
        let _result = lbvh.dispatch(&device, &queue, &bounds(130)).unwrap();
        let (first_capacity, first_generation) = {
            let scratch = lbvh.scratch.lock().unwrap();
            (scratch.collider_capacity, scratch.allocation_generation)
        };
        let _result = lbvh.dispatch(&device, &queue, &bounds(100)).unwrap();
        {
            let scratch = lbvh.scratch.lock().unwrap();
            assert_eq!(scratch.collider_capacity, first_capacity);
            assert_eq!(scratch.allocation_generation, first_generation);
        }
        let _result = lbvh.dispatch(&device, &queue, &bounds(257)).unwrap();
        let scratch = lbvh.scratch.lock().unwrap();
        assert_eq!(scratch.collider_capacity, 257);
        assert_eq!(scratch.allocation_generation, first_generation + 1);
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn dx12_runs_sparse_lbvh() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::DX12,
            ..Default::default()
        });
        let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
        else {
            return;
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let bounds: Vec<_> = (0..70)
            .map(|index| sphere_aabb([index as f32, 0.0, 0.0], 0.51))
            .collect();
        let result = GpuLbvh::new(&device)
            .dispatch(&device, &queue, &bounds)
            .unwrap();
        let actual: BTreeSet<_> = read_pairs(&device, &queue, &result)
            .into_iter()
            .map(|pair| (pair.a, pair.b))
            .collect();
        let expected: BTreeSet<_> = (0..69).map(|index| (index, index + 1)).collect();
        assert_eq!(actual, expected);
    }
}
