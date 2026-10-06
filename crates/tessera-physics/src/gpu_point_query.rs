//! Batched nearest-point queries against live resident rigid-body geometry.
use crate::gpu_rigid_sphere_contact::{
    GpuRigidCollisionGroups, GpuRigidSphereContacts, map_readback,
};
use wgpu::util::DeviceExt;

/// Point projection and collision filtering for one GPU query.
#[derive(Clone, Copy, Debug)]
pub struct GpuRigidPoint {
    /// Finite world-space point.
    pub point: [f32; 3],
    /// Finite inclusive nonnegative distance bound.
    pub max_distance: f32,
    /// Bilateral collider collision group filter.
    pub groups: GpuRigidCollisionGroups,
    /// Optional dense body index to skip. Ground is not a body index.
    pub excluded_body: Option<u32>,
    /// Optional half-open dense body range [start, end], for independent environments.
    pub body_range: Option<[u32; 2]>,
    /// Return the input point when inside a volume.
    pub solid: bool,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PointData {
    point_max: [f32; 4],
    direction: [f32; 4],
    filter: [u32; 4],
    range: [u32; 4],
}

/// GPU-compatible nearest projection; inspect `ids[2]` before other fields.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuRigidPointHit {
    /// World XYZ and unsigned distance.
    pub point_distance: [f32; 4],
    /// Unit direction from the boundary toward an exterior query, or outward
    /// from an interior query toward its nearest boundary. Zero at coincidence.
    pub normal: [f32; 4],
    /// Dense body ID (u32::MAX for ground), feature ID, hit flag, volume containment flag.
    pub ids: [u32; 4],
}
/// Point query validation, allocation, or readback failure.
#[derive(Debug, thiserror::Error)]
pub enum GpuPointQueryError {
    /// Input or device limits are invalid.
    #[error("invalid GPU point query or capacity")]
    InvalidInput,
    /// Resident bounds or scene-tree construction failed.
    #[error("GPU point scene index: {0}")]
    SceneIndex(String),
    /// GPU result mapping failed.
    #[error("GPU point query readback: {0}")]
    Readback(String),
}

/// Immutable point batch bound to live state, shape and collider-group buffers.
/// Recreate after contact or body buffer replacement. Ground settings are captured at creation.
/// Encoding does not submit or map: results can be consumed by another GPU pass.
#[derive(Debug)]
pub struct GpuRigidPointQueries {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    output: wgpu::Buffer,
    count: u32,
}
impl GpuRigidPointQueries {
    /// Validate all points and bind geometry without reading body state back.
    pub fn new(
        device: &wgpu::Device,
        contacts: &GpuRigidSphereContacts,
        rays: &[GpuRigidPoint],
    ) -> Result<Self, GpuPointQueryError> {
        Self::new_impl(device, contacts, rays, None)
    }

    /// Bind an independently built scene tree to accelerate nearest-point search.
    /// Tree leaves must use the same dense body IDs as `contacts`, and conservative
    /// world bounds for the state observed by this pass. Rebuild the tree after
    /// motion, before encoding queries; stale bounds can omit the nearest body.
    pub fn with_scene_tree(
        device: &wgpu::Device,
        contacts: &GpuRigidSphereContacts,
        points: &[GpuRigidPoint],
        tree: &crate::gpu_lbvh::GpuLbvhTree,
    ) -> Result<Self, GpuPointQueryError> {
        Self::new_impl(device, contacts, points, Some(tree))
    }

    fn new_impl(
        device: &wgpu::Device,
        contacts: &GpuRigidSphereContacts,
        rays: &[GpuRigidPoint],
        tree: Option<&crate::gpu_lbvh::GpuLbvhTree>,
    ) -> Result<Self, GpuPointQueryError> {
        let count = u32::try_from(rays.len()).map_err(|_| GpuPointQueryError::InvalidInput)?;
        let body_count =
            u32::try_from(contacts.body_count()).map_err(|_| GpuPointQueryError::InvalidInput)?;
        let limits = device.limits();
        let bytes = u64::from(count.max(1)) * 64;
        if count.div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || bytes > limits.max_buffer_size
            || bytes > u64::from(limits.max_storage_buffer_binding_size)
            || limits.max_storage_buffers_per_shader_stage < 8
        {
            return Err(GpuPointQueryError::InvalidInput);
        }
        if tree.is_some_and(|tree| {
            tree.collider_count != body_count
                || body_count < 2
                || tree.nodes.size() < (u64::from(body_count) * 2 - 1) * 64
                || !tree.nodes.usage().contains(wgpu::BufferUsages::STORAGE)
                || tree.nodes.size() > u64::from(limits.max_storage_buffer_binding_size)
        }) {
            return Err(GpuPointQueryError::InvalidInput);
        }
        let dummy_tree = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera empty point scene tree"),
            size: 64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let mut data = Vec::with_capacity(rays.len().max(1));
        for ray in rays {
            let range = ray.body_range.unwrap_or([0, body_count]);
            if range[0] > range[1] || range[1] > body_count {
                return Err(GpuPointQueryError::InvalidInput);
            }
            if !ray.max_distance.is_finite()
                || ray.max_distance < 0.0
                || ray.point.iter().any(|v| !v.is_finite())
                || ray.excluded_body.is_some_and(|index| index >= body_count)
            {
                return Err(GpuPointQueryError::InvalidInput);
            }
            data.push(PointData {
                point_max: [ray.point[0], ray.point[1], ray.point[2], ray.max_distance],
                direction: [0.0; 4],
                range: [range[0], range[1], 0, 0],
                filter: [
                    ray.groups.memberships,
                    ray.groups.filter,
                    ray.excluded_body.unwrap_or(u32::MAX),
                    u32::from(ray.solid),
                ],
            });
        }
        if data.is_empty() {
            data.push(bytemuck::Zeroable::zeroed());
        }
        let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera points"),
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera point hits"),
            size: u64::from(count.max(1)) * 48,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let (ground, extent, groups) = contacts.ray_query_ground();
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera point counts"),
            contents: bytemuck::cast_slice(&[
                body_count,
                count,
                u32::from(ground),
                extent.to_bits(),
                groups.memberships,
                groups.filter,
                u32::from(tree.is_some()),
                0,
            ]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let geometry = contacts.ray_query_buffers(device);
        let pipeline = contacts.point_query_pipeline(device);
        let layout = pipeline.get_bind_group_layout(0);
        let buffers = [
            &geometry[0],
            &geometry[1],
            &geometry[2],
            &geometry[3],
            &geometry[4],
            &input,
            &output,
            &params,
            tree.map_or(&dummy_tree, |tree| &tree.nodes),
        ];
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera point bindings"),
            layout: &layout,
            entries: &buffers
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: b.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        Ok(Self {
            pipeline,
            bind_group,
            output,
            count,
        })
    }
    /// Append the query pass after any physics state updates in the same encoder.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        if self.count == 0 {
            return;
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Tessera points"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.count.div_ceil(64), 1, 1);
    }
    /// Nearest-hit storage for downstream GPU consumers, 48 bytes per point.
    pub fn result_buffer(&self) -> &wgpu::Buffer {
        &self.output
    }
    /// Copy and map an already encoded query result. Does not rerun queries.
    pub fn readback(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<GpuRigidPointHit>, GpuPointQueryError> {
        if self.count == 0 {
            return Ok(Vec::new());
        }
        let bytes = u64::from(self.count) * 48;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera point staging"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.output, 0, &staging, 0, bytes);
        let _ = queue.submit(Some(encoder.finish()));
        map_readback(device, &staging)
            .map_err(|error| GpuPointQueryError::Readback(error.to_string()))?;
        let view = staging.slice(..).get_mapped_range();
        let hits = view
            .chunks_exact(48)
            .map(bytemuck::pod_read_unaligned)
            .collect();
        drop(view);
        staging.unmap();
        Ok(hits)
    }
}

pub(crate) fn create_pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Tessera point layout"),
        entries: &(0..9)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 7 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: binding != 6,
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect::<Vec<_>>(),
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Tessera point queries"),
        source: wgpu::ShaderSource::Wgsl(include_str!("gpu_point_query.wgsl").into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Tessera point pipeline layout"),
        bind_group_layouts: &[&layout],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Tessera point queries"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("point_main"),
        compilation_options: Default::default(),
        cache: None,
    })
}
