//! Batched ray queries against live resident rigid-body geometry.
use crate::gpu_rigid_sphere_contact::{
    GpuRigidCollisionGroups, GpuRigidSphereContacts, map_readback,
};
use wgpu::util::DeviceExt;

/// Ray parameterization and collision filtering for one GPU query.
#[derive(Clone, Copy, Debug)]
pub struct GpuRigidRay {
    /// Finite world origin.
    pub origin: [f32; 3],
    /// Finite nonzero direction; need not have unit length.
    pub direction: [f32; 3],
    /// Inclusive nonnegative upper bound on the ray parameter.
    pub max_t: f32,
    /// Bilateral collider collision group filter.
    pub groups: GpuRigidCollisionGroups,
    /// Optional dense body index to skip. Ground is not a body index.
    pub excluded_body: Option<u32>,
    /// Optional half-open dense body range [start, end], for independent environments.
    pub body_range: Option<[u32; 2]>,
    /// Return parameter zero when starting inside a volume.
    pub solid: bool,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RayData {
    origin_max: [f32; 4],
    direction: [f32; 4],
    filter: [u32; 4],
    range: [u32; 4],
}

/// GPU-compatible nearest intersection; inspect `ids[2]` before other fields.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuRigidRayHit {
    /// World XYZ and ray parameter t.
    pub point_toi: [f32; 4],
    /// World normal, zero for inside-solid hits and line segments.
    pub normal: [f32; 4],
    /// Dense body ID (u32::MAX for ground), feature ID, hit flag, inside-solid flag.
    pub ids: [u32; 4],
}
/// Ray query validation, allocation, or readback failure.
#[derive(Debug, thiserror::Error)]
pub enum GpuRayQueryError {
    /// Input or device limits are invalid.
    #[error("invalid GPU ray query or capacity")]
    InvalidInput,
    /// Resident bounds or scene-tree construction failed.
    #[error("GPU ray scene index: {0}")]
    SceneIndex(String),
    /// GPU result mapping failed.
    #[error("GPU ray query readback: {0}")]
    Readback(String),
}

/// Immutable ray batch bound to live state, shape and collider-group buffers.
/// Recreate after contact or body buffer replacement. Ground settings are captured at creation.
/// Encoding does not submit or map: results can be consumed by another GPU pass.
#[derive(Debug)]
pub struct GpuRigidRayQueries {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    output: wgpu::Buffer,
    count: u32,
}
impl GpuRigidRayQueries {
    /// Validate all rays and bind geometry without reading body state back.
    pub fn new(
        device: &wgpu::Device,
        contacts: &GpuRigidSphereContacts,
        rays: &[GpuRigidRay],
    ) -> Result<Self, GpuRayQueryError> {
        Self::new_impl(device, contacts, rays, None)
    }

    /// Bind a scene tree with conservative current-state bounds and matching dense body IDs.
    /// Rebuild after motion and encode that build before queries; stale bounds can miss hits.
    pub fn with_scene_tree(
        device: &wgpu::Device,
        contacts: &GpuRigidSphereContacts,
        rays: &[GpuRigidRay],
        tree: &crate::gpu_lbvh::GpuLbvhTree,
    ) -> Result<Self, GpuRayQueryError> {
        Self::new_impl(device, contacts, rays, Some(tree))
    }

    fn new_impl(
        device: &wgpu::Device,
        contacts: &GpuRigidSphereContacts,
        rays: &[GpuRigidRay],
        tree: Option<&crate::gpu_lbvh::GpuLbvhTree>,
    ) -> Result<Self, GpuRayQueryError> {
        let count = u32::try_from(rays.len()).map_err(|_| GpuRayQueryError::InvalidInput)?;
        let body_count =
            u32::try_from(contacts.body_count()).map_err(|_| GpuRayQueryError::InvalidInput)?;
        let limits = device.limits();
        let bytes = u64::from(count.max(1)) * 64;
        if count.div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || bytes > limits.max_buffer_size
            || bytes > u64::from(limits.max_storage_buffer_binding_size)
            || limits.max_storage_buffers_per_shader_stage < 8
        {
            return Err(GpuRayQueryError::InvalidInput);
        }
        if tree.is_some_and(|tree| {
            tree.collider_count != body_count
                || body_count < 2
                || tree.nodes.size() < (u64::from(body_count) * 2 - 1) * 64
                || !tree.nodes.usage().contains(wgpu::BufferUsages::STORAGE)
                || tree.nodes.size() > u64::from(limits.max_storage_buffer_binding_size)
        }) {
            return Err(GpuRayQueryError::InvalidInput);
        }
        let dummy_tree = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera empty ray scene tree"),
            size: 64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let mut data = Vec::with_capacity(rays.len().max(1));
        for ray in rays {
            let range = ray.body_range.unwrap_or([0, body_count]);
            if range[0] > range[1] || range[1] > body_count {
                return Err(GpuRayQueryError::InvalidInput);
            }
            let length = ray.direction.iter().map(|v| v * v).sum::<f32>();
            if !ray.max_t.is_finite()
                || ray.max_t < 0.0
                || !length.is_finite()
                || length <= 0.0
                || ray
                    .origin
                    .iter()
                    .chain(ray.direction.iter())
                    .any(|v| !v.is_finite())
                || (0..3)
                    .any(|axis| !(ray.origin[axis] + ray.max_t * ray.direction[axis]).is_finite())
                || ray.excluded_body.is_some_and(|index| index >= body_count)
            {
                return Err(GpuRayQueryError::InvalidInput);
            }
            data.push(RayData {
                origin_max: [ray.origin[0], ray.origin[1], ray.origin[2], ray.max_t],
                direction: [ray.direction[0], ray.direction[1], ray.direction[2], 0.0],
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
            label: Some("Tessera rays"),
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera ray hits"),
            size: u64::from(count.max(1)) * 48,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let (ground, extent, groups) = contacts.ray_query_ground();
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera ray counts"),
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
        let pipeline = contacts.ray_query_pipeline(device);
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
            label: Some("Tessera ray bindings"),
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
            label: Some("Tessera rays"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.count.div_ceil(64), 1, 1);
    }
    /// Nearest-hit storage for downstream GPU consumers, 48 bytes per ray.
    pub fn result_buffer(&self) -> &wgpu::Buffer {
        &self.output
    }
    /// Copy and map an already encoded query result. Does not rerun queries.
    pub fn readback(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<GpuRigidRayHit>, GpuRayQueryError> {
        if self.count == 0 {
            return Ok(Vec::new());
        }
        let bytes = u64::from(self.count) * 48;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera ray staging"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.output, 0, &staging, 0, bytes);
        let _ = queue.submit(Some(encoder.finish()));
        map_readback(device, &staging)
            .map_err(|error| GpuRayQueryError::Readback(error.to_string()))?;
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
        label: Some("Tessera ray layout"),
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
        label: Some("Tessera ray queries"),
        source: wgpu::ShaderSource::Wgsl(include_str!("gpu_ray_query.wgsl").into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Tessera ray pipeline layout"),
        bind_group_layouts: &[&layout],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Tessera ray queries"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("ray_main"),
        compilation_options: Default::default(),
        cache: None,
    })
}
