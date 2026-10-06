//! GPU-resident pair generation, broad phase, and narrow phase for spheres.
//!
//! The kernels run in submission order. The overlap buffer stays on the GPU;
//! callers can bind the returned contact buffer in a later compute pass.

use core::mem::{size_of, size_of_val};
use core::time::Duration;
use std::ffi::OsStr;
use std::sync::{Arc, mpsc};

use wgpu::util::DeviceExt;

use crate::gpu_box_box_contact::GpuBoxBoxContacts;
use crate::gpu_broad_phase::{GpuAabb, GpuPair};
use crate::gpu_contact_solver::GpuContactSolver;
use crate::gpu_convex_contact::GpuConvexContacts;
use crate::gpu_ground_contact::GpuGroundContacts;
use crate::gpu_lbvh::{GpuLbvh, GpuLbvhError};
use crate::gpu_sphere_box_contact::GpuSphereBoxContacts;
use crate::gpu_sphere_contact::{GpuSphere, GpuSphereContact};

/// Invalid inputs or device capacity exceeded.
#[derive(Debug, thiserror::Error)]
pub enum ContactPipelineError {
    /// A sphere or pair index is invalid, or a sphere AABB cannot be represented.
    #[error("invalid sphere or candidate pair")]
    InvalidInput,
    /// The input cannot fit the selected GPU's buffers or dispatch grid.
    #[error("contact batch exceeds GPU limits")]
    Capacity,
    /// The final contact readback did not complete.
    #[error("contact batch GPU readback failed: {0}")]
    Readback(String),
    /// No compatible compute adapter or device could be acquired.
    #[error("GPU contact device unavailable: {0}")]
    Device(String),
}

/// Owned GPU compute context for a Tessera contact world.
#[derive(Debug)]
pub struct GpuContactDevice {
    identity: Arc<()>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_info: wgpu::AdapterInfo,
    pipeline: GpuContactPipeline,
    ground: GpuGroundContacts,
    sphere_box: GpuSphereBoxContacts,
    box_box: GpuBoxBoxContacts,
    convex: GpuConvexContacts,
    solver: GpuContactSolver,
}

impl GpuContactDevice {
    /// Select a compute adapter and build the sphere contact kernels.
    ///
    /// Software adapters require `TESSERA_ALLOW_SOFTWARE_ADAPTER=1`, intended for
    /// correctness testing rather than real-time physics.
    pub fn new() -> Result<Self, ContactPipelineError> {
        Self::from_instance(wgpu::Instance::default())
    }

    /// Select an adapter from the requested wgpu backends.
    pub fn new_with_backends(backends: wgpu::Backends) -> Result<Self, ContactPipelineError> {
        Self::from_instance(wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        }))
    }

    fn from_instance(instance: wgpu::Instance) -> Result<Self, ContactPipelineError> {
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .map_err(|error| ContactPipelineError::Device(error.to_string()))?;
        let adapter_info = adapter.get_info();
        let software_adapter = std::env::var_os("TESSERA_ALLOW_SOFTWARE_ADAPTER");
        validate_adapter(adapter_info.device_type, software_adapter.as_deref())?;
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .map_err(|error| ContactPipelineError::Device(error.to_string()))?;
        let pipeline = GpuContactPipeline::new(&device);
        let ground = GpuGroundContacts::new(&device);
        let sphere_box = GpuSphereBoxContacts::new(&device);
        let box_box = GpuBoxBoxContacts::new(&device);
        let convex = GpuConvexContacts::new(&device);
        let solver = GpuContactSolver::new(&device);
        Ok(Self {
            identity: Arc::new(()),
            device,
            queue,
            adapter_info,
            pipeline,
            ground,
            sphere_box,
            box_box,
            convex,
            solver,
        })
    }

    /// GPU device used by this context.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub(crate) fn identity(&self) -> &Arc<()> {
        &self.identity
    }

    /// GPU submission queue used by this context.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Adapter selected for this device, including its backend and driver.
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }

    /// Contact pipeline compiled for this device.
    pub fn pipeline(&self) -> &GpuContactPipeline {
        &self.pipeline
    }

    /// Ground contact pipeline compiled for this device.
    pub fn ground(&self) -> &GpuGroundContacts {
        &self.ground
    }

    /// Sphere against oriented box narrow phase.
    pub fn sphere_box(&self) -> &GpuSphereBoxContacts {
        &self.sphere_box
    }

    /// Oriented box pair SAT narrow phase.
    pub fn box_box(&self) -> &GpuBoxBoxContacts {
        &self.box_box
    }

    /// Generic convex hull pair narrow phase.
    pub fn convex(&self) -> &GpuConvexContacts {
        &self.convex
    }

    /// Generalized GPU contact impulse solver.
    pub fn solver(&self) -> &GpuContactSolver {
        &self.solver
    }
}

fn validate_adapter(
    device_type: wgpu::DeviceType,
    software_adapter: Option<&OsStr>,
) -> Result<(), ContactPipelineError> {
    if device_type == wgpu::DeviceType::Cpu && software_adapter != Some(OsStr::new("1")) {
        return Err(ContactPipelineError::Device(
            "only a software compute adapter was available".into(),
        ));
    }
    Ok(())
}

/// A GPU-resident result with one contact slot per candidate pair.
#[derive(Debug)]
pub struct GpuContactBatch {
    /// Storage buffer containing `GpuPair` elements in contact order.
    pub pairs: wgpu::Buffer,
    /// Storage buffer containing `GpuSphereContact` elements in pair order.
    pub contacts: wgpu::Buffer,
    /// Number of valid elements in `contacts`.
    pub pair_count: u32,
}

impl GpuContactBatch {
    /// Transfer a finished batch to the CPU for the reference rigid-body solver.
    ///
    /// This is a synchronization point. A future GPU constraint solver can bind
    /// the public buffers directly and avoid this transfer.
    pub fn readback(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<(GpuPair, GpuSphereContact)>, ContactPipelineError> {
        if self.pair_count == 0 {
            return Ok(Vec::new());
        }
        let pair_bytes = u64::from(self.pair_count) * size_of::<GpuPair>() as u64;
        let contact_bytes = u64::from(self.pair_count) * size_of::<GpuSphereContact>() as u64;
        if pair_bytes + contact_bytes > device.limits().max_buffer_size {
            return Err(ContactPipelineError::Capacity);
        }
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera contact batch readback"),
            size: pair_bytes + contact_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera contact batch readback encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.pairs, 0, &staging, 0, pair_bytes);
        encoder.copy_buffer_to_buffer(&self.contacts, 0, &staging, pair_bytes, contact_bytes);
        let _submission = queue.submit(Some(encoder.finish()));
        let (sender, receiver) = mpsc::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        let _status = device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(5)),
            })
            .map_err(|error| ContactPipelineError::Readback(error.to_string()))?;
        receiver
            .recv_timeout(Duration::from_secs(5))
            .map_err(|error| ContactPipelineError::Readback(error.to_string()))?
            .map_err(|error| ContactPipelineError::Readback(error.to_string()))?;
        let view = staging.slice(..).get_mapped_range();
        let pair_slice = &view[..pair_bytes as usize];
        let contact_slice = &view[pair_bytes as usize..];
        let result = pair_slice
            .chunks_exact(size_of::<GpuPair>())
            .zip(contact_slice.chunks_exact(size_of::<GpuSphereContact>()))
            .map(|(pair, contact)| {
                (
                    bytemuck::pod_read_unaligned(pair),
                    bytemuck::pod_read_unaligned(contact),
                )
            })
            .collect();
        drop(view);
        staging.unmap();
        Ok(result)
    }
}

/// Reusable sphere contact pipeline with optional GPU pair generation.
#[derive(Debug)]
pub struct GpuContactPipeline {
    generate_pairs: wgpu::ComputePipeline,
    broad_phase: wgpu::ComputePipeline,
    narrow_phase: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    lbvh: GpuLbvh,
}

fn valid_sphere(sphere: &GpuSphere) -> bool {
    let [x, y, z, radius] = sphere.center_radius;
    radius.is_finite()
        && radius > 0.0
        && (radius + radius).is_finite()
        && [x, y, z].iter().all(|center| {
            center.is_finite() && (*center - radius).is_finite() && (*center + radius).is_finite()
        })
}

impl GpuContactPipeline {
    /// Compile both kernels for the selected wgpu device.
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera sphere contact pipeline"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_contact_pipeline.wgsl").into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Tessera sphere contact bind group layout"),
            entries: &[0, 1, 2, 3].map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: binding == 0,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Tessera sphere contact pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });
        let broad_phase = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tessera sphere broad phase"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("broad_phase"),
            compilation_options: Default::default(),
            cache: None,
        });
        let generate_pairs = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tessera sphere pair generation"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("generate_pairs"),
            compilation_options: Default::default(),
            cache: None,
        });
        let narrow_phase = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tessera sphere narrow phase"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("narrow_phase"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            generate_pairs,
            broad_phase,
            narrow_phase,
            bind_group_layout,
            lbvh: GpuLbvh::new(device),
        }
    }

    /// Submit both phases without an intermediate CPU readback.
    ///
    /// Pair order is preserved. Misses are represented by zeroed contact slots.
    /// The returned storage buffer can be consumed by subsequent GPU work.
    pub fn dispatch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        spheres: &[GpuSphere],
        pairs: &[GpuPair],
    ) -> Result<GpuContactBatch, ContactPipelineError> {
        let pair_count = u32::try_from(pairs.len()).map_err(|_| ContactPipelineError::Capacity)?;
        let limits = device.limits();
        let contact_bytes = u64::from(pair_count) * size_of::<GpuSphereContact>() as u64;
        let overlap_bytes = u64::from(pair_count) * size_of::<u32>() as u64;
        if pair_count.div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || [
                size_of_val(spheres) as u64,
                size_of_val(pairs) as u64,
                overlap_bytes,
                contact_bytes,
            ]
            .iter()
            .any(|size| {
                *size > u64::from(limits.max_storage_buffer_binding_size)
                    || *size > limits.max_buffer_size
            })
        {
            return Err(ContactPipelineError::Capacity);
        }
        if spheres.iter().any(|sphere| !valid_sphere(sphere))
            || pairs.iter().any(|pair| {
                if pair.a == pair.b
                    || pair.a as usize >= spheres.len()
                    || pair.b as usize >= spheres.len()
                {
                    return true;
                }
                let a = spheres[pair.a as usize].center_radius;
                let b = spheres[pair.b as usize].center_radius;
                !(a[3] + b[3]).is_finite() || (0..3).any(|axis| !(b[axis] - a[axis]).is_finite())
            })
        {
            return Err(ContactPipelineError::InvalidInput);
        }
        let pairs_buffer = if pairs.is_empty() {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Tessera empty pair buffer"),
                size: size_of::<GpuPair>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        } else {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tessera candidate pairs"),
                contents: bytemuck::cast_slice(pairs),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            })
        };
        Ok(self.run(device, queue, spheres, pairs_buffer, pair_count, false))
    }

    /// Build a generic GPU LBVH and transfer its compact AABB pair list.
    ///
    /// This synchronization point supports the current CPU constraint builder.
    /// At most 64 bounds should use the caller's small-scene path.
    pub fn dispatch_aabb_pairs(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bounds: &[GpuAabb],
    ) -> Result<Vec<GpuPair>, ContactPipelineError> {
        let result = self
            .lbvh
            .dispatch(device, queue, bounds)
            .map_err(map_lbvh_error)?;
        result.readback(device, queue).map_err(map_lbvh_error)
    }

    /// Generate candidate sphere pairs on the GPU, then run both contact phases.
    ///
    /// Scenes with at most 64 spheres preserve the brute-force lexicographic
    /// order `(0, 1), (0, 2), ...`. Larger scenes use Morton sorting and LBVH
    /// traversal, so only overlapping AABB pairs are returned and order is not
    /// specified.
    pub fn dispatch_all_pairs(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        spheres: &[GpuSphere],
    ) -> Result<GpuContactBatch, ContactPipelineError> {
        let count = u32::try_from(spheres.len()).map_err(|_| ContactPipelineError::Capacity)?;
        if spheres.iter().any(|sphere| !valid_sphere(sphere)) {
            return Err(ContactPipelineError::InvalidInput);
        }
        if count > 64 {
            let bounds = spheres
                .iter()
                .map(|sphere| {
                    let [x, y, z, radius] = sphere.center_radius;
                    GpuAabb {
                        lower: [x - radius, y - radius, z - radius, 0.0],
                        upper: [x + radius, y + radius, z + radius, 0.0],
                    }
                })
                .collect::<Vec<_>>();
            let candidates = self
                .lbvh
                .dispatch(device, queue, &bounds)
                .map_err(map_lbvh_error)?;
            return Ok(self.run(
                device,
                queue,
                spheres,
                candidates.pairs,
                candidates.pair_count,
                false,
            ));
        }
        let twice_pairs = count
            .checked_mul(count.saturating_sub(1))
            .ok_or(ContactPipelineError::Capacity)?;
        let pair_count = twice_pairs / 2;
        let limits = device.limits();
        if pair_count.div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || [
                size_of_val(spheres) as u64,
                u64::from(pair_count) * size_of::<GpuPair>() as u64,
                u64::from(pair_count) * size_of::<u32>() as u64,
                u64::from(pair_count) * size_of::<GpuSphereContact>() as u64,
            ]
            .iter()
            .any(|size| {
                *size > u64::from(limits.max_storage_buffer_binding_size)
                    || *size > limits.max_buffer_size
            })
        {
            return Err(ContactPipelineError::Capacity);
        }
        let pairs_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera generated pairs"),
            size: (u64::from(pair_count) * size_of::<GpuPair>() as u64)
                .max(size_of::<GpuPair>() as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Ok(self.run(device, queue, spheres, pairs_buffer, pair_count, true))
    }

    fn run(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        spheres: &[GpuSphere],
        pairs_buffer: wgpu::Buffer,
        pair_count: u32,
        generate_pairs: bool,
    ) -> GpuContactBatch {
        let contact_bytes = u64::from(pair_count) * size_of::<GpuSphereContact>() as u64;
        let overlap_bytes = u64::from(pair_count) * size_of::<u32>() as u64;
        let contacts = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera sphere contacts"),
            size: contact_bytes.max(size_of::<GpuSphereContact>() as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        if pair_count == 0 {
            return GpuContactBatch {
                pairs: pairs_buffer,
                contacts,
                pair_count,
            };
        }
        let spheres_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera spheres"),
            contents: bytemuck::cast_slice(spheres),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let overlaps = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera sphere AABB overlaps"),
            size: overlap_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera sphere contact data"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: spheres_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pairs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: overlaps.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: contacts.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera sphere contact encoder"),
        });
        if generate_pairs {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Tessera pair generation pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.generate_pairs);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(pair_count.div_ceil(64), 1, 1);
        }
        for pipeline in [&self.broad_phase, &self.narrow_phase] {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Tessera sphere contact pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(pair_count.div_ceil(64), 1, 1);
        }
        let _submission = queue.submit(Some(encoder.finish()));
        GpuContactBatch {
            pairs: pairs_buffer,
            contacts,
            pair_count,
        }
    }
}

fn map_lbvh_error(error: GpuLbvhError) -> ContactPipelineError {
    match error {
        GpuLbvhError::InvalidInput => ContactPipelineError::InvalidInput,
        GpuLbvhError::Capacity => ContactPipelineError::Capacity,
        GpuLbvhError::Readback(message) => ContactPipelineError::Readback(message),
        GpuLbvhError::State => {
            ContactPipelineError::Device("LBVH reusable state is unavailable".into())
        }
    }
}

#[cfg(test)]
mod tests {
    use core::time::Duration;
    use std::sync::mpsc;

    use super::*;

    #[test]
    fn software_adapter_requires_explicit_opt_in() {
        for (device_type, value, allowed) in [
            (wgpu::DeviceType::Cpu, None, false),
            (wgpu::DeviceType::Cpu, Some(""), false),
            (wgpu::DeviceType::Cpu, Some("0"), false),
            (wgpu::DeviceType::Cpu, Some("true"), false),
            (wgpu::DeviceType::Cpu, Some("1"), true),
            (wgpu::DeviceType::DiscreteGpu, None, true),
            (wgpu::DeviceType::IntegratedGpu, None, true),
            (wgpu::DeviceType::VirtualGpu, None, true),
            (wgpu::DeviceType::Other, None, true),
        ] {
            let result = validate_adapter(device_type, value.map(OsStr::new));
            assert_eq!(result.is_ok(), allowed, "{device_type:?} {value:?}");
        }
    }

    fn read_buffer<T: bytemuck::Pod>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
        count: u32,
    ) -> Vec<T> {
        if count == 0 {
            return Vec::new();
        }
        let size = u64::from(count) * size_of::<T>() as u64;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera test readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera test readback encoder"),
        });
        encoder.copy_buffer_to_buffer(buffer, 0, &readback, 0, size);
        let _submission = queue.submit(Some(encoder.finish()));
        let (sender, receiver) = mpsc::channel();
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        let _status = device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(5)),
            })
            .unwrap();
        receiver
            .recv_timeout(Duration::from_secs(5))
            .unwrap()
            .unwrap();
        let view = readback.slice(..).get_mapped_range();
        let result = view
            .chunks_exact(size_of::<T>())
            .map(bytemuck::pod_read_unaligned)
            .collect();
        drop(view);
        readback.unmap();
        result
    }

    #[tokio::test]
    async fn chained_gpu_phases_match_sphere_reference_across_workgroups() {
        let instance = wgpu::Instance::default();
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
        let spheres = [
            GpuSphere::new([0.0, 0.0, 0.0], 1.0).unwrap(),
            GpuSphere::new([1.5, 0.0, 0.0], 1.0).unwrap(),
            GpuSphere::new([2.0, 2.0, 0.0], 1.0).unwrap(),
            GpuSphere::new([3.1, 0.0, 0.0], 1.0).unwrap(),
            GpuSphere::new([0.0, 0.0, 0.0], 0.5).unwrap(),
            GpuSphere::new([2.0, 0.0, 0.0], 1.0).unwrap(),
        ];
        let pattern = [
            GpuPair { a: 0, b: 1 },
            GpuPair { a: 0, b: 2 },
            GpuPair { a: 0, b: 3 },
            GpuPair { a: 0, b: 4 },
            GpuPair { a: 0, b: 5 },
        ];
        let pairs: Vec<_> = (0..133)
            .map(|index| pattern[index % pattern.len()])
            .collect();
        let pipeline = GpuContactPipeline::new(&device);
        let batch = pipeline
            .dispatch(&device, &queue, &spheres, &pairs)
            .unwrap();
        let contacts: Vec<GpuSphereContact> =
            read_buffer(&device, &queue, &batch.contacts, batch.pair_count);
        assert_eq!(contacts.len(), pairs.len());
        for (index, contact) in contacts.iter().enumerate() {
            match index % pattern.len() {
                0 => {
                    assert!(contact.is_contact());
                    assert!((contact.depth_hit[0] - 0.5).abs() < 1e-6);
                    assert!((contact.normal[0] - 1.0).abs() < 1e-6);
                }
                3 => {
                    assert!(contact.is_contact());
                    assert!((contact.depth_hit[0] - 1.5).abs() < 1e-6);
                }
                4 => {
                    assert!(contact.is_contact());
                    assert_eq!(contact.depth_hit[0], 0.0);
                }
                _ => assert!(!contact.is_contact()),
            }
        }
        assert_eq!(
            pipeline
                .dispatch(&device, &queue, &[], &[])
                .unwrap()
                .pair_count,
            0
        );
        assert!(matches!(
            pipeline.dispatch(&device, &queue, &spheres, &[GpuPair { a: 1, b: 1 }]),
            Err(ContactPipelineError::InvalidInput)
        ));
    }

    #[tokio::test]
    async fn gpu_generated_pairs_match_cpu_order_and_contacts() {
        let instance = wgpu::Instance::default();
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
        let spheres: Vec<_> = (0..14)
            .map(|index| GpuSphere::new([index as f32 * 1.25, 0.0, 0.0], 0.75).unwrap())
            .collect();
        let pipeline = GpuContactPipeline::new(&device);
        let batch = pipeline
            .dispatch_all_pairs(&device, &queue, &spheres)
            .unwrap();
        assert_eq!(batch.pair_count, 91);
        let pairs: Vec<GpuPair> = read_buffer(&device, &queue, &batch.pairs, batch.pair_count);
        let contacts: Vec<GpuSphereContact> =
            read_buffer(&device, &queue, &batch.contacts, batch.pair_count);
        let expected_pairs: Vec<_> = (0..14)
            .flat_map(|a| (a + 1..14).map(move |b| GpuPair { a, b }))
            .collect();
        assert_eq!(pairs.len(), expected_pairs.len());
        for ((pair, expected), contact) in pairs.iter().zip(&expected_pairs).zip(&contacts) {
            assert_eq!((pair.a, pair.b), (expected.a, expected.b));
            let expected_hit = pair.b == pair.a + 1;
            assert_eq!(contact.is_contact(), expected_hit);
            if expected_hit {
                assert!((contact.depth_hit[0] - 0.25).abs() < 1e-6);
                assert!((contact.normal[0] - 1.0).abs() < 1e-6);
            }
        }
        assert_eq!(
            pipeline
                .dispatch_all_pairs(&device, &queue, &[])
                .unwrap()
                .pair_count,
            0
        );
        assert_eq!(
            pipeline
                .dispatch_all_pairs(&device, &queue, &spheres[..1])
                .unwrap()
                .pair_count,
            0
        );
        assert!(matches!(
            pipeline.dispatch_all_pairs(
                &device,
                &queue,
                &[GpuSphere {
                    center_radius: [0.0, 0.0, 0.0, f32::NAN],
                }]
            ),
            Err(ContactPipelineError::InvalidInput)
        ));
        let excessive = vec![spheres[0]; 65_537];
        assert!(matches!(
            pipeline.dispatch_all_pairs(&device, &queue, &excessive),
            Err(ContactPipelineError::Capacity)
        ));
    }

    #[tokio::test]
    async fn lbvh_candidates_feed_the_narrow_phase() {
        let instance = wgpu::Instance::default();
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
        let spheres: Vec<_> = (0..130)
            .map(|index| GpuSphere::new([index as f32, 0.0, 0.0], 0.51).unwrap())
            .collect();
        let batch = GpuContactPipeline::new(&device)
            .dispatch_all_pairs(&device, &queue, &spheres)
            .unwrap();
        assert_eq!(batch.pair_count, 129);
        let mut contacts = batch.readback(&device, &queue).unwrap();
        contacts.sort_by_key(|(pair, _)| (pair.a, pair.b));
        for (index, (pair, contact)) in contacts.iter().enumerate() {
            assert_eq!((pair.a, pair.b), (index as u32, index as u32 + 1));
            assert!(contact.is_contact());
            assert!((contact.depth_hit[0] - 0.02).abs() < 1e-5);
        }
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn dx12_runs_chained_contact_pipeline() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::DX12,
            ..Default::default()
        });
        let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
        else {
            eprintln!("DX12 adapter unavailable; skipping Tessera DX12 pipeline test");
            return;
        };
        let info = adapter.get_info();
        eprintln!(
            "Tessera DX12 adapter: {} ({:?})",
            info.name, info.device_type
        );
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let spheres = [
            GpuSphere::new([0.0, 0.0, 0.0], 1.0).unwrap(),
            GpuSphere::new([1.5, 0.0, 0.0], 1.0).unwrap(),
        ];
        let pairs = [GpuPair { a: 0, b: 1 }];
        let batch = GpuContactPipeline::new(&device)
            .dispatch(&device, &queue, &spheres, &pairs)
            .unwrap();
        let contacts: Vec<GpuSphereContact> =
            read_buffer(&device, &queue, &batch.contacts, batch.pair_count);
        assert_eq!(contacts.len(), 1);
        assert!(contacts[0].is_contact());
        assert!((contacts[0].depth_hit[0] - 0.5).abs() < 1e-6);
        let generated = GpuContactPipeline::new(&device)
            .dispatch_all_pairs(&device, &queue, &spheres)
            .unwrap();
        let generated_pairs: Vec<GpuPair> =
            read_buffer(&device, &queue, &generated.pairs, generated.pair_count);
        assert_eq!(generated_pairs.len(), 1);
        assert_eq!((generated_pairs[0].a, generated_pairs[0].b), (0, 1));
        let generated_contacts: Vec<GpuSphereContact> =
            read_buffer(&device, &queue, &generated.contacts, generated.pair_count);
        assert!(generated_contacts[0].is_contact());
    }
}
