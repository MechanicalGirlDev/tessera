//! GPU construction of an open-addressed sparse MPM stencil grid.
use crate::gpu::GpuMpmError;
use core::mem::size_of;
use wgpu::util::DeviceExt;

#[derive(Clone, Debug)]
pub(crate) struct HashTopology {
    generate: wgpu::ComputePipeline,
    insert: wgpu::ComputePipeline,
    resolve: wgpu::ComputePipeline,
    remap: wgpu::ComputePipeline,
    indirect: wgpu::ComputePipeline,
}

#[derive(Debug)]
pub(crate) struct HashTopologyState {
    owners: wgpu::Buffer,
    indirect: wgpu::Buffer,
    bindings: wgpu::BindGroup,
    slots: u32,
    buckets: u32,
}

impl HashTopology {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera MPM sparse grid topology"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_topology.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Tessera MPM topology layout"),
            entries: &core::array::from_fn::<_, 9, _>(|index| wgpu::BindGroupLayoutEntry {
                binding: index as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if index == 5 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: index == 0,
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Tessera MPM topology pipelines"),
            bind_group_layouts: &[&layout],
            immediate_size: 0,
        });
        let pipeline = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Self {
            generate: pipeline("generate"),
            insert: pipeline("insert"),
            resolve: pipeline("resolve"),
            remap: pipeline("remap"),
            indirect: pipeline("indirect"),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare(
        &self,
        device: &wgpu::Device,
        particles: &wgpu::Buffer,
        indices: &wgpu::Buffer,
        coordinates: &wgpu::Buffer,
        grid: &wgpu::Buffer,
        indirect: &wgpu::Buffer,
        count: u32,
        stride: u32,
        base_offset: u32,
        buckets: u32,
    ) -> Result<HashTopologyState, GpuMpmError> {
        let slots = count.checked_mul(27).ok_or(GpuMpmError::Capacity)?;
        let candidate_bytes = u64::from(slots) * size_of::<[i32; 4]>() as u64;
        let limits = device.limits();
        if candidate_bytes > u64::from(limits.max_storage_buffer_binding_size)
            || candidate_bytes > limits.max_buffer_size
            || u64::from(buckets) * 4 > u64::from(limits.max_storage_buffer_binding_size)
            || u64::from(buckets) * 4 > limits.max_buffer_size
            || buckets.div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || slots.div_ceil(64) > limits.max_compute_workgroups_per_dimension
        {
            return Err(GpuMpmError::Capacity);
        }
        let candidates = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera MPM stencil coordinates"),
            size: candidate_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let owners = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera MPM hash owners"),
            size: u64::from(buckets) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let compact_ids = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera MPM compact node IDs"),
            size: u64::from(buckets) * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera MPM topology parameters"),
            contents: bytemuck::cast_slice(&[count, stride, base_offset, buckets]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera MPM topology bindings"),
            layout: &self.generate.get_bind_group_layout(0),
            entries: &[
                particles,
                &candidates,
                &owners,
                indices,
                coordinates,
                &params,
                &compact_ids,
                indirect,
                grid,
            ]
            .iter()
            .enumerate()
            .map(|(index, buffer)| wgpu::BindGroupEntry {
                binding: index as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>(),
        });
        Ok(HashTopologyState {
            owners,
            indirect: indirect.clone(),
            bindings,
            slots,
            buckets,
        })
    }

    pub(crate) fn encode(&self, encoder: &mut wgpu::CommandEncoder, state: &HashTopologyState) {
        encoder.clear_buffer(&state.owners, 0, None);
        encoder.clear_buffer(&state.indirect, 0, None);
        for (pipeline, count) in [
            (&self.generate, state.slots),
            (&self.insert, state.slots),
            (&self.resolve, state.buckets),
            (&self.remap, state.slots),
            (&self.indirect, 1),
        ] {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &state.bindings, &[]);
            pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
        }
    }
}

impl HashTopologyState {
    pub(crate) fn indirect(&self) -> &wgpu::Buffer {
        &self.indirect
    }
}
