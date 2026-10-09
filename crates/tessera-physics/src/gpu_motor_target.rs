//! Storage-buffer targets for the resident articulated scalar motors.

use wgpu::util::DeviceExt;

/// Which existing native motor target receives an action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuMotorTargetMode {
    /// Position target; the native motor must already have position control enabled.
    Position,
    /// Velocity target, including the feed-forward target of a position motor.
    Velocity,
}

/// One action row's destination in an environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuMotorTargetMapping {
    /// Generalized coordinate, including six leading slots for a floating root.
    pub coordinate: usize,
    /// Stable environment-local child link of the scalar joint.
    /// Scaled or offset mimic links are not accepted.
    pub link: usize,
    /// Target component to replace, retaining gains and effort cap.
    pub mode: GpuMotorTargetMode,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct PackedMotorTargetMapping {
    pub coordinate: u32,
    pub action: u32,
    pub mode: u32,
    pub delay: u32,
}

const _: () = assert!(size_of::<PackedMotorTargetMapping>() == 16);

#[derive(Debug)]
pub(crate) struct GpuMotorTargetControl {
    latch: wgpu::ComputePipeline,
    apply: wgpu::ComputePipeline,
    delay_update: wgpu::ComputePipeline,
    control_layout: wgpu::Buffer,
    mappings: wgpu::Buffer,
    pending: wgpu::Buffer,
    parameters: wgpu::Buffer,
    count: u32,
    destinations: Vec<PackedMotorTargetMapping>,
}

impl GpuMotorTargetControl {
    pub(crate) fn action_bytes(&self) -> u64 {
        u64::from(self.count) * 4
    }

    pub(crate) fn new(
        device: &wgpu::Device,
        parameters: &wgpu::Buffer,
        mappings: &[PackedMotorTargetMapping],
        environment_count: u32,
    ) -> Self {
        let mappings_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera motor target mapping"),
            contents: bytemuck::cast_slice(mappings),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let pending = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera pending motor targets"),
            size: mappings_buffer.size(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera motor target scatter"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_motor_target.wgsl").into()),
        });
        let pipeline = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Self {
            latch: pipeline("latch_targets"),
            apply: pipeline("apply_targets"),
            delay_update: pipeline("update_delays"),
            control_layout: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tessera resident motor control layout"),
                contents: bytemuck::cast_slice(&[environment_count, 0, 0, 0]),
                usage: wgpu::BufferUsages::UNIFORM,
            }),
            mappings: mappings_buffer,
            pending,
            parameters: parameters.clone(),
            count: mappings.len() as u32,
            destinations: mappings.to_vec(),
        }
    }

    pub(crate) fn accepts_inputs(
        &self,
        inputs: &[Vec<crate::gpu_articulated_joint_force::GpuJointForceInput>],
    ) -> bool {
        let coordinates = inputs.iter().flatten().collect::<Vec<_>>();
        self.destinations.iter().all(|mapping| {
            coordinates
                .get(mapping.coordinate as usize)
                .and_then(|input| input.motor)
                .is_some_and(|motor| mapping.mode != 0 || motor.position_target.is_some())
        })
    }

    pub(crate) fn encode_latch(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        actions: &wgpu::Buffer,
    ) {
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera motor target latch"),
            layout: &self.latch.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.mappings.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.pending.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: actions.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&self.latch);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(self.count.div_ceil(64), 1, 1);
    }

    pub(crate) fn encode_delays(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        delays: &wgpu::Buffer,
    ) {
        let buffers = [
            (0, &self.mappings),
            (1, &self.pending),
            (4, delays),
            (5, &self.control_layout),
        ];
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera motor substep delay update"),
            layout: &self.delay_update.get_bind_group_layout(0),
            entries: &buffers
                .iter()
                .map(|&(binding, buffer)| wgpu::BindGroupEntry {
                    binding,
                    resource: buffer.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&self.delay_update);
        pass.set_bind_group(0, &bindings, &[]);
        pass.dispatch_workgroups(self.count.div_ceil(64), 1, 1);
    }

    pub(crate) fn encode_apply(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        let layout = self.apply.get_bind_group_layout(0);
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera motor target application"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.mappings.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.pending.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.parameters.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&self.apply);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(self.count.div_ceil(64), 1, 1);
    }

    pub(crate) fn cancel(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.pending, 0, &vec![0; self.pending.size() as usize]);
    }

    pub(crate) fn encode_cancel_environment(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        environment: usize,
        environment_count: usize,
    ) {
        let size = self.pending.size() / environment_count as u64;
        encoder.clear_buffer(&self.pending, environment as u64 * size, Some(size));
    }
}
