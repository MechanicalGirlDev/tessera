//! Stateless MJCF feedback evaluated against resident generalized state.

use core::{mem::size_of, ops::Range};
use wgpu::util::DeviceExt;

use crate::mjcf::{MjcfActuatorInfo, MjcfActuatorKind};

/// One environment's absolute packed layout and imported actuator declarations.
#[derive(Debug, Clone)]
pub struct GpuMjcfActuatorInput<'a> {
    /// Absolute scalar offsets, identical in position, velocity and force buffers.
    pub coordinates: Range<usize>,
    /// Non-joint workspace slots preceding imported scalar joint coordinates.
    pub joint_prefix: usize,
    /// Imported definitions in declaration order; unsupported definitions never drive.
    pub entries: &'a [MjcfActuatorInfo],
    /// Initial held controls, including unsupported declarations.
    pub controls: &'a [f64],
}

/// Invalid metadata, controls, buffer layout, or GPU capacity.
#[derive(Debug, thiserror::Error)]
pub enum GpuMjcfActuatorError {
    /// Metadata or controls cannot be represented in the supplied resident layout.
    #[error("invalid GPU MJCF actuator input")]
    InvalidInput,
    /// Packed metadata or dispatch exceeds GPU limits.
    #[error("GPU MJCF actuators exceed device capacity")]
    Capacity,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PackedActuator {
    header: [u32; 4],
    gains: [f32; 4],
    bias: [f32; 4],
    clamps: [f32; 4],
}

/// Additive MJCF feedback with deterministic declaration-order reduction.
///
/// Buffers must belong to `device`, have STORAGE usage, and hold scalar `f32`
/// arrays using the same absolute coordinate offsets. Bind the ordinary force
/// assembly's base-force buffer, not its mass-solver scratch vector.
/// Encode after ordinary joint-force assembly and before gravity projection /
/// mass response, once per resident microstep. Each encode reads the then-current
/// resident q/dq and adds to forces; it does not clear or replace baseline forces.
/// Caller state/status handling must keep the bound state finite and prevent
/// stepping faulted environments. There is no host feedback or state readback.
///
/// Held-control uploads precede subsequent queue submissions. To use different
/// controls within one command buffer, submit its earlier steps before updating.
#[derive(Debug)]
pub struct GpuMjcfActuatorBatch {
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bindings: wgpu::BindGroup,
    controls: wgpu::Buffer,
    control_ranges: Vec<Range<usize>>,
    workgroups: u32,
}

impl GpuMjcfActuatorBatch {
    /// Resident held controls captured/restored by native owner reset templates.
    pub(crate) fn controls_buffer(&self) -> &wgpu::Buffer {
        &self.controls
    }

    /// Environment control intervals in declaration order.
    pub(crate) fn control_ranges(&self) -> &[Range<usize>] {
        &self.control_ranges
    }

    /// Bind resident buffers and validate imported stateless actuator metadata.
    ///
    /// Environment ranges must be ordered and nonoverlapping; gaps are allowed.
    /// Imported coordinates are relative to `coordinates.start + joint_prefix`.
    ///
    /// # Errors
    /// Rejects invalid coordinates, ranges, nonfinite/unrepresentable parameters,
    /// incorrect control counts, incompatible buffers, or device capacity.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        positions: &wgpu::Buffer,
        velocities: &wgpu::Buffer,
        forces: &wgpu::Buffer,
        inputs: &[GpuMjcfActuatorInput<'_>],
    ) -> Result<Self, GpuMjcfActuatorError> {
        let mut rows = Vec::<[u32; 4]>::new();
        let mut actuators = Vec::<PackedActuator>::new();
        let mut controls = Vec::<f32>::new();
        let mut control_ranges = Vec::with_capacity(inputs.len());
        let mut previous_end = 0;
        for input in inputs {
            if input.coordinates.start < previous_end
                || input.coordinates.end < input.coordinates.start
                || input.joint_prefix > input.coordinates.len()
            {
                return Err(GpuMjcfActuatorError::InvalidInput);
            }
            previous_end = input.coordinates.end;
            let start = controls.len();
            controls.extend(pack_controls(input.controls, input.entries.len())?);
            control_ranges.push(start..controls.len());
            let joint_count = input.coordinates.len() - input.joint_prefix;
            let mut by_coordinate = vec![Vec::new(); joint_count];
            for (ordinal, entry) in input.entries.iter().enumerate() {
                if entry.kind == MjcfActuatorKind::Unsupported {
                    continue;
                }
                let coordinate = entry.coordinate.ok_or(GpuMjcfActuatorError::InvalidInput)?;
                let target = by_coordinate
                    .get_mut(coordinate)
                    .ok_or(GpuMjcfActuatorError::InvalidInput)?;
                let control = start
                    .checked_add(ordinal)
                    .ok_or(GpuMjcfActuatorError::Capacity)?;
                target.push(pack_actuator(entry, index(control)?)?);
            }
            for (coordinate, definitions) in by_coordinate.into_iter().enumerate() {
                if definitions.is_empty() {
                    continue;
                }
                let slot = input
                    .coordinates
                    .start
                    .checked_add(input.joint_prefix)
                    .and_then(|offset| offset.checked_add(coordinate))
                    .ok_or(GpuMjcfActuatorError::Capacity)?;
                rows.push([
                    index(slot)?,
                    index(actuators.len())?,
                    index(definitions.len())?,
                    0,
                ]);
                actuators.extend(definitions);
            }
        }
        let limits = device.limits();
        let scalar_bytes = u64::try_from(previous_end)
            .map_err(|_| GpuMjcfActuatorError::Capacity)?
            .checked_mul(4)
            .ok_or(GpuMjcfActuatorError::Capacity)?;
        for buffer in [positions, velocities, forces] {
            if buffer.size() < scalar_bytes.max(4)
                || !buffer.usage().contains(wgpu::BufferUsages::STORAGE)
            {
                return Err(GpuMjcfActuatorError::InvalidInput);
            }
            if buffer.size() > u64::from(limits.max_storage_buffer_binding_size) {
                return Err(GpuMjcfActuatorError::Capacity);
            }
        }
        let workgroups = index(rows.len().div_ceil(64))?;
        if workgroups > limits.max_compute_workgroups_per_dimension
            || limits.max_storage_buffers_per_shader_stage < 6
        {
            return Err(GpuMjcfActuatorError::Capacity);
        }
        let upload = |label, bytes: &[u8], stride: usize, writable| {
            let length = bytes.len().max(stride);
            let length = u64::try_from(length).map_err(|_| GpuMjcfActuatorError::Capacity)?;
            if length > limits.max_buffer_size
                || length > u64::from(limits.max_storage_buffer_binding_size)
            {
                return Err(GpuMjcfActuatorError::Capacity);
            }
            let empty = vec![0u8; stride];
            Ok(
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: if bytes.is_empty() { &empty } else { bytes },
                    usage: wgpu::BufferUsages::STORAGE
                        | if writable {
                            wgpu::BufferUsages::COPY_DST
                        } else {
                            wgpu::BufferUsages::empty()
                        },
                }),
            )
        };
        let rows = upload(
            "MJCF coordinate reductions",
            bytemuck::cast_slice(&rows),
            16,
            false,
        )?;
        let actuators = upload(
            "MJCF actuator metadata",
            bytemuck::cast_slice(&actuators),
            size_of::<PackedActuator>(),
            false,
        )?;
        let controls = upload(
            "MJCF held controls",
            bytemuck::cast_slice(&controls),
            4,
            true,
        )?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MJCF resident feedback"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_mjcf_actuator.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MJCF resident feedback"),
            layout: None,
            module: &module,
            entry_point: Some("add_efforts"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MJCF resident feedback bindings"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: rows.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: actuators.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: controls.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: positions.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: velocities.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: forces.as_entire_binding(),
                },
            ],
        });
        Ok(Self {
            queue: queue.clone(),
            pipeline,
            bindings,
            controls,
            control_ranges,
            workgroups,
        })
    }

    /// Atomically validate and replace one environment's held declaration-order controls.
    ///
    /// # Errors
    /// Rejects unknown environments, incorrect lengths, or nonfinite/f32-overflowing controls.
    pub fn update_controls(
        &self,
        environment: usize,
        controls: &[f64],
    ) -> Result<(), GpuMjcfActuatorError> {
        let range = self
            .control_ranges
            .get(environment)
            .ok_or(GpuMjcfActuatorError::InvalidInput)?;
        let packed = pack_controls(controls, range.len())?;
        if !packed.is_empty() {
            let offset = u64::try_from(range.start)
                .map_err(|_| GpuMjcfActuatorError::Capacity)?
                .checked_mul(4)
                .ok_or(GpuMjcfActuatorError::Capacity)?;
            self.queue
                .write_buffer(&self.controls, offset, bytemuck::cast_slice(&packed));
        }
        Ok(())
    }

    /// Add feedback at the current resident state after joint forces, before mass response.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        if self.workgroups == 0 {
            return;
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("MJCF additive feedback"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bindings, &[]);
        pass.dispatch_workgroups(self.workgroups, 1, 1);
    }
}

fn index(value: usize) -> Result<u32, GpuMjcfActuatorError> {
    u32::try_from(value).map_err(|_| GpuMjcfActuatorError::Capacity)
}

fn scalar(value: f64) -> Result<f32, GpuMjcfActuatorError> {
    // GPU state and all native force kernels use f32; reject overflow at upload.
    let packed = value as f32;
    if !value.is_finite() || !packed.is_finite() {
        return Err(GpuMjcfActuatorError::InvalidInput);
    }
    Ok(packed)
}

fn pack_controls(values: &[f64], count: usize) -> Result<Vec<f32>, GpuMjcfActuatorError> {
    if values.len() != count {
        return Err(GpuMjcfActuatorError::InvalidInput);
    }
    values.iter().copied().map(scalar).collect()
}

fn pack_actuator(
    entry: &MjcfActuatorInfo,
    control: u32,
) -> Result<PackedActuator, GpuMjcfActuatorError> {
    let (kind, gains, bias) = match entry.kind {
        MjcfActuatorKind::Unsupported => return Err(GpuMjcfActuatorError::InvalidInput),
        MjcfActuatorKind::Motor => (1, [scalar(entry.gear)?, 0.0, 0.0, 0.0], [0.0; 4]),
        MjcfActuatorKind::Position { kp, kv } => {
            (2, [scalar(kp)?, scalar(kv)?, 0.0, 0.0], [0.0; 4])
        }
        MjcfActuatorKind::Velocity { kv } => (3, [scalar(kv)?, 0.0, 0.0, 0.0], [0.0; 4]),
        MjcfActuatorKind::Damper { gain } => (4, [scalar(gain)?, 0.0, 0.0, 0.0], [0.0; 4]),
        MjcfActuatorKind::General { gain, bias, affine } => {
            if affine {
                (
                    5,
                    [scalar(gain)?, 0.0, 0.0, 0.0],
                    [scalar(bias[0])?, scalar(bias[1])?, scalar(bias[2])?, 0.0],
                )
            } else {
                (6, [scalar(gain)?, scalar(entry.gear)?, 0.0, 0.0], [0.0; 4])
            }
        }
    };
    let range = |value: Option<[f64; 2]>| -> Result<[f32; 2], GpuMjcfActuatorError> {
        if let Some([lower, upper]) = value {
            if lower > upper {
                return Err(GpuMjcfActuatorError::InvalidInput);
            }
            Ok([scalar(lower)?, scalar(upper)?])
        } else {
            Ok([0.0; 2])
        }
    };
    let control_range = range(entry.control_range)?;
    let force_range = range(entry.force_range)?;
    Ok(PackedActuator {
        header: [
            kind,
            control,
            u32::from(entry.control_range.is_some()),
            u32::from(entry.force_range.is_some()),
        ],
        gains,
        bias,
        clamps: [
            control_range[0],
            control_range[1],
            force_range[0],
            force_range[1],
        ],
    })
}
