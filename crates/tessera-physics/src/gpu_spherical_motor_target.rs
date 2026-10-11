//! Resident actions for native quaternion spherical drives.
//!
//! Position values are radians in components of the shortest rotation vector
//! `Log(orientation_target)` in the parent-side joint frame, not Euler angles.
//! Each application replaces mapped components, retains the other components
//! of the current target logarithm, then stores the unit quaternion `Exp(vector)`.
//! Quaternions are canonicalized to positive w; at w=0 the first nonzero xyz
//! component is positive. Values outside the principal ball wrap through Exp;
//! subsequent partial updates use the resulting shortest Log, not unwrapped
//! action history. Velocity values are parent-frame radians per second.
//! The existing Lie quaternion-error physical drive is unchanged; this does
//! not assert numerical solver identity with Nexus.

use crate::gpu_motor_target::GpuMotorTargetMode;
use crate::spherical_drive::SphericalJointDrive;
use wgpu::util::DeviceExt;

/// One action row's destination within an environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuSphericalMotorTargetMapping {
    /// Native spherical joint index in constructor environment/joint order.
    pub joint: usize,
    /// Parent-side joint frame axis: 0=x, 1=y, 2=z.
    pub axis: usize,
    /// Rotation-vector position or angular-velocity component.
    pub mode: GpuMotorTargetMode,
}

/// Invalid action layout or insufficient GPU capacity.
#[derive(Debug, thiserror::Error)]
pub enum GpuSphericalMotorTargetError {
    /// Empty, duplicate, disabled, out-of-range, or mismatched mapping.
    #[error("invalid GPU spherical motor target mapping")]
    InvalidInput,
    /// Packed layout exceeds a device or integer limit.
    #[error("GPU spherical motor targets exceed device capacity")]
    Capacity,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PackedMapping {
    drive: u32,
    action: u32,
    component: u32,
    delay: u32,
}
const _: () = assert!(size_of::<PackedMapping>() == 16);

/// Owner hooks for independent resident spherical action control.
///
/// Latch STORAGE f32 actions in `[action row, environment]` order after the GPU
/// producer. Apply once before spherical drive assembly in every microstep.
/// Delay d skips d apply calls; zero applies on the next call. A new latch
/// replaces pending values and restarts the configured delay. Updating delays
/// also restarts queued delays. No action readback or host target upload occurs.
#[derive(Debug)]
pub(crate) struct GpuSphericalMotorTargetControl {
    latch: wgpu::ComputePipeline,
    apply: wgpu::ComputePipeline,
    delay_update: wgpu::ComputePipeline,
    mappings: wgpu::Buffer,
    pending: wgpu::Buffer,
    parameters: wgpu::Buffer,
    destinations: wgpu::Buffer,
    control_layout: wgpu::Buffer,
    count: u32,
    drive_count: u32,
    environment_count: usize,
    joint_counts: Vec<usize>,
    mapped_joints: Vec<(usize, usize)>,
}

impl GpuSphericalMotorTargetControl {
    /// Parameters are the 96-byte PackedDrive array owned by spherical drive.
    pub(crate) fn new(
        device: &wgpu::Device,
        parameters: &wgpu::Buffer,
        drives: &[Vec<Option<SphericalJointDrive>>],
        mappings: &[Vec<GpuSphericalMotorTargetMapping>],
        delays: &[u32],
    ) -> Result<Self, GpuSphericalMotorTargetError> {
        use GpuSphericalMotorTargetError::{Capacity, InvalidInput};
        let environments = drives.len();
        let rows = mappings.first().map_or(0, Vec::len);
        if environments == 0
            || rows == 0
            || mappings.len() != environments
            || delays.len() != environments
            || mappings.iter().any(|m| m.len() != rows)
        {
            return Err(InvalidInput);
        }
        let count = rows.checked_mul(environments).ok_or(Capacity)?;
        let drive_count = drives
            .iter()
            .try_fold(0usize, |n, d| n.checked_add(d.len()))
            .ok_or(Capacity)?;
        let bytes = u64::try_from(count)
            .map_err(|_| Capacity)?
            .checked_mul(16)
            .ok_or(Capacity)?;
        let destination_bytes = u64::try_from(drive_count)
            .map_err(|_| Capacity)?
            .checked_mul(32)
            .ok_or(Capacity)?;
        let limits = device.limits();
        if drive_count == 0
            || parameters.size()
                != u64::try_from(drive_count)
                    .map_err(|_| Capacity)?
                    .checked_mul(96)
                    .ok_or(Capacity)?
            || !parameters.usage().contains(wgpu::BufferUsages::STORAGE)
        {
            return Err(InvalidInput);
        }
        if bytes.max(destination_bytes) > limits.max_buffer_size
            || bytes.max(destination_bytes) > u64::from(limits.max_storage_buffer_binding_size)
            || count.max(drive_count).div_ceil(64)
                > usize::try_from(limits.max_compute_workgroups_per_dimension)
                    .map_err(|_| Capacity)?
        {
            return Err(Capacity);
        }
        let mut packed = Vec::with_capacity(count);
        let mut destinations = vec![[u32::MAX; 8]; drive_count];
        let mut mapped_joints = Vec::with_capacity(count);
        let mut offset = 0usize;
        for (environment, ((joints, entries), &delay)) in
            drives.iter().zip(mappings).zip(delays).enumerate()
        {
            for (row, mapping) in entries.iter().enumerate() {
                if mapping.axis >= 3 || !joints.get(mapping.joint).is_some_and(Option::is_some) {
                    return Err(InvalidInput);
                }
                let component = mapping.axis
                    + match mapping.mode {
                        GpuMotorTargetMode::Position => 0,
                        GpuMotorTargetMode::Velocity => 3,
                    };
                let drive = offset.checked_add(mapping.joint).ok_or(Capacity)?;
                let slot = destinations
                    .get_mut(drive)
                    .and_then(|d| d.get_mut(component))
                    .ok_or(InvalidInput)?;
                if *slot != u32::MAX {
                    return Err(InvalidInput);
                }
                *slot = u32::try_from(packed.len()).map_err(|_| Capacity)?;
                packed.push(PackedMapping {
                    drive: u32::try_from(drive).map_err(|_| Capacity)?,
                    action: u32::try_from(
                        row.checked_mul(environments)
                            .and_then(|n| n.checked_add(environment))
                            .ok_or(Capacity)?,
                    )
                    .map_err(|_| Capacity)?,
                    component: u32::try_from(component).map_err(|_| Capacity)?,
                    delay,
                });
                mapped_joints.push((environment, mapping.joint));
            }
            offset = offset.checked_add(joints.len()).ok_or(Capacity)?;
        }
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera spherical motor targets"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("gpu_spherical_motor_target.wgsl").into(),
            ),
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
        let buffer = |label, contents: &[u8], usage| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage,
            })
        };
        Ok(Self {
            latch: pipeline("latch_targets"),
            apply: pipeline("apply_targets"),
            delay_update: pipeline("update_delays"),
            mappings: buffer(
                "Spherical target mappings",
                bytemuck::cast_slice(&packed),
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            ),
            pending: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Pending spherical targets"),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            parameters: parameters.clone(),
            destinations: buffer(
                "Spherical target destinations",
                bytemuck::cast_slice(&destinations),
                wgpu::BufferUsages::STORAGE,
            ),
            control_layout: buffer(
                "Spherical target environments",
                bytemuck::cast_slice(&[
                    u32::try_from(environments).map_err(|_| Capacity)?,
                    0,
                    0,
                    0,
                ]),
                wgpu::BufferUsages::UNIFORM,
            ),
            count: u32::try_from(count).map_err(|_| Capacity)?,
            drive_count: u32::try_from(drive_count).map_err(|_| Capacity)?,
            environment_count: environments,
            joint_counts: drives.iter().map(Vec::len).collect(),
            mapped_joints,
        })
    }

    pub(crate) fn action_bytes(&self) -> u64 {
        u64::from(self.count) * 4
    }

    /// Validate a host drive replacement before it disables mapped joints.
    pub(crate) fn accepts_inputs(&self, drives: &[Vec<Option<SphericalJointDrive>>]) -> bool {
        drives.len() == self.environment_count
            && drives
                .iter()
                .zip(&self.joint_counts)
                .all(|(d, &n)| d.len() == n)
            && self.mapped_joints.iter().all(|&(e, j)| {
                drives
                    .get(e)
                    .and_then(|d| d.get(j))
                    .is_some_and(Option::is_some)
            })
    }

    fn encode_pass(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        buffers: &[(u32, &wgpu::Buffer)],
        count: u32,
    ) {
        let entries = buffers
            .iter()
            .map(|&(binding, buffer)| wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>();
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Spherical target bindings"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
    }

    pub(crate) fn encode_latch(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        actions: &wgpu::Buffer,
    ) {
        self.encode_pass(
            device,
            encoder,
            &self.latch,
            &[(0, &self.mappings), (1, &self.pending), (2, actions)],
            self.count,
        );
    }
    pub(crate) fn encode_delays(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        delays: &wgpu::Buffer,
    ) {
        self.encode_pass(
            device,
            encoder,
            &self.delay_update,
            &[
                (0, &self.mappings),
                (1, &self.pending),
                (4, delays),
                (5, &self.control_layout),
            ],
            self.count,
        );
    }
    pub(crate) fn encode_apply(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        self.encode_pass(
            device,
            encoder,
            &self.apply,
            &[
                (1, &self.pending),
                (3, &self.parameters),
                (6, &self.destinations),
            ],
            self.drive_count,
        );
    }
    pub(crate) fn cancel(&self, queue: &wgpu::Queue) -> Result<(), GpuSphericalMotorTargetError> {
        let count =
            usize::try_from(self.count).map_err(|_| GpuSphericalMotorTargetError::Capacity)?;
        queue.write_buffer(
            &self.pending,
            0,
            bytemuck::cast_slice(&vec![[0u32; 4]; count]),
        );
        Ok(())
    }
    /// Clear only selected environment pending actions before restoring its drive snapshot.
    pub(crate) fn encode_cancel_environment(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        environment: usize,
    ) -> Result<(), GpuSphericalMotorTargetError> {
        if environment >= self.environment_count {
            return Err(GpuSphericalMotorTargetError::InvalidInput);
        }
        let stride = self.pending.size()
            / u64::try_from(self.environment_count)
                .map_err(|_| GpuSphericalMotorTargetError::Capacity)?;
        encoder.clear_buffer(
            &self.pending,
            u64::try_from(environment).map_err(|_| GpuSphericalMotorTargetError::Capacity)?
                * stride,
            Some(stride),
        );
        Ok(())
    }
}

#[cfg(test)]
#[path = "gpu_spherical_motor_target_tests.rs"]
mod tests;
