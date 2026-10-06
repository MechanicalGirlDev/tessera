//! GPU-resident rigid-body state and collision-free semi-implicit integration.
//!
//! The state and force buffers stay on the device between steps. Contact and
//! joint kernels can use the same buffers before integration is dispatched.

use core::mem::size_of;
use core::ops::Range;
use core::time::Duration;
use std::sync::mpsc;

use crate::gpu_kinematic_body::PackedKinematicTranslation;
use wgpu::util::DeviceExt;

/// One rigid-body state in GPU storage. Orientation is an XYZW unit quaternion.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuRigidBodyState {
    /// World position in XYZ and inverse mass in W. Zero inverse mass is static or kinematic.
    pub position_inverse_mass: [f32; 4],
    /// World orientation as an XYZW unit quaternion.
    pub orientation: [f32; 4],
    /// World linear velocity in XYZ; W is 1 for a prescribed kinematic body, else 0.
    pub linear_velocity: [f32; 4],
    /// World angular velocity in XYZ.
    pub angular_velocity: [f32; 4],
    /// Body-frame inverse principal inertia in XYZ; W is one when sleeping.
    pub inverse_inertia_sleep: [f32; 4],
}

impl GpuRigidBodyState {
    /// Set prescribed velocities on a zero-mass body. Forces and contact impulses
    /// cannot change its motion. All inputs are validated before mutation.
    pub fn set_kinematic_velocity(
        &mut self,
        linear: [f32; 3],
        angular: [f32; 3],
    ) -> Result<(), GpuRigidStateError> {
        if !self.is_valid()
            || self.position_inverse_mass[3] != 0.0
            || linear
                .iter()
                .chain(&angular)
                .any(|value| !value.is_finite())
        {
            return Err(GpuRigidStateError::InvalidInput);
        }
        self.linear_velocity = [linear[0], linear[1], linear[2], 1.0];
        self.angular_velocity = [angular[0], angular[1], angular[2], 0.0];
        Ok(())
    }

    /// Validate the storage representation before uploading it to the GPU.
    pub fn is_valid(&self) -> bool {
        let finite = self
            .position_inverse_mass
            .iter()
            .chain(self.orientation.iter())
            .chain(self.linear_velocity.iter())
            .chain(self.angular_velocity.iter())
            .chain(self.inverse_inertia_sleep.iter())
            .all(|value| value.is_finite());
        let q_norm = self.orientation.iter().map(|x| x * x).sum::<f32>();
        finite
            && (q_norm - 1.0).abs() <= 1e-3
            && self.position_inverse_mass[3] >= 0.0
            && matches!(self.linear_velocity[3], 0.0 | 1.0)
            && (self.linear_velocity[3] == 0.0 || self.position_inverse_mass[3] == 0.0)
            && self.inverse_inertia_sleep[..3]
                .iter()
                .all(|value| *value >= 0.0)
            && matches!(self.inverse_inertia_sleep[3], 0.0 | 1.0)
            && (self.position_inverse_mass[3] > 0.0
                || (self.inverse_inertia_sleep[..3]
                    .iter()
                    .all(|value| *value == 0.0)
                    && self.inverse_inertia_sleep[3] == 0.0))
    }
}

/// Queued world-space force and torque, consumed by a combined step or frame capture.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuRigidBodyForces {
    /// World-space force in XYZ, in newtons.
    pub force: [f32; 4],
    /// World-space torque in XYZ, in newton metres.
    pub torque: [f32; 4],
}

impl GpuRigidBodyForces {
    fn is_valid(&self) -> bool {
        self.force
            .iter()
            .chain(self.torque.iter())
            .all(|x| x.is_finite())
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct StepParams {
    dt_gravity: [f32; 4],
    count: u32,
    padding: [u32; 3],
}

/// Invalid body state, exceeded GPU capacity, or failed diagnostic readback.
#[derive(Debug, thiserror::Error)]
pub enum GpuRigidStateError {
    /// A body, force, index, or step parameter was invalid.
    #[error("invalid GPU rigid-body input")]
    InvalidInput,
    /// The batch cannot fit in the selected device's storage or dispatch grid.
    #[error("GPU rigid-body batch exceeds device capacity")]
    Capacity,
    /// Mapping a diagnostic state readback failed.
    #[error("GPU rigid-body readback failed: {0}")]
    Readback(String),
}

/// Persistent GPU state for 3D rigid bodies.
///
/// `step` submits work without reading state back. Call `readback` only when
/// synchronizing with a CPU consumer or validating a simulation.
#[derive(Debug)]
pub struct GpuRigidStateSession {
    states: wgpu::Buffer,
    forces: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    capture_forces_pipeline: wgpu::ComputePipeline,
    velocity_pipeline: wgpu::ComputePipeline,
    position_pipeline: wgpu::ComputePipeline,
    clamp_linear_speed_pipeline: wgpu::ComputePipeline,
    motion_pipeline: wgpu::ComputePipeline,
    frame_forces: wgpu::Buffer,
    kinematic_translation: wgpu::Buffer,
    count: usize,
}

impl GpuRigidStateSession {
    /// Upload initial states and allocate reusable storage buffers.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        states: &[GpuRigidBodyState],
    ) -> Result<Self, GpuRigidStateError> {
        if states.iter().any(|state| !state.is_valid()) {
            return Err(GpuRigidStateError::InvalidInput);
        }
        let session = Self::new_uninitialized(device, states.len())?;
        if !states.is_empty() {
            queue.write_buffer(&session.states, 0, bytemuck::cast_slice(states));
            queue.write_buffer(
                &session.forces,
                0,
                bytemuck::cast_slice(&vec![GpuRigidBodyForces::default(); states.len()]),
            );
        }
        Ok(session)
    }

    /// Allocate buffers for a GPU-only topology transfer before filling state.
    pub(crate) fn new_uninitialized(
        device: &wgpu::Device,
        count: usize,
    ) -> Result<Self, GpuRigidStateError> {
        let count_u32 = u32::try_from(count).map_err(|_| GpuRigidStateError::Capacity)?;
        let limits = device.limits();
        let state_size = count as u64 * size_of::<GpuRigidBodyState>() as u64;
        let force_size = count as u64 * size_of::<GpuRigidBodyForces>() as u64;
        if limits.max_storage_buffers_per_shader_stage < 5
            || count_u32.div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || state_size > u64::from(limits.max_storage_buffer_binding_size)
            || force_size > u64::from(limits.max_storage_buffer_binding_size)
            || state_size > limits.max_buffer_size
            || force_size > limits.max_buffer_size
        {
            return Err(GpuRigidStateError::Capacity);
        }
        let states_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera rigid states"),
            size: state_size.max(size_of::<GpuRigidBodyState>() as u64),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let forces = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera rigid forces"),
            size: force_size.max(size_of::<GpuRigidBodyForces>() as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let frame_forces = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera rigid frame forces"),
            size: force_size.max(size_of::<GpuRigidBodyForces>() as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let kinematic_translation =
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Tessera kinematic translation intervals"),
                size: (count as u64 * size_of::<PackedKinematicTranslation>() as u64)
                    .max(size_of::<PackedKinematicTranslation>() as u64),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        let entries: Vec<_> = (0..5)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: binding == 2,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Tessera rigid integration layout"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Tessera rigid integration pipeline layout"),
            bind_group_layouts: &[&layout],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera rigid state integration"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_rigid_state.wgsl").into()),
        });
        let pipeline = |entry_point| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Tessera rigid state pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let motion_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera prescribed rigid motion"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_rigid_motion.wgsl").into()),
        });
        let motion_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tessera prescribed rigid motion"),
            layout: None,
            module: &motion_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Ok(Self {
            states: states_buffer,
            forces,
            frame_forces,
            kinematic_translation,
            pipeline: pipeline("main"),
            capture_forces_pipeline: pipeline("capture_forces"),
            velocity_pipeline: pipeline("velocity"),
            position_pipeline: pipeline("position"),
            clamp_linear_speed_pipeline: pipeline("clamp_linear_speed"),
            motion_pipeline,
            count,
        })
    }

    /// Number of bodies in stable buffer order.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Whether the session contains no bodies.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Storage buffer that collision and constraint kernels can read or update.
    pub fn state_buffer(&self) -> &wgpu::Buffer {
        &self.states
    }

    /// Storage buffer for force-producing kernels. A step clears each entry.
    pub fn force_buffer(&self) -> &wgpu::Buffer {
        &self.forces
    }

    /// Encode a prescribed velocity change without reading or replacing the pose.
    ///
    /// `Some((linear, angular))` enables kinematic motion; `None` stops motion and
    /// restores static behavior. The GPU applies the command only if the body's
    /// current inverse mass is zero. Dynamic bodies are left unchanged, including
    /// their velocities and sleep state. Index and finite values are checked
    /// before encoding. Submit this command before the integration/contact passes
    /// that should observe it. Contact owners must arrange any required wake-up.
    pub fn encode_kinematic_motion(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        index: usize,
        velocities: Option<([f32; 3], [f32; 3])>,
    ) -> Result<(), GpuRigidStateError> {
        let (linear, angular) = velocities.unwrap_or(([0.0; 3], [0.0; 3]));
        if index >= self.count || linear.iter().chain(&angular).any(|v| !v.is_finite()) {
            return Err(GpuRigidStateError::InvalidInput);
        }
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Motion {
            header: [u32; 4],
            linear: [f32; 4],
            angular: [f32; 4],
        }
        let motion = Motion {
            header: [index as u32, 0, 0, 0],
            linear: [
                linear[0],
                linear[1],
                linear[2],
                if velocities.is_some() { 1.0 } else { 0.0 },
            ],
            angular: [angular[0], angular[1], angular[2], 0.0],
        };
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera prescribed motion command"),
            contents: bytemuck::bytes_of(&motion),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera prescribed motion bindings"),
            layout: &self.motion_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.states.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.kinematic_translation.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Tessera prescribed motion update"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.motion_pipeline);
        pass.set_bind_group(0, &bindings, &[]);
        pass.dispatch_workgroups(1, 1, 1);
        Ok(())
    }

    /// Replace one body and clear its queued and captured forces without reading other states.
    pub fn write_body(
        &self,
        queue: &wgpu::Queue,
        index: usize,
        state: GpuRigidBodyState,
    ) -> Result<(), GpuRigidStateError> {
        if index >= self.count || !state.is_valid() {
            return Err(GpuRigidStateError::InvalidInput);
        }
        queue.write_buffer(
            &self.states,
            (index * size_of::<GpuRigidBodyState>()) as u64,
            bytemuck::bytes_of(&state),
        );
        queue.write_buffer(
            &self.forces,
            (index * size_of::<GpuRigidBodyForces>()) as u64,
            bytemuck::bytes_of(&GpuRigidBodyForces::default()),
        );
        queue.write_buffer(
            &self.frame_forces,
            (index * size_of::<GpuRigidBodyForces>()) as u64,
            bytemuck::bytes_of(&GpuRigidBodyForces::default()),
        );
        queue.write_buffer(
            &self.kinematic_translation,
            (index * size_of::<PackedKinematicTranslation>()) as u64,
            bytemuck::bytes_of(&PackedKinematicTranslation::default()),
        );
        Ok(())
    }

    /// Overwrite one body's force and torque for the next step.
    pub fn write_forces(
        &self,
        queue: &wgpu::Queue,
        index: usize,
        value: GpuRigidBodyForces,
    ) -> Result<(), GpuRigidStateError> {
        if index >= self.count || !value.is_valid() {
            return Err(GpuRigidStateError::InvalidInput);
        }
        queue.write_buffer(
            &self.forces,
            (index * size_of::<GpuRigidBodyForces>()) as u64,
            bytemuck::bytes_of(&value),
        );
        Ok(())
    }

    /// Submit one collision-free semi-implicit integration step.
    pub fn step(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        dt: f32,
        gravity: [f32; 3],
    ) -> Result<(), GpuRigidStateError> {
        self.step_with_speed_limit(device, queue, dt, gravity, None)
    }

    /// Submit one integration step with an optional linear speed limit.
    pub fn step_with_speed_limit(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        dt: f32,
        gravity: [f32; 3],
        max_speed: Option<f32>,
    ) -> Result<(), GpuRigidStateError> {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera rigid state encoder"),
        });
        self.encode_step_with_speed_limit(device, &mut encoder, dt, gravity, max_speed)?;
        let _submission = queue.submit(Some(encoder.finish()));
        Ok(())
    }

    /// Encode one step after any contact and constraint passes on the encoder.
    ///
    /// Each dispatch owns its parameters, so several steps can share one
    /// submission while keeping body state and forces entirely on the GPU.
    pub fn encode_step(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        dt: f32,
        gravity: [f32; 3],
    ) -> Result<(), GpuRigidStateError> {
        self.encode_step_with_speed_limit(device, encoder, dt, gravity, None)
    }

    /// Encode one step with an optional world-space linear speed limit.
    pub fn encode_step_with_speed_limit(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        dt: f32,
        gravity: [f32; 3],
        max_speed: Option<f32>,
    ) -> Result<(), GpuRigidStateError> {
        if !dt.is_finite() || dt <= 0.0 || gravity.iter().any(|x| !x.is_finite()) {
            return Err(GpuRigidStateError::InvalidInput);
        }
        if max_speed.is_some_and(|speed| !speed.is_finite() || speed <= 0.0) {
            return Err(GpuRigidStateError::InvalidInput);
        }
        if self.is_empty() {
            return Ok(());
        }
        self.encode_integration(device, encoder, dt, gravity, max_speed, &self.pipeline);
        Ok(())
    }

    /// Snapshot queued forces for all substeps of one temporal frame, then clear the queue.
    ///
    /// Encode once before velocity substeps. The snapshot remains constant until
    /// the next capture; forces written after capture belong to the next frame.
    pub fn encode_capture_forces(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        if !self.is_empty() {
            self.encode_integration(
                device,
                encoder,
                1.0,
                [0.0; 3],
                None,
                &self.capture_forces_pipeline,
            );
        }
    }

    /// Integrate velocities from gravity, gyroscopic torque and the captured frame forces.
    ///
    /// Does not update positions or orientations. Encode force capture once per
    /// frame and use the substep duration so external forces act over the full frame.
    pub fn encode_velocity_step(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        dt: f32,
        gravity: [f32; 3],
    ) -> Result<(), GpuRigidStateError> {
        self.encode_velocity_step_with_speed_limit(device, encoder, dt, gravity, None)
    }

    /// Integrate velocity while limiting the resulting world-space linear speed.
    pub fn encode_velocity_step_with_speed_limit(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        dt: f32,
        gravity: [f32; 3],
        max_speed: Option<f32>,
    ) -> Result<(), GpuRigidStateError> {
        if !dt.is_finite() || dt <= 0.0 || gravity.iter().any(|x| !x.is_finite()) {
            return Err(GpuRigidStateError::InvalidInput);
        }
        if max_speed.is_some_and(|speed| !speed.is_finite() || speed <= 0.0) {
            return Err(GpuRigidStateError::InvalidInput);
        }
        if !self.is_empty() {
            self.encode_integration(
                device,
                encoder,
                dt,
                gravity,
                max_speed,
                &self.velocity_pipeline,
            );
        }
        Ok(())
    }

    /// Integrate positions and orientations from current velocities after a constraint solve.
    pub fn encode_position_step(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        dt: f32,
    ) -> Result<(), GpuRigidStateError> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(GpuRigidStateError::InvalidInput);
        }
        if !self.is_empty() {
            self.encode_integration(device, encoder, dt, [0.0; 3], None, &self.position_pipeline);
        }
        Ok(())
    }

    /// Limit dynamic bodies after contact and joint impulses without reading back state.
    pub fn encode_clamp_linear_speed(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        max_speed: f32,
    ) -> Result<(), GpuRigidStateError> {
        if !max_speed.is_finite() || max_speed <= 0.0 {
            return Err(GpuRigidStateError::InvalidInput);
        }
        if !self.is_empty() {
            self.encode_integration(
                device,
                encoder,
                0.0,
                [0.0; 3],
                Some(max_speed),
                &self.clamp_linear_speed_pipeline,
            );
        }
        Ok(())
    }

    fn encode_integration(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        dt: f32,
        gravity: [f32; 3],
        max_speed: Option<f32>,
        pipeline: &wgpu::ComputePipeline,
    ) {
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera rigid step params"),
            contents: bytemuck::bytes_of(&StepParams {
                dt_gravity: [dt, gravity[0], gravity[1], gravity[2]],
                count: self.count as u32,
                padding: [max_speed.map_or(0, f32::to_bits), 0, 0],
            }),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera rigid state inputs"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.states.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.forces.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.frame_forces.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.kinematic_translation.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Tessera rigid state integration"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((self.count as u32).div_ceil(64), 1, 1);
        }
    }

    /// Read all states for diagnostics or synchronization with a CPU consumer.
    pub fn readback(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<GpuRigidBodyState>, GpuRigidStateError> {
        self.readback_range(device, queue, 0..self.count)
    }

    /// Read only a contiguous body range for one consumer or environment.
    pub fn readback_range(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        range: Range<usize>,
    ) -> Result<Vec<GpuRigidBodyState>, GpuRigidStateError> {
        if range.start > range.end || range.end > self.count {
            return Err(GpuRigidStateError::InvalidInput);
        }
        if range.is_empty() {
            return Ok(Vec::new());
        }
        let size = (range.len() * size_of::<GpuRigidBodyState>()) as u64;
        let offset = (range.start * size_of::<GpuRigidBodyState>()) as u64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera rigid state readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera rigid readback encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.states, offset, &staging, 0, size);
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
            .map_err(|error| GpuRigidStateError::Readback(error.to_string()))?;
        receiver
            .recv_timeout(Duration::from_secs(5))
            .map_err(|error| GpuRigidStateError::Readback(error.to_string()))?
            .map_err(|error| GpuRigidStateError::Readback(error.to_string()))?;
        let view = staging.slice(..).get_mapped_range();
        let result = view
            .chunks_exact(size_of::<GpuRigidBodyState>())
            .map(bytemuck::pod_read_unaligned)
            .collect();
        drop(view);
        staging.unmap();
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_contact_pipeline::GpuContactDevice;

    fn body(position: [f32; 3], mass: f32) -> GpuRigidBodyState {
        GpuRigidBodyState {
            position_inverse_mass: [
                position[0],
                position[1],
                position[2],
                if mass > 0.0 { 1.0 / mass } else { 0.0 },
            ],
            orientation: [0.0, 0.0, 0.0, 1.0],
            linear_velocity: [0.0; 4],
            angular_velocity: [0.0; 4],
            inverse_inertia_sleep: [
                if mass > 0.0 { 0.5 } else { 0.0 },
                if mass > 0.0 { 0.5 } else { 0.0 },
                if mass > 0.0 { 0.5 } else { 0.0 },
                0.0,
            ],
        }
    }

    #[test]
    fn validation_rejects_nonfinite_and_invalid_static_inertia() {
        let mut invalid = body([0.0; 3], 1.0);
        invalid.orientation = [0.0; 4];
        assert!(!invalid.is_valid());
        invalid = body([0.0; 3], 0.0);
        invalid.inverse_inertia_sleep[0] = 1.0;
        assert!(!invalid.is_valid());
        invalid = body([0.0; 3], 1.0);
        invalid.linear_velocity[0] = f32::NAN;
        assert!(!invalid.is_valid());
    }

    #[test]
    fn gpu_kinematic_translation_intervals_preserve_small_steps_and_reset() {
        let velocity = [0.2, -0.3, 0.1];
        let origins = [[1.0, 2.0, -3.0], [10000.0, -10000.0, 1.0]];
        let mut inputs = origins
            .map(|origin| {
                let mut state = body(origin, 0.0);
                state.set_kinematic_velocity(velocity, [0.0; 3]).unwrap();
                state
            })
            .to_vec();
        let stationary = body([100.0, 5.0, 8.0], 0.0);
        inputs.push(stationary);
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            eprintln!("kinematic translation interval backend: {backend:?}");
            let device = context.device();
            let queue = context.queue();
            let session = GpuRigidStateSession::new(device, queue, &inputs).unwrap();
            let mut encoder = device.create_command_encoder(&Default::default());
            for _ in 0..100 {
                session
                    .encode_step(device, &mut encoder, 0.001, [0.0; 3])
                    .unwrap();
            }
            let _ = queue.submit([encoder.finish()]);
            let states = session.readback(device, queue).unwrap();
            for (index, origin) in origins.iter().enumerate() {
                for axis in 0..3 {
                    let expected = (origin[axis] as f64 + velocity[axis] as f64 * 0.1) as f32;
                    if index == 0 {
                        assert!(
                            (states[index].position_inverse_mass[axis] - expected).abs() < 1e-6,
                            "{backend:?}: {:?}",
                            states[index]
                        );
                    } else {
                        assert_eq!(
                            states[index].position_inverse_mass[axis], expected,
                            "large coordinate on {backend:?}"
                        );
                    }
                }
            }
            assert_eq!(
                states[2].position_inverse_mass,
                stationary.position_inverse_mass
            );
            let before = states[0];
            let mut encoder = device.create_command_encoder(&Default::default());
            session
                .encode_kinematic_motion(device, &mut encoder, 0, Some(([0.0, 0.2, 0.0], [0.0; 3])))
                .unwrap();
            for _ in 0..20 {
                session
                    .encode_position_step(device, &mut encoder, 0.005)
                    .unwrap();
            }
            let _ = queue.submit([encoder.finish()]);
            let moved = session.readback(device, queue).unwrap()[0];
            assert_eq!(
                moved.position_inverse_mass[0],
                before.position_inverse_mass[0]
            );
            assert_eq!(
                moved.position_inverse_mass[2],
                before.position_inverse_mass[2]
            );
            assert!(
                (moved.position_inverse_mass[1] - before.position_inverse_mass[1] - 0.02).abs()
                    < 1e-6
            );
            let mut encoder = device.create_command_encoder(&Default::default());
            session
                .encode_kinematic_motion(device, &mut encoder, 0, None)
                .unwrap();
            for _ in 0..10 {
                session
                    .encode_position_step(device, &mut encoder, 0.01)
                    .unwrap();
            }
            let _ = queue.submit([encoder.finish()]);
            let stopped = session.readback(device, queue).unwrap()[0];
            assert_eq!(stopped.position_inverse_mass, moved.position_inverse_mass);
            session.write_body(queue, 0, inputs[0]).unwrap();
            let mut encoder = device.create_command_encoder(&Default::default());
            for _ in 0..10 {
                session
                    .encode_position_step(device, &mut encoder, 0.001)
                    .unwrap();
            }
            let _ = queue.submit([encoder.finish()]);
            let restarted = session.readback(device, queue).unwrap()[0];
            assert!((restarted.position_inverse_mass[0] - 1.002).abs() < 1e-6);
            // A GPU topology transfer or other resident writer must start a fresh interval too.
            let mut replacement = restarted;
            replacement.position_inverse_mass = [5.0, 6.0, 7.0, 0.0];
            replacement.linear_velocity = [0.1, 0.0, 0.0, 1.0];
            queue.write_buffer(session.state_buffer(), 0, bytemuck::bytes_of(&replacement));
            let mut encoder = device.create_command_encoder(&Default::default());
            for _ in 0..10 {
                session
                    .encode_position_step(device, &mut encoder, 0.001)
                    .unwrap();
            }
            let _ = queue.submit([encoder.finish()]);
            let replaced = session.readback(device, queue).unwrap()[0];
            assert!((replaced.position_inverse_mass[0] - 5.001).abs() < 1e-6);
            assert_eq!(replaced.position_inverse_mass[1..], [6.0, 7.0, 0.0]);
        }
        assert!(tested > 0);
    }

    #[test]
    fn gpu_kinematic_motion_ignores_forces_and_preserves_static_behavior() {
        let mut kinematic = body([0.0; 3], 0.0);
        assert!(
            kinematic
                .set_kinematic_velocity([f32::NAN, 0.0, 0.0], [0.0; 3])
                .is_err()
        );
        assert_eq!(kinematic.linear_velocity, [0.0; 4]);
        kinematic
            .set_kinematic_velocity([1.0, 0.0, 0.0], [0.0, 0.0, 1.0])
            .unwrap();
        let mut dynamic = body([0.0; 3], 1.0);
        assert!(dynamic.set_kinematic_velocity([1.0; 3], [0.0; 3]).is_err());
        let mut stationary = body([3.0, 2.0, 1.0], 0.0);
        stationary.linear_velocity[0] = 1.0;
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let device = context.device();
            let queue = context.queue();
            let session =
                GpuRigidStateSession::new(device, queue, &[kinematic, stationary, dynamic])
                    .unwrap();
            session
                .write_forces(
                    queue,
                    0,
                    GpuRigidBodyForces {
                        force: [100.0; 4],
                        torque: [100.0; 4],
                    },
                )
                .unwrap();
            session.step(device, queue, 0.1, [0.0, 0.0, -10.0]).unwrap();
            session.step(device, queue, 0.1, [0.0, 0.0, -10.0]).unwrap();
            let states = session.readback(device, queue).unwrap();
            assert!((states[0].position_inverse_mass[0] - 0.2).abs() < 1e-5);
            assert_eq!(states[0].position_inverse_mass[1..], [0.0; 3]);
            assert_eq!(states[0].linear_velocity, [1.0, 0.0, 0.0, 1.0]);
            assert_eq!(states[0].angular_velocity, [0.0, 0.0, 1.0, 0.0]);
            assert!((states[0].orientation[2] - 0.1f32.sin()).abs() < 1e-5);
            assert_eq!(states[1].position_inverse_mass[..3], [3.0, 2.0, 1.0]);
            let mut encoder = device.create_command_encoder(&Default::default());
            assert!(
                session
                    .encode_kinematic_motion(device, &mut encoder, 3, None)
                    .is_err()
            );
            assert!(
                session
                    .encode_kinematic_motion(
                        device,
                        &mut encoder,
                        0,
                        Some(([f32::NAN; 3], [0.0; 3]))
                    )
                    .is_err()
            );
            session
                .encode_kinematic_motion(device, &mut encoder, 0, Some(([0.0, 2.0, 0.0], [0.0; 3])))
                .unwrap();
            session
                .encode_kinematic_motion(device, &mut encoder, 2, Some(([100.0; 3], [100.0; 3])))
                .unwrap();
            let _submission = queue.submit([encoder.finish()]);
            let updated = session.readback(device, queue).unwrap();
            assert_eq!(
                updated[0].position_inverse_mass,
                states[0].position_inverse_mass
            );
            assert_eq!(updated[0].orientation, states[0].orientation);
            assert_eq!(updated[2].linear_velocity, states[2].linear_velocity);
            assert_eq!(updated[2].angular_velocity, states[2].angular_velocity);
            session.step(device, queue, 0.1, [0.0; 3]).unwrap();
            let moving = session.readback(device, queue).unwrap();
            assert!((moving[0].position_inverse_mass[0] - 0.2).abs() < 1e-5);
            assert!((moving[0].position_inverse_mass[1] - 0.2).abs() < 1e-5);
            let mut encoder = device.create_command_encoder(&Default::default());
            session
                .encode_kinematic_motion(device, &mut encoder, 0, None)
                .unwrap();
            let _submission = queue.submit([encoder.finish()]);
            session.step(device, queue, 0.1, [0.0; 3]).unwrap();
            let stopped = session.readback(device, queue).unwrap();
            assert_eq!(
                stopped[0].position_inverse_mass,
                moving[0].position_inverse_mass
            );
            assert_eq!(stopped[0].orientation, moving[0].orientation);
            assert_eq!(stopped[0].linear_velocity, [0.0; 4]);
            assert_eq!(stopped[0].angular_velocity, [0.0; 4]);
            eprintln!("kinematic state integration passed on {backend:?}");
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn gpu_keeps_state_resident_across_steps_and_consumes_forces() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let device = context.device();
        let queue = context.queue();
        let session = GpuRigidStateSession::new(
            device,
            queue,
            &[body([0.0; 3], 2.0), body([3.0, 2.0, 1.0], 0.0)],
        )
        .unwrap();
        session
            .write_forces(
                queue,
                0,
                GpuRigidBodyForces {
                    force: [10.0, 0.0, 0.0, 0.0],
                    torque: [0.0, 0.0, 2.0, 0.0],
                },
            )
            .unwrap();
        session.step(device, queue, 0.1, [0.0, 0.0, -10.0]).unwrap();
        session.step(device, queue, 0.1, [0.0, 0.0, -10.0]).unwrap();
        let states = session.readback(device, queue).unwrap();
        assert!((states[0].position_inverse_mass[0] - 0.1).abs() < 1e-5);
        assert!((states[0].position_inverse_mass[2] + 0.3).abs() < 1e-5);
        assert!((states[0].linear_velocity[0] - 0.5).abs() < 1e-5);
        assert!((states[0].angular_velocity[2] - 0.1).abs() < 1e-5);
        assert!(states[0].orientation[2] > 0.007);
        assert_eq!(states[1].position_inverse_mass[..3], [3.0, 2.0, 1.0]);
    }

    #[test]
    fn limited_integration_caps_speed_before_position_update() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let device = context.device();
        let queue = context.queue();
        let mut fast = body([0.0; 3], 1.0);
        fast.linear_velocity[0] = 5.0;
        let session = GpuRigidStateSession::new(device, queue, &[fast]).unwrap();
        assert!(
            session
                .step_with_speed_limit(device, queue, 0.1, [0.0; 3], Some(0.0))
                .is_err()
        );
        session
            .step_with_speed_limit(device, queue, 0.1, [0.0; 3], Some(1.0))
            .unwrap();
        let state = session.readback(device, queue).unwrap()[0];
        assert!((state.linear_velocity[0] - 1.0).abs() < 1e-5);
        assert!((state.position_inverse_mass[0] - 0.1).abs() < 1e-5);
    }

    #[test]
    fn gpu_force_wakes_sleeping_body_and_body_can_be_replaced() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let device = context.device();
        let queue = context.queue();
        let mut sleeping = body([0.0, 0.0, 1.0], 1.0);
        sleeping.inverse_inertia_sleep[3] = 1.0;
        let session = GpuRigidStateSession::new(device, queue, &[sleeping]).unwrap();
        session.step(device, queue, 0.1, [0.0, 0.0, -10.0]).unwrap();
        assert_eq!(
            session.readback(device, queue).unwrap()[0].position_inverse_mass[2],
            1.0
        );
        session
            .write_forces(
                queue,
                0,
                GpuRigidBodyForces {
                    force: [0.0, 0.0, 20.0, 0.0],
                    ..GpuRigidBodyForces::default()
                },
            )
            .unwrap();
        session.step(device, queue, 0.1, [0.0, 0.0, -10.0]).unwrap();
        let state = session.readback(device, queue).unwrap()[0];
        assert_eq!(state.inverse_inertia_sleep[3], 0.0);
        assert!((state.position_inverse_mass[2] - 1.1).abs() < 1e-5);
        session
            .write_body(queue, 0, body([1.0, 0.0, 2.0], 1.0))
            .unwrap();
        assert_eq!(
            session.readback(device, queue).unwrap()[0].position_inverse_mass[..3],
            [1.0, 0.0, 2.0]
        );
    }

    #[test]
    fn gpu_encodes_distinct_substeps_in_one_submission() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let device = context.device();
        let queue = context.queue();
        let session = GpuRigidStateSession::new(device, queue, &[body([0.0; 3], 1.0)]).unwrap();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera rigid multi-step test"),
        });
        session
            .encode_step(device, &mut encoder, 0.1, [0.0, 0.0, -10.0])
            .unwrap();
        session
            .encode_step(device, &mut encoder, 0.2, [0.0, 0.0, -10.0])
            .unwrap();
        let _submission = queue.submit(Some(encoder.finish()));
        let state = session.readback(device, queue).unwrap()[0];
        assert!((state.linear_velocity[2] + 3.0).abs() < 1e-5);
        assert!((state.position_inverse_mass[2] + 0.7).abs() < 1e-5);
    }

    #[test]
    fn gpu_asymmetric_body_applies_gyroscopic_acceleration() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut initial = body([0.0; 3], 1.0);
        initial.angular_velocity = [1.0, 2.0, 3.0, 0.0];
        initial.inverse_inertia_sleep = [1.0, 0.5, 0.25, 0.0];
        let session =
            GpuRigidStateSession::new(context.device(), context.queue(), &[initial]).unwrap();
        session
            .step(context.device(), context.queue(), 0.01, [0.0; 3])
            .unwrap();
        let actual = session.readback(context.device(), context.queue()).unwrap()[0];
        assert!((actual.angular_velocity[0] - 0.88).abs() < 1e-5);
        assert!((actual.angular_velocity[1] - 2.045).abs() < 1e-5);
        assert!((actual.angular_velocity[2] - 2.995).abs() < 1e-5);
        let norm = actual.orientation.iter().map(|x| x * x).sum::<f32>();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn empty_gpu_session_steps_without_dispatch_or_readback() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let session = GpuRigidStateSession::new(context.device(), context.queue(), &[]).unwrap();
        assert!(session.is_empty());
        session
            .step(context.device(), context.queue(), 0.01, [0.0, 0.0, -9.81])
            .unwrap();
        assert!(
            session
                .readback(context.device(), context.queue())
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn readback_range_transfers_only_selected_bodies() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let session = GpuRigidStateSession::new(
            context.device(),
            context.queue(),
            &[
                body([1.0, 0.0, 0.0], 1.0),
                body([2.0, 0.0, 0.0], 1.0),
                body([3.0, 0.0, 0.0], 1.0),
            ],
        )
        .unwrap();
        let selected = session
            .readback_range(context.device(), context.queue(), 1..3)
            .unwrap();
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].position_inverse_mass[0], 2.0);
        assert_eq!(selected[1].position_inverse_mass[0], 3.0);
        assert!(
            session
                .readback_range(context.device(), context.queue(), 2..2)
                .unwrap()
                .is_empty()
        );
        assert!(matches!(
            session.readback_range(context.device(), context.queue(), 2..4),
            Err(GpuRigidStateError::InvalidInput)
        ));
    }
}
