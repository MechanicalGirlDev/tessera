//! Device-resident rigid pose synchronization and MPM reaction transfer.

use core::mem::size_of;

use nalgebra::{UnitQuaternion, Vector3};
use tessera_physics::gpu_rigid_shape::GpuRigidShape;
use tessera_physics::gpu_rigid_sphere_world::{GpuRigidSphereWorld, GpuRigidSphereWorldError};
use wgpu::util::DeviceExt;

use crate::obstacle::ObstacleShape;
use crate::rigid_sync::ResidentRigidSyncError;
use crate::{GpuMpmError, GpuMpmResidentSession};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Mapping {
    body: [u32; 4],
    local_center: [f32; 4],
    local_orientation: [f32; 4],
}

impl Mapping {
    fn new(body: u32, center: [f32; 3], orientation: [f32; 4]) -> Self {
        Self {
            body: [body, 0, 0, 0],
            local_center: [center[0], center[1], center[2], 0.0],
            local_orientation: orientation,
        }
    }
}

/// Device mismatch, edited topology, or an invalid GPU coupling setup.
#[derive(Debug, thiserror::Error)]
pub enum GpuRigidMpmCouplerError {
    /// Rigid and MPM buffers belong to different devices.
    #[error("GPU rigid and MPM sessions must use the same wgpu device")]
    DeviceMismatch,
    /// Shape, material, obstacle layout, or body count changed after construction.
    #[error("GPU rigid to MPM coupling topology changed; recreate the coupler")]
    TopologyChanged,
    /// Mapping or dispatch exceeds the device limits.
    #[error("GPU rigid to MPM coupling capacity exceeded")]
    Capacity,
    /// Initial one-way obstacle conversion or upload failed.
    #[error(transparent)]
    Sync(#[from] ResidentRigidSyncError),
    /// The MPM resident step failed.
    #[error(transparent)]
    Mpm(#[from] GpuMpmError),
    /// The rigid contact, joint, or integration step failed.
    #[error(transparent)]
    Rigid(#[from] GpuRigidSphereWorldError),
}

/// Reuses static MPM geometry while updating obstacle poses and velocities on GPU.
///
/// Construction performs one rigid-state readback to install the obstacle set.
/// Subsequent `encode` calls read live rigid state directly on the same device.
/// Recreate after shape, body count, material, or ground changes. The MPM CPU
/// obstacle snapshot keeps the initial pose until an explicit CPU synchronization.
#[derive(Debug)]
pub struct GpuRigidMpmCoupler {
    device: wgpu::Device,
    pipeline: wgpu::ComputePipeline,
    apply_reactions: wgpu::ComputePipeline,
    mappings: wgpu::Buffer,
    count: u32,
    shapes: Vec<GpuRigidShape>,
    obstacle_shapes: Vec<ObstacleShape>,
    obstacle_frictions: Vec<f64>,
    frictions: Vec<f64>,
    ground: Option<f32>,
    ground_friction: f64,
}

impl GpuRigidMpmCoupler {
    /// Submit one live obstacle update without mapping rigid state to the CPU.
    pub fn update(
        &self,
        rigid: &GpuRigidSphereWorld,
        mpm: &GpuMpmResidentSession<'_>,
    ) -> Result<(), GpuRigidMpmCouplerError> {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.encode(&mut encoder, rigid, mpm)?;
        let _ = mpm.queue().submit(Some(encoder.finish()));
        Ok(())
    }

    /// Update obstacles from live rigid state, then run fixed MPM substeps.
    /// Step the rigid world first; this call does not advance rigid bodies.
    pub fn step(
        &self,
        rigid: &GpuRigidSphereWorld,
        mpm: &mut GpuMpmResidentSession<'_>,
        steps: u32,
    ) -> Result<(), GpuRigidMpmCouplerError> {
        if steps == 0 || steps > 64 {
            return Err(GpuMpmError::InvalidInput.into());
        }
        if mpm.has_pending_steps() {
            return Err(GpuMpmError::PendingSteps.into());
        }
        self.update(rigid, mpm)?;
        mpm.step(steps)?;
        Ok(())
    }

    /// Queue a live obstacle update and MPM substeps without particle readback.
    /// Consecutive calls may interleave rigid steps on the same queue. Call
    /// `mpm.synchronize()` before inspecting the resulting CPU particle state.
    pub fn submit_steps(
        &self,
        rigid: &GpuRigidSphereWorld,
        mpm: &mut GpuMpmResidentSession<'_>,
        steps: u32,
    ) -> Result<(), GpuRigidMpmCouplerError> {
        if steps == 0 || steps > 64 {
            return Err(GpuMpmError::InvalidInput.into());
        }
        self.update(rigid, mpm)?;
        mpm.submit_steps(steps)?;
        Ok(())
    }

    /// Queue one MPM substep and apply its reaction to rigid GPU velocities.
    ///
    /// The rigid state is not mapped to the CPU. Consecutive calls can stay
    /// queued; synchronize the MPM session before relying on its CPU snapshot.
    /// The rigid world must be stepped separately to integrate its new velocity.
    pub fn submit_two_way_step(
        &self,
        rigid: &GpuRigidSphereWorld,
        mpm: &mut GpuMpmResidentSession<'_>,
    ) -> Result<(), GpuRigidMpmCouplerError> {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.encode(&mut encoder, rigid, mpm)?;
        let next_pending = mpm.encode_one_substep_with_reactions(&mut encoder)?;
        self.encode_apply_reactions(&mut encoder, rigid, mpm, false)?;
        let _ = mpm.queue().submit(Some(encoder.finish()));
        mpm.mark_one_substep_submitted(next_pending);
        Ok(())
    }

    /// Queue one coupled substep with MPM reactions and the rigid world's
    /// normal integration, contacts, joints, queued forces, and sleep policy.
    ///
    /// The MPM reaction reaches rigid velocities before the rigid step. This
    /// leaves the MPM CPU snapshot pending; call `mpm.synchronize()` before
    /// reading particles. If the rigid step fails after MPM submission, the
    /// MPM session still has a pending substep and can be synchronized.
    pub fn submit_coupled_step(
        &self,
        rigid: &mut GpuRigidSphereWorld,
        mpm: &mut GpuMpmResidentSession<'_>,
    ) -> Result<usize, GpuRigidMpmCouplerError> {
        if rigid.is_faulted() {
            return Err(GpuRigidSphereWorldError::Faulted.into());
        }
        let dt = mpm.substep_dt() as f32;
        if !dt.is_finite() || dt <= 0.0 {
            return Err(GpuMpmError::InvalidInput.into());
        }
        self.submit_two_way_step(rigid, mpm)?;
        Ok(rigid.step(dt)?)
    }

    /// Advance both worlds by one substep and synchronize MPM particles.
    pub fn step_coupled(
        &self,
        rigid: &mut GpuRigidSphereWorld,
        mpm: &mut GpuMpmResidentSession<'_>,
    ) -> Result<usize, GpuRigidMpmCouplerError> {
        if mpm.has_pending_steps() {
            return Err(GpuMpmError::PendingSteps.into());
        }
        let candidates = self.submit_coupled_step(rigid, mpm)?;
        mpm.synchronize()?;
        Ok(candidates)
    }

    /// Queue one MPM-owned rigid step on the same GPU encoder.
    ///
    /// MPM impulses and rigid gravity update velocity before integrating pose.
    /// Call this instead of a rigid-world step for coupled bodies. Rigid contact
    /// constraints, joints, queued rigid forces, and sleep policy are not solved
    /// in this path.
    pub fn submit_mpm_owned_step(
        &self,
        rigid: &GpuRigidSphereWorld,
        mpm: &mut GpuMpmResidentSession<'_>,
    ) -> Result<(), GpuRigidMpmCouplerError> {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.encode(&mut encoder, rigid, mpm)?;
        let next_pending = mpm.encode_one_substep_with_reactions(&mut encoder)?;
        self.encode_apply_reactions(&mut encoder, rigid, mpm, true)?;
        let _ = mpm.queue().submit(Some(encoder.finish()));
        mpm.mark_one_substep_submitted(next_pending);
        Ok(())
    }

    /// Advance one MPM-owned rigid step and synchronize particle state.
    pub fn step_mpm_owned(
        &self,
        rigid: &GpuRigidSphereWorld,
        mpm: &mut GpuMpmResidentSession<'_>,
    ) -> Result<(), GpuRigidMpmCouplerError> {
        if mpm.has_pending_steps() {
            return Err(GpuMpmError::PendingSteps.into());
        }
        self.submit_mpm_owned_step(rigid, mpm)?;
        mpm.synchronize()?;
        Ok(())
    }

    /// Submit one device-side two-way step and synchronize particle state.
    pub fn step_two_way(
        &self,
        rigid: &GpuRigidSphereWorld,
        mpm: &mut GpuMpmResidentSession<'_>,
    ) -> Result<(), GpuRigidMpmCouplerError> {
        if mpm.has_pending_steps() {
            return Err(GpuMpmError::PendingSteps.into());
        }
        self.submit_two_way_step(rigid, mpm)?;
        mpm.synchronize()?;
        Ok(())
    }

    /// Install the current rigid scene as MPM obstacles and prepare a live update pass.
    pub fn new(
        rigid: &GpuRigidSphereWorld,
        mpm: &mut GpuMpmResidentSession<'_>,
    ) -> Result<Self, GpuRigidMpmCouplerError> {
        if rigid.device() != mpm.device() {
            return Err(GpuRigidMpmCouplerError::DeviceMismatch);
        }
        let device = rigid.device();
        let (mut descriptors, shapes) = build_mappings(rigid)?;
        let count =
            u32::try_from(descriptors.len()).map_err(|_| GpuRigidMpmCouplerError::Capacity)?;
        let limits = device.limits();
        let bytes = u64::from(count.max(1)) * size_of::<Mapping>() as u64;
        if bytes > limits.max_buffer_size
            || bytes > u64::from(limits.max_storage_buffer_binding_size)
            || count.div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || limits.max_storage_buffers_per_shader_stage < 3
        {
            return Err(GpuRigidMpmCouplerError::Capacity);
        }
        if descriptors.is_empty() {
            descriptors.push(Mapping::new(u32::MAX, [0.0; 3], [0.0, 0.0, 0.0, 1.0]));
        }
        let mappings = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera rigid MPM obstacle mappings"),
            contents: bytemuck::cast_slice(&descriptors),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera rigid MPM obstacle sync"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_rigid_sync.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Tessera rigid MPM sync layout"),
            entries: &(0..3)
                .map(|binding| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: binding != 2,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect::<Vec<_>>(),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Tessera rigid MPM sync pipeline layout"),
            bind_group_layouts: &[&layout],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tessera rigid MPM sync pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("update"),
            compilation_options: Default::default(),
            cache: None,
        });
        let reaction_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera MPM rigid reaction apply"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_rigid_reaction.wgsl").into()),
        });
        let apply_reactions = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tessera MPM rigid reaction apply"),
            layout: None,
            module: &reaction_module,
            entry_point: Some("apply"),
            compilation_options: Default::default(),
            cache: None,
        });
        mpm.sync_gpu_rigid_world(rigid)?;
        if mpm.world().obstacles.len() != count as usize {
            return Err(GpuRigidMpmCouplerError::TopologyChanged);
        }
        let obstacle_shapes = mpm
            .world()
            .obstacles
            .iter()
            .map(|obstacle| obstacle.shape.clone())
            .collect();
        let obstacle_frictions = mpm
            .world()
            .obstacles
            .iter()
            .map(|obstacle| obstacle.friction)
            .collect();
        let frictions = rigid_frictions(rigid)?;
        let ground = rigid.config().ground_half_extent;
        let ground_friction = rigid
            .ground_material_override()
            .map_or(f64::from(rigid.config().solve.friction), |material| {
                material.friction
            });
        Ok(Self {
            device: device.clone(),
            pipeline,
            apply_reactions,
            mappings,
            count,
            shapes,
            obstacle_shapes,
            obstacle_frictions,
            frictions,
            ground,
            ground_friction,
        })
    }

    /// Append a live rigid-state to MPM-obstacle update to the given encoder.
    /// Submit this pass before the next MPM `step` or `submit_steps` call.
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        rigid: &GpuRigidSphereWorld,
        mpm: &GpuMpmResidentSession<'_>,
    ) -> Result<(), GpuRigidMpmCouplerError> {
        if rigid.device() != &self.device || mpm.device() != &self.device {
            return Err(GpuRigidMpmCouplerError::DeviceMismatch);
        }
        if rigid.len() != self.shapes.len()
            || rigid.config().ground_half_extent != self.ground
            || rigid
                .ground_material_override()
                .map_or(f64::from(rigid.config().solve.friction), |material| {
                    material.friction
                })
                != self.ground_friction
            || rigid_frictions(rigid)? != self.frictions
            || mpm.world().obstacles.len() != self.count as usize
            || mpm
                .world()
                .obstacles
                .iter()
                .map(|obstacle| &obstacle.shape)
                .ne(self.obstacle_shapes.iter())
            || mpm
                .world()
                .obstacles
                .iter()
                .map(|obstacle| obstacle.friction)
                .ne(self.obstacle_frictions.iter().copied())
            || self
                .shapes
                .iter()
                .enumerate()
                .any(|(index, shape)| rigid.shape(index).as_ref() != Some(shape))
        {
            return Err(GpuRigidMpmCouplerError::TopologyChanged);
        }
        if self.count == 0 {
            return Ok(());
        }
        let Some(obstacles) = mpm.obstacle_buffer() else {
            return Ok(());
        };
        if rigid.state_buffer().size() < rigid.len() as u64 * 80
            || obstacles.size() < u64::from(self.count) * 144
        {
            return Err(GpuRigidMpmCouplerError::Capacity);
        }
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera rigid MPM sync bindings"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[rigid.state_buffer(), &self.mappings, obstacles]
                .iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Tessera rigid MPM obstacle update"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(self.count.div_ceil(64), 1, 1);
        Ok(())
    }

    fn encode_apply_reactions(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        rigid: &GpuRigidSphereWorld,
        mpm: &GpuMpmResidentSession<'_>,
        integrate_pose: bool,
    ) -> Result<(), GpuRigidMpmCouplerError> {
        let Some((transfers, particles, obstacles)) = mpm.reaction_output() else {
            return Ok(());
        };
        if obstacles != self.count {
            return Err(GpuRigidMpmCouplerError::TopologyChanged);
        }
        if self.count == 0 || rigid.is_empty() {
            return Ok(());
        }
        let required = u64::from(particles)
            .checked_add(u64::from(obstacles))
            .and_then(|count| count.checked_mul(144))
            .ok_or(GpuRigidMpmCouplerError::Capacity)?;
        let Some(obstacle_buffer) = mpm.obstacle_buffer() else {
            return Err(GpuRigidMpmCouplerError::Capacity);
        };
        if transfers.size() < required
            || obstacle_buffer.size() < u64::from(obstacles) * 144
            || rigid.state_buffer().size() < rigid.len() as u64 * 80
            || u32::try_from(rigid.len())
                .map_err(|_| GpuRigidMpmCouplerError::Capacity)?
                .div_ceil(64)
                > self.device.limits().max_compute_workgroups_per_dimension
        {
            return Err(GpuRigidMpmCouplerError::Capacity);
        }
        let dt = mpm.substep_dt() as f32;
        if !dt.is_finite() || dt <= 0.0 {
            return Err(GpuMpmError::InvalidInput.into());
        }
        let gravity = rigid.config().gravity;
        let max_speed = rigid.max_linear_speed().unwrap_or(0.0);
        let params_words = [
            particles,
            obstacles,
            u32::from(integrate_pose),
            dt.to_bits(),
            gravity[0].to_bits(),
            gravity[1].to_bits(),
            gravity[2].to_bits(),
            max_speed.to_bits(),
        ];
        let params = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tessera MPM rigid reaction counts"),
                contents: bytemuck::cast_slice(&params_words),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bindings = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera MPM rigid reaction bindings"),
            layout: &self.apply_reactions.get_bind_group_layout(0),
            entries: &[
                transfers,
                &self.mappings,
                obstacle_buffer,
                rigid.state_buffer(),
                &params,
            ]
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>(),
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Tessera MPM rigid reaction apply"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.apply_reactions);
        pass.set_bind_group(0, &bindings, &[]);
        pass.dispatch_workgroups((rigid.len() as u32).div_ceil(64), 1, 1);
        Ok(())
    }
}

fn rigid_frictions(rigid: &GpuRigidSphereWorld) -> Result<Vec<f64>, GpuRigidMpmCouplerError> {
    (0..rigid.len())
        .map(|index| {
            rigid
                .body_material_override(index)
                .map(|material| {
                    material.map_or(f64::from(rigid.config().solve.friction), |m| m.friction)
                })
                .ok_or(GpuRigidMpmCouplerError::TopologyChanged)
        })
        .collect()
}

fn build_mappings(
    rigid: &GpuRigidSphereWorld,
) -> Result<(Vec<Mapping>, Vec<GpuRigidShape>), GpuRigidMpmCouplerError> {
    let mut mappings = Vec::new();
    let mut shapes = Vec::new();
    for index in 0..rigid.len() {
        let body = u32::try_from(index).map_err(|_| GpuRigidMpmCouplerError::Capacity)?;
        let shape = rigid
            .shape(index)
            .ok_or(GpuRigidMpmCouplerError::TopologyChanged)?;
        let identity = [0.0, 0.0, 0.0, 1.0];
        match &shape {
            GpuRigidShape::Polyline { vertices, segments } => {
                mappings
                    .try_reserve(segments.len())
                    .map_err(|_| GpuRigidMpmCouplerError::Capacity)?;
                for [a, b] in segments {
                    let a = vertices
                        .get(*a as usize)
                        .ok_or(GpuRigidMpmCouplerError::TopologyChanged)?;
                    let b = vertices
                        .get(*b as usize)
                        .ok_or(GpuRigidMpmCouplerError::TopologyChanged)?;
                    let a = Vector3::from(*a);
                    let b = Vector3::from(*b);
                    let delta = b - a;
                    let length = delta.norm();
                    if !length.is_finite() || length <= 0.0 {
                        return Err(GpuRigidMpmCouplerError::TopologyChanged);
                    }
                    let rotation =
                        UnitQuaternion::rotation_between(&Vector3::z(), &(delta / length))
                            .unwrap_or_else(|| {
                                UnitQuaternion::from_axis_angle(
                                    &Vector3::x_axis(),
                                    core::f32::consts::PI,
                                )
                            });
                    let q = rotation.quaternion().coords;
                    let center = (a + b) / 2.0;
                    mappings.push(Mapping::new(
                        body,
                        [center.x, center.y, center.z],
                        [q[0], q[1], q[2], q[3]],
                    ));
                }
            }
            GpuRigidShape::TriangleMesh {
                vertices,
                triangles,
            } => {
                mappings
                    .try_reserve(triangles.len())
                    .map_err(|_| GpuRigidMpmCouplerError::Capacity)?;
                for [a, b, c] in triangles {
                    let [Some(a), Some(b), Some(c)] =
                        [a, b, c].map(|id| vertices.get(*id as usize))
                    else {
                        return Err(GpuRigidMpmCouplerError::TopologyChanged);
                    };
                    let center = (Vector3::from(*a) + Vector3::from(*b) + Vector3::from(*c)) / 3.0;
                    mappings.push(Mapping::new(body, [center.x, center.y, center.z], identity));
                }
            }
            _ => mappings.push(Mapping::new(body, [0.0; 3], identity)),
        }
        shapes.push(shape);
    }
    if rigid.config().ground_half_extent.is_some() {
        mappings.push(Mapping::new(u32::MAX, [0.0; 3], [0.0, 0.0, 0.0, 1.0]));
    }
    Ok((mappings, shapes))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use core::time::Duration;
    use std::sync::mpsc;

    use nalgebra::Vector3;
    use tessera_physics::gpu_contact_pipeline::GpuContactDevice;
    use tessera_physics::gpu_rigid_ball_joint::GpuRigidBallJoint;
    use tessera_physics::gpu_rigid_sphere_world::{
        GpuRigidPrimitiveWorld, GpuRigidSphereWorldConfig,
    };
    use tessera_physics::gpu_rigid_state::GpuRigidBodyState;

    use super::*;
    use crate::rigid_sync::gpu_rigid_world_obstacles;
    use crate::{GpuMpmTransfers, MaterialModel, MpmParams, MpmParticle, MpmWorld, WorldBounds};

    #[tokio::test]
    async fn mpm_owned_step_integrates_rigid_pose_without_cpu_particle_sync() {
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
        let state = GpuRigidBodyState {
            position_inverse_mass: [0.0, 0.0, 0.0, 1.0],
            orientation: [0.0, 0.0, 0.0, 1.0],
            linear_velocity: [1.0, 0.0, 0.0, 0.0],
            angular_velocity: [0.0, 0.0, 1.0, 0.0],
            inverse_inertia_sleep: [1.0, 1.0, 1.0, 0.0],
        };
        let rigid = GpuRigidPrimitiveWorld::new_primitives(
            &device,
            &queue,
            &[state],
            &[GpuRigidShape::Sphere { radius: 1.0 }],
            GpuRigidSphereWorldConfig {
                gravity: [0.0, 0.0, -9.81],
                ground_half_extent: None,
                ..Default::default()
            },
        )
        .unwrap();
        let particle = MpmParticle::new(
            Vector3::new(3.0, 0.0, 0.0),
            0.04,
            1000.0,
            MaterialModel::elastic(1000.0, 0.2),
        );
        let world = MpmWorld::new(
            vec![particle],
            MpmParams {
                gravity: Vector3::zeros(),
                ..MpmParams::default()
            },
        )
        .unwrap();
        let transfers = GpuMpmTransfers::new(&device);
        let mut mpm =
            GpuMpmResidentSession::new(&transfers, &device, &queue, world, 0.001).unwrap();
        let coupler = GpuRigidMpmCoupler::new(&rigid, &mut mpm).unwrap();
        coupler.submit_mpm_owned_step(&rigid, &mut mpm).unwrap();
        assert_eq!(mpm.pending_substeps(), 1);
        let first = rigid.readback().unwrap()[0];
        assert!((first.position_inverse_mass[0] - 0.001).abs() < 1e-6);
        assert!((first.position_inverse_mass[2] + 9.81e-6).abs() < 1e-7);
        assert!((first.orientation[2] - 0.0005).abs() < 1e-6);
        coupler.submit_mpm_owned_step(&rigid, &mut mpm).unwrap();
        let second = rigid.readback().unwrap()[0];
        assert!(second.position_inverse_mass[0] > first.position_inverse_mass[0]);
        assert!(second.orientation[2] > first.orientation[2]);
        mpm.synchronize().unwrap();
        assert_eq!(mpm.world().substeps, 2);
    }

    #[tokio::test]
    async fn mpm_owned_step_obeys_rigid_world_speed_cap_before_pose_integration() {
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
        let state = GpuRigidBodyState {
            position_inverse_mass: [0.0, 0.0, 0.0, 1.0],
            orientation: [0.0, 0.0, 0.0, 1.0],
            linear_velocity: [2.0, 0.0, 0.0, 0.0],
            angular_velocity: [0.0; 4],
            inverse_inertia_sleep: [1.0, 1.0, 1.0, 0.0],
        };
        let mut rigid = GpuRigidPrimitiveWorld::new_primitives(
            &device,
            &queue,
            &[state],
            &[GpuRigidShape::Sphere { radius: 1.0 }],
            GpuRigidSphereWorldConfig {
                gravity: [0.0; 3],
                ground_half_extent: None,
                ..Default::default()
            },
        )
        .unwrap();
        rigid.set_max_linear_speed(Some(0.5)).unwrap();
        let world = MpmWorld::new(
            vec![MpmParticle::new(
                Vector3::new(3.0, 0.0, 0.0),
                0.04,
                1000.0,
                MaterialModel::elastic(1000.0, 0.2),
            )],
            MpmParams {
                gravity: Vector3::zeros(),
                ..MpmParams::default()
            },
        )
        .unwrap();
        let transfers = GpuMpmTransfers::new(&device);
        let mut mpm =
            GpuMpmResidentSession::new(&transfers, &device, &queue, world, 0.001).unwrap();
        let coupler = GpuRigidMpmCoupler::new(&rigid, &mut mpm).unwrap();
        coupler.submit_mpm_owned_step(&rigid, &mut mpm).unwrap();
        let state = rigid.readback().unwrap()[0];
        assert!((state.linear_velocity[0] - 0.5).abs() < 1e-5);
        assert!((state.position_inverse_mass[0] - 0.0005).abs() < 1e-6);
    }

    #[tokio::test]
    async fn two_way_reaction_updates_rigid_state_before_particle_readback() {
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
        let state = GpuRigidBodyState {
            position_inverse_mass: [0.0, 0.0, 0.0, 1.0],
            orientation: [0.0, 0.0, 0.0, 1.0],
            linear_velocity: [0.0; 4],
            angular_velocity: [0.0; 4],
            inverse_inertia_sleep: [1.0, 1.0, 1.0, 0.0],
        };
        let rigid = GpuRigidPrimitiveWorld::new_primitives(
            &device,
            &queue,
            &[state],
            &[GpuRigidShape::Sphere { radius: 1.0 }],
            GpuRigidSphereWorldConfig {
                gravity: [0.0; 3],
                ground_half_extent: None,
                ..Default::default()
            },
        )
        .unwrap();
        let mut particle = MpmParticle::new(
            Vector3::new(1.02, 0.0, 0.0),
            0.04,
            1000.0,
            MaterialModel::elastic(1000.0, 0.2),
        );
        particle.velocity.x = -1.0;
        let mpm_world = MpmWorld::new(
            vec![particle],
            MpmParams {
                gravity: Vector3::zeros(),
                ..MpmParams::default()
            },
        )
        .unwrap();
        let transfers = GpuMpmTransfers::new(&device);
        let mut mpm =
            GpuMpmResidentSession::new(&transfers, &device, &queue, mpm_world, 0.001).unwrap();
        let coupler = GpuRigidMpmCoupler::new(&rigid, &mut mpm).unwrap();
        coupler.submit_two_way_step(&rigid, &mut mpm).unwrap();
        assert!(mpm.has_pending_steps());
        let states = rigid.readback().unwrap();
        assert!(states[0].linear_velocity[0] < -0.01, "{states:?}");
        mpm.synchronize().unwrap();
        assert_eq!(mpm.world().substeps, 1);
        assert!(mpm.world().particles[0].velocity.x > -1.0);
    }

    #[tokio::test]
    async fn two_way_reaction_reaches_seventeenth_rigid_body() {
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
        let states = (0..17)
            .map(|index| GpuRigidBodyState {
                position_inverse_mass: [
                    if index == 16 {
                        0.0
                    } else {
                        10.0 + index as f32 * 3.0
                    },
                    0.0,
                    0.0,
                    1.0,
                ],
                orientation: [0.0, 0.0, 0.0, 1.0],
                linear_velocity: [0.0; 4],
                angular_velocity: [0.0; 4],
                inverse_inertia_sleep: [1.0, 1.0, 1.0, 0.0],
            })
            .collect::<Vec<_>>();
        let rigid = GpuRigidPrimitiveWorld::new_primitives(
            &device,
            &queue,
            &states,
            &vec![GpuRigidShape::Sphere { radius: 1.0 }; 17],
            GpuRigidSphereWorldConfig {
                gravity: [0.0; 3],
                ground_half_extent: None,
                ..Default::default()
            },
        )
        .unwrap();
        let mut particle = MpmParticle::new(
            Vector3::new(1.02, 0.0, 0.0),
            0.04,
            1000.0,
            MaterialModel::elastic(1000.0, 0.2),
        );
        particle.velocity.x = -1.0;
        let mpm_world = MpmWorld::new(
            vec![particle],
            MpmParams {
                gravity: Vector3::zeros(),
                ..MpmParams::default()
            },
        )
        .unwrap();
        let transfers = GpuMpmTransfers::new(&device);
        let mut mpm =
            GpuMpmResidentSession::new(&transfers, &device, &queue, mpm_world, 0.001).unwrap();
        let coupler = GpuRigidMpmCoupler::new(&rigid, &mut mpm).unwrap();
        coupler.submit_two_way_step(&rigid, &mut mpm).unwrap();
        let actual = rigid.readback().unwrap();
        assert!(
            actual[..16]
                .iter()
                .all(|state| state.linear_velocity[0].abs() < 1e-6)
        );
        assert!(actual[16].linear_velocity[0] < -0.01, "{actual:?}");
        mpm.synchronize().unwrap();
        assert_eq!(mpm.world().substeps, 1);
    }

    #[tokio::test]
    async fn coupled_step_solves_mpm_reaction_and_rigid_ground_contact() {
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
        let mut rigid = GpuRigidPrimitiveWorld::new_primitives(
            &device,
            &queue,
            &[GpuRigidBodyState {
                position_inverse_mass: [0.0, 0.0, 0.95, 1.0],
                orientation: [0.0, 0.0, 0.0, 1.0],
                linear_velocity: [0.0, 0.0, -1.0, 0.0],
                angular_velocity: [0.0; 4],
                inverse_inertia_sleep: [1.0, 1.0, 1.0, 0.0],
            }],
            &[GpuRigidShape::Sphere { radius: 1.0 }],
            GpuRigidSphereWorldConfig {
                gravity: [0.0; 3],
                ground_half_extent: Some(10.0),
                ..Default::default()
            },
        )
        .unwrap();
        let mut particle = MpmParticle::new(
            Vector3::new(1.02, 0.0, 0.95),
            0.04,
            1000.0,
            MaterialModel::elastic(1000.0, 0.2),
        );
        particle.velocity.x = -1.0;
        let world = MpmWorld::new(
            vec![particle],
            MpmParams {
                gravity: Vector3::zeros(),
                ..MpmParams::default()
            },
        )
        .unwrap();
        let transfers = GpuMpmTransfers::new(&device);
        let mut mpm =
            GpuMpmResidentSession::new(&transfers, &device, &queue, world, 0.001).unwrap();
        let coupler = GpuRigidMpmCoupler::new(&rigid, &mut mpm).unwrap();
        let _candidate_count = coupler.submit_coupled_step(&mut rigid, &mut mpm).unwrap();
        assert_eq!(mpm.pending_substeps(), 1);
        assert!(matches!(
            coupler.step_coupled(&mut rigid, &mut mpm),
            Err(GpuRigidMpmCouplerError::Mpm(GpuMpmError::PendingSteps))
        ));
        let state = rigid.readback().unwrap()[0];
        assert!(state.linear_velocity[0] < -0.01, "{state:?}");
        assert!(state.linear_velocity[2] > -0.5, "{state:?}");
        mpm.synchronize().unwrap();
        assert_eq!(mpm.world().substeps, 1);
        assert!(mpm.world().particles[0].velocity.x > -1.0);
        let _candidate_count = coupler.step_coupled(&mut rigid, &mut mpm).unwrap();
        assert_eq!(mpm.world().substeps, 2);
    }

    #[tokio::test]
    async fn coupled_step_transmits_mpm_reaction_through_ball_joint() {
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
        let states = [
            GpuRigidBodyState {
                position_inverse_mass: [0.0, 0.0, 0.0, 0.0],
                orientation: [0.0, 0.0, 0.0, 1.0],
                linear_velocity: [0.0; 4],
                angular_velocity: [0.0; 4],
                inverse_inertia_sleep: [0.0; 4],
            },
            GpuRigidBodyState {
                position_inverse_mass: [1.0, 0.0, 0.0, 1.0],
                orientation: [0.0, 0.0, 0.0, 1.0],
                linear_velocity: [0.0; 4],
                angular_velocity: [0.0; 4],
                inverse_inertia_sleep: [1.0, 1.0, 1.0, 0.0],
            },
        ];
        let shapes = [
            GpuRigidShape::Sphere { radius: 0.1 },
            GpuRigidShape::Sphere { radius: 0.5 },
        ];
        let config = GpuRigidSphereWorldConfig {
            gravity: [0.0; 3],
            ground_half_extent: None,
            ..Default::default()
        };
        let particle = || {
            let mut value = MpmParticle::new(
                Vector3::new(1.52, 0.0, 0.0),
                0.04,
                1000.0,
                MaterialModel::elastic(1000.0, 0.2),
            );
            value.velocity.x = -1.0;
            value
        };
        let transfers = GpuMpmTransfers::new(&device);
        let mut free =
            GpuRigidPrimitiveWorld::new_primitives(&device, &queue, &states, &shapes, config)
                .unwrap();
        let mut jointed =
            GpuRigidPrimitiveWorld::new_primitives(&device, &queue, &states, &shapes, config)
                .unwrap();
        jointed
            .set_ball_joints(&[GpuRigidBallJoint {
                body_a: 0,
                body_b: 1,
                local_anchor_a: [1.0, 0.0, 0.0],
                local_anchor_b: [0.0; 3],
            }])
            .unwrap();
        let new_mpm = || {
            MpmWorld::new(
                vec![particle()],
                MpmParams {
                    gravity: Vector3::zeros(),
                    ..MpmParams::default()
                },
            )
            .unwrap()
        };
        let mut free_mpm =
            GpuMpmResidentSession::new(&transfers, &device, &queue, new_mpm(), 0.001).unwrap();
        let mut jointed_mpm =
            GpuMpmResidentSession::new(&transfers, &device, &queue, new_mpm(), 0.001).unwrap();
        let free_coupler = GpuRigidMpmCoupler::new(&free, &mut free_mpm).unwrap();
        let jointed_coupler = GpuRigidMpmCoupler::new(&jointed, &mut jointed_mpm).unwrap();
        let _free_candidates = free_coupler.step_coupled(&mut free, &mut free_mpm).unwrap();
        let _jointed_candidates = jointed_coupler
            .step_coupled(&mut jointed, &mut jointed_mpm)
            .unwrap();
        let free_velocity = free.readback().unwrap()[1].linear_velocity[0];
        let jointed_velocity = jointed.readback().unwrap()[1].linear_velocity[0];
        assert!(free_velocity < -0.01, "{free_velocity}");
        assert!(
            jointed_velocity.abs() < free_velocity.abs() * 0.5,
            "free={free_velocity}, jointed={jointed_velocity}"
        );
        assert_eq!(free_mpm.world().substeps, 1);
        assert_eq!(jointed_mpm.world().substeps, 1);
    }

    fn read_obstacles(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
        count: usize,
    ) -> Vec<u8> {
        let bytes = count as u64 * 144;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera MPM obstacle sync test staging"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, bytes);
        let _ = queue.submit(Some(encoder.finish()));
        let (sender, receiver) = mpsc::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        let _ = device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(10)),
            })
            .unwrap();
        receiver
            .recv_timeout(Duration::from_secs(10))
            .unwrap()
            .unwrap();
        let view = staging.slice(..).get_mapped_range();
        let bytes = view.to_vec();
        drop(view);
        staging.unmap();
        bytes
    }

    fn f32_at(bytes: &[u8], obstacle: usize, offset: usize) -> f32 {
        let start = obstacle * 144 + offset;
        f32::from_le_bytes(bytes[start..start + 4].try_into().unwrap())
    }

    #[test]
    fn resident_rigid_obstacles_follow_all_shape_poses_without_state_readback() {
        fn check(context: &GpuContactDevice) {
            let cube = [-0.1, 0.1]
                .into_iter()
                .flat_map(|x| {
                    [-0.1, 0.1]
                        .into_iter()
                        .flat_map(move |y| [-0.1, 0.1].into_iter().map(move |z| [x, y, z]))
                })
                .collect();
            let shapes = vec![
                GpuRigidShape::Sphere { radius: 0.1 },
                GpuRigidShape::Box {
                    half_extents: [0.1; 3],
                },
                GpuRigidShape::Capsule {
                    radius: 0.1,
                    half_length: 0.1,
                },
                GpuRigidShape::Cylinder {
                    radius: 0.1,
                    half_length: 0.1,
                },
                GpuRigidShape::Cone {
                    radius: 0.1,
                    half_length: 0.1,
                },
                GpuRigidShape::Convex { vertices: cube },
                GpuRigidShape::Polyline {
                    vertices: vec![[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.2, 0.2, 0.0]],
                    segments: vec![[0, 1], [1, 2]],
                },
                GpuRigidShape::TriangleMesh {
                    vertices: vec![[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0]],
                    triangles: vec![[0, 1, 2]],
                },
            ];
            let mut states = (0..shapes.len())
                .map(|index| GpuRigidBodyState {
                    position_inverse_mass: [index as f32, 0.5, 0.5, 1.0],
                    orientation: [0.0, 0.0, 0.0, 1.0],
                    linear_velocity: [0.0; 4],
                    angular_velocity: [0.0; 4],
                    inverse_inertia_sleep: [1.0, 1.0, 1.0, 0.0],
                })
                .collect::<Vec<_>>();
            let mut rigid = GpuRigidPrimitiveWorld::new_primitives(
                context.device(),
                context.queue(),
                &states,
                &shapes,
                GpuRigidSphereWorldConfig {
                    gravity: [0.0; 3],
                    ground_half_extent: Some(10.0),
                    ..Default::default()
                },
            )
            .unwrap();
            let particle = MpmParticle::new(
                Vector3::new(0.5, 0.5, 0.5),
                0.03,
                1000.0,
                MaterialModel::elastic(1000.0, 0.2),
            );
            let mpm_world = MpmWorld::new(
                vec![particle],
                MpmParams {
                    gravity: Vector3::zeros(),
                    bounds: Some(WorldBounds {
                        min: Vector3::zeros(),
                        max: Vector3::repeat(1.0),
                    }),
                    ..MpmParams::default()
                },
            )
            .unwrap();
            let transfers = GpuMpmTransfers::new(context.device());
            let mut mpm = GpuMpmResidentSession::new(
                &transfers,
                context.device(),
                context.queue(),
                mpm_world,
                0.0001,
            )
            .unwrap();
            let coupler = GpuRigidMpmCoupler::new(&rigid, &mut mpm).unwrap();
            assert_eq!(mpm.world().obstacles.len(), 10);

            let angle = core::f32::consts::FRAC_PI_2;
            for (index, state) in states.iter_mut().enumerate() {
                state.position_inverse_mass[0] += 0.1;
                state.orientation = [0.0, 0.0, (angle / 2.0).sin(), (angle / 2.0).cos()];
                state.linear_velocity = [1.0, 2.0, 3.0, 0.0];
                state.angular_velocity = [0.0, 0.0, 2.0, 0.0];
                rigid.write_body(index, *state).unwrap();
            }
            let mut encoder = context.device().create_command_encoder(&Default::default());
            coupler.encode(&mut encoder, &rigid, &mpm).unwrap();
            let _ = context.queue().submit(Some(encoder.finish()));
            let expected = gpu_rigid_world_obstacles(&rigid).unwrap();
            let actual = read_obstacles(
                context.device(),
                context.queue(),
                mpm.obstacle_buffer().unwrap(),
                expected.len(),
            );
            for (index, obstacle) in expected.iter().enumerate() {
                for axis in 0..3 {
                    assert!(
                        (f64::from(f32_at(&actual, index, axis * 4)) - obstacle.center[axis]).abs()
                            < 2e-5,
                        "center {index} {axis}"
                    );
                    assert!(
                        (f64::from(f32_at(&actual, index, 48 + axis * 4))
                            - obstacle.linear_velocity[axis])
                            .abs()
                            < 2e-5,
                        "linear {index} {axis}"
                    );
                    assert!(
                        (f64::from(f32_at(&actual, index, 64 + axis * 4))
                            - obstacle.angular_velocity[axis])
                            .abs()
                            < 2e-5,
                        "angular {index} {axis}"
                    );
                }
                let q = obstacle.orientation.quaternion().coords;
                for axis in 0..4 {
                    assert!(
                        (f64::from(f32_at(&actual, index, 32 + axis * 4)) - q[axis]).abs() < 2e-5,
                        "orientation {index} {axis}: actual {}, expected {}",
                        f32_at(&actual, index, 32 + axis * 4),
                        q[axis]
                    );
                }
            }
            mpm.step(1).unwrap();
            coupler.step(&rigid, &mut mpm, 1).unwrap();
            assert_eq!(mpm.world().substeps, 2);
            for offset in [0.02, 0.04] {
                states[0].position_inverse_mass[0] += offset;
                rigid.write_body(0, states[0]).unwrap();
                coupler.submit_steps(&rigid, &mut mpm, 1).unwrap();
            }
            assert_eq!(mpm.pending_substeps(), 2);
            assert_eq!(mpm.world().substeps, 2);
            mpm.synchronize().unwrap();
            assert_eq!(mpm.world().substeps, 4);
            let expected = gpu_rigid_world_obstacles(&rigid).unwrap();
            let actual = read_obstacles(
                context.device(),
                context.queue(),
                mpm.obstacle_buffer().unwrap(),
                expected.len(),
            );
            assert!((f64::from(f32_at(&actual, 0, 0)) - expected[0].center.x).abs() < 2e-5);
            rigid
                .set_body_material(
                    0,
                    tessera_physics::material::ColliderMaterial::new(0.8, 0.0),
                )
                .unwrap();
            let mut encoder = context.device().create_command_encoder(&Default::default());
            assert!(matches!(
                coupler.encode(&mut encoder, &rigid, &mpm),
                Err(GpuRigidMpmCouplerError::TopologyChanged)
            ));
        }
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            eprintln!("resident rigid MPM sync backend: {backend:?}");
            check(&context);
        }
        assert!(tested > 0);
    }
}
