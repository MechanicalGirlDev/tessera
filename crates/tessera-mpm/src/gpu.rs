//! WebGPU particle-grid transfers and selected constitutive models for 3D MPM.

use core::mem::{size_of, size_of_val};
use core::time::Duration;
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::mpsc;

use nalgebra::{Matrix3, Vector3};
use wgpu::util::DeviceExt;

use crate::gpu_topology::{HashTopology, HashTopologyState};
use crate::material::{MaterialModel, PlasticState};
use crate::obstacle::{ObstacleShape, RigidObstacle};
use crate::world::{
    MpmError, MpmParams, MpmParticle, MpmWorld, ObstacleReaction, ParticleChunkId, WorldBounds,
};

/// GPU snapshot record for rendering or subsequent compute passes.
///
/// WGSL layout: two consecutive `vec4<f32>` values, with a 32-byte array stride.
/// Records retain world particle order; zero mass marks disabled particles.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuMpmParticleSnapshot {
    /// World position in xyz and enabled mass in w.
    pub position_mass: [f32; 4],
    /// World velocity in xyz and rest volume in w.
    pub velocity_volume: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuParticle {
    position_mass: [f32; 4],
    velocity_volume: [f32; 4],
    force_damping: [f32; 4],
    radius: [f32; 4],
    affine: [[f32; 4]; 3],
    stress: [[f32; 4]; 3],
    deformation: [[f32; 4]; 3],
    material: [f32; 4],
    material_extra: [f32; 4],
    projection: [f32; 4],
    base_cell: [i32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuParams {
    origin: [i32; 4],
    dimensions: [u32; 4],
    scalars: [f32; 4],
    gravity: [f32; 4],
    bound_min: [f32; 4],
    bound_max: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuObstacle {
    center_radius: [f32; 4],
    half_extents_kind: [f32; 4],
    orientation: [f32; 4],
    linear_velocity_friction: [f32; 4],
    angular_velocity: [f32; 4],
    triangle_a: [f32; 4],
    triangle_b: [f32; 4],
    triangle_c: [f32; 4],
    convex_range: [u32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuConvexPlane {
    normal_offset: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GpuTransfer {
    position: [f32; 4],
    velocity: [f32; 4],
    affine: [[f32; 4]; 3],
    deformation: [[f32; 4]; 3],
    plastic: [f32; 4],
}

impl GpuTransfer {
    fn position(self) -> Vector3<f64> {
        Vector3::new(
            f64::from(self.position[0]),
            f64::from(self.position[1]),
            f64::from(self.position[2]),
        )
    }

    fn velocity(self) -> Vector3<f64> {
        Vector3::new(
            f64::from(self.velocity[0]),
            f64::from(self.velocity[1]),
            f64::from(self.velocity[2]),
        )
    }

    fn affine(self) -> Matrix3<f64> {
        unpack_matrix(self.affine)
    }

    fn deformation(self) -> Matrix3<f64> {
        unpack_matrix(self.deformation)
    }

    fn plastic(self) -> PlasticState {
        PlasticState {
            plastic_det: f64::from(self.plastic[0]),
            hardening: f64::from(self.plastic[1]),
            log_volume_gain: f64::from(self.plastic[2]),
        }
    }
}

fn unpack_matrix(columns: [[f32; 4]; 3]) -> Matrix3<f64> {
    Matrix3::from_columns(&columns.map(|column| {
        Vector3::new(
            f64::from(column[0]),
            f64::from(column[1]),
            f64::from(column[2]),
        )
    }))
}

/// Invalid GPU input, capacity exhaustion, or a failed GPU transfer.
#[derive(Debug, thiserror::Error)]
pub enum GpuMpmError {
    /// An MPM particle or parameter cannot be represented by the compute shader.
    #[error("invalid or unrepresentable GPU MPM input")]
    InvalidInput,
    /// The grid, particle index buffer, or dispatch exceeds device capacity.
    #[error("GPU MPM transfer exceeds the device capacity")]
    Capacity,
    /// A submitted step must be synchronized before another step or state edit.
    #[error("GPU MPM steps are pending; synchronize before stepping or editing state")]
    PendingSteps,
    /// GPU completion or readback failed.
    #[error("GPU MPM readback failed: {0}")]
    Readback(String),
    /// The resident path needs a CPU fallback; no world state was applied.
    #[error("GPU MPM resident substeps require a CPU fallback; world state is unchanged")]
    ResidentUnavailable,
    /// The CPU material update failed after GPU transfer.
    #[error(transparent)]
    Reference(#[from] MpmError),
}

/// Reusable WebGPU P2G, grid-update, and G2P compute pipelines.
/// Resident sparse grids rebuild stencil coordinates and hash ownership on GPU
/// each substep; bounded dense grids remain available when they use less storage.
#[derive(Clone, Debug)]
pub struct GpuMpmTransfers {
    topology: HashTopology,
    snapshot: wgpu::ComputePipeline,
    update_cpic_colors: wgpu::ComputePipeline,
    p2g: wgpu::ComputePipeline,
    update_grid: wgpu::ComputePipeline,
    g2p: wgpu::ComputePipeline,
}

#[derive(Debug)]
struct ReactionPipelines {
    grid: wgpu::ComputePipeline,
    particle: wgpu::ComputePipeline,
    finalize: wgpu::ComputePipeline,
}

#[derive(Debug)]
struct PreparedTransfer {
    topology: Option<HashTopologyState>,
    count: u32,
    obstacle_count: u32,
    reaction_capacity: u32,
    cpic_count: u32,
    nodes: u64,
    output_bytes: u64,
    params: GpuParams,
    particle_buffer: wgpu::Buffer,
    grid_buffer: wgpu::Buffer,
    node_dispatch_buffer: wgpu::Buffer,
    velocity_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    readback: wgpu::Buffer,
    config_buffer: wgpu::Buffer,
    obstacle_buffer: wgpu::Buffer,
    convex_plane_buffer: wgpu::Buffer,
    node_index_buffer: wgpu::Buffer,
    node_coord_buffer: wgpu::Buffer,
    bindings: wgpu::BindGroup,
}

/// MPM world with particle and grid buffers reused across GPU steps.
/// Unbounded worlds use a GPU sparse hash grid; bounded worlds may use dense grids.
///
/// The CPU world is synchronized after each call to [`Self::step`]. Particle
/// insertions and removals rebuild the GPU particle and grid buffers.
#[derive(Debug)]
pub struct GpuMpmResidentSession<'a> {
    gpu: Cow<'a, GpuMpmTransfers>,
    device: Cow<'a, wgpu::Device>,
    queue: Cow<'a, wgpu::Queue>,
    world: MpmWorld,
    prepared: Option<PreparedTransfer>,
    substep_dt: f64,
    pending_steps: Option<u64>,
    reaction_pipelines: Option<ReactionPipelines>,
    valid: bool,
}

impl<'a> GpuMpmResidentSession<'a> {
    /// Device owning the resident particle and obstacle buffers.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub(crate) fn obstacle_buffer(&self) -> Option<&wgpu::Buffer> {
        self.prepared
            .as_ref()
            .map(|prepared| &prepared.obstacle_buffer)
    }

    pub(crate) fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Retain cloned GPU handles so this session can outlive its creator.
    ///
    /// Cloning handles preserves the existing pipelines and buffers; it does
    /// not upload particles or build new compute pipelines.
    pub fn into_owned(self) -> GpuMpmResidentSession<'static> {
        GpuMpmResidentSession {
            gpu: Cow::Owned(self.gpu.into_owned()),
            device: Cow::Owned(self.device.into_owned()),
            queue: Cow::Owned(self.queue.into_owned()),
            world: self.world,
            prepared: self.prepared,
            substep_dt: self.substep_dt,
            pending_steps: self.pending_steps,
            reaction_pipelines: self.reaction_pipelines,
            valid: self.valid,
        }
    }

    /// Upload a world and retain its GPU buffers for later steps.
    pub fn new(
        gpu: &'a GpuMpmTransfers,
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        world: MpmWorld,
        substep_dt: f64,
    ) -> Result<Self, GpuMpmError> {
        validate_resident_world(&world, substep_dt)?;
        let prepared = gpu.prepare(
            device,
            &world.particles,
            &world.params,
            &world.obstacles,
            substep_dt,
            true,
        )?;
        Ok(Self {
            gpu: Cow::Borrowed(gpu),
            device: Cow::Borrowed(device),
            queue: Cow::Borrowed(queue),
            world,
            prepared,
            substep_dt,
            pending_steps: None,
            reaction_pipelines: None,
            valid: true,
        })
    }

    /// Run fixed substeps using the buffers retained by this session.
    ///
    /// An error after GPU submission invalidates the session. The last
    /// synchronized CPU world remains available through [`Self::into_world`].
    pub fn step(&mut self, steps: u32) -> Result<(), GpuMpmError> {
        if self.pending_steps.is_some() {
            return Err(GpuMpmError::PendingSteps);
        }
        self.submit_steps(steps)?;
        self.synchronize()
    }

    /// Submit fixed substeps without waiting or reading particle state back to CPU.
    ///
    /// Multiple batches can be queued before [`Self::synchronize`]. Each call
    /// accepts 1 through 64 substeps. Pending counts accumulate with checked
    /// arithmetic. Synchronize before a synchronous step, obstacle edit, or
    /// particle edit. [`Self::world`] continues to
    /// expose the last synchronized state. GPU snapshots can consume the submitted
    /// result in later submissions on this session's queue, but remain provisional
    /// until synchronization checks for capacity, material, and non-finite errors.
    pub fn submit_steps(&mut self, steps: u32) -> Result<(), GpuMpmError> {
        if !self.valid {
            return Err(GpuMpmError::ResidentUnavailable);
        }
        if steps == 0 || steps > 64 {
            return Err(GpuMpmError::InvalidInput);
        }
        let total = self
            .pending_steps
            .unwrap_or(0)
            .checked_add(u64::from(steps))
            .ok_or(GpuMpmError::Capacity)?;
        validate_resident_world(&self.world, self.substep_dt)?;
        if let Some(prepared) = &self.prepared {
            self.gpu
                .submit_prepared(&self.device, &self.queue, prepared, steps)?;
        }
        self.pending_steps = Some(total);
        Ok(())
    }

    pub(crate) fn encode_one_substep_with_reactions(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<u64, GpuMpmError> {
        if !self.valid {
            return Err(GpuMpmError::ResidentUnavailable);
        }
        let total = self
            .pending_steps
            .unwrap_or(0)
            .checked_add(1)
            .ok_or(GpuMpmError::Capacity)?;
        validate_resident_world(&self.world, self.substep_dt)?;
        if let Some(prepared) = &self.prepared {
            if self.reaction_pipelines.is_none() && prepared.obstacle_count > 0 {
                self.reaction_pipelines = Some(self.gpu.create_reaction_pipelines(&self.device));
            }
            self.gpu.encode_prepared(encoder, prepared, 1)?;
            if let Some(pipelines) = &self.reaction_pipelines {
                self.gpu.encode_reactions(encoder, prepared, pipelines)?;
            }
        }
        Ok(total)
    }

    pub(crate) fn reaction_output(&self) -> Option<(&wgpu::Buffer, u32, u32)> {
        self.prepared.as_ref().map(|prepared| {
            (
                &prepared.output_buffer,
                prepared.count,
                prepared.obstacle_count,
            )
        })
    }

    pub(crate) fn substep_dt(&self) -> f64 {
        self.substep_dt
    }

    pub(crate) fn mark_one_substep_submitted(&mut self, total: u64) {
        self.pending_steps = Some(total);
    }

    /// Whether submitted GPU substeps have not yet been checked and synchronized.
    pub fn has_pending_steps(&self) -> bool {
        self.pending_steps.is_some()
    }

    /// Number of submitted substeps not yet checked and synchronized.
    pub fn pending_substeps(&self) -> u64 {
        self.pending_steps.unwrap_or(0)
    }

    /// Wait for submitted substeps, validate their result, and update the CPU world.
    ///
    /// With no pending steps this is a no-op. A GPU or material error invalidates
    /// the session and preserves the last synchronized CPU world. Dropping a
    /// pending session or calling [`Self::into_world`] recovers only that last state.
    pub fn synchronize(&mut self) -> Result<(), GpuMpmError> {
        if !self.valid {
            return Err(GpuMpmError::ResidentUnavailable);
        }
        let Some(steps) = self.pending_steps.take() else {
            return Ok(());
        };
        let transfers = if let Some(prepared) = &self.prepared {
            match self
                .gpu
                .read_prepared(&self.device, &self.queue, prepared, None, None)
            {
                Ok(transfers) => transfers,
                Err(error) => {
                    self.valid = false;
                    return Err(error);
                }
            }
        } else {
            vec![GpuTransfer::default(); self.world.particles.len()]
        };
        let mut updated = self.world.clone();
        if let Err(error) = updated.apply_gpu_transfers(transfers) {
            self.valid = false;
            return Err(error);
        }
        updated.substeps = updated.substeps.saturating_add(steps);
        self.world = updated;
        Ok(())
    }

    /// Await GPU completion without blocking the caller's thread.
    ///
    /// Browser WebGPU delivers map callbacks from its event loop. Native backends
    /// poll on a helper thread. Cancellation unmaps the staging buffer and keeps
    /// pending substeps, so synchronization may be retried. World state changes
    /// only after validation succeeds; errors invalidate the session just as in
    /// [`Self::synchronize`]. No pending work makes this a no-op.
    pub async fn synchronize_async(&mut self) -> Result<(), GpuMpmError> {
        if !self.valid {
            return Err(GpuMpmError::ResidentUnavailable);
        }
        let Some(steps) = self.pending_steps else {
            return Ok(());
        };
        let result = if let Some(prepared) = &self.prepared {
            self.gpu
                .read_prepared_async(&self.device, &self.queue, prepared)
                .await
        } else {
            Ok(vec![GpuTransfer::default(); self.world.particles.len()])
        };
        self.pending_steps = None;
        let transfers = match result {
            Ok(transfers) => transfers,
            Err(error) => {
                self.valid = false;
                return Err(error);
            }
        };
        let mut updated = self.world.clone();
        if let Err(error) = updated.apply_gpu_transfers(transfers) {
            self.valid = false;
            return Err(error);
        }
        updated.substeps = updated.substeps.saturating_add(steps);
        self.world = updated;
        Ok(())
    }

    /// Replace a particle's one-shot force without rebuilding GPU buffers.
    ///
    /// The force is consumed by the next substep. Fixed or disabled particles do
    /// not accelerate. Pending substeps must be synchronized before editing.
    /// Invalid indices, non-finite or unrepresentable forces leave state unchanged.
    pub fn set_particle_force(
        &mut self,
        index: usize,
        force: Vector3<f64>,
    ) -> Result<(), GpuMpmError> {
        if !self.valid {
            return Err(GpuMpmError::ResidentUnavailable);
        }
        if self.pending_steps.is_some() {
            return Err(GpuMpmError::PendingSteps);
        }
        let particle = self
            .world
            .particles
            .get_mut(index)
            .ok_or(GpuMpmError::InvalidInput)?;
        let packed = vec4(force, particle.damping)?;
        if let Some(prepared) = &self.prepared {
            let offset = index as u64 * size_of::<GpuParticle>() as u64
                + core::mem::offset_of!(GpuParticle, force_damping) as u64;
            self.queue.write_buffer(
                &prepared.particle_buffer,
                offset,
                bytemuck::cast_slice(&packed),
            );
        }
        particle.force = force;
        Ok(())
    }

    /// Replace prescribed rigid obstacles, growing GPU buffers when needed.
    ///
    /// The next step sees the new poses, velocities, shapes, and obstacle count.
    /// Invalid input leaves the current world and GPU bindings unchanged.
    pub fn set_obstacles(&mut self, obstacles: Vec<RigidObstacle>) -> Result<(), GpuMpmError> {
        if !self.valid {
            return Err(GpuMpmError::ResidentUnavailable);
        }
        if self.pending_steps.is_some() {
            return Err(GpuMpmError::PendingSteps);
        }
        if obstacles.iter().any(|obstacle| !obstacle.is_valid()) {
            return Err(GpuMpmError::Reference(MpmError::InvalidInput));
        }
        let needs_sparse_rebuild = obstacles
            .iter()
            .any(|obstacle| obstacle.cpic_group.is_some())
            && self
                .prepared
                .as_ref()
                .is_some_and(|prepared| prepared.topology.is_none());
        let needs_reaction_capacity = self
            .prepared
            .as_ref()
            .is_some_and(|prepared| obstacles.len() > prepared.reaction_capacity as usize);
        if needs_sparse_rebuild || needs_reaction_capacity {
            let replacement = self.gpu.prepare(
                &self.device,
                &self.world.particles,
                &self.world.params,
                &obstacles,
                self.substep_dt,
                true,
            )?;
            self.prepared = replacement;
        } else if let Some(prepared) = &mut self.prepared {
            self.gpu
                .replace_obstacles(&self.device, prepared, &obstacles)?;
        }
        self.world.set_obstacles(obstacles)?;
        Ok(())
    }

    /// Append a validated particle batch and return its stable removal handle.
    ///
    /// The current GPU state is replaced only after all new buffers are ready.
    /// The last synchronized particle state and substep count are preserved.
    pub fn add_particles(
        &mut self,
        particles: Vec<MpmParticle>,
    ) -> Result<ParticleChunkId, GpuMpmError> {
        if !self.valid {
            return Err(GpuMpmError::ResidentUnavailable);
        }
        if self.pending_steps.is_some() {
            return Err(GpuMpmError::PendingSteps);
        }
        let mut updated = self.world.clone();
        let chunk = updated.add_particles(particles)?;
        self.replace_particle_world(updated)?;
        Ok(chunk)
    }

    /// Remove a particle chunk while preserving the order of remaining particles.
    /// The old world and GPU buffers remain available if rebuilding fails.
    pub fn remove_chunk(&mut self, chunk: ParticleChunkId) -> Result<usize, GpuMpmError> {
        if !self.valid {
            return Err(GpuMpmError::ResidentUnavailable);
        }
        if self.pending_steps.is_some() {
            return Err(GpuMpmError::PendingSteps);
        }
        let mut updated = self.world.clone();
        let removed = updated.remove_chunk(chunk)?;
        self.replace_particle_world(updated)?;
        Ok(removed)
    }

    fn replace_particle_world(&mut self, updated: MpmWorld) -> Result<(), GpuMpmError> {
        validate_resident_world(&updated, self.substep_dt)?;
        let prepared = self.gpu.prepare(
            &self.device,
            &updated.particles,
            &updated.params,
            &updated.obstacles,
            self.substep_dt,
            true,
        )?;
        self.prepared = prepared;
        self.world = updated;
        Ok(())
    }

    /// Encode a compact position/velocity snapshot without CPU readback or submission.
    ///
    /// `output` must be a STORAGE buffer from this session's device with room for
    /// `world().particles.len()` [`GpuMpmParticleSnapshot`] records. Submit the
    /// encoder on the session's queue before consuming output in another queue
    /// submission; subsequent passes in this encoder can consume it directly.
    /// Returns the number of records written. If no particles are enabled, returns
    /// zero and leaves output untouched. Otherwise disabled particles are included
    /// with zero mass, preserving world order. Invalidated sessions cannot export
    /// partially computed state. Particle edits may change the required capacity.
    ///
    /// This only exports state; [`Self::step`] still synchronizes the CPU world.
    /// After [`Self::submit_steps`], snapshots are provisional until
    /// [`Self::synchronize`] has checked GPU error flags.
    pub fn encode_particle_snapshot(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        output: &wgpu::Buffer,
    ) -> Result<u32, GpuMpmError> {
        if !self.valid {
            return Err(GpuMpmError::ResidentUnavailable);
        }
        let Some(prepared) = &self.prepared else {
            return Ok(0);
        };
        let bytes = u64::from(prepared.count) * size_of::<GpuMpmParticleSnapshot>() as u64;
        if !output.usage().contains(wgpu::BufferUsages::STORAGE) {
            return Err(GpuMpmError::InvalidInput);
        }
        let limits = self.device.limits();
        if output.size() < bytes
            || output.size() > u64::from(limits.max_storage_buffer_binding_size)
        {
            return Err(GpuMpmError::Capacity);
        }
        let config = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tessera MPM snapshot layout"),
                contents: bytemuck::cast_slice(&[
                    prepared.count,
                    (size_of::<GpuParticle>() / 16) as u32,
                    0,
                    0,
                ]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bindings = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tessera MPM snapshot buffers"),
            layout: &self.gpu.snapshot.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: prepared.particle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: config.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&self.gpu.snapshot);
        pass.set_bind_group(0, &bindings, &[]);
        pass.dispatch_workgroups(prepared.count.div_ceil(64), 1, 1);
        Ok(prepared.count)
    }

    /// Last successfully synchronized CPU state.
    pub fn world(&self) -> &MpmWorld {
        &self.world
    }

    /// Recover the last synchronized CPU world, including after a GPU error.
    pub fn into_world(self) -> MpmWorld {
        self.world
    }
}

fn validate_resident_world(world: &MpmWorld, substep_dt: f64) -> Result<(), GpuMpmError> {
    if !substep_dt.is_finite()
        || substep_dt <= 0.0
        || !world.params.max_substep.is_finite()
        || substep_dt > world.stable_timestep()
    {
        return Err(GpuMpmError::InvalidInput);
    }
    if world.obstacles.iter().any(|obstacle| !obstacle.is_valid()) {
        return Err(GpuMpmError::Reference(MpmError::InvalidInput));
    }
    Ok(())
}

struct GridTopology {
    origin: [i32; 3],
    dimensions: [u32; 3],
    nodes: u64,
    particle_nodes: Vec<u32>,
    node_coords: Vec<[i32; 4]>,
}

impl GridTopology {
    fn bounded_dense(
        bounds: WorldBounds,
        inputs: &[GpuParticle],
        inv_h: f32,
        max_nodes: u64,
    ) -> Result<Self, GpuMpmError> {
        if inputs
            .iter()
            .all(|particle| particle.position_mass[3] <= 0.0)
        {
            return Self::build(inputs, max_nodes, 4);
        }
        let mut origin = [0i32; 3];
        let mut dimensions = [0u32; 3];
        for axis in 0..3 {
            let min = finite_f32(bounds.min[axis])?;
            let max = finite_f32(bounds.max[axis])?;
            if min >= max {
                return Err(GpuMpmError::InvalidInput);
            }
            let first = (f64::from(min) * f64::from(inv_h) - 0.5).floor() - 2.0;
            let last = (f64::from(max) * f64::from(inv_h) - 0.5).floor() + 4.0;
            if first < f64::from(i32::MIN)
                || last > f64::from(i32::MAX)
                || !first.is_finite()
                || !last.is_finite()
            {
                return Err(GpuMpmError::Capacity);
            }
            origin[axis] = first as i32;
            dimensions[axis] =
                u32::try_from(last as i64 - first as i64 + 1).map_err(|_| GpuMpmError::Capacity)?;
            for particle in inputs
                .iter()
                .filter(|particle| particle.position_mass[3] > 0.0)
            {
                let position = particle.position_mass[axis];
                let radius = particle.radius[0];
                if radius * 2.0 >= max - min || position < min || position > max {
                    return Err(GpuMpmError::ResidentUnavailable);
                }
            }
        }
        let nodes = dimensions
            .iter()
            .try_fold(1u64, |product, dimension| {
                product.checked_mul(u64::from(*dimension))
            })
            .ok_or(GpuMpmError::Capacity)?;
        if nodes > max_nodes {
            return Err(GpuMpmError::Capacity);
        }
        Ok(Self {
            origin,
            dimensions,
            nodes,
            particle_nodes: vec![0],
            node_coords: vec![[0; 4]],
        })
    }

    fn hashed(inputs: &[GpuParticle], max_nodes: u64) -> Result<Self, GpuMpmError> {
        let slots = u32::try_from(inputs.len())
            .map_err(|_| GpuMpmError::Capacity)?
            .checked_mul(27)
            .ok_or(GpuMpmError::Capacity)?;
        let buckets = slots
            .checked_mul(2)
            .and_then(u32::checked_next_power_of_two)
            .filter(|buckets| u64::from(slots) <= max_nodes && *buckets <= u32::MAX / 2)
            .ok_or(GpuMpmError::Capacity)?;
        Ok(Self {
            origin: [0; 3],
            dimensions: [0, buckets, 0],
            nodes: u64::from(slots),
            particle_nodes: vec![0; slots as usize],
            node_coords: vec![[0; 4]; slots as usize],
        })
    }

    fn local_dense(inputs: &[GpuParticle], max_nodes: u64) -> Option<Self> {
        if inputs
            .iter()
            .any(|particle| particle.radius[2].to_bits() != 0 || particle.radius[3].to_bits() != 0)
        {
            return None;
        }
        let mut lower = [i32::MAX; 3];
        let mut upper = [i32::MIN; 3];
        let mut active = 0u64;
        for particle in inputs
            .iter()
            .filter(|particle| particle.position_mass[3] > 0.0)
        {
            active += 1;
            for axis in 0..3 {
                lower[axis] = lower[axis].min(particle.base_cell[axis]);
                upper[axis] = upper[axis].max(particle.base_cell[axis] + 2);
            }
        }
        if active == 0 {
            return Some(Self {
                origin: [0; 3],
                dimensions: [0; 3],
                nodes: 0,
                particle_nodes: vec![0],
                node_coords: vec![[0; 4]],
            });
        }

        let dimensions = core::array::from_fn(|axis| {
            u32::try_from(i64::from(upper[axis]) - i64::from(lower[axis]) + 1).ok()
        });
        let dense_nodes = dimensions
            .iter()
            .copied()
            .try_fold(1u64, |product, dimension| {
                product.checked_mul(u64::from(dimension?))
            });
        if let ([Some(x), Some(y), Some(z)], Some(nodes)) = (dimensions, dense_nodes)
            && nodes <= max_nodes
            && nodes <= active.saturating_mul(27)
        {
            return Some(Self {
                origin: lower,
                dimensions: [x, y, z],
                nodes,
                particle_nodes: vec![0],
                node_coords: vec![[0; 4]],
            });
        }

        None
    }

    fn gpu_generated(
        inputs: &[GpuParticle],
        max_nodes: u64,
        max_index_bytes: u64,
        max_dispatch: u32,
    ) -> Result<Self, GpuMpmError> {
        if let Some(dense) = Self::local_dense(inputs, max_nodes) {
            return Ok(dense);
        }
        match Self::hashed(inputs, max_nodes) {
            Ok(hash)
                if (inputs.len() as u64 * 27).div_ceil(64) <= u64::from(max_dispatch)
                    && hash.dimensions[1].div_ceil(64) <= max_dispatch =>
            {
                Ok(hash)
            }
            _ => Self::build(inputs, max_nodes, max_index_bytes),
        }
    }

    fn build(
        inputs: &[GpuParticle],
        max_nodes: u64,
        max_index_bytes: u64,
    ) -> Result<Self, GpuMpmError> {
        if let Some(dense) = Self::local_dense(inputs, max_nodes) {
            return Ok(dense);
        }
        let slots = inputs.len().checked_mul(27).ok_or(GpuMpmError::Capacity)?;
        let index_bytes = u64::try_from(slots)
            .map_err(|_| GpuMpmError::Capacity)?
            .checked_mul(4)
            .ok_or(GpuMpmError::Capacity)?;
        if index_bytes > max_index_bytes {
            return Err(GpuMpmError::Capacity);
        }
        let mut particle_nodes = Vec::new();
        particle_nodes
            .try_reserve_exact(slots)
            .map_err(|_| GpuMpmError::Capacity)?;
        particle_nodes.resize(slots, 0);
        let mut node_ids = HashMap::<[i32; 5], u32>::new();
        let mut node_coords = Vec::<[i32; 4]>::new();
        for (particle_index, particle) in inputs.iter().enumerate() {
            if particle.position_mass[3] <= 0.0 {
                continue;
            }
            for x in 0..3usize {
                for y in 0..3usize {
                    for z in 0..3usize {
                        let key = [
                            particle.base_cell[0] + x as i32,
                            particle.base_cell[1] + y as i32,
                            particle.base_cell[2] + z as i32,
                            particle.radius[2].to_bits() as i32,
                            particle.radius[3].to_bits() as i32,
                        ];
                        let node = match node_ids.entry(key) {
                            std::collections::hash_map::Entry::Occupied(entry) => *entry.get(),
                            std::collections::hash_map::Entry::Vacant(entry) => {
                                if node_coords.len() as u64 >= max_nodes {
                                    return Err(GpuMpmError::Capacity);
                                }
                                let id = u32::try_from(node_coords.len())
                                    .map_err(|_| GpuMpmError::Capacity)?;
                                node_coords
                                    .try_reserve(1)
                                    .map_err(|_| GpuMpmError::Capacity)?;
                                node_coords.push([key[0], key[1], key[2], key[3]]);
                                *entry.insert(id)
                            }
                        };
                        particle_nodes[particle_index * 27 + (x * 3 + y) * 3 + z] = node;
                    }
                }
            }
        }
        Ok(Self {
            origin: [0; 3],
            dimensions: [0; 3],
            nodes: node_coords.len() as u64,
            particle_nodes,
            node_coords,
        })
    }
}

impl GpuMpmTransfers {
    /// Compile the three compute stages on a caller-selected WebGPU device.
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tessera MPM transfers"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu.wgsl").into()),
        });
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tessera MPM transfer layout"),
            entries: &[0u32, 1, 2, 3, 4, 5, 6, 7, 8, 9].map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: if binding == 4 || binding == 9 {
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    }
                } else {
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: binding >= 5,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    }
                },
                count: None,
            }),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tessera MPM compute layout"),
            bind_group_layouts: &[&bind_layout],
            immediate_size: 0,
        });
        let pipeline = |label, entry_point| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let snapshot_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tessera MPM particle snapshot"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_snapshot.wgsl").into()),
        });
        let snapshot = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tessera MPM snapshot pipeline"),
            layout: None,
            module: &snapshot_shader,
            entry_point: Some("snapshot"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            snapshot,
            topology: HashTopology::new(device),
            update_cpic_colors: pipeline("tessera MPM CPIC colors", "update_cpic_colors"),
            p2g: pipeline("tessera MPM P2G", "p2g"),
            update_grid: pipeline("tessera MPM grid update", "update_grid"),
            g2p: pipeline("tessera MPM G2P", "g2p"),
        }
    }

    fn create_reaction_pipelines(&self, device: &wgpu::Device) -> ReactionPipelines {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tessera MPM reaction shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu.wgsl").into()),
        });
        let bind_layout = self.p2g.get_bind_group_layout(0);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tessera MPM reaction layout"),
            bind_group_layouts: &[&bind_layout],
            immediate_size: 0,
        });
        let pipeline = |label, entry_point| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        ReactionPipelines {
            grid: pipeline("tessera MPM grid reactions", "collect_grid_reactions"),
            particle: pipeline(
                "tessera MPM particle reactions",
                "collect_particle_reactions",
            ),
            finalize: pipeline("tessera MPM reaction output", "finalize_obstacle_reactions"),
        }
    }

    pub(crate) fn transfer(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        particles: &[MpmParticle],
        params: &MpmParams,
        obstacles: &[RigidObstacle],
        dt: f64,
    ) -> Result<Vec<GpuTransfer>, GpuMpmError> {
        self.transfer_steps((device, queue), particles, params, obstacles, dt, 1)
    }

    fn transfer_with_reactions(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        particles: &[MpmParticle],
        params: &MpmParams,
        obstacles: &[RigidObstacle],
        dt: f64,
    ) -> Result<(Vec<GpuTransfer>, Vec<ObstacleReaction>), GpuMpmError> {
        let Some(prepared) = self.prepare(device, particles, params, obstacles, dt, false)? else {
            return Ok((
                Vec::new(),
                vec![ObstacleReaction::default(); obstacles.len()],
            ));
        };
        if obstacles.is_empty() {
            return Ok((self.run_prepared(device, queue, &prepared, 1)?, Vec::new()));
        }
        let reaction_pipelines = self.create_reaction_pipelines(device);
        let all =
            self.read_prepared(device, queue, &prepared, Some(1), Some(&reaction_pipelines))?;
        let particle_count = prepared.count as usize;
        let mut transfers = all;
        let reaction_records = transfers.split_off(particle_count);
        let reactions = reaction_records
            .iter()
            .map(|record| ObstacleReaction {
                linear: record.position(),
                angular: record.velocity(),
            })
            .collect();
        Ok((transfers, reactions))
    }

    fn transfer_steps(
        &self,
        gpu: (&wgpu::Device, &wgpu::Queue),
        particles: &[MpmParticle],
        params: &MpmParams,
        obstacles: &[RigidObstacle],
        dt: f64,
        steps: u32,
    ) -> Result<Vec<GpuTransfer>, GpuMpmError> {
        let (device, queue) = gpu;
        if steps == 0 || steps > 64 {
            return Err(GpuMpmError::Capacity);
        }
        let Some(prepared) = self.prepare(device, particles, params, obstacles, dt, steps > 1)?
        else {
            return Ok(vec![GpuTransfer::default(); particles.len()]);
        };
        self.run_prepared(device, queue, &prepared, steps)
    }

    fn prepare(
        &self,
        device: &wgpu::Device,
        particles: &[MpmParticle],
        params: &MpmParams,
        obstacles: &[RigidObstacle],
        dt: f64,
        resident: bool,
    ) -> Result<Option<PreparedTransfer>, GpuMpmError> {
        if particles.is_empty() {
            return Ok(None);
        }
        let count = u32::try_from(particles.len()).map_err(|_| GpuMpmError::Capacity)?;
        let limits = device.limits();
        let max_storage = u64::from(limits.max_storage_buffer_binding_size);
        if count.div_ceil(64) > limits.max_compute_workgroups_per_dimension {
            return Err(GpuMpmError::Capacity);
        }
        let reaction_capacity =
            u32::try_from(obstacles.len().max(16)).map_err(|_| GpuMpmError::Capacity)?;
        let scratch_nodes = u64::from(reaction_capacity)
            .checked_mul(2)
            .ok_or(GpuMpmError::Capacity)?;
        let max_nodes = (max_storage.min(limits.max_buffer_size) / 16)
            .checked_sub(scratch_nodes)
            .ok_or(GpuMpmError::Capacity)?;
        let mut inputs = Vec::with_capacity(particles.len());
        let h = finite_f32(params.cell_width)?;
        let step = finite_f32(dt)?;
        let inv_d = finite_f32(4.0 / (params.cell_width * params.cell_width))?;
        if h <= 0.0 || step <= 0.0 {
            return Err(GpuMpmError::InvalidInput);
        }
        let inv_h = h.recip();
        if !inv_h.is_finite() {
            return Err(GpuMpmError::InvalidInput);
        }
        for particle in particles {
            let position_mass = vec4(
                particle.position,
                if particle.enabled { particle.mass } else { 0.0 },
            )?;
            let velocity_volume = vec4(particle.velocity, particle.rest_volume)?;
            let volume = velocity_volume[3];
            if particle.enabled && (position_mass[3] <= 0.0 || volume <= 0.0) {
                return Err(GpuMpmError::InvalidInput);
            }
            let radius = finite_f32(particle.radius)?;
            if radius <= 0.0 || particle.damping < 0.0 {
                return Err(GpuMpmError::InvalidInput);
            }
            let mut base_cell = [0i32; 4];
            if particle.enabled {
                for axis in 0..3 {
                    let coordinate = (position_mass[axis] * inv_h - 0.5).floor();
                    if !coordinate.is_finite()
                        || f64::from(coordinate) < f64::from(i32::MIN)
                        || f64::from(coordinate) > f64::from(i32::MAX - 2)
                    {
                        return Err(GpuMpmError::InvalidInput);
                    }
                    base_cell[axis] = coordinate as i32;
                }
            }
            let (material, material_extra, projection, stress) = if particle.enabled {
                let use_cpu_corotated = match particle.material {
                    MaterialModel::LinearElastic { .. }
                    | MaterialModel::Sand { .. }
                    | MaterialModel::Snow { .. } => {
                        let deformation_norm = particle.deformation.norm();
                        let inverse_norm = particle
                            .deformation
                            .try_inverse()
                            .map_or(f64::INFINITY, |inverse| inverse.norm());
                        particle.deformation.determinant().abs() < 1e-5
                            || !(1e-5..=20.0).contains(&deformation_norm)
                            || deformation_norm * inverse_norm > 100.0
                    }
                    _ => false,
                };
                match particle.material {
                    MaterialModel::NeoHookean {
                        young_modulus,
                        poisson_ratio,
                    } => {
                        let (lambda, mu) = gpu_lame(young_modulus, poisson_ratio)?;
                        ([1.0, lambda, mu, 0.0], [0.0; 4], [0.0; 4], Matrix3::zeros())
                    }
                    MaterialModel::Fluid {
                        bulk_modulus,
                        gamma,
                        viscosity,
                        tensile_stiffness,
                    } => (
                        [
                            2.0,
                            finite_f32(bulk_modulus)?,
                            finite_f32(gamma)?,
                            finite_f32(viscosity)?,
                        ],
                        [finite_f32(tensile_stiffness)?, 0.0, 0.0, 0.0],
                        [0.0; 4],
                        Matrix3::zeros(),
                    ),
                    MaterialModel::LinearElastic {
                        young_modulus,
                        poisson_ratio,
                    } if !use_cpu_corotated => {
                        let (lambda, mu) = gpu_lame(young_modulus, poisson_ratio)?;
                        ([3.0, lambda, mu, 0.0], [0.0; 4], [0.0; 4], Matrix3::zeros())
                    }
                    MaterialModel::Sand {
                        young_modulus,
                        poisson_ratio,
                        friction_angle,
                        cohesion,
                    } if !use_cpu_corotated => {
                        let (lambda, mu) = gpu_lame(young_modulus, poisson_ratio)?;
                        (
                            [5.0, lambda, mu, 0.0],
                            [
                                finite_f32(particle.plastic.plastic_det)?,
                                finite_f32(particle.plastic.hardening)?,
                                finite_f32(particle.plastic.log_volume_gain)?,
                                finite_f32(friction_angle)?,
                            ],
                            [finite_f32(cohesion)?, 0.0, 0.0, 0.0],
                            Matrix3::zeros(),
                        )
                    }
                    MaterialModel::SandNeoHookean {
                        young_modulus,
                        poisson_ratio,
                        friction_angle,
                        cohesion,
                    } => {
                        let (lambda, mu) = gpu_lame(young_modulus, poisson_ratio)?;
                        (
                            [6.0, lambda, mu, 0.0],
                            [
                                finite_f32(particle.plastic.plastic_det)?,
                                finite_f32(particle.plastic.hardening)?,
                                finite_f32(particle.plastic.log_volume_gain)?,
                                finite_f32(friction_angle)?,
                            ],
                            [finite_f32(cohesion)?, 0.0, 0.0, 0.0],
                            Matrix3::zeros(),
                        )
                    }
                    MaterialModel::Snow {
                        young_modulus,
                        poisson_ratio,
                        critical_compression,
                        critical_stretch,
                        hardening,
                        ..
                    } if !use_cpu_corotated => {
                        let (lambda, mu) = gpu_lame(young_modulus, poisson_ratio)?;
                        (
                            [4.0, lambda, mu, finite_f32(hardening)?],
                            [
                                finite_f32(particle.plastic.plastic_det)?,
                                finite_f32(critical_compression)?,
                                finite_f32(critical_stretch)?,
                                0.0,
                            ],
                            [
                                finite_f32(particle.plastic.hardening)?,
                                finite_f32(particle.plastic.log_volume_gain)?,
                                0.0,
                                0.0,
                            ],
                            Matrix3::zeros(),
                        )
                    }
                    _ => (
                        [0.0; 4],
                        [0.0; 4],
                        [0.0; 4],
                        particle.material.kirchhoff_stress(
                            particle.deformation,
                            particle.affine,
                            particle.plastic,
                        ) * (-dt * particle.rest_volume * f64::from(inv_d)),
                    ),
                }
            } else {
                ([0.0; 4], [0.0; 4], [0.0; 4], Matrix3::zeros())
            };
            inputs.push(GpuParticle {
                position_mass,
                velocity_volume,
                force_damping: vec4(particle.force, particle.damping)?,
                radius: [
                    radius,
                    f32::from(u8::from(particle.fixed)),
                    f32::from_bits(0),
                    f32::from_bits(u32::from(particle.transfer_color)),
                ],
                affine: matrix_columns(particle.affine)?,
                stress: matrix_columns(stress)?,
                deformation: matrix_columns(particle.deformation)?,
                material,
                material_extra,
                projection,
                base_cell,
            });
        }
        let has_cpic = obstacles
            .iter()
            .any(|obstacle| obstacle.cpic_group.is_some());
        let topology = if resident {
            if inputs
                .iter()
                .any(|particle| particle.position_mass[3] > 0.0 && particle.material[0] == 0.0)
            {
                return Err(GpuMpmError::ResidentUnavailable);
            }
            let hashed = GridTopology::hashed(&inputs, max_nodes);
            let colored = has_cpic
                || inputs
                    .iter()
                    .any(|particle| particle.radius[3].to_bits() != 0);
            if colored {
                let hash = hashed?;
                if hash.dimensions[1].div_ceil(64) > limits.max_compute_workgroups_per_dimension {
                    return Err(GpuMpmError::Capacity);
                }
                hash
            } else if let Some(bounds) = params.bounds {
                // Dense grids require fixed bounds; sparse grids rebuild their coordinates each substep.
                let dense = GridTopology::bounded_dense(bounds, &inputs, inv_h, u64::MAX)?;
                match hashed {
                    Ok(hash)
                        if dense.nodes > hash.nodes
                            && hash.dimensions[1].div_ceil(64)
                                <= limits.max_compute_workgroups_per_dimension =>
                    {
                        hash
                    }
                    _ if dense.nodes <= max_nodes => dense,
                    _ => return Err(GpuMpmError::Capacity),
                }
            } else {
                let hash = hashed?;
                if hash.dimensions[1].div_ceil(64) > limits.max_compute_workgroups_per_dimension {
                    return Err(GpuMpmError::Capacity);
                }
                hash
            }
        } else if has_cpic {
            let hash = GridTopology::hashed(&inputs, max_nodes)?;
            if hash.dimensions[1].div_ceil(64) > limits.max_compute_workgroups_per_dimension
                || (inputs.len() as u64 * 27).div_ceil(64)
                    > u64::from(limits.max_compute_workgroups_per_dimension)
            {
                return Err(GpuMpmError::Capacity);
            }
            hash
        } else {
            GridTopology::gpu_generated(
                &inputs,
                max_nodes,
                max_storage.min(limits.max_buffer_size),
                limits.max_compute_workgroups_per_dimension,
            )?
        };
        if topology.nodes == 0 {
            return Ok(None);
        }
        let nodes = topology.nodes;
        let particle_bytes = size_of_val(inputs.as_slice()) as u64;
        let obstacle_count = i32::try_from(obstacles.len()).map_err(|_| GpuMpmError::Capacity)?;
        let mut convex_planes = Vec::new();
        let obstacle_inputs = obstacles
            .iter()
            .map(|obstacle| pack_obstacle(obstacle, &mut convex_planes))
            .collect::<Result<Vec<_>, _>>()?;
        let obstacle_buffer_contents = if obstacle_inputs.is_empty() {
            vec![GpuObstacle::default()]
        } else {
            obstacle_inputs
        };
        let obstacle_bytes = size_of_val(obstacle_buffer_contents.as_slice()) as u64;
        if convex_planes.is_empty() {
            convex_planes.push(GpuConvexPlane::default());
        }
        let convex_plane_bytes = size_of_val(convex_planes.as_slice()) as u64;
        let node_index_bytes = size_of_val(topology.particle_nodes.as_slice()) as u64;
        let node_coord_bytes = size_of_val(topology.node_coords.as_slice()) as u64;
        let grid_bytes = nodes
            .checked_add(scratch_nodes)
            .and_then(|count| count.checked_mul(16))
            .ok_or(GpuMpmError::Capacity)?;
        let output_slots = u64::from(count)
            .checked_add(u64::from(reaction_capacity))
            .ok_or(GpuMpmError::Capacity)?;
        let output_bytes = output_slots
            .checked_mul(size_of::<GpuTransfer>() as u64)
            .ok_or(GpuMpmError::Capacity)?;
        if [
            particle_bytes,
            obstacle_bytes,
            convex_plane_bytes,
            node_index_bytes,
            node_coord_bytes,
            grid_bytes,
            output_bytes,
        ]
        .iter()
        .any(|size| *size > max_storage || *size > limits.max_buffer_size)
            || count.div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || nodes.div_ceil(64) > u64::from(limits.max_compute_workgroups_per_dimension)
        {
            return Err(GpuMpmError::Capacity);
        }
        let (bound_min, bound_max) = if let Some(bounds) = params.bounds {
            (vec4(bounds.min, 0.0)?, vec4(bounds.max, 0.0)?)
        } else {
            ([0.0; 4], [0.0; 4])
        };
        if params.bounds.is_some() && (0..3).any(|axis| bound_min[axis] >= bound_max[axis]) {
            return Err(GpuMpmError::InvalidInput);
        }
        let config = GpuParams {
            origin: [
                topology.origin[0],
                topology.origin[1],
                topology.origin[2],
                obstacle_count,
            ],
            dimensions: [
                topology.dimensions[0],
                topology.dimensions[1],
                topology.dimensions[2],
                u32::from(params.bounds.is_some()),
            ],
            scalars: [h, inv_h, inv_d, step],
            gravity: vec4(params.gravity, f64::from(u8::from(resident)))?,
            bound_min,
            bound_max,
        };
        let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tessera MPM particles"),
            contents: bytemuck::cast_slice(&inputs),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let config_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tessera MPM grid parameters"),
            contents: bytemuck::bytes_of(&config),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let obstacle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tessera MPM rigid obstacles"),
            contents: bytemuck::cast_slice(&obstacle_buffer_contents),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let convex_plane_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tessera MPM convex planes"),
            contents: bytemuck::cast_slice(&convex_planes),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let node_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tessera MPM particle node indices"),
            contents: bytemuck::cast_slice(&topology.particle_nodes),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let node_coord_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tessera MPM compact node coordinates"),
            contents: bytemuck::cast_slice(&topology.node_coords),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let grid_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tessera MPM atomic grid"),
            size: grid_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | if cfg!(test) {
                    wgpu::BufferUsages::COPY_SRC
                } else {
                    wgpu::BufferUsages::empty()
                },
            mapped_at_creation: false,
        });
        let velocity_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tessera MPM grid velocity"),
            size: nodes * 16,
            usage: wgpu::BufferUsages::STORAGE
                | if cfg!(test) {
                    wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST
                } else {
                    wgpu::BufferUsages::empty()
                },
            mapped_at_creation: false,
        });
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tessera MPM G2P output"),
            size: output_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tessera MPM G2P readback"),
            size: output_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let node_dispatch_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tessera MPM live node count and dispatch"),
            contents: bytemuck::cast_slice(&[
                u32::try_from(nodes.div_ceil(64)).map_err(|_| GpuMpmError::Capacity)?,
                1,
                1,
                u32::try_from(nodes).map_err(|_| GpuMpmError::Capacity)?,
            ]),
            usage: wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let layout = self.p2g.get_bind_group_layout(0);
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tessera MPM transfer buffers"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grid_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: velocity_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: config_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: obstacle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: convex_plane_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: node_index_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: node_coord_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: node_dispatch_buffer.as_entire_binding(),
                },
            ],
        });
        let hash_topology = if topology.dimensions[0] == 0 && topology.dimensions[1] != 0 {
            Some(self.topology.prepare(
                device,
                &input_buffer,
                &node_index_buffer,
                &node_coord_buffer,
                &grid_buffer,
                &node_dispatch_buffer,
                count,
                (size_of::<GpuParticle>() / 16) as u32,
                (core::mem::offset_of!(GpuParticle, base_cell) / 16) as u32,
                topology.dimensions[1],
            )?)
        } else {
            None
        };
        Ok(Some(PreparedTransfer {
            topology: hash_topology,
            count,
            obstacle_count: obstacles.len() as u32,
            reaction_capacity,
            cpic_count: obstacles
                .iter()
                .filter(|obstacle| obstacle.cpic_group.is_some())
                .count() as u32,
            nodes,
            output_bytes,
            params: config,
            particle_buffer: input_buffer,
            grid_buffer,
            node_dispatch_buffer,
            velocity_buffer,
            output_buffer,
            readback,
            config_buffer,
            obstacle_buffer,
            convex_plane_buffer,
            node_index_buffer,
            node_coord_buffer,
            bindings,
        }))
    }

    fn replace_obstacles(
        &self,
        device: &wgpu::Device,
        prepared: &mut PreparedTransfer,
        obstacles: &[RigidObstacle],
    ) -> Result<(), GpuMpmError> {
        let count = i32::try_from(obstacles.len()).map_err(|_| GpuMpmError::Capacity)?;
        let mut planes = Vec::new();
        let mut packed = obstacles
            .iter()
            .map(|obstacle| pack_obstacle(obstacle, &mut planes))
            .collect::<Result<Vec<_>, _>>()?;
        if packed.is_empty() {
            packed.push(GpuObstacle::default());
        }
        if planes.is_empty() {
            planes.push(GpuConvexPlane::default());
        }
        let limits = device.limits();
        let max_storage =
            u64::from(limits.max_storage_buffer_binding_size).min(limits.max_buffer_size);
        if size_of_val(packed.as_slice()) as u64 > max_storage
            || size_of_val(planes.as_slice()) as u64 > max_storage
        {
            return Err(GpuMpmError::Capacity);
        }
        let mut params = prepared.params;
        params.origin[3] = count;
        let config_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tessera MPM updated grid parameters"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let obstacle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tessera MPM updated rigid obstacles"),
            contents: bytemuck::cast_slice(&packed),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let convex_plane_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tessera MPM updated convex planes"),
            contents: bytemuck::cast_slice(&planes),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tessera MPM refreshed obstacle bindings"),
            layout: &self.p2g.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: prepared.particle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: prepared.grid_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: prepared.velocity_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: prepared.output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: config_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: obstacle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: convex_plane_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: prepared.node_index_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: prepared.node_coord_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: prepared.node_dispatch_buffer.as_entire_binding(),
                },
            ],
        });
        prepared.params = params;
        prepared.obstacle_count = count as u32;
        prepared.cpic_count = obstacles
            .iter()
            .filter(|obstacle| obstacle.cpic_group.is_some())
            .count() as u32;
        prepared.config_buffer = config_buffer;
        prepared.obstacle_buffer = obstacle_buffer;
        prepared.convex_plane_buffer = convex_plane_buffer;
        prepared.bindings = bindings;
        Ok(())
    }

    fn run_prepared(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        prepared: &PreparedTransfer,
        steps: u32,
    ) -> Result<Vec<GpuTransfer>, GpuMpmError> {
        self.read_prepared(device, queue, prepared, Some(steps), None)
    }

    fn submit_prepared(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        prepared: &PreparedTransfer,
        steps: u32,
    ) -> Result<(), GpuMpmError> {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("tessera MPM transfer encoder"),
        });
        self.encode_prepared(&mut encoder, prepared, steps)?;
        let _submission = queue.submit(Some(encoder.finish()));
        Ok(())
    }

    fn encode_prepared(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        prepared: &PreparedTransfer,
        steps: u32,
    ) -> Result<(), GpuMpmError> {
        if steps == 0 || steps > 64 {
            return Err(GpuMpmError::Capacity);
        }
        for _ in 0..steps {
            if prepared.cpic_count > 0 {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("tessera MPM CPIC color update"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.update_cpic_colors);
                pass.set_bind_group(0, &prepared.bindings, &[]);
                pass.dispatch_workgroups(prepared.count.div_ceil(64), 1, 1);
            }
            if let Some(topology) = &prepared.topology {
                self.topology.encode(encoder, topology);
            } else {
                encoder.clear_buffer(&prepared.grid_buffer, 0, None);
            }
            for (pipeline, dispatch, indirect) in [
                (&self.p2g, prepared.count.div_ceil(64), None),
                (
                    &self.update_grid,
                    u32::try_from(prepared.nodes.div_ceil(64))
                        .map_err(|_| GpuMpmError::Capacity)?,
                    prepared.topology.as_ref().map(HashTopologyState::indirect),
                ),
                (&self.g2p, prepared.count.div_ceil(64), None),
            ] {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("tessera MPM transfer stage"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &prepared.bindings, &[]);
                if let Some(buffer) = indirect {
                    pass.dispatch_workgroups_indirect(buffer, 0);
                } else {
                    pass.dispatch_workgroups(dispatch, 1, 1);
                }
            }
        }
        Ok(())
    }

    fn encode_reactions(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        prepared: &PreparedTransfer,
        pipelines: &ReactionPipelines,
    ) -> Result<(), GpuMpmError> {
        if prepared.obstacle_count == 0 {
            return Ok(());
        }
        encoder.clear_buffer(
            &prepared.grid_buffer,
            prepared.nodes * 16,
            Some(u64::from(prepared.obstacle_count) * 32),
        );
        for (pipeline, dispatch, indirect) in [
            (
                &pipelines.grid,
                u32::try_from(prepared.nodes.div_ceil(64)).map_err(|_| GpuMpmError::Capacity)?,
                prepared.topology.as_ref().map(HashTopologyState::indirect),
            ),
            (&pipelines.particle, prepared.count.div_ceil(64), None),
            (
                &pipelines.finalize,
                prepared.obstacle_count.div_ceil(64),
                None,
            ),
        ] {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tessera MPM obstacle reaction stage"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &prepared.bindings, &[]);
            if let Some(buffer) = indirect {
                pass.dispatch_workgroups_indirect(buffer, 0);
            } else {
                pass.dispatch_workgroups(dispatch, 1, 1);
            }
        }
        Ok(())
    }

    fn read_prepared(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        prepared: &PreparedTransfer,
        steps: Option<u32>,
        reaction_pipelines: Option<&ReactionPipelines>,
    ) -> Result<Vec<GpuTransfer>, GpuMpmError> {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("tessera MPM synchronization encoder"),
        });
        if let Some(steps) = steps {
            self.encode_prepared(&mut encoder, prepared, steps)?;
        }
        if let Some(pipelines) = reaction_pipelines {
            self.encode_reactions(&mut encoder, prepared, pipelines)?;
        }
        encoder.copy_buffer_to_buffer(
            &prepared.output_buffer,
            0,
            &prepared.readback,
            0,
            prepared.output_bytes,
        );
        let _submission = queue.submit(Some(encoder.finish()));
        let (sender, receiver) = mpsc::channel();
        prepared
            .readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        let _status = device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(5)),
            })
            .map_err(|error| GpuMpmError::Readback(error.to_string()))?;
        receiver
            .recv_timeout(Duration::from_secs(5))
            .map_err(|error| GpuMpmError::Readback(error.to_string()))?
            .map_err(|error| GpuMpmError::Readback(error.to_string()))?;
        Self::collect_prepared(prepared, reaction_pipelines.is_some())
    }

    async fn read_prepared_async(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        prepared: &PreparedTransfer,
    ) -> Result<Vec<GpuTransfer>, GpuMpmError> {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(
            &prepared.output_buffer,
            0,
            &prepared.readback,
            0,
            prepared.output_bytes,
        );
        let _submission = queue.submit(Some(encoder.finish()));
        // Unmapping on cancellation aborts outstanding maps and permits a retry.
        struct MapGuard(Option<wgpu::Buffer>);
        impl Drop for MapGuard {
            fn drop(&mut self) {
                if let Some(buffer) = self.0.take() {
                    buffer.unmap();
                }
            }
        }
        let mut guard = MapGuard(Some(prepared.readback.clone()));
        let (sender, receiver) = futures_channel::oneshot::channel::<Result<(), (String, bool)>>();
        let completion = std::sync::Arc::new(std::sync::Mutex::new(Some(sender)));
        let callback_completion = completion.clone();
        prepared
            .readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                if let Ok(mut sender) = callback_completion.lock()
                    && let Some(sender) = sender.take()
                {
                    let _sent = sender.send(result.map_err(|error| (error.to_string(), false)));
                }
            });
        #[cfg(not(target_arch = "wasm32"))]
        {
            let device = device.clone();
            let _worker = std::thread::Builder::new()
                .name("tessera-mpm-readback".into())
                .spawn(move || {
                    if let Err(error) = device.poll(wgpu::PollType::Wait {
                        submission_index: None,
                        timeout: Some(Duration::from_secs(5)),
                    }) && let Ok(mut sender) = completion.lock()
                        && let Some(sender) = sender.take()
                    {
                        let _sent = sender.send(Err((error.to_string(), true)));
                    }
                })
                .map_err(|error| GpuMpmError::Readback(error.to_string()))?;
        }
        match receiver
            .await
            .map_err(|error| GpuMpmError::Readback(error.to_string()))?
        {
            Ok(()) => {
                // collect_prepared releases the successful mapping itself.
                guard.0 = None;
                Self::collect_prepared(prepared, false)
            }
            Err((error, still_pending)) => {
                // Map failures already release mapping state; poll timeouts do not.
                if !still_pending {
                    guard.0 = None;
                }
                Err(GpuMpmError::Readback(error))
            }
        }
    }

    fn collect_prepared(
        prepared: &PreparedTransfer,
        with_reactions: bool,
    ) -> Result<Vec<GpuTransfer>, GpuMpmError> {
        let view = prepared.readback.slice(..).get_mapped_range();
        let mut transfers = bytemuck::cast_slice::<u8, GpuTransfer>(&view).to_vec();
        drop(view);
        prepared.readback.unmap();
        if transfers[..prepared.count as usize]
            .iter()
            .any(|transfer| transfer.plastic[3] <= -2.0)
        {
            return Err(GpuMpmError::Capacity);
        }
        if transfers[..prepared.count as usize]
            .iter()
            .any(|transfer| transfer.plastic[3] < 0.0)
        {
            return Err(GpuMpmError::ResidentUnavailable);
        }
        if transfers
            .iter()
            .take(if with_reactions {
                transfers.len()
            } else {
                prepared.count as usize
            })
            .any(|transfer| {
                !transfer.position.iter().all(|value| value.is_finite())
                    || !transfer.velocity.iter().all(|value| value.is_finite())
                    || transfer
                        .affine
                        .iter()
                        .flatten()
                        .any(|value| !value.is_finite())
                    || transfer
                        .deformation
                        .iter()
                        .flatten()
                        .any(|value| !value.is_finite())
                    || transfer.plastic.iter().any(|value| !value.is_finite())
            })
        {
            return Err(GpuMpmError::Reference(MpmError::NonFiniteState));
        }
        if !with_reactions {
            transfers.truncate(prepared.count as usize);
        }
        Ok(transfers)
    }
}

fn finite_f32(value: f64) -> Result<f32, GpuMpmError> {
    let converted = value as f32;
    if converted.is_finite() {
        Ok(converted)
    } else {
        Err(GpuMpmError::InvalidInput)
    }
}

fn gpu_lame(young: f64, poisson: f64) -> Result<(f32, f32), GpuMpmError> {
    Ok((
        finite_f32(young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson)))?,
        finite_f32(young / (2.0 * (1.0 + poisson)))?,
    ))
}

fn vec4(value: Vector3<f64>, fourth: f64) -> Result<[f32; 4], GpuMpmError> {
    Ok([
        finite_f32(value.x)?,
        finite_f32(value.y)?,
        finite_f32(value.z)?,
        finite_f32(fourth)?,
    ])
}

fn matrix_columns(value: Matrix3<f64>) -> Result<[[f32; 4]; 3], GpuMpmError> {
    Ok([
        vec4(value.column(0).into_owned(), 0.0)?,
        vec4(value.column(1).into_owned(), 0.0)?,
        vec4(value.column(2).into_owned(), 0.0)?,
    ])
}

fn pack_obstacle(
    obstacle: &RigidObstacle,
    convex_planes: &mut Vec<GpuConvexPlane>,
) -> Result<GpuObstacle, GpuMpmError> {
    if !obstacle.is_valid() {
        return Err(GpuMpmError::InvalidInput);
    }
    let mut convex_range = [0u32; 4];
    let (radius, half_extents_kind, triangle_vertices) = match &obstacle.shape {
        ObstacleShape::Sphere { radius } => (*radius, [0.0; 4], [[0.0; 4]; 3]),
        ObstacleShape::Box { half_extents } => (0.0, vec4(*half_extents, 1.0)?, [[0.0; 4]; 3]),
        ObstacleShape::Capsule {
            half_height,
            radius,
        } => (
            *radius,
            [finite_f32(*half_height)?, 0.0, 0.0, 2.0],
            [[0.0; 4]; 3],
        ),
        ObstacleShape::Cylinder {
            half_height,
            radius,
        } => (
            *radius,
            [finite_f32(*half_height)?, 0.0, 0.0, 3.0],
            [[0.0; 4]; 3],
        ),
        ObstacleShape::Cone {
            half_height,
            radius,
        } => (
            *radius,
            [finite_f32(*half_height)?, 0.0, 0.0, 4.0],
            [[0.0; 4]; 3],
        ),
        ObstacleShape::Ground { half_extents } => (
            0.0,
            [
                finite_f32(half_extents.x)?,
                finite_f32(half_extents.y)?,
                0.0,
                5.0,
            ],
            [[0.0; 4]; 3],
        ),
        ObstacleShape::TrianglePrism {
            vertices,
            half_thickness,
        } => {
            let packed_vertices = [
                vec4(vertices[0], 0.0)?,
                vec4(vertices[1], 0.0)?,
                vec4(vertices[2], 0.0)?,
            ];
            let a = Vector3::new(
                packed_vertices[0][0],
                packed_vertices[0][1],
                packed_vertices[0][2],
            );
            let b = Vector3::new(
                packed_vertices[1][0],
                packed_vertices[1][1],
                packed_vertices[1][2],
            );
            let c = Vector3::new(
                packed_vertices[2][0],
                packed_vertices[2][1],
                packed_vertices[2][2],
            );
            let thickness = finite_f32(*half_thickness)?;
            let area_squared = (b - a).cross(&(c - a)).norm_squared();
            if thickness <= 0.0 || !area_squared.is_finite() || area_squared <= 0.0 {
                return Err(GpuMpmError::InvalidInput);
            }
            let bound = (vertices.iter().map(Vector3::norm).fold(0.0f64, f64::max)
                + half_thickness)
                * (1.0 + 1e-5);
            (bound, [thickness, 0.0, 0.0, 6.0], packed_vertices)
        }
        ObstacleShape::Convex {
            planes,
            bound_radius,
        } => {
            if planes.len() > 128 {
                return Err(GpuMpmError::Capacity);
            }
            let start = u32::try_from(convex_planes.len()).map_err(|_| GpuMpmError::Capacity)?;
            let count = u32::try_from(planes.len()).map_err(|_| GpuMpmError::Capacity)?;
            convex_planes
                .try_reserve(planes.len())
                .map_err(|_| GpuMpmError::Capacity)?;
            for plane in planes.iter() {
                let packed = [
                    finite_f32(plane[0])?,
                    finite_f32(plane[1])?,
                    finite_f32(plane[2])?,
                    finite_f32(plane[3])?,
                ];
                let norm_squared = packed[..3].iter().map(|value| value * value).sum::<f32>();
                if !norm_squared.is_finite() || norm_squared <= 0.0 {
                    return Err(GpuMpmError::InvalidInput);
                }
                convex_planes.push(GpuConvexPlane {
                    normal_offset: packed,
                });
            }
            convex_range = [start, count, 0, 0];
            (
                bound_radius * (1.0 + 1e-5),
                [0.0, 0.0, 0.0, 7.0],
                [[0.0; 4]; 3],
            )
        }
    };
    let q = obstacle.orientation.quaternion().coords;
    convex_range[2] = obstacle.cpic_group.map_or(0, |group| u32::from(group) + 1);
    Ok(GpuObstacle {
        center_radius: vec4(obstacle.center, radius)?,
        half_extents_kind,
        orientation: [
            finite_f32(q[0])?,
            finite_f32(q[1])?,
            finite_f32(q[2])?,
            finite_f32(q[3])?,
        ],
        linear_velocity_friction: vec4(obstacle.linear_velocity, obstacle.friction)?,
        angular_velocity: vec4(
            obstacle.angular_velocity,
            match obstacle.boundary {
                crate::ObstacleBoundary::Slip => 0.0,
                crate::ObstacleBoundary::Stick => 1.0,
                crate::ObstacleBoundary::Separate => 2.0,
                crate::ObstacleBoundary::NonReflecting => 3.0,
            },
        )?,
        triangle_a: triangle_vertices[0],
        triangle_b: triangle_vertices[1],
        triangle_c: triangle_vertices[2],
        convex_range,
    })
}

impl MpmWorld {
    /// Advance an MPM world with one GPU submission.
    ///
    /// The caller chooses the fixed substep size. At most 64 substeps are
    /// recorded; the initial CFL bound is checked, but the caller must keep
    /// later velocity changes within its chosen stability margin. The GPU
    /// checks the CFL bound after each substep. If it is exceeded, a particle
    /// leaves the grid, or CPU material projection is needed, this returns an
    /// error without applying the GPU result to the world.
    pub fn step_gpu_resident_fixed_substeps(
        &mut self,
        gpu: &GpuMpmTransfers,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        substep_dt: f64,
        steps: u32,
    ) -> Result<(), GpuMpmError> {
        if steps == 0 || steps > 64 {
            return Err(GpuMpmError::InvalidInput);
        }
        validate_resident_world(self, substep_dt)?;
        let transfers = gpu.transfer_steps(
            (device, queue),
            &self.particles,
            &self.params,
            &self.obstacles,
            substep_dt,
            steps,
        )?;
        self.apply_gpu_transfers(transfers)?;
        self.substeps = self.substeps.saturating_add(u64::from(steps));
        Ok(())
    }

    /// Advance with WebGPU P2G, grid, G2P, particle integration, and projection stages.
    ///
    /// Material stress is evaluated in P2G except for ill-conditioned
    /// corotated deformations. G2P integrates deformation and projects fluid,
    /// sand, and snow. It also applies forces, damping, bounds, and rigid
    /// obstacle projection; ill-conditioned plastic states fall back to the CPU.
    pub fn step_with_gpu_transfers(
        &mut self,
        gpu: &GpuMpmTransfers,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        dt: f64,
    ) -> Result<(), GpuMpmError> {
        if !dt.is_finite() || dt <= 0.0 || self.obstacles.iter().any(|o| !o.is_valid()) {
            return Err(GpuMpmError::Reference(MpmError::InvalidInput));
        }
        let mut remaining = dt;
        let mut count = 0usize;
        while remaining > 0.0 {
            let stable = self.stable_timestep();
            if !stable.is_finite() || stable <= 1e-12 || count >= 100_000 {
                return Err(GpuMpmError::Reference(MpmError::ExcessiveSubsteps));
            }
            let substep = remaining.min(stable);
            let transfers = gpu.transfer(
                device,
                queue,
                &self.particles,
                &self.params,
                &self.obstacles,
                substep,
            )?;
            self.apply_gpu_transfers(transfers)?;
            self.substeps = self.substeps.saturating_add(1);
            remaining = (remaining - substep).max(0.0);
            count += 1;
        }
        Ok(())
    }

    /// Advance GPU MPM while collecting obstacle reaction impulses on the GPU.
    ///
    /// Each substep reads particle transfers and the per-obstacle reductions
    /// back to the CPU. Device buffer and dispatch limits are reported as
    /// `Capacity` errors.
    pub fn step_with_gpu_obstacle_reactions(
        &mut self,
        gpu: &GpuMpmTransfers,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        dt: f64,
    ) -> Result<Vec<ObstacleReaction>, GpuMpmError> {
        if !dt.is_finite() || dt <= 0.0 || self.obstacles.iter().any(|o| !o.is_valid()) {
            return Err(GpuMpmError::Reference(MpmError::InvalidInput));
        }
        let mut reactions = vec![ObstacleReaction::default(); self.obstacles.len()];
        let mut remaining = dt;
        let mut count = 0usize;
        while remaining > 0.0 {
            let stable = self.stable_timestep();
            if !stable.is_finite() || stable <= 1e-12 || count >= 100_000 {
                return Err(GpuMpmError::Reference(MpmError::ExcessiveSubsteps));
            }
            let substep = remaining.min(stable);
            let (transfers, next_reactions) = gpu.transfer_with_reactions(
                device,
                queue,
                &self.particles,
                &self.params,
                &self.obstacles,
                substep,
            )?;
            self.apply_gpu_transfers(transfers)?;
            for (reaction, next) in reactions.iter_mut().zip(next_reactions) {
                reaction.linear += next.linear;
                reaction.angular += next.angular;
            }
            self.substeps = self.substeps.saturating_add(1);
            remaining = (remaining - substep).max(0.0);
            count += 1;
        }
        if reactions.iter().any(|reaction| {
            reaction
                .linear
                .iter()
                .chain(reaction.angular.iter())
                .any(|component| !component.is_finite())
        }) {
            return Err(GpuMpmError::Reference(MpmError::NonFiniteState));
        }
        Ok(reactions)
    }

    fn apply_gpu_transfers(&mut self, transfers: Vec<GpuTransfer>) -> Result<(), GpuMpmError> {
        for (particle, transfer) in self.particles.iter_mut().zip(transfers) {
            if !particle.enabled {
                particle.force = Vector3::zeros();
                continue;
            }
            if particle.fixed {
                particle.velocity = Vector3::zeros();
                particle.affine = Matrix3::zeros();
                particle.force = Vector3::zeros();
                continue;
            }
            particle.position = transfer.position();
            particle.velocity = transfer.velocity();
            particle.affine = transfer.affine();
            particle.deformation = transfer.deformation();
            if transfer.plastic[3] == 1.0 {
                particle.plastic = transfer.plastic();
            } else if matches!(
                particle.material,
                MaterialModel::Sand { .. }
                    | MaterialModel::SandNeoHookean { .. }
                    | MaterialModel::Snow { .. }
            ) {
                (particle.deformation, particle.plastic) = particle
                    .material
                    .project_deformation(particle.deformation, particle.plastic);
            }
            particle.force = Vector3::zeros();
            if !crate::world::finite_particle_state(particle) {
                return Err(GpuMpmError::Reference(MpmError::NonFiniteState));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BoxEmitter, MaterialModel, MeshEmitter, WorldBounds};
    use nalgebra::UnitQuaternion;

    #[tokio::test]
    async fn sand_neo_hookean_gpu_matches_cpu_direct_and_resident() {
        #[cfg(target_os = "windows")]
        let backends = [wgpu::Backends::VULKAN, wgpu::Backends::DX12];
        #[cfg(not(target_os = "windows"))]
        let backends = [wgpu::Backends::PRIMARY];
        for backend in backends {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: backend,
                ..Default::default()
            });
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
                .expect("SandNeoHookean parity requires a real WebGPU adapter");
            eprintln!("SandNeoHookean adapter: {:?}", adapter.get_info());
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor::default())
                .await
                .unwrap();
            let rotation = UnitQuaternion::from_euler_angles(0.2, -0.4, 0.7)
                .to_rotation_matrix()
                .into_inner();
            let deformations = [
                Matrix3::identity(),
                rotation,
                Matrix3::identity() * 1.1,
                Matrix3::from_diagonal(&Vector3::new(0.8, 0.79, 0.81)),
                rotation
                    * Matrix3::from_diagonal(&Vector3::new(
                        0.15f64.exp(),
                        (-0.25f64).exp(),
                        (-0.1f64).exp(),
                    )),
            ];
            let particles = deformations
                .into_iter()
                .enumerate()
                .map(|(index, deformation)| {
                    let mut particle = MpmParticle::new(
                        Vector3::new(
                            0.5 + f64::from(u32::try_from(index).unwrap()) * 0.4,
                            0.5,
                            0.5,
                        ),
                        0.04,
                        1_000.0,
                        MaterialModel::sand_neo_hookean(4_000.0, 0.2, 35.0f64.to_radians(), 0.02),
                    );
                    particle.deformation = deformation;
                    particle.plastic.hardening += f64::from(u32::try_from(index).unwrap()) * 0.2;
                    particle
                })
                .collect::<Vec<_>>();
            let params = MpmParams {
                gravity: Vector3::zeros(),
                bounds: Some(WorldBounds {
                    min: Vector3::zeros(),
                    max: Vector3::repeat(3.0),
                }),
                ..MpmParams::default()
            };
            let pipeline = GpuMpmTransfers::new(&device);
            for resident in [false, true] {
                let mut cpu = MpmWorld::new(particles.clone(), params.clone()).unwrap();
                let mut gpu = cpu.clone();
                for _ in 0..4 {
                    cpu.step(0.0001).unwrap();
                }
                if resident {
                    gpu.step_gpu_resident_fixed_substeps(&pipeline, &device, &queue, 0.0001, 4)
                        .unwrap();
                } else {
                    for _ in 0..4 {
                        gpu.step_with_gpu_transfers(&pipeline, &device, &queue, 0.0001)
                            .unwrap();
                    }
                }
                assert_eq!(cpu.substeps, gpu.substeps);
                for (index, (expected, actual)) in
                    cpu.particles.iter().zip(&gpu.particles).enumerate()
                {
                    assert!((expected.position - actual.position).norm() < 2e-5);
                    assert!((expected.velocity - actual.velocity).norm() < 3e-4);
                    assert!(
                        (expected.affine - actual.affine).norm() < 2e-3,
                        "particle={index} resident={resident} affine error={} cpu={:?} gpu={:?} deformation error={} plastic cpu={:?} gpu={:?}",
                        (expected.affine - actual.affine).norm(),
                        expected.affine,
                        actual.affine,
                        (expected.deformation - actual.deformation).norm(),
                        expected.plastic,
                        actual.plastic,
                    );
                    assert!((expected.deformation - actual.deformation).norm() < 2e-4);
                    assert!(
                        (expected.plastic.plastic_det - actual.plastic.plastic_det).abs() < 2e-4
                    );
                    assert!((expected.plastic.hardening - actual.plastic.hardening).abs() < 2e-4);
                    assert!(
                        (expected.plastic.log_volume_gain - actual.plastic.log_volume_gain).abs()
                            < 2e-4
                    );
                }
                assert!(gpu.particles[2].plastic.hardening > particles[2].plastic.hardening);
                assert!(gpu.particles[4].plastic.hardening > particles[4].plastic.hardening);
            }
        }
    }

    #[tokio::test]
    async fn gpu_obstacle_reactions_match_cpu_momentum_transfer() {
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
        let mut particle = MpmParticle::new(
            Vector3::new(1.02, 0.03, 0.0),
            0.04,
            1000.0,
            MaterialModel::elastic(1000.0, 0.2),
        );
        particle.velocity.x = -1.0;
        particle.velocity.y = 1.0;
        let params = MpmParams {
            gravity: Vector3::zeros(),
            ..MpmParams::default()
        };
        let mut cpu = MpmWorld::new(vec![particle.clone()], params.clone()).unwrap();
        let mut gpu = MpmWorld::new(vec![particle], params).unwrap();
        let mut contacting = RigidObstacle::sphere(Vector3::zeros(), 1.0);
        contacting.friction = 0.5;
        let obstacles = vec![
            contacting,
            RigidObstacle::sphere(Vector3::new(10.0, 0.0, 0.0), 1.0),
        ];
        cpu.set_obstacles(obstacles.clone()).unwrap();
        gpu.set_obstacles(obstacles).unwrap();
        let expected = cpu.step_with_obstacle_reactions(0.001).unwrap();
        let transfers = GpuMpmTransfers::new(&device);
        let actual = gpu
            .step_with_gpu_obstacle_reactions(&transfers, &device, &queue, 0.001)
            .unwrap();
        assert_eq!(actual.len(), 2);
        assert!((actual[0].linear - expected[0].linear).norm() < 1e-4);
        assert!((actual[0].angular - expected[0].angular).norm() < 1e-4);
        assert!(actual[0].angular.z.abs() > 1e-3);
        assert!(actual[1].linear.norm() < 1e-6);
        assert!(actual[1].angular.norm() < 1e-6);
        assert!((gpu.particles[0].velocity - cpu.particles[0].velocity).norm() < 1e-4);
    }

    #[tokio::test]
    async fn gpu_reactions_support_more_than_sixteen_obstacles() {
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
        let mut particle = MpmParticle::new(
            Vector3::new(1.02, 0.0, 0.0),
            0.04,
            1000.0,
            MaterialModel::elastic(1000.0, 0.2),
        );
        particle.velocity.x = -1.0;
        let params = MpmParams {
            gravity: Vector3::zeros(),
            ..MpmParams::default()
        };
        let mut cpu = MpmWorld::new(vec![particle.clone()], params.clone()).unwrap();
        let mut gpu = MpmWorld::new(vec![particle], params).unwrap();
        let mut obstacles = (0..16)
            .map(|index| {
                RigidObstacle::sphere(Vector3::new(10.0 + index as f64 * 3.0, 0.0, 0.0), 1.0)
            })
            .collect::<Vec<_>>();
        obstacles.push(RigidObstacle::sphere(Vector3::zeros(), 1.0));
        cpu.set_obstacles(obstacles.clone()).unwrap();
        gpu.set_obstacles(obstacles).unwrap();
        let expected = cpu.step_with_obstacle_reactions(0.001).unwrap();
        let transfers = GpuMpmTransfers::new(&device);
        let actual = gpu
            .step_with_gpu_obstacle_reactions(&transfers, &device, &queue, 0.001)
            .unwrap();
        assert_eq!(actual.len(), 17);
        assert!(
            actual[..16]
                .iter()
                .all(|reaction| reaction.linear.norm() < 1e-6)
        );
        assert!(actual[16].linear.x < -0.01, "{:?}", actual[16]);
        assert!((actual[16].linear - expected[16].linear).norm() < 1e-4);

        let initial = MpmWorld::new(
            vec![MpmParticle::new(
                Vector3::new(1.02, 0.0, 0.0),
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
        let mut session =
            GpuMpmResidentSession::new(&transfers, &device, &queue, initial, 0.001).unwrap();
        assert_eq!(session.prepared.as_ref().unwrap().reaction_capacity, 16);
        session.set_obstacles(cpu.obstacles.clone()).unwrap();
        assert_eq!(session.prepared.as_ref().unwrap().reaction_capacity, 17);
        let mut encoder = device.create_command_encoder(&Default::default());
        let next = session
            .encode_one_substep_with_reactions(&mut encoder)
            .unwrap();
        let _submission = queue.submit(Some(encoder.finish()));
        session.mark_one_substep_submitted(next);
        session.synchronize().unwrap();
        assert_eq!(session.world().substeps, 1);
    }

    #[tokio::test]
    async fn gpu_reactions_preserve_ordered_overlap_attribution() {
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
        let mut particle = MpmParticle::new(
            Vector3::new(1.02, 0.0, 0.0),
            0.04,
            1000.0,
            MaterialModel::elastic(1000.0, 0.2),
        );
        particle.velocity.x = -1.0;
        let params = MpmParams {
            gravity: Vector3::zeros(),
            ..MpmParams::default()
        };
        let mut cpu = MpmWorld::new(vec![particle.clone()], params.clone()).unwrap();
        let mut gpu = MpmWorld::new(vec![particle], params).unwrap();
        let first = RigidObstacle::sphere(Vector3::zeros(), 1.0);
        let mut second = RigidObstacle::sphere(Vector3::new(0.1, 0.0, 0.0), 1.0);
        second.linear_velocity.x = 1.0;
        cpu.set_obstacles(vec![first.clone(), second.clone()])
            .unwrap();
        gpu.set_obstacles(vec![first, second]).unwrap();
        let expected = cpu.step_with_obstacle_reactions(0.001).unwrap();
        let actual = gpu
            .step_with_gpu_obstacle_reactions(
                &GpuMpmTransfers::new(&device),
                &device,
                &queue,
                0.001,
            )
            .unwrap();
        assert!(expected[0].linear.x < 0.0);
        assert!(expected[1].linear.x < 0.0);
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual.linear - expected.linear).norm() < 1e-4);
            assert!((actual.angular - expected.angular).norm() < 1e-4);
        }
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn dx12_obstacle_reactions_match_cpu() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::DX12,
            ..Default::default()
        });
        let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
        else {
            eprintln!("DX12 adapter unavailable for MPM reaction test");
            return;
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let mut particle = MpmParticle::new(
            Vector3::new(1.02, 0.0, 0.0),
            0.04,
            1000.0,
            MaterialModel::elastic(1000.0, 0.2),
        );
        particle.velocity.x = -1.0;
        let params = MpmParams {
            gravity: Vector3::zeros(),
            ..MpmParams::default()
        };
        let mut cpu = MpmWorld::new(vec![particle.clone()], params.clone()).unwrap();
        let mut gpu = MpmWorld::new(vec![particle], params).unwrap();
        let obstacle = RigidObstacle::sphere(Vector3::zeros(), 1.0);
        cpu.set_obstacles(vec![obstacle.clone()]).unwrap();
        gpu.set_obstacles(vec![obstacle]).unwrap();
        let expected = cpu.step_with_obstacle_reactions(0.001).unwrap();
        let actual = gpu
            .step_with_gpu_obstacle_reactions(
                &GpuMpmTransfers::new(&device),
                &device,
                &queue,
                0.001,
            )
            .unwrap();
        assert!((actual[0].linear - expected[0].linear).norm() < 1e-4);
    }

    #[tokio::test]
    async fn sparse_grid_obstacle_reactions_match_cpu() {
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
        let mut impactor = MpmParticle::new(
            Vector3::new(1.02, 0.0, 0.0),
            0.04,
            1000.0,
            MaterialModel::elastic(1000.0, 0.2),
        );
        impactor.velocity.x = -1.0;
        let distant = MpmParticle::new(
            Vector3::new(100.0, 0.0, 0.0),
            0.04,
            1000.0,
            MaterialModel::elastic(1000.0, 0.2),
        );
        let params = MpmParams {
            gravity: Vector3::zeros(),
            ..MpmParams::default()
        };
        let mut cpu =
            MpmWorld::new(vec![impactor.clone(), distant.clone()], params.clone()).unwrap();
        let mut gpu = MpmWorld::new(vec![impactor, distant], params).unwrap();
        let obstacle = RigidObstacle::sphere(Vector3::zeros(), 1.0);
        cpu.set_obstacles(vec![obstacle.clone()]).unwrap();
        gpu.set_obstacles(vec![obstacle]).unwrap();
        let expected = cpu.step_with_obstacle_reactions(0.001).unwrap();
        let actual = gpu
            .step_with_gpu_obstacle_reactions(
                &GpuMpmTransfers::new(&device),
                &device,
                &queue,
                0.001,
            )
            .unwrap();
        assert!((actual[0].linear - expected[0].linear).norm() < 1e-4);
        assert!((actual[0].angular - expected[0].angular).norm() < 1e-4);
        assert!((gpu.particles[1].velocity - cpu.particles[1].velocity).norm() < 1e-4);
    }

    #[tokio::test]
    async fn topology_exhaustion_rejects_incomplete_transfers_and_retains_world() {
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
        let particles = [0.1, 0.8]
            .into_iter()
            .map(|x| {
                MpmParticle::new(
                    Vector3::repeat(x),
                    0.03,
                    1_000.0,
                    MaterialModel::fluid(2_000.0, 7.0, 0.1),
                )
            })
            .collect();
        let world = MpmWorld::new(
            particles,
            MpmParams {
                bounds: Some(WorldBounds {
                    min: Vector3::zeros(),
                    max: Vector3::repeat(1.0),
                }),
                ..MpmParams::default()
            },
        )
        .unwrap();
        let pipeline = GpuMpmTransfers::new(&device);
        let exhaust = |prepared: &mut PreparedTransfer| {
            // Inject a single bucket for distinct stencil keys to exercise GPU failure.
            prepared.topology = Some(
                pipeline
                    .topology
                    .prepare(
                        &device,
                        &prepared.particle_buffer,
                        &prepared.node_index_buffer,
                        &prepared.node_coord_buffer,
                        &prepared.grid_buffer,
                        &prepared.node_dispatch_buffer,
                        prepared.count,
                        (size_of::<GpuParticle>() / 16) as u32,
                        (core::mem::offset_of!(GpuParticle, base_cell) / 16) as u32,
                        1,
                    )
                    .unwrap(),
            );
        };
        let mut prepared = pipeline
            .prepare(&device, &world.particles, &world.params, &[], 0.0001, false)
            .unwrap()
            .unwrap();
        exhaust(&mut prepared);
        assert!(matches!(
            pipeline.run_prepared(&device, &queue, &prepared, 1),
            Err(GpuMpmError::Capacity)
        ));
        let mut session =
            GpuMpmResidentSession::new(&pipeline, &device, &queue, world.clone(), 0.0001).unwrap();
        let original_topology = session.prepared.as_mut().unwrap().topology.take();
        assert!(original_topology.is_some());
        exhaust(session.prepared.as_mut().unwrap());
        session.submit_steps(4).unwrap();
        session.prepared.as_mut().unwrap().topology = original_topology;
        session.submit_steps(2).unwrap();
        assert_eq!(session.pending_substeps(), 6);
        assert!(session.has_pending_steps());
        assert!(matches!(
            session.synchronize_async().await,
            Err(GpuMpmError::Capacity)
        ));
        assert!(!session.has_pending_steps());
        assert_eq!(session.world().substeps, 0);
        for (before, after) in world.particles.iter().zip(&session.world().particles) {
            assert_eq!(before.position, after.position);
            assert_eq!(before.velocity, after.velocity);
            assert_eq!(before.deformation, after.deformation);
        }
        assert!(matches!(
            session.step(1),
            Err(GpuMpmError::ResidentUnavailable)
        ));
    }

    #[tokio::test]
    async fn gpu_snapshot_matches_state_after_steps_and_chunk_edits() {
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: backend,
                ..Default::default()
            });
            let Ok(adapter) = instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
            else {
                continue;
            };
            if adapter.get_info().device_type == wgpu::DeviceType::Cpu {
                continue;
            }
            eprintln!("Snapshot adapter: {:?}", adapter.get_info());
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor::default())
                .await
                .unwrap();
            let mut particle = MpmParticle::new(
                Vector3::new(0.05, 0.2, 0.3),
                0.03,
                1_000.0,
                MaterialModel::fluid(2_000.0, 7.0, 0.1),
            );
            particle.velocity = Vector3::new(0.1, -0.2, 0.3);
            let mut disabled = particle.clone();
            disabled.position.x = 10.0;
            disabled.enabled = false;
            let world = MpmWorld::new(
                vec![particle.clone(), disabled],
                MpmParams {
                    gravity: Vector3::zeros(),
                    cell_width: 0.1,
                    bounds: None,
                    ..MpmParams::default()
                },
            )
            .unwrap();
            let pipeline = GpuMpmTransfers::new(&device);
            let mut session =
                GpuMpmResidentSession::new(&pipeline, &device, &queue, world, 0.001).unwrap();
            let output = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("snapshot consumer storage"),
                size: 128,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("snapshot verification"),
                size: 128,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut chunk = None;
            for phase in 0..3 {
                if phase == 1 {
                    chunk = Some(session.add_particles(vec![particle.clone()]).unwrap());
                }
                if phase == 2 {
                    assert_eq!(session.remove_chunk(chunk.take().unwrap()).unwrap(), 1);
                }
                let mut reference = session.world().clone();
                for _ in 0..3 {
                    reference.step(0.001).unwrap();
                }
                let previous_steps = session.world().substeps;
                session.submit_steps(2).unwrap();
                assert!(session.has_pending_steps());
                assert_eq!(session.world().substeps, previous_steps);
                session.submit_steps(1).unwrap();
                if phase == 0 {
                    session = session.into_owned();
                }
                assert_eq!(session.pending_substeps(), 3);
                assert!(matches!(
                    session.submit_steps(0),
                    Err(GpuMpmError::InvalidInput)
                ));
                assert_eq!(session.pending_substeps(), 3);
                assert!(matches!(session.step(1), Err(GpuMpmError::PendingSteps)));
                assert!(matches!(
                    session.set_obstacles(Vec::new()),
                    Err(GpuMpmError::PendingSteps)
                ));
                assert!(matches!(
                    session.add_particles(vec![particle.clone()]),
                    Err(GpuMpmError::PendingSteps)
                ));
                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                let count = session
                    .encode_particle_snapshot(&mut encoder, &output)
                    .unwrap();
                assert_eq!(count as usize, session.world().particles.len());
                encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, u64::from(count) * 32);
                let _submission = queue.submit(Some(encoder.finish()));
                if phase == 0 {
                    let mut synchronization = Box::pin(session.synchronize_async());
                    let mut context = core::task::Context::from_waker(core::task::Waker::noop());
                    let first_poll = Future::poll(synchronization.as_mut(), &mut context);
                    drop(synchronization);
                    match first_poll {
                        core::task::Poll::Ready(result) => result.unwrap(),
                        core::task::Poll::Pending => {
                            eprintln!("{backend:?}: canceled pending synchronization; retrying");
                            assert_eq!(session.pending_substeps(), 3);
                            assert_eq!(session.world().substeps, previous_steps);
                            session.synchronize_async().await.unwrap();
                        }
                    }
                } else {
                    session.synchronize_async().await.unwrap();
                }
                for (actual, expected) in session.world().particles.iter().zip(&reference.particles)
                {
                    assert!(
                        (actual.position - expected.position).norm() < 1e-4,
                        "{backend:?}: async position"
                    );
                    assert!(
                        (actual.velocity - expected.velocity).norm() < 2e-3,
                        "{backend:?}: async velocity"
                    );
                }
                assert!(!session.has_pending_steps());
                assert_eq!(session.world().substeps, previous_steps + 3);
                session.synchronize_async().await.unwrap();
                assert_eq!(session.world().substeps, previous_steps + 3);
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
                let snapshots = bytemuck::cast_slice::<u8, GpuMpmParticleSnapshot>(&view);
                for (snapshot, expected) in snapshots.iter().zip(&session.world().particles) {
                    for axis in 0..3 {
                        assert!(
                            (f64::from(snapshot.position_mass[axis]) - expected.position[axis])
                                .abs()
                                < 1e-5
                        );
                        assert!(
                            (f64::from(snapshot.velocity_volume[axis]) - expected.velocity[axis])
                                .abs()
                                < 1e-5
                        );
                    }
                    assert_eq!(
                        snapshot.position_mass[3],
                        if expected.enabled {
                            expected.mass as f32
                        } else {
                            0.0
                        }
                    );
                    assert_eq!(snapshot.velocity_volume[3], expected.rest_volume as f32);
                }
                drop(view);
                readback.unmap();
            }
            let too_small = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 16,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            assert!(matches!(
                session.encode_particle_snapshot(&mut encoder, &too_small),
                Err(GpuMpmError::Capacity)
            ));
            assert!(matches!(
                session.encode_particle_snapshot(&mut encoder, &readback),
                Err(GpuMpmError::InvalidInput)
            ));
            session.valid = false;
            assert!(matches!(
                session.encode_particle_snapshot(&mut encoder, &output),
                Err(GpuMpmError::ResidentUnavailable)
            ));
        }
    }

    #[tokio::test]
    async fn compact_topology_dispatch_counts_shared_and_disabled_stencils() {
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: backend,
                ..Default::default()
            });
            let Ok(adapter) = instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
            else {
                continue;
            };
            if adapter.get_info().device_type == wgpu::DeviceType::Cpu {
                continue;
            }
            eprintln!("Compaction adapter: {:?}", adapter.get_info());
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor::default())
                .await
                .unwrap();
            for (positions, enabled, expected) in [
                ([0.05, 0.15, 0.05, 10.05, 20.05], true, [1, 1, 1, 63]),
                ([0.05, 10.05, 20.05, 20.05, 30.05], true, [2, 1, 1, 81]),
                ([0.05, 0.15, 0.05, 10.05, 20.05], false, [0, 1, 1, 0]),
            ] {
                let mut particles = positions
                    .into_iter()
                    .map(|x| {
                        let mut particle = MpmParticle::new(
                            Vector3::new(x, 0.05, 0.05),
                            0.03,
                            1_000.0,
                            MaterialModel::fluid(2_000.0, 7.0, 0.1),
                        );
                        particle.fixed = true;
                        particle.enabled = enabled;
                        particle
                    })
                    .collect::<Vec<_>>();
                particles[4].enabled = false;
                let world = MpmWorld::new(
                    particles,
                    MpmParams {
                        gravity: Vector3::zeros(),
                        cell_width: 0.1,
                        bounds: Some(WorldBounds {
                            min: Vector3::repeat(-100.0),
                            max: Vector3::repeat(100.0),
                        }),
                        ..MpmParams::default()
                    },
                )
                .unwrap();
                let pipeline = GpuMpmTransfers::new(&device);
                let mut session =
                    GpuMpmResidentSession::new(&pipeline, &device, &queue, world, 0.001).unwrap();
                if !enabled {
                    // No active particles require no GPU topology or grid dispatch at all.
                    assert!(session.prepared.is_none());
                    session.step(2).unwrap();
                    assert!(session.prepared.is_none());
                    assert_eq!(session.world().substeps, 2);
                    continue;
                }
                let readback = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("compact topology dispatch readback"),
                    size: 48,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                for _ in 0..2 {
                    let prepared = session.prepared.as_ref().unwrap();
                    let poison = vec![0x7fc0_0000_u32; prepared.nodes as usize * 4];
                    queue.write_buffer(&prepared.grid_buffer, 0, bytemuck::cast_slice(&poison));
                    queue.write_buffer(&prepared.velocity_buffer, 0, bytemuck::cast_slice(&poison));
                    session.step(1).unwrap();
                    let topology = session
                        .prepared
                        .as_ref()
                        .unwrap()
                        .topology
                        .as_ref()
                        .unwrap();
                    let mut encoder =
                        device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                    encoder.copy_buffer_to_buffer(topology.indirect(), 0, &readback, 0, 16);
                    let prepared = session.prepared.as_ref().unwrap();
                    encoder.copy_buffer_to_buffer(
                        &prepared.grid_buffer,
                        (prepared.nodes - 1) * 16,
                        &readback,
                        16,
                        16,
                    );
                    encoder.copy_buffer_to_buffer(
                        &prepared.velocity_buffer,
                        u64::from(expected[3]) * 16,
                        &readback,
                        32,
                        16,
                    );
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
                    // Shared and duplicate nodes must not inflate dispatch; disabled particles add no nodes.
                    assert_eq!(
                        &bytemuck::cast_slice::<u8, u32>(&view)[..4],
                        &expected,
                        "{backend:?}"
                    );
                    // Compaction must clear active nodes without clearing unused capacity.
                    assert_eq!(
                        &bytemuck::cast_slice::<u8, u32>(&view)[4..8],
                        &[0x7fc0_0000_u32; 4]
                    );
                    // Rounded-up lanes must not update even the first unused node velocity.
                    assert_eq!(
                        &bytemuck::cast_slice::<u8, u32>(&view)[8..],
                        &[0x7fc0_0000_u32; 4]
                    );
                    drop(view);
                    readback.unmap();
                }
            }
        }
    }

    #[tokio::test]
    async fn resident_force_edits_reuse_buffers_and_are_consumed_once() {
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: backend,
                ..Default::default()
            });
            let Ok(adapter) = instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
            else {
                continue;
            };
            if adapter.get_info().device_type == wgpu::DeviceType::Cpu {
                continue;
            }
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor::default())
                .await
                .unwrap();
            let mut particles = [-10.0, 10.0, 20.0]
                .into_iter()
                .map(|x| {
                    MpmParticle::new(
                        Vector3::new(x, 0.5, 0.5),
                        0.03,
                        1_000.0,
                        MaterialModel::fluid(1.0, 7.0, 0.0),
                    )
                })
                .collect::<Vec<_>>();
            particles[1].fixed = true;
            particles[2].enabled = false;
            let mut reference = MpmWorld::new(
                particles,
                MpmParams {
                    cell_width: 0.1,
                    bounds: None,
                    gravity: Vector3::zeros(),
                    ..MpmParams::default()
                },
            )
            .unwrap();
            let pipeline = GpuMpmTransfers::new(&device);
            let mut session =
                GpuMpmResidentSession::new(&pipeline, &device, &queue, reference.clone(), 0.001)
                    .unwrap();
            let original_particles = session.prepared.as_ref().unwrap().particle_buffer.clone();
            let original_grid = session.prepared.as_ref().unwrap().grid_buffer.clone();
            let force = Vector3::new(0.2, 0.1, -0.1);
            for phase in 0..2 {
                for index in 0..3 {
                    session.set_particle_force(index, force).unwrap();
                    reference.particles[index].force = force;
                }
                assert_eq!(
                    session.prepared.as_ref().unwrap().particle_buffer,
                    original_particles
                );
                assert_eq!(
                    session.prepared.as_ref().unwrap().grid_buffer,
                    original_grid
                );
                assert!(matches!(
                    session.set_particle_force(0, Vector3::repeat(f64::NAN)),
                    Err(GpuMpmError::InvalidInput)
                ));
                assert!(matches!(
                    session.set_particle_force(0, Vector3::repeat(f64::MAX)),
                    Err(GpuMpmError::InvalidInput)
                ));
                assert!(matches!(
                    session.set_particle_force(usize::MAX, force),
                    Err(GpuMpmError::InvalidInput)
                ));
                assert_eq!(session.world().particles[0].force, force);
                session.submit_steps(1).unwrap();
                assert!(matches!(
                    session.set_particle_force(0, Vector3::zeros()),
                    Err(GpuMpmError::PendingSteps)
                ));
                session.submit_steps(2).unwrap();
                session.synchronize_async().await.unwrap();
                for _ in 0..3 {
                    reference.step(0.001).unwrap();
                }
                for (actual, expected) in session.world().particles.iter().zip(&reference.particles)
                {
                    assert!(
                        (actual.position - expected.position).norm() < 1e-4,
                        "{backend:?}: force position"
                    );
                    assert!(
                        (actual.velocity - expected.velocity).norm() < 1e-5,
                        "{backend:?}: force velocity"
                    );
                    assert_eq!(actual.force, Vector3::zeros());
                }
                let expected_velocity =
                    force * (0.001 * f64::from(phase + 1) / reference.particles[0].mass);
                assert!((session.world().particles[0].velocity - expected_velocity).norm() < 1e-5);
            }
        }
    }

    #[tokio::test]
    async fn unbounded_resident_rejects_cell_overflow_without_applying_state() {
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: backend,
                ..Default::default()
            });
            let Ok(adapter) = instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
            else {
                continue;
            };
            if adapter.get_info().device_type == wgpu::DeviceType::Cpu {
                continue;
            }
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor::default())
                .await
                .unwrap();
            let mut particle = MpmParticle::new(
                Vector3::new(214_748_352.0, 0.05, 0.05),
                0.01,
                1_000.0,
                MaterialModel::fluid(1.0, 7.0, 0.0),
            );
            particle.force.x = 1e15;
            let world = MpmWorld::new(
                vec![particle],
                MpmParams {
                    cell_width: 0.1,
                    bounds: None,
                    gravity: Vector3::zeros(),
                    ..MpmParams::default()
                },
            )
            .unwrap();
            let pipeline = GpuMpmTransfers::new(&device);
            let mut session =
                GpuMpmResidentSession::new(&pipeline, &device, &queue, world.clone(), 0.001)
                    .unwrap();
            session.submit_steps(1).unwrap();
            session.submit_steps(2).unwrap();
            assert!(
                matches!(
                    session.synchronize_async().await,
                    Err(GpuMpmError::ResidentUnavailable)
                ),
                "{backend:?}"
            );
            assert_eq!(session.world().substeps, 0);
            assert_eq!(
                session.world().particles[0].position,
                world.particles[0].position
            );
            assert_eq!(
                session.world().particles[0].velocity,
                world.particles[0].velocity
            );
        }
    }

    #[tokio::test]
    async fn resident_sparse_topology_rebuilds_for_separated_moving_particles() {
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: backend,
                ..Default::default()
            });
            let Ok(adapter) = instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
            else {
                continue;
            };
            if adapter.get_info().device_type == wgpu::DeviceType::Cpu {
                continue;
            }
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor::default())
                .await
                .unwrap();
            let mut particles = [-20.055, -20.015, 20.055, 20.095, -50.0, 0.0]
                .into_iter()
                .map(|x| {
                    MpmParticle::new(
                        Vector3::new(x, 0.5, 0.5),
                        0.03,
                        1_000.0,
                        MaterialModel::fluid(2_000.0, 7.0, 0.1),
                    )
                })
                .collect::<Vec<_>>();
            for (index, particle) in particles[..4].iter_mut().enumerate() {
                particle.velocity.x = if index % 2 == 0 { 1.0 } else { 0.2 };
            }
            particles[4].fixed = true;
            particles[5].enabled = false;
            let params = MpmParams {
                gravity: Vector3::zeros(),
                cell_width: 0.1,
                bounds: None,
                ..MpmParams::default()
            };
            let mut reference = MpmWorld::new(particles, params).unwrap();
            let pipeline = GpuMpmTransfers::new(&device);
            let mut session =
                GpuMpmResidentSession::new(&pipeline, &device, &queue, reference.clone(), 0.001)
                    .unwrap();
            let prepared = session.prepared.as_ref().unwrap();
            assert!(prepared.topology.is_some());
            assert_eq!(prepared.params.dimensions[0], 0);
            assert_eq!(prepared.nodes, 162);
            let initial_cell = (reference.particles[0].position.x / 0.1 - 0.5).floor();
            for _ in 0..4 {
                for _ in 0..8 {
                    reference.step(0.001).unwrap();
                }
                session.step(8).unwrap();
                for (expected, actual) in reference.particles.iter().zip(&session.world().particles)
                {
                    assert!(
                        (expected.position - actual.position).norm() < 2e-4,
                        "{backend:?}: position"
                    );
                    assert!(
                        (expected.velocity - actual.velocity).norm() < 2e-3,
                        "{backend:?}: velocity"
                    );
                    assert!(
                        (expected.affine - actual.affine).norm() < 2e-2,
                        "{backend:?}: affine"
                    );
                }
            }
            assert_ne!(
                (session.world().particles[0].position.x / 0.1 - 0.5).floor(),
                initial_cell
            );
            assert_eq!(session.world().substeps, 32);
            assert_eq!(
                session.world().particles[4].position,
                reference.particles[4].position
            );
            assert_eq!(
                session.world().particles[5].position,
                reference.particles[5].position
            );
        }
    }

    #[tokio::test]
    async fn resident_substeps_project_elastic_sand_and_snow_without_readback() {
        let instance = wgpu::Instance::default();
        let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
        else {
            eprintln!("No WebGPU adapter available for resident material test");
            return;
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let materials = [
            MaterialModel::elastic(4_000.0, 0.2),
            MaterialModel::sand(4_000.0, 0.2, 30.0f64.to_radians(), 0.0),
            MaterialModel::snow(4_000.0, 0.2),
        ];
        let particles = materials
            .into_iter()
            .enumerate()
            .map(|(index, material)| {
                let mut particle = MpmParticle::new(
                    Vector3::new(0.25 + index as f64 * 0.25, 0.5, 0.5),
                    0.04,
                    1_000.0,
                    material,
                );
                particle.deformation = Matrix3::new(1.03, 0.01, 0.0, 0.0, 0.98, 0.0, 0.0, 0.0, 1.0);
                particle
            })
            .collect::<Vec<_>>();
        let params = MpmParams {
            gravity: Vector3::zeros(),
            bounds: Some(WorldBounds {
                min: Vector3::zeros(),
                max: Vector3::repeat(1.0),
            }),
            ..MpmParams::default()
        };
        let mut cpu = MpmWorld::new(particles.clone(), params.clone()).unwrap();
        let mut gpu = MpmWorld::new(particles, params).unwrap();
        let pipeline = GpuMpmTransfers::new(&device);
        for _ in 0..4 {
            cpu.step(0.0001).unwrap();
        }
        gpu.step_gpu_resident_fixed_substeps(&pipeline, &device, &queue, 0.0001, 4)
            .unwrap();
        assert_eq!(cpu.substeps, gpu.substeps);
        for (expected, actual) in cpu.particles.iter().zip(&gpu.particles) {
            assert!((expected.position - actual.position).norm() < 2e-5);
            assert!((expected.velocity - actual.velocity).norm() < 3e-4);
            assert!((expected.deformation - actual.deformation).norm() < 2e-4);
            assert!((expected.plastic.plastic_det - actual.plastic.plastic_det).abs() < 2e-4);
        }
        let mut singular = MpmParticle::new(
            Vector3::repeat(0.5),
            0.04,
            1_000.0,
            MaterialModel::elastic(4_000.0, 0.2),
        );
        singular.deformation[(2, 2)] = 1e-8;
        let mut fallback_world = MpmWorld::new(vec![singular], gpu.params.clone()).unwrap();
        assert!(matches!(
            fallback_world.step_gpu_resident_fixed_substeps(&pipeline, &device, &queue, 0.0001, 4),
            Err(GpuMpmError::ResidentUnavailable)
        ));
        assert_eq!(fallback_world.substeps, 0);
        assert_eq!(fallback_world.particles[0].deformation[(2, 2)], 1e-8);
    }

    #[tokio::test]
    async fn resident_substeps_match_cpu_with_force_boundary_and_fixed_particles() {
        let instance = wgpu::Instance::default();
        let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
        else {
            eprintln!("No WebGPU adapter available for resident MPM test");
            return;
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let mut particles = vec![
            MpmParticle::new(
                Vector3::new(0.42, 0.48, 0.51),
                0.04,
                1_000.0,
                MaterialModel::neo_hookean(2_000.0, 0.2),
            ),
            MpmParticle::new(
                Vector3::new(0.48, 0.48, 0.0401),
                0.04,
                1_000.0,
                MaterialModel::fluid(2_000.0, 7.0, 0.1),
            ),
            MpmParticle::new(
                Vector3::new(0.6, 0.5, 0.5),
                0.04,
                1_000.0,
                MaterialModel::neo_hookean(2_000.0, 0.2),
            ),
        ];
        particles[0].force = Vector3::new(0.2, -0.1, 0.3);
        particles[0].damping = 0.4;
        particles[1].velocity.z = -2.0;
        particles[2].fixed = true;
        particles[2].velocity.x = 1.0;
        let params = MpmParams {
            bounds: Some(WorldBounds {
                min: Vector3::zeros(),
                max: Vector3::repeat(1.0),
            }),
            ..MpmParams::default()
        };
        let mut cpu = MpmWorld::new(particles.clone(), params.clone()).unwrap();
        let mut gpu = MpmWorld::new(particles, params).unwrap();
        let pipeline = GpuMpmTransfers::new(&device);
        let mut session =
            GpuMpmResidentSession::new(&pipeline, &device, &queue, gpu.clone(), 0.0001).unwrap();
        for _ in 0..2 {
            for _ in 0..4 {
                cpu.step(0.0001).unwrap();
            }
            gpu.step_gpu_resident_fixed_substeps(&pipeline, &device, &queue, 0.0001, 4)
                .unwrap();
            session.step(4).unwrap();
            assert_eq!(session.world().substeps, gpu.substeps);
            for (expected, actual) in gpu.particles.iter().zip(&session.world().particles) {
                assert!((expected.position - actual.position).norm() < 1e-6);
                assert!((expected.velocity - actual.velocity).norm() < 1e-6);
            }
        }
        assert_eq!(session.into_world().substeps, 8);
        assert_eq!(cpu.substeps, gpu.substeps);
        for (expected, actual) in cpu.particles.iter().zip(&gpu.particles) {
            assert!((expected.position - actual.position).norm() < 2e-5);
            assert!((expected.velocity - actual.velocity).norm() < 2e-4);
            assert!((expected.deformation - actual.deformation).norm() < 2e-4);
            assert!((expected.affine - actual.affine).norm() < 2e-3);
        }
        let obstacle = RigidObstacle::sphere(Vector3::repeat(0.5), 0.1);
        cpu.set_obstacles(vec![obstacle.clone()]).unwrap();
        gpu.set_obstacles(vec![obstacle]).unwrap();
        for _ in 0..4 {
            cpu.step(0.0001).unwrap();
        }
        gpu.step_gpu_resident_fixed_substeps(&pipeline, &device, &queue, 0.0001, 4)
            .unwrap();
        for (expected, actual) in cpu.particles.iter().zip(&gpu.particles) {
            assert!((expected.position - actual.position).norm() < 2e-5);
            assert!((expected.velocity - actual.velocity).norm() < 2e-4);
        }

        let escape_particle = MpmParticle::new(
            Vector3::new(0.95, 0.5, 0.5),
            0.04,
            1_000.0,
            MaterialModel::fluid(2_000.0, 7.0, 0.1),
        );
        let mut escape_world = MpmWorld::new(vec![escape_particle], gpu.params.clone()).unwrap();
        escape_world
            .set_obstacles(vec![RigidObstacle::sphere(
                Vector3::new(0.9, 0.5, 0.5),
                10.0,
            )])
            .unwrap();
        let original_position = escape_world.particles[0].position;
        assert!(matches!(
            escape_world.step_gpu_resident_fixed_substeps(&pipeline, &device, &queue, 0.0001, 4),
            Err(GpuMpmError::ResidentUnavailable)
        ));
        assert_eq!(escape_world.particles[0].position, original_position);
        assert_eq!(escape_world.substeps, 0);

        let mut accelerated = MpmParticle::new(
            Vector3::repeat(0.5),
            0.04,
            1_000.0,
            MaterialModel::neo_hookean(2_000.0, 0.2),
        );
        accelerated.force.x = 4_000_000.0;
        let mut unstable_world = MpmWorld::new(vec![accelerated], gpu.params.clone()).unwrap();
        let mut unstable_session =
            GpuMpmResidentSession::new(&pipeline, &device, &queue, unstable_world.clone(), 0.0001)
                .unwrap();
        assert!(matches!(
            unstable_session.step(4),
            Err(GpuMpmError::ResidentUnavailable)
        ));
        assert_eq!(unstable_session.world().substeps, 0);
        assert!(matches!(
            unstable_session.step(1),
            Err(GpuMpmError::ResidentUnavailable)
        ));
        assert_eq!(
            unstable_session.into_world().particles[0].force.x,
            4_000_000.0
        );
        assert!(matches!(
            unstable_world.step_gpu_resident_fixed_substeps(&pipeline, &device, &queue, 0.0001, 4),
            Err(GpuMpmError::ResidentUnavailable)
        ));
        assert_eq!(unstable_world.substeps, 0);
        assert_eq!(unstable_world.particles[0].force.x, 4_000_000.0);
        unstable_world
            .step_gpu_resident_fixed_substeps(&pipeline, &device, &queue, 0.000001, 4)
            .unwrap();
        assert_eq!(unstable_world.substeps, 4);
        assert_eq!(unstable_world.particles[0].force.x, 0.0);
    }

    #[test]
    fn grid_topology_uses_dense_indices_for_local_particles_and_compact_indices_for_sparse_ones() {
        let packed = |cell: [i32; 3]| {
            let mut particle: GpuParticle = bytemuck::Zeroable::zeroed();
            particle.position_mass[3] = 1.0;
            particle.base_cell[..3].copy_from_slice(&cell);
            particle
        };
        let dense = GridTopology::build(&[packed([-3, 2, 4]), packed([-2, 2, 4])], 256, 4).unwrap();
        assert_eq!(dense.origin, [-3, 2, 4]);
        assert_eq!(dense.dimensions, [4, 3, 3]);
        assert_eq!(dense.nodes, 36);
        assert_eq!(dense.particle_nodes.len(), 1);

        let sparse = GridTopology::build(
            &[packed([i32::MIN, 0, 0]), packed([i32::MAX - 2, 0, 0])],
            256,
            216,
        )
        .unwrap();
        assert_eq!(sparse.dimensions, [0; 3]);
        assert_eq!(sparse.nodes, 54);
        assert_eq!(sparse.particle_nodes.len(), 54);
        assert_eq!(sparse.node_coords.len(), 54);
        let inputs = [packed([i32::MIN, 0, 0]), packed([i32::MAX - 2, 0, 0])];
        let hash = GridTopology::gpu_generated(&inputs, 256, 216, 4).unwrap();
        assert_eq!(hash.dimensions, [0, 128, 0]);
        assert_eq!(hash.nodes, 54);
        let repeated = [inputs[0], inputs[1], inputs[0]];
        let fallback = GridTopology::gpu_generated(&repeated, 64, 324, 4).unwrap();
        assert_eq!(fallback.dimensions, [0; 3]);
        assert_eq!(fallback.nodes, 54);
        let dispatch_limited = GridTopology::gpu_generated(&inputs, 256, 216, 1).unwrap();
        assert_eq!(dispatch_limited.node_coords, sparse.node_coords);
        assert_eq!(dispatch_limited.particle_nodes, sparse.particle_nodes);
    }

    #[tokio::test]
    async fn resident_session_refreshes_moving_obstacles_without_rebuilding_particles() {
        let instance = wgpu::Instance::default();
        let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
        else {
            eprintln!("No WebGPU adapter available for resident obstacle refresh test");
            return;
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let params = MpmParams {
            gravity: Vector3::zeros(),
            bounds: Some(WorldBounds {
                min: Vector3::zeros(),
                max: Vector3::repeat(1.0),
            }),
            ..MpmParams::default()
        };
        let particle = MpmParticle::new(
            Vector3::new(0.5, 0.5, 0.52),
            0.04,
            1_000.0,
            MaterialModel::fluid(2_000.0, 7.0, 0.1),
        );
        let reference = MpmWorld::new(vec![particle], params).unwrap();
        let pipeline = GpuMpmTransfers::new(&device);
        let mut session =
            GpuMpmResidentSession::new(&pipeline, &device, &queue, reference.clone(), 0.0001)
                .unwrap();
        let mut reference = reference;
        let mut cpu = reference.clone();
        let mut invalid = RigidObstacle::sphere(Vector3::repeat(0.5), 0.1);
        invalid.friction = -1.0;
        assert!(matches!(
            session.set_obstacles(vec![invalid]),
            Err(GpuMpmError::Reference(MpmError::InvalidInput))
        ));
        assert!(session.world().obstacles.is_empty());

        for stage in 0..4 {
            let position = session.world().particles[0].position;
            let obstacles = match stage {
                0 | 1 => {
                    let mut sphere =
                        RigidObstacle::sphere(position - Vector3::new(0.02, 0.0, 0.02), 0.1);
                    sphere.linear_velocity = Vector3::new(0.1 * f64::from(stage), 0.0, 0.0);
                    vec![sphere]
                }
                2 => Vec::new(),
                _ => {
                    let vertices = [-0.07, 0.07]
                        .into_iter()
                        .flat_map(|x| {
                            [-0.07, 0.07].into_iter().flat_map(move |y| {
                                [-0.07, 0.07]
                                    .into_iter()
                                    .map(move |z| Vector3::new(x, y, z))
                            })
                        })
                        .collect::<Vec<_>>();
                    let normals = [
                        Vector3::x(),
                        -Vector3::x(),
                        Vector3::y(),
                        -Vector3::y(),
                        Vector3::z(),
                        -Vector3::z(),
                    ];
                    vec![RigidObstacle::convex(
                        position,
                        &vertices,
                        &normals,
                        UnitQuaternion::identity(),
                    )]
                }
            };
            reference.set_obstacles(obstacles.clone()).unwrap();
            cpu.set_obstacles(obstacles.clone()).unwrap();
            session.set_obstacles(obstacles.clone()).unwrap();
            assert_eq!(session.world().obstacles, obstacles);
            if stage == 0 {
                let too_large = RigidObstacle::sphere(Vector3::repeat(0.5), 1e100);
                assert!(too_large.is_valid());
                assert!(matches!(
                    session.set_obstacles(vec![too_large]),
                    Err(GpuMpmError::InvalidInput)
                ));
                assert_eq!(session.world().obstacles, obstacles);
            }
            assert_eq!(
                session.prepared.as_ref().unwrap().params.origin[3],
                i32::try_from(obstacles.len()).unwrap()
            );
            reference
                .step_gpu_resident_fixed_substeps(&pipeline, &device, &queue, 0.0001, 1)
                .unwrap();
            cpu.step(0.0001).unwrap();
            session.step(1).unwrap();
            assert_eq!(session.world().substeps, reference.substeps);
            assert!(
                (session.world().particles[0].position - reference.particles[0].position).norm()
                    < 1e-6
            );
            assert!(
                (session.world().particles[0].velocity - reference.particles[0].velocity).norm()
                    < 1e-6
            );
            assert!(
                (session.world().particles[0].position - cpu.particles[0].position).norm() < 2e-4
            );
            assert!(
                (session.world().particles[0].velocity - cpu.particles[0].velocity).norm() < 2e-3
            );
        }
    }

    #[tokio::test]
    async fn resident_session_adds_and_removes_particle_chunks_across_empty_state() {
        let instance = wgpu::Instance::default();
        let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
        else {
            eprintln!("No WebGPU adapter available for resident particle edit test");
            return;
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let particle = |x| {
            MpmParticle::new(
                Vector3::new(x, 0.5, 0.5),
                0.03,
                1_000.0,
                MaterialModel::fluid(2_000.0, 7.0, 0.1),
            )
        };
        let params = MpmParams {
            gravity: Vector3::new(0.0, 0.0, -1.0),
            bounds: Some(WorldBounds {
                min: Vector3::zeros(),
                max: Vector3::repeat(1.0),
            }),
            ..MpmParams::default()
        };
        let mut reference = MpmWorld::new(vec![particle(0.4)], params).unwrap();
        let pipeline = GpuMpmTransfers::new(&device);
        let mut session = {
            let local_pipeline = GpuMpmTransfers::new(&device);
            let local_device = device.clone();
            let local_queue = queue.clone();
            GpuMpmResidentSession::new(
                &local_pipeline,
                &local_device,
                &local_queue,
                reference.clone(),
                0.0001,
            )
            .unwrap()
            .into_owned()
        };
        let mut added_chunk = None;
        for stage in 0..5 {
            match stage {
                1 => {
                    let mut invalid = particle(0.5);
                    invalid.damping = -1.0;
                    assert!(matches!(
                        session.add_particles(vec![invalid]),
                        Err(GpuMpmError::Reference(MpmError::InvalidInput))
                    ));
                    assert_eq!(session.world().particles.len(), 1);
                    assert!(matches!(
                        session.add_particles(vec![particle(2.0)]),
                        Err(GpuMpmError::ResidentUnavailable)
                    ));
                    assert_eq!(session.world().particles.len(), 1);
                    let particles = vec![particle(0.55), particle(0.65)];
                    let expected = reference.add_particles(particles.clone()).unwrap();
                    let actual = session.add_particles(particles).unwrap();
                    assert_eq!(actual, expected);
                    added_chunk = Some(actual);
                }
                2 => {
                    assert_eq!(reference.remove_chunk(ParticleChunkId::INITIAL).unwrap(), 1);
                    assert_eq!(session.remove_chunk(ParticleChunkId::INITIAL).unwrap(), 1);
                }
                3 => {
                    let chunk = added_chunk.unwrap();
                    assert_eq!(reference.remove_chunk(chunk).unwrap(), 2);
                    assert_eq!(session.remove_chunk(chunk).unwrap(), 2);
                    assert!(session.prepared.is_none());
                    assert!(matches!(
                        session.remove_chunk(chunk),
                        Err(GpuMpmError::Reference(MpmError::UnknownChunk))
                    ));
                }
                4 => {
                    let obstacle = RigidObstacle::sphere(Vector3::new(0.6, 0.5, 0.5), 0.1);
                    session.set_obstacles(vec![obstacle.clone()]).unwrap();
                    reference.set_obstacles(vec![obstacle]).unwrap();
                    let emitted = vec![particle(0.72)];
                    let expected = reference.add_particles(emitted.clone()).unwrap();
                    assert_eq!(session.add_particles(emitted).unwrap(), expected);
                    assert!(session.prepared.is_some());
                }
                _ => {}
            }
            assert_eq!(session.world().particles.len(), reference.particles.len());
            assert_eq!(session.world().obstacles, reference.obstacles);
            session.step(1).unwrap();
            reference
                .step_gpu_resident_fixed_substeps(&pipeline, &device, &queue, 0.0001, 1)
                .unwrap();
            assert_eq!(session.world().substeps, reference.substeps);
            for (actual, expected) in session.world().particles.iter().zip(&reference.particles) {
                assert_eq!(actual.chunk_id(), expected.chunk_id());
                assert!((actual.position - expected.position).norm() < 1e-6);
                assert!((actual.velocity - expected.velocity).norm() < 1e-6);
                assert!((actual.deformation - expected.deformation).norm() < 1e-6);
            }
        }
    }

    #[tokio::test]
    async fn gpu_transfers_track_cpu_reference_for_elastic_particles() {
        let instance = wgpu::Instance::default();
        let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
        else {
            eprintln!("No WebGPU adapter available for MPM transfer test");
            return;
        };
        let info = adapter.get_info();
        eprintln!(
            "MPM transfer adapter: {} ({:?}, {:?})",
            info.name, info.backend, info.device_type
        );
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let mut particles = vec![
            MpmParticle::new(
                Vector3::new(0.35, 0.43, 0.53),
                0.04,
                1_000.0,
                MaterialModel::elastic(2_000.0, 0.2),
            ),
            MpmParticle::new(
                Vector3::new(0.41, 0.47, 0.56),
                0.04,
                1_000.0,
                MaterialModel::neo_hookean(2_000.0, 0.2),
            ),
        ];
        particles[0].velocity = Vector3::new(0.2, -0.1, -0.5);
        particles[1].velocity = Vector3::new(-0.1, 0.3, 0.2);
        particles[0].deformation[(0, 0)] = 0.97;
        particles[1].deformation[(1, 1)] = 1.04;
        let params = MpmParams {
            bounds: Some(WorldBounds {
                min: Vector3::zeros(),
                max: Vector3::repeat(1.0),
            }),
            ..MpmParams::default()
        };
        let mut cpu = MpmWorld::new(particles.clone(), params.clone()).unwrap();
        let mut gpu = MpmWorld::new(particles, params).unwrap();
        cpu.step(0.002).unwrap();
        gpu.step_with_gpu_transfers(&GpuMpmTransfers::new(&device), &device, &queue, 0.002)
            .unwrap();
        assert_eq!(cpu.substeps, gpu.substeps);
        for (expected, actual) in cpu.particles.iter().zip(&gpu.particles) {
            assert!((expected.position - actual.position).norm() < 1e-5);
            assert!((expected.velocity - actual.velocity).norm() < 1e-4);
            assert!((expected.affine - actual.affine).norm() < 1e-3);
            assert!((expected.deformation - actual.deformation).norm() < 1e-5);
        }
    }

    #[tokio::test]
    async fn closed_mesh_particles_step_on_gpu_like_cpu_reference() {
        let instance = wgpu::Instance::default();
        let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
        else {
            eprintln!("No WebGPU adapter available for closed-mesh MPM test");
            return;
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let emitter = MeshEmitter::new(
            vec![
                Vector3::new(0.5, 0.5, 0.5),
                Vector3::new(1.5, 0.5, 0.5),
                Vector3::new(0.5, 1.5, 0.5),
                Vector3::new(0.5, 0.5, 1.5),
            ],
            vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
            0.25,
            1_000.0,
            MaterialModel::elastic(1_000.0, 0.2),
            64,
            64,
        );
        let particles = emitter.sample().unwrap();
        assert!(!particles.is_empty());
        let params = MpmParams {
            bounds: Some(WorldBounds {
                min: Vector3::zeros(),
                max: Vector3::repeat(2.0),
            }),
            ..MpmParams::default()
        };
        let mut cpu = MpmWorld::new(particles.clone(), params.clone()).unwrap();
        let mut gpu = MpmWorld::new(particles, params).unwrap();
        cpu.step(0.001).unwrap();
        gpu.step_with_gpu_transfers(&GpuMpmTransfers::new(&device), &device, &queue, 0.001)
            .unwrap();
        assert_eq!(cpu.substeps, gpu.substeps);
        assert_eq!(cpu.particles.len(), gpu.particles.len());
        for (expected, actual) in cpu.particles.iter().zip(&gpu.particles) {
            assert!((expected.position - actual.position).norm() < 1e-4);
            assert!((expected.velocity - actual.velocity).norm() < 1e-4);
        }
    }

    #[tokio::test]
    async fn transfer_colors_separate_gpu_grid_momentum_in_direct_and_resident_steps() {
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
        let mut first = MpmParticle::new(
            Vector3::new(0.5, 0.5, 0.5),
            0.04,
            1_000.0,
            MaterialModel::elastic(1_000.0, 0.2),
        );
        first.velocity.x = 1.0;
        let mut second = first.clone();
        second.velocity.x = -1.0;
        second.transfer_color = 1;
        let params = MpmParams {
            gravity: Vector3::zeros(),
            bounds: Some(WorldBounds {
                min: Vector3::zeros(),
                max: Vector3::repeat(1.0),
            }),
            ..MpmParams::default()
        };
        let world = MpmWorld::new(vec![first, second], params).unwrap();
        let mut cpu = world.clone();
        cpu.step(0.001).unwrap();
        let transfers = GpuMpmTransfers::new(&device);
        let mut direct = world.clone();
        direct
            .step_with_gpu_transfers(&transfers, &device, &queue, 0.001)
            .unwrap();
        let mut resident =
            GpuMpmResidentSession::new(&transfers, &device, &queue, world, 0.001).unwrap();
        resident.step(1).unwrap();
        for actual in [&direct, resident.world()] {
            for (expected, particle) in cpu.particles.iter().zip(&actual.particles) {
                assert!((particle.velocity.x - expected.velocity.x).abs() < 1e-4);
                assert!((particle.position.x - expected.position.x).abs() < 1e-5);
            }
        }
    }

    #[tokio::test]
    async fn cpic_groups_separate_gpu_grid_in_direct_and_resident_steps() {
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
        let mut left = MpmParticle::new(
            Vector3::new(0.48, 0.5, 0.5),
            0.005,
            1_000.0,
            MaterialModel::elastic(1_000.0, 0.2),
        );
        left.velocity.x = -1.0;
        let mut right = left.clone();
        right.position.x = 0.52;
        right.velocity.x = 1.0;
        let mut sheet = RigidObstacle::triangle_prism(
            Vector3::repeat(0.5),
            [
                Vector3::new(0.0, -1.0, -1.0),
                Vector3::new(0.0, 1.0, -1.0),
                Vector3::new(0.0, 0.0, 1.0),
            ],
            0.001,
            UnitQuaternion::identity(),
        );
        sheet.boundary = crate::ObstacleBoundary::NonReflecting;
        let params = MpmParams {
            gravity: Vector3::zeros(),
            ..MpmParams::default()
        };
        let transfers = GpuMpmTransfers::new(&device);
        for (group, left_color, all_groups) in
            [(0, 0, false), (16, 1, false), (31, 0, false), (31, 0, true)]
        {
            let mut left = left.clone();
            left.transfer_color = left_color;
            let obstacles = (if all_groups { 0..32 } else { group..group + 1 })
                .map(|group| {
                    let mut sheet = sheet.clone();
                    sheet.cpic_group = Some(group);
                    sheet
                })
                .collect();
            let mut world = MpmWorld::new(vec![left, right.clone()], params.clone()).unwrap();
            world.set_obstacles(obstacles).unwrap();
            let mut cpu = world.clone();
            cpu.step(0.001).unwrap();
            assert!(cpu.particles[0].velocity.x < -0.9, "group {group}");
            assert!(cpu.particles[1].velocity.x > 0.9, "group {group}");
            let mut direct = world.clone();
            direct
                .step_with_gpu_transfers(&transfers, &device, &queue, 0.001)
                .unwrap();
            let mut resident =
                GpuMpmResidentSession::new(&transfers, &device, &queue, world, 0.001).unwrap();
            resident.step(1).unwrap();
            for actual in [&direct, resident.world()] {
                for (expected, particle) in cpu.particles.iter().zip(&actual.particles) {
                    assert!(
                        (particle.velocity.x - expected.velocity.x).abs() < 1e-3,
                        "group {group}"
                    );
                    assert!((particle.position.x - expected.position.x).abs() < 1e-5);
                }
            }
        }
    }

    #[tokio::test]
    async fn gpu_transfers_track_cpu_with_materials_fixed_and_disabled_particles() {
        let instance = wgpu::Instance::default();
        let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
        else {
            eprintln!("No WebGPU adapter available for MPM material test");
            return;
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let materials = [
            MaterialModel::elastic(1_000.0, 0.2),
            MaterialModel::neo_hookean(1_000.0, 0.2),
            MaterialModel::sand(1_000.0, 0.2, 35.0f64.to_radians(), 0.0),
            MaterialModel::fluid(2_000.0, 7.0, 0.01),
            MaterialModel::snow(1_000.0, 0.2),
        ];
        let mut particles: Vec<_> = materials
            .into_iter()
            .enumerate()
            .map(|(index, material)| {
                MpmParticle::new(
                    Vector3::new(0.3 + index as f64 * 0.035, 0.5, 0.5),
                    0.03,
                    1_000.0,
                    material,
                )
            })
            .collect();
        particles[0].force = Vector3::new(0.1, -0.2, 0.3);
        particles[1].damping = 0.4;
        let mut fixed = MpmParticle::new(
            Vector3::new(0.48, 0.5, 0.5),
            0.03,
            1_000.0,
            MaterialModel::elastic(1_000.0, 0.2),
        );
        fixed.fixed = true;
        particles.push(fixed);
        let mut disabled = MpmParticle::new(
            Vector3::new(0.52, 0.5, 0.5),
            0.03,
            1_000.0,
            MaterialModel::elastic(1_000.0, 0.2),
        );
        disabled.enabled = false;
        particles.push(disabled);
        let mut cpu = MpmWorld::new(particles.clone(), MpmParams::default()).unwrap();
        let mut gpu = MpmWorld::new(particles, MpmParams::default()).unwrap();
        cpu.step(0.002).unwrap();
        gpu.step_with_gpu_transfers(&GpuMpmTransfers::new(&device), &device, &queue, 0.002)
            .unwrap();
        for (expected, actual) in cpu.particles.iter().zip(&gpu.particles) {
            assert!((expected.position - actual.position).norm() < 1e-5);
            assert!((expected.velocity - actual.velocity).norm() < 1e-4);
            assert!((expected.deformation - actual.deformation).norm() < 1e-5);
            assert_eq!(expected.enabled, actual.enabled);
            assert_eq!(expected.fixed, actual.fixed);
        }
    }

    #[tokio::test]
    async fn gpu_neo_hookean_and_viscous_fluid_stress_track_cpu() {
        let instance = wgpu::Instance::default();
        let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
        else {
            eprintln!("No WebGPU adapter available for MPM stress test");
            return;
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let mut solid = MpmParticle::new(
            Vector3::new(0.3, 0.4, 0.5),
            0.04,
            1_000.0,
            MaterialModel::neo_hookean(4_000.0, 0.25),
        );
        solid.deformation = Matrix3::new(1.2, 0.1, 0.0, 0.0, 0.85, 0.05, 0.0, 0.0, 1.1);
        solid.affine = Matrix3::new(0.2, 0.1, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0, 0.1);
        let mut fluid = MpmParticle::new(
            Vector3::new(0.7, 0.6, 0.5),
            0.04,
            1_000.0,
            MaterialModel::Fluid {
                bulk_modulus: 3_000.0,
                gamma: 6.0,
                viscosity: 12.0,
                tensile_stiffness: 0.3,
            },
        );
        fluid.deformation = Matrix3::new(0.9, 0.0, 0.0, 0.0, 1.05, 0.0, 0.0, 0.0, 1.0);
        fluid.affine = Matrix3::new(0.4, 0.2, 0.0, -0.1, -0.2, 0.0, 0.0, 0.0, 0.1);
        let mut stretched_fluid = fluid.clone();
        stretched_fluid.position = Vector3::new(0.7, 0.2, 0.5);
        stretched_fluid.deformation = Matrix3::new(1.1, 0.0, 0.0, 0.0, 1.03, 0.0, 0.0, 0.0, 1.0);
        let params = MpmParams {
            gravity: Vector3::zeros(),
            ..Default::default()
        };
        let mut cpu = MpmWorld::new(
            vec![solid.clone(), fluid.clone(), stretched_fluid.clone()],
            params.clone(),
        )
        .unwrap();
        let mut gpu = MpmWorld::new(vec![solid, fluid, stretched_fluid], params).unwrap();
        let pipeline = GpuMpmTransfers::new(&device);
        let first_transfer = pipeline
            .transfer(&device, &queue, &gpu.particles, &gpu.params, &[], 0.001)
            .unwrap();
        for transfer in &first_transfer[1..] {
            let deformation = transfer.deformation();
            let scale = deformation[(0, 0)];
            assert!((deformation - Matrix3::identity() * scale).norm() < 1e-6);
        }
        for _ in 0..3 {
            cpu.step(0.001).unwrap();
            gpu.step_with_gpu_transfers(&pipeline, &device, &queue, 0.001)
                .unwrap();
        }
        assert_eq!(cpu.substeps, gpu.substeps);
        for (expected, actual) in cpu.particles.iter().zip(&gpu.particles) {
            assert!((expected.position - actual.position).norm() < 2e-5);
            assert!((expected.velocity - actual.velocity).norm() < 3e-4);
            assert!((expected.deformation - actual.deformation).norm() < 2e-4);
        }
    }

    #[tokio::test]
    async fn gpu_corotated_stress_tracks_cpu_for_elastic_sand_and_snow() {
        let instance = wgpu::Instance::default();
        let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
        else {
            eprintln!("No WebGPU adapter available for MPM corotated stress test");
            return;
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let materials = [
            MaterialModel::elastic(4_000.0, 0.2),
            MaterialModel::sand(4_000.0, 0.2, 30.0f64.to_radians(), 0.0),
            MaterialModel::snow(4_000.0, 0.2),
        ];
        let mut particles = materials
            .into_iter()
            .enumerate()
            .map(|(index, material)| {
                let mut particle = MpmParticle::new(
                    Vector3::new(0.25 + index as f64 * 0.25, 0.4, 0.5),
                    0.04,
                    1_000.0,
                    material,
                );
                particle.deformation =
                    Matrix3::new(1.1, 0.15, 0.02, 0.0, 0.88, 0.08, 0.0, 0.0, 1.04);
                if index == 2 {
                    particle.plastic.plastic_det = 0.8;
                }
                particle
            })
            .collect::<Vec<_>>();
        for (x, diagonal) in [
            (0.25, Vector3::new(0.95, 0.8, 0.75)),
            (0.5, Vector3::new(1.2, 0.9, 0.7)),
        ] {
            let mut particle = MpmParticle::new(
                Vector3::new(x, 0.7, 0.5),
                0.04,
                1_000.0,
                MaterialModel::sand(4_000.0, 0.2, 30.0f64.to_radians(), 0.0),
            );
            particle.deformation = Matrix3::from_diagonal(&diagonal);
            particles.push(particle);
        }
        let params = MpmParams {
            gravity: Vector3::zeros(),
            ..Default::default()
        };
        let mut cpu = MpmWorld::new(particles.clone(), params.clone()).unwrap();
        let mut gpu = MpmWorld::new(particles, params).unwrap();
        let pipeline = GpuMpmTransfers::new(&device);
        let first_transfer = pipeline
            .transfer(&device, &queue, &gpu.particles, &gpu.params, &[], 0.001)
            .unwrap();
        for index in [1, 2, 3, 4] {
            let particle = &cpu.particles[index];
            let transfer = first_transfer[index];
            assert_eq!(transfer.plastic[3], 1.0);
            let unprojected =
                (Matrix3::identity() + transfer.affine() * 0.001) * particle.deformation;
            let (expected_deformation, expected_plastic) = particle
                .material
                .project_deformation(unprojected, particle.plastic);
            assert!((expected_deformation - transfer.deformation()).norm() < 2e-5);
            assert!((expected_plastic.plastic_det - transfer.plastic().plastic_det).abs() < 2e-5);
            assert!((expected_plastic.hardening - transfer.plastic().hardening).abs() < 2e-5);
            assert!(
                (expected_plastic.log_volume_gain - transfer.plastic().log_volume_gain).abs()
                    < 2e-5
            );
        }
        assert_eq!(first_transfer[3].plastic().hardening, 0.0);
        assert!(first_transfer[4].plastic().hardening > 0.0);
        for _ in 0..12 {
            cpu.step(0.001).unwrap();
            gpu.step_with_gpu_transfers(&pipeline, &device, &queue, 0.001)
                .unwrap();
        }
        assert_eq!(cpu.substeps, gpu.substeps);
        for (expected, actual) in cpu.particles.iter().zip(&gpu.particles) {
            assert!((expected.position - actual.position).norm() < 2e-5);
            assert!((expected.velocity - actual.velocity).norm() < 3e-4);
            assert!((expected.deformation - actual.deformation).norm() < 2e-4);
            assert!((expected.plastic.plastic_det - actual.plastic.plastic_det).abs() < 2e-4);
        }
    }

    #[tokio::test]
    async fn gpu_corotated_near_singular_deformation_matches_cpu_fallback() {
        let instance = wgpu::Instance::default();
        let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
        else {
            eprintln!("No WebGPU adapter available for MPM fallback test");
            return;
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let particles = [
            MaterialModel::elastic(2_000.0, 0.2),
            MaterialModel::sand(2_000.0, 0.2, 30.0f64.to_radians(), 0.0),
            MaterialModel::snow(2_000.0, 0.2),
        ]
        .into_iter()
        .enumerate()
        .map(|(index, material)| {
            let mut particle = MpmParticle::new(
                Vector3::new(0.2 + index as f64 * 0.25, 0.4, 0.4),
                0.04,
                1_000.0,
                material,
            );
            particle.deformation[(0, 0)] = 1e-6;
            particle
        })
        .collect::<Vec<_>>();
        let params = MpmParams {
            gravity: Vector3::zeros(),
            ..Default::default()
        };
        let mut cpu = MpmWorld::new(particles.clone(), params.clone()).unwrap();
        let mut gpu = MpmWorld::new(particles, params).unwrap();
        let pipeline = GpuMpmTransfers::new(&device);
        let first_transfer = pipeline
            .transfer(&device, &queue, &gpu.particles, &gpu.params, &[], 0.001)
            .unwrap();
        assert_eq!(first_transfer[1].plastic[3], 0.0);
        assert_eq!(first_transfer[2].plastic[3], 0.0);
        cpu.step(0.001).unwrap();
        gpu.step_with_gpu_transfers(&pipeline, &device, &queue, 0.001)
            .unwrap();
        for (expected, actual) in cpu.particles.iter().zip(&gpu.particles) {
            assert!((expected.velocity - actual.velocity).norm() < 1e-4);
            assert!((expected.position - actual.position).norm() < 1e-5);
            assert!((expected.deformation - actual.deformation).norm() < 2e-4);
            assert!((expected.plastic.plastic_det - actual.plastic.plastic_det).abs() < 2e-4);
        }
    }

    #[tokio::test]
    async fn gpu_grid_boundary_and_sparse_span_match_cpu() {
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: backend,
                ..Default::default()
            });
            let Ok(adapter) = instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
            else {
                eprintln!("No WebGPU adapter available for MPM boundary test");
                continue;
            };
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor::default())
                .await
                .unwrap();
            let pipeline = GpuMpmTransfers::new(&device);
            let mut falling = MpmParticle::new(
                Vector3::new(0.5, 0.5, 0.05),
                0.04,
                1_000.0,
                MaterialModel::elastic(1_000.0, 0.2),
            );
            falling.velocity.z = -10.0;
            let params = MpmParams {
                gravity: Vector3::zeros(),
                bounds: Some(WorldBounds {
                    min: Vector3::zeros(),
                    max: Vector3::repeat(1.0),
                }),
                ..MpmParams::default()
            };
            let mut cpu = MpmWorld::new(vec![falling.clone()], params.clone()).unwrap();
            let mut gpu = MpmWorld::new(vec![falling], params).unwrap();
            cpu.step(0.001).unwrap();
            gpu.step_with_gpu_transfers(&pipeline, &device, &queue, 0.001)
                .unwrap();
            assert!((cpu.particles[0].velocity - gpu.particles[0].velocity).norm() < 1e-4);
            assert!(gpu.particles[0].position.z >= gpu.particles[0].radius);

            let far = [
                MpmParticle::new(
                    Vector3::repeat(0.1),
                    0.04,
                    1_000.0,
                    MaterialModel::elastic(1_000.0, 0.2),
                ),
                MpmParticle::new(
                    Vector3::repeat(10_000.0),
                    0.04,
                    1_000.0,
                    MaterialModel::elastic(1_000.0, 0.2),
                ),
            ];
            let mut cpu = MpmWorld::new(far.to_vec(), MpmParams::default()).unwrap();
            let mut gpu = MpmWorld::new(far.to_vec(), MpmParams::default()).unwrap();
            let prepared = pipeline
                .prepare(
                    &device,
                    &gpu.particles,
                    &gpu.params,
                    &gpu.obstacles,
                    0.001,
                    false,
                )
                .unwrap()
                .unwrap();
            assert!(prepared.topology.is_some());
            assert_eq!(prepared.nodes, 54);
            cpu.step(0.001).unwrap();
            gpu.step_with_gpu_transfers(&pipeline, &device, &queue, 0.001)
                .unwrap();
            for (expected, actual) in cpu.particles.iter().zip(&gpu.particles) {
                assert!((expected.position - actual.position).norm() < 1e-5);
                assert!((expected.velocity - actual.velocity).norm() < 1e-4);
                assert!((expected.deformation - actual.deformation).norm() < 1e-5);
            }
            let adjacent_negative = [
                MpmParticle::new(
                    Vector3::new(-0.11, -0.08, -0.05),
                    0.04,
                    1_000.0,
                    MaterialModel::elastic(1_000.0, 0.2),
                ),
                MpmParticle::new(
                    Vector3::new(-0.08, -0.07, -0.04),
                    0.04,
                    1_000.0,
                    MaterialModel::elastic(1_000.0, 0.2),
                ),
            ];
            let mut cpu = MpmWorld::new(adjacent_negative.to_vec(), MpmParams::default()).unwrap();
            let mut gpu = MpmWorld::new(adjacent_negative.to_vec(), MpmParams::default()).unwrap();
            cpu.step(0.001).unwrap();
            gpu.step_with_gpu_transfers(&pipeline, &device, &queue, 0.001)
                .unwrap();
            for (expected, actual) in cpu.particles.iter().zip(&gpu.particles) {
                assert!((expected.position - actual.position).norm() < 1e-5);
                assert!((expected.velocity - actual.velocity).norm() < 1e-4);
            }
            let tiny = [MpmParticle::new(
                Vector3::repeat(0.5),
                1e-20,
                1_000.0,
                MaterialModel::elastic(1_000.0, 0.2),
            )];
            assert!(matches!(
                pipeline.transfer(&device, &queue, &tiny, &MpmParams::default(), &[], 0.001),
                Err(GpuMpmError::InvalidInput)
            ));
            let out_of_range = [MpmParticle::new(
                Vector3::repeat(300_000_000.0),
                0.04,
                1_000.0,
                MaterialModel::elastic(1_000.0, 0.2),
            )];
            assert!(matches!(
                pipeline.transfer(
                    &device,
                    &queue,
                    &out_of_range,
                    &MpmParams::default(),
                    &[],
                    0.001
                ),
                Err(GpuMpmError::InvalidInput)
            ));
        }
    }

    #[tokio::test]
    async fn gpu_rigid_obstacles_match_cpu_one_way_coupling() {
        let instance = wgpu::Instance::default();
        let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
        else {
            eprintln!("No WebGPU adapter available for MPM obstacle test");
            return;
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let pipeline = GpuMpmTransfers::new(&device);
        let mut moving_sphere = RigidObstacle::sphere(Vector3::new(0.5, 0.5, 0.5), 0.1);
        moving_sphere.linear_velocity = Vector3::new(1.0, 0.2, 0.0);
        moving_sphere.friction = 0.3;
        let mut moving_box = RigidObstacle::cuboid(
            Vector3::new(0.5, 0.5, 0.5),
            Vector3::repeat(0.1),
            UnitQuaternion::from_axis_angle(&Vector3::z_axis(), core::f64::consts::FRAC_PI_4),
        );
        moving_box.linear_velocity.x = 0.5;
        moving_box.angular_velocity.z = 0.5;
        moving_box.friction = 0.2;
        let mut moving_capsule =
            RigidObstacle::capsule(Vector3::repeat(0.5), 0.1, 0.1, UnitQuaternion::identity());
        moving_capsule.linear_velocity.x = 0.5;
        let mut moving_line =
            RigidObstacle::capsule(Vector3::repeat(0.5), 0.1, 0.0, UnitQuaternion::identity());
        moving_line.linear_velocity.x = 0.5;
        let mut moving_cylinder =
            RigidObstacle::cylinder(Vector3::repeat(0.5), 0.1, 0.1, UnitQuaternion::identity());
        moving_cylinder.linear_velocity.x = 0.5;
        let mut moving_cone =
            RigidObstacle::cone(Vector3::repeat(0.5), 0.1, 0.1, UnitQuaternion::identity());
        moving_cone.linear_velocity.x = 0.5;
        let tilted =
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), core::f64::consts::FRAC_PI_2);
        let mut tilted_capsule = RigidObstacle::capsule(Vector3::repeat(0.5), 0.1, 0.1, tilted);
        tilted_capsule.linear_velocity.z = 0.5;
        let mut tilted_cylinder = RigidObstacle::cylinder(Vector3::repeat(0.5), 0.1, 0.1, tilted);
        tilted_cylinder.linear_velocity.z = 0.5;
        let mut tilted_cone = RigidObstacle::cone(Vector3::repeat(0.5), 0.1, 0.1, tilted);
        tilted_cone.linear_velocity.y = 0.5;
        let triangle_vertices = [
            Vector3::new(-0.15, -0.15, 0.0),
            Vector3::new(0.15, -0.15, 0.0),
            Vector3::new(0.0, 0.15, 0.0),
        ];
        let mut moving_triangle = RigidObstacle::triangle_prism(
            Vector3::repeat(0.5),
            triangle_vertices,
            1e-4,
            UnitQuaternion::identity(),
        );
        moving_triangle.linear_velocity.z = 0.5;
        let mut tilted_triangle =
            RigidObstacle::triangle_prism(Vector3::repeat(0.5), triangle_vertices, 1e-4, tilted);
        tilted_triangle.linear_velocity.x = 0.5;
        let mut moving_convex = RigidObstacle::convex(
            Vector3::new(0.45, 0.45, 0.45),
            &[
                Vector3::zeros(),
                Vector3::x() * 0.2,
                Vector3::y() * 0.2,
                Vector3::z() * 0.2,
            ],
            &[
                -Vector3::x(),
                -Vector3::y(),
                -Vector3::z(),
                Vector3::repeat(1.0).normalize(),
            ],
            UnitQuaternion::identity(),
        );
        moving_convex.linear_velocity.x = -0.5;
        for (obstacle, position) in [
            (moving_sphere, Vector3::new(0.62, 0.5, 0.5)),
            (moving_box, Vector3::new(0.65, 0.5, 0.5)),
            (moving_capsule, Vector3::new(0.62, 0.5, 0.5)),
            (moving_line, Vector3::new(0.54, 0.5, 0.5)),
            (moving_cylinder, Vector3::new(0.62, 0.5, 0.5)),
            (moving_cone, Vector3::new(0.56, 0.5, 0.5)),
            (tilted_capsule, Vector3::new(0.5, 0.5, 0.62)),
            (tilted_cylinder, Vector3::new(0.5, 0.5, 0.62)),
            (tilted_cone, Vector3::new(0.5, 0.56, 0.5)),
            (moving_triangle, Vector3::new(0.5, 0.45, 0.54)),
            (tilted_triangle, Vector3::new(0.54, 0.45, 0.5)),
            (moving_convex.clone(), Vector3::new(0.43, 0.48, 0.48)),
            (moving_convex.clone(), Vector3::new(0.43, 0.42, 0.48)),
            (moving_convex, Vector3::new(0.69, 0.47, 0.47)),
        ] {
            let particle = MpmParticle::new(
                position,
                0.03,
                1_000.0,
                MaterialModel::elastic(1_000.0, 0.2),
            );
            let params = MpmParams {
                gravity: Vector3::zeros(),
                ..MpmParams::default()
            };
            let mut cpu = MpmWorld::new(vec![particle.clone()], params.clone()).unwrap();
            let mut gpu = MpmWorld::new(vec![particle], params).unwrap();
            cpu.set_obstacles(vec![obstacle.clone()]).unwrap();
            gpu.set_obstacles(vec![obstacle.clone()]).unwrap();
            cpu.step(0.002).unwrap();
            gpu.step_with_gpu_transfers(&pipeline, &device, &queue, 0.002)
                .unwrap();
            assert!(
                (cpu.particles[0].position - gpu.particles[0].position).norm() < 1e-5,
                "shape={:?}, cpu={:?}, gpu={:?}",
                obstacle.shape,
                cpu.particles[0].position,
                gpu.particles[0].position,
            );
            assert!(
                (cpu.particles[0].velocity - gpu.particles[0].velocity).norm() < 1e-4,
                "shape={:?}, cpu={:?}, gpu={:?}",
                obstacle.shape,
                cpu.particles[0].velocity,
                gpu.particles[0].velocity,
            );
            assert_eq!(gpu.obstacles, vec![obstacle]);
        }

        let particles = [0.42, 0.82]
            .map(|x| {
                MpmParticle::new(
                    Vector3::new(x, 0.5, 0.5),
                    0.03,
                    1_000.0,
                    MaterialModel::elastic(1_000.0, 0.2),
                )
            })
            .to_vec();
        let mut first = RigidObstacle::sphere(Vector3::new(0.3, 0.5, 0.5), 0.1);
        let mut second = RigidObstacle::sphere(Vector3::new(0.7, 0.5, 0.5), 0.1);
        first.linear_velocity.x = 1.0;
        second.linear_velocity.x = 0.5;
        let params = MpmParams {
            gravity: Vector3::zeros(),
            ..MpmParams::default()
        };
        let mut cpu = MpmWorld::new(particles.clone(), params.clone()).unwrap();
        let mut gpu = MpmWorld::new(particles, params).unwrap();
        cpu.set_obstacles(vec![first.clone(), second.clone()])
            .unwrap();
        gpu.set_obstacles(vec![first, second]).unwrap();
        cpu.step(0.001).unwrap();
        gpu.step_with_gpu_transfers(&pipeline, &device, &queue, 0.001)
            .unwrap();
        for (expected, actual) in cpu.particles.iter().zip(&gpu.particles) {
            assert!((expected.position - actual.position).norm() < 1e-5);
            assert!((expected.velocity - actual.velocity).norm() < 1e-4);
        }

        let make_convex = |center: Vector3<f64>, speed: f64| {
            let mut shape = RigidObstacle::convex(
                center,
                &[
                    Vector3::zeros(),
                    Vector3::x() * 0.15,
                    Vector3::y() * 0.15,
                    Vector3::z() * 0.15,
                ],
                &[
                    -Vector3::x(),
                    -Vector3::y(),
                    -Vector3::z(),
                    Vector3::repeat(1.0).normalize(),
                ],
                UnitQuaternion::identity(),
            );
            shape.linear_velocity.x = speed;
            shape
        };
        let obstacles = vec![
            make_convex(Vector3::new(0.3, 0.45, 0.45), -0.5),
            make_convex(Vector3::new(0.7, 0.45, 0.45), -1.0),
        ];
        let mut packed_planes = Vec::new();
        assert_eq!(
            pack_obstacle(&obstacles[0], &mut packed_planes)
                .unwrap()
                .convex_range,
            [0, 4, 0, 0]
        );
        assert_eq!(
            pack_obstacle(&obstacles[1], &mut packed_planes)
                .unwrap()
                .convex_range,
            [4, 4, 0, 0]
        );
        let particles = [0.28, 0.68]
            .map(|x| {
                MpmParticle::new(
                    Vector3::new(x, 0.48, 0.48),
                    0.03,
                    1_000.0,
                    MaterialModel::elastic(1_000.0, 0.2),
                )
            })
            .to_vec();
        let params = MpmParams {
            gravity: Vector3::zeros(),
            ..MpmParams::default()
        };
        let mut cpu = MpmWorld::new(particles.clone(), params.clone()).unwrap();
        let mut gpu = MpmWorld::new(particles, params).unwrap();
        cpu.set_obstacles(obstacles.clone()).unwrap();
        gpu.set_obstacles(obstacles).unwrap();
        cpu.step(0.001).unwrap();
        gpu.step_with_gpu_transfers(&pipeline, &device, &queue, 0.001)
            .unwrap();
        for (expected, actual) in cpu.particles.iter().zip(&gpu.particles) {
            assert!((expected.position - actual.position).norm() < 1e-5);
            assert!((expected.velocity - actual.velocity).norm() < 1e-4);
        }
    }

    #[test]
    fn convex_gpu_pack_rejects_more_planes_than_shader_capacity() {
        let vertices = [Vector3::zeros(), Vector3::x(), Vector3::y(), Vector3::z()];
        let normals = vec![Vector3::x(); 129];
        let obstacle = RigidObstacle::convex(
            Vector3::zeros(),
            &vertices,
            &normals,
            UnitQuaternion::identity(),
        );
        assert!(obstacle.is_valid());
        assert!(matches!(
            pack_obstacle(&obstacle, &mut Vec::new()),
            Err(GpuMpmError::Capacity)
        ));
    }

    #[tokio::test]
    async fn gpu_finite_ground_matches_cpu_particle_projection() {
        let instance = wgpu::Instance::default();
        let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
        else {
            eprintln!("No WebGPU adapter available for MPM ground test");
            return;
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let ground = RigidObstacle::ground(
            Vector3::zeros(),
            nalgebra::Vector2::repeat(1.0),
            UnitQuaternion::identity(),
        );
        let mut particle = MpmParticle::new(
            Vector3::new(0.5, 0.5, 0.05),
            0.04,
            1000.0,
            MaterialModel::elastic(1000.0, 0.2),
        );
        particle.velocity.z = -10.0;
        let params = MpmParams {
            gravity: Vector3::zeros(),
            ..MpmParams::default()
        };
        let mut cpu = MpmWorld::new(vec![particle.clone()], params.clone()).unwrap();
        let mut gpu = MpmWorld::new(vec![particle], params).unwrap();
        cpu.set_obstacles(vec![ground.clone()]).unwrap();
        gpu.set_obstacles(vec![ground]).unwrap();
        cpu.step(0.002).unwrap();
        gpu.step_with_gpu_transfers(&GpuMpmTransfers::new(&device), &device, &queue, 0.002)
            .unwrap();
        assert!(cpu.particles[0].position.z >= cpu.particles[0].radius - 1e-12);
        assert!((cpu.particles[0].position - gpu.particles[0].position).norm() < 1e-5);
        assert!((cpu.particles[0].velocity - gpu.particles[0].velocity).norm() < 1e-4);
    }

    #[tokio::test]
    async fn gpu_transfer_accepts_emitted_and_removed_chunks() {
        let instance = wgpu::Instance::default();
        let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
        else {
            eprintln!("No WebGPU adapter available for MPM emitter test");
            return;
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let pipeline = GpuMpmTransfers::new(&device);
        let params = MpmParams::default();
        let mut cpu = MpmWorld::new(Vec::new(), params.clone()).unwrap();
        let mut gpu = MpmWorld::new(Vec::new(), params).unwrap();
        let mut emitter = BoxEmitter::new(
            Vector3::new(0.4, 0.5, 0.5),
            [2, 1, 1],
            Vector3::repeat(0.08),
            0.03,
            1_000.0,
            MaterialModel::elastic(1_000.0, 0.2),
        );
        let first_cpu = emitter.emit(&mut cpu).unwrap();
        let first_gpu = emitter.emit(&mut gpu).unwrap();
        assert_eq!(first_cpu, first_gpu);
        cpu.step(0.001).unwrap();
        gpu.step_with_gpu_transfers(&pipeline, &device, &queue, 0.001)
            .unwrap();
        assert_eq!(cpu.remove_chunk(first_cpu).unwrap(), 2);
        assert_eq!(gpu.remove_chunk(first_gpu).unwrap(), 2);
        emitter.center.x = 0.7;
        let second_cpu = emitter.emit(&mut cpu).unwrap();
        let second_gpu = emitter.emit(&mut gpu).unwrap();
        assert_eq!(second_cpu, second_gpu);
        cpu.step(0.002).unwrap();
        gpu.step_with_gpu_transfers(&pipeline, &device, &queue, 0.002)
            .unwrap();
        for (expected, actual) in cpu.particles.iter().zip(&gpu.particles) {
            assert_eq!(expected.chunk_id(), actual.chunk_id());
            assert!((expected.position - actual.position).norm() < 1e-5);
            assert!((expected.velocity - actual.velocity).norm() < 1e-4);
        }
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn dx12_backend_runs_mpm_transfers() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::DX12,
            ..Default::default()
        });
        let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
        else {
            eprintln!("DX12 adapter unavailable for MPM transfer test");
            return;
        };
        let info = adapter.get_info();
        eprintln!("DX12 MPM adapter: {} ({:?})", info.name, info.device_type);
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let particle = MpmParticle::new(
            Vector3::new(0.5, 0.5, 0.5),
            0.04,
            1_000.0,
            MaterialModel::elastic(1_000.0, 0.2),
        );
        let mut world = MpmWorld::new(vec![particle], MpmParams::default()).unwrap();
        let pipeline = GpuMpmTransfers::new(&device);
        world
            .step_with_gpu_transfers(&pipeline, &device, &queue, 0.001)
            .unwrap();
        assert!((world.particles[0].velocity.z + 0.00981).abs() < 1e-5);

        let particle = MpmParticle::new(
            Vector3::new(0.62, 0.5, 0.5),
            0.03,
            1_000.0,
            MaterialModel::elastic(1_000.0, 0.2),
        );
        let params = MpmParams {
            gravity: Vector3::zeros(),
            ..MpmParams::default()
        };
        let mut obstacle = RigidObstacle::sphere(Vector3::repeat(0.5), 0.1);
        obstacle.linear_velocity.x = 1.0;
        let mut cpu = MpmWorld::new(vec![particle.clone()], params.clone()).unwrap();
        let mut gpu = MpmWorld::new(vec![particle], params).unwrap();
        cpu.set_obstacles(vec![obstacle.clone()]).unwrap();
        gpu.set_obstacles(vec![obstacle]).unwrap();
        cpu.step(0.001).unwrap();
        gpu.step_with_gpu_transfers(&pipeline, &device, &queue, 0.001)
            .unwrap();
        assert!((cpu.particles[0].velocity - gpu.particles[0].velocity).norm() < 1e-4);

        let mut first = MpmParticle::new(
            Vector3::repeat(0.5),
            0.04,
            1_000.0,
            MaterialModel::elastic(1_000.0, 0.2),
        );
        first.velocity.x = 1.0;
        let mut second = first.clone();
        second.velocity.x = -1.0;
        second.transfer_color = 1;
        let mut colored = MpmWorld::new(
            vec![first, second],
            MpmParams {
                gravity: Vector3::zeros(),
                ..MpmParams::default()
            },
        )
        .unwrap();
        colored
            .step_with_gpu_transfers(&pipeline, &device, &queue, 0.001)
            .unwrap();
        assert!((colored.particles[0].velocity.x - 1.0).abs() < 1e-4);
        assert!((colored.particles[1].velocity.x + 1.0).abs() < 1e-4);
    }
}
