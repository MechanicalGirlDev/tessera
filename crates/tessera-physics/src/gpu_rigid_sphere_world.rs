//! Owned GPU-resident sphere world with a single-step simulation API.
//!
//! Rigid state stays on the device between steps. Small environments use an
//! exhaustive GPU contact list. Bounded single worlds retain LBVH candidates
//! through contact solving; other large worlds read candidate pairs back.

use core::mem::size_of;
use core::ops::Range;
use std::collections::BTreeMap;

use crate::gpu_broad_phase::GpuPair;
use crate::gpu_lbvh::{GpuLbvh, GpuLbvhResidentPairs};
use crate::gpu_rigid_ball_joint::{
    GpuRigidAxisMotor, GpuRigidAxisServo, GpuRigidBallJoint, GpuRigidBallJointError,
    GpuRigidBallJointSolver, GpuRigidFixedJoint, GpuRigidPrismaticJoint, GpuRigidPrismaticLimit,
    GpuRigidRevoluteJoint, GpuRigidRevoluteLimit, GpuRigidTemporalJointParams,
};
use crate::gpu_rigid_contact_transport::{GpuRigidContactTransport, GpuRigidContactTransportError};
use crate::gpu_rigid_shape::GpuRigidShape;
use crate::gpu_rigid_sphere_contact::{
    GpuRigidCandidateContacts, GpuRigidCollisionGroups, GpuRigidSphereContactError,
    GpuRigidSphereContactReadback, GpuRigidSphereContacts,
};
use crate::gpu_rigid_sphere_solver::{
    GpuRigidCandidateImpulseCache, GpuRigidContactImpulseReadback, GpuRigidSphereImpulseCache,
    GpuRigidSphereSolveError, GpuRigidSphereSolveParams, GpuRigidSphereSolver,
    GpuRigidTemporalSolveParams,
};
use crate::gpu_rigid_state::{
    GpuRigidBodyForces, GpuRigidBodyState, GpuRigidStateError, GpuRigidStateSession,
};
use crate::material::ColliderMaterial;
use crate::sleep::SleepSettings;

// Bound serial island work while removing pair readback for small scenes.
const EXHAUSTIVE_PAIR_BODY_LIMIT: usize = 16;
const MAX_EXHAUSTIVE_PAIRS: usize = 65_536;
const MAX_ENCODED_SUBSTEPS: u32 = 1024;

enum TopologyEdit {
    Insert(usize, Box<GpuRigidBodyState>),
    Remove(usize),
    InsertRange(usize, Vec<GpuRigidBodyState>),
    RemoveRange(Range<usize>),
}

type RemovedEnvironment = (Option<Vec<GpuRigidBodyState>>, Option<Vec<GpuRigidShape>>);

fn remap_joint_bodies(edit: &TopologyEdit, body_a: u32, body_b: u32) -> Option<(u32, u32)> {
    let remap = |body: u32| match edit {
        TopologyEdit::Insert(index, _) if body as usize >= *index => body.checked_add(1),
        TopologyEdit::Remove(index) if body as usize == *index => None,
        TopologyEdit::Remove(index) if body as usize > *index => Some(body - 1),
        TopologyEdit::InsertRange(index, states) if body as usize >= *index => {
            body.checked_add(u32::try_from(states.len()).ok()?)
        }
        TopologyEdit::RemoveRange(range) if range.contains(&(body as usize)) => None,
        TopologyEdit::RemoveRange(range) if body as usize >= range.end => {
            body.checked_sub(u32::try_from(range.len()).ok()?)
        }
        _ => Some(body),
    };
    Some((remap(body_a)?, remap(body_b)?))
}

/// World coefficients and optional finite ground plane.
#[derive(Clone, Copy, Debug)]
pub struct GpuRigidSphereWorldConfig {
    /// World-space gravitational acceleration.
    pub gravity: [f32; 3],
    /// Half-width of the square ground plane; `None` disables it.
    pub ground_half_extent: Option<f32>,
    /// Default contact coefficients, bias, and solver iterations.
    pub solve: GpuRigidSphereSolveParams,
    /// Automatic sleep thresholds.
    pub sleep: SleepSettings,
}

impl Default for GpuRigidSphereWorldConfig {
    fn default() -> Self {
        Self {
            gravity: [0.0, 0.0, -9.81],
            ground_half_extent: Some(10.0),
            solve: GpuRigidSphereSolveParams::default(),
            sleep: SleepSettings::default(),
        }
    }
}

impl GpuRigidSphereWorldConfig {
    /// Whether every coefficient can be represented by the resident kernels.
    pub fn is_valid(self) -> bool {
        let limit = f32::MAX.sqrt();
        self.gravity.iter().all(|value| value.is_finite())
            && self
                .ground_half_extent
                .is_none_or(|half| half.is_finite() && half > 0.0)
            && self.solve.friction.is_finite()
            && (0.0..=limit).contains(&self.solve.friction)
            && self.solve.restitution.is_finite()
            && (0.0..=limit).contains(&self.solve.restitution)
            && self.solve.bias_factor.is_finite()
            && (0.0..=1.0).contains(&self.solve.bias_factor)
            && self.solve.iterations > 0
            && self.sleep.is_valid()
            && (self.sleep.linear_velocity_threshold as f32).is_finite()
            && (self.sleep.angular_velocity_threshold as f32).is_finite()
            && (self.sleep.linear_velocity_threshold as f32)
                .powi(2)
                .is_finite()
            && (self.sleep.angular_velocity_threshold as f32)
                .powi(2)
                .is_finite()
            && (self.sleep.time_threshold as f32).is_finite()
            && (self.sleep.time_threshold as f32) > 0.0
    }
}

struct NewWorldOptions<'a> {
    config: GpuRigidSphereWorldConfig,
    previous_contacts: Option<&'a GpuRigidSphereContacts>,
}

/// Invalid input, GPU pipeline failure, or a world requiring reset.
#[derive(Debug, thiserror::Error)]
pub enum GpuRigidSphereWorldError {
    /// Configuration, body index, or timestep is invalid.
    #[error("invalid GPU rigid sphere world input")]
    InvalidInput,
    /// A step stopped after preparing or submitting GPU work; reset before reuse.
    #[error("GPU rigid sphere world requires reset after an incomplete step")]
    Faulted,
    /// GPU state creation, update, or readback failed.
    #[error(transparent)]
    State(#[from] GpuRigidStateError),
    /// Contact generation or candidate refresh failed.
    #[error(transparent)]
    Contact(#[from] GpuRigidSphereContactError),
    /// Impulse solving exceeded the selected device's capacity.
    #[error(transparent)]
    Solver(#[from] GpuRigidSphereSolveError),
    /// A resident joint constraint is invalid or exceeds device capacity.
    #[error(transparent)]
    BallJoint(#[from] GpuRigidBallJointError),
    /// Temporal contact anchor storage exceeds device capacity.
    #[error(transparent)]
    ContactTransport(#[from] GpuRigidContactTransportError),
}

/// Failure while building or reading a combined ray and point query batch.
#[derive(Debug, thiserror::Error)]
pub enum GpuSceneQueryError {
    /// Environment count or an environment-local query is invalid.
    #[error("invalid GPU scene query input")]
    InvalidInput,
    /// Current-state scene bounds or the scene tree could not be built.
    #[error(transparent)]
    SceneIndex(#[from] GpuRigidSphereContactError),
    /// Ray query input, encoding, or readback failed.
    #[error(transparent)]
    Ray(#[from] crate::gpu_ray_query::GpuRayQueryError),
    /// Point query input, encoding, or readback failed.
    #[error(transparent)]
    Point(#[from] crate::gpu_point_query::GpuPointQueryError),
}

/// Sphere-only 3D world that keeps body state, contacts, and sleep on the GPU.
///
/// Small environments encode integration, contacts, solve, and sleep in one GPU
/// submission. Bounded larger single worlds retain LBVH pairs on the GPU;
/// remaining large worlds read compact pairs back to build islands.
/// Use `readback` only when CPU state is actually needed.
#[derive(Debug)]
pub struct GpuRigidSphereWorld {
    device: wgpu::Device,
    queue: wgpu::Queue,
    state: GpuRigidStateSession,
    environment_ids: Vec<u32>,
    radii: Vec<f32>,
    shapes: Option<Vec<GpuRigidShape>>,
    body_materials: Vec<Option<ColliderMaterial>>,
    ground_material: Option<ColliderMaterial>,
    topology_mutable: bool,
    contacts: GpuRigidSphereContacts,
    broad_phase: Option<GpuLbvh>,
    query_broad_phase: std::sync::OnceLock<GpuLbvh>,
    solver: GpuRigidSphereSolver,
    ball_joints: Option<GpuRigidBallJointSolver>,
    impulse_cache: GpuRigidSphereImpulseCache,
    resident_impulse_cache: GpuRigidCandidateImpulseCache,
    last_resident_contacts: Option<(usize, GpuLbvhResidentPairs, GpuRigidCandidateContacts)>,
    config: GpuRigidSphereWorldConfig,
    max_linear_speed: Option<f32>,
    faulted: bool,
    temporal_active: bool,
}

/// Resident primitive world using the shared rigid solver.
pub type GpuRigidPrimitiveWorld = GpuRigidSphereWorld;

impl GpuRigidSphereWorld {
    /// Allocate a resident world on a caller-selected wgpu device.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        states: &[GpuRigidBodyState],
        radii: &[f32],
        config: GpuRigidSphereWorldConfig,
    ) -> Result<Self, GpuRigidSphereWorldError> {
        let mut world =
            Self::new_grouped(device, queue, states, radii, &vec![0; states.len()], config)?;
        world.topology_mutable = true;
        Ok(world)
    }

    /// Allocate independent environments in one GPU state and solver dispatch.
    /// Bodies only contact others with the same ID. Gravity and ground are shared.
    pub fn new_grouped(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        states: &[GpuRigidBodyState],
        radii: &[f32],
        environment_ids: &[u32],
        config: GpuRigidSphereWorldConfig,
    ) -> Result<Self, GpuRigidSphereWorldError> {
        Self::new_grouped_inner(device, queue, states, radii, None, environment_ids, config)
    }

    /// Allocate a resident primitive world on a caller-selected GPU.
    pub fn new_primitives(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        states: &[GpuRigidBodyState],
        shapes: &[GpuRigidShape],
        config: GpuRigidSphereWorldConfig,
    ) -> Result<Self, GpuRigidSphereWorldError> {
        let mut world = Self::new_grouped_primitives(
            device,
            queue,
            states,
            shapes,
            &vec![0; states.len()],
            config,
        )?;
        world.topology_mutable = true;
        Ok(world)
    }

    /// Allocate independent primitive environments in shared GPU buffers.
    pub fn new_grouped_primitives(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        states: &[GpuRigidBodyState],
        shapes: &[GpuRigidShape],
        environment_ids: &[u32],
        config: GpuRigidSphereWorldConfig,
    ) -> Result<Self, GpuRigidSphereWorldError> {
        let radii = shapes
            .iter()
            .map(|shape| {
                shape
                    .bounding_radius()
                    .ok_or(GpuRigidSphereWorldError::InvalidInput)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::new_grouped_inner(
            device,
            queue,
            states,
            &radii,
            Some(shapes),
            environment_ids,
            config,
        )
    }

    fn new_grouped_inner(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        states: &[GpuRigidBodyState],
        radii: &[f32],
        shapes: Option<&[GpuRigidShape]>,
        environment_ids: &[u32],
        config: GpuRigidSphereWorldConfig,
    ) -> Result<Self, GpuRigidSphereWorldError> {
        if !config.is_valid()
            || environment_ids.len() != states.len()
            || radii.len() != states.len()
            || shapes.is_some_and(|shapes| shapes.len() != states.len())
        {
            return Err(GpuRigidSphereWorldError::InvalidInput);
        }
        let state = GpuRigidStateSession::new(device, queue, states)?;
        Self::new_grouped_with_state(
            device,
            queue,
            state,
            radii,
            shapes,
            environment_ids,
            NewWorldOptions {
                config,
                previous_contacts: None,
            },
        )
    }

    fn new_grouped_with_state(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        state: GpuRigidStateSession,
        radii: &[f32],
        shapes: Option<&[GpuRigidShape]>,
        environment_ids: &[u32],
        options: NewWorldOptions<'_>,
    ) -> Result<Self, GpuRigidSphereWorldError> {
        let NewWorldOptions {
            config,
            previous_contacts,
        } = options;
        if !config.is_valid()
            || environment_ids.len() != state.len()
            || radii.len() != state.len()
            || shapes.is_some_and(|shapes| shapes.len() != state.len())
        {
            return Err(GpuRigidSphereWorldError::InvalidInput);
        }
        let mut groups: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
        for (index, environment) in environment_ids.iter().copied().enumerate() {
            groups.entry(environment).or_default().push(index as u32);
        }
        let pair_count = groups.values().try_fold(0usize, |total, bodies| {
            let pairs = bodies.len().checked_mul(bodies.len().saturating_sub(1))? / 2;
            total.checked_add(pairs)
        });
        let exhaustive_pairs = groups
            .values()
            .all(|bodies| bodies.len() <= EXHAUSTIVE_PAIR_BODY_LIMIT)
            && pair_count.is_some_and(|count| count <= MAX_EXHAUSTIVE_PAIRS);
        let mut pairs = Vec::new();
        if exhaustive_pairs {
            for bodies in groups.values() {
                for (offset, &a) in bodies.iter().enumerate() {
                    for &b in &bodies[(offset + 1)..] {
                        pairs.push(GpuPair { a, b });
                    }
                }
            }
            pairs.sort_unstable_by_key(|pair| (pair.a, pair.b));
        }
        let contacts = if let Some(shapes) = shapes {
            GpuRigidSphereContacts::new_with_shapes_reusing(
                device,
                &state,
                shapes,
                &pairs,
                config.ground_half_extent,
                previous_contacts,
            )?
        } else {
            GpuRigidSphereContacts::new(device, &state, radii, &pairs, config.ground_half_extent)?
        };
        let body_count = state.len();
        Ok(Self {
            device: device.clone(),
            queue: queue.clone(),
            state,
            environment_ids: environment_ids.to_vec(),
            radii: radii.to_vec(),
            shapes: shapes.map(<[GpuRigidShape]>::to_vec),
            body_materials: vec![None; body_count],
            ground_material: None,
            topology_mutable: false,
            contacts,
            broad_phase: (!exhaustive_pairs).then(|| GpuLbvh::new(device)),
            query_broad_phase: std::sync::OnceLock::new(),
            solver: GpuRigidSphereSolver::new(device),
            ball_joints: None,
            impulse_cache: GpuRigidSphereImpulseCache::default(),
            resident_impulse_cache: GpuRigidCandidateImpulseCache::default(),
            last_resident_contacts: None,
            config,
            max_linear_speed: None,
            faulted: false,
            temporal_active: false,
        })
    }

    /// Number of bodies with stable indices in this world.
    pub fn len(&self) -> usize {
        self.state.len()
    }

    /// Whether this world contains no bodies.
    pub fn is_empty(&self) -> bool {
        self.state.is_empty()
    }

    /// GPU-resident rigid state for rendering or additional compute passes.
    ///
    /// The buffer is replaced when bodies are appended or removed. Recreate any
    /// bind group that references it after a topology edit.
    pub fn state_buffer(&self) -> &wgpu::Buffer {
        self.state.state_buffer()
    }

    /// Device owning the resident state and collider buffers.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Queue that orders submissions against the resident rigid state.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Whether a failed step requires a complete body reset.
    pub fn is_faulted(&self) -> bool {
        self.faulted
    }

    /// Number of potential pairs: all pairs in small scenes, LBVH candidates otherwise.
    pub fn candidate_pair_count(&self) -> usize {
        self.last_resident_contacts
            .as_ref()
            .map_or_else(|| self.contacts.pairs().len(), |(count, _, _)| *count)
    }

    /// Whether steps avoid broad-phase readback through an exhaustive pair list.
    pub fn uses_exhaustive_pairs(&self) -> bool {
        self.broad_phase.is_none()
    }

    /// Current world coefficients and ground configuration.
    pub fn config(&self) -> GpuRigidSphereWorldConfig {
        self.config
    }

    /// Optional world-space linear speed cap for dynamic bodies, in metres per second.
    pub fn max_linear_speed(&self) -> Option<f32> {
        self.max_linear_speed
    }

    /// Set a speed cap applied during integration and after contact and joint impulses.
    pub fn set_max_linear_speed(
        &mut self,
        max_speed: Option<f32>,
    ) -> Result<(), GpuRigidSphereWorldError> {
        if max_speed.is_some_and(|speed| !speed.is_finite() || speed <= 0.0) {
            return Err(GpuRigidSphereWorldError::InvalidInput);
        }
        self.max_linear_speed = max_speed;
        Ok(())
    }

    /// Sphere radius or conservative primitive radius in current dense index order.
    pub fn radius(&self, index: usize) -> Option<f32> {
        self.radii.get(index).copied()
    }

    /// Exact collision shape in current dense index order.
    pub fn shape(&self, index: usize) -> Option<GpuRigidShape> {
        if let Some(shapes) = &self.shapes {
            shapes.get(index).cloned()
        } else {
            self.radii
                .get(index)
                .copied()
                .map(|radius| GpuRigidShape::Sphere { radius })
        }
    }

    /// Resident point-to-point joints in stable insertion order.
    pub fn ball_joints(&self) -> &[GpuRigidBallJoint] {
        self.ball_joints
            .as_ref()
            .map_or(&[], GpuRigidBallJointSolver::joints)
    }

    /// Resident fixed joints in stable insertion order.
    pub fn fixed_joints(&self) -> &[GpuRigidFixedJoint] {
        self.ball_joints
            .as_ref()
            .map_or(&[], GpuRigidBallJointSolver::fixed_joints)
    }

    /// Resident revolute joints in stable insertion order.
    pub fn revolute_joints(&self) -> &[GpuRigidRevoluteJoint] {
        self.ball_joints
            .as_ref()
            .map_or(&[], GpuRigidBallJointSolver::revolute_joints)
    }

    /// Resident prismatic joints in stable insertion order.
    pub fn prismatic_joints(&self) -> &[GpuRigidPrismaticJoint] {
        self.ball_joints
            .as_ref()
            .map_or(&[], GpuRigidBallJointSolver::prismatic_joints)
    }

    /// Current velocity motor on a revolute joint.
    pub fn revolute_motor(
        &self,
        index: usize,
    ) -> Result<Option<GpuRigidAxisMotor>, GpuRigidSphereWorldError> {
        self.ball_joints
            .as_ref()
            .and_then(|solver| solver.revolute_motor(index))
            .ok_or(GpuRigidSphereWorldError::InvalidInput)
    }

    /// Set or clear a revolute joint's velocity motor.
    pub fn set_revolute_motor(
        &mut self,
        index: usize,
        motor: Option<GpuRigidAxisMotor>,
    ) -> Result<(), GpuRigidSphereWorldError> {
        if self.faulted {
            return Err(GpuRigidSphereWorldError::Faulted);
        }
        self.ball_joints
            .as_mut()
            .ok_or(GpuRigidSphereWorldError::InvalidInput)?
            .set_revolute_motor(&self.queue, index, motor)?;
        Ok(())
    }

    /// Current position and velocity servo on a revolute joint.
    pub fn revolute_servo(
        &self,
        index: usize,
    ) -> Result<Option<GpuRigidAxisServo>, GpuRigidSphereWorldError> {
        self.ball_joints
            .as_ref()
            .and_then(|solver| solver.revolute_servo(index))
            .ok_or(GpuRigidSphereWorldError::InvalidInput)
    }

    /// Set or clear a revolute joint's position and velocity servo.
    pub fn set_revolute_servo(
        &mut self,
        index: usize,
        servo: Option<GpuRigidAxisServo>,
    ) -> Result<(), GpuRigidSphereWorldError> {
        if self.faulted {
            return Err(GpuRigidSphereWorldError::Faulted);
        }
        self.ball_joints
            .as_mut()
            .ok_or(GpuRigidSphereWorldError::InvalidInput)?
            .set_revolute_servo(&self.queue, index, servo)?;
        Ok(())
    }

    /// Current continuous angle limits on a revolute joint.
    pub fn revolute_limit(
        &self,
        index: usize,
    ) -> Result<Option<GpuRigidRevoluteLimit>, GpuRigidSphereWorldError> {
        self.ball_joints
            .as_ref()
            .and_then(|solver| solver.revolute_limit(index))
            .ok_or(GpuRigidSphereWorldError::InvalidInput)
    }

    /// Set or clear a revolute joint's continuous angle limits.
    pub fn set_revolute_limit(
        &mut self,
        index: usize,
        limit: Option<GpuRigidRevoluteLimit>,
    ) -> Result<(), GpuRigidSphereWorldError> {
        if self.faulted {
            return Err(GpuRigidSphereWorldError::Faulted);
        }
        self.ball_joints
            .as_mut()
            .ok_or(GpuRigidSphereWorldError::InvalidInput)?
            .set_revolute_limit(&self.queue, index, limit)?;
        Ok(())
    }

    /// Read the continuous angle of a revolute joint in radians.
    pub fn readback_revolute_angle(&self, index: usize) -> Result<f32, GpuRigidSphereWorldError> {
        self.ball_joints
            .as_ref()
            .ok_or(GpuRigidSphereWorldError::InvalidInput)?
            .readback_revolute_angle(&self.device, &self.queue, index)
            .map_err(Into::into)
    }

    /// Current velocity motor on a prismatic joint.
    pub fn prismatic_motor(
        &self,
        index: usize,
    ) -> Result<Option<GpuRigidAxisMotor>, GpuRigidSphereWorldError> {
        self.ball_joints
            .as_ref()
            .and_then(|solver| solver.prismatic_motor(index))
            .ok_or(GpuRigidSphereWorldError::InvalidInput)
    }

    /// Set or clear a prismatic joint's velocity motor.
    pub fn set_prismatic_motor(
        &mut self,
        index: usize,
        motor: Option<GpuRigidAxisMotor>,
    ) -> Result<(), GpuRigidSphereWorldError> {
        if self.faulted {
            return Err(GpuRigidSphereWorldError::Faulted);
        }
        self.ball_joints
            .as_mut()
            .ok_or(GpuRigidSphereWorldError::InvalidInput)?
            .set_prismatic_motor(&self.queue, index, motor)?;
        Ok(())
    }

    /// Current position and velocity servo on a prismatic joint.
    pub fn prismatic_servo(
        &self,
        index: usize,
    ) -> Result<Option<GpuRigidAxisServo>, GpuRigidSphereWorldError> {
        self.ball_joints
            .as_ref()
            .and_then(|solver| solver.prismatic_servo(index))
            .ok_or(GpuRigidSphereWorldError::InvalidInput)
    }

    /// Set or clear a prismatic joint's position and velocity servo.
    pub fn set_prismatic_servo(
        &mut self,
        index: usize,
        servo: Option<GpuRigidAxisServo>,
    ) -> Result<(), GpuRigidSphereWorldError> {
        if self.faulted {
            return Err(GpuRigidSphereWorldError::Faulted);
        }
        self.ball_joints
            .as_mut()
            .ok_or(GpuRigidSphereWorldError::InvalidInput)?
            .set_prismatic_servo(&self.queue, index, servo)?;
        Ok(())
    }

    /// Current displacement limits on a prismatic joint.
    pub fn prismatic_limit(
        &self,
        index: usize,
    ) -> Result<Option<GpuRigidPrismaticLimit>, GpuRigidSphereWorldError> {
        self.ball_joints
            .as_ref()
            .and_then(|solver| solver.prismatic_limit(index))
            .ok_or(GpuRigidSphereWorldError::InvalidInput)
    }

    /// Set or clear a prismatic joint's displacement limits.
    pub fn set_prismatic_limit(
        &mut self,
        index: usize,
        limit: Option<GpuRigidPrismaticLimit>,
    ) -> Result<(), GpuRigidSphereWorldError> {
        if self.faulted {
            return Err(GpuRigidSphereWorldError::Faulted);
        }
        self.ball_joints
            .as_mut()
            .ok_or(GpuRigidSphereWorldError::InvalidInput)?
            .set_prismatic_limit(&self.queue, index, limit)?;
        Ok(())
    }

    /// Replace point-to-point joints without transferring rigid state to the CPU.
    ///
    /// Jointed bodies must belong to the same environment. Topology edits preserve
    /// surviving joints and remove joints attached to a deleted body.
    pub fn set_ball_joints(
        &mut self,
        joints: &[GpuRigidBallJoint],
    ) -> Result<(), GpuRigidSphereWorldError> {
        let fixed_joints = self.fixed_joints().to_vec();
        self.set_joints(joints, &fixed_joints)
    }

    /// Replace fixed joints while retaining current point-to-point joints.
    pub fn set_fixed_joints(
        &mut self,
        joints: &[GpuRigidFixedJoint],
    ) -> Result<(), GpuRigidSphereWorldError> {
        let ball_joints = self.ball_joints().to_vec();
        self.set_joints(&ball_joints, joints)
    }

    /// Replace revolute joints while retaining ball and fixed joints.
    pub fn set_revolute_joints(
        &mut self,
        joints: &[GpuRigidRevoluteJoint],
    ) -> Result<(), GpuRigidSphereWorldError> {
        let ball_joints = self.ball_joints().to_vec();
        let fixed_joints = self.fixed_joints().to_vec();
        self.set_all_joints(&ball_joints, &fixed_joints, joints)
    }

    /// Replace prismatic joints while retaining other joint types.
    pub fn set_prismatic_joints(
        &mut self,
        joints: &[GpuRigidPrismaticJoint],
    ) -> Result<(), GpuRigidSphereWorldError> {
        let ball_joints = self.ball_joints().to_vec();
        let fixed_joints = self.fixed_joints().to_vec();
        let revolute_joints = self.revolute_joints().to_vec();
        self.set_all_joints_with_prismatic(&ball_joints, &fixed_joints, &revolute_joints, joints)
    }

    /// Replace all GPU-resident joints as one topology transaction.
    pub fn set_joints(
        &mut self,
        ball_joints: &[GpuRigidBallJoint],
        fixed_joints: &[GpuRigidFixedJoint],
    ) -> Result<(), GpuRigidSphereWorldError> {
        let revolute_joints = self.revolute_joints().to_vec();
        self.set_all_joints(ball_joints, fixed_joints, &revolute_joints)
    }

    /// Replace all GPU-resident joints as one topology transaction.
    pub fn set_all_joints(
        &mut self,
        ball_joints: &[GpuRigidBallJoint],
        fixed_joints: &[GpuRigidFixedJoint],
        revolute_joints: &[GpuRigidRevoluteJoint],
    ) -> Result<(), GpuRigidSphereWorldError> {
        let prismatic_joints = self.prismatic_joints().to_vec();
        self.set_all_joints_with_prismatic(
            ball_joints,
            fixed_joints,
            revolute_joints,
            &prismatic_joints,
        )
    }

    /// Replace every GPU-resident joint type as one topology transaction.
    pub fn set_all_joints_with_prismatic(
        &mut self,
        ball_joints: &[GpuRigidBallJoint],
        fixed_joints: &[GpuRigidFixedJoint],
        revolute_joints: &[GpuRigidRevoluteJoint],
        prismatic_joints: &[GpuRigidPrismaticJoint],
    ) -> Result<(), GpuRigidSphereWorldError> {
        if self.faulted {
            return Err(GpuRigidSphereWorldError::Faulted);
        }
        let mut next = if ball_joints.is_empty()
            && fixed_joints.is_empty()
            && revolute_joints.is_empty()
            && prismatic_joints.is_empty()
        {
            None
        } else {
            Some(GpuRigidBallJointSolver::new_with_prismatic(
                &self.device,
                self.state.state_buffer(),
                self.len(),
                &self.environment_ids,
                ball_joints,
                fixed_joints,
                revolute_joints,
                prismatic_joints,
            )?)
        };
        if let (Some(previous), Some(next_solver)) = (&self.ball_joints, &mut next) {
            let mut angle_encoder =
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Tessera resident joint angle topology transfer"),
                    });
            let mut copied_angles = false;
            let mut used_revolute = vec![false; previous.revolute_joints().len()];
            for (index, joint) in revolute_joints.iter().enumerate() {
                if let Some(old_index) = previous
                    .revolute_joints()
                    .iter()
                    .enumerate()
                    .position(|(old_index, old)| !used_revolute[old_index] && old == joint)
                {
                    used_revolute[old_index] = true;
                    let motor = previous.revolute_motor(old_index).flatten();
                    let servo = previous.revolute_servo(old_index).flatten();
                    let limit = previous.revolute_limit(old_index).flatten();
                    next_solver.set_revolute_motor(&self.queue, index, motor)?;
                    next_solver.set_revolute_servo(&self.queue, index, servo)?;
                    next_solver.set_revolute_limit(&self.queue, index, limit)?;
                    next_solver.encode_copy_revolute_angle_from(
                        previous,
                        old_index,
                        index,
                        &mut angle_encoder,
                    );
                    copied_angles = true;
                }
            }
            if copied_angles {
                let _submission = self.queue.submit(Some(angle_encoder.finish()));
            }
            let mut used_prismatic = vec![false; previous.prismatic_joints().len()];
            for (index, joint) in prismatic_joints.iter().enumerate() {
                if let Some(old_index) = previous
                    .prismatic_joints()
                    .iter()
                    .enumerate()
                    .position(|(old_index, old)| !used_prismatic[old_index] && old == joint)
                {
                    used_prismatic[old_index] = true;
                    let motor = previous.prismatic_motor(old_index).flatten();
                    let servo = previous.prismatic_servo(old_index).flatten();
                    let limit = previous.prismatic_limit(old_index).flatten();
                    next_solver.set_prismatic_motor(&self.queue, index, motor)?;
                    next_solver.set_prismatic_servo(&self.queue, index, servo)?;
                    next_solver.set_prismatic_limit(&self.queue, index, limit)?;
                }
            }
        }
        if let Some(next_solver) = &next
            && !next_solver.revolute_joints().is_empty()
        {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Tessera resident joint angle initialization"),
                });
            next_solver.encode_sample_angles(&mut encoder);
            let _submission = self.queue.submit(Some(encoder.finish()));
        }
        self.ball_joints = next;
        self.impulse_cache.clear();
        self.resident_impulse_cache.clear();
        Ok(())
    }

    /// Contact material override of one sphere, if present.
    pub fn body_material_override(&self, index: usize) -> Option<Option<ColliderMaterial>> {
        self.body_materials.get(index).copied()
    }

    /// Contact material override of the finite ground plane, if present.
    pub fn ground_material_override(&self) -> Option<ColliderMaterial> {
        self.ground_material
    }

    /// Change gravity without replacing the resident body state.
    pub fn set_gravity(&mut self, gravity: [f32; 3]) -> Result<(), GpuRigidSphereWorldError> {
        let next = GpuRigidSphereWorldConfig {
            gravity,
            ..self.config
        };
        if !next.is_valid() {
            return Err(GpuRigidSphereWorldError::InvalidInput);
        }
        self.config = next;
        self.impulse_cache.clear();
        self.resident_impulse_cache.clear();
        if let Some(joints) = &mut self.ball_joints {
            joints.clear_impulse_cache(&self.queue);
        }
        Ok(())
    }

    /// Change default material and impulse solve parameters.
    pub fn set_solve_params(
        &mut self,
        solve: GpuRigidSphereSolveParams,
    ) -> Result<(), GpuRigidSphereWorldError> {
        let next = GpuRigidSphereWorldConfig {
            solve,
            ..self.config
        };
        if !next.is_valid() {
            return Err(GpuRigidSphereWorldError::InvalidInput);
        }
        self.config = next;
        self.impulse_cache.clear();
        self.resident_impulse_cache.clear();
        if let Some(joints) = &mut self.ball_joints {
            joints.clear_impulse_cache(&self.queue);
        }
        Ok(())
    }

    /// Change automatic sleep thresholds for later steps.
    pub fn set_sleep_settings(
        &mut self,
        sleep: SleepSettings,
    ) -> Result<(), GpuRigidSphereWorldError> {
        let next = GpuRigidSphereWorldConfig {
            sleep,
            ..self.config
        };
        if !next.is_valid() {
            return Err(GpuRigidSphereWorldError::InvalidInput);
        }
        if !sleep.enabled && !self.faulted {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Tessera resident sphere disable sleep"),
                });
            self.contacts
                .encode_sleep(&self.device, &mut encoder, 0.01, sleep)?;
            let _submission = self.queue.submit(Some(encoder.finish()));
        }
        self.config = next;
        self.impulse_cache.clear();
        self.resident_impulse_cache.clear();
        Ok(())
    }

    /// Replace one body and clear its idle history without reading state back.
    pub fn write_body(
        &mut self,
        index: usize,
        state: GpuRigidBodyState,
    ) -> Result<(), GpuRigidSphereWorldError> {
        if self.faulted {
            return Err(GpuRigidSphereWorldError::Faulted);
        }
        self.state.write_body(&self.queue, index, state)?;
        self.contacts.reset_sleep_timer(&self.queue, index)?;
        self.impulse_cache.invalidate_body(&self.queue, index);
        self.resident_impulse_cache.clear();
        if let Some(joints) = &mut self.ball_joints {
            joints.invalidate_body(&self.queue, index);
        }
        self.sample_joint_angles();
        Ok(())
    }

    /// Change prescribed motion without replacing the resident pose.
    ///
    /// `None` restores static behavior. Dynamic bodies ignore this command.
    /// Cached impulses involving the body are invalidated. Moving kinematic
    /// contacts wake sleeping dynamic partners when contact motion requires it.
    pub fn set_kinematic_motion(
        &mut self,
        index: usize,
        velocities: Option<([f32; 3], [f32; 3])>,
    ) -> Result<(), GpuRigidSphereWorldError> {
        if self.faulted {
            return Err(GpuRigidSphereWorldError::Faulted);
        }
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.state
            .encode_kinematic_motion(&self.device, &mut encoder, index, velocities)?;
        self.contacts.reset_sleep_timer(&self.queue, index)?;
        self.impulse_cache.invalidate_body(&self.queue, index);
        self.resident_impulse_cache.clear();
        if let Some(joints) = &mut self.ball_joints {
            joints.invalidate_body(&self.queue, index);
        }
        let _submission = self.queue.submit([encoder.finish()]);
        Ok(())
    }

    /// Overwrite one body's force and torque for the next step.
    pub fn write_forces(
        &self,
        index: usize,
        forces: GpuRigidBodyForces,
    ) -> Result<(), GpuRigidSphereWorldError> {
        if self.faulted {
            return Err(GpuRigidSphereWorldError::Faulted);
        }
        self.state.write_forces(&self.queue, index, forces)?;
        Ok(())
    }

    /// Override one body's contact material.
    pub fn set_body_material(
        &mut self,
        index: usize,
        material: ColliderMaterial,
    ) -> Result<(), GpuRigidSphereWorldError> {
        self.contacts
            .set_body_material(&self.queue, index, material)?;
        self.body_materials[index] = Some(material);
        self.impulse_cache.invalidate_body(&self.queue, index);
        self.resident_impulse_cache.clear();
        Ok(())
    }

    /// Restore the configured default material for one body.
    pub fn clear_body_material(&mut self, index: usize) -> Result<(), GpuRigidSphereWorldError> {
        self.contacts.clear_body_material(&self.queue, index)?;
        self.body_materials[index] = None;
        self.impulse_cache.invalidate_body(&self.queue, index);
        self.resident_impulse_cache.clear();
        Ok(())
    }

    /// Override the finite ground plane's contact material.
    pub fn set_ground_material(
        &mut self,
        material: ColliderMaterial,
    ) -> Result<(), GpuRigidSphereWorldError> {
        self.contacts.set_ground_material(&self.queue, material)?;
        self.ground_material = Some(material);
        self.impulse_cache.clear();
        self.resident_impulse_cache.clear();
        Ok(())
    }

    /// Restore the configured default material for the finite ground.
    pub fn clear_ground_material(&mut self) -> Result<(), GpuRigidSphereWorldError> {
        self.contacts.clear_ground_material(&self.queue)?;
        self.ground_material = None;
        self.impulse_cache.clear();
        self.resident_impulse_cache.clear();
        Ok(())
    }

    /// Collision masks for one resident body.
    pub fn body_collision_groups(&self, index: usize) -> Option<GpuRigidCollisionGroups> {
        self.contacts.collision_groups(index)
    }

    /// Change one body's reciprocal collision masks.
    pub fn set_body_collision_groups(
        &mut self,
        index: usize,
        groups: GpuRigidCollisionGroups,
    ) -> Result<(), GpuRigidSphereWorldError> {
        if self.faulted || index >= self.len() {
            return Err(GpuRigidSphereWorldError::InvalidInput);
        }
        if self.broad_phase.is_none() {
            let updated = (0..self.len())
                .map(|body| {
                    if body == index {
                        groups
                    } else {
                        self.contacts.collision_groups(body).unwrap_or_default()
                    }
                })
                .collect::<Vec<_>>();
            let mut pairs = Vec::new();
            let mut environments: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
            for (body, environment) in self.environment_ids.iter().copied().enumerate() {
                environments.entry(environment).or_default().push(body);
            }
            for bodies in environments.values() {
                for (offset, &a) in bodies.iter().enumerate() {
                    for &b in &bodies[(offset + 1)..] {
                        if !updated[a].allows(updated[b]) {
                            continue;
                        }
                        pairs.push(GpuPair {
                            a: a as u32,
                            b: b as u32,
                        });
                    }
                }
            }
            pairs.sort_unstable_by_key(|pair| (pair.a, pair.b));
            self.contacts.set_candidate_pairs(&self.device, &pairs)?;
        }
        self.contacts
            .set_collision_groups(&self.queue, index, groups)?;
        self.impulse_cache.clear();
        self.resident_impulse_cache.clear();
        Ok(())
    }

    /// Collision masks for the finite ground, when enabled.
    pub fn ground_collision_groups(&self) -> Option<GpuRigidCollisionGroups> {
        self.contacts.ground_collision_groups()
    }

    /// Change the finite ground's collision masks.
    pub fn set_ground_collision_groups(
        &mut self,
        groups: GpuRigidCollisionGroups,
    ) -> Result<(), GpuRigidSphereWorldError> {
        if self.faulted {
            return Err(GpuRigidSphereWorldError::Faulted);
        }
        self.contacts
            .set_ground_collision_groups(&self.queue, groups)?;
        self.impulse_cache.clear();
        self.resident_impulse_cache.clear();
        Ok(())
    }

    fn collision_groups_snapshot(&self) -> Vec<GpuRigidCollisionGroups> {
        (0..self.len())
            .map(|index| self.contacts.collision_groups(index).unwrap_or_default())
            .collect()
    }

    fn sample_joint_angles(&self) {
        if let Some(joints) = &self.ball_joints
            && !joints.revolute_joints().is_empty()
        {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Tessera resident joint angle state sample"),
                });
            joints.encode_sample_angles(&mut encoder);
            let _submission = self.queue.submit(Some(encoder.finish()));
        }
    }

    /// Append a sphere, returning its dense index.
    ///
    /// Topology edits copy surviving state on the GPU and rebuild contact buffers. Queued
    /// forces and warm-start impulses are discarded. Surviving joints retain
    /// their settings and angle history. Use the batch API for grouped worlds.
    pub fn append_body(
        &mut self,
        state: GpuRigidBodyState,
        radius: f32,
    ) -> Result<usize, GpuRigidSphereWorldError> {
        if self.faulted {
            return Err(GpuRigidSphereWorldError::Faulted);
        }
        if !self.topology_mutable
            || self.shapes.is_some()
            || !state.is_valid()
            || !radius.is_finite()
            || radius <= 0.0
        {
            return Err(GpuRigidSphereWorldError::InvalidInput);
        }
        let index = self.len();
        let mut radii = self.radii.clone();
        let mut materials = self.body_materials.clone();
        let mut collision_groups = self.collision_groups_snapshot();
        radii.push(radius);
        materials.push(None);
        collision_groups.push(GpuRigidCollisionGroups::default());
        self.rebuild_topology(
            &radii,
            None,
            &materials,
            &collision_groups,
            &vec![0; radii.len()],
            TopologyEdit::Insert(index, Box::new(state)),
        )?;
        Ok(index)
    }

    /// Remove a sphere and return its last GPU state.
    ///
    /// All later dense indices shift down by one. Only the removed body is read
    /// back; surviving state is copied on the GPU. Queued forces, warm-start
    /// impulses, and sleep timers are discarded. Attached joints are removed;
    /// surviving joints retain their settings and angle history.
    pub fn remove_body(
        &mut self,
        index: usize,
    ) -> Result<GpuRigidBodyState, GpuRigidSphereWorldError> {
        self.remove_body_inner(index, true)?
            .ok_or(GpuRigidSphereWorldError::InvalidInput)
    }

    /// Remove a sphere without reading its state back from the GPU.
    pub fn discard_body(&mut self, index: usize) -> Result<(), GpuRigidSphereWorldError> {
        let _removed = self.remove_body_inner(index, false)?;
        Ok(())
    }

    fn remove_body_inner(
        &mut self,
        index: usize,
        readback: bool,
    ) -> Result<Option<GpuRigidBodyState>, GpuRigidSphereWorldError> {
        if self.faulted {
            return Err(GpuRigidSphereWorldError::Faulted);
        }
        if !self.topology_mutable || self.shapes.is_some() || index >= self.len() {
            return Err(GpuRigidSphereWorldError::InvalidInput);
        }
        let removed = if readback {
            Some(self.readback_range(index..index + 1)?[0])
        } else {
            None
        };
        let mut radii = self.radii.clone();
        let _removed_radius = radii.remove(index);
        let mut materials = self.body_materials.clone();
        let _removed_material = materials.remove(index);
        let mut collision_groups = self.collision_groups_snapshot();
        let _removed_groups = collision_groups.remove(index);
        self.rebuild_topology(
            &radii,
            None,
            &materials,
            &collision_groups,
            &vec![0; radii.len()],
            TopologyEdit::Remove(index),
        )?;
        Ok(removed)
    }

    /// Append a primitive to a single mixed-shape world.
    ///
    /// This copies surviving state on the GPU and rebuilds contact buffers. Queued forces,
    /// warm-start impulses, and sleep timers are discarded. Surviving joints
    /// retain their settings and angle history.
    pub fn append_primitive(
        &mut self,
        state: GpuRigidBodyState,
        shape: GpuRigidShape,
    ) -> Result<usize, GpuRigidSphereWorldError> {
        if self.faulted {
            return Err(GpuRigidSphereWorldError::Faulted);
        }
        let Some(radius) = shape.bounding_radius() else {
            return Err(GpuRigidSphereWorldError::InvalidInput);
        };
        if !self.topology_mutable || self.shapes.is_none() || !state.is_valid() {
            return Err(GpuRigidSphereWorldError::InvalidInput);
        }
        let index = self.len();
        let mut radii = self.radii.clone();
        let mut shapes = self
            .shapes
            .clone()
            .ok_or(GpuRigidSphereWorldError::InvalidInput)?;
        let mut materials = self.body_materials.clone();
        let mut collision_groups = self.collision_groups_snapshot();
        radii.push(radius);
        shapes.push(shape);
        materials.push(None);
        collision_groups.push(GpuRigidCollisionGroups::default());
        self.rebuild_topology(
            &radii,
            Some(&shapes),
            &materials,
            &collision_groups,
            &vec![0; radii.len()],
            TopologyEdit::Insert(index, Box::new(state)),
        )?;
        Ok(index)
    }

    /// Remove a primitive and return its last GPU state and collision shape.
    ///
    /// Later dense indices shift down by one. Only the removed body is read back;
    /// surviving state is copied on the GPU. Queued forces, warm-start impulses,
    /// and sleep timers are discarded. Attached joints are removed; surviving
    /// joints retain their settings and angle history.
    pub fn remove_primitive(
        &mut self,
        index: usize,
    ) -> Result<(GpuRigidBodyState, GpuRigidShape), GpuRigidSphereWorldError> {
        let (state, shape) = self.remove_primitive_inner(index, true)?;
        Ok((state.ok_or(GpuRigidSphereWorldError::InvalidInput)?, shape))
    }

    /// Remove a primitive without reading its state back from the GPU.
    pub fn discard_primitive(&mut self, index: usize) -> Result<(), GpuRigidSphereWorldError> {
        let _removed = self.remove_primitive_inner(index, false)?;
        Ok(())
    }

    fn remove_primitive_inner(
        &mut self,
        index: usize,
        readback: bool,
    ) -> Result<(Option<GpuRigidBodyState>, GpuRigidShape), GpuRigidSphereWorldError> {
        if self.faulted {
            return Err(GpuRigidSphereWorldError::Faulted);
        }
        if !self.topology_mutable || self.shapes.is_none() || index >= self.len() {
            return Err(GpuRigidSphereWorldError::InvalidInput);
        }
        let removed_state = if readback {
            Some(self.readback_range(index..index + 1)?[0])
        } else {
            None
        };
        let mut radii = self.radii.clone();
        let _removed_radius = radii.remove(index);
        let mut shapes = self
            .shapes
            .clone()
            .ok_or(GpuRigidSphereWorldError::InvalidInput)?;
        let removed_shape = shapes.remove(index);
        let mut materials = self.body_materials.clone();
        let _removed_material = materials.remove(index);
        let mut collision_groups = self.collision_groups_snapshot();
        let _removed_groups = collision_groups.remove(index);
        self.rebuild_topology(
            &radii,
            Some(&shapes),
            &materials,
            &collision_groups,
            &vec![0; radii.len()],
            TopologyEdit::Remove(index),
        )?;
        Ok((removed_state, removed_shape))
    }

    fn rebuild_topology(
        &mut self,
        radii: &[f32],
        shapes: Option<&[GpuRigidShape]>,
        materials: &[Option<ColliderMaterial>],
        collision_groups: &[GpuRigidCollisionGroups],
        environment_ids: &[u32],
        edit: TopologyEdit,
    ) -> Result<(), GpuRigidSphereWorldError> {
        let expected_count = match &edit {
            TopologyEdit::Insert(index, state) if *index <= self.len() && state.is_valid() => {
                self.len().checked_add(1)
            }
            TopologyEdit::Remove(index) if *index < self.len() => Some(self.len() - 1),
            TopologyEdit::InsertRange(index, states)
                if *index <= self.len() && states.iter().all(GpuRigidBodyState::is_valid) =>
            {
                self.len().checked_add(states.len())
            }
            TopologyEdit::RemoveRange(range)
                if range.start <= range.end && range.end <= self.len() =>
            {
                Some(self.len() - range.len())
            }
            _ => None,
        };
        if expected_count != Some(radii.len())
            || materials.len() != radii.len()
            || collision_groups.len() != radii.len()
            || environment_ids.len() != radii.len()
            || shapes.is_some_and(|shapes| shapes.len() != radii.len())
        {
            return Err(GpuRigidSphereWorldError::InvalidInput);
        }
        let state = GpuRigidStateSession::new_uninitialized(&self.device, radii.len())?;
        let mut next = Self::new_grouped_with_state(
            &self.device,
            &self.queue,
            state,
            radii,
            shapes,
            environment_ids,
            NewWorldOptions {
                config: self.config,
                previous_contacts: Some(&self.contacts),
            },
        )?;
        next.topology_mutable = self.topology_mutable;
        next.max_linear_speed = self.max_linear_speed;
        for (index, material) in materials.iter().copied().enumerate() {
            if let Some(material) = material {
                next.set_body_material(index, material)?;
            }
        }
        if let Some(material) = self.ground_material {
            next.set_ground_material(material)?;
        }
        if let Some(groups) = self.ground_collision_groups() {
            next.set_ground_collision_groups(groups)?;
        }
        for (index, groups) in collision_groups.iter().copied().enumerate() {
            if groups != GpuRigidCollisionGroups::default() {
                next.set_body_collision_groups(index, groups)?;
            }
        }
        let bytes_per_body = size_of::<GpuRigidBodyState>() as u64;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Tessera resident rigid topology state transfer"),
            });
        let mut copied = false;
        match &edit {
            TopologyEdit::Insert(index, state) => {
                let index = *index;
                if index > 0 {
                    encoder.copy_buffer_to_buffer(
                        self.state.state_buffer(),
                        0,
                        next.state.state_buffer(),
                        0,
                        index as u64 * bytes_per_body,
                    );
                    copied = true;
                }
                let suffix = self.len() - index;
                if suffix > 0 {
                    encoder.copy_buffer_to_buffer(
                        self.state.state_buffer(),
                        index as u64 * bytes_per_body,
                        next.state.state_buffer(),
                        (index + 1) as u64 * bytes_per_body,
                        suffix as u64 * bytes_per_body,
                    );
                    copied = true;
                }
                next.state.write_body(&self.queue, index, **state)?;
            }
            TopologyEdit::Remove(index) => {
                let index = *index;
                if index > 0 {
                    encoder.copy_buffer_to_buffer(
                        self.state.state_buffer(),
                        0,
                        next.state.state_buffer(),
                        0,
                        index as u64 * bytes_per_body,
                    );
                    copied = true;
                }
                let suffix = self.len() - index - 1;
                if suffix > 0 {
                    encoder.copy_buffer_to_buffer(
                        self.state.state_buffer(),
                        (index + 1) as u64 * bytes_per_body,
                        next.state.state_buffer(),
                        index as u64 * bytes_per_body,
                        suffix as u64 * bytes_per_body,
                    );
                    copied = true;
                }
            }
            TopologyEdit::InsertRange(index, states) => {
                if *index > 0 {
                    encoder.copy_buffer_to_buffer(
                        self.state.state_buffer(),
                        0,
                        next.state.state_buffer(),
                        0,
                        *index as u64 * bytes_per_body,
                    );
                    copied = true;
                }
                let suffix = self.len() - index;
                if suffix > 0 {
                    encoder.copy_buffer_to_buffer(
                        self.state.state_buffer(),
                        *index as u64 * bytes_per_body,
                        next.state.state_buffer(),
                        (index + states.len()) as u64 * bytes_per_body,
                        suffix as u64 * bytes_per_body,
                    );
                    copied = true;
                }
                for (offset, state) in states.iter().enumerate() {
                    next.state.write_body(&self.queue, index + offset, *state)?;
                }
            }
            TopologyEdit::RemoveRange(range) => {
                if range.start > 0 {
                    encoder.copy_buffer_to_buffer(
                        self.state.state_buffer(),
                        0,
                        next.state.state_buffer(),
                        0,
                        range.start as u64 * bytes_per_body,
                    );
                    copied = true;
                }
                let suffix = self.len() - range.end;
                if suffix > 0 {
                    encoder.copy_buffer_to_buffer(
                        self.state.state_buffer(),
                        range.end as u64 * bytes_per_body,
                        next.state.state_buffer(),
                        range.start as u64 * bytes_per_body,
                        suffix as u64 * bytes_per_body,
                    );
                    copied = true;
                }
            }
        }
        if copied {
            let _submission = self.queue.submit(Some(encoder.finish()));
        }
        self.transfer_remapped_joints(&mut next, &edit)?;
        *self = next;
        Ok(())
    }

    fn transfer_remapped_joints(
        &self,
        next: &mut Self,
        edit: &TopologyEdit,
    ) -> Result<(), GpuRigidSphereWorldError> {
        let Some(previous) = &self.ball_joints else {
            return Ok(());
        };
        let ball_joints = previous
            .joints()
            .iter()
            .filter_map(|joint| {
                let (body_a, body_b) = remap_joint_bodies(edit, joint.body_a, joint.body_b)?;
                Some(GpuRigidBallJoint {
                    body_a,
                    body_b,
                    ..*joint
                })
            })
            .collect::<Vec<_>>();
        let fixed_joints = previous
            .fixed_joints()
            .iter()
            .filter_map(|joint| {
                let (body_a, body_b) = remap_joint_bodies(edit, joint.body_a, joint.body_b)?;
                Some(GpuRigidFixedJoint {
                    body_a,
                    body_b,
                    ..*joint
                })
            })
            .collect::<Vec<_>>();
        let revolute = previous
            .revolute_joints()
            .iter()
            .enumerate()
            .filter_map(|(old_index, joint)| {
                let (body_a, body_b) = remap_joint_bodies(edit, joint.body_a, joint.body_b)?;
                Some((
                    old_index,
                    GpuRigidRevoluteJoint {
                        body_a,
                        body_b,
                        ..*joint
                    },
                ))
            })
            .collect::<Vec<_>>();
        let prismatic = previous
            .prismatic_joints()
            .iter()
            .enumerate()
            .filter_map(|(old_index, joint)| {
                let (body_a, body_b) = remap_joint_bodies(edit, joint.body_a, joint.body_b)?;
                Some((
                    old_index,
                    GpuRigidPrismaticJoint {
                        body_a,
                        body_b,
                        ..*joint
                    },
                ))
            })
            .collect::<Vec<_>>();
        next.set_all_joints_with_prismatic(
            &ball_joints,
            &fixed_joints,
            &revolute.iter().map(|(_, joint)| *joint).collect::<Vec<_>>(),
            &prismatic
                .iter()
                .map(|(_, joint)| *joint)
                .collect::<Vec<_>>(),
        )?;
        let Some(next_solver) = next.ball_joints.as_mut() else {
            return Ok(());
        };
        let mut angle_encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Tessera rigid topology joint angle transfer"),
                });
        for (index, (old_index, _)) in revolute.iter().enumerate() {
            next_solver.set_revolute_motor(
                &self.queue,
                index,
                previous.revolute_motor(*old_index).flatten(),
            )?;
            next_solver.set_revolute_servo(
                &self.queue,
                index,
                previous.revolute_servo(*old_index).flatten(),
            )?;
            next_solver.set_revolute_limit(
                &self.queue,
                index,
                previous.revolute_limit(*old_index).flatten(),
            )?;
            next_solver.encode_copy_revolute_angle_from(
                previous,
                *old_index,
                index,
                &mut angle_encoder,
            );
        }
        if !revolute.is_empty() {
            let _submission = self.queue.submit(Some(angle_encoder.finish()));
        }
        for (index, (old_index, _)) in prismatic.iter().enumerate() {
            next_solver.set_prismatic_motor(
                &self.queue,
                index,
                previous.prismatic_motor(*old_index).flatten(),
            )?;
            next_solver.set_prismatic_servo(
                &self.queue,
                index,
                previous.prismatic_servo(*old_index).flatten(),
            )?;
            next_solver.set_prismatic_limit(
                &self.queue,
                index,
                previous.prismatic_limit(*old_index).flatten(),
            )?;
        }
        Ok(())
    }

    /// Advance one substep and return the current candidate count.
    ///
    /// If a GPU capacity or readback error interrupts a step, the world becomes
    /// faulted. `reset` restores a known state before another step.
    pub fn step(&mut self, dt: f32) -> Result<usize, GpuRigidSphereWorldError> {
        if self.faulted {
            return Err(GpuRigidSphereWorldError::Faulted);
        }
        if !dt.is_finite() || dt <= 0.0 {
            return Err(GpuRigidSphereWorldError::InvalidInput);
        }
        self.select_temporal_mode(false);
        if self.can_use_resident_step() {
            let _buffers = self.step_gpu_resident(dt)?;
            return Ok(self.candidate_pair_count());
        }
        self.resident_impulse_cache.clear();
        if self.broad_phase.is_none() {
            return self.step_exhaustive_substeps(dt, 1);
        }
        if let Some(broad_phase) = &self.broad_phase {
            self.state.step_with_speed_limit(
                &self.device,
                &self.queue,
                dt,
                self.config.gravity,
                self.max_linear_speed,
            )?;
            if let Err(error) = self.contacts.refresh_candidate_pairs_from_state_filtered(
                &self.device,
                &self.queue,
                broad_phase,
                &self.environment_ids,
            ) {
                self.faulted = true;
                return Err(error.into());
            }
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Tessera resident sphere world step"),
            });
        if self.broad_phase.is_some()
            && let Err(error) = self.impulse_cache.remap_candidate_pairs(
                &self.device,
                &mut encoder,
                &self.contacts,
                dt,
            )
        {
            self.faulted = true;
            return Err(error.into());
        }
        if let Some(joints) = &self.ball_joints {
            joints.encode_capture_velocity(&mut encoder);
        }
        if let Err(error) = self.encode_constraints(&mut encoder, dt) {
            self.faulted = true;
            return Err(error);
        }
        self.encode_linear_speed_limit(&mut encoder)?;
        if let Err(error) = self.contacts.encode_sleep_with_joints(
            &self.device,
            &mut encoder,
            dt,
            self.config.sleep,
            self.ball_joints.as_ref(),
        ) {
            self.faulted = true;
            return Err(error.into());
        }
        let _submission = self.queue.submit(Some(encoder.finish()));
        self.last_resident_contacts = None;
        Ok(self.candidate_pair_count())
    }

    fn can_use_resident_step(&self) -> bool {
        if self.broad_phase.is_none() || self.ball_joints.is_some() {
            return false;
        }
        let count = self.len() as u64;
        let Some(pair_capacity) = count.checked_mul(count.saturating_sub(1)).map(|n| n / 2) else {
            return false;
        };
        if pair_capacity > u32::MAX as u64 {
            return false;
        }
        let pair_slots = pair_capacity * u64::from(self.contacts.pair_contact_stride());
        let ground_slots = if self.config.ground_half_extent.is_some() {
            count * u64::from(self.contacts.ground_contact_stride())
        } else {
            0
        };
        let Some(slots) = pair_slots.checked_add(ground_slots) else {
            return false;
        };
        let limits = self.device.limits();
        let maximum_storage = u64::from(limits.max_storage_buffer_binding_size);
        self.config.solve.iterations <= 20_000
            && count <= u64::from(limits.max_compute_workgroups_per_dimension)
            && (1 + 8 * count + 2 * pair_capacity).saturating_mul(4) <= maximum_storage
            && slots.saturating_mul(64) <= maximum_storage
            && pair_slots.saturating_mul(48) <= maximum_storage
            && pair_capacity.saturating_mul(8) <= maximum_storage
            && slots.saturating_mul(64) <= limits.max_buffer_size
    }

    /// Advance a large joint-free world without reading LBVH pairs to the CPU.
    ///
    /// This path reuses GPU impulse history by stable body pair and partitions
    /// larger candidate workloads into independent GPU contact islands. The
    /// returned buffers can be read explicitly for contact diagnostics before
    /// the next step; the ground buffer aliases this world's contact storage.
    /// `readback_contacts` returns these candidates and the current ground slots.
    /// The GPU solve status is checked before the method returns, so a failed step
    /// faults the world and requires `reset` before reuse.
    pub fn step_gpu_resident(
        &mut self,
        dt: f32,
    ) -> Result<(GpuLbvhResidentPairs, GpuRigidCandidateContacts), GpuRigidSphereWorldError> {
        if self.faulted {
            return Err(GpuRigidSphereWorldError::Faulted);
        }
        if !dt.is_finite() || dt <= 0.0 || self.ball_joints.is_some() {
            return Err(GpuRigidSphereWorldError::InvalidInput);
        }
        if self.broad_phase.is_none() {
            return Err(GpuRigidSphereWorldError::InvalidInput);
        }
        self.select_temporal_mode(false);
        let Some(broad_phase) = &self.broad_phase else {
            return Err(GpuRigidSphereWorldError::InvalidInput);
        };
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Tessera resident LBVH world step"),
            });
        self.state.encode_step_with_speed_limit(
            &self.device,
            &mut encoder,
            dt,
            self.config.gravity,
            self.max_linear_speed,
        )?;
        let result = (|| {
            let (candidates, output) = self.contacts.encode_state_lbvh_contacts(
                &self.device,
                &mut encoder,
                broad_phase,
                Some(&self.environment_ids),
            )?;
            let solve = self.solver.encode_resident_candidates_cached(
                &self.device,
                &mut encoder,
                &self.contacts,
                &candidates,
                &output,
                dt,
                self.config.solve,
                &mut self.resident_impulse_cache,
            )?;
            if let Some(max_speed) = self.max_linear_speed {
                self.state
                    .encode_clamp_linear_speed(&self.device, &mut encoder, max_speed)?;
            }
            self.contacts.encode_candidate_sleep(
                &self.device,
                &mut encoder,
                &candidates,
                &output,
                &solve.status,
                dt,
                self.config.sleep,
            )?;
            Ok::<_, GpuRigidSphereWorldError>((candidates, output, solve))
        })();
        let (candidates, output, solve) = match result {
            Ok(value) => value,
            Err(error) => {
                self.faulted = true;
                return Err(error);
            }
        };
        let _submission = self.queue.submit(Some(encoder.finish()));
        let count = match solve.readback_status_and_count(&self.device, &self.queue, &candidates) {
            Ok(count) => count as usize,
            Err(error) => {
                self.faulted = true;
                return Err(error.into());
            }
        };
        self.impulse_cache.clear();
        self.last_resident_contacts = Some((count, candidates.clone(), output.clone()));
        Ok((candidates, output))
    }

    /// Advance several substeps without intermediate CPU observation.
    ///
    /// For at most 16 bodies, all substeps share one GPU submission. Larger
    /// scenes use the same bounded resident or candidate-readback path as `step`
    /// for each substep. The maximum
    /// batch length bounds command-buffer and temporary-buffer allocation.
    pub fn step_substeps(
        &mut self,
        dt: f32,
        count: u32,
    ) -> Result<usize, GpuRigidSphereWorldError> {
        if self.faulted {
            return Err(GpuRigidSphereWorldError::Faulted);
        }
        if !dt.is_finite() || dt <= 0.0 || count == 0 || count > MAX_ENCODED_SUBSTEPS {
            return Err(GpuRigidSphereWorldError::InvalidInput);
        }
        self.select_temporal_mode(false);
        if self.broad_phase.is_none() {
            return self.step_exhaustive_substeps(dt, count);
        }
        let mut candidates = 0;
        for _ in 0..count {
            candidates = self.step(dt)?;
        }
        Ok(candidates)
    }

    fn select_temporal_mode(&mut self, temporal: bool) {
        if self.temporal_active != temporal {
            self.impulse_cache.clear();
            self.resident_impulse_cache.clear();
            self.temporal_active = temporal;
        }
    }

    /// Advance one temporal frame with fixed contact topology and GPU pose refresh.
    ///
    /// `frame_dt` is the whole frame duration. Forces are captured once and reused
    /// across `substeps`. Narrow phase runs at the initial poses; each substep
    /// integrates velocity, solves bias, integrates pose, refreshes anchors, then
    /// solves relaxation. Large worlds currently read LBVH pairs once per frame.
    /// Use the speculative variants to include separated but approaching pairs.
    /// Joints use contact frequency,
    /// damping and sweep count, with a default angular correction speed of 3 rad/s.
    /// Servo spring targets are captured once per substep. Joint limits are soft.
    pub fn step_temporal(
        &mut self,
        frame_dt: f32,
        substeps: u32,
        settings: GpuRigidTemporalSolveParams,
    ) -> Result<usize, GpuRigidSphereWorldError> {
        self.step_temporal_speculative(frame_dt, substeps, settings, 0.0)
    }

    /// Advance a temporal frame with initial sphere contacts inside `margin`.
    /// Positive pair margins use convex distance witnesses or surface BVH queries.
    /// This is speculative contact, not swept CCD.
    /// A four-byte GPU query status is checked after submission. Unconverged
    /// queries fault the world; reset is required before another step.
    pub fn step_temporal_speculative(
        &mut self,
        frame_dt: f32,
        substeps: u32,
        settings: GpuRigidTemporalSolveParams,
        margin: f32,
    ) -> Result<usize, GpuRigidSphereWorldError> {
        self.step_temporal_impl(frame_dt, substeps, settings, margin, false, None)
    }

    /// A conservative initial sphere-pair margin from the configured speed cap.
    ///
    /// The margin covers the maximum relative translation over one frame, with
    /// a small f32 guard. It does not account for curved trajectories or prove
    /// swept collision detection. Mixed primitive worlds must supply a margin
    /// that also covers rotational motion instead.
    pub fn speed_bounded_speculative_margin(
        &self,
        frame_dt: f32,
    ) -> Result<f32, GpuRigidSphereWorldError> {
        let Some(max_speed) = self.max_linear_speed else {
            return Err(GpuRigidSphereWorldError::InvalidInput);
        };
        if !frame_dt.is_finite()
            || frame_dt <= 0.0
            || self.shapes.as_ref().is_some_and(|shapes| {
                shapes
                    .iter()
                    .any(|shape| !matches!(shape, GpuRigidShape::Sphere { .. }))
            })
        {
            return Err(GpuRigidSphereWorldError::InvalidInput);
        }
        let margin = (2.0 * max_speed * frame_dt).mul_add(1.0001, 1.0e-4);
        if !margin.is_finite() {
            return Err(GpuRigidSphereWorldError::InvalidInput);
        }
        Ok(margin)
    }

    /// Advance spheres with a speculative margin derived from the speed cap.
    pub fn step_temporal_speed_bounded(
        &mut self,
        frame_dt: f32,
        substeps: u32,
        settings: GpuRigidTemporalSolveParams,
    ) -> Result<usize, GpuRigidSphereWorldError> {
        let margin = self.speed_bounded_speculative_margin(frame_dt)?;
        self.step_temporal_speculative(frame_dt, substeps, settings, margin)
    }

    /// Advance a temporal frame with speculative ground rows for every shape.
    /// Pair contacts keep ordinary overlap conditions; joint worlds remain unsupported.
    pub fn step_temporal_speculative_ground(
        &mut self,
        frame_dt: f32,
        substeps: u32,
        settings: GpuRigidTemporalSolveParams,
        margin: f32,
    ) -> Result<usize, GpuRigidSphereWorldError> {
        self.step_temporal_impl(frame_dt, substeps, settings, margin, true, None)
    }

    /// Advance a temporal frame with separately configured joint softness.
    ///
    /// A positive margin enables speculative pairs and ground contacts. Each
    /// substep alternates contact and joint sweeps before and after pose integration.
    pub fn step_temporal_with_joints(
        &mut self,
        frame_dt: f32,
        substeps: u32,
        settings: GpuRigidTemporalSolveParams,
        joint_settings: GpuRigidTemporalJointParams,
        margin: f32,
    ) -> Result<usize, GpuRigidSphereWorldError> {
        self.step_temporal_impl(
            frame_dt,
            substeps,
            settings,
            margin,
            false,
            Some(joint_settings),
        )
    }

    /// Advance a temporal frame with independent joint settings and speculative ground only.
    pub fn step_temporal_with_joints_ground(
        &mut self,
        frame_dt: f32,
        substeps: u32,
        settings: GpuRigidTemporalSolveParams,
        joint_settings: GpuRigidTemporalJointParams,
        margin: f32,
    ) -> Result<usize, GpuRigidSphereWorldError> {
        self.step_temporal_impl(
            frame_dt,
            substeps,
            settings,
            margin,
            true,
            Some(joint_settings),
        )
    }

    fn step_temporal_impl(
        &mut self,
        frame_dt: f32,
        substeps: u32,
        settings: GpuRigidTemporalSolveParams,
        margin: f32,
        ground_only: bool,
        joint_settings: Option<GpuRigidTemporalJointParams>,
    ) -> Result<usize, GpuRigidSphereWorldError> {
        if self.faulted {
            return Err(GpuRigidSphereWorldError::Faulted);
        }
        let dt = frame_dt / substeps as f32;
        if !frame_dt.is_finite()
            || frame_dt <= 0.0
            || !dt.is_finite()
            || dt <= 0.0
            || substeps == 0
            || substeps > MAX_ENCODED_SUBSTEPS
            || !margin.is_finite()
            || margin < 0.0
        {
            return Err(GpuRigidSphereWorldError::InvalidInput);
        }
        // Validate all coefficients before any command is submitted or history changed.
        settings.validate(dt)?;
        let joint_settings = joint_settings.unwrap_or(GpuRigidTemporalJointParams {
            frequency: settings.normal_frequency,
            damping_ratio: settings.damping_ratio,
            max_linear_correction_speed: settings.max_corrective_velocity,
            iterations: settings.iterations,
            ..GpuRigidTemporalJointParams::default()
        });
        if let Some(joints) = &self.ball_joints {
            let _coefficients = joints.validate_temporal(dt, joint_settings)?;
        }
        self.select_temporal_mode(true);
        let result = (|| {
            if let Some(broad_phase) = &self.broad_phase {
                self.contacts
                    .refresh_speculative_pairs_from_state_filtered(
                        &self.device,
                        &self.queue,
                        broad_phase,
                        &self.environment_ids,
                        if ground_only { 0.0 } else { margin },
                    )?;
            }
            let transport = GpuRigidContactTransport::new(&self.device, &self.contacts)?;
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Tessera temporal rigid frame"),
                });
            if self.broad_phase.is_some() {
                self.impulse_cache.remap_candidate_pairs(
                    &self.device,
                    &mut encoder,
                    &self.contacts,
                    dt,
                )?;
            }
            let joint_solve = self
                .ball_joints
                .as_mut()
                .map(|joints| {
                    joints.prepare_temporal(&self.device, &mut encoder, dt, joint_settings)
                })
                .transpose()?;
            if let Some(joints) = &joint_solve {
                joints.encode_capture_angles(&mut encoder);
            }
            self.state.encode_capture_forces(&self.device, &mut encoder);
            let status = if margin > 0.0 && ground_only {
                self.contacts
                    .encode_speculative_ground(&self.device, &mut encoder, margin)?;
                None
            } else if margin > 0.0 {
                Some(self.contacts.encode_speculative_checked(
                    &self.device,
                    &mut encoder,
                    margin,
                )?)
            } else {
                self.contacts.encode(&mut encoder);
                None
            };
            transport.encode_capture(&mut encoder);
            let capture = self.solver.prepare_temporal_cached(
                &self.device,
                &self.contacts,
                dt,
                settings,
                &mut self.impulse_cache,
            )?;
            if let Some(capture) = &capture {
                capture.encode_capture_anchors(&mut encoder);
            }
            for _ in 0..substeps {
                self.state.encode_velocity_step_with_speed_limit(
                    &self.device,
                    &mut encoder,
                    dt,
                    self.config.gravity,
                    self.max_linear_speed,
                )?;
                if let Some(joints) = &joint_solve {
                    joints.encode_capture_drives(&mut encoder);
                }
                transport.encode_refresh(&mut encoder);
                let solve = self.solver.prepare_temporal_cached(
                    &self.device,
                    &self.contacts,
                    dt,
                    settings,
                    &mut self.impulse_cache,
                )?;
                if let Some(joints) = &joint_solve {
                    if let Some(solve) = &solve {
                        solve.encode_warm(&mut encoder);
                    }
                    joints.encode_warm(&mut encoder);
                    for iteration in 0..settings.iterations.max(joint_settings.iterations) {
                        if iteration < settings.iterations
                            && let Some(solve) = &solve
                        {
                            solve.encode_bias_iteration(&mut encoder);
                        }
                        if iteration < joint_settings.iterations {
                            joints.encode_bias_iteration(&mut encoder);
                        }
                    }
                } else if let Some(solve) = &solve {
                    solve.encode_bias(&mut encoder);
                }
                self.encode_linear_speed_limit(&mut encoder)?;
                self.state
                    .encode_position_step(&self.device, &mut encoder, dt)?;
                if let Some(joints) = &joint_solve {
                    joints.encode_integrated_angles(&mut encoder);
                }
                transport.encode_refresh(&mut encoder);
                if let Some(joints) = &joint_solve {
                    for iteration in 0..settings.iterations.max(joint_settings.iterations) {
                        if iteration < settings.iterations
                            && let Some(solve) = &solve
                        {
                            solve.encode_relax_iteration(&mut encoder);
                        }
                        if iteration < joint_settings.iterations {
                            joints.encode_relax_iteration(&mut encoder);
                        }
                    }
                } else if let Some(solve) = &solve {
                    solve.encode_relax(&mut encoder);
                }
                self.encode_linear_speed_limit(&mut encoder)?;
                self.contacts.encode_sleep_with_joints(
                    &self.device,
                    &mut encoder,
                    dt,
                    self.config.sleep,
                    self.ball_joints.as_ref(),
                )?;
            }
            let _submission = self.queue.submit(Some(encoder.finish()));
            if let Some(status) = status {
                status.readback(&self.device, &self.queue)?;
            }
            Ok::<_, GpuRigidSphereWorldError>(())
        })();
        if let Err(error) = result {
            self.faulted = true;
            return Err(error);
        }
        self.last_resident_contacts = None;
        Ok(self.candidate_pair_count())
    }

    fn step_exhaustive_substeps(
        &mut self,
        dt: f32,
        count: u32,
    ) -> Result<usize, GpuRigidSphereWorldError> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Tessera resident sphere batched substeps"),
            });
        for _ in 0..count {
            if let Err(error) = self.encode_exhaustive_step(&mut encoder, dt) {
                self.faulted = true;
                return Err(error);
            }
        }
        let _submission = self.queue.submit(Some(encoder.finish()));
        self.last_resident_contacts = None;
        Ok(self.candidate_pair_count())
    }

    fn encode_exhaustive_step(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        dt: f32,
    ) -> Result<(), GpuRigidSphereWorldError> {
        self.state.encode_step_with_speed_limit(
            &self.device,
            encoder,
            dt,
            self.config.gravity,
            self.max_linear_speed,
        )?;
        if let Some(joints) = &self.ball_joints {
            joints.encode_capture_velocity(encoder);
        }
        self.encode_constraints(encoder, dt)?;
        self.encode_linear_speed_limit(encoder)?;
        self.contacts.encode_sleep_with_joints(
            &self.device,
            encoder,
            dt,
            self.config.sleep,
            self.ball_joints.as_ref(),
        )?;
        Ok(())
    }

    fn encode_constraints(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        dt: f32,
    ) -> Result<(), GpuRigidSphereWorldError> {
        self.contacts.encode(encoder);
        if let Some(joints) = &mut self.ball_joints {
            let contacts = self.solver.prepare_coupled_cached(
                &self.device,
                &self.contacts,
                dt,
                self.config.solve,
                &mut self.impulse_cache,
            )?;
            if let Some(contacts) = &contacts {
                contacts.encode_warm(encoder);
            }
            joints.encode_coupled_warm(
                &self.queue,
                encoder,
                dt,
                self.config.solve.iterations,
                self.config.solve.bias_factor,
            )?;
            for iteration in 0..self.config.solve.iterations {
                if let Some(contacts) = &contacts {
                    contacts.encode_iteration(encoder, iteration == 0);
                }
                joints.encode_coupled_iteration(encoder);
            }
            joints.encode_pose_correction(encoder);
            joints.encode_sample_angles(encoder);
        } else {
            self.solver.encode_cached(
                &self.device,
                encoder,
                &self.contacts,
                dt,
                self.config.solve,
                &mut self.impulse_cache,
            )?;
        }
        Ok(())
    }

    fn encode_linear_speed_limit(
        &self,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(), GpuRigidSphereWorldError> {
        if let Some(max_speed) = self.max_linear_speed {
            self.state
                .encode_clamp_linear_speed(&self.device, encoder, max_speed)?;
        }
        Ok(())
    }

    /// Replace every body after a failed step or to restart a scene.
    ///
    /// Geometry, materials, and configuration are retained. The next step
    /// rebuilds candidates from the replaced positions.
    pub fn reset(&mut self, states: &[GpuRigidBodyState]) -> Result<(), GpuRigidSphereWorldError> {
        if states.len() != self.len() || states.iter().any(|state| !state.is_valid()) {
            return Err(GpuRigidSphereWorldError::InvalidInput);
        }
        for (index, state) in states.iter().copied().enumerate() {
            self.state.write_body(&self.queue, index, state)?;
            self.contacts.reset_sleep_timer(&self.queue, index)?;
        }
        self.impulse_cache.clear();
        self.resident_impulse_cache.clear();
        if let Some(joints) = &mut self.ball_joints {
            joints.clear_cache(&self.queue);
        }
        self.sample_joint_angles();
        self.faulted = false;
        Ok(())
    }

    /// Transfer all body states to the CPU for observation or persistence.
    pub fn readback(&self) -> Result<Vec<GpuRigidBodyState>, GpuRigidSphereWorldError> {
        Ok(self.state.readback(&self.device, &self.queue)?)
    }

    /// Transfer only a contiguous body range to the CPU.
    pub fn readback_range(
        &self,
        range: Range<usize>,
    ) -> Result<Vec<GpuRigidBodyState>, GpuRigidSphereWorldError> {
        Ok(self
            .state
            .readback_range(&self.device, &self.queue, range)?)
    }

    /// Read accumulated impulses from the latest contact solve.
    ///
    /// Values describe the final solver substep, not a sum over a frame.
    /// Sleeping contacts may retain warm-start support history. Clearing the
    /// cache returns an empty result with no timestep.
    pub fn readback_contact_impulses(
        &self,
    ) -> Result<GpuRigidContactImpulseReadback, GpuRigidSphereWorldError> {
        if let Some((_, candidates, _)) = &self.last_resident_contacts {
            return Ok(self.resident_impulse_cache.readback(
                &self.device,
                &self.queue,
                candidates,
            )?);
        }
        Ok(self.impulse_cache.readback(&self.device, &self.queue)?)
    }

    /// Run a fresh ray batch against current GPU state and return only query hits.
    pub fn cast_rays(
        &self,
        rays: &[crate::gpu_ray_query::GpuRigidRay],
    ) -> Result<Vec<crate::gpu_ray_query::GpuRigidRayHit>, crate::gpu_ray_query::GpuRayQueryError>
    {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let queries = self.encode_ray_queries(&mut encoder, rays)?;
        let _ = self.queue.submit(Some(encoder.finish()));
        queries.readback(&self.device, &self.queue)
    }

    /// Encode current-state bounds, an independent scene tree and ray queries.
    /// Small scenes use linear search. Large scenes rebuild the tree on every call.
    /// Encode this after physics updates; no state or candidate pairs are mapped.
    pub fn encode_ray_queries(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        rays: &[crate::gpu_ray_query::GpuRigidRay],
    ) -> Result<crate::gpu_ray_query::GpuRigidRayQueries, crate::gpu_ray_query::GpuRayQueryError>
    {
        let tree = if rays.is_empty() {
            None
        } else {
            self.encode_scene_query_tree(encoder)
                .map_err(|e| crate::gpu_ray_query::GpuRayQueryError::SceneIndex(e.to_string()))?
        };
        let queries = if let Some(tree) = tree {
            self.prepare_ray_queries_with_scene_tree(rays, &tree)?
        } else {
            self.prepare_ray_queries(rays)?
        };
        queries.encode(encoder);
        Ok(queries)
    }

    /// Build one current-state scene tree and encode both query types.
    /// The caller submits the encoder and reads back the two returned batches.
    /// Empty query slices produce empty result batches.
    pub fn encode_scene_queries(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        rays: &[crate::gpu_ray_query::GpuRigidRay],
        points: &[crate::gpu_point_query::GpuRigidPoint],
    ) -> Result<
        (
            crate::gpu_ray_query::GpuRigidRayQueries,
            crate::gpu_point_query::GpuRigidPointQueries,
        ),
        GpuSceneQueryError,
    > {
        let tree = if rays.is_empty() && points.is_empty() {
            None
        } else {
            self.encode_scene_query_tree(encoder)?
        };
        let ray_queries = if let Some(tree) = &tree {
            self.prepare_ray_queries_with_scene_tree(rays, tree)?
        } else {
            self.prepare_ray_queries(rays)?
        };
        let point_queries = if let Some(tree) = &tree {
            self.prepare_point_queries_with_scene_tree(points, tree)?
        } else {
            self.prepare_point_queries(points)?
        };
        ray_queries.encode(encoder);
        point_queries.encode(encoder);
        Ok((ray_queries, point_queries))
    }

    /// Query rays and points against one current-state scene snapshot.
    pub fn query_scene(
        &self,
        rays: &[crate::gpu_ray_query::GpuRigidRay],
        points: &[crate::gpu_point_query::GpuRigidPoint],
    ) -> Result<
        (
            Vec<crate::gpu_ray_query::GpuRigidRayHit>,
            Vec<crate::gpu_point_query::GpuRigidPointHit>,
        ),
        GpuSceneQueryError,
    > {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let (ray_queries, point_queries) = self.encode_scene_queries(&mut encoder, rays, points)?;
        let _ = self.queue.submit(Some(encoder.finish()));
        Ok((
            ray_queries.readback(&self.device, &self.queue)?,
            point_queries.readback(&self.device, &self.queue)?,
        ))
    }

    fn encode_scene_query_tree(
        &self,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<Option<crate::gpu_lbvh::GpuLbvhTree>, GpuRigidSphereContactError> {
        if self.len() <= 16 {
            return Ok(None);
        }
        let builder = self.broad_phase.as_ref().unwrap_or_else(|| {
            self.query_broad_phase
                .get_or_init(|| GpuLbvh::new(&self.device))
        });
        let bounds = self.contacts.encode_state_bounds(&self.device, encoder)?;
        let count = u32::try_from(self.len()).map_err(|_| GpuRigidSphereContactError::Capacity)?;
        Ok(Some(builder.encode_tree_resident(
            &self.device,
            encoder,
            &bounds,
            count,
        )?))
    }

    /// Bind an immutable ray batch to this world's live GPU state and geometry.
    /// Prepare after stepping or editing topology: contact buffer replacement invalidates
    /// older bindings. No state is read back.
    pub fn prepare_ray_queries(
        &self,
        rays: &[crate::gpu_ray_query::GpuRigidRay],
    ) -> Result<crate::gpu_ray_query::GpuRigidRayQueries, crate::gpu_ray_query::GpuRayQueryError>
    {
        crate::gpu_ray_query::GpuRigidRayQueries::new(&self.device, &self.contacts, rays)
    }

    /// Bind rays to a scene tree built for the current resident state.
    /// Encode the tree build before these queries; leaves use shared-world body IDs.
    pub fn prepare_ray_queries_with_scene_tree(
        &self,
        rays: &[crate::gpu_ray_query::GpuRigidRay],
        tree: &crate::gpu_lbvh::GpuLbvhTree,
    ) -> Result<crate::gpu_ray_query::GpuRigidRayQueries, crate::gpu_ray_query::GpuRayQueryError>
    {
        crate::gpu_ray_query::GpuRigidRayQueries::with_scene_tree(
            &self.device,
            &self.contacts,
            rays,
            tree,
        )
    }

    /// Bind point projections to current resident geometry without reading body state.
    /// Recreate after stepping or any operation replacing contact or state buffers.
    pub fn prepare_point_queries(
        &self,
        points: &[crate::gpu_point_query::GpuRigidPoint],
    ) -> Result<
        crate::gpu_point_query::GpuRigidPointQueries,
        crate::gpu_point_query::GpuPointQueryError,
    > {
        crate::gpu_point_query::GpuRigidPointQueries::new(&self.device, &self.contacts, points)
    }

    /// Bind projections to a scene tree built for the current resident state.
    /// Encode the tree build before these queries; leaves use shared-world body IDs.
    pub fn prepare_point_queries_with_scene_tree(
        &self,
        points: &[crate::gpu_point_query::GpuRigidPoint],
        tree: &crate::gpu_lbvh::GpuLbvhTree,
    ) -> Result<
        crate::gpu_point_query::GpuRigidPointQueries,
        crate::gpu_point_query::GpuPointQueryError,
    > {
        crate::gpu_point_query::GpuRigidPointQueries::with_scene_tree(
            &self.device,
            &self.contacts,
            points,
            tree,
        )
    }

    /// Project points and synchronously map only the query results.
    pub fn project_points(
        &self,
        points: &[crate::gpu_point_query::GpuRigidPoint],
    ) -> Result<
        Vec<crate::gpu_point_query::GpuRigidPointHit>,
        crate::gpu_point_query::GpuPointQueryError,
    > {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let queries = self.encode_point_queries(&mut encoder, points)?;
        let _ = self.queue.submit(Some(encoder.finish()));
        queries.readback(&self.device, &self.queue)
    }

    /// Encode current-state bounds, an independent scene tree and point queries.
    /// Small scenes use linear search; large scenes rebuild after earlier state updates.
    pub fn encode_point_queries(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        points: &[crate::gpu_point_query::GpuRigidPoint],
    ) -> Result<
        crate::gpu_point_query::GpuRigidPointQueries,
        crate::gpu_point_query::GpuPointQueryError,
    > {
        let tree = if points.is_empty() {
            None
        } else {
            self.encode_scene_query_tree(encoder).map_err(|e| {
                crate::gpu_point_query::GpuPointQueryError::SceneIndex(e.to_string())
            })?
        };
        let queries = if let Some(tree) = tree {
            self.prepare_point_queries_with_scene_tree(points, &tree)?
        } else {
            self.prepare_point_queries(points)?
        };
        queries.encode(encoder);
        Ok(queries)
    }

    /// Transfer the latest pair and ground contact slots for diagnostics.
    pub fn readback_contacts(
        &self,
    ) -> Result<GpuRigidSphereContactReadback, GpuRigidSphereWorldError> {
        if let Some((_, candidates, output)) = &self.last_resident_contacts {
            let mut pairs = output.readback_pairs(&self.device, &self.queue, candidates)?;
            let ground = self.contacts.readback_ground(&self.device, &self.queue)?;
            pairs.ground = ground.ground;
            pairs.ground_extra = ground.ground_extra;
            return Ok(pairs);
        }
        Ok(self.contacts.readback(&self.device, &self.queue)?)
    }
}

/// One independent sphere environment to pack into a shared GPU world.
#[derive(Clone, Copy, Debug)]
pub struct GpuRigidSphereEnvironment<'a> {
    /// Initial rigid states in stable local order.
    pub states: &'a [GpuRigidBodyState],
    /// Sphere radii corresponding to `states`.
    pub radii: &'a [f32],
}

/// One independent mixed primitive environment to pack into a shared GPU world.
#[derive(Clone, Copy, Debug)]
pub struct GpuRigidPrimitiveEnvironment<'a> {
    /// Initial rigid states in stable local order.
    pub states: &'a [GpuRigidBodyState],
    /// Exact collider shapes corresponding to `states`.
    pub shapes: &'a [GpuRigidShape],
}

/// Ray and point queries for one independent GPU environment.
#[derive(Clone, Copy, Debug)]
pub struct GpuRigidEnvironmentSceneQueries<'a> {
    /// Rays with environment-local body exclusions and ranges.
    pub rays: &'a [crate::gpu_ray_query::GpuRigidRay],
    /// Points with environment-local body exclusions and ranges.
    pub points: &'a [crate::gpu_point_query::GpuRigidPoint],
}

/// Ray and point results for one independent GPU environment.
#[derive(Debug)]
pub struct GpuRigidEnvironmentSceneHits {
    /// Ray results with environment-local body IDs.
    pub rays: Vec<crate::gpu_ray_query::GpuRigidRayHit>,
    /// Point results with environment-local body IDs.
    pub points: Vec<crate::gpu_point_query::GpuRigidPointHit>,
}

/// Independent primitive environments sharing one GPU state and solver dispatch.
pub type GpuRigidPrimitiveBatch = GpuRigidSphereBatch;

/// Independent environments sharing one GPU state and solver dispatch.
#[derive(Debug)]
pub struct GpuRigidSphereBatch {
    world: GpuRigidSphereWorld,
    ranges: Vec<Range<usize>>,
}

impl GpuRigidSphereBatch {
    /// Pack environments in order; each has its own contact and solver island.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        environments: &[GpuRigidSphereEnvironment<'_>],
        config: GpuRigidSphereWorldConfig,
    ) -> Result<Self, GpuRigidSphereWorldError> {
        let total = environments.iter().try_fold(0usize, |count, environment| {
            count.checked_add(environment.states.len())
        });
        let Some(total) = total else {
            return Err(GpuRigidSphereWorldError::InvalidInput);
        };
        let mut states = Vec::with_capacity(total);
        let mut radii = Vec::with_capacity(total);
        let mut environment_ids = Vec::with_capacity(total);
        let mut ranges = Vec::with_capacity(environments.len());
        for (index, environment) in environments.iter().enumerate() {
            if environment.states.len() != environment.radii.len() {
                return Err(GpuRigidSphereWorldError::InvalidInput);
            }
            let environment_id =
                u32::try_from(index).map_err(|_| GpuRigidSphereWorldError::InvalidInput)?;
            let start = states.len();
            states.extend_from_slice(environment.states);
            radii.extend_from_slice(environment.radii);
            environment_ids.extend(core::iter::repeat_n(
                environment_id,
                environment.states.len(),
            ));
            ranges.push(start..states.len());
        }
        let world = GpuRigidSphereWorld::new_grouped(
            device,
            queue,
            &states,
            &radii,
            &environment_ids,
            config,
        )?;
        Ok(Self { world, ranges })
    }

    /// Pack mixed primitive environments without generating cross-environment contacts.
    pub fn new_primitives(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        environments: &[GpuRigidPrimitiveEnvironment<'_>],
        config: GpuRigidSphereWorldConfig,
    ) -> Result<Self, GpuRigidSphereWorldError> {
        let total = environments.iter().try_fold(0usize, |count, environment| {
            count.checked_add(environment.states.len())
        });
        let Some(total) = total else {
            return Err(GpuRigidSphereWorldError::InvalidInput);
        };
        let mut states = Vec::with_capacity(total);
        let mut shapes = Vec::with_capacity(total);
        let mut environment_ids = Vec::with_capacity(total);
        let mut ranges = Vec::with_capacity(environments.len());
        for (index, environment) in environments.iter().enumerate() {
            if environment.states.len() != environment.shapes.len() {
                return Err(GpuRigidSphereWorldError::InvalidInput);
            }
            let environment_id =
                u32::try_from(index).map_err(|_| GpuRigidSphereWorldError::InvalidInput)?;
            let start = states.len();
            states.extend_from_slice(environment.states);
            shapes.extend_from_slice(environment.shapes);
            environment_ids.extend(core::iter::repeat_n(
                environment_id,
                environment.states.len(),
            ));
            ranges.push(start..states.len());
        }
        let world = GpuRigidSphereWorld::new_grouped_primitives(
            device,
            queue,
            &states,
            &shapes,
            &environment_ids,
            config,
        )?;
        Ok(Self { world, ranges })
    }

    /// Append an independent sphere environment while preserving existing GPU states.
    /// The new environment may be empty. Topology changes reset contact and sleep history.
    pub fn append_environment(
        &mut self,
        states: &[GpuRigidBodyState],
        radii: &[f32],
    ) -> Result<usize, GpuRigidSphereWorldError> {
        if states.len() != radii.len() {
            return Err(GpuRigidSphereWorldError::InvalidInput);
        }
        self.append_environment_inner(states, radii, None)
    }

    /// Append an independent mixed-shape environment without reading surviving states.
    pub fn append_environment_primitives(
        &mut self,
        states: &[GpuRigidBodyState],
        shapes: &[GpuRigidShape],
    ) -> Result<usize, GpuRigidSphereWorldError> {
        if states.len() != shapes.len() {
            return Err(GpuRigidSphereWorldError::InvalidInput);
        }
        let radii = shapes
            .iter()
            .map(|shape| {
                shape
                    .bounding_radius()
                    .ok_or(GpuRigidSphereWorldError::InvalidInput)
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.append_environment_inner(states, &radii, Some(shapes))
    }

    fn append_environment_inner(
        &mut self,
        states: &[GpuRigidBodyState],
        radii: &[f32],
        shapes: Option<&[GpuRigidShape]>,
    ) -> Result<usize, GpuRigidSphereWorldError> {
        if self.world.faulted {
            return Err(GpuRigidSphereWorldError::Faulted);
        }
        if self.world.shapes.is_some() != shapes.is_some() {
            return Err(GpuRigidSphereWorldError::InvalidInput);
        }
        let environment = self.ranges.len();
        let environment_id =
            u32::try_from(environment).map_err(|_| GpuRigidSphereWorldError::InvalidInput)?;
        let start = self.world.len();
        let mut next_radii = self.world.radii.clone();
        next_radii.extend_from_slice(radii);
        let mut next_shapes = self.world.shapes.clone();
        if let (Some(next), Some(additions)) = (&mut next_shapes, shapes) {
            next.extend_from_slice(additions);
        }
        let mut materials = self.world.body_materials.clone();
        let end = start
            .checked_add(states.len())
            .ok_or(GpuRigidSphereWorldError::InvalidInput)?;
        materials.resize(end, None);
        let mut collision_groups = self.world.collision_groups_snapshot();
        collision_groups.resize(end, GpuRigidCollisionGroups::default());
        let mut environment_ids = self.world.environment_ids.clone();
        environment_ids.extend(core::iter::repeat_n(environment_id, states.len()));
        self.world.rebuild_topology(
            &next_radii,
            next_shapes.as_deref(),
            &materials,
            &collision_groups,
            &environment_ids,
            TopologyEdit::InsertRange(start, states.to_vec()),
        )?;
        self.ranges.push(start..end);
        Ok(environment)
    }

    /// Remove a sphere environment and return only its states to the CPU.
    /// Surviving environments keep their order and joint settings.
    pub fn remove_environment(
        &mut self,
        environment: usize,
    ) -> Result<Vec<GpuRigidBodyState>, GpuRigidSphereWorldError> {
        let (states, shapes) = self.remove_environment_inner(environment, false, true)?;
        debug_assert!(shapes.is_none());
        states.ok_or(GpuRigidSphereWorldError::InvalidInput)
    }

    /// Remove a mixed-shape environment and return its states and shapes.
    pub fn remove_environment_primitives(
        &mut self,
        environment: usize,
    ) -> Result<Vec<(GpuRigidBodyState, GpuRigidShape)>, GpuRigidSphereWorldError> {
        let (states, shapes) = self.remove_environment_inner(environment, true, true)?;
        Ok(states
            .ok_or(GpuRigidSphereWorldError::InvalidInput)?
            .into_iter()
            .zip(shapes.ok_or(GpuRigidSphereWorldError::InvalidInput)?)
            .collect())
    }

    /// Discard a sphere environment without reading its body states from the GPU.
    pub fn discard_environment(
        &mut self,
        environment: usize,
    ) -> Result<(), GpuRigidSphereWorldError> {
        let _removed = self.remove_environment_inner(environment, false, false)?;
        Ok(())
    }

    /// Discard a mixed-shape environment without reading its body states from the GPU.
    pub fn discard_environment_primitives(
        &mut self,
        environment: usize,
    ) -> Result<(), GpuRigidSphereWorldError> {
        let _removed = self.remove_environment_inner(environment, true, false)?;
        Ok(())
    }

    fn remove_environment_inner(
        &mut self,
        environment: usize,
        primitive: bool,
        readback: bool,
    ) -> Result<RemovedEnvironment, GpuRigidSphereWorldError> {
        if self.world.faulted {
            return Err(GpuRigidSphereWorldError::Faulted);
        }
        if self.world.shapes.is_some() != primitive {
            return Err(GpuRigidSphereWorldError::InvalidInput);
        }
        let range = self
            .ranges
            .get(environment)
            .cloned()
            .ok_or(GpuRigidSphereWorldError::InvalidInput)?;
        let states = if readback {
            Some(self.world.readback_range(range.clone())?)
        } else {
            None
        };
        let mut radii = self.world.radii.clone();
        drop(radii.drain(range.clone()));
        let mut shapes = self.world.shapes.clone();
        let removed_shapes = shapes
            .as_mut()
            .map(|shapes| shapes.drain(range.clone()).collect());
        let mut materials = self.world.body_materials.clone();
        drop(materials.drain(range.clone()));
        let mut collision_groups = self.world.collision_groups_snapshot();
        drop(collision_groups.drain(range.clone()));
        let mut environment_ids = self.world.environment_ids.clone();
        drop(environment_ids.drain(range.clone()));
        let removed_id =
            u32::try_from(environment).map_err(|_| GpuRigidSphereWorldError::InvalidInput)?;
        for id in &mut environment_ids {
            if *id > removed_id {
                *id -= 1;
            }
        }
        self.world.rebuild_topology(
            &radii,
            shapes.as_deref(),
            &materials,
            &collision_groups,
            &environment_ids,
            TopologyEdit::RemoveRange(range.clone()),
        )?;
        let _removed_range = self.ranges.remove(environment);
        for later in &mut self.ranges[environment..] {
            later.start -= range.len();
            later.end -= range.len();
        }
        Ok((states, removed_shapes))
    }

    /// Append a sphere to one environment and return its local dense index.
    ///
    /// Surviving body states are copied on the GPU. Topology edits discard
    /// queued forces, sleep timers, and warm-start impulses for the batch.
    /// Surviving joints retain their settings and angle history.
    pub fn append_body_environment(
        &mut self,
        environment: usize,
        state: GpuRigidBodyState,
        radius: f32,
    ) -> Result<usize, GpuRigidSphereWorldError> {
        if !radius.is_finite() || radius <= 0.0 {
            return Err(GpuRigidSphereWorldError::InvalidInput);
        }
        self.insert_environment_body(environment, state, radius, None)
    }

    /// Append a primitive to one environment and return its local dense index.
    pub fn append_primitive_environment(
        &mut self,
        environment: usize,
        state: GpuRigidBodyState,
        shape: GpuRigidShape,
    ) -> Result<usize, GpuRigidSphereWorldError> {
        let radius = shape
            .bounding_radius()
            .ok_or(GpuRigidSphereWorldError::InvalidInput)?;
        self.insert_environment_body(environment, state, radius, Some(shape))
    }

    fn insert_environment_body(
        &mut self,
        environment: usize,
        state: GpuRigidBodyState,
        radius: f32,
        shape: Option<GpuRigidShape>,
    ) -> Result<usize, GpuRigidSphereWorldError> {
        if self.world.faulted {
            return Err(GpuRigidSphereWorldError::Faulted);
        }
        let range = self
            .ranges
            .get(environment)
            .ok_or(GpuRigidSphereWorldError::InvalidInput)?;
        if self.world.shapes.is_some() != shape.is_some() || !state.is_valid() {
            return Err(GpuRigidSphereWorldError::InvalidInput);
        }
        let environment_id =
            u32::try_from(environment).map_err(|_| GpuRigidSphereWorldError::InvalidInput)?;
        let local_index = range.len();
        let index = range.end;
        let mut radii = self.world.radii.clone();
        radii.insert(index, radius);
        let mut shapes = self.world.shapes.clone();
        if let (Some(shapes), Some(shape)) = (&mut shapes, shape) {
            shapes.insert(index, shape);
        }
        let mut materials = self.world.body_materials.clone();
        materials.insert(index, None);
        let mut collision_groups = self.world.collision_groups_snapshot();
        collision_groups.insert(index, GpuRigidCollisionGroups::default());
        let mut environment_ids = self.world.environment_ids.clone();
        environment_ids.insert(index, environment_id);
        self.world.rebuild_topology(
            &radii,
            shapes.as_deref(),
            &materials,
            &collision_groups,
            &environment_ids,
            TopologyEdit::Insert(index, Box::new(state)),
        )?;
        self.ranges[environment].end += 1;
        for range in &mut self.ranges[(environment + 1)..] {
            range.start += 1;
            range.end += 1;
        }
        Ok(local_index)
    }

    /// Remove a sphere by environment-local index and return its GPU state.
    /// Attached joints are removed; surviving joints keep their settings.
    pub fn remove_body_environment(
        &mut self,
        environment: usize,
        body: usize,
    ) -> Result<GpuRigidBodyState, GpuRigidSphereWorldError> {
        let (state, shape) = self.remove_environment_body(environment, body, false, true)?;
        debug_assert!(shape.is_none());
        state.ok_or(GpuRigidSphereWorldError::InvalidInput)
    }

    /// Remove a primitive by environment-local index and return its state and shape.
    /// Attached joints are removed; surviving joints keep their settings.
    pub fn remove_primitive_environment(
        &mut self,
        environment: usize,
        body: usize,
    ) -> Result<(GpuRigidBodyState, GpuRigidShape), GpuRigidSphereWorldError> {
        let (state, shape) = self.remove_environment_body(environment, body, true, true)?;
        Ok((
            state.ok_or(GpuRigidSphereWorldError::InvalidInput)?,
            shape.ok_or(GpuRigidSphereWorldError::InvalidInput)?,
        ))
    }

    /// Discard a sphere by environment-local index without a GPU state readback.
    pub fn discard_body_environment(
        &mut self,
        environment: usize,
        body: usize,
    ) -> Result<(), GpuRigidSphereWorldError> {
        let _removed = self.remove_environment_body(environment, body, false, false)?;
        Ok(())
    }

    /// Discard a primitive by environment-local index without a GPU state readback.
    pub fn discard_primitive_environment(
        &mut self,
        environment: usize,
        body: usize,
    ) -> Result<(), GpuRigidSphereWorldError> {
        let _removed = self.remove_environment_body(environment, body, true, false)?;
        Ok(())
    }

    fn remove_environment_body(
        &mut self,
        environment: usize,
        body: usize,
        primitive: bool,
        readback: bool,
    ) -> Result<(Option<GpuRigidBodyState>, Option<GpuRigidShape>), GpuRigidSphereWorldError> {
        if self.world.faulted {
            return Err(GpuRigidSphereWorldError::Faulted);
        }
        let range = self
            .ranges
            .get(environment)
            .ok_or(GpuRigidSphereWorldError::InvalidInput)?;
        if body >= range.len() || self.world.shapes.is_some() != primitive {
            return Err(GpuRigidSphereWorldError::InvalidInput);
        }
        let index = range.start + body;
        let state = if readback {
            Some(self.world.readback_range(index..index + 1)?[0])
        } else {
            None
        };
        let mut radii = self.world.radii.clone();
        let _removed_radius = radii.remove(index);
        let mut shapes = self.world.shapes.clone();
        let shape = shapes.as_mut().map(|shapes| shapes.remove(index));
        let mut materials = self.world.body_materials.clone();
        let _removed_material = materials.remove(index);
        let mut collision_groups = self.world.collision_groups_snapshot();
        let _removed_groups = collision_groups.remove(index);
        let mut environment_ids = self.world.environment_ids.clone();
        let _removed_environment = environment_ids.remove(index);
        self.world.rebuild_topology(
            &radii,
            shapes.as_deref(),
            &materials,
            &collision_groups,
            &environment_ids,
            TopologyEdit::Remove(index),
        )?;
        self.ranges[environment].end -= 1;
        for range in &mut self.ranges[(environment + 1)..] {
            range.start -= 1;
            range.end -= 1;
        }
        Ok((state, shape))
    }

    /// Number of independent environments.
    pub fn len(&self) -> usize {
        self.ranges.len()
    }

    /// Whether the batch contains no environments.
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// Stable body range for one environment in the shared world.
    pub fn environment_range(&self, index: usize) -> Option<Range<usize>> {
        self.ranges.get(index).cloned()
    }

    /// Run rays in one environment and return environment-local body IDs.
    pub fn cast_rays_environment(
        &self,
        environment: usize,
        rays: &[crate::gpu_ray_query::GpuRigidRay],
    ) -> Result<Vec<crate::gpu_ray_query::GpuRigidRayHit>, crate::gpu_ray_query::GpuRayQueryError>
    {
        let local = self.local_ray_queries_environment(environment, rays)?;
        let mut hits = self.world.cast_rays(&local)?;
        let start = self.ranges[environment].start as u32;
        for hit in &mut hits {
            if hit.ids[2] != 0 && hit.ids[0] != u32::MAX {
                hit.ids[0] -= start;
            }
        }
        Ok(hits)
    }

    /// Query rays and points in one environment with one scene-tree build.
    /// Exclusion, ranges, and returned body IDs are environment-local.
    pub fn query_scene_environment(
        &self,
        environment: usize,
        rays: &[crate::gpu_ray_query::GpuRigidRay],
        points: &[crate::gpu_point_query::GpuRigidPoint],
    ) -> Result<
        (
            Vec<crate::gpu_ray_query::GpuRigidRayHit>,
            Vec<crate::gpu_point_query::GpuRigidPointHit>,
        ),
        GpuSceneQueryError,
    > {
        let local_rays = self.local_ray_queries_environment(environment, rays)?;
        let local_points = self.local_point_queries_environment(environment, points)?;
        let (mut ray_hits, mut point_hits) = self.world.query_scene(&local_rays, &local_points)?;
        let start = self.ranges[environment].start as u32;
        for hit in &mut ray_hits {
            if hit.ids[2] != 0 && hit.ids[0] != u32::MAX {
                hit.ids[0] -= start;
            }
        }
        for hit in &mut point_hits {
            if hit.ids[2] != 0 && hit.ids[0] != u32::MAX {
                hit.ids[0] -= start;
            }
        }
        Ok((ray_hits, point_hits))
    }

    /// Query every environment against one current-state GPU scene snapshot.
    ///
    /// The input must contain one entry per environment. Query ranges, exclusions,
    /// and hit body IDs use environment-local indices. Empty entries are valid.
    pub fn query_scene_environments(
        &self,
        queries: &[GpuRigidEnvironmentSceneQueries<'_>],
    ) -> Result<Vec<GpuRigidEnvironmentSceneHits>, GpuSceneQueryError> {
        if queries.len() != self.ranges.len() {
            return Err(GpuSceneQueryError::InvalidInput);
        }
        let mut rays = Vec::new();
        let mut points = Vec::new();
        for (environment, query) in queries.iter().enumerate() {
            rays.extend(self.local_ray_queries_environment(environment, query.rays)?);
            points.extend(self.local_point_queries_environment(environment, query.points)?);
        }
        let (ray_hits, point_hits) = self.world.query_scene(&rays, &points)?;
        let mut ray_hits = ray_hits.into_iter();
        let mut point_hits = point_hits.into_iter();
        let mut results = Vec::with_capacity(queries.len());
        for (environment, query) in queries.iter().enumerate() {
            let start = u32::try_from(self.ranges[environment].start)
                .map_err(|_| GpuSceneQueryError::InvalidInput)?;
            let mut environment_rays = ray_hits.by_ref().take(query.rays.len()).collect::<Vec<_>>();
            let mut environment_points = point_hits
                .by_ref()
                .take(query.points.len())
                .collect::<Vec<_>>();
            for hit in &mut environment_rays {
                if hit.ids[2] != 0 && hit.ids[0] != u32::MAX {
                    hit.ids[0] -= start;
                }
            }
            for hit in &mut environment_points {
                if hit.ids[2] != 0 && hit.ids[0] != u32::MAX {
                    hit.ids[0] -= start;
                }
            }
            results.push(GpuRigidEnvironmentSceneHits {
                rays: environment_rays,
                points: environment_points,
            });
        }
        Ok(results)
    }

    /// Bind rays to one independent environment. Exclusion and input ranges use local IDs.
    /// Hit body IDs remain shared-world IDs; subtract the environment range start.
    pub fn prepare_ray_queries_environment(
        &self,
        environment: usize,
        rays: &[crate::gpu_ray_query::GpuRigidRay],
    ) -> Result<crate::gpu_ray_query::GpuRigidRayQueries, crate::gpu_ray_query::GpuRayQueryError>
    {
        self.world
            .prepare_ray_queries(&self.local_ray_queries_environment(environment, rays)?)
    }

    fn local_ray_queries_environment(
        &self,
        environment: usize,
        rays: &[crate::gpu_ray_query::GpuRigidRay],
    ) -> Result<Vec<crate::gpu_ray_query::GpuRigidRay>, crate::gpu_ray_query::GpuRayQueryError>
    {
        use crate::gpu_ray_query::GpuRayQueryError;
        let range = self
            .ranges
            .get(environment)
            .ok_or(GpuRayQueryError::InvalidInput)?;
        let start = u32::try_from(range.start).map_err(|_| GpuRayQueryError::InvalidInput)?;
        let count = u32::try_from(range.len()).map_err(|_| GpuRayQueryError::InvalidInput)?;
        let mut local = Vec::with_capacity(rays.len());
        for ray in rays {
            let bounds = ray.body_range.unwrap_or([0, count]);
            if bounds[0] > bounds[1]
                || bounds[1] > count
                || ray.excluded_body.is_some_and(|index| index >= count)
            {
                return Err(GpuRayQueryError::InvalidInput);
            }
            let mut ray = *ray;
            ray.body_range = Some([start + bounds[0], start + bounds[1]]);
            ray.excluded_body = ray.excluded_body.map(|index| start + index);
            local.push(ray);
        }
        Ok(local)
    }

    /// Project points in one environment and return environment-local body IDs.
    pub fn project_points_environment(
        &self,
        environment: usize,
        points: &[crate::gpu_point_query::GpuRigidPoint],
    ) -> Result<
        Vec<crate::gpu_point_query::GpuRigidPointHit>,
        crate::gpu_point_query::GpuPointQueryError,
    > {
        let local = self.local_point_queries_environment(environment, points)?;
        let mut hits = self.world.project_points(&local)?;
        let start = self.ranges[environment].start as u32;
        for hit in &mut hits {
            if hit.ids[2] != 0 && hit.ids[0] != u32::MAX {
                hit.ids[0] -= start;
            }
        }
        Ok(hits)
    }

    /// Bind points to one independent environment. Exclusion and input ranges use local IDs.
    /// Hit body IDs remain shared-world IDs; subtract the environment range start.
    pub fn prepare_point_queries_environment(
        &self,
        environment: usize,
        points: &[crate::gpu_point_query::GpuRigidPoint],
    ) -> Result<
        crate::gpu_point_query::GpuRigidPointQueries,
        crate::gpu_point_query::GpuPointQueryError,
    > {
        self.world
            .prepare_point_queries(&self.local_point_queries_environment(environment, points)?)
    }

    fn local_point_queries_environment(
        &self,
        environment: usize,
        points: &[crate::gpu_point_query::GpuRigidPoint],
    ) -> Result<
        Vec<crate::gpu_point_query::GpuRigidPoint>,
        crate::gpu_point_query::GpuPointQueryError,
    > {
        use crate::gpu_point_query::GpuPointQueryError;
        let range = self
            .ranges
            .get(environment)
            .ok_or(GpuPointQueryError::InvalidInput)?;
        let start = u32::try_from(range.start).map_err(|_| GpuPointQueryError::InvalidInput)?;
        let count = u32::try_from(range.len()).map_err(|_| GpuPointQueryError::InvalidInput)?;
        let mut local = Vec::with_capacity(points.len());
        for point in points {
            let bounds = point.body_range.unwrap_or([0, count]);
            if bounds[0] > bounds[1]
                || bounds[1] > count
                || point.excluded_body.is_some_and(|index| index >= count)
            {
                return Err(GpuPointQueryError::InvalidInput);
            }
            let mut point = *point;
            point.body_range = Some([start + bounds[0], start + bounds[1]]);
            point.excluded_body = point.excluded_body.map(|index| start + index);
            local.push(point);
        }
        Ok(local)
    }

    /// Access the shared world for configuration and per-body materials.
    pub fn world(&self) -> &GpuRigidSphereWorld {
        &self.world
    }

    /// Mutate the shared world without changing environment membership.
    pub fn world_mut(&mut self) -> &mut GpuRigidSphereWorld {
        &mut self.world
    }

    /// Replace one environment's ball joints using environment-local body IDs.
    pub fn set_ball_joints_environment(
        &mut self,
        environment: usize,
        joints: &[GpuRigidBallJoint],
    ) -> Result<(), GpuRigidSphereWorldError> {
        let range = self
            .ranges
            .get(environment)
            .ok_or(GpuRigidSphereWorldError::InvalidInput)?;
        if joints.iter().any(|joint| {
            joint.body_a as usize >= range.len() || joint.body_b as usize >= range.len()
        }) {
            return Err(GpuRigidSphereWorldError::InvalidInput);
        }
        let start =
            u32::try_from(range.start).map_err(|_| GpuRigidSphereWorldError::InvalidInput)?;
        let end = u32::try_from(range.end).map_err(|_| GpuRigidSphereWorldError::InvalidInput)?;
        let mut all = self
            .world
            .ball_joints()
            .iter()
            .copied()
            .filter(|joint| !(start..end).contains(&joint.body_a))
            .collect::<Vec<_>>();
        for joint in joints {
            all.push(GpuRigidBallJoint {
                body_a: joint
                    .body_a
                    .checked_add(start)
                    .ok_or(GpuRigidSphereWorldError::InvalidInput)?,
                body_b: joint
                    .body_b
                    .checked_add(start)
                    .ok_or(GpuRigidSphereWorldError::InvalidInput)?,
                ..*joint
            });
        }
        self.world.set_ball_joints(&all)
    }

    /// One environment's ball joints with local body IDs.
    pub fn ball_joints_environment(
        &self,
        environment: usize,
    ) -> Result<Vec<GpuRigidBallJoint>, GpuRigidSphereWorldError> {
        let range = self
            .ranges
            .get(environment)
            .ok_or(GpuRigidSphereWorldError::InvalidInput)?;
        let start =
            u32::try_from(range.start).map_err(|_| GpuRigidSphereWorldError::InvalidInput)?;
        let end = u32::try_from(range.end).map_err(|_| GpuRigidSphereWorldError::InvalidInput)?;
        Ok(self
            .world
            .ball_joints()
            .iter()
            .filter(|joint| (start..end).contains(&joint.body_a))
            .map(|joint| GpuRigidBallJoint {
                body_a: joint.body_a - start,
                body_b: joint.body_b - start,
                ..*joint
            })
            .collect())
    }

    /// Replace one environment's fixed joints using local body IDs.
    pub fn set_fixed_joints_environment(
        &mut self,
        environment: usize,
        joints: &[GpuRigidFixedJoint],
    ) -> Result<(), GpuRigidSphereWorldError> {
        let range = self
            .ranges
            .get(environment)
            .ok_or(GpuRigidSphereWorldError::InvalidInput)?;
        if joints.iter().any(|joint| {
            joint.body_a as usize >= range.len() || joint.body_b as usize >= range.len()
        }) {
            return Err(GpuRigidSphereWorldError::InvalidInput);
        }
        let start =
            u32::try_from(range.start).map_err(|_| GpuRigidSphereWorldError::InvalidInput)?;
        let end = u32::try_from(range.end).map_err(|_| GpuRigidSphereWorldError::InvalidInput)?;
        let mut all = self
            .world
            .fixed_joints()
            .iter()
            .copied()
            .filter(|joint| !(start..end).contains(&joint.body_a))
            .collect::<Vec<_>>();
        for joint in joints {
            all.push(GpuRigidFixedJoint {
                body_a: joint
                    .body_a
                    .checked_add(start)
                    .ok_or(GpuRigidSphereWorldError::InvalidInput)?,
                body_b: joint
                    .body_b
                    .checked_add(start)
                    .ok_or(GpuRigidSphereWorldError::InvalidInput)?,
                ..*joint
            });
        }
        self.world.set_fixed_joints(&all)
    }

    /// One environment's fixed joints with local body IDs.
    pub fn fixed_joints_environment(
        &self,
        environment: usize,
    ) -> Result<Vec<GpuRigidFixedJoint>, GpuRigidSphereWorldError> {
        let range = self
            .ranges
            .get(environment)
            .ok_or(GpuRigidSphereWorldError::InvalidInput)?;
        let start =
            u32::try_from(range.start).map_err(|_| GpuRigidSphereWorldError::InvalidInput)?;
        let end = u32::try_from(range.end).map_err(|_| GpuRigidSphereWorldError::InvalidInput)?;
        Ok(self
            .world
            .fixed_joints()
            .iter()
            .filter(|joint| (start..end).contains(&joint.body_a))
            .map(|joint| GpuRigidFixedJoint {
                body_a: joint.body_a - start,
                body_b: joint.body_b - start,
                ..*joint
            })
            .collect())
    }

    /// Replace one environment's revolute joints using local body IDs.
    pub fn set_revolute_joints_environment(
        &mut self,
        environment: usize,
        joints: &[GpuRigidRevoluteJoint],
    ) -> Result<(), GpuRigidSphereWorldError> {
        let range = self
            .ranges
            .get(environment)
            .ok_or(GpuRigidSphereWorldError::InvalidInput)?;
        if joints.iter().any(|joint| {
            joint.body_a as usize >= range.len() || joint.body_b as usize >= range.len()
        }) {
            return Err(GpuRigidSphereWorldError::InvalidInput);
        }
        let start =
            u32::try_from(range.start).map_err(|_| GpuRigidSphereWorldError::InvalidInput)?;
        let end = u32::try_from(range.end).map_err(|_| GpuRigidSphereWorldError::InvalidInput)?;
        let mut all = self
            .world
            .revolute_joints()
            .iter()
            .copied()
            .filter(|joint| !(start..end).contains(&joint.body_a))
            .collect::<Vec<_>>();
        for joint in joints {
            all.push(GpuRigidRevoluteJoint {
                body_a: joint
                    .body_a
                    .checked_add(start)
                    .ok_or(GpuRigidSphereWorldError::InvalidInput)?,
                body_b: joint
                    .body_b
                    .checked_add(start)
                    .ok_or(GpuRigidSphereWorldError::InvalidInput)?,
                ..*joint
            });
        }
        self.world.set_revolute_joints(&all)
    }

    /// One environment's revolute joints with local body IDs.
    pub fn revolute_joints_environment(
        &self,
        environment: usize,
    ) -> Result<Vec<GpuRigidRevoluteJoint>, GpuRigidSphereWorldError> {
        let range = self
            .ranges
            .get(environment)
            .ok_or(GpuRigidSphereWorldError::InvalidInput)?;
        let start =
            u32::try_from(range.start).map_err(|_| GpuRigidSphereWorldError::InvalidInput)?;
        let end = u32::try_from(range.end).map_err(|_| GpuRigidSphereWorldError::InvalidInput)?;
        Ok(self
            .world
            .revolute_joints()
            .iter()
            .filter(|joint| (start..end).contains(&joint.body_a))
            .map(|joint| GpuRigidRevoluteJoint {
                body_a: joint.body_a - start,
                body_b: joint.body_b - start,
                ..*joint
            })
            .collect())
    }

    /// Replace one environment's prismatic joints using local body IDs.
    pub fn set_prismatic_joints_environment(
        &mut self,
        environment: usize,
        joints: &[GpuRigidPrismaticJoint],
    ) -> Result<(), GpuRigidSphereWorldError> {
        let range = self
            .ranges
            .get(environment)
            .ok_or(GpuRigidSphereWorldError::InvalidInput)?;
        if joints.iter().any(|joint| {
            joint.body_a as usize >= range.len() || joint.body_b as usize >= range.len()
        }) {
            return Err(GpuRigidSphereWorldError::InvalidInput);
        }
        let start =
            u32::try_from(range.start).map_err(|_| GpuRigidSphereWorldError::InvalidInput)?;
        let end = u32::try_from(range.end).map_err(|_| GpuRigidSphereWorldError::InvalidInput)?;
        let mut all = self
            .world
            .prismatic_joints()
            .iter()
            .copied()
            .filter(|joint| !(start..end).contains(&joint.body_a))
            .collect::<Vec<_>>();
        for joint in joints {
            all.push(GpuRigidPrismaticJoint {
                body_a: joint
                    .body_a
                    .checked_add(start)
                    .ok_or(GpuRigidSphereWorldError::InvalidInput)?,
                body_b: joint
                    .body_b
                    .checked_add(start)
                    .ok_or(GpuRigidSphereWorldError::InvalidInput)?,
                ..*joint
            });
        }
        self.world.set_prismatic_joints(&all)
    }

    /// One environment's prismatic joints with local body IDs.
    pub fn prismatic_joints_environment(
        &self,
        environment: usize,
    ) -> Result<Vec<GpuRigidPrismaticJoint>, GpuRigidSphereWorldError> {
        let range = self
            .ranges
            .get(environment)
            .ok_or(GpuRigidSphereWorldError::InvalidInput)?;
        let start =
            u32::try_from(range.start).map_err(|_| GpuRigidSphereWorldError::InvalidInput)?;
        let end = u32::try_from(range.end).map_err(|_| GpuRigidSphereWorldError::InvalidInput)?;
        Ok(self
            .world
            .prismatic_joints()
            .iter()
            .filter(|joint| (start..end).contains(&joint.body_a))
            .map(|joint| GpuRigidPrismaticJoint {
                body_a: joint.body_a - start,
                body_b: joint.body_b - start,
                ..*joint
            })
            .collect())
    }

    fn axis_joint_index(
        &self,
        environment: usize,
        index: usize,
        prismatic: bool,
    ) -> Result<usize, GpuRigidSphereWorldError> {
        let range = self
            .ranges
            .get(environment)
            .ok_or(GpuRigidSphereWorldError::InvalidInput)?;
        let start =
            u32::try_from(range.start).map_err(|_| GpuRigidSphereWorldError::InvalidInput)?;
        let end = u32::try_from(range.end).map_err(|_| GpuRigidSphereWorldError::InvalidInput)?;
        let joints = if prismatic {
            self.world
                .prismatic_joints()
                .iter()
                .map(|joint| joint.body_a)
                .collect::<Vec<_>>()
        } else {
            self.world
                .revolute_joints()
                .iter()
                .map(|joint| joint.body_a)
                .collect::<Vec<_>>()
        };
        joints
            .iter()
            .enumerate()
            .filter(|(_, body_a)| (start..end).contains(body_a))
            .nth(index)
            .map(|(index, _)| index)
            .ok_or(GpuRigidSphereWorldError::InvalidInput)
    }

    /// Set one environment-local revolute joint's velocity drive.
    pub fn set_revolute_motor_environment(
        &mut self,
        environment: usize,
        index: usize,
        motor: Option<GpuRigidAxisMotor>,
    ) -> Result<(), GpuRigidSphereWorldError> {
        let index = self.axis_joint_index(environment, index, false)?;
        self.world.set_revolute_motor(index, motor)
    }

    /// Read one environment-local revolute joint's velocity drive.
    pub fn revolute_motor_environment(
        &self,
        environment: usize,
        index: usize,
    ) -> Result<Option<GpuRigidAxisMotor>, GpuRigidSphereWorldError> {
        let index = self.axis_joint_index(environment, index, false)?;
        self.world.revolute_motor(index)
    }

    /// Set one environment-local prismatic joint's velocity drive.
    pub fn set_prismatic_motor_environment(
        &mut self,
        environment: usize,
        index: usize,
        motor: Option<GpuRigidAxisMotor>,
    ) -> Result<(), GpuRigidSphereWorldError> {
        let index = self.axis_joint_index(environment, index, true)?;
        self.world.set_prismatic_motor(index, motor)
    }

    /// Read one environment-local prismatic joint's velocity drive.
    pub fn prismatic_motor_environment(
        &self,
        environment: usize,
        index: usize,
    ) -> Result<Option<GpuRigidAxisMotor>, GpuRigidSphereWorldError> {
        let index = self.axis_joint_index(environment, index, true)?;
        self.world.prismatic_motor(index)
    }

    /// Set one environment-local prismatic joint's displacement limits.
    pub fn set_prismatic_limit_environment(
        &mut self,
        environment: usize,
        index: usize,
        limit: Option<GpuRigidPrismaticLimit>,
    ) -> Result<(), GpuRigidSphereWorldError> {
        let index = self.axis_joint_index(environment, index, true)?;
        self.world.set_prismatic_limit(index, limit)
    }

    /// Read one environment-local prismatic joint's displacement limits.
    pub fn prismatic_limit_environment(
        &self,
        environment: usize,
        index: usize,
    ) -> Result<Option<GpuRigidPrismaticLimit>, GpuRigidSphereWorldError> {
        let index = self.axis_joint_index(environment, index, true)?;
        self.world.prismatic_limit(index)
    }

    /// Set one environment-local revolute joint's position servo.
    pub fn set_revolute_servo_environment(
        &mut self,
        environment: usize,
        index: usize,
        servo: Option<GpuRigidAxisServo>,
    ) -> Result<(), GpuRigidSphereWorldError> {
        let index = self.axis_joint_index(environment, index, false)?;
        self.world.set_revolute_servo(index, servo)
    }

    /// Read one environment-local revolute joint's position servo.
    pub fn revolute_servo_environment(
        &self,
        environment: usize,
        index: usize,
    ) -> Result<Option<GpuRigidAxisServo>, GpuRigidSphereWorldError> {
        let index = self.axis_joint_index(environment, index, false)?;
        self.world.revolute_servo(index)
    }

    /// Set one environment-local revolute joint's angle limits.
    pub fn set_revolute_limit_environment(
        &mut self,
        environment: usize,
        index: usize,
        limit: Option<GpuRigidRevoluteLimit>,
    ) -> Result<(), GpuRigidSphereWorldError> {
        let index = self.axis_joint_index(environment, index, false)?;
        self.world.set_revolute_limit(index, limit)
    }

    /// Read one environment-local revolute joint's angle limits.
    pub fn revolute_limit_environment(
        &self,
        environment: usize,
        index: usize,
    ) -> Result<Option<GpuRigidRevoluteLimit>, GpuRigidSphereWorldError> {
        let index = self.axis_joint_index(environment, index, false)?;
        self.world.revolute_limit(index)
    }

    /// Read one environment-local revolute joint's continuous angle.
    pub fn readback_revolute_angle_environment(
        &self,
        environment: usize,
        index: usize,
    ) -> Result<f32, GpuRigidSphereWorldError> {
        let index = self.axis_joint_index(environment, index, false)?;
        self.world.readback_revolute_angle(index)
    }

    /// Set one environment-local prismatic joint's position servo.
    pub fn set_prismatic_servo_environment(
        &mut self,
        environment: usize,
        index: usize,
        servo: Option<GpuRigidAxisServo>,
    ) -> Result<(), GpuRigidSphereWorldError> {
        let index = self.axis_joint_index(environment, index, true)?;
        self.world.set_prismatic_servo(index, servo)
    }

    /// Read one environment-local prismatic joint's position servo.
    pub fn prismatic_servo_environment(
        &self,
        environment: usize,
        index: usize,
    ) -> Result<Option<GpuRigidAxisServo>, GpuRigidSphereWorldError> {
        let index = self.axis_joint_index(environment, index, true)?;
        self.world.prismatic_servo(index)
    }

    /// Advance every environment together by one substep.
    pub fn step(&mut self, dt: f32) -> Result<usize, GpuRigidSphereWorldError> {
        self.world.step(dt)
    }

    /// Advance every environment through several substeps.
    pub fn step_substeps(
        &mut self,
        dt: f32,
        count: u32,
    ) -> Result<usize, GpuRigidSphereWorldError> {
        self.world.step_substeps(dt, count)
    }

    /// Advance every environment through one temporal frame with one force capture.
    pub fn step_temporal(
        &mut self,
        frame_dt: f32,
        substeps: u32,
        settings: GpuRigidTemporalSolveParams,
    ) -> Result<usize, GpuRigidSphereWorldError> {
        self.world.step_temporal(frame_dt, substeps, settings)
    }

    /// Advance every environment with initial speculative pair and ground contacts.
    pub fn step_temporal_speculative(
        &mut self,
        frame_dt: f32,
        substeps: u32,
        settings: GpuRigidTemporalSolveParams,
        margin: f32,
    ) -> Result<usize, GpuRigidSphereWorldError> {
        self.world
            .step_temporal_speculative(frame_dt, substeps, settings, margin)
    }

    /// Advance sphere environments using the shared speed cap as a motion bound.
    pub fn step_temporal_speed_bounded(
        &mut self,
        frame_dt: f32,
        substeps: u32,
        settings: GpuRigidTemporalSolveParams,
    ) -> Result<usize, GpuRigidSphereWorldError> {
        self.world
            .step_temporal_speed_bounded(frame_dt, substeps, settings)
    }

    /// Advance every environment with speculative ground and ordinary pair contacts.
    pub fn step_temporal_speculative_ground(
        &mut self,
        frame_dt: f32,
        substeps: u32,
        settings: GpuRigidTemporalSolveParams,
        margin: f32,
    ) -> Result<usize, GpuRigidSphereWorldError> {
        self.world
            .step_temporal_speculative_ground(frame_dt, substeps, settings, margin)
    }

    /// Advance every environment with independently configured joint softness.
    pub fn step_temporal_with_joints(
        &mut self,
        frame_dt: f32,
        substeps: u32,
        settings: GpuRigidTemporalSolveParams,
        joint_settings: GpuRigidTemporalJointParams,
        margin: f32,
    ) -> Result<usize, GpuRigidSphereWorldError> {
        self.world
            .step_temporal_with_joints(frame_dt, substeps, settings, joint_settings, margin)
    }

    /// Replace one environment's bodies and invalidate only its cached impulses.
    pub fn reset_environment(
        &mut self,
        environment: usize,
        states: &[GpuRigidBodyState],
    ) -> Result<(), GpuRigidSphereWorldError> {
        if self.world.is_faulted() {
            return Err(GpuRigidSphereWorldError::Faulted);
        }
        let range = self
            .ranges
            .get(environment)
            .ok_or(GpuRigidSphereWorldError::InvalidInput)?;
        if states.len() != range.len() || states.iter().any(|state| !state.is_valid()) {
            return Err(GpuRigidSphereWorldError::InvalidInput);
        }
        for (index, state) in range.clone().zip(states.iter().copied()) {
            self.world.write_body(index, state)?;
        }
        Ok(())
    }

    /// Replace all packed bodies after a failed step or full batch reset.
    pub fn reset_all(
        &mut self,
        states: &[GpuRigidBodyState],
    ) -> Result<(), GpuRigidSphereWorldError> {
        self.world.reset(states)
    }

    /// Change prescribed motion for an environment-local body without readback.
    pub fn set_kinematic_motion(
        &mut self,
        environment: usize,
        body: usize,
        velocities: Option<([f32; 3], [f32; 3])>,
    ) -> Result<(), GpuRigidSphereWorldError> {
        let range = self
            .ranges
            .get(environment)
            .ok_or(GpuRigidSphereWorldError::InvalidInput)?;
        if body >= range.len() {
            return Err(GpuRigidSphereWorldError::InvalidInput);
        }
        self.world
            .set_kinematic_motion(range.start + body, velocities)
    }

    /// Set force and torque for a body identified within one environment.
    pub fn write_forces(
        &self,
        environment: usize,
        body: usize,
        forces: GpuRigidBodyForces,
    ) -> Result<(), GpuRigidSphereWorldError> {
        let range = self
            .ranges
            .get(environment)
            .ok_or(GpuRigidSphereWorldError::InvalidInput)?;
        if body >= range.len() {
            return Err(GpuRigidSphereWorldError::InvalidInput);
        }
        self.world.write_forces(range.start + body, forces)
    }

    /// Change one environment-local body's collision masks.
    pub fn set_body_collision_groups_environment(
        &mut self,
        environment: usize,
        body: usize,
        groups: GpuRigidCollisionGroups,
    ) -> Result<(), GpuRigidSphereWorldError> {
        let range = self
            .ranges
            .get(environment)
            .ok_or(GpuRigidSphereWorldError::InvalidInput)?;
        if body >= range.len() {
            return Err(GpuRigidSphereWorldError::InvalidInput);
        }
        self.world
            .set_body_collision_groups(range.start + body, groups)
    }

    /// Read one environment-local body's collision masks.
    pub fn body_collision_groups_environment(
        &self,
        environment: usize,
        body: usize,
    ) -> Option<GpuRigidCollisionGroups> {
        let range = self.ranges.get(environment)?;
        (body < range.len())
            .then(|| self.world.body_collision_groups(range.start + body))
            .flatten()
    }

    /// Read one environment without transferring other environments' bodies.
    pub fn readback_environment(
        &self,
        environment: usize,
    ) -> Result<Vec<GpuRigidBodyState>, GpuRigidSphereWorldError> {
        let range = self
            .ranges
            .get(environment)
            .ok_or(GpuRigidSphereWorldError::InvalidInput)?;
        self.world.readback_range(range.clone())
    }

    /// Read solver impulse history for one environment using local body IDs.
    ///
    /// This currently transfers the world's active impulse slots before filtering.
    pub fn readback_contact_impulses_environment(
        &self,
        environment: usize,
    ) -> Result<GpuRigidContactImpulseReadback, GpuRigidSphereWorldError> {
        let range = self
            .ranges
            .get(environment)
            .ok_or(GpuRigidSphereWorldError::InvalidInput)?;
        let start =
            u32::try_from(range.start).map_err(|_| GpuRigidSphereWorldError::InvalidInput)?;
        let mut result = self.world.readback_contact_impulses()?;
        result.contacts.retain(|contact| {
            range.contains(&(contact.body_b as usize))
                && contact
                    .body_a
                    .is_none_or(|body| range.contains(&(body as usize)))
        });
        for contact in &mut result.contacts {
            contact.body_b -= start;
            contact.body_a = contact.body_a.map(|body| body - start);
        }
        Ok(result)
    }

    /// Read diagnostic contacts for one environment with local body indices.
    ///
    /// The exhaustive path transfers only the selected environment's slots.
    /// The resident candidate path transfers all pairs before selecting this
    /// environment.
    pub fn readback_contacts_environment(
        &self,
        environment: usize,
    ) -> Result<GpuRigidSphereContactReadback, GpuRigidSphereWorldError> {
        let range = self
            .ranges
            .get(environment)
            .ok_or(GpuRigidSphereWorldError::InvalidInput)?;
        let start =
            u32::try_from(range.start).map_err(|_| GpuRigidSphereWorldError::InvalidInput)?;
        if self.world.last_resident_contacts.is_some() {
            let all = self.world.readback_contacts()?;
            let mut selected = GpuRigidSphereContactReadback {
                pairs: Vec::new(),
                pair_extra: Vec::new(),
                ground: all.ground.get(range.clone()).unwrap_or(&[]).to_vec(),
                ground_extra: all.ground_extra.get(range.clone()).unwrap_or(&[]).to_vec(),
            };
            for (index, (mut pair, contact)) in all.pairs.into_iter().enumerate() {
                if range.contains(&(pair.a as usize)) && range.contains(&(pair.b as usize)) {
                    pair.a -= start;
                    pair.b -= start;
                    selected.pairs.push((pair, contact));
                    if let Some(extra) = all.pair_extra.get(index) {
                        selected.pair_extra.push(*extra);
                    }
                }
            }
            return Ok(selected);
        }
        let mut readback = self.world.contacts.readback_body_range(
            &self.world.device,
            &self.world.queue,
            range.clone(),
        )?;
        for (pair, _) in &mut readback.pairs {
            pair.a -= start;
            pair.b -= start;
        }
        Ok(readback)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_contact_pipeline::GpuContactDevice;

    #[test]
    fn resident_heightfield_contacts_distinct_elevations() {
        let geometry = crate::mesh::HeightFieldGeometry::new(
            2,
            4,
            vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            nalgebra::Vector3::new(6.0, 4.0, 1.0),
        )
        .unwrap();
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            let mut states = [body(0.0, 0.0); 4];
            for state in &mut states {
                state.position_inverse_mass = [0.0; 4];
                state.inverse_inertia_sleep = [0.0; 4];
            }
            states[1].position_inverse_mass = [-2.0, 0.25, 0.25, 0.0];
            states[2].position_inverse_mass = [2.0, 0.25, 1.25, 0.0];
            states[3].position_inverse_mass = [0.0, 0.25, 0.75, 0.0];
            let shapes = [
                GpuRigidShape::from_heightfield(&geometry).unwrap(),
                GpuRigidShape::Sphere { radius: 0.5 },
                GpuRigidShape::Sphere { radius: 0.5 },
                GpuRigidShape::Sphere { radius: 0.5 },
            ];
            let mut world = GpuRigidPrimitiveWorld::new_primitives(
                context.device(),
                context.queue(),
                &states,
                &shapes,
                config(),
            )
            .unwrap();
            let _count = world.step(0.01).unwrap();
            let contacts = world.readback_contacts().unwrap();
            for sphere in 1..=3 {
                let contact = contacts
                    .pairs
                    .iter()
                    .find(|(pair, _)| pair.a == 0 && pair.b == sphere)
                    .map(|(_, contact)| contact)
                    .unwrap();
                assert!(contact.is_contact(), "backend={backend:?}, sphere={sphere}");
                let normal_z = if sphere == 3 {
                    2.0_f32 / 5.0_f32.sqrt()
                } else {
                    1.0
                };
                assert!((contact.depth_hit[0] - (0.5 - 0.25 * normal_z)).abs() < 1e-3);
                assert!((contact.normal[2] - normal_z).abs() < 1e-3);
            }
            for state in &mut states[1..] {
                state.position_inverse_mass[2] += 2.0;
            }
            world.reset(&states).unwrap();
            let _count = world.step(0.01).unwrap();
            assert!(
                world
                    .readback_contacts()
                    .unwrap()
                    .pairs
                    .iter()
                    .all(|(_, contact)| !contact.is_contact())
            );
        }
    }

    #[test]
    fn resident_polyline_contacts_sphere_on_gpu() {
        let mut vertices = vec![[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let mut segments = vec![[0, 1]];
        for index in 0..32 {
            let base = vertices.len() as u32;
            vertices.extend([
                [10.0 + index as f32, 0.0, 0.0],
                [10.5 + index as f32, 0.0, 0.0],
            ]);
            segments.push([base, base + 1]);
        }
        let mesh = GpuRigidShape::Polyline { vertices, segments };
        let mut tested_backends = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested_backends += 1;
            eprintln!("polyline contact backend: {backend:?}");
            for sphere_first in [false, true] {
                let mut world = None;
                for (sphere_position, expected_depth) in [
                    ([0.0, 0.0, 0.25], Some(0.25)),
                    ([1.2, 0.0, 0.0], Some(0.3)),
                    ([0.0, 0.0, 1.0], None),
                    ([0.0, 0.0, 0.0], Some(0.5)),
                ] {
                    let mut mesh_state = body(0.0, 0.0);
                    mesh_state.position_inverse_mass = [2.0, 3.0, 4.0, 0.0];
                    let half_angle = core::f32::consts::FRAC_PI_4;
                    mesh_state.orientation = [0.0, 0.0, half_angle.sin(), half_angle.cos()];
                    mesh_state.inverse_inertia_sleep = [0.0; 4];
                    let mut sphere_state = body(0.0, 0.0);
                    sphere_state.position_inverse_mass = [
                        2.0 - sphere_position[1],
                        3.0 + sphere_position[0],
                        4.0 + sphere_position[2],
                        0.0,
                    ];
                    sphere_state.inverse_inertia_sleep = [0.0; 4];
                    let (states, shapes) = if sphere_first {
                        (
                            [sphere_state, mesh_state],
                            [GpuRigidShape::Sphere { radius: 0.5 }, mesh.clone()],
                        )
                    } else {
                        (
                            [mesh_state, sphere_state],
                            [mesh.clone(), GpuRigidShape::Sphere { radius: 0.5 }],
                        )
                    };
                    let world = world.get_or_insert_with(|| {
                        GpuRigidPrimitiveWorld::new_primitives(
                            context.device(),
                            context.queue(),
                            &states,
                            &shapes,
                            config(),
                        )
                        .unwrap()
                    });
                    world.reset(&states).unwrap();
                    let pair_count = world.step(0.01).unwrap();
                    let contact = world.readback_contacts().unwrap().pairs[0].1;
                    assert_eq!(
                        pair_count, 1,
                        "backend: {backend:?}, position: {sphere_position:?}"
                    );
                    assert_eq!(
                        contact.is_contact(),
                        expected_depth.is_some(),
                        "backend: {backend:?}, position: {sphere_position:?}, sphere_first: {sphere_first}, contact: {contact:?}"
                    );
                    if let Some(depth) = expected_depth {
                        assert!((contact.depth_hit[0] - depth).abs() < 1e-3, "{contact:?}");
                        if sphere_position[2] > 0.0 {
                            assert_eq!(contact.normal[2] > 0.0, !sphere_first);
                        }
                    }
                }
            }
        }
        assert!(tested_backends > 0, "no GPU backend available");
    }

    #[test]
    fn resident_polyline_contacts_capsule_on_gpu() {
        let mut vertices = vec![[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let mut segments = vec![[0, 1]];
        for index in 0..32 {
            let base = vertices.len() as u32;
            vertices.extend([
                [10.0 + index as f32, 0.0, 0.0],
                [10.5 + index as f32, 0.0, 0.0],
            ]);
            segments.push([base, base + 1]);
        }
        let mesh = GpuRigidShape::Polyline { vertices, segments };
        let mut tested_backends = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested_backends += 1;
            eprintln!("polyline capsule backend: {backend:?}");
            for sphere_first in [false, true] {
                let mut world = None;
                for (sphere_position, expected_depth, parallel) in [
                    ([0.0, 0.0, 0.55], Some(0.25), false),
                    ([1.2, 0.0, 0.0], Some(0.3), false),
                    ([0.0, 0.0, 1.0], None, false),
                    ([0.0, 0.0, 0.0], Some(0.5), false),
                    ([1.4, 0.0, 0.0], Some(0.4), true),
                    ([0.0, 0.0, 0.25], Some(0.25), true),
                ] {
                    let mut mesh_state = body(0.0, 0.0);
                    mesh_state.position_inverse_mass = [2.0, 3.0, 4.0, 0.0];
                    let half_angle = core::f32::consts::FRAC_PI_4;
                    mesh_state.orientation = [0.0, 0.0, half_angle.sin(), half_angle.cos()];
                    mesh_state.inverse_inertia_sleep = [0.0; 4];
                    let mut sphere_state = body(0.0, 0.0);
                    sphere_state.position_inverse_mass = [
                        2.0 - sphere_position[1],
                        3.0 + sphere_position[0],
                        4.0 + sphere_position[2],
                        0.0,
                    ];
                    sphere_state.inverse_inertia_sleep = [0.0; 4];
                    if parallel {
                        sphere_state.orientation = [half_angle.sin(), 0.0, 0.0, half_angle.cos()];
                    }
                    let (states, shapes) = if sphere_first {
                        (
                            [sphere_state, mesh_state],
                            [
                                GpuRigidShape::Capsule {
                                    radius: 0.5,
                                    half_length: 0.3,
                                },
                                mesh.clone(),
                            ],
                        )
                    } else {
                        (
                            [mesh_state, sphere_state],
                            [
                                mesh.clone(),
                                GpuRigidShape::Capsule {
                                    radius: 0.5,
                                    half_length: 0.3,
                                },
                            ],
                        )
                    };
                    let world = world.get_or_insert_with(|| {
                        GpuRigidPrimitiveWorld::new_primitives(
                            context.device(),
                            context.queue(),
                            &states,
                            &shapes,
                            config(),
                        )
                        .unwrap()
                    });
                    world.reset(&states).unwrap();
                    let pair_count = world.step(0.01).unwrap();
                    let contact = world.readback_contacts().unwrap().pairs[0].1;
                    assert_eq!(
                        pair_count, 1,
                        "backend: {backend:?}, position: {sphere_position:?}"
                    );
                    assert_eq!(
                        contact.is_contact(),
                        expected_depth.is_some(),
                        "backend: {backend:?}, position: {sphere_position:?}, sphere_first: {sphere_first}, contact: {contact:?}"
                    );
                    if let Some(depth) = expected_depth {
                        assert!((contact.depth_hit[0] - depth).abs() < 1e-3, "{contact:?}");
                        if sphere_position[2] > 0.0 {
                            assert_eq!(contact.normal[2] > 0.0, !sphere_first);
                        }
                    }
                }
            }
        }
        assert!(tested_backends > 0, "no GPU backend available");
    }

    #[test]
    fn resident_triangle_mesh_contacts_sphere_on_gpu() {
        let mesh = GpuRigidShape::TriangleMesh {
            vertices: vec![[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]],
            triangles: vec![[0, 1, 2]],
        };
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            for sphere_first in [false, true] {
                let mut world = None;
                for (sphere_position, expected_depth) in [
                    ([0.0, 0.0, 0.25], Some(0.25)),
                    ([0.0, -1.2, 0.0], Some(0.3)),
                    ([0.0, 0.0, 1.0], None),
                ] {
                    let mut mesh_state = body(0.0, 0.0);
                    mesh_state.position_inverse_mass = [0.0, 0.0, 0.0, 0.0];
                    mesh_state.inverse_inertia_sleep = [0.0; 4];
                    let mut sphere_state = body(0.0, 0.0);
                    sphere_state.position_inverse_mass = [
                        sphere_position[0],
                        sphere_position[1],
                        sphere_position[2],
                        0.0,
                    ];
                    sphere_state.inverse_inertia_sleep = [0.0; 4];
                    let (states, shapes) = if sphere_first {
                        (
                            [sphere_state, mesh_state],
                            [GpuRigidShape::Sphere { radius: 0.5 }, mesh.clone()],
                        )
                    } else {
                        (
                            [mesh_state, sphere_state],
                            [mesh.clone(), GpuRigidShape::Sphere { radius: 0.5 }],
                        )
                    };
                    let world = world.get_or_insert_with(|| {
                        GpuRigidPrimitiveWorld::new_primitives(
                            context.device(),
                            context.queue(),
                            &states,
                            &shapes,
                            config(),
                        )
                        .unwrap()
                    });
                    world.reset(&states).unwrap();
                    let pair_count = world.step(0.01).unwrap();
                    let contact = world.readback_contacts().unwrap().pairs[0].1;
                    assert_eq!(
                        pair_count, 1,
                        "backend: {backend:?}, position: {sphere_position:?}"
                    );
                    assert_eq!(
                        contact.is_contact(),
                        expected_depth.is_some(),
                        "backend: {backend:?}, position: {sphere_position:?}, sphere_first: {sphere_first}, contact: {contact:?}"
                    );
                    if let Some(depth) = expected_depth {
                        assert!((contact.depth_hit[0] - depth).abs() < 1e-3, "{contact:?}");
                        if sphere_position[2] > 0.0 {
                            assert_eq!(contact.normal[2] > 0.0, !sphere_first);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn resident_triangle_mesh_contacts_capsule_on_gpu() {
        let mesh = GpuRigidShape::TriangleMesh {
            vertices: vec![[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]],
            triangles: vec![[0, 1, 2]],
        };
        let capsule = GpuRigidShape::Capsule {
            radius: 0.25,
            half_length: 0.3,
        };
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            for capsule_first in [false, true] {
                let mut mesh_state = body(0.0, 0.0);
                mesh_state.position_inverse_mass = [0.0; 4];
                mesh_state.inverse_inertia_sleep = [0.0; 4];
                let mut capsule_state = body(0.0, 0.0);
                capsule_state.position_inverse_mass = [0.0, 0.0, 0.4, 0.0];
                capsule_state.inverse_inertia_sleep = [0.0; 4];
                let (initial_states, shapes) = if capsule_first {
                    ([capsule_state, mesh_state], [capsule.clone(), mesh.clone()])
                } else {
                    ([mesh_state, capsule_state], [mesh.clone(), capsule.clone()])
                };
                let mut world = GpuRigidPrimitiveWorld::new_primitives(
                    context.device(),
                    context.queue(),
                    &initial_states,
                    &shapes,
                    config(),
                )
                .unwrap();
                for (position, expected_depth) in [
                    ([0.0, 0.0, 0.4], Some(0.15)),
                    ([0.0, 0.0, -0.4], Some(0.15)),
                    ([0.0, -1.1, 0.0], Some(0.15)),
                    ([1.1, -1.1, 0.0], Some(0.25 - 0.02_f32.sqrt())),
                    ([0.0, 0.0, 0.0], Some(0.25)),
                    ([0.0, 0.0, 2.0], None),
                ] {
                    capsule_state.position_inverse_mass[..3].copy_from_slice(&position);
                    let states = if capsule_first {
                        [capsule_state, mesh_state]
                    } else {
                        [mesh_state, capsule_state]
                    };
                    world.reset(&states).unwrap();
                    let _pairs = world.step(0.01).unwrap();
                    let contact = world.readback_contacts().unwrap().pairs[0].1;
                    assert_eq!(
                        contact.is_contact(),
                        expected_depth.is_some(),
                        "backend: {backend:?}, position: {position:?}, capsule_first: {capsule_first}, contact: {contact:?}"
                    );
                    if let Some(depth) = expected_depth {
                        assert!((contact.depth_hit[0] - depth).abs() < 1e-3, "{contact:?}");
                        if position[2].abs() > 0.0 && position[2].abs() < 1.0 {
                            assert_eq!(
                                contact.normal[2] > 0.0,
                                (position[2] > 0.0) != capsule_first
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn resident_polyline_convex_contacts_and_manifold() {
        let line = GpuRigidShape::Polyline {
            vertices: vec![[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            segments: vec![[0, 1], [1, 2]],
        };
        let shape = GpuRigidShape::Convex {
            vertices: [-0.5, 0.5]
                .into_iter()
                .flat_map(|x| {
                    [-0.5, 0.5]
                        .into_iter()
                        .flat_map(move |y| [-0.5, 0.5].into_iter().map(move |z| [x, y, z]))
                })
                .collect(),
        };
        let q = core::f32::consts::FRAC_1_SQRT_2;
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            eprintln!("polyline convex backend: {backend:?}");
            for box_first in [false, true] {
                let mut line_state = body(0.0, 0.0);
                line_state.position_inverse_mass = [2.0, 3.0, 4.0, 0.0];
                line_state.orientation = [0.0, 0.0, q, q];
                line_state.inverse_inertia_sleep = [0.0; 4];
                let mut box_state = line_state;
                let (states, shapes) = if box_first {
                    ([box_state, line_state], [shape.clone(), line.clone()])
                } else {
                    ([line_state, box_state], [line.clone(), shape.clone()])
                };
                let mut world = GpuRigidPrimitiveWorld::new_primitives(
                    context.device(),
                    context.queue(),
                    &states,
                    &shapes,
                    config(),
                )
                .unwrap();
                for (position, depth, oblique) in [
                    ([0.0, 0.0, 0.4], Some(0.1), false),
                    ([0.0, 0.0, -0.4], Some(0.1), false),
                    ([0.0, 0.0, 0.5], Some(0.0), false),
                    ([0.0, 0.0, 0.7], None, false),
                    ([2.4, 0.0, 0.0], Some(0.1), false),
                    ([2.6, 0.0, 0.0], None, false),
                    ([0.0, 0.6, 0.6], None, false),
                    ([0.0, 0.0, 0.4], Some(0.1), true),
                ] {
                    box_state.position_inverse_mass =
                        [2.0 - position[1], 3.0 + position[0], 4.0 + position[2], 0.0];
                    let yaw = if oblique {
                        core::f32::consts::PI * 0.375
                    } else {
                        core::f32::consts::FRAC_PI_4
                    };
                    box_state.orientation = [0.0, 0.0, yaw.sin(), yaw.cos()];
                    let states = if box_first {
                        [box_state, line_state]
                    } else {
                        [line_state, box_state]
                    };
                    world.reset(&states).unwrap();
                    let _pairs = world.step(0.01).unwrap();
                    let contacts = world.readback_contacts().unwrap();
                    let point = contacts.pairs[0].1;
                    assert_eq!(
                        point.is_contact(),
                        depth.is_some(),
                        "{backend:?}: {position:?}: {contacts:?}"
                    );
                    if let Some(depth) = depth {
                        assert!((point.depth_hit[0] - depth).abs() < 1e-3, "{contacts:?}");
                        assert!(point.normal.iter().all(|value| value.is_finite()));
                        if position[2].abs() > 0.0 {
                            let sign = if box_first { -1.0 } else { 1.0 };
                            assert!(point.normal[2] * position[2].signum() * sign > 0.99);
                            let extra = contacts.pair_extra[0][0];
                            assert!(extra.is_contact(), "{contacts:?}");
                            assert!(
                                (point.point[1] - extra.point[1]).abs() > 0.99,
                                "{contacts:?}"
                            );
                            assert!(!contacts.pair_extra[0][1].is_contact());
                        }
                    } else {
                        assert!(
                            contacts.pair_extra[0]
                                .iter()
                                .all(|point| !point.is_contact())
                        );
                    }
                }
            }
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn resident_polyline_box_contacts_and_manifold() {
        let line = GpuRigidShape::Polyline {
            vertices: vec![[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            segments: vec![[0, 1], [1, 2]],
        };
        let shape = GpuRigidShape::Box {
            half_extents: [0.5; 3],
        };
        let q = core::f32::consts::FRAC_1_SQRT_2;
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            eprintln!("polyline box backend: {backend:?}");
            for box_first in [false, true] {
                let mut line_state = body(0.0, 0.0);
                line_state.position_inverse_mass = [2.0, 3.0, 4.0, 0.0];
                line_state.orientation = [0.0, 0.0, q, q];
                line_state.inverse_inertia_sleep = [0.0; 4];
                let mut box_state = line_state;
                let (states, shapes) = if box_first {
                    ([box_state, line_state], [shape.clone(), line.clone()])
                } else {
                    ([line_state, box_state], [line.clone(), shape.clone()])
                };
                let mut world = GpuRigidPrimitiveWorld::new_primitives(
                    context.device(),
                    context.queue(),
                    &states,
                    &shapes,
                    config(),
                )
                .unwrap();
                for (position, depth, oblique) in [
                    ([0.0, 0.0, 0.4], Some(0.1), false),
                    ([0.0, 0.0, -0.4], Some(0.1), false),
                    ([0.0, 0.0, 0.5], Some(0.0), false),
                    ([0.0, 0.0, 0.7], None, false),
                    ([2.4, 0.0, 0.0], Some(0.1), false),
                    ([2.6, 0.0, 0.0], None, false),
                    ([0.0, 0.6, 0.6], None, false),
                    ([0.0, 0.0, 0.4], Some(0.1), true),
                ] {
                    box_state.position_inverse_mass =
                        [2.0 - position[1], 3.0 + position[0], 4.0 + position[2], 0.0];
                    let yaw = if oblique {
                        core::f32::consts::PI * 0.375
                    } else {
                        core::f32::consts::FRAC_PI_4
                    };
                    box_state.orientation = [0.0, 0.0, yaw.sin(), yaw.cos()];
                    let states = if box_first {
                        [box_state, line_state]
                    } else {
                        [line_state, box_state]
                    };
                    world.reset(&states).unwrap();
                    let _pairs = world.step(0.01).unwrap();
                    let contacts = world.readback_contacts().unwrap();
                    let point = contacts.pairs[0].1;
                    assert_eq!(
                        point.is_contact(),
                        depth.is_some(),
                        "{backend:?}: {position:?}: {contacts:?}"
                    );
                    if let Some(depth) = depth {
                        assert!((point.depth_hit[0] - depth).abs() < 1e-3, "{contacts:?}");
                        assert!(point.normal.iter().all(|value| value.is_finite()));
                        if position[2].abs() > 0.0 {
                            let sign = if box_first { -1.0 } else { 1.0 };
                            assert!(point.normal[2] * position[2].signum() * sign > 0.99);
                            let extra = contacts.pair_extra[0][0];
                            assert!(extra.is_contact(), "{contacts:?}");
                            assert!(
                                (point.point[1] - extra.point[1]).abs() > 0.99,
                                "{contacts:?}"
                            );
                            assert!(!contacts.pair_extra[0][1].is_contact());
                        }
                    } else {
                        assert!(
                            contacts.pair_extra[0]
                                .iter()
                                .all(|point| !point.is_contact())
                        );
                    }
                }
            }
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn resident_polyline_capsule_manifold_spans_segments() {
        let mesh = GpuRigidShape::Polyline {
            vertices: vec![[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            segments: vec![[0, 1], [1, 2]],
        };
        let capsule = GpuRigidShape::Capsule {
            radius: 0.25,
            half_length: 0.75,
        };
        let quarter_turn = core::f32::consts::FRAC_1_SQRT_2;
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            eprintln!("polyline manifold backend: {backend:?}");
            for capsule_first in [false, true] {
                let mut mesh_state = body(0.0, 0.0);
                mesh_state.position_inverse_mass = [0.0; 4];
                mesh_state.inverse_inertia_sleep = [0.0; 4];
                let mut capsule_state = body(0.0, 0.0);
                capsule_state.position_inverse_mass = [0.0, 0.0, 0.15, 0.0];
                capsule_state.orientation = [0.0, quarter_turn, 0.0, quarter_turn];
                capsule_state.inverse_inertia_sleep = [0.0; 4];
                let (states, shapes) = if capsule_first {
                    ([capsule_state, mesh_state], [capsule.clone(), mesh.clone()])
                } else {
                    ([mesh_state, capsule_state], [mesh.clone(), capsule.clone()])
                };
                let mut world = GpuRigidPrimitiveWorld::new_primitives(
                    context.device(),
                    context.queue(),
                    &states,
                    &shapes,
                    config(),
                )
                .unwrap();
                let _pairs = world.step(0.01).unwrap();
                let contacts = world.readback_contacts().unwrap();
                let first = contacts.pairs[0].1;
                let second = contacts.pair_extra[0][0];
                assert!(
                    first.is_contact() && second.is_contact(),
                    "backend: {backend:?}, capsule_first: {capsule_first}, contacts: {contacts:?}"
                );
                assert!((first.depth_hit[0] - 0.1).abs() < 1e-3, "{first:?}");
                assert!((second.depth_hit[0] - 0.1).abs() < 1e-3, "{second:?}");
                assert!(
                    (first.point[0] - second.point[0]).abs() > 1.4,
                    "{contacts:?}"
                );
                let sign = if capsule_first { -1.0 } else { 1.0 };
                assert!(first.normal[2] * sign > 0.99);
                assert!(second.normal[2] * sign > 0.99);
                let manifold: Vec<_> = core::iter::once(first)
                    .chain(contacts.pair_extra[0])
                    .filter(|point| point.is_contact())
                    .collect();
                for (index, point) in manifold.iter().enumerate() {
                    assert!((point.depth_hit[0] - 0.1).abs() < 1e-3);
                    assert!(point.normal[2] * sign > 0.99);
                    assert!(
                        manifold[..index]
                            .iter()
                            .all(|other| (other.point[0] - point.point[0]).abs() > 0.01)
                    );
                }
                capsule_state.position_inverse_mass[0] = 1.8;
                let overhang = if capsule_first {
                    [capsule_state, mesh_state]
                } else {
                    [mesh_state, capsule_state]
                };
                world.reset(&overhang).unwrap();
                let _pairs = world.step(0.01).unwrap();
                let contacts = world.readback_contacts().unwrap();
                let points: Vec<_> = core::iter::once(contacts.pairs[0].1)
                    .chain(contacts.pair_extra[0])
                    .filter(|point| point.is_contact())
                    .collect();
                let lower = points
                    .iter()
                    .map(|point| point.point[0])
                    .fold(f32::INFINITY, f32::min);
                let upper = points
                    .iter()
                    .map(|point| point.point[0])
                    .fold(f32::NEG_INFINITY, f32::max);
                assert!(upper - lower > 0.9, "{contacts:?}");
                assert!(
                    points
                        .iter()
                        .all(|point| (point.depth_hit[0] - 0.1).abs() < 1e-3)
                );

                capsule_state.position_inverse_mass[3] = 1.0;
                capsule_state.linear_velocity[2] = -1.0;
                let states = if capsule_first {
                    [capsule_state, mesh_state]
                } else {
                    [mesh_state, capsule_state]
                };
                world.reset(&states).unwrap();
                let _pairs = world.step(0.01).unwrap();
                let settled = world.readback().unwrap()[usize::from(!capsule_first)];
                assert!(settled.linear_velocity[2] > -0.1, "{settled:?}");
            }
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn resident_triangle_mesh_capsule_spans_adjacent_triangles() {
        let mesh = GpuRigidShape::TriangleMesh {
            vertices: vec![
                [-2.0, -2.0, 0.0],
                [2.0, -2.0, 0.0],
                [2.0, 2.0, 0.0],
                [-2.0, 2.0, 0.0],
            ],
            triangles: vec![[0, 1, 2], [0, 2, 3]],
        };
        let capsule = GpuRigidShape::Capsule {
            radius: 0.25,
            half_length: 0.75,
        };
        let quarter_turn = core::f32::consts::FRAC_1_SQRT_2;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            for capsule_first in [false, true] {
                let mut mesh_state = body(0.0, 0.0);
                mesh_state.position_inverse_mass = [0.0; 4];
                mesh_state.inverse_inertia_sleep = [0.0; 4];
                let mut capsule_state = body(0.0, 0.0);
                capsule_state.position_inverse_mass = [0.0, 0.0, 0.15, 0.0];
                capsule_state.orientation = [0.0, quarter_turn, 0.0, quarter_turn];
                capsule_state.inverse_inertia_sleep = [0.0; 4];
                let (states, shapes) = if capsule_first {
                    ([capsule_state, mesh_state], [capsule.clone(), mesh.clone()])
                } else {
                    ([mesh_state, capsule_state], [mesh.clone(), capsule.clone()])
                };
                let mut world = GpuRigidPrimitiveWorld::new_primitives(
                    context.device(),
                    context.queue(),
                    &states,
                    &shapes,
                    config(),
                )
                .unwrap();
                let _pairs = world.step(0.01).unwrap();
                let contacts = world.readback_contacts().unwrap();
                let first = contacts.pairs[0].1;
                let second = contacts.pair_extra[0][0];
                assert!(
                    first.is_contact() && second.is_contact(),
                    "backend: {backend:?}, capsule_first: {capsule_first}, contacts: {contacts:?}"
                );
                assert!((first.depth_hit[0] - 0.1).abs() < 1e-3, "{first:?}");
                assert!((second.depth_hit[0] - 0.1).abs() < 1e-3, "{second:?}");
                assert!(
                    (first.point[0] - second.point[0]).abs() > 1.4,
                    "{contacts:?}"
                );
                let sign = if capsule_first { -1.0 } else { 1.0 };
                assert!(first.normal[2] * sign > 0.99);
                assert!(second.normal[2] * sign > 0.99);
                assert!(!contacts.pair_extra[0][1].is_contact());

                capsule_state.position_inverse_mass[3] = 1.0;
                capsule_state.linear_velocity[2] = -1.0;
                let states = if capsule_first {
                    [capsule_state, mesh_state]
                } else {
                    [mesh_state, capsule_state]
                };
                world.reset(&states).unwrap();
                let _pairs = world.step(0.01).unwrap();
                let settled = world.readback().unwrap()[usize::from(!capsule_first)];
                assert!(settled.linear_velocity[2] > -0.1, "{settled:?}");
            }
        }
    }

    #[test]
    fn resident_triangle_mesh_contacts_cylinder_and_cone_on_gpu() {
        let mesh = GpuRigidShape::TriangleMesh {
            vertices: vec![
                [-2.0, -2.0, 0.0],
                [2.0, -2.0, 0.0],
                [2.0, 2.0, 0.0],
                [-2.0, 2.0, 0.0],
            ],
            triangles: vec![[0, 1, 2], [0, 2, 3]],
        };
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            for shape in [
                GpuRigidShape::Cylinder {
                    radius: 0.3,
                    half_length: 0.25,
                },
                GpuRigidShape::Cone {
                    radius: 0.3,
                    half_length: 0.25,
                },
            ] {
                for analytic_first in [false, true] {
                    let mut mesh_state = body(0.0, 0.0);
                    mesh_state.position_inverse_mass = [0.0; 4];
                    mesh_state.inverse_inertia_sleep = [0.0; 4];
                    let mut analytic_state = body(0.0, 0.0);
                    analytic_state.position_inverse_mass = [0.0, 0.0, 0.15, 0.0];
                    analytic_state.inverse_inertia_sleep = [0.0; 4];
                    let (states, shapes) = if analytic_first {
                        ([analytic_state, mesh_state], [shape.clone(), mesh.clone()])
                    } else {
                        ([mesh_state, analytic_state], [mesh.clone(), shape.clone()])
                    };
                    let mut world = GpuRigidPrimitiveWorld::new_primitives(
                        context.device(),
                        context.queue(),
                        &states,
                        &shapes,
                        config(),
                    )
                    .unwrap();
                    for (position, expected) in [
                        ([0.0, 0.0, 0.15], true),
                        ([0.0, 0.0, -0.15], true),
                        ([0.0, 0.0, 1.0], false),
                        ([3.0, 0.0, 0.15], false),
                    ] {
                        analytic_state.position_inverse_mass[..3].copy_from_slice(&position);
                        let states = if analytic_first {
                            [analytic_state, mesh_state]
                        } else {
                            [mesh_state, analytic_state]
                        };
                        world.reset(&states).unwrap();
                        let _pairs = world.step(0.01).unwrap();
                        let contacts = world.readback_contacts().unwrap();
                        let contact = contacts.pairs[0].1;
                        assert_eq!(
                            contact.is_contact(),
                            expected,
                            "backend: {backend:?}, shape: {shape:?}, analytic_first: {analytic_first}, position: {position:?}, contact: {contact:?}"
                        );
                        if expected {
                            let sign = if (position[2] > 0.0) == analytic_first {
                                -1.0
                            } else {
                                1.0
                            };
                            assert!(contact.normal[2] * sign > 0.99, "{contact:?}");
                            assert!((contact.depth_hit[0] - 0.1).abs() < 1e-3, "{contact:?}");
                            let cap_faces_mesh = position[2] > 0.0
                                || matches!(shape, GpuRigidShape::Cylinder { .. });
                            if cap_faces_mesh {
                                assert!(
                                    contacts.pair_extra[0]
                                        .iter()
                                        .all(|point| point.is_contact()),
                                    "{contacts:?}"
                                );
                                let points = [
                                    contact.point,
                                    contacts.pair_extra[0][0].point,
                                    contacts.pair_extra[0][1].point,
                                    contacts.pair_extra[0][2].point,
                                ];
                                for axis in [0, 1] {
                                    let low = points
                                        .iter()
                                        .map(|point| point[axis])
                                        .fold(f32::MAX, f32::min);
                                    let high = points
                                        .iter()
                                        .map(|point| point[axis])
                                        .fold(f32::MIN, f32::max);
                                    assert!(high - low > 0.5, "{contacts:?}");
                                }
                            } else {
                                assert!(
                                    contacts.pair_extra[0]
                                        .iter()
                                        .all(|point| !point.is_contact()),
                                    "{contacts:?}"
                                );
                            }
                        }
                    }
                    for (position, expected) in
                        [([2.1, 0.0, 0.15], true), ([2.5, 0.0, 0.15], false)]
                    {
                        analytic_state.position_inverse_mass[..3].copy_from_slice(&position);
                        let states = if analytic_first {
                            [analytic_state, mesh_state]
                        } else {
                            [mesh_state, analytic_state]
                        };
                        world.reset(&states).unwrap();
                        let _pairs = world.step(0.01).unwrap();
                        let contact = world.readback_contacts().unwrap().pairs[0].1;
                        assert_eq!(
                            contact.is_contact(),
                            expected,
                            "backend: {backend:?}, shape: {shape:?}, analytic_first: {analytic_first}, edge position: {position:?}, contact: {contact:?}"
                        );
                    }

                    let quarter_turn = core::f32::consts::FRAC_1_SQRT_2;
                    let rotated = [0.0, quarter_turn, 0.0, quarter_turn];
                    mesh_state.orientation = rotated;
                    analytic_state.orientation = rotated;
                    analytic_state.position_inverse_mass = [0.15, 0.0, 0.0, 0.0];
                    let states = if analytic_first {
                        [analytic_state, mesh_state]
                    } else {
                        [mesh_state, analytic_state]
                    };
                    world.reset(&states).unwrap();
                    let _pairs = world.step(0.01).unwrap();
                    let contacts = world.readback_contacts().unwrap();
                    let contact = contacts.pairs[0].1;
                    let sign = if analytic_first { -1.0 } else { 1.0 };
                    assert!(
                        contact.is_contact()
                            && contact.normal[0] * sign > 0.99
                            && (contact.depth_hit[0] - 0.1).abs() < 1e-3,
                        "backend: {backend:?}, shape: {shape:?}, analytic_first: {analytic_first}, rotated contact: {contact:?}"
                    );
                    assert!(
                        contacts.pair_extra[0]
                            .iter()
                            .all(|point| point.is_contact()),
                        "{contacts:?}"
                    );

                    mesh_state.orientation = [0.0, 0.0, 0.0, 1.0];
                    analytic_state.orientation = [0.0, 0.0, 0.0, 1.0];
                    analytic_state.position_inverse_mass = [0.0, 0.0, 0.15, 1.0];
                    analytic_state.linear_velocity[2] = -1.0;
                    let states = if analytic_first {
                        [analytic_state, mesh_state]
                    } else {
                        [mesh_state, analytic_state]
                    };
                    world.reset(&states).unwrap();
                    let _pairs = world.step(0.01).unwrap();
                    let settled = world.readback().unwrap()[usize::from(!analytic_first)];
                    assert!(
                        settled.linear_velocity[2] > -0.1,
                        "backend: {backend:?}, shape: {shape:?}, analytic_first: {analytic_first}, state: {settled:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn resident_mesh_cylinder_side_spans_adjacent_triangles() {
        let mesh = GpuRigidShape::TriangleMesh {
            vertices: vec![
                [-2.0, -2.0, 0.0],
                [2.0, -2.0, 0.0],
                [2.0, 2.0, 0.0],
                [-2.0, 2.0, 0.0],
            ],
            triangles: vec![[0, 1, 2], [0, 2, 3]],
        };
        let cylinder = GpuRigidShape::Cylinder {
            radius: 0.25,
            half_length: 0.75,
        };
        let quarter_turn = core::f32::consts::FRAC_1_SQRT_2;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            for cylinder_first in [false, true] {
                let mut mesh_state = body(0.0, 0.0);
                mesh_state.position_inverse_mass = [0.0; 4];
                mesh_state.inverse_inertia_sleep = [0.0; 4];
                let mut cylinder_state = body(0.0, 0.0);
                cylinder_state.position_inverse_mass = [0.0, 0.0, 0.15, 0.0];
                cylinder_state.orientation = [0.0, quarter_turn, 0.0, quarter_turn];
                cylinder_state.inverse_inertia_sleep = [0.0; 4];
                let (states, shapes) = if cylinder_first {
                    (
                        [cylinder_state, mesh_state],
                        [cylinder.clone(), mesh.clone()],
                    )
                } else {
                    (
                        [mesh_state, cylinder_state],
                        [mesh.clone(), cylinder.clone()],
                    )
                };
                let mut world = GpuRigidPrimitiveWorld::new_primitives(
                    context.device(),
                    context.queue(),
                    &states,
                    &shapes,
                    config(),
                )
                .unwrap();
                let _pairs = world.step(0.01).unwrap();
                let contacts = world.readback_contacts().unwrap();
                let first = contacts.pairs[0].1;
                let second = contacts.pair_extra[0][0];
                assert!(
                    first.is_contact() && second.is_contact(),
                    "backend: {backend:?}, cylinder_first: {cylinder_first}, contacts: {contacts:?}"
                );
                assert!((first.depth_hit[0] - 0.1).abs() < 1e-3, "{first:?}");
                assert!((second.depth_hit[0] - 0.1).abs() < 1e-3, "{second:?}");
                assert!((first.point[0] - second.point[0]).abs() > 1.4);
                let sign = if cylinder_first { -1.0 } else { 1.0 };
                assert!(first.normal[2] * sign > 0.99);
                assert!(second.normal[2] * sign > 0.99);
                assert!(
                    contacts.pair_extra[0][1..]
                        .iter()
                        .all(|point| !point.is_contact())
                );

                cylinder_state.position_inverse_mass[0] = 1.5;
                let shifted = if cylinder_first {
                    [cylinder_state, mesh_state]
                } else {
                    [mesh_state, cylinder_state]
                };
                world.reset(&shifted).unwrap();
                let _pairs = world.step(0.01).unwrap();
                let partial = world.readback_contacts().unwrap();
                assert!(partial.pairs[0].1.is_contact(), "{partial:?}");
                assert!(partial.pair_extra[0][0].is_contact(), "{partial:?}");
                assert!(
                    (partial.pairs[0].1.point[0] - partial.pair_extra[0][0].point[0]).abs() > 1.2,
                    "{partial:?}"
                );

                cylinder_state.position_inverse_mass[0] = 2.7;
                let narrow = if cylinder_first {
                    [cylinder_state, mesh_state]
                } else {
                    [mesh_state, cylinder_state]
                };
                world.reset(&narrow).unwrap();
                let _pairs = world.step(0.01).unwrap();
                let narrow_contacts = world.readback_contacts().unwrap();
                assert!(
                    narrow_contacts.pairs[0].1.is_contact(),
                    "{narrow_contacts:?}"
                );
                assert!(
                    narrow_contacts.pair_extra[0]
                        .iter()
                        .all(|point| !point.is_contact()),
                    "{narrow_contacts:?}"
                );

                cylinder_state.position_inverse_mass[0] = 0.0;
                let strip = GpuRigidShape::TriangleMesh {
                    vertices: vec![
                        [-0.5, -2.0, 0.0],
                        [0.5, -2.0, 0.0],
                        [0.5, 2.0, 0.0],
                        [-0.5, 2.0, 0.0],
                    ],
                    triangles: vec![[0, 1, 2], [0, 2, 3]],
                };
                let (strip_states, strip_shapes) = if cylinder_first {
                    ([cylinder_state, mesh_state], [cylinder.clone(), strip])
                } else {
                    ([mesh_state, cylinder_state], [strip, cylinder.clone()])
                };
                let mut strip_world = GpuRigidPrimitiveWorld::new_primitives(
                    context.device(),
                    context.queue(),
                    &strip_states,
                    &strip_shapes,
                    config(),
                )
                .unwrap();
                let _pairs = strip_world.step(0.01).unwrap();
                let strip_contacts = strip_world.readback_contacts().unwrap();
                assert!(strip_contacts.pairs[0].1.is_contact(), "{strip_contacts:?}");
                assert!(
                    strip_contacts.pair_extra[0][0].is_contact(),
                    "{strip_contacts:?}"
                );
                let strip_span = (strip_contacts.pairs[0].1.point[0]
                    - strip_contacts.pair_extra[0][0].point[0])
                    .abs();
                assert!((strip_span - 1.0).abs() < 1e-3, "{strip_contacts:?}");

                cylinder_state.position_inverse_mass = [0.0, 0.0, 0.15, 1.0];
                cylinder_state.linear_velocity[2] = -1.0;
                let dynamic = if cylinder_first {
                    [cylinder_state, mesh_state]
                } else {
                    [mesh_state, cylinder_state]
                };
                world.reset(&dynamic).unwrap();
                let _pairs = world.step(0.01).unwrap();
                let settled = world.readback().unwrap()[usize::from(!cylinder_first)];
                assert!(settled.linear_velocity[2] > -0.1, "{settled:?}");
            }
        }
    }

    #[test]
    fn resident_mesh_analytic_bodies_settle_under_gravity() {
        let mesh = GpuRigidShape::TriangleMesh {
            vertices: vec![
                [-2.0, -2.0, 0.0],
                [2.0, -2.0, 0.0],
                [2.0, 2.0, 0.0],
                [-2.0, 2.0, 0.0],
            ],
            triangles: vec![[0, 1, 2], [0, 2, 3]],
        };
        let quarter_turn = core::f32::consts::FRAC_1_SQRT_2;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            for (shape, orientation) in [
                (
                    GpuRigidShape::Cylinder {
                        radius: 0.25,
                        half_length: 0.25,
                    },
                    [0.0, 0.0, 0.0, 1.0],
                ),
                (
                    GpuRigidShape::Cone {
                        radius: 0.25,
                        half_length: 0.25,
                    },
                    [0.0, 0.0, 0.0, 1.0],
                ),
                (
                    GpuRigidShape::Cylinder {
                        radius: 0.25,
                        half_length: 0.75,
                    },
                    [0.0, quarter_turn, 0.0, quarter_turn],
                ),
            ] {
                let mut terrain = body(0.0, 0.0);
                terrain.position_inverse_mass = [0.0; 4];
                terrain.inverse_inertia_sleep = [0.0; 4];
                let mut dynamic = body(0.0, 0.0);
                dynamic.position_inverse_mass = [0.0, 0.0, 0.35, 1.0];
                dynamic.orientation = orientation;
                let mut settings = config();
                settings.gravity = [0.0, 0.0, -9.81];
                settings.solve.bias_factor = 0.2;
                settings.solve.iterations = 16;
                settings.sleep.enabled = false;
                let mut world = GpuRigidPrimitiveWorld::new_primitives(
                    context.device(),
                    context.queue(),
                    &[terrain, dynamic],
                    &[mesh.clone(), shape.clone()],
                    settings,
                )
                .unwrap();
                for _ in 0..240 {
                    let _pairs = world.step(1.0 / 120.0).unwrap();
                }
                let settled = world.readback().unwrap()[1];
                assert!(settled.is_valid(), "{backend:?}, {shape:?}, {settled:?}");
                assert!(
                    (settled.position_inverse_mass[2] - 0.25).abs() < 0.02,
                    "{backend:?}, {shape:?}, {settled:?}"
                );
                assert!(
                    settled.linear_velocity[..3].iter().all(|v| v.abs() < 0.1),
                    "{backend:?}, {shape:?}, {settled:?}"
                );
                assert!(
                    settled.angular_velocity[..3].iter().all(|v| v.abs() < 0.2),
                    "{backend:?}, {shape:?}, {settled:?}"
                );
            }
        }
    }

    #[test]
    fn resident_mesh_cylinder_material_friction_reduces_sliding() {
        let mesh = GpuRigidShape::TriangleMesh {
            vertices: vec![
                [-3.0, -3.0, 0.0],
                [3.0, -3.0, 0.0],
                [3.0, 3.0, 0.0],
                [-3.0, 3.0, 0.0],
            ],
            triangles: vec![[0, 1, 2], [0, 2, 3]],
        };
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            let mut terrain = body(0.0, 0.0);
            terrain.position_inverse_mass = [0.0; 4];
            terrain.inverse_inertia_sleep = [0.0; 4];
            let mut sliding = body(0.0, 0.5);
            sliding.position_inverse_mass = [0.0, 0.0, 0.25, 1.0];
            let quarter_turn = core::f32::consts::FRAC_1_SQRT_2;
            sliding.orientation = [0.0, quarter_turn, 0.0, quarter_turn];
            sliding.inverse_inertia_sleep = [0.0; 4];
            let mut settings = config();
            settings.gravity = [0.0, 0.0, -9.81];
            settings.solve.bias_factor = 0.2;
            settings.solve.iterations = 16;
            settings.sleep.enabled = false;
            let mut world = GpuRigidPrimitiveWorld::new_primitives(
                context.device(),
                context.queue(),
                &[terrain, sliding],
                &[
                    mesh.clone(),
                    GpuRigidShape::Cylinder {
                        radius: 0.25,
                        half_length: 0.75,
                    },
                ],
                settings,
            )
            .unwrap();
            let mut results = Vec::new();
            for friction in [0.0, 0.6] {
                world.reset(&[terrain, sliding]).unwrap();
                for index in 0..2 {
                    world
                        .set_body_material(index, ColliderMaterial::new(friction, 0.0))
                        .unwrap();
                }
                for _ in 0..120 {
                    let _pairs = world.step(1.0 / 120.0).unwrap();
                }
                results.push(world.readback().unwrap()[1]);
            }
            assert!(
                (results[0].linear_velocity[0] - 0.5).abs() < 0.03,
                "{backend:?}, {results:?}"
            );
            assert!(
                results[1].linear_velocity[0].abs() < 0.05,
                "{backend:?}, {results:?}"
            );
            assert!(
                results[1].position_inverse_mass[0] < results[0].position_inverse_mass[0] * 0.5,
                "{backend:?}, {results:?}"
            );
            for state in results {
                assert!(
                    state.is_valid() && (state.position_inverse_mass[2] - 0.25).abs() < 0.02,
                    "{backend:?}, {state:?}"
                );
            }
        }
    }

    #[test]
    fn resident_triangle_mesh_contacts_rotated_box_on_gpu() {
        let mesh = GpuRigidShape::TriangleMesh {
            vertices: vec![[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]],
            triangles: vec![[0, 1, 2]],
        };
        let box_shape = GpuRigidShape::Box {
            half_extents: [0.2, 0.2, 0.2],
        };
        let tilted = [
            core::f32::consts::FRAC_PI_8.sin(),
            0.0,
            0.0,
            core::f32::consts::FRAC_PI_8.cos(),
        ];
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            for box_first in [false, true] {
                let mut mesh_state = body(0.0, 0.0);
                mesh_state.position_inverse_mass = [0.0; 4];
                mesh_state.inverse_inertia_sleep = [0.0; 4];
                let mut box_state = body(0.0, 0.0);
                box_state.position_inverse_mass = [0.0, 0.0, 0.1, 0.0];
                box_state.inverse_inertia_sleep = [0.0; 4];
                let (initial_states, shapes) = if box_first {
                    ([box_state, mesh_state], [box_shape.clone(), mesh.clone()])
                } else {
                    ([mesh_state, box_state], [mesh.clone(), box_shape.clone()])
                };
                let mut world = GpuRigidPrimitiveWorld::new_primitives(
                    context.device(),
                    context.queue(),
                    &initial_states,
                    &shapes,
                    config(),
                )
                .unwrap();
                for (position, orientation, expected_depth, expected_axis) in [
                    ([0.0, 0.0, 0.1], [0.0, 0.0, 0.0, 1.0], Some(0.1), 2),
                    ([0.0, 0.0, -0.1], [0.0, 0.0, 0.0, 1.0], Some(0.1), 2),
                    ([0.0, -1.1, 0.0], [0.0, 0.0, 0.0, 1.0], Some(0.1), 1),
                    (
                        [0.0, 0.0, 0.25],
                        tilted,
                        Some(0.2 * 2.0_f32.sqrt() - 0.25),
                        2,
                    ),
                    ([0.0, -1.4, 0.0], [0.0, 0.0, 0.0, 1.0], None, 1),
                    ([0.0, 0.0, 0.5], [0.0, 0.0, 0.0, 1.0], None, 2),
                ] {
                    box_state.position_inverse_mass[..3].copy_from_slice(&position);
                    box_state.orientation = orientation;
                    let states = if box_first {
                        [box_state, mesh_state]
                    } else {
                        [mesh_state, box_state]
                    };
                    world.reset(&states).unwrap();
                    let _pairs = world.step(0.01).unwrap();
                    let contact = world.readback_contacts().unwrap().pairs[0].1;
                    assert_eq!(
                        contact.is_contact(),
                        expected_depth.is_some(),
                        "backend: {backend:?}, box_first: {box_first}, position: {position:?}, contact: {contact:?}"
                    );
                    if let Some(depth) = expected_depth {
                        assert!((contact.depth_hit[0] - depth).abs() < 2e-3, "{contact:?}");
                        let side = if expected_axis == 1 {
                            -1.0
                        } else {
                            position[2].signum()
                        };
                        let expected_sign = if box_first { -side } else { side };
                        assert!(
                            contact.normal[expected_axis] * expected_sign > 0.99,
                            "{contact:?}"
                        );
                        assert!(contact.point[2].abs() < 1e-4, "{contact:?}");
                    }
                }

                box_state.position_inverse_mass = [0.0, 0.0, 0.1, 1.0];
                box_state.orientation = [0.0, 0.0, 0.0, 1.0];
                box_state.linear_velocity[2] = -1.0;
                let states = if box_first {
                    [box_state, mesh_state]
                } else {
                    [mesh_state, box_state]
                };
                world.reset(&states).unwrap();
                let _pairs = world.step(0.01).unwrap();
                let settled = world.readback().unwrap()[usize::from(!box_first)];
                assert!(
                    settled.linear_velocity[2] > -0.1,
                    "backend: {backend:?}, box_first: {box_first}, state: {settled:?}"
                );
            }
        }
    }

    #[test]
    fn resident_triangle_mesh_box_manifold_spans_adjacent_triangles() {
        let mesh = GpuRigidShape::TriangleMesh {
            vertices: vec![
                [-2.0, -2.0, 0.0],
                [2.0, -2.0, 0.0],
                [2.0, 2.0, 0.0],
                [-2.0, 2.0, 0.0],
            ],
            triangles: vec![[0, 1, 2], [0, 2, 3]],
        };
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            for box_first in [false, true] {
                let mut mesh_state = body(0.0, 0.0);
                mesh_state.position_inverse_mass = [0.0; 4];
                mesh_state.inverse_inertia_sleep = [0.0; 4];
                let mut box_state = body(0.0, 0.0);
                box_state.position_inverse_mass = [0.0, 0.0, 0.15, 1.0];
                box_state.linear_velocity[2] = -1.0;
                let (states, shapes) = if box_first {
                    (
                        [box_state, mesh_state],
                        [
                            GpuRigidShape::Box {
                                half_extents: [0.2; 3],
                            },
                            mesh.clone(),
                        ],
                    )
                } else {
                    (
                        [mesh_state, box_state],
                        [
                            mesh.clone(),
                            GpuRigidShape::Box {
                                half_extents: [0.2; 3],
                            },
                        ],
                    )
                };
                let mut world = GpuRigidPrimitiveWorld::new_primitives(
                    context.device(),
                    context.queue(),
                    &states,
                    &shapes,
                    config(),
                )
                .unwrap();
                let _pairs = world.step(0.01).unwrap();
                let contacts = world.readback_contacts().unwrap();
                let manifold = [
                    contacts.pairs[0].1,
                    contacts.pair_extra[0][0],
                    contacts.pair_extra[0][1],
                    contacts.pair_extra[0][2],
                ];
                assert!(
                    manifold.iter().all(|contact| contact.is_contact()),
                    "backend: {backend:?}, box_first: {box_first}, manifold: {manifold:?}"
                );
                for (index, contact) in manifold.iter().enumerate() {
                    assert!((contact.depth_hit[0] - 0.06).abs() < 1e-3, "{contact:?}");
                    assert!(contact.normal[2] * if box_first { -1.0 } else { 1.0 } > 0.99);
                    assert!(contact.point[2].abs() < 1e-4, "{contact:?}");
                    assert!(manifold[..index].iter().all(|other| {
                        (other.point[0] - contact.point[0]).abs() > 0.1
                            || (other.point[1] - contact.point[1]).abs() > 0.1
                    }));
                }
                let x_span = manifold
                    .iter()
                    .map(|c| c.point[0])
                    .fold(f32::NEG_INFINITY, f32::max)
                    - manifold
                        .iter()
                        .map(|c| c.point[0])
                        .fold(f32::INFINITY, f32::min);
                let y_span = manifold
                    .iter()
                    .map(|c| c.point[1])
                    .fold(f32::NEG_INFINITY, f32::max)
                    - manifold
                        .iter()
                        .map(|c| c.point[1])
                        .fold(f32::INFINITY, f32::min);
                assert!(x_span > 0.39 && y_span > 0.39, "{manifold:?}");
                let settled = world.readback().unwrap()[usize::from(!box_first)];
                assert!(settled.linear_velocity[2] > -0.1, "{settled:?}");
                assert!(settled.angular_velocity[0].abs() < 0.2, "{settled:?}");
                assert!(settled.angular_velocity[1].abs() < 0.2, "{settled:?}");
            }
        }
    }

    #[test]
    fn resident_triangle_mesh_contacts_convex_hull_on_gpu() {
        let mesh = GpuRigidShape::TriangleMesh {
            vertices: vec![
                [-2.0, -2.0, 0.0],
                [2.0, -2.0, 0.0],
                [2.0, 2.0, 0.0],
                [-2.0, 2.0, 0.0],
            ],
            triangles: vec![[0, 1, 2], [0, 2, 3]],
        };
        let hull = GpuRigidShape::Convex {
            vertices: [-0.2, 0.2]
                .into_iter()
                .flat_map(|x| {
                    [-0.2, 0.2]
                        .into_iter()
                        .flat_map(move |y| [-0.2, 0.2].into_iter().map(move |z| [x, y, z]))
                })
                .collect(),
        };
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            for hull_first in [false, true] {
                let mut mesh_state = body(0.0, 0.0);
                mesh_state.position_inverse_mass = [0.0; 4];
                mesh_state.inverse_inertia_sleep = [0.0; 4];
                let mut hull_state = body(0.0, 0.0);
                hull_state.position_inverse_mass = [0.0, 0.0, 0.15, 0.0];
                hull_state.inverse_inertia_sleep = [0.0; 4];
                let (states, shapes) = if hull_first {
                    ([hull_state, mesh_state], [hull.clone(), mesh.clone()])
                } else {
                    ([mesh_state, hull_state], [mesh.clone(), hull.clone()])
                };
                let mut world = GpuRigidPrimitiveWorld::new_primitives(
                    context.device(),
                    context.queue(),
                    &states,
                    &shapes,
                    config(),
                )
                .unwrap();
                for (position, depth) in [
                    ([0.0, 0.0, 0.15], Some(0.05)),
                    ([0.0, 0.0, -0.15], Some(0.05)),
                    ([0.0, 0.0, 0.5], None),
                ] {
                    hull_state.position_inverse_mass[..3].copy_from_slice(&position);
                    let states = if hull_first {
                        [hull_state, mesh_state]
                    } else {
                        [mesh_state, hull_state]
                    };
                    world.reset(&states).unwrap();
                    let _pairs = world.step(0.01).unwrap();
                    let contacts = world.readback_contacts().unwrap();
                    let primary = contacts.pairs[0].1;
                    assert_eq!(
                        primary.is_contact(),
                        depth.is_some(),
                        "backend: {backend:?}, hull_first: {hull_first}, position: {position:?}, contact: {primary:?}"
                    );
                    if let Some(expected_depth) = depth {
                        assert!(
                            (primary.depth_hit[0] - expected_depth).abs() < 2e-3,
                            "{primary:?}"
                        );
                        let sign = if hull_first {
                            -position[2].signum()
                        } else {
                            position[2].signum()
                        };
                        assert!(primary.normal[2] * sign > 0.99, "{primary:?}");
                        assert!(
                            contacts.pair_extra[0].iter().all(|c| c.is_contact()),
                            "{contacts:?}"
                        );
                    }
                }
                for (position, orientation, expected_depth, axis) in [
                    ([0.0, -2.1, 0.0], [0.0, 0.0, 0.0, 1.0], Some(0.1), 1),
                    (
                        [0.0, 0.0, 0.25],
                        [
                            core::f32::consts::FRAC_PI_8.sin(),
                            0.0,
                            0.0,
                            core::f32::consts::FRAC_PI_8.cos(),
                        ],
                        Some(0.2 * 2.0_f32.sqrt() - 0.25),
                        2,
                    ),
                    ([0.0, -2.5, 0.0], [0.0, 0.0, 0.0, 1.0], None, 1),
                ] {
                    hull_state.position_inverse_mass[..3].copy_from_slice(&position);
                    hull_state.orientation = orientation;
                    let states = if hull_first {
                        [hull_state, mesh_state]
                    } else {
                        [mesh_state, hull_state]
                    };
                    world.reset(&states).unwrap();
                    let _pairs = world.step(0.01).unwrap();
                    let contact = world.readback_contacts().unwrap().pairs[0].1;
                    assert_eq!(
                        contact.is_contact(),
                        expected_depth.is_some(),
                        "backend: {backend:?}, hull_first: {hull_first}, position: {position:?}, contact: {contact:?}"
                    );
                    if let Some(depth) = expected_depth {
                        assert!((contact.depth_hit[0] - depth).abs() < 2e-3, "{contact:?}");
                        let sign = if axis == 1 { -1.0 } else { 1.0 };
                        let order = if hull_first { -1.0 } else { 1.0 };
                        assert!(contact.normal[axis] * sign * order > 0.99, "{contact:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn resident_polyline_hull_pipeline_rebuilds_for_lbvh_candidate_pairs() {
        let mesh = GpuRigidShape::Polyline {
            vertices: vec![[-5.0, -5.0, 0.0], [5.0, 5.0, 0.0]],
            segments: vec![[0, 1]],
        };
        let hull = GpuRigidShape::Convex {
            vertices: [-0.2, 0.2]
                .into_iter()
                .flat_map(|x| {
                    [-0.2, 0.2]
                        .into_iter()
                        .flat_map(move |y| [-0.2, 0.2].into_iter().map(move |z| [x, y, z]))
                })
                .collect(),
        };
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            eprintln!("resident polyline LBVH backend: {backend:?}");
            let mut mesh_state = body(0.0, 0.0);
            mesh_state.position_inverse_mass = [0.0; 4];
            mesh_state.inverse_inertia_sleep = [0.0; 4];
            let mut box_state = body(1.0, 0.0);
            box_state.position_inverse_mass = [1.0, 1.0, 0.15, 0.0];
            box_state.inverse_inertia_sleep = [0.0; 4];
            let mut states = vec![mesh_state, box_state];
            let mut shapes = vec![
                mesh.clone(),
                GpuRigidShape::Box {
                    half_extents: [0.2; 3],
                },
            ];
            for index in 0..15 {
                let mut distant = body(100.0 + index as f32 * 10.0, 0.0);
                distant.position_inverse_mass[3] = 0.0;
                distant.inverse_inertia_sleep = [0.0; 4];
                states.push(distant);
                shapes.push(GpuRigidShape::Sphere { radius: 0.1 });
            }
            let mut world = GpuRigidPrimitiveWorld::new_primitives(
                context.device(),
                context.queue(),
                &states,
                &shapes,
                config(),
            )
            .unwrap();
            assert!(!world.uses_exhaustive_pairs());
            let _pairs = world.step(0.01).unwrap();
            assert!(world.last_resident_contacts.is_some());
            assert!(
                world
                    .readback_contacts()
                    .unwrap()
                    .pairs
                    .iter()
                    .any(|(pair, contact)| { pair.a == 0 && pair.b == 1 && contact.is_contact() })
            );

            let mut hull_state = body(-1.0, 0.0);
            hull_state.position_inverse_mass = [-1.0, -1.0, 0.15, 0.0];
            hull_state.inverse_inertia_sleep = [0.0; 4];
            assert_eq!(
                world.append_primitive(hull_state, hull.clone()).unwrap(),
                17
            );
            assert!(!world.uses_exhaustive_pairs());
            let _pairs = world.step(0.01).unwrap();
            assert!(world.last_resident_contacts.is_some());
            let contacts = world.readback_contacts().unwrap();
            for other in [1, 17] {
                let index = contacts
                    .pairs
                    .iter()
                    .position(|(pair, _)| pair.a == 0 && pair.b == other)
                    .unwrap();
                let contact = contacts.pairs[index].1;
                let second = contacts.pair_extra[index][0];
                assert!(second.is_contact(), "{contacts:?}");
                assert!((second.depth_hit[0] - 0.05).abs() < 2e-3);
                assert!((contact.point[0] - second.point[0]).abs() > 0.39);
                assert!(!contacts.pair_extra[index][1].is_contact());
                assert!(contact.is_contact(), "backend: {backend:?}, other: {other}");
                assert!((contact.depth_hit[0] - 0.05).abs() < 2e-3, "{contact:?}");
                assert!(contact.normal[2] > 0.99, "{contact:?}");
            }
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn resident_mesh_hull_pipeline_rebuilds_for_lbvh_candidate_pairs() {
        let mesh = GpuRigidShape::TriangleMesh {
            vertices: vec![
                [-5.0, -5.0, 0.0],
                [5.0, -5.0, 0.0],
                [5.0, 5.0, 0.0],
                [-5.0, 5.0, 0.0],
            ],
            triangles: vec![[0, 1, 2], [0, 2, 3]],
        };
        let hull = GpuRigidShape::Convex {
            vertices: [-0.2, 0.2]
                .into_iter()
                .flat_map(|x| {
                    [-0.2, 0.2]
                        .into_iter()
                        .flat_map(move |y| [-0.2, 0.2].into_iter().map(move |z| [x, y, z]))
                })
                .collect(),
        };
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            let mut mesh_state = body(0.0, 0.0);
            mesh_state.position_inverse_mass = [0.0; 4];
            mesh_state.inverse_inertia_sleep = [0.0; 4];
            let mut box_state = body(1.0, 0.0);
            box_state.position_inverse_mass = [1.0, 1.0, 0.15, 0.0];
            box_state.inverse_inertia_sleep = [0.0; 4];
            let mut states = vec![mesh_state, box_state];
            let mut shapes = vec![
                mesh.clone(),
                GpuRigidShape::Box {
                    half_extents: [0.2; 3],
                },
            ];
            for index in 0..15 {
                let mut distant = body(100.0 + index as f32 * 10.0, 0.0);
                distant.position_inverse_mass[3] = 0.0;
                distant.inverse_inertia_sleep = [0.0; 4];
                states.push(distant);
                shapes.push(GpuRigidShape::Sphere { radius: 0.1 });
            }
            let mut world = GpuRigidPrimitiveWorld::new_primitives(
                context.device(),
                context.queue(),
                &states,
                &shapes,
                config(),
            )
            .unwrap();
            assert!(!world.uses_exhaustive_pairs());
            let _pairs = world.step(0.01).unwrap();
            assert!(
                world
                    .readback_contacts()
                    .unwrap()
                    .pairs
                    .iter()
                    .any(|(pair, contact)| { pair.a == 0 && pair.b == 1 && contact.is_contact() })
            );

            let mut hull_state = body(-1.0, 0.0);
            hull_state.position_inverse_mass = [-1.0, -1.0, 0.15, 0.0];
            hull_state.inverse_inertia_sleep = [0.0; 4];
            assert_eq!(
                world.append_primitive(hull_state, hull.clone()).unwrap(),
                17
            );
            assert!(!world.uses_exhaustive_pairs());
            let _pairs = world.step(0.01).unwrap();
            let contacts = world.readback_contacts().unwrap();
            for other in [1, 17] {
                let contact = contacts
                    .pairs
                    .iter()
                    .find(|(pair, _)| pair.a == 0 && pair.b == other)
                    .unwrap()
                    .1;
                assert!(contact.is_contact(), "backend: {backend:?}, other: {other}");
                assert!((contact.depth_hit[0] - 0.05).abs() < 2e-3, "{contact:?}");
                assert!(contact.normal[2] > 0.99, "{contact:?}");
            }
        }
    }

    #[test]
    fn resident_mesh_bvh_finds_distant_rotated_triangle_contacts() {
        let mut vertices = Vec::new();
        let mut triangles = Vec::new();
        for index in 0..64 {
            let x = index as f32 * 10.0;
            let first = vertices.len() as u32;
            vertices.extend([[x, 0.0, 0.0], [x + 1.0, 0.0, 0.0], [x, 1.0, 0.0]]);
            triangles.push([first, first + 1, first + 2]);
        }
        let mesh = GpuRigidShape::TriangleMesh {
            vertices,
            triangles,
        };
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            let mut mesh_state = body(2.0, 0.0);
            mesh_state.position_inverse_mass = [2.0, 1.0, 2.0, 0.0];
            mesh_state.orientation = [
                0.0,
                0.0,
                core::f32::consts::FRAC_PI_4.sin(),
                core::f32::consts::FRAC_PI_4.cos(),
            ];
            mesh_state.inverse_inertia_sleep = [0.0; 4];
            let mut sphere = body(0.0, 0.0);
            sphere.position_inverse_mass = [1.8, 371.2, 2.2, 0.0];
            sphere.inverse_inertia_sleep = [0.0; 4];
            let mut capsule = body(0.0, 0.0);
            capsule.position_inverse_mass = [1.8, 531.2, 2.4, 0.0];
            capsule.inverse_inertia_sleep = [0.0; 4];
            let mut box_state = body(0.0, 0.0);
            box_state.position_inverse_mass = [1.8, 431.2, 2.1, 0.0];
            box_state.inverse_inertia_sleep = [0.0; 4];
            let mut hull_state = body(0.0, 0.0);
            hull_state.position_inverse_mass = [1.8, 331.2, 2.1, 0.0];
            hull_state.inverse_inertia_sleep = [0.0; 4];
            let mut tetra_state = body(0.0, 0.0);
            tetra_state.position_inverse_mass = [1.8, 231.2, 2.1, 0.0];
            tetra_state.inverse_inertia_sleep = [0.0; 4];
            let mut cylinder_state = body(0.0, 0.0);
            cylinder_state.position_inverse_mass = [1.8, 171.2, 2.1, 0.0];
            cylinder_state.inverse_inertia_sleep = [0.0; 4];
            let mut cone_state = body(0.0, 0.0);
            cone_state.position_inverse_mass = [1.8, 131.2, 2.1, 0.0];
            cone_state.inverse_inertia_sleep = [0.0; 4];
            let mut world = GpuRigidPrimitiveWorld::new_primitives(
                context.device(),
                context.queue(),
                &[
                    mesh_state,
                    sphere,
                    capsule,
                    box_state,
                    hull_state,
                    tetra_state,
                    cylinder_state,
                    cone_state,
                ],
                &[
                    mesh.clone(),
                    GpuRigidShape::Sphere { radius: 0.5 },
                    GpuRigidShape::Capsule {
                        radius: 0.25,
                        half_length: 0.3,
                    },
                    GpuRigidShape::Box {
                        half_extents: [0.2, 0.2, 0.2],
                    },
                    GpuRigidShape::Convex {
                        vertices: [-0.2, 0.2]
                            .into_iter()
                            .flat_map(|x| {
                                [-0.2, 0.2].into_iter().flat_map(move |y| {
                                    [-0.2, 0.2].into_iter().map(move |z| [x, y, z])
                                })
                            })
                            .collect(),
                    },
                    GpuRigidShape::Convex {
                        vertices: vec![
                            [-0.2, -0.2, -0.2],
                            [0.2, -0.2, -0.2],
                            [0.0, 0.2, -0.2],
                            [0.0, 0.0, 0.2],
                        ],
                    },
                    GpuRigidShape::Cylinder {
                        radius: 0.25,
                        half_length: 0.2,
                    },
                    GpuRigidShape::Cone {
                        radius: 0.25,
                        half_length: 0.2,
                    },
                ],
                config(),
            )
            .unwrap();
            let _pairs = world.step(0.01).unwrap();
            let contacts = world.readback_contacts().unwrap();
            for (body, depth) in [
                (1, 0.3),
                (2, 0.15),
                (3, 0.1),
                (4, 0.1),
                (5, 0.1),
                (6, 0.1),
                (7, 0.1),
            ] {
                let contact = contacts
                    .pairs
                    .iter()
                    .find(|(pair, _)| pair.a == 0 && pair.b == body)
                    .unwrap()
                    .1;
                assert!(contact.is_contact(), "backend: {backend:?}, body: {body}");
                assert!((contact.depth_hit[0] - depth).abs() < 2e-3, "{contact:?}");
                assert!(contact.normal[2] > 0.99);
            }
        }
    }

    #[test]
    fn resident_polyline_ground_uses_referenced_vertices() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mesh = GpuRigidShape::Polyline {
            vertices: vec![
                [-1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0],
                [0.0, 1.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            segments: vec![[0, 1]],
        };
        let mut options = config();
        options.ground_half_extent = Some(10.0);
        let mut state = body(0.0, 0.0);
        state.position_inverse_mass = [0.0, 0.0, 0.0, 0.0];
        state.inverse_inertia_sleep = [0.0; 4];
        let mut world = GpuRigidPrimitiveWorld::new_primitives(
            context.device(),
            context.queue(),
            &[state],
            &[mesh],
            options,
        )
        .unwrap();
        let _pairs = world.step(0.01).unwrap();
        assert!(!world.readback_contacts().unwrap().ground[0].is_contact());
        state.position_inverse_mass[2] = -1.1;
        world.reset(&[state]).unwrap();
        let _pairs = world.step(0.01).unwrap();
        let ground = world.readback_contacts().unwrap().ground[0];
        assert!(ground.is_contact());
        assert!((ground.depth_hit[0] - 0.1).abs() < 1e-3, "{ground:?}");
    }

    #[test]
    fn resident_triangle_mesh_ground_uses_referenced_vertices() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mesh = GpuRigidShape::TriangleMesh {
            vertices: vec![
                [-1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0],
                [0.0, 1.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            triangles: vec![[0, 1, 2]],
        };
        let mut options = config();
        options.ground_half_extent = Some(10.0);
        let mut state = body(0.0, 0.0);
        state.position_inverse_mass = [0.0, 0.0, 0.0, 0.0];
        state.inverse_inertia_sleep = [0.0; 4];
        let mut world = GpuRigidPrimitiveWorld::new_primitives(
            context.device(),
            context.queue(),
            &[state],
            &[mesh],
            options,
        )
        .unwrap();
        let _pairs = world.step(0.01).unwrap();
        assert!(!world.readback_contacts().unwrap().ground[0].is_contact());
        state.position_inverse_mass[2] = -1.1;
        world.reset(&[state]).unwrap();
        let _pairs = world.step(0.01).unwrap();
        let ground = world.readback_contacts().unwrap().ground[0];
        assert!(ground.is_contact());
        assert!((ground.depth_hit[0] - 0.1).abs() < 1e-3, "{ground:?}");
    }

    #[test]
    fn collision_masks_filter_small_sphere_pairs_and_ground() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut a = body(0.0, 0.0);
        a.position_inverse_mass[2] = 0.5;
        let mut b = body(1.5, 0.0);
        b.position_inverse_mass[2] = 0.5;
        let mut options = config();
        options.ground_half_extent = Some(10.0);
        let mut world = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &[a, b],
            &[1.0; 2],
            options,
        )
        .unwrap();
        let group_a = GpuRigidCollisionGroups {
            memberships: 1,
            filter: 1,
        };
        let group_b = GpuRigidCollisionGroups {
            memberships: 2,
            filter: 2,
        };
        world.set_body_collision_groups(0, group_a).unwrap();
        world.set_body_collision_groups(1, group_b).unwrap();
        assert_eq!(world.candidate_pair_count(), 0);
        assert_eq!(world.step(0.01).unwrap(), 0);
        assert_eq!(
            world
                .readback_contacts()
                .unwrap()
                .ground
                .iter()
                .filter(|c| c.is_contact())
                .count(),
            2
        );
        world.set_ground_collision_groups(group_b).unwrap();
        let _pairs = world.step(0.01).unwrap();
        let contacts = world.readback_contacts().unwrap();
        assert!(!contacts.ground[0].is_contact());
        assert!(contacts.ground[1].is_contact());
        world.set_body_collision_groups(1, group_a).unwrap();
        assert_eq!(world.candidate_pair_count(), 1);
        let _pairs = world.step(0.01).unwrap();
        assert!(world.readback_contacts().unwrap().pairs[0].1.is_contact());
    }

    #[test]
    fn collision_masks_filter_primitive_pair_and_ground() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut a = body(0.0, 0.0);
        a.position_inverse_mass[2] = 0.5;
        let mut b = body(1.0, 0.0);
        b.position_inverse_mass[2] = 0.5;
        let mut options = config();
        options.ground_half_extent = Some(10.0);
        let mut world = GpuRigidPrimitiveWorld::new_primitives(
            context.device(),
            context.queue(),
            &[a, b],
            &[
                GpuRigidShape::Box {
                    half_extents: [1.0; 3],
                },
                GpuRigidShape::Sphere { radius: 1.0 },
            ],
            options,
        )
        .unwrap();
        let group_a = GpuRigidCollisionGroups {
            memberships: 1,
            filter: 1,
        };
        let group_b = GpuRigidCollisionGroups {
            memberships: 2,
            filter: 2,
        };
        world.set_body_collision_groups(0, group_a).unwrap();
        world.set_body_collision_groups(1, group_b).unwrap();
        world.set_ground_collision_groups(group_b).unwrap();
        assert_eq!(world.step(0.01).unwrap(), 0);
        let contacts = world.readback_contacts().unwrap();
        assert!(!contacts.ground[0].is_contact());
        assert!(contacts.ground[1].is_contact());
        world.set_body_collision_groups(1, group_a).unwrap();
        assert_eq!(world.step(0.01).unwrap(), 1);
        assert!(world.readback_contacts().unwrap().pairs[0].1.is_contact());
    }

    #[test]
    fn collision_masks_prune_lbvh_candidates_before_readback() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut states = (0..17)
            .map(|index| body(index as f32 * 10.0, 0.0))
            .collect::<Vec<_>>();
        states[1] = body(1.5, 0.0);
        let mut world = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &states,
            &[1.0; 17],
            config(),
        )
        .unwrap();
        assert!(!world.uses_exhaustive_pairs());
        world
            .set_body_collision_groups(
                0,
                GpuRigidCollisionGroups {
                    memberships: 1,
                    filter: 1,
                },
            )
            .unwrap();
        world
            .set_body_collision_groups(
                1,
                GpuRigidCollisionGroups {
                    memberships: 2,
                    filter: 2,
                },
            )
            .unwrap();
        assert_eq!(world.step(0.01).unwrap(), 0);
        world
            .set_body_collision_groups(
                1,
                GpuRigidCollisionGroups {
                    memberships: 1,
                    filter: 1,
                },
            )
            .unwrap();
        assert_eq!(world.step(0.01).unwrap(), 1);
    }

    #[test]
    fn collision_masks_survive_dense_topology_edits() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut options = config();
        options.ground_half_extent = Some(10.0);
        let mut world = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &[body(0.0, 0.0), body(3.0, 0.0)],
            &[1.0; 2],
            options,
        )
        .unwrap();
        let body_groups = GpuRigidCollisionGroups {
            memberships: 4,
            filter: 8,
        };
        let ground_groups = GpuRigidCollisionGroups {
            memberships: 8,
            filter: 4,
        };
        world.set_body_collision_groups(1, body_groups).unwrap();
        world.set_ground_collision_groups(ground_groups).unwrap();
        let _removed = world.remove_body(0).unwrap();
        assert_eq!(world.body_collision_groups(0), Some(body_groups));
        assert_eq!(world.ground_collision_groups(), Some(ground_groups));
        assert_eq!(world.append_body(body(6.0, 0.0), 1.0).unwrap(), 1);
        assert_eq!(world.body_collision_groups(0), Some(body_groups));
        assert_eq!(
            world.body_collision_groups(1),
            Some(GpuRigidCollisionGroups::default())
        );
    }

    #[test]
    fn batch_collision_masks_use_environment_local_indices() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let initial = [body(0.0, 0.0), body(1.5, 0.0)];
        let environments = [
            GpuRigidSphereEnvironment {
                states: &initial,
                radii: &[1.0; 2],
            },
            GpuRigidSphereEnvironment {
                states: &initial,
                radii: &[1.0; 2],
            },
        ];
        let mut batch =
            GpuRigidSphereBatch::new(context.device(), context.queue(), &environments, config())
                .unwrap();
        let group_a = GpuRigidCollisionGroups {
            memberships: 1,
            filter: 1,
        };
        let group_b = GpuRigidCollisionGroups {
            memberships: 2,
            filter: 2,
        };
        batch
            .set_body_collision_groups_environment(0, 0, group_a)
            .unwrap();
        batch
            .set_body_collision_groups_environment(0, 1, group_b)
            .unwrap();
        assert_eq!(batch.body_collision_groups_environment(0, 0), Some(group_a));
        assert_eq!(
            batch.body_collision_groups_environment(1, 0),
            Some(GpuRigidCollisionGroups::default())
        );
        assert_eq!(batch.step(0.01).unwrap(), 1);
        assert!(
            !batch
                .readback_contacts_environment(1)
                .unwrap()
                .pairs
                .is_empty()
        );
        assert!(
            batch
                .readback_contacts_environment(0)
                .unwrap()
                .pairs
                .is_empty()
        );
    }

    fn body(x: f32, velocity_x: f32) -> GpuRigidBodyState {
        GpuRigidBodyState {
            position_inverse_mass: [x, 0.0, 3.0, 1.0],
            orientation: [0.0, 0.0, 0.0, 1.0],
            linear_velocity: [velocity_x, 0.0, 0.0, 0.0],
            angular_velocity: [0.0; 4],
            inverse_inertia_sleep: [1.0, 1.0, 1.0, 0.0],
        }
    }

    fn config() -> GpuRigidSphereWorldConfig {
        GpuRigidSphereWorldConfig {
            gravity: [0.0; 3],
            ground_half_extent: None,
            solve: GpuRigidSphereSolveParams {
                friction: 0.0,
                restitution: 0.0,
                bias_factor: 0.0,
                iterations: 4,
            },
            sleep: SleepSettings {
                time_threshold: 0.015,
                ..Default::default()
            },
        }
    }

    #[test]
    fn resident_kinematic_sphere_pushes_dynamic_body_without_impulse_feedback() {
        let mut moving = body(0.0, 0.0);
        moving.position_inverse_mass[3] = 0.0;
        moving.inverse_inertia_sleep = [0.0; 4];
        let mut sleeping = body(1.9, 0.0);
        sleeping.inverse_inertia_sleep[3] = 1.0;
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let mut world = GpuRigidSphereWorld::new(
                context.device(),
                context.queue(),
                &[moving, sleeping],
                &[1.0; 2],
                config(),
            )
            .unwrap();
            world
                .set_kinematic_motion(0, Some(([1.0, 0.0, 0.0], [0.0; 3])))
                .unwrap();
            let _steps = world.step(0.01).unwrap();
            let _steps = world.step(0.01).unwrap();
            let result = world.readback().unwrap();
            assert!((result[0].position_inverse_mass[0] - 0.02).abs() < 1e-5);
            assert_eq!(result[0].linear_velocity, [1.0, 0.0, 0.0, 1.0]);
            assert_eq!(result[0].inverse_inertia_sleep, [0.0; 4]);
            assert!(result[1].linear_velocity[0] > 0.9);
            assert_eq!(result[1].inverse_inertia_sleep[3], 0.0);
            assert!(
                (result[1].position_inverse_mass[0] - 1.91).abs() < 1e-5,
                "states: {result:?}"
            );
            world.set_kinematic_motion(0, None).unwrap();
            let _steps = world.step(0.01).unwrap();
            let stopped = world.readback().unwrap();
            assert_eq!(
                stopped[0].position_inverse_mass,
                result[0].position_inverse_mass
            );
            assert_eq!(stopped[0].linear_velocity, [0.0; 4]);
            eprintln!("kinematic contact push passed on {backend:?}");
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn moving_kinematic_contact_prevents_sleep_for_slow_and_rotating_support() {
        let mut support = body(0.0, 0.0);
        support.position_inverse_mass[3] = 0.0;
        support.inverse_inertia_sleep = [0.0; 4];
        let mut sleeping = body(1.9, 0.0);
        sleeping.inverse_inertia_sleep[3] = 1.0;
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            for (rotating, resident_lbvh) in
                [(false, false), (true, false), (false, true), (true, true)]
            {
                let mut options = config();
                if rotating {
                    options.solve.friction = 1.0;
                    options.solve.bias_factor = 0.2;
                }
                let mut bodies = vec![support, sleeping];
                if resident_lbvh {
                    for index in 2..=EXHAUSTIVE_PAIR_BODY_LIMIT {
                        let mut distant = support;
                        distant.position_inverse_mass[0] = index as f32 * 10.0;
                        bodies.push(distant);
                    }
                }
                let mut world = GpuRigidSphereWorld::new(
                    context.device(),
                    context.queue(),
                    &bodies,
                    &vec![1.0; bodies.len()],
                    options,
                )
                .unwrap();
                assert_eq!(world.uses_exhaustive_pairs(), !resident_lbvh);
                let velocities = if rotating {
                    ([0.0; 3], [0.0, 0.0, 0.005])
                } else {
                    ([0.005, 0.0, 0.0], [0.0; 3])
                };
                world.set_kinematic_motion(0, Some(velocities)).unwrap();
                for frame in 0..10 {
                    let _steps = world.step(0.01).unwrap();
                    let states = world.readback().unwrap();
                    assert_eq!(
                        states[1].inverse_inertia_sleep[3], 0.0,
                        "frame {frame}, rotating {rotating}, {backend:?}: {states:?}"
                    );
                    if !rotating {
                        assert!(
                            states[1].linear_velocity[0] > 0.004,
                            "frame {frame}: {states:?}"
                        );
                    } else if frame == 0 {
                        assert!(
                            states[1].linear_velocity[1].abs() > 1e-4
                                || states[1].angular_velocity[2].abs() > 1e-4,
                            "rotation must transfer through friction: {states:?}"
                        );
                    }
                }
                eprintln!(
                    "kinematic sustained contact rotating={rotating}, LBVH={resident_lbvh} passed on {backend:?}"
                );
            }
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn slow_kinematic_motion_keeps_a_grounded_joint_chain_awake() {
        let mut support = body(-5.0, 0.0);
        support.position_inverse_mass[2] = 0.99;
        support.position_inverse_mass[3] = 0.0;
        support.inverse_inertia_sleep = [0.0; 4];
        let mut first = body(0.0, 0.0);
        first.position_inverse_mass[2] = 0.99;
        first.inverse_inertia_sleep[3] = 1.0;
        let mut second = first;
        second.position_inverse_mass[0] = 5.0;
        let mut stationary = support;
        stationary.position_inverse_mass[0] = 12.0;
        let mut isolated = first;
        isolated.position_inverse_mass[0] = 17.0;
        let mut options = config();
        options.ground_half_extent = Some(20.0);
        let joints = [
            GpuRigidBallJoint {
                body_a: 0,
                body_b: 1,
                local_anchor_a: [5.0, 0.0, 0.0],
                local_anchor_b: [0.0; 3],
            },
            GpuRigidBallJoint {
                body_a: 1,
                body_b: 2,
                local_anchor_a: [5.0, 0.0, 0.0],
                local_anchor_b: [0.0; 3],
            },
            GpuRigidBallJoint {
                body_a: 3,
                body_b: 4,
                local_anchor_a: [5.0, 0.0, 0.0],
                local_anchor_b: [0.0; 3],
            },
        ];
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let mut world = GpuRigidSphereWorld::new(
                context.device(),
                context.queue(),
                &[support, first, second, stationary, isolated],
                &[1.0; 5],
                options,
            )
            .unwrap();
            world.set_ball_joints(&joints).unwrap();
            for mode in 0..3 {
                world
                    .reset(&[support, first, second, stationary, isolated])
                    .unwrap();
                world
                    .set_kinematic_motion(0, Some(([0.005, 0.0, 0.0], [0.0; 3])))
                    .unwrap();
                for frame in 0..10 {
                    let _steps = match mode {
                        0 => world.step(0.01),
                        1 => world.step_substeps(0.005, 2),
                        _ => world.step_temporal(0.01, 2, GpuRigidTemporalSolveParams::default()),
                    }
                    .unwrap();
                    let states = world.readback().unwrap();
                    assert_eq!(
                        states[4].inverse_inertia_sleep[3], 1.0,
                        "unrelated joint island must stay asleep: {states:?}"
                    );
                    for body in 1..3 {
                        assert_eq!(
                            states[body].inverse_inertia_sleep[3], 0.0,
                            "mode {mode}, frame {frame}, body {body}, {backend:?}: {states:?}"
                        );
                        let minimum = if mode == 2 { 0.0005 } else { 0.004 };
                        assert!(
                            states[body].linear_velocity[0] > minimum,
                            "mode {mode}, frame {frame}, body {body}: {states:?}"
                        );
                    }
                }
                eprintln!(
                    "kinematic grounded joint chain mode={mode} and isolation passed on {backend:?}"
                );
            }
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn kinematic_fixed_hinge_and_slider_preserve_constrained_and_free_motion() {
        let mut support = body(0.0, 0.0);
        support.position_inverse_mass[2] = 0.99;
        support.position_inverse_mass[3] = 0.0;
        support.inverse_inertia_sleep = [0.0; 4];
        let mut dynamic = body(0.0, 0.0);
        dynamic.position_inverse_mass[2] = 0.99;
        dynamic.inverse_inertia_sleep[3] = 1.0;
        let mut options = config();
        options.ground_half_extent = Some(20.0);
        let identity = [0.0, 0.0, 0.0, 1.0];
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            for case in 0..5 {
                let mut world = GpuRigidSphereWorld::new(
                    context.device(),
                    context.queue(),
                    &[support, dynamic],
                    &[1.0; 2],
                    options,
                )
                .unwrap();
                // Disable overlapping body pairs while retaining finite ground support.
                for index in 0..2 {
                    world
                        .set_body_collision_groups(
                            index,
                            GpuRigidCollisionGroups {
                                memberships: 1,
                                filter: 2,
                            },
                        )
                        .unwrap();
                }
                match case {
                    0 => world
                        .set_fixed_joints(&[GpuRigidFixedJoint {
                            body_a: 0,
                            body_b: 1,
                            local_anchor_a: [0.0; 3],
                            local_anchor_b: [0.0; 3],
                            local_rotation_a: identity,
                            local_rotation_b: identity,
                        }])
                        .unwrap(),
                    1 | 2 => world
                        .set_revolute_joints(&[GpuRigidRevoluteJoint {
                            body_a: 0,
                            body_b: 1,
                            local_anchor_a: [0.0; 3],
                            local_anchor_b: [0.0; 3],
                            local_axis_a: [0.0, 0.0, 1.0],
                            local_axis_b: [0.0, 0.0, 1.0],
                        }])
                        .unwrap(),
                    _ => world
                        .set_prismatic_joints(&[GpuRigidPrismaticJoint {
                            body_a: 0,
                            body_b: 1,
                            local_anchor_a: [0.0; 3],
                            local_anchor_b: [0.0; 3],
                            local_rotation_a: identity,
                            local_rotation_b: identity,
                        }])
                        .unwrap(),
                }
                let motion = match case {
                    0 | 2 => ([0.0; 3], [0.0, 0.0, 0.005]),
                    1 => ([0.0; 3], [0.005, 0.0, 0.0]),
                    3 => ([0.005, 0.0, 0.0], [0.0; 3]),
                    _ => ([0.0, 0.0, 0.005], [0.0; 3]),
                };
                world.set_kinematic_motion(0, Some(motion)).unwrap();
                for frame in 0..10 {
                    let _steps = world.step(0.01).unwrap();
                    let states = world.readback().unwrap();
                    assert_eq!(
                        states[1].inverse_inertia_sleep[3], 0.0,
                        "case {case}, frame {frame}, {backend:?}: {states:?}"
                    );
                    if frame == 0 {
                        let contacts = world.readback_contacts().unwrap();
                        assert!(
                            contacts.ground[1].is_contact(),
                            "ground must support the sleeping body"
                        );
                        assert!(
                            contacts
                                .pairs
                                .iter()
                                .all(|(_, contact)| !contact.is_contact()),
                            "body pair contacts must not contribute to joint transmission"
                        );
                    }
                    match case {
                        0 => {
                            assert!(states[1].angular_velocity[2] > 0.004, "{states:?}");
                            assert!(
                                (states[1].orientation[2] - states[0].orientation[2]).abs() < 1e-5,
                                "{states:?}"
                            );
                        }
                        1 => assert!(states[1].angular_velocity[0] > 0.004, "{states:?}"),
                        2 => assert!(
                            states[1].angular_velocity[2].abs() < 1e-6,
                            "free hinge axis must not transmit rotation: {states:?}"
                        ),
                        3 => assert!(states[1].linear_velocity[0] > 0.004, "{states:?}"),
                        _ => assert!(
                            states[1].linear_velocity[2].abs() < 1e-6,
                            "free slider axis must not transmit translation: {states:?}"
                        ),
                    }
                }
                eprintln!("kinematic constrained/free joint case={case} passed on {backend:?}");
            }
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn batch_kinematic_motion_wakes_only_the_selected_environment() {
        let mut stationary = body(0.0, 0.0);
        stationary.position_inverse_mass[3] = 0.0;
        stationary.inverse_inertia_sleep = [0.0; 4];
        let mut sleeping = body(1.9, 0.0);
        sleeping.inverse_inertia_sleep[3] = 1.0;
        let initial = [stationary, sleeping];
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let mut batch = GpuRigidSphereBatch::new(
                context.device(),
                context.queue(),
                &[
                    GpuRigidSphereEnvironment {
                        states: &initial,
                        radii: &[1.0; 2],
                    },
                    GpuRigidSphereEnvironment {
                        states: &initial,
                        radii: &[1.0; 2],
                    },
                ],
                config(),
            )
            .unwrap();
            assert!(batch.set_kinematic_motion(2, 0, None).is_err());
            assert!(batch.set_kinematic_motion(0, 2, None).is_err());
            assert!(
                batch
                    .set_kinematic_motion(1, 0, Some(([f32::NAN; 3], [0.0; 3])))
                    .is_err()
            );
            batch
                .set_kinematic_motion(1, 0, Some(([1.0, 0.0, 0.0], [0.0; 3])))
                .unwrap();
            let _steps = batch.step(0.01).unwrap();
            let _steps = batch.step(0.01).unwrap();
            let unchanged = batch.readback_environment(0).unwrap();
            let moved = batch.readback_environment(1).unwrap();
            assert_eq!(
                unchanged[0].position_inverse_mass,
                initial[0].position_inverse_mass
            );
            assert_eq!(unchanged[0].linear_velocity, [0.0; 4]);
            assert_eq!(
                unchanged[1].position_inverse_mass,
                initial[1].position_inverse_mass
            );
            assert_eq!(unchanged[1].inverse_inertia_sleep[3], 1.0);
            assert!((moved[0].position_inverse_mass[0] - 0.02).abs() < 1e-5);
            assert!(moved[1].linear_velocity[0] > 0.9);
            assert_eq!(moved[1].inverse_inertia_sleep[3], 0.0);
            eprintln!("kinematic batch isolation passed on {backend:?}");
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn batch_environment_topology_preserves_survivors_and_joints() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let first = [body(0.0, 0.0)];
        let second = [body(10.0, 0.0)];
        let mut batch = GpuRigidSphereBatch::new(
            context.device(),
            context.queue(),
            &[
                GpuRigidSphereEnvironment {
                    states: &first,
                    radii: &[0.1],
                },
                GpuRigidSphereEnvironment {
                    states: &second,
                    radii: &[0.1],
                },
            ],
            config(),
        )
        .unwrap();
        assert!(batch.append_environment(&[body(99.0, 0.0)], &[]).is_err());
        assert_eq!(batch.ranges.len(), 2);
        assert_eq!(
            batch
                .append_environment(&[body(20.0, 0.0), body(21.0, 0.0)], &[0.1; 2])
                .unwrap(),
            2
        );
        let joint = GpuRigidBallJoint {
            body_a: 0,
            body_b: 1,
            local_anchor_a: [1.0, 0.0, 0.0],
            local_anchor_b: [0.0; 3],
        };
        batch.set_ball_joints_environment(2, &[joint]).unwrap();
        let groups = GpuRigidCollisionGroups {
            memberships: 2,
            filter: 4,
        };
        batch
            .set_body_collision_groups_environment(1, 0, groups)
            .unwrap();
        assert_eq!(
            batch.remove_environment(0).unwrap()[0].position_inverse_mass[0],
            0.0
        );
        assert_eq!(batch.environment_range(0), Some(0..1));
        assert_eq!(batch.environment_range(1), Some(1..3));
        assert_eq!(batch.body_collision_groups_environment(0, 0), Some(groups));
        assert_eq!(batch.ball_joints_environment(1).unwrap(), vec![joint]);
        assert_eq!(
            batch.readback_environment(0).unwrap()[0].position_inverse_mass[0],
            10.0
        );
        assert_eq!(
            batch.readback_environment(1).unwrap()[0].position_inverse_mass[0],
            20.0
        );
        assert_eq!(batch.append_environment(&[], &[]).unwrap(), 2);
        assert_eq!(batch.environment_range(2), Some(3..3));
        assert!(batch.remove_environment(2).unwrap().is_empty());
        assert!(batch.remove_environment(2).is_err());
        let _contacts = batch.step(0.01).unwrap();
        assert_eq!(batch.ball_joints_environment(1).unwrap(), vec![joint]);
    }

    #[test]
    fn primitive_batch_environment_topology_preserves_shapes() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let sphere = GpuRigidShape::Sphere { radius: 0.1 };
        let box_shape = GpuRigidShape::Box {
            half_extents: [0.2; 3],
        };
        let capsule = GpuRigidShape::Capsule {
            radius: 0.1,
            half_length: 0.3,
        };
        let initial = [body(0.0, 0.0)];
        let mut batch = GpuRigidSphereBatch::new_primitives(
            context.device(),
            context.queue(),
            &[GpuRigidPrimitiveEnvironment {
                states: &initial,
                shapes: core::slice::from_ref(&sphere),
            }],
            config(),
        )
        .unwrap();
        assert!(batch.append_environment(&initial, &[0.1]).is_err());
        assert_eq!(
            batch
                .append_environment_primitives(
                    &[body(10.0, 0.0), body(20.0, 0.0)],
                    &[box_shape.clone(), capsule.clone()],
                )
                .unwrap(),
            1
        );
        let removed = batch.remove_environment_primitives(0).unwrap();
        assert_eq!(removed.len(), 1);
        assert_eq!(removed[0].0.position_inverse_mass[0], 0.0);
        assert_eq!(removed[0].1, sphere);
        assert_eq!(batch.environment_range(0), Some(0..2));
        assert_eq!(
            batch.world.shapes.as_ref().unwrap(),
            &vec![box_shape, capsule]
        );
        assert_eq!(
            batch.readback_environment(0).unwrap()[1].position_inverse_mass[0],
            20.0
        );
        let _contacts = batch.step(0.01).unwrap();
    }

    #[test]
    fn batch_can_repopulate_after_removing_last_environment() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut empty =
            GpuRigidSphereBatch::new(context.device(), context.queue(), &[], config()).unwrap();
        assert_eq!(
            empty.append_environment(&[body(3.0, 0.0)], &[0.1]).unwrap(),
            0
        );
        assert_eq!(
            empty.readback_environment(0).unwrap()[0].position_inverse_mass[0],
            3.0
        );
        let initial = [body(1.0, 0.0)];
        let mut batch = GpuRigidSphereBatch::new(
            context.device(),
            context.queue(),
            &[GpuRigidSphereEnvironment {
                states: &initial,
                radii: &[0.1],
            }],
            config(),
        )
        .unwrap();
        assert_eq!(batch.remove_environment(0).unwrap().len(), 1);
        assert!(batch.is_empty());
        assert_eq!(
            batch.append_environment(&[body(2.0, 0.0)], &[0.1]).unwrap(),
            0
        );
        assert_eq!(
            batch.readback_environment(0).unwrap()[0].position_inverse_mass[0],
            2.0
        );
        let _contacts = batch.step(0.01).unwrap();
    }

    #[test]
    fn discarded_sphere_topology_keeps_survivors_on_gpu() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut world = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &[body(0.0, 0.0), body(10.0, 0.0), body(20.0, 0.0)],
            &[0.1; 3],
            config(),
        )
        .unwrap();
        let _contacts = world.step(0.01).unwrap();
        world.discard_body(1).unwrap();
        assert!(world.discard_body(2).is_err());
        let states = world.readback().unwrap();
        assert_eq!(states.len(), 2);
        assert_eq!(states[0].position_inverse_mass[0], 0.0);
        assert_eq!(states[1].position_inverse_mass[0], 20.0);

        let first = [body(30.0, 0.0)];
        let second = [body(40.0, 0.0), body(41.0, 0.0)];
        let mut batch = GpuRigidSphereBatch::new(
            context.device(),
            context.queue(),
            &[
                GpuRigidSphereEnvironment {
                    states: &first,
                    radii: &[0.1],
                },
                GpuRigidSphereEnvironment {
                    states: &second,
                    radii: &[0.1; 2],
                },
            ],
            config(),
        )
        .unwrap();
        let joint = GpuRigidBallJoint {
            body_a: 0,
            body_b: 1,
            local_anchor_a: [1.0, 0.0, 0.0],
            local_anchor_b: [0.0; 3],
        };
        batch.set_ball_joints_environment(1, &[joint]).unwrap();
        batch.discard_environment(0).unwrap();
        assert_eq!(batch.ball_joints_environment(0).unwrap(), vec![joint]);
        batch.discard_body_environment(0, 1).unwrap();
        assert!(batch.ball_joints_environment(0).unwrap().is_empty());
        assert_eq!(
            batch.readback_environment(0).unwrap()[0].position_inverse_mass[0],
            40.0
        );
        assert_eq!(batch.environment_range(0), Some(0..1));
        assert!(batch.discard_environment(1).is_err());
    }

    #[test]
    fn discarded_primitive_topology_keeps_shape_alignment() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let sphere = GpuRigidShape::Sphere { radius: 0.1 };
        let cube = GpuRigidShape::Box {
            half_extents: [0.2; 3],
        };
        let capsule = GpuRigidShape::Capsule {
            radius: 0.1,
            half_length: 0.3,
        };
        let mut world = GpuRigidSphereWorld::new_primitives(
            context.device(),
            context.queue(),
            &[body(0.0, 0.0), body(10.0, 0.0), body(20.0, 0.0)],
            &[sphere.clone(), cube.clone(), capsule.clone()],
            config(),
        )
        .unwrap();
        world.discard_primitive(1).unwrap();
        assert_eq!(
            world.shapes.as_ref().unwrap(),
            &vec![sphere.clone(), capsule.clone()]
        );
        assert_eq!(world.readback().unwrap()[1].position_inverse_mass[0], 20.0);

        let first = [body(30.0, 0.0)];
        let second = [body(40.0, 0.0), body(50.0, 0.0)];
        let mut batch = GpuRigidSphereBatch::new_primitives(
            context.device(),
            context.queue(),
            &[
                GpuRigidPrimitiveEnvironment {
                    states: &first,
                    shapes: core::slice::from_ref(&sphere),
                },
                GpuRigidPrimitiveEnvironment {
                    states: &second,
                    shapes: &[cube.clone(), capsule.clone()],
                },
            ],
            config(),
        )
        .unwrap();
        batch.discard_environment_primitives(0).unwrap();
        batch.discard_primitive_environment(0, 0).unwrap();
        assert_eq!(batch.world.shapes.as_ref().unwrap(), &vec![capsule]);
        assert_eq!(
            batch.readback_environment(0).unwrap()[0].position_inverse_mass[0],
            50.0
        );
        let _contacts = batch.step(0.01).unwrap();
    }

    #[test]
    fn speed_cap_covers_exhaustive_temporal_and_resident_steps() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut exhaustive = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &[body(0.0, 5.0)],
            &[0.1],
            config(),
        )
        .unwrap();
        assert!(exhaustive.set_max_linear_speed(Some(f32::NAN)).is_err());
        exhaustive.set_max_linear_speed(Some(1.0)).unwrap();
        let _contacts = exhaustive.step(0.1).unwrap();
        let state = exhaustive.readback().unwrap()[0];
        assert!((state.position_inverse_mass[0] - 0.1).abs() < 1e-4);
        assert!((state.linear_velocity[0] - 1.0).abs() < 1e-4);

        let _index = exhaustive.append_body(body(20.0, 0.0), 0.1).unwrap();
        assert_eq!(exhaustive.max_linear_speed(), Some(1.0));
        let _removed = exhaustive.remove_body(1).unwrap();
        assert_eq!(exhaustive.max_linear_speed(), Some(1.0));

        exhaustive.reset(&[body(0.0, 5.0)]).unwrap();
        let _contacts = exhaustive
            .step_temporal(0.1, 2, GpuRigidTemporalSolveParams::default())
            .unwrap();
        let state = exhaustive.readback().unwrap()[0];
        assert!((state.position_inverse_mass[0] - 0.1).abs() < 1e-4);
        assert!((state.linear_velocity[0] - 1.0).abs() < 1e-4);

        let states = (0..70)
            .map(|index| body(index as f32 * 10.0, if index == 0 { 5.0 } else { 0.0 }))
            .collect::<Vec<_>>();
        let mut resident = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &states,
            &vec![0.1; states.len()],
            config(),
        )
        .unwrap();
        resident.set_max_linear_speed(Some(1.0)).unwrap();
        let _result = resident.step_gpu_resident(0.1).unwrap();
        let state = resident.readback().unwrap()[0];
        assert!((state.position_inverse_mass[0] - 0.1).abs() < 1e-4);
        assert!((state.linear_velocity[0] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn speed_cap_applies_after_contact_impulse() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut moving_static = body(1.5, -10.0);
        moving_static.position_inverse_mass[3] = 0.0;
        moving_static.inverse_inertia_sleep = [0.0; 4];
        let states = [body(0.0, 0.0), moving_static];
        let mut uncapped = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &states,
            &[1.0; 2],
            config(),
        )
        .unwrap();
        let _contacts = uncapped.step(0.01).unwrap();
        assert!(uncapped.readback().unwrap()[0].linear_velocity[0].abs() > 1.0);
        let mut world = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &states,
            &[1.0; 2],
            config(),
        )
        .unwrap();
        world.set_max_linear_speed(Some(1.0)).unwrap();
        let _contacts = world.step(0.01).unwrap();
        let actual = world.readback().unwrap();
        assert!(actual[0].linear_velocity[0].abs() <= 1.0 + 1e-4);
        assert!((actual[1].linear_velocity[0] + 10.0).abs() < 1e-4);
    }

    #[test]
    fn dx12_speed_cap_executes_after_solver() {
        let Ok(context) = GpuContactDevice::new_with_backends(wgpu::Backends::DX12) else {
            return;
        };
        let mut world = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &[body(0.0, 5.0)],
            &[0.1],
            config(),
        )
        .unwrap();
        world.set_max_linear_speed(Some(1.0)).unwrap();
        let _contacts = world.step(0.1).unwrap();
        let state = world.readback().unwrap()[0];
        assert!((state.position_inverse_mass[0] - 0.1).abs() < 1e-4);
        assert!((state.linear_velocity[0] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn exhaustive_world_impulse_history_matches_momentum_and_reset() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let states = [body(0.0, 1.0), body(1.5, -1.0)];
        let mut world = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &states,
            &[1.0; 2],
            config(),
        )
        .unwrap();
        assert!(world.uses_exhaustive_pairs());
        assert_eq!(world.readback_contact_impulses().unwrap().dt, None);
        assert_eq!(world.step(0.01).unwrap(), 1);
        let history = world.readback_contact_impulses().unwrap();
        assert_eq!(history.dt, Some(0.01));
        assert_eq!(history.contacts.len(), 1);
        let impulse = history.contacts[0].impulse_on_body_b();
        let actual = world.readback().unwrap();
        for (axis, value) in impulse.into_iter().enumerate() {
            assert!(
                (value - (actual[1].linear_velocity[axis] - states[1].linear_velocity[axis])).abs()
                    < 1e-4
            );
        }
        world.reset(&states).unwrap();
        assert_eq!(world.readback_contact_impulses().unwrap().dt, None);
    }

    #[test]
    fn resident_lbvh_world_step_matches_pair_reference_without_pair_readback() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let states = (0..70)
            .map(|index| match index {
                0 => body(0.0, 1.0),
                1 => body(1.5, -1.0),
                _ => body(index as f32 * 10.0, 0.0),
            })
            .collect::<Vec<_>>();
        let mut settings = config();
        settings.solve.restitution = 1.0;
        let mut resident = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &states,
            &[1.0; 70],
            settings,
        )
        .unwrap();
        let mut reference = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &states,
            &[1.0; 70],
            settings,
        )
        .unwrap();
        let (candidates, output) = resident.step_gpu_resident(0.01).unwrap();
        assert_eq!(
            output
                .readback_pairs(context.device(), context.queue(), &candidates)
                .unwrap()
                .pairs
                .len(),
            1
        );
        assert_eq!(reference.step(0.01).unwrap(), 1);
        let actual = resident.readback().unwrap();
        let expected = reference.readback().unwrap();
        for (left, right) in actual.iter().zip(&expected) {
            for axis in 0..3 {
                assert!((left.linear_velocity[axis] - right.linear_velocity[axis]).abs() < 1e-4);
            }
            assert_eq!(
                left.inverse_inertia_sleep[3],
                right.inverse_inertia_sleep[3]
            );
        }
    }

    #[test]
    fn ordinary_step_uses_resident_candidates_and_preserves_contact_diagnostics() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut states = (0..20)
            .map(|index| body(index as f32 * 10.0, 0.0))
            .collect::<Vec<_>>();
        states[1] = body(1.5, 0.0);
        let mut settings = config();
        settings.ground_half_extent = Some(50.0);
        states[0].position_inverse_mass[2] = 1.0;
        states[1].position_inverse_mass[2] = 1.0;
        let mut world = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &states,
            &[1.0; 20],
            settings,
        )
        .unwrap();
        assert!(!world.uses_exhaustive_pairs());
        assert!(world.can_use_resident_step());
        assert_eq!(world.step(0.01).unwrap(), 1);
        assert_eq!(world.candidate_pair_count(), 1);
        assert!(world.last_resident_contacts.is_some());
        let contacts = world.readback_contacts().unwrap();
        assert_eq!(contacts.pairs.len(), 1);
        assert_eq!((contacts.pairs[0].0.a, contacts.pairs[0].0.b), (0, 1));
        assert!(contacts.pairs[0].1.is_contact());
        assert_eq!(contacts.ground.len(), 20);
        assert!(contacts.ground[0].is_contact());
        world.write_body(1, body(1000.0, 0.0)).unwrap();
        assert_eq!(world.step(0.01).unwrap(), 0);
        let contacts = world.readback_contacts().unwrap();
        assert!(contacts.pairs.is_empty());
        assert_eq!(contacts.ground.len(), 20);
    }

    #[test]
    fn ordinary_step_uses_resident_islands_when_worst_case_exceeds_serial_limit() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut states = (0..70)
            .map(|index| body(index as f32 * 10.0, 0.0))
            .collect::<Vec<_>>();
        states[1] = body(1.5, 0.0);
        let mut world = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &states,
            &[1.0; 70],
            GpuRigidSphereWorldConfig {
                ground_half_extent: None,
                gravity: [0.0; 3],
                ..Default::default()
            },
        )
        .unwrap();
        assert!(world.can_use_resident_step());
        assert_eq!(world.step(0.01).unwrap(), 1);
        assert!(world.last_resident_contacts.is_some());
        assert_eq!(world.readback_contacts().unwrap().pairs.len(), 1);
    }

    fn check_colored_resident_chain(context: &GpuContactDevice) {
        let states = (0..70)
            .map(|index| body(index as f32 * 1.95, if index == 0 { 1.0 } else { 0.0 }))
            .collect::<Vec<_>>();
        let mut settings = config();
        settings.solve.iterations = 12;
        settings.sleep.enabled = false;
        let mut world = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &states,
            &[1.0; 70],
            settings,
        )
        .unwrap();
        assert!(world.can_use_resident_step());
        for _ in 0..120 {
            let _candidate_count = world.step(1.0 / 120.0).unwrap();
        }
        let actual = world.readback().unwrap();
        assert!(actual.iter().all(|state| {
            state.position_inverse_mass.iter().all(|v| v.is_finite())
                && state.linear_velocity.iter().all(|v| v.is_finite())
        }));
        let momentum = actual
            .iter()
            .map(|state| state.linear_velocity[0])
            .sum::<f32>();
        let energy = actual
            .iter()
            .map(|state| state.linear_velocity[0].powi(2))
            .sum::<f32>();
        assert!((momentum - 1.0).abs() < 1e-3, "momentum={momentum}");
        assert!(energy <= 1.001, "energy={energy}");
    }

    #[test]
    fn colored_resident_chain_keeps_momentum_and_finite_energy() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        check_colored_resident_chain(&context);
    }

    #[test]
    fn dx12_colored_resident_chain_keeps_momentum_and_finite_energy() {
        let Ok(context) = GpuContactDevice::new_with_backends(wgpu::Backends::DX12) else {
            return;
        };
        check_colored_resident_chain(&context);
    }

    #[test]
    fn ordinary_primitive_step_returns_resident_face_manifold() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut states = (0..17)
            .map(|index| body(index as f32 * 10.0, 0.0))
            .collect::<Vec<_>>();
        states[0].linear_velocity[0] = 1.0;
        states[1] = body(1.5, -1.0);
        let shapes = vec![
            GpuRigidShape::Box {
                half_extents: [1.0; 3]
            };
            17
        ];
        let mut world = GpuRigidPrimitiveWorld::new_primitives(
            context.device(),
            context.queue(),
            &states,
            &shapes,
            config(),
        )
        .unwrap();
        assert!(world.can_use_resident_step());
        assert_eq!(world.step(0.01).unwrap(), 1);
        let contacts = world.readback_contacts().unwrap();
        assert_eq!(contacts.pairs.len(), 1);
        assert!(contacts.pairs[0].1.is_contact());
        assert_eq!(contacts.pair_extra.len(), 1);
        assert!(
            contacts.pair_extra[0]
                .iter()
                .all(|contact| contact.is_contact())
        );
        let impulses = world.readback_contact_impulses().unwrap();
        assert_eq!(impulses.dt, Some(0.01));
        assert!(!impulses.contacts.is_empty());
        let actual = world.readback().unwrap();
        let mut sum = [0.0; 3];
        for contact in &impulses.contacts {
            assert_eq!((contact.body_a, contact.body_b), (Some(0), 1));
            assert!(contact.point_index < 4);
            for (axis, value) in contact.impulse_on_body_b().into_iter().enumerate() {
                sum[axis] += value;
            }
        }
        for (axis, value) in sum.into_iter().enumerate() {
            assert!(
                (value - (actual[1].linear_velocity[axis] - states[1].linear_velocity[axis])).abs()
                    < 1e-4
            );
        }
        world.reset(&states).unwrap();
        assert_eq!(world.readback_contact_impulses().unwrap().dt, None);
    }

    #[test]
    fn explicit_grouped_resident_step_keeps_local_contact_indices() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut first = (0..17)
            .map(|index| body(index as f32 * 10.0, 0.0))
            .collect::<Vec<_>>();
        let mut second = first.clone();
        first[0].linear_velocity[0] = 1.0;
        first[1] = body(1.5, -1.0);
        second[0].linear_velocity[0] = 1.0;
        second[1] = body(1.5, -1.0);
        let radii = [1.0; 17];
        let mut batch = GpuRigidSphereBatch::new(
            context.device(),
            context.queue(),
            &[
                GpuRigidSphereEnvironment {
                    states: &first,
                    radii: &radii,
                },
                GpuRigidSphereEnvironment {
                    states: &second,
                    radii: &radii,
                },
            ],
            config(),
        )
        .unwrap();
        let _result = batch.world_mut().step_gpu_resident(0.01).unwrap();
        assert_eq!(batch.world().candidate_pair_count(), 2);
        for environment in 0..2 {
            let contacts = batch.readback_contacts_environment(environment).unwrap();
            assert_eq!(contacts.pairs.len(), 1);
            assert_eq!((contacts.pairs[0].0.a, contacts.pairs[0].0.b), (0, 1));
            assert!(contacts.pairs[0].1.is_contact());
            let impulses = batch
                .readback_contact_impulses_environment(environment)
                .unwrap();
            assert_eq!(impulses.dt, Some(0.01));
            assert!(!impulses.contacts.is_empty());
            assert!(
                impulses
                    .contacts
                    .iter()
                    .all(|contact| contact.body_a == Some(0) && contact.body_b == 1)
            );
        }
    }

    #[test]
    fn grouped_step_automatically_uses_resident_candidates() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut first = (0..17)
            .map(|index| body(index as f32 * 10.0, 0.0))
            .collect::<Vec<_>>();
        first[1] = body(1.5, 0.0);
        let mut second = first.clone();
        second[1] = body(25.0, 0.0);
        let radii = [1.0; 17];
        let mut batch = GpuRigidSphereBatch::new(
            context.device(),
            context.queue(),
            &[
                GpuRigidSphereEnvironment {
                    states: &first,
                    radii: &radii,
                },
                GpuRigidSphereEnvironment {
                    states: &second,
                    radii: &radii,
                },
            ],
            config(),
        )
        .unwrap();
        assert!(batch.world().can_use_resident_step());
        assert_eq!(batch.step(0.01).unwrap(), 1);
        assert!(batch.world().last_resident_contacts.is_some());
        let first_contacts = batch.readback_contacts_environment(0).unwrap();
        assert_eq!(first_contacts.pairs.len(), 1);
        assert_eq!(
            (first_contacts.pairs[0].0.a, first_contacts.pairs[0].0.b),
            (0, 1)
        );
        assert!(first_contacts.pairs[0].1.is_contact());
        assert!(
            batch
                .readback_contacts_environment(1)
                .unwrap()
                .pairs
                .is_empty()
        );

        second[1] = body(1.5, 0.0);
        batch.reset_environment(1, &second).unwrap();
        assert_eq!(batch.step_substeps(0.01, 2).unwrap(), 2);
        assert!(batch.world().last_resident_contacts.is_some());
        for environment in 0..2 {
            let contacts = batch.readback_contacts_environment(environment).unwrap();
            assert_eq!(contacts.pairs.len(), 1);
            assert_eq!((contacts.pairs[0].0.a, contacts.pairs[0].0.b), (0, 1));
            assert!(contacts.pairs[0].1.is_contact());
        }
    }

    #[test]
    fn grouped_primitive_step_uses_resident_candidates() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut states = (0..17)
            .map(|index| body(index as f32 * 10.0, 0.0))
            .collect::<Vec<_>>();
        states[1] = body(1.5, 0.0);
        let mut shapes = vec![GpuRigidShape::Sphere { radius: 1.0 }; 17];
        shapes[0] = GpuRigidShape::Box {
            half_extents: [1.0; 3],
        };
        let environments = [
            GpuRigidPrimitiveEnvironment {
                states: &states,
                shapes: &shapes,
            },
            GpuRigidPrimitiveEnvironment {
                states: &states,
                shapes: &shapes,
            },
        ];
        let mut batch = GpuRigidPrimitiveBatch::new_primitives(
            context.device(),
            context.queue(),
            &environments,
            config(),
        )
        .unwrap();
        assert!(batch.world().can_use_resident_step());
        assert_eq!(batch.step(0.01).unwrap(), 2);
        assert!(batch.world().last_resident_contacts.is_some());
        for environment in 0..2 {
            let contacts = batch.readback_contacts_environment(environment).unwrap();
            assert_eq!(contacts.pairs.len(), 1);
            assert_eq!((contacts.pairs[0].0.a, contacts.pairs[0].0.b), (0, 1));
            assert!(contacts.pairs[0].1.is_contact());
            assert_eq!(contacts.pair_extra.len(), 1);
        }
    }

    #[test]
    fn grouped_sphere_topology_edits_preserve_other_environment() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let first = (0..16)
            .map(|index| body(index as f32 * 10.0, 0.0))
            .collect::<Vec<_>>();
        let second = [body(0.0, 0.0), body(1.5, 0.0)];
        let mut batch = GpuRigidSphereBatch::new(
            context.device(),
            context.queue(),
            &[
                GpuRigidSphereEnvironment {
                    states: &first,
                    radii: &[1.0; 16],
                },
                GpuRigidSphereEnvironment {
                    states: &second,
                    radii: &[1.0; 2],
                },
            ],
            config(),
        )
        .unwrap();
        let material = ColliderMaterial::new(0.4, 0.2);
        let groups = GpuRigidCollisionGroups {
            memberships: 3,
            filter: 3,
        };
        batch.world_mut().set_body_material(16, material).unwrap();
        batch
            .set_body_collision_groups_environment(1, 0, groups)
            .unwrap();
        assert!(batch.world().uses_exhaustive_pairs());
        assert_eq!(
            batch
                .append_body_environment(0, body(1.5, 0.0), 1.0)
                .unwrap(),
            16
        );
        assert_eq!(batch.environment_range(0), Some(0..17));
        assert_eq!(batch.environment_range(1), Some(17..19));
        assert!(!batch.world().uses_exhaustive_pairs());
        assert_same_states(&batch.readback_environment(1).unwrap(), &second);
        assert_eq!(
            batch.world().body_material_override(17),
            Some(Some(material))
        );
        assert_eq!(batch.body_collision_groups_environment(1, 0), Some(groups));
        assert_eq!(batch.step(0.01).unwrap(), 2);
        for (environment, expected) in [(0, (0, 16)), (1, (0, 1))] {
            let contacts = batch.readback_contacts_environment(environment).unwrap();
            assert_eq!(contacts.pairs.len(), 1);
            assert_eq!((contacts.pairs[0].0.a, contacts.pairs[0].0.b), expected);
            assert!(contacts.pairs[0].1.is_contact());
        }
        let before = batch.readback_environment(0).unwrap();
        let untouched = batch.readback_environment(1).unwrap();
        let removed = batch.remove_body_environment(0, 0).unwrap();
        assert_same_states(&[removed], &before[..1]);
        assert_eq!(batch.environment_range(0), Some(0..16));
        assert_eq!(batch.environment_range(1), Some(16..18));
        assert!(batch.world().uses_exhaustive_pairs());
        assert_same_states(&batch.readback_environment(1).unwrap(), &untouched);
        assert_eq!(
            batch.world().body_material_override(16),
            Some(Some(material))
        );
        assert_eq!(batch.body_collision_groups_environment(1, 0), Some(groups));

        let _removed = batch.remove_body_environment(1, 1).unwrap();
        let _removed = batch.remove_body_environment(1, 0).unwrap();
        assert_eq!(batch.environment_range(1), Some(16..16));
        assert_eq!(
            batch
                .append_body_environment(1, body(30.0, 0.0), 1.0)
                .unwrap(),
            0
        );
        assert_eq!(batch.environment_range(1), Some(16..17));
    }

    #[test]
    fn grouped_primitive_topology_edits_preserve_shapes_and_indices() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let first = [body(0.0, 0.0), body(1.5, 0.0)];
        let second = [body(0.0, 0.0), body(1.4, 0.0)];
        let first_shapes = [
            GpuRigidShape::Box {
                half_extents: [1.0; 3],
            },
            GpuRigidShape::Sphere { radius: 1.0 },
        ];
        let second_shapes = [
            GpuRigidShape::Capsule {
                radius: 0.5,
                half_length: 0.5,
            },
            GpuRigidShape::Sphere { radius: 1.0 },
        ];
        let mut batch = GpuRigidPrimitiveBatch::new_primitives(
            context.device(),
            context.queue(),
            &[
                GpuRigidPrimitiveEnvironment {
                    states: &first,
                    shapes: &first_shapes,
                },
                GpuRigidPrimitiveEnvironment {
                    states: &second,
                    shapes: &second_shapes,
                },
            ],
            config(),
        )
        .unwrap();
        let added_shape = GpuRigidShape::Sphere { radius: 0.75 };
        assert_eq!(
            batch
                .append_primitive_environment(0, body(50.0, 0.0), added_shape.clone())
                .unwrap(),
            2
        );
        assert_eq!(batch.environment_range(1), Some(3..5));
        assert_eq!(batch.world().shape(2), Some(added_shape));
        assert_eq!(batch.world().shape(3), Some(second_shapes[0].clone()));
        assert_same_states(&batch.readback_environment(1).unwrap(), &second);
        assert_eq!(batch.step(0.01).unwrap(), 4);
        for (environment, candidates) in [(0, 3), (1, 1)] {
            let contacts = batch.readback_contacts_environment(environment).unwrap();
            assert_eq!(contacts.pairs.len(), candidates);
            assert!(
                contacts
                    .pairs
                    .iter()
                    .any(|(pair, contact)| { (pair.a, pair.b) == (0, 1) && contact.is_contact() })
            );
        }
        let untouched = batch.readback_environment(1).unwrap();
        let (removed, shape) = batch.remove_primitive_environment(0, 0).unwrap();
        assert_eq!(shape, first_shapes[0]);
        assert!(removed.is_valid());
        assert_eq!(batch.environment_range(1), Some(2..4));
        assert_eq!(batch.world().shape(2), Some(second_shapes[0].clone()));
        assert_same_states(&batch.readback_environment(1).unwrap(), &untouched);
        assert!(matches!(
            batch.append_body_environment(0, body(0.0, 0.0), 1.0),
            Err(GpuRigidSphereWorldError::InvalidInput)
        ));
        assert!(matches!(
            batch.remove_primitive_environment(0, 9),
            Err(GpuRigidSphereWorldError::InvalidInput)
        ));
        batch
            .set_ball_joints_environment(
                1,
                &[GpuRigidBallJoint {
                    body_a: 0,
                    body_b: 1,
                    local_anchor_a: [0.0; 3],
                    local_anchor_b: [0.0; 3],
                }],
            )
            .unwrap();
        assert_eq!(
            batch
                .append_primitive_environment(
                    0,
                    body(50.0, 0.0),
                    GpuRigidShape::Sphere { radius: 1.0 }
                )
                .unwrap(),
            2
        );
        assert_eq!(batch.environment_range(1), Some(3..5));
        assert_eq!(batch.ball_joints_environment(1).unwrap().len(), 1);
    }

    #[test]
    fn grouped_topology_edits_remap_all_joint_types_and_drives() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let states = [body(0.0, 0.0), body(1.0, 0.0)];
        let environments = [GpuRigidSphereEnvironment {
            states: &states,
            radii: &[0.1; 2],
        }; 4];
        let mut batch =
            GpuRigidSphereBatch::new(context.device(), context.queue(), &environments, config())
                .unwrap();
        let ball = GpuRigidBallJoint {
            body_a: 0,
            body_b: 1,
            local_anchor_a: [1.0, 0.0, 0.0],
            local_anchor_b: [0.0; 3],
        };
        let fixed = GpuRigidFixedJoint {
            body_a: 0,
            body_b: 1,
            local_anchor_a: [1.0, 0.0, 0.0],
            local_anchor_b: [0.0; 3],
            local_rotation_a: [0.0, 0.0, 0.0, 1.0],
            local_rotation_b: [0.0, 0.0, 0.0, 1.0],
        };
        let revolute = GpuRigidRevoluteJoint {
            body_a: 0,
            body_b: 1,
            local_anchor_a: [1.0, 0.0, 0.0],
            local_anchor_b: [0.0; 3],
            local_axis_a: [0.0, 0.0, 1.0],
            local_axis_b: [0.0, 0.0, 1.0],
        };
        let prismatic = GpuRigidPrismaticJoint {
            body_a: 0,
            body_b: 1,
            local_anchor_a: [0.0; 3],
            local_anchor_b: [0.0; 3],
            local_rotation_a: [0.0, 0.0, 0.0, 1.0],
            local_rotation_b: [0.0, 0.0, 0.0, 1.0],
        };
        batch.set_ball_joints_environment(0, &[ball]).unwrap();
        batch.set_fixed_joints_environment(1, &[fixed]).unwrap();
        batch
            .set_revolute_joints_environment(2, &[revolute])
            .unwrap();
        batch
            .set_prismatic_joints_environment(3, &[prismatic])
            .unwrap();
        let servo = GpuRigidAxisServo {
            position_target: 0.2,
            velocity_target: 0.1,
            stiffness: 2.0,
            damping: 1.0,
            max_force: 3.0,
        };
        let motor = GpuRigidAxisMotor {
            target_velocity: 0.3,
            max_force: 2.0,
        };
        let revolute_limit = GpuRigidRevoluteLimit {
            min: -0.5,
            max: 0.5,
        };
        let prismatic_limit = GpuRigidPrismaticLimit {
            min: -0.25,
            max: 0.25,
        };
        batch
            .set_revolute_servo_environment(2, 0, Some(servo))
            .unwrap();
        batch
            .set_revolute_limit_environment(2, 0, Some(revolute_limit))
            .unwrap();
        batch
            .set_prismatic_motor_environment(3, 0, Some(motor))
            .unwrap();
        batch
            .set_prismatic_limit_environment(3, 0, Some(prismatic_limit))
            .unwrap();
        assert_eq!(
            batch
                .append_body_environment(0, body(50.0, 0.0), 0.1)
                .unwrap(),
            2
        );
        assert_eq!(batch.environment_range(3), Some(7..9));
        assert_eq!(batch.ball_joints_environment(0).unwrap(), vec![ball]);
        assert_eq!(batch.fixed_joints_environment(1).unwrap(), vec![fixed]);
        assert_eq!(
            batch.revolute_joints_environment(2).unwrap(),
            vec![revolute]
        );
        assert_eq!(
            batch.prismatic_joints_environment(3).unwrap(),
            vec![prismatic]
        );
        assert_eq!(batch.revolute_servo_environment(2, 0).unwrap(), Some(servo));
        assert_eq!(
            batch.revolute_limit_environment(2, 0).unwrap(),
            Some(revolute_limit)
        );
        assert_eq!(
            batch.prismatic_motor_environment(3, 0).unwrap(),
            Some(motor)
        );
        assert_eq!(
            batch.prismatic_limit_environment(3, 0).unwrap(),
            Some(prismatic_limit)
        );
        assert_eq!(batch.step(0.01).unwrap(), 6);

        let _removed = batch.remove_body_environment(0, 1).unwrap();
        assert!(batch.ball_joints_environment(0).unwrap().is_empty());
        assert_eq!(batch.environment_range(3), Some(6..8));
        assert_eq!(batch.fixed_joints_environment(1).unwrap(), vec![fixed]);
        assert_eq!(
            batch.revolute_joints_environment(2).unwrap(),
            vec![revolute]
        );
        assert_eq!(
            batch.prismatic_joints_environment(3).unwrap(),
            vec![prismatic]
        );
        assert_eq!(batch.revolute_servo_environment(2, 0).unwrap(), Some(servo));
        assert_eq!(
            batch.prismatic_motor_environment(3, 0).unwrap(),
            Some(motor)
        );
        assert_eq!(batch.step(0.01).unwrap(), 4);
    }

    #[test]
    fn grouped_topology_edits_keep_revolute_turn_history() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut fixed = body(0.0, 0.0);
        fixed.position_inverse_mass[3] = 0.0;
        fixed.inverse_inertia_sleep = [0.0; 4];
        let moving = body(1.0, 0.0);
        let mut options = config();
        options.sleep.enabled = false;
        options.solve.bias_factor = 0.2;
        options.solve.iterations = 16;
        let mut batch = GpuRigidSphereBatch::new(
            context.device(),
            context.queue(),
            &[
                GpuRigidSphereEnvironment {
                    states: &[body(100.0, 0.0)],
                    radii: &[0.1],
                },
                GpuRigidSphereEnvironment {
                    states: &[fixed, moving],
                    radii: &[0.1; 2],
                },
            ],
            options,
        )
        .unwrap();
        batch
            .set_revolute_joints_environment(
                1,
                &[GpuRigidRevoluteJoint {
                    body_a: 0,
                    body_b: 1,
                    local_anchor_a: [0.0; 3],
                    local_anchor_b: [-1.0, 0.0, 0.0],
                    local_axis_a: [0.0, 0.0, 1.0],
                    local_axis_b: [0.0, 0.0, 1.0],
                }],
            )
            .unwrap();
        let limit = GpuRigidRevoluteLimit {
            min: 0.0,
            max: core::f32::consts::TAU + 0.3,
        };
        let motor = GpuRigidAxisMotor {
            target_velocity: 3.0,
            max_force: 30.0,
        };
        batch
            .set_revolute_limit_environment(1, 0, Some(limit))
            .unwrap();
        batch
            .set_revolute_motor_environment(1, 0, Some(motor))
            .unwrap();
        for _ in 0..10 {
            let _candidates = batch.step_substeps(0.01, 32).unwrap();
        }
        let before = batch.readback_revolute_angle_environment(1, 0).unwrap();
        assert!(before > core::f32::consts::TAU, "{before}");
        assert_eq!(
            batch
                .append_body_environment(0, body(150.0, 0.0), 0.1)
                .unwrap(),
            1
        );
        assert_eq!(batch.environment_range(1), Some(2..4));
        assert_eq!(batch.revolute_motor_environment(1, 0).unwrap(), Some(motor));
        assert_eq!(batch.revolute_limit_environment(1, 0).unwrap(), Some(limit));
        let after_insert = batch.readback_revolute_angle_environment(1, 0).unwrap();
        assert!(
            (after_insert - before).abs() < 1e-4,
            "{before} {after_insert}"
        );
        let _removed = batch.remove_body_environment(0, 1).unwrap();
        let after_remove = batch.readback_revolute_angle_environment(1, 0).unwrap();
        assert!(
            (after_remove - before).abs() < 1e-4,
            "{before} {after_remove}"
        );
        let _removed = batch.remove_body_environment(1, 1).unwrap();
        assert!(batch.revolute_joints_environment(1).unwrap().is_empty());
    }

    #[test]
    fn single_world_topology_edits_remap_surviving_joint() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut world = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &[body(50.0, 0.0), body(0.0, 0.0), body(1.0, 0.0)],
            &[0.1; 3],
            config(),
        )
        .unwrap();
        world
            .set_ball_joints(&[GpuRigidBallJoint {
                body_a: 1,
                body_b: 2,
                local_anchor_a: [1.0, 0.0, 0.0],
                local_anchor_b: [0.0; 3],
            }])
            .unwrap();
        let _removed = world.remove_body(0).unwrap();
        assert_eq!(world.len(), 2);
        assert_eq!(
            (world.ball_joints()[0].body_a, world.ball_joints()[0].body_b),
            (0, 1)
        );
        assert_eq!(world.append_body(body(100.0, 0.0), 0.1).unwrap(), 2);
        assert_eq!(
            (world.ball_joints()[0].body_a, world.ball_joints()[0].body_b),
            (0, 1)
        );
        assert_eq!(world.step(0.01).unwrap(), 3);
    }

    #[test]
    fn resident_lbvh_world_sleep_uses_candidate_contacts() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut states = (0..17)
            .map(|index| body(index as f32 * 10.0, 0.0))
            .collect::<Vec<_>>();
        states[0].position_inverse_mass[2] = 1.0;
        let mut settings = config();
        settings.ground_half_extent = Some(100.0);
        let mut world = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &states,
            &[1.0; 17],
            settings,
        )
        .unwrap();
        for _ in 0..3 {
            let _result = world.step_gpu_resident(0.01).unwrap();
        }
        let actual = world.readback().unwrap();
        assert_eq!(actual[0].inverse_inertia_sleep[3], 1.0);
        assert_eq!(actual[1].inverse_inertia_sleep[3], 0.0);
    }

    #[test]
    fn resident_lbvh_world_solves_box_manifold() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let states = (0..17)
            .map(|index| {
                if index == 1 {
                    body(1.5, -1.0)
                } else {
                    body(index as f32 * 10.0, if index == 0 { 1.0 } else { 0.0 })
                }
            })
            .collect::<Vec<_>>();
        let shapes = vec![
            GpuRigidShape::Box {
                half_extents: [1.0; 3],
            };
            17
        ];
        let mut settings = config();
        settings.solve.restitution = 1.0;
        let mut world = GpuRigidPrimitiveWorld::new_primitives(
            context.device(),
            context.queue(),
            &states,
            &shapes,
            settings,
        )
        .unwrap();
        let (candidates, output) = world.step_gpu_resident(0.01).unwrap();
        assert_eq!(output.pair_stride, 4);
        let readback = output
            .readback_pairs(context.device(), context.queue(), &candidates)
            .unwrap();
        assert_eq!(readback.pairs.len(), 1);
        assert!(readback.pairs[0].1.is_contact());
        assert!(
            readback.pair_extra[0]
                .iter()
                .all(|contact| contact.is_contact())
        );
        let actual = world.readback().unwrap();
        assert!(actual[0].linear_velocity[0] < 1.0);
        assert!(actual[1].linear_velocity[0] > -1.0);
    }

    #[test]
    fn resident_lbvh_world_capacity_faults_until_reset() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let states = (0..70)
            .map(|index| body(index as f32 * 1.9, 0.0))
            .collect::<Vec<_>>();
        let mut settings = config();
        settings.solve.iterations = 400;
        let mut world = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &states,
            &[1.0; 70],
            settings,
        )
        .unwrap();
        assert!(matches!(
            world.step_gpu_resident(0.01),
            Err(GpuRigidSphereWorldError::Solver(
                GpuRigidSphereSolveError::Capacity
            ))
        ));
        assert!(world.is_faulted());
        assert!(matches!(
            world.step_gpu_resident(0.01),
            Err(GpuRigidSphereWorldError::Faulted)
        ));
        world.reset(&states).unwrap();
        assert!(!world.is_faulted());
    }

    #[test]
    fn resident_lbvh_cached_impulses_reduce_one_iteration_stack_error() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut states = (0..17)
            .map(|index| body(index as f32 * 10.0, 0.0))
            .collect::<Vec<_>>();
        states[0].position_inverse_mass[2] = 1.0;
        states[1].position_inverse_mass = [0.0, 0.0, 3.0, 1.0];
        let mut settings = config();
        settings.gravity = [0.0, 0.0, -9.81];
        settings.ground_half_extent = Some(5.0);
        settings.solve.iterations = 1;
        settings.solve.bias_factor = 0.2;
        settings.sleep.enabled = false;
        let simulate = |cached: bool| {
            let mut world = GpuRigidSphereWorld::new(
                context.device(),
                context.queue(),
                &states,
                &[1.0; 17],
                settings,
            )
            .unwrap();
            for _ in 0..20 {
                if !cached {
                    world.resident_impulse_cache.clear();
                }
                let _result = world.step_gpu_resident(0.01).unwrap();
            }
            world.readback().unwrap()
        };
        let warm = simulate(true);
        let cold = simulate(false);
        let warm_error = (warm[0].position_inverse_mass[2] - 1.0).abs()
            + (warm[1].position_inverse_mass[2] - 3.0).abs();
        let cold_error = (cold[0].position_inverse_mass[2] - 1.0).abs()
            + (cold[1].position_inverse_mass[2] - 3.0).abs();
        assert!(
            warm_error < cold_error,
            "warm={warm_error}, cold={cold_error}"
        );
    }

    #[test]
    fn resident_lbvh_pair_order_change_preserves_independent_stack() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut states = (0..17)
            .map(|index| body(index as f32 * 20.0, 0.0))
            .collect::<Vec<_>>();
        states[0] = body(100.0, 5.0);
        states[1] = body(103.0, -5.0);
        states[3].position_inverse_mass = [0.0, 0.0, 1.0, 1.0];
        states[4].position_inverse_mass = [0.0, 0.0, 3.0, 1.0];
        let mut settings = config();
        settings.gravity = [0.0, 0.0, -9.81];
        settings.ground_half_extent = Some(5.0);
        settings.solve.iterations = 1;
        settings.solve.bias_factor = 0.2;
        settings.sleep.enabled = false;
        let create = || {
            GpuRigidSphereWorld::new(
                context.device(),
                context.queue(),
                &states,
                &[1.0; 17],
                settings,
            )
            .unwrap()
        };
        let mut changing = create();
        let mut baseline = create();
        baseline
            .set_body_collision_groups(
                0,
                GpuRigidCollisionGroups {
                    memberships: 1,
                    filter: 1,
                },
            )
            .unwrap();
        baseline
            .set_body_collision_groups(
                1,
                GpuRigidCollisionGroups {
                    memberships: 2,
                    filter: 2,
                },
            )
            .unwrap();
        let mut saw_new_pair = false;
        for _ in 0..20 {
            let (candidates, output) = changing.step_gpu_resident(0.01).unwrap();
            let pairs = output
                .readback_pairs(context.device(), context.queue(), &candidates)
                .unwrap();
            saw_new_pair |= pairs
                .pairs
                .iter()
                .any(|(pair, _)| pair.a == 0 && pair.b == 1);
            let _baseline = baseline.step_gpu_resident(0.01).unwrap();
        }
        assert!(saw_new_pair);
        let actual = changing.readback().unwrap();
        let expected = baseline.readback().unwrap();
        for body_index in [3, 4] {
            assert!(
                (actual[body_index].position_inverse_mass[2]
                    - expected[body_index].position_inverse_mass[2])
                    .abs()
                    < 1e-4
            );
            assert!(
                (actual[body_index].linear_velocity[2] - expected[body_index].linear_velocity[2])
                    .abs()
                    < 1e-4
            );
        }
    }

    #[test]
    fn resident_lbvh_reappearing_pair_discards_stale_impulse() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut states = (0..17)
            .map(|index| body(index as f32 * 10.0, 0.0))
            .collect::<Vec<_>>();
        states[1] = body(1.5, 0.0);
        let mut settings = config();
        settings.solve.bias_factor = 0.2;
        settings.sleep.enabled = false;
        let create = || {
            GpuRigidSphereWorld::new(
                context.device(),
                context.queue(),
                &states,
                &[1.0; 17],
                settings,
            )
            .unwrap()
        };
        let mut cached = create();
        let mut cold = create();
        for world in [&mut cached, &mut cold] {
            let _first = world.step_gpu_resident(0.01).unwrap();
            assert!(world.readback().unwrap()[1].linear_velocity[0] > 0.0);
            world
                .state
                .write_body(context.queue(), 0, body(0.0, 0.0))
                .unwrap();
            world
                .state
                .write_body(context.queue(), 1, body(10.0, 0.0))
                .unwrap();
            let _absent = world.step_gpu_resident(0.01).unwrap();
            world
                .state
                .write_body(context.queue(), 0, body(0.0, 0.0))
                .unwrap();
            world
                .state
                .write_body(context.queue(), 1, body(1.5, 0.0))
                .unwrap();
        }
        cold.resident_impulse_cache.clear();
        let _cached = cached.step_gpu_resident(0.01).unwrap();
        let _cold = cold.step_gpu_resident(0.01).unwrap();
        let warm_states = cached.readback().unwrap();
        let cold_states = cold.readback().unwrap();
        for index in [0, 1] {
            assert!(
                (warm_states[index].linear_velocity[0] - cold_states[index].linear_velocity[0])
                    .abs()
                    < 1e-5
            );
        }
    }

    fn assert_same_states(left: &[GpuRigidBodyState], right: &[GpuRigidBodyState]) {
        assert_eq!(left.len(), right.len());
        for (a, b) in left.iter().zip(right) {
            assert_eq!(a.position_inverse_mass, b.position_inverse_mass);
            assert_eq!(a.orientation, b.orientation);
            assert_eq!(a.linear_velocity, b.linear_velocity);
            assert_eq!(a.angular_velocity, b.angular_velocity);
            assert_eq!(a.inverse_inertia_sleep, b.inverse_inertia_sleep);
        }
    }

    #[test]
    fn world_step_finds_contacts_solves_and_sleeps() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut world = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &[body(-1.0, 1.0), body(1.0, -1.0)],
            &[1.0; 2],
            config(),
        )
        .unwrap();
        assert_eq!(world.step(0.01).unwrap(), 1);
        let actual = world.readback().unwrap();
        assert!(
            actual
                .iter()
                .all(|body| body.linear_velocity[0].abs() < 1e-4)
        );
        assert_eq!(world.step(0.01).unwrap(), 1);
        assert!(
            world
                .readback()
                .unwrap()
                .iter()
                .all(|body| body.inverse_inertia_sleep[3] == 1.0)
        );
        world
            .set_sleep_settings(SleepSettings {
                enabled: false,
                ..world.config().sleep
            })
            .unwrap();
        assert!(
            world
                .readback()
                .unwrap()
                .iter()
                .all(|body| body.inverse_inertia_sleep[3] == 0.0)
        );
    }

    #[test]
    fn two_sphere_stack_sleeps_and_stays_supported() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut lower = body(0.0, 0.0);
        lower.position_inverse_mass[2] = 1.0;
        let mut upper = body(0.0, 0.0);
        upper.position_inverse_mass[2] = 3.2;
        let mut world = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &[lower, upper],
            &[1.0; 2],
            GpuRigidSphereWorldConfig::default(),
        )
        .unwrap();
        for _ in 0..240 {
            let _candidate_count = world.step(1.0 / 120.0).unwrap();
        }
        let actual = world.readback().unwrap();
        assert!((actual[0].position_inverse_mass[2] - 1.0).abs() < 0.02);
        assert!((actual[1].position_inverse_mass[2] - 3.0).abs() < 0.02);
        assert!(
            actual
                .iter()
                .all(|body| body.inverse_inertia_sleep[3] == 1.0)
        );
        assert!(actual.iter().all(|body| body.linear_velocity[2] == 0.0));
        for _ in 0..240 {
            let _candidate_count = world.step(1.0 / 120.0).unwrap();
        }
        assert!(
            world
                .readback()
                .unwrap()
                .iter()
                .all(|body| body.inverse_inertia_sleep[3] == 1.0)
        );
    }

    #[test]
    fn batched_exhaustive_substeps_match_sequential_steps() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut lower = body(0.0, 0.0);
        lower.position_inverse_mass[2] = 1.0;
        let mut upper = body(0.0, 0.0);
        upper.position_inverse_mass[2] = 3.2;
        let initial = [lower, upper];
        let make_world = || {
            GpuRigidSphereWorld::new(
                context.device(),
                context.queue(),
                &initial,
                &[1.0; 2],
                GpuRigidSphereWorldConfig::default(),
            )
            .unwrap()
        };
        let mut batched = make_world();
        let mut sequential = make_world();
        assert!(matches!(
            batched.step_substeps(1.0 / 120.0, 0),
            Err(GpuRigidSphereWorldError::InvalidInput)
        ));
        assert_eq!(batched.readback().unwrap()[1].position_inverse_mass[2], 3.2);
        assert_eq!(batched.step_substeps(1.0 / 120.0, 120).unwrap(), 1);
        for _ in 0..120 {
            let _candidate_count = sequential.step(1.0 / 120.0).unwrap();
        }
        let left = batched.readback().unwrap();
        let right = sequential.readback().unwrap();
        for (a, b) in left.iter().zip(right) {
            assert!((a.position_inverse_mass[2] - b.position_inverse_mass[2]).abs() < 1e-5);
            assert!((a.linear_velocity[2] - b.linear_velocity[2]).abs() < 1e-5);
            assert_eq!(a.inverse_inertia_sleep[3], b.inverse_inertia_sleep[3]);
        }
    }

    #[test]
    fn packed_environments_keep_overlapping_scenes_isolated() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let initial = [body(-1.0, 1.0), body(1.0, -1.0)];
        let environments = [
            GpuRigidSphereEnvironment {
                states: &initial,
                radii: &[1.0; 2],
            },
            GpuRigidSphereEnvironment {
                states: &initial,
                radii: &[1.0; 2],
            },
        ];
        let mut batch =
            GpuRigidSphereBatch::new(context.device(), context.queue(), &environments, config())
                .unwrap();
        let mut reference = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &initial,
            &[1.0; 2],
            config(),
        )
        .unwrap();
        assert!(batch.world().uses_exhaustive_pairs());
        assert_eq!(batch.world().candidate_pair_count(), 2);
        assert_eq!(batch.step_substeps(0.01, 4).unwrap(), 2);
        assert_eq!(reference.step_substeps(0.01, 4).unwrap(), 1);
        for environment in 0..2 {
            let contacts = batch.readback_contacts_environment(environment).unwrap();
            assert_eq!(contacts.pairs.len(), 1);
            assert_eq!((contacts.pairs[0].0.a, contacts.pairs[0].0.b), (0, 1));
            assert!(contacts.pair_extra.is_empty());
            assert!(contacts.ground.is_empty());
        }
        let expected = reference.readback().unwrap();
        for environment in 0..2 {
            let actual = batch.readback_environment(environment).unwrap();
            for (left, right) in actual.iter().zip(&expected) {
                assert!(
                    (left.position_inverse_mass[0] - right.position_inverse_mass[0]).abs() < 1e-5
                );
                assert!((left.linear_velocity[0] - right.linear_velocity[0]).abs() < 1e-5);
            }
        }
        let untouched = batch.readback_environment(1).unwrap();
        let reset = [body(-5.0, 0.0), body(5.0, 0.0)];
        batch.reset_environment(0, &reset).unwrap();
        assert_same_states(&batch.readback_environment(1).unwrap(), &untouched);
        assert_same_states(&batch.readback_environment(0).unwrap(), &reset);
    }

    #[test]
    fn packed_primitive_environments_keep_shapes_contacts_and_resets_isolated() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let box_states = [body(0.0, 0.0), body(1.4, 0.0)];
        let capsule_states = [body(0.0, 0.0), body(0.9, 0.0)];
        let box_shapes = [
            GpuRigidShape::Box {
                half_extents: [1.0; 3],
            },
            GpuRigidShape::Sphere { radius: 0.5 },
        ];
        let capsule_shapes = [
            GpuRigidShape::Capsule {
                radius: 0.5,
                half_length: 0.5,
            },
            GpuRigidShape::Sphere { radius: 0.5 },
        ];
        let environments = [
            GpuRigidPrimitiveEnvironment {
                states: &box_states,
                shapes: &box_shapes,
            },
            GpuRigidPrimitiveEnvironment {
                states: &capsule_states,
                shapes: &capsule_shapes,
            },
        ];
        let mut batch = GpuRigidPrimitiveBatch::new_primitives(
            context.device(),
            context.queue(),
            &environments,
            config(),
        )
        .unwrap();
        assert_eq!(batch.len(), 2);
        assert_eq!(batch.environment_range(1), Some(2..4));
        assert_eq!(batch.world().candidate_pair_count(), 2);
        assert_eq!(batch.world().shape(2), Some(capsule_shapes[0].clone()));
        assert_eq!(batch.step(0.01).unwrap(), 2);
        let full_contacts = batch.world().readback_contacts().unwrap();
        for environment in 0..2 {
            let contacts = batch.readback_contacts_environment(environment).unwrap();
            assert_eq!(contacts.pairs.len(), 1);
            assert_eq!((contacts.pairs[0].0.a, contacts.pairs[0].0.b), (0, 1));
            assert!(contacts.pairs[0].1.is_contact());
            assert_eq!(contacts.pair_extra.len(), 1);
            assert!(contacts.ground.is_empty());
            assert_eq!(
                bytemuck::bytes_of(&contacts.pairs[0].1),
                bytemuck::bytes_of(&full_contacts.pairs[environment].1)
            );
            assert_eq!(
                bytemuck::bytes_of(&contacts.pair_extra[0]),
                bytemuck::bytes_of(&full_contacts.pair_extra[environment])
            );
        }
        assert_eq!(full_contacts.pairs.len(), 2);
        assert_eq!(
            (full_contacts.pairs[0].0.a, full_contacts.pairs[0].0.b),
            (0, 1)
        );
        assert_eq!(
            (full_contacts.pairs[1].0.a, full_contacts.pairs[1].0.b),
            (2, 3)
        );
        assert!(
            full_contacts
                .pairs
                .iter()
                .all(|(_, contact)| contact.is_contact())
        );

        let untouched = batch.readback_environment(1).unwrap();
        batch.reset_environment(0, &box_states).unwrap();
        assert_same_states(&batch.readback_environment(0).unwrap(), &box_states);
        assert_same_states(&batch.readback_environment(1).unwrap(), &untouched);
        assert!(matches!(
            GpuRigidPrimitiveBatch::new_primitives(
                context.device(),
                context.queue(),
                &[GpuRigidPrimitiveEnvironment {
                    states: &box_states,
                    shapes: &box_shapes[..1],
                }],
                config(),
            ),
            Err(GpuRigidSphereWorldError::InvalidInput)
        ));
    }

    #[test]
    fn resident_ball_joint_keeps_rotating_local_anchors_together() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut fixed = body(0.0, 0.0);
        fixed.position_inverse_mass[3] = 0.0;
        fixed.inverse_inertia_sleep = [0.0; 4];
        let moving = body(1.0, 0.0);
        let mut options = config();
        options.sleep.enabled = false;
        let mut world = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &[fixed, moving],
            &[0.1, 0.1],
            options,
        )
        .unwrap();
        let joint = GpuRigidBallJoint {
            body_a: 0,
            body_b: 1,
            local_anchor_a: [0.0; 3],
            local_anchor_b: [-1.0, 0.0, 0.0],
        };
        world.set_ball_joints(&[joint]).unwrap();
        assert_eq!(world.ball_joints(), &[joint]);
        let before_edit = world.readback().unwrap();
        assert_eq!(world.append_body(body(2.0, 0.0), 0.1).unwrap(), 2);
        assert_eq!(world.ball_joints(), &[joint]);
        assert_same_states(&world.readback().unwrap()[..2], &before_edit);
        let removed = world.remove_body(2).unwrap();
        assert_eq!(
            removed.position_inverse_mass,
            body(2.0, 0.0).position_inverse_mass
        );
        assert_eq!(world.ball_joints(), &[joint]);
        assert_same_states(&world.readback().unwrap(), &before_edit);
        for _ in 0..120 {
            world
                .write_forces(
                    1,
                    GpuRigidBodyForces {
                        force: [0.0; 4],
                        torque: [0.0, 0.0, 1.0, 0.0],
                    },
                )
                .unwrap();
            let _candidates = world.step(0.005).unwrap();
        }
        let actual = world.readback().unwrap();
        let moving = actual[1];
        let q = moving.orientation;
        let rotated_anchor = [
            -1.0 + 2.0 * (q[1] * q[1] + q[2] * q[2]),
            -2.0 * (q[0] * q[1] + q[2] * q[3]),
            -2.0 * (q[0] * q[2] - q[1] * q[3]),
        ];
        let error = (0..3)
            .map(|axis| {
                let difference = moving.position_inverse_mass[axis] + rotated_anchor[axis]
                    - fixed.position_inverse_mass[axis];
                difference * difference
            })
            .sum::<f32>()
            .sqrt();
        assert!(error < 0.05, "anchor drift: {error}, state: {moving:?}");
        assert!(moving.position_inverse_mass[1].abs() > 0.05);
        world.set_ball_joints(&[]).unwrap();
        assert!(world.ball_joints().is_empty());
        assert_eq!(world.append_body(body(2.0, 0.0), 0.1).unwrap(), 2);
    }

    #[test]
    fn resident_fixed_joint_resists_torque_and_force() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut fixed = body(0.0, 0.0);
        fixed.position_inverse_mass[3] = 0.0;
        fixed.inverse_inertia_sleep = [0.0; 4];
        let mut options = config();
        options.sleep.enabled = false;
        options.solve.bias_factor = 0.2;
        options.solve.iterations = 12;
        let mut world = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &[fixed, body(1.0, 0.0)],
            &[0.1; 2],
            options,
        )
        .unwrap();
        let joint = GpuRigidFixedJoint {
            body_a: 0,
            body_b: 1,
            local_anchor_a: [1.0, 0.0, 0.0],
            local_anchor_b: [0.0; 3],
            local_rotation_a: [0.0, 0.0, 0.0, 1.0],
            local_rotation_b: [0.0, 0.0, 0.0, 1.0],
        };
        world.set_fixed_joints(&[joint]).unwrap();
        assert_eq!(world.fixed_joints(), &[joint]);
        let before_edit = world.readback().unwrap();
        assert_eq!(world.append_body(body(3.0, 0.0), 0.1).unwrap(), 2);
        assert_eq!(world.fixed_joints(), &[joint]);
        assert_same_states(&world.readback().unwrap()[..2], &before_edit);
        let removed = world.remove_body(2).unwrap();
        assert_eq!(
            removed.position_inverse_mass,
            body(3.0, 0.0).position_inverse_mass
        );
        assert_eq!(world.fixed_joints(), &[joint]);
        assert_same_states(&world.readback().unwrap(), &before_edit);
        for _ in 0..120 {
            world
                .write_forces(
                    1,
                    GpuRigidBodyForces {
                        force: [0.0, 10.0, 0.0, 0.0],
                        torque: [0.0, 0.0, 5.0, 0.0],
                    },
                )
                .unwrap();
            let _candidates = world.step(0.005).unwrap();
        }
        let state = world.readback().unwrap()[1];
        assert!(
            (state.position_inverse_mass[0] - 1.0).abs() < 0.05,
            "{state:?}"
        );
        assert!(state.position_inverse_mass[1].abs() < 0.05, "{state:?}");
        assert!(state.orientation[2].abs() < 0.05, "{state:?}");
        world.set_fixed_joints(&[]).unwrap();
        assert!(world.fixed_joints().is_empty());
        assert_eq!(world.append_body(body(3.0, 0.0), 0.1).unwrap(), 2);
    }

    #[test]
    fn resident_fixed_joint_rejects_invalid_frame_without_losing_topology() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut world = GpuRigidSphereWorld::new_grouped(
            context.device(),
            context.queue(),
            &[body(0.0, 0.0), body(1.0, 0.0)],
            &[0.1; 2],
            &[0, 1],
            config(),
        )
        .unwrap();
        let joint = GpuRigidFixedJoint {
            body_a: 0,
            body_b: 1,
            local_anchor_a: [0.0; 3],
            local_anchor_b: [0.0; 3],
            local_rotation_a: [0.0, 0.0, 0.0, 1.0],
            local_rotation_b: [0.0, 0.0, 0.0, 1.0],
        };
        assert!(world.set_fixed_joints(&[joint]).is_err());
        assert!(world.fixed_joints().is_empty());
        let mut invalid = joint;
        invalid.body_b = 0;
        assert!(world.set_fixed_joints(&[invalid]).is_err());
        invalid.body_b = 1;
        invalid.local_rotation_b = [0.0; 4];
        assert!(world.set_fixed_joints(&[invalid]).is_err());
    }

    #[test]
    fn resident_fixed_joint_preserves_rotated_local_frame() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut fixed = body(0.0, 0.0);
        fixed.position_inverse_mass[3] = 0.0;
        fixed.inverse_inertia_sleep = [0.0; 4];
        let mut moving = body(1.0, 0.0);
        let half = 0.5f32.sqrt();
        moving.orientation = [0.0, 0.0, half, half];
        let mut options = config();
        options.sleep.enabled = false;
        options.solve.bias_factor = 0.2;
        let mut world = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &[fixed, moving],
            &[0.1; 2],
            options,
        )
        .unwrap();
        world
            .set_fixed_joints(&[GpuRigidFixedJoint {
                body_a: 0,
                body_b: 1,
                local_anchor_a: [1.0, 0.0, 0.0],
                local_anchor_b: [0.0; 3],
                local_rotation_a: [0.0, 0.0, 0.0, 1.0],
                local_rotation_b: [0.0, 0.0, -half, half],
            }])
            .unwrap();
        for _ in 0..80 {
            world
                .write_forces(
                    1,
                    GpuRigidBodyForces {
                        force: [0.0; 4],
                        torque: [0.0, 0.0, 4.0, 0.0],
                    },
                )
                .unwrap();
            let _candidates = world.step(0.005).unwrap();
        }
        let state = world.readback().unwrap()[1];
        let alignment = (state.orientation[2] * half + state.orientation[3] * half).abs();
        assert!(alignment > 0.998, "{state:?}");
        assert!((state.position_inverse_mass[0] - 1.0).abs() < 0.05);
    }

    #[test]
    fn resident_revolute_joint_allows_only_hinge_rotation() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut fixed = body(0.0, 0.0);
        fixed.position_inverse_mass[3] = 0.0;
        fixed.inverse_inertia_sleep = [0.0; 4];
        let mut options = config();
        options.sleep.enabled = false;
        options.solve.bias_factor = 0.2;
        options.solve.iterations = 16;
        let mut world = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &[fixed, body(1.0, 0.0)],
            &[0.1; 2],
            options,
        )
        .unwrap();
        let hinge = GpuRigidRevoluteJoint {
            body_a: 0,
            body_b: 1,
            local_anchor_a: [0.0; 3],
            local_anchor_b: [-1.0, 0.0, 0.0],
            local_axis_a: [0.0, 0.0, 1.0],
            local_axis_b: [0.0, 0.0, 1.0],
        };
        world.set_revolute_joints(&[hinge]).unwrap();
        assert_eq!(world.revolute_joints(), &[hinge]);
        for _ in 0..120 {
            world
                .write_forces(
                    1,
                    GpuRigidBodyForces {
                        force: [0.0; 4],
                        torque: [4.0, 0.0, 3.0, 0.0],
                    },
                )
                .unwrap();
            let _candidates = world.step(0.005).unwrap();
        }
        let state = world.readback().unwrap()[1];
        let q = state.orientation;
        let anchor = [
            state.position_inverse_mass[0] - 1.0 + 2.0 * (q[1] * q[1] + q[2] * q[2]),
            state.position_inverse_mass[1] - 2.0 * (q[0] * q[1] + q[2] * q[3]),
            state.position_inverse_mass[2]
                - fixed.position_inverse_mass[2]
                - 2.0 * (q[0] * q[2] - q[1] * q[3]),
        ];
        let drift = anchor.into_iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(drift < 0.05, "{state:?}");
        assert!(q[2].abs() > 0.05, "{state:?}");
        assert!(q[0].abs() < 0.05 && q[1].abs() < 0.05, "{state:?}");
        world.set_revolute_joints(&[]).unwrap();
        assert!(world.revolute_joints().is_empty());
    }

    #[test]
    fn resident_prismatic_joint_allows_only_axis_translation() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut fixed = body(0.0, 0.0);
        fixed.position_inverse_mass[3] = 0.0;
        fixed.inverse_inertia_sleep = [0.0; 4];
        let mut moving = body(0.0, 0.0);
        moving.position_inverse_mass[2] = 4.0;
        let mut options = config();
        options.sleep.enabled = false;
        options.solve.bias_factor = 0.2;
        options.solve.iterations = 16;
        let mut world = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &[fixed, moving],
            &[0.1; 2],
            options,
        )
        .unwrap();
        let slider = GpuRigidPrismaticJoint {
            body_a: 0,
            body_b: 1,
            local_anchor_a: [0.0; 3],
            local_anchor_b: [0.0; 3],
            local_rotation_a: [0.0, 0.0, 0.0, 1.0],
            local_rotation_b: [0.0, 0.0, 0.0, 1.0],
        };
        world.set_prismatic_joints(&[slider]).unwrap();
        assert_eq!(world.prismatic_joints(), &[slider]);
        for _ in 0..100 {
            world
                .write_forces(
                    1,
                    GpuRigidBodyForces {
                        force: [4.0, 3.0, 10.0, 0.0],
                        torque: [0.0, 0.0, 4.0, 0.0],
                    },
                )
                .unwrap();
            let _candidates = world.step(0.005).unwrap();
        }
        let state = world.readback().unwrap()[1];
        assert!(state.position_inverse_mass[2] > 4.5, "{state:?}");
        assert!(state.position_inverse_mass[0].abs() < 0.05, "{state:?}");
        assert!(state.position_inverse_mass[1].abs() < 0.05, "{state:?}");
        assert!(state.orientation[2].abs() < 0.05, "{state:?}");
        world.set_prismatic_joints(&[]).unwrap();
        assert!(world.prismatic_joints().is_empty());
    }

    #[test]
    fn resident_prismatic_joint_validates_frames_and_environments() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut world = GpuRigidSphereWorld::new_grouped(
            context.device(),
            context.queue(),
            &[body(0.0, 0.0), body(1.0, 0.0)],
            &[0.1; 2],
            &[0, 1],
            config(),
        )
        .unwrap();
        let slider = GpuRigidPrismaticJoint {
            body_a: 0,
            body_b: 1,
            local_anchor_a: [0.0; 3],
            local_anchor_b: [0.0; 3],
            local_rotation_a: [0.0, 0.0, 0.0, 1.0],
            local_rotation_b: [0.0, 0.0, 0.0, 1.0],
        };
        assert!(world.set_prismatic_joints(&[slider]).is_err());
        assert!(world.prismatic_joints().is_empty());
        let mut invalid = slider;
        invalid.body_b = 0;
        assert!(world.set_prismatic_joints(&[invalid]).is_err());
        invalid.body_b = 1;
        invalid.local_rotation_a = [0.0; 4];
        assert!(world.set_prismatic_joints(&[invalid]).is_err());
    }

    #[test]
    fn resident_prismatic_joint_uses_rotated_local_axis() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut fixed = body(0.0, 0.0);
        fixed.position_inverse_mass[3] = 0.0;
        fixed.inverse_inertia_sleep = [0.0; 4];
        let mut options = config();
        options.sleep.enabled = false;
        options.solve.bias_factor = 0.2;
        options.solve.iterations = 16;
        let mut world = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &[fixed, body(1.0, 0.0)],
            &[0.1; 2],
            options,
        )
        .unwrap();
        let half = 0.5f32.sqrt();
        world
            .set_prismatic_joints(&[GpuRigidPrismaticJoint {
                body_a: 0,
                body_b: 1,
                local_anchor_a: [0.0; 3],
                local_anchor_b: [0.0; 3],
                local_rotation_a: [0.0, half, 0.0, half],
                local_rotation_b: [0.0, half, 0.0, half],
            }])
            .unwrap();
        for _ in 0..100 {
            world
                .write_forces(
                    1,
                    GpuRigidBodyForces {
                        force: [10.0, 4.0, 3.0, 0.0],
                        torque: [0.0, 0.0, 4.0, 0.0],
                    },
                )
                .unwrap();
            let _candidates = world.step(0.005).unwrap();
        }
        let state = world.readback().unwrap()[1];
        assert!(state.position_inverse_mass[0] > 1.5, "{state:?}");
        assert!(state.position_inverse_mass[1].abs() < 0.05, "{state:?}");
        assert!(
            (state.position_inverse_mass[2] - 3.0).abs() < 0.05,
            "{state:?}"
        );
        assert!(state.orientation[2].abs() < 0.05, "{state:?}");
    }

    #[test]
    fn resident_prismatic_motor_and_limit_bound_slide() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut fixed = body(0.0, 0.0);
        fixed.position_inverse_mass[3] = 0.0;
        fixed.inverse_inertia_sleep = [0.0; 4];
        let mut moving = body(0.0, 0.0);
        moving.position_inverse_mass[2] = 4.0;
        let mut options = config();
        options.sleep.enabled = false;
        options.solve.bias_factor = 0.2;
        options.solve.iterations = 16;
        let mut world = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &[fixed, moving],
            &[0.1; 2],
            options,
        )
        .unwrap();
        let slider = GpuRigidPrismaticJoint {
            body_a: 0,
            body_b: 1,
            local_anchor_a: [0.0; 3],
            local_anchor_b: [0.0, 0.0, -1.0],
            local_rotation_a: [0.0, 0.0, 0.0, 1.0],
            local_rotation_b: [0.0, 0.0, 0.0, 1.0],
        };
        world.set_prismatic_joints(&[slider]).unwrap();
        let motor = GpuRigidAxisMotor {
            target_velocity: 2.0,
            max_force: 4.0,
        };
        let limit = GpuRigidPrismaticLimit { min: 0.0, max: 0.2 };
        world.set_prismatic_motor(0, Some(motor)).unwrap();
        world.set_prismatic_limit(0, Some(limit)).unwrap();
        assert_eq!(world.prismatic_motor(0).unwrap(), Some(motor));
        assert_eq!(world.prismatic_limit(0).unwrap(), Some(limit));
        assert!(
            world
                .set_prismatic_limit(0, Some(GpuRigidPrismaticLimit { min: 1.0, max: 0.0 }))
                .is_err()
        );
        assert_eq!(world.prismatic_limit(0).unwrap(), Some(limit));
        for _ in 0..100 {
            let _candidates = world.step(0.005).unwrap();
        }
        let state = world.readback().unwrap()[1];
        assert!(state.position_inverse_mass[2] > 4.1, "{state:?}");
        assert!(state.position_inverse_mass[2] < 4.25, "{state:?}");
        assert!(state.position_inverse_mass[0].abs() < 0.02, "{state:?}");
        let reverse = GpuRigidAxisMotor {
            target_velocity: -2.0,
            ..motor
        };
        world.set_prismatic_motor(0, Some(reverse)).unwrap();
        for _ in 0..120 {
            let _candidates = world.step(0.005).unwrap();
        }
        let state = world.readback().unwrap()[1];
        assert!(state.position_inverse_mass[2] >= 3.95, "{state:?}");
        assert!(state.position_inverse_mass[2] < 4.1, "{state:?}");
        assert!(
            world
                .set_prismatic_motor(
                    0,
                    Some(GpuRigidAxisMotor {
                        target_velocity: f32::NAN,
                        ..motor
                    })
                )
                .is_err()
        );
        assert_eq!(world.prismatic_motor(0).unwrap(), Some(reverse));
        world.set_prismatic_limit(0, None).unwrap();
        assert_eq!(world.prismatic_limit(0).unwrap(), None);
    }

    #[test]
    fn resident_revolute_motor_drives_hinge_with_torque_cap() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut fixed = body(0.0, 0.0);
        fixed.position_inverse_mass[3] = 0.0;
        fixed.inverse_inertia_sleep = [0.0; 4];
        let mut options = config();
        options.sleep.enabled = false;
        options.solve.bias_factor = 0.2;
        options.solve.iterations = 16;
        let mut world = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &[fixed, body(1.0, 0.0)],
            &[0.1; 2],
            options,
        )
        .unwrap();
        world
            .set_revolute_joints(&[GpuRigidRevoluteJoint {
                body_a: 0,
                body_b: 1,
                local_anchor_a: [0.0; 3],
                local_anchor_b: [-1.0, 0.0, 0.0],
                local_axis_a: [0.0, 0.0, 1.0],
                local_axis_b: [0.0, 0.0, 1.0],
            }])
            .unwrap();
        let motor = GpuRigidAxisMotor {
            target_velocity: 1.0,
            max_force: 4.0,
        };
        world.set_revolute_motor(0, Some(motor)).unwrap();
        for _ in 0..100 {
            let _candidates = world.step(0.005).unwrap();
        }
        let state = world.readback().unwrap()[1];
        assert!(state.orientation[2] > 0.05, "{state:?}");
        assert!(state.angular_velocity[2] > 0.2, "{state:?}");
        assert!(state.angular_velocity[2] < 1.2, "{state:?}");
        assert_eq!(world.revolute_motor(0).unwrap(), Some(motor));
        world.reset(&[fixed, body(1.0, 0.0)]).unwrap();
        let capped = GpuRigidAxisMotor {
            target_velocity: 10.0,
            max_force: 0.2,
        };
        world.set_revolute_motor(0, Some(capped)).unwrap();
        let _candidates = world.step(0.005).unwrap();
        let state = world.readback().unwrap()[1];
        assert!(state.angular_velocity[2] > 0.0, "{state:?}");
        assert!(state.angular_velocity[2] < 0.005, "{state:?}");
    }

    #[test]
    fn resident_duplicate_slider_settings_survive_topology_rebuild() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut world = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &[body(0.0, 0.0), body(1.0, 0.0)],
            &[0.1; 2],
            config(),
        )
        .unwrap();
        let slider = GpuRigidPrismaticJoint {
            body_a: 0,
            body_b: 1,
            local_anchor_a: [0.0; 3],
            local_anchor_b: [0.0; 3],
            local_rotation_a: [0.0, 0.0, 0.0, 1.0],
            local_rotation_b: [0.0, 0.0, 0.0, 1.0],
        };
        world.set_prismatic_joints(&[slider, slider]).unwrap();
        let first = GpuRigidAxisMotor {
            target_velocity: 1.0,
            max_force: 2.0,
        };
        let second = GpuRigidAxisMotor {
            target_velocity: -1.0,
            max_force: 3.0,
        };
        world.set_prismatic_motor(0, Some(first)).unwrap();
        world.set_prismatic_motor(1, Some(second)).unwrap();
        world.set_ball_joints(&[]).unwrap();
        assert_eq!(world.prismatic_motor(0).unwrap(), Some(first));
        assert_eq!(world.prismatic_motor(1).unwrap(), Some(second));
    }

    #[test]
    fn resident_revolute_limit_and_position_servo() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut fixed = body(0.0, 0.0);
        fixed.position_inverse_mass[3] = 0.0;
        fixed.inverse_inertia_sleep = [0.0; 4];
        let mut options = config();
        options.sleep.enabled = false;
        options.solve.bias_factor = 0.2;
        options.solve.iterations = 16;
        let mut world = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &[fixed, body(1.0, 0.0)],
            &[0.1; 2],
            options,
        )
        .unwrap();
        world
            .set_revolute_joints(&[GpuRigidRevoluteJoint {
                body_a: 0,
                body_b: 1,
                local_anchor_a: [0.0; 3],
                local_anchor_b: [-1.0, 0.0, 0.0],
                local_axis_a: [0.0, 0.0, 1.0],
                local_axis_b: [0.0, 0.0, 1.0],
            }])
            .unwrap();
        let limit = GpuRigidRevoluteLimit { min: 0.0, max: 0.3 };
        world.set_revolute_limit(0, Some(limit)).unwrap();
        world
            .set_revolute_motor(
                0,
                Some(GpuRigidAxisMotor {
                    target_velocity: 2.0,
                    max_force: 4.0,
                }),
            )
            .unwrap();
        for _ in 0..120 {
            let _candidates = world.step(0.005).unwrap();
        }
        let state = world.readback().unwrap()[1];
        let angle = 2.0 * state.orientation[2].atan2(state.orientation[3]);
        assert!(angle > 0.15 && angle < 0.35, "{state:?}");
        let servo = GpuRigidAxisServo {
            position_target: 0.1,
            velocity_target: 0.0,
            stiffness: 30.0,
            damping: 8.0,
            max_force: 4.0,
        };
        world.set_revolute_servo(0, Some(servo)).unwrap();
        assert_eq!(world.revolute_motor(0).unwrap(), None);
        assert_eq!(world.revolute_servo(0).unwrap(), Some(servo));
        assert_eq!(world.revolute_limit(0).unwrap(), Some(limit));
        assert!(
            world
                .set_revolute_servo(
                    0,
                    Some(GpuRigidAxisServo {
                        position_target: f32::NAN,
                        ..servo
                    })
                )
                .is_err()
        );
        assert_eq!(world.revolute_servo(0).unwrap(), Some(servo));
        for _ in 0..200 {
            let _candidates = world.step(0.005).unwrap();
        }
        let state = world.readback().unwrap()[1];
        let angle = 2.0 * state.orientation[2].atan2(state.orientation[3]);
        assert!((angle - 0.1).abs() < 0.05, "{state:?}");
        assert!(
            world
                .set_revolute_limit(0, Some(GpuRigidRevoluteLimit { min: 1.0, max: 0.0 }))
                .is_err()
        );
        assert_eq!(world.revolute_limit(0).unwrap(), Some(limit));
    }

    #[test]
    fn resident_revolute_tracks_multiple_turns_for_limit_and_servo() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut fixed = body(0.0, 0.0);
        fixed.position_inverse_mass[3] = 0.0;
        fixed.inverse_inertia_sleep = [0.0; 4];
        let moving = body(1.0, 0.0);
        let mut options = config();
        options.sleep.enabled = false;
        options.solve.bias_factor = 0.2;
        options.solve.iterations = 16;
        let mut world = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &[fixed, moving],
            &[0.1; 2],
            options,
        )
        .unwrap();
        world
            .set_revolute_joints(&[GpuRigidRevoluteJoint {
                body_a: 0,
                body_b: 1,
                local_anchor_a: [0.0; 3],
                local_anchor_b: [-1.0, 0.0, 0.0],
                local_axis_a: [0.0, 0.0, 1.0],
                local_axis_b: [0.0, 0.0, 1.0],
            }])
            .unwrap();
        let upper = core::f32::consts::TAU + 0.3;
        world
            .set_revolute_limit(
                0,
                Some(GpuRigidRevoluteLimit {
                    min: 0.0,
                    max: upper,
                }),
            )
            .unwrap();
        world
            .set_revolute_motor(
                0,
                Some(GpuRigidAxisMotor {
                    target_velocity: 3.0,
                    max_force: 30.0,
                }),
            )
            .unwrap();
        assert!(world.readback_revolute_angle(0).unwrap().abs() < 1e-4);
        for _ in 0..10 {
            let _candidates = world.step_substeps(0.01, 32).unwrap();
        }
        let limited = world.readback_revolute_angle(0).unwrap();
        assert!(
            limited > core::f32::consts::TAU && limited < upper + 0.15,
            "{limited}"
        );
        world.set_gravity([0.0; 3]).unwrap();
        assert!((world.readback_revolute_angle(0).unwrap() - limited).abs() < 1e-4);
        world.set_solve_params(options.solve).unwrap();
        assert!((world.readback_revolute_angle(0).unwrap() - limited).abs() < 1e-4);
        world.set_ball_joints(&[]).unwrap();
        assert!((world.readback_revolute_angle(0).unwrap() - limited).abs() < 1e-4);
        let servo = GpuRigidAxisServo {
            position_target: 4.0,
            velocity_target: 0.0,
            stiffness: 30.0,
            damping: 10.0,
            max_force: 20.0,
        };
        world.set_revolute_servo(0, Some(servo)).unwrap();
        for _ in 0..10 {
            let _candidates = world.step_substeps(0.01, 32).unwrap();
        }
        let settled = world.readback_revolute_angle(0).unwrap();
        assert!((settled - 4.0).abs() < 0.15, "{settled}");
        world.reset(&[fixed, moving]).unwrap();
        assert!(world.readback_revolute_angle(0).unwrap().abs() < 1e-4);
        world
            .set_revolute_limit(
                0,
                Some(GpuRigidRevoluteLimit {
                    min: -upper,
                    max: 0.0,
                }),
            )
            .unwrap();
        world
            .set_revolute_motor(
                0,
                Some(GpuRigidAxisMotor {
                    target_velocity: -3.0,
                    max_force: 30.0,
                }),
            )
            .unwrap();
        for _ in 0..10 {
            let _candidates = world.step_substeps(0.01, 32).unwrap();
        }
        let negative = world.readback_revolute_angle(0).unwrap();
        assert!(negative < -core::f32::consts::TAU && negative > -upper - 0.15);
    }

    #[test]
    fn resident_revolute_tracks_more_than_half_turn_per_substep() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut fixed = body(0.0, 0.0);
        fixed.position_inverse_mass[3] = 0.0;
        fixed.inverse_inertia_sleep = [0.0; 4];
        let mut spinning = body(0.0, 0.0);
        spinning.angular_velocity[2] = 400.0;
        let mut options = config();
        options.sleep.enabled = false;
        let mut world = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &[fixed, spinning],
            &[0.1; 2],
            options,
        )
        .unwrap();
        world
            .set_revolute_joints(&[GpuRigidRevoluteJoint {
                body_a: 0,
                body_b: 1,
                local_anchor_a: [0.0; 3],
                local_anchor_b: [0.0; 3],
                local_axis_a: [0.0, 0.0, 1.0],
                local_axis_b: [0.0, 0.0, 1.0],
            }])
            .unwrap();
        world
            .set_body_collision_groups(
                0,
                GpuRigidCollisionGroups {
                    memberships: 1,
                    filter: 1,
                },
            )
            .unwrap();
        world
            .set_body_collision_groups(
                1,
                GpuRigidCollisionGroups {
                    memberships: 2,
                    filter: 2,
                },
            )
            .unwrap();
        assert_eq!(world.step(0.01).unwrap(), 0);
        let first = world.readback_revolute_angle(0).unwrap();
        assert!(
            (first - 4.0).abs() < 0.05,
            "angle={first}, state={:?}",
            world.readback().unwrap()[1]
        );
        assert_eq!(world.readback_revolute_angle(0).unwrap(), first);
        assert_eq!(world.step(0.01).unwrap(), 0);
        assert!((world.readback_revolute_angle(0).unwrap() - 8.0).abs() < 0.1);
        let mut negative = spinning;
        negative.angular_velocity[2] = -400.0;
        world.reset(&[fixed, negative]).unwrap();
        assert_eq!(world.step(0.01).unwrap(), 0);
        assert!((world.readback_revolute_angle(0).unwrap() + 4.0).abs() < 0.05);
        let mut fast = spinning;
        fast.angular_velocity[2] = 800.0;
        world.write_body(1, fast).unwrap();
        assert_eq!(world.step(0.01).unwrap(), 0);
        assert!((world.readback_revolute_angle(0).unwrap() - 8.0).abs() < 0.1);
        world.reset(&[fixed, spinning]).unwrap();
        assert_eq!(world.step_substeps(0.01, 2).unwrap(), 0);
        assert!((world.readback_revolute_angle(0).unwrap() - 8.0).abs() < 0.1);
    }

    #[test]
    fn resident_prismatic_position_servo_tracks_displacement() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut fixed = body(0.0, 0.0);
        fixed.position_inverse_mass[3] = 0.0;
        fixed.inverse_inertia_sleep = [0.0; 4];
        let mut moving = body(0.0, 0.0);
        moving.position_inverse_mass[2] = 4.0;
        let mut options = config();
        options.sleep.enabled = false;
        options.solve.bias_factor = 0.2;
        options.solve.iterations = 16;
        let mut world = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &[fixed, moving],
            &[0.1; 2],
            options,
        )
        .unwrap();
        world
            .set_prismatic_joints(&[GpuRigidPrismaticJoint {
                body_a: 0,
                body_b: 1,
                local_anchor_a: [0.0; 3],
                local_anchor_b: [0.0, 0.0, -1.0],
                local_rotation_a: [0.0, 0.0, 0.0, 1.0],
                local_rotation_b: [0.0, 0.0, 0.0, 1.0],
            }])
            .unwrap();
        let servo = GpuRigidAxisServo {
            position_target: 0.4,
            velocity_target: 0.0,
            stiffness: 20.0,
            damping: 8.0,
            max_force: 5.0,
        };
        world.set_prismatic_servo(0, Some(servo)).unwrap();
        for _ in 0..200 {
            let _candidates = world.step(0.005).unwrap();
        }
        let state = world.readback().unwrap()[1];
        assert!(
            (state.position_inverse_mass[2] - 4.4).abs() < 0.06,
            "{state:?}"
        );
    }

    #[test]
    fn resident_revolute_lower_limit_uses_rotated_axis() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut fixed = body(0.0, 0.0);
        fixed.position_inverse_mass[3] = 0.0;
        fixed.inverse_inertia_sleep = [0.0; 4];
        let mut options = config();
        options.sleep.enabled = false;
        options.solve.bias_factor = 0.2;
        options.solve.iterations = 16;
        let mut world = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &[fixed, body(1.0, 0.0)],
            &[0.1; 2],
            options,
        )
        .unwrap();
        world
            .set_revolute_joints(&[GpuRigidRevoluteJoint {
                body_a: 0,
                body_b: 1,
                local_anchor_a: [0.0; 3],
                local_anchor_b: [-1.0, 0.0, 0.0],
                local_axis_a: [1.0, 0.0, 0.0],
                local_axis_b: [1.0, 0.0, 0.0],
            }])
            .unwrap();
        world
            .set_revolute_limit(
                0,
                Some(GpuRigidRevoluteLimit {
                    min: -0.3,
                    max: 0.0,
                }),
            )
            .unwrap();
        world
            .set_revolute_motor(
                0,
                Some(GpuRigidAxisMotor {
                    target_velocity: -2.0,
                    max_force: 4.0,
                }),
            )
            .unwrap();
        for _ in 0..120 {
            let _candidates = world.step(0.005).unwrap();
        }
        let state = world.readback().unwrap()[1];
        let angle = 2.0 * state.orientation[0].atan2(state.orientation[3]);
        assert!(angle < -0.15 && angle > -0.35, "{state:?}");
    }

    #[test]
    fn resident_prismatic_joint_batch_uses_local_ids() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut fixed = body(0.0, 0.0);
        fixed.position_inverse_mass[3] = 0.0;
        fixed.inverse_inertia_sleep = [0.0; 4];
        let states = [fixed, body(1.0, 0.0)];
        let environments = [
            GpuRigidSphereEnvironment {
                states: &states,
                radii: &[0.1; 2],
            },
            GpuRigidSphereEnvironment {
                states: &states,
                radii: &[0.1; 2],
            },
        ];
        let mut batch =
            GpuRigidSphereBatch::new(context.device(), context.queue(), &environments, config())
                .unwrap();
        let slider = GpuRigidPrismaticJoint {
            body_a: 0,
            body_b: 1,
            local_anchor_a: [0.0; 3],
            local_anchor_b: [0.0; 3],
            local_rotation_a: [0.0, 0.5f32.sqrt(), 0.0, 0.5f32.sqrt()],
            local_rotation_b: [0.0, 0.5f32.sqrt(), 0.0, 0.5f32.sqrt()],
        };
        batch
            .set_prismatic_joints_environment(0, &[slider])
            .unwrap();
        let motor = GpuRigidAxisMotor {
            target_velocity: 1.0,
            max_force: 2.0,
        };
        let limit = GpuRigidPrismaticLimit {
            min: -0.2,
            max: 0.2,
        };
        batch
            .set_prismatic_motor_environment(0, 0, Some(motor))
            .unwrap();
        batch
            .set_prismatic_limit_environment(0, 0, Some(limit))
            .unwrap();
        batch
            .set_prismatic_joints_environment(1, &[slider])
            .unwrap();
        assert_eq!(
            batch.prismatic_motor_environment(0, 0).unwrap(),
            Some(motor)
        );
        assert_eq!(
            batch.prismatic_limit_environment(0, 0).unwrap(),
            Some(limit)
        );
        assert_eq!(batch.prismatic_motor_environment(1, 0).unwrap(), None);
        let servo = GpuRigidAxisServo {
            position_target: 0.1,
            velocity_target: 0.0,
            stiffness: 10.0,
            damping: 2.0,
            max_force: 3.0,
        };
        batch
            .set_prismatic_servo_environment(0, 0, Some(servo))
            .unwrap();
        assert_eq!(
            batch.prismatic_servo_environment(0, 0).unwrap(),
            Some(servo)
        );
        assert_eq!(batch.prismatic_motor_environment(0, 0).unwrap(), None);
        assert_eq!(
            batch.prismatic_limit_environment(0, 0).unwrap(),
            Some(limit)
        );
        assert!(
            batch
                .set_prismatic_motor_environment(1, 1, Some(motor))
                .is_err()
        );
        assert_eq!(batch.prismatic_joints_environment(0).unwrap(), vec![slider]);
        assert_eq!(batch.prismatic_joints_environment(1).unwrap(), vec![slider]);
        assert!(
            batch
                .set_prismatic_joints_environment(
                    0,
                    &[GpuRigidPrismaticJoint {
                        body_b: 2,
                        ..slider
                    }]
                )
                .is_err()
        );
        batch.set_prismatic_joints_environment(0, &[]).unwrap();
        assert!(batch.prismatic_joints_environment(0).unwrap().is_empty());
        assert_eq!(batch.prismatic_joints_environment(1).unwrap(), vec![slider]);
    }

    #[test]
    fn resident_revolute_joint_light_body_tracks_hinge_inertia() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut fixed = body(0.0, 0.0);
        fixed.position_inverse_mass[3] = 0.0;
        fixed.inverse_inertia_sleep = [0.0; 4];
        let mut moving = body(1.0, 0.0);
        moving.inverse_inertia_sleep = [250.0, 250.0, 250.0, 0.0];
        let mut options = config();
        options.solve = GpuRigidSphereSolveParams::default();
        options.sleep.enabled = false;
        let mut world = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &[fixed, moving],
            &[0.1; 2],
            options,
        )
        .unwrap();
        world
            .set_revolute_joints(&[GpuRigidRevoluteJoint {
                body_a: 0,
                body_b: 1,
                local_anchor_a: [0.0; 3],
                local_anchor_b: [-1.0, 0.0, 0.0],
                local_axis_a: [0.0, 0.0, 1.0],
                local_axis_b: [0.0, 0.0, 1.0],
            }])
            .unwrap();
        for _ in 0..120 {
            world
                .write_forces(
                    1,
                    GpuRigidBodyForces {
                        force: [0.0; 4],
                        torque: [0.0, 0.0, 3.0, 0.0],
                    },
                )
                .unwrap();
            let _candidates = world.step(0.005).unwrap();
        }
        let state = world.readback().unwrap()[1];
        let angle = 2.0 * state.orientation[2].atan2(state.orientation[3]);
        let expected_angular_speed = 3.0 * 0.6 / 1.004;
        let expected_angle = 0.5 * expected_angular_speed * 0.6;
        assert!(
            (state.angular_velocity[2] - expected_angular_speed).abs() < 0.15,
            "{state:?}"
        );
        assert!((angle - expected_angle).abs() < 0.08, "{state:?}");
        let x_error = state.position_inverse_mass[0] - angle.cos();
        let y_error = state.position_inverse_mass[1] - angle.sin();
        assert!(
            (x_error * x_error + y_error * y_error).sqrt() < 0.05,
            "{state:?}"
        );
    }

    #[test]
    fn resident_revolute_joint_batched_substeps_match_sequential_steps() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut fixed = body(0.0, 0.0);
        fixed.position_inverse_mass[3] = 0.0;
        fixed.inverse_inertia_sleep = [0.0; 4];
        let mut moving = body(1.0, 0.0);
        moving.linear_velocity = [0.0, 2.0, 0.0, 0.0];
        moving.angular_velocity = [0.0, 0.0, 2.0, 0.0];
        let mut options = config();
        options.sleep.enabled = false;
        options.solve.bias_factor = 0.2;
        let joint = GpuRigidRevoluteJoint {
            body_a: 0,
            body_b: 1,
            local_anchor_a: [0.0; 3],
            local_anchor_b: [-1.0, 0.0, 0.0],
            local_axis_a: [0.0, 0.0, 1.0],
            local_axis_b: [0.0, 0.0, 1.0],
        };
        let make_world = || {
            let mut world = GpuRigidSphereWorld::new(
                context.device(),
                context.queue(),
                &[fixed, moving],
                &[0.1; 2],
                options,
            )
            .unwrap();
            world.set_revolute_joints(&[joint]).unwrap();
            world
        };
        let mut batched = make_world();
        let mut sequential = make_world();
        let _candidates = batched.step_substeps(0.005, 100).unwrap();
        for _ in 0..100 {
            let _candidates = sequential.step(0.005).unwrap();
        }
        let left = batched.readback().unwrap()[1];
        let right = sequential.readback().unwrap()[1];
        for (a, b) in left
            .position_inverse_mass
            .into_iter()
            .zip(right.position_inverse_mass)
        {
            assert!((a - b).abs() < 1e-4, "{left:?} != {right:?}");
        }
        for (a, b) in left.orientation.into_iter().zip(right.orientation) {
            assert!((a - b).abs() < 1e-4, "{left:?} != {right:?}");
        }
    }

    #[tokio::test]
    async fn vulkan_resident_revolute_joint_tracks_positive_hinge_torque() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
        else {
            eprintln!("Vulkan adapter unavailable; skipping resident revolute test");
            return;
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let mut fixed = body(0.0, 0.0);
        fixed.position_inverse_mass[3] = 0.0;
        fixed.inverse_inertia_sleep = [0.0; 4];
        let mut moving = body(1.0, 0.0);
        moving.inverse_inertia_sleep = [250.0, 250.0, 250.0, 0.0];
        let mut options = config();
        options.solve = GpuRigidSphereSolveParams::default();
        options.sleep.enabled = false;
        let mut world =
            GpuRigidSphereWorld::new(&device, &queue, &[fixed, moving], &[0.1; 2], options)
                .unwrap();
        world
            .set_revolute_joints(&[GpuRigidRevoluteJoint {
                body_a: 0,
                body_b: 1,
                local_anchor_a: [0.0; 3],
                local_anchor_b: [-1.0, 0.0, 0.0],
                local_axis_a: [0.0, 0.0, 1.0],
                local_axis_b: [0.0, 0.0, 1.0],
            }])
            .unwrap();
        for _ in 0..60 {
            world
                .write_forces(
                    1,
                    GpuRigidBodyForces {
                        force: [0.0; 4],
                        torque: [0.0, 0.0, 3.0, 0.0],
                    },
                )
                .unwrap();
            let _candidates = world.step(0.005).unwrap();
        }
        let moving = world.readback().unwrap()[1];
        assert!(moving.angular_velocity[2] > 0.7, "{moving:?}");
        assert!(moving.orientation[2] > 0.03, "{moving:?}");
    }

    #[tokio::test]
    async fn vulkan_resident_prismatic_motor_respects_limit() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
        else {
            eprintln!("Vulkan adapter unavailable; skipping resident prismatic motor test");
            return;
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let mut fixed = body(0.0, 0.0);
        fixed.position_inverse_mass[3] = 0.0;
        fixed.inverse_inertia_sleep = [0.0; 4];
        let mut moving = body(0.0, 0.0);
        moving.position_inverse_mass[2] = 4.0;
        let mut options = config();
        options.sleep.enabled = false;
        options.solve.bias_factor = 0.2;
        options.solve.iterations = 16;
        let mut world =
            GpuRigidSphereWorld::new(&device, &queue, &[fixed, moving], &[0.1; 2], options)
                .unwrap();
        world
            .set_prismatic_joints(&[GpuRigidPrismaticJoint {
                body_a: 0,
                body_b: 1,
                local_anchor_a: [0.0; 3],
                local_anchor_b: [0.0, 0.0, -1.0],
                local_rotation_a: [0.0, 0.0, 0.0, 1.0],
                local_rotation_b: [0.0, 0.0, 0.0, 1.0],
            }])
            .unwrap();
        world
            .set_prismatic_motor(
                0,
                Some(GpuRigidAxisMotor {
                    target_velocity: 2.0,
                    max_force: 4.0,
                }),
            )
            .unwrap();
        world
            .set_prismatic_limit(0, Some(GpuRigidPrismaticLimit { min: 0.0, max: 0.2 }))
            .unwrap();
        for _ in 0..100 {
            let _candidates = world.step(0.005).unwrap();
        }
        let moving = world.readback().unwrap()[1];
        assert!(
            (4.1..4.25).contains(&moving.position_inverse_mass[2]),
            "{moving:?}"
        );
    }

    #[test]
    fn resident_revolute_joint_rejects_invalid_axis_and_cross_environment() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut world = GpuRigidSphereWorld::new_grouped(
            context.device(),
            context.queue(),
            &[body(0.0, 0.0), body(1.0, 0.0)],
            &[0.1; 2],
            &[0, 1],
            config(),
        )
        .unwrap();
        let hinge = GpuRigidRevoluteJoint {
            body_a: 0,
            body_b: 1,
            local_anchor_a: [0.0; 3],
            local_anchor_b: [0.0; 3],
            local_axis_a: [0.0, 0.0, 1.0],
            local_axis_b: [0.0, 0.0, 1.0],
        };
        assert!(world.set_revolute_joints(&[hinge]).is_err());
        assert!(world.revolute_joints().is_empty());
        let mut invalid = hinge;
        invalid.body_b = 0;
        invalid.local_axis_b = [0.0; 3];
        assert!(world.set_revolute_joints(&[invalid]).is_err());
    }

    #[test]
    fn resident_revolute_joint_batch_uses_local_body_ids() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut fixed = body(0.0, 0.0);
        fixed.position_inverse_mass[3] = 0.0;
        fixed.inverse_inertia_sleep = [0.0; 4];
        let states = [fixed, body(1.0, 0.0)];
        let environments = [
            GpuRigidSphereEnvironment {
                states: &states,
                radii: &[0.1; 2],
            },
            GpuRigidSphereEnvironment {
                states: &states,
                radii: &[0.1; 2],
            },
        ];
        let mut options = config();
        options.sleep.enabled = false;
        options.solve.bias_factor = 0.2;
        let mut batch =
            GpuRigidSphereBatch::new(context.device(), context.queue(), &environments, options)
                .unwrap();
        let hinge = GpuRigidRevoluteJoint {
            body_a: 0,
            body_b: 1,
            local_anchor_a: [0.0; 3],
            local_anchor_b: [-1.0, 0.0, 0.0],
            local_axis_a: [0.0, 0.0, 1.0],
            local_axis_b: [0.0, 0.0, 1.0],
        };
        batch.set_revolute_joints_environment(0, &[hinge]).unwrap();
        batch.set_revolute_joints_environment(1, &[hinge]).unwrap();
        assert_eq!(batch.revolute_joints_environment(0).unwrap(), vec![hinge]);
        assert_eq!(batch.revolute_joints_environment(1).unwrap(), vec![hinge]);
        assert!(
            batch
                .set_revolute_joints_environment(0, &[GpuRigidRevoluteJoint { body_b: 2, ..hinge }])
                .is_err()
        );
        for _ in 0..60 {
            batch
                .write_forces(
                    0,
                    1,
                    GpuRigidBodyForces {
                        force: [0.0; 4],
                        torque: [0.0, 0.0, 3.0, 0.0],
                    },
                )
                .unwrap();
            let _candidates = batch.step(0.005).unwrap();
        }
        assert!(batch.readback_environment(0).unwrap()[1].orientation[2].abs() > 0.01);
        assert!(batch.readback_environment(1).unwrap()[1].orientation[2].abs() < 1e-5);
        assert!(batch.readback_revolute_angle_environment(0, 0).unwrap() > 0.02);
        assert!(
            batch
                .readback_revolute_angle_environment(1, 0)
                .unwrap()
                .abs()
                < 1e-5
        );
        let limit = GpuRigidRevoluteLimit {
            min: -0.2,
            max: 0.5,
        };
        let servo = GpuRigidAxisServo {
            position_target: 0.1,
            velocity_target: 0.0,
            stiffness: 10.0,
            damping: 2.0,
            max_force: 3.0,
        };
        batch
            .set_revolute_limit_environment(0, 0, Some(limit))
            .unwrap();
        batch
            .set_revolute_servo_environment(0, 0, Some(servo))
            .unwrap();
        batch.set_revolute_joints_environment(1, &[hinge]).unwrap();
        assert_eq!(batch.revolute_limit_environment(0, 0).unwrap(), Some(limit));
        assert_eq!(batch.revolute_servo_environment(0, 0).unwrap(), Some(servo));
        assert_eq!(batch.revolute_servo_environment(1, 0).unwrap(), None);
        batch.set_revolute_joints_environment(0, &[]).unwrap();
        assert!(batch.revolute_joints_environment(0).unwrap().is_empty());
        assert_eq!(batch.revolute_joints_environment(1).unwrap(), vec![hinge]);
    }

    #[test]
    fn resident_contact_joint_iterations_support_a_linked_body() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut lower = body(0.0, 0.0);
        lower.position_inverse_mass[2] = 0.4;
        let mut upper = body(0.0, 0.0);
        upper.position_inverse_mass[2] = 1.4;
        upper.linear_velocity[2] = -2.0;
        let mut options = config();
        options.ground_half_extent = Some(10.0);
        options.solve.bias_factor = 0.2;
        options.solve.iterations = 8;
        options.sleep.enabled = false;
        let joint = GpuRigidFixedJoint {
            body_a: 0,
            body_b: 1,
            local_anchor_a: [0.0, 0.0, 1.0],
            local_anchor_b: [0.0; 3],
            local_rotation_a: [0.0, 0.0, 0.0, 1.0],
            local_rotation_b: [0.0, 0.0, 0.0, 1.0],
        };
        let new_world = || {
            let mut world = GpuRigidSphereWorld::new(
                context.device(),
                context.queue(),
                &[lower, upper],
                &[0.5, 0.2],
                options,
            )
            .unwrap();
            world.set_fixed_joints(&[joint]).unwrap();
            world
        };
        let mut coupled = new_world();
        let _candidates = coupled.step(0.01).unwrap();
        let coupled = coupled.readback().unwrap();

        let mut split = new_world();
        let mut encoder =
            context
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Tessera split contact and joint baseline"),
                });
        split
            .state
            .encode_step(context.device(), &mut encoder, 0.01, options.gravity)
            .unwrap();
        split
            .ball_joints
            .as_ref()
            .unwrap()
            .encode_capture_velocity(&mut encoder);
        split.contacts.encode(&mut encoder);
        split
            .solver
            .encode_cached(
                context.device(),
                &mut encoder,
                &split.contacts,
                0.01,
                options.solve,
                &mut split.impulse_cache,
            )
            .unwrap();
        split
            .ball_joints
            .as_mut()
            .unwrap()
            .encode(
                context.queue(),
                &mut encoder,
                0.01,
                options.solve.iterations,
                options.solve.bias_factor,
            )
            .unwrap();
        let _submission = context.queue().submit(Some(encoder.finish()));
        let split = split.readback().unwrap();
        assert!(
            coupled[0].linear_velocity[2] > split[0].linear_velocity[2] + 0.5,
            "coupled: {coupled:?}, split: {split:?}"
        );
        assert!(
            (coupled[0].linear_velocity[2] - coupled[1].linear_velocity[2]).abs() < 0.5,
            "{coupled:?}"
        );
    }

    #[test]
    fn resident_mixed_ball_fixed_revolute_chain_stays_finite() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut fixed = body(0.0, 0.0);
        fixed.position_inverse_mass[3] = 0.0;
        fixed.inverse_inertia_sleep = [0.0; 4];
        let mut options = config();
        options.sleep.enabled = false;
        options.solve.bias_factor = 0.2;
        options.solve.iterations = 16;
        let mut world = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &[fixed, body(1.0, 0.0), body(2.0, 0.0), body(3.0, 0.0)],
            &[0.1; 4],
            options,
        )
        .unwrap();
        let ball = GpuRigidBallJoint {
            body_a: 2,
            body_b: 3,
            local_anchor_a: [1.0, 0.0, 0.0],
            local_anchor_b: [0.0; 3],
        };
        let fixed_joint = GpuRigidFixedJoint {
            body_a: 1,
            body_b: 2,
            local_anchor_a: [1.0, 0.0, 0.0],
            local_anchor_b: [0.0; 3],
            local_rotation_a: [0.0, 0.0, 0.0, 1.0],
            local_rotation_b: [0.0, 0.0, 0.0, 1.0],
        };
        let hinge = GpuRigidRevoluteJoint {
            body_a: 0,
            body_b: 1,
            local_anchor_a: [0.0; 3],
            local_anchor_b: [-1.0, 0.0, 0.0],
            local_axis_a: [0.0, 0.0, 1.0],
            local_axis_b: [0.0, 0.0, 1.0],
        };
        world
            .set_all_joints(&[ball], &[fixed_joint], &[hinge])
            .unwrap();
        for _ in 0..100 {
            world
                .write_forces(
                    3,
                    GpuRigidBodyForces {
                        force: [0.0, 1.0, 0.0, 0.0],
                        torque: [0.0; 4],
                    },
                )
                .unwrap();
            let _candidates = world.step(0.005).unwrap();
        }
        let states = world.readback().unwrap();
        assert!(
            states
                .iter()
                .flat_map(|state| state.position_inverse_mass)
                .all(f32::is_finite)
        );
        assert!(
            states
                .iter()
                .flat_map(|state| state.orientation)
                .all(f32::is_finite)
        );
        assert!(states[1].position_inverse_mass[1].abs() < 0.1);
        assert!(states[2].position_inverse_mass[1].abs() < 0.1);
        assert!(states[3].position_inverse_mass[1].abs() < 0.15);
        let angle = world.readback_revolute_angle(0).unwrap();
        let expected = 2.0 * states[1].orientation[2].atan2(states[1].orientation[3]);
        assert!((angle - expected).abs() < 1e-3, "{angle}, {expected}");
    }

    #[test]
    fn resident_mixed_joint_chain_shares_one_island() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut fixed = body(0.0, 0.0);
        fixed.position_inverse_mass[3] = 0.0;
        fixed.inverse_inertia_sleep = [0.0; 4];
        let mut options = config();
        options.sleep.enabled = false;
        options.solve.bias_factor = 0.2;
        options.solve.iterations = 16;
        let mut world = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &[fixed, body(1.0, 0.0), body(2.0, 0.0)],
            &[0.1; 3],
            options,
        )
        .unwrap();
        let ball = GpuRigidBallJoint {
            body_a: 1,
            body_b: 2,
            local_anchor_a: [1.0, 0.0, 0.0],
            local_anchor_b: [0.0; 3],
        };
        let joint = GpuRigidFixedJoint {
            body_a: 0,
            body_b: 1,
            local_anchor_a: [1.0, 0.0, 0.0],
            local_anchor_b: [0.0; 3],
            local_rotation_a: [0.0, 0.0, 0.0, 1.0],
            local_rotation_b: [0.0, 0.0, 0.0, 1.0],
        };
        world.set_joints(&[ball], &[joint]).unwrap();
        assert_eq!(world.ball_joints(), &[ball]);
        assert_eq!(world.fixed_joints(), &[joint]);
        for _ in 0..100 {
            world
                .write_forces(
                    2,
                    GpuRigidBodyForces {
                        force: [0.0, 10.0, 0.0, 0.0],
                        torque: [0.0; 4],
                    },
                )
                .unwrap();
            let _candidates = world.step(0.005).unwrap();
        }
        let actual = world.readback().unwrap();
        assert!(
            actual[1].position_inverse_mass[1].abs() < 0.05,
            "{actual:?}"
        );
        assert!(actual[1].orientation[2].abs() < 0.05, "{actual:?}");
        assert!(actual[2].position_inverse_mass[1].abs() < 0.1, "{actual:?}");
    }

    #[test]
    fn resident_fixed_joint_batch_uses_local_body_ids() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut fixed = body(0.0, 0.0);
        fixed.position_inverse_mass[3] = 0.0;
        fixed.inverse_inertia_sleep = [0.0; 4];
        let states = [fixed, body(1.0, 0.0)];
        let environments = [
            GpuRigidSphereEnvironment {
                states: &states,
                radii: &[0.1; 2],
            },
            GpuRigidSphereEnvironment {
                states: &states,
                radii: &[0.1; 2],
            },
        ];
        let mut options = config();
        options.sleep.enabled = false;
        options.solve.bias_factor = 0.2;
        let mut batch =
            GpuRigidSphereBatch::new(context.device(), context.queue(), &environments, options)
                .unwrap();
        let joint = GpuRigidFixedJoint {
            body_a: 0,
            body_b: 1,
            local_anchor_a: [1.0, 0.0, 0.0],
            local_anchor_b: [0.0; 3],
            local_rotation_a: [0.0, 0.0, 0.0, 1.0],
            local_rotation_b: [0.0, 0.0, 0.0, 1.0],
        };
        batch.set_fixed_joints_environment(0, &[joint]).unwrap();
        batch.set_fixed_joints_environment(1, &[joint]).unwrap();
        assert_eq!(batch.fixed_joints_environment(0).unwrap(), vec![joint]);
        assert_eq!(batch.fixed_joints_environment(1).unwrap(), vec![joint]);
        assert!(
            batch
                .set_fixed_joints_environment(0, &[GpuRigidFixedJoint { body_b: 2, ..joint }])
                .is_err()
        );
        for _ in 0..60 {
            batch
                .write_forces(
                    0,
                    1,
                    GpuRigidBodyForces {
                        force: [0.0, 10.0, 0.0, 0.0],
                        torque: [0.0, 0.0, 5.0, 0.0],
                    },
                )
                .unwrap();
            let _candidates = batch.step(0.005).unwrap();
        }
        assert!(batch.readback_environment(0).unwrap()[1].orientation[2].abs() < 0.05);
        assert!(batch.readback_environment(1).unwrap()[1].orientation[2].abs() < 1e-5);
        batch.set_fixed_joints_environment(0, &[]).unwrap();
        assert!(batch.fixed_joints_environment(0).unwrap().is_empty());
        assert_eq!(batch.fixed_joints_environment(1).unwrap(), vec![joint]);
    }

    #[test]
    fn resident_ball_joint_rejects_cross_environment_links() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let states = [body(0.0, 0.0), body(1.0, 0.0)];
        let mut world = GpuRigidSphereWorld::new_grouped(
            context.device(),
            context.queue(),
            &states,
            &[0.1; 2],
            &[0, 1],
            config(),
        )
        .unwrap();
        let joint = GpuRigidBallJoint {
            body_a: 0,
            body_b: 1,
            local_anchor_a: [0.0; 3],
            local_anchor_b: [0.0; 3],
        };
        assert!(matches!(
            world.set_ball_joints(&[joint]),
            Err(GpuRigidSphereWorldError::BallJoint(
                GpuRigidBallJointError::InvalidInput
            ))
        ));
        assert!(world.ball_joints().is_empty());
    }

    #[test]
    fn resident_ball_joint_chain_solves_shared_body_without_gpu_races() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut fixed = body(0.0, 0.0);
        fixed.position_inverse_mass[3] = 0.0;
        fixed.inverse_inertia_sleep = [0.0; 4];
        let mut middle = body(1.0, 0.0);
        middle.inverse_inertia_sleep = [0.0; 4];
        let initial = [fixed, middle, body(2.0, 0.0)];
        let mut options = config();
        options.sleep.enabled = false;
        options.solve.bias_factor = 0.2;
        options.solve.iterations = 12;
        let mut world = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &initial,
            &[0.1; 3],
            options,
        )
        .unwrap();
        world
            .set_ball_joints(&[
                GpuRigidBallJoint {
                    body_a: 0,
                    body_b: 1,
                    local_anchor_a: [1.0, 0.0, 0.0],
                    local_anchor_b: [0.0; 3],
                },
                GpuRigidBallJoint {
                    body_a: 1,
                    body_b: 2,
                    local_anchor_a: [1.0, 0.0, 0.0],
                    local_anchor_b: [0.0; 3],
                },
            ])
            .unwrap();
        for _ in 0..100 {
            world
                .write_forces(
                    2,
                    GpuRigidBodyForces {
                        force: [0.0, 10.0, 0.0, 0.0],
                        torque: [0.0; 4],
                    },
                )
                .unwrap();
            let _candidates = world.step(0.01).unwrap();
        }
        let actual = world.readback().unwrap();
        assert!(actual[1].position_inverse_mass[1].abs() < 0.1);
        assert!(actual[2].position_inverse_mass[1].abs() < 0.1);
        world.reset(&initial).unwrap();
        let _candidates = world.step(0.01).unwrap();
        let reset = world.readback().unwrap();
        assert!(reset[1].position_inverse_mass[1].abs() < 1e-4);
        assert!(reset[2].position_inverse_mass[1].abs() < 1e-4);
    }

    #[test]
    fn resident_ball_joint_batch_keeps_environment_constraints_independent() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut fixed = body(0.0, 0.0);
        fixed.position_inverse_mass[3] = 0.0;
        fixed.inverse_inertia_sleep = [0.0; 4];
        let states = [fixed, body(1.0, 0.0)];
        let environments = [
            GpuRigidSphereEnvironment {
                states: &states,
                radii: &[0.1; 2],
            },
            GpuRigidSphereEnvironment {
                states: &states,
                radii: &[0.1; 2],
            },
        ];
        let mut options = config();
        options.solve.bias_factor = 0.2;
        options.sleep.enabled = false;
        let mut batch =
            GpuRigidSphereBatch::new(context.device(), context.queue(), &environments, options)
                .unwrap();
        let joint = GpuRigidBallJoint {
            body_a: 0,
            body_b: 1,
            local_anchor_a: [1.0, 0.0, 0.0],
            local_anchor_b: [0.0; 3],
        };
        batch.set_ball_joints_environment(0, &[joint]).unwrap();
        batch.set_ball_joints_environment(1, &[joint]).unwrap();
        assert_eq!(batch.ball_joints_environment(0).unwrap(), vec![joint]);
        assert_eq!(batch.ball_joints_environment(1).unwrap(), vec![joint]);
        assert!(matches!(
            batch.set_ball_joints_environment(0, &[GpuRigidBallJoint { body_b: 2, ..joint }]),
            Err(GpuRigidSphereWorldError::InvalidInput)
        ));
        for _ in 0..60 {
            batch
                .write_forces(
                    0,
                    1,
                    GpuRigidBodyForces {
                        force: [0.0, 10.0, 0.0, 0.0],
                        torque: [0.0; 4],
                    },
                )
                .unwrap();
            let _candidates = batch.step(0.01).unwrap();
        }
        assert!(batch.readback_environment(0).unwrap()[1].position_inverse_mass[1].abs() < 0.05);
        assert!(batch.readback_environment(1).unwrap()[1].position_inverse_mass[1].abs() < 1e-5);
        batch.set_ball_joints_environment(0, &[]).unwrap();
        assert!(batch.ball_joints_environment(0).unwrap().is_empty());
        assert_eq!(batch.ball_joints_environment(1).unwrap(), vec![joint]);
    }

    #[test]
    fn primitive_batch_ground_contacts_use_environment_local_bodies() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut state = body(0.0, 0.0);
        state.position_inverse_mass[2] = 0.45;
        let states = [state];
        let mut lower_state = state;
        lower_state.position_inverse_mass[2] = 0.25;
        let lower_states = [lower_state];
        let shapes = [GpuRigidShape::Box {
            half_extents: [0.5; 3],
        }];
        let environments = [
            GpuRigidPrimitiveEnvironment {
                states: &states,
                shapes: &shapes,
            },
            GpuRigidPrimitiveEnvironment {
                states: &lower_states,
                shapes: &shapes,
            },
        ];
        let mut settings = config();
        settings.ground_half_extent = Some(2.0);
        let mut batch = GpuRigidPrimitiveBatch::new_primitives(
            context.device(),
            context.queue(),
            &environments,
            settings,
        )
        .unwrap();
        assert_eq!(batch.step(0.01).unwrap(), 0);
        let full = batch.world().readback_contacts().unwrap();
        for environment in 0..2 {
            let contacts = batch.readback_contacts_environment(environment).unwrap();
            assert!(contacts.pairs.is_empty());
            assert_eq!(contacts.ground.len(), 1);
            assert_eq!(contacts.ground_extra.len(), 1);
            assert!(contacts.ground[0].is_contact());
            assert!(
                contacts.ground_extra[0]
                    .iter()
                    .all(|contact| contact.is_contact())
            );
            assert_eq!(
                bytemuck::bytes_of(&contacts.ground[0]),
                bytemuck::bytes_of(&full.ground[environment])
            );
            assert_eq!(
                bytemuck::bytes_of(&contacts.ground_extra[0]),
                bytemuck::bytes_of(&full.ground_extra[environment])
            );
        }
    }

    #[test]
    fn resetting_one_environment_preserves_other_warm_start_history() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut lower = body(0.0, 0.0);
        lower.position_inverse_mass[2] = 1.0;
        let mut upper = body(0.0, 0.0);
        upper.position_inverse_mass[2] = 3.0;
        let initial = [lower, upper];
        let environments = [
            GpuRigidSphereEnvironment {
                states: &initial,
                radii: &[1.0; 2],
            },
            GpuRigidSphereEnvironment {
                states: &initial,
                radii: &[1.0; 2],
            },
        ];
        let mut settings = GpuRigidSphereWorldConfig::default();
        settings.solve.iterations = 1;
        settings.sleep.enabled = false;
        let mut batch =
            GpuRigidSphereBatch::new(context.device(), context.queue(), &environments, settings)
                .unwrap();
        let mut reference = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &initial,
            &[1.0; 2],
            settings,
        )
        .unwrap();
        let _batch_candidates = batch.step_substeps(0.01, 10).unwrap();
        let _reference_candidates = reference.step_substeps(0.01, 10).unwrap();
        batch.reset_environment(0, &initial).unwrap();
        let _batch_candidates = batch.step_substeps(0.01, 10).unwrap();
        let _reference_candidates = reference.step_substeps(0.01, 10).unwrap();
        let actual = batch.readback_environment(1).unwrap();
        let expected = reference.readback().unwrap();
        for (left, right) in actual.iter().zip(expected) {
            assert!((left.position_inverse_mass[2] - right.position_inverse_mass[2]).abs() < 1e-5);
            assert!((left.linear_velocity[2] - right.linear_velocity[2]).abs() < 1e-5);
        }
    }

    #[test]
    fn packed_force_targets_only_one_environment() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let initial = [body(0.0, 0.0)];
        let environments = [
            GpuRigidSphereEnvironment {
                states: &initial,
                radii: &[1.0],
            },
            GpuRigidSphereEnvironment {
                states: &initial,
                radii: &[1.0],
            },
        ];
        let mut batch =
            GpuRigidSphereBatch::new(context.device(), context.queue(), &environments, config())
                .unwrap();
        batch
            .write_forces(
                0,
                0,
                GpuRigidBodyForces {
                    force: [10.0, 0.0, 0.0, 0.0],
                    torque: [0.0; 4],
                },
            )
            .unwrap();
        let _candidates = batch.step(0.1).unwrap();
        assert!((batch.readback_environment(0).unwrap()[0].linear_velocity[0] - 1.0).abs() < 1e-5);
        assert_eq!(
            batch.readback_environment(1).unwrap()[0].linear_velocity[0],
            0.0
        );
        assert!(matches!(
            batch.write_forces(2, 0, GpuRigidBodyForces::default()),
            Err(GpuRigidSphereWorldError::InvalidInput)
        ));
    }

    #[test]
    fn many_small_environments_use_parallel_exhaustive_islands() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let initial = [body(-1.0, 1.0), body(1.0, -1.0)];
        let environments = (0..100)
            .map(|_| GpuRigidSphereEnvironment {
                states: &initial,
                radii: &[1.0; 2],
            })
            .collect::<Vec<_>>();
        let mut batch =
            GpuRigidSphereBatch::new(context.device(), context.queue(), &environments, config())
                .unwrap();
        assert!(batch.world().uses_exhaustive_pairs());
        assert_eq!(batch.world().candidate_pair_count(), 100);
        assert_eq!(batch.step_substeps(0.01, 4).unwrap(), 100);
        let expected = batch.readback_environment(0).unwrap();
        for environment in [1, 63, 64, 99] {
            assert_same_states(&batch.readback_environment(environment).unwrap(), &expected);
        }
    }

    #[test]
    fn grouped_lbvh_discards_cross_environment_candidates() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut bodies = (0..17)
            .map(|index| body(index as f32 * 10.0, 0.0))
            .collect::<Vec<_>>();
        bodies.push(body(0.0, 0.0));
        let mut environment_ids = vec![0; 17];
        environment_ids.push(1);
        let mut world = GpuRigidSphereWorld::new_grouped(
            context.device(),
            context.queue(),
            &bodies,
            &vec![1.0; bodies.len()],
            &environment_ids,
            config(),
        )
        .unwrap();
        assert!(!world.uses_exhaustive_pairs());
        assert_eq!(world.step(0.01).unwrap(), 0);
        world.write_body(1, body(1.5, 0.0)).unwrap();
        assert_eq!(world.step(0.01).unwrap(), 1);
        assert!(matches!(
            GpuRigidSphereWorld::new_grouped(
                context.device(),
                context.queue(),
                &bodies,
                &vec![1.0; bodies.len()],
                &[0],
                config(),
            ),
            Err(GpuRigidSphereWorldError::InvalidInput)
        ));
    }

    #[test]
    fn slow_dynamic_contact_does_not_wake_sleeping_support() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut support = body(0.0, 0.0);
        support.inverse_inertia_sleep[3] = 1.0;
        let mut world = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &[support, body(2.0, -0.1)],
            &[1.0; 2],
            config(),
        )
        .unwrap();
        assert_eq!(world.step(0.01).unwrap(), 1);
        let actual = world.readback().unwrap();
        assert_eq!(actual[0].inverse_inertia_sleep[3], 1.0);
        assert_eq!(actual[0].position_inverse_mass[0], 0.0);
    }

    #[test]
    fn exhaustive_pairs_refresh_contacts_after_body_writes_and_reset() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut world = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &[body(0.0, 0.0), body(5.0, 0.0)],
            &[1.0; 2],
            config(),
        )
        .unwrap();
        assert!(world.uses_exhaustive_pairs());
        assert_eq!(world.step(0.01).unwrap(), 1);
        assert_eq!(
            world.readback_contacts().unwrap().pairs[0].1.depth_hit[1],
            0.0
        );
        world.write_body(1, body(1.5, 0.0)).unwrap();
        assert_eq!(world.step(0.01).unwrap(), 1);
        assert_ne!(
            world.readback_contacts().unwrap().pairs[0].1.depth_hit[1],
            0.0
        );
        world.reset(&[body(0.0, 0.0), body(5.0, 0.0)]).unwrap();
        assert_eq!(world.step(0.01).unwrap(), 1);
        assert_eq!(
            world.readback_contacts().unwrap().pairs[0].1.depth_hit[1],
            0.0
        );
    }

    #[test]
    fn larger_scene_keeps_dynamic_lbvh_candidate_refresh() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let bodies = (0..(EXHAUSTIVE_PAIR_BODY_LIMIT + 1))
            .map(|index| body(index as f32 * 10.0, 0.0))
            .collect::<Vec<_>>();
        let mut world = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &bodies,
            &vec![1.0; bodies.len()],
            config(),
        )
        .unwrap();
        assert!(!world.uses_exhaustive_pairs());
        assert_eq!(world.step(0.01).unwrap(), 0);
        world.write_body(1, body(1.5, 0.0)).unwrap();
        assert_eq!(world.step(0.01).unwrap(), 1);
        assert_eq!(world.step_substeps(0.01, 2).unwrap(), 1);
    }

    #[test]
    fn invalid_inputs_do_not_advance_the_world() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut invalid_config = config();
        invalid_config.sleep.time_threshold = 0.0;
        assert!(matches!(
            GpuRigidSphereWorld::new(
                context.device(),
                context.queue(),
                &[body(0.0, 0.0)],
                &[1.0],
                invalid_config,
            ),
            Err(GpuRigidSphereWorldError::InvalidInput)
        ));
        let mut world = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &[body(0.0, 1.0)],
            &[1.0],
            config(),
        )
        .unwrap();
        assert!(matches!(
            world.step(0.0),
            Err(GpuRigidSphereWorldError::InvalidInput)
        ));
        assert_eq!(world.readback().unwrap()[0].position_inverse_mass[0], 0.0);
        assert!(!world.is_faulted());
    }

    #[test]
    fn faulted_step_requires_reset_and_can_recover_with_new_solver_params() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut world_config = config();
        world_config.solve.iterations = 20_001;
        let initial = [body(-1.0, 1.0), body(1.0, -1.0)];
        let mut world = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &initial,
            &[1.0; 2],
            world_config,
        )
        .unwrap();
        assert!(matches!(
            world.step(0.01),
            Err(GpuRigidSphereWorldError::Solver(
                GpuRigidSphereSolveError::Capacity
            ))
        ));
        assert!(world.is_faulted());
        assert!(matches!(
            world.step(0.01),
            Err(GpuRigidSphereWorldError::Faulted)
        ));
        world.set_solve_params(config().solve).unwrap();
        world.reset(&initial).unwrap();
        assert!(!world.is_faulted());
        assert_eq!(world.step(0.01).unwrap(), 1);
    }

    #[test]
    fn world_material_override_and_clear_change_collision_response() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let initial = [body(-1.0, 1.0), body(1.0, -1.0)];
        let mut world = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &initial,
            &[1.0; 2],
            config(),
        )
        .unwrap();
        for index in 0..2 {
            world
                .set_body_material(index, ColliderMaterial::new(0.0, 1.0))
                .unwrap();
        }
        let _pair_count = world.step(0.01).unwrap();
        let elastic = world.readback().unwrap();
        assert!((elastic[0].linear_velocity[0] + 1.0).abs() < 1e-4);
        assert!((elastic[1].linear_velocity[0] - 1.0).abs() < 1e-4);

        for index in 0..2 {
            world.clear_body_material(index).unwrap();
        }
        world.reset(&initial).unwrap();
        let _pair_count = world.step(0.01).unwrap();
        assert!(
            world
                .readback()
                .unwrap()
                .iter()
                .all(|body| body.linear_velocity[0].abs() < 1e-4)
        );
    }

    #[test]
    fn topology_edits_preserve_state_materials_and_dense_order() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut options = config();
        options.ground_half_extent = Some(10.0);
        let mut world = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &[body(0.0, 0.0), body(5.0, 0.0)],
            &[1.0, 2.0],
            options,
        )
        .unwrap();
        let override_material = ColliderMaterial::new(0.2, 0.8);
        world.set_body_material(1, override_material).unwrap();
        world.set_ground_material(override_material).unwrap();
        let _pairs = world.step(0.01).unwrap();
        let before = world.readback().unwrap();

        assert_eq!(world.append_body(body(10.0, 0.0), 0.5).unwrap(), 2);
        assert_eq!(world.len(), 3);
        assert_same_states(&world.readback().unwrap()[..2], &before);
        assert_eq!(world.radius(2), Some(0.5));
        assert_eq!(
            world.body_material_override(1),
            Some(Some(override_material))
        );
        assert_eq!(world.ground_material_override(), Some(override_material));
        let removed = world.remove_body(1).unwrap();
        assert_same_states(&[removed], &before[1..2]);
        assert_eq!(world.len(), 2);
        assert_eq!(world.radius(1), Some(0.5));
        assert_eq!(world.body_material_override(1), Some(None));
        assert_eq!(world.readback().unwrap()[1].position_inverse_mass[0], 10.0);
        assert!(world.append_body(body(20.0, 0.0), f32::NAN).is_err());
        assert!(world.remove_body(5).is_err());
        assert_eq!(world.len(), 2);
        let _pairs = world.step(0.01).unwrap();

        let _last = world.remove_body(1).unwrap();
        let _first = world.remove_body(0).unwrap();
        assert!(world.is_empty());
        assert_eq!(world.append_body(body(1.0, 0.0), 1.0).unwrap(), 0);
        assert_eq!(world.ground_material_override(), Some(override_material));
    }

    #[test]
    fn topology_edits_preserve_queued_gpu_steps_and_clear_forces() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut world = GpuRigidSphereWorld::new(
            context.device(),
            context.queue(),
            &[body(0.0, 1.0), body(10.0, 2.0), body(20.0, 3.0)],
            &[0.1; 3],
            config(),
        )
        .unwrap();

        let _pairs = world.step(0.01).unwrap();
        assert_eq!(world.append_body(body(30.0, 4.0), 0.1).unwrap(), 3);
        let after_append = world.readback().unwrap();
        for (index, expected) in [0.01, 10.02, 20.03, 30.0].into_iter().enumerate() {
            assert!((after_append[index].position_inverse_mass[0] - expected).abs() < 1e-4);
        }

        world
            .write_forces(
                2,
                GpuRigidBodyForces {
                    force: [100.0, 0.0, 0.0, 0.0],
                    torque: [0.0; 4],
                },
            )
            .unwrap();
        let removed = world.remove_body(1).unwrap();
        assert_same_states(&[removed], &after_append[1..2]);
        let _pairs = world.step(0.01).unwrap();
        let after_remove = world.readback().unwrap();
        for (index, expected) in [0.02, 20.06, 30.04].into_iter().enumerate() {
            assert!((after_remove[index].position_inverse_mass[0] - expected).abs() < 1e-4);
        }
        assert!((after_remove[1].linear_velocity[0] - 3.0).abs() < 1e-4);
    }

    #[test]
    fn grouped_world_rejects_topology_edits() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let initial = [body(0.0, 0.0), body(5.0, 0.0)];
        let mut world = GpuRigidSphereWorld::new_grouped(
            context.device(),
            context.queue(),
            &initial,
            &[1.0; 2],
            &[0, 1],
            config(),
        )
        .unwrap();
        assert!(world.append_body(body(10.0, 0.0), 1.0).is_err());
        assert!(world.remove_body(0).is_err());
        assert_same_states(&world.readback().unwrap(), &initial);
    }

    #[test]
    fn resident_primitive_pairs_and_ground_produce_contacts() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut options = config();
        options.ground_half_extent = Some(10.0);
        let mut box_state = body(0.0, 0.0);
        box_state.position_inverse_mass[2] = 0.4;
        box_state.position_inverse_mass[3] = 0.0;
        box_state.inverse_inertia_sleep = [0.0; 4];
        let mut sphere_state = body(1.4, 0.0);
        sphere_state.position_inverse_mass[2] = 0.4;
        let shapes = [
            GpuRigidShape::Box {
                half_extents: [1.0, 1.0, 0.5],
            },
            GpuRigidShape::Sphere { radius: 0.5 },
        ];
        let mut world = GpuRigidPrimitiveWorld::new_primitives(
            context.device(),
            context.queue(),
            &[box_state, sphere_state],
            &shapes,
            options,
        )
        .unwrap();
        assert_eq!(world.shape(0), Some(shapes[0].clone()));
        assert_eq!(world.step(0.01).unwrap(), 1);
        let contacts = world.readback_contacts().unwrap();
        assert_eq!(contacts.pairs[0].1.depth_hit[1], 1.0);
        assert!((contacts.pairs[0].1.depth_hit[0] - 0.1).abs() < 1e-4);
        assert!(contacts.pairs[0].1.normal[0] > 0.99);
        assert_eq!(contacts.ground[0].depth_hit[1], 1.0);
        assert!((contacts.ground[0].depth_hit[0] - 0.1).abs() < 1e-4);

        let mut second_box = body(1.5, 0.0);
        second_box.position_inverse_mass[2] = 0.4;
        let mut boxes = GpuRigidPrimitiveWorld::new_primitives(
            context.device(),
            context.queue(),
            &[box_state, second_box],
            &[
                GpuRigidShape::Box {
                    half_extents: [1.0; 3],
                },
                GpuRigidShape::Box {
                    half_extents: [1.0; 3],
                },
            ],
            config(),
        )
        .unwrap();
        let _pairs = boxes.step(0.01).unwrap();
        let contact = boxes.readback_contacts().unwrap().pairs[0].1;
        assert_eq!(contact.depth_hit[1], 1.0);
        assert!((contact.depth_hit[0] - 0.5).abs() < 1e-4);
        assert!(contact.normal[0] > 0.99);
    }

    #[test]
    fn resident_primitive_lbvh_refresh_keeps_box_sphere_contacts() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut states = (0..17)
            .map(|index| body(index as f32 * 10.0, 0.0))
            .collect::<Vec<_>>();
        states[1] = body(1.4, 0.0);
        let mut shapes = vec![GpuRigidShape::Sphere { radius: 0.5 }; 17];
        shapes[0] = GpuRigidShape::Box {
            half_extents: [1.0; 3],
        };
        let mut world = GpuRigidPrimitiveWorld::new_primitives(
            context.device(),
            context.queue(),
            &states,
            &shapes,
            config(),
        )
        .unwrap();
        assert!(!world.uses_exhaustive_pairs());
        assert!(world.step(0.01).unwrap() >= 1);
        assert!(
            world
                .readback_contacts()
                .unwrap()
                .pairs
                .iter()
                .any(|(pair, contact)| {
                    pair.a == 0 && pair.b == 1 && contact.depth_hit[1] == 1.0
                })
        );
    }

    #[test]
    fn primitive_sleep_uses_the_selected_bodys_manifold_stride() {
        let airborne = body(0.0, 0.0);
        let mut supported = body(3.0, 0.0);
        supported.position_inverse_mass[2] = 0.49;
        let mut options = config();
        options.ground_half_extent = Some(10.0);
        let shapes = vec![
            GpuRigidShape::Box {
                half_extents: [0.5; 3]
            };
            2
        ];
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let mut world = GpuRigidPrimitiveWorld::new_primitives(
                context.device(),
                context.queue(),
                &[airborne, supported],
                &shapes,
                options,
            )
            .unwrap();
            let _steps = world.step(0.01).unwrap();
            let _steps = world.step(0.01).unwrap();
            let states = world.readback().unwrap();
            assert_eq!(states[0].inverse_inertia_sleep[3], 0.0);
            assert_eq!(
                states[1].inverse_inertia_sleep[3], 1.0,
                "{backend:?}: {states:?}"
            );
            eprintln!("primitive sleep manifold stride passed on {backend:?}");
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn resident_box_ground_uses_four_corner_manifold() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut state = body(0.0, 0.0);
        state.position_inverse_mass[2] = 0.45;
        let mut options = config();
        options.ground_half_extent = Some(2.0);
        options.solve.bias_factor = 0.2;
        options.solve.iterations = 12;
        let shape = GpuRigidShape::Box {
            half_extents: [0.5; 3],
        };
        let mut world = GpuRigidPrimitiveWorld::new_primitives(
            context.device(),
            context.queue(),
            &[state],
            core::slice::from_ref(&shape),
            options,
        )
        .unwrap();
        let _ = world.step(0.01).unwrap();
        let contacts = world.readback_contacts().unwrap();
        assert_eq!(contacts.ground_extra.len(), 1);
        let manifold = [
            contacts.ground[0],
            contacts.ground_extra[0][0],
            contacts.ground_extra[0][1],
            contacts.ground_extra[0][2],
        ];
        assert!(manifold.iter().all(|contact| contact.is_contact()));
        for (index, contact) in manifold.iter().enumerate() {
            assert!((contact.depth_hit[0] - 0.05).abs() < 1e-5);
            assert_eq!(contact.normal[2], 1.0);
            assert!(manifold[..index].iter().all(|other| {
                (other.point[0] - contact.point[0]).abs() > 0.9
                    || (other.point[1] - contact.point[1]).abs() > 0.9
            }));
        }
        let actual = world.readback().unwrap()[0];
        assert!(actual.linear_velocity[2] > 0.0);
        assert!(actual.angular_velocity[0].abs() < 0.1);
        assert!(actual.angular_velocity[1].abs() < 0.1);

        state.angular_velocity[1] = 1.0;
        let mut spinning = GpuRigidPrimitiveWorld::new_primitives(
            context.device(),
            context.queue(),
            &[state],
            core::slice::from_ref(&shape),
            options,
        )
        .unwrap();
        let _ = spinning.step(0.01).unwrap();
        let actual = spinning.readback().unwrap()[0];
        assert!(actual.angular_velocity[1].abs() < 1.0);

        state.angular_velocity[1] = 0.0;
        state.position_inverse_mass[2] = 0.5;
        options.gravity = [0.0, 0.0, -9.81];
        let mut resting = GpuRigidPrimitiveWorld::new_primitives(
            context.device(),
            context.queue(),
            &[state],
            &[shape],
            options,
        )
        .unwrap();
        let _ = resting.step_substeps(0.005, 120).unwrap();
        let actual = resting.readback().unwrap()[0];
        assert!((actual.position_inverse_mass[2] - 0.5).abs() < 0.1);
        assert!(actual.linear_velocity[2].abs() < 0.2);
        assert!(actual.angular_velocity[0].abs() < 0.1);
        assert!(actual.angular_velocity[1].abs() < 0.1);
    }

    #[test]
    fn resident_convex_ground_selects_four_spread_support_points() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut vertices = [-0.5, 0.5]
            .into_iter()
            .flat_map(|x| {
                [-0.5, 0.5]
                    .into_iter()
                    .flat_map(move |y| [-0.5, 0.5].into_iter().map(move |z| [x, y, z]))
            })
            .collect::<Vec<_>>();
        vertices.push([0.0, 0.0, -0.5]);
        let shape = GpuRigidShape::Convex { vertices };
        let mut state = body(0.0, 0.0);
        state.position_inverse_mass[2] = 0.45;
        let mut options = config();
        options.ground_half_extent = Some(2.0);
        options.solve.bias_factor = 0.2;
        options.solve.iterations = 12;
        let mut world = GpuRigidPrimitiveWorld::new_primitives(
            context.device(),
            context.queue(),
            &[state],
            core::slice::from_ref(&shape),
            options,
        )
        .unwrap();
        let _ = world.step(0.01).unwrap();
        let contacts = world.readback_contacts().unwrap();
        let manifold = [
            contacts.ground[0],
            contacts.ground_extra[0][0],
            contacts.ground_extra[0][1],
            contacts.ground_extra[0][2],
        ];
        assert!(manifold.iter().all(|contact| contact.is_contact()));
        for (index, contact) in manifold.iter().enumerate() {
            assert!((contact.depth_hit[0] - 0.05).abs() < 1e-5);
            assert!((contact.point[0].abs() - 0.5).abs() < 1e-5);
            assert!((contact.point[1].abs() - 0.5).abs() < 1e-5);
            assert!(manifold[..index].iter().all(|other| {
                (other.point[0] - contact.point[0]).abs() > 0.9
                    || (other.point[1] - contact.point[1]).abs() > 0.9
            }));
        }
        let actual = world.readback().unwrap()[0];
        assert!(actual.linear_velocity[2] > 0.0);
        assert!(actual.angular_velocity[0].abs() < 0.1);
        assert!(actual.angular_velocity[1].abs() < 0.1);

        state.position_inverse_mass[2] = 0.5;
        options.gravity = [0.0, 0.0, -9.81];
        let mut resting = GpuRigidPrimitiveWorld::new_primitives(
            context.device(),
            context.queue(),
            &[state],
            &[shape],
            options,
        )
        .unwrap();
        let _ = resting.step_substeps(0.005, 120).unwrap();
        let actual = resting.readback().unwrap()[0];
        assert!((actual.position_inverse_mass[2] - 0.5).abs() < 0.1);
        assert!(actual.linear_velocity[2].abs() < 0.2);
        assert!(actual.angular_velocity[0].abs() < 0.1);
        assert!(actual.angular_velocity[1].abs() < 0.1);
    }

    #[test]
    fn resident_box_pair_clips_four_face_contacts() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let box_shape = GpuRigidShape::Box {
            half_extents: [0.5; 3],
        };
        for orientation in [
            [0.0, 0.0, 0.0, 1.0],
            [
                (core::f32::consts::FRAC_PI_4 * 0.5).sin(),
                0.0,
                0.0,
                (core::f32::consts::FRAC_PI_4 * 0.5).cos(),
            ],
        ] {
            let mut a = body(0.0, 0.0);
            let mut b = body(0.9, 0.0);
            a.position_inverse_mass[3] = 0.0;
            b.position_inverse_mass[3] = 0.0;
            a.inverse_inertia_sleep = [0.0; 4];
            b.inverse_inertia_sleep = [0.0; 4];
            b.orientation = orientation;
            let mut world = GpuRigidPrimitiveWorld::new_primitives(
                context.device(),
                context.queue(),
                &[a, b],
                &[box_shape.clone(), box_shape.clone()],
                config(),
            )
            .unwrap();
            let _ = world.step(0.01).unwrap();
            let contacts = world.readback_contacts().unwrap();
            assert_eq!(contacts.pair_extra.len(), 1);
            let manifold = [
                contacts.pairs[0].1,
                contacts.pair_extra[0][0],
                contacts.pair_extra[0][1],
                contacts.pair_extra[0][2],
            ];
            assert!(
                manifold.iter().all(|contact| contact.is_contact()),
                "orientation: {orientation:?}, manifold: {manifold:?}"
            );
            for (index, contact) in manifold.iter().enumerate() {
                assert!((contact.depth_hit[0] - 0.1).abs() < 1e-4);
                assert!(contact.normal[0] > 0.99);
                assert!(manifold[..index].iter().all(|other| {
                    (other.point[1] - contact.point[1]).abs() > 0.1
                        || (other.point[2] - contact.point[2]).abs() > 0.1
                }));
            }
        }

        let mut a = body(0.0, 0.0);
        let mut b = body(0.0, 0.0);
        a.position_inverse_mass[3] = 0.0;
        b.position_inverse_mass = [0.9 / 2.0f32.sqrt(), 0.9 / 2.0f32.sqrt(), 3.0, 0.0];
        a.inverse_inertia_sleep = [0.0; 4];
        b.inverse_inertia_sleep = [0.0; 4];
        b.orientation = [
            0.0,
            0.0,
            (core::f32::consts::FRAC_PI_4 * 0.5).sin(),
            (core::f32::consts::FRAC_PI_4 * 0.5).cos(),
        ];
        let mut rotated = GpuRigidPrimitiveWorld::new_primitives(
            context.device(),
            context.queue(),
            &[a, b],
            &[box_shape.clone(), box_shape.clone()],
            config(),
        )
        .unwrap();
        let _ = rotated.step(0.01).unwrap();
        let contacts = rotated.readback_contacts().unwrap();
        let face_contacts = core::iter::once(&contacts.pairs[0].1)
            .chain(contacts.pair_extra[0].iter())
            .filter(|contact| contact.is_contact())
            .count();
        assert!(face_contacts >= 2, "contacts: {contacts:?}");
        assert!(contacts.pairs[0].1.normal[0] > 0.5);
        assert!(contacts.pairs[0].1.normal[1] > 0.5);

        let mut support = body(0.0, 0.0);
        support.position_inverse_mass[2] = 0.5;
        support.position_inverse_mass[3] = 0.0;
        support.inverse_inertia_sleep = [0.0; 4];
        let mut supported = body(0.0, 0.0);
        supported.position_inverse_mass[2] = 1.45;
        let mut options = config();
        options.gravity = [0.0, 0.0, -9.81];
        options.solve.bias_factor = 0.2;
        options.solve.iterations = 12;
        let mut stack = GpuRigidPrimitiveWorld::new_primitives(
            context.device(),
            context.queue(),
            &[support, supported],
            &[box_shape.clone(), box_shape],
            options,
        )
        .unwrap();
        let _ = stack.step_substeps(0.005, 120).unwrap();
        let top = stack.readback().unwrap()[1];
        assert!((top.position_inverse_mass[2] - 1.5).abs() < 0.1);
        assert!(top.linear_velocity[2].abs() < 0.2);
        assert!(top.angular_velocity[0].abs() < 0.1);
        assert!(top.angular_velocity[1].abs() < 0.1);
    }

    #[test]
    fn resident_primitive_lbvh_uses_oriented_shape_bounds() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let quarter_turn = core::f32::consts::FRAC_1_SQRT_2;
        let around_z = [0.0, 0.0, quarter_turn, quarter_turn];
        let around_y = [0.0, quarter_turn, 0.0, quarter_turn];
        let elongated_hull = GpuRigidShape::Convex {
            vertices: [-4.0, 4.0]
                .into_iter()
                .flat_map(|x| {
                    [-0.1, 0.1]
                        .into_iter()
                        .flat_map(move |y| [-0.1, 0.1].into_iter().map(move |z| [x, y, z]))
                })
                .collect(),
        };
        let cases = [
            (
                GpuRigidShape::Box {
                    half_extents: [4.0, 0.1, 0.1],
                },
                around_z,
                [1.0, 0.0],
            ),
            (
                GpuRigidShape::Capsule {
                    radius: 0.1,
                    half_length: 4.0,
                },
                around_y,
                [0.0, 1.0],
            ),
            (
                GpuRigidShape::Cylinder {
                    radius: 0.1,
                    half_length: 4.0,
                },
                around_y,
                [0.0, 1.0],
            ),
            (
                GpuRigidShape::Cone {
                    radius: 0.1,
                    half_length: 4.0,
                },
                around_y,
                [0.0, 1.0],
            ),
            (elongated_hull, around_z, [1.0, 0.0]),
        ];
        for (shape, orientation, probe) in cases {
            let mut states = (0..17)
                .map(|index| body(index as f32 * 100.0, 0.0))
                .collect::<Vec<_>>();
            states[0].orientation = orientation;
            states[1] = body(probe[0], 0.0);
            states[1].position_inverse_mass[1] = probe[1];
            let mut shapes = vec![GpuRigidShape::Sphere { radius: 0.1 }; states.len()];
            shapes[0] = shape.clone();
            let mut world = GpuRigidPrimitiveWorld::new_primitives(
                context.device(),
                context.queue(),
                &states,
                &shapes,
                config(),
            )
            .unwrap();
            assert!(!world.uses_exhaustive_pairs());
            assert_eq!(world.step(0.01).unwrap(), 0, "shape: {shape:?}");
            world.write_body(1, body(0.0, 0.0)).unwrap();
            assert_eq!(world.step(0.01).unwrap(), 1, "shape: {shape:?}");
        }
    }

    #[test]
    fn resident_capsule_contacts_cover_sphere_box_capsule_and_ground() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let capsule = GpuRigidShape::Capsule {
            radius: 0.5,
            half_length: 1.0,
        };
        let cases = [
            (
                [capsule.clone(), GpuRigidShape::Sphere { radius: 0.25 }],
                0.65,
                0.1,
            ),
            ([capsule.clone(), capsule.clone()], 0.8, 0.2),
            (
                [
                    GpuRigidShape::Box {
                        half_extents: [0.5; 3],
                    },
                    capsule.clone(),
                ],
                0.9,
                0.1,
            ),
        ];
        for (shapes, x, expected_depth) in cases {
            let mut first = body(0.0, 0.0);
            first.position_inverse_mass[3] = 0.0;
            first.inverse_inertia_sleep = [0.0; 4];
            let mut second = body(x, 0.0);
            second.position_inverse_mass[3] = 0.0;
            second.inverse_inertia_sleep = [0.0; 4];
            let mut world = GpuRigidPrimitiveWorld::new_primitives(
                context.device(),
                context.queue(),
                &[first, second],
                &shapes,
                config(),
            )
            .unwrap();
            let _pairs = world.step(0.01).unwrap();
            let contact = world.readback_contacts().unwrap().pairs[0].1;
            assert_eq!(contact.depth_hit[1], 1.0, "shapes: {shapes:?}");
            assert!((contact.depth_hit[0] - expected_depth).abs() < 1e-3);
            assert!(contact.normal[0] > 0.99);
        }

        let mut rotated = body(0.0, 0.0);
        rotated.position_inverse_mass[3] = 0.0;
        rotated.inverse_inertia_sleep = [0.0; 4];
        rotated.orientation = [
            0.0,
            core::f32::consts::FRAC_1_SQRT_2,
            0.0,
            core::f32::consts::FRAC_1_SQRT_2,
        ];
        let mut sphere = body(1.7, 0.0);
        sphere.position_inverse_mass[3] = 0.0;
        sphere.inverse_inertia_sleep = [0.0; 4];
        let mut world = GpuRigidPrimitiveWorld::new_primitives(
            context.device(),
            context.queue(),
            &[rotated, sphere],
            &[capsule.clone(), GpuRigidShape::Sphere { radius: 0.25 }],
            config(),
        )
        .unwrap();
        let _pairs = world.step(0.01).unwrap();
        let contact = world.readback_contacts().unwrap().pairs[0].1;
        assert_eq!(contact.depth_hit[1], 1.0);
        assert!((contact.depth_hit[0] - 0.05).abs() < 1e-3);

        let mut ground_config = config();
        ground_config.ground_half_extent = Some(10.0);
        let mut state = body(0.0, 0.0);
        state.position_inverse_mass = [0.0, 0.0, 1.3, 0.0];
        state.inverse_inertia_sleep = [0.0; 4];
        let mut ground_world = GpuRigidPrimitiveWorld::new_primitives(
            context.device(),
            context.queue(),
            &[state],
            &[capsule],
            ground_config,
        )
        .unwrap();
        let _pairs = ground_world.step(0.01).unwrap();
        let contact = ground_world.readback_contacts().unwrap().ground[0];
        assert_eq!(contact.depth_hit[1], 1.0);
        assert!((contact.depth_hit[0] - 0.2).abs() < 1e-3);
    }

    #[test]
    fn resident_capsule_convex_packed_contacts_on_gpu_backends() {
        let hull = GpuRigidShape::Convex {
            vertices: [-0.5, 0.5]
                .into_iter()
                .flat_map(|x| {
                    [-0.5, 0.5]
                        .into_iter()
                        .flat_map(move |y| [-0.5, 0.5].into_iter().map(move |z| [x, y, z]))
                })
                .collect(),
        };
        let cases = [
            (false, false, 0.0f32),
            (false, true, 0.0),
            (false, false, 0.01),
            (false, true, 0.01),
            (true, false, 0.0),
            (true, true, 0.0),
        ];
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let mut states = Vec::new();
            let mut shapes = Vec::new();
            for (group, &(corner, reversed, pitch)) in cases.iter().enumerate() {
                let offset = group as f32 * 10.0;
                let mut support = body(offset, 0.0);
                support.position_inverse_mass[3] = 0.0;
                support.inverse_inertia_sleep = [0.0; 4];
                support.orientation = [0.0, (pitch * 0.5).sin(), 0.0, (pitch * 0.5).cos()];
                let mut rounded = body(offset + if corner { 0.8 } else { 0.9 }, 0.0);
                rounded.position_inverse_mass[3] = 0.0;
                rounded.inverse_inertia_sleep = [0.0; 4];
                if corner {
                    rounded.position_inverse_mass[1] = 0.8;
                    rounded.position_inverse_mass[2] = 3.7;
                }
                let capsule = GpuRigidShape::Capsule {
                    radius: 0.5,
                    half_length: if corner { 0.1 } else { 2.0 },
                };
                if reversed {
                    states.extend([rounded, support]);
                    shapes.extend([capsule, hull.clone()]);
                } else {
                    states.extend([support, rounded]);
                    shapes.extend([hull.clone(), capsule]);
                }
            }
            let mut support = body(60.0, 0.0);
            support.position_inverse_mass[3] = 0.0;
            support.inverse_inertia_sleep = [0.0; 4];
            states.extend([support, body(60.9, 0.0)]);
            shapes.extend([
                hull.clone(),
                GpuRigidShape::Capsule {
                    radius: 0.5,
                    half_length: 1.0,
                },
            ]);
            let mut settings = config();
            settings.gravity = [-9.81, 0.0, 0.0];
            settings.solve.bias_factor = 0.2;
            settings.solve.iterations = 12;
            settings.sleep.enabled = false;
            let mut world = GpuRigidPrimitiveWorld::new_primitives(
                context.device(),
                context.queue(),
                &states,
                &shapes,
                settings,
            )
            .unwrap();
            let _ = world.step(1.0 / 120.0).unwrap();
            let contacts = world.readback_contacts().unwrap();
            for (group, &(corner, reversed, pitch)) in cases.iter().enumerate() {
                let index = contacts
                    .pairs
                    .iter()
                    .position(|(pair, _)| {
                        pair.a == group as u32 * 2 && pair.b == group as u32 * 2 + 1
                    })
                    .unwrap();
                let first = contacts.pairs[index].1;
                assert!(first.is_contact(), "{backend:?}, group {group}: {first:?}");
                let sign = if reversed { -1.0 } else { 1.0 };
                if corner {
                    let distance = (0.3f32 * 0.3 * 2.0 + 0.1 * 0.1).sqrt();
                    assert!((first.depth_hit[0] - (0.5 - distance)).abs() < 1e-4);
                    for (axis, delta) in [0.3, 0.3, 0.1].into_iter().enumerate() {
                        assert!((first.normal[axis] * sign - delta / distance).abs() < 1e-4);
                        let vertex = [group as f32 * 10.0 + 0.5, 0.5, 3.5][axis];
                        assert!(
                            (first.point[axis]
                                - vertex
                                - delta / distance * (distance - 0.5) * 0.5)
                                .abs()
                                < 1e-4
                        );
                    }
                    assert!(
                        contacts.pair_extra[index]
                            .iter()
                            .all(|point| !point.is_contact())
                    );
                } else {
                    let second = contacts.pair_extra[index][0];
                    assert!(second.is_contact());
                    assert!((first.point[2] - second.point[2]).abs() > 0.9);
                    for point in [first, second] {
                        assert!((point.normal[0] * sign - pitch.cos()).abs() < 1e-4);
                        assert!((point.normal[2] * sign + pitch.sin()).abs() < 1e-4);
                    }
                }
            }
            let _ = world.step_substeps(1.0 / 120.0, 119).unwrap();
            let state = world.readback().unwrap()[13];
            assert!(
                state.position_inverse_mass[0] > 60.8,
                "{backend:?}: {state:?}"
            );
            assert!(
                state.linear_velocity[0].abs() < 0.2 && state.angular_velocity[1].abs() < 0.2,
                "{backend:?}: {state:?}"
            );
            let contacts = world.readback_contacts().unwrap();
            let index = contacts
                .pairs
                .iter()
                .position(|(pair, _)| pair.a == 12 && pair.b == 13)
                .unwrap();
            assert!(
                contacts.pairs[index].1.is_contact() && contacts.pair_extra[index][0].is_contact()
            );
            eprintln!("packed capsule convex contact and support passed on {backend:?}");
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn resident_capsule_convex_corner_uses_segment_vertex_axis() {
        let context = GpuContactDevice::new().unwrap();
        let hull = GpuRigidShape::Convex {
            vertices: [-0.5, 0.5]
                .into_iter()
                .flat_map(|x| {
                    [-0.5, 0.5]
                        .into_iter()
                        .flat_map(move |y| [-0.5, 0.5].into_iter().map(move |z| [x, y, z]))
                })
                .collect(),
        };
        let capsule = GpuRigidShape::Capsule {
            radius: 0.5,
            half_length: 0.1,
        };
        let mut hull_state = body(0.0, 0.0);
        hull_state.position_inverse_mass[3] = 0.0;
        hull_state.inverse_inertia_sleep = [0.0; 4];
        let mut capsule_state = hull_state;
        capsule_state.position_inverse_mass = [0.8, 0.8, 3.7, 0.0];
        for reversed in [false, true] {
            let (states, shapes) = if reversed {
                ([capsule_state, hull_state], [capsule.clone(), hull.clone()])
            } else {
                ([hull_state, capsule_state], [hull.clone(), capsule.clone()])
            };
            let mut world = GpuRigidPrimitiveWorld::new_primitives(
                context.device(),
                context.queue(),
                &states,
                &shapes,
                config(),
            )
            .unwrap();
            let _ = world.step(0.001).unwrap();
            let contacts = world.readback_contacts().unwrap();
            let contact = contacts.pairs[0].1;
            let distance = (0.3f32 * 0.3 * 2.0 + 0.1 * 0.1).sqrt();
            assert!(contact.is_contact());
            assert!(
                (contact.depth_hit[0] - (0.5 - distance)).abs() < 1e-4,
                "{contact:?}"
            );
            let sign = if reversed { -1.0 } else { 1.0 };
            for (axis, delta) in [0.3, 0.3, 0.1].into_iter().enumerate() {
                assert!(
                    (contact.normal[axis] * sign - delta / distance).abs() < 1e-4,
                    "{contact:?}"
                );
                let corner = [0.5, 0.5, 3.5][axis];
                let expected_point = corner + delta / distance * (distance - 0.5) * 0.5;
                assert!(
                    (contact.point[axis] - expected_point).abs() < 1e-4,
                    "{contact:?}"
                );
            }
            assert!(
                contacts.pair_extra[0]
                    .iter()
                    .all(|contact| !contact.is_contact())
            );
        }
    }

    #[test]
    fn resident_capsule_convex_face_uses_two_clipped_contact_points() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let box_shape = GpuRigidShape::Convex {
            vertices: [-0.5, 0.5]
                .into_iter()
                .flat_map(|x| {
                    [-0.5, 0.5]
                        .into_iter()
                        .flat_map(move |y| [-0.5, 0.5].into_iter().map(move |z| [x, y, z]))
                })
                .collect(),
        };
        let capsule = GpuRigidShape::Capsule {
            radius: 0.5,
            half_length: 2.0,
        };
        for reversed in [false, true] {
            let (states, shapes) = if reversed {
                (
                    [body(0.9, 0.0), body(0.0, 0.0)],
                    [capsule.clone(), box_shape.clone()],
                )
            } else {
                (
                    [body(0.0, 0.0), body(0.9, 0.0)],
                    [box_shape.clone(), capsule.clone()],
                )
            };
            let mut world = GpuRigidPrimitiveWorld::new_primitives(
                context.device(),
                context.queue(),
                &states,
                &shapes,
                config(),
            )
            .unwrap();
            assert_eq!(world.step(0.01).unwrap(), 1);
            let contacts = world.readback_contacts().unwrap();
            let first = contacts.pairs[0].1;
            let second = contacts.pair_extra[0][0];
            assert!(first.is_contact());
            assert!(second.is_contact());
            assert!((first.depth_hit[0] - 0.1).abs() < 1e-3);
            assert!((second.depth_hit[0] - 0.1).abs() < 1e-3);
            assert!(((first.point[2] - 3.0).abs() - 0.5).abs() < 1e-4);
            assert!(((second.point[2] - 3.0).abs() - 0.5).abs() < 1e-4);
            assert!((first.point[2] - second.point[2]).abs() > 0.9);
            let expected_normal = if reversed { -1.0 } else { 1.0 };
            assert!(first.normal[0] * expected_normal > 0.99);
            assert!(second.normal[0] * expected_normal > 0.99);
            assert!(!contacts.pair_extra[0][1].is_contact());
            assert!(!contacts.pair_extra[0][2].is_contact());
        }
    }

    #[test]
    fn resident_capsule_box_face_uses_two_distinct_contact_points() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let box_shape = GpuRigidShape::Box {
            half_extents: [0.5; 3],
        };
        let capsule = GpuRigidShape::Capsule {
            radius: 0.5,
            half_length: 1.0,
        };
        for reversed in [false, true] {
            let (states, shapes) = if reversed {
                (
                    [body(0.9, 0.0), body(0.0, 0.0)],
                    [capsule.clone(), box_shape.clone()],
                )
            } else {
                (
                    [body(0.0, 0.0), body(0.9, 0.0)],
                    [box_shape.clone(), capsule.clone()],
                )
            };
            let mut world = GpuRigidPrimitiveWorld::new_primitives(
                context.device(),
                context.queue(),
                &states,
                &shapes,
                config(),
            )
            .unwrap();
            assert_eq!(world.step(0.01).unwrap(), 1);
            let contacts = world.readback_contacts().unwrap();
            let first = contacts.pairs[0].1;
            let second = contacts.pair_extra[0][0];
            assert!(first.is_contact());
            assert!(second.is_contact());
            assert!((first.depth_hit[0] - 0.1).abs() < 1e-3);
            assert!((second.depth_hit[0] - 0.1).abs() < 1e-3);
            assert!((first.point[2] - second.point[2]).abs() > 0.9);
            let expected_normal = if reversed { -1.0 } else { 1.0 };
            assert!(first.normal[0] * expected_normal > 0.99);
            assert!(second.normal[0] * expected_normal > 0.99);
            assert!(!contacts.pair_extra[0][1].is_contact());
            assert!(!contacts.pair_extra[0][2].is_contact());
        }
    }

    #[test]
    fn resident_capsule_convex_face_impulses_support_dynamic_capsule() {
        let Ok(context) = GpuContactDevice::new_with_backends(wgpu::Backends::VULKAN)
            .or_else(|_| GpuContactDevice::new())
        else {
            return;
        };
        let mut fixed_box = body(0.0, 0.0);
        fixed_box.position_inverse_mass[3] = 0.0;
        fixed_box.inverse_inertia_sleep = [0.0; 4];
        let capsule = body(0.9, 0.0);
        let mut settings = config();
        settings.gravity = [-9.81, 0.0, 0.0];
        settings.solve.bias_factor = 0.2;
        settings.solve.iterations = 12;
        settings.sleep.enabled = false;
        let mut world = GpuRigidPrimitiveWorld::new_primitives(
            context.device(),
            context.queue(),
            &[fixed_box, capsule],
            &[
                GpuRigidShape::Convex {
                    vertices: [-0.5, 0.5]
                        .into_iter()
                        .flat_map(|x| {
                            [-0.5, 0.5]
                                .into_iter()
                                .flat_map(move |y| [-0.5, 0.5].into_iter().map(move |z| [x, y, z]))
                        })
                        .collect(),
                },
                GpuRigidShape::Capsule {
                    radius: 0.5,
                    half_length: 1.0,
                },
            ],
            settings,
        )
        .unwrap();
        assert_eq!(world.step_substeps(1.0 / 120.0, 120).unwrap(), 1);
        let state = world.readback().unwrap()[1];
        assert!(state.position_inverse_mass[0] > 0.8, "{state:?}");
        assert!(state.linear_velocity[0].abs() < 0.2, "{state:?}");
        assert!(state.angular_velocity[1].abs() < 0.2, "{state:?}");
        let contacts = world.readback_contacts().unwrap();
        assert!(contacts.pairs[0].1.is_contact());
        assert!(
            contacts.pair_extra[0][0].is_contact(),
            "{state:?}, {contacts:?}"
        );
    }

    #[test]
    fn resident_capsule_box_face_impulses_support_dynamic_capsule() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut fixed_box = body(0.0, 0.0);
        fixed_box.position_inverse_mass[3] = 0.0;
        fixed_box.inverse_inertia_sleep = [0.0; 4];
        let capsule = body(0.9, 0.0);
        let mut settings = config();
        settings.gravity = [-9.81, 0.0, 0.0];
        settings.solve.bias_factor = 0.2;
        settings.solve.iterations = 12;
        settings.sleep.enabled = false;
        let mut world = GpuRigidPrimitiveWorld::new_primitives(
            context.device(),
            context.queue(),
            &[fixed_box, capsule],
            &[
                GpuRigidShape::Box {
                    half_extents: [0.5; 3],
                },
                GpuRigidShape::Capsule {
                    radius: 0.5,
                    half_length: 1.0,
                },
            ],
            settings,
        )
        .unwrap();
        assert_eq!(world.step_substeps(1.0 / 120.0, 120).unwrap(), 1);
        let state = world.readback().unwrap()[1];
        assert!(state.position_inverse_mass[0] > 0.8, "{state:?}");
        assert!(state.linear_velocity[0].abs() < 0.2, "{state:?}");
        assert!(state.angular_velocity[1].abs() < 0.2, "{state:?}");
        let contacts = world.readback_contacts().unwrap();
        assert!(contacts.pairs[0].1.is_contact());
        assert!(contacts.pair_extra[0][0].is_contact());
    }

    #[test]
    fn resident_parallel_capsules_use_two_side_contacts() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let shape = GpuRigidShape::Capsule {
            radius: 0.5,
            half_length: 1.0,
        };
        for inverted in [false, true] {
            let first = body(0.0, 0.0);
            let mut second = body(0.8, 0.0);
            if inverted {
                second.orientation = [1.0, 0.0, 0.0, 0.0];
            }
            let mut world = GpuRigidPrimitiveWorld::new_primitives(
                context.device(),
                context.queue(),
                &[first, second],
                &[shape.clone(), shape.clone()],
                config(),
            )
            .unwrap();
            assert_eq!(world.step(0.01).unwrap(), 1);
            let contacts = world.readback_contacts().unwrap();
            let first_contact = contacts.pairs[0].1;
            let second_contact = contacts.pair_extra[0][0];
            assert!(first_contact.is_contact());
            assert!(second_contact.is_contact());
            assert!((first_contact.depth_hit[0] - 0.2).abs() < 1e-3);
            assert!((second_contact.depth_hit[0] - 0.2).abs() < 1e-3);
            assert!((first_contact.point[2] - second_contact.point[2]).abs() > 1.9);
            assert!(first_contact.normal[0] > 0.99);
            assert!(second_contact.normal[0] > 0.99);
            assert!(!contacts.pair_extra[0][1].is_contact());
        }
    }

    #[test]
    fn resident_parallel_cylinders_use_side_manifolds() {
        let Ok(context) = GpuContactDevice::new_with_backends(wgpu::Backends::VULKAN) else {
            return;
        };
        let cylinder = GpuRigidShape::Cylinder {
            radius: 0.5,
            half_length: 1.0,
        };
        let capsule = GpuRigidShape::Capsule {
            radius: 0.5,
            half_length: 1.0,
        };
        let box_shape = GpuRigidShape::Box {
            half_extents: [0.5; 3],
        };
        for (first, second, separation) in [
            (cylinder.clone(), cylinder.clone(), 0.8),
            (cylinder.clone(), capsule.clone(), 0.8),
            (capsule, cylinder.clone(), 0.8),
            (cylinder.clone(), box_shape.clone(), 0.9),
            (box_shape, cylinder.clone(), 0.9),
        ] {
            let mut world = GpuRigidPrimitiveWorld::new_primitives(
                context.device(),
                context.queue(),
                &[body(0.0, 0.0), body(separation, 0.0)],
                &[first, second],
                config(),
            )
            .unwrap();
            assert_eq!(world.step(0.01).unwrap(), 1);
            let contacts = world.readback_contacts().unwrap();
            let first = contacts.pairs[0].1;
            let second = contacts.pair_extra[0][0];
            assert!(first.is_contact(), "{contacts:?}");
            assert!(second.is_contact(), "{contacts:?}");
            assert!((first.point[2] - second.point[2]).abs() > 0.9);
            assert!(first.normal[0] > 0.98);
            assert!(second.normal[0] > 0.98);
            assert!(!contacts.pair_extra[0][1].is_contact());
        }

        let mut above = body(0.0, 0.0);
        above.position_inverse_mass[2] = 1.9;
        let mut world = GpuRigidPrimitiveWorld::new_primitives(
            context.device(),
            context.queue(),
            &[body(0.0, 0.0), above],
            &[cylinder.clone(), cylinder],
            config(),
        )
        .unwrap();
        assert_eq!(world.step(0.01).unwrap(), 1);
        let contacts = world.readback_contacts().unwrap();
        assert!(contacts.pairs[0].1.is_contact());
        assert!(!contacts.pair_extra[0][0].is_contact());
    }

    #[test]
    fn resident_linear_side_manifolds_run_on_available_backends() {
        let box_shape = GpuRigidShape::Box {
            half_extents: [0.5; 3],
        };
        let capsule = GpuRigidShape::Capsule {
            radius: 0.5,
            half_length: 1.0,
        };
        let cylinder = GpuRigidShape::Cylinder {
            radius: 0.5,
            half_length: 1.0,
        };
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            for (states, shapes) in [
                (
                    [body(0.0, 0.0), body(0.9, 0.0)],
                    [box_shape.clone(), capsule.clone()],
                ),
                (
                    [body(0.0, 0.0), body(0.8, 0.0)],
                    [capsule.clone(), capsule.clone()],
                ),
                (
                    [body(0.0, 0.0), body(0.8, 0.0)],
                    [cylinder.clone(), cylinder.clone()],
                ),
                (
                    [body(0.0, 0.0), body(0.9, 0.0)],
                    [box_shape.clone(), cylinder.clone()],
                ),
            ] {
                let mut world = GpuRigidPrimitiveWorld::new_primitives(
                    context.device(),
                    context.queue(),
                    &states,
                    &shapes,
                    config(),
                )
                .unwrap();
                let _count = world.step(0.01).unwrap();
                let contacts = world.readback_contacts().unwrap();
                assert!(contacts.pairs[0].1.is_contact());
                assert!(contacts.pair_extra[0][0].is_contact());
            }
        }
    }

    #[test]
    fn resident_capsule_ground_impulse_supports_dynamic_body() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut options = config();
        options.gravity = [0.0, 0.0, -9.81];
        options.ground_half_extent = Some(10.0);
        options.solve.bias_factor = 0.2;
        let mut state = body(0.0, 0.0);
        state.position_inverse_mass[2] = 2.0;
        let mut world = GpuRigidPrimitiveWorld::new_primitives(
            context.device(),
            context.queue(),
            &[state],
            &[GpuRigidShape::Capsule {
                radius: 0.5,
                half_length: 1.0,
            }],
            options,
        )
        .unwrap();
        for _ in 0..120 {
            let _pairs = world.step(1.0 / 120.0).unwrap();
        }
        let state = world.readback().unwrap()[0];
        assert!(state.position_inverse_mass[2] > 1.4);
        assert!(state.position_inverse_mass[2] < 1.6);
        assert!(state.linear_velocity[2].abs() < 0.2);
    }

    #[test]
    fn resident_convex_pairs_clip_face_manifolds_on_gpu() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let vertices = [-0.5, 0.5]
            .into_iter()
            .flat_map(|x| {
                [-0.5, 0.5]
                    .into_iter()
                    .flat_map(move |y| [-0.5, 0.5].into_iter().map(move |z| [x, y, z]))
            })
            .chain(core::iter::once([0.0, 0.0, -0.5]))
            .collect::<Vec<_>>();
        let hull = GpuRigidShape::Convex { vertices };
        let box_shape = GpuRigidShape::Box {
            half_extents: [0.5; 3],
        };
        for shapes in [
            [hull.clone(), hull.clone()],
            [box_shape.clone(), hull.clone()],
            [hull.clone(), box_shape],
        ] {
            let mut support = body(0.0, 0.0);
            support.position_inverse_mass = [0.0, 0.0, 0.5, 0.0];
            support.inverse_inertia_sleep = [0.0; 4];
            let mut top = body(0.0, 0.0);
            top.position_inverse_mass[2] = 1.45;
            let mut world = GpuRigidPrimitiveWorld::new_primitives(
                context.device(),
                context.queue(),
                &[support, top],
                &shapes,
                config(),
            )
            .unwrap();
            let _ = world.step(0.01).unwrap();
            let contacts = world.readback_contacts().unwrap();
            let manifold = [
                contacts.pairs[0].1,
                contacts.pair_extra[0][0],
                contacts.pair_extra[0][1],
                contacts.pair_extra[0][2],
            ];
            assert!(
                manifold.iter().all(|contact| contact.is_contact()),
                "shapes: {shapes:?}, contacts: {contacts:?}"
            );
            for (index, contact) in manifold.iter().enumerate() {
                assert!((contact.depth_hit[0] - 0.05).abs() < 1e-4);
                assert!(contact.normal[2] > 0.99);
                assert!(manifold[..index].iter().all(|other| {
                    (other.point[0] - contact.point[0]).abs() > 0.9
                        || (other.point[1] - contact.point[1]).abs() > 0.9
                }));
            }
        }

        let octagon = GpuRigidShape::Convex {
            vertices: (0..8)
                .flat_map(|corner| {
                    let angle = core::f32::consts::TAU * corner as f32 / 8.0;
                    [-0.5, 0.5]
                        .into_iter()
                        .map(move |z| [0.5 * angle.cos(), 0.5 * angle.sin(), z])
                })
                .collect(),
        };
        let mut support = body(0.0, 0.0);
        support.position_inverse_mass = [0.0, 0.0, 0.5, 0.0];
        support.inverse_inertia_sleep = [0.0; 4];
        let mut top = body(0.0, 0.0);
        top.position_inverse_mass = [0.0, 0.0, 1.45, 0.0];
        top.inverse_inertia_sleep = [0.0; 4];
        let mut octagonal_pair = GpuRigidPrimitiveWorld::new_primitives(
            context.device(),
            context.queue(),
            &[support, top],
            &[octagon.clone(), octagon],
            config(),
        )
        .unwrap();
        let _ = octagonal_pair.step(0.01).unwrap();
        let contacts = octagonal_pair.readback_contacts().unwrap();
        let manifold = [
            contacts.pairs[0].1,
            contacts.pair_extra[0][0],
            contacts.pair_extra[0][1],
            contacts.pair_extra[0][2],
        ];
        assert!(
            manifold.iter().all(|contact| contact.is_contact()),
            "octagonal contacts: {contacts:?}"
        );
        for (index, contact) in manifold.iter().enumerate() {
            assert!(manifold[..index].iter().all(|other| {
                (other.point[0] - contact.point[0]).abs() > 0.1
                    || (other.point[1] - contact.point[1]).abs() > 0.1
            }));
        }

        let mut support = body(0.0, 0.0);
        support.position_inverse_mass = [0.0, 0.0, 0.5, 0.0];
        support.inverse_inertia_sleep = [0.0; 4];
        let mut top = body(0.0, 0.0);
        top.position_inverse_mass[2] = 1.45;
        top.orientation = [
            0.0,
            0.0,
            (core::f32::consts::FRAC_PI_4 * 0.5).sin(),
            (core::f32::consts::FRAC_PI_4 * 0.5).cos(),
        ];
        let mut options = config();
        options.gravity = [0.0, 0.0, -9.81];
        options.solve.bias_factor = 0.2;
        options.solve.iterations = 12;
        let mut stack = GpuRigidPrimitiveWorld::new_primitives(
            context.device(),
            context.queue(),
            &[support, top],
            &[hull.clone(), hull],
            options,
        )
        .unwrap();
        let _ = stack.step(0.005).unwrap();
        let contacts = stack.readback_contacts().unwrap();
        let count = core::iter::once(&contacts.pairs[0].1)
            .chain(contacts.pair_extra[0].iter())
            .filter(|contact| contact.is_contact())
            .count();
        assert!(count >= 2, "rotated convex contacts: {contacts:?}");
        let _ = stack.step_substeps(0.005, 120).unwrap();
        let actual = stack.readback().unwrap()[1];
        assert!((actual.position_inverse_mass[2] - 1.5).abs() < 0.1);
        assert!(actual.linear_velocity[2].abs() < 0.2);
    }

    #[test]
    fn resident_primitive_topology_edits_keep_shapes_states_and_materials() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let box_shape = GpuRigidShape::Box {
            half_extents: [0.5; 3],
        };
        let capsule = GpuRigidShape::Capsule {
            radius: 0.5,
            half_length: 1.0,
        };
        let sphere = GpuRigidShape::Sphere { radius: 0.5 };
        let mut world = GpuRigidPrimitiveWorld::new_primitives(
            context.device(),
            context.queue(),
            &[body(0.0, 0.0), body(5.0, 0.0)],
            &[box_shape.clone(), capsule.clone()],
            config(),
        )
        .unwrap();
        let material = ColliderMaterial::new(0.2, 0.7);
        world.set_body_material(1, material).unwrap();
        let before = world.readback().unwrap();
        assert_eq!(
            world
                .append_primitive(body(5.6, 0.0), sphere.clone())
                .unwrap(),
            2
        );
        assert_same_states(&world.readback().unwrap()[..2], &before);
        assert_eq!(world.shape(2), Some(sphere.clone()));
        assert_eq!(world.body_material_override(1), Some(Some(material)));
        assert!(world.append_body(body(10.0, 0.0), 0.5).is_err());
        assert!(world.remove_body(0).is_err());
        let _pairs = world.step(0.01).unwrap();
        assert!(
            world
                .readback_contacts()
                .unwrap()
                .pairs
                .iter()
                .any(|(pair, contact)| { pair.a == 1 && pair.b == 2 && contact.is_contact() })
        );
        let (removed, removed_shape) = world.remove_primitive(0).unwrap();
        assert_eq!(removed_shape, box_shape);
        assert_eq!(
            removed.position_inverse_mass[0],
            before[0].position_inverse_mass[0]
        );
        assert_eq!(world.shape(0), Some(capsule));
        assert_eq!(world.shape(1), Some(sphere));
        assert_eq!(world.body_material_override(0), Some(Some(material)));
        assert!(world.remove_primitive(5).is_err());
        assert!(
            world
                .append_primitive(body(9.0, 0.0), GpuRigidShape::Sphere { radius: f32::NAN })
                .is_err()
        );
    }

    #[test]
    fn resident_primitive_topology_rebuilds_keep_contacts_on_available_backends() {
        let box_shape = GpuRigidShape::Box {
            half_extents: [0.5; 3],
        };
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            let mut fixed = body(0.0, 0.0);
            fixed.position_inverse_mass[3] = 0.0;
            fixed.inverse_inertia_sleep = [0.0; 4];
            let mut added = body(0.9, 0.0);
            added.position_inverse_mass[3] = 0.0;
            added.inverse_inertia_sleep = [0.0; 4];
            let mut world = GpuRigidPrimitiveWorld::new_primitives(
                context.device(),
                context.queue(),
                &[fixed],
                core::slice::from_ref(&box_shape),
                config(),
            )
            .unwrap();
            for shape in [
                GpuRigidShape::Capsule {
                    radius: 0.5,
                    half_length: 1.0,
                },
                GpuRigidShape::Cylinder {
                    radius: 0.5,
                    half_length: 1.0,
                },
            ] {
                assert_eq!(world.append_primitive(added, shape.clone()).unwrap(), 1);
                assert_eq!(world.step(0.01).unwrap(), 1);
                let contacts = world.readback_contacts().unwrap();
                assert!(contacts.pairs[0].1.is_contact(), "{backend:?}: {shape:?}");
                assert!(
                    contacts.pair_extra[0][0].is_contact(),
                    "{backend:?}: {shape:?}"
                );
                let (_, removed) = world.remove_primitive(1).unwrap();
                assert_eq!(removed, shape);
            }
        }
    }

    #[test]
    fn resident_cylinder_and_cone_contacts_cover_mixed_pairs_and_ground() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let cylinder = GpuRigidShape::Cylinder {
            radius: 1.0,
            half_length: 1.0,
        };
        let cone = GpuRigidShape::Cone {
            radius: 1.0,
            half_length: 1.0,
        };
        let sphere = GpuRigidShape::Sphere { radius: 0.5 };
        let cases = [
            ([cylinder.clone(), sphere.clone()], 1.4, 0.1),
            ([cylinder.clone(), cylinder.clone()], 1.8, 0.2),
            ([cylinder.clone(), cylinder.clone()], 2.0, 0.0),
            ([cone.clone(), sphere.clone()], 0.9, 0.142229),
            (
                [
                    GpuRigidShape::Box {
                        half_extents: [0.5; 3],
                    },
                    GpuRigidShape::Cylinder {
                        radius: 0.5,
                        half_length: 1.0,
                    },
                ],
                0.9,
                0.1,
            ),
        ];
        for (shapes, x, expected_depth) in cases {
            let mut first = body(0.0, 0.0);
            first.position_inverse_mass[3] = 0.0;
            first.inverse_inertia_sleep = [0.0; 4];
            let mut second = body(x, 0.0);
            second.position_inverse_mass[3] = 0.0;
            second.inverse_inertia_sleep = [0.0; 4];
            let mut world = GpuRigidPrimitiveWorld::new_primitives(
                context.device(),
                context.queue(),
                &[first, second],
                &shapes,
                config(),
            )
            .unwrap();
            let _pairs = world.step(0.01).unwrap();
            let contact = world.readback_contacts().unwrap().pairs[0].1;
            assert!(contact.is_contact(), "shapes: {shapes:?}");
            assert!(
                (contact.depth_hit[0] - expected_depth).abs() < 2e-2,
                "shapes: {shapes:?}, depth: {}",
                contact.depth_hit[0]
            );
            assert!(contact.normal[0] > 0.7, "shapes: {shapes:?}");
        }
        for shape in [cylinder.clone(), cone.clone()] {
            let mut analytic = body(0.0, 0.0);
            analytic.position_inverse_mass[3] = 0.0;
            analytic.inverse_inertia_sleep = [0.0; 4];
            analytic.orientation = [
                0.0,
                core::f32::consts::FRAC_1_SQRT_2,
                0.0,
                core::f32::consts::FRAC_1_SQRT_2,
            ];
            let mut sphere_state = body(1.4, 0.0);
            sphere_state.position_inverse_mass[3] = 0.0;
            sphere_state.inverse_inertia_sleep = [0.0; 4];
            let mut world = GpuRigidPrimitiveWorld::new_primitives(
                context.device(),
                context.queue(),
                &[analytic, sphere_state],
                &[shape.clone(), sphere.clone()],
                config(),
            )
            .unwrap();
            let _pairs = world.step(0.01).unwrap();
            let contact = world.readback_contacts().unwrap().pairs[0].1;
            assert!(contact.is_contact(), "shape: {shape:?}");
            assert!((contact.depth_hit[0] - 0.1).abs() < 1e-3);
            assert!(contact.normal[0] > 0.99);
        }
        let mut options = config();
        options.ground_half_extent = Some(10.0);
        for shape in [cylinder, cone] {
            let mut state = body(0.0, 0.0);
            state.position_inverse_mass = [0.0, 0.0, 0.8, 0.0];
            state.inverse_inertia_sleep = [0.0; 4];
            let mut world = GpuRigidPrimitiveWorld::new_primitives(
                context.device(),
                context.queue(),
                &[state],
                &[shape],
                options,
            )
            .unwrap();
            let _pairs = world.step(0.01).unwrap();
            let contact = world.readback_contacts().unwrap().ground[0];
            assert!(contact.is_contact());
            assert!((contact.depth_hit[0] - 0.2).abs() < 1e-4);
        }
    }

    #[test]
    fn resident_analytic_shapes_detect_mixed_overlap_and_separation() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let cylinder = GpuRigidShape::Cylinder {
            radius: 1.0,
            half_length: 1.0,
        };
        let cone = GpuRigidShape::Cone {
            radius: 1.0,
            half_length: 1.0,
        };
        let capsule = GpuRigidShape::Capsule {
            radius: 0.5,
            half_length: 1.0,
        };
        let box_shape = GpuRigidShape::Box {
            half_extents: [0.5; 3],
        };
        for shapes in [
            [cylinder.clone(), capsule.clone()],
            [cone.clone(), box_shape],
            [cone.clone(), capsule],
            [cone.clone(), cylinder],
            [cone.clone(), cone],
        ] {
            for x in [0.5, 5.0] {
                let mut first = body(0.0, 0.0);
                first.position_inverse_mass[3] = 0.0;
                first.inverse_inertia_sleep = [0.0; 4];
                let mut second = body(x, 0.0);
                second.position_inverse_mass[3] = 0.0;
                second.inverse_inertia_sleep = [0.0; 4];
                let mut world = GpuRigidPrimitiveWorld::new_primitives(
                    context.device(),
                    context.queue(),
                    &[first, second],
                    &shapes,
                    config(),
                )
                .unwrap();
                let _pairs = world.step(0.01).unwrap();
                let contact = world.readback_contacts().unwrap().pairs[0].1;
                assert_eq!(contact.is_contact(), x < 1.0, "shapes: {shapes:?}, x: {x}");
                if contact.is_contact() {
                    assert!(contact.depth_hit[0].is_finite());
                    assert!(contact.depth_hit[0] > 0.0);
                    assert!(
                        contact.normal[..3]
                            .iter()
                            .all(|component| component.is_finite())
                    );
                }
            }
        }
    }

    #[test]
    fn resident_convex_vertices_contact_sphere_ground_and_lbvh() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let hull = GpuRigidShape::Convex {
            vertices: [-1.0, 1.0]
                .into_iter()
                .flat_map(|x| {
                    [-1.0, 1.0]
                        .into_iter()
                        .flat_map(move |y| [-0.5, 0.5].into_iter().map(move |z| [x, y, z]))
                })
                .collect(),
        };
        let mut options = config();
        options.ground_half_extent = Some(10.0);
        let mut hull_state = body(0.0, 0.0);
        hull_state.position_inverse_mass = [0.0, 0.0, 0.4, 0.0];
        hull_state.inverse_inertia_sleep = [0.0; 4];
        let mut sphere_state = body(1.4, 0.0);
        sphere_state.position_inverse_mass = [1.4, 0.0, 0.4, 0.0];
        sphere_state.inverse_inertia_sleep = [0.0; 4];
        let mut world = GpuRigidPrimitiveWorld::new_primitives(
            context.device(),
            context.queue(),
            &[hull_state, sphere_state],
            &[hull.clone(), GpuRigidShape::Sphere { radius: 0.5 }],
            options,
        )
        .unwrap();
        assert_eq!(world.shape(0), Some(hull.clone()));
        let _pairs = world.step(0.01).unwrap();
        let contacts = world.readback_contacts().unwrap();
        assert!(contacts.pairs[0].1.is_contact());
        assert!(
            (contacts.pairs[0].1.depth_hit[0] - 0.1).abs() < 1e-3,
            "contact: {:?}",
            contacts.pairs[0].1
        );
        assert!(contacts.pairs[0].1.normal[0] > 0.99);
        assert!(contacts.ground[0].is_contact());
        assert!((contacts.ground[0].depth_hit[0] - 0.1).abs() < 1e-3);

        let mut states = (0..17)
            .map(|index| body(index as f32 * 10.0, 0.0))
            .collect::<Vec<_>>();
        states[0] = hull_state;
        states[1] = sphere_state;
        let mut shapes = vec![GpuRigidShape::Sphere { radius: 0.5 }; 17];
        shapes[0] = hull.clone();
        let mut large = GpuRigidPrimitiveWorld::new_primitives(
            context.device(),
            context.queue(),
            &states,
            &shapes,
            config(),
        )
        .unwrap();
        assert!(!large.uses_exhaustive_pairs());
        let _pairs = large.step(0.01).unwrap();
        assert!(
            large
                .readback_contacts()
                .unwrap()
                .pairs
                .iter()
                .any(|(pair, contact)| { pair.a == 0 && pair.b == 1 && contact.is_contact() })
        );
        let removed = large.remove_primitive(0).unwrap();
        assert_eq!(removed.1, hull);

        let elongated = GpuRigidShape::Convex {
            vertices: [-2.0, 2.0]
                .into_iter()
                .flat_map(|x| {
                    [-0.25, 0.25]
                        .into_iter()
                        .flat_map(move |y| [-0.25, 0.25].into_iter().map(move |z| [x, y, z]))
                })
                .collect(),
        };
        let mut rotated = hull_state;
        rotated.orientation = [
            0.0,
            0.0,
            core::f32::consts::FRAC_1_SQRT_2,
            core::f32::consts::FRAC_1_SQRT_2,
        ];
        rotated.position_inverse_mass[2] = 2.0;
        let mut neighbor = sphere_state;
        neighbor.position_inverse_mass = [0.0, 2.4, 2.0, 0.0];
        let mut rotated_world = GpuRigidPrimitiveWorld::new_primitives(
            context.device(),
            context.queue(),
            &[rotated, neighbor],
            &[elongated, GpuRigidShape::Sphere { radius: 0.5 }],
            config(),
        )
        .unwrap();
        let _pairs = rotated_world.step(0.01).unwrap();
        let contact = rotated_world.readback_contacts().unwrap().pairs[0].1;
        assert!(contact.is_contact());
        assert!((contact.depth_hit[0] - 0.1).abs() < 1e-3, "{contact:?}");
        assert!(contact.normal[1] > 0.99);
    }

    #[test]
    fn resident_tetrahedron_face_normal_controls_sphere_penetration() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let hull = GpuRigidShape::Convex {
            vertices: vec![
                [2.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [2.0, 2.0, 0.0],
                [2.0, 0.0, 2.0],
            ],
        };
        let mut fixed = body(0.0, 0.0);
        fixed.position_inverse_mass = [0.0, 0.0, 0.0, 0.0];
        fixed.inverse_inertia_sleep = [0.0; 4];
        for (z, expected) in [(0.35, true), (0.61, false)] {
            let mut sphere = body(3.0, 0.75);
            sphere.position_inverse_mass = [3.0, 0.75, z, 0.0];
            sphere.inverse_inertia_sleep = [0.0; 4];
            let mut world = GpuRigidPrimitiveWorld::new_primitives(
                context.device(),
                context.queue(),
                &[fixed, sphere],
                &[hull.clone(), GpuRigidShape::Sphere { radius: 0.2 }],
                config(),
            )
            .unwrap();
            let _pairs = world.step(0.01).unwrap();
            let contact = world.readback_contacts().unwrap().pairs[0].1;
            assert_eq!(contact.is_contact(), expected, "{contact:?}");
            if expected {
                let depth = 0.2 - 0.1 / 3.0_f32.sqrt();
                assert!((contact.depth_hit[0] - depth).abs() < 1e-3, "{contact:?}");
                for component in &contact.normal[..3] {
                    assert!((*component - 3.0_f32.recip().sqrt()).abs() < 1e-3);
                }
            }
        }
        let mut sphere = body(3.0, 0.75);
        sphere.position_inverse_mass = [3.0, 0.75, 0.35, 0.0];
        sphere.inverse_inertia_sleep = [0.0; 4];
        let mut reversed = GpuRigidPrimitiveWorld::new_primitives(
            context.device(),
            context.queue(),
            &[sphere, fixed],
            &[GpuRigidShape::Sphere { radius: 0.2 }, hull],
            config(),
        )
        .unwrap();
        let _pairs = reversed.step(0.01).unwrap();
        let contact = reversed.readback_contacts().unwrap().pairs[0].1;
        assert!(contact.is_contact());
        assert!((contact.depth_hit[0] - (0.2 - 0.1 / 3.0_f32.sqrt())).abs() < 1e-3);
        assert!(contact.normal[..3].iter().all(|value| *value < -0.57));
    }

    #[test]
    fn resident_convex_edge_axes_match_rotated_box_sat() {
        use nalgebra::{Isometry3, Point3, Translation3, UnitQuaternion, Vector3};

        use crate::articulated_world::box_box_contact;
        use crate::convex::convex_face_contact;

        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let half_a = Vector3::new(0.4, 0.2, 0.3);
        let half_b = Vector3::new(0.3, 0.4, 0.2);
        let vertices = |half: Vector3<f64>| {
            [-1.0, 1.0]
                .into_iter()
                .flat_map(|x| {
                    [-1.0, 1.0].into_iter().flat_map(move |y| {
                        [-1.0, 1.0]
                            .into_iter()
                            .map(move |z| [half.x * x, half.y * y, half.z * z])
                    })
                })
                .collect::<Vec<_>>()
        };
        let local_a = vertices(half_a);
        let local_b = vertices(half_b);
        let axes = [Vector3::x(), Vector3::y(), Vector3::z()];
        let mut selected = None;
        for index in 0..500 {
            let phase = index as f64 * 0.37;
            let pose_a = Isometry3::from_parts(
                Translation3::identity(),
                UnitQuaternion::from_euler_angles(phase * 0.13, phase * 0.07, phase * 0.21),
            );
            let pose_b = Isometry3::from_parts(
                Translation3::new(
                    phase.sin() * 0.8,
                    (phase * 1.7).cos() * 0.6,
                    (phase * 0.9).sin() * 0.55,
                ),
                UnitQuaternion::from_euler_angles(phase * 0.31, phase * 0.19, phase * 0.11),
            );
            let Some((normal, _, depth)) = box_box_contact(pose_a, half_a, pose_b, half_b) else {
                continue;
            };
            if depth < 1e-4 {
                continue;
            }
            let world_a = local_a
                .iter()
                .map(|v| {
                    pose_a
                        .transform_point(&Point3::new(v[0], v[1], v[2]))
                        .coords
                })
                .collect::<Vec<_>>();
            let world_b = local_b
                .iter()
                .map(|v| {
                    pose_b
                        .transform_point(&Point3::new(v[0], v[1], v[2]))
                        .coords
                })
                .collect::<Vec<_>>();
            let normals_a = axes.map(|axis| pose_a.rotation * axis);
            let normals_b = axes.map(|axis| pose_b.rotation * axis);
            if convex_face_contact(&world_a, &normals_a, &world_b, &normals_b)
                .is_some_and(|face| face.penetration - depth > 1e-4)
            {
                selected = Some((pose_a, pose_b, normal, depth));
                break;
            }
        }
        let (pose_a, pose_b, expected_normal, expected_depth) =
            selected.expect("an edge-axis contact case");
        let to_state = |pose: Isometry3<f64>| {
            let mut state = body(0.0, 0.0);
            state.position_inverse_mass = [
                pose.translation.vector.x as f32,
                pose.translation.vector.y as f32,
                pose.translation.vector.z as f32,
                0.0,
            ];
            let q = pose.rotation.quaternion();
            state.orientation = [q.i as f32, q.j as f32, q.k as f32, q.w as f32];
            state.inverse_inertia_sleep = [0.0; 4];
            state
        };
        let hull_a = GpuRigidShape::Convex {
            vertices: local_a
                .iter()
                .map(|v| v.map(|value| value as f32))
                .collect(),
        };
        for shape_b in [
            GpuRigidShape::Box {
                half_extents: [0.3, 0.4, 0.2],
            },
            GpuRigidShape::Convex {
                vertices: local_b
                    .iter()
                    .map(|v| v.map(|value| value as f32))
                    .collect(),
            },
        ] {
            let mut world = GpuRigidPrimitiveWorld::new_primitives(
                context.device(),
                context.queue(),
                &[to_state(pose_a), to_state(pose_b)],
                &[hull_a.clone(), shape_b],
                config(),
            )
            .unwrap();
            let _pairs = world.step(0.01).unwrap();
            let contact = world.readback_contacts().unwrap().pairs[0].1;
            assert!(contact.is_contact(), "{contact:?}");
            assert!(
                (f64::from(contact.depth_hit[0]) - expected_depth).abs() < 2e-3,
                "{contact:?}, expected {expected_depth}"
            );
            let alignment = contact.normal[..3]
                .iter()
                .zip(expected_normal.iter())
                .map(|(a, b)| f64::from(*a) * b)
                .sum::<f64>();
            assert!(
                alignment > 0.99,
                "{contact:?}, expected {expected_normal:?}"
            );
        }
    }

    #[tokio::test]
    async fn vulkan_resident_cylinder_and_convex_sphere_contacts() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
        else {
            eprintln!("Vulkan adapter unavailable; skipping resident primitive test");
            return;
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let mut world = GpuRigidPrimitiveWorld::new_primitives(
            &device,
            &queue,
            &[body(0.0, 0.0), body(1.4, 0.0)],
            &[
                GpuRigidShape::Cylinder {
                    radius: 1.0,
                    half_length: 1.0,
                },
                GpuRigidShape::Sphere { radius: 0.5 },
            ],
            config(),
        )
        .unwrap();
        let _pairs = world.step(0.01).unwrap();
        let contact = world.readback_contacts().unwrap().pairs[0].1;
        assert!(contact.is_contact());
        assert!((contact.depth_hit[0] - 0.1).abs() < 1e-4);
        let hull = GpuRigidShape::Convex {
            vertices: [-1.0, 1.0]
                .into_iter()
                .flat_map(|x| {
                    [-1.0, 1.0]
                        .into_iter()
                        .flat_map(move |y| [-0.5, 0.5].into_iter().map(move |z| [x, y, z]))
                })
                .collect(),
        };
        let mut convex = GpuRigidPrimitiveWorld::new_primitives(
            &device,
            &queue,
            &[body(0.0, 0.0), body(1.4, 0.0)],
            &[hull, GpuRigidShape::Sphere { radius: 0.5 }],
            config(),
        )
        .unwrap();
        let _pairs = convex.step(0.01).unwrap();
        let contact = convex.readback_contacts().unwrap().pairs[0].1;
        assert!(contact.is_contact());
        assert!((contact.depth_hit[0] - 0.1).abs() < 1e-4);
    }
}
