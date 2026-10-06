//! GPU-resident point-to-point constraints between rigid bodies.

use core::mem::size_of;
use core::time::Duration;
use std::collections::BTreeMap;
use std::sync::mpsc;

use bytemuck::Zeroable;
use wgpu::util::DeviceExt;

/// Local anchor points constrained to coincide in world space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuRigidBallJoint {
    /// First body in the resident world's dense index order.
    pub body_a: u32,
    /// Second body in the resident world's dense index order.
    pub body_b: u32,
    /// First anchor in body A's local frame.
    pub local_anchor_a: [f32; 3],
    /// Second anchor in body B's local frame.
    pub local_anchor_b: [f32; 3],
}

impl GpuRigidBallJoint {
    fn is_valid(self, body_count: usize, environment_ids: &[u32]) -> bool {
        let a = self.body_a as usize;
        let b = self.body_b as usize;
        a < body_count
            && b < body_count
            && a != b
            && environment_ids[a] == environment_ids[b]
            && [self.local_anchor_a, self.local_anchor_b]
                .into_iter()
                .flatten()
                .all(f32::is_finite)
            && [self.local_anchor_a, self.local_anchor_b]
                .into_iter()
                .map(|anchor| anchor.into_iter().map(|value| value * value).sum::<f32>())
                .all(f32::is_finite)
    }
}

/// Local frames constrained to share both position and orientation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuRigidFixedJoint {
    /// First body in the resident world's dense index order.
    pub body_a: u32,
    /// Second body in the resident world's dense index order.
    pub body_b: u32,
    /// First frame origin in body A's local coordinates.
    pub local_anchor_a: [f32; 3],
    /// Second frame origin in body B's local coordinates.
    pub local_anchor_b: [f32; 3],
    /// First local frame orientation as a unit XYZW quaternion.
    pub local_rotation_a: [f32; 4],
    /// Second local frame orientation as a unit XYZW quaternion.
    pub local_rotation_b: [f32; 4],
}

impl GpuRigidFixedJoint {
    fn is_valid(self, body_count: usize, environment_ids: &[u32]) -> bool {
        GpuRigidBallJoint {
            body_a: self.body_a,
            body_b: self.body_b,
            local_anchor_a: self.local_anchor_a,
            local_anchor_b: self.local_anchor_b,
        }
        .is_valid(body_count, environment_ids)
            && [self.local_rotation_a, self.local_rotation_b]
                .into_iter()
                .all(|rotation| {
                    let norm_sq = rotation.into_iter().map(|v| v * v).sum::<f32>();
                    norm_sq.is_finite() && (norm_sq - 1.0).abs() <= 1e-3
                })
    }
}

/// Hinge with coincident anchors and one free relative rotation axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuRigidRevoluteJoint {
    /// First body in the resident world's dense index order.
    pub body_a: u32,
    /// Second body in the resident world's dense index order.
    pub body_b: u32,
    /// First hinge origin in body A's local coordinates.
    pub local_anchor_a: [f32; 3],
    /// Second hinge origin in body B's local coordinates.
    pub local_anchor_b: [f32; 3],
    /// Unit hinge axis in body A's local coordinates.
    pub local_axis_a: [f32; 3],
    /// Unit hinge axis in body B's local coordinates.
    pub local_axis_b: [f32; 3],
}

impl GpuRigidRevoluteJoint {
    fn is_valid(self, body_count: usize, environment_ids: &[u32]) -> bool {
        GpuRigidBallJoint {
            body_a: self.body_a,
            body_b: self.body_b,
            local_anchor_a: self.local_anchor_a,
            local_anchor_b: self.local_anchor_b,
        }
        .is_valid(body_count, environment_ids)
            && [self.local_axis_a, self.local_axis_b]
                .into_iter()
                .all(|axis| {
                    let norm_sq = axis.into_iter().map(|v| v * v).sum::<f32>();
                    norm_sq.is_finite() && (norm_sq - 1.0).abs() <= 1e-3
                })
    }
}

/// Slider with one free translation along the local frame's Z axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuRigidPrismaticJoint {
    /// First body in the resident world's dense index order.
    pub body_a: u32,
    /// Second body in the resident world's dense index order.
    pub body_b: u32,
    /// First frame origin in body A's local coordinates.
    pub local_anchor_a: [f32; 3],
    /// Second frame origin in body B's local coordinates.
    pub local_anchor_b: [f32; 3],
    /// First local frame orientation as a unit XYZW quaternion.
    pub local_rotation_a: [f32; 4],
    /// Second local frame orientation as a unit XYZW quaternion.
    pub local_rotation_b: [f32; 4],
}

impl GpuRigidPrismaticJoint {
    fn is_valid(self, body_count: usize, environment_ids: &[u32]) -> bool {
        GpuRigidFixedJoint {
            body_a: self.body_a,
            body_b: self.body_b,
            local_anchor_a: self.local_anchor_a,
            local_anchor_b: self.local_anchor_b,
            local_rotation_a: self.local_rotation_a,
            local_rotation_b: self.local_rotation_b,
        }
        .is_valid(body_count, environment_ids)
    }
}

/// Velocity drive on a revolute or prismatic joint's free axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuRigidAxisMotor {
    /// Relative angular or linear velocity requested along the joint axis.
    pub target_velocity: f32,
    /// Maximum torque or force magnitude.
    pub max_force: f32,
}

impl GpuRigidAxisMotor {
    fn is_valid(self) -> bool {
        self.target_velocity.is_finite()
            && self.max_force.is_finite()
            && self.max_force >= 0.0
            && self.max_force <= f32::MAX.sqrt()
    }
}

/// Allowed displacement between prismatic anchor frames along the slide axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuRigidPrismaticLimit {
    /// Lower displacement bound in meters.
    pub min: f32,
    /// Upper displacement bound in meters.
    pub max: f32,
}

impl GpuRigidPrismaticLimit {
    fn is_valid(self) -> bool {
        self.min.is_finite() && self.max.is_finite() && self.min <= self.max
    }
}

/// Allowed continuous revolute angle in radians relative to local tangents.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuRigidRevoluteLimit {
    /// Lower relative angle bound in radians.
    pub min: f32,
    /// Upper relative angle bound in radians.
    pub max: f32,
}

impl GpuRigidRevoluteLimit {
    fn is_valid(self) -> bool {
        self.min.is_finite() && self.max.is_finite() && self.min <= self.max
    }
}

/// Force-limited position and velocity target for a joint's free axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuRigidAxisServo {
    /// Target relative angle in radians or displacement in meters.
    pub position_target: f32,
    /// Target relative angular or linear velocity.
    pub velocity_target: f32,
    /// Force or torque per unit position error.
    pub stiffness: f32,
    /// Force or torque per unit velocity error.
    pub damping: f32,
    /// Maximum force or torque magnitude.
    pub max_force: f32,
}

impl GpuRigidAxisServo {
    fn is_valid(self) -> bool {
        self.position_target.is_finite()
            && self.velocity_target.is_finite()
            && self.stiffness.is_finite()
            && self.stiffness >= 0.0
            && self.damping.is_finite()
            && self.damping >= 0.0
            && self.max_force.is_finite()
            && self.max_force >= 0.0
            && self.max_force <= f32::MAX.sqrt()
    }
}

/// Invalid topology, device capacity, or solve parameters.
#[derive(Debug, thiserror::Error)]
pub enum GpuRigidBallJointError {
    /// A body, environment, anchor, or timestep is invalid.
    #[error("invalid GPU resident joint input")]
    InvalidInput,
    /// Joint data or serial work exceeds device limits.
    #[error("GPU resident joint solve exceeds capacity")]
    Capacity,
    /// The requested GPU diagnostic transfer failed.
    #[error("GPU resident joint readback failed: {0}")]
    Readback(String),
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct PackedJoint {
    bodies: [u32; 4],
    anchor_a: [f32; 4],
    anchor_b: [f32; 4],
    frame_a: [f32; 4],
    frame_b: [f32; 4],
    drive: [f32; 4],
    servo: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct IslandRange {
    offset: u32,
    count: u32,
    padding: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SolveParams {
    dt: f32,
    bias: f32,
    iterations: u32,
    islands: u32,
    bodies: u32,
    padding: [u32; 3],
    temporal: [f32; 4],
    correction_limits: [f32; 4],
}

/// Soft joint coefficients for a temporal substep.
#[derive(Clone, Copy, Debug)]
pub struct GpuRigidTemporalJointParams {
    /// Spring frequency in hertz, shared by linear and angular rows.
    pub frequency: f32,
    /// Nonnegative spring damping ratio.
    pub damping_ratio: f32,
    /// Maximum linear position correction speed in metres per second.
    pub max_linear_correction_speed: f32,
    /// Maximum angular position correction speed in radians per second.
    pub max_angular_correction_speed: f32,
    /// Number of sweeps in each bias or relaxation pass.
    pub iterations: u32,
}

impl Default for GpuRigidTemporalJointParams {
    fn default() -> Self {
        Self {
            frequency: 30.0,
            damping_ratio: 1.0,
            max_linear_correction_speed: 3.0,
            max_angular_correction_speed: 3.0,
            iterations: 1,
        }
    }
}

/// Immutable bias and relaxation bindings for temporal joint constraints.
#[derive(Debug)]
pub struct GpuRigidTemporalJointStep {
    bias: wgpu::BindGroup,
    relax: wgpu::BindGroup,
    warm_pipeline: wgpu::ComputePipeline,
    iteration_pipeline: wgpu::ComputePipeline,
    angle_pipeline: wgpu::ComputePipeline,
    sample_angle_pipeline: wgpu::ComputePipeline,
    drive_capture_pipeline: wgpu::ComputePipeline,
    joint_count: u32,
    islands: u32,
    iterations: u32,
}

impl GpuRigidTemporalJointStep {
    /// Capture each servo's spring target once before warm starting a substep.
    pub fn encode_capture_drives(&self, encoder: &mut wgpu::CommandEncoder) {
        self.encode_angles(encoder, &self.drive_capture_pipeline);
    }
    /// Sample the initial wrapped angle before integrating any substep.
    pub fn encode_capture_angles(&self, encoder: &mut wgpu::CommandEncoder) {
        self.encode_angles(encoder, &self.sample_angle_pipeline);
    }
    /// Track integrated revolute turns using the velocity that advanced the pose.
    pub fn encode_integrated_angles(&self, encoder: &mut wgpu::CommandEncoder) {
        self.encode_angles(encoder, &self.angle_pipeline);
    }
    fn encode_angles(&self, encoder: &mut wgpu::CommandEncoder, pipeline: &wgpu::ComputePipeline) {
        if self.joint_count == 0 {
            return;
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Tessera temporal joint angle tracking"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.bias, &[]);
        pass.dispatch_workgroups(self.joint_count.div_ceil(64), 1, 1);
    }
    /// Warm start once at the beginning of a temporal substep.
    pub fn encode_warm(&self, encoder: &mut wgpu::CommandEncoder) {
        self.encode(encoder, &self.warm_pipeline, &self.bias);
    }
    /// Solve soft joint errors before position integration.
    pub fn encode_bias(&self, encoder: &mut wgpu::CommandEncoder) {
        for _ in 0..self.iterations {
            self.encode_bias_iteration(encoder);
        }
    }
    /// Solve one sweep for alternating contact and joint constraints.
    pub fn encode_bias_iteration(&self, encoder: &mut wgpu::CommandEncoder) {
        self.encode(encoder, &self.iteration_pipeline, &self.bias);
    }
    /// Relax velocity after position integration, without another warm start.
    pub fn encode_relax(&self, encoder: &mut wgpu::CommandEncoder) {
        for _ in 0..self.iterations {
            self.encode_relax_iteration(encoder);
        }
    }
    /// Solve one relaxation sweep for alternating constraints.
    pub fn encode_relax_iteration(&self, encoder: &mut wgpu::CommandEncoder) {
        self.encode(encoder, &self.iteration_pipeline, &self.relax);
    }
    fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        group: &wgpu::BindGroup,
    ) {
        if self.islands == 0 {
            return;
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Tessera temporal joint pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(self.islands.div_ceil(64), 1, 1);
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AngleState {
    angle: f32,
    wrapped: f32,
    initialized: f32,
    padding: f32,
}

/// Bilateral joint impulses on the same GPU state used by rigid contacts.
///
/// Independent joint islands run in parallel. Each island solves its joints
/// sequentially, so bodies shared by a chain cannot race across invocations.
#[derive(Debug)]
pub struct GpuRigidBallJointSolver {
    joints: Vec<GpuRigidBallJoint>,
    fixed_joints: Vec<GpuRigidFixedJoint>,
    revolute_joints: Vec<GpuRigidRevoluteJoint>,
    prismatic_joints: Vec<GpuRigidPrismaticJoint>,
    revolute_motors: Vec<Option<GpuRigidAxisMotor>>,
    revolute_servos: Vec<Option<GpuRigidAxisServo>>,
    revolute_limits: Vec<Option<GpuRigidRevoluteLimit>>,
    prismatic_motors: Vec<Option<GpuRigidAxisMotor>>,
    prismatic_servos: Vec<Option<GpuRigidAxisServo>>,
    prismatic_limits: Vec<Option<GpuRigidPrismaticLimit>>,
    packed_joints: Vec<PackedJoint>,
    joint_buffer: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    coupled_warm_pipeline: wgpu::ComputePipeline,
    coupled_iteration_pipeline: wgpu::ComputePipeline,
    capture_pipeline: wgpu::ComputePipeline,
    activity_pipeline: wgpu::ComputePipeline,
    correction_pipeline: wgpu::ComputePipeline,
    angle_pipeline: wgpu::ComputePipeline,
    sample_angle_pipeline: wgpu::ComputePipeline,
    drive_capture_pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    impulses: wgpu::Buffer,
    angular_impulses: wgpu::Buffer,
    angles: wgpu::Buffer,
    params: wgpu::Buffer,
    island_count: u32,
    body_count: u32,
    max_island_joints: u32,
    last_dt_bits: Option<u32>,
    temporal_resources: [wgpu::Buffer; 4],
    temporal_active: bool,
}

impl GpuRigidBallJointSolver {
    /// Build joint islands for a resident rigid state buffer.
    pub fn new(
        device: &wgpu::Device,
        state_buffer: &wgpu::Buffer,
        body_count: usize,
        environment_ids: &[u32],
        joints: &[GpuRigidBallJoint],
    ) -> Result<Self, GpuRigidBallJointError> {
        Self::new_mixed(
            device,
            state_buffer,
            body_count,
            environment_ids,
            joints,
            &[],
        )
    }

    /// Build connected islands containing ball and fixed joints together.
    pub fn new_mixed(
        device: &wgpu::Device,
        state_buffer: &wgpu::Buffer,
        body_count: usize,
        environment_ids: &[u32],
        joints: &[GpuRigidBallJoint],
        fixed_joints: &[GpuRigidFixedJoint],
    ) -> Result<Self, GpuRigidBallJointError> {
        Self::new_with_revolute(
            device,
            state_buffer,
            body_count,
            environment_ids,
            joints,
            fixed_joints,
            &[],
        )
    }

    /// Build connected islands containing all supported resident joints.
    pub fn new_with_revolute(
        device: &wgpu::Device,
        state_buffer: &wgpu::Buffer,
        body_count: usize,
        environment_ids: &[u32],
        joints: &[GpuRigidBallJoint],
        fixed_joints: &[GpuRigidFixedJoint],
        revolute_joints: &[GpuRigidRevoluteJoint],
    ) -> Result<Self, GpuRigidBallJointError> {
        Self::new_with_prismatic(
            device,
            state_buffer,
            body_count,
            environment_ids,
            joints,
            fixed_joints,
            revolute_joints,
            &[],
        )
    }

    /// Build connected islands containing ball, fixed, revolute, and prismatic joints.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_prismatic(
        device: &wgpu::Device,
        state_buffer: &wgpu::Buffer,
        body_count: usize,
        environment_ids: &[u32],
        joints: &[GpuRigidBallJoint],
        fixed_joints: &[GpuRigidFixedJoint],
        revolute_joints: &[GpuRigidRevoluteJoint],
        prismatic_joints: &[GpuRigidPrismaticJoint],
    ) -> Result<Self, GpuRigidBallJointError> {
        if joints.is_empty()
            && fixed_joints.is_empty()
            && revolute_joints.is_empty()
            && prismatic_joints.is_empty()
            || joints
                .len()
                .checked_add(fixed_joints.len())
                .and_then(|count| count.checked_add(revolute_joints.len()))
                .and_then(|count| count.checked_add(prismatic_joints.len()))
                .is_none()
            || environment_ids.len() != body_count
            || u32::try_from(body_count).is_err()
            || joints
                .iter()
                .any(|joint| !joint.is_valid(body_count, environment_ids))
            || fixed_joints
                .iter()
                .any(|joint| !joint.is_valid(body_count, environment_ids))
            || revolute_joints
                .iter()
                .any(|joint| !joint.is_valid(body_count, environment_ids))
            || prismatic_joints
                .iter()
                .any(|joint| !joint.is_valid(body_count, environment_ids))
        {
            return Err(GpuRigidBallJointError::InvalidInput);
        }
        let edges = joints
            .iter()
            .map(|joint| (joint.body_a, joint.body_b))
            .chain(
                fixed_joints
                    .iter()
                    .map(|joint| (joint.body_a, joint.body_b)),
            )
            .chain(
                revolute_joints
                    .iter()
                    .map(|joint| (joint.body_a, joint.body_b)),
            )
            .chain(
                prismatic_joints
                    .iter()
                    .map(|joint| (joint.body_a, joint.body_b)),
            )
            .collect::<Vec<_>>();
        let mut parents = (0..body_count).collect::<Vec<_>>();
        for &(body_a, body_b) in &edges {
            let a = root(&mut parents, body_a as usize);
            let b = root(&mut parents, body_b as usize);
            parents[b] = a;
        }
        let mut groups = BTreeMap::<usize, Vec<u32>>::new();
        for (index, &(body_a, _)) in edges.iter().enumerate() {
            let representative = root(&mut parents, body_a as usize);
            let index = u32::try_from(index).map_err(|_| GpuRigidBallJointError::Capacity)?;
            groups.entry(representative).or_default().push(index);
        }
        let island_count =
            u32::try_from(groups.len()).map_err(|_| GpuRigidBallJointError::Capacity)?;
        let max_island_joints = groups.values().map(Vec::len).max().unwrap_or(0);
        let max_island_joints =
            u32::try_from(max_island_joints).map_err(|_| GpuRigidBallJointError::Capacity)?;
        let mut ranges = Vec::with_capacity(groups.len());
        let mut indices = Vec::with_capacity(edges.len());
        for group in groups.into_values() {
            ranges.push(IslandRange {
                offset: u32::try_from(indices.len())
                    .map_err(|_| GpuRigidBallJointError::Capacity)?,
                count: u32::try_from(group.len()).map_err(|_| GpuRigidBallJointError::Capacity)?,
                padding: [0; 2],
            });
            indices.extend(group);
        }
        let packed = joints
            .iter()
            .map(|joint| PackedJoint {
                bodies: [joint.body_a, joint.body_b, 0, 0],
                anchor_a: [
                    joint.local_anchor_a[0],
                    joint.local_anchor_a[1],
                    joint.local_anchor_a[2],
                    0.0,
                ],
                anchor_b: [
                    joint.local_anchor_b[0],
                    joint.local_anchor_b[1],
                    joint.local_anchor_b[2],
                    0.0,
                ],
                frame_a: [0.0, 0.0, 0.0, 1.0],
                frame_b: [0.0, 0.0, 0.0, 1.0],
                drive: [0.0; 4],
                servo: [0.0; 4],
            })
            .chain(fixed_joints.iter().map(|joint| PackedJoint {
                bodies: [joint.body_a, joint.body_b, 1, 0],
                anchor_a: [
                    joint.local_anchor_a[0],
                    joint.local_anchor_a[1],
                    joint.local_anchor_a[2],
                    0.0,
                ],
                anchor_b: [
                    joint.local_anchor_b[0],
                    joint.local_anchor_b[1],
                    joint.local_anchor_b[2],
                    0.0,
                ],
                frame_a: joint.local_rotation_a,
                frame_b: joint.local_rotation_b,
                drive: [0.0; 4],
                servo: [0.0; 4],
            }))
            .chain(revolute_joints.iter().map(|joint| PackedJoint {
                bodies: [joint.body_a, joint.body_b, 2, 0],
                anchor_a: [
                    joint.local_anchor_a[0],
                    joint.local_anchor_a[1],
                    joint.local_anchor_a[2],
                    0.0,
                ],
                anchor_b: [
                    joint.local_anchor_b[0],
                    joint.local_anchor_b[1],
                    joint.local_anchor_b[2],
                    0.0,
                ],
                frame_a: [
                    joint.local_axis_a[0],
                    joint.local_axis_a[1],
                    joint.local_axis_a[2],
                    0.0,
                ],
                frame_b: [
                    joint.local_axis_b[0],
                    joint.local_axis_b[1],
                    joint.local_axis_b[2],
                    0.0,
                ],
                drive: [0.0; 4],
                servo: [0.0; 4],
            }))
            .chain(prismatic_joints.iter().map(|joint| PackedJoint {
                bodies: [joint.body_a, joint.body_b, 3, 0],
                anchor_a: [
                    joint.local_anchor_a[0],
                    joint.local_anchor_a[1],
                    joint.local_anchor_a[2],
                    0.0,
                ],
                anchor_b: [
                    joint.local_anchor_b[0],
                    joint.local_anchor_b[1],
                    joint.local_anchor_b[2],
                    0.0,
                ],
                frame_a: joint.local_rotation_a,
                frame_b: joint.local_rotation_b,
                drive: [0.0; 4],
                servo: [0.0; 4],
            }))
            .collect::<Vec<_>>();
        let limits = device.limits();
        let max_storage = u64::from(limits.max_storage_buffer_binding_size);
        let joint_bytes = packed.len() as u64 * size_of::<PackedJoint>() as u64;
        let range_bytes = ranges.len() as u64 * size_of::<IslandRange>() as u64;
        let index_bytes = indices.len() as u64 * size_of::<u32>() as u64;
        let impulse_bytes = packed.len() as u64 * size_of::<[f32; 4]>() as u64;
        let velocity_bytes = body_count as u64 * size_of::<[[f32; 4]; 2]>() as u64;
        let body_count = u32::try_from(body_count).map_err(|_| GpuRigidBallJointError::Capacity)?;
        if [
            joint_bytes,
            range_bytes,
            index_bytes,
            impulse_bytes,
            velocity_bytes,
            impulse_bytes,
        ]
        .into_iter()
        .any(|bytes| bytes > max_storage || bytes > limits.max_buffer_size)
            || limits.max_storage_buffers_per_shader_stage < 8
            || island_count.div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || body_count.div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || packed.len().div_ceil(64) as u64
                > u64::from(limits.max_compute_workgroups_per_dimension)
        {
            return Err(GpuRigidBallJointError::Capacity);
        }
        let joint_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera resident ball joints"),
            contents: bytemuck::cast_slice(&packed),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let range_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera resident ball joint islands"),
            contents: bytemuck::cast_slice(&ranges),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera resident ball joint island indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let impulses = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera resident ball joint warm impulses"),
            contents: bytemuck::cast_slice(&vec![[0.0f32; 4]; packed.len()]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let angular_impulses = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera resident fixed joint angular impulses"),
            contents: bytemuck::cast_slice(&vec![[0.0f32; 4]; packed.len()]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let angles = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera resident revolute continuous angles"),
            contents: bytemuck::cast_slice(&vec![AngleState::zeroed(); packed.len()]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let velocities_before_solve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera resident joint pre-solve velocities"),
            size: velocity_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera resident ball joint solve parameters"),
            size: size_of::<SolveParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera resident ball joint shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_rigid_ball_joint.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Tessera resident joint bindings"),
            entries: &(0..9)
                .map(|binding| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: if binding == 5 {
                            wgpu::BufferBindingType::Uniform
                        } else {
                            wgpu::BufferBindingType::Storage {
                                read_only: matches!(binding, 1..=3),
                            }
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect::<Vec<_>>(),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Tessera resident joint pipeline layout"),
            bind_group_layouts: &[&layout],
            immediate_size: 0,
        });
        let activity_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera prescribed joint island activity"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_rigid_joint_activity.wgsl").into()),
        });
        let activity_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tessera prescribed joint island activity"),
            layout: None,
            module: &activity_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tessera resident ball joint solver"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let coupled_warm_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Tessera resident coupled joint warm"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("coupled_warm"),
                compilation_options: Default::default(),
                cache: None,
            });
        let coupled_iteration_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Tessera resident coupled joint iteration"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("coupled_iteration"),
                compilation_options: Default::default(),
                cache: None,
            });
        let capture_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tessera resident joint velocity capture"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("capture_velocity"),
            compilation_options: Default::default(),
            cache: None,
        });
        let correction_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Tessera resident joint pose correction"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("correct_pose"),
                compilation_options: Default::default(),
                cache: None,
            });
        let angle_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tessera resident revolute angle tracking"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("update_angles"),
            compilation_options: Default::default(),
            cache: None,
        });
        let sample_angle_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Tessera resident revolute angle sampling"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("sample_angles"),
                compilation_options: Default::default(),
                cache: None,
            });
        let drive_capture_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Tessera temporal servo target capture"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("capture_temporal_drives"),
                compilation_options: Default::default(),
                cache: None,
            });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera resident ball joint bind group"),
            layout: &layout,
            entries: &[
                binding(0, state_buffer),
                binding(1, &joint_buffer),
                binding(2, &range_buffer),
                binding(3, &index_buffer),
                binding(4, &impulses),
                binding(5, &params),
                binding(6, &angular_impulses),
                binding(7, &velocities_before_solve),
                binding(8, &angles),
            ],
        });
        Ok(Self {
            joints: joints.to_vec(),
            fixed_joints: fixed_joints.to_vec(),
            revolute_joints: revolute_joints.to_vec(),
            prismatic_joints: prismatic_joints.to_vec(),
            revolute_motors: vec![None; revolute_joints.len()],
            revolute_servos: vec![None; revolute_joints.len()],
            revolute_limits: vec![None; revolute_joints.len()],
            prismatic_motors: vec![None; prismatic_joints.len()],
            prismatic_servos: vec![None; prismatic_joints.len()],
            prismatic_limits: vec![None; prismatic_joints.len()],
            packed_joints: packed,
            joint_buffer,
            pipeline,
            coupled_warm_pipeline,
            coupled_iteration_pipeline,
            capture_pipeline,
            activity_pipeline,
            correction_pipeline,
            angle_pipeline,
            sample_angle_pipeline,
            drive_capture_pipeline,
            bind_group,
            impulses,
            angular_impulses,
            angles,
            params,
            island_count,
            body_count,
            max_island_joints,
            last_dt_bits: None,
            temporal_resources: [
                state_buffer.clone(),
                range_buffer,
                index_buffer,
                velocities_before_solve,
            ],
            temporal_active: false,
        })
    }

    /// Current joint topology in stable insertion order.
    pub fn joints(&self) -> &[GpuRigidBallJoint] {
        &self.joints
    }

    /// Current fixed joints in stable insertion order.
    pub fn fixed_joints(&self) -> &[GpuRigidFixedJoint] {
        &self.fixed_joints
    }

    /// Current revolute joints in stable insertion order.
    pub fn revolute_joints(&self) -> &[GpuRigidRevoluteJoint] {
        &self.revolute_joints
    }

    /// Current prismatic joints in stable insertion order.
    pub fn prismatic_joints(&self) -> &[GpuRigidPrismaticJoint] {
        &self.prismatic_joints
    }

    /// Current motor on a revolute joint, if configured.
    pub fn revolute_motor(&self, index: usize) -> Option<Option<GpuRigidAxisMotor>> {
        self.revolute_motors.get(index).copied()
    }

    /// Current position and velocity servo on a revolute joint.
    pub fn revolute_servo(&self, index: usize) -> Option<Option<GpuRigidAxisServo>> {
        self.revolute_servos.get(index).copied()
    }

    /// Current wrapped angle limits on a revolute joint.
    pub fn revolute_limit(&self, index: usize) -> Option<Option<GpuRigidRevoluteLimit>> {
        self.revolute_limits.get(index).copied()
    }

    /// Current motor on a prismatic joint, if configured.
    pub fn prismatic_motor(&self, index: usize) -> Option<Option<GpuRigidAxisMotor>> {
        self.prismatic_motors.get(index).copied()
    }

    /// Current position and velocity servo on a prismatic joint.
    pub fn prismatic_servo(&self, index: usize) -> Option<Option<GpuRigidAxisServo>> {
        self.prismatic_servos.get(index).copied()
    }

    /// Current displacement limits on a prismatic joint, if configured.
    pub fn prismatic_limit(&self, index: usize) -> Option<Option<GpuRigidPrismaticLimit>> {
        self.prismatic_limits.get(index).copied()
    }

    /// Change the velocity drive on one revolute joint.
    pub fn set_revolute_motor(
        &mut self,
        queue: &wgpu::Queue,
        index: usize,
        motor: Option<GpuRigidAxisMotor>,
    ) -> Result<(), GpuRigidBallJointError> {
        if index >= self.revolute_motors.len() || motor.is_some_and(|motor| !motor.is_valid()) {
            return Err(GpuRigidBallJointError::InvalidInput);
        }
        self.revolute_motors[index] = motor;
        if motor.is_some() {
            self.revolute_servos[index] = None;
        }
        self.refresh_revolute_settings(queue, index);
        Ok(())
    }

    /// Change the position and velocity servo on one revolute joint.
    pub fn set_revolute_servo(
        &mut self,
        queue: &wgpu::Queue,
        index: usize,
        servo: Option<GpuRigidAxisServo>,
    ) -> Result<(), GpuRigidBallJointError> {
        if index >= self.revolute_servos.len() || servo.is_some_and(|servo| !servo.is_valid()) {
            return Err(GpuRigidBallJointError::InvalidInput);
        }
        self.revolute_servos[index] = servo;
        if servo.is_some() {
            self.revolute_motors[index] = None;
        }
        self.refresh_revolute_settings(queue, index);
        Ok(())
    }

    /// Change the continuous angle limits on one revolute joint.
    pub fn set_revolute_limit(
        &mut self,
        queue: &wgpu::Queue,
        index: usize,
        limit: Option<GpuRigidRevoluteLimit>,
    ) -> Result<(), GpuRigidBallJointError> {
        if index >= self.revolute_limits.len() || limit.is_some_and(|limit| !limit.is_valid()) {
            return Err(GpuRigidBallJointError::InvalidInput);
        }
        self.revolute_limits[index] = limit;
        self.refresh_revolute_settings(queue, index);
        Ok(())
    }

    /// Change the velocity drive on one prismatic joint.
    pub fn set_prismatic_motor(
        &mut self,
        queue: &wgpu::Queue,
        index: usize,
        motor: Option<GpuRigidAxisMotor>,
    ) -> Result<(), GpuRigidBallJointError> {
        if index >= self.prismatic_motors.len() || motor.is_some_and(|motor| !motor.is_valid()) {
            return Err(GpuRigidBallJointError::InvalidInput);
        }
        self.prismatic_motors[index] = motor;
        if motor.is_some() {
            self.prismatic_servos[index] = None;
        }
        self.refresh_prismatic_settings(queue, index);
        Ok(())
    }

    /// Change the position and velocity servo on one prismatic joint.
    pub fn set_prismatic_servo(
        &mut self,
        queue: &wgpu::Queue,
        index: usize,
        servo: Option<GpuRigidAxisServo>,
    ) -> Result<(), GpuRigidBallJointError> {
        if index >= self.prismatic_servos.len() || servo.is_some_and(|servo| !servo.is_valid()) {
            return Err(GpuRigidBallJointError::InvalidInput);
        }
        self.prismatic_servos[index] = servo;
        if servo.is_some() {
            self.prismatic_motors[index] = None;
        }
        self.refresh_prismatic_settings(queue, index);
        Ok(())
    }

    /// Change the displacement limits on one prismatic joint.
    pub fn set_prismatic_limit(
        &mut self,
        queue: &wgpu::Queue,
        index: usize,
        limit: Option<GpuRigidPrismaticLimit>,
    ) -> Result<(), GpuRigidBallJointError> {
        if index >= self.prismatic_limits.len() || limit.is_some_and(|limit| !limit.is_valid()) {
            return Err(GpuRigidBallJointError::InvalidInput);
        }
        self.prismatic_limits[index] = limit;
        self.refresh_prismatic_settings(queue, index);
        Ok(())
    }

    fn refresh_revolute_settings(&mut self, queue: &wgpu::Queue, index: usize) {
        let packed_index = self.joints.len() + self.fixed_joints.len() + index;
        let limit = self.revolute_limits[index].map(|limit| (limit.min, limit.max));
        self.update_axis_settings(
            queue,
            packed_index,
            self.revolute_motors[index],
            self.revolute_servos[index],
            limit,
        );
    }

    fn refresh_prismatic_settings(&mut self, queue: &wgpu::Queue, index: usize) {
        let packed_index =
            self.joints.len() + self.fixed_joints.len() + self.revolute_joints.len() + index;
        let limit = self.prismatic_limits[index].map(|limit| (limit.min, limit.max));
        self.update_axis_settings(
            queue,
            packed_index,
            self.prismatic_motors[index],
            self.prismatic_servos[index],
            limit,
        );
    }

    fn update_axis_settings(
        &mut self,
        queue: &wgpu::Queue,
        index: usize,
        motor: Option<GpuRigidAxisMotor>,
        servo: Option<GpuRigidAxisServo>,
        limit: Option<(f32, f32)>,
    ) {
        let packed = &mut self.packed_joints[index];
        packed.bodies[3] = u32::from(motor.is_some())
            | (u32::from(limit.is_some()) << 1)
            | (u32::from(servo.is_some()) << 2);
        packed.drive = [
            motor.map_or_else(
                || servo.map_or(0.0, |servo| servo.velocity_target),
                |motor| motor.target_velocity,
            ),
            motor.map_or_else(
                || servo.map_or(0.0, |servo| servo.max_force),
                |motor| motor.max_force,
            ),
            limit.map_or(0.0, |limit| limit.0),
            limit.map_or(0.0, |limit| limit.1),
        ];
        packed.servo = [
            servo.map_or(0.0, |servo| servo.position_target),
            servo.map_or(0.0, |servo| servo.stiffness),
            servo.map_or(0.0, |servo| servo.damping),
            0.0,
        ];
        queue.write_buffer(
            &self.joint_buffer,
            (index * size_of::<PackedJoint>()) as u64,
            bytemuck::bytes_of(packed),
        );
        self.clear_impulse_cache(queue);
    }

    /// Forget warm impulses while keeping continuous revolute angle history.
    pub fn clear_impulse_cache(&mut self, queue: &wgpu::Queue) {
        let zeros = vec![
            [0.0f32; 4];
            self.joints.len()
                + self.fixed_joints.len()
                + self.revolute_joints.len()
                + self.prismatic_joints.len()
        ];
        queue.write_buffer(&self.impulses, 0, bytemuck::cast_slice(&zeros));
        queue.write_buffer(&self.angular_impulses, 0, bytemuck::cast_slice(&zeros));
        self.last_dt_bits = None;
    }

    /// Forget impulses and angle history after an explicit world reset.
    pub fn clear_cache(&mut self, queue: &wgpu::Queue) {
        self.clear_impulse_cache(queue);
        queue.write_buffer(
            &self.angles,
            0,
            bytemuck::cast_slice(&vec![AngleState::zeroed(); self.packed_joints.len()]),
        );
    }

    /// Copy one unchanged revolute joint's angle history across topology rebuilds.
    pub(crate) fn encode_copy_revolute_angle_from(
        &self,
        previous: &Self,
        previous_index: usize,
        index: usize,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let previous_slot = previous.joints.len() + previous.fixed_joints.len() + previous_index;
        let slot = self.joints.len() + self.fixed_joints.len() + index;
        encoder.copy_buffer_to_buffer(
            &previous.angles,
            (previous_slot * size_of::<AngleState>()) as u64,
            &self.angles,
            (slot * size_of::<AngleState>()) as u64,
            size_of::<AngleState>() as u64,
        );
    }

    /// Forget impulses touching one replaced body.
    pub fn invalidate_body(&mut self, queue: &wgpu::Queue, body: usize) {
        let edges = self
            .joints
            .iter()
            .map(|joint| (joint.body_a, joint.body_b))
            .chain(
                self.fixed_joints
                    .iter()
                    .map(|joint| (joint.body_a, joint.body_b)),
            )
            .chain(
                self.revolute_joints
                    .iter()
                    .map(|joint| (joint.body_a, joint.body_b)),
            )
            .chain(
                self.prismatic_joints
                    .iter()
                    .map(|joint| (joint.body_a, joint.body_b)),
            );
        for (index, (body_a, body_b)) in edges.enumerate() {
            if body_a as usize == body || body_b as usize == body {
                queue.write_buffer(
                    &self.impulses,
                    (index * size_of::<[f32; 4]>()) as u64,
                    bytemuck::bytes_of(&[0.0f32; 4]),
                );
                queue.write_buffer(
                    &self.angular_impulses,
                    (index * size_of::<[f32; 4]>()) as u64,
                    bytemuck::bytes_of(&[0.0f32; 4]),
                );
                queue.write_buffer(
                    &self.angles,
                    (index * size_of::<AngleState>()) as u64,
                    bytemuck::bytes_of(&AngleState::zeroed()),
                );
            }
        }
    }

    /// Propagate captured prescribed motion through disjoint joint islands.
    pub(crate) fn encode_kinematic_activity(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        moving: &wgpu::Buffer,
    ) {
        if self.island_count == 0 {
            return;
        }
        let buffers = [
            &self.joint_buffer,
            &self.temporal_resources[1],
            &self.temporal_resources[2],
            moving,
        ];
        let entries = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>();
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera prescribed joint island activity"),
            layout: &self.activity_pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Tessera propagate prescribed joint island motion"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.activity_pipeline);
        pass.set_bind_group(0, &bindings, &[]);
        pass.dispatch_workgroups(self.island_count.div_ceil(64), 1, 1);
    }

    /// Snapshot integrated velocities before contact and joint impulses.
    pub fn encode_capture_velocity(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Tessera resident joint velocity capture pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.capture_pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.body_count.div_ceil(64), 1, 1);
    }

    /// Integrate the difference between pre-solve and post-solve velocity.
    pub fn encode_pose_correction(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Tessera resident joint pose correction pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.correction_pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.body_count.div_ceil(64), 1, 1);
    }

    /// Record a continuous angle sample for every revolute joint.
    pub fn encode_update_angles(&self, encoder: &mut wgpu::CommandEncoder) {
        if self.revolute_joints.is_empty() {
            return;
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Tessera resident revolute angle update pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.angle_pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.packed_joints.len().div_ceil(64) as u32, 1, 1);
    }

    /// Sample current orientation without inferring a new full turn from velocity.
    pub fn encode_sample_angles(&self, encoder: &mut wgpu::CommandEncoder) {
        if self.revolute_joints.is_empty() {
            return;
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Tessera resident revolute angle sample pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.sample_angle_pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.packed_joints.len().div_ceil(64) as u32, 1, 1);
    }

    /// Read the latest unwrapped angle of one revolute joint.
    pub fn readback_revolute_angle(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        index: usize,
    ) -> Result<f32, GpuRigidBallJointError> {
        if index >= self.revolute_joints.len() {
            return Err(GpuRigidBallJointError::InvalidInput);
        }
        let packed_index = self.joints.len() + self.fixed_joints.len() + index;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera resident revolute angle readback"),
            size: size_of::<AngleState>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera resident revolute angle readback encoder"),
        });
        self.encode_sample_angles(&mut encoder);
        encoder.copy_buffer_to_buffer(
            &self.angles,
            (packed_index * size_of::<AngleState>()) as u64,
            &staging,
            0,
            size_of::<AngleState>() as u64,
        );
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
            .map_err(|error| GpuRigidBallJointError::Readback(error.to_string()))?;
        receiver
            .recv_timeout(Duration::from_secs(5))
            .map_err(|error| GpuRigidBallJointError::Readback(error.to_string()))?
            .map_err(|error| GpuRigidBallJointError::Readback(error.to_string()))?;
        let view = staging.slice(..).get_mapped_range();
        let state = bytemuck::pod_read_unaligned::<AngleState>(&view);
        drop(view);
        staging.unmap();
        Ok(state.angle)
    }

    pub(crate) fn validate_temporal(
        &self,
        dt: f32,
        settings: GpuRigidTemporalJointParams,
    ) -> Result<[f32; 4], GpuRigidBallJointError> {
        if !dt.is_finite()
            || dt <= 0.0
            || !settings.frequency.is_finite()
            || settings.frequency <= 0.0
            || !settings.damping_ratio.is_finite()
            || settings.damping_ratio < 0.0
            || !settings.max_linear_correction_speed.is_finite()
            || settings.max_linear_correction_speed <= 0.0
            || !settings.max_angular_correction_speed.is_finite()
            || settings.max_angular_correction_speed <= 0.0
            || settings.iterations == 0
        {
            return Err(GpuRigidBallJointError::InvalidInput);
        }
        if u64::from(settings.iterations) * u64::from(self.max_island_joints) > 20_000 {
            return Err(GpuRigidBallJointError::Capacity);
        }
        let omega = 2.0 * core::f32::consts::PI * settings.frequency;
        let a1 = 2.0 * settings.damping_ratio + dt * omega;
        let a2 = dt * omega * a1;
        let inverse = 1.0 / (1.0 + a2);
        let coefficients = [a2 * inverse, inverse, omega / a1, 1.0];
        if coefficients.iter().any(|value| !value.is_finite()) {
            return Err(GpuRigidBallJointError::InvalidInput);
        }
        Ok(coefficients)
    }

    /// Prepare immutable soft bias and rigid relaxation coefficients for joint constraints.
    ///
    /// Clear incompatible impulse history in the supplied encoder before encoding
    /// the returned passes. Capture servo targets once before each substep warm start.
    pub fn prepare_temporal(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        dt: f32,
        settings: GpuRigidTemporalJointParams,
    ) -> Result<GpuRigidTemporalJointStep, GpuRigidBallJointError> {
        let coefficients = self.validate_temporal(dt, settings)?;
        let group = |relax: bool| {
            let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tessera immutable temporal joint coefficients"),
                contents: bytemuck::bytes_of(&SolveParams {
                    dt,
                    bias: 0.0,
                    iterations: settings.iterations,
                    islands: self.island_count,
                    bodies: self.body_count,
                    padding: [0; 3],
                    temporal: [
                        coefficients[0],
                        coefficients[1],
                        coefficients[2],
                        if relax { -1.0 } else { 1.0 },
                    ],
                    correction_limits: [
                        settings.max_linear_correction_speed,
                        settings.max_angular_correction_speed,
                        0.0,
                        0.0,
                    ],
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Tessera temporal joint bindings"),
                layout: &self.coupled_iteration_pipeline.get_bind_group_layout(0),
                entries: &[
                    binding(0, &self.temporal_resources[0]),
                    binding(1, &self.joint_buffer),
                    binding(2, &self.temporal_resources[1]),
                    binding(3, &self.temporal_resources[2]),
                    binding(4, &self.impulses),
                    binding(5, &params),
                    binding(6, &self.angular_impulses),
                    binding(7, &self.temporal_resources[3]),
                    binding(8, &self.angles),
                ],
            })
        };
        let result = GpuRigidTemporalJointStep {
            bias: group(false),
            relax: group(true),
            warm_pipeline: self.coupled_warm_pipeline.clone(),
            iteration_pipeline: self.coupled_iteration_pipeline.clone(),
            angle_pipeline: self.angle_pipeline.clone(),
            sample_angle_pipeline: self.sample_angle_pipeline.clone(),
            drive_capture_pipeline: self.drive_capture_pipeline.clone(),
            joint_count: self.packed_joints.len() as u32,
            islands: self.island_count,
            iterations: settings.iterations,
        };
        if !self.temporal_active || self.last_dt_bits != Some(dt.to_bits()) {
            encoder.clear_buffer(&self.impulses, 0, None);
            encoder.clear_buffer(&self.angular_impulses, 0, None);
        }
        self.last_dt_bits = Some(dt.to_bits());
        self.temporal_active = true;
        Ok(result)
    }

    /// Encode joint impulses after rigid contact resolution.
    pub fn encode(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        dt: f32,
        iterations: u32,
        bias: f32,
    ) -> Result<(), GpuRigidBallJointError> {
        self.prepare_solve(queue, encoder, dt, iterations, bias)?;
        self.encode_update_angles(encoder);
        self.encode_pipeline(encoder, &self.pipeline);
        Ok(())
    }

    /// Warm start once before alternating contact and joint iterations.
    pub fn encode_coupled_warm(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        dt: f32,
        iterations: u32,
        bias: f32,
    ) -> Result<(), GpuRigidBallJointError> {
        self.prepare_solve(queue, encoder, dt, iterations, bias)?;
        self.encode_update_angles(encoder);
        self.encode_pipeline(encoder, &self.coupled_warm_pipeline);
        Ok(())
    }

    /// Solve one joint iteration using impulses accumulated by earlier passes.
    pub fn encode_coupled_iteration(&self, encoder: &mut wgpu::CommandEncoder) {
        self.encode_pipeline(encoder, &self.coupled_iteration_pipeline);
    }

    fn encode_pipeline(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Tessera resident joint constraint pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.island_count.div_ceil(64), 1, 1);
    }

    fn prepare_solve(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        dt: f32,
        iterations: u32,
        bias: f32,
    ) -> Result<(), GpuRigidBallJointError> {
        if !dt.is_finite()
            || dt <= 0.0
            || iterations == 0
            || !bias.is_finite()
            || !(0.0..=1.0).contains(&bias)
        {
            return Err(GpuRigidBallJointError::InvalidInput);
        }
        if u64::from(iterations) * u64::from(self.max_island_joints) > 20_000 {
            return Err(GpuRigidBallJointError::Capacity);
        }
        if self.temporal_active || self.last_dt_bits != Some(dt.to_bits()) {
            encoder.clear_buffer(&self.impulses, 0, None);
            encoder.clear_buffer(&self.angular_impulses, 0, None);
            self.last_dt_bits = Some(dt.to_bits());
        }
        self.temporal_active = false;
        queue.write_buffer(
            &self.params,
            0,
            bytemuck::bytes_of(&SolveParams {
                dt,
                bias,
                iterations,
                islands: self.island_count,
                bodies: self.body_count,
                padding: [0; 3],
                temporal: [0.0; 4],
                correction_limits: [0.0; 4],
            }),
        );
        Ok(())
    }
}

fn root(parents: &mut [usize], mut node: usize) -> usize {
    while parents[node] != node {
        parents[node] = parents[parents[node]];
        node = parents[node];
    }
    node
}

fn binding(index: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding: index,
        resource: buffer.as_entire_binding(),
    }
}
