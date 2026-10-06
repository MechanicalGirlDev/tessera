//! Fixed-base articulated kinematics and generalized rigid-body inertia.
//!
//! Joint slots follow the input joint order. This module is independent of URDF
//! and supplies the coordinates needed by Tessera's contact impulse solver.

use std::collections::BTreeSet;

use nalgebra::{DMatrix, DVector, Isometry3, Matrix3, Point3, Unit, UnitQuaternion, Vector3};

/// Validate a disjoint partition that preserves every required mobility group.
/// Unspecified links remain singleton groups; coarser groups are allowed.
pub(crate) fn validate_mobility_partition(
    link_count: usize,
    required: &[Vec<usize>],
    groups: &[Vec<usize>],
) -> Result<(), ArticulationError> {
    let mut labels = (0..link_count).collect::<Vec<_>>();
    let mut used = vec![false; link_count];
    for group in groups {
        let label = group
            .iter()
            .copied()
            .min()
            .ok_or(ArticulationError::InvalidInput)?;
        for &link in group {
            if link >= link_count || used[link] {
                return Err(ArticulationError::InvalidInput);
            }
            used[link] = true;
            labels[link] = label;
        }
    }
    for group in required {
        let first = group
            .first()
            .and_then(|&link| labels.get(link))
            .ok_or(ArticulationError::InvalidInput)?;
        if group.iter().any(|&link| labels.get(link) != Some(first)) {
            return Err(ArticulationError::InvalidInput);
        }
    }
    Ok(())
}

fn append_free_branch(
    links: &mut Vec<LinkSpec>,
    joints: &mut Vec<JointSpec>,
    root: usize,
    body: usize,
    empty: &LinkSpec,
) -> usize {
    let mut parent = root;
    for axis in [Vector3::x(), Vector3::y(), Vector3::z()] {
        let child = links.len();
        links.push(empty.clone());
        joints.push(JointSpec {
            parent,
            child,
            kind: JointKind::Prismatic,
            origin: Isometry3::identity(),
            axis,
            limits: None,
        });
        parent = child;
    }
    let edge = joints.len();
    joints.push(JointSpec {
        parent,
        child: body,
        kind: JointKind::Spherical,
        origin: Isometry3::identity(),
        axis: Vector3::z(),
        limits: None,
    });
    edge
}

fn ordered_pair(a: usize, b: usize) -> (usize, usize) {
    if a < b { (a, b) } else { (b, a) }
}

fn joint_width(kind: JointKind) -> usize {
    match kind {
        JointKind::Fixed => 0,
        JointKind::Revolute | JointKind::Prismatic => 1,
        JointKind::Spherical => 3,
    }
}

fn armature_contribution(armature: f64, scale: f64) -> Option<f64> {
    if armature == 0.0 {
        return Some(0.0);
    }
    let value = armature * scale * scale;
    value.is_finite().then_some(value)
}

/// Mass and center-of-mass properties in a link's local frame.
#[derive(Debug, Clone, PartialEq)]
pub struct LinkSpec {
    /// Positive mass in kilograms. Zero-mass frame links are allowed.
    pub mass: f64,
    /// Center of mass in link coordinates, metres.
    pub center_of_mass: Vector3<f64>,
    /// Inertia tensor about the center of mass, in link coordinates.
    pub inertia: Matrix3<f64>,
}

/// Joint motion type supported by the fixed-base reference model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JointKind {
    /// A rigid weld.
    Fixed,
    /// One rotation about the joint-frame axis.
    Revolute,
    /// One translation along the joint-frame axis.
    Prismatic,
    /// Three rotations about intrinsic joint-frame X, Y, then Z axes.
    Spherical,
}

/// A parent-to-child edge in the articulated tree.
#[derive(Debug, Clone, PartialEq)]
pub struct JointSpec {
    /// Index of the parent link.
    pub parent: usize,
    /// Index of the child link.
    pub child: usize,
    /// Parent-link-to-joint transform at zero displacement.
    pub origin: Isometry3<f64>,
    /// Motion type.
    pub kind: JointKind,
    /// Rotation or translation axis in the joint frame. Ignored by spherical joints.
    pub axis: Vector3<f64>,
    /// Optional hard position limits in radians.
    pub limits: Option<(f64, f64)>,
}

/// Invalid geometry or a disconnected/cyclic joint graph.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ArticulationError {
    /// Link, joint, or state input is invalid.
    #[error("invalid articulation input")]
    InvalidInput,
    /// Joints do not describe one rooted tree.
    #[error("articulation joints must form one rooted tree")]
    InvalidTree,
}

/// Evaluated poses and joint axes for one generalized position.
#[derive(Debug, Clone)]
pub struct ArticulationPose {
    /// World pose of each link in input link order.
    pub links: Vec<Isometry3<f64>>,
    /// World origin of each revolute joint in DOF order.
    pub joint_origins: Vec<Vector3<f64>>,
    /// World axis of each revolute joint in DOF order.
    pub joint_axes: Vec<Vector3<f64>>,
    edge_origins: Vec<Vector3<f64>>,
    edge_axes: Vec<[Vector3<f64>; 3]>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct MimicRelation {
    source: usize,
    multiplier: f64,
    offset: f64,
}

/// Inertia and gravity force at one articulated pose.
#[derive(Debug, Clone)]
pub struct ArticulationDynamics {
    /// Generalized inertia matrix in revolute joint order.
    pub mass: DMatrix<f64>,
    /// Generalized gravity force in the same order.
    pub gravity_force: DVector<f64>,
}

/// Generalized rigid-body dynamics, including an optional floating root.
#[derive(Debug, Clone)]
pub struct GeneralizedDynamics {
    /// Mass matrix in base-linear, base-angular, then joint order.
    pub mass: DMatrix<f64>,
    /// Generalized gravity forces in the same order.
    pub gravity_force: DVector<f64>,
    /// Velocity-dependent generalized force subtracted from applied forces.
    pub velocity_bias: DVector<f64>,
}

#[derive(Default)]
struct DynamicsEvaluation<'a> {
    assemble_mass: bool,
    spherical: Option<&'a [Option<UnitQuaternion<f64>>]>,
}

/// Fixed-base revolute, prismatic, spherical, and fixed tree with stable input indices.
#[derive(Debug, Clone, PartialEq)]
pub struct Articulation {
    links: Vec<LinkSpec>,
    joints: Vec<JointSpec>,
    root: usize,
    incoming: Vec<Option<usize>>,
    traversal: Vec<usize>,
    joint_dofs: Vec<Option<usize>>,
    mimics: Vec<Option<MimicRelation>>,
    joint_scales: Vec<f64>,
    joint_offsets: Vec<f64>,
    joint_armatures: Vec<[f64; 3]>,
    coordinate_limits: Vec<Option<(f64, f64)>>,
    dof: usize,
    collision_exclusions: BTreeSet<(usize, usize)>,
}

/// Stable link and generalized slots for one independent free rigid body.
#[derive(Debug, Clone, PartialEq)]
pub struct FreeBodyCoordinates {
    /// Physical body link, excluding the massless translation links.
    pub link: usize,
    /// Three world-frame translation coordinates and linear velocities.
    pub translation: core::ops::Range<usize>,
    /// Three world-frame angular velocities; position values are workspace.
    pub angular: core::ops::Range<usize>,
    /// Edge receiving the body's independent quaternion orientation.
    pub spherical_edge: usize,
}

/// One inertial-root tree containing a robot and independent scene bodies.
#[derive(Debug, Clone, PartialEq)]
pub struct SceneArticulation {
    /// Composed tree for tangent quaternion mass and contact assembly.
    pub articulation: Articulation,
    /// Original robot joint slots, preserving reduced mimic coordinate order.
    pub robot_coordinates: core::ops::Range<usize>,
    /// Number of original robot edges preceding the appended scene branches.
    pub robot_edge_count: usize,
    /// Floating robot root slots, if requested.
    pub robot_root: Option<FreeBodyCoordinates>,
    /// Scene bodies in supplied order.
    pub bodies: Vec<FreeBodyCoordinates>,
}

/// World pose and origin velocities for one scene or floating robot body.
#[derive(Debug, Clone, Copy)]
pub struct SceneFreeBodyState {
    /// World pose of the physical body frame.
    pub pose: Isometry3<f64>,
    /// World-frame linear velocity of the body origin, not its center of mass.
    pub linear_velocity: Vector3<f64>,
    /// World-frame angular velocity.
    pub angular_velocity: Vector3<f64>,
}

/// Packed robot/scene coordinates with independent spherical orientations.
#[derive(Debug, Clone)]
pub struct SceneGeneralizedState {
    /// Scalar positions; angular position slots are unused workspace.
    pub positions: DVector<f64>,
    /// Physical generalized velocities in the composed coordinate order.
    pub velocities: DVector<f64>,
    /// One entry per composed edge, populated only for spherical joints.
    pub orientations: Vec<Option<UnitQuaternion<f64>>>,
}

/// Robot coordinates and independent body states extracted from a composed solve.
#[derive(Debug, Clone)]
pub struct SceneStateOutput {
    /// Original robot joint positions, retaining spherical workspace slots.
    pub robot_positions: DVector<f64>,
    /// Original robot joint velocities, using tangent spherical angular velocities.
    pub robot_velocities: DVector<f64>,
    /// Original robot edge orientations; scalar edges contain None.
    pub robot_orientations: Vec<Option<UnitQuaternion<f64>>>,
    /// World state of the floating robot origin, if enabled.
    pub robot_root: Option<SceneFreeBodyState>,
    /// Scene physical body states in supplied order, excluding massless helper links.
    pub bodies: Vec<SceneFreeBodyState>,
}

impl SceneArticulation {
    /// Map applied coordinate freezes to complete scene-body sleep diagnostics.
    /// A body sleeps only when all three translation and all three angular slots
    /// were frozen. Robot and inertial-root slots do not represent scene bodies.
    pub fn sleeping_scene_bodies(
        &self,
        coordinates: &[bool],
    ) -> Result<Vec<bool>, ArticulationError> {
        self.validate_layout()?;
        if coordinates.len() != self.articulation.dof() {
            return Err(ArticulationError::InvalidInput);
        }
        Ok(self
            .bodies
            .iter()
            .map(|body| {
                body.translation
                    .clone()
                    .chain(body.angular.clone())
                    .all(|slot| coordinates[slot])
            })
            .collect())
    }

    /// Expand robot/body sleep settings to physical and massless helper links.
    /// Body settings follow composed body order. The inertial root stays disabled.
    /// None keeps the corresponding robot/body group awake. This validates the
    /// composed layout and every supplied setting without changing the scene.
    pub fn contact_sleep_settings(
        &self,
        robot: Option<crate::sleep::SleepSettings>,
        bodies: &[Option<crate::sleep::SleepSettings>],
    ) -> Result<Vec<Option<crate::sleep::SleepSettings>>, ArticulationError> {
        if bodies.len() != self.bodies.len()
            || robot
                .iter()
                .chain(bodies.iter().flatten())
                .any(|setting| !setting.is_valid())
        {
            return Err(ArticulationError::InvalidInput);
        }
        let groups = self.contact_mobility_groups()?;
        let mut settings = vec![None; self.articulation.link_count()];
        for (group, setting) in groups
            .iter()
            .zip(core::iter::once(robot).chain(bodies.iter().copied()))
        {
            for &link in group {
                settings[link] = setting;
            }
        }
        Ok(settings)
    }

    /// Mobility groups for contact connectivity: the original robot is one group,
    /// while each free scene body owns a separate group with its translation helpers.
    /// The fixed inertial root is excluded and must never merge independent branches.
    pub fn contact_mobility_groups(&self) -> Result<Vec<Vec<usize>>, ArticulationError> {
        self.validate_layout()?;
        let inertial = self.articulation.root;
        if inertial == 0
            || self.articulation.joints[..self.robot_edge_count]
                .iter()
                .any(|joint| joint.parent >= inertial || joint.child >= inertial)
        {
            return Err(ArticulationError::InvalidInput);
        }
        let free_group = |layout: &FreeBodyCoordinates| -> Result<Vec<usize>, ArticulationError> {
            let mut group = vec![layout.link];
            let mut helper = self.articulation.joints[layout.spherical_edge].parent;
            for _ in 0..3 {
                if helper <= inertial {
                    return Err(ArticulationError::InvalidInput);
                }
                group.push(helper);
                let edge =
                    self.articulation.incoming[helper].ok_or(ArticulationError::InvalidInput)?;
                let joint = &self.articulation.joints[edge];
                if joint.kind != JointKind::Prismatic {
                    return Err(ArticulationError::InvalidInput);
                }
                helper = joint.parent;
            }
            if helper != inertial {
                return Err(ArticulationError::InvalidInput);
            }
            Ok(group)
        };
        let mut robot = (0..inertial).collect::<Vec<_>>();
        if let Some(root) = &self.robot_root {
            robot.extend(free_group(root)?.into_iter().skip(1));
        }
        let mut groups = vec![robot];
        for body in &self.bodies {
            groups.push(free_group(body)?);
        }
        let mut used = vec![false; self.articulation.link_count()];
        used[inertial] = true;
        for group in &groups {
            for &link in group {
                if used[link] {
                    return Err(ArticulationError::InvalidInput);
                }
                used[link] = true;
            }
        }
        if used.iter().any(|&value| !value) {
            return Err(ArticulationError::InvalidInput);
        }
        Ok(groups)
    }

    fn validate_layout(&self) -> Result<(), ArticulationError> {
        let n = self.articulation.dof();
        if self.robot_coordinates.start != 0
            || self.robot_coordinates.end > n
            || self.robot_edge_count > self.articulation.joints.len()
        {
            return Err(ArticulationError::InvalidInput);
        }
        let mut occupied = vec![false; n];
        occupied[self.robot_coordinates.clone()].fill(true);
        for layout in self.robot_root.iter().chain(&self.bodies) {
            if layout.translation.len() != 3
                || layout.angular.len() != 3
                || layout.translation.end > n
                || layout.angular.end > n
                || layout.translation.end != layout.angular.start
                || self
                    .articulation
                    .joint_coordinate_range(layout.spherical_edge)
                    != Some(layout.angular.clone())
                || !self
                    .articulation
                    .joints
                    .get(layout.spherical_edge)
                    .is_some_and(|joint| {
                        joint.kind == JointKind::Spherical && joint.child == layout.link
                    })
            {
                return Err(ArticulationError::InvalidInput);
            }
            let range = layout.translation.start..layout.angular.end;
            if occupied[range.clone()].iter().any(|&taken| taken) {
                return Err(ArticulationError::InvalidInput);
            }
            occupied[range].fill(true);
        }
        if occupied.iter().any(|&taken| !taken) {
            return Err(ArticulationError::InvalidInput);
        }
        Ok(())
    }

    /// Pack robot coordinates and world-frame free-body state without Euler conversion.
    /// Robot spherical velocities use their parent-side joint frames. Its pose
    /// array has one entry per original edge; scalar edges must contain None.
    pub fn pack_state(
        &self,
        robot_positions: &[f64],
        robot_velocities: &[f64],
        robot_orientations: &[Option<UnitQuaternion<f64>>],
        robot_root: Option<&SceneFreeBodyState>,
        bodies: &[SceneFreeBodyState],
    ) -> Result<SceneGeneralizedState, ArticulationError> {
        self.validate_layout()?;
        if robot_positions.len() != self.robot_coordinates.len()
            || robot_velocities.len() != self.robot_coordinates.len()
            || robot_orientations.len() != self.robot_edge_count
            || robot_root.is_some() != self.robot_root.is_some()
            || bodies.len() != self.bodies.len()
        {
            return Err(ArticulationError::InvalidInput);
        }
        let mut state = SceneGeneralizedState {
            positions: DVector::zeros(self.articulation.dof()),
            velocities: DVector::zeros(self.articulation.dof()),
            orientations: vec![None; self.articulation.joints.len()],
        };
        state.positions.as_mut_slice()[self.robot_coordinates.clone()]
            .copy_from_slice(robot_positions);
        state.velocities.as_mut_slice()[self.robot_coordinates.clone()]
            .copy_from_slice(robot_velocities);
        state.orientations[..self.robot_edge_count].copy_from_slice(robot_orientations);
        let mut write_body = |layout: &FreeBodyCoordinates, body: &SceneFreeBodyState| {
            state.positions.as_mut_slice()[layout.translation.clone()]
                .copy_from_slice(body.pose.translation.vector.as_slice());
            state.velocities.as_mut_slice()[layout.translation.clone()]
                .copy_from_slice(body.linear_velocity.as_slice());
            state.velocities.as_mut_slice()[layout.angular.clone()]
                .copy_from_slice(body.angular_velocity.as_slice());
            state.orientations[layout.spherical_edge] = Some(body.pose.rotation);
        };
        if let (Some(layout), Some(body)) = (&self.robot_root, robot_root) {
            write_body(layout, body);
        }
        for (layout, body) in self.bodies.iter().zip(bodies) {
            write_body(layout, body);
        }
        if state.velocities.iter().any(|v| !v.is_finite()) {
            return Err(ArticulationError::InvalidInput);
        }
        let _ = self.articulation.pose_with_spherical_orientations(
            Isometry3::identity(),
            state.positions.as_slice(),
            &state.orientations,
        )?;
        Ok(state)
    }

    /// Extract physical body origins and world velocities from a validated composed state.
    pub fn unpack_state(
        &self,
        state: &SceneGeneralizedState,
    ) -> Result<SceneStateOutput, ArticulationError> {
        self.validate_layout()?;
        if state.velocities.len() != self.articulation.dof()
            || state.velocities.iter().any(|v| !v.is_finite())
        {
            return Err(ArticulationError::InvalidInput);
        }
        let pose = self.articulation.pose_with_spherical_orientations(
            Isometry3::identity(),
            state.positions.as_slice(),
            &state.orientations,
        )?;
        let body = |layout: &FreeBodyCoordinates| SceneFreeBodyState {
            pose: pose.links[layout.link],
            linear_velocity: Vector3::from_column_slice(
                &state.velocities.as_slice()[layout.translation.clone()],
            ),
            angular_velocity: Vector3::from_column_slice(
                &state.velocities.as_slice()[layout.angular.clone()],
            ),
        };
        Ok(SceneStateOutput {
            robot_positions: DVector::from_column_slice(
                &state.positions.as_slice()[self.robot_coordinates.clone()],
            ),
            robot_velocities: DVector::from_column_slice(
                &state.velocities.as_slice()[self.robot_coordinates.clone()],
            ),
            robot_orientations: state.orientations[..self.robot_edge_count].to_vec(),
            robot_root: self.robot_root.as_ref().map(body),
            bodies: self.bodies.iter().map(body).collect(),
        })
    }

    /// Validate and split one GPU environment output; contact diagnostics remain on the source output.
    /// The composed world root must remain fixed at identity. Quaternion indices
    /// are matched by physical velocity slots, independent of registration order.
    #[cfg(feature = "gpu-contact")]
    pub fn split_gpu_output(
        &self,
        output: &crate::gpu_articulated_dynamics::GpuArticulatedDynamicsOutput,
    ) -> Result<SceneStateOutput, ArticulationError> {
        self.validate_layout()?;
        if output.floating_root
            || output
                .root_pose
                .translation
                .vector
                .iter()
                .any(|v| !v.is_finite())
            || output
                .root_pose
                .rotation
                .coords
                .iter()
                .any(|v| !v.is_finite())
            || output.root_pose.translation.vector.norm() > 1e-10
            || output.root_pose.rotation.angle() > 1e-10
            || (output.root_pose.rotation.quaternion().norm_squared() - 1.0).abs() > 1e-8
        {
            return Err(ArticulationError::InvalidInput);
        }
        let mut orientations = vec![None; self.articulation.joints.len()];
        if let Some(joints) = &output.spherical_joints {
            for joint in joints {
                let edge = self
                    .articulation
                    .joints
                    .iter()
                    .enumerate()
                    .find_map(|(edge, spec)| {
                        (spec.kind == JointKind::Spherical
                            && self
                                .articulation
                                .joint_coordinate_range(edge)
                                .is_some_and(|range| range.start == joint.velocity_slot))
                        .then_some(edge)
                    })
                    .ok_or(ArticulationError::InvalidInput)?;
                if orientations[edge].replace(joint.orientation).is_some() {
                    return Err(ArticulationError::InvalidInput);
                }
            }
        }
        self.unpack_state(&SceneGeneralizedState {
            positions: output.state.positions.clone(),
            velocities: output.state.velocities.clone(),
            orientations,
        })
    }

    /// Create resident quaternion inputs in stable edge order for this composed state.
    #[cfg(feature = "gpu-contact")]
    pub fn gpu_spherical_state(
        &self,
        state: &SceneGeneralizedState,
    ) -> Result<Vec<crate::gpu_articulated_spherical::GpuSphericalJointState>, ArticulationError>
    {
        self.validate_layout()?;
        let _ = self.articulation.pose_with_spherical_orientations(
            Isometry3::identity(),
            state.positions.as_slice(),
            &state.orientations,
        )?;
        if state.velocities.len() != self.articulation.dof()
            || state.velocities.iter().any(|v| !v.is_finite())
        {
            return Err(ArticulationError::InvalidInput);
        }
        state
            .orientations
            .iter()
            .enumerate()
            .filter_map(|(edge, orientation)| {
                orientation.map(|orientation| {
                    self.articulation
                        .joint_coordinate_range(edge)
                        .map(
                            |range| crate::gpu_articulated_spherical::GpuSphericalJointState {
                                velocity_slot: range.start,
                                orientation,
                            },
                        )
                        .ok_or(ArticulationError::InvalidInput)
                })
            })
            .collect()
    }
}

impl Articulation {
    /// Validate and build one connected tree. Movable slots keep joint order.
    pub fn new(
        links: Vec<LinkSpec>,
        mut joints: Vec<JointSpec>,
        root: usize,
    ) -> Result<Self, ArticulationError> {
        if links.is_empty() || root >= links.len() || joints.len() + 1 != links.len() {
            return Err(ArticulationError::InvalidTree);
        }
        if links.iter().any(|link| {
            !link.mass.is_finite()
                || link.mass < 0.0
                || link.center_of_mass.iter().any(|value| !value.is_finite())
                || link.inertia.iter().any(|value| !value.is_finite())
                || (0..3).any(|axis| link.inertia[(axis, axis)] < 0.0)
                || (link.inertia - link.inertia.transpose()).norm() > 1e-9
        }) {
            return Err(ArticulationError::InvalidInput);
        }
        let mut incoming = vec![None; links.len()];
        let mut joint_dofs = Vec::with_capacity(joints.len());
        let mut dof = 0;
        for (index, joint) in joints.iter_mut().enumerate() {
            if joint.parent >= links.len()
                || joint.child >= links.len()
                || joint.parent == joint.child
                || joint.child == root
                || incoming[joint.child].replace(index).is_some()
                || !joint
                    .origin
                    .translation
                    .vector
                    .iter()
                    .all(|value| value.is_finite())
                || !joint
                    .origin
                    .rotation
                    .coords
                    .iter()
                    .all(|value| value.is_finite())
                || joint.axis.iter().any(|value| !value.is_finite())
                || joint.limits.is_some_and(|(lower, upper)| {
                    !lower.is_finite() || !upper.is_finite() || lower > upper
                })
            {
                return Err(ArticulationError::InvalidInput);
            }
            if joint.kind == JointKind::Spherical && joint.limits.is_some() {
                return Err(ArticulationError::InvalidInput);
            }
            if matches!(joint.kind, JointKind::Revolute | JointKind::Prismatic) {
                let norm = joint.axis.norm();
                if !norm.is_finite() || norm <= 1e-12 {
                    return Err(ArticulationError::InvalidInput);
                }
                joint.axis /= norm;
                joint_dofs.push(Some(dof));
                dof += 1;
            } else if joint.kind == JointKind::Spherical {
                joint_dofs.push(Some(dof));
                dof += 3;
            } else {
                joint_dofs.push(None);
            }
        }
        if incoming.iter().enumerate().any(|(index, edge)| {
            (index == root && edge.is_some()) || (index != root && edge.is_none())
        }) {
            return Err(ArticulationError::InvalidTree);
        }
        let mut traversal = Vec::with_capacity(links.len() - 1);
        let mut visited = vec![false; links.len()];
        visited[root] = true;
        let mut frontier = vec![root];
        while let Some(parent) = frontier.pop() {
            for (index, joint) in joints.iter().enumerate() {
                if joint.parent == parent && !visited[joint.child] {
                    visited[joint.child] = true;
                    traversal.push(index);
                    frontier.push(joint.child);
                }
            }
        }
        if visited.iter().any(|seen| !seen) {
            return Err(ArticulationError::InvalidTree);
        }
        let collision_exclusions = joints
            .iter()
            .map(|joint| ordered_pair(joint.parent, joint.child))
            .collect();
        let joint_count = joints.len();
        let mut articulation = Self {
            links,
            joints,
            root,
            incoming,
            traversal,
            joint_dofs,
            mimics: vec![None; joint_count],
            joint_scales: vec![1.0; joint_count],
            joint_offsets: vec![0.0; joint_count],
            joint_armatures: vec![[0.0; 3]; joint_count],
            coordinate_limits: Vec::new(),
            dof,
            collision_exclusions,
        };
        articulation.set_mimics(&[])?;
        Ok(articulation)
    }

    /// Number of generalized joint coordinates.
    pub fn dof(&self) -> usize {
        self.dof
    }

    /// Compose a robot and free scene bodies under a fixed inertial world root.
    /// Existing robot link/edge indices, reduced coordinates, mimics, armatures,
    /// and collision exclusions are preserved. Each free body adds six physical
    /// velocity slots and three massless translation links; orientation must use
    /// the quaternion tangent APIs. The composed root pose must be identity.
    /// A fixed robot mount uses `robot_pose`; a floating robot pose is supplied
    /// through its returned translation slots and spherical orientation instead.
    pub fn compose_scene(
        &self,
        floating_robot: bool,
        robot_pose: Isometry3<f64>,
        bodies: &[LinkSpec],
    ) -> Result<SceneArticulation, ArticulationError> {
        if robot_pose
            .translation
            .vector
            .iter()
            .chain(robot_pose.rotation.coords.iter())
            .any(|v| !v.is_finite())
            || (robot_pose.rotation.quaternion().norm_squared() - 1.0).abs() > 1e-8
        {
            return Err(ArticulationError::InvalidInput);
        }
        let empty = LinkSpec {
            mass: 0.0,
            center_of_mass: Vector3::zeros(),
            inertia: Matrix3::zeros(),
        };
        let mut links = self.links.clone();
        let inertial_root = links.len();
        links.push(empty.clone());
        let mut joints = self.joints.clone();
        let mut root_edge = None;
        if floating_robot {
            root_edge = Some(append_free_branch(
                &mut links,
                &mut joints,
                inertial_root,
                self.root,
                &empty,
            ));
        } else {
            joints.push(JointSpec {
                parent: inertial_root,
                child: self.root,
                kind: JointKind::Fixed,
                origin: robot_pose,
                axis: Vector3::z(),
                limits: None,
            });
        }
        let mut body_edges = Vec::with_capacity(bodies.len());
        for body in bodies {
            let link = links.len();
            links.push(body.clone());
            let edge = append_free_branch(&mut links, &mut joints, inertial_root, link, &empty);
            body_edges.push((link, edge));
        }
        let mut articulation = Self::new(links, joints, inertial_root)?;
        let relations = self
            .mimics
            .iter()
            .enumerate()
            .filter_map(|(dependent, relation)| {
                relation.map(|relation| {
                    (
                        dependent,
                        relation.source,
                        relation.multiplier,
                        relation.offset,
                    )
                })
            })
            .collect::<Vec<_>>();
        articulation.set_mimics(&relations)?;
        for edge in 0..self.joints.len() {
            articulation.set_joint_armature(
                edge,
                self.joint_armature(edge)
                    .ok_or(ArticulationError::InvalidInput)?,
            )?;
        }
        articulation
            .collision_exclusions
            .extend(self.collision_exclusions.iter().copied());
        let layout = |link, edge| -> Result<FreeBodyCoordinates, ArticulationError> {
            let angular = articulation
                .joint_coordinate_range(edge)
                .ok_or(ArticulationError::InvalidInput)?;
            let translation_start = articulation
                .joint_coordinate_range(edge - 3)
                .ok_or(ArticulationError::InvalidInput)?
                .start;
            Ok(FreeBodyCoordinates {
                link,
                translation: translation_start..translation_start + 3,
                angular,
                spherical_edge: edge,
            })
        };
        let robot_root = root_edge.map(|edge| layout(self.root, edge)).transpose()?;
        let bodies = body_edges
            .into_iter()
            .map(|(link, edge)| layout(link, edge))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(SceneArticulation {
            articulation,
            robot_coordinates: 0..self.dof,
            robot_edge_count: self.joints.len(),
            robot_root,
            bodies,
        })
    }

    /// Number of links in stable input order.
    pub fn link_count(&self) -> usize {
        self.links.len()
    }

    #[cfg(feature = "gpu-contact")]
    pub(crate) fn gpu_pose_topology(&self) -> (usize, &[usize], &[JointSpec]) {
        (self.root, &self.traversal, &self.joints)
    }

    #[cfg(feature = "gpu-contact")]
    pub(crate) fn joint_coordinate_offset(&self, edge: usize) -> Option<f64> {
        self.joints.get(edge).map(|_| self.joint_offsets[edge])
    }

    /// Mass properties for one link in stable input order.
    pub fn link(&self, index: usize) -> Option<&LinkSpec> {
        self.links.get(index)
    }

    /// Couple scalar joints without adding dependent generalized coordinates.
    ///
    /// Each entry is `(dependent_edge, source_edge, multiplier, offset)` and
    /// imposes `q_dependent = multiplier * q_source + offset`. Chains are allowed;
    /// cycles, incompatible joints, and contradictory limits are rejected.
    pub fn set_mimics(
        &mut self,
        relations: &[(usize, usize, f64, f64)],
    ) -> Result<(), ArticulationError> {
        let mut mimics = vec![None; self.joints.len()];
        for &(dependent, source, multiplier, offset) in relations {
            if dependent >= self.joints.len()
                || source >= self.joints.len()
                || dependent == source
                || !multiplier.is_finite()
                || !offset.is_finite()
                || !matches!(
                    self.joints[dependent].kind,
                    JointKind::Revolute | JointKind::Prismatic
                )
                || !matches!(
                    self.joints[source].kind,
                    JointKind::Revolute | JointKind::Prismatic
                )
                || mimics[dependent].is_some()
            {
                return Err(ArticulationError::InvalidInput);
            }
            mimics[dependent] = Some(MimicRelation {
                source,
                multiplier,
                offset,
            });
        }
        let mut joint_dofs = vec![None; self.joints.len()];
        let mut dof = 0;
        for (edge, joint) in self.joints.iter().enumerate() {
            if mimics[edge].is_some() {
                continue;
            }
            match joint.kind {
                JointKind::Revolute | JointKind::Prismatic => {
                    joint_dofs[edge] = Some(dof);
                    dof += 1;
                }
                JointKind::Spherical => {
                    joint_dofs[edge] = Some(dof);
                    dof += 3;
                }
                JointKind::Fixed => {}
            }
        }
        let mut joint_scales = vec![1.0; self.joints.len()];
        let mut joint_offsets = vec![0.0; self.joints.len()];
        for edge in 0..self.joints.len() {
            if mimics[edge].is_none() {
                continue;
            }
            let mut seen = BTreeSet::new();
            let mut current = edge;
            let mut scale = 1.0;
            let mut offset = 0.0;
            while let Some(relation) = mimics[current] {
                if !seen.insert(current) {
                    return Err(ArticulationError::InvalidInput);
                }
                offset += scale * relation.offset;
                scale *= relation.multiplier;
                current = relation.source;
            }
            if !scale.is_finite() || !offset.is_finite() {
                return Err(ArticulationError::InvalidInput);
            }
            joint_dofs[edge] = joint_dofs[current];
            joint_scales[edge] = scale;
            joint_offsets[edge] = offset;
        }
        for (edge, armatures) in self.joint_armatures.iter().enumerate() {
            let width = joint_width(self.joints[edge].kind);
            if armatures[..width]
                .iter()
                .any(|armature| armature_contribution(*armature, joint_scales[edge]).is_none())
            {
                return Err(ArticulationError::InvalidInput);
            }
        }
        let mut coordinate_limits: Vec<Option<(f64, f64)>> = vec![None; dof];
        for (edge, joint) in self.joints.iter().enumerate() {
            let Some((lower, upper)) = joint.limits else {
                continue;
            };
            let Some(slot) = joint_dofs[edge] else {
                return Err(ArticulationError::InvalidInput);
            };
            let scale = joint_scales[edge];
            let offset = joint_offsets[edge];
            if scale == 0.0 {
                if offset < lower || offset > upper {
                    return Err(ArticulationError::InvalidInput);
                }
                continue;
            }
            let first = (lower - offset) / scale;
            let second = (upper - offset) / scale;
            if !first.is_finite() || !second.is_finite() {
                return Err(ArticulationError::InvalidInput);
            }
            let interval = (first.min(second), first.max(second));
            coordinate_limits[slot] = Some(match coordinate_limits[slot] {
                Some((old_lower, old_upper)) => {
                    (old_lower.max(interval.0), old_upper.min(interval.1))
                }
                None => interval,
            });
            if coordinate_limits[slot].is_some_and(|(low, high)| low > high) {
                return Err(ArticulationError::InvalidInput);
            }
        }
        self.mimics = mimics;
        self.joint_dofs = joint_dofs;
        self.joint_scales = joint_scales;
        self.joint_offsets = joint_offsets;
        self.coordinate_limits = coordinate_limits;
        self.dof = dof;
        Ok(())
    }

    /// Generalized-coordinate range used by an edge. Mimics share their source range.
    pub fn joint_coordinate_range(&self, edge: usize) -> Option<core::ops::Range<usize>> {
        let joint = self.joints.get(edge)?;
        if let Some(start) = self.joint_dofs[edge] {
            let width = if joint.kind == JointKind::Spherical {
                3
            } else {
                1
            };
            return Some(start..start + width);
        }
        let start = self
            .joints
            .iter()
            .enumerate()
            .take(edge)
            .filter(|(index, _)| self.mimics[*index].is_none())
            .map(|(_, joint)| match joint.kind {
                JointKind::Revolute | JointKind::Prismatic => 1,
                JointKind::Spherical => 3,
                JointKind::Fixed => 0,
            })
            .sum();
        Some(start..start)
    }

    /// Links whose poses can depend on each generalized coordinate.
    ///
    /// Floating-root translation and tangent angular coordinates occupy the first
    /// six entries. Joint coordinates follow in reduced mimic order. Shared mimic
    /// coordinates include descendants of every nonzero-scale dependent edge.
    /// Owners use stable link order, include massless helpers, and are independent
    /// of the current pose or accidental Jacobian zeros. This is a kinematic map;
    /// dynamically coupled coordinates also need a common mobility group.
    pub fn coordinate_affected_links(&self, floating_root: bool) -> Vec<Vec<usize>> {
        let root_dofs = if floating_root { 6 } else { 0 };
        let mut owners = vec![Vec::new(); root_dofs + self.dof];
        for group in &mut owners[..root_dofs] {
            group.extend(0..self.links.len());
        }
        for link in 0..self.links.len() {
            let mut ancestor = link;
            while let Some(edge) = self.incoming[ancestor] {
                if self.joint_scales[edge] != 0.0
                    && let Some(range) = self.joint_coordinate_range(edge)
                {
                    for coordinate in range {
                        let group = &mut owners[root_dofs + coordinate];
                        if group.last() != Some(&link) {
                            group.push(link);
                        }
                    }
                }
                ancestor = self.joints[edge].parent;
            }
        }
        owners
    }

    /// Conservative mobility groups for generalized-coordinate sleep decisions.
    ///
    /// A fixed root separates independent branches. Every other joint, including
    /// fixed edges and massless helpers, connects its parent and child. Mimic
    /// owners and explicit coordinate couplings merge branches. Floating roots
    /// connect the whole tree. Fixed roots are omitted from the returned groups.
    /// Contact and bilateral constraints may merge these groups further at runtime.
    pub fn coordinate_mobility_groups(
        &self,
        floating_root: bool,
        coordinate_couplings: &[(usize, usize)],
    ) -> Result<Vec<Vec<usize>>, ArticulationError> {
        let owners = self.coordinate_affected_links(floating_root);
        if coordinate_couplings
            .iter()
            .any(|&(a, b)| a >= owners.len() || b >= owners.len())
        {
            return Err(ArticulationError::InvalidInput);
        }
        let mut parents = (0..self.links.len()).collect::<Vec<_>>();
        fn root(parents: &[usize], mut link: usize) -> usize {
            while parents[link] != link {
                link = parents[link];
            }
            link
        }
        let connect = |parents: &mut [usize], a, b| {
            let a = root(parents, a);
            let b = root(parents, b);
            parents[a.max(b)] = a.min(b);
        };
        for joint in &self.joints {
            if floating_root || joint.parent != self.root {
                connect(&mut parents, joint.parent, joint.child);
            }
        }
        for owner in &owners {
            if let Some(&first) = owner.first() {
                for &link in &owner[1..] {
                    connect(&mut parents, first, link);
                }
            }
        }
        for &(a, b) in coordinate_couplings {
            if let (Some(&first), Some(&second)) = (owners[a].first(), owners[b].first()) {
                connect(&mut parents, first, second);
            }
        }
        let mut groups = vec![Vec::new(); self.links.len()];
        for link in 0..self.links.len() {
            if floating_root || link != self.root {
                groups[root(&parents, link)].push(link);
            }
        }
        Ok(groups
            .into_iter()
            .filter(|group| !group.is_empty())
            .collect())
    }

    /// Check that supplied sleep groups retain every topology and coupling group.
    /// Missing singleton links remain independent; larger conservative groups are valid.
    pub fn validate_coordinate_mobility_groups(
        &self,
        floating_root: bool,
        coordinate_couplings: &[(usize, usize)],
        groups: &[Vec<usize>],
    ) -> Result<(), ArticulationError> {
        let required = self.coordinate_mobility_groups(floating_root, coordinate_couplings)?;
        validate_mobility_partition(self.links.len(), &required, groups)
    }

    /// Multiplier from a shared generalized coordinate to this joint edge.
    pub fn joint_coordinate_scale(&self, edge: usize) -> Option<f64> {
        self.joints.get(edge).map(|_| self.joint_scales[edge])
    }

    /// Total mass and world-space center of mass at an evaluated pose.
    pub fn center_of_mass(
        &self,
        pose: &ArticulationPose,
    ) -> Result<(f64, Vector3<f64>), ArticulationError> {
        if pose.links.len() != self.links.len() {
            return Err(ArticulationError::InvalidInput);
        }
        let mut mass = 0.0;
        let mut weighted_position = Vector3::zeros();
        for (link, frame) in self.links.iter().zip(&pose.links) {
            mass += link.mass;
            weighted_position += link.mass
                * frame
                    .transform_point(&Point3::from(link.center_of_mass))
                    .coords;
        }
        if !mass.is_finite() || mass <= 0.0 {
            return Err(ArticulationError::InvalidInput);
        }
        Ok((mass, weighted_position / mass))
    }

    /// Exclude one valid link pair from self-collision detection.
    ///
    /// Direct joint edges are excluded automatically. Loaders may add an exclusion
    /// when internal massless links expand one logical parent-child body edge.
    pub fn exclude_collision_pair(&mut self, a: usize, b: usize) -> Result<(), ArticulationError> {
        if a >= self.links.len() || b >= self.links.len() || a == b {
            return Err(ArticulationError::InvalidInput);
        }
        let _inserted = self.collision_exclusions.insert(ordered_pair(a, b));
        Ok(())
    }

    /// Whether a pair should skip articulated self-collision.
    pub fn adjacent(&self, a: usize, b: usize) -> bool {
        self.collision_exclusions.contains(&ordered_pair(a, b))
    }

    /// Hard position limit of a scalar revolute or prismatic slot, when present.
    pub fn joint_limit(&self, slot: usize) -> Option<(f64, f64)> {
        self.coordinate_limits.get(slot).copied().flatten()
    }

    /// Reflected rotor inertia for one joint edge, in its local coordinate order.
    /// Mimic joints contribute the square of their coupling multiplier to the
    /// shared generalized mass diagonal.
    pub fn set_joint_armature(
        &mut self,
        edge: usize,
        armatures: &[f64],
    ) -> Result<(), ArticulationError> {
        let joint = self
            .joints
            .get(edge)
            .ok_or(ArticulationError::InvalidInput)?;
        let width = joint_width(joint.kind);
        if armatures.len() != width
            || armatures.iter().any(|armature| {
                !armature.is_finite()
                    || *armature < 0.0
                    || armature_contribution(*armature, self.joint_scales[edge]).is_none()
            })
        {
            return Err(ArticulationError::InvalidInput);
        }
        self.joint_armatures[edge][..width].copy_from_slice(armatures);
        Ok(())
    }

    /// Reflected rotor inertia assigned to one joint edge.
    pub fn joint_armature(&self, edge: usize) -> Option<&[f64]> {
        let width = joint_width(self.joints.get(edge)?.kind);
        Some(&self.joint_armatures[edge][..width])
    }

    fn add_joint_armature(
        &self,
        mass: &mut DMatrix<f64>,
        base_offset: usize,
    ) -> Result<(), ArticulationError> {
        for (edge, armatures) in self.joint_armatures.iter().enumerate() {
            let Some(start) = self.joint_dofs[edge] else {
                continue;
            };
            for (axis, armature) in armatures[..joint_width(self.joints[edge].kind)]
                .iter()
                .enumerate()
            {
                let contribution = armature_contribution(*armature, self.joint_scales[edge])
                    .ok_or(ArticulationError::InvalidInput)?;
                let slot = base_offset + start + axis;
                mass[(slot, slot)] += contribution;
                if !mass[(slot, slot)].is_finite() {
                    return Err(ArticulationError::InvalidInput);
                }
            }
        }
        Ok(())
    }

    /// Compute all link transforms and movable joint frames.
    pub fn pose(
        &self,
        root_pose: Isometry3<f64>,
        angles: &[f64],
    ) -> Result<ArticulationPose, ArticulationError> {
        self.pose_impl(root_pose, angles, None)
    }

    /// Evaluate quaternion spherical joints with joint-frame angular velocity slots.
    ///
    /// Supply one entry per tree edge: Some for spherical edges and None otherwise.
    /// Scalar coordinates retain their usual meaning; the three spherical position
    /// slots are ignored. Spherical Jacobian columns represent angular velocity in
    /// the parent-side joint frame, rather than intrinsic XYZ angle derivatives.
    /// This pose can be passed to point Jacobians and `dynamics` for tangent-space
    /// mass/gravity assembly. Legacy `generalized_dynamics` still uses Euler state.
    pub fn pose_with_spherical_orientations(
        &self,
        root_pose: Isometry3<f64>,
        coordinates: &[f64],
        orientations: &[Option<UnitQuaternion<f64>>],
    ) -> Result<ArticulationPose, ArticulationError> {
        if root_pose.translation.vector.iter().any(|v| !v.is_finite())
            || root_pose.rotation.coords.iter().any(|v| !v.is_finite())
            || (root_pose.rotation.norm_squared() - 1.0).abs() > 1e-8
            || orientations.len() != self.joints.len()
            || self
                .joints
                .iter()
                .zip(orientations)
                .any(|(joint, orientation)| {
                    (joint.kind == JointKind::Spherical) != orientation.is_some()
                        || orientation.is_some_and(|rotation| {
                            rotation.coords.iter().any(|v| !v.is_finite())
                                || (rotation.norm_squared() - 1.0).abs() > 1e-8
                        })
                })
        {
            return Err(ArticulationError::InvalidInput);
        }
        self.pose_impl(root_pose, coordinates, Some(orientations))
    }

    fn pose_impl(
        &self,
        root_pose: Isometry3<f64>,
        angles: &[f64],
        spherical: Option<&[Option<UnitQuaternion<f64>>]>,
    ) -> Result<ArticulationPose, ArticulationError> {
        if angles.len() != self.dof || angles.iter().any(|angle| !angle.is_finite()) {
            return Err(ArticulationError::InvalidInput);
        }
        let mut links = vec![Isometry3::identity(); self.links.len()];
        links[self.root] = root_pose;
        let mut joint_origins = vec![Vector3::zeros(); self.dof];
        let mut joint_axes = vec![Vector3::zeros(); self.dof];
        let mut edge_origins = vec![Vector3::zeros(); self.joints.len()];
        let mut edge_axes = vec![[Vector3::zeros(); 3]; self.joints.len()];
        for &index in &self.traversal {
            let joint = &self.joints[index];
            let frame = links[joint.parent] * joint.origin;
            let child_pose = if let Some(slot) = self.joint_dofs[index] {
                match joint.kind {
                    JointKind::Revolute => {
                        edge_origins[index] = frame.translation.vector;
                        edge_axes[index][0] = frame.rotation * joint.axis;
                        if self.mimics[index].is_none() {
                            joint_origins[slot] = edge_origins[index];
                            joint_axes[slot] = edge_axes[index][0];
                        }
                        let axis = Unit::new_normalize(joint.axis);
                        frame
                            * Isometry3::from_parts(
                                nalgebra::Translation3::identity(),
                                UnitQuaternion::from_axis_angle(
                                    &axis,
                                    angles[slot] * self.joint_scales[index]
                                        + self.joint_offsets[index],
                                ),
                            )
                    }
                    JointKind::Prismatic => {
                        edge_origins[index] = frame.translation.vector;
                        edge_axes[index][0] = frame.rotation * joint.axis;
                        if self.mimics[index].is_none() {
                            joint_origins[slot] = edge_origins[index];
                            joint_axes[slot] = edge_axes[index][0];
                        }
                        let position =
                            angles[slot] * self.joint_scales[index] + self.joint_offsets[index];
                        frame
                            * Isometry3::translation(
                                joint.axis.x * position,
                                joint.axis.y * position,
                                joint.axis.z * position,
                            )
                    }
                    JointKind::Spherical => {
                        let rotation = if let Some(rotation) =
                            spherical.and_then(|values| values[index])
                        {
                            joint_axes[slot] = frame.rotation * Vector3::x();
                            joint_axes[slot + 1] = frame.rotation * Vector3::y();
                            joint_axes[slot + 2] = frame.rotation * Vector3::z();
                            rotation
                        } else {
                            let rotation_x =
                                UnitQuaternion::from_axis_angle(&Vector3::x_axis(), angles[slot]);
                            let rotation_y = UnitQuaternion::from_axis_angle(
                                &Vector3::y_axis(),
                                angles[slot + 1],
                            );
                            let rotation_z = UnitQuaternion::from_axis_angle(
                                &Vector3::z_axis(),
                                angles[slot + 2],
                            );
                            joint_axes[slot] = frame.rotation * Vector3::x();
                            joint_axes[slot + 1] = frame.rotation * (rotation_x * Vector3::y());
                            joint_axes[slot + 2] =
                                frame.rotation * (rotation_x * rotation_y * Vector3::z());
                            rotation_x * rotation_y * rotation_z
                        };
                        joint_origins[slot..(slot + 3)].fill(frame.translation.vector);
                        edge_origins[index] = frame.translation.vector;
                        edge_axes[index].copy_from_slice(&joint_axes[slot..slot + 3]);
                        frame * Isometry3::from_parts(nalgebra::Translation3::identity(), rotation)
                    }
                    JointKind::Fixed => frame,
                }
            } else {
                frame
            };
            links[joint.child] = child_pose;
        }
        Ok(ArticulationPose {
            links,
            joint_origins,
            joint_axes,
            edge_origins,
            edge_axes,
        })
    }

    /// Point linear and angular Jacobians in generalized joint order.
    pub fn point_jacobians(
        &self,
        pose: &ArticulationPose,
        link: usize,
        local_point: Vector3<f64>,
    ) -> Result<(DMatrix<f64>, DMatrix<f64>), ArticulationError> {
        let Some(link_pose) = pose.links.get(link) else {
            return Err(ArticulationError::InvalidInput);
        };
        let point = link_pose.transform_point(&Point3::from(local_point)).coords;
        let mut linear = DMatrix::zeros(3, self.dof);
        let mut angular = DMatrix::zeros(3, self.dof);
        let mut current = link;
        while let Some(edge) = self.incoming[current] {
            if let Some(slot) = self.joint_dofs[edge] {
                match self.joints[edge].kind {
                    JointKind::Revolute => {
                        let axis = pose.edge_axes[edge][0];
                        let velocity = axis.cross(&(point - pose.edge_origins[edge]));
                        let scale = self.joint_scales[edge];
                        for row in 0..3 {
                            linear[(row, slot)] += velocity[row] * scale;
                            angular[(row, slot)] += axis[row] * scale;
                        }
                    }
                    JointKind::Prismatic => {
                        let axis = pose.edge_axes[edge][0];
                        let scale = self.joint_scales[edge];
                        for row in 0..3 {
                            linear[(row, slot)] += axis[row] * scale;
                        }
                    }
                    JointKind::Spherical => {
                        for component in 0..3 {
                            let spherical_slot = slot + component;
                            let axis = pose.edge_axes[edge][component];
                            let velocity = axis.cross(&(point - pose.edge_origins[edge]));
                            for row in 0..3 {
                                linear[(row, spherical_slot)] += velocity[row];
                                angular[(row, spherical_slot)] += axis[row];
                            }
                        }
                    }
                    JointKind::Fixed => {}
                }
            }
            current = self.joints[edge].parent;
        }
        Ok((linear, angular))
    }

    /// Point Jacobians in world-linear, world-angular, then joint order.
    /// With a fixed root this is identical to [`Self::point_jacobians`].
    pub fn generalized_point_jacobians(
        &self,
        pose: &ArticulationPose,
        link: usize,
        local_point: Vector3<f64>,
        floating: bool,
    ) -> Result<(DMatrix<f64>, DMatrix<f64>), ArticulationError> {
        let (joint_linear, joint_angular) = self.point_jacobians(pose, link, local_point)?;
        if !floating {
            return Ok((joint_linear, joint_angular));
        }
        let point = pose.links[link]
            .transform_point(&Point3::from(local_point))
            .coords;
        let offset = point - pose.links[self.root].translation.vector;
        let mut linear = DMatrix::zeros(3, 6 + self.dof);
        let mut angular = DMatrix::zeros(3, 6 + self.dof);
        for axis in 0..3 {
            linear[(axis, axis)] = 1.0;
            angular[(axis, axis + 3)] = 1.0;
            let mut basis = Vector3::zeros();
            basis[axis] = 1.0;
            let column = basis.cross(&offset);
            for row in 0..3 {
                linear[(row, axis + 3)] = column[row];
            }
        }
        for slot in 0..self.dof {
            for row in 0..3 {
                linear[(row, 6 + slot)] = joint_linear[(row, slot)];
                angular[(row, 6 + slot)] = joint_angular[(row, slot)];
            }
        }
        Ok((linear, angular))
    }

    /// Assemble mass, gravity and velocity bias for fixed or floating dynamics.
    /// The convective acceleration is a central difference of point Jacobians.
    pub fn generalized_dynamics(
        &self,
        root_pose: Isometry3<f64>,
        angles: &[f64],
        velocity: &DVector<f64>,
        floating: bool,
        gravity: Vector3<f64>,
    ) -> Result<GeneralizedDynamics, ArticulationError> {
        self.generalized_dynamics_impl(
            root_pose,
            angles,
            velocity,
            floating,
            gravity,
            DynamicsEvaluation {
                assemble_mass: true,
                spherical: None,
            },
        )
    }

    /// Assemble dynamics using quaternion spherical poses and joint-frame angular velocities.
    /// Root velocity slots remain world-frame twists. Scalar joint slots retain
    /// their usual meanings; spherical position slots are ignored. The convective
    /// term perturbs spherical quaternions with a left exponential update.
    pub fn generalized_dynamics_with_spherical_orientations(
        &self,
        root_pose: Isometry3<f64>,
        coordinates: &[f64],
        orientations: &[Option<UnitQuaternion<f64>>],
        velocity: &DVector<f64>,
        floating: bool,
        gravity: Vector3<f64>,
    ) -> Result<GeneralizedDynamics, ArticulationError> {
        self.generalized_dynamics_impl(
            root_pose,
            coordinates,
            velocity,
            floating,
            gravity,
            DynamicsEvaluation {
                assemble_mass: true,
                spherical: Some(orientations),
            },
        )
    }

    /// Assemble force terms while leaving mass-matrix construction to GPU.
    #[cfg(feature = "gpu-contact")]
    pub(crate) fn generalized_force_terms(
        &self,
        root_pose: Isometry3<f64>,
        angles: &[f64],
        velocity: &DVector<f64>,
        floating: bool,
        gravity: Vector3<f64>,
    ) -> Result<(DVector<f64>, DVector<f64>), ArticulationError> {
        let terms = self.generalized_dynamics_impl(
            root_pose,
            angles,
            velocity,
            floating,
            gravity,
            DynamicsEvaluation::default(),
        )?;
        Ok((terms.gravity_force, terms.velocity_bias))
    }

    fn generalized_dynamics_impl(
        &self,
        root_pose: Isometry3<f64>,
        angles: &[f64],
        velocity: &DVector<f64>,
        floating: bool,
        gravity: Vector3<f64>,
        evaluation: DynamicsEvaluation<'_>,
    ) -> Result<GeneralizedDynamics, ArticulationError> {
        let offset = if floating { 6 } else { 0 };
        if velocity.len() != offset + self.dof
            || velocity.iter().any(|value| !value.is_finite())
            || gravity.iter().any(|value| !value.is_finite())
        {
            return Err(ArticulationError::InvalidInput);
        }
        let assemble_mass = evaluation.assemble_mass;
        let pose = match evaluation.spherical {
            Some(orientations) => {
                self.pose_with_spherical_orientations(root_pose, angles, orientations)?
            }
            None => self.pose(root_pose, angles)?,
        };
        let moving = velocity.norm_squared() > 0.0;
        let epsilon = 1e-5;
        let shifted = |sign: f64| {
            let mut shifted_root = root_pose;
            if floating {
                shifted_root.translation.vector +=
                    Vector3::new(velocity[0], velocity[1], velocity[2]) * sign * epsilon;
                shifted_root.rotation = UnitQuaternion::from_scaled_axis(
                    Vector3::new(velocity[3], velocity[4], velocity[5]) * sign * epsilon,
                ) * root_pose.rotation;
            }
            let shifted_angles = (0..self.dof)
                .map(|slot| angles[slot] + velocity[offset + slot] * sign * epsilon)
                .collect::<Vec<_>>();
            if let Some(orientations) = evaluation.spherical {
                let mut shifted_orientations = orientations.to_vec();
                for (edge, rotation) in shifted_orientations.iter_mut().enumerate() {
                    if let Some(rotation) = rotation {
                        let slot =
                            self.joint_dofs[edge].ok_or(ArticulationError::InvalidInput)? + offset;
                        let angular =
                            Vector3::new(velocity[slot], velocity[slot + 1], velocity[slot + 2]);
                        *rotation =
                            UnitQuaternion::from_scaled_axis(angular * sign * epsilon) * *rotation;
                    }
                }
                self.pose_with_spherical_orientations(
                    shifted_root,
                    &shifted_angles,
                    &shifted_orientations,
                )
            } else {
                self.pose(shifted_root, &shifted_angles)
            }
        };
        let plus = if moving { Some(shifted(1.0)?) } else { None };
        let minus = if moving { Some(shifted(-1.0)?) } else { None };
        let mut mass = if assemble_mass {
            DMatrix::zeros(velocity.len(), velocity.len())
        } else {
            DMatrix::zeros(0, 0)
        };
        let mut gravity_force = DVector::zeros(velocity.len());
        let mut velocity_bias = DVector::zeros(velocity.len());
        for (link_index, link) in self.links.iter().enumerate() {
            if link.mass == 0.0 {
                continue;
            }
            let (linear, angular) =
                self.generalized_point_jacobians(&pose, link_index, link.center_of_mass, floating)?;
            let rotation = pose.links[link_index].rotation.to_rotation_matrix();
            let inertia = rotation.matrix() * link.inertia * rotation.matrix().transpose();
            if assemble_mass {
                mass += link.mass * linear.transpose() * &linear
                    + angular.transpose() * inertia * &angular;
            }
            gravity_force += link.mass * linear.transpose() * gravity;
            if let (Some(plus), Some(minus)) = (&plus, &minus) {
                let (linear_plus, angular_plus) = self.generalized_point_jacobians(
                    plus,
                    link_index,
                    link.center_of_mass,
                    floating,
                )?;
                let (linear_minus, angular_minus) = self.generalized_point_jacobians(
                    minus,
                    link_index,
                    link.center_of_mass,
                    floating,
                )?;
                let drift_linear = (linear_plus - linear_minus) * velocity * (0.5 / epsilon);
                let drift_angular = (angular_plus - angular_minus) * velocity * (0.5 / epsilon);
                let omega = &angular * velocity;
                let omega = Vector3::new(omega[0], omega[1], omega[2]);
                let gyroscopic = omega.cross(&(inertia * omega));
                velocity_bias += link.mass * linear.transpose() * drift_linear
                    + angular.transpose() * (inertia * drift_angular + gyroscopic);
            }
        }
        if assemble_mass {
            self.add_joint_armature(&mut mass, offset)?;
        }
        Ok(GeneralizedDynamics {
            mass,
            gravity_force,
            velocity_bias,
        })
    }

    /// Reflected rotor inertia in generalized-coordinate order.
    #[cfg(feature = "gpu-contact")]
    pub(crate) fn generalized_armature_diagonal(
        &self,
        floating: bool,
    ) -> Result<DVector<f64>, ArticulationError> {
        let offset = if floating { 6 } else { 0 };
        let mut diagonal = DVector::<f64>::zeros(offset + self.dof);
        for (edge, armatures) in self.joint_armatures.iter().enumerate() {
            let Some(start) = self.joint_dofs[edge] else {
                continue;
            };
            for (axis, armature) in armatures[..joint_width(self.joints[edge].kind)]
                .iter()
                .enumerate()
            {
                diagonal[offset + start + axis] +=
                    armature_contribution(*armature, self.joint_scales[edge])
                        .ok_or(ArticulationError::InvalidInput)?;
            }
        }
        if diagonal.iter().any(|value| !value.is_finite()) {
            return Err(ArticulationError::InvalidInput);
        }
        Ok(diagonal)
    }

    /// Assemble joint-space inertia and gravity forces for this pose.
    pub fn dynamics(
        &self,
        pose: &ArticulationPose,
        gravity: Vector3<f64>,
    ) -> Result<ArticulationDynamics, ArticulationError> {
        if gravity.iter().any(|value| !value.is_finite()) {
            return Err(ArticulationError::InvalidInput);
        }
        let mut mass = DMatrix::zeros(self.dof, self.dof);
        let mut gravity_force = DVector::zeros(self.dof);
        for (index, link) in self.links.iter().enumerate() {
            if link.mass == 0.0 {
                continue;
            }
            let (linear, angular) = self.point_jacobians(pose, index, link.center_of_mass)?;
            let rotation = pose.links[index].rotation.to_rotation_matrix();
            let inertia_world = rotation.matrix() * link.inertia * rotation.matrix().transpose();
            mass += link.mass * linear.transpose() * &linear
                + angular.transpose() * inertia_world * &angular;
            gravity_force += link.mass * linear.transpose() * gravity;
        }
        self.add_joint_armature(&mut mass, 0)?;
        Ok(ArticulationDynamics {
            mass,
            gravity_force,
        })
    }

    /// Evaluate Coriolis and centrifugal force with central mass derivatives.
    /// This is a deliberately slow CPU reference for later GPU comparison.
    pub fn velocity_bias(
        &self,
        root_pose: Isometry3<f64>,
        angles: &[f64],
        velocity: &DVector<f64>,
    ) -> Result<DVector<f64>, ArticulationError> {
        if angles.len() != self.dof
            || velocity.len() != self.dof
            || velocity.iter().any(|value| !value.is_finite())
        {
            return Err(ArticulationError::InvalidInput);
        }
        let step = 1e-5;
        let mut derivatives = Vec::with_capacity(self.dof);
        for slot in 0..self.dof {
            let mut shifted = angles.to_vec();
            shifted[slot] += step;
            let plus = self
                .dynamics(&self.pose(root_pose, &shifted)?, Vector3::zeros())?
                .mass;
            shifted[slot] -= 2.0 * step;
            let minus = self
                .dynamics(&self.pose(root_pose, &shifted)?, Vector3::zeros())?
                .mass;
            derivatives.push((plus - minus) * (0.5 / step));
        }
        let mut bias = DVector::zeros(self.dof);
        for i in 0..self.dof {
            for j in 0..self.dof {
                for k in 0..self.dof {
                    let christoffel = 0.5
                        * (derivatives[k][(i, j)] + derivatives[j][(i, k)]
                            - derivatives[i][(j, k)]);
                    bias[i] += christoffel * velocity[j] * velocity[k];
                }
            }
        }
        Ok(bias)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn link(mass: f64, com: [f64; 3]) -> LinkSpec {
        LinkSpec {
            mass,
            center_of_mass: Vector3::from(com),
            inertia: Matrix3::from_diagonal_element(0.1),
        }
    }

    #[test]
    fn sleep_mobility_partition_preserves_required_groups_before_upload() {
        let required = vec![vec![0, 2, 4], vec![1, 3], vec![5]];
        assert!(validate_mobility_partition(7, &required, &required).is_ok());
        assert!(validate_mobility_partition(7, &required, &[vec![4, 2, 0, 3, 1]]).is_ok());
        assert!(validate_mobility_partition(7, &required, &[vec![0, 2, 4], vec![1, 3]]).is_ok());
        for invalid in [
            vec![vec![0, 2], vec![4], vec![1, 3]],
            vec![vec![0, 2, 4], vec![1], vec![3]],
            vec![vec![0, 2, 4], vec![1, 3], vec![4]],
            vec![vec![0, 2, 4], vec![1, 3], vec![7]],
            vec![vec![0, 2, 4], vec![1, 3], vec![]],
            vec![vec![0, 2, 4], vec![1, 3, 3]],
            vec![],
        ] {
            assert!(validate_mobility_partition(7, &required, &invalid).is_err());
        }
        assert!(validate_mobility_partition(7, &[vec![7]], &[]).is_err());
        assert!(validate_mobility_partition(7, &[vec![]], &[]).is_err());
    }

    #[test]
    fn coordinate_owners_follow_descendants_mimics_and_nonleading_roots() {
        let joints = [
            (3, 4, JointKind::Revolute),
            (4, 1, JointKind::Fixed),
            (3, 0, JointKind::Spherical),
            (0, 2, JointKind::Revolute),
        ]
        .map(|(parent, child, kind)| JointSpec {
            parent,
            child,
            kind,
            origin: Isometry3::identity(),
            axis: Vector3::z(),
            limits: None,
        })
        .to_vec();
        let mut art = Articulation::new(vec![link(1.0, [0.0; 3]); 5], joints, 3).unwrap();
        assert_eq!(
            art.coordinate_affected_links(false),
            vec![vec![1, 4], vec![0, 2], vec![0, 2], vec![0, 2], vec![2]]
        );
        assert_eq!(
            art.coordinate_mobility_groups(false, &[]).unwrap(),
            vec![vec![0, 2], vec![1, 4]]
        );
        assert_eq!(
            art.coordinate_mobility_groups(false, &[(0, 1)]).unwrap(),
            vec![vec![0, 1, 2, 4]]
        );
        assert_eq!(
            art.coordinate_mobility_groups(true, &[]).unwrap(),
            vec![vec![0, 1, 2, 3, 4]]
        );
        assert!(art.coordinate_mobility_groups(false, &[(0, 5)]).is_err());
        let floating = art.coordinate_affected_links(true);
        assert_eq!(&floating[..6], vec![vec![0, 1, 2, 3, 4]; 6]);
        assert_eq!(&floating[6..], art.coordinate_affected_links(false));
        art.set_mimics(&[(3, 0, -2.0, 0.3)]).unwrap();
        assert_eq!(
            art.coordinate_affected_links(false),
            vec![vec![1, 2, 4], vec![0, 2], vec![0, 2], vec![0, 2]]
        );
        assert_eq!(
            art.coordinate_mobility_groups(false, &[]).unwrap(),
            vec![vec![0, 1, 2, 4]]
        );
        art.set_mimics(&[(3, 0, 0.0, 0.3)]).unwrap();
        assert_eq!(
            art.coordinate_mobility_groups(false, &[]).unwrap(),
            vec![vec![0, 2], vec![1, 4]]
        );
        assert_eq!(art.coordinate_affected_links(false)[0], vec![1, 4]);
        // Composition appends floating robot/free-body coordinates after the
        // original robot prefix. Every owner remains in its movable branch.
        for floating in [false, true] {
            let scene = art
                .compose_scene(floating, Isometry3::identity(), &[link(2.0, [0.0; 3])])
                .unwrap();
            let owners = scene.articulation.coordinate_affected_links(false);
            let groups = scene.contact_mobility_groups().unwrap();
            let sleep = crate::sleep::SleepSettings::default();
            let settings = scene.contact_sleep_settings(None, &[Some(sleep)]).unwrap();
            for &link in &groups[0] {
                assert_eq!(settings[link], None);
            }
            for &link in &groups[1] {
                assert_eq!(settings[link], Some(sleep));
            }
            assert_eq!(settings[scene.articulation.root], None);
            let mut frozen = vec![false; scene.articulation.dof()];
            let body = &scene.bodies[0];
            for slot in body.translation.clone().chain(body.angular.clone()) {
                frozen[slot] = true;
            }
            assert_eq!(scene.sleeping_scene_bodies(&frozen).unwrap(), [true]);
            frozen[body.angular.start] = false;
            assert_eq!(scene.sleeping_scene_bodies(&frozen).unwrap(), [false]);
            assert!(scene.sleeping_scene_bodies(&[]).is_err());
            assert!(scene.contact_sleep_settings(None, &[]).is_err());
            assert!(
                scene
                    .contact_sleep_settings(
                        None,
                        &[Some(crate::sleep::SleepSettings {
                            time_threshold: f64::NAN,
                            ..sleep
                        })]
                    )
                    .is_err()
            );
            let generated = scene
                .articulation
                .coordinate_mobility_groups(false, &[])
                .unwrap();
            let mut expected = groups.clone();
            for group in &mut expected {
                group.sort_unstable();
            }
            expected.sort_unstable();
            assert_eq!(generated, expected);
            for owner in &owners {
                assert!(!owner.is_empty());
                assert!(
                    groups
                        .iter()
                        .any(|group| owner.iter().all(|link| group.contains(link)))
                );
                assert!(!owner.contains(&scene.articulation.root));
            }
            for slot in scene.bodies[0]
                .translation
                .clone()
                .chain(scene.bodies[0].angular.clone())
            {
                assert!(owners[slot].contains(&scene.bodies[0].link));
                assert!(owners[slot].iter().all(|link| groups[1].contains(link)));
            }
        }
    }

    #[test]
    fn scene_contact_mobility_keeps_robot_helpers_and_free_branches_separate() {
        let robot = Articulation::new(
            vec![link(1.0, [0.0; 3]); 3],
            [0, 2]
                .map(|child| JointSpec {
                    parent: 1,
                    child,
                    kind: JointKind::Revolute,
                    origin: Isometry3::identity(),
                    axis: Vector3::z(),
                    limits: None,
                })
                .to_vec(),
            1,
        )
        .unwrap();
        for floating in [false, true] {
            let scene = robot
                .compose_scene(
                    floating,
                    Isometry3::identity(),
                    &[link(2.0, [0.0; 3]), link(2.0, [0.0; 3])],
                )
                .unwrap();
            let groups = scene.contact_mobility_groups().unwrap();
            assert_eq!(groups.len(), 3);
            assert_eq!(&groups[0][..3], &[0, 1, 2]);
            assert_eq!(groups[0].len(), if floating { 6 } else { 3 });
            for (group, body) in groups[1..].iter().zip(&scene.bodies) {
                assert_eq!(group.len(), 4);
                assert_eq!(group[0], body.link);
            }
            let all = groups.iter().flatten().copied().collect::<BTreeSet<_>>();
            assert_eq!(all.len(), scene.articulation.link_count() - 1);
            assert!(!all.contains(&scene.articulation.root));
            let mut invalid = scene.clone();
            invalid.bodies[0] = invalid.bodies[1].clone();
            assert!(invalid.contact_mobility_groups().is_err());
        }
    }

    #[test]
    fn scene_state_packing_validates_layout_and_world_frame_velocities() {
        let body = LinkSpec {
            mass: 1.0,
            center_of_mass: Vector3::zeros(),
            inertia: Matrix3::identity(),
        };
        let robot = Articulation::new(vec![body.clone()], vec![], 0).unwrap();
        let scene = robot
            .compose_scene(true, Isometry3::identity(), &[body])
            .unwrap();
        let state = SceneFreeBodyState {
            pose: Isometry3::translation(1.0, 2.0, 3.0)
                * Isometry3::rotation(Vector3::new(0.2, -0.3, 0.4)),
            linear_velocity: Vector3::new(0.4, -0.5, 0.6),
            angular_velocity: Vector3::new(-0.7, 0.8, -0.9),
        };
        let packed = scene
            .pack_state(&[], &[], &[], Some(&state), &[state])
            .unwrap();
        for layout in scene.robot_root.iter().chain(&scene.bodies) {
            assert_eq!(
                &packed.positions.as_slice()[layout.translation.clone()],
                &[1.0, 2.0, 3.0]
            );
            assert_eq!(
                &packed.velocities.as_slice()[layout.translation.clone()],
                state.linear_velocity.as_slice()
            );
            assert_eq!(
                &packed.velocities.as_slice()[layout.angular.clone()],
                state.angular_velocity.as_slice()
            );
            assert!(
                (packed.orientations[layout.spherical_edge]
                    .unwrap()
                    .inverse()
                    * state.pose.rotation)
                    .angle()
                    < 1e-12
            );
        }
        let unpacked = scene.unpack_state(&packed).unwrap();
        for actual in unpacked.robot_root.iter().chain(&unpacked.bodies) {
            assert!(
                (actual.pose.translation.vector - state.pose.translation.vector).norm() < 1e-12
            );
            assert!((actual.pose.rotation.inverse() * state.pose.rotation).angle() < 1e-12);
            assert_eq!(actual.linear_velocity, state.linear_velocity);
            assert_eq!(actual.angular_velocity, state.angular_velocity);
        }
        let mut invalid_state = packed.clone();
        invalid_state.orientations.fill(None);
        assert!(scene.unpack_state(&invalid_state).is_err());
        invalid_state = packed.clone();
        invalid_state.velocities[0] = f64::INFINITY;
        assert!(scene.unpack_state(&invalid_state).is_err());
        assert!(scene.pack_state(&[], &[], &[], None, &[state]).is_err());
        assert!(scene.pack_state(&[], &[], &[], Some(&state), &[]).is_err());
        let mut invalid = state;
        invalid.angular_velocity.x = f64::NAN;
        assert!(
            scene
                .pack_state(&[], &[], &[], Some(&state), &[invalid])
                .is_err()
        );
        let mut invalid_layout = scene.clone();
        invalid_layout.bodies[0].translation = 999..1002;
        assert!(
            invalid_layout
                .pack_state(&[], &[], &[], Some(&state), &[state])
                .is_err()
        );
        invalid_layout = scene.clone();
        invalid_layout.bodies[0] = invalid_layout.robot_root.clone().unwrap();
        assert!(
            invalid_layout
                .pack_state(&[], &[], &[], Some(&state), &[state])
                .is_err()
        );
    }

    #[test]
    fn scene_composition_preserves_robot_dynamics_and_free_body_mass_blocks() {
        let body = |mass| LinkSpec {
            mass,
            center_of_mass: Vector3::new(0.2, -0.1, 0.3),
            inertia: Matrix3::from_diagonal(&Vector3::new(0.4, 0.6, 0.8)),
        };
        let mut robot = Articulation::new(
            vec![body(1.0), body(2.0), body(1.5)],
            vec![
                JointSpec {
                    parent: 1,
                    child: 0,
                    kind: JointKind::Revolute,
                    origin: Isometry3::translation(0.3, 0.1, 0.0),
                    axis: Vector3::z(),
                    limits: Some((-1.0, 1.0)),
                },
                JointSpec {
                    parent: 0,
                    child: 2,
                    kind: JointKind::Revolute,
                    origin: Isometry3::translation(0.4, 0.0, 0.2),
                    axis: Vector3::y(),
                    limits: None,
                },
            ],
            1,
        )
        .unwrap();
        robot.set_mimics(&[(1, 0, -2.0, 0.1)]).unwrap();
        robot.set_joint_armature(0, &[0.2]).unwrap();
        robot.set_joint_armature(1, &[0.3]).unwrap();
        robot.exclude_collision_pair(1, 2).unwrap();
        let robot_pose = Isometry3::translation(0.5, -0.3, 1.2)
            * Isometry3::rotation(Vector3::new(0.2, -0.3, 0.4));
        let scene_bodies = [body(3.0), body(4.0)];
        for floating in [false, true] {
            let scene = robot
                .compose_scene(floating, robot_pose, &scene_bodies)
                .unwrap();
            assert_eq!(scene.robot_coordinates, 0..1);
            assert_eq!(scene.articulation.joint_limit(0), robot.joint_limit(0));
            assert!(scene.articulation.adjacent(1, 2));
            let mut q = DVector::zeros(scene.articulation.dof());
            q[0] = 0.3;
            let mut orientations = vec![None; scene.articulation.joints.len()];
            if let Some(root) = &scene.robot_root {
                q.as_mut_slice()[root.translation.clone()]
                    .copy_from_slice(robot_pose.translation.vector.as_slice());
                orientations[root.spherical_edge] = Some(robot_pose.rotation);
            }
            let poses = [
                Isometry3::translation(-1.0, 0.2, 0.4)
                    * Isometry3::rotation(Vector3::new(0.5, -0.4, 0.3)),
                Isometry3::translation(2.0, -0.3, 0.7)
                    * Isometry3::rotation(Vector3::new(-0.2, 0.8, 0.1)),
            ];
            for (layout, pose) in scene.bodies.iter().zip(poses) {
                q.as_mut_slice()[layout.translation.clone()]
                    .copy_from_slice(pose.translation.vector.as_slice());
                q.as_mut_slice()[layout.angular.clone()].fill(123.0);
                orientations[layout.spherical_edge] = Some(pose.rotation);
            }
            let pose = scene
                .articulation
                .pose_with_spherical_orientations(
                    Isometry3::identity(),
                    q.as_slice(),
                    &orientations,
                )
                .unwrap();
            let original_pose = robot.pose(robot_pose, &[0.3]).unwrap();
            for i in 0..robot.link_count() {
                assert!(
                    (pose.links[i].translation.vector - original_pose.links[i].translation.vector)
                        .norm()
                        < 1e-12
                );
                assert!(
                    (pose.links[i].rotation.inverse() * original_pose.links[i].rotation).angle()
                        < 1e-12
                );
            }
            let gravity = Vector3::new(0.1, -0.2, -9.81);
            let dynamics = scene
                .articulation
                .generalized_dynamics_with_spherical_orientations(
                    Isometry3::identity(),
                    q.as_slice(),
                    &orientations,
                    &DVector::zeros(q.len()),
                    false,
                    gravity,
                )
                .unwrap();
            let original = robot
                .generalized_dynamics(
                    robot_pose,
                    &[0.3],
                    &DVector::zeros(if floating { 7 } else { 1 }),
                    floating,
                    gravity,
                )
                .unwrap();
            let mapping = if let Some(root) = &scene.robot_root {
                root.translation
                    .clone()
                    .chain(root.angular.clone())
                    .chain(0..1)
                    .collect::<Vec<_>>()
            } else {
                vec![0]
            };
            for (i, &a) in mapping.iter().enumerate() {
                assert!((dynamics.gravity_force[a] - original.gravity_force[i]).abs() < 1e-10);
                for (j, &b) in mapping.iter().enumerate() {
                    assert!((dynamics.mass[(a, b)] - original.mass[(i, j)]).abs() < 1e-10);
                }
            }
            for (index, layout) in scene.bodies.iter().enumerate() {
                let properties = &scene_bodies[index];
                assert!(
                    (pose.links[layout.link].translation.vector - poses[index].translation.vector)
                        .norm()
                        < 1e-12
                );
                let radius = poses[index].rotation * properties.center_of_mass;
                let skew = Matrix3::new(
                    0.0, -radius.z, radius.y, radius.z, 0.0, -radius.x, -radius.y, radius.x, 0.0,
                );
                let rotation = poses[index].rotation.to_rotation_matrix();
                let angular_mass =
                    rotation.matrix() * properties.inertia * rotation.matrix().transpose()
                        + properties.mass * skew.transpose() * skew;
                let slots = layout
                    .translation
                    .clone()
                    .chain(layout.angular.clone())
                    .collect::<Vec<_>>();
                for i in 0..3 {
                    assert!(
                        (dynamics.gravity_force[slots[i]] - properties.mass * gravity[i]).abs()
                            < 1e-10
                    );
                    for j in 0..3 {
                        assert!(
                            (dynamics.mass[(slots[i], slots[j])]
                                - if i == j { properties.mass } else { 0.0 })
                            .abs()
                                < 1e-10
                        );
                        assert!(
                            (dynamics.mass[(slots[i], slots[j + 3])]
                                + properties.mass * skew[(i, j)])
                                .abs()
                                < 1e-10
                        );
                        assert!(
                            (dynamics.mass[(slots[i + 3], slots[j + 3])] - angular_mass[(i, j)])
                                .abs()
                                < 1e-10
                        );
                    }
                }
                for &robot_slot in &mapping {
                    for &body_slot in &slots {
                        assert!(dynamics.mass[(robot_slot, body_slot)].abs() < 1e-12);
                    }
                }
            }
            // A shared contact witness must exchange equal and opposite spatial impulse.
            let witness = Point3::new(-0.2, 0.1, 0.8);
            let normal = Vector3::new(0.6, -0.8, 0.0);
            let mut row = DVector::zeros(q.len());
            for (sign, layout) in [(1.0, &scene.bodies[0]), (-1.0, &scene.bodies[1])] {
                let local = pose.links[layout.link]
                    .inverse_transform_point(&witness)
                    .coords;
                let (linear, _) = scene
                    .articulation
                    .generalized_point_jacobians(&pose, layout.link, local, false)
                    .unwrap();
                row += linear.transpose() * normal * sign;
            }
            let delta = dynamics.mass.clone().lu().solve(&row).unwrap();
            for &slot in &mapping {
                assert!(delta[slot].abs() < 1e-12);
            }
            let mut linear_impulse = Vector3::zeros();
            let mut angular_impulse = Vector3::zeros();
            for (index, layout) in scene.bodies.iter().enumerate() {
                let properties = &scene_bodies[index];
                let (linear, angular) = scene
                    .articulation
                    .generalized_point_jacobians(
                        &pose,
                        layout.link,
                        properties.center_of_mass,
                        false,
                    )
                    .unwrap();
                let momentum =
                    Vector3::from_column_slice((linear * &delta).as_slice()) * properties.mass;
                let omega = Vector3::from_column_slice((angular * &delta).as_slice());
                let center = pose.links[layout.link]
                    .transform_point(&Point3::from(properties.center_of_mass))
                    .coords;
                let rotation = pose.links[layout.link].rotation.to_rotation_matrix();
                linear_impulse += momentum;
                angular_impulse += center.cross(&momentum)
                    + rotation.matrix()
                        * properties.inertia
                        * rotation.matrix().transpose()
                        * omega;
            }
            assert!(linear_impulse.norm() < 1e-10);
            assert!(angular_impulse.norm() < 1e-10);
            assert!(dynamics.mass.symmetric_eigen().eigenvalues.min() > 0.01);
        }
    }

    #[test]
    fn quaternion_spherical_pose_has_full_rank_tangent_mass_at_gimbal_lock() {
        let origin = Isometry3::translation(0.4, -0.2, 0.1)
            * Isometry3::rotation(Vector3::new(0.2, 0.1, -0.3));
        let tree = Articulation::new(
            vec![link(1.0, [0.0; 3]), link(1.0, [0.3, -0.2, 0.1])],
            vec![JointSpec {
                parent: 0,
                child: 1,
                origin,
                kind: JointKind::Spherical,
                axis: Vector3::zeros(),
                limits: None,
            }],
            0,
        )
        .unwrap();
        let root = Isometry3::translation(1.0, 2.0, 3.0)
            * Isometry3::rotation(Vector3::new(-0.4, 0.5, 0.2));
        let euler = [0.2, core::f64::consts::FRAC_PI_2, -0.3];
        let rotation = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), euler[0])
            * UnitQuaternion::from_axis_angle(&Vector3::y_axis(), euler[1])
            * UnitQuaternion::from_axis_angle(&Vector3::z_axis(), euler[2]);
        let legacy = tree.pose(root, &euler).unwrap();
        let legacy_mass = tree.dynamics(&legacy, Vector3::zeros()).unwrap().mass;
        assert!(legacy_mass.symmetric_eigen().eigenvalues.min().abs() < 1e-12);
        let pose = tree
            .pose_with_spherical_orientations(root, &[100.0, 200.0, 300.0], &[Some(rotation)])
            .unwrap();
        assert!(pose.links[1].rotation.angle_to(&legacy.links[1].rotation) < 1e-12);
        assert!(
            (pose.links[1].translation.vector - legacy.links[1].translation.vector).norm() < 1e-12
        );
        let gravity = Vector3::new(0.0, 0.0, -9.81);
        let dynamics = tree.dynamics(&pose, gravity).unwrap();
        assert!(dynamics.mass.clone().symmetric_eigen().eigenvalues.min() > 0.099999);
        let point = Vector3::new(0.3, -0.2, 0.1);
        let (linear, angular) = tree.point_jacobians(&pose, 1, point).unwrap();
        assert!((linear.transpose() * gravity - dynamics.gravity_force).norm() < 1e-12);
        let step = 1e-6;
        for axis in 0..3 {
            let mut shift = Vector3::zeros();
            shift[axis] = step;
            let plus = tree
                .pose_with_spherical_orientations(
                    root,
                    &[0.0; 3],
                    &[Some(UnitQuaternion::from_scaled_axis(shift) * rotation)],
                )
                .unwrap();
            let minus = tree
                .pose_with_spherical_orientations(
                    root,
                    &[0.0; 3],
                    &[Some(UnitQuaternion::from_scaled_axis(-shift) * rotation)],
                )
                .unwrap();
            let numerical_linear = (plus.links[1].transform_point(&Point3::from(point)).coords
                - minus.links[1].transform_point(&Point3::from(point)).coords)
                / (2.0 * step);
            let numerical_angular = (plus.links[1].rotation * minus.links[1].rotation.inverse())
                .scaled_axis()
                / (2.0 * step);
            assert!((linear.column(axis) - numerical_linear).norm() < 1e-9);
            assert!((angular.column(axis) - numerical_angular).norm() < 1e-9);
        }
        assert!(
            tree.pose_with_spherical_orientations(root, &[0.0; 3], &[None])
                .is_err()
        );
        assert!(
            tree.pose_with_spherical_orientations(root, &[0.0; 3], &[])
                .is_err()
        );
        let invalid = UnitQuaternion::new_unchecked(nalgebra::Quaternion::new(2.0, 0.0, 0.0, 0.0));
        assert!(
            tree.pose_with_spherical_orientations(root, &[0.0; 3], &[Some(invalid)])
                .is_err()
        );
    }

    #[test]
    fn one_revolute_joint_has_expected_pose_inertia_and_gravity() {
        let tree = Articulation::new(
            vec![link(1.0, [0.0; 3]), link(1.0, [1.0, 0.0, 0.0])],
            vec![JointSpec {
                parent: 0,
                child: 1,
                origin: Isometry3::identity(),
                kind: JointKind::Revolute,
                axis: Vector3::y(),
                limits: None,
            }],
            0,
        )
        .unwrap();
        let pose = tree.pose(Isometry3::identity(), &[0.0]).unwrap();
        let dynamics = tree.dynamics(&pose, Vector3::new(0.0, 0.0, -9.81)).unwrap();
        assert!((dynamics.mass[(0, 0)] - 1.1).abs() < 1e-10);
        assert!((dynamics.gravity_force[0] - 9.81).abs() < 1e-10);
        let bias = tree
            .velocity_bias(Isometry3::identity(), &[0.0], &DVector::from_vec(vec![2.0]))
            .unwrap();
        assert!(bias[0].abs() < 1e-9);
        let turned = tree
            .pose(Isometry3::identity(), &[core::f64::consts::FRAC_PI_2])
            .unwrap();
        let com = turned.links[1].transform_point(&Point3::new(1.0, 0.0, 0.0));
        assert!((com.z + 1.0).abs() < 1e-10);
    }

    #[test]
    fn fixed_child_transmits_ancestor_jacobian() {
        let tree = Articulation::new(
            vec![link(1.0, [0.0; 3]); 3],
            vec![
                JointSpec {
                    parent: 0,
                    child: 1,
                    origin: Isometry3::identity(),
                    kind: JointKind::Revolute,
                    axis: Vector3::y(),
                    limits: None,
                },
                JointSpec {
                    parent: 1,
                    child: 2,
                    origin: Isometry3::translation(1.0, 0.0, 0.0),
                    kind: JointKind::Fixed,
                    axis: Vector3::z(),
                    limits: None,
                },
            ],
            0,
        )
        .unwrap();
        let pose = tree.pose(Isometry3::identity(), &[0.0]).unwrap();
        let (linear, angular) = tree.point_jacobians(&pose, 2, Vector3::zeros()).unwrap();
        assert!((linear[(2, 0)] + 1.0).abs() < 1e-10);
        assert!((angular[(1, 0)] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn mimic_reduces_mass_gravity_and_jacobians_exactly() {
        let joints = vec![
            JointSpec {
                parent: 0,
                child: 1,
                origin: Isometry3::translation(-1.0, 0.0, 0.0),
                kind: JointKind::Revolute,
                axis: Vector3::z(),
                limits: None,
            },
            JointSpec {
                parent: 0,
                child: 2,
                origin: Isometry3::translation(1.0, 0.0, 0.0),
                kind: JointKind::Revolute,
                axis: Vector3::z(),
                limits: None,
            },
        ];
        let links = vec![
            link(0.0, [0.0; 3]),
            link(1.0, [1.0, 0.0, 0.0]),
            link(2.0, [1.0, 0.0, 0.0]),
        ];
        let full = Articulation::new(links.clone(), joints.clone(), 0).unwrap();
        let mut reduced = Articulation::new(links, joints, 0).unwrap();
        reduced.set_mimics(&[(1, 0, -2.0, 0.3)]).unwrap();
        assert_eq!(reduced.dof(), 1);
        assert_eq!(reduced.joint_coordinate_range(1), Some(0..1));
        let full_pose = full.pose(Isometry3::identity(), &[0.2, -0.1]).unwrap();
        let reduced_pose = reduced.pose(Isometry3::identity(), &[0.2]).unwrap();
        for link in 0..3 {
            assert!(
                (full_pose.links[link].translation.vector
                    - reduced_pose.links[link].translation.vector)
                    .norm()
                    < 1e-12
            );
            assert!(
                (full_pose.links[link].rotation.coords - reduced_pose.links[link].rotation.coords)
                    .norm()
                    < 1e-12
            );
        }
        let (full_linear, full_angular) = full
            .point_jacobians(&full_pose, 2, Vector3::new(0.5, 0.0, 0.0))
            .unwrap();
        let (reduced_linear, reduced_angular) = reduced
            .point_jacobians(&reduced_pose, 2, Vector3::new(0.5, 0.0, 0.0))
            .unwrap();
        for row in 0..3 {
            assert!(
                (reduced_linear[(row, 0)] - (full_linear[(row, 0)] - 2.0 * full_linear[(row, 1)]))
                    .abs()
                    < 1e-12
            );
            assert!(
                (reduced_angular[(row, 0)]
                    - (full_angular[(row, 0)] - 2.0 * full_angular[(row, 1)]))
                    .abs()
                    < 1e-12
            );
        }
        let gravity = Vector3::new(0.0, -9.81, 0.0);
        let full_dynamics = full.dynamics(&full_pose, gravity).unwrap();
        let reduced_dynamics = reduced.dynamics(&reduced_pose, gravity).unwrap();
        let mapping = DVector::from_vec(vec![1.0, -2.0]);
        let expected_mass = (mapping.transpose() * &full_dynamics.mass * &mapping)[(0, 0)];
        let expected_gravity = (mapping.transpose() * &full_dynamics.gravity_force)[0];
        assert!((reduced_dynamics.mass[(0, 0)] - expected_mass).abs() < 1e-10);
        assert!((reduced_dynamics.gravity_force[0] - expected_gravity).abs() < 1e-10);
    }

    #[test]
    fn armature_adds_reflected_inertia_to_fixed_and_floating_dynamics() {
        let joints = vec![
            JointSpec {
                parent: 0,
                child: 1,
                origin: Isometry3::identity(),
                kind: JointKind::Revolute,
                axis: Vector3::z(),
                limits: None,
            },
            JointSpec {
                parent: 0,
                child: 2,
                origin: Isometry3::identity(),
                kind: JointKind::Revolute,
                axis: Vector3::z(),
                limits: None,
            },
        ];
        let mut tree = Articulation::new(vec![link(1.0, [0.0; 3]); 3], joints, 0).unwrap();
        tree.set_mimics(&[(1, 0, -2.0, 0.0)]).unwrap();
        let pose = tree.pose(Isometry3::identity(), &[0.2]).unwrap();
        let base = tree.dynamics(&pose, Vector3::zeros()).unwrap().mass;
        let floating_base = tree
            .generalized_dynamics(
                Isometry3::identity(),
                &[0.2],
                &DVector::zeros(7),
                true,
                Vector3::zeros(),
            )
            .unwrap()
            .mass;
        assert!(tree.set_joint_armature(0, &[-1.0]).is_err());
        assert!(tree.set_joint_armature(0, &[f64::INFINITY]).is_err());
        assert!(tree.set_joint_armature(1, &[]).is_err());
        assert_eq!(tree.joint_armature(0), Some([0.0].as_slice()));
        tree.set_joint_armature(0, &[0.5]).unwrap();
        tree.set_joint_armature(1, &[0.25]).unwrap();
        assert_eq!(tree.joint_armature(1), Some([0.25].as_slice()));
        let fixed = tree.dynamics(&pose, Vector3::zeros()).unwrap();
        assert!((fixed.mass[(0, 0)] - base[(0, 0)] - 1.5).abs() < 1e-12);
        let floating = tree
            .generalized_dynamics(
                Isometry3::identity(),
                &[0.2],
                &DVector::zeros(7),
                true,
                Vector3::zeros(),
            )
            .unwrap();
        assert!((floating.mass[(6, 6)] - fixed.mass[(0, 0)]).abs() < 1e-12);
        assert!((floating.mass[(6, 6)] - floating_base[(6, 6)] - 1.5).abs() < 1e-12);
        for row in 0..6 {
            assert!((floating.mass[(row, 6)] - floating_base[(row, 6)]).abs() < 1e-12);
        }
    }

    #[test]
    fn mimic_chains_compose_and_contradictory_limits_fail() {
        let joints = (1..4)
            .map(|child| JointSpec {
                parent: 0,
                child,
                origin: Isometry3::identity(),
                kind: JointKind::Prismatic,
                axis: Vector3::x(),
                limits: if child == 1 {
                    Some((-1.0, 1.0))
                } else if child == 2 {
                    Some((-0.5, 0.5))
                } else {
                    None
                },
            })
            .collect::<Vec<_>>();
        let mut tree = Articulation::new(vec![link(1.0, [0.0; 3]); 4], joints, 0).unwrap();
        tree.set_mimics(&[(1, 0, -2.0, 0.0), (2, 1, -1.0, 0.1)])
            .unwrap();
        assert_eq!(tree.dof(), 1);
        assert_eq!(tree.joint_limit(0), Some((-0.25, 0.25)));
        let pose = tree.pose(Isometry3::identity(), &[0.2]).unwrap();
        assert!((pose.links[2].translation.vector.x + 0.4).abs() < 1e-12);
        assert!((pose.links[3].translation.vector.x - 0.5).abs() < 1e-12);
        assert!(
            tree.set_mimics(&[(0, 1, 1.0, 0.0), (1, 0, 1.0, 0.0)])
                .is_err()
        );
        assert_eq!(tree.dof(), 1);
        assert!(tree.set_mimics(&[(1, 0, 0.0, 1.0)]).is_err());
        assert_eq!(tree.dof(), 1);
    }

    #[test]
    fn one_prismatic_joint_has_expected_pose_inertia_and_gravity() {
        let tree = Articulation::new(
            vec![link(0.0, [0.0; 3]), link(2.0, [0.0; 3])],
            vec![JointSpec {
                parent: 0,
                child: 1,
                origin: Isometry3::identity(),
                kind: JointKind::Prismatic,
                axis: Vector3::z(),
                limits: Some((-0.5, 1.0)),
            }],
            0,
        )
        .unwrap();
        let pose = tree.pose(Isometry3::identity(), &[0.4]).unwrap();
        assert!((pose.links[1].translation.vector.z - 0.4).abs() < 1e-12);
        let (linear, angular) = tree
            .point_jacobians(&pose, 1, Vector3::new(1.0, 0.0, 0.0))
            .unwrap();
        assert!((linear[(2, 0)] - 1.0).abs() < 1e-12);
        assert!(angular.column(0).norm() < 1e-12);
        let dynamics = tree.dynamics(&pose, Vector3::new(0.0, 0.0, -9.81)).unwrap();
        assert!((dynamics.mass[(0, 0)] - 2.0).abs() < 1e-12);
        assert!((dynamics.gravity_force[0] + 19.62).abs() < 1e-12);
        assert_eq!(tree.joint_limit(0), Some((-0.5, 1.0)));
    }

    #[test]
    fn spherical_joint_has_three_dofs_pose_jacobian_and_inertia() {
        let mut tree = Articulation::new(
            vec![link(0.0, [0.0; 3]), link(1.0, [1.0, 0.0, 0.0])],
            vec![JointSpec {
                parent: 0,
                child: 1,
                origin: Isometry3::identity(),
                kind: JointKind::Spherical,
                axis: Vector3::zeros(),
                limits: None,
            }],
            0,
        )
        .unwrap();
        assert_eq!(tree.dof(), 3);
        assert_eq!(tree.joint_limit(0), None);
        assert_eq!(tree.joint_limit(1), None);
        assert_eq!(tree.joint_limit(2), None);

        let pose = tree.pose(Isometry3::identity(), &[0.0; 3]).unwrap();
        let (linear, angular) = tree
            .point_jacobians(&pose, 1, Vector3::new(1.0, 0.0, 0.0))
            .unwrap();
        for row in 0..3 {
            for column in 0..3 {
                let expected = if row == column { 1.0 } else { 0.0 };
                assert!((angular[(row, column)] - expected).abs() < 1e-12);
            }
        }
        assert!(linear.column(0).norm() < 1e-12);
        let linear_y = Vector3::new(linear[(0, 1)], linear[(1, 1)], linear[(2, 1)]);
        let linear_z = Vector3::new(linear[(0, 2)], linear[(1, 2)], linear[(2, 2)]);
        assert!((linear_y + Vector3::z()).norm() < 1e-12);
        assert!((linear_z - Vector3::y()).norm() < 1e-12);
        let dynamics = tree.dynamics(&pose, Vector3::zeros()).unwrap();
        assert!((dynamics.mass[(0, 0)] - 0.1).abs() < 1e-12);
        assert!((dynamics.mass[(1, 1)] - 1.1).abs() < 1e-12);
        assert!((dynamics.mass[(2, 2)] - 1.1).abs() < 1e-12);
        tree.set_joint_armature(0, &[0.2, 0.3, 0.4]).unwrap();
        let with_armature = tree.dynamics(&pose, Vector3::zeros()).unwrap();
        for (axis, added) in [0.2, 0.3, 0.4].into_iter().enumerate() {
            assert!(
                (with_armature.mass[(axis, axis)] - dynamics.mass[(axis, axis)] - added).abs()
                    < 1e-12
            );
        }

        let turned = tree
            .pose(
                Isometry3::identity(),
                &[core::f64::consts::FRAC_PI_2, 0.0, 0.0],
            )
            .unwrap();
        assert!((turned.joint_axes[0] - Vector3::x()).norm() < 1e-12);
        assert!((turned.joint_axes[1] - Vector3::z()).norm() < 1e-12);
        assert!((turned.joint_axes[2] + Vector3::y()).norm() < 1e-12);
        let rotated_y = turned.links[1].rotation * Vector3::y();
        assert!((rotated_y - Vector3::z()).norm() < 1e-12);
    }

    #[test]
    fn spherical_joint_rejects_scalar_limits() {
        let result = Articulation::new(
            vec![link(0.0, [0.0; 3]), link(1.0, [0.0; 3])],
            vec![JointSpec {
                parent: 0,
                child: 1,
                origin: Isometry3::identity(),
                kind: JointKind::Spherical,
                axis: Vector3::zeros(),
                limits: Some((-1.0, 1.0)),
            }],
            0,
        );
        assert_eq!(result.unwrap_err(), ArticulationError::InvalidInput);
    }

    #[test]
    fn rejects_disconnected_cycle() {
        let result = Articulation::new(
            vec![link(1.0, [0.0; 3]); 3],
            vec![
                JointSpec {
                    parent: 2,
                    child: 1,
                    origin: Isometry3::identity(),
                    kind: JointKind::Fixed,
                    axis: Vector3::z(),
                    limits: None,
                },
                JointSpec {
                    parent: 1,
                    child: 2,
                    origin: Isometry3::identity(),
                    kind: JointKind::Fixed,
                    axis: Vector3::z(),
                    limits: None,
                },
            ],
            0,
        );
        assert_eq!(result.unwrap_err(), ArticulationError::InvalidTree);
    }

    #[test]
    fn floating_single_body_has_translational_and_rotational_inertia() {
        let tree = Articulation::new(
            vec![LinkSpec {
                mass: 2.0,
                center_of_mass: Vector3::zeros(),
                inertia: Matrix3::identity() * 0.4,
            }],
            vec![],
            0,
        )
        .unwrap();
        let state = tree
            .generalized_dynamics(
                Isometry3::identity(),
                &[],
                &DVector::zeros(6),
                true,
                Vector3::new(0.0, 0.0, -9.81),
            )
            .unwrap();
        for axis in 0..3 {
            assert!((state.mass[(axis, axis)] - 2.0).abs() < 1e-12);
            assert!((state.mass[(axis + 3, axis + 3)] - 0.4).abs() < 1e-12);
        }
        assert!((state.gravity_force[2] + 19.62).abs() < 1e-12);
        assert!(state.velocity_bias.norm() < 1e-12);
    }

    #[test]
    fn anisotropic_floating_body_has_gyroscopic_bias() {
        let tree = Articulation::new(
            vec![LinkSpec {
                mass: 1.0,
                center_of_mass: Vector3::zeros(),
                inertia: Matrix3::from_diagonal(&Vector3::new(1.0, 2.0, 3.0)),
            }],
            vec![],
            0,
        )
        .unwrap();
        let velocity = DVector::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0]);
        let state = tree
            .generalized_dynamics(
                Isometry3::identity(),
                &[],
                &velocity,
                true,
                Vector3::zeros(),
            )
            .unwrap();
        assert!(state.velocity_bias[0].abs() < 1e-8);
        assert!(state.velocity_bias[1].abs() < 1e-8);
        assert!(state.velocity_bias[2].abs() < 1e-8);
        assert!(state.velocity_bias[5] > 0.99 && state.velocity_bias[5] < 1.01);
    }
}
