//! Composed device-resident dynamics steps for fixed-root articulated trees.

use crate::articulated_world::{JointPolynomialCoupling, LinkFixedConstraint, LinkPointConstraint};
use nalgebra::{DMatrix, DVector, Isometry3, Vector3};

use crate::articulation::{Articulation, ArticulationError};
use crate::gpu_articulated_force::{
    GpuArticulatedForceBatch, GpuArticulatedForceError, GpuMassLinkLoad,
};
use crate::gpu_articulated_ground_contact::{
    GpuArticulatedAxialBoxPair, GpuArticulatedAxialCapsulePair, GpuArticulatedAxialConvexPair,
    GpuArticulatedAxialPair, GpuArticulatedAxialSpherePair, GpuArticulatedBoxPair,
    GpuArticulatedCapsuleBoxPair, GpuArticulatedCapsulePair, GpuArticulatedCapsuleSpherePair,
    GpuArticulatedCombineRules, GpuArticulatedContactSettings, GpuArticulatedConvexCapsulePair,
    GpuArticulatedConvexPair, GpuArticulatedConvexSpherePair, GpuArticulatedGroundAxialShape,
    GpuArticulatedGroundBox, GpuArticulatedGroundCapsule, GpuArticulatedGroundContactBatch,
    GpuArticulatedGroundContactError, GpuArticulatedGroundSphere, GpuArticulatedLinkBox,
    GpuArticulatedLinkCapsule, GpuArticulatedLinkSphere, GpuArticulatedSceneConvexCapsulePair,
    GpuArticulatedSceneConvexSpherePair, GpuArticulatedSceneMeshAxialPair,
    GpuArticulatedSceneMeshBoxPair, GpuArticulatedSceneMeshCapsulePair,
    GpuArticulatedSceneMeshConvexPair, GpuArticulatedSceneMeshSpherePair,
    GpuArticulatedScenePolylineAxialPair, GpuArticulatedScenePolylineBoxPair,
    GpuArticulatedScenePolylineCapsulePair, GpuArticulatedScenePolylineConvexPair,
    GpuArticulatedScenePolylineSpherePair, GpuArticulatedSphereBoxPair, GpuArticulatedSpherePair,
    GpuArticulatedStaticAxialBoxPair, GpuArticulatedStaticAxialCapsulePair,
    GpuArticulatedStaticAxialConvexPair, GpuArticulatedStaticAxialSpherePair,
    GpuArticulatedStaticBoxCapsulePair, GpuArticulatedStaticBoxPair,
    GpuArticulatedStaticBoxSpherePair, GpuArticulatedStaticCapsuleBoxPair,
    GpuArticulatedStaticCapsulePair, GpuArticulatedStaticCapsuleSpherePair,
    GpuArticulatedStaticConvexCapsulePair, GpuArticulatedStaticConvexPair,
    GpuArticulatedStaticConvexSpherePair, GpuArticulatedStaticSphereBoxPair,
    GpuArticulatedStaticSphereCapsulePair, GpuArticulatedStaticSpherePair, validate_link_box,
    validate_link_capsule,
};
#[cfg(test)]
use crate::gpu_articulated_ground_contact::{
    self_contact_box_pairs, self_contact_capsule_box_pairs, self_contact_capsule_pairs,
    self_contact_capsule_sphere_pairs, self_contact_sphere_box_pairs,
};
use crate::gpu_articulated_joint_force::{
    GpuArticulatedJointForceBatch, GpuArticulatedJointForceError, GpuJointForceInput,
};
use crate::gpu_articulated_joint_limit::{
    GpuArticulatedJointLimitBatch, GpuArticulatedJointLimitError,
};
use crate::gpu_articulated_link_terms::{
    GpuArticulatedLinkTermsBatch, GpuArticulatedLinkTermsError,
};
use crate::gpu_articulated_mass::GpuArticulatedMassError;
use crate::gpu_articulated_mass_assembly::{
    GpuArticulatedMassAssemblyBatch, GpuMassAssemblySystem, GpuMassLink,
};
use crate::gpu_articulated_pose::{GpuArticulatedPoseBatch, GpuArticulatedPoseError};
use crate::gpu_articulated_root::{GpuArticulatedRootBatch, GpuArticulatedRootError};
use crate::gpu_articulated_shape_bounds::{
    GpuArticulatedBoundsShape, GpuArticulatedShapeBoundsBatch, GpuArticulatedShapeBoundsError,
};
use crate::gpu_articulated_sleep_freeze::{GpuArticulatedSleepFreezeBatch, GpuSleepFreezeError};
use crate::gpu_articulated_spherical::{
    GpuArticulatedSphericalBatch, GpuArticulatedSphericalError, GpuSphericalJointState,
};
use crate::gpu_articulated_spherical_drive::{
    GpuArticulatedSphericalDriveBatch, GpuSphericalDriveError,
};
use crate::gpu_articulated_state::{
    GpuGeneralizedState, GpuGeneralizedStateBatch, GpuGeneralizedStateError,
};
use crate::gpu_articulated_velocity_bias::{
    GpuArticulatedVelocityBiasBatch, GpuArticulatedVelocityBiasError,
};
use crate::gpu_contact_pipeline::GpuContactDevice;
use crate::gpu_lbvh::GpuLbvh;
use crate::gpu_motor_target::{
    GpuMotorTargetControl, GpuMotorTargetMapping, GpuMotorTargetMode, PackedMotorTargetMapping,
};
use crate::spherical_drive::SphericalJointDrive;

#[path = "gpu_dynamics_reset.rs"]
mod resident_reset;
pub use resident_reset::GpuArticulatedDynamicsResetTemplates;

#[cfg(test)]
use crate::gpu_articulated_ground_contact::self_contact_sphere_pairs;

/// Initial model, state, and forces for one independent articulated environment.
#[derive(Debug, Clone)]
pub struct GpuArticulatedDynamicsInput<'a> {
    /// Tree topology and inertial properties; reconstruction is required after changes.
    pub articulation: &'a Articulation,
    /// Initial world pose of the root link.
    pub root_pose: Isometry3<f64>,
    /// Joint state, after six root workspace slots when using a floating layout.
    pub state: GpuGeneralizedState,
    /// World gravity acceleration.
    pub gravity: Vector3<f64>,
    /// One drive per generalized slot; floating root slots contain only raw efforts.
    pub joints: Vec<GpuJointForceInput>,
    /// Coulomb friction effort bound for each independent coordinate; empty omits updateable rows.
    pub joint_friction: Vec<f64>,
    /// Linear or quartic equality constraints solved in the contact sweeps.
    pub joint_couplings: Vec<JointPolynomialCoupling>,
    /// Point constraints between links or to the fixed world.
    pub link_point_constraints: Vec<LinkPointConstraint>,
    /// Frame constraints between links or to the fixed world.
    pub link_fixed_constraints: Vec<LinkFixedConstraint>,
    /// Optional shared absolute speed cap for all independent joint coordinates.
    pub joint_velocity_limit: Option<f64>,
    /// Optional per-coordinate caps overriding the uniform cap; floating root entries must be None.
    pub coordinate_velocity_limits: Option<Vec<Option<f64>>>,
    /// World loads in stable articulation link order, including massless links.
    pub link_loads: Vec<GpuMassLinkLoad>,
    /// Optional link spheres against static world planes.
    pub ground_spheres: Vec<GpuArticulatedGroundSphere>,
    /// First ground sphere included in per-link support-point reduction, if any.
    pub ground_manifold_start: Option<usize>,
    /// Optional link capsules against static world planes.
    pub ground_capsules: Vec<GpuArticulatedGroundCapsule>,
    /// Optional link boxes against static world planes.
    pub ground_boxes: Vec<GpuArticulatedGroundBox>,
    /// Optional link cylinders and cones against static world planes.
    pub ground_axial_shapes: Vec<GpuArticulatedGroundAxialShape>,
    /// Explicit sphere pairs on different links of the same articulation.
    pub sphere_pairs: Vec<GpuArticulatedSpherePair>,
    /// Link spheres paired with stationary world-space spheres.
    pub static_sphere_pairs: Vec<GpuArticulatedStaticSpherePair>,
    /// Optional initial prescribed body state for external sphere pairs.
    /// Some enables GPU origin, orientation, and sphere center integration.
    pub external_sphere_bodies:
        Option<crate::gpu_articulated_ground_contact::GpuArticulatedExternalSphereBodies>,
    /// Prescribed world-capsule body states in reserved pair order.
    pub external_capsule_bodies:
        Option<crate::gpu_articulated_ground_contact::GpuArticulatedExternalCapsuleBodies>,
    /// Prescribed world-cylinder/cone body states in reserved pair order.
    pub external_axial_bodies:
        Option<crate::gpu_articulated_ground_contact::GpuArticulatedExternalAxialBodies>,
    /// Initial prescribed world-convex states in reserved contact order.
    pub external_convex_bodies:
        Option<crate::gpu_articulated_ground_contact::GpuArticulatedExternalConvexBodies>,
    /// Initial external point-anchor bodies followed by fixed-anchor bodies.
    /// Constraints with a second link do not consume entries. None keeps anchors static.
    pub external_constraint_bodies:
        Option<Vec<Option<crate::gpu_kinematic_body::GpuKinematicBody>>>,
    /// Initial mesh/polyline body motion in the packed indexed contact-family order.
    /// Each pair consumes one entry; None keeps its geometry static.
    pub external_indexed_bodies: Option<Vec<Option<crate::gpu_kinematic_body::GpuKinematicBody>>>,
    /// Optional prescribed box-body states for reserved primitive/box pairs.
    pub external_box_bodies:
        Option<crate::gpu_articulated_ground_contact::GpuArticulatedExternalBoxBodies>,
    /// Moving link capsules against stationary world-space spheres.
    pub static_capsule_sphere_pairs: Vec<GpuArticulatedStaticCapsuleSpherePair>,
    /// Moving link boxes against stationary world-space spheres.
    pub static_box_sphere_pairs: Vec<GpuArticulatedStaticBoxSpherePair>,
    /// Moving link spheres against stationary world-space capsules.
    pub static_sphere_capsule_pairs: Vec<GpuArticulatedStaticSphereCapsulePair>,
    /// Moving link capsules against stationary world-space capsules.
    pub static_capsule_pairs: Vec<GpuArticulatedStaticCapsulePair>,
    /// Moving link spheres against stationary world-space boxes.
    pub static_sphere_box_pairs: Vec<GpuArticulatedStaticSphereBoxPair>,
    /// Moving link capsules against stationary world-space boxes.
    pub static_capsule_box_pairs: Vec<GpuArticulatedStaticCapsuleBoxPair>,
    /// Moving link boxes against stationary world-space boxes.
    pub static_box_pairs: Vec<GpuArticulatedStaticBoxPair>,
    /// Moving link boxes against stationary world-space capsules.
    pub static_box_capsule_pairs: Vec<GpuArticulatedStaticBoxCapsulePair>,
    /// Moving link cylinders and cones against stationary world-space spheres.
    pub static_axial_sphere_pairs: Vec<GpuArticulatedStaticAxialSpherePair>,
    /// Moving link cylinders and cones against stationary world-space capsules.
    pub static_axial_capsule_pairs: Vec<GpuArticulatedStaticAxialCapsulePair>,
    /// Moving link cylinders and cones against stationary world-space boxes.
    pub static_axial_box_pairs: Vec<GpuArticulatedStaticAxialBoxPair>,
    /// Moving link cylinders and cones against stationary world-space convex hulls.
    pub static_axial_convex_pairs: Vec<GpuArticulatedStaticAxialConvexPair>,
    /// Link cylinders and cones paired with convex hulls on other links.
    pub axial_convex_pairs: Vec<GpuArticulatedAxialConvexPair>,
    /// Explicit cylinder/cone-sphere pairs on different links.
    pub axial_sphere_pairs: Vec<GpuArticulatedAxialSpherePair>,
    /// Explicit cylinder/cone-box pairs on different links.
    pub axial_box_pairs: Vec<GpuArticulatedAxialBoxPair>,
    /// Explicit cylinder/cone-capsule pairs on different links.
    pub axial_capsule_pairs: Vec<GpuArticulatedAxialCapsulePair>,
    /// Explicit cylinder/cone pairs on different links.
    pub axial_pairs: Vec<GpuArticulatedAxialPair>,
    /// Explicit convex-hull/sphere pairs on different links.
    pub convex_sphere_pairs: Vec<GpuArticulatedConvexSpherePair>,
    /// Link convex hulls paired with stationary world-space spheres.
    pub static_convex_sphere_pairs: Vec<GpuArticulatedStaticConvexSpherePair>,
    /// Link convex hulls paired with stationary world-space capsules.
    pub static_convex_capsule_pairs: Vec<GpuArticulatedStaticConvexCapsulePair>,
    /// Explicit convex-hull/capsule pairs on different links.
    pub convex_capsule_pairs: Vec<GpuArticulatedConvexCapsulePair>,
    /// Explicit convex polyhedron pairs on distinct links.
    pub convex_pairs: Vec<GpuArticulatedConvexPair>,
    /// Link convex hulls paired with stationary world-space convex hulls.
    pub static_convex_pairs: Vec<GpuArticulatedStaticConvexPair>,
    /// Stationary world-space convex hulls paired with link spheres.
    pub scene_convex_sphere_pairs: Vec<GpuArticulatedSceneConvexSpherePair>,
    /// Stationary indexed meshes paired with link spheres.
    pub scene_mesh_sphere_pairs: Vec<GpuArticulatedSceneMeshSpherePair>,
    /// Stationary indexed polylines paired with link spheres.
    pub scene_polyline_sphere_pairs: Vec<GpuArticulatedScenePolylineSpherePair>,
    /// Stationary indexed meshes paired with link capsules.
    pub scene_mesh_capsule_pairs: Vec<GpuArticulatedSceneMeshCapsulePair>,
    /// Stationary indexed polylines paired with link capsules.
    pub scene_polyline_capsule_pairs: Vec<GpuArticulatedScenePolylineCapsulePair>,
    /// Stationary indexed meshes paired with link boxes.
    pub scene_mesh_box_pairs: Vec<GpuArticulatedSceneMeshBoxPair>,
    /// Stationary indexed polylines against moving link boxes.
    pub scene_polyline_box_pairs: Vec<GpuArticulatedScenePolylineBoxPair>,
    /// Stationary indexed meshes paired with link cylinders and cones.
    pub scene_mesh_axial_pairs: Vec<GpuArticulatedSceneMeshAxialPair>,
    /// Stationary indexed polylines against moving link cylinders and cones.
    pub scene_polyline_axial_pairs: Vec<GpuArticulatedScenePolylineAxialPair>,
    /// Stationary indexed meshes paired with link convex hulls.
    pub scene_mesh_convex_pairs: Vec<GpuArticulatedSceneMeshConvexPair>,
    /// Stationary indexed polylines against moving link convex hulls.
    pub scene_polyline_convex_pairs: Vec<GpuArticulatedScenePolylineConvexPair>,
    /// Stationary world-space convex hulls paired with link capsules.
    pub scene_convex_capsule_pairs: Vec<GpuArticulatedSceneConvexCapsulePair>,
    /// Explicit capsule-sphere pairs on different links of the same articulation.
    pub capsule_sphere_pairs: Vec<GpuArticulatedCapsuleSpherePair>,
    /// Explicit capsule pairs on different links of the same articulation.
    pub capsule_pairs: Vec<GpuArticulatedCapsulePair>,
    /// Explicit sphere-box pairs on different links of the same articulation.
    pub sphere_box_pairs: Vec<GpuArticulatedSphereBoxPair>,
    /// Explicit capsule-box pairs on different links of the same articulation.
    pub capsule_box_pairs: Vec<GpuArticulatedCapsuleBoxPair>,
    /// Explicit box pairs on different links of the same articulation.
    pub box_pairs: Vec<GpuArticulatedBoxPair>,
    /// Link spheres paired from GPU LBVH candidates at each step.
    pub link_spheres: Vec<GpuArticulatedLinkSphere>,
    /// Link capsules paired with spheres and other capsules from GPU LBVH candidates.
    pub link_capsules: Vec<GpuArticulatedLinkCapsule>,
    /// Link boxes paired with spheres, capsules, and other boxes from GPU LBVH candidates.
    pub link_boxes: Vec<GpuArticulatedLinkBox>,
    /// Rules for dynamic spheres, capsules, then boxes; empty uses legacy GPU combination.
    pub dynamic_material_rules: Vec<GpuArticulatedCombineRules>,
    /// Projected contact sweeps per step, in `1..=64`.
    pub contact_iterations: u32,
    /// Reuse damped impulses from the previous step when contact geometry is stable.
    pub contact_warm_start: bool,
}

impl<'a> GpuArticulatedDynamicsInput<'a> {
    /// Create an unforced, collision-free environment with explicit state and gravity.
    /// Add shape/contact inputs before constructing the resident dynamics batch.
    /// This initializes generalized drives for the supplied state dimension; the
    /// batch constructor validates fixed/floating and tangent coordinate layouts.
    pub fn new(
        articulation: &'a Articulation,
        root_pose: Isometry3<f64>,
        state: GpuGeneralizedState,
        gravity: Vector3<f64>,
    ) -> Self {
        let n = state.velocities.len();
        Self {
            articulation,
            root_pose,
            state,
            gravity,
            joints: vec![GpuJointForceInput::default(); n],
            link_loads: vec![GpuMassLinkLoad::default(); articulation.link_count()],
            joint_velocity_limit: None,
            coordinate_velocity_limits: None,
            ground_manifold_start: None,
            contact_iterations: 8,
            contact_warm_start: false,
            joint_friction: Vec::new(),
            joint_couplings: Vec::new(),
            link_point_constraints: Vec::new(),
            link_fixed_constraints: Vec::new(),
            ground_spheres: Vec::new(),
            ground_capsules: Vec::new(),
            ground_boxes: Vec::new(),
            ground_axial_shapes: Vec::new(),
            sphere_pairs: Vec::new(),
            static_sphere_pairs: Vec::new(),
            external_sphere_bodies: None,
            external_box_bodies: None,
            external_capsule_bodies: None,
            external_axial_bodies: None,
            external_convex_bodies: None,
            external_constraint_bodies: None,
            external_indexed_bodies: None,
            static_capsule_sphere_pairs: Vec::new(),
            static_box_sphere_pairs: Vec::new(),
            static_sphere_capsule_pairs: Vec::new(),
            static_capsule_pairs: Vec::new(),
            static_sphere_box_pairs: Vec::new(),
            static_capsule_box_pairs: Vec::new(),
            static_box_pairs: Vec::new(),
            static_box_capsule_pairs: Vec::new(),
            static_axial_sphere_pairs: Vec::new(),
            static_axial_capsule_pairs: Vec::new(),
            static_axial_box_pairs: Vec::new(),
            static_axial_convex_pairs: Vec::new(),
            axial_convex_pairs: Vec::new(),
            axial_sphere_pairs: Vec::new(),
            axial_box_pairs: Vec::new(),
            axial_capsule_pairs: Vec::new(),
            axial_pairs: Vec::new(),
            convex_sphere_pairs: Vec::new(),
            static_convex_sphere_pairs: Vec::new(),
            static_convex_capsule_pairs: Vec::new(),
            convex_capsule_pairs: Vec::new(),
            convex_pairs: Vec::new(),
            static_convex_pairs: Vec::new(),
            scene_convex_sphere_pairs: Vec::new(),
            scene_mesh_sphere_pairs: Vec::new(),
            scene_polyline_sphere_pairs: Vec::new(),
            scene_mesh_capsule_pairs: Vec::new(),
            scene_polyline_capsule_pairs: Vec::new(),
            scene_mesh_box_pairs: Vec::new(),
            scene_polyline_box_pairs: Vec::new(),
            scene_mesh_axial_pairs: Vec::new(),
            scene_polyline_axial_pairs: Vec::new(),
            scene_mesh_convex_pairs: Vec::new(),
            scene_polyline_convex_pairs: Vec::new(),
            scene_convex_capsule_pairs: Vec::new(),
            capsule_sphere_pairs: Vec::new(),
            capsule_pairs: Vec::new(),
            sphere_box_pairs: Vec::new(),
            capsule_box_pairs: Vec::new(),
            box_pairs: Vec::new(),
            link_spheres: Vec::new(),
            link_capsules: Vec::new(),
            link_boxes: Vec::new(),
            dynamic_material_rules: Vec::new(),
        }
    }
}

/// Invalid environment data or a dependent GPU operation failed.
#[derive(Debug, thiserror::Error)]
pub enum GpuArticulatedDynamicsError {
    /// Initial input dimensions do not match their articulations.
    #[error("invalid fixed-root articulated dynamics input")]
    InvalidInput,
    /// Articulation armature could not be evaluated.
    #[error(transparent)]
    Articulation(#[from] ArticulationError),
    /// Mass assembly or solve failed.
    #[error(transparent)]
    Mass(#[from] GpuArticulatedMassError),
    /// Generalized state creation or update failed.
    #[error(transparent)]
    State(#[from] GpuGeneralizedStateError),
    /// Link pose creation or readback failed.
    #[error(transparent)]
    Pose(#[from] GpuArticulatedPoseError),
    /// Link Jacobian construction failed.
    #[error(transparent)]
    LinkTerms(#[from] GpuArticulatedLinkTermsError),
    /// Gravity or external wrench input failed.
    #[error(transparent)]
    Force(#[from] GpuArticulatedForceError),
    /// Joint drive or implicit passive input failed.
    #[error(transparent)]
    JointForce(#[from] GpuArticulatedJointForceError),
    /// Hard position or velocity limit input failed.
    #[error(transparent)]
    JointLimit(#[from] GpuArticulatedJointLimitError),
    /// Velocity-bias construction failed.
    #[error(transparent)]
    VelocityBias(#[from] GpuArticulatedVelocityBiasError),
    /// Ground or paired-link contact construction failed.
    #[error(transparent)]
    GroundContact(#[from] GpuArticulatedGroundContactError),
    /// GPU articulated shape bounds or candidates failed.
    #[error(transparent)]
    ShapeBounds(#[from] GpuArticulatedShapeBoundsError),
    /// Floating root integration construction failed.
    #[error(transparent)]
    Root(#[from] GpuArticulatedRootError),
    /// Quaternion state construction or readback failed.
    #[error(transparent)]
    Spherical(#[from] GpuArticulatedSphericalError),
    /// Quaternion drive construction or update failed.
    #[error(transparent)]
    SphericalDrive(#[from] GpuSphericalDriveError),
    /// Sleep coordinate ownership or GPU capacity is invalid.
    #[error(transparent)]
    SleepFreeze(#[from] GpuSleepFreezeError),
}

/// State and contact diagnostics for one resident environment.
#[derive(Debug, Clone)]
pub struct GpuArticulatedDynamicsOutput {
    /// Generalized state, with six leading twist slots for a floating root.
    pub state: GpuGeneralizedState,
    /// Separate root pose; leading generalized positions are workspace values.
    pub root_pose: Isometry3<f64>,
    /// Contact wrenches evaluated before the last integration step.
    pub contacts: Vec<crate::gpu_articulated_ground_contact::GpuArticulatedLinkContact>,
    /// Whether the generalized state includes the six root slots.
    pub floating_root: bool,
    /// Separate quaternion poses and angular velocity slot locations, if enabled.
    /// Generalized spherical positions are workspace, not Euler coordinates.
    pub spherical_joints: Option<Vec<GpuSphericalJointState>>,
}

/// Reusable articulated dynamics on a wgpu device.
///
/// Forward kinematics, link Jacobians, drives, passive tangents, velocity bias,
/// gravity, mass solve, and semi-implicit integration stay on the GPU across
/// successive steps. Hard joint bounds and optional speed caps apply during
/// integration. Optional link spheres, capsules, boxes, cylinders, cones, and
/// convex vertices resolve contact against static planes. Supported shape pairs
/// resolve contact between links; link spheres also contact stationary world spheres.
/// Its fixed timestep is chosen at construction and used by both implicit
/// passive correction and state integration.
#[derive(Debug)]
pub struct GpuArticulatedDynamicsBatch {
    device: wgpu::Device,
    queue: wgpu::Queue,
    mass: GpuArticulatedMassAssemblyBatch,
    state: GpuGeneralizedStateBatch,
    poses: GpuArticulatedPoseBatch,
    link_terms: GpuArticulatedLinkTermsBatch,
    joint_forces: GpuArticulatedJointForceBatch,
    motor_targets: Option<GpuMotorTargetControl>,
    scalar_motor_links: Vec<Vec<(usize, usize)>>,
    accepted_joints: std::sync::Mutex<Vec<Vec<GpuJointForceInput>>>,
    joint_limits: GpuArticulatedJointLimitBatch,
    velocity_bias: GpuArticulatedVelocityBiasBatch,
    forces: GpuArticulatedForceBatch,
    ground_contact: Option<GpuArticulatedGroundContactBatch>,
    integrate_external_spheres: bool,
    shape_bounds: Option<GpuArticulatedShapeBoundsBatch>,
    lbvh: Option<GpuLbvh>,
    timestep: f64,
    dimensions: Vec<usize>,
    joint_bounds: Vec<Vec<Option<(f64, f64)>>>,
    topology_mobility: Vec<Vec<Vec<usize>>>,
    coordinate_owners: Vec<Vec<Vec<usize>>>,
    sleep_freeze: Option<GpuArticulatedSleepFreezeBatch>,
    actuation_efforts: Option<wgpu::Buffer>,
    root_integration: Option<GpuArticulatedRootBatch>,
    root_dofs: Vec<usize>,
    spherical: Option<GpuArticulatedSphericalBatch>,
    spherical_drives: Option<GpuArticulatedSphericalDriveBatch>,
    accepted_spherical_drives: std::sync::Mutex<Vec<Vec<Option<SphericalJointDrive>>>>,
}

impl GpuArticulatedDynamicsBatch {
    /// Initialize prescribed external sphere motion before the first step.
    /// Works after any fixed, floating, or spherical batch constructor.
    /// Pair ordering matches the corresponding center update methods.
    /// Enabling integration advances sphere centers on the GPU; otherwise
    /// the caller supplies updated centers separately. Invalid input returns
    /// an error without exposing a partially configured batch.
    pub fn with_external_sphere_motions(
        mut self,
        motions: &crate::gpu_articulated_ground_contact::GpuArticulatedExternalSphereMotions,
        integrate_centers: bool,
    ) -> Result<Self, GpuArticulatedDynamicsError> {
        let _ = self.update_external_sphere_motions(motions)?;
        self.set_static_sphere_integration(integrate_centers);
        Ok(self)
    }

    #[cfg(test)]
    pub(crate) fn readback_dynamic_row_materials(&self) -> Vec<[f32; 4]> {
        self.ground_contact
            .as_ref()
            .map(|contact| contact.readback_dynamic_row_materials(&self.queue))
            .unwrap_or_default()
    }

    #[cfg(test)]
    pub(crate) fn readback_ground_manifold_state(&self) -> Vec<(u32, [f32; 4])> {
        self.ground_contact
            .as_ref()
            .map(|contact| contact.readback_ground_manifold_state(&self.queue))
            .unwrap_or_default()
    }

    /// Allocate a packed batch of independent fixed-root trees.
    pub fn new(
        context: &GpuContactDevice,
        inputs: &[GpuArticulatedDynamicsInput<'_>],
        timestep: f64,
    ) -> Result<Self, GpuArticulatedDynamicsError> {
        Self::new_with_floating_roots(context, inputs, timestep, &vec![false; inputs.len()])
    }

    /// Allocate mixed fixed/floating environments with generalized coordinate inputs.
    ///
    /// Floating state and drive inputs have six leading world linear/angular
    /// slots. Root drives contain only applied effort; joint coupling indices
    /// are generalized indices after the prefix. Root friction entries are zero.
    /// Joint speed caps apply only to the trailing joint coordinates.
    pub fn new_with_floating_roots(
        context: &GpuContactDevice,
        inputs: &[GpuArticulatedDynamicsInput<'_>],
        timestep: f64,
        floating_roots: &[bool],
    ) -> Result<Self, GpuArticulatedDynamicsError> {
        Self::build(context, inputs, timestep, floating_roots, None, None)
    }

    /// Allocate tangent spherical dynamics with independent quaternion state.
    /// Spherical velocity slots are joint-frame angular velocities. Their scalar
    /// drive inputs must contain baseline effort only; use quaternion drives here.
    /// Scalar coordinate couplings on spherical slots are rejected.
    pub fn new_with_spherical_state(
        context: &GpuContactDevice,
        inputs: &[GpuArticulatedDynamicsInput<'_>],
        timestep: f64,
        floating_roots: &[bool],
        spherical: &[Vec<GpuSphericalJointState>],
        drives: &[Vec<Option<SphericalJointDrive>>],
    ) -> Result<Self, GpuArticulatedDynamicsError> {
        Self::build(
            context,
            inputs,
            timestep,
            floating_roots,
            Some(spherical),
            Some(drives),
        )
    }

    fn build(
        context: &GpuContactDevice,
        inputs: &[GpuArticulatedDynamicsInput<'_>],
        timestep: f64,
        floating_roots: &[bool],
        spherical_input: Option<&[Vec<GpuSphericalJointState>]>,
        spherical_drive_input: Option<&[Vec<Option<SphericalJointDrive>>]>,
    ) -> Result<Self, GpuArticulatedDynamicsError> {
        if let Some(items) = spherical_input {
            if items.len() != inputs.len() {
                return Err(GpuArticulatedDynamicsError::InvalidInput);
            }
            for (input, items) in inputs.iter().zip(items) {
                let slots = items.iter().map(|j| j.velocity_slot).collect::<Vec<_>>();
                if !spherical_baseline_only(&input.joints, &slots)
                    || input.joint_couplings.iter().any(|c| {
                        slots.iter().any(|&slot| {
                            (slot..slot.saturating_add(3)).contains(&c.follower)
                                || c.source.is_some_and(|source| {
                                    (slot..slot.saturating_add(3)).contains(&source)
                                })
                        })
                    })
                {
                    return Err(GpuArticulatedDynamicsError::InvalidInput);
                }
            }
        }
        if inputs.is_empty()
            || floating_roots.len() != inputs.len()
            || !timestep.is_finite()
            || timestep <= 0.0
        {
            return Err(GpuArticulatedDynamicsError::InvalidInput);
        }
        let mut systems = Vec::with_capacity(inputs.len());
        let mut states = Vec::with_capacity(inputs.len());
        let mut articulations = Vec::with_capacity(inputs.len());
        let mut roots = Vec::with_capacity(inputs.len());
        let mut gravities = Vec::with_capacity(inputs.len());
        let mut joints = Vec::with_capacity(inputs.len());
        let mut loads = Vec::with_capacity(inputs.len());
        let mut armatures = Vec::with_capacity(inputs.len());
        let mut dimensions = Vec::with_capacity(inputs.len());
        let mut speed_limits = Vec::with_capacity(inputs.len());
        let mut joint_bounds = Vec::with_capacity(inputs.len());
        let mut ground_spheres = Vec::with_capacity(inputs.len());
        let mut ground_manifold_starts = Vec::with_capacity(inputs.len());
        let mut joint_frictions = Vec::with_capacity(inputs.len());
        let mut joint_couplings = Vec::with_capacity(inputs.len());
        let mut link_point_constraints = Vec::with_capacity(inputs.len());
        let mut link_fixed_constraints = Vec::with_capacity(inputs.len());
        let mut ground_boxes = Vec::with_capacity(inputs.len());
        let mut ground_axial_shapes = Vec::with_capacity(inputs.len());
        let mut sphere_pairs = Vec::with_capacity(inputs.len());
        let mut static_sphere_pairs = Vec::with_capacity(inputs.len());
        let mut static_capsule_sphere_pairs = Vec::with_capacity(inputs.len());
        let mut static_box_sphere_pairs = Vec::with_capacity(inputs.len());
        let mut static_sphere_capsule_pairs = Vec::with_capacity(inputs.len());
        let mut static_capsule_pairs = Vec::with_capacity(inputs.len());
        let mut static_sphere_box_pairs = Vec::with_capacity(inputs.len());
        let mut static_capsule_box_pairs = Vec::with_capacity(inputs.len());
        let mut static_box_pairs = Vec::with_capacity(inputs.len());
        let mut static_box_capsule_pairs = Vec::with_capacity(inputs.len());
        let mut static_axial_sphere_pairs = Vec::with_capacity(inputs.len());
        let mut static_axial_capsule_pairs = Vec::with_capacity(inputs.len());
        let mut static_axial_box_pairs = Vec::with_capacity(inputs.len());
        let mut static_axial_convex_pairs = Vec::with_capacity(inputs.len());
        let mut axial_convex_pairs = Vec::with_capacity(inputs.len());
        let mut axial_sphere_pairs = Vec::with_capacity(inputs.len());
        let mut axial_box_pairs = Vec::with_capacity(inputs.len());
        let mut axial_capsule_pairs = Vec::with_capacity(inputs.len());
        let mut axial_pairs = Vec::with_capacity(inputs.len());
        let mut convex_sphere_pairs = Vec::with_capacity(inputs.len());
        let mut static_convex_sphere_pairs = Vec::with_capacity(inputs.len());
        let mut static_convex_capsule_pairs = Vec::with_capacity(inputs.len());
        let mut convex_capsule_pairs = Vec::with_capacity(inputs.len());
        let mut convex_pairs = Vec::with_capacity(inputs.len());
        let mut static_convex_pairs = Vec::with_capacity(inputs.len());
        let mut scene_convex_sphere_pairs = Vec::with_capacity(inputs.len());
        let mut scene_mesh_sphere_pairs = Vec::with_capacity(inputs.len());
        let mut scene_polyline_sphere_pairs = Vec::with_capacity(inputs.len());
        let mut scene_mesh_capsule_pairs = Vec::with_capacity(inputs.len());
        let mut scene_polyline_capsule_pairs = Vec::with_capacity(inputs.len());
        let mut scene_mesh_box_pairs = Vec::with_capacity(inputs.len());
        let mut scene_polyline_box_pairs = Vec::with_capacity(inputs.len());
        let mut scene_mesh_axial_pairs = Vec::with_capacity(inputs.len());
        let mut scene_polyline_axial_pairs = Vec::with_capacity(inputs.len());
        let mut scene_mesh_convex_pairs = Vec::with_capacity(inputs.len());
        let mut scene_polyline_convex_pairs = Vec::with_capacity(inputs.len());
        let mut scene_convex_capsule_pairs = Vec::with_capacity(inputs.len());
        let mut dynamic_spheres = Vec::with_capacity(inputs.len());
        let mut dynamic_capsules = Vec::with_capacity(inputs.len());
        let mut dynamic_boxes = Vec::with_capacity(inputs.len());
        let mut dynamic_material_rules = Vec::with_capacity(inputs.len());
        let mut capsule_sphere_pairs = Vec::with_capacity(inputs.len());
        let mut capsule_pairs = Vec::with_capacity(inputs.len());
        let mut sphere_box_pairs = Vec::with_capacity(inputs.len());
        let mut capsule_box_pairs = Vec::with_capacity(inputs.len());
        let mut box_pairs = Vec::with_capacity(inputs.len());
        let mut contact_iterations = Vec::with_capacity(inputs.len());
        let mut contact_warm_start = Vec::with_capacity(inputs.len());
        let root_dofs = floating_roots
            .iter()
            .map(|&flag| if flag { 6 } else { 0 })
            .collect::<Vec<_>>();
        for (input, &root_dof) in inputs.iter().zip(&root_dofs) {
            let articulation = input.articulation;
            let n = articulation
                .dof()
                .checked_add(root_dof)
                .ok_or(GpuArticulatedDynamicsError::InvalidInput)?;
            if n == 0
                || input.state.positions.len() != n
                || input.state.velocities.len() != n
                || input.joints.len() != n
                || (!input.joint_friction.is_empty() && input.joint_friction.len() != n)
                || input
                    .joint_friction
                    .iter()
                    .any(|&value| !value.is_finite() || value < 0.0)
                || input.link_loads.len() != articulation.link_count()
                || !(1..=64).contains(&input.contact_iterations)
            {
                return Err(GpuArticulatedDynamicsError::InvalidInput);
            }
            if input.joints[..root_dof].iter().any(|drive| {
                drive.motor.is_some()
                    || drive.passive != Default::default()
                    || drive.nonlinear != Default::default()
            }) || input
                .joint_friction
                .iter()
                .take(root_dof)
                .any(|&value| value != 0.0)
                || input.joint_couplings.iter().any(|c| {
                    c.follower < root_dof || c.source.is_some_and(|source| source < root_dof)
                })
            {
                return Err(GpuArticulatedDynamicsError::InvalidInput);
            }
            let armature = articulation.generalized_armature_diagonal(root_dof != 0)?;
            let bounds = core::iter::repeat_n(None, root_dof)
                .chain((0..articulation.dof()).map(|slot| articulation.joint_limit(slot)))
                .collect::<Vec<_>>();
            if !drives_within_bounds(&input.joints, &bounds) {
                return Err(GpuArticulatedDynamicsError::InvalidInput);
            }
            let mut links = Vec::with_capacity(articulation.link_count());
            for index in 0..articulation.link_count() {
                let link = articulation
                    .link(index)
                    .ok_or(GpuArticulatedDynamicsError::InvalidInput)?;
                links.push(GpuMassLink {
                    mass: link.mass,
                    inertia_world: link.inertia,
                    linear_jacobian: DMatrix::zeros(3, n),
                    angular_jacobian: DMatrix::zeros(3, n),
                });
            }
            systems.push(GpuMassAssemblySystem {
                links,
                armature: armature.clone(),
                force: DVector::zeros(n),
            });
            states.push(input.state.clone());
            articulations.push(articulation);
            roots.push(input.root_pose);
            gravities.push(input.gravity);
            joints.push(input.joints.clone());
            joint_frictions.push(input.joint_friction.clone());
            joint_couplings.push(input.joint_couplings.clone());
            link_point_constraints.push(input.link_point_constraints.clone());
            link_fixed_constraints.push(input.link_fixed_constraints.clone());
            loads.push(input.link_loads.clone());
            armatures.push(armature);
            dimensions.push(n);
            speed_limits.push(input.joint_velocity_limit);
            joint_bounds.push(bounds);
            if input
                .ground_manifold_start
                .is_some_and(|start| start > input.ground_spheres.len())
            {
                return Err(GpuArticulatedDynamicsError::InvalidInput);
            }
            let mut grounds = input.ground_spheres.clone();
            let manifold_start = input
                .ground_manifold_start
                .or_else(|| (!input.ground_capsules.is_empty()).then_some(grounds.len()));
            for capsule in &input.ground_capsules {
                grounds.extend(capsule.endpoint_contacts());
            }
            ground_spheres.push(grounds);
            ground_manifold_starts.push(manifold_start);
            ground_boxes.push(input.ground_boxes.clone());
            ground_axial_shapes.push(input.ground_axial_shapes.clone());
            sphere_pairs.push(input.sphere_pairs.clone());
            static_sphere_pairs.push(input.static_sphere_pairs.clone());
            static_capsule_sphere_pairs.push(input.static_capsule_sphere_pairs.clone());
            static_box_sphere_pairs.push(input.static_box_sphere_pairs.clone());
            static_sphere_capsule_pairs.push(input.static_sphere_capsule_pairs.clone());
            static_capsule_pairs.push(input.static_capsule_pairs.clone());
            static_sphere_box_pairs.push(input.static_sphere_box_pairs.clone());
            static_capsule_box_pairs.push(input.static_capsule_box_pairs.clone());
            static_box_pairs.push(input.static_box_pairs.clone());
            static_box_capsule_pairs.push(input.static_box_capsule_pairs.clone());
            static_axial_sphere_pairs.push(input.static_axial_sphere_pairs.clone());
            static_axial_capsule_pairs.push(input.static_axial_capsule_pairs.clone());
            static_axial_box_pairs.push(input.static_axial_box_pairs.clone());
            static_axial_convex_pairs.push(input.static_axial_convex_pairs.clone());
            axial_convex_pairs.push(input.axial_convex_pairs.clone());
            axial_sphere_pairs.push(input.axial_sphere_pairs.clone());
            axial_box_pairs.push(input.axial_box_pairs.clone());
            axial_capsule_pairs.push(input.axial_capsule_pairs.clone());
            axial_pairs.push(input.axial_pairs.clone());
            convex_sphere_pairs.push(input.convex_sphere_pairs.clone());
            static_convex_sphere_pairs.push(input.static_convex_sphere_pairs.clone());
            static_convex_capsule_pairs.push(input.static_convex_capsule_pairs.clone());
            convex_capsule_pairs.push(input.convex_capsule_pairs.clone());
            convex_pairs.push(input.convex_pairs.clone());
            static_convex_pairs.push(input.static_convex_pairs.clone());
            scene_convex_sphere_pairs.push(input.scene_convex_sphere_pairs.clone());
            scene_mesh_sphere_pairs.push(input.scene_mesh_sphere_pairs.clone());
            scene_polyline_sphere_pairs.push(input.scene_polyline_sphere_pairs.clone());
            scene_mesh_capsule_pairs.push(input.scene_mesh_capsule_pairs.clone());
            scene_polyline_capsule_pairs.push(input.scene_polyline_capsule_pairs.clone());
            scene_mesh_box_pairs.push(input.scene_mesh_box_pairs.clone());
            scene_polyline_box_pairs.push(input.scene_polyline_box_pairs.clone());
            scene_mesh_axial_pairs.push(input.scene_mesh_axial_pairs.clone());
            scene_polyline_axial_pairs.push(input.scene_polyline_axial_pairs.clone());
            scene_mesh_convex_pairs.push(input.scene_mesh_convex_pairs.clone());
            scene_polyline_convex_pairs.push(input.scene_polyline_convex_pairs.clone());
            scene_convex_capsule_pairs.push(input.scene_convex_capsule_pairs.clone());
            dynamic_spheres.push(input.link_spheres.clone());
            for capsule in &input.link_capsules {
                validate_link_capsule(articulation, capsule)?;
            }
            dynamic_capsules.push(input.link_capsules.clone());
            for box_shape in &input.link_boxes {
                validate_link_box(articulation, box_shape)?;
            }
            dynamic_boxes.push(input.link_boxes.clone());
            dynamic_material_rules.push(input.dynamic_material_rules.clone());
            capsule_sphere_pairs.push(input.capsule_sphere_pairs.clone());
            capsule_pairs.push(input.capsule_pairs.clone());
            sphere_box_pairs.push(input.sphere_box_pairs.clone());
            capsule_box_pairs.push(input.capsule_box_pairs.clone());
            box_pairs.push(input.box_pairs.clone());
            contact_iterations.push(input.contact_iterations);
            contact_warm_start.push(input.contact_warm_start);
        }
        let mass =
            GpuArticulatedMassAssemblyBatch::new(context.device(), context.queue(), &systems)?;
        let mut state = GpuGeneralizedStateBatch::from_assembly_batch(&mass, &states)?;
        let spherical = spherical_input
            .map(|items| GpuArticulatedSphericalBatch::new(&state, items, timestep))
            .transpose()?;
        if let Some(spherical) = &spherical {
            let mut masks = dimensions
                .iter()
                .map(|&n| vec![true; n])
                .collect::<Vec<_>>();
            for ((mask, slots), &root_dof) in masks
                .iter_mut()
                .zip(spherical.velocity_slots())
                .zip(&root_dofs)
            {
                mask[..root_dof].fill(false);
                for &slot in slots {
                    mask[slot..slot + 3].fill(false);
                }
            }
            state.set_position_integration(&masks)?;
        }
        let poses = if let Some(spherical) = &spherical {
            GpuArticulatedPoseBatch::new_with_spherical_state(
                &state,
                &articulations,
                &roots,
                floating_roots,
                spherical,
            )?
        } else {
            GpuArticulatedPoseBatch::new_with_floating_roots(
                &state,
                &articulations,
                &roots,
                floating_roots,
            )?
        };
        let link_terms = GpuArticulatedLinkTermsBatch::new(&poses, &mass, &articulations)?;
        let has_dynamic_pairs = dynamic_spheres
            .iter()
            .zip(&dynamic_capsules)
            .zip(&dynamic_boxes)
            .any(|((spheres, capsules), boxes)| {
                spheres.len() >= 2
                    || (!spheres.is_empty() && !capsules.is_empty())
                    || capsules.len() >= 2
                    || (!spheres.is_empty() && !boxes.is_empty())
                    || (!capsules.is_empty() && !boxes.is_empty())
                    || boxes.len() >= 2
            });
        let shape_bounds = if has_dynamic_pairs {
            let bounds_shapes = dynamic_spheres
                .iter()
                .zip(&dynamic_capsules)
                .zip(&dynamic_boxes)
                .map(|((spheres, capsules), boxes)| {
                    let mut shapes = spheres
                        .iter()
                        .map(|shape| GpuArticulatedBoundsShape::Sphere {
                            link: shape.link,
                            center: shape.local_center,
                            radius: shape.radius,
                        })
                        .collect::<Vec<_>>();
                    shapes.extend(capsules.iter().map(|shape| {
                        GpuArticulatedBoundsShape::Capsule {
                            link: shape.link,
                            a: shape.local_a,
                            b: shape.local_b,
                            radius: shape.radius,
                        }
                    }));
                    shapes.extend(boxes.iter().map(|shape| GpuArticulatedBoundsShape::Box {
                        link: shape.link,
                        pose: shape.local_pose,
                        half_extents: shape.half_extents,
                    }));
                    shapes
                })
                .collect::<Vec<_>>();
            let shape_slices = bounds_shapes.iter().map(Vec::as_slice).collect::<Vec<_>>();
            Some(
                GpuArticulatedShapeBoundsBatch::new(
                    context.device(),
                    &poses,
                    &articulations,
                    &shape_slices,
                )?
                .with_motion(
                    &poses,
                    &state,
                    &mass,
                    &articulations,
                    &shape_slices,
                    timestep,
                )?,
            )
        } else {
            None
        };
        let lbvh = shape_bounds
            .as_ref()
            .map(|_| GpuLbvh::new(context.device()));
        let zero_forces = dimensions
            .iter()
            .map(|&n| DVector::zeros(n))
            .collect::<Vec<_>>();
        let forces = GpuArticulatedForceBatch::new(&mass, &zero_forces, &gravities)?;
        forces.set_link_loads(&loads)?;
        let joint_forces = GpuArticulatedJointForceBatch::new_implicit(
            &state, &forces, &joints, &armatures, timestep,
        )?;
        let mut joint_limits = GpuArticulatedJointLimitBatch::new_with_floating_roots(
            &state,
            &mass,
            &articulations,
            &speed_limits,
            timestep,
            floating_roots,
        )?;
        if inputs
            .iter()
            .any(|input| input.coordinate_velocity_limits.is_some())
        {
            let caps = inputs
                .iter()
                .zip(floating_roots)
                .map(|(input, &floating)| {
                    input.coordinate_velocity_limits.clone().unwrap_or_else(|| {
                        let mut caps =
                            vec![input.joint_velocity_limit; input.state.velocities.len()];
                        if floating {
                            caps[..6].fill(None);
                        }
                        caps
                    })
                })
                .collect::<Vec<_>>();
            joint_limits.update_coordinate_velocity_limits(&caps)?;
        }
        let spherical_drives = match (&spherical, spherical_drive_input) {
            (Some(spherical), Some(drives)) => Some(GpuArticulatedSphericalDriveBatch::new(
                &state, spherical, &forces, drives,
            )?),
            _ => None,
        };
        let velocity_bias = if let Some(spherical) = &spherical {
            GpuArticulatedVelocityBiasBatch::new_with_spherical_state(
                &state,
                &mass,
                &forces,
                &articulations,
                &poses,
                spherical,
            )?
        } else {
            GpuArticulatedVelocityBiasBatch::new_with_floating_roots(
                &state,
                &mass,
                &forces,
                &articulations,
                &poses,
            )?
        };
        let root_integration = if floating_roots.iter().any(|&flag| flag) {
            Some(GpuArticulatedRootBatch::new(&poses, &state, timestep)?)
        } else {
            None
        };
        let ground_contact = if ground_spheres.iter().any(|contacts| !contacts.is_empty())
            || joint_couplings.iter().any(|rows| !rows.is_empty())
            || link_point_constraints.iter().any(|rows| !rows.is_empty())
            || link_fixed_constraints.iter().any(|rows| !rows.is_empty())
            || joint_frictions
                .iter()
                .any(|frictions| !frictions.is_empty())
            || ground_boxes.iter().any(|contacts| !contacts.is_empty())
            || ground_axial_shapes
                .iter()
                .any(|contacts| !contacts.is_empty())
            || sphere_pairs.iter().any(|pairs| !pairs.is_empty())
            || static_sphere_pairs.iter().any(|pairs| !pairs.is_empty())
            || static_capsule_sphere_pairs
                .iter()
                .any(|pairs| !pairs.is_empty())
            || static_box_sphere_pairs
                .iter()
                .any(|pairs| !pairs.is_empty())
            || static_sphere_capsule_pairs
                .iter()
                .any(|pairs| !pairs.is_empty())
            || static_capsule_pairs.iter().any(|pairs| !pairs.is_empty())
            || static_sphere_box_pairs
                .iter()
                .any(|pairs| !pairs.is_empty())
            || static_capsule_box_pairs
                .iter()
                .any(|pairs| !pairs.is_empty())
            || static_box_pairs.iter().any(|pairs| !pairs.is_empty())
            || static_box_capsule_pairs
                .iter()
                .any(|pairs| !pairs.is_empty())
            || static_axial_sphere_pairs
                .iter()
                .any(|pairs| !pairs.is_empty())
            || static_axial_capsule_pairs
                .iter()
                .any(|pairs| !pairs.is_empty())
            || static_axial_box_pairs.iter().any(|pairs| !pairs.is_empty())
            || static_axial_convex_pairs
                .iter()
                .any(|pairs| !pairs.is_empty())
            || axial_convex_pairs.iter().any(|pairs| !pairs.is_empty())
            || axial_sphere_pairs.iter().any(|pairs| !pairs.is_empty())
            || axial_box_pairs.iter().any(|pairs| !pairs.is_empty())
            || axial_capsule_pairs.iter().any(|pairs| !pairs.is_empty())
            || axial_pairs.iter().any(|pairs| !pairs.is_empty())
            || convex_sphere_pairs.iter().any(|pairs| !pairs.is_empty())
            || static_convex_sphere_pairs
                .iter()
                .any(|pairs| !pairs.is_empty())
            || static_convex_capsule_pairs
                .iter()
                .any(|pairs| !pairs.is_empty())
            || convex_capsule_pairs.iter().any(|pairs| !pairs.is_empty())
            || convex_pairs.iter().any(|pairs| !pairs.is_empty())
            || static_convex_pairs.iter().any(|pairs| !pairs.is_empty())
            || scene_convex_sphere_pairs
                .iter()
                .any(|pairs| !pairs.is_empty())
            || scene_mesh_sphere_pairs
                .iter()
                .any(|pairs| !pairs.is_empty())
            || scene_polyline_sphere_pairs
                .iter()
                .any(|pairs| !pairs.is_empty())
            || scene_mesh_capsule_pairs
                .iter()
                .any(|pairs| !pairs.is_empty())
            || scene_polyline_capsule_pairs
                .iter()
                .any(|pairs| !pairs.is_empty())
            || scene_mesh_box_pairs.iter().any(|pairs| !pairs.is_empty())
            || scene_polyline_box_pairs
                .iter()
                .any(|pairs| !pairs.is_empty())
            || scene_mesh_axial_pairs.iter().any(|pairs| !pairs.is_empty())
            || scene_polyline_axial_pairs
                .iter()
                .any(|pairs| !pairs.is_empty())
            || scene_polyline_convex_pairs
                .iter()
                .any(|pairs| !pairs.is_empty())
            || scene_mesh_convex_pairs
                .iter()
                .any(|pairs| !pairs.is_empty())
            || scene_convex_capsule_pairs
                .iter()
                .any(|pairs| !pairs.is_empty())
            || has_dynamic_pairs
            || capsule_sphere_pairs.iter().any(|pairs| !pairs.is_empty())
            || capsule_pairs.iter().any(|pairs| !pairs.is_empty())
            || sphere_box_pairs.iter().any(|pairs| !pairs.is_empty())
            || capsule_box_pairs.iter().any(|pairs| !pairs.is_empty())
            || box_pairs.iter().any(|pairs| !pairs.is_empty())
        {
            Some(GpuArticulatedGroundContactBatch::new_with_settings(
                &state,
                &poses,
                &mass,
                &articulations,
                &ground_spheres,
                GpuArticulatedContactSettings {
                    boxes: &ground_boxes,
                    axial_shapes: &ground_axial_shapes,
                    manifold_starts: &ground_manifold_starts,
                    joint_frictions: &joint_frictions,
                    joint_couplings: &joint_couplings,
                    link_point_constraints: &link_point_constraints,
                    link_fixed_constraints: &link_fixed_constraints,
                    pairs: &sphere_pairs,
                    static_sphere_pairs: &static_sphere_pairs,
                    static_capsule_sphere_pairs: &static_capsule_sphere_pairs,
                    static_box_sphere_pairs: &static_box_sphere_pairs,
                    static_sphere_capsule_pairs: &static_sphere_capsule_pairs,
                    static_capsule_pairs: &static_capsule_pairs,
                    static_sphere_box_pairs: &static_sphere_box_pairs,
                    static_capsule_box_pairs: &static_capsule_box_pairs,
                    static_box_pairs: &static_box_pairs,
                    static_box_capsule_pairs: &static_box_capsule_pairs,
                    static_axial_sphere_pairs: &static_axial_sphere_pairs,
                    static_axial_capsule_pairs: &static_axial_capsule_pairs,
                    static_axial_box_pairs: &static_axial_box_pairs,
                    static_axial_convex_pairs: &static_axial_convex_pairs,
                    axial_convex_pairs: &axial_convex_pairs,
                    axial_sphere_pairs: &axial_sphere_pairs,
                    axial_box_pairs: &axial_box_pairs,
                    axial_capsule_pairs: &axial_capsule_pairs,
                    axial_pairs: &axial_pairs,
                    convex_sphere_pairs: &convex_sphere_pairs,
                    static_convex_sphere_pairs: &static_convex_sphere_pairs,
                    static_convex_capsule_pairs: &static_convex_capsule_pairs,
                    convex_capsule_pairs: &convex_capsule_pairs,
                    convex_pairs: &convex_pairs,
                    static_convex_pairs: &static_convex_pairs,
                    scene_convex_sphere_pairs: &scene_convex_sphere_pairs,
                    scene_mesh_sphere_pairs: &scene_mesh_sphere_pairs,
                    scene_polyline_sphere_pairs: &scene_polyline_sphere_pairs,
                    scene_mesh_capsule_pairs: &scene_mesh_capsule_pairs,
                    scene_polyline_capsule_pairs: &scene_polyline_capsule_pairs,
                    scene_mesh_box_pairs: &scene_mesh_box_pairs,
                    scene_polyline_box_pairs: &scene_polyline_box_pairs,
                    scene_mesh_axial_pairs: &scene_mesh_axial_pairs,
                    scene_polyline_axial_pairs: &scene_polyline_axial_pairs,
                    scene_mesh_convex_pairs: &scene_mesh_convex_pairs,
                    scene_polyline_convex_pairs: &scene_polyline_convex_pairs,
                    scene_convex_capsule_pairs: &scene_convex_capsule_pairs,
                    dynamic_spheres: &dynamic_spheres,
                    dynamic_capsules: &dynamic_capsules,
                    dynamic_boxes: &dynamic_boxes,
                    dynamic_material_rules: &dynamic_material_rules,
                    capsule_sphere_pairs: &capsule_sphere_pairs,
                    capsule_pairs: &capsule_pairs,
                    sphere_box_pairs: &sphere_box_pairs,
                    capsule_box_pairs: &capsule_box_pairs,
                    box_pairs: &box_pairs,
                    iterations: &contact_iterations,
                    warm_start: &contact_warm_start,
                },
                timestep,
            )?)
        } else {
            None
        };
        let topology_mobility = inputs
            .iter()
            .zip(&root_dofs)
            .map(|(input, &root_dofs)| {
                let couplings = input
                    .joint_couplings
                    .iter()
                    .filter_map(|coupling| {
                        coupling.source.map(|source| (coupling.follower, source))
                    })
                    .collect::<Vec<_>>();
                input
                    .articulation
                    .coordinate_mobility_groups(root_dofs != 0, &couplings)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let coordinate_owners = inputs
            .iter()
            .zip(&root_dofs)
            .map(|(input, &root_dofs)| input.articulation.coordinate_affected_links(root_dofs != 0))
            .collect();
        let mut batch = Self {
            motor_targets: None,
            scalar_motor_links: inputs
                .iter()
                .zip(&root_dofs)
                .map(|(input, &root)| {
                    let (_, _, joints) = input.articulation.gpu_pose_topology();
                    joints
                        .iter()
                        .enumerate()
                        .filter_map(|(edge, joint)| {
                            let range = input.articulation.joint_coordinate_range(edge)?;
                            (range.len() == 1
                                && input.articulation.joint_coordinate_scale(edge) == Some(1.0)
                                && input.articulation.joint_coordinate_offset(edge) == Some(0.0))
                            .then_some((root + range.start, joint.child))
                        })
                        .collect()
                })
                .collect(),
            coordinate_owners,
            sleep_freeze: None,
            actuation_efforts: None,
            topology_mobility,
            device: context.device().clone(),
            queue: context.queue().clone(),
            mass,
            state,
            poses,
            link_terms,
            joint_forces,
            accepted_joints: std::sync::Mutex::new(
                inputs.iter().map(|input| input.joints.clone()).collect(),
            ),
            joint_limits,
            velocity_bias,
            forces,
            ground_contact,
            integrate_external_spheres: false,
            shape_bounds,
            lbvh,
            timestep,
            dimensions,
            joint_bounds,
            root_integration,
            root_dofs,
            spherical,
            spherical_drives,
            accepted_spherical_drives: std::sync::Mutex::new(
                spherical_drive_input.map_or_else(Vec::new, <[_]>::to_vec),
            ),
        };
        batch.initialize_external_sphere_bodies(inputs)?;
        batch.initialize_external_box_bodies(inputs)?;
        batch.initialize_external_capsule_bodies(inputs)?;
        batch.initialize_external_axial_bodies(inputs)?;
        batch.initialize_external_convex_bodies(inputs)?;
        batch.initialize_external_constraint_bodies(inputs)?;
        batch.initialize_external_indexed_bodies(inputs)?;
        Ok(batch)
    }

    pub(crate) fn initialize_external_sphere_bodies(
        &mut self,
        inputs: &[GpuArticulatedDynamicsInput<'_>],
    ) -> Result<(), GpuArticulatedDynamicsError> {
        use crate::gpu_articulated_ground_contact::{
            GpuArticulatedExternalSphereMotions, GpuArticulatedSphereMotion,
        };
        if inputs
            .iter()
            .all(|input| input.external_sphere_bodies.is_none())
        {
            return Ok(());
        }
        let mut motions = GpuArticulatedExternalSphereMotions::default();
        let mut orbits =
            crate::gpu_articulated_ground_contact::GpuArticulatedExternalSphereOrbits::default();
        for input in inputs {
            let counts = [
                input.static_sphere_pairs.len(),
                input.static_capsule_sphere_pairs.len(),
                input.static_box_sphere_pairs.len(),
                input
                    .static_axial_sphere_pairs
                    .iter()
                    .filter(|pair| !pair.axial_is_static)
                    .count(),
                input.static_convex_sphere_pairs.len(),
            ];
            let values = input.external_sphere_bodies.as_ref().map(|bodies| {
                [
                    &bodies.spheres,
                    &bodies.capsules,
                    &bodies.boxes,
                    &bodies.axial,
                    &bodies.convex,
                ]
            });
            if values.as_ref().is_some_and(|values| {
                values
                    .iter()
                    .zip(counts)
                    .any(|(values, count)| values.len() != count)
            }) {
                return Err(GpuArticulatedDynamicsError::InvalidInput);
            }
            for (index, (motion_field, orbit_field)) in [
                (&mut motions.spheres, &mut orbits.spheres),
                (&mut motions.capsules, &mut orbits.capsules),
                (&mut motions.boxes, &mut orbits.boxes),
                (&mut motions.axial, &mut orbits.axial),
                (&mut motions.convex, &mut orbits.convex),
            ]
            .into_iter()
            .enumerate()
            {
                let bodies = values
                    .as_ref()
                    .map_or_else(|| vec![None; counts[index]], |values| values[index].clone());
                motion_field.push(
                    bodies
                        .iter()
                        .map(|body| {
                            body.map_or_else(GpuArticulatedSphereMotion::default, |body| {
                                GpuArticulatedSphereMotion {
                                    linear_velocity: body.orbit.linear_velocity,
                                    angular_velocity: body.angular_velocity,
                                }
                            })
                        })
                        .collect(),
                );
                orbit_field.push(
                    bodies
                        .iter()
                        .map(|body| body.map(|body| body.orbit))
                        .collect(),
                );
            }
        }
        if self.ground_contact.is_none() {
            if motions
                .spheres
                .iter()
                .chain(&motions.capsules)
                .chain(&motions.boxes)
                .chain(&motions.axial)
                .chain(&motions.convex)
                .any(|values| !values.is_empty())
            {
                return Err(GpuArticulatedDynamicsError::InvalidInput);
            }
            return Ok(());
        }
        let _ = self.update_external_sphere_motions(&motions)?;
        let _ = self.update_static_sphere_orbits(&orbits.spheres)?;
        let _ = self.update_static_capsule_sphere_orbits(&orbits.capsules)?;
        let _ = self.update_static_box_sphere_orbits(&orbits.boxes)?;
        let _ = self.update_static_axial_sphere_orbits(&orbits.axial)?;
        let _ = self.update_static_convex_sphere_orbits(&orbits.convex)?;
        self.set_static_sphere_integration(true);
        Ok(())
    }

    pub(crate) fn has_contact_rows(&self) -> bool {
        self.ground_contact.is_some()
    }

    pub(crate) fn initialize_external_box_bodies(
        &mut self,
        inputs: &[GpuArticulatedDynamicsInput<'_>],
    ) -> Result<(), GpuArticulatedDynamicsError> {
        use crate::gpu_articulated_ground_contact::GpuArticulatedBoxContactKind as Kind;
        if inputs
            .iter()
            .all(|input| input.external_box_bodies.is_none())
        {
            return Ok(());
        }
        let mut groups = [Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for input in inputs {
            let counts = [
                input.static_sphere_box_pairs.len(),
                input.static_capsule_box_pairs.len(),
                input.static_box_pairs.len(),
                input
                    .static_axial_box_pairs
                    .iter()
                    .filter(|pair| !pair.axial_is_static)
                    .count(),
                input.static_convex_pairs.len(),
            ];
            let fields = input.external_box_bodies.as_ref().map(|body| {
                [
                    &body.spheres,
                    &body.capsules,
                    &body.boxes,
                    &body.axial,
                    &body.convex,
                ]
            });
            for index in 0..5 {
                let values = fields
                    .as_ref()
                    .map_or_else(|| vec![None; counts[index]], |fields| fields[index].clone());
                if values.len() != counts[index] {
                    return Err(GpuArticulatedDynamicsError::InvalidInput);
                }
                groups[index].push(values);
            }
        }
        if !self.has_contact_rows() {
            if groups.iter().flatten().any(|values| !values.is_empty()) {
                return Err(GpuArticulatedDynamicsError::InvalidInput);
            }
            return Ok(());
        }
        for (kind, bodies) in [
            Kind::Sphere,
            Kind::Capsule,
            Kind::Box,
            Kind::Axial,
            Kind::Convex,
        ]
        .into_iter()
        .zip(&groups)
        {
            self.update_prescribed_box_bodies(kind, bodies)?;
        }
        Ok(())
    }

    pub(crate) fn initialize_external_capsule_bodies(
        &mut self,
        inputs: &[GpuArticulatedDynamicsInput<'_>],
    ) -> Result<(), GpuArticulatedDynamicsError> {
        use crate::gpu_articulated_ground_contact::GpuArticulatedCapsuleContactKind as Kind;
        if inputs
            .iter()
            .all(|input| input.external_capsule_bodies.is_none())
        {
            return Ok(());
        }
        let mut groups = [Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for input in inputs {
            let counts = [
                input.static_sphere_capsule_pairs.len(),
                input.static_capsule_pairs.len(),
                input.static_box_capsule_pairs.len(),
                input
                    .static_axial_capsule_pairs
                    .iter()
                    .filter(|pair| !pair.axial_is_static)
                    .count(),
                input.static_convex_capsule_pairs.len(),
            ];
            let fields = input.external_capsule_bodies.as_ref().map(|body| {
                [
                    &body.spheres,
                    &body.capsules,
                    &body.boxes,
                    &body.axial,
                    &body.convex,
                ]
            });
            for index in 0..5 {
                let values = fields
                    .as_ref()
                    .map_or_else(|| vec![None; counts[index]], |fields| fields[index].clone());
                if values.len() != counts[index] {
                    return Err(GpuArticulatedDynamicsError::InvalidInput);
                }
                groups[index].push(values);
            }
        }
        if !self.has_contact_rows() {
            if groups.iter().flatten().any(|values| !values.is_empty()) {
                return Err(GpuArticulatedDynamicsError::InvalidInput);
            }
            return Ok(());
        }
        for (kind, bodies) in [
            Kind::Sphere,
            Kind::Capsule,
            Kind::Box,
            Kind::Axial,
            Kind::Convex,
        ]
        .into_iter()
        .zip(&groups)
        {
            self.update_prescribed_capsule_bodies(kind, bodies)?;
        }
        Ok(())
    }

    pub(crate) fn initialize_external_axial_bodies(
        &mut self,
        inputs: &[GpuArticulatedDynamicsInput<'_>],
    ) -> Result<(), GpuArticulatedDynamicsError> {
        use crate::gpu_articulated_ground_contact::GpuArticulatedAxialContactKind as Kind;
        if inputs
            .iter()
            .all(|input| input.external_axial_bodies.is_none())
        {
            return Ok(());
        }
        let mut groups = [Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for input in inputs {
            let counts = [
                input
                    .static_axial_sphere_pairs
                    .iter()
                    .filter(|pair| pair.axial_is_static)
                    .count(),
                input
                    .static_axial_capsule_pairs
                    .iter()
                    .filter(|pair| pair.axial_is_static)
                    .count(),
                input
                    .static_axial_box_pairs
                    .iter()
                    .filter(|pair| pair.axial_is_static)
                    .count(),
                input
                    .axial_pairs
                    .iter()
                    .filter(|pair| pair.first_is_static)
                    .count(),
                input
                    .axial_convex_pairs
                    .iter()
                    .filter(|pair| pair.axial_is_static)
                    .count(),
            ];
            let fields = input.external_axial_bodies.as_ref().map(|body| {
                [
                    &body.spheres,
                    &body.capsules,
                    &body.boxes,
                    &body.axial,
                    &body.convex,
                ]
            });
            for index in 0..5 {
                let values = fields
                    .as_ref()
                    .map_or_else(|| vec![None; counts[index]], |fields| fields[index].clone());
                if values.len() != counts[index] {
                    return Err(GpuArticulatedDynamicsError::InvalidInput);
                }
                groups[index].push(values);
            }
        }
        if !self.has_contact_rows() {
            if groups.iter().flatten().any(|values| !values.is_empty()) {
                return Err(GpuArticulatedDynamicsError::InvalidInput);
            }
            return Ok(());
        }
        for (kind, bodies) in [
            Kind::Sphere,
            Kind::Capsule,
            Kind::Box,
            Kind::Axial,
            Kind::Convex,
        ]
        .into_iter()
        .zip(&groups)
        {
            self.update_prescribed_axial_bodies(kind, bodies)?;
        }
        Ok(())
    }

    pub(crate) fn initialize_external_indexed_bodies(
        &mut self,
        inputs: &[GpuArticulatedDynamicsInput<'_>],
    ) -> Result<(), GpuArticulatedDynamicsError> {
        if inputs
            .iter()
            .all(|input| input.external_indexed_bodies.is_none())
        {
            return Ok(());
        }
        let bodies = inputs
            .iter()
            .map(|input| {
                let count = input.scene_mesh_sphere_pairs.len()
                    + input.scene_polyline_sphere_pairs.len()
                    + input.scene_mesh_capsule_pairs.len()
                    + input.scene_polyline_capsule_pairs.len()
                    + input.scene_mesh_box_pairs.len()
                    + input.scene_polyline_box_pairs.len()
                    + input.scene_mesh_axial_pairs.len()
                    + input.scene_polyline_axial_pairs.len()
                    + input.scene_mesh_convex_pairs.len()
                    + input.scene_polyline_convex_pairs.len();
                let values = input
                    .external_indexed_bodies
                    .clone()
                    .unwrap_or_else(|| vec![None; count]);
                if values.len() != count {
                    return Err(GpuArticulatedDynamicsError::InvalidInput);
                }
                Ok(values)
            })
            .collect::<Result<Vec<_>, _>>()?;
        if bodies.iter().flatten().all(Option::is_none) {
            return Ok(());
        }
        self.update_prescribed_indexed_bodies(&bodies)
    }

    pub(crate) fn initialize_external_constraint_bodies(
        &mut self,
        inputs: &[GpuArticulatedDynamicsInput<'_>],
    ) -> Result<(), GpuArticulatedDynamicsError> {
        if inputs
            .iter()
            .all(|input| input.external_constraint_bodies.is_none())
        {
            return Ok(());
        }
        let bodies = inputs
            .iter()
            .map(|input| {
                let count = input
                    .link_point_constraints
                    .iter()
                    .filter(|c| c.link_b.is_none())
                    .count()
                    + input
                        .link_fixed_constraints
                        .iter()
                        .filter(|c| c.link_b.is_none())
                        .count();
                let values = input
                    .external_constraint_bodies
                    .clone()
                    .unwrap_or_else(|| vec![None; count]);
                if values.len() != count {
                    return Err(GpuArticulatedDynamicsError::InvalidInput);
                }
                Ok(values)
            })
            .collect::<Result<Vec<_>, _>>()?;
        if bodies.iter().flatten().all(Option::is_none) {
            return Ok(());
        }
        self.update_prescribed_constraint_bodies(&bodies)
    }

    pub(crate) fn initialize_external_convex_bodies(
        &mut self,
        inputs: &[GpuArticulatedDynamicsInput<'_>],
    ) -> Result<(), GpuArticulatedDynamicsError> {
        if inputs
            .iter()
            .all(|input| input.external_convex_bodies.is_none())
        {
            return Ok(());
        }
        let mut rounded = Vec::new();
        let mut other = Vec::new();
        for input in inputs {
            let counts = [
                input.scene_convex_sphere_pairs.len() + input.scene_convex_capsule_pairs.len(),
                input.static_convex_pairs.len() + input.static_axial_convex_pairs.len(),
            ];
            let (a, b) = input.external_convex_bodies.as_ref().map_or_else(
                || (vec![None; counts[0]], vec![None; counts[1]]),
                |body| (body.rounded.clone(), body.polyhedron_axial.clone()),
            );
            if a.len() != counts[0] || b.len() != counts[1] {
                return Err(GpuArticulatedDynamicsError::InvalidInput);
            }
            rounded.push(a);
            other.push(b);
        }
        if !self.has_contact_rows() {
            if rounded
                .iter()
                .chain(&other)
                .any(|values| !values.is_empty())
            {
                return Err(GpuArticulatedDynamicsError::InvalidInput);
            }
            return Ok(());
        }
        self.update_prescribed_convex_rounded_bodies(&rounded)?;
        self.update_prescribed_convex_other_bodies(&other)?;
        Ok(())
    }

    fn clear_sleep_diagnostics(&self) {
        if let Some(freeze) = &self.sleep_freeze {
            let mut encoder = self.device.create_command_encoder(&Default::default());
            freeze.encode_clear_frozen(&mut encoder);
            let _ = self.queue.submit(Some(encoder.finish()));
        }
    }

    /// Enable resident threshold-based sleeping with conservative topology groups.
    /// Requires contact activity enabled. Gravity-opposing static contact or a
    /// fixed world anchor is required; point anchors alone do not permit sleeping.
    /// Persistent loads, actuation, gravity changes, motion and lost contact wake
    /// components. This direction criterion is not a force/torque equilibrium test.
    /// Configuration resets idle history. Invalid settings preserve the old setup.
    /// Other setup errors leave automatic freezing
    /// disabled, but may retain already configured tracking and wake passes.
    pub fn enable_contact_sleep(
        &mut self,
        settings: &[Option<crate::sleep::SleepSettings>],
    ) -> Result<(), GpuArticulatedDynamicsError> {
        // Validate settings before changing mobility or enabling auxiliary passes.
        self.update_contact_idle_settings(settings)?;
        self.finish_enable_contact_sleep()
    }

    /// Enable resident sleeping with separate thresholds for every local link.
    /// All physical members of an island must allow sleep and meet their own
    /// thresholds. Massless helpers inherit the physical component's decision.
    /// Uses the same support and wake policy as enable_contact_sleep.
    pub fn enable_contact_link_sleep(
        &mut self,
        settings: &[Vec<Option<crate::sleep::SleepSettings>>],
    ) -> Result<(), GpuArticulatedDynamicsError> {
        self.update_contact_link_idle_settings(settings)?;
        self.finish_enable_contact_sleep()
    }

    fn finish_enable_contact_sleep(&mut self) -> Result<(), GpuArticulatedDynamicsError> {
        self.sleep_freeze = None;
        let freeze = self.build_contact_sleep_freeze()?;
        self.update_contact_mobility_from_topology()?;
        self.enable_contact_load_wake()?;
        self.enable_contact_actuation_wake()?;
        self.enable_contact_gravity_wake()?;
        self.update_contact_loss_wake(true)?;
        self.update_contact_idle_requires_static_support(true)?;
        self.sleep_freeze = Some(freeze);
        Ok(())
    }

    /// Stop applying sleep freezes, retaining tracking settings and wake passes.
    pub fn disable_contact_sleep(&mut self) {
        self.sleep_freeze = None;
    }

    /// Bind a freeze pass to this batch's resident state and complete topology owners.
    /// Requires contact activity enabled. This does not enable sleeping or encode
    /// the pass automatically: support policy and all wake sources must be configured
    /// before using candidate flags to stop integration. The returned pass is a
    /// building block for a split step; encoding it after encode_step is too late.
    /// It must run after contact/idle evaluation and before every integrator.
    pub fn build_contact_sleep_freeze(
        &self,
    ) -> Result<GpuArticulatedSleepFreezeBatch, GpuArticulatedDynamicsError> {
        let contact = self
            .ground_contact
            .as_ref()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?;
        Ok(GpuArticulatedSleepFreezeBatch::new(
            &self.state,
            &self.mass,
            contact.contact_sleep_candidate_buffer()?,
            self.poses.link_ranges(),
            &self.coordinate_owners,
        )?)
    }

    /// Encode one dynamics step without downloading intermediate state.
    pub fn encode_step(
        &self,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(), GpuArticulatedDynamicsError> {
        self.encode_contact_step(encoder)?;
        self.encode_integration(encoder);
        Ok(())
    }

    /// Evaluate forces, contact, wake propagation and idle without integration.
    /// Pair exactly once with encode_integration on the same encoder for each
    /// timestep. A sleep freeze pass may be inserted between these two phases.
    /// Submit neither phase alone as a complete dynamics step.
    pub fn encode_contact_step(
        &self,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(), GpuArticulatedDynamicsError> {
        if let Some(targets) = &self.motor_targets {
            targets.encode_apply(&self.device, encoder);
        }
        self.poses.encode(encoder);
        self.link_terms.encode(encoder);
        self.joint_forces.encode(encoder);
        if let Some(drives) = &self.spherical_drives {
            drives.encode(encoder);
        }
        // Wake from user drives/passive forces, not inertial velocity bias.
        if let Some(efforts) = &self.actuation_efforts {
            encoder.copy_buffer_to_buffer(
                self.forces.base_force_buffer(),
                0,
                efforts,
                0,
                efforts.size(),
            );
        }
        self.velocity_bias.encode(encoder);
        self.forces.encode(encoder);
        if let Some(contact) = &self.ground_contact {
            self.mass.encode_with_inverse(encoder);
            if let (Some(bounds), Some(lbvh)) = (&self.shape_bounds, &self.lbvh) {
                let candidates = bounds.encode_candidates(&self.device, encoder, lbvh)?;
                contact.encode_dynamic_shape_rows(encoder, lbvh, &candidates)?;
            }
            contact.encode(encoder);
        } else {
            self.mass.encode(encoder);
        }
        Ok(())
    }

    /// Integrate scalar, spherical and floating-root state from the evaluated step.
    /// Must follow encode_contact_step exactly once on the same encoder. Optional
    /// externally supplied freezing belongs immediately before this phase.
    /// Configured automatic sleeping is applied here before every integrator,
    /// including when callers use the split-step API directly.
    pub fn encode_integration(&self, encoder: &mut wgpu::CommandEncoder) {
        if let Some(freeze) = &self.sleep_freeze {
            freeze.encode(encoder);
        }
        self.joint_limits.encode(encoder);
        if let Some(spherical) = &self.spherical {
            spherical.encode(encoder);
        }
        if let Some(root) = &self.root_integration {
            root.encode(encoder);
        }
        if self.integrate_external_spheres
            && let Some(contact) = &self.ground_contact
        {
            contact.encode_static_sphere_integration(encoder);
        }
    }

    /// Submit any number of successive steps on the owning device.
    pub fn submit_steps(&self, count: usize) -> Result<(), GpuArticulatedDynamicsError> {
        if count == 0 {
            return Ok(());
        }
        let mut encoder = self.device.create_command_encoder(&Default::default());
        for _ in 0..count {
            self.encode_step(&mut encoder)?;
        }
        let _ = self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    /// Fixed integration timestep in seconds.
    pub fn timestep(&self) -> f64 {
        self.timestep
    }

    /// Configure resident scalar motor targets with one equally sized mapping per environment.
    ///
    /// Each mapping row is an action coordinate. Destinations must be unique within
    /// an environment and name a scalar joint's child link and generalized slot.
    /// Floating-root and spherical slots, scaled/offset mimic links, disabled motors,
    /// and position targets on velocity-only motors are rejected before any change.
    /// `delays[environment]` counts physics substeps that retain the previous target.
    /// Reconfiguration cancels pending actions but retains currently applied targets.
    ///
    /// # Errors
    /// Returns `InvalidInput` for incompatible mappings, delays, native motors, or GPU capacity.
    pub fn enable_motor_target_control(
        &mut self,
        mappings: &[Vec<GpuMotorTargetMapping>],
        delays: &[u32],
    ) -> Result<(), GpuArticulatedDynamicsError> {
        let env_count = self.dimensions.len();
        if mappings.len() != env_count || delays.len() != env_count {
            return Err(GpuArticulatedDynamicsError::InvalidInput);
        }
        let width = mappings.first().map_or(0, Vec::len);
        if width == 0 || mappings.iter().any(|mapping| mapping.len() != width) {
            return Err(GpuArticulatedDynamicsError::InvalidInput);
        }
        let count = width
            .checked_mul(env_count)
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?;
        let limits = self.device.limits();
        if count > u32::MAX as usize
            || count.div_ceil(64) > limits.max_compute_workgroups_per_dimension as usize
            || count as u64 * 16 > u64::from(limits.max_storage_buffer_binding_size)
            || count as u64 * 16 > limits.max_buffer_size
        {
            return Err(GpuArticulatedDynamicsError::InvalidInput);
        }
        let accepted = self
            .accepted_joints
            .lock()
            .map_err(|_| GpuArticulatedDynamicsError::InvalidInput)?;
        let mut packed = Vec::with_capacity(count);
        let mut offset = 0;
        for (env, mapping) in mappings.iter().enumerate() {
            let mut used = std::collections::BTreeSet::new();
            for (row, target) in mapping.iter().enumerate() {
                let motor = accepted[env]
                    .get(target.coordinate)
                    .and_then(|joint| joint.motor)
                    .ok_or(GpuArticulatedDynamicsError::InvalidInput)?;
                if !used.insert(target.coordinate)
                    || !self.scalar_motor_links[env].contains(&(target.coordinate, target.link))
                    || (target.mode == GpuMotorTargetMode::Position
                        && motor.position_target.is_none())
                {
                    return Err(GpuArticulatedDynamicsError::InvalidInput);
                }
                let mode = match target.mode {
                    GpuMotorTargetMode::Position => 0,
                    GpuMotorTargetMode::Velocity => 1,
                };
                packed.push(PackedMotorTargetMapping {
                    coordinate: u32::try_from(offset + target.coordinate)
                        .map_err(|_| GpuArticulatedDynamicsError::InvalidInput)?,
                    action: (row * env_count + env) as u32,
                    mode,
                    delay: delays[env],
                });
            }
            offset += self.dimensions[env];
        }
        self.motor_targets = Some(GpuMotorTargetControl::new(
            &self.device,
            self.joint_forces.parameter_buffer(),
            &packed,
            env_count as u32,
        ));
        Ok(())
    }

    /// Latch actions from a STORAGE f32 buffer in `[actuated coordinate, environment]` order.
    ///
    /// The buffer must contain exactly mapping-width times environment-count values,
    /// all finite, with position targets inside configured joint bounds. The caller
    /// may encode its GPU action producer immediately before this pass on the same
    /// encoder; there is no readback or CPU target upload. Each latch supersedes any
    /// pending action and restarts each environment's configured delay. Delay zero
    /// applies before the next force evaluation; delay k retains the old target for
    /// exactly k contact/integration substeps, then applies before substep k + 1.
    /// Both whole and split resident stepping APIs consume the delay on the GPU.
    ///
    /// # Errors
    /// Returns `InvalidInput` if control is not configured or buffer size/usage is incompatible.
    pub fn encode_motor_targets(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        actions: &wgpu::Buffer,
    ) -> Result<(), GpuArticulatedDynamicsError> {
        let targets = self
            .motor_targets
            .as_ref()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?;
        if !actions.usage().contains(wgpu::BufferUsages::STORAGE)
            || actions.size() != targets.action_bytes()
        {
            return Err(GpuArticulatedDynamicsError::InvalidInput);
        }
        targets.encode_latch(&self.device, encoder, actions);
        Ok(())
    }

    /// Update per-environment substep delays from a resident `u32` STORAGE buffer.
    ///
    /// Supply one delay per environment. Pending actions restart their countdown;
    /// applied targets stay unchanged. This may precede action scatter on the same
    /// encoder after a GPU policy/randomization producer, without host readback.
    pub fn encode_motor_target_delays(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        delays: &wgpu::Buffer,
    ) -> Result<(), GpuArticulatedDynamicsError> {
        let targets = self
            .motor_targets
            .as_ref()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?;
        if !delays.usage().contains(wgpu::BufferUsages::STORAGE)
            || delays.size() != self.dimensions.len() as u64 * 4
        {
            return Err(GpuArticulatedDynamicsError::InvalidInput);
        }
        targets.encode_delays(&self.device, encoder, delays);
        Ok(())
    }

    /// Update raw joint efforts, motors, and passive laws for the next step.
    pub fn update_joints(
        &self,
        joints: &[Vec<GpuJointForceInput>],
    ) -> Result<(), GpuArticulatedDynamicsError> {
        if self
            .motor_targets
            .as_ref()
            .is_some_and(|targets| !targets.accepts_inputs(joints))
        {
            return Err(GpuArticulatedDynamicsError::InvalidInput);
        }
        if joints.len() != self.joint_bounds.len()
            || joints
                .iter()
                .zip(&self.joint_bounds)
                .any(|(drives, bounds)| !drives_within_bounds(drives, bounds))
            || joints
                .iter()
                .zip(&self.root_dofs)
                .any(|(drives, &root_dof)| {
                    drives.iter().take(root_dof).any(|drive| {
                        drive.motor.is_some()
                            || drive.passive != Default::default()
                            || drive.nonlinear != Default::default()
                    })
                })
        {
            return Err(GpuArticulatedDynamicsError::InvalidInput);
        }
        if let Some(spherical) = &self.spherical
            && joints
                .iter()
                .zip(spherical.velocity_slots())
                .any(|(joints, slots)| !spherical_baseline_only(joints, slots))
        {
            return Err(GpuArticulatedDynamicsError::InvalidInput);
        }
        let mut accepted = self
            .accepted_joints
            .lock()
            .map_err(|_| GpuArticulatedDynamicsError::InvalidInput)?;
        self.joint_forces.update_inputs(joints)?;
        // Host replacement owns the complete drive parameters, so stale GPU actions
        // must not overwrite it on a later substep.
        if let Some(targets) = &self.motor_targets {
            targets.cancel(&self.queue);
        }
        if self.sleep_freeze.is_some() {
            let mut requests = self
                .poses
                .link_ranges()
                .iter()
                .map(|range| vec![false; range.len()])
                .collect::<Vec<_>>();
            for (env, (old, new)) in accepted.iter().zip(joints).enumerate() {
                for (coordinate, (old, new)) in old.iter().zip(new).enumerate() {
                    if old != new {
                        for &link in &self.coordinate_owners[env][coordinate] {
                            requests[env][link] = true;
                        }
                    }
                }
            }
            self.ground_contact
                .as_ref()
                .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
                .add_contact_wake_requests(&self.queue, &requests)?;
        }
        *accepted = joints.to_vec();
        Ok(())
    }

    /// Update quaternion drives while retaining resident spherical poses.
    pub fn update_spherical_drives(
        &self,
        drives: &[Vec<Option<SphericalJointDrive>>],
    ) -> Result<(), GpuArticulatedDynamicsError> {
        let mut accepted = self
            .accepted_spherical_drives
            .lock()
            .map_err(|_| GpuArticulatedDynamicsError::InvalidInput)?;
        self.spherical_drives
            .as_ref()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .update(drives)?;
        if self.sleep_freeze.is_some() {
            let spherical = self
                .spherical
                .as_ref()
                .ok_or(GpuArticulatedDynamicsError::InvalidInput)?;
            let mut requests = self
                .poses
                .link_ranges()
                .iter()
                .map(|range| vec![false; range.len()])
                .collect::<Vec<_>>();
            for (env, ((old, new), slots)) in accepted
                .iter()
                .zip(drives)
                .zip(spherical.velocity_slots())
                .enumerate()
            {
                for ((old, new), &slot) in old.iter().zip(new).zip(slots) {
                    if old != new {
                        for coordinate in slot..slot + 3 {
                            for &link in &self.coordinate_owners[env][coordinate] {
                                requests[env][link] = true;
                            }
                        }
                    }
                }
            }
            self.ground_contact
                .as_ref()
                .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
                .add_contact_wake_requests(&self.queue, &requests)?;
        }
        *accepted = drives.to_vec();
        Ok(())
    }

    /// Reset spherical orientations. Reset generalized state separately to clear its faults.
    pub fn reset_spherical_orientations(
        &self,
        joints: &[Vec<GpuSphericalJointState>],
    ) -> Result<(), GpuArticulatedDynamicsError> {
        self.spherical
            .as_ref()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .reset(joints)?;
        if let Some(contact) = &self.ground_contact {
            contact.clear_cached_impulses(&self.queue);
        }
        self.clear_sleep_diagnostics();
        Ok(())
    }

    /// Update friction efforts on reserved rows before the next step.
    ///
    /// Inputs created with an empty friction vector cannot enable friction without
    /// rebuilding the batch. A full vector, including zeros, reserves every axis.
    pub fn update_joint_frictions(
        &self,
        frictions: &[Vec<f64>],
    ) -> Result<(), GpuArticulatedDynamicsError> {
        if frictions.len() != self.dimensions.len()
            || frictions
                .iter()
                .zip(&self.dimensions)
                .any(|(values, &dimension)| values.len() != dimension)
            || frictions
                .iter()
                .zip(&self.root_dofs)
                .any(|(values, &root_dof)| values.iter().take(root_dof).any(|&value| value != 0.0))
        {
            return Err(GpuArticulatedDynamicsError::InvalidInput);
        }
        if let Some(contact) = &self.ground_contact {
            contact.update_joint_frictions(&self.queue, frictions)?;
        } else if frictions
            .iter()
            .flatten()
            .any(|&value| !value.is_finite() || value != 0.0)
        {
            return Err(GpuArticulatedDynamicsError::InvalidInput);
        }
        Ok(())
    }

    /// Update the absolute speed cap of each environment.
    pub fn update_velocity_limits(
        &mut self,
        velocity_limits: &[Option<f64>],
    ) -> Result<(), GpuArticulatedDynamicsError> {
        self.joint_limits.update_velocity_limits(velocity_limits)?;
        Ok(())
    }

    /// Update speed caps per generalized coordinate without changing position bounds.
    pub fn update_coordinate_velocity_limits(
        &mut self,
        limits: &[Vec<Option<f64>>],
    ) -> Result<(), GpuArticulatedDynamicsError> {
        self.joint_limits
            .update_coordinate_velocity_limits(limits)?;
        Ok(())
    }

    /// Update world gravities without replacing device-resident joint drives.
    pub fn update_gravity(
        &self,
        gravities: &[Vector3<f64>],
    ) -> Result<(), GpuArticulatedDynamicsError> {
        let zero_forces = self
            .dimensions
            .iter()
            .map(|&n| DVector::zeros(n))
            .collect::<Vec<_>>();
        self.forces.update_inputs(&zero_forces, gravities)?;
        Ok(())
    }

    /// Replace all world-frame link loads in stable articulation order.
    pub fn update_link_loads(
        &self,
        loads: &[Vec<GpuMassLinkLoad>],
    ) -> Result<(), GpuArticulatedDynamicsError> {
        self.forces.set_link_loads(loads)?;
        Ok(())
    }

    /// Move stationary centers in the reserved sphere/static-sphere contact pairs.
    /// Returns whether any represented center changed. Radii and pair ordering
    /// stay fixed. Changed rows wake their owners and clear applied sleep diagnostics.
    /// Invalid input retains all geometry, pending requests and diagnostics.
    pub fn update_static_sphere_centers(
        &mut self,
        centers: &[Vec<Vector3<f64>>],
    ) -> Result<bool, GpuArticulatedDynamicsError> {
        let changed = self
            .ground_contact
            .as_mut()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .update_static_sphere_centers(&self.queue, centers)?;
        if changed {
            self.clear_sleep_diagnostics();
        }
        Ok(changed)
    }

    /// Enable GPU center integration for prescribed external spheres.
    /// Disabled by default so externally supplied pose updates keep their meaning.
    pub fn set_static_sphere_integration(&mut self, enabled: bool) {
        self.integrate_external_spheres = enabled;
    }

    /// Configure prescribed bodies for link-sphere/world-box pairs. The current
    /// box collision poses are retained; supplied body poses define their local
    /// offsets. None stops a box at its current pose. Integration runs on GPU.
    pub fn update_prescribed_sphere_box_bodies(
        &mut self,
        bodies: &[Vec<Option<crate::gpu_kinematic_body::GpuKinematicBody>>],
    ) -> Result<(), GpuArticulatedDynamicsError> {
        self.ground_contact
            .as_mut()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .update_prescribed_sphere_box_bodies(&self.queue, bodies)?;
        self.integrate_external_spheres = true;
        self.clear_sleep_diagnostics();
        Ok(())
    }

    /// Download current world-box poses for link-sphere/box pairs, including
    /// prescribed translation and rotation. This synchronizes the owning queue.
    pub fn readback_static_sphere_box_poses(
        &self,
    ) -> Result<Vec<Vec<Isometry3<f64>>>, GpuArticulatedDynamicsError> {
        Ok(self
            .ground_contact
            .as_ref()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .readback_static_sphere_box_poses(&self.queue)?)
    }

    /// Configure prescribed boxes for a reserved contact family. Each pair
    /// updates all endpoint or manifold rows, retaining its current collision
    /// pose. None stops motion. Invalid packed input leaves all buffers intact.
    pub fn update_prescribed_box_bodies(
        &mut self,
        kind: crate::gpu_articulated_ground_contact::GpuArticulatedBoxContactKind,
        bodies: &[Vec<Option<crate::gpu_kinematic_body::GpuKinematicBody>>],
    ) -> Result<(), GpuArticulatedDynamicsError> {
        self.ground_contact
            .as_mut()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .update_prescribed_box_bodies(&self.queue, kind, bodies)?;
        self.integrate_external_spheres = true;
        self.clear_sleep_diagnostics();
        Ok(())
    }

    /// Download world-box transforms for the selected reserved contact family.
    pub fn readback_prescribed_box_poses(
        &self,
        kind: crate::gpu_articulated_ground_contact::GpuArticulatedBoxContactKind,
    ) -> Result<Vec<Vec<Isometry3<f64>>>, GpuArticulatedDynamicsError> {
        Ok(self
            .ground_contact
            .as_ref()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .readback_prescribed_box_poses(&self.queue, kind)?)
    }

    /// Configure prescribed convex hulls in reserved sphere then capsule pair order.
    /// None stops at the current pose. Invalid input preserves all buffers.
    pub fn update_prescribed_convex_rounded_bodies(
        &mut self,
        bodies: &[Vec<Option<crate::gpu_kinematic_body::GpuKinematicBody>>],
    ) -> Result<(), GpuArticulatedDynamicsError> {
        self.ground_contact
            .as_mut()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .update_prescribed_convex_rounded_bodies(&self.queue, bodies)?;
        self.integrate_external_spheres = true;
        self.clear_sleep_diagnostics();
        Ok(())
    }

    /// Download prescribed convex hull transforms in reserved rounded-pair order.
    pub fn readback_prescribed_convex_rounded_poses(
        &self,
    ) -> Result<Vec<Vec<Isometry3<f64>>>, GpuArticulatedDynamicsError> {
        Ok(self
            .ground_contact
            .as_ref()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .readback_prescribed_convex_rounded_poses(&self.queue)?)
    }

    /// Configure prescribed external point anchors followed by fixed frames.
    /// Constraints with a second articulated link are excluded from this ordering.
    pub fn update_prescribed_constraint_bodies(
        &mut self,
        bodies: &[Vec<Option<crate::gpu_kinematic_body::GpuKinematicBody>>],
    ) -> Result<(), GpuArticulatedDynamicsError> {
        self.ground_contact
            .as_mut()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .update_prescribed_constraint_bodies(&self.queue, bodies)?;
        self.integrate_external_spheres = true;
        self.clear_sleep_diagnostics();
        Ok(())
    }

    /// Configure prescribed mesh/polyline pairs in packed family order.
    /// Pair order is mesh/sphere, polyline/sphere, mesh/capsule,
    /// polyline/capsule, mesh/box, polyline/box, mesh/axial, polyline/axial,
    /// mesh/convex, polyline/convex; preserve input order within each family.
    pub fn update_prescribed_indexed_bodies(
        &mut self,
        bodies: &[Vec<Option<crate::gpu_kinematic_body::GpuKinematicBody>>],
    ) -> Result<(), GpuArticulatedDynamicsError> {
        self.ground_contact
            .as_mut()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .update_prescribed_indexed_bodies(&self.queue, bodies)?;
        self.integrate_external_spheres = true;
        self.clear_sleep_diagnostics();
        Ok(())
    }

    /// Read the current reserved mesh/polyline geometry poses.
    pub fn readback_prescribed_indexed_poses(
        &self,
    ) -> Result<Vec<Vec<Isometry3<f64>>>, GpuArticulatedDynamicsError> {
        Ok(self
            .ground_contact
            .as_ref()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .readback_prescribed_indexed_poses(&self.queue)?)
    }

    /// Update external anchor points/frames while retaining their prescribed motion.
    pub fn update_external_constraint_frames(
        &mut self,
        frames: &[Vec<Isometry3<f64>>],
    ) -> Result<(), GpuArticulatedDynamicsError> {
        self.ground_contact
            .as_mut()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .update_external_constraint_frames(&self.queue, frames)?;
        self.clear_sleep_diagnostics();
        Ok(())
    }

    /// Update mesh/polyline geometry poses without changing topology or prescribed motion.
    pub fn update_indexed_geometry_poses(
        &mut self,
        frames: &[Vec<Isometry3<f64>>],
    ) -> Result<(), GpuArticulatedDynamicsError> {
        self.ground_contact
            .as_mut()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .update_indexed_geometry_poses(&self.queue, frames)?;
        self.clear_sleep_diagnostics();
        Ok(())
    }

    /// Read external anchor frames in point-then-fixed input order.
    /// The frame translation is the anchor point, not the prescribed body origin.
    pub fn readback_prescribed_constraint_frames(
        &self,
    ) -> Result<Vec<Vec<Isometry3<f64>>>, GpuArticulatedDynamicsError> {
        Ok(self
            .ground_contact
            .as_ref()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .readback_prescribed_constraint_frames(&self.queue)?)
    }

    /// Configure prescribed convex hulls in reserved polyhedron then axial pair order.
    /// None stops at the current pose. Invalid input preserves all buffers.
    pub fn update_prescribed_convex_other_bodies(
        &mut self,
        bodies: &[Vec<Option<crate::gpu_kinematic_body::GpuKinematicBody>>],
    ) -> Result<(), GpuArticulatedDynamicsError> {
        self.ground_contact
            .as_mut()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .update_prescribed_convex_other_bodies(&self.queue, bodies)?;
        self.integrate_external_spheres = true;
        self.clear_sleep_diagnostics();
        Ok(())
    }

    /// Download prescribed convex hull transforms in reserved polyhedron/axial pair order.
    pub fn readback_prescribed_convex_other_poses(
        &self,
    ) -> Result<Vec<Vec<Isometry3<f64>>>, GpuArticulatedDynamicsError> {
        Ok(self
            .ground_contact
            .as_ref()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .readback_prescribed_convex_other_poses(&self.queue)?)
    }

    /// Configure prescribed cylinders and cones, retaining their collision poses.
    pub fn update_prescribed_axial_bodies(
        &mut self,
        kind: crate::gpu_articulated_ground_contact::GpuArticulatedAxialContactKind,
        bodies: &[Vec<Option<crate::gpu_kinematic_body::GpuKinematicBody>>],
    ) -> Result<(), GpuArticulatedDynamicsError> {
        self.ground_contact
            .as_mut()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .update_prescribed_axial_bodies(&self.queue, kind, bodies)?;
        self.integrate_external_spheres = true;
        self.clear_sleep_diagnostics();
        Ok(())
    }

    /// Download world-cylinder and world-cone transforms for the selected reserved contact family.
    pub fn readback_prescribed_axial_poses(
        &self,
        kind: crate::gpu_articulated_ground_contact::GpuArticulatedAxialContactKind,
    ) -> Result<Vec<Vec<Isometry3<f64>>>, GpuArticulatedDynamicsError> {
        Ok(self
            .ground_contact
            .as_ref()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .readback_prescribed_axial_poses(&self.queue, kind)?)
    }

    /// Configure prescribed capsules for a reserved contact family. Each pair
    /// updates all endpoint or manifold rows, retaining its current collision
    /// pose. None stops motion. Invalid packed input leaves all buffers intact.
    pub fn update_prescribed_capsule_bodies(
        &mut self,
        kind: crate::gpu_articulated_ground_contact::GpuArticulatedCapsuleContactKind,
        bodies: &[Vec<Option<crate::gpu_kinematic_body::GpuKinematicBody>>],
    ) -> Result<(), GpuArticulatedDynamicsError> {
        self.ground_contact
            .as_mut()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .update_prescribed_capsule_bodies(&self.queue, kind, bodies)?;
        self.integrate_external_spheres = true;
        self.clear_sleep_diagnostics();
        Ok(())
    }

    /// Download world-capsule endpoints for the selected reserved contact family.
    pub fn readback_prescribed_capsule_endpoints(
        &self,
        kind: crate::gpu_articulated_ground_contact::GpuArticulatedCapsuleContactKind,
    ) -> Result<Vec<Vec<[Vector3<f64>; 2]>>, GpuArticulatedDynamicsError> {
        Ok(self
            .ground_contact
            .as_ref()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .readback_prescribed_capsule_endpoints(&self.queue, kind)?)
    }

    /// Set external sphere motion in environment/static-sphere-pair order.
    /// Center positions are supplied separately unless GPU sphere integration
    /// is enabled. No geometry or mass is rebuilt.
    pub fn update_static_sphere_motion(
        &mut self,
        motions: &[Vec<crate::gpu_articulated_ground_contact::GpuArticulatedSphereMotion>],
    ) -> Result<bool, GpuArticulatedDynamicsError> {
        let changed = self
            .ground_contact
            .as_mut()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .update_static_sphere_motion(&self.queue, motions)?;
        if changed {
            self.clear_sleep_diagnostics();
        }
        Ok(changed)
    }

    /// Set prescribed external sphere motion for capsule/static-sphere contacts.
    pub fn update_static_capsule_sphere_motion(
        &mut self,
        motions: &[Vec<crate::gpu_articulated_ground_contact::GpuArticulatedSphereMotion>],
    ) -> Result<bool, GpuArticulatedDynamicsError> {
        let changed = self
            .ground_contact
            .as_mut()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .update_static_capsule_sphere_motion(&self.queue, motions)?;
        if changed {
            self.clear_sleep_diagnostics();
        }
        Ok(changed)
    }

    /// Set prescribed external sphere motion for box/static-sphere contacts.
    pub fn update_static_box_sphere_motion(
        &mut self,
        motions: &[Vec<crate::gpu_articulated_ground_contact::GpuArticulatedSphereMotion>],
    ) -> Result<bool, GpuArticulatedDynamicsError> {
        let changed = self
            .ground_contact
            .as_mut()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .update_static_box_sphere_motion(&self.queue, motions)?;
        if changed {
            self.clear_sleep_diagnostics();
        }
        Ok(changed)
    }

    /// Set external sphere motion for dynamic cylinder/cone contacts.
    /// Input excludes pairs with a stationary axial shape.
    pub fn update_static_axial_sphere_motion(
        &mut self,
        motions: &[Vec<crate::gpu_articulated_ground_contact::GpuArticulatedSphereMotion>],
    ) -> Result<bool, GpuArticulatedDynamicsError> {
        let changed = self
            .ground_contact
            .as_mut()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .update_static_axial_sphere_motion(&self.queue, motions)?;
        if changed {
            self.clear_sleep_diagnostics();
        }
        Ok(changed)
    }

    /// Set external sphere motion for dynamic convex contacts.
    pub fn update_static_convex_sphere_motion(
        &mut self,
        motions: &[Vec<crate::gpu_articulated_ground_contact::GpuArticulatedSphereMotion>],
    ) -> Result<bool, GpuArticulatedDynamicsError> {
        let changed = self
            .ground_contact
            .as_mut()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .update_static_convex_sphere_motion(&self.queue, motions)?;
        if changed {
            self.clear_sleep_diagnostics();
        }
        Ok(changed)
    }

    /// Move stationary centers in the reserved capsule/static-sphere contact pairs.
    /// Returns whether any represented center changed. Radii and pair ordering
    /// stay fixed. Changed rows wake their owners and clear applied sleep diagnostics.
    /// Invalid input retains all geometry, pending requests and diagnostics.
    pub fn update_static_capsule_sphere_centers(
        &mut self,
        centers: &[Vec<Vector3<f64>>],
    ) -> Result<bool, GpuArticulatedDynamicsError> {
        let changed = self
            .ground_contact
            .as_mut()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .update_static_capsule_sphere_centers(&self.queue, centers)?;
        if changed {
            self.clear_sleep_diagnostics();
        }
        Ok(changed)
    }

    /// Move stationary centers in the reserved box/static-sphere contact pairs.
    /// Returns whether any represented center changed. Radii and pair ordering
    /// stay fixed. Changed rows wake their owners and clear applied sleep diagnostics.
    /// Invalid input retains all geometry, pending requests and diagnostics.
    pub fn update_static_box_sphere_centers(
        &mut self,
        centers: &[Vec<Vector3<f64>>],
    ) -> Result<bool, GpuArticulatedDynamicsError> {
        let changed = self
            .ground_contact
            .as_mut()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .update_static_box_sphere_centers(&self.queue, centers)?;
        if changed {
            self.clear_sleep_diagnostics();
        }
        Ok(changed)
    }

    /// Move stationary centers in the reserved axial/static-sphere contact pairs.
    /// Input omits pairs with axial_is_static == true.
    /// Returns whether any represented center changed. Radii and pair ordering
    /// stay fixed. Changed rows wake their owners and clear applied sleep diagnostics.
    /// Invalid input retains all geometry, pending requests and diagnostics.
    pub fn update_static_axial_sphere_centers(
        &mut self,
        centers: &[Vec<Vector3<f64>>],
    ) -> Result<bool, GpuArticulatedDynamicsError> {
        let changed = self
            .ground_contact
            .as_mut()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .update_static_axial_sphere_centers(&self.queue, centers)?;
        if changed {
            self.clear_sleep_diagnostics();
        }
        Ok(changed)
    }

    /// Move stationary centers in the reserved convex/static-sphere contact pairs.
    /// Returns whether any represented center changed. Radii and pair ordering
    /// stay fixed. Changed rows wake their owners and clear applied sleep diagnostics.
    /// Invalid input retains all geometry, pending requests and diagnostics.
    pub fn update_static_convex_sphere_centers(
        &mut self,
        centers: &[Vec<Vector3<f64>>],
    ) -> Result<bool, GpuArticulatedDynamicsError> {
        let changed = self
            .ground_contact
            .as_mut()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .update_static_convex_sphere_centers(&self.queue, centers)?;
        if changed {
            self.clear_sleep_diagnostics();
        }
        Ok(changed)
    }

    /// Update stationary box transforms reserved for sphere/box contacts.
    /// Changed poses wake owners and clear applied sleep diagnostics. Invalid
    /// inputs preserve geometry and diagnostics. Box dimensions stay fixed.
    pub fn update_static_sphere_box_poses(
        &mut self,
        poses: &[Vec<Isometry3<f64>>],
    ) -> Result<bool, GpuArticulatedDynamicsError> {
        let changed = self
            .ground_contact
            .as_mut()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .update_static_sphere_box_poses(&self.queue, poses)?;
        if changed {
            self.clear_sleep_diagnostics();
        }
        Ok(changed)
    }

    /// Update stationary box transforms reserved for axial/box contacts.
    /// Input omits pairs with axial_is_static == true.
    /// Changed poses wake owners and clear applied sleep diagnostics. Invalid
    /// inputs preserve geometry and diagnostics. Box dimensions stay fixed.
    pub fn update_static_axial_box_poses(
        &mut self,
        poses: &[Vec<Isometry3<f64>>],
    ) -> Result<bool, GpuArticulatedDynamicsError> {
        let changed = self
            .ground_contact
            .as_mut()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .update_static_axial_box_poses(&self.queue, poses)?;
        if changed {
            self.clear_sleep_diagnostics();
        }
        Ok(changed)
    }

    /// Update stationary box transforms reserved for capsule/box contacts.
    /// Changed poses wake owners and clear applied sleep diagnostics. Invalid
    /// inputs preserve geometry and diagnostics. Box dimensions stay fixed.
    pub fn update_static_capsule_box_poses(
        &mut self,
        poses: &[Vec<Isometry3<f64>>],
    ) -> Result<bool, GpuArticulatedDynamicsError> {
        let changed = self
            .ground_contact
            .as_mut()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .update_static_capsule_box_poses(&self.queue, poses)?;
        if changed {
            self.clear_sleep_diagnostics();
        }
        Ok(changed)
    }

    /// Update stationary box transforms reserved for box/box contacts.
    /// Changed poses wake owners and clear applied sleep diagnostics. Invalid
    /// inputs preserve geometry and diagnostics. Box dimensions stay fixed.
    pub fn update_static_box_box_poses(
        &mut self,
        poses: &[Vec<Isometry3<f64>>],
    ) -> Result<bool, GpuArticulatedDynamicsError> {
        let changed = self
            .ground_contact
            .as_mut()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .update_static_box_box_poses(&self.queue, poses)?;
        if changed {
            self.clear_sleep_diagnostics();
        }
        Ok(changed)
    }

    /// Update stationary convex transforms reserved for static convex pair contacts.
    /// Changed poses wake owners and clear applied sleep diagnostics. Invalid
    /// inputs preserve geometry and diagnostics. Convex geometry stay fixed.
    pub fn update_static_convex_pair_poses(
        &mut self,
        poses: &[Vec<Isometry3<f64>>],
    ) -> Result<bool, GpuArticulatedDynamicsError> {
        let changed = self
            .ground_contact
            .as_mut()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .update_static_convex_pair_poses(&self.queue, poses)?;
        if changed {
            self.clear_sleep_diagnostics();
        }
        Ok(changed)
    }

    /// Update stationary convex transforms reserved for static axial-convex contacts.
    /// Changed poses wake owners and clear applied sleep diagnostics. Invalid
    /// inputs preserve geometry and diagnostics. Convex geometry stays fixed.
    pub fn update_static_axial_convex_poses(
        &mut self,
        poses: &[Vec<Isometry3<f64>>],
    ) -> Result<bool, GpuArticulatedDynamicsError> {
        let changed = self
            .ground_contact
            .as_mut()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .update_static_axial_convex_poses(&self.queue, poses)?;
        if changed {
            self.clear_sleep_diagnostics();
        }
        Ok(changed)
    }

    /// Update stationary convex transforms reserved for scene convex sphere then capsule contacts.
    /// Changed poses wake owners and clear applied sleep diagnostics. Invalid
    /// inputs preserve geometry and diagnostics. Convex geometry stays fixed.
    pub fn update_scene_convex_rounded_poses(
        &mut self,
        poses: &[Vec<Isometry3<f64>>],
    ) -> Result<bool, GpuArticulatedDynamicsError> {
        let changed = self
            .ground_contact
            .as_mut()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .update_scene_convex_rounded_poses(&self.queue, poses)?;
        if changed {
            self.clear_sleep_diagnostics();
        }
        Ok(changed)
    }

    /// Set a root pose for subsequent steps without changing its velocity.
    pub fn set_root_pose(
        &self,
        environment: usize,
        root: Isometry3<f64>,
    ) -> Result<(), GpuArticulatedDynamicsError> {
        self.poses.set_root_pose(environment, root)?;
        self.velocity_bias.set_root_pose(environment, root)?;
        if let Some(contact) = &self.ground_contact {
            contact.clear_cached_impulses(&self.queue);
        }
        self.clear_sleep_diagnostics();
        Ok(())
    }

    /// Reset generalized states and clear faults, retaining separate root poses.
    /// Use `set_root_pose` as well when resetting a floating environment's pose.
    /// Restore generalized state and independent orientations after validating both.
    /// Invalid input preserves coordinates, velocities, poses, fault flags and caches.
    /// Quaternion registration order and velocity slots must match this batch.
    /// Separate floating-root poses are unchanged; update those with `set_root_pose`.
    pub fn reset_with_spherical_state(
        &self,
        states: &[GpuGeneralizedState],
        joints: &[Vec<GpuSphericalJointState>],
    ) -> Result<(), GpuArticulatedDynamicsError> {
        let spherical = self
            .spherical
            .as_ref()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?;
        self.state.validate_reset(states)?;
        spherical.validate_reset(joints)?;
        self.reset(states)?;
        spherical.reset(joints)?;
        Ok(())
    }

    /// Restore generalized coordinates and clear contact caches; quaternion poses are separate.
    pub fn reset(&self, states: &[GpuGeneralizedState]) -> Result<(), GpuArticulatedDynamicsError> {
        self.state.reset(states)?;
        if let Some(targets) = &self.motor_targets {
            targets.cancel(&self.queue);
        }
        if let Some(contact) = &self.ground_contact {
            contact.clear_cached_impulses(&self.queue);
        }
        self.clear_sleep_diagnostics();
        Ok(())
    }

    /// Download the current generalized states.
    pub fn readback(&self) -> Result<Vec<GpuGeneralizedState>, GpuArticulatedDynamicsError> {
        Ok(self.state.readback()?)
    }

    /// Enable optional per-link GPU contact-impulse reduction before the next step.
    /// A batch without contact rows cannot enable this pass.
    pub fn enable_contact_activity(&mut self) -> Result<(), GpuArticulatedDynamicsError> {
        self.ground_contact
            .as_mut()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .enable_contact_activity()?;
        Ok(())
    }

    /// Create an explicitly sampled resident normal-impulse sensor for selected links.
    /// Encode the returned sensor after the desired solved substep on the same encoder.
    ///
    /// # Errors
    /// Returns `InvalidInput` without contact rows, or propagates invalid link selection/capacity.
    pub fn normal_impulse_sensor(
        &self,
        links: &[usize],
    ) -> Result<crate::gpu_contact_sensor::GpuContactImpulseSensor, GpuArticulatedDynamicsError>
    {
        Ok(self
            .ground_contact
            .as_ref()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .normal_impulse_sensor(links)?)
    }

    /// Update resident contact correction and warm-start policy without reallocating rows.
    ///
    /// # Errors
    /// Returns `InvalidInput` without contact rows, or propagates invalid policy values.
    pub fn update_contact_policy(
        &self,
        policy: crate::gpu_contact_policy::GpuContactPolicy,
    ) -> Result<(), GpuArticulatedDynamicsError> {
        Ok(self
            .ground_contact
            .as_ref()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .update_contact_policy(&self.queue, policy)?)
    }

    /// Download solved-impulse activity per environment and stable link index.
    /// Requires `enable_contact_activity`; bilateral and zero-impulse rows are excluded.
    pub fn readback_contact_activity(&self) -> Result<Vec<Vec<bool>>, GpuArticulatedDynamicsError> {
        Ok(self
            .ground_contact
            .as_ref()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .readback_contact_activity(&self.queue)?)
    }

    /// Initialize mobility from the batch's topology, mimics and scalar couplings.
    /// Requires activity enabled. This clears existing activity, wake and idle history.
    /// Bilateral constraints and contact may connect these groups further each step.
    pub fn update_contact_mobility_from_topology(
        &mut self,
    ) -> Result<(), GpuArticulatedDynamicsError> {
        self.ground_contact
            .as_mut()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .update_contact_mobility_groups(&self.queue, &self.topology_mobility)?;
        self.clear_sleep_diagnostics();
        Ok(())
    }

    /// Set mobility for sleep, rejecting splits of topology/mimic/coupling groups.
    /// Coarser groups are allowed. All environments are checked before any upload.
    /// Invalid input retains existing labels, wake requests and idle history.
    pub fn update_contact_sleep_mobility_groups(
        &mut self,
        groups: &[Vec<Vec<usize>>],
    ) -> Result<(), GpuArticulatedDynamicsError> {
        if groups.len() != self.topology_mobility.len() {
            return Err(GpuArticulatedDynamicsError::InvalidInput);
        }
        for ((required, supplied), links) in self
            .topology_mobility
            .iter()
            .zip(groups)
            .zip(self.poses.link_ranges())
        {
            crate::articulation::validate_mobility_partition(links.len(), required, supplied)?;
        }
        self.update_contact_mobility_groups(groups)
    }

    /// Supply disjoint articulation mobility groups in each environment's local links.
    /// Requires activity enabled; successful updates clear previous activity/labels.
    pub fn update_contact_mobility_groups(
        &mut self,
        groups: &[Vec<Vec<usize>>],
    ) -> Result<(), GpuArticulatedDynamicsError> {
        if self.sleep_freeze.is_some() {
            if groups.len() != self.topology_mobility.len() {
                return Err(GpuArticulatedDynamicsError::InvalidInput);
            }
            for ((required, supplied), links) in self
                .topology_mobility
                .iter()
                .zip(groups)
                .zip(self.poses.link_ranges())
            {
                crate::articulation::validate_mobility_partition(links.len(), required, supplied)?;
            }
        }
        self.ground_contact
            .as_mut()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .update_contact_mobility_groups(&self.queue, groups)?;
        self.clear_sleep_diagnostics();
        Ok(())
    }

    /// Download contact/bilateral/mobility component labels. Unspecified joints
    /// and scalar couplings must be combined before treating these as sleep islands.
    pub fn readback_contact_components(
        &self,
    ) -> Result<Vec<Vec<usize>>, GpuArticulatedDynamicsError> {
        Ok(self
            .ground_contact
            .as_ref()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .readback_contact_components(&self.queue)?)
    }

    /// Configure one-step wake when a link loses its last geometric contact.
    /// Requires activity enabled. Reset clears geometry history and retains this setting.
    pub fn update_contact_loss_wake(
        &self,
        enabled: bool,
    ) -> Result<(), GpuArticulatedDynamicsError> {
        if self.sleep_freeze.is_some() && !enabled {
            return Err(GpuArticulatedDynamicsError::InvalidInput);
        }
        self.ground_contact
            .as_ref()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .update_contact_loss_wake(&self.queue, enabled)?;
        Ok(())
    }

    /// Download touching/penetrating contact presence, including zero-impulse rows.
    pub fn readback_geometric_contacts(
        &self,
    ) -> Result<Vec<Vec<bool>>, GpuArticulatedDynamicsError> {
        Ok(self
            .ground_contact
            .as_ref()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .readback_geometric_contacts(&self.queue)?)
    }

    /// Download one-sided geometric contacts against external static geometry.
    /// This is not a gravity support test.
    pub fn readback_external_static_contacts(
        &self,
    ) -> Result<Vec<Vec<bool>>, GpuArticulatedDynamicsError> {
        Ok(self
            .ground_contact
            .as_ref()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .readback_external_static_contacts(&self.queue)?)
    }

    /// Enable static-contact normal tests against the resident environment gravity.
    /// Requires activity enabled. Gravity updates are observed without CPU readback.
    pub fn enable_contact_gravity_support(&mut self) -> Result<(), GpuArticulatedDynamicsError> {
        self.ground_contact
            .as_mut()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .enable_contact_gravity_support(self.forces.gravity_buffer())?;
        Ok(())
    }

    /// Download static touching contacts whose signed normal opposes gravity.
    pub fn readback_contact_gravity_support(
        &self,
    ) -> Result<Vec<Vec<bool>>, GpuArticulatedDynamicsError> {
        Ok(self
            .ground_contact
            .as_ref()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .readback_contact_gravity_support(&self.queue)?)
    }

    /// Download bilateral point/fixed links anchored directly to world.
    /// A point anchor alone does not constrain every rotation.
    pub fn readback_contact_world_anchors(
        &self,
    ) -> Result<Vec<Vec<bool>>, GpuArticulatedDynamicsError> {
        Ok(self
            .ground_contact
            .as_ref()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .readback_contact_world_anchors(&self.queue)?)
    }

    /// Download full fixed world anchors, excluding point anchors.
    pub fn readback_contact_fixed_world_anchors(
        &self,
    ) -> Result<Vec<Vec<bool>>, GpuArticulatedDynamicsError> {
        Ok(self
            .ground_contact
            .as_ref()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .readback_contact_fixed_world_anchors(&self.queue)?)
    }

    /// Enable automatic wake from persistent world force/torque loads, including
    /// loads on massless frames. Gravity is excluded. Requires activity enabled.
    /// Reset retains loads and this pass, so nonzero loads generate new requests.
    pub fn enable_contact_load_wake(&mut self) -> Result<(), GpuArticulatedDynamicsError> {
        self.ground_contact
            .as_mut()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .enable_contact_load_wake(self.forces.link_load_buffer())?;
        Ok(())
    }

    /// Enable one-step wake when an environment's represented gravity changes.
    /// Enabling captures the current gravity; repeating an identical update does not wake.
    /// Requires activity enabled. Reset retains the last observed gravity.
    pub fn enable_contact_gravity_wake(&mut self) -> Result<(), GpuArticulatedDynamicsError> {
        self.ground_contact
            .as_mut()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .enable_contact_gravity_wake(&self.queue, self.forces.gravity_buffer())?;
        Ok(())
    }

    /// Enable persistent wake from evaluated efforts, motors and passive laws.
    /// Tangent Jacobians select affected physical links before component propagation.
    /// Gravity and external link loads are handled separately. Requires activity.
    pub fn enable_contact_actuation_wake(&mut self) -> Result<(), GpuArticulatedDynamicsError> {
        if self.ground_contact.is_none() {
            return Err(GpuArticulatedDynamicsError::InvalidInput);
        }
        let efforts = self.actuation_efforts.get_or_insert_with(|| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Tessera drive efforts before velocity bias"),
                size: self.forces.base_force_buffer().size(),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        self.ground_contact
            .as_mut()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .enable_contact_actuation_wake(efforts)?;
        Ok(())
    }

    /// Configure GPU component idle tracking and motion wake thresholds.
    /// Candidates stop integration only when enable_contact_sleep has succeeded.
    /// Requires activity enabled. Persistent load wake should also be enabled.
    pub fn update_contact_idle_settings(
        &mut self,
        settings: &[Option<crate::sleep::SleepSettings>],
    ) -> Result<(), GpuArticulatedDynamicsError> {
        self.ground_contact
            .as_mut()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .update_contact_idle_settings(&self.queue, settings)?;
        self.clear_sleep_diagnostics();
        Ok(())
    }

    /// Set per-link idle and predicted-motion thresholds together.
    /// Invalid dimensions or settings preserve history and applied diagnostics.
    /// Valid changes clear idle history and applied coordinate freeze diagnostics.
    pub fn update_contact_link_idle_settings(
        &mut self,
        settings: &[Vec<Option<crate::sleep::SleepSettings>>],
    ) -> Result<(), GpuArticulatedDynamicsError> {
        self.ground_contact
            .as_mut()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .update_contact_link_idle_settings(&self.queue, settings)?;
        self.clear_sleep_diagnostics();
        Ok(())
    }

    /// Require gravity-opposing static contact for idle candidates, enabling its evaluation.
    /// False restores geometric-contact-only tracking. Policy changes clear idle history.
    /// This is a component support-direction criterion, not a full equilibrium proof.
    pub fn update_contact_idle_requires_static_support(
        &mut self,
        required: bool,
    ) -> Result<(), GpuArticulatedDynamicsError> {
        if self.sleep_freeze.is_some() && !required {
            return Err(GpuArticulatedDynamicsError::InvalidInput);
        }
        if required {
            self.enable_contact_gravity_support()?;
        }
        self.ground_contact
            .as_ref()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .update_contact_idle_requires_static_support(&self.queue, required)?;
        self.clear_sleep_diagnostics();
        Ok(())
    }

    /// Download coordinate freezes applied during the last completed step.
    /// This is distinct from per-link candidates. Reset and configuration changes
    /// clear these diagnostics; disabled automatic sleeping returns an error.
    /// Serialize submissions and readback externally; this is not a queue snapshot.
    pub fn readback_sleeping_coordinates(
        &self,
    ) -> Result<Vec<Vec<bool>>, GpuArticulatedDynamicsError> {
        let freeze = self
            .sleep_freeze
            .as_ref()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?;
        // Reject stale diagnostics from any source state or mass fault.
        let _ = self.readback_contact_sleep_candidates()?;
        let bytes = crate::gpu_articulated_mass::read_buffer(
            &self.device,
            &self.queue,
            freeze.frozen_buffer(),
        )?;
        let flags = bytes
            .chunks_exact(4)
            .map(|v| u32::from_ne_bytes([v[0], v[1], v[2], v[3]]) != 0)
            .collect::<Vec<_>>();
        Ok(self
            .state
            .ranges()
            .iter()
            .map(|range| flags[range.clone()].to_vec())
            .collect())
    }

    /// Download component sleep candidates, not an applied sleeping state.
    pub fn readback_contact_sleep_candidates(
        &self,
    ) -> Result<Vec<Vec<bool>>, GpuArticulatedDynamicsError> {
        Ok(self
            .ground_contact
            .as_ref()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .readback_contact_sleep_candidates(&self.queue)?)
    }

    /// Configure GPU predicted-motion wake at positive-mass link origins.
    /// Requires activity enabled; None disables the corresponding environment.
    /// This generates requests and does not update idle time or sleeping state.
    /// Rejected while automatic sleeping is active; use update_contact_idle_settings
    /// to change matching idle and motion thresholds together.
    pub fn update_contact_motion_wake(
        &mut self,
        settings: &[Option<crate::sleep::SleepSettings>],
    ) -> Result<(), GpuArticulatedDynamicsError> {
        // Active sleeping shares motion thresholds with idle tracking.
        // Configure both through update_contact_idle_settings instead.
        if self.sleep_freeze.is_some() {
            return Err(GpuArticulatedDynamicsError::InvalidInput);
        }
        self.ground_contact
            .as_mut()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .update_contact_motion_wake(&self.queue, settings)?;
        Ok(())
    }

    /// Replace one-step wake requests in environment and stable link order.
    /// Requires activity enabled; requests are consumed by the next step.
    pub fn update_contact_wake_requests(
        &self,
        requests: &[Vec<bool>],
    ) -> Result<(), GpuArticulatedDynamicsError> {
        self.ground_contact
            .as_ref()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .update_contact_wake_requests(&self.queue, requests)?;
        Ok(())
    }

    /// Download propagated one-step wake requests, not actual sleeping state.
    pub fn readback_contact_wake_requests(
        &self,
    ) -> Result<Vec<Vec<bool>>, GpuArticulatedDynamicsError> {
        Ok(self
            .ground_contact
            .as_ref()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .readback_contact_wake_requests(&self.queue)?)
    }

    /// Set rigid sphere offsets about prescribed body origins for sphere pairs.
    /// Set angular velocity first with update_static_sphere_motion. None stops
    /// the corresponding sphere. GPU center integration must be enabled.
    pub fn update_static_sphere_orbits(
        &mut self,
        orbits: &[Vec<Option<crate::gpu_articulated_ground_contact::GpuArticulatedSphereOrbit>>],
    ) -> Result<bool, GpuArticulatedDynamicsError> {
        let changed = self
            .ground_contact
            .as_mut()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .update_static_sphere_orbits(&self.queue, orbits)?;
        if changed {
            self.clear_sleep_diagnostics();
        }
        Ok(changed)
    }

    /// Set rigid sphere offsets about prescribed body origins for capsule pairs.
    pub fn update_static_capsule_sphere_orbits(
        &mut self,
        orbits: &[Vec<Option<crate::gpu_articulated_ground_contact::GpuArticulatedSphereOrbit>>],
    ) -> Result<bool, GpuArticulatedDynamicsError> {
        let changed = self
            .ground_contact
            .as_mut()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .update_static_capsule_sphere_orbits(&self.queue, orbits)?;
        if changed {
            self.clear_sleep_diagnostics();
        }
        Ok(changed)
    }

    /// Set rigid sphere offsets about prescribed body origins for box pairs.
    pub fn update_static_box_sphere_orbits(
        &mut self,
        orbits: &[Vec<Option<crate::gpu_articulated_ground_contact::GpuArticulatedSphereOrbit>>],
    ) -> Result<bool, GpuArticulatedDynamicsError> {
        let changed = self
            .ground_contact
            .as_mut()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .update_static_box_sphere_orbits(&self.queue, orbits)?;
        if changed {
            self.clear_sleep_diagnostics();
        }
        Ok(changed)
    }

    /// Set rigid sphere offsets about prescribed body origins for axial pairs.
    pub fn update_static_axial_sphere_orbits(
        &mut self,
        orbits: &[Vec<Option<crate::gpu_articulated_ground_contact::GpuArticulatedSphereOrbit>>],
    ) -> Result<bool, GpuArticulatedDynamicsError> {
        let changed = self
            .ground_contact
            .as_mut()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .update_static_axial_sphere_orbits(&self.queue, orbits)?;
        if changed {
            self.clear_sleep_diagnostics();
        }
        Ok(changed)
    }

    /// Set rigid sphere offsets about prescribed body origins for convex pairs.
    pub fn update_static_convex_sphere_orbits(
        &mut self,
        orbits: &[Vec<Option<crate::gpu_articulated_ground_contact::GpuArticulatedSphereOrbit>>],
    ) -> Result<bool, GpuArticulatedDynamicsError> {
        let changed = self
            .ground_contact
            .as_mut()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .update_static_convex_sphere_orbits(&self.queue, orbits)?;
        if changed {
            self.clear_sleep_diagnostics();
        }
        Ok(changed)
    }

    /// Replace prescribed motion for all external sphere contact categories.
    /// All categories are validated before uploading any change.
    pub fn update_external_sphere_motions(
        &mut self,
        motions: &crate::gpu_articulated_ground_contact::GpuArticulatedExternalSphereMotions,
    ) -> Result<bool, GpuArticulatedDynamicsError> {
        let changed = self
            .ground_contact
            .as_mut()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .update_external_sphere_motions(&self.queue, motions)?;
        if changed {
            self.clear_sleep_diagnostics();
        }
        Ok(changed)
    }

    /// Download GPU-integrated body origins in environment and contact pair order.
    /// None denotes center-based motion without a tracked body origin.
    /// Rejects faulted source environments. Serialize queue use externally.
    pub fn readback_external_sphere_orbits(
        &self,
    ) -> Result<
        crate::gpu_articulated_ground_contact::GpuArticulatedExternalSphereOrbits,
        GpuArticulatedDynamicsError,
    > {
        Ok(self
            .ground_contact
            .as_ref()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .readback_external_sphere_orbits(&self.queue)?)
    }

    /// Download external sphere centers in environment and contact pair order.
    /// Includes GPU-integrated motion and rejects faulted source environments.
    /// Serialize submissions and readback externally.
    pub fn readback_external_sphere_centers(
        &self,
    ) -> Result<
        crate::gpu_articulated_ground_contact::GpuArticulatedExternalSphereCenters,
        GpuArticulatedDynamicsError,
    > {
        Ok(self
            .ground_contact
            .as_ref()
            .ok_or(GpuArticulatedDynamicsError::InvalidInput)?
            .readback_external_sphere_centers(&self.queue)?)
    }

    /// Download per-link contact wrenches from the last solved step.
    pub fn readback_contacts(
        &self,
    ) -> Result<
        Vec<Vec<crate::gpu_articulated_ground_contact::GpuArticulatedLinkContact>>,
        GpuArticulatedDynamicsError,
    > {
        match &self.ground_contact {
            Some(contact) => Ok(contact.readback_contacts(&self.queue)?),
            None => Ok(vec![Vec::new(); self.poses.link_ranges().len()]),
        }
    }

    /// Download states, root poses, and diagnostics together in environment order.
    /// Serialize submissions and readback externally: this performs multiple
    /// buffer copies and is not a concurrent queue snapshot.
    pub fn readback_output(
        &self,
    ) -> Result<Vec<GpuArticulatedDynamicsOutput>, GpuArticulatedDynamicsError> {
        let states = self.readback()?;
        let roots = self.readback_root_poses()?;
        let contacts = self.readback_contacts()?;
        let spherical_orientations = self
            .spherical
            .as_ref()
            .map(|batch| batch.readback())
            .transpose()?;
        Ok(states
            .into_iter()
            .zip(roots)
            .zip(contacts)
            .enumerate()
            .map(
                |(environment, ((state, root_pose), contacts))| GpuArticulatedDynamicsOutput {
                    state,
                    root_pose,
                    contacts,
                    floating_root: self.poses.floating_roots()[environment],
                    spherical_joints: spherical_orientations.as_ref().map(|orientations| {
                        orientations[environment]
                            .iter()
                            .zip(
                                self.spherical
                                    .as_ref()
                                    .map(|s| s.velocity_slots()[environment].as_slice())
                                    .unwrap_or(&[]),
                            )
                            .map(|(&orientation, &velocity_slot)| GpuSphericalJointState {
                                velocity_slot,
                                orientation,
                            })
                            .collect()
                    }),
                },
            )
            .collect())
    }

    /// Download separate root poses in environment order without recomputing FK.
    pub fn readback_root_poses(&self) -> Result<Vec<Isometry3<f64>>, GpuArticulatedDynamicsError> {
        Ok(self.poses.readback_roots()?)
    }

    /// Recompute and download link poses for the current generalized states.
    pub fn readback_link_poses(
        &self,
    ) -> Result<Vec<Vec<Isometry3<f64>>>, GpuArticulatedDynamicsError> {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.poses.encode(&mut encoder);
        let _ = self.queue.submit(Some(encoder.finish()));
        Ok(self.poses.readback()?)
    }
}

fn drives_within_bounds(drives: &[GpuJointForceInput], bounds: &[Option<(f64, f64)>]) -> bool {
    drives.len() == bounds.len()
        && drives.iter().zip(bounds).all(|(drive, bound)| {
            !drive.motor.is_some_and(|motor| {
                motor.position_target.is_some_and(|target| {
                    bound.is_some_and(|(lower, upper)| target < lower || target > upper)
                })
            })
        })
}

#[cfg(test)]
fn same_sphere_box_geometry(
    a: &GpuArticulatedSphereBoxPair,
    b: &GpuArticulatedSphereBoxPair,
) -> bool {
    a.sphere_link == b.sphere_link
        && a.sphere_local_center == b.sphere_local_center
        && a.sphere_radius == b.sphere_radius
        && a.box_link == b.box_link
        && a.box_local_pose == b.box_local_pose
        && a.box_half_extents == b.box_half_extents
}

fn spherical_baseline_only(joints: &[GpuJointForceInput], slots: &[usize]) -> bool {
    slots.iter().all(|&slot| {
        slot.checked_add(3)
            .and_then(|end| joints.get(slot..end))
            .is_some_and(|drives| {
                drives.iter().all(|drive| {
                    drive.motor.is_none()
                        && drive.passive == Default::default()
                        && drive.nonlinear == Default::default()
                })
            })
    })
}

#[cfg(test)]
#[path = "gpu_motor_target_tests.rs"]
mod motor_target_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::articulated_world::{JointMotor, JointNonlinearPassive, JointPassive};
    use crate::articulation::{JointKind, JointSpec, LinkSpec};
    use nalgebra::Matrix3;

    #[test]
    fn composed_scene_ball_and_fixed_constraints_exchange_gpu_momentum() {
        use crate::articulated_world::{
            ArticulatedWorld, ArticulatedWorldParams, LinkSceneFixedConstraint,
            LinkScenePointConstraint, SceneBody, SceneCollider,
        };
        let pose = Isometry3::translation(0.3, -0.2, 1.0)
            * Isometry3::rotation(Vector3::new(0.2, -0.3, 0.4));
        let properties = |mass| LinkSpec {
            mass,
            center_of_mass: Vector3::zeros(),
            inertia: Matrix3::identity() * (mass * 0.2),
        };
        let make_world = |fixed| {
            let robot = Articulation::new(vec![properties(2.0)], Vec::new(), 0).unwrap();
            let mut world = ArticulatedWorld::new(
                robot,
                pose,
                Vec::new(),
                ArticulatedWorldParams {
                    gravity: [0.0; 3],
                    ..Default::default()
                },
            )
            .unwrap();
            world.floating = true;
            world.joint_velocity_limit = Some(0.1);
            world.base_linear_velocity = Vector3::x();
            world.base_angular_velocity = Vector3::z();
            let mut body = SceneBody::new(
                pose,
                3.0,
                properties(3.0).inertia,
                vec![SceneCollider::Sphere {
                    center: Vector3::zeros(),
                    radius: 0.1,
                }],
            )
            .unwrap();
            body.linear_velocity = -Vector3::x();
            body.angular_velocity = -Vector3::z();
            let slot = world.add_scene_body(body);
            if fixed {
                world
                    .set_link_scene_fixed_constraints(vec![LinkSceneFixedConstraint {
                        link: 0,
                        link_frame: Isometry3::identity(),
                        body: slot,
                        body_frame: Isometry3::identity(),
                    }])
                    .unwrap();
            } else {
                world
                    .set_link_scene_point_constraints(vec![LinkScenePointConstraint {
                        link: 0,
                        link_point: [0.0; 3],
                        body: slot,
                        body_point: [0.0; 3],
                    }])
                    .unwrap();
            }
            world
        };
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let mut worlds = [make_world(false), make_world(true)];
            let sources = worlds
                .iter()
                .map(|world| world.compose_scene_dynamics().unwrap())
                .collect::<Vec<_>>();
            let inputs = sources
                .iter()
                .map(|source| {
                    let mut input = source.gpu_dynamics_input(&[]).unwrap();
                    input.contact_iterations = 16;
                    input
                })
                .collect::<Vec<_>>();
            let orientations = sources
                .iter()
                .map(|source| source.scene.gpu_spherical_state(&source.state).unwrap())
                .collect::<Vec<_>>();
            let drives = orientations
                .iter()
                .map(|joints| vec![None; joints.len()])
                .collect::<Vec<_>>();
            let batch = GpuArticulatedDynamicsBatch::new_with_spherical_state(
                &context,
                &inputs,
                0.001,
                &[false, false],
                &orientations,
                &drives,
            )
            .unwrap();
            batch.submit_steps(20).unwrap();
            let outputs = batch.readback_output().unwrap();
            for (index, world) in worlds.iter_mut().enumerate() {
                let _ = world
                    .apply_scene_gpu_output(&sources[index], &outputs[index])
                    .unwrap();
                let body = &world.scene_bodies[0];
                assert!(
                    (world.base_linear_velocity + Vector3::x() * 0.2).norm() < 3e-4,
                    "{backend:?} environment {index}: {:?}",
                    world.base_linear_velocity
                );
                assert!((body.linear_velocity + Vector3::x() * 0.2).norm() < 3e-4);
                assert!(
                    (world.root_pose.translation.vector - body.pose.translation.vector).norm()
                        < 2e-5
                );
                let expected_omega = if index == 0 {
                    Vector3::z()
                } else {
                    -Vector3::z() * 0.2
                };
                let expected_body_omega = if index == 0 {
                    -Vector3::z()
                } else {
                    expected_omega
                };
                assert!((world.base_angular_velocity - expected_omega).norm() < 3e-4);
                assert!((body.angular_velocity - expected_body_omega).norm() < 3e-4);
                let momentum = world.base_linear_velocity * 2.0 + body.linear_velocity * 3.0;
                assert!((momentum + Vector3::x()).norm() < 5e-4);
                let angular = world
                    .root_pose
                    .translation
                    .vector
                    .cross(&(world.base_linear_velocity * 2.0))
                    + body
                        .pose
                        .translation
                        .vector
                        .cross(&(body.linear_velocity * 3.0))
                    + world.base_angular_velocity * 0.4
                    + body.angular_velocity * 0.6;
                let initial = pose.translation.vector.cross(&(-Vector3::x())) - Vector3::z() * 0.2;
                assert!((angular - initial).norm() < 8e-4);
                if index == 1 {
                    assert!(
                        (world.root_pose.rotation.inverse() * body.pose.rotation).angle() < 2e-5
                    );
                }
            }
            eprintln!("composed scene ball/fixed constraints passed on {backend:?}");
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn composed_robot_scene_contact_transfers_reaction_and_conserves_momentum() {
        use crate::articulation::SceneFreeBodyState;
        let properties = |mass| LinkSpec {
            mass,
            center_of_mass: Vector3::zeros(),
            inertia: Matrix3::from_diagonal(&Vector3::new(0.4, 0.6, 0.8)),
        };
        let robot = Articulation::new(vec![properties(2.0)], vec![], 0).unwrap();
        let robot_state = SceneFreeBodyState {
            pose: Isometry3::translation(0.0, 0.0, 2.0)
                * Isometry3::rotation(Vector3::new(0.2, -0.3, 0.4)),
            linear_velocity: Vector3::x(),
            angular_velocity: Vector3::zeros(),
        };
        let body_state = SceneFreeBodyState {
            pose: Isometry3::translation(1.0, 0.0, 2.0)
                * Isometry3::rotation(Vector3::new(-0.3, 0.2, 0.5)),
            linear_velocity: -Vector3::x(),
            angular_velocity: Vector3::zeros(),
        };
        use crate::articulated_world::{
            ArticulatedWorld, ArticulatedWorldParams, LinkSphere, SceneBody, SceneCollider,
        };
        let make_world = || {
            let mut world = ArticulatedWorld::new(
                robot.clone(),
                robot_state.pose,
                vec![LinkSphere {
                    link: 0,
                    center: Vector3::zeros(),
                    radius: 0.5,
                }],
                ArticulatedWorldParams {
                    gravity: [0.0; 3],
                    restitution: 1.0,
                    friction: 0.0,
                    ..Default::default()
                },
            )
            .unwrap();
            world.floating = true;
            world.base_linear_velocity = robot_state.linear_velocity;
            let mut body = SceneBody::new(
                body_state.pose,
                3.0,
                properties(3.0).inertia,
                vec![SceneCollider::Sphere {
                    center: Vector3::zeros(),
                    radius: 0.5,
                }],
            )
            .unwrap();
            body.linear_velocity = body_state.linear_velocity;
            let _ = world.add_scene_body(body);
            world
        };
        let template = make_world();
        let source = template.compose_scene_dynamics().unwrap();
        let scene = &source.scene;
        let spherical = scene.gpu_spherical_state(&source.state).unwrap();
        let robot_layout = scene.robot_root.as_ref().unwrap();
        let body_layout = &scene.bodies[0];
        let input = template.gpu_scene_dynamics_input(&source, &[]).unwrap();
        let drives = vec![None; spherical.len()];
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let mut batch = GpuArticulatedDynamicsBatch::new_with_spherical_state(
                &context,
                core::slice::from_ref(&input),
                0.001,
                &[false],
                core::slice::from_ref(&spherical),
                core::slice::from_ref(&drives),
            )
            .unwrap();
            batch.enable_contact_activity().unwrap();
            let mobility = scene.contact_mobility_groups().unwrap();
            batch.update_contact_mobility_from_topology().unwrap();
            let freeze = batch.build_contact_sleep_freeze().unwrap();
            assert_eq!(
                freeze.frozen_buffer().size(),
                batch.state.velocity_buffer().size()
            );
            batch
                .update_contact_sleep_mobility_groups(core::slice::from_ref(&mobility))
                .unwrap();
            let mut wake = vec![false; scene.articulation.link_count()];
            wake[body_layout.link] = true;
            batch.update_contact_wake_requests(&[wake]).unwrap();
            assert!(batch.update_contact_wake_requests(&[Vec::new()]).is_err());
            batch.submit_steps(1).unwrap();
            let components = batch.readback_contact_components().unwrap();
            assert_eq!(
                components[0][robot_layout.link],
                components[0][body_layout.link]
            );
            let mut expected = (0..scene.articulation.link_count()).collect::<Vec<_>>();
            for &link in mobility.iter().flatten() {
                expected[link] = robot_layout.link.min(body_layout.link);
            }
            assert_eq!(components[0], expected);
            let mut invalid_groups = mobility.clone();
            let duplicate_link = invalid_groups[0][0];
            invalid_groups[0].push(duplicate_link);
            assert!(
                batch
                    .update_contact_mobility_groups(&[invalid_groups])
                    .is_err()
            );
            assert_eq!(batch.readback_contact_components().unwrap(), components);
            let output = batch.readback_output().unwrap();
            let wake = batch.readback_contact_wake_requests().unwrap();
            for (link, &requested) in wake[0].iter().enumerate() {
                assert_eq!(
                    requested,
                    expected[link] == robot_layout.link.min(body_layout.link)
                );
            }
            let result = &output[0];
            // Apply the actual device result through the world bridge, not just raw slots.
            let mut world = make_world();
            let diagnostics = world.apply_scene_gpu_output(&source, result).unwrap();
            assert!((world.base_linear_velocity - Vector3::new(-1.4, 0.0, 0.0)).norm() < 2e-4);
            assert!(
                (world.scene_bodies[0].linear_velocity - Vector3::new(0.6, 0.0, 0.0)).norm() < 2e-4
            );
            assert!((world.contact_forces[0] + diagnostics.forces[0]).norm() < 1e-2);
            assert!(!diagnostics.samples[0].is_empty());
            let split = scene.split_gpu_output(result).unwrap();
            assert!(
                (split.robot_root.unwrap().linear_velocity - Vector3::new(-1.4, 0.0, 0.0)).norm()
                    < 2e-4
            );
            assert!((split.bodies[0].linear_velocity - Vector3::new(0.6, 0.0, 0.0)).norm() < 2e-4);
            assert!((split.robot_root.unwrap().pose.translation.vector.x + 0.0014).abs() < 2e-6);
            assert!((split.bodies[0].pose.translation.vector.x - 1.0006).abs() < 2e-6);
            let mut reversed = result.clone();
            reversed.spherical_joints.as_mut().unwrap().reverse();
            assert!(
                (scene.split_gpu_output(&reversed).unwrap().bodies[0]
                    .pose
                    .rotation
                    .inverse()
                    * split.bodies[0].pose.rotation)
                    .angle()
                    < 1e-12
            );
            let duplicate = reversed.spherical_joints.as_ref().unwrap()[0];
            reversed.spherical_joints.as_mut().unwrap().push(duplicate);
            assert!(scene.split_gpu_output(&reversed).is_err());
            let mut invalid_root = result.clone();
            invalid_root.root_pose.translation.vector.x = 1.0;
            assert!(scene.split_gpu_output(&invalid_root).is_err());
            let robot_velocity = Vector3::from_column_slice(
                &result.state.velocities.as_slice()[robot_layout.translation.clone()],
            );
            let body_velocity = Vector3::from_column_slice(
                &result.state.velocities.as_slice()[body_layout.translation.clone()],
            );
            assert!(
                (robot_velocity - Vector3::new(-1.4, 0.0, 0.0)).norm() < 2e-4,
                "{backend:?}: {robot_velocity:?}"
            );
            assert!((body_velocity - Vector3::new(0.6, 0.0, 0.0)).norm() < 2e-4);
            assert!((robot_velocity * 2.0 + body_velocity * 3.0 + Vector3::x()).norm() < 2e-4);
            let robot_force = result
                .contacts
                .iter()
                .filter(|c| c.link == robot_layout.link)
                .fold(Vector3::zeros(), |sum, c| sum + c.force);
            let body_force = result
                .contacts
                .iter()
                .filter(|c| c.link == body_layout.link)
                .fold(Vector3::zeros(), |sum, c| sum + c.force);
            assert!(robot_force.x < -1000.0 && body_force.x > 1000.0);
            assert!((robot_force + body_force).norm() < 1e-2);
            let poses = batch.readback_link_poses().unwrap();
            assert!((poses[0][robot_layout.link].translation.vector.x + 0.0014).abs() < 2e-6);
            assert!((poses[0][body_layout.link].translation.vector.x - 1.0006).abs() < 2e-6);
            let mut invalid_joints = spherical.clone();
            invalid_joints[0].velocity_slot += 1000;
            assert!(
                batch
                    .reset_with_spherical_state(
                        core::slice::from_ref(&input.state),
                        &[invalid_joints]
                    )
                    .is_err()
            );
            assert!(
                (batch.readback().unwrap()[0].velocities.clone() - result.state.velocities.clone())
                    .norm()
                    < 1e-6
            );
            let mut invalid_state = input.state.clone();
            invalid_state.velocities[0] = f64::NAN;
            assert!(
                batch
                    .reset_with_spherical_state(&[invalid_state], core::slice::from_ref(&spherical))
                    .is_err()
            );
            assert!(
                (batch.readback().unwrap()[0].positions.clone() - result.state.positions.clone())
                    .norm()
                    < 1e-6
            );
            batch
                .reset_with_spherical_state(
                    core::slice::from_ref(&input.state),
                    core::slice::from_ref(&spherical),
                )
                .unwrap();
            assert!(
                batch.readback_contact_wake_requests().unwrap()[0]
                    .iter()
                    .all(|v| !v)
            );
            batch
                .update_contact_motion_wake(&[Some(crate::sleep::SleepSettings::default())])
                .unwrap();
            for threshold in [f64::NAN, f64::MAX, 1e-100, -1.0] {
                let invalid = crate::sleep::SleepSettings {
                    linear_velocity_threshold: threshold,
                    ..Default::default()
                };
                assert!(batch.update_contact_motion_wake(&[Some(invalid)]).is_err());
            }
            batch.enable_contact_activity().unwrap();
            batch.submit_steps(1).unwrap();
            let components = batch.readback_contact_components().unwrap();
            assert_eq!(
                components[0][robot_layout.link],
                components[0][body_layout.link]
            );
            let mut expected = (0..scene.articulation.link_count()).collect::<Vec<_>>();
            for &link in mobility.iter().flatten() {
                expected[link] = robot_layout.link.min(body_layout.link);
            }
            assert_eq!(components[0], expected);
            let mut invalid_groups = mobility.clone();
            let duplicate_link = invalid_groups[0][0];
            invalid_groups[0].push(duplicate_link);
            assert!(
                batch
                    .update_contact_mobility_groups(&[invalid_groups])
                    .is_err()
            );
            assert_eq!(batch.readback_contact_components().unwrap(), components);
            assert!(
                (batch.readback().unwrap()[0].velocities.clone() - result.state.velocities.clone())
                    .norm()
                    < 1e-6
            );
            let predicted_wake = batch.readback_contact_wake_requests().unwrap();
            for (link, &requested) in predicted_wake[0].iter().enumerate() {
                assert_eq!(
                    requested,
                    expected[link] == robot_layout.link.min(body_layout.link)
                );
            }
            // A load far below the speed threshold must still wake its free body
            // every step, without waking the now-separated robot branch.
            let quiet = crate::sleep::SleepSettings {
                linear_velocity_threshold: 100.0,
                angular_velocity_threshold: 100.0,
                ..Default::default()
            };
            batch.update_contact_motion_wake(&[Some(quiet)]).unwrap();
            batch.enable_contact_load_wake().unwrap();
            let mut loads = vec![GpuMassLinkLoad::default(); scene.articulation.link_count()];
            loads[body_layout.link].force.x = 1e-8;
            batch
                .update_link_loads(core::slice::from_ref(&loads))
                .unwrap();
            assert!(batch.update_link_loads(&[Vec::new()]).is_err());
            batch.submit_steps(2).unwrap();
            let force_wake = batch.readback_contact_wake_requests().unwrap();
            for (link, &requested) in force_wake[0].iter().enumerate() {
                assert_eq!(requested, mobility[1].contains(&link));
            }
            batch.enable_contact_load_wake().unwrap();
            assert_eq!(batch.readback_contact_wake_requests().unwrap(), force_wake);
            loads[body_layout.link].force.x = 0.0;
            batch.update_link_loads(&[loads]).unwrap();
            batch.submit_steps(1).unwrap();
            assert!(
                batch.readback_contact_wake_requests().unwrap()[0]
                    .iter()
                    .all(|v| !v)
            );
            // A resting geometric contact has no solved impulse, yet connects
            // both mobility groups. Losing it wakes each separated group once.
            let mut resting = input.state.clone();
            resting.velocities.fill(0.0);
            batch.update_contact_loss_wake(true).unwrap();
            batch
                .reset_with_spherical_state(&[resting], core::slice::from_ref(&spherical))
                .unwrap();
            batch.submit_steps(1).unwrap();
            assert!(
                batch.readback_contact_activity().unwrap()[0]
                    .iter()
                    .all(|v| !v)
            );
            let geometry = batch.readback_geometric_contacts().unwrap();
            for (link, &touching) in geometry[0].iter().enumerate() {
                assert_eq!(
                    touching,
                    link == robot_layout.link || link == body_layout.link
                );
            }
            let components = batch.readback_contact_components().unwrap();
            assert_eq!(
                components[0][robot_layout.link],
                components[0][body_layout.link]
            );
            assert!(
                batch.readback_contact_wake_requests().unwrap()[0]
                    .iter()
                    .all(|v| !v)
            );
            let mut separation_loads =
                vec![GpuMassLinkLoad::default(); scene.articulation.link_count()];
            separation_loads[body_layout.link].force.x = 300.0;
            batch
                .update_link_loads(core::slice::from_ref(&separation_loads))
                .unwrap();
            batch.submit_steps(1).unwrap();
            separation_loads[body_layout.link].force.x = 0.0;
            batch.update_link_loads(&[separation_loads]).unwrap();
            batch.submit_steps(1).unwrap();
            assert!(
                batch.readback_geometric_contacts().unwrap()[0]
                    .iter()
                    .all(|v| !v)
            );
            assert!(
                batch.readback_contact_activity().unwrap()[0]
                    .iter()
                    .all(|v| !v)
            );
            let loss_wake = batch.readback_contact_wake_requests().unwrap();
            for (link, &requested) in loss_wake[0].iter().enumerate() {
                assert_eq!(
                    requested,
                    mobility.iter().any(|group| group.contains(&link))
                );
            }
            let components = batch.readback_contact_components().unwrap();
            assert_ne!(
                components[0][robot_layout.link],
                components[0][body_layout.link]
            );
            batch.submit_steps(1).unwrap();
            assert!(
                batch.readback_contact_wake_requests().unwrap()[0]
                    .iter()
                    .all(|v| !v)
            );
            let mut idle_state = input.state.clone();
            idle_state.velocities.fill(0.0);
            let idle_settings = crate::sleep::SleepSettings {
                time_threshold: 0.002,
                linear_velocity_threshold: 100.0,
                angular_velocity_threshold: 100.0,
                ..Default::default()
            };
            batch
                .update_contact_idle_settings(&[Some(idle_settings)])
                .unwrap();
            batch
                .reset_with_spherical_state(
                    core::slice::from_ref(&idle_state),
                    core::slice::from_ref(&spherical),
                )
                .unwrap();
            batch.submit_steps(1).unwrap();
            assert!(
                batch.readback_contact_sleep_candidates().unwrap()[0]
                    .iter()
                    .all(|v| !v)
            );
            for time_threshold in [f64::NAN, f64::MAX, 1e-100, 0.0] {
                let invalid = crate::sleep::SleepSettings {
                    time_threshold,
                    ..idle_settings
                };
                assert!(
                    batch
                        .update_contact_idle_settings(&[Some(invalid)])
                        .is_err()
                );
            }
            batch.submit_steps(1).unwrap();
            let candidates = batch.readback_contact_sleep_candidates().unwrap();
            let labels = batch.readback_contact_components().unwrap();
            let mut split_groups = mobility.clone();
            let separated = split_groups[1].pop().unwrap();
            split_groups.push(vec![separated]);
            assert!(
                batch
                    .update_contact_sleep_mobility_groups(&[split_groups])
                    .is_err()
            );
            assert!(batch.update_contact_sleep_mobility_groups(&[]).is_err());
            assert_eq!(
                batch.readback_contact_sleep_candidates().unwrap(),
                candidates
            );
            assert_eq!(batch.readback_contact_components().unwrap(), labels);
            for (link, &candidate) in candidates[0].iter().enumerate() {
                assert_eq!(
                    candidate,
                    mobility.iter().any(|group| group.contains(&link))
                );
            }
            let mut tiny_loads = vec![GpuMassLinkLoad::default(); scene.articulation.link_count()];
            tiny_loads[body_layout.link].force.x = 1e-8;
            batch
                .update_link_loads(core::slice::from_ref(&tiny_loads))
                .unwrap();
            batch.submit_steps(1).unwrap();
            assert!(
                batch.readback_contact_sleep_candidates().unwrap()[0]
                    .iter()
                    .all(|v| !v)
            );
            tiny_loads[body_layout.link].force.x = 0.0;
            batch.update_link_loads(&[tiny_loads]).unwrap();
            batch.submit_steps(1).unwrap();
            assert!(
                batch.readback_contact_sleep_candidates().unwrap()[0]
                    .iter()
                    .all(|v| !v)
            );
            batch.submit_steps(1).unwrap();
            assert_eq!(
                batch.readback_contact_sleep_candidates().unwrap(),
                candidates
            );
            batch
                .reset_with_spherical_state(&[idle_state], core::slice::from_ref(&spherical))
                .unwrap();
            assert!(
                batch.readback_contact_sleep_candidates().unwrap()[0]
                    .iter()
                    .all(|v| !v)
            );
            batch.submit_steps(2).unwrap();
            assert_eq!(
                batch.readback_contact_sleep_candidates().unwrap(),
                candidates
            );
            batch.enable_contact_actuation_wake().unwrap();
            let mut actuated = input.joints.clone();
            actuated[body_layout.translation.start].base_force = 1e-8;
            batch
                .update_joints(core::slice::from_ref(&actuated))
                .unwrap();
            let mut invalid = actuated.clone();
            invalid[body_layout.translation.start].base_force = f64::NAN;
            assert!(batch.update_joints(&[invalid]).is_err());
            for _ in 0..2 {
                batch.submit_steps(1).unwrap();
                assert!(
                    batch.readback_contact_sleep_candidates().unwrap()[0]
                        .iter()
                        .all(|v| !v)
                );
                let wake = batch.readback_contact_wake_requests().unwrap();
                for (link, &requested) in wake[0].iter().enumerate() {
                    assert_eq!(
                        requested,
                        mobility.iter().any(|group| group.contains(&link))
                    );
                }
            }
            batch
                .update_joints(core::slice::from_ref(&input.joints))
                .unwrap();
            batch.submit_steps(1).unwrap();
            assert!(
                batch.readback_contact_sleep_candidates().unwrap()[0]
                    .iter()
                    .all(|v| !v)
            );
            batch.submit_steps(1).unwrap();
            assert_eq!(
                batch.readback_contact_sleep_candidates().unwrap(),
                candidates
            );
            batch.enable_contact_gravity_wake().unwrap();
            batch.submit_steps(1).unwrap();
            assert_eq!(
                batch.readback_contact_sleep_candidates().unwrap(),
                candidates
            );
            let changed_gravity = Vector3::new(0.0, 1e-8, 0.0);
            batch
                .update_gravity(core::slice::from_ref(&changed_gravity))
                .unwrap();
            assert!(
                batch
                    .update_gravity(&[Vector3::new(f64::NAN, 0.0, 0.0)])
                    .is_err()
            );
            batch.submit_steps(1).unwrap();
            assert!(
                batch.readback_contact_sleep_candidates().unwrap()[0]
                    .iter()
                    .all(|v| !v)
            );
            let gravity_wake = batch.readback_contact_wake_requests().unwrap();
            for (link, &requested) in gravity_wake[0].iter().enumerate() {
                assert_eq!(
                    requested,
                    mobility.iter().any(|group| group.contains(&link))
                );
            }
            batch.update_gravity(&[changed_gravity]).unwrap();
            batch.submit_steps(1).unwrap();
            assert!(
                batch.readback_contact_wake_requests().unwrap()[0]
                    .iter()
                    .all(|v| !v)
            );
            batch.submit_steps(1).unwrap();
            assert_eq!(
                batch.readback_contact_sleep_candidates().unwrap(),
                candidates
            );
            let mut pending = vec![false; scene.articulation.link_count()];
            pending[body_layout.link] = true;
            batch.update_contact_wake_requests(&[pending]).unwrap();
            batch
                .update_contact_idle_requires_static_support(true)
                .unwrap();
            assert!(
                batch.readback_contact_sleep_candidates().unwrap()[0]
                    .iter()
                    .all(|v| !v)
            );
            batch.submit_steps(1).unwrap();
            let propagated = batch.readback_contact_wake_requests().unwrap();
            for (link, &requested) in propagated[0].iter().enumerate() {
                assert_eq!(
                    requested,
                    mobility.iter().any(|group| group.contains(&link))
                );
            }
            batch.submit_steps(2).unwrap();
            assert!(
                batch.readback_contact_wake_requests().unwrap()[0]
                    .iter()
                    .all(|v| !v)
            );
            let geometry = batch.readback_geometric_contacts().unwrap();
            assert!(geometry[0][robot_layout.link] && geometry[0][body_layout.link]);
            assert!(
                batch.readback_external_static_contacts().unwrap()[0]
                    .iter()
                    .all(|v| !v)
            );
            assert!(
                batch.readback_contact_gravity_support().unwrap()[0]
                    .iter()
                    .all(|v| !v)
            );
            assert!(
                batch.readback_contact_world_anchors().unwrap()[0]
                    .iter()
                    .all(|v| !v)
            );
            assert!(
                batch.readback_contact_fixed_world_anchors().unwrap()[0]
                    .iter()
                    .all(|v| !v)
            );
            assert!(
                batch.readback_contact_sleep_candidates().unwrap()[0]
                    .iter()
                    .all(|v| !v)
            );
            batch
                .update_contact_idle_requires_static_support(false)
                .unwrap();
            batch.submit_steps(1).unwrap();
            assert!(
                batch.readback_contact_sleep_candidates().unwrap()[0]
                    .iter()
                    .all(|v| !v)
            );
            batch.submit_steps(1).unwrap();
            assert_eq!(
                batch.readback_contact_sleep_candidates().unwrap(),
                candidates
            );
            // Exercise the split-step freeze mechanism independently of final sleep policy.
            let before = batch.readback().unwrap();
            let mut encoder = batch.device.create_command_encoder(&Default::default());
            batch.encode_contact_step(&mut encoder).unwrap();
            freeze.encode(&mut encoder);
            batch.encode_integration(&mut encoder);
            let _ = batch.queue.submit(Some(encoder.finish()));
            let frozen = batch.readback().unwrap();
            assert_eq!(frozen[0].positions, before[0].positions);
            assert!(frozen[0].velocities.iter().all(|v| *v == 0.0));
            let flags = crate::gpu_articulated_mass::read_buffer(
                &batch.device,
                &batch.queue,
                freeze.frozen_buffer(),
            )
            .unwrap();
            assert!(
                flags
                    .chunks_exact(4)
                    .all(|v| u32::from_ne_bytes(v.try_into().unwrap()) == 1)
            );
            let mut pending = vec![false; scene.articulation.link_count()];
            pending[body_layout.link] = true;
            batch.update_contact_wake_requests(&[pending]).unwrap();
            let mut encoder = batch.device.create_command_encoder(&Default::default());
            batch.encode_contact_step(&mut encoder).unwrap();
            freeze.encode(&mut encoder);
            batch.encode_integration(&mut encoder);
            let _ = batch.queue.submit(Some(encoder.finish()));
            let awake = batch.readback().unwrap();
            assert!(awake[0].velocities.iter().any(|v| *v != 0.0));
            let flags = crate::gpu_articulated_mass::read_buffer(
                &batch.device,
                &batch.queue,
                freeze.frozen_buffer(),
            )
            .unwrap();
            assert!(flags.iter().all(|v| *v == 0));
            batch.enable_contact_sleep(&[Some(idle_settings)]).unwrap();
            let zero_drive = SphericalJointDrive {
                orientation_target: nalgebra::UnitQuaternion::identity(),
                velocity_target: Vector3::zeros(),
                stiffness: Vector3::zeros(),
                damping: Vector3::zeros(),
                max_torque: Vector3::zeros(),
            };
            let changed = vec![Some(zero_drive); spherical.len()];
            batch
                .update_spherical_drives(core::slice::from_ref(&changed))
                .unwrap();
            let invalid = vec![
                Some(SphericalJointDrive {
                    stiffness: Vector3::new(f64::NAN, 0.0, 0.0),
                    ..zero_drive
                });
                spherical.len()
            ];
            assert!(
                batch
                    .update_spherical_drives(core::slice::from_ref(&invalid))
                    .is_err()
            );
            batch.submit_steps(1).unwrap();
            let wake = batch.readback_contact_wake_requests().unwrap();
            for (link, &requested) in wake[0].iter().enumerate() {
                assert_eq!(
                    requested,
                    mobility.iter().any(|group| group.contains(&link))
                );
            }
            batch
                .update_spherical_drives(core::slice::from_ref(&changed))
                .unwrap();
            batch.submit_steps(1).unwrap();
            assert!(
                batch.readback_contact_wake_requests().unwrap()[0]
                    .iter()
                    .all(|v| !v)
            );
            eprintln!(
                "composed scene contact, idle, wake, static-support and split freeze passed on {backend:?}"
            );
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn resident_quaternion_dynamics_crosses_gimbal_lock_and_matches_cpu() {
        use nalgebra::UnitQuaternion;
        let body = LinkSpec {
            mass: 1.0,
            center_of_mass: Vector3::zeros(),
            inertia: Matrix3::identity() * 0.5,
        };
        let art = Articulation::new(
            vec![body.clone(), body],
            vec![JointSpec {
                parent: 0,
                child: 1,
                kind: JointKind::Spherical,
                origin: Isometry3::identity(),
                axis: Vector3::z(),
                limits: None,
            }],
            0,
        )
        .unwrap();
        let flags = [false, true];
        let q0 = UnitQuaternion::from_scaled_axis(Vector3::y() * 1.4);
        let drive = SphericalJointDrive {
            orientation_target: UnitQuaternion::from_scaled_axis(Vector3::y() * 1.8),
            velocity_target: Vector3::zeros(),
            stiffness: Vector3::repeat(2.0),
            damping: Vector3::repeat(0.2),
            max_torque: Vector3::repeat(10.0),
        };
        let inputs = flags.map(|floating| {
            let n = if floating { 9 } else { 3 };
            let mut velocity = DVector::zeros(n);
            velocity[n - 2] = 4.0;
            GpuArticulatedDynamicsInput {
                articulation: &art,
                root_pose: Isometry3::identity(),
                state: GpuGeneralizedState {
                    positions: DVector::from_element(n, 123.0),
                    velocities: velocity,
                },
                gravity: Vector3::zeros(),
                joints: vec![GpuJointForceInput::default(); n],
                joint_velocity_limit: None,
                coordinate_velocity_limits: None,
                link_loads: vec![GpuMassLinkLoad::default(); 2],
                ground_manifold_start: None,
                contact_iterations: 8,
                contact_warm_start: false,
                joint_friction: Vec::new(),
                joint_couplings: Vec::new(),
                link_point_constraints: Vec::new(),
                link_fixed_constraints: Vec::new(),
                ground_spheres: Vec::new(),
                ground_capsules: Vec::new(),
                ground_boxes: Vec::new(),
                ground_axial_shapes: Vec::new(),
                sphere_pairs: Vec::new(),
                static_sphere_pairs: Vec::new(),
                external_sphere_bodies: None,
                external_box_bodies: None,
                external_capsule_bodies: None,
                external_axial_bodies: None,
                external_convex_bodies: None,
                external_constraint_bodies: None,
                external_indexed_bodies: None,
                static_capsule_sphere_pairs: Vec::new(),
                static_box_sphere_pairs: Vec::new(),
                static_sphere_capsule_pairs: Vec::new(),
                static_capsule_pairs: Vec::new(),
                static_sphere_box_pairs: Vec::new(),
                static_capsule_box_pairs: Vec::new(),
                static_box_pairs: Vec::new(),
                static_box_capsule_pairs: Vec::new(),
                static_axial_sphere_pairs: Vec::new(),
                static_axial_capsule_pairs: Vec::new(),
                static_axial_box_pairs: Vec::new(),
                static_axial_convex_pairs: Vec::new(),
                axial_convex_pairs: Vec::new(),
                axial_sphere_pairs: Vec::new(),
                axial_box_pairs: Vec::new(),
                axial_capsule_pairs: Vec::new(),
                axial_pairs: Vec::new(),
                convex_sphere_pairs: Vec::new(),
                static_convex_sphere_pairs: Vec::new(),
                static_convex_capsule_pairs: Vec::new(),
                convex_capsule_pairs: Vec::new(),
                convex_pairs: Vec::new(),
                static_convex_pairs: Vec::new(),
                scene_convex_sphere_pairs: Vec::new(),
                scene_mesh_sphere_pairs: Vec::new(),
                scene_polyline_sphere_pairs: Vec::new(),
                scene_mesh_capsule_pairs: Vec::new(),
                scene_polyline_capsule_pairs: Vec::new(),
                scene_mesh_box_pairs: Vec::new(),
                scene_polyline_box_pairs: Vec::new(),
                scene_mesh_axial_pairs: Vec::new(),
                scene_polyline_axial_pairs: Vec::new(),
                scene_mesh_convex_pairs: Vec::new(),
                scene_polyline_convex_pairs: Vec::new(),
                scene_convex_capsule_pairs: Vec::new(),
                capsule_sphere_pairs: Vec::new(),
                capsule_pairs: Vec::new(),
                sphere_box_pairs: Vec::new(),
                capsule_box_pairs: Vec::new(),
                box_pairs: Vec::new(),
                link_spheres: Vec::new(),
                link_capsules: Vec::new(),
                link_boxes: Vec::new(),
                dynamic_material_rules: Vec::new(),
            }
        });
        let spherical = flags.map(|floating| {
            vec![GpuSphericalJointState {
                velocity_slot: if floating { 6 } else { 0 },
                orientation: q0,
            }]
        });
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let batch = GpuArticulatedDynamicsBatch::new_with_spherical_state(
                &context,
                &inputs,
                0.001,
                &flags,
                &spherical,
                &[vec![Some(drive)], vec![Some(drive)]],
            )
            .unwrap();
            let mut velocities = inputs.clone().map(|input| input.state.velocities);
            let mut roots = [Isometry3::identity(); 2];
            let mut orientations = [q0; 2];
            for _ in 0..1000 {
                for i in 0..2 {
                    let start = if flags[i] { 6 } else { 0 };
                    let omega =
                        Vector3::from_column_slice(&velocities[i].as_slice()[start..start + 3]);
                    let dynamics = art
                        .generalized_dynamics_with_spherical_orientations(
                            roots[i],
                            &[123.0; 3],
                            &[Some(orientations[i])],
                            &velocities[i],
                            flags[i],
                            Vector3::zeros(),
                        )
                        .unwrap();
                    let torque = drive.torque(orientations[i], omega).unwrap();
                    let mut force = -dynamics.velocity_bias;
                    for j in 0..3 {
                        force[start + j] += torque[j];
                    }
                    velocities[i] += dynamics.mass.lu().solve(&force).unwrap() * 0.001;
                    let omega =
                        Vector3::from_column_slice(&velocities[i].as_slice()[start..start + 3]);
                    orientations[i] =
                        UnitQuaternion::from_scaled_axis(omega * 0.001) * orientations[i];
                    if flags[i] {
                        roots[i].translation.vector +=
                            Vector3::from_column_slice(&velocities[i].as_slice()[..3]) * 0.001;
                        roots[i].rotation = UnitQuaternion::from_scaled_axis(
                            Vector3::from_column_slice(&velocities[i].as_slice()[3..6]) * 0.001,
                        ) * roots[i].rotation;
                    }
                }
            }
            batch.submit_steps(1000).unwrap();
            let output = batch.readback_output().unwrap();
            for i in 0..2 {
                assert!((&output[i].state.velocities - &velocities[i]).norm() < 2e-3);
                assert_eq!(output[i].state.positions, inputs[i].state.positions);
                let actual = output[i].spherical_joints.as_ref().unwrap()[0];
                assert_eq!(actual.velocity_slot, spherical[i][0].velocity_slot);
                assert!((actual.orientation.inverse() * orientations[i]).angle() < 2e-3);
                assert!(
                    (output[i].root_pose.rotation.inverse() * roots[i].rotation).angle() < 2e-3
                );
            }
            for i in 0..2 {
                use crate::articulated_world::{ArticulatedWorld, ArticulatedWorldParams};
                let mut world = if flags[i] {
                    ArticulatedWorld::new_floating(
                        art.clone(),
                        Isometry3::identity(),
                        vec![],
                        ArticulatedWorldParams::default(),
                    )
                    .unwrap()
                } else {
                    ArticulatedWorld::new(
                        art.clone(),
                        Isometry3::identity(),
                        vec![],
                        ArticulatedWorldParams::default(),
                    )
                    .unwrap()
                };
                let start = if flags[i] { 6 } else { 0 };
                world
                    .set_tangent_spherical_state(
                        &spherical[i],
                        &DVector::from_column_slice(
                            &inputs[i].state.velocities.as_slice()[start..],
                        ),
                    )
                    .unwrap();
                world.apply_gpu_dynamics_output(&output[i]).unwrap();
                let poses = world.link_poses().unwrap();
                let expected = art
                    .pose_with_spherical_orientations(
                        output[i].root_pose,
                        &[0.0; 3],
                        &[Some(
                            output[i].spherical_joints.as_ref().unwrap()[0].orientation,
                        )],
                    )
                    .unwrap();
                for (actual, expected) in poses.iter().zip(expected.links) {
                    assert!((actual.rotation.inverse() * expected.rotation).angle() < 1e-9);
                }
                let local_point = Vector3::new(0.2, 0.3, 0.4);
                let twist = world.link_point_twist(1, local_point).unwrap();
                let acceleration = world
                    .link_point_acceleration(
                        1,
                        local_point,
                        &vec![0.0; if flags[i] { 9 } else { 3 }],
                    )
                    .unwrap();
                let radius = poses[1].rotation * local_point;
                let centripetal = twist.angular.cross(&twist.angular.cross(&radius));
                assert!((acceleration.linear - centripetal).norm() < 1e-6);
                assert!(acceleration.angular.norm() < 1e-6);
                let snapshot = world.snapshot();
                let prior = world.gpu_spherical_state().unwrap()[0].orientation;
                let mut invalid = output[i].clone();
                invalid.spherical_joints.as_mut().unwrap()[0].velocity_slot += 1;
                assert!(world.apply_gpu_dynamics_output(&invalid).is_err());
                assert!(
                    (world.gpu_spherical_state().unwrap()[0]
                        .orientation
                        .inverse()
                        * prior)
                        .angle()
                        < 1e-12
                );
                world
                    .set_tangent_spherical_state(&spherical[i], &DVector::zeros(3))
                    .unwrap();
                world.restore_snapshot(&snapshot).unwrap();
                assert!(
                    (world.gpu_spherical_state().unwrap()[0]
                        .orientation
                        .inverse()
                        * prior)
                        .angle()
                        < 1e-12
                );
                assert!(world.step(0.001, &[0.0; 3]).is_err());
            }
            let mut bad = inputs
                .iter()
                .map(|input| input.joints.clone())
                .collect::<Vec<_>>();
            bad[0][0].passive.stiffness = 1.0;
            assert!(batch.update_joints(&bad).is_err());
            batch.reset_spherical_orientations(&spherical).unwrap();
            batch
                .reset(
                    &inputs
                        .iter()
                        .map(|input| input.state.clone())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            for i in 0..2 {
                batch.set_root_pose(i, Isometry3::identity()).unwrap();
            }
            let output = batch.readback_output().unwrap();
            for environment in output {
                assert!(
                    (environment.spherical_joints.unwrap()[0]
                        .orientation
                        .inverse()
                        * q0)
                        .angle()
                        < 1e-6
                );
            }
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn mixed_floating_dynamics_contact_and_joint_equality_match_cpu_world() {
        use crate::articulated_world::{ArticulatedWorld, ArticulatedWorldParams, LinkSphere};
        let model = Articulation::new(
            vec![link(2.0), link(1.0)],
            vec![JointSpec {
                parent: 1,
                child: 0,
                kind: JointKind::Revolute,
                origin: Isometry3::translation(0.4, 0.0, 0.1),
                axis: Vector3::y(),
                limits: None,
            }],
            1,
        )
        .unwrap();
        let flags = [true, true, false, true, true];
        let roots = [
            Isometry3::translation(0.0, 0.0, 2.0),
            Isometry3::translation(0.0, 0.0, 0.099),
            Isometry3::translation(0.0, 0.0, 0.099),
            Isometry3::translation(0.0, 0.0, 0.099),
        ];
        let mut templates = Vec::new();
        let mut cpu = Vec::new();
        for i in 0..4 {
            let colliders = if i == 0 {
                vec![]
            } else {
                vec![LinkSphere {
                    link: 1,
                    center: Vector3::new(0.15, 0.0, 0.0),
                    radius: 0.1,
                }]
            };
            templates.push(
                ArticulatedWorld::new(
                    model.clone(),
                    roots[i],
                    colliders.clone(),
                    ArticulatedWorldParams::default(),
                )
                .unwrap(),
            );
            let mut world = if flags[i] {
                ArticulatedWorld::new_floating(
                    model.clone(),
                    roots[i],
                    colliders,
                    ArticulatedWorldParams::default(),
                )
                .unwrap()
            } else {
                ArticulatedWorld::new(
                    model.clone(),
                    roots[i],
                    colliders,
                    ArticulatedWorldParams::default(),
                )
                .unwrap()
            };
            world.positions[0] = 0.001;
            world.velocities[0] = 0.02;
            world.set_implicit_coriolis(false);
            if flags[i] {
                world.base_linear_velocity = Vector3::new(0.1, -0.1, -0.01);
                world.base_angular_velocity = Vector3::new(0.05, -0.03, 0.02);
                world.base_force = Vector3::new(0.1, 0.2, 0.3);
            }
            if i == 3 {
                world
                    .set_joint_polynomial_couplings(vec![JointPolynomialCoupling {
                        follower: 0,
                        source: None,
                        coefficients: [0.001, 0.0, 0.0, 0.0, 0.0],
                        follower_reference: 0.0,
                        source_reference: 0.0,
                    }])
                    .unwrap();
            }
            cpu.push(world);
        }
        let mut inputs = templates
            .iter()
            .map(|world| world.gpu_dynamics_input(&[0.2]).unwrap())
            .collect::<Vec<_>>();
        for i in 0..4 {
            if flags[i] {
                let mut positions = vec![0.0; 6];
                positions.push(0.001);
                let v = &cpu[i];
                inputs[i].state = GpuGeneralizedState {
                    positions: DVector::from_vec(positions),
                    velocities: DVector::from_vec(vec![
                        v.base_linear_velocity.x,
                        v.base_linear_velocity.y,
                        v.base_linear_velocity.z,
                        v.base_angular_velocity.x,
                        v.base_angular_velocity.y,
                        v.base_angular_velocity.z,
                        0.02,
                    ]),
                };
                let mut drives = vec![GpuJointForceInput::default(); 6];
                for (slot, effort) in [0.1, 0.2, 0.3].into_iter().enumerate() {
                    drives[slot].base_force = effort;
                }
                drives.extend(inputs[i].joints.clone());
                inputs[i].joints = drives;
                inputs[i].joint_friction = vec![0.0; 7];
            } else {
                inputs[i].state.positions[0] = 0.001;
                inputs[i].state.velocities[0] = 0.02;
            }
            if i == 3 {
                inputs[i].joint_couplings = vec![JointPolynomialCoupling {
                    follower: 6,
                    source: None,
                    coefficients: [0.001, 0.0, 0.0, 0.0, 0.0],
                    follower_reference: 0.0,
                    source_reference: 0.0,
                }];
            }
        }
        let pure = Articulation::new(vec![link(1.5)], vec![], 0).unwrap();
        let mut pure_input = inputs[0].clone();
        pure_input.articulation = &pure;
        pure_input.state.positions = DVector::zeros(6);
        pure_input.state.velocities = DVector::from_vec(vec![0.1, -0.1, -0.01, 0.05, -0.03, 0.02]);
        pure_input.joints.truncate(6);
        pure_input.joint_friction.truncate(6);
        pure_input.link_loads.truncate(1);
        inputs.push(pure_input);
        let mut pure_world = ArticulatedWorld::new_floating(
            pure.clone(),
            roots[0],
            vec![],
            ArticulatedWorldParams::default(),
        )
        .unwrap();
        pure_world.base_linear_velocity = Vector3::new(0.1, -0.1, -0.01);
        pure_world.base_angular_velocity = Vector3::new(0.05, -0.03, 0.02);
        pure_world.base_force = Vector3::new(0.1, 0.2, 0.3);
        cpu.push(pure_world);
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            eprintln!("mixed floating dynamics contact: {backend:?}");
            assert!(GpuArticulatedDynamicsBatch::new(&context, &inputs, 0.001).is_err());
            let mut invalid = inputs.clone();
            invalid[0].joints[0].passive.stiffness = 1.0;
            assert!(
                GpuArticulatedDynamicsBatch::new_with_floating_roots(
                    &context, &invalid, 0.001, &flags
                )
                .is_err()
            );
            invalid = inputs.clone();
            invalid[0].joint_friction[0] = 1.0;
            assert!(
                GpuArticulatedDynamicsBatch::new_with_floating_roots(
                    &context, &invalid, 0.001, &flags
                )
                .is_err()
            );
            let batch = GpuArticulatedDynamicsBatch::new_with_floating_roots(
                &context, &inputs, 0.001, &flags,
            )
            .unwrap();
            let mut expected = cpu
                .iter()
                .map(|w| {
                    let mut world = if w.floating {
                        ArticulatedWorld::new_floating(
                            w.articulation.clone(),
                            w.root_pose,
                            w.colliders.clone(),
                            ArticulatedWorldParams::default(),
                        )
                        .unwrap()
                    } else {
                        ArticulatedWorld::new(
                            w.articulation.clone(),
                            w.root_pose,
                            w.colliders.clone(),
                            ArticulatedWorldParams::default(),
                        )
                        .unwrap()
                    };
                    world.positions = w.positions.clone();
                    world.velocities = w.velocities.clone();
                    world.base_linear_velocity = w.base_linear_velocity;
                    world.base_angular_velocity = w.base_angular_velocity;
                    world.base_force = w.base_force;
                    world.set_implicit_coriolis(false);
                    world
                })
                .collect::<Vec<_>>();
            expected[3]
                .set_joint_polynomial_couplings(vec![JointPolynomialCoupling {
                    follower: 0,
                    source: None,
                    coefficients: [0.001, 0.0, 0.0, 0.0, 0.0],
                    follower_reference: 0.0,
                    source_reference: 0.0,
                }])
                .unwrap();
            for count in [1, 31] {
                batch.submit_steps(count).unwrap();
                for _ in 0..count {
                    for world in &mut expected {
                        world
                            .step(0.001, &vec![0.2; world.articulation.dof()])
                            .unwrap();
                    }
                }
                let output = batch.readback_output().unwrap();
                assert_eq!(output.len(), inputs.len());
                let states = output.iter().map(|o| &o.state).collect::<Vec<_>>();
                let poses = batch.readback_link_poses().unwrap();
                for i in 0..inputs.len() {
                    assert_eq!(output[i].floating_root, flags[i]);
                    let reference_world = &expected[i];
                    let mut reflected = if flags[i] {
                        ArticulatedWorld::new_floating(
                            reference_world.articulation.clone(),
                            reference_world.root_pose,
                            reference_world.colliders.clone(),
                            *reference_world.params(),
                        )
                        .unwrap()
                    } else {
                        ArticulatedWorld::new(
                            reference_world.articulation.clone(),
                            reference_world.root_pose,
                            reference_world.colliders.clone(),
                            *reference_world.params(),
                        )
                        .unwrap()
                    };
                    reflected.apply_gpu_dynamics_output(&output[i]).unwrap();
                    assert_eq!(reflected.root_pose, output[i].root_pose);
                    assert_eq!(
                        reflected.velocities.as_slice(),
                        &output[i].state.velocities.as_slice()[if flags[i] { 6 } else { 0 }..]
                    );
                    for link in 0..reference_world.articulation.link_count() {
                        let force_error = (reflected.contact_forces[link]
                            - reference_world.contact_forces[link])
                            .norm();
                        assert!(
                            force_error
                                < 0.05 + 0.001 * reference_world.contact_forces[link].norm(),
                            "env {i}, link {link}, force after {count}: {:?} {:?}",
                            reflected.contact_forces[link],
                            reference_world.contact_forces[link]
                        );
                    }
                    let offset = if flags[i] { 6 } else { 0 };
                    if expected[i].articulation.dof() != 0 {
                        assert!(
                            (states[i].positions[offset] - expected[i].positions[0]).abs() < 2e-4,
                            "env {i} joint q"
                        );
                        assert!(
                            (states[i].velocities[offset] - expected[i].velocities[0]).abs() < 2e-3,
                            "env {i} joint v after {count}: GPU {:?}, CPU {:?}; GPU base {:?}, CPU base {:?} {:?}",
                            states[i].velocities,
                            expected[i].velocities,
                            states[i].positions,
                            expected[i].base_linear_velocity,
                            expected[i].base_angular_velocity
                        );
                    }
                    if flags[i] {
                        let v = &states[i].velocities;
                        assert!(
                            (Vector3::new(v[0], v[1], v[2]) - expected[i].base_linear_velocity)
                                .norm()
                                < 2e-3,
                            "env {i} base v"
                        );
                        assert!(
                            (Vector3::new(v[3], v[4], v[5]) - expected[i].base_angular_velocity)
                                .norm()
                                < 2e-3,
                            "env {i} base omega"
                        );
                    }
                    let reference = expected[i]
                        .articulation
                        .pose(expected[i].root_pose, expected[i].positions.as_slice())
                        .unwrap();
                    for (actual, reference) in poses[i].iter().zip(reference.links) {
                        assert!(
                            (actual.translation.vector - reference.translation.vector).norm()
                                < 2e-4,
                            "env {i} pose"
                        );
                        assert!(
                            (actual.rotation.inverse() * reference.rotation).angle() < 2e-4,
                            "env {i} rotation"
                        );
                    }
                }
            }
        }
        assert!(tested > 0, "no GPU backend available");
    }

    fn link(mass: f64) -> LinkSpec {
        LinkSpec {
            mass,
            center_of_mass: Vector3::new(0.25, 0.12, 0.18),
            inertia: Matrix3::from_diagonal(&Vector3::new(0.2, 0.3, 0.4)),
        }
    }

    fn drive_force(input: GpuJointForceInput, q: f64, v: f64) -> (f64, f64, f64) {
        let d = q - input.passive.rest_position;
        let spring_tangent = input.passive.stiffness
            + 2.0 * input.nonlinear.spring_quadratic * d
            + 3.0 * input.nonlinear.spring_cubic * d * d;
        let damping_tangent = input.passive.damping
            + 2.0 * input.nonlinear.damping_quadratic * v.abs()
            + 3.0 * input.nonlinear.damping_cubic * v * v;
        let passive = -input.passive.stiffness * d
            - input.passive.damping * v
            - input.nonlinear.spring_quadratic * d * d
            - input.nonlinear.spring_cubic * d * d * d
            - input.nonlinear.damping_quadratic * v * v.abs()
            - input.nonlinear.damping_cubic * v * v * v;
        let motor = input.motor.map_or(0.0, |motor| {
            (motor.stiffness * motor.position_target.map_or(0.0, |target| target - q)
                + motor.damping * (motor.velocity_target - v))
                .clamp(-motor.max_force, motor.max_force)
        });
        (
            input.base_force + passive + motor,
            spring_tangent,
            damping_tangent,
        )
    }

    fn cpu_step(input: &GpuArticulatedDynamicsInput<'_>, state: &mut GpuGeneralizedState, dt: f64) {
        let articulation = input.articulation;
        let pose = articulation
            .pose(input.root_pose, state.positions.as_slice())
            .unwrap();
        let dynamics = articulation
            .generalized_dynamics(
                input.root_pose,
                state.positions.as_slice(),
                &state.velocities,
                false,
                input.gravity,
            )
            .unwrap();
        let mut force = dynamics.gravity_force - dynamics.velocity_bias;
        let mut mass = dynamics.mass;
        for (index, load) in input.link_loads.iter().enumerate() {
            let link = articulation.link(index).unwrap();
            let (linear, angular) = articulation
                .point_jacobians(&pose, index, link.center_of_mass)
                .unwrap();
            force += linear.transpose()
                * (load.force + input.gravity * (link.mass * (load.gravity_scale - 1.0)))
                + angular.transpose() * load.torque;
        }
        for (axis, &drive) in input.joints.iter().enumerate() {
            let v = state.velocities[axis];
            let (applied, spring_tangent, damping_tangent) =
                drive_force(drive, state.positions[axis], v);
            force[axis] += applied - dt * spring_tangent * v;
            mass[(axis, axis)] += dt * damping_tangent + dt * dt * spring_tangent;
        }
        let acceleration = mass.lu().solve(&force).unwrap();
        state.velocities += acceleration * dt;
        for axis in 0..articulation.dof() {
            if let Some(cap) = input.joint_velocity_limit {
                state.velocities[axis] = state.velocities[axis].clamp(-cap, cap);
            }
            let candidate = state.positions[axis] + state.velocities[axis] * dt;
            state.positions[axis] = if let Some((lower, upper)) = articulation.joint_limit(axis) {
                let bounded = candidate.clamp(lower, upper);
                if bounded != candidate {
                    state.velocities[axis] = 0.0;
                }
                bounded
            } else {
                candidate
            };
        }
    }

    #[test]
    fn composed_gpu_dynamics_matches_cpu_across_two_environments_and_steps() {
        let joint = |parent, child, kind, axis, origin| JointSpec {
            parent,
            child,
            kind,
            origin,
            axis,
            limits: None,
        };
        let serial = Articulation::new(
            vec![link(0.0), link(0.0), link(1.7)],
            vec![
                joint(
                    0,
                    1,
                    JointKind::Revolute,
                    Vector3::y(),
                    Isometry3::translation(0.5, 0.0, 0.0),
                ),
                joint(
                    1,
                    2,
                    JointKind::Revolute,
                    Vector3::z(),
                    Isometry3::translation(0.8, 0.1, 0.0),
                ),
            ],
            0,
        )
        .unwrap();
        let slider = Articulation::new(
            vec![link(0.0), link(2.0)],
            vec![JointSpec {
                limits: Some((0.399, 0.5)),
                ..joint(
                    0,
                    1,
                    JointKind::Prismatic,
                    Vector3::z(),
                    Isometry3::identity(),
                )
            }],
            0,
        )
        .unwrap();
        let inputs = [
            GpuArticulatedDynamicsInput {
                articulation: &serial,
                root_pose: Isometry3::rotation(Vector3::new(0.1, 0.0, -0.2)),
                state: GpuGeneralizedState {
                    positions: DVector::from_column_slice(&[0.3, -0.2]),
                    velocities: DVector::from_column_slice(&[0.6, -0.4]),
                },
                gravity: Vector3::new(0.0, 0.0, -9.0),
                joints: vec![
                    GpuJointForceInput {
                        base_force: 0.3,
                        passive: JointPassive {
                            stiffness: 4.0,
                            damping: 0.5,
                            rest_position: 0.1,
                        },
                        nonlinear: JointNonlinearPassive {
                            spring_quadratic: 0.2,
                            spring_cubic: -0.1,
                            damping_quadratic: 0.1,
                            damping_cubic: 0.05,
                        },
                        motor: Some(JointMotor {
                            position_target: Some(0.0),
                            velocity_target: 0.2,
                            stiffness: 3.0,
                            damping: 0.7,
                            max_force: 0.8,
                        }),
                    },
                    GpuJointForceInput {
                        base_force: -0.2,
                        passive: JointPassive {
                            stiffness: 2.0,
                            damping: 0.3,
                            rest_position: -0.1,
                        },
                        ..GpuJointForceInput::default()
                    },
                ],
                joint_velocity_limit: None,
                coordinate_velocity_limits: None,
                link_loads: vec![
                    GpuMassLinkLoad::default(),
                    GpuMassLinkLoad {
                        torque: Vector3::new(0.0, 0.8, 0.1),
                        ..GpuMassLinkLoad::default()
                    },
                    GpuMassLinkLoad {
                        force: Vector3::new(0.5, -0.3, 0.2),
                        gravity_scale: 0.9,
                        ..GpuMassLinkLoad::default()
                    },
                ],
                ground_spheres: Vec::new(),
                ground_capsules: Vec::new(),
                ground_boxes: Vec::new(),
                ground_axial_shapes: Vec::new(),
                sphere_pairs: Vec::new(),
                static_sphere_pairs: Vec::new(),
                external_sphere_bodies: None,
                external_box_bodies: None,
                external_capsule_bodies: None,
                external_axial_bodies: None,
                external_convex_bodies: None,
                external_constraint_bodies: None,
                external_indexed_bodies: None,
                static_capsule_sphere_pairs: Vec::new(),
                static_box_sphere_pairs: Vec::new(),
                static_sphere_capsule_pairs: Vec::new(),
                static_capsule_pairs: Vec::new(),
                static_sphere_box_pairs: Vec::new(),
                static_capsule_box_pairs: Vec::new(),
                static_box_pairs: Vec::new(),
                static_box_capsule_pairs: Vec::new(),
                static_axial_sphere_pairs: Vec::new(),
                static_axial_capsule_pairs: Vec::new(),
                static_axial_box_pairs: Vec::new(),
                static_axial_convex_pairs: Vec::new(),
                axial_convex_pairs: Vec::new(),
                axial_sphere_pairs: Vec::new(),
                axial_box_pairs: Vec::new(),
                axial_capsule_pairs: Vec::new(),
                axial_pairs: Vec::new(),
                convex_sphere_pairs: Vec::new(),
                static_convex_sphere_pairs: Vec::new(),
                static_convex_capsule_pairs: Vec::new(),
                convex_capsule_pairs: Vec::new(),
                convex_pairs: Vec::new(),
                static_convex_pairs: Vec::new(),
                scene_convex_sphere_pairs: Vec::new(),
                scene_mesh_sphere_pairs: Vec::new(),
                scene_polyline_sphere_pairs: Vec::new(),
                scene_mesh_capsule_pairs: Vec::new(),
                scene_polyline_capsule_pairs: Vec::new(),
                scene_mesh_box_pairs: Vec::new(),
                scene_polyline_box_pairs: Vec::new(),
                scene_mesh_axial_pairs: Vec::new(),
                scene_polyline_axial_pairs: Vec::new(),
                scene_mesh_convex_pairs: Vec::new(),
                scene_polyline_convex_pairs: Vec::new(),
                scene_convex_capsule_pairs: Vec::new(),
                capsule_sphere_pairs: Vec::new(),
                capsule_pairs: Vec::new(),
                sphere_box_pairs: Vec::new(),
                capsule_box_pairs: Vec::new(),
                link_spheres: Vec::new(),
                link_capsules: Vec::new(),
                link_boxes: Vec::new(),
                box_pairs: Vec::new(),
                contact_iterations: 8,
                dynamic_material_rules: Vec::new(),
                ground_manifold_start: None,
                joint_friction: Vec::new(),
                joint_couplings: Vec::new(),
                link_point_constraints: Vec::new(),
                link_fixed_constraints: Vec::new(),
                contact_warm_start: false,
            },
            GpuArticulatedDynamicsInput {
                articulation: &slider,
                root_pose: Isometry3::translation(1.0, -0.5, 0.2),
                state: GpuGeneralizedState {
                    positions: DVector::from_element(1, 0.4),
                    velocities: DVector::from_element(1, -0.1),
                },
                gravity: Vector3::new(0.0, 0.0, -9.81),
                joints: vec![GpuJointForceInput {
                    base_force: 1.0,
                    passive: JointPassive {
                        stiffness: 6.0,
                        damping: 1.0,
                        rest_position: 0.25,
                    },
                    ..GpuJointForceInput::default()
                }],
                joint_velocity_limit: Some(0.15),
                coordinate_velocity_limits: None,
                link_loads: vec![
                    GpuMassLinkLoad::default(),
                    GpuMassLinkLoad {
                        force: Vector3::new(0.0, 0.0, 2.0),
                        ..GpuMassLinkLoad::default()
                    },
                ],
                ground_spheres: Vec::new(),
                ground_capsules: Vec::new(),
                ground_boxes: Vec::new(),
                ground_axial_shapes: Vec::new(),
                sphere_pairs: Vec::new(),
                static_sphere_pairs: Vec::new(),
                external_sphere_bodies: None,
                external_box_bodies: None,
                external_capsule_bodies: None,
                external_axial_bodies: None,
                external_convex_bodies: None,
                external_constraint_bodies: None,
                external_indexed_bodies: None,
                static_capsule_sphere_pairs: Vec::new(),
                static_box_sphere_pairs: Vec::new(),
                static_sphere_capsule_pairs: Vec::new(),
                static_capsule_pairs: Vec::new(),
                static_sphere_box_pairs: Vec::new(),
                static_capsule_box_pairs: Vec::new(),
                static_box_pairs: Vec::new(),
                static_box_capsule_pairs: Vec::new(),
                static_axial_sphere_pairs: Vec::new(),
                static_axial_capsule_pairs: Vec::new(),
                static_axial_box_pairs: Vec::new(),
                static_axial_convex_pairs: Vec::new(),
                axial_convex_pairs: Vec::new(),
                axial_sphere_pairs: Vec::new(),
                axial_box_pairs: Vec::new(),
                axial_capsule_pairs: Vec::new(),
                axial_pairs: Vec::new(),
                convex_sphere_pairs: Vec::new(),
                static_convex_sphere_pairs: Vec::new(),
                static_convex_capsule_pairs: Vec::new(),
                convex_capsule_pairs: Vec::new(),
                convex_pairs: Vec::new(),
                static_convex_pairs: Vec::new(),
                scene_convex_sphere_pairs: Vec::new(),
                scene_mesh_sphere_pairs: Vec::new(),
                scene_polyline_sphere_pairs: Vec::new(),
                scene_mesh_capsule_pairs: Vec::new(),
                scene_polyline_capsule_pairs: Vec::new(),
                scene_mesh_box_pairs: Vec::new(),
                scene_polyline_box_pairs: Vec::new(),
                scene_mesh_axial_pairs: Vec::new(),
                scene_polyline_axial_pairs: Vec::new(),
                scene_mesh_convex_pairs: Vec::new(),
                scene_polyline_convex_pairs: Vec::new(),
                scene_convex_capsule_pairs: Vec::new(),
                capsule_sphere_pairs: Vec::new(),
                capsule_pairs: Vec::new(),
                sphere_box_pairs: Vec::new(),
                capsule_box_pairs: Vec::new(),
                link_spheres: Vec::new(),
                link_capsules: Vec::new(),
                link_boxes: Vec::new(),
                box_pairs: Vec::new(),
                contact_iterations: 8,
                dynamic_material_rules: Vec::new(),
                ground_manifold_start: None,
                joint_friction: Vec::new(),
                joint_couplings: Vec::new(),
                link_point_constraints: Vec::new(),
                link_fixed_constraints: Vec::new(),
                contact_warm_start: false,
            },
        ];
        let dt = 0.01;
        let mut reference: [GpuGeneralizedState; 2] =
            core::array::from_fn(|index| inputs[index].state.clone());
        for _ in 0..3 {
            for (input, state) in inputs.iter().zip(&mut reference) {
                cpu_step(input, state, dt);
            }
        }
        assert!((reference[1].positions[0] - 0.399).abs() < 1e-12);
        assert!(reference[1].velocities[0].abs() < 1e-12);
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let mut batch = GpuArticulatedDynamicsBatch::new(&context, &inputs, dt).unwrap();
            batch.submit_steps(3).unwrap();
            let actual = batch.readback().unwrap();
            for (actual, expected) in actual.iter().zip(&reference) {
                assert!((&actual.positions - &expected.positions).norm() < 3e-3);
                assert!((&actual.velocities - &expected.velocities).norm() < 3e-3);
            }
            let poses = batch.readback_link_poses().unwrap();
            let expected_pose = serial
                .pose(inputs[0].root_pose, reference[0].positions.as_slice())
                .unwrap();
            assert!(
                (poses[0][2].translation.vector - expected_pose.links[2].translation.vector).norm()
                    < 1e-4
            );
            batch
                .reset(
                    &inputs
                        .iter()
                        .map(|input| input.state.clone())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            batch.submit_steps(3).unwrap();
            let repeated = batch.readback().unwrap();
            for (actual, expected) in repeated.iter().zip(&reference) {
                assert!((&actual.positions - &expected.positions).norm() < 3e-3);
                assert!((&actual.velocities - &expected.velocities).norm() < 3e-3);
            }
            let mut changed = inputs.clone();
            changed[0].root_pose = Isometry3::rotation(Vector3::new(-0.1, 0.2, 0.05));
            changed[0].joints[0].base_force = -0.4;
            changed[0].link_loads[1].torque = Vector3::new(0.0, -0.3, 0.2);
            changed[1].gravity.z = -8.0;
            changed[1].joint_velocity_limit = Some(0.05);
            let mut changed_reference: [GpuGeneralizedState; 2] =
                core::array::from_fn(|index| changed[index].state.clone());
            for _ in 0..2 {
                for (input, state) in changed.iter().zip(&mut changed_reference) {
                    cpu_step(input, state, dt);
                }
            }
            batch
                .reset(
                    &changed
                        .iter()
                        .map(|input| input.state.clone())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            batch
                .update_joints(
                    &changed
                        .iter()
                        .map(|input| input.joints.clone())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            batch
                .update_gravity(
                    &changed
                        .iter()
                        .map(|input| input.gravity)
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            batch
                .update_link_loads(
                    &changed
                        .iter()
                        .map(|input| input.link_loads.clone())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            batch
                .update_velocity_limits(
                    &changed
                        .iter()
                        .map(|input| input.joint_velocity_limit)
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            batch.set_root_pose(0, changed[0].root_pose).unwrap();
            batch.submit_steps(2).unwrap();
            let updated = batch.readback().unwrap();
            for (actual, expected) in updated.iter().zip(&changed_reference) {
                assert!((&actual.positions - &expected.positions).norm() < 3e-3);
                assert!((&actual.velocities - &expected.velocities).norm() < 3e-3);
            }
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn resident_supported_sleep_freezes_and_external_load_wakes() {
        let articulation = Articulation::new(
            vec![link(0.0), link(1.0)],
            vec![JointSpec {
                parent: 0,
                child: 1,
                kind: JointKind::Prismatic,
                origin: Isometry3::identity(),
                axis: Vector3::z(),
                limits: None,
            }],
            0,
        )
        .unwrap();
        let state = GpuGeneralizedState {
            positions: DVector::zeros(1),
            velocities: DVector::zeros(1),
        };
        let mut input = GpuArticulatedDynamicsInput::new(
            &articulation,
            Isometry3::translation(0.0, 0.0, 0.5),
            state.clone(),
            Vector3::new(0.0, 0.0, -9.81),
        );
        input.ground_spheres.push(GpuArticulatedGroundSphere {
            link: 1,
            local_center: Vector3::zeros(),
            radius: 0.5,
            plane_normal: Vector3::z(),
            plane_offset: 0.0,
            plane_xy_half_extent: None,
            restitution: 0.0,
            friction: 0.0,
        });
        let settings = crate::sleep::SleepSettings {
            time_threshold: 0.002,
            linear_velocity_threshold: 0.01,
            angular_velocity_threshold: 0.01,
            ..Default::default()
        };
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let mut batch =
                GpuArticulatedDynamicsBatch::new(&context, &[input.clone()], 0.001).unwrap();
            let external_spheres = batch.readback_external_sphere_orbits().unwrap().spheres;
            assert_eq!(external_spheres.len(), 1);
            assert!(external_spheres[0].is_empty());
            assert!(batch.enable_contact_sleep(&[Some(settings)]).is_err());
            batch.enable_contact_activity().unwrap();
            let per_link = vec![None, Some(settings)];
            batch
                .enable_contact_link_sleep(core::slice::from_ref(&per_link))
                .unwrap();
            assert!(
                batch
                    .update_contact_idle_requires_static_support(false)
                    .is_err()
            );
            assert!(batch.update_contact_loss_wake(false).is_err());
            assert!(batch.update_contact_motion_wake(&[None]).is_err());
            batch.submit_steps(1).unwrap();
            let before_sleep = batch.readback().unwrap()[0].clone();
            batch.submit_steps(1).unwrap();
            assert_eq!(
                batch.readback_contact_sleep_candidates().unwrap()[0],
                [false, true]
            );
            let sleeping = batch.readback().unwrap()[0].clone();
            assert_eq!(sleeping.positions, before_sleep.positions);
            assert!(sleeping.velocities.iter().all(|v| *v == 0.0));
            let read_flags = |batch: &GpuArticulatedDynamicsBatch| {
                crate::gpu_articulated_mass::read_buffer(
                    &batch.device,
                    &batch.queue,
                    batch.sleep_freeze.as_ref().unwrap().frozen_buffer(),
                )
                .unwrap()
            };
            assert_eq!(read_flags(&batch), 1u32.to_ne_bytes());
            assert_eq!(batch.readback_sleeping_coordinates().unwrap(), [vec![true]]);
            let before_split = batch.readback().unwrap();
            let mut encoder = batch.device.create_command_encoder(&Default::default());
            batch.encode_contact_step(&mut encoder).unwrap();
            batch.encode_integration(&mut encoder);
            let _ = batch.queue.submit(Some(encoder.finish()));
            assert_eq!(batch.readback().unwrap(), before_split);
            assert_eq!(batch.readback_sleeping_coordinates().unwrap(), [vec![true]]);
            assert!(batch.update_contact_link_idle_settings(&[]).is_err());
            assert!(
                batch
                    .update_contact_link_idle_settings(&[vec![Some(settings)]])
                    .is_err()
            );
            let mut invalid_link = per_link.clone();
            invalid_link[1] = Some(crate::sleep::SleepSettings {
                angular_velocity_threshold: f64::NAN,
                ..settings
            });
            assert!(
                batch
                    .update_contact_link_idle_settings(&[invalid_link])
                    .is_err()
            );
            assert_eq!(batch.readback_sleeping_coordinates().unwrap(), [vec![true]]);
            let invalid = crate::sleep::SleepSettings {
                time_threshold: f64::NAN,
                ..settings
            };
            assert!(batch.enable_contact_sleep(&[Some(invalid)]).is_err());
            assert_eq!(read_flags(&batch), 1u32.to_ne_bytes());
            batch.reset(core::slice::from_ref(&state)).unwrap();
            assert_eq!(read_flags(&batch), 0u32.to_ne_bytes());
            batch.submit_steps(2).unwrap();
            assert_eq!(read_flags(&batch), 1u32.to_ne_bytes());
            let mut invalid_joints = input.joints.clone();
            invalid_joints[0].base_force = f64::NAN;
            assert!(
                batch
                    .update_joints(core::slice::from_ref(&invalid_joints))
                    .is_err()
            );
            assert_eq!(batch.readback_sleeping_coordinates().unwrap(), [vec![true]]);
            let mut changed = input.joints.clone();
            changed[0].motor = Some(JointMotor {
                position_target: Some(1.0),
                velocity_target: 0.0,
                stiffness: 0.0,
                damping: 0.0,
                max_force: 0.0,
            });
            batch
                .update_joints(core::slice::from_ref(&changed))
                .unwrap();
            batch.submit_steps(1).unwrap();
            assert_eq!(
                batch.readback_contact_wake_requests().unwrap()[0],
                [false, true]
            );
            assert_eq!(read_flags(&batch), 0u32.to_ne_bytes());
            batch
                .update_joints(core::slice::from_ref(&changed))
                .unwrap();
            batch.submit_steps(2).unwrap();
            assert_eq!(read_flags(&batch), 1u32.to_ne_bytes());
            // A persistent represented drive effort must still wake after the
            // one-shot configuration request has been consumed.
            changed[0].base_force = 1e-30;
            batch
                .update_joints(core::slice::from_ref(&changed))
                .unwrap();
            batch.submit_steps(3).unwrap();
            assert_eq!(
                batch.readback_contact_wake_requests().unwrap()[0],
                [false, true]
            );
            assert_eq!(read_flags(&batch), 0u32.to_ne_bytes());
            changed[0].base_force = 0.0;
            batch
                .update_joints(core::slice::from_ref(&changed))
                .unwrap();
            batch.submit_steps(3).unwrap();
            assert_eq!(read_flags(&batch), 1u32.to_ne_bytes());
            let mut loads = vec![GpuMassLinkLoad::default(); 2];
            loads[1].force = Vector3::z() * 20.0;
            batch.update_link_loads(&[loads]).unwrap();
            batch.submit_steps(1).unwrap();
            assert_eq!(read_flags(&batch), 0u32.to_ne_bytes());
            assert!(batch.readback().unwrap()[0].velocities[0] > 0.0);
            batch.disable_contact_sleep();
            assert!(batch.readback_sleeping_coordinates().is_err());
            assert!(
                batch
                    .update_contact_idle_requires_static_support(false)
                    .is_ok()
            );
            eprintln!("resident supported sleep and load wake passed on {backend:?}");
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn resident_static_sphere_center_update_wakes_without_rebuilding() {
        let articulation = Articulation::new(
            vec![link(0.0), link(1.0)],
            vec![JointSpec {
                parent: 0,
                child: 1,
                kind: JointKind::Prismatic,
                origin: Isometry3::identity(),
                axis: Vector3::z(),
                limits: None,
            }],
            0,
        )
        .unwrap();
        let state = GpuGeneralizedState {
            positions: DVector::zeros(1),
            velocities: DVector::zeros(1),
        };
        let mut input = GpuArticulatedDynamicsInput::new(
            &articulation,
            Isometry3::translation(0.0, 0.0, 0.5),
            state,
            Vector3::new(0.0, 0.0, -9.81),
        );
        let center = Vector3::new(0.0, 0.0, -0.5);
        input
            .static_sphere_pairs
            .push(GpuArticulatedStaticSpherePair {
                link: 1,
                local_center: Vector3::zeros(),
                radius: 0.5,
                static_center: center,
                static_radius: 0.5,
                restitution: 0.0,
                friction: 0.0,
            });
        let settings = crate::sleep::SleepSettings {
            time_threshold: 0.002,
            linear_velocity_threshold: 100.0,
            angular_velocity_threshold: 100.0,
            ..Default::default()
        };
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let mut batch =
                GpuArticulatedDynamicsBatch::new(&context, &[input.clone()], 0.001).unwrap();
            batch.enable_contact_activity().unwrap();
            batch.enable_contact_sleep(&[Some(settings)]).unwrap();
            batch.submit_steps(2).unwrap();
            assert_eq!(batch.readback_sleeping_coordinates().unwrap(), [vec![true]]);
            assert!(!batch.update_static_sphere_centers(&[vec![center]]).unwrap());
            assert!(batch.update_static_sphere_centers(&[]).is_err());
            assert!(
                batch
                    .update_static_sphere_centers(&[vec![Vector3::new(f64::MAX, 0.0, 0.0)]])
                    .is_err()
            );
            assert_eq!(batch.readback_sleeping_coordinates().unwrap(), [vec![true]]);
            batch
                .update_contact_wake_requests(&[vec![true, false]])
                .unwrap();
            assert!(
                batch
                    .update_static_sphere_centers(&[vec![Vector3::new(2.0, 0.0, -0.5)]])
                    .unwrap()
            );
            assert_eq!(
                batch.readback_sleeping_coordinates().unwrap(),
                [vec![false]]
            );
            batch.submit_steps(1).unwrap();
            assert_eq!(
                batch.readback_contact_wake_requests().unwrap()[0],
                [true, true]
            );
            assert_eq!(
                batch.readback_sleeping_coordinates().unwrap(),
                [vec![false]]
            );
            assert!(batch.readback().unwrap()[0].velocities[0] < 0.0);
            assert_eq!(batch.readback_geometric_contacts().unwrap()[0], [false; 2]);
            batch.submit_steps(1).unwrap();
            assert_eq!(
                batch.readback_contact_wake_requests().unwrap()[0],
                [false; 2]
            );
            eprintln!("resident static sphere center update and wake passed on {backend:?}");
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn resident_prescribed_mesh_translates_and_stops_without_partial_upload() {
        let context = GpuContactDevice::new_with_backends(wgpu::Backends::VULKAN).unwrap();
        let articulation = Articulation::new(
            vec![link(0.0), link(1.0)],
            vec![JointSpec {
                parent: 0,
                child: 1,
                kind: JointKind::Prismatic,
                origin: Isometry3::identity(),
                axis: Vector3::z(),
                limits: None,
            }],
            0,
        )
        .unwrap();
        let mut input = GpuArticulatedDynamicsInput::new(
            &articulation,
            Isometry3::identity(),
            GpuGeneralizedState {
                positions: DVector::zeros(1),
                velocities: DVector::zeros(1),
            },
            -Vector3::z() * 0.1,
        );
        input
            .scene_mesh_sphere_pairs
            .push(GpuArticulatedSceneMeshSpherePair {
                mesh_world_pose: Isometry3::identity(),
                mesh: std::sync::Arc::new(
                    crate::mesh::TriangleMeshGeometry::new(
                        vec![
                            Vector3::new(-5.0, -5.0, 0.0),
                            Vector3::new(5.0, -5.0, 0.0),
                            Vector3::new(0.0, 5.0, 0.0),
                        ],
                        vec![[0, 1, 2]],
                    )
                    .unwrap(),
                ),
                sphere_link: 1,
                sphere_local_center: Vector3::z() * 0.5,
                sphere_radius: 0.5,
                restitution: 0.0,
                friction: 0.0,
            });
        input
            .scene_polyline_sphere_pairs
            .push(GpuArticulatedScenePolylineSpherePair {
                polyline_world_pose: Isometry3::translation(0.0, 0.0, -1.0),
                polyline: std::sync::Arc::new(
                    crate::mesh::PolylineGeometry::new(
                        vec![-Vector3::x(), Vector3::x()],
                        vec![[0, 1]],
                    )
                    .unwrap(),
                ),
                sphere_link: 1,
                sphere_local_center: Vector3::z() * 0.5,
                sphere_radius: 0.5,
                restitution: 0.0,
                friction: 0.0,
            });
        let body = crate::gpu_kinematic_body::GpuKinematicBody {
            pose: Isometry3::identity(),
            linear_velocity: Vector3::z() * 0.2,
            angular_velocity: Vector3::zeros(),
        };
        let polyline = crate::gpu_kinematic_body::GpuKinematicBody {
            pose: Isometry3::translation(0.0, 0.0, -1.0),
            linear_velocity: -Vector3::z() * 0.1,
            angular_velocity: Vector3::zeros(),
        };
        let stationary = input.clone();
        input.external_indexed_bodies = Some(vec![Some(body.clone()), Some(polyline)]);
        let mut batch =
            GpuArticulatedDynamicsBatch::new(&context, &[input, stationary], 0.001).unwrap();
        batch.enable_contact_activity().unwrap();
        let sleep = crate::sleep::SleepSettings {
            time_threshold: 0.002,
            linear_velocity_threshold: 100.0,
            angular_velocity_threshold: 100.0,
            ..Default::default()
        };
        batch
            .enable_contact_sleep(&[Some(sleep), Some(sleep)])
            .unwrap();
        batch.submit_steps(100).unwrap();
        assert_eq!(
            batch.readback_sleeping_coordinates().unwrap(),
            [vec![false], vec![true]]
        );

        let poses = batch.readback_prescribed_indexed_poses().unwrap();
        assert_eq!(poses.iter().map(Vec::len).collect::<Vec<_>>(), [2, 2]);
        assert!(
            (poses[0][1].translation.vector.z + 1.01).abs() < 1e-5,
            "{poses:?}"
        );
        assert!((poses[1][1].translation.vector.z + 1.0).abs() < 1e-6);
        let state = batch.readback().unwrap();
        assert!((poses[0][0].translation.vector.z - 0.02).abs() < 1e-5);
        assert!(poses[1][0].translation.vector.norm() < 1e-6);
        assert!((state[0].positions[0] - 0.02).abs() < 1e-5, "{state:?}");
        assert!((state[0].velocities[0] - 0.2).abs() < 1e-5, "{state:?}");
        assert!(state[1].positions[0].abs() < 1e-6);
        let mut invalid = body.clone();
        invalid.linear_velocity.x = f64::NAN;
        assert!(
            batch
                .update_prescribed_indexed_bodies(&[
                    vec![Some(body), None],
                    vec![Some(invalid), None]
                ])
                .is_err()
        );
        assert_eq!(batch.readback_prescribed_indexed_poses().unwrap(), poses);
        batch
            .update_prescribed_indexed_bodies(&[vec![None, None], vec![None, None]])
            .unwrap();
        batch.submit_steps(10).unwrap();
        assert_eq!(batch.readback_prescribed_indexed_poses().unwrap(), poses);
        // The separating sphere keeps its momentum while gravity decelerates it.
        let stopped = batch.readback().unwrap();
        assert!((stopped[0].velocities[0] - 0.199).abs() < 1e-5);
        assert!((stopped[0].positions[0] - 0.022).abs() < 1e-5);
        let mut changed = poses.clone();
        changed[0][0].translation.vector.x = 0.25;
        changed[0][1].translation.vector.x = -0.25;
        let mut invalid = changed.clone();
        invalid[1][1].translation.vector.z = f64::NAN;
        assert!(batch.update_indexed_geometry_poses(&invalid).is_err());
        assert_eq!(batch.readback_prescribed_indexed_poses().unwrap(), poses);
        batch.update_indexed_geometry_poses(&changed).unwrap();
        assert_eq!(batch.readback_prescribed_indexed_poses().unwrap(), changed);
        batch.submit_steps(1).unwrap();
        assert_eq!(batch.readback_prescribed_indexed_poses().unwrap(), changed);
        batch.submit_steps(10).unwrap();
        assert!(batch.readback_sleeping_coordinates().unwrap()[1][0]);
        let wake_motion = crate::gpu_kinematic_body::GpuKinematicBody {
            pose: Isometry3::identity(),
            linear_velocity: Vector3::z() * 0.1,
            angular_velocity: Vector3::zeros(),
        };
        batch
            .update_prescribed_indexed_bodies(&[vec![None, None], vec![Some(wake_motion), None]])
            .unwrap();
        batch.submit_steps(10).unwrap();
        assert!(!batch.readback_sleeping_coordinates().unwrap()[1][0]);
        let awakened = batch.readback().unwrap();
        assert!(
            (awakened[1].positions[0] - 0.001).abs() < 1e-5,
            "{awakened:?}"
        );
        assert!(
            (awakened[1].velocities[0] - 0.1).abs() < 1e-5,
            "{awakened:?}"
        );
        assert!(
            (batch.readback_prescribed_indexed_poses().unwrap()[1][0]
                .translation
                .vector
                .z
                - 0.001)
                .abs()
                < 1e-5
        );
    }

    #[test]
    fn resident_prescribed_indexed_rotation_tracks_contact_velocity_and_geometry_pose() {
        let context = GpuContactDevice::new_with_backends(wgpu::Backends::VULKAN).unwrap();
        let articulation = Articulation::new(
            vec![link(0.0), link(1.0)],
            vec![JointSpec {
                parent: 0,
                child: 1,
                kind: JointKind::Prismatic,
                origin: Isometry3::identity(),
                axis: Vector3::z(),
                limits: None,
            }],
            0,
        )
        .unwrap();
        let mut input = GpuArticulatedDynamicsInput::new(
            &articulation,
            Isometry3::identity(),
            GpuGeneralizedState {
                positions: DVector::zeros(1),
                velocities: DVector::zeros(1),
            },
            Vector3::zeros(),
        );
        input
            .scene_mesh_sphere_pairs
            .push(GpuArticulatedSceneMeshSpherePair {
                mesh_world_pose: Isometry3::identity(),
                mesh: std::sync::Arc::new(
                    crate::mesh::TriangleMeshGeometry::new(
                        vec![
                            Vector3::new(-5.0, -5.0, 0.0),
                            Vector3::new(5.0, -5.0, 0.0),
                            Vector3::new(0.0, 5.0, 0.0),
                        ],
                        vec![[0, 1, 2]],
                    )
                    .unwrap(),
                ),
                sphere_link: 1,
                sphere_local_center: Vector3::new(1.0, 0.0, 0.5),
                sphere_radius: 0.5,
                restitution: 0.0,
                friction: 0.0,
            });
        input
            .scene_polyline_sphere_pairs
            .push(GpuArticulatedScenePolylineSpherePair {
                polyline_world_pose: Isometry3::identity(),
                polyline: std::sync::Arc::new(
                    crate::mesh::PolylineGeometry::new(
                        vec![-Vector3::x(), Vector3::x()],
                        vec![[0, 1]],
                    )
                    .unwrap(),
                ),
                sphere_link: 1,
                sphere_local_center: Vector3::new(1.0, 0.0, 0.5),
                sphere_radius: 0.5,
                restitution: 0.0,
                friction: 0.0,
            });
        let body = crate::gpu_kinematic_body::GpuKinematicBody {
            pose: Isometry3::identity(),
            linear_velocity: Vector3::zeros(),
            angular_velocity: -Vector3::y() * 0.2,
        };
        let mut mesh = input.clone();
        mesh.scene_polyline_sphere_pairs.clear();
        mesh.external_indexed_bodies = Some(vec![Some(body.clone())]);
        let mut polyline = input;
        polyline.scene_mesh_sphere_pairs.clear();
        polyline.external_indexed_bodies = Some(vec![Some(body.clone())]);
        let mut offset_body = body;
        offset_body.pose.translation.vector.x = -0.5;
        let mut offset_mesh = mesh.clone();
        offset_mesh.external_indexed_bodies = Some(vec![Some(offset_body.clone())]);
        let mut offset_polyline = polyline.clone();
        offset_polyline.external_indexed_bodies = Some(vec![Some(offset_body)]);
        let mut batch = GpuArticulatedDynamicsBatch::new(
            &context,
            &[mesh, polyline, offset_mesh, offset_polyline],
            0.001,
        )
        .unwrap();
        batch.submit_steps(1).unwrap();
        for (state, speed) in batch.readback().unwrap().iter().zip([0.2, 0.2, 0.3, 0.3]) {
            assert!((state.velocities[0] - speed).abs() < 1e-5, "{state:?}");
        }
        batch.submit_steps(99).unwrap();
        let poses = batch.readback_prescribed_indexed_poses().unwrap();

        for (environment, (pose, state)) in poses.iter().zip(batch.readback().unwrap()).enumerate()
        {
            let lever = if environment < 2 { 1.0 } else { 1.5 };
            let expected_position = if environment % 2 == 0 {
                0.5 / 0.02_f64.cos() + lever * 0.02_f64.tan() - 0.5
            } else {
                // The finite rotating segment touches at its endpoint.
                let horizontal = lever * (1.0 - 0.02_f64.cos());
                lever * 0.02_f64.sin() + (0.25 - horizontal * horizontal).sqrt() - 0.5
            };
            let expected_origin = if environment < 2 {
                Vector3::zeros()
            } else {
                Vector3::new(-0.5 + 0.5 * 0.02_f64.cos(), 0.0, 0.5 * 0.02_f64.sin())
            };
            assert!((pose[0].translation.vector - expected_origin).norm() < 1e-5);
            assert!((pose[0].rotation.scaled_axis().y + 0.02).abs() < 1e-5);
            assert!(
                (state.positions[0] - expected_position).abs() < 1e-5,
                "{state:?}"
            );
        }
        batch
            .update_prescribed_indexed_bodies(&[vec![None], vec![None], vec![None], vec![None]])
            .unwrap();
        batch.submit_steps(10).unwrap();
        assert_eq!(batch.readback_prescribed_indexed_poses().unwrap(), poses);
    }

    #[test]
    fn resident_rotating_indexed_pose_update_preserves_contact_point_velocity() {
        let context = GpuContactDevice::new_with_backends(wgpu::Backends::VULKAN).unwrap();
        let articulation = Articulation::new(
            vec![link(0.0), link(1.0)],
            vec![JointSpec {
                parent: 0,
                child: 1,
                kind: JointKind::Prismatic,
                origin: Isometry3::identity(),
                axis: Vector3::z(),
                limits: None,
            }],
            0,
        )
        .unwrap();
        let mut input = GpuArticulatedDynamicsInput::new(
            &articulation,
            Isometry3::identity(),
            GpuGeneralizedState {
                positions: DVector::zeros(1),
                velocities: DVector::zeros(1),
            },
            Vector3::zeros(),
        );
        input
            .scene_mesh_sphere_pairs
            .push(GpuArticulatedSceneMeshSpherePair {
                mesh_world_pose: Isometry3::identity(),
                mesh: std::sync::Arc::new(
                    crate::mesh::TriangleMeshGeometry::new(
                        vec![
                            Vector3::new(-5.0, -5.0, 0.0),
                            Vector3::new(5.0, -5.0, 0.0),
                            Vector3::new(0.0, 5.0, 0.0),
                        ],
                        vec![[0, 1, 2]],
                    )
                    .unwrap(),
                ),
                sphere_link: 1,
                sphere_local_center: Vector3::new(1.0, 0.0, 0.5),
                sphere_radius: 0.5,
                restitution: 0.0,
                friction: 0.0,
            });
        input
            .scene_polyline_sphere_pairs
            .push(GpuArticulatedScenePolylineSpherePair {
                polyline_world_pose: Isometry3::identity(),
                polyline: std::sync::Arc::new(
                    crate::mesh::PolylineGeometry::new(
                        vec![-Vector3::x(), Vector3::x()],
                        vec![[0, 1]],
                    )
                    .unwrap(),
                ),
                sphere_link: 1,
                sphere_local_center: Vector3::new(1.0, 0.0, 0.5),
                sphere_radius: 0.5,
                restitution: 0.0,
                friction: 0.0,
            });
        let body = crate::gpu_kinematic_body::GpuKinematicBody {
            pose: Isometry3::identity(),
            linear_velocity: Vector3::zeros(),
            angular_velocity: -Vector3::y() * 0.2,
        };
        let mut mesh = input.clone();
        mesh.scene_polyline_sphere_pairs.clear();
        mesh.external_indexed_bodies = Some(vec![Some(body.clone())]);
        let mut polyline = input;
        polyline.scene_mesh_sphere_pairs.clear();
        polyline.external_indexed_bodies = Some(vec![Some(body.clone())]);
        let mut offset_body = body;
        offset_body.pose.translation.vector.x = -0.5;
        let mut offset_mesh = mesh.clone();
        offset_mesh.external_indexed_bodies = Some(vec![Some(offset_body.clone())]);
        let mut offset_polyline = polyline.clone();
        offset_polyline.external_indexed_bodies = Some(vec![Some(offset_body)]);
        let mut batch = GpuArticulatedDynamicsBatch::new(
            &context,
            &[mesh, polyline, offset_mesh, offset_polyline],
            0.001,
        )
        .unwrap();
        let mut changed = batch.readback_prescribed_indexed_poses().unwrap();
        for environment in &mut changed {
            environment[0].translation.vector.x += 0.25;
        }
        batch.update_indexed_geometry_poses(&changed).unwrap();
        batch.submit_steps(1).unwrap();
        for (state, speed) in batch.readback().unwrap().iter().zip([0.2, 0.2, 0.3, 0.3]) {
            assert!((state.velocities[0] - speed).abs() < 1e-5, "{state:?}");
        }
        batch.submit_steps(99).unwrap();
        let poses = batch.readback_prescribed_indexed_poses().unwrap();

        for (environment, (pose, state)) in poses.iter().zip(batch.readback().unwrap()).enumerate()
        {
            let lever = if environment < 2 { 1.0 } else { 1.5 };
            // The translated segment now covers the contact inside its endpoints.
            let expected_position = 0.5 / 0.02_f64.cos() + lever * 0.02_f64.tan() - 0.5;
            let expected_origin = if environment < 2 {
                Vector3::new(0.25 * 0.02_f64.cos(), 0.0, 0.25 * 0.02_f64.sin())
            } else {
                Vector3::new(-0.5 + 0.75 * 0.02_f64.cos(), 0.0, 0.75 * 0.02_f64.sin())
            };
            assert!((pose[0].translation.vector - expected_origin).norm() < 1e-5);
            assert!((pose[0].rotation.scaled_axis().y + 0.02).abs() < 1e-5);
            assert!(
                (state.positions[0] - expected_position).abs() < 1e-5,
                "{state:?}"
            );
        }
        batch
            .update_prescribed_indexed_bodies(&[vec![None], vec![None], vec![None], vec![None]])
            .unwrap();
        batch.submit_steps(10).unwrap();
        assert_eq!(batch.readback_prescribed_indexed_poses().unwrap(), poses);
    }

    #[test]
    fn resident_prescribed_point_anchor_translates_and_stops() {
        let context = GpuContactDevice::new_with_backends(wgpu::Backends::VULKAN).unwrap();
        let articulation = Articulation::new(
            vec![link(0.0), link(1.0)],
            vec![JointSpec {
                parent: 0,
                child: 1,
                kind: JointKind::Prismatic,
                origin: Isometry3::identity(),
                axis: Vector3::z(),
                limits: None,
            }],
            0,
        )
        .unwrap();
        let mut input = GpuArticulatedDynamicsInput::new(
            &articulation,
            Isometry3::identity(),
            GpuGeneralizedState {
                positions: DVector::zeros(1),
                velocities: DVector::zeros(1),
            },
            Vector3::zeros(),
        );
        input.link_point_constraints.push(LinkPointConstraint {
            link_a: 1,
            point_a: [0.0; 3],
            link_b: None,
            point_b: [0.0; 3],
        });
        input.external_constraint_bodies =
            Some(vec![Some(crate::gpu_kinematic_body::GpuKinematicBody {
                pose: Isometry3::identity(),
                linear_velocity: Vector3::z() * 0.2,
                angular_velocity: Vector3::zeros(),
            })]);
        let mut batch = GpuArticulatedDynamicsBatch::new(&context, &[input], 0.001).unwrap();
        batch.submit_steps(100).unwrap();
        let anchor = batch.readback_prescribed_constraint_frames().unwrap()[0][0];
        assert!((anchor.translation.vector.z - 0.02).abs() < 1e-5);
        let state = batch.readback().unwrap().remove(0);
        assert!((state.positions[0] - 0.02).abs() < 1e-5, "{state:?}");
        assert!((state.velocities[0] - 0.2).abs() < 1e-5, "{state:?}");
        batch
            .update_prescribed_constraint_bodies(&[vec![None]])
            .unwrap();
        batch.submit_steps(10).unwrap();
        let stopped_anchor = batch.readback_prescribed_constraint_frames().unwrap()[0][0];
        assert!((stopped_anchor.translation.vector - anchor.translation.vector).norm() < 1e-6);
        let stopped = batch.readback().unwrap().remove(0);
        assert!(
            (stopped.positions[0] - state.positions[0]).abs() < 1e-5,
            "{stopped:?}"
        );
        assert!(stopped.velocities[0].abs() < 1e-5, "{stopped:?}");
    }

    #[test]
    fn resident_immovable_point_anchor_rejects_prescribed_velocity_immediately() {
        let context = GpuContactDevice::new_with_backends(wgpu::Backends::VULKAN).unwrap();
        let articulation = Articulation::new(
            vec![link(0.0), link(1.0)],
            vec![JointSpec {
                parent: 0,
                child: 1,
                kind: JointKind::Prismatic,
                origin: Isometry3::identity(),
                axis: Vector3::z(),
                limits: None,
            }],
            0,
        )
        .unwrap();
        let mut input = GpuArticulatedDynamicsInput::new(
            &articulation,
            Isometry3::identity(),
            GpuGeneralizedState {
                positions: DVector::zeros(1),
                velocities: DVector::zeros(1),
            },
            Vector3::zeros(),
        );
        input.link_point_constraints.push(LinkPointConstraint {
            link_a: 0,
            point_a: [0.0; 3],
            link_b: None,
            point_b: [0.0; 3],
        });
        input.external_constraint_bodies =
            Some(vec![Some(crate::gpu_kinematic_body::GpuKinematicBody {
                pose: Isometry3::identity(),
                linear_velocity: Vector3::z() * 0.2,
                angular_velocity: Vector3::zeros(),
            })]);
        let mut batch = GpuArticulatedDynamicsBatch::new(&context, &[input], 0.001).unwrap();
        batch.submit_steps(1).unwrap();
        assert!(batch.readback().is_err());
        assert!(batch.readback_prescribed_constraint_frames().is_err());
        assert!(
            batch
                .update_external_constraint_frames(&[vec![Isometry3::identity()]])
                .is_err()
        );
        batch
            .reset(&[GpuGeneralizedState {
                positions: DVector::zeros(1),
                velocities: DVector::zeros(1),
            }])
            .unwrap();
        let frame = batch.readback_prescribed_constraint_frames().unwrap()[0][0];
        assert!(frame.translation.vector.norm() < 1e-6);
        batch
            .update_prescribed_constraint_bodies(&[vec![None]])
            .unwrap();
        batch.submit_steps(10).unwrap();
        let recovered = batch.readback().unwrap().remove(0);
        assert!(recovered.positions[0].abs() < 1e-6);
        assert!(recovered.velocities[0].abs() < 1e-6);
        assert!(
            batch.readback_prescribed_constraint_frames().unwrap()[0][0]
                .translation
                .vector
                .norm()
                < 1e-6
        );
    }

    #[test]
    fn resident_prescribed_anchor_initialization_preserves_mixed_order() {
        let context = GpuContactDevice::new_with_backends(wgpu::Backends::VULKAN).unwrap();
        let articulation = Articulation::new(
            vec![link(0.0), link(1.0), link(0.0)],
            vec![
                JointSpec {
                    parent: 0,
                    child: 1,
                    kind: JointKind::Prismatic,
                    origin: Isometry3::identity(),
                    axis: Vector3::z(),
                    limits: None,
                },
                JointSpec {
                    parent: 0,
                    child: 2,
                    kind: JointKind::Fixed,
                    origin: Isometry3::identity(),
                    axis: Vector3::z(),
                    limits: None,
                },
            ],
            0,
        )
        .unwrap();
        let mut input = GpuArticulatedDynamicsInput::new(
            &articulation,
            Isometry3::identity(),
            GpuGeneralizedState {
                positions: DVector::zeros(1),
                velocities: DVector::zeros(1),
            },
            Vector3::zeros(),
        );
        input.link_point_constraints.push(LinkPointConstraint {
            link_a: 0,
            point_a: [0.0; 3],
            link_b: Some(2),
            point_b: [0.0; 3],
        });
        input.link_point_constraints.push(LinkPointConstraint {
            link_a: 1,
            point_a: [0.0; 3],
            link_b: None,
            point_b: [0.0; 3],
        });
        let stationary = input.clone();
        input.link_fixed_constraints.push(LinkFixedConstraint {
            link_a: 0,
            frame_a: Isometry3::identity(),
            link_b: None,
            frame_b: Isometry3::identity(),
        });
        input.external_constraint_bodies = Some(vec![
            Some(crate::gpu_kinematic_body::GpuKinematicBody {
                pose: Isometry3::identity(),
                linear_velocity: Vector3::z() * 0.2,
                angular_velocity: Vector3::zeros(),
            }),
            None,
        ]);
        let mut batch =
            GpuArticulatedDynamicsBatch::new(&context, &[input, stationary], 0.001).unwrap();
        batch.submit_steps(100).unwrap();
        let frames = batch.readback_prescribed_constraint_frames().unwrap();
        assert_eq!(frames[0].len(), 2);
        assert_eq!(frames[1].len(), 1);
        assert!(frames[0][1].translation.vector.norm() < 1e-6);
        let mut invalid = frames.clone();
        invalid[0][0].translation.vector.z = 9.0;
        invalid[1][0].translation.vector.x = f64::NAN;
        assert!(batch.update_external_constraint_frames(&invalid).is_err());
        assert_eq!(
            batch.readback_prescribed_constraint_frames().unwrap(),
            frames
        );
        invalid[1][0] = Isometry3::identity();
        invalid[1][0].translation.vector.x = f64::MAX;
        assert!(batch.update_external_constraint_frames(&invalid).is_err());
        assert_eq!(
            batch.readback_prescribed_constraint_frames().unwrap(),
            frames
        );
        assert!(
            batch
                .update_external_constraint_frames(&[frames[0].clone(), Vec::new()])
                .is_err()
        );
        assert_eq!(
            batch.readback_prescribed_constraint_frames().unwrap(),
            frames
        );
        let stationary_state = &batch.readback().unwrap()[1];
        assert!(stationary_state.positions[0].abs() < 1e-6);
        let anchor = batch.readback_prescribed_constraint_frames().unwrap()[0][0];
        assert!((anchor.translation.vector.z - 0.02).abs() < 1e-5);
        let state = batch.readback().unwrap().remove(0);
        assert!((state.positions[0] - 0.02).abs() < 1e-5, "{state:?}");
        assert!((state.velocities[0] - 0.2).abs() < 1e-5, "{state:?}");
        batch
            .update_prescribed_constraint_bodies(&[vec![None, None], vec![None]])
            .unwrap();
        batch.submit_steps(10).unwrap();
        let stopped_anchor = batch.readback_prescribed_constraint_frames().unwrap()[0][0];
        assert!((stopped_anchor.translation.vector - anchor.translation.vector).norm() < 1e-6);
        let stopped = batch.readback().unwrap().remove(0);
        assert!(
            (stopped.positions[0] - state.positions[0]).abs() < 1e-5,
            "{stopped:?}"
        );
        assert!(stopped.velocities[0].abs() < 1e-5, "{stopped:?}");
    }

    #[test]
    fn resident_prescribed_fixed_anchor_rotates_and_stops() {
        let context = GpuContactDevice::new_with_backends(wgpu::Backends::VULKAN).unwrap();
        let articulation = Articulation::new(
            vec![link(0.0), link(1.0)],
            vec![JointSpec {
                parent: 0,
                child: 1,
                kind: JointKind::Revolute,
                origin: Isometry3::identity(),
                axis: Vector3::z(),
                limits: None,
            }],
            0,
        )
        .unwrap();
        let mut input = GpuArticulatedDynamicsInput::new(
            &articulation,
            Isometry3::identity(),
            GpuGeneralizedState {
                positions: DVector::zeros(1),
                velocities: DVector::zeros(1),
            },
            Vector3::zeros(),
        );
        input.link_fixed_constraints.push(LinkFixedConstraint {
            link_a: 1,
            frame_a: Isometry3::identity(),
            link_b: None,
            frame_b: Isometry3::identity(),
        });
        let mut batch = GpuArticulatedDynamicsBatch::new(&context, &[input], 0.001).unwrap();
        batch
            .update_prescribed_constraint_bodies(&[vec![Some(
                crate::gpu_kinematic_body::GpuKinematicBody {
                    pose: Isometry3::identity(),
                    linear_velocity: Vector3::zeros(),
                    angular_velocity: Vector3::z() * 0.2,
                },
            )]])
            .unwrap();
        batch.submit_steps(100).unwrap();
        let anchor = batch.readback_prescribed_constraint_frames().unwrap()[0][0];
        assert!((anchor.rotation.scaled_axis().z - 0.02).abs() < 1e-5);
        let state = batch.readback().unwrap().remove(0);
        assert!((state.positions[0] - 0.02).abs() < 1e-5, "{state:?}");
        assert!((state.velocities[0] - 0.2).abs() < 1e-5, "{state:?}");
        batch
            .update_prescribed_constraint_bodies(&[vec![None]])
            .unwrap();
        batch.submit_steps(10).unwrap();
        let stopped_anchor = batch.readback_prescribed_constraint_frames().unwrap()[0][0];
        assert!(stopped_anchor.rotation.angle_to(&anchor.rotation) < 1e-6);
        let stopped = batch.readback().unwrap().remove(0);
        assert!(
            (stopped.positions[0] - state.positions[0]).abs() < 1e-5,
            "{stopped:?}"
        );
        assert!(stopped.velocities[0].abs() < 1e-5, "{stopped:?}");
    }

    #[test]
    fn resident_prescribed_offset_point_anchor_orbits_and_stops() {
        let context = GpuContactDevice::new_with_backends(wgpu::Backends::VULKAN).unwrap();
        let articulation = Articulation::new(
            vec![link(0.0), link(1.0)],
            vec![JointSpec {
                parent: 0,
                child: 1,
                kind: JointKind::Revolute,
                origin: Isometry3::identity(),
                axis: Vector3::z(),
                limits: None,
            }],
            0,
        )
        .unwrap();
        let mut input = GpuArticulatedDynamicsInput::new(
            &articulation,
            Isometry3::identity(),
            GpuGeneralizedState {
                positions: DVector::zeros(1),
                velocities: DVector::zeros(1),
            },
            Vector3::zeros(),
        );
        input.link_point_constraints.push(LinkPointConstraint {
            link_a: 1,
            point_a: [1.0, 0.0, 0.0],
            link_b: None,
            point_b: [1.0, 0.0, 0.0],
        });
        let mut batch = GpuArticulatedDynamicsBatch::new(&context, &[input], 0.001).unwrap();
        batch
            .update_prescribed_constraint_bodies(&[vec![Some(
                crate::gpu_kinematic_body::GpuKinematicBody {
                    pose: Isometry3::identity(),
                    linear_velocity: Vector3::zeros(),
                    angular_velocity: Vector3::z() * 0.2,
                },
            )]])
            .unwrap();
        batch.submit_steps(100).unwrap();
        let anchor = batch.readback_prescribed_constraint_frames().unwrap()[0][0];
        assert!(
            (anchor.translation.vector - Vector3::new(0.02f64.cos(), 0.02f64.sin(), 0.0)).norm()
                < 1e-5
        );
        let state = batch.readback().unwrap().remove(0);
        assert!((state.positions[0] - 0.02).abs() < 1e-5, "{state:?}");
        assert!((state.velocities[0] - 0.2).abs() < 5e-5, "{state:?}");
        batch
            .update_prescribed_constraint_bodies(&[vec![None]])
            .unwrap();
        batch.submit_steps(10).unwrap();
        let stopped_anchor = batch.readback_prescribed_constraint_frames().unwrap()[0][0];
        assert!((stopped_anchor.translation.vector - anchor.translation.vector).norm() < 1e-6);
        let stopped = batch.readback().unwrap().remove(0);
        assert!(
            (stopped.positions[0] - state.positions[0]).abs() < 1e-5,
            "{stopped:?}"
        );
        assert!(stopped.velocities[0].abs() < 5e-5, "{stopped:?}");
    }

    #[test]
    fn resident_prescribed_axial_angular_velocity_reaches_contact_point() {
        use crate::gpu_articulated_ground_contact::{
            GpuArticulatedAxialContactKind as Kind, GpuArticulatedGroundAxialKind,
            GpuArticulatedStaticAxialSpherePair,
        };
        use crate::gpu_kinematic_body::GpuKinematicBody;
        for backend in [
            wgpu::Backends::VULKAN,
            #[cfg(windows)]
            wgpu::Backends::DX12,
        ] {
            let context = GpuContactDevice::new_with_backends(backend).unwrap();
            let articulation = Articulation::new(
                vec![link(0.0), link(1.0)],
                vec![JointSpec {
                    parent: 0,
                    child: 1,
                    kind: JointKind::Prismatic,
                    origin: Isometry3::identity(),
                    axis: Vector3::z(),
                    limits: None,
                }],
                0,
            )
            .unwrap();
            let moving = GpuKinematicBody {
                pose: Isometry3::translation(-0.2, 0.0, -0.5),
                linear_velocity: Vector3::zeros(),
                angular_velocity: -Vector3::y() * 0.4,
            };
            let inputs: Vec<_> = [
                GpuArticulatedGroundAxialKind::Cylinder,
                GpuArticulatedGroundAxialKind::Cone,
            ]
            .into_iter()
            .map(|kind| {
                let mut input = GpuArticulatedDynamicsInput::new(
                    &articulation,
                    Isometry3::translation(0.3, 0.0, 0.5),
                    GpuGeneralizedState {
                        positions: DVector::zeros(1),
                        velocities: DVector::zeros(1),
                    },
                    Vector3::zeros(),
                );
                input
                    .static_axial_sphere_pairs
                    .push(GpuArticulatedStaticAxialSpherePair {
                        link: 1,
                        axial_is_static: true,
                        kind,
                        local_pose: Isometry3::translation(0.0, 0.0, -0.5)
                            * if kind == GpuArticulatedGroundAxialKind::Cone {
                                Isometry3::rotation(Vector3::x() * core::f64::consts::PI)
                            } else {
                                Isometry3::identity()
                            },
                        half_height: 0.5,
                        radius: 0.5,
                        static_center: Vector3::zeros(),
                        static_radius: 0.5,
                        restitution: 0.0,
                        friction: 0.0,
                    });
                input.external_axial_bodies = Some(
                    crate::gpu_articulated_ground_contact::GpuArticulatedExternalAxialBodies {
                        spheres: vec![Some(moving.clone())],
                        ..Default::default()
                    },
                );
                input
            })
            .collect();
            let mut batch = GpuArticulatedDynamicsBatch::new(&context, &inputs, 0.001).unwrap();
            batch.enable_contact_activity().unwrap();
            // At the flat face, omega x (point - body_origin) gives +0.2 m/s Z.
            // Shape-center velocity alone gives +0.08 m/s and cannot satisfy this check.
            batch.submit_steps(1).unwrap();
            for state in batch.readback().unwrap() {
                assert!((state.velocities[0] - 0.2).abs() < 5e-4, "{state:?}");
            }
            assert_eq!(
                batch.readback_contact_wake_requests().unwrap(),
                vec![vec![false, true], vec![false, true]]
            );
            for (index, poses) in batch
                .readback_prescribed_axial_poses(Kind::Sphere)
                .unwrap()
                .iter()
                .enumerate()
            {
                let delta = nalgebra::UnitQuaternion::from_scaled_axis(-Vector3::y() * 0.0004);
                let initial = if index == 1 {
                    nalgebra::UnitQuaternion::from_scaled_axis(Vector3::x() * core::f64::consts::PI)
                } else {
                    nalgebra::UnitQuaternion::identity()
                };
                assert!(poses[0].rotation.angle_to(&(delta * initial)) < 1e-5);
            }
            eprintln!("prescribed axial angular contact passed on {backend:?}");
        }
    }

    #[test]
    fn resident_prescribed_axial_overflow_rejects_readback_and_recovers() {
        use crate::gpu_articulated_ground_contact::{
            GpuArticulatedAxialContactKind as Kind, GpuArticulatedGroundAxialKind,
            GpuArticulatedStaticAxialSpherePair,
        };
        use crate::gpu_kinematic_body::GpuKinematicBody;
        let context = GpuContactDevice::new_with_backends(wgpu::Backends::VULKAN).unwrap();
        let articulation = Articulation::new(
            vec![link(0.0), link(1.0)],
            vec![JointSpec {
                parent: 0,
                child: 1,
                kind: JointKind::Prismatic,
                origin: Isometry3::identity(),
                axis: Vector3::z(),
                limits: None,
            }],
            0,
        )
        .unwrap();
        let moving = GpuKinematicBody {
            pose: Isometry3::translation(0.0, 0.0, -0.5),
            linear_velocity: Vector3::x() * 3e38,
            angular_velocity: Vector3::zeros(),
        };
        let inputs: Vec<_> = [
            GpuArticulatedGroundAxialKind::Cylinder,
            GpuArticulatedGroundAxialKind::Cone,
        ]
        .into_iter()
        .map(|kind| {
            let mut input = GpuArticulatedDynamicsInput::new(
                &articulation,
                Isometry3::translation(0.0, 0.0, 10.0),
                GpuGeneralizedState {
                    positions: DVector::zeros(1),
                    velocities: DVector::zeros(1),
                },
                Vector3::zeros(),
            );
            input
                .static_axial_sphere_pairs
                .push(GpuArticulatedStaticAxialSpherePair {
                    link: 1,
                    axial_is_static: true,
                    kind,
                    local_pose: Isometry3::translation(0.0, 0.0, -0.5),
                    half_height: 0.5,
                    radius: 0.5,
                    static_center: Vector3::zeros(),
                    static_radius: 0.5,
                    restitution: 0.0,
                    friction: 0.0,
                });
            input.external_axial_bodies = Some(
                crate::gpu_articulated_ground_contact::GpuArticulatedExternalAxialBodies {
                    spheres: vec![if kind == GpuArticulatedGroundAxialKind::Cylinder {
                        Some(moving.clone())
                    } else {
                        None
                    }],
                    ..Default::default()
                },
            );
            input
        })
        .collect();
        let mut batch = GpuArticulatedDynamicsBatch::new(&context, &inputs, 2.0).unwrap();
        let before = batch.readback_prescribed_axial_poses(Kind::Sphere).unwrap();
        batch.submit_steps(1).unwrap();
        assert!(matches!(
            batch.readback(),
            Err(GpuArticulatedDynamicsError::State(
                GpuGeneralizedStateError::NonFinite(0)
            ))
        ));
        assert!(matches!(
            batch.readback_prescribed_axial_poses(Kind::Sphere),
            Err(GpuArticulatedDynamicsError::GroundContact(
                GpuArticulatedGroundContactError::SourceFault(0)
            ))
        ));
        assert!(matches!(
            batch.update_prescribed_axial_bodies(Kind::Sphere, &[vec![None], vec![None]]),
            Err(GpuArticulatedDynamicsError::GroundContact(
                GpuArticulatedGroundContactError::SourceFault(0)
            ))
        ));
        let states: Vec<_> = inputs.iter().map(|input| input.state.clone()).collect();
        batch.reset(&states).unwrap();
        assert_eq!(
            batch.readback_prescribed_axial_poses(Kind::Sphere).unwrap(),
            before
        );
        batch
            .update_prescribed_axial_bodies(Kind::Sphere, &[vec![None], vec![None]])
            .unwrap();
        batch.submit_steps(1).unwrap();
        assert_eq!(batch.readback().unwrap(), states);
        assert_eq!(
            batch.readback_prescribed_axial_poses(Kind::Sphere).unwrap(),
            before
        );
    }

    #[test]
    fn resident_prescribed_convex_rounded_motion_stop_and_orbit() {
        use crate::gpu_articulated_ground_contact::{
            GpuArticulatedSceneConvexCapsulePair, GpuArticulatedSceneConvexSpherePair,
        };
        use crate::gpu_kinematic_body::GpuKinematicBody;
        let context = GpuContactDevice::new_with_backends(wgpu::Backends::VULKAN).unwrap();
        let articulation = Articulation::new(
            vec![link(0.0), link(1.0)],
            vec![JointSpec {
                parent: 0,
                child: 1,
                kind: JointKind::Prismatic,
                origin: Isometry3::identity(),
                axis: Vector3::z(),
                limits: None,
            }],
            0,
        )
        .unwrap();
        let vertices: Vec<_> = [-0.5, 0.5]
            .into_iter()
            .flat_map(|x| {
                [-0.5, 0.5]
                    .into_iter()
                    .flat_map(move |y| [-0.5, 0.5].into_iter().map(move |z| Vector3::new(x, y, z)))
            })
            .collect();
        let normals = vec![
            Vector3::x(),
            -Vector3::x(),
            Vector3::y(),
            -Vector3::y(),
            Vector3::z(),
            -Vector3::z(),
        ];
        let body = GpuKinematicBody {
            pose: Isometry3::translation(0.0, 0.0, -0.5),
            linear_velocity: Vector3::z() * 0.2,
            angular_velocity: Vector3::zeros(),
        };
        let inputs: Vec<_> = [false, true]
            .into_iter()
            .map(|capsule| {
                let mut input = GpuArticulatedDynamicsInput::new(
                    &articulation,
                    Isometry3::translation(0.0, 0.0, 0.5),
                    GpuGeneralizedState {
                        positions: DVector::zeros(1),
                        velocities: DVector::zeros(1),
                    },
                    Vector3::zeros(),
                );
                if capsule {
                    input
                        .scene_convex_capsule_pairs
                        .push(GpuArticulatedSceneConvexCapsulePair {
                            convex_world_pose: Isometry3::translation(0.0, 0.0, -0.5),
                            vertices: vertices.clone(),
                            face_normals: normals.clone(),
                            edge_directions: vec![Vector3::x(), Vector3::y(), Vector3::z()],
                            capsule_link: 1,
                            capsule_local_a: -Vector3::x() * 0.2,
                            capsule_local_b: Vector3::x() * 0.2,
                            capsule_radius: 0.5,
                            restitution: 0.0,
                            friction: 0.0,
                        });
                } else {
                    input
                        .scene_convex_sphere_pairs
                        .push(GpuArticulatedSceneConvexSpherePair {
                            convex_world_pose: Isometry3::translation(0.0, 0.0, -0.5),
                            vertices: vertices.clone(),
                            face_normals: normals.clone(),
                            sphere_link: 1,
                            sphere_local_center: Vector3::zeros(),
                            sphere_radius: 0.5,
                            restitution: 0.0,
                            friction: 0.0,
                        });
                }
                input.external_convex_bodies = Some(
                    crate::gpu_articulated_ground_contact::GpuArticulatedExternalConvexBodies {
                        rounded: vec![Some(body.clone())],
                        polyhedron_axial: Vec::new(),
                    },
                );
                input
            })
            .collect();
        let mut batch = GpuArticulatedDynamicsBatch::new(&context, &inputs, 0.001).unwrap();
        batch.submit_steps(100).unwrap();
        for state in batch.readback().unwrap() {
            assert!((state.velocities[0] - 0.2).abs() < 5e-4, "{state:?}");
        }
        let before = batch.readback_prescribed_convex_rounded_poses().unwrap();
        for poses in &before {
            assert!((poses[0].translation.z + 0.48).abs() < 2e-6);
        }
        batch
            .update_prescribed_convex_rounded_bodies(&[vec![None], vec![None]])
            .unwrap();
        batch.submit_steps(10).unwrap();
        assert_eq!(
            batch.readback_prescribed_convex_rounded_poses().unwrap(),
            before
        );
        let original = vec![vec![Isometry3::translation(0.0, 0.0, -0.5)]; 2];
        assert!(batch.update_scene_convex_rounded_poses(&original).unwrap());
        assert_eq!(
            batch.readback_prescribed_convex_rounded_poses().unwrap(),
            original
        );
        assert!(!batch.update_scene_convex_rounded_poses(&original).unwrap());
        for environment in 0..2 {
            batch
                .set_root_pose(environment, Isometry3::translation(0.0, 0.0, 10.0))
                .unwrap();
        }
        let orbit = GpuKinematicBody {
            pose: Isometry3::translation(-1.0, 0.0, -0.5),
            linear_velocity: Vector3::x() * 0.1,
            angular_velocity: Vector3::y() * 0.3,
        };
        batch
            .update_prescribed_convex_rounded_bodies(&[
                vec![Some(orbit.clone())],
                vec![Some(orbit.clone())],
            ])
            .unwrap();
        batch.submit_steps(100).unwrap();
        let rotation = nalgebra::UnitQuaternion::from_scaled_axis(Vector3::y() * 0.03);
        let expected = Vector3::new(-0.99, 0.0, -0.5) + rotation * Vector3::new(1.0, 0.0, 0.0);
        for poses in batch.readback_prescribed_convex_rounded_poses().unwrap() {
            assert!((poses[0].translation.vector - expected).norm() < 1e-5);
            assert!(poses[0].rotation.angle_to(&rotation) < 1e-5);
        }
    }

    #[test]
    fn resident_prescribed_axial_sphere_motion_stop_and_orbit() {
        use crate::gpu_articulated_ground_contact::{
            GpuArticulatedAxialContactKind as Kind, GpuArticulatedGroundAxialKind,
            GpuArticulatedStaticAxialSpherePair,
        };
        use crate::gpu_kinematic_body::GpuKinematicBody;
        let context = GpuContactDevice::new_with_backends(wgpu::Backends::VULKAN).unwrap();
        let articulation = Articulation::new(
            vec![link(0.0), link(1.0)],
            vec![JointSpec {
                parent: 0,
                child: 1,
                kind: JointKind::Prismatic,
                origin: Isometry3::identity(),
                axis: Vector3::z(),
                limits: None,
            }],
            0,
        )
        .unwrap();
        let moving = GpuKinematicBody {
            pose: Isometry3::translation(0.0, 0.0, -0.5),
            linear_velocity: Vector3::z() * 0.2,
            angular_velocity: Vector3::zeros(),
        };
        let inputs: Vec<_> = [
            GpuArticulatedGroundAxialKind::Cylinder,
            GpuArticulatedGroundAxialKind::Cone,
        ]
        .into_iter()
        .map(|kind| {
            let mut input = GpuArticulatedDynamicsInput::new(
                &articulation,
                Isometry3::translation(0.0, 0.0, 0.5),
                GpuGeneralizedState {
                    positions: DVector::zeros(1),
                    velocities: DVector::zeros(1),
                },
                Vector3::zeros(),
            );
            input
                .static_axial_sphere_pairs
                .push(GpuArticulatedStaticAxialSpherePair {
                    link: 1,
                    axial_is_static: true,
                    kind,
                    local_pose: Isometry3::translation(0.0, 0.0, -0.5),
                    half_height: 0.5,
                    radius: 0.5,
                    static_center: Vector3::zeros(),
                    static_radius: 0.5,
                    restitution: 0.0,
                    friction: 0.0,
                });
            input.external_axial_bodies = Some(
                crate::gpu_articulated_ground_contact::GpuArticulatedExternalAxialBodies {
                    spheres: vec![Some(moving.clone())],
                    ..Default::default()
                },
            );
            input
        })
        .collect();
        let mut batch = GpuArticulatedDynamicsBatch::new(&context, &inputs, 0.001).unwrap();
        batch.enable_contact_activity().unwrap();
        batch.submit_steps(1).unwrap();
        assert_eq!(
            batch.readback_contact_wake_requests().unwrap(),
            vec![vec![false, true], vec![false, true]]
        );
        batch.submit_steps(99).unwrap();
        for state in batch.readback().unwrap() {
            assert!((state.velocities[0] - 0.2).abs() < 5e-4, "{state:?}");
        }
        let before = batch.readback_prescribed_axial_poses(Kind::Sphere).unwrap();
        for poses in &before {
            assert!((poses[0].translation.z + 0.48).abs() < 2e-6);
        }
        let invalid = GpuKinematicBody {
            linear_velocity: Vector3::new(f64::INFINITY, 0.0, 0.0),
            ..moving
        };
        assert!(
            batch
                .update_prescribed_axial_bodies(
                    Kind::Sphere,
                    &[vec![Some(moving.clone())], vec![Some(invalid)]]
                )
                .is_err()
        );
        assert_eq!(
            batch.readback_prescribed_axial_poses(Kind::Sphere).unwrap(),
            before
        );
        batch
            .update_prescribed_axial_bodies(Kind::Sphere, &[vec![None], vec![None]])
            .unwrap();
        batch.submit_steps(10).unwrap();
        assert_eq!(
            batch.readback_prescribed_axial_poses(Kind::Sphere).unwrap(),
            before
        );
        for environment in 0..2 {
            batch
                .set_root_pose(environment, Isometry3::translation(0.0, 0.0, 10.0))
                .unwrap();
        }
        let orbit = GpuKinematicBody {
            pose: Isometry3::translation(-1.0, 0.0, -0.5),
            linear_velocity: Vector3::x() * 0.1,
            angular_velocity: Vector3::y() * 0.3,
        };
        batch
            .update_prescribed_axial_bodies(
                Kind::Sphere,
                &[vec![Some(orbit.clone())], vec![Some(orbit.clone())]],
            )
            .unwrap();
        batch.submit_steps(100).unwrap();
        let rotation = nalgebra::UnitQuaternion::from_scaled_axis(Vector3::y() * 0.03);
        let expected = Vector3::new(-0.99, 0.0, -0.5) + rotation * Vector3::new(1.0, 0.0, 0.02);
        for poses in batch.readback_prescribed_axial_poses(Kind::Sphere).unwrap() {
            assert!(
                (poses[0].translation.vector - expected).norm() < 1e-5,
                "{:?}",
                poses[0]
            );
            assert!(poses[0].rotation.angle_to(&rotation) < 1e-5);
        }
    }

    #[test]
    fn resident_prescribed_capsules_cover_primitive_and_convex_contact_families() {
        use crate::gpu_articulated_ground_contact::{
            GpuArticulatedCapsuleContactKind as Kind, GpuArticulatedGroundAxialKind,
            GpuArticulatedStaticAxialCapsulePair, GpuArticulatedStaticBoxCapsulePair,
            GpuArticulatedStaticCapsulePair, GpuArticulatedStaticConvexCapsulePair,
            GpuArticulatedStaticSphereCapsulePair,
        };
        use crate::gpu_kinematic_body::GpuKinematicBody;
        let articulation = Articulation::new(
            vec![link(0.0), link(1.0)],
            vec![JointSpec {
                parent: 0,
                child: 1,
                kind: JointKind::Prismatic,
                origin: Isometry3::identity(),
                axis: Vector3::z(),
                limits: None,
            }],
            0,
        )
        .unwrap();
        let pose = Isometry3::translation(0.0, 0.0, -0.5);
        let hull = crate::convex::ConvexGeometry::new(
            [-0.5, 0.5]
                .into_iter()
                .flat_map(|x| {
                    [-0.5, 0.5].into_iter().flat_map(move |y| {
                        [-0.5, 0.5].into_iter().map(move |z| Vector3::new(x, y, z))
                    })
                })
                .collect(),
            vec![
                Vector3::x(),
                -Vector3::x(),
                Vector3::y(),
                -Vector3::y(),
                Vector3::z(),
                -Vector3::z(),
            ],
            vec![Vector3::x(), Vector3::y(), Vector3::z()],
        )
        .unwrap();
        let kinds = [
            Kind::Sphere,
            Kind::Capsule,
            Kind::Box,
            Kind::Axial,
            Kind::Axial,
            Kind::Convex,
            Kind::Sphere,
        ];
        let inputs: Vec<_> =
            kinds
                .iter()
                .enumerate()
                .map(|(index, kind)| {
                    let mut input = GpuArticulatedDynamicsInput::new(
                        &articulation,
                        Isometry3::translation(0.0, 0.0, 0.5),
                        GpuGeneralizedState {
                            positions: DVector::zeros(1),
                            velocities: DVector::zeros(1),
                        },
                        Vector3::zeros(),
                    );
                    let static_a = Vector3::new(-0.2, 0.0, -0.5);
                    let static_b = Vector3::new(0.2, 0.0, -0.5);
                    match kind {
                        Kind::Sphere => input.static_sphere_capsule_pairs.push(
                            GpuArticulatedStaticSphereCapsulePair {
                                link: 1,
                                local_center: Vector3::zeros(),
                                radius: 0.5,
                                static_a,
                                static_b,
                                static_radius: 0.5,
                                restitution: 0.0,
                                friction: 0.0,
                            },
                        ),
                        Kind::Capsule => {
                            input
                                .static_capsule_pairs
                                .push(GpuArticulatedStaticCapsulePair {
                                    link: 1,
                                    local_a: -Vector3::x() * 0.2,
                                    local_b: Vector3::x() * 0.2,
                                    radius: 0.5,
                                    static_a,
                                    static_b,
                                    static_radius: 0.5,
                                    restitution: 0.0,
                                    friction: 0.0,
                                })
                        }
                        Kind::Box => input.static_box_capsule_pairs.push(
                            GpuArticulatedStaticBoxCapsulePair {
                                link: 1,
                                local_pose: Isometry3::identity(),
                                half_extents: Vector3::repeat(0.5),
                                static_a,
                                static_b,
                                static_radius: 0.5,
                                restitution: 0.0,
                                friction: 0.0,
                            },
                        ),
                        Kind::Axial => input.static_axial_capsule_pairs.push(
                            GpuArticulatedStaticAxialCapsulePair {
                                link: 1,
                                axial_is_static: false,
                                kind: if index == 3 {
                                    GpuArticulatedGroundAxialKind::Cylinder
                                } else {
                                    GpuArticulatedGroundAxialKind::Cone
                                },
                                local_pose: Isometry3::identity(),
                                radius: 0.3,
                                half_height: 0.5,
                                static_a,
                                static_b,
                                static_radius: 0.5,
                                restitution: 0.0,
                                friction: 0.0,
                            },
                        ),
                        Kind::Convex => input.static_convex_capsule_pairs.push(
                            GpuArticulatedStaticConvexCapsulePair {
                                convex_link: 1,
                                convex_local_pose: Isometry3::identity(),
                                vertices: hull.vertices.clone(),
                                face_normals: hull.face_normals.clone(),
                                edge_directions: hull.edge_directions.clone(),
                                static_a,
                                static_b,
                                static_radius: 0.5,
                                restitution: 0.0,
                                friction: 0.0,
                            },
                        ),
                    }
                    input
                })
                .collect();
        let families = [
            Kind::Sphere,
            Kind::Capsule,
            Kind::Box,
            Kind::Axial,
            Kind::Convex,
        ];
        let body_inputs = |kind, body: Option<GpuKinematicBody>| {
            kinds
                .iter()
                .enumerate()
                .map(|(index, actual)| {
                    if *actual == kind {
                        vec![if index == 6 { None } else { body.clone() }]
                    } else {
                        vec![]
                    }
                })
                .collect::<Vec<_>>()
        };
        let moving = GpuKinematicBody {
            pose,
            linear_velocity: Vector3::z() * 0.2,
            angular_velocity: Vector3::zeros(),
        };
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let mut batch = GpuArticulatedDynamicsBatch::new(&context, &inputs, 0.001).unwrap();
            batch.enable_contact_activity().unwrap();
            for kind in families {
                batch
                    .update_prescribed_capsule_bodies(
                        kind,
                        &body_inputs(kind, Some(moving.clone())),
                    )
                    .unwrap();
            }
            batch.submit_steps(1).unwrap();
            assert_eq!(
                batch.readback_contact_wake_requests().unwrap(),
                vec![vec![false, true]; 7]
            );
            batch.submit_steps(9).unwrap();
            let output = batch.readback().unwrap();
            for (environment, state) in output.iter().enumerate() {
                let expected = if environment == 6 { 0.0 } else { 0.2 };
                assert!(
                    (state.velocities[0] - expected).abs() < 5e-4,
                    "{backend:?}, env {environment}: {:?}",
                    state.velocities
                );
            }
            let before = batch
                .readback_prescribed_capsule_endpoints(Kind::Capsule)
                .unwrap();
            let mut invalid = moving.clone();
            invalid.linear_velocity.x = f64::INFINITY;
            assert!(
                batch
                    .update_prescribed_capsule_bodies(
                        Kind::Capsule,
                        &body_inputs(Kind::Capsule, Some(invalid))
                    )
                    .is_err()
            );
            assert_eq!(
                batch
                    .readback_prescribed_capsule_endpoints(Kind::Capsule)
                    .unwrap(),
                before
            );
            batch.submit_steps(10).unwrap();
            for kind in families {
                batch
                    .update_prescribed_capsule_bodies(kind, &body_inputs(kind, None))
                    .unwrap();
            }
            let stopped =
                families.map(|kind| batch.readback_prescribed_capsule_endpoints(kind).unwrap());
            batch.submit_steps(10).unwrap();
            for (kind, expected) in families.into_iter().zip(&stopped) {
                assert_eq!(
                    &batch.readback_prescribed_capsule_endpoints(kind).unwrap(),
                    expected
                );
            }
            let mut distant = inputs.clone();
            for input in &mut distant {
                input.root_pose = Isometry3::translation(0.0, 0.0, 10.0);
            }
            let body = GpuKinematicBody {
                pose: Isometry3::translation(-1.0, 0.0, -0.5),
                linear_velocity: Vector3::x() * 0.1,
                angular_velocity: Vector3::y() * 0.3,
            };
            let mut orbit = GpuArticulatedDynamicsBatch::new(&context, &distant, 0.001).unwrap();
            for kind in families {
                orbit
                    .update_prescribed_capsule_bodies(kind, &body_inputs(kind, Some(body.clone())))
                    .unwrap();
            }
            orbit.submit_steps(100).unwrap();
            let rotation = nalgebra::UnitQuaternion::from_scaled_axis(body.angular_velocity * 0.1);
            for kind in families {
                for (environment, pairs) in orbit
                    .readback_prescribed_capsule_endpoints(kind)
                    .unwrap()
                    .iter()
                    .enumerate()
                {
                    for actual in pairs {
                        for (endpoint, initial) in actual
                            .iter()
                            .zip([Vector3::new(-0.2, 0.0, -0.5), Vector3::new(0.2, 0.0, -0.5)])
                        {
                            let expected = if environment == 6 {
                                initial
                            } else {
                                body.pose.translation.vector
                                    + body.linear_velocity * 0.1
                                    + rotation * (initial - body.pose.translation.vector)
                            };
                            assert!(
                                (endpoint - expected).norm() < 2e-5,
                                "{backend:?}, {kind:?}, env {environment}: {endpoint:?} vs {expected:?}"
                            );
                        }
                    }
                }
            }
            eprintln!("prescribed capsule primitive and convex families passed on {backend:?}");
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn resident_prescribed_boxes_cover_primitive_and_convex_contact_families() {
        use crate::gpu_articulated_ground_contact::{
            GpuArticulatedBoxContactKind as Kind, GpuArticulatedGroundAxialKind,
            GpuArticulatedStaticAxialBoxPair, GpuArticulatedStaticBoxPair,
            GpuArticulatedStaticCapsuleBoxPair, GpuArticulatedStaticSphereBoxPair,
        };
        use crate::gpu_kinematic_body::GpuKinematicBody;
        let articulation = Articulation::new(
            vec![link(0.0), link(1.0)],
            vec![JointSpec {
                parent: 0,
                child: 1,
                kind: JointKind::Prismatic,
                origin: Isometry3::identity(),
                axis: Vector3::z(),
                limits: None,
            }],
            0,
        )
        .unwrap();
        let pose = Isometry3::translation(0.0, 0.0, -0.5);
        let hull = crate::convex::ConvexGeometry::new(
            [-0.5, 0.5]
                .into_iter()
                .flat_map(|x| {
                    [-0.5, 0.5].into_iter().flat_map(move |y| {
                        [-0.5, 0.5].into_iter().map(move |z| Vector3::new(x, y, z))
                    })
                })
                .collect(),
            vec![
                Vector3::x(),
                -Vector3::x(),
                Vector3::y(),
                -Vector3::y(),
                Vector3::z(),
                -Vector3::z(),
            ],
            vec![Vector3::x(), Vector3::y(), Vector3::z()],
        )
        .unwrap();
        let kinds = [
            Kind::Sphere,
            Kind::Capsule,
            Kind::Box,
            Kind::Axial,
            Kind::Axial,
            Kind::Convex,
            Kind::Sphere,
        ];
        let inputs: Vec<_> =
            kinds
                .iter()
                .enumerate()
                .map(|(index, kind)| {
                    let mut input = GpuArticulatedDynamicsInput::new(
                        &articulation,
                        Isometry3::translation(0.0, 0.0, 0.5),
                        GpuGeneralizedState {
                            positions: DVector::zeros(1),
                            velocities: DVector::zeros(1),
                        },
                        Vector3::zeros(),
                    );
                    match kind {
                        Kind::Sphere => {
                            input
                                .static_sphere_box_pairs
                                .push(GpuArticulatedStaticSphereBoxPair {
                                    link: 1,
                                    local_center: Vector3::zeros(),
                                    radius: 0.5,
                                    static_pose: pose,
                                    half_extents: Vector3::repeat(0.5),
                                    restitution: 0.0,
                                    friction: 0.0,
                                })
                        }
                        Kind::Capsule => input.static_capsule_box_pairs.push(
                            GpuArticulatedStaticCapsuleBoxPair {
                                link: 1,
                                local_a: -Vector3::x() * 0.2,
                                local_b: Vector3::x() * 0.2,
                                radius: 0.5,
                                static_pose: pose,
                                half_extents: Vector3::repeat(0.5),
                                restitution: 0.0,
                                friction: 0.0,
                            },
                        ),
                        Kind::Box => input.static_box_pairs.push(GpuArticulatedStaticBoxPair {
                            link: 1,
                            local_pose: Isometry3::identity(),
                            half_extents: Vector3::repeat(0.5),
                            static_pose: pose,
                            static_half_extents: Vector3::repeat(0.5),
                            restitution: 0.0,
                            friction: 0.0,
                        }),
                        Kind::Convex => {
                            input
                                .static_convex_pairs
                                .push(GpuArticulatedStaticConvexPair {
                                    first_link: 1,
                                    first_local_pose: Isometry3::identity(),
                                    first_geometry: hull.clone(),
                                    second_world_pose: pose,
                                    second_geometry: hull.clone(),
                                    restitution: 0.0,
                                    friction: 0.0,
                                })
                        }
                        Kind::Axial => {
                            input
                                .static_axial_box_pairs
                                .push(GpuArticulatedStaticAxialBoxPair {
                                    link: 1,
                                    axial_is_static: false,
                                    kind: if index == 3 {
                                        GpuArticulatedGroundAxialKind::Cylinder
                                    } else {
                                        GpuArticulatedGroundAxialKind::Cone
                                    },
                                    local_pose: Isometry3::identity(),
                                    radius: 0.3,
                                    half_height: 0.5,
                                    static_pose: pose,
                                    static_half_extents: Vector3::repeat(0.5),
                                    restitution: 0.0,
                                    friction: 0.0,
                                })
                        }
                    }
                    input
                })
                .collect();
        let moving = GpuKinematicBody {
            pose,
            linear_velocity: Vector3::z() * 0.2,
            angular_velocity: Vector3::zeros(),
        };
        let body_inputs = |kind: Kind, body: Option<GpuKinematicBody>| {
            kinds
                .iter()
                .enumerate()
                .map(|(index, actual)| {
                    if *actual == kind {
                        vec![if index == 6 { None } else { body.clone() }]
                    } else {
                        vec![]
                    }
                })
                .collect::<Vec<_>>()
        };
        let families = [
            Kind::Sphere,
            Kind::Capsule,
            Kind::Box,
            Kind::Axial,
            Kind::Convex,
        ];
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let mut batch = GpuArticulatedDynamicsBatch::new(&context, &inputs, 0.001).unwrap();
            for kind in families {
                batch
                    .update_prescribed_box_bodies(kind, &body_inputs(kind, Some(moving.clone())))
                    .unwrap();
            }
            batch.submit_steps(10).unwrap();
            let output = batch.readback_output().unwrap();
            for (index, environment) in output.iter().enumerate() {
                let expected = if index == 6 { 0.0 } else { 0.2 };
                assert!(
                    (environment.state.velocities[0] - expected).abs() < 5e-4,
                    "{backend:?} {index} {:?}: {:?}",
                    kinds[index],
                    environment.state.velocities
                );
            }
            for kind in families {
                let poses = batch.readback_prescribed_box_poses(kind).unwrap();
                for (index, values) in poses.iter().enumerate() {
                    for actual in values {
                        let expected = if index == 6 { -0.5 } else { -0.498 };
                        assert!((actual.translation.z - expected).abs() < 2e-6);
                    }
                }
            }
            let mut invalid = moving.clone();
            invalid.linear_velocity.x = f64::INFINITY;
            let before = batch.readback_prescribed_box_poses(Kind::Capsule).unwrap();
            assert!(
                batch
                    .update_prescribed_box_bodies(
                        Kind::Capsule,
                        &body_inputs(Kind::Capsule, Some(invalid))
                    )
                    .is_err()
            );
            assert_eq!(
                batch.readback_prescribed_box_poses(Kind::Capsule).unwrap(),
                before
            );
            batch.submit_steps(10).unwrap();
            for kind in families {
                batch
                    .update_prescribed_box_bodies(kind, &body_inputs(kind, None))
                    .unwrap();
            }
            let stopped = families.map(|kind| batch.readback_prescribed_box_poses(kind).unwrap());
            batch.submit_steps(10).unwrap();
            for (kind, expected) in families.into_iter().zip(&stopped) {
                assert_eq!(
                    &batch.readback_prescribed_box_poses(kind).unwrap(),
                    expected
                );
            }
            let static_poses = |kind| {
                kinds
                    .iter()
                    .map(|actual| if *actual == kind { vec![pose] } else { vec![] })
                    .collect::<Vec<_>>()
            };
            assert!(
                batch
                    .update_static_sphere_box_poses(&static_poses(Kind::Sphere))
                    .unwrap()
            );
            assert!(
                batch
                    .update_static_capsule_box_poses(&static_poses(Kind::Capsule))
                    .unwrap()
            );
            assert!(
                batch
                    .update_static_box_box_poses(&static_poses(Kind::Box))
                    .unwrap()
            );
            assert!(
                batch
                    .update_static_axial_box_poses(&static_poses(Kind::Axial))
                    .unwrap()
            );
            assert!(
                batch
                    .update_static_convex_pair_poses(&static_poses(Kind::Convex))
                    .unwrap()
            );
            for kind in families {
                assert_eq!(
                    batch.readback_prescribed_box_poses(kind).unwrap(),
                    static_poses(kind)
                );
            }

            let mut distant = inputs.clone();
            for input in &mut distant {
                input.root_pose = Isometry3::translation(0.0, 0.0, 10.0);
            }
            let body = GpuKinematicBody {
                pose: Isometry3::translation(-1.0, 0.0, -0.5),
                linear_velocity: Vector3::x() * 0.1,
                angular_velocity: Vector3::y() * 0.3,
            };
            let mut orbit = GpuArticulatedDynamicsBatch::new(&context, &distant, 0.001).unwrap();
            for kind in families {
                orbit
                    .update_prescribed_box_bodies(kind, &body_inputs(kind, Some(body.clone())))
                    .unwrap();
            }
            orbit.submit_steps(100).unwrap();
            let rotation = nalgebra::UnitQuaternion::from_scaled_axis(body.angular_velocity * 0.1);
            let expected =
                body.pose.translation.vector + body.linear_velocity * 0.1 + rotation * Vector3::x();
            for kind in families {
                for (index, values) in orbit
                    .readback_prescribed_box_poses(kind)
                    .unwrap()
                    .iter()
                    .enumerate()
                {
                    for actual in values {
                        if index == 6 {
                            assert_eq!(*actual, pose);
                            continue;
                        }
                        assert!(
                            (actual.translation.vector - expected).norm() < 2e-5,
                            "{backend:?}: {kind:?}"
                        );
                        assert!(actual.rotation.angle_to(&rotation) < 1e-5);
                    }
                }
            }
            eprintln!("prescribed box primitive and convex families passed on {backend:?}");
        }
        assert!(tested > 0, "no hardware backend available");
    }

    #[test]
    fn resident_prescribed_box_motion_transmits_velocity_and_rotates_offsets() {
        use crate::gpu_articulated_ground_contact::GpuArticulatedStaticSphereBoxPair;
        use crate::gpu_kinematic_body::GpuKinematicBody;
        let articulation = Articulation::new(
            vec![link(0.0), link(1.0)],
            vec![JointSpec {
                parent: 0,
                child: 1,
                kind: JointKind::Prismatic,
                origin: Isometry3::identity(),
                axis: Vector3::z(),
                limits: None,
            }],
            0,
        )
        .unwrap();
        let box_pose = Isometry3::translation(0.0, 0.0, -0.5);
        let mut input = GpuArticulatedDynamicsInput::new(
            &articulation,
            Isometry3::translation(0.0, 0.0, 0.5),
            GpuGeneralizedState {
                positions: DVector::zeros(1),
                velocities: DVector::zeros(1),
            },
            Vector3::zeros(),
        );
        input
            .static_sphere_box_pairs
            .push(GpuArticulatedStaticSphereBoxPair {
                link: 1,
                local_center: Vector3::zeros(),
                radius: 0.5,
                static_pose: box_pose,
                half_extents: Vector3::repeat(0.5),
                restitution: 0.0,
                friction: 0.5,
            });
        let moving = GpuKinematicBody {
            pose: box_pose,
            linear_velocity: Vector3::z() * 0.2,
            angular_velocity: Vector3::zeros(),
        };
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let mut batch =
                GpuArticulatedDynamicsBatch::new(&context, &[input.clone(), input.clone()], 0.001)
                    .unwrap();
            batch
                .update_prescribed_sphere_box_bodies(&[vec![Some(moving.clone())], vec![None]])
                .unwrap();
            batch.submit_steps(10).unwrap();
            let output = batch.readback_output().unwrap();
            assert!(
                (output[0].state.velocities[0] - 0.2).abs() < 2e-4,
                "{backend:?}: {:?}",
                output[0].state.velocities
            );
            assert!(output[1].state.velocities[0].abs() < 1e-6);
            let poses = batch.readback_static_sphere_box_poses().unwrap();
            assert!((poses[0][0].translation.z + 0.498).abs() < 1e-6);
            assert_eq!(poses[1][0], box_pose);
            let mut invalid = moving.clone();
            invalid.angular_velocity.x = f64::INFINITY;
            assert!(
                batch
                    .update_prescribed_sphere_box_bodies(&[vec![None], vec![Some(invalid)]])
                    .is_err()
            );
            assert_eq!(batch.readback_static_sphere_box_poses().unwrap(), poses);
            batch.submit_steps(10).unwrap();
            assert!(
                (batch.readback_static_sphere_box_poses().unwrap()[0][0]
                    .translation
                    .z
                    + 0.496)
                    .abs()
                    < 1e-6
            );
            batch
                .update_prescribed_sphere_box_bodies(&[vec![None], vec![None]])
                .unwrap();
            let stopped = batch.readback_static_sphere_box_poses().unwrap();
            batch.submit_steps(10).unwrap();
            assert_eq!(batch.readback_static_sphere_box_poses().unwrap(), stopped);
            assert!(
                batch
                    .update_static_sphere_box_poses(&[vec![box_pose], vec![box_pose]])
                    .unwrap()
            );
            assert_eq!(
                batch.readback_static_sphere_box_poses().unwrap(),
                vec![vec![box_pose], vec![box_pose]]
            );

            let mut offset_input = input.clone();
            offset_input.root_pose = Isometry3::translation(0.0, 0.0, 10.0);
            let orientation = nalgebra::UnitQuaternion::from_euler_angles(0.1, -0.2, 0.3);
            offset_input.static_sphere_box_pairs[0].static_pose.rotation = orientation;
            let body = GpuKinematicBody {
                pose: Isometry3::translation(-1.0, 0.0, -0.5),
                linear_velocity: Vector3::x() * 0.1,
                angular_velocity: Vector3::y() * 0.3,
            };
            let mut offset =
                GpuArticulatedDynamicsBatch::new(&context, &[offset_input], 0.001).unwrap();
            offset
                .update_prescribed_sphere_box_bodies(&[vec![Some(body.clone())]])
                .unwrap();
            offset.submit_steps(100).unwrap();
            let pose = offset.readback_static_sphere_box_poses().unwrap()[0][0];
            let delta = nalgebra::UnitQuaternion::from_scaled_axis(body.angular_velocity * 0.1);
            let expected =
                body.pose.translation.vector + body.linear_velocity * 0.1 + delta * Vector3::x();
            assert!(
                (pose.translation.vector - expected).norm() < 2e-5,
                "{backend:?}: {:?} vs {expected:?}",
                pose.translation.vector
            );
            assert!(pose.rotation.angle_to(&(delta * orientation)) < 1e-5);
            eprintln!("prescribed sphere/box motion passed on {backend:?}");
        }
        assert!(tested > 0, "no hardware backend available");
    }

    #[test]
    fn resident_prescribed_sphere_motion_changes_contact_velocity() {
        use crate::gpu_articulated_ground_contact::{
            GpuArticulatedExternalSphereMotions, GpuArticulatedGroundAxialKind,
            GpuArticulatedSphereMotion, GpuArticulatedSphereOrbit,
        };
        let articulation = Articulation::new(
            vec![link(0.0), link(1.0)],
            vec![JointSpec {
                parent: 0,
                child: 1,
                kind: JointKind::Prismatic,
                origin: Isometry3::identity(),
                axis: Vector3::z(),
                limits: None,
            }],
            0,
        )
        .unwrap();
        let mut input = GpuArticulatedDynamicsInput::new(
            &articulation,
            Isometry3::translation(0.0, 0.0, 0.5),
            GpuGeneralizedState {
                positions: DVector::zeros(1),
                velocities: DVector::zeros(1),
            },
            Vector3::new(0.0, 0.0, -9.81),
        );
        input
            .static_sphere_pairs
            .push(GpuArticulatedStaticSpherePair {
                link: 1,
                local_center: Vector3::zeros(),
                radius: 0.5,
                static_center: Vector3::new(0.0, 0.0, -0.5),
                static_radius: 0.5,
                restitution: 0.0,
                friction: 0.5,
            });
        let motion = GpuArticulatedSphereMotion {
            linear_velocity: Vector3::z() * 0.2,
            angular_velocity: Vector3::zeros(),
        };
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let mut batch =
                GpuArticulatedDynamicsBatch::new(&context, &[input.clone()], 0.001).unwrap();
            assert!(batch.update_static_sphere_motion(&[]).is_err());
            assert!(
                batch
                    .update_static_sphere_motion(&[vec![GpuArticulatedSphereMotion {
                        linear_velocity: Vector3::x() * f64::MAX,
                        ..motion
                    }]])
                    .is_err()
            );
            let mut all_motion = GpuArticulatedExternalSphereMotions {
                spheres: vec![vec![motion]],
                capsules: vec![vec![]],
                boxes: vec![vec![]],
                axial: vec![vec![]],
                convex: vec![vec![]],
            };
            let mut invalid = all_motion.clone();
            invalid.spheres[0][0].linear_velocity = Vector3::x();
            invalid.convex[0].push(motion);
            assert!(batch.update_external_sphere_motions(&invalid).is_err());
            all_motion.spheres[0][0] = GpuArticulatedSphereMotion::default();
            assert!(!batch.update_external_sphere_motions(&all_motion).unwrap());
            all_motion.spheres[0][0] = motion;
            assert!(batch.update_external_sphere_motions(&all_motion).unwrap());
            assert!(!batch.update_external_sphere_motions(&all_motion).unwrap());
            assert!(!batch.update_static_sphere_motion(&[vec![motion]]).unwrap());
            batch.set_static_sphere_integration(true);
            batch.submit_steps(1).unwrap();
            assert!(batch.readback().unwrap()[0].velocities[0] > 0.19);
            batch.submit_steps(9).unwrap();
            assert!((batch.readback().unwrap()[0].positions[0] - 0.002).abs() < 1e-5);
            let integrated = batch.readback_external_sphere_centers().unwrap();
            assert!((integrated.spheres[0][0].z + 0.498).abs() < 1e-6);
            assert_eq!(integrated.capsules.len(), 1);
            assert!(integrated.capsules[0].is_empty());
            assert!(
                batch
                    .update_static_sphere_motion(&[vec![GpuArticulatedSphereMotion::default()]])
                    .unwrap()
            );
            batch.submit_steps(1).unwrap();
            assert!(batch.readback().unwrap()[0].velocities[0] < 0.199);
            assert_eq!(
                batch.readback_external_sphere_centers().unwrap(),
                integrated
            );
            // The original CPU shadow equals this requested center, but the GPU
            // has moved it. Explicit pose updates must still overwrite it.
            assert!(
                batch
                    .update_static_sphere_centers(&[vec![Vector3::new(0.0, 0.0, -0.5)]])
                    .unwrap()
            );
            assert!(
                !batch
                    .update_static_sphere_centers(&[vec![Vector3::new(0.0, 0.0, -0.5)]])
                    .unwrap()
            );
            assert_eq!(
                batch.readback_external_sphere_centers().unwrap().spheres,
                vec![vec![Vector3::new(0.0, 0.0, -0.5)]]
            );
            let orbit = GpuArticulatedSphereOrbit {
                origin: Vector3::new(1.0, 0.0, -0.5),
                orientation: nalgebra::UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 0.3),
                linear_velocity: Vector3::y() * 0.1,
            };
            let orbit_motion = GpuArticulatedSphereMotion {
                linear_velocity: Vector3::zeros(),
                angular_velocity: Vector3::y() * 2.0,
            };
            assert!(
                batch
                    .update_static_sphere_motion(&[vec![orbit_motion]])
                    .unwrap()
            );
            assert!(batch.update_static_sphere_orbits(&[]).is_err());
            assert!(
                batch
                    .update_static_sphere_orbits(&[vec![Some(GpuArticulatedSphereOrbit {
                        origin: Vector3::x() * f64::MAX,
                        ..orbit
                    })]])
                    .is_err()
            );
            assert!(
                batch
                    .update_static_sphere_orbits(&[vec![Some(orbit)]])
                    .unwrap()
            );
            batch.submit_steps(100).unwrap();
            let expected_orbit_center = Vector3::new(1.0 - 0.2f64.cos(), 0.01, -0.5 + 0.2f64.sin());
            let actual_orbit_center =
                batch.readback_external_sphere_centers().unwrap().spheres[0][0];
            let actual_orbit =
                batch.readback_external_sphere_orbits().unwrap().spheres[0][0].unwrap();
            assert!((actual_orbit.origin - Vector3::new(1.0, 0.01, -0.5)).norm() < 1e-6);
            assert!((actual_orbit.linear_velocity - orbit.linear_velocity).norm() < 1e-7);
            let expected_orientation =
                nalgebra::UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.2)
                    * orbit.orientation;
            assert!(actual_orbit.orientation.angle_to(&expected_orientation) < 1e-5);
            assert!(
                (actual_orbit_center - expected_orbit_center).norm() < 1e-5,
                "{actual_orbit_center:?} vs {expected_orbit_center:?}"
            );
            assert!(batch.update_static_sphere_orbits(&[vec![None]]).unwrap());
            assert_eq!(
                batch.readback_external_sphere_orbits().unwrap().spheres,
                vec![vec![None]]
            );
            batch.submit_steps(3).unwrap();
            assert_eq!(
                batch.readback_external_sphere_centers().unwrap().spheres[0][0],
                actual_orbit_center
            );
            let rotating_articulation = Articulation::new(
                vec![link(0.0), link(1.0), link(1.0)],
                vec![
                    JointSpec {
                        parent: 0,
                        child: 1,
                        kind: JointKind::Prismatic,
                        origin: Isometry3::identity(),
                        axis: Vector3::z(),
                        limits: None,
                    },
                    JointSpec {
                        parent: 1,
                        child: 2,
                        kind: JointKind::Prismatic,
                        origin: Isometry3::identity(),
                        axis: Vector3::x(),
                        limits: None,
                    },
                ],
                0,
            )
            .unwrap();
            let mut rotating_input = GpuArticulatedDynamicsInput::new(
                &rotating_articulation,
                Isometry3::translation(0.0, 0.0, 0.5),
                GpuGeneralizedState {
                    positions: DVector::zeros(2),
                    velocities: DVector::zeros(2),
                },
                Vector3::new(0.0, 0.0, -9.81),
            );
            rotating_input
                .static_sphere_pairs
                .push(GpuArticulatedStaticSpherePair {
                    link: 2,
                    ..input.static_sphere_pairs[0]
                });
            let mut rotating =
                GpuArticulatedDynamicsBatch::new(&context, &[rotating_input], 0.001).unwrap();
            rotating.enable_contact_activity().unwrap();
            rotating
                .enable_contact_sleep(&[Some(crate::sleep::SleepSettings {
                    time_threshold: 0.002,
                    linear_velocity_threshold: 100.0,
                    angular_velocity_threshold: 100.0,
                    ..Default::default()
                })])
                .unwrap();
            rotating.submit_steps(2).unwrap();
            assert_eq!(
                rotating.readback_sleeping_coordinates().unwrap(),
                [vec![true; 2]]
            );
            assert!(
                rotating
                    .update_static_sphere_motion(&[vec![GpuArticulatedSphereMotion {
                        linear_velocity: Vector3::zeros(),
                        angular_velocity: Vector3::y() * 0.2,
                    }]])
                    .unwrap()
            );
            rotating.submit_steps(1).unwrap();
            let state = rotating.readback().unwrap().remove(0);
            assert!(
                state.velocities[1] > 0.005,
                "rotating support must transfer positive X velocity: {:?}",
                state.velocities
            );
            assert_eq!(
                rotating.readback_sleeping_coordinates().unwrap(),
                [vec![false; 2]]
            );
            for _ in 0..10 {
                rotating.submit_steps(1).unwrap();
                assert_eq!(
                    rotating.readback_sleeping_coordinates().unwrap(),
                    [vec![false; 2]]
                );
            }
            assert!(
                rotating
                    .update_static_sphere_motion(&[vec![GpuArticulatedSphereMotion::default()]])
                    .unwrap()
            );
            rotating.submit_steps(3).unwrap();
            assert_eq!(
                rotating.readback_sleeping_coordinates().unwrap(),
                [vec![true; 2]]
            );
            let mut capsule_input = GpuArticulatedDynamicsInput::new(
                &rotating_articulation,
                Isometry3::translation(0.0, 0.0, 0.5),
                GpuGeneralizedState {
                    positions: DVector::zeros(2),
                    velocities: DVector::zeros(2),
                },
                Vector3::new(0.0, 0.0, -9.81),
            );
            capsule_input
                .static_capsule_sphere_pairs
                .push(GpuArticulatedStaticCapsuleSpherePair {
                    link: 2,
                    local_a: Vector3::new(-0.2, 0.0, 0.0),
                    local_b: Vector3::new(0.2, 0.0, 0.0),
                    radius: 0.5,
                    static_center: Vector3::new(0.0, 0.0, -0.5),
                    static_radius: 0.5,
                    restitution: 0.0,
                    friction: 0.5,
                });
            let mut capsule =
                GpuArticulatedDynamicsBatch::new(&context, &[capsule_input], 0.001).unwrap();
            capsule.set_static_sphere_integration(true);
            assert!(capsule.update_static_capsule_sphere_motion(&[]).is_err());
            let motion = GpuArticulatedSphereMotion {
                linear_velocity: Vector3::z() * 0.2,
                angular_velocity: Vector3::y() * 0.2,
            };
            assert!(
                capsule
                    .update_static_capsule_sphere_motion(&[vec![motion]])
                    .unwrap()
            );
            assert!(
                !capsule
                    .update_static_capsule_sphere_motion(&[vec![motion]])
                    .unwrap()
            );
            capsule.submit_steps(1).unwrap();
            let state = capsule.readback().unwrap().remove(0);
            assert!(state.velocities[0] > 0.19);
            assert!(
                state.velocities[1] > 0.005,
                "rotating sphere must push capsule tangentially: {:?}",
                state.velocities
            );
            capsule.submit_steps(9).unwrap();
            assert!((capsule.readback().unwrap()[0].positions[0] - 0.002).abs() < 1e-5);
            let centers = capsule.readback_external_sphere_centers().unwrap();
            assert!((centers.capsules[0][0].z + 0.498).abs() < 1e-6);
            assert_eq!(centers.spheres.len(), 1);
            assert!(centers.spheres[0].is_empty());
            assert!(
                capsule
                    .update_static_capsule_sphere_motion(&[vec![
                        GpuArticulatedSphereMotion::default()
                    ]])
                    .unwrap()
            );
            assert!(
                capsule
                    .update_static_capsule_sphere_centers(&[vec![Vector3::new(0.0, 0.0, -0.5)]])
                    .unwrap()
            );
            assert!(
                !capsule
                    .update_static_capsule_sphere_centers(&[vec![Vector3::new(0.0, 0.0, -0.5)]])
                    .unwrap()
            );
            let mut box_input = GpuArticulatedDynamicsInput::new(
                &rotating_articulation,
                Isometry3::translation(0.0, 0.0, 0.5),
                GpuGeneralizedState {
                    positions: DVector::zeros(2),
                    velocities: DVector::zeros(2),
                },
                Vector3::new(0.0, 0.0, -9.81),
            );
            box_input
                .static_box_sphere_pairs
                .push(GpuArticulatedStaticBoxSpherePair {
                    link: 2,
                    local_pose: Isometry3::identity(),
                    half_extents: Vector3::new(0.2, 0.2, 0.5),
                    static_center: Vector3::new(0.0, 0.0, -0.5),
                    static_radius: 0.5,
                    restitution: 0.0,
                    friction: 0.5,
                });
            let mut boxes =
                GpuArticulatedDynamicsBatch::new(&context, &[box_input.clone(), box_input], 0.001)
                    .unwrap();
            boxes.set_static_sphere_integration(true);
            assert!(boxes.update_static_box_sphere_motion(&[]).is_err());
            let motions = vec![vec![motion], vec![GpuArticulatedSphereMotion::default()]];
            assert!(boxes.update_static_box_sphere_motion(&motions).unwrap());
            assert!(!boxes.update_static_box_sphere_motion(&motions).unwrap());
            boxes.submit_steps(1).unwrap();
            let states = boxes.readback().unwrap();
            assert!(states[0].velocities[0] > 0.19);
            assert!(
                states[0].velocities[1] > 0.005,
                "rotating sphere must push box tangentially: {:?}",
                states[0].velocities
            );
            assert!(states[1].velocities.norm() < 1e-5);
            boxes.submit_steps(9).unwrap();
            let states = boxes.readback().unwrap();
            assert!((states[0].positions[0] - 0.002).abs() < 1e-5);
            assert!(states[1].positions.norm() < 1e-5);
            let centers = boxes.readback_external_sphere_centers().unwrap();
            assert!((centers.boxes[0][0].z + 0.498).abs() < 1e-6);
            assert_eq!(centers.boxes[1], vec![Vector3::new(0.0, 0.0, -0.5)]);
            assert!(
                boxes
                    .update_static_box_sphere_motion(&vec![
                        vec![
                            GpuArticulatedSphereMotion::default()
                        ];
                        2
                    ])
                    .unwrap()
            );
            let centers = vec![vec![Vector3::new(0.0, 0.0, -0.5)]; 2];
            assert!(boxes.update_static_box_sphere_centers(&centers).unwrap());
            assert!(!boxes.update_static_box_sphere_centers(&centers).unwrap());
            let mut shapes = (0..3)
                .map(|_| {
                    GpuArticulatedDynamicsInput::new(
                        &rotating_articulation,
                        Isometry3::translation(0.0, 0.0, 0.5),
                        GpuGeneralizedState {
                            positions: DVector::zeros(2),
                            velocities: DVector::zeros(2),
                        },
                        Vector3::new(0.0, 0.0, -9.81),
                    )
                })
                .collect::<Vec<_>>();
            for (index, kind) in [
                GpuArticulatedGroundAxialKind::Cylinder,
                GpuArticulatedGroundAxialKind::Cone,
            ]
            .into_iter()
            .enumerate()
            {
                shapes[index]
                    .static_axial_sphere_pairs
                    .push(GpuArticulatedStaticAxialSpherePair {
                        link: 2,
                        axial_is_static: false,
                        kind,
                        local_pose: Isometry3::identity(),
                        half_height: 0.5,
                        radius: 0.3,
                        static_center: Vector3::new(0.0, 0.0, -0.5),
                        static_radius: 0.5,
                        restitution: 0.0,
                        friction: 0.5,
                    });
            }
            shapes[2]
                .static_convex_sphere_pairs
                .push(GpuArticulatedStaticConvexSpherePair {
                    convex_link: 2,
                    convex_local_pose: Isometry3::identity(),
                    vertices: [-0.2, 0.2]
                        .into_iter()
                        .flat_map(|x| {
                            [-0.2, 0.2].into_iter().flat_map(move |y| {
                                [-0.5, 0.5].into_iter().map(move |z| Vector3::new(x, y, z))
                            })
                        })
                        .collect(),
                    face_normals: vec![
                        Vector3::x(),
                        -Vector3::x(),
                        Vector3::y(),
                        -Vector3::y(),
                        Vector3::z(),
                        -Vector3::z(),
                    ],
                    static_center: Vector3::new(0.0, 0.0, -0.5),
                    static_radius: 0.5,
                    restitution: 0.0,
                    friction: 0.5,
                });
            let axial_motion = vec![vec![motion], vec![motion], vec![]];
            let convex_motion = vec![vec![], vec![], vec![motion]];
            let initial_motion = GpuArticulatedExternalSphereMotions {
                spheres: vec![vec![]; 3],
                capsules: vec![vec![]; 3],
                boxes: vec![vec![]; 3],
                axial: axial_motion.clone(),
                convex: convex_motion.clone(),
            };
            let mut shaped = GpuArticulatedDynamicsBatch::new(&context, &shapes, 0.001)
                .unwrap()
                .with_external_sphere_motions(&initial_motion, true)
                .unwrap();
            assert!(
                !shaped
                    .update_external_sphere_motions(&initial_motion)
                    .unwrap()
            );
            assert!(
                !shaped
                    .update_static_axial_sphere_motion(&axial_motion)
                    .unwrap()
            );
            assert!(
                !shaped
                    .update_static_convex_sphere_motion(&convex_motion)
                    .unwrap()
            );
            shaped.submit_steps(1).unwrap();
            for (index, state) in shaped.readback().unwrap().iter().enumerate() {
                assert!(
                    state.velocities[0] > 0.19,
                    "shape {index}: {:?}",
                    state.velocities
                );
                assert!(
                    state.velocities[1] > 0.005,
                    "rotating sphere must push shape {index} tangentially: {:?}",
                    state.velocities
                );
            }
            shaped.submit_steps(9).unwrap();
            for (index, state) in shaped.readback().unwrap().iter().enumerate() {
                assert!(
                    (state.positions[0] - 0.002).abs() < 1e-5,
                    "shape {index}: {:?}",
                    state.positions
                );
            }
            assert!(
                shaped
                    .update_static_axial_sphere_motion(&[
                        vec![GpuArticulatedSphereMotion::default()],
                        vec![GpuArticulatedSphereMotion::default()],
                        vec![]
                    ])
                    .unwrap()
            );
            assert!(
                shaped
                    .update_static_convex_sphere_motion(&[
                        vec![],
                        vec![],
                        vec![GpuArticulatedSphereMotion::default()]
                    ])
                    .unwrap()
            );
            let centers = shaped.readback_external_sphere_centers().unwrap();
            assert!((centers.axial[0][0].z + 0.498).abs() < 1e-6);
            assert!((centers.axial[1][0].z + 0.498).abs() < 1e-6);
            assert!(centers.axial[2].is_empty());
            assert!(centers.convex[0].is_empty());
            assert!(centers.convex[1].is_empty());
            assert!((centers.convex[2][0].z + 0.498).abs() < 1e-6);
            let axial_centers = vec![
                vec![Vector3::new(0.0, 0.0, -0.5)],
                vec![Vector3::new(0.0, 0.0, -0.5)],
                vec![],
            ];
            let convex_centers = vec![vec![], vec![], vec![Vector3::new(0.0, 0.0, -0.5)]];
            assert!(
                shaped
                    .update_static_axial_sphere_centers(&axial_centers)
                    .unwrap()
            );
            assert!(
                shaped
                    .update_static_convex_sphere_centers(&convex_centers)
                    .unwrap()
            );
            assert!(
                !shaped
                    .update_static_axial_sphere_centers(&axial_centers)
                    .unwrap()
            );
            assert!(
                !shaped
                    .update_static_convex_sphere_centers(&convex_centers)
                    .unwrap()
            );
            let centers = shaped.readback_external_sphere_centers().unwrap();
            assert_eq!(centers.axial, axial_centers);
            assert_eq!(centers.convex, convex_centers);
            let axial_motion = vec![vec![orbit_motion], vec![orbit_motion], vec![]];
            let convex_motion = vec![vec![], vec![], vec![orbit_motion]];
            assert!(
                shaped
                    .update_static_axial_sphere_motion(&axial_motion)
                    .unwrap()
            );
            assert!(
                shaped
                    .update_static_convex_sphere_motion(&convex_motion)
                    .unwrap()
            );
            assert!(
                shaped
                    .update_static_axial_sphere_orbits(&[
                        vec![Some(orbit)],
                        vec![Some(orbit)],
                        vec![]
                    ])
                    .unwrap()
            );
            assert!(
                shaped
                    .update_static_convex_sphere_orbits(&[vec![], vec![], vec![Some(orbit)]])
                    .unwrap()
            );
            shaped.submit_steps(100).unwrap();
            let centers = shaped.readback_external_sphere_centers().unwrap();
            for center in [
                centers.axial[0][0],
                centers.axial[1][0],
                centers.convex[2][0],
            ] {
                assert!(
                    (center - expected_orbit_center).norm() < 1e-5,
                    "{center:?} vs {expected_orbit_center:?}"
                );
            }
            let orbits = shaped.readback_external_sphere_orbits().unwrap();
            assert!(orbits.axial[2].is_empty());
            assert!(orbits.convex[0].is_empty());
            assert!(orbits.convex[1].is_empty());
            for actual in [orbits.axial[0][0], orbits.axial[1][0], orbits.convex[2][0]] {
                let actual = actual.unwrap();
                assert!((actual.origin - Vector3::new(1.0, 0.01, -0.5)).norm() < 1e-6);
                assert!((actual.linear_velocity - orbit.linear_velocity).norm() < 1e-7);
                assert!(actual.orientation.angle_to(&expected_orientation) < 1e-5);
            }
            shaped
                .queue
                .write_buffer(shaped.state.status_buffer(), 4, bytemuck::bytes_of(&1u32));
            assert!(matches!(
                shaped.readback_external_sphere_centers(),
                Err(GpuArticulatedDynamicsError::GroundContact(
                    GpuArticulatedGroundContactError::SourceFault(1)
                ))
            ));
            assert!(matches!(
                shaped.readback_external_sphere_orbits(),
                Err(GpuArticulatedDynamicsError::GroundContact(
                    GpuArticulatedGroundContactError::SourceFault(1)
                ))
            ));
            eprintln!("resident prescribed primitive/convex contact passed on {backend:?}");
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn resident_static_sphere_box_pose_update_wakes_without_rebuilding() {
        let articulation = Articulation::new(
            vec![link(0.0), link(1.0)],
            vec![JointSpec {
                parent: 0,
                child: 1,
                kind: JointKind::Prismatic,
                origin: Isometry3::identity(),
                axis: Vector3::z(),
                limits: None,
            }],
            0,
        )
        .unwrap();
        let state = GpuGeneralizedState {
            positions: DVector::zeros(1),
            velocities: DVector::zeros(1),
        };
        let mut input = GpuArticulatedDynamicsInput::new(
            &articulation,
            Isometry3::translation(0.0, 0.0, 0.5),
            state,
            Vector3::new(0.0, 0.0, -9.81),
        );
        let center = Isometry3::translation(0.0, 0.0, -0.5);
        input
            .static_sphere_box_pairs
            .push(GpuArticulatedStaticSphereBoxPair {
                link: 1,
                local_center: Vector3::zeros(),
                radius: 0.5,
                static_pose: center,
                half_extents: Vector3::new(0.1, 0.5, 0.5),
                restitution: 0.0,
                friction: 0.0,
            });
        let settings = crate::sleep::SleepSettings {
            time_threshold: 0.002,
            linear_velocity_threshold: 100.0,
            angular_velocity_threshold: 100.0,
            ..Default::default()
        };
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let mut batch =
                GpuArticulatedDynamicsBatch::new(&context, &[input.clone()], 0.001).unwrap();
            batch.enable_contact_activity().unwrap();
            batch.enable_contact_sleep(&[Some(settings)]).unwrap();
            batch.submit_steps(2).unwrap();
            assert_eq!(batch.readback_sleeping_coordinates().unwrap(), [vec![true]]);
            assert!(
                !batch
                    .update_static_sphere_box_poses(&[vec![center]])
                    .unwrap()
            );
            assert!(batch.update_static_sphere_box_poses(&[]).is_err());
            assert!(
                batch
                    .update_static_sphere_box_poses(&[vec![Isometry3::translation(
                        f64::MAX,
                        0.0,
                        0.0
                    )]])
                    .is_err()
            );
            let invalid_rotation = Isometry3::from_parts(
                center.translation,
                nalgebra::UnitQuaternion::new_unchecked(nalgebra::Quaternion::new(
                    2.0, 0.0, 0.0, 0.0,
                )),
            );
            assert!(
                batch
                    .update_static_sphere_box_poses(&[vec![invalid_rotation]])
                    .is_err()
            );
            assert_eq!(batch.readback_sleeping_coordinates().unwrap(), [vec![true]]);
            batch
                .update_contact_wake_requests(&[vec![true, false]])
                .unwrap();
            assert!(
                batch
                    .update_static_sphere_box_poses(&[vec![Isometry3::from_parts(
                        center.translation,
                        nalgebra::UnitQuaternion::from_axis_angle(
                            &Vector3::y_axis(),
                            core::f64::consts::FRAC_PI_2
                        )
                    )]])
                    .unwrap()
            );
            assert_eq!(
                batch.readback_sleeping_coordinates().unwrap(),
                [vec![false]]
            );
            batch.submit_steps(1).unwrap();
            assert_eq!(
                batch.readback_contact_wake_requests().unwrap()[0],
                [true, true]
            );
            assert_eq!(
                batch.readback_sleeping_coordinates().unwrap(),
                [vec![false]]
            );
            assert!(batch.readback().unwrap()[0].velocities[0] < 0.0);
            assert_eq!(batch.readback_geometric_contacts().unwrap()[0], [false; 2]);
            batch.submit_steps(1).unwrap();
            assert_eq!(
                batch.readback_contact_wake_requests().unwrap()[0],
                [false; 2]
            );
            eprintln!("resident static sphere/box pose update and wake passed on {backend:?}");
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn resident_static_axial_box_pose_update_wakes_without_rebuilding() {
        let articulation = Articulation::new(
            vec![link(0.0), link(1.0)],
            vec![JointSpec {
                parent: 0,
                child: 1,
                kind: JointKind::Prismatic,
                origin: Isometry3::identity(),
                axis: Vector3::z(),
                limits: None,
            }],
            0,
        )
        .unwrap();
        let state = GpuGeneralizedState {
            positions: DVector::zeros(1),
            velocities: DVector::zeros(1),
        };
        let mut input = GpuArticulatedDynamicsInput::new(
            &articulation,
            Isometry3::translation(0.0, 0.0, 0.5),
            state,
            Vector3::new(0.0, 0.0, -9.81),
        );
        let center = Isometry3::translation(0.0, 0.0, -0.5);
        input
            .static_axial_box_pairs
            .push(GpuArticulatedStaticAxialBoxPair {
                link: 1,
                local_pose: Isometry3::identity(),
                axial_is_static: false,
                kind:
                    crate::gpu_articulated_ground_contact::GpuArticulatedGroundAxialKind::Cylinder,
                half_height: 0.5,
                radius: 0.5,
                static_pose: center,
                static_half_extents: Vector3::new(0.1, 0.5, 0.5),
                restitution: 0.0,
                friction: 0.0,
            });
        let mut reverse = input.static_axial_box_pairs[0];
        reverse.axial_is_static = true;
        reverse.local_pose = Isometry3::translation(100.0, 0.0, 0.0);
        reverse.static_pose = Isometry3::identity();
        input.static_axial_box_pairs.push(reverse);
        let settings = crate::sleep::SleepSettings {
            time_threshold: 0.002,
            linear_velocity_threshold: 100.0,
            angular_velocity_threshold: 100.0,
            ..Default::default()
        };
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let mut batch =
                GpuArticulatedDynamicsBatch::new(&context, &[input.clone()], 0.001).unwrap();
            batch.enable_contact_activity().unwrap();
            batch.enable_contact_sleep(&[Some(settings)]).unwrap();
            batch.submit_steps(2).unwrap();
            assert_eq!(batch.readback_sleeping_coordinates().unwrap(), [vec![true]]);
            assert!(
                !batch
                    .update_static_axial_box_poses(&[vec![center]])
                    .unwrap()
            );
            assert!(batch.update_static_axial_box_poses(&[]).is_err());
            assert!(
                batch
                    .update_static_axial_box_poses(&[vec![center, Isometry3::identity()]])
                    .is_err()
            );
            assert!(
                batch
                    .update_static_axial_box_poses(&[vec![Isometry3::translation(
                        f64::MAX,
                        0.0,
                        0.0
                    )]])
                    .is_err()
            );
            let invalid_rotation = Isometry3::from_parts(
                center.translation,
                nalgebra::UnitQuaternion::new_unchecked(nalgebra::Quaternion::new(
                    2.0, 0.0, 0.0, 0.0,
                )),
            );
            assert!(
                batch
                    .update_static_axial_box_poses(&[vec![invalid_rotation]])
                    .is_err()
            );
            assert_eq!(batch.readback_sleeping_coordinates().unwrap(), [vec![true]]);
            batch
                .update_contact_wake_requests(&[vec![true, false]])
                .unwrap();
            assert!(
                batch
                    .update_static_axial_box_poses(&[vec![Isometry3::from_parts(
                        center.translation,
                        nalgebra::UnitQuaternion::from_axis_angle(
                            &Vector3::y_axis(),
                            core::f64::consts::FRAC_PI_2
                        )
                    )]])
                    .unwrap()
            );
            assert_eq!(
                batch.readback_sleeping_coordinates().unwrap(),
                [vec![false]]
            );
            batch.submit_steps(1).unwrap();
            assert_eq!(
                batch.readback_contact_wake_requests().unwrap()[0],
                [true, true]
            );
            assert_eq!(
                batch.readback_sleeping_coordinates().unwrap(),
                [vec![false]]
            );
            assert!(batch.readback().unwrap()[0].velocities[0] < 0.0);
            assert_eq!(batch.readback_geometric_contacts().unwrap()[0], [false; 2]);
            batch.submit_steps(1).unwrap();
            assert_eq!(
                batch.readback_contact_wake_requests().unwrap()[0],
                [false; 2]
            );
            eprintln!("resident static axial/box pose update and wake passed on {backend:?}");
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn resident_static_box_box_pose_update_wakes_without_rebuilding() {
        let articulation = Articulation::new(
            vec![link(0.0), link(1.0)],
            vec![JointSpec {
                parent: 0,
                child: 1,
                kind: JointKind::Prismatic,
                origin: Isometry3::identity(),
                axis: Vector3::z(),
                limits: None,
            }],
            0,
        )
        .unwrap();
        let state = GpuGeneralizedState {
            positions: DVector::zeros(1),
            velocities: DVector::zeros(1),
        };
        let mut input = GpuArticulatedDynamicsInput::new(
            &articulation,
            Isometry3::translation(0.0, 0.0, 0.5),
            state,
            Vector3::new(0.0, 0.0, -9.81),
        );
        let center = Isometry3::translation(0.0, 0.0, -0.5);
        input.static_box_pairs.push(GpuArticulatedStaticBoxPair {
            link: 1,
            local_pose: Isometry3::identity(),
            half_extents: Vector3::repeat(0.5),
            static_pose: center,
            static_half_extents: Vector3::new(0.1, 0.5, 0.5),
            restitution: 0.0,
            friction: 0.0,
        });
        let settings = crate::sleep::SleepSettings {
            time_threshold: 0.002,
            linear_velocity_threshold: 100.0,
            angular_velocity_threshold: 100.0,
            ..Default::default()
        };
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let mut batch =
                GpuArticulatedDynamicsBatch::new(&context, &[input.clone()], 0.001).unwrap();
            batch.enable_contact_activity().unwrap();
            batch.enable_contact_sleep(&[Some(settings)]).unwrap();
            batch.submit_steps(2).unwrap();
            assert_eq!(batch.readback_sleeping_coordinates().unwrap(), [vec![true]]);
            assert!(!batch.update_static_box_box_poses(&[vec![center]]).unwrap());
            assert!(batch.update_static_box_box_poses(&[]).is_err());
            assert!(
                batch
                    .update_static_box_box_poses(&[vec![Isometry3::translation(
                        f64::MAX,
                        0.0,
                        0.0
                    )]])
                    .is_err()
            );
            let invalid_rotation = Isometry3::from_parts(
                center.translation,
                nalgebra::UnitQuaternion::new_unchecked(nalgebra::Quaternion::new(
                    2.0, 0.0, 0.0, 0.0,
                )),
            );
            assert!(
                batch
                    .update_static_box_box_poses(&[vec![invalid_rotation]])
                    .is_err()
            );
            assert_eq!(batch.readback_sleeping_coordinates().unwrap(), [vec![true]]);
            batch
                .update_contact_wake_requests(&[vec![true, false]])
                .unwrap();
            assert!(
                batch
                    .update_static_box_box_poses(&[vec![Isometry3::from_parts(
                        center.translation,
                        nalgebra::UnitQuaternion::from_axis_angle(
                            &Vector3::y_axis(),
                            core::f64::consts::FRAC_PI_2
                        )
                    )]])
                    .unwrap()
            );
            assert_eq!(
                batch.readback_sleeping_coordinates().unwrap(),
                [vec![false]]
            );
            batch.submit_steps(1).unwrap();
            assert_eq!(
                batch.readback_contact_wake_requests().unwrap()[0],
                [true, true]
            );
            assert_eq!(
                batch.readback_sleeping_coordinates().unwrap(),
                [vec![false]]
            );
            assert!(batch.readback().unwrap()[0].velocities[0] < 0.0);
            assert_eq!(batch.readback_geometric_contacts().unwrap()[0], [false; 2]);
            batch.submit_steps(1).unwrap();
            assert_eq!(
                batch.readback_contact_wake_requests().unwrap()[0],
                [false; 2]
            );
            eprintln!("resident static box/box pose update and wake passed on {backend:?}");
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn resident_static_convex_pair_pose_update_wakes_without_rebuilding() {
        let articulation = Articulation::new(
            vec![link(0.0), link(1.0)],
            vec![JointSpec {
                parent: 0,
                child: 1,
                kind: JointKind::Prismatic,
                origin: Isometry3::identity(),
                axis: Vector3::z(),
                limits: None,
            }],
            0,
        )
        .unwrap();
        let state = GpuGeneralizedState {
            positions: DVector::zeros(1),
            velocities: DVector::zeros(1),
        };
        let mut input = GpuArticulatedDynamicsInput::new(
            &articulation,
            Isometry3::translation(0.0, 0.0, 0.5),
            state,
            Vector3::new(0.0, 0.0, -9.81),
        );
        let hull = |half: Vector3<f64>| {
            crate::convex::ConvexGeometry::new(
                [-1.0, 1.0]
                    .into_iter()
                    .flat_map(|x| {
                        [-1.0, 1.0].into_iter().flat_map(move |y| {
                            [-1.0, 1.0]
                                .into_iter()
                                .map(move |z| Vector3::new(x * half.x, y * half.y, z * half.z))
                        })
                    })
                    .collect(),
                vec![
                    Vector3::x(),
                    -Vector3::x(),
                    Vector3::y(),
                    -Vector3::y(),
                    Vector3::z(),
                    -Vector3::z(),
                ],
                vec![Vector3::x(), Vector3::y(), Vector3::z()],
            )
            .unwrap()
        };
        let center = Isometry3::translation(0.0, 0.0, -0.5);
        input
            .static_convex_pairs
            .push(GpuArticulatedStaticConvexPair {
                first_link: 1,
                first_local_pose: Isometry3::identity(),
                first_geometry: hull(Vector3::repeat(0.5)),
                second_world_pose: center,
                second_geometry: hull(Vector3::new(0.1, 0.5, 0.5)),
                restitution: 0.0,
                friction: 0.0,
            });
        let settings = crate::sleep::SleepSettings {
            time_threshold: 0.002,
            linear_velocity_threshold: 100.0,
            angular_velocity_threshold: 100.0,
            ..Default::default()
        };
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let mut batch =
                GpuArticulatedDynamicsBatch::new(&context, &[input.clone()], 0.001).unwrap();
            batch.enable_contact_activity().unwrap();
            batch.enable_contact_sleep(&[Some(settings)]).unwrap();
            batch.submit_steps(2).unwrap();
            assert_eq!(batch.readback_sleeping_coordinates().unwrap(), [vec![true]]);
            assert!(
                !batch
                    .update_static_convex_pair_poses(&[vec![center]])
                    .unwrap()
            );
            assert!(batch.update_static_convex_pair_poses(&[]).is_err());
            assert!(
                batch
                    .update_static_convex_pair_poses(&[vec![Isometry3::translation(
                        f64::MAX,
                        0.0,
                        0.0
                    )]])
                    .is_err()
            );
            let invalid_rotation = Isometry3::from_parts(
                center.translation,
                nalgebra::UnitQuaternion::new_unchecked(nalgebra::Quaternion::new(
                    2.0, 0.0, 0.0, 0.0,
                )),
            );
            assert!(
                batch
                    .update_static_convex_pair_poses(&[vec![invalid_rotation]])
                    .is_err()
            );
            assert_eq!(batch.readback_sleeping_coordinates().unwrap(), [vec![true]]);
            batch
                .update_contact_wake_requests(&[vec![true, false]])
                .unwrap();
            assert!(
                batch
                    .update_static_convex_pair_poses(&[vec![Isometry3::from_parts(
                        center.translation,
                        nalgebra::UnitQuaternion::from_axis_angle(
                            &Vector3::y_axis(),
                            core::f64::consts::FRAC_PI_2
                        )
                    )]])
                    .unwrap()
            );
            assert_eq!(
                batch.readback_sleeping_coordinates().unwrap(),
                [vec![false]]
            );
            batch.submit_steps(1).unwrap();
            assert_eq!(
                batch.readback_contact_wake_requests().unwrap()[0],
                [true, true]
            );
            assert_eq!(
                batch.readback_sleeping_coordinates().unwrap(),
                [vec![false]]
            );
            assert!(batch.readback().unwrap()[0].velocities[0] < 0.0);
            assert_eq!(batch.readback_geometric_contacts().unwrap()[0], [false; 2]);
            batch.submit_steps(1).unwrap();
            assert_eq!(
                batch.readback_contact_wake_requests().unwrap()[0],
                [false; 2]
            );
            eprintln!("resident static convex pair pose update and wake passed on {backend:?}");
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn resident_static_axial_convex_pose_update_wakes_without_rebuilding() {
        let articulation = Articulation::new(
            vec![link(0.0), link(1.0)],
            vec![JointSpec {
                parent: 0,
                child: 1,
                kind: JointKind::Prismatic,
                origin: Isometry3::identity(),
                axis: Vector3::z(),
                limits: None,
            }],
            0,
        )
        .unwrap();
        let state = GpuGeneralizedState {
            positions: DVector::zeros(1),
            velocities: DVector::zeros(1),
        };
        let mut input = GpuArticulatedDynamicsInput::new(
            &articulation,
            Isometry3::translation(0.0, 0.0, 0.5),
            state,
            Vector3::new(0.0, 0.0, -9.81),
        );
        let hull = |half: Vector3<f64>| {
            crate::convex::ConvexGeometry::new(
                [-1.0, 1.0]
                    .into_iter()
                    .flat_map(|x| {
                        [-1.0, 1.0].into_iter().flat_map(move |y| {
                            [-1.0, 1.0]
                                .into_iter()
                                .map(move |z| Vector3::new(x * half.x, y * half.y, z * half.z))
                        })
                    })
                    .collect(),
                vec![
                    Vector3::x(),
                    -Vector3::x(),
                    Vector3::y(),
                    -Vector3::y(),
                    Vector3::z(),
                    -Vector3::z(),
                ],
                vec![Vector3::x(), Vector3::y(), Vector3::z()],
            )
            .unwrap()
        };
        let center = Isometry3::translation(0.0, 0.0, -0.5);
        input
            .static_axial_convex_pairs
            .push(GpuArticulatedStaticAxialConvexPair {
                link: 1,
                local_pose: Isometry3::identity(),
                kind:
                    crate::gpu_articulated_ground_contact::GpuArticulatedGroundAxialKind::Cylinder,
                half_height: 0.5,
                radius: 0.1,
                static_pose: center,
                static_geometry: hull(Vector3::new(0.1, 0.5, 0.5)),
                restitution: 0.0,
                friction: 0.0,
            });
        let settings = crate::sleep::SleepSettings {
            time_threshold: 0.002,
            linear_velocity_threshold: 100.0,
            angular_velocity_threshold: 100.0,
            ..Default::default()
        };
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let mut batch =
                GpuArticulatedDynamicsBatch::new(&context, &[input.clone()], 0.001).unwrap();
            batch.enable_contact_activity().unwrap();
            batch.enable_contact_sleep(&[Some(settings)]).unwrap();
            batch.submit_steps(2).unwrap();
            assert_eq!(batch.readback_sleeping_coordinates().unwrap(), [vec![true]]);
            assert!(
                !batch
                    .update_static_axial_convex_poses(&[vec![center]])
                    .unwrap()
            );
            assert!(batch.update_static_axial_convex_poses(&[]).is_err());
            assert!(
                batch
                    .update_static_axial_convex_poses(&[vec![Isometry3::translation(
                        f64::MAX,
                        0.0,
                        0.0
                    )]])
                    .is_err()
            );
            let invalid_rotation = Isometry3::from_parts(
                center.translation,
                nalgebra::UnitQuaternion::new_unchecked(nalgebra::Quaternion::new(
                    2.0, 0.0, 0.0, 0.0,
                )),
            );
            assert!(
                batch
                    .update_static_axial_convex_poses(&[vec![invalid_rotation]])
                    .is_err()
            );
            assert_eq!(batch.readback_sleeping_coordinates().unwrap(), [vec![true]]);
            batch
                .update_contact_wake_requests(&[vec![true, false]])
                .unwrap();
            assert!(
                batch
                    .update_static_axial_convex_poses(&[vec![Isometry3::from_parts(
                        center.translation,
                        nalgebra::UnitQuaternion::from_axis_angle(
                            &Vector3::y_axis(),
                            core::f64::consts::FRAC_PI_2
                        )
                    )]])
                    .unwrap()
            );
            assert_eq!(
                batch.readback_sleeping_coordinates().unwrap(),
                [vec![false]]
            );
            batch.submit_steps(1).unwrap();
            assert_eq!(
                batch.readback_contact_wake_requests().unwrap()[0],
                [true, true]
            );
            assert_eq!(
                batch.readback_sleeping_coordinates().unwrap(),
                [vec![false]]
            );
            assert!(batch.readback().unwrap()[0].velocities[0] < 0.0);
            assert_eq!(batch.readback_geometric_contacts().unwrap()[0], [false; 2]);
            batch.submit_steps(1).unwrap();
            assert_eq!(
                batch.readback_contact_wake_requests().unwrap()[0],
                [false; 2]
            );
            eprintln!("resident static axial convex pose update and wake passed on {backend:?}");
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn resident_static_cone_convex_pose_update_wakes_without_rebuilding() {
        let articulation = Articulation::new(
            vec![link(0.0), link(1.0)],
            vec![JointSpec {
                parent: 0,
                child: 1,
                kind: JointKind::Prismatic,
                origin: Isometry3::identity(),
                axis: Vector3::z(),
                limits: None,
            }],
            0,
        )
        .unwrap();
        let state = GpuGeneralizedState {
            positions: DVector::zeros(1),
            velocities: DVector::zeros(1),
        };
        let mut input = GpuArticulatedDynamicsInput::new(
            &articulation,
            Isometry3::translation(0.0, 0.0, 0.5),
            state,
            Vector3::new(0.0, 0.0, -9.81),
        );
        let hull = |half: Vector3<f64>| {
            crate::convex::ConvexGeometry::new(
                [-1.0, 1.0]
                    .into_iter()
                    .flat_map(|x| {
                        [-1.0, 1.0].into_iter().flat_map(move |y| {
                            [-1.0, 1.0]
                                .into_iter()
                                .map(move |z| Vector3::new(x * half.x, y * half.y, z * half.z))
                        })
                    })
                    .collect(),
                vec![
                    Vector3::x(),
                    -Vector3::x(),
                    Vector3::y(),
                    -Vector3::y(),
                    Vector3::z(),
                    -Vector3::z(),
                ],
                vec![Vector3::x(), Vector3::y(), Vector3::z()],
            )
            .unwrap()
        };
        let center = Isometry3::translation(0.0, 0.0, -0.5);
        input
            .static_axial_convex_pairs
            .push(GpuArticulatedStaticAxialConvexPair {
                link: 1,
                local_pose: Isometry3::identity(),
                kind: crate::gpu_articulated_ground_contact::GpuArticulatedGroundAxialKind::Cone,
                half_height: 0.5,
                radius: 0.1,
                static_pose: center,
                static_geometry: hull(Vector3::new(0.1, 0.5, 0.5)),
                restitution: 0.0,
                friction: 0.0,
            });
        let settings = crate::sleep::SleepSettings {
            time_threshold: 0.002,
            linear_velocity_threshold: 100.0,
            angular_velocity_threshold: 100.0,
            ..Default::default()
        };
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let mut batch =
                GpuArticulatedDynamicsBatch::new(&context, &[input.clone()], 0.001).unwrap();
            batch.enable_contact_activity().unwrap();
            batch.enable_contact_sleep(&[Some(settings)]).unwrap();
            batch.submit_steps(2).unwrap();
            assert_eq!(batch.readback_sleeping_coordinates().unwrap(), [vec![true]]);
            assert!(
                !batch
                    .update_static_axial_convex_poses(&[vec![center]])
                    .unwrap()
            );
            assert!(batch.update_static_axial_convex_poses(&[]).is_err());
            assert!(
                batch
                    .update_static_axial_convex_poses(&[vec![Isometry3::translation(
                        f64::MAX,
                        0.0,
                        0.0
                    )]])
                    .is_err()
            );
            let invalid_rotation = Isometry3::from_parts(
                center.translation,
                nalgebra::UnitQuaternion::new_unchecked(nalgebra::Quaternion::new(
                    2.0, 0.0, 0.0, 0.0,
                )),
            );
            assert!(
                batch
                    .update_static_axial_convex_poses(&[vec![invalid_rotation]])
                    .is_err()
            );
            assert_eq!(batch.readback_sleeping_coordinates().unwrap(), [vec![true]]);
            batch
                .update_contact_wake_requests(&[vec![true, false]])
                .unwrap();
            assert!(
                batch
                    .update_static_axial_convex_poses(&[vec![Isometry3::from_parts(
                        center.translation,
                        nalgebra::UnitQuaternion::from_axis_angle(
                            &Vector3::y_axis(),
                            core::f64::consts::FRAC_PI_2
                        )
                    )]])
                    .unwrap()
            );
            assert_eq!(
                batch.readback_sleeping_coordinates().unwrap(),
                [vec![false]]
            );
            batch.submit_steps(1).unwrap();
            assert_eq!(
                batch.readback_contact_wake_requests().unwrap()[0],
                [true, true]
            );
            assert_eq!(
                batch.readback_sleeping_coordinates().unwrap(),
                [vec![false]]
            );
            assert!(batch.readback().unwrap()[0].velocities[0] < 0.0);
            assert_eq!(batch.readback_geometric_contacts().unwrap()[0], [false; 2]);
            batch.submit_steps(1).unwrap();
            assert_eq!(
                batch.readback_contact_wake_requests().unwrap()[0],
                [false; 2]
            );
            eprintln!("resident static axial convex pose update and wake passed on {backend:?}");
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn resident_scene_convex_rounded_pose_update_wakes_without_rebuilding() {
        let articulation = Articulation::new(
            vec![link(0.0), link(1.0)],
            vec![JointSpec {
                parent: 0,
                child: 1,
                kind: JointKind::Prismatic,
                origin: Isometry3::identity(),
                axis: Vector3::z(),
                limits: None,
            }],
            0,
        )
        .unwrap();
        let state = GpuGeneralizedState {
            positions: DVector::zeros(1),
            velocities: DVector::zeros(1),
        };
        let mut input = GpuArticulatedDynamicsInput::new(
            &articulation,
            Isometry3::translation(0.0, 0.0, 0.5),
            state,
            Vector3::new(0.0, 0.0, -9.81),
        );
        let hull = |half: Vector3<f64>| {
            crate::convex::ConvexGeometry::new(
                [-1.0, 1.0]
                    .into_iter()
                    .flat_map(|x| {
                        [-1.0, 1.0].into_iter().flat_map(move |y| {
                            [-1.0, 1.0]
                                .into_iter()
                                .map(move |z| Vector3::new(x * half.x, y * half.y, z * half.z))
                        })
                    })
                    .collect(),
                vec![
                    Vector3::x(),
                    -Vector3::x(),
                    Vector3::y(),
                    -Vector3::y(),
                    Vector3::z(),
                    -Vector3::z(),
                ],
                vec![Vector3::x(), Vector3::y(), Vector3::z()],
            )
            .unwrap()
        };
        let center = Isometry3::translation(0.0, 0.0, -0.5);
        let geometry = hull(Vector3::new(0.1, 0.5, 0.5));
        input
            .scene_convex_sphere_pairs
            .push(GpuArticulatedSceneConvexSpherePair {
                convex_world_pose: center,
                vertices: geometry.vertices.clone(),
                face_normals: geometry.face_normals.clone(),
                sphere_link: 1,
                sphere_local_center: Vector3::zeros(),
                sphere_radius: 0.5,
                restitution: 0.0,
                friction: 0.0,
            });
        let settings = crate::sleep::SleepSettings {
            time_threshold: 0.002,
            linear_velocity_threshold: 100.0,
            angular_velocity_threshold: 100.0,
            ..Default::default()
        };
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let mut batch =
                GpuArticulatedDynamicsBatch::new(&context, &[input.clone()], 0.001).unwrap();
            batch.enable_contact_activity().unwrap();
            batch.enable_contact_sleep(&[Some(settings)]).unwrap();
            batch.submit_steps(2).unwrap();
            assert_eq!(batch.readback_sleeping_coordinates().unwrap(), [vec![true]]);
            assert!(
                !batch
                    .update_scene_convex_rounded_poses(&[vec![center]])
                    .unwrap()
            );
            assert!(batch.update_scene_convex_rounded_poses(&[]).is_err());
            assert!(
                batch
                    .update_scene_convex_rounded_poses(&[vec![Isometry3::translation(
                        f64::MAX,
                        0.0,
                        0.0
                    )]])
                    .is_err()
            );
            let invalid_rotation = Isometry3::from_parts(
                center.translation,
                nalgebra::UnitQuaternion::new_unchecked(nalgebra::Quaternion::new(
                    2.0, 0.0, 0.0, 0.0,
                )),
            );
            assert!(
                batch
                    .update_scene_convex_rounded_poses(&[vec![invalid_rotation]])
                    .is_err()
            );
            assert_eq!(batch.readback_sleeping_coordinates().unwrap(), [vec![true]]);
            batch
                .update_contact_wake_requests(&[vec![true, false]])
                .unwrap();
            assert!(
                batch
                    .update_scene_convex_rounded_poses(&[vec![Isometry3::from_parts(
                        center.translation,
                        nalgebra::UnitQuaternion::from_axis_angle(
                            &Vector3::y_axis(),
                            core::f64::consts::FRAC_PI_2
                        )
                    )]])
                    .unwrap()
            );
            assert_eq!(
                batch.readback_sleeping_coordinates().unwrap(),
                [vec![false]]
            );
            batch.submit_steps(1).unwrap();
            assert_eq!(
                batch.readback_contact_wake_requests().unwrap()[0],
                [true, true]
            );
            assert_eq!(
                batch.readback_sleeping_coordinates().unwrap(),
                [vec![false]]
            );
            assert!(batch.readback().unwrap()[0].velocities[0] < 0.0);
            assert_eq!(batch.readback_geometric_contacts().unwrap()[0], [false; 2]);
            batch.submit_steps(1).unwrap();
            assert_eq!(
                batch.readback_contact_wake_requests().unwrap()[0],
                [false; 2]
            );
            eprintln!("resident scene convex rounded pose update and wake passed on {backend:?}");
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn resident_scene_convex_capsule_pose_update_wakes_without_rebuilding() {
        let articulation = Articulation::new(
            vec![link(0.0), link(1.0)],
            vec![JointSpec {
                parent: 0,
                child: 1,
                kind: JointKind::Prismatic,
                origin: Isometry3::identity(),
                axis: Vector3::z(),
                limits: None,
            }],
            0,
        )
        .unwrap();
        let state = GpuGeneralizedState {
            positions: DVector::zeros(1),
            velocities: DVector::zeros(1),
        };
        let mut input = GpuArticulatedDynamicsInput::new(
            &articulation,
            Isometry3::translation(0.0, 0.0, 0.5),
            state,
            Vector3::new(0.0, 0.0, -9.81),
        );
        let hull = |half: Vector3<f64>| {
            crate::convex::ConvexGeometry::new(
                [-1.0, 1.0]
                    .into_iter()
                    .flat_map(|x| {
                        [-1.0, 1.0].into_iter().flat_map(move |y| {
                            [-1.0, 1.0]
                                .into_iter()
                                .map(move |z| Vector3::new(x * half.x, y * half.y, z * half.z))
                        })
                    })
                    .collect(),
                vec![
                    Vector3::x(),
                    -Vector3::x(),
                    Vector3::y(),
                    -Vector3::y(),
                    Vector3::z(),
                    -Vector3::z(),
                ],
                vec![Vector3::x(), Vector3::y(), Vector3::z()],
            )
            .unwrap()
        };
        let center = Isometry3::translation(0.0, 0.0, -0.5);
        let geometry = hull(Vector3::new(0.1, 0.5, 0.5));
        input
            .scene_convex_capsule_pairs
            .push(GpuArticulatedSceneConvexCapsulePair {
                convex_world_pose: center,
                vertices: geometry.vertices.clone(),
                face_normals: geometry.face_normals.clone(),
                capsule_link: 1,
                capsule_local_a: Vector3::zeros(),
                capsule_local_b: Vector3::new(0.0, 0.0, 0.25),
                edge_directions: geometry.edge_directions.clone(),
                capsule_radius: 0.5,
                restitution: 0.0,
                friction: 0.0,
            });
        let settings = crate::sleep::SleepSettings {
            time_threshold: 0.002,
            linear_velocity_threshold: 100.0,
            angular_velocity_threshold: 100.0,
            ..Default::default()
        };
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let mut batch =
                GpuArticulatedDynamicsBatch::new(&context, &[input.clone()], 0.001).unwrap();
            batch.enable_contact_activity().unwrap();
            batch.enable_contact_sleep(&[Some(settings)]).unwrap();
            batch.submit_steps(2).unwrap();
            assert_eq!(batch.readback_sleeping_coordinates().unwrap(), [vec![true]]);
            assert!(
                !batch
                    .update_scene_convex_rounded_poses(&[vec![center]])
                    .unwrap()
            );
            assert!(batch.update_scene_convex_rounded_poses(&[]).is_err());
            assert!(
                batch
                    .update_scene_convex_rounded_poses(&[vec![Isometry3::translation(
                        f64::MAX,
                        0.0,
                        0.0
                    )]])
                    .is_err()
            );
            let invalid_rotation = Isometry3::from_parts(
                center.translation,
                nalgebra::UnitQuaternion::new_unchecked(nalgebra::Quaternion::new(
                    2.0, 0.0, 0.0, 0.0,
                )),
            );
            assert!(
                batch
                    .update_scene_convex_rounded_poses(&[vec![invalid_rotation]])
                    .is_err()
            );
            assert_eq!(batch.readback_sleeping_coordinates().unwrap(), [vec![true]]);
            batch
                .update_contact_wake_requests(&[vec![true, false]])
                .unwrap();
            assert!(
                batch
                    .update_scene_convex_rounded_poses(&[vec![Isometry3::from_parts(
                        center.translation,
                        nalgebra::UnitQuaternion::from_axis_angle(
                            &Vector3::y_axis(),
                            core::f64::consts::FRAC_PI_2
                        )
                    )]])
                    .unwrap()
            );
            assert_eq!(
                batch.readback_sleeping_coordinates().unwrap(),
                [vec![false]]
            );
            batch.submit_steps(1).unwrap();
            assert_eq!(
                batch.readback_contact_wake_requests().unwrap()[0],
                [true, true]
            );
            assert_eq!(
                batch.readback_sleeping_coordinates().unwrap(),
                [vec![false]]
            );
            assert!(batch.readback().unwrap()[0].velocities[0] < 0.0);
            assert_eq!(batch.readback_geometric_contacts().unwrap()[0], [false; 2]);
            batch.submit_steps(1).unwrap();
            assert_eq!(
                batch.readback_contact_wake_requests().unwrap()[0],
                [false; 2]
            );
            eprintln!("resident scene convex rounded pose update and wake passed on {backend:?}");
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn resident_static_capsule_box_pose_update_wakes_without_rebuilding() {
        let articulation = Articulation::new(
            vec![link(0.0), link(1.0)],
            vec![JointSpec {
                parent: 0,
                child: 1,
                kind: JointKind::Prismatic,
                origin: Isometry3::identity(),
                axis: Vector3::z(),
                limits: None,
            }],
            0,
        )
        .unwrap();
        let state = GpuGeneralizedState {
            positions: DVector::zeros(1),
            velocities: DVector::zeros(1),
        };
        let mut input = GpuArticulatedDynamicsInput::new(
            &articulation,
            Isometry3::translation(0.0, 0.0, 0.5),
            state,
            Vector3::new(0.0, 0.0, -9.81),
        );
        let center = Isometry3::translation(0.0, 0.0, -0.5);
        input
            .static_capsule_box_pairs
            .push(GpuArticulatedStaticCapsuleBoxPair {
                link: 1,
                local_a: Vector3::zeros(),
                local_b: Vector3::new(0.0, 0.0, 0.25),
                radius: 0.5,
                static_pose: center,
                half_extents: Vector3::new(0.1, 0.5, 0.5),
                restitution: 0.0,
                friction: 0.0,
            });
        let settings = crate::sleep::SleepSettings {
            time_threshold: 0.002,
            linear_velocity_threshold: 100.0,
            angular_velocity_threshold: 100.0,
            ..Default::default()
        };
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let mut batch =
                GpuArticulatedDynamicsBatch::new(&context, &[input.clone()], 0.001).unwrap();
            batch.enable_contact_activity().unwrap();
            batch.enable_contact_sleep(&[Some(settings)]).unwrap();
            batch.submit_steps(2).unwrap();
            assert_eq!(batch.readback_sleeping_coordinates().unwrap(), [vec![true]]);
            assert!(
                !batch
                    .update_static_capsule_box_poses(&[vec![center]])
                    .unwrap()
            );
            assert!(batch.update_static_capsule_box_poses(&[]).is_err());
            assert!(
                batch
                    .update_static_capsule_box_poses(&[vec![Isometry3::translation(
                        f64::MAX,
                        0.0,
                        0.0
                    )]])
                    .is_err()
            );
            let invalid_rotation = Isometry3::from_parts(
                center.translation,
                nalgebra::UnitQuaternion::new_unchecked(nalgebra::Quaternion::new(
                    2.0, 0.0, 0.0, 0.0,
                )),
            );
            assert!(
                batch
                    .update_static_capsule_box_poses(&[vec![invalid_rotation]])
                    .is_err()
            );
            assert_eq!(batch.readback_sleeping_coordinates().unwrap(), [vec![true]]);
            batch
                .update_contact_wake_requests(&[vec![true, false]])
                .unwrap();
            assert!(
                batch
                    .update_static_capsule_box_poses(&[vec![Isometry3::from_parts(
                        center.translation,
                        nalgebra::UnitQuaternion::from_axis_angle(
                            &Vector3::y_axis(),
                            core::f64::consts::FRAC_PI_2
                        )
                    )]])
                    .unwrap()
            );
            assert_eq!(
                batch.readback_sleeping_coordinates().unwrap(),
                [vec![false]]
            );
            batch.submit_steps(1).unwrap();
            assert_eq!(
                batch.readback_contact_wake_requests().unwrap()[0],
                [true, true]
            );
            assert_eq!(
                batch.readback_sleeping_coordinates().unwrap(),
                [vec![false]]
            );
            assert!(batch.readback().unwrap()[0].velocities[0] < 0.0);
            assert_eq!(batch.readback_geometric_contacts().unwrap()[0], [false; 2]);
            batch.submit_steps(1).unwrap();
            assert_eq!(
                batch.readback_contact_wake_requests().unwrap()[0],
                [false; 2]
            );
            eprintln!("resident static capsule/box pose update and wake passed on {backend:?}");
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn resident_static_capsule_sphere_center_update_wakes_without_rebuilding() {
        let articulation = Articulation::new(
            vec![link(0.0), link(1.0)],
            vec![JointSpec {
                parent: 0,
                child: 1,
                kind: JointKind::Prismatic,
                origin: Isometry3::identity(),
                axis: Vector3::z(),
                limits: None,
            }],
            0,
        )
        .unwrap();
        let state = GpuGeneralizedState {
            positions: DVector::zeros(1),
            velocities: DVector::zeros(1),
        };
        let mut input = GpuArticulatedDynamicsInput::new(
            &articulation,
            Isometry3::translation(0.0, 0.0, 0.5),
            state,
            Vector3::new(0.0, 0.0, -9.81),
        );
        let center = Vector3::new(0.0, 0.0, -0.5);
        input
            .static_capsule_sphere_pairs
            .push(GpuArticulatedStaticCapsuleSpherePair {
                link: 1,
                local_a: Vector3::zeros(),
                local_b: Vector3::new(0.0, 0.0, 0.25),
                radius: 0.5,
                static_center: center,
                static_radius: 0.5,
                restitution: 0.0,
                friction: 0.0,
            });
        let settings = crate::sleep::SleepSettings {
            time_threshold: 0.002,
            linear_velocity_threshold: 100.0,
            angular_velocity_threshold: 100.0,
            ..Default::default()
        };
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let mut batch =
                GpuArticulatedDynamicsBatch::new(&context, &[input.clone()], 0.001).unwrap();
            batch.enable_contact_activity().unwrap();
            batch.enable_contact_sleep(&[Some(settings)]).unwrap();
            batch.submit_steps(2).unwrap();
            assert_eq!(batch.readback_sleeping_coordinates().unwrap(), [vec![true]]);
            assert!(
                !batch
                    .update_static_capsule_sphere_centers(&[vec![center]])
                    .unwrap()
            );
            assert!(batch.update_static_capsule_sphere_centers(&[]).is_err());
            assert!(
                batch
                    .update_static_capsule_sphere_centers(&[vec![Vector3::new(f64::MAX, 0.0, 0.0)]])
                    .is_err()
            );
            assert_eq!(batch.readback_sleeping_coordinates().unwrap(), [vec![true]]);
            batch
                .update_contact_wake_requests(&[vec![true, false]])
                .unwrap();
            assert!(
                batch
                    .update_static_capsule_sphere_centers(&[vec![Vector3::new(2.0, 0.0, -0.5)]])
                    .unwrap()
            );
            assert_eq!(
                batch.readback_sleeping_coordinates().unwrap(),
                [vec![false]]
            );
            batch.submit_steps(1).unwrap();
            assert_eq!(
                batch.readback_contact_wake_requests().unwrap()[0],
                [true, true]
            );
            assert_eq!(
                batch.readback_sleeping_coordinates().unwrap(),
                [vec![false]]
            );
            assert!(batch.readback().unwrap()[0].velocities[0] < 0.0);
            assert_eq!(batch.readback_geometric_contacts().unwrap()[0], [false; 2]);
            batch.submit_steps(1).unwrap();
            assert_eq!(
                batch.readback_contact_wake_requests().unwrap()[0],
                [false; 2]
            );
            eprintln!(
                "resident static capsule/sphere center update and wake passed on {backend:?}"
            );
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn resident_static_box_sphere_center_update_wakes_without_rebuilding() {
        let articulation = Articulation::new(
            vec![link(0.0), link(1.0)],
            vec![JointSpec {
                parent: 0,
                child: 1,
                kind: JointKind::Prismatic,
                origin: Isometry3::identity(),
                axis: Vector3::z(),
                limits: None,
            }],
            0,
        )
        .unwrap();
        let state = GpuGeneralizedState {
            positions: DVector::zeros(1),
            velocities: DVector::zeros(1),
        };
        let mut input = GpuArticulatedDynamicsInput::new(
            &articulation,
            Isometry3::translation(0.0, 0.0, 0.5),
            state,
            Vector3::new(0.0, 0.0, -9.81),
        );
        let center = Vector3::new(0.0, 0.0, -0.5);
        input
            .static_box_sphere_pairs
            .push(GpuArticulatedStaticBoxSpherePair {
                link: 1,
                local_pose: Isometry3::identity(),
                half_extents: Vector3::repeat(0.5),
                static_center: center,
                static_radius: 0.5,
                restitution: 0.0,
                friction: 0.0,
            });
        let settings = crate::sleep::SleepSettings {
            time_threshold: 0.002,
            linear_velocity_threshold: 100.0,
            angular_velocity_threshold: 100.0,
            ..Default::default()
        };
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let mut batch =
                GpuArticulatedDynamicsBatch::new(&context, &[input.clone()], 0.001).unwrap();
            batch.enable_contact_activity().unwrap();
            batch.enable_contact_sleep(&[Some(settings)]).unwrap();
            batch.submit_steps(2).unwrap();
            assert_eq!(batch.readback_sleeping_coordinates().unwrap(), [vec![true]]);
            assert!(
                !batch
                    .update_static_box_sphere_centers(&[vec![center]])
                    .unwrap()
            );
            assert!(batch.update_static_box_sphere_centers(&[]).is_err());
            assert!(
                batch
                    .update_static_box_sphere_centers(&[vec![Vector3::new(f64::MAX, 0.0, 0.0)]])
                    .is_err()
            );
            assert_eq!(batch.readback_sleeping_coordinates().unwrap(), [vec![true]]);
            batch
                .update_contact_wake_requests(&[vec![true, false]])
                .unwrap();
            assert!(
                batch
                    .update_static_box_sphere_centers(&[vec![Vector3::new(2.0, 0.0, -0.5)]])
                    .unwrap()
            );
            assert_eq!(
                batch.readback_sleeping_coordinates().unwrap(),
                [vec![false]]
            );
            batch.submit_steps(1).unwrap();
            assert_eq!(
                batch.readback_contact_wake_requests().unwrap()[0],
                [true, true]
            );
            assert_eq!(
                batch.readback_sleeping_coordinates().unwrap(),
                [vec![false]]
            );
            assert!(batch.readback().unwrap()[0].velocities[0] < 0.0);
            assert_eq!(batch.readback_geometric_contacts().unwrap()[0], [false; 2]);
            batch.submit_steps(1).unwrap();
            assert_eq!(
                batch.readback_contact_wake_requests().unwrap()[0],
                [false; 2]
            );
            eprintln!("resident static box/sphere center update and wake passed on {backend:?}");
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn resident_static_convex_sphere_center_update_wakes_without_rebuilding() {
        let articulation = Articulation::new(
            vec![link(0.0), link(1.0)],
            vec![JointSpec {
                parent: 0,
                child: 1,
                kind: JointKind::Prismatic,
                origin: Isometry3::identity(),
                axis: Vector3::z(),
                limits: None,
            }],
            0,
        )
        .unwrap();
        let state = GpuGeneralizedState {
            positions: DVector::zeros(1),
            velocities: DVector::zeros(1),
        };
        let mut input = GpuArticulatedDynamicsInput::new(
            &articulation,
            Isometry3::translation(0.0, 0.0, 0.5),
            state,
            Vector3::new(0.0, 0.0, -9.81),
        );
        let center = Vector3::new(0.0, 0.0, -0.5);
        input
            .static_convex_sphere_pairs
            .push(GpuArticulatedStaticConvexSpherePair {
                convex_link: 1,
                convex_local_pose: Isometry3::identity(),
                vertices: [-0.5, 0.5]
                    .into_iter()
                    .flat_map(|x| {
                        [-0.5, 0.5].into_iter().flat_map(move |y| {
                            [-0.5, 0.5].into_iter().map(move |z| Vector3::new(x, y, z))
                        })
                    })
                    .collect(),
                face_normals: vec![
                    Vector3::x(),
                    -Vector3::x(),
                    Vector3::y(),
                    -Vector3::y(),
                    Vector3::z(),
                    -Vector3::z(),
                ],
                static_center: center,
                static_radius: 0.5,
                restitution: 0.0,
                friction: 0.0,
            });
        let settings = crate::sleep::SleepSettings {
            time_threshold: 0.002,
            linear_velocity_threshold: 100.0,
            angular_velocity_threshold: 100.0,
            ..Default::default()
        };
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let mut batch =
                GpuArticulatedDynamicsBatch::new(&context, &[input.clone()], 0.001).unwrap();
            batch.enable_contact_activity().unwrap();
            batch.enable_contact_sleep(&[Some(settings)]).unwrap();
            batch.submit_steps(2).unwrap();
            assert_eq!(batch.readback_sleeping_coordinates().unwrap(), [vec![true]]);
            assert!(
                !batch
                    .update_static_convex_sphere_centers(&[vec![center]])
                    .unwrap()
            );
            assert!(batch.update_static_convex_sphere_centers(&[]).is_err());
            assert!(
                batch
                    .update_static_convex_sphere_centers(&[vec![Vector3::new(f64::MAX, 0.0, 0.0)]])
                    .is_err()
            );
            assert_eq!(batch.readback_sleeping_coordinates().unwrap(), [vec![true]]);
            batch
                .update_contact_wake_requests(&[vec![true, false]])
                .unwrap();
            assert!(
                batch
                    .update_static_convex_sphere_centers(&[vec![Vector3::new(2.0, 0.0, -0.5)]])
                    .unwrap()
            );
            assert_eq!(
                batch.readback_sleeping_coordinates().unwrap(),
                [vec![false]]
            );
            batch.submit_steps(1).unwrap();
            assert_eq!(
                batch.readback_contact_wake_requests().unwrap()[0],
                [true, true]
            );
            assert_eq!(
                batch.readback_sleeping_coordinates().unwrap(),
                [vec![false]]
            );
            assert!(batch.readback().unwrap()[0].velocities[0] < 0.0);
            assert_eq!(batch.readback_geometric_contacts().unwrap()[0], [false; 2]);
            batch.submit_steps(1).unwrap();
            assert_eq!(
                batch.readback_contact_wake_requests().unwrap()[0],
                [false; 2]
            );
            eprintln!("resident static convex/sphere center update and wake passed on {backend:?}");
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn resident_static_axial_sphere_center_update_wakes_without_rebuilding() {
        let articulation = Articulation::new(
            vec![link(0.0), link(1.0)],
            vec![JointSpec {
                parent: 0,
                child: 1,
                kind: JointKind::Prismatic,
                origin: Isometry3::identity(),
                axis: Vector3::z(),
                limits: None,
            }],
            0,
        )
        .unwrap();
        let state = GpuGeneralizedState {
            positions: DVector::zeros(1),
            velocities: DVector::zeros(1),
        };
        let mut input = GpuArticulatedDynamicsInput::new(
            &articulation,
            Isometry3::translation(0.0, 0.0, 0.5),
            state,
            Vector3::new(0.0, 0.0, -9.81),
        );
        let center = Vector3::new(0.0, 0.0, -0.5);
        input
            .static_axial_sphere_pairs
            .push(GpuArticulatedStaticAxialSpherePair {
                link: 1,
                local_pose: Isometry3::identity(),
                axial_is_static: false,
                kind:
                    crate::gpu_articulated_ground_contact::GpuArticulatedGroundAxialKind::Cylinder,
                half_height: 0.5,
                radius: 0.5,
                static_center: center,
                static_radius: 0.5,
                restitution: 0.0,
                friction: 0.0,
            });
        // The reverse orientation owns a link-local sphere center and must not
        // appear in the stationary-sphere update ordering.
        let mut reverse = input.static_axial_sphere_pairs[0];
        reverse.axial_is_static = true;
        reverse.local_pose = Isometry3::translation(100.0, 0.0, 0.0);
        reverse.static_center = Vector3::zeros();
        input.static_axial_sphere_pairs.push(reverse);
        let settings = crate::sleep::SleepSettings {
            time_threshold: 0.002,
            linear_velocity_threshold: 100.0,
            angular_velocity_threshold: 100.0,
            ..Default::default()
        };
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let mut batch =
                GpuArticulatedDynamicsBatch::new(&context, &[input.clone()], 0.001).unwrap();
            batch.enable_contact_activity().unwrap();
            batch.enable_contact_sleep(&[Some(settings)]).unwrap();
            batch.submit_steps(2).unwrap();
            assert_eq!(batch.readback_sleeping_coordinates().unwrap(), [vec![true]]);
            assert!(
                !batch
                    .update_static_axial_sphere_centers(&[vec![center]])
                    .unwrap()
            );
            assert!(batch.update_static_axial_sphere_centers(&[]).is_err());
            assert!(
                batch
                    .update_static_axial_sphere_centers(&[vec![center, Vector3::zeros()]])
                    .is_err()
            );
            assert!(
                batch
                    .update_static_axial_sphere_centers(&[vec![Vector3::new(f64::MAX, 0.0, 0.0)]])
                    .is_err()
            );
            assert_eq!(batch.readback_sleeping_coordinates().unwrap(), [vec![true]]);
            batch
                .update_contact_wake_requests(&[vec![true, false]])
                .unwrap();
            assert!(
                batch
                    .update_static_axial_sphere_centers(&[vec![Vector3::new(2.0, 0.0, -0.5)]])
                    .unwrap()
            );
            assert_eq!(
                batch.readback_sleeping_coordinates().unwrap(),
                [vec![false]]
            );
            batch.submit_steps(1).unwrap();
            assert_eq!(
                batch.readback_contact_wake_requests().unwrap()[0],
                [true, true]
            );
            assert_eq!(
                batch.readback_sleeping_coordinates().unwrap(),
                [vec![false]]
            );
            assert!(batch.readback().unwrap()[0].velocities[0] < 0.0);
            assert_eq!(batch.readback_geometric_contacts().unwrap()[0], [false; 2]);
            batch.submit_steps(1).unwrap();
            assert_eq!(
                batch.readback_contact_wake_requests().unwrap()[0],
                [false; 2]
            );
            eprintln!("resident static axial/sphere center update and wake passed on {backend:?}");
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn resident_sphere_plane_contact_changes_coupled_joint_velocities() {
        let articulation = Articulation::new(
            vec![link(0.0), link(1.0), link(1.5)],
            vec![
                JointSpec {
                    parent: 0,
                    child: 1,
                    kind: JointKind::Prismatic,
                    origin: Isometry3::identity(),
                    axis: Vector3::z(),
                    limits: None,
                },
                JointSpec {
                    parent: 1,
                    child: 2,
                    kind: JointKind::Prismatic,
                    origin: Isometry3::identity(),
                    axis: Vector3::new(0.6, 0.0, 0.8),
                    limits: None,
                },
            ],
            0,
        )
        .unwrap();
        let dt = 0.1;
        let input = GpuArticulatedDynamicsInput {
            articulation: &articulation,
            root_pose: Isometry3::translation(0.0, 0.0, 0.55),
            state: GpuGeneralizedState {
                positions: DVector::zeros(2),
                velocities: DVector::from_column_slice(&[-1.0, -0.5]),
            },
            gravity: Vector3::zeros(),
            joints: vec![GpuJointForceInput::default(); 2],
            joint_velocity_limit: None,
            coordinate_velocity_limits: None,
            link_loads: vec![GpuMassLinkLoad::default(); 3],
            ground_spheres: vec![GpuArticulatedGroundSphere {
                link: 2,
                local_center: Vector3::zeros(),
                radius: 0.5,
                plane_normal: Vector3::z(),
                plane_offset: 0.0,
                plane_xy_half_extent: None,
                restitution: 0.0,
                friction: 0.0,
            }],
            ground_capsules: Vec::new(),
            ground_boxes: Vec::new(),
            ground_axial_shapes: Vec::new(),
            sphere_pairs: Vec::new(),
            static_sphere_pairs: Vec::new(),
            external_sphere_bodies: None,
            external_box_bodies: None,
            external_capsule_bodies: None,
            external_axial_bodies: None,
            external_convex_bodies: None,
            external_constraint_bodies: None,
            external_indexed_bodies: None,
            static_capsule_sphere_pairs: Vec::new(),
            static_box_sphere_pairs: Vec::new(),
            static_sphere_capsule_pairs: Vec::new(),
            static_capsule_pairs: Vec::new(),
            static_sphere_box_pairs: Vec::new(),
            static_capsule_box_pairs: Vec::new(),
            static_box_pairs: Vec::new(),
            static_box_capsule_pairs: Vec::new(),
            static_axial_sphere_pairs: Vec::new(),
            static_axial_capsule_pairs: Vec::new(),
            static_axial_box_pairs: Vec::new(),
            static_axial_convex_pairs: Vec::new(),
            axial_convex_pairs: Vec::new(),
            axial_sphere_pairs: Vec::new(),
            axial_box_pairs: Vec::new(),
            axial_capsule_pairs: Vec::new(),
            axial_pairs: Vec::new(),
            convex_sphere_pairs: Vec::new(),
            static_convex_sphere_pairs: Vec::new(),
            static_convex_capsule_pairs: Vec::new(),
            convex_capsule_pairs: Vec::new(),
            convex_pairs: Vec::new(),
            static_convex_pairs: Vec::new(),
            scene_convex_sphere_pairs: Vec::new(),
            scene_mesh_sphere_pairs: Vec::new(),
            scene_polyline_sphere_pairs: Vec::new(),
            scene_mesh_capsule_pairs: Vec::new(),
            scene_polyline_capsule_pairs: Vec::new(),
            scene_mesh_box_pairs: Vec::new(),
            scene_polyline_box_pairs: Vec::new(),
            scene_mesh_axial_pairs: Vec::new(),
            scene_polyline_axial_pairs: Vec::new(),
            scene_mesh_convex_pairs: Vec::new(),
            scene_polyline_convex_pairs: Vec::new(),
            scene_convex_capsule_pairs: Vec::new(),
            capsule_sphere_pairs: Vec::new(),
            capsule_pairs: Vec::new(),
            sphere_box_pairs: Vec::new(),
            capsule_box_pairs: Vec::new(),
            link_spheres: Vec::new(),
            link_capsules: Vec::new(),
            link_boxes: Vec::new(),
            box_pairs: Vec::new(),
            contact_iterations: 8,
            dynamic_material_rules: Vec::new(),
            ground_manifold_start: None,
            joint_friction: Vec::new(),
            joint_couplings: Vec::new(),
            link_point_constraints: Vec::new(),
            link_fixed_constraints: Vec::new(),
            contact_warm_start: false,
        };
        let dynamics = articulation
            .generalized_dynamics(
                input.root_pose,
                input.state.positions.as_slice(),
                &input.state.velocities,
                false,
                Vector3::zeros(),
            )
            .unwrap();
        let pose = articulation
            .pose(input.root_pose, input.state.positions.as_slice())
            .unwrap();
        let (linear, _) = articulation
            .point_jacobians(&pose, 2, Vector3::new(0.0, 0.0, -0.5))
            .unwrap();
        let jacobian = linear.row(2).transpose().into_owned();
        let response = dynamics.mass.lu().solve(&jacobian).unwrap();
        let effective = jacobian.dot(&response);
        let normal_velocity = jacobian.dot(&input.state.velocities);
        let distance = 0.05;
        let impulse = (-distance / dt - normal_velocity) / effective;
        assert!(impulse > 0.0);
        let expected_velocity = &input.state.velocities + &response * impulse;
        let expected_position = &input.state.positions + &expected_velocity * dt;
        let second_impulse = (0.0 - jacobian.dot(&expected_velocity)) / effective;
        let second_velocity = &expected_velocity + &response * second_impulse;
        let second_position = &expected_position + &second_velocity * dt;
        assert!((expected_velocity[0] - input.state.velocities[0]).abs() > 0.1);
        assert!((expected_velocity[1] - input.state.velocities[1]).abs() > 0.1);
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let mut batch =
                GpuArticulatedDynamicsBatch::new(&context, core::slice::from_ref(&input), dt)
                    .unwrap();
            assert!(batch.readback_contact_activity().is_err());
            batch.enable_contact_activity().unwrap();
            assert_eq!(batch.readback_contact_activity().unwrap()[0], [false; 3]);
            batch.submit_steps(1).unwrap();
            assert_eq!(
                batch.readback_contact_activity().unwrap()[0],
                [false, false, true]
            );
            batch.enable_contact_activity().unwrap();
            assert!(batch.readback_contact_activity().unwrap()[0][2]);
            let actual = batch.readback().unwrap();
            assert!(
                (&actual[0].velocities - &expected_velocity).norm() < 2e-4,
                "actual {:?}, expected {:?}",
                actual[0].velocities,
                expected_velocity,
            );
            assert!((&actual[0].positions - &expected_position).norm() < 2e-4);
            let actual_pose = batch.readback_link_poses().unwrap();
            assert!((actual_pose[0][2].translation.vector.z - 0.5).abs() < 2e-4);
            batch.submit_steps(1).unwrap();
            let supported = batch.readback().unwrap();
            assert!((&supported[0].velocities - &second_velocity).norm() < 2e-4);
            assert!((&supported[0].positions - &second_position).norm() < 2e-4);
            let mut rebound_input = input.clone();
            rebound_input.root_pose.translation.vector.z = 0.5;
            rebound_input.ground_spheres[0].restitution = 0.5;
            let rebound =
                GpuArticulatedDynamicsBatch::new(&context, &[rebound_input.clone()], dt).unwrap();
            rebound.submit_steps(1).unwrap();
            let rebound_velocity = rebound.readback().unwrap()[0].velocities.clone();
            let restitution_target = -normal_velocity * 0.5;
            let expected_rebound = &input.state.velocities
                + &response * ((restitution_target - normal_velocity) / effective);
            assert!((&rebound_velocity - expected_rebound).norm() < 2e-4);
            rebound_input.contact_warm_start = true;
            let warm_rebound =
                GpuArticulatedDynamicsBatch::new(&context, &[rebound_input], dt).unwrap();
            warm_rebound.submit_steps(1).unwrap();
            let first = warm_rebound.readback().unwrap()[0].velocities.clone();
            warm_rebound.submit_steps(1).unwrap();
            let separated = warm_rebound.readback().unwrap()[0].velocities.clone();
            assert!((&separated - &first).norm() < 2e-4);
            batch.reset(core::slice::from_ref(&input.state)).unwrap();
            assert_eq!(batch.readback_contact_activity().unwrap()[0], [false; 3]);
            batch.submit_steps(1).unwrap();
            assert_eq!(
                batch.readback_contact_activity().unwrap()[0],
                [false, false, true]
            );
            batch
                .set_root_pose(0, input.root_pose * Isometry3::translation(0.0, 0.0, 10.0))
                .unwrap();
            assert_eq!(batch.readback_contact_activity().unwrap()[0], [false; 3]);
            batch.submit_steps(1).unwrap();
            assert_eq!(batch.readback_contact_activity().unwrap()[0], [false; 3]);
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn rotating_link_sphere_contact_uses_angular_jacobian() {
        let articulation = Articulation::new(
            vec![link(0.0), link(1.2)],
            vec![JointSpec {
                parent: 0,
                child: 1,
                kind: JointKind::Revolute,
                origin: Isometry3::identity(),
                axis: Vector3::y(),
                limits: None,
            }],
            0,
        )
        .unwrap();
        let dt = 0.1;
        let input = GpuArticulatedDynamicsInput {
            articulation: &articulation,
            root_pose: Isometry3::translation(0.0, 0.0, 0.25),
            state: GpuGeneralizedState {
                positions: DVector::zeros(1),
                velocities: DVector::from_element(1, 1.2),
            },
            gravity: Vector3::zeros(),
            joints: vec![GpuJointForceInput::default()],
            joint_velocity_limit: None,
            coordinate_velocity_limits: None,
            link_loads: vec![GpuMassLinkLoad::default(); 2],
            ground_spheres: vec![GpuArticulatedGroundSphere {
                link: 1,
                local_center: Vector3::new(0.5, 0.0, 0.0),
                radius: 0.2,
                plane_normal: Vector3::new(0.0, 0.0, 2.0),
                plane_offset: 0.0,
                plane_xy_half_extent: None,
                restitution: 0.0,
                friction: 0.0,
            }],
            ground_capsules: Vec::new(),
            ground_boxes: Vec::new(),
            ground_axial_shapes: Vec::new(),
            sphere_pairs: Vec::new(),
            static_sphere_pairs: Vec::new(),
            external_sphere_bodies: None,
            external_box_bodies: None,
            external_capsule_bodies: None,
            external_axial_bodies: None,
            external_convex_bodies: None,
            external_constraint_bodies: None,
            external_indexed_bodies: None,
            static_capsule_sphere_pairs: Vec::new(),
            static_box_sphere_pairs: Vec::new(),
            static_sphere_capsule_pairs: Vec::new(),
            static_capsule_pairs: Vec::new(),
            static_sphere_box_pairs: Vec::new(),
            static_capsule_box_pairs: Vec::new(),
            static_box_pairs: Vec::new(),
            static_box_capsule_pairs: Vec::new(),
            static_axial_sphere_pairs: Vec::new(),
            static_axial_capsule_pairs: Vec::new(),
            static_axial_box_pairs: Vec::new(),
            static_axial_convex_pairs: Vec::new(),
            axial_convex_pairs: Vec::new(),
            axial_sphere_pairs: Vec::new(),
            axial_box_pairs: Vec::new(),
            axial_capsule_pairs: Vec::new(),
            axial_pairs: Vec::new(),
            convex_sphere_pairs: Vec::new(),
            static_convex_sphere_pairs: Vec::new(),
            static_convex_capsule_pairs: Vec::new(),
            convex_capsule_pairs: Vec::new(),
            convex_pairs: Vec::new(),
            static_convex_pairs: Vec::new(),
            scene_convex_sphere_pairs: Vec::new(),
            scene_mesh_sphere_pairs: Vec::new(),
            scene_polyline_sphere_pairs: Vec::new(),
            scene_mesh_capsule_pairs: Vec::new(),
            scene_polyline_capsule_pairs: Vec::new(),
            scene_mesh_box_pairs: Vec::new(),
            scene_polyline_box_pairs: Vec::new(),
            scene_mesh_axial_pairs: Vec::new(),
            scene_polyline_axial_pairs: Vec::new(),
            scene_mesh_convex_pairs: Vec::new(),
            scene_polyline_convex_pairs: Vec::new(),
            scene_convex_capsule_pairs: Vec::new(),
            capsule_sphere_pairs: Vec::new(),
            capsule_pairs: Vec::new(),
            sphere_box_pairs: Vec::new(),
            capsule_box_pairs: Vec::new(),
            link_spheres: Vec::new(),
            link_capsules: Vec::new(),
            link_boxes: Vec::new(),
            box_pairs: Vec::new(),
            contact_iterations: 8,
            dynamic_material_rules: Vec::new(),
            ground_manifold_start: None,
            joint_friction: Vec::new(),
            joint_couplings: Vec::new(),
            link_point_constraints: Vec::new(),
            link_fixed_constraints: Vec::new(),
            contact_warm_start: false,
        };
        let pose = articulation
            .pose(input.root_pose, input.state.positions.as_slice())
            .unwrap();
        let (linear, _) = articulation
            .point_jacobians(&pose, 1, Vector3::new(0.5, 0.0, -0.2))
            .unwrap();
        let jacobian = linear[(2, 0)];
        assert!((jacobian + 0.5).abs() < 1e-12);
        let dynamics = articulation
            .generalized_dynamics(
                input.root_pose,
                input.state.positions.as_slice(),
                &input.state.velocities,
                false,
                Vector3::zeros(),
            )
            .unwrap();
        let predicted =
            input.state.velocities[0] - dt * dynamics.velocity_bias[0] / dynamics.mass[(0, 0)];
        let distance = 0.05;
        let normal_velocity = jacobian * predicted;
        assert!(distance + dt * normal_velocity < 0.0);
        let effective = jacobian * jacobian / dynamics.mass[(0, 0)];
        let impulse = (-distance / dt - normal_velocity) / effective;
        let expected_velocity = predicted + jacobian / dynamics.mass[(0, 0)] * impulse;
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let batch =
                GpuArticulatedDynamicsBatch::new(&context, core::slice::from_ref(&input), dt)
                    .unwrap();
            batch.submit_steps(1).unwrap();
            let actual = batch.readback().unwrap();
            assert!((actual[0].velocities[0] - expected_velocity).abs() < 2e-4);
            assert!((actual[0].positions[0] - expected_velocity * dt).abs() < 2e-4);
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn packed_ground_contacts_use_their_own_environment_offsets() {
        let articulation = Articulation::new(
            vec![link(0.0), link(1.0)],
            vec![JointSpec {
                parent: 0,
                child: 1,
                kind: JointKind::Prismatic,
                origin: Isometry3::identity(),
                axis: Vector3::z(),
                limits: None,
            }],
            0,
        )
        .unwrap();
        let make_input = |height: f64, velocity: f64| GpuArticulatedDynamicsInput {
            articulation: &articulation,
            root_pose: Isometry3::translation(0.0, 0.0, height),
            state: GpuGeneralizedState {
                positions: DVector::zeros(1),
                velocities: DVector::from_element(1, velocity),
            },
            gravity: Vector3::zeros(),
            joints: vec![GpuJointForceInput::default()],
            joint_velocity_limit: None,
            coordinate_velocity_limits: None,
            link_loads: vec![GpuMassLinkLoad::default(); 2],
            ground_spheres: vec![GpuArticulatedGroundSphere {
                link: 1,
                local_center: Vector3::zeros(),
                radius: 0.5,
                plane_normal: Vector3::z(),
                plane_offset: 0.0,
                plane_xy_half_extent: None,
                restitution: 0.0,
                friction: 0.0,
            }],
            ground_capsules: Vec::new(),
            ground_boxes: Vec::new(),
            ground_axial_shapes: Vec::new(),
            sphere_pairs: Vec::new(),
            static_sphere_pairs: Vec::new(),
            external_sphere_bodies: None,
            external_box_bodies: None,
            external_capsule_bodies: None,
            external_axial_bodies: None,
            external_convex_bodies: None,
            external_constraint_bodies: None,
            external_indexed_bodies: None,
            static_capsule_sphere_pairs: Vec::new(),
            static_box_sphere_pairs: Vec::new(),
            static_sphere_capsule_pairs: Vec::new(),
            static_capsule_pairs: Vec::new(),
            static_sphere_box_pairs: Vec::new(),
            static_capsule_box_pairs: Vec::new(),
            static_box_pairs: Vec::new(),
            static_box_capsule_pairs: Vec::new(),
            static_axial_sphere_pairs: Vec::new(),
            static_axial_capsule_pairs: Vec::new(),
            static_axial_box_pairs: Vec::new(),
            static_axial_convex_pairs: Vec::new(),
            axial_convex_pairs: Vec::new(),
            axial_sphere_pairs: Vec::new(),
            axial_box_pairs: Vec::new(),
            axial_capsule_pairs: Vec::new(),
            axial_pairs: Vec::new(),
            convex_sphere_pairs: Vec::new(),
            static_convex_sphere_pairs: Vec::new(),
            static_convex_capsule_pairs: Vec::new(),
            convex_capsule_pairs: Vec::new(),
            convex_pairs: Vec::new(),
            static_convex_pairs: Vec::new(),
            scene_convex_sphere_pairs: Vec::new(),
            scene_mesh_sphere_pairs: Vec::new(),
            scene_polyline_sphere_pairs: Vec::new(),
            scene_mesh_capsule_pairs: Vec::new(),
            scene_polyline_capsule_pairs: Vec::new(),
            scene_mesh_box_pairs: Vec::new(),
            scene_polyline_box_pairs: Vec::new(),
            scene_mesh_axial_pairs: Vec::new(),
            scene_polyline_axial_pairs: Vec::new(),
            scene_mesh_convex_pairs: Vec::new(),
            scene_polyline_convex_pairs: Vec::new(),
            scene_convex_capsule_pairs: Vec::new(),
            capsule_sphere_pairs: Vec::new(),
            capsule_pairs: Vec::new(),
            sphere_box_pairs: Vec::new(),
            capsule_box_pairs: Vec::new(),
            link_spheres: Vec::new(),
            link_capsules: Vec::new(),
            link_boxes: Vec::new(),
            box_pairs: Vec::new(),
            contact_iterations: 8,
            dynamic_material_rules: Vec::new(),
            ground_manifold_start: None,
            joint_friction: Vec::new(),
            joint_couplings: Vec::new(),
            link_point_constraints: Vec::new(),
            link_fixed_constraints: Vec::new(),
            contact_warm_start: false,
        };
        let inputs = [make_input(1.0, 0.0), make_input(0.55, -1.0)];
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let batch = GpuArticulatedDynamicsBatch::new(&context, &inputs, 0.1).unwrap();
            batch.submit_steps(1).unwrap();
            let actual = batch.readback().unwrap();
            assert!(actual[0].velocities[0].abs() < 1e-6);
            assert!(actual[0].positions[0].abs() < 1e-6);
            assert!((actual[1].velocities[0] + 0.5).abs() < 1e-5);
            assert!((actual[1].positions[0] + 0.05).abs() < 1e-5);
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn sphere_plane_friction_projects_onto_coulomb_disk() {
        let centered_link = |mass| LinkSpec {
            mass,
            center_of_mass: Vector3::zeros(),
            inertia: Matrix3::identity() * mass,
        };
        let articulation = Articulation::new(
            vec![
                centered_link(0.0),
                centered_link(0.0),
                centered_link(0.0),
                centered_link(1.0),
            ],
            [Vector3::x(), Vector3::y(), Vector3::z()]
                .into_iter()
                .enumerate()
                .map(|(index, axis)| JointSpec {
                    parent: index,
                    child: index + 1,
                    kind: JointKind::Prismatic,
                    origin: Isometry3::identity(),
                    axis,
                    limits: None,
                })
                .collect(),
            0,
        )
        .unwrap();
        let input = |friction| GpuArticulatedDynamicsInput {
            articulation: &articulation,
            root_pose: Isometry3::translation(0.0, 0.0, 0.5),
            state: GpuGeneralizedState {
                positions: DVector::zeros(3),
                velocities: DVector::from_column_slice(&[1.0, 1.0, 0.0]),
            },
            gravity: Vector3::new(0.0, 0.0, -10.0),
            joints: vec![GpuJointForceInput::default(); 3],
            joint_velocity_limit: None,
            coordinate_velocity_limits: None,
            link_loads: vec![GpuMassLinkLoad::default(); 4],
            ground_spheres: vec![GpuArticulatedGroundSphere {
                link: 3,
                local_center: Vector3::zeros(),
                radius: 0.5,
                plane_normal: Vector3::z(),
                plane_offset: 0.0,
                plane_xy_half_extent: None,
                restitution: 0.0,
                friction,
            }],
            ground_capsules: Vec::new(),
            ground_boxes: Vec::new(),
            ground_axial_shapes: Vec::new(),
            sphere_pairs: Vec::new(),
            static_sphere_pairs: Vec::new(),
            external_sphere_bodies: None,
            external_box_bodies: None,
            external_capsule_bodies: None,
            external_axial_bodies: None,
            external_convex_bodies: None,
            external_constraint_bodies: None,
            external_indexed_bodies: None,
            static_capsule_sphere_pairs: Vec::new(),
            static_box_sphere_pairs: Vec::new(),
            static_sphere_capsule_pairs: Vec::new(),
            static_capsule_pairs: Vec::new(),
            static_sphere_box_pairs: Vec::new(),
            static_capsule_box_pairs: Vec::new(),
            static_box_pairs: Vec::new(),
            static_box_capsule_pairs: Vec::new(),
            static_axial_sphere_pairs: Vec::new(),
            static_axial_capsule_pairs: Vec::new(),
            static_axial_box_pairs: Vec::new(),
            static_axial_convex_pairs: Vec::new(),
            axial_convex_pairs: Vec::new(),
            axial_sphere_pairs: Vec::new(),
            axial_box_pairs: Vec::new(),
            axial_capsule_pairs: Vec::new(),
            axial_pairs: Vec::new(),
            convex_sphere_pairs: Vec::new(),
            static_convex_sphere_pairs: Vec::new(),
            static_convex_capsule_pairs: Vec::new(),
            convex_capsule_pairs: Vec::new(),
            convex_pairs: Vec::new(),
            static_convex_pairs: Vec::new(),
            scene_convex_sphere_pairs: Vec::new(),
            scene_mesh_sphere_pairs: Vec::new(),
            scene_polyline_sphere_pairs: Vec::new(),
            scene_mesh_capsule_pairs: Vec::new(),
            scene_polyline_capsule_pairs: Vec::new(),
            scene_mesh_box_pairs: Vec::new(),
            scene_polyline_box_pairs: Vec::new(),
            scene_mesh_axial_pairs: Vec::new(),
            scene_polyline_axial_pairs: Vec::new(),
            scene_mesh_convex_pairs: Vec::new(),
            scene_polyline_convex_pairs: Vec::new(),
            scene_convex_capsule_pairs: Vec::new(),
            capsule_sphere_pairs: Vec::new(),
            capsule_pairs: Vec::new(),
            sphere_box_pairs: Vec::new(),
            capsule_box_pairs: Vec::new(),
            link_spheres: Vec::new(),
            link_capsules: Vec::new(),
            link_boxes: Vec::new(),
            box_pairs: Vec::new(),
            contact_iterations: 8,
            dynamic_material_rules: Vec::new(),
            ground_manifold_start: None,
            joint_friction: Vec::new(),
            joint_couplings: Vec::new(),
            link_point_constraints: Vec::new(),
            link_fixed_constraints: Vec::new(),
            contact_warm_start: false,
        };
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let batch = GpuArticulatedDynamicsBatch::new(&context, &[input(0.5)], 0.1).unwrap();
            batch.submit_steps(1).unwrap();
            let velocity = batch.readback().unwrap()[0].velocities.clone();
            let expected = 1.0 - 0.5 / 2.0_f64.sqrt();
            assert!((velocity[0] - expected).abs() < 2e-4, "{velocity:?}");
            assert!((velocity[1] - expected).abs() < 2e-4, "{velocity:?}");
            assert!(velocity[2].abs() < 2e-4, "{velocity:?}");
            let no_friction =
                GpuArticulatedDynamicsBatch::new(&context, &[input(0.0)], 0.1).unwrap();
            no_friction.submit_steps(1).unwrap();
            let free = no_friction.readback().unwrap()[0].velocities.clone();
            assert!((free[0] - 1.0).abs() < 2e-4, "{free:?}");
            assert!((free[1] - 1.0).abs() < 2e-4, "{free:?}");
            assert!(free[2].abs() < 2e-4, "{free:?}");
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn articulated_sphere_pair_transfers_impulse_and_friction_between_branches() {
        let centered_link = |mass| LinkSpec {
            mass,
            center_of_mass: Vector3::zeros(),
            inertia: Matrix3::identity() * mass,
        };
        let joint = |parent, child, axis| JointSpec {
            parent,
            child,
            kind: JointKind::Prismatic,
            origin: Isometry3::identity(),
            axis,
            limits: None,
        };
        let articulation = Articulation::new(
            vec![
                centered_link(0.0),
                centered_link(0.0),
                centered_link(1.0),
                centered_link(0.0),
                centered_link(1.0),
            ],
            vec![
                joint(0, 1, Vector3::x()),
                joint(1, 2, Vector3::y()),
                joint(0, 3, Vector3::x()),
                joint(3, 4, Vector3::y()),
            ],
            0,
        )
        .unwrap();
        let input = GpuArticulatedDynamicsInput {
            articulation: &articulation,
            root_pose: Isometry3::identity(),
            state: GpuGeneralizedState {
                positions: DVector::zeros(4),
                velocities: DVector::from_column_slice(&[1.0, 1.0, -1.0, -1.0]),
            },
            gravity: Vector3::zeros(),
            joints: vec![GpuJointForceInput::default(); 4],
            joint_velocity_limit: None,
            coordinate_velocity_limits: None,
            link_loads: vec![GpuMassLinkLoad::default(); 5],
            ground_spheres: Vec::new(),
            ground_capsules: Vec::new(),
            ground_boxes: Vec::new(),
            ground_axial_shapes: Vec::new(),
            sphere_pairs: vec![GpuArticulatedSpherePair {
                first_link: 2,
                first_local_center: Vector3::new(-0.55, 0.0, 0.0),
                first_radius: 0.5,
                second_link: 4,
                second_local_center: Vector3::new(0.55, 0.0, 0.0),
                second_radius: 0.5,
                restitution: 0.0,
                friction: 0.5,
            }],
            static_sphere_pairs: Vec::new(),
            external_sphere_bodies: None,
            external_box_bodies: None,
            external_capsule_bodies: None,
            external_axial_bodies: None,
            external_convex_bodies: None,
            external_constraint_bodies: None,
            external_indexed_bodies: None,
            static_capsule_sphere_pairs: Vec::new(),
            static_box_sphere_pairs: Vec::new(),
            static_sphere_capsule_pairs: Vec::new(),
            static_capsule_pairs: Vec::new(),
            static_sphere_box_pairs: Vec::new(),
            static_capsule_box_pairs: Vec::new(),
            static_box_pairs: Vec::new(),
            static_box_capsule_pairs: Vec::new(),
            static_axial_sphere_pairs: Vec::new(),
            static_axial_capsule_pairs: Vec::new(),
            static_axial_box_pairs: Vec::new(),
            static_axial_convex_pairs: Vec::new(),
            axial_convex_pairs: Vec::new(),
            axial_sphere_pairs: Vec::new(),
            axial_box_pairs: Vec::new(),
            axial_capsule_pairs: Vec::new(),
            axial_pairs: Vec::new(),
            convex_sphere_pairs: Vec::new(),
            static_convex_sphere_pairs: Vec::new(),
            static_convex_capsule_pairs: Vec::new(),
            convex_capsule_pairs: Vec::new(),
            convex_pairs: Vec::new(),
            static_convex_pairs: Vec::new(),
            scene_convex_sphere_pairs: Vec::new(),
            scene_mesh_sphere_pairs: Vec::new(),
            scene_polyline_sphere_pairs: Vec::new(),
            scene_mesh_capsule_pairs: Vec::new(),
            scene_polyline_capsule_pairs: Vec::new(),
            scene_mesh_box_pairs: Vec::new(),
            scene_polyline_box_pairs: Vec::new(),
            scene_mesh_axial_pairs: Vec::new(),
            scene_polyline_axial_pairs: Vec::new(),
            scene_mesh_convex_pairs: Vec::new(),
            scene_polyline_convex_pairs: Vec::new(),
            scene_convex_capsule_pairs: Vec::new(),
            capsule_sphere_pairs: Vec::new(),
            capsule_pairs: Vec::new(),
            sphere_box_pairs: Vec::new(),
            capsule_box_pairs: Vec::new(),
            link_spheres: Vec::new(),
            link_capsules: Vec::new(),
            link_boxes: Vec::new(),
            box_pairs: Vec::new(),
            contact_iterations: 8,
            dynamic_material_rules: Vec::new(),
            ground_manifold_start: None,
            joint_friction: Vec::new(),
            joint_couplings: Vec::new(),
            link_point_constraints: Vec::new(),
            link_fixed_constraints: Vec::new(),
            contact_warm_start: false,
        };
        let expected = DVector::from_column_slice(&[0.5, 0.75, -0.5, -0.75]);
        let automatic_spheres = vec![
            GpuArticulatedLinkSphere {
                link: 2,
                local_center: Vector3::new(-0.55, 0.0, 0.0),
                radius: 0.5,
                restitution: 0.0,
                friction: 0.5,
            },
            GpuArticulatedLinkSphere {
                link: 4,
                local_center: Vector3::new(0.55, 0.0, 0.0),
                radius: 0.5,
                restitution: 0.0,
                friction: 0.5,
            },
        ];
        assert_eq!(
            self_contact_sphere_pairs(&articulation, &automatic_spheres)
                .unwrap()
                .len(),
            1,
        );
        let mut material_spheres = automatic_spheres.clone();
        material_spheres[0].restitution = 0.2;
        material_spheres[1].restitution = 0.7;
        material_spheres[0].friction = 0.25;
        material_spheres[1].friction = 1.0;
        let combined = self_contact_sphere_pairs(&articulation, &material_spheres).unwrap();
        assert!((combined[0].restitution - 0.7).abs() < 1e-12);
        assert!((combined[0].friction - 0.5).abs() < 1e-12);
        let mut excluded = articulation.clone();
        excluded.exclude_collision_pair(2, 4).unwrap();
        assert!(
            self_contact_sphere_pairs(&excluded, &automatic_spheres)
                .unwrap()
                .is_empty()
        );
        let mut invalid = automatic_spheres.clone();
        invalid[0].radius = -0.5;
        assert!(self_contact_sphere_pairs(&articulation, &invalid).is_err());
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let batch =
                GpuArticulatedDynamicsBatch::new(&context, core::slice::from_ref(&input), 0.1)
                    .unwrap();
            batch.submit_steps(1).unwrap();
            let actual = batch.readback().unwrap();
            assert!(
                (&actual[0].velocities - &expected).norm() < 2e-4,
                "actual {:?}, expected {expected:?}",
                actual[0].velocities,
            );
            assert!((&actual[0].positions - &expected * 0.1).norm() < 2e-4);
            batch.submit_steps(1).unwrap();
            let second = batch.readback().unwrap();
            assert!(second[0].velocities.iter().all(|value| value.is_finite()));
            let second_poses = batch.readback_link_poses().unwrap();
            let first_center =
                second_poses[0][2].translation.vector + input.sphere_pairs[0].first_local_center;
            let second_center =
                second_poses[0][4].translation.vector + input.sphere_pairs[0].second_local_center;
            assert!((second_center - first_center).norm() > 1.0 - 2e-4);
            let mut separated_input = input.clone();
            separated_input.sphere_pairs[0].second_local_center.x = 2.0;
            let separated =
                GpuArticulatedDynamicsBatch::new(&context, &[input.clone(), separated_input], 0.1)
                    .unwrap();
            separated.submit_steps(1).unwrap();
            let free = separated.readback().unwrap();
            assert!((&free[0].velocities - &expected).norm() < 2e-4);
            assert!((&free[1].velocities - &input.state.velocities).norm() < 2e-4);
            let mut automatic_input = input.clone();
            automatic_input.sphere_pairs.clear();
            automatic_input.link_spheres = automatic_spheres.clone();
            let automatic =
                GpuArticulatedDynamicsBatch::new(&context, &[automatic_input], 0.1).unwrap();
            automatic.submit_steps(1).unwrap();
            let detected = automatic.readback().unwrap();
            assert!(
                (&detected[0].velocities - &expected).norm() < 2e-4,
                "dynamic sphere velocities: {:?}",
                detected[0].velocities
            );
            automatic
                .reset(core::slice::from_ref(&input.state))
                .unwrap();
            automatic.submit_steps(1).unwrap();
            let restarted = automatic.readback().unwrap();
            assert!((&restarted[0].velocities - &expected).norm() < 2e-4);
            let mut mixed_input = input.clone();
            mixed_input.link_spheres = automatic_spheres.clone();
            let mixed = GpuArticulatedDynamicsBatch::new(&context, &[mixed_input], 0.1).unwrap();
            mixed.submit_steps(1).unwrap();
            let deduplicated = mixed.readback().unwrap();
            assert!((&deduplicated[0].velocities - &expected).norm() < 2e-4);

            let mut accelerated_input = input.clone();
            accelerated_input.state.velocities.fill(0.0);
            accelerated_input.joints[0].base_force = 20.0;
            accelerated_input.joints[2].base_force = -20.0;
            let explicit =
                GpuArticulatedDynamicsBatch::new(&context, &[accelerated_input.clone()], 0.1)
                    .unwrap();
            explicit.submit_steps(1).unwrap();
            let explicit_result = explicit.readback().unwrap();
            let mut dynamic_input = accelerated_input;
            dynamic_input.sphere_pairs.clear();
            dynamic_input.link_spheres = automatic_spheres.clone();
            let dynamic =
                GpuArticulatedDynamicsBatch::new(&context, &[dynamic_input], 0.1).unwrap();
            dynamic.submit_steps(1).unwrap();
            let dynamic_result = dynamic.readback().unwrap();
            assert!(
                (&dynamic_result[0].velocities - &explicit_result[0].velocities).norm() < 2e-4,
                "accelerated dynamic {:?}, explicit {:?}",
                dynamic_result[0].velocities,
                explicit_result[0].velocities,
            );
            assert!(explicit_result[0].velocities[0] < 2.0);

            let mut pressed = input.clone();
            pressed.sphere_pairs[0].first_local_center.x = -0.5;
            pressed.sphere_pairs[0].second_local_center.x = 0.5;
            pressed.state.velocities.fill(0.0);
            pressed.joints[0].base_force = 20.0;
            pressed.joints[2].base_force = -20.0;
            pressed.contact_warm_start = true;
            let mut dynamic_pressed = pressed.clone();
            dynamic_pressed.sphere_pairs.clear();
            dynamic_pressed.link_spheres = automatic_spheres.clone();
            dynamic_pressed.link_spheres[0].local_center.x = -0.5;
            dynamic_pressed.link_spheres[1].local_center.x = 0.5;
            let explicit =
                GpuArticulatedDynamicsBatch::new(&context, &[pressed.clone()], 0.1).unwrap();
            let dynamic =
                GpuArticulatedDynamicsBatch::new(&context, &[dynamic_pressed], 0.1).unwrap();
            for step in 0..3 {
                explicit.submit_steps(1).unwrap();
                dynamic.submit_steps(1).unwrap();
                let reference = explicit.readback().unwrap();
                let actual = dynamic.readback().unwrap();
                assert!(
                    (&actual[0].velocities - &reference[0].velocities).norm() < 3e-4,
                    "step {step}: dynamic {:?}, explicit {:?}",
                    actual[0].velocities,
                    reference[0].velocities,
                );
                let rows = dynamic
                    .ground_contact
                    .as_ref()
                    .unwrap()
                    .readback_dynamic_row_state(context.queue());
                assert_eq!(rows.len(), 1);
                assert!(
                    rows[0].0 > 0.0 && rows[0].1[0] > 0.0,
                    "step {step}: {rows:?}"
                );
            }
            let mut outward = pressed.joints;
            outward[0].base_force = -20.0;
            outward[2].base_force = 20.0;
            dynamic.update_joints(&[outward]).unwrap();
            dynamic.submit_steps(3).unwrap();
            let rows = dynamic
                .ground_contact
                .as_ref()
                .unwrap()
                .readback_dynamic_row_state(context.queue());
            assert_eq!(rows[0].0, 0.0);
            assert_eq!(rows[0].1, [0.0; 4]);

            let mut many_input = input.clone();
            many_input.sphere_pairs.clear();
            many_input.link_spheres = (0..35)
                .flat_map(|index| {
                    let offset = f64::from(index) * 10.0;
                    [
                        GpuArticulatedLinkSphere {
                            local_center: Vector3::new(-0.55 + offset, 0.0, 0.0),
                            ..automatic_spheres[0]
                        },
                        GpuArticulatedLinkSphere {
                            local_center: Vector3::new(0.55 + offset, 0.0, 0.0),
                            ..automatic_spheres[1]
                        },
                    ]
                })
                .collect();
            let many = GpuArticulatedDynamicsBatch::new(&context, &[many_input], 0.1).unwrap();
            many.submit_steps(1).unwrap();
            let many_result = many.readback().unwrap();
            assert_eq!(
                many.ground_contact
                    .as_ref()
                    .unwrap()
                    .readback_dynamic_row_state(context.queue())
                    .len(),
                35 * 35,
            );
            assert!(
                (&many_result[0].velocities - &expected).norm() < 3e-4,
                "many dynamic spheres: {:?}",
                many_result[0].velocities,
            );
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn coupled_plane_contacts_revisit_earlier_normal_constraint() {
        let centered_link = |mass| LinkSpec {
            mass,
            center_of_mass: Vector3::zeros(),
            inertia: Matrix3::identity() * mass,
        };
        let articulation = Articulation::new(
            vec![centered_link(0.0), centered_link(0.0), centered_link(1.0)],
            vec![
                JointSpec {
                    parent: 0,
                    child: 1,
                    kind: JointKind::Prismatic,
                    origin: Isometry3::identity(),
                    axis: Vector3::x(),
                    limits: None,
                },
                JointSpec {
                    parent: 1,
                    child: 2,
                    kind: JointKind::Prismatic,
                    origin: Isometry3::identity(),
                    axis: Vector3::z(),
                    limits: None,
                },
            ],
            0,
        )
        .unwrap();
        let slant = Vector3::new(1.0, 0.0, -1.0).normalize();
        let ground = |normal: Vector3<f64>, offset| GpuArticulatedGroundSphere {
            link: 2,
            local_center: Vector3::zeros(),
            radius: 0.5,
            plane_normal: normal,
            plane_offset: offset,
            plane_xy_half_extent: None,
            restitution: 0.0,
            friction: 0.0,
        };
        let input = GpuArticulatedDynamicsInput {
            articulation: &articulation,
            root_pose: Isometry3::translation(0.0, 0.0, 0.5),
            state: GpuGeneralizedState {
                positions: DVector::zeros(2),
                velocities: DVector::from_column_slice(&[-1.0, -1.0]),
            },
            gravity: Vector3::zeros(),
            joints: vec![GpuJointForceInput::default(); 2],
            joint_velocity_limit: None,
            coordinate_velocity_limits: None,
            link_loads: vec![GpuMassLinkLoad::default(); 3],
            ground_spheres: vec![
                ground(Vector3::z(), 0.0),
                ground(slant, slant.dot(&Vector3::new(0.0, 0.0, 0.5)) - 0.5),
            ],
            ground_capsules: Vec::new(),
            ground_boxes: Vec::new(),
            ground_axial_shapes: Vec::new(),
            sphere_pairs: Vec::new(),
            static_sphere_pairs: Vec::new(),
            external_sphere_bodies: None,
            external_box_bodies: None,
            external_capsule_bodies: None,
            external_axial_bodies: None,
            external_convex_bodies: None,
            external_constraint_bodies: None,
            external_indexed_bodies: None,
            static_capsule_sphere_pairs: Vec::new(),
            static_box_sphere_pairs: Vec::new(),
            static_sphere_capsule_pairs: Vec::new(),
            static_capsule_pairs: Vec::new(),
            static_sphere_box_pairs: Vec::new(),
            static_capsule_box_pairs: Vec::new(),
            static_box_pairs: Vec::new(),
            static_box_capsule_pairs: Vec::new(),
            static_axial_sphere_pairs: Vec::new(),
            static_axial_capsule_pairs: Vec::new(),
            static_axial_box_pairs: Vec::new(),
            static_axial_convex_pairs: Vec::new(),
            axial_convex_pairs: Vec::new(),
            axial_sphere_pairs: Vec::new(),
            axial_box_pairs: Vec::new(),
            axial_capsule_pairs: Vec::new(),
            axial_pairs: Vec::new(),
            convex_sphere_pairs: Vec::new(),
            static_convex_sphere_pairs: Vec::new(),
            static_convex_capsule_pairs: Vec::new(),
            convex_capsule_pairs: Vec::new(),
            convex_pairs: Vec::new(),
            static_convex_pairs: Vec::new(),
            scene_convex_sphere_pairs: Vec::new(),
            scene_mesh_sphere_pairs: Vec::new(),
            scene_polyline_sphere_pairs: Vec::new(),
            scene_mesh_capsule_pairs: Vec::new(),
            scene_polyline_capsule_pairs: Vec::new(),
            scene_mesh_box_pairs: Vec::new(),
            scene_polyline_box_pairs: Vec::new(),
            scene_mesh_axial_pairs: Vec::new(),
            scene_polyline_axial_pairs: Vec::new(),
            scene_mesh_convex_pairs: Vec::new(),
            scene_polyline_convex_pairs: Vec::new(),
            scene_convex_capsule_pairs: Vec::new(),
            capsule_sphere_pairs: Vec::new(),
            capsule_pairs: Vec::new(),
            sphere_box_pairs: Vec::new(),
            capsule_box_pairs: Vec::new(),
            link_spheres: Vec::new(),
            link_capsules: Vec::new(),
            link_boxes: Vec::new(),
            box_pairs: Vec::new(),
            contact_iterations: 8,
            dynamic_material_rules: Vec::new(),
            ground_manifold_start: None,
            joint_friction: Vec::new(),
            joint_couplings: Vec::new(),
            link_point_constraints: Vec::new(),
            link_fixed_constraints: Vec::new(),
            contact_warm_start: false,
        };
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let batch =
                GpuArticulatedDynamicsBatch::new(&context, core::slice::from_ref(&input), 0.1)
                    .unwrap();
            batch.submit_steps(1).unwrap();
            let velocity = batch.readback().unwrap()[0].velocities.clone();
            let expected = -1.0 / 256.0;
            assert!((velocity[0] - expected).abs() < 2e-4, "{velocity:?}");
            assert!((velocity[1] - expected).abs() < 2e-4, "{velocity:?}");
            assert!(velocity[1] > -0.01);
            let mut one_sweep_input = input.clone();
            one_sweep_input.contact_iterations = 1;
            let one_sweep =
                GpuArticulatedDynamicsBatch::new(&context, &[input.clone(), one_sweep_input], 0.1)
                    .unwrap();
            one_sweep.submit_steps(1).unwrap();
            let mixed_iterations = one_sweep.readback().unwrap();
            assert!((mixed_iterations[0].velocities[1] - expected).abs() < 2e-4);
            let once = mixed_iterations[1].velocities.clone();
            assert!((once[0] + 0.5).abs() < 2e-4, "{once:?}");
            assert!((once[1] + 0.5).abs() < 2e-4, "{once:?}");
            let mut invalid_input = input.clone();
            invalid_input.contact_iterations = 0;
            assert!(GpuArticulatedDynamicsBatch::new(&context, &[invalid_input], 0.1).is_err());
            let mut invalid_extent = input.clone();
            invalid_extent.ground_spheres[1].plane_xy_half_extent = Some(0.5);
            assert!(GpuArticulatedDynamicsBatch::new(&context, &[invalid_extent], 0.1).is_err());
            let mut invalid_extent = input.clone();
            invalid_extent.ground_spheres[0].plane_xy_half_extent = Some(f64::MIN_POSITIVE);
            assert!(GpuArticulatedDynamicsBatch::new(&context, &[invalid_extent], 0.1).is_err());
            let mut rebound_input = input.clone();
            rebound_input.ground_spheres[0].restitution = 0.5;
            let rebound =
                GpuArticulatedDynamicsBatch::new(&context, &[rebound_input], 0.1).unwrap();
            rebound.submit_steps(1).unwrap();
            let rebound_velocity = rebound.readback().unwrap()[0].velocities.clone();
            let rebound_target = 0.5 - 1.5 / 256.0;
            assert!(
                (rebound_velocity[0] - rebound_target).abs() < 2e-4,
                "{rebound_velocity:?}"
            );
            assert!(
                (rebound_velocity[1] - rebound_target).abs() < 2e-4,
                "{rebound_velocity:?}"
            );
            let mut supported_input = input.clone();
            supported_input.state.velocities.fill(0.0);
            supported_input.gravity = Vector3::new(-10.0, 0.0, -10.0);
            let supported = GpuArticulatedDynamicsBatch::new(
                &context,
                core::slice::from_ref(&supported_input),
                0.01,
            )
            .unwrap();
            supported.submit_steps(100).unwrap();
            let settled = supported.readback().unwrap();
            assert!(settled[0].positions.norm() < 1e-3, "{:?}", settled[0]);
            assert!(settled[0].velocities.norm() < 1e-2, "{:?}", settled[0]);
            let mut one_sweep_support = supported_input.clone();
            one_sweep_support.contact_iterations = 1;
            let once_supported =
                GpuArticulatedDynamicsBatch::new(&context, &[one_sweep_support], 0.01).unwrap();
            once_supported.submit_steps(100).unwrap();
            let once_settled = once_supported.readback().unwrap();
            assert!(
                once_settled[0].velocities.norm() < 1e-2,
                "{:?}",
                once_settled[0]
            );
            once_supported
                .reset(&[supported_input.state.clone()])
                .unwrap();
            once_supported.submit_steps(2).unwrap();
            let cold = once_supported.readback().unwrap()[0].velocities.clone();
            let mut warm_input = supported_input.clone();
            warm_input.contact_iterations = 1;
            warm_input.contact_warm_start = true;
            let warm = GpuArticulatedDynamicsBatch::new(&context, &[warm_input], 0.01).unwrap();
            warm.submit_steps(2).unwrap();
            let warmed = warm.readback().unwrap()[0].velocities.clone();
            assert!(
                warmed.norm() < cold.norm(),
                "warm {warmed:?}, cold {cold:?}"
            );
            warm.reset(&[supported_input.state.clone()]).unwrap();
            warm.submit_steps(1).unwrap();
            let restarted = warm.readback().unwrap()[0].velocities.clone();
            assert!((restarted[0] + 0.05).abs() < 2e-4, "{restarted:?}");
            assert!((restarted[1] + 0.05).abs() < 2e-4, "{restarted:?}");
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn articulated_capsule_plane_uses_both_endpoint_contacts() {
        let centered_link = |mass| LinkSpec {
            mass,
            center_of_mass: Vector3::zeros(),
            inertia: Matrix3::identity() * mass,
        };
        let articulation = Articulation::new(
            vec![centered_link(0.0), centered_link(0.0), centered_link(1.0)],
            vec![
                JointSpec {
                    parent: 0,
                    child: 1,
                    kind: JointKind::Prismatic,
                    origin: Isometry3::identity(),
                    axis: Vector3::z(),
                    limits: None,
                },
                JointSpec {
                    parent: 1,
                    child: 2,
                    kind: JointKind::Revolute,
                    origin: Isometry3::identity(),
                    axis: Vector3::y(),
                    limits: None,
                },
            ],
            0,
        )
        .unwrap();
        let capsule = GpuArticulatedGroundCapsule {
            link: 2,
            local_a: Vector3::new(-0.5, 0.0, 0.0),
            local_b: Vector3::new(0.5, 0.0, 0.0),
            radius: 0.2,
            plane_normal: Vector3::z(),
            plane_offset: 0.0,
            restitution: 0.0,
            friction: 0.0,
        };
        assert_eq!(capsule.endpoint_contacts().len(), 2);
        let mut collapsed = capsule;
        collapsed.local_b = collapsed.local_a;
        assert_eq!(collapsed.endpoint_contacts().len(), 1);
        let input = GpuArticulatedDynamicsInput {
            articulation: &articulation,
            root_pose: Isometry3::translation(0.0, 0.0, 0.2),
            state: GpuGeneralizedState {
                positions: DVector::zeros(2),
                velocities: DVector::from_column_slice(&[-1.0, 0.2]),
            },
            gravity: Vector3::zeros(),
            joints: vec![GpuJointForceInput::default(); 2],
            joint_velocity_limit: None,
            coordinate_velocity_limits: None,
            link_loads: vec![GpuMassLinkLoad::default(); 3],
            ground_spheres: Vec::new(),
            ground_capsules: vec![capsule],
            ground_boxes: Vec::new(),
            ground_axial_shapes: Vec::new(),
            sphere_pairs: Vec::new(),
            static_sphere_pairs: Vec::new(),
            external_sphere_bodies: None,
            external_box_bodies: None,
            external_capsule_bodies: None,
            external_axial_bodies: None,
            external_convex_bodies: None,
            external_constraint_bodies: None,
            external_indexed_bodies: None,
            static_capsule_sphere_pairs: Vec::new(),
            static_box_sphere_pairs: Vec::new(),
            static_sphere_capsule_pairs: Vec::new(),
            static_capsule_pairs: Vec::new(),
            static_sphere_box_pairs: Vec::new(),
            static_capsule_box_pairs: Vec::new(),
            static_box_pairs: Vec::new(),
            static_box_capsule_pairs: Vec::new(),
            static_axial_sphere_pairs: Vec::new(),
            static_axial_capsule_pairs: Vec::new(),
            static_axial_box_pairs: Vec::new(),
            static_axial_convex_pairs: Vec::new(),
            axial_convex_pairs: Vec::new(),
            axial_sphere_pairs: Vec::new(),
            axial_box_pairs: Vec::new(),
            axial_capsule_pairs: Vec::new(),
            axial_pairs: Vec::new(),
            convex_sphere_pairs: Vec::new(),
            static_convex_sphere_pairs: Vec::new(),
            static_convex_capsule_pairs: Vec::new(),
            convex_capsule_pairs: Vec::new(),
            convex_pairs: Vec::new(),
            static_convex_pairs: Vec::new(),
            scene_convex_sphere_pairs: Vec::new(),
            scene_mesh_sphere_pairs: Vec::new(),
            scene_polyline_sphere_pairs: Vec::new(),
            scene_mesh_capsule_pairs: Vec::new(),
            scene_polyline_capsule_pairs: Vec::new(),
            scene_mesh_box_pairs: Vec::new(),
            scene_polyline_box_pairs: Vec::new(),
            scene_mesh_axial_pairs: Vec::new(),
            scene_polyline_axial_pairs: Vec::new(),
            scene_mesh_convex_pairs: Vec::new(),
            scene_polyline_convex_pairs: Vec::new(),
            scene_convex_capsule_pairs: Vec::new(),
            capsule_sphere_pairs: Vec::new(),
            capsule_pairs: Vec::new(),
            sphere_box_pairs: Vec::new(),
            capsule_box_pairs: Vec::new(),
            link_spheres: Vec::new(),
            link_capsules: Vec::new(),
            link_boxes: Vec::new(),
            box_pairs: Vec::new(),
            contact_iterations: 16,
            dynamic_material_rules: Vec::new(),
            ground_manifold_start: None,
            joint_friction: Vec::new(),
            joint_couplings: Vec::new(),
            link_point_constraints: Vec::new(),
            link_fixed_constraints: Vec::new(),
            contact_warm_start: false,
        };
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let batch =
                GpuArticulatedDynamicsBatch::new(&context, core::slice::from_ref(&input), 0.1)
                    .unwrap();
            batch.submit_steps(1).unwrap();
            let actual = batch.readback().unwrap();
            assert!(actual[0].velocities.norm() < 1e-3, "{:?}", actual[0]);
            assert!(actual[0].positions.norm() < 1e-4, "{:?}", actual[0]);
            let mut tilted = input.clone();
            tilted.root_pose.translation.vector.z = 0.4;
            tilted.state.velocities = DVector::from_column_slice(&[-1.0, 0.0]);
            tilted.ground_capsules[0].local_a.z = -0.2;
            tilted.ground_capsules[0].local_b.z = 0.2;
            let tilted_batch =
                GpuArticulatedDynamicsBatch::new(&context, &[tilted.clone()], 0.1).unwrap();
            tilted_batch.submit_steps(1).unwrap();
            let tilted_state = tilted_batch.readback().unwrap();
            let mut endpoint = tilted;
            endpoint.ground_spheres = vec![endpoint.ground_capsules[0].endpoint_contacts()[0]];
            endpoint.ground_capsules.clear();
            let endpoint_batch =
                GpuArticulatedDynamicsBatch::new(&context, &[endpoint], 0.1).unwrap();
            endpoint_batch.submit_steps(1).unwrap();
            let endpoint_state = endpoint_batch.readback().unwrap();
            assert!((&tilted_state[0].velocities - &endpoint_state[0].velocities).norm() < 2e-4);
            assert!((&tilted_state[0].positions - &endpoint_state[0].positions).norm() < 2e-4);
            let mut invalid = input.clone();
            invalid.ground_capsules[0].local_b.x = f64::NAN;
            assert!(GpuArticulatedDynamicsBatch::new(&context, &[invalid], 0.1).is_err());
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn articulated_capsule_pairs_use_closest_axis_points() {
        let centered_link = |mass| LinkSpec {
            mass,
            center_of_mass: Vector3::zeros(),
            inertia: Matrix3::identity() * mass,
        };
        let articulation = Articulation::new(
            vec![centered_link(0.0), centered_link(1.0), centered_link(1.0)],
            vec![
                JointSpec {
                    parent: 0,
                    child: 1,
                    kind: JointKind::Prismatic,
                    origin: Isometry3::identity(),
                    axis: Vector3::x(),
                    limits: None,
                },
                JointSpec {
                    parent: 0,
                    child: 2,
                    kind: JointKind::Prismatic,
                    origin: Isometry3::identity(),
                    axis: Vector3::x(),
                    limits: None,
                },
            ],
            0,
        )
        .unwrap();
        let input = GpuArticulatedDynamicsInput {
            articulation: &articulation,
            root_pose: Isometry3::identity(),
            state: GpuGeneralizedState {
                positions: DVector::zeros(2),
                velocities: DVector::from_column_slice(&[1.0, -1.0]),
            },
            gravity: Vector3::zeros(),
            joints: vec![GpuJointForceInput::default(); 2],
            joint_velocity_limit: None,
            coordinate_velocity_limits: None,
            link_loads: vec![GpuMassLinkLoad::default(); 3],
            ground_spheres: Vec::new(),
            ground_capsules: Vec::new(),
            ground_boxes: Vec::new(),
            ground_axial_shapes: Vec::new(),
            sphere_pairs: Vec::new(),
            static_sphere_pairs: Vec::new(),
            external_sphere_bodies: None,
            external_box_bodies: None,
            external_capsule_bodies: None,
            external_axial_bodies: None,
            external_convex_bodies: None,
            external_constraint_bodies: None,
            external_indexed_bodies: None,
            static_capsule_sphere_pairs: Vec::new(),
            static_box_sphere_pairs: Vec::new(),
            static_sphere_capsule_pairs: Vec::new(),
            static_capsule_pairs: Vec::new(),
            static_sphere_box_pairs: Vec::new(),
            static_capsule_box_pairs: Vec::new(),
            static_box_pairs: Vec::new(),
            static_box_capsule_pairs: Vec::new(),
            static_axial_sphere_pairs: Vec::new(),
            static_axial_capsule_pairs: Vec::new(),
            static_axial_box_pairs: Vec::new(),
            static_axial_convex_pairs: Vec::new(),
            axial_convex_pairs: Vec::new(),
            axial_sphere_pairs: Vec::new(),
            axial_box_pairs: Vec::new(),
            axial_capsule_pairs: Vec::new(),
            axial_pairs: Vec::new(),
            convex_sphere_pairs: Vec::new(),
            static_convex_sphere_pairs: Vec::new(),
            static_convex_capsule_pairs: Vec::new(),
            convex_capsule_pairs: Vec::new(),
            convex_pairs: Vec::new(),
            static_convex_pairs: Vec::new(),
            scene_convex_sphere_pairs: Vec::new(),
            scene_mesh_sphere_pairs: Vec::new(),
            scene_polyline_sphere_pairs: Vec::new(),
            scene_mesh_capsule_pairs: Vec::new(),
            scene_polyline_capsule_pairs: Vec::new(),
            scene_mesh_box_pairs: Vec::new(),
            scene_polyline_box_pairs: Vec::new(),
            scene_mesh_axial_pairs: Vec::new(),
            scene_polyline_axial_pairs: Vec::new(),
            scene_mesh_convex_pairs: Vec::new(),
            scene_polyline_convex_pairs: Vec::new(),
            scene_convex_capsule_pairs: Vec::new(),
            capsule_sphere_pairs: vec![GpuArticulatedCapsuleSpherePair {
                capsule_link: 1,
                capsule_local_a: Vector3::new(0.0, -0.5, 0.0),
                capsule_local_b: Vector3::new(0.0, 0.5, 0.0),
                capsule_radius: 0.5,
                sphere_link: 2,
                sphere_local_center: Vector3::new(1.05, 0.25, 0.0),
                sphere_radius: 0.5,
                restitution: 0.0,
                friction: 0.0,
            }],
            capsule_pairs: Vec::new(),
            sphere_box_pairs: Vec::new(),
            capsule_box_pairs: Vec::new(),
            link_spheres: Vec::new(),
            link_capsules: Vec::new(),
            link_boxes: Vec::new(),
            box_pairs: Vec::new(),
            contact_iterations: 8,
            dynamic_material_rules: Vec::new(),
            ground_manifold_start: None,
            joint_friction: Vec::new(),
            joint_couplings: Vec::new(),
            link_point_constraints: Vec::new(),
            link_fixed_constraints: Vec::new(),
            contact_warm_start: false,
        };
        let link_capsules = vec![
            GpuArticulatedLinkCapsule {
                link: 1,
                local_a: Vector3::new(0.0, -0.5, 0.0),
                local_b: Vector3::new(0.0, 0.5, 0.0),
                radius: 0.5,
                restitution: 0.2,
                friction: 0.25,
            },
            GpuArticulatedLinkCapsule {
                link: 2,
                local_a: Vector3::new(1.05, 0.25, -0.5),
                local_b: Vector3::new(1.05, 0.25, 0.5),
                radius: 0.5,
                restitution: 0.7,
                friction: 1.0,
            },
        ];
        let link_spheres = vec![GpuArticulatedLinkSphere {
            link: 2,
            local_center: Vector3::new(1.05, 0.25, 0.0),
            radius: 0.5,
            restitution: 0.7,
            friction: 1.0,
        }];
        let generated_capsules = self_contact_capsule_pairs(&articulation, &link_capsules).unwrap();
        assert_eq!(generated_capsules.len(), 1);
        assert!((generated_capsules[0].restitution - 0.7).abs() < 1e-12);
        assert!((generated_capsules[0].friction - 0.5).abs() < 1e-12);
        let generated_mixed =
            self_contact_capsule_sphere_pairs(&articulation, &link_capsules, &link_spheres)
                .unwrap();
        assert_eq!(generated_mixed.len(), 1);
        assert!((generated_mixed[0].restitution - 0.7).abs() < 1e-12);
        assert!((generated_mixed[0].friction - 0.5).abs() < 1e-12);
        let mut excluded = articulation.clone();
        excluded.exclude_collision_pair(1, 2).unwrap();
        assert!(
            self_contact_capsule_pairs(&excluded, &link_capsules)
                .unwrap()
                .is_empty()
        );
        assert!(
            self_contact_capsule_sphere_pairs(&excluded, &link_capsules, &link_spheres)
                .unwrap()
                .is_empty()
        );
        let mut invalid_capsules = link_capsules.clone();
        invalid_capsules[0].local_b.x = f64::NAN;
        assert!(self_contact_capsule_pairs(&articulation, &invalid_capsules).is_err());
        let expected = DVector::from_column_slice(&[0.25, -0.25]);
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let batch =
                GpuArticulatedDynamicsBatch::new(&context, core::slice::from_ref(&input), 0.1)
                    .unwrap();
            batch.submit_steps(1).unwrap();
            let first = batch.readback().unwrap();
            assert!(
                (&first[0].velocities - &expected).norm() < 2e-4,
                "{:?}",
                first[0]
            );
            assert!((&first[0].positions - &expected * 0.1).norm() < 2e-4);
            let mut automatic_mixed = input.clone();
            automatic_mixed.capsule_sphere_pairs.clear();
            let mut automatic_capsule = link_capsules[0];
            automatic_capsule.restitution = 0.0;
            automatic_capsule.friction = 0.0;
            let mut automatic_sphere = link_spheres[0];
            automatic_sphere.restitution = 0.0;
            automatic_sphere.friction = 0.0;
            automatic_mixed.link_capsules.push(automatic_capsule);
            automatic_mixed.link_spheres.push(automatic_sphere);
            let automatic_mixed_batch =
                GpuArticulatedDynamicsBatch::new(&context, &[automatic_mixed.clone()], 0.1)
                    .unwrap();
            automatic_mixed_batch.submit_steps(1).unwrap();
            let automatic_mixed_state = automatic_mixed_batch.readback().unwrap();
            assert!((&automatic_mixed_state[0].velocities - &expected).norm() < 2e-4);
            let rows = automatic_mixed_batch
                .ground_contact
                .as_ref()
                .unwrap()
                .readback_dynamic_row_state(context.queue());
            assert_eq!(rows.len(), 1);
            assert!(rows[0].0 > 0.0 && rows[0].1[0] > 0.0, "{rows:?}");
            let mut multiple_shapes = automatic_mixed.clone();
            multiple_shapes.link_spheres.push(GpuArticulatedLinkSphere {
                local_center: Vector3::new(10.0, 0.0, 0.0),
                ..automatic_sphere
            });
            let multiple_shapes_batch =
                GpuArticulatedDynamicsBatch::new(&context, &[multiple_shapes], 0.1).unwrap();
            multiple_shapes_batch.submit_steps(1).unwrap();
            let rows = multiple_shapes_batch
                .ground_contact
                .as_ref()
                .unwrap()
                .readback_dynamic_row_state(context.queue());
            assert_eq!(rows.len(), 2);
            assert!(rows[0].0 > 0.0 && rows[0].1[0] > 0.0, "{rows:?}");
            assert_eq!(rows[1], (0.0, [0.0; 4]));
            let mut separated_mixed = automatic_mixed;
            separated_mixed.link_spheres[0].local_center.x = 3.0;
            let separated_mixed_batch =
                GpuArticulatedDynamicsBatch::new(&context, &[separated_mixed], 0.1).unwrap();
            separated_mixed_batch.submit_steps(1).unwrap();
            let separated_mixed_state = separated_mixed_batch.readback().unwrap();
            assert!((&separated_mixed_state[0].velocities - &input.state.velocities).norm() < 2e-4);
            let rows = separated_mixed_batch
                .ground_contact
                .as_ref()
                .unwrap()
                .readback_dynamic_row_state(context.queue());
            assert_eq!(rows, vec![(0.0, [0.0; 4])]);
            let mut prioritized_mixed = input.clone();
            let mut opposing_capsule = link_capsules[0];
            opposing_capsule.restitution = 1.0;
            core::mem::swap(&mut opposing_capsule.local_a, &mut opposing_capsule.local_b);
            let mut opposing_sphere = link_spheres[0];
            opposing_sphere.restitution = 1.0;
            prioritized_mixed.link_capsules.push(opposing_capsule);
            prioritized_mixed.link_spheres.push(opposing_sphere);
            let prioritized_mixed_batch =
                GpuArticulatedDynamicsBatch::new(&context, &[prioritized_mixed], 0.1).unwrap();
            prioritized_mixed_batch.submit_steps(1).unwrap();
            let prioritized_mixed_state = prioritized_mixed_batch.readback().unwrap();
            assert!((&prioritized_mixed_state[0].velocities - &expected).norm() < 2e-4);
            let rows = prioritized_mixed_batch
                .ground_contact
                .as_ref()
                .unwrap()
                .readback_dynamic_row_state(context.queue());
            assert!(rows.is_empty());
            batch.submit_steps(1).unwrap();
            let second = batch.readback().unwrap();
            assert!(second[0].velocities.norm() < 2e-4, "{:?}", second[0]);
            let mut collapsed = input.clone();
            collapsed.capsule_sphere_pairs[0].capsule_local_a = Vector3::new(0.0, 0.25, 0.0);
            collapsed.capsule_sphere_pairs[0].capsule_local_b = Vector3::new(0.0, 0.25, 0.0);
            let collapsed_batch =
                GpuArticulatedDynamicsBatch::new(&context, &[collapsed], 0.1).unwrap();
            collapsed_batch.submit_steps(1).unwrap();
            let collapsed_state = collapsed_batch.readback().unwrap();
            assert!((&collapsed_state[0].velocities - &expected).norm() < 2e-4);
            let mut two_capsules = input.clone();
            two_capsules.capsule_sphere_pairs.clear();
            two_capsules.capsule_pairs.push(GpuArticulatedCapsulePair {
                first_link: 1,
                first_local_a: Vector3::new(0.0, -0.5, 0.0),
                first_local_b: Vector3::new(0.0, 0.5, 0.0),
                first_radius: 0.5,
                second_link: 2,
                second_local_a: Vector3::new(1.05, 0.25, -0.5),
                second_local_b: Vector3::new(1.05, 0.25, 0.5),
                second_radius: 0.5,
                restitution: 0.0,
                friction: 0.0,
            });
            let capsule_batch =
                GpuArticulatedDynamicsBatch::new(&context, &[two_capsules.clone()], 0.1).unwrap();
            capsule_batch.submit_steps(1).unwrap();
            let capsule_state = capsule_batch.readback().unwrap();
            assert!((&capsule_state[0].velocities - &expected).norm() < 2e-4);
            let mut automatic_capsules = two_capsules.clone();
            automatic_capsules.capsule_pairs.clear();
            let mut second_capsule = link_capsules[1];
            second_capsule.restitution = 0.0;
            second_capsule.friction = 0.0;
            automatic_capsules.link_capsules = vec![automatic_capsule, second_capsule];
            let automatic_capsule_batch =
                GpuArticulatedDynamicsBatch::new(&context, &[automatic_capsules.clone()], 0.1)
                    .unwrap();
            automatic_capsule_batch.submit_steps(1).unwrap();
            let automatic_capsule_state = automatic_capsule_batch.readback().unwrap();
            assert!((&automatic_capsule_state[0].velocities - &expected).norm() < 2e-4);
            let rows = automatic_capsule_batch
                .ground_contact
                .as_ref()
                .unwrap()
                .readback_dynamic_row_state(context.queue());
            assert_eq!(rows.len(), 2);
            assert!(rows[0].0 > 0.0 && rows[0].1[0] > 0.0, "{rows:?}");
            let mut outward = automatic_capsules.joints.clone();
            outward[0].base_force = -20.0;
            outward[1].base_force = 20.0;
            automatic_capsule_batch.update_joints(&[outward]).unwrap();
            automatic_capsule_batch.submit_steps(3).unwrap();
            let rows = automatic_capsule_batch
                .ground_contact
                .as_ref()
                .unwrap()
                .readback_dynamic_row_state(context.queue());
            assert_eq!(rows, vec![(0.0, [0.0; 4]); 2]);
            let mut three_capsules = automatic_capsules.clone();
            three_capsules
                .link_capsules
                .push(GpuArticulatedLinkCapsule {
                    local_a: Vector3::new(10.0, -0.5, 0.0),
                    local_b: Vector3::new(10.0, 0.5, 0.0),
                    ..second_capsule
                });
            let three_capsule_batch =
                GpuArticulatedDynamicsBatch::new(&context, &[three_capsules], 0.1).unwrap();
            three_capsule_batch.submit_steps(1).unwrap();
            let rows = three_capsule_batch
                .ground_contact
                .as_ref()
                .unwrap()
                .readback_dynamic_row_state(context.queue());
            assert_eq!(rows.len(), 4);
            assert!(rows[0].0 > 0.0 && rows[0].1[0] > 0.0, "{rows:?}");
            assert!(rows[2..].iter().all(|row| *row == (0.0, [0.0; 4])));
            let mut separated_capsules = automatic_capsules;
            separated_capsules.link_capsules[1].local_a.x = 3.0;
            separated_capsules.link_capsules[1].local_b.x = 3.0;
            let separated_capsule_batch =
                GpuArticulatedDynamicsBatch::new(&context, &[separated_capsules], 0.1).unwrap();
            separated_capsule_batch.submit_steps(1).unwrap();
            let rows = separated_capsule_batch
                .ground_contact
                .as_ref()
                .unwrap()
                .readback_dynamic_row_state(context.queue());
            assert_eq!(rows, vec![(0.0, [0.0; 4]); 2]);
            let mut prioritized_capsules = two_capsules.clone();
            let mut opposing_first = link_capsules[0];
            opposing_first.restitution = 1.0;
            let mut opposing_second = link_capsules[1];
            opposing_second.restitution = 1.0;
            prioritized_capsules.link_capsules = vec![opposing_second, opposing_first];
            let prioritized_capsule_batch =
                GpuArticulatedDynamicsBatch::new(&context, &[prioritized_capsules], 0.1).unwrap();
            prioritized_capsule_batch.submit_steps(1).unwrap();
            let prioritized_capsule_state = prioritized_capsule_batch.readback().unwrap();
            assert!((&prioritized_capsule_state[0].velocities - &expected).norm() < 2e-4);
            let rows = prioritized_capsule_batch
                .ground_contact
                .as_ref()
                .unwrap()
                .readback_dynamic_row_state(context.queue());
            assert!(rows.is_empty());
            capsule_batch.submit_steps(1).unwrap();
            let stopped = capsule_batch.readback().unwrap();
            assert!(stopped[0].velocities.norm() < 2e-4);
            two_capsules.capsule_pairs[0].second_local_a.z = 0.0;
            two_capsules.capsule_pairs[0].second_local_b.z = 0.0;
            let degenerate_batch =
                GpuArticulatedDynamicsBatch::new(&context, &[two_capsules.clone()], 0.1).unwrap();
            degenerate_batch.submit_steps(1).unwrap();
            let degenerate = degenerate_batch.readback().unwrap();
            assert!((&degenerate[0].velocities - &expected).norm() < 2e-4);
            two_capsules.capsule_pairs[0].first_local_a.y = 0.25;
            two_capsules.capsule_pairs[0].first_local_b.y = 0.25;
            let both_degenerate_batch =
                GpuArticulatedDynamicsBatch::new(&context, &[two_capsules.clone()], 0.1).unwrap();
            both_degenerate_batch.submit_steps(1).unwrap();
            let both_degenerate = both_degenerate_batch.readback().unwrap();
            assert!((&both_degenerate[0].velocities - &expected).norm() < 2e-4);
            two_capsules.capsule_pairs[0].second_local_a.x = 3.0;
            two_capsules.capsule_pairs[0].second_local_b.x = 3.0;
            let separated_batch =
                GpuArticulatedDynamicsBatch::new(&context, &[two_capsules.clone()], 0.1).unwrap();
            separated_batch.submit_steps(1).unwrap();
            let separated = separated_batch.readback().unwrap();
            assert!((&separated[0].velocities - &input.state.velocities).norm() < 2e-4);
            two_capsules.capsule_pairs[0].second_radius = -0.5;
            assert!(GpuArticulatedDynamicsBatch::new(&context, &[two_capsules], 0.1).is_err());
            let mut invalid = input.clone();
            invalid.capsule_sphere_pairs[0].capsule_radius = -0.5;
            assert!(GpuArticulatedDynamicsBatch::new(&context, &[invalid], 0.1).is_err());
            let mut isolated_invalid = input.clone();
            isolated_invalid.capsule_sphere_pairs.clear();
            isolated_invalid
                .link_capsules
                .push(GpuArticulatedLinkCapsule {
                    radius: -0.5,
                    ..automatic_capsule
                });
            assert!(GpuArticulatedDynamicsBatch::new(&context, &[isolated_invalid], 0.1).is_err());
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn parallel_capsule_side_contacts_balance_angular_impulses() {
        let link = |mass| LinkSpec {
            mass,
            center_of_mass: Vector3::zeros(),
            inertia: Matrix3::identity() * mass,
        };
        let joint = |parent, child, kind, axis| JointSpec {
            parent,
            child,
            kind,
            origin: Isometry3::identity(),
            axis,
            limits: None,
        };
        let articulation = Articulation::new(
            vec![link(0.0), link(0.0), link(1.0), link(0.0), link(1.0)],
            vec![
                joint(0, 1, JointKind::Prismatic, Vector3::x()),
                joint(1, 2, JointKind::Revolute, Vector3::z()),
                joint(0, 3, JointKind::Prismatic, Vector3::x()),
                joint(3, 4, JointKind::Revolute, Vector3::z()),
            ],
            0,
        )
        .unwrap();
        let input = GpuArticulatedDynamicsInput {
            articulation: &articulation,
            root_pose: Isometry3::identity(),
            state: GpuGeneralizedState {
                positions: DVector::zeros(4),
                velocities: DVector::from_column_slice(&[1.0, 0.0, -1.0, 0.0]),
            },
            gravity: Vector3::zeros(),
            joints: vec![GpuJointForceInput::default(); 4],
            joint_velocity_limit: None,
            coordinate_velocity_limits: None,
            link_loads: vec![GpuMassLinkLoad::default(); 5],
            ground_spheres: Vec::new(),
            ground_capsules: Vec::new(),
            ground_boxes: Vec::new(),
            ground_axial_shapes: Vec::new(),
            sphere_pairs: Vec::new(),
            static_sphere_pairs: Vec::new(),
            external_sphere_bodies: None,
            external_box_bodies: None,
            external_capsule_bodies: None,
            external_axial_bodies: None,
            external_convex_bodies: None,
            external_constraint_bodies: None,
            external_indexed_bodies: None,
            static_capsule_sphere_pairs: Vec::new(),
            static_box_sphere_pairs: Vec::new(),
            static_sphere_capsule_pairs: Vec::new(),
            static_capsule_pairs: Vec::new(),
            static_sphere_box_pairs: Vec::new(),
            static_capsule_box_pairs: Vec::new(),
            static_box_pairs: Vec::new(),
            static_box_capsule_pairs: Vec::new(),
            static_axial_sphere_pairs: Vec::new(),
            static_axial_capsule_pairs: Vec::new(),
            static_axial_box_pairs: Vec::new(),
            static_axial_convex_pairs: Vec::new(),
            axial_convex_pairs: Vec::new(),
            axial_sphere_pairs: Vec::new(),
            axial_box_pairs: Vec::new(),
            axial_capsule_pairs: Vec::new(),
            axial_pairs: Vec::new(),
            convex_sphere_pairs: Vec::new(),
            static_convex_sphere_pairs: Vec::new(),
            static_convex_capsule_pairs: Vec::new(),
            convex_capsule_pairs: Vec::new(),
            convex_pairs: Vec::new(),
            static_convex_pairs: Vec::new(),
            scene_convex_sphere_pairs: Vec::new(),
            scene_mesh_sphere_pairs: Vec::new(),
            scene_polyline_sphere_pairs: Vec::new(),
            scene_mesh_capsule_pairs: Vec::new(),
            scene_polyline_capsule_pairs: Vec::new(),
            scene_mesh_box_pairs: Vec::new(),
            scene_polyline_box_pairs: Vec::new(),
            scene_mesh_axial_pairs: Vec::new(),
            scene_polyline_axial_pairs: Vec::new(),
            scene_mesh_convex_pairs: Vec::new(),
            scene_polyline_convex_pairs: Vec::new(),
            scene_convex_capsule_pairs: Vec::new(),
            capsule_sphere_pairs: Vec::new(),
            capsule_pairs: vec![GpuArticulatedCapsulePair {
                first_link: 2,
                first_local_a: Vector3::new(0.0, -0.5, 0.0),
                first_local_b: Vector3::new(0.0, 0.5, 0.0),
                first_radius: 0.5,
                second_link: 4,
                second_local_a: Vector3::new(1.05, -0.5, 0.0),
                second_local_b: Vector3::new(1.05, 0.5, 0.0),
                second_radius: 0.5,
                restitution: 0.0,
                friction: 0.0,
            }],
            sphere_box_pairs: Vec::new(),
            capsule_box_pairs: Vec::new(),
            link_spheres: Vec::new(),
            link_capsules: Vec::new(),
            link_boxes: Vec::new(),
            box_pairs: Vec::new(),
            contact_iterations: 16,
            dynamic_material_rules: Vec::new(),
            ground_manifold_start: None,
            joint_friction: Vec::new(),
            joint_couplings: Vec::new(),
            link_point_constraints: Vec::new(),
            link_fixed_constraints: Vec::new(),
            contact_warm_start: false,
        };
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let batch =
                GpuArticulatedDynamicsBatch::new(&context, core::slice::from_ref(&input), 0.1)
                    .unwrap();
            batch.submit_steps(1).unwrap();
            let state = batch.readback().unwrap();
            assert!(
                (state[0].velocities[0] - 0.25).abs() < 2e-3,
                "{:?}",
                state[0]
            );
            assert!(
                (state[0].velocities[2] + 0.25).abs() < 2e-3,
                "{:?}",
                state[0]
            );
            assert!(state[0].velocities[1].abs() < 2e-3, "{:?}", state[0]);
            assert!(state[0].velocities[3].abs() < 2e-3, "{:?}", state[0]);
            let pair = input.capsule_pairs[0];
            let mut dynamic_input = input.clone();
            dynamic_input.capsule_pairs.clear();
            dynamic_input.link_capsules = vec![
                GpuArticulatedLinkCapsule {
                    link: pair.first_link,
                    local_a: pair.first_local_a,
                    local_b: pair.first_local_b,
                    radius: pair.first_radius,
                    restitution: pair.restitution,
                    friction: pair.friction,
                },
                GpuArticulatedLinkCapsule {
                    link: pair.second_link,
                    local_a: pair.second_local_a,
                    local_b: pair.second_local_b,
                    radius: pair.second_radius,
                    restitution: pair.restitution,
                    friction: pair.friction,
                },
            ];
            let dynamic_batch =
                GpuArticulatedDynamicsBatch::new(&context, &[dynamic_input], 0.1).unwrap();
            dynamic_batch.submit_steps(1).unwrap();
            let dynamic_state = dynamic_batch.readback().unwrap();
            assert!((&dynamic_state[0].velocities - &state[0].velocities).norm() < 2e-3);
            let rows = dynamic_batch
                .ground_contact
                .as_ref()
                .unwrap()
                .readback_dynamic_row_state(context.queue());
            assert_eq!(rows.len(), 2);
            assert!(
                rows.iter().all(|row| row.0 > 0.0 && row.1[0] > 0.0),
                "{rows:?}"
            );
            batch.submit_steps(99).unwrap();
            let settled = batch.readback().unwrap();
            assert!(settled[0].velocities.norm() < 2e-3, "{:?}", settled[0]);
            assert!(settled[0].positions[1].abs() < 2e-3, "{:?}", settled[0]);
            assert!(settled[0].positions[3].abs() < 2e-3, "{:?}", settled[0]);
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn articulated_box_plane_uses_four_support_corners() {
        let link = |mass| LinkSpec {
            mass,
            center_of_mass: Vector3::zeros(),
            inertia: Matrix3::identity() * mass,
        };
        let articulation = Articulation::new(
            vec![link(0.0), link(0.0), link(1.0)],
            vec![
                JointSpec {
                    parent: 0,
                    child: 1,
                    kind: JointKind::Prismatic,
                    origin: Isometry3::identity(),
                    axis: Vector3::z(),
                    limits: None,
                },
                JointSpec {
                    parent: 1,
                    child: 2,
                    kind: JointKind::Revolute,
                    origin: Isometry3::identity(),
                    axis: Vector3::y(),
                    limits: None,
                },
            ],
            0,
        )
        .unwrap();
        let input = GpuArticulatedDynamicsInput {
            articulation: &articulation,
            root_pose: Isometry3::translation(0.0, 0.0, 0.55),
            state: GpuGeneralizedState {
                positions: DVector::zeros(2),
                velocities: DVector::from_column_slice(&[-1.0, 0.0]),
            },
            gravity: Vector3::zeros(),
            joints: vec![GpuJointForceInput::default(); 2],
            joint_velocity_limit: None,
            coordinate_velocity_limits: None,
            link_loads: vec![GpuMassLinkLoad::default(); 3],
            ground_spheres: Vec::new(),
            ground_capsules: Vec::new(),
            ground_boxes: vec![GpuArticulatedGroundBox {
                link: 2,
                local_pose: Isometry3::identity(),
                half_extents: Vector3::new(0.5, 0.5, 0.5),
                plane_normal: Vector3::z(),
                plane_offset: 0.0,
                plane_xy_half_extent: None,
                restitution: 0.0,
                friction: 0.0,
            }],
            ground_axial_shapes: Vec::new(),
            sphere_pairs: Vec::new(),
            static_sphere_pairs: Vec::new(),
            external_sphere_bodies: None,
            external_box_bodies: None,
            external_capsule_bodies: None,
            external_axial_bodies: None,
            external_convex_bodies: None,
            external_constraint_bodies: None,
            external_indexed_bodies: None,
            static_capsule_sphere_pairs: Vec::new(),
            static_box_sphere_pairs: Vec::new(),
            static_sphere_capsule_pairs: Vec::new(),
            static_capsule_pairs: Vec::new(),
            static_sphere_box_pairs: Vec::new(),
            static_capsule_box_pairs: Vec::new(),
            static_box_pairs: Vec::new(),
            static_box_capsule_pairs: Vec::new(),
            static_axial_sphere_pairs: Vec::new(),
            static_axial_capsule_pairs: Vec::new(),
            static_axial_box_pairs: Vec::new(),
            static_axial_convex_pairs: Vec::new(),
            axial_convex_pairs: Vec::new(),
            axial_sphere_pairs: Vec::new(),
            axial_box_pairs: Vec::new(),
            axial_capsule_pairs: Vec::new(),
            axial_pairs: Vec::new(),
            convex_sphere_pairs: Vec::new(),
            static_convex_sphere_pairs: Vec::new(),
            static_convex_capsule_pairs: Vec::new(),
            convex_capsule_pairs: Vec::new(),
            convex_pairs: Vec::new(),
            static_convex_pairs: Vec::new(),
            scene_convex_sphere_pairs: Vec::new(),
            scene_mesh_sphere_pairs: Vec::new(),
            scene_polyline_sphere_pairs: Vec::new(),
            scene_mesh_capsule_pairs: Vec::new(),
            scene_polyline_capsule_pairs: Vec::new(),
            scene_mesh_box_pairs: Vec::new(),
            scene_polyline_box_pairs: Vec::new(),
            scene_mesh_axial_pairs: Vec::new(),
            scene_polyline_axial_pairs: Vec::new(),
            scene_mesh_convex_pairs: Vec::new(),
            scene_polyline_convex_pairs: Vec::new(),
            scene_convex_capsule_pairs: Vec::new(),
            capsule_sphere_pairs: Vec::new(),
            capsule_pairs: Vec::new(),
            sphere_box_pairs: Vec::new(),
            capsule_box_pairs: Vec::new(),
            link_spheres: Vec::new(),
            link_capsules: Vec::new(),
            link_boxes: Vec::new(),
            box_pairs: Vec::new(),
            contact_iterations: 16,
            dynamic_material_rules: Vec::new(),
            ground_manifold_start: None,
            joint_friction: Vec::new(),
            joint_couplings: Vec::new(),
            link_point_constraints: Vec::new(),
            link_fixed_constraints: Vec::new(),
            contact_warm_start: false,
        };
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let batch =
                GpuArticulatedDynamicsBatch::new(&context, core::slice::from_ref(&input), 0.1)
                    .unwrap();
            batch.submit_steps(1).unwrap();
            let first = batch.readback().unwrap();
            assert!(
                (first[0].velocities[0] + 0.5).abs() < 2e-4,
                "{:?}",
                first[0]
            );
            assert!(first[0].velocities[1].abs() < 2e-4, "{:?}", first[0]);
            batch.submit_steps(99).unwrap();
            let settled = batch.readback().unwrap();
            assert!(settled[0].velocities.norm() < 2e-3, "{:?}", settled[0]);
            assert!(settled[0].positions[1].abs() < 2e-3, "{:?}", settled[0]);

            let mut tilted = input.clone();
            tilted.ground_boxes[0].local_pose =
                Isometry3::new(Vector3::zeros(), Vector3::new(0.0, 0.2, 0.0));
            let tilted_batch = GpuArticulatedDynamicsBatch::new(&context, &[tilted], 0.1).unwrap();
            tilted_batch.submit_steps(1).unwrap();
            let tilted_state = tilted_batch.readback().unwrap();
            assert!(
                tilted_state[0].velocities[1].abs() > 0.01,
                "{:?}",
                tilted_state[0]
            );

            let mut rotated_link = input.clone();
            rotated_link.state.positions[1] = 0.2;
            let rotated_link_batch =
                GpuArticulatedDynamicsBatch::new(&context, &[rotated_link], 0.1).unwrap();
            rotated_link_batch.submit_steps(1).unwrap();
            let rotated_link_state = rotated_link_batch.readback().unwrap();
            assert!(
                (&rotated_link_state[0].velocities - &tilted_state[0].velocities).norm() < 2e-3,
                "link {:?}, local {:?}",
                rotated_link_state[0],
                tilted_state[0],
            );

            let mut raised = input.clone();
            raised.ground_boxes[0].local_pose.translation.vector.z = 0.2;
            let raised_batch = GpuArticulatedDynamicsBatch::new(&context, &[raised], 0.1).unwrap();
            raised_batch.submit_steps(1).unwrap();
            let raised_state = raised_batch.readback().unwrap();
            assert!((raised_state[0].velocities[0] + 1.0).abs() < 2e-4);

            let mut scaled_plane = input.clone();
            scaled_plane.root_pose.translation.vector.z += 0.1;
            scaled_plane.ground_boxes[0].plane_normal = Vector3::z() * 2.0;
            scaled_plane.ground_boxes[0].plane_offset = 0.2;
            let scaled_batch =
                GpuArticulatedDynamicsBatch::new(&context, &[scaled_plane], 0.1).unwrap();
            scaled_batch.submit_steps(1).unwrap();
            let scaled_state = scaled_batch.readback().unwrap();
            assert!((&scaled_state[0].velocities - &first[0].velocities).norm() < 2e-4);

            let mut invalid_extent = input.clone();
            invalid_extent.ground_boxes[0].plane_xy_half_extent = Some(0.5);
            invalid_extent.ground_boxes[0].plane_normal = Vector3::new(1.0, 0.0, 1.0);
            assert!(GpuArticulatedDynamicsBatch::new(&context, &[invalid_extent], 0.1).is_err());

            let mut invalid = input.clone();
            invalid.ground_boxes[0].half_extents.x = -0.5;
            assert!(GpuArticulatedDynamicsBatch::new(&context, &[invalid], 0.1).is_err());
            let mut invalid_pose = input.clone();
            invalid_pose.ground_boxes[0].local_pose.translation.vector.x = f64::NAN;
            assert!(GpuArticulatedDynamicsBatch::new(&context, &[invalid_pose], 0.1).is_err());
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn articulated_sphere_box_pair_transfers_impulse_and_handles_interior() {
        let link = |mass| LinkSpec {
            mass,
            center_of_mass: Vector3::zeros(),
            inertia: Matrix3::identity() * mass,
        };
        let joint = |child| JointSpec {
            parent: 0,
            child,
            kind: JointKind::Prismatic,
            origin: Isometry3::identity(),
            axis: Vector3::x(),
            limits: None,
        };
        let articulation = Articulation::new(
            vec![link(0.0), link(1.0), link(1.0)],
            vec![joint(1), joint(2)],
            0,
        )
        .unwrap();
        let input = GpuArticulatedDynamicsInput {
            articulation: &articulation,
            root_pose: Isometry3::identity(),
            state: GpuGeneralizedState {
                positions: DVector::zeros(2),
                velocities: DVector::from_column_slice(&[1.0, -1.0]),
            },
            gravity: Vector3::zeros(),
            joints: vec![GpuJointForceInput::default(); 2],
            joint_velocity_limit: None,
            coordinate_velocity_limits: None,
            link_loads: vec![GpuMassLinkLoad::default(); 3],
            ground_spheres: Vec::new(),
            ground_capsules: Vec::new(),
            ground_boxes: Vec::new(),
            ground_axial_shapes: Vec::new(),
            sphere_pairs: Vec::new(),
            static_sphere_pairs: Vec::new(),
            external_sphere_bodies: None,
            external_box_bodies: None,
            external_capsule_bodies: None,
            external_axial_bodies: None,
            external_convex_bodies: None,
            external_constraint_bodies: None,
            external_indexed_bodies: None,
            static_capsule_sphere_pairs: Vec::new(),
            static_box_sphere_pairs: Vec::new(),
            static_sphere_capsule_pairs: Vec::new(),
            static_capsule_pairs: Vec::new(),
            static_sphere_box_pairs: Vec::new(),
            static_capsule_box_pairs: Vec::new(),
            static_box_pairs: Vec::new(),
            static_box_capsule_pairs: Vec::new(),
            static_axial_sphere_pairs: Vec::new(),
            static_axial_capsule_pairs: Vec::new(),
            static_axial_box_pairs: Vec::new(),
            static_axial_convex_pairs: Vec::new(),
            axial_convex_pairs: Vec::new(),
            axial_sphere_pairs: Vec::new(),
            axial_box_pairs: Vec::new(),
            axial_capsule_pairs: Vec::new(),
            axial_pairs: Vec::new(),
            convex_sphere_pairs: Vec::new(),
            static_convex_sphere_pairs: Vec::new(),
            static_convex_capsule_pairs: Vec::new(),
            convex_capsule_pairs: Vec::new(),
            convex_pairs: Vec::new(),
            static_convex_pairs: Vec::new(),
            scene_convex_sphere_pairs: Vec::new(),
            scene_mesh_sphere_pairs: Vec::new(),
            scene_polyline_sphere_pairs: Vec::new(),
            scene_mesh_capsule_pairs: Vec::new(),
            scene_polyline_capsule_pairs: Vec::new(),
            scene_mesh_box_pairs: Vec::new(),
            scene_polyline_box_pairs: Vec::new(),
            scene_mesh_axial_pairs: Vec::new(),
            scene_polyline_axial_pairs: Vec::new(),
            scene_mesh_convex_pairs: Vec::new(),
            scene_polyline_convex_pairs: Vec::new(),
            scene_convex_capsule_pairs: Vec::new(),
            capsule_sphere_pairs: Vec::new(),
            capsule_pairs: Vec::new(),
            sphere_box_pairs: vec![GpuArticulatedSphereBoxPair {
                sphere_link: 1,
                sphere_local_center: Vector3::zeros(),
                sphere_radius: 0.5,
                box_link: 2,
                box_local_pose: Isometry3::translation(1.05, 0.0, 0.0),
                box_half_extents: Vector3::repeat(0.5),
                restitution: 0.0,
                friction: 0.0,
            }],
            capsule_box_pairs: Vec::new(),
            link_spheres: Vec::new(),
            link_capsules: Vec::new(),
            link_boxes: Vec::new(),
            box_pairs: Vec::new(),
            contact_iterations: 8,
            dynamic_material_rules: Vec::new(),
            ground_manifold_start: None,
            joint_friction: Vec::new(),
            joint_couplings: Vec::new(),
            link_point_constraints: Vec::new(),
            link_fixed_constraints: Vec::new(),
            contact_warm_start: false,
        };
        let expected = DVector::from_column_slice(&[0.25, -0.25]);
        let sphere = GpuArticulatedLinkSphere {
            link: 1,
            local_center: Vector3::zeros(),
            radius: 0.5,
            restitution: 0.2,
            friction: 0.25,
        };
        let box_shape = GpuArticulatedLinkBox {
            link: 2,
            local_pose: Isometry3::translation(1.05, 0.0, 0.0),
            half_extents: Vector3::repeat(0.5),
            restitution: 0.7,
            friction: 1.0,
        };
        let generated =
            self_contact_sphere_box_pairs(&articulation, &[sphere], &[box_shape]).unwrap();
        assert_eq!(generated.len(), 1);
        assert_eq!(generated[0].restitution, 0.7);
        assert_eq!(generated[0].friction, 0.5);
        assert!(same_sphere_box_geometry(
            &generated[0],
            &input.sphere_box_pairs[0]
        ));
        let mut excluded = articulation.clone();
        excluded.exclude_collision_pair(1, 2).unwrap();
        assert!(
            self_contact_sphere_box_pairs(&excluded, &[sphere], &[box_shape])
                .unwrap()
                .is_empty()
        );
        assert!(
            self_contact_sphere_box_pairs(
                &articulation,
                &[sphere],
                &[GpuArticulatedLinkBox {
                    link: 0,
                    ..box_shape
                }],
            )
            .unwrap()
            .is_empty()
        );
        assert!(
            self_contact_sphere_box_pairs(
                &articulation,
                &[sphere],
                &[GpuArticulatedLinkBox {
                    link: 1,
                    ..box_shape
                }],
            )
            .unwrap()
            .is_empty()
        );
        for invalid in [
            GpuArticulatedLinkBox {
                half_extents: Vector3::new(0.0, 0.5, 0.5),
                ..box_shape
            },
            GpuArticulatedLinkBox {
                local_pose: Isometry3::translation(f64::NAN, 0.0, 0.0),
                ..box_shape
            },
            GpuArticulatedLinkBox {
                friction: f64::INFINITY,
                ..box_shape
            },
            GpuArticulatedLinkBox {
                half_extents: Vector3::new(f64::MAX, 0.5, 0.5),
                ..box_shape
            },
        ] {
            assert!(self_contact_sphere_box_pairs(&articulation, &[sphere], &[invalid]).is_err());
        }
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let batch =
                GpuArticulatedDynamicsBatch::new(&context, core::slice::from_ref(&input), 0.1)
                    .unwrap();
            batch.submit_steps(1).unwrap();
            let first = batch.readback().unwrap();
            assert!(
                (&first[0].velocities - &expected).norm() < 2e-4,
                "{:?}",
                first[0]
            );
            let mut automatic = input.clone();
            automatic.sphere_box_pairs.clear();
            automatic.link_spheres = vec![GpuArticulatedLinkSphere {
                restitution: 0.0,
                friction: 0.0,
                ..sphere
            }];
            automatic.link_boxes = vec![GpuArticulatedLinkBox {
                restitution: 0.0,
                friction: 0.0,
                ..box_shape
            }];
            let automatic_batch =
                GpuArticulatedDynamicsBatch::new(&context, &[automatic.clone()], 0.1).unwrap();
            automatic_batch.submit_steps(1).unwrap();
            let automatic_state = automatic_batch.readback().unwrap();
            assert!(
                (&automatic_state[0].velocities - &expected).norm() < 2e-4,
                "{:?}",
                automatic_state[0]
            );
            let rows = automatic_batch
                .ground_contact
                .as_ref()
                .unwrap()
                .readback_dynamic_row_state(context.queue());
            assert_eq!(rows.len(), 1);
            assert!(rows[0].0 > 0.0 && rows[0].1[0] > 0.0, "{rows:?}");
            let mut multiple_shapes = automatic.clone();
            let nearby_sphere = multiple_shapes.link_spheres[0];
            multiple_shapes.link_spheres.push(GpuArticulatedLinkSphere {
                local_center: Vector3::new(10.0, 0.0, 0.0),
                ..nearby_sphere
            });
            let multiple_batch =
                GpuArticulatedDynamicsBatch::new(&context, &[multiple_shapes], 0.1).unwrap();
            multiple_batch.submit_steps(1).unwrap();
            let rows = multiple_batch
                .ground_contact
                .as_ref()
                .unwrap()
                .readback_dynamic_row_state(context.queue());
            assert_eq!(rows.len(), 2);
            assert!(rows[0].0 > 0.0 && rows[0].1[0] > 0.0, "{rows:?}");
            assert_eq!(rows[1], (0.0, [0.0; 4]));
            let mut mixed_shapes = automatic.clone();
            mixed_shapes.link_capsules.push(GpuArticulatedLinkCapsule {
                link: sphere.link,
                local_a: Vector3::new(10.0, 0.0, 0.0),
                local_b: Vector3::new(10.0, 1.0, 0.0),
                radius: 0.2,
                restitution: 0.0,
                friction: 0.0,
            });
            let mixed_batch =
                GpuArticulatedDynamicsBatch::new(&context, &[mixed_shapes], 0.1).unwrap();
            mixed_batch.submit_steps(1).unwrap();
            let rows = mixed_batch
                .ground_contact
                .as_ref()
                .unwrap()
                .readback_dynamic_row_state(context.queue());
            assert_eq!(rows.len(), 3);
            assert!(rows[0].0 > 0.0 && rows[0].1[0] > 0.0, "{rows:?}");
            assert_eq!(rows[1], (0.0, [0.0; 4]));
            assert_eq!(rows[2], (0.0, [0.0; 4]));
            let mut separated_dynamic = automatic.clone();
            separated_dynamic.link_boxes[0]
                .local_pose
                .translation
                .vector
                .x = 3.0;
            let separated_dynamic_batch =
                GpuArticulatedDynamicsBatch::new(&context, &[separated_dynamic], 0.1).unwrap();
            separated_dynamic_batch.submit_steps(1).unwrap();
            let rows = separated_dynamic_batch
                .ground_contact
                .as_ref()
                .unwrap()
                .readback_dynamic_row_state(context.queue());
            assert_eq!(rows, vec![(0.0, [0.0; 4])]);
            let mut outward = automatic.joints.clone();
            outward[0].base_force = -20.0;
            outward[1].base_force = 20.0;
            automatic_batch.update_joints(&[outward]).unwrap();
            automatic_batch.submit_steps(3).unwrap();
            let rows = automatic_batch
                .ground_contact
                .as_ref()
                .unwrap()
                .readback_dynamic_row_state(context.queue());
            assert_eq!(rows, vec![(0.0, [0.0; 4])]);
            automatic.sphere_box_pairs = input.sphere_box_pairs.clone();
            automatic.link_spheres[0].restitution = 1.0;
            automatic.link_boxes[0].restitution = 1.0;
            let explicit_batch =
                GpuArticulatedDynamicsBatch::new(&context, &[automatic], 0.1).unwrap();
            explicit_batch.submit_steps(1).unwrap();
            let explicit_state = explicit_batch.readback().unwrap();
            assert!(
                (&explicit_state[0].velocities - &expected).norm() < 2e-4,
                "{:?}",
                explicit_state[0]
            );
            let rows = explicit_batch
                .ground_contact
                .as_ref()
                .unwrap()
                .readback_dynamic_row_state(context.queue());
            assert!(rows.is_empty());
            batch.submit_steps(1).unwrap();
            let stopped = batch.readback().unwrap();
            assert!(stopped[0].velocities.norm() < 2e-4, "{:?}", stopped[0]);

            let mut rotated = input.clone();
            rotated.sphere_box_pairs[0].box_local_pose =
                Isometry3::new(Vector3::new(1.05, 0.0, 0.0), Vector3::new(0.0, 0.0, 0.2));
            let rotated_batch =
                GpuArticulatedDynamicsBatch::new(&context, &[rotated.clone()], 0.1).unwrap();
            rotated_batch.submit_steps(1).unwrap();
            let rotated_state = rotated_batch.readback().unwrap();
            assert!(
                rotated_state[0].velocities[0] < 1.0,
                "{:?}",
                rotated_state[0]
            );
            assert!(
                rotated_state[0].velocities[1] > -1.0,
                "{:?}",
                rotated_state[0]
            );
            let rotated_pair = rotated.sphere_box_pairs.remove(0);
            rotated.link_spheres.push(GpuArticulatedLinkSphere {
                restitution: rotated_pair.restitution,
                friction: rotated_pair.friction,
                ..sphere
            });
            rotated.link_boxes.push(GpuArticulatedLinkBox {
                local_pose: rotated_pair.box_local_pose,
                restitution: rotated_pair.restitution,
                friction: rotated_pair.friction,
                ..box_shape
            });
            let rotated_dynamic =
                GpuArticulatedDynamicsBatch::new(&context, &[rotated], 0.1).unwrap();
            rotated_dynamic.submit_steps(1).unwrap();
            let rotated_dynamic_state = rotated_dynamic.readback().unwrap();
            assert!(
                (&rotated_dynamic_state[0].velocities - &rotated_state[0].velocities).norm() < 2e-4
            );

            let mut inside = input.clone();
            inside.state.velocities.fill(0.0);
            inside.sphere_box_pairs[0].sphere_local_center.x = 0.25;
            inside.sphere_box_pairs[0].box_local_pose = Isometry3::identity();
            let inside_batch = GpuArticulatedDynamicsBatch::new(&context, &[inside], 0.1).unwrap();
            inside_batch.submit_steps(1).unwrap();
            let inside_state = inside_batch.readback().unwrap();
            assert!(
                (inside_state[0].velocities[0] - 0.75).abs() < 2e-4,
                "{:?}",
                inside_state[0]
            );
            assert!((inside_state[0].velocities[1] + 0.75).abs() < 2e-4);

            let mut separated = input.clone();
            separated.sphere_box_pairs[0]
                .box_local_pose
                .translation
                .vector
                .x = 3.0;
            let separated_batch =
                GpuArticulatedDynamicsBatch::new(&context, &[separated], 0.1).unwrap();
            separated_batch.submit_steps(1).unwrap();
            let separated_state = separated_batch.readback().unwrap();
            assert!((&separated_state[0].velocities - &input.state.velocities).norm() < 2e-4);

            let mut invalid = input.clone();
            invalid.sphere_box_pairs[0].box_half_extents.x = -0.5;
            assert!(GpuArticulatedDynamicsBatch::new(&context, &[invalid], 0.1).is_err());
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn articulated_box_pair_solves_four_face_contacts() {
        let link = |mass| LinkSpec {
            mass,
            center_of_mass: Vector3::zeros(),
            inertia: Matrix3::identity() * mass,
        };
        let joint = |child| JointSpec {
            parent: 0,
            child,
            kind: JointKind::Prismatic,
            origin: Isometry3::identity(),
            axis: Vector3::x(),
            limits: None,
        };
        let articulation = Articulation::new(
            vec![link(0.0), link(1.0), link(1.0)],
            vec![joint(1), joint(2)],
            0,
        )
        .unwrap();
        let first = GpuArticulatedLinkBox {
            link: 1,
            local_pose: Isometry3::identity(),
            half_extents: Vector3::repeat(0.5),
            restitution: 0.2,
            friction: 0.25,
        };
        let second = GpuArticulatedLinkBox {
            link: 2,
            local_pose: Isometry3::translation(1.05, 0.0, 0.0),
            half_extents: Vector3::repeat(0.5),
            restitution: 0.7,
            friction: 1.0,
        };
        let generated = self_contact_box_pairs(&articulation, &[first, second]).unwrap();
        assert_eq!(generated.len(), 1);
        assert_eq!(generated[0].restitution, 0.7);
        assert_eq!(generated[0].friction, 0.5);
        let mut excluded = articulation.clone();
        excluded.exclude_collision_pair(1, 2).unwrap();
        assert!(
            self_contact_box_pairs(&excluded, &[first, second])
                .unwrap()
                .is_empty()
        );
        assert!(
            self_contact_box_pairs(
                &articulation,
                &[first, GpuArticulatedLinkBox { link: 1, ..second }],
            )
            .unwrap()
            .is_empty()
        );
        let mut invalid = second;
        invalid.half_extents.z = 0.0;
        assert!(self_contact_box_pairs(&articulation, &[first, invalid]).is_err());
        let input = GpuArticulatedDynamicsInput {
            articulation: &articulation,
            root_pose: Isometry3::identity(),
            state: GpuGeneralizedState {
                positions: DVector::zeros(2),
                velocities: DVector::from_column_slice(&[1.0, -1.0]),
            },
            gravity: Vector3::zeros(),
            joints: vec![GpuJointForceInput::default(); 2],
            joint_velocity_limit: None,
            coordinate_velocity_limits: None,
            link_loads: vec![GpuMassLinkLoad::default(); 3],
            ground_spheres: Vec::new(),
            ground_capsules: Vec::new(),
            ground_boxes: Vec::new(),
            ground_axial_shapes: Vec::new(),
            sphere_pairs: Vec::new(),
            static_sphere_pairs: Vec::new(),
            external_sphere_bodies: None,
            external_box_bodies: None,
            external_capsule_bodies: None,
            external_axial_bodies: None,
            external_convex_bodies: None,
            external_constraint_bodies: None,
            external_indexed_bodies: None,
            static_capsule_sphere_pairs: Vec::new(),
            static_box_sphere_pairs: Vec::new(),
            static_sphere_capsule_pairs: Vec::new(),
            static_capsule_pairs: Vec::new(),
            static_sphere_box_pairs: Vec::new(),
            static_capsule_box_pairs: Vec::new(),
            static_box_pairs: Vec::new(),
            static_box_capsule_pairs: Vec::new(),
            static_axial_sphere_pairs: Vec::new(),
            static_axial_capsule_pairs: Vec::new(),
            static_axial_box_pairs: Vec::new(),
            static_axial_convex_pairs: Vec::new(),
            axial_convex_pairs: Vec::new(),
            axial_sphere_pairs: Vec::new(),
            axial_box_pairs: Vec::new(),
            axial_capsule_pairs: Vec::new(),
            axial_pairs: Vec::new(),
            convex_sphere_pairs: Vec::new(),
            static_convex_sphere_pairs: Vec::new(),
            static_convex_capsule_pairs: Vec::new(),
            convex_capsule_pairs: Vec::new(),
            convex_pairs: Vec::new(),
            static_convex_pairs: Vec::new(),
            scene_convex_sphere_pairs: Vec::new(),
            scene_mesh_sphere_pairs: Vec::new(),
            scene_polyline_sphere_pairs: Vec::new(),
            scene_mesh_capsule_pairs: Vec::new(),
            scene_polyline_capsule_pairs: Vec::new(),
            scene_mesh_box_pairs: Vec::new(),
            scene_polyline_box_pairs: Vec::new(),
            scene_mesh_axial_pairs: Vec::new(),
            scene_polyline_axial_pairs: Vec::new(),
            scene_mesh_convex_pairs: Vec::new(),
            scene_polyline_convex_pairs: Vec::new(),
            scene_convex_capsule_pairs: Vec::new(),
            capsule_sphere_pairs: Vec::new(),
            capsule_pairs: Vec::new(),
            sphere_box_pairs: Vec::new(),
            capsule_box_pairs: Vec::new(),
            box_pairs: vec![GpuArticulatedBoxPair {
                restitution: 0.0,
                friction: 0.0,
                ..generated[0]
            }],
            link_spheres: Vec::new(),
            link_capsules: Vec::new(),
            link_boxes: Vec::new(),
            contact_iterations: 16,
            dynamic_material_rules: Vec::new(),
            ground_manifold_start: None,
            joint_friction: Vec::new(),
            joint_couplings: Vec::new(),
            link_point_constraints: Vec::new(),
            link_fixed_constraints: Vec::new(),
            contact_warm_start: false,
        };
        let expected = DVector::from_column_slice(&[0.25, -0.25]);
        let phase: f64 = 23.0 * 0.37;
        let first_edge_pose = Isometry3::from_parts(
            nalgebra::Translation3::identity(),
            nalgebra::UnitQuaternion::from_euler_angles(phase * 0.13, phase * 0.07, phase * 0.21),
        );
        let second_edge_pose = Isometry3::from_parts(
            nalgebra::Translation3::new(
                phase.sin() * 0.8,
                (phase * 1.7).cos() * 0.6,
                (phase * 0.9).sin() * 0.55,
            ),
            nalgebra::UnitQuaternion::from_euler_angles(phase * 0.31, phase * 0.19, phase * 0.11),
        );
        let first_extent = Vector3::new(0.4, 0.2, 0.3);
        let second_extent = Vector3::new(0.3, 0.4, 0.2);
        let basis = [Vector3::x(), Vector3::y(), Vector3::z()];
        let first_axes = basis.map(|axis| first_edge_pose.rotation * axis);
        let second_axes = basis.map(|axis| second_edge_pose.rotation * axis);
        let delta = second_edge_pose.translation.vector - first_edge_pose.translation.vector;
        let mut best = (f64::NEG_INFINITY, 0usize);
        for (index, axis) in first_axes
            .into_iter()
            .chain(second_axes)
            .chain(first_axes.into_iter().flat_map(|first| {
                second_axes
                    .into_iter()
                    .map(move |second| first.cross(&second))
            }))
            .enumerate()
        {
            if axis.norm_squared() < 1e-10 {
                continue;
            }
            let direction = axis.normalize();
            let separation = delta.dot(&direction).abs()
                - (0..3)
                    .map(|slot| {
                        first_extent[slot] * first_axes[slot].dot(&direction).abs()
                            + second_extent[slot] * second_axes[slot].dot(&direction).abs()
                    })
                    .sum::<f64>();
            if separation > best.0 {
                best = (separation, index);
            }
        }
        assert!(best.1 >= 6 && best.0 < -0.05, "{best:?}");
        let angular_articulation = Articulation::new(
            vec![link(0.0), link(1.0), link(0.0), link(1.0)],
            vec![
                joint(1),
                joint(2),
                JointSpec {
                    parent: 2,
                    child: 3,
                    kind: JointKind::Revolute,
                    origin: Isometry3::identity(),
                    axis: Vector3::z(),
                    limits: None,
                },
            ],
            0,
        )
        .unwrap();
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let batch =
                GpuArticulatedDynamicsBatch::new(&context, core::slice::from_ref(&input), 0.1)
                    .unwrap();
            batch.submit_steps(1).unwrap();
            let state = batch.readback().unwrap();
            assert!(
                (&state[0].velocities - &expected).norm() < 2e-4,
                "{:?}",
                state[0]
            );

            let mut automatic = input.clone();
            automatic.box_pairs.clear();
            automatic.link_boxes = vec![
                GpuArticulatedLinkBox {
                    restitution: 0.0,
                    friction: 0.0,
                    ..first
                },
                GpuArticulatedLinkBox {
                    restitution: 0.0,
                    friction: 0.0,
                    ..second
                },
            ];
            let automatic_batch =
                GpuArticulatedDynamicsBatch::new(&context, &[automatic.clone()], 0.1).unwrap();
            automatic_batch.submit_steps(1).unwrap();
            let automatic_state = automatic_batch.readback().unwrap();
            assert!(
                (&automatic_state[0].velocities - &expected).norm() < 2e-4,
                "{:?}",
                automatic_state[0]
            );
            let rows = automatic_batch
                .ground_contact
                .as_ref()
                .unwrap()
                .readback_dynamic_row_state(context.queue());
            assert_eq!(rows.len(), 4);
            assert!(rows.iter().all(|row| row.0 > 0.0), "{rows:?}");
            assert!(rows.iter().any(|row| row.1[0] > 0.0), "{rows:?}");
            let mut three_boxes = automatic.clone();
            three_boxes.link_boxes.push(GpuArticulatedLinkBox {
                local_pose: Isometry3::translation(10.0, 0.0, 0.0),
                ..first
            });
            let three_batch =
                GpuArticulatedDynamicsBatch::new(&context, &[three_boxes], 0.1).unwrap();
            three_batch.submit_steps(1).unwrap();
            let rows = three_batch
                .ground_contact
                .as_ref()
                .unwrap()
                .readback_dynamic_row_state(context.queue());
            assert_eq!(rows.len(), 8);
            assert!(rows[0..4].iter().all(|row| row.0 > 0.0));
            assert!(rows[4..].iter().all(|row| *row == (0.0, [0.0; 4])));
            let mut separated_dynamic = automatic.clone();
            separated_dynamic.link_boxes[1].local_pose = Isometry3::translation(3.0, 0.0, 0.0);
            let separated_dynamic_batch =
                GpuArticulatedDynamicsBatch::new(&context, &[separated_dynamic], 0.1).unwrap();
            separated_dynamic_batch.submit_steps(1).unwrap();
            let rows = separated_dynamic_batch
                .ground_contact
                .as_ref()
                .unwrap()
                .readback_dynamic_row_state(context.queue());
            assert_eq!(rows, vec![(0.0, [0.0; 4]); 4]);
            let mut outward = automatic.joints.clone();
            outward[0].base_force = -20.0;
            outward[1].base_force = 20.0;
            automatic_batch.update_joints(&[outward]).unwrap();
            automatic_batch.submit_steps(3).unwrap();
            let rows = automatic_batch
                .ground_contact
                .as_ref()
                .unwrap()
                .readback_dynamic_row_state(context.queue());
            assert_eq!(rows, vec![(0.0, [0.0; 4]); 4]);
            automatic.box_pairs = input.box_pairs.clone();
            automatic.link_boxes[0].restitution = 1.0;
            automatic.link_boxes[1].restitution = 1.0;
            automatic.link_boxes.swap(0, 1);
            let prioritized =
                GpuArticulatedDynamicsBatch::new(&context, &[automatic], 0.1).unwrap();
            prioritized.submit_steps(1).unwrap();
            let priority_state = prioritized.readback().unwrap();
            assert!(
                (&priority_state[0].velocities - &expected).norm() < 2e-4,
                "{:?}",
                priority_state[0]
            );
            let rows = prioritized
                .ground_contact
                .as_ref()
                .unwrap()
                .readback_dynamic_row_state(context.queue());
            assert!(rows.is_empty());

            let mut separated = input.clone();
            separated.box_pairs[0].second_local_pose = Isometry3::translation(3.0, 0.0, 0.0);
            let separated_batch =
                GpuArticulatedDynamicsBatch::new(&context, &[separated], 0.1).unwrap();
            separated_batch.submit_steps(1).unwrap();
            let separated_state = separated_batch.readback().unwrap();
            assert!(
                (&separated_state[0].velocities - &input.state.velocities).norm() < 2e-4,
                "{:?}",
                separated_state[0]
            );

            let mut overlapping = input.clone();
            overlapping.state.velocities.fill(0.0);
            overlapping.box_pairs[0].second_local_pose = Isometry3::translation(0.9, 0.0, 0.0);
            let overlapping_batch =
                GpuArticulatedDynamicsBatch::new(&context, &[overlapping], 0.1).unwrap();
            overlapping_batch.submit_steps(1).unwrap();
            let overlapping_state = overlapping_batch.readback().unwrap();
            assert!(
                overlapping_state[0].velocities[0] < -0.1,
                "{:?}",
                overlapping_state[0]
            );
            assert!(
                overlapping_state[0].velocities[1] > 0.1,
                "{:?}",
                overlapping_state[0]
            );

            let mut rotated = input.clone();
            rotated.box_pairs[0].second_local_pose =
                Isometry3::new(Vector3::new(1.05, 0.0, 0.0), Vector3::new(0.0, 0.0, 0.2));
            let rotated_batch =
                GpuArticulatedDynamicsBatch::new(&context, &[rotated], 0.1).unwrap();
            rotated_batch.submit_steps(1).unwrap();
            let rotated_state = rotated_batch.readback().unwrap();
            assert!(
                rotated_state[0].velocities[0] < 1.0,
                "{:?}",
                rotated_state[0]
            );
            assert!(
                rotated_state[0].velocities[1] > -1.0,
                "{:?}",
                rotated_state[0]
            );

            let mut edge = input.clone();
            edge.state.velocities.fill(0.0);
            edge.box_pairs[0].first_local_pose = first_edge_pose;
            edge.box_pairs[0].first_half_extents = first_extent;
            edge.box_pairs[0].second_local_pose = second_edge_pose;
            edge.box_pairs[0].second_half_extents = second_extent;
            let edge_batch = GpuArticulatedDynamicsBatch::new(&context, &[edge], 0.1).unwrap();
            edge_batch.submit_steps(1).unwrap();
            let edge_state = edge_batch.readback().unwrap();
            assert!(edge_state[0].velocities[0] < -0.05, "{:?}", edge_state[0]);
            assert!(edge_state[0].velocities[1] > 0.05, "{:?}", edge_state[0]);

            let mut angular_edge = input.clone();
            angular_edge.articulation = &angular_articulation;
            angular_edge.state = GpuGeneralizedState {
                positions: DVector::zeros(3),
                velocities: DVector::zeros(3),
            };
            angular_edge.joints = vec![GpuJointForceInput::default(); 3];
            angular_edge.link_loads = vec![GpuMassLinkLoad::default(); 4];
            angular_edge.box_pairs[0].first_local_pose = first_edge_pose;
            angular_edge.box_pairs[0].first_half_extents = first_extent;
            angular_edge.box_pairs[0].second_link = 3;
            angular_edge.box_pairs[0].second_local_pose = second_edge_pose;
            angular_edge.box_pairs[0].second_half_extents = second_extent;
            let angular_batch =
                GpuArticulatedDynamicsBatch::new(&context, &[angular_edge], 0.1).unwrap();
            angular_batch.submit_steps(1).unwrap();
            let angular_state = angular_batch.readback().unwrap();
            assert!(
                angular_state[0].velocities[2] > 0.015,
                "{:?}",
                angular_state[0]
            );

            let mut invalid_pair = input.clone();
            invalid_pair.box_pairs[0].second_half_extents.x = -0.5;
            assert!(GpuArticulatedDynamicsBatch::new(&context, &[invalid_pair], 0.1).is_err());
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn articulated_capsule_box_pair_detects_axis_side_contact() {
        let link = |mass| LinkSpec {
            mass,
            center_of_mass: Vector3::zeros(),
            inertia: Matrix3::identity() * mass,
        };
        let joint = |child| JointSpec {
            parent: 0,
            child,
            kind: JointKind::Prismatic,
            origin: Isometry3::identity(),
            axis: Vector3::x(),
            limits: None,
        };
        let articulation = Articulation::new(
            vec![link(0.0), link(1.0), link(1.0)],
            vec![joint(1), joint(2)],
            0,
        )
        .unwrap();
        let capsule = GpuArticulatedLinkCapsule {
            link: 1,
            local_a: Vector3::new(0.0, -1.0, 0.0),
            local_b: Vector3::new(0.0, 1.0, 0.0),
            radius: 0.5,
            restitution: 0.2,
            friction: 0.25,
        };
        let box_shape = GpuArticulatedLinkBox {
            link: 2,
            local_pose: Isometry3::translation(1.05, 0.0, 0.0),
            half_extents: Vector3::new(0.5, 0.25, 0.5),
            restitution: 0.7,
            friction: 1.0,
        };
        let generated =
            self_contact_capsule_box_pairs(&articulation, &[capsule], &[box_shape]).unwrap();
        assert_eq!(generated.len(), 1);
        assert_eq!(generated[0].restitution, 0.7);
        assert_eq!(generated[0].friction, 0.5);
        let mut excluded = articulation.clone();
        excluded.exclude_collision_pair(1, 2).unwrap();
        assert!(
            self_contact_capsule_box_pairs(&excluded, &[capsule], &[box_shape])
                .unwrap()
                .is_empty()
        );
        assert!(
            self_contact_capsule_box_pairs(
                &articulation,
                &[capsule],
                &[GpuArticulatedLinkBox {
                    link: 1,
                    ..box_shape
                }],
            )
            .unwrap()
            .is_empty()
        );
        let mut invalid = box_shape;
        invalid.half_extents.y = 0.0;
        assert!(self_contact_capsule_box_pairs(&articulation, &[capsule], &[invalid]).is_err());
        let input = GpuArticulatedDynamicsInput {
            articulation: &articulation,
            root_pose: Isometry3::identity(),
            state: GpuGeneralizedState {
                positions: DVector::zeros(2),
                velocities: DVector::from_column_slice(&[1.0, -1.0]),
            },
            gravity: Vector3::zeros(),
            joints: vec![GpuJointForceInput::default(); 2],
            joint_velocity_limit: None,
            coordinate_velocity_limits: None,
            link_loads: vec![GpuMassLinkLoad::default(); 3],
            ground_spheres: Vec::new(),
            ground_capsules: Vec::new(),
            ground_boxes: Vec::new(),
            ground_axial_shapes: Vec::new(),
            sphere_pairs: Vec::new(),
            static_sphere_pairs: Vec::new(),
            external_sphere_bodies: None,
            external_box_bodies: None,
            external_capsule_bodies: None,
            external_axial_bodies: None,
            external_convex_bodies: None,
            external_constraint_bodies: None,
            external_indexed_bodies: None,
            static_capsule_sphere_pairs: Vec::new(),
            static_box_sphere_pairs: Vec::new(),
            static_sphere_capsule_pairs: Vec::new(),
            static_capsule_pairs: Vec::new(),
            static_sphere_box_pairs: Vec::new(),
            static_capsule_box_pairs: Vec::new(),
            static_box_pairs: Vec::new(),
            static_box_capsule_pairs: Vec::new(),
            static_axial_sphere_pairs: Vec::new(),
            static_axial_capsule_pairs: Vec::new(),
            static_axial_box_pairs: Vec::new(),
            static_axial_convex_pairs: Vec::new(),
            axial_convex_pairs: Vec::new(),
            axial_sphere_pairs: Vec::new(),
            axial_box_pairs: Vec::new(),
            axial_capsule_pairs: Vec::new(),
            axial_pairs: Vec::new(),
            convex_sphere_pairs: Vec::new(),
            static_convex_sphere_pairs: Vec::new(),
            static_convex_capsule_pairs: Vec::new(),
            convex_capsule_pairs: Vec::new(),
            convex_pairs: Vec::new(),
            static_convex_pairs: Vec::new(),
            scene_convex_sphere_pairs: Vec::new(),
            scene_mesh_sphere_pairs: Vec::new(),
            scene_polyline_sphere_pairs: Vec::new(),
            scene_mesh_capsule_pairs: Vec::new(),
            scene_polyline_capsule_pairs: Vec::new(),
            scene_mesh_box_pairs: Vec::new(),
            scene_polyline_box_pairs: Vec::new(),
            scene_mesh_axial_pairs: Vec::new(),
            scene_polyline_axial_pairs: Vec::new(),
            scene_mesh_convex_pairs: Vec::new(),
            scene_polyline_convex_pairs: Vec::new(),
            scene_convex_capsule_pairs: Vec::new(),
            capsule_sphere_pairs: Vec::new(),
            capsule_pairs: Vec::new(),
            sphere_box_pairs: Vec::new(),
            capsule_box_pairs: vec![GpuArticulatedCapsuleBoxPair {
                restitution: 0.0,
                friction: 0.0,
                ..generated[0]
            }],
            link_spheres: Vec::new(),
            link_capsules: Vec::new(),
            link_boxes: Vec::new(),
            box_pairs: Vec::new(),
            contact_iterations: 8,
            dynamic_material_rules: Vec::new(),
            ground_manifold_start: None,
            joint_friction: Vec::new(),
            joint_couplings: Vec::new(),
            link_point_constraints: Vec::new(),
            link_fixed_constraints: Vec::new(),
            contact_warm_start: false,
        };
        let expected = DVector::from_column_slice(&[0.25, -0.25]);
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let batch =
                GpuArticulatedDynamicsBatch::new(&context, core::slice::from_ref(&input), 0.1)
                    .unwrap();
            batch.submit_steps(1).unwrap();
            let state = batch.readback().unwrap();
            assert!(
                (&state[0].velocities - &expected).norm() < 2e-4,
                "{:?}",
                state[0]
            );

            let mut automatic = input.clone();
            automatic.capsule_box_pairs.clear();
            automatic.link_capsules.push(GpuArticulatedLinkCapsule {
                restitution: 0.0,
                friction: 0.0,
                ..capsule
            });
            automatic.link_boxes.push(GpuArticulatedLinkBox {
                restitution: 0.0,
                friction: 0.0,
                ..box_shape
            });
            let automatic_batch =
                GpuArticulatedDynamicsBatch::new(&context, &[automatic.clone()], 0.1).unwrap();
            automatic_batch.submit_steps(1).unwrap();
            let automatic_state = automatic_batch.readback().unwrap();
            assert!(
                (&automatic_state[0].velocities - &expected).norm() < 2e-4,
                "{:?}",
                automatic_state[0]
            );

            let rows = automatic_batch
                .ground_contact
                .as_ref()
                .unwrap()
                .readback_dynamic_row_state(context.queue());
            assert_eq!(rows.len(), 2);
            assert!(rows[0].0 > 0.0 && rows[0].1[0] > 0.0, "{rows:?}");
            let mut mixed_shapes = automatic.clone();
            mixed_shapes.link_spheres.push(GpuArticulatedLinkSphere {
                link: capsule.link,
                local_center: Vector3::new(10.0, 0.0, 0.0),
                radius: 0.2,
                restitution: 0.0,
                friction: 0.0,
            });
            let mixed_batch =
                GpuArticulatedDynamicsBatch::new(&context, &[mixed_shapes], 0.1).unwrap();
            mixed_batch.submit_steps(1).unwrap();
            let rows = mixed_batch
                .ground_contact
                .as_ref()
                .unwrap()
                .readback_dynamic_row_state(context.queue());
            assert_eq!(rows.len(), 3);
            assert_eq!(rows[0], (0.0, [0.0; 4]));
            assert!(rows[1].0 > 0.0 && rows[1].1[0] > 0.0, "{rows:?}");
            let mut separated_dynamic = automatic.clone();
            separated_dynamic.link_boxes[0].local_pose = Isometry3::translation(3.0, 0.0, 0.0);
            let separated_dynamic_batch =
                GpuArticulatedDynamicsBatch::new(&context, &[separated_dynamic], 0.1).unwrap();
            separated_dynamic_batch.submit_steps(1).unwrap();
            let rows = separated_dynamic_batch
                .ground_contact
                .as_ref()
                .unwrap()
                .readback_dynamic_row_state(context.queue());
            assert_eq!(rows, vec![(0.0, [0.0; 4]); 2]);
            let mut outward = automatic.joints.clone();
            outward[0].base_force = -20.0;
            outward[1].base_force = 20.0;
            automatic_batch.update_joints(&[outward]).unwrap();
            automatic_batch.submit_steps(3).unwrap();
            let rows = automatic_batch
                .ground_contact
                .as_ref()
                .unwrap()
                .readback_dynamic_row_state(context.queue());
            assert_eq!(rows, vec![(0.0, [0.0; 4]); 2]);

            automatic.capsule_box_pairs = input.capsule_box_pairs.clone();
            automatic.link_capsules[0].restitution = 1.0;
            automatic.link_capsules[0].local_a = capsule.local_b;
            automatic.link_capsules[0].local_b = capsule.local_a;
            automatic.link_boxes[0].restitution = 1.0;
            let prioritized =
                GpuArticulatedDynamicsBatch::new(&context, &[automatic], 0.1).unwrap();
            prioritized.submit_steps(1).unwrap();
            let priority_state = prioritized.readback().unwrap();
            assert!(
                (&priority_state[0].velocities - &expected).norm() < 2e-4,
                "{:?}",
                priority_state[0]
            );
            let rows = prioritized
                .ground_contact
                .as_ref()
                .unwrap()
                .readback_dynamic_row_state(context.queue());
            assert!(rows.is_empty());

            let mut separated = input.clone();
            separated.capsule_box_pairs[0].box_local_pose = Isometry3::translation(3.0, 0.0, 0.0);
            let separated_batch =
                GpuArticulatedDynamicsBatch::new(&context, &[separated], 0.1).unwrap();
            separated_batch.submit_steps(1).unwrap();
            let separated_state = separated_batch.readback().unwrap();
            assert!(
                (&separated_state[0].velocities - &input.state.velocities).norm() < 2e-4,
                "{:?}",
                separated_state[0]
            );

            let mut inside = input.clone();
            inside.state.velocities.fill(0.0);
            inside.capsule_box_pairs[0].capsule_local_a.x = 0.25;
            inside.capsule_box_pairs[0].capsule_local_b.x = 0.25;
            inside.capsule_box_pairs[0].box_local_pose = Isometry3::identity();
            inside.capsule_box_pairs[0].box_half_extents.y = 0.5;
            let inside_batch = GpuArticulatedDynamicsBatch::new(&context, &[inside], 0.1).unwrap();
            inside_batch.submit_steps(1).unwrap();
            let inside_state = inside_batch.readback().unwrap();
            assert!(inside_state[0].velocities[0] > 0.0, "{:?}", inside_state[0]);
            assert!(inside_state[0].velocities[1] < 0.0, "{:?}", inside_state[0]);

            let mut collapsed = input.clone();
            collapsed.capsule_box_pairs[0].capsule_local_a = Vector3::zeros();
            collapsed.capsule_box_pairs[0].capsule_local_b = Vector3::zeros();
            let collapsed_batch =
                GpuArticulatedDynamicsBatch::new(&context, &[collapsed], 0.1).unwrap();
            collapsed_batch.submit_steps(1).unwrap();
            let collapsed_state = collapsed_batch.readback().unwrap();
            assert!(
                (&collapsed_state[0].velocities - &expected).norm() < 2e-4,
                "{:?}",
                collapsed_state[0]
            );

            let mut rotated = input.clone();
            rotated.capsule_box_pairs[0].box_local_pose =
                Isometry3::new(Vector3::new(1.05, 0.0, 0.0), Vector3::new(0.0, 0.0, 0.2));
            let rotated_batch =
                GpuArticulatedDynamicsBatch::new(&context, &[rotated], 0.1).unwrap();
            rotated_batch.submit_steps(1).unwrap();
            let rotated_state = rotated_batch.readback().unwrap();
            assert!(
                rotated_state[0].velocities[0] < 1.0,
                "{:?}",
                rotated_state[0]
            );
            assert!(
                rotated_state[0].velocities[1] > -1.0,
                "{:?}",
                rotated_state[0]
            );

            let mut invalid_pair = input.clone();
            invalid_pair.capsule_box_pairs[0].capsule_radius = -0.5;
            assert!(GpuArticulatedDynamicsBatch::new(&context, &[invalid_pair], 0.1).is_err());
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn parallel_capsule_box_face_uses_two_angularly_distinct_contacts() {
        let link = |mass| LinkSpec {
            mass,
            center_of_mass: Vector3::zeros(),
            inertia: Matrix3::identity() * mass,
        };
        let articulation = Articulation::new(
            vec![link(0.0), link(1.0), link(1.0)],
            vec![
                JointSpec {
                    parent: 0,
                    child: 1,
                    kind: JointKind::Revolute,
                    origin: Isometry3::identity(),
                    axis: Vector3::z(),
                    limits: None,
                },
                JointSpec {
                    parent: 0,
                    child: 2,
                    kind: JointKind::Prismatic,
                    origin: Isometry3::identity(),
                    axis: Vector3::x(),
                    limits: None,
                },
            ],
            0,
        )
        .unwrap();
        let input = GpuArticulatedDynamicsInput {
            articulation: &articulation,
            root_pose: Isometry3::identity(),
            state: GpuGeneralizedState {
                positions: DVector::zeros(2),
                velocities: DVector::from_column_slice(&[0.0, -1.0]),
            },
            gravity: Vector3::zeros(),
            joints: vec![GpuJointForceInput::default(); 2],
            joint_velocity_limit: None,
            coordinate_velocity_limits: None,
            link_loads: vec![GpuMassLinkLoad::default(); 3],
            ground_spheres: Vec::new(),
            ground_capsules: Vec::new(),
            ground_boxes: Vec::new(),
            ground_axial_shapes: Vec::new(),
            sphere_pairs: Vec::new(),
            static_sphere_pairs: Vec::new(),
            external_sphere_bodies: None,
            external_box_bodies: None,
            external_capsule_bodies: None,
            external_axial_bodies: None,
            external_convex_bodies: None,
            external_constraint_bodies: None,
            external_indexed_bodies: None,
            static_capsule_sphere_pairs: Vec::new(),
            static_box_sphere_pairs: Vec::new(),
            static_sphere_capsule_pairs: Vec::new(),
            static_capsule_pairs: Vec::new(),
            static_sphere_box_pairs: Vec::new(),
            static_capsule_box_pairs: Vec::new(),
            static_box_pairs: Vec::new(),
            static_box_capsule_pairs: Vec::new(),
            static_axial_sphere_pairs: Vec::new(),
            static_axial_capsule_pairs: Vec::new(),
            static_axial_box_pairs: Vec::new(),
            static_axial_convex_pairs: Vec::new(),
            axial_convex_pairs: Vec::new(),
            axial_sphere_pairs: Vec::new(),
            axial_box_pairs: Vec::new(),
            axial_capsule_pairs: Vec::new(),
            axial_pairs: Vec::new(),
            convex_sphere_pairs: Vec::new(),
            static_convex_sphere_pairs: Vec::new(),
            static_convex_capsule_pairs: Vec::new(),
            convex_capsule_pairs: Vec::new(),
            convex_pairs: Vec::new(),
            static_convex_pairs: Vec::new(),
            scene_convex_sphere_pairs: Vec::new(),
            scene_mesh_sphere_pairs: Vec::new(),
            scene_polyline_sphere_pairs: Vec::new(),
            scene_mesh_capsule_pairs: Vec::new(),
            scene_polyline_capsule_pairs: Vec::new(),
            scene_mesh_box_pairs: Vec::new(),
            scene_polyline_box_pairs: Vec::new(),
            scene_mesh_axial_pairs: Vec::new(),
            scene_polyline_axial_pairs: Vec::new(),
            scene_mesh_convex_pairs: Vec::new(),
            scene_polyline_convex_pairs: Vec::new(),
            scene_convex_capsule_pairs: Vec::new(),
            capsule_sphere_pairs: Vec::new(),
            capsule_pairs: Vec::new(),
            sphere_box_pairs: Vec::new(),
            capsule_box_pairs: Vec::new(),
            link_spheres: Vec::new(),
            link_capsules: vec![GpuArticulatedLinkCapsule {
                link: 1,
                local_a: Vector3::new(0.0, -1.0, 0.0),
                local_b: Vector3::new(0.0, 1.0, 0.0),
                radius: 0.5,
                restitution: 0.0,
                friction: 0.0,
            }],
            link_boxes: vec![GpuArticulatedLinkBox {
                link: 2,
                local_pose: Isometry3::translation(1.05, 0.0, 0.0),
                half_extents: Vector3::new(0.5, 0.5, 0.5),
                restitution: 0.0,
                friction: 0.0,
            }],
            box_pairs: Vec::new(),
            contact_iterations: 16,
            dynamic_material_rules: Vec::new(),
            ground_manifold_start: None,
            joint_friction: Vec::new(),
            joint_couplings: Vec::new(),
            link_point_constraints: Vec::new(),
            link_fixed_constraints: Vec::new(),
            contact_warm_start: false,
        };
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let batch =
                GpuArticulatedDynamicsBatch::new(&context, core::slice::from_ref(&input), 0.1)
                    .unwrap();
            batch.submit_steps(1).unwrap();
            let rows = batch
                .ground_contact
                .as_ref()
                .unwrap()
                .readback_dynamic_row_state(context.queue());
            assert_eq!(rows.len(), 2);
            assert!(rows[0].1[0] > 0.0 && rows[1].1[0] > 0.0, "{rows:?}");

            let mut explicit = input.clone();
            explicit
                .capsule_box_pairs
                .push(GpuArticulatedCapsuleBoxPair {
                    capsule_link: 1,
                    capsule_local_a: Vector3::new(0.0, -1.0, 0.0),
                    capsule_local_b: Vector3::new(0.0, 1.0, 0.0),
                    capsule_radius: 0.5,
                    box_link: 2,
                    box_local_pose: Isometry3::translation(1.05, 0.0, 0.0),
                    box_half_extents: Vector3::new(0.5, 0.5, 0.5),
                    restitution: 0.0,
                    friction: 0.0,
                });
            let explicit_batch =
                GpuArticulatedDynamicsBatch::new(&context, &[explicit], 0.1).unwrap();
            explicit_batch.submit_steps(1).unwrap();
            let excluded_rows = explicit_batch
                .ground_contact
                .as_ref()
                .unwrap()
                .readback_dynamic_row_state(context.queue());
            assert!(excluded_rows.is_empty());
            let dynamic_state = batch.readback().unwrap();
            let explicit_state = explicit_batch.readback().unwrap();
            assert!(
                (&dynamic_state[0].velocities - &explicit_state[0].velocities).norm() < 1e-4,
                "dynamic={:?} explicit={:?}",
                dynamic_state[0],
                explicit_state[0]
            );
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn articulated_sphere_box_pair_uses_box_angular_jacobian() {
        let link = |mass| LinkSpec {
            mass,
            center_of_mass: Vector3::zeros(),
            inertia: Matrix3::identity() * mass,
        };
        let articulation = Articulation::new(
            vec![link(0.0), link(1.0), link(0.0), link(1.0)],
            vec![
                JointSpec {
                    parent: 0,
                    child: 1,
                    kind: JointKind::Prismatic,
                    origin: Isometry3::identity(),
                    axis: Vector3::x(),
                    limits: None,
                },
                JointSpec {
                    parent: 0,
                    child: 2,
                    kind: JointKind::Prismatic,
                    origin: Isometry3::identity(),
                    axis: Vector3::x(),
                    limits: None,
                },
                JointSpec {
                    parent: 2,
                    child: 3,
                    kind: JointKind::Revolute,
                    origin: Isometry3::identity(),
                    axis: Vector3::z(),
                    limits: None,
                },
            ],
            0,
        )
        .unwrap();
        let input = GpuArticulatedDynamicsInput {
            articulation: &articulation,
            root_pose: Isometry3::identity(),
            state: GpuGeneralizedState {
                positions: DVector::zeros(3),
                velocities: DVector::from_column_slice(&[1.0, -1.0, 0.0]),
            },
            gravity: Vector3::zeros(),
            joints: vec![GpuJointForceInput::default(); 3],
            joint_velocity_limit: None,
            coordinate_velocity_limits: None,
            link_loads: vec![GpuMassLinkLoad::default(); 4],
            ground_spheres: Vec::new(),
            ground_capsules: Vec::new(),
            ground_boxes: Vec::new(),
            ground_axial_shapes: Vec::new(),
            sphere_pairs: Vec::new(),
            static_sphere_pairs: Vec::new(),
            external_sphere_bodies: None,
            external_box_bodies: None,
            external_capsule_bodies: None,
            external_axial_bodies: None,
            external_convex_bodies: None,
            external_constraint_bodies: None,
            external_indexed_bodies: None,
            static_capsule_sphere_pairs: Vec::new(),
            static_box_sphere_pairs: Vec::new(),
            static_sphere_capsule_pairs: Vec::new(),
            static_capsule_pairs: Vec::new(),
            static_sphere_box_pairs: Vec::new(),
            static_capsule_box_pairs: Vec::new(),
            static_box_pairs: Vec::new(),
            static_box_capsule_pairs: Vec::new(),
            static_axial_sphere_pairs: Vec::new(),
            static_axial_capsule_pairs: Vec::new(),
            static_axial_box_pairs: Vec::new(),
            static_axial_convex_pairs: Vec::new(),
            axial_convex_pairs: Vec::new(),
            axial_sphere_pairs: Vec::new(),
            axial_box_pairs: Vec::new(),
            axial_capsule_pairs: Vec::new(),
            axial_pairs: Vec::new(),
            convex_sphere_pairs: Vec::new(),
            static_convex_sphere_pairs: Vec::new(),
            static_convex_capsule_pairs: Vec::new(),
            convex_capsule_pairs: Vec::new(),
            convex_pairs: Vec::new(),
            static_convex_pairs: Vec::new(),
            scene_convex_sphere_pairs: Vec::new(),
            scene_mesh_sphere_pairs: Vec::new(),
            scene_polyline_sphere_pairs: Vec::new(),
            scene_mesh_capsule_pairs: Vec::new(),
            scene_polyline_capsule_pairs: Vec::new(),
            scene_mesh_box_pairs: Vec::new(),
            scene_polyline_box_pairs: Vec::new(),
            scene_mesh_axial_pairs: Vec::new(),
            scene_polyline_axial_pairs: Vec::new(),
            scene_mesh_convex_pairs: Vec::new(),
            scene_polyline_convex_pairs: Vec::new(),
            scene_convex_capsule_pairs: Vec::new(),
            capsule_sphere_pairs: Vec::new(),
            capsule_pairs: Vec::new(),
            sphere_box_pairs: vec![GpuArticulatedSphereBoxPair {
                sphere_link: 1,
                sphere_local_center: Vector3::new(0.0, 0.25, 0.0),
                sphere_radius: 0.5,
                box_link: 3,
                box_local_pose: Isometry3::translation(1.05, 0.0, 0.0),
                box_half_extents: Vector3::repeat(0.5),
                restitution: 0.0,
                friction: 0.0,
            }],
            capsule_box_pairs: Vec::new(),
            link_spheres: Vec::new(),
            link_capsules: Vec::new(),
            link_boxes: Vec::new(),
            box_pairs: Vec::new(),
            contact_iterations: 8,
            dynamic_material_rules: Vec::new(),
            ground_manifold_start: None,
            joint_friction: Vec::new(),
            joint_couplings: Vec::new(),
            link_point_constraints: Vec::new(),
            link_fixed_constraints: Vec::new(),
            contact_warm_start: false,
        };
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let batch =
                GpuArticulatedDynamicsBatch::new(&context, core::slice::from_ref(&input), 0.1)
                    .unwrap();
            batch.submit_steps(1).unwrap();
            let state = batch.readback().unwrap();
            assert!(state[0].velocities[0] < 1.0, "{:?}", state[0]);
            assert!(state[0].velocities[1] > -1.0, "{:?}", state[0]);
            assert!(state[0].velocities[2] < -0.05, "{:?}", state[0]);
            let pair = input.sphere_box_pairs[0];
            let mut dynamic_input = input.clone();
            dynamic_input.sphere_box_pairs.clear();
            dynamic_input.link_spheres.push(GpuArticulatedLinkSphere {
                link: pair.sphere_link,
                local_center: pair.sphere_local_center,
                radius: pair.sphere_radius,
                restitution: pair.restitution,
                friction: pair.friction,
            });
            dynamic_input.link_boxes.push(GpuArticulatedLinkBox {
                link: pair.box_link,
                local_pose: pair.box_local_pose,
                half_extents: pair.box_half_extents,
                restitution: pair.restitution,
                friction: pair.friction,
            });
            let dynamic_batch =
                GpuArticulatedDynamicsBatch::new(&context, &[dynamic_input], 0.1).unwrap();
            dynamic_batch.submit_steps(1).unwrap();
            let dynamic_state = dynamic_batch.readback().unwrap();
            assert!((&dynamic_state[0].velocities - &state[0].velocities).norm() < 2e-4);
            let rows = dynamic_batch
                .ground_contact
                .as_ref()
                .unwrap()
                .readback_dynamic_row_state(context.queue());
            assert_eq!(rows.len(), 1);
            assert!(rows[0].0 > 0.0 && rows[0].1[0] > 0.0, "{rows:?}");

            let mut face_pair = input.clone();
            face_pair.sphere_box_pairs.clear();
            face_pair.box_pairs = vec![GpuArticulatedBoxPair {
                first_link: 1,
                first_local_pose: Isometry3::identity(),
                first_half_extents: Vector3::repeat(0.5),
                second_link: 3,
                second_local_pose: Isometry3::translation(1.05, 0.0, 0.0),
                second_half_extents: Vector3::repeat(0.5),
                restitution: 0.0,
                friction: 0.0,
            }];
            face_pair.contact_iterations = 32;
            let face_batch = GpuArticulatedDynamicsBatch::new(&context, &[face_pair], 0.1).unwrap();
            face_batch.submit_steps(1).unwrap();
            let face_state = face_batch.readback().unwrap();
            assert!(face_state[0].velocities[0] < 1.0, "{:?}", face_state[0]);
            assert!(face_state[0].velocities[1] > -1.0, "{:?}", face_state[0]);
            assert!(
                face_state[0].velocities[2].abs() < 0.05,
                "{:?}",
                face_state[0]
            );

            let mut capsule_input = input.clone();
            capsule_input.sphere_box_pairs.clear();
            capsule_input.capsule_box_pairs = vec![GpuArticulatedCapsuleBoxPair {
                capsule_link: 1,
                capsule_local_a: Vector3::new(0.0, 0.25, 0.0),
                capsule_local_b: Vector3::new(0.0, 0.75, 0.0),
                capsule_radius: 0.5,
                box_link: 3,
                box_local_pose: Isometry3::translation(1.05, 0.0, 0.0),
                box_half_extents: Vector3::repeat(0.5),
                restitution: 0.0,
                friction: 0.0,
            }];
            let capsule_batch =
                GpuArticulatedDynamicsBatch::new(&context, &[capsule_input], 0.1).unwrap();
            capsule_batch.submit_steps(1).unwrap();
            let capsule_state = capsule_batch.readback().unwrap();
            assert!(
                capsule_state[0].velocities[0] < 1.0,
                "{:?}",
                capsule_state[0]
            );
            assert!(
                capsule_state[0].velocities[1] > -1.0,
                "{:?}",
                capsule_state[0]
            );
            assert!(
                capsule_state[0].velocities[2] < -0.05,
                "{:?}",
                capsule_state[0]
            );
        }
        assert!(tested > 0, "no GPU backend available");
    }
}
