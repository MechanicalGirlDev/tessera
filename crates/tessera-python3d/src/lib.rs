//! UniFFI boundary for Tessera 3D physics.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use std::sync::{Mutex, MutexGuard};

use nalgebra::{
    Isometry3, Matrix3, Quaternion as NaQuaternion, Translation3, UnitQuaternion, Vector3,
};
use tessera_physics::articulated_world::{
    ArticulatedWorld as CoreArticulatedWorld, JointCoupling as CoreJointCoupling,
    JointMotor as CoreJointMotor, JointNonlinearPassive as CoreJointNonlinearPassive,
    JointPassive as CoreJointPassive, JointPolynomialCoupling as CoreJointPolynomialCoupling,
    LinkAcceleration as CoreLinkAcceleration, LinkExternalLoad as CoreLinkExternalLoad,
    LinkFixedConstraint as CoreLinkFixedConstraint, LinkImuReading as CoreLinkImuReading,
    LinkPointConstraint as CoreLinkPointConstraint,
    LinkSceneFixedConstraint as CoreLinkSceneFixedConstraint,
    LinkScenePointConstraint as CoreLinkScenePointConstraint, LinkTwist as CoreLinkTwist,
    SceneBody as CoreSceneBody, SceneCollider as CoreSceneCollider,
};
use tessera_physics::batch::{
    ArticulatedBatch as CoreArticulatedBatch, ArticulatedDeviceMassStepCache, EnvironmentId,
};
use tessera_physics::convex::ConvexGeometry;
use tessera_physics::gpu_articulated_mass_assembly::GpuArticulatedMassAssemblyBatch;
use tessera_physics::gpu_contact_pipeline::GpuContactDevice;
use tessera_physics::gpu_rigid_ball_joint::{
    GpuRigidAxisMotor, GpuRigidAxisServo, GpuRigidBallJoint, GpuRigidFixedJoint,
    GpuRigidPrismaticJoint, GpuRigidPrismaticLimit, GpuRigidRevoluteJoint, GpuRigidRevoluteLimit,
    GpuRigidTemporalJointParams,
};
use tessera_physics::gpu_rigid_shape::GpuRigidShape;
use tessera_physics::gpu_rigid_sphere_contact::{
    GpuRigidCollisionGroups, GpuRigidSphereContactReadback,
};
use tessera_physics::gpu_rigid_sphere_solver::GpuRigidTemporalSolveParams;
use tessera_physics::gpu_rigid_sphere_world::{
    GpuRigidEnvironmentSceneQueries, GpuRigidPrimitiveWorld as CoreGpuPrimitiveWorld,
    GpuRigidSphereBatch, GpuRigidSphereEnvironment, GpuRigidSphereWorld as CoreGpuSphereWorld,
    GpuRigidSphereWorldConfig,
};
use tessera_physics::gpu_rigid_state::{GpuRigidBodyForces, GpuRigidBodyState};
use tessera_physics::gpu_scene_dynamics::{GpuSceneDynamics, GpuSceneDynamicsBatch};
use tessera_physics::inverse_kinematics::{
    IkConfig as CoreIkConfig, IkState as CoreIkState, IkTarget as CoreIkTarget,
    forward_kinematics as core_forward_kinematics, inverse_kinematics as core_inverse_kinematics,
};
use tessera_physics::material::{CoefficientCombineRule, ColliderMaterial};
use tessera_physics::mesh::{HeightFieldGeometry, PolylineGeometry, TriangleMeshGeometry};
use tessera_physics::mjcf::{
    LoadedMjcf, MjcfActuatorKind as CoreMjcfActuatorKind, MjcfActuators, MjcfLoadOptions,
    load_mjcf_str, load_mjcf_str_with_mesh_resolver,
};
use tessera_physics::sphere_world::{
    SphereBody, SphereWorld as CoreSphereWorld, SphereWorldParams,
};
use tessera_physics::urdf::{
    LoadedUrdf, UrdfLoadOptions, load_urdf_str, load_urdf_str_with_mesh_resolver,
};

mod mpm;
pub use mpm::*;
mod primitive_batch;
pub use primitive_batch::GpuPrimitiveBatch;

uniffi::setup_scaffolding!();

/// Invalid input or a physics backend failure.
#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum TesseraError {
    /// An operation failed with a diagnostic.
    #[error("{reason}")]
    Failed {
        /// Diagnostic from the Rust physics API.
        reason: String,
    },
}

fn failed(error: impl core::fmt::Display) -> TesseraError {
    TesseraError::Failed {
        reason: error.to_string(),
    }
}

fn locked<T>(mutex: &Mutex<T>) -> Result<MutexGuard<'_, T>, TesseraError> {
    mutex
        .lock()
        .map_err(|_| failed("physics world lock poisoned"))
}

/// World-space XYZ vector.
#[derive(Clone, Copy, Debug, uniffi::Record)]
pub struct Vec3 {
    /// X component.
    pub x: f64,
    /// Y component.
    pub y: f64,
    /// Z component.
    pub z: f64,
}

impl Vec3 {
    fn nalgebra(self) -> Vector3<f64> {
        Vector3::new(self.x, self.y, self.z)
    }

    fn gpu(self) -> [f32; 3] {
        [self.x as f32, self.y as f32, self.z as f32]
    }
}

impl From<Vector3<f64>> for Vec3 {
    fn from(value: Vector3<f64>) -> Self {
        Self {
            x: value.x,
            y: value.y,
            z: value.z,
        }
    }
}

/// Soft temporal contact coefficients and initial speculative distance.
#[derive(Clone, Copy, Debug, uniffi::Record)]
pub struct GpuTemporalSettings {
    /// Default friction coefficient, overridden by collider materials.
    pub friction: f32,
    /// Default restitution coefficient, overridden by collider materials.
    pub restitution: f32,
    /// Normal spring frequency in hertz.
    pub normal_frequency: f32,
    /// Nonnegative normal damping ratio.
    pub damping_ratio: f32,
    /// Normal spring frequency for ground and fixed-body contacts.
    pub static_normal_frequency: f32,
    /// Normal damping ratio for ground and fixed-body contacts.
    pub static_damping_ratio: f32,
    /// Maximum penetration correction speed in metres per second.
    pub max_corrective_velocity: f32,
    /// Constraint iterations per substep phase.
    pub iterations: u32,
    /// Initial surface gap admitted as a speculative contact, in metres.
    pub speculative_margin: f32,
    /// Restrict speculative distance to ground; ordinary pair contacts remain active.
    pub ground_only: bool,
}

/// Return the engine defaults for a temporal frame without a speculative margin.
#[uniffi::export]
pub fn default_gpu_temporal_settings() -> GpuTemporalSettings {
    let settings = GpuRigidTemporalSolveParams::default();
    GpuTemporalSettings {
        friction: settings.friction,
        restitution: settings.restitution,
        normal_frequency: settings.normal_frequency,
        damping_ratio: settings.damping_ratio,
        static_normal_frequency: settings.static_normal_frequency,
        static_damping_ratio: settings.static_damping_ratio,
        max_corrective_velocity: settings.max_corrective_velocity,
        iterations: settings.iterations,
        speculative_margin: 0.0,
        ground_only: false,
    }
}

impl From<GpuTemporalSettings> for GpuRigidTemporalSolveParams {
    fn from(settings: GpuTemporalSettings) -> Self {
        Self {
            friction: settings.friction,
            restitution: settings.restitution,
            normal_frequency: settings.normal_frequency,
            damping_ratio: settings.damping_ratio,
            static_normal_frequency: settings.static_normal_frequency,
            static_damping_ratio: settings.static_damping_ratio,
            max_corrective_velocity: settings.max_corrective_velocity,
            iterations: settings.iterations,
        }
    }
}

/// Independent temporal coefficients for passive joints and joint limits.
#[derive(Clone, Copy, Debug, uniffi::Record)]
pub struct GpuTemporalJointSettings {
    /// Joint spring frequency in hertz.
    pub frequency: f32,
    /// Nonnegative joint damping ratio.
    pub damping_ratio: f32,
    /// Maximum linear correction speed in metres per second.
    pub max_linear_correction_speed: f32,
    /// Maximum angular correction speed in radians per second.
    pub max_angular_correction_speed: f32,
    /// Joint constraint sweeps per substep phase.
    pub iterations: u32,
}

/// Return the engine defaults for independent temporal joint coefficients.
#[uniffi::export]
pub fn default_gpu_temporal_joint_settings() -> GpuTemporalJointSettings {
    let settings = GpuRigidTemporalJointParams::default();
    GpuTemporalJointSettings {
        frequency: settings.frequency,
        damping_ratio: settings.damping_ratio,
        max_linear_correction_speed: settings.max_linear_correction_speed,
        max_angular_correction_speed: settings.max_angular_correction_speed,
        iterations: settings.iterations,
    }
}

impl From<GpuTemporalJointSettings> for GpuRigidTemporalJointParams {
    fn from(value: GpuTemporalJointSettings) -> Self {
        Self {
            frequency: value.frequency,
            damping_ratio: value.damping_ratio,
            max_linear_correction_speed: value.max_linear_correction_speed,
            max_angular_correction_speed: value.max_angular_correction_speed,
            iterations: value.iterations,
        }
    }
}

fn temporal_step_binding(
    world: &mut CoreGpuSphereWorld,
    frame_dt: f32,
    substeps: u32,
    settings: Option<GpuTemporalSettings>,
    joint_settings: Option<GpuTemporalJointSettings>,
) -> Result<u32, TesseraError> {
    let settings = settings.unwrap_or_else(default_gpu_temporal_settings);
    let solve = settings.into();
    let count = if let Some(joints) = joint_settings {
        if settings.ground_only {
            world.step_temporal_with_joints_ground(
                frame_dt,
                substeps,
                solve,
                joints.into(),
                settings.speculative_margin,
            )
        } else {
            world.step_temporal_with_joints(
                frame_dt,
                substeps,
                solve,
                joints.into(),
                settings.speculative_margin,
            )
        }
    } else if settings.ground_only {
        world.step_temporal_speculative_ground(
            frame_dt,
            substeps,
            solve,
            settings.speculative_margin,
        )
    } else {
        world.step_temporal_speculative(frame_dt, substeps, solve, settings.speculative_margin)
    }
    .map_err(failed)?;
    u32::try_from(count).map_err(failed)
}

fn temporal_step_speed_bounded_binding(
    world: &mut CoreGpuSphereWorld,
    frame_dt: f32,
    substeps: u32,
    settings: Option<GpuTemporalSettings>,
    joint_settings: Option<GpuTemporalJointSettings>,
) -> Result<u32, TesseraError> {
    let mut settings = settings.unwrap_or_else(default_gpu_temporal_settings);
    if settings.ground_only
        || !settings.speculative_margin.is_finite()
        || settings.speculative_margin < 0.0
    {
        return Err(failed(
            "speed-bounded speculative contacts require pair mode and a valid margin",
        ));
    }
    let margin = world
        .speed_bounded_speculative_margin(frame_dt)
        .map_err(failed)?;
    settings.speculative_margin = settings.speculative_margin.max(margin);
    temporal_step_binding(world, frame_dt, substeps, Some(settings), joint_settings)
}

/// Half-open environment-local body range used by ray queries.
#[derive(Clone, Copy, Debug, uniffi::Record)]
pub struct GpuRayBodyRange {
    /// Inclusive first body index.
    pub start: u32,
    /// Exclusive last body index.
    pub end: u32,
}

/// Ray parameterization and filters for resident GPU geometry.
#[derive(Clone, Debug, uniffi::Record)]
pub struct GpuRay {
    /// Finite world origin.
    pub origin: Vec3,
    /// Finite nonzero direction, not necessarily normalized.
    pub direction: Vec3,
    /// Inclusive maximum ray parameter.
    pub max_t: f32,
    /// Bilateral collider group filter.
    pub groups: GpuCollisionGroups,
    /// Optional body to exclude, local to the queried environment.
    pub excluded_body: Option<u32>,
    /// Optional half-open range, local to the queried environment.
    pub body_range: Option<GpuRayBodyRange>,
    /// Return t=0 if the origin is inside a solid shape.
    pub solid: bool,
}

/// Nearest ray intersection; a miss is returned as None.
#[derive(Clone, Debug, uniffi::Record)]
pub struct GpuRayHit {
    /// Body index local to the world or environment, None for ground.
    pub body: Option<u32>,
    /// Triangle or segment ID for thin surfaces, zero for volume primitives.
    pub feature: u32,
    /// World intersection point.
    pub point: Vec3,
    /// World normal; zero for inside-solid hits and line segments.
    pub normal: Vec3,
    /// Parameter t in origin + direction * t.
    pub toi: f32,
    /// Whether the origin was inside a volume and solid=true.
    pub inside_solid: bool,
}

fn gpu_rays(rays: Vec<GpuRay>) -> Vec<tessera_physics::gpu_ray_query::GpuRigidRay> {
    rays.into_iter()
        .map(|ray| tessera_physics::gpu_ray_query::GpuRigidRay {
            origin: ray.origin.gpu(),
            direction: ray.direction.gpu(),
            max_t: ray.max_t,
            groups: ray.groups.into(),
            excluded_body: ray.excluded_body,
            body_range: ray.body_range.map(|range| [range.start, range.end]),
            solid: ray.solid,
        })
        .collect()
}
fn gpu_ray_hits(
    hits: Vec<tessera_physics::gpu_ray_query::GpuRigidRayHit>,
) -> Vec<Option<GpuRayHit>> {
    hits.into_iter()
        .map(|hit| {
            (hit.ids[2] != 0).then(|| GpuRayHit {
                body: (hit.ids[0] != u32::MAX).then_some(hit.ids[0]),
                feature: hit.ids[1],
                point: Vec3 {
                    x: f64::from(hit.point_toi[0]),
                    y: f64::from(hit.point_toi[1]),
                    z: f64::from(hit.point_toi[2]),
                },
                normal: Vec3 {
                    x: f64::from(hit.normal[0]),
                    y: f64::from(hit.normal[1]),
                    z: f64::from(hit.normal[2]),
                },
                toi: hit.point_toi[3],
                inside_solid: hit.ids[3] != 0,
            })
        })
        .collect()
}

/// Point projection and filters for resident GPU geometry.
#[derive(Clone, Debug, uniffi::Record)]
pub struct GpuPointQuery {
    /// Finite world-space point.
    pub point: Vec3,
    /// Finite inclusive maximum distance.
    pub max_distance: f32,
    /// Bilateral collider group filter.
    pub groups: GpuCollisionGroups,
    /// Optional body to exclude, local to the queried environment.
    pub excluded_body: Option<u32>,
    /// Optional half-open range, local to the queried environment.
    pub body_range: Option<GpuRayBodyRange>,
    /// Return the input point if inside a volume.
    pub solid: bool,
}

/// Nearest point projection; a miss is returned as None.
#[derive(Clone, Debug, uniffi::Record)]
pub struct GpuPointHit {
    /// Body index local to the world or environment, None for ground.
    pub body: Option<u32>,
    /// Triangle or segment ID for thin surfaces, zero for volume primitives.
    pub feature: u32,
    /// World-space projected point.
    pub point: Vec3,
    /// Unit direction from the boundary toward an exterior query, or outward
    /// from an interior query toward its nearest boundary. Zero at coincidence.
    pub normal: Vec3,
    /// Unsigned distance to the projected point.
    pub distance: f32,
    /// Whether the point belongs to a volume, regardless of solid mode.
    pub is_inside: bool,
}

/// Ray and point results evaluated against one resident scene snapshot.
#[derive(Clone, Debug, uniffi::Record)]
pub struct GpuSceneQueryHits {
    /// Nearest ray intersections in input order.
    pub rays: Vec<Option<GpuRayHit>>,
    /// Nearest point projections in input order.
    pub points: Vec<Option<GpuPointHit>>,
}

fn gpu_scene_query_hits(
    hits: (
        Vec<tessera_physics::gpu_ray_query::GpuRigidRayHit>,
        Vec<tessera_physics::gpu_point_query::GpuRigidPointHit>,
    ),
) -> GpuSceneQueryHits {
    GpuSceneQueryHits {
        rays: gpu_ray_hits(hits.0),
        points: gpu_point_hits(hits.1),
    }
}

fn gpu_scene_query_hits_environments(
    batch: &GpuRigidSphereBatch,
    rays: Vec<Vec<GpuRay>>,
    points: Vec<Vec<GpuPointQuery>>,
) -> Result<Vec<GpuSceneQueryHits>, TesseraError> {
    if rays.len() != batch.len() || points.len() != batch.len() {
        return Err(failed(
            "one ray and point query list is required per environment",
        ));
    }
    let rays = rays.into_iter().map(gpu_rays).collect::<Vec<_>>();
    let points = points.into_iter().map(gpu_points).collect::<Vec<_>>();
    let queries = rays
        .iter()
        .zip(&points)
        .map(|(rays, points)| GpuRigidEnvironmentSceneQueries { rays, points })
        .collect::<Vec<_>>();
    Ok(batch
        .query_scene_environments(&queries)
        .map_err(failed)?
        .into_iter()
        .map(|hits| gpu_scene_query_hits((hits.rays, hits.points)))
        .collect())
}

fn gpu_points(rays: Vec<GpuPointQuery>) -> Vec<tessera_physics::gpu_point_query::GpuRigidPoint> {
    rays.into_iter()
        .map(|ray| tessera_physics::gpu_point_query::GpuRigidPoint {
            point: ray.point.gpu(),
            max_distance: ray.max_distance,
            groups: ray.groups.into(),
            excluded_body: ray.excluded_body,
            body_range: ray.body_range.map(|range| [range.start, range.end]),
            solid: ray.solid,
        })
        .collect()
}
fn gpu_point_hits(
    hits: Vec<tessera_physics::gpu_point_query::GpuRigidPointHit>,
) -> Vec<Option<GpuPointHit>> {
    hits.into_iter()
        .map(|hit| {
            (hit.ids[2] != 0).then(|| GpuPointHit {
                body: (hit.ids[0] != u32::MAX).then_some(hit.ids[0]),
                feature: hit.ids[1],
                point: Vec3 {
                    x: f64::from(hit.point_distance[0]),
                    y: f64::from(hit.point_distance[1]),
                    z: f64::from(hit.point_distance[2]),
                },
                normal: Vec3 {
                    x: f64::from(hit.normal[0]),
                    y: f64::from(hit.normal[1]),
                    z: f64::from(hit.normal[2]),
                },
                distance: hit.point_distance[3],
                is_inside: hit.ids[3] != 0,
            })
        })
        .collect()
}

/// Sphere input; mass zero makes the body static.
#[derive(Clone, Debug, uniffi::Record)]
pub struct SphereInput {
    /// Initial world centre.
    pub center: Vec3,
    /// Initial linear velocity.
    pub velocity: Vec3,
    /// Positive radius.
    pub radius: f64,
    /// Nonnegative mass.
    pub mass: f64,
}

/// Observed CPU sphere state.
#[derive(Clone, Debug, uniffi::Record)]
pub struct SphereState {
    /// World centre.
    pub center: Vec3,
    /// Linear velocity.
    pub velocity: Vec3,
    /// Radius.
    pub radius: f64,
    /// Mass.
    pub mass: f64,
}

/// Observed GPU sphere state.
#[derive(Clone, Debug, uniffi::Record)]
pub struct GpuSphereState {
    /// World centre.
    pub center: Vec3,
    /// Orientation in XYZW order.
    pub orientation: Quaternion,
    /// Linear velocity.
    pub velocity: Vec3,
    /// Angular velocity.
    pub angular_velocity: Vec3,
    /// Whether automatic sleep is active.
    pub sleeping: bool,
}

/// Observed GPU state of a primitive rigid body.
#[derive(Clone, Debug, uniffi::Record)]
pub struct GpuPrimitiveState {
    /// World-space body origin.
    pub center: Vec3,
    /// Orientation in XYZW order.
    pub orientation: Quaternion,
    /// World-space linear velocity.
    pub velocity: Vec3,
    /// World-space angular velocity.
    pub angular_velocity: Vec3,
    /// Whether automatic sleep is active.
    pub sleeping: bool,
}

/// Three vertex indices of a GPU-resident triangle.
#[derive(Clone, Copy, Debug, uniffi::Record)]
pub struct GpuTriangle {
    /// First vertex index.
    pub a: u32,
    /// Second vertex index.
    pub b: u32,
    /// Third vertex index.
    pub c: u32,
}

/// Two vertex indices of a GPU-resident line segment.
#[derive(Clone, Copy, Debug, uniffi::Record)]
pub struct GpuSegment {
    /// First endpoint index.
    pub a: u32,
    /// Second endpoint index.
    pub b: u32,
}

/// One GPU-resident primitive collider attached to a body origin.
#[derive(Clone, Debug, uniffi::Enum)]
pub enum GpuPrimitiveShape {
    /// Sphere with a positive radius.
    Sphere {
        /// Collision radius in metres.
        radius: f64,
    },
    /// Oriented box with positive body-frame half extents.
    Box {
        /// Half extents along the body-frame XYZ axes.
        half_extents: Vec3,
    },
    /// Z-axis capsule with hemispheres at plus and minus half length.
    Capsule {
        /// Rounded-end radius in metres.
        radius: f64,
        /// Distance from the body origin to either hemisphere center.
        half_length: f64,
    },
    /// Z-axis cylinder with flat end caps.
    Cylinder {
        /// Circular cross-section radius in metres.
        radius: f64,
        /// Distance from the body origin to either cap.
        half_length: f64,
    },
    /// Z-axis cone with its apex at positive half length.
    Cone {
        /// Flat-base radius in metres.
        radius: f64,
        /// Distance from the body origin to the apex or base plane.
        half_length: f64,
    },
    /// Three-dimensional convex hull from body-frame vertices.
    Convex {
        /// Vertices spanning a nondegenerate hull in body coordinates.
        vertices: Vec<Vec3>,
    },
    /// Centered z-up heightfield, triangulated identically to the CPU geometry.
    Heightfield {
        /// Number of rows along the local y axis, at least two.
        rows: u32,
        /// Number of columns along the local x axis, at least two.
        columns: u32,
        /// Row-major height samples before vertical scaling.
        heights: Vec<f64>,
        /// Full x/y extents and vertical height multiplier, all positive.
        scale: Vec3,
    },
    /// Zero-thickness segments supporting sphere, capsule, box, convex hull, and ground contacts.
    Polyline {
        /// Finite body-local vertices.
        vertices: Vec<Vec3>,
        /// Nondegenerate indexed segments.
        segments: Vec<GpuSegment>,
    },
    /// Two-sided triangle surface in body coordinates.
    TriangleMesh {
        /// Finite body-local vertices.
        vertices: Vec<Vec3>,
        /// Nondegenerate indexed triangles.
        triangles: Vec<GpuTriangle>,
    },
}

/// Prescribed world-space motion for a zero-mass GPU body.
#[derive(Clone, Debug, uniffi::Record)]
pub struct GpuKinematicMotion {
    /// Prescribed linear velocity in metres per second.
    pub linear: Vec3,
    /// Prescribed angular velocity in radians per second.
    pub angular: Vec3,
}

impl GpuKinematicMotion {
    fn gpu(self) -> ([f32; 3], [f32; 3]) {
        (self.linear.gpu(), self.angular.gpu())
    }
}

/// Initial state and mass properties of a GPU-resident primitive.
#[derive(Clone, Debug, uniffi::Record)]
pub struct GpuPrimitiveBody {
    /// Collision shape centered on the body origin.
    pub shape: GpuPrimitiveShape,
    /// World-space body origin.
    pub center: Vec3,
    /// XYZW unit orientation.
    pub orientation: Quaternion,
    /// Initial world-space linear velocity.
    pub velocity: Vec3,
    /// Initial world-space angular velocity.
    pub angular_velocity: Vec3,
    /// Nonnegative body mass; zero means static.
    pub mass: f64,
    /// Positive body-frame principal moments for a dynamic body; zero when static.
    pub principal_inertia: Vec3,
}

/// Point-to-point joint between two GPU-resident rigid bodies.
#[derive(Clone, Debug, uniffi::Record)]
pub struct GpuBallJoint {
    /// First body in the world's or environment's dense index order.
    pub body_a: u32,
    /// Second body in the world's or environment's dense index order.
    pub body_b: u32,
    /// Anchor point in body A's local frame.
    pub local_anchor_a: Vec3,
    /// Anchor point in body B's local frame.
    pub local_anchor_b: Vec3,
}

impl From<GpuBallJoint> for GpuRigidBallJoint {
    fn from(value: GpuBallJoint) -> Self {
        Self {
            body_a: value.body_a,
            body_b: value.body_b,
            local_anchor_a: value.local_anchor_a.gpu(),
            local_anchor_b: value.local_anchor_b.gpu(),
        }
    }
}

impl From<GpuRigidBallJoint> for GpuBallJoint {
    fn from(value: GpuRigidBallJoint) -> Self {
        Self {
            body_a: value.body_a,
            body_b: value.body_b,
            local_anchor_a: Vec3 {
                x: f64::from(value.local_anchor_a[0]),
                y: f64::from(value.local_anchor_a[1]),
                z: f64::from(value.local_anchor_a[2]),
            },
            local_anchor_b: Vec3 {
                x: f64::from(value.local_anchor_b[0]),
                y: f64::from(value.local_anchor_b[1]),
                z: f64::from(value.local_anchor_b[2]),
            },
        }
    }
}

/// Position and orientation lock between two GPU-resident local frames.
#[derive(Clone, Debug, uniffi::Record)]
pub struct GpuFixedJoint {
    /// First body in the world's or environment's dense index order.
    pub body_a: u32,
    /// Second body in the world's or environment's dense index order.
    pub body_b: u32,
    /// First frame origin in body A's local coordinates.
    pub local_anchor_a: Vec3,
    /// Second frame origin in body B's local coordinates.
    pub local_anchor_b: Vec3,
    /// First frame orientation in body A's local coordinates.
    pub local_rotation_a: Quaternion,
    /// Second frame orientation in body B's local coordinates.
    pub local_rotation_b: Quaternion,
}

impl From<GpuFixedJoint> for GpuRigidFixedJoint {
    fn from(value: GpuFixedJoint) -> Self {
        Self {
            body_a: value.body_a,
            body_b: value.body_b,
            local_anchor_a: value.local_anchor_a.gpu(),
            local_anchor_b: value.local_anchor_b.gpu(),
            local_rotation_a: [
                value.local_rotation_a.x as f32,
                value.local_rotation_a.y as f32,
                value.local_rotation_a.z as f32,
                value.local_rotation_a.w as f32,
            ],
            local_rotation_b: [
                value.local_rotation_b.x as f32,
                value.local_rotation_b.y as f32,
                value.local_rotation_b.z as f32,
                value.local_rotation_b.w as f32,
            ],
        }
    }
}

impl From<GpuRigidFixedJoint> for GpuFixedJoint {
    fn from(value: GpuRigidFixedJoint) -> Self {
        let quaternion = |rotation: [f32; 4]| Quaternion {
            x: f64::from(rotation[0]),
            y: f64::from(rotation[1]),
            z: f64::from(rotation[2]),
            w: f64::from(rotation[3]),
        };
        Self {
            body_a: value.body_a,
            body_b: value.body_b,
            local_anchor_a: Vec3 {
                x: f64::from(value.local_anchor_a[0]),
                y: f64::from(value.local_anchor_a[1]),
                z: f64::from(value.local_anchor_a[2]),
            },
            local_anchor_b: Vec3 {
                x: f64::from(value.local_anchor_b[0]),
                y: f64::from(value.local_anchor_b[1]),
                z: f64::from(value.local_anchor_b[2]),
            },
            local_rotation_a: quaternion(value.local_rotation_a),
            local_rotation_b: quaternion(value.local_rotation_b),
        }
    }
}

/// Hinge between two GPU-resident bodies with one free rotation axis.
#[derive(Clone, Debug, uniffi::Record)]
pub struct GpuRevoluteJoint {
    /// First body in the world's or environment's dense index order.
    pub body_a: u32,
    /// Second body in the world's or environment's dense index order.
    pub body_b: u32,
    /// First hinge origin in body A's local coordinates.
    pub local_anchor_a: Vec3,
    /// Second hinge origin in body B's local coordinates.
    pub local_anchor_b: Vec3,
    /// Unit hinge axis in body A's local coordinates.
    pub local_axis_a: Vec3,
    /// Unit hinge axis in body B's local coordinates.
    pub local_axis_b: Vec3,
}

impl From<GpuRevoluteJoint> for GpuRigidRevoluteJoint {
    fn from(value: GpuRevoluteJoint) -> Self {
        Self {
            body_a: value.body_a,
            body_b: value.body_b,
            local_anchor_a: value.local_anchor_a.gpu(),
            local_anchor_b: value.local_anchor_b.gpu(),
            local_axis_a: value.local_axis_a.gpu(),
            local_axis_b: value.local_axis_b.gpu(),
        }
    }
}

impl From<GpuRigidRevoluteJoint> for GpuRevoluteJoint {
    fn from(value: GpuRigidRevoluteJoint) -> Self {
        let vector = |v: [f32; 3]| Vec3 {
            x: f64::from(v[0]),
            y: f64::from(v[1]),
            z: f64::from(v[2]),
        };
        Self {
            body_a: value.body_a,
            body_b: value.body_b,
            local_anchor_a: vector(value.local_anchor_a),
            local_anchor_b: vector(value.local_anchor_b),
            local_axis_a: vector(value.local_axis_a),
            local_axis_b: vector(value.local_axis_b),
        }
    }
}

/// Slider between two GPU-resident bodies with one free local Z translation.
#[derive(Clone, Debug, uniffi::Record)]
pub struct GpuPrismaticJoint {
    /// First body in the world's or environment's dense index order.
    pub body_a: u32,
    /// Second body in the world's or environment's dense index order.
    pub body_b: u32,
    /// First frame origin in body A's local coordinates.
    pub local_anchor_a: Vec3,
    /// Second frame origin in body B's local coordinates.
    pub local_anchor_b: Vec3,
    /// First frame orientation as a unit XYZW quaternion.
    pub local_rotation_a: Quaternion,
    /// Second frame orientation as a unit XYZW quaternion.
    pub local_rotation_b: Quaternion,
}

impl From<GpuPrismaticJoint> for GpuRigidPrismaticJoint {
    fn from(value: GpuPrismaticJoint) -> Self {
        Self {
            body_a: value.body_a,
            body_b: value.body_b,
            local_anchor_a: value.local_anchor_a.gpu(),
            local_anchor_b: value.local_anchor_b.gpu(),
            local_rotation_a: [
                value.local_rotation_a.x as f32,
                value.local_rotation_a.y as f32,
                value.local_rotation_a.z as f32,
                value.local_rotation_a.w as f32,
            ],
            local_rotation_b: [
                value.local_rotation_b.x as f32,
                value.local_rotation_b.y as f32,
                value.local_rotation_b.z as f32,
                value.local_rotation_b.w as f32,
            ],
        }
    }
}

impl From<GpuRigidPrismaticJoint> for GpuPrismaticJoint {
    fn from(value: GpuRigidPrismaticJoint) -> Self {
        let vector = |v: [f32; 3]| Vec3 {
            x: f64::from(v[0]),
            y: f64::from(v[1]),
            z: f64::from(v[2]),
        };
        let quaternion = |v: [f32; 4]| Quaternion {
            x: f64::from(v[0]),
            y: f64::from(v[1]),
            z: f64::from(v[2]),
            w: f64::from(v[3]),
        };
        Self {
            body_a: value.body_a,
            body_b: value.body_b,
            local_anchor_a: vector(value.local_anchor_a),
            local_anchor_b: vector(value.local_anchor_b),
            local_rotation_a: quaternion(value.local_rotation_a),
            local_rotation_b: quaternion(value.local_rotation_b),
        }
    }
}

/// Velocity drive on a GPU-resident hinge or slider axis.
#[derive(Clone, Debug, uniffi::Record)]
pub struct GpuAxisMotor {
    /// Target relative velocity in radians or meters per second.
    pub target_velocity: f64,
    /// Maximum torque or force magnitude.
    pub max_force: f64,
}

impl From<GpuAxisMotor> for GpuRigidAxisMotor {
    fn from(value: GpuAxisMotor) -> Self {
        Self {
            target_velocity: value.target_velocity as f32,
            max_force: value.max_force as f32,
        }
    }
}

impl From<GpuRigidAxisMotor> for GpuAxisMotor {
    fn from(value: GpuRigidAxisMotor) -> Self {
        Self {
            target_velocity: f64::from(value.target_velocity),
            max_force: f64::from(value.max_force),
        }
    }
}

/// Force-limited position and velocity servo on a GPU-resident joint axis.
#[derive(Clone, Debug, uniffi::Record)]
pub struct GpuAxisServo {
    /// Target relative angle or displacement.
    pub position_target: f64,
    /// Target relative angular or linear velocity.
    pub velocity_target: f64,
    /// Force or torque per unit position error.
    pub stiffness: f64,
    /// Force or torque per unit velocity error.
    pub damping: f64,
    /// Maximum force or torque magnitude.
    pub max_force: f64,
}

impl From<GpuAxisServo> for GpuRigidAxisServo {
    fn from(value: GpuAxisServo) -> Self {
        Self {
            position_target: value.position_target as f32,
            velocity_target: value.velocity_target as f32,
            stiffness: value.stiffness as f32,
            damping: value.damping as f32,
            max_force: value.max_force as f32,
        }
    }
}

impl From<GpuRigidAxisServo> for GpuAxisServo {
    fn from(value: GpuRigidAxisServo) -> Self {
        Self {
            position_target: f64::from(value.position_target),
            velocity_target: f64::from(value.velocity_target),
            stiffness: f64::from(value.stiffness),
            damping: f64::from(value.damping),
            max_force: f64::from(value.max_force),
        }
    }
}

/// Reciprocal collision membership and filter masks for GPU-resident bodies.
#[derive(Clone, Copy, Debug, uniffi::Record)]
pub struct GpuCollisionGroups {
    /// Layers occupied by the collider.
    pub memberships: u32,
    /// Layers this collider accepts.
    pub filter: u32,
}

impl From<GpuCollisionGroups> for GpuRigidCollisionGroups {
    fn from(value: GpuCollisionGroups) -> Self {
        Self {
            memberships: value.memberships,
            filter: value.filter,
        }
    }
}

impl From<GpuRigidCollisionGroups> for GpuCollisionGroups {
    fn from(value: GpuRigidCollisionGroups) -> Self {
        Self {
            memberships: value.memberships,
            filter: value.filter,
        }
    }
}

/// Wrapped relative hinge angle bounds in radians.
#[derive(Clone, Debug, uniffi::Record)]
pub struct GpuRevoluteLimit {
    /// Lower relative angle bound.
    pub min: f64,
    /// Upper relative angle bound.
    pub max: f64,
}

impl From<GpuRevoluteLimit> for GpuRigidRevoluteLimit {
    fn from(value: GpuRevoluteLimit) -> Self {
        Self {
            min: value.min as f32,
            max: value.max as f32,
        }
    }
}

impl From<GpuRigidRevoluteLimit> for GpuRevoluteLimit {
    fn from(value: GpuRigidRevoluteLimit) -> Self {
        Self {
            min: f64::from(value.min),
            max: f64::from(value.max),
        }
    }
}

/// Allowed slider displacement between the two joint frames.
#[derive(Clone, Debug, uniffi::Record)]
pub struct GpuPrismaticLimit {
    /// Lower displacement bound in meters.
    pub min: f64,
    /// Upper displacement bound in meters.
    pub max: f64,
}

impl From<GpuPrismaticLimit> for GpuRigidPrismaticLimit {
    fn from(value: GpuPrismaticLimit) -> Self {
        Self {
            min: value.min as f32,
            max: value.max as f32,
        }
    }
}

impl From<GpuRigidPrismaticLimit> for GpuPrismaticLimit {
    fn from(value: GpuRigidPrismaticLimit) -> Self {
        Self {
            min: f64::from(value.min),
            max: f64::from(value.max),
        }
    }
}

/// Accumulated solver impulse at one GPU contact manifold point.
///
/// The impulse acts on `body_b` and its opposite acts on `body_a`.
/// Ground has no first body. Sleeping support can retain historical values.
#[derive(Clone, Debug, uniffi::Record)]
pub struct GpuContactImpulse {
    /// First body, absent for finite ground.
    pub body_a: Option<u32>,
    /// Body receiving the positive impulse.
    pub body_b: u32,
    /// Point index within the manifold.
    pub point_index: u32,
    /// Nonnegative normal impulse.
    pub normal_impulse: f64,
    /// World-space normal toward the second body.
    pub normal: Vec3,
    /// World-space friction impulse.
    pub tangent_impulse: Vec3,
    /// Combined normal and friction impulse on the second body.
    pub impulse_on_body_b: Vec3,
}

/// Sparse history from the last GPU solver substep, not a frame-wide sum.
#[derive(Clone, Debug, uniffi::Record)]
pub struct GpuContactImpulseReadback {
    /// Solver substep duration; absent when the cache has been cleared.
    pub dt: Option<f64>,
    /// Manifold points with positive normal impulse.
    pub contacts: Vec<GpuContactImpulse>,
}

impl From<tessera_physics::gpu_rigid_sphere_solver::GpuRigidContactImpulseReadback>
    for GpuContactImpulseReadback
{
    fn from(
        value: tessera_physics::gpu_rigid_sphere_solver::GpuRigidContactImpulseReadback,
    ) -> Self {
        let vector = |v: [f32; 3]| Vec3 {
            x: f64::from(v[0]),
            y: f64::from(v[1]),
            z: f64::from(v[2]),
        };
        Self {
            dt: value.dt.map(f64::from),
            contacts: value
                .contacts
                .into_iter()
                .map(|contact| GpuContactImpulse {
                    body_a: contact.body_a,
                    body_b: contact.body_b,
                    point_index: contact.point_index,
                    normal_impulse: f64::from(contact.normal_impulse),
                    normal: vector(contact.normal),
                    tangent_impulse: vector(contact.tangent_impulse),
                    impulse_on_body_b: vector(contact.impulse_on_body_b()),
                })
                .collect(),
        }
    }
}

/// One reported GPU contact after a step.
#[derive(Clone, Debug, uniffi::Record)]
pub struct GpuPrimitiveContact {
    /// Dense index of the first body.
    pub body_a: u32,
    /// Dense index of the second body; absent for finite-ground contact.
    pub body_b: Option<u32>,
    /// Midpoint of the contact witnesses.
    pub point: Vec3,
    /// Unit normal from the first body to the second, or upward for ground.
    pub normal: Vec3,
    /// Nonnegative penetration depth.
    pub depth: f64,
}

/// Last state and shape returned after removing a GPU primitive.
#[derive(Clone, Debug, uniffi::Record)]
pub struct RemovedGpuPrimitive {
    /// Body state at the time of removal.
    pub state: GpuPrimitiveState,
    /// Collision shape of the removed body.
    pub shape: GpuPrimitiveShape,
}

/// State and collision radius returned when a GPU sphere is removed.
#[derive(Clone, Debug, uniffi::Record)]
pub struct RemovedGpuSphere {
    /// Last GPU state before removal.
    pub state: GpuSphereState,
    /// Collision radius of the removed sphere.
    pub radius: f64,
}

impl From<GpuRigidBodyState> for GpuSphereState {
    fn from(state: GpuRigidBodyState) -> Self {
        Self {
            center: Vec3 {
                x: f64::from(state.position_inverse_mass[0]),
                y: f64::from(state.position_inverse_mass[1]),
                z: f64::from(state.position_inverse_mass[2]),
            },
            orientation: Quaternion {
                x: f64::from(state.orientation[0]),
                y: f64::from(state.orientation[1]),
                z: f64::from(state.orientation[2]),
                w: f64::from(state.orientation[3]),
            },
            velocity: Vec3 {
                x: f64::from(state.linear_velocity[0]),
                y: f64::from(state.linear_velocity[1]),
                z: f64::from(state.linear_velocity[2]),
            },
            angular_velocity: Vec3 {
                x: f64::from(state.angular_velocity[0]),
                y: f64::from(state.angular_velocity[1]),
                z: f64::from(state.angular_velocity[2]),
            },
            sleeping: state.inverse_inertia_sleep[3] == 1.0,
        }
    }
}

impl From<GpuRigidBodyState> for GpuPrimitiveState {
    fn from(state: GpuRigidBodyState) -> Self {
        let observed: GpuSphereState = state.into();
        Self {
            center: observed.center,
            orientation: observed.orientation,
            velocity: observed.velocity,
            angular_velocity: observed.angular_velocity,
            sleeping: observed.sleeping,
        }
    }
}

/// Named generalized-coordinate range.
#[derive(Clone, Debug, uniffi::Record)]
pub struct JointRange {
    /// Joint name.
    pub name: String,
    /// Inclusive start coordinate.
    pub start: u32,
    /// Exclusive end coordinate.
    pub end: u32,
}

/// Quaternion in XYZW order.
#[derive(Clone, Copy, Debug, uniffi::Record)]
pub struct Quaternion {
    /// X imaginary component.
    pub x: f64,
    /// Y imaginary component.
    pub y: f64,
    /// Z imaginary component.
    pub z: f64,
    /// Scalar component.
    pub w: f64,
}

/// World-space link transform.
#[derive(Clone, Copy, Debug, uniffi::Record)]
pub struct LinkPose {
    /// Link origin.
    pub position: Vec3,
    /// Link orientation.
    pub orientation: Quaternion,
}

/// World-space velocity at a point attached to an articulated link.
#[derive(Clone, Copy, Debug, uniffi::Record)]
pub struct LinkTwist {
    /// Linear velocity of the selected point in metres per second.
    pub linear: Vec3,
    /// Angular velocity of the link in radians per second.
    pub angular: Vec3,
}

impl From<CoreLinkTwist> for LinkTwist {
    fn from(value: CoreLinkTwist) -> Self {
        Self {
            linear: value.linear.into(),
            angular: value.angular.into(),
        }
    }
}

/// World-space acceleration at a point attached to a link.
#[derive(Clone, Copy, Debug, uniffi::Record)]
pub struct LinkAcceleration {
    /// Linear acceleration in metres per second squared.
    pub linear: Vec3,
    /// Angular acceleration in radians per second squared.
    pub angular: Vec3,
}

impl From<CoreLinkAcceleration> for LinkAcceleration {
    fn from(value: CoreLinkAcceleration) -> Self {
        Self {
            linear: value.linear.into(),
            angular: value.angular.into(),
        }
    }
}

/// Ideal accelerometer and gyroscope reading in the link frame.
#[derive(Clone, Copy, Debug, uniffi::Record)]
pub struct LinkImuReading {
    /// Proper acceleration at the sensor point in metres per second squared.
    pub specific_force: Vec3,
    /// Angular velocity in radians per second.
    pub angular_velocity: Vec3,
    /// Angular acceleration in radians per second squared.
    pub angular_acceleration: Vec3,
}

impl From<CoreLinkImuReading> for LinkImuReading {
    fn from(value: CoreLinkImuReading) -> Self {
        Self {
            specific_force: value.specific_force.into(),
            angular_velocity: value.angular_velocity.into(),
            angular_acceleration: value.angular_acceleration.into(),
        }
    }
}

/// Pose and velocity of one independent scene body.
#[derive(Clone, Debug, uniffi::Record)]
pub struct SceneBodyState {
    /// World-space body frame.
    pub pose: LinkPose,
    /// World-space origin velocity.
    pub linear_velocity: Vec3,
    /// World-space angular velocity.
    pub angular_velocity: Vec3,
    /// Mass in kilograms; zero indicates static or prescribed motion.
    pub mass: f64,
    /// Whether this zero-mass body integrates prescribed velocity.
    pub kinematic: bool,
    /// Body-frame inertia tensor about the origin in row-major order.
    pub inertia_tensor: Vec<f64>,
    /// Persistent world-space force at the body origin.
    pub force: Vec3,
    /// Persistent world-space torque about the body origin.
    pub torque: Vec3,
    /// Number of attached colliders.
    pub collider_count: u32,
}

/// Collision shape attached to an independent scene body.
#[derive(Clone, Debug, uniffi::Enum)]
pub enum SceneShape {
    /// Sphere centered at the collider frame origin.
    Sphere {
        /// Positive collision radius.
        radius: f64,
    },
    /// Oriented box with positive half extents.
    Box {
        /// Positive local half extents.
        half_extents: Vec3,
    },
    /// Z-axis capsule with a rounded border.
    Capsule {
        /// Half the central segment length.
        half_height: f64,
        /// Positive rounded border radius.
        radius: f64,
    },
    /// Z-axis cylinder with flat end caps.
    Cylinder {
        /// Half the full cylinder height.
        half_height: f64,
        /// Positive circular radius.
        radius: f64,
    },
    /// Z-axis cone with its apex at positive half height.
    Cone {
        /// Half the full cone height.
        half_height: f64,
        /// Positive base radius.
        radius: f64,
    },
    /// Prepared convex hull topology.
    Convex {
        /// Validated hull vertices, normals, and edge directions.
        geometry: ConvexMeshPart,
    },
    /// Indexed triangle surface.
    TriangleMesh {
        /// Collider-local finite vertices.
        vertices: Vec<Vec3>,
        /// Nondegenerate indexed triangles.
        triangles: Vec<GpuTriangle>,
    },
    /// Regular centered z-up heightfield.
    HeightField {
        /// Number of rows along local y.
        rows: u32,
        /// Number of columns along local x.
        columns: u32,
        /// Row-major height samples.
        heights: Vec<f64>,
        /// Full x/y extents and positive vertical multiplier.
        scale: Vec3,
    },
    /// Indexed zero-thickness line segments.
    Polyline {
        /// Collider-local finite vertices.
        vertices: Vec<Vec3>,
        /// Nondegenerate indexed segments.
        segments: Vec<GpuSegment>,
    },
}

/// One scene collider and its body-local frame.
#[derive(Clone, Debug, uniffi::Record)]
pub struct SceneColliderInput {
    /// Body-local collider frame.
    pub frame: LinkPose,
    /// Collision geometry in the collider frame.
    pub shape: SceneShape,
}

/// Initial state, mass properties, and colliders of a scene body.
#[derive(Clone, Debug, uniffi::Record)]
pub struct SceneBodyInput {
    /// World-space body frame.
    pub pose: LinkPose,
    /// World-space origin velocity.
    pub linear_velocity: Vec3,
    /// World-space angular velocity.
    pub angular_velocity: Vec3,
    /// Nonnegative mass; zero creates a static body.
    pub mass: f64,
    /// Body-frame inertia tensor about the origin in row-major order.
    pub inertia_tensor: Vec<f64>,
    /// Persistent world-space force at the body origin.
    pub force: Vec3,
    /// Body-attached colliders; an empty list creates a body without collision geometry.
    pub colliders: Vec<SceneColliderInput>,
}

/// Material coefficient combination rule.
#[derive(Clone, Copy, Debug, uniffi::Enum)]
pub enum CombineRule {
    /// Arithmetic mean.
    Average,
    /// Smaller coefficient.
    Min,
    /// Product of coefficients.
    Multiply,
    /// Larger coefficient.
    Max,
}

impl From<CombineRule> for CoefficientCombineRule {
    fn from(value: CombineRule) -> Self {
        match value {
            CombineRule::Average => Self::Average,
            CombineRule::Min => Self::Min,
            CombineRule::Multiply => Self::Multiply,
            CombineRule::Max => Self::Max,
        }
    }
}

impl From<CoefficientCombineRule> for CombineRule {
    fn from(value: CoefficientCombineRule) -> Self {
        match value {
            CoefficientCombineRule::Average => Self::Average,
            CoefficientCombineRule::Min => Self::Min,
            CoefficientCombineRule::Multiply => Self::Multiply,
            CoefficientCombineRule::Max => Self::Max,
        }
    }
}

/// Friction and restitution for one collider.
#[derive(Clone, Copy, Debug, uniffi::Record)]
pub struct Material {
    /// Coulomb friction coefficient.
    pub friction: f64,
    /// Normal restitution coefficient.
    pub restitution: f64,
    /// Combination rule for friction.
    pub friction_rule: CombineRule,
    /// Combination rule for restitution.
    pub restitution_rule: CombineRule,
}

impl From<Material> for ColliderMaterial {
    fn from(value: Material) -> Self {
        Self {
            friction: value.friction,
            restitution: value.restitution,
            friction_combine_rule: value.friction_rule.into(),
            restitution_combine_rule: value.restitution_rule.into(),
        }
    }
}

impl From<ColliderMaterial> for Material {
    fn from(value: ColliderMaterial) -> Self {
        Self {
            friction: value.friction,
            restitution: value.restitution,
            friction_rule: value.friction_combine_rule.into(),
            restitution_rule: value.restitution_combine_rule.into(),
        }
    }
}

/// Optional position target and a bounded generalized-force motor.
#[derive(Clone, Copy, Debug, uniffi::Record)]
pub struct JointMotor {
    /// Position target, or none for velocity-only drive.
    pub position_target: Option<f64>,
    /// Velocity target.
    pub velocity_target: f64,
    /// Position gain.
    pub stiffness: f64,
    /// Velocity gain.
    pub damping: f64,
    /// Maximum absolute motor force.
    pub max_force: f64,
}

impl From<JointMotor> for CoreJointMotor {
    fn from(value: JointMotor) -> Self {
        Self {
            position_target: value.position_target,
            velocity_target: value.velocity_target,
            stiffness: value.stiffness,
            damping: value.damping,
            max_force: value.max_force,
        }
    }
}

impl From<CoreJointMotor> for JointMotor {
    fn from(value: CoreJointMotor) -> Self {
        Self {
            position_target: value.position_target,
            velocity_target: value.velocity_target,
            stiffness: value.stiffness,
            damping: value.damping,
            max_force: value.max_force,
        }
    }
}

/// Passive joint spring and viscous damping, separate from motor effort.
#[derive(Clone, Copy, Debug, uniffi::Record)]
pub struct JointPassive {
    /// Nonnegative spring stiffness.
    pub stiffness: f64,
    /// Nonnegative viscous damping.
    pub damping: f64,
    /// Zero-force spring position.
    pub rest_position: f64,
}

impl From<JointPassive> for CoreJointPassive {
    fn from(value: JointPassive) -> Self {
        Self {
            stiffness: value.stiffness,
            damping: value.damping,
            rest_position: value.rest_position,
        }
    }
}

impl From<CoreJointPassive> for JointPassive {
    fn from(value: CoreJointPassive) -> Self {
        Self {
            stiffness: value.stiffness,
            damping: value.damping,
            rest_position: value.rest_position,
        }
    }
}

/// Nonlinear passive spring and damping coefficients.
#[derive(Clone, Copy, Debug, uniffi::Record)]
pub struct JointNonlinearPassive {
    /// Quadratic spring coefficient.
    pub spring_quadratic: f64,
    /// Cubic spring coefficient.
    pub spring_cubic: f64,
    /// Velocity-times-absolute-velocity damping coefficient.
    pub damping_quadratic: f64,
    /// Cubic damping coefficient.
    pub damping_cubic: f64,
}

impl From<JointNonlinearPassive> for CoreJointNonlinearPassive {
    fn from(value: JointNonlinearPassive) -> Self {
        Self {
            spring_quadratic: value.spring_quadratic,
            spring_cubic: value.spring_cubic,
            damping_quadratic: value.damping_quadratic,
            damping_cubic: value.damping_cubic,
        }
    }
}

impl From<CoreJointNonlinearPassive> for JointNonlinearPassive {
    fn from(value: CoreJointNonlinearPassive) -> Self {
        Self {
            spring_quadratic: value.spring_quadratic,
            spring_cubic: value.spring_cubic,
            damping_quadratic: value.damping_quadratic,
            damping_cubic: value.damping_cubic,
        }
    }
}

/// Holonomic relation between two independent generalized coordinates.
#[derive(Clone, Copy, Debug, uniffi::Record)]
pub struct JointCoupling {
    /// Source coordinate index.
    pub source: u32,
    /// Follower coordinate index.
    pub follower: u32,
    /// Follower displacement per source displacement.
    pub multiplier: f64,
    /// Follower position when the source is zero.
    pub offset: f64,
}

impl From<JointCoupling> for CoreJointCoupling {
    fn from(value: JointCoupling) -> Self {
        Self {
            source: value.source as usize,
            follower: value.follower as usize,
            multiplier: value.multiplier,
            offset: value.offset,
        }
    }
}

impl From<CoreJointCoupling> for JointCoupling {
    fn from(value: CoreJointCoupling) -> Self {
        Self {
            source: value.source as u32,
            follower: value.follower as u32,
            multiplier: value.multiplier,
            offset: value.offset,
        }
    }
}

/// Quartic equality between two scalar joint coordinates.
#[derive(Clone, Debug, uniffi::Record)]
pub struct JointPolynomialCoupling {
    /// Constrained coordinate index.
    pub follower: u32,
    /// Driving coordinate index, or None to lock the follower.
    pub source: Option<u32>,
    /// Polynomial coefficients from constant through quartic order.
    pub coefficients: Vec<f64>,
    /// Follower coordinate in the reference configuration.
    pub follower_reference: f64,
    /// Source coordinate in the reference configuration.
    pub source_reference: f64,
}

impl TryFrom<JointPolynomialCoupling> for CoreJointPolynomialCoupling {
    type Error = TesseraError;

    fn try_from(value: JointPolynomialCoupling) -> Result<Self, Self::Error> {
        let coefficients = value
            .coefficients
            .try_into()
            .map_err(|_| failed("joint polynomial coupling needs five coefficients"))?;
        Ok(Self {
            follower: value.follower as usize,
            source: value.source.map(|source| source as usize),
            coefficients,
            follower_reference: value.follower_reference,
            source_reference: value.source_reference,
        })
    }
}

impl From<CoreJointPolynomialCoupling> for JointPolynomialCoupling {
    fn from(value: CoreJointPolynomialCoupling) -> Self {
        Self {
            follower: value.follower as u32,
            source: value.source.map(|source| source as u32),
            coefficients: value.coefficients.to_vec(),
            follower_reference: value.follower_reference,
            source_reference: value.source_reference,
        }
    }
}

/// Ball constraint between two articulation links or one link and the world.
#[derive(Clone, Copy, Debug, uniffi::Record)]
pub struct LinkPointConstraint {
    /// First link index.
    pub link_a: u32,
    /// Local point on the first link.
    pub point_a: Vec3,
    /// Second link index, or None for a fixed world-space point.
    pub link_b: Option<u32>,
    /// Local point on the second link, or world-space point when link_b is None.
    pub point_b: Vec3,
}

impl From<LinkPointConstraint> for CoreLinkPointConstraint {
    fn from(value: LinkPointConstraint) -> Self {
        Self {
            link_a: value.link_a as usize,
            point_a: [value.point_a.x, value.point_a.y, value.point_a.z],
            link_b: value.link_b.map(|link| link as usize),
            point_b: [value.point_b.x, value.point_b.y, value.point_b.z],
        }
    }
}

impl From<CoreLinkPointConstraint> for LinkPointConstraint {
    fn from(value: CoreLinkPointConstraint) -> Self {
        Self {
            link_a: value.link_a as u32,
            point_a: Vector3::from(value.point_a).into(),
            link_b: value.link_b.map(|link| link as u32),
            point_b: Vector3::from(value.point_b).into(),
        }
    }
}

/// Fixed-frame constraint between two articulation links or a link and the world.
#[derive(Clone, Copy, Debug, uniffi::Record)]
pub struct LinkFixedConstraint {
    /// First link index.
    pub link_a: u32,
    /// Local frame on the first link.
    pub frame_a: LinkPose,
    /// Second link index, or None for a fixed world-space frame.
    pub link_b: Option<u32>,
    /// Local frame on the second link, or world-space frame when link_b is None.
    pub frame_b: LinkPose,
}

fn link_pose_isometry(pose: LinkPose) -> Result<Isometry3<f64>, TesseraError> {
    let q = pose.orientation;
    let quaternion = NaQuaternion::new(q.w, q.x, q.y, q.z);
    let norm = quaternion.norm_squared();
    if !norm.is_finite()
        || norm <= 1e-24
        || ![pose.position.x, pose.position.y, pose.position.z]
            .iter()
            .all(|value| value.is_finite())
    {
        return Err(failed("frame must be finite with nonzero quaternion"));
    }
    Ok(Isometry3::from_parts(
        Translation3::from(Vector3::new(
            pose.position.x,
            pose.position.y,
            pose.position.z,
        )),
        UnitQuaternion::new_normalize(quaternion),
    ))
}

fn isometry_link_pose(pose: Isometry3<f64>) -> LinkPose {
    let q = pose.rotation.quaternion();
    LinkPose {
        position: pose.translation.vector.into(),
        orientation: Quaternion {
            x: q.i,
            y: q.j,
            z: q.k,
            w: q.w,
        },
    }
}

/// Pure kinematic configuration, separate from simulation velocities and caches.
#[derive(Clone, Debug, uniffi::Record)]
pub struct IkState {
    /// Root world pose.
    pub root_pose: LinkPose,
    /// Reduced joint coordinates; explicit spherical slots are workspace.
    pub positions: Vec<f64>,
    /// Optional per-edge spherical quaternions, with None on scalar/fixed edges.
    pub orientations: Option<Vec<Option<Quaternion>>>,
}

impl IkState {
    fn native(self) -> Result<CoreIkState, TesseraError> {
        let orientations = self
            .orientations
            .map(|values| {
                values
                    .into_iter()
                    .map(|value| {
                        value
                            .map(|orientation| {
                                link_pose_isometry(LinkPose {
                                    position: Vec3 {
                                        x: 0.0,
                                        y: 0.0,
                                        z: 0.0,
                                    },
                                    orientation,
                                })
                                .map(|pose| pose.rotation)
                            })
                            .transpose()
                    })
                    .collect::<Result<Vec<_>, TesseraError>>()
            })
            .transpose()?;
        Ok(CoreIkState {
            root_pose: link_pose_isometry(self.root_pose)?,
            positions: self.positions,
            orientations,
        })
    }
}

impl From<CoreIkState> for IkState {
    fn from(value: CoreIkState) -> Self {
        Self {
            root_pose: isometry_link_pose(value.root_pose),
            positions: value.positions,
            orientations: value.orientations.map(|values| {
                values
                    .into_iter()
                    .map(|value| {
                        value.map(|rotation| {
                            isometry_link_pose(Isometry3::from_parts(
                                Translation3::identity(),
                                rotation,
                            ))
                            .orientation
                        })
                    })
                    .collect()
            }),
        }
    }
}

/// Damped-least-squares iteration and convergence controls.
#[derive(Clone, Debug, uniffi::Record)]
pub struct IkConfig {
    /// Maximum number of displacement updates.
    pub max_iterations: u32,
    /// Positive normal-equation damping.
    pub damping: f64,
    /// Constrained position tolerance in metres.
    pub position_tolerance: f64,
    /// Constrained rotation tolerance in radians.
    pub rotation_tolerance: f64,
    /// Scalar coupling tolerance.
    pub coupling_tolerance: f64,
    /// Movable slots; floating world-linear/world-angular slots precede joints.
    pub dofs: Option<Vec<u32>>,
}

impl Default for IkConfig {
    fn default() -> Self {
        let native = CoreIkConfig::default();
        Self {
            max_iterations: native.max_iterations as u32,
            damping: native.damping,
            position_tolerance: native.position_tolerance,
            rotation_tolerance: native.rotation_tolerance,
            coupling_tolerance: native.coupling_tolerance,
            dofs: None,
        }
    }
}

/// Defaults matching the Nexus CPU IK contract.
#[uniffi::export]
pub fn default_ik_config() -> IkConfig {
    IkConfig::default()
}

/// One world-frame link target.
#[derive(Clone, Debug, uniffi::Record)]
pub struct IkTarget {
    /// Stable link index.
    pub link: u32,
    /// Desired local-point world position and link orientation.
    pub pose: LinkPose,
    /// Point expressed in the link frame.
    pub local_point: Vec3,
    /// Exactly six world linear X/Y/Z then angular X/Y/Z constraint flags.
    pub constrained_axes: Vec<bool>,
}

/// Actual IK convergence and final state, including unreachable targets.
#[derive(Clone, Debug, uniffi::Record)]
pub struct IkResult {
    /// Solved configuration.
    pub state: IkState,
    /// Whether pose and scalar coupling tolerances were met.
    pub converged: bool,
    /// Number of displacement updates applied.
    pub iterations: u32,
    /// Six world-frame errors with disabled components zeroed.
    pub residual: Vec<f64>,
    /// Maximum absolute scalar equality error.
    pub coupling_residual: f64,
}

fn scene_sphere_body(
    pose: LinkPose,
    radius: f64,
    mass: f64,
) -> Result<CoreSceneBody, TesseraError> {
    if !radius.is_finite() || radius <= 0.0 {
        return Err(failed("scene sphere radius must be positive and finite"));
    }
    let inertia = if mass == 0.0 {
        1.0
    } else {
        0.4 * mass * radius * radius
    };
    CoreSceneBody::new(
        link_pose_isometry(pose)?,
        mass,
        Matrix3::identity() * inertia,
        vec![CoreSceneCollider::Sphere {
            center: Vector3::zeros(),
            radius,
        }],
    )
    .map_err(failed)
}

impl TryFrom<SceneColliderInput> for CoreSceneCollider {
    type Error = TesseraError;

    fn try_from(input: SceneColliderInput) -> Result<Self, Self::Error> {
        let frame = link_pose_isometry(input.frame)?;
        Ok(match input.shape {
            SceneShape::Sphere { radius } => Self::Sphere {
                center: frame.translation.vector,
                radius,
            },
            SceneShape::Box { half_extents } => Self::Box {
                origin: frame,
                half_extents: half_extents.nalgebra(),
            },
            SceneShape::Capsule {
                half_height,
                radius,
            } => Self::Capsule {
                origin: frame,
                half_height,
                radius,
            },
            SceneShape::Cylinder {
                half_height,
                radius,
            } => Self::Cylinder {
                origin: frame,
                half_height,
                radius,
            },
            SceneShape::Cone {
                half_height,
                radius,
            } => Self::Cone {
                origin: frame,
                half_height,
                radius,
            },
            SceneShape::Convex { geometry } => Self::Convex {
                origin: frame,
                geometry: ConvexGeometry::new(
                    geometry.vertices.into_iter().map(Vec3::nalgebra).collect(),
                    geometry
                        .face_normals
                        .into_iter()
                        .map(Vec3::nalgebra)
                        .collect(),
                    geometry
                        .edge_directions
                        .into_iter()
                        .map(Vec3::nalgebra)
                        .collect(),
                )
                .map_err(failed)?,
            },
            SceneShape::TriangleMesh {
                vertices,
                triangles,
            } => Self::TriangleMesh {
                origin: frame,
                geometry: TriangleMeshGeometry::new(
                    vertices.into_iter().map(Vec3::nalgebra).collect(),
                    triangles
                        .into_iter()
                        .map(|triangle| [triangle.a, triangle.b, triangle.c])
                        .collect(),
                )
                .map_err(failed)?,
            },
            SceneShape::HeightField {
                rows,
                columns,
                heights,
                scale,
            } => Self::HeightField {
                origin: frame,
                geometry: HeightFieldGeometry::new(
                    rows as usize,
                    columns as usize,
                    heights,
                    scale.nalgebra(),
                )
                .map_err(failed)?,
            },
            SceneShape::Polyline { vertices, segments } => Self::Polyline {
                origin: frame,
                geometry: PolylineGeometry::new(
                    vertices.into_iter().map(Vec3::nalgebra).collect(),
                    segments
                        .into_iter()
                        .map(|segment| [segment.a, segment.b])
                        .collect(),
                )
                .map_err(failed)?,
            },
        })
    }
}

impl TryFrom<SceneBodyInput> for CoreSceneBody {
    type Error = TesseraError;

    fn try_from(input: SceneBodyInput) -> Result<Self, Self::Error> {
        let linear_velocity = input.linear_velocity.nalgebra();
        let angular_velocity = input.angular_velocity.nalgebra();
        let force = input.force.nalgebra();
        if linear_velocity
            .iter()
            .chain(angular_velocity.iter())
            .chain(force.iter())
            .any(|value| !value.is_finite())
        {
            return Err(failed("scene body velocity and force must be finite"));
        }
        if input.inertia_tensor.len() != 9 {
            return Err(failed(
                "scene body inertia tensor requires nine row-major values",
            ));
        }
        let colliders = input
            .colliders
            .into_iter()
            .map(TryInto::try_into)
            .collect::<Result<Vec<_>, _>>()?;
        let mut body = CoreSceneBody::new(
            link_pose_isometry(input.pose)?,
            input.mass,
            Matrix3::from_row_slice(&input.inertia_tensor),
            colliders,
        )
        .map_err(failed)?;
        body.linear_velocity = linear_velocity;
        body.angular_velocity = angular_velocity;
        body.force = force;
        Ok(body)
    }
}

impl From<&CoreSceneBody> for SceneBodyState {
    fn from(body: &CoreSceneBody) -> Self {
        Self {
            pose: isometry_link_pose(body.pose),
            linear_velocity: body.linear_velocity.into(),
            angular_velocity: body.angular_velocity.into(),
            mass: body.mass,
            kinematic: body.kinematic,
            inertia_tensor: (0..3)
                .flat_map(|row| (0..3).map(move |column| body.inertia[(row, column)]))
                .collect(),
            force: body.force.into(),
            torque: body.torque.into(),
            collider_count: body.colliders.len() as u32,
        }
    }
}

impl TryFrom<LinkFixedConstraint> for CoreLinkFixedConstraint {
    type Error = TesseraError;

    fn try_from(value: LinkFixedConstraint) -> Result<Self, Self::Error> {
        Ok(Self {
            link_a: value.link_a as usize,
            frame_a: link_pose_isometry(value.frame_a)?,
            link_b: value.link_b.map(|link| link as usize),
            frame_b: link_pose_isometry(value.frame_b)?,
        })
    }
}

impl From<CoreLinkFixedConstraint> for LinkFixedConstraint {
    fn from(value: CoreLinkFixedConstraint) -> Self {
        Self {
            link_a: value.link_a as u32,
            frame_a: isometry_link_pose(value.frame_a),
            link_b: value.link_b.map(|link| link as u32),
            frame_b: isometry_link_pose(value.frame_b),
        }
    }
}

/// Ball constraint between an articulation link and an independent scene body.
#[derive(Clone, Copy, Debug, uniffi::Record)]
pub struct LinkScenePointConstraint {
    /// Link index.
    pub link: u32,
    /// Link-local point.
    pub link_point: Vec3,
    /// Scene body index.
    pub body: u32,
    /// Scene-body-local point.
    pub body_point: Vec3,
}

impl From<LinkScenePointConstraint> for CoreLinkScenePointConstraint {
    fn from(value: LinkScenePointConstraint) -> Self {
        Self {
            link: value.link as usize,
            link_point: [value.link_point.x, value.link_point.y, value.link_point.z],
            body: value.body as usize,
            body_point: [value.body_point.x, value.body_point.y, value.body_point.z],
        }
    }
}

impl From<CoreLinkScenePointConstraint> for LinkScenePointConstraint {
    fn from(value: CoreLinkScenePointConstraint) -> Self {
        Self {
            link: value.link as u32,
            link_point: Vector3::from(value.link_point).into(),
            body: value.body as u32,
            body_point: Vector3::from(value.body_point).into(),
        }
    }
}

/// Fixed-frame constraint between an articulation link and a scene body.
#[derive(Clone, Copy, Debug, uniffi::Record)]
pub struct LinkSceneFixedConstraint {
    /// Link index.
    pub link: u32,
    /// Link-local frame.
    pub link_frame: LinkPose,
    /// Scene body index.
    pub body: u32,
    /// Scene-body-local frame.
    pub body_frame: LinkPose,
}

impl TryFrom<LinkSceneFixedConstraint> for CoreLinkSceneFixedConstraint {
    type Error = TesseraError;

    fn try_from(value: LinkSceneFixedConstraint) -> Result<Self, Self::Error> {
        Ok(Self {
            link: value.link as usize,
            link_frame: link_pose_isometry(value.link_frame)?,
            body: value.body as usize,
            body_frame: link_pose_isometry(value.body_frame)?,
        })
    }
}

impl From<CoreLinkSceneFixedConstraint> for LinkSceneFixedConstraint {
    fn from(value: CoreLinkSceneFixedConstraint) -> Self {
        Self {
            link: value.link as u32,
            link_frame: isometry_link_pose(value.link_frame),
            body: value.body as u32,
            body_frame: isometry_link_pose(value.body_frame),
        }
    }
}

/// Persistent world-frame load of one articulated link.
#[derive(Clone, Copy, Debug, uniffi::Record)]
pub struct LinkLoad {
    /// Persistent force in world coordinates, applied at the link COM.
    pub force: Vec3,
    /// Persistent torque in world coordinates about the link COM.
    pub torque: Vec3,
    /// Multiplier for this link's gravity contribution.
    pub gravity_scale: f64,
}

impl From<CoreLinkExternalLoad> for LinkLoad {
    fn from(value: CoreLinkExternalLoad) -> Self {
        Self {
            force: value.force.into(),
            torque: value.torque.into(),
            gravity_scale: value.gravity_scale,
        }
    }
}

/// Contact wrench on a link from the last integration substep.
#[derive(Clone, Copy, Debug, uniffi::Record)]
pub struct LinkContactWrench {
    /// World-frame force in newtons.
    pub force: Vec3,
    /// World-frame torque about the link origin in newton metres.
    pub torque: Vec3,
}

/// One solved scene-body contact from the final resident substep.
#[derive(Clone, Debug, uniffi::Record)]
pub struct SceneContactSample {
    /// World application point in metres, before integration.
    pub position: Vec3,
    /// World normal pointing into the body.
    pub normal: Vec3,
    /// World force including friction, in newtons.
    pub force: Vec3,
    /// World impulse including friction, in newton seconds.
    pub impulse: Vec3,
    /// Signed geometric separation in metres.
    pub distance: f64,
}

/// Solved scene-body wrench and contacts from the final resident substep.
#[derive(Clone, Debug, uniffi::Record)]
pub struct SceneContactReport {
    /// World force in newtons.
    pub force: Vec3,
    /// World torque about the body origin before integration, in newton metres.
    pub torque: Vec3,
    /// Individual solved witnesses. Static collision-only bodies have no samples.
    pub samples: Vec<SceneContactSample>,
}

fn scene_contact_reports(
    diagnostics: tessera_physics::articulated_world::SceneContactDiagnostics,
    timestep: f64,
) -> Vec<SceneContactReport> {
    diagnostics
        .forces
        .into_iter()
        .zip(diagnostics.torques)
        .zip(diagnostics.samples)
        .map(|((force, torque), samples)| SceneContactReport {
            force: force.into(),
            torque: torque.into(),
            samples: samples
                .into_iter()
                .map(|sample| SceneContactSample {
                    position: sample.position.into(),
                    normal: sample.normal.into(),
                    force: sample.force.into(),
                    impulse: (sample.force * timestep).into(),
                    distance: sample.distance,
                })
                .collect(),
        })
        .collect()
}

/// Type of collider attached to an articulated link.
#[derive(Clone, Copy, Debug, uniffi::Enum)]
pub enum LinkColliderKind {
    /// Sphere collider.
    Sphere,
    /// Point sampling the ground plane.
    GroundPoint,
    /// Oriented box collider.
    Box,
    /// Cylinder collider.
    Cylinder,
    /// Convex hull collider.
    Convex,
}

/// Unscaled convex collision part supplied for one external mesh URI.
#[derive(Clone, Debug, uniffi::Record)]
pub struct ConvexMeshPart {
    /// Convex hull vertices in original model coordinates.
    pub vertices: Vec<Vec3>,
    /// Outward unit face normals.
    pub face_normals: Vec<Vec3>,
    /// Unit edge directions.
    pub edge_directions: Vec<Vec3>,
}

/// One external mesh URI and its convex decomposition.
#[derive(Clone, Debug, uniffi::Record)]
pub struct MeshAsset {
    /// URI as used by the URDF or MJCF document.
    pub uri: String,
    /// Nonempty convex decomposition.
    pub parts: Vec<ConvexMeshPart>,
}

/// Source format for one articulated batch environment.
#[derive(Clone, Copy, Debug, uniffi::Enum)]
pub enum ModelFormat {
    /// URDF document.
    Urdf,
    /// MJCF document.
    Mjcf,
}

/// One independently simulated articulated model.
#[derive(Clone, Debug, uniffi::Record)]
pub struct BatchModel {
    /// Document format.
    pub format: ModelFormat,
    /// Model XML text.
    pub xml: String,
    /// Floating root for URDF; ignored for MJCF.
    pub floating_base: bool,
    /// External convex collision meshes referenced by this model.
    pub meshes: Vec<MeshAsset>,
}

/// Stable metadata for one articulated batch environment.
#[derive(Clone, Debug, uniffi::Record)]
pub struct BatchEnvironmentInfo {
    /// Model name, or empty if MJCF omitted it.
    pub name: String,
    /// Link names in articulation order.
    pub link_names: Vec<String>,
    /// Generalized-coordinate ranges in joint order.
    pub joint_ranges: Vec<JointRange>,
}

fn mesh_assets(
    assets: Vec<MeshAsset>,
) -> Result<BTreeMap<String, Vec<ConvexMeshPart>>, TesseraError> {
    let mut by_uri = BTreeMap::new();
    for asset in assets {
        if asset.uri.is_empty() || asset.parts.is_empty() || by_uri.contains_key(&asset.uri) {
            return Err(failed(
                "mesh URIs must be unique, nonempty, and have convex parts",
            ));
        }
        for part in &asset.parts {
            let _ = ConvexGeometry::new(
                part.vertices.iter().copied().map(Vec3::nalgebra).collect(),
                part.face_normals
                    .iter()
                    .copied()
                    .map(Vec3::nalgebra)
                    .collect(),
                part.edge_directions
                    .iter()
                    .copied()
                    .map(Vec3::nalgebra)
                    .collect(),
            )
            .map_err(failed)?;
        }
        let _ = by_uri.insert(asset.uri, asset.parts);
    }
    Ok(by_uri)
}

fn resolve_mesh_parts(
    assets: &BTreeMap<String, Vec<ConvexMeshPart>>,
    filename: &str,
    scale: [f64; 3],
) -> Result<Vec<ConvexGeometry>, String> {
    if scale
        .iter()
        .any(|value| !value.is_finite() || *value == 0.0)
    {
        return Err("mesh scale must be finite and nonzero".into());
    }
    let parts = assets
        .get(filename)
        .ok_or_else(|| format!("mesh URI not provided: {filename}"))?;
    parts
        .iter()
        .map(|part| {
            let vertices = part
                .vertices
                .iter()
                .map(|point| point.nalgebra().component_mul(&Vector3::from(scale)))
                .collect();
            let normals = part
                .face_normals
                .iter()
                .map(|normal| {
                    Vector3::new(
                        normal.x / scale[0],
                        normal.y / scale[1],
                        normal.z / scale[2],
                    )
                    .try_normalize(1e-12)
                    .ok_or_else(|| "scaled mesh has a degenerate face normal".to_owned())
                })
                .collect::<Result<Vec<_>, _>>()?;
            let edges = part
                .edge_directions
                .iter()
                .map(|edge| {
                    edge.nalgebra()
                        .component_mul(&Vector3::from(scale))
                        .try_normalize(1e-12)
                        .ok_or_else(|| "scaled mesh has a degenerate edge direction".to_owned())
                })
                .collect::<Result<Vec<_>, _>>()?;
            ConvexGeometry::new(vertices, normals, edges).map_err(|error| error.to_string())
        })
        .collect()
}

fn gpu_sphere(input: SphereInput) -> Result<(GpuRigidBodyState, f32), TesseraError> {
    if !input.radius.is_finite()
        || input.radius <= 0.0
        || !input.mass.is_finite()
        || input.mass < 0.0
    {
        return Err(failed("radius must be positive and mass nonnegative"));
    }
    let inverse_mass = if input.mass > 0.0 {
        (1.0 / input.mass) as f32
    } else {
        0.0
    };
    let inverse_inertia = if input.mass > 0.0 {
        (2.5 / (input.mass * input.radius * input.radius)) as f32
    } else {
        0.0
    };
    let [x, y, z] = input.center.gpu();
    let [vx, vy, vz] = input.velocity.gpu();
    let radius = input.radius as f32;
    let state = GpuRigidBodyState {
        position_inverse_mass: [x, y, z, inverse_mass],
        orientation: [0.0, 0.0, 0.0, 1.0],
        linear_velocity: [vx, vy, vz, 0.0],
        angular_velocity: [0.0; 4],
        inverse_inertia_sleep: [inverse_inertia, inverse_inertia, inverse_inertia, 0.0],
    };
    if !state.is_valid() || !radius.is_finite() || radius <= 0.0 {
        return Err(failed("body cannot be represented by GPU f32 state"));
    }
    Ok((state, radius))
}

fn gpu_primitive(
    input: GpuPrimitiveBody,
) -> Result<(GpuRigidBodyState, GpuRigidShape), TesseraError> {
    let shape = match input.shape {
        GpuPrimitiveShape::Sphere { radius } => GpuRigidShape::Sphere {
            radius: radius as f32,
        },
        GpuPrimitiveShape::Box { half_extents } => GpuRigidShape::Box {
            half_extents: half_extents.gpu(),
        },
        GpuPrimitiveShape::Capsule {
            radius,
            half_length,
        } => GpuRigidShape::Capsule {
            radius: radius as f32,
            half_length: half_length as f32,
        },
        GpuPrimitiveShape::Cylinder {
            radius,
            half_length,
        } => GpuRigidShape::Cylinder {
            radius: radius as f32,
            half_length: half_length as f32,
        },
        GpuPrimitiveShape::Cone {
            radius,
            half_length,
        } => GpuRigidShape::Cone {
            radius: radius as f32,
            half_length: half_length as f32,
        },
        GpuPrimitiveShape::Convex { vertices } => GpuRigidShape::Convex {
            vertices: vertices.into_iter().map(|vertex| vertex.gpu()).collect(),
        },
        GpuPrimitiveShape::Heightfield {
            rows,
            columns,
            heights,
            scale,
        } => {
            let geometry = HeightFieldGeometry::new(
                rows as usize,
                columns as usize,
                heights,
                Vector3::new(scale.x, scale.y, scale.z),
            )
            .map_err(failed)?;
            GpuRigidShape::from_heightfield(&geometry).map_err(failed)?
        }
        GpuPrimitiveShape::Polyline { vertices, segments } => GpuRigidShape::Polyline {
            vertices: vertices.into_iter().map(|vertex| vertex.gpu()).collect(),
            segments: segments
                .into_iter()
                .map(|segment| [segment.a, segment.b])
                .collect(),
        },
        GpuPrimitiveShape::TriangleMesh {
            vertices,
            triangles,
        } => GpuRigidShape::TriangleMesh {
            vertices: vertices.into_iter().map(|vertex| vertex.gpu()).collect(),
            triangles: triangles
                .into_iter()
                .map(|triangle| [triangle.a, triangle.b, triangle.c])
                .collect(),
        },
    };
    if shape.bounding_radius().is_none() || !input.mass.is_finite() || input.mass < 0.0 {
        return Err(failed("invalid GPU primitive shape or mass"));
    }
    let inertia = [
        input.principal_inertia.x,
        input.principal_inertia.y,
        input.principal_inertia.z,
    ];
    if inertia
        .iter()
        .any(|value| !value.is_finite() || (input.mass > 0.0 && *value <= 0.0))
        || (input.mass == 0.0 && inertia.iter().any(|value| *value != 0.0))
    {
        return Err(failed(
            "dynamic inertia must be positive; static inertia must be zero",
        ));
    }
    let inverse_mass = if input.mass > 0.0 {
        (1.0 / input.mass) as f32
    } else {
        0.0
    };
    let inverse_inertia = if input.mass > 0.0 {
        [
            (1.0 / inertia[0]) as f32,
            (1.0 / inertia[1]) as f32,
            (1.0 / inertia[2]) as f32,
        ]
    } else {
        [0.0; 3]
    };
    let [x, y, z] = input.center.gpu();
    let [vx, vy, vz] = input.velocity.gpu();
    let [wx, wy, wz] = input.angular_velocity.gpu();
    let state = GpuRigidBodyState {
        position_inverse_mass: [x, y, z, inverse_mass],
        orientation: [
            input.orientation.x as f32,
            input.orientation.y as f32,
            input.orientation.z as f32,
            input.orientation.w as f32,
        ],
        linear_velocity: [vx, vy, vz, 0.0],
        angular_velocity: [wx, wy, wz, 0.0],
        inverse_inertia_sleep: [
            inverse_inertia[0],
            inverse_inertia[1],
            inverse_inertia[2],
            0.0,
        ],
    };
    if !state.is_valid() || (input.mass > 0.0 && inverse_mass == 0.0) {
        return Err(failed("primitive state cannot be represented by GPU f32"));
    }
    Ok((state, shape))
}

fn primitive_contact(
    body_a: u32,
    body_b: Option<u32>,
    contact: tessera_physics::gpu_sphere_contact::GpuSphereContact,
) -> GpuPrimitiveContact {
    GpuPrimitiveContact {
        body_a,
        body_b,
        point: Vec3 {
            x: f64::from(contact.point[0]),
            y: f64::from(contact.point[1]),
            z: f64::from(contact.point[2]),
        },
        normal: Vec3 {
            x: f64::from(contact.normal[0]),
            y: f64::from(contact.normal[1]),
            z: f64::from(contact.normal[2]),
        },
        depth: f64::from(contact.depth_hit[0]),
    }
}

fn primitive_contacts(
    readback: GpuRigidSphereContactReadback,
) -> Result<Vec<GpuPrimitiveContact>, TesseraError> {
    let mut contacts = Vec::new();
    for (index, (pair, contact)) in readback.pairs.into_iter().enumerate() {
        let extra = readback
            .pair_extra
            .get(index)
            .into_iter()
            .flat_map(|slots| slots.iter().copied());
        for point in core::iter::once(contact).chain(extra) {
            if point.is_contact() {
                contacts.push(primitive_contact(pair.a, Some(pair.b), point));
            }
        }
    }
    for (index, contact) in readback.ground.into_iter().enumerate() {
        let body = u32::try_from(index).map_err(failed)?;
        let extra = readback
            .ground_extra
            .get(index)
            .into_iter()
            .flat_map(|slots| slots.iter().copied());
        for point in core::iter::once(contact).chain(extra) {
            if point.is_contact() {
                contacts.push(primitive_contact(body, None, point));
            }
        }
    }
    Ok(contacts)
}

fn gpu_config(
    gravity: Vec3,
    ground_half_extent: f32,
) -> Result<GpuRigidSphereWorldConfig, TesseraError> {
    let config = GpuRigidSphereWorldConfig {
        gravity: gravity.gpu(),
        ground_half_extent: Some(ground_half_extent),
        ..Default::default()
    };
    if !config.is_valid() {
        return Err(failed("invalid GPU world configuration"));
    }
    Ok(config)
}

/// CPU f64 sphere world.
#[derive(Debug, uniffi::Object)]
pub struct SphereWorld {
    world: Mutex<CoreSphereWorld>,
}

#[uniffi::export]
impl SphereWorld {
    /// Construct a sphere world with explicit integration settings.
    #[uniffi::constructor]
    pub fn new(
        bodies: Vec<SphereInput>,
        gravity: Vec3,
        ground_half_extent: f64,
        friction: f64,
        restitution: f64,
        max_substep: f64,
        solver_iterations: u32,
    ) -> Result<Arc<Self>, TesseraError> {
        let params = SphereWorldParams {
            gravity: [gravity.x, gravity.y, gravity.z],
            ground_half_extent,
            friction,
            restitution,
            max_substep,
            solver_iterations: solver_iterations as usize,
            ..Default::default()
        };
        let bodies = bodies
            .into_iter()
            .map(|body| SphereBody {
                center: body.center.nalgebra(),
                velocity: body.velocity.nalgebra(),
                radius: body.radius,
                mass: body.mass,
            })
            .collect();
        let world = CoreSphereWorld::new(bodies, params).map_err(failed)?;
        Ok(Arc::new(Self {
            world: Mutex::new(world),
        }))
    }

    /// Advance the world by seconds.
    pub fn step(&self, dt: f64) -> Result<(), TesseraError> {
        locked(&self.world)?.step(dt).map_err(failed)
    }

    /// Optional dynamic-body linear speed limit in metres per second.
    pub fn max_linear_speed(&self) -> Result<Option<f64>, TesseraError> {
        Ok(locked(&self.world)?.max_linear_speed())
    }

    /// Apply a speed limit before and after contact solving, or disable it.
    pub fn set_max_linear_speed(&self, limit: Option<f64>) -> Result<(), TesseraError> {
        locked(&self.world)?
            .set_max_linear_speed(limit)
            .map_err(failed)
    }

    /// Copy all body states in insertion order.
    pub fn states(&self) -> Result<Vec<SphereState>, TesseraError> {
        Ok(locked(&self.world)?
            .bodies
            .iter()
            .map(|body| SphereState {
                center: body.center.into(),
                velocity: body.velocity.into(),
                radius: body.radius,
                mass: body.mass,
            })
            .collect())
    }

    /// Net force on one body from the last substep.
    pub fn contact_force(&self, index: u32) -> Result<Vec3, TesseraError> {
        locked(&self.world)?
            .body_contact_force(index as usize)
            .map(|[x, y, z]| Vec3 { x, y, z })
            .ok_or_else(|| failed("body index out of range"))
    }

    /// Material assigned to one sphere.
    pub fn body_material(&self, index: u32) -> Result<Material, TesseraError> {
        locked(&self.world)?
            .body_material(index as usize)
            .map(Into::into)
            .ok_or_else(|| failed("body index out of range"))
    }

    /// Assign a validated contact material to one sphere.
    pub fn set_body_material(&self, index: u32, material: Material) -> Result<(), TesseraError> {
        locked(&self.world)?
            .set_body_material(index as usize, material.into())
            .map_err(failed)
    }

    /// Contact material of the finite ground plane.
    pub fn ground_material(&self) -> Result<Material, TesseraError> {
        Ok(locked(&self.world)?.ground_material().into())
    }

    /// Assign a validated material to the finite ground plane.
    pub fn set_ground_material(&self, material: Material) -> Result<(), TesseraError> {
        locked(&self.world)?
            .set_ground_material(material.into())
            .map_err(failed)
    }
}

/// Imported stateless MJCF actuator dynamics.
#[derive(Clone, Debug, uniffi::Enum)]
pub enum MjcfActuatorDynamics {
    /// Geared constant-force motor.
    Motor,
    /// Scalar position servo.
    Position {
        /// Position gain.
        kp: f64,
        /// Velocity damping.
        kv: f64,
    },
    /// Scalar velocity servo.
    Velocity {
        /// Velocity gain.
        kv: f64,
    },
    /// Control-scaled zero-velocity damper.
    Damper {
        /// Damping gain.
        gain: f64,
    },
    /// Fixed-gain general actuator.
    General {
        /// Control gain.
        gain: f64,
        /// Constant, position and velocity bias coefficients.
        bias: Vec<f64>,
        /// Restoring affine servo branch.
        affine: bool,
    },
}

/// MJCF actuator metadata in stable XML declaration order.
#[derive(Clone, Debug, uniffi::Record)]
pub struct MjcfActuatorInfo {
    /// Explicit or generated actuator name.
    pub name: String,
    /// Referenced scalar joint.
    pub joint: String,
    /// Native generalized-coordinate index.
    pub coordinate: u32,
    /// Supported dynamics.
    pub dynamics: MjcfActuatorDynamics,
    /// Motor gear.
    pub gear: f64,
    /// Enabled control clamp; empty when unlimited.
    pub control_range: Vec<f64>,
    /// Enabled force clamp; empty when unlimited.
    pub force_range: Vec<f64>,
}

#[derive(Debug)]
struct ArticulatedInner {
    world: CoreArticulatedWorld,
    actuators: MjcfActuators,
    gpu: Option<GpuContactDevice>,
    resident: Option<(f64, GpuSceneDynamics)>,
}

/// URDF or MJCF articulated world.
#[derive(Debug, uniffi::Object)]
pub struct ArticulatedWorld {
    inner: Mutex<ArticulatedInner>,
    name: String,
    link_names: Vec<String>,
    joint_ranges: Vec<JointRange>,
}

impl ArticulatedWorld {
    fn from_loaded_urdf(loaded: LoadedUrdf) -> Result<Arc<Self>, TesseraError> {
        let joint_ranges = loaded
            .joints
            .into_iter()
            .map(|joint| {
                Ok(JointRange {
                    name: joint.name,
                    start: u32::try_from(joint.dofs.start).map_err(failed)?,
                    end: u32::try_from(joint.dofs.end).map_err(failed)?,
                })
            })
            .collect::<Result<Vec<_>, TesseraError>>()?;
        Ok(Arc::new(Self {
            inner: Mutex::new(ArticulatedInner {
                world: loaded.world,
                actuators: MjcfActuators::default(),
                gpu: None,
                resident: None,
            }),
            name: loaded.robot_name,
            link_names: loaded.link_names,
            joint_ranges,
        }))
    }

    fn from_loaded_mjcf(loaded: LoadedMjcf) -> Result<Arc<Self>, TesseraError> {
        let joint_ranges = loaded
            .joints
            .into_iter()
            .map(|joint| {
                Ok(JointRange {
                    name: joint.name,
                    start: u32::try_from(joint.dofs.start).map_err(failed)?,
                    end: u32::try_from(joint.dofs.end).map_err(failed)?,
                })
            })
            .collect::<Result<Vec<_>, TesseraError>>()?;
        Ok(Arc::new(Self {
            inner: Mutex::new(ArticulatedInner {
                world: loaded.world,
                actuators: loaded.actuators,
                gpu: None,
                resident: None,
            }),
            name: loaded.model_name.unwrap_or_default(),
            link_names: loaded.link_names,
            joint_ranges,
        }))
    }
}

#[uniffi::export]
impl ArticulatedWorld {
    /// Parse a self-contained URDF document.
    #[uniffi::constructor]
    pub fn from_urdf(xml: String, floating_base: bool) -> Result<Arc<Self>, TesseraError> {
        let loaded = load_urdf_str(
            &xml,
            UrdfLoadOptions {
                floating_base,
                ..Default::default()
            },
        )
        .map_err(failed)?;
        Self::from_loaded_urdf(loaded)
    }

    /// Parse URDF with externally supplied convex collision meshes.
    #[uniffi::constructor]
    pub fn from_urdf_with_meshes(
        xml: String,
        floating_base: bool,
        assets: Vec<MeshAsset>,
    ) -> Result<Arc<Self>, TesseraError> {
        let assets = mesh_assets(assets)?;
        let mut resolver =
            |filename: &str, scale: [f64; 3]| resolve_mesh_parts(&assets, filename, scale);
        let loaded = load_urdf_str_with_mesh_resolver(
            &xml,
            UrdfLoadOptions {
                floating_base,
                ..Default::default()
            },
            &mut resolver,
        )
        .map_err(failed)?;
        Self::from_loaded_urdf(loaded)
    }

    /// Parse a self-contained MJCF document.
    #[uniffi::constructor]
    pub fn from_mjcf(xml: String) -> Result<Arc<Self>, TesseraError> {
        let loaded = load_mjcf_str(&xml, MjcfLoadOptions::default()).map_err(failed)?;
        Self::from_loaded_mjcf(loaded)
    }

    /// Parse MJCF with externally supplied convex collision meshes.
    #[uniffi::constructor]
    pub fn from_mjcf_with_meshes(
        xml: String,
        assets: Vec<MeshAsset>,
    ) -> Result<Arc<Self>, TesseraError> {
        let assets = mesh_assets(assets)?;
        let mut resolver =
            |filename: &str, scale: [f64; 3]| resolve_mesh_parts(&assets, filename, scale);
        let loaded =
            load_mjcf_str_with_mesh_resolver(&xml, MjcfLoadOptions::default(), &mut resolver)
                .map_err(failed)?;
        Self::from_loaded_mjcf(loaded)
    }

    /// Model name.
    pub fn name(&self) -> String {
        self.name.clone()
    }

    /// Stable link names.
    pub fn link_names(&self) -> Vec<String> {
        self.link_names.clone()
    }

    /// Joint names and generalized-coordinate ranges.
    pub fn joint_ranges(&self) -> Vec<JointRange> {
        self.joint_ranges.clone()
    }

    /// Imported actuator metadata in XML declaration order.
    pub fn actuator_info(&self) -> Result<Vec<MjcfActuatorInfo>, TesseraError> {
        locked(&self.inner)?
            .actuators
            .entries()
            .iter()
            .map(|a| {
                let dynamics = match a.kind {
                    CoreMjcfActuatorKind::Motor => MjcfActuatorDynamics::Motor,
                    CoreMjcfActuatorKind::Position { kp, kv } => {
                        MjcfActuatorDynamics::Position { kp, kv }
                    }
                    CoreMjcfActuatorKind::Velocity { kv } => MjcfActuatorDynamics::Velocity { kv },
                    CoreMjcfActuatorKind::Damper { gain } => MjcfActuatorDynamics::Damper { gain },
                    CoreMjcfActuatorKind::General { gain, bias, affine } => {
                        MjcfActuatorDynamics::General {
                            gain,
                            bias: bias.to_vec(),
                            affine,
                        }
                    }
                };
                Ok(MjcfActuatorInfo {
                    name: a.name.clone(),
                    joint: a.joint.clone(),
                    coordinate: u32::try_from(a.coordinate).map_err(failed)?,
                    dynamics,
                    gear: a.gear,
                    control_range: a.control_range.map_or_else(Vec::new, |r| r.to_vec()),
                    force_range: a.force_range.map_or_else(Vec::new, |r| r.to_vec()),
                })
            })
            .collect()
    }

    /// Stable imported actuator names, independent of coordinate order.
    pub fn actuator_names(&self) -> Result<Vec<String>, TesseraError> {
        Ok(locked(&self.inner)?
            .actuators
            .entries()
            .iter()
            .map(|a| a.name.clone())
            .collect())
    }

    /// Last accepted actuator-order controls.
    pub fn controls(&self) -> Result<Vec<f64>, TesseraError> {
        Ok(locked(&self.inner)?.actuators.controls().to_vec())
    }

    /// Atomically replace actuator-order controls.
    pub fn set_controls(&self, controls: Vec<f64>) -> Result<(), TesseraError> {
        locked(&self.inner)?
            .actuators
            .set_controls(&controls)
            .map_err(failed)
    }

    /// Advance CPU dynamics using held imported actuator controls.
    pub fn step_controls(&self, dt: f64) -> Result<(), TesseraError> {
        let mut inner = locked(&self.inner)?;
        inner.resident = None;
        let ArticulatedInner {
            world, actuators, ..
        } = &mut *inner;
        actuators.step(world, dt).map_err(failed)
    }

    /// Enable or disable contacts between nonadjacent links.
    pub fn set_self_contacts_enabled(&self, enabled: bool) -> Result<(), TesseraError> {
        locked(&self.inner)?
            .world
            .set_self_contacts_enabled(enabled);
        Ok(())
    }

    /// Whether contacts between nonadjacent links are enabled.
    pub fn self_contacts_enabled(&self) -> Result<bool, TesseraError> {
        Ok(locked(&self.inner)?.world.self_contacts_enabled())
    }

    /// Enable midpoint Coriolis iterations during contact steps.
    pub fn set_implicit_coriolis(&self, enabled: bool) -> Result<(), TesseraError> {
        locked(&self.inner)?.world.set_implicit_coriolis(enabled);
        Ok(())
    }

    /// Whether midpoint Coriolis iterations are enabled.
    pub fn implicit_coriolis(&self) -> Result<bool, TesseraError> {
        Ok(locked(&self.inner)?.world.implicit_coriolis())
    }

    /// Optional motor for one generalized coordinate.
    pub fn joint_motor(&self, slot: u32) -> Result<Option<JointMotor>, TesseraError> {
        let inner = locked(&self.inner)?;
        if slot as usize >= inner.world.positions.len() {
            return Err(failed("joint coordinate out of range"));
        }
        Ok(inner.world.joint_motor(slot as usize).map(Into::into))
    }

    /// Configure or remove a generalized-coordinate motor.
    pub fn set_joint_motor(
        &self,
        slot: u32,
        motor: Option<JointMotor>,
    ) -> Result<(), TesseraError> {
        locked(&self.inner)?
            .world
            .set_joint_motor(slot as usize, motor.map(Into::into))
            .map_err(failed)
    }

    /// Passive spring and damping for one generalized coordinate.
    pub fn joint_passive(&self, slot: u32) -> Result<JointPassive, TesseraError> {
        locked(&self.inner)?
            .world
            .joint_passive(slot as usize)
            .map(Into::into)
            .ok_or_else(|| failed("joint coordinate out of range"))
    }

    /// Set passive spring and damping for one generalized coordinate.
    pub fn set_joint_passive(&self, slot: u32, passive: JointPassive) -> Result<(), TesseraError> {
        locked(&self.inner)?
            .world
            .set_joint_passive(slot as usize, passive.into())
            .map_err(failed)
    }

    /// Nonlinear passive terms of one generalized coordinate.
    pub fn joint_nonlinear_passive(
        &self,
        slot: u32,
    ) -> Result<JointNonlinearPassive, TesseraError> {
        locked(&self.inner)?
            .world
            .joint_nonlinear_passive(slot as usize)
            .map(Into::into)
            .ok_or_else(|| failed("joint coordinate out of range"))
    }

    /// Set nonlinear passive terms of one generalized coordinate.
    pub fn set_joint_nonlinear_passive(
        &self,
        slot: u32,
        passive: JointNonlinearPassive,
    ) -> Result<(), TesseraError> {
        locked(&self.inner)?
            .world
            .set_joint_nonlinear_passive(slot as usize, passive.into())
            .map_err(failed)
    }

    /// Dry friction force or torque for one generalized coordinate.
    pub fn joint_friction(&self, slot: u32) -> Result<f64, TesseraError> {
        locked(&self.inner)?
            .world
            .joint_friction(slot as usize)
            .ok_or_else(|| failed("joint coordinate out of range"))
    }

    /// Set dry friction force or torque for one generalized coordinate.
    pub fn set_joint_friction(&self, slot: u32, friction: f64) -> Result<(), TesseraError> {
        locked(&self.inner)?
            .world
            .set_joint_friction(slot as usize, friction)
            .map_err(failed)
    }

    /// Holonomic couplings between independent generalized coordinates.
    pub fn joint_couplings(&self) -> Result<Vec<JointCoupling>, TesseraError> {
        Ok(locked(&self.inner)?
            .world
            .joint_couplings()
            .iter()
            .copied()
            .map(Into::into)
            .collect())
    }

    /// Replace the holonomic joint couplings.
    pub fn set_joint_couplings(&self, couplings: Vec<JointCoupling>) -> Result<(), TesseraError> {
        locked(&self.inner)?
            .world
            .set_joint_couplings(couplings.into_iter().map(Into::into).collect())
            .map_err(failed)
    }

    /// Polynomial joint equalities, including those loaded from MJCF.
    pub fn joint_polynomial_couplings(&self) -> Result<Vec<JointPolynomialCoupling>, TesseraError> {
        Ok(locked(&self.inner)?
            .world
            .joint_polynomial_couplings()
            .iter()
            .copied()
            .map(Into::into)
            .collect())
    }

    /// Replace the polynomial joint equalities.
    pub fn set_joint_polynomial_couplings(
        &self,
        couplings: Vec<JointPolynomialCoupling>,
    ) -> Result<(), TesseraError> {
        let couplings = couplings
            .into_iter()
            .map(TryInto::try_into)
            .collect::<Result<Vec<_>, _>>()?;
        locked(&self.inner)?
            .world
            .set_joint_polynomial_couplings(couplings)
            .map_err(failed)
    }

    /// Ball constraints between articulation links or a link and the world.
    pub fn link_point_constraints(&self) -> Result<Vec<LinkPointConstraint>, TesseraError> {
        Ok(locked(&self.inner)?
            .world
            .link_point_constraints()
            .iter()
            .copied()
            .map(Into::into)
            .collect())
    }

    /// Replace ball constraints between articulation links or the world.
    pub fn set_link_point_constraints(
        &self,
        constraints: Vec<LinkPointConstraint>,
    ) -> Result<(), TesseraError> {
        locked(&self.inner)?
            .world
            .set_link_point_constraints(constraints.into_iter().map(Into::into).collect())
            .map_err(failed)
    }

    /// Fixed-frame constraints between articulation links or the world.
    pub fn link_fixed_constraints(&self) -> Result<Vec<LinkFixedConstraint>, TesseraError> {
        Ok(locked(&self.inner)?
            .world
            .link_fixed_constraints()
            .iter()
            .copied()
            .map(Into::into)
            .collect())
    }

    /// Replace fixed-frame constraints between articulation links or the world.
    pub fn set_link_fixed_constraints(
        &self,
        constraints: Vec<LinkFixedConstraint>,
    ) -> Result<(), TesseraError> {
        let constraints = constraints
            .into_iter()
            .map(TryInto::try_into)
            .collect::<Result<Vec<_>, _>>()?;
        locked(&self.inner)?
            .world
            .set_link_fixed_constraints(constraints)
            .map_err(failed)
    }

    /// Add an independent scene body with one or more colliders.
    pub fn add_scene_body(&self, input: SceneBodyInput) -> Result<u32, TesseraError> {
        let body = input.try_into()?;
        Ok(locked(&self.inner)?.world.add_scene_body(body) as u32)
    }

    /// Add an independent dynamic or static scene sphere.
    pub fn add_scene_sphere(
        &self,
        pose: LinkPose,
        radius: f64,
        mass: f64,
    ) -> Result<u32, TesseraError> {
        let body = scene_sphere_body(pose, radius, mass)?;
        Ok(locked(&self.inner)?.world.add_scene_body(body) as u32)
    }

    /// Read the state of an independent scene body.
    pub fn scene_body_state(&self, body: u32) -> Result<SceneBodyState, TesseraError> {
        locked(&self.inner)?
            .world
            .scene_bodies
            .get(body as usize)
            .map(Into::into)
            .ok_or_else(|| failed("scene body index out of range"))
    }

    /// Teleport a scene body and invalidate cached contacts.
    pub fn set_scene_body_pose(&self, body: u32, pose: LinkPose) -> Result<(), TesseraError> {
        let pose = link_pose_isometry(pose)?;
        locked(&self.inner)?
            .world
            .set_scene_body_pose(body as usize, pose)
            .map_err(failed)
    }

    /// Prescribe a zero-mass scene body's motion, or stop it with two absent velocities.
    pub fn set_scene_body_kinematic_motion(
        &self,
        body: u32,
        linear: Option<Vec3>,
        angular: Option<Vec3>,
    ) -> Result<(), TesseraError> {
        let motion = match (linear, angular) {
            (Some(linear), Some(angular)) => Some((linear.nalgebra(), angular.nalgebra())),
            (None, None) => None,
            _ => {
                return Err(failed(
                    "linear and angular velocities must both be present or absent",
                ));
            }
        };
        locked(&self.inner)?
            .world
            .set_scene_body_kinematic_motion(body as usize, motion)
            .map_err(failed)
    }

    /// Set a dynamic scene body's world-space linear and angular velocities.
    pub fn set_scene_body_velocity(
        &self,
        body: u32,
        linear: Vec3,
        angular: Vec3,
    ) -> Result<(), TesseraError> {
        locked(&self.inner)?
            .world
            .set_scene_body_velocity(body as usize, linear.nalgebra(), angular.nalgebra())
            .map_err(failed)
    }

    /// Set a persistent world-frame force at a scene body origin.
    pub fn set_scene_body_force(&self, body: u32, force: Vec3) -> Result<(), TesseraError> {
        let force = Vector3::new(force.x, force.y, force.z);
        if !force.iter().all(|value| value.is_finite()) {
            return Err(failed("scene body force must be finite"));
        }
        let mut inner = locked(&self.inner)?;
        let scene = inner
            .world
            .scene_bodies
            .get_mut(body as usize)
            .ok_or_else(|| failed("scene body index out of range"))?;
        scene.force = force;
        Ok(())
    }

    /// Set persistent world-frame force and torque on a dynamic scene body.
    pub fn set_scene_body_wrench(
        &self,
        body: u32,
        force: Vec3,
        torque: Vec3,
    ) -> Result<(), TesseraError> {
        locked(&self.inner)?
            .world
            .set_scene_body_wrench(body as usize, force.nalgebra(), torque.nalgebra())
            .map_err(failed)
    }

    /// Apply world-frame impulses; angular impulse is about the body origin.
    pub fn apply_scene_body_impulse(
        &self,
        body: u32,
        linear: Vec3,
        angular: Vec3,
    ) -> Result<(), TesseraError> {
        locked(&self.inner)?
            .world
            .apply_scene_body_impulse(body as usize, linear.nalgebra(), angular.nalgebra())
            .map_err(failed)
    }

    /// Remove a scene body and renumber constraints referring to later bodies.
    pub fn remove_scene_body(&self, body: u32) -> Result<bool, TesseraError> {
        Ok(locked(&self.inner)?.world.remove_scene_body(body as usize))
    }

    /// Ball constraints between articulation links and scene bodies.
    pub fn link_scene_point_constraints(
        &self,
    ) -> Result<Vec<LinkScenePointConstraint>, TesseraError> {
        Ok(locked(&self.inner)?
            .world
            .link_scene_point_constraints()
            .iter()
            .copied()
            .map(Into::into)
            .collect())
    }

    /// Replace ball constraints between articulation links and scene bodies.
    pub fn set_link_scene_point_constraints(
        &self,
        constraints: Vec<LinkScenePointConstraint>,
    ) -> Result<(), TesseraError> {
        locked(&self.inner)?
            .world
            .set_link_scene_point_constraints(constraints.into_iter().map(Into::into).collect())
            .map_err(failed)
    }

    /// Fixed-frame constraints between articulation links and scene bodies.
    pub fn link_scene_fixed_constraints(
        &self,
    ) -> Result<Vec<LinkSceneFixedConstraint>, TesseraError> {
        Ok(locked(&self.inner)?
            .world
            .link_scene_fixed_constraints()
            .iter()
            .copied()
            .map(Into::into)
            .collect())
    }

    /// Replace fixed-frame constraints between articulation links and scene bodies.
    pub fn set_link_scene_fixed_constraints(
        &self,
        constraints: Vec<LinkSceneFixedConstraint>,
    ) -> Result<(), TesseraError> {
        let constraints = constraints
            .into_iter()
            .map(TryInto::try_into)
            .collect::<Result<Vec<_>, _>>()?;
        locked(&self.inner)?
            .world
            .set_link_scene_fixed_constraints(constraints)
            .map_err(failed)
    }

    /// Reflected rotor inertia on one articulation edge, in local axis order.
    pub fn joint_armature(&self, edge: u32) -> Result<Vec<f64>, TesseraError> {
        locked(&self.inner)?
            .world
            .articulation
            .joint_armature(edge as usize)
            .map(<[f64]>::to_vec)
            .ok_or_else(|| failed("joint edge out of range"))
    }

    /// Set reflected rotor inertia on one articulation edge.
    pub fn set_joint_armature(&self, edge: u32, armatures: Vec<f64>) -> Result<(), TesseraError> {
        locked(&self.inner)?
            .world
            .articulation
            .set_joint_armature(edge as usize, &armatures)
            .map_err(failed)
    }

    /// Persistent world-frame force, torque, and gravity multiplier of a link.
    pub fn link_load(&self, link: u32) -> Result<LinkLoad, TesseraError> {
        locked(&self.inner)?
            .world
            .link_external_load(link as usize)
            .map(Into::into)
            .ok_or_else(|| failed("link index out of range"))
    }

    /// Apply a persistent world-frame force at a link COM and torque about it.
    pub fn set_link_external_wrench(
        &self,
        link: u32,
        force: Vec3,
        torque: Vec3,
    ) -> Result<(), TesseraError> {
        locked(&self.inner)?
            .world
            .set_link_external_wrench(link as usize, force.nalgebra(), torque.nalgebra())
            .map_err(failed)
    }

    /// Set one link's gravity multiplier without changing its mass.
    pub fn set_link_gravity_scale(&self, link: u32, scale: f64) -> Result<(), TesseraError> {
        locked(&self.inner)?
            .world
            .set_link_gravity_scale(link as usize, scale)
            .map_err(failed)
    }

    /// Material of one articulated-link collider.
    pub fn link_material(
        &self,
        kind: LinkColliderKind,
        index: u32,
    ) -> Result<Material, TesseraError> {
        let inner = locked(&self.inner)?;
        let index = index as usize;
        let material = match kind {
            LinkColliderKind::Sphere => inner.world.link_sphere_material(index),
            LinkColliderKind::GroundPoint => inner.world.ground_point_material(index),
            LinkColliderKind::Box => inner.world.link_box_material(index),
            LinkColliderKind::Cylinder => inner.world.link_cylinder_material(index),
            LinkColliderKind::Convex => inner.world.link_convex_material(index),
        };
        material
            .map(Into::into)
            .ok_or_else(|| failed("link collider index out of range"))
    }

    /// Assign a validated material to one articulated-link collider.
    pub fn set_link_material(
        &self,
        kind: LinkColliderKind,
        index: u32,
        material: Material,
    ) -> Result<(), TesseraError> {
        let mut inner = locked(&self.inner)?;
        let material = material.into();
        let result = match kind {
            LinkColliderKind::Sphere => inner
                .world
                .set_link_sphere_material(index as usize, material),
            LinkColliderKind::GroundPoint => inner
                .world
                .set_ground_point_material(index as usize, material),
            LinkColliderKind::Box => inner.world.set_link_box_material(index as usize, material),
            LinkColliderKind::Cylinder => inner
                .world
                .set_link_cylinder_material(index as usize, material),
            LinkColliderKind::Convex => inner
                .world
                .set_link_convex_material(index as usize, material),
        };
        result.map_err(failed)
    }

    /// Material of one scene-body collider.
    pub fn scene_collider_material(
        &self,
        body_index: u32,
        collider_index: u32,
    ) -> Result<Material, TesseraError> {
        locked(&self.inner)?
            .world
            .scene_collider_material(body_index as usize, collider_index as usize)
            .map(Into::into)
            .ok_or_else(|| failed("scene collider index out of range"))
    }

    /// Assign a validated material to a scene-body collider.
    pub fn set_scene_collider_material(
        &self,
        body_index: u32,
        collider_index: u32,
        material: Material,
    ) -> Result<(), TesseraError> {
        locked(&self.inner)?
            .world
            .set_scene_collider_material(
                body_index as usize,
                collider_index as usize,
                material.into(),
            )
            .map_err(failed)
    }

    /// Contact material of the finite ground plane.
    pub fn ground_material(&self) -> Result<Material, TesseraError> {
        Ok(locked(&self.inner)?.world.ground_material().into())
    }

    /// Assign a validated contact material to the finite ground plane.
    pub fn set_ground_material(&self, material: Material) -> Result<(), TesseraError> {
        locked(&self.inner)?
            .world
            .set_ground_material(material.into())
            .map_err(failed)
    }

    /// Generalized positions.
    pub fn positions(&self) -> Result<Vec<f64>, TesseraError> {
        Ok(locked(&self.inner)?
            .world
            .positions
            .iter()
            .copied()
            .collect())
    }

    /// Read the current native configuration without changing contacts or GPU state.
    pub fn kinematics_state(&self) -> Result<IkState, TesseraError> {
        let inner = locked(&self.inner)?;
        let world = &inner.world;
        let orientations = world.gpu_spherical_state().map(|joints| {
            let prefix = if world.floating { 6 } else { 0 };
            let mut values = vec![None; world.articulation.link_count() - 1];
            for joint in joints {
                for (edge, value) in values.iter_mut().enumerate() {
                    if world
                        .articulation
                        .joint_coordinate_range(edge)
                        .is_some_and(|range| {
                            range.len() == 3 && range.start + prefix == joint.velocity_slot
                        })
                    {
                        *value = Some(joint.orientation);
                        break;
                    }
                }
            }
            values
        });
        Ok(CoreIkState {
            root_pose: world.root_pose,
            positions: world.positions.iter().copied().collect(),
            orientations,
        }
        .into())
    }

    /// Pure CPU forward kinematics at a supplied configuration.
    pub fn forward_kinematics(&self, state: IkState) -> Result<Vec<LinkPose>, TesseraError> {
        let state = state.native()?;
        let inner = locked(&self.inner)?;
        Ok(core_forward_kinematics(&inner.world.articulation, &state)
            .map_err(failed)?
            .links
            .into_iter()
            .map(isometry_link_pose)
            .collect())
    }

    /// Pure CPU IK with reduced mimic coordinates, limits, and scalar equalities.
    /// This does not teleport the robot, clear contacts, or change resident state.
    pub fn inverse_kinematics(
        &self,
        initial: IkState,
        target: IkTarget,
        config: IkConfig,
    ) -> Result<IkResult, TesseraError> {
        let initial = initial.native()?;
        let target = CoreIkTarget {
            link: target.link as usize,
            pose: link_pose_isometry(target.pose)?,
            local_point: target.local_point.nalgebra(),
            constrained_axes: target
                .constrained_axes
                .try_into()
                .map_err(|_| failed("IK constrained_axes must contain exactly six flags"))?,
        };
        let config = CoreIkConfig {
            max_iterations: config.max_iterations as usize,
            damping: config.damping,
            position_tolerance: config.position_tolerance,
            rotation_tolerance: config.rotation_tolerance,
            coupling_tolerance: config.coupling_tolerance,
            dofs: config
                .dofs
                .map(|dofs| dofs.into_iter().map(|slot| slot as usize).collect()),
        };
        let inner = locked(&self.inner)?;
        let world = &inner.world;
        let couplings = world
            .joint_couplings()
            .iter()
            .map(|coupling| CoreJointPolynomialCoupling {
                follower: coupling.follower,
                source: Some(coupling.source),
                coefficients: [coupling.offset, coupling.multiplier, 0.0, 0.0, 0.0],
                follower_reference: 0.0,
                source_reference: 0.0,
            })
            .chain(world.joint_polynomial_couplings().iter().copied())
            .collect::<Vec<_>>();
        let result = core_inverse_kinematics(
            &world.articulation,
            &initial,
            world.floating,
            &target,
            &config,
            &couplings,
        )
        .map_err(failed)?;
        Ok(IkResult {
            state: result.state.into(),
            converged: result.converged,
            iterations: result.iterations as u32,
            residual: result.residual.to_vec(),
            coupling_residual: result.coupling_residual,
        })
    }

    /// Set all generalized positions and discard cached contacts.
    pub fn set_positions(&self, values: Vec<f64>) -> Result<(), TesseraError> {
        let mut inner = locked(&self.inner)?;
        if values.len() != inner.world.positions.len() || values.iter().any(|x| !x.is_finite()) {
            return Err(failed("positions must be finite and match the joint count"));
        }
        inner
            .world
            .positions
            .as_mut_slice()
            .copy_from_slice(&values);
        inner.world.clear_contact_cache();
        Ok(())
    }

    /// Generalized velocities.
    pub fn velocities(&self) -> Result<Vec<f64>, TesseraError> {
        Ok(locked(&self.inner)?
            .world
            .velocities
            .iter()
            .copied()
            .collect())
    }

    /// Contact wrench on one link from the last integration substep.
    pub fn link_contact_wrench(&self, link: u32) -> Result<LinkContactWrench, TesseraError> {
        let inner = locked(&self.inner)?;
        let index = link as usize;
        let force = inner
            .world
            .contact_forces
            .get(index)
            .ok_or_else(|| failed("link index out of range"))?;
        let torque = &inner.world.contact_torques[index];
        Ok(LinkContactWrench {
            force: (*force).into(),
            torque: (*torque).into(),
        })
    }

    /// Set all generalized velocities and discard cached contacts.
    pub fn set_velocities(&self, values: Vec<f64>) -> Result<(), TesseraError> {
        let mut inner = locked(&self.inner)?;
        if values.len() != inner.world.velocities.len() || values.iter().any(|x| !x.is_finite()) {
            return Err(failed(
                "velocities must be finite and match the joint count",
            ));
        }
        inner
            .world
            .velocities
            .as_mut_slice()
            .copy_from_slice(&values);
        inner.world.clear_contact_cache();
        Ok(())
    }

    /// Advance CPU dynamics and contact.
    pub fn step(&self, dt: f64, torques: Vec<f64>) -> Result<(), TesseraError> {
        let mut inner = locked(&self.inner)?;
        inner.resident = None;
        inner.world.step(dt, &torques).map_err(failed)
    }

    /// Advance using GPU collision and impulse kernels.
    pub fn step_gpu(&self, dt: f64, torques: Vec<f64>) -> Result<(), TesseraError> {
        let mut inner = locked(&self.inner)?;
        inner.resident = None;
        if inner.gpu.is_none() {
            inner.gpu = Some(GpuContactDevice::new().map_err(failed)?);
        }
        let ArticulatedInner { world, gpu, .. } = &mut *inner;
        let context = gpu.as_ref().ok_or_else(|| failed("GPU unavailable"))?;
        world.step_gpu(dt, &torques, context).map_err(failed)
    }

    /// Advance with GPU mass inversion and contact impulse solves.
    pub fn step_gpu_device_mass(&self, dt: f64, torques: Vec<f64>) -> Result<(), TesseraError> {
        let mut inner = locked(&self.inner)?;
        inner.resident = None;
        if inner.gpu.is_none() {
            inner.gpu = Some(GpuContactDevice::new().map_err(failed)?);
        }
        let ArticulatedInner { world, gpu, .. } = &mut *inner;
        let context = gpu.as_ref().ok_or_else(|| failed("GPU unavailable"))?;
        world
            .step_gpu_device_mass(dt, &torques, context)
            .map_err(failed)
    }

    /// Advance a reusable resident robot/scene batch for fixed-timestep steps.
    /// Readback synchronizes robot and scene states before returning. Repeated
    /// calls with the same timestep reuse the batch. Reset after geometry,
    /// material, solver, state, or prescribed motion edits. External state edits
    /// are rejected rather than overwritten by stale device state.
    pub fn step_gpu_resident(
        &self,
        timestep: f64,
        torques: Vec<f64>,
        steps: u32,
    ) -> Result<(), TesseraError> {
        self.step_gpu_resident_with_contacts(timestep, torques, steps)
            .map(|_| ())
    }

    /// Advance the resident solver and return final-substep scene-body contacts.
    /// Reports follow original scene-body order.
    pub fn step_gpu_resident_with_contacts(
        &self,
        timestep: f64,
        torques: Vec<f64>,
        steps: u32,
    ) -> Result<Vec<SceneContactReport>, TesseraError> {
        if !timestep.is_finite() || timestep <= 0.0 || steps == 0 {
            return Err(failed("resident timestep and step count must be positive"));
        }
        let mut inner = locked(&self.inner)?;
        if inner.gpu.is_none() {
            inner.gpu = Some(GpuContactDevice::new().map_err(failed)?);
        }
        let ArticulatedInner {
            world,
            gpu,
            resident,
            ..
        } = &mut *inner;
        if !resident.as_ref().is_some_and(|(dt, _)| *dt == timestep) {
            let context = gpu.as_ref().ok_or_else(|| failed("GPU unavailable"))?;
            let session = GpuSceneDynamics::new(world, context, timestep).map_err(failed)?;
            *resident = Some((timestep, session));
        }
        let (_, session) = resident
            .as_mut()
            .ok_or_else(|| failed("resident GPU unavailable"))?;
        let diagnostics = session
            .step(world, &torques, steps as usize)
            .map_err(failed)?;
        Ok(scene_contact_reports(diagnostics, timestep))
    }

    /// Latest synchronized sleep state of one scene body.
    pub fn scene_body_is_sleeping(&self, body: u32) -> Result<bool, TesseraError> {
        locked(&self.inner)?
            .world
            .scene_body_is_sleeping(body as usize)
            .ok_or_else(|| failed("scene body index out of range"))
    }

    /// Upload edited static sphere, box, and convex poses to the current session.
    /// Geometry, materials, and prescribed body state must remain unchanged.
    pub fn update_gpu_resident_static_scene_poses(&self) -> Result<(), TesseraError> {
        let mut inner = locked(&self.inner)?;
        let ArticulatedInner {
            world, resident, ..
        } = &mut *inner;
        let (_, session) = resident
            .as_mut()
            .ok_or_else(|| failed("resident session is not initialized"))?;
        session.update_static_scene_poses(world).map_err(failed)
    }

    /// Enable automatic scene-body sleep on an initialized resident session.
    /// Call after a resident step; reset or timestep changes discard this setting.
    pub fn enable_gpu_resident_scene_sleep(&self) -> Result<(), TesseraError> {
        let mut inner = locked(&self.inner)?;
        let ArticulatedInner {
            world, resident, ..
        } = &mut *inner;
        let (_, session) = resident.as_mut().ok_or_else(|| {
            failed("initialize the resident session with step_gpu_resident first")
        })?;
        session.enable_scene_sleep(world).map_err(failed)
    }

    /// Release resident buffers before changing their persistent configuration.
    /// The next resident step rebuilds from the current host state.
    pub fn reset_gpu_resident(&self) -> Result<(), TesseraError> {
        locked(&self.inner)?.resident = None;
        Ok(())
    }

    /// Solve the current generalized acceleration on GPU without advancing state.
    pub fn generalized_acceleration_gpu(
        &self,
        torques: Vec<f64>,
    ) -> Result<Vec<f64>, TesseraError> {
        let mut inner = locked(&self.inner)?;
        if inner.gpu.is_none() {
            inner.gpu = Some(GpuContactDevice::new().map_err(failed)?);
        }
        let ArticulatedInner { world, gpu, .. } = &mut *inner;
        let system = world
            .generalized_mass_assembly_system(&torques)
            .map_err(failed)?;
        if system.force.is_empty() {
            return Ok(Vec::new());
        }
        let context = gpu.as_ref().ok_or_else(|| failed("GPU unavailable"))?;
        let batch =
            GpuArticulatedMassAssemblyBatch::new(context.device(), context.queue(), &[system])
                .map_err(failed)?;
        batch.submit();
        batch
            .readback()
            .map_err(failed)?
            .into_iter()
            .next()
            .map(|values| values.iter().copied().collect())
            .ok_or_else(|| failed("GPU mass solve returned no result"))
    }

    /// World-space poses in link order.
    pub fn link_poses(&self) -> Result<Vec<LinkPose>, TesseraError> {
        locked(&self.inner)?
            .world
            .link_poses()
            .map_err(failed)
            .map(|poses| {
                poses
                    .into_iter()
                    .map(|pose| {
                        let q = pose.rotation.quaternion();
                        LinkPose {
                            position: pose.translation.vector.into(),
                            orientation: Quaternion {
                                x: q.i,
                                y: q.j,
                                z: q.k,
                                w: q.w,
                            },
                        }
                    })
                    .collect()
            })
    }

    /// World-space velocities at all link origins in model order.
    pub fn link_twists(&self) -> Result<Vec<LinkTwist>, TesseraError> {
        locked(&self.inner)?
            .world
            .link_twists()
            .map_err(failed)
            .map(|twists| twists.into_iter().map(Into::into).collect())
    }

    /// World-space velocity at a point expressed in one link frame.
    pub fn link_point_twist(
        &self,
        link: u32,
        local_point: Vec3,
    ) -> Result<LinkTwist, TesseraError> {
        locked(&self.inner)?
            .world
            .link_point_twist(link as usize, local_point.nalgebra())
            .map(Into::into)
            .map_err(failed)
    }

    /// Kinematic acceleration at a link-frame point from generalized qdd.
    pub fn link_point_acceleration(
        &self,
        link: u32,
        local_point: Vec3,
        generalized_acceleration: Vec<f64>,
    ) -> Result<LinkAcceleration, TesseraError> {
        locked(&self.inner)?
            .world
            .link_point_acceleration(
                link as usize,
                local_point.nalgebra(),
                &generalized_acceleration,
            )
            .map(Into::into)
            .map_err(failed)
    }

    /// Ideal link-frame inertial sensor values from generalized qdd.
    pub fn link_imu(
        &self,
        link: u32,
        local_point: Vec3,
        generalized_acceleration: Vec<f64>,
    ) -> Result<LinkImuReading, TesseraError> {
        locked(&self.inner)?
            .world
            .link_imu(
                link as usize,
                local_point.nalgebra(),
                &generalized_acceleration,
            )
            .map(Into::into)
            .map_err(failed)
    }
}

#[derive(Debug)]
struct ArticulatedBatchInner {
    batch: CoreArticulatedBatch,
    gpu: Option<GpuContactDevice>,
    device_mass_cache: ArticulatedDeviceMassStepCache,
    resident: Option<(f64, GpuSceneDynamicsBatch)>,
}

/// Independent articulated environments with packed GPU contact solves.
#[derive(Debug, uniffi::Object)]
pub struct ArticulatedBatch {
    inner: Mutex<ArticulatedBatchInner>,
    ids: Vec<EnvironmentId>,
    info: Vec<BatchEnvironmentInfo>,
}

impl ArticulatedBatch {
    fn id(&self, index: u32) -> Result<EnvironmentId, TesseraError> {
        self.ids
            .get(index as usize)
            .copied()
            .ok_or_else(|| failed("environment index out of range"))
    }
}

#[uniffi::export]
impl ArticulatedBatch {
    /// Load independent URDF and MJCF environments in insertion order.
    #[uniffi::constructor]
    pub fn new(models: Vec<BatchModel>) -> Result<Arc<Self>, TesseraError> {
        if models.len() > u32::MAX as usize {
            return Err(failed("too many environments"));
        }
        let mut batch = CoreArticulatedBatch::new();
        let mut ids = Vec::with_capacity(models.len());
        let mut info = Vec::with_capacity(models.len());
        for model in models {
            let assets = mesh_assets(model.meshes)?;
            let mut resolver =
                |filename: &str, scale: [f64; 3]| resolve_mesh_parts(&assets, filename, scale);
            let (world, metadata) = match model.format {
                ModelFormat::Urdf => {
                    let loaded = load_urdf_str_with_mesh_resolver(
                        &model.xml,
                        UrdfLoadOptions {
                            floating_base: model.floating_base,
                            ..Default::default()
                        },
                        &mut resolver,
                    )
                    .map_err(failed)?;
                    let ranges = loaded
                        .joints
                        .iter()
                        .map(|joint| {
                            Ok(JointRange {
                                name: joint.name.clone(),
                                start: u32::try_from(joint.dofs.start).map_err(failed)?,
                                end: u32::try_from(joint.dofs.end).map_err(failed)?,
                            })
                        })
                        .collect::<Result<Vec<_>, TesseraError>>()?;
                    (
                        loaded.world,
                        BatchEnvironmentInfo {
                            name: loaded.robot_name,
                            link_names: loaded.link_names,
                            joint_ranges: ranges,
                        },
                    )
                }
                ModelFormat::Mjcf => {
                    let loaded = load_mjcf_str_with_mesh_resolver(
                        &model.xml,
                        MjcfLoadOptions::default(),
                        &mut resolver,
                    )
                    .map_err(failed)?;
                    let ranges = loaded
                        .joints
                        .iter()
                        .map(|joint| {
                            Ok(JointRange {
                                name: joint.name.clone(),
                                start: u32::try_from(joint.dofs.start).map_err(failed)?,
                                end: u32::try_from(joint.dofs.end).map_err(failed)?,
                            })
                        })
                        .collect::<Result<Vec<_>, TesseraError>>()?;
                    (
                        loaded.world,
                        BatchEnvironmentInfo {
                            name: loaded.model_name.unwrap_or_default(),
                            link_names: loaded.link_names,
                            joint_ranges: ranges,
                        },
                    )
                }
            };
            ids.push(batch.add_environment(world));
            info.push(metadata);
        }
        Ok(Arc::new(Self {
            inner: Mutex::new(ArticulatedBatchInner {
                batch,
                gpu: None,
                device_mass_cache: ArticulatedDeviceMassStepCache::new(),
                resident: None,
            }),
            ids,
            info,
        }))
    }

    /// Number of independent environments.
    pub fn len(&self) -> u32 {
        self.ids.len() as u32
    }

    /// Whether the batch has no environments.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Stable names and coordinate ranges for one environment.
    pub fn environment_info(&self, index: u32) -> Result<BatchEnvironmentInfo, TesseraError> {
        self.info
            .get(index as usize)
            .cloned()
            .ok_or_else(|| failed("environment index out of range"))
    }

    /// Generalized positions for one environment.
    pub fn positions(&self, index: u32) -> Result<Vec<f64>, TesseraError> {
        let id = self.id(index)?;
        let inner = locked(&self.inner)?;
        let world = inner
            .batch
            .environment(id)
            .ok_or_else(|| failed("environment index out of range"))?;
        Ok(world.positions.iter().copied().collect())
    }

    /// Set all generalized positions and discard cached contact impulses.
    pub fn set_positions(&self, index: u32, values: Vec<f64>) -> Result<(), TesseraError> {
        let id = self.id(index)?;
        let mut inner = locked(&self.inner)?;
        let world = inner
            .batch
            .environment_mut(id)
            .ok_or_else(|| failed("environment index out of range"))?;
        if values.len() != world.positions.len() || values.iter().any(|x| !x.is_finite()) {
            return Err(failed("positions must be finite and match the joint count"));
        }
        world.positions.as_mut_slice().copy_from_slice(&values);
        world.clear_contact_cache();
        Ok(())
    }

    /// Generalized velocities for one environment.
    pub fn velocities(&self, index: u32) -> Result<Vec<f64>, TesseraError> {
        let id = self.id(index)?;
        let inner = locked(&self.inner)?;
        let world = inner
            .batch
            .environment(id)
            .ok_or_else(|| failed("environment index out of range"))?;
        Ok(world.velocities.iter().copied().collect())
    }

    /// Set all generalized velocities and discard cached contact impulses.
    pub fn set_velocities(&self, index: u32, values: Vec<f64>) -> Result<(), TesseraError> {
        let id = self.id(index)?;
        let mut inner = locked(&self.inner)?;
        let world = inner
            .batch
            .environment_mut(id)
            .ok_or_else(|| failed("environment index out of range"))?;
        if values.len() != world.velocities.len() || values.iter().any(|x| !x.is_finite()) {
            return Err(failed(
                "velocities must be finite and match the joint count",
            ));
        }
        world.velocities.as_mut_slice().copy_from_slice(&values);
        world.clear_contact_cache();
        Ok(())
    }

    /// Enable or disable contacts between nonadjacent links in one environment.
    pub fn set_self_contacts_enabled(&self, index: u32, enabled: bool) -> Result<(), TesseraError> {
        let id = self.id(index)?;
        locked(&self.inner)?
            .batch
            .environment_mut(id)
            .ok_or_else(|| failed("environment index out of range"))?
            .set_self_contacts_enabled(enabled);
        Ok(())
    }

    /// Whether contacts between nonadjacent links are enabled in one environment.
    pub fn self_contacts_enabled(&self, index: u32) -> Result<bool, TesseraError> {
        let id = self.id(index)?;
        Ok(locked(&self.inner)?
            .batch
            .environment(id)
            .ok_or_else(|| failed("environment index out of range"))?
            .self_contacts_enabled())
    }

    /// Enable midpoint Coriolis iterations in one environment.
    pub fn set_implicit_coriolis(&self, index: u32, enabled: bool) -> Result<(), TesseraError> {
        let id = self.id(index)?;
        locked(&self.inner)?
            .batch
            .environment_mut(id)
            .ok_or_else(|| failed("environment index out of range"))?
            .set_implicit_coriolis(enabled);
        Ok(())
    }

    /// Whether midpoint Coriolis iterations are enabled in one environment.
    pub fn implicit_coriolis(&self, index: u32) -> Result<bool, TesseraError> {
        let id = self.id(index)?;
        Ok(locked(&self.inner)?
            .batch
            .environment(id)
            .ok_or_else(|| failed("environment index out of range"))?
            .implicit_coriolis())
    }

    /// Configure or remove one environment's generalized-coordinate motor.
    pub fn set_joint_motor(
        &self,
        index: u32,
        slot: u32,
        motor: Option<JointMotor>,
    ) -> Result<(), TesseraError> {
        let id = self.id(index)?;
        let mut inner = locked(&self.inner)?;
        inner
            .batch
            .environment_mut(id)
            .ok_or_else(|| failed("environment index out of range"))?
            .set_joint_motor(slot as usize, motor.map(Into::into))
            .map_err(failed)
    }

    /// Passive spring and damping in one environment.
    pub fn joint_passive(&self, index: u32, slot: u32) -> Result<JointPassive, TesseraError> {
        let id = self.id(index)?;
        locked(&self.inner)?
            .batch
            .environment(id)
            .ok_or_else(|| failed("environment index out of range"))?
            .joint_passive(slot as usize)
            .map(Into::into)
            .ok_or_else(|| failed("joint coordinate out of range"))
    }

    /// Set passive spring and damping in one environment.
    pub fn set_joint_passive(
        &self,
        index: u32,
        slot: u32,
        passive: JointPassive,
    ) -> Result<(), TesseraError> {
        let id = self.id(index)?;
        locked(&self.inner)?
            .batch
            .environment_mut(id)
            .ok_or_else(|| failed("environment index out of range"))?
            .set_joint_passive(slot as usize, passive.into())
            .map_err(failed)
    }

    /// Nonlinear passive terms in one environment.
    pub fn joint_nonlinear_passive(
        &self,
        index: u32,
        slot: u32,
    ) -> Result<JointNonlinearPassive, TesseraError> {
        let id = self.id(index)?;
        locked(&self.inner)?
            .batch
            .environment(id)
            .ok_or_else(|| failed("environment index out of range"))?
            .joint_nonlinear_passive(slot as usize)
            .map(Into::into)
            .ok_or_else(|| failed("joint coordinate out of range"))
    }

    /// Set nonlinear passive terms in one environment.
    pub fn set_joint_nonlinear_passive(
        &self,
        index: u32,
        slot: u32,
        passive: JointNonlinearPassive,
    ) -> Result<(), TesseraError> {
        let id = self.id(index)?;
        locked(&self.inner)?
            .batch
            .environment_mut(id)
            .ok_or_else(|| failed("environment index out of range"))?
            .set_joint_nonlinear_passive(slot as usize, passive.into())
            .map_err(failed)
    }

    /// Dry friction force or torque in one environment.
    pub fn joint_friction(&self, index: u32, slot: u32) -> Result<f64, TesseraError> {
        let id = self.id(index)?;
        locked(&self.inner)?
            .batch
            .environment(id)
            .ok_or_else(|| failed("environment index out of range"))?
            .joint_friction(slot as usize)
            .ok_or_else(|| failed("joint coordinate out of range"))
    }

    /// Set dry friction force or torque in one environment.
    pub fn set_joint_friction(
        &self,
        index: u32,
        slot: u32,
        friction: f64,
    ) -> Result<(), TesseraError> {
        let id = self.id(index)?;
        locked(&self.inner)?
            .batch
            .environment_mut(id)
            .ok_or_else(|| failed("environment index out of range"))?
            .set_joint_friction(slot as usize, friction)
            .map_err(failed)
    }

    /// Holonomic couplings in one environment.
    pub fn joint_couplings(&self, index: u32) -> Result<Vec<JointCoupling>, TesseraError> {
        let id = self.id(index)?;
        Ok(locked(&self.inner)?
            .batch
            .environment(id)
            .ok_or_else(|| failed("environment index out of range"))?
            .joint_couplings()
            .iter()
            .copied()
            .map(Into::into)
            .collect())
    }

    /// Replace the holonomic joint couplings in one environment.
    pub fn set_joint_couplings(
        &self,
        index: u32,
        couplings: Vec<JointCoupling>,
    ) -> Result<(), TesseraError> {
        let id = self.id(index)?;
        locked(&self.inner)?
            .batch
            .environment_mut(id)
            .ok_or_else(|| failed("environment index out of range"))?
            .set_joint_couplings(couplings.into_iter().map(Into::into).collect())
            .map_err(failed)
    }

    /// Polynomial joint equalities in one environment.
    pub fn joint_polynomial_couplings(
        &self,
        index: u32,
    ) -> Result<Vec<JointPolynomialCoupling>, TesseraError> {
        let id = self.id(index)?;
        Ok(locked(&self.inner)?
            .batch
            .environment(id)
            .ok_or_else(|| failed("environment index out of range"))?
            .joint_polynomial_couplings()
            .iter()
            .copied()
            .map(Into::into)
            .collect())
    }

    /// Replace polynomial joint equalities in one environment.
    pub fn set_joint_polynomial_couplings(
        &self,
        index: u32,
        couplings: Vec<JointPolynomialCoupling>,
    ) -> Result<(), TesseraError> {
        let id = self.id(index)?;
        let couplings = couplings
            .into_iter()
            .map(TryInto::try_into)
            .collect::<Result<Vec<_>, _>>()?;
        locked(&self.inner)?
            .batch
            .environment_mut(id)
            .ok_or_else(|| failed("environment index out of range"))?
            .set_joint_polynomial_couplings(couplings)
            .map_err(failed)
    }

    /// Ball constraints in one environment.
    pub fn link_point_constraints(
        &self,
        index: u32,
    ) -> Result<Vec<LinkPointConstraint>, TesseraError> {
        let id = self.id(index)?;
        Ok(locked(&self.inner)?
            .batch
            .environment(id)
            .ok_or_else(|| failed("environment index out of range"))?
            .link_point_constraints()
            .iter()
            .copied()
            .map(Into::into)
            .collect())
    }

    /// Replace ball constraints in one environment.
    pub fn set_link_point_constraints(
        &self,
        index: u32,
        constraints: Vec<LinkPointConstraint>,
    ) -> Result<(), TesseraError> {
        let id = self.id(index)?;
        locked(&self.inner)?
            .batch
            .environment_mut(id)
            .ok_or_else(|| failed("environment index out of range"))?
            .set_link_point_constraints(constraints.into_iter().map(Into::into).collect())
            .map_err(failed)
    }

    /// Fixed-frame constraints in one environment.
    pub fn link_fixed_constraints(
        &self,
        index: u32,
    ) -> Result<Vec<LinkFixedConstraint>, TesseraError> {
        let id = self.id(index)?;
        Ok(locked(&self.inner)?
            .batch
            .environment(id)
            .ok_or_else(|| failed("environment index out of range"))?
            .link_fixed_constraints()
            .iter()
            .copied()
            .map(Into::into)
            .collect())
    }

    /// Replace fixed-frame constraints in one environment.
    pub fn set_link_fixed_constraints(
        &self,
        index: u32,
        constraints: Vec<LinkFixedConstraint>,
    ) -> Result<(), TesseraError> {
        let id = self.id(index)?;
        let constraints = constraints
            .into_iter()
            .map(TryInto::try_into)
            .collect::<Result<Vec<_>, _>>()?;
        locked(&self.inner)?
            .batch
            .environment_mut(id)
            .ok_or_else(|| failed("environment index out of range"))?
            .set_link_fixed_constraints(constraints)
            .map_err(failed)
    }

    /// Add an independent scene body to one environment.
    pub fn add_scene_body(&self, index: u32, input: SceneBodyInput) -> Result<u32, TesseraError> {
        let id = self.id(index)?;
        let body = input.try_into()?;
        Ok(locked(&self.inner)?
            .batch
            .environment_mut(id)
            .ok_or_else(|| failed("environment index out of range"))?
            .add_scene_body(body) as u32)
    }

    /// Add an independent scene sphere to one environment.
    pub fn add_scene_sphere(
        &self,
        index: u32,
        pose: LinkPose,
        radius: f64,
        mass: f64,
    ) -> Result<u32, TesseraError> {
        let id = self.id(index)?;
        let body = scene_sphere_body(pose, radius, mass)?;
        Ok(locked(&self.inner)?
            .batch
            .environment_mut(id)
            .ok_or_else(|| failed("environment index out of range"))?
            .add_scene_body(body) as u32)
    }

    /// Read one environment's scene body state.
    pub fn scene_body_state(&self, index: u32, body: u32) -> Result<SceneBodyState, TesseraError> {
        let id = self.id(index)?;
        locked(&self.inner)?
            .batch
            .environment(id)
            .ok_or_else(|| failed("environment index out of range"))?
            .scene_bodies
            .get(body as usize)
            .map(Into::into)
            .ok_or_else(|| failed("scene body index out of range"))
    }

    /// Teleport one environment's scene body and invalidate cached contacts.
    pub fn set_scene_body_pose(
        &self,
        index: u32,
        body: u32,
        pose: LinkPose,
    ) -> Result<(), TesseraError> {
        let id = self.id(index)?;
        let pose = link_pose_isometry(pose)?;
        locked(&self.inner)?
            .batch
            .environment_mut(id)
            .ok_or_else(|| failed("environment index out of range"))?
            .set_scene_body_pose(body as usize, pose)
            .map_err(failed)
    }

    /// Prescribe one environment's zero-mass scene body motion, or stop it.
    pub fn set_scene_body_kinematic_motion(
        &self,
        index: u32,
        body: u32,
        linear: Option<Vec3>,
        angular: Option<Vec3>,
    ) -> Result<(), TesseraError> {
        let id = self.id(index)?;
        let motion = match (linear, angular) {
            (Some(linear), Some(angular)) => Some((linear.nalgebra(), angular.nalgebra())),
            (None, None) => None,
            _ => {
                return Err(failed(
                    "linear and angular velocities must both be present or absent",
                ));
            }
        };
        locked(&self.inner)?
            .batch
            .environment_mut(id)
            .ok_or_else(|| failed("environment index out of range"))?
            .set_scene_body_kinematic_motion(body as usize, motion)
            .map_err(failed)
    }

    /// Set one environment's dynamic scene body velocity.
    pub fn set_scene_body_velocity(
        &self,
        index: u32,
        body: u32,
        linear: Vec3,
        angular: Vec3,
    ) -> Result<(), TesseraError> {
        let id = self.id(index)?;
        locked(&self.inner)?
            .batch
            .environment_mut(id)
            .ok_or_else(|| failed("environment index out of range"))?
            .set_scene_body_velocity(body as usize, linear.nalgebra(), angular.nalgebra())
            .map_err(failed)
    }

    /// Set a persistent force on one environment's scene body.
    pub fn set_scene_body_force(
        &self,
        index: u32,
        body: u32,
        force: Vec3,
    ) -> Result<(), TesseraError> {
        let id = self.id(index)?;
        let force = Vector3::new(force.x, force.y, force.z);
        if !force.iter().all(|value| value.is_finite()) {
            return Err(failed("scene body force must be finite"));
        }
        let mut inner = locked(&self.inner)?;
        let scene = inner
            .batch
            .environment_mut(id)
            .ok_or_else(|| failed("environment index out of range"))?
            .scene_bodies
            .get_mut(body as usize)
            .ok_or_else(|| failed("scene body index out of range"))?;
        scene.force = force;
        Ok(())
    }

    /// Set one environment's persistent dynamic-body force and torque.
    pub fn set_scene_body_wrench(
        &self,
        index: u32,
        body: u32,
        force: Vec3,
        torque: Vec3,
    ) -> Result<(), TesseraError> {
        let id = self.id(index)?;
        locked(&self.inner)?
            .batch
            .environment_mut(id)
            .ok_or_else(|| failed("environment index out of range"))?
            .set_scene_body_wrench(body as usize, force.nalgebra(), torque.nalgebra())
            .map_err(failed)
    }

    /// Apply world-frame impulses to one environment's dynamic body.
    pub fn apply_scene_body_impulse(
        &self,
        index: u32,
        body: u32,
        linear: Vec3,
        angular: Vec3,
    ) -> Result<(), TesseraError> {
        let id = self.id(index)?;
        locked(&self.inner)?
            .batch
            .environment_mut(id)
            .ok_or_else(|| failed("environment index out of range"))?
            .apply_scene_body_impulse(body as usize, linear.nalgebra(), angular.nalgebra())
            .map_err(failed)
    }

    /// Remove one environment's scene body and renumber dependent constraints.
    pub fn remove_scene_body(&self, index: u32, body: u32) -> Result<bool, TesseraError> {
        let id = self.id(index)?;
        Ok(locked(&self.inner)?
            .batch
            .environment_mut(id)
            .ok_or_else(|| failed("environment index out of range"))?
            .remove_scene_body(body as usize))
    }

    /// Ball constraints between articulation links and scene bodies in one environment.
    pub fn link_scene_point_constraints(
        &self,
        index: u32,
    ) -> Result<Vec<LinkScenePointConstraint>, TesseraError> {
        let id = self.id(index)?;
        Ok(locked(&self.inner)?
            .batch
            .environment(id)
            .ok_or_else(|| failed("environment index out of range"))?
            .link_scene_point_constraints()
            .iter()
            .copied()
            .map(Into::into)
            .collect())
    }

    /// Replace ball constraints between articulation links and scene bodies in one environment.
    pub fn set_link_scene_point_constraints(
        &self,
        index: u32,
        constraints: Vec<LinkScenePointConstraint>,
    ) -> Result<(), TesseraError> {
        let id = self.id(index)?;
        locked(&self.inner)?
            .batch
            .environment_mut(id)
            .ok_or_else(|| failed("environment index out of range"))?
            .set_link_scene_point_constraints(constraints.into_iter().map(Into::into).collect())
            .map_err(failed)
    }

    /// Fixed-frame constraints between articulation links and scene bodies in one environment.
    pub fn link_scene_fixed_constraints(
        &self,
        index: u32,
    ) -> Result<Vec<LinkSceneFixedConstraint>, TesseraError> {
        let id = self.id(index)?;
        Ok(locked(&self.inner)?
            .batch
            .environment(id)
            .ok_or_else(|| failed("environment index out of range"))?
            .link_scene_fixed_constraints()
            .iter()
            .copied()
            .map(Into::into)
            .collect())
    }

    /// Replace fixed-frame constraints between articulation links and scene bodies in one environment.
    pub fn set_link_scene_fixed_constraints(
        &self,
        index: u32,
        constraints: Vec<LinkSceneFixedConstraint>,
    ) -> Result<(), TesseraError> {
        let id = self.id(index)?;
        let constraints = constraints
            .into_iter()
            .map(TryInto::try_into)
            .collect::<Result<Vec<_>, _>>()?;
        locked(&self.inner)?
            .batch
            .environment_mut(id)
            .ok_or_else(|| failed("environment index out of range"))?
            .set_link_scene_fixed_constraints(constraints)
            .map_err(failed)
    }

    /// Reflected rotor inertia on one environment's articulation edge.
    pub fn joint_armature(&self, index: u32, edge: u32) -> Result<Vec<f64>, TesseraError> {
        let id = self.id(index)?;
        locked(&self.inner)?
            .batch
            .environment(id)
            .ok_or_else(|| failed("environment index out of range"))?
            .articulation
            .joint_armature(edge as usize)
            .map(<[f64]>::to_vec)
            .ok_or_else(|| failed("joint edge out of range"))
    }

    /// Set reflected rotor inertia on one environment's articulation edge.
    pub fn set_joint_armature(
        &self,
        index: u32,
        edge: u32,
        armatures: Vec<f64>,
    ) -> Result<(), TesseraError> {
        let id = self.id(index)?;
        locked(&self.inner)?
            .batch
            .environment_mut(id)
            .ok_or_else(|| failed("environment index out of range"))?
            .articulation
            .set_joint_armature(edge as usize, &armatures)
            .map_err(failed)
    }

    /// Persistent world-frame force, torque, and gravity multiplier of one link.
    pub fn link_load(&self, index: u32, link: u32) -> Result<LinkLoad, TesseraError> {
        let id = self.id(index)?;
        let inner = locked(&self.inner)?;
        inner
            .batch
            .environment(id)
            .ok_or_else(|| failed("environment index out of range"))?
            .link_external_load(link as usize)
            .map(Into::into)
            .ok_or_else(|| failed("link index out of range"))
    }

    /// Apply a persistent world-frame force and torque to one environment link.
    pub fn set_link_external_wrench(
        &self,
        index: u32,
        link: u32,
        force: Vec3,
        torque: Vec3,
    ) -> Result<(), TesseraError> {
        let id = self.id(index)?;
        let mut inner = locked(&self.inner)?;
        inner
            .batch
            .environment_mut(id)
            .ok_or_else(|| failed("environment index out of range"))?
            .set_link_external_wrench(link as usize, force.nalgebra(), torque.nalgebra())
            .map_err(failed)
    }

    /// Scale one link's gravity without changing its inertia.
    pub fn set_link_gravity_scale(
        &self,
        index: u32,
        link: u32,
        scale: f64,
    ) -> Result<(), TesseraError> {
        let id = self.id(index)?;
        let mut inner = locked(&self.inner)?;
        inner
            .batch
            .environment_mut(id)
            .ok_or_else(|| failed("environment index out of range"))?
            .set_link_gravity_scale(link as usize, scale)
            .map_err(failed)
    }

    /// Optional motor for one environment's generalized coordinate.
    pub fn joint_motor(&self, index: u32, slot: u32) -> Result<Option<JointMotor>, TesseraError> {
        let id = self.id(index)?;
        let inner = locked(&self.inner)?;
        let world = inner
            .batch
            .environment(id)
            .ok_or_else(|| failed("environment index out of range"))?;
        if slot as usize >= world.positions.len() {
            return Err(failed("joint coordinate out of range"));
        }
        Ok(world.joint_motor(slot as usize).map(Into::into))
    }

    /// Material of the finite ground plane in one environment.
    pub fn ground_material(&self, index: u32) -> Result<Material, TesseraError> {
        let id = self.id(index)?;
        let inner = locked(&self.inner)?;
        inner
            .batch
            .environment(id)
            .map(|world| world.ground_material().into())
            .ok_or_else(|| failed("environment index out of range"))
    }

    /// Set one environment's finite ground material.
    pub fn set_ground_material(&self, index: u32, material: Material) -> Result<(), TesseraError> {
        let id = self.id(index)?;
        let mut inner = locked(&self.inner)?;
        inner
            .batch
            .environment_mut(id)
            .ok_or_else(|| failed("environment index out of range"))?
            .set_ground_material(material.into())
            .map_err(failed)
    }

    /// Material of one articulated-link collider in an environment.
    pub fn link_material(
        &self,
        environment: u32,
        kind: LinkColliderKind,
        index: u32,
    ) -> Result<Material, TesseraError> {
        let id = self.id(environment)?;
        let inner = locked(&self.inner)?;
        let world = inner
            .batch
            .environment(id)
            .ok_or_else(|| failed("environment index out of range"))?;
        let index = index as usize;
        let material = match kind {
            LinkColliderKind::Sphere => world.link_sphere_material(index),
            LinkColliderKind::GroundPoint => world.ground_point_material(index),
            LinkColliderKind::Box => world.link_box_material(index),
            LinkColliderKind::Cylinder => world.link_cylinder_material(index),
            LinkColliderKind::Convex => world.link_convex_material(index),
        };
        material
            .map(Into::into)
            .ok_or_else(|| failed("link collider index out of range"))
    }

    /// Set one articulated-link collider material in an environment.
    pub fn set_link_material(
        &self,
        environment: u32,
        kind: LinkColliderKind,
        index: u32,
        material: Material,
    ) -> Result<(), TesseraError> {
        let id = self.id(environment)?;
        let mut inner = locked(&self.inner)?;
        let world = inner
            .batch
            .environment_mut(id)
            .ok_or_else(|| failed("environment index out of range"))?;
        let material = material.into();
        let result = match kind {
            LinkColliderKind::Sphere => world.set_link_sphere_material(index as usize, material),
            LinkColliderKind::GroundPoint => {
                world.set_ground_point_material(index as usize, material)
            }
            LinkColliderKind::Box => world.set_link_box_material(index as usize, material),
            LinkColliderKind::Cylinder => {
                world.set_link_cylinder_material(index as usize, material)
            }
            LinkColliderKind::Convex => world.set_link_convex_material(index as usize, material),
        };
        result.map_err(failed)
    }

    /// Material of one scene-body collider in an environment.
    pub fn scene_collider_material(
        &self,
        environment: u32,
        body_index: u32,
        collider_index: u32,
    ) -> Result<Material, TesseraError> {
        let id = self.id(environment)?;
        let inner = locked(&self.inner)?;
        inner
            .batch
            .environment(id)
            .ok_or_else(|| failed("environment index out of range"))?
            .scene_collider_material(body_index as usize, collider_index as usize)
            .map(Into::into)
            .ok_or_else(|| failed("scene collider index out of range"))
    }

    /// Set one scene-body collider material in an environment.
    pub fn set_scene_collider_material(
        &self,
        environment: u32,
        body_index: u32,
        collider_index: u32,
        material: Material,
    ) -> Result<(), TesseraError> {
        let id = self.id(environment)?;
        let mut inner = locked(&self.inner)?;
        inner
            .batch
            .environment_mut(id)
            .ok_or_else(|| failed("environment index out of range"))?
            .set_scene_collider_material(
                body_index as usize,
                collider_index as usize,
                material.into(),
            )
            .map_err(failed)
    }

    /// World-space link poses in articulation order for one environment.
    pub fn link_poses(&self, index: u32) -> Result<Vec<LinkPose>, TesseraError> {
        let id = self.id(index)?;
        let inner = locked(&self.inner)?;
        let world = inner
            .batch
            .environment(id)
            .ok_or_else(|| failed("environment index out of range"))?;
        Ok(world
            .link_poses()
            .map_err(failed)?
            .into_iter()
            .map(|pose| {
                let q = pose.rotation.quaternion();
                LinkPose {
                    position: pose.translation.vector.into(),
                    orientation: Quaternion {
                        x: q.i,
                        y: q.j,
                        z: q.k,
                        w: q.w,
                    },
                }
            })
            .collect())
    }

    /// World-space link-origin velocities for one environment.
    pub fn link_twists(&self, index: u32) -> Result<Vec<LinkTwist>, TesseraError> {
        let id = self.id(index)?;
        let inner = locked(&self.inner)?;
        inner
            .batch
            .environment(id)
            .ok_or_else(|| failed("environment index out of range"))?
            .link_twists()
            .map_err(failed)
            .map(|twists| twists.into_iter().map(Into::into).collect())
    }

    /// World-space velocity at a link-frame point in one environment.
    pub fn link_point_twist(
        &self,
        index: u32,
        link: u32,
        local_point: Vec3,
    ) -> Result<LinkTwist, TesseraError> {
        let id = self.id(index)?;
        let inner = locked(&self.inner)?;
        inner
            .batch
            .environment(id)
            .ok_or_else(|| failed("environment index out of range"))?
            .link_point_twist(link as usize, local_point.nalgebra())
            .map(Into::into)
            .map_err(failed)
    }

    /// Kinematic acceleration at a link-frame point in one environment.
    pub fn link_point_acceleration(
        &self,
        index: u32,
        link: u32,
        local_point: Vec3,
        generalized_acceleration: Vec<f64>,
    ) -> Result<LinkAcceleration, TesseraError> {
        let id = self.id(index)?;
        let inner = locked(&self.inner)?;
        inner
            .batch
            .environment(id)
            .ok_or_else(|| failed("environment index out of range"))?
            .link_point_acceleration(
                link as usize,
                local_point.nalgebra(),
                &generalized_acceleration,
            )
            .map(Into::into)
            .map_err(failed)
    }

    /// Ideal link-frame inertial sensor values in one environment.
    pub fn link_imu(
        &self,
        index: u32,
        link: u32,
        local_point: Vec3,
        generalized_acceleration: Vec<f64>,
    ) -> Result<LinkImuReading, TesseraError> {
        let id = self.id(index)?;
        let inner = locked(&self.inner)?;
        inner
            .batch
            .environment(id)
            .ok_or_else(|| failed("environment index out of range"))?
            .link_imu(
                link as usize,
                local_point.nalgebra(),
                &generalized_acceleration,
            )
            .map(Into::into)
            .map_err(failed)
    }

    /// Contact wrench on one link in an environment's last substep.
    pub fn link_contact_wrench(
        &self,
        environment: u32,
        link: u32,
    ) -> Result<LinkContactWrench, TesseraError> {
        let id = self.id(environment)?;
        let inner = locked(&self.inner)?;
        let world = inner
            .batch
            .environment(id)
            .ok_or_else(|| failed("environment index out of range"))?;
        let index = link as usize;
        let force = world
            .contact_forces
            .get(index)
            .ok_or_else(|| failed("link index out of range"))?;
        let torque = &world.contact_torques[index];
        Ok(LinkContactWrench {
            force: (*force).into(),
            torque: (*torque).into(),
        })
    }

    /// Publish the current state as one environment's future reset template.
    pub fn publish_reset_template(&self, index: u32) -> Result<(), TesseraError> {
        let id = self.id(index)?;
        locked(&self.inner)?
            .batch
            .publish_reset_template(id)
            .map_err(failed)
    }

    /// Restore one environment without touching the other environments.
    pub fn reset_environment(&self, index: u32) -> Result<(), TesseraError> {
        let id = self.id(index)?;
        locked(&self.inner)?
            .batch
            .reset_environment(id)
            .map_err(failed)
    }

    /// Advance all environments using CPU contact solves.
    pub fn step(&self, dt: f64, torques: Vec<Vec<f64>>) -> Result<(), TesseraError> {
        let refs = torques.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let mut inner = locked(&self.inner)?;
        inner.resident = None;
        inner.batch.step(dt, &refs).map_err(failed)
    }

    /// Advance all environments with packed GPU contact solves.
    pub fn step_gpu(&self, dt: f64, torques: Vec<Vec<f64>>) -> Result<(), TesseraError> {
        let refs = torques.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let mut inner = locked(&self.inner)?;
        inner.resident = None;
        if inner.gpu.is_none() {
            inner.gpu = Some(GpuContactDevice::new().map_err(failed)?);
        }
        let ArticulatedBatchInner { batch, gpu, .. } = &mut *inner;
        let context = gpu.as_ref().ok_or_else(|| failed("GPU unavailable"))?;
        batch.step_gpu(dt, &refs, context).map_err(failed)
    }

    /// Advance all environments with packed GPU mass inversion and contacts.
    pub fn step_gpu_device_mass(
        &self,
        dt: f64,
        torques: Vec<Vec<f64>>,
    ) -> Result<(), TesseraError> {
        let refs = torques.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let mut inner = locked(&self.inner)?;
        inner.resident = None;
        if inner.gpu.is_none() {
            inner.gpu = Some(GpuContactDevice::new().map_err(failed)?);
        }
        let ArticulatedBatchInner {
            batch,
            gpu,
            device_mass_cache,
            ..
        } = &mut *inner;
        let context = gpu.as_ref().ok_or_else(|| failed("GPU unavailable"))?;
        device_mass_cache
            .step(batch, dt, &refs, context)
            .map_err(failed)
    }

    /// Advance mixed-DOF environments in one reusable packed resident batch.
    /// Reset after state, geometry, material, solver, or prescribed motion edits.
    /// All environment states and effort lengths are checked before submission.
    pub fn step_gpu_resident(
        &self,
        timestep: f64,
        torques: Vec<Vec<f64>>,
        steps: u32,
    ) -> Result<(), TesseraError> {
        self.step_gpu_resident_with_contacts(timestep, torques, steps)
            .map(|_| ())
    }

    /// Advance the resident solver and return final-substep scene-body contacts.
    /// Reports follow original scene-body order, nested by environment.
    pub fn step_gpu_resident_with_contacts(
        &self,
        timestep: f64,
        torques: Vec<Vec<f64>>,
        steps: u32,
    ) -> Result<Vec<Vec<SceneContactReport>>, TesseraError> {
        if !timestep.is_finite() || timestep <= 0.0 || steps == 0 {
            return Err(failed("resident timestep and step count must be positive"));
        }
        let refs = torques.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let mut inner = locked(&self.inner)?;
        if inner.gpu.is_none() {
            inner.gpu = Some(GpuContactDevice::new().map_err(failed)?);
        }
        let ArticulatedBatchInner {
            batch,
            gpu,
            resident,
            ..
        } = &mut *inner;
        if !resident.as_ref().is_some_and(|(dt, _)| *dt == timestep) {
            let context = gpu.as_ref().ok_or_else(|| failed("GPU unavailable"))?;
            let session = GpuSceneDynamicsBatch::new(batch.environments(), context, timestep)
                .map_err(failed)?;
            *resident = Some((timestep, session));
        }
        let (_, session) = resident
            .as_mut()
            .ok_or_else(|| failed("resident GPU unavailable"))?;
        let diagnostics = session
            .step(batch.environments_mut(), &refs, steps as usize)
            .map_err(failed)?;
        Ok(diagnostics
            .into_iter()
            .map(|item| scene_contact_reports(item, timestep))
            .collect())
    }

    /// Latest synchronized sleep state of a scene body in one environment.
    pub fn scene_body_is_sleeping(
        &self,
        environment: u32,
        body: u32,
    ) -> Result<bool, TesseraError> {
        let id = self.id(environment)?;
        locked(&self.inner)?
            .batch
            .environment(id)
            .ok_or_else(|| failed("environment index out of range"))?
            .scene_body_is_sleeping(body as usize)
            .ok_or_else(|| failed("scene body index out of range"))
    }

    /// Upload edited static poses after checking every environment's layout.
    pub fn update_gpu_resident_static_scene_poses(&self) -> Result<(), TesseraError> {
        let mut inner = locked(&self.inner)?;
        let ArticulatedBatchInner {
            batch, resident, ..
        } = &mut *inner;
        let (_, session) = resident
            .as_mut()
            .ok_or_else(|| failed("resident session is not initialized"))?;
        session
            .update_static_scene_poses(batch.environments())
            .map_err(failed)
    }

    /// Enable independent scene-body sleeping in the initialized packed session.
    /// Reset or timestep changes discard the setting for every environment.
    pub fn enable_gpu_resident_scene_sleep(&self) -> Result<(), TesseraError> {
        let mut inner = locked(&self.inner)?;
        let ArticulatedBatchInner {
            batch, resident, ..
        } = &mut *inner;
        let (_, session) = resident.as_mut().ok_or_else(|| {
            failed("initialize the resident session with step_gpu_resident first")
        })?;
        session
            .enable_scene_sleep(batch.environments())
            .map_err(failed)
    }

    /// Release packed resident buffers before changing persistent configuration.
    pub fn reset_gpu_resident(&self) -> Result<(), TesseraError> {
        locked(&self.inner)?.resident = None;
        Ok(())
    }

    /// Solve current generalized accelerations in one GPU dispatch.
    /// This does not advance any environment or resolve contacts.
    pub fn generalized_accelerations_gpu(
        &self,
        torques: Vec<Vec<f64>>,
    ) -> Result<Vec<Vec<f64>>, TesseraError> {
        let refs = torques.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let mut inner = locked(&self.inner)?;
        if inner.gpu.is_none() {
            inner.gpu = Some(GpuContactDevice::new().map_err(failed)?);
        }
        let ArticulatedBatchInner { batch, gpu, .. } = &mut *inner;
        let context = gpu.as_ref().ok_or_else(|| failed("GPU unavailable"))?;
        batch
            .generalized_accelerations_gpu(&refs, context)
            .map_err(failed)
            .map(|values| {
                values
                    .into_iter()
                    .map(|value| value.iter().copied().collect())
                    .collect()
            })
    }
}

/// GPU-resident mixed-primitive world.
#[derive(Debug, uniffi::Object)]
pub struct GpuPrimitiveWorld {
    inner: Mutex<GpuPrimitiveWorldInner>,
}

#[derive(Debug)]
struct GpuPrimitiveWorldInner {
    world: CoreGpuPrimitiveWorld,
    initial: Vec<GpuRigidBodyState>,
    shapes: Vec<GpuPrimitiveShape>,
}

#[uniffi::export]
impl GpuPrimitiveWorld {
    /// Limit dynamic bodies' world-space linear speed in metres per second.
    pub fn set_max_linear_speed(&self, max_speed: Option<f32>) -> Result<(), TesseraError> {
        locked(&self.inner)?
            .world
            .set_max_linear_speed(max_speed)
            .map_err(failed)
    }

    /// Current linear speed limit, or None when disabled.
    pub fn max_linear_speed(&self) -> Result<Option<f32>, TesseraError> {
        Ok(locked(&self.inner)?.world.max_linear_speed())
    }

    /// Read accumulated contact impulses from the last solver substep.
    pub fn readback_contact_impulses(&self) -> Result<GpuContactImpulseReadback, TesseraError> {
        Ok(locked(&self.inner)?
            .world
            .readback_contact_impulses()
            .map_err(failed)?
            .into())
    }

    /// Construct a mixed-shape world on a GPU adapter.
    #[uniffi::constructor]
    pub fn new(
        bodies: Vec<GpuPrimitiveBody>,
        gravity: Vec3,
        ground_half_extent: f32,
    ) -> Result<Arc<Self>, TesseraError> {
        let shapes = bodies.iter().map(|body| body.shape.clone()).collect();
        let (states, colliders): (Vec<_>, Vec<_>) = bodies
            .into_iter()
            .map(gpu_primitive)
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .unzip();
        let config = gpu_config(gravity, ground_half_extent)?;
        let gpu = GpuContactDevice::new().map_err(failed)?;
        let world = CoreGpuPrimitiveWorld::new_primitives(
            gpu.device(),
            gpu.queue(),
            &states,
            &colliders,
            config,
        )
        .map_err(failed)?;
        Ok(Arc::new(Self {
            inner: Mutex::new(GpuPrimitiveWorldInner {
                world,
                initial: states,
                shapes,
            }),
        }))
    }

    /// Number of bodies in dense index order.
    pub fn len(&self) -> Result<u32, TesseraError> {
        u32::try_from(locked(&self.inner)?.world.len()).map_err(failed)
    }

    /// Whether this world has no bodies.
    pub fn is_empty(&self) -> Result<bool, TesseraError> {
        Ok(locked(&self.inner)?.world.is_empty())
    }

    /// Shape of one body in dense index order.
    pub fn shape(&self, index: u32) -> Result<GpuPrimitiveShape, TesseraError> {
        locked(&self.inner)?
            .shapes
            .get(index as usize)
            .cloned()
            .ok_or_else(|| failed("body index out of range"))
    }

    /// Replace point-to-point joints between bodies in this world.
    pub fn set_ball_joints(&self, joints: Vec<GpuBallJoint>) -> Result<(), TesseraError> {
        let joints = joints.into_iter().map(Into::into).collect::<Vec<_>>();
        locked(&self.inner)?
            .world
            .set_ball_joints(&joints)
            .map_err(failed)
    }

    /// Current point-to-point joints in stable insertion order.
    pub fn ball_joints(&self) -> Result<Vec<GpuBallJoint>, TesseraError> {
        Ok(locked(&self.inner)?
            .world
            .ball_joints()
            .iter()
            .copied()
            .map(Into::into)
            .collect())
    }

    /// Replace position and orientation locks between bodies in this world.
    pub fn set_fixed_joints(&self, joints: Vec<GpuFixedJoint>) -> Result<(), TesseraError> {
        let joints = joints.into_iter().map(Into::into).collect::<Vec<_>>();
        locked(&self.inner)?
            .world
            .set_fixed_joints(&joints)
            .map_err(failed)
    }

    /// Current fixed joints in stable insertion order.
    pub fn fixed_joints(&self) -> Result<Vec<GpuFixedJoint>, TesseraError> {
        Ok(locked(&self.inner)?
            .world
            .fixed_joints()
            .iter()
            .copied()
            .map(Into::into)
            .collect())
    }

    /// Replace hinge joints between bodies in this world.
    pub fn set_revolute_joints(&self, joints: Vec<GpuRevoluteJoint>) -> Result<(), TesseraError> {
        let joints = joints.into_iter().map(Into::into).collect::<Vec<_>>();
        locked(&self.inner)?
            .world
            .set_revolute_joints(&joints)
            .map_err(failed)
    }

    /// Current revolute joints in stable insertion order.
    pub fn revolute_joints(&self) -> Result<Vec<GpuRevoluteJoint>, TesseraError> {
        Ok(locked(&self.inner)?
            .world
            .revolute_joints()
            .iter()
            .copied()
            .map(Into::into)
            .collect())
    }

    /// Replace slider joints between bodies in this world.
    pub fn set_prismatic_joints(&self, joints: Vec<GpuPrismaticJoint>) -> Result<(), TesseraError> {
        let joints = joints.into_iter().map(Into::into).collect::<Vec<_>>();
        locked(&self.inner)?
            .world
            .set_prismatic_joints(&joints)
            .map_err(failed)
    }

    /// Current slider joints in stable insertion order.
    pub fn prismatic_joints(&self) -> Result<Vec<GpuPrismaticJoint>, TesseraError> {
        Ok(locked(&self.inner)?
            .world
            .prismatic_joints()
            .iter()
            .copied()
            .map(Into::into)
            .collect())
    }

    /// Set or clear a hinge's velocity motor.
    pub fn set_revolute_motor(
        &self,
        index: u32,
        motor: Option<GpuAxisMotor>,
    ) -> Result<(), TesseraError> {
        locked(&self.inner)?
            .world
            .set_revolute_motor(index as usize, motor.map(Into::into))
            .map_err(failed)
    }

    /// Current hinge velocity motor.
    pub fn revolute_motor(&self, index: u32) -> Result<Option<GpuAxisMotor>, TesseraError> {
        Ok(locked(&self.inner)?
            .world
            .revolute_motor(index as usize)
            .map_err(failed)?
            .map(Into::into))
    }

    /// Set or clear a slider's velocity motor.
    pub fn set_prismatic_motor(
        &self,
        index: u32,
        motor: Option<GpuAxisMotor>,
    ) -> Result<(), TesseraError> {
        locked(&self.inner)?
            .world
            .set_prismatic_motor(index as usize, motor.map(Into::into))
            .map_err(failed)
    }

    /// Current slider velocity motor.
    pub fn prismatic_motor(&self, index: u32) -> Result<Option<GpuAxisMotor>, TesseraError> {
        Ok(locked(&self.inner)?
            .world
            .prismatic_motor(index as usize)
            .map_err(failed)?
            .map(Into::into))
    }

    /// Set or clear a slider's displacement limits.
    pub fn set_prismatic_limit(
        &self,
        index: u32,
        limit: Option<GpuPrismaticLimit>,
    ) -> Result<(), TesseraError> {
        locked(&self.inner)?
            .world
            .set_prismatic_limit(index as usize, limit.map(Into::into))
            .map_err(failed)
    }

    /// Current slider displacement limits.
    pub fn prismatic_limit(&self, index: u32) -> Result<Option<GpuPrismaticLimit>, TesseraError> {
        Ok(locked(&self.inner)?
            .world
            .prismatic_limit(index as usize)
            .map_err(failed)?
            .map(Into::into))
    }

    /// Set or clear a hinge's position servo.
    pub fn set_revolute_servo(
        &self,
        index: u32,
        servo: Option<GpuAxisServo>,
    ) -> Result<(), TesseraError> {
        locked(&self.inner)?
            .world
            .set_revolute_servo(index as usize, servo.map(Into::into))
            .map_err(failed)
    }

    /// Current hinge position servo.
    pub fn revolute_servo(&self, index: u32) -> Result<Option<GpuAxisServo>, TesseraError> {
        Ok(locked(&self.inner)?
            .world
            .revolute_servo(index as usize)
            .map_err(failed)?
            .map(Into::into))
    }

    /// Set or clear a hinge's continuous angle limits.
    pub fn set_revolute_limit(
        &self,
        index: u32,
        limit: Option<GpuRevoluteLimit>,
    ) -> Result<(), TesseraError> {
        locked(&self.inner)?
            .world
            .set_revolute_limit(index as usize, limit.map(Into::into))
            .map_err(failed)
    }

    /// Current hinge angle limits.
    pub fn revolute_limit(&self, index: u32) -> Result<Option<GpuRevoluteLimit>, TesseraError> {
        Ok(locked(&self.inner)?
            .world
            .revolute_limit(index as usize)
            .map_err(failed)?
            .map(Into::into))
    }

    /// Read a hinge's continuous angle in radians.
    pub fn readback_revolute_angle(&self, index: u32) -> Result<f32, TesseraError> {
        locked(&self.inner)?
            .world
            .readback_revolute_angle(index as usize)
            .map_err(failed)
    }

    /// Set or clear a slider's position servo.
    pub fn set_prismatic_servo(
        &self,
        index: u32,
        servo: Option<GpuAxisServo>,
    ) -> Result<(), TesseraError> {
        locked(&self.inner)?
            .world
            .set_prismatic_servo(index as usize, servo.map(Into::into))
            .map_err(failed)
    }

    /// Current slider position servo.
    pub fn prismatic_servo(&self, index: u32) -> Result<Option<GpuAxisServo>, TesseraError> {
        Ok(locked(&self.inner)?
            .world
            .prismatic_servo(index as usize)
            .map_err(failed)?
            .map(Into::into))
    }

    /// Append a primitive and return its dense index.
    ///
    /// This synchronizes GPU state and rebuilds contact buffers. Queued forces,
    /// warm-start impulses, and sleep timers are discarded.
    pub fn add_body(&self, body: GpuPrimitiveBody) -> Result<u32, TesseraError> {
        let shape = body.shape.clone();
        let (state, collider) = gpu_primitive(body)?;
        let mut inner = locked(&self.inner)?;
        let index = inner
            .world
            .append_primitive(state, collider)
            .map_err(failed)?;
        inner.initial.push(state);
        inner.shapes.push(shape);
        u32::try_from(index).map_err(failed)
    }

    /// Remove a primitive and shift all later dense indices down by one.
    ///
    /// This synchronizes GPU state and rebuilds contact buffers. Queued forces,
    /// warm-start impulses, and sleep timers are discarded.
    pub fn remove_body(&self, index: u32) -> Result<RemovedGpuPrimitive, TesseraError> {
        let mut inner = locked(&self.inner)?;
        let (state, _shape) = inner
            .world
            .remove_primitive(index as usize)
            .map_err(failed)?;
        let _initial = inner.initial.remove(index as usize);
        let shape = inner.shapes.remove(index as usize);
        Ok(RemovedGpuPrimitive {
            state: state.into(),
            shape,
        })
    }

    /// Discard a primitive without reading its GPU state back.
    pub fn discard_body(&self, index: u32) -> Result<(), TesseraError> {
        let mut inner = locked(&self.inner)?;
        inner
            .world
            .discard_primitive(index as usize)
            .map_err(failed)?;
        let _initial = inner.initial.remove(index as usize);
        let _shape = inner.shapes.remove(index as usize);
        Ok(())
    }

    /// Advance one substep and return the broad-phase candidate count.
    pub fn step(&self, dt: f32) -> Result<u32, TesseraError> {
        let count = locked(&self.inner)?.world.step(dt).map_err(failed)?;
        u32::try_from(count).map_err(failed)
    }

    /// Advance several substeps in one call.
    pub fn step_substeps(&self, dt: f32, count: u32) -> Result<u32, TesseraError> {
        let candidates = locked(&self.inner)?
            .world
            .step_substeps(dt, count)
            .map_err(failed)?;
        u32::try_from(candidates).map_err(failed)
    }

    /// Advance one temporal frame. Forces are captured once for all substeps.
    /// `None` selects defaults with zero speculative margin.
    pub fn step_temporal(
        &self,
        frame_dt: f32,
        substeps: u32,
        settings: Option<GpuTemporalSettings>,
    ) -> Result<u32, TesseraError> {
        let mut state = locked(&self.inner)?;
        temporal_step_binding(&mut state.world, frame_dt, substeps, settings, None)
    }

    /// Advance a sphere-only temporal frame using the configured speed cap.
    pub fn step_temporal_speed_bounded(
        &self,
        frame_dt: f32,
        substeps: u32,
        settings: Option<GpuTemporalSettings>,
        joint_settings: Option<GpuTemporalJointSettings>,
    ) -> Result<u32, TesseraError> {
        let mut state = locked(&self.inner)?;
        temporal_step_speed_bounded_binding(
            &mut state.world,
            frame_dt,
            substeps,
            settings,
            joint_settings,
        )
    }

    /// Advance a temporal frame with independent contact and joint coefficients.
    pub fn step_temporal_with_joints(
        &self,
        frame_dt: f32,
        substeps: u32,
        settings: Option<GpuTemporalSettings>,
        joint_settings: GpuTemporalJointSettings,
    ) -> Result<u32, TesseraError> {
        let mut state = locked(&self.inner)?;
        temporal_step_binding(
            &mut state.world,
            frame_dt,
            substeps,
            settings,
            Some(joint_settings),
        )
    }

    /// Set prescribed motion without reading the pose; None restores static behavior.
    /// Dynamic bodies ignore the command. Nonfinite velocities are rejected.
    pub fn set_kinematic_motion(
        &self,
        index: u32,
        motion: Option<GpuKinematicMotion>,
    ) -> Result<(), TesseraError> {
        locked(&self.inner)?
            .world
            .set_kinematic_motion(index as usize, motion.map(GpuKinematicMotion::gpu))
            .map_err(failed)
    }

    /// Apply a world-space force and torque to one body for its next step.
    pub fn write_wrench(&self, index: u32, force: Vec3, torque: Vec3) -> Result<(), TesseraError> {
        let [x, y, z] = force.gpu();
        let [tx, ty, tz] = torque.gpu();
        locked(&self.inner)?
            .world
            .write_forces(
                index as usize,
                GpuRigidBodyForces {
                    force: [x, y, z, 0.0],
                    torque: [tx, ty, tz, 0.0],
                },
            )
            .map_err(failed)
    }

    /// Override one body's contact material.
    pub fn set_body_material(&self, index: u32, material: Material) -> Result<(), TesseraError> {
        locked(&self.inner)?
            .world
            .set_body_material(index as usize, material.into())
            .map_err(failed)
    }

    /// Restore one body's default contact material.
    pub fn clear_body_material(&self, index: u32) -> Result<(), TesseraError> {
        locked(&self.inner)?
            .world
            .clear_body_material(index as usize)
            .map_err(failed)
    }

    /// Set reciprocal collision masks for one primitive.
    pub fn set_body_collision_groups(
        &self,
        index: u32,
        groups: GpuCollisionGroups,
    ) -> Result<(), TesseraError> {
        locked(&self.inner)?
            .world
            .set_body_collision_groups(index as usize, groups.into())
            .map_err(failed)
    }

    /// Read reciprocal collision masks for one primitive.
    pub fn body_collision_groups(&self, index: u32) -> Result<GpuCollisionGroups, TesseraError> {
        locked(&self.inner)?
            .world
            .body_collision_groups(index as usize)
            .map(Into::into)
            .ok_or_else(|| failed("body index out of range"))
    }

    /// Override the finite ground's contact material.
    pub fn set_ground_material(&self, material: Material) -> Result<(), TesseraError> {
        locked(&self.inner)?
            .world
            .set_ground_material(material.into())
            .map_err(failed)
    }

    /// Restore the configured ground contact material.
    pub fn clear_ground_material(&self) -> Result<(), TesseraError> {
        locked(&self.inner)?
            .world
            .clear_ground_material()
            .map_err(failed)
    }

    /// Set reciprocal collision masks for the finite ground.
    pub fn set_ground_collision_groups(
        &self,
        groups: GpuCollisionGroups,
    ) -> Result<(), TesseraError> {
        locked(&self.inner)?
            .world
            .set_ground_collision_groups(groups.into())
            .map_err(failed)
    }

    /// Read reciprocal collision masks for the finite ground.
    pub fn ground_collision_groups(&self) -> Result<GpuCollisionGroups, TesseraError> {
        locked(&self.inner)?
            .world
            .ground_collision_groups()
            .map(Into::into)
            .ok_or_else(|| failed("ground is disabled"))
    }

    /// Restore the initial body states while preserving shapes and materials.
    pub fn reset(&self) -> Result<(), TesseraError> {
        let mut inner = locked(&self.inner)?;
        let GpuPrimitiveWorldInner { world, initial, .. } = &mut *inner;
        world.reset(initial).map_err(failed)
    }

    /// Cast rays against current resident geometry without transferring body state.
    pub fn cast_rays(&self, rays: Vec<GpuRay>) -> Result<Vec<Option<GpuRayHit>>, TesseraError> {
        let state = locked(&self.inner)?;
        let hits = (state.world).cast_rays(&gpu_rays(rays)).map_err(failed)?;
        Ok(gpu_ray_hits(hits))
    }

    /// Project points against current resident geometry without transferring body state.
    pub fn project_points(
        &self,
        points: Vec<GpuPointQuery>,
    ) -> Result<Vec<Option<GpuPointHit>>, TesseraError> {
        let state = locked(&self.inner)?;
        let hits = (state.world)
            .project_points(&gpu_points(points))
            .map_err(failed)?;
        Ok(gpu_point_hits(hits))
    }

    /// Evaluate rays and points with one current-state GPU scene tree.
    pub fn query_scene(
        &self,
        rays: Vec<GpuRay>,
        points: Vec<GpuPointQuery>,
    ) -> Result<GpuSceneQueryHits, TesseraError> {
        let state = locked(&self.inner)?;
        let hits = state
            .world
            .query_scene(&gpu_rays(rays), &gpu_points(points))
            .map_err(failed)?;
        Ok(gpu_scene_query_hits(hits))
    }

    /// Transfer all body states to the CPU.
    pub fn readback(&self) -> Result<Vec<GpuPrimitiveState>, TesseraError> {
        Ok(locked(&self.inner)?
            .world
            .readback()
            .map_err(failed)?
            .into_iter()
            .map(Into::into)
            .collect())
    }

    /// Transfer active pair and finite-ground manifold contacts to the CPU.
    pub fn readback_contacts(&self) -> Result<Vec<GpuPrimitiveContact>, TesseraError> {
        let readback = locked(&self.inner)?
            .world
            .readback_contacts()
            .map_err(failed)?;
        primitive_contacts(readback)
    }
}

/// GPU-resident sphere world.
#[derive(Debug, uniffi::Object)]
pub struct GpuSphereWorld {
    world: Mutex<CoreGpuSphereWorld>,
}

#[uniffi::export]
impl GpuSphereWorld {
    /// Limit dynamic bodies' world-space linear speed in metres per second.
    pub fn set_max_linear_speed(&self, max_speed: Option<f32>) -> Result<(), TesseraError> {
        locked(&self.world)?
            .set_max_linear_speed(max_speed)
            .map_err(failed)
    }

    /// Current linear speed limit, or None when disabled.
    pub fn max_linear_speed(&self) -> Result<Option<f32>, TesseraError> {
        Ok(locked(&self.world)?.max_linear_speed())
    }

    /// Read accumulated contact impulses from the last solver substep.
    pub fn readback_contact_impulses(&self) -> Result<GpuContactImpulseReadback, TesseraError> {
        Ok(locked(&self.world)?
            .readback_contact_impulses()
            .map_err(failed)?
            .into())
    }

    /// Construct an independent sphere world on a GPU adapter.
    #[uniffi::constructor]
    pub fn new(
        bodies: Vec<SphereInput>,
        gravity: Vec3,
        ground_half_extent: f32,
    ) -> Result<Arc<Self>, TesseraError> {
        let (states, radii): (Vec<_>, Vec<_>) = bodies
            .into_iter()
            .map(gpu_sphere)
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .unzip();
        let config = gpu_config(gravity, ground_half_extent)?;
        let gpu = GpuContactDevice::new().map_err(failed)?;
        let world = CoreGpuSphereWorld::new(gpu.device(), gpu.queue(), &states, &radii, config)
            .map_err(failed)?;
        Ok(Arc::new(Self {
            world: Mutex::new(world),
        }))
    }

    /// Number of spheres in the current dense index order.
    pub fn len(&self) -> Result<u32, TesseraError> {
        u32::try_from(locked(&self.world)?.len()).map_err(failed)
    }

    /// Whether this world currently has no spheres.
    pub fn is_empty(&self) -> Result<bool, TesseraError> {
        Ok(locked(&self.world)?.is_empty())
    }

    /// Collision radius of one sphere.
    pub fn radius(&self, index: u32) -> Result<f64, TesseraError> {
        locked(&self.world)?
            .radius(index as usize)
            .map(f64::from)
            .ok_or_else(|| failed("body index out of range"))
    }

    /// Replace point-to-point joints between spheres in this world.
    pub fn set_ball_joints(&self, joints: Vec<GpuBallJoint>) -> Result<(), TesseraError> {
        let joints = joints.into_iter().map(Into::into).collect::<Vec<_>>();
        locked(&self.world)?
            .set_ball_joints(&joints)
            .map_err(failed)
    }

    /// Current point-to-point joints in stable insertion order.
    pub fn ball_joints(&self) -> Result<Vec<GpuBallJoint>, TesseraError> {
        Ok(locked(&self.world)?
            .ball_joints()
            .iter()
            .copied()
            .map(Into::into)
            .collect())
    }

    /// Replace position and orientation locks between spheres in this world.
    pub fn set_fixed_joints(&self, joints: Vec<GpuFixedJoint>) -> Result<(), TesseraError> {
        let joints = joints.into_iter().map(Into::into).collect::<Vec<_>>();
        locked(&self.world)?
            .set_fixed_joints(&joints)
            .map_err(failed)
    }

    /// Current fixed joints in stable insertion order.
    pub fn fixed_joints(&self) -> Result<Vec<GpuFixedJoint>, TesseraError> {
        Ok(locked(&self.world)?
            .fixed_joints()
            .iter()
            .copied()
            .map(Into::into)
            .collect())
    }

    /// Replace hinge joints between spheres in this world.
    pub fn set_revolute_joints(&self, joints: Vec<GpuRevoluteJoint>) -> Result<(), TesseraError> {
        let joints = joints.into_iter().map(Into::into).collect::<Vec<_>>();
        locked(&self.world)?
            .set_revolute_joints(&joints)
            .map_err(failed)
    }

    /// Current revolute joints in stable insertion order.
    pub fn revolute_joints(&self) -> Result<Vec<GpuRevoluteJoint>, TesseraError> {
        Ok(locked(&self.world)?
            .revolute_joints()
            .iter()
            .copied()
            .map(Into::into)
            .collect())
    }

    /// Replace slider joints between spheres in this world.
    pub fn set_prismatic_joints(&self, joints: Vec<GpuPrismaticJoint>) -> Result<(), TesseraError> {
        let joints = joints.into_iter().map(Into::into).collect::<Vec<_>>();
        locked(&self.world)?
            .set_prismatic_joints(&joints)
            .map_err(failed)
    }

    /// Current slider joints in stable insertion order.
    pub fn prismatic_joints(&self) -> Result<Vec<GpuPrismaticJoint>, TesseraError> {
        Ok(locked(&self.world)?
            .prismatic_joints()
            .iter()
            .copied()
            .map(Into::into)
            .collect())
    }

    /// Set or clear a hinge's velocity motor.
    pub fn set_revolute_motor(
        &self,
        index: u32,
        motor: Option<GpuAxisMotor>,
    ) -> Result<(), TesseraError> {
        locked(&self.world)?
            .set_revolute_motor(index as usize, motor.map(Into::into))
            .map_err(failed)
    }

    /// Current hinge velocity motor.
    pub fn revolute_motor(&self, index: u32) -> Result<Option<GpuAxisMotor>, TesseraError> {
        Ok(locked(&self.world)?
            .revolute_motor(index as usize)
            .map_err(failed)?
            .map(Into::into))
    }

    /// Set or clear a slider's velocity motor.
    pub fn set_prismatic_motor(
        &self,
        index: u32,
        motor: Option<GpuAxisMotor>,
    ) -> Result<(), TesseraError> {
        locked(&self.world)?
            .set_prismatic_motor(index as usize, motor.map(Into::into))
            .map_err(failed)
    }

    /// Current slider velocity motor.
    pub fn prismatic_motor(&self, index: u32) -> Result<Option<GpuAxisMotor>, TesseraError> {
        Ok(locked(&self.world)?
            .prismatic_motor(index as usize)
            .map_err(failed)?
            .map(Into::into))
    }

    /// Set or clear a slider's displacement limits.
    pub fn set_prismatic_limit(
        &self,
        index: u32,
        limit: Option<GpuPrismaticLimit>,
    ) -> Result<(), TesseraError> {
        locked(&self.world)?
            .set_prismatic_limit(index as usize, limit.map(Into::into))
            .map_err(failed)
    }

    /// Current slider displacement limits.
    pub fn prismatic_limit(&self, index: u32) -> Result<Option<GpuPrismaticLimit>, TesseraError> {
        Ok(locked(&self.world)?
            .prismatic_limit(index as usize)
            .map_err(failed)?
            .map(Into::into))
    }

    /// Set or clear a hinge's position servo.
    pub fn set_revolute_servo(
        &self,
        index: u32,
        servo: Option<GpuAxisServo>,
    ) -> Result<(), TesseraError> {
        locked(&self.world)?
            .set_revolute_servo(index as usize, servo.map(Into::into))
            .map_err(failed)
    }

    /// Current hinge position servo.
    pub fn revolute_servo(&self, index: u32) -> Result<Option<GpuAxisServo>, TesseraError> {
        Ok(locked(&self.world)?
            .revolute_servo(index as usize)
            .map_err(failed)?
            .map(Into::into))
    }

    /// Set or clear a hinge's continuous angle limits.
    pub fn set_revolute_limit(
        &self,
        index: u32,
        limit: Option<GpuRevoluteLimit>,
    ) -> Result<(), TesseraError> {
        locked(&self.world)?
            .set_revolute_limit(index as usize, limit.map(Into::into))
            .map_err(failed)
    }

    /// Current hinge angle limits.
    pub fn revolute_limit(&self, index: u32) -> Result<Option<GpuRevoluteLimit>, TesseraError> {
        Ok(locked(&self.world)?
            .revolute_limit(index as usize)
            .map_err(failed)?
            .map(Into::into))
    }

    /// Read a hinge's continuous angle in radians.
    pub fn readback_revolute_angle(&self, index: u32) -> Result<f32, TesseraError> {
        locked(&self.world)?
            .readback_revolute_angle(index as usize)
            .map_err(failed)
    }

    /// Set or clear a slider's position servo.
    pub fn set_prismatic_servo(
        &self,
        index: u32,
        servo: Option<GpuAxisServo>,
    ) -> Result<(), TesseraError> {
        locked(&self.world)?
            .set_prismatic_servo(index as usize, servo.map(Into::into))
            .map_err(failed)
    }

    /// Current slider position servo.
    pub fn prismatic_servo(&self, index: u32) -> Result<Option<GpuAxisServo>, TesseraError> {
        Ok(locked(&self.world)?
            .prismatic_servo(index as usize)
            .map_err(failed)?
            .map(Into::into))
    }

    /// Append a sphere, returning its dense index.
    ///
    /// This synchronizes and rebuilds GPU contact buffers. Queued forces and
    /// warm-start impulses are discarded.
    pub fn add_body(&self, body: SphereInput) -> Result<u32, TesseraError> {
        let (state, radius) = gpu_sphere(body)?;
        let index = locked(&self.world)?
            .append_body(state, radius)
            .map_err(failed)?;
        u32::try_from(index).map_err(failed)
    }

    /// Remove a sphere and shift all later dense indices down by one.
    ///
    /// This synchronizes and rebuilds GPU contact buffers. Queued forces and
    /// warm-start impulses are discarded.
    pub fn remove_body(&self, index: u32) -> Result<RemovedGpuSphere, TesseraError> {
        let mut world = locked(&self.world)?;
        let radius = world
            .radius(index as usize)
            .ok_or_else(|| failed("body index out of range"))?;
        let state = world.remove_body(index as usize).map_err(failed)?;
        Ok(RemovedGpuSphere {
            state: state.into(),
            radius: f64::from(radius),
        })
    }

    /// Discard a sphere without reading its GPU state back.
    pub fn discard_body(&self, index: u32) -> Result<(), TesseraError> {
        locked(&self.world)?
            .discard_body(index as usize)
            .map_err(failed)
    }

    /// Advance one substep and return the candidate count.
    pub fn step(&self, dt: f32) -> Result<u32, TesseraError> {
        let count = locked(&self.world)?.step(dt).map_err(failed)?;
        u32::try_from(count).map_err(failed)
    }

    /// Advance several substeps in one call.
    pub fn step_substeps(&self, dt: f32, count: u32) -> Result<u32, TesseraError> {
        let candidates = locked(&self.world)?
            .step_substeps(dt, count)
            .map_err(failed)?;
        u32::try_from(candidates).map_err(failed)
    }

    /// Advance one temporal frame. Forces are captured once for all substeps.
    /// `None` selects defaults with zero speculative margin.
    pub fn step_temporal(
        &self,
        frame_dt: f32,
        substeps: u32,
        settings: Option<GpuTemporalSettings>,
    ) -> Result<u32, TesseraError> {
        let mut state = locked(&self.world)?;
        temporal_step_binding(&mut state, frame_dt, substeps, settings, None)
    }

    /// Advance a sphere-only temporal frame using the configured speed cap.
    pub fn step_temporal_speed_bounded(
        &self,
        frame_dt: f32,
        substeps: u32,
        settings: Option<GpuTemporalSettings>,
        joint_settings: Option<GpuTemporalJointSettings>,
    ) -> Result<u32, TesseraError> {
        let mut state = locked(&self.world)?;
        temporal_step_speed_bounded_binding(
            &mut state,
            frame_dt,
            substeps,
            settings,
            joint_settings,
        )
    }

    /// Advance a temporal frame with independent contact and joint coefficients.
    pub fn step_temporal_with_joints(
        &self,
        frame_dt: f32,
        substeps: u32,
        settings: Option<GpuTemporalSettings>,
        joint_settings: GpuTemporalJointSettings,
    ) -> Result<u32, TesseraError> {
        let mut state = locked(&self.world)?;
        temporal_step_binding(
            &mut state,
            frame_dt,
            substeps,
            settings,
            Some(joint_settings),
        )
    }

    /// Apply a force to one body for its next step.
    pub fn write_force(&self, index: u32, force: Vec3) -> Result<(), TesseraError> {
        self.write_wrench(
            index,
            force,
            Vec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
        )
    }

    /// Set prescribed motion without reading the pose; None restores static behavior.
    /// Dynamic bodies ignore the command. Nonfinite velocities are rejected.
    pub fn set_kinematic_motion(
        &self,
        index: u32,
        motion: Option<GpuKinematicMotion>,
    ) -> Result<(), TesseraError> {
        locked(&self.world)?
            .set_kinematic_motion(index as usize, motion.map(GpuKinematicMotion::gpu))
            .map_err(failed)
    }

    /// Apply a force and torque to one body for its next step.
    pub fn write_wrench(&self, index: u32, force: Vec3, torque: Vec3) -> Result<(), TesseraError> {
        let [x, y, z] = force.gpu();
        let [tx, ty, tz] = torque.gpu();
        locked(&self.world)?
            .write_forces(
                index as usize,
                GpuRigidBodyForces {
                    force: [x, y, z, 0.0],
                    torque: [tx, ty, tz, 0.0],
                },
            )
            .map_err(failed)
    }

    /// Override one sphere's contact material.
    pub fn set_body_material(&self, index: u32, material: Material) -> Result<(), TesseraError> {
        locked(&self.world)?
            .set_body_material(index as usize, material.into())
            .map_err(failed)
    }

    /// Restore the configured default material for one sphere.
    pub fn clear_body_material(&self, index: u32) -> Result<(), TesseraError> {
        locked(&self.world)?
            .clear_body_material(index as usize)
            .map_err(failed)
    }

    /// Set reciprocal collision masks for one sphere.
    pub fn set_body_collision_groups(
        &self,
        index: u32,
        groups: GpuCollisionGroups,
    ) -> Result<(), TesseraError> {
        locked(&self.world)?
            .set_body_collision_groups(index as usize, groups.into())
            .map_err(failed)
    }

    /// Read reciprocal collision masks for one sphere.
    pub fn body_collision_groups(&self, index: u32) -> Result<GpuCollisionGroups, TesseraError> {
        locked(&self.world)?
            .body_collision_groups(index as usize)
            .map(Into::into)
            .ok_or_else(|| failed("body index out of range"))
    }

    /// Override the finite ground plane's contact material.
    pub fn set_ground_material(&self, material: Material) -> Result<(), TesseraError> {
        locked(&self.world)?
            .set_ground_material(material.into())
            .map_err(failed)
    }

    /// Restore the configured default ground material.
    pub fn clear_ground_material(&self) -> Result<(), TesseraError> {
        locked(&self.world)?.clear_ground_material().map_err(failed)
    }

    /// Set reciprocal collision masks for the finite ground.
    pub fn set_ground_collision_groups(
        &self,
        groups: GpuCollisionGroups,
    ) -> Result<(), TesseraError> {
        locked(&self.world)?
            .set_ground_collision_groups(groups.into())
            .map_err(failed)
    }

    /// Read reciprocal collision masks for the finite ground.
    pub fn ground_collision_groups(&self) -> Result<GpuCollisionGroups, TesseraError> {
        locked(&self.world)?
            .ground_collision_groups()
            .map(Into::into)
            .ok_or_else(|| failed("ground is disabled"))
    }

    /// Cast rays against current resident geometry without transferring body state.
    pub fn cast_rays(&self, rays: Vec<GpuRay>) -> Result<Vec<Option<GpuRayHit>>, TesseraError> {
        let state = locked(&self.world)?;
        let hits = (*state).cast_rays(&gpu_rays(rays)).map_err(failed)?;
        Ok(gpu_ray_hits(hits))
    }

    /// Project points against current resident geometry without transferring body state.
    pub fn project_points(
        &self,
        points: Vec<GpuPointQuery>,
    ) -> Result<Vec<Option<GpuPointHit>>, TesseraError> {
        let state = locked(&self.world)?;
        let hits = (*state)
            .project_points(&gpu_points(points))
            .map_err(failed)?;
        Ok(gpu_point_hits(hits))
    }

    /// Evaluate rays and points with one current-state GPU scene tree.
    pub fn query_scene(
        &self,
        rays: Vec<GpuRay>,
        points: Vec<GpuPointQuery>,
    ) -> Result<GpuSceneQueryHits, TesseraError> {
        let state = locked(&self.world)?;
        let hits = (*state)
            .query_scene(&gpu_rays(rays), &gpu_points(points))
            .map_err(failed)?;
        Ok(gpu_scene_query_hits(hits))
    }

    /// Transfer all states to the CPU.
    pub fn readback(&self) -> Result<Vec<GpuSphereState>, TesseraError> {
        Ok(locked(&self.world)?
            .readback()
            .map_err(failed)?
            .into_iter()
            .map(Into::into)
            .collect())
    }
}

/// Independent GPU sphere environments packed into one state buffer.
#[derive(Debug, uniffi::Object)]
pub struct GpuSphereBatch {
    batch: Mutex<GpuRigidSphereBatch>,
    radii: Mutex<Vec<Vec<f32>>>,
}

#[uniffi::export]
impl GpuSphereBatch {
    /// Set a shared linear speed limit for all environments.
    pub fn set_max_linear_speed(&self, max_speed: Option<f32>) -> Result<(), TesseraError> {
        locked(&self.batch)?
            .world_mut()
            .set_max_linear_speed(max_speed)
            .map_err(failed)
    }

    /// Current shared linear speed limit, or None when disabled.
    pub fn max_linear_speed(&self) -> Result<Option<f32>, TesseraError> {
        Ok(locked(&self.batch)?.world().max_linear_speed())
    }

    /// Read contact impulse history with environment-local body IDs.
    pub fn readback_contact_impulses_environment(
        &self,
        environment: u32,
    ) -> Result<GpuContactImpulseReadback, TesseraError> {
        Ok(locked(&self.batch)?
            .readback_contact_impulses_environment(environment as usize)
            .map_err(failed)?
            .into())
    }

    /// Construct a batch. Each nested body list is one environment.
    #[uniffi::constructor]
    pub fn new(
        environments: Vec<Vec<SphereInput>>,
        gravity: Vec3,
        ground_half_extent: f32,
    ) -> Result<Arc<Self>, TesseraError> {
        let mut groups = Vec::with_capacity(environments.len());
        for bodies in environments {
            let (states, radii): (Vec<_>, Vec<_>) = bodies
                .into_iter()
                .map(gpu_sphere)
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .unzip();
            groups.push((states, radii));
        }
        let config = gpu_config(gravity, ground_half_extent)?;
        let gpu = GpuContactDevice::new().map_err(failed)?;
        let environments = groups
            .iter()
            .map(|(states, radii)| GpuRigidSphereEnvironment { states, radii })
            .collect::<Vec<_>>();
        let batch = GpuRigidSphereBatch::new(gpu.device(), gpu.queue(), &environments, config)
            .map_err(failed)?;
        let radii = groups.into_iter().map(|(_, radii)| radii).collect();
        Ok(Arc::new(Self {
            batch: Mutex::new(batch),
            radii: Mutex::new(radii),
        }))
    }

    /// Number of independent environments.
    pub fn len(&self) -> Result<u32, TesseraError> {
        u32::try_from(locked(&self.batch)?.len()).map_err(failed)
    }

    /// Whether this batch has no environments.
    pub fn is_empty(&self) -> Result<bool, TesseraError> {
        Ok(locked(&self.batch)?.is_empty())
    }

    /// Append one independent sphere environment, including an empty environment.
    pub fn add_environment(&self, bodies: Vec<SphereInput>) -> Result<u32, TesseraError> {
        let (states, radii): (Vec<_>, Vec<_>) = bodies
            .into_iter()
            .map(gpu_sphere)
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .unzip();
        let mut batch = locked(&self.batch)?;
        let mut stored_radii = locked(&self.radii)?;
        let index = batch.append_environment(&states, &radii).map_err(failed)?;
        stored_radii.push(radii);
        u32::try_from(index).map_err(failed)
    }

    /// Remove an environment and return its last body states and radii.
    pub fn remove_environment(
        &self,
        environment: u32,
    ) -> Result<Vec<RemovedGpuSphere>, TesseraError> {
        let mut batch = locked(&self.batch)?;
        let mut radii = locked(&self.radii)?;
        let states = batch
            .remove_environment(environment as usize)
            .map_err(failed)?;
        let removed_radii = radii.remove(environment as usize);
        Ok(states
            .into_iter()
            .zip(removed_radii)
            .map(|(state, radius)| RemovedGpuSphere {
                state: state.into(),
                radius: f64::from(radius),
            })
            .collect())
    }

    /// Discard one environment without reading body states back.
    pub fn discard_environment(&self, environment: u32) -> Result<(), TesseraError> {
        let mut batch = locked(&self.batch)?;
        let mut radii = locked(&self.radii)?;
        batch
            .discard_environment(environment as usize)
            .map_err(failed)?;
        let _removed = radii.remove(environment as usize);
        Ok(())
    }

    /// Append a sphere to one environment and return its local body index.
    pub fn add_body(&self, environment: u32, body: SphereInput) -> Result<u32, TesseraError> {
        let (state, radius) = gpu_sphere(body)?;
        let mut batch = locked(&self.batch)?;
        let mut radii = locked(&self.radii)?;
        let index = batch
            .append_body_environment(environment as usize, state, radius)
            .map_err(failed)?;
        radii
            .get_mut(environment as usize)
            .ok_or_else(|| failed("environment index out of range"))?
            .push(radius);
        u32::try_from(index).map_err(failed)
    }

    /// Remove an environment-local sphere and return its last state and radius.
    pub fn remove_body(
        &self,
        environment: u32,
        body: u32,
    ) -> Result<RemovedGpuSphere, TesseraError> {
        let mut batch = locked(&self.batch)?;
        let mut radii = locked(&self.radii)?;
        let state = batch
            .remove_body_environment(environment as usize, body as usize)
            .map_err(failed)?;
        let radius = radii
            .get_mut(environment as usize)
            .ok_or_else(|| failed("environment index out of range"))?
            .remove(body as usize);
        Ok(RemovedGpuSphere {
            state: state.into(),
            radius: f64::from(radius),
        })
    }

    /// Discard an environment-local sphere without reading its GPU state back.
    pub fn discard_body(&self, environment: u32, body: u32) -> Result<(), TesseraError> {
        let mut batch = locked(&self.batch)?;
        let mut radii = locked(&self.radii)?;
        batch
            .discard_body_environment(environment as usize, body as usize)
            .map_err(failed)?;
        let _removed = radii[environment as usize].remove(body as usize);
        Ok(())
    }

    /// Replace one environment's point-to-point joints using local body IDs.
    pub fn set_ball_joints_environment(
        &self,
        environment: u32,
        joints: Vec<GpuBallJoint>,
    ) -> Result<(), TesseraError> {
        let joints = joints.into_iter().map(Into::into).collect::<Vec<_>>();
        locked(&self.batch)?
            .set_ball_joints_environment(environment as usize, &joints)
            .map_err(failed)
    }

    /// One environment's point-to-point joints using local body IDs.
    pub fn ball_joints_environment(
        &self,
        environment: u32,
    ) -> Result<Vec<GpuBallJoint>, TesseraError> {
        Ok(locked(&self.batch)?
            .ball_joints_environment(environment as usize)
            .map_err(failed)?
            .into_iter()
            .map(Into::into)
            .collect())
    }

    /// Replace one environment's fixed joints using local body IDs.
    pub fn set_fixed_joints_environment(
        &self,
        environment: u32,
        joints: Vec<GpuFixedJoint>,
    ) -> Result<(), TesseraError> {
        let joints = joints.into_iter().map(Into::into).collect::<Vec<_>>();
        locked(&self.batch)?
            .set_fixed_joints_environment(environment as usize, &joints)
            .map_err(failed)
    }

    /// One environment's fixed joints using local body IDs.
    pub fn fixed_joints_environment(
        &self,
        environment: u32,
    ) -> Result<Vec<GpuFixedJoint>, TesseraError> {
        Ok(locked(&self.batch)?
            .fixed_joints_environment(environment as usize)
            .map_err(failed)?
            .into_iter()
            .map(Into::into)
            .collect())
    }

    /// Replace one environment's revolute joints using local body IDs.
    pub fn set_revolute_joints_environment(
        &self,
        environment: u32,
        joints: Vec<GpuRevoluteJoint>,
    ) -> Result<(), TesseraError> {
        let joints = joints.into_iter().map(Into::into).collect::<Vec<_>>();
        locked(&self.batch)?
            .set_revolute_joints_environment(environment as usize, &joints)
            .map_err(failed)
    }

    /// One environment's revolute joints using local body IDs.
    pub fn revolute_joints_environment(
        &self,
        environment: u32,
    ) -> Result<Vec<GpuRevoluteJoint>, TesseraError> {
        Ok(locked(&self.batch)?
            .revolute_joints_environment(environment as usize)
            .map_err(failed)?
            .into_iter()
            .map(Into::into)
            .collect())
    }

    /// Replace one environment's slider joints using local body IDs.
    pub fn set_prismatic_joints_environment(
        &self,
        environment: u32,
        joints: Vec<GpuPrismaticJoint>,
    ) -> Result<(), TesseraError> {
        let joints = joints.into_iter().map(Into::into).collect::<Vec<_>>();
        locked(&self.batch)?
            .set_prismatic_joints_environment(environment as usize, &joints)
            .map_err(failed)
    }

    /// One environment's slider joints with local body IDs.
    pub fn prismatic_joints_environment(
        &self,
        environment: u32,
    ) -> Result<Vec<GpuPrismaticJoint>, TesseraError> {
        Ok(locked(&self.batch)?
            .prismatic_joints_environment(environment as usize)
            .map_err(failed)?
            .into_iter()
            .map(Into::into)
            .collect())
    }

    /// Set one environment-local hinge's velocity motor.
    pub fn set_revolute_motor_environment(
        &self,
        environment: u32,
        index: u32,
        motor: Option<GpuAxisMotor>,
    ) -> Result<(), TesseraError> {
        locked(&self.batch)?
            .set_revolute_motor_environment(
                environment as usize,
                index as usize,
                motor.map(Into::into),
            )
            .map_err(failed)
    }

    /// Read one environment-local hinge's velocity motor.
    pub fn revolute_motor_environment(
        &self,
        environment: u32,
        index: u32,
    ) -> Result<Option<GpuAxisMotor>, TesseraError> {
        Ok(locked(&self.batch)?
            .revolute_motor_environment(environment as usize, index as usize)
            .map_err(failed)?
            .map(Into::into))
    }

    /// Set one environment-local slider's velocity motor.
    pub fn set_prismatic_motor_environment(
        &self,
        environment: u32,
        index: u32,
        motor: Option<GpuAxisMotor>,
    ) -> Result<(), TesseraError> {
        locked(&self.batch)?
            .set_prismatic_motor_environment(
                environment as usize,
                index as usize,
                motor.map(Into::into),
            )
            .map_err(failed)
    }

    /// Read one environment-local slider's velocity motor.
    pub fn prismatic_motor_environment(
        &self,
        environment: u32,
        index: u32,
    ) -> Result<Option<GpuAxisMotor>, TesseraError> {
        Ok(locked(&self.batch)?
            .prismatic_motor_environment(environment as usize, index as usize)
            .map_err(failed)?
            .map(Into::into))
    }

    /// Set one environment-local slider's displacement limits.
    pub fn set_prismatic_limit_environment(
        &self,
        environment: u32,
        index: u32,
        limit: Option<GpuPrismaticLimit>,
    ) -> Result<(), TesseraError> {
        locked(&self.batch)?
            .set_prismatic_limit_environment(
                environment as usize,
                index as usize,
                limit.map(Into::into),
            )
            .map_err(failed)
    }

    /// Read one environment-local slider's displacement limits.
    pub fn prismatic_limit_environment(
        &self,
        environment: u32,
        index: u32,
    ) -> Result<Option<GpuPrismaticLimit>, TesseraError> {
        Ok(locked(&self.batch)?
            .prismatic_limit_environment(environment as usize, index as usize)
            .map_err(failed)?
            .map(Into::into))
    }

    /// Set one environment-local hinge's position servo.
    pub fn set_revolute_servo_environment(
        &self,
        environment: u32,
        index: u32,
        servo: Option<GpuAxisServo>,
    ) -> Result<(), TesseraError> {
        locked(&self.batch)?
            .set_revolute_servo_environment(
                environment as usize,
                index as usize,
                servo.map(Into::into),
            )
            .map_err(failed)
    }

    /// Read one environment-local hinge's position servo.
    pub fn revolute_servo_environment(
        &self,
        environment: u32,
        index: u32,
    ) -> Result<Option<GpuAxisServo>, TesseraError> {
        Ok(locked(&self.batch)?
            .revolute_servo_environment(environment as usize, index as usize)
            .map_err(failed)?
            .map(Into::into))
    }

    /// Set one environment-local hinge's wrapped angle limits.
    pub fn set_revolute_limit_environment(
        &self,
        environment: u32,
        index: u32,
        limit: Option<GpuRevoluteLimit>,
    ) -> Result<(), TesseraError> {
        locked(&self.batch)?
            .set_revolute_limit_environment(
                environment as usize,
                index as usize,
                limit.map(Into::into),
            )
            .map_err(failed)
    }

    /// Read one environment-local hinge's angle limits.
    pub fn revolute_limit_environment(
        &self,
        environment: u32,
        index: u32,
    ) -> Result<Option<GpuRevoluteLimit>, TesseraError> {
        Ok(locked(&self.batch)?
            .revolute_limit_environment(environment as usize, index as usize)
            .map_err(failed)?
            .map(Into::into))
    }

    /// Read one environment-local hinge's continuous angle.
    pub fn readback_revolute_angle_environment(
        &self,
        environment: u32,
        index: u32,
    ) -> Result<f32, TesseraError> {
        locked(&self.batch)?
            .readback_revolute_angle_environment(environment as usize, index as usize)
            .map_err(failed)
    }

    /// Set one environment-local slider's position servo.
    pub fn set_prismatic_servo_environment(
        &self,
        environment: u32,
        index: u32,
        servo: Option<GpuAxisServo>,
    ) -> Result<(), TesseraError> {
        locked(&self.batch)?
            .set_prismatic_servo_environment(
                environment as usize,
                index as usize,
                servo.map(Into::into),
            )
            .map_err(failed)
    }

    /// Read one environment-local slider's position servo.
    pub fn prismatic_servo_environment(
        &self,
        environment: u32,
        index: u32,
    ) -> Result<Option<GpuAxisServo>, TesseraError> {
        Ok(locked(&self.batch)?
            .prismatic_servo_environment(environment as usize, index as usize)
            .map_err(failed)?
            .map(Into::into))
    }

    /// Advance every environment together.
    pub fn step(&self, dt: f32) -> Result<u32, TesseraError> {
        let count = locked(&self.batch)?.step(dt).map_err(failed)?;
        u32::try_from(count).map_err(failed)
    }

    /// Advance every environment through several substeps.
    pub fn step_substeps(&self, dt: f32, count: u32) -> Result<u32, TesseraError> {
        let candidates = locked(&self.batch)?
            .step_substeps(dt, count)
            .map_err(failed)?;
        u32::try_from(candidates).map_err(failed)
    }

    /// Advance one temporal frame. Forces are captured once for all substeps.
    /// `None` selects defaults with zero speculative margin.
    pub fn step_temporal(
        &self,
        frame_dt: f32,
        substeps: u32,
        settings: Option<GpuTemporalSettings>,
    ) -> Result<u32, TesseraError> {
        let mut state = locked(&self.batch)?;
        temporal_step_binding(state.world_mut(), frame_dt, substeps, settings, None)
    }

    /// Advance sphere environments using their shared linear speed cap.
    pub fn step_temporal_speed_bounded(
        &self,
        frame_dt: f32,
        substeps: u32,
        settings: Option<GpuTemporalSettings>,
        joint_settings: Option<GpuTemporalJointSettings>,
    ) -> Result<u32, TesseraError> {
        let mut state = locked(&self.batch)?;
        temporal_step_speed_bounded_binding(
            state.world_mut(),
            frame_dt,
            substeps,
            settings,
            joint_settings,
        )
    }

    /// Advance a temporal frame with independent contact and joint coefficients.
    pub fn step_temporal_with_joints(
        &self,
        frame_dt: f32,
        substeps: u32,
        settings: Option<GpuTemporalSettings>,
        joint_settings: GpuTemporalJointSettings,
    ) -> Result<u32, TesseraError> {
        let mut state = locked(&self.batch)?;
        temporal_step_binding(
            state.world_mut(),
            frame_dt,
            substeps,
            settings,
            Some(joint_settings),
        )
    }

    /// Apply force to one body within one environment.
    pub fn write_force(
        &self,
        environment: u32,
        body: u32,
        force: Vec3,
    ) -> Result<(), TesseraError> {
        self.write_wrench(
            environment,
            body,
            force,
            Vec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
        )
    }

    /// Set environment-local prescribed motion; None restores static behavior.
    /// Dynamic bodies ignore the command. Nonfinite velocities are rejected.
    pub fn set_kinematic_motion(
        &self,
        environment: u32,
        body: u32,
        motion: Option<GpuKinematicMotion>,
    ) -> Result<(), TesseraError> {
        locked(&self.batch)?
            .set_kinematic_motion(
                environment as usize,
                body as usize,
                motion.map(GpuKinematicMotion::gpu),
            )
            .map_err(failed)
    }

    /// Apply force and torque to a body within one environment.
    pub fn write_wrench(
        &self,
        environment: u32,
        body: u32,
        force: Vec3,
        torque: Vec3,
    ) -> Result<(), TesseraError> {
        let [x, y, z] = force.gpu();
        let [tx, ty, tz] = torque.gpu();
        locked(&self.batch)?
            .write_forces(
                environment as usize,
                body as usize,
                GpuRigidBodyForces {
                    force: [x, y, z, 0.0],
                    torque: [tx, ty, tz, 0.0],
                },
            )
            .map_err(failed)
    }

    /// Override the material of one sphere within an environment.
    pub fn set_body_material(
        &self,
        environment: u32,
        body: u32,
        material: Material,
    ) -> Result<(), TesseraError> {
        let mut batch = locked(&self.batch)?;
        let range = batch
            .environment_range(environment as usize)
            .ok_or_else(|| failed("environment index out of range"))?;
        if body as usize >= range.len() {
            return Err(failed("body index out of range"));
        }
        batch
            .world_mut()
            .set_body_material(range.start + body as usize, material.into())
            .map_err(failed)
    }

    /// Restore the default material of one sphere within an environment.
    pub fn clear_body_material(&self, environment: u32, body: u32) -> Result<(), TesseraError> {
        let mut batch = locked(&self.batch)?;
        let range = batch
            .environment_range(environment as usize)
            .ok_or_else(|| failed("environment index out of range"))?;
        if body as usize >= range.len() {
            return Err(failed("body index out of range"));
        }
        batch
            .world_mut()
            .clear_body_material(range.start + body as usize)
            .map_err(failed)
    }

    /// Set reciprocal collision masks for one environment-local sphere.
    pub fn set_body_collision_groups(
        &self,
        environment: u32,
        body: u32,
        groups: GpuCollisionGroups,
    ) -> Result<(), TesseraError> {
        locked(&self.batch)?
            .set_body_collision_groups_environment(
                environment as usize,
                body as usize,
                groups.into(),
            )
            .map_err(failed)
    }

    /// Read reciprocal collision masks for one environment-local sphere.
    pub fn body_collision_groups(
        &self,
        environment: u32,
        body: u32,
    ) -> Result<GpuCollisionGroups, TesseraError> {
        locked(&self.batch)?
            .body_collision_groups_environment(environment as usize, body as usize)
            .map(Into::into)
            .ok_or_else(|| failed("body index out of range"))
    }

    /// Override the finite ground material shared by all environments.
    pub fn set_ground_material(&self, material: Material) -> Result<(), TesseraError> {
        locked(&self.batch)?
            .world_mut()
            .set_ground_material(material.into())
            .map_err(failed)
    }

    /// Restore the configured default ground material for all environments.
    pub fn clear_ground_material(&self) -> Result<(), TesseraError> {
        locked(&self.batch)?
            .world_mut()
            .clear_ground_material()
            .map_err(failed)
    }

    /// Set reciprocal collision masks for the shared finite ground.
    pub fn set_ground_collision_groups(
        &self,
        groups: GpuCollisionGroups,
    ) -> Result<(), TesseraError> {
        locked(&self.batch)?
            .world_mut()
            .set_ground_collision_groups(groups.into())
            .map_err(failed)
    }

    /// Cast rays in one environment; hit and exclusion IDs are environment-local.
    pub fn cast_rays_environment(
        &self,
        environment: u32,
        rays: Vec<GpuRay>,
    ) -> Result<Vec<Option<GpuRayHit>>, TesseraError> {
        let hits = locked(&self.batch)?
            .cast_rays_environment(environment as usize, &gpu_rays(rays))
            .map_err(failed)?;
        Ok(gpu_ray_hits(hits))
    }

    /// Project points in one environment; hit and exclusion IDs are environment-local.
    pub fn project_points_environment(
        &self,
        environment: u32,
        points: Vec<GpuPointQuery>,
    ) -> Result<Vec<Option<GpuPointHit>>, TesseraError> {
        let hits = locked(&self.batch)?
            .project_points_environment(environment as usize, &gpu_points(points))
            .map_err(failed)?;
        Ok(gpu_point_hits(hits))
    }

    /// Evaluate rays and points in one environment with one GPU scene tree.
    pub fn query_scene_environment(
        &self,
        environment: u32,
        rays: Vec<GpuRay>,
        points: Vec<GpuPointQuery>,
    ) -> Result<GpuSceneQueryHits, TesseraError> {
        let hits = locked(&self.batch)?
            .query_scene_environment(environment as usize, &gpu_rays(rays), &gpu_points(points))
            .map_err(failed)?;
        Ok(gpu_scene_query_hits(hits))
    }

    /// Evaluate all environments against one current-state GPU scene snapshot.
    pub fn query_scene_environments(
        &self,
        rays: Vec<Vec<GpuRay>>,
        points: Vec<Vec<GpuPointQuery>>,
    ) -> Result<Vec<GpuSceneQueryHits>, TesseraError> {
        let batch = locked(&self.batch)?;
        gpu_scene_query_hits_environments(&batch, rays, points)
    }

    /// Transfer one environment without reading the others.
    pub fn readback_environment(
        &self,
        environment: u32,
    ) -> Result<Vec<GpuSphereState>, TesseraError> {
        Ok(locked(&self.batch)?
            .readback_environment(environment as usize)
            .map_err(failed)?
            .into_iter()
            .map(Into::into)
            .collect())
    }

    /// Replace one environment while preserving every other state and cache.
    pub fn reset_environment(
        &self,
        environment: u32,
        bodies: Vec<SphereInput>,
    ) -> Result<(), TesseraError> {
        let (states, radii): (Vec<_>, Vec<_>) = bodies
            .into_iter()
            .map(gpu_sphere)
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .unzip();
        let mut batch = locked(&self.batch)?;
        if locked(&self.radii)?.get(environment as usize) != Some(&radii) {
            return Err(failed("environment body count or sphere radius changed"));
        }
        batch
            .reset_environment(environment as usize, &states)
            .map_err(failed)
    }

    /// Reset all environments, including after an incomplete GPU step.
    pub fn reset_all(&self, environments: Vec<Vec<SphereInput>>) -> Result<(), TesseraError> {
        let mut batch = locked(&self.batch)?;
        let radii_config = locked(&self.radii)?;
        if environments.len() != radii_config.len() {
            return Err(failed("environment count changed"));
        }
        let mut states = Vec::new();
        for (environment, bodies) in environments.into_iter().enumerate() {
            let (next_states, radii): (Vec<_>, Vec<_>) = bodies
                .into_iter()
                .map(gpu_sphere)
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .unzip();
            if radii_config[environment] != radii {
                return Err(failed("environment body count or sphere radius changed"));
            }
            states.extend(next_states);
        }
        batch.reset_all(&states).map_err(failed)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn gpu_point_hit_carries_projection_normal() {
        let hits = gpu_point_hits(vec![tessera_physics::gpu_point_query::GpuRigidPointHit {
            point_distance: [1.0, 2.0, 3.0, 0.5],
            normal: [0.0, 0.0, 1.0, 0.0],
            ids: [4, 0, 1, 0],
        }]);
        let hit = hits[0].as_ref().unwrap();
        assert_eq!(hit.body, Some(4));
        assert_eq!(hit.normal.z, 1.0);
    }

    #[test]
    fn temporal_settings_forward_static_contact_softness() {
        let mut settings = default_gpu_temporal_settings();
        assert!(settings.static_normal_frequency > settings.normal_frequency);
        settings.static_normal_frequency = 85.0;
        settings.static_damping_ratio = 2.5;
        let solve: GpuRigidTemporalSolveParams = settings.into();
        assert_eq!(solve.static_normal_frequency, 85.0);
        assert_eq!(solve.static_damping_ratio, 2.5);
    }

    fn v(x: f64, y: f64, z: f64) -> Vec3 {
        Vec3 { x, y, z }
    }

    #[test]
    fn cpu_scene_wrench_and_impulse_binding_updates_native_state() {
        // Given: a dynamic scene sphere with known mass and isotropic inertia.
        let world = ArticulatedWorld::from_urdf(
            "<robot name=\"root\"><link name=\"base\"/></robot>".into(),
            false,
        )
        .unwrap();
        let pose = LinkPose {
            position: v(0.0, 0.0, 10.0),
            orientation: Quaternion {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 1.0,
            },
        };
        let body = world.add_scene_sphere(pose, 0.5, 2.0).unwrap();
        world
            .set_scene_body_wrench(body, v(6.0, 0.0, 0.0), v(0.0, 0.0, 0.4))
            .unwrap();

        // When: impulses and one native substep act on that body.
        world
            .apply_scene_body_impulse(body, v(4.0, 0.0, 0.0), v(0.0, 0.0, 0.2))
            .unwrap();
        world.step(0.001, vec![]).unwrap();

        // Then: the binding exposes both persistent loads and the correct velocities.
        let state = world.scene_body_state(body).unwrap();
        assert!((state.linear_velocity.x - 2.003).abs() < 1e-12);
        assert!((state.angular_velocity.z - 1.002).abs() < 1e-12);
        assert!((state.force.x - 6.0).abs() < 1e-12);
        assert!((state.torque.z - 0.4).abs() < 1e-12);
        assert!(
            world
                .set_scene_body_wrench(body, v(f64::NAN, 0.0, 0.0), v(0.0, 0.0, 0.0))
                .is_err()
        );
        assert!((world.scene_body_state(body).unwrap().force.x - 6.0).abs() < 1e-12);
    }

    #[test]
    fn cpu_ik_binding_reachable_partial_limits_errors_and_nonmutation() {
        let xml = r#"<robot name="slider"><link name="base"/><link name="tip">
          <inertial><mass value="1"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial>
          </link><joint name="slide" type="prismatic"><parent link="base"/><child link="tip"/>
          <axis xyz="1 0 0"/><limit lower="-0.5" upper="0.5" effort="10" velocity="10"/></joint></robot>"#;
        let world = ArticulatedWorld::from_urdf(xml.into(), false).unwrap();
        world.set_positions(vec![0.1]).unwrap();
        {
            let mut inner = locked(&world.inner).unwrap();
            inner.world.velocities[0] = 0.2;
            inner.world.contact_forces[1] = Vector3::new(1.0, 2.0, 3.0);
            inner.world.contact_torques[1] = Vector3::new(4.0, 5.0, 6.0);
        }
        let initial = world.kinematics_state().unwrap();
        let before = world.link_poses().unwrap();
        let mut target = IkTarget {
            link: 1,
            pose: isometry_link_pose(Isometry3::translation(0.3, 0.0, 0.0)),
            local_point: v(0.0, 0.0, 0.0),
            constrained_axes: vec![true; 6],
        };
        let result = world
            .inverse_kinematics(initial.clone(), target.clone(), default_ik_config())
            .unwrap();
        assert!(result.converged && result.iterations > 0);
        assert!((result.state.positions[0] - 0.3).abs() < 1e-4);
        let fk = world.forward_kinematics(result.state).unwrap();
        assert!((fk[1].position.x - 0.3).abs() < 1e-4);
        target.pose.position = v(0.4, 7.0, -8.0);
        target.constrained_axes = vec![true, false, false, false, false, false];
        let partial = world
            .inverse_kinematics(initial.clone(), target.clone(), default_ik_config())
            .unwrap();
        assert!(partial.converged);
        assert!((partial.state.positions[0] - 0.4).abs() < 1e-4);
        assert_eq!(&partial.residual[1..], &[0.0; 5]);
        target.pose.position.x = 2.0;
        let unreachable = world
            .inverse_kinematics(initial.clone(), target.clone(), default_ik_config())
            .unwrap();
        assert!(!unreachable.converged);
        assert_eq!(unreachable.state.positions, [0.5]);
        assert_eq!(unreachable.iterations, 100);
        assert!((unreachable.residual[0] - 1.5).abs() < 1e-12);
        target.link = 2;
        assert!(
            world
                .inverse_kinematics(initial.clone(), target.clone(), default_ik_config())
                .is_err()
        );
        target.link = 1;
        let mut config = default_ik_config();
        config.damping = -0.1;
        assert!(
            world
                .inverse_kinematics(initial.clone(), target.clone(), config)
                .is_err()
        );
        target.constrained_axes = vec![true; 5];
        assert!(
            world
                .inverse_kinematics(initial.clone(), target, default_ik_config())
                .is_err()
        );
        let mut invalid = initial.clone();
        invalid.positions.clear();
        assert!(world.forward_kinematics(invalid).is_err());
        assert_eq!(world.positions().unwrap(), initial.positions);
        assert_eq!(world.velocities().unwrap(), [0.2]);
        assert_eq!(
            world.link_poses().unwrap()[1].position.x,
            before[1].position.x
        );
        let wrench = world.link_contact_wrench(1).unwrap();
        assert_eq!(wrench.force.z, 3.0);
        assert_eq!(wrench.torque.z, 6.0);
    }

    #[test]
    fn mjcf_actuator_binding_preserves_metadata_and_native_controls() {
        let xml = r#"<mujoco><option gravity="0 0 0"/><worldbody>
          <body name="root"><inertial mass="1" diaginertia="1 1 1"/>
            <body name="first"><joint name="a" type="slide" axis="1 0 0"/>
              <inertial mass="1" diaginertia="1 1 1"/></body>
            <body name="second"><joint name="b"/>
              <inertial mass="1" diaginertia="1 1 1"/></body>
          </body></worldbody><actuator>
            <motor name="motor_b" joint="b" gear="-2" forcerange="-3 3"/>
            <position name="servo_a" joint="a" kp="25" kv="8" ctrlrange="-1 1"/>
          </actuator></mujoco>"#;
        let world = ArticulatedWorld::from_mjcf(xml.into()).unwrap();
        assert_eq!(world.actuator_names().unwrap(), ["motor_b", "servo_a"]);
        let info = world.actuator_info().unwrap();
        assert_eq!(info[0].coordinate, 1);
        assert_eq!(info[0].gear, -2.0);
        assert_eq!(info[0].force_range, [-3.0, 3.0]);
        assert_eq!(info[1].coordinate, 0);
        assert_eq!(info[1].control_range, [-1.0, 1.0]);
        assert!(matches!(
            info[1].dynamics,
            MjcfActuatorDynamics::Position { kp: 25.0, kv: 8.0 }
        ));
        world.set_controls(vec![2.0, 0.5]).unwrap();
        for invalid in [vec![2.0], vec![2.0, f64::NAN]] {
            assert!(world.set_controls(invalid).is_err());
            assert_eq!(world.controls().unwrap(), [2.0, 0.5]);
            assert_eq!(world.positions().unwrap(), [0.0, 0.0]);
        }
        let mut native = load_mjcf_str(xml, MjcfLoadOptions::default()).unwrap();
        native.set_controls(&[2.0, 0.5]).unwrap();
        for _ in 0..1000 {
            world.step_controls(0.002).unwrap();
            native.step(0.002).unwrap();
        }
        assert_eq!(
            world.positions().unwrap(),
            native.world.positions.as_slice()
        );
        assert_eq!(
            world.velocities().unwrap(),
            native.world.velocities.as_slice()
        );
        assert!((world.positions().unwrap()[0] - 0.5).abs() < 0.01);
        assert!(world.velocities().unwrap()[1] < 0.0);
    }

    #[test]
    fn cpu_ik_binding_preserves_explicit_spherical_state() {
        let xml = r#"<mujoco><worldbody><body name="ball">
          <joint name="ball_joint" type="ball"/><geom type="sphere" size="0.1" mass="1"/>
          </body></worldbody></mujoco>"#;
        let world = ArticulatedWorld::from_mjcf(xml.into()).unwrap();
        {
            let mut inner = locked(&world.inner).unwrap();
            inner
                .world
                .set_tangent_spherical_state(
                    &[
                        tessera_physics::gpu_articulated_spherical::GpuSphericalJointState {
                            velocity_slot: 0,
                            orientation: UnitQuaternion::identity(),
                        },
                    ],
                    &nalgebra::DVector::zeros(3),
                )
                .unwrap();
        }
        let initial = world.kinematics_state().unwrap();
        assert!(initial.orientations.is_some());
        let link = world
            .link_names
            .iter()
            .position(|name| name == "ball")
            .unwrap() as u32;
        let mut pose = world.forward_kinematics(initial.clone()).unwrap()[link as usize];
        pose.orientation =
            isometry_link_pose(Isometry3::rotation(Vector3::new(0.2, -0.3, 0.4))).orientation;
        let target = IkTarget {
            link,
            pose,
            local_point: v(0.0, 0.0, 0.0),
            constrained_axes: vec![false, false, false, true, true, true],
        };
        let result = world
            .inverse_kinematics(initial.clone(), target, default_ik_config())
            .unwrap();
        assert!(result.converged && result.iterations > 0);
        assert_eq!(result.state.positions, initial.positions);
        let solved = world.forward_kinematics(result.state).unwrap()[link as usize];
        assert!(
            link_pose_isometry(solved)
                .unwrap()
                .rotation
                .angle_to(&link_pose_isometry(pose).unwrap().rotation,)
                < 1e-3
        );
        let after = world.kinematics_state().unwrap();
        let before_q = initial
            .orientations
            .unwrap()
            .into_iter()
            .flatten()
            .next()
            .unwrap();
        let after_q = after
            .orientations
            .unwrap()
            .into_iter()
            .flatten()
            .next()
            .unwrap();
        assert_eq!(
            [before_q.x, before_q.y, before_q.z, before_q.w],
            [after_q.x, after_q.y, after_q.z, after_q.w]
        );
        assert_eq!(world.velocities().unwrap(), [0.0; 3]);
    }

    #[test]
    fn cpu_sphere_and_urdf_cross_ffi_boundary() {
        let world = SphereWorld::new(
            vec![SphereInput {
                center: v(0.0, 0.0, 0.4),
                velocity: v(0.0, 0.0, -1.0),
                radius: 0.5,
                mass: 1.0,
            }],
            v(0.0, 0.0, 0.0),
            10.0,
            1.0,
            0.0,
            0.001,
            12,
        )
        .unwrap();
        world.step(0.001).unwrap();
        assert!(world.contact_force(0).unwrap().z > 0.0);

        let xml = r#"<robot name="arm"><link name="base"/><link name="tip">
            <inertial><mass value="1"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial>
            </link><joint name="slide" type="prismatic"><parent link="base"/><child link="tip"/>
            <axis xyz="1 0 0"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint></robot>"#;
        let robot = ArticulatedWorld::from_urdf(xml.into(), false).unwrap();
        assert_eq!(robot.joint_armature(0).unwrap(), [0.0]);
        robot.set_joint_armature(0, vec![0.6]).unwrap();
        assert_eq!(robot.joint_armature(0).unwrap(), [0.6]);
        assert!(robot.set_joint_armature(0, vec![-1.0]).is_err());
        assert!(robot.joint_armature(1).is_err());
        assert_eq!(robot.joint_ranges()[0].start, 0);
        assert!(robot.self_contacts_enabled().unwrap());
        robot.set_self_contacts_enabled(false).unwrap();
        assert!(!robot.self_contacts_enabled().unwrap());
        robot.set_positions(vec![0.2]).unwrap();
        robot.step(0.001, vec![0.0]).unwrap();
        assert_eq!(robot.link_poses().unwrap().len(), 2);
        let wrench = robot.link_contact_wrench(1).unwrap();
        assert_eq!(wrench.force.z, 0.0);
        assert_eq!(wrench.torque.y, 0.0);
        assert!(robot.link_contact_wrench(2).is_err());
    }

    #[test]
    fn articulated_link_twists_cross_single_and_batch_python_boundaries() {
        let xml = r#"<robot name="slider"><link name="base"/><link name="tip">
          <inertial><mass value="1"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial>
          </link><joint name="slide" type="prismatic"><parent link="base"/><child link="tip"/>
          <axis xyz="1 0 0"/><limit lower="-2" upper="2" effort="10" velocity="10"/></joint></robot>"#;
        let world = ArticulatedWorld::from_urdf(xml.into(), false).unwrap();
        locked(&world.inner).unwrap().world.velocities[0] = 2.0;
        let twists = world.link_twists().unwrap();
        assert_eq!(twists.len(), 2);
        assert_eq!(twists[0].linear.x, 0.0);
        assert_eq!(twists[1].linear.x, 2.0);
        assert_eq!(twists[1].angular.z, 0.0);
        assert_eq!(
            world
                .link_point_twist(1, v(0.0, 1.0, 0.0))
                .unwrap()
                .linear
                .x,
            2.0
        );
        assert!(world.link_point_twist(2, v(0.0, 0.0, 0.0)).is_err());
        let acceleration = world
            .link_point_acceleration(1, v(0.0, 1.0, 0.0), vec![3.0])
            .unwrap();
        assert!((acceleration.linear.x - 3.0).abs() < 1e-10);
        let imu = world.link_imu(1, v(0.0, 1.0, 0.0), vec![3.0]).unwrap();
        assert!((imu.specific_force.x - 3.0).abs() < 1e-10);
        assert!((imu.specific_force.z - 9.81).abs() < 1e-10);
        assert!(world.link_imu(1, v(0.0, 0.0, 0.0), vec![]).is_err());

        let batch = ArticulatedBatch::new(vec![BatchModel {
            format: ModelFormat::Urdf,
            xml: xml.into(),
            floating_base: false,
            meshes: Vec::new(),
        }])
        .unwrap();
        {
            let mut inner = locked(&batch.inner).unwrap();
            let id = batch.id(0).unwrap();
            inner.batch.environment_mut(id).unwrap().velocities[0] = -3.0;
        }
        assert_eq!(batch.link_twists(0).unwrap()[1].linear.x, -3.0);
        assert_eq!(
            batch
                .link_point_twist(0, 1, v(0.0, 1.0, 0.0))
                .unwrap()
                .linear
                .x,
            -3.0
        );
        assert!(batch.link_twists(1).is_err());
        assert!(
            (batch
                .link_point_acceleration(0, 1, v(0.0, 0.0, 0.0), vec![-4.0])
                .unwrap()
                .linear
                .x
                + 4.0)
                .abs()
                < 1e-10
        );
        assert!(
            (batch
                .link_imu(0, 1, v(0.0, 0.0, 0.0), vec![-4.0])
                .unwrap()
                .specific_force
                .x
                + 4.0)
                .abs()
                < 1e-10
        );
    }

    fn tetrahedron() -> ConvexMeshPart {
        let diagonal = (1.0 / 3.0f64).sqrt();
        ConvexMeshPart {
            vertices: vec![
                v(0.0, 0.0, 0.0),
                v(1.0, 0.0, 0.0),
                v(0.0, 1.0, 0.0),
                v(0.0, 0.0, 1.0),
            ],
            face_normals: vec![
                v(-1.0, 0.0, 0.0),
                v(0.0, -1.0, 0.0),
                v(0.0, 0.0, -1.0),
                v(diagonal, diagonal, diagonal),
            ],
            edge_directions: vec![v(1.0, 0.0, 0.0), v(0.0, 1.0, 0.0), v(0.0, 0.0, 1.0)],
        }
    }

    fn material() -> Material {
        Material {
            friction: 0.7,
            restitution: 0.3,
            friction_rule: CombineRule::Max,
            restitution_rule: CombineRule::Min,
        }
    }

    #[test]
    fn mesh_resolver_scales_urdf_and_mjcf_and_rejects_invalid_assets() {
        let asset = MeshAsset {
            uri: "mesh.obj".into(),
            parts: vec![tetrahedron()],
        };
        let urdf = r#"<robot name="mesh"><link name="base">
          <inertial><mass value="1"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
          <collision><geometry><mesh filename="mesh.obj" scale="2 3 4"/></geometry></collision>
        </link></robot>"#;
        let world =
            ArticulatedWorld::from_urdf_with_meshes(urdf.into(), false, vec![asset.clone()])
                .unwrap();
        let inner = locked(&world.inner).unwrap();
        let geometry = &inner.world.convex_shapes[0].geometry;
        assert!(geometry.vertices.contains(&Vector3::new(2.0, 0.0, 0.0)));
        assert!(geometry.vertices.contains(&Vector3::new(0.0, 3.0, 0.0)));
        assert!(geometry.vertices.contains(&Vector3::new(0.0, 0.0, 4.0)));
        assert!((geometry.face_normals[3].norm() - 1.0).abs() < 1e-12);
        drop(inner);
        world
            .set_link_material(LinkColliderKind::Convex, 0, material())
            .unwrap();
        assert_eq!(
            world
                .link_material(LinkColliderKind::Convex, 0)
                .unwrap()
                .friction,
            0.7
        );

        let mjcf = r#"<mujoco><asset><mesh name="hull" file="mesh.obj" scale="2 3 4"/></asset>
          <worldbody><body name="root"><inertial mass="1" diaginertia="1 1 1"/>
          <geom type="mesh" mesh="hull"/></body></worldbody></mujoco>"#;
        let world =
            ArticulatedWorld::from_mjcf_with_meshes(mjcf.into(), vec![asset.clone()]).unwrap();
        assert_eq!(locked(&world.inner).unwrap().world.convex_shapes.len(), 1);
        let batch = ArticulatedBatch::new(vec![BatchModel {
            format: ModelFormat::Mjcf,
            xml: mjcf.into(),
            floating_base: false,
            meshes: vec![asset.clone()],
        }])
        .unwrap();
        assert_eq!(batch.environment_info(0).unwrap().link_names, ["root"]);
        assert!(batch.link_material(0, LinkColliderKind::Convex, 0).is_ok());
        assert_eq!(batch.link_contact_wrench(0, 0).unwrap().torque.y, 0.0);
        assert!(batch.link_contact_wrench(0, 1).is_err());
        assert!(ArticulatedWorld::from_mjcf_with_meshes(mjcf.into(), Vec::new()).is_err());
        assert!(ArticulatedWorld::from_mjcf_with_meshes(
            "<mujoco/>".into(),
            vec![asset.clone(), asset],
        )
        .is_err());
    }

    #[test]
    fn material_and_motor_round_trip_and_validate() {
        let ball = SphereInput {
            center: v(0.0, 0.0, 1.0),
            velocity: v(0.0, 0.0, 0.0),
            radius: 0.5,
            mass: 1.0,
        };
        let world =
            SphereWorld::new(vec![ball], v(0.0, 0.0, -9.81), 10.0, 0.5, 0.0, 0.001, 8).unwrap();
        world.set_body_material(0, material()).unwrap();
        world.set_ground_material(material()).unwrap();
        assert_eq!(world.body_material(0).unwrap().friction, 0.7);
        assert_eq!(world.ground_material().unwrap().restitution, 0.3);
        assert!(world.set_body_material(1, material()).is_err());
        let mut invalid = material();
        invalid.friction = -1.0;
        assert!(world.set_ground_material(invalid).is_err());

        let xml = r#"<robot name="arm"><link name="base"/><link name="tip">
          <inertial><mass value="1"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial>
          </link><joint name="slide" type="prismatic"><parent link="base"/><child link="tip"/>
          <axis xyz="1 0 0"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint></robot>"#;
        let robot = ArticulatedWorld::from_urdf(xml.into(), false).unwrap();
        let motor = JointMotor {
            position_target: Some(0.4),
            velocity_target: 0.0,
            stiffness: 10.0,
            damping: 1.0,
            max_force: 5.0,
        };
        robot.set_joint_motor(0, Some(motor)).unwrap();
        assert_eq!(
            robot.joint_motor(0).unwrap().unwrap().position_target,
            Some(0.4)
        );
        robot.step(0.001, vec![0.0]).unwrap();
        assert!(robot.velocities().unwrap()[0] > 0.0);
        assert!(robot.joint_motor(1).is_err());
        robot.set_joint_motor(0, None).unwrap();
        assert!(robot.joint_motor(0).unwrap().is_none());
        let passive = JointPassive {
            stiffness: 3.0,
            damping: 0.5,
            rest_position: 0.2,
        };
        robot.set_joint_passive(0, passive).unwrap();
        assert_eq!(robot.joint_passive(0).unwrap().stiffness, 3.0);
        let nonlinear = JointNonlinearPassive {
            spring_quadratic: 1.0,
            spring_cubic: 2.0,
            damping_quadratic: 3.0,
            damping_cubic: 4.0,
        };
        robot.set_joint_nonlinear_passive(0, nonlinear).unwrap();
        assert_eq!(robot.joint_nonlinear_passive(0).unwrap().spring_cubic, 2.0);
        assert!(robot.joint_nonlinear_passive(1).is_err());
        assert!(
            robot
                .set_joint_nonlinear_passive(
                    0,
                    JointNonlinearPassive {
                        spring_cubic: f64::NAN,
                        ..nonlinear
                    },
                )
                .is_err()
        );
        robot.set_joint_friction(0, 0.6).unwrap();
        assert_eq!(robot.joint_friction(0).unwrap(), 0.6);
        assert!(robot.set_joint_friction(0, -1.0).is_err());
        assert!(robot.joint_friction(1).is_err());
        assert!(robot.joint_passive(1).is_err());
        assert!(
            robot
                .set_joint_passive(
                    0,
                    JointPassive {
                        damping: -1.0,
                        ..passive
                    }
                )
                .is_err()
        );
    }

    #[test]
    fn articulated_device_mass_bindings_step_world_and_batch() {
        if GpuContactDevice::new().is_err() {
            return;
        }
        let xml = r#"<robot name="arm"><link name="base"/><link name="tip">
          <inertial><mass value="1"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial>
          </link><joint name="slide" type="prismatic"><parent link="base"/><child link="tip"/>
          <axis xyz="1 0 0"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint></robot>"#;
        let world = ArticulatedWorld::from_urdf(xml.into(), false).unwrap();
        let batch = ArticulatedBatch::new(vec![
            BatchModel {
                format: ModelFormat::Urdf,
                xml: xml.into(),
                floating_base: false,
                meshes: Vec::new(),
            },
            BatchModel {
                format: ModelFormat::Urdf,
                xml: xml.into(),
                floating_base: false,
                meshes: Vec::new(),
            },
        ])
        .unwrap();
        world.step_gpu_device_mass(0.001, vec![0.2]).unwrap();
        batch
            .step_gpu_device_mass(0.001, vec![vec![0.2], vec![-0.2]])
            .unwrap();
        assert!((world.positions().unwrap()[0] - batch.positions(0).unwrap()[0]).abs() < 1e-6);
        assert!((world.velocities().unwrap()[0] - batch.velocities(0).unwrap()[0]).abs() < 1e-6);
        assert!(batch.velocities(1).unwrap()[0] < 0.0);
        world.step_gpu_device_mass(0.001, vec![0.2]).unwrap();
        batch
            .step_gpu_device_mass(0.001, vec![vec![0.2], vec![-0.2]])
            .unwrap();
        assert!((world.positions().unwrap()[0] - batch.positions(0).unwrap()[0]).abs() < 1e-6);
        assert!((world.velocities().unwrap()[0] - batch.velocities(0).unwrap()[0]).abs() < 1e-6);
        assert!(world.implicit_coriolis().unwrap());
        world.set_implicit_coriolis(false).unwrap();
        assert!(!world.implicit_coriolis().unwrap());
        assert!(batch.implicit_coriolis(0).unwrap());
        batch.set_implicit_coriolis(0, false).unwrap();
        batch.publish_reset_template(0).unwrap();
        batch.set_implicit_coriolis(0, true).unwrap();
        batch.reset_environment(0).unwrap();
        assert!(!batch.implicit_coriolis(0).unwrap());
        assert!(batch.implicit_coriolis(1).unwrap());
        assert!(batch.set_implicit_coriolis(2, false).is_err());
    }

    #[test]
    fn scene_body_constraints_round_trip_through_world_and_batch() {
        let xml = r#"<mujoco><option gravity="0 0 0"/><worldbody>
          <body name="root"><inertial mass="1" diaginertia="1 1 1"/>
            <body name="x"><joint name="x" type="slide" axis="1 0 0"/>
              <inertial mass="1" diaginertia="1 1 1"/></body>
          </body></worldbody></mujoco>"#;
        let world = ArticulatedWorld::from_mjcf(xml.into()).unwrap();
        let model = BatchModel {
            format: ModelFormat::Mjcf,
            xml: xml.into(),
            floating_base: false,
            meshes: Vec::new(),
        };
        let batch = ArticulatedBatch::new(vec![model.clone(), model]).unwrap();
        let identity_frame = LinkPose {
            position: Vec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            orientation: Quaternion {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 1.0,
            },
        };
        let first_scene = world.add_scene_sphere(identity_frame, 0.1, 1.0).unwrap();
        let second_scene = world.add_scene_sphere(identity_frame, 0.1, 1.0).unwrap();
        let prescribed = world
            .add_scene_sphere(
                LinkPose {
                    position: Vec3 {
                        x: 20.0,
                        y: 0.0,
                        z: 10.0,
                    },
                    ..identity_frame
                },
                0.1,
                0.0,
            )
            .unwrap();
        let motion = Vec3 {
            x: 0.5,
            y: 0.0,
            z: 0.0,
        };
        let zero = Vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        assert!(
            world
                .set_scene_body_kinematic_motion(prescribed, Some(motion), None)
                .is_err()
        );
        assert!(
            world
                .set_scene_body_kinematic_motion(first_scene, Some(motion), Some(zero))
                .is_err()
        );
        world
            .set_scene_body_kinematic_motion(prescribed, Some(motion), Some(zero))
            .unwrap();
        world.step(0.01, vec![0.0]).unwrap();
        assert!(
            (world.scene_body_state(prescribed).unwrap().pose.position.x - 20.005).abs() < 1.0e-9
        );
        world
            .set_scene_body_kinematic_motion(prescribed, None, None)
            .unwrap();
        world.step(0.01, vec![0.0]).unwrap();
        assert!(
            (world.scene_body_state(prescribed).unwrap().pose.position.x - 20.005).abs() < 1.0e-9
        );
        assert_eq!(world.scene_body_state(second_scene).unwrap().mass, 1.0);
        assert!(world.add_scene_sphere(identity_frame, -1.0, 1.0).is_err());
        let scene_point = LinkScenePointConstraint {
            link: 1,
            link_point: identity_frame.position,
            body: second_scene,
            body_point: identity_frame.position,
        };
        let scene_fixed = LinkSceneFixedConstraint {
            link: 1,
            link_frame: identity_frame,
            body: second_scene,
            body_frame: identity_frame,
        };
        world
            .set_link_scene_point_constraints(vec![scene_point])
            .unwrap();
        world
            .set_link_scene_fixed_constraints(vec![scene_fixed])
            .unwrap();
        assert_eq!(
            world.link_scene_point_constraints().unwrap()[0].body,
            second_scene
        );
        assert_eq!(
            world.link_scene_fixed_constraints().unwrap()[0].body,
            second_scene
        );
        world
            .set_scene_body_force(
                second_scene,
                Vec3 {
                    x: 1.0,
                    y: 0.0,
                    z: 0.0,
                },
            )
            .unwrap();
        assert_eq!(world.scene_body_state(second_scene).unwrap().force.x, 1.0);
        assert!(world.remove_scene_body(first_scene).unwrap());
        assert_eq!(world.link_scene_point_constraints().unwrap()[0].body, 0);
        assert_eq!(world.link_scene_fixed_constraints().unwrap()[0].body, 0);
        assert!(world.remove_scene_body(0).unwrap());
        assert!(world.link_scene_point_constraints().unwrap().is_empty());
        assert!(world.link_scene_fixed_constraints().unwrap().is_empty());

        let batch_scene = batch.add_scene_sphere(0, identity_frame, 0.1, 1.0).unwrap();
        assert_eq!(batch_scene, 0);
        batch
            .set_link_scene_point_constraints(
                0,
                vec![LinkScenePointConstraint {
                    body: batch_scene,
                    ..scene_point
                }],
            )
            .unwrap();
        batch
            .set_link_scene_fixed_constraints(
                0,
                vec![LinkSceneFixedConstraint {
                    body: batch_scene,
                    ..scene_fixed
                }],
            )
            .unwrap();
        assert_eq!(batch.scene_body_state(0, batch_scene).unwrap().mass, 1.0);
        assert!(batch.scene_body_state(1, batch_scene).is_err());
        batch.publish_reset_template(0).unwrap();
        assert!(batch.remove_scene_body(0, batch_scene).unwrap());
        assert!(batch.link_scene_point_constraints(0).unwrap().is_empty());
        batch.reset_environment(0).unwrap();
        assert_eq!(batch.link_scene_point_constraints(0).unwrap()[0].body, 0);
        assert_eq!(batch.link_scene_fixed_constraints(0).unwrap()[0].body, 0);
        assert!(
            batch
                .set_link_scene_point_constraints(1, vec![scene_point])
                .is_err()
        );
        assert!(
            batch
                .set_link_scene_fixed_constraints(1, vec![scene_fixed])
                .is_err()
        );
    }

    #[test]
    fn scene_body_input_supports_all_scene_collider_shapes() {
        let xml = r#"<mujoco><option gravity="0 0 0"/><worldbody>
          <body name="root"><inertial mass="1" diaginertia="1 1 1"/>
            <body name="x"><joint name="x" type="slide" axis="1 0 0"/>
              <inertial mass="1" diaginertia="1 1 1"/></body>
          </body></worldbody></mujoco>"#;
        let world = ArticulatedWorld::from_mjcf(xml.into()).unwrap();
        let identity = LinkPose {
            position: v(0.0, 0.0, 0.0),
            orientation: Quaternion {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 1.0,
            },
        };
        let normal = 1.0 / 3.0_f64.sqrt();
        let shapes = vec![
            SceneShape::Sphere { radius: 0.2 },
            SceneShape::Box {
                half_extents: v(0.2, 0.3, 0.4),
            },
            SceneShape::Capsule {
                half_height: 0.2,
                radius: 0.1,
            },
            SceneShape::Cylinder {
                half_height: 0.2,
                radius: 0.1,
            },
            SceneShape::Cone {
                half_height: 0.2,
                radius: 0.1,
            },
            SceneShape::Convex {
                geometry: ConvexMeshPart {
                    vertices: vec![
                        v(0.0, 0.0, 0.0),
                        v(1.0, 0.0, 0.0),
                        v(0.0, 1.0, 0.0),
                        v(0.0, 0.0, 1.0),
                    ],
                    face_normals: vec![
                        v(-1.0, 0.0, 0.0),
                        v(0.0, -1.0, 0.0),
                        v(0.0, 0.0, -1.0),
                        v(normal, normal, normal),
                    ],
                    edge_directions: vec![v(1.0, 0.0, 0.0), v(0.0, 1.0, 0.0), v(0.0, 0.0, 1.0)],
                },
            },
            SceneShape::TriangleMesh {
                vertices: vec![v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0), v(0.0, 1.0, 0.0)],
                triangles: vec![GpuTriangle { a: 0, b: 1, c: 2 }],
            },
            SceneShape::HeightField {
                rows: 2,
                columns: 2,
                heights: vec![0.0; 4],
                scale: v(1.0, 1.0, 1.0),
            },
            SceneShape::Polyline {
                vertices: vec![v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0)],
                segments: vec![GpuSegment { a: 0, b: 1 }],
            },
        ];
        let base = SceneBodyInput {
            pose: identity,
            linear_velocity: v(0.0, 0.0, 0.0),
            angular_velocity: v(0.0, 0.0, 0.0),
            mass: 1.0,
            inertia_tensor: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            force: v(0.0, 0.0, 0.0),
            colliders: Vec::new(),
        };
        for (index, shape) in shapes.into_iter().enumerate() {
            let input = SceneBodyInput {
                pose: LinkPose {
                    position: v(index as f64 * 5.0, 0.0, 10.0),
                    ..identity
                },
                colliders: vec![SceneColliderInput {
                    frame: identity,
                    shape,
                }],
                ..base.clone()
            };
            assert_eq!(world.add_scene_body(input).unwrap(), index as u32);
        }
        assert_eq!(world.scene_body_state(8).unwrap().mass, 1.0);
        world
            .set_scene_body_pose(
                0,
                LinkPose {
                    position: v(0.0, 0.0, 12.0),
                    ..identity
                },
            )
            .unwrap();
        world
            .set_scene_body_velocity(0, v(0.1, 0.0, 0.0), v(0.0, 0.0, 0.2))
            .unwrap();
        assert_eq!(world.scene_body_state(0).unwrap().pose.position.z, 12.0);
        assert_eq!(world.scene_body_state(0).unwrap().linear_velocity.x, 0.1);
        assert_eq!(world.scene_body_state(0).unwrap().angular_velocity.z, 0.2);
        assert!(
            world
                .set_scene_body_velocity(9, v(0.0, 0.0, 0.0), v(0.0, 0.0, 0.0))
                .is_err()
        );
        world.step(0.001, vec![0.0]).unwrap();
        assert!(
            world
                .add_scene_body(SceneBodyInput {
                    colliders: vec![SceneColliderInput {
                        frame: identity,
                        shape: SceneShape::Sphere { radius: 0.0 }
                    }],
                    ..base.clone()
                })
                .is_err()
        );
        assert!(
            world
                .add_scene_body(SceneBodyInput {
                    linear_velocity: v(f64::NAN, 0.0, 0.0),
                    colliders: vec![SceneColliderInput {
                        frame: identity,
                        shape: SceneShape::Sphere { radius: 0.2 }
                    }],
                    ..base.clone()
                })
                .is_err()
        );
        assert!(
            world
                .add_scene_body(SceneBodyInput {
                    inertia_tensor: vec![1.0; 8],
                    colliders: vec![SceneColliderInput {
                        frame: identity,
                        shape: SceneShape::Sphere { radius: 0.2 },
                    }],
                    ..base.clone()
                })
                .is_err()
        );
        assert!(
            world
                .add_scene_body(SceneBodyInput {
                    inertia_tensor: vec![1.0, 0.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    colliders: vec![SceneColliderInput {
                        frame: identity,
                        shape: SceneShape::Sphere { radius: 0.2 },
                    }],
                    ..base.clone()
                })
                .is_err()
        );
        assert!(world.scene_body_state(9).is_err());

        let batch = ArticulatedBatch::new(vec![BatchModel {
            format: ModelFormat::Mjcf,
            xml: xml.into(),
            floating_base: false,
            meshes: Vec::new(),
        }])
        .unwrap();
        let compound = SceneBodyInput {
            pose: LinkPose {
                position: v(0.0, 0.0, 10.0),
                ..identity
            },
            inertia_tensor: vec![1.0, 0.1, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0, 1.0],
            colliders: vec![
                SceneColliderInput {
                    frame: identity,
                    shape: SceneShape::Sphere { radius: 0.2 },
                },
                SceneColliderInput {
                    frame: LinkPose {
                        position: v(0.5, 0.0, 0.0),
                        ..identity
                    },
                    shape: SceneShape::Box {
                        half_extents: v(0.1, 0.1, 0.1),
                    },
                },
            ],
            ..base
        };
        assert_eq!(batch.add_scene_body(0, compound).unwrap(), 0);
        batch
            .set_scene_body_pose(
                0,
                0,
                LinkPose {
                    position: v(0.0, 0.0, 11.0),
                    ..identity
                },
            )
            .unwrap();
        batch
            .set_scene_body_velocity(0, 0, v(0.0, 0.1, 0.0), v(0.0, 0.0, 0.2))
            .unwrap();
        assert_eq!(batch.scene_body_state(0, 0).unwrap().collider_count, 2);
        assert!((batch.scene_body_state(0, 0).unwrap().inertia_tensor[1] - 0.1).abs() < 1e-12);
        {
            let inner = locked(&batch.inner).unwrap();
            let scene = &inner
                .batch
                .environment(batch.id(0).unwrap())
                .unwrap()
                .scene_bodies[0];
            assert_eq!(scene.colliders.len(), 2);
            assert!((scene.inertia[(0, 1)] - 0.1).abs() < 1e-12);
            assert!(matches!(scene.colliders[1], CoreSceneCollider::Box { .. }));
        }
        batch.publish_reset_template(0).unwrap();
        assert!(batch.remove_scene_body(0, 0).unwrap());
        batch.reset_environment(0).unwrap();
        assert_eq!(batch.scene_body_state(0, 0).unwrap().mass, 1.0);
        assert_eq!(batch.scene_body_state(0, 0).unwrap().pose.position.z, 11.0);
    }

    #[test]
    fn joint_couplings_round_trip_through_world_and_batch() {
        let xml = r#"<mujoco><option gravity="0 0 0"/><worldbody>
          <body name="root"><inertial mass="1" diaginertia="1 1 1"/>
            <body name="x"><joint name="x" type="slide" axis="1 0 0"/>
              <inertial mass="1" diaginertia="1 1 1"/></body>
            <body name="y"><joint name="y" type="slide" axis="0 1 0"/>
              <inertial mass="1" diaginertia="1 1 1"/></body>
          </body></worldbody></mujoco>"#;
        let coupling = JointCoupling {
            source: 0,
            follower: 1,
            multiplier: -2.0,
            offset: 0.0,
        };
        let world = ArticulatedWorld::from_mjcf(xml.into()).unwrap();
        let nonlinear_xml = xml.replace(
            "name=\"x\" type=\"slide\"",
            "name=\"x\" type=\"slide\" stiffness=\"1 2 3\" damping=\"4 5 6\"",
        );
        let nonlinear_world = ArticulatedWorld::from_mjcf(nonlinear_xml).unwrap();
        assert_eq!(
            nonlinear_world
                .joint_nonlinear_passive(0)
                .unwrap()
                .spring_cubic,
            3.0
        );
        let reference_xml = xml
            .replace(
                "name=\"x\" type=\"slide\"",
                "name=\"x\" type=\"slide\" ref=\"0.3\"",
            )
            .replace(
                "name=\"y\" type=\"slide\"",
                "name=\"y\" type=\"slide\" ref=\"0.5\"",
            );
        let reference_world = ArticulatedWorld::from_mjcf(reference_xml).unwrap();
        assert_eq!(reference_world.positions().unwrap(), vec![0.3, 0.5]);
        assert!(world.joint_couplings().unwrap().is_empty());
        world.set_joint_couplings(vec![coupling]).unwrap();
        assert_eq!(world.joint_couplings().unwrap()[0].multiplier, -2.0);
        world.step(0.001, vec![1.0, 0.0]).unwrap();
        let velocity = world.velocities().unwrap();
        assert!((velocity[1] + 2.0 * velocity[0]).abs() < 1e-8);
        let model = BatchModel {
            format: ModelFormat::Mjcf,
            xml: xml.into(),
            floating_base: false,
            meshes: Vec::new(),
        };
        let batch = ArticulatedBatch::new(vec![model.clone(), model]).unwrap();
        batch.set_joint_couplings(0, vec![coupling]).unwrap();
        assert_eq!(batch.joint_couplings(0).unwrap()[0].follower, 1);
        assert!(batch.joint_couplings(1).unwrap().is_empty());
        assert!(
            batch
                .set_joint_couplings(
                    1,
                    vec![JointCoupling {
                        follower: 0,
                        ..coupling
                    }]
                )
                .is_err()
        );
        let polynomial = JointPolynomialCoupling {
            follower: 1,
            source: Some(0),
            coefficients: vec![0.0, 2.0, 1.0, 0.0, 0.5],
            follower_reference: 0.0,
            source_reference: 0.0,
        };
        batch
            .set_joint_polynomial_couplings(0, vec![polynomial.clone()])
            .unwrap();
        assert_eq!(
            batch.joint_polynomial_couplings(0).unwrap()[0].coefficients,
            polynomial.coefficients
        );
        assert!(batch.joint_polynomial_couplings(1).unwrap().is_empty());
        assert!(
            batch
                .set_joint_polynomial_couplings(
                    1,
                    vec![JointPolynomialCoupling {
                        coefficients: vec![1.0, 2.0],
                        ..polynomial.clone()
                    }]
                )
                .is_err()
        );
        let xml_equality = xml.replace("</mujoco>",
            "<equality><joint joint1=\"y\" joint2=\"x\" polycoef=\"0 2 1 0 0.5\"/></equality></mujoco>");
        let loaded = ArticulatedWorld::from_mjcf(xml_equality).unwrap();
        assert_eq!(
            loaded.joint_polynomial_couplings().unwrap()[0].coefficients,
            polynomial.coefficients
        );
        let point_joint = LinkPointConstraint {
            link_a: 1,
            point_a: Vec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            link_b: Some(2),
            point_b: Vec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
        };
        world.set_link_point_constraints(vec![point_joint]).unwrap();
        assert_eq!(world.link_point_constraints().unwrap()[0].link_b, Some(2));
        batch
            .set_link_point_constraints(0, vec![point_joint])
            .unwrap();
        assert_eq!(batch.link_point_constraints(0).unwrap()[0].link_a, 1);
        assert!(batch.link_point_constraints(1).unwrap().is_empty());
        batch.publish_reset_template(0).unwrap();
        batch.set_link_point_constraints(0, Vec::new()).unwrap();
        batch.reset_environment(0).unwrap();
        assert_eq!(batch.link_point_constraints(0).unwrap()[0].link_b, Some(2));
        assert!(
            batch
                .set_link_point_constraints(
                    1,
                    vec![LinkPointConstraint {
                        link_a: 99,
                        ..point_joint
                    }],
                )
                .is_err()
        );
        let identity_frame = LinkPose {
            position: Vec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            orientation: Quaternion {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 1.0,
            },
        };
        let fixed_joint = LinkFixedConstraint {
            link_a: 1,
            frame_a: identity_frame,
            link_b: Some(2),
            frame_b: identity_frame,
        };
        world.set_link_fixed_constraints(vec![fixed_joint]).unwrap();
        assert_eq!(world.link_fixed_constraints().unwrap()[0].link_b, Some(2));
        batch
            .set_link_fixed_constraints(0, vec![fixed_joint])
            .unwrap();
        assert_eq!(batch.link_fixed_constraints(0).unwrap()[0].link_a, 1);
        assert!(batch.link_fixed_constraints(1).unwrap().is_empty());
        batch.publish_reset_template(0).unwrap();
        batch.set_link_fixed_constraints(0, Vec::new()).unwrap();
        batch.reset_environment(0).unwrap();
        assert_eq!(batch.link_fixed_constraints(0).unwrap()[0].link_b, Some(2));
        assert!(
            world
                .set_link_fixed_constraints(vec![LinkFixedConstraint {
                    frame_a: LinkPose {
                        orientation: Quaternion {
                            w: 0.0,
                            ..identity_frame.orientation
                        },
                        ..identity_frame
                    },
                    ..fixed_joint
                }])
                .is_err()
        );
        assert!(
            batch
                .set_link_fixed_constraints(
                    1,
                    vec![LinkFixedConstraint {
                        link_a: 99,
                        ..fixed_joint
                    }]
                )
                .is_err()
        );
    }

    #[test]
    fn articulated_batch_isolates_environments_and_restores_published_state() {
        let urdf = r#"<robot name="arm"><link name="base"/><link name="tip">
          <inertial><mass value="1"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial>
          <collision><geometry><sphere radius="0.1"/></geometry></collision>
          </link><joint name="slide" type="prismatic"><parent link="base"/><child link="tip"/>
          <axis xyz="1 0 0"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint></robot>"#;
        let model = BatchModel {
            format: ModelFormat::Urdf,
            xml: urdf.into(),
            floating_base: false,
            meshes: Vec::new(),
        };
        let batch = ArticulatedBatch::new(vec![model.clone(), model]).unwrap();
        assert_eq!(batch.len(), 2);
        let passive = JointPassive {
            stiffness: 4.0,
            damping: 1.0,
            rest_position: 0.1,
        };
        batch.set_joint_passive(0, 0, passive).unwrap();
        assert_eq!(batch.joint_passive(0, 0).unwrap().damping, 1.0);
        assert_eq!(batch.joint_passive(1, 0).unwrap().damping, 0.0);
        let nonlinear = JointNonlinearPassive {
            spring_quadratic: 1.0,
            spring_cubic: 2.0,
            damping_quadratic: 3.0,
            damping_cubic: 4.0,
        };
        batch.set_joint_nonlinear_passive(0, 0, nonlinear).unwrap();
        assert_eq!(
            batch.joint_nonlinear_passive(0, 0).unwrap().damping_cubic,
            4.0
        );
        assert_eq!(
            batch.joint_nonlinear_passive(1, 0).unwrap().damping_cubic,
            0.0
        );
        batch.set_joint_friction(0, 0, 0.4).unwrap();
        assert_eq!(batch.joint_friction(0, 0).unwrap(), 0.4);
        assert_eq!(batch.joint_friction(1, 0).unwrap(), 0.0);
        assert!(batch.set_joint_friction(1, 0, f64::NAN).is_err());
        batch.set_joint_armature(0, 0, vec![0.4]).unwrap();
        assert_eq!(batch.joint_armature(0, 0).unwrap(), [0.4]);
        assert_eq!(batch.joint_armature(1, 0).unwrap(), [0.0]);
        assert!(batch.set_joint_armature(1, 0, vec![-0.1]).is_err());
        batch.set_self_contacts_enabled(0, false).unwrap();
        assert!(!batch.self_contacts_enabled(0).unwrap());
        assert!(batch.self_contacts_enabled(1).unwrap());
        assert!(batch.set_self_contacts_enabled(2, false).is_err());
        assert_eq!(
            batch.environment_info(1).unwrap().joint_ranges[0].name,
            "slide"
        );
        batch.set_positions(0, vec![0.2]).unwrap();
        batch.publish_reset_template(0).unwrap();
        batch.set_self_contacts_enabled(0, true).unwrap();
        batch.set_positions(0, vec![0.5]).unwrap();
        batch.set_positions(1, vec![-0.4]).unwrap();
        batch.set_ground_material(0, material()).unwrap();
        batch
            .set_link_material(1, LinkColliderKind::Sphere, 0, material())
            .unwrap();
        assert_eq!(batch.ground_material(0).unwrap().friction, 0.7);
        assert_eq!(
            batch
                .link_material(1, LinkColliderKind::Sphere, 0)
                .unwrap()
                .restitution,
            0.3
        );
        batch.reset_environment(0).unwrap();
        assert_eq!(batch.joint_passive(0, 0).unwrap().stiffness, 4.0);
        assert_eq!(batch.joint_armature(0, 0).unwrap(), [0.4]);
        assert!(!batch.self_contacts_enabled(0).unwrap());
        assert_eq!(batch.positions(0).unwrap(), [0.2]);
        assert_eq!(batch.positions(1).unwrap(), [-0.4]);
        assert_eq!(batch.link_poses(1).unwrap().len(), 2);
        assert!(batch.step(0.001, vec![vec![0.0]]).is_err());
        batch.step(0.001, vec![vec![0.0], vec![0.0]]).unwrap();
        assert!(batch.reset_environment(2).is_err());
    }

    #[test]
    fn gpu_primitive_input_validates_shape_mass_and_inertia() {
        let input = GpuPrimitiveBody {
            shape: GpuPrimitiveShape::Capsule {
                radius: 0.5,
                half_length: 1.0,
            },
            center: v(0.0, 0.0, 2.0),
            orientation: Quaternion {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 1.0,
            },
            velocity: v(0.0, 0.0, 0.0),
            angular_velocity: v(0.0, 0.0, 0.0),
            mass: 1.0,
            principal_inertia: v(0.2, 0.2, 0.1),
        };
        let (_, shape) = gpu_primitive(input.clone()).unwrap();
        assert_eq!(shape.bounding_radius(), Some(1.5));
        for primitive in [
            GpuPrimitiveShape::Cylinder {
                radius: 3.0,
                half_length: 4.0,
            },
            GpuPrimitiveShape::Cone {
                radius: 3.0,
                half_length: 4.0,
            },
        ] {
            let mut analytic = input.clone();
            analytic.shape = primitive;
            let (_, shape) = gpu_primitive(analytic).unwrap();
            assert_eq!(shape.bounding_radius(), Some(5.0));
        }
        let mut convex = input.clone();
        convex.shape = GpuPrimitiveShape::Convex {
            vertices: vec![
                v(1.0, 0.0, 0.0),
                v(0.0, 1.0, 0.0),
                v(0.0, 0.0, 1.0),
                v(-1.0, -1.0, -1.0),
            ],
        };
        let (_, hull) = gpu_primitive(convex.clone()).unwrap();
        assert!((hull.bounding_radius().unwrap() - 3.0_f32.sqrt()).abs() < 1e-6);
        if let GpuPrimitiveShape::Convex { vertices } = &mut convex.shape {
            vertices[3].z = f64::NAN;
        }
        assert!(gpu_primitive(convex).is_err());
        let mut mesh = input.clone();
        mesh.shape = GpuPrimitiveShape::TriangleMesh {
            vertices: vec![v(-1.0, -1.0, 0.0), v(1.0, -1.0, 0.0), v(0.0, 1.0, 0.0)],
            triangles: vec![GpuTriangle { a: 0, b: 1, c: 2 }],
        };
        assert!(matches!(
            gpu_primitive(mesh.clone()).unwrap().1,
            GpuRigidShape::TriangleMesh { .. }
        ));
        if let GpuPrimitiveShape::TriangleMesh { triangles, .. } = &mut mesh.shape {
            triangles[0].c = 3;
        }
        assert!(gpu_primitive(mesh).is_err());
        let mut line = input.clone();
        line.shape = GpuPrimitiveShape::Polyline {
            vertices: vec![v(-1.0, 0.0, 0.0), v(1.0, 0.0, 0.0)],
            segments: vec![GpuSegment { a: 0, b: 1 }],
        };
        assert!(matches!(
            gpu_primitive(line.clone()).unwrap().1,
            GpuRigidShape::Polyline { .. }
        ));
        if let GpuPrimitiveShape::Polyline { segments, .. } = &mut line.shape {
            segments[0].b = 2;
        }
        assert!(gpu_primitive(line).is_err());
        let mut terrain = input.clone();
        terrain.shape = GpuPrimitiveShape::Heightfield {
            rows: 2,
            columns: 2,
            heights: vec![0.0, 0.0, 1.0, 1.0],
            scale: v(2.0, 2.0, 1.0),
        };
        let (_, shape) = gpu_primitive(terrain.clone()).unwrap();
        assert!(matches!(shape, GpuRigidShape::TriangleMesh { .. }));
        for (rows, heights, scale) in [
            (1, vec![0.0; 4], v(2.0, 2.0, 1.0)),
            (2, vec![0.0; 3], v(2.0, 2.0, 1.0)),
            (2, vec![f64::NAN; 4], v(2.0, 2.0, 1.0)),
            (2, vec![0.0; 4], v(0.0, 2.0, 1.0)),
            (2, vec![1e40; 4], v(2.0, 2.0, 1.0)),
        ] {
            terrain.shape = GpuPrimitiveShape::Heightfield {
                rows,
                columns: 2,
                heights,
                scale,
            };
            assert!(gpu_primitive(terrain.clone()).is_err());
        }
        let mut invalid = input.clone();
        invalid.principal_inertia.x = 0.0;
        assert!(gpu_primitive(invalid).is_err());
        let mut invalid = input;
        invalid.orientation.w = 0.5;
        assert!(gpu_primitive(invalid).is_err());
    }

    #[test]
    fn gpu_primitive_world_exposes_mixed_contacts_and_reset() {
        let Ok(_device) = GpuContactDevice::new() else {
            return;
        };
        let orientation = Quaternion {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        };
        let bodies = vec![
            GpuPrimitiveBody {
                shape: GpuPrimitiveShape::Box {
                    half_extents: v(1.0, 1.0, 0.5),
                },
                center: v(0.0, 0.0, 3.0),
                orientation,
                velocity: v(0.0, 0.0, 0.0),
                angular_velocity: v(0.0, 0.0, 0.0),
                mass: 0.0,
                principal_inertia: v(0.0, 0.0, 0.0),
            },
            GpuPrimitiveBody {
                shape: GpuPrimitiveShape::Sphere { radius: 0.5 },
                center: v(1.4, 0.0, 3.0),
                orientation,
                velocity: v(0.0, 0.0, 0.0),
                angular_velocity: v(0.0, 0.0, 0.0),
                mass: 1.0,
                principal_inertia: v(0.1, 0.1, 0.1),
            },
            GpuPrimitiveBody {
                shape: GpuPrimitiveShape::Capsule {
                    radius: 0.25,
                    half_length: 0.5,
                },
                center: v(10.0, 0.0, 3.0),
                orientation,
                velocity: v(0.0, 0.0, 0.0),
                angular_velocity: v(0.0, 0.0, 0.0),
                mass: 0.0,
                principal_inertia: v(0.0, 0.0, 0.0),
            },
        ];
        let world = GpuPrimitiveWorld::new(bodies, v(0.0, 0.0, 0.0), 10.0).unwrap();
        world.set_max_linear_speed(Some(2.0)).unwrap();
        assert_eq!(world.max_linear_speed().unwrap(), Some(2.0));
        assert!(
            world
                .step_temporal_speed_bounded(0.1, 4, None, None)
                .is_err()
        );
        assert_eq!(world.len().unwrap(), 3);
        assert!(matches!(
            world.shape(2).unwrap(),
            GpuPrimitiveShape::Capsule { .. }
        ));
        let _candidates = world.step(0.01).unwrap();
        let contacts = world.readback_contacts().unwrap();
        assert!(contacts.iter().any(|contact| {
            contact.body_a == 0 && contact.body_b == Some(1) && contact.depth > 0.0
        }));
        world.reset().unwrap();
        assert!((world.readback().unwrap()[1].center.x - 1.4).abs() < 1e-5);
        let added = world
            .add_body(GpuPrimitiveBody {
                shape: GpuPrimitiveShape::Sphere { radius: 0.25 },
                center: v(20.0, 0.0, 3.0),
                orientation,
                velocity: v(0.0, 0.0, 0.0),
                angular_velocity: v(0.0, 0.0, 0.0),
                mass: 0.0,
                principal_inertia: v(0.0, 0.0, 0.0),
            })
            .unwrap();
        assert_eq!(added, 3);
        let removed = world.remove_body(0).unwrap();
        assert!(matches!(removed.shape, GpuPrimitiveShape::Box { .. }));
        assert_eq!(world.len().unwrap(), 3);
        assert!(matches!(
            world.shape(1).unwrap(),
            GpuPrimitiveShape::Capsule { .. }
        ));
        world.reset().unwrap();
        assert!((world.readback().unwrap()[2].center.x - 20.0).abs() < 1e-5);
    }

    #[test]
    fn python_sphere_world_uses_speed_bounded_speculative_contacts() {
        let Ok(_device) = GpuContactDevice::new() else {
            return;
        };
        let world = GpuSphereWorld::new(
            vec![
                SphereInput {
                    center: v(0.0, 0.0, 5.0),
                    velocity: v(0.0, 0.0, 0.0),
                    radius: 1.0,
                    mass: 0.0,
                },
                SphereInput {
                    center: v(2.5, 0.0, 5.0),
                    velocity: v(-10.0, 0.0, 0.0),
                    radius: 1.0,
                    mass: 1.0,
                },
            ],
            v(0.0, 0.0, 0.0),
            10.0,
        )
        .unwrap();
        assert!(
            world
                .step_temporal_speed_bounded(0.1, 4, None, None)
                .is_err()
        );
        world.set_max_linear_speed(Some(10.0)).unwrap();
        let count = world
            .step_temporal_speed_bounded(0.1, 4, None, None)
            .unwrap();
        assert_eq!(count, 1);
        assert!(world.readback().unwrap()[1].center.x > 1.99);
    }

    #[test]
    fn gpu_primitive_world_resolves_convex_sphere_contact() {
        let Ok(_device) = GpuContactDevice::new() else {
            return;
        };
        let orientation = Quaternion {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        };
        let make_body = |shape, x, mass, inertia| GpuPrimitiveBody {
            shape,
            center: v(x, 0.0, 2.0),
            orientation,
            velocity: v(0.0, 0.0, 0.0),
            angular_velocity: v(0.0, 0.0, 0.0),
            mass,
            principal_inertia: v(inertia, inertia, inertia),
        };
        let hull = GpuPrimitiveShape::Convex {
            vertices: [-1.0, 1.0]
                .into_iter()
                .flat_map(|x| {
                    [-1.0, 1.0]
                        .into_iter()
                        .flat_map(move |y| [-0.5, 0.5].into_iter().map(move |z| v(x, y, z)))
                })
                .collect(),
        };
        let world = GpuPrimitiveWorld::new(
            vec![
                make_body(hull, 0.0, 0.0, 0.0),
                make_body(GpuPrimitiveShape::Sphere { radius: 0.5 }, 1.4, 1.0, 0.1),
            ],
            v(0.0, 0.0, 0.0),
            10.0,
        )
        .unwrap();
        let _pairs = world.step(0.01).unwrap();
        let contacts = world.readback_contacts().unwrap();
        assert!(contacts.iter().any(|contact| {
            contact.body_a == 0 && contact.body_b == Some(1) && (contact.depth - 0.1).abs() < 1e-3
        }));
    }

    #[test]
    fn gpu_primitive_world_resolves_triangle_mesh_sphere_contact() {
        let Ok(_device) = GpuContactDevice::new() else {
            return;
        };
        let orientation = Quaternion {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        };
        let make_body = |shape, z, mass, inertia| GpuPrimitiveBody {
            shape,
            center: v(0.0, 0.0, z),
            orientation,
            velocity: v(0.0, 0.0, 0.0),
            angular_velocity: v(0.0, 0.0, 0.0),
            mass,
            principal_inertia: v(inertia, inertia, inertia),
        };
        let mesh = GpuPrimitiveShape::TriangleMesh {
            vertices: vec![v(-1.0, -1.0, 0.0), v(1.0, -1.0, 0.0), v(0.0, 1.0, 0.0)],
            triangles: vec![GpuTriangle { a: 0, b: 1, c: 2 }],
        };
        let mut capsule = make_body(
            GpuPrimitiveShape::Capsule {
                radius: 0.25,
                half_length: 0.3,
            },
            2.0,
            1.0,
            0.1,
        );
        capsule.center.y = -1.1;
        let world = GpuPrimitiveWorld::new(
            vec![
                make_body(mesh, 2.0, 0.0, 0.0),
                make_body(GpuPrimitiveShape::Sphere { radius: 0.5 }, 2.25, 1.0, 0.1),
                capsule,
            ],
            v(0.0, 0.0, 0.0),
            10.0,
        )
        .unwrap();
        let _pairs = world.step(0.01).unwrap();
        let contacts = world.readback_contacts().unwrap();
        assert!(contacts.iter().any(|contact| {
            contact.body_a == 0 && contact.body_b == Some(1) && (contact.depth - 0.25).abs() < 1e-3
        }));
        assert!(contacts.iter().any(|contact| {
            contact.body_a == 0 && contact.body_b == Some(2) && (contact.depth - 0.15).abs() < 1e-3
        }));
        let states = world.readback().unwrap();
        assert!(states[1].velocity.z > 0.0);
        assert!(states[2].velocity.y < 0.0);
        assert!(matches!(
            world.shape(0).unwrap(),
            GpuPrimitiveShape::TriangleMesh { .. }
        ));
    }

    #[test]
    fn gpu_primitive_world_exposes_box_ground_manifold() {
        let Ok(_device) = GpuContactDevice::new() else {
            return;
        };
        let world = GpuPrimitiveWorld::new(
            vec![GpuPrimitiveBody {
                shape: GpuPrimitiveShape::Box {
                    half_extents: v(0.5, 0.5, 0.5),
                },
                center: v(0.0, 0.0, 0.45),
                orientation: Quaternion {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                    w: 1.0,
                },
                velocity: v(0.0, 0.0, 0.0),
                angular_velocity: v(0.0, 0.0, 0.0),
                mass: 1.0,
                principal_inertia: v(1.0, 1.0, 1.0),
            }],
            v(0.0, 0.0, 0.0),
            2.0,
        )
        .unwrap();
        let _ = world.step(0.01).unwrap();
        let contacts = world.readback_contacts().unwrap();
        assert_eq!(contacts.len(), 4);
        assert!(contacts.iter().all(|contact| {
            contact.body_a == 0 && contact.body_b.is_none() && contact.depth > 0.0
        }));
    }

    #[test]
    fn gpu_primitive_world_exposes_box_pair_manifold() {
        let Ok(_device) = GpuContactDevice::new() else {
            return;
        };
        let box_body = |x| GpuPrimitiveBody {
            shape: GpuPrimitiveShape::Box {
                half_extents: v(0.5, 0.5, 0.5),
            },
            center: v(x, 0.0, 3.0),
            orientation: Quaternion {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 1.0,
            },
            velocity: v(0.0, 0.0, 0.0),
            angular_velocity: v(0.0, 0.0, 0.0),
            mass: 0.0,
            principal_inertia: v(0.0, 0.0, 0.0),
        };
        let world =
            GpuPrimitiveWorld::new(vec![box_body(0.0), box_body(0.9)], v(0.0, 0.0, 0.0), 2.0)
                .unwrap();
        let _ = world.step(0.01).unwrap();
        let contacts = world.readback_contacts().unwrap();
        assert_eq!(contacts.len(), 4);
        assert!(contacts.iter().all(|contact| {
            contact.body_a == 0 && contact.body_b == Some(1) && (contact.depth - 0.1).abs() < 1e-4
        }));
    }

    #[test]
    fn gpu_primitive_world_exposes_convex_pair_manifold() {
        let Ok(_device) = GpuContactDevice::new() else {
            return;
        };
        let shape = GpuPrimitiveShape::Convex {
            vertices: [-0.5, 0.5]
                .into_iter()
                .flat_map(|x| {
                    [-0.5, 0.5]
                        .into_iter()
                        .flat_map(move |y| [-0.5, 0.5].into_iter().map(move |z| v(x, y, z)))
                })
                .collect(),
        };
        let make_body = |z| GpuPrimitiveBody {
            shape: shape.clone(),
            center: v(0.0, 0.0, z),
            orientation: Quaternion {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 1.0,
            },
            velocity: v(0.0, 0.0, 0.0),
            angular_velocity: v(0.0, 0.0, 0.0),
            mass: 0.0,
            principal_inertia: v(0.0, 0.0, 0.0),
        };
        let world =
            GpuPrimitiveWorld::new(vec![make_body(0.5), make_body(1.45)], v(0.0, 0.0, 0.0), 2.0)
                .unwrap();
        let _ = world.step(0.01).unwrap();
        let contacts = world.readback_contacts().unwrap();
        let pair_contacts = contacts
            .iter()
            .filter(|contact| contact.body_b == Some(1))
            .collect::<Vec<_>>();
        assert_eq!(pair_contacts.len(), 4);
        assert!(pair_contacts.iter().all(|contact| {
            contact.body_a == 0 && contact.body_b == Some(1) && (contact.depth - 0.05).abs() < 1e-4
        }));
    }

    #[test]
    fn gpu_sphere_batch_exposes_environment_local_topology_edits() {
        let Ok(_device) = GpuContactDevice::new() else {
            return;
        };
        let first = SphereInput {
            center: v(0.0, 0.0, 3.0),
            velocity: v(0.0, 0.0, 0.0),
            radius: 0.5,
            mass: 1.0,
        };
        let second = SphereInput {
            center: v(4.0, 0.0, 3.0),
            ..first.clone()
        };
        let batch = GpuSphereBatch::new(
            vec![vec![first.clone()], vec![second.clone()]],
            v(0.0, 0.0, 0.0),
            10.0,
        )
        .unwrap();
        batch.set_max_linear_speed(Some(3.0)).unwrap();
        assert_eq!(batch.max_linear_speed().unwrap(), Some(3.0));
        let added = SphereInput {
            center: v(8.0, 0.0, 3.0),
            radius: 0.75,
            ..first.clone()
        };
        assert_eq!(batch.add_body(0, added.clone()).unwrap(), 1);
        assert_eq!(batch.readback_environment(0).unwrap().len(), 2);
        assert_eq!(batch.readback_environment(1).unwrap()[0].center.x, 4.0);
        batch
            .reset_environment(0, vec![first.clone(), added])
            .unwrap();
        let removed = batch.remove_body(0, 1).unwrap();
        assert_eq!(removed.state.center.x, 8.0);
        assert_eq!(removed.radius, 0.75);
        batch.reset_environment(0, vec![first.clone()]).unwrap();
        batch.reset_all(vec![vec![first], vec![second]]).unwrap();
    }

    #[test]
    fn gpu_sphere_batch_discard_keeps_python_radii_aligned() {
        let Ok(_device) = GpuContactDevice::new() else {
            return;
        };
        let first = SphereInput {
            center: v(0.0, 0.0, 3.0),
            velocity: v(0.0, 0.0, 0.0),
            radius: 0.5,
            mass: 1.0,
        };
        let second = SphereInput {
            center: v(4.0, 0.0, 3.0),
            radius: 0.75,
            ..first.clone()
        };
        let batch = GpuSphereBatch::new(
            vec![vec![first.clone()], vec![second.clone(), first.clone()]],
            v(0.0, 0.0, 0.0),
            10.0,
        )
        .unwrap();
        batch.discard_environment(0).unwrap();
        batch.discard_body(0, 0).unwrap();
        assert_eq!(batch.readback_environment(0).unwrap()[0].center.x, 0.0);
        assert_eq!(batch.add_environment(vec![second]).unwrap(), 1);
        let point = GpuPointQuery {
            point: v(2.0, 0.0, 3.0),
            max_distance: 10.0,
            groups: GpuCollisionGroups {
                memberships: u32::MAX,
                filter: u32::MAX,
            },
            excluded_body: None,
            body_range: None,
            solid: false,
        };
        let hits = batch
            .query_scene_environments(vec![vec![], vec![]], vec![vec![point.clone()], vec![point]])
            .unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].points[0].as_ref().unwrap().body, Some(0));
        assert_eq!(hits[1].points[0].as_ref().unwrap().body, Some(0));
        assert!(batch.query_scene_environments(vec![], vec![]).is_err());
        let removed = batch.remove_body(1, 0).unwrap();
        assert_eq!(removed.radius, 0.75);
        assert_eq!(batch.len().unwrap(), 2);
    }
}
