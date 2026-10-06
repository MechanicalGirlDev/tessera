//! Device-resident sphere, capsule, and box contact for fixed-root articulated links.

use core::{
    mem::{offset_of, size_of},
    ops::Range,
    sync::atomic::{AtomicBool, Ordering},
};
use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, OnceLock},
};

use bytemuck::Zeroable;
use nalgebra::{Isometry3, Quaternion, UnitQuaternion, Vector3};
use wgpu::util::DeviceExt;

use crate::articulated_world::{JointPolynomialCoupling, LinkFixedConstraint, LinkPointConstraint};
use crate::articulation::Articulation;
use crate::convex::ConvexGeometry;
use crate::gpu_articulated_mass_assembly::GpuArticulatedMassAssemblyBatch;
use crate::gpu_articulated_pose::GpuArticulatedPoseBatch;
use crate::gpu_articulated_state::GpuGeneralizedStateBatch;
use crate::gpu_lbvh::{GpuLbvh, GpuLbvhError, GpuLbvhResidentPairs};
use crate::gpu_rigid_shape::mesh_bvh_nodes;
use crate::material::CoefficientCombineRule;
use crate::mesh::{PolylineGeometry, TriangleMeshGeometry};
use crate::sleep::SleepSettings;

/// A sphere or zero-radius point fixed to one link, colliding against a static world plane.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuArticulatedGroundSphere {
    /// Link index in the articulation's stable input order.
    pub link: usize,
    /// Sphere center in link-local coordinates.
    pub local_center: Vector3<f64>,
    /// Nonnegative sphere radius; zero represents a ground contact point.
    pub radius: f64,
    /// Plane normal in world coordinates; the plane is `normal dot point = offset`.
    pub plane_normal: Vector3<f64>,
    /// Plane offset in the units of `plane_normal`.
    pub plane_offset: f64,
    /// Optional half-extent of a square ground region in world X and Y.
    /// This is supported only for a plane with a positive world Z normal.
    pub plane_xy_half_extent: Option<f64>,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

/// A capsule fixed to one link, colliding against a static world plane.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuArticulatedGroundCapsule {
    /// Link index in the articulation's stable input order.
    pub link: usize,
    /// First endpoint of the capsule axis in link-local coordinates.
    pub local_a: Vector3<f64>,
    /// Second endpoint of the capsule axis in link-local coordinates.
    pub local_b: Vector3<f64>,
    /// Positive radius around the capsule axis.
    pub radius: f64,
    /// Plane normal in world coordinates; the plane is `normal dot point = offset`.
    pub plane_normal: Vector3<f64>,
    /// Plane offset in the units of `plane_normal`.
    pub plane_offset: f64,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

impl GpuArticulatedGroundCapsule {
    /// Return endpoint contacts; a zero-length capsule has one sphere contact.
    pub fn endpoint_contacts(self) -> Vec<GpuArticulatedGroundSphere> {
        let first = GpuArticulatedGroundSphere {
            link: self.link,
            local_center: self.local_a,
            radius: self.radius,
            plane_normal: self.plane_normal,
            plane_offset: self.plane_offset,
            plane_xy_half_extent: None,
            restitution: self.restitution,
            friction: self.friction,
        };
        if self.local_a == self.local_b {
            vec![first]
        } else {
            vec![
                first,
                GpuArticulatedGroundSphere {
                    local_center: self.local_b,
                    ..first
                },
            ]
        }
    }
}

/// A pair of spheres fixed to different links of the same articulation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuArticulatedSpherePair {
    /// First link index in stable articulation order.
    pub first_link: usize,
    /// First sphere center in its link-local coordinates.
    pub first_local_center: Vector3<f64>,
    /// Positive first sphere radius.
    pub first_radius: f64,
    /// Second link index in stable articulation order.
    pub second_link: usize,
    /// Second sphere center in its link-local coordinates.
    pub second_local_center: Vector3<f64>,
    /// Positive second sphere radius.
    pub second_radius: f64,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

/// Prescribed world-space velocity of an external sphere at its center.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct GpuArticulatedSphereMotion {
    /// Linear velocity of the sphere center.
    pub linear_velocity: Vector3<f64>,
    /// Angular velocity about the sphere center.
    pub angular_velocity: Vector3<f64>,
}

/// World-space body pose and origin velocity for a rigidly offset external sphere.
/// Angular velocity is supplied by the corresponding sphere motion update.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuArticulatedSphereOrbit {
    /// Current body origin in world coordinates.
    pub origin: Vector3<f64>,
    /// Body orientation in world coordinates, independent of sphere geometry.
    pub orientation: UnitQuaternion<f64>,
    /// Prescribed world-space linear velocity of the body origin.
    pub linear_velocity: Vector3<f64>,
}

/// Current rigid sphere body poses and origin velocities grouped by contact pair.
/// None indicates center-based motion without a tracked body origin.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GpuArticulatedExternalSphereOrbits {
    /// Origins for external spheres paired with link spheres.
    pub spheres: Vec<Vec<Option<GpuArticulatedSphereOrbit>>>,
    /// Origins for external spheres paired with link capsules.
    pub capsules: Vec<Vec<Option<GpuArticulatedSphereOrbit>>>,
    /// Origins for external spheres paired with link boxes.
    pub boxes: Vec<Vec<Option<GpuArticulatedSphereOrbit>>>,
    /// Origins for external spheres paired with moving cylinders or cones.
    pub axial: Vec<Vec<Option<GpuArticulatedSphereOrbit>>>,
    /// Origins for external spheres paired with moving convex hulls.
    pub convex: Vec<Vec<Option<GpuArticulatedSphereOrbit>>>,
}

/// Initial prescribed rigid body state for one external sphere contact pair.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuArticulatedSphereBodyInput {
    /// Optional caller-owned body identifier, preserved in the input mapping.
    pub user_data: Option<usize>,
    /// Body pose and origin linear velocity.
    pub orbit: GpuArticulatedSphereOrbit,
    /// Prescribed world-space body angular velocity.
    pub angular_velocity: Vector3<f64>,
}

/// Initial external sphere bodies for one dynamics environment.
/// Each field follows its corresponding contact-pair order; None is stationary.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GpuArticulatedExternalSphereBodies {
    /// Bodies for external spheres paired with link spheres.
    pub spheres: Vec<Option<GpuArticulatedSphereBodyInput>>,
    /// Bodies for external spheres paired with link capsules.
    pub capsules: Vec<Option<GpuArticulatedSphereBodyInput>>,
    /// Bodies for external spheres paired with link boxes.
    pub boxes: Vec<Option<GpuArticulatedSphereBodyInput>>,
    /// Bodies for external spheres paired with moving cylinders or cones.
    pub axial: Vec<Option<GpuArticulatedSphereBodyInput>>,
    /// Bodies for external spheres paired with moving convex hulls.
    pub convex: Vec<Option<GpuArticulatedSphereBodyInput>>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PackedSphereOrbit {
    origin: [f32; 4],
    linear: [f32; 4],
    orientation: [f32; 4],
    translation_anchor: [f32; 4],
}

#[derive(Clone, Copy)]
enum ExternalSphereContactKind {
    Sphere,
    Capsule,
    Box,
    Axial,
    Convex,
}

/// Current external sphere centers grouped by environment and contact pair.
/// Each field follows the ordering of its corresponding center update method.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GpuArticulatedExternalSphereCenters {
    /// External spheres paired with link spheres.
    pub spheres: Vec<Vec<Vector3<f64>>>,
    /// External spheres paired with link capsules.
    pub capsules: Vec<Vec<Vector3<f64>>>,
    /// External spheres paired with link boxes.
    pub boxes: Vec<Vec<Vector3<f64>>>,
    /// External spheres paired with moving cylinders or cones.
    pub axial: Vec<Vec<Vector3<f64>>>,
    /// External spheres paired with moving convex hulls.
    pub convex: Vec<Vec<Vector3<f64>>>,
}

/// Prescribed external sphere motion grouped by environment and contact pair.
/// All fields must contain one vector per environment, including empty pairs.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GpuArticulatedExternalSphereMotions {
    /// Motion for external spheres paired with link spheres.
    pub spheres: Vec<Vec<GpuArticulatedSphereMotion>>,
    /// Motion for external spheres paired with link capsules.
    pub capsules: Vec<Vec<GpuArticulatedSphereMotion>>,
    /// Motion for external spheres paired with link boxes.
    pub boxes: Vec<Vec<GpuArticulatedSphereMotion>>,
    /// Motion for external spheres paired with moving cylinders or cones.
    pub axial: Vec<Vec<GpuArticulatedSphereMotion>>,
    /// Motion for external spheres paired with moving convex hulls.
    pub convex: Vec<Vec<GpuArticulatedSphereMotion>>,
}

/// A link sphere colliding with a stationary world-space sphere.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuArticulatedStaticSpherePair {
    /// Link carrying the moving sphere.
    pub link: usize,
    /// Moving sphere center in link-local coordinates.
    pub local_center: Vector3<f64>,
    /// Positive moving sphere radius.
    pub radius: f64,
    /// Stationary sphere center in world coordinates.
    pub static_center: Vector3<f64>,
    /// Positive stationary sphere radius.
    pub static_radius: f64,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

/// A link capsule colliding with a stationary world-space sphere.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuArticulatedStaticCapsuleSpherePair {
    /// Link carrying the moving capsule.
    pub link: usize,
    /// First capsule axis endpoint in link-local coordinates.
    pub local_a: Vector3<f64>,
    /// Second capsule axis endpoint in link-local coordinates.
    pub local_b: Vector3<f64>,
    /// Positive capsule radius.
    pub radius: f64,
    /// Stationary sphere center in world coordinates.
    pub static_center: Vector3<f64>,
    /// Positive stationary sphere radius.
    pub static_radius: f64,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

/// A link box colliding with a stationary world-space sphere.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuArticulatedStaticBoxSpherePair {
    /// Link carrying the moving box.
    pub link: usize,
    /// Box pose in link-local coordinates.
    pub local_pose: Isometry3<f64>,
    /// Positive box half extents.
    pub half_extents: Vector3<f64>,
    /// Stationary sphere center in world coordinates.
    pub static_center: Vector3<f64>,
    /// Positive stationary sphere radius.
    pub static_radius: f64,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

/// A link sphere colliding with a stationary world-space capsule.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuArticulatedStaticSphereCapsulePair {
    /// Link carrying the moving sphere.
    pub link: usize,
    /// Moving sphere center in link-local coordinates.
    pub local_center: Vector3<f64>,
    /// Positive moving sphere radius.
    pub radius: f64,
    /// First stationary capsule axis endpoint in world coordinates.
    pub static_a: Vector3<f64>,
    /// Second stationary capsule axis endpoint in world coordinates.
    pub static_b: Vector3<f64>,
    /// Nonnegative stationary capsule radius; zero represents a polyline segment.
    pub static_radius: f64,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

/// A link capsule colliding with a stationary world-space capsule.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuArticulatedStaticCapsulePair {
    /// Link carrying the moving capsule.
    pub link: usize,
    /// First moving capsule axis endpoint in link-local coordinates.
    pub local_a: Vector3<f64>,
    /// Second moving capsule axis endpoint in link-local coordinates.
    pub local_b: Vector3<f64>,
    /// Positive moving capsule radius.
    pub radius: f64,
    /// First stationary capsule axis endpoint in world coordinates.
    pub static_a: Vector3<f64>,
    /// Second stationary capsule axis endpoint in world coordinates.
    pub static_b: Vector3<f64>,
    /// Nonnegative stationary capsule radius; zero represents a polyline segment.
    pub static_radius: f64,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

/// A link sphere colliding with a stationary world-space box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuArticulatedStaticSphereBoxPair {
    /// Link carrying the moving sphere.
    pub link: usize,
    /// Moving sphere center in link-local coordinates.
    pub local_center: Vector3<f64>,
    /// Positive moving sphere radius.
    pub radius: f64,
    /// Stationary box pose in world coordinates.
    pub static_pose: Isometry3<f64>,
    /// Positive stationary box half extents.
    pub half_extents: Vector3<f64>,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

/// A link capsule colliding with a stationary world-space box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuArticulatedStaticCapsuleBoxPair {
    /// Link carrying the moving capsule.
    pub link: usize,
    /// First moving capsule axis endpoint in link-local coordinates.
    pub local_a: Vector3<f64>,
    /// Second moving capsule axis endpoint in link-local coordinates.
    pub local_b: Vector3<f64>,
    /// Positive moving capsule radius.
    pub radius: f64,
    /// Stationary box pose in world coordinates.
    pub static_pose: Isometry3<f64>,
    /// Positive stationary box half extents.
    pub half_extents: Vector3<f64>,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

/// A link box colliding with a stationary world-space box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuArticulatedStaticBoxPair {
    /// Link carrying the moving box.
    pub link: usize,
    /// Moving box pose in link-local coordinates.
    pub local_pose: Isometry3<f64>,
    /// Positive moving box half extents.
    pub half_extents: Vector3<f64>,
    /// Stationary box pose in world coordinates.
    pub static_pose: Isometry3<f64>,
    /// Positive stationary box half extents.
    pub static_half_extents: Vector3<f64>,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

/// A link box colliding with a stationary world-space capsule.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuArticulatedStaticBoxCapsulePair {
    /// Link carrying the moving box.
    pub link: usize,
    /// Moving box pose in link-local coordinates.
    pub local_pose: Isometry3<f64>,
    /// Positive moving box half extents.
    pub half_extents: Vector3<f64>,
    /// First stationary capsule axis endpoint in world coordinates.
    pub static_a: Vector3<f64>,
    /// Second stationary capsule axis endpoint in world coordinates.
    pub static_b: Vector3<f64>,
    /// Nonnegative stationary capsule radius; zero represents a polyline segment.
    pub static_radius: f64,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

/// An axial shape and sphere where exactly one belongs to a stationary scene body.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuArticulatedStaticAxialSpherePair {
    /// Link carrying the moving shape.
    pub link: usize,
    /// The axial pose is in world coordinates and the sphere center is link-local when true.
    pub axial_is_static: bool,
    /// Analytic shape kind.
    pub kind: GpuArticulatedGroundAxialKind,
    /// Axial shape pose in link-local coordinates, or world coordinates when static.
    pub local_pose: Isometry3<f64>,
    /// Positive shape half-height.
    pub half_height: f64,
    /// Positive shape radius.
    pub radius: f64,
    /// Sphere center in world coordinates, or link-local coordinates when axial is static.
    pub static_center: Vector3<f64>,
    /// Positive sphere radius.
    pub static_radius: f64,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

/// An axial shape and capsule where exactly one belongs to a stationary scene body.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuArticulatedStaticAxialCapsulePair {
    /// Link carrying the moving shape.
    pub link: usize,
    /// The axial pose is in world coordinates and capsule endpoints are link-local when true.
    pub axial_is_static: bool,
    /// Analytic shape kind.
    pub kind: GpuArticulatedGroundAxialKind,
    /// Axial shape pose in link-local coordinates, or world coordinates when static.
    pub local_pose: Isometry3<f64>,
    /// Positive shape half-height.
    pub half_height: f64,
    /// Positive shape radius.
    pub radius: f64,
    /// First capsule endpoint in world coordinates, or link-local coordinates when axial is static.
    pub static_a: Vector3<f64>,
    /// Second capsule endpoint in world coordinates, or link-local coordinates when axial is static.
    pub static_b: Vector3<f64>,
    /// Nonnegative capsule radius; zero represents a stationary polyline segment.
    pub static_radius: f64,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

/// An axial shape and box where exactly one belongs to a stationary scene body.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuArticulatedStaticAxialBoxPair {
    /// Link carrying the moving shape.
    pub link: usize,
    /// The axial pose is in world coordinates and the box pose is link-local when true.
    pub axial_is_static: bool,
    /// Analytic shape kind.
    pub kind: GpuArticulatedGroundAxialKind,
    /// Axial shape pose in link-local coordinates, or world coordinates when static.
    pub local_pose: Isometry3<f64>,
    /// Positive shape half-height.
    pub half_height: f64,
    /// Positive shape radius.
    pub radius: f64,
    /// Box pose in world coordinates, or link-local coordinates when axial is static.
    pub static_pose: Isometry3<f64>,
    /// Positive box half extents.
    pub static_half_extents: Vector3<f64>,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

/// A link cylinder or cone colliding with a stationary world-space convex hull.
#[derive(Debug, Clone)]
pub struct GpuArticulatedStaticAxialConvexPair {
    /// Link carrying the moving axial shape.
    pub link: usize,
    /// Analytic shape kind.
    pub kind: GpuArticulatedGroundAxialKind,
    /// Shape pose in link-local coordinates.
    pub local_pose: Isometry3<f64>,
    /// Positive shape half-height.
    pub half_height: f64,
    /// Positive shape radius.
    pub radius: f64,
    /// Stationary hull pose in world coordinates.
    pub static_pose: Isometry3<f64>,
    /// Stationary hull geometry in its shape coordinates.
    pub static_geometry: ConvexGeometry,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

/// A cylinder or cone colliding with a link convex hull, optionally from a stationary scene.
#[derive(Debug, Clone)]
pub struct GpuArticulatedAxialConvexPair {
    /// Link carrying the axial shape; ignored when the axial shape is stationary.
    pub axial_link: usize,
    /// Whether the axial shape uses a stationary world-space pose.
    pub axial_is_static: bool,
    /// Analytic axial shape kind.
    pub kind: GpuArticulatedGroundAxialKind,
    /// Axial pose in link coordinates, or world coordinates when stationary.
    pub axial_local_pose: Isometry3<f64>,
    /// Positive axial half-height.
    pub half_height: f64,
    /// Positive axial radius.
    pub radius: f64,
    /// Link carrying the convex hull.
    pub convex_link: usize,
    /// Convex hull pose in its link coordinates.
    pub convex_local_pose: Isometry3<f64>,
    /// Hull vertices, face normals, and edge directions.
    pub convex_geometry: ConvexGeometry,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

/// A link cylinder or cone colliding with a sphere on another link.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuArticulatedAxialSpherePair {
    /// Link carrying the moving axial shape.
    pub axial_link: usize,
    /// Analytic shape kind.
    pub kind: GpuArticulatedGroundAxialKind,
    /// Axial shape pose in its link coordinates.
    pub axial_local_pose: Isometry3<f64>,
    /// Positive axial shape half-height.
    pub half_height: f64,
    /// Positive axial shape radius.
    pub radius: f64,
    /// Link carrying the sphere.
    pub sphere_link: usize,
    /// Sphere center in its link coordinates.
    pub sphere_local_center: Vector3<f64>,
    /// Positive sphere radius.
    pub sphere_radius: f64,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

/// A link cylinder or cone colliding with a box on another link.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuArticulatedAxialBoxPair {
    /// Link carrying the axial shape.
    pub axial_link: usize,
    /// Analytic axial shape kind.
    pub kind: GpuArticulatedGroundAxialKind,
    /// Axial shape pose in its link coordinates.
    pub axial_local_pose: Isometry3<f64>,
    /// Positive axial shape half-height.
    pub half_height: f64,
    /// Positive axial shape radius.
    pub radius: f64,
    /// Link carrying the box.
    pub box_link: usize,
    /// Box pose in its link coordinates.
    pub box_local_pose: Isometry3<f64>,
    /// Positive box half extents.
    pub box_half_extents: Vector3<f64>,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

/// A link cylinder or cone colliding with a capsule on another link.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuArticulatedAxialCapsulePair {
    /// Link carrying the axial shape.
    pub axial_link: usize,
    /// Analytic axial shape kind.
    pub kind: GpuArticulatedGroundAxialKind,
    /// Axial shape pose in its link coordinates.
    pub axial_local_pose: Isometry3<f64>,
    /// Positive axial shape half-height.
    pub half_height: f64,
    /// Positive axial shape radius.
    pub radius: f64,
    /// Link carrying the capsule.
    pub capsule_link: usize,
    /// First capsule-axis endpoint in its link coordinates.
    pub capsule_local_a: Vector3<f64>,
    /// Second capsule-axis endpoint in its link coordinates.
    pub capsule_local_b: Vector3<f64>,
    /// Positive capsule radius.
    pub capsule_radius: f64,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

/// Two cylinders or cones, optionally with the first stationary in world coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuArticulatedAxialPair {
    /// Link carrying the first shape; ignored when the first shape is stationary.
    pub first_link: usize,
    /// Whether the first shape uses a stationary world-space pose.
    pub first_is_static: bool,
    /// Kind of the first shape.
    pub first_kind: GpuArticulatedGroundAxialKind,
    /// First shape pose in link coordinates, or world coordinates when stationary.
    pub first_local_pose: Isometry3<f64>,
    /// Positive first shape half-height.
    pub first_half_height: f64,
    /// Positive first shape radius.
    pub first_radius: f64,
    /// Link carrying the second shape.
    pub second_link: usize,
    /// Kind of the second shape.
    pub second_kind: GpuArticulatedGroundAxialKind,
    /// Second shape pose in its link coordinates.
    pub second_local_pose: Isometry3<f64>,
    /// Positive second shape half-height.
    pub second_half_height: f64,
    /// Positive second shape radius.
    pub second_radius: f64,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

/// A link convex hull colliding with a sphere on another link.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuArticulatedConvexSpherePair {
    /// Link carrying the convex hull.
    pub convex_link: usize,
    /// Hull pose in its link coordinates.
    pub convex_local_pose: Isometry3<f64>,
    /// Hull vertices in hull-local coordinates.
    pub vertices: Vec<Vector3<f64>>,
    /// Outward hull face normals in hull-local coordinates.
    pub face_normals: Vec<Vector3<f64>>,
    /// Link carrying the sphere.
    pub sphere_link: usize,
    /// Sphere center in its link coordinates.
    pub sphere_local_center: Vector3<f64>,
    /// Positive sphere radius.
    pub sphere_radius: f64,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

/// A moving link convex hull colliding with a stationary world-space sphere.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuArticulatedStaticConvexSpherePair {
    /// Link carrying the convex hull.
    pub convex_link: usize,
    /// Hull pose in its link coordinates.
    pub convex_local_pose: Isometry3<f64>,
    /// Hull vertices in hull-local coordinates.
    pub vertices: Vec<Vector3<f64>>,
    /// Outward hull face normals in hull-local coordinates.
    pub face_normals: Vec<Vector3<f64>>,
    /// Fixed sphere center in world coordinates.
    pub static_center: Vector3<f64>,
    /// Positive sphere radius.
    pub static_radius: f64,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

/// A stationary world-space convex hull colliding with a moving link sphere.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuArticulatedSceneConvexSpherePair {
    /// Fixed hull pose in world coordinates.
    pub convex_world_pose: Isometry3<f64>,
    /// Hull vertices in hull-local coordinates.
    pub vertices: Vec<Vector3<f64>>,
    /// Outward hull face normals in hull-local coordinates.
    pub face_normals: Vec<Vector3<f64>>,
    /// Link carrying the sphere.
    pub sphere_link: usize,
    /// Sphere center in link coordinates.
    pub sphere_local_center: Vector3<f64>,
    /// Positive sphere radius.
    pub sphere_radius: f64,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

/// A stationary indexed mesh paired with a moving link sphere.
#[derive(Debug, Clone)]
pub struct GpuArticulatedSceneMeshSpherePair {
    /// Fixed mesh pose in world coordinates.
    pub mesh_world_pose: Isometry3<f64>,
    /// Indexed triangle mesh shared by all pairs using the same scene collider.
    pub mesh: Arc<TriangleMeshGeometry>,
    /// Link carrying the sphere.
    pub sphere_link: usize,
    /// Sphere center in link coordinates.
    pub sphere_local_center: Vector3<f64>,
    /// Positive sphere radius.
    pub sphere_radius: f64,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

/// A stationary indexed polyline paired with a moving link sphere.
#[derive(Debug, Clone)]
pub struct GpuArticulatedScenePolylineSpherePair {
    /// Fixed polyline pose in world coordinates.
    pub polyline_world_pose: Isometry3<f64>,
    /// Indexed segments shared by all pairs using the same scene collider.
    pub polyline: Arc<PolylineGeometry>,
    /// Link carrying the sphere.
    pub sphere_link: usize,
    /// Sphere center in link coordinates.
    pub sphere_local_center: Vector3<f64>,
    /// Positive sphere radius.
    pub sphere_radius: f64,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

/// A stationary indexed mesh paired with a moving link capsule.
#[derive(Debug, Clone)]
pub struct GpuArticulatedSceneMeshCapsulePair {
    /// Fixed mesh pose in world coordinates.
    pub mesh_world_pose: Isometry3<f64>,
    /// Indexed triangle mesh shared by all pairs using the same scene collider.
    pub mesh: Arc<TriangleMeshGeometry>,
    /// Link carrying the capsule.
    pub capsule_link: usize,
    /// First capsule segment endpoint in link coordinates.
    pub capsule_local_a: Vector3<f64>,
    /// Second capsule segment endpoint in link coordinates.
    pub capsule_local_b: Vector3<f64>,
    /// Positive capsule radius.
    pub capsule_radius: f64,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

/// A stationary indexed polyline paired with a moving link capsule.
#[derive(Debug, Clone)]
pub struct GpuArticulatedScenePolylineCapsulePair {
    /// Fixed polyline pose in world coordinates.
    pub polyline_world_pose: Isometry3<f64>,
    /// Indexed segments shared by all pairs using the same scene collider.
    pub polyline: Arc<PolylineGeometry>,
    /// Link carrying the capsule.
    pub capsule_link: usize,
    /// First capsule segment endpoint in link coordinates.
    pub capsule_local_a: Vector3<f64>,
    /// Second capsule segment endpoint in link coordinates.
    pub capsule_local_b: Vector3<f64>,
    /// Positive capsule radius.
    pub capsule_radius: f64,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

/// A stationary indexed mesh paired with a moving link box.
#[derive(Debug, Clone)]
pub struct GpuArticulatedSceneMeshBoxPair {
    /// Fixed mesh pose in world coordinates.
    pub mesh_world_pose: Isometry3<f64>,
    /// Indexed triangle mesh shared by all pairs using the same scene collider.
    pub mesh: Arc<TriangleMeshGeometry>,
    /// Link carrying the box.
    pub box_link: usize,
    /// Box pose in link coordinates.
    pub box_local_pose: Isometry3<f64>,
    /// Positive box half extents.
    pub box_half_extents: Vector3<f64>,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

/// A stationary indexed polyline paired with a moving link box.
#[derive(Debug, Clone)]
pub struct GpuArticulatedScenePolylineBoxPair {
    /// Fixed polyline pose in world coordinates.
    pub polyline_world_pose: Isometry3<f64>,
    /// Indexed segments shared by all pairs using the same scene collider.
    pub polyline: Arc<PolylineGeometry>,
    /// Link carrying the box.
    pub box_link: usize,
    /// Box pose in link coordinates.
    pub box_local_pose: Isometry3<f64>,
    /// Positive box half extents.
    pub box_half_extents: Vector3<f64>,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

/// A stationary indexed mesh paired with a moving link cylinder or cone.
#[derive(Debug, Clone)]
pub struct GpuArticulatedSceneMeshAxialPair {
    /// Fixed mesh pose in world coordinates.
    pub mesh_world_pose: Isometry3<f64>,
    /// Indexed triangle mesh shared by all pairs using the same scene collider.
    pub mesh: Arc<TriangleMeshGeometry>,
    /// Link carrying the axial shape.
    pub link: usize,
    /// Analytic shape kind.
    pub kind: GpuArticulatedGroundAxialKind,
    /// Shape pose in link coordinates.
    pub local_pose: Isometry3<f64>,
    /// Positive shape half-height.
    pub half_height: f64,
    /// Positive shape radius.
    pub radius: f64,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

/// A stationary indexed polyline paired with a moving link cylinder or cone.
#[derive(Debug, Clone)]
pub struct GpuArticulatedScenePolylineAxialPair {
    /// Fixed polyline pose in world coordinates.
    pub polyline_world_pose: Isometry3<f64>,
    /// Indexed segments shared by all pairs using the same scene collider.
    pub polyline: Arc<PolylineGeometry>,
    /// Link carrying the axial shape.
    pub link: usize,
    /// Analytic shape kind.
    pub kind: GpuArticulatedGroundAxialKind,
    /// Shape pose in link coordinates.
    pub local_pose: Isometry3<f64>,
    /// Positive shape half-height.
    pub half_height: f64,
    /// Positive shape radius.
    pub radius: f64,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

/// A stationary indexed mesh paired with a moving link convex hull.
#[derive(Debug, Clone)]
pub struct GpuArticulatedSceneMeshConvexPair {
    /// Fixed mesh pose in world coordinates.
    pub mesh_world_pose: Isometry3<f64>,
    /// Indexed triangle mesh shared by all pairs using the same scene collider.
    pub mesh: Arc<TriangleMeshGeometry>,
    /// Link carrying the convex hull.
    pub convex_link: usize,
    /// Hull pose in link coordinates.
    pub convex_local_pose: Isometry3<f64>,
    /// Convex hull vertices, face normals, and edge directions.
    pub convex_geometry: ConvexGeometry,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

/// A stationary indexed polyline paired with a moving link convex hull.
#[derive(Debug, Clone)]
pub struct GpuArticulatedScenePolylineConvexPair {
    /// Fixed polyline pose in world coordinates.
    pub polyline_world_pose: Isometry3<f64>,
    /// Indexed segments shared by all pairs using the same scene collider.
    pub polyline: Arc<PolylineGeometry>,
    /// Link carrying the convex hull.
    pub convex_link: usize,
    /// Convex hull pose in link coordinates.
    pub convex_local_pose: Isometry3<f64>,
    /// Convex hull geometry.
    pub convex_geometry: Arc<ConvexGeometry>,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

/// A stationary world-space convex hull colliding with a moving link capsule.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuArticulatedSceneConvexCapsulePair {
    /// Fixed hull pose in world coordinates.
    pub convex_world_pose: Isometry3<f64>,
    /// Hull vertices in hull-local coordinates.
    pub vertices: Vec<Vector3<f64>>,
    /// Outward hull face normals in hull-local coordinates.
    pub face_normals: Vec<Vector3<f64>>,
    /// Hull edge directions in hull-local coordinates.
    pub edge_directions: Vec<Vector3<f64>>,
    /// Link carrying the capsule.
    pub capsule_link: usize,
    /// First capsule segment endpoint in link coordinates.
    pub capsule_local_a: Vector3<f64>,
    /// Second capsule segment endpoint in link coordinates.
    pub capsule_local_b: Vector3<f64>,
    /// Positive capsule radius.
    pub capsule_radius: f64,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

/// A moving link convex hull colliding with a stationary world-space capsule.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuArticulatedStaticConvexCapsulePair {
    /// Link carrying the convex hull.
    pub convex_link: usize,
    /// Hull pose in its link coordinates.
    pub convex_local_pose: Isometry3<f64>,
    /// Hull vertices in hull-local coordinates.
    pub vertices: Vec<Vector3<f64>>,
    /// Outward hull face normals in hull-local coordinates.
    pub face_normals: Vec<Vector3<f64>>,
    /// Unique unit hull edge directions in hull-local coordinates.
    pub edge_directions: Vec<Vector3<f64>>,
    /// First fixed capsule axis endpoint in world coordinates.
    pub static_a: Vector3<f64>,
    /// Second fixed capsule axis endpoint in world coordinates.
    pub static_b: Vector3<f64>,
    /// Nonnegative capsule radius; zero represents a polyline segment.
    pub static_radius: f64,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

/// A link convex hull colliding with a capsule on another link.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuArticulatedConvexCapsulePair {
    /// Link carrying the convex hull.
    pub convex_link: usize,
    /// Hull pose in its link coordinates.
    pub convex_local_pose: Isometry3<f64>,
    /// Hull vertices in hull-local coordinates.
    pub vertices: Vec<Vector3<f64>>,
    /// Outward hull face normals in hull-local coordinates.
    pub face_normals: Vec<Vector3<f64>>,
    /// Unique unit hull edge directions in hull-local coordinates.
    pub edge_directions: Vec<Vector3<f64>>,
    /// Link carrying the capsule.
    pub capsule_link: usize,
    /// First capsule axis endpoint in link coordinates.
    pub capsule_local_a: Vector3<f64>,
    /// Second capsule axis endpoint in link coordinates.
    pub capsule_local_b: Vector3<f64>,
    /// Positive capsule radius.
    pub capsule_radius: f64,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

/// Two convex polyhedra fixed to different links of one articulation.
#[derive(Debug, Clone)]
pub struct GpuArticulatedConvexPair {
    /// First hull link index.
    pub first_link: usize,
    /// First hull pose in link coordinates.
    pub first_local_pose: Isometry3<f64>,
    /// First hull geometry in its shape coordinates.
    pub first_geometry: ConvexGeometry,
    /// Second hull link index.
    pub second_link: usize,
    /// Second hull pose in link coordinates.
    pub second_local_pose: Isometry3<f64>,
    /// Second hull geometry in its shape coordinates.
    pub second_geometry: ConvexGeometry,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

/// A moving link convex hull colliding with a stationary world-space convex hull.
#[derive(Debug, Clone)]
pub struct GpuArticulatedStaticConvexPair {
    /// Moving hull link index.
    pub first_link: usize,
    /// Moving hull pose in link coordinates.
    pub first_local_pose: Isometry3<f64>,
    /// Moving hull geometry in its shape coordinates.
    pub first_geometry: ConvexGeometry,
    /// Fixed hull pose in world coordinates.
    pub second_world_pose: Isometry3<f64>,
    /// Fixed hull geometry in its shape coordinates.
    pub second_geometry: ConvexGeometry,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

/// A capsule and a sphere fixed to different links of one articulation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuArticulatedCapsuleSpherePair {
    /// Link carrying the capsule.
    pub capsule_link: usize,
    /// First capsule-axis endpoint in link-local coordinates.
    pub capsule_local_a: Vector3<f64>,
    /// Second capsule-axis endpoint in link-local coordinates.
    pub capsule_local_b: Vector3<f64>,
    /// Positive capsule radius.
    pub capsule_radius: f64,
    /// Link carrying the sphere.
    pub sphere_link: usize,
    /// Sphere center in link-local coordinates.
    pub sphere_local_center: Vector3<f64>,
    /// Positive sphere radius.
    pub sphere_radius: f64,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

/// A link-local box colliding against a static world plane.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuArticulatedGroundBox {
    /// Link index in the articulation's stable input order.
    pub link: usize,
    /// Box pose in link-local coordinates.
    pub local_pose: Isometry3<f64>,
    /// Positive half extents along the box-local axes.
    pub half_extents: Vector3<f64>,
    /// Plane normal in world coordinates; the plane is `normal dot point = offset`.
    pub plane_normal: Vector3<f64>,
    /// Plane offset in the units of `plane_normal`.
    pub plane_offset: f64,
    /// Optional half-extent of a square ground region in world X and Y.
    /// This is supported only for a plane with a positive world Z normal.
    pub plane_xy_half_extent: Option<f64>,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

/// Analytic axial primitive against a static plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuArticulatedGroundAxialKind {
    /// A circular cylinder with two planar caps.
    Cylinder,
    /// A circular cone with its tip along positive local Z.
    Cone,
}

/// Link-local cylinder or cone against a static world plane.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuArticulatedGroundAxialShape {
    /// Owning link index.
    pub link: usize,
    /// Kind of analytic primitive.
    pub kind: GpuArticulatedGroundAxialKind,
    /// Shape frame in link coordinates.
    pub local_pose: Isometry3<f64>,
    /// Half of the full axial length in metres.
    pub half_height: f64,
    /// Circular radius in metres.
    pub radius: f64,
    /// Plane normal in world coordinates.
    pub plane_normal: Vector3<f64>,
    /// Plane offset in the units of `plane_normal`.
    pub plane_offset: f64,
    /// Optional half-extent of a square ground region in world X and Y.
    pub plane_xy_half_extent: Option<f64>,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

/// A sphere and a box fixed to different links of one articulation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuArticulatedSphereBoxPair {
    /// Link carrying the sphere.
    pub sphere_link: usize,
    /// Sphere center in sphere-link coordinates.
    pub sphere_local_center: Vector3<f64>,
    /// Positive sphere radius.
    pub sphere_radius: f64,
    /// Link carrying the box.
    pub box_link: usize,
    /// Box pose in box-link coordinates.
    pub box_local_pose: Isometry3<f64>,
    /// Positive half extents along the box-local axes.
    pub box_half_extents: Vector3<f64>,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

/// Two capsules fixed to different links of one articulation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuArticulatedCapsulePair {
    /// First link index in stable articulation order.
    pub first_link: usize,
    /// First capsule-axis endpoint in first-link coordinates.
    pub first_local_a: Vector3<f64>,
    /// Second capsule-axis endpoint in first-link coordinates.
    pub first_local_b: Vector3<f64>,
    /// Positive first capsule radius.
    pub first_radius: f64,
    /// Second link index in stable articulation order.
    pub second_link: usize,
    /// First capsule-axis endpoint in second-link coordinates.
    pub second_local_a: Vector3<f64>,
    /// Second capsule-axis endpoint in second-link coordinates.
    pub second_local_b: Vector3<f64>,
    /// Positive second capsule radius.
    pub second_radius: f64,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

/// A link-local sphere whose eligible self-contact pairs are generated at construction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuArticulatedLinkSphere {
    /// Link index in stable articulation order.
    pub link: usize,
    /// Sphere center in link-local coordinates.
    pub local_center: Vector3<f64>,
    /// Positive sphere radius.
    pub radius: f64,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

/// A link-local capsule whose eligible self-contact pairs are generated at construction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuArticulatedLinkCapsule {
    /// Link index in stable articulation order.
    pub link: usize,
    /// First capsule-axis endpoint in link-local coordinates.
    pub local_a: Vector3<f64>,
    /// Second capsule-axis endpoint in link-local coordinates.
    pub local_b: Vector3<f64>,
    /// Positive capsule radius.
    pub radius: f64,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

/// A capsule and a box fixed to different links of one articulation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuArticulatedCapsuleBoxPair {
    /// Link carrying the capsule.
    pub capsule_link: usize,
    /// First capsule-axis endpoint in capsule-link coordinates.
    pub capsule_local_a: Vector3<f64>,
    /// Second capsule-axis endpoint in capsule-link coordinates.
    pub capsule_local_b: Vector3<f64>,
    /// Positive capsule radius.
    pub capsule_radius: f64,
    /// Link carrying the box.
    pub box_link: usize,
    /// Box pose in box-link coordinates.
    pub box_local_pose: Isometry3<f64>,
    /// Positive half extents along the box-local axes.
    pub box_half_extents: Vector3<f64>,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

/// A link-local box whose eligible sphere contacts are generated at construction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuArticulatedLinkBox {
    /// Link index in stable articulation order.
    pub link: usize,
    /// Box pose in link-local coordinates.
    pub local_pose: Isometry3<f64>,
    /// Positive half extents along the box-local axes.
    pub half_extents: Vector3<f64>,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

/// Two boxes fixed to different links of one articulation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuArticulatedBoxPair {
    /// First link index in stable articulation order.
    pub first_link: usize,
    /// First box pose in first-link coordinates.
    pub first_local_pose: Isometry3<f64>,
    /// Positive first box half extents.
    pub first_half_extents: Vector3<f64>,
    /// Second link index in stable articulation order.
    pub second_link: usize,
    /// Second box pose in second-link coordinates.
    pub second_local_pose: Isometry3<f64>,
    /// Positive second box half extents.
    pub second_half_extents: Vector3<f64>,
    /// Coefficient of restitution in `[0, 1]`.
    pub restitution: f64,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

/// Material rules for one shape participating in GPU-generated pairs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuArticulatedCombineRules {
    /// Friction rule requested by this shape.
    pub friction: CoefficientCombineRule,
    /// Restitution rule requested by this shape.
    pub restitution: CoefficientCombineRule,
}

/// Per-environment shape-pair and solver-iteration settings.
#[derive(Debug, Clone, Copy)]
pub struct GpuArticulatedContactSettings<'a> {
    /// Link boxes against static planes for each environment.
    pub boxes: &'a [Vec<GpuArticulatedGroundBox>],
    /// Cylinders and cones against static planes for each environment.
    pub axial_shapes: &'a [Vec<GpuArticulatedGroundAxialShape>],
    /// First ground sphere joining support-point reduction in each environment.
    pub manifold_starts: &'a [Option<usize>],
    /// Per-coordinate Coulomb friction effort bounds for each environment.
    pub joint_frictions: &'a [Vec<f64>],
    /// Polynomial coordinate equalities solved in the same sweeps as contact.
    pub joint_couplings: &'a [Vec<JointPolynomialCoupling>],
    /// Bilateral link point constraints solved with contact.
    pub link_point_constraints: &'a [Vec<LinkPointConstraint>],
    /// Bilateral link frame constraints solved with contact.
    pub link_fixed_constraints: &'a [Vec<LinkFixedConstraint>],
    /// Explicit or generated sphere pairs for each environment.
    pub pairs: &'a [Vec<GpuArticulatedSpherePair>],
    /// Moving link spheres against stationary world spheres.
    pub static_sphere_pairs: &'a [Vec<GpuArticulatedStaticSpherePair>],
    /// Moving link capsules against stationary world spheres.
    pub static_capsule_sphere_pairs: &'a [Vec<GpuArticulatedStaticCapsuleSpherePair>],
    /// Moving link boxes against stationary world spheres.
    pub static_box_sphere_pairs: &'a [Vec<GpuArticulatedStaticBoxSpherePair>],
    /// Moving link spheres against stationary world capsules.
    pub static_sphere_capsule_pairs: &'a [Vec<GpuArticulatedStaticSphereCapsulePair>],
    /// Moving link capsules against stationary world capsules.
    pub static_capsule_pairs: &'a [Vec<GpuArticulatedStaticCapsulePair>],
    /// Moving link spheres against stationary world boxes.
    pub static_sphere_box_pairs: &'a [Vec<GpuArticulatedStaticSphereBoxPair>],
    /// Moving link capsules against stationary world boxes.
    pub static_capsule_box_pairs: &'a [Vec<GpuArticulatedStaticCapsuleBoxPair>],
    /// Moving link boxes against stationary world boxes.
    pub static_box_pairs: &'a [Vec<GpuArticulatedStaticBoxPair>],
    /// Moving link boxes against stationary world capsules.
    pub static_box_capsule_pairs: &'a [Vec<GpuArticulatedStaticBoxCapsulePair>],
    /// Moving link cylinders and cones against stationary world spheres.
    pub static_axial_sphere_pairs: &'a [Vec<GpuArticulatedStaticAxialSpherePair>],
    /// Moving link cylinders and cones against stationary world capsules.
    pub static_axial_capsule_pairs: &'a [Vec<GpuArticulatedStaticAxialCapsulePair>],
    /// Moving link cylinders and cones against stationary world boxes.
    pub static_axial_box_pairs: &'a [Vec<GpuArticulatedStaticAxialBoxPair>],
    /// Moving link cylinders and cones against stationary world convex hulls.
    pub static_axial_convex_pairs: &'a [Vec<GpuArticulatedStaticAxialConvexPair>],
    /// Link cylinders and cones paired with convex hulls on other links.
    pub axial_convex_pairs: &'a [Vec<GpuArticulatedAxialConvexPair>],
    /// Moving link cylinders and cones against spheres on other links.
    pub axial_sphere_pairs: &'a [Vec<GpuArticulatedAxialSpherePair>],
    /// Moving link cylinders and cones against boxes on other links.
    pub axial_box_pairs: &'a [Vec<GpuArticulatedAxialBoxPair>],
    /// Moving link cylinders and cones against capsules on other links.
    pub axial_capsule_pairs: &'a [Vec<GpuArticulatedAxialCapsulePair>],
    /// Cylinder and cone pairs on distinct links.
    pub axial_pairs: &'a [Vec<GpuArticulatedAxialPair>],
    /// Link convex hulls against spheres on other links.
    pub convex_sphere_pairs: &'a [Vec<GpuArticulatedConvexSpherePair>],
    /// Link convex hulls against stationary world-space spheres.
    pub static_convex_sphere_pairs: &'a [Vec<GpuArticulatedStaticConvexSpherePair>],
    /// Link convex hulls against stationary world-space capsules.
    pub static_convex_capsule_pairs: &'a [Vec<GpuArticulatedStaticConvexCapsulePair>],
    /// Link convex hulls against capsules on other links.
    pub convex_capsule_pairs: &'a [Vec<GpuArticulatedConvexCapsulePair>],
    /// Convex polyhedra on distinct links, including boxes represented as hulls.
    pub convex_pairs: &'a [Vec<GpuArticulatedConvexPair>],
    /// Link convex hulls against stationary world-space convex hulls.
    pub static_convex_pairs: &'a [Vec<GpuArticulatedStaticConvexPair>],
    /// Stationary world-space convex hulls against moving link spheres.
    pub scene_convex_sphere_pairs: &'a [Vec<GpuArticulatedSceneConvexSpherePair>],
    /// Stationary indexed meshes against moving link spheres.
    pub scene_mesh_sphere_pairs: &'a [Vec<GpuArticulatedSceneMeshSpherePair>],
    /// Stationary indexed polylines against moving link spheres.
    pub scene_polyline_sphere_pairs: &'a [Vec<GpuArticulatedScenePolylineSpherePair>],
    /// Stationary indexed meshes against moving link capsules.
    pub scene_mesh_capsule_pairs: &'a [Vec<GpuArticulatedSceneMeshCapsulePair>],
    /// Stationary indexed polylines against moving link capsules.
    pub scene_polyline_capsule_pairs: &'a [Vec<GpuArticulatedScenePolylineCapsulePair>],
    /// Stationary indexed meshes against moving link boxes.
    pub scene_mesh_box_pairs: &'a [Vec<GpuArticulatedSceneMeshBoxPair>],
    /// Stationary indexed polylines against moving link boxes.
    pub scene_polyline_box_pairs: &'a [Vec<GpuArticulatedScenePolylineBoxPair>],
    /// Stationary indexed meshes against moving link cylinders and cones.
    pub scene_mesh_axial_pairs: &'a [Vec<GpuArticulatedSceneMeshAxialPair>],
    /// Stationary indexed polylines against moving link cylinders and cones.
    pub scene_polyline_axial_pairs: &'a [Vec<GpuArticulatedScenePolylineAxialPair>],
    /// Static indexed meshes paired with link convex hulls.
    pub scene_mesh_convex_pairs: &'a [Vec<GpuArticulatedSceneMeshConvexPair>],
    /// Stationary indexed polylines against moving link convex hulls.
    pub scene_polyline_convex_pairs: &'a [Vec<GpuArticulatedScenePolylineConvexPair>],
    /// Stationary world-space convex hulls against moving link capsules.
    pub scene_convex_capsule_pairs: &'a [Vec<GpuArticulatedSceneConvexCapsulePair>],
    /// Link spheres whose eligible pairs are populated from GPU LBVH candidates.
    pub dynamic_spheres: &'a [Vec<GpuArticulatedLinkSphere>],
    /// Link capsules whose eligible sphere contacts come from GPU LBVH candidates.
    pub dynamic_capsules: &'a [Vec<GpuArticulatedLinkCapsule>],
    /// Link boxes whose eligible sphere contacts come from GPU LBVH candidates.
    pub dynamic_boxes: &'a [Vec<GpuArticulatedLinkBox>],
    /// Combination rules in sphere, capsule, then box order. Empty keeps legacy GPU rules.
    pub dynamic_material_rules: &'a [Vec<GpuArticulatedCombineRules>],
    /// Capsule-sphere pairs for each environment.
    pub capsule_sphere_pairs: &'a [Vec<GpuArticulatedCapsuleSpherePair>],
    /// Explicit capsule-capsule pairs for each environment.
    pub capsule_pairs: &'a [Vec<GpuArticulatedCapsulePair>],
    /// Sphere-box pairs for each environment.
    pub sphere_box_pairs: &'a [Vec<GpuArticulatedSphereBoxPair>],
    /// Capsule-box pairs for each environment.
    pub capsule_box_pairs: &'a [Vec<GpuArticulatedCapsuleBoxPair>],
    /// Box-box pairs for each environment.
    pub box_pairs: &'a [Vec<GpuArticulatedBoxPair>],
    /// Projected solver sweeps for each environment, in `1..=64`.
    pub iterations: &'a [u32],
    /// Whether each environment reuses a damped contact impulse from the previous step.
    pub warm_start: &'a [bool],
}

/// Expand link spheres into candidate pairs, excluding same-link and configured link exclusions.
pub fn self_contact_sphere_pairs(
    articulation: &Articulation,
    spheres: &[GpuArticulatedLinkSphere],
) -> Result<Vec<GpuArticulatedSpherePair>, GpuArticulatedGroundContactError> {
    for sphere in spheres {
        validate_link_sphere(articulation, sphere)?;
    }
    let mut pairs = Vec::new();
    for (first_index, first) in spheres.iter().enumerate() {
        for second in &spheres[first_index + 1..] {
            if first.link == second.link || articulation.adjacent(first.link, second.link) {
                continue;
            }
            pairs.push(GpuArticulatedSpherePair {
                first_link: first.link,
                first_local_center: first.local_center,
                first_radius: first.radius,
                second_link: second.link,
                second_local_center: second.local_center,
                second_radius: second.radius,
                restitution: first.restitution.max(second.restitution),
                friction: first.friction.sqrt() * second.friction.sqrt(),
            });
        }
    }
    Ok(pairs)
}

/// Expand link capsules into nonexcluded capsule-capsule candidates.
pub fn self_contact_capsule_pairs(
    articulation: &Articulation,
    capsules: &[GpuArticulatedLinkCapsule],
) -> Result<Vec<GpuArticulatedCapsulePair>, GpuArticulatedGroundContactError> {
    for capsule in capsules {
        validate_link_capsule(articulation, capsule)?;
    }
    let mut pairs = Vec::new();
    for (first_index, first) in capsules.iter().enumerate() {
        for second in &capsules[first_index + 1..] {
            if first.link == second.link || articulation.adjacent(first.link, second.link) {
                continue;
            }
            pairs.push(GpuArticulatedCapsulePair {
                first_link: first.link,
                first_local_a: first.local_a,
                first_local_b: first.local_b,
                first_radius: first.radius,
                second_link: second.link,
                second_local_a: second.local_a,
                second_local_b: second.local_b,
                second_radius: second.radius,
                restitution: first.restitution.max(second.restitution),
                friction: first.friction.sqrt() * second.friction.sqrt(),
            });
        }
    }
    Ok(pairs)
}

/// Expand link capsules and spheres into nonexcluded capsule-sphere candidates.
pub fn self_contact_capsule_sphere_pairs(
    articulation: &Articulation,
    capsules: &[GpuArticulatedLinkCapsule],
    spheres: &[GpuArticulatedLinkSphere],
) -> Result<Vec<GpuArticulatedCapsuleSpherePair>, GpuArticulatedGroundContactError> {
    for capsule in capsules {
        validate_link_capsule(articulation, capsule)?;
    }
    for sphere in spheres {
        validate_link_sphere(articulation, sphere)?;
    }
    let mut pairs = Vec::new();
    for capsule in capsules {
        for sphere in spheres {
            if capsule.link == sphere.link || articulation.adjacent(capsule.link, sphere.link) {
                continue;
            }
            pairs.push(GpuArticulatedCapsuleSpherePair {
                capsule_link: capsule.link,
                capsule_local_a: capsule.local_a,
                capsule_local_b: capsule.local_b,
                capsule_radius: capsule.radius,
                sphere_link: sphere.link,
                sphere_local_center: sphere.local_center,
                sphere_radius: sphere.radius,
                restitution: capsule.restitution.max(sphere.restitution),
                friction: capsule.friction.sqrt() * sphere.friction.sqrt(),
            });
        }
    }
    Ok(pairs)
}

/// Expand link spheres and boxes into nonexcluded sphere-box candidates.
pub fn self_contact_sphere_box_pairs(
    articulation: &Articulation,
    spheres: &[GpuArticulatedLinkSphere],
    boxes: &[GpuArticulatedLinkBox],
) -> Result<Vec<GpuArticulatedSphereBoxPair>, GpuArticulatedGroundContactError> {
    for sphere in spheres {
        validate_link_sphere(articulation, sphere)?;
    }
    for box_shape in boxes {
        validate_link_box(articulation, box_shape)?;
    }
    let mut pairs = Vec::new();
    for sphere in spheres {
        for box_shape in boxes {
            if sphere.link == box_shape.link || articulation.adjacent(sphere.link, box_shape.link) {
                continue;
            }
            pairs.push(GpuArticulatedSphereBoxPair {
                sphere_link: sphere.link,
                sphere_local_center: sphere.local_center,
                sphere_radius: sphere.radius,
                box_link: box_shape.link,
                box_local_pose: box_shape.local_pose,
                box_half_extents: box_shape.half_extents,
                restitution: sphere.restitution.max(box_shape.restitution),
                friction: sphere.friction.sqrt() * box_shape.friction.sqrt(),
            });
        }
    }
    Ok(pairs)
}

/// Expand link capsules and boxes into nonexcluded capsule-box candidates.
pub fn self_contact_capsule_box_pairs(
    articulation: &Articulation,
    capsules: &[GpuArticulatedLinkCapsule],
    boxes: &[GpuArticulatedLinkBox],
) -> Result<Vec<GpuArticulatedCapsuleBoxPair>, GpuArticulatedGroundContactError> {
    for capsule in capsules {
        validate_link_capsule(articulation, capsule)?;
    }
    for box_shape in boxes {
        validate_link_box(articulation, box_shape)?;
    }
    let mut pairs = Vec::new();
    for capsule in capsules {
        for box_shape in boxes {
            if capsule.link == box_shape.link || articulation.adjacent(capsule.link, box_shape.link)
            {
                continue;
            }
            pairs.push(GpuArticulatedCapsuleBoxPair {
                capsule_link: capsule.link,
                capsule_local_a: capsule.local_a,
                capsule_local_b: capsule.local_b,
                capsule_radius: capsule.radius,
                box_link: box_shape.link,
                box_local_pose: box_shape.local_pose,
                box_half_extents: box_shape.half_extents,
                restitution: capsule.restitution.max(box_shape.restitution),
                friction: capsule.friction.sqrt() * box_shape.friction.sqrt(),
            });
        }
    }
    Ok(pairs)
}

/// Expand link boxes into nonexcluded box-box candidates.
pub fn self_contact_box_pairs(
    articulation: &Articulation,
    boxes: &[GpuArticulatedLinkBox],
) -> Result<Vec<GpuArticulatedBoxPair>, GpuArticulatedGroundContactError> {
    for box_shape in boxes {
        validate_link_box(articulation, box_shape)?;
    }
    let mut pairs = Vec::new();
    for (first_index, first) in boxes.iter().enumerate() {
        for second in &boxes[first_index + 1..] {
            if first.link == second.link || articulation.adjacent(first.link, second.link) {
                continue;
            }
            pairs.push(GpuArticulatedBoxPair {
                first_link: first.link,
                first_local_pose: first.local_pose,
                first_half_extents: first.half_extents,
                second_link: second.link,
                second_local_pose: second.local_pose,
                second_half_extents: second.half_extents,
                restitution: first.restitution.max(second.restitution),
                friction: first.friction.sqrt() * second.friction.sqrt(),
            });
        }
    }
    Ok(pairs)
}

fn validate_link_sphere(
    articulation: &Articulation,
    sphere: &GpuArticulatedLinkSphere,
) -> Result<(), GpuArticulatedGroundContactError> {
    if articulation.link(sphere.link).is_none()
        || sphere.radius <= 0.0
        || !(0.0..=1.0).contains(&sphere.restitution)
        || sphere.friction < 0.0
    {
        return Err(GpuArticulatedGroundContactError::InvalidInput);
    }
    for value in [
        sphere.local_center.x,
        sphere.local_center.y,
        sphere.local_center.z,
        sphere.radius,
        sphere.restitution,
        sphere.friction,
    ] {
        let _ = finite_f32(value)?;
    }
    Ok(())
}

fn same_sphere_shapes(
    pair: &GpuArticulatedSpherePair,
    first: &GpuArticulatedLinkSphere,
    second: &GpuArticulatedLinkSphere,
) -> bool {
    (pair.first_link == first.link
        && pair.second_link == second.link
        && pair.first_local_center == first.local_center
        && pair.second_local_center == second.local_center
        && pair.first_radius == first.radius
        && pair.second_radius == second.radius)
        || (pair.first_link == second.link
            && pair.second_link == first.link
            && pair.first_local_center == second.local_center
            && pair.second_local_center == first.local_center
            && pair.first_radius == second.radius
            && pair.second_radius == first.radius)
}

fn same_capsule_sphere_shapes(
    pair: &GpuArticulatedCapsuleSpherePair,
    capsule: &GpuArticulatedLinkCapsule,
    sphere: &GpuArticulatedLinkSphere,
) -> bool {
    pair.capsule_link == capsule.link
        && pair.sphere_link == sphere.link
        && ((pair.capsule_local_a == capsule.local_a && pair.capsule_local_b == capsule.local_b)
            || (pair.capsule_local_a == capsule.local_b && pair.capsule_local_b == capsule.local_a))
        && pair.capsule_radius == capsule.radius
        && pair.sphere_local_center == sphere.local_center
        && pair.sphere_radius == sphere.radius
}

fn same_capsule_pair_shapes(
    pair: &GpuArticulatedCapsulePair,
    first: &GpuArticulatedLinkCapsule,
    second: &GpuArticulatedLinkCapsule,
) -> bool {
    let axis_matches = |a: Vector3<f64>, b: Vector3<f64>, capsule: &GpuArticulatedLinkCapsule| {
        (a == capsule.local_a && b == capsule.local_b)
            || (a == capsule.local_b && b == capsule.local_a)
    };
    (pair.first_link == first.link
        && pair.second_link == second.link
        && axis_matches(pair.first_local_a, pair.first_local_b, first)
        && axis_matches(pair.second_local_a, pair.second_local_b, second)
        && pair.first_radius == first.radius
        && pair.second_radius == second.radius)
        || (pair.first_link == second.link
            && pair.second_link == first.link
            && axis_matches(pair.first_local_a, pair.first_local_b, second)
            && axis_matches(pair.second_local_a, pair.second_local_b, first)
            && pair.first_radius == second.radius
            && pair.second_radius == first.radius)
}

fn same_sphere_box_shapes(
    pair: &GpuArticulatedSphereBoxPair,
    sphere: &GpuArticulatedLinkSphere,
    box_shape: &GpuArticulatedLinkBox,
) -> bool {
    pair.sphere_link == sphere.link
        && pair.sphere_local_center == sphere.local_center
        && pair.sphere_radius == sphere.radius
        && pair.box_link == box_shape.link
        && pair.box_local_pose == box_shape.local_pose
        && pair.box_half_extents == box_shape.half_extents
}

fn same_capsule_box_shapes(
    pair: &GpuArticulatedCapsuleBoxPair,
    capsule: &GpuArticulatedLinkCapsule,
    box_shape: &GpuArticulatedLinkBox,
) -> bool {
    pair.capsule_link == capsule.link
        && ((pair.capsule_local_a == capsule.local_a && pair.capsule_local_b == capsule.local_b)
            || (pair.capsule_local_a == capsule.local_b && pair.capsule_local_b == capsule.local_a))
        && pair.capsule_radius == capsule.radius
        && pair.box_link == box_shape.link
        && pair.box_local_pose == box_shape.local_pose
        && pair.box_half_extents == box_shape.half_extents
}

fn same_box_pair_shapes(
    pair: &GpuArticulatedBoxPair,
    first: &GpuArticulatedLinkBox,
    second: &GpuArticulatedLinkBox,
) -> bool {
    (pair.first_link == first.link
        && pair.first_local_pose == first.local_pose
        && pair.first_half_extents == first.half_extents
        && pair.second_link == second.link
        && pair.second_local_pose == second.local_pose
        && pair.second_half_extents == second.half_extents)
        || (pair.first_link == second.link
            && pair.first_local_pose == second.local_pose
            && pair.first_half_extents == second.half_extents
            && pair.second_link == first.link
            && pair.second_local_pose == first.local_pose
            && pair.second_half_extents == first.half_extents)
}

pub(crate) fn validate_link_capsule(
    articulation: &Articulation,
    capsule: &GpuArticulatedLinkCapsule,
) -> Result<(), GpuArticulatedGroundContactError> {
    if articulation.link(capsule.link).is_none()
        || capsule.radius <= 0.0
        || !(0.0..=1.0).contains(&capsule.restitution)
        || capsule.friction < 0.0
    {
        return Err(GpuArticulatedGroundContactError::InvalidInput);
    }
    for value in [
        capsule.local_a.x,
        capsule.local_a.y,
        capsule.local_a.z,
        capsule.local_b.x,
        capsule.local_b.y,
        capsule.local_b.z,
        capsule.radius,
        capsule.restitution,
        capsule.friction,
    ] {
        let _ = finite_f32(value)?;
    }
    Ok(())
}

pub(crate) fn validate_link_box(
    articulation: &Articulation,
    box_shape: &GpuArticulatedLinkBox,
) -> Result<(), GpuArticulatedGroundContactError> {
    if articulation.link(box_shape.link).is_none()
        || box_shape.half_extents.iter().any(|value| *value <= 0.0)
        || !(0.0..=1.0).contains(&box_shape.restitution)
        || box_shape.friction < 0.0
    {
        return Err(GpuArticulatedGroundContactError::InvalidInput);
    }
    let translation = box_shape.local_pose.translation.vector;
    let rotation = box_shape.local_pose.rotation.quaternion();
    for value in [
        translation.x,
        translation.y,
        translation.z,
        rotation.i,
        rotation.j,
        rotation.k,
        rotation.w,
        box_shape.half_extents.x,
        box_shape.half_extents.y,
        box_shape.half_extents.z,
        box_shape.restitution,
        box_shape.friction,
    ] {
        let _ = finite_f32(value)?;
    }
    Ok(())
}

/// Invalid shape, incompatible layout, or exhausted GPU capacity.
#[derive(Debug, thiserror::Error)]
pub enum GpuArticulatedGroundContactError {
    /// A shape value or linked articulation layout is invalid.
    #[error("invalid articulated contact")]
    InvalidInput,
    /// Packed contact buffers exceed GPU limits.
    #[error("articulated contact exceeds GPU capacity")]
    Capacity,
    /// GPU readback failed.
    #[error("contact readback failed: {0}")]
    Readback(String),
    /// A source environment has a numerical fault.
    #[error("contact source fault in environment {0}")]
    SourceFault(usize),
    /// The GPU candidate dispatch exceeds device limits.
    #[error(transparent)]
    Lbvh(#[from] GpuLbvhError),
}

/// Contact wrench on one link, evaluated at the last solver step before integration.
#[derive(Debug, Clone)]
pub struct GpuArticulatedLinkContact {
    /// Environment-local link index.
    pub link: usize,
    /// World-space witness point used by the solver.
    pub position: Vector3<f64>,
    /// World-space normal pointing in the direction of the link's normal impulse.
    pub normal: Vector3<f64>,
    /// Average force in newtons, including friction.
    pub force: Vector3<f64>,
    /// Torque in newton-metres about the link origin during the solve.
    pub torque: Vector3<f64>,
    /// Signed geometric separation.
    pub distance: f64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PackedSystem {
    indices: [u32; 4],
    inverse: [u32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PackedMotionLink {
    indices: [u32; 4],
    center_of_mass: [f32; 4],
    thresholds: [f32; 4],
}

#[derive(Debug)]
struct MotionWake {
    pipeline: wgpu::ComputePipeline,
    metadata: wgpu::Buffer,
    link_count: u32,
}

#[derive(Debug)]
struct LoadWake {
    pipeline: wgpu::ComputePipeline,
    environments: wgpu::Buffer,
    loads: wgpu::Buffer,
    link_count: u32,
}

impl LoadWake {
    fn new(
        device: &wgpu::Device,
        loads: &wgpu::Buffer,
        ranges: &[Range<usize>],
    ) -> Result<Self, GpuArticulatedGroundContactError> {
        let owners = ranges
            .iter()
            .enumerate()
            .flat_map(|(environment, range)| core::iter::repeat_n(environment as u32, range.len()))
            .collect::<Vec<_>>();
        let link_count = checked_u32(owners.len())?;
        if link_count == 0
            || loads.size() != u64::from(link_count) * 32
            || link_count.div_ceil(64) > device.limits().max_compute_workgroups_per_dimension
        {
            return Err(GpuArticulatedGroundContactError::InvalidInput);
        }
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera articulated external load wake"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_articulated_load_wake.wgsl").into()),
        });
        Ok(Self {
            pipeline: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Tessera articulated external load wake"),
                layout: None,
                module: &shader,
                entry_point: Some("request_load_wake"),
                compilation_options: Default::default(),
                cache: None,
            }),
            environments: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tessera articulated load wake environments"),
                contents: bytemuck::cast_slice(&owners),
                usage: wgpu::BufferUsages::STORAGE,
            }),
            loads: loads.clone(),
            link_count,
        })
    }

    fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        status: &wgpu::Buffer,
        requests: &wgpu::Buffer,
        mass_status: &wgpu::Buffer,
    ) {
        let buffers = [
            &self.loads,
            &self.environments,
            status,
            requests,
            mass_status,
        ];
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera articulated load wake bindings"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &buffers
                .iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Tessera articulated persistent load wake"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(self.link_count.div_ceil(64), 1, 1);
    }
}

#[derive(Debug)]
struct GravityWake {
    detect: wgpu::ComputePipeline,
    broadcast: wgpu::ComputePipeline,
    gravity: wgpu::Buffer,
    previous: wgpu::Buffer,
    changed: wgpu::Buffer,
    links: wgpu::Buffer,
    environment_count: u32,
    link_count: u32,
}

impl GravityWake {
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        gravity: &wgpu::Buffer,
        links: &[PackedMotionLink],
        environment_count: usize,
    ) -> Result<Self, GpuArticulatedGroundContactError> {
        let environment_count = checked_u32(environment_count)?;
        let link_count = checked_u32(links.len())?;
        let bytes = size_of_val(links) as u64;
        if gravity.size() != u64::from(environment_count) * 16
            || !gravity.usage().contains(wgpu::BufferUsages::COPY_SRC)
            || links
                .iter()
                .any(|link| link.indices[0] >= environment_count)
        {
            return Err(GpuArticulatedGroundContactError::InvalidInput);
        }
        if bytes > u64::from(device.limits().max_storage_buffer_binding_size)
            || bytes > device.limits().max_buffer_size
            || environment_count.max(link_count).div_ceil(64)
                > device.limits().max_compute_workgroups_per_dimension
        {
            return Err(GpuArticulatedGroundContactError::Capacity);
        }
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera gravity change wake"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("gpu_articulated_gravity_wake.wgsl").into(),
            ),
        });
        let pipeline = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Tessera gravity change wake"),
                layout: None,
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let previous = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera previous gravity"),
            size: gravity.size(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(gravity, 0, &previous, 0, gravity.size());
        let _ = queue.submit(Some(encoder.finish()));
        Ok(Self {
            detect: pipeline("detect"),
            broadcast: pipeline("broadcast"),
            gravity: gravity.clone(),
            previous,
            changed: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tessera gravity change flags"),
                contents: bytemuck::cast_slice(&vec![0u32; environment_count as usize]),
                usage: wgpu::BufferUsages::STORAGE,
            }),
            links: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tessera gravity wake links"),
                contents: bytemuck::cast_slice(links),
                usage: wgpu::BufferUsages::STORAGE,
            }),
            environment_count,
            link_count,
        })
    }

    fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        status: &wgpu::Buffer,
        mass_status: &wgpu::Buffer,
        requests: &wgpu::Buffer,
    ) {
        for (pipeline, bindings, count) in [
            (
                &self.detect,
                vec![
                    (0, &self.gravity),
                    (1, &self.previous),
                    (2, &self.changed),
                    (3, status),
                    (4, mass_status),
                ],
                self.environment_count,
            ),
            (
                &self.broadcast,
                vec![(2, &self.changed), (5, &self.links), (6, requests)],
                self.link_count,
            ),
        ] {
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Tessera gravity wake bindings"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &bindings
                    .into_iter()
                    .map(|(binding, buffer)| wgpu::BindGroupEntry {
                        binding,
                        resource: buffer.as_entire_binding(),
                    })
                    .collect::<Vec<_>>(),
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Tessera gravity wake"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
        }
    }
}

impl MotionWake {
    fn new(
        device: &wgpu::Device,
        links: &[PackedMotionLink],
    ) -> Result<Self, GpuArticulatedGroundContactError> {
        Self::build(
            device,
            links,
            include_str!("gpu_articulated_motion_wake.wgsl"),
            "request_motion_wake",
        )
    }

    fn build(
        device: &wgpu::Device,
        links: &[PackedMotionLink],
        source: &str,
        entry: &str,
    ) -> Result<Self, GpuArticulatedGroundContactError> {
        let link_count = checked_u32(links.len())?;
        let bytes = size_of_val(links) as u64;
        if link_count.div_ceil(64) > device.limits().max_compute_workgroups_per_dimension
            || bytes > device.limits().max_storage_buffer_binding_size as u64
            || bytes > device.limits().max_buffer_size
        {
            return Err(GpuArticulatedGroundContactError::Capacity);
        }
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera articulated motion wake"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        Ok(Self {
            pipeline: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Tessera articulated motion wake"),
                layout: None,
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            }),
            metadata: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tessera articulated motion wake metadata"),
                contents: bytemuck::cast_slice(links),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }),
            link_count,
        })
    }

    fn encode<const N: usize>(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        inputs: [&wgpu::Buffer; N],
    ) {
        let buffers = core::iter::once(&self.metadata).chain(inputs);
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera articulated motion wake bindings"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &buffers
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Tessera articulated predicted motion wake"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(self.link_count.div_ceil(64), 1, 1);
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PackedSphere {
    indices: [u32; 4],
    center_radius: [f32; 4],
    center_of_mass: [f32; 4],
    plane: [f32; 4],
    material: [f32; 4],
    other_center_radius: [f32; 4],
    other_center_of_mass: [f32; 4],
    second_axis_end: [f32; 4],
    first_axis_end: [f32; 4],
    impulses: [f32; 4],
    previous_normal: [f32; 4],
    diagnostic_first: [f32; 4],
    diagnostic_second: [f32; 4],
    diagnostic_first_origin: [f32; 4],
    diagnostic_second_origin: [f32; 4],
    prescribed_linear: [f32; 4],
    prescribed_angular: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PackedDynamicShape {
    indices: [u32; 4],
    layout: [u32; 4],
    center_radius: [f32; 4],
    center_of_mass: [f32; 4],
    material: [f32; 4],
    axis_end: [f32; 4],
    orientation: [f32; 4],
    counts: [u32; 4],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PackedShapePair {
    a: u32,
    b: u32,
}

#[derive(Debug)]
struct CouplingPreparation {
    pipeline: wgpu::ComputePipeline,
    indices: wgpu::Buffer,
    positions: wgpu::Buffer,
    count: usize,
}

#[derive(Debug)]
struct ContactActivity {
    pipeline: wgpu::ComputePipeline,
    initialize: wgpu::ComputePipeline,
    reduce_wake: wgpu::ComputePipeline,
    broadcast_wake: wgpu::ComputePipeline,
    track_geometry: wgpu::ComputePipeline,
    idle_enabled: bool,
    idle_guard: wgpu::ComputePipeline,
    idle_environments: wgpu::Buffer,
    idle_sources: Option<[wgpu::Buffer; 2]>,
    idle_policy: wgpu::Buffer,
    idle_pipelines: [wgpu::ComputePipeline; 4],
    idle_parameters: wgpu::Buffer,
    idle_time: wgpu::Buffer,
    component_idle: wgpu::Buffer,
    idle_flags: wgpu::Buffer,
    sleep_candidates: wgpu::Buffer,
    owners: wgpu::Buffer,
    flags: wgpu::Buffer,
    parents: wgpu::Buffer,
    mobility_roots: wgpu::Buffer,
    seed_parents: Vec<u32>,
    wake_requests: wgpu::Buffer,
    component_wake: wgpu::Buffer,
    link_wake: wgpu::Buffer,
    previous_geometry: wgpu::Buffer,
    loss_params: wgpu::Buffer,
    support_gravity: wgpu::Buffer,
    row_count: u32,
}

impl ContactActivity {
    fn new(
        device: &wgpu::Device,
        contacts: &[Range<usize>],
        links: &[Range<usize>],
    ) -> Result<Self, GpuArticulatedGroundContactError> {
        let mut owners = Vec::<[u32; 4]>::new();
        for (environment, (rows, bounds)) in contacts.iter().zip(links).enumerate() {
            for row in rows.clone() {
                owners.push([
                    checked_u32(row)?,
                    checked_u32(environment)?,
                    checked_u32(bounds.start)?,
                    checked_u32(bounds.end)?,
                ]);
            }
        }
        let row_count = checked_u32(owners.len())?;
        if row_count.div_ceil(64) > device.limits().max_compute_workgroups_per_dimension {
            return Err(GpuArticulatedGroundContactError::Capacity);
        }
        if owners.is_empty() {
            owners.push([0; 4]);
        }
        let link_count = links.last().map_or(0, |range| range.end);
        let seed_parents = (0..checked_u32(link_count.max(1))?).collect::<Vec<_>>();
        if seed_parents.len().div_ceil(64)
            > device.limits().max_compute_workgroups_per_dimension as usize
        {
            return Err(GpuArticulatedGroundContactError::Capacity);
        }
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera articulated contact activity"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("gpu_articulated_contact_activity.wgsl").into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tessera articulated contact activity"),
            layout: None,
            module: &shader,
            entry_point: Some("collect_activity"),
            compilation_options: Default::default(),
            cache: None,
        });
        let initialize = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tessera articulated contact component initialization"),
            layout: None,
            module: &shader,
            entry_point: Some("initialize_components"),
            compilation_options: Default::default(),
            cache: None,
        });
        let wake_pipeline = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Tessera articulated component wake"),
                layout: None,
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let wake_buffer = |label| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&vec![0u32; seed_parents.len()]),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            })
        };
        if (seed_parents.len() as u64) * 16
            > u64::from(device.limits().max_storage_buffer_binding_size)
            || (seed_parents.len() as u64) * 16 > device.limits().max_buffer_size
        {
            return Err(GpuArticulatedGroundContactError::Capacity);
        }
        let idle_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera articulated component idle"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_articulated_sleep.wgsl").into()),
        });
        let idle_pipelines = ["initialize", "reduce", "reduce_ready", "finish"].map(|entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Tessera articulated component idle"),
                layout: None,
                module: &idle_shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        Ok(Self {
            idle_enabled: false,
            idle_guard: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Tessera articulated idle source guard"),
                layout: None,
                module: &idle_shader,
                entry_point: Some("guard_sources"),
                compilation_options: Default::default(),
                cache: None,
            }),
            idle_environments: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tessera articulated idle environments"),
                contents: bytemuck::cast_slice(
                    &links
                        .iter()
                        .enumerate()
                        .flat_map(|(env, links)| core::iter::repeat_n(env as u32, links.len()))
                        .collect::<Vec<_>>(),
                ),
                usage: wgpu::BufferUsages::STORAGE,
            }),
            idle_sources: None,
            idle_policy: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tessera component idle support policy"),
                contents: bytemuck::cast_slice(&[2u32, 0, 0, 0]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }),
            idle_pipelines,
            idle_parameters: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tessera articulated idle parameters"),
                contents: bytemuck::cast_slice(&vec![[0.0f32; 4]; seed_parents.len()]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }),
            idle_time: wake_buffer("Tessera articulated idle time"),
            component_idle: wake_buffer("Tessera articulated component idle time"),
            idle_flags: wake_buffer("Tessera articulated component idle flags"),
            sleep_candidates: wake_buffer("Tessera articulated sleep candidates"),
            pipeline,
            initialize,
            reduce_wake: wake_pipeline("reduce_wake_requests"),
            broadcast_wake: wake_pipeline("broadcast_wake_requests"),
            track_geometry: wake_pipeline("track_geometric_contact"),
            wake_requests: wake_buffer("Tessera articulated pending wake requests"),
            component_wake: wake_buffer("Tessera articulated component wake flags"),
            link_wake: wake_buffer("Tessera articulated link wake flags"),
            previous_geometry: wake_buffer("Tessera articulated previous geometric contact"),
            support_gravity: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tessera contact support gravity"),
                contents: bytemuck::cast_slice(&vec![[0.0f32; 4]; links.len().max(1)]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }),
            loss_params: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tessera articulated contact loss parameters"),
                contents: bytemuck::cast_slice(&[0u32; 4]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }),
            owners: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tessera articulated contact activity owners"),
                contents: bytemuck::cast_slice(&owners),
                usage: wgpu::BufferUsages::STORAGE,
            }),
            flags: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tessera articulated contact activity flags"),
                contents: bytemuck::cast_slice(&vec![0u32; link_count.max(1)]),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            }),
            parents: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tessera articulated contact component parents"),
                contents: bytemuck::cast_slice(&seed_parents),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            }),
            mobility_roots: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tessera articulated mobility roots"),
                contents: bytemuck::cast_slice(&seed_parents),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }),
            seed_parents,
            row_count,
        })
    }

    fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        rows: &wgpu::Buffer,
        status: &wgpu::Buffer,
    ) {
        encoder.clear_buffer(&self.flags, 0, None);
        let initialize_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera articulated component initialization bindings"),
            layout: &self.initialize.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.parents.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.mobility_roots.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Tessera articulated component initialization"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.initialize);
            pass.set_bind_group(0, &initialize_group, &[]);
            pass.dispatch_workgroups((self.parents.size() / 4).div_ceil(64) as u32, 1, 1);
        }
        if self.row_count == 0 {
            self.encode_wake(device, encoder);
            return;
        }
        let buffers = [rows, &self.owners, status, &self.flags, &self.parents];
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera articulated contact activity bindings"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &buffers
                .iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .chain(core::iter::once(wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.wake_requests.as_entire_binding(),
                }))
                .chain(core::iter::once(wgpu::BindGroupEntry {
                    binding: 11,
                    resource: self.support_gravity.as_entire_binding(),
                }))
                .collect::<Vec<_>>(),
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Tessera articulated contact activity"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(self.row_count.div_ceil(64), 1, 1);
        drop(pass);
        self.encode_wake(device, encoder);
    }

    fn encode_wake(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        let geometry_buffers = [
            (3, &self.flags),
            (6, &self.wake_requests),
            (9, &self.previous_geometry),
            (10, &self.loss_params),
        ];
        let geometry_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera articulated geometric contact history bindings"),
            layout: &self.track_geometry.get_bind_group_layout(0),
            entries: &geometry_buffers
                .iter()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: *binding,
                    resource: buffer.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Tessera articulated geometric contact loss"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.track_geometry);
            pass.set_bind_group(0, &geometry_group, &[]);
            pass.dispatch_workgroups((self.parents.size() / 4).div_ceil(64) as u32, 1, 1);
        }
        encoder.clear_buffer(&self.component_wake, 0, None);
        let reduce = [
            (4, &self.parents),
            (6, &self.wake_requests),
            (7, &self.component_wake),
        ];
        let broadcast = [
            (4, &self.parents),
            (7, &self.component_wake),
            (8, &self.link_wake),
        ];
        for (pipeline, buffers) in [
            (&self.reduce_wake, reduce),
            (&self.broadcast_wake, broadcast),
        ] {
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Tessera articulated component wake bindings"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &buffers
                    .iter()
                    .map(|(binding, buffer)| wgpu::BindGroupEntry {
                        binding: *binding,
                        resource: buffer.as_entire_binding(),
                    })
                    .collect::<Vec<_>>(),
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Tessera articulated component wake propagation"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups((self.parents.size() / 4).div_ceil(64) as u32, 1, 1);
        }
        if self.idle_enabled {
            self.encode_idle(device, encoder);
        }
    }

    fn encode_idle(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        if let Some(sources) = &self.idle_sources {
            let buffers = [
                &self.idle_environments,
                &sources[0],
                &sources[1],
                &self.link_wake,
            ];
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Tessera articulated idle source guard bindings"),
                layout: &self.idle_guard.get_bind_group_layout(0),
                entries: &buffers
                    .iter()
                    .enumerate()
                    .map(|(i, buffer)| wgpu::BindGroupEntry {
                        binding: 8 + i as u32,
                        resource: buffer.as_entire_binding(),
                    })
                    .collect::<Vec<_>>(),
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Tessera articulated idle source guard"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.idle_guard);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups((self.parents.size() / 4).div_ceil(64) as u32, 1, 1);
        }
        let buffers = [
            &self.idle_parameters,
            &self.parents,
            &self.flags,
            &self.link_wake,
            &self.idle_time,
            &self.component_idle,
            &self.idle_flags,
            &self.sleep_candidates,
            &self.idle_policy,
        ];
        for (pipeline, bindings) in self.idle_pipelines.iter().zip([
            vec![5usize, 6],
            vec![0, 1, 2, 3, 4, 5, 6, 8],
            vec![0, 1, 5, 6],
            vec![0, 1, 4, 5, 6, 7],
        ]) {
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Tessera articulated idle bindings"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &bindings
                    .into_iter()
                    .map(|index| wgpu::BindGroupEntry {
                        binding: if index == 8 { 12 } else { index as u32 },
                        resource: buffers[index].as_entire_binding(),
                    })
                    .collect::<Vec<_>>(),
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Tessera articulated idle evaluation"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups((self.parents.size() / 4).div_ceil(64) as u32, 1, 1);
        }
    }

    fn clear_wake(&self, queue: &wgpu::Queue) {
        let zeros = vec![0u8; self.parents.size() as usize];
        for buffer in [
            &self.wake_requests,
            &self.component_wake,
            &self.link_wake,
            &self.idle_time,
            &self.component_idle,
            &self.idle_flags,
            &self.sleep_candidates,
        ] {
            queue.write_buffer(buffer, 0, &zeros);
        }
    }
}

/// Reserved link geometry paired with an externally prescribed world-space box.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuArticulatedBoxContactKind {
    /// Link spheres against world boxes.
    Sphere,
    /// Link capsules against world boxes; includes endpoint and side rows.
    Capsule,
    /// Link boxes against world boxes; includes all manifold rows.
    Box,
    /// Link cylinders and cones against world boxes; excludes static axial shapes.
    Axial,
    /// Link convex hulls against world boxes represented as convex hulls.
    Convex,
}

/// Link geometry paired with a prescribed world cylinder or cone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuArticulatedAxialContactKind {
    /// Link spheres.
    Sphere,
    /// Link capsules.
    Capsule,
    /// Link boxes.
    Box,
    /// Link cylinders and cones.
    Axial,
    /// Link convex hulls.
    Convex,
}

impl GpuArticulatedAxialContactKind {
    fn index(self) -> usize {
        match self {
            Self::Sphere => 0,
            Self::Capsule => 1,
            Self::Box => 2,
            Self::Axial => 3,
            Self::Convex => 4,
        }
    }
    fn accepts(self, mode: f32) -> bool {
        match self {
            Self::Sphere => matches!(mode, 60.0 | 61.0),
            Self::Capsule => matches!(mode, 62.0 | 63.0),
            Self::Box => matches!(mode, 64.0 | 65.0),
            Self::Axial => matches!(mode, 66.0 | 67.0 | 68.0 | 69.0),
            Self::Convex => matches!(mode, 70.0 | 71.0),
        }
    }
}

/// Link geometry paired with an externally prescribed world-space capsule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuArticulatedCapsuleContactKind {
    /// Link spheres against world capsules.
    Sphere,
    /// Link capsules against world capsules; endpoint and side rows.
    Capsule,
    /// Link boxes against world capsules; endpoint and side rows.
    Box,
    /// Link cylinders and cones against world capsules.
    Axial,
    /// Link convex hulls against world capsules; includes both face-manifold rows.
    Convex,
}

impl GpuArticulatedCapsuleContactKind {
    fn index(self) -> usize {
        match self {
            Self::Sphere => 0,
            Self::Capsule => 1,
            Self::Box => 2,
            Self::Axial => 3,
            Self::Convex => 4,
        }
    }
    fn row_count(self) -> usize {
        match self {
            Self::Capsule | Self::Box | Self::Convex => 2,
            _ => 1,
        }
    }
    fn accepts(self, mode: f32) -> bool {
        match self {
            Self::Sphere => mode == 17.0,
            Self::Capsule => mode == 18.0,
            Self::Box => mode == 24.0,
            Self::Axial => mode == 28.0 || mode == 29.0,
            Self::Convex => mode == 46.0,
        }
    }
}

fn prescribed_capsule_endpoints(row: &PackedSphere) -> [Vector3<f64>; 2] {
    let (a, b) = match row.material[3] {
        17.0 => (row.plane, row.other_center_radius),
        24.0 | 25.0 => (row.center_radius, row.first_axis_end),
        28.0 | 29.0 => (row.plane, row.first_axis_end),
        _ => (row.plane, row.second_axis_end),
    };
    [
        Vector3::new(a[0] as f64, a[1] as f64, a[2] as f64),
        Vector3::new(b[0] as f64, b[1] as f64, b[2] as f64),
    ]
}

/// Initial prescribed box-body states in one environment's reserved pair order.
/// Static boxes use None. Each capsule/box pair addresses every manifold row.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GpuArticulatedExternalBoxBodies {
    /// Link-sphere/world-box pairs.
    pub spheres: Vec<Option<crate::gpu_kinematic_body::GpuKinematicBody>>,
    /// Link-capsule/world-box pairs.
    pub capsules: Vec<Option<crate::gpu_kinematic_body::GpuKinematicBody>>,
    /// Link-box/world-box pairs.
    pub boxes: Vec<Option<crate::gpu_kinematic_body::GpuKinematicBody>>,
    /// Link-cylinder/cone pairs with world boxes; excludes static axial shapes.
    pub axial: Vec<Option<crate::gpu_kinematic_body::GpuKinematicBody>>,
    /// Link-convex/world-hull pairs. Non-prescribed world hulls use None.
    pub convex: Vec<Option<crate::gpu_kinematic_body::GpuKinematicBody>>,
}

/// Initial prescribed world-convex states in reserved contact-pair order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GpuArticulatedExternalConvexBodies {
    /// World-convex/sphere pairs followed by world-convex/capsule pairs.
    pub rounded: Vec<Option<crate::gpu_kinematic_body::GpuKinematicBody>>,
    /// Static polyhedron pairs (including box proxies) followed by axial/convex pairs.
    /// Use None for pairs whose external shape is stationary or is not a convex hull.
    pub polyhedron_axial: Vec<Option<crate::gpu_kinematic_body::GpuKinematicBody>>,
}

/// Initial prescribed capsule-body states in one environment's reserved pair order.
/// Static capsules use None. Each pair addresses all of its reserved contact rows.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GpuArticulatedExternalCapsuleBodies {
    /// Link-sphere/world-capsule pairs.
    pub spheres: Vec<Option<crate::gpu_kinematic_body::GpuKinematicBody>>,
    /// Link-capsule/world-capsule pairs.
    pub capsules: Vec<Option<crate::gpu_kinematic_body::GpuKinematicBody>>,
    /// Link-box/world-capsule pairs.
    pub boxes: Vec<Option<crate::gpu_kinematic_body::GpuKinematicBody>>,
    /// Link-cylinder/cone pairs with world capsules; excludes static axial shapes.
    pub axial: Vec<Option<crate::gpu_kinematic_body::GpuKinematicBody>>,
    /// Link-convex/world-capsule pairs.
    pub convex: Vec<Option<crate::gpu_kinematic_body::GpuKinematicBody>>,
}

/// Initial prescribed cylinder/cone-body states in one environment's reserved pair order.
/// Static cylinders and cones use None. Each pair addresses all of its reserved contact rows.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GpuArticulatedExternalAxialBodies {
    /// Link-sphere/world-cylinder/cone pairs.
    pub spheres: Vec<Option<crate::gpu_kinematic_body::GpuKinematicBody>>,
    /// Link-capsule/world-cylinder/cone pairs.
    pub capsules: Vec<Option<crate::gpu_kinematic_body::GpuKinematicBody>>,
    /// Link-box/world-cylinder/cone pairs.
    pub boxes: Vec<Option<crate::gpu_kinematic_body::GpuKinematicBody>>,
    /// Link-cylinder/cone pairs with a stationary first axial shape.
    pub axial: Vec<Option<crate::gpu_kinematic_body::GpuKinematicBody>>,
    /// Link-convex/world-cylinder/cone pairs.
    pub convex: Vec<Option<crate::gpu_kinematic_body::GpuKinematicBody>>,
}

impl GpuArticulatedBoxContactKind {
    fn row_count(self) -> usize {
        match self {
            Self::Sphere | Self::Axial => 1,
            Self::Capsule => 2,
            Self::Box | Self::Convex => 4,
        }
    }
}

/// Projects sphere, capsule, and box contact impulses through the articulated inverse mass.
///
/// Each environment is solved serially in one GPU invocation with projected
/// Gauss-Seidel sweeps, so contacts on the same tree share updated generalized
/// velocities. Contact impulses are reset or damped and reused at the start of
/// each step, then accumulated across sweeps. The pass edits the acceleration vector after
/// `encode_with_inverse`; a subsequent integration
/// pass consumes the corrected acceleration. It handles normal impulses,
/// restitution, and Coulomb friction against static planes and paired links.
#[derive(Debug)]
pub struct GpuArticulatedGroundContactBatch {
    device: wgpu::Device,
    pipeline: wgpu::ComputePipeline,
    systems: wgpu::Buffer,
    spheres: wgpu::Buffer,
    static_sphere_rows: Vec<Vec<(usize, usize)>>,
    static_sphere_centers: Vec<Vec<[f32; 3]>>,
    static_sphere_motion: Vec<Vec<[f32; 6]>>,
    sphere_motion_pipeline: OnceLock<wgpu::ComputePipeline>,
    sphere_orbits: wgpu::Buffer,
    sphere_orbit_enabled: Vec<bool>,
    integrated_sphere_centers: AtomicBool,
    integrated_sphere_box_poses: AtomicBool,
    integrated_capsule_box_poses: AtomicBool,
    integrated_box_box_poses: AtomicBool,
    integrated_axial_box_poses: AtomicBool,
    integrated_prescribed_convex_poses: AtomicBool,
    integrated_axial_convex_poses: AtomicBool,
    integrated_convex_rounded_poses: AtomicBool,
    static_capsule_sphere_motion: Vec<Vec<[f32; 6]>>,
    integrated_capsule_sphere_centers: AtomicBool,
    static_box_sphere_motion: Vec<Vec<[f32; 6]>>,
    integrated_box_sphere_centers: AtomicBool,
    static_axial_sphere_motion: Vec<Vec<[f32; 6]>>,
    integrated_axial_sphere_centers: AtomicBool,
    static_convex_sphere_motion: Vec<Vec<[f32; 6]>>,
    integrated_convex_sphere_centers: AtomicBool,
    static_capsule_sphere_rows: Vec<Vec<(usize, usize)>>,
    static_capsule_sphere_centers: Vec<Vec<[f32; 3]>>,
    static_box_sphere_rows: Vec<Vec<(usize, usize)>>,
    static_box_sphere_centers: Vec<Vec<[f32; 3]>>,
    static_axial_sphere_rows: Vec<Vec<(usize, usize)>>,
    static_axial_sphere_centers: Vec<Vec<[f32; 3]>>,
    static_convex_sphere_rows: Vec<Vec<(usize, usize)>>,
    static_convex_sphere_centers: Vec<Vec<[f32; 3]>>,
    prescribed_capsule_rows: [Vec<Vec<(usize, usize)>>; 5],
    prescribed_axial_rows: [Vec<Vec<(usize, usize)>>; 5],
    static_sphere_box_rows: Vec<Vec<(usize, usize)>>,
    static_sphere_box_poses: Vec<Vec<[f32; 7]>>,
    static_capsule_box_rows: Vec<Vec<(usize, usize)>>,
    static_capsule_box_poses: Vec<Vec<[f32; 7]>>,
    static_box_box_rows: Vec<Vec<(usize, usize)>>,
    static_box_box_poses: Vec<Vec<[f32; 7]>>,
    scene_convex_rounded_rows: Vec<Vec<(usize, usize, usize)>>,
    scene_convex_rounded_poses: Vec<Vec<[f32; 7]>>,
    static_axial_convex_rows: Vec<Vec<(usize, usize)>>,
    static_axial_convex_poses: Vec<Vec<[f32; 7]>>,
    static_convex_pair_rows: Vec<Vec<(usize, usize)>>,
    static_constraint_rows: Vec<Vec<(usize, usize, usize)>>,
    prescribed_indexed_rows: Vec<Vec<(usize, usize, usize)>>,
    static_convex_pair_poses: Vec<Vec<[f32; 7]>>,
    static_axial_box_rows: Vec<Vec<(usize, usize)>>,
    static_axial_box_poses: Vec<Vec<[f32; 7]>>,
    coupling_preparation: Option<CouplingPreparation>,
    dynamic_shapes: Option<wgpu::Buffer>,
    dynamic_pair_slots: Option<wgpu::Buffer>,
    dynamic_pipeline: Option<wgpu::ComputePipeline>,
    dynamic_inactive_pipeline: Option<wgpu::ComputePipeline>,
    dynamic_flags: Option<wgpu::Buffer>,
    dynamic_row_indices: Option<wgpu::Buffer>,
    dynamic_rows: Vec<Range<usize>>,
    friction_rows: Vec<Option<Range<usize>>>,
    friction_dimensions: Vec<usize>,
    timestep: f64,
    poses: wgpu::Buffer,
    link_terms: wgpu::Buffer,
    inverse: wgpu::Buffer,
    velocities: wgpu::Buffer,
    accelerations: wgpu::Buffer,
    state_status: wgpu::Buffer,
    environment_count: usize,
    sphere_count: usize,
    contact_ranges: Vec<Range<usize>>,
    link_ranges: Vec<Range<usize>>,
    contact_activity: Option<ContactActivity>,
    motion_layout: Vec<PackedMotionLink>,
    motion_wake: Option<MotionWake>,
    load_wake: Option<LoadWake>,
    actuation_wake: Option<(MotionWake, wgpu::Buffer)>,
    gravity_wake: Option<GravityWake>,
    mass_status: wgpu::Buffer,
}

impl GpuArticulatedGroundContactBatch {
    /// Bind sphere contacts to fixed-root batches containing every link term.
    pub fn new(
        state: &GpuGeneralizedStateBatch,
        poses: &GpuArticulatedPoseBatch,
        mass: &GpuArticulatedMassAssemblyBatch,
        articulations: &[&Articulation],
        contacts: &[Vec<GpuArticulatedGroundSphere>],
        timestep: f64,
    ) -> Result<Self, GpuArticulatedGroundContactError> {
        Self::new_with_pairs(
            state,
            poses,
            mass,
            articulations,
            contacts,
            &vec![Vec::new(); articulations.len()],
            timestep,
        )
    }

    /// Bind ground contacts and explicit sphere pairs to fixed-root batches.
    pub fn new_with_pairs(
        state: &GpuGeneralizedStateBatch,
        poses: &GpuArticulatedPoseBatch,
        mass: &GpuArticulatedMassAssemblyBatch,
        articulations: &[&Articulation],
        contacts: &[Vec<GpuArticulatedGroundSphere>],
        pairs: &[Vec<GpuArticulatedSpherePair>],
        timestep: f64,
    ) -> Result<Self, GpuArticulatedGroundContactError> {
        Self::new_with_settings(
            state,
            poses,
            mass,
            articulations,
            contacts,
            GpuArticulatedContactSettings {
                boxes: &vec![Vec::new(); articulations.len()],
                axial_shapes: &vec![Vec::new(); articulations.len()],
                manifold_starts: &vec![None; articulations.len()],
                joint_frictions: &vec![Vec::new(); articulations.len()],
                joint_couplings: &vec![Vec::new(); articulations.len()],
                link_point_constraints: &vec![Vec::new(); articulations.len()],
                link_fixed_constraints: &vec![Vec::new(); articulations.len()],
                pairs,
                static_sphere_pairs: &vec![Vec::new(); articulations.len()],
                static_capsule_sphere_pairs: &vec![Vec::new(); articulations.len()],
                static_box_sphere_pairs: &vec![Vec::new(); articulations.len()],
                static_sphere_capsule_pairs: &vec![Vec::new(); articulations.len()],
                static_capsule_pairs: &vec![Vec::new(); articulations.len()],
                static_sphere_box_pairs: &vec![Vec::new(); articulations.len()],
                static_capsule_box_pairs: &vec![Vec::new(); articulations.len()],
                static_box_pairs: &vec![Vec::new(); articulations.len()],
                static_box_capsule_pairs: &vec![Vec::new(); articulations.len()],
                static_axial_sphere_pairs: &vec![Vec::new(); articulations.len()],
                static_axial_capsule_pairs: &vec![Vec::new(); articulations.len()],
                static_axial_box_pairs: &vec![Vec::new(); articulations.len()],
                static_axial_convex_pairs: &vec![Vec::new(); articulations.len()],
                axial_convex_pairs: &vec![Vec::new(); articulations.len()],
                axial_sphere_pairs: &vec![Vec::new(); articulations.len()],
                axial_box_pairs: &vec![Vec::new(); articulations.len()],
                axial_capsule_pairs: &vec![Vec::new(); articulations.len()],
                axial_pairs: &vec![Vec::new(); articulations.len()],
                convex_sphere_pairs: &vec![Vec::new(); articulations.len()],
                static_convex_sphere_pairs: &vec![Vec::new(); articulations.len()],
                static_convex_capsule_pairs: &vec![Vec::new(); articulations.len()],
                convex_capsule_pairs: &vec![Vec::new(); articulations.len()],
                convex_pairs: &vec![Vec::new(); articulations.len()],
                static_convex_pairs: &vec![Vec::new(); articulations.len()],
                scene_convex_sphere_pairs: &vec![Vec::new(); articulations.len()],
                scene_mesh_sphere_pairs: &vec![Vec::new(); articulations.len()],
                scene_polyline_sphere_pairs: &vec![Vec::new(); articulations.len()],
                scene_mesh_capsule_pairs: &vec![Vec::new(); articulations.len()],
                scene_polyline_capsule_pairs: &vec![Vec::new(); articulations.len()],
                scene_mesh_box_pairs: &vec![Vec::new(); articulations.len()],
                scene_polyline_box_pairs: &vec![Vec::new(); articulations.len()],
                scene_mesh_axial_pairs: &vec![Vec::new(); articulations.len()],
                scene_polyline_axial_pairs: &vec![Vec::new(); articulations.len()],
                scene_mesh_convex_pairs: &vec![Vec::new(); articulations.len()],
                scene_polyline_convex_pairs: &vec![Vec::new(); articulations.len()],
                scene_convex_capsule_pairs: &vec![Vec::new(); articulations.len()],
                dynamic_spheres: &vec![Vec::new(); articulations.len()],
                dynamic_capsules: &vec![Vec::new(); articulations.len()],
                dynamic_boxes: &vec![Vec::new(); articulations.len()],
                dynamic_material_rules: &vec![Vec::new(); articulations.len()],
                capsule_sphere_pairs: &vec![Vec::new(); articulations.len()],
                capsule_pairs: &vec![Vec::new(); articulations.len()],
                sphere_box_pairs: &vec![Vec::new(); articulations.len()],
                capsule_box_pairs: &vec![Vec::new(); articulations.len()],
                box_pairs: &vec![Vec::new(); articulations.len()],
                iterations: &vec![8; articulations.len()],
                warm_start: &vec![false; articulations.len()],
            },
            timestep,
        )
    }

    /// Bind contacts with an independent iteration count for each environment.
    pub fn new_with_settings(
        state: &GpuGeneralizedStateBatch,
        poses: &GpuArticulatedPoseBatch,
        mass: &GpuArticulatedMassAssemblyBatch,
        articulations: &[&Articulation],
        contacts: &[Vec<GpuArticulatedGroundSphere>],
        settings: GpuArticulatedContactSettings<'_>,
        timestep: f64,
    ) -> Result<Self, GpuArticulatedGroundContactError> {
        let boxes = settings.boxes;
        let axial_shapes = settings.axial_shapes;
        let manifold_starts = settings.manifold_starts;
        let joint_frictions = settings.joint_frictions;
        let joint_couplings = settings.joint_couplings;
        let link_point_constraints = settings.link_point_constraints;
        let link_fixed_constraints = settings.link_fixed_constraints;
        let pairs = settings.pairs;
        let static_sphere_pairs = settings.static_sphere_pairs;
        let static_capsule_sphere_pairs = settings.static_capsule_sphere_pairs;
        let static_box_sphere_pairs = settings.static_box_sphere_pairs;
        let static_sphere_capsule_pairs = settings.static_sphere_capsule_pairs;
        let static_capsule_pairs = settings.static_capsule_pairs;
        let static_sphere_box_pairs = settings.static_sphere_box_pairs;
        let static_capsule_box_pairs = settings.static_capsule_box_pairs;
        let static_box_pairs = settings.static_box_pairs;
        let static_box_capsule_pairs = settings.static_box_capsule_pairs;
        let static_axial_sphere_pairs = settings.static_axial_sphere_pairs;
        let static_axial_capsule_pairs = settings.static_axial_capsule_pairs;
        let static_axial_box_pairs = settings.static_axial_box_pairs;
        let static_axial_convex_pairs = settings.static_axial_convex_pairs;
        let axial_convex_pairs = settings.axial_convex_pairs;
        let axial_sphere_pairs = settings.axial_sphere_pairs;
        let axial_box_pairs = settings.axial_box_pairs;
        let axial_capsule_pairs = settings.axial_capsule_pairs;
        let axial_pairs = settings.axial_pairs;
        let convex_sphere_pairs = settings.convex_sphere_pairs;
        let static_convex_sphere_pairs = settings.static_convex_sphere_pairs;
        let static_convex_capsule_pairs = settings.static_convex_capsule_pairs;
        let convex_capsule_pairs = settings.convex_capsule_pairs;
        let convex_pairs = settings.convex_pairs;
        let static_convex_pairs = settings.static_convex_pairs;
        let scene_convex_sphere_pairs = settings.scene_convex_sphere_pairs;
        let scene_mesh_sphere_pairs = settings.scene_mesh_sphere_pairs;
        let scene_polyline_sphere_pairs = settings.scene_polyline_sphere_pairs;
        let scene_mesh_capsule_pairs = settings.scene_mesh_capsule_pairs;
        let scene_polyline_capsule_pairs = settings.scene_polyline_capsule_pairs;
        let scene_mesh_box_pairs = settings.scene_mesh_box_pairs;
        let scene_polyline_box_pairs = settings.scene_polyline_box_pairs;
        let scene_mesh_axial_pairs = settings.scene_mesh_axial_pairs;
        let scene_polyline_axial_pairs = settings.scene_polyline_axial_pairs;
        let scene_mesh_convex_pairs = settings.scene_mesh_convex_pairs;
        let scene_polyline_convex_pairs = settings.scene_polyline_convex_pairs;
        let scene_convex_capsule_pairs = settings.scene_convex_capsule_pairs;
        let dynamic_spheres = settings.dynamic_spheres;
        let dynamic_capsules = settings.dynamic_capsules;
        let dynamic_boxes = settings.dynamic_boxes;
        let dynamic_material_rules = settings.dynamic_material_rules;
        let capsule_sphere_pairs = settings.capsule_sphere_pairs;
        let capsule_pairs = settings.capsule_pairs;
        let sphere_box_pairs = settings.sphere_box_pairs;
        let capsule_box_pairs = settings.capsule_box_pairs;
        let box_pairs = settings.box_pairs;
        let iterations = settings.iterations;
        let warm_start = settings.warm_start;
        if articulations.is_empty()
            || articulations.len() != contacts.len()
            || articulations.len() != boxes.len()
            || articulations.len() != axial_shapes.len()
            || articulations.len() != manifold_starts.len()
            || articulations.len() != joint_frictions.len()
            || articulations.len() != joint_couplings.len()
            || articulations.len() != link_point_constraints.len()
            || articulations.len() != link_fixed_constraints.len()
            || articulations.len() != pairs.len()
            || articulations.len() != static_sphere_pairs.len()
            || articulations.len() != static_capsule_sphere_pairs.len()
            || articulations.len() != static_box_sphere_pairs.len()
            || articulations.len() != static_sphere_capsule_pairs.len()
            || articulations.len() != static_capsule_pairs.len()
            || articulations.len() != static_sphere_box_pairs.len()
            || articulations.len() != static_capsule_box_pairs.len()
            || articulations.len() != static_box_pairs.len()
            || articulations.len() != static_box_capsule_pairs.len()
            || articulations.len() != static_axial_sphere_pairs.len()
            || articulations.len() != static_axial_capsule_pairs.len()
            || articulations.len() != static_axial_box_pairs.len()
            || articulations.len() != static_axial_convex_pairs.len()
            || articulations.len() != axial_convex_pairs.len()
            || articulations.len() != axial_sphere_pairs.len()
            || articulations.len() != axial_box_pairs.len()
            || articulations.len() != axial_capsule_pairs.len()
            || articulations.len() != axial_pairs.len()
            || articulations.len() != convex_sphere_pairs.len()
            || articulations.len() != static_convex_sphere_pairs.len()
            || articulations.len() != static_convex_capsule_pairs.len()
            || articulations.len() != convex_capsule_pairs.len()
            || articulations.len() != convex_pairs.len()
            || articulations.len() != static_convex_pairs.len()
            || articulations.len() != scene_convex_sphere_pairs.len()
            || articulations.len() != scene_mesh_sphere_pairs.len()
            || articulations.len() != scene_polyline_sphere_pairs.len()
            || articulations.len() != scene_mesh_capsule_pairs.len()
            || articulations.len() != scene_polyline_capsule_pairs.len()
            || articulations.len() != scene_mesh_box_pairs.len()
            || articulations.len() != scene_polyline_box_pairs.len()
            || articulations.len() != scene_mesh_axial_pairs.len()
            || articulations.len() != scene_polyline_axial_pairs.len()
            || articulations.len() != scene_mesh_convex_pairs.len()
            || articulations.len() != scene_polyline_convex_pairs.len()
            || articulations.len() != scene_convex_capsule_pairs.len()
            || articulations.len() != dynamic_spheres.len()
            || articulations.len() != dynamic_capsules.len()
            || articulations.len() != dynamic_boxes.len()
            || articulations.len() != dynamic_material_rules.len()
            || articulations.len() != capsule_sphere_pairs.len()
            || articulations.len() != capsule_pairs.len()
            || articulations.len() != sphere_box_pairs.len()
            || articulations.len() != capsule_box_pairs.len()
            || articulations.len() != box_pairs.len()
            || articulations.len() != iterations.len()
            || articulations.len() != warm_start.len()
            || iterations.iter().any(|&count| !(1..=64).contains(&count))
            || articulations.len() != state.ranges().len()
            || articulations.len() != poses.link_ranges().len()
            || articulations.len() != mass.dimensions().len()
            || !timestep.is_finite()
            || timestep <= 0.0
        {
            return Err(GpuArticulatedGroundContactError::InvalidInput);
        }
        let dt = finite_f32(timestep)?;
        let mut systems = Vec::with_capacity(articulations.len());
        let mut motion_layout = Vec::new();
        let mut spheres = Vec::new();
        let mut static_sphere_rows = vec![Vec::new(); articulations.len()];
        let mut static_sphere_centers = vec![Vec::new(); articulations.len()];
        let mut static_capsule_sphere_rows = vec![Vec::new(); articulations.len()];
        let mut static_capsule_sphere_centers = vec![Vec::new(); articulations.len()];
        let mut static_box_sphere_rows = vec![Vec::new(); articulations.len()];
        let mut static_box_sphere_centers = vec![Vec::new(); articulations.len()];
        let mut static_axial_sphere_rows = vec![Vec::new(); articulations.len()];
        let mut static_axial_sphere_centers = vec![Vec::new(); articulations.len()];
        let mut static_convex_sphere_rows = vec![Vec::new(); articulations.len()];
        let mut static_convex_sphere_centers = vec![Vec::new(); articulations.len()];
        let mut static_sphere_box_rows = vec![Vec::new(); articulations.len()];
        let mut static_sphere_box_poses = vec![Vec::new(); articulations.len()];
        let mut static_capsule_box_rows = vec![Vec::new(); articulations.len()];
        let mut static_capsule_box_poses = vec![Vec::new(); articulations.len()];
        let mut static_box_box_rows = vec![Vec::new(); articulations.len()];
        let mut static_box_box_poses = vec![Vec::new(); articulations.len()];
        let mut scene_convex_rounded_rows = vec![Vec::new(); articulations.len()];
        let mut scene_convex_rounded_poses = vec![Vec::new(); articulations.len()];
        let mut static_axial_convex_rows = vec![Vec::new(); articulations.len()];
        let mut static_axial_convex_poses = vec![Vec::new(); articulations.len()];
        let mut static_convex_pair_rows = vec![Vec::new(); articulations.len()];
        let mut static_constraint_rows = vec![Vec::new(); articulations.len()];
        let mut static_convex_pair_poses = vec![Vec::new(); articulations.len()];
        let mut static_axial_box_rows = vec![Vec::new(); articulations.len()];
        let mut static_axial_box_poses = vec![Vec::new(); articulations.len()];
        let mut convex_geometry = Vec::<PackedSphere>::new();
        let mut convex_contact_rows = Vec::<usize>::new();
        let mut mesh_geometry_offsets = HashMap::<usize, [f32; 4]>::new();
        let mut packed_dynamic_shapes = Vec::new();
        let mut explicit_shape_exclusions = HashSet::<PackedShapePair>::new();
        let mut dynamic_pair_slots = Vec::<u32>::new();
        let mut dynamic_rows = Vec::with_capacity(articulations.len());
        let mut friction_rows = Vec::with_capacity(articulations.len());
        let mut coupling_rows = Vec::new();
        let mut friction_dimensions = Vec::with_capacity(articulations.len());
        let mut dynamic_slot_count = 0usize;
        let mut link_offset = 0usize;
        for (index, ((((articulation, contact_list), pair_list), range), (&n, &link_count))) in
            articulations
                .iter()
                .zip(contacts)
                .zip(pairs)
                .zip(state.ranges())
                .zip(mass.dimensions().iter().zip(mass.link_counts()))
                .enumerate()
        {
            let root_dofs = if poses.floating_roots()[index] { 6 } else { 0 };
            if articulation.dof().checked_add(root_dofs) != Some(n)
                || range.len() != n
                || link_count != articulation.link_count()
                || poses.link_ranges()[index].len() != link_count
                || (!joint_frictions[index].is_empty() && joint_frictions[index].len() != n)
            {
                return Err(GpuArticulatedGroundContactError::InvalidInput);
            }
            let collider_start = spheres.len();
            let stride = 10usize
                .checked_add(
                    n.checked_mul(6)
                        .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                )
                .ok_or(GpuArticulatedGroundContactError::Capacity)?;
            if manifold_starts[index].is_some_and(|start| start > contact_list.len()) {
                return Err(GpuArticulatedGroundContactError::InvalidInput);
            }
            for (contact_index, contact) in contact_list.iter().enumerate() {
                let link = articulation
                    .link(contact.link)
                    .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
                if contact.radius < 0.0
                    || !contact.radius.is_finite()
                    || !contact.plane_offset.is_finite()
                    || !contact.restitution.is_finite()
                    || !(0.0..=1.0).contains(&contact.restitution)
                    || !contact.friction.is_finite()
                    || contact.friction < 0.0
                    || contact.local_center.iter().any(|value| !value.is_finite())
                    || contact.plane_normal.iter().any(|value| !value.is_finite())
                {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                let normal_length = contact.plane_normal.norm();
                if !normal_length.is_finite() || normal_length <= 1e-12 {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                let normal = contact.plane_normal / normal_length;
                let plane_xy_half_extent = match contact.plane_xy_half_extent {
                    Some(extent)
                        if extent.is_finite()
                            && extent > 0.0
                            && normal.x.abs() <= 1e-12
                            && normal.y.abs() <= 1e-12
                            && normal.z > 0.0 =>
                    {
                        let packed = finite_f32(extent)?;
                        if packed <= 0.0 {
                            return Err(GpuArticulatedGroundContactError::InvalidInput);
                        }
                        packed
                    }
                    Some(_) => return Err(GpuArticulatedGroundContactError::InvalidInput),
                    None => 0.0,
                };
                spheres.push(PackedSphere {
                    indices: [
                        checked_u32(poses.link_ranges()[index].start + contact.link)?,
                        checked_u32(
                            link_offset
                                .checked_add(
                                    contact
                                        .link
                                        .checked_mul(stride)
                                        .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                                )
                                .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                        )?,
                        0,
                        0,
                    ],
                    center_radius: [
                        finite_f32(contact.local_center.x)?,
                        finite_f32(contact.local_center.y)?,
                        finite_f32(contact.local_center.z)?,
                        finite_f32(contact.radius)?,
                    ],
                    center_of_mass: [
                        finite_f32(link.center_of_mass.x)?,
                        finite_f32(link.center_of_mass.y)?,
                        finite_f32(link.center_of_mass.z)?,
                        0.0,
                    ],
                    plane: [
                        finite_f32(normal.x)?,
                        finite_f32(normal.y)?,
                        finite_f32(normal.z)?,
                        finite_f32(contact.plane_offset / normal_length)?,
                    ],
                    material: [
                        finite_f32(contact.restitution)?,
                        dt,
                        finite_f32(contact.friction)?,
                        if contact.radius == 0.0
                            || manifold_starts[index].is_some_and(|start| contact_index >= start)
                        {
                            10.0
                        } else {
                            0.0
                        },
                    ],
                    other_center_radius: [0.0, 0.0, 0.0, plane_xy_half_extent],
                    other_center_of_mass: [0.0; 4],
                    second_axis_end: [0.0; 4],
                    first_axis_end: [0.0; 4],
                    impulses: [0.0; 4],
                    previous_normal: [0.0; 4],
                    diagnostic_first: [0.0; 4],
                    diagnostic_second: [0.0; 4],
                    diagnostic_first_origin: [0.0; 4],
                    diagnostic_second_origin: [0.0; 4],
                    prescribed_linear: [0.0; 4],
                    prescribed_angular: [0.0; 4],
                });
            }
            for contact in &boxes[index] {
                let link = articulation
                    .link(contact.link)
                    .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
                if contact
                    .half_extents
                    .iter()
                    .any(|value| !value.is_finite() || *value <= 0.0)
                    || !contact.plane_offset.is_finite()
                    || !(0.0..=1.0).contains(&contact.restitution)
                    || !contact.friction.is_finite()
                    || contact.friction < 0.0
                    || contact
                        .local_pose
                        .translation
                        .vector
                        .iter()
                        .any(|value| !value.is_finite())
                    || contact
                        .local_pose
                        .rotation
                        .coords
                        .iter()
                        .any(|value| !value.is_finite())
                    || contact.plane_normal.iter().any(|value| !value.is_finite())
                {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                let normal_length = contact.plane_normal.norm();
                if !normal_length.is_finite() || normal_length <= 1e-12 {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                let normal = contact.plane_normal / normal_length;
                let plane_xy_half_extent = match contact.plane_xy_half_extent {
                    Some(extent)
                        if extent.is_finite()
                            && extent > 0.0
                            && normal.x.abs() <= 1e-12
                            && normal.y.abs() <= 1e-12
                            && normal.z > 0.0 =>
                    {
                        let packed = finite_f32(extent)?;
                        if packed <= 0.0 {
                            return Err(GpuArticulatedGroundContactError::InvalidInput);
                        }
                        packed
                    }
                    Some(_) => return Err(GpuArticulatedGroundContactError::InvalidInput),
                    None => 0.0,
                };
                let local = contact.local_pose.translation.vector;
                let rotation = contact.local_pose.rotation.quaternion();
                let packed = PackedSphere {
                    indices: [
                        checked_u32(poses.link_ranges()[index].start + contact.link)?,
                        checked_u32(
                            link_offset
                                .checked_add(
                                    contact
                                        .link
                                        .checked_mul(stride)
                                        .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                                )
                                .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                        )?,
                        0,
                        0,
                    ],
                    center_radius: [
                        finite_f32(local.x)?,
                        finite_f32(local.y)?,
                        finite_f32(local.z)?,
                        0.0,
                    ],
                    center_of_mass: [
                        finite_f32(link.center_of_mass.x)?,
                        finite_f32(link.center_of_mass.y)?,
                        finite_f32(link.center_of_mass.z)?,
                        0.0,
                    ],
                    plane: [
                        finite_f32(normal.x)?,
                        finite_f32(normal.y)?,
                        finite_f32(normal.z)?,
                        finite_f32(contact.plane_offset / normal_length)?,
                    ],
                    material: [
                        finite_f32(contact.restitution)?,
                        dt,
                        finite_f32(contact.friction)?,
                        5.0,
                    ],
                    other_center_radius: [
                        finite_f32(contact.half_extents.x)?,
                        finite_f32(contact.half_extents.y)?,
                        finite_f32(contact.half_extents.z)?,
                        plane_xy_half_extent,
                    ],
                    other_center_of_mass: [0.0; 4],
                    second_axis_end: [
                        finite_f32(rotation.i)?,
                        finite_f32(rotation.j)?,
                        finite_f32(rotation.k)?,
                        finite_f32(rotation.w)?,
                    ],
                    first_axis_end: [0.0; 4],
                    impulses: [0.0; 4],
                    previous_normal: [0.0; 4],
                    diagnostic_first: [0.0; 4],
                    diagnostic_second: [0.0; 4],
                    diagnostic_first_origin: [0.0; 4],
                    diagnostic_second_origin: [0.0; 4],
                    prescribed_linear: [0.0; 4],
                    prescribed_angular: [0.0; 4],
                };
                for corner in 0..4 {
                    let mut row = packed;
                    row.indices[2] = corner;
                    spheres.push(row);
                }
            }
            for contact in &axial_shapes[index] {
                let link = articulation
                    .link(contact.link)
                    .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
                if !contact.half_height.is_finite()
                    || contact.half_height <= 0.0
                    || !contact.radius.is_finite()
                    || contact.radius <= 0.0
                    || !contact.plane_offset.is_finite()
                    || !(0.0..=1.0).contains(&contact.restitution)
                    || !contact.friction.is_finite()
                    || contact.friction < 0.0
                    || contact
                        .local_pose
                        .translation
                        .vector
                        .iter()
                        .any(|value| !value.is_finite())
                    || contact
                        .local_pose
                        .rotation
                        .coords
                        .iter()
                        .any(|value| !value.is_finite())
                    || contact.plane_normal.iter().any(|value| !value.is_finite())
                {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                let normal_length = contact.plane_normal.norm();
                if !normal_length.is_finite() || normal_length <= 1e-12 {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                let normal = contact.plane_normal / normal_length;
                let plane_xy_half_extent = match contact.plane_xy_half_extent {
                    Some(extent)
                        if extent.is_finite()
                            && extent > 0.0
                            && normal.x.abs() <= 1e-12
                            && normal.y.abs() <= 1e-12
                            && normal.z > 0.0 =>
                    {
                        let packed = finite_f32(extent)?;
                        if packed <= 0.0 {
                            return Err(GpuArticulatedGroundContactError::InvalidInput);
                        }
                        packed
                    }
                    Some(_) => return Err(GpuArticulatedGroundContactError::InvalidInput),
                    None => 0.0,
                };
                let local = contact.local_pose.translation.vector;
                let rotation = contact.local_pose.rotation.quaternion();
                let packed = PackedSphere {
                    indices: [
                        checked_u32(poses.link_ranges()[index].start + contact.link)?,
                        checked_u32(
                            link_offset
                                .checked_add(
                                    contact
                                        .link
                                        .checked_mul(stride)
                                        .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                                )
                                .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                        )?,
                        0,
                        0,
                    ],
                    center_radius: [
                        finite_f32(local.x)?,
                        finite_f32(local.y)?,
                        finite_f32(local.z)?,
                        0.0,
                    ],
                    center_of_mass: [
                        finite_f32(link.center_of_mass.x)?,
                        finite_f32(link.center_of_mass.y)?,
                        finite_f32(link.center_of_mass.z)?,
                        0.0,
                    ],
                    plane: [
                        finite_f32(normal.x)?,
                        finite_f32(normal.y)?,
                        finite_f32(normal.z)?,
                        finite_f32(contact.plane_offset / normal_length)?,
                    ],
                    material: [
                        finite_f32(contact.restitution)?,
                        dt,
                        finite_f32(contact.friction)?,
                        match contact.kind {
                            GpuArticulatedGroundAxialKind::Cylinder => 12.0,
                            GpuArticulatedGroundAxialKind::Cone => 13.0,
                        },
                    ],
                    other_center_radius: [
                        finite_f32(contact.radius)?,
                        finite_f32(contact.half_height)?,
                        0.0,
                        plane_xy_half_extent,
                    ],
                    second_axis_end: [
                        finite_f32(rotation.i)?,
                        finite_f32(rotation.j)?,
                        finite_f32(rotation.k)?,
                        finite_f32(rotation.w)?,
                    ],
                    ..PackedSphere::zeroed()
                };
                for point in 0..4 {
                    let mut row = packed;
                    row.indices[2] = point;
                    spheres.push(row);
                }
            }
            for pair in pair_list {
                let first = articulation
                    .link(pair.first_link)
                    .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
                let second = articulation
                    .link(pair.second_link)
                    .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
                if pair.first_link == pair.second_link
                    || !pair.first_radius.is_finite()
                    || pair.first_radius <= 0.0
                    || !pair.second_radius.is_finite()
                    || pair.second_radius <= 0.0
                    || !pair.restitution.is_finite()
                    || !(0.0..=1.0).contains(&pair.restitution)
                    || !pair.friction.is_finite()
                    || pair.friction < 0.0
                    || pair
                        .first_local_center
                        .iter()
                        .any(|value| !value.is_finite())
                    || pair
                        .second_local_center
                        .iter()
                        .any(|value| !value.is_finite())
                {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                let term_offset = |link: usize| -> Result<u32, GpuArticulatedGroundContactError> {
                    checked_u32(
                        link_offset
                            .checked_add(
                                link.checked_mul(stride)
                                    .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                            )
                            .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                    )
                };
                spheres.push(PackedSphere {
                    indices: [
                        checked_u32(poses.link_ranges()[index].start + pair.first_link)?,
                        term_offset(pair.first_link)?,
                        checked_u32(poses.link_ranges()[index].start + pair.second_link)?,
                        term_offset(pair.second_link)?,
                    ],
                    center_radius: [
                        finite_f32(pair.first_local_center.x)?,
                        finite_f32(pair.first_local_center.y)?,
                        finite_f32(pair.first_local_center.z)?,
                        finite_f32(pair.first_radius)?,
                    ],
                    center_of_mass: [
                        finite_f32(first.center_of_mass.x)?,
                        finite_f32(first.center_of_mass.y)?,
                        finite_f32(first.center_of_mass.z)?,
                        0.0,
                    ],
                    plane: [0.0; 4],
                    material: [
                        finite_f32(pair.restitution)?,
                        dt,
                        finite_f32(pair.friction)?,
                        1.0,
                    ],
                    other_center_radius: [
                        finite_f32(pair.second_local_center.x)?,
                        finite_f32(pair.second_local_center.y)?,
                        finite_f32(pair.second_local_center.z)?,
                        finite_f32(pair.second_radius)?,
                    ],
                    other_center_of_mass: [
                        finite_f32(second.center_of_mass.x)?,
                        finite_f32(second.center_of_mass.y)?,
                        finite_f32(second.center_of_mass.z)?,
                        0.0,
                    ],
                    second_axis_end: [0.0; 4],
                    first_axis_end: [0.0; 4],
                    impulses: [0.0; 4],
                    previous_normal: [0.0; 4],
                    diagnostic_first: [0.0; 4],
                    diagnostic_second: [0.0; 4],
                    diagnostic_first_origin: [0.0; 4],
                    diagnostic_second_origin: [0.0; 4],
                    prescribed_linear: [0.0; 4],
                    prescribed_angular: [0.0; 4],
                });
            }
            for pair in &static_sphere_pairs[index] {
                let link = articulation
                    .link(pair.link)
                    .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
                if !pair.radius.is_finite()
                    || pair.radius <= 0.0
                    || !pair.static_radius.is_finite()
                    || pair.static_radius <= 0.0
                    || !(0.0..=1.0).contains(&pair.restitution)
                    || !pair.friction.is_finite()
                    || pair.friction < 0.0
                    || pair.local_center.iter().any(|value| !value.is_finite())
                    || pair.static_center.iter().any(|value| !value.is_finite())
                {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                let term_offset = link_offset
                    .checked_add(
                        pair.link
                            .checked_mul(stride)
                            .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                    )
                    .ok_or(GpuArticulatedGroundContactError::Capacity)?;
                let center = [
                    finite_f32(pair.static_center.x)?,
                    finite_f32(pair.static_center.y)?,
                    finite_f32(pair.static_center.z)?,
                ];
                static_sphere_rows[index]
                    .push((spheres.len(), poses.link_ranges()[index].start + pair.link));
                static_sphere_centers[index].push(center);
                spheres.push(PackedSphere {
                    indices: [
                        checked_u32(poses.link_ranges()[index].start + pair.link)?,
                        checked_u32(term_offset)?,
                        0,
                        0,
                    ],
                    center_radius: [
                        finite_f32(pair.local_center.x)?,
                        finite_f32(pair.local_center.y)?,
                        finite_f32(pair.local_center.z)?,
                        finite_f32(pair.radius)?,
                    ],
                    center_of_mass: [
                        finite_f32(link.center_of_mass.x)?,
                        finite_f32(link.center_of_mass.y)?,
                        finite_f32(link.center_of_mass.z)?,
                        0.0,
                    ],
                    material: [
                        finite_f32(pair.restitution)?,
                        dt,
                        finite_f32(pair.friction)?,
                        14.0,
                    ],
                    other_center_radius: [
                        finite_f32(pair.static_center.x)?,
                        finite_f32(pair.static_center.y)?,
                        finite_f32(pair.static_center.z)?,
                        finite_f32(pair.static_radius)?,
                    ],
                    ..PackedSphere::zeroed()
                });
            }
            for pair in &static_capsule_sphere_pairs[index] {
                let link = articulation
                    .link(pair.link)
                    .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
                if !pair.radius.is_finite()
                    || pair.radius <= 0.0
                    || !pair.static_radius.is_finite()
                    || pair.static_radius <= 0.0
                    || !(0.0..=1.0).contains(&pair.restitution)
                    || !pair.friction.is_finite()
                    || pair.friction < 0.0
                    || pair.local_a.iter().any(|value| !value.is_finite())
                    || pair.local_b.iter().any(|value| !value.is_finite())
                    || pair.static_center.iter().any(|value| !value.is_finite())
                {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                let term_offset = link_offset
                    .checked_add(
                        pair.link
                            .checked_mul(stride)
                            .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                    )
                    .ok_or(GpuArticulatedGroundContactError::Capacity)?;
                static_capsule_sphere_rows[index]
                    .push((spheres.len(), poses.link_ranges()[index].start + pair.link));
                static_capsule_sphere_centers[index].push([
                    finite_f32(pair.static_center.x)?,
                    finite_f32(pair.static_center.y)?,
                    finite_f32(pair.static_center.z)?,
                ]);
                spheres.push(PackedSphere {
                    indices: [
                        checked_u32(poses.link_ranges()[index].start + pair.link)?,
                        checked_u32(term_offset)?,
                        0,
                        0,
                    ],
                    center_radius: [
                        finite_f32(pair.local_a.x)?,
                        finite_f32(pair.local_a.y)?,
                        finite_f32(pair.local_a.z)?,
                        finite_f32(pair.radius)?,
                    ],
                    center_of_mass: [
                        finite_f32(link.center_of_mass.x)?,
                        finite_f32(link.center_of_mass.y)?,
                        finite_f32(link.center_of_mass.z)?,
                        0.0,
                    ],
                    material: [
                        finite_f32(pair.restitution)?,
                        dt,
                        finite_f32(pair.friction)?,
                        15.0,
                    ],
                    other_center_radius: [
                        finite_f32(pair.static_center.x)?,
                        finite_f32(pair.static_center.y)?,
                        finite_f32(pair.static_center.z)?,
                        finite_f32(pair.static_radius)?,
                    ],
                    first_axis_end: [
                        finite_f32(pair.local_b.x)?,
                        finite_f32(pair.local_b.y)?,
                        finite_f32(pair.local_b.z)?,
                        0.0,
                    ],
                    ..PackedSphere::zeroed()
                });
            }
            for pair in &static_box_sphere_pairs[index] {
                let link = articulation
                    .link(pair.link)
                    .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
                if pair
                    .half_extents
                    .iter()
                    .any(|value| !value.is_finite() || *value <= 0.0)
                    || !pair.static_radius.is_finite()
                    || pair.static_radius <= 0.0
                    || !(0.0..=1.0).contains(&pair.restitution)
                    || !pair.friction.is_finite()
                    || pair.friction < 0.0
                    || pair
                        .local_pose
                        .translation
                        .vector
                        .iter()
                        .chain(pair.local_pose.rotation.coords.iter())
                        .chain(pair.static_center.iter())
                        .any(|value| !value.is_finite())
                {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                let term_offset = link_offset
                    .checked_add(
                        pair.link
                            .checked_mul(stride)
                            .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                    )
                    .ok_or(GpuArticulatedGroundContactError::Capacity)?;
                let local = pair.local_pose.translation.vector;
                let rotation = pair.local_pose.rotation.quaternion();
                static_box_sphere_rows[index]
                    .push((spheres.len(), poses.link_ranges()[index].start + pair.link));
                static_box_sphere_centers[index].push([
                    finite_f32(pair.static_center.x)?,
                    finite_f32(pair.static_center.y)?,
                    finite_f32(pair.static_center.z)?,
                ]);
                spheres.push(PackedSphere {
                    indices: [
                        checked_u32(poses.link_ranges()[index].start + pair.link)?,
                        checked_u32(term_offset)?,
                        0,
                        0,
                    ],
                    center_radius: [
                        finite_f32(local.x)?,
                        finite_f32(local.y)?,
                        finite_f32(local.z)?,
                        finite_f32(pair.static_radius)?,
                    ],
                    center_of_mass: [
                        finite_f32(link.center_of_mass.x)?,
                        finite_f32(link.center_of_mass.y)?,
                        finite_f32(link.center_of_mass.z)?,
                        0.0,
                    ],
                    plane: [
                        finite_f32(pair.static_center.x)?,
                        finite_f32(pair.static_center.y)?,
                        finite_f32(pair.static_center.z)?,
                        0.0,
                    ],
                    material: [
                        finite_f32(pair.restitution)?,
                        dt,
                        finite_f32(pair.friction)?,
                        16.0,
                    ],
                    other_center_radius: [
                        finite_f32(pair.half_extents.x)?,
                        finite_f32(pair.half_extents.y)?,
                        finite_f32(pair.half_extents.z)?,
                        0.0,
                    ],
                    second_axis_end: [
                        finite_f32(rotation.i)?,
                        finite_f32(rotation.j)?,
                        finite_f32(rotation.k)?,
                        finite_f32(rotation.w)?,
                    ],
                    ..PackedSphere::zeroed()
                });
            }
            for pair in &static_sphere_capsule_pairs[index] {
                let link = articulation
                    .link(pair.link)
                    .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
                if !pair.radius.is_finite()
                    || pair.radius <= 0.0
                    || !pair.static_radius.is_finite()
                    || pair.static_radius < 0.0
                    || !(0.0..=1.0).contains(&pair.restitution)
                    || !pair.friction.is_finite()
                    || pair.friction < 0.0
                    || pair
                        .local_center
                        .iter()
                        .chain(pair.static_a.iter())
                        .chain(pair.static_b.iter())
                        .any(|value| !value.is_finite())
                {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                let term_offset = link_offset
                    .checked_add(
                        pair.link
                            .checked_mul(stride)
                            .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                    )
                    .ok_or(GpuArticulatedGroundContactError::Capacity)?;
                spheres.push(PackedSphere {
                    indices: [
                        checked_u32(poses.link_ranges()[index].start + pair.link)?,
                        checked_u32(term_offset)?,
                        0,
                        0,
                    ],
                    center_radius: [
                        finite_f32(pair.local_center.x)?,
                        finite_f32(pair.local_center.y)?,
                        finite_f32(pair.local_center.z)?,
                        finite_f32(pair.radius)?,
                    ],
                    center_of_mass: [
                        finite_f32(link.center_of_mass.x)?,
                        finite_f32(link.center_of_mass.y)?,
                        finite_f32(link.center_of_mass.z)?,
                        0.0,
                    ],
                    plane: [
                        finite_f32(pair.static_a.x)?,
                        finite_f32(pair.static_a.y)?,
                        finite_f32(pair.static_a.z)?,
                        finite_f32(pair.static_radius)?,
                    ],
                    material: [
                        finite_f32(pair.restitution)?,
                        dt,
                        finite_f32(pair.friction)?,
                        17.0,
                    ],
                    other_center_radius: [
                        finite_f32(pair.static_b.x)?,
                        finite_f32(pair.static_b.y)?,
                        finite_f32(pair.static_b.z)?,
                        0.0,
                    ],
                    ..PackedSphere::zeroed()
                });
            }
            for pair in &static_capsule_pairs[index] {
                let link = articulation
                    .link(pair.link)
                    .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
                if !pair.radius.is_finite()
                    || pair.radius <= 0.0
                    || !pair.static_radius.is_finite()
                    || pair.static_radius < 0.0
                    || !(0.0..=1.0).contains(&pair.restitution)
                    || !pair.friction.is_finite()
                    || pair.friction < 0.0
                    || pair
                        .local_a
                        .iter()
                        .chain(pair.local_b.iter())
                        .chain(pair.static_a.iter())
                        .chain(pair.static_b.iter())
                        .any(|value| !value.is_finite())
                {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                let term_offset = link_offset
                    .checked_add(
                        pair.link
                            .checked_mul(stride)
                            .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                    )
                    .ok_or(GpuArticulatedGroundContactError::Capacity)?;
                let packed = PackedSphere {
                    indices: [
                        checked_u32(poses.link_ranges()[index].start + pair.link)?,
                        checked_u32(term_offset)?,
                        0,
                        0,
                    ],
                    center_radius: [
                        finite_f32(pair.local_a.x)?,
                        finite_f32(pair.local_a.y)?,
                        finite_f32(pair.local_a.z)?,
                        finite_f32(pair.radius)?,
                    ],
                    center_of_mass: [
                        finite_f32(link.center_of_mass.x)?,
                        finite_f32(link.center_of_mass.y)?,
                        finite_f32(link.center_of_mass.z)?,
                        0.0,
                    ],
                    plane: [
                        finite_f32(pair.static_a.x)?,
                        finite_f32(pair.static_a.y)?,
                        finite_f32(pair.static_a.z)?,
                        finite_f32(pair.static_radius)?,
                    ],
                    material: [
                        finite_f32(pair.restitution)?,
                        dt,
                        finite_f32(pair.friction)?,
                        18.0,
                    ],
                    other_center_radius: [
                        finite_f32(pair.local_b.x)?,
                        finite_f32(pair.local_b.y)?,
                        finite_f32(pair.local_b.z)?,
                        0.0,
                    ],
                    second_axis_end: [
                        finite_f32(pair.static_b.x)?,
                        finite_f32(pair.static_b.y)?,
                        finite_f32(pair.static_b.z)?,
                        0.0,
                    ],
                    ..PackedSphere::zeroed()
                };
                spheres.push(packed);
                let mut side = packed;
                side.material[3] = 19.0;
                spheres.push(side);
            }
            for pair in &static_sphere_box_pairs[index] {
                let link = articulation
                    .link(pair.link)
                    .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
                if !pair.radius.is_finite()
                    || pair.radius <= 0.0
                    || pair
                        .half_extents
                        .iter()
                        .any(|value| !value.is_finite() || *value <= 0.0)
                    || !(0.0..=1.0).contains(&pair.restitution)
                    || !pair.friction.is_finite()
                    || pair.friction < 0.0
                    || pair
                        .local_center
                        .iter()
                        .chain(pair.static_pose.translation.vector.iter())
                        .chain(pair.static_pose.rotation.coords.iter())
                        .any(|value| !value.is_finite())
                {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                let term_offset = link_offset
                    .checked_add(
                        pair.link
                            .checked_mul(stride)
                            .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                    )
                    .ok_or(GpuArticulatedGroundContactError::Capacity)?;
                let center = pair.static_pose.translation.vector;
                let rotation = pair.static_pose.rotation.quaternion();
                static_sphere_box_rows[index]
                    .push((spheres.len(), poses.link_ranges()[index].start + pair.link));
                static_sphere_box_poses[index].push([
                    finite_f32(center.x)?,
                    finite_f32(center.y)?,
                    finite_f32(center.z)?,
                    finite_f32(rotation.i)?,
                    finite_f32(rotation.j)?,
                    finite_f32(rotation.k)?,
                    finite_f32(rotation.w)?,
                ]);
                spheres.push(PackedSphere {
                    indices: [
                        checked_u32(poses.link_ranges()[index].start + pair.link)?,
                        checked_u32(term_offset)?,
                        0,
                        0,
                    ],
                    center_radius: [
                        finite_f32(pair.local_center.x)?,
                        finite_f32(pair.local_center.y)?,
                        finite_f32(pair.local_center.z)?,
                        finite_f32(pair.radius)?,
                    ],
                    center_of_mass: [
                        finite_f32(link.center_of_mass.x)?,
                        finite_f32(link.center_of_mass.y)?,
                        finite_f32(link.center_of_mass.z)?,
                        0.0,
                    ],
                    plane: [
                        finite_f32(center.x)?,
                        finite_f32(center.y)?,
                        finite_f32(center.z)?,
                        0.0,
                    ],
                    material: [
                        finite_f32(pair.restitution)?,
                        dt,
                        finite_f32(pair.friction)?,
                        20.0,
                    ],
                    other_center_radius: [
                        finite_f32(pair.half_extents.x)?,
                        finite_f32(pair.half_extents.y)?,
                        finite_f32(pair.half_extents.z)?,
                        0.0,
                    ],
                    second_axis_end: [
                        finite_f32(rotation.i)?,
                        finite_f32(rotation.j)?,
                        finite_f32(rotation.k)?,
                        finite_f32(rotation.w)?,
                    ],
                    ..PackedSphere::zeroed()
                });
            }
            for pair in &static_capsule_box_pairs[index] {
                let link = articulation
                    .link(pair.link)
                    .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
                if !pair.radius.is_finite()
                    || pair.radius <= 0.0
                    || pair
                        .half_extents
                        .iter()
                        .any(|value| !value.is_finite() || *value <= 0.0)
                    || !(0.0..=1.0).contains(&pair.restitution)
                    || !pair.friction.is_finite()
                    || pair.friction < 0.0
                    || pair
                        .local_a
                        .iter()
                        .chain(pair.local_b.iter())
                        .chain(pair.static_pose.translation.vector.iter())
                        .chain(pair.static_pose.rotation.coords.iter())
                        .any(|value| !value.is_finite())
                {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                let term_offset = link_offset
                    .checked_add(
                        pair.link
                            .checked_mul(stride)
                            .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                    )
                    .ok_or(GpuArticulatedGroundContactError::Capacity)?;
                let center = pair.static_pose.translation.vector;
                let rotation = pair.static_pose.rotation.quaternion();
                static_capsule_box_rows[index]
                    .push((spheres.len(), poses.link_ranges()[index].start + pair.link));
                static_capsule_box_poses[index].push([
                    finite_f32(center.x)?,
                    finite_f32(center.y)?,
                    finite_f32(center.z)?,
                    finite_f32(rotation.i)?,
                    finite_f32(rotation.j)?,
                    finite_f32(rotation.k)?,
                    finite_f32(rotation.w)?,
                ]);
                let packed = PackedSphere {
                    indices: [
                        checked_u32(poses.link_ranges()[index].start + pair.link)?,
                        checked_u32(term_offset)?,
                        0,
                        0,
                    ],
                    center_radius: [
                        finite_f32(pair.local_a.x)?,
                        finite_f32(pair.local_a.y)?,
                        finite_f32(pair.local_a.z)?,
                        finite_f32(pair.radius)?,
                    ],
                    center_of_mass: [
                        finite_f32(link.center_of_mass.x)?,
                        finite_f32(link.center_of_mass.y)?,
                        finite_f32(link.center_of_mass.z)?,
                        0.0,
                    ],
                    plane: [
                        finite_f32(center.x)?,
                        finite_f32(center.y)?,
                        finite_f32(center.z)?,
                        0.0,
                    ],
                    material: [
                        finite_f32(pair.restitution)?,
                        dt,
                        finite_f32(pair.friction)?,
                        21.0,
                    ],
                    other_center_radius: [
                        finite_f32(pair.half_extents.x)?,
                        finite_f32(pair.half_extents.y)?,
                        finite_f32(pair.half_extents.z)?,
                        0.0,
                    ],
                    first_axis_end: [
                        finite_f32(pair.local_b.x)?,
                        finite_f32(pair.local_b.y)?,
                        finite_f32(pair.local_b.z)?,
                        0.0,
                    ],
                    second_axis_end: [
                        finite_f32(rotation.i)?,
                        finite_f32(rotation.j)?,
                        finite_f32(rotation.k)?,
                        finite_f32(rotation.w)?,
                    ],
                    ..PackedSphere::zeroed()
                };
                spheres.push(packed);
                let mut side = packed;
                side.material[3] = 22.0;
                spheres.push(side);
            }
            for pair in &static_box_pairs[index] {
                validate_link_box(
                    articulation,
                    &GpuArticulatedLinkBox {
                        link: pair.link,
                        local_pose: pair.local_pose,
                        half_extents: pair.half_extents,
                        restitution: pair.restitution,
                        friction: pair.friction,
                    },
                )?;
                if pair
                    .static_half_extents
                    .iter()
                    .any(|value| !value.is_finite() || *value <= 0.0)
                    || pair
                        .static_pose
                        .translation
                        .vector
                        .iter()
                        .chain(pair.static_pose.rotation.coords.iter())
                        .any(|value| !value.is_finite())
                {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                let link = articulation
                    .link(pair.link)
                    .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
                let term_offset = checked_u32(
                    link_offset
                        .checked_add(
                            pair.link
                                .checked_mul(stride)
                                .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                        )
                        .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                )?;
                let first_center = pair.local_pose.translation.vector;
                let first_rotation = pair.local_pose.rotation.quaternion();
                let second_center = pair.static_pose.translation.vector;
                let second_rotation = pair.static_pose.rotation.quaternion();
                static_box_box_rows[index]
                    .push((spheres.len(), poses.link_ranges()[index].start + pair.link));
                static_box_box_poses[index].push([
                    finite_f32(second_center.x)?,
                    finite_f32(second_center.y)?,
                    finite_f32(second_center.z)?,
                    finite_f32(second_rotation.i)?,
                    finite_f32(second_rotation.j)?,
                    finite_f32(second_rotation.k)?,
                    finite_f32(second_rotation.w)?,
                ]);
                // Mode 23 uses the mode 8 box geometry with a world-space second box.
                let packed = PackedSphere {
                    indices: [
                        checked_u32(poses.link_ranges()[index].start + pair.link)?,
                        term_offset,
                        0,
                        0,
                    ],
                    center_radius: [
                        finite_f32(first_center.x)?,
                        finite_f32(first_center.y)?,
                        finite_f32(first_center.z)?,
                        finite_f32(pair.half_extents.x)?,
                    ],
                    center_of_mass: [
                        finite_f32(link.center_of_mass.x)?,
                        finite_f32(link.center_of_mass.y)?,
                        finite_f32(link.center_of_mass.z)?,
                        finite_f32(pair.half_extents.z)?,
                    ],
                    plane: [
                        finite_f32(second_center.x)?,
                        finite_f32(second_center.y)?,
                        finite_f32(second_center.z)?,
                        finite_f32(pair.half_extents.y)?,
                    ],
                    material: [
                        finite_f32(pair.restitution)?,
                        dt,
                        finite_f32(pair.friction)?,
                        23.0,
                    ],
                    other_center_radius: [
                        finite_f32(pair.static_half_extents.x)?,
                        finite_f32(pair.static_half_extents.y)?,
                        finite_f32(pair.static_half_extents.z)?,
                        0.0,
                    ],
                    second_axis_end: [
                        finite_f32(second_rotation.i)?,
                        finite_f32(second_rotation.j)?,
                        finite_f32(second_rotation.k)?,
                        finite_f32(second_rotation.w)?,
                    ],
                    first_axis_end: [
                        finite_f32(first_rotation.i)?,
                        finite_f32(first_rotation.j)?,
                        finite_f32(first_rotation.k)?,
                        finite_f32(first_rotation.w)?,
                    ],
                    ..PackedSphere::zeroed()
                };
                for row in 0..4 {
                    let mut contact = packed;
                    contact.other_center_radius[3] = row as f32;
                    spheres.push(contact);
                }
            }
            for pair in &static_box_capsule_pairs[index] {
                validate_link_box(
                    articulation,
                    &GpuArticulatedLinkBox {
                        link: pair.link,
                        local_pose: pair.local_pose,
                        half_extents: pair.half_extents,
                        restitution: pair.restitution,
                        friction: pair.friction,
                    },
                )?;
                if !pair.static_radius.is_finite()
                    || pair.static_radius < 0.0
                    || pair
                        .static_a
                        .iter()
                        .chain(pair.static_b.iter())
                        .any(|value| !value.is_finite())
                {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                let link = articulation
                    .link(pair.link)
                    .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
                let term_offset = checked_u32(
                    link_offset
                        .checked_add(
                            pair.link
                                .checked_mul(stride)
                                .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                        )
                        .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                )?;
                let center = pair.local_pose.translation.vector;
                let rotation = pair.local_pose.rotation.quaternion();
                let packed = PackedSphere {
                    indices: [
                        checked_u32(poses.link_ranges()[index].start + pair.link)?,
                        term_offset,
                        0,
                        0,
                    ],
                    center_radius: [
                        finite_f32(pair.static_a.x)?,
                        finite_f32(pair.static_a.y)?,
                        finite_f32(pair.static_a.z)?,
                        finite_f32(pair.static_radius)?,
                    ],
                    center_of_mass: [
                        finite_f32(link.center_of_mass.x)?,
                        finite_f32(link.center_of_mass.y)?,
                        finite_f32(link.center_of_mass.z)?,
                        0.0,
                    ],
                    plane: [
                        finite_f32(center.x)?,
                        finite_f32(center.y)?,
                        finite_f32(center.z)?,
                        0.0,
                    ],
                    material: [
                        finite_f32(pair.restitution)?,
                        dt,
                        finite_f32(pair.friction)?,
                        24.0,
                    ],
                    other_center_radius: [
                        finite_f32(pair.half_extents.x)?,
                        finite_f32(pair.half_extents.y)?,
                        finite_f32(pair.half_extents.z)?,
                        0.0,
                    ],
                    first_axis_end: [
                        finite_f32(pair.static_b.x)?,
                        finite_f32(pair.static_b.y)?,
                        finite_f32(pair.static_b.z)?,
                        0.0,
                    ],
                    second_axis_end: [
                        finite_f32(rotation.i)?,
                        finite_f32(rotation.j)?,
                        finite_f32(rotation.k)?,
                        finite_f32(rotation.w)?,
                    ],
                    ..PackedSphere::zeroed()
                };
                spheres.push(packed);
                let mut side = packed;
                side.material[3] = 25.0;
                spheres.push(side);
            }
            for pair in &static_axial_sphere_pairs[index] {
                let link = articulation
                    .link(pair.link)
                    .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
                if !pair.half_height.is_finite()
                    || pair.half_height <= 0.0
                    || !pair.radius.is_finite()
                    || pair.radius <= 0.0
                    || !pair.static_radius.is_finite()
                    || pair.static_radius <= 0.0
                    || !(0.0..=1.0).contains(&pair.restitution)
                    || !pair.friction.is_finite()
                    || pair.friction < 0.0
                    || pair
                        .local_pose
                        .translation
                        .vector
                        .iter()
                        .chain(pair.local_pose.rotation.coords.iter())
                        .chain(pair.static_center.iter())
                        .any(|value| !value.is_finite())
                {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                let term_offset = checked_u32(
                    link_offset
                        .checked_add(
                            pair.link
                                .checked_mul(stride)
                                .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                        )
                        .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                )?;
                let local = pair.local_pose.translation.vector;
                let rotation = pair.local_pose.rotation.quaternion();
                if !pair.axial_is_static {
                    static_axial_sphere_rows[index]
                        .push((spheres.len(), poses.link_ranges()[index].start + pair.link));
                    static_axial_sphere_centers[index].push([
                        finite_f32(pair.static_center.x)?,
                        finite_f32(pair.static_center.y)?,
                        finite_f32(pair.static_center.z)?,
                    ]);
                }
                spheres.push(PackedSphere {
                    indices: [
                        if pair.axial_is_static {
                            0
                        } else {
                            checked_u32(poses.link_ranges()[index].start + pair.link)?
                        },
                        if pair.axial_is_static { 0 } else { term_offset },
                        if pair.axial_is_static {
                            checked_u32(poses.link_ranges()[index].start + pair.link)?
                        } else {
                            0
                        },
                        if pair.axial_is_static { term_offset } else { 0 },
                    ],
                    center_radius: [
                        finite_f32(local.x)?,
                        finite_f32(local.y)?,
                        finite_f32(local.z)?,
                        finite_f32(pair.static_radius)?,
                    ],
                    center_of_mass: [
                        if pair.axial_is_static {
                            0.0
                        } else {
                            finite_f32(link.center_of_mass.x)?
                        },
                        if pair.axial_is_static {
                            0.0
                        } else {
                            finite_f32(link.center_of_mass.y)?
                        },
                        if pair.axial_is_static {
                            0.0
                        } else {
                            finite_f32(link.center_of_mass.z)?
                        },
                        0.0,
                    ],
                    plane: [
                        finite_f32(pair.static_center.x)?,
                        finite_f32(pair.static_center.y)?,
                        finite_f32(pair.static_center.z)?,
                        0.0,
                    ],
                    material: [
                        finite_f32(pair.restitution)?,
                        dt,
                        finite_f32(pair.friction)?,
                        match (pair.axial_is_static, pair.kind) {
                            (false, GpuArticulatedGroundAxialKind::Cylinder) => 26.0,
                            (false, GpuArticulatedGroundAxialKind::Cone) => 27.0,
                            (true, GpuArticulatedGroundAxialKind::Cylinder) => 60.0,
                            (true, GpuArticulatedGroundAxialKind::Cone) => 61.0,
                        },
                    ],
                    other_center_radius: [
                        finite_f32(pair.radius)?,
                        finite_f32(pair.half_height)?,
                        0.0,
                        0.0,
                    ],
                    second_axis_end: [
                        finite_f32(rotation.i)?,
                        finite_f32(rotation.j)?,
                        finite_f32(rotation.k)?,
                        finite_f32(rotation.w)?,
                    ],
                    other_center_of_mass: if pair.axial_is_static {
                        [
                            finite_f32(link.center_of_mass.x)?,
                            finite_f32(link.center_of_mass.y)?,
                            finite_f32(link.center_of_mass.z)?,
                            0.0,
                        ]
                    } else {
                        [0.0; 4]
                    },
                    ..PackedSphere::zeroed()
                });
            }
            for pair in &static_axial_capsule_pairs[index] {
                let link = articulation
                    .link(pair.link)
                    .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
                if !pair.half_height.is_finite()
                    || pair.half_height <= 0.0
                    || !pair.radius.is_finite()
                    || pair.radius <= 0.0
                    || !pair.static_radius.is_finite()
                    || pair.static_radius < 0.0
                    || !(0.0..=1.0).contains(&pair.restitution)
                    || !pair.friction.is_finite()
                    || pair.friction < 0.0
                    || pair
                        .local_pose
                        .translation
                        .vector
                        .iter()
                        .chain(pair.local_pose.rotation.coords.iter())
                        .chain(pair.static_a.iter())
                        .chain(pair.static_b.iter())
                        .any(|value| !value.is_finite())
                {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                let term_offset = checked_u32(
                    link_offset
                        .checked_add(
                            pair.link
                                .checked_mul(stride)
                                .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                        )
                        .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                )?;
                let local = pair.local_pose.translation.vector;
                let rotation = pair.local_pose.rotation.quaternion();
                spheres.push(PackedSphere {
                    indices: [
                        if pair.axial_is_static {
                            0
                        } else {
                            checked_u32(poses.link_ranges()[index].start + pair.link)?
                        },
                        if pair.axial_is_static { 0 } else { term_offset },
                        if pair.axial_is_static {
                            checked_u32(poses.link_ranges()[index].start + pair.link)?
                        } else {
                            0
                        },
                        if pair.axial_is_static { term_offset } else { 0 },
                    ],
                    center_radius: [
                        finite_f32(local.x)?,
                        finite_f32(local.y)?,
                        finite_f32(local.z)?,
                        finite_f32(pair.static_radius)?,
                    ],
                    center_of_mass: [
                        if pair.axial_is_static {
                            0.0
                        } else {
                            finite_f32(link.center_of_mass.x)?
                        },
                        if pair.axial_is_static {
                            0.0
                        } else {
                            finite_f32(link.center_of_mass.y)?
                        },
                        if pair.axial_is_static {
                            0.0
                        } else {
                            finite_f32(link.center_of_mass.z)?
                        },
                        0.0,
                    ],
                    plane: [
                        finite_f32(pair.static_a.x)?,
                        finite_f32(pair.static_a.y)?,
                        finite_f32(pair.static_a.z)?,
                        0.0,
                    ],
                    material: [
                        finite_f32(pair.restitution)?,
                        dt,
                        finite_f32(pair.friction)?,
                        match (pair.axial_is_static, pair.kind) {
                            (false, GpuArticulatedGroundAxialKind::Cylinder) => 28.0,
                            (false, GpuArticulatedGroundAxialKind::Cone) => 29.0,
                            (true, GpuArticulatedGroundAxialKind::Cylinder) => 62.0,
                            (true, GpuArticulatedGroundAxialKind::Cone) => 63.0,
                        },
                    ],
                    other_center_radius: [
                        finite_f32(pair.radius)?,
                        finite_f32(pair.half_height)?,
                        0.0,
                        0.0,
                    ],
                    first_axis_end: [
                        finite_f32(pair.static_b.x)?,
                        finite_f32(pair.static_b.y)?,
                        finite_f32(pair.static_b.z)?,
                        0.0,
                    ],
                    second_axis_end: [
                        finite_f32(rotation.i)?,
                        finite_f32(rotation.j)?,
                        finite_f32(rotation.k)?,
                        finite_f32(rotation.w)?,
                    ],
                    other_center_of_mass: if pair.axial_is_static {
                        [
                            finite_f32(link.center_of_mass.x)?,
                            finite_f32(link.center_of_mass.y)?,
                            finite_f32(link.center_of_mass.z)?,
                            0.0,
                        ]
                    } else {
                        [0.0; 4]
                    },
                    ..PackedSphere::zeroed()
                });
            }
            for pair in &static_axial_box_pairs[index] {
                let link = articulation
                    .link(pair.link)
                    .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
                if !pair.half_height.is_finite()
                    || pair.half_height <= 0.0
                    || !pair.radius.is_finite()
                    || pair.radius <= 0.0
                    || pair
                        .static_half_extents
                        .iter()
                        .any(|value| !value.is_finite() || *value <= 0.0)
                    || !(0.0..=1.0).contains(&pair.restitution)
                    || !pair.friction.is_finite()
                    || pair.friction < 0.0
                    || pair
                        .local_pose
                        .translation
                        .vector
                        .iter()
                        .chain(pair.local_pose.rotation.coords.iter())
                        .chain(pair.static_pose.translation.vector.iter())
                        .chain(pair.static_pose.rotation.coords.iter())
                        .any(|value| !value.is_finite())
                {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                let term_offset = checked_u32(
                    link_offset
                        .checked_add(
                            pair.link
                                .checked_mul(stride)
                                .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                        )
                        .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                )?;
                let local = pair.local_pose.translation.vector;
                let local_rotation = pair.local_pose.rotation.quaternion();
                let static_center = pair.static_pose.translation.vector;
                let static_rotation = pair.static_pose.rotation.quaternion();
                if !pair.axial_is_static {
                    static_axial_box_rows[index]
                        .push((spheres.len(), poses.link_ranges()[index].start + pair.link));
                    static_axial_box_poses[index].push([
                        finite_f32(static_center.x)?,
                        finite_f32(static_center.y)?,
                        finite_f32(static_center.z)?,
                        finite_f32(static_rotation.i)?,
                        finite_f32(static_rotation.j)?,
                        finite_f32(static_rotation.k)?,
                        finite_f32(static_rotation.w)?,
                    ]);
                }
                spheres.push(PackedSphere {
                    indices: [
                        if pair.axial_is_static {
                            0
                        } else {
                            checked_u32(poses.link_ranges()[index].start + pair.link)?
                        },
                        if pair.axial_is_static { 0 } else { term_offset },
                        if pair.axial_is_static {
                            checked_u32(poses.link_ranges()[index].start + pair.link)?
                        } else {
                            0
                        },
                        if pair.axial_is_static { term_offset } else { 0 },
                    ],
                    center_radius: [
                        finite_f32(local.x)?,
                        finite_f32(local.y)?,
                        finite_f32(local.z)?,
                        finite_f32(pair.radius)?,
                    ],
                    center_of_mass: [
                        if pair.axial_is_static {
                            0.0
                        } else {
                            finite_f32(link.center_of_mass.x)?
                        },
                        if pair.axial_is_static {
                            0.0
                        } else {
                            finite_f32(link.center_of_mass.y)?
                        },
                        if pair.axial_is_static {
                            0.0
                        } else {
                            finite_f32(link.center_of_mass.z)?
                        },
                        0.0,
                    ],
                    plane: [
                        finite_f32(static_center.x)?,
                        finite_f32(static_center.y)?,
                        finite_f32(static_center.z)?,
                        finite_f32(pair.half_height)?,
                    ],
                    material: [
                        finite_f32(pair.restitution)?,
                        dt,
                        finite_f32(pair.friction)?,
                        match (pair.axial_is_static, pair.kind) {
                            (false, GpuArticulatedGroundAxialKind::Cylinder) => 30.0,
                            (false, GpuArticulatedGroundAxialKind::Cone) => 31.0,
                            (true, GpuArticulatedGroundAxialKind::Cylinder) => 64.0,
                            (true, GpuArticulatedGroundAxialKind::Cone) => 65.0,
                        },
                    ],
                    other_center_radius: [
                        finite_f32(pair.static_half_extents.x)?,
                        finite_f32(pair.static_half_extents.y)?,
                        finite_f32(pair.static_half_extents.z)?,
                        0.0,
                    ],
                    first_axis_end: [
                        finite_f32(local_rotation.i)?,
                        finite_f32(local_rotation.j)?,
                        finite_f32(local_rotation.k)?,
                        finite_f32(local_rotation.w)?,
                    ],
                    second_axis_end: [
                        finite_f32(static_rotation.i)?,
                        finite_f32(static_rotation.j)?,
                        finite_f32(static_rotation.k)?,
                        finite_f32(static_rotation.w)?,
                    ],
                    other_center_of_mass: if pair.axial_is_static {
                        [
                            finite_f32(link.center_of_mass.x)?,
                            finite_f32(link.center_of_mass.y)?,
                            finite_f32(link.center_of_mass.z)?,
                            0.0,
                        ]
                    } else {
                        [0.0; 4]
                    },
                    ..PackedSphere::zeroed()
                });
            }
            for pair in &axial_sphere_pairs[index] {
                let axial_link = articulation
                    .link(pair.axial_link)
                    .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
                let sphere_link = articulation
                    .link(pair.sphere_link)
                    .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
                if pair.axial_link == pair.sphere_link
                    || !pair.half_height.is_finite()
                    || pair.half_height <= 0.0
                    || !pair.radius.is_finite()
                    || pair.radius <= 0.0
                    || !pair.sphere_radius.is_finite()
                    || pair.sphere_radius <= 0.0
                    || !(0.0..=1.0).contains(&pair.restitution)
                    || !pair.friction.is_finite()
                    || pair.friction < 0.0
                    || pair
                        .axial_local_pose
                        .translation
                        .vector
                        .iter()
                        .chain(pair.axial_local_pose.rotation.coords.iter())
                        .chain(pair.sphere_local_center.iter())
                        .any(|value| !value.is_finite())
                {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                let term_offset = |link: usize| -> Result<u32, GpuArticulatedGroundContactError> {
                    checked_u32(
                        link_offset
                            .checked_add(
                                link.checked_mul(stride)
                                    .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                            )
                            .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                    )
                };
                let local = pair.axial_local_pose.translation.vector;
                let rotation = pair.axial_local_pose.rotation.quaternion();
                spheres.push(PackedSphere {
                    indices: [
                        checked_u32(poses.link_ranges()[index].start + pair.axial_link)?,
                        term_offset(pair.axial_link)?,
                        checked_u32(poses.link_ranges()[index].start + pair.sphere_link)?,
                        term_offset(pair.sphere_link)?,
                    ],
                    center_radius: [
                        finite_f32(local.x)?,
                        finite_f32(local.y)?,
                        finite_f32(local.z)?,
                        finite_f32(pair.sphere_radius)?,
                    ],
                    center_of_mass: [
                        finite_f32(axial_link.center_of_mass.x)?,
                        finite_f32(axial_link.center_of_mass.y)?,
                        finite_f32(axial_link.center_of_mass.z)?,
                        0.0,
                    ],
                    plane: [
                        finite_f32(pair.sphere_local_center.x)?,
                        finite_f32(pair.sphere_local_center.y)?,
                        finite_f32(pair.sphere_local_center.z)?,
                        0.0,
                    ],
                    material: [
                        finite_f32(pair.restitution)?,
                        dt,
                        finite_f32(pair.friction)?,
                        match pair.kind {
                            GpuArticulatedGroundAxialKind::Cylinder => 32.0,
                            GpuArticulatedGroundAxialKind::Cone => 33.0,
                        },
                    ],
                    other_center_radius: [
                        finite_f32(pair.radius)?,
                        finite_f32(pair.half_height)?,
                        0.0,
                        0.0,
                    ],
                    other_center_of_mass: [
                        finite_f32(sphere_link.center_of_mass.x)?,
                        finite_f32(sphere_link.center_of_mass.y)?,
                        finite_f32(sphere_link.center_of_mass.z)?,
                        0.0,
                    ],
                    second_axis_end: [
                        finite_f32(rotation.i)?,
                        finite_f32(rotation.j)?,
                        finite_f32(rotation.k)?,
                        finite_f32(rotation.w)?,
                    ],
                    ..PackedSphere::zeroed()
                });
            }
            for pair in &axial_capsule_pairs[index] {
                let axial_link = articulation
                    .link(pair.axial_link)
                    .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
                let capsule_link = articulation
                    .link(pair.capsule_link)
                    .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
                if pair.axial_link == pair.capsule_link
                    || !pair.half_height.is_finite()
                    || pair.half_height <= 0.0
                    || !pair.radius.is_finite()
                    || pair.radius <= 0.0
                    || !pair.capsule_radius.is_finite()
                    || pair.capsule_radius <= 0.0
                    || !(0.0..=1.0).contains(&pair.restitution)
                    || !pair.friction.is_finite()
                    || pair.friction < 0.0
                    || pair
                        .axial_local_pose
                        .translation
                        .vector
                        .iter()
                        .chain(pair.axial_local_pose.rotation.coords.iter())
                        .chain(pair.capsule_local_a.iter())
                        .chain(pair.capsule_local_b.iter())
                        .any(|value| !value.is_finite())
                {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                let term_offset = |link: usize| -> Result<u32, GpuArticulatedGroundContactError> {
                    checked_u32(
                        link_offset
                            .checked_add(
                                link.checked_mul(stride)
                                    .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                            )
                            .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                    )
                };
                let local = pair.axial_local_pose.translation.vector;
                let rotation = pair.axial_local_pose.rotation.quaternion();
                spheres.push(PackedSphere {
                    indices: [
                        checked_u32(poses.link_ranges()[index].start + pair.axial_link)?,
                        term_offset(pair.axial_link)?,
                        checked_u32(poses.link_ranges()[index].start + pair.capsule_link)?,
                        term_offset(pair.capsule_link)?,
                    ],
                    center_radius: [
                        finite_f32(local.x)?,
                        finite_f32(local.y)?,
                        finite_f32(local.z)?,
                        finite_f32(pair.capsule_radius)?,
                    ],
                    center_of_mass: [
                        finite_f32(axial_link.center_of_mass.x)?,
                        finite_f32(axial_link.center_of_mass.y)?,
                        finite_f32(axial_link.center_of_mass.z)?,
                        0.0,
                    ],
                    plane: [
                        finite_f32(pair.capsule_local_a.x)?,
                        finite_f32(pair.capsule_local_a.y)?,
                        finite_f32(pair.capsule_local_a.z)?,
                        0.0,
                    ],
                    material: [
                        finite_f32(pair.restitution)?,
                        dt,
                        finite_f32(pair.friction)?,
                        match pair.kind {
                            GpuArticulatedGroundAxialKind::Cylinder => 36.0,
                            GpuArticulatedGroundAxialKind::Cone => 37.0,
                        },
                    ],
                    other_center_radius: [
                        finite_f32(pair.radius)?,
                        finite_f32(pair.half_height)?,
                        0.0,
                        0.0,
                    ],
                    other_center_of_mass: [
                        finite_f32(capsule_link.center_of_mass.x)?,
                        finite_f32(capsule_link.center_of_mass.y)?,
                        finite_f32(capsule_link.center_of_mass.z)?,
                        0.0,
                    ],
                    first_axis_end: [
                        finite_f32(pair.capsule_local_b.x)?,
                        finite_f32(pair.capsule_local_b.y)?,
                        finite_f32(pair.capsule_local_b.z)?,
                        0.0,
                    ],
                    second_axis_end: [
                        finite_f32(rotation.i)?,
                        finite_f32(rotation.j)?,
                        finite_f32(rotation.k)?,
                        finite_f32(rotation.w)?,
                    ],
                    ..PackedSphere::zeroed()
                });
            }
            for pair in &axial_pairs[index] {
                let first_center_of_mass = if pair.first_is_static {
                    Vector3::zeros()
                } else {
                    articulation
                        .link(pair.first_link)
                        .ok_or(GpuArticulatedGroundContactError::InvalidInput)?
                        .center_of_mass
                };
                let second_link = articulation
                    .link(pair.second_link)
                    .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
                if (!pair.first_is_static && pair.first_link == pair.second_link)
                    || !pair.first_half_height.is_finite()
                    || pair.first_half_height <= 0.0
                    || !pair.first_radius.is_finite()
                    || pair.first_radius <= 0.0
                    || !pair.second_half_height.is_finite()
                    || pair.second_half_height <= 0.0
                    || !pair.second_radius.is_finite()
                    || pair.second_radius <= 0.0
                    || !(0.0..=1.0).contains(&pair.restitution)
                    || !pair.friction.is_finite()
                    || pair.friction < 0.0
                    || pair
                        .first_local_pose
                        .translation
                        .vector
                        .iter()
                        .chain(pair.first_local_pose.rotation.coords.iter())
                        .chain(pair.second_local_pose.translation.vector.iter())
                        .chain(pair.second_local_pose.rotation.coords.iter())
                        .any(|value| !value.is_finite())
                {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                let term_offset = |link: usize| -> Result<u32, GpuArticulatedGroundContactError> {
                    checked_u32(
                        link_offset
                            .checked_add(
                                link.checked_mul(stride)
                                    .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                            )
                            .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                    )
                };
                let first_local = pair.first_local_pose.translation.vector;
                let first_rotation = pair.first_local_pose.rotation.quaternion();
                let second_local = pair.second_local_pose.translation.vector;
                let second_rotation = pair.second_local_pose.rotation.quaternion();
                spheres.push(PackedSphere {
                    indices: [
                        if pair.first_is_static {
                            0
                        } else {
                            checked_u32(poses.link_ranges()[index].start + pair.first_link)?
                        },
                        if pair.first_is_static {
                            0
                        } else {
                            term_offset(pair.first_link)?
                        },
                        checked_u32(poses.link_ranges()[index].start + pair.second_link)?,
                        term_offset(pair.second_link)?,
                    ],
                    center_radius: [
                        finite_f32(first_local.x)?,
                        finite_f32(first_local.y)?,
                        finite_f32(first_local.z)?,
                        finite_f32(pair.first_radius)?,
                    ],
                    center_of_mass: [
                        finite_f32(first_center_of_mass.x)?,
                        finite_f32(first_center_of_mass.y)?,
                        finite_f32(first_center_of_mass.z)?,
                        0.0,
                    ],
                    plane: [
                        finite_f32(second_local.x)?,
                        finite_f32(second_local.y)?,
                        finite_f32(second_local.z)?,
                        finite_f32(pair.first_half_height)?,
                    ],
                    material: [
                        finite_f32(pair.restitution)?,
                        dt,
                        finite_f32(pair.friction)?,
                        match (pair.first_is_static, pair.first_kind, pair.second_kind) {
                            (
                                false,
                                GpuArticulatedGroundAxialKind::Cylinder,
                                GpuArticulatedGroundAxialKind::Cylinder,
                            ) => 38.0,
                            (
                                false,
                                GpuArticulatedGroundAxialKind::Cylinder,
                                GpuArticulatedGroundAxialKind::Cone,
                            ) => 39.0,
                            (
                                false,
                                GpuArticulatedGroundAxialKind::Cone,
                                GpuArticulatedGroundAxialKind::Cylinder,
                            ) => 40.0,
                            (
                                false,
                                GpuArticulatedGroundAxialKind::Cone,
                                GpuArticulatedGroundAxialKind::Cone,
                            ) => 41.0,
                            (
                                true,
                                GpuArticulatedGroundAxialKind::Cylinder,
                                GpuArticulatedGroundAxialKind::Cylinder,
                            ) => 66.0,
                            (
                                true,
                                GpuArticulatedGroundAxialKind::Cylinder,
                                GpuArticulatedGroundAxialKind::Cone,
                            ) => 67.0,
                            (
                                true,
                                GpuArticulatedGroundAxialKind::Cone,
                                GpuArticulatedGroundAxialKind::Cylinder,
                            ) => 68.0,
                            (
                                true,
                                GpuArticulatedGroundAxialKind::Cone,
                                GpuArticulatedGroundAxialKind::Cone,
                            ) => 69.0,
                        },
                    ],
                    other_center_radius: [
                        finite_f32(pair.second_radius)?,
                        finite_f32(pair.second_half_height)?,
                        0.0,
                        0.0,
                    ],
                    other_center_of_mass: [
                        finite_f32(second_link.center_of_mass.x)?,
                        finite_f32(second_link.center_of_mass.y)?,
                        finite_f32(second_link.center_of_mass.z)?,
                        0.0,
                    ],
                    first_axis_end: [
                        finite_f32(first_rotation.i)?,
                        finite_f32(first_rotation.j)?,
                        finite_f32(first_rotation.k)?,
                        finite_f32(first_rotation.w)?,
                    ],
                    second_axis_end: [
                        finite_f32(second_rotation.i)?,
                        finite_f32(second_rotation.j)?,
                        finite_f32(second_rotation.k)?,
                        finite_f32(second_rotation.w)?,
                    ],
                    ..PackedSphere::zeroed()
                });
            }
            struct ConvexRoundedPairRef<'a> {
                convex_link: Option<usize>,
                convex_local_pose: Isometry3<f64>,
                vertices: &'a [Vector3<f64>],
                face_normals: &'a [Vector3<f64>],
                edge_directions: &'a [Vector3<f64>],
                sphere_link: Option<usize>,
                sphere_local_center: Vector3<f64>,
                capsule_local_b: Vector3<f64>,
                sphere_radius: f64,
                restitution: f64,
                friction: f64,
                mode: f32,
            }
            let rounded_pairs =
                convex_sphere_pairs[index]
                    .iter()
                    .map(|pair| ConvexRoundedPairRef {
                        convex_link: Some(pair.convex_link),
                        convex_local_pose: pair.convex_local_pose,
                        vertices: &pair.vertices,
                        face_normals: &pair.face_normals,
                        edge_directions: &[],
                        sphere_link: Some(pair.sphere_link),
                        sphere_local_center: pair.sphere_local_center,
                        capsule_local_b: pair.sphere_local_center,
                        sphere_radius: pair.sphere_radius,
                        restitution: pair.restitution,
                        friction: pair.friction,
                        mode: 42.0,
                    })
                    .chain(
                        convex_capsule_pairs[index]
                            .iter()
                            .map(|pair| ConvexRoundedPairRef {
                                convex_link: Some(pair.convex_link),
                                convex_local_pose: pair.convex_local_pose,
                                vertices: &pair.vertices,
                                face_normals: &pair.face_normals,
                                edge_directions: &pair.edge_directions,
                                sphere_link: Some(pair.capsule_link),
                                sphere_local_center: pair.capsule_local_a,
                                capsule_local_b: pair.capsule_local_b,
                                sphere_radius: pair.capsule_radius,
                                restitution: pair.restitution,
                                friction: pair.friction,
                                mode: 43.0,
                            }),
                    )
                    .chain(static_convex_sphere_pairs[index].iter().map(|pair| {
                        ConvexRoundedPairRef {
                            convex_link: Some(pair.convex_link),
                            convex_local_pose: pair.convex_local_pose,
                            vertices: &pair.vertices,
                            face_normals: &pair.face_normals,
                            edge_directions: &[],
                            sphere_link: None,
                            sphere_local_center: pair.static_center,
                            capsule_local_b: pair.static_center,
                            sphere_radius: pair.static_radius,
                            restitution: pair.restitution,
                            friction: pair.friction,
                            mode: 45.0,
                        }
                    }))
                    .chain(static_convex_capsule_pairs[index].iter().map(|pair| {
                        ConvexRoundedPairRef {
                            convex_link: Some(pair.convex_link),
                            convex_local_pose: pair.convex_local_pose,
                            vertices: &pair.vertices,
                            face_normals: &pair.face_normals,
                            edge_directions: &pair.edge_directions,
                            sphere_link: None,
                            sphere_local_center: pair.static_a,
                            capsule_local_b: pair.static_b,
                            sphere_radius: pair.static_radius,
                            restitution: pair.restitution,
                            friction: pair.friction,
                            mode: 46.0,
                        }
                    }))
                    .chain(scene_convex_sphere_pairs[index].iter().map(|pair| {
                        ConvexRoundedPairRef {
                            convex_link: None,
                            convex_local_pose: pair.convex_world_pose,
                            vertices: &pair.vertices,
                            face_normals: &pair.face_normals,
                            edge_directions: &[],
                            sphere_link: Some(pair.sphere_link),
                            sphere_local_center: pair.sphere_local_center,
                            capsule_local_b: pair.sphere_local_center,
                            sphere_radius: pair.sphere_radius,
                            restitution: pair.restitution,
                            friction: pair.friction,
                            mode: 48.0,
                        }
                    }))
                    .chain(scene_convex_capsule_pairs[index].iter().map(|pair| {
                        ConvexRoundedPairRef {
                            convex_link: None,
                            convex_local_pose: pair.convex_world_pose,
                            vertices: &pair.vertices,
                            face_normals: &pair.face_normals,
                            edge_directions: &pair.edge_directions,
                            sphere_link: Some(pair.capsule_link),
                            sphere_local_center: pair.capsule_local_a,
                            capsule_local_b: pair.capsule_local_b,
                            sphere_radius: pair.capsule_radius,
                            restitution: pair.restitution,
                            friction: pair.friction,
                            mode: 49.0,
                        }
                    }));
            for pair in rounded_pairs {
                let convex_link = pair.convex_link.and_then(|link| articulation.link(link));
                let sphere_link = pair.sphere_link.and_then(|link| articulation.link(link));
                if pair.convex_link == pair.sphere_link
                    || (pair.convex_link.is_some() && convex_link.is_none())
                    || (pair.sphere_link.is_some() && sphere_link.is_none())
                    || (pair.convex_link.is_none() && pair.sphere_link.is_none())
                    || pair.vertices.len() < 4
                    || pair.face_normals.len() < 4
                    || !pair.sphere_radius.is_finite()
                    || (pair.sphere_radius <= 0.0
                        && !(pair.mode == 46.0 && pair.sphere_radius == 0.0))
                    || !(0.0..=1.0).contains(&pair.restitution)
                    || !pair.friction.is_finite()
                    || pair.friction < 0.0
                    || pair
                        .convex_local_pose
                        .translation
                        .vector
                        .iter()
                        .chain(pair.convex_local_pose.rotation.coords.iter())
                        .chain(pair.sphere_local_center.iter())
                        .chain(pair.capsule_local_b.iter())
                        .chain(pair.vertices.iter().flat_map(|point| point.iter()))
                        .chain(pair.face_normals.iter().flat_map(|normal| normal.iter()))
                        .chain(pair.edge_directions.iter().flat_map(|edge| edge.iter()))
                        .any(|value| !value.is_finite())
                    || pair
                        .face_normals
                        .iter()
                        .chain(pair.edge_directions.iter())
                        .any(|normal| (normal.norm_squared() - 1.0).abs() > 1e-4)
                    || ((pair.mode == 43.0 || pair.mode == 46.0 || pair.mode == 49.0)
                        && pair.edge_directions.len() < 3)
                {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                let indices_as_f32 =
                    |value: usize| -> Result<f32, GpuArticulatedGroundContactError> {
                        if value > (1 << 24) {
                            return Err(GpuArticulatedGroundContactError::Capacity);
                        }
                        Ok(checked_u32(value)? as f32)
                    };
                let vertex_start = indices_as_f32(convex_geometry.len())?;
                for vertex in pair.vertices {
                    convex_geometry.push(PackedSphere {
                        center_radius: [
                            finite_f32(vertex.x)?,
                            finite_f32(vertex.y)?,
                            finite_f32(vertex.z)?,
                            0.0,
                        ],
                        ..PackedSphere::zeroed()
                    });
                }
                let normal_start = indices_as_f32(convex_geometry.len())?;
                for normal in pair.face_normals {
                    convex_geometry.push(PackedSphere {
                        center_radius: [
                            finite_f32(normal.x)?,
                            finite_f32(normal.y)?,
                            finite_f32(normal.z)?,
                            0.0,
                        ],
                        ..PackedSphere::zeroed()
                    });
                }
                let edge_start = indices_as_f32(convex_geometry.len())?;
                for edge in pair.edge_directions {
                    convex_geometry.push(PackedSphere {
                        center_radius: [
                            finite_f32(edge.x)?,
                            finite_f32(edge.y)?,
                            finite_f32(edge.z)?,
                            0.0,
                        ],
                        ..PackedSphere::zeroed()
                    });
                }
                let term_offset = |link: usize| -> Result<u32, GpuArticulatedGroundContactError> {
                    checked_u32(
                        link_offset
                            .checked_add(
                                link.checked_mul(stride)
                                    .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                            )
                            .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                    )
                };
                let local = pair.convex_local_pose.translation.vector;
                let rotation = pair.convex_local_pose.rotation.quaternion();
                convex_contact_rows.push(spheres.len());
                if pair.mode == 48.0 || pair.mode == 49.0 {
                    let owner = pair
                        .sphere_link
                        .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
                    scene_convex_rounded_rows[index].push((
                        spheres.len(),
                        poses.link_ranges()[index].start + owner,
                        if pair.mode == 49.0 { 2 } else { 1 },
                    ));
                    scene_convex_rounded_poses[index].push([
                        finite_f32(local.x)?,
                        finite_f32(local.y)?,
                        finite_f32(local.z)?,
                        finite_f32(rotation.i)?,
                        finite_f32(rotation.j)?,
                        finite_f32(rotation.k)?,
                        finite_f32(rotation.w)?,
                    ]);
                }
                if pair.mode == 45.0 {
                    let owner = pair
                        .convex_link
                        .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
                    static_convex_sphere_rows[index]
                        .push((spheres.len(), poses.link_ranges()[index].start + owner));
                    static_convex_sphere_centers[index].push([
                        finite_f32(pair.sphere_local_center.x)?,
                        finite_f32(pair.sphere_local_center.y)?,
                        finite_f32(pair.sphere_local_center.z)?,
                    ]);
                }
                let packed = PackedSphere {
                    indices: [
                        pair.convex_link
                            .map(|link| checked_u32(poses.link_ranges()[index].start + link))
                            .transpose()?
                            .unwrap_or(0),
                        pair.convex_link.map(term_offset).transpose()?.unwrap_or(0),
                        pair.sphere_link
                            .map(|link| checked_u32(poses.link_ranges()[index].start + link))
                            .transpose()?
                            .unwrap_or(0),
                        pair.sphere_link.map(term_offset).transpose()?.unwrap_or(0),
                    ],
                    center_radius: [
                        finite_f32(local.x)?,
                        finite_f32(local.y)?,
                        finite_f32(local.z)?,
                        finite_f32(pair.sphere_radius)?,
                    ],
                    center_of_mass: [
                        finite_f32(convex_link.map_or(0.0, |link| link.center_of_mass.x))?,
                        finite_f32(convex_link.map_or(0.0, |link| link.center_of_mass.y))?,
                        finite_f32(convex_link.map_or(0.0, |link| link.center_of_mass.z))?,
                        0.0,
                    ],
                    plane: [
                        finite_f32(pair.sphere_local_center.x)?,
                        finite_f32(pair.sphere_local_center.y)?,
                        finite_f32(pair.sphere_local_center.z)?,
                        vertex_start,
                    ],
                    material: [
                        finite_f32(pair.restitution)?,
                        dt,
                        finite_f32(pair.friction)?,
                        pair.mode,
                    ],
                    other_center_radius: [
                        indices_as_f32(pair.vertices.len())?,
                        normal_start,
                        indices_as_f32(pair.face_normals.len())?,
                        indices_as_f32(pair.edge_directions.len())?,
                    ],
                    other_center_of_mass: [
                        finite_f32(sphere_link.map_or(0.0, |link| link.center_of_mass.x))?,
                        finite_f32(sphere_link.map_or(0.0, |link| link.center_of_mass.y))?,
                        finite_f32(sphere_link.map_or(0.0, |link| link.center_of_mass.z))?,
                        0.0,
                    ],
                    first_axis_end: [
                        finite_f32(rotation.i)?,
                        finite_f32(rotation.j)?,
                        finite_f32(rotation.k)?,
                        finite_f32(rotation.w)?,
                    ],
                    second_axis_end: [
                        finite_f32(pair.capsule_local_b.x)?,
                        finite_f32(pair.capsule_local_b.y)?,
                        finite_f32(pair.capsule_local_b.z)?,
                        edge_start,
                    ],
                    ..PackedSphere::zeroed()
                };
                spheres.push(packed);
                if matches!(pair.mode, 43.0 | 46.0 | 49.0) {
                    let mut second = packed;
                    second.material[3] = match pair.mode {
                        43.0 => 82.0,
                        49.0 => 83.0,
                        _ => 81.0,
                    };
                    convex_contact_rows.push(spheres.len());
                    spheres.push(second);
                }
            }
            for pair in &scene_mesh_sphere_pairs[index] {
                let Some(link) = articulation.link(pair.sphere_link) else {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                };
                if !pair.sphere_radius.is_finite()
                    || pair.sphere_radius <= 0.0
                    || !(0.0..=1.0).contains(&pair.restitution)
                    || !pair.friction.is_finite()
                    || pair.friction < 0.0
                {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                let offsets = pack_mesh_geometry(
                    &pair.mesh,
                    &mut convex_geometry,
                    &mut mesh_geometry_offsets,
                )?;
                let translation = pair.mesh_world_pose.translation.vector;
                let rotation = pair.mesh_world_pose.rotation.quaternion();
                let center = pair.sphere_local_center;
                for manifold_slot in 0..4 {
                    convex_contact_rows.push(spheres.len());
                    spheres.push(PackedSphere {
                        indices: [
                            checked_u32(poses.link_ranges()[index].start + pair.sphere_link)?,
                            checked_u32(
                                link_offset
                                    .checked_add(
                                        pair.sphere_link
                                            .checked_mul(stride)
                                            .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                                    )
                                    .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                            )?,
                            0,
                            0,
                        ],
                        center_radius: [
                            finite_f32(center.x)?,
                            finite_f32(center.y)?,
                            finite_f32(center.z)?,
                            finite_f32(pair.sphere_radius)?,
                        ],
                        center_of_mass: [
                            finite_f32(link.center_of_mass.x)?,
                            finite_f32(link.center_of_mass.y)?,
                            finite_f32(link.center_of_mass.z)?,
                            0.0,
                        ],
                        plane: [
                            finite_f32(translation.x)?,
                            finite_f32(translation.y)?,
                            finite_f32(translation.z)?,
                            0.0,
                        ],
                        material: [
                            finite_f32(pair.restitution)?,
                            dt,
                            finite_f32(pair.friction)?,
                            52.0,
                        ],
                        other_center_radius: offsets,
                        first_axis_end: [
                            finite_f32(rotation.i)?,
                            finite_f32(rotation.j)?,
                            finite_f32(rotation.k)?,
                            finite_f32(rotation.w)?,
                        ],
                        second_axis_end: [0.0, 0.0, 0.0, manifold_slot as f32],
                        ..PackedSphere::zeroed()
                    });
                }
            }
            for pair in &scene_polyline_sphere_pairs[index] {
                let Some(link) = articulation.link(pair.sphere_link) else {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                };
                if !pair.sphere_radius.is_finite()
                    || pair.sphere_radius <= 0.0
                    || !(0.0..=1.0).contains(&pair.restitution)
                    || !pair.friction.is_finite()
                    || pair.friction < 0.0
                {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                let offsets = pack_polyline_geometry(
                    &pair.polyline,
                    &mut convex_geometry,
                    &mut mesh_geometry_offsets,
                )?;
                let translation = pair.polyline_world_pose.translation.vector;
                let rotation = pair.polyline_world_pose.rotation.quaternion();
                let center = pair.sphere_local_center;
                for manifold_slot in 0..4 {
                    convex_contact_rows.push(spheres.len());
                    spheres.push(PackedSphere {
                        indices: [
                            checked_u32(poses.link_ranges()[index].start + pair.sphere_link)?,
                            checked_u32(
                                link_offset
                                    .checked_add(
                                        pair.sphere_link
                                            .checked_mul(stride)
                                            .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                                    )
                                    .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                            )?,
                            0,
                            0,
                        ],
                        center_radius: [
                            finite_f32(center.x)?,
                            finite_f32(center.y)?,
                            finite_f32(center.z)?,
                            finite_f32(pair.sphere_radius)?,
                        ],
                        center_of_mass: [
                            finite_f32(link.center_of_mass.x)?,
                            finite_f32(link.center_of_mass.y)?,
                            finite_f32(link.center_of_mass.z)?,
                            0.0,
                        ],
                        plane: [
                            finite_f32(translation.x)?,
                            finite_f32(translation.y)?,
                            finite_f32(translation.z)?,
                            0.0,
                        ],
                        material: [
                            finite_f32(pair.restitution)?,
                            dt,
                            finite_f32(pair.friction)?,
                            72.0,
                        ],
                        other_center_radius: offsets,
                        first_axis_end: [
                            finite_f32(rotation.i)?,
                            finite_f32(rotation.j)?,
                            finite_f32(rotation.k)?,
                            finite_f32(rotation.w)?,
                        ],
                        second_axis_end: [0.0, 0.0, 0.0, manifold_slot as f32],
                        ..PackedSphere::zeroed()
                    });
                }
            }
            for pair in &scene_mesh_capsule_pairs[index] {
                let Some(link) = articulation.link(pair.capsule_link) else {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                };
                if !pair.capsule_radius.is_finite()
                    || pair.capsule_radius <= 0.0
                    || !(0.0..=1.0).contains(&pair.restitution)
                    || !pair.friction.is_finite()
                    || pair.friction < 0.0
                {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                let offsets = pack_mesh_geometry(
                    &pair.mesh,
                    &mut convex_geometry,
                    &mut mesh_geometry_offsets,
                )?;
                let translation = pair.mesh_world_pose.translation.vector;
                let rotation = pair.mesh_world_pose.rotation.quaternion();
                let local_a = pair.capsule_local_a;
                let local_b = pair.capsule_local_b;
                for manifold_slot in 0..4 {
                    convex_contact_rows.push(spheres.len());
                    spheres.push(PackedSphere {
                        indices: [
                            checked_u32(poses.link_ranges()[index].start + pair.capsule_link)?,
                            checked_u32(
                                link_offset
                                    .checked_add(
                                        pair.capsule_link
                                            .checked_mul(stride)
                                            .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                                    )
                                    .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                            )?,
                            0,
                            0,
                        ],
                        center_radius: [
                            finite_f32(local_a.x)?,
                            finite_f32(local_a.y)?,
                            finite_f32(local_a.z)?,
                            finite_f32(pair.capsule_radius)?,
                        ],
                        center_of_mass: [
                            finite_f32(link.center_of_mass.x)?,
                            finite_f32(link.center_of_mass.y)?,
                            finite_f32(link.center_of_mass.z)?,
                            0.0,
                        ],
                        plane: [
                            finite_f32(local_b.x)?,
                            finite_f32(local_b.y)?,
                            finite_f32(local_b.z)?,
                            0.0,
                        ],
                        material: [
                            finite_f32(pair.restitution)?,
                            dt,
                            finite_f32(pair.friction)?,
                            53.0,
                        ],
                        other_center_radius: offsets,
                        first_axis_end: [
                            finite_f32(rotation.i)?,
                            finite_f32(rotation.j)?,
                            finite_f32(rotation.k)?,
                            finite_f32(rotation.w)?,
                        ],
                        second_axis_end: [
                            finite_f32(translation.x)?,
                            finite_f32(translation.y)?,
                            finite_f32(translation.z)?,
                            manifold_slot as f32,
                        ],
                        ..PackedSphere::zeroed()
                    });
                }
            }
            for pair in &scene_polyline_capsule_pairs[index] {
                let Some(link) = articulation.link(pair.capsule_link) else {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                };
                if !pair.capsule_radius.is_finite()
                    || pair.capsule_radius <= 0.0
                    || !(0.0..=1.0).contains(&pair.restitution)
                    || !pair.friction.is_finite()
                    || pair.friction < 0.0
                {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                let offsets = pack_polyline_geometry(
                    &pair.polyline,
                    &mut convex_geometry,
                    &mut mesh_geometry_offsets,
                )?;
                let translation = pair.polyline_world_pose.translation.vector;
                let rotation = pair.polyline_world_pose.rotation.quaternion();
                let local_a = pair.capsule_local_a;
                let local_b = pair.capsule_local_b;
                for manifold_slot in 0..4 {
                    convex_contact_rows.push(spheres.len());
                    spheres.push(PackedSphere {
                        indices: [
                            checked_u32(poses.link_ranges()[index].start + pair.capsule_link)?,
                            checked_u32(
                                link_offset
                                    .checked_add(
                                        pair.capsule_link
                                            .checked_mul(stride)
                                            .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                                    )
                                    .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                            )?,
                            0,
                            0,
                        ],
                        center_radius: [
                            finite_f32(local_a.x)?,
                            finite_f32(local_a.y)?,
                            finite_f32(local_a.z)?,
                            finite_f32(pair.capsule_radius)?,
                        ],
                        center_of_mass: [
                            finite_f32(link.center_of_mass.x)?,
                            finite_f32(link.center_of_mass.y)?,
                            finite_f32(link.center_of_mass.z)?,
                            0.0,
                        ],
                        plane: [
                            finite_f32(local_b.x)?,
                            finite_f32(local_b.y)?,
                            finite_f32(local_b.z)?,
                            0.0,
                        ],
                        material: [
                            finite_f32(pair.restitution)?,
                            dt,
                            finite_f32(pair.friction)?,
                            73.0,
                        ],
                        other_center_radius: offsets,
                        first_axis_end: [
                            finite_f32(rotation.i)?,
                            finite_f32(rotation.j)?,
                            finite_f32(rotation.k)?,
                            finite_f32(rotation.w)?,
                        ],
                        second_axis_end: [
                            finite_f32(translation.x)?,
                            finite_f32(translation.y)?,
                            finite_f32(translation.z)?,
                            manifold_slot as f32,
                        ],
                        ..PackedSphere::zeroed()
                    });
                }
            }
            for pair in &scene_mesh_box_pairs[index] {
                let Some(link) = articulation.link(pair.box_link) else {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                };
                if pair
                    .box_half_extents
                    .iter()
                    .any(|extent| !extent.is_finite() || *extent <= 0.0)
                    || !(0.0..=1.0).contains(&pair.restitution)
                    || !pair.friction.is_finite()
                    || pair.friction < 0.0
                {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                let offsets = pack_mesh_geometry(
                    &pair.mesh,
                    &mut convex_geometry,
                    &mut mesh_geometry_offsets,
                )?;
                let translation = pair.mesh_world_pose.translation.vector;
                let mesh_rotation = pair.mesh_world_pose.rotation.quaternion();
                let local_center = pair.box_local_pose.translation.vector;
                let box_rotation = pair.box_local_pose.rotation.quaternion();
                let box_geometry =
                    crate::articulated_world::gpu_box_convex_geometry(pair.box_half_extents)
                        .map_err(|_| GpuArticulatedGroundContactError::InvalidInput)?;
                let hull_header = pack_convex_geometry(&mut convex_geometry, &box_geometry)?;
                for manifold_slot in 0..4 {
                    convex_contact_rows.push(spheres.len());
                    spheres.push(PackedSphere {
                        indices: [
                            checked_u32(poses.link_ranges()[index].start + pair.box_link)?,
                            checked_u32(
                                link_offset
                                    .checked_add(
                                        pair.box_link
                                            .checked_mul(stride)
                                            .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                                    )
                                    .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                            )?,
                            0,
                            0,
                        ],
                        center_radius: [
                            finite_f32(local_center.x)?,
                            finite_f32(local_center.y)?,
                            finite_f32(local_center.z)?,
                            hull_header,
                        ],
                        center_of_mass: [
                            finite_f32(link.center_of_mass.x)?,
                            finite_f32(link.center_of_mass.y)?,
                            finite_f32(link.center_of_mass.z)?,
                            0.0,
                        ],
                        plane: [
                            finite_f32(translation.x)?,
                            finite_f32(translation.y)?,
                            finite_f32(translation.z)?,
                            manifold_slot as f32,
                        ],
                        material: [
                            finite_f32(pair.restitution)?,
                            dt,
                            finite_f32(pair.friction)?,
                            57.0,
                        ],
                        other_center_radius: offsets,
                        first_axis_end: [
                            finite_f32(box_rotation.i)?,
                            finite_f32(box_rotation.j)?,
                            finite_f32(box_rotation.k)?,
                            finite_f32(box_rotation.w)?,
                        ],
                        second_axis_end: [
                            finite_f32(mesh_rotation.i)?,
                            finite_f32(mesh_rotation.j)?,
                            finite_f32(mesh_rotation.k)?,
                            finite_f32(mesh_rotation.w)?,
                        ],
                        ..PackedSphere::zeroed()
                    });
                }
            }
            for pair in &scene_polyline_box_pairs[index] {
                let Some(link) = articulation.link(pair.box_link) else {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                };
                if pair
                    .box_half_extents
                    .iter()
                    .any(|extent| !extent.is_finite() || *extent <= 0.0)
                    || !(0.0..=1.0).contains(&pair.restitution)
                    || !pair.friction.is_finite()
                    || pair.friction < 0.0
                {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                let offsets = pack_polyline_geometry(
                    &pair.polyline,
                    &mut convex_geometry,
                    &mut mesh_geometry_offsets,
                )?;
                let translation = pair.polyline_world_pose.translation.vector;
                let mesh_rotation = pair.polyline_world_pose.rotation.quaternion();
                let local_center = pair.box_local_pose.translation.vector;
                let box_rotation = pair.box_local_pose.rotation.quaternion();
                for manifold_slot in 0..4 {
                    convex_contact_rows.push(spheres.len());
                    spheres.push(PackedSphere {
                        indices: [
                            checked_u32(poses.link_ranges()[index].start + pair.box_link)?,
                            checked_u32(
                                link_offset
                                    .checked_add(
                                        pair.box_link
                                            .checked_mul(stride)
                                            .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                                    )
                                    .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                            )?,
                            0,
                            0,
                        ],
                        center_radius: [
                            finite_f32(local_center.x)?,
                            finite_f32(local_center.y)?,
                            finite_f32(local_center.z)?,
                            finite_f32(pair.box_half_extents.x)?,
                        ],
                        center_of_mass: [
                            finite_f32(link.center_of_mass.x)?,
                            finite_f32(link.center_of_mass.y)?,
                            finite_f32(link.center_of_mass.z)?,
                            finite_f32(pair.box_half_extents.y)?,
                        ],
                        plane: [
                            finite_f32(box_rotation.i)?,
                            finite_f32(box_rotation.j)?,
                            finite_f32(box_rotation.k)?,
                            finite_f32(box_rotation.w)?,
                        ],
                        material: [
                            finite_f32(pair.restitution)?,
                            dt,
                            finite_f32(pair.friction)?,
                            74.0,
                        ],
                        other_center_radius: offsets,
                        other_center_of_mass: [0.0, 0.0, 0.0, finite_f32(pair.box_half_extents.z)?],
                        first_axis_end: [
                            finite_f32(mesh_rotation.i)?,
                            finite_f32(mesh_rotation.j)?,
                            finite_f32(mesh_rotation.k)?,
                            finite_f32(mesh_rotation.w)?,
                        ],
                        second_axis_end: [
                            finite_f32(translation.x)?,
                            finite_f32(translation.y)?,
                            finite_f32(translation.z)?,
                            manifold_slot as f32,
                        ],
                        ..PackedSphere::zeroed()
                    });
                }
            }
            for pair in &scene_mesh_axial_pairs[index] {
                let Some(link) = articulation.link(pair.link) else {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                };
                if !pair.half_height.is_finite()
                    || pair.half_height <= 0.0
                    || !pair.radius.is_finite()
                    || pair.radius <= 0.0
                    || !(0.0..=1.0).contains(&pair.restitution)
                    || !pair.friction.is_finite()
                    || pair.friction < 0.0
                {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                let offsets = pack_mesh_geometry(
                    &pair.mesh,
                    &mut convex_geometry,
                    &mut mesh_geometry_offsets,
                )?;
                let translation = pair.mesh_world_pose.translation.vector;
                let mesh_rotation = pair.mesh_world_pose.rotation.quaternion();
                let local_center = pair.local_pose.translation.vector;
                let axial_rotation = pair.local_pose.rotation.quaternion();
                convex_contact_rows.push(spheres.len());
                spheres.push(PackedSphere {
                    indices: [
                        checked_u32(poses.link_ranges()[index].start + pair.link)?,
                        checked_u32(
                            link_offset
                                .checked_add(
                                    pair.link
                                        .checked_mul(stride)
                                        .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                                )
                                .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                        )?,
                        0,
                        0,
                    ],
                    center_radius: [
                        finite_f32(local_center.x)?,
                        finite_f32(local_center.y)?,
                        finite_f32(local_center.z)?,
                        finite_f32(pair.radius)?,
                    ],
                    center_of_mass: [
                        finite_f32(link.center_of_mass.x)?,
                        finite_f32(link.center_of_mass.y)?,
                        finite_f32(link.center_of_mass.z)?,
                        finite_f32(pair.half_height)?,
                    ],
                    plane: [
                        finite_f32(axial_rotation.i)?,
                        finite_f32(axial_rotation.j)?,
                        finite_f32(axial_rotation.k)?,
                        finite_f32(axial_rotation.w)?,
                    ],
                    material: [
                        finite_f32(pair.restitution)?,
                        dt,
                        finite_f32(pair.friction)?,
                        match pair.kind {
                            GpuArticulatedGroundAxialKind::Cylinder => 55.0,
                            GpuArticulatedGroundAxialKind::Cone => 56.0,
                        },
                    ],
                    other_center_radius: offsets,
                    first_axis_end: [
                        finite_f32(mesh_rotation.i)?,
                        finite_f32(mesh_rotation.j)?,
                        finite_f32(mesh_rotation.k)?,
                        finite_f32(mesh_rotation.w)?,
                    ],
                    second_axis_end: [
                        finite_f32(translation.x)?,
                        finite_f32(translation.y)?,
                        finite_f32(translation.z)?,
                        0.0,
                    ],
                    ..PackedSphere::zeroed()
                });
                if pair.kind == GpuArticulatedGroundAxialKind::Cylinder {
                    let mut endpoint = spheres[spheres.len() - 1];
                    endpoint.second_axis_end[3] = 1.0;
                    convex_contact_rows.push(spheres.len());
                    spheres.push(endpoint);
                }
            }
            for pair in &scene_polyline_axial_pairs[index] {
                let Some(link) = articulation.link(pair.link) else {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                };
                if !pair.half_height.is_finite()
                    || pair.half_height <= 0.0
                    || !pair.radius.is_finite()
                    || pair.radius <= 0.0
                    || !(0.0..=1.0).contains(&pair.restitution)
                    || !pair.friction.is_finite()
                    || pair.friction < 0.0
                {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                let offsets = pack_polyline_geometry(
                    &pair.polyline,
                    &mut convex_geometry,
                    &mut mesh_geometry_offsets,
                )?;
                let translation = pair.polyline_world_pose.translation.vector;
                let mesh_rotation = pair.polyline_world_pose.rotation.quaternion();
                let local_center = pair.local_pose.translation.vector;
                let axial_rotation = pair.local_pose.rotation.quaternion();
                for manifold_slot in 0..4 {
                    convex_contact_rows.push(spheres.len());
                    spheres.push(PackedSphere {
                        indices: [
                            checked_u32(poses.link_ranges()[index].start + pair.link)?,
                            checked_u32(
                                link_offset
                                    .checked_add(
                                        pair.link
                                            .checked_mul(stride)
                                            .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                                    )
                                    .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                            )?,
                            0,
                            0,
                        ],
                        center_radius: [
                            finite_f32(local_center.x)?,
                            finite_f32(local_center.y)?,
                            finite_f32(local_center.z)?,
                            finite_f32(pair.radius)?,
                        ],
                        center_of_mass: [
                            finite_f32(link.center_of_mass.x)?,
                            finite_f32(link.center_of_mass.y)?,
                            finite_f32(link.center_of_mass.z)?,
                            finite_f32(pair.half_height)?,
                        ],
                        plane: [
                            finite_f32(axial_rotation.i)?,
                            finite_f32(axial_rotation.j)?,
                            finite_f32(axial_rotation.k)?,
                            finite_f32(axial_rotation.w)?,
                        ],
                        material: [
                            finite_f32(pair.restitution)?,
                            dt,
                            finite_f32(pair.friction)?,
                            match pair.kind {
                                GpuArticulatedGroundAxialKind::Cylinder => 75.0,
                                GpuArticulatedGroundAxialKind::Cone => 76.0,
                            },
                        ],
                        other_center_radius: offsets,
                        first_axis_end: [
                            finite_f32(mesh_rotation.i)?,
                            finite_f32(mesh_rotation.j)?,
                            finite_f32(mesh_rotation.k)?,
                            finite_f32(mesh_rotation.w)?,
                        ],
                        second_axis_end: [
                            finite_f32(translation.x)?,
                            finite_f32(translation.y)?,
                            finite_f32(translation.z)?,
                            manifold_slot as f32,
                        ],
                        ..PackedSphere::zeroed()
                    });
                }
            }
            for pair in &scene_mesh_convex_pairs[index] {
                let Some(link) = articulation.link(pair.convex_link) else {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                };
                if !valid_convex_geometry(&pair.convex_geometry)
                    || !(0.0..=1.0).contains(&pair.restitution)
                    || !pair.friction.is_finite()
                    || pair.friction < 0.0
                    || pair
                        .convex_local_pose
                        .translation
                        .vector
                        .iter()
                        .chain(pair.convex_local_pose.rotation.coords.iter())
                        .chain(pair.mesh_world_pose.translation.vector.iter())
                        .chain(pair.mesh_world_pose.rotation.coords.iter())
                        .any(|value| !value.is_finite())
                {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                let offsets = pack_mesh_geometry(
                    &pair.mesh,
                    &mut convex_geometry,
                    &mut mesh_geometry_offsets,
                )?;
                let hull_header =
                    pack_convex_geometry(&mut convex_geometry, &pair.convex_geometry)?;
                let local_position = pair.convex_local_pose.translation.vector;
                let local_rotation = pair.convex_local_pose.rotation.quaternion();
                let mesh_position = pair.mesh_world_pose.translation.vector;
                let mesh_rotation = pair.mesh_world_pose.rotation.quaternion();
                for manifold_slot in 0..4 {
                    convex_contact_rows.push(spheres.len());
                    spheres.push(PackedSphere {
                        indices: [
                            checked_u32(poses.link_ranges()[index].start + pair.convex_link)?,
                            checked_u32(
                                link_offset
                                    .checked_add(
                                        pair.convex_link
                                            .checked_mul(stride)
                                            .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                                    )
                                    .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                            )?,
                            0,
                            0,
                        ],
                        center_radius: [
                            finite_f32(local_position.x)?,
                            finite_f32(local_position.y)?,
                            finite_f32(local_position.z)?,
                            hull_header,
                        ],
                        center_of_mass: [
                            finite_f32(link.center_of_mass.x)?,
                            finite_f32(link.center_of_mass.y)?,
                            finite_f32(link.center_of_mass.z)?,
                            0.0,
                        ],
                        plane: [
                            finite_f32(mesh_position.x)?,
                            finite_f32(mesh_position.y)?,
                            finite_f32(mesh_position.z)?,
                            manifold_slot as f32,
                        ],
                        material: [
                            finite_f32(pair.restitution)?,
                            dt,
                            finite_f32(pair.friction)?,
                            57.0,
                        ],
                        other_center_radius: offsets,
                        first_axis_end: [
                            finite_f32(local_rotation.i)?,
                            finite_f32(local_rotation.j)?,
                            finite_f32(local_rotation.k)?,
                            finite_f32(local_rotation.w)?,
                        ],
                        second_axis_end: [
                            finite_f32(mesh_rotation.i)?,
                            finite_f32(mesh_rotation.j)?,
                            finite_f32(mesh_rotation.k)?,
                            finite_f32(mesh_rotation.w)?,
                        ],
                        ..PackedSphere::zeroed()
                    });
                }
            }
            for pair in &scene_polyline_convex_pairs[index] {
                let Some(link) = articulation.link(pair.convex_link) else {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                };
                if !valid_convex_geometry(&pair.convex_geometry)
                    || !(0.0..=1.0).contains(&pair.restitution)
                    || !pair.friction.is_finite()
                    || pair.friction < 0.0
                    || pair
                        .convex_local_pose
                        .translation
                        .vector
                        .iter()
                        .chain(pair.convex_local_pose.rotation.coords.iter())
                        .chain(pair.polyline_world_pose.translation.vector.iter())
                        .chain(pair.polyline_world_pose.rotation.coords.iter())
                        .any(|value| !value.is_finite())
                {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                let offsets = pack_polyline_geometry(
                    &pair.polyline,
                    &mut convex_geometry,
                    &mut mesh_geometry_offsets,
                )?;
                let hull_header =
                    pack_convex_geometry(&mut convex_geometry, &pair.convex_geometry)?;
                let local_position = pair.convex_local_pose.translation.vector;
                let local_rotation = pair.convex_local_pose.rotation.quaternion();
                let mesh_position = pair.polyline_world_pose.translation.vector;
                let mesh_rotation = pair.polyline_world_pose.rotation.quaternion();
                for manifold_slot in 0..4 {
                    convex_contact_rows.push(spheres.len());
                    spheres.push(PackedSphere {
                        indices: [
                            checked_u32(poses.link_ranges()[index].start + pair.convex_link)?,
                            checked_u32(
                                link_offset
                                    .checked_add(
                                        pair.convex_link
                                            .checked_mul(stride)
                                            .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                                    )
                                    .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                            )?,
                            0,
                            0,
                        ],
                        center_radius: [
                            finite_f32(local_position.x)?,
                            finite_f32(local_position.y)?,
                            finite_f32(local_position.z)?,
                            hull_header,
                        ],
                        center_of_mass: [
                            finite_f32(link.center_of_mass.x)?,
                            finite_f32(link.center_of_mass.y)?,
                            finite_f32(link.center_of_mass.z)?,
                            0.0,
                        ],
                        plane: [
                            finite_f32(mesh_position.x)?,
                            finite_f32(mesh_position.y)?,
                            finite_f32(mesh_position.z)?,
                            manifold_slot as f32,
                        ],
                        material: [
                            finite_f32(pair.restitution)?,
                            dt,
                            finite_f32(pair.friction)?,
                            77.0,
                        ],
                        other_center_radius: offsets,
                        first_axis_end: [
                            finite_f32(local_rotation.i)?,
                            finite_f32(local_rotation.j)?,
                            finite_f32(local_rotation.k)?,
                            finite_f32(local_rotation.w)?,
                        ],
                        second_axis_end: [
                            finite_f32(mesh_rotation.i)?,
                            finite_f32(mesh_rotation.j)?,
                            finite_f32(mesh_rotation.k)?,
                            finite_f32(mesh_rotation.w)?,
                        ],
                        ..PackedSphere::zeroed()
                    });
                }
            }
            struct ConvexPairRef<'a> {
                first_link: usize,
                first_local_pose: Isometry3<f64>,
                first_geometry: &'a ConvexGeometry,
                second_link: Option<usize>,
                second_pose: Isometry3<f64>,
                second_geometry: &'a ConvexGeometry,
                restitution: f64,
                friction: f64,
                mode: f32,
            }
            let polyhedron_pairs = convex_pairs[index]
                .iter()
                .map(|pair| ConvexPairRef {
                    first_link: pair.first_link,
                    first_local_pose: pair.first_local_pose,
                    first_geometry: &pair.first_geometry,
                    second_link: Some(pair.second_link),
                    second_pose: pair.second_local_pose,
                    second_geometry: &pair.second_geometry,
                    restitution: pair.restitution,
                    friction: pair.friction,
                    mode: 44.0,
                })
                .chain(static_convex_pairs[index].iter().map(|pair| ConvexPairRef {
                    first_link: pair.first_link,
                    first_local_pose: pair.first_local_pose,
                    first_geometry: &pair.first_geometry,
                    second_link: None,
                    second_pose: pair.second_world_pose,
                    second_geometry: &pair.second_geometry,
                    restitution: pair.restitution,
                    friction: pair.friction,
                    mode: 47.0,
                }));
            for pair in polyhedron_pairs {
                let first_link = articulation
                    .link(pair.first_link)
                    .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
                let second_link = pair.second_link.and_then(|link| articulation.link(link));
                if pair.second_link == Some(pair.first_link)
                    || (pair.second_link.is_some() && second_link.is_none())
                    || !valid_convex_geometry(pair.first_geometry)
                    || !valid_convex_geometry(pair.second_geometry)
                    || !(0.0..=1.0).contains(&pair.restitution)
                    || !pair.friction.is_finite()
                    || pair.friction < 0.0
                    || pair
                        .first_local_pose
                        .translation
                        .vector
                        .iter()
                        .chain(pair.first_local_pose.rotation.coords.iter())
                        .chain(pair.second_pose.translation.vector.iter())
                        .chain(pair.second_pose.rotation.coords.iter())
                        .any(|value| !value.is_finite())
                {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                let first_header = pack_convex_geometry(&mut convex_geometry, pair.first_geometry)?;
                let second_header =
                    pack_convex_geometry(&mut convex_geometry, pair.second_geometry)?;
                let term_offset = |link: usize| -> Result<u32, GpuArticulatedGroundContactError> {
                    checked_u32(
                        link_offset
                            .checked_add(
                                link.checked_mul(stride)
                                    .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                            )
                            .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                    )
                };
                let first_position = pair.first_local_pose.translation.vector;
                let second_position = pair.second_pose.translation.vector;
                let first_rotation = pair.first_local_pose.rotation.quaternion();
                let second_rotation = pair.second_pose.rotation.quaternion();
                if pair.mode == 47.0 {
                    static_convex_pair_rows[index].push((
                        spheres.len(),
                        poses.link_ranges()[index].start + pair.first_link,
                    ));
                    static_convex_pair_poses[index].push([
                        finite_f32(second_position.x)?,
                        finite_f32(second_position.y)?,
                        finite_f32(second_position.z)?,
                        finite_f32(second_rotation.i)?,
                        finite_f32(second_rotation.j)?,
                        finite_f32(second_rotation.k)?,
                        finite_f32(second_rotation.w)?,
                    ]);
                }
                let packed = PackedSphere {
                    indices: [
                        checked_u32(poses.link_ranges()[index].start + pair.first_link)?,
                        term_offset(pair.first_link)?,
                        pair.second_link
                            .map(|link| checked_u32(poses.link_ranges()[index].start + link))
                            .transpose()?
                            .unwrap_or(0),
                        pair.second_link.map(term_offset).transpose()?.unwrap_or(0),
                    ],
                    center_radius: [
                        finite_f32(first_position.x)?,
                        finite_f32(first_position.y)?,
                        finite_f32(first_position.z)?,
                        first_header,
                    ],
                    center_of_mass: [
                        finite_f32(first_link.center_of_mass.x)?,
                        finite_f32(first_link.center_of_mass.y)?,
                        finite_f32(first_link.center_of_mass.z)?,
                        0.0,
                    ],
                    plane: [
                        finite_f32(second_position.x)?,
                        finite_f32(second_position.y)?,
                        finite_f32(second_position.z)?,
                        second_header,
                    ],
                    material: [
                        finite_f32(pair.restitution)?,
                        dt,
                        finite_f32(pair.friction)?,
                        pair.mode,
                    ],
                    other_center_of_mass: [
                        finite_f32(second_link.map_or(0.0, |link| link.center_of_mass.x))?,
                        finite_f32(second_link.map_or(0.0, |link| link.center_of_mass.y))?,
                        finite_f32(second_link.map_or(0.0, |link| link.center_of_mass.z))?,
                        0.0,
                    ],
                    first_axis_end: [
                        finite_f32(first_rotation.i)?,
                        finite_f32(first_rotation.j)?,
                        finite_f32(first_rotation.k)?,
                        finite_f32(first_rotation.w)?,
                    ],
                    second_axis_end: [
                        finite_f32(second_rotation.i)?,
                        finite_f32(second_rotation.j)?,
                        finite_f32(second_rotation.k)?,
                        finite_f32(second_rotation.w)?,
                    ],
                    ..PackedSphere::zeroed()
                };
                for slot in 0..4 {
                    convex_contact_rows.push(spheres.len());
                    let mut row = packed;
                    row.other_center_radius[3] = slot as f32;
                    spheres.push(row);
                }
            }
            for pair in &static_axial_convex_pairs[index] {
                let link = articulation
                    .link(pair.link)
                    .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
                if !pair.half_height.is_finite()
                    || pair.half_height <= 0.0
                    || !pair.radius.is_finite()
                    || pair.radius <= 0.0
                    || !valid_convex_geometry(&pair.static_geometry)
                    || !(0.0..=1.0).contains(&pair.restitution)
                    || !pair.friction.is_finite()
                    || pair.friction < 0.0
                    || pair
                        .local_pose
                        .translation
                        .vector
                        .iter()
                        .chain(pair.local_pose.rotation.coords.iter())
                        .chain(pair.static_pose.translation.vector.iter())
                        .chain(pair.static_pose.rotation.coords.iter())
                        .any(|value| !value.is_finite())
                {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                let header = pack_convex_geometry(&mut convex_geometry, &pair.static_geometry)?;
                let term_offset = checked_u32(
                    link_offset
                        .checked_add(
                            pair.link
                                .checked_mul(stride)
                                .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                        )
                        .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                )?;
                let local = pair.local_pose.translation.vector;
                let local_rotation = pair.local_pose.rotation.quaternion();
                let static_center = pair.static_pose.translation.vector;
                let static_rotation = pair.static_pose.rotation.quaternion();
                static_axial_convex_rows[index]
                    .push((spheres.len(), poses.link_ranges()[index].start + pair.link));
                static_axial_convex_poses[index].push([
                    finite_f32(static_center.x)?,
                    finite_f32(static_center.y)?,
                    finite_f32(static_center.z)?,
                    finite_f32(static_rotation.i)?,
                    finite_f32(static_rotation.j)?,
                    finite_f32(static_rotation.k)?,
                    finite_f32(static_rotation.w)?,
                ]);
                convex_contact_rows.push(spheres.len());
                spheres.push(PackedSphere {
                    indices: [
                        checked_u32(poses.link_ranges()[index].start + pair.link)?,
                        term_offset,
                        0,
                        0,
                    ],
                    center_radius: [
                        finite_f32(local.x)?,
                        finite_f32(local.y)?,
                        finite_f32(local.z)?,
                        finite_f32(pair.radius)?,
                    ],
                    center_of_mass: [
                        finite_f32(link.center_of_mass.x)?,
                        finite_f32(link.center_of_mass.y)?,
                        finite_f32(link.center_of_mass.z)?,
                        finite_f32(pair.half_height)?,
                    ],
                    plane: [
                        finite_f32(static_center.x)?,
                        finite_f32(static_center.y)?,
                        finite_f32(static_center.z)?,
                        header,
                    ],
                    material: [
                        finite_f32(pair.restitution)?,
                        dt,
                        finite_f32(pair.friction)?,
                        match pair.kind {
                            GpuArticulatedGroundAxialKind::Cylinder => 50.0,
                            GpuArticulatedGroundAxialKind::Cone => 51.0,
                        },
                    ],
                    first_axis_end: [
                        finite_f32(local_rotation.i)?,
                        finite_f32(local_rotation.j)?,
                        finite_f32(local_rotation.k)?,
                        finite_f32(local_rotation.w)?,
                    ],
                    second_axis_end: [
                        finite_f32(static_rotation.i)?,
                        finite_f32(static_rotation.j)?,
                        finite_f32(static_rotation.k)?,
                        finite_f32(static_rotation.w)?,
                    ],
                    ..PackedSphere::zeroed()
                });
            }
            for pair in &axial_convex_pairs[index] {
                let axial_center_of_mass = if pair.axial_is_static {
                    Vector3::zeros()
                } else {
                    articulation
                        .link(pair.axial_link)
                        .ok_or(GpuArticulatedGroundContactError::InvalidInput)?
                        .center_of_mass
                };
                let convex_link = articulation
                    .link(pair.convex_link)
                    .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
                if (!pair.axial_is_static && pair.axial_link == pair.convex_link)
                    || !pair.half_height.is_finite()
                    || pair.half_height <= 0.0
                    || !pair.radius.is_finite()
                    || pair.radius <= 0.0
                    || !valid_convex_geometry(&pair.convex_geometry)
                    || !(0.0..=1.0).contains(&pair.restitution)
                    || !pair.friction.is_finite()
                    || pair.friction < 0.0
                    || pair
                        .axial_local_pose
                        .translation
                        .vector
                        .iter()
                        .chain(pair.axial_local_pose.rotation.coords.iter())
                        .chain(pair.convex_local_pose.translation.vector.iter())
                        .chain(pair.convex_local_pose.rotation.coords.iter())
                        .any(|value| !value.is_finite())
                {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                let header = pack_convex_geometry(&mut convex_geometry, &pair.convex_geometry)?;
                let term_offset = |link: usize| -> Result<u32, GpuArticulatedGroundContactError> {
                    checked_u32(
                        link_offset
                            .checked_add(
                                link.checked_mul(stride)
                                    .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                            )
                            .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                    )
                };
                let local = pair.axial_local_pose.translation.vector;
                let local_rotation = pair.axial_local_pose.rotation.quaternion();
                let convex_center = pair.convex_local_pose.translation.vector;
                let convex_rotation = pair.convex_local_pose.rotation.quaternion();
                convex_contact_rows.push(spheres.len());
                spheres.push(PackedSphere {
                    indices: [
                        if pair.axial_is_static {
                            0
                        } else {
                            checked_u32(poses.link_ranges()[index].start + pair.axial_link)?
                        },
                        if pair.axial_is_static {
                            0
                        } else {
                            term_offset(pair.axial_link)?
                        },
                        checked_u32(poses.link_ranges()[index].start + pair.convex_link)?,
                        term_offset(pair.convex_link)?,
                    ],
                    center_radius: [
                        finite_f32(local.x)?,
                        finite_f32(local.y)?,
                        finite_f32(local.z)?,
                        finite_f32(pair.radius)?,
                    ],
                    center_of_mass: [
                        finite_f32(axial_center_of_mass.x)?,
                        finite_f32(axial_center_of_mass.y)?,
                        finite_f32(axial_center_of_mass.z)?,
                        finite_f32(pair.half_height)?,
                    ],
                    plane: [
                        finite_f32(convex_center.x)?,
                        finite_f32(convex_center.y)?,
                        finite_f32(convex_center.z)?,
                        header,
                    ],
                    material: [
                        finite_f32(pair.restitution)?,
                        dt,
                        finite_f32(pair.friction)?,
                        match (pair.axial_is_static, pair.kind) {
                            (false, GpuArticulatedGroundAxialKind::Cylinder) => 58.0,
                            (false, GpuArticulatedGroundAxialKind::Cone) => 59.0,
                            (true, GpuArticulatedGroundAxialKind::Cylinder) => 70.0,
                            (true, GpuArticulatedGroundAxialKind::Cone) => 71.0,
                        },
                    ],
                    other_center_of_mass: [
                        finite_f32(convex_link.center_of_mass.x)?,
                        finite_f32(convex_link.center_of_mass.y)?,
                        finite_f32(convex_link.center_of_mass.z)?,
                        0.0,
                    ],
                    first_axis_end: [
                        finite_f32(local_rotation.i)?,
                        finite_f32(local_rotation.j)?,
                        finite_f32(local_rotation.k)?,
                        finite_f32(local_rotation.w)?,
                    ],
                    second_axis_end: [
                        finite_f32(convex_rotation.i)?,
                        finite_f32(convex_rotation.j)?,
                        finite_f32(convex_rotation.k)?,
                        finite_f32(convex_rotation.w)?,
                    ],
                    ..PackedSphere::zeroed()
                });
            }
            for pair in &axial_box_pairs[index] {
                let axial_link = articulation
                    .link(pair.axial_link)
                    .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
                let box_link = articulation
                    .link(pair.box_link)
                    .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
                if pair.axial_link == pair.box_link
                    || !pair.half_height.is_finite()
                    || pair.half_height <= 0.0
                    || !pair.radius.is_finite()
                    || pair.radius <= 0.0
                    || pair
                        .box_half_extents
                        .iter()
                        .any(|value| !value.is_finite() || *value <= 0.0)
                    || !(0.0..=1.0).contains(&pair.restitution)
                    || !pair.friction.is_finite()
                    || pair.friction < 0.0
                    || pair
                        .axial_local_pose
                        .translation
                        .vector
                        .iter()
                        .chain(pair.axial_local_pose.rotation.coords.iter())
                        .chain(pair.box_local_pose.translation.vector.iter())
                        .chain(pair.box_local_pose.rotation.coords.iter())
                        .any(|value| !value.is_finite())
                {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                let term_offset = |link: usize| -> Result<u32, GpuArticulatedGroundContactError> {
                    checked_u32(
                        link_offset
                            .checked_add(
                                link.checked_mul(stride)
                                    .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                            )
                            .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                    )
                };
                let axial_local = pair.axial_local_pose.translation.vector;
                let axial_rotation = pair.axial_local_pose.rotation.quaternion();
                let box_local = pair.box_local_pose.translation.vector;
                let box_rotation = pair.box_local_pose.rotation.quaternion();
                spheres.push(PackedSphere {
                    indices: [
                        checked_u32(poses.link_ranges()[index].start + pair.axial_link)?,
                        term_offset(pair.axial_link)?,
                        checked_u32(poses.link_ranges()[index].start + pair.box_link)?,
                        term_offset(pair.box_link)?,
                    ],
                    center_radius: [
                        finite_f32(axial_local.x)?,
                        finite_f32(axial_local.y)?,
                        finite_f32(axial_local.z)?,
                        finite_f32(pair.radius)?,
                    ],
                    center_of_mass: [
                        finite_f32(axial_link.center_of_mass.x)?,
                        finite_f32(axial_link.center_of_mass.y)?,
                        finite_f32(axial_link.center_of_mass.z)?,
                        0.0,
                    ],
                    plane: [
                        finite_f32(box_local.x)?,
                        finite_f32(box_local.y)?,
                        finite_f32(box_local.z)?,
                        finite_f32(pair.half_height)?,
                    ],
                    material: [
                        finite_f32(pair.restitution)?,
                        dt,
                        finite_f32(pair.friction)?,
                        match pair.kind {
                            GpuArticulatedGroundAxialKind::Cylinder => 34.0,
                            GpuArticulatedGroundAxialKind::Cone => 35.0,
                        },
                    ],
                    other_center_radius: [
                        finite_f32(pair.box_half_extents.x)?,
                        finite_f32(pair.box_half_extents.y)?,
                        finite_f32(pair.box_half_extents.z)?,
                        0.0,
                    ],
                    other_center_of_mass: [
                        finite_f32(box_link.center_of_mass.x)?,
                        finite_f32(box_link.center_of_mass.y)?,
                        finite_f32(box_link.center_of_mass.z)?,
                        0.0,
                    ],
                    first_axis_end: [
                        finite_f32(axial_rotation.i)?,
                        finite_f32(axial_rotation.j)?,
                        finite_f32(axial_rotation.k)?,
                        finite_f32(axial_rotation.w)?,
                    ],
                    second_axis_end: [
                        finite_f32(box_rotation.i)?,
                        finite_f32(box_rotation.j)?,
                        finite_f32(box_rotation.k)?,
                        finite_f32(box_rotation.w)?,
                    ],
                    ..PackedSphere::zeroed()
                });
            }
            for pair in &capsule_sphere_pairs[index] {
                let capsule = articulation
                    .link(pair.capsule_link)
                    .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
                let sphere = articulation
                    .link(pair.sphere_link)
                    .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
                if pair.capsule_link == pair.sphere_link
                    || !pair.capsule_radius.is_finite()
                    || pair.capsule_radius <= 0.0
                    || !pair.sphere_radius.is_finite()
                    || pair.sphere_radius <= 0.0
                    || !pair.restitution.is_finite()
                    || !(0.0..=1.0).contains(&pair.restitution)
                    || !pair.friction.is_finite()
                    || pair.friction < 0.0
                    || pair.capsule_local_a.iter().any(|value| !value.is_finite())
                    || pair.capsule_local_b.iter().any(|value| !value.is_finite())
                    || pair
                        .sphere_local_center
                        .iter()
                        .any(|value| !value.is_finite())
                {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                let term_offset = |link: usize| -> Result<u32, GpuArticulatedGroundContactError> {
                    checked_u32(
                        link_offset
                            .checked_add(
                                link.checked_mul(stride)
                                    .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                            )
                            .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                    )
                };
                spheres.push(PackedSphere {
                    indices: [
                        checked_u32(poses.link_ranges()[index].start + pair.capsule_link)?,
                        term_offset(pair.capsule_link)?,
                        checked_u32(poses.link_ranges()[index].start + pair.sphere_link)?,
                        term_offset(pair.sphere_link)?,
                    ],
                    center_radius: [
                        finite_f32(pair.capsule_local_a.x)?,
                        finite_f32(pair.capsule_local_a.y)?,
                        finite_f32(pair.capsule_local_a.z)?,
                        finite_f32(pair.capsule_radius)?,
                    ],
                    center_of_mass: [
                        finite_f32(capsule.center_of_mass.x)?,
                        finite_f32(capsule.center_of_mass.y)?,
                        finite_f32(capsule.center_of_mass.z)?,
                        0.0,
                    ],
                    plane: [
                        finite_f32(pair.sphere_local_center.x)?,
                        finite_f32(pair.sphere_local_center.y)?,
                        finite_f32(pair.sphere_local_center.z)?,
                        finite_f32(pair.sphere_radius)?,
                    ],
                    material: [
                        finite_f32(pair.restitution)?,
                        dt,
                        finite_f32(pair.friction)?,
                        2.0,
                    ],
                    other_center_radius: [
                        finite_f32(pair.capsule_local_b.x)?,
                        finite_f32(pair.capsule_local_b.y)?,
                        finite_f32(pair.capsule_local_b.z)?,
                        0.0,
                    ],
                    other_center_of_mass: [
                        finite_f32(sphere.center_of_mass.x)?,
                        finite_f32(sphere.center_of_mass.y)?,
                        finite_f32(sphere.center_of_mass.z)?,
                        0.0,
                    ],
                    second_axis_end: [0.0; 4],
                    first_axis_end: [0.0; 4],
                    impulses: [0.0; 4],
                    previous_normal: [0.0; 4],
                    diagnostic_first: [0.0; 4],
                    diagnostic_second: [0.0; 4],
                    diagnostic_first_origin: [0.0; 4],
                    diagnostic_second_origin: [0.0; 4],
                    prescribed_linear: [0.0; 4],
                    prescribed_angular: [0.0; 4],
                });
            }
            for pair in &capsule_pairs[index] {
                let first = articulation
                    .link(pair.first_link)
                    .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
                let second = articulation
                    .link(pair.second_link)
                    .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
                if pair.first_link == pair.second_link
                    || !pair.first_radius.is_finite()
                    || pair.first_radius <= 0.0
                    || !pair.second_radius.is_finite()
                    || pair.second_radius <= 0.0
                    || !pair.restitution.is_finite()
                    || !(0.0..=1.0).contains(&pair.restitution)
                    || !pair.friction.is_finite()
                    || pair.friction < 0.0
                    || pair.first_local_a.iter().any(|value| !value.is_finite())
                    || pair.first_local_b.iter().any(|value| !value.is_finite())
                    || pair.second_local_a.iter().any(|value| !value.is_finite())
                    || pair.second_local_b.iter().any(|value| !value.is_finite())
                {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                let term_offset = |link: usize| -> Result<u32, GpuArticulatedGroundContactError> {
                    checked_u32(
                        link_offset
                            .checked_add(
                                link.checked_mul(stride)
                                    .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                            )
                            .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                    )
                };
                let packed = PackedSphere {
                    indices: [
                        checked_u32(poses.link_ranges()[index].start + pair.first_link)?,
                        term_offset(pair.first_link)?,
                        checked_u32(poses.link_ranges()[index].start + pair.second_link)?,
                        term_offset(pair.second_link)?,
                    ],
                    center_radius: [
                        finite_f32(pair.first_local_a.x)?,
                        finite_f32(pair.first_local_a.y)?,
                        finite_f32(pair.first_local_a.z)?,
                        finite_f32(pair.first_radius)?,
                    ],
                    center_of_mass: [
                        finite_f32(first.center_of_mass.x)?,
                        finite_f32(first.center_of_mass.y)?,
                        finite_f32(first.center_of_mass.z)?,
                        0.0,
                    ],
                    plane: [
                        finite_f32(pair.second_local_a.x)?,
                        finite_f32(pair.second_local_a.y)?,
                        finite_f32(pair.second_local_a.z)?,
                        finite_f32(pair.second_radius)?,
                    ],
                    material: [
                        finite_f32(pair.restitution)?,
                        dt,
                        finite_f32(pair.friction)?,
                        3.0,
                    ],
                    other_center_radius: [
                        finite_f32(pair.first_local_b.x)?,
                        finite_f32(pair.first_local_b.y)?,
                        finite_f32(pair.first_local_b.z)?,
                        0.0,
                    ],
                    other_center_of_mass: [
                        finite_f32(second.center_of_mass.x)?,
                        finite_f32(second.center_of_mass.y)?,
                        finite_f32(second.center_of_mass.z)?,
                        0.0,
                    ],
                    second_axis_end: [
                        finite_f32(pair.second_local_b.x)?,
                        finite_f32(pair.second_local_b.y)?,
                        finite_f32(pair.second_local_b.z)?,
                        0.0,
                    ],
                    first_axis_end: [0.0; 4],
                    impulses: [0.0; 4],
                    previous_normal: [0.0; 4],
                    diagnostic_first: [0.0; 4],
                    diagnostic_second: [0.0; 4],
                    diagnostic_first_origin: [0.0; 4],
                    diagnostic_second_origin: [0.0; 4],
                    prescribed_linear: [0.0; 4],
                    prescribed_angular: [0.0; 4],
                };
                spheres.push(packed);
                let mut side = packed;
                side.material[3] = 4.0;
                spheres.push(side);
            }
            for pair in &sphere_box_pairs[index] {
                let sphere = articulation
                    .link(pair.sphere_link)
                    .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
                let box_link = articulation
                    .link(pair.box_link)
                    .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
                if pair.sphere_link == pair.box_link
                    || !pair.sphere_radius.is_finite()
                    || pair.sphere_radius <= 0.0
                    || pair
                        .box_half_extents
                        .iter()
                        .any(|value| !value.is_finite() || *value <= 0.0)
                    || !(0.0..=1.0).contains(&pair.restitution)
                    || !pair.friction.is_finite()
                    || pair.friction < 0.0
                    || pair
                        .sphere_local_center
                        .iter()
                        .any(|value| !value.is_finite())
                    || pair
                        .box_local_pose
                        .translation
                        .vector
                        .iter()
                        .any(|value| !value.is_finite())
                    || pair
                        .box_local_pose
                        .rotation
                        .coords
                        .iter()
                        .any(|value| !value.is_finite())
                {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                let term_offset = |link: usize| -> Result<u32, GpuArticulatedGroundContactError> {
                    checked_u32(
                        link_offset
                            .checked_add(
                                link.checked_mul(stride)
                                    .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                            )
                            .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                    )
                };
                let local = pair.box_local_pose.translation.vector;
                let rotation = pair.box_local_pose.rotation.quaternion();
                spheres.push(PackedSphere {
                    indices: [
                        checked_u32(poses.link_ranges()[index].start + pair.sphere_link)?,
                        term_offset(pair.sphere_link)?,
                        checked_u32(poses.link_ranges()[index].start + pair.box_link)?,
                        term_offset(pair.box_link)?,
                    ],
                    center_radius: [
                        finite_f32(pair.sphere_local_center.x)?,
                        finite_f32(pair.sphere_local_center.y)?,
                        finite_f32(pair.sphere_local_center.z)?,
                        finite_f32(pair.sphere_radius)?,
                    ],
                    center_of_mass: [
                        finite_f32(sphere.center_of_mass.x)?,
                        finite_f32(sphere.center_of_mass.y)?,
                        finite_f32(sphere.center_of_mass.z)?,
                        0.0,
                    ],
                    plane: [
                        finite_f32(local.x)?,
                        finite_f32(local.y)?,
                        finite_f32(local.z)?,
                        0.0,
                    ],
                    material: [
                        finite_f32(pair.restitution)?,
                        dt,
                        finite_f32(pair.friction)?,
                        6.0,
                    ],
                    other_center_radius: [
                        finite_f32(pair.box_half_extents.x)?,
                        finite_f32(pair.box_half_extents.y)?,
                        finite_f32(pair.box_half_extents.z)?,
                        0.0,
                    ],
                    other_center_of_mass: [
                        finite_f32(box_link.center_of_mass.x)?,
                        finite_f32(box_link.center_of_mass.y)?,
                        finite_f32(box_link.center_of_mass.z)?,
                        0.0,
                    ],
                    second_axis_end: [
                        finite_f32(rotation.i)?,
                        finite_f32(rotation.j)?,
                        finite_f32(rotation.k)?,
                        finite_f32(rotation.w)?,
                    ],
                    first_axis_end: [0.0; 4],
                    impulses: [0.0; 4],
                    previous_normal: [0.0; 4],
                    diagnostic_first: [0.0; 4],
                    diagnostic_second: [0.0; 4],
                    diagnostic_first_origin: [0.0; 4],
                    diagnostic_second_origin: [0.0; 4],
                    prescribed_linear: [0.0; 4],
                    prescribed_angular: [0.0; 4],
                });
            }
            for pair in &capsule_box_pairs[index] {
                if pair.capsule_link == pair.box_link {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                validate_link_capsule(
                    articulation,
                    &GpuArticulatedLinkCapsule {
                        link: pair.capsule_link,
                        local_a: pair.capsule_local_a,
                        local_b: pair.capsule_local_b,
                        radius: pair.capsule_radius,
                        restitution: pair.restitution,
                        friction: pair.friction,
                    },
                )?;
                validate_link_box(
                    articulation,
                    &GpuArticulatedLinkBox {
                        link: pair.box_link,
                        local_pose: pair.box_local_pose,
                        half_extents: pair.box_half_extents,
                        restitution: pair.restitution,
                        friction: pair.friction,
                    },
                )?;
                let capsule = articulation
                    .link(pair.capsule_link)
                    .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
                let box_link = articulation
                    .link(pair.box_link)
                    .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
                let term_offset = |link: usize| -> Result<u32, GpuArticulatedGroundContactError> {
                    checked_u32(
                        link_offset
                            .checked_add(
                                link.checked_mul(stride)
                                    .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                            )
                            .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                    )
                };
                let local = pair.box_local_pose.translation.vector;
                let rotation = pair.box_local_pose.rotation.quaternion();
                let packed = PackedSphere {
                    indices: [
                        checked_u32(poses.link_ranges()[index].start + pair.capsule_link)?,
                        term_offset(pair.capsule_link)?,
                        checked_u32(poses.link_ranges()[index].start + pair.box_link)?,
                        term_offset(pair.box_link)?,
                    ],
                    center_radius: [
                        finite_f32(pair.capsule_local_a.x)?,
                        finite_f32(pair.capsule_local_a.y)?,
                        finite_f32(pair.capsule_local_a.z)?,
                        finite_f32(pair.capsule_radius)?,
                    ],
                    center_of_mass: [
                        finite_f32(capsule.center_of_mass.x)?,
                        finite_f32(capsule.center_of_mass.y)?,
                        finite_f32(capsule.center_of_mass.z)?,
                        0.0,
                    ],
                    plane: [
                        finite_f32(local.x)?,
                        finite_f32(local.y)?,
                        finite_f32(local.z)?,
                        0.0,
                    ],
                    material: [
                        finite_f32(pair.restitution)?,
                        dt,
                        finite_f32(pair.friction)?,
                        7.0,
                    ],
                    other_center_radius: [
                        finite_f32(pair.box_half_extents.x)?,
                        finite_f32(pair.box_half_extents.y)?,
                        finite_f32(pair.box_half_extents.z)?,
                        0.0,
                    ],
                    other_center_of_mass: [
                        finite_f32(box_link.center_of_mass.x)?,
                        finite_f32(box_link.center_of_mass.y)?,
                        finite_f32(box_link.center_of_mass.z)?,
                        0.0,
                    ],
                    second_axis_end: [
                        finite_f32(rotation.i)?,
                        finite_f32(rotation.j)?,
                        finite_f32(rotation.k)?,
                        finite_f32(rotation.w)?,
                    ],
                    first_axis_end: [
                        finite_f32(pair.capsule_local_b.x)?,
                        finite_f32(pair.capsule_local_b.y)?,
                        finite_f32(pair.capsule_local_b.z)?,
                        0.0,
                    ],
                    impulses: [0.0; 4],
                    previous_normal: [0.0; 4],
                    diagnostic_first: [0.0; 4],
                    diagnostic_second: [0.0; 4],
                    diagnostic_first_origin: [0.0; 4],
                    diagnostic_second_origin: [0.0; 4],
                    prescribed_linear: [0.0; 4],
                    prescribed_angular: [0.0; 4],
                };
                spheres.push(packed);
                let mut side = packed;
                side.material[3] = 9.0;
                spheres.push(side);
            }
            for pair in &box_pairs[index] {
                if pair.first_link == pair.second_link {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                validate_link_box(
                    articulation,
                    &GpuArticulatedLinkBox {
                        link: pair.first_link,
                        local_pose: pair.first_local_pose,
                        half_extents: pair.first_half_extents,
                        restitution: pair.restitution,
                        friction: pair.friction,
                    },
                )?;
                validate_link_box(
                    articulation,
                    &GpuArticulatedLinkBox {
                        link: pair.second_link,
                        local_pose: pair.second_local_pose,
                        half_extents: pair.second_half_extents,
                        restitution: pair.restitution,
                        friction: pair.friction,
                    },
                )?;
                let first = articulation
                    .link(pair.first_link)
                    .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
                let second = articulation
                    .link(pair.second_link)
                    .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
                let term_offset = |link: usize| -> Result<u32, GpuArticulatedGroundContactError> {
                    checked_u32(
                        link_offset
                            .checked_add(
                                link.checked_mul(stride)
                                    .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                            )
                            .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                    )
                };
                let first_center = pair.first_local_pose.translation.vector;
                let second_center = pair.second_local_pose.translation.vector;
                let first_rotation = pair.first_local_pose.rotation.quaternion();
                let second_rotation = pair.second_local_pose.rotation.quaternion();
                // Mode 8 stores first-box extents in three otherwise unused w lanes.
                let packed = PackedSphere {
                    indices: [
                        checked_u32(poses.link_ranges()[index].start + pair.first_link)?,
                        term_offset(pair.first_link)?,
                        checked_u32(poses.link_ranges()[index].start + pair.second_link)?,
                        term_offset(pair.second_link)?,
                    ],
                    center_radius: [
                        finite_f32(first_center.x)?,
                        finite_f32(first_center.y)?,
                        finite_f32(first_center.z)?,
                        finite_f32(pair.first_half_extents.x)?,
                    ],
                    center_of_mass: [
                        finite_f32(first.center_of_mass.x)?,
                        finite_f32(first.center_of_mass.y)?,
                        finite_f32(first.center_of_mass.z)?,
                        finite_f32(pair.first_half_extents.z)?,
                    ],
                    plane: [
                        finite_f32(second_center.x)?,
                        finite_f32(second_center.y)?,
                        finite_f32(second_center.z)?,
                        finite_f32(pair.first_half_extents.y)?,
                    ],
                    material: [
                        finite_f32(pair.restitution)?,
                        dt,
                        finite_f32(pair.friction)?,
                        8.0,
                    ],
                    other_center_radius: [
                        finite_f32(pair.second_half_extents.x)?,
                        finite_f32(pair.second_half_extents.y)?,
                        finite_f32(pair.second_half_extents.z)?,
                        0.0,
                    ],
                    other_center_of_mass: [
                        finite_f32(second.center_of_mass.x)?,
                        finite_f32(second.center_of_mass.y)?,
                        finite_f32(second.center_of_mass.z)?,
                        0.0,
                    ],
                    second_axis_end: [
                        finite_f32(second_rotation.i)?,
                        finite_f32(second_rotation.j)?,
                        finite_f32(second_rotation.k)?,
                        finite_f32(second_rotation.w)?,
                    ],
                    first_axis_end: [
                        finite_f32(first_rotation.i)?,
                        finite_f32(first_rotation.j)?,
                        finite_f32(first_rotation.k)?,
                        finite_f32(first_rotation.w)?,
                    ],
                    impulses: [0.0; 4],
                    previous_normal: [0.0; 4],
                    diagnostic_first: [0.0; 4],
                    diagnostic_second: [0.0; 4],
                    diagnostic_first_origin: [0.0; 4],
                    diagnostic_second_origin: [0.0; 4],
                    prescribed_linear: [0.0; 4],
                    prescribed_angular: [0.0; 4],
                };
                for point in 0..4 {
                    let mut contact = packed;
                    contact.other_center_radius[3] = point as f32;
                    spheres.push(contact);
                }
            }
            for coupling in &joint_couplings[index] {
                if !coupling.is_valid(n) {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                coupling_rows.push(checked_u32(spheres.len())?);
                let coefficients = coupling.coefficients.map(finite_f32);
                let [a0, a1, a2, a3, a4] = coefficients;
                spheres.push(PackedSphere {
                    indices: [
                        checked_u32(coupling.follower)?,
                        coupling
                            .source
                            .map(checked_u32)
                            .transpose()?
                            .unwrap_or(u32::MAX),
                        checked_u32(state.ranges()[index].start)?,
                        checked_u32(index)?,
                    ],
                    center_radius: [a0?, a1?, a2?, a3?],
                    center_of_mass: [
                        finite_f32(coupling.follower_reference)?,
                        finite_f32(coupling.source_reference)?,
                        0.0,
                        0.0,
                    ],
                    plane: [a4?, 0.0, 0.0, 0.0],
                    material: [0.2, dt, 2.0, 78.0],
                    ..PackedSphere::zeroed()
                });
            }
            let mut link_constraints = Vec::new();
            for constraint in &link_point_constraints[index] {
                if !constraint.is_valid(articulation.link_count()) {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                link_constraints.push((
                    constraint.link_a,
                    constraint.link_b,
                    Vector3::from(constraint.point_a),
                    Vector3::from(constraint.point_b),
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0],
                    false,
                ));
            }
            for constraint in &link_fixed_constraints[index] {
                if !constraint.is_valid(articulation.link_count()) {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                let a = constraint.frame_a.rotation.quaternion();
                let b = constraint.frame_b.rotation.quaternion();
                link_constraints.push((
                    constraint.link_a,
                    constraint.link_b,
                    constraint.frame_a.translation.vector,
                    constraint.frame_b.translation.vector,
                    [a.i, a.j, a.k, a.w],
                    [b.i, b.j, b.k, b.w],
                    true,
                ));
            }
            for (link_a, link_b, point_a, point_b, rotation_a, rotation_b, fixed) in
                link_constraints
            {
                let first = articulation
                    .link(link_a)
                    .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
                let second_com = link_b
                    .and_then(|link| articulation.link(link))
                    .map_or(Vector3::zeros(), |link| link.center_of_mass);
                let term_index = |link: usize| -> Result<u32, GpuArticulatedGroundContactError> {
                    checked_u32(
                        link_offset
                            .checked_add(
                                link.checked_mul(stride)
                                    .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                            )
                            .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                    )
                };
                if link_b.is_none() {
                    static_constraint_rows[index].push((
                        spheres.len(),
                        poses.link_ranges()[index].start + link_a,
                        if fixed { 6 } else { 3 },
                    ));
                }
                for component in 0..if fixed { 6 } else { 3 } {
                    let mut axis = [0.0; 4];
                    axis[component % 3] = 1.0;
                    let packed_vector = |value: Vector3<f64>| -> Result<[f32; 4], GpuArticulatedGroundContactError> {
                        Ok([finite_f32(value.x)?, finite_f32(value.y)?, finite_f32(value.z)?, 0.0])
                    };
                    let packed_quat =
                        |value: [f64; 4]| -> Result<[f32; 4], GpuArticulatedGroundContactError> {
                            Ok([
                                finite_f32(value[0])?,
                                finite_f32(value[1])?,
                                finite_f32(value[2])?,
                                finite_f32(value[3])?,
                            ])
                        };
                    spheres.push(PackedSphere {
                        indices: [
                            checked_u32(poses.link_ranges()[index].start + link_a)?,
                            term_index(link_a)?,
                            link_b
                                .map(|link| checked_u32(poses.link_ranges()[index].start + link))
                                .transpose()?
                                .unwrap_or(u32::MAX),
                            link_b.map(term_index).transpose()?.unwrap_or(0),
                        ],
                        center_radius: packed_vector(point_a)?,
                        other_center_radius: packed_vector(point_b)?,
                        center_of_mass: packed_vector(first.center_of_mass)?,
                        other_center_of_mass: packed_vector(second_com)?,
                        first_axis_end: packed_quat(rotation_a)?,
                        second_axis_end: packed_quat(rotation_b)?,
                        plane: axis,
                        material: [0.2, dt, 2.0, if component < 3 { 79.0 } else { 80.0 }],
                        ..PackedSphere::zeroed()
                    });
                }
            }
            let friction_start = spheres.len();
            for (coordinate, &friction) in joint_frictions[index].iter().enumerate() {
                if !friction.is_finite() || friction < 0.0 {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                let bound = finite_f32(friction * timestep)?;
                if friction > 0.0 && bound <= 0.0 {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                spheres.push(PackedSphere {
                    indices: [checked_u32(coordinate)?, 0, 0, 0],
                    material: [bound, dt, 0.0, 11.0],
                    ..PackedSphere::zeroed()
                });
            }
            friction_rows
                .push((friction_start != spheres.len()).then_some(friction_start..spheres.len()));
            friction_dimensions.push(n);
            let dynamic_start = spheres.len();
            let sphere_count = dynamic_spheres[index].len();
            let capsule_count = dynamic_capsules[index].len();
            let box_count = dynamic_boxes[index].len();
            let shape_count = sphere_count
                .checked_add(capsule_count)
                .and_then(|count| count.checked_add(box_count))
                .ok_or(GpuArticulatedGroundContactError::Capacity)?;
            if !dynamic_material_rules[index].is_empty()
                && dynamic_material_rules[index].len() != shape_count
            {
                return Err(GpuArticulatedGroundContactError::InvalidInput);
            }
            let packed_rules = |local_index: usize| {
                dynamic_material_rules[index]
                    .get(local_index)
                    .map(|rules| [rules.friction as u32 + 1, rules.restitution as u32 + 1])
                    .unwrap_or([0, 0])
            };
            let shape_base = packed_dynamic_shapes.len();
            for first in 0..sphere_count {
                for second in first + 1..sphere_count {
                    if pair_list.iter().any(|pair| {
                        same_sphere_shapes(
                            pair,
                            &dynamic_spheres[index][first],
                            &dynamic_spheres[index][second],
                        )
                    }) {
                        let _inserted = explicit_shape_exclusions.insert(PackedShapePair {
                            a: checked_u32(shape_base + first)?,
                            b: checked_u32(shape_base + second)?,
                        });
                    }
                }
            }
            for (sphere_index, sphere) in dynamic_spheres[index].iter().enumerate() {
                for (capsule_index, capsule) in dynamic_capsules[index].iter().enumerate() {
                    if capsule_sphere_pairs[index]
                        .iter()
                        .any(|pair| same_capsule_sphere_shapes(pair, capsule, sphere))
                    {
                        let _inserted = explicit_shape_exclusions.insert(PackedShapePair {
                            a: checked_u32(shape_base + sphere_index)?,
                            b: checked_u32(shape_base + sphere_count + capsule_index)?,
                        });
                    }
                }
            }
            for (first, capsule) in dynamic_capsules[index].iter().enumerate() {
                for (second, other) in dynamic_capsules[index].iter().enumerate().skip(first + 1) {
                    if capsule_pairs[index]
                        .iter()
                        .any(|pair| same_capsule_pair_shapes(pair, capsule, other))
                    {
                        let _inserted = explicit_shape_exclusions.insert(PackedShapePair {
                            a: checked_u32(shape_base + sphere_count + first)?,
                            b: checked_u32(shape_base + sphere_count + second)?,
                        });
                    }
                }
            }
            for (sphere_index, sphere) in dynamic_spheres[index].iter().enumerate() {
                for (box_index, box_shape) in dynamic_boxes[index].iter().enumerate() {
                    if sphere_box_pairs[index]
                        .iter()
                        .any(|pair| same_sphere_box_shapes(pair, sphere, box_shape))
                    {
                        let _inserted = explicit_shape_exclusions.insert(PackedShapePair {
                            a: checked_u32(shape_base + sphere_index)?,
                            b: checked_u32(shape_base + sphere_count + capsule_count + box_index)?,
                        });
                    }
                }
            }
            for (capsule_index, capsule) in dynamic_capsules[index].iter().enumerate() {
                for (box_index, box_shape) in dynamic_boxes[index].iter().enumerate() {
                    if capsule_box_pairs[index]
                        .iter()
                        .any(|pair| same_capsule_box_shapes(pair, capsule, box_shape))
                    {
                        let _inserted = explicit_shape_exclusions.insert(PackedShapePair {
                            a: checked_u32(shape_base + sphere_count + capsule_index)?,
                            b: checked_u32(shape_base + sphere_count + capsule_count + box_index)?,
                        });
                    }
                }
            }
            for (first_index, first) in dynamic_boxes[index].iter().enumerate() {
                for (second_index, second) in dynamic_boxes[index]
                    .iter()
                    .enumerate()
                    .skip(first_index + 1)
                {
                    if box_pairs[index]
                        .iter()
                        .any(|pair| same_box_pair_shapes(pair, first, second))
                    {
                        let _inserted = explicit_shape_exclusions.insert(PackedShapePair {
                            a: checked_u32(
                                shape_base + sphere_count + capsule_count + first_index,
                            )?,
                            b: checked_u32(
                                shape_base + sphere_count + capsule_count + second_index,
                            )?,
                        });
                    }
                }
            }
            let pair_map_start = checked_u32(dynamic_pair_slots.len())?;
            let shape_links = dynamic_spheres[index]
                .iter()
                .map(|shape| shape.link)
                .chain(dynamic_capsules[index].iter().map(|shape| shape.link))
                .chain(dynamic_boxes[index].iter().map(|shape| shape.link))
                .collect::<Vec<_>>();
            let mut dynamic_capacity = 0usize;
            for first in 0..shape_count {
                for second in first + 1..shape_count {
                    let excluded = shape_links[first] == shape_links[second]
                        || articulation.adjacent(shape_links[first], shape_links[second])
                        || explicit_shape_exclusions.contains(&PackedShapePair {
                            a: checked_u32(shape_base + first)?,
                            b: checked_u32(shape_base + second)?,
                        });
                    if excluded {
                        dynamic_pair_slots.push(u32::MAX);
                        continue;
                    }
                    dynamic_pair_slots.push(checked_u32(dynamic_capacity)?);
                    let first_kind = if first < sphere_count {
                        0
                    } else if first < sphere_count + capsule_count {
                        1
                    } else {
                        2
                    };
                    let second_kind = if second < sphere_count {
                        0
                    } else if second < sphere_count + capsule_count {
                        1
                    } else {
                        2
                    };
                    let rows = match (first_kind, second_kind) {
                        (1, 1) | (1, 2) => 2,
                        (2, 2) => 4,
                        _ => 1,
                    };
                    dynamic_capacity = dynamic_capacity
                        .checked_add(rows)
                        .ok_or(GpuArticulatedGroundContactError::Capacity)?;
                }
            }
            let dynamic_end = dynamic_start
                .checked_add(dynamic_capacity)
                .ok_or(GpuArticulatedGroundContactError::Capacity)?;
            for (local_index, shape) in dynamic_spheres[index].iter().enumerate() {
                validate_link_sphere(articulation, shape)?;
                let link = articulation
                    .link(shape.link)
                    .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
                packed_dynamic_shapes.push(PackedDynamicShape {
                    indices: [
                        checked_u32(poses.link_ranges()[index].start + shape.link)?,
                        checked_u32(
                            link_offset
                                .checked_add(
                                    shape
                                        .link
                                        .checked_mul(stride)
                                        .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                                )
                                .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                        )?,
                        0,
                        checked_u32(sphere_count)?,
                    ],
                    layout: [
                        checked_u32(shape_count)?,
                        checked_u32(local_index)?,
                        checked_u32(dynamic_start)?,
                        checked_u32(dynamic_slot_count)?,
                    ],
                    center_radius: [
                        finite_f32(shape.local_center.x)?,
                        finite_f32(shape.local_center.y)?,
                        finite_f32(shape.local_center.z)?,
                        finite_f32(shape.radius)?,
                    ],
                    center_of_mass: [
                        finite_f32(link.center_of_mass.x)?,
                        finite_f32(link.center_of_mass.y)?,
                        finite_f32(link.center_of_mass.z)?,
                        0.0,
                    ],
                    material: [
                        finite_f32(shape.restitution)?,
                        dt,
                        finite_f32(shape.friction)?,
                        0.0,
                    ],
                    axis_end: [0.0; 4],
                    orientation: [0.0, 0.0, 0.0, 1.0],
                    counts: [
                        checked_u32(capsule_count)?,
                        packed_rules(local_index)[0],
                        packed_rules(local_index)[1],
                        pair_map_start,
                    ],
                });
            }
            for (capsule_index, shape) in dynamic_capsules[index].iter().enumerate() {
                validate_link_capsule(articulation, shape)?;
                let link = articulation
                    .link(shape.link)
                    .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
                packed_dynamic_shapes.push(PackedDynamicShape {
                    indices: [
                        checked_u32(poses.link_ranges()[index].start + shape.link)?,
                        checked_u32(
                            link_offset
                                .checked_add(
                                    shape
                                        .link
                                        .checked_mul(stride)
                                        .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                                )
                                .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                        )?,
                        1,
                        checked_u32(sphere_count)?,
                    ],
                    layout: [
                        checked_u32(shape_count)?,
                        checked_u32(sphere_count + capsule_index)?,
                        checked_u32(dynamic_start)?,
                        checked_u32(dynamic_slot_count)?,
                    ],
                    center_radius: [
                        finite_f32(shape.local_a.x)?,
                        finite_f32(shape.local_a.y)?,
                        finite_f32(shape.local_a.z)?,
                        finite_f32(shape.radius)?,
                    ],
                    center_of_mass: [
                        finite_f32(link.center_of_mass.x)?,
                        finite_f32(link.center_of_mass.y)?,
                        finite_f32(link.center_of_mass.z)?,
                        0.0,
                    ],
                    material: [
                        finite_f32(shape.restitution)?,
                        dt,
                        finite_f32(shape.friction)?,
                        0.0,
                    ],
                    axis_end: [
                        finite_f32(shape.local_b.x)?,
                        finite_f32(shape.local_b.y)?,
                        finite_f32(shape.local_b.z)?,
                        0.0,
                    ],
                    orientation: [0.0, 0.0, 0.0, 1.0],
                    counts: [
                        checked_u32(capsule_count)?,
                        packed_rules(sphere_count + capsule_index)[0],
                        packed_rules(sphere_count + capsule_index)[1],
                        pair_map_start,
                    ],
                });
            }
            for (box_index, shape) in dynamic_boxes[index].iter().enumerate() {
                validate_link_box(articulation, shape)?;
                let link = articulation
                    .link(shape.link)
                    .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
                let center = shape.local_pose.translation.vector;
                let rotation = shape.local_pose.rotation.quaternion();
                packed_dynamic_shapes.push(PackedDynamicShape {
                    indices: [
                        checked_u32(poses.link_ranges()[index].start + shape.link)?,
                        checked_u32(
                            link_offset
                                .checked_add(
                                    shape
                                        .link
                                        .checked_mul(stride)
                                        .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                                )
                                .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                        )?,
                        2,
                        checked_u32(sphere_count)?,
                    ],
                    layout: [
                        checked_u32(shape_count)?,
                        checked_u32(sphere_count + capsule_count + box_index)?,
                        checked_u32(dynamic_start)?,
                        checked_u32(dynamic_slot_count)?,
                    ],
                    center_radius: [
                        finite_f32(center.x)?,
                        finite_f32(center.y)?,
                        finite_f32(center.z)?,
                        0.0,
                    ],
                    center_of_mass: [
                        finite_f32(link.center_of_mass.x)?,
                        finite_f32(link.center_of_mass.y)?,
                        finite_f32(link.center_of_mass.z)?,
                        0.0,
                    ],
                    material: [
                        finite_f32(shape.restitution)?,
                        dt,
                        finite_f32(shape.friction)?,
                        0.0,
                    ],
                    axis_end: [
                        finite_f32(shape.half_extents.x)?,
                        finite_f32(shape.half_extents.y)?,
                        finite_f32(shape.half_extents.z)?,
                        0.0,
                    ],
                    orientation: [
                        finite_f32(rotation.i)?,
                        finite_f32(rotation.j)?,
                        finite_f32(rotation.k)?,
                        finite_f32(rotation.w)?,
                    ],
                    counts: [
                        checked_u32(capsule_count)?,
                        packed_rules(sphere_count + capsule_count + box_index)[0],
                        packed_rules(sphere_count + capsule_count + box_index)[1],
                        pair_map_start,
                    ],
                });
            }
            spheres.resize(dynamic_end, PackedSphere::zeroed());
            dynamic_rows.push(dynamic_start..dynamic_end);
            dynamic_slot_count = dynamic_slot_count
                .checked_add(dynamic_capacity)
                .ok_or(GpuArticulatedGroundContactError::Capacity)?;
            systems.push(PackedSystem {
                indices: [
                    checked_u32(range.start)?,
                    checked_u32(n)?,
                    checked_u32(collider_start)?,
                    checked_u32(dynamic_end - collider_start)?,
                ],
                inverse: [
                    checked_u32(mass.inverse_ranges()[index].start)?,
                    checked_u32(mass.inverse_ranges()[index].end - 1)?,
                    iterations[index],
                    u32::from(warm_start[index]),
                ],
            });
            for local in 0..link_count {
                let term_offset = local
                    .checked_mul(stride)
                    .and_then(|offset| link_offset.checked_add(offset))
                    .ok_or(GpuArticulatedGroundContactError::Capacity)?;
                let link = articulation
                    .link(local)
                    .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
                motion_layout.push(PackedMotionLink {
                    indices: [
                        checked_u32(index)?,
                        checked_u32(range.start)?,
                        checked_u32(n)?,
                        checked_u32(term_offset)?,
                    ],
                    center_of_mass: [
                        finite_f32(link.center_of_mass.x)?,
                        finite_f32(link.center_of_mass.y)?,
                        finite_f32(link.center_of_mass.z)?,
                        if link.mass > 0.0 { 1.0 } else { 0.0 },
                    ],
                    thresholds: [0.0, 0.0, dt, 0.0],
                });
            }
            link_offset = link_offset
                .checked_add(
                    link_count
                        .checked_mul(stride)
                        .ok_or(GpuArticulatedGroundContactError::Capacity)?,
                )
                .ok_or(GpuArticulatedGroundContactError::Capacity)?;
        }
        if spheres.is_empty() {
            return Err(GpuArticulatedGroundContactError::InvalidInput);
        }
        let contact_row_count = spheres.len();
        let geometry_end = contact_row_count
            .checked_add(convex_geometry.len())
            .ok_or(GpuArticulatedGroundContactError::Capacity)?;
        if geometry_end > (1 << 24) {
            return Err(GpuArticulatedGroundContactError::Capacity);
        }
        let geometry_base = contact_row_count as f32;
        for row in convex_contact_rows {
            if (52.0..=57.0).contains(&spheres[row].material[3])
                || (72.0..=77.0).contains(&spheres[row].material[3])
            {
                for offset in &mut spheres[row].other_center_radius[..3] {
                    *offset += geometry_base;
                }
                if spheres[row].material[3] == 57.0 || spheres[row].material[3] == 77.0 {
                    spheres[row].center_radius[3] += geometry_base;
                }
            } else if spheres[row].material[3] == 50.0
                || spheres[row].material[3] == 51.0
                || spheres[row].material[3] == 58.0
                || spheres[row].material[3] == 59.0
                || spheres[row].material[3] == 70.0
                || spheres[row].material[3] == 71.0
            {
                spheres[row].plane[3] += geometry_base;
            } else if spheres[row].material[3] == 44.0 || spheres[row].material[3] == 47.0 {
                spheres[row].center_radius[3] += geometry_base;
                spheres[row].plane[3] += geometry_base;
            } else {
                spheres[row].plane[3] += geometry_base;
                spheres[row].other_center_radius[1] += geometry_base;
            }
            if matches!(
                spheres[row].material[3],
                43.0 | 46.0 | 49.0 | 81.0 | 82.0 | 83.0
            ) {
                spheres[row].second_axis_end[3] += geometry_base;
            }
        }
        spheres.extend(convex_geometry);
        let device = state.device();
        let device_limits = device.limits();
        for bytes in [
            systems.len() * size_of::<PackedSystem>(),
            coupling_rows.len() * size_of::<u32>(),
            spheres.len() * size_of::<PackedSphere>(),
            packed_dynamic_shapes.len() * size_of::<PackedDynamicShape>(),
            dynamic_pair_slots.len() * size_of::<u32>(),
            dynamic_slot_count * size_of::<u32>(),
        ] {
            if bytes as u64 > device_limits.max_buffer_size
                || bytes as u64 > u64::from(device_limits.max_storage_buffer_binding_size)
            {
                return Err(GpuArticulatedGroundContactError::Capacity);
            }
        }
        if coupling_rows.len().div_ceil(64)
            > device_limits.max_compute_workgroups_per_dimension as usize
            || systems.len() > device_limits.max_compute_workgroups_per_dimension as usize
            || dynamic_slot_count.div_ceil(64)
                > device_limits.max_compute_workgroups_per_dimension as usize
            || device_limits.max_storage_buffers_per_shader_stage < 8
        {
            return Err(GpuArticulatedGroundContactError::Capacity);
        }
        let system_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera articulated contact systems"),
            contents: bytemuck::cast_slice(&systems),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let sphere_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera articulated contact spheres"),
            contents: bytemuck::cast_slice(&spheres),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let (
            dynamic_shape_buffer,
            dynamic_pair_slots_buffer,
            dynamic_pipeline,
            dynamic_inactive_pipeline,
            dynamic_flags,
            dynamic_row_indices,
        ) = if dynamic_rows.iter().any(|range| !range.is_empty()) {
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tessera articulated dynamic shape metadata"),
                contents: bytemuck::cast_slice(&packed_dynamic_shapes),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let pair_slots = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tessera articulated dynamic shape pair slots"),
                contents: bytemuck::cast_slice(&dynamic_pair_slots),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Tessera articulated dynamic shape rows"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("gpu_articulated_dynamic_shape_rows.wgsl").into(),
                ),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Tessera articulated dynamic shape rows"),
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            let inactive_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Tessera articulated inactive dynamic rows"),
                    layout: None,
                    module: &shader,
                    entry_point: Some("clear_inactive"),
                    compilation_options: Default::default(),
                    cache: None,
                });
            let flags = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Tessera articulated dynamic candidate flags"),
                size: dynamic_slot_count as u64 * size_of::<u32>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let row_indices = dynamic_rows
                .iter()
                .flat_map(|range| range.clone())
                .map(checked_u32)
                .collect::<Result<Vec<_>, _>>()?;
            let row_indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tessera articulated dynamic row indices"),
                contents: bytemuck::cast_slice(&row_indices),
                usage: wgpu::BufferUsages::STORAGE,
            });
            (
                Some(buffer),
                Some(pair_slots),
                Some(pipeline),
                Some(inactive_pipeline),
                Some(flags),
                Some(row_indices),
            )
        } else {
            (None, None, None, None, None, None)
        };
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera articulated sphere contact"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("gpu_articulated_ground_contact.wgsl").into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tessera articulated sphere contact"),
            layout: None,
            module: &shader,
            entry_point: Some("resolve_ground_contacts"),
            compilation_options: Default::default(),
            cache: None,
        });
        let coupling_preparation = if coupling_rows.is_empty() {
            None
        } else {
            let coupling_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Tessera articulated coupling preparation"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("gpu_articulated_coupling.wgsl").into(),
                ),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Tessera articulated coupling preparation"),
                layout: None,
                module: &coupling_shader,
                entry_point: Some("prepare_couplings"),
                compilation_options: Default::default(),
                cache: None,
            });
            let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tessera articulated coupling row indices"),
                contents: bytemuck::cast_slice(&coupling_rows),
                usage: wgpu::BufferUsages::STORAGE,
            });
            Some(CouplingPreparation {
                pipeline,
                indices,
                positions: state.position_buffer().clone(),
                count: coupling_rows.len(),
            })
        };
        let contact_ranges: Vec<_> = systems
            .iter()
            .map(|system| {
                let start = system.indices[2] as usize;
                start..start + system.indices[3] as usize
            })
            .collect();
        let prescribed_indexed_rows = contact_ranges
            .iter()
            .map(|range| {
                range
                    .clone()
                    .filter(|&row| {
                        let mode = spheres[row].material[3];
                        ((52.0..=57.0).contains(&mode) || (72.0..=77.0).contains(&mode))
                            && if mode == 57.0 || mode == 77.0 {
                                spheres[row].plane[3] == 0.0
                            } else {
                                spheres[row].second_axis_end[3] == 0.0
                            }
                    })
                    .map(|row| {
                        let mode = spheres[row].material[3];
                        let count = if mode == 55.0 {
                            2
                        } else if mode == 56.0 {
                            1
                        } else {
                            4
                        };
                        (row, spheres[row].indices[0] as usize, count)
                    })
                    .collect()
            })
            .collect();
        let prescribed_capsule_rows = [
            GpuArticulatedCapsuleContactKind::Sphere,
            GpuArticulatedCapsuleContactKind::Capsule,
            GpuArticulatedCapsuleContactKind::Box,
            GpuArticulatedCapsuleContactKind::Axial,
            GpuArticulatedCapsuleContactKind::Convex,
        ]
        .map(|kind| {
            contact_ranges
                .iter()
                .map(|range| {
                    range
                        .clone()
                        .filter(|&row| kind.accepts(spheres[row].material[3]))
                        .map(|row| (row, spheres[row].indices[0] as usize))
                        .collect()
                })
                .collect()
        });
        let prescribed_axial_rows = [
            GpuArticulatedAxialContactKind::Sphere,
            GpuArticulatedAxialContactKind::Capsule,
            GpuArticulatedAxialContactKind::Box,
            GpuArticulatedAxialContactKind::Axial,
            GpuArticulatedAxialContactKind::Convex,
        ]
        .map(|kind| {
            contact_ranges
                .iter()
                .map(|range| {
                    range
                        .clone()
                        .filter(|&row| kind.accepts(spheres[row].material[3]))
                        .map(|row| (row, spheres[row].indices[2] as usize))
                        .collect()
                })
                .collect()
        });
        Ok(Self {
            device: device.clone(),
            pipeline,
            systems: system_buffer,
            prescribed_capsule_rows,
            prescribed_axial_rows,
            spheres: sphere_buffer,
            sphere_orbits: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tessera external sphere body origins"),
                contents: bytemuck::cast_slice(&vec![
                    PackedSphereOrbit::zeroed();
                    spheres.len().max(1)
                ]),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            }),
            sphere_orbit_enabled: vec![false; spheres.len()],
            static_sphere_rows,
            static_sphere_motion: static_sphere_centers
                .iter()
                .map(|rows| vec![[0.0; 6]; rows.len()])
                .collect(),
            static_sphere_centers,
            sphere_motion_pipeline: OnceLock::new(),
            integrated_sphere_centers: AtomicBool::new(false),
            integrated_sphere_box_poses: AtomicBool::new(false),
            integrated_capsule_box_poses: AtomicBool::new(false),
            integrated_box_box_poses: AtomicBool::new(false),
            integrated_axial_box_poses: AtomicBool::new(false),
            integrated_prescribed_convex_poses: AtomicBool::new(false),
            integrated_axial_convex_poses: AtomicBool::new(false),
            integrated_convex_rounded_poses: AtomicBool::new(false),
            static_capsule_sphere_rows,
            static_capsule_sphere_motion: static_capsule_sphere_centers
                .iter()
                .map(|rows| vec![[0.0; 6]; rows.len()])
                .collect(),
            integrated_capsule_sphere_centers: AtomicBool::new(false),
            static_capsule_sphere_centers,
            static_box_sphere_rows,
            static_box_sphere_motion: static_box_sphere_centers
                .iter()
                .map(|rows| vec![[0.0; 6]; rows.len()])
                .collect(),
            integrated_box_sphere_centers: AtomicBool::new(false),
            static_box_sphere_centers,
            static_axial_sphere_rows,
            static_axial_sphere_motion: static_axial_sphere_centers
                .iter()
                .map(|rows| vec![[0.0; 6]; rows.len()])
                .collect(),
            integrated_axial_sphere_centers: AtomicBool::new(false),
            static_axial_sphere_centers,
            static_convex_sphere_rows,
            static_convex_sphere_motion: static_convex_sphere_centers
                .iter()
                .map(|rows| vec![[0.0; 6]; rows.len()])
                .collect(),
            integrated_convex_sphere_centers: AtomicBool::new(false),
            static_convex_sphere_centers,
            static_sphere_box_rows,
            static_sphere_box_poses,
            static_capsule_box_rows,
            static_capsule_box_poses,
            static_box_box_rows,
            static_box_box_poses,
            scene_convex_rounded_rows,
            scene_convex_rounded_poses,
            static_axial_convex_rows,
            static_axial_convex_poses,
            static_convex_pair_rows,
            static_constraint_rows,
            prescribed_indexed_rows,
            static_convex_pair_poses,
            static_axial_box_rows,
            static_axial_box_poses,
            coupling_preparation,
            dynamic_shapes: dynamic_shape_buffer,
            dynamic_pair_slots: dynamic_pair_slots_buffer,
            dynamic_pipeline,
            dynamic_inactive_pipeline,
            dynamic_flags,
            dynamic_row_indices,
            dynamic_rows,
            friction_rows,
            friction_dimensions,
            timestep,
            poses: poses.link_pose_buffer().clone(),
            link_terms: mass.link_terms_buffer().clone(),
            inverse: mass.inverse_buffer_or_init().clone(),
            velocities: state.velocity_buffer().clone(),
            accelerations: mass.solution_buffer().clone(),
            state_status: state.status_buffer().clone(),
            environment_count: articulations.len(),
            sphere_count: contact_row_count,
            contact_ranges,
            link_ranges: poses.link_ranges().to_vec(),
            contact_activity: None,
            motion_layout,
            motion_wake: None,
            load_wake: None,
            actuation_wake: None,
            gravity_wake: None,
            mass_status: mass.status_buffer().clone(),
        })
    }

    /// Move reserved stationary spheres without rebuilding the contact batch.
    /// Input follows environment/static-sphere-pair order; radii and topology stay
    /// unchanged. Invalid input uploads nothing. Identical represented centers
    /// preserve history. Changed rows discard warm impulses and add owner wake
    /// requests without clearing geometry history or other pending requests.
    pub fn update_static_sphere_centers(
        &mut self,
        queue: &wgpu::Queue,
        centers: &[Vec<Vector3<f64>>],
    ) -> Result<bool, GpuArticulatedGroundContactError> {
        if centers.len() != self.static_sphere_rows.len()
            || centers
                .iter()
                .zip(&self.static_sphere_rows)
                .any(|(v, r)| v.len() != r.len())
        {
            return Err(GpuArticulatedGroundContactError::InvalidInput);
        }
        let packed = centers
            .iter()
            .map(|values| {
                values
                    .iter()
                    .map(|center| {
                        Ok([
                            finite_f32(center.x)?,
                            finite_f32(center.y)?,
                            finite_f32(center.z)?,
                        ])
                    })
                    .collect::<Result<Vec<_>, GpuArticulatedGroundContactError>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut changed = false;
        for (env, values) in packed.iter().enumerate() {
            for (pair, center) in values.iter().enumerate() {
                if !self.integrated_sphere_centers.load(Ordering::Relaxed)
                    && self.static_sphere_centers[env][pair] == *center
                {
                    continue;
                }
                let (row, owner) = self.static_sphere_rows[env][pair];
                let start = row as u64 * size_of::<PackedSphere>() as u64;
                queue.write_buffer(
                    &self.spheres,
                    start + core::mem::offset_of!(PackedSphere, other_center_radius) as u64,
                    bytemuck::cast_slice(center),
                );
                queue.write_buffer(
                    &self.spheres,
                    start + core::mem::offset_of!(PackedSphere, impulses) as u64,
                    bytemuck::cast_slice(&[0.0f32; 4]),
                );
                if let Some(activity) = &self.contact_activity {
                    queue.write_buffer(
                        &activity.wake_requests,
                        owner as u64 * 4,
                        &1u32.to_ne_bytes(),
                    );
                }
                changed = true;
            }
        }
        self.static_sphere_centers = packed;
        self.integrated_sphere_centers
            .store(false, Ordering::Relaxed);
        Ok(changed)
    }

    /// Integrate prescribed external sphere centers on the GPU after contact solving.
    /// A later explicit center update always overrides the integrated positions.
    pub fn encode_static_sphere_integration(&self, encoder: &mut wgpu::CommandEncoder) {
        let moving = |motions: &[Vec<[f32; 6]>]| {
            motions
                .iter()
                .flatten()
                .any(|motion| motion[..3].iter().any(|value| *value != 0.0))
        };
        let orbit = |rows: &[Vec<(usize, usize)>]| {
            rows.iter()
                .flatten()
                .any(|&(row, _)| self.sphere_orbit_enabled[row])
        };
        let spheres = moving(&self.static_sphere_motion) || orbit(&self.static_sphere_rows);
        let capsules =
            moving(&self.static_capsule_sphere_motion) || orbit(&self.static_capsule_sphere_rows);
        let boxes = moving(&self.static_box_sphere_motion) || orbit(&self.static_box_sphere_rows);
        let axial =
            moving(&self.static_axial_sphere_motion) || orbit(&self.static_axial_sphere_rows);
        let convex =
            moving(&self.static_convex_sphere_motion) || orbit(&self.static_convex_sphere_rows);
        let prescribed_boxes = [
            orbit(&self.static_sphere_box_rows),
            orbit(&self.static_capsule_box_rows),
            orbit(&self.static_box_box_rows),
            orbit(&self.static_axial_box_rows),
            orbit(&self.static_convex_pair_rows),
        ];
        if !spheres
            && !capsules
            && !boxes
            && !axial
            && !convex
            && !prescribed_boxes.iter().any(|active| *active)
            && !self.prescribed_capsule_rows.iter().any(|rows| orbit(rows))
            && !self.prescribed_axial_rows.iter().any(|rows| orbit(rows))
            && !orbit(&self.static_axial_convex_rows)
            && !self
                .prescribed_indexed_rows
                .iter()
                .flatten()
                .any(|&(row, _, _)| self.sphere_orbit_enabled[row])
            && !self
                .static_constraint_rows
                .iter()
                .flatten()
                .any(|&(row, _, _)| self.sphere_orbit_enabled[row])
            && !self
                .scene_convex_rounded_rows
                .iter()
                .flatten()
                .any(|&(row, _, _)| self.sphere_orbit_enabled[row])
        {
            return;
        }
        if orbit(&self.static_axial_convex_rows) {
            self.integrated_axial_convex_poses
                .store(true, Ordering::Relaxed);
        }
        if self
            .scene_convex_rounded_rows
            .iter()
            .flatten()
            .any(|&(row, _, _)| self.sphere_orbit_enabled[row])
        {
            self.integrated_convex_rounded_poses
                .store(true, Ordering::Relaxed);
        }
        for (active, flag) in prescribed_boxes.iter().zip([
            &self.integrated_sphere_box_poses,
            &self.integrated_capsule_box_poses,
            &self.integrated_box_box_poses,
            &self.integrated_axial_box_poses,
            &self.integrated_prescribed_convex_poses,
        ]) {
            if *active {
                flag.store(true, Ordering::Relaxed);
            }
        }
        if spheres {
            self.integrated_sphere_centers
                .store(true, Ordering::Relaxed);
        }
        if capsules {
            self.integrated_capsule_sphere_centers
                .store(true, Ordering::Relaxed);
        }
        if boxes {
            self.integrated_box_sphere_centers
                .store(true, Ordering::Relaxed);
        }
        if axial {
            self.integrated_axial_sphere_centers
                .store(true, Ordering::Relaxed);
        }
        if convex {
            self.integrated_convex_sphere_centers
                .store(true, Ordering::Relaxed);
        }
        let pipeline = self.sphere_motion_pipeline.get_or_init(|| {
            let module = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Tessera prescribed external sphere integration"),
                    source: wgpu::ShaderSource::Wgsl(
                        include_str!("gpu_articulated_external_sphere_motion.wgsl").into(),
                    ),
                });
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Tessera prescribed external sphere integration"),
                    layout: None,
                    module: &module,
                    entry_point: Some("integrate"),
                    compilation_options: Default::default(),
                    cache: None,
                })
        });
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera prescribed sphere integration bindings"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.spheres.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.systems.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.state_status.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.sphere_orbits.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(
            (self.systems.size() / size_of::<PackedSystem>() as u64).div_ceil(64) as u32,
            1,
            1,
        );
    }

    /// Set external sphere velocities used by contact restitution and friction.
    /// Geometry is moved separately with `update_static_sphere_centers`.
    /// Input follows environment/static-sphere-pair order. Invalid input uploads
    /// nothing; changed rows invalidate warm impulses and request owner wake.
    pub fn update_static_sphere_motion(
        &mut self,
        queue: &wgpu::Queue,
        motions: &[Vec<GpuArticulatedSphereMotion>],
    ) -> Result<bool, GpuArticulatedGroundContactError> {
        self.update_external_sphere_motion(queue, motions, ExternalSphereContactKind::Sphere)
    }

    /// Set external sphere motion for reserved capsule/static-sphere pairs.
    /// Validation and warm-start invalidation follow `update_static_sphere_motion`.
    pub fn update_static_capsule_sphere_motion(
        &mut self,
        queue: &wgpu::Queue,
        motions: &[Vec<GpuArticulatedSphereMotion>],
    ) -> Result<bool, GpuArticulatedGroundContactError> {
        self.update_external_sphere_motion(queue, motions, ExternalSphereContactKind::Capsule)
    }

    /// Set external sphere motion for reserved box/static-sphere pairs.
    /// Validation and warm-start invalidation follow `update_static_sphere_motion`.
    pub fn update_static_box_sphere_motion(
        &mut self,
        queue: &wgpu::Queue,
        motions: &[Vec<GpuArticulatedSphereMotion>],
    ) -> Result<bool, GpuArticulatedGroundContactError> {
        self.update_external_sphere_motion(queue, motions, ExternalSphereContactKind::Box)
    }

    /// Set external sphere motion for dynamic axial/static-sphere pairs.
    /// Input excludes pairs with a stationary axial shape.
    pub fn update_static_axial_sphere_motion(
        &mut self,
        queue: &wgpu::Queue,
        motions: &[Vec<GpuArticulatedSphereMotion>],
    ) -> Result<bool, GpuArticulatedGroundContactError> {
        self.update_external_sphere_motion(queue, motions, ExternalSphereContactKind::Axial)
    }

    /// Set external sphere motion for dynamic convex/static-sphere pairs.
    /// Input follows the same ordering as `update_static_convex_sphere_centers`.
    pub fn update_static_convex_sphere_motion(
        &mut self,
        queue: &wgpu::Queue,
        motions: &[Vec<GpuArticulatedSphereMotion>],
    ) -> Result<bool, GpuArticulatedGroundContactError> {
        self.update_external_sphere_motion(queue, motions, ExternalSphereContactKind::Convex)
    }

    /// Replace all external sphere motion after validating every shape category.
    /// Invalid input uploads nothing. Pair ordering matches the center update APIs.
    pub fn update_external_sphere_motions(
        &mut self,
        queue: &wgpu::Queue,
        motions: &GpuArticulatedExternalSphereMotions,
    ) -> Result<bool, GpuArticulatedGroundContactError> {
        for (values, rows) in [
            (&motions.spheres, &self.static_sphere_rows),
            (&motions.capsules, &self.static_capsule_sphere_rows),
            (&motions.boxes, &self.static_box_sphere_rows),
            (&motions.axial, &self.static_axial_sphere_rows),
            (&motions.convex, &self.static_convex_sphere_rows),
        ] {
            drop(Self::pack_external_sphere_motion(values, rows)?);
        }
        let mut changed = false;
        for (values, kind) in [
            (&motions.spheres, ExternalSphereContactKind::Sphere),
            (&motions.capsules, ExternalSphereContactKind::Capsule),
            (&motions.boxes, ExternalSphereContactKind::Box),
            (&motions.axial, ExternalSphereContactKind::Axial),
            (&motions.convex, ExternalSphereContactKind::Convex),
        ] {
            changed |= self.update_external_sphere_motion(queue, values, kind)?;
        }
        Ok(changed)
    }

    /// Set body origins for external spheres paired with link spheres.
    /// Current centers define the rigid offset. None stops the corresponding
    /// sphere. Angular velocity comes from the preceding motion update.
    /// Explicit origins reset warm impulses. Serialize queue use externally.
    pub fn update_static_sphere_orbits(
        &mut self,
        queue: &wgpu::Queue,
        orbits: &[Vec<Option<GpuArticulatedSphereOrbit>>],
    ) -> Result<bool, GpuArticulatedGroundContactError> {
        self.update_external_sphere_orbits(queue, orbits, ExternalSphereContactKind::Sphere)
    }

    /// Set body origins for external spheres paired with link capsules.
    pub fn update_static_capsule_sphere_orbits(
        &mut self,
        queue: &wgpu::Queue,
        orbits: &[Vec<Option<GpuArticulatedSphereOrbit>>],
    ) -> Result<bool, GpuArticulatedGroundContactError> {
        self.update_external_sphere_orbits(queue, orbits, ExternalSphereContactKind::Capsule)
    }

    /// Set body origins for external spheres paired with link boxes.
    pub fn update_static_box_sphere_orbits(
        &mut self,
        queue: &wgpu::Queue,
        orbits: &[Vec<Option<GpuArticulatedSphereOrbit>>],
    ) -> Result<bool, GpuArticulatedGroundContactError> {
        self.update_external_sphere_orbits(queue, orbits, ExternalSphereContactKind::Box)
    }

    /// Set body origins for external spheres paired with moving cylinders or cones.
    pub fn update_static_axial_sphere_orbits(
        &mut self,
        queue: &wgpu::Queue,
        orbits: &[Vec<Option<GpuArticulatedSphereOrbit>>],
    ) -> Result<bool, GpuArticulatedGroundContactError> {
        self.update_external_sphere_orbits(queue, orbits, ExternalSphereContactKind::Axial)
    }

    /// Set body origins for external spheres paired with moving convex hulls.
    pub fn update_static_convex_sphere_orbits(
        &mut self,
        queue: &wgpu::Queue,
        orbits: &[Vec<Option<GpuArticulatedSphereOrbit>>],
    ) -> Result<bool, GpuArticulatedGroundContactError> {
        self.update_external_sphere_orbits(queue, orbits, ExternalSphereContactKind::Convex)
    }

    fn update_external_sphere_orbits(
        &mut self,
        queue: &wgpu::Queue,
        orbits: &[Vec<Option<GpuArticulatedSphereOrbit>>],
        kind: ExternalSphereContactKind,
    ) -> Result<bool, GpuArticulatedGroundContactError> {
        let (rows, previous) = match kind {
            ExternalSphereContactKind::Sphere => {
                (&self.static_sphere_rows, &self.static_sphere_motion)
            }
            ExternalSphereContactKind::Capsule => (
                &self.static_capsule_sphere_rows,
                &self.static_capsule_sphere_motion,
            ),
            ExternalSphereContactKind::Box => {
                (&self.static_box_sphere_rows, &self.static_box_sphere_motion)
            }
            ExternalSphereContactKind::Axial => (
                &self.static_axial_sphere_rows,
                &self.static_axial_sphere_motion,
            ),
            ExternalSphereContactKind::Convex => (
                &self.static_convex_sphere_rows,
                &self.static_convex_sphere_motion,
            ),
        };
        if orbits.len() != rows.len()
            || orbits
                .iter()
                .zip(rows)
                .any(|(orbits, rows)| orbits.len() != rows.len())
        {
            return Err(GpuArticulatedGroundContactError::InvalidInput);
        }
        let packed = orbits
            .iter()
            .map(|environment| {
                environment
                    .iter()
                    .map(|orbit| {
                        orbit
                            .map(|orbit| {
                                if (orbit.orientation.quaternion().norm_squared() - 1.0).abs()
                                    > 1e-6
                                {
                                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                                }
                                let quaternion = orbit.orientation.quaternion();
                                Ok(PackedSphereOrbit {
                                    origin: [
                                        finite_f32(orbit.origin.x)?,
                                        finite_f32(orbit.origin.y)?,
                                        finite_f32(orbit.origin.z)?,
                                        1.0,
                                    ],
                                    linear: [
                                        finite_f32(orbit.linear_velocity.x)?,
                                        finite_f32(orbit.linear_velocity.y)?,
                                        finite_f32(orbit.linear_velocity.z)?,
                                        0.0,
                                    ],
                                    orientation: [
                                        finite_f32(quaternion.i)?,
                                        finite_f32(quaternion.j)?,
                                        finite_f32(quaternion.k)?,
                                        finite_f32(quaternion.w)?,
                                    ],
                                    translation_anchor: [0.0; 4],
                                })
                            })
                            .transpose()
                    })
                    .collect::<Result<Vec<_>, GpuArticulatedGroundContactError>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let snapshot = self.readback_external_sphere_centers(queue)?;
        let centers = match kind {
            ExternalSphereContactKind::Sphere => snapshot.spheres,
            ExternalSphereContactKind::Capsule => snapshot.capsules,
            ExternalSphereContactKind::Box => snapshot.boxes,
            ExternalSphereContactKind::Axial => snapshot.axial,
            ExternalSphereContactKind::Convex => snapshot.convex,
        };
        let mut motions = Vec::with_capacity(orbits.len());
        for (environment, values) in packed.iter().enumerate() {
            let mut environment_motions = Vec::with_capacity(values.len());
            for (pair, orbit) in values.iter().enumerate() {
                let motion = if let Some(orbit) = orbit {
                    let angular = Vector3::new(
                        previous[environment][pair][3] as f64,
                        previous[environment][pair][4] as f64,
                        previous[environment][pair][5] as f64,
                    );
                    let origin = Vector3::new(
                        orbit.origin[0] as f64,
                        orbit.origin[1] as f64,
                        orbit.origin[2] as f64,
                    );
                    let linear = Vector3::new(
                        orbit.linear[0] as f64,
                        orbit.linear[1] as f64,
                        orbit.linear[2] as f64,
                    );
                    GpuArticulatedSphereMotion {
                        linear_velocity: linear
                            + angular.cross(&(centers[environment][pair] - origin)),
                        angular_velocity: angular,
                    }
                } else {
                    GpuArticulatedSphereMotion::default()
                };
                environment_motions.push(motion);
            }
            motions.push(environment_motions);
        }
        drop(Self::pack_external_sphere_motion(&motions, rows)?);
        let rows = rows.clone();
        let mut changed = self.update_external_sphere_motion(queue, &motions, kind)?;
        for (environment, values) in packed.iter().enumerate() {
            for (pair, orbit) in values.iter().enumerate() {
                if let Some(orbit) = orbit {
                    let (row, owner) = rows[environment][pair];
                    queue.write_buffer(
                        &self.sphere_orbits,
                        row as u64 * size_of::<PackedSphereOrbit>() as u64,
                        bytemuck::bytes_of(orbit),
                    );
                    self.sphere_orbit_enabled[row] = true;
                    queue.write_buffer(
                        &self.spheres,
                        row as u64 * size_of::<PackedSphere>() as u64
                            + core::mem::offset_of!(PackedSphere, impulses) as u64,
                        bytemuck::cast_slice(&[0.0f32; 4]),
                    );
                    if let Some(activity) = &self.contact_activity {
                        queue.write_buffer(
                            &activity.wake_requests,
                            owner as u64 * 4,
                            &1u32.to_ne_bytes(),
                        );
                    }
                    changed = true;
                }
            }
        }
        Ok(changed)
    }

    fn pack_external_sphere_motion(
        motions: &[Vec<GpuArticulatedSphereMotion>],
        rows: &[Vec<(usize, usize)>],
    ) -> Result<Vec<Vec<[f32; 6]>>, GpuArticulatedGroundContactError> {
        if motions.len() != rows.len()
            || motions
                .iter()
                .zip(rows)
                .any(|(motions, rows)| motions.len() != rows.len())
        {
            return Err(GpuArticulatedGroundContactError::InvalidInput);
        }
        motions
            .iter()
            .map(|motions| {
                motions
                    .iter()
                    .map(|motion| {
                        Ok([
                            finite_f32(motion.linear_velocity.x)?,
                            finite_f32(motion.linear_velocity.y)?,
                            finite_f32(motion.linear_velocity.z)?,
                            finite_f32(motion.angular_velocity.x)?,
                            finite_f32(motion.angular_velocity.y)?,
                            finite_f32(motion.angular_velocity.z)?,
                        ])
                    })
                    .collect::<Result<Vec<_>, GpuArticulatedGroundContactError>>()
            })
            .collect()
    }

    fn update_external_sphere_motion(
        &mut self,
        queue: &wgpu::Queue,
        motions: &[Vec<GpuArticulatedSphereMotion>],
        kind: ExternalSphereContactKind,
    ) -> Result<bool, GpuArticulatedGroundContactError> {
        let rows = match kind {
            ExternalSphereContactKind::Sphere => &self.static_sphere_rows,
            ExternalSphereContactKind::Capsule => &self.static_capsule_sphere_rows,
            ExternalSphereContactKind::Box => &self.static_box_sphere_rows,
            ExternalSphereContactKind::Axial => &self.static_axial_sphere_rows,
            ExternalSphereContactKind::Convex => &self.static_convex_sphere_rows,
        };
        let previous = match kind {
            ExternalSphereContactKind::Sphere => &self.static_sphere_motion,
            ExternalSphereContactKind::Capsule => &self.static_capsule_sphere_motion,
            ExternalSphereContactKind::Box => &self.static_box_sphere_motion,
            ExternalSphereContactKind::Axial => &self.static_axial_sphere_motion,
            ExternalSphereContactKind::Convex => &self.static_convex_sphere_motion,
        };
        let packed = Self::pack_external_sphere_motion(motions, rows)?;
        let mut changed = false;
        for (environment, motions) in packed.iter().enumerate() {
            for (pair, motion) in motions.iter().enumerate() {
                let (row, owner) = rows[environment][pair];
                if *motion == previous[environment][pair] && !self.sphere_orbit_enabled[row] {
                    continue;
                }
                if self.sphere_orbit_enabled[row] {
                    queue.write_buffer(
                        &self.sphere_orbits,
                        row as u64 * size_of::<PackedSphereOrbit>() as u64,
                        bytemuck::bytes_of(&PackedSphereOrbit::zeroed()),
                    );
                    self.sphere_orbit_enabled[row] = false;
                }
                let offset = row as u64 * size_of::<PackedSphere>() as u64;
                queue.write_buffer(
                    &self.spheres,
                    offset + core::mem::offset_of!(PackedSphere, prescribed_linear) as u64,
                    bytemuck::cast_slice(&motion[..3]),
                );
                queue.write_buffer(
                    &self.spheres,
                    offset + core::mem::offset_of!(PackedSphere, prescribed_angular) as u64,
                    bytemuck::cast_slice(&motion[3..]),
                );
                queue.write_buffer(
                    &self.spheres,
                    offset + core::mem::offset_of!(PackedSphere, impulses) as u64,
                    bytemuck::cast_slice(&[0.0f32; 4]),
                );
                if let Some(activity) = &self.contact_activity {
                    queue.write_buffer(
                        &activity.wake_requests,
                        owner as u64 * 4,
                        &1u32.to_ne_bytes(),
                    );
                }
                changed = true;
            }
        }
        match kind {
            ExternalSphereContactKind::Sphere => self.static_sphere_motion = packed,
            ExternalSphereContactKind::Capsule => self.static_capsule_sphere_motion = packed,
            ExternalSphereContactKind::Box => self.static_box_sphere_motion = packed,
            ExternalSphereContactKind::Axial => self.static_axial_sphere_motion = packed,
            ExternalSphereContactKind::Convex => self.static_convex_sphere_motion = packed,
        }
        Ok(changed)
    }

    /// Move reserved stationary spheres without rebuilding the contact batch.
    /// Input follows environment/static-capsule-sphere-pair order; radii and topology stay
    /// unchanged. Invalid input uploads nothing. Identical represented centers
    /// preserve history. Changed rows discard warm impulses and add owner wake
    /// requests without clearing geometry history or other pending requests.
    pub fn update_static_capsule_sphere_centers(
        &mut self,
        queue: &wgpu::Queue,
        centers: &[Vec<Vector3<f64>>],
    ) -> Result<bool, GpuArticulatedGroundContactError> {
        if centers.len() != self.static_capsule_sphere_rows.len()
            || centers
                .iter()
                .zip(&self.static_capsule_sphere_rows)
                .any(|(v, r)| v.len() != r.len())
        {
            return Err(GpuArticulatedGroundContactError::InvalidInput);
        }
        let packed = centers
            .iter()
            .map(|values| {
                values
                    .iter()
                    .map(|center| {
                        Ok([
                            finite_f32(center.x)?,
                            finite_f32(center.y)?,
                            finite_f32(center.z)?,
                        ])
                    })
                    .collect::<Result<Vec<_>, GpuArticulatedGroundContactError>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut changed = false;
        for (env, values) in packed.iter().enumerate() {
            for (pair, center) in values.iter().enumerate() {
                if !self
                    .integrated_capsule_sphere_centers
                    .load(Ordering::Relaxed)
                    && self.static_capsule_sphere_centers[env][pair] == *center
                {
                    continue;
                }
                let (row, owner) = self.static_capsule_sphere_rows[env][pair];
                let start = row as u64 * size_of::<PackedSphere>() as u64;
                queue.write_buffer(
                    &self.spheres,
                    start + core::mem::offset_of!(PackedSphere, other_center_radius) as u64,
                    bytemuck::cast_slice(center),
                );
                queue.write_buffer(
                    &self.spheres,
                    start + core::mem::offset_of!(PackedSphere, impulses) as u64,
                    bytemuck::cast_slice(&[0.0f32; 4]),
                );
                if let Some(activity) = &self.contact_activity {
                    queue.write_buffer(
                        &activity.wake_requests,
                        owner as u64 * 4,
                        &1u32.to_ne_bytes(),
                    );
                }
                changed = true;
            }
        }
        self.static_capsule_sphere_centers = packed;
        self.integrated_capsule_sphere_centers
            .store(false, Ordering::Relaxed);
        Ok(changed)
    }

    /// Move reserved stationary spheres without rebuilding the contact batch.
    /// Input follows environment/static-box-sphere-pair order; radii and topology stay
    /// unchanged. Invalid input uploads nothing. Identical represented centers
    /// preserve history. Changed rows discard warm impulses and add owner wake
    /// requests without clearing geometry history or other pending requests.
    pub fn update_static_box_sphere_centers(
        &mut self,
        queue: &wgpu::Queue,
        centers: &[Vec<Vector3<f64>>],
    ) -> Result<bool, GpuArticulatedGroundContactError> {
        if centers.len() != self.static_box_sphere_rows.len()
            || centers
                .iter()
                .zip(&self.static_box_sphere_rows)
                .any(|(v, r)| v.len() != r.len())
        {
            return Err(GpuArticulatedGroundContactError::InvalidInput);
        }
        let packed = centers
            .iter()
            .map(|values| {
                values
                    .iter()
                    .map(|center| {
                        Ok([
                            finite_f32(center.x)?,
                            finite_f32(center.y)?,
                            finite_f32(center.z)?,
                        ])
                    })
                    .collect::<Result<Vec<_>, GpuArticulatedGroundContactError>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut changed = false;
        for (env, values) in packed.iter().enumerate() {
            for (pair, center) in values.iter().enumerate() {
                if !self.integrated_box_sphere_centers.load(Ordering::Relaxed)
                    && self.static_box_sphere_centers[env][pair] == *center
                {
                    continue;
                }
                let (row, owner) = self.static_box_sphere_rows[env][pair];
                let start = row as u64 * size_of::<PackedSphere>() as u64;
                queue.write_buffer(
                    &self.spheres,
                    start + core::mem::offset_of!(PackedSphere, plane) as u64,
                    bytemuck::cast_slice(center),
                );
                queue.write_buffer(
                    &self.spheres,
                    start + core::mem::offset_of!(PackedSphere, impulses) as u64,
                    bytemuck::cast_slice(&[0.0f32; 4]),
                );
                if let Some(activity) = &self.contact_activity {
                    queue.write_buffer(
                        &activity.wake_requests,
                        owner as u64 * 4,
                        &1u32.to_ne_bytes(),
                    );
                }
                changed = true;
            }
        }
        self.static_box_sphere_centers = packed;
        self.integrated_box_sphere_centers
            .store(false, Ordering::Relaxed);
        Ok(changed)
    }

    /// Move reserved stationary spheres without rebuilding the contact batch.
    /// Input follows environment/static-axial-sphere-pair order, filtered to axial_is_static == false; radii and topology stay
    /// unchanged. Invalid input uploads nothing. Identical represented centers
    /// preserve history. Changed rows discard warm impulses and add owner wake
    /// requests without clearing geometry history or other pending requests.
    pub fn update_static_axial_sphere_centers(
        &mut self,
        queue: &wgpu::Queue,
        centers: &[Vec<Vector3<f64>>],
    ) -> Result<bool, GpuArticulatedGroundContactError> {
        if centers.len() != self.static_axial_sphere_rows.len()
            || centers
                .iter()
                .zip(&self.static_axial_sphere_rows)
                .any(|(v, r)| v.len() != r.len())
        {
            return Err(GpuArticulatedGroundContactError::InvalidInput);
        }
        let packed = centers
            .iter()
            .map(|values| {
                values
                    .iter()
                    .map(|center| {
                        Ok([
                            finite_f32(center.x)?,
                            finite_f32(center.y)?,
                            finite_f32(center.z)?,
                        ])
                    })
                    .collect::<Result<Vec<_>, GpuArticulatedGroundContactError>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut changed = false;
        for (env, values) in packed.iter().enumerate() {
            for (pair, center) in values.iter().enumerate() {
                if !self.integrated_axial_sphere_centers.load(Ordering::Relaxed)
                    && self.static_axial_sphere_centers[env][pair] == *center
                {
                    continue;
                }
                let (row, owner) = self.static_axial_sphere_rows[env][pair];
                let start = row as u64 * size_of::<PackedSphere>() as u64;
                queue.write_buffer(
                    &self.spheres,
                    start + core::mem::offset_of!(PackedSphere, plane) as u64,
                    bytemuck::cast_slice(center),
                );
                queue.write_buffer(
                    &self.spheres,
                    start + core::mem::offset_of!(PackedSphere, impulses) as u64,
                    bytemuck::cast_slice(&[0.0f32; 4]),
                );
                if let Some(activity) = &self.contact_activity {
                    queue.write_buffer(
                        &activity.wake_requests,
                        owner as u64 * 4,
                        &1u32.to_ne_bytes(),
                    );
                }
                changed = true;
            }
        }
        self.static_axial_sphere_centers = packed;
        self.integrated_axial_sphere_centers
            .store(false, Ordering::Relaxed);
        Ok(changed)
    }

    /// Move reserved stationary spheres without rebuilding the contact batch.
    /// Input follows environment/static-convex-sphere-pair order; radii and topology stay
    /// unchanged. Invalid input uploads nothing. Identical represented centers
    /// preserve history. Changed rows discard warm impulses and add owner wake
    /// requests without clearing geometry history or other pending requests.
    pub fn update_static_convex_sphere_centers(
        &mut self,
        queue: &wgpu::Queue,
        centers: &[Vec<Vector3<f64>>],
    ) -> Result<bool, GpuArticulatedGroundContactError> {
        if centers.len() != self.static_convex_sphere_rows.len()
            || centers
                .iter()
                .zip(&self.static_convex_sphere_rows)
                .any(|(v, r)| v.len() != r.len())
        {
            return Err(GpuArticulatedGroundContactError::InvalidInput);
        }
        let packed = centers
            .iter()
            .map(|values| {
                values
                    .iter()
                    .map(|center| {
                        Ok([
                            finite_f32(center.x)?,
                            finite_f32(center.y)?,
                            finite_f32(center.z)?,
                        ])
                    })
                    .collect::<Result<Vec<_>, GpuArticulatedGroundContactError>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut changed = false;
        for (env, values) in packed.iter().enumerate() {
            for (pair, center) in values.iter().enumerate() {
                if !self
                    .integrated_convex_sphere_centers
                    .load(Ordering::Relaxed)
                    && self.static_convex_sphere_centers[env][pair] == *center
                {
                    continue;
                }
                let (row, owner) = self.static_convex_sphere_rows[env][pair];
                let start = row as u64 * size_of::<PackedSphere>() as u64;
                queue.write_buffer(
                    &self.spheres,
                    start + core::mem::offset_of!(PackedSphere, plane) as u64,
                    bytemuck::cast_slice(center),
                );
                queue.write_buffer(
                    &self.spheres,
                    start + core::mem::offset_of!(PackedSphere, second_axis_end) as u64,
                    bytemuck::cast_slice(center),
                );
                queue.write_buffer(
                    &self.spheres,
                    start + core::mem::offset_of!(PackedSphere, impulses) as u64,
                    bytemuck::cast_slice(&[0.0f32; 4]),
                );
                if let Some(activity) = &self.contact_activity {
                    queue.write_buffer(
                        &activity.wake_requests,
                        owner as u64 * 4,
                        &1u32.to_ne_bytes(),
                    );
                }
                changed = true;
            }
        }
        self.static_convex_sphere_centers = packed;
        self.integrated_convex_sphere_centers
            .store(false, Ordering::Relaxed);
        Ok(changed)
    }

    /// Configure prescribed bodies for reserved link-sphere/world-box pairs.
    /// The collision pose remains unchanged. Supply the current body origin and
    /// orientation; the box may have a fixed local offset from that body frame.
    /// None stops motion while retaining the current GPU collision pose. Validate
    /// the entire packed input before uploading, and wake affected link owners.
    pub fn update_prescribed_sphere_box_bodies(
        &mut self,
        queue: &wgpu::Queue,
        bodies: &[Vec<Option<crate::gpu_kinematic_body::GpuKinematicBody>>],
    ) -> Result<(), GpuArticulatedGroundContactError> {
        self.update_prescribed_box_bodies(queue, GpuArticulatedBoxContactKind::Sphere, bodies)
    }

    fn prescribed_box_mapping(&self, kind: GpuArticulatedBoxContactKind) -> &[Vec<(usize, usize)>] {
        match kind {
            GpuArticulatedBoxContactKind::Sphere => &self.static_sphere_box_rows,
            GpuArticulatedBoxContactKind::Capsule => &self.static_capsule_box_rows,
            GpuArticulatedBoxContactKind::Box => &self.static_box_box_rows,
            GpuArticulatedBoxContactKind::Axial => &self.static_axial_box_rows,
            GpuArticulatedBoxContactKind::Convex => &self.static_convex_pair_rows,
        }
    }

    /// Configure prescribed world boxes for one reserved contact family.
    /// Each input addresses a pair, including all of its manifold rows. The
    /// current collision pose is retained. None stops at that pose. Validate all
    /// inputs before uploads; serialize commands on the owning queue externally.
    pub fn update_prescribed_box_bodies(
        &mut self,
        queue: &wgpu::Queue,
        kind: GpuArticulatedBoxContactKind,
        bodies: &[Vec<Option<crate::gpu_kinematic_body::GpuKinematicBody>>],
    ) -> Result<(), GpuArticulatedGroundContactError> {
        let mapping = self.prescribed_box_mapping(kind);
        if bodies.len() != mapping.len()
            || bodies.iter().zip(mapping).any(|(b, r)| b.len() != r.len())
        {
            return Err(GpuArticulatedGroundContactError::InvalidInput);
        }
        let mut inputs = Vec::new();
        for (environment, values) in bodies.iter().enumerate() {
            for (pair, body) in values.iter().enumerate() {
                let (row, owner) = mapping[environment][pair];
                let (orbit, angular) = if let Some(body) = body {
                    let q = body.pose.rotation.quaternion();
                    if !q.norm_squared().is_finite() || (q.norm_squared() - 1.0).abs() > 1e-6 {
                        return Err(GpuArticulatedGroundContactError::InvalidInput);
                    }
                    let origin = body.pose.translation.vector;
                    let v = body.linear_velocity;
                    let w = body.angular_velocity;
                    (
                        PackedSphereOrbit {
                            origin: [
                                finite_f32(origin.x)?,
                                finite_f32(origin.y)?,
                                finite_f32(origin.z)?,
                                1.0,
                            ],
                            linear: [finite_f32(v.x)?, finite_f32(v.y)?, finite_f32(v.z)?, 0.0],
                            orientation: [
                                finite_f32(q.i)?,
                                finite_f32(q.j)?,
                                finite_f32(q.k)?,
                                finite_f32(q.w)?,
                            ],
                            translation_anchor: [0.0; 4],
                        },
                        [finite_f32(w.x)?, finite_f32(w.y)?, finite_f32(w.z)?, 0.0],
                    )
                } else {
                    (PackedSphereOrbit::zeroed(), [0.0; 4])
                };
                for index in row..row + kind.row_count() {
                    inputs.push((index, owner, orbit, angular));
                }
            }
        }
        let rows = self.read_prescribed_box_rows(queue)?;
        let uploads = inputs
            .into_iter()
            .map(|(row, owner, orbit, angular)| {
                let center = Vector3::new(
                    rows[row].plane[0] as f64,
                    rows[row].plane[1] as f64,
                    rows[row].plane[2] as f64,
                );
                let origin = Vector3::new(
                    orbit.origin[0] as f64,
                    orbit.origin[1] as f64,
                    orbit.origin[2] as f64,
                );
                let w = Vector3::new(angular[0] as f64, angular[1] as f64, angular[2] as f64);
                let v = Vector3::new(
                    orbit.linear[0] as f64,
                    orbit.linear[1] as f64,
                    orbit.linear[2] as f64,
                ) + w.cross(&(center - origin));
                Ok((
                    row,
                    owner,
                    orbit,
                    angular,
                    [finite_f32(v.x)?, finite_f32(v.y)?, finite_f32(v.z)?, 0.0],
                ))
            })
            .collect::<Result<Vec<_>, GpuArticulatedGroundContactError>>()?;
        for (row, owner, orbit, angular, linear) in uploads {
            let start = row as u64 * size_of::<PackedSphere>() as u64;
            queue.write_buffer(
                &self.sphere_orbits,
                row as u64 * size_of::<PackedSphereOrbit>() as u64,
                bytemuck::bytes_of(&orbit),
            );
            queue.write_buffer(
                &self.spheres,
                start + offset_of!(PackedSphere, prescribed_linear) as u64,
                bytemuck::cast_slice(&linear),
            );
            queue.write_buffer(
                &self.spheres,
                start + offset_of!(PackedSphere, prescribed_angular) as u64,
                bytemuck::cast_slice(&angular),
            );
            queue.write_buffer(
                &self.spheres,
                start + offset_of!(PackedSphere, impulses) as u64,
                bytemuck::cast_slice(&[0.0f32; 4]),
            );
            self.sphere_orbit_enabled[row] = orbit.origin[3] != 0.0;
            if let Some(activity) = &self.contact_activity {
                queue.write_buffer(
                    &activity.wake_requests,
                    owner as u64 * 4,
                    &1u32.to_ne_bytes(),
                );
            }
        }
        Ok(())
    }

    /// Configure reserved world-cylinder or world-cone contact rows.
    /// Validate every input before uploads; None stops at the current collision pose.
    pub fn update_prescribed_axial_bodies(
        &mut self,
        queue: &wgpu::Queue,
        kind: GpuArticulatedAxialContactKind,
        bodies: &[Vec<Option<crate::gpu_kinematic_body::GpuKinematicBody>>],
    ) -> Result<(), GpuArticulatedGroundContactError> {
        let mapping = &self.prescribed_axial_rows[kind.index()];
        if bodies.len() != mapping.len()
            || bodies.iter().zip(mapping).any(|(b, r)| b.len() != r.len())
        {
            return Err(GpuArticulatedGroundContactError::InvalidInput);
        }
        let mut inputs = Vec::new();
        for (environment, values) in bodies.iter().enumerate() {
            for (pair, body) in values.iter().enumerate() {
                let (row, owner) = mapping[environment][pair];
                let (orbit, angular) = if let Some(body) = body {
                    let q = body.pose.rotation.quaternion();
                    if !q.norm_squared().is_finite() || (q.norm_squared() - 1.0).abs() > 1e-6 {
                        return Err(GpuArticulatedGroundContactError::InvalidInput);
                    }
                    let origin = body.pose.translation.vector;
                    let v = body.linear_velocity;
                    let w = body.angular_velocity;
                    (
                        PackedSphereOrbit {
                            origin: [
                                finite_f32(origin.x)?,
                                finite_f32(origin.y)?,
                                finite_f32(origin.z)?,
                                1.0,
                            ],
                            linear: [finite_f32(v.x)?, finite_f32(v.y)?, finite_f32(v.z)?, 0.0],
                            orientation: [
                                finite_f32(q.i)?,
                                finite_f32(q.j)?,
                                finite_f32(q.k)?,
                                finite_f32(q.w)?,
                            ],
                            translation_anchor: [0.0; 4],
                        },
                        [finite_f32(w.x)?, finite_f32(w.y)?, finite_f32(w.z)?, 0.0],
                    )
                } else {
                    (PackedSphereOrbit::zeroed(), [0.0; 4])
                };
                for index in row..row + 1 {
                    inputs.push((index, owner, orbit, angular));
                }
            }
        }
        let rows = self.read_prescribed_box_rows(queue)?;
        let uploads = inputs
            .into_iter()
            .map(|(row, owner, orbit, angular)| {
                let center = Vector3::new(
                    rows[row].center_radius[0] as f64,
                    rows[row].center_radius[1] as f64,
                    rows[row].center_radius[2] as f64,
                );
                let origin = Vector3::new(
                    orbit.origin[0] as f64,
                    orbit.origin[1] as f64,
                    orbit.origin[2] as f64,
                );
                let w = Vector3::new(angular[0] as f64, angular[1] as f64, angular[2] as f64);
                let v = Vector3::new(
                    orbit.linear[0] as f64,
                    orbit.linear[1] as f64,
                    orbit.linear[2] as f64,
                ) + w.cross(&(center - origin));
                Ok((
                    row,
                    owner,
                    orbit,
                    angular,
                    [finite_f32(v.x)?, finite_f32(v.y)?, finite_f32(v.z)?, 0.0],
                ))
            })
            .collect::<Result<Vec<_>, GpuArticulatedGroundContactError>>()?;
        for (row, owner, orbit, angular, linear) in uploads {
            let start = row as u64 * size_of::<PackedSphere>() as u64;
            queue.write_buffer(
                &self.sphere_orbits,
                row as u64 * size_of::<PackedSphereOrbit>() as u64,
                bytemuck::bytes_of(&orbit),
            );
            queue.write_buffer(
                &self.spheres,
                start + offset_of!(PackedSphere, prescribed_linear) as u64,
                bytemuck::cast_slice(&linear),
            );
            queue.write_buffer(
                &self.spheres,
                start + offset_of!(PackedSphere, prescribed_angular) as u64,
                bytemuck::cast_slice(&angular),
            );
            queue.write_buffer(
                &self.spheres,
                start + offset_of!(PackedSphere, impulses) as u64,
                bytemuck::cast_slice(&[0.0f32; 4]),
            );
            self.sphere_orbit_enabled[row] = orbit.origin[3] != 0.0;
            if let Some(activity) = &self.contact_activity {
                queue.write_buffer(
                    &activity.wake_requests,
                    owner as u64 * 4,
                    &1u32.to_ne_bytes(),
                );
            }
        }
        Ok(())
    }

    /// Configure world-convex/sphere then world-convex/capsule pairs in reserved order.
    /// Validate every input before uploads; None stops at the current collision pose.
    pub fn update_prescribed_convex_rounded_bodies(
        &mut self,
        queue: &wgpu::Queue,
        bodies: &[Vec<Option<crate::gpu_kinematic_body::GpuKinematicBody>>],
    ) -> Result<(), GpuArticulatedGroundContactError> {
        let mapping = &self.scene_convex_rounded_rows;
        if bodies.len() != mapping.len()
            || bodies.iter().zip(mapping).any(|(b, r)| b.len() != r.len())
        {
            return Err(GpuArticulatedGroundContactError::InvalidInput);
        }
        let mut inputs = Vec::new();
        for (environment, values) in bodies.iter().enumerate() {
            for (pair, body) in values.iter().enumerate() {
                let (row, owner, count) = mapping[environment][pair];
                let (orbit, angular) = if let Some(body) = body {
                    let q = body.pose.rotation.quaternion();
                    if !q.norm_squared().is_finite() || (q.norm_squared() - 1.0).abs() > 1e-6 {
                        return Err(GpuArticulatedGroundContactError::InvalidInput);
                    }
                    let origin = body.pose.translation.vector;
                    let v = body.linear_velocity;
                    let w = body.angular_velocity;
                    (
                        PackedSphereOrbit {
                            origin: [
                                finite_f32(origin.x)?,
                                finite_f32(origin.y)?,
                                finite_f32(origin.z)?,
                                1.0,
                            ],
                            linear: [finite_f32(v.x)?, finite_f32(v.y)?, finite_f32(v.z)?, 0.0],
                            orientation: [
                                finite_f32(q.i)?,
                                finite_f32(q.j)?,
                                finite_f32(q.k)?,
                                finite_f32(q.w)?,
                            ],
                            translation_anchor: [0.0; 4],
                        },
                        [finite_f32(w.x)?, finite_f32(w.y)?, finite_f32(w.z)?, 0.0],
                    )
                } else {
                    (PackedSphereOrbit::zeroed(), [0.0; 4])
                };
                for index in row..row + count {
                    inputs.push((index, owner, orbit, angular));
                }
            }
        }
        let rows = self.read_prescribed_box_rows(queue)?;
        let uploads = inputs
            .into_iter()
            .map(|(row, owner, orbit, angular)| {
                let center = Vector3::new(
                    rows[row].center_radius[0] as f64,
                    rows[row].center_radius[1] as f64,
                    rows[row].center_radius[2] as f64,
                );
                let origin = Vector3::new(
                    orbit.origin[0] as f64,
                    orbit.origin[1] as f64,
                    orbit.origin[2] as f64,
                );
                let w = Vector3::new(angular[0] as f64, angular[1] as f64, angular[2] as f64);
                let v = Vector3::new(
                    orbit.linear[0] as f64,
                    orbit.linear[1] as f64,
                    orbit.linear[2] as f64,
                ) + w.cross(&(center - origin));
                Ok((
                    row,
                    owner,
                    orbit,
                    angular,
                    [finite_f32(v.x)?, finite_f32(v.y)?, finite_f32(v.z)?, 0.0],
                ))
            })
            .collect::<Result<Vec<_>, GpuArticulatedGroundContactError>>()?;
        for (row, owner, orbit, angular, linear) in uploads {
            let start = row as u64 * size_of::<PackedSphere>() as u64;
            queue.write_buffer(
                &self.sphere_orbits,
                row as u64 * size_of::<PackedSphereOrbit>() as u64,
                bytemuck::bytes_of(&orbit),
            );
            queue.write_buffer(
                &self.spheres,
                start + offset_of!(PackedSphere, prescribed_linear) as u64,
                bytemuck::cast_slice(&linear),
            );
            queue.write_buffer(
                &self.spheres,
                start + offset_of!(PackedSphere, prescribed_angular) as u64,
                bytemuck::cast_slice(&angular),
            );
            queue.write_buffer(
                &self.spheres,
                start + offset_of!(PackedSphere, impulses) as u64,
                bytemuck::cast_slice(&[0.0f32; 4]),
            );
            self.sphere_orbit_enabled[row] = orbit.origin[3] != 0.0;
            if let Some(activity) = &self.contact_activity {
                queue.write_buffer(
                    &activity.wake_requests,
                    owner as u64 * 4,
                    &1u32.to_ne_bytes(),
                );
            }
        }
        Ok(())
    }

    fn prescribed_convex_other_rows(&self) -> Vec<Vec<(usize, usize, usize)>> {
        self.static_convex_pair_rows
            .iter()
            .zip(&self.static_axial_convex_rows)
            .map(|(polyhedra, axial)| {
                polyhedra
                    .iter()
                    .map(|&(row, owner)| (row, owner, 4))
                    .chain(axial.iter().map(|&(row, owner)| (row, owner, 1)))
                    .collect()
            })
            .collect()
    }

    /// Set prescribed bodies for external point constraints followed by fixed constraints.
    /// Only constraints without a second articulated link participate, in input order.
    /// The body pose supplies the orbit origin; None stops at the current anchor pose.
    pub fn update_prescribed_constraint_bodies(
        &mut self,
        queue: &wgpu::Queue,
        bodies: &[Vec<Option<crate::gpu_kinematic_body::GpuKinematicBody>>],
    ) -> Result<(), GpuArticulatedGroundContactError> {
        let owned_mapping = self.static_constraint_rows.clone();
        let mapping = &owned_mapping;
        if bodies.len() != mapping.len()
            || bodies.iter().zip(mapping).any(|(b, r)| b.len() != r.len())
        {
            return Err(GpuArticulatedGroundContactError::InvalidInput);
        }
        let mut inputs = Vec::new();
        for (environment, values) in bodies.iter().enumerate() {
            for (pair, body) in values.iter().enumerate() {
                let (row, owner, count) = mapping[environment][pair];
                let (orbit, angular) = if let Some(body) = body {
                    let q = body.pose.rotation.quaternion();
                    if !q.norm_squared().is_finite() || (q.norm_squared() - 1.0).abs() > 1e-6 {
                        return Err(GpuArticulatedGroundContactError::InvalidInput);
                    }
                    let origin = body.pose.translation.vector;
                    let v = body.linear_velocity;
                    let w = body.angular_velocity;
                    (
                        PackedSphereOrbit {
                            origin: [
                                finite_f32(origin.x)?,
                                finite_f32(origin.y)?,
                                finite_f32(origin.z)?,
                                1.0,
                            ],
                            linear: [finite_f32(v.x)?, finite_f32(v.y)?, finite_f32(v.z)?, 0.0],
                            orientation: [
                                finite_f32(q.i)?,
                                finite_f32(q.j)?,
                                finite_f32(q.k)?,
                                finite_f32(q.w)?,
                            ],
                            translation_anchor: [0.0; 4],
                        },
                        [finite_f32(w.x)?, finite_f32(w.y)?, finite_f32(w.z)?, 0.0],
                    )
                } else {
                    (PackedSphereOrbit::zeroed(), [0.0; 4])
                };
                for index in row..row + count {
                    inputs.push((index, owner, orbit, angular));
                }
            }
        }
        let rows = self.read_prescribed_box_rows(queue)?;
        let uploads = inputs
            .into_iter()
            .map(|(row, owner, orbit, angular)| {
                let center = Vector3::new(
                    rows[row].other_center_radius[0] as f64,
                    rows[row].other_center_radius[1] as f64,
                    rows[row].other_center_radius[2] as f64,
                );
                let origin = Vector3::new(
                    orbit.origin[0] as f64,
                    orbit.origin[1] as f64,
                    orbit.origin[2] as f64,
                );
                let w = Vector3::new(angular[0] as f64, angular[1] as f64, angular[2] as f64);
                let v = Vector3::new(
                    orbit.linear[0] as f64,
                    orbit.linear[1] as f64,
                    orbit.linear[2] as f64,
                ) + w.cross(&(center - origin));
                Ok((
                    row,
                    owner,
                    orbit,
                    angular,
                    [finite_f32(v.x)?, finite_f32(v.y)?, finite_f32(v.z)?, 0.0],
                ))
            })
            .collect::<Result<Vec<_>, GpuArticulatedGroundContactError>>()?;
        for (row, owner, orbit, angular, linear) in uploads {
            let start = row as u64 * size_of::<PackedSphere>() as u64;
            queue.write_buffer(
                &self.sphere_orbits,
                row as u64 * size_of::<PackedSphereOrbit>() as u64,
                bytemuck::bytes_of(&orbit),
            );
            queue.write_buffer(
                &self.spheres,
                start + offset_of!(PackedSphere, prescribed_linear) as u64,
                bytemuck::cast_slice(&linear),
            );
            queue.write_buffer(
                &self.spheres,
                start + offset_of!(PackedSphere, prescribed_angular) as u64,
                bytemuck::cast_slice(&angular),
            );
            queue.write_buffer(
                &self.spheres,
                start + offset_of!(PackedSphere, impulses) as u64,
                bytemuck::cast_slice(&[0.0f32; 4]),
            );
            self.sphere_orbit_enabled[row] = orbit.origin[3] != 0.0;
            if let Some(activity) = &self.contact_activity {
                queue.write_buffer(
                    &activity.wake_requests,
                    owner as u64 * 4,
                    &1u32.to_ne_bytes(),
                );
            }
        }
        Ok(())
    }

    /// Configure reserved mesh/polyline pairs in packed contact-family order.
    /// All four manifold rows share the body origin and motion. None stops the
    /// current geometry pose; validate all inputs and source faults before uploads.
    /// Pair order is mesh/sphere, polyline/sphere, mesh/capsule,
    /// polyline/capsule, mesh/box, polyline/box, mesh/axial, polyline/axial,
    /// mesh/convex, polyline/convex; preserve input order within each family.
    pub fn update_prescribed_indexed_bodies(
        &mut self,
        queue: &wgpu::Queue,
        bodies: &[Vec<Option<crate::gpu_kinematic_body::GpuKinematicBody>>],
    ) -> Result<(), GpuArticulatedGroundContactError> {
        let owned_mapping = self.prescribed_indexed_rows.clone();
        let mapping = &owned_mapping;
        if bodies.len() != mapping.len()
            || bodies.iter().zip(mapping).any(|(b, r)| b.len() != r.len())
        {
            return Err(GpuArticulatedGroundContactError::InvalidInput);
        }
        let mut inputs = Vec::new();
        for (environment, values) in bodies.iter().enumerate() {
            for (pair, body) in values.iter().enumerate() {
                let (row, owner, count) = mapping[environment][pair];
                let (orbit, angular) = if let Some(body) = body {
                    let q = body.pose.rotation.quaternion();
                    if !q.norm_squared().is_finite() || (q.norm_squared() - 1.0).abs() > 1e-6 {
                        return Err(GpuArticulatedGroundContactError::InvalidInput);
                    }
                    let origin = body.pose.translation.vector;
                    let v = body.linear_velocity;
                    let w = body.angular_velocity;
                    (
                        PackedSphereOrbit {
                            origin: [
                                finite_f32(origin.x)?,
                                finite_f32(origin.y)?,
                                finite_f32(origin.z)?,
                                1.0,
                            ],
                            linear: [finite_f32(v.x)?, finite_f32(v.y)?, finite_f32(v.z)?, 0.0],
                            orientation: [
                                finite_f32(q.i)?,
                                finite_f32(q.j)?,
                                finite_f32(q.k)?,
                                finite_f32(q.w)?,
                            ],
                            translation_anchor: [0.0; 4],
                        },
                        [finite_f32(w.x)?, finite_f32(w.y)?, finite_f32(w.z)?, 0.0],
                    )
                } else {
                    (PackedSphereOrbit::zeroed(), [0.0; 4])
                };
                for index in row..row + count {
                    inputs.push((index, owner, orbit, angular));
                }
            }
        }
        let rows = self.read_prescribed_box_rows(queue)?;
        let uploads = inputs
            .into_iter()
            .map(|(row, owner, orbit, angular)| {
                let packed = &rows[row];
                let point = if matches!(packed.material[3], 53.0..=56.0 | 73.0..=76.0) {
                    packed.second_axis_end
                } else {
                    packed.plane
                };
                let center = Vector3::new(point[0] as f64, point[1] as f64, point[2] as f64);
                let origin = Vector3::new(
                    orbit.origin[0] as f64,
                    orbit.origin[1] as f64,
                    orbit.origin[2] as f64,
                );
                let w = Vector3::new(angular[0] as f64, angular[1] as f64, angular[2] as f64);
                let v = Vector3::new(
                    orbit.linear[0] as f64,
                    orbit.linear[1] as f64,
                    orbit.linear[2] as f64,
                ) + w.cross(&(center - origin));
                Ok((
                    row,
                    owner,
                    orbit,
                    angular,
                    [finite_f32(v.x)?, finite_f32(v.y)?, finite_f32(v.z)?, 0.0],
                ))
            })
            .collect::<Result<Vec<_>, GpuArticulatedGroundContactError>>()?;
        for (row, owner, orbit, angular, linear) in uploads {
            let start = row as u64 * size_of::<PackedSphere>() as u64;
            queue.write_buffer(
                &self.sphere_orbits,
                row as u64 * size_of::<PackedSphereOrbit>() as u64,
                bytemuck::bytes_of(&orbit),
            );
            queue.write_buffer(
                &self.spheres,
                start + offset_of!(PackedSphere, prescribed_linear) as u64,
                bytemuck::cast_slice(&linear),
            );
            queue.write_buffer(
                &self.spheres,
                start + offset_of!(PackedSphere, prescribed_angular) as u64,
                bytemuck::cast_slice(&angular),
            );
            queue.write_buffer(
                &self.spheres,
                start + offset_of!(PackedSphere, impulses) as u64,
                bytemuck::cast_slice(&[0.0f32; 4]),
            );
            self.sphere_orbit_enabled[row] = orbit.origin[3] != 0.0;
            if let Some(activity) = &self.contact_activity {
                queue.write_buffer(
                    &activity.wake_requests,
                    owner as u64 * 4,
                    &1u32.to_ne_bytes(),
                );
            }
        }
        Ok(())
    }

    /// Configure world-convex polyhedron pairs followed by axial pairs in reserved order.
    /// Validate every input before uploads; None stops at the current collision pose.
    pub fn update_prescribed_convex_other_bodies(
        &mut self,
        queue: &wgpu::Queue,
        bodies: &[Vec<Option<crate::gpu_kinematic_body::GpuKinematicBody>>],
    ) -> Result<(), GpuArticulatedGroundContactError> {
        let owned_mapping = self.prescribed_convex_other_rows();
        let mapping = &owned_mapping;
        if bodies.len() != mapping.len()
            || bodies.iter().zip(mapping).any(|(b, r)| b.len() != r.len())
        {
            return Err(GpuArticulatedGroundContactError::InvalidInput);
        }
        let mut inputs = Vec::new();
        for (environment, values) in bodies.iter().enumerate() {
            for (pair, body) in values.iter().enumerate() {
                let (row, owner, count) = mapping[environment][pair];
                let (orbit, angular) = if let Some(body) = body {
                    let q = body.pose.rotation.quaternion();
                    if !q.norm_squared().is_finite() || (q.norm_squared() - 1.0).abs() > 1e-6 {
                        return Err(GpuArticulatedGroundContactError::InvalidInput);
                    }
                    let origin = body.pose.translation.vector;
                    let v = body.linear_velocity;
                    let w = body.angular_velocity;
                    (
                        PackedSphereOrbit {
                            origin: [
                                finite_f32(origin.x)?,
                                finite_f32(origin.y)?,
                                finite_f32(origin.z)?,
                                1.0,
                            ],
                            linear: [finite_f32(v.x)?, finite_f32(v.y)?, finite_f32(v.z)?, 0.0],
                            orientation: [
                                finite_f32(q.i)?,
                                finite_f32(q.j)?,
                                finite_f32(q.k)?,
                                finite_f32(q.w)?,
                            ],
                            translation_anchor: [0.0; 4],
                        },
                        [finite_f32(w.x)?, finite_f32(w.y)?, finite_f32(w.z)?, 0.0],
                    )
                } else {
                    (PackedSphereOrbit::zeroed(), [0.0; 4])
                };
                for index in row..row + count {
                    inputs.push((index, owner, orbit, angular));
                }
            }
        }
        let rows = self.read_prescribed_box_rows(queue)?;
        let uploads = inputs
            .into_iter()
            .map(|(row, owner, orbit, angular)| {
                let center = Vector3::new(
                    rows[row].plane[0] as f64,
                    rows[row].plane[1] as f64,
                    rows[row].plane[2] as f64,
                );
                let origin = Vector3::new(
                    orbit.origin[0] as f64,
                    orbit.origin[1] as f64,
                    orbit.origin[2] as f64,
                );
                let w = Vector3::new(angular[0] as f64, angular[1] as f64, angular[2] as f64);
                let v = Vector3::new(
                    orbit.linear[0] as f64,
                    orbit.linear[1] as f64,
                    orbit.linear[2] as f64,
                ) + w.cross(&(center - origin));
                Ok((
                    row,
                    owner,
                    orbit,
                    angular,
                    [finite_f32(v.x)?, finite_f32(v.y)?, finite_f32(v.z)?, 0.0],
                ))
            })
            .collect::<Result<Vec<_>, GpuArticulatedGroundContactError>>()?;
        for (row, owner, orbit, angular, linear) in uploads {
            let start = row as u64 * size_of::<PackedSphere>() as u64;
            queue.write_buffer(
                &self.sphere_orbits,
                row as u64 * size_of::<PackedSphereOrbit>() as u64,
                bytemuck::bytes_of(&orbit),
            );
            queue.write_buffer(
                &self.spheres,
                start + offset_of!(PackedSphere, prescribed_linear) as u64,
                bytemuck::cast_slice(&linear),
            );
            queue.write_buffer(
                &self.spheres,
                start + offset_of!(PackedSphere, prescribed_angular) as u64,
                bytemuck::cast_slice(&angular),
            );
            queue.write_buffer(
                &self.spheres,
                start + offset_of!(PackedSphere, impulses) as u64,
                bytemuck::cast_slice(&[0.0f32; 4]),
            );
            self.sphere_orbit_enabled[row] = orbit.origin[3] != 0.0;
            if let Some(activity) = &self.contact_activity {
                queue.write_buffer(
                    &activity.wake_requests,
                    owner as u64 * 4,
                    &1u32.to_ne_bytes(),
                );
            }
        }
        Ok(())
    }

    /// Configure prescribed world capsules for one reserved contact family.
    /// Each input addresses a pair, including all of its manifold rows. The
    /// current collision pose is retained. None stops at that pose. Validate all
    /// inputs before uploads; serialize commands on the owning queue externally.
    pub fn update_prescribed_capsule_bodies(
        &mut self,
        queue: &wgpu::Queue,
        kind: GpuArticulatedCapsuleContactKind,
        bodies: &[Vec<Option<crate::gpu_kinematic_body::GpuKinematicBody>>],
    ) -> Result<(), GpuArticulatedGroundContactError> {
        let mapping = &self.prescribed_capsule_rows[kind.index()];
        if bodies.len() != mapping.len()
            || bodies.iter().zip(mapping).any(|(b, r)| b.len() != r.len())
        {
            return Err(GpuArticulatedGroundContactError::InvalidInput);
        }
        let mut inputs = Vec::new();
        for (environment, values) in bodies.iter().enumerate() {
            for (pair, body) in values.iter().enumerate() {
                let (row, owner) = mapping[environment][pair];
                let (orbit, angular) = if let Some(body) = body {
                    let q = body.pose.rotation.quaternion();
                    if !q.norm_squared().is_finite() || (q.norm_squared() - 1.0).abs() > 1e-6 {
                        return Err(GpuArticulatedGroundContactError::InvalidInput);
                    }
                    let origin = body.pose.translation.vector;
                    let v = body.linear_velocity;
                    let w = body.angular_velocity;
                    (
                        PackedSphereOrbit {
                            origin: [
                                finite_f32(origin.x)?,
                                finite_f32(origin.y)?,
                                finite_f32(origin.z)?,
                                1.0,
                            ],
                            linear: [finite_f32(v.x)?, finite_f32(v.y)?, finite_f32(v.z)?, 0.0],
                            orientation: [
                                finite_f32(q.i)?,
                                finite_f32(q.j)?,
                                finite_f32(q.k)?,
                                finite_f32(q.w)?,
                            ],
                            translation_anchor: [0.0; 4],
                        },
                        [finite_f32(w.x)?, finite_f32(w.y)?, finite_f32(w.z)?, 0.0],
                    )
                } else {
                    (PackedSphereOrbit::zeroed(), [0.0; 4])
                };
                for index in row..row + kind.row_count() {
                    inputs.push((index, owner, orbit, angular));
                }
            }
        }
        let rows = self.read_prescribed_box_rows(queue)?;
        let uploads = inputs
            .into_iter()
            .map(|(row, owner, orbit, angular)| {
                let [a, b] = prescribed_capsule_endpoints(&rows[row]);
                let center = (a + b) * 0.5;
                let origin = Vector3::new(
                    orbit.origin[0] as f64,
                    orbit.origin[1] as f64,
                    orbit.origin[2] as f64,
                );
                let w = Vector3::new(angular[0] as f64, angular[1] as f64, angular[2] as f64);
                let v = Vector3::new(
                    orbit.linear[0] as f64,
                    orbit.linear[1] as f64,
                    orbit.linear[2] as f64,
                ) + w.cross(&(center - origin));
                Ok((
                    row,
                    owner,
                    orbit,
                    angular,
                    [finite_f32(v.x)?, finite_f32(v.y)?, finite_f32(v.z)?, 0.0],
                ))
            })
            .collect::<Result<Vec<_>, GpuArticulatedGroundContactError>>()?;
        for (row, owner, orbit, angular, linear) in uploads {
            let start = row as u64 * size_of::<PackedSphere>() as u64;
            queue.write_buffer(
                &self.sphere_orbits,
                row as u64 * size_of::<PackedSphereOrbit>() as u64,
                bytemuck::bytes_of(&orbit),
            );
            queue.write_buffer(
                &self.spheres,
                start + offset_of!(PackedSphere, prescribed_linear) as u64,
                bytemuck::cast_slice(&linear),
            );
            queue.write_buffer(
                &self.spheres,
                start + offset_of!(PackedSphere, prescribed_angular) as u64,
                bytemuck::cast_slice(&angular),
            );
            queue.write_buffer(
                &self.spheres,
                start + offset_of!(PackedSphere, impulses) as u64,
                bytemuck::cast_slice(&[0.0f32; 4]),
            );
            self.sphere_orbit_enabled[row] = orbit.origin[3] != 0.0;
            if let Some(activity) = &self.contact_activity {
                queue.write_buffer(
                    &activity.wake_requests,
                    owner as u64 * 4,
                    &1u32.to_ne_bytes(),
                );
            }
        }
        Ok(())
    }

    /// Download current capsule endpoints in reserved pair order.
    /// Rejects faulted sources, nonfinite geometry and inconsistent duplicate rows.
    pub fn readback_prescribed_capsule_endpoints(
        &self,
        queue: &wgpu::Queue,
        kind: GpuArticulatedCapsuleContactKind,
    ) -> Result<Vec<Vec<[Vector3<f64>; 2]>>, GpuArticulatedGroundContactError> {
        let rows = self.read_prescribed_box_rows(queue)?;
        self.prescribed_capsule_rows[kind.index()]
            .iter()
            .map(|pairs| {
                pairs
                    .iter()
                    .map(|&(row, _)| {
                        let endpoints = prescribed_capsule_endpoints(&rows[row]);
                        if endpoints
                            .iter()
                            .flat_map(|v| v.iter())
                            .any(|v| !v.is_finite())
                        {
                            return Err(GpuArticulatedGroundContactError::InvalidInput);
                        }
                        for duplicate in &rows[row + 1..row + kind.row_count()] {
                            let other = prescribed_capsule_endpoints(duplicate);
                            if other.iter().zip(&endpoints).any(|(a, b)| {
                                !a.iter().all(|v| v.is_finite()) || (a - b).norm() > 1e-5
                            }) {
                                return Err(GpuArticulatedGroundContactError::InvalidInput);
                            }
                        }
                        Ok(endpoints)
                    })
                    .collect()
            })
            .collect()
    }

    fn read_prescribed_box_rows(
        &self,
        queue: &wgpu::Queue,
    ) -> Result<Vec<PackedSphere>, GpuArticulatedGroundContactError> {
        let read = |buffer| {
            crate::gpu_articulated_mass::read_buffer(&self.device, queue, buffer)
                .map_err(|e| GpuArticulatedGroundContactError::Readback(e.to_string()))
        };
        let status = read(&self.state_status)?;
        for (environment, value) in status.chunks_exact(4).enumerate() {
            if value != [0, 0, 0, 0] {
                return Err(GpuArticulatedGroundContactError::SourceFault(environment));
            }
        }
        let bytes = read(&self.spheres)?;
        Ok(bytes
            .chunks_exact(size_of::<PackedSphere>())
            .map(bytemuck::pod_read_unaligned)
            .collect())
    }

    /// Download current world-box poses in reserved link-sphere/box pair order.
    /// Includes GPU-integrated transforms and rejects faulted source environments.
    pub fn readback_static_sphere_box_poses(
        &self,
        queue: &wgpu::Queue,
    ) -> Result<Vec<Vec<Isometry3<f64>>>, GpuArticulatedGroundContactError> {
        self.readback_prescribed_box_poses(queue, GpuArticulatedBoxContactKind::Sphere)
    }

    /// Download world-box transforms in the selected reserved pair order.
    /// Rejects faulted environments and includes current GPU motion.
    pub fn readback_prescribed_box_poses(
        &self,
        queue: &wgpu::Queue,
        kind: GpuArticulatedBoxContactKind,
    ) -> Result<Vec<Vec<Isometry3<f64>>>, GpuArticulatedGroundContactError> {
        let rows = self.read_prescribed_box_rows(queue)?;
        self.prescribed_box_mapping(kind)
            .iter()
            .map(|environment| {
                environment
                    .iter()
                    .map(|&(index, _)| {
                        let row = &rows[index];
                        for other in &rows[index + 1..index + kind.row_count()] {
                            if other.plane[..3]
                                .iter()
                                .chain(&other.second_axis_end)
                                .any(|v| !v.is_finite())
                                || row.plane[..3]
                                    .iter()
                                    .zip(&other.plane[..3])
                                    .chain(row.second_axis_end.iter().zip(&other.second_axis_end))
                                    .any(|(a, b)| (a - b).abs() > 1e-5)
                            {
                                return Err(GpuArticulatedGroundContactError::InvalidInput);
                            }
                        }
                        let q = Quaternion::new(
                            row.second_axis_end[3] as f64,
                            row.second_axis_end[0] as f64,
                            row.second_axis_end[1] as f64,
                            row.second_axis_end[2] as f64,
                        );
                        if row.plane[..3]
                            .iter()
                            .chain(&row.second_axis_end)
                            .any(|v| !v.is_finite())
                            || (q.norm_squared() - 1.0).abs() > 1e-3
                        {
                            return Err(GpuArticulatedGroundContactError::InvalidInput);
                        }
                        Ok(Isometry3::from_parts(
                            nalgebra::Translation3::new(
                                row.plane[0] as f64,
                                row.plane[1] as f64,
                                row.plane[2] as f64,
                            ),
                            UnitQuaternion::new_normalize(q),
                        ))
                    })
                    .collect()
            })
            .collect()
    }

    /// Read current GPU world-cylinder and world-cone collision poses.
    /// Reject faulted environments, nonfinite poses, and invalid orientations.
    pub fn readback_prescribed_axial_poses(
        &self,
        queue: &wgpu::Queue,
        kind: GpuArticulatedAxialContactKind,
    ) -> Result<Vec<Vec<Isometry3<f64>>>, GpuArticulatedGroundContactError> {
        let rows = self.read_prescribed_box_rows(queue)?;
        self.prescribed_axial_rows[kind.index()]
            .iter()
            .map(|environment| {
                environment
                    .iter()
                    .map(|&(index, _)| {
                        let row = &rows[index];
                        let rotation = if row.material[3] < 66.0 {
                            row.second_axis_end
                        } else {
                            row.first_axis_end
                        };
                        let q = Quaternion::new(
                            rotation[3] as f64,
                            rotation[0] as f64,
                            rotation[1] as f64,
                            rotation[2] as f64,
                        );
                        if row.center_radius[..3]
                            .iter()
                            .chain(&rotation)
                            .any(|v| !v.is_finite())
                            || (q.norm_squared() - 1.0).abs() > 1e-3
                        {
                            return Err(GpuArticulatedGroundContactError::InvalidInput);
                        }
                        Ok(Isometry3::from_parts(
                            nalgebra::Translation3::new(
                                row.center_radius[0] as f64,
                                row.center_radius[1] as f64,
                                row.center_radius[2] as f64,
                            ),
                            UnitQuaternion::new_normalize(q),
                        ))
                    })
                    .collect()
            })
            .collect()
    }

    /// Read current GPU world-convex rounded-contact collision poses.
    /// Reject faulted environments, nonfinite poses, and invalid orientations.
    pub fn readback_prescribed_convex_rounded_poses(
        &self,
        queue: &wgpu::Queue,
    ) -> Result<Vec<Vec<Isometry3<f64>>>, GpuArticulatedGroundContactError> {
        let rows = self.read_prescribed_box_rows(queue)?;
        self.scene_convex_rounded_rows
            .iter()
            .map(|environment| {
                environment
                    .iter()
                    .map(|&(index, _, _)| {
                        let row = &rows[index];
                        let rotation = row.first_axis_end;
                        let q = Quaternion::new(
                            rotation[3] as f64,
                            rotation[0] as f64,
                            rotation[1] as f64,
                            rotation[2] as f64,
                        );
                        if row.center_radius[..3]
                            .iter()
                            .chain(&rotation)
                            .any(|v| !v.is_finite())
                            || (q.norm_squared() - 1.0).abs() > 1e-3
                        {
                            return Err(GpuArticulatedGroundContactError::InvalidInput);
                        }
                        Ok(Isometry3::from_parts(
                            nalgebra::Translation3::new(
                                row.center_radius[0] as f64,
                                row.center_radius[1] as f64,
                                row.center_radius[2] as f64,
                            ),
                            UnitQuaternion::new_normalize(q),
                        ))
                    })
                    .collect()
            })
            .collect()
    }

    /// Read current GPU world-convex rounded-contact collision poses.
    /// Reject faulted environments, nonfinite poses, and invalid orientations.
    pub fn readback_prescribed_convex_other_poses(
        &self,
        queue: &wgpu::Queue,
    ) -> Result<Vec<Vec<Isometry3<f64>>>, GpuArticulatedGroundContactError> {
        let rows = self.read_prescribed_box_rows(queue)?;
        self.prescribed_convex_other_rows()
            .iter()
            .map(|environment| {
                environment
                    .iter()
                    .map(|&(index, _, _)| {
                        let row = &rows[index];
                        let rotation = row.second_axis_end;
                        let q = Quaternion::new(
                            rotation[3] as f64,
                            rotation[0] as f64,
                            rotation[1] as f64,
                            rotation[2] as f64,
                        );
                        if row.plane[..3]
                            .iter()
                            .chain(&rotation)
                            .any(|v| !v.is_finite())
                            || (q.norm_squared() - 1.0).abs() > 1e-3
                        {
                            return Err(GpuArticulatedGroundContactError::InvalidInput);
                        }
                        Ok(Isometry3::from_parts(
                            nalgebra::Translation3::new(
                                row.plane[0] as f64,
                                row.plane[1] as f64,
                                row.plane[2] as f64,
                            ),
                            UnitQuaternion::new_normalize(q),
                        ))
                    })
                    .collect()
            })
            .collect()
    }

    /// Replace external anchor frames without changing prescribed motion or topology.
    /// Validate all frames and source status before uploading any row.
    pub fn update_external_constraint_frames(
        &mut self,
        queue: &wgpu::Queue,
        frames: &[Vec<Isometry3<f64>>],
    ) -> Result<(), GpuArticulatedGroundContactError> {
        if frames.len() != self.static_constraint_rows.len()
            || frames
                .iter()
                .zip(&self.static_constraint_rows)
                .any(|(f, r)| f.len() != r.len())
        {
            return Err(GpuArticulatedGroundContactError::InvalidInput);
        }
        let mut updates = Vec::new();
        for (values, mapping) in frames.iter().zip(&self.static_constraint_rows) {
            for (frame, &(row, owner, count)) in values.iter().zip(mapping) {
                let q = frame.rotation.quaternion();
                if !q.norm_squared().is_finite() || (q.norm_squared() - 1.0).abs() > 1e-6 {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                let p = frame.translation.vector;
                let point = [finite_f32(p.x)?, finite_f32(p.y)?, finite_f32(p.z)?, 0.0];
                let rotation = [
                    finite_f32(q.i)?,
                    finite_f32(q.j)?,
                    finite_f32(q.k)?,
                    finite_f32(q.w)?,
                ];
                updates.push((row, owner, count, point, rotation));
            }
        }
        let _ = self.read_prescribed_box_rows(queue)?;
        for (row, owner, count, point, rotation) in updates {
            for index in row..row + count {
                let start = index as u64 * size_of::<PackedSphere>() as u64;
                queue.write_buffer(
                    &self.spheres,
                    start + offset_of!(PackedSphere, other_center_radius) as u64,
                    bytemuck::cast_slice(&point),
                );
                queue.write_buffer(
                    &self.spheres,
                    start + offset_of!(PackedSphere, second_axis_end) as u64,
                    bytemuck::cast_slice(&rotation),
                );
                queue.write_buffer(
                    &self.spheres,
                    start + offset_of!(PackedSphere, impulses) as u64,
                    bytemuck::cast_slice(&[0.0f32; 4]),
                );
            }
            if let Some(activity) = &self.contact_activity {
                queue.write_buffer(
                    &activity.wake_requests,
                    owner as u64 * 4,
                    &1u32.to_ne_bytes(),
                );
            }
        }
        Ok(())
    }

    /// Replace indexed geometry poses while retaining topology and prescribed motion.
    /// Recompute the origin velocity for its new offset from the same body orbit.
    /// Validate all poses, derived velocities, and source status before any upload.
    pub fn update_indexed_geometry_poses(
        &mut self,
        queue: &wgpu::Queue,
        frames: &[Vec<Isometry3<f64>>],
    ) -> Result<(), GpuArticulatedGroundContactError> {
        if frames.len() != self.prescribed_indexed_rows.len()
            || frames
                .iter()
                .zip(&self.prescribed_indexed_rows)
                .any(|(f, r)| f.len() != r.len())
        {
            return Err(GpuArticulatedGroundContactError::InvalidInput);
        }
        let mut updates = Vec::new();
        for (values, mapping) in frames.iter().zip(&self.prescribed_indexed_rows) {
            for (frame, &(row, owner, count)) in values.iter().zip(mapping) {
                let q = frame.rotation.quaternion();
                if !q.norm_squared().is_finite() || (q.norm_squared() - 1.0).abs() > 1e-6 {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                let p = frame.translation.vector;
                let point = [finite_f32(p.x)?, finite_f32(p.y)?, finite_f32(p.z)?, 0.0];
                let rotation = [
                    finite_f32(q.i)?,
                    finite_f32(q.j)?,
                    finite_f32(q.k)?,
                    finite_f32(q.w)?,
                ];
                updates.push((row, owner, count, point, rotation));
            }
        }
        let rows = self.read_prescribed_box_rows(queue)?;
        let updates = updates
            .into_iter()
            .map(|(row, owner, count, point, rotation)| {
                let old = &rows[row];
                let previous = if matches!(old.material[3], 53.0..=56.0 | 73.0..=76.0) {
                    old.second_axis_end
                } else {
                    old.plane
                };
                let delta = Vector3::new(
                    (point[0] - previous[0]) as f64,
                    (point[1] - previous[1]) as f64,
                    (point[2] - previous[2]) as f64,
                );
                let angular = Vector3::new(
                    old.prescribed_angular[0] as f64,
                    old.prescribed_angular[1] as f64,
                    old.prescribed_angular[2] as f64,
                );
                let linear = Vector3::new(
                    old.prescribed_linear[0] as f64,
                    old.prescribed_linear[1] as f64,
                    old.prescribed_linear[2] as f64,
                ) + angular.cross(&delta);
                let linear = [
                    finite_f32(linear.x)?,
                    finite_f32(linear.y)?,
                    finite_f32(linear.z)?,
                    0.0,
                ];
                Ok((row, owner, count, point, rotation, linear))
            })
            .collect::<Result<Vec<_>, GpuArticulatedGroundContactError>>()?;
        for (row, owner, count, point, rotation, linear) in updates {
            for (index, packed) in rows.iter().enumerate().skip(row).take(count) {
                let start = index as u64 * size_of::<PackedSphere>() as u64;
                let mode = packed.material[3];
                let position_offset = if matches!(mode, 53.0..=56.0 | 73.0..=76.0) {
                    offset_of!(PackedSphere, second_axis_end)
                } else {
                    offset_of!(PackedSphere, plane)
                };
                let rotation_offset = if mode == 57.0 || mode == 77.0 {
                    offset_of!(PackedSphere, second_axis_end)
                } else {
                    offset_of!(PackedSphere, first_axis_end)
                };
                queue.write_buffer(
                    &self.spheres,
                    start + position_offset as u64,
                    bytemuck::cast_slice(&point[..3]),
                );
                queue.write_buffer(
                    &self.spheres,
                    start + rotation_offset as u64,
                    bytemuck::cast_slice(&rotation),
                );
                queue.write_buffer(
                    &self.spheres,
                    start + offset_of!(PackedSphere, prescribed_linear) as u64,
                    bytemuck::cast_slice(&linear),
                );
                queue.write_buffer(
                    &self.spheres,
                    start + offset_of!(PackedSphere, impulses) as u64,
                    bytemuck::cast_slice(&[0.0f32; 4]),
                );
            }
            if let Some(activity) = &self.contact_activity {
                queue.write_buffer(
                    &activity.wake_requests,
                    owner as u64 * 4,
                    &1u32.to_ne_bytes(),
                );
            }
        }
        Ok(())
    }

    /// Read external point-anchor positions and fixed-frame poses in reserved order.
    /// Point anchors use an identity initial orientation; this is not the body origin.
    /// Reject source faults and invalid frame values before returning any result.
    pub fn readback_prescribed_constraint_frames(
        &self,
        queue: &wgpu::Queue,
    ) -> Result<Vec<Vec<Isometry3<f64>>>, GpuArticulatedGroundContactError> {
        let rows = self.read_prescribed_box_rows(queue)?;
        self.static_constraint_rows
            .iter()
            .map(|environment| {
                environment
                    .iter()
                    .map(|&(index, _, _)| {
                        let row = &rows[index];
                        let rotation = row.second_axis_end;
                        let q = Quaternion::new(
                            rotation[3] as f64,
                            rotation[0] as f64,
                            rotation[1] as f64,
                            rotation[2] as f64,
                        );
                        if row.other_center_radius[..3]
                            .iter()
                            .chain(&rotation)
                            .any(|v| !v.is_finite())
                            || (q.norm_squared() - 1.0).abs() > 1e-3
                        {
                            return Err(GpuArticulatedGroundContactError::InvalidInput);
                        }
                        Ok(Isometry3::from_parts(
                            nalgebra::Translation3::new(
                                row.other_center_radius[0] as f64,
                                row.other_center_radius[1] as f64,
                                row.other_center_radius[2] as f64,
                            ),
                            UnitQuaternion::new_normalize(q),
                        ))
                    })
                    .collect()
            })
            .collect()
    }

    /// Read current mesh/polyline poses in the reserved packed pair order.
    /// Reject source faults and nonfinite or invalid orientations.
    pub fn readback_prescribed_indexed_poses(
        &self,
        queue: &wgpu::Queue,
    ) -> Result<Vec<Vec<Isometry3<f64>>>, GpuArticulatedGroundContactError> {
        let rows = self.read_prescribed_box_rows(queue)?;
        self.prescribed_indexed_rows
            .iter()
            .map(|environment| {
                environment
                    .iter()
                    .map(|&(index, _, _)| {
                        let row = &rows[index];
                        let rotation = if row.material[3] == 57.0 || row.material[3] == 77.0 {
                            row.second_axis_end
                        } else {
                            row.first_axis_end
                        };
                        let point = if matches!(row.material[3], 53.0..=56.0 | 73.0..=76.0) {
                            row.second_axis_end
                        } else {
                            row.plane
                        };
                        let q = Quaternion::new(
                            rotation[3] as f64,
                            rotation[0] as f64,
                            rotation[1] as f64,
                            rotation[2] as f64,
                        );
                        if point[..3].iter().chain(&rotation).any(|v| !v.is_finite())
                            || (q.norm_squared() - 1.0).abs() > 1e-3
                        {
                            return Err(GpuArticulatedGroundContactError::InvalidInput);
                        }
                        Ok(Isometry3::from_parts(
                            nalgebra::Translation3::new(
                                point[0] as f64,
                                point[1] as f64,
                                point[2] as f64,
                            ),
                            UnitQuaternion::new_normalize(q),
                        ))
                    })
                    .collect()
            })
            .collect()
    }

    /// Update stationary box transforms in reserved sphere/box pair order.
    /// Dimensions and topology stay fixed. Validate all poses before any upload;
    /// changed rows lose warm impulses and add owner wake without clearing other
    /// pending requests or geometry history. Quaternion inputs must be unit length.
    pub fn update_static_sphere_box_poses(
        &mut self,
        queue: &wgpu::Queue,
        poses: &[Vec<Isometry3<f64>>],
    ) -> Result<bool, GpuArticulatedGroundContactError> {
        if poses.len() != self.static_sphere_box_rows.len()
            || poses
                .iter()
                .zip(&self.static_sphere_box_rows)
                .any(|(p, r)| p.len() != r.len())
        {
            return Err(GpuArticulatedGroundContactError::InvalidInput);
        }
        let packed = poses
            .iter()
            .map(|values| {
                values
                    .iter()
                    .map(|pose| {
                        let q = pose.rotation.quaternion();
                        if !q.norm_squared().is_finite() || (q.norm_squared() - 1.0).abs() > 1e-5 {
                            return Err(GpuArticulatedGroundContactError::InvalidInput);
                        }
                        let c = pose.translation.vector;
                        Ok([
                            finite_f32(c.x)?,
                            finite_f32(c.y)?,
                            finite_f32(c.z)?,
                            finite_f32(q.i)?,
                            finite_f32(q.j)?,
                            finite_f32(q.k)?,
                            finite_f32(q.w)?,
                        ])
                    })
                    .collect::<Result<Vec<_>, GpuArticulatedGroundContactError>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let integrated = self
            .integrated_sphere_box_poses
            .swap(false, Ordering::Relaxed);
        let mut changed = false;
        for (env, values) in packed.iter().enumerate() {
            for (pair, pose) in values.iter().enumerate() {
                if !integrated && self.static_sphere_box_poses[env][pair] == *pose {
                    continue;
                }
                let (row, owner) = self.static_sphere_box_rows[env][pair];
                let start = row as u64 * size_of::<PackedSphere>() as u64;
                queue.write_buffer(
                    &self.spheres,
                    start + core::mem::offset_of!(PackedSphere, plane) as u64,
                    bytemuck::cast_slice(&pose[..3]),
                );
                queue.write_buffer(
                    &self.spheres,
                    start + core::mem::offset_of!(PackedSphere, second_axis_end) as u64,
                    bytemuck::cast_slice(&pose[3..]),
                );
                queue.write_buffer(
                    &self.spheres,
                    start + core::mem::offset_of!(PackedSphere, impulses) as u64,
                    bytemuck::cast_slice(&[0.0f32; 4]),
                );
                if let Some(activity) = &self.contact_activity {
                    queue.write_buffer(
                        &activity.wake_requests,
                        owner as u64 * 4,
                        &1u32.to_ne_bytes(),
                    );
                }
                changed = true;
            }
        }
        self.static_sphere_box_poses = packed;
        Ok(changed)
    }

    /// Update stationary box transforms in reserved axial/box pair order.
    /// Input omits pairs with axial_is_static == true.
    /// Dimensions and topology stay fixed. Validate all poses before any upload;
    /// changed rows lose warm impulses and add owner wake without clearing other
    /// pending requests or geometry history. Quaternion inputs must be unit length.
    pub fn update_static_axial_box_poses(
        &mut self,
        queue: &wgpu::Queue,
        poses: &[Vec<Isometry3<f64>>],
    ) -> Result<bool, GpuArticulatedGroundContactError> {
        if poses.len() != self.static_axial_box_rows.len()
            || poses
                .iter()
                .zip(&self.static_axial_box_rows)
                .any(|(p, r)| p.len() != r.len())
        {
            return Err(GpuArticulatedGroundContactError::InvalidInput);
        }
        let packed = poses
            .iter()
            .map(|values| {
                values
                    .iter()
                    .map(|pose| {
                        let q = pose.rotation.quaternion();
                        if !q.norm_squared().is_finite() || (q.norm_squared() - 1.0).abs() > 1e-5 {
                            return Err(GpuArticulatedGroundContactError::InvalidInput);
                        }
                        let c = pose.translation.vector;
                        Ok([
                            finite_f32(c.x)?,
                            finite_f32(c.y)?,
                            finite_f32(c.z)?,
                            finite_f32(q.i)?,
                            finite_f32(q.j)?,
                            finite_f32(q.k)?,
                            finite_f32(q.w)?,
                        ])
                    })
                    .collect::<Result<Vec<_>, GpuArticulatedGroundContactError>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let integrated = self
            .integrated_axial_box_poses
            .swap(false, Ordering::Relaxed);
        let mut changed = false;
        for (env, values) in packed.iter().enumerate() {
            for (pair, pose) in values.iter().enumerate() {
                if !integrated && self.static_axial_box_poses[env][pair] == *pose {
                    continue;
                }
                let (row, owner) = self.static_axial_box_rows[env][pair];
                let start = row as u64 * size_of::<PackedSphere>() as u64;
                queue.write_buffer(
                    &self.spheres,
                    start + core::mem::offset_of!(PackedSphere, plane) as u64,
                    bytemuck::cast_slice(&pose[..3]),
                );
                queue.write_buffer(
                    &self.spheres,
                    start + core::mem::offset_of!(PackedSphere, second_axis_end) as u64,
                    bytemuck::cast_slice(&pose[3..]),
                );
                queue.write_buffer(
                    &self.spheres,
                    start + core::mem::offset_of!(PackedSphere, impulses) as u64,
                    bytemuck::cast_slice(&[0.0f32; 4]),
                );
                if let Some(activity) = &self.contact_activity {
                    queue.write_buffer(
                        &activity.wake_requests,
                        owner as u64 * 4,
                        &1u32.to_ne_bytes(),
                    );
                }
                changed = true;
            }
        }
        self.static_axial_box_poses = packed;
        Ok(changed)
    }

    /// Update stationary box transforms in reserved capsule/box pair order.
    /// Dimensions and topology stay fixed. Validate all poses before any upload;
    /// changed rows lose warm impulses and add owner wake without clearing other
    /// pending requests or geometry history. Quaternion inputs must be unit length.
    pub fn update_static_capsule_box_poses(
        &mut self,
        queue: &wgpu::Queue,
        poses: &[Vec<Isometry3<f64>>],
    ) -> Result<bool, GpuArticulatedGroundContactError> {
        if poses.len() != self.static_capsule_box_rows.len()
            || poses
                .iter()
                .zip(&self.static_capsule_box_rows)
                .any(|(p, r)| p.len() != r.len())
        {
            return Err(GpuArticulatedGroundContactError::InvalidInput);
        }
        let packed = poses
            .iter()
            .map(|values| {
                values
                    .iter()
                    .map(|pose| {
                        let q = pose.rotation.quaternion();
                        if !q.norm_squared().is_finite() || (q.norm_squared() - 1.0).abs() > 1e-5 {
                            return Err(GpuArticulatedGroundContactError::InvalidInput);
                        }
                        let c = pose.translation.vector;
                        Ok([
                            finite_f32(c.x)?,
                            finite_f32(c.y)?,
                            finite_f32(c.z)?,
                            finite_f32(q.i)?,
                            finite_f32(q.j)?,
                            finite_f32(q.k)?,
                            finite_f32(q.w)?,
                        ])
                    })
                    .collect::<Result<Vec<_>, GpuArticulatedGroundContactError>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let integrated = self
            .integrated_capsule_box_poses
            .swap(false, Ordering::Relaxed);
        let mut changed = false;
        for (env, values) in packed.iter().enumerate() {
            for (pair, pose) in values.iter().enumerate() {
                if !integrated && self.static_capsule_box_poses[env][pair] == *pose {
                    continue;
                }
                let (row, owner) = self.static_capsule_box_rows[env][pair];
                // Both endpoint and side manifold rows share this stationary pose.
                for contact_row in row..row + 2 {
                    let start = contact_row as u64 * size_of::<PackedSphere>() as u64;
                    queue.write_buffer(
                        &self.spheres,
                        start + core::mem::offset_of!(PackedSphere, plane) as u64,
                        bytemuck::cast_slice(&pose[..3]),
                    );
                    queue.write_buffer(
                        &self.spheres,
                        start + core::mem::offset_of!(PackedSphere, second_axis_end) as u64,
                        bytemuck::cast_slice(&pose[3..]),
                    );
                    queue.write_buffer(
                        &self.spheres,
                        start + core::mem::offset_of!(PackedSphere, impulses) as u64,
                        bytemuck::cast_slice(&[0.0f32; 4]),
                    );
                }
                if let Some(activity) = &self.contact_activity {
                    queue.write_buffer(
                        &activity.wake_requests,
                        owner as u64 * 4,
                        &1u32.to_ne_bytes(),
                    );
                }
                changed = true;
            }
        }
        self.static_capsule_box_poses = packed;
        Ok(changed)
    }

    /// Update stationary box transforms in reserved box/box pair order.
    /// Dimensions and topology stay fixed. Validate all poses before any upload;
    /// changed rows lose warm impulses and add owner wake without clearing other
    /// pending requests or geometry history. Quaternion inputs must be unit length.
    pub fn update_static_box_box_poses(
        &mut self,
        queue: &wgpu::Queue,
        poses: &[Vec<Isometry3<f64>>],
    ) -> Result<bool, GpuArticulatedGroundContactError> {
        if poses.len() != self.static_box_box_rows.len()
            || poses
                .iter()
                .zip(&self.static_box_box_rows)
                .any(|(p, r)| p.len() != r.len())
        {
            return Err(GpuArticulatedGroundContactError::InvalidInput);
        }
        let packed = poses
            .iter()
            .map(|values| {
                values
                    .iter()
                    .map(|pose| {
                        let q = pose.rotation.quaternion();
                        if !q.norm_squared().is_finite() || (q.norm_squared() - 1.0).abs() > 1e-5 {
                            return Err(GpuArticulatedGroundContactError::InvalidInput);
                        }
                        let c = pose.translation.vector;
                        Ok([
                            finite_f32(c.x)?,
                            finite_f32(c.y)?,
                            finite_f32(c.z)?,
                            finite_f32(q.i)?,
                            finite_f32(q.j)?,
                            finite_f32(q.k)?,
                            finite_f32(q.w)?,
                        ])
                    })
                    .collect::<Result<Vec<_>, GpuArticulatedGroundContactError>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let integrated = self.integrated_box_box_poses.swap(false, Ordering::Relaxed);
        let mut changed = false;
        for (env, values) in packed.iter().enumerate() {
            for (pair, pose) in values.iter().enumerate() {
                if !integrated && self.static_box_box_poses[env][pair] == *pose {
                    continue;
                }
                let (row, owner) = self.static_box_box_rows[env][pair];
                // All four manifold rows share this stationary pose.
                for contact_row in row..row + 4 {
                    let start = contact_row as u64 * size_of::<PackedSphere>() as u64;
                    queue.write_buffer(
                        &self.spheres,
                        start + core::mem::offset_of!(PackedSphere, plane) as u64,
                        bytemuck::cast_slice(&pose[..3]),
                    );
                    queue.write_buffer(
                        &self.spheres,
                        start + core::mem::offset_of!(PackedSphere, second_axis_end) as u64,
                        bytemuck::cast_slice(&pose[3..]),
                    );
                    queue.write_buffer(
                        &self.spheres,
                        start + core::mem::offset_of!(PackedSphere, impulses) as u64,
                        bytemuck::cast_slice(&[0.0f32; 4]),
                    );
                }
                if let Some(activity) = &self.contact_activity {
                    queue.write_buffer(
                        &activity.wake_requests,
                        owner as u64 * 4,
                        &1u32.to_ne_bytes(),
                    );
                }
                changed = true;
            }
        }
        self.static_box_box_poses = packed;
        Ok(changed)
    }

    /// Update stationary convex transforms in reserved static-convex-pair order.
    /// Dimensions and topology stay fixed. Validate all poses before any upload;
    /// changed rows lose warm impulses and add owner wake without clearing other
    /// pending requests or geometry history. Quaternion inputs must be unit length.
    pub fn update_static_convex_pair_poses(
        &mut self,
        queue: &wgpu::Queue,
        poses: &[Vec<Isometry3<f64>>],
    ) -> Result<bool, GpuArticulatedGroundContactError> {
        if poses.len() != self.static_convex_pair_rows.len()
            || poses
                .iter()
                .zip(&self.static_convex_pair_rows)
                .any(|(p, r)| p.len() != r.len())
        {
            return Err(GpuArticulatedGroundContactError::InvalidInput);
        }
        let packed = poses
            .iter()
            .map(|values| {
                values
                    .iter()
                    .map(|pose| {
                        let q = pose.rotation.quaternion();
                        if !q.norm_squared().is_finite() || (q.norm_squared() - 1.0).abs() > 1e-5 {
                            return Err(GpuArticulatedGroundContactError::InvalidInput);
                        }
                        let c = pose.translation.vector;
                        Ok([
                            finite_f32(c.x)?,
                            finite_f32(c.y)?,
                            finite_f32(c.z)?,
                            finite_f32(q.i)?,
                            finite_f32(q.j)?,
                            finite_f32(q.k)?,
                            finite_f32(q.w)?,
                        ])
                    })
                    .collect::<Result<Vec<_>, GpuArticulatedGroundContactError>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let integrated = self
            .integrated_prescribed_convex_poses
            .swap(false, Ordering::Relaxed);
        let mut changed = false;
        for (env, values) in packed.iter().enumerate() {
            for (pair, pose) in values.iter().enumerate() {
                if !integrated && self.static_convex_pair_poses[env][pair] == *pose {
                    continue;
                }
                let (row, owner) = self.static_convex_pair_rows[env][pair];
                // All four manifold rows share this stationary pose.
                for contact_row in row..row + 4 {
                    let start = contact_row as u64 * size_of::<PackedSphere>() as u64;
                    queue.write_buffer(
                        &self.spheres,
                        start + core::mem::offset_of!(PackedSphere, plane) as u64,
                        bytemuck::cast_slice(&pose[..3]),
                    );
                    queue.write_buffer(
                        &self.spheres,
                        start + core::mem::offset_of!(PackedSphere, second_axis_end) as u64,
                        bytemuck::cast_slice(&pose[3..]),
                    );
                    queue.write_buffer(
                        &self.spheres,
                        start + core::mem::offset_of!(PackedSphere, impulses) as u64,
                        bytemuck::cast_slice(&[0.0f32; 4]),
                    );
                }
                if let Some(activity) = &self.contact_activity {
                    queue.write_buffer(
                        &activity.wake_requests,
                        owner as u64 * 4,
                        &1u32.to_ne_bytes(),
                    );
                }
                changed = true;
            }
        }
        self.static_convex_pair_poses = packed;
        Ok(changed)
    }

    /// Update stationary convex transforms in reserved static-axial-convex-pair order.
    /// Dimensions and topology stay fixed. Validate all poses before any upload;
    /// changed rows lose warm impulses and add owner wake without clearing other
    /// pending requests or geometry history. Quaternion inputs must be unit length.
    pub fn update_static_axial_convex_poses(
        &mut self,
        queue: &wgpu::Queue,
        poses: &[Vec<Isometry3<f64>>],
    ) -> Result<bool, GpuArticulatedGroundContactError> {
        if poses.len() != self.static_axial_convex_rows.len()
            || poses
                .iter()
                .zip(&self.static_axial_convex_rows)
                .any(|(p, r)| p.len() != r.len())
        {
            return Err(GpuArticulatedGroundContactError::InvalidInput);
        }
        let packed = poses
            .iter()
            .map(|values| {
                values
                    .iter()
                    .map(|pose| {
                        let q = pose.rotation.quaternion();
                        if !q.norm_squared().is_finite() || (q.norm_squared() - 1.0).abs() > 1e-5 {
                            return Err(GpuArticulatedGroundContactError::InvalidInput);
                        }
                        let c = pose.translation.vector;
                        Ok([
                            finite_f32(c.x)?,
                            finite_f32(c.y)?,
                            finite_f32(c.z)?,
                            finite_f32(q.i)?,
                            finite_f32(q.j)?,
                            finite_f32(q.k)?,
                            finite_f32(q.w)?,
                        ])
                    })
                    .collect::<Result<Vec<_>, GpuArticulatedGroundContactError>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let integrated = self
            .integrated_axial_convex_poses
            .swap(false, Ordering::Relaxed);
        let mut changed = false;
        for (env, values) in packed.iter().enumerate() {
            for (pair, pose) in values.iter().enumerate() {
                if !integrated && self.static_axial_convex_poses[env][pair] == *pose {
                    continue;
                }
                let (row, owner) = self.static_axial_convex_rows[env][pair];
                // This contact row owns the stationary convex pose.
                for contact_row in row..row + 1 {
                    let start = contact_row as u64 * size_of::<PackedSphere>() as u64;
                    queue.write_buffer(
                        &self.spheres,
                        start + core::mem::offset_of!(PackedSphere, plane) as u64,
                        bytemuck::cast_slice(&pose[..3]),
                    );
                    queue.write_buffer(
                        &self.spheres,
                        start + core::mem::offset_of!(PackedSphere, second_axis_end) as u64,
                        bytemuck::cast_slice(&pose[3..]),
                    );
                    queue.write_buffer(
                        &self.spheres,
                        start + core::mem::offset_of!(PackedSphere, impulses) as u64,
                        bytemuck::cast_slice(&[0.0f32; 4]),
                    );
                }
                if let Some(activity) = &self.contact_activity {
                    queue.write_buffer(
                        &activity.wake_requests,
                        owner as u64 * 4,
                        &1u32.to_ne_bytes(),
                    );
                }
                changed = true;
            }
        }
        self.static_axial_convex_poses = packed;
        Ok(changed)
    }

    /// Update stationary convex transforms in reserved scene-convex-sphere then scene-convex-capsule order.
    /// Dimensions and topology stay fixed. Validate all poses before any upload;
    /// changed rows lose warm impulses and add owner wake without clearing other
    /// pending requests or geometry history. Quaternion inputs must be unit length.
    pub fn update_scene_convex_rounded_poses(
        &mut self,
        queue: &wgpu::Queue,
        poses: &[Vec<Isometry3<f64>>],
    ) -> Result<bool, GpuArticulatedGroundContactError> {
        if poses.len() != self.scene_convex_rounded_rows.len()
            || poses
                .iter()
                .zip(&self.scene_convex_rounded_rows)
                .any(|(p, r)| p.len() != r.len())
        {
            return Err(GpuArticulatedGroundContactError::InvalidInput);
        }
        let packed = poses
            .iter()
            .map(|values| {
                values
                    .iter()
                    .map(|pose| {
                        let q = pose.rotation.quaternion();
                        if !q.norm_squared().is_finite() || (q.norm_squared() - 1.0).abs() > 1e-5 {
                            return Err(GpuArticulatedGroundContactError::InvalidInput);
                        }
                        let c = pose.translation.vector;
                        Ok([
                            finite_f32(c.x)?,
                            finite_f32(c.y)?,
                            finite_f32(c.z)?,
                            finite_f32(q.i)?,
                            finite_f32(q.j)?,
                            finite_f32(q.k)?,
                            finite_f32(q.w)?,
                        ])
                    })
                    .collect::<Result<Vec<_>, GpuArticulatedGroundContactError>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let integrated = self
            .integrated_convex_rounded_poses
            .swap(false, Ordering::Relaxed);
        let mut changed = false;
        for (env, values) in packed.iter().enumerate() {
            for (pair, pose) in values.iter().enumerate() {
                if !integrated && self.scene_convex_rounded_poses[env][pair] == *pose {
                    continue;
                }
                let (row, owner, count) = self.scene_convex_rounded_rows[env][pair];
                // Every manifold row owns the stationary convex pose.
                for contact_row in row..row + count {
                    let start = contact_row as u64 * size_of::<PackedSphere>() as u64;
                    queue.write_buffer(
                        &self.spheres,
                        start + core::mem::offset_of!(PackedSphere, center_radius) as u64,
                        bytemuck::cast_slice(&pose[..3]),
                    );
                    queue.write_buffer(
                        &self.spheres,
                        start + core::mem::offset_of!(PackedSphere, first_axis_end) as u64,
                        bytemuck::cast_slice(&pose[3..]),
                    );
                    queue.write_buffer(
                        &self.spheres,
                        start + core::mem::offset_of!(PackedSphere, impulses) as u64,
                        bytemuck::cast_slice(&[0.0f32; 4]),
                    );
                }
                if let Some(activity) = &self.contact_activity {
                    queue.write_buffer(
                        &activity.wake_requests,
                        owner as u64 * 4,
                        &1u32.to_ne_bytes(),
                    );
                }
                changed = true;
            }
        }
        self.scene_convex_rounded_poses = packed;
        Ok(changed)
    }

    /// Discard contact impulses after a state reset or root teleport.
    pub fn clear_cached_impulses(&self, queue: &wgpu::Queue) {
        if let Some(activity) = &self.contact_activity {
            activity.clear_wake(queue);
            queue.write_buffer(
                &activity.previous_geometry,
                0,
                &vec![0; activity.previous_geometry.size() as usize],
            );
            queue.write_buffer(&activity.flags, 0, &vec![0; activity.flags.size() as usize]);
            queue.write_buffer(
                &activity.parents,
                0,
                bytemuck::cast_slice(&activity.seed_parents),
            );
        }
        let zeros = [0u8; 32];
        let mut previous = 0;
        for range in &self.dynamic_rows {
            for index in previous..range.start {
                let offset = index * size_of::<PackedSphere>() + offset_of!(PackedSphere, impulses);
                queue.write_buffer(&self.spheres, offset as u64, &zeros);
            }
            previous = range.end;
        }
        for index in previous..self.sphere_count {
            let offset = index * size_of::<PackedSphere>() + offset_of!(PackedSphere, impulses);
            queue.write_buffer(&self.spheres, offset as u64, &zeros);
        }
        for range in &self.dynamic_rows {
            if !range.is_empty() {
                let offset = range.start as u64 * size_of::<PackedSphere>() as u64;
                let bytes = vec![0u8; range.len() * size_of::<PackedSphere>()];
                queue.write_buffer(&self.spheres, offset, &bytes);
            }
        }
    }

    /// Update reserved joint-friction bounds before the next GPU submission.
    ///
    /// An environment with no friction rows can only keep zero friction.
    pub fn update_joint_frictions(
        &self,
        queue: &wgpu::Queue,
        frictions: &[Vec<f64>],
    ) -> Result<(), GpuArticulatedGroundContactError> {
        if frictions.len() != self.friction_dimensions.len() {
            return Err(GpuArticulatedGroundContactError::InvalidInput);
        }
        let mut updates = Vec::new();
        for (environment, values) in frictions.iter().enumerate() {
            if values.len() != self.friction_dimensions[environment] {
                return Err(GpuArticulatedGroundContactError::InvalidInput);
            }
            for (coordinate, &friction) in values.iter().enumerate() {
                if !friction.is_finite() || friction < 0.0 {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                let bound = finite_f32(friction * self.timestep)?;
                if friction > 0.0 && bound <= 0.0 {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                if let Some(rows) = &self.friction_rows[environment] {
                    let offset = (rows.start + coordinate)
                        .checked_mul(size_of::<PackedSphere>())
                        .and_then(|value| value.checked_add(offset_of!(PackedSphere, material)))
                        .ok_or(GpuArticulatedGroundContactError::Capacity)?;
                    updates.push((offset as u64, bound));
                } else if friction > 0.0 {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
            }
        }
        for (offset, bound) in updates {
            queue.write_buffer(&self.spheres, offset, bytemuck::bytes_of(&bound));
        }
        Ok(())
    }

    /// Populate sphere and capsule contact rows from eligible resident LBVH candidates.
    ///
    /// Continued pairs retain their impulse history; inactive pairs lose it.
    /// The pair-to-row mapping is stable, independent of candidate order.
    pub fn encode_dynamic_shape_rows(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        lbvh: &GpuLbvh,
        candidates: &GpuLbvhResidentPairs,
    ) -> Result<(), GpuArticulatedGroundContactError> {
        let (
            Some(metadata),
            Some(pair_slots),
            Some(pipeline),
            Some(inactive_pipeline),
            Some(flags),
            Some(row_indices),
        ) = (
            &self.dynamic_shapes,
            &self.dynamic_pair_slots,
            &self.dynamic_pipeline,
            &self.dynamic_inactive_pipeline,
            &self.dynamic_flags,
            &self.dynamic_row_indices,
        )
        else {
            return Ok(());
        };
        encoder.clear_buffer(flags, 0, None);
        let dispatch =
            lbvh.encode_candidate_dispatch_args(&self.device, encoder, candidates, 64)?;
        let buffers = [
            &candidates.pairs,
            &candidates.counter,
            metadata,
            &self.spheres,
            pair_slots,
            flags,
        ];
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera articulated dynamic shape row bindings"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &buffers
                .iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Tessera articulated dynamic shape rows"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups_indirect(&dispatch, 0);
        drop(pass);
        let inactive_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera articulated inactive row bindings"),
            layout: &inactive_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.spheres.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: flags.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: row_indices.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Tessera articulated inactive dynamic rows"),
            timestamp_writes: None,
        });
        pass.set_pipeline(inactive_pipeline);
        pass.set_bind_group(0, &inactive_group, &[]);
        pass.dispatch_workgroups(
            (row_indices.size() / size_of::<u32>() as u64).div_ceil(64) as u32,
            1,
            1,
        );
        Ok(())
    }

    /// Enable optional GPU reduction of solved contact impulses into per-link flags.
    /// Newly enabled flags are zero until the next encoded step. Enabling twice
    /// preserves the previous result. Bilateral constraints do not count as contacts.
    pub fn enable_contact_activity(&mut self) -> Result<(), GpuArticulatedGroundContactError> {
        if self.contact_activity.is_none() {
            self.contact_activity = Some(ContactActivity::new(
                &self.device,
                &self.contact_ranges,
                &self.link_ranges,
            )?);
        }
        Ok(())
    }

    /// Set disjoint mobility groups in local link indices for every environment.
    /// Ungrouped links remain independent. All input is validated before upload.
    /// A successful update clears activity and replaces component labels, retaining
    /// source state/faults. Reset/teleport retain these groups. Requires activity enabled.
    pub fn update_contact_mobility_groups(
        &mut self,
        queue: &wgpu::Queue,
        groups: &[Vec<Vec<usize>>],
    ) -> Result<(), GpuArticulatedGroundContactError> {
        let activity = self
            .contact_activity
            .as_mut()
            .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
        let seeds = pack_mobility_groups(&self.link_ranges, groups)?;
        queue.write_buffer(&activity.mobility_roots, 0, bytemuck::cast_slice(&seeds));
        queue.write_buffer(&activity.parents, 0, bytemuck::cast_slice(&seeds));
        queue.write_buffer(&activity.flags, 0, &vec![0; activity.flags.size() as usize]);
        activity.seed_parents = seeds;
        activity.clear_wake(queue);
        queue.write_buffer(
            &activity.previous_geometry,
            0,
            &vec![0; activity.previous_geometry.size() as usize],
        );
        Ok(())
    }

    pub(crate) fn enable_contact_load_wake(
        &mut self,
        loads: &wgpu::Buffer,
    ) -> Result<(), GpuArticulatedGroundContactError> {
        if self.contact_activity.is_none() {
            return Err(GpuArticulatedGroundContactError::InvalidInput);
        }
        if self.load_wake.is_none() {
            self.load_wake = Some(LoadWake::new(&self.device, loads, &self.link_ranges)?);
        }
        Ok(())
    }

    /// Enable/disable one-step wake requests when a link loses its last geometric
    /// contact. Geometry history is tracked even while disabled; reset, teleport
    /// and mobility updates clear that history. Requires activity enabled.
    pub fn update_contact_loss_wake(
        &self,
        queue: &wgpu::Queue,
        enabled: bool,
    ) -> Result<(), GpuArticulatedGroundContactError> {
        let activity = self
            .contact_activity
            .as_ref()
            .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
        queue.write_buffer(
            &activity.loss_params,
            0,
            bytemuck::cast_slice(&[u32::from(enabled), 0, 0, 0]),
        );
        Ok(())
    }

    /// Download touching/penetrating contact presence from the last solve, including
    /// zero-impulse contacts. Speculative positive-separation impulses are excluded.
    /// Bilateral constraints are not geometric contacts. Requires activity enabled.
    pub fn readback_geometric_contacts(
        &self,
        queue: &wgpu::Queue,
    ) -> Result<Vec<Vec<bool>>, GpuArticulatedGroundContactError> {
        self.readback_activity_flags(queue, 2)
    }

    pub(crate) fn enable_contact_gravity_wake(
        &mut self,
        queue: &wgpu::Queue,
        gravity: &wgpu::Buffer,
    ) -> Result<(), GpuArticulatedGroundContactError> {
        if self.contact_activity.is_none() {
            return Err(GpuArticulatedGroundContactError::InvalidInput);
        }
        if self.gravity_wake.is_none() {
            self.gravity_wake = Some(GravityWake::new(
                &self.device,
                queue,
                gravity,
                &self.motion_layout,
                self.environment_count,
            )?);
        }
        Ok(())
    }

    pub(crate) fn enable_contact_actuation_wake(
        &mut self,
        efforts: &wgpu::Buffer,
    ) -> Result<(), GpuArticulatedGroundContactError> {
        if self.contact_activity.is_none() || efforts.size() != self.velocities.size() {
            return Err(GpuArticulatedGroundContactError::InvalidInput);
        }
        if self.actuation_wake.is_none() {
            let pass = MotionWake::build(
                &self.device,
                &self.motion_layout,
                include_str!("gpu_articulated_actuation_wake.wgsl"),
                "request_actuation_wake",
            )?;
            self.actuation_wake = Some((pass, efforts.clone()));
        }
        Ok(())
    }

    /// Download geometric contacts against an external static shape or plane.
    /// This identifies one-sided touching rows; it does not prove support against gravity.
    pub fn readback_external_static_contacts(
        &self,
        queue: &wgpu::Queue,
    ) -> Result<Vec<Vec<bool>>, GpuArticulatedGroundContactError> {
        self.readback_activity_flags(queue, 4)
    }

    pub(crate) fn enable_contact_gravity_support(
        &mut self,
        gravity: &wgpu::Buffer,
    ) -> Result<(), GpuArticulatedGroundContactError> {
        if gravity.size() != self.environment_count as u64 * 16 {
            return Err(GpuArticulatedGroundContactError::InvalidInput);
        }
        let activity = self
            .contact_activity
            .as_mut()
            .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
        activity.support_gravity = gravity.clone();
        Ok(())
    }

    /// Download touching static contacts whose signed normal opposes gravity.
    /// This is a local support direction test, not a full equilibrium proof.
    pub fn readback_contact_gravity_support(
        &self,
        queue: &wgpu::Queue,
    ) -> Result<Vec<Vec<bool>>, GpuArticulatedGroundContactError> {
        self.readback_activity_flags(queue, 8)
    }

    /// Download links constrained directly to world by a bilateral point or fixed row.
    /// This is an anchor classification; point anchors do not constrain all rotations.
    pub fn readback_contact_world_anchors(
        &self,
        queue: &wgpu::Queue,
    ) -> Result<Vec<Vec<bool>>, GpuArticulatedGroundContactError> {
        self.readback_activity_flags(queue, 16)
    }

    /// Download links with a full fixed bilateral world anchor, excluding point anchors.
    pub fn readback_contact_fixed_world_anchors(
        &self,
        queue: &wgpu::Queue,
    ) -> Result<Vec<Vec<bool>>, GpuArticulatedGroundContactError> {
        self.readback_activity_flags(queue, 32)
    }

    /// Configure component idle tracking and predicted-motion wake thresholds.
    /// This only identifies sleep candidates; integration is not frozen yet.
    /// Every positive-mass member must remain quiet, and some member must touch.
    /// Invalid settings preserve thresholds and history. Valid updates clear idle history.
    pub fn update_contact_idle_settings(
        &mut self,
        queue: &wgpu::Queue,
        settings: &[Option<SleepSettings>],
    ) -> Result<(), GpuArticulatedGroundContactError> {
        if settings.len() != self.environment_count {
            return Err(GpuArticulatedGroundContactError::InvalidInput);
        }
        let links = settings
            .iter()
            .zip(&self.link_ranges)
            .map(|(&setting, range)| vec![setting; range.len()])
            .collect::<Vec<_>>();
        self.update_contact_link_idle_settings(queue, &links)
    }

    /// Configure idle and motion thresholds per environment/local link.
    /// None disables a physical member and vetoes its entire component sleeping.
    /// Massless helpers inherit the physical component's candidate flags regardless
    /// of their own thresholds or enabled setting. All inputs are validated
    /// before upload; successful changes reset idle history and candidates.
    pub fn update_contact_link_idle_settings(
        &mut self,
        queue: &wgpu::Queue,
        settings: &[Vec<Option<SleepSettings>>],
    ) -> Result<(), GpuArticulatedGroundContactError> {
        if self.contact_activity.is_none()
            || settings.len() != self.link_ranges.len()
            || settings
                .iter()
                .zip(&self.link_ranges)
                .any(|(s, r)| s.len() != r.len())
        {
            return Err(GpuArticulatedGroundContactError::InvalidInput);
        }
        let mut limits = Vec::with_capacity(self.motion_layout.len());
        let mut times = Vec::with_capacity(self.motion_layout.len());
        for setting in settings.iter().flatten() {
            if let Some(setting) = setting {
                if !setting.is_valid() {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                let time = finite_f32(setting.time_threshold)?;
                let linear = finite_f32(setting.linear_velocity_threshold)?;
                let angular = finite_f32(setting.angular_velocity_threshold)?;
                if time == 0.0
                    || (setting.linear_velocity_threshold > 0.0 && linear == 0.0)
                    || (setting.angular_velocity_threshold > 0.0 && angular == 0.0)
                {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                let enabled = if setting.enabled { 1.0 } else { 0.0 };
                limits.push([linear, angular, enabled]);
                times.push((time, enabled));
            } else {
                limits.push([0.0; 3]);
                times.push((1.0, 0.0));
            }
        }
        let mut links = self.motion_layout.clone();
        for (link, limit) in links.iter_mut().zip(&limits) {
            link.thresholds[0] = limit[0];
            link.thresholds[1] = limit[1];
            link.thresholds[3] = limit[2];
        }
        let packed = links
            .iter()
            .zip(&times)
            .map(|(link, &(time, enabled))| {
                [link.thresholds[2], time, enabled, link.center_of_mass[3]]
            })
            .collect::<Vec<_>>();
        if let Some(motion) = &self.motion_wake {
            queue.write_buffer(&motion.metadata, 0, bytemuck::cast_slice(&links));
        } else {
            self.motion_wake = Some(MotionWake::new(&self.device, &links)?);
        }
        let activity = self
            .contact_activity
            .as_mut()
            .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
        queue.write_buffer(&activity.idle_parameters, 0, bytemuck::cast_slice(&packed));
        let zeros = vec![0u8; activity.idle_time.size() as usize];
        queue.write_buffer(&activity.idle_time, 0, &zeros);
        queue.write_buffer(&activity.sleep_candidates, 0, &zeros);
        activity.idle_enabled = true;
        activity.idle_sources = Some([self.state_status.clone(), self.mass_status.clone()]);
        Ok(())
    }

    /// Require a gravity-opposing external static contact somewhere in each idle component.
    /// False restores the geometric-contact criterion. Changing policy clears idle history.
    /// Gravity support must be enabled separately. This does not stop integration.
    pub fn update_contact_idle_requires_static_support(
        &self,
        queue: &wgpu::Queue,
        required: bool,
    ) -> Result<(), GpuArticulatedGroundContactError> {
        let activity = self
            .contact_activity
            .as_ref()
            .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
        queue.write_buffer(
            &activity.idle_policy,
            0,
            bytemuck::cast_slice(&[if required { 40u32 } else { 2u32 }, 0, 0, 0]),
        );
        let zeros = vec![0u8; activity.idle_time.size() as usize];
        queue.write_buffer(&activity.idle_time, 0, &zeros);
        queue.write_buffer(&activity.sleep_candidates, 0, &zeros);
        Ok(())
    }

    pub(crate) fn contact_sleep_candidate_buffer(
        &self,
    ) -> Result<&wgpu::Buffer, GpuArticulatedGroundContactError> {
        self.contact_activity
            .as_ref()
            .map(|activity| &activity.sleep_candidates)
            .ok_or(GpuArticulatedGroundContactError::InvalidInput)
    }

    /// Download candidate flags after validating the source state.
    /// These flags do not imply that integration has been stopped.
    pub fn readback_contact_sleep_candidates(
        &self,
        queue: &wgpu::Queue,
    ) -> Result<Vec<Vec<bool>>, GpuArticulatedGroundContactError> {
        let _ = self.readback_contact_activity(queue)?;
        let activity = self
            .contact_activity
            .as_ref()
            .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
        let bytes = crate::gpu_articulated_mass::read_buffer(
            &self.device,
            queue,
            &activity.sleep_candidates,
        )
        .map_err(|e| GpuArticulatedGroundContactError::Readback(e.to_string()))?;
        let values = bytes
            .chunks_exact(4)
            .map(|v| u32::from_ne_bytes([v[0], v[1], v[2], v[3]]) != 0)
            .collect::<Vec<_>>();
        Ok(self
            .link_ranges
            .iter()
            .map(|r| values[r.clone()].to_vec())
            .collect())
    }

    /// Configure automatic predicted-motion wake requests per environment.
    /// None disables motion wake for that environment. This evaluates positive-mass
    /// links at their origins using tangent Jacobians before joint-limit integration.
    /// Requires activity enabled. Invalid input preserves the old configuration.
    /// Idle time and sleeping state are not updated by this pass.
    pub fn update_contact_motion_wake(
        &mut self,
        queue: &wgpu::Queue,
        settings: &[Option<SleepSettings>],
    ) -> Result<(), GpuArticulatedGroundContactError> {
        if self.contact_activity.is_none() || settings.len() != self.environment_count {
            return Err(GpuArticulatedGroundContactError::InvalidInput);
        }
        let mut limits = Vec::with_capacity(settings.len());
        for setting in settings {
            let Some(setting) = setting else {
                limits.push([0.0; 3]);
                continue;
            };
            if !setting.is_valid() {
                return Err(GpuArticulatedGroundContactError::InvalidInput);
            }
            let linear = finite_f32(setting.linear_velocity_threshold)?;
            let angular = finite_f32(setting.angular_velocity_threshold)?;
            if (setting.linear_velocity_threshold > 0.0 && linear == 0.0)
                || (setting.angular_velocity_threshold > 0.0 && angular == 0.0)
            {
                return Err(GpuArticulatedGroundContactError::InvalidInput);
            }
            limits.push([linear, angular, if setting.enabled { 1.0 } else { 0.0 }]);
        }
        let mut links = self.motion_layout.clone();
        for link in &mut links {
            let limit = limits[link.indices[0] as usize];
            link.thresholds[0] = limit[0];
            link.thresholds[1] = limit[1];
            link.thresholds[3] = limit[2];
        }
        if let Some(motion) = &self.motion_wake {
            queue.write_buffer(&motion.metadata, 0, bytemuck::cast_slice(&links));
        } else {
            self.motion_wake = Some(MotionWake::new(&self.device, &links)?);
        }
        Ok(())
    }

    /// Replace pending one-step wake requests in environment/stable link order.
    /// Requests are consumed on the next encoded step and propagated through the
    /// resulting contact/bilateral/mobility components. This does not change sleep
    /// state or velocities. Reset, teleport and mobility updates clear requests.
    /// Invalid dimensions preserve pending requests and previous diagnostics.
    pub fn update_contact_wake_requests(
        &self,
        queue: &wgpu::Queue,
        requests: &[Vec<bool>],
    ) -> Result<(), GpuArticulatedGroundContactError> {
        let activity = self
            .contact_activity
            .as_ref()
            .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
        if requests.len() != self.link_ranges.len()
            || requests
                .iter()
                .zip(&self.link_ranges)
                .any(|(values, range)| values.len() != range.len())
        {
            return Err(GpuArticulatedGroundContactError::InvalidInput);
        }
        let mut packed = vec![0u32; activity.seed_parents.len()];
        for (values, range) in requests.iter().zip(&self.link_ranges) {
            for (slot, &request) in packed[range.clone()].iter_mut().zip(values) {
                *slot = u32::from(request);
            }
        }
        queue.write_buffer(&activity.wake_requests, 0, bytemuck::cast_slice(&packed));
        Ok(())
    }

    /// Add requests without clearing pending requests from other wake sources.
    pub(crate) fn add_contact_wake_requests(
        &self,
        queue: &wgpu::Queue,
        requests: &[Vec<bool>],
    ) -> Result<(), GpuArticulatedGroundContactError> {
        let activity = self
            .contact_activity
            .as_ref()
            .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
        if requests.len() != self.link_ranges.len()
            || requests
                .iter()
                .zip(&self.link_ranges)
                .any(|(v, r)| v.len() != r.len())
        {
            return Err(GpuArticulatedGroundContactError::InvalidInput);
        }
        for (values, range) in requests.iter().zip(&self.link_ranges) {
            for (link, &requested) in values.iter().enumerate() {
                if requested {
                    queue.write_buffer(
                        &activity.wake_requests,
                        (range.start + link) as u64 * 4,
                        &1u32.to_ne_bytes(),
                    );
                }
            }
        }
        Ok(())
    }

    /// Download wake requests propagated through the last step's components.
    /// Source faults reject the result. This reports requests, not sleeping state.
    pub fn readback_contact_wake_requests(
        &self,
        queue: &wgpu::Queue,
    ) -> Result<Vec<Vec<bool>>, GpuArticulatedGroundContactError> {
        let _ = self.readback_contact_activity(queue)?;
        let activity = self
            .contact_activity
            .as_ref()
            .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
        let bytes =
            crate::gpu_articulated_mass::read_buffer(&self.device, queue, &activity.link_wake)
                .map_err(|e| GpuArticulatedGroundContactError::Readback(e.to_string()))?;
        let values = bytes
            .chunks_exact(4)
            .map(|v| u32::from_ne_bytes([v[0], v[1], v[2], v[3]]) != 0)
            .collect::<Vec<_>>();
        Ok(self
            .link_ranges
            .iter()
            .map(|range| values[range.clone()].to_vec())
            .collect())
    }

    /// Download per-link solved-impulse flags, in environment and stable link order.
    /// Zero-impulse geometric contacts are not reported. This is not a sleep state.
    /// Requires `enable_contact_activity`; source faults reject the whole result.
    /// Serialize submissions and readback externally.
    pub fn readback_contact_activity(
        &self,
        queue: &wgpu::Queue,
    ) -> Result<Vec<Vec<bool>>, GpuArticulatedGroundContactError> {
        self.readback_activity_flags(queue, 1)
    }

    fn readback_activity_flags(
        &self,
        queue: &wgpu::Queue,
        mask: u32,
    ) -> Result<Vec<Vec<bool>>, GpuArticulatedGroundContactError> {
        let activity = self
            .contact_activity
            .as_ref()
            .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
        let read = |buffer| {
            crate::gpu_articulated_mass::read_buffer(&self.device, queue, buffer)
                .map_err(|e| GpuArticulatedGroundContactError::Readback(e.to_string()))
        };
        for status_buffer in [&self.state_status, &self.mass_status] {
            let status = read(status_buffer)?;
            for (environment, bytes) in status.chunks_exact(4).enumerate() {
                if u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) != 0 {
                    return Err(GpuArticulatedGroundContactError::SourceFault(environment));
                }
            }
        }
        let bytes = read(&activity.flags)?;
        let flags = bytes
            .chunks_exact(4)
            .map(|value| u32::from_ne_bytes([value[0], value[1], value[2], value[3]]) & mask != 0)
            .collect::<Vec<_>>();
        Ok(self
            .link_ranges
            .iter()
            .map(|range| flags[range.clone()].to_vec())
            .collect())
    }

    /// Download contact/bilateral connectivity labels in stable link order.
    /// Each label is the smallest local link index in the connected component.
    /// Bilateral link constraints connect even with zero impulse. Mobility groups
    /// supplied by `update_contact_mobility_groups` are included. Unspecified
    /// articulation joints, scalar couplings and zero-impulse geometry are not edges.
    /// Requires `enable_contact_activity`; faults reject the result.
    pub fn readback_contact_components(
        &self,
        queue: &wgpu::Queue,
    ) -> Result<Vec<Vec<usize>>, GpuArticulatedGroundContactError> {
        let _ = self.readback_contact_activity(queue)?;
        let activity = self
            .contact_activity
            .as_ref()
            .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
        let bytes =
            crate::gpu_articulated_mass::read_buffer(&self.device, queue, &activity.parents)
                .map_err(|e| GpuArticulatedGroundContactError::Readback(e.to_string()))?;
        let parents = bytes
            .chunks_exact(4)
            .map(|v| u32::from_ne_bytes([v[0], v[1], v[2], v[3]]) as usize)
            .collect::<Vec<_>>();
        let mut result = Vec::with_capacity(self.link_ranges.len());
        for range in &self.link_ranges {
            let mut labels = Vec::with_capacity(range.len());
            for link in range.clone() {
                let mut root = link;
                loop {
                    let parent = parents[root];
                    if !range.contains(&parent) || parent > root {
                        return Err(GpuArticulatedGroundContactError::InvalidInput);
                    }
                    if parent == root {
                        break;
                    }
                    root = parent;
                }
                labels.push(root - range.start);
            }
            result.push(labels);
        }
        Ok(result)
    }

    /// Download current body origins, orientations, and origin linear velocities.
    /// Rejects faulted environments. Serialize submissions and readback externally.
    pub fn readback_external_sphere_orbits(
        &self,
        queue: &wgpu::Queue,
    ) -> Result<GpuArticulatedExternalSphereOrbits, GpuArticulatedGroundContactError> {
        let read = |buffer| {
            crate::gpu_articulated_mass::read_buffer(&self.device, queue, buffer)
                .map_err(|e| GpuArticulatedGroundContactError::Readback(e.to_string()))
        };
        let status = read(&self.state_status)?;
        for (environment, bytes) in status.chunks_exact(4).enumerate() {
            if u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) != 0 {
                return Err(GpuArticulatedGroundContactError::SourceFault(environment));
            }
        }
        let bytes = read(&self.sphere_orbits)?;
        let rows: Vec<PackedSphereOrbit> = bytes
            .chunks_exact(size_of::<PackedSphereOrbit>())
            .map(bytemuck::pod_read_unaligned)
            .collect();
        let orbits = |mapping: &[Vec<(usize, usize)>]| {
            mapping
                .iter()
                .map(|pairs| {
                    pairs
                        .iter()
                        .map(|&(index, _)| {
                            let row = rows
                                .get(index)
                                .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
                            if row.origin[3] == 0.0 {
                                return Ok(None);
                            }
                            if row.origin[3] != 1.0
                                || row.origin[..3]
                                    .iter()
                                    .chain(&row.linear[..3])
                                    .chain(&row.orientation)
                                    .any(|value| !value.is_finite())
                            {
                                return Err(GpuArticulatedGroundContactError::InvalidInput);
                            }
                            let quaternion = Quaternion::new(
                                row.orientation[3] as f64,
                                row.orientation[0] as f64,
                                row.orientation[1] as f64,
                                row.orientation[2] as f64,
                            );
                            if (quaternion.norm_squared() - 1.0).abs() > 1e-3 {
                                return Err(GpuArticulatedGroundContactError::InvalidInput);
                            }
                            Ok(Some(GpuArticulatedSphereOrbit {
                                origin: Vector3::new(
                                    row.origin[0] as f64,
                                    row.origin[1] as f64,
                                    row.origin[2] as f64,
                                ),
                                linear_velocity: Vector3::new(
                                    row.linear[0] as f64,
                                    row.linear[1] as f64,
                                    row.linear[2] as f64,
                                ),
                                orientation: UnitQuaternion::new_normalize(quaternion),
                            }))
                        })
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<Vec<_>, _>>()
        };
        Ok(GpuArticulatedExternalSphereOrbits {
            spheres: orbits(&self.static_sphere_rows)?,
            capsules: orbits(&self.static_capsule_sphere_rows)?,
            boxes: orbits(&self.static_box_sphere_rows)?,
            axial: orbits(&self.static_axial_sphere_rows)?,
            convex: orbits(&self.static_convex_sphere_rows)?,
        })
    }

    /// Download current external sphere centers, including GPU-integrated motion.
    /// Rejects faulted environments. Serialize submissions and readback externally.
    pub fn readback_external_sphere_centers(
        &self,
        queue: &wgpu::Queue,
    ) -> Result<GpuArticulatedExternalSphereCenters, GpuArticulatedGroundContactError> {
        let read = |buffer| {
            crate::gpu_articulated_mass::read_buffer(&self.device, queue, buffer)
                .map_err(|e| GpuArticulatedGroundContactError::Readback(e.to_string()))
        };
        let status = read(&self.state_status)?;
        for (environment, bytes) in status.chunks_exact(4).enumerate() {
            if u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) != 0 {
                return Err(GpuArticulatedGroundContactError::SourceFault(environment));
            }
        }
        let bytes = read(&self.spheres)?;
        let rows: Vec<PackedSphere> = bytes
            .chunks_exact(size_of::<PackedSphere>())
            .map(bytemuck::pod_read_unaligned)
            .collect();
        let centers = |mapping: &[Vec<(usize, usize)>], use_other: bool| {
            mapping
                .iter()
                .map(|pairs| {
                    pairs
                        .iter()
                        .map(|&(index, _)| {
                            let row = rows
                                .get(index)
                                .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
                            let packed = if use_other {
                                row.other_center_radius
                            } else {
                                row.plane
                            };
                            if packed[..3].iter().any(|value| !value.is_finite()) {
                                return Err(GpuArticulatedGroundContactError::InvalidInput);
                            }
                            Ok(Vector3::new(
                                packed[0] as f64,
                                packed[1] as f64,
                                packed[2] as f64,
                            ))
                        })
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<Vec<_>, _>>()
        };
        Ok(GpuArticulatedExternalSphereCenters {
            spheres: centers(&self.static_sphere_rows, true)?,
            capsules: centers(&self.static_capsule_sphere_rows, true)?,
            boxes: centers(&self.static_box_sphere_rows, false)?,
            axial: centers(&self.static_axial_sphere_rows, false)?,
            convex: centers(&self.static_convex_sphere_rows, false)?,
        })
    }

    /// Download contact wrenches from the last solved step in environment order.
    /// Scalar joint and bilateral constraint rows are excluded.
    pub fn readback_contacts(
        &self,
        queue: &wgpu::Queue,
    ) -> Result<Vec<Vec<GpuArticulatedLinkContact>>, GpuArticulatedGroundContactError> {
        let read = |buffer| {
            crate::gpu_articulated_mass::read_buffer(&self.device, queue, buffer)
                .map_err(|e| GpuArticulatedGroundContactError::Readback(e.to_string()))
        };
        let status = read(&self.state_status)?;
        for (environment, bytes) in status.chunks_exact(4).enumerate() {
            if u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) != 0 {
                return Err(GpuArticulatedGroundContactError::SourceFault(environment));
            }
        }
        let bytes = read(&self.spheres)?;
        let rows: Vec<PackedSphere> = bytes
            .chunks_exact(size_of::<PackedSphere>())
            .map(bytemuck::pod_read_unaligned)
            .collect();
        let vector = |v: [f32; 4]| Vector3::new(v[0] as f64, v[1] as f64, v[2] as f64);
        let mut result = vec![Vec::new(); self.environment_count];
        for (environment, range) in self.contact_ranges.iter().enumerate() {
            for index in range.clone() {
                let row = rows
                    .get(index)
                    .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
                if row.impulses[..3].iter().all(|v| *v == 0.0) {
                    continue;
                }
                let owners = [
                    (
                        row.indices[0],
                        row.diagnostic_first,
                        row.diagnostic_first_origin,
                    ),
                    (
                        row.indices[2],
                        row.diagnostic_second,
                        row.diagnostic_second_origin,
                    ),
                ];
                if owners.iter().all(|(_, _, origin)| origin[3] == 0.0) {
                    continue;
                }
                let normal = vector(row.previous_normal);
                if !normal.iter().all(|v| v.is_finite())
                    || (normal.norm_squared() - 1.0).abs() > 1e-4
                    || row.impulses[..3].iter().any(|v| !v.is_finite())
                {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                let reference = if normal.z.abs() >= 0.9 {
                    Vector3::x()
                } else {
                    Vector3::z()
                };
                let tangent = normal.cross(&reference).normalize();
                let force = (normal * row.impulses[0] as f64
                    + tangent * row.impulses[1] as f64
                    + normal.cross(&tangent) * row.impulses[2] as f64)
                    / self.timestep;
                for (link, point, origin) in owners {
                    let sign = origin[3] as f64;
                    if sign == 0.0 {
                        continue;
                    }
                    if !self.link_ranges[environment].contains(&(link as usize))
                        || sign.abs() != 1.0
                        || point.iter().chain(origin.iter()).any(|v| !v.is_finite())
                    {
                        return Err(GpuArticulatedGroundContactError::InvalidInput);
                    }
                    let position = vector(point);
                    let force = force * sign;
                    result[environment].push(GpuArticulatedLinkContact {
                        link: link as usize - self.link_ranges[environment].start,
                        position,
                        normal: normal * sign,
                        force,
                        torque: (position - vector(origin)).cross(&force),
                        distance: point[3] as f64,
                    });
                }
            }
        }
        Ok(result)
    }

    #[cfg(test)]
    pub(crate) fn readback_dynamic_row_state(&self, queue: &wgpu::Queue) -> Vec<(f32, [f32; 4])> {
        let bytes =
            crate::gpu_articulated_mass::read_buffer(&self.device, queue, &self.spheres).unwrap();
        let rows = bytes
            .chunks_exact(size_of::<PackedSphere>())
            .map(bytemuck::pod_read_unaligned::<PackedSphere>)
            .collect::<Vec<_>>();
        self.dynamic_rows
            .iter()
            .flat_map(|range| range.clone())
            .map(|index| (rows[index].center_radius[3], rows[index].impulses))
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn readback_dynamic_row_materials(&self, queue: &wgpu::Queue) -> Vec<[f32; 4]> {
        let bytes =
            crate::gpu_articulated_mass::read_buffer(&self.device, queue, &self.spheres).unwrap();
        let rows = bytes
            .chunks_exact(size_of::<PackedSphere>())
            .map(bytemuck::pod_read_unaligned::<PackedSphere>)
            .collect::<Vec<_>>();
        self.dynamic_rows
            .iter()
            .flat_map(|range| range.clone())
            .map(|index| rows[index].material)
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn readback_ground_manifold_state(
        &self,
        queue: &wgpu::Queue,
    ) -> Vec<(u32, [f32; 4])> {
        let bytes =
            crate::gpu_articulated_mass::read_buffer(&self.device, queue, &self.spheres).unwrap();
        bytes
            .chunks_exact(size_of::<PackedSphere>())
            .map(bytemuck::pod_read_unaligned::<PackedSphere>)
            .filter(|row| {
                row.material[3] == 5.0
                    || row.material[3] == 10.0
                    || row.material[3] == 12.0
                    || row.material[3] == 13.0
            })
            .map(|row| (row.indices[3], row.impulses))
            .collect()
    }

    /// Encode contact impulses after mass assembly with inverse and before state integration.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        if let Some(preparation) = &self.coupling_preparation {
            let buffers = [
                &self.spheres,
                &preparation.positions,
                &self.state_status,
                &preparation.indices,
            ];
            let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Tessera articulated coupling bindings"),
                layout: &preparation.pipeline.get_bind_group_layout(0),
                entries: &buffers
                    .iter()
                    .enumerate()
                    .map(|(binding, buffer)| wgpu::BindGroupEntry {
                        binding: binding as u32,
                        resource: buffer.as_entire_binding(),
                    })
                    .collect::<Vec<_>>(),
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Tessera articulated coupling preparation"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&preparation.pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(preparation.count.div_ceil(64) as u32, 1, 1);
        }
        let buffers = [
            &self.systems,
            &self.spheres,
            &self.poses,
            &self.link_terms,
            &self.inverse,
            &self.velocities,
            &self.accelerations,
            &self.state_status,
        ];
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera articulated contact bindings"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &buffers
                .iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Tessera articulated contact"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(self.environment_count as u32, 1, 1);
        drop(pass);
        if let Some(activity) = &self.contact_activity {
            if let Some(load) = &self.load_wake {
                load.encode(
                    &self.device,
                    encoder,
                    &self.state_status,
                    &activity.wake_requests,
                    &self.mass_status,
                );
            }
            if let Some(gravity) = &self.gravity_wake {
                gravity.encode(
                    &self.device,
                    encoder,
                    &self.state_status,
                    &self.mass_status,
                    &activity.wake_requests,
                );
            }
            if let Some((actuation, efforts)) = &self.actuation_wake {
                actuation.encode(
                    &self.device,
                    encoder,
                    [
                        &self.link_terms,
                        efforts,
                        &self.state_status,
                        &activity.wake_requests,
                        &self.mass_status,
                    ],
                );
            }
            if let Some(motion) = &self.motion_wake {
                motion.encode(
                    &self.device,
                    encoder,
                    [
                        &self.link_terms,
                        &self.poses,
                        &self.velocities,
                        &self.accelerations,
                        &self.state_status,
                        &activity.wake_requests,
                        &self.mass_status,
                    ],
                );
            }
            activity.encode(&self.device, encoder, &self.spheres, &self.state_status);
        }
    }
}

fn checked_u32(value: usize) -> Result<u32, GpuArticulatedGroundContactError> {
    u32::try_from(value).map_err(|_| GpuArticulatedGroundContactError::Capacity)
}

fn pack_mobility_groups(
    links: &[Range<usize>],
    groups: &[Vec<Vec<usize>>],
) -> Result<Vec<u32>, GpuArticulatedGroundContactError> {
    if links.len() != groups.len() {
        return Err(GpuArticulatedGroundContactError::InvalidInput);
    }
    let count = links.last().map_or(0, |range| range.end).max(1);
    let mut seeds = (0..checked_u32(count)?).collect::<Vec<_>>();
    for (range, groups) in links.iter().zip(groups) {
        let mut used = vec![false; range.len()];
        for group in groups {
            let root = group
                .iter()
                .copied()
                .min()
                .ok_or(GpuArticulatedGroundContactError::InvalidInput)?;
            for &link in group {
                if link >= range.len() || used[link] {
                    return Err(GpuArticulatedGroundContactError::InvalidInput);
                }
                used[link] = true;
                seeds[range.start + link] = checked_u32(range.start + root)?;
            }
        }
    }
    Ok(seeds)
}

fn mesh_index_f32(value: usize) -> Result<f32, GpuArticulatedGroundContactError> {
    if value > (1 << 24) {
        return Err(GpuArticulatedGroundContactError::Capacity);
    }
    Ok(checked_u32(value)? as f32)
}

fn pack_mesh_geometry(
    mesh: &Arc<TriangleMeshGeometry>,
    geometry: &mut Vec<PackedSphere>,
    cache: &mut HashMap<usize, [f32; 4]>,
) -> Result<[f32; 4], GpuArticulatedGroundContactError> {
    let key = Arc::as_ptr(mesh) as usize;
    if let Some(&offsets) = cache.get(&key) {
        return Ok(offsets);
    }
    let vertices = mesh
        .vertices()
        .iter()
        .map(|vertex| {
            Ok([
                finite_f32(vertex.x)?,
                finite_f32(vertex.y)?,
                finite_f32(vertex.z)?,
            ])
        })
        .collect::<Result<Vec<_>, GpuArticulatedGroundContactError>>()?;
    let vertex_start = geometry.len();
    for vertex in &vertices {
        geometry.push(PackedSphere {
            center_radius: [vertex[0], vertex[1], vertex[2], 0.0],
            ..PackedSphere::zeroed()
        });
    }
    let triangle_start = geometry.len();
    for &triangle in mesh.triangles() {
        geometry.push(PackedSphere {
            indices: [triangle[0], triangle[1], triangle[2], 0],
            ..PackedSphere::zeroed()
        });
    }
    let node_start = geometry.len();
    let nodes = mesh_bvh_nodes(&vertices, mesh.triangles());
    for node in &nodes {
        geometry.push(PackedSphere {
            indices: [
                node.triangle
                    .map(checked_u32)
                    .transpose()?
                    .unwrap_or(u32::MAX),
                checked_u32(node.escape)?,
                0,
                0,
            ],
            center_radius: [node.lower[0], node.lower[1], node.lower[2], 0.0],
            plane: [node.upper[0], node.upper[1], node.upper[2], 0.0],
            ..PackedSphere::zeroed()
        });
    }
    let offsets = [
        mesh_index_f32(vertex_start)?,
        mesh_index_f32(triangle_start)?,
        mesh_index_f32(node_start)?,
        mesh_index_f32(nodes.len())?,
    ];
    let _ = cache.insert(key, offsets);
    Ok(offsets)
}

fn pack_polyline_geometry(
    polyline: &Arc<PolylineGeometry>,
    geometry: &mut Vec<PackedSphere>,
    cache: &mut HashMap<usize, [f32; 4]>,
) -> Result<[f32; 4], GpuArticulatedGroundContactError> {
    let key = Arc::as_ptr(polyline) as usize;
    if let Some(&offsets) = cache.get(&key) {
        return Ok(offsets);
    }
    let vertices = polyline
        .vertices()
        .iter()
        .map(|vertex| {
            Ok([
                finite_f32(vertex.x)?,
                finite_f32(vertex.y)?,
                finite_f32(vertex.z)?,
            ])
        })
        .collect::<Result<Vec<_>, GpuArticulatedGroundContactError>>()?;
    let vertex_start = geometry.len();
    for vertex in &vertices {
        geometry.push(PackedSphere {
            center_radius: [vertex[0], vertex[1], vertex[2], 0.0],
            ..PackedSphere::zeroed()
        });
    }
    let segment_start = geometry.len();
    for &segment in polyline.segments() {
        if vertices[segment[0] as usize] == vertices[segment[1] as usize] {
            return Err(GpuArticulatedGroundContactError::InvalidInput);
        }
        geometry.push(PackedSphere {
            indices: [segment[0], segment[1], 0, 0],
            ..PackedSphere::zeroed()
        });
    }
    let node_start = geometry.len();
    // Degenerate triangles preserve segment bounds in the shared BVH builder.
    let degenerate_triangles = polyline
        .segments()
        .iter()
        .map(|&[a, b]| [a, b, b])
        .collect::<Vec<_>>();
    let nodes = mesh_bvh_nodes(&vertices, &degenerate_triangles);
    for node in &nodes {
        geometry.push(PackedSphere {
            indices: [
                node.triangle
                    .map(checked_u32)
                    .transpose()?
                    .unwrap_or(u32::MAX),
                checked_u32(node.escape)?,
                0,
                0,
            ],
            center_radius: [node.lower[0], node.lower[1], node.lower[2], 0.0],
            plane: [node.upper[0], node.upper[1], node.upper[2], 0.0],
            ..PackedSphere::zeroed()
        });
    }
    let offsets = [
        mesh_index_f32(vertex_start)?,
        mesh_index_f32(segment_start)?,
        mesh_index_f32(node_start)?,
        mesh_index_f32(nodes.len())?,
    ];
    let _ = cache.insert(key, offsets);
    Ok(offsets)
}

fn finite_f32(value: f64) -> Result<f32, GpuArticulatedGroundContactError> {
    let narrowed = value as f32;
    if !value.is_finite() || !narrowed.is_finite() || (value != 0.0 && narrowed == 0.0) {
        return Err(GpuArticulatedGroundContactError::InvalidInput);
    }
    Ok(narrowed)
}

fn valid_convex_geometry(geometry: &ConvexGeometry) -> bool {
    geometry.vertices.len() >= 4
        && geometry.face_normals.len() >= 4
        && geometry.edge_directions.len() >= 3
        && geometry
            .vertices
            .iter()
            .all(|point| point.iter().all(|value| value.is_finite()))
        && geometry
            .face_normals
            .iter()
            .chain(&geometry.edge_directions)
            .all(|direction| {
                direction.iter().all(|value| value.is_finite())
                    && (direction.norm_squared() - 1.0).abs() <= 1e-4
            })
}

fn pack_convex_geometry(
    output: &mut Vec<PackedSphere>,
    geometry: &ConvexGeometry,
) -> Result<f32, GpuArticulatedGroundContactError> {
    let index = output.len();
    if index > (1 << 24) {
        return Err(GpuArticulatedGroundContactError::Capacity);
    }
    output.push(PackedSphere {
        indices: [
            checked_u32(geometry.vertices.len())?,
            checked_u32(geometry.face_normals.len())?,
            checked_u32(geometry.edge_directions.len())?,
            0,
        ],
        ..PackedSphere::zeroed()
    });
    for point in geometry
        .vertices
        .iter()
        .chain(&geometry.face_normals)
        .chain(&geometry.edge_directions)
    {
        output.push(PackedSphere {
            center_radius: [
                finite_f32(point.x)?,
                finite_f32(point.y)?,
                finite_f32(point.z)?,
                0.0,
            ],
            ..PackedSphere::zeroed()
        });
    }
    Ok(index as f32)
}

#[cfg(test)]
mod activity_tests {
    use super::*;
    use crate::gpu_contact_pipeline::GpuContactDevice;

    #[test]
    fn load_wake_preserves_tiny_loads_excludes_gravity_and_is_environment_local() {
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let device = context.device();
            let queue = context.queue();
            let buffer = |label, bytes: &[u8]| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytes,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                })
            };
            let mut loads = [[0f32; 8]; 6];
            loads[0][3] = 2.0;
            loads[1][0] = 1e-8;
            loads[2][5] = -1e-10;
            loads[3][0] = -0.0;
            loads[3][4] = -0.0;
            loads[3][3] = -100.0;
            loads[4][2] = f32::from_bits(1);
            let load_buffer = buffer("load wake test loads", bytemuck::cast_slice(&loads));
            let status = buffer("load wake test status", bytemuck::cast_slice(&[0u32; 2]));
            let mass_status = buffer(
                "load wake test mass status",
                bytemuck::cast_slice(&[0u32; 2]),
            );
            let requests = buffer(
                "load wake test requests",
                bytemuck::cast_slice(&[0u32, 0, 0, 0, 0, 1]),
            );
            let wake = LoadWake::new(device, &load_buffer, &[0..3, 3..6]).unwrap();
            assert!(LoadWake::new(device, &load_buffer, core::slice::from_ref(&(0..2))).is_err());
            let submit = || {
                let mut encoder = device.create_command_encoder(&Default::default());
                wake.encode(device, &mut encoder, &status, &requests, &mass_status);
                let _ = queue.submit(Some(encoder.finish()));
            };
            let read = |buffer| {
                crate::gpu_articulated_mass::read_buffer(device, queue, buffer)
                    .unwrap()
                    .chunks_exact(4)
                    .map(|v| u32::from_ne_bytes(v.try_into().unwrap()))
                    .collect::<Vec<_>>()
            };
            submit();
            assert_eq!(read(&requests), [0, 1, 1, 0, 1, 1]);
            assert_eq!(read(&status), [0; 2]);
            queue.write_buffer(&requests, 0, bytemuck::cast_slice(&[0u32; 6]));
            submit();
            assert_eq!(read(&requests), [0, 1, 1, 0, 1, 0]);
            loads[1][0] = 0.0;
            loads[2][5] = 0.0;
            loads[4][2] = 0.0;
            queue.write_buffer(&load_buffer, 0, bytemuck::cast_slice(&loads));
            queue.write_buffer(&requests, 0, bytemuck::cast_slice(&[0u32; 6]));
            submit();
            assert_eq!(read(&requests), [0; 6]);
            loads[0][0] = f32::NAN;
            loads[4][0] = 1e-8;
            queue.write_buffer(&load_buffer, 0, bytemuck::cast_slice(&loads));
            submit();
            assert_eq!(read(&status), [1, 0]);
            assert_eq!(&read(&requests)[3..], &[0, 1, 0]);
            queue.write_buffer(&status, 0, bytemuck::cast_slice(&[8u32, 0]));
            queue.write_buffer(&requests, 0, bytemuck::cast_slice(&[0u32; 6]));
            submit();
            assert_eq!(read(&status), [8, 0]);
            assert_eq!(read(&requests), [0, 0, 0, 0, 1, 0]);
            queue.write_buffer(&status, 0, bytemuck::cast_slice(&[0u32; 2]));
            queue.write_buffer(&mass_status, 0, bytemuck::cast_slice(&[0u32, 2]));
            queue.write_buffer(&requests, 0, bytemuck::cast_slice(&[0u32; 6]));
            submit();
            assert_eq!(read(&status), [1, 1]);
            assert_eq!(&read(&requests)[3..], &[0; 3]);
            eprintln!("external load wake passed on {backend:?}");
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn motion_wake_uses_predicted_origin_twist_and_rejects_source_faults() {
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let device = context.device();
            let queue = context.queue();
            let buffer = |label, bytes: &[u8]| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytes,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                })
            };
            let mut links = vec![
                PackedMotionLink {
                    indices: [0, 0, 2, 0],
                    center_of_mass: [1.0, 0.0, 0.0, 1.0],
                    thresholds: [0.01, 0.3, 0.5, 1.0],
                };
                5
            ];
            links[1].thresholds[3] = 0.0;
            links[2].center_of_mass[3] = 0.0;
            links[4].thresholds[1] = 0.1;
            // World COM offset is Y after a 90-degree rotation. The rotational
            // linear Jacobian contributes -X, which must cancel at the origin.
            let mut terms = vec![0f32; 22];
            terms[10] = 1.0;
            terms[11] = -1.0;
            terms[21] = 1.0;
            let terms = buffer("motion test link terms", bytemuck::cast_slice(&terms));
            let half = core::f32::consts::FRAC_1_SQRT_2;
            let poses = buffer(
                "motion test poses",
                bytemuck::cast_slice(&[[0f32, 0.0, 0.0, 0.0, 0.0, 0.0, half, half]; 5]),
            );
            let velocities = buffer(
                "motion test velocities",
                bytemuck::cast_slice(&[0.002f32, 0.2]),
            );
            let accelerations = buffer(
                "motion test accelerations",
                bytemuck::cast_slice(&[0f32; 2]),
            );
            let status = buffer("motion test status", bytemuck::cast_slice(&[0u32]));
            let requests = buffer(
                "motion test requests",
                bytemuck::cast_slice(&[0u32, 0, 0, 1, 0]),
            );
            let mass_status = buffer("motion test mass status", bytemuck::cast_slice(&[0u32]));
            let motion = MotionWake::new(device, &links).unwrap();
            let submit = || {
                let mut encoder = device.create_command_encoder(&Default::default());
                motion.encode(
                    device,
                    &mut encoder,
                    [
                        &terms,
                        &poses,
                        &velocities,
                        &accelerations,
                        &status,
                        &requests,
                        &mass_status,
                    ],
                );
                let _ = queue.submit(Some(encoder.finish()));
            };
            let read = |buffer| {
                crate::gpu_articulated_mass::read_buffer(device, queue, buffer)
                    .unwrap()
                    .chunks_exact(4)
                    .map(|v| u32::from_ne_bytes(v.try_into().unwrap()))
                    .collect::<Vec<_>>()
            };
            submit();
            assert_eq!(read(&requests), [0, 0, 0, 1, 1]);
            assert_eq!(read(&status), [0]);
            queue.write_buffer(&requests, 0, bytemuck::cast_slice(&[0u32; 5]));
            queue.write_buffer(&accelerations, 0, bytemuck::cast_slice(&[0.04f32, 0.0]));
            submit();
            assert_eq!(read(&requests), [1, 0, 0, 1, 1]);
            assert_eq!(read(&status), [0]);
            for (velocity, source_status) in [(f32::NAN, 0u32), (0.002, 2)] {
                queue.write_buffer(&velocities, 0, bytemuck::cast_slice(&[velocity, 0.2]));
                queue.write_buffer(&requests, 0, bytemuck::cast_slice(&[0u32; 5]));
                queue.write_buffer(&status, 0, bytemuck::cast_slice(&[0u32]));
                queue.write_buffer(&mass_status, 0, bytemuck::cast_slice(&[source_status]));
                submit();
                assert_ne!(read(&status), [0]);
                assert_eq!(read(&requests), [0; 5]);
            }
            queue.write_buffer(&status, 0, bytemuck::cast_slice(&[8u32]));
            submit();
            assert_eq!(read(&status), [8]);
            assert_eq!(read(&requests), [0; 5]);
            // Distinct environment dimensions and offsets must remain independent.
            let mut multi_terms = vec![0f32; 38];
            multi_terms[10] = 1.0;
            multi_terms[32] = 1.0;
            let multi_terms = buffer("multi motion terms", bytemuck::cast_slice(&multi_terms));
            let multi_velocities = buffer(
                "multi motion velocities",
                bytemuck::cast_slice(&[0.002f32, 0.0, 0.1]),
            );
            let multi_accelerations = buffer(
                "multi motion accelerations",
                bytemuck::cast_slice(&[0f32; 3]),
            );
            let multi_status = buffer("multi motion status", bytemuck::cast_slice(&[0u32; 2]));
            let multi_mass = buffer("multi motion mass status", bytemuck::cast_slice(&[0u32; 2]));
            let multi_requests = buffer("multi motion requests", bytemuck::cast_slice(&[0u32; 2]));
            let multi_poses = buffer(
                "multi motion poses",
                bytemuck::cast_slice(&[[0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]; 2]),
            );
            let multi_links = [[0, 0, 2, 0], [1, 2, 1, 22]].map(|indices| PackedMotionLink {
                indices,
                center_of_mass: [0.0, 0.0, 0.0, 1.0],
                thresholds: [0.01, 0.3, 0.5, 1.0],
            });
            let multi_motion = MotionWake::new(device, &multi_links).unwrap();
            let multi_submit = || {
                let mut encoder = device.create_command_encoder(&Default::default());
                multi_motion.encode(
                    device,
                    &mut encoder,
                    [
                        &multi_terms,
                        &multi_poses,
                        &multi_velocities,
                        &multi_accelerations,
                        &multi_status,
                        &multi_requests,
                        &multi_mass,
                    ],
                );
                let _ = queue.submit(Some(encoder.finish()));
            };
            multi_submit();
            assert_eq!(read(&multi_requests), [0, 1]);
            queue.write_buffer(&multi_requests, 0, bytemuck::cast_slice(&[0u32; 2]));
            queue.write_buffer(
                &multi_accelerations,
                0,
                bytemuck::cast_slice(&[0f32, 0.0, -0.2]),
            );
            multi_submit();
            assert_eq!(read(&multi_requests), [0; 2]);
            assert_eq!(read(&multi_status), [0; 2]);
            eprintln!("predicted origin motion wake passed on {backend:?}");
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn mobility_seeds_validate_disjoint_local_groups_in_all_environments() {
        let ranges = [0..3, 3..5];
        let groups = vec![vec![vec![2, 0]], vec![vec![1, 0]]];
        assert_eq!(
            pack_mobility_groups(&ranges, &groups).unwrap(),
            [0, 1, 0, 3, 3]
        );
        let mut bad = groups.clone();
        bad[1][0].push(2);
        assert!(pack_mobility_groups(&ranges, &bad).is_err());
        bad = groups.clone();
        bad[1].push(vec![0]);
        assert!(pack_mobility_groups(&ranges, &bad).is_err());
        bad = groups.clone();
        bad[1][0] = vec![0, 0];
        assert!(pack_mobility_groups(&ranges, &bad).is_err());
        bad = groups.clone();
        bad[1][0].clear();
        assert!(pack_mobility_groups(&ranges, &bad).is_err());
        assert!(pack_mobility_groups(&ranges, &groups[..1]).is_err());
    }

    #[test]
    fn gravity_wake_is_one_shot_environment_local_and_ignores_signed_zero() {
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let device = context.device();
            let queue = context.queue();
            let links = (0..6)
                .map(|i| PackedMotionLink {
                    indices: [i / 3, 0, 0, 0],
                    center_of_mass: [0.0, 0.0, 0.0, if i % 3 == 1 { 0.0 } else { 1.0 }],
                    thresholds: [0.0; 4],
                })
                .collect::<Vec<_>>();
            let buffer = |label, bytes: &[u8]| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytes,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                })
            };
            let mut gravities = [[0.0f32; 4]; 2];
            let gravity = buffer(
                "gravity wake test gravity",
                bytemuck::cast_slice(&gravities),
            );
            let status = buffer("gravity wake test status", bytemuck::cast_slice(&[0u32; 2]));
            let mass_status = buffer(
                "gravity wake test mass status",
                bytemuck::cast_slice(&[0u32; 2]),
            );
            let requests = buffer(
                "gravity wake test requests",
                bytemuck::cast_slice(&[0u32; 6]),
            );
            assert!(GravityWake::new(device, queue, &gravity, &links, 1).is_err());
            let pass = GravityWake::new(device, queue, &gravity, &links, 2).unwrap();
            let step = || {
                let mut encoder = device.create_command_encoder(&Default::default());
                pass.encode(device, &mut encoder, &status, &mass_status, &requests);
                let _ = queue.submit(Some(encoder.finish()));
            };
            let read = |buffer| {
                crate::gpu_articulated_mass::read_buffer(device, queue, buffer)
                    .unwrap()
                    .chunks_exact(4)
                    .map(|v| u32::from_ne_bytes(v.try_into().unwrap()))
                    .collect::<Vec<_>>()
            };
            step();
            assert_eq!(read(&requests), [0; 6]);
            gravities[0][2] = -9.81;
            queue.write_buffer(&gravity, 0, bytemuck::cast_slice(&gravities));
            step();
            assert_eq!(read(&requests), [1, 0, 1, 0, 0, 0]);
            queue.write_buffer(&requests, 0, bytemuck::cast_slice(&[0u32; 6]));
            step();
            assert_eq!(read(&requests), [0; 6]);
            gravities[1][0] = f32::from_bits(1);
            gravities[1][1] = -0.0;
            queue.write_buffer(&gravity, 0, bytemuck::cast_slice(&gravities));
            step();
            assert_eq!(read(&requests), [0, 0, 0, 1, 0, 1]);
            gravities[1][1] = 0.0;
            queue.write_buffer(&gravity, 0, bytemuck::cast_slice(&gravities));
            queue.write_buffer(&requests, 0, bytemuck::cast_slice(&[0u32, 0, 1, 0, 0, 0]));
            step();
            assert_eq!(read(&requests), [0, 0, 1, 0, 0, 0]);
            queue.write_buffer(&requests, 0, bytemuck::cast_slice(&[0u32; 6]));
            queue.write_buffer(&mass_status, 0, bytemuck::cast_slice(&[1u32, 0]));
            step();
            assert_eq!(read(&status), [1, 0]);
            assert_eq!(read(&requests), [0; 6]);
            gravities[1][0] = f32::NAN;
            queue.write_buffer(&gravity, 0, bytemuck::cast_slice(&gravities));
            step();
            assert_eq!(read(&status), [1, 1]);
            assert_eq!(read(&requests), [0; 6]);
            gravities[0][2] = -1.0;
            gravities[1][0] = f32::from_bits(1);
            queue.write_buffer(&gravity, 0, bytemuck::cast_slice(&gravities));
            queue.write_buffer(&mass_status, 0, bytemuck::cast_slice(&[0u32; 2]));
            queue.write_buffer(&status, 0, bytemuck::cast_slice(&[8u32, 0]));
            step();
            assert_eq!(read(&requests), [0; 6]);
            queue.write_buffer(&status, 0, bytemuck::cast_slice(&[0u32; 2]));
            step();
            assert_eq!(read(&requests), [1, 0, 1, 0, 0, 0]);
            eprintln!("gravity change wake passed on {backend:?}");
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn actuation_wake_selects_jacobian_links_preserves_tiny_efforts_and_faults() {
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let device = context.device();
            let queue = context.queue();
            let links = [
                PackedMotionLink {
                    indices: [0, 0, 1, 0],
                    center_of_mass: [0.0, 0.0, 0.0, 1.0],
                    thresholds: [0.0; 4],
                },
                PackedMotionLink {
                    indices: [0, 0, 1, 16],
                    center_of_mass: [0.0; 4],
                    thresholds: [0.0; 4],
                },
                PackedMotionLink {
                    indices: [1, 1, 2, 32],
                    center_of_mass: [0.0, 0.0, 0.0, 1.0],
                    thresholds: [0.0; 4],
                },
                PackedMotionLink {
                    indices: [1, 1, 2, 54],
                    center_of_mass: [0.0, 0.0, 0.0, 1.0],
                    thresholds: [0.0; 4],
                },
            ];
            let mut terms = vec![0.0f32; 76];
            terms[10] = 1.0;
            terms[26] = 1.0; // Massless helper must not request wake.
            terms[48] = 1.0; // Angular Jacobian for the first env-1 coordinate.
            terms[65] = 1.0; // Linear Jacobian for the second env-1 coordinate.
            let buffer = |label, bytes: &[u8]| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytes,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                })
            };
            let jacobians = buffer("actuation test terms", bytemuck::cast_slice(&terms));
            let efforts = buffer(
                "actuation test efforts",
                bytemuck::cast_slice(&[1e-8f32, 0.0, f32::from_bits(1)]),
            );
            let status = buffer("actuation test status", bytemuck::cast_slice(&[0u32; 2]));
            let mass_status = buffer(
                "actuation test mass status",
                bytemuck::cast_slice(&[0u32; 2]),
            );
            let requests = buffer("actuation test requests", bytemuck::cast_slice(&[0u32; 4]));
            let pass = MotionWake::build(
                device,
                &links,
                include_str!("gpu_articulated_actuation_wake.wgsl"),
                "request_actuation_wake",
            )
            .unwrap();
            let step = || {
                let mut encoder = device.create_command_encoder(&Default::default());
                pass.encode(
                    device,
                    &mut encoder,
                    [&jacobians, &efforts, &status, &requests, &mass_status],
                );
                let _ = queue.submit(Some(encoder.finish()));
            };
            let read = |buffer| {
                crate::gpu_articulated_mass::read_buffer(device, queue, buffer)
                    .unwrap()
                    .chunks_exact(4)
                    .map(|v| u32::from_ne_bytes(v.try_into().unwrap()))
                    .collect::<Vec<_>>()
            };
            step();
            assert_eq!(read(&requests), [1, 0, 0, 1]);
            queue.write_buffer(&requests, 0, bytemuck::cast_slice(&[0u32; 4]));
            step();
            assert_eq!(read(&requests), [1, 0, 0, 1]);
            queue.write_buffer(&efforts, 0, bytemuck::cast_slice(&[0.0f32, -0.0, 0.0]));
            queue.write_buffer(&requests, 0, bytemuck::cast_slice(&[0u32; 4]));
            step();
            assert_eq!(read(&requests), [0; 4]);
            queue.write_buffer(&efforts, 0, bytemuck::cast_slice(&[1e-8f32, 1e-10, 0.0]));
            step();
            assert_eq!(read(&requests), [1, 0, 1, 0]);
            queue.write_buffer(&mass_status, 0, bytemuck::cast_slice(&[1u32, 0]));
            queue.write_buffer(&requests, 0, bytemuck::cast_slice(&[0u32; 4]));
            step();
            assert_eq!(read(&requests), [0, 0, 1, 0]);
            assert_eq!(read(&status), [1, 0]);
            queue.write_buffer(&mass_status, 0, bytemuck::cast_slice(&[0u32; 2]));
            queue.write_buffer(&status, 0, bytemuck::cast_slice(&[0u32; 2]));
            terms[48] = f32::NAN;
            queue.write_buffer(&jacobians, 0, bytemuck::cast_slice(&terms));
            queue.write_buffer(&requests, 0, bytemuck::cast_slice(&[0u32; 4]));
            step();
            assert_eq!(read(&requests), [1, 0, 0, 0]);
            assert_eq!(read(&status), [0, 1]);
            terms[48] = 1.0;
            queue.write_buffer(&jacobians, 0, bytemuck::cast_slice(&terms));
            queue.write_buffer(&efforts, 0, bytemuck::cast_slice(&[1e-8f32, f32::NAN, 0.0]));
            queue.write_buffer(&status, 0, bytemuck::cast_slice(&[0u32; 2]));
            queue.write_buffer(&requests, 0, bytemuck::cast_slice(&[0u32; 4]));
            step();
            assert_eq!(read(&status), [0, 1]);
            queue.write_buffer(&status, 0, bytemuck::cast_slice(&[8u32, 8]));
            queue.write_buffer(&requests, 0, bytemuck::cast_slice(&[0u32, 0, 1, 0]));
            step();
            assert_eq!(read(&requests), [0, 0, 1, 0]);
            assert_eq!(read(&status), [8, 8]);
            eprintln!("persistent actuation wake passed on {backend:?}");
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn component_idle_requires_quiet_contact_and_resets_on_merge_wake_and_disable() {
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let device = context.device();
            let queue = context.queue();
            let mut activity = ContactActivity::new(device, &[0..0, 0..0], &[0..5, 5..7]).unwrap();
            let source = |label| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytemuck::cast_slice(&[0u32; 2]),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                })
            };
            let state_status = source("idle test state status");
            let mass_status = source("idle test mass status");
            activity.idle_sources = Some([state_status.clone(), mass_status.clone()]);
            let mut parameters = vec![[0.01f32, 0.02, 1.0, 1.0]; 7];
            parameters[2][3] = 0.0;
            parameters[2][2] = 0.0;
            parameters[2][1] = 1.0;
            parameters[6][3] = 0.0;
            queue.write_buffer(
                &activity.idle_parameters,
                0,
                bytemuck::cast_slice(&parameters),
            );
            queue.write_buffer(
                &activity.parents,
                0,
                bytemuck::cast_slice(&[0u32, 0, 0, 3, 3, 5, 6]),
            );
            queue.write_buffer(
                &activity.flags,
                0,
                bytemuck::cast_slice(&[2u32, 0, 0, 2, 0, 0, 0]),
            );
            queue.write_buffer(
                &activity.link_wake,
                0,
                bytemuck::cast_slice(&[0u32, 1, 0, 0, 0, 0, 0]),
            );
            let step = || {
                let mut encoder = device.create_command_encoder(&Default::default());
                activity.encode_idle(device, &mut encoder);
                let _ = queue.submit(Some(encoder.finish()));
            };
            let read = |buffer| {
                crate::gpu_articulated_mass::read_buffer(device, queue, buffer)
                    .unwrap()
                    .chunks_exact(4)
                    .map(|v| u32::from_ne_bytes(v.try_into().unwrap()))
                    .collect::<Vec<_>>()
            };
            step();
            assert_eq!(read(&activity.sleep_candidates), [0; 7]);
            step();
            assert_eq!(read(&activity.sleep_candidates), [0, 0, 0, 1, 1, 0, 0]);
            queue.write_buffer(&activity.link_wake, 0, bytemuck::cast_slice(&[0u32; 7]));
            step();
            step();
            assert_eq!(read(&activity.sleep_candidates), [1, 1, 1, 1, 1, 0, 0]);
            // A newly active member joins the old sleeping component. Every
            // member must wait for the minimum previous idle history again.
            queue.write_buffer(
                &activity.parents,
                0,
                bytemuck::cast_slice(&[0u32, 0, 0, 0, 3, 5, 6]),
            );
            queue.write_buffer(&activity.idle_time, 4 * 4, bytemuck::cast_slice(&[0.0f32]));
            step();
            assert_eq!(read(&activity.sleep_candidates), [0; 7]);
            step();
            assert_eq!(read(&activity.sleep_candidates), [1, 1, 1, 1, 1, 0, 0]);
            // The detached component loses its only geometric contact.
            queue.write_buffer(
                &activity.parents,
                0,
                bytemuck::cast_slice(&[0u32, 0, 0, 3, 3, 5, 6]),
            );
            queue.write_buffer(
                &activity.flags,
                0,
                bytemuck::cast_slice(&[2u32, 0, 0, 0, 0, 0, 0]),
            );
            step();
            assert_eq!(read(&activity.sleep_candidates), [1, 1, 1, 0, 0, 0, 0]);
            parameters[1][2] = 0.0;
            queue.write_buffer(
                &activity.idle_parameters,
                0,
                bytemuck::cast_slice(&parameters),
            );
            step();
            assert_eq!(read(&activity.sleep_candidates), [0; 7]);
            assert_eq!(read(&activity.idle_time), [0; 7]);
            parameters[1][2] = 1.0;
            queue.write_buffer(
                &activity.idle_parameters,
                0,
                bytemuck::cast_slice(&parameters),
            );
            queue.write_buffer(
                &activity.flags,
                0,
                bytemuck::cast_slice(&[2u32, 0, 0, 2, 0, 0, 0]),
            );
            step();
            step();
            assert_eq!(read(&activity.sleep_candidates), [1, 1, 1, 1, 1, 0, 0]);
            queue.write_buffer(
                &activity.idle_policy,
                0,
                bytemuck::cast_slice(&[8u32, 0, 0, 0]),
            );
            step();
            assert_eq!(read(&activity.sleep_candidates), [0; 7]);
            queue.write_buffer(
                &activity.flags,
                0,
                bytemuck::cast_slice(&[10u32, 0, 0, 2, 0, 0, 0]),
            );
            step();
            step();
            assert_eq!(read(&activity.sleep_candidates), [1, 1, 1, 0, 0, 0, 0]);
            queue.write_buffer(
                &activity.idle_policy,
                0,
                bytemuck::cast_slice(&[24u32, 0, 0, 0]),
            );
            queue.write_buffer(
                &activity.flags,
                0,
                bytemuck::cast_slice(&[10u32, 0, 0, 16, 0, 0, 0]),
            );
            step();
            step();
            assert_eq!(read(&activity.sleep_candidates), [1, 1, 1, 1, 1, 0, 0]);
            queue.write_buffer(
                &activity.idle_policy,
                0,
                bytemuck::cast_slice(&[40u32, 0, 0, 0]),
            );
            step();
            assert_eq!(read(&activity.sleep_candidates), [1, 1, 1, 0, 0, 0, 0]);
            queue.write_buffer(
                &activity.flags,
                0,
                bytemuck::cast_slice(&[10u32, 0, 0, 48, 0, 0, 0]),
            );
            step();
            step();
            assert_eq!(read(&activity.sleep_candidates), [1, 1, 1, 1, 1, 0, 0]);
            queue.write_buffer(
                &activity.flags,
                0,
                bytemuck::cast_slice(&[2u32, 0, 0, 2, 0, 0, 0]),
            );
            queue.write_buffer(
                &activity.idle_policy,
                0,
                bytemuck::cast_slice(&[2u32, 0, 0, 0]),
            );
            queue.write_buffer(&mass_status, 0, bytemuck::cast_slice(&[1u32, 0]));
            step();
            assert_eq!(read(&activity.sleep_candidates), [0; 7]);
            queue.write_buffer(&mass_status, 0, bytemuck::cast_slice(&[0u32; 2]));
            queue.write_buffer(&activity.link_wake, 0, bytemuck::cast_slice(&[0u32; 7]));
            step();
            step();
            assert_eq!(read(&activity.sleep_candidates), [1, 1, 1, 1, 1, 0, 0]);
            queue.write_buffer(&state_status, 0, bytemuck::cast_slice(&[8u32, 0]));
            step();
            assert_eq!(read(&activity.sleep_candidates), [0; 7]);
            assert_eq!(read(&activity.idle_time), [0; 7]);
            activity.clear_wake(queue);
            assert_eq!(read(&activity.sleep_candidates), [0; 7]);
            queue.write_buffer(&state_status, 0, bytemuck::cast_slice(&[0u32; 2]));
            queue.write_buffer(&activity.link_wake, 0, bytemuck::cast_slice(&[0u32; 7]));
            parameters[1][1] = 0.04;
            queue.write_buffer(
                &activity.idle_parameters,
                0,
                bytemuck::cast_slice(&parameters),
            );
            step();
            step();
            // A short-duration member must not sleep before its slower island peer.
            assert_eq!(read(&activity.sleep_candidates), [0, 0, 0, 1, 1, 0, 0]);
            step();
            assert_eq!(read(&activity.sleep_candidates), [0, 0, 0, 1, 1, 0, 0]);
            step();
            assert_eq!(read(&activity.sleep_candidates), [1, 1, 1, 1, 1, 0, 0]);
            // Merge an already-idle island with a newly quiet island. All members
            // must use the shared minimum history and the longest waiting period.
            queue.write_buffer(
                &activity.parents,
                0,
                bytemuck::cast_slice(&[0u32, 0, 0, 0, 0, 5, 6]),
            );
            queue.write_buffer(
                &activity.idle_time,
                3 * 4,
                bytemuck::cast_slice(&[0.0f32; 2]),
            );
            step();
            assert_eq!(read(&activity.sleep_candidates), [0; 7]);
            step();
            assert_eq!(read(&activity.sleep_candidates), [0; 7]);
            step();
            assert_eq!(read(&activity.sleep_candidates), [0; 7]);
            step();
            assert_eq!(read(&activity.sleep_candidates), [1, 1, 1, 1, 1, 0, 0]);
            eprintln!("component idle tracking passed on {backend:?}");
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn contact_activity_reduces_owners_clears_stale_flags_and_propagates_faults() {
        let contact = |first, second, signs: [f32; 2]| PackedSphere {
            indices: [first, 0, second, 0],
            impulses: [1.0, 0.2, 0.0, 0.0],
            previous_normal: [0.0, 0.0, 1.0, 0.0],
            diagnostic_first_origin: [0.0, 0.0, 0.0, signs[0]],
            diagnostic_second_origin: [0.0, 0.0, 0.0, signs[1]],
            prescribed_linear: [0.0; 4],
            prescribed_angular: [0.0; 4],
            ..PackedSphere::zeroed()
        };
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let device = context.device();
            let queue = context.queue();
            // Cross-workgroup duplicates exercise atomic aggregation. The two
            // environments use different global link offsets and row offsets.
            let mut rows = vec![contact(1, 2, [1.0, -1.0]); 65];
            rows.push(contact(99, 3, [0.0, 1.0])); // one-sided dynamic owner
            rows.push(contact(4, 99, [1.0, 0.0]));
            rows.push(contact(3, 4, [0.0, 0.0])); // bilateral impulse
            rows[67].material[3] = 79.0;
            let mut inactive = contact(3, 4, [0.0, 0.0]);
            inactive.impulses = [0.0; 4];
            rows.push(inactive);
            let activity = ContactActivity::new(device, &[0..65, 65..69], &[0..3, 3..5]).unwrap();
            let buffer = |label, bytes: &[u8]| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytes,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                })
            };
            let row_buffer = buffer("activity test rows", bytemuck::cast_slice(&rows));
            let status = buffer("activity test status", bytemuck::cast_slice(&[0u32; 2]));
            let submit = || {
                let mut encoder = device.create_command_encoder(&Default::default());
                activity.encode(device, &mut encoder, &row_buffer, &status);
                let _ = queue.submit(Some(encoder.finish()));
            };
            let read = |buffer| {
                crate::gpu_articulated_mass::read_buffer(device, queue, buffer)
                    .unwrap()
                    .chunks_exact(4)
                    .map(|bytes| u32::from_ne_bytes(bytes.try_into().unwrap()))
                    .collect::<Vec<_>>()
            };
            queue.write_buffer(
                &activity.wake_requests,
                0,
                bytemuck::cast_slice(&[0u32, 0, 1, 0, 0]),
            );
            submit();
            assert_eq!(read(&activity.flags), [0, 3, 3, 7, 7]);
            queue.write_buffer(
                &activity.support_gravity,
                0,
                bytemuck::cast_slice(&[[0.0f32, 0.0, -9.81, 0.0]; 2]),
            );
            submit();
            assert_eq!(read(&activity.flags), [0, 3, 3, 15, 15]);
            queue.write_buffer(
                &activity.support_gravity,
                0,
                bytemuck::cast_slice(&[[0.0f32, 0.0, 9.81, 0.0]; 2]),
            );
            submit();
            assert_eq!(read(&activity.flags), [0, 3, 3, 7, 7]);
            rows[65].diagnostic_second_origin[3] = -1.0;
            rows[66].previous_normal = [1.0, 0.0, 0.0, 0.0];
            queue.write_buffer(&row_buffer, 0, bytemuck::cast_slice(&rows));
            submit();
            assert_eq!(read(&activity.flags), [0, 3, 3, 15, 7]);
            queue.write_buffer(
                &activity.support_gravity,
                0,
                bytemuck::cast_slice(&[[0.0f32, 9.81, 0.0, 0.0]; 2]),
            );
            submit();
            assert_eq!(read(&activity.flags), [0, 3, 3, 7, 7]);
            rows[65].diagnostic_second_origin[3] = 1.0;
            queue.write_buffer(&row_buffer, 0, bytemuck::cast_slice(&rows));
            queue.write_buffer(
                &activity.support_gravity,
                0,
                bytemuck::cast_slice(&[[0.0f32, 0.0, -f32::MAX, 0.0]; 2]),
            );
            submit();
            assert_eq!(read(&activity.flags), [0, 3, 3, 15, 7]);
            queue.write_buffer(
                &activity.support_gravity,
                0,
                bytemuck::cast_slice(&[[0.0f32; 4], [f32::NAN, 0.0, 0.0, 0.0]]),
            );
            submit();
            assert_eq!(read(&status), [0, 1]);
            assert_eq!(read(&activity.flags), [0, 3, 3, 0, 0]);
            rows[66].previous_normal = [0.0, 0.0, 1.0, 0.0];
            queue.write_buffer(&row_buffer, 0, bytemuck::cast_slice(&rows));
            queue.write_buffer(&status, 0, bytemuck::cast_slice(&[0u32; 2]));
            queue.write_buffer(
                &activity.support_gravity,
                0,
                bytemuck::cast_slice(&[[0.0f32; 4]; 2]),
            );
            queue.write_buffer(
                &activity.wake_requests,
                0,
                bytemuck::cast_slice(&[0u32, 0, 1, 0, 0]),
            );
            submit();
            assert_eq!(read(&status), [0, 0]);
            assert_eq!(read(&activity.parents), [0, 1, 1, 3, 3]);
            assert_eq!(read(&activity.link_wake), [0, 1, 1, 0, 0]);
            assert_eq!(read(&activity.wake_requests), [0; 5]);
            queue.write_buffer(
                &activity.wake_requests,
                0,
                bytemuck::cast_slice(&[1u32, 0, 0, 0, 1]),
            );
            submit();
            assert_eq!(read(&activity.link_wake), [1, 0, 0, 1, 1]);
            // Geometric diagnostics remain but the next solve has no impulse.
            for row in &mut rows {
                row.impulses = [0.0; 4];
            }
            queue.write_buffer(&row_buffer, 0, bytemuck::cast_slice(&rows));
            submit();
            assert_eq!(read(&activity.flags), [0, 2, 2, 6, 6]);
            assert_eq!(read(&activity.parents), [0, 1, 1, 3, 3]);
            assert_eq!(read(&activity.link_wake), [0; 5]);
            queue.write_buffer(
                &activity.loss_params,
                0,
                bytemuck::cast_slice(&[1u32, 0, 0, 0]),
            );
            for row in &mut rows {
                row.diagnostic_first_origin = [0.0; 4];
                row.diagnostic_second_origin = [0.0; 4];
            }
            queue.write_buffer(&row_buffer, 0, bytemuck::cast_slice(&rows));
            submit();
            assert_eq!(read(&activity.flags), [0; 5]);
            assert_eq!(read(&activity.parents), [0, 1, 2, 3, 3]);
            assert_eq!(read(&activity.link_wake), [0, 1, 1, 1, 1]);
            submit();
            assert_eq!(read(&activity.link_wake), [0; 5]);
            queue.write_buffer(&activity.loss_params, 0, bytemuck::cast_slice(&[0u32; 4]));
            rows[67].indices[2] = u32::MAX;
            for mode in [79.0, 80.0] {
                rows[67].material[3] = mode;
                queue.write_buffer(&row_buffer, 0, bytemuck::cast_slice(&rows));
                submit();
                assert_eq!(
                    read(&activity.flags),
                    [0, 0, 0, if mode == 80.0 { 48 } else { 16 }, 0]
                );
                assert_eq!(read(&activity.parents), [0, 1, 2, 3, 4]);
            }
            rows[67].indices[2] = 4;
            rows[67].material[3] = 79.0;
            rows[65] = contact(99, 3, [0.0, 1.0]);
            let mut invalid_rows = Vec::new();
            let mut invalid = rows[65];
            invalid.indices[2] = 2; // owner belongs to the other environment
            invalid_rows.push(invalid);
            invalid = rows[65];
            invalid.diagnostic_second_origin[3] = 0.5;
            invalid_rows.push(invalid);
            invalid = rows[65];
            invalid.impulses[1] = f32::NAN;
            invalid_rows.push(invalid);
            invalid = rows[65];
            invalid.previous_normal[2] = 2.0;
            invalid_rows.push(invalid);
            invalid = rows[65];
            invalid.diagnostic_second[0] = f32::INFINITY;
            invalid_rows.push(invalid);
            for invalid in invalid_rows {
                rows[65] = invalid;
                queue.write_buffer(&row_buffer, 0, bytemuck::cast_slice(&rows));
                queue.write_buffer(&status, 0, bytemuck::cast_slice(&[0u32; 2]));
                submit();
                assert_eq!(read(&status), [0, 1]);
                assert_eq!(read(&activity.flags), [0; 5]);
            }
            // A pre-existing source fault prevents otherwise valid rows publishing.
            rows[65] = contact(99, 3, [0.0, 1.0]);
            queue.write_buffer(&row_buffer, 0, bytemuck::cast_slice(&rows));
            queue.write_buffer(&status, 0, bytemuck::cast_slice(&[0u32, 8]));
            submit();
            assert_eq!(read(&activity.flags), [0; 5]);
            assert_eq!(read(&status), [0, 8]);
            // Seeded articulation mobility merges with transient contact edges;
            // merging does not invent contact activity on the other group links.
            let seeds =
                pack_mobility_groups(&[0..3, 3..5], &[vec![vec![0, 1]], Vec::new()]).unwrap();
            queue.write_buffer(&activity.mobility_roots, 0, bytemuck::cast_slice(&seeds));
            rows[0] = contact(1, 2, [1.0, -1.0]);
            queue.write_buffer(
                &activity.wake_requests,
                0,
                bytemuck::cast_slice(&[0u32, 0, 1, 0, 0]),
            );
            queue.write_buffer(&row_buffer, 0, bytemuck::cast_slice(&rows));
            queue.write_buffer(&status, 0, bytemuck::cast_slice(&[0u32; 2]));
            submit();
            assert_eq!(read(&activity.parents), [0, 0, 0, 3, 3]);
            assert_eq!(read(&activity.flags), [0, 3, 3, 7, 0]);
            assert_eq!(read(&activity.link_wake), [1, 1, 1, 0, 0]);
            rows[0].impulses = [0.0; 4];
            rows[0].diagnostic_first[3] = 0.1;
            rows[0].diagnostic_second[3] = 0.1;
            queue.write_buffer(&row_buffer, 0, bytemuck::cast_slice(&rows));
            submit();
            assert_eq!(read(&activity.parents), [0, 0, 2, 3, 3]);
            assert_eq!(read(&activity.flags), [0, 0, 0, 7, 0]);
            assert_eq!(read(&activity.link_wake), [0; 5]);
            // Reverse-order chains, duplicate edges, and multiple workgroups
            // exercise concurrent unions where an observed root has changed.
            let chain = (0..128)
                .map(|i| {
                    let first = 63 - i % 64;
                    contact(first, first + 1, [1.0, -1.0])
                })
                .collect::<Vec<_>>();
            let chain_rows = buffer("activity chain rows", bytemuck::cast_slice(&chain));
            let components = ContactActivity::new(
                device,
                core::slice::from_ref(&(0..128)),
                core::slice::from_ref(&(0..65)),
            )
            .unwrap();
            queue.write_buffer(&components.wake_requests, 64 * 4, bytemuck::bytes_of(&1u32));
            for iteration in 0..5 {
                queue.write_buffer(&status, 0, bytemuck::cast_slice(&[0u32; 2]));
                let mut encoder = device.create_command_encoder(&Default::default());
                components.encode(device, &mut encoder, &chain_rows, &status);
                let _ = queue.submit(Some(encoder.finish()));
                let parents = read(&components.parents);
                for link in 0..65 {
                    let mut root = link;
                    while parents[root] as usize != root {
                        assert!((parents[root] as usize) < root);
                        root = parents[root] as usize;
                    }
                    assert_eq!(root, 0);
                }
                assert_eq!(read(&components.flags), [3; 65]);
                assert_eq!(read(&status), [0; 2]);
                assert_eq!(read(&components.link_wake), [u32::from(iteration == 0); 65]);
            }
            // Environments with no contact rows still propagate through mobility.
            let isolated = ContactActivity::new(
                device,
                core::slice::from_ref(&(0..0)),
                core::slice::from_ref(&(0..3)),
            )
            .unwrap();
            queue.write_buffer(
                &isolated.mobility_roots,
                0,
                bytemuck::cast_slice(&[0u32, 1, 1]),
            );
            queue.write_buffer(
                &isolated.wake_requests,
                0,
                bytemuck::cast_slice(&[0u32, 0, 1]),
            );
            let mut encoder = device.create_command_encoder(&Default::default());
            isolated.encode(device, &mut encoder, &row_buffer, &status);
            let _ = queue.submit(Some(encoder.finish()));
            assert_eq!(read(&isolated.link_wake), [0, 1, 1]);
            assert_eq!(read(&isolated.flags), [0; 3]);
            // Only a supporting normal impulse permits positive sub-ULP gaps.
            rows.fill(PackedSphere::zeroed());
            queue.write_buffer(&status, 0, bytemuck::cast_slice(&[0u32; 2]));
            queue.write_buffer(
                &activity.support_gravity,
                0,
                bytemuck::cast_slice(&[[0.0f32, 0.0, -9.81, 0.0]; 2]),
            );
            for (gap, impulse, flags) in [
                (1e-8, 1.0, 15u32),
                (1e-8, 0.0, 0),
                (1e-4, 1.0, 1),
                (0.0, 0.0, 14),
            ] {
                rows[65] = contact(99, 3, [0.0, 1.0]);
                rows[65].impulses = [impulse, 0.0, 0.0, 0.0];
                rows[65].diagnostic_second = [0.0, 0.0, 1.5, gap];
                rows[65].diagnostic_second_origin = [0.0, 0.0, 1.6, 1.0];
                queue.write_buffer(&row_buffer, 0, bytemuck::cast_slice(&rows));
                submit();
                assert_eq!(read(&activity.flags), [0, 0, 0, flags, 0]);
                assert_eq!(read(&status), [0, 0]);
            }
            eprintln!("contact activity reduction passed on {backend:?}");
        }
        assert!(tested > 0, "no GPU backend available");
    }
}
