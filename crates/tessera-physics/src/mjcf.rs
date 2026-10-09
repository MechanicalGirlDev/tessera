//! MJCF loader for Tessera articulated worlds.
//!
//! The loader consumes XML text supplied by the caller and does not resolve
//! repository-relative paths. It supports the rigid-body subset needed to build
//! an [`ArticulatedWorld`]: nested bodies, explicit or primitive-derived inertials,
//! default classes, primitive geoms, and fixed, hinge, slide, ball, or root free joints.

use core::ops::Range;
use std::collections::BTreeMap;

use nalgebra::{
    DVector, Isometry3, Matrix3, Point3, Quaternion, Translation3, Unit, UnitQuaternion, Vector3,
};
use quick_xml::Reader;
use quick_xml::events::{BytesStart, Event};

use crate::articulated_world::{
    ArticulatedWorld, ArticulatedWorldError, ArticulatedWorldParams, JointNonlinearPassive,
    JointPassive, JointPolynomialCoupling, LinkBox, LinkConvex, LinkCylinder, LinkFixedConstraint,
    LinkPointConstraint, LinkSphere, SceneBody, SceneCollider,
};
use crate::articulation::{Articulation, ArticulationError, JointKind, JointSpec, LinkSpec};
use crate::convex::ConvexGeometry;
use crate::mesh::ConvexMeshResolver;

#[path = "mjcf_actuator.rs"]
mod actuator;
pub use actuator::{MjcfActuatorInfo, MjcfActuatorKind, MjcfActuators};

/// Mesh resolver accepted by [`load_mjcf_str_with_mesh_resolver`].
pub use crate::mesh::ConvexMeshResolver as MjcfMeshResolver;

/// Options applied while constructing an articulated world from MJCF.
#[derive(Debug, Clone)]
pub struct MjcfLoadOptions {
    /// Additional world transform applied before each top-level MJCF body pose.
    pub root_pose: Isometry3<f64>,
    /// Tessera integration parameters. An MJCF `option.gravity` overrides gravity.
    pub world: ArticulatedWorldParams,
}

impl Default for MjcfLoadOptions {
    fn default() -> Self {
        Self {
            root_pose: Isometry3::identity(),
            world: ArticulatedWorldParams::default(),
        }
    }
}

/// Stable metadata for one loaded MJCF joint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MjcfJointInfo {
    /// Joint name, explicit or generated.
    pub name: String,
    /// Index of the child link in [`LoadedMjcf::link_names`].
    pub child_link: usize,
    /// Tessera joint kind.
    pub kind: JointKind,
    /// Generalized-coordinate slots owned by the joint.
    pub dofs: Range<usize>,
}

/// A loaded MJCF model and its ready-to-step Tessera world.
#[derive(Debug)]
pub struct LoadedMjcf {
    /// Optional name from the root `mujoco` element.
    pub model_name: Option<String>,
    /// Link names in articulation order. A synthetic root prepends `__mjcf_world`.
    pub link_names: Vec<String>,
    /// Joint metadata in articulation edge order. A multi-root `freejoint`
    /// expands into named `tx`, `ty`, `tz`, and `rotation` joints.
    pub joints: Vec<MjcfJointInfo>,
    /// Imported actuators and controls in XML declaration order.
    pub actuators: MjcfActuators,
    /// Constructed fixed-root or floating-root physics world.
    pub world: ArticulatedWorld,
}

/// MJCF parsing or model-conversion failure.
#[derive(Debug, thiserror::Error)]
pub enum MjcfLoadError {
    /// Malformed XML or invalid UTF-8 in an attribute.
    #[error("invalid MJCF XML: {0}")]
    Xml(String),
    /// A recognized MJCF construct cannot be represented by this loader yet.
    #[error("unsupported MJCF construct: {0}")]
    Unsupported(String),
    /// An attribute or model relationship is invalid.
    #[error("invalid MJCF model: {0}")]
    Invalid(String),
    /// An injected mesh resolver failed.
    #[error("failed to resolve MJCF mesh `{filename}`: {message}")]
    Mesh {
        /// URI exactly as written in the MJCF document.
        filename: String,
        /// Resolver-provided diagnostic.
        message: String,
    },
    /// The converted articulation is invalid.
    #[error("invalid MJCF articulation: {0}")]
    Articulation(#[from] ArticulationError),
    /// The converted Tessera world is invalid.
    #[error("invalid MJCF world: {0}")]
    World(#[from] ArticulatedWorldError),
}

#[derive(Debug, Clone, Copy)]
enum AngleUnit {
    Degree,
    Radian,
}

impl AngleUnit {
    fn value(self, angle: f64) -> f64 {
        match self {
            Self::Degree => angle.to_radians(),
            Self::Radian => angle,
        }
    }
}

#[derive(Debug, Clone)]
struct ParsedInertial {
    mass: f64,
    center: Vector3<f64>,
    inertia: Matrix3<f64>,
}

#[derive(Debug, Clone)]
struct ParsedJoint {
    name: String,
    kind: ParsedJointKind,
    position: Vector3<f64>,
    axis: Vector3<f64>,
    limits: Option<(f64, f64)>,
    armature: f64,
    reference: f64,
    passive: JointPassive,
    nonlinear_passive: JointNonlinearPassive,
    frictionloss: f64,
    springdamper: Option<(f64, f64)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParsedJointKind {
    Revolute,
    Prismatic,
    Spherical,
    Free,
}

#[derive(Debug, Clone)]
struct ParsedGeom {
    pose: Isometry3<f64>,
    kind: ParsedGeomKind,
    mass: Option<f64>,
    density: f64,
}

#[derive(Debug, Clone)]
enum ParsedGeomKind {
    Sphere { radius: f64 },
    Box { half_extents: Vector3<f64> },
    Capsule { radius: f64, half_height: f64 },
    Cylinder { radius: f64, half_height: f64 },
    Convex { geometries: Vec<ConvexGeometry> },
    Plane,
}

#[derive(Debug, Clone)]
struct ParsedBody {
    name: String,
    pose: Isometry3<f64>,
    inertial: Option<ParsedInertial>,
    joints: Vec<ParsedJoint>,
    geoms: Vec<ParsedGeom>,
    sites: Vec<ParsedSite>,
    children: Vec<ParsedBody>,
    child_class: Option<String>,
}

#[derive(Debug, Clone)]
struct ParsedSite {
    name: String,
    pose: Isometry3<f64>,
}

#[derive(Debug, Clone, Default)]
struct DefaultClass {
    geom: BTreeMap<String, String>,
    joint: BTreeMap<String, String>,
    actuators: BTreeMap<String, BTreeMap<String, String>>,
    actuator_parent: Option<String>,
}

#[derive(Debug, Clone, Copy)]
enum DefaultElement {
    Geom,
    Joint,
    Actuator(&'static str),
}

#[derive(Debug)]
struct ParsedModel {
    name: Option<String>,
    gravity: Option<[f64; 3]>,
    roots: Vec<ParsedBody>,
    world_geoms: Vec<ParsedGeom>,
    world_sites: Vec<ParsedSite>,
    equalities: Vec<ParsedEquality>,
}

#[derive(Debug)]
enum ParsedEquality {
    Joint(ParsedEqualityJoint),
    Connect(ParsedEqualityConnect),
    Weld(ParsedEqualityWeld),
}

#[derive(Debug)]
enum EqualityTarget {
    Bodies {
        first: String,
        second: Option<String>,
    },
    Sites {
        first: String,
        second: String,
    },
}

#[derive(Debug)]
struct ParsedEqualityConnect {
    target: EqualityTarget,
    anchor: Option<Vector3<f64>>,
}

#[derive(Debug)]
struct ParsedEqualityWeld {
    target: EqualityTarget,
    anchor: Vector3<f64>,
    relpose: Option<Isometry3<f64>>,
    torquescale: f64,
}

#[derive(Debug, Clone, Copy)]
struct FrameRef {
    link: usize,
    frame: Isometry3<f64>,
}

#[derive(Debug, Clone, Copy)]
enum SiteFrame {
    Link(FrameRef),
    World(Isometry3<f64>),
}

struct EqualityFrames {
    first: FrameRef,
    second: Option<FrameRef>,
    world_target: Option<Isometry3<f64>>,
}

#[derive(Debug)]
struct ParsedEqualityJoint {
    follower_name: String,
    source_name: Option<String>,
    coefficients: [f64; 5],
}

#[derive(Default)]
struct CompileState {
    links: Vec<LinkSpec>,
    link_names: Vec<String>,
    body_refs: BTreeMap<String, Vec<FrameRef>>,
    site_refs: BTreeMap<String, Vec<FrameRef>>,
    joints: Vec<JointSpec>,
    joint_armatures: Vec<f64>,
    joint_references: Vec<f64>,
    joint_passives: Vec<JointPassive>,
    joint_nonlinear_passives: Vec<JointNonlinearPassive>,
    joint_frictions: Vec<f64>,
    joint_springdampers: Vec<Option<(f64, f64)>>,
    joint_info: Vec<MjcfJointInfo>,
    spheres: Vec<LinkSphere>,
    boxes: Vec<LinkBox>,
    cylinders: Vec<LinkCylinder>,
    convex_shapes: Vec<LinkConvex>,
    collision_exclusions: Vec<(usize, usize)>,
    dof: usize,
}

/// Parse one self-contained MJCF document and construct a Tessera articulated world.
///
/// A single root `freejoint` selects a floating root. Multiple top-level bodies
/// and top-level movable joints share a fixed, massless world frame; each
/// `freejoint` in that frame is represented by three translations and a spherical
/// joint, occupying six generalized coordinates.
/// Primitive geom mass properties use MJCF's
/// `mass`-over-`density` precedence and default density of 1000 kg/m³. Inline
/// triangular convex mesh assets and serial joints on a non-root body are accepted.
/// External mesh assets require an injected resolver. Non-`xyz` compiler Euler
/// sequences return explicit errors instead of being approximated silently. Capsules
/// are represented exactly as the union of one cylinder and two endpoint spheres.
pub fn load_mjcf_str(xml: &str, options: MjcfLoadOptions) -> Result<LoadedMjcf, MjcfLoadError> {
    load_mjcf_inner(xml, options, None)
}

/// Parse MJCF and resolve every external mesh asset through `resolver`.
pub fn load_mjcf_str_with_mesh_resolver(
    xml: &str,
    options: MjcfLoadOptions,
    resolver: &mut dyn ConvexMeshResolver,
) -> Result<LoadedMjcf, MjcfLoadError> {
    load_mjcf_inner(xml, options, Some(resolver))
}

fn load_mjcf_inner(
    xml: &str,
    options: MjcfLoadOptions,
    resolver: Option<&mut dyn ConvexMeshResolver>,
) -> Result<LoadedMjcf, MjcfLoadError> {
    let parsed = parse_document(xml, resolver)?;
    if parsed.roots.is_empty() {
        return Err(MjcfLoadError::Unsupported(format!(
            "expected at least one top-level body, found {}",
            parsed.roots.len()
        )));
    }
    let mut state = CompileState::default();
    let direct_root = parsed.roots.len() == 1
        && parsed.roots[0].joints.len() <= 1
        && parsed.roots[0]
            .joints
            .first()
            .is_none_or(|joint| joint.kind == ParsedJointKind::Free);
    let (floating, root_pose) = if direct_root {
        let root = &parsed.roots[0];
        let floating = root
            .joints
            .first()
            .is_some_and(|joint| joint.kind == ParsedJointKind::Free);
        compile_root(root, &mut state)?;
        (floating, options.root_pose * root.pose)
    } else {
        state.links.push(empty_link_spec());
        state.link_names.push("__mjcf_world".into());
        for root in &parsed.roots {
            if root
                .joints
                .iter()
                .any(|joint| joint.kind == ParsedJointKind::Free)
            {
                compile_free_root(root, &mut state)?;
            } else {
                compile_child(root, 0, Isometry3::identity(), &mut state)?;
            }
        }
        (false, options.root_pose)
    };
    let mut articulation = Articulation::new(state.links, state.joints, 0)?;
    for (edge, armature) in state.joint_armatures.iter().enumerate() {
        let width = joint_width(state.joint_info[edge].kind);
        articulation.set_joint_armature(edge, &vec![*armature; width])?;
    }
    for (a, b) in state.collision_exclusions {
        articulation.exclude_collision_pair(a, b)?;
    }
    let mut params = options.world;
    if let Some(gravity) = parsed.gravity {
        params.gravity = gravity;
    }
    let mut world = if floating {
        ArticulatedWorld::new_floating(articulation, root_pose, state.spheres, params)?
    } else {
        ArticulatedWorld::new(articulation, root_pose, state.spheres, params)?
    };
    for (edge, reference) in state.joint_references.iter().copied().enumerate() {
        for slot in state.joint_info[edge].dofs.clone() {
            world.positions[slot] = reference;
        }
    }
    for (edge, passive) in state.joint_passives.iter().copied().enumerate() {
        for slot in state.joint_info[edge].dofs.clone() {
            world.set_joint_passive(slot, passive)?;
            world.set_joint_nonlinear_passive(slot, state.joint_nonlinear_passives[edge])?;
            world.set_joint_friction(slot, state.joint_frictions[edge])?;
        }
    }
    let equalities = parsed
        .equalities
        .iter()
        .filter_map(|equality| match equality {
            ParsedEquality::Joint(joint) => Some(joint),
            _ => None,
        })
        .map(|equality| {
            let follower = equality_slot(&state.joint_info, &equality.follower_name)?;
            let source = equality
                .source_name
                .as_deref()
                .map(|name| equality_slot(&state.joint_info, name))
                .transpose()?;
            Ok(JointPolynomialCoupling {
                follower,
                source,
                coefficients: equality.coefficients,
                follower_reference: world.positions[follower],
                source_reference: source.map_or(0.0, |slot| world.positions[slot]),
            })
        })
        .collect::<Result<Vec<_>, MjcfLoadError>>()?;
    world.set_joint_polynomial_couplings(equalities)?;
    let reference_pose = world
        .articulation
        .pose(world.root_pose, world.positions.as_slice())?;
    let mut point_constraints = Vec::new();
    let mut fixed_constraints = Vec::new();
    for equality in &parsed.equalities {
        match equality {
            ParsedEquality::Joint(_) => {}
            ParsedEquality::Connect(connect) => {
                let EqualityFrames {
                    first,
                    second,
                    world_target,
                } = equality_frames(
                    &connect.target,
                    &state.body_refs,
                    &state.site_refs,
                    &parsed.world_sites,
                    options.root_pose,
                )?;
                let anchor = connect.anchor.unwrap_or_else(Vector3::zeros);
                let point_a = first.frame * Point3::from(anchor);
                let world_point = reference_pose.links[first.link] * point_a;
                let point_b = if let Some(frame) = world_target {
                    Point3::from(frame.translation.vector)
                } else {
                    match (&connect.target, second) {
                        (EqualityTarget::Sites { .. }, Some(second)) => {
                            Point3::from(second.frame.translation.vector)
                        }
                        (_, Some(second)) => {
                            reference_pose.links[second.link].inverse() * world_point
                        }
                        (_, None) => world_point,
                    }
                };
                point_constraints.push(LinkPointConstraint {
                    link_a: first.link,
                    point_a: point_a.coords.into(),
                    link_b: second.map(|frame| frame.link),
                    point_b: point_b.coords.into(),
                });
            }
            ParsedEquality::Weld(weld) => {
                let EqualityFrames {
                    first,
                    second,
                    world_target,
                } = equality_frames(
                    &weld.target,
                    &state.body_refs,
                    &state.site_refs,
                    &parsed.world_sites,
                    options.root_pose,
                )?;
                let (frame_a, frame_b) = match &weld.target {
                    EqualityTarget::Sites { .. } => {
                        let frame_b = if let Some(frame) = world_target {
                            frame
                        } else if let Some(second) = second {
                            second.frame
                        } else {
                            return Err(MjcfLoadError::Invalid(
                                "equality weld has no second site frame".into(),
                            ));
                        };
                        (first.frame, frame_b)
                    }
                    EqualityTarget::Bodies { .. } => {
                        let anchor =
                            Isometry3::translation(weld.anchor.x, weld.anchor.y, weld.anchor.z);
                        let (frame_b, world_frame_b) = if let Some(second) = second {
                            let frame_b = second.frame * anchor;
                            (frame_b, reference_pose.links[second.link] * frame_b)
                        } else {
                            let world_frame_b = options.root_pose * anchor;
                            (world_frame_b, world_frame_b)
                        };
                        let frame_a = weld.relpose.map_or_else(
                            || reference_pose.links[first.link].inverse() * world_frame_b,
                            |relpose| first.frame * relpose,
                        );
                        (frame_a, frame_b)
                    }
                };
                if weld.torquescale == 0.0 {
                    point_constraints.push(LinkPointConstraint {
                        link_a: first.link,
                        point_a: frame_a.translation.vector.into(),
                        link_b: second.map(|frame| frame.link),
                        point_b: frame_b.translation.vector.into(),
                    });
                } else {
                    fixed_constraints.push(LinkFixedConstraint {
                        link_a: first.link,
                        frame_a,
                        link_b: second.map(|frame| frame.link),
                        frame_b,
                    });
                }
            }
        }
    }
    world.set_link_point_constraints(point_constraints)?;
    world.set_link_fixed_constraints(fixed_constraints)?;
    if state.joint_springdampers.iter().any(Option::is_some) {
        let dimensions = world.articulation.dof() + if world.floating { 6 } else { 0 };
        let reference = world.articulation.generalized_dynamics(
            world.root_pose,
            world.positions.as_slice(),
            &DVector::zeros(dimensions),
            world.floating,
            Vector3::zeros(),
        )?;
        let base_offset = if world.floating { 6 } else { 0 };
        for (edge, parameters) in state.joint_springdampers.iter().enumerate() {
            let Some((time_constant, damping_ratio)) = *parameters else {
                continue;
            };
            for slot in state.joint_info[edge].dofs.clone() {
                let inertia = reference.mass[(base_offset + slot, base_offset + slot)];
                let stiffness = inertia / (time_constant * damping_ratio).powi(2);
                let damping = 2.0 * inertia / time_constant;
                if !inertia.is_finite()
                    || inertia <= 0.0
                    || !stiffness.is_finite()
                    || !damping.is_finite()
                {
                    return Err(MjcfLoadError::Invalid(format!(
                        "joint `{}` has invalid springdamper inertia",
                        state.joint_info[edge].name
                    )));
                }
                let rest_position = world
                    .joint_passive(slot)
                    .ok_or(ArticulatedWorldError::InvalidInput)?
                    .rest_position;
                world.set_joint_passive(
                    slot,
                    JointPassive {
                        stiffness,
                        damping,
                        rest_position,
                    },
                )?;
                world.set_joint_nonlinear_passive(slot, JointNonlinearPassive::default())?;
            }
        }
    }
    world.set_boxes(state.boxes)?;
    world.set_cylinders(state.cylinders)?;
    world.set_convex_shapes(state.convex_shapes)?;
    add_world_geometries(&mut world, parsed.world_geoms)?;

    let actuators = actuator::import(xml, &state.joint_info)?;
    Ok(LoadedMjcf {
        model_name: parsed.name,
        link_names: state.link_names,
        joints: state.joint_info,
        actuators,
        world,
    })
}

fn equality_slot(joints: &[MjcfJointInfo], name: &str) -> Result<usize, MjcfLoadError> {
    let mut matches = joints.iter().filter(|joint| joint.name == name);
    let joint = matches.next().ok_or_else(|| {
        MjcfLoadError::Invalid(format!("equality references unknown joint `{name}`"))
    })?;
    if matches.next().is_some() {
        return Err(MjcfLoadError::Invalid(format!(
            "equality references ambiguous joint `{name}`"
        )));
    }
    if !matches!(joint.kind, JointKind::Revolute | JointKind::Prismatic) || joint.dofs.len() != 1 {
        return Err(MjcfLoadError::Unsupported(format!(
            "equality joint `{name}` must be scalar hinge or slide"
        )));
    }
    Ok(joint.dofs.start)
}

fn register_body_frames(
    body: &ParsedBody,
    link: usize,
    link_from_body: Isometry3<f64>,
    state: &mut CompileState,
) {
    state
        .body_refs
        .entry(body.name.clone())
        .or_default()
        .push(FrameRef {
            link,
            frame: link_from_body,
        });
    for site in &body.sites {
        if site.name.is_empty() {
            continue;
        }
        state
            .site_refs
            .entry(site.name.clone())
            .or_default()
            .push(FrameRef {
                link,
                frame: link_from_body * site.pose,
            });
    }
}

fn named_frame(
    frames: &BTreeMap<String, Vec<FrameRef>>,
    kind: &str,
    name: &str,
) -> Result<FrameRef, MjcfLoadError> {
    let matches = frames.get(name).ok_or_else(|| {
        MjcfLoadError::Invalid(format!("equality references unknown {kind} `{name}`"))
    })?;
    if matches.len() != 1 {
        return Err(MjcfLoadError::Invalid(format!(
            "equality references ambiguous {kind} `{name}`"
        )));
    }
    Ok(matches[0])
}

fn equality_frames(
    target: &EqualityTarget,
    body_refs: &BTreeMap<String, Vec<FrameRef>>,
    site_refs: &BTreeMap<String, Vec<FrameRef>>,
    world_sites: &[ParsedSite],
    world_pose: Isometry3<f64>,
) -> Result<EqualityFrames, MjcfLoadError> {
    let (first, second, world_target) = match target {
        EqualityTarget::Bodies { first, second } => (
            named_frame(body_refs, "body", first)?,
            second
                .as_deref()
                .map(|name| named_frame(body_refs, "body", name))
                .transpose()?,
            None,
        ),
        EqualityTarget::Sites { first, second } => {
            let first = named_site_frame(site_refs, world_sites, world_pose, first)?;
            let second = named_site_frame(site_refs, world_sites, world_pose, second)?;
            match (first, second) {
                (SiteFrame::Link(first), SiteFrame::Link(second)) => (first, Some(second), None),
                (SiteFrame::Link(first), SiteFrame::World(second))
                | (SiteFrame::World(second), SiteFrame::Link(first)) => (first, None, Some(second)),
                (SiteFrame::World(_), SiteFrame::World(_)) => {
                    return Err(MjcfLoadError::Invalid(
                        "equality requires at least one dynamic site".into(),
                    ));
                }
            }
        }
    };
    if second.is_some_and(|frame| frame.link == first.link) {
        return Err(MjcfLoadError::Invalid(
            "equality references two frames on the same link".into(),
        ));
    }
    Ok(EqualityFrames {
        first,
        second,
        world_target,
    })
}

fn named_site_frame(
    site_refs: &BTreeMap<String, Vec<FrameRef>>,
    world_sites: &[ParsedSite],
    world_pose: Isometry3<f64>,
    name: &str,
) -> Result<SiteFrame, MjcfLoadError> {
    let link_sites = site_refs.get(name).map_or(&[][..], Vec::as_slice);
    let world_sites = world_sites
        .iter()
        .filter(|site| site.name == name)
        .collect::<Vec<_>>();
    match (link_sites, world_sites.as_slice()) {
        ([site], []) => Ok(SiteFrame::Link(*site)),
        ([], [site]) => Ok(SiteFrame::World(world_pose * site.pose)),
        ([], []) => Err(MjcfLoadError::Invalid(format!(
            "equality references unknown site `{name}`"
        ))),
        _ => Err(MjcfLoadError::Invalid(format!(
            "equality references ambiguous site `{name}`"
        ))),
    }
}

fn compile_root(body: &ParsedBody, state: &mut CompileState) -> Result<(), MjcfLoadError> {
    state.link_names.push(body.name.clone());
    state
        .links
        .push(body_link_spec(body, Isometry3::identity())?);
    register_body_frames(body, 0, Isometry3::identity(), state);
    add_link_geometries(0, &body.geoms, Isometry3::identity(), state)?;
    for child in &body.children {
        compile_child(child, 0, Isometry3::identity(), state)?;
    }
    Ok(())
}

fn compile_free_root(body: &ParsedBody, state: &mut CompileState) -> Result<(), MjcfLoadError> {
    if body.joints.len() != 1 || body.joints[0].kind != ParsedJointKind::Free {
        return Err(MjcfLoadError::Unsupported(format!(
            "top-level body `{}` combines freejoint with another joint",
            body.name
        )));
    }
    let name = &body.joints[0].name;
    let frame_rotation = body.pose.rotation.inverse();
    let joints = [
        (
            "tx",
            ParsedJointKind::Prismatic,
            frame_rotation * Vector3::x(),
        ),
        (
            "ty",
            ParsedJointKind::Prismatic,
            frame_rotation * Vector3::y(),
        ),
        (
            "tz",
            ParsedJointKind::Prismatic,
            frame_rotation * Vector3::z(),
        ),
        ("rotation", ParsedJointKind::Spherical, Vector3::z()),
    ]
    .into_iter()
    .map(|(suffix, kind, axis)| ParsedJoint {
        name: format!("{name}/{suffix}"),
        kind,
        position: Vector3::zeros(),
        axis,
        limits: None,
        armature: 0.0,
        reference: 0.0,
        passive: JointPassive::default(),
        nonlinear_passive: JointNonlinearPassive::default(),
        frictionloss: 0.0,
        springdamper: None,
    })
    .collect::<Vec<_>>();
    compile_child_joints(body, &joints, 0, Isometry3::identity(), state)
}

fn compile_child(
    body: &ParsedBody,
    parent: usize,
    parent_body_from_link: Isometry3<f64>,
    state: &mut CompileState,
) -> Result<(), MjcfLoadError> {
    compile_child_joints(body, &body.joints, parent, parent_body_from_link, state)
}

fn compile_child_joints(
    body: &ParsedBody,
    joints: &[ParsedJoint],
    parent: usize,
    parent_body_from_link: Isometry3<f64>,
    state: &mut CompileState,
) -> Result<(), MjcfLoadError> {
    if joints
        .iter()
        .any(|joint| joint.kind == ParsedJointKind::Free)
    {
        return Err(MjcfLoadError::Unsupported(format!(
            "non-root body `{}` has a freejoint",
            body.name
        )));
    }
    if joints.is_empty() {
        let child = state.links.len();
        state.link_names.push(body.name.clone());
        state
            .links
            .push(body_link_spec(body, Isometry3::identity())?);
        register_body_frames(body, child, Isometry3::identity(), state);
        state.joints.push(JointSpec {
            parent,
            child,
            origin: parent_body_from_link.inverse() * body.pose,
            kind: JointKind::Fixed,
            axis: Vector3::z(),
            limits: None,
        });
        state.joint_armatures.push(0.0);
        state.joint_references.push(0.0);
        state.joint_passives.push(JointPassive::default());
        state
            .joint_nonlinear_passives
            .push(JointNonlinearPassive::default());
        state.joint_frictions.push(0.0);
        state.joint_springdampers.push(None);
        state.joint_info.push(MjcfJointInfo {
            name: format!("{}/fixed", body.name),
            child_link: child,
            kind: JointKind::Fixed,
            dofs: state.dof..state.dof,
        });
        add_link_geometries(child, &body.geoms, Isometry3::identity(), state)?;
        for descendant in &body.children {
            compile_child(descendant, child, Isometry3::identity(), state)?;
        }
        return Ok(());
    }

    let logical_parent = parent;
    let mut chain_parent = parent;
    let mut previous_body_from_link = parent_body_from_link;
    for (index, joint) in joints.iter().enumerate() {
        let body_from_link =
            Isometry3::translation(joint.position.x, joint.position.y, joint.position.z);
        let link_from_body = body_from_link.inverse();
        let child = state.links.len();
        let is_physical_link = index + 1 == joints.len();
        state.link_names.push(if is_physical_link {
            body.name.clone()
        } else {
            format!("{}@{}", body.name, joint.name)
        });
        state.links.push(if is_physical_link {
            body_link_spec(body, link_from_body)?
        } else {
            empty_link_spec()
        });
        if is_physical_link {
            register_body_frames(body, child, link_from_body, state);
        }
        let kind = parsed_joint_kind(joint)?;
        let origin = if index == 0 {
            parent_body_from_link.inverse() * body.pose * body_from_link
        } else {
            previous_body_from_link.inverse() * body_from_link
        };
        let axis = link_from_body.rotation * joint.axis;
        let reference_inverse = match kind {
            JointKind::Revolute => Isometry3::from_parts(
                Translation3::identity(),
                UnitQuaternion::from_axis_angle(&Unit::new_normalize(axis), -joint.reference),
            ),
            JointKind::Prismatic => Isometry3::translation(
                -axis.x * joint.reference,
                -axis.y * joint.reference,
                -axis.z * joint.reference,
            ),
            _ => Isometry3::identity(),
        };
        state.joints.push(JointSpec {
            parent: chain_parent,
            child,
            origin: origin * reference_inverse,
            kind,
            axis,
            limits: joint.limits,
        });
        state.joint_armatures.push(joint.armature);
        state.joint_references.push(joint.reference);
        state.joint_passives.push(joint.passive);
        state.joint_nonlinear_passives.push(joint.nonlinear_passive);
        state.joint_frictions.push(joint.frictionloss);
        state.joint_springdampers.push(joint.springdamper);
        let width = joint_width(kind);
        state.joint_info.push(MjcfJointInfo {
            name: joint.name.clone(),
            child_link: child,
            kind,
            dofs: state.dof..(state.dof + width),
        });
        state.dof += width;
        chain_parent = child;
        previous_body_from_link = body_from_link;
    }
    if joints.len() > 1 {
        state
            .collision_exclusions
            .push((logical_parent, chain_parent));
    }
    let link_from_body = previous_body_from_link.inverse();
    add_link_geometries(chain_parent, &body.geoms, link_from_body, state)?;
    for descendant in &body.children {
        compile_child(descendant, chain_parent, previous_body_from_link, state)?;
    }
    Ok(())
}

fn parsed_joint_kind(joint: &ParsedJoint) -> Result<JointKind, MjcfLoadError> {
    match joint.kind {
        ParsedJointKind::Revolute => Ok(JointKind::Revolute),
        ParsedJointKind::Prismatic => Ok(JointKind::Prismatic),
        ParsedJointKind::Spherical => Ok(JointKind::Spherical),
        ParsedJointKind::Free => Err(MjcfLoadError::Unsupported(format!(
            "joint `{}` is a non-root freejoint",
            joint.name
        ))),
    }
}

fn joint_width(kind: JointKind) -> usize {
    match kind {
        JointKind::Fixed => 0,
        JointKind::Revolute | JointKind::Prismatic => 1,
        JointKind::Spherical => 3,
    }
}

fn empty_link_spec() -> LinkSpec {
    LinkSpec {
        mass: 0.0,
        center_of_mass: Vector3::zeros(),
        inertia: Matrix3::zeros(),
    }
}

fn body_link_spec(
    body: &ParsedBody,
    link_from_body: Isometry3<f64>,
) -> Result<LinkSpec, MjcfLoadError> {
    let inferred;
    let inertial = if let Some(inertial) = body.inertial.as_ref() {
        Some(inertial)
    } else if body.geoms.is_empty() {
        None
    } else {
        inferred = infer_inertial(&body.geoms)?;
        Some(&inferred)
    };
    Ok(link_spec(inertial, link_from_body))
}

fn link_spec(inertial: Option<&ParsedInertial>, link_from_body: Isometry3<f64>) -> LinkSpec {
    let Some(inertial) = inertial else {
        return LinkSpec {
            mass: 0.0,
            center_of_mass: Vector3::zeros(),
            inertia: Matrix3::zeros(),
        };
    };
    let rotation = link_from_body.rotation.to_rotation_matrix();
    LinkSpec {
        mass: inertial.mass,
        center_of_mass: link_from_body
            .transform_point(&inertial.center.into())
            .coords,
        inertia: rotation.matrix() * inertial.inertia * rotation.matrix().transpose(),
    }
}

fn infer_inertial(geoms: &[ParsedGeom]) -> Result<ParsedInertial, MjcfLoadError> {
    let mut contributions = Vec::with_capacity(geoms.len());
    let mut mass = 0.0;
    let mut weighted_center = Vector3::zeros();
    for geom in geoms {
        let (geom_mass, local_inertia) = geom.mass_properties()?;
        if geom_mass == 0.0 {
            continue;
        }
        let center = geom.pose.translation.vector;
        let rotation = geom.pose.rotation.to_rotation_matrix();
        let body_inertia = rotation.matrix() * local_inertia * rotation.matrix().transpose();
        mass += geom_mass;
        weighted_center += geom_mass * center;
        contributions.push((geom_mass, center, body_inertia));
    }
    if mass == 0.0 {
        return Ok(ParsedInertial {
            mass: 0.0,
            center: Vector3::zeros(),
            inertia: Matrix3::zeros(),
        });
    }
    let center = weighted_center / mass;
    let identity = Matrix3::identity();
    let inertia = contributions.into_iter().fold(
        Matrix3::zeros(),
        |sum, (geom_mass, geom_center, geom_inertia)| {
            let offset = geom_center - center;
            sum + geom_inertia
                + geom_mass * (offset.norm_squared() * identity - offset * offset.transpose())
        },
    );
    Ok(ParsedInertial {
        mass,
        center,
        inertia,
    })
}

impl ParsedGeom {
    fn mass_properties(&self) -> Result<(f64, Matrix3<f64>), MjcfLoadError> {
        let (volume, unit_inertia) = match self.kind {
            ParsedGeomKind::Sphere { radius } => {
                let volume = 4.0 * core::f64::consts::PI * radius.powi(3) / 3.0;
                let inertia = Matrix3::identity() * (2.0 * radius.powi(2) / 5.0);
                (volume, inertia)
            }
            ParsedGeomKind::Box { half_extents } => {
                let volume = 8.0 * half_extents.x * half_extents.y * half_extents.z;
                let inertia = Matrix3::from_diagonal(&Vector3::new(
                    (half_extents.y.powi(2) + half_extents.z.powi(2)) / 3.0,
                    (half_extents.x.powi(2) + half_extents.z.powi(2)) / 3.0,
                    (half_extents.x.powi(2) + half_extents.y.powi(2)) / 3.0,
                ));
                (volume, inertia)
            }
            ParsedGeomKind::Cylinder {
                radius,
                half_height,
            } => {
                let volume = 2.0 * core::f64::consts::PI * radius.powi(2) * half_height;
                let transverse = (3.0 * radius.powi(2) + 4.0 * half_height.powi(2)) / 12.0;
                let inertia = Matrix3::from_diagonal(&Vector3::new(
                    transverse,
                    transverse,
                    radius.powi(2) / 2.0,
                ));
                (volume, inertia)
            }
            ParsedGeomKind::Capsule {
                radius,
                half_height,
            } => {
                let cylinder_volume = 2.0 * core::f64::consts::PI * radius.powi(2) * half_height;
                let sphere_volume = 4.0 * core::f64::consts::PI * radius.powi(3) / 3.0;
                let volume = cylinder_volume + sphere_volume;
                let cylinder_fraction = cylinder_volume / volume;
                let sphere_fraction = sphere_volume / volume;
                let cylinder_transverse = (3.0 * radius.powi(2) + 4.0 * half_height.powi(2)) / 12.0;
                let cap_transverse =
                    83.0 * radius.powi(2) / 320.0 + (half_height + 3.0 * radius / 8.0).powi(2);
                let transverse =
                    cylinder_fraction * cylinder_transverse + sphere_fraction * cap_transverse;
                let axial = cylinder_fraction * radius.powi(2) / 2.0
                    + sphere_fraction * 2.0 * radius.powi(2) / 5.0;
                (
                    volume,
                    Matrix3::from_diagonal(&Vector3::new(transverse, transverse, axial)),
                )
            }
            ParsedGeomKind::Convex { .. } => {
                return Err(MjcfLoadError::Unsupported(
                    "mesh geom inertia needs an explicit body inertial".into(),
                ));
            }
            ParsedGeomKind::Plane => {
                return Err(MjcfLoadError::Unsupported(
                    "body plane cannot provide finite inertia".into(),
                ));
            }
        };
        let mass = self.mass.unwrap_or(self.density * volume);
        Ok((mass, unit_inertia * mass))
    }
}

fn add_link_geometries(
    link: usize,
    geoms: &[ParsedGeom],
    link_from_body: Isometry3<f64>,
    state: &mut CompileState,
) -> Result<(), MjcfLoadError> {
    for geom in geoms {
        let origin = link_from_body * geom.pose;
        match &geom.kind {
            ParsedGeomKind::Sphere { radius } => state.spheres.push(LinkSphere {
                link,
                center: origin.translation.vector,
                radius: *radius,
            }),
            ParsedGeomKind::Box { half_extents } => state.boxes.push(LinkBox {
                link,
                origin,
                half_extents: *half_extents,
            }),
            ParsedGeomKind::Cylinder {
                radius,
                half_height,
            } => state.cylinders.push(LinkCylinder {
                link,
                origin,
                half_height: *half_height,
                radius: *radius,
            }),
            ParsedGeomKind::Capsule {
                radius,
                half_height,
            } => {
                state.cylinders.push(LinkCylinder {
                    link,
                    origin,
                    half_height: *half_height,
                    radius: *radius,
                });
                for direction in [-1.0, 1.0] {
                    state.spheres.push(LinkSphere {
                        link,
                        center: origin
                            .transform_point(
                                &Vector3::new(0.0, 0.0, direction * half_height).into(),
                            )
                            .coords,
                        radius: *radius,
                    });
                }
            }
            ParsedGeomKind::Convex { geometries } => {
                state
                    .convex_shapes
                    .extend(geometries.iter().cloned().map(|geometry| LinkConvex {
                        link,
                        origin,
                        geometry,
                    }));
            }
            ParsedGeomKind::Plane => {
                return Err(MjcfLoadError::Unsupported(
                    "a plane geom attached to a body".into(),
                ));
            }
        }
    }
    Ok(())
}

fn add_world_geometries(
    world: &mut ArticulatedWorld,
    geoms: Vec<ParsedGeom>,
) -> Result<(), MjcfLoadError> {
    let mut colliders = Vec::new();
    for geom in geoms {
        match geom.kind {
            ParsedGeomKind::Plane => {
                let normal = geom.pose.rotation * Vector3::z();
                if normal.dot(&Vector3::z()).abs() < 1.0 - 1e-9
                    || geom.pose.translation.vector.z.abs() > 1e-9
                {
                    return Err(MjcfLoadError::Unsupported(
                        "only the z=0 world plane maps to Tessera ground".into(),
                    ));
                }
            }
            ParsedGeomKind::Sphere { radius } => colliders.push(SceneCollider::Sphere {
                center: geom.pose.translation.vector,
                radius,
            }),
            ParsedGeomKind::Box { half_extents } => colliders.push(SceneCollider::Box {
                origin: geom.pose,
                half_extents,
            }),
            ParsedGeomKind::Cylinder {
                radius,
                half_height,
            } => colliders.push(SceneCollider::Cylinder {
                origin: geom.pose,
                half_height,
                radius,
            }),
            ParsedGeomKind::Capsule {
                radius,
                half_height,
            } => colliders.push(SceneCollider::Capsule {
                origin: geom.pose,
                half_height,
                radius,
            }),
            ParsedGeomKind::Convex { geometries } => {
                colliders.extend(
                    geometries
                        .into_iter()
                        .map(|geometry| SceneCollider::Convex {
                            origin: geom.pose,
                            geometry,
                        }),
                );
            }
        }
    }
    if !colliders.is_empty() {
        let body = SceneBody::new(Isometry3::identity(), 0.0, Matrix3::zeros(), colliders)?;
        let _slot = world.add_scene_body(body);
    }
    Ok(())
}

fn collect_mesh_assets(
    xml: &str,
    mut resolver: Option<&mut dyn ConvexMeshResolver>,
) -> Result<BTreeMap<String, Vec<ConvexGeometry>>, MjcfLoadError> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);
    let mut in_asset = false;
    let mut meshes = BTreeMap::new();
    loop {
        match reader.read_event() {
            Ok(Event::Start(element)) if element.name().as_ref() == b"asset" => in_asset = true,
            Ok(Event::End(element)) if element.name().as_ref() == b"asset" => in_asset = false,
            Ok(Event::Start(element)) | Ok(Event::Empty(element))
                if in_asset && element.name().as_ref() == b"mesh" =>
            {
                let attrs = attributes(&element)?;
                let name = attrs.get("name").cloned().ok_or_else(|| {
                    MjcfLoadError::Invalid("inline mesh asset needs a name".into())
                })?;
                let geometries = if let Some(filename) = attrs.get("file") {
                    if attrs.contains_key("vertex") || attrs.contains_key("face") {
                        return Err(MjcfLoadError::Invalid(format!(
                            "mesh asset `{name}` mixes external and inline data"
                        )));
                    }
                    let scale = parse_mesh_scale(&name, attrs.get("scale"))?;
                    let resolver = resolver.as_deref_mut().ok_or_else(|| {
                        MjcfLoadError::Unsupported(format!(
                            "mesh asset `{name}` references external file `{filename}` without a resolver"
                        ))
                    })?;
                    let parts = resolver
                        .resolve(filename, [scale.x, scale.y, scale.z])
                        .map_err(|message| MjcfLoadError::Mesh {
                            filename: filename.clone(),
                            message,
                        })?;
                    if parts.is_empty() {
                        return Err(MjcfLoadError::Invalid(format!(
                            "mesh asset `{name}` resolved to no convex parts"
                        )));
                    }
                    parts
                } else {
                    vec![parse_inline_mesh(&name, &attrs)?]
                };
                if meshes.insert(name.clone(), geometries).is_some() {
                    return Err(MjcfLoadError::Invalid(format!(
                        "duplicate mesh asset `{name}`"
                    )));
                }
            }
            Ok(Event::Eof) => break,
            Err(error) => return Err(MjcfLoadError::Xml(error.to_string())),
            _ => {}
        }
    }
    Ok(meshes)
}

fn parse_inline_mesh(
    name: &str,
    attrs: &BTreeMap<String, String>,
) -> Result<ConvexGeometry, MjcfLoadError> {
    let coordinates = attrs
        .get("vertex")
        .ok_or_else(|| MjcfLoadError::Unsupported(format!("mesh asset `{name}` has no vertices")))
        .and_then(|value| parse_number_text("vertex", value))?;
    if coordinates.len() < 12 || coordinates.len() % 3 != 0 {
        return Err(MjcfLoadError::Invalid(format!(
            "mesh asset `{name}` needs at least four 3D vertices"
        )));
    }
    let scale = parse_mesh_scale(name, attrs.get("scale"))?;
    let vertices = coordinates
        .chunks_exact(3)
        .map(|point| Vector3::new(point[0], point[1], point[2]).component_mul(&scale))
        .collect::<Vec<_>>();
    let face_text = attrs.get("face").ok_or_else(|| {
        MjcfLoadError::Unsupported(format!("mesh asset `{name}` needs inline triangle faces"))
    })?;
    let indices = face_text
        .split_ascii_whitespace()
        .map(|part| {
            part.parse::<usize>().map_err(|error| {
                MjcfLoadError::Invalid(format!("mesh asset `{name}` has invalid face: {error}"))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if indices.len() < 12 || indices.len() % 3 != 0 {
        return Err(MjcfLoadError::Invalid(format!(
            "mesh asset `{name}` needs at least four triangle faces"
        )));
    }
    let centroid = vertices.iter().copied().sum::<Vector3<f64>>() / vertices.len() as f64;
    let mut normals = Vec::new();
    let mut edge_faces: BTreeMap<(usize, usize), Vec<Vector3<f64>>> = BTreeMap::new();
    for face in indices.chunks_exact(3) {
        if face.iter().any(|index| *index >= vertices.len())
            || face[0] == face[1]
            || face[1] == face[2]
            || face[2] == face[0]
        {
            return Err(MjcfLoadError::Invalid(format!(
                "mesh asset `{name}` has an invalid face index"
            )));
        }
        let a = vertices[face[0]];
        let b = vertices[face[1]];
        let c = vertices[face[2]];
        let Some(mut normal) = (b - a).cross(&(c - a)).try_normalize(1e-12) else {
            return Err(MjcfLoadError::Invalid(format!(
                "mesh asset `{name}` has a degenerate face"
            )));
        };
        if normal.dot(&(centroid - a)) > 0.0 {
            normal = -normal;
        }
        if vertices
            .iter()
            .any(|vertex| normal.dot(&(vertex - a)) > 1e-8)
        {
            return Err(MjcfLoadError::Invalid(format!(
                "mesh asset `{name}` is not a convex closed hull"
            )));
        }
        push_unique_axis(&mut normals, normal);
        for [start, end] in [[face[0], face[1]], [face[1], face[2]], [face[2], face[0]]] {
            let key = if start < end {
                (start, end)
            } else {
                (end, start)
            };
            edge_faces.entry(key).or_default().push(normal);
        }
    }
    let mut edges = Vec::new();
    for ((start, end), adjacent) in edge_faces {
        if adjacent.len() != 2 {
            return Err(MjcfLoadError::Invalid(format!(
                "mesh asset `{name}` is not a closed two-manifold"
            )));
        }
        if adjacent[0].dot(&adjacent[1]).abs() >= 1.0 - 1e-8 {
            continue;
        }
        let Some(direction) = (vertices[end] - vertices[start]).try_normalize(1e-12) else {
            return Err(MjcfLoadError::Invalid(format!(
                "mesh asset `{name}` has a zero edge"
            )));
        };
        push_unique_axis(&mut edges, direction);
    }
    ConvexGeometry::new(vertices, normals, edges)
        .map_err(|error| MjcfLoadError::Invalid(format!("mesh asset `{name}`: {error}")))
}

fn parse_mesh_scale(name: &str, value: Option<&String>) -> Result<Vector3<f64>, MjcfLoadError> {
    let Some(value) = value else {
        return Ok(Vector3::repeat(1.0));
    };
    let values = parse_number_text("scale", value)?;
    match values.as_slice() {
        [uniform] if uniform.is_finite() && *uniform != 0.0 => Ok(Vector3::repeat(*uniform)),
        [x, y, z]
            if x.is_finite()
                && y.is_finite()
                && z.is_finite()
                && *x != 0.0
                && *y != 0.0
                && *z != 0.0 =>
        {
            Ok(Vector3::new(*x, *y, *z))
        }
        _ => Err(MjcfLoadError::Invalid(format!(
            "mesh asset `{name}` has invalid scale"
        ))),
    }
}

fn push_unique_axis(axes: &mut Vec<Vector3<f64>>, candidate: Vector3<f64>) {
    if axes
        .iter()
        .all(|axis| axis.dot(&candidate).abs() < 1.0 - 1e-8)
    {
        axes.push(candidate);
    }
}

fn collect_defaults(xml: &str) -> Result<BTreeMap<String, DefaultClass>, MjcfLoadError> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);
    let mut classes = BTreeMap::new();
    let _global = classes.insert(String::new(), DefaultClass::default());
    let mut stack: Vec<String> = Vec::new();
    loop {
        match reader.read_event() {
            Ok(Event::Start(element)) if element.name().as_ref() == b"default" => {
                push_default_class(&attributes(&element)?, &mut classes, &mut stack)?;
            }
            Ok(Event::Empty(element)) if element.name().as_ref() == b"default" => {
                push_default_class(&attributes(&element)?, &mut classes, &mut stack)?;
                let _closed = stack.pop();
            }
            Ok(Event::Start(element)) | Ok(Event::Empty(element)) if !stack.is_empty() => {
                let kind = match element.name().as_ref() {
                    b"geom" => Some(DefaultElement::Geom),
                    b"joint" => Some(DefaultElement::Joint),
                    b"motor" => Some(DefaultElement::Actuator("motor")),
                    b"position" => Some(DefaultElement::Actuator("position")),
                    b"velocity" => Some(DefaultElement::Actuator("velocity")),
                    b"damper" => Some(DefaultElement::Actuator("damper")),
                    b"general" => Some(DefaultElement::Actuator("general")),
                    _ => None,
                };
                if let Some(kind) = kind {
                    update_default_class(
                        stack.last().map(String::as_str).unwrap_or_default(),
                        kind,
                        attributes(&element)?,
                        &mut classes,
                    )?;
                }
            }
            Ok(Event::End(element)) if element.name().as_ref() == b"default" => {
                if stack.pop().is_none() {
                    return Err(MjcfLoadError::Xml(
                        "default close without matching open".into(),
                    ));
                }
            }
            Ok(Event::Eof) => break,
            Err(error) => return Err(MjcfLoadError::Xml(error.to_string())),
            _ => {}
        }
    }
    if !stack.is_empty() {
        return Err(MjcfLoadError::Xml("unclosed default element".into()));
    }
    Ok(classes)
}

fn push_default_class(
    attrs: &BTreeMap<String, String>,
    classes: &mut BTreeMap<String, DefaultClass>,
    stack: &mut Vec<String>,
) -> Result<(), MjcfLoadError> {
    let class = attrs.get("class").cloned().unwrap_or_default();
    if class.is_empty() && !stack.is_empty() {
        return Err(MjcfLoadError::Invalid(
            "a nested default needs a class name".into(),
        ));
    }
    if !class.is_empty() && classes.contains_key(&class) {
        return Err(MjcfLoadError::Invalid(format!(
            "duplicate default class `{class}`"
        )));
    }
    if !classes.contains_key(&class) {
        let parent = stack.last().map(String::as_str).unwrap_or_default();
        let mut inherited = classes.get(parent).cloned().ok_or_else(|| {
            MjcfLoadError::Invalid(format!("unknown parent default class `{parent}`"))
        })?;
        inherited.actuators.clear();
        inherited.actuator_parent = Some(parent.into());
        let _previous = classes.insert(class.clone(), inherited);
    }
    stack.push(class);
    Ok(())
}

fn update_default_class(
    class: &str,
    kind: DefaultElement,
    attrs: BTreeMap<String, String>,
    classes: &mut BTreeMap<String, DefaultClass>,
) -> Result<(), MjcfLoadError> {
    let defaults = classes
        .get_mut(class)
        .ok_or_else(|| MjcfLoadError::Invalid(format!("unknown default class `{class}`")))?;
    let target = match kind {
        DefaultElement::Geom => &mut defaults.geom,
        DefaultElement::Joint => &mut defaults.joint,
        DefaultElement::Actuator(kind) => defaults.actuators.entry(kind.into()).or_default(),
    };
    for (key, value) in attrs {
        if key != "class" {
            let _previous = target.insert(key, value);
        }
    }
    Ok(())
}

fn resolve_defaults(
    explicit: BTreeMap<String, String>,
    active_class: Option<&str>,
    kind: DefaultElement,
    classes: &BTreeMap<String, DefaultClass>,
) -> Result<BTreeMap<String, String>, MjcfLoadError> {
    let selected = explicit
        .get("class")
        .map(String::as_str)
        .or(active_class)
        .unwrap_or_default();
    let defaults = classes
        .get(selected)
        .ok_or_else(|| MjcfLoadError::Invalid(format!("unknown default class `{selected}`")))?;
    let mut merged = match kind {
        DefaultElement::Geom => defaults.geom.clone(),
        DefaultElement::Joint => defaults.joint.clone(),
        DefaultElement::Actuator(kind) => {
            let mut chain = vec![defaults];
            let mut current = defaults;
            while let Some(parent) = &current.actuator_parent {
                current = classes.get(parent).ok_or_else(|| {
                    MjcfLoadError::Invalid(format!("unknown parent default class `{parent}`"))
                })?;
                chain.push(current);
            }
            let mut merged = BTreeMap::new();
            for class in chain.into_iter().rev() {
                if let Some(attrs) = class.actuators.get(kind) {
                    merged.extend(attrs.clone());
                }
                if let Some(attrs) = class.actuators.get("general") {
                    merged.extend(attrs.clone());
                }
            }
            merged
        }
    };
    for (key, value) in explicit {
        let _previous = merged.insert(key, value);
    }
    Ok(merged)
}

fn parse_document(
    xml: &str,
    resolver: Option<&mut dyn ConvexMeshResolver>,
) -> Result<ParsedModel, MjcfLoadError> {
    let defaults = collect_defaults(xml)?;
    let meshes = collect_mesh_assets(xml, resolver)?;
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);
    let mut angle = AngleUnit::Degree;
    let mut model_name = None;
    let mut gravity = None;
    let mut body_stack: Vec<ParsedBody> = Vec::new();
    let mut roots = Vec::new();
    let mut world_geoms = Vec::new();
    let mut world_sites = Vec::new();
    let mut equalities = Vec::new();
    let mut in_worldbody = false;
    let mut in_equality = false;
    let mut generated_body = 0usize;
    let mut generated_joint = 0usize;

    loop {
        match reader.read_event() {
            Ok(Event::Start(element)) => {
                let name = element.name();
                match name.as_ref() {
                    b"mujoco" => {
                        model_name = attributes(&element)?.remove("model");
                    }
                    b"compiler" => {
                        parse_compiler(&attributes(&element)?, &mut angle)?;
                    }
                    b"option" => {
                        gravity =
                            parse_optional_vec3(&attributes(&element)?, "gravity")?.map(Into::into);
                    }
                    b"worldbody" => in_worldbody = true,
                    b"equality" => in_equality = true,
                    b"joint" if in_equality => {
                        if let Some(equality) = parse_equality_joint(&attributes(&element)?)? {
                            equalities.push(ParsedEquality::Joint(equality));
                        }
                    }
                    b"connect" if in_equality => {
                        if let Some(equality) = parse_equality_connect(&attributes(&element)?)? {
                            equalities.push(ParsedEquality::Connect(equality));
                        }
                    }
                    b"weld" if in_equality => {
                        if let Some(equality) = parse_equality_weld(&attributes(&element)?)? {
                            equalities.push(ParsedEquality::Weld(equality));
                        }
                    }
                    _ if in_equality => {
                        return Err(MjcfLoadError::Unsupported(
                            "MJCF equality type other than joint, connect or weld".into(),
                        ));
                    }
                    b"body" if in_worldbody => {
                        let attrs = attributes(&element)?;
                        let inherited = body_stack
                            .last()
                            .and_then(|body| body.child_class.as_deref());
                        let body = parse_body(&attrs, angle, generated_body, inherited)?;
                        if let Some(class) = body.child_class.as_deref()
                            && !defaults.contains_key(class)
                        {
                            return Err(MjcfLoadError::Invalid(format!(
                                "unknown child default class `{class}`"
                            )));
                        }
                        body_stack.push(body);
                        generated_body += 1;
                    }
                    b"inertial" if !body_stack.is_empty() => {
                        let inertial = parse_inertial(&attributes(&element)?, angle)?;
                        let Some(body) = body_stack.last_mut() else {
                            return Err(MjcfLoadError::Xml("inertial outside body".into()));
                        };
                        if body.inertial.replace(inertial).is_some() {
                            return Err(MjcfLoadError::Invalid(format!(
                                "body `{}` has multiple inertials",
                                body.name
                            )));
                        }
                    }
                    b"joint" if !body_stack.is_empty() => {
                        let active_class = body_stack
                            .last()
                            .and_then(|body| body.child_class.as_deref());
                        let attrs = resolve_defaults(
                            attributes(&element)?,
                            active_class,
                            DefaultElement::Joint,
                            &defaults,
                        )?;
                        let joint = parse_joint(&attrs, angle, generated_joint)?;
                        generated_joint += 1;
                        let Some(body) = body_stack.last_mut() else {
                            return Err(MjcfLoadError::Xml("joint outside body".into()));
                        };
                        body.joints.push(joint);
                    }
                    b"freejoint" if !body_stack.is_empty() => {
                        let attrs = attributes(&element)?;
                        let name = attrs
                            .get("name")
                            .cloned()
                            .unwrap_or_else(|| format!("joint_{generated_joint}"));
                        generated_joint += 1;
                        let Some(body) = body_stack.last_mut() else {
                            return Err(MjcfLoadError::Xml("freejoint outside body".into()));
                        };
                        body.joints.push(ParsedJoint {
                            name,
                            kind: ParsedJointKind::Free,
                            position: Vector3::zeros(),
                            axis: Vector3::z(),
                            limits: None,
                            armature: 0.0,
                            reference: 0.0,
                            passive: JointPassive::default(),
                            nonlinear_passive: JointNonlinearPassive::default(),
                            frictionloss: 0.0,
                            springdamper: None,
                        });
                    }
                    b"geom" if in_worldbody => {
                        let active_class = body_stack
                            .last()
                            .and_then(|body| body.child_class.as_deref());
                        let attrs = resolve_defaults(
                            attributes(&element)?,
                            active_class,
                            DefaultElement::Geom,
                            &defaults,
                        )?;
                        let geom = parse_geom(&attrs, angle, &meshes)?;
                        if let Some(body) = body_stack.last_mut() {
                            body.geoms.push(geom);
                        } else {
                            world_geoms.push(geom);
                        }
                    }
                    b"site" if in_worldbody => {
                        let site = parse_site(&attributes(&element)?, angle)?;
                        if let Some(body) = body_stack.last_mut() {
                            body.sites.push(site);
                        } else {
                            world_sites.push(site);
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::Empty(element)) => {
                if element.name().as_ref() == b"compiler" {
                    parse_compiler(&attributes(&element)?, &mut angle)?;
                } else if element.name().as_ref() == b"joint" && in_equality {
                    if let Some(equality) = parse_equality_joint(&attributes(&element)?)? {
                        equalities.push(ParsedEquality::Joint(equality));
                    }
                } else if element.name().as_ref() == b"connect" && in_equality {
                    if let Some(equality) = parse_equality_connect(&attributes(&element)?)? {
                        equalities.push(ParsedEquality::Connect(equality));
                    }
                } else if element.name().as_ref() == b"weld" && in_equality {
                    if let Some(equality) = parse_equality_weld(&attributes(&element)?)? {
                        equalities.push(ParsedEquality::Weld(equality));
                    }
                } else if in_equality {
                    return Err(MjcfLoadError::Unsupported(
                        "MJCF equality type other than joint, connect or weld".into(),
                    ));
                } else if element.name().as_ref() == b"site" && in_worldbody {
                    let site = parse_site(&attributes(&element)?, angle)?;
                    if let Some(body) = body_stack.last_mut() {
                        body.sites.push(site);
                    } else {
                        world_sites.push(site);
                    }
                } else {
                    parse_empty_element(
                        &element,
                        angle,
                        in_worldbody,
                        body_stack.as_mut_slice(),
                        &mut roots,
                        &mut world_geoms,
                        &mut generated_body,
                        &mut generated_joint,
                        &mut gravity,
                        &defaults,
                        &meshes,
                    )?;
                }
            }
            Ok(Event::End(element)) => match element.name().as_ref() {
                b"body" if in_worldbody => finish_body(&mut body_stack, &mut roots)?,
                b"worldbody" => in_worldbody = false,
                b"equality" => in_equality = false,
                _ => {}
            },
            Ok(Event::Eof) => break,
            Err(error) => return Err(MjcfLoadError::Xml(error.to_string())),
            _ => {}
        }
    }
    if !body_stack.is_empty() {
        return Err(MjcfLoadError::Xml("unclosed body element".into()));
    }
    Ok(ParsedModel {
        name: model_name,
        gravity,
        roots,
        world_geoms,
        world_sites,
        equalities,
    })
}

fn parse_equality_joint(
    attrs: &BTreeMap<String, String>,
) -> Result<Option<ParsedEqualityJoint>, MjcfLoadError> {
    match attrs.get("active").map(String::as_str) {
        Some("false") => return Ok(None),
        Some("true") | None => {}
        Some(value) => {
            return Err(MjcfLoadError::Invalid(format!(
                "equality joint has invalid active value `{value}`"
            )));
        }
    }
    if let Some(attribute) = attrs.keys().find(|key| {
        !matches!(
            key.as_str(),
            "name" | "active" | "joint1" | "joint2" | "polycoef"
        )
    }) {
        return Err(MjcfLoadError::Unsupported(format!(
            "equality joint attribute `{attribute}`"
        )));
    }
    let follower_name = attrs
        .get("joint1")
        .filter(|name| !name.is_empty())
        .cloned()
        .ok_or_else(|| MjcfLoadError::Invalid("equality joint needs joint1".into()))?;
    let source_name = attrs.get("joint2").cloned();
    let coefficients = match parse_optional_numbers(attrs, "polycoef")? {
        None => [0.0, 1.0, 0.0, 0.0, 0.0],
        Some(values) => values.try_into().map_err(|_| {
            MjcfLoadError::Invalid("equality joint polycoef needs five values".into())
        })?,
    };
    Ok(Some(ParsedEqualityJoint {
        follower_name,
        source_name,
        coefficients,
    }))
}

fn parse_equality_active(
    attrs: &BTreeMap<String, String>,
    kind: &str,
) -> Result<bool, MjcfLoadError> {
    match attrs.get("active").map(String::as_str) {
        Some("false") => Ok(false),
        Some("true") | None => Ok(true),
        Some(value) => Err(MjcfLoadError::Invalid(format!(
            "equality {kind} has invalid active value `{value}`"
        ))),
    }
}

fn parse_equality_target(
    attrs: &BTreeMap<String, String>,
    kind: &str,
) -> Result<EqualityTarget, MjcfLoadError> {
    let body1 = attrs.get("body1");
    let body2 = attrs.get("body2");
    let site1 = attrs.get("site1");
    let site2 = attrs.get("site2");
    if site1.is_some() || site2.is_some() {
        if body1.is_some() || body2.is_some() {
            return Err(MjcfLoadError::Invalid(format!(
                "equality {kind} mixes body and site references"
            )));
        }
        let first = site1.filter(|name| !name.is_empty()).ok_or_else(|| {
            MjcfLoadError::Invalid(format!("equality {kind} needs site1 and site2"))
        })?;
        let second = site2.filter(|name| !name.is_empty()).ok_or_else(|| {
            MjcfLoadError::Invalid(format!("equality {kind} needs site1 and site2"))
        })?;
        return Ok(EqualityTarget::Sites {
            first: first.clone(),
            second: second.clone(),
        });
    }
    let first = body1
        .filter(|name| !name.is_empty())
        .ok_or_else(|| MjcfLoadError::Invalid(format!("equality {kind} needs body1")))?;
    Ok(EqualityTarget::Bodies {
        first: first.clone(),
        second: body2.cloned(),
    })
}

fn check_equality_attributes(
    attrs: &BTreeMap<String, String>,
    kind: &str,
    extra: &[&str],
) -> Result<(), MjcfLoadError> {
    if let Some(attribute) = attrs.keys().find(|key| {
        !matches!(
            key.as_str(),
            "name" | "active" | "body1" | "body2" | "site1" | "site2"
        ) && !extra.contains(&key.as_str())
    }) {
        return Err(MjcfLoadError::Unsupported(format!(
            "equality {kind} attribute `{attribute}`"
        )));
    }
    Ok(())
}

fn parse_equality_connect(
    attrs: &BTreeMap<String, String>,
) -> Result<Option<ParsedEqualityConnect>, MjcfLoadError> {
    if !parse_equality_active(attrs, "connect")? {
        return Ok(None);
    }
    check_equality_attributes(attrs, "connect", &["anchor"])?;
    let target = parse_equality_target(attrs, "connect")?;
    let anchor = parse_optional_vec3(attrs, "anchor")?;
    match target {
        EqualityTarget::Bodies { .. } if anchor.is_none() => Err(MjcfLoadError::Invalid(
            "equality connect with bodies needs anchor".into(),
        )),
        EqualityTarget::Sites { .. } if anchor.is_some() => Err(MjcfLoadError::Invalid(
            "equality connect with sites cannot specify anchor".into(),
        )),
        _ => Ok(Some(ParsedEqualityConnect { target, anchor })),
    }
}

fn parse_equality_weld(
    attrs: &BTreeMap<String, String>,
) -> Result<Option<ParsedEqualityWeld>, MjcfLoadError> {
    if !parse_equality_active(attrs, "weld")? {
        return Ok(None);
    }
    check_equality_attributes(attrs, "weld", &["anchor", "relpose", "torquescale"])?;
    let target = parse_equality_target(attrs, "weld")?;
    if matches!(target, EqualityTarget::Sites { .. })
        && (attrs.contains_key("anchor") || attrs.contains_key("relpose"))
    {
        return Err(MjcfLoadError::Invalid(
            "equality weld with sites cannot specify anchor or relpose".into(),
        ));
    }
    let anchor = parse_optional_vec3(attrs, "anchor")?.unwrap_or_else(Vector3::zeros);
    let torquescale = optional_scalar(attrs, "torquescale")?.unwrap_or(1.0);
    if !matches!(torquescale, 0.0 | 1.0) {
        return Err(MjcfLoadError::Unsupported(format!(
            "equality weld torquescale `{torquescale}`"
        )));
    }
    let relpose = parse_optional_numbers(attrs, "relpose")?
        .map(|values| {
            if values.len() != 7 {
                return Err(MjcfLoadError::Invalid(
                    "equality weld relpose needs seven values".into(),
                ));
            }
            let quaternion = Quaternion::new(values[3], values[4], values[5], values[6]);
            if quaternion.coords.iter().all(|value| *value == 0.0) {
                return Ok(None);
            }
            if quaternion.norm() <= 1e-12 || !quaternion.norm().is_finite() {
                return Err(MjcfLoadError::Invalid(
                    "equality weld relpose has invalid quaternion".into(),
                ));
            }
            Ok(Some(Isometry3::from_parts(
                Translation3::new(values[0], values[1], values[2]),
                UnitQuaternion::new_normalize(quaternion),
            )))
        })
        .transpose()?
        .flatten();
    Ok(Some(ParsedEqualityWeld {
        target,
        anchor,
        relpose,
        torquescale,
    }))
}

fn parse_site(
    attrs: &BTreeMap<String, String>,
    angle: AngleUnit,
) -> Result<ParsedSite, MjcfLoadError> {
    let name = attrs.get("name").cloned().unwrap_or_default();
    if attrs.contains_key("fromto") || attrs.contains_key("xyaxes") || attrs.contains_key("zaxis") {
        return Err(MjcfLoadError::Unsupported(format!(
            "site `{name}` orientation other than quat, euler or axisangle"
        )));
    }
    Ok(ParsedSite {
        name,
        pose: parse_pose(attrs, angle)?,
    })
}

#[allow(clippy::too_many_arguments)]
fn parse_empty_element(
    element: &BytesStart<'_>,
    angle: AngleUnit,
    in_worldbody: bool,
    body_stack: &mut [ParsedBody],
    roots: &mut Vec<ParsedBody>,
    world_geoms: &mut Vec<ParsedGeom>,
    generated_body: &mut usize,
    generated_joint: &mut usize,
    gravity: &mut Option<[f64; 3]>,
    defaults: &BTreeMap<String, DefaultClass>,
    meshes: &BTreeMap<String, Vec<ConvexGeometry>>,
) -> Result<(), MjcfLoadError> {
    let attrs = attributes(element)?;
    match element.name().as_ref() {
        b"option" => {
            *gravity = parse_optional_vec3(&attrs, "gravity")?.map(Into::into);
        }
        b"body" if in_worldbody => {
            let inherited = body_stack
                .last()
                .and_then(|body| body.child_class.as_deref());
            let body = parse_body(&attrs, angle, *generated_body, inherited)?;
            if let Some(class) = body.child_class.as_deref()
                && !defaults.contains_key(class)
            {
                return Err(MjcfLoadError::Invalid(format!(
                    "unknown child default class `{class}`"
                )));
            }
            *generated_body += 1;
            if let Some(parent) = body_stack.last_mut() {
                parent.children.push(body);
            } else {
                roots.push(body);
            }
        }
        b"inertial" if !body_stack.is_empty() => {
            let Some(body) = body_stack.last_mut() else {
                return Err(MjcfLoadError::Xml("inertial outside body".into()));
            };
            let inertial = parse_inertial(&attrs, angle)?;
            if body.inertial.replace(inertial).is_some() {
                return Err(MjcfLoadError::Invalid(format!(
                    "body `{}` has multiple inertials",
                    body.name
                )));
            }
        }
        b"joint" if !body_stack.is_empty() => {
            let active_class = body_stack
                .last()
                .and_then(|body| body.child_class.as_deref());
            let attrs = resolve_defaults(attrs, active_class, DefaultElement::Joint, defaults)?;
            let joint = parse_joint(&attrs, angle, *generated_joint)?;
            *generated_joint += 1;
            let Some(body) = body_stack.last_mut() else {
                return Err(MjcfLoadError::Xml("joint outside body".into()));
            };
            body.joints.push(joint);
        }
        b"freejoint" if !body_stack.is_empty() => {
            let name = attrs
                .get("name")
                .cloned()
                .unwrap_or_else(|| format!("joint_{}", *generated_joint));
            *generated_joint += 1;
            let Some(body) = body_stack.last_mut() else {
                return Err(MjcfLoadError::Xml("freejoint outside body".into()));
            };
            body.joints.push(ParsedJoint {
                name,
                kind: ParsedJointKind::Free,
                position: Vector3::zeros(),
                axis: Vector3::z(),
                limits: None,
                armature: 0.0,
                reference: 0.0,
                passive: JointPassive::default(),
                nonlinear_passive: JointNonlinearPassive::default(),
                frictionloss: 0.0,
                springdamper: None,
            });
        }
        b"geom" if in_worldbody => {
            let active_class = body_stack
                .last()
                .and_then(|body| body.child_class.as_deref());
            let attrs = resolve_defaults(attrs, active_class, DefaultElement::Geom, defaults)?;
            let geom = parse_geom(&attrs, angle, meshes)?;
            if let Some(body) = body_stack.last_mut() {
                body.geoms.push(geom);
            } else {
                world_geoms.push(geom);
            }
        }
        _ => {}
    }
    Ok(())
}

fn finish_body(
    stack: &mut Vec<ParsedBody>,
    roots: &mut Vec<ParsedBody>,
) -> Result<(), MjcfLoadError> {
    let body = stack
        .pop()
        .ok_or_else(|| MjcfLoadError::Xml("body close without open".into()))?;
    if let Some(parent) = stack.last_mut() {
        parent.children.push(body);
    } else {
        roots.push(body);
    }
    Ok(())
}

fn attributes(element: &BytesStart<'_>) -> Result<BTreeMap<String, String>, MjcfLoadError> {
    let mut values = BTreeMap::new();
    for attribute in element.attributes().with_checks(false) {
        let attribute = attribute.map_err(|error| MjcfLoadError::Xml(error.to_string()))?;
        let key = core::str::from_utf8(attribute.key.as_ref())
            .map_err(|error| MjcfLoadError::Xml(error.to_string()))?;
        let raw = core::str::from_utf8(attribute.value.as_ref())
            .map_err(|error| MjcfLoadError::Xml(error.to_string()))?;
        let value = quick_xml::escape::unescape(raw)
            .map_err(|error| MjcfLoadError::Xml(error.to_string()))?;
        let _previous = values.insert(key.to_owned(), value.into_owned());
    }
    Ok(values)
}

fn parse_compiler(
    attrs: &BTreeMap<String, String>,
    angle: &mut AngleUnit,
) -> Result<(), MjcfLoadError> {
    if let Some(sequence) = attrs.get("eulerseq")
        && !sequence.eq_ignore_ascii_case("xyz")
    {
        return Err(MjcfLoadError::Unsupported(format!(
            "compiler eulerseq `{sequence}`"
        )));
    }
    if let Some(value) = attrs.get("angle") {
        *angle = match value.as_str() {
            "degree" => AngleUnit::Degree,
            "radian" => AngleUnit::Radian,
            _ => {
                return Err(MjcfLoadError::Invalid(format!(
                    "unknown compiler angle `{value}`"
                )));
            }
        };
    }
    Ok(())
}

fn parse_body(
    attrs: &BTreeMap<String, String>,
    angle: AngleUnit,
    generated: usize,
    inherited_child_class: Option<&str>,
) -> Result<ParsedBody, MjcfLoadError> {
    Ok(ParsedBody {
        name: attrs
            .get("name")
            .cloned()
            .unwrap_or_else(|| format!("body_{generated}")),
        pose: parse_pose(attrs, angle)?,
        inertial: None,
        joints: Vec::new(),
        geoms: Vec::new(),
        sites: Vec::new(),
        children: Vec::new(),
        child_class: attrs
            .get("childclass")
            .cloned()
            .or_else(|| inherited_child_class.map(str::to_owned)),
    })
}

fn parse_inertial(
    attrs: &BTreeMap<String, String>,
    angle: AngleUnit,
) -> Result<ParsedInertial, MjcfLoadError> {
    let mass = required_scalar(attrs, "mass")?;
    if mass <= 0.0 {
        return Err(MjcfLoadError::Invalid(
            "inertial mass must be positive".into(),
        ));
    }
    let local = if let Some(values) = parse_optional_numbers(attrs, "diaginertia")? {
        if values.len() != 3 || values.iter().any(|value| *value < 0.0) {
            return Err(MjcfLoadError::Invalid("invalid diaginertia".into()));
        }
        Matrix3::from_diagonal(&Vector3::new(values[0], values[1], values[2]))
    } else if let Some(values) = parse_optional_numbers(attrs, "fullinertia")? {
        if values.len() != 6 {
            return Err(MjcfLoadError::Invalid("invalid fullinertia".into()));
        }
        Matrix3::new(
            values[0], values[3], values[4], values[3], values[1], values[5], values[4], values[5],
            values[2],
        )
    } else {
        return Err(MjcfLoadError::Invalid(
            "inertial needs diaginertia or fullinertia".into(),
        ));
    };
    if local
        .symmetric_eigen()
        .eigenvalues
        .iter()
        .any(|value| *value <= 0.0)
    {
        return Err(MjcfLoadError::Invalid(
            "inertial tensor must be positive definite".into(),
        ));
    }
    let pose = parse_pose(attrs, angle)?;
    let rotation = pose.rotation.to_rotation_matrix();
    Ok(ParsedInertial {
        mass,
        center: pose.translation.vector,
        inertia: rotation.matrix() * local * rotation.matrix().transpose(),
    })
}

fn parse_joint(
    attrs: &BTreeMap<String, String>,
    angle: AngleUnit,
    generated: usize,
) -> Result<ParsedJoint, MjcfLoadError> {
    let name = attrs
        .get("name")
        .cloned()
        .unwrap_or_else(|| format!("joint_{generated}"));
    let kind = match attrs.get("type").map(String::as_str).unwrap_or("hinge") {
        "hinge" => ParsedJointKind::Revolute,
        "slide" => ParsedJointKind::Prismatic,
        "ball" => ParsedJointKind::Spherical,
        "free" => ParsedJointKind::Free,
        other => return Err(MjcfLoadError::Unsupported(format!("joint type `{other}`"))),
    };
    let axis = parse_optional_vec3(attrs, "axis")?.unwrap_or_else(Vector3::z);
    let armature =
        parse_optional_numbers(attrs, "armature")?.map_or(Ok(0.0), |values| {
            match values.as_slice() {
                [value] if value.is_finite() && *value >= 0.0 => Ok(*value),
                _ => Err(MjcfLoadError::Invalid(format!(
                    "joint `{name}` has an invalid armature"
                ))),
            }
        })?;
    if kind == ParsedJointKind::Free && armature != 0.0 {
        return Err(MjcfLoadError::Unsupported(format!(
            "free joint `{name}` armature"
        )));
    }
    let scalar = |attribute: &str| -> Result<f64, MjcfLoadError> {
        match parse_optional_numbers(attrs, attribute)? {
            None => Ok(0.0),
            Some(values) if values.len() == 1 && values[0].is_finite() => Ok(values[0]),
            _ => Err(MjcfLoadError::Invalid(format!(
                "joint `{name}` has an invalid {attribute}"
            ))),
        }
    };
    let coefficients = |attribute: &str| -> Result<[f64; 3], MjcfLoadError> {
        match parse_optional_numbers(attrs, attribute)? {
            None => Ok([0.0; 3]),
            Some(values) if values.len() == 1 && values[0].is_finite() => Ok([values[0], 0.0, 0.0]),
            Some(values) if values.len() == 3 && values.iter().all(|value| value.is_finite()) => {
                Ok([values[0], values[1], values[2]])
            }
            _ => Err(MjcfLoadError::Invalid(format!(
                "joint `{name}` has invalid {attribute} coefficients"
            ))),
        }
    };
    let [stiffness, spring_quadratic, spring_cubic] = coefficients("stiffness")?;
    let [damping, damping_quadratic, damping_cubic] = coefficients("damping")?;
    let reference = scalar("ref")?;
    if !matches!(kind, ParsedJointKind::Revolute | ParsedJointKind::Prismatic) && reference != 0.0 {
        return Err(MjcfLoadError::Unsupported(format!(
            "joint `{name}` ref requires hinge or slide"
        )));
    }
    let reference = if kind == ParsedJointKind::Revolute {
        angle.value(reference)
    } else {
        reference
    };
    let springref = scalar("springref")?;
    let frictionloss = scalar("frictionloss")?;
    if frictionloss < 0.0 {
        return Err(MjcfLoadError::Invalid(format!(
            "joint `{name}` has negative frictionloss"
        )));
    }
    if kind == ParsedJointKind::Free && frictionloss != 0.0 {
        return Err(MjcfLoadError::Unsupported(format!(
            "free joint `{name}` frictionloss"
        )));
    }
    if stiffness < 0.0 || damping < 0.0 {
        return Err(MjcfLoadError::Invalid(format!(
            "joint `{name}` has negative passive coefficients"
        )));
    }
    if kind == ParsedJointKind::Free && (stiffness != 0.0 || damping != 0.0 || springref != 0.0) {
        return Err(MjcfLoadError::Unsupported(format!(
            "free joint `{name}` passive coefficients"
        )));
    }
    if kind == ParsedJointKind::Spherical && springref != 0.0 {
        return Err(MjcfLoadError::Unsupported(format!(
            "ball joint `{name}` springref"
        )));
    }
    let passive = JointPassive {
        stiffness,
        damping,
        rest_position: if kind == ParsedJointKind::Revolute {
            angle.value(springref)
        } else {
            springref
        },
    };
    if matches!(kind, ParsedJointKind::Free | ParsedJointKind::Spherical)
        && [
            spring_quadratic,
            spring_cubic,
            damping_quadratic,
            damping_cubic,
        ]
        .iter()
        .any(|value| *value != 0.0)
    {
        return Err(MjcfLoadError::Unsupported(format!(
            "joint `{name}` nonlinear passive coefficients require hinge or slide"
        )));
    }
    let nonlinear_passive = JointNonlinearPassive {
        spring_quadratic,
        spring_cubic,
        damping_quadratic,
        damping_cubic,
    };
    let springdamper = match parse_optional_numbers(attrs, "springdamper")? {
        None => None,
        Some(values) if values.len() == 2 && values == [0.0, 0.0] => None,
        Some(values)
            if values.len() == 2
                && values[0].is_finite()
                && values[0] > 0.0
                && values[1].is_finite()
                && values[1] > 0.0 =>
        {
            Some((values[0], values[1]))
        }
        _ => {
            return Err(MjcfLoadError::Invalid(format!(
                "joint `{name}` has an invalid springdamper"
            )));
        }
    };
    if kind == ParsedJointKind::Free && springdamper.is_some() {
        return Err(MjcfLoadError::Unsupported(format!(
            "free joint `{name}` springdamper"
        )));
    }
    if matches!(kind, ParsedJointKind::Revolute | ParsedJointKind::Prismatic)
        && (!axis.norm().is_finite() || axis.norm() <= 1e-12)
    {
        return Err(MjcfLoadError::Invalid(format!(
            "joint `{name}` has a zero axis"
        )));
    }
    let limited = attrs.get("limited").map(String::as_str);
    let limits = if limited == Some("false") {
        None
    } else if let Some(values) = parse_optional_numbers(attrs, "range")? {
        if values.len() != 2 {
            return Err(MjcfLoadError::Invalid(format!(
                "joint `{name}` has an invalid range"
            )));
        }
        if kind == ParsedJointKind::Spherical {
            return Err(MjcfLoadError::Unsupported(format!(
                "ball joint `{name}` range"
            )));
        }
        let mut lower = values[0];
        let mut upper = values[1];
        if kind == ParsedJointKind::Revolute {
            lower = angle.value(lower);
            upper = angle.value(upper);
        }
        Some((lower, upper))
    } else {
        None
    };
    Ok(ParsedJoint {
        name,
        kind,
        position: parse_optional_vec3(attrs, "pos")?.unwrap_or_else(Vector3::zeros),
        axis,
        limits,
        armature,
        reference,
        passive,
        nonlinear_passive,
        frictionloss,
        springdamper,
    })
}

fn parse_geom(
    attrs: &BTreeMap<String, String>,
    angle: AngleUnit,
    meshes: &BTreeMap<String, Vec<ConvexGeometry>>,
) -> Result<ParsedGeom, MjcfLoadError> {
    let geom_type = attrs.get("type").map(String::as_str).unwrap_or("sphere");
    let sizes = parse_optional_numbers(attrs, "size")?.unwrap_or_default();
    let mut pose = parse_pose(attrs, angle)?;
    let mut fromto_half_height = None;
    if let Some(values) = parse_optional_numbers(attrs, "fromto")? {
        if values.len() != 6 {
            return Err(MjcfLoadError::Invalid("geom fromto needs 6 values".into()));
        }
        let start = Vector3::new(values[0], values[1], values[2]);
        let end = Vector3::new(values[3], values[4], values[5]);
        let segment = end - start;
        let length = segment.norm();
        if !length.is_finite() || length <= 1e-12 {
            return Err(MjcfLoadError::Invalid("geom fromto is degenerate".into()));
        }
        let rotation = UnitQuaternion::rotation_between(&Vector3::z(), &(segment / length))
            .ok_or_else(|| MjcfLoadError::Invalid("cannot orient geom fromto".into()))?;
        pose = Isometry3::from_parts(Translation3::from((start + end) * 0.5), rotation);
        fromto_half_height = Some(length * 0.5);
    }
    let positive = |index: usize| -> Result<f64, MjcfLoadError> {
        let value = *sizes
            .get(index)
            .ok_or_else(|| MjcfLoadError::Invalid(format!("{geom_type} geom size is missing")))?;
        if !value.is_finite() || value <= 0.0 {
            return Err(MjcfLoadError::Invalid(format!(
                "{geom_type} geom size must be positive"
            )));
        }
        Ok(value)
    };
    let kind = match geom_type {
        "sphere" => ParsedGeomKind::Sphere {
            radius: positive(0)?,
        },
        "box" => ParsedGeomKind::Box {
            half_extents: Vector3::new(positive(0)?, positive(1)?, positive(2)?),
        },
        "capsule" => ParsedGeomKind::Capsule {
            radius: positive(0)?,
            half_height: match fromto_half_height {
                Some(value) => value,
                None => positive(1)?,
            },
        },
        "cylinder" => ParsedGeomKind::Cylinder {
            radius: positive(0)?,
            half_height: match fromto_half_height {
                Some(value) => value,
                None => positive(1)?,
            },
        },
        "mesh" => {
            let name = attrs.get("mesh").ok_or_else(|| {
                MjcfLoadError::Invalid("mesh geom needs a mesh asset name".into())
            })?;
            let geometry = meshes
                .get(name)
                .cloned()
                .ok_or_else(|| MjcfLoadError::Invalid(format!("unknown mesh asset `{name}`")))?;
            ParsedGeomKind::Convex {
                geometries: geometry,
            }
        }
        "plane" => ParsedGeomKind::Plane,
        other => return Err(MjcfLoadError::Unsupported(format!("geom type `{other}`"))),
    };
    let mass = optional_scalar(attrs, "mass")?;
    if mass.is_some_and(|value| value < 0.0) {
        return Err(MjcfLoadError::Invalid(
            "geom mass must be non-negative".into(),
        ));
    }
    let density = optional_scalar(attrs, "density")?.unwrap_or(1000.0);
    if density < 0.0 {
        return Err(MjcfLoadError::Invalid(
            "geom density must be non-negative".into(),
        ));
    }
    Ok(ParsedGeom {
        pose,
        kind,
        mass,
        density,
    })
}

fn parse_pose(
    attrs: &BTreeMap<String, String>,
    angle: AngleUnit,
) -> Result<Isometry3<f64>, MjcfLoadError> {
    let translation = parse_optional_vec3(attrs, "pos")?.unwrap_or_else(Vector3::zeros);
    let orientation_count = ["quat", "euler", "axisangle"]
        .into_iter()
        .filter(|key| attrs.contains_key(*key))
        .count();
    if orientation_count > 1 {
        return Err(MjcfLoadError::Invalid(
            "an element has multiple orientation attributes".into(),
        ));
    }
    let rotation = if let Some(values) = parse_optional_numbers(attrs, "quat")? {
        if values.len() != 4 {
            return Err(MjcfLoadError::Invalid("quat needs 4 values".into()));
        }
        let quaternion = Quaternion::new(values[0], values[1], values[2], values[3]);
        if !quaternion.norm().is_finite() || quaternion.norm() <= 1e-12 {
            return Err(MjcfLoadError::Invalid("zero quaternion".into()));
        }
        UnitQuaternion::new_normalize(quaternion)
    } else if let Some(values) = parse_optional_numbers(attrs, "euler")? {
        if values.len() != 3 {
            return Err(MjcfLoadError::Invalid("euler needs 3 values".into()));
        }
        UnitQuaternion::from_euler_angles(
            angle.value(values[0]),
            angle.value(values[1]),
            angle.value(values[2]),
        )
    } else if let Some(values) = parse_optional_numbers(attrs, "axisangle")? {
        if values.len() != 4 {
            return Err(MjcfLoadError::Invalid("axisangle needs 4 values".into()));
        }
        let axis = Vector3::new(values[0], values[1], values[2]);
        if !axis.norm().is_finite() || axis.norm() <= 1e-12 {
            return Err(MjcfLoadError::Invalid("zero axisangle axis".into()));
        }
        UnitQuaternion::from_axis_angle(&Unit::new_normalize(axis), angle.value(values[3]))
    } else {
        UnitQuaternion::identity()
    };
    Ok(Isometry3::from_parts(
        Translation3::from(translation),
        rotation,
    ))
}

fn required_scalar(attrs: &BTreeMap<String, String>, key: &str) -> Result<f64, MjcfLoadError> {
    let values = parse_optional_numbers(attrs, key)?
        .ok_or_else(|| MjcfLoadError::Invalid(format!("missing `{key}` attribute")))?;
    if values.len() != 1 {
        return Err(MjcfLoadError::Invalid(format!("`{key}` needs one value")));
    }
    Ok(values[0])
}

fn optional_scalar(
    attrs: &BTreeMap<String, String>,
    key: &str,
) -> Result<Option<f64>, MjcfLoadError> {
    let Some(values) = parse_optional_numbers(attrs, key)? else {
        return Ok(None);
    };
    if values.len() != 1 {
        return Err(MjcfLoadError::Invalid(format!("`{key}` needs one value")));
    }
    Ok(Some(values[0]))
}

fn parse_optional_vec3(
    attrs: &BTreeMap<String, String>,
    key: &str,
) -> Result<Option<Vector3<f64>>, MjcfLoadError> {
    let Some(values) = parse_optional_numbers(attrs, key)? else {
        return Ok(None);
    };
    if values.len() != 3 {
        return Err(MjcfLoadError::Invalid(format!("`{key}` needs 3 values")));
    }
    Ok(Some(Vector3::new(values[0], values[1], values[2])))
}

fn parse_optional_numbers(
    attrs: &BTreeMap<String, String>,
    key: &str,
) -> Result<Option<Vec<f64>>, MjcfLoadError> {
    attrs
        .get(key)
        .map(|text| parse_number_text(key, text))
        .transpose()
}

fn parse_number_text(key: &str, text: &str) -> Result<Vec<f64>, MjcfLoadError> {
    let values = text
        .split_ascii_whitespace()
        .map(|part| {
            part.parse::<f64>()
                .map_err(|error| MjcfLoadError::Invalid(format!("invalid `{key}` number: {error}")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if values.iter().all(|value| value.is_finite()) {
        Ok(values)
    } else {
        Err(MjcfLoadError::Invalid(format!("non-finite `{key}` value")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROBOT: &str = r#"
        <mujoco model="loader-fixture">
          <compiler angle="degree"/>
          <option gravity="0 0 -3"/>
          <worldbody>
            <geom type="plane" size="10 10 0.1"/>
            <geom type="box" size="1 2 0.25" pos="3 0 0.25"/>
            <body name="base" pos="0 0 2">
              <freejoint name="base_free"/>
              <inertial mass="2" pos="0 0 0.1" diaginertia="1 2 3"/>
              <geom type="box" size="0.4 0.3 0.2"/>
              <body name="arm" pos="0 0 0.5">
                <joint name="shoulder" type="hinge" pos="0 0 0.1" axis="0 1 0" range="-90 90"/>
                <inertial mass="1" pos="0 0 0.2" diaginertia="0.1 0.2 0.3"/>
                <geom type="capsule" size="0.05 0.2" pos="0 0 0.2"/>
                <body name="slider" pos="0 0 0.4">
                  <joint name="extension" type="slide" axis="0 0 1" range="0 0.3"/>
                  <inertial mass="0.5" diaginertia="0.01 0.01 0.01"/>
                  <geom type="sphere" size="0.08"/>
                  <body name="wrist" pos="0 0 0.1">
                    <joint name="wrist_ball" type="ball"/>
                    <inertial mass="0.2" diaginertia="0.01 0.01 0.01"/>
                    <geom type="cylinder" size="0.04 0.06"/>
                  </body>
                </body>
              </body>
            </body>
          </worldbody>
        </mujoco>
    "#;

    #[test]
    fn loads_hierarchy_primitives_and_all_joint_kinds() {
        let loaded = load_mjcf_str(ROBOT, MjcfLoadOptions::default()).unwrap();
        assert_eq!(loaded.model_name.as_deref(), Some("loader-fixture"));
        assert_eq!(loaded.link_names, ["base", "arm", "slider", "wrist"]);
        assert!(loaded.world.floating);
        assert_eq!(loaded.world.articulation.dof(), 5);
        assert_eq!(loaded.joints.len(), 3);
        assert_eq!(loaded.joints[0].dofs, 0..1);
        assert_eq!(loaded.joints[1].dofs, 1..2);
        assert_eq!(loaded.joints[2].dofs, 2..5);
        assert_eq!(loaded.world.boxes.len(), 1);
        assert_eq!(loaded.world.cylinders.len(), 2);
        assert_eq!(loaded.world.colliders.len(), 3);
        assert_eq!(loaded.world.scene_bodies.len(), 1);
        assert!((loaded.world.root_pose.translation.vector.z - 2.0).abs() < 1e-12);

        let pose = loaded
            .world
            .articulation
            .pose(loaded.world.root_pose, loaded.world.positions.as_slice())
            .unwrap();
        assert!((pose.links[1].translation.vector.z - 2.6).abs() < 1e-12);
        assert_eq!(loaded.world.params().gravity, [0.0, 0.0, -3.0]);
    }

    #[test]
    fn converts_degree_joint_limits_and_nonzero_joint_position() {
        let loaded = load_mjcf_str(ROBOT, MjcfLoadOptions::default()).unwrap();
        let limit = loaded.world.articulation.joint_limit(0).unwrap();
        assert!((limit.0 + core::f64::consts::FRAC_PI_2).abs() < 1e-12);
        assert!((limit.1 - core::f64::consts::FRAC_PI_2).abs() < 1e-12);
        assert!((loaded.world.articulation.joint_limit(1).unwrap().1 - 0.3).abs() < 1e-12);
        assert!((loaded.world.cylinders[0].origin.translation.vector.z - 0.1).abs() < 1e-12);
    }

    #[test]
    fn fromto_orients_capsules_and_world_shapes_are_static() {
        let xml = r#"
            <mujoco><worldbody>
              <body name="root"><inertial mass="1" diaginertia="1 1 1"/>
                <geom type="capsule" size="0.1" fromto="-1 0 0 1 0 0"/>
              </body>
            </worldbody></mujoco>
        "#;
        let loaded = load_mjcf_str(xml, MjcfLoadOptions::default()).unwrap();
        let cylinder = &loaded.world.cylinders[0];
        assert!((cylinder.half_height - 1.0).abs() < 1e-12);
        assert!((cylinder.origin.rotation * Vector3::z() - Vector3::x()).norm() < 1e-12);
        assert_eq!(loaded.world.colliders.len(), 2);
    }

    #[test]
    fn rejects_ambiguous_and_unsupported_models() {
        let no_roots = r#"<mujoco><worldbody/></mujoco>"#;
        assert!(matches!(
            load_mjcf_str(no_roots, MjcfLoadOptions::default()),
            Err(MjcfLoadError::Unsupported(_))
        ));
        let combined_free = r#"
            <mujoco><worldbody>
              <body name="a"><freejoint/><joint type="hinge"/></body>
              <body name="b"/>
            </worldbody></mujoco>
        "#;
        assert!(matches!(
            load_mjcf_str(combined_free, MjcfLoadOptions::default()),
            Err(MjcfLoadError::Unsupported(_))
        ));
        let mesh = r#"
            <mujoco><asset><mesh name="asset" file="model.obj"/></asset>
              <worldbody><body><geom type="mesh" mesh="asset"/></body></worldbody></mujoco>
        "#;
        assert!(matches!(
            load_mjcf_str(mesh, MjcfLoadOptions::default()),
            Err(MjcfLoadError::Unsupported(_))
        ));
    }

    #[test]
    fn multiple_fixed_roots_preserve_poses_and_joint_motion() {
        let xml = r#"
            <mujoco><option gravity="0 0 0"/><worldbody>
              <body name="left" pos="-2 0 1">
                <geom type="sphere" size="0.2"/>
                <body name="arm" pos="0 0 1">
                  <joint name="slide" type="slide" axis="0 0 1"/>
                  <geom type="sphere" size="0.1"/>
                </body>
              </body>
              <body name="right" pos="2 0 1"><geom type="box" size="0.2 0.2 0.2"/></body>
            </worldbody></mujoco>
        "#;
        let options = MjcfLoadOptions {
            root_pose: Isometry3::translation(1.0, 0.0, 0.0),
            ..Default::default()
        };
        let mut loaded = load_mjcf_str(xml, options).unwrap();
        assert_eq!(loaded.link_names, ["__mjcf_world", "left", "arm", "right"]);
        assert_eq!(loaded.world.articulation.dof(), 1);
        assert_eq!(loaded.joints.len(), 3);
        assert_eq!(loaded.world.colliders.len(), 2);
        assert_eq!(loaded.world.boxes.len(), 1);
        let poses = loaded.world.link_poses().unwrap();
        assert!((poses[1].translation.vector - Vector3::new(-1.0, 0.0, 1.0)).norm() < 1e-12);
        assert!((poses[2].translation.vector - Vector3::new(-1.0, 0.0, 2.0)).norm() < 1e-12);
        assert!((poses[3].translation.vector - Vector3::new(3.0, 0.0, 1.0)).norm() < 1e-12);
        loaded.world.step(0.01, &[1.0]).unwrap();
        assert!(loaded.world.velocities[0] > 0.0);
    }

    #[test]
    fn single_movable_root_uses_serial_joint_chain() {
        let xml = r#"
            <mujoco><option gravity="0 0 0"/><worldbody>
              <body name="platform" pos="0 0 2">
                <joint name="translation" type="slide" axis="1 0 0"/>
                <joint name="rotation" type="hinge" axis="0 0 1"/>
                <geom type="box" size="0.2 0.2 0.2" mass="1"/>
              </body>
            </worldbody></mujoco>
        "#;
        let mut loaded = load_mjcf_str(xml, MjcfLoadOptions::default()).unwrap();
        assert_eq!(
            loaded.link_names,
            ["__mjcf_world", "platform@translation", "platform"]
        );
        assert_eq!(loaded.world.articulation.dof(), 2);
        assert_eq!(loaded.joints[0].dofs, 0..1);
        assert_eq!(loaded.joints[1].dofs, 1..2);
        assert_eq!(loaded.world.boxes.len(), 1);
        assert!((loaded.world.link_poses().unwrap()[2].translation.vector.z - 2.0).abs() < 1e-12);
        loaded.world.step(0.01, &[1.0, 0.0]).unwrap();
        assert!(loaded.world.velocities[0] > 0.0);
    }

    #[test]
    fn multiple_free_roots_have_independent_six_dof_states() {
        let xml = r#"
            <mujoco><option gravity="0 0 0"/><worldbody>
              <body name="a" pos="-3 0 2">
                <freejoint name="free_a"/>
                <geom type="sphere" size="0.2" mass="1"/>
                <body name="sensor" pos="0 0 0.5"/>
              </body>
              <body name="b" pos="3 0 2">
                <freejoint name="free_b"/>
                <geom type="sphere" size="0.2" mass="1"/>
              </body>
            </worldbody></mujoco>
        "#;
        let mut loaded = load_mjcf_str(xml, MjcfLoadOptions::default()).unwrap();
        assert!(!loaded.world.floating);
        assert_eq!(loaded.world.articulation.dof(), 12);
        assert_eq!(loaded.joints[0].name, "free_a/tx");
        assert_eq!(loaded.joints[3].dofs, 3..6);
        assert_eq!(loaded.joints[5].name, "free_b/tx");
        assert_eq!(loaded.joints[8].dofs, 9..12);
        assert_eq!(loaded.world.colliders.len(), 2);
        let sensor = loaded
            .link_names
            .iter()
            .position(|name| name == "sensor")
            .unwrap();
        assert!(
            (loaded.world.link_poses().unwrap()[sensor]
                .translation
                .vector
                - Vector3::new(-3.0, 0.0, 2.5))
            .norm()
                < 1e-12
        );
        let mut forces = [0.0; 12];
        forces[0] = 1.0;
        loaded.world.step(0.01, &forces).unwrap();
        assert!(loaded.world.velocities[0] > 0.0);
        assert!(loaded.world.velocities[6].abs() < 1e-12);
    }

    #[test]
    fn multiple_free_roots_resolve_mutual_contact() {
        let xml = r#"
            <mujoco><option gravity="0 0 0"/><worldbody>
              <body name="a" pos="-0.15 0 1">
                <freejoint/><geom type="sphere" size="0.2" mass="1"/>
              </body>
              <body name="b" pos="0.15 0 1">
                <freejoint/><geom type="sphere" size="0.2" mass="1"/>
              </body>
            </worldbody></mujoco>
        "#;
        let mut loaded = load_mjcf_str(xml, MjcfLoadOptions::default()).unwrap();
        loaded.world.step(0.01, &[0.0; 12]).unwrap();
        assert!(loaded.world.velocities[0] < 0.0);
        assert!(loaded.world.velocities[6] > 0.0);

        #[cfg(feature = "gpu-contact")]
        if let Ok(context) = crate::gpu_contact_pipeline::GpuContactDevice::new() {
            let mut gpu = load_mjcf_str(xml, MjcfLoadOptions::default()).unwrap();
            gpu.world.step_gpu(0.01, &[0.0; 12], &context).unwrap();
            assert!((gpu.world.velocities[0] - loaded.world.velocities[0]).abs() < 1e-4);
            assert!((gpu.world.velocities[6] - loaded.world.velocities[6]).abs() < 1e-4);
        }
    }

    #[test]
    fn rotated_free_root_translates_in_world_axes() {
        let xml = r#"
            <mujoco><worldbody>
              <body name="rotated" pos="0 0 2" euler="0 0 90">
                <freejoint/><geom type="sphere" size="0.2" mass="1"/>
              </body>
              <body name="anchor" pos="4 0 2"/>
            </worldbody></mujoco>
        "#;
        let mut loaded = load_mjcf_str(xml, MjcfLoadOptions::default()).unwrap();
        let body = loaded
            .link_names
            .iter()
            .position(|name| name == "rotated")
            .unwrap();
        loaded.world.positions[0] = 1.0;
        let pose = loaded.world.link_poses().unwrap()[body];
        assert!((pose.translation.vector - Vector3::new(1.0, 0.0, 2.0)).norm() < 1e-12);
        assert!((pose.rotation * Vector3::x() - Vector3::y()).norm() < 1e-12);
    }

    #[test]
    fn loaded_world_steps_with_contacts() {
        let mut loaded = load_mjcf_str(ROBOT, MjcfLoadOptions::default()).unwrap();
        loaded.world.root_pose.translation.vector.z = 0.15;
        loaded.world.base_linear_velocity.z = -1.0;
        loaded
            .world
            .step(0.002, &[0.0; 5])
            .expect("loaded world should step");
        assert!(
            loaded
                .world
                .base_linear_velocity
                .iter()
                .all(|value| value.is_finite())
        );
    }

    #[test]
    fn self_closing_radian_compiler_preserves_radian_limits() {
        let xml = r#"
            <mujoco><compiler angle="radian"/><worldbody>
              <body name="root"><inertial mass="1" diaginertia="1 1 1"/>
                <body name="child"><joint range="-1 2"/>
                  <inertial mass="1" diaginertia="1 1 1"/>
                </body>
              </body>
            </worldbody></mujoco>
        "#;
        let loaded = load_mjcf_str(xml, MjcfLoadOptions::default()).unwrap();
        assert_eq!(loaded.world.articulation.joint_limit(0), Some((-1.0, 2.0)));
    }

    #[test]
    fn explicit_unlimited_joint_ignores_range() {
        let xml = r#"
            <mujoco><worldbody>
              <body name="root"><inertial mass="1" diaginertia="1 1 1"/>
                <body name="child"><joint limited="false" range="-10 10"/>
                  <inertial mass="1" diaginertia="1 1 1"/>
                </body>
              </body>
            </worldbody></mujoco>
        "#;
        let loaded = load_mjcf_str(xml, MjcfLoadOptions::default()).unwrap();
        assert_eq!(loaded.world.articulation.joint_limit(0), None);
    }

    #[test]
    fn loads_joint_armature_from_defaults_and_overrides() {
        let xml = r#"
            <mujoco>
              <default><joint armature="0.2"/></default>
              <worldbody><body name="root">
                <inertial mass="1" diaginertia="1 1 1"/>
                <body name="hinge">
                  <joint type="hinge" armature="0.5"/>
                  <inertial mass="1" diaginertia="1 1 1"/>
                  <body name="slider">
                    <joint type="slide"/>
                    <inertial mass="1" diaginertia="1 1 1"/>
                  </body>
                </body>
              </body></worldbody>
            </mujoco>
        "#;
        let loaded = load_mjcf_str(xml, MjcfLoadOptions::default()).unwrap();
        let mut baseline = loaded.world.articulation.clone();
        assert_eq!(baseline.joint_armature(0), Some([0.5].as_slice()));
        assert_eq!(baseline.joint_armature(1), Some([0.2].as_slice()));
        baseline.set_joint_armature(0, &[0.0]).unwrap();
        baseline.set_joint_armature(1, &[0.0]).unwrap();
        let pose = loaded
            .world
            .articulation
            .pose(Isometry3::identity(), &[0.0, 0.0])
            .unwrap();
        let actual = loaded
            .world
            .articulation
            .dynamics(&pose, Vector3::zeros())
            .unwrap();
        let without = baseline.dynamics(&pose, Vector3::zeros()).unwrap();
        assert!((actual.mass[(0, 0)] - without.mass[(0, 0)] - 0.5).abs() < 1e-12);
        assert!((actual.mass[(1, 1)] - without.mass[(1, 1)] - 0.2).abs() < 1e-12);
        assert!((actual.mass[(0, 1)] - without.mass[(0, 1)]).abs() < 1e-12);

        let invalid = xml.replace("armature=\"0.5\"", "armature=\"-0.5\"");
        assert!(load_mjcf_str(&invalid, MjcfLoadOptions::default()).is_err());
    }

    #[test]
    fn loads_passive_joint_coefficients_from_defaults() {
        let xml = r#"
            <mujoco><compiler angle="degree"/>
              <default><joint damping="2" stiffness="3" springref="90" frictionloss="0.5"/></default>
              <worldbody><body name="root">
                <inertial mass="1" diaginertia="1 1 1"/>
                <body name="hinge"><joint type="hinge" stiffness="4" frictionloss="0.7"/>
                  <inertial mass="1" diaginertia="1 1 1"/>
                  <body name="slider"><joint type="slide" springref="0.5"/>
                    <inertial mass="1" diaginertia="1 1 1"/>
                  </body>
                </body>
              </body></worldbody>
            </mujoco>
        "#;
        let loaded = load_mjcf_str(xml, MjcfLoadOptions::default()).unwrap();
        assert_eq!(
            loaded.world.joint_passive(0),
            Some(JointPassive {
                stiffness: 4.0,
                damping: 2.0,
                rest_position: core::f64::consts::FRAC_PI_2,
            })
        );
        assert_eq!(
            loaded.world.joint_passive(1),
            Some(JointPassive {
                stiffness: 3.0,
                damping: 2.0,
                rest_position: 0.5,
            })
        );
        assert_eq!(loaded.world.joint_friction(0), Some(0.7));
        assert_eq!(loaded.world.joint_friction(1), Some(0.5));
        assert!(
            load_mjcf_str(
                &xml.replace("damping=\"2\"", "damping=\"-1\""),
                MjcfLoadOptions::default()
            )
            .is_err()
        );
        assert!(
            load_mjcf_str(
                &xml.replace("frictionloss=\"0.5\"", "frictionloss=\"-1\""),
                MjcfLoadOptions::default()
            )
            .is_err()
        );
    }

    #[test]
    fn equality_connect_and_weld_compile_body_and_site_frames() {
        let xml = r#"<mujoco><option gravity="0 0 0"/><worldbody>
          <site name="world_site" pos="2 3 4"/>
          <body name="root" pos="1 0 0"><inertial mass="1" diaginertia="1 1 1"/>
            <body name="left" pos="0 1 0">
              <joint name="left_joint" type="hinge" pos="0.2 0 0"/>
              <inertial mass="1" diaginertia="1 1 1"/>
              <site name="left_site" pos="0.4 0 0"/>
            </body>
            <body name="right" pos="0 -1 0">
              <joint name="right_joint" type="slide" axis="1 0 0"/>
              <inertial mass="1" diaginertia="1 1 1"/>
              <site name="right_site" pos="0 0.5 0"/>
            </body>
          </body></worldbody>
          <equality>
            <connect body1="left" body2="right" anchor="0.4 0 0"/>
            <connect body1="left" anchor="0.4 0 0"/>
            <connect site1="left_site" site2="right_site"/>
            <connect site1="world_site" site2="left_site"/>
            <weld body1="left" body2="right"/>
            <weld body1="left"/>
            <weld site1="left_site" site2="right_site"/>
            <weld site1="world_site" site2="left_site"/>
            <connect body1="left" anchor="0 0 0" active="false"/>
          </equality></mujoco>"#;
        let loaded = load_mjcf_str(xml, MjcfLoadOptions::default()).unwrap();
        let points = loaded.world.link_point_constraints();
        assert_eq!(points.len(), 4);
        assert_eq!(points[0].link_a, 1);
        assert_eq!(points[0].link_b, Some(2));
        assert_eq!(points[0].point_a, [0.2, 0.0, 0.0]);
        assert!((points[0].point_b[0] - 0.4).abs() < 1e-12);
        assert_eq!(&points[0].point_b[1..], &[2.0, 0.0]);
        assert_eq!(points[1].link_b, None);
        assert_eq!(points[1].point_b, [1.4, 1.0, 0.0]);
        assert_eq!(points[2].point_a, [0.2, 0.0, 0.0]);
        assert_eq!(points[2].point_b, [0.0, 0.5, 0.0]);
        assert_eq!(points[3].link_a, 1);
        assert_eq!(points[3].link_b, None);
        assert_eq!(points[3].point_b, [2.0, 3.0, 4.0]);
        let fixed = loaded.world.link_fixed_constraints();
        assert_eq!(fixed.len(), 4);
        assert_eq!(fixed[0].link_b, Some(2));
        assert_eq!(fixed[1].link_b, None);
        assert_eq!(fixed[2].frame_a.translation.vector.x, 0.2);
        assert_eq!(fixed[2].frame_b.translation.vector.y, 0.5);
        assert_eq!(fixed[3].link_b, None);
        assert_eq!(
            fixed[3].frame_b.translation.vector,
            Vector3::new(2.0, 3.0, 4.0)
        );
    }

    #[test]
    fn equality_connect_and_weld_reject_unrepresentable_attributes() {
        let base = r#"<mujoco><worldbody><body name="root">
            <inertial mass="1" diaginertia="1 1 1"/>
            <site name="site"/>
          </body></worldbody><equality>{}</equality></mujoco>"#;
        for equality in [
            r#"<connect body1="root"/>"#,
            r#"<connect site1="site" site2="missing" anchor="0 0 0"/>"#,
            r#"<weld body1="root" relpose="0 0 0 1 0 0"/>"#,
            r#"<weld site1="site" site2="missing" anchor="0 0 0"/>"#,
            r#"<weld body1="root" torquescale="2"/>"#,
            r#"<weld body1="root" torquescale="-1"/>"#,
            r#"<weld body1="missing"/>"#,
            r#"<connect body1="root" anchor="0 0 0" solref="0.02 1"/>"#,
        ] {
            let xml = base.replace("{}", equality);
            assert!(load_mjcf_str(&xml, MjcfLoadOptions::default()).is_err());
        }
    }

    #[test]
    fn equality_weld_uses_body2_anchor_and_explicit_relative_pose() {
        let xml = r#"<mujoco><option gravity="0 0 0"/><worldbody>
          <body name="root"><inertial mass="1" diaginertia="1 1 1"/>
            <body name="first" pos="1 0 0">
              <joint type="slide" pos="0.2 0 0" axis="1 0 0"/>
              <inertial mass="1" diaginertia="1 1 1"/>
            </body>
            <body name="second" pos="3 0 0" quat="0.7071067811865476 0 0 0.7071067811865476">
              <joint type="slide" axis="1 0 0"/>
              <inertial mass="1" diaginertia="1 1 1"/>
            </body>
          </body></worldbody><equality>
            <weld body1="first" body2="second" anchor="0 1 0" torquescale="1"/>
            <weld body1="first" body2="second" anchor="0 1 0"
                  relpose="0.5 0 0 0.7071067811865476 0 0 0.7071067811865476"/>
            <weld body1="first" body2="second" anchor="0 1 0"
                  relpose="99 0 0 0 0 0 0"/>
            <weld body1="first" body2="second" anchor="0 1 0" torquescale="0"/>
          </equality></mujoco>"#;
        let loaded = load_mjcf_str(xml, MjcfLoadOptions::default()).unwrap();
        let constraints = loaded.world.link_fixed_constraints();
        assert_eq!(constraints.len(), 3);
        assert_eq!(constraints[0].link_a, 1);
        assert_eq!(constraints[0].link_b, Some(2));
        assert!((constraints[0].frame_a.translation.vector.x - 0.8).abs() < 1e-12);
        assert!(constraints[0].frame_a.translation.vector.y.abs() < 1e-12);
        assert!(
            (constraints[0].frame_a.rotation.angle() - core::f64::consts::FRAC_PI_2).abs() < 1e-12
        );
        assert_eq!(constraints[0].frame_b.translation.vector, Vector3::y());
        assert!((constraints[1].frame_a.translation.vector.x - 0.3).abs() < 1e-12);
        assert!(
            (constraints[1].frame_a.rotation.angle() - core::f64::consts::FRAC_PI_2).abs() < 1e-12
        );
        assert_eq!(constraints[1].frame_b.translation.vector, Vector3::y());
        assert_eq!(constraints[2], constraints[0]);
        let points = loaded.world.link_point_constraints();
        assert_eq!(points.len(), 1);
        assert!((points[0].point_a[0] - constraints[0].frame_a.translation.vector.x).abs() < 1e-12);
        assert!((points[0].point_a[1] - constraints[0].frame_a.translation.vector.y).abs() < 1e-12);
        assert_eq!(points[0].point_b, [0.0, 1.0, 0.0]);
    }

    #[test]
    fn equality_weld_world_anchor_respects_root_pose() {
        let xml = r#"<mujoco><worldbody><body name="root" pos="1 0 0">
            <inertial mass="1" diaginertia="1 1 1"/>
          </body></worldbody><equality>
            <weld body1="root" anchor="2 0 0"/>
          </equality></mujoco>"#;
        let options = MjcfLoadOptions {
            root_pose: Isometry3::translation(5.0, 0.0, 0.0),
            ..MjcfLoadOptions::default()
        };
        let loaded = load_mjcf_str(xml, options).unwrap();
        let weld = loaded.world.link_fixed_constraints()[0];
        assert_eq!(weld.frame_a.translation.vector, Vector3::x());
        assert_eq!(weld.frame_b.translation.vector, Vector3::new(7.0, 0.0, 0.0));
    }

    #[test]
    fn equality_weld_explicit_orientation_drives_joint_toward_target() {
        let xml = r#"<mujoco><option gravity="0 0 0"/><worldbody>
          <body name="rotor"><joint type="hinge" axis="0 0 1"/>
            <inertial mass="1" diaginertia="1 1 1"/>
          </body></worldbody><equality>
            <weld body1="rotor" relpose="0 0 0 0.7071067811865476 0 0 0.7071067811865476"/>
          </equality></mujoco>"#;
        let mut loaded = load_mjcf_str(xml, MjcfLoadOptions::default()).unwrap();
        loaded.world.step(0.001, &[0.0]).unwrap();
        assert!(loaded.world.velocities[0] < 0.0);
        assert!(loaded.world.positions[0] < 0.0);
    }

    #[test]
    fn equality_joint_uses_quartic_position_and_derivative() {
        let xml = r#"<mujoco><option gravity="0 0 0"/><worldbody>
          <body name="root"><inertial mass="1" diaginertia="1 1 1"/>
            <body name="source"><joint name="source" type="slide" axis="1 0 0"/>
              <inertial mass="1" diaginertia="1 1 1"/></body>
            <body name="follower"><joint name="follower" type="slide" axis="0 1 0"/>
              <inertial mass="1" diaginertia="1 1 1"/></body>
          </body></worldbody>
          <equality><joint joint1="follower" joint2="source" polycoef="0 2 1 0 0.5"/></equality>
        </mujoco>"#;
        let mut loaded = load_mjcf_str(xml, MjcfLoadOptions::default()).unwrap();
        let equality = loaded.world.joint_polynomial_couplings()[0];
        assert_eq!(equality.follower, 1);
        assert_eq!(equality.source, Some(0));
        assert_eq!(equality.coefficients, [0.0, 2.0, 1.0, 0.0, 0.5]);
        let snapshot = loaded.world.snapshot();
        loaded
            .world
            .set_joint_polynomial_couplings(Vec::new())
            .unwrap();
        loaded.world.restore_snapshot(&snapshot).unwrap();
        assert_eq!(loaded.world.joint_polynomial_couplings(), &[equality]);
        loaded.world.positions[0] = 0.5;
        loaded.world.positions[1] = 1.28125;
        loaded.world.step(0.001, &[1.0, 0.0]).unwrap();
        assert!((loaded.world.velocities[1] - 3.25 * loaded.world.velocities[0]).abs() < 1e-8);
        #[cfg(feature = "gpu-contact")]
        if let Ok(gpu) = crate::gpu_contact_pipeline::GpuContactDevice::new() {
            let mut actual = load_mjcf_str(xml, MjcfLoadOptions::default()).unwrap();
            actual.world.positions[0] = 0.5;
            actual.world.positions[1] = 1.28125;
            actual.world.step_gpu(0.001, &[1.0, 0.0], &gpu).unwrap();
            assert!((actual.world.velocities[0] - loaded.world.velocities[0]).abs() < 1e-6);
            assert!((actual.world.velocities[1] - loaded.world.velocities[1]).abs() < 1e-6);
        }
        let inactive = xml.replace(
            "joint1=\"follower\"",
            "active=\"false\" joint1=\"follower\"",
        );
        assert!(
            load_mjcf_str(&inactive, MjcfLoadOptions::default())
                .unwrap()
                .world
                .joint_polynomial_couplings()
                .is_empty()
        );
        let invalid = xml.replace("joint2=\"source\"", "joint2=\"missing\"");
        assert!(load_mjcf_str(&invalid, MjcfLoadOptions::default()).is_err());
        let invalid = xml.replace("0 2 1 0 0.5", "0 2 1");
        assert!(load_mjcf_str(&invalid, MjcfLoadOptions::default()).is_err());
        let unsupported = xml.replace("<joint joint1", "<tendon tendon1");
        assert!(matches!(
            load_mjcf_str(&unsupported, MjcfLoadOptions::default()),
            Err(MjcfLoadError::Unsupported(_))
        ));
        let unsupported = xml.replace(
            "joint1=\"follower\"",
            "solref=\"0.02 1\" joint1=\"follower\"",
        );
        assert!(matches!(
            load_mjcf_str(&unsupported, MjcfLoadOptions::default()),
            Err(MjcfLoadError::Unsupported(_))
        ));
        let locked = xml
            .replace(" joint2=\"source\"", "")
            .replace("0 2 1 0 0.5", "0.2 1 0 0 0");
        let mut locked = load_mjcf_str(&locked, MjcfLoadOptions::default()).unwrap();
        locked.world.step(0.001, &[0.0, 0.0]).unwrap();
        assert!(locked.world.velocities[1] > 0.0);
    }

    #[test]
    fn nonlinear_joint_passive_matches_force_and_implicit_tangent() {
        let xml = r#"<mujoco><option gravity="0 0 0"/><worldbody>
          <body name="root"><inertial mass="1" diaginertia="1 1 1"/>
            <body name="slider"><joint name="slide" type="slide" axis="1 0 0"
              stiffness="2 3 4" damping="5 6 7"/>
              <inertial mass="1" diaginertia="1 1 1"/></body>
          </body></worldbody></mujoco>"#;
        let mut options = MjcfLoadOptions::default();
        options.world.max_substep = 0.01;
        let mut loaded = load_mjcf_str(xml, options.clone()).unwrap();
        let nonlinear = JointNonlinearPassive {
            spring_quadratic: 3.0,
            spring_cubic: 4.0,
            damping_quadratic: 6.0,
            damping_cubic: 7.0,
        };
        assert_eq!(loaded.world.joint_nonlinear_passive(0), Some(nonlinear));
        let snapshot = loaded.world.snapshot();
        loaded
            .world
            .set_joint_nonlinear_passive(0, JointNonlinearPassive::default())
            .unwrap();
        loaded.world.restore_snapshot(&snapshot).unwrap();
        assert_eq!(loaded.world.joint_nonlinear_passive(0), Some(nonlinear));
        loaded.world.positions[0] = 0.5;
        loaded.world.velocities[0] = 0.2;
        let dt = 0.01;
        let force = -(2.0 * 0.5 + 3.0 * 0.5_f64.powi(2) + 4.0 * 0.5_f64.powi(3))
            - (5.0 * 0.2 + 6.0 * 0.2_f64.powi(2) + 7.0 * 0.2_f64.powi(3));
        let spring_tangent = 2.0 + 2.0 * 3.0 * 0.5 + 3.0 * 4.0 * 0.5_f64.powi(2);
        let damping_tangent = 5.0 + 2.0 * 6.0 * 0.2 + 3.0 * 7.0 * 0.2_f64.powi(2);
        let expected = 0.2
            + dt * (force - dt * spring_tangent * 0.2)
                / (1.0 + dt * damping_tangent + dt * dt * spring_tangent);
        loaded.world.step(dt, &[0.0]).unwrap();
        assert!(
            (loaded.world.velocities[0] - expected).abs() < 1e-10,
            "actual={}, expected={expected}",
            loaded.world.velocities[0]
        );
        #[cfg(feature = "gpu-contact")]
        if let Ok(gpu) = crate::gpu_contact_pipeline::GpuContactDevice::new() {
            let mut actual = load_mjcf_str(xml, options).unwrap();
            actual.world.positions[0] = 0.5;
            actual.world.velocities[0] = 0.2;
            actual.world.step_gpu(dt, &[0.0], &gpu).unwrap();
            assert!((actual.world.velocities[0] - expected).abs() < 1e-6);
        }
        let invalid = xml.replace("stiffness=\"2 3 4\"", "stiffness=\"2 3\"");
        assert!(load_mjcf_str(&invalid, MjcfLoadOptions::default()).is_err());
    }

    #[test]
    fn joint_ref_preserves_initial_geometry_and_sets_absolute_coordinate() {
        let slide = r#"<mujoco><option gravity="0 0 0"/><worldbody>
          <body name="root"><inertial mass="1" diaginertia="1 1 1"/>
            <body name="child" pos="0.2 0.3 0.4">
              <joint name="slide" type="slide" axis="1 0 0" ref="0.4"
                range="0.2 0.8" stiffness="2" springref="0.6"/>
              <inertial mass="1" diaginertia="1 1 1"/>
            </body>
          </body></worldbody></mujoco>"#;
        let mut loaded = load_mjcf_str(slide, MjcfLoadOptions::default()).unwrap();
        let baseline = load_mjcf_str(
            &slide.replace(" ref=\"0.4\"", ""),
            MjcfLoadOptions::default(),
        )
        .unwrap();
        assert!((loaded.world.positions[0] - 0.4).abs() < 1e-12);
        let initial = loaded
            .world
            .articulation
            .pose(loaded.world.root_pose, loaded.world.positions.as_slice())
            .unwrap()
            .links[1];
        let baseline_pose = baseline
            .world
            .articulation
            .pose(
                baseline.world.root_pose,
                baseline.world.positions.as_slice(),
            )
            .unwrap()
            .links[1];
        assert!((initial.translation.vector - baseline_pose.translation.vector).norm() < 1e-12);
        assert!((initial.rotation.inverse() * baseline_pose.rotation).angle() < 1e-12);
        let snapshot = loaded.world.snapshot();
        loaded.world.positions[0] = 0.7;
        let moved = loaded
            .world
            .articulation
            .pose(loaded.world.root_pose, loaded.world.positions.as_slice())
            .unwrap()
            .links[1];
        assert!(
            (moved.translation.vector - initial.translation.vector - Vector3::new(0.3, 0.0, 0.0))
                .norm()
                < 1e-12
        );
        loaded.world.restore_snapshot(&snapshot).unwrap();
        assert!((loaded.world.positions[0] - 0.4).abs() < 1e-12);
        loaded.world.step(0.001, &[0.0]).unwrap();
        assert!(loaded.world.velocities[0] > 0.0);
        loaded.world.positions[0] = 0.9;
        loaded.world.velocities[0] = 0.0;
        loaded.world.step(0.001, &[0.0]).unwrap();
        assert!(loaded.world.positions[0] <= 0.8);
        let hinge = slide
            .replace(
                "type=\"slide\" axis=\"1 0 0\" ref=\"0.4\"",
                "type=\"hinge\" axis=\"0 0 1\" ref=\"90\"",
            )
            .replace("range=\"0.2 0.8\"", "range=\"0 180\"");
        let mut loaded = load_mjcf_str(&hinge, MjcfLoadOptions::default()).unwrap();
        let baseline = load_mjcf_str(
            &hinge.replace(" ref=\"90\"", ""),
            MjcfLoadOptions::default(),
        )
        .unwrap();
        assert!((loaded.world.positions[0] - core::f64::consts::FRAC_PI_2).abs() < 1e-12);
        let initial = loaded
            .world
            .articulation
            .pose(loaded.world.root_pose, loaded.world.positions.as_slice())
            .unwrap()
            .links[1];
        let baseline_pose = baseline
            .world
            .articulation
            .pose(
                baseline.world.root_pose,
                baseline.world.positions.as_slice(),
            )
            .unwrap()
            .links[1];
        assert!((initial.rotation.inverse() * baseline_pose.rotation).angle() < 1e-12);
        loaded.world.positions[0] += core::f64::consts::FRAC_PI_2;
        let moved = loaded
            .world
            .articulation
            .pose(loaded.world.root_pose, loaded.world.positions.as_slice())
            .unwrap()
            .links[1];
        let expected = initial.rotation
            * UnitQuaternion::from_axis_angle(&Vector3::z_axis(), core::f64::consts::FRAC_PI_2);
        assert!((moved.rotation.inverse() * expected).angle() < 1e-12);
        let radians = hinge
            .replace("<mujoco>", "<mujoco><compiler angle=\"radian\"/>")
            .replace("ref=\"90\"", "ref=\"1.2\"");
        let radians = load_mjcf_str(&radians, MjcfLoadOptions::default()).unwrap();
        assert!((radians.world.positions[0] - 1.2).abs() < 1e-12);
    }

    #[test]
    fn joint_equality_uses_mjcf_reference_coordinates() {
        let xml = r#"<mujoco><option gravity="0 0 0"/><worldbody>
          <body name="root"><inertial mass="1" diaginertia="1 1 1"/>
            <body name="source"><joint name="source" type="slide" axis="1 0 0" ref="0.3"/>
              <inertial mass="1" diaginertia="1 1 1"/></body>
            <body name="follower"><joint name="follower" type="slide" axis="0 1 0" ref="0.5"/>
              <inertial mass="1" diaginertia="1 1 1"/></body>
          </body></worldbody>
          <equality><joint joint1="follower" joint2="source" polycoef="0 2 0 0 0"/></equality>
        </mujoco>"#;
        let mut loaded = load_mjcf_str(xml, MjcfLoadOptions::default()).unwrap();
        assert_eq!(loaded.world.positions.as_slice(), &[0.3, 0.5]);
        let coupling = loaded.world.joint_polynomial_couplings()[0];
        assert_eq!(coupling.source_reference, 0.3);
        assert_eq!(coupling.follower_reference, 0.5);
        loaded.world.positions[0] = 0.4;
        loaded.world.positions[1] = 0.7;
        loaded.world.step(0.001, &[1.0, 0.0]).unwrap();
        assert!((loaded.world.velocities[1] - 2.0 * loaded.world.velocities[0]).abs() < 1e-8);
        #[cfg(feature = "gpu-contact")]
        if let Ok(gpu) = crate::gpu_contact_pipeline::GpuContactDevice::new() {
            let mut actual = load_mjcf_str(xml, MjcfLoadOptions::default()).unwrap();
            actual.world.positions[0] = 0.4;
            actual.world.positions[1] = 0.7;
            actual.world.step_gpu(0.001, &[1.0, 0.0], &gpu).unwrap();
            assert!((actual.world.velocities[0] - loaded.world.velocities[0]).abs() < 1e-6);
            assert!((actual.world.velocities[1] - loaded.world.velocities[1]).abs() < 1e-6);
        }
        let bad = xml.replace(
            "type=\"slide\" axis=\"1 0 0\" ref=\"0.3\"",
            "type=\"ball\" ref=\"0.3\"",
        );
        assert!(matches!(
            load_mjcf_str(&bad, MjcfLoadOptions::default()),
            Err(MjcfLoadError::Unsupported(_))
        ));
    }

    #[test]
    fn springdamper_uses_reference_inertia_and_overrides_direct_coefficients() {
        let xml = r#"
            <mujoco>
              <default><joint springdamper="0.2 0.5" stiffness="999 3 4" damping="999 6 7"/></default>
              <worldbody><body name="root">
                <inertial mass="1" diaginertia="1 1 1"/>
                <body name="slide"><joint type="slide" axis="1 0 0"
                    armature="1" springref="0.3"/>
                  <inertial mass="2" diaginertia="1 1 1"/>
                </body>
              </body></worldbody>
            </mujoco>
        "#;
        let loaded = load_mjcf_str(xml, MjcfLoadOptions::default()).unwrap();
        let passive = loaded.world.joint_passive(0).unwrap();
        assert!((passive.stiffness - 300.0).abs() < 1e-10);
        assert!((passive.damping - 30.0).abs() < 1e-10);
        assert!((passive.rest_position - 0.3).abs() < 1e-10);
        assert_eq!(
            loaded.world.joint_nonlinear_passive(0),
            Some(JointNonlinearPassive::default())
        );
        let floating = xml.replace(
            "<inertial mass=\"1\" diaginertia=\"1 1 1\"/>",
            "<freejoint/><inertial mass=\"1\" diaginertia=\"1 1 1\"/>",
        );
        let floating = load_mjcf_str(&floating, MjcfLoadOptions::default()).unwrap();
        assert!(floating.world.floating);
        assert!((floating.world.joint_passive(0).unwrap().stiffness - 300.0).abs() < 1e-10);
        for invalid in ["0.2 0", "0 0.5", "-0.2 0.5", "nan 0.5"] {
            assert!(
                load_mjcf_str(
                    &xml.replace(
                        "springdamper=\"0.2 0.5\"",
                        &format!("springdamper=\"{invalid}\"")
                    ),
                    MjcfLoadOptions::default()
                )
                .is_err()
            );
        }
    }

    #[test]
    fn infers_primitive_mass_center_and_inertia() {
        let xml = r#"
            <mujoco><worldbody><body name="sphere">
              <geom type="sphere" size="2" density="3" pos="1 2 3"/>
            </body></worldbody></mujoco>
        "#;
        let loaded = load_mjcf_str(xml, MjcfLoadOptions::default()).unwrap();
        let link = loaded.world.articulation.link(0).unwrap();
        let expected_mass = 32.0 * core::f64::consts::PI;
        assert!((link.mass - expected_mass).abs() < 1e-10);
        assert!((link.center_of_mass - Vector3::new(1.0, 2.0, 3.0)).norm() < 1e-12);
        let expected_inertia = 1.6 * expected_mass;
        assert!((link.inertia - Matrix3::identity() * expected_inertia).norm() < 1e-10);
    }

    #[test]
    fn applies_global_nested_and_child_default_classes() {
        let xml = r#"
            <mujoco>
              <default>
                <geom density="10"/>
                <default class="arm">
                  <geom type="box" size="0.1 0.2 0.3"/>
                  <joint axis="0 1 0" range="-30 60"/>
                </default>
                <default class="light"><geom type="sphere" size="0.05" mass="0.1"/></default>
              </default>
              <worldbody><body name="root" childclass="arm">
                <geom mass="2"/>
                <geom class="light" pos="0 0 1"/>
                <body name="child"><joint name="hinge"/><geom/></body>
              </body></worldbody>
            </mujoco>
        "#;
        let loaded = load_mjcf_str(xml, MjcfLoadOptions::default()).unwrap();
        assert_eq!(loaded.world.boxes.len(), 2);
        assert_eq!(loaded.world.colliders.len(), 1);
        assert!((loaded.world.articulation.link(0).unwrap().mass - 2.1).abs() < 1e-12);
        assert!((loaded.world.articulation.link(1).unwrap().mass - 0.48).abs() < 1e-12);
        let limit = loaded.world.articulation.joint_limit(0).unwrap();
        assert!((limit.0 + core::f64::consts::FRAC_PI_6).abs() < 1e-12);
        assert!((limit.1 - core::f64::consts::FRAC_PI_3).abs() < 1e-12);
    }

    #[test]
    fn loads_inline_convex_mesh_asset() {
        let xml = r#"
            <mujoco>
              <asset><mesh name="tetra" vertex="0 0 0  1 0 0  0 1 0  0 0 1"
                face="0 2 1  0 1 3  1 2 3  2 0 3" scale="2 1 0.5"/></asset>
              <worldbody><body name="root">
                <inertial mass="1" diaginertia="1 1 1"/>
                <geom type="mesh" mesh="tetra"/>
              </body></worldbody>
            </mujoco>
        "#;
        let loaded = load_mjcf_str(xml, MjcfLoadOptions::default()).unwrap();
        assert_eq!(loaded.world.convex_shapes.len(), 1);
        let geometry = &loaded.world.convex_shapes[0].geometry;
        assert_eq!(geometry.vertices.len(), 4);
        assert!(geometry.vertices.contains(&Vector3::new(2.0, 0.0, 0.0)));
        assert!(geometry.vertices.contains(&Vector3::new(0.0, 0.0, 0.5)));
    }

    #[test]
    fn resolves_external_mesh_asset_into_convex_parts() {
        let xml = r#"
            <mujoco>
              <asset><mesh name="hull" file="meshes/hull.obj" scale="2 3 4"/></asset>
              <worldbody><body name="root">
                <inertial mass="1" diaginertia="1 1 1"/>
                <geom type="mesh" mesh="hull" pos="1 2 3"/>
              </body></worldbody>
            </mujoco>
        "#;
        let mut calls = Vec::new();
        let mut resolver = |filename: &str, scale: [f64; 3]| {
            calls.push((filename.to_owned(), scale));
            Ok(vec![test_tetrahedron(), test_tetrahedron()])
        };
        let loaded =
            load_mjcf_str_with_mesh_resolver(xml, MjcfLoadOptions::default(), &mut resolver)
                .unwrap();
        assert_eq!(calls, [("meshes/hull.obj".into(), [2.0, 3.0, 4.0])]);
        assert_eq!(loaded.world.convex_shapes.len(), 2);
        assert!(
            loaded
                .world
                .convex_shapes
                .iter()
                .all(|shape| { shape.origin.translation.vector == Vector3::new(1.0, 2.0, 3.0) })
        );
    }

    #[test]
    fn expands_multiple_body_joints_into_massless_frames() {
        let xml = r#"
            <mujoco><worldbody>
              <body name="root"><inertial mass="1" diaginertia="1 1 1"/>
                <body name="gimbal" pos="0 0 1">
                  <joint name="yaw" type="hinge" axis="0 0 1"/>
                  <joint name="pitch" type="hinge" axis="0 1 0"/>
                  <inertial mass="2" diaginertia="1 1 1"/>
                  <geom type="sphere" size="0.1"/>
                </body>
              </body>
            </worldbody></mujoco>
        "#;
        let mut loaded = load_mjcf_str(xml, MjcfLoadOptions::default()).unwrap();
        assert_eq!(loaded.link_names, ["root", "gimbal@yaw", "gimbal"]);
        assert_eq!(loaded.world.articulation.dof(), 2);
        assert_eq!(loaded.joints[0].dofs, 0..1);
        assert_eq!(loaded.joints[1].dofs, 1..2);
        assert_eq!(loaded.world.articulation.link(1).unwrap().mass, 0.0);
        assert_eq!(loaded.world.articulation.link(2).unwrap().mass, 2.0);
        assert_eq!(loaded.world.colliders[0].link, 2);
        assert!(loaded.world.articulation.adjacent(0, 2));

        loaded.world.positions[0] = core::f64::consts::FRAC_PI_2;
        loaded.world.positions[1] = core::f64::consts::FRAC_PI_2;
        let pose = loaded
            .world
            .articulation
            .pose(loaded.world.root_pose, loaded.world.positions.as_slice())
            .unwrap();
        let expected =
            UnitQuaternion::from_axis_angle(&Vector3::z_axis(), core::f64::consts::FRAC_PI_2)
                * UnitQuaternion::from_axis_angle(&Vector3::y_axis(), core::f64::consts::FRAC_PI_2);
        assert!((pose.links[2].rotation.angle_to(&expected)).abs() < 1e-12);
        assert!((pose.links[2].translation.vector - Vector3::new(0.0, 0.0, 1.0)).norm() < 1e-12);

        loaded.world.step(0.002, &[0.0; 2]).unwrap();
        assert!(loaded.world.positions.iter().all(|value| value.is_finite()));
    }

    fn test_tetrahedron() -> ConvexGeometry {
        ConvexGeometry::new(
            vec![
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(0.0, 1.0, 0.0),
                Vector3::new(0.0, 0.0, 1.0),
            ],
            vec![
                -Vector3::x(),
                -Vector3::y(),
                -Vector3::z(),
                Vector3::new(1.0, 1.0, 1.0).normalize(),
            ],
            vec![Vector3::x(), Vector3::y(), Vector3::z()],
        )
        .unwrap()
    }
}
