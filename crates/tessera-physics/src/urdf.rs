//! URDF loader for Tessera articulated worlds.
//!
//! XML is supplied by the caller. External collision meshes are resolved through
//! an injected resolver so this crate does not depend on any repository layout.

use core::ops::Range;
use std::collections::{BTreeMap, BTreeSet};

use nalgebra::{Isometry3, Matrix3, Translation3, UnitQuaternion, Vector3};

use crate::articulated_world::{
    ArticulatedWorld, ArticulatedWorldError, ArticulatedWorldParams, JointPassive, LinkBox,
    LinkConvex, LinkCylinder, LinkSphere,
};
use crate::articulation::{Articulation, ArticulationError, JointKind, JointSpec, LinkSpec};
#[cfg(test)]
use crate::convex::ConvexGeometry;
use crate::mesh::ConvexMeshResolver;

/// Mesh resolver accepted by [`load_urdf_str_with_mesh_resolver`].
pub use crate::mesh::ConvexMeshResolver as UrdfMeshResolver;

/// Options applied while constructing a Tessera world from URDF.
#[derive(Debug, Clone)]
pub struct UrdfLoadOptions {
    /// World transform of the URDF root link.
    pub root_pose: Isometry3<f64>,
    /// Whether the root link contributes six floating-base coordinates.
    pub floating_base: bool,
    /// Tessera integration and contact parameters.
    pub world: ArticulatedWorldParams,
}

impl Default for UrdfLoadOptions {
    fn default() -> Self {
        Self {
            root_pose: Isometry3::identity(),
            floating_base: false,
            world: ArticulatedWorldParams::default(),
        }
    }
}

/// Stable metadata for one loaded URDF joint.
#[derive(Debug, Clone, PartialEq)]
pub struct UrdfJointInfo {
    /// Joint name from the URDF document.
    pub name: String,
    /// Parent link index in [`LoadedUrdf::link_names`].
    pub parent_link: usize,
    /// Child link index in [`LoadedUrdf::link_names`].
    pub child_link: usize,
    /// Tessera joint kind.
    pub kind: JointKind,
    /// Generalized-coordinate slots used by this joint. Mimics share source slots.
    pub dofs: Range<usize>,
    /// Affine source relation when this joint mimics another joint.
    pub mimic: Option<UrdfMimicInfo>,
    /// Positive URDF effort limit, when specified.
    pub effort_limit: Option<f64>,
    /// Positive URDF velocity limit, when specified.
    pub velocity_limit: Option<f64>,
}

/// Affine URDF mimic relation, `q = multiplier * q_source + offset`.
#[derive(Debug, Clone, PartialEq)]
pub struct UrdfMimicInfo {
    /// Name of the source joint in the URDF document.
    pub source_joint: String,
    /// Source-coordinate multiplier.
    pub multiplier: f64,
    /// Dependent-coordinate offset.
    pub offset: f64,
}

/// A loaded URDF model and its ready-to-step Tessera world.
#[derive(Debug)]
pub struct LoadedUrdf {
    /// Robot name from the root `robot` element.
    pub robot_name: String,
    /// Link names in articulation index order. Compound joints and multiple
    /// roots append virtual frames.
    pub link_names: Vec<String>,
    /// Joint metadata in URDF joint order. Planar and floating joints expand
    /// into individually named translation and rotation components.
    pub joints: Vec<UrdfJointInfo>,
    /// Constructed fixed-root or floating-root physics world.
    pub world: ArticulatedWorld,
}

/// URDF parsing or model-conversion failure.
#[derive(Debug, thiserror::Error)]
pub enum UrdfLoadError {
    /// Malformed XML or invalid URDF structure.
    #[error("invalid URDF XML: {0}")]
    Xml(String),
    /// A recognized URDF construct cannot be represented by this loader yet.
    #[error("unsupported URDF construct: {0}")]
    Unsupported(String),
    /// An attribute or model relationship is invalid.
    #[error("invalid URDF model: {0}")]
    Invalid(String),
    /// An injected mesh resolver failed.
    #[error("failed to resolve URDF mesh `{filename}`: {message}")]
    Mesh {
        /// URI exactly as written in the URDF document.
        filename: String,
        /// Resolver-provided diagnostic.
        message: String,
    },
    /// The converted articulation is invalid.
    #[error("invalid URDF articulation: {0}")]
    Articulation(#[from] ArticulationError),
    /// The converted Tessera world is invalid.
    #[error("invalid URDF world: {0}")]
    World(#[from] ArticulatedWorldError),
}

/// Parse a self-contained URDF document without external collision meshes.
pub fn load_urdf_str(xml: &str, options: UrdfLoadOptions) -> Result<LoadedUrdf, UrdfLoadError> {
    load_urdf_inner(xml, options, None)
}

/// Parse URDF and resolve every external collision mesh through `resolver`.
pub fn load_urdf_str_with_mesh_resolver(
    xml: &str,
    options: UrdfLoadOptions,
    resolver: &mut dyn ConvexMeshResolver,
) -> Result<LoadedUrdf, UrdfLoadError> {
    load_urdf_inner(xml, options, Some(resolver))
}

fn load_urdf_inner(
    xml: &str,
    options: UrdfLoadOptions,
    mut resolver: Option<&mut dyn ConvexMeshResolver>,
) -> Result<LoadedUrdf, UrdfLoadError> {
    let robot =
        urdf_rs::read_from_string(xml).map_err(|error| UrdfLoadError::Xml(error.to_string()))?;
    if robot.links.is_empty() {
        return Err(UrdfLoadError::Invalid("document has no links".into()));
    }

    let mut link_indices = BTreeMap::new();
    for (index, link) in robot.links.iter().enumerate() {
        if link.name.is_empty() {
            return Err(UrdfLoadError::Invalid("link name is empty".into()));
        }
        if link_indices.insert(link.name.clone(), index).is_some() {
            return Err(UrdfLoadError::Invalid(format!(
                "duplicate link name `{}`",
                link.name
            )));
        }
    }

    let mut links = robot
        .links
        .iter()
        .map(link_spec)
        .collect::<Result<Vec<_>, _>>()?;
    let mut link_names = robot
        .links
        .iter()
        .map(|link| link.name.clone())
        .collect::<Vec<_>>();
    let mut child_links = BTreeSet::new();
    let mut used_link_names = link_indices.keys().cloned().collect::<BTreeSet<_>>();
    let mut joint_names = BTreeSet::new();
    let mut scalar_joint_edges = BTreeMap::new();
    let mut mimic_requests = Vec::new();
    let mut joints = Vec::with_capacity(robot.joints.len());
    let mut joint_info = Vec::with_capacity(robot.joints.len());
    let mut joint_dampings = Vec::with_capacity(robot.joints.len());
    let mut joint_frictions = Vec::with_capacity(robot.joints.len());
    let mut dof = 0;
    for joint in &robot.joints {
        if !joint_names.insert(&joint.name) {
            return Err(UrdfLoadError::Invalid(format!(
                "duplicate joint name `{}`",
                joint.name
            )));
        }
        let parent = link_index(&link_indices, &joint.parent.link, "parent", &joint.name)?;
        let child = link_index(&link_indices, &joint.child.link, "child", &joint.name)?;
        if !child_links.insert(child) {
            return Err(UrdfLoadError::Invalid(format!(
                "link `{}` has multiple parent joints",
                joint.child.link
            )));
        }
        let components = joint_components(joint)?;
        let damping = joint.dynamics.as_ref().map_or(0.0, |value| value.damping);
        let friction = joint.dynamics.as_ref().map_or(0.0, |value| value.friction);
        if !damping.is_finite() || damping < 0.0 {
            return Err(UrdfLoadError::Invalid(format!(
                "joint `{}` has invalid damping",
                joint.name
            )));
        }
        if !friction.is_finite() || friction < 0.0 {
            return Err(UrdfLoadError::Invalid(format!(
                "joint `{}` has invalid friction",
                joint.name
            )));
        }
        let component_count = components.len();
        let first_edge = joints.len();
        let scalar = component_count == 1
            && matches!(
                components[0].kind,
                JointKind::Revolute | JointKind::Prismatic
            );
        let _previous = scalar_joint_edges.insert(joint.name.clone(), scalar.then_some(first_edge));
        let mimic = joint.mimic.as_ref().map(|mimic| UrdfMimicInfo {
            source_joint: mimic.joint.clone(),
            multiplier: mimic.multiplier.unwrap_or(1.0),
            offset: mimic.offset.unwrap_or(0.0),
        });
        if let Some(relation) = &mimic {
            if !scalar || !relation.multiplier.is_finite() || !relation.offset.is_finite() {
                return Err(UrdfLoadError::Invalid(format!(
                    "joint `{}` has an invalid mimic relation",
                    joint.name
                )));
            }
            mimic_requests.push((first_edge, relation.clone()));
        }
        let mut chain_parent = parent;
        for (index, component) in components.into_iter().enumerate() {
            let chain_child = if index + 1 == component_count {
                child
            } else {
                let virtual_link = links.len();
                links.push(LinkSpec {
                    mass: 0.0,
                    center_of_mass: Vector3::zeros(),
                    inertia: Matrix3::zeros(),
                });
                let base_name = format!("__urdf_{}_{index}", joint.name);
                let mut name = base_name.clone();
                let mut suffix = 0usize;
                while !used_link_names.insert(name.clone()) {
                    suffix += 1;
                    name = format!("{base_name}_{suffix}");
                }
                link_names.push(name);
                virtual_link
            };
            let start = dof;
            dof += component.width;
            joints.push(JointSpec {
                parent: chain_parent,
                child: chain_child,
                origin: if index == 0 {
                    pose(&joint.origin)
                } else {
                    Isometry3::identity()
                },
                kind: component.kind,
                axis: component.axis,
                limits: component.limits,
            });
            joint_info.push(UrdfJointInfo {
                name: component.suffix.map_or_else(
                    || joint.name.clone(),
                    |suffix| format!("{}/{suffix}", joint.name),
                ),
                parent_link: chain_parent,
                child_link: chain_child,
                kind: component.kind,
                dofs: start..dof,
                mimic: mimic.clone(),
                effort_limit: positive_limit(joint.limit.effort),
                velocity_limit: positive_limit(joint.limit.velocity),
            });
            joint_dampings.push(damping);
            joint_frictions.push(friction);
            chain_parent = chain_child;
        }
    }

    let roots = (0..robot.links.len())
        .filter(|index| !child_links.contains(index))
        .collect::<Vec<_>>();
    if roots.is_empty() {
        return Err(UrdfLoadError::Invalid(format!(
            "expected at least one root link, found {}",
            roots.len()
        )));
    }
    let root = if roots.len() == 1 {
        roots[0]
    } else {
        let root = links.len();
        links.push(LinkSpec {
            mass: 0.0,
            center_of_mass: Vector3::zeros(),
            inertia: Matrix3::zeros(),
        });
        let mut name = "__urdf_world".to_owned();
        let mut suffix = 0usize;
        while !used_link_names.insert(name.clone()) {
            suffix += 1;
            name = format!("__urdf_world_{suffix}");
        }
        link_names.push(name);
        for child in roots {
            joints.push(JointSpec {
                parent: root,
                child,
                origin: Isometry3::identity(),
                kind: JointKind::Fixed,
                axis: Vector3::z(),
                limits: None,
            });
        }
        root
    };
    let mut articulation = Articulation::new(links, joints, root)?;
    let mimic_edges = mimic_requests
        .iter()
        .map(|(dependent, relation)| {
            let source = scalar_joint_edges
                .get(&relation.source_joint)
                .copied()
                .flatten()
                .ok_or_else(|| {
                    UrdfLoadError::Invalid(format!(
                        "mimic source `{}` is missing or not scalar",
                        relation.source_joint
                    ))
                })?;
            Ok((*dependent, source, relation.multiplier, relation.offset))
        })
        .collect::<Result<Vec<_>, UrdfLoadError>>()?;
    articulation.set_mimics(&mimic_edges).map_err(|_| {
        UrdfLoadError::Invalid("mimic chain is cyclic or has conflicting limits".into())
    })?;
    for (edge, info) in joint_info.iter_mut().enumerate() {
        info.dofs = articulation
            .joint_coordinate_range(edge)
            .ok_or_else(|| UrdfLoadError::Invalid("invalid joint metadata index".into()))?;
    }

    let mut spheres = Vec::new();
    let mut boxes = Vec::new();
    let mut cylinders = Vec::new();
    let mut convex_shapes = Vec::new();
    for (link_index, link) in robot.links.iter().enumerate() {
        for collision in &link.collision {
            let origin = pose(&collision.origin);
            match &collision.geometry {
                urdf_rs::Geometry::Sphere { radius } => {
                    validate_positive(*radius, "sphere radius")?;
                    spheres.push(LinkSphere {
                        link: link_index,
                        center: origin.translation.vector,
                        radius: *radius,
                    });
                }
                urdf_rs::Geometry::Box { size } => {
                    let half_extents = Vector3::new(size[0], size[1], size[2]) * 0.5;
                    if half_extents
                        .iter()
                        .any(|value| !value.is_finite() || *value <= 0.0)
                    {
                        return Err(UrdfLoadError::Invalid(
                            "box dimensions must be finite and positive".into(),
                        ));
                    }
                    boxes.push(LinkBox {
                        link: link_index,
                        origin,
                        half_extents,
                    });
                }
                urdf_rs::Geometry::Cylinder { radius, length } => {
                    validate_positive(*radius, "cylinder radius")?;
                    validate_positive(*length, "cylinder length")?;
                    cylinders.push(LinkCylinder {
                        link: link_index,
                        origin,
                        half_height: length * 0.5,
                        radius: *radius,
                    });
                }
                urdf_rs::Geometry::Capsule { radius, length } => {
                    validate_positive(*radius, "capsule radius")?;
                    validate_nonnegative(*length, "capsule length")?;
                    let half_height = length * 0.5;
                    if half_height > 0.0 {
                        cylinders.push(LinkCylinder {
                            link: link_index,
                            origin,
                            half_height,
                            radius: *radius,
                        });
                    }
                    let endpoints: &[f64] = if half_height > 0.0 {
                        &[-half_height, half_height]
                    } else {
                        &[0.0]
                    };
                    for &z in endpoints {
                        spheres.push(LinkSphere {
                            link: link_index,
                            center: origin
                                .transform_point(&nalgebra::Point3::new(0.0, 0.0, z))
                                .coords,
                            radius: *radius,
                        });
                    }
                }
                urdf_rs::Geometry::Mesh { filename, scale } => {
                    let scale = scale.as_ref().map_or([1.0; 3], |value| value.0);
                    if scale
                        .iter()
                        .any(|value| !value.is_finite() || *value == 0.0)
                    {
                        return Err(UrdfLoadError::Invalid(format!(
                            "mesh `{filename}` has invalid scale"
                        )));
                    }
                    let resolver = resolver.as_deref_mut().ok_or_else(|| {
                        UrdfLoadError::Unsupported(format!(
                            "mesh `{filename}` requires a mesh resolver"
                        ))
                    })?;
                    let parts = resolver.resolve(filename, scale).map_err(|message| {
                        UrdfLoadError::Mesh {
                            filename: filename.clone(),
                            message,
                        }
                    })?;
                    if parts.is_empty() {
                        return Err(UrdfLoadError::Invalid(format!(
                            "mesh `{filename}` resolved to no convex parts"
                        )));
                    }
                    convex_shapes.extend(parts.into_iter().map(|geometry| LinkConvex {
                        link: link_index,
                        origin,
                        geometry,
                    }));
                }
            }
        }
    }

    let mut world = if options.floating_base {
        ArticulatedWorld::new_floating(articulation, options.root_pose, spheres, options.world)?
    } else {
        ArticulatedWorld::new(articulation, options.root_pose, spheres, options.world)?
    };
    let mut generalized_damping = vec![0.0; world.articulation.dof()];
    let mut generalized_friction = vec![0.0; world.articulation.dof()];
    for (edge, damping) in joint_dampings.into_iter().enumerate() {
        let scale = world
            .articulation
            .joint_coordinate_scale(edge)
            .ok_or_else(|| UrdfLoadError::Invalid("invalid joint damping index".into()))?;
        for slot in joint_info[edge].dofs.clone() {
            generalized_damping[slot] += damping * scale * scale;
            generalized_friction[slot] += joint_frictions[edge] * scale.abs();
        }
    }
    for (slot, damping) in generalized_damping.into_iter().enumerate() {
        world.set_joint_passive(
            slot,
            JointPassive {
                damping,
                ..JointPassive::default()
            },
        )?;
        world.set_joint_friction(slot, generalized_friction[slot])?;
    }
    world.set_boxes(boxes)?;
    world.set_cylinders(cylinders)?;
    world.set_convex_shapes(convex_shapes)?;

    Ok(LoadedUrdf {
        robot_name: robot.name,
        link_names,
        joints: joint_info,
        world,
    })
}

fn link_index(
    indices: &BTreeMap<String, usize>,
    name: &str,
    role: &str,
    joint: &str,
) -> Result<usize, UrdfLoadError> {
    indices.get(name).copied().ok_or_else(|| {
        UrdfLoadError::Invalid(format!(
            "joint `{joint}` references missing {role} link `{name}`"
        ))
    })
}

fn link_spec(link: &urdf_rs::Link) -> Result<LinkSpec, UrdfLoadError> {
    let inertial = &link.inertial;
    let local = Matrix3::new(
        inertial.inertia.ixx,
        inertial.inertia.ixy,
        inertial.inertia.ixz,
        inertial.inertia.ixy,
        inertial.inertia.iyy,
        inertial.inertia.iyz,
        inertial.inertia.ixz,
        inertial.inertia.iyz,
        inertial.inertia.izz,
    );
    let rotation = pose(&inertial.origin).rotation.to_rotation_matrix();
    let inertia = rotation.matrix() * local * rotation.matrix().transpose();
    let spec = LinkSpec {
        mass: inertial.mass.value,
        center_of_mass: Vector3::new(
            inertial.origin.xyz[0],
            inertial.origin.xyz[1],
            inertial.origin.xyz[2],
        ),
        inertia,
    };
    if spec.mass.is_finite()
        && spec.mass >= 0.0
        && spec.center_of_mass.iter().all(|value| value.is_finite())
        && spec.inertia.iter().all(|value| value.is_finite())
    {
        Ok(spec)
    } else {
        Err(UrdfLoadError::Invalid(format!(
            "link `{}` has invalid inertial properties",
            link.name
        )))
    }
}

struct JointMapping {
    kind: JointKind,
    limits: Option<(f64, f64)>,
    width: usize,
    axis: Vector3<f64>,
    suffix: Option<&'static str>,
}

fn joint_components(joint: &urdf_rs::Joint) -> Result<Vec<JointMapping>, UrdfLoadError> {
    let axis = Vector3::new(joint.axis.xyz[0], joint.axis.xyz[1], joint.axis.xyz[2]);
    let mapping = |kind, limits, width, axis, suffix| JointMapping {
        kind,
        limits,
        width,
        axis,
        suffix,
    };
    let components = match joint.joint_type {
        urdf_rs::JointType::Fixed => vec![mapping(JointKind::Fixed, None, 0, axis, None)],
        urdf_rs::JointType::Continuous => {
            vec![mapping(JointKind::Revolute, None, 1, axis, None)]
        }
        urdf_rs::JointType::Revolute => vec![mapping(
            JointKind::Revolute,
            Some(valid_limits(joint, "revolute")?),
            1,
            axis,
            None,
        )],
        urdf_rs::JointType::Prismatic => vec![mapping(
            JointKind::Prismatic,
            Some(valid_limits(joint, "prismatic")?),
            1,
            axis,
            None,
        )],
        urdf_rs::JointType::Spherical => {
            vec![mapping(JointKind::Spherical, None, 3, axis, None)]
        }
        urdf_rs::JointType::Floating => vec![
            mapping(JointKind::Prismatic, None, 1, Vector3::x(), Some("tx")),
            mapping(JointKind::Prismatic, None, 1, Vector3::y(), Some("ty")),
            mapping(JointKind::Prismatic, None, 1, Vector3::z(), Some("tz")),
            mapping(
                JointKind::Spherical,
                None,
                3,
                Vector3::z(),
                Some("rotation"),
            ),
        ],
        urdf_rs::JointType::Planar => {
            let length = axis.norm();
            if !length.is_finite() || length <= 1e-12 {
                return Err(UrdfLoadError::Invalid(format!(
                    "planar joint `{}` has a zero normal",
                    joint.name
                )));
            }
            let normal = axis / length;
            let reference = if normal.x.abs() < 0.9 {
                Vector3::x()
            } else {
                Vector3::y()
            };
            let tangent = (reference - normal * reference.dot(&normal)).normalize();
            let bitangent = normal.cross(&tangent);
            vec![
                mapping(JointKind::Prismatic, None, 1, tangent, Some("u")),
                mapping(JointKind::Prismatic, None, 1, bitangent, Some("v")),
                mapping(JointKind::Revolute, None, 1, normal, Some("rotation")),
            ]
        }
    };
    Ok(components)
}

fn valid_limits(joint: &urdf_rs::Joint, kind: &str) -> Result<(f64, f64), UrdfLoadError> {
    let limits = (joint.limit.lower, joint.limit.upper);
    if limits.0.is_finite() && limits.1.is_finite() && limits.0 <= limits.1 {
        Ok(limits)
    } else {
        Err(UrdfLoadError::Invalid(format!(
            "{kind} joint `{}` has invalid limits",
            joint.name
        )))
    }
}

fn positive_limit(value: f64) -> Option<f64> {
    (value.is_finite() && value > 0.0).then_some(value)
}

fn validate_positive(value: f64, label: &str) -> Result<(), UrdfLoadError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(UrdfLoadError::Invalid(format!(
            "{label} must be finite and positive"
        )))
    }
}

fn validate_nonnegative(value: f64, label: &str) -> Result<(), UrdfLoadError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(UrdfLoadError::Invalid(format!(
            "{label} must be finite and nonnegative"
        )))
    }
}

fn pose(pose: &urdf_rs::Pose) -> Isometry3<f64> {
    Isometry3::from_parts(
        Translation3::new(pose.xyz[0], pose.xyz[1], pose.xyz[2]),
        UnitQuaternion::from_euler_angles(pose.rpy[0], pose.rpy[1], pose.rpy[2]),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROBOT: &str = r#"
        <robot name="all-joints">
          <link name="base">
            <inertial><mass value="2"/><inertia ixx="2" ixy="0" ixz="0" iyy="1" iyz="0" izz="3"/>
              <origin xyz="0 0 0.1" rpy="0 0 1.5707963267948966"/>
            </inertial>
            <collision><geometry><box size="1 2 3"/></geometry></collision>
          </link>
          <link name="arm">
            <inertial><mass value="1"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
            <collision><origin xyz="0 0 0.5"/><geometry><capsule radius="0.1" length="1"/></geometry></collision>
          </link>
          <link name="slider"><inertial><mass value="1"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
            <collision><geometry><sphere radius="0.2"/></geometry></collision>
          </link>
          <link name="wrist"><inertial><mass value="1"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
            <collision><geometry><cylinder radius="0.1" length="0.4"/></geometry></collision>
          </link>
          <joint name="shoulder" type="revolute"><parent link="base"/><child link="arm"/>
            <origin xyz="0 0 1"/><axis xyz="0 1 0"/><limit lower="-1" upper="2" effort="3" velocity="4"/>
          </joint>
          <joint name="extension" type="prismatic"><parent link="arm"/><child link="slider"/>
            <axis xyz="0 0 1"/><limit lower="0" upper="0.5" effort="5" velocity="6"/>
          </joint>
          <joint name="wrist-ball" type="spherical"><parent link="slider"/><child link="wrist"/></joint>
        </robot>
    "#;

    #[test]
    fn loads_tree_inertials_primitives_and_joint_metadata() {
        let options = UrdfLoadOptions {
            floating_base: true,
            ..UrdfLoadOptions::default()
        };
        let loaded = load_urdf_str(ROBOT, options).unwrap();
        assert_eq!(loaded.robot_name, "all-joints");
        assert_eq!(loaded.link_names, ["base", "arm", "slider", "wrist"]);
        assert!(loaded.world.floating);
        assert_eq!(loaded.world.articulation.dof(), 5);
        assert_eq!(loaded.joints[0].dofs, 0..1);
        assert_eq!(loaded.joints[1].dofs, 1..2);
        assert_eq!(loaded.joints[2].dofs, 2..5);
        assert_eq!(loaded.joints[0].effort_limit, Some(3.0));
        assert_eq!(loaded.joints[1].velocity_limit, Some(6.0));
        assert_eq!(loaded.world.boxes.len(), 1);
        assert_eq!(loaded.world.cylinders.len(), 2);
        assert_eq!(loaded.world.colliders.len(), 3);

        let root = loaded.world.articulation.link(0).unwrap();
        assert!((root.center_of_mass - Vector3::new(0.0, 0.0, 0.1)).norm() < 1e-12);
        assert!((root.inertia[(0, 0)] - 1.0).abs() < 1e-12);
        assert!((root.inertia[(1, 1)] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn resolves_scaled_meshes_with_caller_policy() {
        let xml = r#"
            <robot name="mesh"><link name="base">
              <inertial><mass value="1"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
              <collision><origin xyz="1 2 3"/><geometry><mesh filename="package://robot/hull.obj" scale="2 3 4"/></geometry></collision>
            </link></robot>
        "#;
        let mut calls = Vec::new();
        let mut resolver = |filename: &str, scale: [f64; 3]| {
            calls.push((filename.to_owned(), scale));
            Ok(vec![tetrahedron()])
        };
        let loaded =
            load_urdf_str_with_mesh_resolver(xml, UrdfLoadOptions::default(), &mut resolver)
                .unwrap();
        assert_eq!(
            calls,
            [("package://robot/hull.obj".into(), [2.0, 3.0, 4.0])]
        );
        assert_eq!(loaded.world.convex_shapes.len(), 1);
        assert_eq!(
            loaded.world.convex_shapes[0].origin.translation.vector,
            Vector3::new(1.0, 2.0, 3.0)
        );
    }

    #[test]
    fn rejects_mesh_without_resolver_and_invalid_planar_normal() {
        let mesh = r#"<robot name="mesh"><link name="base"><collision><geometry><mesh filename="a.stl"/></geometry></collision></link></robot>"#;
        assert!(matches!(
            load_urdf_str(mesh, UrdfLoadOptions::default()),
            Err(UrdfLoadError::Unsupported(_))
        ));
        let planar = r#"
            <robot name="planar"><link name="a"/><link name="b"/>
              <joint name="j" type="planar"><parent link="a"/><child link="b"/>
                <axis xyz="0 0 0"/></joint>
            </robot>
        "#;
        assert!(matches!(
            load_urdf_str(planar, UrdfLoadOptions::default()),
            Err(UrdfLoadError::Invalid(_))
        ));
    }

    #[test]
    fn planar_joint_moves_in_its_plane_and_rotates_about_normal() {
        let xml = r#"
            <robot name="planar"><link name="base"/><link name="platform">
              <inertial><mass value="1"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
              <collision><geometry><box size="0.2 0.2 0.2"/></geometry></collision>
            </link>
              <joint name="plane" type="planar"><parent link="base"/><child link="platform"/>
                <origin xyz="0 0 2"/><axis xyz="0 0 1"/></joint>
            </robot>
        "#;
        let mut loaded = load_urdf_str(xml, UrdfLoadOptions::default()).unwrap();
        assert_eq!(loaded.world.articulation.dof(), 3);
        assert_eq!(loaded.link_names.len(), 4);
        assert_eq!(
            loaded
                .joints
                .iter()
                .map(|joint| joint.name.as_str())
                .collect::<Vec<_>>(),
            ["plane/u", "plane/v", "plane/rotation"]
        );
        assert_eq!(loaded.joints[2].child_link, 1);
        loaded.world.positions[0] = 1.0;
        loaded.world.positions[1] = 2.0;
        loaded.world.positions[2] = core::f64::consts::FRAC_PI_2;
        let pose = loaded.world.link_poses().unwrap()[1];
        assert!((pose.translation.vector - Vector3::new(1.0, 2.0, 2.0)).norm() < 1e-12);
        assert!((pose.rotation * Vector3::x() - Vector3::y()).norm() < 1e-12);
        loaded.world.step(0.01, &[1.0, 0.0, 0.0]).unwrap();
        assert!(loaded.world.velocities[0] > 0.0);
    }

    #[test]
    fn floating_joint_has_six_generalized_coordinates() {
        let xml = r#"
            <robot name="floating"><link name="__urdf_free_0"/><link name="body">
              <inertial><mass value="1"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
              <collision><geometry><sphere radius="0.2"/></geometry></collision>
            </link>
              <joint name="free" type="floating"><parent link="__urdf_free_0"/><child link="body"/>
                <origin xyz="0 0 2"/></joint>
            </robot>
        "#;
        let mut loaded = load_urdf_str(xml, UrdfLoadOptions::default()).unwrap();
        assert_eq!(loaded.world.articulation.dof(), 6);
        assert_eq!(loaded.link_names.len(), 5);
        assert_eq!(loaded.link_names[2], "__urdf_free_0_1");
        assert_eq!(loaded.joints[0].name, "free/tx");
        assert_eq!(loaded.joints[3].name, "free/rotation");
        assert_eq!(loaded.joints[3].dofs, 3..6);
        assert_eq!(loaded.joints[3].child_link, 1);
        loaded.world.positions[0] = 1.0;
        loaded.world.positions[1] = 2.0;
        loaded.world.positions[2] = 3.0;
        loaded.world.positions[5] = core::f64::consts::FRAC_PI_2;
        let pose = loaded.world.link_poses().unwrap()[1];
        assert!((pose.translation.vector - Vector3::new(1.0, 2.0, 5.0)).norm() < 1e-12);
        assert!((pose.rotation * Vector3::x() - Vector3::y()).norm() < 1e-12);
        loaded.world.step(0.01, &[0.0; 6]).unwrap();
        assert!(
            loaded
                .world
                .velocities
                .iter()
                .all(|value| value.is_finite())
        );
    }

    #[test]
    fn floating_children_exchange_contact_impulses() {
        let xml = r#"
            <robot name="floating-pair"><link name="base"/>
              <link name="left"><inertial><mass value="1"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
                <collision><geometry><sphere radius="0.2"/></geometry></collision>
              </link>
              <link name="right"><inertial><mass value="1"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
                <collision><geometry><sphere radius="0.2"/></geometry></collision>
              </link>
              <joint name="left_free" type="floating"><parent link="base"/><child link="left"/>
                <origin xyz="-0.15 0 1"/></joint>
              <joint name="right_free" type="floating"><parent link="base"/><child link="right"/>
                <origin xyz="0.15 0 1"/></joint>
            </robot>
        "#;
        let options = UrdfLoadOptions {
            world: ArticulatedWorldParams {
                gravity: [0.0; 3],
                ..Default::default()
            },
            ..Default::default()
        };
        let mut cpu = load_urdf_str(xml, options.clone()).unwrap();
        assert_eq!(cpu.world.articulation.dof(), 12);
        cpu.world.step(0.01, &[0.0; 12]).unwrap();
        assert!(cpu.world.velocities[0] < 0.0);
        assert!(cpu.world.velocities[6] > 0.0);

        #[cfg(feature = "gpu-contact")]
        if let Ok(context) = crate::gpu_contact_pipeline::GpuContactDevice::new() {
            let mut gpu = load_urdf_str(xml, options).unwrap();
            gpu.world.step_gpu(0.01, &[0.0; 12], &context).unwrap();
            assert!((gpu.world.velocities[0] - cpu.world.velocities[0]).abs() < 1e-4);
            assert!((gpu.world.velocities[6] - cpu.world.velocities[6]).abs() < 1e-4);
        }
    }

    #[test]
    fn multiple_roots_share_world_frame_and_collide() {
        let xml = r#"
            <robot name="forest"><link name="__urdf_world"/><link name="root_b"/>
              <link name="left"><inertial><mass value="1"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
                <collision><geometry><sphere radius="0.2"/></geometry></collision>
              </link>
              <link name="right"><inertial><mass value="1"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
                <collision><geometry><sphere radius="0.2"/></geometry></collision>
              </link>
              <joint name="a_free" type="floating"><parent link="__urdf_world"/><child link="left"/>
                <origin xyz="-0.15 0 1"/></joint>
              <joint name="b_free" type="floating"><parent link="root_b"/><child link="right"/>
                <origin xyz="0.15 0 1"/></joint>
            </robot>
        "#;
        let options = UrdfLoadOptions {
            root_pose: Isometry3::translation(1.0, 0.0, 0.0),
            world: ArticulatedWorldParams {
                gravity: [0.0; 3],
                ..Default::default()
            },
            ..Default::default()
        };
        let mut cpu = load_urdf_str(xml, options.clone()).unwrap();
        assert_eq!(cpu.world.articulation.dof(), 12);
        assert_eq!(cpu.link_names.last().unwrap(), "__urdf_world_1");
        let poses = cpu.world.link_poses().unwrap();
        assert!((poses[2].translation.vector.x - 0.85).abs() < 1e-12);
        assert!((poses[3].translation.vector.x - 1.15).abs() < 1e-12);
        cpu.world.step(0.01, &[0.0; 12]).unwrap();
        assert!(cpu.world.velocities[0] < 0.0);
        assert!(cpu.world.velocities[6] > 0.0);

        #[cfg(feature = "gpu-contact")]
        if let Ok(context) = crate::gpu_contact_pipeline::GpuContactDevice::new() {
            let mut gpu = load_urdf_str(xml, options).unwrap();
            gpu.world.step_gpu(0.01, &[0.0; 12], &context).unwrap();
            assert!((gpu.world.velocities[0] - cpu.world.velocities[0]).abs() < 1e-4);
            assert!((gpu.world.velocities[6] - cpu.world.velocities[6]).abs() < 1e-4);
        }
    }

    #[test]
    fn mimic_joint_couples_contact_dynamics_on_cpu_and_gpu() {
        let xml = r#"
            <robot name="gripper"><link name="base"/>
              <link name="left"><inertial><origin xyz="0 0.2 0"/><mass value="1"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial>
                <collision><origin xyz="0 0.2 0"/><geometry><sphere radius="0.2"/></geometry></collision>
              </link>
              <link name="right"><inertial><origin xyz="0 0.2 0"/><mass value="1"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial>
                <collision><origin xyz="0 0.2 0"/><geometry><sphere radius="0.2"/></geometry></collision>
              </link>
              <joint name="left_hinge" type="revolute"><parent link="base"/><child link="left"/>
                <origin xyz="-0.15 0 1"/><axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="10" velocity="10"/>
              </joint>
              <joint name="right_hinge" type="revolute"><parent link="base"/><child link="right"/>
                <origin xyz="0.15 0 1"/><axis xyz="0 0 1"/><limit lower="-0.5" upper="0.5" effort="10" velocity="10"/>
                <mimic joint="left_hinge" multiplier="-1"/>
              </joint>
            </robot>
        "#;
        let options = UrdfLoadOptions {
            world: ArticulatedWorldParams {
                gravity: [0.0; 3],
                ..Default::default()
            },
            ..Default::default()
        };
        let mut cpu = load_urdf_str(xml, options.clone()).unwrap();
        assert_eq!(cpu.world.articulation.dof(), 1);
        assert_eq!(cpu.joints[0].dofs, 0..1);
        assert_eq!(cpu.joints[1].dofs, 0..1);
        assert_eq!(cpu.world.articulation.joint_limit(0), Some((-0.5, 0.5)));
        assert_eq!(
            cpu.joints[1].mimic.as_ref().unwrap().source_joint,
            "left_hinge"
        );
        cpu.world.step(0.01, &[0.0]).unwrap();
        assert!(cpu.world.velocities[0].abs() > 1e-5);
        let poses = cpu.world.link_poses().unwrap();
        let left_axis = poses[1].rotation * Vector3::x();
        let right_axis = poses[2].rotation * Vector3::x();
        assert!((left_axis.y + right_axis.y).abs() < 1e-8);

        #[cfg(feature = "gpu-contact")]
        if let Ok(context) = crate::gpu_contact_pipeline::GpuContactDevice::new() {
            let mut gpu = load_urdf_str(xml, options).unwrap();
            gpu.world.step_gpu(0.01, &[0.0], &context).unwrap();
            assert!((gpu.world.velocities[0] - cpu.world.velocities[0]).abs() < 1e-4);
        }
    }

    #[test]
    fn loads_joint_damping_and_reflects_mimic_multiplier() {
        let xml = r#"
            <robot name="damped"><link name="base"/><link name="a"/><link name="b"/>
              <joint name="source" type="continuous"><parent link="base"/><child link="a"/>
                <dynamics damping="0.5" friction="0.2"/></joint>
              <joint name="follower" type="continuous"><parent link="base"/><child link="b"/>
                <mimic joint="source" multiplier="-2"/>
                <dynamics damping="0.25" friction="0.3"/></joint>
            </robot>
        "#;
        let loaded = load_urdf_str(xml, UrdfLoadOptions::default()).unwrap();
        assert_eq!(loaded.world.articulation.dof(), 1);
        assert_eq!(loaded.world.joint_passive(0).unwrap().damping, 1.5);
        assert!((loaded.world.joint_friction(0).unwrap() - 0.8).abs() < 1e-12);
        let invalid = xml.replace("damping=\"0.25\"", "damping=\"-0.25\"");
        assert!(load_urdf_str(&invalid, UrdfLoadOptions::default()).is_err());
        let invalid = xml.replace("friction=\"0.3\"", "friction=\"-0.3\"");
        assert!(load_urdf_str(&invalid, UrdfLoadOptions::default()).is_err());
    }

    #[test]
    fn mimic_rejects_missing_sources_and_cycles() {
        let model = |source: &str| {
            format!(
                r#"
            <robot name="invalid"><link name="base"/><link name="a"/><link name="b"/>
              <joint name="first" type="continuous"><parent link="base"/><child link="a"/>
                <mimic joint="{source}"/></joint>
              <joint name="second" type="continuous"><parent link="base"/><child link="b"/>
                <mimic joint="first"/></joint>
            </robot>
        "#
            )
        };
        assert!(matches!(
            load_urdf_str(&model("missing"), UrdfLoadOptions::default()),
            Err(UrdfLoadError::Invalid(_))
        ));
        assert!(matches!(
            load_urdf_str(&model("second"), UrdfLoadOptions::default()),
            Err(UrdfLoadError::Invalid(_))
        ));
        let forward = r#"
            <robot name="forward"><link name="base"/><link name="a"/><link name="b"/>
              <joint name="first" type="continuous"><parent link="base"/><child link="a"/>
                <mimic joint="second"/></joint>
              <joint name="second" type="continuous"><parent link="base"/><child link="b"/></joint>
            </robot>
        "#;
        let loaded = load_urdf_str(forward, UrdfLoadOptions::default()).unwrap();
        assert_eq!(loaded.world.articulation.dof(), 1);
        assert_eq!(loaded.joints[0].dofs, 0..1);
        assert_eq!(loaded.joints[0].mimic.as_ref().unwrap().multiplier, 1.0);
    }

    #[test]
    fn supports_fixed_and_continuous_edges() {
        let xml = r#"
            <robot name="serial"><link name="base"/><link name="mount"/><link name="wheel">
              <inertial><mass value="1"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
            </link>
              <joint name="weld" type="fixed"><parent link="base"/><child link="mount"/></joint>
              <joint name="spin" type="continuous"><parent link="mount"/><child link="wheel"/><axis xyz="1 0 0"/></joint>
            </robot>
        "#;
        let loaded = load_urdf_str(xml, UrdfLoadOptions::default()).unwrap();
        assert_eq!(loaded.world.articulation.dof(), 1);
        assert_eq!(loaded.joints[0].kind, JointKind::Fixed);
        assert_eq!(loaded.joints[0].dofs, 0..0);
        assert_eq!(loaded.joints[1].kind, JointKind::Revolute);
        assert_eq!(loaded.joints[1].dofs, 0..1);
        assert_eq!(loaded.world.articulation.joint_limit(0), None);
    }

    #[test]
    fn zero_length_capsule_becomes_one_sphere() {
        let xml = r#"
            <robot name="capsule"><link name="base">
              <inertial><mass value="1"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
              <collision><origin xyz="1 2 3"/><geometry><capsule radius="0.2" length="0"/></geometry></collision>
            </link></robot>
        "#;
        let loaded = load_urdf_str(xml, UrdfLoadOptions::default()).unwrap();
        assert!(loaded.world.cylinders.is_empty());
        assert_eq!(loaded.world.colliders.len(), 1);
        assert_eq!(
            loaded.world.colliders[0].center,
            Vector3::new(1.0, 2.0, 3.0)
        );
    }

    #[test]
    fn loaded_world_steps_with_finite_state() {
        let mut loaded = load_urdf_str(ROBOT, UrdfLoadOptions::default()).unwrap();
        loaded.world.step(0.002, &[0.0; 5]).unwrap();
        assert!(loaded.world.positions.iter().all(|value| value.is_finite()));
        assert!(
            loaded
                .world
                .velocities
                .iter()
                .all(|value| value.is_finite())
        );
    }

    fn tetrahedron() -> ConvexGeometry {
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
