//! Native loader policy and complete imported joint metadata.

use super::*;
use tessera_physics::articulation::JointKind as CoreJointKind;

/// Policies used when importing a URDF robot.
#[derive(Clone, Debug, uniffi::Record)]
pub struct UrdfOptions {
    /// Uniform length scale, leaving mass and inertia magnitudes unchanged.
    pub scale: f64,
    /// World transform left-multiplied onto the scaled root pose.
    pub shift: LinkPose,
    /// Import the document's mass, COM, and inertia.
    pub apply_imported_mass_props: bool,
    /// Create collision geometry from collision elements.
    pub create_colliders_from_collision_shapes: bool,
    /// Create collision geometry from visual elements.
    pub create_colliders_from_visual_shapes: bool,
    /// Allow contacts between links connected by joints.
    pub enable_joint_collisions: bool,
    /// Resolve external meshes as convex parts.
    pub convex_hull: bool,
    /// World root pose before applying shift.
    pub root_pose: LinkPose,
    /// Include floating-root dynamics.
    pub floating_base: bool,
}

impl UrdfOptions {
    pub(super) fn native(self) -> Result<UrdfLoadOptions, TesseraError> {
        Ok(UrdfLoadOptions {
            scale: self.scale,
            shift: link_pose_isometry(self.shift)?,
            apply_imported_mass_props: self.apply_imported_mass_props,
            create_colliders_from_collision_shapes: self.create_colliders_from_collision_shapes,
            create_colliders_from_visual_shapes: self.create_colliders_from_visual_shapes,
            enable_joint_collisions: self.enable_joint_collisions,
            convex_hull: self.convex_hull,
            root_pose: link_pose_isometry(self.root_pose)?,
            floating_base: self.floating_base,
            ..Default::default()
        })
    }
}

/// Native generalized joint family.
#[derive(Clone, Copy, Debug, uniffi::Enum)]
pub enum JointType {
    /// Zero generalized coordinates.
    Fixed,
    /// One angular coordinate.
    Revolute,
    /// One linear coordinate.
    Prismatic,
    /// Three angular coordinates or an explicit quaternion with tangent velocity.
    Spherical,
}

impl From<CoreJointKind> for JointType {
    fn from(kind: CoreJointKind) -> Self {
        match kind {
            CoreJointKind::Fixed => Self::Fixed,
            CoreJointKind::Revolute => Self::Revolute,
            CoreJointKind::Prismatic => Self::Prismatic,
            CoreJointKind::Spherical => Self::Spherical,
        }
    }
}

/// An imported affine joint dependency.
#[derive(Clone, Debug, uniffi::Record)]
pub struct JointMimic {
    /// Name of the source joint from the document.
    pub source_joint: String,
    /// Coordinate multiplier.
    pub multiplier: f64,
    /// Coordinate offset.
    pub offset: f64,
}

/// Original source identity, physical endpoints, and native coordinate mapping.
#[derive(Clone, Debug, uniffi::Record)]
pub struct JointInfo {
    /// Stable native component name.
    pub name: String,
    /// Original joint name before planar/floating expansion.
    pub source_name: String,
    /// Original source kind, preserving continuous/planar/floating distinctions.
    pub source_kind: String,
    /// Native joint family.
    pub kind: JointType,
    /// Physical parent link, or none for a floating root.
    pub parent_link: Option<u32>,
    /// Physical child link.
    pub child_link: u32,
    /// Original physical links before URDF virtual-frame expansion.
    pub source_links: Vec<u32>,
    /// Native generalized-coordinate interval.
    pub coordinates: JointRange,
    /// Affine dependency when imported from a URDF mimic joint.
    pub mimic: Option<JointMimic>,
    /// Imported positive effort limit.
    pub effort_limit: Option<f64>,
    /// Imported positive velocity limit.
    pub velocity_limit: Option<f64>,
}

pub(super) fn urdf_joints(loaded: &LoadedUrdf) -> Result<Vec<JointInfo>, TesseraError> {
    loaded
        .joints
        .iter()
        .map(|joint| {
            Ok(JointInfo {
                name: joint.name.clone(),
                source_name: joint.source_name.clone(),
                source_kind: format!("{:?}", joint.source_kind),
                kind: joint.kind.into(),
                parent_link: Some(u32::try_from(joint.parent_link).map_err(failed)?),
                child_link: u32::try_from(joint.child_link).map_err(failed)?,
                source_links: joint
                    .source_links
                    .iter()
                    .copied()
                    .map(|link| u32::try_from(link).map_err(failed))
                    .collect::<Result<_, _>>()?,
                coordinates: JointRange {
                    name: joint.name.clone(),
                    start: u32::try_from(joint.dofs.start).map_err(failed)?,
                    end: u32::try_from(joint.dofs.end).map_err(failed)?,
                },
                mimic: joint.mimic.as_ref().map(|mimic| JointMimic {
                    source_joint: mimic.source_joint.clone(),
                    multiplier: mimic.multiplier,
                    offset: mimic.offset,
                }),
                effort_limit: joint.effort_limit,
                velocity_limit: joint.velocity_limit,
            })
        })
        .collect()
}

pub(super) fn mjcf_joints(loaded: &LoadedMjcf) -> Result<Vec<JointInfo>, TesseraError> {
    loaded
        .joints
        .iter()
        .map(|joint| {
            let parent = loaded
                .world
                .articulation
                .joint_specs()
                .iter()
                .find(|spec| spec.child == joint.child_link)
                .map(|spec| spec.parent);
            Ok(JointInfo {
                name: joint.name.clone(),
                source_name: joint.name.clone(),
                source_kind: format!("{:?}", joint.kind),
                kind: joint.kind.into(),
                parent_link: parent.map(u32::try_from).transpose().map_err(failed)?,
                child_link: u32::try_from(joint.child_link).map_err(failed)?,
                source_links: parent
                    .into_iter()
                    .chain([joint.child_link])
                    .map(|link| u32::try_from(link).map_err(failed))
                    .collect::<Result<_, _>>()?,
                coordinates: JointRange {
                    name: joint.name.clone(),
                    start: u32::try_from(joint.dofs.start).map_err(failed)?,
                    end: u32::try_from(joint.dofs.end).map_err(failed)?,
                },
                mimic: None,
                effort_limit: None,
                velocity_limit: None,
            })
        })
        .collect()
}

#[uniffi::export]
impl ArticulatedWorld {
    /// Import a URDF with explicit native loader policies and external meshes.
    #[uniffi::constructor]
    pub fn from_urdf_with_options(
        xml: String,
        options: UrdfOptions,
        assets: Vec<MeshAsset>,
    ) -> Result<Arc<Self>, TesseraError> {
        let mut resolver = MeshAssetResolver {
            assets: mesh_assets(assets)?,
        };
        let loaded = load_urdf_str_with_mesh_resolver(&xml, options.native()?, &mut resolver)
            .map_err(failed)?;
        Self::from_loaded_urdf(loaded)
    }

    /// Complete imported joint metadata, including physical endpoints and mimics.
    pub fn joints(&self) -> Vec<JointInfo> {
        self.joint_metadata.clone()
    }

    /// Find a physical or generated link by its stable name.
    pub fn link_index(&self, name: String) -> Option<u32> {
        self.link_names
            .iter()
            .position(|candidate| candidate == &name)
            .and_then(|index| u32::try_from(index).ok())
    }

    /// Find a native component or all components of one original source joint.
    pub fn joint_components(&self, source_name: String) -> Vec<JointInfo> {
        self.joint_metadata
            .iter()
            .filter(|joint| joint.source_name == source_name)
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loader_metadata_resolves_indexed_assets_for_single_and_batch_urdf()
    -> Result<(), TesseraError> {
        let xml = r#"<robot name="indexed"><link name="base">
          <collision><geometry><mesh filename="surface.obj" scale="2 3 4"/></geometry></collision>
          </link></robot>"#;
        let pose = isometry_link_pose(Isometry3::identity());
        let options = UrdfOptions {
            scale: 2.0,
            shift: pose,
            root_pose: pose,
            floating_base: false,
            apply_imported_mass_props: true,
            create_colliders_from_collision_shapes: true,
            create_colliders_from_visual_shapes: false,
            enable_joint_collisions: true,
            convex_hull: false,
        };
        let asset = MeshAsset {
            uri: "surface.obj".into(),
            parts: vec![],
            indexed_parts: vec![IndexedMeshPart {
                vertices: vec![
                    Vec3 {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                    },
                    Vec3 {
                        x: 1.0,
                        y: 0.0,
                        z: 0.0,
                    },
                    Vec3 {
                        x: 0.0,
                        y: 1.0,
                        z: 0.0,
                    },
                ],
                triangles: vec![GpuTriangle { a: 0, b: 1, c: 2 }],
            }],
        };
        let robot = ArticulatedWorld::from_urdf_with_options(
            xml.into(),
            options.clone(),
            vec![asset.clone()],
        )?;
        let inner = locked(&robot.inner)?;
        assert!(inner.world.convex_shapes.is_empty());
        assert_eq!(inner.world.triangle_meshes().len(), 1);
        assert_eq!(
            inner.world.triangle_meshes()[0].geometry.vertices()[1].x,
            4.0
        );
        assert!(inner.world.joint_contacts_enabled());
        drop(inner);
        let batch = ArticulatedBatch::new(vec![BatchModel {
            format: ModelFormat::Urdf,
            xml: xml.into(),
            floating_base: false,
            meshes: vec![asset.clone()],
            urdf_options: Some(options),
        }])?;
        assert_eq!(
            locked(&batch.inner)?.batch.environments()[0]
                .triangle_meshes()
                .len(),
            1
        );
        let mut invalid = asset;
        invalid.indexed_parts[0].triangles[0].c = 5;
        assert!(mesh_assets(vec![invalid]).is_err());
        Ok(())
    }

    #[test]
    fn loader_metadata_keeps_original_joint_identity_and_loader_transform()
    -> Result<(), TesseraError> {
        let xml = r#"<robot name="metadata">
          <link name="base"><inertial><mass value="1"/>
            <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
          </inertial></link>
          <link name="tip"><inertial><mass value="1"/>
            <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
          </inertial></link>
          <joint name="slide" type="prismatic"><parent link="base"/><child link="tip"/>
            <axis xyz="1 0 0"/><limit lower="-1" upper="1" effort="3" velocity="4"/>
          </joint></robot>"#;
        let pose = isometry_link_pose(Isometry3::identity());
        let robot = ArticulatedWorld::from_urdf_with_options(
            xml.into(),
            UrdfOptions {
                scale: 2.0,
                shift: LinkPose {
                    position: Vec3 {
                        x: 1.0,
                        y: 2.0,
                        z: 3.0,
                    },
                    ..pose
                },
                apply_imported_mass_props: true,
                create_colliders_from_collision_shapes: true,
                create_colliders_from_visual_shapes: false,
                enable_joint_collisions: false,
                convex_hull: true,
                root_pose: pose,
                floating_base: false,
            },
            Vec::new(),
        )?;

        let joints = robot.joint_components("slide".into());

        assert_eq!(joints.len(), 1);
        assert_eq!(joints[0].source_kind, "Prismatic");
        assert!(matches!(joints[0].kind, JointType::Prismatic));
        assert_eq!(joints[0].parent_link, Some(0));
        assert_eq!(joints[0].child_link, 1);
        assert_eq!(joints[0].source_links, vec![0, 1]);
        assert_eq!(joints[0].effort_limit, Some(3.0));
        assert_eq!(joints[0].velocity_limit, Some(4.0));
        assert_eq!(robot.link_index("tip".into()), Some(1));
        assert_eq!(
            robot.simulation_state()?.configuration.root_pose.position.z,
            3.0
        );
        Ok(())
    }
}
