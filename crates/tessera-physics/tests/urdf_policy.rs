//! Native URDF policy semantics, matching rapier3d-urdf 0.36 length units.

use nalgebra::{Isometry3, Matrix3, UnitQuaternion, Vector3};
use tessera_physics::convex::ConvexGeometry;
use tessera_physics::urdf::{
    UrdfLoadError, UrdfLoadOptions, load_urdf_str, load_urdf_str_with_mesh_resolver,
};

type TestResult = Result<(), Box<dyn core::error::Error>>;

const MODEL: &str = r#"<robot name="policies">
  <link name="base">
    <inertial><origin xyz="1 2 3" rpy="0 0 1.5707963267948966"/>
      <mass value="7"/><inertia ixx="2" ixy="0" ixz="0" iyy="3" iyz="0" izz="4"/>
    </inertial>
    <collision><origin xyz="1 0 0"/><geometry><box size="2 4 6"/></geometry></collision>
    <collision><geometry><cylinder radius="2" length="4"/></geometry></collision>
    <collision><geometry><capsule radius="1" length="6"/></geometry></collision>
    <visual><origin xyz="0 1 0"/><geometry><sphere radius="3"/></geometry></visual>
  </link>
  <link name="child"/>
  <joint name="slide" type="prismatic"><parent link="base"/><child link="child"/>
    <origin xyz="1 2 3"/><axis xyz="1 0 0"/>
    <limit lower="-1" upper="2" effort="5" velocity="6"/>
  </joint>
</robot>"#;

#[test]
fn global_scale_changes_lengths_but_not_imported_mass_or_inertia() -> TestResult {
    let normal = load_urdf_str(MODEL, UrdfLoadOptions::default())?;
    let scaled = load_urdf_str(
        MODEL,
        UrdfLoadOptions {
            scale: 2.0,
            ..Default::default()
        },
    )?;
    let original = normal
        .world
        .articulation
        .link(0)
        .ok_or("missing fixture link")?;
    let link = scaled
        .world
        .articulation
        .link(0)
        .ok_or("missing fixture link")?;
    assert_eq!(link.mass, 7.0);
    assert_eq!(link.inertia, original.inertia);
    assert_eq!(link.center_of_mass, Vector3::new(2.0, 4.0, 6.0));
    assert_eq!(
        scaled.world.boxes[0].half_extents,
        Vector3::new(2.0, 4.0, 6.0)
    );
    assert_eq!(
        scaled.world.boxes[0].origin.translation.vector,
        Vector3::new(2.0, 0.0, 0.0)
    );
    assert_eq!(scaled.world.cylinders[0].radius, 4.0);
    assert_eq!(scaled.world.cylinders[0].half_height, 4.0);
    assert_eq!(scaled.world.cylinders[1].radius, 2.0);
    assert_eq!(scaled.world.cylinders[1].half_height, 6.0);
    assert_eq!(scaled.world.colliders[0].radius, 2.0);
    assert_eq!(scaled.world.colliders[0].center.z, -6.0);
    assert_eq!(scaled.world.articulation.joint_limit(0), Some((-2.0, 4.0)));
    assert_eq!(
        scaled.world.link_poses()?[1].translation.vector,
        Vector3::new(2.0, 4.0, 6.0)
    );
    assert_eq!(scaled.joints[0].effort_limit, Some(5.0));
    assert_eq!(scaled.joints[0].velocity_limit, Some(6.0));
    assert_eq!(scaled.options.scale, 2.0);
    Ok(())
}

#[test]
fn shift_is_applied_in_world_space_after_scaled_link_positions() -> TestResult {
    let root = Isometry3::translation(1.0, 0.0, 0.0);
    let shift = Isometry3::translation(0.0, 10.0, 0.0)
        * Isometry3::rotation(Vector3::z() * core::f64::consts::FRAC_PI_2);
    for floating_base in [false, true] {
        let loaded = load_urdf_str(
            MODEL,
            UrdfLoadOptions {
                scale: 2.0,
                root_pose: root,
                shift,
                floating_base,
                ..Default::default()
            },
        )?;
        let expected = shift * root * Isometry3::translation(2.0, 4.0, 6.0);
        assert!(
            (loaded.world.link_poses()?[1].translation.vector - expected.translation.vector).norm()
                < 1e-12
        );
        assert_eq!(loaded.world.root_pose, shift * root);
    }
    Ok(())
}

#[test]
fn disabling_imported_mass_props_produces_zero_density_mass_properties() -> TestResult {
    let loaded = load_urdf_str(
        MODEL,
        UrdfLoadOptions {
            apply_imported_mass_props: false,
            ..Default::default()
        },
    )?;
    let link = loaded
        .world
        .articulation
        .link(0)
        .ok_or("missing fixture link")?;
    assert_eq!(link.mass, 0.0);
    assert_eq!(link.center_of_mass, Vector3::zeros());
    assert_eq!(link.inertia, Matrix3::zeros());
    assert_eq!(loaded.world.boxes.len(), 1);
    Ok(())
}

#[test]
fn collision_and_visual_selection_are_independent_and_additive() -> TestResult {
    for collision in [false, true] {
        for visual in [false, true] {
            let loaded = load_urdf_str(
                MODEL,
                UrdfLoadOptions {
                    create_colliders_from_collision_shapes: collision,
                    create_colliders_from_visual_shapes: visual,
                    ..Default::default()
                },
            )?;
            assert_eq!(loaded.world.boxes.len(), usize::from(collision));
            assert_eq!(
                loaded.world.colliders.len(),
                2 * usize::from(collision) + usize::from(visual)
            );
            if visual {
                let sphere = loaded
                    .world
                    .colliders
                    .last()
                    .ok_or("missing selected visual collider")?;
                assert_eq!(sphere.radius, 3.0);
                assert_eq!(sphere.center, Vector3::y());
            }
        }
    }
    Ok(())
}

#[test]
fn logical_compound_joint_endpoints_are_excluded_and_metadata_is_retained() -> TestResult {
    let xml = r#"<robot name="compound"><link name="base"/><link name="child"/>
      <joint name="plane" type="planar"><parent link="base"/><child link="child"/>
        <axis xyz="0 0 1"/><limit effort="7" velocity="8"/>
      </joint></robot>"#;
    let loaded = load_urdf_str(xml, UrdfLoadOptions::default())?;
    assert!(loaded.world.articulation.adjacent(0, 1));
    assert_eq!(loaded.link_index("child"), Some(1));
    assert_eq!(loaded.joint_components("plane").count(), 3);
    let rotation = loaded
        .joint("plane/rotation")
        .ok_or("missing planar rotation component")?;
    assert_eq!(rotation.source_kind, urdf_rs::JointType::Planar);
    assert_eq!(rotation.source_links, [0, 1]);
    assert_eq!(rotation.effort_limit, Some(7.0));
    assert_eq!(rotation.velocity_limit, Some(8.0));
    assert!(loaded.joint("missing").is_none());
    Ok(())
}

fn hull(scale: [f64; 3]) -> Result<ConvexGeometry, tessera_physics::convex::ConvexGeometryError> {
    ConvexGeometry::new(
        vec![
            Vector3::zeros(),
            Vector3::x() * scale[0],
            Vector3::y() * scale[1],
            Vector3::z() * scale[2],
        ],
        vec![
            -Vector3::x(),
            -Vector3::y(),
            -Vector3::z(),
            Vector3::new(1.0 / scale[0], 1.0 / scale[1], 1.0 / scale[2]).normalize(),
        ],
        vec![Vector3::x(), Vector3::y(), Vector3::z()],
    )
}

#[test]
fn mixed_meshes_receive_combined_scale_and_indexed_policy_never_falls_back() -> TestResult {
    let xml = r#"<robot name="mixed"><link name="base">
      <collision><geometry><box size="1 1 1"/></geometry></collision>
      <collision><origin xyz="1 0 0"/><geometry><mesh filename="a.obj" scale="2 3 4"/></geometry></collision>
      <collision><geometry><mesh filename="b.stl"/></geometry></collision>
      <visual><geometry><mesh filename="ignored.dae"/></geometry></visual>
    </link></robot>"#;
    let mut calls = Vec::new();
    let mut resolver = |name: &str, scale: [f64; 3]| {
        calls.push((name.to_owned(), scale));
        Ok(vec![hull(scale).map_err(|error| error.to_string())?])
    };
    let options = UrdfLoadOptions {
        scale: 2.0,
        ..Default::default()
    };
    let loaded = load_urdf_str_with_mesh_resolver(xml, options.clone(), &mut resolver)?;
    assert_eq!(loaded.world.boxes.len(), 1);
    assert_eq!(loaded.world.convex_shapes.len(), 2);
    assert_eq!(
        loaded.world.convex_shapes[0].geometry.vertices[1],
        Vector3::new(4.0, 0.0, 0.0)
    );
    assert_eq!(
        loaded.world.convex_shapes[0].origin.translation.vector,
        Vector3::new(2.0, 0.0, 0.0)
    );
    assert_eq!(
        calls,
        [
            ("a.obj".into(), [4.0, 6.0, 8.0]),
            ("b.stl".into(), [2.0; 3])
        ]
    );
    let mut resolver = |_: &str, _: [f64; 3]| -> Result<Vec<ConvexGeometry>, String> {
        panic!("indexed policy must not invoke a convex resolver")
    };
    assert!(matches!(
        load_urdf_str_with_mesh_resolver(xml, UrdfLoadOptions { convex_hull: false, ..options }, &mut resolver),
        Err(UrdfLoadError::Mesh { message, .. }) if message.contains("indexed mesh")
    ));
    // An unselected mesh does not make an otherwise representable model fail.
    let loaded = load_urdf_str(
        xml,
        UrdfLoadOptions {
            convex_hull: false,
            create_colliders_from_collision_shapes: false,
            ..Default::default()
        },
    )?;
    assert!(loaded.world.convex_shapes.is_empty());
    Ok(())
}

#[test]
fn invalid_options_and_unrepresentable_collision_policy_fail_explicitly() -> TestResult {
    for scale in [0.0, -1.0, f64::INFINITY, f64::NAN] {
        assert!(matches!(
            load_urdf_str(
                MODEL,
                UrdfLoadOptions {
                    scale,
                    ..Default::default()
                }
            ),
            Err(UrdfLoadError::Invalid(_))
        ));
    }
    let bad_shift = Isometry3::translation(f64::NAN, 0.0, 0.0);
    assert!(matches!(
        load_urdf_str(
            MODEL,
            UrdfLoadOptions {
                shift: bad_shift,
                ..Default::default()
            }
        ),
        Err(UrdfLoadError::Invalid(_))
    ));
    let bad_root = Isometry3::from_parts(
        nalgebra::Translation3::identity(),
        UnitQuaternion::new_unchecked(nalgebra::Quaternion::new(2.0, 0.0, 0.0, 0.0)),
    );
    assert!(matches!(
        load_urdf_str(
            MODEL,
            UrdfLoadOptions {
                root_pose: bad_root,
                ..Default::default()
            }
        ),
        Err(UrdfLoadError::Invalid(_))
    ));
    let loaded = load_urdf_str(
        MODEL,
        UrdfLoadOptions {
            enable_joint_collisions: true,
            ..Default::default()
        },
    )?;
    assert!(loaded.world.joint_contacts_enabled());
    Ok(())
}
