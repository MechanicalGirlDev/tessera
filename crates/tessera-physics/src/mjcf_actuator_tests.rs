//! Deterministic native MJCF actuator import and control regressions.

use super::*;

fn model(actuators: &str) -> String {
    format!(
        r#"<mujoco><option gravity="0 0 0"/>
      <worldbody><body name="root"><inertial mass="1" diaginertia="1 1 1"/>
        <body name="first"><joint name="a" type="slide" axis="1 0 0"/>
          <inertial mass="1" diaginertia="1 1 1"/></body>
        <body name="second"><joint name="b" type="hinge" axis="0 0 1"/>
          <inertial mass="1" diaginertia="1 1 1"/></body>
      </body></worldbody><actuator>{actuators}</actuator></mujoco>"#
    )
}

fn load(actuators: &str) -> LoadedMjcf {
    load_mjcf_str(&model(actuators), MjcfLoadOptions::default()).unwrap()
}

#[test]
fn mjcf_actuator_metadata_order_defaults_and_scalar_coordinates() {
    let xml = model(
        r#"<motor name="b_motor" joint="b" gear="-2"/>
      <position name="a_servo" joint="a" class="servo"/>
      <velocity joint="b" kv="3"/>"#,
    )
    .replace(
        "<worldbody>",
        r#"<default><default class="servo">
          <position kp="25" kv="4" ctrlrange="-1 1"/>
          </default></default><worldbody>"#,
    );
    let loaded = load_mjcf_str(&xml, MjcfLoadOptions::default()).unwrap();
    let entries = loaded.actuators.entries();
    assert_eq!(
        entries.iter().map(|a| a.name.as_str()).collect::<Vec<_>>(),
        ["b_motor", "a_servo", "actuator_2"]
    );
    assert_eq!(
        entries.iter().map(|a| a.coordinate).collect::<Vec<_>>(),
        [1, 0, 1]
    );
    assert_eq!(
        entries[1].kind,
        MjcfActuatorKind::Position { kp: 25.0, kv: 4.0 }
    );
    assert_eq!(entries[1].control_range, Some([-1.0, 1.0]));
    assert_eq!(loaded.actuators.controls(), [0.0; 3]);
}

#[test]
fn mjcf_actuator_general_defaults_inherit_and_child_kind_overrides() {
    let xml = model(r#"<position joint="a" class="servo"/>"#).replace(
        "<worldbody>",
        r#"<default><general ctrlrange="-2 2" forcerange="-3 3" kp="2"/>
          <default class="servo"><position kp="25" kv="4"/></default>
          </default><worldbody>"#,
    );
    let loaded = load_mjcf_str(&xml, MjcfLoadOptions::default()).unwrap();
    let info = &loaded.actuators.entries()[0];
    assert_eq!(info.kind, MjcfActuatorKind::Position { kp: 25.0, kv: 4.0 });
    assert_eq!(info.control_range, Some([-2.0, 2.0]));
    assert_eq!(info.force_range, Some([-3.0, 3.0]));
    let unsupported = xml.replace("kp=\"2\"", "dyntype=\"filter\"");
    assert!(matches!(
        load_mjcf_str(&unsupported, MjcfLoadOptions::default()),
        Err(MjcfLoadError::Unsupported(_))
    ));
}

#[test]
fn mjcf_actuator_distinct_geared_motors_drive_coordinate_order() {
    let mut loaded = load(r#"<motor joint="b" gear="-2"/><motor joint="a" gear="3"/>"#);
    let mut reference = load("");
    loaded.set_controls(&[1.0, 2.0]).unwrap();
    assert_eq!(
        loaded.actuators.efforts(&loaded.world).unwrap(),
        [6.0, -2.0]
    );
    loaded.step(0.01).unwrap();
    reference.world.step(0.01, &[6.0, -2.0]).unwrap();
    assert_eq!(loaded.world.positions, reference.world.positions);
    assert_eq!(loaded.world.velocities, reference.world.velocities);
    assert!(loaded.world.velocities[0] > 0.0);
    assert!(loaded.world.velocities[1] < 0.0);
}

#[test]
fn mjcf_actuator_position_servo_moves_and_damps() {
    let mut loaded = load(r#"<position joint="a" kp="25" kv="8"/>"#);
    loaded.set_controls(&[0.5]).unwrap();
    for _ in 0..1000 {
        loaded.step(0.002).unwrap();
    }
    assert!((loaded.world.positions[0] - 0.5).abs() < 0.01);
    assert!(loaded.world.velocities[0].abs() < 0.02);
    assert_eq!(loaded.world.positions[1], 0.0);
}

#[test]
fn mjcf_actuator_feedback_is_reevaluated_at_native_substep_boundaries() {
    // Given: identical position servos and held actuator-order controls.
    let mut frame = load(r#"<position joint="a" kp="100" kv="8"/>"#);
    let mut substeps = load(r#"<position joint="a" kp="100" kv="8"/>"#);
    frame.set_controls(&[0.5]).unwrap();
    substeps.set_controls(&[0.5]).unwrap();
    let dt = frame.world.params().max_substep;

    // When: a control frame advances the same duration as explicit native substeps.
    frame.step(dt * 40.0).unwrap();
    for _ in 0..40 {
        substeps.step(dt).unwrap();
    }

    // Then: feedback is not frozen at the beginning of the enclosing frame.
    assert!((&frame.world.positions - &substeps.world.positions).norm() < 1e-12);
    assert!((&frame.world.velocities - &substeps.world.velocities).norm() < 1e-12);
}

#[test]
fn mjcf_actuator_velocity_servo_and_reference_servo_gear_semantics() {
    let mut loaded = load(r#"<velocity joint="a" kv="4" gear="7"/>"#);
    loaded.set_controls(&[0.5]).unwrap();
    loaded.world.velocities[0] = 0.25;
    assert_eq!(loaded.actuators.efforts(&loaded.world).unwrap(), [1.0, 0.0]);
    for _ in 0..500 {
        loaded.step(0.002).unwrap();
    }
    assert!((loaded.world.velocities[0] - 0.5).abs() < 0.02);
    assert!(loaded.world.positions[0] > 0.3);
}

#[test]
fn mjcf_actuator_clamps_controls_and_signed_forces_and_sums_shared_joint() {
    let mut loaded = load(
        r#"<motor joint="a" gear="3" ctrlrange="-1 2" forcerange="-2 4"/>
      <motor joint="a" gear="2" ctrllimited="false" ctrlrange="-1 1"
      forcelimited="false" forcerange="-1 1"/>"#,
    );
    loaded.set_controls(&[9.0, 2.0]).unwrap();
    assert_eq!(loaded.actuators.efforts(&loaded.world).unwrap(), [8.0, 0.0]);
    loaded.set_controls(&[-9.0, 0.0]).unwrap();
    assert_eq!(
        loaded.actuators.efforts(&loaded.world).unwrap(),
        [-2.0, 0.0]
    );
    let mut reference = load("");
    loaded.step(0.01).unwrap();
    reference.world.step(0.01, &[-2.0, 0.0]).unwrap();
    assert_eq!(loaded.world.velocities, reference.world.velocities);
}

#[test]
fn mjcf_actuator_invalid_control_vector_is_atomic() {
    let mut loaded = load(r#"<motor joint="b"/><position joint="a"/>"#);
    loaded.set_controls(&[2.0, 0.5]).unwrap();
    let before = loaded.actuators.efforts(&loaded.world).unwrap();
    let positions = loaded.world.positions.clone();
    let velocities = loaded.world.velocities.clone();
    for invalid in [vec![3.0], vec![3.0, f64::NAN], vec![f64::INFINITY, 1.0]] {
        assert!(loaded.set_controls(&invalid).is_err());
        assert_eq!(loaded.actuators.controls(), [2.0, 0.5]);
        assert_eq!(loaded.actuators.efforts(&loaded.world).unwrap(), before);
        assert_eq!(loaded.world.positions, positions);
        assert_eq!(loaded.world.velocities, velocities);
    }
}

#[test]
fn mjcf_actuator_damper_and_general_reference_branches() {
    let mut loaded = load(
        r#"<damper joint="a" gainprm="3"/>
      <general joint="b" gainprm="4" biasprm="1 -5 -2" biastype="affine"/>"#,
    );
    loaded.world.positions[1] = 0.5;
    loaded.world.velocities[0] = 2.0;
    loaded.world.velocities[1] = 0.25;
    loaded.set_controls(&[-2.0, 1.0]).unwrap();
    assert_eq!(
        loaded.actuators.efforts(&loaded.world).unwrap(),
        [-12.0, 2.0]
    );
    let mut constant = load(r#"<general joint="a" gainprm="4" gear="-2"/>"#);
    constant.set_controls(&[3.0]).unwrap();
    assert_eq!(
        constant.actuators.efforts(&constant.world).unwrap(),
        [-24.0, 0.0]
    );
}

#[test]
fn mjcf_actuator_unsupported_dynamics_are_typed_errors() {
    for actuator in [
        r#"<intvelocity joint="a"/>"#,
        r#"<muscle joint="a"/>"#,
        r#"<general joint="a" dyntype="filter"/>"#,
        r#"<general joint="a" gaintype="affine"/>"#,
        r#"<motor tendon="cable"/>"#,
        r#"<motor joint="a" gear="1 2 0 0 0 0"/>"#,
    ] {
        assert!(matches!(
            load_mjcf_str(&model(actuator), MjcfLoadOptions::default()),
            Err(MjcfLoadError::Unsupported(_))
        ));
    }
    assert!(matches!(
        load_mjcf_str(
            &model(r#"<motor joint="missing"/>"#),
            MjcfLoadOptions::default()
        ),
        Err(MjcfLoadError::Invalid(_))
    ));
}
