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
        [Some(1), Some(0), Some(1)]
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
    let retained = load_mjcf_str(&unsupported, MjcfLoadOptions::default()).unwrap();
    assert_eq!(
        retained.actuators.entries()[0].kind,
        MjcfActuatorKind::Unsupported
    );
    assert_eq!(
        retained.actuators.entries()[0].attributes["dyntype"],
        "filter"
    );
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
fn mjcf_actuator_unsupported_definitions_are_retained_and_undriven() {
    for actuator in [
        r#"<intvelocity joint="a"/>"#,
        r#"<muscle joint="a"/>"#,
        r#"<general joint="a" dyntype="filter"/>"#,
        r#"<general joint="a" gaintype="affine"/>"#,
        r#"<motor tendon="cable"/>"#,
        r#"<motor site="attachment" refsite="reference"/>"#,
        r#"<motor joint="a" gear="1 2 0 0 0 0"/>"#,
    ] {
        let mut loaded = load(actuator);
        let info = &loaded.actuators.entries()[0];
        assert_eq!(info.kind, MjcfActuatorKind::Unsupported);
        assert!(info.unsupported_reason.is_some());
        loaded.set_controls(&[f64::MAX]).unwrap();
        assert_eq!(loaded.actuators.efforts(&loaded.world).unwrap(), [0.0, 0.0]);
        loaded.step(0.01).unwrap();
        assert_eq!(loaded.world.positions.as_slice(), [0.0, 0.0]);
        assert_eq!(loaded.world.velocities.as_slice(), [0.0, 0.0]);
    }
    assert!(matches!(
        load_mjcf_str(
            &model(r#"<motor joint="missing"/>"#),
            MjcfLoadOptions::default()
        ),
        Err(MjcfLoadError::Invalid(_))
    ));
}

#[test]
fn mjcf_actuator_nested_plugin_configuration_is_not_an_extra_control() {
    let mut loaded = load(
        r#"<plugin name="plugin_drive" plugin="example">
        <config key="gain" value="100"/>
        </plugin><motor joint="b"/>"#,
    );
    assert_eq!(loaded.actuators.entries().len(), 2);
    assert_eq!(loaded.actuators.entries()[0].source_kind, "plugin");
    assert_eq!(
        loaded.actuators.entries()[0].attributes["plugin"],
        "example"
    );
    assert_eq!(loaded.actuators.entries()[1].name, "actuator_1");
    loaded.set_controls(&[100.0, 2.0]).unwrap();
    assert_eq!(loaded.actuators.efforts(&loaded.world).unwrap(), [0.0, 2.0]);
}

#[test]
fn mjcf_actuator_mixed_metadata_controls_and_native_drive() {
    let xml = model(
        r#"<motor name="drive" joint="b" gear="-2"/>
        <muscle name="activation" joint="a" class="muscle"/>
        <motor tendon="cable" gear="1 2 3 4 5 6"/>
        <general joint="a" dyntype="filter" dynprm="0.25" actrange="0 1" actlimited="true"/>
        <position name="servo" joint="a" kp="4" kv="2"/>"#,
    )
    .replace(
        "<worldbody>",
        r#"<default><default class="muscle">
        <muscle timeconst="0.01 0.04" ctrlrange="0 1"/>
        </default></default><worldbody>"#,
    );
    let mut loaded = load_mjcf_str(&xml, MjcfLoadOptions::default()).unwrap();
    let entries = loaded.actuators.entries();
    assert_eq!(
        entries.iter().map(|a| a.name.as_str()).collect::<Vec<_>>(),
        ["drive", "activation", "actuator_2", "actuator_3", "servo"]
    );
    assert_eq!(entries[1].source_kind, "muscle");
    assert_eq!(entries[1].attributes["timeconst"], "0.01 0.04");
    assert_eq!(entries[1].control_range, Some([0.0, 1.0]));
    assert_eq!(entries[2].attributes["tendon"], "cable");
    assert_eq!(entries[2].attributes["gear"], "1 2 3 4 5 6");
    assert_eq!(entries[2].coordinate, None);
    assert_eq!(entries[3].attributes["dyntype"], "filter");
    assert_eq!(entries[3].attributes["dynprm"], "0.25");
    assert_eq!(entries[3].attributes["actrange"], "0 1");
    assert_eq!(entries[3].attributes["actlimited"], "true");
    assert!(entries[0].unsupported_reason.is_none());
    assert!(entries[4].unsupported_reason.is_none());
    let controls = [3.0, 100.0, 200.0, 300.0, 0.5];
    loaded.set_controls(&controls).unwrap();
    for invalid in [vec![1.0; 4], vec![3.0, f64::NAN, 200.0, 300.0, 0.5]] {
        assert!(loaded.set_controls(&invalid).is_err());
        assert_eq!(loaded.actuators.controls(), controls);
    }
    assert_eq!(
        loaded.actuators.efforts(&loaded.world).unwrap(),
        [2.0, -6.0]
    );
    let mut reference = load(r#"<motor joint="b" gear="-2"/><position joint="a" kp="4" kv="2"/>"#);
    reference.set_controls(&[3.0, 0.5]).unwrap();
    loaded.step(0.02).unwrap();
    reference.step(0.02).unwrap();
    assert_eq!(loaded.world.positions, reference.world.positions);
    assert_eq!(loaded.world.velocities, reference.world.velocities);
}

#[test]
fn mjcf_actuator_malformed_supported_definitions_still_fail() {
    for actuator in [
        r#"<motor joint="missing"/>"#,
        r#"<motor joint="a" gear="NaN"/>"#,
        r#"<position joint="a" kp="-1"/>"#,
        r#"<velocity joint="a" kv="1 2"/>"#,
        r#"<motor joint="a" ctrlrange="2 -1"/>"#,
        r#"<motor joint="a" forcelimited="true"/>"#,
        r#"<motor joint="a" ctrllimited="maybe"/>"#,
    ] {
        assert!(matches!(
            load_mjcf_str(&model(actuator), MjcfLoadOptions::default()),
            Err(MjcfLoadError::Invalid(_))
        ));
    }
    assert!(matches!(
        load_mjcf_str(
            &model(r#"<damper joint="a" gainprm="-1"/>"#),
            MjcfLoadOptions::default()
        ),
        Err(MjcfLoadError::Unsupported(_))
    ));
}
