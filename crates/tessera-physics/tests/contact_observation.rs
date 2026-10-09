//! Selected-link contact observations from an actual resident articulated batch.
#![cfg(feature = "gpu-contact")]

use nalgebra::Isometry3;
use tessera_physics::articulated_world::ArticulatedWorldParams;
use tessera_physics::gpu_articulated_dynamics::GpuArticulatedDynamicsBatch;
use tessera_physics::gpu_contact_pipeline::GpuContactDevice;
use tessera_physics::urdf::{UrdfLoadOptions, load_urdf_str};

#[test]
fn resident_contact_observation_tracks_only_selected_links_in_independent_environments()
-> Result<(), Box<dyn core::error::Error>> {
    // Given: identical vertical sliders, one touching ground and one in free fall.
    let xml = r#"<robot name="contact_slider">
      <link name="base"/>
      <link name="slider">
        <inertial><mass value="1"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
        <collision><geometry><sphere radius="0.5"/></geometry></collision>
      </link>
      <joint name="slide" type="prismatic">
        <parent link="base"/><child link="slider"/><origin xyz="0 0 0.5"/>
        <axis xyz="0 0 1"/><limit lower="-5" upper="5" effort="100" velocity="100"/>
      </joint>
    </robot>"#;
    let options = |height| UrdfLoadOptions {
        root_pose: Isometry3::translation(0.0, 0.0, height),
        world: ArticulatedWorldParams {
            max_substep: 0.005,
            ..ArticulatedWorldParams::default()
        },
        ..UrdfLoadOptions::default()
    };
    let supported = load_urdf_str(xml, options(0.0))?;
    let falling = load_urdf_str(xml, options(3.0))?;
    let context = GpuContactDevice::new()?;
    let batch = GpuArticulatedDynamicsBatch::new(
        &context,
        &[
            supported.world.gpu_dynamics_input(&[0.0])?,
            falling.world.gpu_dynamics_input(&[0.0])?,
        ],
        0.005,
    )?;
    let sensor = batch.normal_impulse_sensor(&[1, 0])?;
    assert!(batch.normal_impulse_sensor(&[]).is_err());
    assert!(batch.normal_impulse_sensor(&[1, 1]).is_err());
    assert!(batch.normal_impulse_sensor(&[2]).is_err());

    // When: contact solves and the last observation share one GPU command encoder.
    let mut encoder = context.device().create_command_encoder(&Default::default());
    for _ in 0..8 {
        batch.encode_step(&mut encoder)?;
    }
    sensor.encode(&mut encoder);
    let _submission = context.queue().submit(Some(encoder.finish()));
    let observed = sensor.readback(context.queue())?;
    let contacts = batch.readback_contacts()?;
    let states = batch.readback()?;

    // Then: only the supported slider has normal impulses, in selected-link order.
    assert!(observed[0][0] > 0.04 && observed[0][0] < 0.06);
    assert!(observed[0][1].abs() < 1e-7);
    assert!(observed[1].iter().all(|value| value.abs() < 1e-7));
    let normal_impulse: f64 = contacts[0]
        .iter()
        .filter(|contact| contact.link == 1)
        .map(|contact| contact.force.dot(&contact.normal).max(0.0) * 0.005)
        .sum();
    assert!((f64::from(observed[0][0]) - normal_impulse).abs() < 1e-6);
    assert!(states[0].positions[0].abs() < 0.01);
    assert!(states[1].velocities[0] < -0.3);
    Ok(())
}
