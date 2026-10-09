//! Behavioral verification of configurable resident contact recovery.
#![cfg(feature = "gpu-contact")]

use nalgebra::Isometry3;
use tessera_physics::articulated_world::ArticulatedWorldParams;
use tessera_physics::gpu_articulated_dynamics::GpuArticulatedDynamicsBatch;
use tessera_physics::gpu_contact_pipeline::GpuContactDevice;
use tessera_physics::gpu_contact_policy::GpuContactPolicy;
use tessera_physics::urdf::{UrdfLoadOptions, load_urdf_str};

#[test]
fn resident_contact_policy_controls_recovery_gain_speed_cap_and_penetration_tolerance()
-> Result<(), Box<dyn core::error::Error>> {
    // Given: a one-kilogram slider penetrating ground by 0.1 m, without gravity.
    let xml = r#"<robot name="penetrated_slider">
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
    let loaded = load_urdf_str(
        xml,
        UrdfLoadOptions {
            root_pose: Isometry3::translation(0.0, 0.0, -0.1),
            world: ArticulatedWorldParams {
                gravity: [0.0; 3],
                ..ArticulatedWorldParams::default()
            },
            ..UrdfLoadOptions::default()
        },
    )?;
    let context = GpuContactDevice::new()?;
    let batch = GpuArticulatedDynamicsBatch::new(
        &context,
        &[loaded.world.gpu_dynamics_input(&[0.0])?],
        0.005,
    )?;
    let initial = batch.readback()?;
    let cases = [
        (GpuContactPolicy::default(), 2.0),
        (
            GpuContactPolicy {
                position_gain: 0.01,
                ..GpuContactPolicy::default()
            },
            0.2,
        ),
        (
            GpuContactPolicy {
                max_correction_speed: 0.1,
                ..GpuContactPolicy::default()
            },
            0.1,
        ),
        (
            GpuContactPolicy {
                allowed_penetration: 0.2,
                ..GpuContactPolicy::default()
            },
            0.0,
        ),
    ];
    for invalid in [
        GpuContactPolicy {
            position_gain: -0.1,
            ..GpuContactPolicy::default()
        },
        GpuContactPolicy {
            position_gain: 1.1,
            ..GpuContactPolicy::default()
        },
        GpuContactPolicy {
            max_correction_speed: f32::INFINITY,
            ..GpuContactPolicy::default()
        },
        GpuContactPolicy {
            warm_start_coefficient: f32::NAN,
            ..GpuContactPolicy::default()
        },
        GpuContactPolicy {
            warm_start_coefficient: 1.1,
            ..GpuContactPolicy::default()
        },
        GpuContactPolicy {
            allowed_penetration: -0.1,
            ..GpuContactPolicy::default()
        },
    ] {
        assert!(batch.update_contact_policy(invalid).is_err());
    }

    // When: each policy is applied to the same initial resident state.
    for (policy, expected_velocity) in cases {
        batch.reset(&initial)?;
        batch.update_contact_policy(policy)?;
        batch.submit_steps(1)?;

        // Then: separating speed follows the native policy, not old shader literals.
        let velocity = batch.readback()?[0].velocities[0];
        assert!(
            (velocity - expected_velocity).abs() < 1e-5,
            "policy {policy:?}: velocity={velocity}, expected={expected_velocity}"
        );
    }
    Ok(())
}
