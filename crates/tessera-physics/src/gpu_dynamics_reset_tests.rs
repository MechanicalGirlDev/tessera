use super::*;
use crate::articulation::{Articulation, JointKind, JointSpec, LinkSpec};
use crate::gpu_articulated_dynamics::GpuArticulatedDynamicsInput;
use crate::gpu_articulated_spherical::GpuSphericalJointState;
use crate::gpu_articulated_state::GpuGeneralizedState;
use crate::gpu_contact_pipeline::GpuContactDevice;
use nalgebra::{DVector, Isometry3, Matrix3, UnitQuaternion, Vector3};

#[test]
fn owner_reset_restores_floating_quaternion_and_drive_templates_without_cross_environment_writes()
-> Result<(), Box<dyn core::error::Error>> {
    // Given: two floating robots with different native spherical orientations.
    let link = LinkSpec {
        mass: 1.0,
        center_of_mass: Vector3::zeros(),
        inertia: Matrix3::identity(),
    };
    let art = Articulation::new(
        vec![link.clone(), link],
        vec![JointSpec {
            parent: 0,
            child: 1,
            kind: JointKind::Spherical,
            origin: Isometry3::identity(),
            axis: Vector3::z(),
            limits: None,
        }],
        0,
    )?;
    let initial = GpuGeneralizedState {
        positions: DVector::zeros(9),
        velocities: DVector::from_vec(vec![0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    };
    let roots = [
        Isometry3::translation(1.0, 2.0, 3.0),
        Isometry3::translation(4.0, 5.0, 6.0),
    ];
    let joints = [
        vec![GpuSphericalJointState {
            velocity_slot: 6,
            orientation: UnitQuaternion::from_scaled_axis(Vector3::z() * 0.1),
        }],
        vec![GpuSphericalJointState {
            velocity_slot: 6,
            orientation: UnitQuaternion::from_scaled_axis(Vector3::y() * -0.2),
        }],
    ];
    let zero_drive = SphericalJointDrive {
        orientation_target: UnitQuaternion::identity(),
        velocity_target: Vector3::zeros(),
        stiffness: Vector3::zeros(),
        damping: Vector3::zeros(),
        max_torque: Vector3::repeat(100.0),
    };
    let context = GpuContactDevice::new()?;
    let inputs = roots.map(|root| {
        GpuArticulatedDynamicsInput::new(&art, root, initial.clone(), Vector3::zeros())
    });
    let batch = GpuArticulatedDynamicsBatch::new_with_spherical_state(
        &context,
        &inputs,
        0.01,
        &[true, true],
        &joints,
        &[vec![Some(zero_drive)], vec![Some(zero_drive)]],
    )?;
    let mut encoder = context.device().create_command_encoder(&Default::default());
    let templates = batch.snapshot_reset_templates(&mut encoder)?;
    let _submission = context.queue().submit(Some(encoder.finish()));
    batch.update_spherical_drives(&[
        vec![Some(SphericalJointDrive {
            orientation_target: UnitQuaternion::from_scaled_axis(Vector3::z() * 0.5),
            stiffness: Vector3::repeat(10.0),
            ..zero_drive
        })],
        vec![Some(zero_drive)],
    ])?;
    batch.submit_steps(2)?;
    let before = batch.readback_output()?;

    // When: environment one resets from environment zero's initial device template.
    let mut encoder = context.device().create_command_encoder(&Default::default());
    batch.encode_reset_envs_from_templates(
        &mut encoder,
        &templates,
        &[GpuArticulatedResetSelection {
            environment: 1,
            template: 0,
            root_translation: Some([10.0, 0.0, 0.0]),
            root_velocity: Some([1.0, 2.0, 3.0, 0.0, 0.0, 0.0]),
        }],
    )?;
    let _submission = context.queue().submit(Some(encoder.finish()));
    let reset = batch.readback_output()?;
    batch.submit_steps(1)?;
    let stepped = batch.readback_output()?;

    // Then: the selected root/quaternion/drives restore and other controls remain live.
    assert_eq!(reset[0].state, before[0].state);
    assert_eq!(reset[0].root_pose, before[0].root_pose);
    assert_eq!(
        reset[1].root_pose.translation.vector,
        Vector3::new(11.0, 2.0, 3.0)
    );
    assert_eq!(
        reset[1].state.velocities.as_slice(),
        &[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    );
    let orientation = reset[1]
        .spherical_joints
        .as_ref()
        .ok_or("missing native quaternion output")?[0]
        .orientation;
    assert!((orientation.inverse() * joints[0][0].orientation).angle() < 1e-6);
    assert!(stepped[1].state.velocities.rows(6, 3).norm() < 1e-6);
    assert!(stepped[0].state.velocities[8] > before[0].state.velocities[8]);
    assert!(
        (stepped[1].root_pose.translation.vector - Vector3::new(11.01, 2.02, 3.03)).norm() < 1e-5
    );
    Ok(())
}
