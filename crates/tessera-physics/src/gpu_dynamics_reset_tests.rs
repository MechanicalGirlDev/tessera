use super::*;
use crate::articulation::{Articulation, JointKind, JointSpec, LinkSpec};
use crate::gpu_articulated_dynamics::GpuArticulatedDynamicsInput;
use crate::gpu_articulated_spherical::GpuSphericalJointState;
use crate::gpu_articulated_state::GpuGeneralizedState;
use crate::gpu_contact_pipeline::GpuContactDevice;
use nalgebra::{DVector, Isometry3, Matrix3, UnitQuaternion, Vector3};

fn physical_model(
    mass: f64,
    armature: f64,
    axis: Vector3<f64>,
    kind: JointKind,
) -> Result<Articulation, crate::articulation::ArticulationError> {
    let mut articulation = Articulation::new(
        vec![
            LinkSpec {
                mass: 1.0,
                center_of_mass: Vector3::zeros(),
                inertia: Matrix3::identity(),
            },
            LinkSpec {
                mass,
                center_of_mass: Vector3::new(0.1 * mass, 0.2, 0.3),
                inertia: Matrix3::from_diagonal(&Vector3::new(mass, mass * 2.0, mass * 3.0)),
            },
        ],
        vec![JointSpec {
            parent: 0,
            child: 1,
            kind,
            origin: Isometry3::translation(mass, 0.2, 0.3),
            axis,
            limits: Some((-mass, mass)),
        }],
        0,
    )?;
    articulation.set_joint_armature(0, &[armature])?;
    Ok(articulation)
}

#[test]
fn owner_reset_restores_selected_prescribed_masks_and_compliance()
-> Result<(), Box<dyn core::error::Error>> {
    use crate::prescribed_dof::ConstraintCompliance;
    let model = physical_model(2.0, 0.5, Vector3::x(), JointKind::Prismatic)?;
    let context = GpuContactDevice::new()?;
    let inputs = [0, 1].map(|_| {
        GpuArticulatedDynamicsInput::new(
            &model,
            Isometry3::identity(),
            GpuGeneralizedState {
                positions: DVector::zeros(1),
                velocities: DVector::from_element(1, 0.25),
            },
            Vector3::zeros(),
        )
    });
    let mut batch = GpuArticulatedDynamicsBatch::new(&context, &inputs, 0.01)?;
    let saved = ConstraintCompliance {
        erp: 0.3,
        cfm: 0.002,
    };
    let changed = ConstraintCompliance {
        erp: 0.4,
        cfm: 0.003,
    };
    batch.set_prescribed_coordinates(&[vec![true], vec![false]])?;
    batch.set_joint_limit_compliance(&[vec![saved], vec![ConstraintCompliance::default()]])?;
    let mut encoder = context.device().create_command_encoder(&Default::default());
    let templates = batch.snapshot_reset_templates(&mut encoder)?;
    let _submission = context.queue().submit(Some(encoder.finish()));
    batch.set_prescribed_coordinates(&[vec![false], vec![false]])?;
    batch.set_joint_limit_compliance(&[vec![changed], vec![changed]])?;
    let mut encoder = context.device().create_command_encoder(&Default::default());
    batch.encode_reset_envs_from_templates(
        &mut encoder,
        &templates,
        &[GpuArticulatedResetSelection {
            environment: 1,
            template: 0,
            root_translation: None,
            root_velocity: None,
        }],
    )?;
    let _submission = context.queue().submit(Some(encoder.finish()));
    assert_eq!(batch.prescribed_coordinates(), &[vec![false], vec![true]]);
    assert_eq!(
        batch.joint_limit_compliance(),
        &[vec![changed], vec![saved]]
    );
    batch.update_joints(&vec![
        vec![GpuJointForceInput {
            base_force: 10.0,
            ..Default::default()
        }];
        2
    ])?;
    batch.submit_steps(1)?;
    let output = batch.readback_output()?;
    assert!(output[0].state.velocities[0] > 0.28);
    assert!((output[1].state.velocities[0] - 0.25).abs() < 1e-6);
    Ok(())
}

#[test]
fn owner_reset_restores_selected_imported_held_controls() -> Result<(), Box<dyn core::error::Error>>
{
    use crate::mjcf::{MjcfLoadOptions, load_mjcf_str};
    let imported = load_mjcf_str(
        r#"<mujoco><worldbody><body><joint name="slide" type="slide" axis="1 0 0"/>
        <geom type="sphere" size=".1" mass="1"/></body></worldbody>
        <actuator><motor joint="slide"/></actuator></mujoco>"#,
        MjcfLoadOptions::default(),
    )?;
    let model = physical_model(2.0, 0.5, Vector3::x(), JointKind::Prismatic)?;
    let context = GpuContactDevice::new()?;
    let inputs = [0, 1].map(|_| {
        GpuArticulatedDynamicsInput::new(
            &model,
            Isometry3::identity(),
            GpuGeneralizedState {
                positions: DVector::zeros(1),
                velocities: DVector::zeros(1),
            },
            Vector3::zeros(),
        )
    });
    let mut batch = GpuArticulatedDynamicsBatch::new(&context, &inputs, 0.01)?;
    let mut actuator = imported.actuators;
    actuator.set_controls(&[5.0])?;
    batch.enable_mjcf_actuators(&[actuator.clone(), actuator], &[0, 0])?;
    let mut encoder = context.device().create_command_encoder(&Default::default());
    let templates = batch.snapshot_reset_templates(&mut encoder)?;
    let _submission = context.queue().submit(Some(encoder.finish()));
    batch.update_mjcf_controls(0, &[10.0])?;
    batch.update_mjcf_controls(1, &[-10.0])?;
    let mut encoder = context.device().create_command_encoder(&Default::default());
    batch.encode_reset_envs_from_templates(
        &mut encoder,
        &templates,
        &[GpuArticulatedResetSelection {
            environment: 1,
            template: 0,
            root_translation: None,
            root_velocity: None,
        }],
    )?;
    let _submission = context.queue().submit(Some(encoder.finish()));
    batch.submit_steps(1)?;
    let output = batch.readback_output()?;
    assert!((output[0].state.velocities[0] - 0.04).abs() < 1e-6);
    assert!((output[1].state.velocities[0] - 0.02).abs() < 1e-6);
    Ok(())
}

#[test]
fn owner_reset_restores_physical_descriptors_loads_and_armature_with_destination_indices()
-> Result<(), Box<dyn core::error::Error>> {
    use crate::gpu_articulated_force::GpuMassLinkLoad;
    use crate::gpu_articulated_mass::read_buffer;
    let source = physical_model(2.0, 0.5, Vector3::x(), JointKind::Prismatic)?;
    let target = physical_model(8.0, 3.0, Vector3::y(), JointKind::Prismatic)?;
    let state = GpuGeneralizedState {
        positions: DVector::from_element(1, 0.1),
        velocities: DVector::zeros(1),
    };
    let mut inputs = [&source, &target].map(|articulation| {
        GpuArticulatedDynamicsInput::new(
            articulation,
            Isometry3::identity(),
            state.clone(),
            Vector3::x() * 2.0,
        )
    });
    for input in &mut inputs {
        input.joints[0].base_force = 1.0;
        input.link_loads[1] = GpuMassLinkLoad {
            force: Vector3::x() * 5.0,
            torque: Vector3::zeros(),
            gravity_scale: 0.5,
        };
    }
    let context = GpuContactDevice::new()?;
    let mut batch = GpuArticulatedDynamicsBatch::new(&context, &inputs, 0.01)?;
    let mut encoder = context.device().create_command_encoder(&Default::default());
    let templates = batch.snapshot_reset_templates(&mut encoder)?;
    let _submission = context.queue().submit(Some(encoder.finish()));
    batch.update_link_loads(&vec![vec![GpuMassLinkLoad::default(); 2]; 2])?;
    batch.update_gravity(&[Vector3::zeros(), Vector3::zeros()])?;
    batch.update_joints(&vec![
        vec![GpuJointForceInput {
            base_force: 3.0,
            ..Default::default()
        }];
        2
    ])?;
    batch.submit_steps(1)?;
    let before = batch.readback_output()?;
    let before_poses = batch.readback_link_poses()?;
    let mut encoder = context.device().create_command_encoder(&Default::default());
    let live = batch.snapshot_reset_templates(&mut encoder)?;
    let _submission = context.queue().submit(Some(encoder.finish()));

    let mut encoder = context.device().create_command_encoder(&Default::default());
    batch.encode_reset_envs_from_templates(
        &mut encoder,
        &templates,
        &[GpuArticulatedResetSelection {
            environment: 1,
            template: 0,
            root_translation: None,
            root_velocity: None,
        }],
    )?;
    let _submission = context.queue().submit(Some(encoder.finish()));
    let reset = batch.readback_output()?;
    assert_eq!(reset[0].state, before[0].state);
    assert_eq!(batch.readback_link_poses()?[0], before_poses[0]);
    assert!((reset[1].state.positions[0] - 0.1).abs() < 1e-6);
    assert!(reset[1].state.velocities[0].abs() < 1e-8);
    assert_eq!(batch.joint_bounds[1], batch.joint_bounds[0]);

    let mut encoder = context.device().create_command_encoder(&Default::default());
    let restored = batch.snapshot_reset_templates(&mut encoder)?;
    let _submission = context.queue().submit(Some(encoder.finish()));
    for (saved, changed, actual, stride, rows, payload) in [
        (
            &templates.link_metadata,
            &live.link_metadata,
            &restored.link_metadata,
            96,
            2,
            16..96,
        ),
        (
            &templates.pose_joints,
            &live.pose_joints,
            &restored.pose_joints,
            96,
            1,
            16..80,
        ),
        (
            &templates.link_loads,
            &live.link_loads,
            &restored.link_loads,
            32,
            2,
            0..32,
        ),
        (
            &templates.limits,
            &live.limits,
            &restored.limits,
            32,
            1,
            0..32,
        ),
        (
            &templates.gravities,
            &live.gravities,
            &restored.gravities,
            16,
            1,
            0..16,
        ),
    ] {
        let saved = read_buffer(context.device(), context.queue(), saved)?;
        let changed = read_buffer(context.device(), context.queue(), changed)?;
        let actual = read_buffer(context.device(), context.queue(), actual)?;
        let environment_bytes = stride * rows;
        assert_eq!(&actual[..environment_bytes], &changed[..environment_bytes]);
        for row in 0..rows {
            let source = row * stride;
            let destination = environment_bytes + source;
            assert_eq!(
                &actual[destination + payload.start..destination + payload.end],
                &saved[source + payload.start..source + payload.end],
            );
            assert_eq!(
                &actual[destination..destination + payload.start],
                &changed[destination..destination + payload.start],
            );
        }
    }
    // Updating drives after reset must use the restored CPU armature baseline too.
    let mut next_drives = vec![
        vec![GpuJointForceInput {
            base_force: 3.0,
            ..Default::default()
        }];
        2
    ];
    next_drives[1][0].base_force = 1.0;
    batch.update_joints(&next_drives)?;
    batch.submit_steps(1)?;
    let stepped = batch.readback_output()?;
    // (base effort 1 + persistent load 5 + gravity 2 * mass 2 * scale .5) / (2 + .5).
    assert!((stepped[1].state.velocities[0] - 0.032).abs() < 1e-6);
    assert!((stepped[1].state.positions[0] - 0.10032).abs() < 1e-6);
    Ok(())
}

#[test]
fn owner_reset_rejects_equal_dof_incompatible_topology_before_any_encoding()
-> Result<(), Box<dyn core::error::Error>> {
    let prismatic = physical_model(2.0, 0.5, Vector3::x(), JointKind::Prismatic)?;
    let revolute = physical_model(2.0, 0.5, Vector3::x(), JointKind::Revolute)?;
    let context = GpuContactDevice::new()?;
    let inputs = [&prismatic, &revolute].map(|model| {
        GpuArticulatedDynamicsInput::new(
            model,
            Isometry3::identity(),
            GpuGeneralizedState {
                positions: DVector::zeros(1),
                velocities: DVector::from_element(1, 0.25),
            },
            Vector3::zeros(),
        )
    });
    let mut batch = GpuArticulatedDynamicsBatch::new(&context, &inputs, 0.01)?;
    let mut encoder = context.device().create_command_encoder(&Default::default());
    let templates = batch.snapshot_reset_templates(&mut encoder)?;
    let _submission = context.queue().submit(Some(encoder.finish()));
    batch.submit_steps(1)?;
    let before = batch.readback_output()?;
    let mut encoder = context.device().create_command_encoder(&Default::default());
    let selections = [0, 1].map(|environment| GpuArticulatedResetSelection {
        environment,
        template: 0,
        root_translation: None,
        root_velocity: None,
    });
    assert!(
        batch
            .encode_reset_envs_from_templates(&mut encoder, &templates, &selections)
            .is_err()
    );
    let _submission = context.queue().submit(Some(encoder.finish()));
    let after = batch.readback_output()?;
    for (before, after) in before.iter().zip(after) {
        assert_eq!(before.state, after.state);
        assert_eq!(before.root_pose, after.root_pose);
    }
    Ok(())
}

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
        model: crate::articulated_world::MotorModel::ForceBased,
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
    let mut batch = GpuArticulatedDynamicsBatch::new_with_spherical_state(
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
        vec![None],
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
    let drive_parameters = crate::gpu_articulated_mass::read_buffer(
        context.device(),
        context.queue(),
        batch
            .spherical_drives
            .as_ref()
            .ok_or("missing spherical drives")?
            .parameter_buffer(),
    )?;
    let enabled = usize::try_from(templates.spherical_ranges[1].start)? + 12;
    assert_eq!(&drive_parameters[enabled..enabled + 4], &1u32.to_ne_bytes());
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
