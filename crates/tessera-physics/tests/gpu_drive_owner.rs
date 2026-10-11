//! Drive mass projection and resident spherical actions through the dynamics owner.
#![cfg(feature = "gpu-contact")]

use nalgebra::{DVector, Isometry3, Matrix3, UnitQuaternion, Vector3};
use tessera_physics::articulated_world::{JointMotor, MotorModel};
use tessera_physics::articulation::{Articulation, JointKind, JointSpec, LinkSpec};
use tessera_physics::gpu_articulated_dynamics::{
    GpuArticulatedDynamicsBatch, GpuArticulatedDynamicsInput,
};
use tessera_physics::gpu_articulated_spherical::GpuSphericalJointState;
use tessera_physics::gpu_articulated_state::{
    GpuGeneralizedState, reset::GpuArticulatedResetSelection,
};
use tessera_physics::gpu_contact_pipeline::GpuContactDevice;
use tessera_physics::gpu_motor_target::GpuMotorTargetMode;
use tessera_physics::gpu_spherical_motor_target::GpuSphericalMotorTargetMapping;
use tessera_physics::spherical_drive::SphericalJointDrive;
use wgpu::util::DeviceExt;

fn model(
    mass: f64,
    kind: JointKind,
) -> Result<Articulation, tessera_physics::articulation::ArticulationError> {
    Articulation::new(
        vec![
            LinkSpec {
                mass: 0.0,
                center_of_mass: Vector3::zeros(),
                inertia: Matrix3::zeros(),
            },
            LinkSpec {
                mass,
                center_of_mass: Vector3::zeros(),
                inertia: Matrix3::from_diagonal(&Vector3::new(mass, mass * 2.0, mass * 3.0)),
            },
        ],
        vec![JointSpec {
            parent: 0,
            child: 1,
            kind,
            origin: Isometry3::identity(),
            axis: Vector3::x(),
            limits: None,
        }],
        0,
    )
}

#[test]
fn dynamics_owner_projects_acceleration_motor_mass_before_assembly()
-> Result<(), Box<dyn core::error::Error>> {
    let context = GpuContactDevice::new()?;
    let models = [
        model(2.0, JointKind::Prismatic)?,
        model(7.0, JointKind::Prismatic)?,
    ];
    let inputs = models
        .iter()
        .map(|model| {
            let mut input = GpuArticulatedDynamicsInput::new(
                model,
                Isometry3::identity(),
                GpuGeneralizedState {
                    positions: DVector::zeros(1),
                    velocities: DVector::zeros(1),
                },
                Vector3::zeros(),
            );
            input.joints[0].motor = Some(JointMotor {
                model: MotorModel::AccelerationBased,
                position_target: Some(1.0),
                velocity_target: 0.0,
                stiffness: 3.0,
                damping: 0.0,
                max_force: f64::INFINITY,
            });
            input
        })
        .collect::<Vec<_>>();
    let batch = GpuArticulatedDynamicsBatch::new(&context, &inputs, 0.001)?;
    batch.submit_steps(1)?;
    for output in batch.readback_output()? {
        assert!((output.state.velocities[0] - 0.003).abs() < 1e-7);
    }
    Ok(())
}

#[test]
fn dynamics_owner_spherical_scatter_delays_and_selected_reset_are_resident()
-> Result<(), Box<dyn core::error::Error>> {
    let context = GpuContactDevice::new()?;
    let models = [
        model(2.0, JointKind::Spherical)?,
        model(7.0, JointKind::Spherical)?,
    ];
    let inputs = models
        .iter()
        .map(|model| {
            GpuArticulatedDynamicsInput::new(
                model,
                Isometry3::identity(),
                GpuGeneralizedState {
                    positions: DVector::zeros(3),
                    velocities: DVector::zeros(3),
                },
                Vector3::zeros(),
            )
        })
        .collect::<Vec<_>>();
    let states = vec![
        vec![GpuSphericalJointState {
            velocity_slot: 0,
            orientation: UnitQuaternion::identity()
        }];
        2
    ];
    let drive = SphericalJointDrive {
        model: MotorModel::AccelerationBased,
        orientation_target: UnitQuaternion::identity(),
        velocity_target: Vector3::zeros(),
        stiffness: Vector3::repeat(10.0),
        damping: Vector3::zeros(),
        max_torque: Vector3::repeat(f64::INFINITY),
    };
    let mut batch = GpuArticulatedDynamicsBatch::new_with_spherical_state(
        &context,
        &inputs,
        0.001,
        &[false, false],
        &states,
        &vec![vec![Some(drive)]; 2],
    )?;
    batch.enable_spherical_motor_target_control(
        &vec![
            vec![GpuSphericalMotorTargetMapping {
                joint: 0,
                axis: 2,
                mode: GpuMotorTargetMode::Position
            }];
            2
        ],
        &[1, 0],
    )?;
    let mut encoder = context.device().create_command_encoder(&Default::default());
    let templates = batch.snapshot_reset_templates(&mut encoder)?;
    let actions = context
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Owner spherical resident actions"),
            contents: bytemuck::cast_slice(&[0.4f32, -0.2]),
            usage: wgpu::BufferUsages::STORAGE,
        });
    batch.encode_spherical_motor_targets(&mut encoder, &actions)?;
    batch.encode_step(&mut encoder)?;
    let _submission = context.queue().submit([encoder.finish()]);
    let first = batch.readback_output()?;
    assert!(first[0].state.velocities[2].abs() < 1e-7);
    assert!((first[1].state.velocities[2] + 0.002).abs() < 1e-7);
    batch.submit_steps(1)?;
    let second = batch.readback_output()?;
    assert!((second[0].state.velocities[2] - 0.004).abs() < 1e-7);
    let untouched = second[1].state.velocities.clone();
    let mut encoder = context.device().create_command_encoder(&Default::default());
    batch.encode_reset_envs_from_templates(
        &mut encoder,
        &templates,
        &[GpuArticulatedResetSelection {
            environment: 0,
            template: 0,
            root_translation: None,
            root_velocity: None,
        }],
    )?;
    let _submission = context.queue().submit([encoder.finish()]);
    let reset = batch.readback_output()?;
    assert!(reset[0].state.velocities.norm() < 1e-7);
    assert_eq!(reset[1].state.velocities, untouched);
    batch.submit_steps(1)?;
    assert!(batch.readback_output()?[0].state.velocities.norm() < 1e-7);
    Ok(())
}
