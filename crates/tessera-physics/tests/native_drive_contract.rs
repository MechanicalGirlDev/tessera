//! Coupled native inertia and physical motor caps on CPU and real GPU backends.
#![cfg(feature = "gpu-contact")]

use nalgebra::{DVector, Isometry3, Matrix3, UnitQuaternion, Vector3};
use tessera_physics::articulated_world::{
    ArticulatedWorld, ArticulatedWorldParams, JointMotor, MotorModel,
};
use tessera_physics::articulation::{Articulation, JointKind, JointSpec, LinkSpec};
use tessera_physics::gpu_articulated_force::GpuArticulatedForceBatch;
use tessera_physics::gpu_articulated_joint_force::{
    GpuArticulatedJointForceBatch, GpuJointForceInput,
};
use tessera_physics::gpu_articulated_mass_assembly::{
    GpuArticulatedMassAssemblyBatch, GpuMassAssemblySystem,
};
use tessera_physics::gpu_articulated_spherical::{
    GpuArticulatedSphericalBatch, GpuSphericalJointState,
};
use tessera_physics::gpu_articulated_spherical_drive::GpuArticulatedSphericalDriveBatch;
use tessera_physics::gpu_articulated_state::{GpuGeneralizedState, GpuGeneralizedStateBatch};
use tessera_physics::gpu_contact_pipeline::GpuContactDevice;
use tessera_physics::spherical_drive::SphericalJointDrive;

type TestResult = Result<(), Box<dyn core::error::Error>>;

#[test]
fn acceleration_motor_uses_coupled_inverse_inertia_and_physical_caps() -> TestResult {
    for model in [MotorModel::ForceBased, MotorModel::AccelerationBased] {
        for mass in [1.0, 7.0] {
            for cap in [0.0, 0.5, f64::INFINITY] {
                let articulation = Articulation::new(
                    vec![
                        LinkSpec {
                            mass: 0.0,
                            center_of_mass: Vector3::zeros(),
                            inertia: Matrix3::zeros(),
                        },
                        LinkSpec {
                            mass,
                            center_of_mass: Vector3::zeros(),
                            inertia: Matrix3::identity() * mass,
                        },
                        LinkSpec {
                            mass: mass * 3.0,
                            center_of_mass: Vector3::zeros(),
                            inertia: Matrix3::identity() * mass,
                        },
                    ],
                    (0..2)
                        .map(|parent| JointSpec {
                            parent,
                            child: parent + 1,
                            kind: JointKind::Prismatic,
                            origin: Isometry3::identity(),
                            axis: Vector3::x(),
                            limits: None,
                        })
                        .collect(),
                    0,
                )?;
                let mut world = ArticulatedWorld::new(
                    articulation,
                    Isometry3::identity(),
                    Vec::new(),
                    ArticulatedWorldParams {
                        gravity: [0.0; 3],
                        ..Default::default()
                    },
                )?;
                let motor = JointMotor {
                    model,
                    position_target: Some(1.0),
                    velocity_target: 0.0,
                    stiffness: 3.0,
                    damping: 0.0,
                    max_force: cap,
                };
                world.set_joint_motor(0, Some(motor))?;
                let force = if model == MotorModel::AccelerationBased {
                    3.0 * mass
                } else {
                    3.0
                };
                let acceleration = force.min(cap) / mass;
                world.step(0.001, &[0.0, 0.0])?;
                assert!((world.velocities[0] - 0.001 * acceleration).abs() < 1e-12);
                assert!((world.velocities[1] + 0.001 * acceleration).abs() < 1e-12);
                assert_eq!(motor.force(0.0, 0.0, 1.0 / mass), Some(force.min(cap)));
            }
        }
    }
    let motor = JointMotor {
        model: MotorModel::AccelerationBased,
        position_target: None,
        velocity_target: 1.0,
        stiffness: 0.0,
        damping: 2.0,
        max_force: f64::INFINITY,
    };
    assert_eq!(motor.force(0.0, 0.0, 0.0), Some(0.0));
    for invalid in [f64::NAN, f64::NEG_INFINITY, -1.0] {
        assert!(
            JointMotor {
                max_force: invalid,
                ..motor
            }
            .force(0.0, 0.0, 1.0)
            .is_none()
        );
        assert!(motor.force(0.0, 0.0, invalid).is_none());
    }
    Ok(())
}

#[test]
fn acceleration_tangent_drives_use_projected_inverse_inertia_and_physical_caps() -> TestResult {
    use nalgebra::Matrix4;
    let orientation = UnitQuaternion::from_euler_angles(0.4, 1.57, -0.2);
    let error = Vector3::new(0.1, -0.2, 0.3);
    let mut drive = SphericalJointDrive {
        model: MotorModel::AccelerationBased,
        orientation_target: UnitQuaternion::from_scaled_axis(error) * orientation,
        velocity_target: Vector3::zeros(),
        stiffness: Vector3::repeat(10.0),
        damping: Vector3::repeat(2.0),
        max_torque: Vector3::repeat(f64::INFINITY),
    };
    let native =
        Matrix4::from_diagonal(&nalgebra::Vector4::new(1.0, 2.0, 3.0, 4.0)) + Matrix4::repeat(1.0);
    let inverse = native.try_inverse().ok_or("singular fixture inertia")?;
    let projected = Vector3::new(inverse[(1, 1)], inverse[(2, 2)], inverse[(3, 3)]);
    let torque = drive.torque_with_inverse_mass(orientation, Vector3::zeros(), projected)?;
    assert!((torque.component_mul(&projected) - error * 10.0).norm() < 1e-12);
    // Scaling the entire coupled articulation scales physical torque, not gains.
    let scaled = drive.torque_with_inverse_mass(orientation, Vector3::zeros(), projected / 7.0)?;
    assert!((scaled - torque * 7.0).norm() < 1e-12);
    assert!(drive.torque(orientation, Vector3::zeros()).is_err());
    drive.max_torque = Vector3::new(0.0, 0.5, f64::INFINITY);
    let capped = drive.torque_with_inverse_mass(orientation, Vector3::zeros(), projected / 7.0)?;
    assert_eq!(capped.x, 0.0);
    assert_eq!(capped.y, -0.5);
    assert!((capped.z - scaled.z).abs() < 1e-12);
    let negated = UnitQuaternion::new_unchecked(-orientation.into_inner());
    assert!(
        (drive.torque_with_inverse_mass(negated, Vector3::zeros(), projected / 7.0)? - capped)
            .norm()
            < 1e-12
    );
    Ok(())
}

#[test]
fn acceleration_gpu_motor_projects_coupled_inverse_and_preserves_physical_caps() -> TestResult {
    use nalgebra::{DMatrix, Matrix3, Vector3};
    use tessera_physics::gpu_articulated_mass_assembly::GpuMassLink;
    let mut systems = Vec::new();
    let mut inputs = Vec::new();
    let mut expected = Vec::new();
    for model in [MotorModel::ForceBased, MotorModel::AccelerationBased] {
        for scale in [1.0, 7.0] {
            for cap in [0.0, 2.0, f64::INFINITY] {
                let matrix = DMatrix::from_row_slice(2, 2, &[2.0, 1.0, 1.0, 3.0]) * scale;
                let inverse = matrix.try_inverse().ok_or("singular fixture inertia")?;
                let motor = JointMotor {
                    model,
                    position_target: Some(1.0),
                    velocity_target: 0.0,
                    stiffness: 4.0,
                    damping: 0.0,
                    max_force: cap,
                };
                let force = motor
                    .force(0.0, 0.0, inverse[(0, 0)])
                    .ok_or("invalid fixture motor")?;
                expected.push(&inverse * DVector::from_column_slice(&[force, 0.0]));
                systems.push(GpuMassAssemblySystem {
                    links: vec![GpuMassLink {
                        mass: scale,
                        inertia_world: Matrix3::zeros(),
                        linear_jacobian: DMatrix::from_row_slice(
                            3,
                            2,
                            &[1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        ),
                        angular_jacobian: DMatrix::zeros(3, 2),
                    }],
                    armature: DVector::from_column_slice(&[scale, 2.0 * scale]),
                    force: DVector::zeros(2),
                });
                inputs.push(vec![
                    GpuJointForceInput {
                        motor: Some(motor),
                        ..Default::default()
                    },
                    GpuJointForceInput::default(),
                ]);
            }
        }
    }
    let states = vec![
        GpuGeneralizedState {
            positions: DVector::zeros(2),
            velocities: DVector::zeros(2)
        };
        systems.len()
    ];
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        tested += 1;
        let mass =
            GpuArticulatedMassAssemblyBatch::new(context.device(), context.queue(), &systems)?;
        let state = GpuGeneralizedStateBatch::from_assembly_batch(&mass, &states)?;
        let forces = GpuArticulatedForceBatch::new(
            &mass,
            &vec![DVector::zeros(2); systems.len()],
            &vec![Vector3::zeros(); systems.len()],
        )?;
        assert!(GpuArticulatedJointForceBatch::new(&state, &forces, &inputs).is_err());
        let joint = GpuArticulatedJointForceBatch::new_implicit_with_mass(
            &state,
            &forces,
            &inputs,
            &systems
                .iter()
                .map(|s| s.armature.clone())
                .collect::<Vec<_>>(),
            0.01,
            &mass,
        )?;
        let mut encoder = context.device().create_command_encoder(&Default::default());
        joint.encode_prepare_mass(&mut encoder);
        mass.encode_with_inverse(&mut encoder);
        joint.encode(&mut encoder);
        forces.encode(&mut encoder);
        mass.encode(&mut encoder);
        let _ = context.queue().submit(Some(encoder.finish()));
        let actual = mass.readback()?;
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (actual - expected).norm() < 3e-5,
                "{backend:?} environment {index}: {actual:?} != {expected:?}"
            );
        }
        // Poison the previous effective armature with implicit passive terms.
        // A native prepass must restore it before normalizing next-step drives.
        let mut passive = inputs.clone();
        for row in &mut passive {
            row[0].passive.damping = 500.0;
        }
        joint.update_inputs(&passive)?;
        let mut encoder = context.device().create_command_encoder(&Default::default());
        joint.encode(&mut encoder);
        joint.encode_prepare_mass(&mut encoder);
        mass.encode_with_inverse(&mut encoder);
        let _ = context.queue().submit(Some(encoder.finish()));
        joint.update_inputs(&inputs)?;
        let mut encoder = context.device().create_command_encoder(&Default::default());
        joint.encode(&mut encoder);
        forces.encode(&mut encoder);
        mass.encode(&mut encoder);
        let _ = context.queue().submit(Some(encoder.finish()));
        let actual = mass.readback()?;
        for (actual, expected) in actual.iter().zip(&expected) {
            assert!((actual - expected).norm() < 3e-5);
        }
    }
    assert!(tested > 0, "no GPU backend available");
    Ok(())
}
#[test]
fn acceleration_gpu_spherical_drives_project_native_coupling_and_caps() -> TestResult {
    use nalgebra::{DMatrix, Matrix3};
    use tessera_physics::gpu_articulated_mass_assembly::GpuMassLink;
    let orientation = UnitQuaternion::from_euler_angles(0.4, 1.57, -0.2);
    let error = Vector3::new(0.1, -0.2, 0.3);
    let mut systems = Vec::new();
    let mut drives = Vec::new();
    let mut expected = Vec::new();
    for model in [MotorModel::ForceBased, MotorModel::AccelerationBased] {
        for scale in [1.0, 7.0] {
            for cap in [Vector3::repeat(f64::INFINITY), Vector3::new(0.0, 0.5, 2.0)] {
                let matrix =
                    (DMatrix::from_diagonal(&DVector::from_column_slice(&[1.0, 2.0, 3.0, 4.0]))
                        + DMatrix::from_element(4, 4, 1.0))
                        * scale;
                let inverse = matrix.try_inverse().ok_or("singular fixture inertia")?;
                let drive = SphericalJointDrive {
                    model,
                    orientation_target: UnitQuaternion::from_scaled_axis(error) * orientation,
                    velocity_target: Vector3::zeros(),
                    stiffness: Vector3::repeat(10.0),
                    damping: Vector3::repeat(2.0),
                    max_torque: cap,
                };
                let projected = Vector3::new(inverse[(1, 1)], inverse[(2, 2)], inverse[(3, 3)]);
                let torque =
                    drive.torque_with_inverse_mass(orientation, Vector3::zeros(), projected)?;
                expected.push(
                    &inverse * DVector::from_column_slice(&[0.0, torque.x, torque.y, torque.z]),
                );
                drives.push(vec![Some(drive)]);
                systems.push(GpuMassAssemblySystem {
                    links: vec![GpuMassLink {
                        mass: scale,
                        inertia_world: Matrix3::zeros(),
                        linear_jacobian: DMatrix::from_row_slice(
                            3,
                            4,
                            &[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        ),
                        angular_jacobian: DMatrix::zeros(3, 4),
                    }],
                    armature: DVector::from_column_slice(&[
                        scale,
                        2.0 * scale,
                        3.0 * scale,
                        4.0 * scale,
                    ]),
                    force: DVector::zeros(4),
                });
            }
        }
    }
    let states = vec![
        GpuGeneralizedState {
            positions: DVector::zeros(4),
            velocities: DVector::zeros(4)
        };
        systems.len()
    ];
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        tested += 1;
        let mass =
            GpuArticulatedMassAssemblyBatch::new(context.device(), context.queue(), &systems)?;
        let state = GpuGeneralizedStateBatch::from_assembly_batch(&mass, &states)?;
        let spherical = GpuArticulatedSphericalBatch::new(
            &state,
            &(0..systems.len())
                .map(|i| {
                    vec![GpuSphericalJointState {
                        velocity_slot: 1,
                        orientation: if i % 2 == 0 {
                            orientation
                        } else {
                            UnitQuaternion::new_unchecked(-orientation.into_inner())
                        },
                    }]
                })
                .collect::<Vec<_>>(),
            0.01,
        )?;
        let forces = GpuArticulatedForceBatch::new(
            &mass,
            &vec![DVector::zeros(4); systems.len()],
            &vec![Vector3::zeros(); systems.len()],
        )?;
        assert!(
            GpuArticulatedSphericalDriveBatch::new(&state, &spherical, &forces, &drives).is_err()
        );
        let batch = GpuArticulatedSphericalDriveBatch::new_with_mass(
            &state, &spherical, &forces, &drives, &mass,
        )?;
        let mut encoder = context.device().create_command_encoder(&Default::default());
        mass.encode_with_inverse(&mut encoder);
        batch.encode(&mut encoder);
        forces.encode(&mut encoder);
        mass.encode(&mut encoder);
        let _ = context.queue().submit(Some(encoder.finish()));
        let actual = mass.readback()?;
        for (i, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (actual - expected).norm() < 5e-5,
                "{backend:?} environment {i}: {actual:?} != {expected:?}"
            );
        }
        assert!((&actual[4] - &actual[6]).norm() < 5e-5);
        assert!((&actual[0] - &actual[2] * 7.0).norm() < 5e-5);
    }
    assert!(tested > 0, "no GPU backend available");
    Ok(())
}
