use super::*;
use crate::articulated_world::JointMotor;
use crate::articulation::{JointKind, JointSpec, LinkSpec};
use nalgebra::Matrix3;
use wgpu::util::DeviceExt;

#[test]
fn resident_gpu_motor_targets_scatter_delay_boundaries_reset_and_validation() {
    let properties = LinkSpec {
        mass: 1.0,
        center_of_mass: Vector3::zeros(),
        inertia: Matrix3::identity(),
    };
    let articulation = Articulation::new(
        vec![properties.clone(), properties.clone(), properties],
        vec![
            JointSpec {
                parent: 0,
                child: 1,
                origin: Isometry3::identity(),
                kind: JointKind::Prismatic,
                axis: Vector3::x(),
                limits: None,
            },
            JointSpec {
                parent: 1,
                child: 2,
                origin: Isometry3::identity(),
                kind: JointKind::Prismatic,
                axis: Vector3::y(),
                limits: None,
            },
        ],
        0,
    )
    .unwrap();
    let initial = GpuGeneralizedState {
        positions: DVector::zeros(2),
        velocities: DVector::zeros(2),
    };
    let mut input = GpuArticulatedDynamicsInput::new(
        &articulation,
        Isometry3::identity(),
        initial.clone(),
        Vector3::zeros(),
    );
    input.joints = vec![
        GpuJointForceInput {
            motor: Some(JointMotor {
                position_target: Some(0.0),
                velocity_target: 0.0,
                stiffness: 1.0,
                damping: 0.0,
                max_force: 100.0,
            }),
            ..Default::default()
        },
        GpuJointForceInput {
            motor: Some(JointMotor {
                position_target: None,
                velocity_target: 0.0,
                stiffness: 0.0,
                damping: 1.0,
                max_force: 100.0,
            }),
            ..Default::default()
        },
    ];
    let mapping = vec![
        GpuMotorTargetMapping {
            coordinate: 1,
            link: 2,
            mode: GpuMotorTargetMode::Velocity,
        },
        GpuMotorTargetMapping {
            coordinate: 0,
            link: 1,
            mode: GpuMotorTargetMode::Position,
        },
    ];
    let mappings = vec![mapping.clone(), mapping.clone()];
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        tested += 1;
        let mut batch =
            GpuArticulatedDynamicsBatch::new(&context, &[input.clone(), input.clone()], 0.01)
                .unwrap();
        for bad in [
            vec![],
            vec![mapping.clone()],
            vec![vec![], vec![]],
            vec![mapping.clone(), vec![mapping[0]]],
            vec![vec![mapping[0], mapping[0]]; 2],
            vec![
                vec![GpuMotorTargetMapping {
                    coordinate: 9,
                    ..mapping[0]
                }];
                2
            ],
            vec![
                vec![GpuMotorTargetMapping {
                    link: 1,
                    ..mapping[0]
                }];
                2
            ],
            vec![
                vec![GpuMotorTargetMapping {
                    mode: GpuMotorTargetMode::Position,
                    ..mapping[0]
                }];
                2
            ],
        ] {
            assert!(batch.enable_motor_target_control(&bad, &[0, 2]).is_err());
        }
        assert!(batch.enable_motor_target_control(&mappings, &[0]).is_err());
        batch
            .enable_motor_target_control(&mappings, &[0, 2])
            .unwrap();
        let actions = context.device().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // A GPU-side producer copies distinct action rows into the storage buffer
        // immediately before latch, with no submit/readback between producer and use.
        let producer = context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&[2.0_f32, -3.0, 4.0, -5.0]),
                usage: wgpu::BufferUsages::COPY_SRC,
            });
        let mut encoder = context.device().create_command_encoder(&Default::default());
        assert!(batch.encode_motor_targets(&mut encoder, &producer).is_err());
        let short = context.device().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        assert!(batch.encode_motor_targets(&mut encoder, &short).is_err());
        encoder.copy_buffer_to_buffer(&producer, 0, &actions, 0, 16);
        batch.encode_motor_targets(&mut encoder, &actions).unwrap();
        batch.encode_step(&mut encoder).unwrap();
        batch.encode_contact_step(&mut encoder).unwrap();
        batch.encode_integration(&mut encoder);
        let _ = context.queue().submit(Some(encoder.finish()));
        let after_two = batch.readback().unwrap();
        assert!(after_two[0].positions[0] > 0.0 && after_two[0].positions[1] > 0.0);
        assert_eq!(after_two[1].positions, initial.positions);
        assert_eq!(after_two[1].velocities, initial.velocities);
        batch.submit_steps(1).unwrap();
        let after_three = batch.readback().unwrap();
        assert!((after_three[1].velocities[0] + 0.025).abs() < 1e-6);
        assert!((after_three[1].velocities[1] + 0.03).abs() < 1e-6);
        assert!((after_three[0].velocities[0] - 0.059996).abs() < 1e-6);
        assert!((after_three[0].velocities[1] - 0.059402).abs() < 1e-6);

        // Reset cancels an action still waiting, but preserves the applied targets.
        let replacement = context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&[7.0_f32, 8.0, 9.0, 10.0]),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let mut encoder = context.device().create_command_encoder(&Default::default());
        batch
            .encode_motor_targets(&mut encoder, &replacement)
            .unwrap();
        let _ = context.queue().submit(Some(encoder.finish()));
        batch.reset(&[initial.clone(), initial.clone()]).unwrap();
        batch.submit_steps(1).unwrap();
        let reset = batch.readback().unwrap();
        assert!((reset[0].velocities[0] - 0.02).abs() < 1e-6);
        assert!((reset[0].velocities[1] - 0.02).abs() < 1e-6);
        assert!((reset[1].velocities[0] + 0.025).abs() < 1e-6);
        assert!((reset[1].velocities[1] + 0.03).abs() < 1e-6);

        // Invalid reset preserves pending state; repeated latches restart the delay.
        batch.reset(&[initial.clone(), initial.clone()]).unwrap();
        context.queue().write_buffer(
            &actions,
            0,
            bytemuck::cast_slice(&[2.0_f32, -4.0, 4.0, -7.0]),
        );
        let mut encoder = context.device().create_command_encoder(&Default::default());
        batch
            .encode_motor_targets(&mut encoder, &replacement)
            .unwrap();
        batch.encode_step(&mut encoder).unwrap();
        batch.encode_motor_targets(&mut encoder, &actions).unwrap();
        batch.encode_step(&mut encoder).unwrap();
        let _ = context.queue().submit(Some(encoder.finish()));
        assert!(batch.reset(&[]).is_err());
        batch.submit_steps(1).unwrap();
        let before_apply = batch.readback().unwrap();
        assert!((before_apply[1].velocities[0] + 0.074995).abs() < 1e-6);
        assert!((before_apply[1].velocities[1] + 0.089103).abs() < 1e-6);
        batch.submit_steps(1).unwrap();
        let applied = batch.readback().unwrap();
        // Env 1 retains (-5,-3) for the restarted delay, then receives (-7,-4).
        // The superseded (+10,+8) never applies, and invalid reset did not cancel.
        assert!(before_apply[1].velocities[0] < 0.0);
        assert!((applied[1].velocities[0] + 0.1099875).abs() < 1e-6);
        assert!((applied[1].velocities[1] + 0.12821197).abs() < 1e-6);
        assert!(applied[1].velocities[0] < before_apply[1].velocities[0]);
        assert!(applied[1].velocities[1] < before_apply[1].velocities[1]);

        // Host updates cancel pending actions and cannot invalidate configured modes.
        let mut invalid = input.joints.clone();
        invalid[0].motor = None;
        assert!(batch.update_joints(&[invalid.clone(), invalid]).is_err());
        batch.reset(&[initial.clone(), initial.clone()]).unwrap();
        let mut encoder = context.device().create_command_encoder(&Default::default());
        batch
            .encode_motor_targets(&mut encoder, &replacement)
            .unwrap();
        let _ = context.queue().submit(Some(encoder.finish()));
        batch
            .update_joints(&[input.joints.clone(), input.joints.clone()])
            .unwrap();
        batch.submit_steps(4).unwrap();
        for state in batch.readback().unwrap() {
            assert_eq!(state.positions, initial.positions);
            assert_eq!(state.velocities, initial.velocities);
        }
    }
    assert!(tested > 0, "no GPU backend available");
}
