//! Incompatible reset layouts and foreign owner rejection.

use super::*;

#[test]
fn selective_reset_rejects_incompatible_layouts_and_foreign_owner() {
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
            kind: JointKind::Revolute,
            axis: Vector3::z(),
            origin: Isometry3::identity(),
            limits: None,
        }],
        0,
    )
    .unwrap();
    let initial = [
        GpuGeneralizedState {
            positions: DVector::from_element(1, 0.25),
            velocities: DVector::zeros(1),
        },
        GpuGeneralizedState {
            positions: DVector::from_element(7, -0.5),
            velocities: DVector::zeros(7),
        },
    ];
    let systems = initial
        .iter()
        .map(|s| GpuArticulatedMassSystem {
            mass: DMatrix::identity(s.positions.len(), s.positions.len()),
            force: DVector::zeros(s.positions.len()),
        })
        .collect::<Vec<_>>();
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        tested += 1;
        let mass =
            GpuArticulatedMassBatch::new(context.device(), context.queue(), &systems).unwrap();
        let state = GpuGeneralizedStateBatch::from_mass_batch(&mass, &initial).unwrap();
        let foreign = GpuGeneralizedStateBatch::from_mass_batch(&mass, &initial).unwrap();
        let poses = GpuArticulatedPoseBatch::new_with_floating_roots(
            &state,
            &[&art, &art],
            &[Isometry3::identity(); 2],
            &[false, true],
        )
        .unwrap();
        let mut encoder = context.device().create_command_encoder(&Default::default());
        assert!(
            foreign
                .snapshot_reset_templates(&mut encoder, &poses, None)
                .is_err()
        );
        let templates = state
            .snapshot_reset_templates(&mut encoder, &poses, None)
            .unwrap();
        let selection = GpuArticulatedResetSelection {
            environment: 0,
            template: 0,
            root_translation: None,
            root_velocity: None,
        };
        assert!(
            foreign
                .encode_reset_envs_from_templates(&mut encoder, &templates, &[selection])
                .is_err()
        );
        for invalid in [
            GpuArticulatedResetSelection {
                template: 1,
                ..selection
            },
            GpuArticulatedResetSelection {
                root_translation: Some([0.0; 3]),
                ..selection
            },
            GpuArticulatedResetSelection {
                root_velocity: Some([0.0; 6]),
                ..selection
            },
        ] {
            assert!(
                state
                    .encode_reset_envs_from_templates(&mut encoder, &templates, &[invalid])
                    .is_err()
            );
        }
        state
            .encode_reset_envs_from_templates(&mut encoder, &templates, &[selection])
            .unwrap();
        state
            .encode_reset_envs_from_templates(&mut encoder, &templates, &[])
            .unwrap();
        let _submission = context.queue().submit(Some(encoder.finish()));
        assert_eq!(state.readback().unwrap(), initial);
    }
    assert!(tested > 0, "no native GPU backend tested");
}
