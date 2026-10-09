use super::*;
use crate::articulation::{Articulation, JointKind, JointSpec, LinkSpec};
use crate::gpu_articulated_mass::{GpuArticulatedMassBatch, GpuArticulatedMassSystem, read_buffer};
use crate::gpu_articulated_root::GpuArticulatedRootBatch;
use crate::gpu_articulated_spherical::GpuSphericalJointState;
use crate::gpu_articulated_state::GpuGeneralizedState;
use crate::gpu_contact_pipeline::GpuContactDevice;
use nalgebra::{DMatrix, DVector, Isometry3, Matrix3, UnitQuaternion, Vector3};

#[path = "gpu_reset_fault_tests.rs"]
mod faults;
#[path = "gpu_reset_fixture.rs"]
mod fixture;
#[path = "gpu_reset_layout_tests.rs"]
mod layouts;

#[test]
fn selective_resident_reset_preserves_other_environment_and_native_layouts() {
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Some(fixture) = fixture::build(backend) else {
            continue;
        };
        let fixture::Fixture {
            context,
            mass,
            state,
            spherical,
            poses,
            roots,
            initial,
            joints,
            root_poses,
        } = fixture;
        tested += 1;
        let mut encoder = context.device().create_command_encoder(&Default::default());
        let templates = state
            .snapshot_reset_templates(&mut encoder, &poses, Some(&spherical))
            .unwrap();
        mass.encode(&mut encoder);
        state.encode_step(&mut encoder, 0.125).unwrap();
        roots.encode(&mut encoder);
        spherical.encode(&mut encoder);
        poses.encode(&mut encoder);
        let _submission = context.queue().submit(Some(encoder.finish()));
        let before = state.readback().unwrap();
        let roots_before = poses.readback_roots().unwrap();
        let spherical_before = spherical.readback().unwrap();
        assert_ne!(roots_before[0], root_poses[0]);
        assert_ne!(spherical_before[0][0], joints[0][0].orientation);
        let selection = GpuArticulatedResetSelection {
            environment: 1,
            template: 0,
            root_translation: Some([10.0, -4.0, 2.0]),
            root_velocity: Some([1.0, 2.0, 3.0, -1.0, -2.0, -3.0]),
        };
        let raw_before = [
            read_buffer(context.device(), context.queue(), state.position_buffer()).unwrap(),
            read_buffer(context.device(), context.queue(), state.velocity_buffer()).unwrap(),
            read_buffer(context.device(), context.queue(), poses.root_pose_buffer()).unwrap(),
            read_buffer(
                context.device(),
                context.queue(),
                spherical.orientation_buffer(),
            )
            .unwrap(),
            read_buffer(context.device(), context.queue(), state.status_buffer()).unwrap(),
            read_buffer(
                context.device(),
                context.queue(),
                state.mass_status_buffer(),
            )
            .unwrap(),
        ];
        // A valid first row followed by any invalid row must record no writes.
        for invalid in [
            GpuArticulatedResetSelection {
                environment: 9,
                ..selection
            },
            GpuArticulatedResetSelection {
                template: 9,
                ..selection
            },
            GpuArticulatedResetSelection {
                root_translation: Some([f64::NAN, 0.0, 0.0]),
                ..selection
            },
            GpuArticulatedResetSelection {
                root_velocity: Some([f64::INFINITY; 6]),
                ..selection
            },
            selection,
        ] {
            let mut encoder = context.device().create_command_encoder(&Default::default());
            assert!(
                state
                    .encode_reset_envs_from_templates(
                        &mut encoder,
                        &templates,
                        &[selection, invalid]
                    )
                    .is_err()
            );
            let _submission = context.queue().submit(Some(encoder.finish()));
        }
        for (buffer, expected) in [
            state.position_buffer(),
            state.velocity_buffer(),
            poses.root_pose_buffer(),
            spherical.orientation_buffer(),
            state.status_buffer(),
            state.mass_status_buffer(),
        ]
        .into_iter()
        .zip(&raw_before)
        {
            assert_eq!(
                &read_buffer(context.device(), context.queue(), buffer).unwrap(),
                expected
            );
        }
        // Recover a persistent integration fault using a healthy captured template.
        context
            .queue()
            .write_buffer(state.status_buffer(), 4, bytemuck::bytes_of(&1u32));
        let mut encoder = context.device().create_command_encoder(&Default::default());
        state
            .encode_reset_envs_from_templates(&mut encoder, &templates, &[selection])
            .unwrap();
        poses.encode(&mut encoder);
        let _submission = context.queue().submit(Some(encoder.finish()));
        let after = state.readback().unwrap();
        let roots_after = poses.readback_roots().unwrap();
        let spherical_after = spherical.readback().unwrap();
        assert_eq!(after[0], before[0]);
        assert_eq!(roots_after[0], roots_before[0]);
        assert_eq!(spherical_after[0], spherical_before[0]);
        assert_eq!(after[1].positions, initial[0].positions);
        assert_eq!(
            after[1].velocities.as_slice(),
            &[1.0, 2.0, 3.0, -1.0, -2.0, -3.0, 0.5, 0.5, 0.5]
        );
        assert_eq!(
            roots_after[1].translation.vector,
            Vector3::new(11.0, -2.0, 5.0)
        );
        assert_eq!(roots_after[1].rotation, root_poses[0].rotation);
        assert!((spherical_after[1][0].inverse() * joints[0][0].orientation).angle() < 1e-6);
        // The next integration advances both environments from their respective states.
        let mut encoder = context.device().create_command_encoder(&Default::default());
        mass.encode(&mut encoder);
        state.encode_step(&mut encoder, 0.125).unwrap();
        roots.encode(&mut encoder);
        spherical.encode(&mut encoder);
        let _submission = context.queue().submit(Some(encoder.finish()));
        let next_roots = poses.readback_roots().unwrap();
        assert!(
            (next_roots[0].translation.vector
                - roots_after[0].translation.vector
                - Vector3::repeat(0.0625))
            .norm()
                < 1e-6
        );
        assert!(
            (next_roots[1].translation.vector
                - roots_after[1].translation.vector
                - Vector3::new(0.125, 0.25, 0.375))
            .norm()
                < 1e-6
        );
        // Reuse proves stepping/reset never mutates the immutable templates.
        let mut encoder = context.device().create_command_encoder(&Default::default());
        state
            .encode_reset_envs_from_templates(&mut encoder, &templates, &[selection])
            .unwrap();
        let _submission = context.queue().submit(Some(encoder.finish()));
        assert_eq!(state.readback().unwrap(), after);
        assert_eq!(poses.readback_roots().unwrap()[1], roots_after[1]);
    }
    assert!(tested > 0, "no native GPU backend tested");
}
