//! Environment-local ray selection with shared live sphere state.
#![cfg(feature = "gpu-contact")]
use tessera_physics::{
    gpu_contact_pipeline::GpuContactDevice,
    gpu_point_query::GpuRigidPoint,
    gpu_rigid_sphere_contact::GpuRigidCollisionGroups,
    gpu_rigid_sphere_world::{
        GpuRigidSphereBatch, GpuRigidSphereEnvironment, GpuRigidSphereWorldConfig,
    },
    gpu_rigid_state::GpuRigidBodyState,
};
#[test]
fn point_batch_filters_identical_environments_without_state_readback()
-> Result<(), Box<dyn core::error::Error>> {
    let state = GpuRigidBodyState {
        position_inverse_mass: [0.0, 0.0, 3.0, 1.0],
        orientation: [0.0, 0.0, 0.0, 1.0],
        linear_velocity: [0.0; 4],
        angular_velocity: [0.0; 4],
        inverse_inertia_sleep: [1.0, 1.0, 1.0, 0.0],
    };
    let ray = GpuRigidPoint {
        point: [2.0, 0.0, 3.0],
        max_distance: 2.0,
        groups: GpuRigidCollisionGroups::default(),
        excluded_body: None,
        body_range: None,
        solid: false,
    };
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        tested += 1;
        let env = GpuRigidSphereEnvironment {
            states: &[state],
            radii: &[0.5],
        };
        let mut batch = GpuRigidSphereBatch::new(
            context.device(),
            context.queue(),
            &[env, env],
            GpuRigidSphereWorldConfig {
                gravity: [0.0; 3],
                ground_half_extent: None,
                ..Default::default()
            },
        )?;
        let mut exclude = ray;
        exclude.excluded_body = Some(0);
        let queries = batch.prepare_point_queries_environment(1, &[ray, exclude])?;
        let mut encoder = context.device().create_command_encoder(&Default::default());
        queries.encode(&mut encoder);
        let _ = context.queue().submit(Some(encoder.finish()));
        let hits = queries.readback(context.device(), context.queue())?;
        assert_eq!(hits[0].ids[0], 1);
        assert_eq!(hits[0].ids[2], 1);
        assert!((hits[0].point_distance[3] - 1.5).abs() < 1e-5);
        assert_eq!(hits[1].ids[2], 0);
        assert_eq!(hits[0].point_distance[..3], [0.5, 0.0, 3.0]);
        let limited = GpuRigidPoint {
            max_distance: 1.5,
            ..ray
        };
        let masked = GpuRigidPoint {
            groups: GpuRigidCollisionGroups {
                memberships: 0,
                filter: u32::MAX,
            },
            ..ray
        };
        let filter_zero = GpuRigidPoint {
            groups: GpuRigidCollisionGroups {
                memberships: u32::MAX,
                filter: 0,
            },
            ..ray
        };
        let boundary = batch.project_points_environment(1, &[limited, masked, filter_zero])?;
        assert_eq!(boundary[0].ids, [0, 0, 1, 0]);
        assert_eq!(boundary[0].point_distance[3], 1.5);
        assert_eq!(boundary[1].ids[2], 0);
        assert_eq!(boundary[2].ids[2], 0);
        let tie = batch.world().project_points(&[ray])?;
        assert_eq!(tie[0].ids[0], 0);
        assert!(
            batch
                .project_points_environment(
                    1,
                    &[GpuRigidPoint {
                        point: [f32::NAN, 0.0, 0.0],
                        ..ray
                    }]
                )
                .is_err()
        );
        let mut moved = state;
        moved.position_inverse_mass[0] = 1.0;
        batch.world_mut().write_body(1, moved)?;
        let mut encoder = context.device().create_command_encoder(&Default::default());
        queries.encode(&mut encoder);
        let _ = context.queue().submit(Some(encoder.finish()));
        let hits = queries.readback(context.device(), context.queue())?;
        assert!((hits[0].point_distance[3] - 0.5).abs() < 1e-5);
        assert_eq!(hits[0].ids[0], 1);
        // Fresh batches share the cached pipeline but bind current state and metadata.
        for _ in 0..2 {
            let local = batch.project_points_environment(1, &[ray, exclude])?;
            assert_eq!(local[0].ids[0], 0);
            assert!((local[0].point_distance[3] - 0.5).abs() < 1e-5);
            assert_eq!(local[1].ids[2], 0);
        }
        let global = batch.world().project_points(&[ray])?;
        assert_eq!(global[0].ids[0], 1);
        assert!(batch.prepare_point_queries_environment(2, &[ray]).is_err());
        let mut invalid = ray;
        invalid.body_range = Some([0, 2]);
        assert!(
            batch
                .prepare_point_queries_environment(1, &[invalid])
                .is_err()
        );
        let mut added = state;
        added.position_inverse_mass = [1.8, 0.0, 3.0, 0.0];
        added.inverse_inertia_sleep = [0.0; 4];
        assert_eq!(batch.append_body_environment(1, added, 0.1)?, 1);
        let after = batch.project_points_environment(1, &[ray])?;
        assert_eq!(after[0].ids[0], 1);
        assert!(
            (after[0].point_distance[3] - 0.1).abs() < 1e-5,
            "{backend:?}: {:?}",
            after[0]
        );
    }
    assert!(tested > 0);
    Ok(())
}
