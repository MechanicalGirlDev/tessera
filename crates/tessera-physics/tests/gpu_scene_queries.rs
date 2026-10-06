//! World and batch automatically rebuild GPU scene indexes without state readback.
#![cfg(feature = "gpu-contact")]
use tessera_physics::{
    gpu_contact_pipeline::GpuContactDevice,
    gpu_point_query::GpuRigidPoint,
    gpu_ray_query::GpuRigidRay,
    gpu_rigid_sphere_contact::GpuRigidCollisionGroups,
    gpu_rigid_sphere_world::{
        GpuRigidEnvironmentSceneQueries, GpuRigidSphereBatch, GpuRigidSphereEnvironment,
        GpuRigidSphereWorldConfig,
    },
    gpu_rigid_state::GpuRigidBodyState,
};
#[test]
fn automatic_scene_queries_observe_motion_forces_and_topology()
-> Result<(), Box<dyn core::error::Error>> {
    let states = (0..16)
        .map(|i| GpuRigidBodyState {
            position_inverse_mass: [i as f32 * 4.0, 0.0, 3.0, 1.0],
            orientation: [0.0, 0.0, 0.0, 1.0],
            linear_velocity: [0.0; 4],
            angular_velocity: [0.0; 4],
            inverse_inertia_sleep: [1.0, 1.0, 1.0, 0.0],
        })
        .collect::<Vec<_>>();
    let ray = GpuRigidRay {
        origin: [2.0, 0.0, 3.0],
        direction: [-2.0, 0.0, 0.0],
        max_t: 1.0,
        groups: GpuRigidCollisionGroups::default(),
        excluded_body: None,
        body_range: None,
        solid: false,
    };
    let point = GpuRigidPoint {
        point: ray.origin,
        max_distance: 2.0,
        groups: ray.groups,
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
            states: &states,
            radii: &[0.5; 16],
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
        for environment in 0..2 {
            let (ray_hits, point_hits) =
                batch.query_scene_environment(environment, &[ray], &[point])?;
            assert_eq!(ray_hits[0].ids[0], 0);
            assert_eq!(point_hits[0].ids[0], 0);
            let hit = batch.cast_rays_environment(environment, &[ray])?[0];
            assert_eq!(hit.ids, [0, 0, 1, 0]);
            assert_eq!(hit.point_toi, [0.5, 0.0, 3.0, 0.75]);
            let hit = batch.project_points_environment(environment, &[point])?[0];
            assert_eq!(hit.ids, [0, 0, 1, 0]);
            assert_eq!(hit.point_distance, [0.5, 0.0, 3.0, 1.5]);
        }
        let excluded = GpuRigidPoint {
            excluded_body: Some(0),
            ..point
        };
        let batched = batch.query_scene_environments(&[
            GpuRigidEnvironmentSceneQueries {
                rays: &[ray],
                points: &[point, excluded],
            },
            GpuRigidEnvironmentSceneQueries {
                rays: &[
                    ray,
                    GpuRigidRay {
                        excluded_body: Some(0),
                        ..ray
                    },
                ],
                points: &[point],
            },
        ])?;
        assert_eq!(batched.len(), 2);
        assert_eq!(batched[0].rays[0].ids[0], 0);
        assert_eq!(batched[0].points[0].ids[0], 0);
        assert_eq!(batched[0].points[1].ids[0], 1);
        assert_eq!(batched[1].rays[0].ids[0], 0);
        assert_eq!(batched[1].rays[1].ids[2], 0);
        assert_eq!(batched[1].points[0].ids[0], 0);
        assert!(batch.query_scene_environments(&[]).is_err());
        assert!(
            batch
                .query_scene_environments(&[
                    GpuRigidEnvironmentSceneQueries {
                        rays: &[],
                        points: &[]
                    },
                    GpuRigidEnvironmentSceneQueries {
                        rays: &[],
                        points: &[GpuRigidPoint {
                            excluded_body: Some(16),
                            ..point
                        }]
                    },
                ])
                .is_err()
        );
        let mut encoder = context.device().create_command_encoder(&Default::default());
        let (ray_pass, point_pass) = batch.world().encode_scene_queries(
            &mut encoder,
            &[GpuRigidRay {
                body_range: Some([16, 32]),
                ..ray
            }],
            &[GpuRigidPoint {
                body_range: Some([16, 32]),
                ..point
            }],
        )?;
        let _ = context.queue().submit(Some(encoder.finish()));
        assert_eq!(
            ray_pass.readback(context.device(), context.queue())?[0].ids[0],
            16
        );
        assert_eq!(
            point_pass.readback(context.device(), context.queue())?[0].ids[0],
            16
        );
        let (ray_hits, point_hits) = batch.world().query_scene(&[ray], &[point])?;
        assert_eq!(ray_hits[0].ids[0], 0);
        assert_eq!(point_hits[0].ids[0], 0);
        let mut moved = states[0];
        moved.position_inverse_mass[0] = 1.0;
        batch.world_mut().write_body(16, moved)?;
        let (ray_hits, point_hits) = batch.world().query_scene(
            &[GpuRigidRay {
                body_range: Some([16, 32]),
                ..ray
            }],
            &[GpuRigidPoint {
                body_range: Some([16, 32]),
                ..point
            }],
        )?;
        assert!((ray_hits[0].point_toi[3] - 0.25).abs() < 2e-5);
        assert!((point_hits[0].point_distance[3] - 0.5).abs() < 2e-5);
        assert_eq!(
            batch.cast_rays_environment(1, &[ray])?[0].point_toi,
            [1.5, 0.0, 3.0, 0.25]
        );
        assert_eq!(
            batch.project_points_environment(1, &[point])?[0].point_distance,
            [1.5, 0.0, 3.0, 0.5]
        );
        batch.world_mut().write_forces(
            16,
            tessera_physics::gpu_rigid_state::GpuRigidBodyForces {
                force: [10.0, 0.0, 0.0, 0.0],
                torque: [0.0; 4],
            },
        )?;
        assert!(
            batch
                .cast_rays_environment(1, &[GpuRigidRay { max_t: -1.0, ..ray }])
                .is_err()
        );
        assert_eq!(
            batch.project_points_environment(1, &[point])?[0].point_distance[3],
            0.5
        );
        let _ = batch.step(0.1)?;
        assert!((batch.cast_rays_environment(1, &[ray])?[0].point_toi[3] - 0.2).abs() < 2e-5);
        assert!(
            (batch.project_points_environment(1, &[point])?[0].point_distance[3] - 0.4).abs()
                < 2e-5
        );
        assert_eq!(
            batch.cast_rays_environment(0, &[ray])?[0].point_toi[3],
            0.75
        );
        let mut added = states[0];
        added.position_inverse_mass = [1.8, 0.0, 3.0, 0.0];
        added.inverse_inertia_sleep = [0.0; 4];
        assert_eq!(batch.append_body_environment(1, added, 0.1)?, 16);
        let ray_hit = batch.cast_rays_environment(1, &[ray])?[0];
        let point_hit = batch.project_points_environment(1, &[point])?[0];
        assert_eq!(ray_hit.ids[0], 16);
        assert_eq!(point_hit.ids[0], 16);
        assert!((ray_hit.point_toi[3] - 0.05).abs() < 2e-5);
        assert!((point_hit.point_distance[3] - 0.1).abs() < 2e-5);
        let batched = batch.query_scene_environments(&[
            GpuRigidEnvironmentSceneQueries {
                rays: &[],
                points: &[],
            },
            GpuRigidEnvironmentSceneQueries {
                rays: &[ray],
                points: &[point],
            },
        ])?;
        assert!(batched[0].rays.is_empty() && batched[0].points.is_empty());
        assert_eq!(batched[1].rays[0].ids[0], 16);
        assert_eq!(batched[1].points[0].ids[0], 16);
    }
    assert!(tested > 0);
    Ok(())
}
