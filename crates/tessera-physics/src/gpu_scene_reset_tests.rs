use super::*;
use crate::articulated_world::ArticulatedWorldParams;
use crate::articulation::{Articulation, JointKind, JointSpec, LinkSpec};
use crate::prescribed_dof::{ArticulatedConstraintPolicy, ConstraintCompliance};
use nalgebra::{Isometry3, Matrix3, Vector3};

fn world(floating: bool, prescribed: bool) -> Result<ArticulatedWorld, ArticulatedWorldError> {
    let link = LinkSpec {
        mass: 1.0,
        center_of_mass: Vector3::zeros(),
        inertia: Matrix3::identity(),
    };
    let articulation = Articulation::new(
        vec![link.clone(), link],
        vec![JointSpec {
            parent: 0,
            child: 1,
            kind: JointKind::Prismatic,
            origin: Isometry3::identity(),
            axis: Vector3::x(),
            limits: Some((-1.0, 1.0)),
        }],
        0,
    )?;
    let params = ArticulatedWorldParams {
        gravity: [0.0; 3],
        ..Default::default()
    };
    let mut world = if floating {
        ArticulatedWorld::new_floating(articulation, Isometry3::identity(), vec![], params)?
    } else {
        ArticulatedWorld::new(articulation, Isometry3::identity(), vec![], params)?
    };
    let mut mask = vec![false; if floating { 7 } else { 1 }];
    if floating {
        mask[0] = true;
        mask[4] = true;
    }
    let last = mask.len() - 1;
    mask[last] = prescribed;
    world.set_prescribed_coordinates(&mask)?;
    world.set_joint_limit_compliance(&[ConstraintCompliance {
        erp: 0.3,
        cfm: 0.002,
    }])?;
    world.set_constraint_policy(ArticulatedConstraintPolicy {
        fixed_contact: ConstraintCompliance {
            erp: 0.4,
            cfm: 0.001,
        },
        ..Default::default()
    })?;
    world.velocities[0] = 0.25;
    Ok(world)
}

#[test]
fn scene_owner_reset_policy_construction_maps_root_and_joint_masks_and_compliance()
-> Result<(), Box<dyn core::error::Error>> {
    let context = GpuContactDevice::new()?;
    let mut floating = world(true, true)?;
    let mut single = GpuSceneDynamics::new(&floating, &context, 0.01)?;
    let scene = &single.source.scene;
    let root = scene
        .robot_root
        .as_ref()
        .ok_or("missing floating robot layout")?;
    let mask = &single.batch.prescribed_coordinates()[0];
    assert!(mask[root.translation.start]);
    assert!(mask[root.angular.start + 1]);
    assert!(mask[scene.robot_coordinates.start]);
    assert_eq!(mask.iter().filter(|&&selected| selected).count(), 3);
    assert_eq!(
        single.batch.joint_limit_compliance()[0][scene.robot_coordinates.clone()],
        *floating.joint_limit_compliance(),
    );
    let _diagnostics = single.step(&mut floating, &[100.0], 1)?;
    assert!((floating.velocities[0] - 0.25).abs() < 1e-6);

    let mut worlds = [world(false, true)?, world(false, false)?];
    let mut batch = GpuSceneDynamicsBatch::new(&worlds, &context, 0.01)?;
    assert_eq!(
        batch.batch.prescribed_coordinates(),
        &[vec![true], vec![false]]
    );
    assert_eq!(
        batch.batch.joint_limit_compliance()[0],
        worlds[0].joint_limit_compliance()
    );
    let _diagnostics = batch.step(&mut worlds, &[&[10.0], &[10.0]], 1)?;
    assert!((worlds[0].velocities[0] - 0.25).abs() < 1e-6);
    assert!(worlds[1].velocities[0] > 0.3);
    Ok(())
}

#[test]
fn scene_owner_reset_policy_construction_rejects_nonshared_contact_policy()
-> Result<(), Box<dyn core::error::Error>> {
    let context = GpuContactDevice::new()?;
    let mut worlds = [world(false, true)?, world(false, false)?];
    worlds[1].set_constraint_policy(ArticulatedConstraintPolicy::default())?;
    assert!(matches!(
        GpuSceneDynamicsBatch::new(&worlds, &context, 0.01),
        Err(GpuSceneDynamicsError::ConstraintPolicyMismatch),
    ));
    Ok(())
}

fn scene_world() -> Result<ArticulatedWorld, ArticulatedWorldError> {
    use crate::articulated_world::{SceneBody, SceneCollider};
    let mut world = world(true, false)?;
    let mut dynamic = SceneBody::new(
        Isometry3::translation(0.0, 3.0, 1.0),
        1.0,
        Matrix3::identity(),
        vec![SceneCollider::Sphere {
            center: Vector3::zeros(),
            radius: 0.1,
        }],
    )?;
    dynamic.linear_velocity = Vector3::x() * 0.4;
    let _dynamic = world.add_scene_body(dynamic);
    for (index, paired) in [true, false].into_iter().enumerate() {
        let mut body = SceneBody::new(
            Isometry3::translation(3.0 + index as f64, 0.0, 1.0),
            0.0,
            Matrix3::identity(),
            if paired {
                vec![SceneCollider::Sphere {
                    center: Vector3::zeros(),
                    radius: 0.1,
                }]
            } else {
                vec![]
            },
        )?;
        body.kinematic = true;
        body.linear_velocity = Vector3::new(0.1, 0.2, 0.3);
        body.angular_velocity = Vector3::z() * 0.3;
        let _prescribed = world.add_scene_body(body);
    }
    let _static = world.add_scene_body(SceneBody::new(
        Isometry3::translation(5.0, 0.0, 0.0),
        0.0,
        Matrix3::identity(),
        vec![SceneCollider::Sphere {
            center: Vector3::zeros(),
            radius: 0.1,
        }],
    )?);
    Ok(world)
}

#[test]
fn scene_owner_reset_restores_all_body_kinds_and_preserves_unselected_environment()
-> Result<(), Box<dyn core::error::Error>> {
    let context = GpuContactDevice::new()?;
    let mut worlds = [scene_world()?, scene_world()?];
    let mut batch = GpuSceneDynamicsBatch::new(&worlds, &context, 0.01)?;
    let templates = batch.snapshot_reset_templates(&worlds)?;
    let initial = worlds[0].scene_bodies.clone();
    let initial_root = worlds[0].root_pose;
    let _diagnostics = batch.step(&mut worlds, &[&[1.0], &[2.0]], 3)?;
    let untouched_bodies = worlds[1].scene_bodies.clone();
    let untouched_state = worlds[1].positions.clone();
    assert_ne!(worlds[0].scene_bodies[1].pose, initial[1].pose);
    assert_ne!(worlds[0].scene_bodies[2].pose, initial[2].pose);
    worlds[0].scene_bodies[3].pose.translation.vector.x += 0.5;
    batch.update_static_scene_poses(&worlds)?;
    assert!(
        batch
            .reset_envs_from_templates(&mut worlds, &templates, &[0, 0])
            .is_err()
    );
    assert_eq!(worlds[1].scene_bodies, untouched_bodies);
    batch.reset_envs_from_templates(&mut worlds, &templates, &[0])?;
    assert_eq!(worlds[0].scene_bodies, initial);
    assert_eq!(worlds[0].root_pose, initial_root);
    assert_eq!(worlds[1].scene_bodies, untouched_bodies);
    assert_eq!(worlds[1].positions, untouched_state);
    let _diagnostics = batch.step(&mut worlds, &[&[1.0], &[2.0]], 1)?;
    let mut fresh = scene_world()?;
    let mut single = GpuSceneDynamics::new(&fresh, &context, 0.01)?;
    let _diagnostics = single.step(&mut fresh, &[1.0], 1)?;
    for (actual, expected) in worlds[0].scene_bodies.iter().zip(&fresh.scene_bodies) {
        assert!((actual.pose.translation.vector - expected.pose.translation.vector).norm() < 1e-6);
        assert!((actual.pose.rotation.inverse() * expected.pose.rotation).angle() < 1e-6);
        assert!((actual.linear_velocity - expected.linear_velocity).norm() < 1e-6);
    }
    let single_template = single.snapshot_reset_templates(&fresh)?;
    let before = fresh.scene_bodies.clone();
    let _diagnostics = single.step(&mut fresh, &[1.0], 2)?;
    single.reset_from_templates(&mut fresh, &single_template)?;
    assert_eq!(fresh.scene_bodies, before);
    let _diagnostics = single.step(&mut fresh, &[1.0], 1)?;
    Ok(())
}
