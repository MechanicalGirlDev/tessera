//! Full-owner prescribed motion and accumulated compliant constraint regressions.

use nalgebra::{DVector, Isometry3, Matrix3, Vector3};
use tessera_physics::articulated_world::{
    ArticulatedWorld, ArticulatedWorldParams, JointPolynomialCoupling, LinkFixedConstraint,
    LinkPointConstraint, LinkSphere,
};
use tessera_physics::articulation::{Articulation, JointKind, JointSpec, LinkSpec};
use tessera_physics::prescribed_dof::{ArticulatedConstraintPolicy, ConstraintCompliance};

type TestResult = Result<(), Box<dyn core::error::Error>>;

const DT: f64 = 0.005;
const SOFT: ConstraintCompliance = ConstraintCompliance {
    erp: 0.2,
    cfm: 0.02,
};

fn link(mass: f64) -> LinkSpec {
    LinkSpec {
        mass,
        center_of_mass: Vector3::zeros(),
        inertia: Matrix3::identity() * mass,
    }
}

fn tree(
    chain: bool,
    count: usize,
    bounds: Option<(f64, f64)>,
) -> Result<Articulation, tessera_physics::articulation::ArticulationError> {
    Articulation::new(
        core::iter::once(link(0.0))
            .chain((0..count).map(|_| link(1.0)))
            .collect(),
        (0..count)
            .map(|slot| JointSpec {
                parent: if chain { slot } else { 0 },
                child: slot + 1,
                kind: JointKind::Prismatic,
                origin: Isometry3::identity(),
                axis: Vector3::z(),
                limits: bounds,
            })
            .collect(),
        0,
    )
}

fn world(
    art: Articulation,
) -> Result<ArticulatedWorld, tessera_physics::articulated_world::ArticulatedWorldError> {
    ArticulatedWorld::new(
        art,
        Isometry3::identity(),
        Vec::new(),
        ArticulatedWorldParams {
            gravity: [0.0; 3],
            max_substep: DT,
            ..Default::default()
        },
    )
}

fn configure_row(
    world: &mut ArticulatedWorld,
    family: usize,
) -> Result<(), tessera_physics::articulated_world::ArticulatedWorldError> {
    match family {
        0 => world.set_joint_polynomial_couplings(vec![JointPolynomialCoupling {
            follower: 0,
            source: None,
            coefficients: [0.0; 5],
            follower_reference: 0.0,
            source_reference: 0.0,
        }])?,
        1 => world.set_link_point_constraints(vec![LinkPointConstraint {
            link_a: 1,
            point_a: [0.0; 3],
            link_b: None,
            point_b: [0.0; 3],
        }])?,
        2 => world.set_link_fixed_constraints(vec![LinkFixedConstraint {
            link_a: 1,
            frame_a: Isometry3::identity(),
            link_b: None,
            frame_b: Isometry3::identity(),
        }])?,
        3 => world.set_joint_limit_compliance(&[SOFT])?,
        _ => unreachable!(),
    }
    let mut policy = ArticulatedConstraintPolicy::default();
    match family {
        0 => policy.coupling = SOFT,
        1 => policy.point_loop = SOFT,
        2 => policy.fixed_loop = SOFT,
        3 => {}
        _ => unreachable!(),
    }
    world.set_constraint_policy(policy)?;
    Ok(())
}

#[test]
fn cpu_prescribed_mass_contact_and_snapshot_are_owner_behaviors() -> TestResult {
    let mut model = world(tree(true, 2, None)?)?;
    model.positions = DVector::from_vec(vec![0.2, 0.3]);
    model.velocities = DVector::from_vec(vec![-1.0, 0.0]);
    model.set_prescribed_coordinates(&[true, false])?;
    model.step(DT, &[1000.0, 2.0])?;
    assert!((model.velocities[0] + 1.0).abs() < 1e-12);
    assert!((model.velocities[1] - 2.0 * DT).abs() < 1e-12);

    model.positions = DVector::from_vec(vec![0.2, 0.3]);
    model.velocities = DVector::from_vec(vec![-1.0, 0.0]);
    model.colliders.push(LinkSphere {
        link: 2,
        center: Vector3::zeros(),
        radius: 0.5,
    });
    let snapshot = model.snapshot();
    model.step(DT, &[1000.0, 2.0])?;
    assert!((model.velocities[0] + 1.0).abs() < 1e-12);
    assert!((model.velocities[1] - 1.0).abs() < 1e-10);
    assert!(model.contact_forces[2].z > 0.0);
    assert!(model.set_prescribed_coordinates(&[true]).is_err());
    assert_eq!(model.prescribed_coordinates(), &[true, false]);
    model.set_prescribed_coordinates(&[false, true])?;
    model.restore_snapshot(&snapshot)?;
    assert_eq!(model.prescribed_coordinates(), &[true, false]);
    Ok(())
}

#[test]
fn cpu_prescribed_slots_bypass_caps_stops_and_floating_projection() -> TestResult {
    let mut model = world(tree(false, 1, Some((-0.01, 0.0)))?)?;
    model.velocities[0] = 2.0;
    model.joint_velocity_limit = Some(0.1);
    model.set_prescribed_coordinates(&[true])?;
    model.step(DT, &[-1000.0])?;
    assert!((model.positions[0] - 2.0 * DT).abs() < 1e-12);
    assert!((model.velocities[0] - 2.0).abs() < 1e-12);

    let art = Articulation::new(vec![link(2.0)], Vec::new(), 0)?;
    let mut floating = ArticulatedWorld::new_floating(
        art,
        Isometry3::identity(),
        Vec::new(),
        ArticulatedWorldParams {
            gravity: [0.0, 0.0, -10.0],
            max_substep: DT,
            ..Default::default()
        },
    )?;
    floating.base_linear_velocity = Vector3::new(0.0, 0.0, 2.0);
    floating.base_force = Vector3::new(10.0, 0.0, -1000.0);
    floating.set_prescribed_coordinates(&[false, false, true, false, false, false])?;
    floating.step(DT, &[])?;
    assert!((floating.base_linear_velocity.z - 2.0).abs() < 1e-12);
    assert!((floating.base_linear_velocity.x - 5.0 * DT).abs() < 1e-12);
    assert!((floating.root_pose.translation.z - 2.0 * DT).abs() < 1e-12);
    Ok(())
}

#[test]
fn cpu_each_compliant_row_accumulates_impulses_and_retains_its_policy() -> TestResult {
    for family in 0..4 {
        let mut model = world(tree(false, 1, (family == 3).then_some((-1.0, 0.0)))?)?;
        configure_row(&mut model, family)?;
        let snapshot = model.snapshot();
        let policy = model.constraint_policy();
        assert!(
            model
                .set_constraint_policy(ArticulatedConstraintPolicy {
                    fixed_contact: ConstraintCompliance {
                        erp: f64::NAN,
                        cfm: 0.0
                    },
                    ..policy
                })
                .is_err()
        );
        for _ in 0..400 {
            model.step(DT, &[5.0])?;
        }
        let expected = SOFT.cfm * 5.0 * DT / SOFT.erp;
        assert!(
            (model.positions[0] - expected).abs() < 1e-8,
            "family={family} q={}",
            model.positions[0]
        );
        model.set_constraint_policy(ArticulatedConstraintPolicy::default())?;
        model.restore_snapshot(&snapshot)?;
        assert_eq!(model.constraint_policy(), policy);
        model.step(DT, &[5.0])?;
        // Twelve sweeps must not repeatedly erase a compliant row's impulse.
        let gamma = SOFT.cfm / DT;
        assert!((model.velocities[0] - 5.0 * DT * gamma / (1.0 + gamma)).abs() < 1e-12);
    }
    Ok(())
}

#[test]
fn cpu_contact_softness_distinguishes_fixed_and_dynamic_response() -> TestResult {
    for dynamic in [false, true] {
        let mut model = world(tree(false, if dynamic { 2 } else { 1 }, None)?)?;
        model.root_pose = Isometry3::translation(0.0, 0.0, 2.0);
        model.positions[0] = 1.0;
        model.colliders.push(LinkSphere {
            link: 1,
            center: Vector3::zeros(),
            radius: 0.5,
        });
        model.colliders.push(LinkSphere {
            link: if dynamic { 2 } else { 0 },
            center: Vector3::zeros(),
            radius: 0.5,
        });
        model.set_joint_contacts_enabled(true);
        model.set_constraint_policy(ArticulatedConstraintPolicy {
            dynamic_contact: ConstraintCompliance {
                erp: 0.2,
                cfm: 0.04,
            },
            fixed_contact: ConstraintCompliance {
                erp: 0.2,
                cfm: 0.01,
            },
            ..Default::default()
        })?;
        for _ in 0..400 {
            model.step(DT, if dynamic { &[-5.0, 5.0] } else { &[-5.0] })?;
        }
        let distance = model.positions[0] - if dynamic { model.positions[1] } else { 0.0 };
        let expected = if dynamic { 0.04 } else { 0.01 } * 5.0 * DT / 0.2;
        assert!(
            (1.0 - distance - expected).abs() < 1e-8,
            "dynamic={dynamic} distance={distance}"
        );
    }
    Ok(())
}

#[test]
fn joint_contact_enablement_preserves_explicit_exclusions_and_snapshots() -> TestResult {
    let mut model = world(tree(false, 1, None)?)?;
    model.root_pose = Isometry3::translation(0.0, 0.0, 2.0);
    model.positions[0] = 0.9;
    model.colliders = vec![
        LinkSphere {
            link: 0,
            center: Vector3::zeros(),
            radius: 0.5,
        },
        LinkSphere {
            link: 1,
            center: Vector3::zeros(),
            radius: 0.5,
        },
    ];
    let initial = model.snapshot();
    model.step(DT, &[0.0])?;
    assert!(model.velocities[0].abs() < 1e-12);
    model.restore_snapshot(&initial)?;
    model.set_joint_contacts_enabled(true);
    model.step(DT, &[0.0])?;
    assert!(model.velocities[0] > 0.0);
    model.restore_snapshot(&initial)?;
    assert!(!model.joint_contacts_enabled());
    model.articulation.exclude_collision_pair(0, 1)?;
    model.set_joint_contacts_enabled(true);
    assert!(model.articulation.collision_pair_excluded(0, 1));
    model.step(DT, &[0.0])?;
    assert!(model.velocities[0].abs() < 1e-12);
    Ok(())
}

#[cfg(feature = "gpu-contact")]
mod resident {
    use super::*;
    use nalgebra::UnitQuaternion;
    use tessera_physics::gpu_articulated_dynamics::{
        GpuArticulatedDynamicsBatch, GpuArticulatedDynamicsInput,
    };
    use tessera_physics::gpu_articulated_spherical::GpuSphericalJointState;
    use tessera_physics::gpu_articulated_state::GpuGeneralizedState;
    use tessera_physics::gpu_contact_pipeline::GpuContactDevice;

    #[test]
    fn selected_resident_masks_preserve_motion_under_coupled_inertia_and_contact() -> TestResult {
        let context = GpuContactDevice::new_with_backends(wgpu::Backends::VULKAN)?;
        for contact in [false, true] {
            let mut model = world(tree(true, 2, None)?)?;
            model.positions = DVector::from_vec(vec![0.2, 0.3]);
            model.velocities = DVector::from_vec(vec![-1.0, 0.0]);
            if contact {
                model.colliders.push(LinkSphere {
                    link: 2,
                    center: Vector3::zeros(),
                    radius: 0.5,
                });
            }
            let inputs = [
                model.gpu_dynamics_input(&[1000.0, 2.0])?,
                model.gpu_dynamics_input(&[1000.0, 2.0])?,
            ];
            let mut batch = GpuArticulatedDynamicsBatch::new(&context, &inputs, DT)?;
            let masks = vec![vec![true, false], vec![false, true]];
            batch.set_prescribed_coordinates(&masks)?;
            assert!(
                batch
                    .set_prescribed_coordinates(&[vec![true; 2], vec![true]])
                    .is_err()
            );
            assert_eq!(batch.prescribed_coordinates(), &masks);
            batch.submit_steps(1)?;
            let actual = batch.readback()?;
            assert!((actual[0].velocities[0] + 1.0).abs() < 1e-6);
            assert!((actual[0].velocities[1] - if contact { 1.0 } else { 2.0 * DT }).abs() < 1e-5);
            assert!(actual[1].velocities[1].abs() < 1e-6);
            if !contact {
                assert!((actual[1].velocities[0] - 1.5).abs() < 1e-5);
            }
        }
        Ok(())
    }

    #[test]
    fn owner_compliant_rows_execute_without_low_level_solver_calls() -> TestResult {
        let context = GpuContactDevice::new_with_backends(wgpu::Backends::VULKAN)?;
        for family in 0..4 {
            let mut model = world(tree(false, 1, (family == 3).then_some((-1.0, 0.0)))?)?;
            configure_row(&mut model, family)?;
            let mut batch = GpuArticulatedDynamicsBatch::new(
                &context,
                &[model.gpu_dynamics_input(&[5.0])?],
                DT,
            )?;
            batch.set_constraint_policy(model.constraint_policy())?;
            batch.set_joint_limit_compliance(&[model.joint_limit_compliance().to_vec()])?;
            batch.submit_steps(1)?;
            let gamma = SOFT.cfm / DT;
            assert!(
                (batch.readback()?[0].velocities[0] - 5.0 * DT * gamma / (1.0 + gamma)).abs()
                    < 1e-6,
                "family={family}"
            );
            batch.submit_steps(399)?;
            let q = batch.readback()?[0].positions[0];
            assert!(
                (q - SOFT.cfm * 5.0 * DT / SOFT.erp).abs() < 2e-5,
                "family={family} q={q}"
            );
        }
        Ok(())
    }

    #[test]
    fn resident_normal_policy_uses_prescribed_response_not_nonzero_jacobians() -> TestResult {
        let context = GpuContactDevice::new_with_backends(wgpu::Backends::VULKAN)?;
        let mut model = world(tree(false, 2, None)?)?;
        model.root_pose = Isometry3::translation(0.0, 0.0, 2.0);
        model.positions[0] = 1.0;
        model.colliders = vec![
            LinkSphere {
                link: 1,
                center: Vector3::zeros(),
                radius: 0.5,
            },
            LinkSphere {
                link: 2,
                center: Vector3::zeros(),
                radius: 0.5,
            },
        ];
        let input = model.gpu_dynamics_input(&[-5.0, 5.0])?;
        let mut batch = GpuArticulatedDynamicsBatch::new(&context, &[input.clone(), input], DT)?;
        batch.set_prescribed_coordinates(&[vec![false, false], vec![false, true]])?;
        batch.set_constraint_policy(ArticulatedConstraintPolicy {
            dynamic_contact: ConstraintCompliance {
                erp: 0.2,
                cfm: 0.04,
            },
            fixed_contact: ConstraintCompliance {
                erp: 0.2,
                cfm: 0.01,
            },
            ..Default::default()
        })?;
        batch.submit_steps(400)?;
        for (actual, cfm) in batch.readback()?.iter().zip([0.04, 0.01]) {
            let penetration = 1.0 - actual.positions[0] + actual.positions[1];
            assert!((penetration - cfm * 5.0 * DT / 0.2).abs() < 2e-5);
        }
        Ok(())
    }

    #[test]
    fn resident_prescribed_coordinate_bypasses_speed_position_and_sleep_clamps() -> TestResult {
        let context = GpuContactDevice::new_with_backends(wgpu::Backends::VULKAN)?;
        let art = Articulation::new(
            vec![link(0.0), link(1.0), link(1.0)],
            vec![
                JointSpec {
                    parent: 0,
                    child: 1,
                    kind: JointKind::Prismatic,
                    origin: Isometry3::identity(),
                    axis: Vector3::x(),
                    limits: Some((-1.0, 0.0)),
                },
                JointSpec {
                    parent: 1,
                    child: 2,
                    kind: JointKind::Prismatic,
                    origin: Isometry3::identity(),
                    axis: Vector3::z(),
                    limits: None,
                },
            ],
            0,
        )?;
        let mut model = world(art)?;
        model.root_pose = Isometry3::translation(0.0, 0.0, 0.5);
        model.velocities[0] = 0.001;
        model.joint_velocity_limit = Some(0.0001);
        model.colliders.push(LinkSphere {
            link: 2,
            center: Vector3::zeros(),
            radius: 0.5,
        });
        let mut input = model.gpu_dynamics_input(&[0.0, 0.0])?;
        input.gravity = Vector3::new(0.0, 0.0, -9.81);
        let mut batch = GpuArticulatedDynamicsBatch::new(&context, &[input], DT)?;
        batch.enable_contact_activity()?;
        batch.enable_contact_sleep(&[Some(tessera_physics::sleep::SleepSettings {
            enabled: true,
            linear_velocity_threshold: 1.0,
            angular_velocity_threshold: 1.0,
            time_threshold: DT,
        })])?;
        // Updating after sleep setup must rebuild the owned freeze selection.
        batch.set_prescribed_coordinates(&[vec![true, false]])?;
        batch.submit_steps(4)?;
        let actual = batch.readback()?;
        assert!((actual[0].velocities[0] - 0.001).abs() < 1e-9);
        assert!((actual[0].positions[0] - 0.001 * DT * 4.0).abs() < 1e-9);
        assert_eq!(batch.readback_sleeping_coordinates()?, [vec![false, true]]);
        Ok(())
    }

    #[test]
    fn resident_joint_contact_filter_respects_explicit_exclusions() -> TestResult {
        let context = GpuContactDevice::new_with_backends(wgpu::Backends::VULKAN)?;
        for (enabled, excluded) in [(false, false), (true, false), (true, true)] {
            let mut model = world(tree(false, 1, None)?)?;
            model.root_pose = Isometry3::translation(0.0, 0.0, 2.0);
            model.positions[0] = 0.9;
            model.colliders = vec![
                LinkSphere {
                    link: 0,
                    center: Vector3::zeros(),
                    radius: 0.5,
                },
                LinkSphere {
                    link: 1,
                    center: Vector3::zeros(),
                    radius: 0.5,
                },
            ];
            model.set_joint_contacts_enabled(enabled);
            if excluded {
                model.articulation.exclude_collision_pair(0, 1)?;
            }
            let batch = GpuArticulatedDynamicsBatch::new(
                &context,
                &[model.gpu_dynamics_input(&[0.0])?],
                DT,
            )?;
            batch.submit_steps(1)?;
            let speed = batch.readback()?[0].velocities[0];
            assert_eq!(speed > 1e-6, enabled && !excluded);
        }
        Ok(())
    }

    #[test]
    fn resident_and_cpu_prescribed_tangent_and_floating_modes_integrate() -> TestResult {
        let context = GpuContactDevice::new_with_backends(wgpu::Backends::VULKAN)?;
        let spherical = Articulation::new(
            vec![link(0.0), link(1.0)],
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
        let orientation = UnitQuaternion::from_euler_angles(0.2, 1.55, -0.3);
        let joints = vec![GpuSphericalJointState {
            velocity_slot: 0,
            orientation,
        }];
        let mut model = world(spherical)?;
        model.set_tangent_spherical_state(&joints, &DVector::from_vec(vec![0.0, 0.0, 2.0]))?;
        model.set_prescribed_coordinates(&[false, false, true])?;
        model.step(DT, &[0.0, 0.0, 1000.0])?;
        assert!((model.velocities[2] - 2.0).abs() < 1e-12);
        let expected = UnitQuaternion::from_scaled_axis(Vector3::z() * (2.0 * DT)) * orientation;
        assert!(
            model
                .gpu_spherical_state()
                .ok_or("missing tangent spherical state")?[0]
                .orientation
                .angle_to(&expected)
                < 1e-8
        );
        let state = GpuGeneralizedState {
            positions: DVector::zeros(3),
            velocities: DVector::from_vec(vec![0.0, 0.0, 2.0]),
        };
        let mut input = GpuArticulatedDynamicsInput::new(
            &model.articulation,
            Isometry3::identity(),
            state,
            Vector3::zeros(),
        );
        input.joints[2].base_force = 1000.0;
        let mut batch = GpuArticulatedDynamicsBatch::new_with_spherical_state(
            &context,
            &[input],
            DT,
            &[false],
            &[joints],
            &[vec![None]],
        )?;
        batch.set_prescribed_coordinates(&[vec![false, false, true]])?;
        batch.submit_steps(1)?;
        assert!((batch.readback()?[0].velocities[2] - 2.0).abs() < 1e-6);
        assert!(
            batch.readback_output()?[0]
                .spherical_joints
                .as_ref()
                .ok_or("missing spherical output")?[0]
                .orientation
                .angle_to(&expected)
                < 1e-5
        );

        let art = Articulation::new(vec![link(2.0)], Vec::new(), 0)?;
        let state = GpuGeneralizedState {
            positions: DVector::zeros(6),
            velocities: DVector::from_vec(vec![0.0, 0.0, 2.0, 0.0, 0.0, 0.0]),
        };
        let mut input = GpuArticulatedDynamicsInput::new(
            &art,
            Isometry3::identity(),
            state,
            Vector3::new(0.0, 0.0, -10.0),
        );
        input.joints = vec![Default::default(); 6];
        input.joint_friction = vec![0.0; 6];
        let mut batch =
            GpuArticulatedDynamicsBatch::new_with_floating_roots(&context, &[input], DT, &[true])?;
        batch.set_prescribed_coordinates(&[vec![false, false, true, false, false, false]])?;
        batch.submit_steps(1)?;
        assert!((batch.readback()?[0].velocities[2] - 2.0).abs() < 1e-6);
        assert!((batch.readback_root_poses()?[0].translation.z - 2.0 * DT).abs() < 1e-6);
        Ok(())
    }
}
