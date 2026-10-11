//! Deterministic real-device load response of native articulated ERP/CFM rows.
#![cfg(feature = "gpu-contact")]

use nalgebra::{DMatrix, DVector, Isometry3, Matrix3, Vector3};
use tessera_physics::articulated_world::{LinkFixedConstraint, LinkPointConstraint};
use tessera_physics::articulation::{Articulation, JointKind, JointSpec, LinkSpec};
use tessera_physics::gpu_articulated_dynamics::{
    GpuArticulatedDynamicsBatch, GpuArticulatedDynamicsInput,
};
use tessera_physics::gpu_articulated_ground_contact::{
    GpuArticulatedExternalSphereBodies, GpuArticulatedGroundSphere, GpuArticulatedSphereBodyInput,
    GpuArticulatedSphereOrbit, GpuArticulatedSpherePair, GpuArticulatedStaticConvexCapsulePair,
    GpuArticulatedStaticSpherePair,
};
use tessera_physics::gpu_articulated_joint_limit::GpuArticulatedJointLimitBatch;
use tessera_physics::gpu_articulated_mass_assembly::{
    GpuArticulatedMassAssemblyBatch, GpuMassAssemblySystem, GpuMassLink,
};
use tessera_physics::gpu_articulated_state::{GpuGeneralizedState, GpuGeneralizedStateBatch};
use tessera_physics::gpu_contact_pipeline::GpuContactDevice;
use tessera_physics::gpu_contact_policy::{GpuConstraintCompliance, GpuContactPolicy};

const DT: f64 = 0.005;
const STEPS: usize = 400;

fn tree(
    slots: usize,
    kind: JointKind,
    bounds: Option<(f64, f64)>,
) -> Result<Articulation, tessera_physics::articulation::ArticulationError> {
    let mut links = vec![
        LinkSpec {
            mass: 1.0,
            center_of_mass: Vector3::zeros(),
            inertia: Matrix3::identity(),
        };
        slots + 1
    ];
    links[0].mass = 0.0;
    links[0].inertia = Matrix3::zeros();
    Articulation::new(
        links,
        (0..slots)
            .map(|slot| JointSpec {
                parent: 0,
                child: slot + 1,
                kind,
                origin: Isometry3::identity(),
                axis: Vector3::z(),
                limits: bounds,
            })
            .collect(),
        0,
    )
}

fn state(positions: Vec<f64>) -> GpuGeneralizedState {
    GpuGeneralizedState {
        velocities: DVector::zeros(positions.len()),
        positions: DVector::from_vec(positions),
    }
}

fn soft(cfm: f32) -> GpuConstraintCompliance {
    GpuConstraintCompliance { erp: 0.2, cfm }
}

#[test]
fn normal_compliance_distinguishes_dynamic_fixed_and_prescribed_loads()
-> Result<(), Box<dyn core::error::Error>> {
    let context = GpuContactDevice::new_with_backends(wgpu::Backends::VULKAN)?;
    for warm_start in [false, true] {
        for mode in 0..4 {
            let art = tree(if mode == 1 { 2 } else { 1 }, JointKind::Prismatic, None)?;
            let mut displacement = Vec::new();
            for force in [5.0, 10.0] {
                let mut input = GpuArticulatedDynamicsInput::new(
                    &art,
                    Isometry3::identity(),
                    state(if mode == 1 { vec![1.0, 0.0] } else { vec![1.0] }),
                    Vector3::zeros(),
                );
                input.joints[0].base_force = -force;
                input.contact_warm_start = warm_start;
                if mode == 0 {
                    input.ground_spheres.push(GpuArticulatedGroundSphere {
                        link: 1,
                        local_center: Vector3::zeros(),
                        radius: 0.5,
                        plane_normal: Vector3::z(),
                        plane_offset: 0.5,
                        plane_xy_half_extent: None,
                        restitution: 0.0,
                        friction: 0.0,
                    });
                } else if mode == 3 {
                    input
                        .static_sphere_pairs
                        .push(GpuArticulatedStaticSpherePair {
                            link: 1,
                            local_center: Vector3::zeros(),
                            radius: 0.5,
                            static_center: Vector3::zeros(),
                            static_radius: 0.5,
                            restitution: 0.0,
                            friction: 0.0,
                        });
                    input.external_sphere_bodies = Some(GpuArticulatedExternalSphereBodies {
                        spheres: vec![Some(GpuArticulatedSphereBodyInput {
                            user_data: None,
                            orbit: GpuArticulatedSphereOrbit {
                                origin: Vector3::zeros(),
                                orientation: Default::default(),
                                linear_velocity: Vector3::zeros(),
                            },
                            angular_velocity: Vector3::new(0.37, 0.0, 0.0),
                        })],
                        ..Default::default()
                    });
                } else {
                    if mode == 1 {
                        input.joints[1].base_force = force;
                    }
                    input.sphere_pairs.push(GpuArticulatedSpherePair {
                        first_link: 1,
                        first_local_center: Vector3::zeros(),
                        first_radius: 0.5,
                        second_link: if mode == 1 { 2 } else { 0 },
                        second_local_center: Vector3::zeros(),
                        second_radius: 0.5,
                        restitution: 0.0,
                        friction: 0.0,
                    });
                }
                let batch = GpuArticulatedDynamicsBatch::new(&context, &[input], DT)?;
                let initial = batch.readback()?;
                batch.submit_steps(STEPS)?;
                let hard = batch.readback()?[0].positions.clone();
                assert!((hard[0] - 1.0).abs() < 2e-5);
                batch.reset(&initial)?;
                batch.update_contact_policy(GpuContactPolicy {
                    dynamic_normal: soft(0.04),
                    static_normal: soft(0.01),
                    ..Default::default()
                })?;
                // Malformed updates must leave the complete previous policy intact.
                assert!(
                    batch
                        .update_contact_policy(GpuContactPolicy {
                            dynamic_normal: soft(0.08),
                            static_normal: soft(f32::NAN),
                            ..Default::default()
                        })
                        .is_err()
                );
                batch.submit_steps(STEPS)?;
                let output = batch.readback()?;
                let q = &output[0].positions;
                let penetration = if mode == 1 {
                    1.0 - (q[0] - q[1])
                } else {
                    1.0 - q[0]
                };
                let cfm = if mode == 1 { 0.04 } else { 0.01 };
                let expected = cfm * force * DT / 0.2;
                assert!(
                    (penetration - expected).abs() < 2e-5,
                    "mode={mode} warm={warm_start} force={force}: {penetration} != {expected}"
                );
                displacement.push(penetration);
            }
            assert!((displacement[1] / displacement[0] - 2.0).abs() < 0.02);
        }
    }
    Ok(())
}

#[test]
fn point_and_fixed_loop_rows_have_load_dependent_response()
-> Result<(), Box<dyn core::error::Error>> {
    let context = GpuContactDevice::new_with_backends(wgpu::Backends::VULKAN)?;
    for mode in 0..3 {
        let art = tree(
            1,
            if mode == 2 {
                JointKind::Revolute
            } else {
                JointKind::Prismatic
            },
            None,
        )?;
        for force in [5.0, 10.0] {
            let mut input = GpuArticulatedDynamicsInput::new(
                &art,
                Isometry3::identity(),
                state(vec![0.0]),
                Vector3::zeros(),
            );
            input.joints[0].base_force = force;
            input.contact_warm_start = true;
            if mode == 0 {
                input.link_point_constraints.push(LinkPointConstraint {
                    link_a: 1,
                    point_a: [0.0; 3],
                    link_b: None,
                    point_b: [0.0; 3],
                });
            } else {
                input.link_fixed_constraints.push(LinkFixedConstraint {
                    link_a: 1,
                    frame_a: Isometry3::identity(),
                    link_b: None,
                    frame_b: Isometry3::identity(),
                });
            }
            let batch = GpuArticulatedDynamicsBatch::new(&context, &[input], DT)?;
            let initial = batch.readback()?;
            batch.submit_steps(STEPS)?;
            assert!(batch.readback()?[0].positions[0].abs() < 2e-5);
            batch.reset(&initial)?;
            batch.update_contact_policy(GpuContactPolicy {
                loop_constraint: soft(0.01),
                fixed_loop_constraint: soft(0.01),
                ..Default::default()
            })?;
            batch.submit_steps(STEPS)?;
            let actual = batch.readback()?[0].positions[0];
            let expected = 0.01 * force * DT / 0.2;
            assert!(
                (actual - expected).abs() < 2e-5,
                "loop mode={mode} load={force}: {actual} != {expected}"
            );
        }
    }
    Ok(())
}

#[test]
fn compliant_stops_allow_load_dependent_overtravel_without_hard_clamping()
-> Result<(), Box<dyn core::error::Error>> {
    let context = GpuContactDevice::new_with_backends(wgpu::Backends::VULKAN)?;
    let art = tree(1, JointKind::Prismatic, Some((0.0, 1.0)))?;
    for force in [5.0, 10.0, -5.0, -10.0] {
        let systems = vec![
            GpuMassAssemblySystem {
                links: Vec::new(),
                armature: DVector::from_element(1, 1.0),
                force: DVector::from_element(1, force),
            };
            2
        ];
        let mass =
            GpuArticulatedMassAssemblyBatch::new(context.device(), context.queue(), &systems)?;
        let boundary = if force > 0.0 { 1.0 } else { 0.0 };
        let state = GpuGeneralizedStateBatch::from_assembly_batch(
            &mass,
            &[state(vec![boundary]), state(vec![boundary])],
        )?;
        let mut limits =
            GpuArticulatedJointLimitBatch::new(&state, &mass, &[&art, &art], &[None; 2], DT)?;
        limits.update_coordinate_compliance(&[vec![soft(0.0)], vec![soft(0.01)]])?;
        for invalid in [
            soft(-1.0),
            soft(f32::INFINITY),
            GpuConstraintCompliance {
                erp: 1.1,
                cfm: 0.01,
            },
        ] {
            assert!(
                limits
                    .update_coordinate_compliance(&[vec![soft(0.02)], vec![invalid],])
                    .is_err()
            );
        }
        let mut encoder = context.device().create_command_encoder(&Default::default());
        for _ in 0..STEPS {
            mass.encode_with_inverse(&mut encoder);
            limits.encode(&mut encoder);
        }
        let _ = context.queue().submit(Some(encoder.finish()));
        let output = state.readback()?;
        assert!((output[0].positions[0] - boundary).abs() < 1e-7);
        let overtravel = output[1].positions[0] - boundary;
        let expected = 0.01 * force * DT / 0.2;
        assert!(
            (overtravel - expected).abs() < 2e-5,
            "stop force={force}: {overtravel} != {expected}"
        );
        context
            .queue()
            .write_buffer(state.velocity_buffer(), 4, bytemuck::bytes_of(&f32::NAN));
        let mut encoder = context.device().create_command_encoder(&Default::default());
        mass.encode_with_inverse(&mut encoder);
        limits.encode(&mut encoder);
        let _ = context.queue().submit(Some(encoder.finish()));
        assert!(
            state.readback().is_err(),
            "nonfinite stop input must propagate a state fault"
        );
        context
            .queue()
            .write_buffer(state.velocity_buffer(), 4, bytemuck::bytes_of(&0.0f32));
        assert!(
            state.readback().is_err(),
            "repairing velocity must not erase the persistent fault"
        );
    }
    Ok(())
}

#[test]
fn capsule_face_normal_block_preserves_static_compliance() -> Result<(), Box<dyn core::error::Error>>
{
    let context = GpuContactDevice::new_with_backends(wgpu::Backends::VULKAN)?;
    let art = Articulation::new(
        vec![
            LinkSpec {
                mass: 0.0,
                center_of_mass: Vector3::zeros(),
                inertia: Matrix3::zeros(),
            },
            LinkSpec {
                mass: 0.0,
                center_of_mass: Vector3::zeros(),
                inertia: Matrix3::zeros(),
            },
            LinkSpec {
                mass: 1.0,
                center_of_mass: Vector3::zeros(),
                inertia: Matrix3::identity(),
            },
        ],
        vec![
            JointSpec {
                parent: 0,
                child: 1,
                kind: JointKind::Prismatic,
                origin: Isometry3::identity(),
                axis: Vector3::z(),
                limits: None,
            },
            JointSpec {
                parent: 1,
                child: 2,
                kind: JointKind::Revolute,
                origin: Isometry3::identity(),
                axis: Vector3::y(),
                limits: None,
            },
        ],
        0,
    )?;
    for force in [5.0, 10.0] {
        let mut input = GpuArticulatedDynamicsInput::new(
            &art,
            Isometry3::identity(),
            state(vec![0.0, 0.0]),
            Vector3::zeros(),
        );
        input.joints[0].base_force = force;
        input.contact_warm_start = true;
        input
            .static_convex_capsule_pairs
            .push(GpuArticulatedStaticConvexCapsulePair {
                convex_link: 2,
                convex_local_pose: Isometry3::identity(),
                vertices: [-0.5, 0.5]
                    .into_iter()
                    .flat_map(|x| {
                        [-0.5, 0.5].into_iter().flat_map(move |y| {
                            [-0.5, 0.5].into_iter().map(move |z| Vector3::new(x, y, z))
                        })
                    })
                    .collect(),
                face_normals: vec![
                    Vector3::x(),
                    -Vector3::x(),
                    Vector3::y(),
                    -Vector3::y(),
                    Vector3::z(),
                    -Vector3::z(),
                ],
                edge_directions: vec![Vector3::x(), Vector3::y(), Vector3::z()],
                static_a: Vector3::new(-0.4, 0.0, 0.6),
                static_b: Vector3::new(0.4, 0.0, 0.6),
                static_radius: 0.1,
                restitution: 0.0,
                friction: 0.0,
            });
        let batch = GpuArticulatedDynamicsBatch::new(&context, &[input], DT)?;
        let initial = batch.readback()?;
        batch.submit_steps(STEPS)?;
        assert!(batch.readback()?[0].positions[0].abs() < 2e-5);
        batch.reset(&initial)?;
        batch.update_contact_policy(GpuContactPolicy {
            dynamic_normal: soft(0.04),
            static_normal: soft(0.01),
            ..Default::default()
        })?;
        batch.submit_steps(STEPS)?;
        let output = batch.readback()?;
        let expected = 0.01 * force * DT / (2.0 * 0.2);
        assert!(
            (output[0].positions[0] - expected).abs() < 2e-5,
            "capsule block load={force}: {} != {expected}",
            output[0].positions[0]
        );
        assert!(
            output[0].positions[1].abs() < 2e-5,
            "symmetric block must not tilt"
        );
    }
    Ok(())
}

#[test]
fn soft_stop_impulse_uses_the_full_inverse_mass_column() -> Result<(), Box<dyn core::error::Error>>
{
    let context = GpuContactDevice::new_with_backends(wgpu::Backends::VULKAN)?;
    let art = Articulation::new(
        vec![
            LinkSpec {
                mass: 0.0,
                center_of_mass: Vector3::zeros(),
                inertia: Matrix3::zeros()
            };
            3
        ],
        vec![
            JointSpec {
                parent: 0,
                child: 1,
                kind: JointKind::Prismatic,
                origin: Isometry3::identity(),
                axis: Vector3::z(),
                limits: Some((0.0, 1.0)),
            },
            JointSpec {
                parent: 0,
                child: 2,
                kind: JointKind::Prismatic,
                origin: Isometry3::identity(),
                axis: Vector3::z(),
                limits: None,
            },
        ],
        0,
    )?;
    // M = [[2,1],[1,1]], so its inverse first column is [1,-1].
    let mass = GpuArticulatedMassAssemblyBatch::new(
        context.device(),
        context.queue(),
        &[GpuMassAssemblySystem {
            links: vec![GpuMassLink {
                mass: 1.0,
                inertia_world: Matrix3::zeros(),
                linear_jacobian: DMatrix::from_row_slice(3, 2, &[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
                angular_jacobian: DMatrix::zeros(3, 2),
            }],
            armature: DVector::from_vec(vec![1.0, 0.0]),
            force: DVector::from_vec(vec![10.0, 0.0]),
        }],
    )?;
    let state = GpuGeneralizedStateBatch::from_assembly_batch(&mass, &[state(vec![1.0, 0.0])])?;
    let mut limits = GpuArticulatedJointLimitBatch::new(&state, &mass, &[&art], &[None], DT)?;
    assert!(
        limits
            .update_coordinate_compliance(&[vec![soft(0.01), soft(0.01)]])
            .is_err()
    );
    limits.update_coordinate_compliance(&[vec![soft(0.01), soft(0.0)]])?;
    let mut encoder = context.device().create_command_encoder(&Default::default());
    for _ in 0..STEPS {
        mass.encode_with_inverse(&mut encoder);
        limits.encode(&mut encoder);
    }
    let _ = context.queue().submit(Some(encoder.finish()));
    let output = state.readback()?;
    assert!((output[0].positions[0] - 1.0025).abs() < 2e-5);
    assert!((output[0].positions[1] + 0.0025).abs() < 2e-5);
    Ok(())
}
