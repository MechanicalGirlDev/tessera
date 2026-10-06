//! Analytic soft bias, speculative rows and unbiased relaxation on GPU backends.
#![cfg(feature = "gpu-contact")]
use tessera_physics::{
    gpu_broad_phase::GpuPair,
    gpu_contact_pipeline::GpuContactDevice,
    gpu_rigid_contact_transport::GpuRigidContactTransport,
    gpu_rigid_sphere_contact::GpuRigidSphereContacts,
    gpu_rigid_sphere_solver::{
        GpuRigidSphereImpulseCache, GpuRigidSphereSolver, GpuRigidTemporalSolveParams,
    },
    gpu_rigid_state::{GpuRigidBodyState, GpuRigidStateSession},
};

fn solve_pair_with_static_frequency(
    context: &GpuContactDevice,
    fixed_a: bool,
    static_frequency: f32,
) -> Result<f32, Box<dyn core::error::Error>> {
    let body = |z: f32, inverse_mass: f32| GpuRigidBodyState {
        position_inverse_mass: [0.0, 0.0, z, inverse_mass],
        orientation: [0.0, 0.0, 0.0, 1.0],
        linear_velocity: [0.0; 4],
        angular_velocity: [0.0; 4],
        inverse_inertia_sleep: [inverse_mass, inverse_mass, inverse_mass, 0.0],
    };
    let session = GpuRigidStateSession::new(
        context.device(),
        context.queue(),
        &[body(0.0, if fixed_a { 0.0 } else { 1.0 }), body(1.9, 1.0)],
    )?;
    let contacts = GpuRigidSphereContacts::new(
        context.device(),
        &session,
        &[1.0, 1.0],
        &[GpuPair { a: 0, b: 1 }],
        None,
    )?;
    let settings = GpuRigidTemporalSolveParams {
        friction: 0.0,
        restitution: 0.0,
        normal_frequency: 20.0,
        damping_ratio: 1.0,
        static_normal_frequency: static_frequency,
        static_damping_ratio: 1.0,
        max_corrective_velocity: 3.0,
        iterations: 1,
    };
    let solver = GpuRigidSphereSolver::new(context.device());
    let mut cache = GpuRigidSphereImpulseCache::default();
    let step = solver
        .prepare_temporal_cached(context.device(), &contacts, 0.01, settings, &mut cache)?
        .ok_or("missing pair solve")?;
    let mut encoder = context.device().create_command_encoder(&Default::default());
    contacts.encode(&mut encoder);
    step.encode_capture_anchors(&mut encoder);
    step.encode_bias(&mut encoder);
    let _ = context.queue().submit(Some(encoder.finish()));
    Ok(session.readback(context.device(), context.queue())?[1].linear_velocity[2])
}

fn solve_ground_with_static_frequency(
    context: &GpuContactDevice,
    static_frequency: f32,
) -> Result<f32, Box<dyn core::error::Error>> {
    let body = GpuRigidBodyState {
        position_inverse_mass: [0.0, 0.0, 0.9, 1.0],
        orientation: [0.0, 0.0, 0.0, 1.0],
        linear_velocity: [0.0; 4],
        angular_velocity: [0.0; 4],
        inverse_inertia_sleep: [1.0, 1.0, 1.0, 0.0],
    };
    let session = GpuRigidStateSession::new(context.device(), context.queue(), &[body])?;
    let contacts =
        GpuRigidSphereContacts::new(context.device(), &session, &[1.0], &[], Some(10.0))?;
    let settings = GpuRigidTemporalSolveParams {
        normal_frequency: 20.0,
        static_normal_frequency: static_frequency,
        ..GpuRigidTemporalSolveParams::default()
    };
    let solver = GpuRigidSphereSolver::new(context.device());
    let mut cache = GpuRigidSphereImpulseCache::default();
    let step = solver
        .prepare_temporal_cached(context.device(), &contacts, 0.01, settings, &mut cache)?
        .ok_or("missing ground solve")?;
    let mut encoder = context.device().create_command_encoder(&Default::default());
    contacts.encode(&mut encoder);
    step.encode_capture_anchors(&mut encoder);
    step.encode_bias(&mut encoder);
    let _ = context.queue().submit(Some(encoder.finish()));
    Ok(session.readback(context.device(), context.queue())?[0].linear_velocity[2])
}

#[test]
fn static_contact_softness_only_changes_fixed_body_pairs() -> Result<(), Box<dyn core::error::Error>>
{
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        tested += 1;
        let fixed_soft = solve_pair_with_static_frequency(&context, true, 20.0)?;
        let fixed_stiff = solve_pair_with_static_frequency(&context, true, 80.0)?;
        let dynamic_soft = solve_pair_with_static_frequency(&context, false, 20.0)?;
        let dynamic_stiff = solve_pair_with_static_frequency(&context, false, 80.0)?;
        let ground_soft = solve_ground_with_static_frequency(&context, 20.0)?;
        let ground_stiff = solve_ground_with_static_frequency(&context, 80.0)?;
        assert!(
            fixed_stiff > fixed_soft + 0.25,
            "{backend:?}: {fixed_soft} {fixed_stiff}"
        );
        assert!((dynamic_soft - dynamic_stiff).abs() < 1e-5, "{backend:?}");
        assert!(
            ground_stiff > ground_soft + 0.25,
            "{backend:?}: {ground_soft} {ground_stiff}"
        );
    }
    assert!(tested > 0, "no GPU backend available");
    Ok(())
}
#[test]
fn restitution_target_is_captured_once_per_frame() -> Result<(), Box<dyn core::error::Error>> {
    let base = GpuRigidBodyState {
        position_inverse_mass: [0.0, 0.0, 1.0, 1.0],
        orientation: [0.0, 0.0, 0.0, 1.0],
        linear_velocity: [0.0, 0.0, -10.0, 0.0],
        angular_velocity: [0.0; 4],
        inverse_inertia_sleep: [1.0, 1.0, 1.0, 0.0],
    };
    let settings = GpuRigidTemporalSolveParams {
        friction: 0.0,
        restitution: 0.5,
        normal_frequency: 20.0,
        damping_ratio: 1.0,
        static_normal_frequency: 20.0,
        static_damping_ratio: 1.0,
        max_corrective_velocity: 3.0,
        iterations: 1,
    };
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        tested += 1;
        let session = GpuRigidStateSession::new(context.device(), context.queue(), &[base])?;
        let contacts =
            GpuRigidSphereContacts::new(context.device(), &session, &[1.0], &[], Some(10.0))?;
        let solver = GpuRigidSphereSolver::new(context.device());
        let mut cache = GpuRigidSphereImpulseCache::default();
        let step = solver
            .prepare_temporal_cached(context.device(), &contacts, 0.01, settings, &mut cache)?
            .ok_or("missing ground solve")?;
        let mut capture = context.device().create_command_encoder(&Default::default());
        contacts.encode(&mut capture);
        step.encode_capture_anchors(&mut capture);
        let _ = context.queue().submit(Some(capture.finish()));
        // Later substeps use the captured target of +5, even when closing speed changes.
        for velocity in [-10.0, -20.0, -2.0] {
            let current = GpuRigidBodyState {
                linear_velocity: [0.0, 0.0, velocity, 0.0],
                ..base
            };
            session.write_body(context.queue(), 0, current)?;
            let step = solver
                .prepare_temporal_cached(context.device(), &contacts, 0.01, settings, &mut cache)?
                .ok_or("missing cached solve")?;
            let mut encoder = context.device().create_command_encoder(&Default::default());
            step.encode_bias(&mut encoder);
            step.encode_relax(&mut encoder);
            let _ = context.queue().submit(Some(encoder.finish()));
            let actual = session.readback(context.device(), context.queue())?[0];
            let expected = 5.0;
            assert!(
                (actual.linear_velocity[2] - expected).abs() < 3e-5,
                "{backend:?}: velocity={velocity}: {actual:?}"
            );
        }
        // A new frame refreshes the target from its new incoming velocity.
        session.write_body(
            context.queue(),
            0,
            GpuRigidBodyState {
                linear_velocity: [0.0, 0.0, -4.0, 0.0],
                ..base
            },
        )?;
        let step = solver
            .prepare_temporal_cached(context.device(), &contacts, 0.01, settings, &mut cache)?
            .ok_or("missing next frame solve")?;
        let mut encoder = context.device().create_command_encoder(&Default::default());
        contacts.encode(&mut encoder);
        step.encode_capture_anchors(&mut encoder);
        step.encode_bias(&mut encoder);
        let _ = context.queue().submit(Some(encoder.finish()));
        let actual = session.readback(context.device(), context.queue())?[0];
        assert!((actual.linear_velocity[2] - (-4.0 + 0.803_629_34 * 6.0)).abs() < 3e-5);
    }
    assert!(tested > 0, "no GPU backend available");
    Ok(())
}
#[test]
fn rotated_pair_uses_individual_anchors_through_warm_start()
-> Result<(), Box<dyn core::error::Error>> {
    let fixed = GpuRigidBodyState {
        position_inverse_mass: [0.0; 4],
        orientation: [0.0, 0.0, 0.0, 1.0],
        linear_velocity: [0.0; 4],
        angular_velocity: [0.0; 4],
        inverse_inertia_sleep: [0.0; 4],
    };
    let dynamic = GpuRigidBodyState {
        position_inverse_mass: [1.5, 0.0, 0.0, 1.0],
        inverse_inertia_sleep: [1.0, 1.0, 1.0, 0.0],
        ..fixed
    };
    let settings = GpuRigidTemporalSolveParams {
        friction: 0.0,
        restitution: 0.0,
        normal_frequency: 20.0,
        damping_ratio: 1.0,
        static_normal_frequency: 20.0,
        static_damping_ratio: 1.0,
        max_corrective_velocity: 3.0,
        iterations: 1,
    };
    // Rotation opens a 0.25 gap: rigid target -25 versus incoming -40,
    // with individual anchor inverse effective mass 1 + 0.75^2.
    let expected_impulse = 15.0 / 1.5625;
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        tested += 1;
        for swapped in [false, true] {
            let bodies = if swapped {
                [dynamic, fixed]
            } else {
                [fixed, dynamic]
            };
            let moving = if swapped { 0 } else { 1 };
            let session = GpuRigidStateSession::new(context.device(), context.queue(), &bodies)?;
            let contacts = GpuRigidSphereContacts::new(
                context.device(),
                &session,
                &[1.0, 1.0],
                &[GpuPair { a: 0, b: 1 }],
                None,
            )?;
            let transport = GpuRigidContactTransport::new(context.device(), &contacts)?;
            let solver = GpuRigidSphereSolver::new(context.device());
            let mut cache = GpuRigidSphereImpulseCache::default();
            let step = solver
                .prepare_temporal_cached(context.device(), &contacts, 0.01, settings, &mut cache)?
                .ok_or("missing pair solve")?;
            let mut capture = context.device().create_command_encoder(&Default::default());
            contacts.encode(&mut capture);
            transport.encode_capture(&mut capture);
            step.encode_capture_anchors(&mut capture);
            let _ = context.queue().submit(Some(capture.finish()));
            let rotated = GpuRigidBodyState {
                orientation: [
                    0.0,
                    0.0,
                    core::f32::consts::FRAC_1_SQRT_2,
                    core::f32::consts::FRAC_1_SQRT_2,
                ],
                linear_velocity: [-40.0, 0.0, 0.0, 0.0],
                ..dynamic
            };
            // Reset velocity between solves to check both initial solve and cached warm start.
            for _ in 0..2 {
                session.write_body(context.queue(), moving, rotated)?;
                let step = solver
                    .prepare_temporal_cached(
                        context.device(),
                        &contacts,
                        0.01,
                        settings,
                        &mut cache,
                    )?
                    .ok_or("missing cached pair solve")?;
                let mut encoder = context.device().create_command_encoder(&Default::default());
                transport.encode_refresh(&mut encoder);
                step.encode_bias(&mut encoder);
                let _ = context.queue().submit(Some(encoder.finish()));
                let actual = session.readback(context.device(), context.queue())?;
                assert_eq!(
                    actual[1 - moving].position_inverse_mass,
                    fixed.position_inverse_mass
                );
                assert_eq!(actual[1 - moving].linear_velocity, fixed.linear_velocity);
                assert_eq!(actual[1 - moving].angular_velocity, fixed.angular_velocity);
                assert!(
                    (actual[moving].linear_velocity[0] + 40.0 - expected_impulse).abs() < 3e-5,
                    "{backend:?}: swapped={swapped}: {:?}",
                    actual[moving]
                );
                assert!(
                    (actual[moving].angular_velocity[2] - 0.75 * expected_impulse).abs() < 3e-5
                );
                let history = cache.readback(context.device(), context.queue())?;
                let impulse = history
                    .contacts
                    .iter()
                    .map(|point| point.impulse_on_body_b()[0])
                    .sum::<f32>();
                let sign = if swapped { -1.0 } else { 1.0 };
                assert!((sign * impulse - expected_impulse).abs() < 3e-5);
            }
        }
    }
    assert!(tested > 0, "no GPU backend available");
    Ok(())
}
#[test]
fn soft_bias_and_relax_match_analytic_impulses_and_preserve_speculative_closing_speed()
-> Result<(), Box<dyn core::error::Error>> {
    let base = GpuRigidBodyState {
        position_inverse_mass: [0.0, 0.0, 0.9, 1.0],
        orientation: [0.0, 0.0, 0.0, 1.0],
        linear_velocity: [0.0; 4],
        angular_velocity: [0.0; 4],
        inverse_inertia_sleep: [1.0, 1.0, 1.0, 0.0],
    };
    let settings = GpuRigidTemporalSolveParams {
        friction: 0.0,
        restitution: 0.0,
        normal_frequency: 20.0,
        damping_ratio: 1.0,
        static_normal_frequency: 20.0,
        static_damping_ratio: 1.0,
        max_corrective_velocity: 3.0,
        iterations: 1,
    };
    // Reference value for omega=2*pi*20, h=.01, damping=1: a2/(1+a2).
    let mass_scale = 0.803_629_34;
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        tested += 1;
        eprintln!("temporal solve backend: {backend:?}");
        let session = GpuRigidStateSession::new(context.device(), context.queue(), &[base])?;
        let contacts =
            GpuRigidSphereContacts::new(context.device(), &session, &[1.0], &[], Some(10.0))?;
        let anchors = GpuRigidContactTransport::new(context.device(), &contacts)?;
        let solver = GpuRigidSphereSolver::new(context.device());
        for (z, velocity, expected_bias_velocity, expected_relax_velocity) in [
            (0.9, 0.0, mass_scale * 3.0, 0.0),
            (1.05, -10.0, -5.0, 0.0),
            (1.05, -1.0, -1.0, -1.0),
        ] {
            session.write_body(context.queue(), 0, base)?;
            let mut capture = context.device().create_command_encoder(&Default::default());
            contacts.encode(&mut capture);
            anchors.encode_capture(&mut capture);
            let _ = context.queue().submit(Some(capture.finish()));
            let mut current = base;
            current.position_inverse_mass[2] = z;
            current.linear_velocity[2] = velocity;
            session.write_body(context.queue(), 0, current)?;
            let mut cache = GpuRigidSphereImpulseCache::default();
            let step = solver
                .prepare_temporal_cached(context.device(), &contacts, 0.01, settings, &mut cache)?
                .ok_or("missing temporal ground solve")?;
            let mut bias = context.device().create_command_encoder(&Default::default());
            anchors.encode_refresh(&mut bias);
            step.encode_bias(&mut bias);
            let _ = context.queue().submit(Some(bias.finish()));
            let biased = session.readback(context.device(), context.queue())?[0];
            assert!(
                (biased.linear_velocity[2] - expected_bias_velocity).abs() < 2e-5,
                "{backend:?}: z={z}: initial velocity={velocity}: {biased:?}"
            );
            assert_eq!(biased.position_inverse_mass, current.position_inverse_mass);
            let mut relax = context.device().create_command_encoder(&Default::default());
            session.encode_position_step(context.device(), &mut relax, 0.01)?;
            anchors.encode_refresh(&mut relax);
            step.encode_relax(&mut relax);
            let _ = context.queue().submit(Some(relax.finish()));
            let actual = session.readback(context.device(), context.queue())?[0];
            assert!(
                (actual.position_inverse_mass[2] - z - 0.01 * expected_bias_velocity).abs() < 2e-5
            );
            assert!(
                (actual.linear_velocity[2] - expected_relax_velocity).abs() < 2e-5,
                "{backend:?}: z={z}: initial velocity={velocity}: {actual:?}"
            );
            let history = cache.readback(context.device(), context.queue())?;
            let impulse = history
                .contacts
                .iter()
                .map(|point| point.impulse_on_body_b()[2])
                .sum::<f32>();
            assert!((impulse - (expected_relax_velocity - velocity)).abs() < 2e-5);
        }
        // Guard an underflowed spring mass scale against an infinite separation target.
        session.write_body(context.queue(), 0, base)?;
        let mut capture = context.device().create_command_encoder(&Default::default());
        contacts.encode(&mut capture);
        anchors.encode_capture(&mut capture);
        let _ = context.queue().submit(Some(capture.finish()));
        let mut separated = base;
        separated.position_inverse_mass[2] = 1.05;
        separated.linear_velocity[2] = -1.0;
        session.write_body(context.queue(), 0, separated)?;
        let tiny = GpuRigidTemporalSolveParams {
            normal_frequency: f32::MIN_POSITIVE,
            ..settings
        };
        let step = solver
            .prepare_temporal_cached(
                context.device(),
                &contacts,
                f32::from_bits(1),
                tiny,
                &mut GpuRigidSphereImpulseCache::default(),
            )?
            .ok_or("missing tiny temporal solve")?;
        let mut encoder = context.device().create_command_encoder(&Default::default());
        anchors.encode_refresh(&mut encoder);
        step.encode_bias(&mut encoder);
        step.encode_relax(&mut encoder);
        let _ = context.queue().submit(Some(encoder.finish()));
        let actual = session.readback(context.device(), context.queue())?[0];
        assert_eq!(actual.linear_velocity, separated.linear_velocity);
        for invalid in [
            GpuRigidTemporalSolveParams {
                normal_frequency: 0.0,
                ..settings
            },
            GpuRigidTemporalSolveParams {
                damping_ratio: f32::NAN,
                ..settings
            },
            GpuRigidTemporalSolveParams {
                static_normal_frequency: 0.0,
                ..settings
            },
            GpuRigidTemporalSolveParams {
                static_damping_ratio: f32::NAN,
                ..settings
            },
            GpuRigidTemporalSolveParams {
                max_corrective_velocity: 0.0,
                ..settings
            },
            GpuRigidTemporalSolveParams {
                normal_frequency: f32::MAX,
                ..settings
            },
        ] {
            assert!(
                solver
                    .prepare_temporal_cached(
                        context.device(),
                        &contacts,
                        0.01,
                        invalid,
                        &mut GpuRigidSphereImpulseCache::default()
                    )
                    .is_err()
            );
        }
    }
    assert!(tested > 0, "no GPU backend available");
    Ok(())
}
