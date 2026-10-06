//! Analytic soft joint impulses and immutable bias/relax coefficients.
#![cfg(feature = "gpu-contact")]
use tessera_physics::{
    gpu_contact_pipeline::GpuContactDevice,
    gpu_rigid_ball_joint::{
        GpuRigidBallJoint, GpuRigidBallJointSolver, GpuRigidFixedJoint, GpuRigidTemporalJointParams,
    },
    gpu_rigid_state::{GpuRigidBodyState, GpuRigidStateSession},
};

fn body(x: f32, inverse_mass: f32) -> GpuRigidBodyState {
    GpuRigidBodyState {
        position_inverse_mass: [x, 0.0, 0.0, inverse_mass],
        orientation: [0.0, 0.0, 0.0, 1.0],
        linear_velocity: [0.0; 4],
        angular_velocity: [0.0; 4],
        inverse_inertia_sleep: [inverse_mass, inverse_mass, inverse_mass, 0.0],
    }
}

#[test]
fn passive_soft_joint_bias_relax_and_pgs_switch_match_scalar_reference()
-> Result<(), Box<dyn core::error::Error>> {
    let mut linear = body(0.1, 1.0);
    linear.linear_velocity[0] = -2.0;
    let mut angular = body(10.0, 1.0);
    angular.orientation = [0.0, 0.0, (0.1_f32).sin(), (0.1_f32).cos()];
    angular.angular_velocity[2] = 1.0;
    let bodies = [body(0.0, 0.0), linear, body(10.0, 0.0), angular];
    let ball = GpuRigidBallJoint {
        body_a: 0,
        body_b: 1,
        local_anchor_a: [0.0; 3],
        local_anchor_b: [0.0; 3],
    };
    let fixed = GpuRigidFixedJoint {
        body_a: 2,
        body_b: 3,
        local_anchor_a: [0.0; 3],
        local_anchor_b: [0.0; 3],
        local_rotation_a: [0.0, 0.0, 0.0, 1.0],
        local_rotation_b: [0.0, 0.0, 0.0, 1.0],
    };
    let dt = 0.01;
    let settings = GpuRigidTemporalJointParams {
        max_angular_correction_speed: 0.25,
        ..GpuRigidTemporalJointParams::default()
    };
    let omega = 2.0 * core::f32::consts::PI * settings.frequency;
    let a2 = dt * omega * (2.0 * settings.damping_ratio + dt * omega);
    let mass_scale = a2 / (1.0 + a2);
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        tested += 1;
        eprintln!("temporal joint: {backend:?}");
        let session = GpuRigidStateSession::new(context.device(), context.queue(), &bodies)?;
        let mut solver = GpuRigidBallJointSolver::new_mixed(
            context.device(),
            session.state_buffer(),
            4,
            &[0; 4],
            &[ball],
            &[fixed],
        )?;
        let mut encoder = context.device().create_command_encoder(&Default::default());
        assert!(
            solver
                .prepare_temporal(
                    context.device(),
                    &mut encoder,
                    dt,
                    GpuRigidTemporalJointParams {
                        frequency: 0.0,
                        ..settings
                    }
                )
                .is_err()
        );
        let step = solver.prepare_temporal(context.device(), &mut encoder, dt, settings)?;
        step.encode_warm(&mut encoder);
        step.encode_bias(&mut encoder);
        let _ = context.queue().submit(Some(encoder.finish()));
        let biased = session.readback(context.device(), context.queue())?;
        assert!(
            (biased[1].linear_velocity[0] - (-2.0 - mass_scale)).abs() < 1e-5,
            "{backend:?}: {:?}",
            biased[1]
        );
        assert!(
            (biased[3].angular_velocity[2] - (1.0 - 1.25 * mass_scale)).abs() < 1e-5,
            "{backend:?}: {:?}",
            biased[3]
        );
        let mut encoder = context.device().create_command_encoder(&Default::default());
        step.encode_relax(&mut encoder);
        let _ = context.queue().submit(Some(encoder.finish()));
        let relaxed = session.readback(context.device(), context.queue())?;
        assert!(relaxed[1].linear_velocity[0].abs() < 1e-5);
        assert!(relaxed[3].angular_velocity[2].abs() < 1e-5);
        for (index, initial) in bodies.iter().enumerate() {
            session.write_body(context.queue(), index, *initial)?;
        }
        solver.clear_impulse_cache(context.queue());
        let mut encoder = context.device().create_command_encoder(&Default::default());
        // Both bindings coexist in one submission; queue writes cannot replace their coefficients.
        step.encode_warm(&mut encoder);
        step.encode_bias(&mut encoder);
        step.encode_relax(&mut encoder);
        let _ = context.queue().submit(Some(encoder.finish()));
        let combined = session.readback(context.device(), context.queue())?;
        for index in [1, 3] {
            for axis in 0..3 {
                assert!(
                    (combined[index].linear_velocity[axis] - relaxed[index].linear_velocity[axis])
                        .abs()
                        < 1e-5
                );
                assert!(
                    (combined[index].angular_velocity[axis]
                        - relaxed[index].angular_velocity[axis])
                        .abs()
                        < 1e-5
                );
            }
        }
        for index in [0, 2] {
            assert_eq!(
                combined[index].position_inverse_mass,
                bodies[index].position_inverse_mass
            );
        }
        let mut encoder = context.device().create_command_encoder(&Default::default());
        solver.encode(context.queue(), &mut encoder, dt, 1, 0.0)?;
        let _ = context.queue().submit(Some(encoder.finish()));
        let pgs = session.readback(context.device(), context.queue())?;
        assert!(pgs[1].linear_velocity[0].abs() < 1e-5);
        assert!(pgs[3].angular_velocity[2].abs() < 1e-5);
    }
    assert!(tested > 0, "no GPU backend available");
    Ok(())
}
