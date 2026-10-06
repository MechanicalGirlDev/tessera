//! Friction belongs to the stabilization sweep of the temporal solver.
#![cfg(feature = "gpu-contact")]
use tessera_physics::{
    gpu_contact_pipeline::GpuContactDevice,
    gpu_rigid_sphere_contact::GpuRigidSphereContacts,
    gpu_rigid_sphere_solver::{
        GpuRigidSphereImpulseCache, GpuRigidSphereSolver, GpuRigidTemporalSolveParams,
    },
    gpu_rigid_state::{GpuRigidBodyState, GpuRigidStateSession},
};

#[test]
fn bias_preserves_tangent_velocity_and_relax_enforces_coulomb_friction()
-> Result<(), Box<dyn core::error::Error>> {
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        tested += 1;
        let body = GpuRigidBodyState {
            position_inverse_mass: [0.0, 0.0, 1.0, 1.0],
            orientation: [0.0, 0.0, 0.0, 1.0],
            linear_velocity: [1.0, 0.0, -2.0, 0.0],
            angular_velocity: [0.0; 4],
            inverse_inertia_sleep: [1.0, 1.0, 1.0, 0.0],
        };
        let session = GpuRigidStateSession::new(context.device(), context.queue(), &[body])?;
        let contacts =
            GpuRigidSphereContacts::new(context.device(), &session, &[1.0], &[], Some(10.0))?;
        let solver = GpuRigidSphereSolver::new(context.device());
        let mut cache = GpuRigidSphereImpulseCache::default();
        let step = solver
            .prepare_temporal_cached(
                context.device(),
                &contacts,
                0.01,
                GpuRigidTemporalSolveParams {
                    friction: 1.0,
                    iterations: 1,
                    ..Default::default()
                },
                &mut cache,
            )?
            .ok_or("missing solve")?;
        let mut encoder = context.device().create_command_encoder(&Default::default());
        contacts.encode(&mut encoder);
        step.encode_capture_anchors(&mut encoder);
        step.encode_bias(&mut encoder);
        let _ = context.queue().submit(Some(encoder.finish()));
        let biased = session.readback(context.device(), context.queue())?[0];
        assert!(
            (biased.linear_velocity[0] - 1.0).abs() < 1e-6,
            "{backend:?}: {biased:?}"
        );
        assert!(biased.angular_velocity[..3].iter().all(|v| v.abs() < 1e-6));
        let mut encoder = context.device().create_command_encoder(&Default::default());
        step.encode_relax(&mut encoder);
        let _ = context.queue().submit(Some(encoder.finish()));
        let relaxed = session.readback(context.device(), context.queue())?[0];
        // Effective tangent inverse mass is 1 + r^2 I^-1 = 2.
        assert!(
            (relaxed.linear_velocity[0] - 0.5).abs() < 1e-5,
            "{backend:?}: {relaxed:?}"
        );
        assert!((relaxed.angular_velocity[1] - 0.5).abs() < 1e-5);
        assert!(relaxed.linear_velocity[2].abs() < 1e-5);
        let history = cache.readback(context.device(), context.queue())?;
        let impulse = history.contacts[0].impulse_on_body_b();
        assert!((impulse[2] - 2.0).abs() < 1e-5);
        assert!((impulse[0] + 0.5).abs() < 1e-5);
        // An upward perturbation reduces normal support while contact tangent speed
        // is zero. The cached tangent impulse must still obey the smaller cone.
        let mut current = relaxed;
        for (upward, normal, tangent) in [(1.75, 0.25, 0.25), (0.25, 0.0, 0.0)] {
            current.linear_velocity[2] = upward;
            session.write_body(context.queue(), 0, current)?;
            let mut encoder = context.device().create_command_encoder(&Default::default());
            step.encode_relax(&mut encoder);
            let _ = context.queue().submit(Some(encoder.finish()));
            current = session.readback(context.device(), context.queue())?[0];
            assert!(
                (current.linear_velocity[0] - (1.0 - tangent)).abs() < 1e-5,
                "{backend:?}: normal={normal}: {current:?}"
            );
            assert!((current.angular_velocity[1] - tangent).abs() < 1e-5);
            let history = cache.readback(context.device(), context.queue())?;
            if normal == 0.0 {
                // Readback only reports rows with a nonzero normal impulse.
                assert!(history.contacts.is_empty());
            } else {
                let impulse = history.contacts[0].impulse_on_body_b();
                assert!((impulse[2] - normal).abs() < 1e-5);
                assert!((impulse[0] + tangent).abs() < 1e-5);
            }
        }
    }
    assert!(tested > 0, "no GPU backend available");
    Ok(())
}
