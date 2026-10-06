//! Analytic pose changes of captured contact anchors on GPU backends.
#![cfg(feature = "gpu-contact")]
use tessera_physics::{
    gpu_broad_phase::GpuPair,
    gpu_contact_pipeline::GpuContactDevice,
    gpu_rigid_contact_transport::GpuRigidContactTransport,
    gpu_rigid_sphere_contact::GpuRigidSphereContacts,
    gpu_rigid_state::{GpuRigidBodyState, GpuRigidStateSession},
};
#[test]
fn temporal_anchors_preserve_rotation_translation_and_signed_separation()
-> Result<(), Box<dyn core::error::Error>> {
    let base = GpuRigidBodyState {
        position_inverse_mass: [0.0, 0.0, 1.0, 0.0],
        orientation: [0.0, 0.0, 0.0, 1.0],
        linear_velocity: [0.0; 4],
        angular_velocity: [0.0; 4],
        inverse_inertia_sleep: [0.0; 4],
    };
    let mut initial = [base; 3];
    initial[1].position_inverse_mass[0] = 1.5;
    initial[2].position_inverse_mass[0] = 10.0;
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        tested += 1;
        eprintln!("contact transport backend: {backend:?}");
        let session = GpuRigidStateSession::new(context.device(), context.queue(), &initial)?;
        let contacts = GpuRigidSphereContacts::new(
            context.device(),
            &session,
            &[1.0; 3],
            &[GpuPair { a: 0, b: 1 }, GpuPair { a: 0, b: 2 }],
            Some(20.0),
        )?;
        let transport = GpuRigidContactTransport::new(context.device(), &contacts)?;
        let mut encoder = context.device().create_command_encoder(&Default::default());
        contacts.encode(&mut encoder);
        transport.encode_capture(&mut encoder);
        transport.encode_refresh(&mut encoder);
        let _ = context.queue().submit(Some(encoder.finish()));
        let original = contacts.readback(context.device(), context.queue())?;
        assert!(original.pairs[0].1.is_contact());
        assert!((original.pairs[0].1.depth_hit[0] - 0.5).abs() < 1e-5);
        assert!(!original.pairs[1].1.is_contact());
        for (x, z, rotated, expected_point, expected_depth, ground_depth) in [
            (1.75, 1.25, false, [0.875, 0.0, 1.125], 0.25, -0.25),
            (2.25, 1.0, false, [1.125, 0.0, 1.0], -0.25, 0.0),
            (1.5, 1.0, true, [1.125, -0.375, 1.0], -0.25, 0.0),
        ] {
            let mut moved = initial[1];
            moved.position_inverse_mass = [x, 0.0, z, 0.0];
            if rotated {
                let q = core::f32::consts::FRAC_1_SQRT_2;
                moved.orientation = [0.0, 0.0, q, q];
            }
            session.write_body(context.queue(), 1, moved)?;
            let mut third = initial[2];
            third.position_inverse_mass[0] = 0.5;
            session.write_body(context.queue(), 2, third)?;
            let mut encoder = context.device().create_command_encoder(&Default::default());
            transport.encode_refresh(&mut encoder);
            // Repeated refresh must use the original anchors, not updated contact points.
            transport.encode_refresh(&mut encoder);
            let _ = context.queue().submit(Some(encoder.finish()));
            let refreshed = contacts.readback(context.device(), context.queue())?;
            let contact = refreshed.pairs[0].1;
            assert!(contact.is_contact());
            assert!(
                (contact.depth_hit[0] - expected_depth).abs() < 1e-5,
                "{contact:?}"
            );
            for (actual, expected) in contact.point[..3].iter().zip(expected_point) {
                assert!((actual - expected).abs() < 1e-5, "{contact:?}");
            }
            assert!(
                (contact.normal[0] - 1.0).abs() < 1e-5
                    && contact.normal[1].abs() < 1e-5
                    && contact.normal[2].abs() < 1e-5
            );
            assert!(
                !refreshed.pairs[1].1.is_contact(),
                "inactive slot became active"
            );
            assert!((refreshed.ground[1].depth_hit[0] - ground_depth).abs() < 1e-5);
            assert!(refreshed.ground[1].normal[2] > 0.99);
        }
        let mut encoder = context.device().create_command_encoder(&Default::default());
        contacts.encode(&mut encoder);
        transport.encode_capture(&mut encoder);
        transport.encode_refresh(&mut encoder);
        let _ = context.queue().submit(Some(encoder.finish()));
        let recaptured = contacts.readback(context.device(), context.queue())?;
        assert!(
            recaptured.pairs[1].1.is_contact(),
            "new narrow phase did not recapture contact"
        );
    }
    assert!(tested > 0, "no GPU backend available");
    Ok(())
}
