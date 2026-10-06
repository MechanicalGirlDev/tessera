//! GPU segment/hull clipping against a non-box convex polytope.

#![cfg(feature = "gpu-contact")]

use tessera_physics::{
    gpu_contact_pipeline::GpuContactDevice,
    gpu_rigid_shape::GpuRigidShape,
    gpu_rigid_sphere_world::{GpuRigidPrimitiveWorld, GpuRigidSphereWorldConfig},
    gpu_rigid_state::GpuRigidBodyState,
};

#[test]
fn polyline_clips_against_tetrahedron_faces() -> Result<(), Box<dyn core::error::Error>> {
    let line = GpuRigidShape::Polyline {
        vertices: vec![[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        segments: vec![[0, 1]],
    };
    let hull = GpuRigidShape::Convex {
        vertices: vec![
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [0.0, 1.0, -1.0],
            [0.0, 0.0, 1.0],
        ],
    };
    let state = GpuRigidBodyState {
        position_inverse_mass: [0.0; 4],
        orientation: [0.0, 0.0, 0.0, 1.0],
        linear_velocity: [0.0; 4],
        angular_velocity: [0.0; 4],
        inverse_inertia_sleep: [0.0; 4],
    };
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        tested += 1;
        eprintln!("tetrahedron contact backend: {backend:?}");
        let mut world = GpuRigidPrimitiveWorld::new_primitives(
            context.device(),
            context.queue(),
            &[state; 2],
            &[line.clone(), hull.clone()],
            GpuRigidSphereWorldConfig {
                gravity: [0.0; 3],
                ground_half_extent: None,
                ..Default::default()
            },
        )?;
        let _pairs = world.step(0.01)?;
        let contacts = world.readback_contacts()?;
        let first = contacts.pairs[0].1;
        let second = contacts.pair_extra[0][0];
        assert!(first.is_contact() && second.is_contact(), "{contacts:?}");
        for point in [first, second] {
            // The nearest separating plane is (0, 2, 1), at distance 1/sqrt(5).
            assert!(
                (point.depth_hit[0] - 0.2_f32.sqrt()).abs() < 1e-3,
                "{point:?}"
            );
            assert!((point.point[0].abs() - 0.25).abs() < 1e-3, "{point:?}");
            let normal_length = point.normal[..3]
                .iter()
                .map(|value| value * value)
                .sum::<f32>();
            assert!((normal_length - 1.0).abs() < 1e-3);
        }
        assert!((first.point[0] - second.point[0]).abs() > 0.49);
        let mut shifted = state;
        shifted.position_inverse_mass[1] = 2.0;
        world.reset(&[state, shifted])?;
        let _pairs = world.step(0.01)?;
        let contacts = world.readback_contacts()?;
        assert!(!contacts.pairs[0].1.is_contact());
        assert!(
            contacts.pair_extra[0]
                .iter()
                .all(|point| !point.is_contact())
        );
    }
    assert!(tested > 0, "no GPU backend available");
    Ok(())
}
