//! Speculative convex face manifold through owned-world temporal scheduling.
#![cfg(feature = "gpu-contact")]
use tessera_physics::{
    gpu_contact_pipeline::GpuContactDevice,
    gpu_rigid_shape::GpuRigidShape,
    gpu_rigid_sphere_solver::GpuRigidTemporalSolveParams,
    gpu_rigid_sphere_world::{GpuRigidSphereWorld, GpuRigidSphereWorldConfig},
    gpu_rigid_state::GpuRigidBodyState,
    sleep::SleepSettings,
};
#[test]
fn separated_box_faces_support_translation_without_spurious_spin()
-> Result<(), Box<dyn core::error::Error>> {
    let fixed = GpuRigidBodyState {
        position_inverse_mass: [0.0; 4],
        orientation: [0.0, 0.0, 0.0, 1.0],
        linear_velocity: [0.0; 4],
        angular_velocity: [0.0; 4],
        inverse_inertia_sleep: [0.0; 4],
    };
    let moving = GpuRigidBodyState {
        position_inverse_mass: [0.0, 0.0, 1.05, 1.0],
        linear_velocity: [0.0, 0.0, -1.0, 0.0],
        inverse_inertia_sleep: [1.0, 1.0, 1.0, 0.0],
        ..fixed
    };
    let shape = GpuRigidShape::Box {
        half_extents: [0.5; 3],
    };
    let settings = GpuRigidTemporalSolveParams {
        friction: 0.0,
        iterations: 8,
        ..GpuRigidTemporalSolveParams::default()
    };
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        tested += 1;
        eprintln!("temporal convex faces: {backend:?}");
        let mut world = GpuRigidSphereWorld::new_primitives(
            context.device(),
            context.queue(),
            &[fixed, moving],
            &[shape.clone(), shape.clone()],
            GpuRigidSphereWorldConfig {
                gravity: [0.0; 3],
                ground_half_extent: None,
                sleep: SleepSettings {
                    enabled: false,
                    ..SleepSettings::default()
                },
                ..GpuRigidSphereWorldConfig::default()
            },
        )?;
        for frame in 0..3 {
            let count = world.step_temporal_speculative(0.1, 16, settings, 0.1)?;
            assert_eq!(count, 1);
            assert!(!world.is_faulted());
            let states = world.readback()?;
            assert_eq!(states[0].position_inverse_mass, fixed.position_inverse_mass);
            assert_eq!(states[0].linear_velocity, fixed.linear_velocity);
            let actual = states[1];
            assert!(
                actual.position_inverse_mass[2] > 0.995,
                "{backend:?}: frame={frame}: {actual:?}"
            );
            assert!(
                actual.angular_velocity[..3].iter().all(|v| v.abs() < 0.03),
                "{backend:?}: frame={frame}: {actual:?}"
            );
            assert!(
                actual.orientation[..3].iter().all(|v| v.abs() < 0.005),
                "{backend:?}: frame={frame}: {actual:?}"
            );
            let contacts = world.readback_contacts()?;
            assert!(contacts.pairs[0].1.is_contact());
            assert!(
                contacts.pair_extra[0].iter().all(|c| c.is_contact()),
                "{contacts:?}"
            );
        }
    }
    assert!(tested > 0, "no GPU backend available");
    Ok(())
}
