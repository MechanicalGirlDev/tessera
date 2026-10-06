//! LBVH topology changes and impulse remapping during temporal convex contact.
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
fn temporal_convex_lbvh_tracks_entering_and_leaving_support_pairs()
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
    let mut states = vec![
        fixed,
        moving,
        GpuRigidBodyState {
            position_inverse_mass: [10.0, 0.0, 0.0, 0.0],
            ..fixed
        },
        GpuRigidBodyState {
            position_inverse_mass: [10.0, 0.0, 5.0, 1.0],
            ..moving
        },
    ];
    for i in 4..17 {
        states.push(GpuRigidBodyState {
            position_inverse_mass: [100.0 + 3.0 * i as f32, 0.0, 0.0, 0.0],
            ..fixed
        });
    }
    let shapes = vec![
        GpuRigidShape::Box {
            half_extents: [0.5; 3]
        };
        17
    ];
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
        eprintln!("temporal convex LBVH: {backend:?}");
        let mut world = GpuRigidSphereWorld::new_primitives(
            context.device(),
            context.queue(),
            &states,
            &shapes,
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
        assert!(!world.uses_exhaustive_pairs());
        for (phase, expected) in [(0, 1), (1, 2), (2, 1), (3, 2)] {
            match phase {
                1 => world.write_body(
                    3,
                    GpuRigidBodyState {
                        position_inverse_mass: [10.0, 0.0, 1.05, 1.0],
                        ..moving
                    },
                )?,
                2 => world.write_body(
                    0,
                    GpuRigidBodyState {
                        position_inverse_mass: [30.0, 0.0, 0.0, 0.0],
                        ..fixed
                    },
                )?,
                3 => world.write_body(0, fixed)?,
                _ => (),
            }
            let count = world.step_temporal_speculative(0.1, 16, settings, 0.1)?;
            assert_eq!(count, expected, "{backend:?}: phase={phase}");
            assert!(!world.is_faulted());
            let actual = world.readback()?;
            for index in if phase == 0 { vec![1] } else { vec![1, 3] } {
                assert!(
                    actual[index].position_inverse_mass[2] > 0.99,
                    "{backend:?}: phase={phase}: {:?}",
                    actual[index]
                );
                assert!(
                    actual[index].angular_velocity[..3]
                        .iter()
                        .all(|v| v.abs() < 0.03),
                    "{backend:?}: phase={phase}: {:?}",
                    actual[index]
                );
            }
            let contacts = world.readback_contacts()?;
            assert_eq!(contacts.pairs.len(), expected);
            for (slot, (_, point)) in contacts.pairs.iter().enumerate() {
                assert!(point.is_contact(), "{contacts:?}");
                assert!(
                    contacts.pair_extra[slot].iter().all(|c| c.is_contact()),
                    "{contacts:?}"
                );
            }
        }
    }
    assert!(tested > 0, "no GPU backend available");
    Ok(())
}
