//! Zero-thickness polyline intersection and overlap on GPU backends.
#![cfg(feature = "gpu-contact")]
use tessera_physics::{
    gpu_contact_pipeline::GpuContactDevice,
    gpu_rigid_shape::GpuRigidShape,
    gpu_rigid_sphere_world::{GpuRigidPrimitiveWorld, GpuRigidSphereWorldConfig},
    gpu_rigid_state::GpuRigidBodyState,
};

#[test]
fn polyline_pair_crossing_overlap_and_separation() -> Result<(), Box<dyn core::error::Error>> {
    let base = GpuRigidBodyState {
        position_inverse_mass: [0.0; 4],
        orientation: [0.0, 0.0, 0.0, 1.0],
        linear_velocity: [0.0; 4],
        angular_velocity: [0.0; 4],
        inverse_inertia_sleep: [0.0; 4],
    };
    let line = GpuRigidShape::Polyline {
        vertices: vec![[-1.0, 0.0, 0.0], [0.0; 3], [1.0, 0.0, 0.0]],
        segments: vec![[0, 1], [1, 2]],
    };
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        tested += 1;
        eprintln!("polyline pair backend: {backend:?}");
        let mut world = GpuRigidPrimitiveWorld::new_primitives(
            context.device(),
            context.queue(),
            &[base; 2],
            &[line.clone(), line.clone()],
            GpuRigidSphereWorldConfig {
                gravity: [0.0; 3],
                ground_half_extent: None,
                ..Default::default()
            },
        )?;
        for rotated in [false, true] {
            for swapped in [false, true] {
                for (offset, crossing, expected_hit, span) in [
                    ([0.0; 3], true, true, 0.0),
                    ([0.0, 0.0, 0.01], true, false, 0.0),
                    ([0.5, 0.0, 0.0], false, true, 1.5),
                    ([2.0, 0.0, 0.0], false, true, 0.0),
                    ([0.0, 0.01, 0.0], false, false, 0.0),
                    ([2.1, 0.0, 0.0], false, false, 0.0),
                ] {
                    let q = core::f32::consts::FRAC_1_SQRT_2;
                    let mut states = [base; 2];
                    for (index, state) in states.iter_mut().enumerate() {
                        let p = if index == 1 { offset } else { [0.0; 3] };
                        state.position_inverse_mass = if rotated {
                            [p[2] - 3.0, p[1] + 5.0, -p[0] + 2.0, 0.0]
                        } else {
                            [p[0], p[1], p[2], 0.0]
                        };
                        state.orientation = match (rotated, crossing && index == 1) {
                            (false, false) => [0.0, 0.0, 0.0, 1.0],
                            (false, true) => [0.0, 0.0, q, q],
                            (true, false) => [0.0, q, 0.0, q],
                            (true, true) => [0.5; 4],
                        };
                    }
                    if swapped {
                        states.swap(0, 1);
                    }
                    world.reset(&states)?;
                    let _ = world.step(0.01)?;
                    let contacts = world.readback_contacts()?;
                    let first = contacts.pairs.first().ok_or("missing exhaustive pair")?.1;
                    assert_eq!(
                        first.is_contact(),
                        expected_hit,
                        "{backend:?}: {rotated}: {swapped}: {offset:?}: {first:?}"
                    );
                    let points: Vec<_> = core::iter::once(first)
                        .chain(contacts.pair_extra[0])
                        .filter(|point| point.is_contact())
                        .collect();
                    if !expected_hit {
                        assert!(points.is_empty());
                        continue;
                    }
                    for (index, point) in points.iter().enumerate() {
                        assert_eq!(point.depth_hit[0], 0.0);
                        let norm = point.normal[..3].iter().map(|v| v * v).sum::<f32>();
                        assert!((norm - 1.0).abs() < 1e-4, "{point:?}");
                        let mut p = [point.point[0], point.point[1], point.point[2]];
                        if rotated {
                            p = [2.0 - p[2], p[1] - 5.0, p[0] + 3.0];
                        }
                        assert!(p[1].abs() < 2e-5 && p[2].abs() < 2e-5, "{point:?}");
                        assert!(p[0] >= -1.0 - 2e-5 && p[0] <= 1.0 + 2e-5);
                        if crossing {
                            assert!(p[0].abs() < 2e-5);
                        } else {
                            assert!(
                                p[0] >= offset[0] - 1.0 - 2e-5 && p[0] <= offset[0] + 1.0 + 2e-5
                            );
                        }
                        assert!(points[..index].iter().all(|other| {
                            point.point[..3]
                                .iter()
                                .zip(&other.point[..3])
                                .map(|(a, b)| (a - b).powi(2))
                                .sum::<f32>()
                                > 1e-10
                        }));
                    }
                    let component = if rotated { 2 } else { 0 };
                    let low = points
                        .iter()
                        .map(|p| p.point[component])
                        .fold(f32::INFINITY, f32::min);
                    let high = points
                        .iter()
                        .map(|p| p.point[component])
                        .fold(f32::NEG_INFINITY, f32::max);
                    assert!((high - low - span).abs() < 2e-5, "{points:?}");
                    if span == 0.0 {
                        assert_eq!(points.len(), 1);
                    }
                }
            }
        }
    }
    assert!(tested > 0, "no GPU backend available");
    Ok(())
}
