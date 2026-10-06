//! Thin triangle-surface intersections on actual GPU backends.
#![cfg(feature = "gpu-contact")]
use tessera_physics::{
    gpu_contact_pipeline::GpuContactDevice,
    gpu_rigid_shape::GpuRigidShape,
    gpu_rigid_sphere_world::{GpuRigidPrimitiveWorld, GpuRigidSphereWorldConfig},
    gpu_rigid_state::GpuRigidBodyState,
};
#[test]
fn mesh_pair_containment_transverse_and_vertex_contacts() -> Result<(), Box<dyn core::error::Error>>
{
    let base = GpuRigidBodyState {
        position_inverse_mass: [0.0; 4],
        orientation: [0.0, 0.0, 0.0, 1.0],
        linear_velocity: [0.0; 4],
        angular_velocity: [0.0; 4],
        inverse_inertia_sleep: [0.0; 4],
    };
    let large = GpuRigidShape::TriangleMesh {
        vertices: vec![[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]],
        triangles: vec![[0, 1, 2]],
    };
    let small = GpuRigidShape::TriangleMesh {
        vertices: vec![[-0.25, -0.25, 0.0], [0.25, -0.25, 0.0], [0.0, 0.25, 0.0]],
        triangles: vec![[0, 1, 2]],
    };
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        tested += 1;
        eprintln!("mesh pair backend: {backend:?}");
        let mut world = GpuRigidPrimitiveWorld::new_primitives(
            context.device(),
            context.queue(),
            &[base; 3],
            &[large.clone(), small.clone(), large.clone()],
            GpuRigidSphereWorldConfig {
                gravity: [0.0; 3],
                ground_half_extent: None,
                ..Default::default()
            },
        )?;
        for rotated in [false, true] {
            for active_large in [0, 2] {
                for (offset, transverse, hit, expected_span, expected_count) in [
                    ([0.0; 3], false, true, 0.5, 3),
                    ([0.0; 3], true, true, 0.25, 2),
                    ([0.5, 0.0, 0.0], true, true, 0.125, 2),
                    ([1.25, -0.75, 0.0], false, true, 0.0, 1),
                    ([1.26, -0.75, 0.0], false, false, 0.0, 0),
                    ([0.0, 0.0, 0.01], false, false, 0.0, 0),
                    ([2.0, 0.0, 0.0], false, false, 0.0, 0),
                ] {
                    let q = core::f32::consts::FRAC_1_SQRT_2;
                    let mut states = [base; 3];
                    for (index, state) in states.iter_mut().enumerate() {
                        let p = if index == 1 {
                            offset
                        } else if index == active_large {
                            [0.0; 3]
                        } else {
                            [0.0, 0.0, 100.0]
                        };
                        state.position_inverse_mass = if rotated {
                            [p[2] - 3.0, p[1] + 5.0, -p[0] + 2.0, 0.0]
                        } else {
                            [p[0], p[1], p[2], 0.0]
                        };
                        state.orientation = match (rotated, transverse && index == 1) {
                            (false, false) => [0.0, 0.0, 0.0, 1.0],
                            (false, true) => [q, 0.0, 0.0, q],
                            (true, false) => [0.0, q, 0.0, q],
                            (true, true) => [0.5, 0.5, -0.5, 0.5],
                        };
                    }
                    world.reset(&states)?;
                    let _ = world.step(0.01)?;
                    let contacts = world.readback_contacts()?;
                    let (pair_index, (_, first)) = contacts
                        .pairs
                        .iter()
                        .enumerate()
                        .find(|(_, (pair, _))| {
                            (pair.a as usize == active_large && pair.b == 1)
                                || (pair.a == 1 && pair.b as usize == active_large)
                        })
                        .ok_or("missing exhaustive candidate")?;
                    assert_eq!(
                        first.is_contact(),
                        hit,
                        "{backend:?}: {rotated}: {active_large}: {offset:?}: {first:?}"
                    );
                    let points: Vec<_> = core::iter::once(*first)
                        .chain(contacts.pair_extra[pair_index])
                        .filter(|p| p.is_contact())
                        .collect();
                    assert_eq!(points.len(), expected_count, "{points:?}");
                    if !hit {
                        continue;
                    }
                    for (index, point) in points.iter().enumerate() {
                        assert_eq!(point.depth_hit[0], 0.0);
                        assert!(
                            (point.normal[..3].iter().map(|v| v * v).sum::<f32>() - 1.0).abs()
                                < 1e-4
                        );
                        let p = if rotated {
                            [
                                2.0 - point.point[2],
                                point.point[1] - 5.0,
                                point.point[0] + 3.0,
                            ]
                        } else {
                            [point.point[0], point.point[1], point.point[2]]
                        };
                        assert!(p[2].abs() < 2e-5 && p[1] >= -1.0 - 2e-5 && p[1] <= 1.0 + 2e-5);
                        assert!(p[0].abs() <= (1.0 - p[1]) * 0.5 + 2e-5, "{point:?}");
                        let local = if transverse {
                            [p[0] - offset[0], p[2] - offset[2], -(p[1] - offset[1])]
                        } else {
                            [p[0] - offset[0], p[1] - offset[1], p[2] - offset[2]]
                        };
                        assert!(
                            local[2].abs() < 2e-5
                                && local[1] >= -0.25 - 2e-5
                                && local[1] <= 0.25 + 2e-5
                        );
                        assert!(
                            local[0].abs() <= (0.25 - local[1]) * 0.5 + 2e-5,
                            "{point:?}"
                        );
                        assert!(points[..index].iter().all(|other| {
                            point.point[..3]
                                .iter()
                                .zip(&other.point[..3])
                                .map(|(a, b)| (a - b).powi(2))
                                .sum::<f32>()
                                > 1e-10
                        }));
                        assert!(
                            point
                                .normal
                                .iter()
                                .zip(first.normal.iter())
                                .all(|(a, b)| (a - b).abs() < 1e-5)
                        );
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
                    assert!((high - low - expected_span).abs() < 2e-5, "{points:?}");
                }
            }
        }
        world.reset(&[base; 3])?;
        let _ = world.remove_primitive(2)?;
        let _ = world.remove_primitive(1)?;
        let _ = world.step(0.01)?;
        assert!(world.readback_contacts()?.pairs.is_empty());
        assert_eq!(world.append_primitive(base, large.clone())?, 1);
        let _ = world.step(0.01)?;
        let contacts = world.readback_contacts()?;
        assert_eq!(contacts.pairs.len(), 1);
        assert!(contacts.pairs[0].1.is_contact());
        assert_eq!(
            contacts.pair_extra[0]
                .iter()
                .filter(|p| p.is_contact())
                .count(),
            2
        );
        for index in 0..15 {
            let mut distant = base;
            distant.position_inverse_mass = [100.0 + index as f32 * 10.0, 100.0, 0.0, 0.0];
            assert_eq!(
                world.append_primitive(distant, GpuRigidShape::Sphere { radius: 0.1 })?,
                index + 2
            );
        }
        assert!(!world.uses_exhaustive_pairs());
        let (candidates, output) = world.step_gpu_resident(0.01)?;
        let contacts = output.readback_pairs(context.device(), context.queue(), &candidates)?;
        assert_eq!(contacts.pairs.len(), 1);
        assert_eq!((contacts.pairs[0].0.a, contacts.pairs[0].0.b), (0, 1));
        assert!(contacts.pairs[0].1.is_contact());
        assert_eq!(
            contacts.pair_extra[0]
                .iter()
                .filter(|p| p.is_contact())
                .count(),
            2
        );
    }
    assert!(tested > 0, "no GPU backend available");
    Ok(())
}
