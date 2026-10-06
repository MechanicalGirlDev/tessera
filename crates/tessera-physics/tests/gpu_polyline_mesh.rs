//! Segment intersections with triangle surfaces on actual GPU backends.
#![cfg(feature = "gpu-contact")]
use tessera_physics::{
    gpu_contact_pipeline::GpuContactDevice,
    gpu_rigid_shape::GpuRigidShape,
    gpu_rigid_sphere_world::{GpuRigidPrimitiveWorld, GpuRigidSphereWorldConfig},
    gpu_rigid_state::GpuRigidBodyState,
};
#[test]
fn polyline_mesh_transverse_coplanar_and_boundary_contacts()
-> Result<(), Box<dyn core::error::Error>> {
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
    let mut vertices = vec![
        [-0.5, -0.5, 0.0],
        [0.5, -0.5, 0.0],
        [0.5, 0.5, 0.0],
        [-0.5, 0.5, 0.0],
    ];
    let mut triangles = vec![[0, 1, 2], [0, 2, 3]];
    for index in 0..62 {
        let first = vertices.len() as u32;
        let x = 100.0 + index as f32 * 2.0;
        vertices.extend([[x, 100.0, 0.0], [x + 1.0, 100.0, 0.0], [x, 101.0, 0.0]]);
        triangles.push([first, first + 1, first + 2]);
    }
    let mesh = GpuRigidShape::TriangleMesh {
        vertices,
        triangles,
    };
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        tested += 1;
        eprintln!("polyline mesh backend: {backend:?}");
        for swapped in [false, true] {
            let shapes = if swapped {
                [mesh.clone(), line.clone()]
            } else {
                [line.clone(), mesh.clone()]
            };
            let mut world = GpuRigidPrimitiveWorld::new_primitives(
                context.device(),
                context.queue(),
                &[base; 2],
                &shapes,
                GpuRigidSphereWorldConfig {
                    gravity: [0.0; 3],
                    ground_half_extent: None,
                    ..Default::default()
                },
            )?;
            for rotated in [false, true] {
                for (offset, transverse, hit, expected_span) in [
                    ([0.0; 3], false, true, 1.0),
                    ([0.0; 3], true, true, 0.0),
                    ([1.5, 0.0, 0.0], false, true, 0.0),
                    ([0.0, 0.5, 0.0], false, true, 1.0),
                    ([0.0, 0.0, 0.01], false, false, 0.0),
                    ([2.0, 0.0, 0.0], false, false, 0.0),
                    ([1.0, 0.0, 0.0], true, false, 0.0),
                ] {
                    let q = core::f32::consts::FRAC_1_SQRT_2;
                    let mut states = [base; 2];
                    let line_index = usize::from(swapped);
                    for (index, state) in states.iter_mut().enumerate() {
                        let p = if index == line_index {
                            offset
                        } else {
                            [0.0; 3]
                        };
                        state.position_inverse_mass = if rotated {
                            [p[2] - 3.0, p[1] + 5.0, -p[0] + 2.0, 0.0]
                        } else {
                            [p[0], p[1], p[2], 0.0]
                        };
                        state.orientation = match (rotated, transverse && index == line_index) {
                            (false, false) => [0.0, 0.0, 0.0, 1.0],
                            (false, true) | (true, false) => [0.0, q, 0.0, q],
                            (true, true) => [0.0, 1.0, 0.0, 0.0],
                        };
                    }
                    world.reset(&states)?;
                    let _ = world.step(0.01)?;
                    let contacts = world.readback_contacts()?;
                    let first = contacts.pairs.first().ok_or("missing pair")?.1;
                    assert_eq!(
                        first.is_contact(),
                        hit,
                        "{backend:?}: {rotated}: {swapped}: {offset:?}: {first:?}"
                    );
                    let points: Vec<_> = core::iter::once(first)
                        .chain(contacts.pair_extra[0])
                        .filter(|p| p.is_contact())
                        .collect();
                    if !hit {
                        assert!(points.is_empty());
                        continue;
                    }
                    for (index, point) in points.iter().enumerate() {
                        assert_eq!(point.depth_hit[0], 0.0);
                        let normal_axis = if rotated { 0 } else { 2 };
                        assert!(point.normal[normal_axis].abs() > 0.99, "{point:?}");
                        if !rotated {
                            assert!(point.normal[2] * if swapped { -1.0 } else { 1.0 } > 0.99);
                        }
                        let p = if rotated {
                            [
                                2.0 - point.point[2],
                                point.point[1] - 5.0,
                                point.point[0] + 3.0,
                            ]
                        } else {
                            [point.point[0], point.point[1], point.point[2]]
                        };
                        assert!(
                            p[2].abs() < 2e-5
                                && p[0].abs() <= 0.5 + 2e-5
                                && p[1].abs() <= 0.5 + 2e-5,
                            "{point:?}"
                        );
                        assert!((p[1] - offset[1]).abs() < 2e-5);
                        if transverse {
                            assert!((p[0] - offset[0]).abs() < 2e-5);
                        } else {
                            assert!((p[0] - offset[0]).abs() <= 1.0 + 2e-5);
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
                    assert!((high - low - expected_span).abs() < 2e-5, "{points:?}");
                    if expected_span == 0.0 {
                        assert_eq!(points.len(), 1);
                    }
                }
            }
        }
    }
    assert!(tested > 0, "no GPU backend available");
    Ok(())
}
