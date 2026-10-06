//! GPU support-map overlap and bounded face/edge-axis contacts for convex pairs.

use core::mem::{size_of, size_of_val};
use core::time::Duration;
use std::sync::mpsc;

use wgpu::util::DeviceExt;

use crate::gpu_contact_pipeline::ContactPipelineError;
use crate::gpu_sphere_contact::GpuSphereContact;

/// Analytic support-map parameters evaluated directly by the GPU kernel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuAnalyticShape {
    /// Circular cylinder around a world-space unit axis.
    Cylinder {
        /// Unit axis from the bottom cap toward the top cap.
        axis: [f32; 3],
        /// Half the full height.
        half_height: f32,
        /// Circular radius.
        radius: f32,
    },
    /// Circular cone around a world-space unit axis.
    Cone {
        /// Unit axis from the base centre toward the apex.
        axis: [f32; 3],
        /// Half the full height.
        half_height: f32,
        /// Base radius.
        radius: f32,
    },
}

/// World-space convex hull prepared for one collision dispatch.
#[derive(Debug, Clone)]
pub struct GpuConvexHull {
    /// Vertices in world coordinates.
    pub vertices: Vec<[f32; 4]>,
    /// Unique world-space face normals.
    pub normals: Vec<[f32; 4]>,
    /// Spherical border radius around the vertex core.
    ///
    /// One vertex represents a sphere and two vertices represent a capsule.
    pub radius: f32,
    /// Optional exact analytic support map. The vertex list still supplies the
    /// world-space centre and conservative bounds to shared callers.
    pub analytic: Option<GpuAnalyticShape>,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HullRange {
    ranges: [u32; 4],
    center: [f32; 4],
    axis_kind: [f32; 4],
    params: [f32; 4],
    edge_range: [u32; 4],
}

/// Reusable GJK and bounded separating-axis GPU narrow phase.
#[derive(Debug)]
pub struct GpuConvexContacts {
    pipeline: wgpu::ComputePipeline,
}

impl GpuConvexContacts {
    /// Compile the convex contact kernel for the selected device.
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera convex contacts"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_convex_contact.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tessera convex contact pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { pipeline }
    }

    /// Detect candidate pairs; results follow the input pair order.
    pub fn detect(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        hulls: &[GpuConvexHull],
        pairs: &[(usize, usize)],
    ) -> Result<Vec<GpuSphereContact>, ContactPipelineError> {
        self.detect_with_mesh_prisms(device, queue, hulls, pairs, &[])
    }

    /// Detect pairs with an optional per-hull mesh-prism mask. An empty mask
    /// disables the specialized sphere-face contact path.
    pub fn detect_with_mesh_prisms(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        hulls: &[GpuConvexHull],
        pairs: &[(usize, usize)],
        mesh_prisms: &[bool],
    ) -> Result<Vec<GpuSphereContact>, ContactPipelineError> {
        self.detect_with_mesh_prisms_and_edges(device, queue, hulls, pairs, mesh_prisms, &[])
    }

    /// Detect pairs using optional per-hull edge directions when the bounded
    /// convex-polytope separating-axis work fits the shader limit.
    pub fn detect_with_edges(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        hulls: &[GpuConvexHull],
        pairs: &[(usize, usize)],
        edge_directions: &[Vec<[f32; 4]>],
    ) -> Result<Vec<GpuSphereContact>, ContactPipelineError> {
        self.detect_with_mesh_prisms_and_edges(device, queue, hulls, pairs, &[], edge_directions)
    }

    /// Detect convex pairs with optional mesh-prism and edge-axis metadata.
    pub fn detect_with_mesh_prisms_and_edges(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        hulls: &[GpuConvexHull],
        pairs: &[(usize, usize)],
        mesh_prisms: &[bool],
        edge_directions: &[Vec<[f32; 4]>],
    ) -> Result<Vec<GpuSphereContact>, ContactPipelineError> {
        if !mesh_prisms.is_empty() && mesh_prisms.len() != hulls.len() {
            return Err(ContactPipelineError::InvalidInput);
        }
        if (!edge_directions.is_empty() && edge_directions.len() != hulls.len())
            || edge_directions.iter().flatten().any(|direction| {
                direction.iter().any(|value| !value.is_finite())
                    || direction[..3]
                        .iter()
                        .map(|value| value * value)
                        .sum::<f32>()
                        <= 1e-12
            })
        {
            return Err(ContactPipelineError::InvalidInput);
        }
        if pairs.is_empty() {
            return Ok(Vec::new());
        }
        if hulls.iter().enumerate().any(|(index, hull)| {
            hull.vertices.is_empty()
                || hull.normals.is_empty()
                || !hull.radius.is_finite()
                || hull.radius < 0.0
                || (mesh_prisms.get(index).copied().unwrap_or(false)
                    && (hull.vertices.len() != 6 || hull.radius != 0.0 || hull.analytic.is_some()))
                || hull
                    .vertices
                    .iter()
                    .chain(&hull.normals)
                    .flatten()
                    .any(|value| !value.is_finite())
                || hull.analytic.is_some_and(|shape| match shape {
                    GpuAnalyticShape::Cylinder {
                        axis,
                        half_height,
                        radius,
                    }
                    | GpuAnalyticShape::Cone {
                        axis,
                        half_height,
                        radius,
                    } => {
                        axis.iter().any(|value| !value.is_finite())
                            || (axis.iter().map(|value| value * value).sum::<f32>() - 1.0).abs()
                                > 1e-4
                            || !half_height.is_finite()
                            || half_height <= 0.0
                            || !radius.is_finite()
                            || radius <= 0.0
                    }
                })
        }) || pairs
            .iter()
            .any(|&(a, b)| a >= hulls.len() || b >= hulls.len() || a == b)
        {
            return Err(ContactPipelineError::InvalidInput);
        }
        let mut ranges = Vec::with_capacity(hulls.len());
        let mut vertices = Vec::new();
        let mut normals = Vec::new();
        let mut edges = Vec::new();
        for (index, hull) in hulls.iter().enumerate() {
            let vertex_offset =
                u32::try_from(vertices.len()).map_err(|_| ContactPipelineError::Capacity)?;
            let normal_offset =
                u32::try_from(normals.len()).map_err(|_| ContactPipelineError::Capacity)?;
            let vertex_count =
                u32::try_from(hull.vertices.len()).map_err(|_| ContactPipelineError::Capacity)?;
            let normal_count =
                u32::try_from(hull.normals.len()).map_err(|_| ContactPipelineError::Capacity)?;
            let edge_offset =
                u32::try_from(edges.len()).map_err(|_| ContactPipelineError::Capacity)?;
            let edge_count = u32::try_from(edge_directions.get(index).map_or(0, Vec::len))
                .map_err(|_| ContactPipelineError::Capacity)?;
            let mut center = [0.0; 4];
            for vertex in &hull.vertices {
                for axis in 0..3 {
                    center[axis] += vertex[axis] / vertex_count as f32;
                }
            }
            if center.iter().any(|value| !value.is_finite()) {
                return Err(ContactPipelineError::InvalidInput);
            }
            center[3] = hull.radius;
            let (axis_kind, mut params) = match hull.analytic {
                None => ([0.0; 4], [0.0; 4]),
                Some(GpuAnalyticShape::Cylinder {
                    axis,
                    half_height,
                    radius,
                }) => (
                    [axis[0], axis[1], axis[2], 1.0],
                    [half_height, radius, 0.0, 0.0],
                ),
                Some(GpuAnalyticShape::Cone {
                    axis,
                    half_height,
                    radius,
                }) => (
                    [axis[0], axis[1], axis[2], 2.0],
                    [half_height, radius, 0.0, 0.0],
                ),
            };
            params[2] = f32::from(u8::from(mesh_prisms.get(index).copied().unwrap_or(false)));
            ranges.push(HullRange {
                ranges: [vertex_offset, vertex_count, normal_offset, normal_count],
                center,
                axis_kind,
                params,
                edge_range: [edge_offset, edge_count, 0, 0],
            });
            vertices.extend_from_slice(&hull.vertices);
            normals.extend_from_slice(&hull.normals);
            if let Some(directions) = edge_directions.get(index) {
                edges.extend_from_slice(directions);
            }
        }
        if edges.is_empty() {
            edges.push([0.0; 4]);
        }
        let pairs = pairs
            .iter()
            .map(|&(a, b)| {
                Ok::<_, ContactPipelineError>([
                    u32::try_from(a).map_err(|_| ContactPipelineError::Capacity)?,
                    u32::try_from(b).map_err(|_| ContactPipelineError::Capacity)?,
                    0,
                    0,
                ])
            })
            .collect::<Result<Vec<_>, _>>()?;
        let count = u32::try_from(pairs.len()).map_err(|_| ContactPipelineError::Capacity)?;
        let output_bytes = u64::from(count) * size_of::<GpuSphereContact>() as u64;
        let limits = device.limits();
        if count.div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || [
                size_of_val(ranges.as_slice()) as u64,
                size_of_val(vertices.as_slice()) as u64,
                size_of_val(normals.as_slice()) as u64,
                size_of_val(edges.as_slice()) as u64,
                size_of_val(pairs.as_slice()) as u64,
                output_bytes,
            ]
            .iter()
            .any(|size| {
                *size > u64::from(limits.max_storage_buffer_binding_size)
                    || *size > limits.max_buffer_size
            })
        {
            return Err(ContactPipelineError::Capacity);
        }
        let upload = |label, contents: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let range_buffer = upload("Tessera convex ranges", bytemuck::cast_slice(&ranges));
        let vertex_buffer = upload("Tessera convex vertices", bytemuck::cast_slice(&vertices));
        let normal_buffer = upload("Tessera convex normals", bytemuck::cast_slice(&normals));
        let edge_buffer = upload("Tessera convex edges", bytemuck::cast_slice(&edges));
        let pair_buffer = upload("Tessera convex pairs", bytemuck::cast_slice(&pairs));
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera convex contacts output"),
            size: output_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera convex contacts readback"),
            size: output_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera convex bindings"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: [
                range_buffer.as_entire_binding(),
                vertex_buffer.as_entire_binding(),
                normal_buffer.as_entire_binding(),
                pair_buffer.as_entire_binding(),
                output.as_entire_binding(),
                edge_buffer.as_entire_binding(),
            ]
            .into_iter()
            .enumerate()
            .map(|(binding, resource)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource,
            })
            .collect::<Vec<_>>()
            .as_slice(),
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera convex contact encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Tessera convex contact pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, output_bytes);
        let _submission = queue.submit(Some(encoder.finish()));
        let (sender, receiver) = mpsc::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        let _status = device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(5)),
            })
            .map_err(|error| ContactPipelineError::Readback(error.to_string()))?;
        receiver
            .recv_timeout(Duration::from_secs(5))
            .map_err(|error| ContactPipelineError::Readback(error.to_string()))?
            .map_err(|error| ContactPipelineError::Readback(error.to_string()))?;
        let view = staging.slice(..).get_mapped_range();
        let result = view
            .chunks_exact(size_of::<GpuSphereContact>())
            .map(bytemuck::pod_read_unaligned)
            .collect();
        drop(view);
        staging.unmap();
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::articulated_world::box_box_contact;
    use crate::convex::{
        ConvexContact, ConvexEdgeAxisShape, cone_sphere_contact, cone_support,
        convex_edge_contact_from_axis, convex_edge_manifold, convex_face_manifold_from_axis,
        convex_sphere_contact, cylinder_sphere_contact, cylinder_support, rounded_convex_contact,
        support_map_contact,
    };
    use crate::gpu_contact_pipeline::GpuContactDevice;
    use crate::mesh::{TRIANGLE_HALF_THICKNESS, triangle_prism};
    use nalgebra::{Isometry3, Point3, Translation3, UnitQuaternion, Vector3};

    fn cube(center: [f32; 3]) -> GpuConvexHull {
        let vertices = [-0.5, 0.5]
            .into_iter()
            .flat_map(|x| {
                [-0.5, 0.5].into_iter().flat_map(move |y| {
                    [-0.5, 0.5]
                        .into_iter()
                        .map(move |z| [center[0] + x, center[1] + y, center[2] + z, 0.0])
                })
            })
            .collect();
        GpuConvexHull {
            vertices,
            normals: vec![
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            radius: 0.0,
            analytic: None,
        }
    }

    #[test]
    fn gpu_convex_overlap_and_separation() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let hulls = [cube([0.0; 3]), cube([0.8, 0.0, 0.0]), cube([3.0, 0.0, 0.0])];
        let contacts = context
            .convex()
            .detect(context.device(), context.queue(), &hulls, &[(0, 1), (0, 2)])
            .unwrap();
        assert!(contacts[0].is_contact());
        assert!((contacts[0].depth_hit[0] - 0.2).abs() < 1e-4);
        assert!(contacts[0].normal[0] > 0.99);
        assert!(!contacts[1].is_contact());
    }

    #[test]
    fn gpu_face_witness_reconstructs_four_point_manifold() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let hulls = [cube([0.0; 3]), cube([0.75, 0.0, 0.0])];
        let edges = hulls
            .iter()
            .map(|hull| hull.normals.clone())
            .collect::<Vec<_>>();
        let result = context
            .convex()
            .detect_with_edges(context.device(), context.queue(), &hulls, &[(0, 1)], &edges)
            .unwrap()[0];
        assert!(result.is_contact());
        assert!(
            matches!(result.depth_hit[2], 6.0 | 7.0),
            "unexpected feature {} with depth {} and normal {:?}",
            result.depth_hit[2],
            result.depth_hit[0],
            result.normal
        );
        let unpack = |value: &[f32; 4]| {
            Vector3::new(
                f64::from(value[0]),
                f64::from(value[1]),
                f64::from(value[2]),
            )
        };
        let vertices_a = hulls[0].vertices.iter().map(&unpack).collect::<Vec<_>>();
        let vertices_b = hulls[1].vertices.iter().map(&unpack).collect::<Vec<_>>();
        let normals_a = hulls[0].normals.iter().map(&unpack).collect::<Vec<_>>();
        let normals_b = hulls[1].normals.iter().map(&unpack).collect::<Vec<_>>();
        let selected = result.point[3] as usize;
        let axis = if result.depth_hit[2] == 6.0 {
            normals_a[selected]
        } else {
            normals_b[selected]
        };
        let manifold = convex_face_manifold_from_axis(
            &vertices_a,
            &vertices_b,
            axis,
            ConvexContact {
                point: unpack(&result.point),
                normal: unpack(&result.normal),
                penetration: f64::from(result.depth_hit[0]),
            },
        )
        .unwrap();
        let reference = convex_edge_manifold(
            &vertices_a,
            &normals_a,
            &normals_a,
            &vertices_b,
            &normals_b,
            &normals_b,
        );
        assert_eq!(manifold.len(), 4);
        assert_eq!(reference.len(), 4);
        for contact in &manifold {
            assert!((contact.penetration - 0.25).abs() < 1e-5);
            assert!(contact.normal.x > 0.99);
            assert!(
                reference
                    .iter()
                    .any(|expected| { (contact.point - expected.point).norm() < 1e-5 })
            );
        }
    }

    #[test]
    fn gpu_convex_agrees_with_rotated_box_sat() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let half_a = Vector3::new(0.4, 0.2, 0.3);
        let half_b = Vector3::new(0.3, 0.4, 0.2);
        let pose_a = Isometry3::from_parts(
            Translation3::identity(),
            UnitQuaternion::from_euler_angles(0.3, 0.2, -0.1),
        );
        let as_hull = |pose: Isometry3<f64>, half: Vector3<f64>| GpuConvexHull {
            vertices: [-1.0, 1.0]
                .into_iter()
                .flat_map(|x| {
                    [-1.0, 1.0].into_iter().flat_map(move |y| {
                        [-1.0, 1.0].into_iter().map(move |z| {
                            let point = pose.transform_point(&Point3::from(
                                half.component_mul(&Vector3::new(x, y, z)),
                            ));
                            [point.x as f32, point.y as f32, point.z as f32, 0.0]
                        })
                    })
                })
                .collect(),
            normals: [Vector3::x(), Vector3::y(), Vector3::z()]
                .map(|axis| {
                    let normal = pose.rotation * axis;
                    [normal.x as f32, normal.y as f32, normal.z as f32, 0.0]
                })
                .to_vec(),
            radius: 0.0,
            analytic: None,
        };
        let mut hulls = vec![as_hull(pose_a, half_a)];
        let mut expected = Vec::new();
        let mut pairs = Vec::new();
        for index in 0..250 {
            let phase = index as f64 * 0.37;
            let pose_b = Isometry3::from_parts(
                Translation3::new(
                    (phase * 0.23).sin() * 1.3,
                    (phase * 0.43).cos() * 0.9,
                    (phase * 0.71).sin() * 0.6,
                ),
                UnitQuaternion::from_euler_angles(phase * 0.09, phase * 0.14, phase * 0.19),
            );
            expected.push(box_box_contact(pose_a, half_a, pose_b, half_b));
            pairs.push((0, hulls.len()));
            hulls.push(as_hull(pose_b, half_b));
        }
        let contacts = context
            .convex()
            .detect(context.device(), context.queue(), &hulls, &pairs)
            .unwrap();
        let edge_directions = hulls
            .iter()
            .map(|hull| hull.normals.clone())
            .collect::<Vec<_>>();
        let edge_contacts = context
            .convex()
            .detect_with_edges(
                context.device(),
                context.queue(),
                &hulls,
                &pairs,
                &edge_directions,
            )
            .unwrap();
        let mut edge_improvements = 0;
        let mut direct_edge_witnesses = 0;
        for (index, ((face_only, contact), expected)) in contacts
            .iter()
            .zip(&edge_contacts)
            .zip(expected)
            .enumerate()
        {
            assert_eq!(contact.is_contact(), expected.is_some(), "case {index}");
            if let Some((normal, _, depth)) = expected
                && depth > 1e-3
            {
                assert!(
                    (f64::from(contact.depth_hit[0]) - depth).abs() < 2e-4,
                    "case {index}: GPU depth {} != CPU depth {depth}",
                    contact.depth_hit[0]
                );
                if f64::from(face_only.depth_hit[0]) - depth > 1e-3 {
                    let gpu_normal = Vector3::new(
                        f64::from(contact.normal[0]),
                        f64::from(contact.normal[1]),
                        f64::from(contact.normal[2]),
                    );
                    assert!(gpu_normal.dot(&normal) > 0.99, "case {index}");
                    edge_improvements += 1;
                    if contact.depth_hit[2] == 5.0 {
                        let first = &hulls[0];
                        let second = &hulls[index + 1];
                        let unpack = |point: &[f32; 4]| {
                            Vector3::new(
                                f64::from(point[0]),
                                f64::from(point[1]),
                                f64::from(point[2]),
                            )
                        };
                        let vertices_a = first.vertices.iter().map(&unpack).collect::<Vec<_>>();
                        let vertices_b = second.vertices.iter().map(&unpack).collect::<Vec<_>>();
                        let edges_a = first.normals.iter().map(&unpack).collect::<Vec<_>>();
                        let edges_b = second.normals.iter().map(&unpack).collect::<Vec<_>>();
                        let reconstructed = convex_edge_contact_from_axis(
                            ConvexEdgeAxisShape {
                                vertices: &vertices_a,
                                directions: &edges_a,
                                selected: contact.point[3] as usize,
                            },
                            ConvexEdgeAxisShape {
                                vertices: &vertices_b,
                                directions: &edges_b,
                                selected: contact.normal[3] as usize,
                            },
                            ConvexContact {
                                point: Vector3::zeros(),
                                normal: gpu_normal,
                                penetration: f64::from(contact.depth_hit[0]),
                            },
                        )
                        .unwrap();
                        assert!((reconstructed.penetration - depth).abs() < 2e-4);
                        assert!(reconstructed.normal.dot(&normal) > 0.99);
                        let reference = convex_edge_manifold(
                            &vertices_a,
                            &edges_a,
                            &edges_a,
                            &vertices_b,
                            &edges_b,
                            &edges_b,
                        );
                        assert_eq!(reference.len(), 1);
                        assert!((reconstructed.point - reference[0].point).norm() < 1e-3);
                        direct_edge_witnesses += 1;
                    }
                }
            }
        }
        assert!(edge_improvements > 0);
        assert!(direct_edge_witnesses > 0);
    }

    #[test]
    fn gpu_sphere_against_convex_hull() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let sphere = |x| GpuConvexHull {
            vertices: vec![[x, 0.0, 0.0, 0.0]],
            normals: vec![[1.0, 0.0, 0.0, 0.0]],
            radius: 0.4,
            analytic: None,
        };
        let hulls = [cube([0.0; 3]), sphere(0.8), sphere(2.0)];
        let contacts = context
            .convex()
            .detect(context.device(), context.queue(), &hulls, &[(0, 1), (0, 2)])
            .unwrap();
        assert!(contacts[0].is_contact());
        assert!((contacts[0].depth_hit[0] - 0.1).abs() < 1e-4);
        assert!(contacts[0].normal[0] > 0.99);
        assert!(!contacts[1].is_contact());
    }

    #[test]
    fn gpu_mesh_prism_sphere_face_reports_point_and_both_orientations() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let prism = triangle_prism([
            Vector3::new(-1.0, -1.0, 0.0),
            Vector3::new(1.0, -1.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        ])
        .unwrap();
        let pack = |value: &Vector3<f64>| [value.x as f32, value.y as f32, value.z as f32, 0.0];
        let hulls = [
            GpuConvexHull {
                vertices: prism.vertices.iter().map(&pack).collect(),
                normals: prism.face_normals.iter().map(&pack).collect(),
                radius: 0.0,
                analytic: None,
            },
            GpuConvexHull {
                vertices: vec![[0.0, 0.0, 0.5, 0.0]],
                normals: vec![[0.0, 0.0, 1.0, 0.0]],
                radius: 0.5,
                analytic: None,
            },
        ];
        let contacts = context
            .convex()
            .detect_with_mesh_prisms(
                context.device(),
                context.queue(),
                &hulls,
                &[(0, 1), (1, 0)],
                &[true, false],
            )
            .unwrap();
        for (index, contact) in contacts.iter().enumerate() {
            assert!(contact.is_contact());
            assert_eq!(contact.depth_hit[2], 1.0);
            assert!(contact.point[0].abs() < 1e-6);
            assert!(contact.point[1].abs() < 1e-6);
            assert!((f64::from(contact.point[2]) - TRIANGLE_HALF_THICKNESS * 0.5).abs() < 1e-6);
            assert!((f64::from(contact.depth_hit[0]) - TRIANGLE_HALF_THICKNESS).abs() < 1e-6);
            let expected_z = if index == 0 { 1.0 } else { -1.0 };
            assert!((contact.normal[2] - expected_z).abs() < 1e-6);
        }
    }

    #[test]
    fn gpu_mesh_prism_edge_contact_reports_exact_witness() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let prism = triangle_prism([
            Vector3::new(-1.0, -1.0, 0.0),
            Vector3::new(1.0, -1.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        ])
        .unwrap();
        let center = Vector3::new(0.0, -1.15, 0.0);
        assert!(convex_sphere_contact(&prism.vertices, &prism.face_normals, center, 0.3).is_some());
        let pack = |value: &Vector3<f64>| [value.x as f32, value.y as f32, value.z as f32, 0.0];
        let mut hulls = [
            GpuConvexHull {
                vertices: prism.vertices.iter().map(&pack).collect(),
                normals: prism.face_normals.iter().map(&pack).collect(),
                radius: 0.0,
                analytic: None,
            },
            GpuConvexHull {
                vertices: vec![pack(&center)],
                normals: vec![[0.0, 0.0, 1.0, 0.0]],
                radius: 0.3,
                analytic: None,
            },
        ];
        let contact = context
            .convex()
            .detect_with_mesh_prisms(
                context.device(),
                context.queue(),
                &hulls,
                &[(0, 1)],
                &[true, false],
            )
            .unwrap()[0];
        assert!(contact.is_contact());
        assert_eq!(contact.depth_hit[2], 2.0);
        assert!((contact.depth_hit[0] - 0.15).abs() < 1e-5);
        assert!((contact.point[1] + 0.925).abs() < 1e-5);
        assert!((contact.normal[1] + 1.0).abs() < 1e-5);

        hulls[1].vertices[0] = [-1.15, -1.15, 0.0, 0.0];
        let corners = context
            .convex()
            .detect_with_mesh_prisms(
                context.device(),
                context.queue(),
                &hulls,
                &[(0, 1), (1, 0)],
                &[true, false],
            )
            .unwrap();
        for (index, corner) in corners.iter().enumerate() {
            assert!(corner.is_contact());
            assert_eq!(corner.depth_hit[2], 2.0);
            assert!((corner.depth_hit[0] - (0.3 - 0.15 * 2.0_f32.sqrt())).abs() < 1e-5);
            let direction = if index == 0 { -1.0 } else { 1.0 };
            assert!((corner.normal[0] - direction / 2.0_f32.sqrt()).abs() < 1e-5);
            assert!((corner.normal[1] - direction / 2.0_f32.sqrt()).abs() < 1e-5);
        }

        hulls[1].vertices[0] = [0.0, -1.31, 0.0, 0.0];
        let miss = context
            .convex()
            .detect_with_mesh_prisms(
                context.device(),
                context.queue(),
                &hulls,
                &[(0, 1)],
                &[true, false],
            )
            .unwrap()[0];
        assert!(!miss.is_contact());
        assert_eq!(miss.depth_hit[2], 2.0);
    }

    #[test]
    fn gpu_sphere_convex_agrees_with_cpu_across_positions() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let cube = cube([0.0; 3]);
        let cpu_vertices = cube
            .vertices
            .iter()
            .map(|point| {
                Vector3::new(
                    f64::from(point[0]),
                    f64::from(point[1]),
                    f64::from(point[2]),
                )
            })
            .collect::<Vec<_>>();
        let cpu_normals = cube
            .normals
            .iter()
            .map(|point| {
                Vector3::new(
                    f64::from(point[0]),
                    f64::from(point[1]),
                    f64::from(point[2]),
                )
            })
            .collect::<Vec<_>>();
        let mut hulls = vec![cube];
        let mut pairs = Vec::new();
        let mut expected = Vec::new();
        for index in 0..500 {
            let phase = index as f64 * 0.37;
            let center = Vector3::new(
                (phase * 0.23).sin() * 1.1,
                (phase * 0.43).cos() * 1.1,
                (phase * 0.71).sin() * 1.1,
            );
            expected.push(convex_sphere_contact(
                &cpu_vertices,
                &cpu_normals,
                center,
                0.4,
            ));
            pairs.push((0, hulls.len()));
            hulls.push(GpuConvexHull {
                vertices: vec![[center.x as f32, center.y as f32, center.z as f32, 0.0]],
                normals: vec![[1.0, 0.0, 0.0, 0.0]],
                radius: 0.4,
                analytic: None,
            });
        }
        let contacts = context
            .convex()
            .detect(context.device(), context.queue(), &hulls, &pairs)
            .unwrap();
        for (index, (contact, expected)) in contacts.iter().zip(expected).enumerate() {
            assert_eq!(contact.is_contact(), expected.is_some(), "case {index}");
            if let Some(expected) = expected {
                let point = Vector3::new(
                    f64::from(contact.point[0]),
                    f64::from(contact.point[1]),
                    f64::from(contact.point[2]),
                );
                let normal = Vector3::new(
                    f64::from(contact.normal[0]),
                    f64::from(contact.normal[1]),
                    f64::from(contact.normal[2]),
                );
                assert!((point - expected.point).norm() < 1e-4, "point case {index}");
                assert!(
                    (normal - expected.normal).norm() < 1e-4,
                    "normal case {index}"
                );
                assert!(
                    (f64::from(contact.depth_hit[0]) - expected.penetration).abs() < 1e-4,
                    "depth case {index}"
                );
            }
        }
    }

    #[test]
    fn gpu_rotated_sphere_convex_contact_matches_cpu_geometry() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let rotation = UnitQuaternion::from_euler_angles(0.27, -0.41, 0.19);
        let mut cube = cube([0.0; 3]);
        for vertex in &mut cube.vertices {
            let rotated = rotation * Vector3::new(vertex[0], vertex[1], vertex[2]);
            vertex[..3].copy_from_slice(rotated.as_slice());
        }
        for normal in &mut cube.normals {
            let rotated = rotation * Vector3::new(normal[0], normal[1], normal[2]);
            normal[..3].copy_from_slice(rotated.as_slice());
        }
        let vertices = cube
            .vertices
            .iter()
            .map(|point| {
                Vector3::new(
                    f64::from(point[0]),
                    f64::from(point[1]),
                    f64::from(point[2]),
                )
            })
            .collect::<Vec<_>>();
        let normals = cube
            .normals
            .iter()
            .map(|point| {
                Vector3::new(
                    f64::from(point[0]),
                    f64::from(point[1]),
                    f64::from(point[2]),
                )
            })
            .collect::<Vec<_>>();
        let mut hulls = vec![cube];
        let mut pairs = Vec::new();
        let mut expected = Vec::new();
        for index in 0..250 {
            let phase = index as f64 * 0.37;
            let center = Vector3::new(
                (phase * 0.23).sin() * 1.1,
                (phase * 0.43).cos() * 1.1,
                (phase * 0.71).sin() * 1.1,
            );
            expected.push(convex_sphere_contact(&vertices, &normals, center, 0.4));
            pairs.push((0, hulls.len()));
            hulls.push(GpuConvexHull {
                vertices: vec![[center.x as f32, center.y as f32, center.z as f32, 0.0]],
                normals: vec![[1.0, 0.0, 0.0, 0.0]],
                radius: 0.4,
                analytic: None,
            });
        }
        let contacts = context
            .convex()
            .detect(context.device(), context.queue(), &hulls, &pairs)
            .unwrap();
        for (index, (actual, reference)) in contacts.iter().zip(expected).enumerate() {
            assert_eq!(actual.is_contact(), reference.is_some(), "case {index}");
            if let Some(reference) = reference {
                let point = Vector3::new(
                    f64::from(actual.point[0]),
                    f64::from(actual.point[1]),
                    f64::from(actual.point[2]),
                );
                let normal = Vector3::new(
                    f64::from(actual.normal[0]),
                    f64::from(actual.normal[1]),
                    f64::from(actual.normal[2]),
                );
                assert!(
                    (point - reference.point).norm() < 1e-4,
                    "point case {index}"
                );
                assert!(
                    (normal - reference.normal).norm() < 1e-4,
                    "normal case {index}"
                );
                assert!(
                    (f64::from(actual.depth_hit[0]) - reference.penetration).abs() < 1e-4,
                    "depth case {index}"
                );
            }
        }
    }

    #[test]
    fn gpu_rounded_segments_detect_capsule_overlap() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let capsule = |x: f32| GpuConvexHull {
            vertices: vec![[x, 0.0, -1.0, 0.0], [x, 0.0, 1.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0, 0.0]],
            radius: 0.5,
            analytic: None,
        };
        let hulls = [capsule(0.0), capsule(0.8), capsule(1.2)];
        let contacts = context
            .convex()
            .detect(context.device(), context.queue(), &hulls, &[(0, 1), (0, 2)])
            .unwrap();
        assert!(contacts[0].is_contact());
        assert_eq!(contacts[0].depth_hit[2], 3.0);
        assert!((contacts[0].depth_hit[0] - 0.2).abs() < 1e-4);
        assert!((contacts[0].point[0] - 0.4).abs() < 1e-5);
        assert!((contacts[0].normal[0] - 1.0).abs() < 1e-5);
        assert!(!contacts[1].is_contact());
        assert_eq!(contacts[1].depth_hit[2], 3.0);
    }

    #[test]
    fn gpu_sphere_capsule_contact_matches_exact_cpu() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let cap_vertices = [Vector3::new(0.0, 0.0, -1.0), Vector3::new(0.0, 0.0, 1.0)];
        let centers = [
            Vector3::new(0.6, 0.0, 0.0),
            Vector3::new(0.6, 0.0, 1.2),
            Vector3::new(1.2, 0.0, 0.0),
        ];
        let mut hulls = vec![GpuConvexHull {
            vertices: cap_vertices
                .iter()
                .map(|point| [point.x as f32, point.y as f32, point.z as f32, 0.0])
                .collect(),
            normals: vec![[0.0, 0.0, 1.0, 0.0]],
            radius: 0.5,
            analytic: None,
        }];
        hulls.extend(centers.iter().map(|center| GpuConvexHull {
            vertices: vec![[center.x as f32, center.y as f32, center.z as f32, 0.0]],
            normals: vec![[1.0, 0.0, 0.0, 0.0]],
            radius: 0.3,
            analytic: None,
        }));
        let pairs = (1..hulls.len())
            .flat_map(|index| [(0, index), (index, 0)])
            .collect::<Vec<_>>();
        let actual = context
            .convex()
            .detect(context.device(), context.queue(), &hulls, &pairs)
            .unwrap();
        for (case, center) in centers.iter().enumerate() {
            let reference = rounded_convex_contact(
                &cap_vertices,
                &[Vector3::z()],
                &[Vector3::z()],
                0.5,
                &[*center],
                &[Vector3::x()],
                &[],
                0.3,
            );
            for orientation in 0..2 {
                let contact = actual[case * 2 + orientation];
                assert_eq!(contact.is_contact(), reference.is_some());
                assert_eq!(contact.depth_hit[2], 3.0);
                if let Some(reference) = reference {
                    let point = Vector3::new(
                        f64::from(contact.point[0]),
                        f64::from(contact.point[1]),
                        f64::from(contact.point[2]),
                    );
                    let normal = Vector3::new(
                        f64::from(contact.normal[0]),
                        f64::from(contact.normal[1]),
                        f64::from(contact.normal[2]),
                    );
                    let expected_normal = if orientation == 0 {
                        reference.normal
                    } else {
                        -reference.normal
                    };
                    assert!((point - reference.point).norm() < 1e-5);
                    assert!((normal - expected_normal).norm() < 1e-5);
                    assert!((f64::from(contact.depth_hit[0]) - reference.penetration).abs() < 1e-5);
                }
            }
        }
    }

    #[test]
    fn gpu_skew_capsules_match_exact_cpu() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let first = [Vector3::new(0.0, 0.0, -1.0), Vector3::new(0.0, 0.0, 1.0)];
        let others = [
            [Vector3::new(0.4, -1.0, 0.0), Vector3::new(0.4, 1.0, 0.0)],
            [Vector3::new(0.4, 0.0, 1.2), Vector3::new(0.4, 0.0, 2.2)],
        ];
        let pack = |points: &[Vector3<f64>; 2]| GpuConvexHull {
            vertices: points
                .iter()
                .map(|point| [point.x as f32, point.y as f32, point.z as f32, 0.0])
                .collect(),
            normals: vec![[0.0, 0.0, 1.0, 0.0]],
            radius: 0.3,
            analytic: None,
        };
        let hulls = [pack(&first), pack(&others[0]), pack(&others[1])];
        let contacts = context
            .convex()
            .detect(context.device(), context.queue(), &hulls, &[(0, 1), (0, 2)])
            .unwrap();
        for (index, other) in others.iter().enumerate() {
            let reference = rounded_convex_contact(
                &first,
                &[Vector3::z()],
                &[Vector3::z()],
                0.3,
                other,
                &[Vector3::z()],
                &[other[1] - other[0]],
                0.3,
            )
            .unwrap();
            let actual = contacts[index];
            assert!(actual.is_contact());
            assert_eq!(actual.depth_hit[2], 3.0);
            let point = Vector3::new(
                f64::from(actual.point[0]),
                f64::from(actual.point[1]),
                f64::from(actual.point[2]),
            );
            let normal = Vector3::new(
                f64::from(actual.normal[0]),
                f64::from(actual.normal[1]),
                f64::from(actual.normal[2]),
            );
            assert!((point - reference.point).norm() < 1e-5, "case {index}");
            assert!((normal - reference.normal).norm() < 1e-5, "case {index}");
            assert!(
                (f64::from(actual.depth_hit[0]) - reference.penetration).abs() < 1e-5,
                "case {index}"
            );
        }
    }

    #[test]
    fn gpu_analytic_cylinder_and_cone_match_cpu_support_maps() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let cylinder = GpuConvexHull {
            vertices: vec![[0.0, 0.0, 0.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0, 0.0]],
            radius: 0.0,
            analytic: Some(GpuAnalyticShape::Cylinder {
                axis: [0.0, 0.0, 1.0],
                half_height: 1.0,
                radius: 0.5,
            }),
        };
        let cone = GpuConvexHull {
            vertices: vec![[3.0, 0.0, 0.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0, 0.0]],
            radius: 0.0,
            analytic: Some(GpuAnalyticShape::Cone {
                axis: [0.0, 0.0, 1.0],
                half_height: 1.0,
                radius: 0.5,
            }),
        };
        let spheres = [
            ([0.8, 0.0, 0.0], 0.4),
            ([1.2, 0.0, 0.0], 0.4),
            ([3.35, 0.0, -0.8], 0.3),
            ([4.0, 0.0, 0.0], 0.2),
            ([0.0, 0.0, 1.2], 0.3),
            ([0.1, 0.0, 0.0], 0.2),
            ([3.2, 0.0, -1.2], 0.3),
            ([3.0, 0.0, 1.2], 0.3),
        ];
        let mut hulls = vec![cylinder, cone];
        hulls.extend(spheres.map(|(center, radius)| GpuConvexHull {
            vertices: vec![[center[0], center[1], center[2], 0.0]],
            normals: vec![[1.0, 0.0, 0.0, 0.0]],
            radius,
            analytic: None,
        }));
        let pairs = [
            (0, 2),
            (0, 3),
            (1, 4),
            (1, 5),
            (0, 6),
            (0, 7),
            (1, 8),
            (1, 9),
        ];
        let contacts = context
            .convex()
            .detect(context.device(), context.queue(), &hulls, &pairs)
            .unwrap();
        let cpu = pairs.map(|(analytic, sphere)| {
            let center = Vector3::from(spheres[sphere - 2].0.map(f64::from));
            let radius = f64::from(spheres[sphere - 2].1);
            let analytic_center = if analytic == 0 {
                Vector3::zeros()
            } else {
                Vector3::new(3.0, 0.0, 0.0)
            };
            support_map_contact(
                analytic_center,
                |direction| {
                    if analytic == 0 {
                        cylinder_support(analytic_center, Vector3::z(), 1.0, 0.5, direction)
                    } else {
                        cone_support(analytic_center, Vector3::z(), 1.0, 0.5, direction)
                    }
                },
                center,
                |direction| {
                    center + direction.try_normalize(1e-12).unwrap_or_else(Vector3::x) * radius
                },
            )
            .is_some()
        });
        for (index, (contact, expected)) in contacts.iter().zip(cpu).enumerate() {
            assert_eq!(contact.is_contact(), expected);
            assert_eq!(contact.depth_hit[2], 4.0);
            let (analytic, sphere) = pairs[index];
            let center = Vector3::from(spheres[sphere - 2].0.map(f64::from));
            let radius = f64::from(spheres[sphere - 2].1);
            let reference = if analytic == 0 {
                cylinder_sphere_contact(Vector3::zeros(), Vector3::z(), 1.0, 0.5, center, radius)
            } else {
                cone_sphere_contact(
                    Vector3::new(3.0, 0.0, 0.0),
                    Vector3::z(),
                    1.0,
                    0.5,
                    center,
                    radius,
                )
            };
            assert_eq!(contact.is_contact(), reference.is_some());
            if let Some(reference) = reference {
                let point = Vector3::new(
                    f64::from(contact.point[0]),
                    f64::from(contact.point[1]),
                    f64::from(contact.point[2]),
                );
                let normal = Vector3::new(
                    f64::from(contact.normal[0]),
                    f64::from(contact.normal[1]),
                    f64::from(contact.normal[2]),
                );
                assert!(
                    (point - reference.point).norm() < 1e-5,
                    "point case {index}"
                );
                assert!(
                    (normal - reference.normal).norm() < 1e-5,
                    "normal case {index}"
                );
                assert!(
                    (f64::from(contact.depth_hit[0]) - reference.penetration).abs() < 1e-5,
                    "depth case {index}"
                );
            }
        }
        let reverse_pairs = pairs.map(|(a, b)| (b, a));
        let reverse = context
            .convex()
            .detect(context.device(), context.queue(), &hulls, &reverse_pairs)
            .unwrap();
        for (forward, reverse) in contacts.iter().zip(reverse) {
            assert_eq!(forward.is_contact(), reverse.is_contact());
            assert_eq!(reverse.depth_hit[2], 4.0);
            if forward.is_contact() {
                for axis in 0..3 {
                    assert!((forward.point[axis] - reverse.point[axis]).abs() < 1e-5);
                    assert!((forward.normal[axis] + reverse.normal[axis]).abs() < 1e-5);
                }
            }
        }
    }

    #[test]
    fn gpu_rotated_analytic_sphere_contacts_match_cpu() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let axis = Vector3::y();
        let cases = [
            (
                GpuAnalyticShape::Cylinder {
                    axis: [0.0, 1.0, 0.0],
                    half_height: 1.0,
                    radius: 0.5,
                },
                Vector3::zeros(),
                Vector3::new(0.0, 0.5, 0.8),
            ),
            (
                GpuAnalyticShape::Cone {
                    axis: [0.0, 1.0, 0.0],
                    half_height: 1.0,
                    radius: 0.5,
                },
                Vector3::new(3.0, 0.0, 0.0),
                Vector3::new(3.0, 1.2, 0.0),
            ),
        ];
        let mut hulls = Vec::new();
        let mut pairs = Vec::new();
        for (analytic, shape_center, sphere_center) in cases {
            let index = hulls.len();
            hulls.push(GpuConvexHull {
                vertices: vec![[
                    shape_center.x as f32,
                    shape_center.y as f32,
                    shape_center.z as f32,
                    0.0,
                ]],
                normals: vec![[0.0, 1.0, 0.0, 0.0]],
                radius: 0.0,
                analytic: Some(analytic),
            });
            hulls.push(GpuConvexHull {
                vertices: vec![[
                    sphere_center.x as f32,
                    sphere_center.y as f32,
                    sphere_center.z as f32,
                    0.0,
                ]],
                normals: vec![[1.0, 0.0, 0.0, 0.0]],
                radius: 0.4,
                analytic: None,
            });
            pairs.extend([(index, index + 1), (index + 1, index)]);
        }
        let contacts = context
            .convex()
            .detect(context.device(), context.queue(), &hulls, &pairs)
            .unwrap();
        for (index, (_, shape_center, sphere_center)) in cases.into_iter().enumerate() {
            let reference = if index == 0 {
                cylinder_sphere_contact(shape_center, axis, 1.0, 0.5, sphere_center, 0.4)
            } else {
                cone_sphere_contact(shape_center, axis, 1.0, 0.5, sphere_center, 0.4)
            }
            .unwrap();
            for orientation in 0..2 {
                let contact = contacts[index * 2 + orientation];
                assert!(contact.is_contact());
                assert_eq!(contact.depth_hit[2], 4.0);
                let expected_normal = if orientation == 0 {
                    reference.normal
                } else {
                    -reference.normal
                };
                let point = Vector3::new(
                    f64::from(contact.point[0]),
                    f64::from(contact.point[1]),
                    f64::from(contact.point[2]),
                );
                let normal = Vector3::new(
                    f64::from(contact.normal[0]),
                    f64::from(contact.normal[1]),
                    f64::from(contact.normal[2]),
                );
                assert!((point - reference.point).norm() < 1e-5);
                assert!((normal - expected_normal).norm() < 1e-5);
                assert!((f64::from(contact.depth_hit[0]) - reference.penetration).abs() < 1e-5);
            }
        }
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn dx12_convex_overlap_and_separation() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::DX12,
            ..Default::default()
        });
        let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
        else {
            eprintln!("DX12 compute adapter unavailable; skipping backend test");
            return;
        };
        if adapter.get_info().device_type == wgpu::DeviceType::Cpu {
            return;
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let hulls = [cube([0.0; 3]), cube([0.8, 0.0, 0.0]), cube([3.0, 0.0, 0.0])];
        let edges = hulls
            .iter()
            .map(|hull| hull.normals.clone())
            .collect::<Vec<_>>();
        let contacts = GpuConvexContacts::new(&device)
            .detect_with_edges(&device, &queue, &hulls, &[(0, 1), (0, 2)], &edges)
            .unwrap();
        assert!(contacts[0].is_contact());
        assert!(!contacts[1].is_contact());
    }
}
