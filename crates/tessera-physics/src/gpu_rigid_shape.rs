//! Collision shapes attached to GPU-resident rigid-body states.

/// One primitive collider centered on a rigid body's local origin.
#[derive(Clone, Debug, PartialEq)]
pub enum GpuRigidShape {
    /// Sphere with a positive radius.
    Sphere {
        /// Radius in metres.
        radius: f32,
    },
    /// Oriented box with positive body-frame half extents.
    Box {
        /// Positive half extents in body-frame XYZ.
        half_extents: [f32; 3],
    },
    /// Z-axis capsule centered on the body origin.
    Capsule {
        /// Radius of the rounded ends in metres.
        radius: f32,
        /// Distance from the body origin to either hemisphere center.
        half_length: f32,
    },
    /// Z-axis cylinder centered on the body origin.
    Cylinder {
        /// Radius of the circular cross section in metres.
        radius: f32,
        /// Distance from the body origin to either flat cap.
        half_length: f32,
    },
    /// Z-axis cone with its apex at positive half length.
    Cone {
        /// Radius of the flat base in metres.
        radius: f32,
        /// Distance from the body origin to the apex or base plane.
        half_length: f32,
    },
    /// Convex hull formed by body-frame vertices, including any interior points.
    Convex {
        /// Finite local vertices spanning a nondegenerate three-dimensional hull.
        vertices: Vec<[f32; 3]>,
    },
    /// Indexed zero-thickness line segments in body-local coordinates.
    /// Primitive and convex hull contacts traverse a local BVH on GPU.
    Polyline {
        /// Finite local vertices.
        vertices: Vec<[f32; 3]>,
        /// Nondegenerate segments indexing the vertices.
        segments: Vec<[u32; 2]>,
    },
    /// Indexed, two-sided triangle surface in body-local coordinates.
    /// A local BVH is built when the GPU world is created and traversed on GPU.
    TriangleMesh {
        /// Finite body-local vertices.
        vertices: Vec<[f32; 3]>,
        /// Nondegenerate triangles indexing the vertices.
        triangles: Vec<[u32; 3]>,
    },
}

impl GpuRigidShape {
    /// Convert a centered z-up heightfield into its exact triangle surface.
    ///
    /// Cell diagonals and winding match the CPU geometry. The resulting mesh
    /// uses the resident GPU BVH/contact path. Reject coordinates that overflow
    /// or collapse triangles when converted from f64 to GPU f32.
    pub fn from_heightfield(
        geometry: &crate::mesh::HeightFieldGeometry,
    ) -> Result<Self, crate::mesh::MeshGeometryError> {
        let shape = Self::TriangleMesh {
            vertices: geometry
                .mesh()
                .vertices()
                .iter()
                .map(|vertex| [vertex.x as f32, vertex.y as f32, vertex.z as f32])
                .collect(),
            triangles: geometry.mesh().triangles().to_vec(),
        };
        let _radius = shape
            .bounding_radius()
            .ok_or(crate::mesh::MeshGeometryError::Invalid)?;
        Ok(shape)
    }

    /// Convert an indexed CPU polyline while validating f32 coordinates and segments.
    pub fn from_polyline(
        geometry: &crate::mesh::PolylineGeometry,
    ) -> Result<Self, crate::mesh::MeshGeometryError> {
        let shape = Self::Polyline {
            vertices: geometry
                .vertices()
                .iter()
                .map(|vertex| [vertex.x as f32, vertex.y as f32, vertex.z as f32])
                .collect(),
            segments: geometry.segments().to_vec(),
        };
        let _radius = shape
            .bounding_radius()
            .ok_or(crate::mesh::MeshGeometryError::Invalid)?;
        Ok(shape)
    }

    /// Conservative sphere radius used for broad-phase candidate generation.
    pub fn bounding_radius(&self) -> Option<f32> {
        match self {
            Self::Sphere { radius } => (radius.is_finite() && *radius > 0.0).then_some(*radius),
            Self::Box { half_extents } => {
                if half_extents
                    .iter()
                    .any(|value| !value.is_finite() || *value <= 0.0)
                {
                    return None;
                }
                let radius = half_extents[0]
                    .hypot(half_extents[1])
                    .hypot(half_extents[2]);
                radius.is_finite().then_some(radius)
            }
            Self::Capsule {
                radius,
                half_length,
            } => {
                let extent = radius + half_length;
                (radius.is_finite()
                    && *radius > 0.0
                    && half_length.is_finite()
                    && *half_length >= 0.0
                    && extent.is_finite())
                .then_some(extent)
            }
            Self::Cylinder {
                radius,
                half_length,
            }
            | Self::Cone {
                radius,
                half_length,
            } => {
                if !radius.is_finite()
                    || *radius <= 0.0
                    || !half_length.is_finite()
                    || *half_length <= 0.0
                {
                    return None;
                }
                let extent = radius.hypot(*half_length);
                extent.is_finite().then_some(extent)
            }
            Self::Convex { vertices } => convex_radius(vertices),
            Self::Polyline { vertices, segments } => polyline_radius(vertices, segments),
            Self::TriangleMesh {
                vertices,
                triangles,
            } => triangle_mesh_radius(vertices, triangles),
        }
    }

    pub(crate) fn packed(
        &self,
        first_vertex: u32,
        first_normal: u32,
        normal_count: u32,
        first_edge: u32,
        edge_count: u32,
    ) -> GpuRigidShapeData {
        match self {
            Self::Sphere { radius } => GpuRigidShapeData {
                kind: [0, 0, 0, 0],
                feature_counts: [0; 4],
                dimensions: [*radius, 0.0, 0.0, 0.0],
            },
            Self::Box { half_extents } => GpuRigidShapeData {
                kind: [1, 0, 0, 0],
                feature_counts: [0; 4],
                dimensions: [half_extents[0], half_extents[1], half_extents[2], 0.0],
            },
            Self::Capsule {
                radius,
                half_length,
            } => GpuRigidShapeData {
                kind: [2, 0, 0, 0],
                feature_counts: [0; 4],
                dimensions: [*radius, *half_length, 0.0, 0.0],
            },
            Self::Cylinder {
                radius,
                half_length,
            } => GpuRigidShapeData {
                kind: [3, 0, 0, 0],
                feature_counts: [0; 4],
                dimensions: [*radius, *half_length, 0.0, 0.0],
            },
            Self::Cone {
                radius,
                half_length,
            } => GpuRigidShapeData {
                kind: [4, 0, 0, 0],
                feature_counts: [0; 4],
                dimensions: [*radius, *half_length, 0.0, 0.0],
            },
            Self::Convex { vertices } => GpuRigidShapeData {
                kind: [5, first_vertex, vertices.len() as u32, first_normal],
                feature_counts: [normal_count, first_edge, edge_count, 0],
                dimensions: [self.bounding_radius().unwrap_or(0.0), 0.0, 0.0, 0.0],
            },
            Self::Polyline { vertices, segments } => GpuRigidShapeData {
                kind: [7, first_vertex, vertices.len() as u32, first_edge],
                feature_counts: [segments.len() as u32, first_normal, normal_count, 0],
                dimensions: [self.bounding_radius().unwrap_or(0.0), 0.0, 0.0, 0.0],
            },
            Self::TriangleMesh {
                vertices,
                triangles,
            } => GpuRigidShapeData {
                kind: [6, first_vertex, vertices.len() as u32, first_edge],
                feature_counts: [triangles.len() as u32, first_normal, normal_count, 0],
                dimensions: [self.bounding_radius().unwrap_or(0.0), 0.0, 0.0, 0.0],
            },
        }
    }
}

fn polyline_radius(vertices: &[[f32; 3]], segments: &[[u32; 2]]) -> Option<f32> {
    if vertices.len() < 2
        || vertices.len() > u32::MAX as usize
        || segments.is_empty()
        || segments.len() > u32::MAX as usize / 3
    {
        return None;
    }
    let mut radius = 0.0_f32;
    for vertex in vertices {
        if vertex.iter().any(|value| !value.is_finite()) {
            return None;
        }
        radius = radius.max(vertex[0].hypot(vertex[1]).hypot(vertex[2]));
    }
    if !radius.is_finite() || radius <= 0.0 {
        return None;
    }
    for &[a, b] in segments {
        let [a, b] = [a as usize, b as usize];
        if a >= vertices.len() || b >= vertices.len() || a == b {
            return None;
        }
        let squared = (0..3)
            .map(|axis| (f64::from(vertices[a][axis]) - f64::from(vertices[b][axis])).powi(2))
            .sum::<f64>();
        if squared <= 1e-20 {
            return None;
        }
    }
    Some(radius)
}

fn triangle_mesh_radius(vertices: &[[f32; 3]], triangles: &[[u32; 3]]) -> Option<f32> {
    if vertices.len() < 3
        || vertices.len() > u32::MAX as usize
        || triangles.is_empty()
        || triangles.len() > u32::MAX as usize / 3
    {
        return None;
    }
    let mut radius = 0.0_f32;
    for vertex in vertices {
        if vertex.iter().any(|value| !value.is_finite()) {
            return None;
        }
        radius = radius.max(vertex[0].hypot(vertex[1]).hypot(vertex[2]));
    }
    if !radius.is_finite() || radius <= 0.0 {
        return None;
    }
    for &[a, b, c] in triangles {
        let [a, b, c] = [a as usize, b as usize, c as usize];
        if a >= vertices.len()
            || b >= vertices.len()
            || c >= vertices.len()
            || a == b
            || b == c
            || c == a
        {
            return None;
        }
        let edge_ab: [f64; 3] = core::array::from_fn(|axis| {
            f64::from(vertices[b][axis]) - f64::from(vertices[a][axis])
        });
        let edge_ac: [f64; 3] = core::array::from_fn(|axis| {
            f64::from(vertices[c][axis]) - f64::from(vertices[a][axis])
        });
        let cross = [
            edge_ab[1] * edge_ac[2] - edge_ab[2] * edge_ac[1],
            edge_ab[2] * edge_ac[0] - edge_ab[0] * edge_ac[2],
            edge_ab[0] * edge_ac[1] - edge_ab[1] * edge_ac[0],
        ];
        if cross.iter().map(|value| value * value).sum::<f64>() <= 1e-20 {
            return None;
        }
    }
    Some(radius)
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct MeshBvhNode {
    pub lower: [f32; 3],
    pub upper: [f32; 3],
    pub triangle: Option<usize>,
    pub escape: usize,
}

pub(crate) fn mesh_bvh_nodes(vertices: &[[f32; 3]], triangles: &[[u32; 3]]) -> Vec<MeshBvhNode> {
    fn triangle_bounds(vertices: &[[f32; 3]], triangle: [u32; 3]) -> ([f32; 3], [f32; 3]) {
        let a = vertices[triangle[0] as usize];
        let b = vertices[triangle[1] as usize];
        let c = vertices[triangle[2] as usize];
        (
            core::array::from_fn(|axis| a[axis].min(b[axis]).min(c[axis])),
            core::array::from_fn(|axis| a[axis].max(b[axis]).max(c[axis])),
        )
    }

    fn centroid_sum(vertices: &[[f32; 3]], triangle: [u32; 3], axis: usize) -> f64 {
        triangle
            .into_iter()
            .map(|index| f64::from(vertices[index as usize][axis]))
            .sum()
    }

    fn append(
        order: &mut [usize],
        vertices: &[[f32; 3]],
        triangles: &[[u32; 3]],
        nodes: &mut Vec<MeshBvhNode>,
    ) {
        let index = nodes.len();
        nodes.push(MeshBvhNode {
            lower: [0.0; 3],
            upper: [0.0; 3],
            triangle: None,
            escape: 0,
        });
        let (mut lower, mut upper) = triangle_bounds(vertices, triangles[order[0]]);
        for &triangle in &order[1..] {
            let (next_lower, next_upper) = triangle_bounds(vertices, triangles[triangle]);
            for axis in 0..3 {
                lower[axis] = lower[axis].min(next_lower[axis]);
                upper[axis] = upper[axis].max(next_upper[axis]);
            }
        }
        if order.len() > 1 {
            let axis = (0..3)
                .max_by(|&a, &b| {
                    (f64::from(upper[a]) - f64::from(lower[a]))
                        .total_cmp(&(f64::from(upper[b]) - f64::from(lower[b])))
                })
                .unwrap_or(0);
            order.sort_unstable_by(|&a, &b| {
                centroid_sum(vertices, triangles[a], axis)
                    .total_cmp(&centroid_sum(vertices, triangles[b], axis))
                    .then_with(|| a.cmp(&b))
            });
            let middle = order.len() / 2;
            let (left, right) = order.split_at_mut(middle);
            append(left, vertices, triangles, nodes);
            append(right, vertices, triangles, nodes);
        }
        nodes[index] = MeshBvhNode {
            lower,
            upper,
            triangle: (order.len() == 1).then_some(order[0]),
            escape: nodes.len(),
        };
    }

    let mut order = (0..triangles.len()).collect::<Vec<_>>();
    let mut nodes = Vec::with_capacity(triangles.len().saturating_mul(2).saturating_sub(1));
    if !order.is_empty() {
        append(&mut order, vertices, triangles, &mut nodes);
    }
    nodes
}

/// Find outward supporting face normals of a nondegenerate GPU convex hull.
pub fn convex_face_normals(vertices: &[[f32; 3]]) -> Vec<[f32; 3]> {
    let mut normals = Vec::<[f32; 3]>::new();
    let span = convex_span(vertices);
    let tolerance = span * 1e-7;
    for i in 0..vertices.len() {
        for j in (i + 1)..vertices.len() {
            for k in (j + 1)..vertices.len() {
                let origin = vertices[i].map(f64::from);
                let a = core::array::from_fn::<_, 3, _>(|axis| {
                    f64::from(vertices[j][axis]) - origin[axis]
                });
                let b = core::array::from_fn::<_, 3, _>(|axis| {
                    f64::from(vertices[k][axis]) - origin[axis]
                });
                let mut normal = [
                    a[1] * b[2] - a[2] * b[1],
                    a[2] * b[0] - a[0] * b[2],
                    a[0] * b[1] - a[1] * b[0],
                ];
                let length = normal.iter().map(|value| value * value).sum::<f64>().sqrt();
                if length <= span * span * 1e-10 {
                    continue;
                }
                for component in &mut normal {
                    *component /= length;
                }
                let mut positive = false;
                let mut negative = false;
                for vertex in vertices {
                    let offset = core::array::from_fn::<_, 3, _>(|axis| {
                        f64::from(vertex[axis]) - origin[axis]
                    });
                    let distance = normal.iter().zip(offset).map(|(x, y)| x * y).sum::<f64>();
                    positive |= distance > tolerance;
                    negative |= distance < -tolerance;
                    if positive && negative {
                        break;
                    }
                }
                if positive && negative {
                    continue;
                }
                if positive {
                    for component in &mut normal {
                        *component = -*component;
                    }
                }
                if normals.iter().any(|known| {
                    normal
                        .iter()
                        .zip(known)
                        .map(|(x, y)| x * f64::from(*y))
                        .sum::<f64>()
                        > 1.0 - 1e-6
                }) {
                    continue;
                }
                normals.push(normal.map(|value| value as f32));
            }
        }
    }
    normals
}

pub(crate) fn convex_edges(vertices: &[[f32; 3]], normals: &[[f32; 3]]) -> Vec<[u32; 2]> {
    let tolerance = convex_span(vertices) * 1e-6;
    let planes = normals
        .iter()
        .map(|normal| {
            let offset = vertices
                .iter()
                .map(|vertex| {
                    normal
                        .iter()
                        .zip(vertex)
                        .map(|(a, b)| f64::from(*a) * f64::from(*b))
                        .sum::<f64>()
                })
                .fold(f64::NEG_INFINITY, f64::max);
            (*normal, offset)
        })
        .collect::<Vec<_>>();
    let mut edges = Vec::new();
    for i in 0..vertices.len() {
        for j in (i + 1)..vertices.len() {
            let separation = vertices[i]
                .iter()
                .zip(vertices[j])
                .map(|(a, b)| {
                    let delta = f64::from(*a) - f64::from(b);
                    delta * delta
                })
                .sum::<f64>();
            if separation <= tolerance * tolerance {
                continue;
            }
            let shared = planes
                .iter()
                .filter(|(normal, offset)| {
                    [i, j].iter().all(|&index| {
                        let projection = normal
                            .iter()
                            .zip(vertices[index])
                            .map(|(a, b)| f64::from(*a) * f64::from(b))
                            .sum::<f64>();
                        (projection - offset).abs() <= tolerance
                    })
                })
                .take(2)
                .count();
            if shared == 2 {
                edges.push([i as u32, j as u32]);
            }
        }
    }
    edges
}

fn convex_radius(vertices: &[[f32; 3]]) -> Option<f32> {
    if vertices.len() < 4 || vertices.len() > u32::MAX as usize {
        return None;
    }
    let mut radius = 0.0_f32;
    for vertex in vertices {
        if vertex.iter().any(|value| !value.is_finite()) {
            return None;
        }
        radius = radius.max(vertex[0].hypot(vertex[1]).hypot(vertex[2]));
    }
    if !radius.is_finite() || radius <= 0.0 {
        return None;
    }
    let origin = vertices[0];
    let mut directions = [[0.0_f64; 3]; 3];
    let mut rank = 0;
    let tolerance = convex_span(vertices) * 1e-6;
    for vertex in &vertices[1..] {
        let mut vector = [
            f64::from(vertex[0]) - f64::from(origin[0]),
            f64::from(vertex[1]) - f64::from(origin[1]),
            f64::from(vertex[2]) - f64::from(origin[2]),
        ];
        for basis in directions.iter().take(rank) {
            let dot = vector.iter().zip(basis).map(|(a, b)| a * b).sum::<f64>();
            for axis in 0..3 {
                vector[axis] -= dot * basis[axis];
            }
        }
        let length = vector.iter().map(|value| value * value).sum::<f64>().sqrt();
        if length > tolerance {
            for axis in &mut vector {
                *axis /= length;
            }
            directions[rank] = vector;
            rank += 1;
            if rank == 3 {
                return Some(radius);
            }
        }
    }
    None
}

fn convex_span(vertices: &[[f32; 3]]) -> f64 {
    let Some(origin) = vertices.first() else {
        return 0.0;
    };
    vertices
        .iter()
        .map(|vertex| {
            vertex
                .iter()
                .zip(origin)
                .map(|(value, base)| {
                    let delta = f64::from(*value) - f64::from(*base);
                    delta * delta
                })
                .sum::<f64>()
                .sqrt()
        })
        .fold(0.0, f64::max)
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GpuRigidShapeData {
    kind: [u32; 4],
    feature_counts: [u32; 4],
    dimensions: [f32; 4],
}

impl GpuRigidShapeData {
    pub(crate) fn kind(&self) -> u32 {
        self.kind[0]
    }
}

#[cfg(test)]
mod tests {
    use super::{GpuRigidShape, convex_edges, convex_face_normals, mesh_bvh_nodes};

    #[test]
    fn polyline_rejects_invalid_segment_geometry() {
        let valid = GpuRigidShape::Polyline {
            vertices: vec![[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            segments: vec![[0, 1]],
        };
        assert_eq!(valid.bounding_radius(), Some(1.0));
        for segments in [vec![], vec![[0, 0]], vec![[0, 2]]] {
            let invalid = GpuRigidShape::Polyline {
                vertices: vec![[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                segments,
            };
            assert!(invalid.bounding_radius().is_none());
        }
        for vertices in [vec![[0.0; 3]; 2], vec![[f32::NAN; 3], [1.0; 3]]] {
            assert!(
                GpuRigidShape::Polyline {
                    vertices,
                    segments: vec![[0, 1]]
                }
                .bounding_radius()
                .is_none()
            );
        }
    }

    #[test]
    fn heightfield_conversion_preserves_grid_and_rejects_f32_overflow() {
        use crate::mesh::HeightFieldGeometry;
        use nalgebra::Vector3;
        let geometry = HeightFieldGeometry::new(
            2,
            3,
            vec![0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
            Vector3::new(4.0, 2.0, 2.0),
        )
        .unwrap();
        let GpuRigidShape::TriangleMesh {
            vertices,
            triangles,
        } = GpuRigidShape::from_heightfield(&geometry).unwrap()
        else {
            panic!("expected triangle surface");
        };
        assert_eq!(
            vertices,
            vec![
                [-2.0, -1.0, 0.0],
                [0.0, -1.0, 1.0],
                [2.0, -1.0, 2.0],
                [-2.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [2.0, 1.0, 2.0]
            ]
        );
        assert_eq!(triangles, geometry.mesh().triangles());
        assert_eq!(triangles, vec![[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4]]);
        let huge =
            HeightFieldGeometry::new(2, 2, vec![1e40; 4], Vector3::new(2.0, 2.0, 1.0)).unwrap();
        assert!(GpuRigidShape::from_heightfield(&huge).is_err());
    }

    #[test]
    fn bounding_spheres_contain_valid_primitives() {
        assert_eq!(
            GpuRigidShape::Sphere { radius: 2.0 }.bounding_radius(),
            Some(2.0)
        );
        assert_eq!(
            GpuRigidShape::Box {
                half_extents: [1.0, 2.0, 2.0],
            }
            .bounding_radius(),
            Some(3.0)
        );
        assert!(
            GpuRigidShape::Sphere { radius: 0.0 }
                .bounding_radius()
                .is_none()
        );
        assert!(
            GpuRigidShape::Box {
                half_extents: [1.0, f32::NAN, 1.0],
            }
            .bounding_radius()
            .is_none()
        );
        assert_eq!(
            GpuRigidShape::Capsule {
                radius: 0.25,
                half_length: 1.5,
            }
            .bounding_radius(),
            Some(1.75)
        );
        assert!(
            GpuRigidShape::Capsule {
                radius: 0.25,
                half_length: -1.0,
            }
            .bounding_radius()
            .is_none()
        );
        for shape in [
            GpuRigidShape::Cylinder {
                radius: 3.0,
                half_length: 4.0,
            },
            GpuRigidShape::Cone {
                radius: 3.0,
                half_length: 4.0,
            },
        ] {
            assert_eq!(shape.bounding_radius(), Some(5.0));
        }
    }

    #[test]
    fn convex_requires_finite_three_dimensional_vertices() {
        let tetrahedron = GpuRigidShape::Convex {
            vertices: vec![
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [-1.0, -1.0, -1.0],
            ],
        };
        assert_eq!(tetrahedron.bounding_radius(), Some(3.0_f32.sqrt()));
        for vertices in [
            vec![[0.0, 0.0, 0.0]; 4],
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [f32::NAN, 0.0, 1.0],
            ],
        ] {
            assert!(
                GpuRigidShape::Convex { vertices }
                    .bounding_radius()
                    .is_none()
            );
        }
    }

    #[test]
    fn triangle_mesh_requires_valid_indexed_faces() {
        let vertices = vec![[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]];
        let mesh = GpuRigidShape::TriangleMesh {
            vertices: vertices.clone(),
            triangles: vec![[0, 1, 2]],
        };
        assert_eq!(mesh.bounding_radius(), Some(2.0_f32.sqrt()));
        assert_eq!(mesh.packed(4, 0, 0, 9, 0).kind, [6, 4, 3, 9]);
        for triangles in [vec![], vec![[0, 1, 3]], vec![[0, 1, 1]]] {
            assert!(
                GpuRigidShape::TriangleMesh {
                    vertices: vertices.clone(),
                    triangles,
                }
                .bounding_radius()
                .is_none()
            );
        }
        assert!(
            GpuRigidShape::TriangleMesh {
                vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                triangles: vec![[0, 1, 2]],
            }
            .bounding_radius()
            .is_none()
        );
    }

    #[test]
    fn mesh_bvh_escape_links_skip_distant_triangles() {
        let mut vertices = Vec::new();
        let mut triangles = Vec::new();
        for x in [0.0, 10.0, 20.0, 30.0] {
            let first = vertices.len() as u32;
            vertices.extend([[x, 0.0, 0.0], [x + 1.0, 0.0, 0.0], [x, 1.0, 0.0]]);
            triangles.push([first, first + 1, first + 2]);
        }
        let nodes = mesh_bvh_nodes(&vertices, &triangles);
        assert_eq!(nodes.len(), 7);
        assert_eq!(nodes[0].lower, [0.0, 0.0, 0.0]);
        assert_eq!(nodes[0].upper, [31.0, 1.0, 0.0]);
        assert_eq!(nodes[0].escape, nodes.len());
        let mut visited = Vec::new();
        let mut cursor = 0;
        while cursor < nodes.len() {
            let node = nodes[cursor];
            assert!(node.escape > cursor && node.escape <= nodes.len());
            if node.lower[0] > 10.5 || node.upper[0] < 10.5 {
                cursor = node.escape;
                continue;
            }
            if let Some(triangle) = node.triangle {
                visited.push(triangle);
            }
            cursor += 1;
        }
        assert_eq!(visited, [1]);
    }

    #[test]
    fn convex_face_normals_follow_supporting_planes() {
        let cube = [-1.0, 1.0]
            .into_iter()
            .flat_map(|x| {
                [-1.0, 1.0]
                    .into_iter()
                    .flat_map(move |y| [-1.0, 1.0].into_iter().map(move |z| [x, y, z]))
            })
            .collect::<Vec<_>>();
        let normals = convex_face_normals(&cube);
        assert_eq!(normals.len(), 6);
        assert_eq!(convex_edges(&cube, &normals).len(), 12);
        let translated = cube
            .iter()
            .map(|vertex| [vertex[0] + 1_000_000.0, vertex[1], vertex[2]])
            .collect::<Vec<_>>();
        assert!(
            GpuRigidShape::Convex {
                vertices: translated.clone()
            }
            .bounding_radius()
            .is_some()
        );
        assert_eq!(convex_face_normals(&translated).len(), 6);
        assert_eq!(
            convex_edges(&translated, &convex_face_normals(&translated)).len(),
            12
        );
        let mut with_interior = cube.clone();
        for index in 0..32 {
            let fraction = index as f32 / 32.0;
            with_interior.push([fraction * 0.5, fraction * 0.25, -fraction * 0.5]);
        }
        assert_eq!(
            convex_edges(&with_interior, &convex_face_normals(&with_interior)).len(),
            12
        );
        for expected in [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ] {
            assert!(normals.iter().any(|normal| {
                normal
                    .iter()
                    .zip(expected)
                    .all(|(a, b)| (a - b).abs() < 1e-6)
            }));
        }
        let tetrahedron = [
            [2.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [2.0, 0.0, 2.0],
        ];
        let normals = convex_face_normals(&tetrahedron);
        assert_eq!(normals.len(), 4);
        assert_eq!(convex_edges(&tetrahedron, &normals).len(), 6);
        assert!(normals.iter().any(|normal| {
            normal
                .iter()
                .all(|component| (*component - 3.0_f32.recip().sqrt()).abs() < 1e-6)
        }));
    }
}
