//! Validated polyline, triangle-mesh, and heightfield collision geometry.

use nalgebra::Vector3;

use crate::convex::{ConvexGeometry, ConvexGeometryError};

/// Resolves one model mesh reference into one or more convex collision parts.
///
/// The resolver owns URI, filesystem, and file-format policy. Returned vertices
/// must already include the requested non-uniform scale.
pub trait ConvexMeshResolver {
    /// Resolve `filename` into validated convex parts.
    fn resolve(&mut self, filename: &str, scale: [f64; 3]) -> Result<Vec<ConvexGeometry>, String>;
}

impl<F> ConvexMeshResolver for F
where
    F: FnMut(&str, [f64; 3]) -> Result<Vec<ConvexGeometry>, String>,
{
    fn resolve(&mut self, filename: &str, scale: [f64; 3]) -> Result<Vec<ConvexGeometry>, String> {
        self(filename, scale)
    }
}

/// Half-thickness used to give surface triangles a robust support volume.
pub const TRIANGLE_HALF_THICKNESS: f64 = 1e-4;

/// Invalid or degenerate mesh input.
#[derive(Debug, thiserror::Error, Clone, Copy, PartialEq, Eq)]
pub enum MeshGeometryError {
    /// Vertices, indices, dimensions, or scale do not describe finite geometry.
    #[error("invalid mesh geometry")]
    Invalid,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct LocalAabb {
    lower: Vector3<f64>,
    upper: Vector3<f64>,
}

impl LocalAabb {
    fn union(self, other: Self) -> Self {
        Self {
            lower: self.lower.inf(&other.lower),
            upper: self.upper.sup(&other.upper),
        }
    }

    fn from_nonempty_slice(points: &[Vector3<f64>]) -> Self {
        let first = points.first().copied().unwrap_or_else(Vector3::zeros);
        points.iter().copied().skip(1).fold(
            Self {
                lower: first,
                upper: first,
            },
            |bounds, point| Self {
                lower: bounds.lower.inf(&point),
                upper: bounds.upper.sup(&point),
            },
        )
    }

    fn overlaps(self, other: Self) -> bool {
        (0..3).all(|axis| {
            self.lower[axis] <= other.upper[axis] && other.lower[axis] <= self.upper[axis]
        })
    }

    fn centroid(self) -> Vector3<f64> {
        (self.lower + self.upper) * 0.5
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct BvhNode {
    bounds: LocalAabb,
    children: Option<[usize; 2]>,
    range: [usize; 2],
}

#[derive(Debug, Clone, PartialEq)]
struct PrimitiveBvh {
    nodes: Vec<BvhNode>,
    indices: Vec<usize>,
    primitive_bounds: Vec<LocalAabb>,
}

impl PrimitiveBvh {
    const LEAF_SIZE: usize = 4;

    fn new(bounds: &[LocalAabb]) -> Self {
        let mut indices = (0..bounds.len()).collect::<Vec<_>>();
        let mut nodes = Vec::with_capacity(bounds.len().saturating_mul(2));
        let count = indices.len();
        let _root = Self::build_node(bounds, &mut indices, 0, count, &mut nodes);
        Self {
            nodes,
            indices,
            primitive_bounds: bounds.to_vec(),
        }
    }

    fn build_node(
        primitive_bounds: &[LocalAabb],
        indices: &mut [usize],
        start: usize,
        end: usize,
        nodes: &mut Vec<BvhNode>,
    ) -> usize {
        let first = primitive_bounds[indices[start]];
        let bounds = indices[(start + 1)..end]
            .iter()
            .fold(first, |bounds, &index| {
                bounds.union(primitive_bounds[index])
            });
        let node_index = nodes.len();
        nodes.push(BvhNode {
            bounds,
            children: None,
            range: [start, end],
        });
        if end - start <= Self::LEAF_SIZE {
            return node_index;
        }
        let extent = bounds.upper - bounds.lower;
        let axis = if extent.x >= extent.y && extent.x >= extent.z {
            0
        } else if extent.y >= extent.z {
            1
        } else {
            2
        };
        indices[start..end].sort_unstable_by(|left, right| {
            primitive_bounds[*left].centroid()[axis]
                .total_cmp(&primitive_bounds[*right].centroid()[axis])
        });
        let middle = start + (end - start) / 2;
        let left = Self::build_node(primitive_bounds, indices, start, middle, nodes);
        let right = Self::build_node(primitive_bounds, indices, middle, end, nodes);
        nodes[node_index].children = Some([left, right]);
        node_index
    }

    fn bounds(&self) -> LocalAabb {
        self.nodes[0].bounds
    }

    fn query(&self, bounds: LocalAabb) -> Vec<usize> {
        let mut result = Vec::new();
        let mut stack = vec![0];
        while let Some(index) = stack.pop() {
            let node = self.nodes[index];
            if !node.bounds.overlaps(bounds) {
                continue;
            }
            if let Some([left, right]) = node.children {
                stack.push(right);
                stack.push(left);
            } else {
                result.extend(
                    self.indices[node.range[0]..node.range[1]]
                        .iter()
                        .copied()
                        .filter(|&primitive| self.primitive_bounds[primitive].overlaps(bounds)),
                );
            }
        }
        result
    }
}

/// Indexed line segments in shape-local coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct PolylineGeometry {
    vertices: Vec<Vector3<f64>>,
    segments: Vec<[u32; 2]>,
    bvh: PrimitiveBvh,
}

impl PolylineGeometry {
    /// Validate and construct an indexed polyline.
    pub fn new(
        vertices: Vec<Vector3<f64>>,
        segments: Vec<[u32; 2]>,
    ) -> Result<Self, MeshGeometryError> {
        if vertices.len() < 2
            || segments.is_empty()
            || vertices
                .iter()
                .any(|vertex| vertex.iter().any(|value| !value.is_finite()))
        {
            return Err(MeshGeometryError::Invalid);
        }
        for segment in &segments {
            let [a, b] = segment.map(|index| index as usize);
            if a >= vertices.len()
                || b >= vertices.len()
                || a == b
                || (vertices[b] - vertices[a]).norm_squared() <= 1e-20
            {
                return Err(MeshGeometryError::Invalid);
            }
        }
        let bounds = segments
            .iter()
            .map(|segment| {
                let [a, b] = segment.map(|index| vertices[index as usize]);
                LocalAabb {
                    lower: a.inf(&b),
                    upper: a.sup(&b),
                }
            })
            .collect::<Vec<_>>();
        Ok(Self {
            vertices,
            segments,
            bvh: PrimitiveBvh::new(&bounds),
        })
    }

    /// Local vertices shared by all segments.
    pub fn vertices(&self) -> &[Vector3<f64>] {
        &self.vertices
    }

    /// Segment endpoint indices.
    pub fn segments(&self) -> &[[u32; 2]] {
        &self.segments
    }

    /// Resolve one segment to two local-space endpoints.
    pub fn segment_vertices(&self, index: usize) -> Option<[Vector3<f64>; 2]> {
        let segment = self.segments.get(index)?;
        Some(segment.map(|vertex| self.vertices[vertex as usize]))
    }

    /// Axis-aligned local bounds of the complete polyline.
    pub fn local_bounds(&self) -> (Vector3<f64>, Vector3<f64>) {
        let bounds = self.bvh.bounds();
        (bounds.lower, bounds.upper)
    }

    /// Segment indices whose local bounds overlap the query box.
    pub fn segments_in_aabb(&self, lower: Vector3<f64>, upper: Vector3<f64>) -> Vec<usize> {
        query_bvh(&self.bvh, lower, upper)
    }
}

/// Indexed triangle mesh in shape-local coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct TriangleMeshGeometry {
    vertices: Vec<Vector3<f64>>,
    triangles: Vec<[u32; 3]>,
    prisms: Vec<ConvexGeometry>,
    bvh: PrimitiveBvh,
}

impl TriangleMeshGeometry {
    /// Validate and construct an indexed triangle mesh.
    pub fn new(
        vertices: Vec<Vector3<f64>>,
        triangles: Vec<[u32; 3]>,
    ) -> Result<Self, MeshGeometryError> {
        if vertices.len() < 3
            || triangles.is_empty()
            || vertices
                .iter()
                .any(|vertex| vertex.iter().any(|value| !value.is_finite()))
        {
            return Err(MeshGeometryError::Invalid);
        }
        for triangle in &triangles {
            let [a, b, c] = triangle.map(|index| index as usize);
            if a >= vertices.len()
                || b >= vertices.len()
                || c >= vertices.len()
                || a == b
                || b == c
                || c == a
                || (vertices[b] - vertices[a])
                    .cross(&(vertices[c] - vertices[a]))
                    .norm_squared()
                    <= 1e-20
            {
                return Err(MeshGeometryError::Invalid);
            }
        }
        let prisms = triangles
            .iter()
            .map(|triangle| {
                triangle_prism(triangle.map(|vertex| vertices[vertex as usize]))
                    .map_err(|_| MeshGeometryError::Invalid)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let bounds = prisms
            .iter()
            .map(|prism| LocalAabb::from_nonempty_slice(&prism.vertices))
            .collect::<Vec<_>>();
        Ok(Self {
            vertices,
            triangles,
            prisms,
            bvh: PrimitiveBvh::new(&bounds),
        })
    }

    /// Local vertices shared by all triangles.
    pub fn vertices(&self) -> &[Vector3<f64>] {
        &self.vertices
    }

    /// Triangle vertex indices.
    pub fn triangles(&self) -> &[[u32; 3]] {
        &self.triangles
    }

    /// Resolve one triangle to three local-space vertices.
    pub fn triangle_vertices(&self, index: usize) -> Option<[Vector3<f64>; 3]> {
        let triangle = self.triangles.get(index)?;
        Some(triangle.map(|vertex| self.vertices[vertex as usize]))
    }

    /// Axis-aligned local bounds of the complete triangle mesh.
    pub fn local_bounds(&self) -> (Vector3<f64>, Vector3<f64>) {
        let bounds = self.bvh.bounds();
        (bounds.lower, bounds.upper)
    }

    /// Triangle indices whose thickened local bounds overlap the query box.
    pub fn triangles_in_aabb(&self, lower: Vector3<f64>, upper: Vector3<f64>) -> Vec<usize> {
        query_bvh(&self.bvh, lower, upper)
    }

    pub(crate) fn prisms(&self) -> &[ConvexGeometry] {
        &self.prisms
    }
}

fn query_bvh(bvh: &PrimitiveBvh, lower: Vector3<f64>, upper: Vector3<f64>) -> Vec<usize> {
    if lower.iter().any(|value| !value.is_finite())
        || upper.iter().any(|value| !value.is_finite())
        || (0..3).any(|axis| lower[axis] > upper[axis])
    {
        return Vec::new();
    }
    bvh.query(LocalAabb { lower, upper })
}

pub(crate) fn triangle_prism(
    triangle: [Vector3<f64>; 3],
) -> Result<ConvexGeometry, ConvexGeometryError> {
    let [a, b, c] = triangle;
    let normal = (b - a)
        .cross(&(c - a))
        .try_normalize(1e-12)
        .ok_or(ConvexGeometryError::Invalid)?;
    let edges = [b - a, c - b, a - c].map(|edge| edge.normalize());
    let offset = normal * TRIANGLE_HALF_THICKNESS;
    let vertices = vec![
        a + offset,
        b + offset,
        c + offset,
        a - offset,
        b - offset,
        c - offset,
    ];
    let mut face_normals = vec![normal];
    face_normals.extend(edges.map(|edge| edge.cross(&normal).normalize()));
    let mut edge_directions = edges.to_vec();
    edge_directions.push(normal);
    ConvexGeometry::new(vertices, face_normals, edge_directions)
}

/// Regular height samples converted to a z-up triangle mesh at construction.
#[derive(Debug, Clone, PartialEq)]
pub struct HeightFieldGeometry {
    rows: usize,
    columns: usize,
    heights: Vec<f64>,
    scale: Vector3<f64>,
    mesh: TriangleMeshGeometry,
}

impl HeightFieldGeometry {
    /// Construct a centered z-up heightfield from row-major samples.
    ///
    /// `scale.x` and `scale.y` are the full horizontal extents. `scale.z`
    /// multiplies each height sample.
    pub fn new(
        rows: usize,
        columns: usize,
        heights: Vec<f64>,
        scale: Vector3<f64>,
    ) -> Result<Self, MeshGeometryError> {
        let sample_count = rows
            .checked_mul(columns)
            .ok_or(MeshGeometryError::Invalid)?;
        if rows < 2
            || columns < 2
            || heights.len() != sample_count
            || heights.iter().any(|height| !height.is_finite())
            || scale
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
            || sample_count > u32::MAX as usize
        {
            return Err(MeshGeometryError::Invalid);
        }

        let mut vertices = Vec::with_capacity(sample_count);
        for row in 0..rows {
            let y = (row as f64 / (rows - 1) as f64 - 0.5) * scale.y;
            for column in 0..columns {
                let x = (column as f64 / (columns - 1) as f64 - 0.5) * scale.x;
                let z = heights[row * columns + column] * scale.z;
                vertices.push(Vector3::new(x, y, z));
            }
        }

        let triangle_capacity = (rows - 1)
            .checked_mul(columns - 1)
            .and_then(|count| count.checked_mul(2))
            .ok_or(MeshGeometryError::Invalid)?;
        let mut triangles = Vec::with_capacity(triangle_capacity);
        for row in 0..(rows - 1) {
            for column in 0..(columns - 1) {
                let a = u32::try_from(row * columns + column)
                    .map_err(|_| MeshGeometryError::Invalid)?;
                let b = a + 1;
                let d = u32::try_from((row + 1) * columns + column)
                    .map_err(|_| MeshGeometryError::Invalid)?;
                let c = d + 1;
                triangles.push([a, b, c]);
                triangles.push([a, c, d]);
            }
        }
        let mesh = TriangleMeshGeometry::new(vertices, triangles)?;
        Ok(Self {
            rows,
            columns,
            heights,
            scale,
            mesh,
        })
    }

    /// Number of height rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of height columns.
    pub fn columns(&self) -> usize {
        self.columns
    }

    /// Row-major height samples before z scaling.
    pub fn heights(&self) -> &[f64] {
        &self.heights
    }

    /// Full x/y extents and height multiplier.
    pub fn scale(&self) -> Vector3<f64> {
        self.scale
    }

    /// Converted triangle mesh used by collision detection.
    pub fn mesh(&self) -> &TriangleMeshGeometry {
        &self.mesh
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_mesh_indices_and_degenerate_triangles() {
        let vertices = vec![Vector3::zeros(), Vector3::x(), Vector3::y()];
        assert_eq!(
            TriangleMeshGeometry::new(vertices.clone(), vec![[0, 1, 3]]).unwrap_err(),
            MeshGeometryError::Invalid
        );
        assert_eq!(
            TriangleMeshGeometry::new(vertices, vec![[0, 1, 1]]).unwrap_err(),
            MeshGeometryError::Invalid
        );
    }

    #[test]
    fn validates_indexed_polyline_segments() {
        let vertices = vec![Vector3::zeros(), Vector3::x(), Vector3::y()];
        let polyline = PolylineGeometry::new(vertices.clone(), vec![[0, 1], [1, 2]]).unwrap();
        assert_eq!(
            polyline.segment_vertices(1),
            Some([Vector3::x(), Vector3::y()])
        );
        assert_eq!(
            PolylineGeometry::new(vertices.clone(), vec![[0, 3]]).unwrap_err(),
            MeshGeometryError::Invalid
        );
        assert_eq!(
            PolylineGeometry::new(vertices, vec![[1, 1]]).unwrap_err(),
            MeshGeometryError::Invalid
        );
    }

    #[test]
    fn internal_bvhs_return_only_overlapping_primitives() {
        let polyline = PolylineGeometry::new(
            vec![
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(10.0, 0.0, 0.0),
                Vector3::new(11.0, 0.0, 0.0),
                Vector3::new(20.0, 0.0, 0.0),
                Vector3::new(21.0, 0.0, 0.0),
            ],
            vec![[0, 1], [2, 3], [4, 5]],
        )
        .unwrap();
        assert_eq!(
            polyline.segments_in_aabb(Vector3::new(9.5, -0.1, -0.1), Vector3::new(11.5, 0.1, 0.1)),
            vec![1]
        );

        let mesh = TriangleMeshGeometry::new(
            vec![
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(0.0, 1.0, 0.0),
                Vector3::new(10.0, 0.0, 0.0),
                Vector3::new(11.0, 0.0, 0.0),
                Vector3::new(10.0, 1.0, 0.0),
                Vector3::new(20.0, 0.0, 0.0),
                Vector3::new(21.0, 0.0, 0.0),
                Vector3::new(20.0, 1.0, 0.0),
            ],
            vec![[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        )
        .unwrap();
        assert_eq!(
            mesh.triangles_in_aabb(Vector3::new(9.5, -0.1, -0.1), Vector3::new(11.5, 1.1, 0.1)),
            vec![1]
        );
        assert!(
            mesh.triangles_in_aabb(Vector3::repeat(1.0), Vector3::repeat(-1.0))
                .is_empty()
        );
    }

    #[test]
    fn heightfield_builds_upward_wound_centered_triangles() {
        let field = HeightFieldGeometry::new(
            2,
            3,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            Vector3::new(4.0, 2.0, 0.5),
        )
        .unwrap();
        assert_eq!(field.mesh().triangles().len(), 4);
        assert_eq!(field.mesh().vertices()[0], Vector3::new(-2.0, -1.0, 0.0));
        let [a, b, c] = field.mesh().triangle_vertices(0).unwrap();
        assert!((b - a).cross(&(c - a)).z > 0.0);
    }

    #[test]
    fn triangle_prism_has_finite_thickness_and_topology() {
        let prism = triangle_prism([Vector3::zeros(), Vector3::x(), Vector3::y()]).unwrap();
        assert_eq!(prism.vertices.len(), 6);
        assert_eq!(prism.face_normals.len(), 4);
        assert_eq!(prism.edge_directions.len(), 4);
        let extent = prism
            .vertices
            .iter()
            .map(|vertex| vertex.z)
            .fold((f64::INFINITY, -f64::INFINITY), |range, z| {
                (range.0.min(z), range.1.max(z))
            });
        assert!((extent.1 - extent.0 - TRIANGLE_HALF_THICKNESS * 2.0).abs() < 1e-12);
    }
}
