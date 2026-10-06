//! Deterministic triangle-mesh surface samples for rigid-MPM coupling.

use std::collections::{BTreeMap, BTreeSet};

use nalgebra::Vector3;

/// A local surface sample tied to a source triangle by barycentric weights.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TriangleSurfaceSample {
    /// Point in the mesh's local frame.
    pub point: Vector3<f64>,
    /// Index of the source triangle.
    pub triangle_id: u32,
    /// Nonnegative weights for the triangle's three vertices.
    pub barycentric: [f64; 3],
}

/// Invalid triangle mesh or an explicit sample-count limit was reached.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SurfaceSamplingError {
    /// The mesh, spacing, or sample limit is invalid.
    #[error("invalid triangle mesh sampling input")]
    InvalidInput,
    /// The requested sampling exceeds the caller's sample limit.
    #[error("triangle mesh surface sample capacity exceeded")]
    Capacity,
}

/// Invalid closed mesh or a volume-sampling work or output limit was reached.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum VolumeSamplingError {
    /// Geometry, spacing, or limits are invalid; the mesh must be a closed oriented manifold.
    #[error("invalid closed triangle mesh volume sampling input")]
    InvalidInput,
    /// The candidate lattice or accepted samples exceed the caller's limits.
    #[error("triangle mesh volume sampling capacity exceeded")]
    Capacity,
}

/// Place deterministic lattice-center samples inside a closed triangle mesh.
///
/// Each undirected edge must occur exactly twice with opposing directions. A
/// non-self-intersecting surface is required. The absolute solid angle handles
/// either consistent overall winding. `max_candidates` bounds the full AABB
/// lattice scan and `max_samples` bounds output allocation. Returned positions
/// are mesh-local and can initialize CPU or GPU MPM particles.
pub fn sample_closed_mesh_volume(
    vertices: &[Vector3<f64>],
    triangles: &[[u32; 3]],
    spacing: f64,
    max_candidates: usize,
    max_samples: usize,
) -> Result<Vec<Vector3<f64>>, VolumeSamplingError> {
    if vertices.is_empty()
        || triangles.is_empty()
        || !spacing.is_finite()
        || spacing <= 0.0
        || max_candidates == 0
        || max_samples == 0
        || vertices
            .iter()
            .any(|vertex| vertex.iter().any(|component| !component.is_finite()))
    {
        return Err(VolumeSamplingError::InvalidInput);
    }
    let mut edges = BTreeMap::<(u32, u32), [u32; 2]>::new();
    for triangle in triangles {
        if triangle.iter().any(|&id| id as usize >= vertices.len())
            || triangle[0] == triangle[1]
            || triangle[1] == triangle[2]
            || triangle[2] == triangle[0]
        {
            return Err(VolumeSamplingError::InvalidInput);
        }
        let [a, b, c] = triangle.map(|id| vertices[id as usize]);
        let area2 = (b - a).cross(&(c - a)).norm_squared();
        if !area2.is_finite() || area2 <= 1e-24 {
            return Err(VolumeSamplingError::InvalidInput);
        }
        for (a, b) in [
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ] {
            let entry = edges.entry((a.min(b), a.max(b))).or_default();
            let orientation = usize::from(a > b);
            entry[orientation] = entry[orientation]
                .checked_add(1)
                .ok_or(VolumeSamplingError::InvalidInput)?;
        }
    }
    if edges.values().any(|count| *count != [1, 1]) {
        return Err(VolumeSamplingError::InvalidInput);
    }
    let mut lower = vertices[0];
    let mut upper = vertices[0];
    for vertex in &vertices[1..] {
        lower = lower.inf(vertex);
        upper = upper.sup(vertex);
    }
    let mut counts = [0usize; 3];
    let mut candidate_count = 1usize;
    for axis in 0..3 {
        let count = ((upper[axis] - lower[axis]) / spacing).ceil();
        if !count.is_finite() || count < 1.0 || count > max_candidates as f64 {
            return Err(VolumeSamplingError::Capacity);
        }
        counts[axis] = count as usize;
        candidate_count = candidate_count
            .checked_mul(counts[axis])
            .filter(|&count| count <= max_candidates)
            .ok_or(VolumeSamplingError::Capacity)?;
    }
    let mut samples = Vec::new();
    for x in 0..counts[0] {
        for y in 0..counts[1] {
            for z in 0..counts[2] {
                let point = Vector3::new(
                    lower.x + (x as f64 + 0.5) * spacing,
                    lower.y + (y as f64 + 0.5) * spacing,
                    lower.z + (z as f64 + 0.5) * spacing,
                );
                if point.x >= upper.x || point.y >= upper.y || point.z >= upper.z {
                    continue;
                }
                let mut angle = 0.0;
                for triangle in triangles {
                    let [a, b, c] = triangle.map(|id| vertices[id as usize] - point);
                    let la = a.norm();
                    let lb = b.norm();
                    let lc = c.norm();
                    let numerator = a.dot(&b.cross(&c));
                    let denominator =
                        la * lb * lc + a.dot(&b) * lc + b.dot(&c) * la + c.dot(&a) * lb;
                    angle += 2.0 * numerator.atan2(denominator);
                }
                if angle.abs() > core::f64::consts::TAU {
                    if samples.len() == max_samples {
                        return Err(VolumeSamplingError::Capacity);
                    }
                    samples.push(point);
                }
            }
        }
    }
    Ok(samples)
}

/// Sample vertices, unique edges, and triangle interiors at a target spacing.
///
/// Shared vertices and edges are emitted only once. The maximum distance
/// between adjacent barycentric grid samples along a triangle edge is at most
/// `spacing`. `max_samples` is a hard memory bound, not a target count.
pub fn sample_triangle_mesh(
    vertices: &[Vector3<f64>],
    triangles: &[[u32; 3]],
    spacing: f64,
    max_samples: usize,
) -> Result<Vec<TriangleSurfaceSample>, SurfaceSamplingError> {
    if vertices.is_empty()
        || triangles.is_empty()
        || !spacing.is_finite()
        || spacing <= 0.0
        || max_samples == 0
        || vertices
            .iter()
            .any(|vertex| vertex.iter().any(|component| !component.is_finite()))
    {
        return Err(SurfaceSamplingError::InvalidInput);
    }
    for triangle in triangles {
        if triangle.iter().any(|id| *id as usize >= vertices.len())
            || triangle[0] == triangle[1]
            || triangle[1] == triangle[2]
            || triangle[2] == triangle[0]
        {
            return Err(SurfaceSamplingError::InvalidInput);
        }
        let a = vertices[triangle[0] as usize];
        let b = vertices[triangle[1] as usize];
        let c = vertices[triangle[2] as usize];
        let cross = (b - a).cross(&(c - a));
        if !cross.norm_squared().is_finite() || cross.norm_squared() <= 1e-24 {
            return Err(SurfaceSamplingError::InvalidInput);
        }
    }
    let mut samples = Vec::new();
    let mut visited_vertices = BTreeSet::new();
    let mut visited_edges = BTreeSet::new();
    for (triangle_index, triangle) in triangles.iter().enumerate() {
        let triangle_id =
            u32::try_from(triangle_index).map_err(|_| SurfaceSamplingError::Capacity)?;
        let points = triangle.map(|id| vertices[id as usize]);
        for local in 0..3 {
            if visited_vertices.insert(triangle[local]) {
                let mut barycentric = [0.0; 3];
                barycentric[local] = 1.0;
                push_sample(
                    &mut samples,
                    max_samples,
                    TriangleSurfaceSample {
                        point: points[local],
                        triangle_id,
                        barycentric,
                    },
                )?;
            }
        }
        for (local_a, local_b) in [(0usize, 1usize), (1, 2), (2, 0)] {
            let a = triangle[local_a];
            let b = triangle[local_b];
            if !visited_edges.insert((a.min(b), a.max(b))) {
                continue;
            }
            let subdivisions = subdivision_count(
                (points[local_b] - points[local_a]).norm(),
                spacing,
                max_samples,
            )?;
            for step in 1..subdivisions {
                let t = step as f64 / subdivisions as f64;
                let mut barycentric = [0.0; 3];
                barycentric[local_a] = 1.0 - t;
                barycentric[local_b] = t;
                push_sample(
                    &mut samples,
                    max_samples,
                    TriangleSurfaceSample {
                        point: points[local_a] * (1.0 - t) + points[local_b] * t,
                        triangle_id,
                        barycentric,
                    },
                )?;
            }
        }
        let longest_edge = (points[1] - points[0])
            .norm()
            .max((points[2] - points[1]).norm())
            .max((points[0] - points[2]).norm());
        let subdivisions = subdivision_count(longest_edge, spacing, max_samples)?;
        for i in 1..subdivisions {
            for j in 1..subdivisions - i {
                let weight_b = i as f64 / subdivisions as f64;
                let weight_c = j as f64 / subdivisions as f64;
                let weight_a = 1.0 - weight_b - weight_c;
                let barycentric = [weight_a, weight_b, weight_c];
                push_sample(
                    &mut samples,
                    max_samples,
                    TriangleSurfaceSample {
                        point: points[0] * weight_a + points[1] * weight_b + points[2] * weight_c,
                        triangle_id,
                        barycentric,
                    },
                )?;
            }
        }
    }
    Ok(samples)
}

fn subdivision_count(
    length: f64,
    spacing: f64,
    max_samples: usize,
) -> Result<usize, SurfaceSamplingError> {
    let count = (length / spacing).ceil().max(1.0);
    if !count.is_finite() || count > max_samples as f64 || count > usize::MAX as f64 {
        return Err(SurfaceSamplingError::Capacity);
    }
    Ok(count as usize)
}

fn push_sample(
    samples: &mut Vec<TriangleSurfaceSample>,
    max_samples: usize,
    sample: TriangleSurfaceSample,
) -> Result<(), SurfaceSamplingError> {
    if samples.len() >= max_samples {
        return Err(SurfaceSamplingError::Capacity);
    }
    samples.push(sample);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cube() -> ([Vector3<f64>; 8], [[u32; 3]; 12]) {
        let vertices = [
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(1.0, 0.0, 1.0),
            Vector3::new(1.0, 1.0, 1.0),
            Vector3::new(0.0, 1.0, 1.0),
        ];
        let triangles = [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ];
        (vertices, triangles)
    }

    #[test]
    fn closed_mesh_volume_samples_lattice_and_reversed_winding() {
        let (vertices, triangles) = cube();
        let samples = sample_closed_mesh_volume(&vertices, &triangles, 0.25, 64, 64).unwrap();
        assert_eq!(samples.len(), 64);
        assert_eq!(samples[0], Vector3::new(0.125, 0.125, 0.125));
        assert_eq!(samples[63], Vector3::new(0.875, 0.875, 0.875));
        let reversed = triangles.map(|[a, b, c]| [a, c, b]);
        assert_eq!(
            samples,
            sample_closed_mesh_volume(&vertices, &reversed, 0.25, 64, 64).unwrap()
        );
        let mut translated = vertices;
        for vertex in &mut translated {
            vertex.x += 2.0;
        }
        let mut joined_vertices = vertices.to_vec();
        joined_vertices.extend(translated);
        let mut joined_triangles = triangles.to_vec();
        joined_triangles.extend(triangles.map(|ids| ids.map(|id| id + 8)));
        let joined =
            sample_closed_mesh_volume(&joined_vertices, &joined_triangles, 0.25, 192, 128).unwrap();
        assert_eq!(joined.len(), 128);
        assert!(joined.iter().all(|point| point.x < 1.0 || point.x > 2.0));
    }

    #[test]
    fn closed_mesh_volume_rejects_open_or_badly_oriented_surfaces_and_limits() {
        let (vertices, triangles) = cube();
        assert_eq!(
            sample_closed_mesh_volume(&vertices, &triangles[..11], 0.25, 64, 64),
            Err(VolumeSamplingError::InvalidInput)
        );
        let mut reversed_face = triangles;
        reversed_face[0].swap(1, 2);
        assert_eq!(
            sample_closed_mesh_volume(&vertices, &reversed_face, 0.25, 64, 64),
            Err(VolumeSamplingError::InvalidInput)
        );
        assert_eq!(
            sample_closed_mesh_volume(&vertices, &triangles, 0.25, 63, 64),
            Err(VolumeSamplingError::Capacity)
        );
        assert_eq!(
            sample_closed_mesh_volume(&vertices, &triangles, 0.25, 64, 63),
            Err(VolumeSamplingError::Capacity)
        );
        assert_eq!(
            sample_closed_mesh_volume(&vertices, &triangles, f64::NAN, 64, 64),
            Err(VolumeSamplingError::InvalidInput)
        );
    }

    #[test]
    fn shared_edge_and_vertices_are_sampled_once_with_valid_barycentric_coordinates() {
        let vertices = [
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        ];
        let triangles = [[0, 1, 2], [0, 2, 3]];
        let samples = sample_triangle_mesh(&vertices, &triangles, 0.25, 1_000).unwrap();
        assert!(samples.len() > 4);
        let origin_count = samples
            .iter()
            .filter(|sample| sample.point == Vector3::zeros())
            .count();
        assert_eq!(origin_count, 1);
        let shared_midpoint_count = samples
            .iter()
            .filter(|sample| (sample.point - Vector3::new(0.5, 0.5, 0.0)).norm() < 1e-12)
            .count();
        assert_eq!(shared_midpoint_count, 1);
        for sample in &samples {
            let triangle = triangles[sample.triangle_id as usize];
            let reconstructed = (0..3)
                .map(|axis| vertices[triangle[axis] as usize] * sample.barycentric[axis])
                .sum::<Vector3<f64>>();
            assert!((reconstructed - sample.point).norm() < 1e-12);
            assert!((sample.barycentric.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        }
        assert_eq!(
            samples,
            sample_triangle_mesh(&vertices, &triangles, 0.25, 1_000).unwrap()
        );
    }

    #[test]
    fn thin_triangle_still_samples_all_vertices_and_edges() {
        let vertices = [
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.5, 1e-4, 0.0),
        ];
        let samples = sample_triangle_mesh(&vertices, &[[0, 1, 2]], 0.2, 100).unwrap();
        for vertex in vertices {
            assert!(samples.iter().any(|sample| sample.point == vertex));
        }
        assert!(samples.len() >= 8);
    }

    #[test]
    fn rejects_invalid_mesh_and_enforces_sample_budget() {
        let vertices = [
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        ];
        assert_eq!(
            sample_triangle_mesh(&vertices, &[[0, 1, 3]], 0.1, 100),
            Err(SurfaceSamplingError::InvalidInput)
        );
        assert_eq!(
            sample_triangle_mesh(&vertices, &[[0, 1, 2]], 0.1, 3),
            Err(SurfaceSamplingError::Capacity)
        );
        assert_eq!(
            sample_triangle_mesh(&vertices, &[[0, 1, 2]], 0.0, 100),
            Err(SurfaceSamplingError::InvalidInput)
        );
    }
}
