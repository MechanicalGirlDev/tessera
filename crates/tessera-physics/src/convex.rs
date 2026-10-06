//! Convex collision geometry independent of URDF and hull construction.

use nalgebra::{Vector2, Vector3};

/// Precomputed convex hull topology used by GPU shape-pair narrow phases.
#[derive(Debug, Clone, PartialEq)]
pub struct ConvexGeometry {
    /// Convex hull vertices in shape coordinates.
    pub vertices: Vec<Vector3<f64>>,
    /// Outward unit normals of unique hull faces.
    pub face_normals: Vec<Vector3<f64>>,
    /// Unit directions of unique hull edges.
    pub edge_directions: Vec<Vector3<f64>>,
}

/// Invalid or degenerate convex shape data.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ConvexGeometryError {
    /// A hull lacks topology or contains a non-finite or zero direction.
    #[error("invalid convex collision geometry")]
    Invalid,
}

impl ConvexGeometry {
    /// Validate finite vertices and normalized directions from a hull builder.
    pub fn new(
        vertices: Vec<Vector3<f64>>,
        face_normals: Vec<Vector3<f64>>,
        edge_directions: Vec<Vector3<f64>>,
    ) -> Result<Self, ConvexGeometryError> {
        if vertices.len() < 4
            || face_normals.len() < 4
            || edge_directions.len() < 3
            || vertices
                .iter()
                .any(|point| point.iter().any(|value| !value.is_finite()))
            || face_normals.iter().chain(&edge_directions).any(|axis| {
                axis.iter().any(|value| !value.is_finite())
                    || (axis.norm_squared() - 1.0).abs() > 1e-4
            })
        {
            return Err(ConvexGeometryError::Invalid);
        }
        Ok(Self {
            vertices,
            face_normals,
            edge_directions,
        })
    }
}

/// Test two world-space convex vertex sets with a bounded GJK simplex search.
/// Touching shapes count as intersecting.
pub fn gjk_intersects(a: &[Vector3<f64>], b: &[Vector3<f64>]) -> bool {
    if a.is_empty() || b.is_empty() {
        return false;
    }
    let support = |direction: Vector3<f64>| {
        let a_point = a
            .iter()
            .max_by(|x, y| x.dot(&direction).total_cmp(&y.dot(&direction)));
        let b_point = b
            .iter()
            .max_by(|x, y| x.dot(&-direction).total_cmp(&y.dot(&-direction)));
        match (a_point, b_point) {
            (Some(a_point), Some(b_point)) => *a_point - *b_point,
            _ => Vector3::zeros(),
        }
    };
    gjk_with_support(a[0] - b[0], support)
}

fn gjk_with_support(
    initial_direction: Vector3<f64>,
    support: impl Fn(Vector3<f64>) -> Vector3<f64>,
) -> bool {
    let mut direction = initial_direction;
    if direction.norm_squared() < 1e-20 {
        direction = Vector3::x();
    }
    let mut simplex = vec![support(direction)];
    direction = -simplex[0];
    for _ in 0..32 {
        if direction.norm_squared() < 1e-20 {
            return true;
        }
        let point = support(direction);
        let projection = point.dot(&direction);
        if projection < -1e-10 {
            return false;
        }
        if simplex
            .iter()
            .any(|vertex| (point - vertex).norm_squared() < 1e-20)
        {
            return projection.abs() < 1e-10;
        }
        simplex.push(point);
        if update_simplex(&mut simplex, &mut direction) {
            return true;
        }
    }
    false
}

/// Approximate one convex-to-sphere contact using exact spherical support.
/// Edge-distance penetration still needs a clipped hull manifold.
pub fn convex_sphere_contact(
    vertices: &[Vector3<f64>],
    normals: &[Vector3<f64>],
    center: Vector3<f64>,
    radius: f64,
) -> Option<ConvexContact> {
    if vertices.is_empty() || normals.is_empty() || radius <= 0.0 || !radius.is_finite() {
        return None;
    }
    let hull_center = vertices.iter().copied().sum::<Vector3<f64>>() / vertices.len() as f64;
    let support = |direction: Vector3<f64>| {
        let hull = vertices
            .iter()
            .max_by(|a, b| a.dot(&direction).total_cmp(&b.dot(&direction)))
            .copied()
            .unwrap_or_else(Vector3::zeros);
        let sphere = center - direction.try_normalize(1e-12).unwrap_or_else(Vector3::x) * radius;
        hull - sphere
    };
    if !gjk_with_support(hull_center - center, support) {
        return None;
    }
    let mut best_depth = f64::INFINITY;
    let mut best_normal = Vector3::x();
    for candidate in normals
        .iter()
        .copied()
        .chain(vertices.iter().map(|vertex| center - vertex))
        .chain(core::iter::once(center - hull_center))
    {
        let Some(axis) = candidate.try_normalize(1e-12) else {
            continue;
        };
        let (min_hull, max_hull) =
            vertices
                .iter()
                .fold((f64::INFINITY, -f64::INFINITY), |(min, max), point| {
                    let value = point.dot(&axis);
                    (min.min(value), max.max(value))
                });
        let sphere = center.dot(&axis);
        let forward = max_hull - (sphere - radius);
        let backward = (sphere + radius) - min_hull;
        let (depth, normal) = if forward < backward {
            (forward, axis)
        } else {
            (backward, -axis)
        };
        if depth < best_depth {
            best_depth = depth;
            best_normal = normal;
        }
    }
    if !best_depth.is_finite() || best_depth < -1e-8 {
        return None;
    }
    let hull_witness = vertices
        .iter()
        .max_by(|a, b| a.dot(&best_normal).total_cmp(&b.dot(&best_normal)))?;
    let sphere_witness = center - best_normal * radius;
    let witness_midpoint = (*hull_witness + sphere_witness) * 0.5;
    let center_midpoint = (hull_center + center) * 0.5;
    let point =
        center_midpoint + best_normal * (witness_midpoint - center_midpoint).dot(&best_normal);
    Some(ConvexContact {
        point,
        normal: best_normal,
        penetration: best_depth.max(0.0),
    })
}

/// Contact from GJK overlap and both hulls' face normals.
/// [`convex_edge_manifold`] also evaluates cross products of edge directions.
#[derive(Debug, Clone, Copy)]
pub struct ConvexContact {
    /// World-space contact point.
    pub point: Vector3<f64>,
    /// Unit direction from the first shape toward the second.
    pub normal: Vector3<f64>,
    /// Nonnegative penetration depth along the selected axis.
    pub penetration: f64,
}

#[derive(Debug, Clone, Copy)]
struct SupportVertex {
    difference: Vector3<f64>,
    point_a: Vector3<f64>,
    point_b: Vector3<f64>,
}

#[derive(Debug, Clone, Copy)]
struct EpaFace {
    indices: [usize; 3],
    normal: Vector3<f64>,
    distance: f64,
}

fn support_vertex(
    direction: Vector3<f64>,
    support_a: &impl Fn(Vector3<f64>) -> Vector3<f64>,
    support_b: &impl Fn(Vector3<f64>) -> Vector3<f64>,
) -> SupportVertex {
    let point_a = support_a(direction);
    let point_b = support_b(-direction);
    SupportVertex {
        difference: point_a - point_b,
        point_a,
        point_b,
    }
}

fn update_support_simplex(simplex: &mut Vec<SupportVertex>, direction: &mut Vector3<f64>) -> bool {
    let Some(a) = simplex.last().copied() else {
        return false;
    };
    let ao = -a.difference;
    match simplex.len() {
        2 => {
            let b = simplex[0];
            if (b.difference - a.difference).dot(&ao) > 0.0 {
                *direction = line_direction(a.difference, b.difference, ao);
            } else {
                simplex.clear();
                simplex.push(a);
                *direction = ao;
            }
        }
        3 => {
            let b = simplex[1];
            let c = simplex[0];
            let ab = b.difference - a.difference;
            let ac = c.difference - a.difference;
            let normal = ab.cross(&ac);
            if normal.cross(&ac).dot(&ao) > 0.0 {
                if ac.dot(&ao) > 0.0 {
                    simplex.clear();
                    simplex.extend([c, a]);
                    *direction = line_direction(a.difference, c.difference, ao);
                } else {
                    simplex.clear();
                    simplex.extend([b, a]);
                    *direction = line_direction(a.difference, b.difference, ao);
                }
            } else if ab.cross(&normal).dot(&ao) > 0.0 {
                simplex.clear();
                simplex.extend([b, a]);
                *direction = line_direction(a.difference, b.difference, ao);
            } else if normal.dot(&ao) > 0.0 {
                *direction = normal;
            } else {
                simplex.swap(0, 1);
                *direction = -normal;
            }
        }
        4 => {
            let b = simplex[2];
            let c = simplex[1];
            let d = simplex[0];
            for (p, q, opposite) in [(b, c, d), (c, d, b), (d, b, c)] {
                let mut normal =
                    (p.difference - a.difference).cross(&(q.difference - a.difference));
                if normal.dot(&(opposite.difference - a.difference)) > 0.0 {
                    normal = -normal;
                }
                if normal.dot(&ao) > 0.0 {
                    simplex.clear();
                    simplex.extend([q, p, a]);
                    *direction = normal;
                    return false;
                }
            }
            return true;
        }
        _ => return false,
    }
    false
}

fn epa_face(vertices: &[SupportVertex], mut indices: [usize; 3]) -> Option<EpaFace> {
    let a = vertices[indices[0]].difference;
    let b = vertices[indices[1]].difference;
    let c = vertices[indices[2]].difference;
    let mut normal = (b - a).cross(&(c - a)).try_normalize(1e-12)?;
    let mut distance = normal.dot(&a);
    if distance < 0.0 {
        indices.swap(1, 2);
        normal = -normal;
        distance = -distance;
    }
    Some(EpaFace {
        indices,
        normal,
        distance,
    })
}

fn triangle_barycentric(
    point: Vector3<f64>,
    a: Vector3<f64>,
    b: Vector3<f64>,
    c: Vector3<f64>,
) -> [f64; 3] {
    let ab = b - a;
    let ac = c - a;
    let ap = point - a;
    let d00 = ab.dot(&ab);
    let d01 = ab.dot(&ac);
    let d11 = ac.dot(&ac);
    let d20 = ap.dot(&ab);
    let d21 = ap.dot(&ac);
    let denominator = d00 * d11 - d01 * d01;
    if denominator.abs() < 1e-20 {
        return [1.0, 0.0, 0.0];
    }
    let v = (d11 * d20 - d01 * d21) / denominator;
    let w = (d00 * d21 - d01 * d20) / denominator;
    let u = 1.0 - v - w;
    let mut weights = [u.max(0.0), v.max(0.0), w.max(0.0)];
    let sum = weights.iter().sum::<f64>();
    if sum > 1e-20 {
        for weight in &mut weights {
            *weight /= sum;
        }
    }
    weights
}

fn epa_contact(
    mut vertices: Vec<SupportVertex>,
    support_a: &impl Fn(Vector3<f64>) -> Vector3<f64>,
    support_b: &impl Fn(Vector3<f64>) -> Vector3<f64>,
) -> Option<ConvexContact> {
    if vertices.len() != 4 {
        return None;
    }
    let mut faces = [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]]
        .into_iter()
        .filter_map(|indices| epa_face(&vertices, indices))
        .collect::<Vec<_>>();
    for _ in 0..64 {
        let (closest_index, closest) = faces
            .iter()
            .enumerate()
            .min_by(|(_, left), (_, right)| left.distance.total_cmp(&right.distance))?;
        let closest = *closest;
        let next = support_vertex(closest.normal, support_a, support_b);
        let next_distance = closest.normal.dot(&next.difference);
        let duplicate = vertices
            .iter()
            .any(|vertex| (vertex.difference - next.difference).norm_squared() < 1e-20);
        // A repeated support point does not imply convergence while the face is short of it.
        if next_distance - closest.distance > 1e-8 && duplicate {
            return None;
        }
        if next_distance - closest.distance <= 1e-8 || duplicate {
            let points = closest.indices.map(|index| vertices[index]);
            let weights = triangle_barycentric(
                closest.normal * closest.distance,
                points[0].difference,
                points[1].difference,
                points[2].difference,
            );
            let point_a = points
                .iter()
                .zip(weights)
                .map(|(vertex, weight)| vertex.point_a * weight)
                .sum::<Vector3<f64>>();
            let point_b = points
                .iter()
                .zip(weights)
                .map(|(vertex, weight)| vertex.point_b * weight)
                .sum::<Vector3<f64>>();
            return Some(ConvexContact {
                point: (point_a + point_b) * 0.5,
                normal: closest.normal,
                penetration: closest.distance.max(0.0),
            });
        }

        let new_index = vertices.len();
        vertices.push(next);
        let mut boundary = Vec::<(usize, usize)>::new();
        let mut kept = Vec::with_capacity(faces.len());
        for (index, face) in faces.into_iter().enumerate() {
            if index == closest_index
                || face
                    .normal
                    .dot(&(next.difference - vertices[face.indices[0]].difference))
                    > 1e-10
            {
                for edge in [
                    (face.indices[0], face.indices[1]),
                    (face.indices[1], face.indices[2]),
                    (face.indices[2], face.indices[0]),
                ] {
                    if let Some(reverse) = boundary
                        .iter()
                        .position(|candidate| *candidate == (edge.1, edge.0))
                    {
                        let _ = boundary.swap_remove(reverse);
                    } else {
                        boundary.push(edge);
                    }
                }
            } else {
                kept.push(face);
            }
        }
        kept.extend(
            boundary
                .into_iter()
                .filter_map(|(a, b)| epa_face(&vertices, [a, b, new_index])),
        );
        if kept.is_empty() {
            return None;
        }
        faces = kept;
    }
    None
}

fn support_axis_contact(
    center_a: Vector3<f64>,
    support_a: &impl Fn(Vector3<f64>) -> Vector3<f64>,
    center_b: Vector3<f64>,
    support_b: &impl Fn(Vector3<f64>) -> Vector3<f64>,
) -> Option<ConvexContact> {
    let evaluate = |candidate: Vector3<f64>| {
        let axis = candidate.try_normalize(1e-12)?;
        let max_a = support_a(axis).dot(&axis);
        let min_a = support_a(-axis).dot(&axis);
        let max_b = support_b(axis).dot(&axis);
        let min_b = support_b(-axis).dot(&axis);
        let forward = max_a - min_b;
        let backward = max_b - min_a;
        if forward < backward {
            Some((forward, axis))
        } else {
            Some((backward, -axis))
        }
    };
    let mut best = [
        Vector3::x(),
        Vector3::y(),
        Vector3::z(),
        center_b - center_a,
    ]
    .into_iter()
    .filter_map(evaluate)
    .min_by(|left, right| left.0.total_cmp(&right.0))?;
    const SAMPLE_COUNT: usize = 128;
    let golden_angle = core::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    for index in 0..SAMPLE_COUNT {
        let z = 1.0 - 2.0 * (index as f64 + 0.5) / SAMPLE_COUNT as f64;
        let radial = (1.0 - z * z).sqrt();
        let azimuth = golden_angle * index as f64;
        let candidate = Vector3::new(radial * azimuth.cos(), radial * azimuth.sin(), z);
        if let Some(value) = evaluate(candidate)
            && value.0 < best.0
        {
            best = value;
        }
    }
    let mut step = 0.2;
    for _ in 0..16 {
        let tangent_a = best
            .1
            .cross(&if best.1.x.abs() < 0.9 {
                Vector3::x()
            } else {
                Vector3::y()
            })
            .normalize();
        let tangent_b = best.1.cross(&tangent_a);
        let mut improved = false;
        for tangent in [tangent_a, -tangent_a, tangent_b, -tangent_b] {
            let candidate = (best.1 + tangent * step).normalize();
            if let Some(value) = evaluate(candidate)
                && value.0 < best.0
            {
                best = value;
                improved = true;
            }
        }
        if !improved {
            step *= 0.5;
        }
    }
    if best.0 < -1e-8 || !best.0.is_finite() {
        return None;
    }
    let point_a = support_a(best.1);
    let point_b = support_b(-best.1);
    Some(ConvexContact {
        point: (point_a + point_b) * 0.5,
        normal: best.1,
        penetration: best.0.max(0.0),
    })
}

/// Compute one contact between arbitrary convex support-map shapes.
///
/// The support callbacks return world-space extreme points. GJK determines
/// overlap and EPA reconstructs the minimum translation, normal, and witnesses.
/// Degenerate EPA polytopes fall back to refined support-axis projection.
pub fn support_map_contact(
    center_a: Vector3<f64>,
    support_a: impl Fn(Vector3<f64>) -> Vector3<f64>,
    center_b: Vector3<f64>,
    support_b: impl Fn(Vector3<f64>) -> Vector3<f64>,
) -> Option<ConvexContact> {
    let mut direction = center_a - center_b;
    if direction.norm_squared() < 1e-20 {
        direction = Vector3::x();
    }
    let mut simplex = vec![support_vertex(direction, &support_a, &support_b)];
    direction = -simplex[0].difference;
    for iteration in 0..64 {
        if direction.norm_squared() < 1e-20 {
            let normal = (center_b - center_a)
                .try_normalize(1e-12)
                .unwrap_or_else(Vector3::x);
            return Some(ConvexContact {
                point: (support_a(normal) + support_b(-normal)) * 0.5,
                normal,
                penetration: 0.0,
            });
        }
        let point = support_vertex(direction, &support_a, &support_b);
        if point.difference.dot(&direction) < -1e-10 {
            return None;
        }
        if simplex
            .iter()
            .any(|vertex| (vertex.difference - point.difference).norm_squared() < 1e-20)
        {
            let projection = point.difference.dot(&direction);
            if projection.abs() < 1e-10 {
                let normal = (center_b - center_a)
                    .try_normalize(1e-12)
                    .unwrap_or_else(Vector3::x);
                return Some(ConvexContact {
                    point: (support_a(normal) + support_b(-normal)) * 0.5,
                    normal,
                    penetration: 0.0,
                });
            }
            let basis = match iteration % 3 {
                0 => Vector3::x(),
                1 => Vector3::y(),
                _ => Vector3::z(),
            };
            let orthogonal = direction.cross(&basis).try_normalize(1e-12).or_else(|| {
                direction
                    .cross(&Vector3::new(basis.y, basis.z, basis.x))
                    .try_normalize(1e-12)
            })?;
            direction += orthogonal * direction.norm().max(1.0) * 1e-5;
            continue;
        }
        simplex.push(point);
        if update_support_simplex(&mut simplex, &mut direction) {
            return epa_contact(simplex, &support_a, &support_b)
                .or_else(|| support_axis_contact(center_a, &support_a, center_b, &support_b));
        }
    }
    None
}

/// World-space support point of a circular cylinder.
///
/// `axis` must be a unit vector from the bottom cap toward the top cap.
pub fn cylinder_support(
    center: Vector3<f64>,
    axis: Vector3<f64>,
    half_height: f64,
    radius: f64,
    direction: Vector3<f64>,
) -> Vector3<f64> {
    let axial = direction.dot(&axis);
    let radial = direction - axis * axial;
    let rim = radial
        .try_normalize(1e-12)
        .map_or_else(Vector3::zeros, |unit| unit * radius);
    center + axis * half_height.copysign(axial) + rim
}

/// World-space support point of a circular cone.
///
/// `axis` is a unit vector from the base centre toward the apex. The full
/// height is twice `half_height`.
pub fn cone_support(
    center: Vector3<f64>,
    axis: Vector3<f64>,
    half_height: f64,
    radius: f64,
    direction: Vector3<f64>,
) -> Vector3<f64> {
    let axial = direction.dot(&axis);
    let radial = direction - axis * axial;
    let radial_length = radial.norm();
    let apex = center + axis * half_height;
    if radial_length <= 1e-12 {
        return if axial >= 0.0 {
            apex
        } else {
            center - axis * half_height
        };
    }
    let base = center - axis * half_height + radial * (radius / radial_length);
    if apex.dot(&direction) > base.dot(&direction) {
        apex
    } else {
        base
    }
}

fn analytic_sphere_witness(
    sphere_center: Vector3<f64>,
    sphere_radius: f64,
    signed_distance: f64,
    outward: Vector3<f64>,
) -> Option<ConvexContact> {
    if !signed_distance.is_finite()
        || !sphere_radius.is_finite()
        || sphere_radius <= 0.0
        || signed_distance > sphere_radius + 1e-10
    {
        return None;
    }
    let surface = sphere_center - outward * signed_distance;
    let sphere_witness = sphere_center - outward * sphere_radius;
    Some(ConvexContact {
        point: (surface + sphere_witness) * 0.5,
        normal: outward,
        penetration: (sphere_radius - signed_distance).max(0.0),
    })
}

fn radial_frame(
    shape_center: Vector3<f64>,
    axis: Vector3<f64>,
    sphere_center: Vector3<f64>,
) -> (f64, f64, Vector3<f64>) {
    let relative = sphere_center - shape_center;
    let height = relative.dot(&axis);
    let radial = relative - axis * height;
    let length = radial.norm();
    let radial_direction = radial.try_normalize(1e-12).unwrap_or_else(|| {
        let reference = if axis.x.abs() < 0.9 {
            Vector3::x()
        } else {
            Vector3::y()
        };
        axis.cross(&reference).normalize()
    });
    (length, height, radial_direction)
}

pub(crate) fn cylinder_sphere_contact(
    center: Vector3<f64>,
    axis: Vector3<f64>,
    half_height: f64,
    radius: f64,
    sphere_center: Vector3<f64>,
    sphere_radius: f64,
) -> Option<ConvexContact> {
    let (radial, height, radial_direction) = radial_frame(center, axis, sphere_center);
    let radial_distance = radial - radius;
    let cap_distance = height.abs() - half_height;
    let radial_outside = radial_distance.max(0.0);
    let cap_outside = cap_distance.max(0.0);
    let cap_direction = axis * if height < 0.0 { -1.0 } else { 1.0 };
    let outside = radial_direction * radial_outside + cap_direction * cap_outside;
    let outside_length = outside.norm();
    let (signed_distance, outward) = if outside_length > 1e-12 {
        (outside_length, outside / outside_length)
    } else if radial_distance >= cap_distance {
        (radial_distance, radial_direction)
    } else {
        (cap_distance, cap_direction)
    };
    analytic_sphere_witness(sphere_center, sphere_radius, signed_distance, outward)
}

pub(crate) fn cone_sphere_contact(
    center: Vector3<f64>,
    axis: Vector3<f64>,
    half_height: f64,
    radius: f64,
    sphere_center: Vector3<f64>,
    sphere_radius: f64,
) -> Option<ConvexContact> {
    let (radial, height, radial_direction) = radial_frame(center, axis, sphere_center);
    let point = Vector2::new(radial, height);
    let base = Vector2::new(radial.clamp(0.0, radius), -half_height);
    let side_start = Vector2::new(radius, -half_height);
    let side_direction = Vector2::new(-radius, 2.0 * half_height);
    let side_t =
        ((point - side_start).dot(&side_direction) / side_direction.norm_squared()).clamp(0.0, 1.0);
    let side = side_start + side_direction * side_t;
    let base_distance_squared = (point - base).norm_squared();
    let side_distance_squared = (point - side).norm_squared();
    let use_base = base_distance_squared <= side_distance_squared;
    let closest = if use_base { base } else { side };
    let delta = point - closest;
    let distance = delta.norm();
    let edge_epsilon = half_height.max(radius) * 1e-6;
    let inside = height >= -half_height - edge_epsilon
        && height <= half_height + edge_epsilon
        && radial <= radius * (half_height - height) / (2.0 * half_height) + edge_epsilon;
    let base_normal = Vector2::new(0.0, -1.0);
    let side_normal = Vector2::new(2.0 * half_height, radius).normalize();
    let near_corner = base_distance_squared <= edge_epsilon * edge_epsilon
        && side_distance_squared <= edge_epsilon * edge_epsilon;
    let feature_normal = if near_corner {
        (base_normal + side_normal).normalize()
    } else if use_base {
        base_normal
    } else {
        side_normal
    };
    let normal_2d = if inside || distance <= edge_epsilon {
        feature_normal
    } else {
        delta / distance
    };
    let outward = radial_direction * normal_2d.x + axis * normal_2d.y;
    let signed_distance = if inside { -distance } else { distance };
    analytic_sphere_witness(sphere_center, sphere_radius, signed_distance, outward)
}

fn rounded_support(
    vertices: &[Vector3<f64>],
    radius: f64,
    direction: Vector3<f64>,
) -> Vector3<f64> {
    let core = vertices
        .iter()
        .max_by(|a, b| a.dot(&direction).total_cmp(&b.dot(&direction)))
        .copied()
        .unwrap_or_else(Vector3::zeros);
    core + direction.try_normalize(1e-12).unwrap_or_else(Vector3::x) * radius
}

fn closest_segment_points(
    a0: Vector3<f64>,
    a1: Vector3<f64>,
    b0: Vector3<f64>,
    b1: Vector3<f64>,
) -> (Vector3<f64>, Vector3<f64>) {
    let da = a1 - a0;
    let db = b1 - b0;
    let offset = a0 - b0;
    let aa = da.dot(&da);
    let bb = db.dot(&db);
    let ab = da.dot(&db);
    let ar = da.dot(&offset);
    let br = db.dot(&offset);
    let denominator = aa * bb - ab * ab;
    let mut s = if denominator > 1e-20 {
        ((ab * br - ar * bb) / denominator).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let mut t = if bb > 1e-20 {
        ((ab * s + br) / bb).clamp(0.0, 1.0)
    } else {
        0.0
    };
    if aa > 1e-20 {
        s = ((ab * t - ar) / aa).clamp(0.0, 1.0);
    }
    if bb > 1e-20 {
        t = ((ab * s + br) / bb).clamp(0.0, 1.0);
    }
    (a0 + da * s, b0 + db * t)
}

/// Contact between convex cores expanded by spherical border radii.
///
/// One- and two-vertex cores represent spheres and exact capsules. Larger
/// cores use their face, edge, and vertex-feature separating axes.
#[allow(clippy::too_many_arguments)]
pub fn rounded_convex_contact(
    a: &[Vector3<f64>],
    normals_a: &[Vector3<f64>],
    edges_a: &[Vector3<f64>],
    radius_a: f64,
    b: &[Vector3<f64>],
    normals_b: &[Vector3<f64>],
    edges_b: &[Vector3<f64>],
    radius_b: f64,
) -> Option<ConvexContact> {
    if a.is_empty()
        || b.is_empty()
        || !radius_a.is_finite()
        || !radius_b.is_finite()
        || radius_a < 0.0
        || radius_b < 0.0
    {
        return None;
    }
    if a.len() <= 2 && b.len() <= 2 && (radius_a > 0.0 || radius_b > 0.0) {
        let (point_a, point_b) = closest_segment_points(
            a[0],
            *a.get(1).unwrap_or(&a[0]),
            b[0],
            *b.get(1).unwrap_or(&b[0]),
        );
        let delta = point_b - point_a;
        let distance = delta.norm();
        let radii = radius_a + radius_b;
        if distance > radii + 1e-10 {
            return None;
        }
        let normal = delta.try_normalize(1e-12).unwrap_or_else(|| {
            let centers = (b[0] + *b.get(1).unwrap_or(&b[0])) - (a[0] + *a.get(1).unwrap_or(&a[0]));
            centers.try_normalize(1e-12).unwrap_or_else(Vector3::x)
        });
        let witness_a = point_a + normal * radius_a;
        let witness_b = point_b - normal * radius_b;
        return Some(ConvexContact {
            point: (witness_a + witness_b) * 0.5,
            normal,
            penetration: (radii - distance).max(0.0),
        });
    }

    let support = |direction| {
        rounded_support(a, radius_a, direction) - rounded_support(b, radius_b, -direction)
    };
    let center_a = a.iter().copied().sum::<Vector3<f64>>() / a.len() as f64;
    let center_b = b.iter().copied().sum::<Vector3<f64>>() / b.len() as f64;
    if !gjk_with_support(center_a - center_b, support) {
        return None;
    }

    let mut axes = Vec::with_capacity(
        normals_a.len() + normals_b.len() + edges_a.len() * edges_b.len() + a.len() * b.len() + 1,
    );
    axes.extend(normals_a.iter().copied());
    axes.extend(normals_b.iter().copied());
    axes.extend(
        edges_a
            .iter()
            .flat_map(|edge_a| edges_b.iter().map(move |edge_b| edge_a.cross(edge_b))),
    );
    axes.extend(
        a.iter()
            .flat_map(|point_a| b.iter().map(move |point_b| point_b - point_a)),
    );
    if a.len() == 2 {
        let segment = a[1] - a[0];
        let length_squared = segment.norm_squared();
        axes.extend(b.iter().map(|point| {
            let t = if length_squared > 1e-20 {
                (point - a[0]).dot(&segment) / length_squared
            } else {
                0.0
            }
            .clamp(0.0, 1.0);
            *point - (a[0] + segment * t)
        }));
    }
    if b.len() == 2 {
        let segment = b[1] - b[0];
        let length_squared = segment.norm_squared();
        axes.extend(a.iter().map(|point| {
            let t = if length_squared > 1e-20 {
                (point - b[0]).dot(&segment) / length_squared
            } else {
                0.0
            }
            .clamp(0.0, 1.0);
            (b[0] + segment * t) - *point
        }));
    }
    axes.push(center_b - center_a);

    let project = |vertices: &[Vector3<f64>], radius: f64, axis: Vector3<f64>| {
        let (min, max) =
            vertices
                .iter()
                .fold((f64::INFINITY, -f64::INFINITY), |(min, max), point| {
                    let projection = point.dot(&axis);
                    (min.min(projection), max.max(projection))
                });
        (min - radius, max + radius)
    };
    let mut best_depth = f64::INFINITY;
    let mut best_normal = Vector3::x();
    for candidate in axes {
        let Some(axis) = candidate.try_normalize(1e-10) else {
            continue;
        };
        let (min_a, max_a) = project(a, radius_a, axis);
        let (min_b, max_b) = project(b, radius_b, axis);
        let forward = max_a - min_b;
        let backward = max_b - min_a;
        let (depth, normal) = if forward < backward {
            (forward, axis)
        } else {
            (backward, -axis)
        };
        if depth < -1e-8 {
            return None;
        }
        if depth < best_depth {
            best_depth = depth;
            best_normal = normal;
        }
    }
    if !best_depth.is_finite() {
        return None;
    }
    let witness_a = rounded_support(a, radius_a, best_normal);
    let witness_b = rounded_support(b, radius_b, -best_normal);
    Some(ConvexContact {
        point: (witness_a + witness_b) * 0.5,
        normal: best_normal,
        penetration: best_depth.max(0.0),
    })
}

/// Clip a capsule parallel to a hull face into its endpoint contact manifold.
pub(crate) fn convex_capsule_manifold(
    vertices: &[Vector3<f64>],
    normals: &[Vector3<f64>],
    edges: &[Vector3<f64>],
    capsule: [Vector3<f64>; 2],
    capsule_normals: &[Vector3<f64>],
    capsule_edges: &[Vector3<f64>],
    radius: f64,
) -> Vec<ConvexContact> {
    let segment = capsule[1] - capsule[0];
    let Some(contact) = rounded_convex_contact(
        vertices,
        normals,
        edges,
        0.0,
        &capsule,
        capsule_normals,
        capsule_edges,
        radius,
    ) else {
        return Vec::new();
    };
    let segment_squared = segment.norm_squared();
    if segment_squared <= 1e-12 {
        return vec![contact];
    }
    let Some(normal) = normals
        .iter()
        .copied()
        .filter(|normal| normal.dot(&contact.normal) >= 0.9995)
        .max_by(|a, b| a.dot(&contact.normal).total_cmp(&b.dot(&contact.normal)))
    else {
        return vec![contact];
    };
    if segment.dot(&normal).abs() > 0.05 * segment_squared.sqrt() {
        return vec![contact];
    }

    let support = vertices
        .iter()
        .map(|vertex| vertex.dot(&normal))
        .fold(f64::NEG_INFINITY, f64::max);
    let projected_start = capsule[0] + normal * (support - capsule[0].dot(&normal));
    let face_segment = segment - normal * segment.dot(&normal);
    let mut low = 0.0_f64;
    let mut high = 1.0_f64;
    for axis in normals {
        if axis.dot(&normal) >= 0.9995 {
            continue;
        }
        let height = vertices
            .iter()
            .map(|vertex| vertex.dot(axis))
            .fold(f64::NEG_INFINITY, f64::max);
        let distance = height - axis.dot(&projected_start);
        let rate = axis.dot(&face_segment);
        if rate.abs() <= 1e-12 {
            if distance < -1e-6 {
                return vec![contact];
            }
        } else if rate > 0.0 {
            high = high.min(distance / rate);
        } else {
            low = low.max(distance / rate);
        }
    }
    if high < low {
        return vec![contact];
    }

    let mut manifold = Vec::with_capacity(2);
    for (index, parameter) in [low, high].into_iter().enumerate() {
        if index == 1 && high - low <= 1e-12 {
            continue;
        }
        let center = capsule[0] + segment * parameter;
        let distance = center.dot(&normal) - radius - support;
        if distance > 1e-8 {
            continue;
        }
        let hull_point = center + normal * (support - center.dot(&normal));
        let capsule_point = center - normal * radius;
        manifold.push(ConvexContact {
            point: (hull_point + capsule_point) * 0.5,
            normal,
            penetration: (-distance).max(0.0),
        });
    }
    if manifold.is_empty() {
        vec![contact]
    } else {
        manifold
    }
}

/// Return a one-point contact when two convex hulls intersect.
pub fn convex_face_contact(
    a: &[Vector3<f64>],
    normals_a: &[Vector3<f64>],
    b: &[Vector3<f64>],
    normals_b: &[Vector3<f64>],
) -> Option<ConvexContact> {
    if !gjk_intersects(a, b) {
        return None;
    }
    let center_a = a.iter().copied().sum::<Vector3<f64>>() / a.len() as f64;
    let center_b = b.iter().copied().sum::<Vector3<f64>>() / b.len() as f64;
    let mut best_depth = f64::INFINITY;
    let mut best_normal = Vector3::x();
    for candidate in normals_a.iter().chain(normals_b) {
        let norm = candidate.norm();
        if norm <= 1e-12 {
            continue;
        }
        let axis = candidate / norm;
        let project = |vertices: &[Vector3<f64>]| {
            vertices
                .iter()
                .fold((f64::INFINITY, -f64::INFINITY), |range, point| {
                    let value = point.dot(&axis);
                    (range.0.min(value), range.1.max(value))
                })
        };
        let (min_a, max_a) = project(a);
        let (min_b, max_b) = project(b);
        let positive = max_a - min_b;
        let negative = max_b - min_a;
        let (depth, normal) = if positive < negative {
            (positive, axis)
        } else {
            (negative, -axis)
        };
        if depth < best_depth {
            best_depth = depth;
            best_normal = normal;
        }
    }
    if !best_depth.is_finite() || best_depth < -1e-8 {
        return None;
    }
    let support_a = a
        .iter()
        .max_by(|x, y| x.dot(&best_normal).total_cmp(&y.dot(&best_normal)))?;
    let support_b = b
        .iter()
        .min_by(|x, y| x.dot(&best_normal).total_cmp(&y.dot(&best_normal)))?;
    let normal_midpoint = (*support_a + *support_b) * 0.5;
    let center_midpoint = (center_a + center_b) * 0.5;
    let point =
        center_midpoint + best_normal * (normal_midpoint - center_midpoint).dot(&best_normal);
    Some(ConvexContact {
        point,
        normal: best_normal,
        penetration: best_depth.max(0.0),
    })
}

/// Clip two supporting faces into up to four contact points.
///
/// Vertex-face and edge contacts retain the single witness returned by
/// [`convex_face_contact`]. The polygon path is used only when both supporting
/// sets form nondegenerate faces in the contact plane.
pub fn convex_face_manifold(
    a: &[Vector3<f64>],
    normals_a: &[Vector3<f64>],
    b: &[Vector3<f64>],
    normals_b: &[Vector3<f64>],
) -> Vec<ConvexContact> {
    let Some(witness) = convex_face_contact(a, normals_a, b, normals_b) else {
        return Vec::new();
    };
    clip_supporting_faces(a, b, witness)
}

/// Resolve face and edge axes, then clip a face pair or keep one edge contact.
/// Edge directions must be unit vectors in world coordinates.
pub fn convex_edge_manifold(
    a: &[Vector3<f64>],
    normals_a: &[Vector3<f64>],
    edges_a: &[Vector3<f64>],
    b: &[Vector3<f64>],
    normals_b: &[Vector3<f64>],
    edges_b: &[Vector3<f64>],
) -> Vec<ConvexContact> {
    // Evaluate edge axes for ordinary hulls. Larger hull pairs still use the
    // face path to bound the cubic projection cost on the CPU fallback.
    const MAX_EDGE_PROJECTION_WORK: usize = 4096;
    let work = edges_a
        .len()
        .saturating_mul(edges_b.len())
        .saturating_mul(a.len().saturating_add(b.len()));
    if work > MAX_EDGE_PROJECTION_WORK {
        return convex_face_manifold(a, normals_a, b, normals_b);
    }
    let Some(mut witness) = convex_face_contact(a, normals_a, b, normals_b) else {
        return Vec::new();
    };
    let mut edge_selected = false;
    for edge_a in edges_a {
        for edge_b in edges_b {
            let cross = edge_a.cross(edge_b);
            let Some(axis) = cross.try_normalize(1e-5) else {
                continue;
            };
            let mut min_a = f64::INFINITY;
            let mut max_a = -f64::INFINITY;
            let mut min_b = f64::INFINITY;
            let mut max_b = -f64::INFINITY;
            let mut cannot_improve = false;
            for index in 0..a.len().max(b.len()) {
                if let Some(point) = a.get(index) {
                    let value = point.dot(&axis);
                    min_a = min_a.min(value);
                    max_a = max_a.max(value);
                }
                if let Some(point) = b.get(index) {
                    let value = point.dot(&axis);
                    min_b = min_b.min(value);
                    max_b = max_b.max(value);
                }
                // Both partial overlaps only grow as further vertices are read.
                if max_a - min_b >= witness.penetration && max_b - min_a >= witness.penetration {
                    cannot_improve = true;
                    break;
                }
            }
            if cannot_improve {
                continue;
            }
            let forward = max_a - min_b;
            let backward = max_b - min_a;
            let (depth, normal) = if forward < backward {
                (forward, axis)
            } else {
                (backward, -axis)
            };
            if depth < -1e-8 {
                return Vec::new();
            }
            if depth + 1e-8 < witness.penetration {
                witness.normal = normal;
                witness.penetration = depth.max(0.0);
                witness.point = edge_contact_point(a, *edge_a, b, *edge_b, normal);
                edge_selected = true;
            }
        }
    }
    if edge_selected {
        vec![witness]
    } else {
        clip_supporting_faces(a, b, witness)
    }
}

/// Clip the face selected by a GPU separating-axis witness in f64.
///
/// A mismatched witness returns `None` so the caller can run the complete CPU
/// manifold path.
#[cfg(feature = "gpu-contact")]
pub(crate) fn convex_face_manifold_from_axis(
    a: &[Vector3<f64>],
    b: &[Vector3<f64>],
    selected_axis: Vector3<f64>,
    gpu: ConvexContact,
) -> Option<Vec<ConvexContact>> {
    let mut axis = selected_axis.try_normalize(1e-12)?;
    if axis.dot(&gpu.normal) < 0.0 {
        axis = -axis;
    }
    let project = |vertices: &[Vector3<f64>]| {
        vertices.iter().fold(
            (f64::INFINITY, -f64::INFINITY),
            |(minimum, maximum), vertex| {
                let value = vertex.dot(&axis);
                (minimum.min(value), maximum.max(value))
            },
        )
    };
    let (min_a, max_a) = project(a);
    let (min_b, max_b) = project(b);
    let forward = max_a - min_b;
    let backward = max_b - min_a;
    let (normal, depth) = if forward < backward {
        (axis, forward)
    } else {
        (-axis, backward)
    };
    let extent = (max_a - min_a).max(max_b - min_b).max(1.0);
    if !gpu.penetration.is_finite()
        || gpu.penetration < 0.0
        || depth < -1e-8
        || normal.dot(&gpu.normal) < 0.9999
        || (depth.max(0.0) - gpu.penetration).abs() > 1e-3 * extent
    {
        return None;
    }
    let support_a = a
        .iter()
        .max_by(|left, right| left.dot(&normal).total_cmp(&right.dot(&normal)))?;
    let support_b = b
        .iter()
        .min_by(|left, right| left.dot(&normal).total_cmp(&right.dot(&normal)))?;
    let center_a = a.iter().copied().sum::<Vector3<f64>>() / a.len() as f64;
    let center_b = b.iter().copied().sum::<Vector3<f64>>() / b.len() as f64;
    let midpoint = (center_a + center_b) * 0.5;
    let witness_midpoint = (*support_a + *support_b) * 0.5;
    let witness = ConvexContact {
        point: midpoint + normal * (witness_midpoint - midpoint).dot(&normal),
        normal,
        penetration: depth.max(0.0),
    };
    Some(clip_supporting_faces(a, b, witness))
}

/// One hull and its GPU-selected edge direction.
#[cfg(feature = "gpu-contact")]
pub(crate) struct ConvexEdgeAxisShape<'a> {
    pub vertices: &'a [Vector3<f64>],
    pub directions: &'a [Vector3<f64>],
    pub selected: usize,
}

/// Reconstruct a selected GPU edge-axis witness with f64 support points.
///
/// A mismatched or numerically ambiguous GPU witness returns `None` so the
/// caller can run the full CPU manifold path.
#[cfg(feature = "gpu-contact")]
pub(crate) fn convex_edge_contact_from_axis(
    a: ConvexEdgeAxisShape<'_>,
    b: ConvexEdgeAxisShape<'_>,
    gpu: ConvexContact,
) -> Option<ConvexContact> {
    let direction_a = *a.directions.get(a.selected)?;
    let direction_b = *b.directions.get(b.selected)?;
    let mut axis = direction_a.cross(&direction_b).try_normalize(1e-5)?;
    if axis.dot(&gpu.normal) < 0.0 {
        axis = -axis;
    }
    let project = |vertices: &[Vector3<f64>]| {
        vertices.iter().fold(
            (f64::INFINITY, -f64::INFINITY),
            |(minimum, maximum), vertex| {
                let value = vertex.dot(&axis);
                (minimum.min(value), maximum.max(value))
            },
        )
    };
    let (min_a, max_a) = project(a.vertices);
    let (min_b, max_b) = project(b.vertices);
    let forward = max_a - min_b;
    let backward = max_b - min_a;
    let (normal, depth) = if forward < backward {
        (axis, forward)
    } else {
        (-axis, backward)
    };
    let extent = (max_a - min_a).max(max_b - min_b).max(1.0);
    if !gpu.penetration.is_finite()
        || gpu.penetration < 0.0
        || depth < -1e-8
        || normal.dot(&gpu.normal) < 0.99
        || (depth.max(0.0) - gpu.penetration).abs() > 1e-3 * extent
    {
        return None;
    }
    Some(ConvexContact {
        point: edge_contact_point(a.vertices, direction_a, b.vertices, direction_b, normal),
        normal,
        penetration: depth.max(0.0),
    })
}

fn edge_contact_point(
    a: &[Vector3<f64>],
    direction_a: Vector3<f64>,
    b: &[Vector3<f64>],
    direction_b: Vector3<f64>,
    normal: Vector3<f64>,
) -> Vector3<f64> {
    let max_a = a
        .iter()
        .map(|point| point.dot(&normal))
        .fold(-f64::INFINITY, f64::max);
    let min_b = b
        .iter()
        .map(|point| point.dot(&normal))
        .fold(f64::INFINITY, f64::min);
    let extent = a.iter().chain(b).map(Vector3::norm).fold(1.0_f64, f64::max);
    let tolerance = 1e-7 * extent;
    let support_a = a
        .iter()
        .filter(|point| max_a - point.dot(&normal) <= tolerance)
        .copied()
        .collect::<Vec<_>>();
    let support_b = b
        .iter()
        .filter(|point| point.dot(&normal) - min_b <= tolerance)
        .copied()
        .collect::<Vec<_>>();
    let endpoint = |points: &[Vector3<f64>], direction: Vector3<f64>, maximum: bool| {
        points
            .iter()
            .max_by(|left, right| {
                let left = left.dot(&direction);
                let right = right.dot(&direction);
                if maximum {
                    left.total_cmp(&right)
                } else {
                    right.total_cmp(&left)
                }
            })
            .copied()
            .unwrap_or_else(Vector3::zeros)
    };
    let a0 = endpoint(&support_a, direction_a, false);
    let a1 = endpoint(&support_a, direction_a, true);
    let b0 = endpoint(&support_b, direction_b, false);
    let b1 = endpoint(&support_b, direction_b, true);
    let delta_a = a1 - a0;
    let delta_b = b1 - b0;
    let offset = a0 - b0;
    let aa = delta_a.norm_squared();
    let bb = delta_b.norm_squared();
    let ab = delta_a.dot(&delta_b);
    let determinant = aa * bb - ab * ab;
    if determinant <= 1e-16 {
        return (a0 + a1 + b0 + b1) * 0.25;
    }
    let a_offset = delta_a.dot(&offset);
    let b_offset = delta_b.dot(&offset);
    let t = ((ab * b_offset - bb * a_offset) / determinant).clamp(0.0, 1.0);
    let u = ((aa * b_offset - ab * a_offset) / determinant).clamp(0.0, 1.0);
    (a0 + delta_a * t + b0 + delta_b * u) * 0.5
}

fn clip_supporting_faces(
    a: &[Vector3<f64>],
    b: &[Vector3<f64>],
    witness: ConvexContact,
) -> Vec<ConvexContact> {
    let normal = witness.normal;
    let tangent = normal
        .cross(&if normal.x.abs() < 0.9 {
            Vector3::x()
        } else {
            Vector3::y()
        })
        .normalize();
    let bitangent = normal.cross(&tangent);
    let max_a = a
        .iter()
        .map(|point| point.dot(&normal))
        .fold(-f64::INFINITY, f64::max);
    let min_b = b
        .iter()
        .map(|point| point.dot(&normal))
        .fold(f64::INFINITY, f64::min);
    let extent = a.iter().chain(b).map(Vector3::norm).fold(1.0_f64, f64::max);
    let tolerance = 1e-7 * extent;
    let projected = |point: &Vector3<f64>| Vector2::new(point.dot(&tangent), point.dot(&bitangent));
    let face_a = convex_hull_2d(
        a.iter()
            .filter(|point| max_a - point.dot(&normal) <= tolerance)
            .map(&projected)
            .collect(),
    );
    let face_b = convex_hull_2d(
        b.iter()
            .filter(|point| point.dot(&normal) - min_b <= tolerance)
            .map(&projected)
            .collect(),
    );
    if face_a.len() < 3 || face_b.len() < 3 {
        return vec![witness];
    }
    let mut polygon = face_a;
    for index in 0..face_b.len() {
        let start = face_b[index];
        let end = face_b[(index + 1) % face_b.len()];
        let input = core::mem::take(&mut polygon);
        if input.is_empty() {
            return vec![witness];
        }
        let side = |point: Vector2<f64>| cross_2d(end - start, point - start);
        let mut previous = *input.last().unwrap_or(&input[0]);
        for current in input {
            let old_side = side(previous);
            let new_side = side(current);
            if (old_side >= -tolerance) != (new_side >= -tolerance) {
                let denominator = old_side - new_side;
                if denominator.abs() > 1e-14 {
                    polygon.push(previous + (current - previous) * (old_side / denominator));
                }
            }
            if new_side >= -tolerance {
                polygon.push(current);
            }
            previous = current;
        }
    }
    let mut points = convex_hull_2d(polygon);
    if points.is_empty() {
        return vec![witness];
    }
    if points.len() > 4 {
        let mut selected = vec![points.remove(0)];
        while selected.len() < 4 {
            let Some((index, _)) = points.iter().enumerate().max_by(|(_, left), (_, right)| {
                let spread = |candidate: &Vector2<f64>| {
                    selected
                        .iter()
                        .map(|chosen| (candidate - chosen).norm_squared())
                        .fold(f64::INFINITY, f64::min)
                };
                spread(left).total_cmp(&spread(right))
            }) else {
                break;
            };
            selected.push(points.swap_remove(index));
        }
        points = selected;
    }
    let plane_height = (max_a + min_b) * 0.5;
    points
        .into_iter()
        .map(|point| ConvexContact {
            point: tangent * point.x + bitangent * point.y + normal * plane_height,
            ..witness
        })
        .collect()
}

fn cross_2d(a: Vector2<f64>, b: Vector2<f64>) -> f64 {
    a.x * b.y - a.y * b.x
}

fn convex_hull_2d(mut points: Vec<Vector2<f64>>) -> Vec<Vector2<f64>> {
    points.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
    points.dedup_by(|a, b| (*a - *b).norm_squared() < 1e-20);
    if points.len() < 3 {
        return points;
    }
    let mut lower = Vec::new();
    for point in &points {
        while lower.len() >= 2 {
            let last = lower.len();
            if cross_2d(lower[last - 1] - lower[last - 2], *point - lower[last - 1]) > 1e-12 {
                break;
            }
            let _ = lower.pop();
        }
        lower.push(*point);
    }
    let mut upper = Vec::new();
    for point in points.iter().rev() {
        while upper.len() >= 2 {
            let last = upper.len();
            if cross_2d(upper[last - 1] - upper[last - 2], *point - upper[last - 1]) > 1e-12 {
                break;
            }
            let _ = upper.pop();
        }
        upper.push(*point);
    }
    let _ = lower.pop();
    let _ = upper.pop();
    lower.extend(upper);
    lower
}

fn line_direction(a: Vector3<f64>, b: Vector3<f64>, ao: Vector3<f64>) -> Vector3<f64> {
    let ab = b - a;
    let perpendicular = ab.cross(&ao).cross(&ab);
    if perpendicular.norm_squared() < 1e-20 {
        let axis = if ab.x.abs() < 0.9 * ab.norm() {
            Vector3::x()
        } else {
            Vector3::y()
        };
        ab.cross(&axis)
    } else {
        perpendicular
    }
}

fn update_simplex(simplex: &mut Vec<Vector3<f64>>, direction: &mut Vector3<f64>) -> bool {
    let Some(&a) = simplex.last() else {
        return false;
    };
    let ao = -a;
    match simplex.len() {
        2 => {
            let b = simplex[0];
            if (b - a).dot(&ao) > 0.0 {
                *direction = line_direction(a, b, ao);
            } else {
                simplex.clear();
                simplex.push(a);
                *direction = ao;
            }
        }
        3 => {
            let b = simplex[1];
            let c = simplex[0];
            let ab = b - a;
            let ac = c - a;
            let normal = ab.cross(&ac);
            if normal.cross(&ac).dot(&ao) > 0.0 {
                if ac.dot(&ao) > 0.0 {
                    simplex.clear();
                    simplex.extend([c, a]);
                    *direction = line_direction(a, c, ao);
                } else {
                    simplex.clear();
                    simplex.extend([b, a]);
                    *direction = line_direction(a, b, ao);
                }
            } else if ab.cross(&normal).dot(&ao) > 0.0 {
                simplex.clear();
                simplex.extend([b, a]);
                *direction = line_direction(a, b, ao);
            } else if normal.dot(&ao) > 0.0 {
                *direction = normal;
            } else {
                simplex.swap(0, 1);
                *direction = -normal;
            }
        }
        4 => {
            let b = simplex[2];
            let c = simplex[1];
            let d = simplex[0];
            for (p, q, opposite) in [(b, c, d), (c, d, b), (d, b, c)] {
                let mut normal = (p - a).cross(&(q - a));
                if normal.dot(&(opposite - a)) > 0.0 {
                    normal = -normal;
                }
                if normal.dot(&ao) > 0.0 {
                    simplex.clear();
                    simplex.extend([q, p, a]);
                    *direction = normal;
                    return false;
                }
            }
            return true;
        }
        _ => return false,
    }
    false
}

#[cfg(test)]
mod tests {
    use nalgebra::{Isometry3, Translation3, UnitQuaternion};

    use super::*;
    use crate::articulated_world::box_box_contact;

    #[test]
    fn rejects_degenerate_hulls() {
        assert_eq!(
            ConvexGeometry::new(
                vec![Vector3::zeros(); 4],
                vec![Vector3::zeros(); 4],
                vec![Vector3::x(); 3],
            )
            .unwrap_err(),
            ConvexGeometryError::Invalid
        );
    }

    #[test]
    fn gjk_detects_overlap_separation_and_touching() {
        let cube = |center: Vector3<f64>| {
            [-0.5, 0.5]
                .into_iter()
                .flat_map(move |x| {
                    [-0.5, 0.5].into_iter().flat_map(move |y| {
                        [-0.5, 0.5]
                            .into_iter()
                            .map(move |z| center + Vector3::new(x, y, z))
                    })
                })
                .collect::<Vec<_>>()
        };
        let first = cube(Vector3::zeros());
        assert!(gjk_intersects(&first, &cube(Vector3::new(0.8, 0.0, 0.0))));
        assert!(!gjk_intersects(&first, &cube(Vector3::new(2.0, 0.0, 0.0))));
        assert!(gjk_intersects(&first, &cube(Vector3::new(1.0, 0.0, 0.0))));
    }

    #[test]
    fn sphere_convex_reports_contact_and_separation() {
        let vertices = [-0.5, 0.5]
            .into_iter()
            .flat_map(|x| {
                [-0.5, 0.5]
                    .into_iter()
                    .flat_map(move |y| [-0.5, 0.5].into_iter().map(move |z| Vector3::new(x, y, z)))
            })
            .collect::<Vec<_>>();
        let normals = [Vector3::x(), Vector3::y(), Vector3::z()];
        let contact =
            convex_sphere_contact(&vertices, &normals, Vector3::new(0.8, 0.0, 0.0), 0.4).unwrap();
        assert!((contact.penetration - 0.1).abs() < 1e-8);
        assert!(contact.normal.x > 0.99);
        assert!(
            convex_sphere_contact(&vertices, &normals, Vector3::new(2.0, 0.0, 0.0), 0.4).is_none()
        );
    }

    #[test]
    fn capsule_hull_manifold_clips_endpoints_and_preserves_other_features() {
        let vertices = [-1.0, 1.0]
            .into_iter()
            .flat_map(|x| {
                [-1.0, 1.0]
                    .into_iter()
                    .flat_map(move |y| [-1.0, 1.0].into_iter().map(move |z| Vector3::new(x, y, z)))
            })
            .collect::<Vec<_>>();
        let normals = [
            Vector3::x(),
            -Vector3::x(),
            Vector3::y(),
            -Vector3::y(),
            Vector3::z(),
            -Vector3::z(),
        ];
        let edges = [Vector3::x(), Vector3::y(), Vector3::z()];
        let capsule = [Vector3::new(-2.0, 0.0, 1.1), Vector3::new(2.0, 0.0, 1.1)];
        let manifold =
            convex_capsule_manifold(&vertices, &normals, &edges, capsule, &edges, &edges, 0.2);
        assert_eq!(manifold.len(), 2);
        for (contact, x) in manifold.iter().zip([-1.0, 1.0]) {
            assert!((contact.point - Vector3::new(x, 0.0, 0.95)).norm() < 1e-8);
            assert!((contact.normal - Vector3::z()).norm() < 1e-8);
            assert!((contact.penetration - 0.1).abs() < 1e-8);
        }

        let tilted = [Vector3::new(-0.3, 0.0, 0.9), Vector3::new(0.3, 0.0, 1.5)];
        let segment = tilted[1] - tilted[0];
        let original = rounded_convex_contact(
            &vertices,
            &normals,
            &edges,
            0.0,
            &tilted,
            &[segment],
            &[segment],
            0.2,
        )
        .unwrap();
        let manifold = convex_capsule_manifold(
            &vertices,
            &normals,
            &edges,
            tilted,
            &[segment],
            &[segment],
            0.2,
        );
        assert_eq!(manifold.len(), 1);
        assert!((manifold[0].point - original.point).norm() < 1e-12);
        assert!((manifold[0].normal - original.normal).norm() < 1e-12);
        assert!((manifold[0].penetration - original.penetration).abs() < 1e-12);

        let point = Vector3::new(0.0, 0.0, 1.1);
        assert_eq!(
            convex_capsule_manifold(&vertices, &normals, &edges, [point; 2], &edges, &edges, 0.2,)
                .len(),
            1
        );
        let separated = [Vector3::new(-2.0, 0.0, 3.0), Vector3::new(2.0, 0.0, 3.0)];
        assert!(
            convex_capsule_manifold(&vertices, &normals, &edges, separated, &edges, &edges, 0.2,)
                .is_empty()
        );
    }

    #[test]
    fn rounded_segments_resolve_exact_capsule_contact() {
        let a = [Vector3::new(0.0, 0.0, -1.0), Vector3::new(0.0, 0.0, 1.0)];
        let b = [Vector3::new(0.8, 0.0, -1.0), Vector3::new(0.8, 0.0, 1.0)];
        let contact = rounded_convex_contact(
            &a,
            &[Vector3::z()],
            &[Vector3::z()],
            0.5,
            &b,
            &[Vector3::z()],
            &[Vector3::z()],
            0.5,
        )
        .unwrap();
        assert!((contact.penetration - 0.2).abs() < 1e-10);
        assert!((contact.normal - Vector3::x()).norm() < 1e-10);

        let separated = b.map(|point| point + Vector3::new(0.3, 0.0, 0.0));
        assert!(
            rounded_convex_contact(
                &a,
                &[Vector3::z()],
                &[Vector3::z()],
                0.5,
                &separated,
                &[Vector3::z()],
                &[Vector3::z()],
                0.5,
            )
            .is_none()
        );
    }

    #[test]
    fn support_map_epa_recovers_box_penetration_and_normal() {
        let half = Vector3::<f64>::repeat(0.5);
        let center_a = Vector3::zeros();
        let center_b = Vector3::new(0.8, 0.0, 0.0);
        let support = |center: Vector3<f64>, direction: Vector3<f64>| {
            center
                + Vector3::new(
                    half.x.copysign(direction.x),
                    half.y.copysign(direction.y),
                    half.z.copysign(direction.z),
                )
        };
        let contact = support_map_contact(
            center_a,
            |direction| support(center_a, direction),
            center_b,
            |direction| support(center_b, direction),
        )
        .unwrap();

        assert!((contact.penetration - 0.2).abs() < 1e-8);
        assert!((contact.normal - Vector3::x()).norm() < 1e-8);
        assert!((contact.point.x - 0.4).abs() < 1e-8);
    }

    #[test]
    fn support_map_detects_aligned_cylinder_cap_and_box() {
        let center = Vector3::<f64>::new(0.0, 0.0, 2.0);
        let box_center = Vector3::new(0.0, 0.0, 2.65);
        let half = Vector3::<f64>::new(0.2, 0.25, 0.2);
        let vertices = [-1.0, 1.0]
            .into_iter()
            .flat_map(|x| {
                [-1.0, 1.0].into_iter().flat_map(move |y| {
                    [-1.0, 1.0]
                        .into_iter()
                        .map(move |z| box_center + half.component_mul(&Vector3::new(x, y, z)))
                })
            })
            .collect::<Vec<_>>();
        let support_a = |direction| cylinder_support(center, Vector3::z(), 0.5, 0.3, direction);
        let support_b = |direction: Vector3<f64>| {
            *vertices
                .iter()
                .max_by(|a, b| a.dot(&direction).total_cmp(&b.dot(&direction)))
                .unwrap()
        };
        let contact = support_map_contact(center, support_a, box_center, support_b);
        assert!(contact.is_some(), "{contact:?}");
        let contact = contact.unwrap();
        assert!((contact.penetration - 0.05).abs() < 1e-6, "{contact:?}");
        assert!(contact.normal.z > 0.9, "{contact:?}");

        let shift = Vector3::z() * 2.0;
        let shifted_vertices = vertices
            .iter()
            .map(|vertex| vertex - shift)
            .collect::<Vec<_>>();
        let shifted = support_map_contact(
            center - shift,
            |direction| cylinder_support(center - shift, Vector3::z(), 0.5, 0.3, direction),
            box_center - shift,
            |direction| {
                *shifted_vertices
                    .iter()
                    .max_by(|a, b| a.dot(&direction).total_cmp(&b.dot(&direction)))
                    .unwrap()
            },
        )
        .unwrap();
        assert!((shifted.penetration - 0.05).abs() < 1e-6, "{shifted:?}");
        assert!(shifted.normal.z > 0.9, "{shifted:?}");

        let touching_center = box_center + Vector3::z() * 0.05;
        let touching_vertices = vertices
            .iter()
            .map(|vertex| vertex + Vector3::z() * 0.05)
            .collect::<Vec<_>>();
        let touching = support_map_contact(center, support_a, touching_center, |direction| {
            *touching_vertices
                .iter()
                .max_by(|a, b| a.dot(&direction).total_cmp(&b.dot(&direction)))
                .unwrap()
        });
        assert!(touching.is_some_and(|contact| contact.penetration < 1e-6));

        let separated_center = box_center + Vector3::z() * 0.1;
        let separated_vertices = vertices
            .iter()
            .map(|vertex| vertex + Vector3::z() * 0.1)
            .collect::<Vec<_>>();
        assert!(
            support_map_contact(center, support_a, separated_center, |direction| {
                *separated_vertices
                    .iter()
                    .max_by(|a, b| a.dot(&direction).total_cmp(&b.dot(&direction)))
                    .unwrap()
            })
            .is_none()
        );
    }

    #[test]
    fn analytic_cylinder_support_and_contact_are_exact() {
        let center = Vector3::zeros();
        let axis = Vector3::z();
        let support = |direction| cylinder_support(center, axis, 1.0, 0.5, direction);
        let diagonal = support(Vector3::new(1.0, 0.0, 1.0));
        assert!((diagonal - Vector3::new(0.5, 0.0, 1.0)).norm() < 1e-12);

        let sphere_center = Vector3::new(0.8, 0.0, 0.0);
        let sphere_support = |direction: Vector3<f64>| {
            sphere_center + direction.try_normalize(1e-12).unwrap_or_else(Vector3::x) * 0.5
        };
        let contact = support_map_contact(center, support, sphere_center, sphere_support).unwrap();
        assert!((contact.penetration - 0.2).abs() < 1e-8);
        assert!((contact.normal - Vector3::x()).norm() < 1e-4);
    }

    #[test]
    fn analytic_cone_support_selects_apex_and_base_rim() {
        let center = Vector3::zeros();
        let axis = Vector3::z();
        assert_eq!(
            cone_support(center, axis, 1.0, 0.5, Vector3::z()),
            Vector3::z()
        );
        assert_eq!(
            cone_support(center, axis, 1.0, 0.5, -Vector3::z()),
            -Vector3::z()
        );
        assert_eq!(
            cone_support(center, axis, 1.0, 0.5, Vector3::x()),
            Vector3::new(0.5, 0.0, -1.0)
        );

        let sphere_center = Vector3::new(0.0, 0.0, 1.6);
        assert!(gjk_with_support(center - sphere_center, |direction| {
            cone_support(center, axis, 1.0, 0.5, direction)
                - (sphere_center - direction.try_normalize(1e-12).unwrap_or_else(Vector3::x) * 0.75)
        }));
        let contact = support_map_contact(
            center,
            |direction| cone_support(center, axis, 1.0, 0.5, direction),
            sphere_center,
            |direction| {
                sphere_center + direction.try_normalize(1e-12).unwrap_or_else(Vector3::x) * 0.75
            },
        );
        assert!(contact.is_some());
    }

    #[test]
    fn analytic_sphere_witness_uses_cylinder_side_and_cap() {
        let side = cylinder_sphere_contact(
            Vector3::zeros(),
            Vector3::z(),
            1.0,
            0.5,
            Vector3::new(0.8, 0.0, 0.0),
            0.4,
        )
        .unwrap();
        assert!((side.point - Vector3::new(0.45, 0.0, 0.0)).norm() < 1e-12);
        assert_eq!(side.normal, Vector3::x());
        assert!((side.penetration - 0.1).abs() < 1e-12);

        let cap = cylinder_sphere_contact(
            Vector3::zeros(),
            Vector3::z(),
            1.0,
            0.5,
            Vector3::new(0.0, 0.0, 1.2),
            0.3,
        )
        .unwrap();
        assert!((cap.point - Vector3::new(0.0, 0.0, 0.95)).norm() < 1e-12);
        assert_eq!(cap.normal, Vector3::z());
        assert!((cap.penetration - 0.1).abs() < 1e-12);
        assert!(
            cylinder_sphere_contact(
                Vector3::zeros(),
                Vector3::z(),
                1.0,
                0.5,
                Vector3::new(1.0, 0.0, 0.0),
                0.3,
            )
            .is_none()
        );
    }

    #[test]
    fn analytic_sphere_witness_uses_cone_base_and_apex() {
        let base = cone_sphere_contact(
            Vector3::zeros(),
            Vector3::z(),
            1.0,
            0.5,
            Vector3::new(0.2, 0.0, -1.2),
            0.3,
        )
        .unwrap();
        assert!((base.point - Vector3::new(0.2, 0.0, -0.95)).norm() < 1e-12);
        assert_eq!(base.normal, -Vector3::z());
        assert!((base.penetration - 0.1).abs() < 1e-12);

        let apex = cone_sphere_contact(
            Vector3::zeros(),
            Vector3::z(),
            1.0,
            0.5,
            Vector3::new(0.0, 0.0, 1.2),
            0.3,
        )
        .unwrap();
        assert!((apex.point - Vector3::new(0.0, 0.0, 0.95)).norm() < 1e-12);
        assert_eq!(apex.normal, Vector3::z());
        assert!((apex.penetration - 0.1).abs() < 1e-12);
    }

    #[test]
    fn gjk_agrees_with_oriented_box_sat() {
        let vertices = |pose: Isometry3<f64>, half: Vector3<f64>| {
            [-1.0, 1.0]
                .into_iter()
                .flat_map(move |x| {
                    [-1.0, 1.0].into_iter().flat_map(move |y| {
                        [-1.0, 1.0].into_iter().map(move |z| {
                            pose.transform_point(&nalgebra::Point3::from(Vector3::new(
                                x * half.x,
                                y * half.y,
                                z * half.z,
                            )))
                            .coords
                        })
                    })
                })
                .collect::<Vec<_>>()
        };
        let half_a = Vector3::new(0.4, 0.2, 0.3);
        let half_b = Vector3::new(0.3, 0.4, 0.2);
        for index in 0..500 {
            let phase = index as f64 * 0.37;
            let pose_a = Isometry3::from_parts(
                Translation3::identity(),
                UnitQuaternion::from_euler_angles(phase * 0.13, phase * 0.07, phase * 0.21),
            );
            let pose_b = Isometry3::from_parts(
                Translation3::new(
                    phase.sin() * 0.8,
                    (phase * 1.7).cos() * 0.6,
                    (phase * 0.9).sin() * 0.55,
                ),
                UnitQuaternion::from_euler_angles(phase * 0.31, phase * 0.19, phase * 0.11),
            );
            let sat = box_box_contact(pose_a, half_a, pose_b, half_b);
            if sat.is_some_and(|(_, _, depth)| depth < 1e-6) {
                continue;
            }
            let gjk = gjk_intersects(&vertices(pose_a, half_a), &vertices(pose_b, half_b));
            assert_eq!(gjk, sat.is_some(), "case {index}");
        }
    }

    #[test]
    fn face_contact_returns_separating_normal_and_depth() {
        let cube = |center: Vector3<f64>| {
            [-0.5, 0.5]
                .into_iter()
                .flat_map(move |x| {
                    [-0.5, 0.5].into_iter().flat_map(move |y| {
                        [-0.5, 0.5]
                            .into_iter()
                            .map(move |z| center + Vector3::new(x, y, z))
                    })
                })
                .collect::<Vec<_>>()
        };
        let normals = [Vector3::x(), Vector3::y(), Vector3::z()];
        let contact = convex_face_contact(
            &cube(Vector3::zeros()),
            &normals,
            &cube(Vector3::new(0.8, 0.0, 0.0)),
            &normals,
        )
        .unwrap();
        assert!((contact.penetration - 0.2).abs() < 1e-12);
        assert!((contact.normal - Vector3::x()).norm() < 1e-12);
        assert!(!gjk_intersects(
            &cube(Vector3::zeros()),
            &cube(Vector3::new(2.0, 0.0, 0.0)),
        ));
    }

    #[test]
    fn aligned_faces_generate_four_distinct_contact_points() {
        let cube = |center: Vector3<f64>| {
            [-0.5, 0.5]
                .into_iter()
                .flat_map(move |x| {
                    [-0.5, 0.5].into_iter().flat_map(move |y| {
                        [-0.5, 0.5]
                            .into_iter()
                            .map(move |z| center + Vector3::new(x, y, z))
                    })
                })
                .collect::<Vec<_>>()
        };
        let normals = [Vector3::x(), Vector3::y(), Vector3::z()];
        let manifold = convex_face_manifold(
            &cube(Vector3::zeros()),
            &normals,
            &cube(Vector3::new(0.0, 0.0, 0.9)),
            &normals,
        );
        assert_eq!(manifold.len(), 4);
        for contact in &manifold {
            assert!((contact.normal - Vector3::z()).norm() < 1e-12);
            assert!((contact.penetration - 0.1).abs() < 1e-12);
            assert!((contact.point.z - 0.45).abs() < 1e-12);
            assert!((contact.point.x.abs() - 0.5).abs() < 1e-12);
            assert!((contact.point.y.abs() - 0.5).abs() < 1e-12);
        }
    }

    #[test]
    fn edge_axis_depth_matches_rotated_box_sat() {
        use crate::articulated_world::box_box_contact;
        use nalgebra::{Isometry3, Point3, Translation3, UnitQuaternion};

        let half_a = Vector3::new(0.4, 0.2, 0.3);
        let half_b = Vector3::new(0.3, 0.4, 0.2);
        let vertices = |pose: Isometry3<f64>, half: Vector3<f64>| {
            [-1.0, 1.0]
                .into_iter()
                .flat_map(|x| {
                    [-1.0, 1.0].into_iter().flat_map(move |y| {
                        [-1.0, 1.0].into_iter().map(move |z| {
                            pose.transform_point(&Point3::from(
                                half.component_mul(&Vector3::new(x, y, z)),
                            ))
                            .coords
                        })
                    })
                })
                .collect::<Vec<_>>()
        };
        let mut checked = 0;
        let mut edge_cases = 0;
        for index in 0..500 {
            let phase = index as f64 * 0.37;
            let pose_a = Isometry3::from_parts(
                Translation3::identity(),
                UnitQuaternion::from_euler_angles(phase * 0.13, phase * 0.07, phase * 0.21),
            );
            let pose_b = Isometry3::from_parts(
                Translation3::new(
                    phase.sin() * 0.8,
                    (phase * 1.7).cos() * 0.6,
                    (phase * 0.9).sin() * 0.55,
                ),
                UnitQuaternion::from_euler_angles(phase * 0.31, phase * 0.19, phase * 0.11),
            );
            let Some((sat_normal, _, sat_depth)) = box_box_contact(pose_a, half_a, pose_b, half_b)
            else {
                continue;
            };
            if sat_depth < 1e-5 {
                continue;
            }
            let axes = [Vector3::x(), Vector3::y(), Vector3::z()];
            let axes_a = axes.map(|axis| pose_a.rotation * axis);
            let axes_b = axes.map(|axis| pose_b.rotation * axis);
            let a = vertices(pose_a, half_a);
            let b = vertices(pose_b, half_b);
            let manifold = convex_edge_manifold(&a, &axes_a, &axes_a, &b, &axes_b, &axes_b);
            assert!(!manifold.is_empty(), "case {index}");
            assert!(
                (manifold[0].penetration - sat_depth).abs() < 1e-6,
                "case {index}: {} vs {sat_depth}",
                manifold[0].penetration
            );
            assert!(manifold[0].normal.dot(&sat_normal) > 0.999, "case {index}");
            if convex_face_contact(&a, &axes_a, &b, &axes_b)
                .is_some_and(|face| face.penetration - sat_depth > 1e-6)
            {
                edge_cases += 1;
            }
            checked += 1;
        }
        assert!(checked > 100);
        assert!(edge_cases > 0);
    }
}
