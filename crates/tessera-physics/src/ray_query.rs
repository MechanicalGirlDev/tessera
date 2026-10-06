//! CPU reference ray queries for rigid collision geometry.
use crate::convex::ConvexGeometry;
use crate::mesh::{HeightFieldGeometry, PolylineGeometry, TriangleMeshGeometry};
use nalgebra::{Isometry3, Point3, Vector3};

/// Borrowed local collision geometry for a ray query.
#[derive(Clone, Copy, Debug)]
pub enum RayShape<'a> {
    /// Sphere radius in metres.
    Sphere(f64),
    /// Positive XYZ half extents.
    Box(Vector3<f64>),
    /// Z-axis capsule radius and half distance between hemisphere centres.
    Capsule {
        /// Radius in metres.
        radius: f64,
        /// Nonnegative half length for a capsule, positive for cylinders and cones.
        half_length: f64,
    },
    /// Z-axis cylinder radius and half height.
    Cylinder {
        /// Radius in metres.
        radius: f64,
        /// Nonnegative half length for a capsule, positive for cylinders and cones.
        half_length: f64,
    },
    /// Z-axis cone with apex at positive half height and base at negative half height.
    Cone {
        /// Radius in metres.
        radius: f64,
        /// Nonnegative half length for a capsule, positive for cylinders and cones.
        half_length: f64,
    },
    /// Validated convex hull topology.
    Convex(&'a ConvexGeometry),
    /// Two-sided triangle surface with a local BVH.
    TriangleMesh(&'a TriangleMeshGeometry),
    /// Zero-thickness segments with a local BVH.
    Polyline(&'a PolylineGeometry),
    /// Heightfield using its exact triangle mesh.
    Heightfield(&'a HeightFieldGeometry),
}

/// Ray parameterization `origin + direction * t`; direction need not be unit length.
#[derive(Clone, Copy, Debug)]
pub struct Ray {
    /// Finite world-space origin.
    pub origin: Vector3<f64>,
    /// Finite nonzero world-space direction.
    pub direction: Vector3<f64>,
}

/// First accepted ray intersection.
#[derive(Clone, Copy, Debug)]
pub struct RayIntersection {
    /// Parameter t, within the inclusive query interval.
    pub toi: f64,
    /// World-space intersection point.
    pub point: Vector3<f64>,
    /// Outward normal for solids; opposes the ray for triangle surfaces.
    /// Zero for a solid query starting inside, or a segment with no unique normal.
    pub normal: Vector3<f64>,
    /// Triangle or segment index for thin surfaces, zero for volume primitives.
    pub feature: usize,
}

/// Invalid ray, pose, interval, or primitive dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum RayQueryError {
    /// The query cannot be evaluated with finite nondegenerate inputs.
    #[error("invalid ray query input")]
    InvalidInput,
}

fn roots(a: f64, b: f64, c: f64) -> Vec<f64> {
    let scale = a.abs().max(b.abs()).max(c.abs());
    if scale == 0.0 || !scale.is_finite() {
        return Vec::new();
    }
    let (a, b, c) = (a / scale, b / scale, c / scale);
    if a == 0.0 {
        return if b == 0.0 { Vec::new() } else { vec![-c / b] };
    }
    let discriminant = b * b - 4.0 * a * c;
    let tolerance = 32.0 * f64::EPSILON * (b * b + (4.0 * a * c).abs());
    if discriminant < -tolerance {
        return Vec::new();
    }
    let q = -0.5 * (b + discriminant.max(0.0).sqrt().copysign(b));
    if q == 0.0 {
        vec![-b / (2.0 * a)]
    } else {
        vec![q / a, c / q]
    }
}

/// Cast against one posed shape. A solid query returns t=0 when inside a volume;
/// a hollow query returns its first boundary crossing. Meshes and polylines are
/// always thin. Coplanar triangle rays are misses; segment normals are non-unique.
/// Local BVHs prune mesh and polyline features; no simulation state is changed.
pub fn cast_ray(
    ray: Ray,
    pose: &Isometry3<f64>,
    shape: RayShape<'_>,
    max_t: f64,
    solid: bool,
) -> Result<Option<RayIntersection>, RayQueryError> {
    if !max_t.is_finite()
        || max_t < 0.0
        || ray
            .origin
            .iter()
            .chain(ray.direction.iter())
            .any(|x| !x.is_finite())
        || ray.direction.norm_squared() <= 0.0
        || !ray.direction.norm_squared().is_finite()
        || pose
            .translation
            .vector
            .iter()
            .chain(pose.rotation.coords.iter())
            .any(|x| !x.is_finite())
        || (pose.rotation.norm_squared() - 1.0).abs() > 1e-8
    {
        return Err(RayQueryError::InvalidInput);
    }
    let o = pose
        .inverse_transform_point(&Point3::from(ray.origin))
        .coords;
    let d = pose.inverse_transform_vector(&ray.direction);
    let end = o + d * max_t;
    if o.iter().chain(end.iter()).any(|x| !x.is_finite()) {
        return Err(RayQueryError::InvalidInput);
    }
    let mut hits: Vec<(f64, Vector3<f64>, usize)> = Vec::new();
    let mut inside = false;
    let positive = |v: f64| v.is_finite() && v > 0.0;
    let mut sphere = |center: Vector3<f64>, radius: f64, clip: Option<(bool, f64)>| {
        let offset = o - center;
        let squared_speed = d.dot(&d);
        let closest = -offset.dot(&d) / squared_speed;
        let perpendicular = offset + d * closest;
        let squared_distance = perpendicular.dot(&perpendicular);
        let gap = radius * radius - squared_distance;
        let tolerance = 32.0 * f64::EPSILON * (radius * radius + squared_distance);
        if gap < -tolerance {
            return;
        }
        let span = (gap.max(0.0) / squared_speed).sqrt();
        for t in [closest - span, closest + span] {
            let p = o + d * t;
            if clip.is_none_or(|(upper, z)| if upper { p.z >= z } else { p.z <= z }) {
                hits.push((t, (p - center) / radius, 0));
            }
        }
    };
    match shape {
        RayShape::Sphere(r) => {
            if !positive(r) {
                return Err(RayQueryError::InvalidInput);
            }
            inside = o.norm_squared() <= r * r;
            sphere(Vector3::zeros(), r, None);
        }
        RayShape::Capsule {
            radius: r,
            half_length: h,
        } => {
            if !positive(r) || !h.is_finite() || h < 0.0 {
                return Err(RayQueryError::InvalidInput);
            }
            inside = (o - Vector3::new(0.0, 0.0, o.z.clamp(-h, h))).norm_squared() <= r * r;
            sphere(Vector3::new(0.0, 0.0, h), r, Some((true, h)));
            sphere(Vector3::new(0.0, 0.0, -h), r, Some((false, -h)));
            for t in roots(
                d.x * d.x + d.y * d.y,
                2.0 * (o.x * d.x + o.y * d.y),
                o.x * o.x + o.y * o.y - r * r,
            ) {
                let p = o + d * t;
                if p.z.abs() <= h {
                    hits.push((t, Vector3::new(p.x / r, p.y / r, 0.0), 0));
                }
            }
        }
        RayShape::Cylinder {
            radius: r,
            half_length: h,
        }
        | RayShape::Cone {
            radius: r,
            half_length: h,
        } => {
            if !positive(r) || !positive(h) {
                return Err(RayQueryError::InvalidInput);
            }
            let cone = matches!(shape, RayShape::Cone { .. });
            let k = r / (2.0 * h);
            let local_radius = if cone { k * (h - o.z) } else { r };
            inside = o.z.abs() <= h && o.x * o.x + o.y * o.y <= local_radius * local_radius;
            let k2 = if cone { k * k } else { 0.0 };
            let a = d.x * d.x + d.y * d.y - k2 * d.z * d.z;
            let b = 2.0 * (o.x * d.x + o.y * d.y + k2 * (h - o.z) * d.z);
            let c = o.x * o.x + o.y * o.y
                - if cone {
                    k2 * (h - o.z) * (h - o.z)
                } else {
                    r * r
                };
            for t in roots(a, b, c) {
                let p = o + d * t;
                if p.z.abs() <= h {
                    let n = Vector3::new(p.x, p.y, k2 * (h - p.z));
                    hits.push((t, n.try_normalize(0.0).unwrap_or_else(Vector3::z), 0));
                }
            }
            if d.z != 0.0 {
                for (z, cap_radius, normal) in [
                    (-h, r, -Vector3::z()),
                    (h, if cone { 0.0 } else { r }, Vector3::z()),
                ] {
                    let t = (z - o.z) / d.z;
                    let p = o + d * t;
                    if p.x * p.x + p.y * p.y <= cap_radius * cap_radius {
                        hits.push((t, normal, 0));
                    }
                }
            }
        }
        RayShape::Box(half) => {
            if half.iter().any(|v| !positive(*v)) {
                return Err(RayQueryError::InvalidInput);
            }
            let mut planes = Vec::new();
            for axis in 0..3 {
                let mut n = Vector3::zeros();
                n[axis] = 1.0;
                planes.push((n, half[axis]));
                planes.push((-n, half[axis]));
            }
            if let Some((enter, exit, normal_enter, normal_exit, is_inside)) =
                clip_planes(o, d, &planes)
            {
                inside = is_inside;
                hits.push((enter, normal_enter, 0));
                hits.push((exit, normal_exit, 0));
            }
        }
        RayShape::Convex(hull) => {
            if hull.vertices.len() < 4
                || hull.face_normals.len() < 4
                || hull
                    .vertices
                    .iter()
                    .chain(hull.face_normals.iter())
                    .any(|v| v.iter().any(|x| !x.is_finite()))
                || hull
                    .face_normals
                    .iter()
                    .any(|n| (n.norm_squared() - 1.0).abs() > 1e-4)
            {
                return Err(RayQueryError::InvalidInput);
            }
            let planes = hull
                .face_normals
                .iter()
                .map(|n| {
                    (
                        *n,
                        hull.vertices
                            .iter()
                            .map(|v| n.dot(v))
                            .fold(f64::NEG_INFINITY, f64::max),
                    )
                })
                .collect::<Vec<_>>();
            if let Some((enter, exit, normal_enter, normal_exit, is_inside)) =
                clip_planes(o, d, &planes)
            {
                inside = is_inside;
                hits.push((enter, normal_enter, 0));
                hits.push((exit, normal_exit, 0));
            }
        }
        RayShape::TriangleMesh(mesh) => mesh_hits(mesh, o, d, end, &mut hits),
        RayShape::Heightfield(field) => mesh_hits(field.mesh(), o, d, end, &mut hits),
        RayShape::Polyline(line) => {
            let (lower, upper) = line.local_bounds();
            let radius = lower.abs().sup(&upper.abs()).norm();
            let margin = Vector3::repeat(64.0 * f64::EPSILON * (1.0 + o.norm() + 2.0 * radius));
            for index in line.segments_in_aabb(o.inf(&end) - margin, o.sup(&end) + margin) {
                if let Some([a, b]) = line.segment_vertices(index) {
                    let e = b - a;
                    let w = o - a;
                    let aa = d.dot(&d);
                    let cross = d.cross(&e);
                    let denominator = cross.norm_squared();
                    let tolerance = 64.0 * f64::EPSILON * (1.0 + o.norm() + a.norm() + b.norm());
                    if denominator > 0.0 {
                        // Cross products retain the angle of nearly parallel segments.
                        let t = (-w).cross(&e).dot(&cross) / denominator;
                        let u = (-w).cross(&d).dot(&cross) / denominator;
                        if (0.0..=1.0).contains(&u) && (o + d * t - a - e * u).norm() <= tolerance {
                            hits.push((t, Vector3::zeros(), index));
                        }
                    } else if w.cross(&d).norm() <= tolerance * d.norm() {
                        let t0 = (a - o).dot(&d) / aa;
                        let t1 = (b - o).dot(&d) / aa;
                        if t0.max(t1) >= 0.0 {
                            hits.push((t0.min(t1).max(0.0), Vector3::zeros(), index));
                        }
                    }
                }
            }
        }
    }
    if solid && inside {
        return Ok(Some(RayIntersection {
            toi: 0.0,
            point: ray.origin,
            normal: Vector3::zeros(),
            feature: 0,
        }));
    }
    let best = hits
        .into_iter()
        .filter(|(t, _, _)| t.is_finite() && *t >= 0.0 && *t <= max_t)
        .min_by(|a, b| a.0.total_cmp(&b.0).then(a.2.cmp(&b.2)));
    Ok(best.map(|(toi, normal, feature)| RayIntersection {
        toi,
        point: ray.origin + ray.direction * toi,
        normal: pose.transform_vector(&normal),
        feature,
    }))
}

type ClipResult = (f64, f64, Vector3<f64>, Vector3<f64>, bool);
fn clip_planes(
    o: Vector3<f64>,
    d: Vector3<f64>,
    planes: &[(Vector3<f64>, f64)],
) -> Option<ClipResult> {
    let mut enter = f64::NEG_INFINITY;
    let mut exit = f64::INFINITY;
    let mut ne = Vector3::zeros();
    let mut nx = ne;
    let mut inside = true;
    for (n, offset) in planes {
        let distance = n.dot(&o) - offset;
        let speed = n.dot(&d);
        inside &= distance <= 0.0;
        if speed == 0.0 {
            if distance > 0.0 {
                return None;
            }
            continue;
        }
        let t = -distance / speed;
        if speed < 0.0 && t > enter {
            enter = t;
            ne = *n;
        }
        if speed > 0.0 && t < exit {
            exit = t;
            nx = *n;
        }
        if enter > exit {
            return None;
        }
    }
    Some((enter, exit, ne, nx, inside))
}
fn mesh_hits(
    mesh: &TriangleMeshGeometry,
    o: Vector3<f64>,
    d: Vector3<f64>,
    end: Vector3<f64>,
    hits: &mut Vec<(f64, Vector3<f64>, usize)>,
) {
    for index in mesh.triangles_in_aabb(o.inf(&end), o.sup(&end)) {
        if let Some([a, b, c]) = mesh.triangle_vertices(index) {
            let e1 = b - a;
            let e2 = c - a;
            let p = d.cross(&e2);
            let det = e1.dot(&p);
            if det.abs() <= 16.0 * f64::EPSILON * e1.norm() * e2.norm() * d.norm() {
                continue;
            }
            let delta = o - a;
            let u = delta.dot(&p) / det;
            let q = delta.cross(&e1);
            let v = d.dot(&q) / det;
            if u >= 0.0 && v >= 0.0 && u + v <= 1.0 {
                let t = e2.dot(&q) / det;
                let mut normal = e1.cross(&e2).normalize();
                if normal.dot(&d) > 0.0 {
                    normal = -normal;
                }
                hits.push((t, normal, index));
            }
        }
    }
}
