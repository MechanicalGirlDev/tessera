//! CPU reference closest-point queries for rigid collision geometry.
use crate::mesh::TriangleMeshGeometry;
use crate::ray_query::RayShape;
use nalgebra::{Isometry3, Point3, Vector3};

/// Result of a posed-shape point projection.
#[derive(Clone, Copy, Debug)]
pub struct PointProjection {
    /// Closest world-space boundary point, or the input for an inside-solid query.
    pub point: Vector3<f64>,
    /// Unit direction from the boundary toward an exterior query, or outward
    /// from an interior query toward its nearest boundary. Zero at coincidence.
    pub normal: Vector3<f64>,
    /// Whether the input belongs to a volume, including its boundary.
    /// Thin surfaces do not perform volume containment tests.
    pub is_inside: bool,
    /// Unsigned distance to the returned point.
    pub distance: f64,
    /// Triangle or segment ID for surfaces, zero for volume primitives.
    pub feature: usize,
}
/// Invalid point, pose, or geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PointQueryError {
    /// A finite nondegenerate projection could not be evaluated.
    #[error("invalid point projection input")]
    InvalidInput,
}
/// Borrowed scene entry. Its index in the scene is the returned body ID.
#[derive(Clone, Copy, Debug)]
pub struct PointQueryBody<'a> {
    /// World-space pose.
    pub pose: Isometry3<f64>,
    /// Local collision geometry.
    pub shape: RayShape<'a>,
    /// Membership bits intersected with the query mask.
    pub groups: u32,
}
/// Scene projection with the selected body index.
#[derive(Clone, Copy, Debug)]
pub struct ScenePointProjection {
    /// Index in the supplied scene.
    pub body: usize,
    /// Projection onto that body's geometry.
    pub projection: PointProjection,
}
/// Find the nearest accepted body within an inclusive distance bound.
///
/// Equal distances select the lowest scene index, including overlapping solids.
/// Filtered bodies are not evaluated. Invalid accepted geometry is an error,
/// even if an earlier body already lies at zero distance. Positive infinity is
/// permitted as the bound. This reference implementation scans the scene.
pub fn project_point_scene(
    point: Vector3<f64>,
    bodies: &[PointQueryBody<'_>],
    max_distance: f64,
    groups: u32,
    excluded_body: Option<usize>,
    solid: bool,
) -> Result<Option<ScenePointProjection>, PointQueryError> {
    if point.iter().any(|v| !v.is_finite())
        || max_distance.is_nan()
        || max_distance < 0.0
        || excluded_body.is_some_and(|body| body >= bodies.len())
    {
        return Err(PointQueryError::InvalidInput);
    }
    let mut best: Option<ScenePointProjection> = None;
    for (body, entry) in bodies.iter().enumerate() {
        if excluded_body == Some(body) || entry.groups & groups == 0 {
            continue;
        }
        let projection = project_point(point, &entry.pose, entry.shape, solid)?;
        if projection.distance <= max_distance
            && best.is_none_or(|old| projection.distance < old.projection.distance)
        {
            best = Some(ScenePointProjection { body, projection });
        }
    }
    Ok(best)
}
fn length(v: Vector3<f64>) -> f64 {
    v.x.hypot(v.y).hypot(v.z)
}
fn segment(p: Vector3<f64>, a: Vector3<f64>, b: Vector3<f64>) -> Vector3<f64> {
    let e = b - a;
    let span = length(e);
    if span == 0.0 {
        return a;
    }
    let direction = e / span;
    a + e * ((p - a).dot(&direction) / span).clamp(0.0, 1.0)
}
fn nearer(p: Vector3<f64>, a: Vector3<f64>, b: Vector3<f64>) -> Vector3<f64> {
    if length(b - p) < length(a - p) { b } else { a }
}
fn triangle(p: Vector3<f64>, a: Vector3<f64>, b: Vector3<f64>, c: Vector3<f64>) -> Vector3<f64> {
    let e1 = b - a;
    let e2 = c - a;
    let n = e1.cross(&e2);
    let squared = n.norm_squared();
    if squared > 0.0 {
        let projected = p - n * ((p - a).dot(&n) / squared);
        let q = projected - a;
        let u = q.cross(&e2).dot(&n) / squared;
        let v = e1.cross(&q).dot(&n) / squared;
        if u >= 0.0 && v >= 0.0 && u + v <= 1.0 {
            return projected;
        }
    }
    nearer(
        p,
        nearer(p, segment(p, a, b), segment(p, a, c)),
        segment(p, b, c),
    )
}
fn mesh_projection(
    mesh: &TriangleMeshGeometry,
    p: Vector3<f64>,
) -> Result<(Vector3<f64>, usize), PointQueryError> {
    let [a, b, c] = mesh
        .triangle_vertices(0)
        .ok_or(PointQueryError::InvalidInput)?;
    let mut best = triangle(p, a, b, c);
    let mut feature = 0;
    let radius = (best - p).norm();
    for index in mesh.triangles_in_aabb(p - Vector3::repeat(radius), p + Vector3::repeat(radius)) {
        if let Some([a, b, c]) = mesh.triangle_vertices(index) {
            let next = triangle(p, a, b, c);
            let distance = (next - p).norm_squared();
            let old = (best - p).norm_squared();
            if distance < old || (distance == old && index < feature) {
                best = next;
                feature = index;
            }
        }
    }
    Ok((best, feature))
}
fn face_polygon(
    points: &[Vector3<f64>],
    normal: Vector3<f64>,
    tolerance: f64,
) -> Vec<Vector3<f64>> {
    let offset = points
        .iter()
        .map(|v| v.dot(&normal))
        .fold(f64::NEG_INFINITY, f64::max);
    let basis = if normal.x.abs() < 0.9 {
        Vector3::x()
    } else {
        Vector3::y()
    };
    let u = normal.cross(&basis).normalize();
    let v = normal.cross(&u);
    let origin = points[0];
    let mut face = points
        .iter()
        .copied()
        .filter(|p| offset - p.dot(&normal) <= tolerance)
        .map(|p| ((p - origin).dot(&u), (p - origin).dot(&v), p))
        .collect::<Vec<_>>();
    face.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    face.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);
    if face.len() < 3 {
        return Vec::new();
    }
    let cross =
        |a: (f64, f64, Vector3<f64>), b: (f64, f64, Vector3<f64>), c: (f64, f64, Vector3<f64>)| {
            (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
        };
    let mut hull: Vec<(f64, f64, Vector3<f64>)> = Vec::new();
    for p in face.iter().copied() {
        while hull.len() >= 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            let _ = hull.pop();
        }
        hull.push(p);
    }
    let lower = hull.len();
    for p in face.iter().rev().skip(1).copied() {
        while hull.len() > lower && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            let _ = hull.pop();
        }
        hull.push(p);
    }
    let _ = hull.pop();
    hull.into_iter().map(|p| p.2).collect()
}

/// Project a world point onto one posed shape. For volumes, solid=true returns
/// the input when inside; otherwise return the closest boundary. Mesh, heightfield
/// and polyline always return a surface point and is_inside=false. The operation
/// changes no simulation state. Equal-distance primitive features use stable axis order.
pub fn project_point(
    point: Vector3<f64>,
    pose: &Isometry3<f64>,
    shape: RayShape<'_>,
    solid: bool,
) -> Result<PointProjection, PointQueryError> {
    if point
        .iter()
        .chain(pose.translation.vector.iter())
        .chain(pose.rotation.coords.iter())
        .any(|v| !v.is_finite())
        || (pose.rotation.norm_squared() - 1.0).abs() > 1e-8
    {
        return Err(PointQueryError::InvalidInput);
    }
    let p = pose.inverse_transform_point(&Point3::from(point)).coords;
    if p.iter().any(|v| !v.is_finite()) {
        return Err(PointQueryError::InvalidInput);
    }
    let positive = |v: f64| v.is_finite() && v > 0.0;
    let radial = |v: Vector3<f64>| {
        let span = length(v);
        if span > 0.0 { v / span } else { Vector3::x() }
    };
    let mut inside = false;
    let mut feature = 0;
    let projected = match shape {
        RayShape::Sphere(r) => {
            if !positive(r) {
                return Err(PointQueryError::InvalidInput);
            }
            inside = length(p) <= r;
            radial(p) * r
        }
        RayShape::Box(h) => {
            if h.iter().any(|v| !positive(*v)) {
                return Err(PointQueryError::InvalidInput);
            }
            inside = (0..3).all(|axis| p[axis].abs() <= h[axis]);
            let mut q = p.sup(&-h).inf(&h);
            if inside {
                let axis = (0..3)
                    .min_by(|a, b| (h[*a] - p[*a].abs()).total_cmp(&(h[*b] - p[*b].abs())))
                    .ok_or(PointQueryError::InvalidInput)?;
                q[axis] = if p[axis] < 0.0 { -h[axis] } else { h[axis] };
            }
            q
        }
        RayShape::Capsule {
            radius: r,
            half_length: h,
        } => {
            if !positive(r) || !h.is_finite() || h < 0.0 {
                return Err(PointQueryError::InvalidInput);
            }
            let axis = Vector3::new(0.0, 0.0, p.z.clamp(-h, h));
            let offset = p - axis;
            inside = length(offset) <= r;
            axis + radial(offset) * r
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
                return Err(PointQueryError::InvalidInput);
            }
            let rho = p.x.hypot(p.y);
            let direction = radial(Vector3::new(p.x, p.y, 0.0));
            let cone = matches!(shape, RayShape::Cone { .. });
            let local_radius = if cone { r * (h - p.z) / (2.0 * h) } else { r };
            inside = p.z.abs() <= h && rho <= local_radius;
            let base = direction * rho.min(r) - Vector3::z() * h;
            if cone {
                let q = segment(
                    Vector3::new(rho, 0.0, p.z),
                    Vector3::new(r, 0.0, -h),
                    Vector3::new(0.0, 0.0, h),
                );
                nearer(p, base, direction * q.x + Vector3::z() * q.z)
            } else {
                let side = direction * r + Vector3::z() * p.z.clamp(-h, h);
                let top = direction * rho.min(r) + Vector3::z() * h;
                nearer(p, nearer(p, side, base), top)
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
                    .any(|v| (v.norm_squared() - 1.0).abs() > 1e-4)
            {
                return Err(PointQueryError::InvalidInput);
            }
            let tolerance = 64.0
                * f64::EPSILON
                * (1.0 + hull.vertices.iter().map(|v| v.norm()).fold(0.0, f64::max));
            inside = true;
            let mut best: Option<Vector3<f64>> = None;
            for normal in &hull.face_normals {
                let offset = hull
                    .vertices
                    .iter()
                    .map(|v| normal.dot(v))
                    .fold(f64::NEG_INFINITY, f64::max);
                inside &= normal.dot(&p) <= offset;
                let face = face_polygon(&hull.vertices, *normal, tolerance);
                if face.len() < 3 {
                    return Err(PointQueryError::InvalidInput);
                }
                for index in 1..face.len() - 1 {
                    let q = triangle(p, face[0], face[index], face[index + 1]);
                    best = Some(best.map_or(q, |old| nearer(p, old, q)));
                }
            }
            best.ok_or(PointQueryError::InvalidInput)?
        }
        RayShape::TriangleMesh(mesh) => {
            let (q, index) = mesh_projection(mesh, p)?;
            feature = index;
            q
        }
        RayShape::Heightfield(field) => {
            let (q, index) = mesh_projection(field.mesh(), p)?;
            feature = index;
            q
        }
        RayShape::Polyline(line) => {
            let [a, b] = line
                .segment_vertices(0)
                .ok_or(PointQueryError::InvalidInput)?;
            let mut best = segment(p, a, b);
            let radius = (best - p).norm();
            for index in
                line.segments_in_aabb(p - Vector3::repeat(radius), p + Vector3::repeat(radius))
            {
                if let Some([a, b]) = line.segment_vertices(index) {
                    let next = segment(p, a, b);
                    let distance = (next - p).norm_squared();
                    let old = (best - p).norm_squared();
                    if distance < old || (distance == old && index < feature) {
                        best = next;
                        feature = index;
                    }
                }
            }
            best
        }
    };
    let boundary = pose.transform_point(&Point3::from(projected)).coords;
    let displacement = if inside {
        boundary - point
    } else {
        point - boundary
    };
    let normal_span = length(displacement);
    let normal = if normal_span > 0.0 {
        displacement / normal_span
    } else {
        Vector3::zeros()
    };
    let q = if solid && inside { point } else { boundary };
    let distance = length(q - point);
    if q.iter().any(|v| !v.is_finite()) || !distance.is_finite() {
        return Err(PointQueryError::InvalidInput);
    }
    Ok(PointProjection {
        point: q,
        normal,
        is_inside: inside,
        distance,
        feature,
    })
}
