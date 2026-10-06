struct Particle {
    position_mass: vec4<f32>,
    velocity_volume: vec4<f32>,
    force_damping: vec4<f32>,
    radius: vec4<f32>,
    affine0: vec4<f32>,
    affine1: vec4<f32>,
    affine2: vec4<f32>,
    stress0: vec4<f32>,
    stress1: vec4<f32>,
    stress2: vec4<f32>,
    deformation0: vec4<f32>,
    deformation1: vec4<f32>,
    deformation2: vec4<f32>,
    material: vec4<f32>,
    material_extra: vec4<f32>,
    projection: vec4<f32>,
    base_cell: vec4<i32>,
}

struct GridNode {
    mass: atomic<u32>,
    momentum_x: atomic<u32>,
    momentum_y: atomic<u32>,
    momentum_z: atomic<u32>,
}

struct Transfer {
    position: vec4<f32>,
    velocity: vec4<f32>,
    affine0: vec4<f32>,
    affine1: vec4<f32>,
    affine2: vec4<f32>,
    deformation0: vec4<f32>,
    deformation1: vec4<f32>,
    deformation2: vec4<f32>,
    plastic: vec4<f32>,
}

struct Params {
    origin: vec4<i32>,
    dimensions: vec4<u32>,
    scalars: vec4<f32>,
    gravity: vec4<f32>,
    bound_min: vec4<f32>,
    bound_max: vec4<f32>,
}

struct Obstacle {
    center_radius: vec4<f32>,
    half_extents_kind: vec4<f32>,
    orientation: vec4<f32>,
    linear_velocity_friction: vec4<f32>,
    angular_velocity: vec4<f32>,
    triangle_a: vec4<f32>,
    triangle_b: vec4<f32>,
    triangle_c: vec4<f32>,
    convex_range: vec4<u32>,
}

struct ConvexPlane {
    normal_offset: vec4<f32>,
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> grid: array<GridNode>;
@group(0) @binding(2) var<storage, read_write> grid_velocity: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> transfers: array<Transfer>;
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var<storage, read> obstacles: array<Obstacle>;
@group(0) @binding(6) var<storage, read> convex_planes: array<ConvexPlane>;
@group(0) @binding(7) var<storage, read> particle_nodes: array<u32>;
@group(0) @binding(8) var<storage, read> node_coords: array<vec4<i32>>;
struct NodeDispatch { values: vec4<u32>, }
@group(0) @binding(9) var<uniform> node_dispatch: NodeDispatch;

fn rotate_quaternion(q: vec4<f32>, value: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, value);
    return value + q.w * t + cross(q.xyz, t);
}

fn cpic_triangle_side(obstacle: Obstacle, point: vec3<f32>) -> vec2<f32> {
    let q = obstacle.orientation;
    let local = rotate_quaternion(vec4<f32>(-q.xyz, q.w), point - obstacle.center_radius.xyz);
    let a = obstacle.triangle_a.xyz;
    let edge_b = obstacle.triangle_b.xyz - a;
    let edge_c = obstacle.triangle_c.xyz - a;
    let normal = normalize(cross(edge_b, edge_c));
    let side_distance = dot(local - a, normal);
    let projected = local - normal * side_distance - a;
    let d00 = dot(edge_b, edge_b);
    let d01 = dot(edge_b, edge_c);
    let d11 = dot(edge_c, edge_c);
    let d20 = dot(projected, edge_b);
    let d21 = dot(projected, edge_c);
    let denominator = d00 * d11 - d01 * d01;
    let v = (d11 * d20 - d01 * d21) / denominator;
    let w = (d00 * d21 - d01 * d20) / denominator;
    if v >= -1e-5 && w >= -1e-5 && v + w <= 1.0 + 1e-5 {
        return vec2<f32>(abs(side_distance), select(0.0, 1.0, side_distance >= 0.0));
    }
    return vec2<f32>(1e30, 0.0);
}

@compute @workgroup_size(64)
fn update_cpic_colors(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    if index >= arrayLength(&particles) { return; }
    var distances: array<f32, 32>;
    var sides: array<u32, 32>;
    for (var group = 0u; group < 32u; group++) {
        distances[group] = 1e30;
        sides[group] = 0u;
    }
    let position = particles[index].position_mass.xyz;
    for (var obstacle_index = 0; obstacle_index < params.origin.w; obstacle_index++) {
        let obstacle = obstacles[obstacle_index];
        if obstacle.convex_range.z == 0u { continue; }
        let group = obstacle.convex_range.z - 1u;
        let side = cpic_triangle_side(obstacle, position);
        if side.x < distances[group] {
            distances[group] = side.x;
            sides[group] = u32(side.y);
        }
    }
    var mask = 0u;
    for (var group = 0u; group < 32u; group++) {
        mask |= sides[group] << group;
    }
    particles[index].radius.z = bitcast<f32>(mask);
}

fn closest_point_triangle(point: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    let ab = b - a;
    let ac = c - a;
    let ap = point - a;
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 { return a; }
    let bp = point - b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 { return b; }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return a + ab * (d1 / (d1 - d3));
    }
    let cp = point - c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 { return c; }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return a + ac * (d2 / (d2 - d6));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    let reciprocal = 1.0 / (va + vb + vc);
    return a + ab * (vb * reciprocal) + ac * (vc * reciprocal);
}

fn triangle_prism_surface(obstacle: Obstacle, point: vec3<f32>) -> vec4<f32> {
    let a = obstacle.triangle_a.xyz;
    let b = obstacle.triangle_b.xyz;
    let c = obstacle.triangle_c.xyz;
    let normal = normalize(cross(b - a, c - a));
    let offset = normal * obstacle.half_extents_kind.x;
    let prism = array<vec3<f32>, 6>(a + offset, b + offset, c + offset,
        a - offset, b - offset, c - offset);
    let faces = array<vec3<u32>, 8>(
        vec3<u32>(0u, 1u, 2u), vec3<u32>(3u, 5u, 4u),
        vec3<u32>(3u, 4u, 1u), vec3<u32>(3u, 1u, 0u),
        vec3<u32>(4u, 5u, 2u), vec3<u32>(4u, 2u, 1u),
        vec3<u32>(5u, 3u, 0u), vec3<u32>(5u, 0u, 2u)
    );
    let edge_scale = max(max(length(b - a), length(c - b)),
        max(length(a - c), obstacle.half_extents_kind.x));
    let epsilon = edge_scale * 1e-6;
    var inside = true;
    var closest = vec3<f32>(0.0);
    var best_distance_squared = 1e30;
    var best_normal = normal;
    var tied_normals = vec3<f32>(0.0);
    for (var face_index = 0u; face_index < 8u; face_index++) {
        let face = faces[face_index];
        let va = prism[face.x];
        let vb = prism[face.y];
        let vc = prism[face.z];
        let face_normal = normalize(cross(vb - va, vc - va));
        inside = inside && dot(point - va, face_normal) <= epsilon;
        let candidate = closest_point_triangle(point, va, vb, vc);
        let distance_squared = dot(point - candidate, point - candidate);
        if distance_squared + epsilon * epsilon < best_distance_squared {
            best_distance_squared = distance_squared;
            closest = candidate;
            best_normal = face_normal;
            tied_normals = face_normal;
        } else if abs(distance_squared - best_distance_squared) <= epsilon * epsilon {
            tied_normals += face_normal;
        }
    }
    let distance = sqrt(best_distance_squared);
    var feature_normal = best_normal;
    if length(tied_normals) > 1e-12 {
        feature_normal = normalize(tied_normals);
    }
    var outward = feature_normal;
    if !inside && distance > epsilon {
        outward = (point - closest) / distance;
    }
    return vec4<f32>(outward, select(distance, -distance, inside));
}

fn convex_surface(obstacle: Obstacle, point: vec3<f32>) -> vec4<f32> {
    let start = obstacle.convex_range.x;
    let count = obstacle.convex_range.y;
    var maximum = -1e30;
    var nearest_normal = vec3<f32>(0.0, 0.0, 1.0);
    for (var i = 0u; i < count; i++) {
        let plane = convex_planes[start + i].normal_offset;
        let signed_distance = dot(plane.xyz, point) - plane.w;
        if signed_distance > maximum {
            maximum = signed_distance;
            nearest_normal = plane.xyz;
        }
    }
    if maximum <= 0.0 {
        return vec4<f32>(nearest_normal, maximum);
    }
    var projected = point;
    var corrections: array<vec3<f32>, 128>;
    for (var i = 0u; i < count; i++) {
        corrections[i] = vec3<f32>(0.0);
    }
    let tolerance = max(obstacle.center_radius.w, 1.0) * 1e-5;
    for (var iteration = 0u; iteration < 64u; iteration++) {
        let previous = projected;
        for (var i = 0u; i < count; i++) {
            let plane = convex_planes[start + i].normal_offset;
            let shifted = projected + corrections[i];
            let violation = max(dot(plane.xyz, shifted) - plane.w, 0.0);
            projected = shifted - plane.xyz * violation;
            corrections[i] = shifted - projected;
        }
        if distance(projected, previous) <= tolerance {
            break;
        }
    }
    let offset = point - projected;
    let signed_distance = length(offset);
    if signed_distance > tolerance {
        return vec4<f32>(offset / signed_distance, signed_distance);
    }
    return vec4<f32>(nearest_normal, maximum);
}

fn obstacle_surface(obstacle: Obstacle, point: vec3<f32>) -> vec4<f32> {
    let relative = point - obstacle.center_radius.xyz;
    if obstacle.half_extents_kind.w == 0.0 {
        let radius = length(relative);
        let normal = select(vec3<f32>(0.0, 0.0, 1.0), relative / max(radius, 1e-12), radius > 1e-12);
        return vec4<f32>(normal, radius - obstacle.center_radius.w);
    }
    let q = obstacle.orientation;
    let local = rotate_quaternion(vec4<f32>(-q.xyz, q.w), relative);
    if obstacle.half_extents_kind.w == 2.0 {
        let half_height = obstacle.half_extents_kind.x;
        let segment_point = vec3<f32>(0.0, 0.0, clamp(local.z, -half_height, half_height));
        let delta = local - segment_point;
        let distance = length(delta);
        let axis_epsilon = max(half_height, obstacle.center_radius.w) * 1e-6;
        let normal = select(vec3<f32>(1.0, 0.0, 0.0), delta / max(distance, 1e-12), distance > axis_epsilon);
        return vec4<f32>(rotate_quaternion(q, normal), distance - obstacle.center_radius.w);
    }
    if obstacle.half_extents_kind.w == 3.0 {
        let half_height = obstacle.half_extents_kind.x;
        let radial = length(local.xy);
        let radial_normal = select(vec2<f32>(1.0, 0.0), local.xy / max(radial, 1e-12), radial > 1e-12);
        let radial_distance = radial - obstacle.center_radius.w;
        let cap_distance = abs(local.z) - half_height;
        let edge_epsilon = max(half_height, obstacle.center_radius.w) * 1e-6;
        if abs(radial_distance) <= edge_epsilon && abs(cap_distance) <= edge_epsilon {
            let cap_normal = vec3<f32>(0.0, 0.0, select(1.0, -1.0, local.z < 0.0));
            let normal = normalize(vec3<f32>(radial_normal, 0.0) + cap_normal);
            return vec4<f32>(rotate_quaternion(q, normal), max(radial_distance, cap_distance));
        }
        let outside = vec3<f32>(
            radial_normal * max(radial_distance, 0.0),
            select(1.0, -1.0, local.z < 0.0) * max(cap_distance, 0.0)
        );
        let outside_length = length(outside);
        if outside_length > 1e-12 {
            return vec4<f32>(rotate_quaternion(q, outside / outside_length), outside_length);
        }
        if radial_distance >= cap_distance {
            return vec4<f32>(rotate_quaternion(q, vec3<f32>(radial_normal, 0.0)), radial_distance);
        }
        let normal = vec3<f32>(0.0, 0.0, select(1.0, -1.0, local.z < 0.0));
        return vec4<f32>(rotate_quaternion(q, normal), cap_distance);
    }
    if obstacle.half_extents_kind.w == 4.0 {
        let half_height = obstacle.half_extents_kind.x;
        let cone_radius = obstacle.center_radius.w;
        let radial = length(local.xy);
        let radial_normal = select(vec2<f32>(1.0, 0.0), local.xy / max(radial, 1e-12), radial > 1e-12);
        let point_2d = vec2<f32>(radial, local.z);
        let base = vec2<f32>(clamp(radial, 0.0, cone_radius), -half_height);
        let side_start = vec2<f32>(cone_radius, -half_height);
        let side_direction = vec2<f32>(-cone_radius, 2.0 * half_height);
        let side_t = clamp(dot(point_2d - side_start, side_direction)
            / dot(side_direction, side_direction), 0.0, 1.0);
        let side = side_start + side_direction * side_t;
        let edge_epsilon = max(half_height, cone_radius) * 1e-6;
        let base_distance_squared = dot(point_2d - base, point_2d - base);
        let side_distance_squared = dot(point_2d - side, point_2d - side);
        let use_base = base_distance_squared <= side_distance_squared;
        let closest = select(side, base, use_base);
        let delta = point_2d - closest;
        let distance = length(delta);
        let inside = local.z >= -half_height - edge_epsilon && local.z <= half_height + edge_epsilon
            && radial <= cone_radius * (half_height - local.z) / (2.0 * half_height) + edge_epsilon;
        let base_normal = vec2<f32>(0.0, -1.0);
        let side_normal = normalize(vec2<f32>(2.0 * half_height, cone_radius));
        let near_corner = base_distance_squared <= edge_epsilon * edge_epsilon
            && side_distance_squared <= edge_epsilon * edge_epsilon;
        var feature_normal = select(side_normal, base_normal, use_base);
        if near_corner {
            feature_normal = normalize(base_normal + side_normal);
        }
        let normal_2d = select(delta / max(distance, 1e-12), feature_normal, inside || distance <= edge_epsilon);
        let normal = vec3<f32>(radial_normal * normal_2d.x, normal_2d.y);
        return vec4<f32>(rotate_quaternion(q, normal), select(distance, -distance, inside));
    }
    if obstacle.half_extents_kind.w == 5.0 {
        let half_extents = obstacle.half_extents_kind.xy;
        let outside_xy = local.xy - clamp(local.xy, -half_extents, half_extents);
        if length(outside_xy) <= 1e-12 {
            return vec4<f32>(rotate_quaternion(q, vec3<f32>(0.0, 0.0, 1.0)), local.z);
        }
        let outside = vec3<f32>(outside_xy, local.z);
        let distance = length(outside);
        return vec4<f32>(rotate_quaternion(q, outside / distance), distance);
    }
    if obstacle.half_extents_kind.w == 6.0 {
        let surface = triangle_prism_surface(obstacle, local);
        return vec4<f32>(rotate_quaternion(q, surface.xyz), surface.w);
    }
    if obstacle.half_extents_kind.w == 7.0 {
        let surface = convex_surface(obstacle, local);
        return vec4<f32>(rotate_quaternion(q, surface.xyz), surface.w);
    }
    let half_extents = obstacle.half_extents_kind.xyz;
    let closest = clamp(local, -half_extents, half_extents);
    let outside = local - closest;
    let outside_length = length(outside);
    if outside_length > 1e-12 {
        return vec4<f32>(rotate_quaternion(q, outside / outside_length), outside_length);
    }
    let distances = half_extents - abs(local);
    var axis = 0;
    if distances.y < distances.x && distances.y <= distances.z {
        axis = 1;
    } else if distances.z < distances.x && distances.z < distances.y {
        axis = 2;
    }
    if axis == 0 {
        let sign = select(1.0, -1.0, local.x < 0.0);
        return vec4<f32>(rotate_quaternion(q, vec3<f32>(sign, 0.0, 0.0)), -distances.x);
    }
    if axis == 1 {
        let sign = select(1.0, -1.0, local.y < 0.0);
        return vec4<f32>(rotate_quaternion(q, vec3<f32>(0.0, sign, 0.0)), -distances.y);
    }
    let sign = select(1.0, -1.0, local.z < 0.0);
    return vec4<f32>(rotate_quaternion(q, vec3<f32>(0.0, 0.0, sign)), -distances.z);
}

fn obstacle_contact_velocity(
    obstacle: Obstacle,
    point: vec3<f32>,
    velocity: vec3<f32>,
    normal: vec3<f32>
) -> vec3<f32> {
    let surface_velocity = obstacle.linear_velocity_friction.xyz
        + cross(obstacle.angular_velocity.xyz, point - obstacle.center_radius.xyz);
    let boundary = obstacle.angular_velocity.w;
    if boundary == 3.0 {
        return velocity;
    }
    if boundary == 1.0 {
        return surface_velocity;
    }
    let relative = velocity - surface_velocity;
    let normal_speed = dot(relative, normal);
    if normal_speed >= 0.0 {
        return velocity;
    }
    let tangent = relative - normal * normal_speed;
    if boundary == 2.0 {
        return surface_velocity + tangent;
    }
    let tangent_speed = length(tangent);
    let friction_scale = select(
        0.0,
        max(0.0, 1.0 - obstacle.linear_velocity_friction.w * -normal_speed / max(tangent_speed, 1e-12)),
        tangent_speed > 1e-12
    );
    return surface_velocity + tangent * friction_scale;
}

fn obstacle_may_contact(obstacle: Obstacle, position: vec3<f32>, margin: f32) -> bool {
    if obstacle.half_extents_kind.w == 2.0 {
        return distance(position, obstacle.center_radius.xyz)
            <= (obstacle.half_extents_kind.x + obstacle.center_radius.w + margin) * (1.0 + 1e-5);
    }
    if obstacle.half_extents_kind.w == 6.0 || obstacle.half_extents_kind.w == 7.0 {
        return distance(position, obstacle.center_radius.xyz)
            <= obstacle.center_radius.w + margin;
    }
    return true;
}

fn quadratic_weight(value: f32, offset: i32) -> f32 {
    if offset == 0 {
        return 0.5 * (1.5 - value) * (1.5 - value);
    }
    if offset == 1 {
        return 0.75 - (value - 1.0) * (value - 1.0);
    }
    return 0.5 * (value - 0.5) * (value - 0.5);
}

fn node_index(particle_index: u32, offset: vec3<i32>) -> u32 {
    if params.dimensions.x != 0u {
        let cell = vec3<u32>(particles[particle_index].base_cell.xyz + offset - params.origin.xyz);
        return (cell.x * params.dimensions.y + cell.y) * params.dimensions.z + cell.z;
    }
    let local_index = u32((offset.x * 3 + offset.y) * 3 + offset.z);
    return particle_nodes[particle_index * 27u + local_index];
}

fn finite_vec3(value: vec3<f32>) -> bool {
    return all(abs(value) < vec3<f32>(1e30));
}

fn regular_corotated(deformation: mat3x3<f32>) -> bool {
    let a = deformation[0];
    let b = deformation[1];
    let c = deformation[2];
    let det = dot(a, cross(b, c));
    let norm = sqrt(dot(a, a) + dot(b, b) + dot(c, c));
    if abs(det) < 1e-5 || norm < 1e-5 || norm > 20.0 {
        return false;
    }
    let inv0 = cross(b, c) / det;
    let inv1 = cross(c, a) / det;
    let inv2 = cross(a, b) / det;
    let inverse_norm = sqrt(dot(inv0, inv0) + dot(inv1, inv1) + dot(inv2, inv2));
    return norm * inverse_norm <= 100.0;
}

fn polar_rotation(deformation: mat3x3<f32>) -> mat3x3<f32> {
    var rotation = deformation;
    for (var iteration = 0u; iteration < 10u; iteration++) {
        let a = rotation[0];
        let b = rotation[1];
        let c = rotation[2];
        let determinant_value = dot(a, cross(b, c));
        let inverse_transpose = mat3x3<f32>(
            cross(b, c) / determinant_value,
            cross(c, a) / determinant_value,
            cross(a, b) / determinant_value
        );
        rotation = (rotation + inverse_transpose) * 0.5;
    }
    return rotation;
}

struct EigenState {
    matrix: mat3x3<f32>,
    vectors: mat3x3<f32>,
}

fn jacobi_pair(input: EigenState, p: u32, q: u32) -> EigenState {
    var matrix = input.matrix;
    var vectors = input.vectors;
    let off_diagonal = matrix[q][p];
    if abs(off_diagonal) < 1e-8 {
        return input;
    }
    let tau = (matrix[q][q] - matrix[p][p]) / (2.0 * off_diagonal);
    let sign = select(-1.0, 1.0, tau >= 0.0);
    let tangent = sign / (abs(tau) + sqrt(1.0 + tau * tau));
    let cosine = inverseSqrt(1.0 + tangent * tangent);
    let sine = tangent * cosine;
    let pp = matrix[p][p];
    let qq = matrix[q][q];
    for (var r = 0u; r < 3u; r++) {
        if r == p || r == q { continue; }
        let rp = matrix[p][r];
        let rq = matrix[q][r];
        matrix[p][r] = cosine * rp - sine * rq;
        matrix[r][p] = matrix[p][r];
        matrix[q][r] = sine * rp + cosine * rq;
        matrix[r][q] = matrix[q][r];
    }
    matrix[p][p] = pp - tangent * off_diagonal;
    matrix[q][q] = qq + tangent * off_diagonal;
    matrix[p][q] = 0.0;
    matrix[q][p] = 0.0;
    let vp = vectors[p];
    let vq = vectors[q];
    vectors[p] = cosine * vp - sine * vq;
    vectors[q] = sine * vp + cosine * vq;
    return EigenState(matrix, vectors);
}

fn principal_axes(deformation: mat3x3<f32>) -> EigenState {
    // Jacobi sweeps diagonalize F^T F; the columns of vectors are right singular axes.
    let identity = mat3x3<f32>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0)
    );
    var state = EigenState(transpose(deformation) * deformation, identity);
    for (var sweep = 0u; sweep < 8u; sweep++) {
        state = jacobi_pair(state, 0u, 1u);
        state = jacobi_pair(state, 0u, 2u);
        state = jacobi_pair(state, 1u, 2u);
    }
    return state;
}

fn replace_stretches(
    deformation: mat3x3<f32>,
    axes: mat3x3<f32>,
    singular: vec3<f32>,
    projected: vec3<f32>
) -> mat3x3<f32> {
    // F V diag(projected / singular) V^T replaces principal stretches without U.
    let scale = projected / singular;
    let scaled_axes = mat3x3<f32>(
        axes[0] * scale.x,
        axes[1] * scale.y,
        axes[2] * scale.z
    );
    return deformation * scaled_axes * transpose(axes);
}

struct PlasticProjection {
    deformation: mat3x3<f32>,
    plastic: vec4<f32>,
}

fn project_plastic(deformation: mat3x3<f32>, particle: Particle) -> PlasticProjection {
    let axes = principal_axes(deformation);
    let singular = sqrt(max(
        vec3<f32>(axes.matrix[0][0], axes.matrix[1][1], axes.matrix[2][2]),
        vec3<f32>(0.0)
    ));
    if min(min(singular.x, singular.y), singular.z) < 1e-6
        || max(max(singular.x, singular.y), singular.z)
            / min(min(singular.x, singular.y), singular.z) > 100.0 {
        return PlasticProjection(deformation, vec4<f32>(0.0));
    }
    if particle.material.x == 4.0 {
        let clamped = clamp(
            singular,
            vec3<f32>(1.0 - particle.material_extra.y),
            vec3<f32>(1.0 + particle.material_extra.z)
        );
        let old_det = singular.x * singular.y * singular.z;
        let new_det = max(clamped.x * clamped.y * clamped.z, 1e-12);
        let plastic_det = clamp(particle.material_extra.x * old_det / new_det, 0.1, 4.0);
        return PlasticProjection(
            replace_stretches(deformation, axes.vectors, singular, clamped),
            vec4<f32>(plastic_det, particle.projection.x, particle.projection.y, 1.0)
        );
    }
    let strain = log(max(singular, vec3<f32>(1e-12)))
        + vec3<f32>(particle.material_extra.z / 3.0);
    let trace = strain.x + strain.y + strain.z;
    let deviatoric = strain - vec3<f32>(trace / 3.0);
    let norm = length(deviatoric);
    let cohesion = particle.projection.x;
    let shifted_trace = trace - cohesion;
    let angle_sine = sin(particle.material_extra.w);
    let alpha = sqrt(2.0 / 3.0) * (2.0 * angle_sine) / (3.0 - angle_sine);
    var plastic = vec3<f32>(
        particle.material_extra.x,
        particle.material_extra.y,
        particle.material_extra.z
    );
    var projected_log = vec3<f32>(cohesion / 3.0);
    if shifted_trace <= 0.0 && norm > 1e-12 {
        let lambda = particle.material.y;
        let mu = particle.material.z;
        let gamma = norm + (3.0 * lambda + 2.0 * mu)
            / (2.0 * mu) * shifted_trace * alpha;
        if gamma <= 0.0 {
            return PlasticProjection(deformation, vec4<f32>(plastic, 1.0));
        }
        plastic.y += gamma;
        projected_log = strain - deviatoric * (gamma / norm);
    }
    let projected = exp(projected_log);
    let old_det = max(singular.x * singular.y * singular.z, 1e-12);
    let new_det = max(projected.x * projected.y * projected.z, 1e-12);
    plastic.x = clamp(plastic.x * old_det / new_det, 0.01, 100.0);
    plastic.z += log(old_det) - log(new_det);
    return PlasticProjection(
        replace_stretches(deformation, axes.vectors, singular, projected),
        vec4<f32>(plastic, 1.0)
    );
}

fn particle_stress(particle: Particle) -> mat3x3<f32> {
    let fallback = mat3x3<f32>(particle.stress0.xyz, particle.stress1.xyz, particle.stress2.xyz);
    if particle.material.x == 0.0 {
        return fallback;
    }
    let deformation = mat3x3<f32>(
        particle.deformation0.xyz,
        particle.deformation1.xyz,
        particle.deformation2.xyz
    );
    let j = determinant(deformation);
    var stress = fallback;
    if particle.material.x == 1.0 {
        let lambda = particle.material.y;
        let mu = particle.material.z;
        let safe_j = max(j, 1e-10);
        let isotropic = lambda * log(safe_j) - mu;
        stress = deformation * transpose(deformation) * mu
            + mat3x3<f32>(
                vec3<f32>(isotropic, 0.0, 0.0),
                vec3<f32>(0.0, isotropic, 0.0),
                vec3<f32>(0.0, 0.0, isotropic)
            );
    } else if particle.material.x == 2.0 {
        let bulk = particle.material.y;
        let gamma = particle.material.z;
        let viscosity = particle.material.w;
        let safe_j = max(j, 1e-6);
        var pressure = -bulk * particle.material_extra.x * (safe_j - 1.0);
        if safe_j <= 1.0 {
            pressure = bulk * (pow(safe_j, -gamma) - 1.0);
        }
        let affine = mat3x3<f32>(particle.affine0.xyz, particle.affine1.xyz, particle.affine2.xyz);
        let strain_rate = (affine + transpose(affine)) * 0.5;
        let trace_third = (strain_rate[0].x + strain_rate[1].y + strain_rate[2].z) / 3.0;
        let diagonal = -2.0 * viscosity * safe_j * trace_third - pressure * safe_j;
        stress = strain_rate * (2.0 * viscosity * safe_j)
            + mat3x3<f32>(
                vec3<f32>(diagonal, 0.0, 0.0),
                vec3<f32>(0.0, diagonal, 0.0),
                vec3<f32>(0.0, 0.0, diagonal)
            );
    } else {
        let lambda = particle.material.y;
        let mu = particle.material.z;
        var factor = 1.0;
        if particle.material.x == 4.0 {
            factor = clamp(
                exp(particle.material.w * (1.0 - particle.material_extra.x)),
                0.01,
                100.0
            );
        }
        let rotation = polar_rotation(deformation);
        let isotropic = lambda * factor * (j - 1.0) * j;
        stress = (deformation - rotation) * transpose(deformation) * (2.0 * mu * factor)
            + mat3x3<f32>(
                vec3<f32>(isotropic, 0.0, 0.0),
                vec3<f32>(0.0, isotropic, 0.0),
                vec3<f32>(0.0, 0.0, isotropic)
            );
    }
    return stress * (-params.scalars.w * particle.velocity_volume.w * params.scalars.z);
}

fn topology_valid(particle: u32) -> bool {
    if params.dimensions.x != 0u { return true; }
    for (var point = 0u; point < 27u; point++) {
        if particle_nodes[particle * 27u + point] >= node_dispatch.values.w { return false; }
    }
    return true;
}

@compute @workgroup_size(64)
fn p2g(@builtin(global_invocation_id) invocation: vec3<u32>) {
    if invocation.x >= arrayLength(&particles) {
        return;
    }
    let particle = particles[invocation.x];
    let mass = particle.position_mass.w;
    if mass <= 0.0 || !topology_valid(invocation.x) || (params.gravity.w != 0.0 && particle.base_cell.w != 0) {
        return;
    }
    let grid_pos = particle.position_mass.xyz * params.scalars.y;
    let base = particle.base_cell.xyz;
    let fx = grid_pos - vec3<f32>(base);
    let stress = particle_stress(particle);
    for (var x = 0; x < 3; x++) {
        for (var y = 0; y < 3; y++) {
            for (var z = 0; z < 3; z++) {
                let offset = vec3<i32>(x, y, z);
                let weight = quadratic_weight(fx.x, x)
                    * quadratic_weight(fx.y, y)
                    * quadratic_weight(fx.z, z);
                let dpos = (vec3<f32>(offset) - fx) * params.scalars.x;
                let affine_momentum =
                    (particle.affine0.xyz * mass + stress[0]) * dpos.x
                    + (particle.affine1.xyz * mass + stress[1]) * dpos.y
                    + (particle.affine2.xyz * mass + stress[2]) * dpos.z;
                let momentum = weight * (mass * particle.velocity_volume.xyz + affine_momentum);
                let index = node_index(invocation.x, offset);
                loop {
                    let old_bits = atomicLoad(&grid[index].mass);
                    let next = bitcast<u32>(bitcast<f32>(old_bits) + weight * mass);
                    if atomicCompareExchangeWeak(&grid[index].mass, old_bits, next).exchanged {
                        break;
                    }
                }
                loop {
                    let old_bits = atomicLoad(&grid[index].momentum_x);
                    let next = bitcast<u32>(bitcast<f32>(old_bits) + momentum.x);
                    if atomicCompareExchangeWeak(&grid[index].momentum_x, old_bits, next).exchanged {
                        break;
                    }
                }
                loop {
                    let old_bits = atomicLoad(&grid[index].momentum_y);
                    let next = bitcast<u32>(bitcast<f32>(old_bits) + momentum.y);
                    if atomicCompareExchangeWeak(&grid[index].momentum_y, old_bits, next).exchanged {
                        break;
                    }
                }
                loop {
                    let old_bits = atomicLoad(&grid[index].momentum_z);
                    let next = bitcast<u32>(bitcast<f32>(old_bits) + momentum.z);
                    if atomicCompareExchangeWeak(&grid[index].momentum_z, old_bits, next).exchanged {
                        break;
                    }
                }
            }
        }
    }
}

@compute @workgroup_size(64)
fn update_grid(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    if index >= node_dispatch.values.w || index >= arrayLength(&grid) {
        return;
    }
    let mass = bitcast<f32>(atomicLoad(&grid[index].mass));
    if mass <= 0.0 {
        grid_velocity[index] = vec4<f32>(0.0);
        return;
    }
    var velocity = vec3<f32>(
        bitcast<f32>(atomicLoad(&grid[index].momentum_x)),
        bitcast<f32>(atomicLoad(&grid[index].momentum_y)),
        bitcast<f32>(atomicLoad(&grid[index].momentum_z))
    ) / mass + params.gravity.xyz * params.scalars.w;
    var cell = vec3<i32>(0);
    if params.dimensions.x != 0u {
        let yz = params.dimensions.y * params.dimensions.z;
        cell = params.origin.xyz + vec3<i32>(
            i32(index / yz),
            i32((index / params.dimensions.z) % params.dimensions.y),
            i32(index % params.dimensions.z)
        );
    } else {
        cell = node_coords[index].xyz;
    }
    let position = vec3<f32>(cell) * params.scalars.x;
    if params.dimensions.w != 0u {
        for (var axis = 0; axis < 3; axis++) {
            if (position[axis] <= params.bound_min[axis] + params.scalars.x && velocity[axis] < 0.0)
                || (position[axis] >= params.bound_max[axis] - params.scalars.x && velocity[axis] > 0.0) {
                velocity[axis] = 0.0;
            }
        }
    }
    for (var obstacle_index = 0; obstacle_index < params.origin.w; obstacle_index++) {
        let obstacle = obstacles[obstacle_index];
        if !obstacle_may_contact(obstacle, position, params.scalars.x) {
            continue;
        }
        let surface = obstacle_surface(obstacle, position);
        if surface.w <= params.scalars.x * (1.0 + 1e-5) {
            velocity = obstacle_contact_velocity(obstacle, position, velocity, surface.xyz);
        }
    }
    grid_velocity[index] = vec4<f32>(velocity, 0.0);
}

@compute @workgroup_size(64)
fn g2p(@builtin(global_invocation_id) invocation: vec3<u32>) {
    if invocation.x >= arrayLength(&particles) {
        return;
    }
    let particle = particles[invocation.x];
    if particle.position_mass.w <= 0.0 {
        transfers[invocation.x] = Transfer(
            vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0),
            vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0),
            vec4<f32>(0.0)
        );
        return;
    }
    let topology_failed = !topology_valid(invocation.x) || particle.base_cell.w == 2;
    if topology_failed || (params.gravity.w != 0.0 && particle.base_cell.w != 0) {
        let failure = select(1, 2, topology_failed);
        particles[invocation.x].base_cell.w = failure;
        transfers[invocation.x] = Transfer(
            particle.position_mass, particle.velocity_volume,
            particle.affine0, particle.affine1, particle.affine2,
            particle.deformation0, particle.deformation1, particle.deformation2,
            vec4<f32>(0.0, 0.0, 0.0, -f32(failure))
        );
        return;
    }
    if particle.radius.y != 0.0 {
        var fixed_particle = particle;
        fixed_particle.velocity_volume = vec4<f32>(0.0, 0.0, 0.0, particle.velocity_volume.w);
        fixed_particle.force_damping = vec4<f32>(0.0, 0.0, 0.0, particle.force_damping.w);
        fixed_particle.affine0 = vec4<f32>(0.0);
        fixed_particle.affine1 = vec4<f32>(0.0);
        fixed_particle.affine2 = vec4<f32>(0.0);
        particles[invocation.x] = fixed_particle;
        transfers[invocation.x] = Transfer(
            particle.position_mass, vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0),
            vec4<f32>(0.0), particle.deformation0, particle.deformation1,
            particle.deformation2, vec4<f32>(0.0)
        );
        return;
    }
    let grid_pos = particle.position_mass.xyz * params.scalars.y;
    let base = particle.base_cell.xyz;
    let fx = grid_pos - vec3<f32>(base);
    var velocity = vec3<f32>(0.0);
    var affine0 = vec3<f32>(0.0);
    var affine1 = vec3<f32>(0.0);
    var affine2 = vec3<f32>(0.0);
    for (var x = 0; x < 3; x++) {
        for (var y = 0; y < 3; y++) {
            for (var z = 0; z < 3; z++) {
                let offset = vec3<i32>(x, y, z);
                let weight = quadratic_weight(fx.x, x)
                    * quadratic_weight(fx.y, y)
                    * quadratic_weight(fx.z, z);
                let dpos = (vec3<f32>(offset) - fx) * params.scalars.x;
                let node_velocity = grid_velocity[node_index(invocation.x, offset)].xyz;
                velocity += node_velocity * weight;
                affine0 += node_velocity * (dpos.x * weight * params.scalars.z);
                affine1 += node_velocity * (dpos.y * weight * params.scalars.z);
                affine2 += node_velocity * (dpos.z * weight * params.scalars.z);
            }
        }
    }
    velocity += particle.force_damping.xyz / particle.position_mass.w * params.scalars.w;
    velocity *= exp(-particle.force_damping.w * params.scalars.w);
    var position = particle.position_mass.xyz + velocity * params.scalars.w;
    let radius = particle.radius.x;
    if params.dimensions.w != 0u {
        for (var axis = 0; axis < 3; axis++) {
            let lower = params.bound_min[axis] + radius;
            let upper = params.bound_max[axis] - radius;
            if position[axis] < lower {
                position[axis] = lower;
                velocity[axis] = max(velocity[axis], 0.0);
            } else if position[axis] > upper {
                position[axis] = upper;
                velocity[axis] = min(velocity[axis], 0.0);
            }
        }
    }
    let pre_contact_position = position;
    let pre_contact_velocity = velocity;
    for (var obstacle_index = 0; obstacle_index < params.origin.w; obstacle_index++) {
        let obstacle = obstacles[obstacle_index];
        if !obstacle_may_contact(obstacle, position, radius) {
            continue;
        }
        let surface = obstacle_surface(obstacle, position);
        if surface.w < radius {
            position += surface.xyz * (radius - surface.w);
            velocity = obstacle_contact_velocity(obstacle, position, velocity, surface.xyz);
        }
    }
    let affine = mat3x3<f32>(affine0, affine1, affine2);
    let old_deformation = mat3x3<f32>(
        particle.deformation0.xyz,
        particle.deformation1.xyz,
        particle.deformation2.xyz
    );
    let identity = mat3x3<f32>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0)
    );
    var deformation = (identity + affine * params.scalars.w) * old_deformation;
    var plastic = vec4<f32>(0.0);
    if particle.material.x == 2.0 {
        let scale = pow(max(determinant(deformation), 1e-9), 1.0 / 3.0);
        deformation = identity * scale;
    } else if particle.material.x == 4.0 || particle.material.x == 5.0 {
        let projection = project_plastic(deformation, particle);
        deformation = projection.deformation;
        plastic = projection.plastic;
    }
    var next_particle = particle;
    next_particle.position_mass = vec4<f32>(position, particle.position_mass.w);
    next_particle.velocity_volume = vec4<f32>(velocity, particle.velocity_volume.w);
    next_particle.force_damping = vec4<f32>(0.0, 0.0, 0.0, particle.force_damping.w);
    next_particle.affine0 = vec4<f32>(affine0, 0.0);
    next_particle.affine1 = vec4<f32>(affine1, 0.0);
    next_particle.affine2 = vec4<f32>(affine2, 0.0);
    next_particle.deformation0 = vec4<f32>(deformation[0], 0.0);
    next_particle.deformation1 = vec4<f32>(deformation[1], 0.0);
    next_particle.deformation2 = vec4<f32>(deformation[2], 0.0);
    let finite_position = finite_vec3(position);
    let next_cell = floor(position * params.scalars.y - 0.5);
    // The strict upper bound leaves room for all stencil offsets after f32 rounding.
    let representable_cell = all(next_cell >= vec3<f32>(-2147483648.0))
        && all(next_cell < vec3<f32>(2147483648.0));
    var next_base = particle.base_cell.xyz;
    if finite_position && representable_cell {
        next_base = vec3<i32>(next_cell);
    }
    var resident_failed = false;
    if params.gravity.w != 0.0 {
        var first_base = params.origin.xyz;
        var last_base = params.origin.xyz + vec3<i32>(params.dimensions.xyz) - vec3<i32>(3);
        if params.dimensions.x == 0u {
            first_base = vec3<i32>(-2147483647 - 1);
            last_base = vec3<i32>(2147483645);
            if params.dimensions.w != 0u {
                first_base = vec3<i32>(floor(params.bound_min.xyz * params.scalars.y - 0.5)) - vec3<i32>(2);
                last_base = vec3<i32>(floor(params.bound_max.xyz * params.scalars.y - 0.5)) + vec3<i32>(2);
            }
        }
        resident_failed = any(next_base < first_base) || any(next_base > last_base)
            || !finite_position || !representable_cell || !finite_vec3(velocity)
            || !finite_vec3(affine0) || !finite_vec3(affine1) || !finite_vec3(affine2)
            || !finite_vec3(deformation[0]) || !finite_vec3(deformation[1])
            || !finite_vec3(deformation[2]);
        if particle.material.x == 4.0 || particle.material.x == 5.0 {
            resident_failed = resident_failed || plastic.w != 1.0;
        }
        if particle.material.x == 3.0 || particle.material.x == 4.0 || particle.material.x == 5.0 {
            resident_failed = resident_failed || !regular_corotated(deformation);
        }
        let density = max(particle.position_mass.w / particle.velocity_volume.w, 1e-12);
        var wave_speed = sqrt((particle.material.y + 2.0 * particle.material.z) / density);
        if particle.material.x == 2.0 {
            wave_speed = sqrt(particle.material.y * particle.material.z / density);
        }
        resident_failed = resident_failed
            || params.scalars.w * (wave_speed + length(velocity))
                > 0.4 * params.scalars.x * (1.0 + 1e-4);
    }
    next_particle.base_cell = vec4<i32>(next_base, i32(resident_failed));
    if plastic.w == 1.0 && particle.material.x == 4.0 {
        next_particle.material_extra.x = plastic.x;
        next_particle.projection.x = plastic.y;
        next_particle.projection.y = plastic.z;
    } else if plastic.w == 1.0 && particle.material.x == 5.0 {
        next_particle.material_extra.x = plastic.x;
        next_particle.material_extra.y = plastic.y;
        next_particle.material_extra.z = plastic.z;
    }
    particles[invocation.x] = next_particle;
    if resident_failed {
        plastic.w = -1.0;
    }
    transfers[invocation.x] = Transfer(
        vec4<f32>(position, pre_contact_position.x),
        vec4<f32>(velocity, pre_contact_position.y),
        vec4<f32>(affine0, pre_contact_position.z),
        vec4<f32>(affine1, pre_contact_velocity.x),
        vec4<f32>(affine2, pre_contact_velocity.y),
        vec4<f32>(deformation[0], pre_contact_velocity.z),
        vec4<f32>(deformation[1], 0.0),
        vec4<f32>(deformation[2], 0.0),
        plastic
    );
}

fn add_obstacle_reaction(obstacle_index: u32, impulse: vec3<f32>, moment: vec3<f32>) {
    let capacity = arrayLength(&transfers) - arrayLength(&particles);
    let first = arrayLength(&grid) - 2u * capacity + obstacle_index * 2u;
    loop {
        let bits = atomicLoad(&grid[first].mass);
        let next = bitcast<u32>(bitcast<f32>(bits) + impulse.x);
        if atomicCompareExchangeWeak(&grid[first].mass, bits, next).exchanged { break; }
    }
    loop {
        let bits = atomicLoad(&grid[first].momentum_x);
        let next = bitcast<u32>(bitcast<f32>(bits) + impulse.y);
        if atomicCompareExchangeWeak(&grid[first].momentum_x, bits, next).exchanged { break; }
    }
    loop {
        let bits = atomicLoad(&grid[first].momentum_y);
        let next = bitcast<u32>(bitcast<f32>(bits) + impulse.z);
        if atomicCompareExchangeWeak(&grid[first].momentum_y, bits, next).exchanged { break; }
    }
    loop {
        let bits = atomicLoad(&grid[first].momentum_z);
        let next = bitcast<u32>(bitcast<f32>(bits) + moment.x);
        if atomicCompareExchangeWeak(&grid[first].momentum_z, bits, next).exchanged { break; }
    }
    loop {
        let bits = atomicLoad(&grid[first + 1u].mass);
        let next = bitcast<u32>(bitcast<f32>(bits) + moment.y);
        if atomicCompareExchangeWeak(&grid[first + 1u].mass, bits, next).exchanged { break; }
    }
    loop {
        let bits = atomicLoad(&grid[first + 1u].momentum_x);
        let next = bitcast<u32>(bitcast<f32>(bits) + moment.z);
        if atomicCompareExchangeWeak(&grid[first + 1u].momentum_x, bits, next).exchanged { break; }
    }
}

@compute @workgroup_size(64)
fn collect_grid_reactions(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let grid_index = invocation.x;
    if grid_index >= node_dispatch.values.w {
        return;
    }
    let mass = bitcast<f32>(atomicLoad(&grid[grid_index].mass));
    if mass <= 0.0 {
        return;
    }
    var velocity = vec3<f32>(
        bitcast<f32>(atomicLoad(&grid[grid_index].momentum_x)),
        bitcast<f32>(atomicLoad(&grid[grid_index].momentum_y)),
        bitcast<f32>(atomicLoad(&grid[grid_index].momentum_z))
    ) / mass + params.gravity.xyz * params.scalars.w;
    var cell = vec3<i32>(0);
    if params.dimensions.x != 0u {
        let yz = params.dimensions.y * params.dimensions.z;
        cell = params.origin.xyz + vec3<i32>(
            i32(grid_index / yz),
            i32((grid_index / params.dimensions.z) % params.dimensions.y),
            i32(grid_index % params.dimensions.z)
        );
    } else {
        cell = node_coords[grid_index].xyz;
    }
    let position = vec3<f32>(cell) * params.scalars.x;
    if params.dimensions.w != 0u {
        for (var axis = 0; axis < 3; axis++) {
            if (position[axis] <= params.bound_min[axis] + params.scalars.x && velocity[axis] < 0.0)
                || (position[axis] >= params.bound_max[axis] - params.scalars.x && velocity[axis] > 0.0) {
                velocity[axis] = 0.0;
            }
        }
    }
    for (var obstacle_index = 0; obstacle_index < params.origin.w; obstacle_index++) {
        let obstacle = obstacles[obstacle_index];
        if !obstacle_may_contact(obstacle, position, params.scalars.x) {
            continue;
        }
        let surface = obstacle_surface(obstacle, position);
        if surface.w <= params.scalars.x * (1.0 + 1e-5) {
            let before = velocity;
            velocity = obstacle_contact_velocity(obstacle, position, velocity, surface.xyz);
            let impulse = (before - velocity) * mass;
            let contact_point = position - surface.xyz * surface.w;
            add_obstacle_reaction(
                u32(obstacle_index), impulse,
                cross(contact_point - obstacle.center_radius.xyz, impulse)
            );
        }
    }
}

@compute @workgroup_size(64)
fn collect_particle_reactions(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let particle_index = invocation.x;
    if particle_index >= arrayLength(&particles) {
        return;
    }
    let particle = particles[particle_index];
    let transfer = transfers[particle_index];
    if particle.position_mass.w <= 0.0 || particle.radius.y != 0.0 || transfer.plastic.w < 0.0 {
        return;
    }
    var position = vec3<f32>(transfer.position.w, transfer.velocity.w, transfer.affine0.w);
    var velocity = vec3<f32>(transfer.affine1.w, transfer.affine2.w, transfer.deformation0.w);
    let radius = particle.radius.x;
    for (var obstacle_index = 0; obstacle_index < params.origin.w; obstacle_index++) {
        let obstacle = obstacles[obstacle_index];
        if !obstacle_may_contact(obstacle, position, radius) {
            continue;
        }
        let surface = obstacle_surface(obstacle, position);
        if surface.w < radius {
            position += surface.xyz * (radius - surface.w);
            let before = velocity;
            velocity = obstacle_contact_velocity(obstacle, position, velocity, surface.xyz);
            let impulse = (before - velocity) * particle.position_mass.w;
            let contact_point = position - surface.xyz * radius;
            add_obstacle_reaction(
                u32(obstacle_index), impulse,
                cross(contact_point - obstacle.center_radius.xyz, impulse)
            );
        }
    }
}

@compute @workgroup_size(64)
fn finalize_obstacle_reactions(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let obstacle_index = invocation.x;
    if obstacle_index >= u32(params.origin.w) {
        return;
    }
    let capacity = arrayLength(&transfers) - arrayLength(&particles);
    let first = arrayLength(&grid) - 2u * capacity + obstacle_index * 2u;
    let linear = vec3<f32>(
        bitcast<f32>(atomicLoad(&grid[first].mass)),
        bitcast<f32>(atomicLoad(&grid[first].momentum_x)),
        bitcast<f32>(atomicLoad(&grid[first].momentum_y))
    );
    let angular = vec3<f32>(
        bitcast<f32>(atomicLoad(&grid[first].momentum_z)),
        bitcast<f32>(atomicLoad(&grid[first + 1u].mass)),
        bitcast<f32>(atomicLoad(&grid[first + 1u].momentum_x))
    );
    transfers[arrayLength(&particles) + obstacle_index] = Transfer(
        vec4<f32>(linear, 0.0), vec4<f32>(angular, 0.0),
        vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0),
        vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0)
    );
}
