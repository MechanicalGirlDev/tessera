struct System {
    indices: vec4<u32>,
    inverse: vec4<u32>,
};

struct Sphere {
    indices: vec4<u32>,
    center_radius: vec4<f32>,
    center_of_mass: vec4<f32>,
    plane: vec4<f32>,
    material: vec4<f32>,
    other_center_radius: vec4<f32>,
    other_center_of_mass: vec4<f32>,
    second_axis_end: vec4<f32>,
    first_axis_end: vec4<f32>,
    impulses: vec4<f32>,
    previous_normal: vec4<f32>,
    diagnostic_first: vec4<f32>,
    diagnostic_second: vec4<f32>,
    diagnostic_first_origin: vec4<f32>,
    diagnostic_second_origin: vec4<f32>,
    prescribed_linear: vec4<f32>,
    prescribed_angular: vec4<f32>,
};

struct Pose {
    position: vec4<f32>,
    orientation: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> systems: array<System>;
@group(0) @binding(1) var<storage, read_write> spheres: array<Sphere>;
@group(0) @binding(2) var<storage, read> poses: array<Pose>;
@group(0) @binding(3) var<storage, read> link_terms: array<f32>;
@group(0) @binding(4) var<storage, read> inverse: array<f32>;
@group(0) @binding(5) var<storage, read> velocities: array<f32>;
@group(0) @binding(6) var<storage, read_write> accelerations: array<f32>;
@group(0) @binding(7) var<storage, read_write> state_status: array<atomic<u32>>;
@group(0) @binding(8) var<uniform> contact_policy: vec4<f32>;

fn contact_recovery_velocity(distance: f32, dt: f32) -> f32 {
    return min(
        contact_policy.x * max(-distance - contact_policy.w, 0.0) / dt,
        contact_policy.y);
}

fn quat_rotate(q: vec4<f32>, value: vec3<f32>) -> vec3<f32> {
    let doubled = 2.0 * cross(q.xyz, value);
    return value + q.w * doubled + cross(q.xyz, doubled);
}

struct ArticulatedBox {
    center: vec3<f32>,
    axis_x: vec3<f32>,
    axis_y: vec3<f32>,
    axis_z: vec3<f32>,
    half_extents: vec3<f32>,
};

struct BoxAxisResult {
    separation: f32,
    normal: vec3<f32>,
    axis_index: u32,
};

struct BoxFacePolygon {
    points: array<vec3<f32>, 8>,
    count: u32,
};

struct BoxContactGeometry {
    enabled: bool,
    normal: vec3<f32>,
    point: vec3<f32>,
    distance: f32,
};

struct BoxEdge {
    start: vec3<f32>,
    end: vec3<f32>,
};

struct AxialSimplex {
    a: vec3<f32>,
    b: vec3<f32>,
    c: vec3<f32>,
    d: vec3<f32>,
    count: u32,
    direction: vec3<f32>,
    inside: bool,
};

fn set_box_face_point(polygon: ptr<function, BoxFacePolygon>, index: u32,
    point: vec3<f32>) {
    switch index {
        case 0u: { (*polygon).points[0u] = point; }
        case 1u: { (*polygon).points[1u] = point; }
        case 2u: { (*polygon).points[2u] = point; }
        case 3u: { (*polygon).points[3u] = point; }
        case 4u: { (*polygon).points[4u] = point; }
        case 5u: { (*polygon).points[5u] = point; }
        case 6u: { (*polygon).points[6u] = point; }
        case 7u: { (*polygon).points[7u] = point; }
        default: {}
    }
}

fn make_articulated_box(pose: Pose, local_center: vec3<f32>,
    local_rotation: vec4<f32>, half_extents: vec3<f32>) -> ArticulatedBox {
    return ArticulatedBox(
        pose.position.xyz + quat_rotate(pose.orientation, local_center),
        quat_rotate(pose.orientation, quat_rotate(local_rotation, vec3<f32>(1.0, 0.0, 0.0))),
        quat_rotate(pose.orientation, quat_rotate(local_rotation, vec3<f32>(0.0, 1.0, 0.0))),
        quat_rotate(pose.orientation, quat_rotate(local_rotation, vec3<f32>(0.0, 0.0, 1.0))),
        half_extents);
}

fn articulated_box_radius(shape: ArticulatedBox, axis: vec3<f32>) -> f32 {
    return shape.half_extents.x * abs(dot(shape.axis_x, axis))
        + shape.half_extents.y * abs(dot(shape.axis_y, axis))
        + shape.half_extents.z * abs(dot(shape.axis_z, axis));
}

fn consider_box_axis(first: ArticulatedBox, second: ArticulatedBox,
    delta: vec3<f32>, candidate: vec3<f32>, index: u32,
    best: BoxAxisResult) -> BoxAxisResult {
    let length_squared = dot(candidate, candidate);
    if (length_squared < 1e-10) { return best; }
    let axis = candidate * inverseSqrt(length_squared);
    let projection = dot(delta, axis);
    let separation = abs(projection)
        - articulated_box_radius(first, axis)
        - articulated_box_radius(second, axis);
    if (separation > best.separation) {
        return BoxAxisResult(separation,
            axis * select(-1.0, 1.0, projection >= 0.0), index);
    }
    return best;
}

fn articulated_box_support(shape: ArticulatedBox, direction: vec3<f32>) -> vec3<f32> {
    return shape.center
        + select(-1.0, 1.0, dot(shape.axis_x, direction) >= 0.0)
            * shape.half_extents.x * shape.axis_x
        + select(-1.0, 1.0, dot(shape.axis_y, direction) >= 0.0)
            * shape.half_extents.y * shape.axis_y
        + select(-1.0, 1.0, dot(shape.axis_z, direction) >= 0.0)
            * shape.half_extents.z * shape.axis_z;
}

fn articulated_box_edge(shape: ArticulatedBox, edge_axis: u32,
    direction: vec3<f32>) -> BoxEdge {
    let axes = array<vec3<f32>, 3>(shape.axis_x, shape.axis_y, shape.axis_z);
    var midpoint = shape.center;
    for (var axis = 0u; axis < 3u; axis++) {
        if (axis == edge_axis) { continue; }
        midpoint += select(-1.0, 1.0, dot(axes[axis], direction) >= 0.0)
            * shape.half_extents[axis] * axes[axis];
    }
    let half_axis = shape.half_extents[edge_axis] * axes[edge_axis];
    return BoxEdge(midpoint - half_axis, midpoint + half_axis);
}

fn closest_box_edge_midpoint(first: BoxEdge, second: BoxEdge) -> vec3<f32> {
    let first_axis = first.end - first.start;
    let second_axis = second.end - second.start;
    let offset = first.start - second.start;
    let aa = dot(first_axis, first_axis);
    let bb = dot(first_axis, second_axis);
    let cc = dot(second_axis, second_axis);
    let dd = dot(first_axis, offset);
    let ee = dot(second_axis, offset);
    let denominator = aa * cc - bb * bb;
    if (aa > 1e-12 && denominator > 1e-12 * aa * cc) {
        let first_fraction = (bb * ee - cc * dd) / denominator;
        let second_fraction = (aa * ee - bb * dd) / denominator;
        if (first_fraction >= 0.0 && first_fraction <= 1.0
            && second_fraction >= 0.0 && second_fraction <= 1.0) {
            return 0.5 * (first.start + first_axis * first_fraction
                + second.start + second_axis * second_fraction);
        }
    }
    var best_distance = 1e30;
    var best_midpoint = 0.5 * (first.start + second.start);
    for (var endpoint = 0u; endpoint < 2u; endpoint++) {
        let first_fraction = f32(endpoint);
        var second_fraction = 0.0;
        if (cc > 1e-12) {
            second_fraction = clamp((bb * first_fraction + ee) / cc, 0.0, 1.0);
        }
        let first_point = first.start + first_axis * first_fraction;
        let second_point = second.start + second_axis * second_fraction;
        let distance_squared = dot(first_point - second_point, first_point - second_point);
        if (distance_squared < best_distance) {
            best_distance = distance_squared;
            best_midpoint = 0.5 * (first_point + second_point);
        }
    }
    for (var endpoint = 0u; endpoint < 2u; endpoint++) {
        let second_fraction = f32(endpoint);
        var first_fraction = 0.0;
        if (aa > 1e-12) {
            first_fraction = clamp((bb * second_fraction - dd) / aa, 0.0, 1.0);
        }
        let first_point = first.start + first_axis * first_fraction;
        let second_point = second.start + second_axis * second_fraction;
        let distance_squared = dot(first_point - second_point, first_point - second_point);
        if (distance_squared < best_distance) {
            best_distance = distance_squared;
            best_midpoint = 0.5 * (first_point + second_point);
        }
    }
    return best_midpoint;
}

fn clip_box_face(input: BoxFacePolygon, center: vec3<f32>,
    axis: vec3<f32>, limit: f32) -> BoxFacePolygon {
    var output: BoxFacePolygon;
    if (input.count == 0u) { return output; }
    var previous = input.points[input.count - 1u];
    var previous_distance = dot(previous - center, axis) - limit;
    for (var index = 0u; index < input.count; index++) {
        let current = input.points[index];
        let current_distance = dot(current - center, axis) - limit;
        let previous_inside = previous_distance <= 0.0;
        let current_inside = current_distance <= 0.0;
        if (previous_inside != current_inside && output.count < 8u) {
            let fraction = previous_distance / (previous_distance - current_distance);
            set_box_face_point(&output, output.count,
                previous + (current - previous) * fraction);
            output.count++;
        }
        if (current_inside && output.count < 8u) {
            set_box_face_point(&output, output.count, current);
            output.count++;
        }
        previous = current;
        previous_distance = current_distance;
    }
    return output;
}

fn box_box_contact_geometry(sphere: Sphere, first_pose: Pose, second_pose: Pose,
    row: u32) -> BoxContactGeometry {
    // Modes 8 and 23 store first-box extents in otherwise unused w lanes.
    let first = make_articulated_box(first_pose, sphere.center_radius.xyz,
        sphere.first_axis_end, vec3<f32>(sphere.center_radius.w,
            sphere.plane.w, sphere.center_of_mass.w));
    let second = make_articulated_box(second_pose, sphere.plane.xyz,
        sphere.second_axis_end, sphere.other_center_radius.xyz);
    let first_axes = array<vec3<f32>, 3>(first.axis_x, first.axis_y, first.axis_z);
    let second_axes = array<vec3<f32>, 3>(second.axis_x, second.axis_y, second.axis_z);
    let delta = second.center - first.center;
    var best = BoxAxisResult(-1e30, vec3<f32>(1.0, 0.0, 0.0), 0u);
    for (var axis = 0u; axis < 3u; axis++) {
        best = consider_box_axis(first, second, delta, first_axes[axis], axis, best);
        best = consider_box_axis(first, second, delta, second_axes[axis], axis + 3u, best);
    }
    for (var first_axis = 0u; first_axis < 3u; first_axis++) {
        for (var second_axis = 0u; second_axis < 3u; second_axis++) {
            best = consider_box_axis(first, second, delta,
                cross(first_axes[first_axis], second_axes[second_axis]),
                6u + first_axis * 3u + second_axis, best);
        }
    }
    var fallback_point = vec3<f32>(0.0);
    if (best.axis_index >= 6u) {
        let first_edge = articulated_box_edge(first,
            (best.axis_index - 6u) / 3u, best.normal);
        let second_edge = articulated_box_edge(second,
            (best.axis_index - 6u) % 3u, -best.normal);
        fallback_point = closest_box_edge_midpoint(first_edge, second_edge);
    } else {
        let witness_first = articulated_box_support(first, best.normal);
        let witness_second = articulated_box_support(second, -best.normal);
        fallback_point = 0.5 * (witness_first + witness_second);
    }
    let fallback = BoxContactGeometry(row == 0u, best.normal,
        fallback_point, best.separation);
    if (best.axis_index >= 6u) { return fallback; }

    let reference_is_second = best.axis_index >= 3u;
    var reference = first;
    var incident = second;
    var reference_normal = best.normal;
    if (reference_is_second) {
        reference = second;
        incident = first;
        reference_normal = -best.normal;
    }
    let reference_axes = array<vec3<f32>, 3>(
        reference.axis_x, reference.axis_y, reference.axis_z);
    let incident_axes = array<vec3<f32>, 3>(
        incident.axis_x, incident.axis_y, incident.axis_z);
    let reference_axis = best.axis_index % 3u;
    let face_sign = select(-1.0, 1.0,
        dot(reference_normal, reference_axes[reference_axis]) >= 0.0);
    let face_center = reference.center + face_sign
        * reference.half_extents[reference_axis]
        * reference_axes[reference_axis];
    let reference_u = (reference_axis + 1u) % 3u;
    let reference_v = (reference_axis + 2u) % 3u;
    var incident_axis = 0u;
    var incident_alignment = 0.0;
    for (var axis = 0u; axis < 3u; axis++) {
        let alignment = abs(dot(reference_normal, incident_axes[axis]));
        if (alignment > incident_alignment) {
            incident_axis = axis;
            incident_alignment = alignment;
        }
    }
    let incident_sign = select(1.0, -1.0,
        dot(reference_normal, incident_axes[incident_axis]) >= 0.0);
    let incident_center = incident.center + incident_sign
        * incident.half_extents[incident_axis]
        * incident_axes[incident_axis];
    let incident_u = (incident_axis + 1u) % 3u;
    let incident_v = (incident_axis + 2u) % 3u;
    var polygon: BoxFacePolygon;
    polygon.count = 4u;
    for (var corner = 0u; corner < 4u; corner++) {
        let sign_u = select(-1.0, 1.0, corner == 1u || corner == 2u);
        let sign_v = select(-1.0, 1.0, corner >= 2u);
        set_box_face_point(&polygon, corner, incident_center
            + sign_u * incident.half_extents[incident_u] * incident_axes[incident_u]
            + sign_v * incident.half_extents[incident_v] * incident_axes[incident_v]);
    }
    polygon = clip_box_face(polygon, face_center,
        reference_axes[reference_u], reference.half_extents[reference_u]);
    polygon = clip_box_face(polygon, face_center,
        -reference_axes[reference_u], reference.half_extents[reference_u]);
    polygon = clip_box_face(polygon, face_center,
        reference_axes[reference_v], reference.half_extents[reference_v]);
    polygon = clip_box_face(polygon, face_center,
        -reference_axes[reference_v], reference.half_extents[reference_v]);
    var valid: BoxFacePolygon;
    let tolerance = 1e-6 * max(max(reference.half_extents.x,
        reference.half_extents.y), reference.half_extents.z);
    for (var candidate = 0u; candidate < polygon.count; candidate++) {
        var duplicate = false;
        for (var known = 0u; known < valid.count; known++) {
            let offset = valid.points[known] - polygon.points[candidate];
            duplicate = duplicate || dot(offset, offset) <= tolerance * tolerance;
        }
        if (!duplicate && valid.count < 8u) {
            set_box_face_point(&valid, valid.count, polygon.points[candidate]);
            valid.count++;
        }
    }
    if (valid.count == 0u) { return fallback; }
    let count = min(valid.count, 4u);
    if (row >= count) {
        return BoxContactGeometry(false, best.normal, vec3<f32>(0.0), best.separation);
    }
    let selected = valid.points[(row * valid.count) / count];
    let depth = dot(face_center - selected, reference_normal);
    return BoxContactGeometry(true, best.normal,
        selected + reference_normal * (depth * 0.5), -depth);
}

fn static_axial_sphere_geometry(
    sphere: Sphere, pose: Pose, sphere_center: vec3<f32>) -> BoxContactGeometry {
    let center = pose.position.xyz
        + quat_rotate(pose.orientation, sphere.center_radius.xyz);
    let axis = quat_rotate(pose.orientation,
        quat_rotate(sphere.second_axis_end, vec3<f32>(0.0, 0.0, 1.0)));
    let relative = sphere_center - center;
    let height = dot(relative, axis);
    let radial_vector = relative - axis * height;
    let radial = length(radial_vector);
    var radial_direction: vec3<f32>;
    if (radial > 1e-12) {
        radial_direction = radial_vector / radial;
    } else {
        let reference = select(vec3<f32>(0.0, 1.0, 0.0),
            vec3<f32>(1.0, 0.0, 0.0), abs(axis.x) < 0.9);
        radial_direction = normalize(cross(axis, reference));
    }
    let radius = sphere.other_center_radius.x;
    let half_height = sphere.other_center_radius.y;
    var signed_distance: f32;
    var outward: vec3<f32>;
    if (sphere.material.w == 26.0 || sphere.material.w == 28.0
        || sphere.material.w == 32.0 || sphere.material.w == 36.0
        || sphere.material.w == 60.0 || sphere.material.w == 62.0) {
        let radial_distance = radial - radius;
        let cap_distance = abs(height) - half_height;
        let cap_direction = axis * select(-1.0, 1.0, height >= 0.0);
        let outside = radial_direction * max(radial_distance, 0.0)
            + cap_direction * max(cap_distance, 0.0);
        let outside_length = length(outside);
        if (outside_length > 1e-12) {
            signed_distance = outside_length;
            outward = outside / outside_length;
        } else if (radial_distance >= cap_distance) {
            signed_distance = radial_distance;
            outward = radial_direction;
        } else {
            signed_distance = cap_distance;
            outward = cap_direction;
        }
    } else {
        let point = vec2<f32>(radial, height);
        let base = vec2<f32>(clamp(radial, 0.0, radius), -half_height);
        let side_start = vec2<f32>(radius, -half_height);
        let side_direction = vec2<f32>(-radius, 2.0 * half_height);
        let side_t = clamp(dot(point - side_start, side_direction)
            / dot(side_direction, side_direction), 0.0, 1.0);
        let side = side_start + side_direction * side_t;
        let base_distance_squared = dot(point - base, point - base);
        let side_distance_squared = dot(point - side, point - side);
        let use_base = base_distance_squared <= side_distance_squared;
        let closest = select(side, base, use_base);
        let offset = point - closest;
        let distance = length(offset);
        let edge_epsilon = max(half_height, radius) * 1e-6;
        let inside = height >= -half_height - edge_epsilon
            && height <= half_height + edge_epsilon
            && radial <= radius * (half_height - height) / (2.0 * half_height)
                + edge_epsilon;
        let base_normal = vec2<f32>(0.0, -1.0);
        let side_normal = normalize(vec2<f32>(2.0 * half_height, radius));
        let near_corner = base_distance_squared <= edge_epsilon * edge_epsilon
            && side_distance_squared <= edge_epsilon * edge_epsilon;
        var feature_normal = select(side_normal, base_normal, use_base);
        if (near_corner) { feature_normal = normalize(base_normal + side_normal); }
        let normal_2d = select(offset / max(distance, 1e-12),
            feature_normal, inside || distance <= edge_epsilon);
        outward = radial_direction * normal_2d.x + axis * normal_2d.y;
        signed_distance = select(distance, -distance, inside);
    }
    let surface = sphere_center - outward * signed_distance;
    let sphere_witness = sphere_center - outward * sphere.center_radius.w;
    return BoxContactGeometry(true, outward,
        0.5 * (surface + sphere_witness), signed_distance - sphere.center_radius.w);
}

fn static_axial_capsule_geometry(sphere: Sphere, pose: Pose,
    start: vec3<f32>, end: vec3<f32>) -> BoxContactGeometry {
    let segment = end - start;
    var lower = 0.0;
    var upper = 1.0;
    for (var iteration = 0u; iteration < 24u; iteration++) {
        let fraction = 0.5 * (lower + upper);
        let geometry = static_axial_sphere_geometry(sphere, pose,
            start + segment * fraction);
        if (dot(geometry.normal, segment) < 0.0) {
            lower = fraction;
        } else {
            upper = fraction;
        }
    }
    return static_axial_sphere_geometry(sphere, pose,
        start + segment * (0.5 * (lower + upper)));
}

fn axial_support(center: vec3<f32>, axis: vec3<f32>, half_height: f32,
    radius: f32, kind: f32, direction: vec3<f32>) -> vec3<f32> {
    let axial = dot(direction, axis);
    let radial = direction - axis * axial;
    let radial_squared = dot(radial, radial);
    let rim = select(vec3<f32>(0.0),
        radial * (radius * inverseSqrt(max(radial_squared, 1e-20))),
        radial_squared > 1e-12 * max(dot(direction, direction), 1e-20));
    let base = center - axis * half_height + rim;
    if (kind == 30.0 || kind == 34.0 || kind == 64.0) {
        return center + axis * select(-half_height, half_height, axial >= 0.0) + rim;
    }
    let apex = center + axis * half_height;
    return select(base, apex, dot(apex, direction) > dot(base, direction));
}

// Choose the nearest witness on a supporting face/edge instead of an arbitrary
// corner when the normal has zero tangential components. Support projections
// remain unchanged, while centered face contacts avoid artificial torque.
fn box_support_near(shape: ArticulatedBox, direction: vec3<f32>, reference: vec3<f32>) -> vec3<f32> {
    let delta = reference - shape.center;
    let local = vec3<f32>(dot(delta, shape.axis_x), dot(delta, shape.axis_y), dot(delta, shape.axis_z));
    let projected = vec3<f32>(dot(direction, shape.axis_x), dot(direction, shape.axis_y), dot(direction, shape.axis_z));
    let corner = shape.half_extents * select(vec3<f32>(-1.0), vec3<f32>(1.0), projected >= vec3<f32>(0.0));
    let closest = clamp(local, -shape.half_extents, shape.half_extents);
    let witness = select(closest, corner, abs(projected) > vec3<f32>(1e-6));
    return shape.center + shape.axis_x * witness.x + shape.axis_y * witness.y + shape.axis_z * witness.z;
}

fn axial_box_support(sphere: Sphere, center: vec3<f32>, axis: vec3<f32>,
    shape: ArticulatedBox, direction: vec3<f32>) -> vec3<f32> {
    return axial_support(center, axis, sphere.plane.w, sphere.center_radius.w,
        sphere.material.w, direction) - articulated_box_support(shape, -direction);
}

fn axial_line_direction(a: vec3<f32>, b: vec3<f32>, toward: vec3<f32>) -> vec3<f32> {
    let edge = b - a;
    let perpendicular = cross(cross(edge, toward), edge);
    if (dot(perpendicular, perpendicular) > 1e-16) { return perpendicular; }
    let reference = select(vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(1.0, 0.0, 0.0), abs(edge.x) < 0.9 * length(edge));
    return cross(edge, reference);
}

fn expand_axial_simplex(simplex: AxialSimplex, point: vec3<f32>) -> AxialSimplex {
    let a = point;
    let b = simplex.a;
    let c = simplex.b;
    let d = simplex.c;
    let toward = -a;
    if (simplex.count == 1u) {
        if (dot(b - a, toward) > 0.0) {
            return AxialSimplex(a, b, c, d, 2u,
                axial_line_direction(a, b, toward), false);
        }
        return AxialSimplex(a, b, c, d, 1u, toward, false);
    }
    if (simplex.count == 2u) {
        let ab = b - a;
        let ac = c - a;
        let face = cross(ab, ac);
        if (dot(cross(face, ac), toward) > 0.0) {
            if (dot(ac, toward) > 0.0) {
                return AxialSimplex(a, c, b, d, 2u,
                    axial_line_direction(a, c, toward), false);
            }
            return AxialSimplex(a, b, c, d, 2u,
                axial_line_direction(a, b, toward), false);
        }
        if (dot(cross(ab, face), toward) > 0.0) {
            return AxialSimplex(a, b, c, d, 2u,
                axial_line_direction(a, b, toward), false);
        }
        if (dot(face, toward) > 0.0) {
            return AxialSimplex(a, b, c, d, 3u, face, false);
        }
        return AxialSimplex(a, c, b, d, 3u, -face, false);
    }
    var face = cross(b - a, c - a);
    if (dot(face, d - a) > 0.0) { face = -face; }
    if (dot(face, toward) > 0.0) { return AxialSimplex(a, b, c, d, 3u, face, false); }
    face = cross(c - a, d - a);
    if (dot(face, b - a) > 0.0) { face = -face; }
    if (dot(face, toward) > 0.0) { return AxialSimplex(a, c, d, b, 3u, face, false); }
    face = cross(d - a, b - a);
    if (dot(face, c - a) > 0.0) { face = -face; }
    if (dot(face, toward) > 0.0) { return AxialSimplex(a, d, b, c, 3u, face, false); }
    return AxialSimplex(a, b, c, d, 4u, toward, true);
}

fn axial_box_intersects(sphere: Sphere, center: vec3<f32>, axis: vec3<f32>,
    shape: ArticulatedBox) -> bool {
    var direction = shape.center - center;
    if (dot(direction, direction) < 1e-16) { direction = vec3<f32>(1.0, 0.0, 0.0); }
    let first = axial_box_support(sphere, center, axis, shape, direction);
    var simplex = AxialSimplex(first, vec3<f32>(0.0), vec3<f32>(0.0),
        vec3<f32>(0.0), 1u, -first, false);
    for (var iteration = 0u; iteration < 32u; iteration++) {
        if (dot(simplex.direction, simplex.direction) < 1e-16) { return true; }
        let point = axial_box_support(sphere, center, axis, shape, simplex.direction);
        let projection = dot(point, simplex.direction);
        if (projection < -1e-6) { return false; }
        if (dot(point - simplex.a, point - simplex.a) < 1e-14 ||
            (simplex.count > 1u && dot(point - simplex.b, point - simplex.b) < 1e-14) ||
            (simplex.count > 2u && dot(point - simplex.c, point - simplex.c) < 1e-14)) {
            return abs(projection) < 1e-6;
        }
        simplex = expand_axial_simplex(simplex, point);
        if (simplex.inside) { return true; }
    }
    return false;
}

fn consider_axial_box_axis(sphere: Sphere, center: vec3<f32>, axis: vec3<f32>,
    shape: ArticulatedBox, candidate: vec3<f32>, best: BoxAxisResult) -> BoxAxisResult {
    let squared = dot(candidate, candidate);
    if (squared < 1e-10) { return best; }
    let normal = candidate * inverseSqrt(squared);
    let axial_min = dot(axial_support(center, axis, sphere.plane.w,
        sphere.center_radius.w, sphere.material.w, -normal), normal);
    let axial_max = dot(axial_support(center, axis, sphere.plane.w,
        sphere.center_radius.w, sphere.material.w, normal), normal);
    let box_min = dot(articulated_box_support(shape, -normal), normal);
    let box_max = dot(articulated_box_support(shape, normal), normal);
    let forward = axial_max - box_min;
    let backward = box_max - axial_min;
    let separation = -min(forward, backward);
    if (separation > best.separation) {
        return BoxAxisResult(separation,
            normal * select(-1.0, 1.0, forward < backward), 0u);
    }
    return best;
}

fn static_axial_box_geometry(sphere: Sphere, pose: Pose,
    box_pose: Pose) -> BoxContactGeometry {
    let center = pose.position.xyz
        + quat_rotate(pose.orientation, sphere.center_radius.xyz);
    let axis = quat_rotate(pose.orientation,
        quat_rotate(sphere.first_axis_end, vec3<f32>(0.0, 0.0, 1.0)));
    let shape = make_articulated_box(box_pose, sphere.plane.xyz,
        sphere.second_axis_end, sphere.other_center_radius.xyz);
    let axes = array<vec3<f32>, 3>(shape.axis_x, shape.axis_y, shape.axis_z);
    let delta = shape.center - center;
    var best = BoxAxisResult(-1e30, vec3<f32>(1.0, 0.0, 0.0), 0u);
    best = consider_axial_box_axis(sphere, center, axis, shape, delta, best);
    best = consider_axial_box_axis(sphere, center, axis, shape, axis, best);
    for (var i = 0u; i < 3u; i++) {
        best = consider_axial_box_axis(sphere, center, axis, shape, axes[i], best);
        best = consider_axial_box_axis(sphere, center, axis, shape,
            cross(axis, axes[i]), best);
        for (var side = 0u; side < 2u; side++) {
            let face = shape.center + axes[i] * shape.half_extents[i]
                * select(-1.0, 1.0, side != 0u);
            let relative = face - center;
            let radial = relative - axis * dot(relative, axis);
            if ((sphere.material.w == 31.0 || sphere.material.w == 35.0
                || sphere.material.w == 65.0)
                && dot(radial, radial) > 1e-12) {
                best = consider_axial_box_axis(sphere, center, axis, shape,
                    normalize(radial) * (2.0 * sphere.plane.w)
                        + axis * sphere.center_radius.w, best);
            }
        }
    }
    for (var corner = 0u; corner < 8u; corner++) {
        let x = select(-1.0, 1.0, (corner & 1u) != 0u);
        let y = select(-1.0, 1.0, (corner & 2u) != 0u);
        let z = select(-1.0, 1.0, (corner & 4u) != 0u);
        let vertex = shape.center
            + shape.axis_x * shape.half_extents.x * x
            + shape.axis_y * shape.half_extents.y * y
            + shape.axis_z * shape.half_extents.z * z;
        let relative = vertex - center;
        let radial = relative - axis * dot(relative, axis);
        best = consider_axial_box_axis(sphere, center, axis, shape, radial, best);
        if ((sphere.material.w == 31.0 || sphere.material.w == 35.0
            || sphere.material.w == 65.0)
            && dot(radial, radial) > 1e-12) {
            best = consider_axial_box_axis(sphere, center, axis, shape,
                normalize(radial) * (2.0 * sphere.plane.w)
                    + axis * sphere.center_radius.w, best);
        }
    }
    let witness_axial = axial_support(center, axis, sphere.plane.w,
        sphere.center_radius.w, sphere.material.w, best.normal);
    let witness_box = box_support_near(shape, -best.normal, witness_axial);
    let point = 0.5 * (witness_axial + witness_box);
    let enabled = best.separation > 0.0
        || axial_box_intersects(sphere, center, axis, shape);
    return BoxContactGeometry(enabled, best.normal, point, best.separation);
}

struct AxialShape {
    center: vec3<f32>,
    axis: vec3<f32>,
    half_height: f32,
    radius: f32,
    kind: f32,
}

fn axial_shape_support(shape: AxialShape, direction: vec3<f32>) -> vec3<f32> {
    return axial_support(shape.center, shape.axis, shape.half_height,
        shape.radius, shape.kind, direction);
}

fn axial_pair_support(first: AxialShape, second: AxialShape,
    direction: vec3<f32>) -> vec3<f32> {
    return axial_shape_support(first, direction)
        - axial_shape_support(second, -direction);
}

fn axial_pair_intersects(first: AxialShape, second: AxialShape) -> bool {
    var direction = second.center - first.center;
    if (dot(direction, direction) < 1e-16) { direction = vec3<f32>(1.0, 0.0, 0.0); }
    let support = axial_pair_support(first, second, direction);
    var simplex = AxialSimplex(support, vec3<f32>(0.0), vec3<f32>(0.0),
        vec3<f32>(0.0), 1u, -support, false);
    for (var iteration = 0u; iteration < 32u; iteration++) {
        if (dot(simplex.direction, simplex.direction) < 1e-16) { return true; }
        let point = axial_pair_support(first, second, simplex.direction);
        let projection = dot(point, simplex.direction);
        if (projection < -1e-6) { return false; }
        if (dot(point - simplex.a, point - simplex.a) < 1e-14 ||
            (simplex.count > 1u && dot(point - simplex.b, point - simplex.b) < 1e-14) ||
            (simplex.count > 2u && dot(point - simplex.c, point - simplex.c) < 1e-14)) {
            return abs(projection) < 1e-6;
        }
        simplex = expand_axial_simplex(simplex, point);
        if (simplex.inside) { return true; }
    }
    return false;
}

fn consider_axial_pair_axis(first: AxialShape, second: AxialShape,
    candidate: vec3<f32>, best: BoxAxisResult) -> BoxAxisResult {
    let squared = dot(candidate, candidate);
    if (squared < 1e-10) { return best; }
    let normal = candidate * inverseSqrt(squared);
    let first_min = dot(axial_shape_support(first, -normal), normal);
    let first_max = dot(axial_shape_support(first, normal), normal);
    let second_min = dot(axial_shape_support(second, -normal), normal);
    let second_max = dot(axial_shape_support(second, normal), normal);
    let forward = first_max - second_min;
    let backward = second_max - first_min;
    let separation = -min(forward, backward);
    if (separation > best.separation) {
        return BoxAxisResult(separation,
            normal * select(-1.0, 1.0, forward < backward), 0u);
    }
    return best;
}

fn axial_pair_geometry(sphere: Sphere, first_pose: Pose,
    second_pose: Pose) -> BoxContactGeometry {
    let first = AxialShape(
        first_pose.position.xyz
            + quat_rotate(first_pose.orientation, sphere.center_radius.xyz),
        quat_rotate(first_pose.orientation,
            quat_rotate(sphere.first_axis_end, vec3<f32>(0.0, 0.0, 1.0))),
        sphere.plane.w, sphere.center_radius.w,
        select(31.0, 30.0, sphere.material.w == 38.0 || sphere.material.w == 39.0
            || sphere.material.w == 66.0 || sphere.material.w == 67.0));
    let second = AxialShape(
        second_pose.position.xyz
            + quat_rotate(second_pose.orientation, sphere.plane.xyz),
        quat_rotate(second_pose.orientation,
            quat_rotate(sphere.second_axis_end, vec3<f32>(0.0, 0.0, 1.0))),
        sphere.other_center_radius.y, sphere.other_center_radius.x,
        select(31.0, 30.0, sphere.material.w == 38.0 || sphere.material.w == 40.0
            || sphere.material.w == 66.0 || sphere.material.w == 68.0));
    let delta = second.center - first.center;
    var best = BoxAxisResult(-1e30, vec3<f32>(1.0, 0.0, 0.0), 0u);
    best = consider_axial_pair_axis(first, second, delta, best);
    best = consider_axial_pair_axis(first, second, first.axis, best);
    best = consider_axial_pair_axis(first, second, second.axis, best);
    best = consider_axial_pair_axis(first, second,
        cross(first.axis, second.axis), best);
    let first_radial = delta - first.axis * dot(delta, first.axis);
    let second_radial = delta - second.axis * dot(delta, second.axis);
    best = consider_axial_pair_axis(first, second, first_radial, best);
    best = consider_axial_pair_axis(first, second, second_radial, best);
    if (first.kind == 31.0 && dot(first_radial, first_radial) > 1e-12) {
        best = consider_axial_pair_axis(first, second,
            normalize(first_radial) * (2.0 * first.half_height)
                + first.axis * first.radius, best);
    }
    if (second.kind == 31.0 && dot(second_radial, second_radial) > 1e-12) {
        best = consider_axial_pair_axis(first, second,
            normalize(second_radial) * (2.0 * second.half_height)
                + second.axis * second.radius, best);
    }
    let first_witness = axial_shape_support(first, best.normal);
    let second_witness = axial_shape_support(second, -best.normal);
    let point = 0.5 * (first_witness + second_witness);
    let enabled = best.separation > 0.0 || axial_pair_intersects(first, second);
    return BoxContactGeometry(enabled, best.normal, point, best.separation);
}

fn closest_mesh_triangle(point: vec3<f32>, a: vec3<f32>, b: vec3<f32>,
    c: vec3<f32>) -> vec3<f32> {
    let ab = b - a;
    let ac = c - a;
    let ap = point - a;
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) { return a; }
    let bp = point - b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) { return b; }
    let vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        return a + ab * (d1 / (d1 - d3));
    }
    let cp = point - c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) { return c; }
    let vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        return a + ac * (d2 / (d2 - d6));
    }
    let va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0) {
        let edge = c - b;
        return b + edge * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    let inverse = 1.0 / (va + vb + vc);
    return a + ab * (vb * inverse) + ac * (vc * inverse);
}

struct MeshSegmentWitness {
    segment: vec3<f32>,
    triangle: vec3<f32>,
};

fn closest_mesh_segments(start: vec3<f32>, end: vec3<f32>,
    edge_start: vec3<f32>, edge_end: vec3<f32>) -> MeshSegmentWitness {
    let u = end - start;
    let v = edge_end - edge_start;
    let w = start - edge_start;
    let aa = dot(u, u);
    let bb = dot(u, v);
    let cc = dot(v, v);
    let dd = dot(u, w);
    let ee = dot(v, w);
    if (aa < 1e-12) {
        let t = clamp(ee / max(cc, 1e-12), 0.0, 1.0);
        return MeshSegmentWitness(start, edge_start + v * t);
    }
    if (cc < 1e-12) {
        let s = clamp(-dd / aa, 0.0, 1.0);
        return MeshSegmentWitness(start + u * s, edge_start);
    }
    let denominator = aa * cc - bb * bb;
    var s = 0.0;
    if (denominator > 1e-12) {
        s = clamp((bb * ee - cc * dd) / denominator, 0.0, 1.0);
    }
    var t = (bb * s + ee) / cc;
    if (t < 0.0) {
        t = 0.0;
        s = clamp(-dd / aa, 0.0, 1.0);
    } else if (t > 1.0) {
        t = 1.0;
        s = clamp((bb - dd) / aa, 0.0, 1.0);
    }
    return MeshSegmentWitness(start + u * s, edge_start + v * t);
}

fn closest_mesh_segment_triangle(start: vec3<f32>, end: vec3<f32>,
    a: vec3<f32>, b: vec3<f32>, c: vec3<f32>,
    face: vec3<f32>) -> MeshSegmentWitness {
    let closest_start = closest_mesh_triangle(start, a, b, c);
    var best = MeshSegmentWitness(start, closest_start);
    var delta = best.segment - best.triangle;
    var best_distance = dot(delta, delta);
    let closest_end = closest_mesh_triangle(end, a, b, c);
    delta = end - closest_end;
    let end_distance = dot(delta, delta);
    if (end_distance < best_distance) {
        best = MeshSegmentWitness(end, closest_end);
        best_distance = end_distance;
    }
    let ab = closest_mesh_segments(start, end, a, b);
    delta = ab.segment - ab.triangle;
    let ab_distance = dot(delta, delta);
    if (ab_distance < best_distance) {
        best = ab;
        best_distance = ab_distance;
    }
    let bc = closest_mesh_segments(start, end, b, c);
    delta = bc.segment - bc.triangle;
    let bc_distance = dot(delta, delta);
    if (bc_distance < best_distance) {
        best = bc;
        best_distance = bc_distance;
    }
    let ca = closest_mesh_segments(start, end, c, a);
    delta = ca.segment - ca.triangle;
    let ca_distance = dot(delta, delta);
    if (ca_distance < best_distance) { best = ca; }
    let direction = end - start;
    let denominator = dot(face, direction);
    if (abs(denominator) > 1e-10) {
        let t = dot(face, a - start) / denominator;
        if (t >= 0.0 && t <= 1.0) {
            let point = start + direction * t;
            let tolerance = -1e-6 * dot(face, face);
            if (dot(cross(b - a, point - a), face) >= tolerance &&
                dot(cross(c - b, point - b), face) >= tolerance &&
                dot(cross(a - c, point - c), face) >= tolerance) {
                return MeshSegmentWitness(point, point);
            }
        }
    }
    return best;
}

fn mesh_capsule_candidate(segment_point: vec3<f32>, triangle_point: vec3<f32>,
    face: vec3<f32>, face_point: vec3<f32>, capsule_center: vec3<f32>,
    radius: f32, mesh_translation: vec3<f32>, mesh_rotation: vec4<f32>)
    -> BoxContactGeometry {
    let offset = segment_point - triangle_point;
    let distance_squared = dot(offset, offset);
    // Retain near-touching witnesses for the speculative velocity constraint.
    // Keep the actual separation so the solver does not attract separated bodies.
    if (distance_squared > (radius + 1e-6) * (radius + 1e-6)) {
        return BoxContactGeometry(false, vec3<f32>(1.0, 0.0, 0.0),
            vec3<f32>(0.0), 1.0);
    }
    let distance = sqrt(distance_squared);
    var mesh_to_capsule = normalize(face);
    if (distance > 1e-7) {
        mesh_to_capsule = offset / distance;
    } else if (dot(capsule_center - face_point, mesh_to_capsule) < 0.0) {
        mesh_to_capsule = -mesh_to_capsule;
    }
    let capsule_witness = segment_point - mesh_to_capsule * radius;
    let local_point = 0.5 * (triangle_point + capsule_witness);
    return BoxContactGeometry(true,
        -quat_rotate(mesh_rotation, mesh_to_capsule),
        mesh_translation + quat_rotate(mesh_rotation, local_point),
        distance - radius);
}

fn scene_mesh_sphere_geometry(sphere: Sphere, pose: Pose) -> BoxContactGeometry {
    let center = pose.position.xyz
        + quat_rotate(pose.orientation, sphere.center_radius.xyz);
    let mesh_rotation = sphere.first_axis_end;
    let inverse_rotation = vec4<f32>(-mesh_rotation.xyz, mesh_rotation.w);
    let local_center = quat_rotate(inverse_rotation, center - sphere.plane.xyz);
    let radius = sphere.center_radius.w;
    let vertex_start = u32(sphere.other_center_radius.x);
    let triangle_start = u32(sphere.other_center_radius.y);
    let node_start = u32(sphere.other_center_radius.z);
    let node_count = u32(sphere.other_center_radius.w);
    let absent = BoxContactGeometry(false, vec3<f32>(1.0, 0.0, 0.0), center, 1.0);
    var first = absent;
    var second = absent;
    var third = absent;
    var fourth = absent;
    var cursor = 0u;
    while (cursor < node_count) {
        let node = spheres[node_start + cursor];
        let margin = vec3<f32>(radius + 1e-4);
        if (any(local_center < node.center_radius.xyz - margin)
            || any(local_center > node.plane.xyz + margin)) {
            cursor = node.indices.y;
            continue;
        }
        cursor++;
        if (node.indices.x == 0xffffffffu) { continue; }
        let triangle = spheres[triangle_start + node.indices.x].indices;
        let a = spheres[vertex_start + triangle.x].center_radius.xyz;
        let b = spheres[vertex_start + triangle.y].center_radius.xyz;
        let c = spheres[vertex_start + triangle.z].center_radius.xyz;
        let face = cross(b - a, c - a);
        if (dot(face, face) < 1e-20) { continue; }
        let closest = closest_mesh_triangle(local_center, a, b, c);
        let offset = local_center - closest;
        let distance_squared = dot(offset, offset);
        // Preserve near-touching candidates through f32 roundoff.
        if (distance_squared > (radius + 1e-6) * (radius + 1e-6)) { continue; }
        let distance = sqrt(distance_squared);
        var mesh_to_sphere = normalize(face);
        if (distance > 1e-7) {
            mesh_to_sphere = offset / distance;
        } else if (dot(local_center - a, mesh_to_sphere) < 0.0) {
            mesh_to_sphere = -mesh_to_sphere;
        }
        let world_normal = quat_rotate(mesh_rotation, mesh_to_sphere);
        let world_closest = sphere.plane.xyz + quat_rotate(mesh_rotation, closest);
        let witness = center - world_normal * radius;
        let depth = radius - distance;
        let candidate = BoxContactGeometry(true, -world_normal,
            0.5 * (world_closest + witness), -depth);
        let duplicate = (first.enabled
                && dot(candidate.point - first.point, candidate.point - first.point) <= 1e-10)
            || (second.enabled
                && dot(candidate.point - second.point, candidate.point - second.point) <= 1e-10)
            || (third.enabled
                && dot(candidate.point - third.point, candidate.point - third.point) <= 1e-10)
            || (fourth.enabled
                && dot(candidate.point - fourth.point, candidate.point - fourth.point) <= 1e-10);
        if (duplicate) { continue; }
        if (!first.enabled || candidate.distance < first.distance) {
            fourth = third;
            third = second;
            second = first;
            first = candidate;
        } else if (!second.enabled || candidate.distance < second.distance) {
            fourth = third;
            third = second;
            second = candidate;
        } else if (!third.enabled || candidate.distance < third.distance) {
            fourth = third;
            third = candidate;
        } else if (!fourth.enabled || candidate.distance < fourth.distance) {
            fourth = candidate;
        }
    }
    let slot = u32(sphere.second_axis_end.w);
    if (slot == 0u) { return first; }
    if (slot == 1u) { return second; }
    if (slot == 2u) { return third; }
    return fourth;
}

fn scene_polyline_sphere_geometry(sphere: Sphere, pose: Pose) -> BoxContactGeometry {
    let center = pose.position.xyz
        + quat_rotate(pose.orientation, sphere.center_radius.xyz);
    let polyline_rotation = sphere.first_axis_end;
    let inverse_rotation = vec4<f32>(-polyline_rotation.xyz, polyline_rotation.w);
    let local_center = quat_rotate(inverse_rotation, center - sphere.plane.xyz);
    let radius = sphere.center_radius.w;
    let vertex_start = u32(sphere.other_center_radius.x);
    let segment_start = u32(sphere.other_center_radius.y);
    let node_start = u32(sphere.other_center_radius.z);
    let node_count = u32(sphere.other_center_radius.w);
    let absent = BoxContactGeometry(false, vec3<f32>(1.0, 0.0, 0.0), center, 1.0);
    var first = absent;
    var second = absent;
    var third = absent;
    var fourth = absent;
    var cursor = 0u;
    while (cursor < node_count) {
        let node = spheres[node_start + cursor];
        let margin = vec3<f32>(radius + 1e-4);
        if (any(local_center < node.center_radius.xyz - margin)
            || any(local_center > node.plane.xyz + margin)) {
            cursor = node.indices.y;
            continue;
        }
        cursor++;
        if (node.indices.x == 0xffffffffu) { continue; }
        let segment = spheres[segment_start + node.indices.x].indices;
        let a = spheres[vertex_start + segment.x].center_radius.xyz;
        let b = spheres[vertex_start + segment.y].center_radius.xyz;
        let axis = b - a;
        let fraction = clamp(dot(local_center - a, axis) / max(dot(axis, axis), 1e-20),
            0.0, 1.0);
        let closest = a + axis * fraction;
        let offset = local_center - closest;
        let distance_squared = dot(offset, offset);
        // Preserve near-touching candidates through f32 roundoff.
        if (distance_squared > (radius + 1e-6) * (radius + 1e-6)) { continue; }
        let distance = sqrt(distance_squared);
        var line_to_sphere = vec3<f32>(1.0, 0.0, 0.0);
        if (distance > 1e-7) {
            line_to_sphere = offset / distance;
        } else {
            let reference = select(vec3<f32>(0.0, 0.0, 1.0),
                vec3<f32>(1.0, 0.0, 0.0), abs(axis.z) > 0.9 * length(axis));
            line_to_sphere = normalize(cross(axis, reference));
        }
        let world_normal = quat_rotate(polyline_rotation, line_to_sphere);
        let world_closest = sphere.plane.xyz + quat_rotate(polyline_rotation, closest);
        let witness = center - world_normal * radius;
        let candidate = BoxContactGeometry(true, -world_normal,
            0.5 * (world_closest + witness), distance - radius);
        let duplicate = (first.enabled
                && dot(candidate.point - first.point, candidate.point - first.point) <= 1e-10)
            || (second.enabled
                && dot(candidate.point - second.point, candidate.point - second.point) <= 1e-10)
            || (third.enabled
                && dot(candidate.point - third.point, candidate.point - third.point) <= 1e-10)
            || (fourth.enabled
                && dot(candidate.point - fourth.point, candidate.point - fourth.point) <= 1e-10);
        if (duplicate) { continue; }
        if (!first.enabled || candidate.distance < first.distance) {
            fourth = third;
            third = second;
            second = first;
            first = candidate;
        } else if (!second.enabled || candidate.distance < second.distance) {
            fourth = third;
            third = second;
            second = candidate;
        } else if (!third.enabled || candidate.distance < third.distance) {
            fourth = third;
            third = candidate;
        } else if (!fourth.enabled || candidate.distance < fourth.distance) {
            fourth = candidate;
        }
    }
    let slot = u32(sphere.second_axis_end.w);
    if (slot == 0u) { return first; }
    if (slot == 1u) { return second; }
    if (slot == 2u) { return third; }
    return fourth;
}

fn scene_mesh_capsule_geometry(sphere: Sphere, pose: Pose) -> BoxContactGeometry {
    let world_a = pose.position.xyz
        + quat_rotate(pose.orientation, sphere.center_radius.xyz);
    let world_b = pose.position.xyz
        + quat_rotate(pose.orientation, sphere.plane.xyz);
    let mesh_translation = sphere.second_axis_end.xyz;
    let mesh_rotation = sphere.first_axis_end;
    let inverse_rotation = vec4<f32>(-mesh_rotation.xyz, mesh_rotation.w);
    let local_a = quat_rotate(inverse_rotation, world_a - mesh_translation);
    let local_b = quat_rotate(inverse_rotation, world_b - mesh_translation);
    let capsule_center = 0.5 * (local_a + local_b);
    let radius = sphere.center_radius.w;
    let vertex_start = u32(sphere.other_center_radius.x);
    let triangle_start = u32(sphere.other_center_radius.y);
    let node_start = u32(sphere.other_center_radius.z);
    let node_count = u32(sphere.other_center_radius.w);
    let absent = BoxContactGeometry(false, vec3<f32>(1.0, 0.0, 0.0),
        0.5 * (world_a + world_b), 1.0);
    var first = absent;
    var second = absent;
    var third = absent;
    var fourth = absent;
    let distinct_squared = max(1e-8, dot(local_b - local_a, local_b - local_a) * 1e-4);
    var cursor = 0u;
    while (cursor < node_count) {
        let node = spheres[node_start + cursor];
        let margin = vec3<f32>(radius + 1e-4);
        if (any(max(local_a, local_b) < node.center_radius.xyz - margin)
            || any(min(local_a, local_b) > node.plane.xyz + margin)) {
            cursor = node.indices.y;
            continue;
        }
        cursor++;
        if (node.indices.x == 0xffffffffu) { continue; }
        let triangle = spheres[triangle_start + node.indices.x].indices;
        let a = spheres[vertex_start + triangle.x].center_radius.xyz;
        let b = spheres[vertex_start + triangle.y].center_radius.xyz;
        let c = spheres[vertex_start + triangle.z].center_radius.xyz;
        let face = cross(b - a, c - a);
        if (dot(face, face) < 1e-20) { continue; }
        for (var candidate_index = 0u; candidate_index < 3u; candidate_index++) {
            var witness = closest_mesh_segment_triangle(local_a, local_b, a, b, c, face);
            if (candidate_index == 1u) {
                witness = MeshSegmentWitness(local_a, closest_mesh_triangle(local_a, a, b, c));
            } else if (candidate_index == 2u) {
                witness = MeshSegmentWitness(local_b, closest_mesh_triangle(local_b, a, b, c));
            }
            let candidate = mesh_capsule_candidate(witness.segment, witness.triangle,
                face, a, capsule_center, radius, mesh_translation, mesh_rotation);
            if (!candidate.enabled) { continue; }
            let duplicate = (first.enabled
                    && dot(candidate.point - first.point, candidate.point - first.point) <= distinct_squared)
                || (second.enabled
                    && dot(candidate.point - second.point, candidate.point - second.point) <= distinct_squared)
                || (third.enabled
                    && dot(candidate.point - third.point, candidate.point - third.point) <= distinct_squared)
                || (fourth.enabled
                    && dot(candidate.point - fourth.point, candidate.point - fourth.point) <= distinct_squared);
            if (duplicate) { continue; }
            if (!first.enabled || candidate.distance < first.distance) {
                fourth = third;
                third = second;
                second = first;
                first = candidate;
            } else if (!second.enabled || candidate.distance < second.distance) {
                fourth = third;
                third = second;
                second = candidate;
            } else if (!third.enabled || candidate.distance < third.distance) {
                fourth = third;
                third = candidate;
            } else if (!fourth.enabled || candidate.distance < fourth.distance) {
                fourth = candidate;
            }
        }
    }
    let slot = u32(sphere.second_axis_end.w);
    if (slot == 0u) { return first; }
    if (slot == 1u) { return second; }
    if (slot == 2u) { return third; }
    return fourth;
}

fn scene_polyline_capsule_geometry(sphere: Sphere, pose: Pose) -> BoxContactGeometry {
    let world_a = pose.position.xyz
        + quat_rotate(pose.orientation, sphere.center_radius.xyz);
    let world_b = pose.position.xyz
        + quat_rotate(pose.orientation, sphere.plane.xyz);
    let line_translation = sphere.second_axis_end.xyz;
    let line_rotation = sphere.first_axis_end;
    let inverse_rotation = vec4<f32>(-line_rotation.xyz, line_rotation.w);
    let local_a = quat_rotate(inverse_rotation, world_a - line_translation);
    let local_b = quat_rotate(inverse_rotation, world_b - line_translation);
    let capsule_center = 0.5 * (local_a + local_b);
    let radius = sphere.center_radius.w;
    let vertex_start = u32(sphere.other_center_radius.x);
    let segment_start = u32(sphere.other_center_radius.y);
    let node_start = u32(sphere.other_center_radius.z);
    let node_count = u32(sphere.other_center_radius.w);
    let absent = BoxContactGeometry(false, vec3<f32>(1.0, 0.0, 0.0),
        0.5 * (world_a + world_b), 1.0);
    var first = absent;
    var second = absent;
    var third = absent;
    var fourth = absent;
    let distinct_squared = max(1e-8, dot(local_b - local_a, local_b - local_a) * 1e-4);
    var cursor = 0u;
    while (cursor < node_count) {
        let node = spheres[node_start + cursor];
        let margin = vec3<f32>(radius + 1e-4);
        if (any(max(local_a, local_b) < node.center_radius.xyz - margin)
            || any(min(local_a, local_b) > node.plane.xyz + margin)) {
            cursor = node.indices.y;
            continue;
        }
        cursor++;
        if (node.indices.x == 0xffffffffu) { continue; }
        let segment = spheres[segment_start + node.indices.x].indices;
        let a = spheres[vertex_start + segment.x].center_radius.xyz;
        let b = spheres[vertex_start + segment.y].center_radius.xyz;
        let edge = b - a;
        let reference = select(vec3<f32>(0.0, 0.0, 1.0),
            vec3<f32>(1.0, 0.0, 0.0), abs(edge.z) > 0.9 * length(edge));
        let fallback_normal = cross(edge, reference);
        for (var candidate_index = 0u; candidate_index < 3u; candidate_index++) {
            var witness = closest_mesh_segments(local_a, local_b, a, b);
            if (candidate_index == 1u) {
                let fraction = clamp(dot(local_a - a, edge) / max(dot(edge, edge), 1e-20),
                    0.0, 1.0);
                witness = MeshSegmentWitness(local_a, a + edge * fraction);
            } else if (candidate_index == 2u) {
                let fraction = clamp(dot(local_b - a, edge) / max(dot(edge, edge), 1e-20),
                    0.0, 1.0);
                witness = MeshSegmentWitness(local_b, a + edge * fraction);
            }
            let candidate = mesh_capsule_candidate(witness.segment, witness.triangle,
                fallback_normal, a, capsule_center, radius,
                line_translation, line_rotation);
            if (!candidate.enabled) { continue; }
            let duplicate = (first.enabled
                    && dot(candidate.point - first.point, candidate.point - first.point) <= distinct_squared)
                || (second.enabled
                    && dot(candidate.point - second.point, candidate.point - second.point) <= distinct_squared)
                || (third.enabled
                    && dot(candidate.point - third.point, candidate.point - third.point) <= distinct_squared)
                || (fourth.enabled
                    && dot(candidate.point - fourth.point, candidate.point - fourth.point) <= distinct_squared);
            if (duplicate) { continue; }
            if (!first.enabled || candidate.distance < first.distance) {
                fourth = third;
                third = second;
                second = first;
                first = candidate;
            } else if (!second.enabled || candidate.distance < second.distance) {
                fourth = third;
                third = second;
                second = candidate;
            } else if (!third.enabled || candidate.distance < third.distance) {
                fourth = third;
                third = candidate;
            } else if (!fourth.enabled || candidate.distance < fourth.distance) {
                fourth = candidate;
            }
        }
    }
    let slot = u32(sphere.second_axis_end.w);
    if (slot == 0u) { return first; }
    if (slot == 1u) { return second; }
    if (slot == 2u) { return third; }
    return fourth;
}

fn consider_polyline_box_axis(a: vec3<f32>, b: vec3<f32>, half_extents: vec3<f32>,
    candidate: vec3<f32>, best: BoxAxisResult) -> BoxAxisResult {
    let squared = dot(candidate, candidate);
    if (squared < 1e-12) { return best; }
    let normal = candidate * inverseSqrt(squared);
    let radius = dot(abs(normal), half_extents);
    let first = dot(a, normal);
    let second = dot(b, normal);
    let forward = min(first, second) - radius;
    let backward = -radius - max(first, second);
    let separation = max(forward, backward);
    if (separation > best.separation) {
        return BoxAxisResult(separation, normal * select(-1.0, 1.0, forward >= backward), 0u);
    }
    return best;
}

fn scene_polyline_box_geometry(sphere: Sphere, pose: Pose) -> BoxContactGeometry {
    let box_world = make_articulated_box(pose, sphere.center_radius.xyz, sphere.plane,
        vec3<f32>(sphere.center_radius.w, sphere.center_of_mass.w,
            sphere.other_center_of_mass.w));
    let line_translation = sphere.second_axis_end.xyz;
    let line_rotation = sphere.first_axis_end;
    let inverse_line_rotation = vec4<f32>(-line_rotation.xyz, line_rotation.w);
    let box_center = quat_rotate(inverse_line_rotation,
        box_world.center - line_translation);
    let reach = abs(quat_rotate(inverse_line_rotation, box_world.axis_x))
            * box_world.half_extents.x
        + abs(quat_rotate(inverse_line_rotation, box_world.axis_y))
            * box_world.half_extents.y
        + abs(quat_rotate(inverse_line_rotation, box_world.axis_z))
            * box_world.half_extents.z + vec3<f32>(1e-4);
    let vertex_start = u32(sphere.other_center_radius.x);
    let segment_start = u32(sphere.other_center_radius.y);
    let node_start = u32(sphere.other_center_radius.z);
    let node_count = u32(sphere.other_center_radius.w);
    let absent = BoxContactGeometry(false, vec3<f32>(1.0, 0.0, 0.0),
        box_world.center, 1.0);
    var first = absent;
    var second = absent;
    var third = absent;
    var fourth = absent;
    let distinct_squared = max(1e-8,
        dot(box_world.half_extents, box_world.half_extents) * 1e-4);
    var cursor = 0u;
    while (cursor < node_count) {
        let node = spheres[node_start + cursor];
        if (any(box_center + reach < node.center_radius.xyz)
            || any(box_center - reach > node.plane.xyz)) {
            cursor = node.indices.y;
            continue;
        }
        cursor++;
        if (node.indices.x == 0xffffffffu) { continue; }
        let segment = spheres[segment_start + node.indices.x].indices;
        let a = spheres[vertex_start + segment.x].center_radius.xyz;
        let b = spheres[vertex_start + segment.y].center_radius.xyz;
        let world_a = line_translation + quat_rotate(line_rotation, a);
        let world_b = line_translation + quat_rotate(line_rotation, b);
        let offset_a = world_a - box_world.center;
        let offset_b = world_b - box_world.center;
        let local_a = vec3<f32>(dot(offset_a, box_world.axis_x),
            dot(offset_a, box_world.axis_y), dot(offset_a, box_world.axis_z));
        let local_b = vec3<f32>(dot(offset_b, box_world.axis_x),
            dot(offset_b, box_world.axis_y), dot(offset_b, box_world.axis_z));
        let local_axis = local_b - local_a;
        let half_extents = box_world.half_extents;
        var best = BoxAxisResult(-1e30, vec3<f32>(1.0, 0.0, 0.0), 0u);
        let axes = array<vec3<f32>, 3>(vec3<f32>(1.0, 0.0, 0.0),
            vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0));
        for (var axis = 0u; axis < 3u; axis++) {
            best = consider_polyline_box_axis(local_a, local_b, half_extents, axes[axis], best);
            best = consider_polyline_box_axis(local_a, local_b, half_extents,
                cross(local_axis, axes[axis]), best);
        }
        if (best.separation > 1e-6) { continue; }
        let support_plane = dot(abs(best.normal), half_extents);
        let projected_a = local_a + best.normal * (support_plane - dot(local_a, best.normal));
        let face_axis = local_axis - best.normal * dot(local_axis, best.normal);
        var lower = 0.0;
        var upper = 1.0;
        var clipped = true;
        // Clip the line on the supporting feature, retaining both ends of face contacts.
        for (var axis = 0u; axis < 3u; axis++) {
            let slope = face_axis[axis];
            if (abs(slope) < 1e-8) {
                if (abs(projected_a[axis]) > half_extents[axis] + 1e-6) { clipped = false; }
            } else {
                // Near an edge, a tiny face slope amplifies f32 error in the interval.
                // Use the same geometric tolerance as the near-contact separation test.
                let first_fraction = (-half_extents[axis] - 1e-6 - projected_a[axis]) / slope;
                let second_fraction = (half_extents[axis] + 1e-6 - projected_a[axis]) / slope;
                lower = max(lower, min(first_fraction, second_fraction));
                upper = min(upper, max(first_fraction, second_fraction));
            }
        }
        if (!clipped || upper < lower - 1e-6) { continue; }
        if (dot(face_axis, face_axis) < 1e-12) {
            lower = select(1.0, 0.0, dot(local_axis, best.normal) >= 0.0);
            upper = lower;
        } else if (upper < lower) {
            lower = clamp(0.5 * (lower + upper), 0.0, 1.0);
            upper = lower;
        }
        for (var endpoint = 0u; endpoint < 2u; endpoint++) {
            let fraction = select(lower, upper, endpoint == 1u);
            let line_point = local_a + local_axis * fraction;
            let distance = dot(line_point, best.normal) - support_plane;
            let box_point = line_point - best.normal * distance;
            let midpoint = 0.5 * (line_point + box_point);
            let candidate = BoxContactGeometry(true,
                box_world.axis_x * best.normal.x
                    + box_world.axis_y * best.normal.y + box_world.axis_z * best.normal.z,
                box_world.center + box_world.axis_x * midpoint.x
                    + box_world.axis_y * midpoint.y + box_world.axis_z * midpoint.z,
                distance);
            let duplicate = (first.enabled
                    && dot(candidate.point - first.point, candidate.point - first.point) <= distinct_squared)
                || (second.enabled
                    && dot(candidate.point - second.point, candidate.point - second.point) <= distinct_squared)
                || (third.enabled
                    && dot(candidate.point - third.point, candidate.point - third.point) <= distinct_squared)
                || (fourth.enabled
                    && dot(candidate.point - fourth.point, candidate.point - fourth.point) <= distinct_squared);
            if (duplicate) { continue; }
            if (!first.enabled || candidate.distance < first.distance) {
                fourth = third;
                third = second;
                second = first;
                first = candidate;
            } else if (!second.enabled || candidate.distance < second.distance) {
                fourth = third;
                third = second;
                second = candidate;
            } else if (!third.enabled || candidate.distance < third.distance) {
                fourth = third;
                third = candidate;
            } else if (!fourth.enabled || candidate.distance < fourth.distance) {
                fourth = candidate;
            }
        }
    }
    let slot = u32(sphere.second_axis_end.w);
    if (slot == 0u) { return first; }
    if (slot == 1u) { return second; }
    if (slot == 2u) { return third; }
    return fourth;
}

fn mesh_box_axis(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>,
    shape: ArticulatedBox, candidate: vec3<f32>, best: BoxAxisResult) -> BoxAxisResult {
    let length_squared = dot(candidate, candidate);
    if (length_squared < 1e-10 || best.separation > 0.0) { return best; }
    let axis = candidate * inverseSqrt(length_squared);
    let pa = dot(a, axis);
    let pb = dot(b, axis);
    let pc = dot(c, axis);
    let triangle_min = min(pa, min(pb, pc));
    let triangle_max = max(pa, max(pb, pc));
    let box_center = dot(shape.center, axis);
    let radius = articulated_box_radius(shape, axis);
    let positive_depth = triangle_max - (box_center - radius);
    let negative_depth = box_center + radius - triangle_min;
    let separation = -min(positive_depth, negative_depth);
    if (separation > best.separation) {
        let direction = axis * select(-1.0, 1.0, positive_depth <= negative_depth);
        return BoxAxisResult(separation, direction, 0u);
    }
    return best;
}

fn scene_mesh_box_geometry(sphere: Sphere, pose: Pose) -> BoxContactGeometry {
    let box_world = make_articulated_box(pose, sphere.center_radius.xyz, sphere.plane,
        vec3<f32>(sphere.center_radius.w, sphere.center_of_mass.w,
            sphere.other_center_of_mass.w));
    let mesh_translation = sphere.second_axis_end.xyz;
    let mesh_rotation = sphere.first_axis_end;
    let inverse_rotation = vec4<f32>(-mesh_rotation.xyz, mesh_rotation.w);
    let box_shape = ArticulatedBox(
        quat_rotate(inverse_rotation, box_world.center - mesh_translation),
        quat_rotate(inverse_rotation, box_world.axis_x),
        quat_rotate(inverse_rotation, box_world.axis_y),
        quat_rotate(inverse_rotation, box_world.axis_z),
        box_world.half_extents);
    let axes = array<vec3<f32>, 3>(box_shape.axis_x, box_shape.axis_y, box_shape.axis_z);
    let reach = abs(box_shape.axis_x) * box_shape.half_extents.x
        + abs(box_shape.axis_y) * box_shape.half_extents.y
        + abs(box_shape.axis_z) * box_shape.half_extents.z
        + vec3<f32>(1e-4);
    let vertex_start = u32(sphere.other_center_radius.x);
    let triangle_start = u32(sphere.other_center_radius.y);
    let node_start = u32(sphere.other_center_radius.z);
    let node_count = u32(sphere.other_center_radius.w);
    let absent = BoxContactGeometry(false, vec3<f32>(1.0, 0.0, 0.0),
        box_world.center, 1.0);
    var first = absent;
    var second = absent;
    var third = absent;
    var fourth = absent;
    let distinct_squared = max(1e-8,
        dot(box_shape.half_extents, box_shape.half_extents) * 1e-4);
    var cursor = 0u;
    while (cursor < node_count) {
        let node = spheres[node_start + cursor];
        if (any(box_shape.center + reach < node.center_radius.xyz)
            || any(box_shape.center - reach > node.plane.xyz)) {
            cursor = node.indices.y;
            continue;
        }
        cursor++;
        if (node.indices.x == 0xffffffffu) { continue; }
        let triangle = spheres[triangle_start + node.indices.x].indices;
        let a = spheres[vertex_start + triangle.x].center_radius.xyz;
        let b = spheres[vertex_start + triangle.y].center_radius.xyz;
        let c = spheres[vertex_start + triangle.z].center_radius.xyz;
        let edges = array<vec3<f32>, 3>(b - a, c - b, a - c);
        let face = cross(edges[0], c - a);
        if (dot(face, face) < 1e-20) { continue; }
        var best = BoxAxisResult(-1e30, vec3<f32>(0.0, 0.0, 1.0), 0u);
        best = mesh_box_axis(a, b, c, box_shape, face, best);
        for (var axis = 0u; axis < 3u; axis++) {
            best = mesh_box_axis(a, b, c, box_shape, axes[axis], best);
            for (var edge = 0u; edge < 3u; edge++) {
                best = mesh_box_axis(a, b, c, box_shape,
                    cross(axes[axis], edges[edge]), best);
            }
        }
        if (best.separation > 0.0) { continue; }
        var polygon: BoxFacePolygon;
        polygon.count = 3u;
        set_box_face_point(&polygon, 0u, a);
        set_box_face_point(&polygon, 1u, b);
        set_box_face_point(&polygon, 2u, c);
        for (var axis = 0u; axis < 3u; axis++) {
            polygon = clip_box_face(polygon, box_shape.center, axes[axis],
                box_shape.half_extents[axis]);
            polygon = clip_box_face(polygon, box_shape.center, -axes[axis],
                box_shape.half_extents[axis]);
        }
        for (var vertex = 0u; vertex < polygon.count; vertex++) {
            let local_point = polygon.points[vertex]
                + best.normal * best.separation * 0.5;
            let world_point = mesh_translation + quat_rotate(mesh_rotation, local_point);
            let candidate = BoxContactGeometry(true,
                -quat_rotate(mesh_rotation, best.normal), world_point, best.separation);
            let duplicate = (first.enabled
                    && dot(candidate.point - first.point, candidate.point - first.point) <= distinct_squared)
                || (second.enabled
                    && dot(candidate.point - second.point, candidate.point - second.point) <= distinct_squared)
                || (third.enabled
                    && dot(candidate.point - third.point, candidate.point - third.point) <= distinct_squared)
                || (fourth.enabled
                    && dot(candidate.point - fourth.point, candidate.point - fourth.point) <= distinct_squared);
            if (duplicate) { continue; }
            if (!first.enabled || candidate.distance < first.distance) {
                fourth = third;
                third = second;
                second = first;
                first = candidate;
            } else if (!second.enabled || candidate.distance < second.distance) {
                fourth = third;
                third = second;
                second = candidate;
            } else if (!third.enabled || candidate.distance < third.distance) {
                fourth = third;
                third = candidate;
            } else if (!fourth.enabled || candidate.distance < fourth.distance) {
                fourth = candidate;
            }
        }
    }
    let slot = u32(sphere.second_axis_end.w);
    if (slot == 0u) { return first; }
    if (slot == 1u) { return second; }
    if (slot == 2u) { return third; }
    return fourth;
}

fn scene_polyline_axial_geometry(sphere: Sphere, pose: Pose) -> BoxContactGeometry {
    let world_center = pose.position.xyz
        + quat_rotate(pose.orientation, sphere.center_radius.xyz);
    let world_axis = quat_rotate(pose.orientation,
        quat_rotate(sphere.plane, vec3<f32>(0.0, 0.0, 1.0)));
    let line_translation = sphere.second_axis_end.xyz;
    let line_rotation = sphere.first_axis_end;
    let inverse_rotation = vec4<f32>(-line_rotation.xyz, line_rotation.w);
    let local_center = quat_rotate(inverse_rotation, world_center - line_translation);
    let local_axis = quat_rotate(inverse_rotation, world_axis);
    let reach = vec3<f32>(sphere.center_radius.w + 1e-4)
        + abs(local_axis) * sphere.center_of_mass.w;
    let vertex_start = u32(sphere.other_center_radius.x);
    let segment_start = u32(sphere.other_center_radius.y);
    let node_start = u32(sphere.other_center_radius.z);
    let node_count = u32(sphere.other_center_radius.w);
    let absent = BoxContactGeometry(false, vec3<f32>(1.0, 0.0, 0.0),
        world_center, 1.0);
    var first = absent;
    var second = absent;
    var third = absent;
    var fourth = absent;
    let distinct_squared = max(1e-8,
        sphere.center_radius.w * sphere.center_radius.w * 1e-4);
    var proxy = sphere;
    proxy.center_radius = vec4<f32>(sphere.center_radius.xyz, 0.0);
    proxy.other_center_radius = vec4<f32>(sphere.center_radius.w,
        sphere.center_of_mass.w, 0.0, 0.0);
    proxy.second_axis_end = sphere.plane;
    proxy.material = vec4<f32>(sphere.material.xyz,
        select(29.0, 28.0, sphere.material.w == 75.0));
    var cursor = 0u;
    while (cursor < node_count) {
        let node = spheres[node_start + cursor];
        if (any(local_center + reach < node.center_radius.xyz)
            || any(local_center - reach > node.plane.xyz)) {
            cursor = node.indices.y;
            continue;
        }
        cursor++;
        if (node.indices.x == 0xffffffffu) { continue; }
        let segment = spheres[segment_start + node.indices.x].indices;
        let a = spheres[vertex_start + segment.x].center_radius.xyz;
        let b = spheres[vertex_start + segment.y].center_radius.xyz;
        let start = line_translation + quat_rotate(line_rotation, a);
        let end = line_translation + quat_rotate(line_rotation, b);
        for (var feature = 0u; feature < select(1u, 4u, sphere.material.w == 75.0); feature++) {
            var candidate = static_axial_capsule_geometry(proxy, pose, start, end);
            if (feature != 0u) {
                // A line can touch an entire cylinder generator; keep both ends of that contact patch.
                var reference = world_center;
                // Sample just inside a cylinder cap so roundoff cannot choose a purely axial corner normal.
                let half_height = sphere.center_of_mass.w
                    * select(1.0, 0.99999, sphere.material.w == 75.0);
                if (feature == 1u) { reference -= world_axis * half_height; }
                if (feature == 2u) { reference += world_axis * half_height; }
                let direction = end - start;
                let fraction = clamp(dot(reference - start, direction) / dot(direction, direction), 0.0, 1.0);
                candidate = static_axial_sphere_geometry(proxy, pose, start + fraction * direction);
            }
            if (candidate.distance > 1e-6) { continue; }
            if (sphere.material.w == 75.0 || sphere.material.w == 76.0) {
                // At an axial rim the closest-feature normal is not unique. An interior line
                // contact must use the normal perpendicular to the line, including at that rim.
                let direction = end - start;
                let squared = dot(direction, direction);
                let line_point = candidate.point + candidate.normal * (0.5 * candidate.distance);
                let fraction = dot(line_point - start, direction) / squared;
                let transverse = candidate.normal - direction * (dot(candidate.normal, direction) / squared);
                if (fraction > 1e-6 && fraction < 1.0 - 1e-6 && dot(transverse, transverse) > 1e-12) {
                    candidate.normal = normalize(transverse);
                    candidate.point = line_point - candidate.normal * (0.5 * candidate.distance);
                    if (sphere.material.w == 75.0) {
                        let shape = AxialShape(world_center, normalize(world_axis),
                            sphere.center_of_mass.w, sphere.center_radius.w, 30.0);
                        var support = axial_shape_support(shape, candidate.normal);
                        if (abs(dot(shape.axis, candidate.normal)) < 1e-6) {
                            // Preserve both witnesses when the supporting generator is parallel.
                            let height = clamp(dot(line_point - shape.center, shape.axis),
                                -shape.half_height, shape.half_height);
                            support += shape.axis * (height - dot(support - shape.center, shape.axis));
                        }
                        let support_fraction = clamp(dot(support - start, direction) / squared, 0.0, 1.0);
                        let witness = start + direction * support_fraction;
                        candidate.distance = dot(witness - support, candidate.normal);
                        candidate.point = 0.5 * (witness + support);
                    }
                }
            }
            let duplicate = (first.enabled
                    && dot(candidate.point - first.point, candidate.point - first.point) <= distinct_squared)
                || (second.enabled
                    && dot(candidate.point - second.point, candidate.point - second.point) <= distinct_squared)
                || (third.enabled
                    && dot(candidate.point - third.point, candidate.point - third.point) <= distinct_squared)
                || (fourth.enabled
                    && dot(candidate.point - fourth.point, candidate.point - fourth.point) <= distinct_squared);
            if (duplicate) { continue; }
            if (!first.enabled || candidate.distance < first.distance) {
                fourth = third;
                third = second;
                second = first;
                first = candidate;
            } else if (!second.enabled || candidate.distance < second.distance) {
                fourth = third;
                third = second;
                second = candidate;
            } else if (!third.enabled || candidate.distance < third.distance) {
                fourth = third;
                third = candidate;
            } else if (!fourth.enabled || candidate.distance < fourth.distance) {
                fourth = candidate;
            }
        }
    }
    let slot = u32(sphere.second_axis_end.w);
    if (slot == 0u) { return first; }
    if (slot == 1u) { return second; }
    if (slot == 2u) { return third; }
    return fourth;
}

fn scene_polyline_convex_manifold(sphere: Sphere, pose: Pose) -> SceneMeshConvexManifold {
    let header = u32(sphere.center_radius.w);
    let counts = spheres[header].indices;
    let hull = ContactHull(pose, sphere.center_radius.xyz, sphere.first_axis_end,
        header, counts.x, counts.y, counts.z);
    let world_center = contact_hull_center(hull);
    let line_translation = sphere.plane.xyz;
    let line_rotation = sphere.second_axis_end;
    let inverse_rotation = vec4<f32>(-line_rotation.xyz, line_rotation.w);
    var lower = vec3<f32>(1e30);
    var upper = vec3<f32>(-1e30);
    for (var i = 0u; i < hull.vertex_count; i++) {
        let local = quat_rotate(inverse_rotation,
            contact_hull_vertex(hull, i) - line_translation);
        lower = min(lower, local);
        upper = max(upper, local);
    }
    lower -= vec3<f32>(1e-4);
    upper += vec3<f32>(1e-4);
    let vertex_start = u32(sphere.other_center_radius.x);
    let segment_start = u32(sphere.other_center_radius.y);
    let node_start = u32(sphere.other_center_radius.z);
    let node_count = u32(sphere.other_center_radius.w);
    let absent = BoxContactGeometry(false, vec3<f32>(1.0, 0.0, 0.0),
        world_center, 1.0);
    var first = absent;
    var second = absent;
    var third = absent;
    var fourth = absent;
    let distinct_squared = max(1e-8, dot(upper - lower, upper - lower) * 1e-4);
    var proxy = sphere;
    proxy.center_radius.w = 0.0;
    proxy.plane.w = f32(header + 1u);
    proxy.other_center_radius = vec4<f32>(f32(counts.x),
        f32(header + 1u + counts.x), f32(counts.y), f32(counts.z));
    proxy.material.w = 46.0;
    let identity = Pose(vec4<f32>(0.0), vec4<f32>(0.0, 0.0, 0.0, 1.0));
    var cursor = 0u;
    while (cursor < node_count) {
        let node = spheres[node_start + cursor];
        if (any(upper < node.center_radius.xyz)
            || any(lower > node.plane.xyz)) {
            cursor = node.indices.y;
            continue;
        }
        cursor++;
        if (node.indices.x == 0xffffffffu) { continue; }
        let segment = spheres[segment_start + node.indices.x].indices;
        let a = spheres[vertex_start + segment.x].center_radius.xyz;
        let b = spheres[vertex_start + segment.y].center_radius.xyz;
        let start = line_translation + quat_rotate(line_rotation, a);
        let end = line_translation + quat_rotate(line_rotation, b);
        proxy.plane = vec4<f32>(start, proxy.plane.w);
        proxy.second_axis_end = vec4<f32>(end, f32(header + 1u + counts.x + counts.y));
        for (var endpoint = 0u; endpoint < 2u; endpoint++) {
            // Reuse the clipped capsule manifold to retain both ends of a line/face contact.
            proxy.material.w = select(46.0, 81.0, endpoint == 1u);
            let candidate = convex_rounded_geometry(proxy, pose, identity);
            if (!candidate.enabled || candidate.distance > 1e-6) { continue; }
            let duplicate = (first.enabled
                    && dot(candidate.point - first.point, candidate.point - first.point) <= distinct_squared)
                || (second.enabled
                    && dot(candidate.point - second.point, candidate.point - second.point) <= distinct_squared)
                || (third.enabled
                    && dot(candidate.point - third.point, candidate.point - third.point) <= distinct_squared)
                || (fourth.enabled
                    && dot(candidate.point - fourth.point, candidate.point - fourth.point) <= distinct_squared);
            if (duplicate) { continue; }
            if (!first.enabled || candidate.distance < first.distance) {
                fourth = third;
                third = second;
                second = first;
                first = candidate;
            } else if (!second.enabled || candidate.distance < second.distance) {
                fourth = third;
                third = second;
                second = candidate;
            } else if (!third.enabled || candidate.distance < third.distance) {
                fourth = third;
                third = candidate;
            } else if (!fourth.enabled || candidate.distance < fourth.distance) {
                fourth = candidate;
            }
        }
    }
    return SceneMeshConvexManifold(first, second, third, fourth);
}

fn mesh_axial_triangle_support(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>,
    face: vec3<f32>, shape: AxialShape, direction: vec3<f32>) -> vec3<f32> {
    var vertex = a;
    if (dot(b, direction) > dot(vertex, direction)) { vertex = b; }
    if (dot(c, direction) > dot(vertex, direction)) { vertex = c; }
    let thickness = select(-1e-5, 1e-5, dot(face, direction) >= 0.0);
    return vertex + face * thickness - axial_shape_support(shape, -direction);
}

fn mesh_axial_triangle_intersects(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>,
    face: vec3<f32>, shape: AxialShape) -> bool {
    var direction = shape.center - (a + b + c) / 3.0;
    if (dot(direction, direction) < 1e-16) { direction = face; }
    let first = mesh_axial_triangle_support(a, b, c, face, shape, direction);
    var simplex = AxialSimplex(first, vec3<f32>(0.0), vec3<f32>(0.0),
        vec3<f32>(0.0), 1u, -first, false);
    for (var iteration = 0u; iteration < 32u; iteration++) {
        if (dot(simplex.direction, simplex.direction) < 1e-16) { return true; }
        let point = mesh_axial_triangle_support(a, b, c, face, shape,
            simplex.direction);
        let projection = dot(point, simplex.direction);
        if (projection < -1e-6) { return false; }
        if (dot(point - simplex.a, point - simplex.a) < 1e-14 ||
            (simplex.count > 1u && dot(point - simplex.b, point - simplex.b) < 1e-14) ||
            (simplex.count > 2u && dot(point - simplex.c, point - simplex.c) < 1e-14)) {
            return abs(projection) < 1e-6;
        }
        simplex = expand_axial_simplex(simplex, point);
        if (simplex.inside) { return true; }
    }
    return false;
}

fn mesh_axial_triangle_axis(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>,
    shape: AxialShape, candidate: vec3<f32>, best: BoxAxisResult) -> BoxAxisResult {
    let length_squared = dot(candidate, candidate);
    if (length_squared < 1e-12 || best.separation > 0.0) { return best; }
    let axis = candidate * inverseSqrt(length_squared);
    let triangle_min = min(dot(a, axis), min(dot(b, axis), dot(c, axis)));
    let triangle_max = max(dot(a, axis), max(dot(b, axis), dot(c, axis)));
    let shape_min = dot(axial_shape_support(shape, -axis), axis);
    let shape_max = dot(axial_shape_support(shape, axis), axis);
    let forward = triangle_max - shape_min;
    let backward = shape_max - triangle_min;
    let separation = -min(forward, backward);
    if (separation > best.separation) {
        return BoxAxisResult(separation,
            axis * select(-1.0, 1.0, forward < backward), 0u);
    }
    return best;
}

fn scene_mesh_axial_geometry(sphere: Sphere, pose: Pose) -> BoxContactGeometry {
    let world_center = pose.position.xyz
        + quat_rotate(pose.orientation, sphere.center_radius.xyz);
    let world_axis = quat_rotate(pose.orientation,
        quat_rotate(sphere.plane, vec3<f32>(0.0, 0.0, 1.0)));
    let mesh_translation = sphere.second_axis_end.xyz;
    let mesh_rotation = sphere.first_axis_end;
    let inverse_rotation = vec4<f32>(-mesh_rotation.xyz, mesh_rotation.w);
    let shape = AxialShape(
        quat_rotate(inverse_rotation, world_center - mesh_translation),
        normalize(quat_rotate(inverse_rotation, world_axis)),
        sphere.center_of_mass.w, sphere.center_radius.w,
        select(31.0, 30.0, sphere.material.w == 55.0));
    let reach = vec3<f32>(shape.radius + 1e-4)
        + abs(shape.axis) * shape.half_height;
    let vertex_start = u32(sphere.other_center_radius.x);
    let triangle_start = u32(sphere.other_center_radius.y);
    let node_start = u32(sphere.other_center_radius.z);
    let node_count = u32(sphere.other_center_radius.w);
    var best_contact = BoxContactGeometry(false, vec3<f32>(1.0, 0.0, 0.0),
        world_center, 1.0);
    var cursor = 0u;
    while (cursor < node_count) {
        let node = spheres[node_start + cursor];
        if (any(shape.center + reach < node.center_radius.xyz)
            || any(shape.center - reach > node.plane.xyz)) {
            cursor = node.indices.y;
            continue;
        }
        cursor++;
        if (node.indices.x == 0xffffffffu) { continue; }
        let triangle = spheres[triangle_start + node.indices.x].indices;
        let a = spheres[vertex_start + triangle.x].center_radius.xyz;
        let b = spheres[vertex_start + triangle.y].center_radius.xyz;
        let c = spheres[vertex_start + triangle.z].center_radius.xyz;
        let face_unnormalized = cross(b - a, c - a);
        if (dot(face_unnormalized, face_unnormalized) < 1e-20) { continue; }
        let face = normalize(face_unnormalized);
        if (!mesh_axial_triangle_intersects(a, b, c, face, shape)) { continue; }
        let edges = array<vec3<f32>, 3>(b - a, c - b, a - c);
        var best = BoxAxisResult(-1e30, face, 0u);
        best = mesh_axial_triangle_axis(a, b, c, shape, face, best);
        if (abs(dot(shape.axis, face)) < 0.99) {
            best = mesh_axial_triangle_axis(a, b, c, shape, shape.axis, best);
            best = mesh_axial_triangle_axis(a, b, c, shape,
                shape.center - closest_mesh_triangle(shape.center, a, b, c), best);
            for (var edge = 0u; edge < 3u; edge++) {
                best = mesh_axial_triangle_axis(a, b, c, shape,
                    cross(edges[edge], shape.axis), best);
            }
        }
        if (best.separation > 1e-5) { continue; }
        var shape_point = axial_shape_support(shape, -best.normal);
        if (shape.kind == 30.0 && abs(dot(shape.axis, best.normal)) < 1e-6) {
            // A side contact supports a generator, not an arbitrary cap endpoint.
            let radial = -best.normal + shape.axis * dot(shape.axis, best.normal);
            let center = shape.center + normalize(radial) * shape.radius;
            let start = center - shape.axis * shape.half_height;
            let end = center + shape.axis * shape.half_height;
            var lower = 0.0;
            var upper = 1.0;
            for (var edge = 0u; edge < 3u; edge++) {
                let vertex = select(select(a, b, edge == 1u), c, edge == 2u);
                let side_start = dot(cross(edges[edge], start - vertex), face);
                let side_end = dot(cross(edges[edge], end - vertex), face);
                if (side_start < -1e-6 && side_end < -1e-6) {
                    upper = -1.0;
                    break;
                }
                if ((side_start < -1e-6) != (side_end < -1e-6)) {
                    let fraction = side_start / (side_start - side_end);
                    if (side_start < -1e-6) { lower = max(lower, fraction); }
                    else { upper = min(upper, fraction); }
                }
            }
            if (lower > upper) { continue; }
            shape_point = mix(start, end, select(lower, upper, sphere.second_axis_end.w == 1.0));
        } else if (sphere.second_axis_end.w == 1.0) {
            continue;
        }
        // Match the mesh witness to the shape's contact feature, preserving its tangential lever arm.
        let mesh_point = closest_mesh_triangle(shape_point, a, b, c);
        let local_point = 0.5 * (mesh_point + shape_point);
        let candidate = BoxContactGeometry(true,
            -quat_rotate(mesh_rotation, best.normal),
            mesh_translation + quat_rotate(mesh_rotation, local_point),
            min(best.separation, 0.0));
        if (!best_contact.enabled || candidate.distance < best_contact.distance) {
            best_contact = candidate;
        }
    }
    return best_contact;
}

fn convex_world_vertex(sphere: Sphere, pose: Pose, vertex_index: u32) -> vec3<f32> {
    let local = spheres[u32(sphere.plane.w) + vertex_index].center_radius.xyz;
    return pose.position.xyz + quat_rotate(pose.orientation,
        sphere.center_radius.xyz + quat_rotate(sphere.first_axis_end, local));
}

fn convex_world_normal(sphere: Sphere, pose: Pose, normal_index: u32) -> vec3<f32> {
    let local = spheres[u32(sphere.other_center_radius.y) + normal_index].center_radius.xyz;
    return quat_rotate(pose.orientation,
        quat_rotate(sphere.first_axis_end, local));
}

fn convex_rounded_support(sphere: Sphere, pose: Pose,
    rounded_a: vec3<f32>, rounded_b: vec3<f32>, direction: vec3<f32>) -> vec3<f32> {
    var best = convex_world_vertex(sphere, pose, 0u);
    var projection = dot(best, direction);
    for (var i = 1u; i < u32(sphere.other_center_radius.x); i++) {
        let vertex = convex_world_vertex(sphere, pose, i);
        let candidate = dot(vertex, direction);
        if (candidate > projection) {
            best = vertex;
            projection = candidate;
        }
    }
    let squared = dot(direction, direction);
    let unit = select(vec3<f32>(1.0, 0.0, 0.0),
        direction * inverseSqrt(max(squared, 1e-20)), squared > 1e-20);
    let rounded_point = select(rounded_b, rounded_a,
        dot(rounded_a, direction) < dot(rounded_b, direction));
    return best - (rounded_point - unit * sphere.center_radius.w);
}

fn convex_rounded_intersects(sphere: Sphere, pose: Pose,
    hull_center: vec3<f32>, rounded_a: vec3<f32>, rounded_b: vec3<f32>) -> bool {
    var direction = 0.5 * (rounded_a + rounded_b) - hull_center;
    if (dot(direction, direction) < 1e-16) { direction = vec3<f32>(1.0, 0.0, 0.0); }
    let support = convex_rounded_support(sphere, pose, rounded_a, rounded_b, direction);
    var simplex = AxialSimplex(support, vec3<f32>(0.0), vec3<f32>(0.0),
        vec3<f32>(0.0), 1u, -support, false);
    for (var iteration = 0u; iteration < 32u; iteration++) {
        if (dot(simplex.direction, simplex.direction) < 1e-16) { return true; }
        let point = convex_rounded_support(sphere, pose,
            rounded_a, rounded_b, simplex.direction);
        let projection = dot(point, simplex.direction);
        if (projection < -1e-6) { return false; }
        if (dot(point - simplex.a, point - simplex.a) < 1e-14 ||
            (simplex.count > 1u && dot(point - simplex.b, point - simplex.b) < 1e-14) ||
            (simplex.count > 2u && dot(point - simplex.c, point - simplex.c) < 1e-14)) {
            return abs(projection) < 1e-6;
        }
        simplex = expand_axial_simplex(simplex, point);
        if (simplex.inside) { return true; }
    }
    return false;
}

fn consider_convex_rounded_axis(sphere: Sphere, pose: Pose,
    rounded_a: vec3<f32>, rounded_b: vec3<f32>, candidate: vec3<f32>,
    best: BoxAxisResult) -> BoxAxisResult {
    let squared = dot(candidate, candidate);
    if (squared < 1e-12) { return best; }
    let axis = candidate * inverseSqrt(squared);
    var hull_min = 1e30;
    var hull_max = -1e30;
    for (var i = 0u; i < u32(sphere.other_center_radius.x); i++) {
        let projection = dot(convex_world_vertex(sphere, pose, i), axis);
        hull_min = min(hull_min, projection);
        hull_max = max(hull_max, projection);
    }
    let first_projection = dot(rounded_a, axis);
    let second_projection = dot(rounded_b, axis);
    let rounded_min = min(first_projection, second_projection) - sphere.center_radius.w;
    let rounded_max = max(first_projection, second_projection) + sphere.center_radius.w;
    let forward = hull_max - rounded_min;
    let backward = rounded_max - hull_min;
    let separation = -min(forward, backward);
    if (separation > best.separation) {
        return BoxAxisResult(separation,
            axis * select(-1.0, 1.0, forward < backward), 0u);
    }
    return best;
}

// Project onto the support plane, then clamp to the hull support feature.
// Opposing face normals are retained by the hull builder. Interior projections
// need no edge search; outside projections use support-vertex segments.
fn convex_support_near(sphere: Sphere, pose: Pose, normal: vec3<f32>,
    projection: f32, reference: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let point = reference + normal * (projection - dot(reference, normal));
    var inside = true;
    for (var face = 0u; face < u32(sphere.other_center_radius.z); face++) {
        let axis = convex_world_normal(sphere, pose, face);
        var bound = -1e30;
        for (var vertex = 0u; vertex < u32(sphere.other_center_radius.x); vertex++) {
            bound = max(bound, dot(convex_world_vertex(sphere, pose, vertex), axis));
        }
        let tolerance = select(1e-6, 0.0, sphere.center_radius.w == 0.0);
        if dot(point, axis) > bound + tolerance { inside = false; }
    }
    if inside { return point; }
    var nearest = fallback;
    var distance = dot(point - nearest, point - nearest);
    for (var first = 0u; first < u32(sphere.other_center_radius.x); first++) {
        let a = convex_world_vertex(sphere, pose, first);
        if abs(dot(a, normal) - projection) > 1e-6 { continue; }
        for (var second = first; second < u32(sphere.other_center_radius.x); second++) {
            let b = convex_world_vertex(sphere, pose, second);
            if abs(dot(b, normal) - projection) > 1e-6 { continue; }
            let edge = b - a;
            let t = clamp(dot(point - a, edge) / max(dot(edge, edge), 1e-20), 0.0, 1.0);
            let candidate = a + edge * t;
            let squared = dot(point - candidate, point - candidate);
            if squared < distance { nearest = candidate; distance = squared; }
        }
    }
    return nearest;
}

fn convex_rounded_geometry(sphere: Sphere, pose: Pose,
    second_pose: Pose) -> BoxContactGeometry {
    let rounded_a = second_pose.position.xyz
        + quat_rotate(second_pose.orientation, sphere.plane.xyz);
    let rounded_b = second_pose.position.xyz
        + quat_rotate(second_pose.orientation, sphere.second_axis_end.xyz);
    let rounded_center = 0.5 * (rounded_a + rounded_b);
    let segment = rounded_b - rounded_a;
    let segment_squared = dot(segment, segment);
    var hull_center = vec3<f32>(0.0);
    var hull_min = vec3<f32>(1e30);
    var hull_max = vec3<f32>(-1e30);
    for (var i = 0u; i < u32(sphere.other_center_radius.x); i++) {
        let vertex = convex_world_vertex(sphere, pose, i);
        hull_center += vertex;
        hull_min = min(hull_min, vertex);
        hull_max = max(hull_max, vertex);
    }
    hull_center /= sphere.other_center_radius.x;
    let margin = vec3<f32>(sphere.center_radius.w + 1e-5);
    if (any(hull_max + margin < min(rounded_a, rounded_b))
        || any(max(rounded_a, rounded_b) + margin < hull_min)) {
        return BoxContactGeometry(false, vec3<f32>(1.0, 0.0, 0.0),
            0.5 * (hull_center + rounded_center), 1.0);
    }
    var best = BoxAxisResult(-1e30, vec3<f32>(1.0, 0.0, 0.0), 0u);
    for (var i = 0u; i < u32(sphere.other_center_radius.z); i++) {
        best = consider_convex_rounded_axis(sphere, pose, rounded_a, rounded_b,
            convex_world_normal(sphere, pose, i), best);
    }
    for (var i = 0u; i < u32(sphere.other_center_radius.x); i++) {
        let vertex = convex_world_vertex(sphere, pose, i);
        let t = clamp(dot(vertex - rounded_a, segment) / max(segment_squared, 1e-20),
            0.0, 1.0);
        best = consider_convex_rounded_axis(sphere, pose, rounded_a, rounded_b,
            rounded_a + segment * t - vertex, best);
        if (sphere.material.w == 43.0 || (sphere.material.w == 46.0 || sphere.material.w == 81.0 || sphere.material.w == 82.0 || sphere.material.w == 83.0)
            || sphere.material.w == 49.0) {
            best = consider_convex_rounded_axis(sphere, pose, rounded_a, rounded_b,
                rounded_a - vertex, best);
            best = consider_convex_rounded_axis(sphere, pose, rounded_a, rounded_b,
                rounded_b - vertex, best);
        }
    }
    best = consider_convex_rounded_axis(sphere, pose, rounded_a, rounded_b,
        rounded_center - hull_center, best);
    best = consider_convex_rounded_axis(sphere, pose, rounded_a, rounded_b,
        segment, best);
    for (var i = 0u; i < u32(sphere.other_center_radius.w); i++) {
        let local_edge = spheres[u32(sphere.second_axis_end.w) + i].center_radius.xyz;
        let world_edge = quat_rotate(pose.orientation,
            quat_rotate(sphere.first_axis_end, local_edge));
        best = consider_convex_rounded_axis(sphere, pose, rounded_a, rounded_b,
            cross(world_edge, segment), best);
    }
    let support = convex_world_vertex(sphere, pose, 0u);
    var hull_witness = support;
    var projection = dot(support, best.normal);
    for (var i = 1u; i < u32(sphere.other_center_radius.x); i++) {
        let vertex = convex_world_vertex(sphere, pose, i);
        let candidate = dot(vertex, best.normal);
        if (candidate >= projection) {
            hull_witness = vertex;
            projection = candidate;
        }
    }
    // A capsule parallel to a supporting hull face produces a clipped line manifold.
    // Modes 81 through 83 reserve the second endpoint; all other feature contacts use one row.
    if (sphere.material.w == 43.0 || sphere.material.w == 46.0 || sphere.material.w == 49.0 || sphere.material.w == 81.0 || sphere.material.w == 82.0 || sphere.material.w == 83.0) {
        let absent = BoxContactGeometry(false, best.normal, rounded_center, best.separation);
        var face_normal = false;
        var support_normal = best.normal;
        var alignment = 0.9995;
        for (var i = 0u; i < u32(sphere.other_center_radius.z); i++) {
            let axis = convex_world_normal(sphere, pose, i);
            let score = dot(axis, best.normal);
            if (score >= alignment) {
                face_normal = true;
                alignment = score;
                // A zero-radius line retains its SAT normal, perpendicular to its interior.
                support_normal = select(best.normal, axis, sphere.center_radius.w > 0.0);
            }
        }
        if (face_normal && segment_squared > 1e-12
            && abs(dot(segment, support_normal)) <= 0.05 * sqrt(segment_squared)) {
            // Capsules use the actual hull face; lines keep their SAT support plane.
            var support_plane = -1e30;
            for (var vertex = 0u; vertex < u32(sphere.other_center_radius.x); vertex++) {
                support_plane = max(support_plane,
                    dot(convex_world_vertex(sphere, pose, vertex), support_normal));
            }
            let projected_a = rounded_a + support_normal * (support_plane - dot(rounded_a, support_normal));
            let face_segment = segment - support_normal * dot(segment, support_normal);
            var low = 0.0;
            var high = 1.0;
            var clipped = true;
            let clip_tolerance = select(0.0, 1e-6, sphere.center_radius.w == 0.0);
            // Hull half spaces clip against the actual face perimeter, including
            // capsules whose original endpoints extend beyond the hull face.
            for (var i = 0u; i < u32(sphere.other_center_radius.z); i++) {
                let axis = convex_world_normal(sphere, pose, i);
                var height = -1e30;
                for (var vertex = 0u; vertex < u32(sphere.other_center_radius.x); vertex++) {
                    height = max(height, dot(convex_world_vertex(sphere, pose, vertex), axis));
                }
                let delta = dot(projected_a, axis) - height;
                let slope = dot(face_segment, axis);
                if (abs(slope) < 1e-8) {
                    if (delta > 1e-6) { clipped = false; }
                } else if (slope > 0.0) {
                    high = min(high, (clip_tolerance - delta) / slope);
                } else {
                    low = max(low, (clip_tolerance - delta) / slope);
                }
            }
            let span = face_segment * max(0.0, high - low);
            let distinct_squared = max(1e-10, dot(hull_max - hull_min, hull_max - hull_min) * 1e-8);
            if (clipped && high >= low) {
                if (sphere.material.w >= 81.0 && dot(span, span) <= distinct_squared) { return absent; }
                let fraction = select(low, high, sphere.material.w >= 81.0);
                let axis_point = rounded_a + segment * fraction;
                var local_distance = dot(axis_point, support_normal) - sphere.center_radius.w - support_plane;
                var hull_point = axis_point + support_normal * (support_plane - dot(axis_point, support_normal));
                var rounded_point = axis_point - support_normal * sphere.center_radius.w;
                if (sphere.center_radius.w == 0.0) {
                    // Acceptance tolerance must not enlarge the physical support feature.
                    hull_point = convex_support_near(sphere, pose, support_normal,
                        support_plane, hull_point, hull_witness);
                    let matched_fraction = clamp(dot(hull_point - rounded_a, segment) / segment_squared, 0.0, 1.0);
                    rounded_point = rounded_a + segment * matched_fraction;
                    local_distance = dot(rounded_point - hull_point, support_normal);
                }
                let point = (hull_point + rounded_point) * 0.5;
                let enabled = convex_rounded_intersects(sphere, pose, hull_center, rounded_a, rounded_b);
                return BoxContactGeometry(enabled, support_normal, point, local_distance);
            }
        }
        if (sphere.material.w >= 81.0) { return absent; }
    }
    var rounded_witness_axis = select(rounded_b, rounded_a,
        dot(rounded_a, best.normal) < dot(rounded_b, best.normal));
    let line_interior_support = sphere.center_radius.w == 0.0 && segment_squared > 1e-12
        && abs(dot(segment, best.normal)) <= 1e-6 * sqrt(segment_squared);
    if (line_interior_support) {
        let fraction = clamp(dot(hull_witness - rounded_a, segment) / segment_squared, 0.0, 1.0);
        rounded_witness_axis = rounded_a + segment * fraction;
    }
    var rounded_witness = rounded_witness_axis
        - best.normal * sphere.center_radius.w;
    // Preserve external rounded witness tangent coordinates on a supporting hull face.
    // Center projection shifts off-axis sphere contacts and creates artificial torque.
    // Nonparallel self-contact capsules retain the CPU rounded-core support witnesses.
    if (sphere.material.w != 43.0) {
        hull_witness = convex_support_near(sphere, pose, best.normal,
            projection, rounded_witness, hull_witness);
    }
    var distance = best.separation;
    if (line_interior_support) {
        // Degenerate clipping still needs paired witnesses, not a distant line endpoint.
        let fraction = clamp(dot(hull_witness - rounded_a, segment) / segment_squared, 0.0, 1.0);
        rounded_witness = rounded_a + segment * fraction;
        distance = dot(rounded_witness - hull_witness, best.normal);
    }
    let point = 0.5 * (hull_witness + rounded_witness);
    let enabled = convex_rounded_intersects(sphere, pose,
        hull_center, rounded_a, rounded_b)
        && best.separation <= 1e-6;
    return BoxContactGeometry(enabled, best.normal, point, distance);
}

struct ContactHull {
    pose: Pose,
    translation: vec3<f32>,
    rotation: vec4<f32>,
    header: u32,
    vertex_count: u32,
    normal_count: u32,
    edge_count: u32,
};

struct SceneMeshConvexManifold {
    first: BoxContactGeometry,
    second: BoxContactGeometry,
    third: BoxContactGeometry,
    fourth: BoxContactGeometry,
};

struct HullFacePolygon {
    points: array<vec2<f32>, 8>,
    count: u32,
    overflow: bool,
};

fn set_contact_hull_face_point(face: ptr<function, HullFacePolygon>,
    index: u32, point: vec2<f32>) {
    switch index {
        case 0u: { (*face).points[0u] = point; }
        case 1u: { (*face).points[1u] = point; }
        case 2u: { (*face).points[2u] = point; }
        case 3u: { (*face).points[3u] = point; }
        case 4u: { (*face).points[4u] = point; }
        case 5u: { (*face).points[5u] = point; }
        case 6u: { (*face).points[6u] = point; }
        case 7u: { (*face).points[7u] = point; }
        default: {}
    }
}

fn hull_face_cross(first: vec2<f32>, second: vec2<f32>) -> f32 {
    return first.x * second.y - first.y * second.x;
}

fn contact_hull_2d(input: HullFacePolygon, tolerance: f32) -> HullFacePolygon {
    var sorted = input;
    for (var i = 0u; i < sorted.count; i++) {
        for (var j = i + 1u; j < sorted.count; j++) {
            let first = sorted.points[i];
            let second = sorted.points[j];
            if (second.x < first.x || (second.x == first.x && second.y < first.y)) {
                set_contact_hull_face_point(&sorted, i, second);
                set_contact_hull_face_point(&sorted, j, first);
            }
        }
    }
    var unique: HullFacePolygon;
    for (var i = 0u; i < sorted.count; i++) {
        let point = sorted.points[i];
        if (unique.count > 0u) {
            let offset = point - unique.points[unique.count - 1u];
            if (dot(offset, offset) <= tolerance * tolerance) { continue; }
        }
        set_contact_hull_face_point(&unique, unique.count, point);
        unique.count++;
    }
    if (unique.count < 3u) { return unique; }
    var hull: HullFacePolygon;
    for (var i = 0u; i < unique.count; i++) {
        let point = unique.points[i];
        while (hull.count >= 2u) {
            let last = hull.count;
            if (hull_face_cross(hull.points[last - 1u] - hull.points[last - 2u],
                point - hull.points[last - 1u]) > 1e-7) { break; }
            hull.count--;
        }
        set_contact_hull_face_point(&hull, hull.count, point);
        hull.count++;
    }
    let lower_count = hull.count + 1u;
    for (var i = unique.count - 2u; i < unique.count; i--) {
        let point = unique.points[i];
        while (hull.count >= lower_count) {
            let last = hull.count;
            if (hull_face_cross(hull.points[last - 1u] - hull.points[last - 2u],
                point - hull.points[last - 1u]) > 1e-7) { break; }
            hull.count--;
        }
        if (hull.count >= 8u) {
            hull.overflow = true;
            return hull;
        }
        set_contact_hull_face_point(&hull, hull.count, point);
        hull.count++;
        if (i == 0u) { break; }
    }
    hull.count--;
    return hull;
}

fn contact_hull_face(hull: ContactHull, normal: vec3<f32>,
    support_height: f32, tangent: vec3<f32>, bitangent: vec3<f32>,
    tolerance: f32) -> HullFacePolygon {
    var face: HullFacePolygon;
    for (var i = 0u; i < hull.vertex_count; i++) {
        let vertex = contact_hull_vertex(hull, i);
        if (abs(dot(vertex, normal) - support_height) <= tolerance) {
            if (face.count >= 6u) {
                face.overflow = true;
                return face;
            }
            set_contact_hull_face_point(&face, face.count,
                vec2<f32>(dot(vertex, tangent), dot(vertex, bitangent)));
            face.count++;
        }
    }
    return contact_hull_2d(face, tolerance);
}

fn clip_contact_hull_face(input: HullFacePolygon,
    start: vec2<f32>, end: vec2<f32>, tolerance: f32) -> HullFacePolygon {
    var output: HullFacePolygon;
    if (input.count == 0u) { return output; }
    var previous = input.points[input.count - 1u];
    var old_side = hull_face_cross(end - start, previous - start);
    for (var i = 0u; i < input.count; i++) {
        let current = input.points[i];
        let new_side = hull_face_cross(end - start, current - start);
        if ((old_side >= -tolerance) != (new_side >= -tolerance)) {
            let denominator = old_side - new_side;
            if (abs(denominator) > 1e-12) {
                if (output.count >= 8u) {
                    output.overflow = true;
                    return output;
                }
                set_contact_hull_face_point(&output, output.count,
                    previous + (current - previous) * (old_side / denominator));
                output.count++;
            }
        }
        if (new_side >= -tolerance) {
            if (output.count >= 8u) {
                output.overflow = true;
                return output;
            }
            set_contact_hull_face_point(&output, output.count, current);
            output.count++;
        }
        previous = current;
        old_side = new_side;
    }
    return output;
}

fn contact_hull_vertex(hull: ContactHull, index: u32) -> vec3<f32> {
    let local = spheres[hull.header + 1u + index].center_radius.xyz;
    return hull.pose.position.xyz + quat_rotate(hull.pose.orientation,
        hull.translation + quat_rotate(hull.rotation, local));
}

fn contact_hull_direction(hull: ContactHull, index: u32, edge: bool) -> vec3<f32> {
    let offset = hull.header + 1u + hull.vertex_count + index
        + select(0u, hull.normal_count, edge);
    let local = spheres[offset].center_radius.xyz;
    return quat_rotate(hull.pose.orientation, quat_rotate(hull.rotation, local));
}

fn contact_hull_center(hull: ContactHull) -> vec3<f32> {
    var center = vec3<f32>(0.0);
    for (var i = 0u; i < hull.vertex_count; i++) {
        center += contact_hull_vertex(hull, i);
    }
    return center / f32(hull.vertex_count);
}

fn contact_hull_support(hull: ContactHull, direction: vec3<f32>) -> vec3<f32> {
    var best = contact_hull_vertex(hull, 0u);
    var projection = dot(best, direction);
    for (var i = 1u; i < hull.vertex_count; i++) {
        let candidate = contact_hull_vertex(hull, i);
        let value = dot(candidate, direction);
        if (value >= projection) {
            best = candidate;
            projection = value;
        }
    }
    return best;
}

fn scene_mesh_convex_axis(hull: ContactHull, a: vec3<f32>, b: vec3<f32>,
    c: vec3<f32>, candidate: vec3<f32>, best: BoxAxisResult) -> BoxAxisResult {
    let squared = dot(candidate, candidate);
    if (squared < 1e-12) { return best; }
    let axis = candidate * inverseSqrt(squared);
    let pa = dot(a, axis);
    let pb = dot(b, axis);
    let pc = dot(c, axis);
    let triangle_min = min(pa, min(pb, pc));
    let triangle_max = max(pa, max(pb, pc));
    let hull_min = dot(contact_hull_support(hull, -axis), axis);
    let hull_max = dot(contact_hull_support(hull, axis), axis);
    let forward = hull_max - triangle_min;
    let backward = triangle_max - hull_min;
    let separation = -min(forward, backward);
    if (separation > best.separation) {
        return BoxAxisResult(separation,
            axis * select(-1.0, 1.0, forward < backward), 0u);
    }
    return best;
}

fn scene_mesh_convex_manifold(sphere: Sphere, pose: Pose) -> SceneMeshConvexManifold {
    let header = u32(sphere.center_radius.w);
    let counts = spheres[header].indices;
    let hull = ContactHull(pose, sphere.center_radius.xyz, sphere.first_axis_end,
        header, counts.x, counts.y, counts.z);
    let mesh_translation = sphere.plane.xyz;
    let mesh_rotation = sphere.second_axis_end;
    let inverse_rotation = vec4<f32>(-mesh_rotation.xyz, mesh_rotation.w);
    let hull_center = contact_hull_center(hull);
    var lower = vec3<f32>(1e30);
    var upper = vec3<f32>(-1e30);
    for (var i = 0u; i < hull.vertex_count; i++) {
        let local = quat_rotate(inverse_rotation,
            contact_hull_vertex(hull, i) - mesh_translation);
        lower = min(lower, local);
        upper = max(upper, local);
    }
    lower -= vec3<f32>(1e-4);
    upper += vec3<f32>(1e-4);
    let vertex_start = u32(sphere.other_center_radius.x);
    let triangle_start = u32(sphere.other_center_radius.y);
    let node_start = u32(sphere.other_center_radius.z);
    let node_count = u32(sphere.other_center_radius.w);
    let absent = BoxContactGeometry(false, vec3<f32>(1.0, 0.0, 0.0),
        hull_center, 1.0);
    var first = absent;
    var second = absent;
    var third = absent;
    var fourth = absent;
    let diagonal = upper - lower;
    let distinct_squared = max(1e-8, dot(diagonal, diagonal) * 1e-4);
    var cursor = 0u;
    while (cursor < node_count) {
        let node = spheres[node_start + cursor];
        if (any(upper < node.center_radius.xyz)
            || any(lower > node.plane.xyz)) {
            cursor = node.indices.y;
            continue;
        }
        cursor++;
        if (node.indices.x == 0xffffffffu) { continue; }
        let triangle = spheres[triangle_start + node.indices.x].indices;
        let a = mesh_translation + quat_rotate(mesh_rotation,
            spheres[vertex_start + triangle.x].center_radius.xyz);
        let b = mesh_translation + quat_rotate(mesh_rotation,
            spheres[vertex_start + triangle.y].center_radius.xyz);
        let c = mesh_translation + quat_rotate(mesh_rotation,
            spheres[vertex_start + triangle.z].center_radius.xyz);
        let edges = array<vec3<f32>, 3>(b - a, c - b, a - c);
        let face = cross(edges[0], c - a);
        if (dot(face, face) < 1e-20) { continue; }
        var best = BoxAxisResult(-1e30, vec3<f32>(0.0, 0.0, 1.0), 0u);
        best = scene_mesh_convex_axis(hull, a, b, c, face, best);
        for (var i = 0u; i < hull.normal_count; i++) {
            best = scene_mesh_convex_axis(hull, a, b, c,
                contact_hull_direction(hull, i, false), best);
        }
        for (var i = 0u; i < hull.edge_count; i++) {
            let edge = contact_hull_direction(hull, i, true);
            for (var j = 0u; j < 3u; j++) {
                best = scene_mesh_convex_axis(hull, a, b, c,
                    cross(edge, edges[j]), best);
            }
        }
        if (best.separation > 1e-5) { continue; }
        var polygon: BoxFacePolygon;
        var reference = vec3<f32>(0.0, 0.0, 1.0);
        if (abs(best.normal.z) >= 0.9) { reference = vec3<f32>(1.0, 0.0, 0.0); }
        let tangent = normalize(cross(best.normal, reference));
        let bitangent = cross(best.normal, tangent);
        let hull_height = dot(contact_hull_support(hull, best.normal), best.normal);
        let tolerance = 1e-5 * (length(hull_center) + 1.0);
        let hull_face = contact_hull_face(hull, best.normal,
            hull_height, tangent, bitangent, tolerance);
        var triangle_face: HullFacePolygon;
        triangle_face.count = 3u;
        set_contact_hull_face_point(&triangle_face, 0u,
            vec2<f32>(dot(a, tangent), dot(a, bitangent)));
        set_contact_hull_face_point(&triangle_face, 1u,
            vec2<f32>(dot(b, tangent), dot(b, bitangent)));
        set_contact_hull_face_point(&triangle_face, 2u,
            vec2<f32>(dot(c, tangent), dot(c, bitangent)));
        let triangle_polygon = contact_hull_2d(triangle_face, tolerance);
        let triangle_normal = face * inverseSqrt(dot(face, face));
        let triangle_projection = dot(triangle_normal, best.normal);
        var projected = false;
        // Clip support features tangentially: SAT admits near contacts without volume overlap.
        if (!hull_face.overflow && hull_face.count > 0u
            && triangle_polygon.count == 3u && abs(triangle_projection) > 1e-6) {
            var clipped = hull_face;
            for (var i = 0u; i < triangle_polygon.count; i++) {
                clipped = clip_contact_hull_face(clipped, triangle_polygon.points[i],
                    triangle_polygon.points[(i + 1u) % triangle_polygon.count], tolerance);
            }
            let valid = contact_hull_2d(clipped, tolerance);
            if (!clipped.overflow && !valid.overflow) {
                projected = true;
                polygon.count = valid.count;
                for (var i = 0u; i < valid.count; i++) {
                    let point = tangent * valid.points[i].x + bitangent * valid.points[i].y;
                    let height = dot(a - point, triangle_normal) / triangle_projection;
                    set_box_face_point(&polygon, i, point + best.normal * height);
                }
            }
        }
        if (!projected) {
            polygon.count = 3u;
            set_box_face_point(&polygon, 0u, a);
            set_box_face_point(&polygon, 1u, b);
            set_box_face_point(&polygon, 2u, c);
            for (var i = 0u; i < hull.normal_count; i++) {
                let normal = contact_hull_direction(hull, i, false);
                let limit = dot(contact_hull_support(hull, normal), normal);
                polygon = clip_box_face(polygon, vec3<f32>(0.0), normal, limit);
            }
        }
        for (var i = 0u; i < polygon.count; i++) {
            let separation = select(best.separation,
                dot(polygon.points[i], best.normal) - hull_height, projected);
            if (separation > 1e-5) { continue; }
            let candidate = BoxContactGeometry(true, best.normal,
                polygon.points[i] - best.normal * separation * 0.5, separation);
            let duplicate = (first.enabled
                    && dot(candidate.point - first.point, candidate.point - first.point) <= distinct_squared)
                || (second.enabled
                    && dot(candidate.point - second.point, candidate.point - second.point) <= distinct_squared)
                || (third.enabled
                    && dot(candidate.point - third.point, candidate.point - third.point) <= distinct_squared)
                || (fourth.enabled
                    && dot(candidate.point - fourth.point, candidate.point - fourth.point) <= distinct_squared);
            if (duplicate) { continue; }
            if (!first.enabled || candidate.distance < first.distance) {
                fourth = third;
                third = second;
                second = first;
                first = candidate;
            } else if (!second.enabled || candidate.distance < second.distance) {
                fourth = third;
                third = second;
                second = candidate;
            } else if (!third.enabled || candidate.distance < third.distance) {
                fourth = third;
                third = candidate;
            } else if (!fourth.enabled || candidate.distance < fourth.distance) {
                fourth = candidate;
            }
        }
    }
    return SceneMeshConvexManifold(first, second, third, fourth);
}

fn cache_scene_mesh_convex_contact(row: u32, geometry: BoxContactGeometry,
    first_row: bool) {
    spheres[row].other_center_of_mass = vec4<f32>(geometry.point,
        select(1.0, geometry.distance, geometry.enabled));
    // Only the first row needs mesh offsets on later steps. Other rows keep
    // their previous normals intact for warm-start validation.
    if (first_row) {
        spheres[row].previous_normal = vec4<f32>(geometry.normal,
            select(0.0, 1.0, geometry.enabled));
    } else {
        spheres[row].other_center_radius = vec4<f32>(geometry.normal,
            spheres[row].other_center_radius.w);
        spheres[row].previous_normal.w = select(0.0, 1.0, geometry.enabled);
    }
}

fn contact_hulls_intersect(first: ContactHull, second: ContactHull,
    first_center: vec3<f32>, second_center: vec3<f32>) -> bool {
    var direction = second_center - first_center;
    if (dot(direction, direction) < 1e-16) { direction = vec3<f32>(1.0, 0.0, 0.0); }
    let initial = contact_hull_support(first, direction)
        - contact_hull_support(second, -direction);
    var simplex = AxialSimplex(initial, vec3<f32>(0.0), vec3<f32>(0.0),
        vec3<f32>(0.0), 1u, -initial, false);
    for (var iteration = 0u; iteration < 32u; iteration++) {
        if (dot(simplex.direction, simplex.direction) < 1e-16) { return true; }
        let point = contact_hull_support(first, simplex.direction)
            - contact_hull_support(second, -simplex.direction);
        let projection = dot(point, simplex.direction);
        if (projection < -1e-6) { return false; }
        if (dot(point - simplex.a, point - simplex.a) < 1e-14 ||
            (simplex.count > 1u && dot(point - simplex.b, point - simplex.b) < 1e-14) ||
            (simplex.count > 2u && dot(point - simplex.c, point - simplex.c) < 1e-14)) {
            return abs(projection) < 1e-6;
        }
        simplex = expand_axial_simplex(simplex, point);
        if (simplex.inside) { return true; }
    }
    return false;
}

fn consider_contact_hull_axis(first: ContactHull, second: ContactHull,
    candidate: vec3<f32>, best: BoxAxisResult) -> BoxAxisResult {
    let squared = dot(candidate, candidate);
    if (squared < 1e-12) { return best; }
    let axis = candidate * inverseSqrt(squared);
    let first_min = dot(contact_hull_support(first, -axis), axis);
    let first_max = dot(contact_hull_support(first, axis), axis);
    let second_min = dot(contact_hull_support(second, -axis), axis);
    let second_max = dot(contact_hull_support(second, axis), axis);
    let forward = first_max - second_min;
    let backward = second_max - first_min;
    let separation = -min(forward, backward);
    if (separation > best.separation) {
        return BoxAxisResult(separation,
            axis * select(-1.0, 1.0, forward < backward), 0u);
    }
    return best;
}

fn contact_hull_pair_geometry(sphere: Sphere, first_pose: Pose,
    second_pose: Pose, row: u32) -> BoxContactGeometry {
    let first_index = u32(sphere.center_radius.w);
    let second_index = u32(sphere.plane.w);
    let first_counts = spheres[first_index].indices;
    let second_counts = spheres[second_index].indices;
    let first = ContactHull(first_pose, sphere.center_radius.xyz,
        sphere.first_axis_end, first_index, first_counts.x, first_counts.y, first_counts.z);
    let second = ContactHull(second_pose, sphere.plane.xyz,
        sphere.second_axis_end, second_index, second_counts.x, second_counts.y, second_counts.z);
    let first_center = contact_hull_center(first);
    let second_center = contact_hull_center(second);
    var best = BoxAxisResult(-1e30, vec3<f32>(1.0, 0.0, 0.0), 0u);
    for (var i = 0u; i < first.normal_count; i++) {
        best = consider_contact_hull_axis(first, second,
            contact_hull_direction(first, i, false), best);
    }
    for (var i = 0u; i < second.normal_count; i++) {
        best = consider_contact_hull_axis(first, second,
            contact_hull_direction(second, i, false), best);
    }
    for (var i = 0u; i < first.edge_count; i++) {
        let first_edge = contact_hull_direction(first, i, true);
        for (var j = 0u; j < second.edge_count; j++) {
            let next = consider_contact_hull_axis(first, second,
                cross(first_edge, contact_hull_direction(second, j, true)), best);
            if (next.separation > best.separation + 1e-6) {
                best = BoxAxisResult(next.separation, next.normal, 1u);
            }
        }
    }
    best = consider_contact_hull_axis(first, second, second_center - first_center, best);
    let first_witness = contact_hull_support(first, best.normal);
    let second_witness = contact_hull_support(second, -best.normal);
    let witness_midpoint = 0.5 * (first_witness + second_witness);
    let center_midpoint = 0.5 * (first_center + second_center);
    let point = center_midpoint + best.normal
        * dot(witness_midpoint - center_midpoint, best.normal);
    let enabled = best.separation <= 1e-6
        && contact_hulls_intersect(first, second, first_center, second_center);
    let fallback = BoxContactGeometry(enabled && row == 0u,
        best.normal, point, best.separation);
    if (!enabled || best.axis_index == 1u) { return fallback; }

    var reference = vec3<f32>(0.0, 0.0, 1.0);
    if (abs(best.normal.z) >= 0.9) { reference = vec3<f32>(1.0, 0.0, 0.0); }
    let tangent = normalize(cross(best.normal, reference));
    let bitangent = cross(best.normal, tangent);
    let first_height = dot(contact_hull_support(first, best.normal), best.normal);
    let second_height = dot(contact_hull_support(second, -best.normal), best.normal);
    let extent = max(length(first_center), length(second_center)) + 1.0;
    let tolerance = 1e-5 * extent;
    let first_face = contact_hull_face(first, best.normal,
        first_height, tangent, bitangent, tolerance);
    let second_face = contact_hull_face(second, best.normal,
        second_height, tangent, bitangent, tolerance);
    if (first_face.overflow || second_face.overflow
        || first_face.count < 3u || second_face.count < 3u) { return fallback; }
    var polygon = first_face;
    for (var i = 0u; i < second_face.count; i++) {
        polygon = clip_contact_hull_face(polygon, second_face.points[i],
            second_face.points[(i + 1u) % second_face.count], tolerance);
        if (polygon.overflow || polygon.count == 0u) { return fallback; }
    }
    let valid = contact_hull_2d(polygon, tolerance);
    if (valid.overflow || valid.count == 0u) { return fallback; }
    var selected = valid;
    if (valid.count > 4u) {
        var remaining = valid;
        selected.count = 1u;
        for (var i = 1u; i < remaining.count; i++) {
            set_contact_hull_face_point(&remaining, i - 1u, remaining.points[i]);
        }
        remaining.count--;
        while (selected.count < 4u) {
            var best_index = 0u;
            var best_spread = -1.0;
            for (var i = 0u; i < remaining.count; i++) {
                var spread = 1e30;
                for (var j = 0u; j < selected.count; j++) {
                    let offset = remaining.points[i] - selected.points[j];
                    spread = min(spread, dot(offset, offset));
                }
                if (spread >= best_spread) {
                    best_spread = spread;
                    best_index = i;
                }
            }
            set_contact_hull_face_point(&selected, selected.count,
                remaining.points[best_index]);
            selected.count++;
            set_contact_hull_face_point(&remaining, best_index,
                remaining.points[remaining.count - 1u]);
            remaining.count--;
        }
    }
    let count = selected.count;
    if (row >= count) {
        return BoxContactGeometry(false, best.normal, point, best.separation);
    }
    let selected_point = selected.points[row];
    return BoxContactGeometry(true, best.normal,
        tangent * selected_point.x + bitangent * selected_point.y
            + best.normal * (0.5 * (first_height + second_height)),
        best.separation);
}

fn axial_hull_support(shape: AxialShape, hull: ContactHull,
    direction: vec3<f32>) -> vec3<f32> {
    return axial_shape_support(shape, direction)
        - contact_hull_support(hull, -direction);
}

fn axial_hull_intersects(shape: AxialShape, hull: ContactHull,
    hull_center: vec3<f32>) -> bool {
    var direction = hull_center - shape.center;
    if (dot(direction, direction) < 1e-16) { direction = vec3<f32>(1.0, 0.0, 0.0); }
    let support = axial_hull_support(shape, hull, direction);
    var simplex = AxialSimplex(support, vec3<f32>(0.0), vec3<f32>(0.0),
        vec3<f32>(0.0), 1u, -support, false);
    for (var iteration = 0u; iteration < 32u; iteration++) {
        if (dot(simplex.direction, simplex.direction) < 1e-16) { return true; }
        let point = axial_hull_support(shape, hull, simplex.direction);
        let projection = dot(point, simplex.direction);
        if (projection < -1e-6) { return false; }
        if (dot(point - simplex.a, point - simplex.a) < 1e-14 ||
            (simplex.count > 1u && dot(point - simplex.b, point - simplex.b) < 1e-14) ||
            (simplex.count > 2u && dot(point - simplex.c, point - simplex.c) < 1e-14)) {
            return abs(projection) < 1e-6;
        }
        simplex = expand_axial_simplex(simplex, point);
        if (simplex.inside) { return true; }
    }
    return false;
}

fn consider_axial_hull_axis(shape: AxialShape, hull: ContactHull,
    candidate: vec3<f32>, best: BoxAxisResult) -> BoxAxisResult {
    let squared = dot(candidate, candidate);
    if (squared < 1e-10) { return best; }
    let normal = candidate * inverseSqrt(squared);
    let axial_min = dot(axial_shape_support(shape, -normal), normal);
    let axial_max = dot(axial_shape_support(shape, normal), normal);
    let hull_min = dot(contact_hull_support(hull, -normal), normal);
    let hull_max = dot(contact_hull_support(hull, normal), normal);
    let forward = axial_max - hull_min;
    let backward = hull_max - axial_min;
    let separation = -min(forward, backward);
    if (separation > best.separation) {
        return BoxAxisResult(separation,
            normal * select(-1.0, 1.0, forward < backward), 0u);
    }
    return best;
}

// Project onto the support plane, then clamp to the hull support feature.
// Opposing face normals are retained by the hull builder. Interior projections
// need no edge search; outside projections use support-vertex segments.
fn contact_hull_support_near(hull: ContactHull, normal: vec3<f32>,
    projection: f32, reference: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let point = reference + normal * (projection - dot(reference, normal));
    var inside = true;
    for (var face = 0u; face < hull.normal_count; face++) {
        let axis = contact_hull_direction(hull, face, false);
        var bound = -1e30;
        for (var vertex = 0u; vertex < hull.vertex_count; vertex++) {
            bound = max(bound, dot(contact_hull_vertex(hull, vertex), axis));
        }
        if dot(point, axis) > bound + 1e-6 { inside = false; }
    }
    if inside { return point; }
    var nearest = fallback;
    var distance = dot(point - nearest, point - nearest);
    for (var first = 0u; first < hull.vertex_count; first++) {
        let a = contact_hull_vertex(hull, first);
        if abs(dot(a, normal) - projection) > 1e-6 { continue; }
        for (var second = first; second < hull.vertex_count; second++) {
            let b = contact_hull_vertex(hull, second);
            if abs(dot(b, normal) - projection) > 1e-6 { continue; }
            let edge = b - a;
            let t = clamp(dot(point - a, edge) / max(dot(edge, edge), 1e-20), 0.0, 1.0);
            let candidate = a + edge * t;
            let squared = dot(point - candidate, point - candidate);
            if squared < distance { nearest = candidate; distance = squared; }
        }
    }
    return nearest;
}

fn axial_hull_geometry(sphere: Sphere, pose: Pose,
    second_pose: Pose) -> BoxContactGeometry {
    let shape = AxialShape(
        pose.position.xyz + quat_rotate(pose.orientation, sphere.center_radius.xyz),
        quat_rotate(pose.orientation,
            quat_rotate(sphere.first_axis_end, vec3<f32>(0.0, 0.0, 1.0))),
        sphere.center_of_mass.w, sphere.center_radius.w,
        select(31.0, 30.0,
            sphere.material.w == 50.0 || sphere.material.w == 58.0
                || sphere.material.w == 70.0));
    let header = u32(sphere.plane.w);
    let counts = spheres[header].indices;
    let hull = ContactHull(
        second_pose, sphere.plane.xyz, sphere.second_axis_end,
        header, counts.x, counts.y, counts.z);
    let hull_center = contact_hull_center(hull);
    let delta = hull_center - shape.center;
    var best = BoxAxisResult(-1e30, vec3<f32>(1.0, 0.0, 0.0), 0u);
    best = consider_axial_hull_axis(shape, hull, delta, best);
    best = consider_axial_hull_axis(shape, hull, shape.axis, best);
    for (var i = 0u; i < hull.normal_count; i++) {
        best = consider_axial_hull_axis(shape, hull,
            contact_hull_direction(hull, i, false), best);
    }
    for (var i = 0u; i < hull.edge_count; i++) {
        best = consider_axial_hull_axis(shape, hull,
            cross(shape.axis, contact_hull_direction(hull, i, true)), best);
    }
    for (var i = 0u; i < hull.vertex_count; i++) {
        let vertex = contact_hull_vertex(hull, i);
        let relative = vertex - shape.center;
        let radial = relative - shape.axis * dot(relative, shape.axis);
        best = consider_axial_hull_axis(shape, hull, radial, best);
        if (shape.kind == 31.0 && dot(radial, radial) > 1e-12) {
            best = consider_axial_hull_axis(shape, hull,
                normalize(radial) * (2.0 * shape.half_height)
                    + shape.axis * shape.radius, best);
        }
    }
    let first_witness = axial_shape_support(shape, best.normal);
    let support = contact_hull_support(hull, -best.normal);
    let second_witness = contact_hull_support_near(hull, -best.normal,
        dot(support, -best.normal), first_witness, support);
    let point = 0.5 * (first_witness + second_witness);
    let enabled = best.separation > 0.0
        || axial_hull_intersects(shape, hull, hull_center);
    return BoxContactGeometry(enabled, best.normal, point, best.separation);
}

fn axis_jacobian(
    term_offset: u32,
    n: u32,
    column: u32,
    axis: vec3<f32>,
    contact_offset: vec3<f32>,
) -> f32 {
    let linear = term_offset + 10u;
    let angular = linear + 3u * n;
    let linear_column = vec3<f32>(
        link_terms[linear + column],
        link_terms[linear + n + column],
        link_terms[linear + 2u * n + column],
    );
    let angular_column = vec3<f32>(
        link_terms[angular + column],
        link_terms[angular + n + column],
        link_terms[angular + 2u * n + column],
    );
    return dot(axis, linear_column + cross(angular_column, contact_offset));
}


fn constraint_quat_product(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz),
        a.w * b.w - dot(a.xyz, b.xyz));
}

fn link_constraint_jacobian(sphere: Sphere, n: u32, column: u32,
    first_offset: vec3<f32>, second_offset: vec3<f32>) -> f32 {
    let axis = sphere.plane.xyz;
    if (sphere.material.w == 79.0) {
        var result = axis_jacobian(sphere.indices.y, n, column, axis, first_offset);
        if (sphere.indices.z != 0xffffffffu) {
            result -= axis_jacobian(sphere.indices.w, n, column, axis, second_offset);
        }
        return result;
    }
    let first = sphere.indices.y + 10u + 3u * n;
    var result = dot(axis, vec3<f32>(link_terms[first + column],
        link_terms[first + n + column], link_terms[first + 2u * n + column]));
    if (sphere.indices.z != 0xffffffffu) {
        let second = sphere.indices.w + 10u + 3u * n;
        result -= dot(axis, vec3<f32>(link_terms[second + column],
            link_terms[second + n + column], link_terms[second + 2u * n + column]));
    }
    return result;
}

fn contact_owner_mode(kind: f32) -> u32 {
    if (kind == 48.0 || kind == 49.0 || kind == 83.0
        || kind == 60.0 || kind == 61.0
        || kind == 62.0 || kind == 63.0
        || kind == 64.0 || kind == 65.0
        || (kind >= 66.0 && kind <= 71.0)) {
        return 2u;
    }
    if (kind == 14.0 || kind == 15.0
        || kind == 16.0 || kind == 17.0
        || kind == 18.0 || kind == 19.0
        || kind == 20.0 || kind == 21.0
        || kind == 22.0 || kind == 23.0
        || kind == 26.0 || kind == 27.0
        || kind == 28.0 || kind == 29.0
        || kind == 30.0 || kind == 31.0
        || kind == 45.0 || (kind == 46.0 || kind == 81.0)
        || kind == 47.0 || kind == 50.0
        || kind == 51.0 || kind == 52.0
        || kind == 53.0 || kind == 54.0
        || kind == 55.0 || kind == 56.0
        || kind == 57.0 || kind == 72.0
        || kind == 73.0 || kind == 74.0
        || kind == 75.0 || kind == 76.0
        || kind == 77.0) { return 1u; }
    if (kind == 0.0 || kind == 5.0
        || kind == 10.0 || kind == 12.0
        || kind == 13.0 || kind == 24.0
        || kind == 25.0) { return 0u; }
    return 3u;
}

fn contact_jacobian(
    sphere: Sphere,
    n: u32,
    column: u32,
    axis: vec3<f32>,
    first_offset: vec3<f32>,
    second_offset: vec3<f32>,
) -> f32 {
    if (sphere.material.w == 48.0 || sphere.material.w == 49.0 || sphere.material.w == 83.0
        || sphere.material.w == 60.0 || sphere.material.w == 61.0
        || sphere.material.w == 62.0 || sphere.material.w == 63.0
        || sphere.material.w == 64.0 || sphere.material.w == 65.0
        || (sphere.material.w >= 66.0 && sphere.material.w <= 71.0)) {
        return axis_jacobian(sphere.indices.w, n, column, axis, second_offset);
    }
    let first = axis_jacobian(sphere.indices.y, n, column, axis, first_offset);
    if (sphere.material.w == 14.0 || sphere.material.w == 15.0
        || sphere.material.w == 16.0 || sphere.material.w == 17.0
        || sphere.material.w == 18.0 || sphere.material.w == 19.0
        || sphere.material.w == 20.0 || sphere.material.w == 21.0
        || sphere.material.w == 22.0 || sphere.material.w == 23.0
        || sphere.material.w == 26.0 || sphere.material.w == 27.0
        || sphere.material.w == 28.0 || sphere.material.w == 29.0
        || sphere.material.w == 30.0 || sphere.material.w == 31.0
        || sphere.material.w == 45.0 || (sphere.material.w == 46.0 || sphere.material.w == 81.0)
        || sphere.material.w == 47.0 || sphere.material.w == 50.0
        || sphere.material.w == 51.0 || sphere.material.w == 52.0
        || sphere.material.w == 53.0 || sphere.material.w == 54.0
        || sphere.material.w == 55.0 || sphere.material.w == 56.0
        || sphere.material.w == 57.0 || sphere.material.w == 72.0
        || sphere.material.w == 73.0 || sphere.material.w == 74.0
        || sphere.material.w == 75.0 || sphere.material.w == 76.0
        || sphere.material.w == 77.0) { return -first; }
    if (sphere.material.w == 0.0 || sphere.material.w == 5.0
        || sphere.material.w == 10.0 || sphere.material.w == 12.0
        || sphere.material.w == 13.0 || sphere.material.w == 24.0
        || sphere.material.w == 25.0) { return first; }
    return axis_jacobian(sphere.indices.w, n, column, axis, second_offset) - first;
}

fn apply_cached_impulse(
    system: System,
    sphere: Sphere,
    normal: vec3<f32>,
    tangent_one: vec3<f32>,
    tangent_two: vec3<f32>,
    first_offset: vec3<f32>,
    second_offset: vec3<f32>,
    impulse: vec3<f32>,
) {
    let n = system.indices.y;
    for (var row = 0u; row < n; row++) {
        var response = 0.0;
        for (var column = 0u; column < n; column++) {
            let jacobian = impulse.x * contact_jacobian(sphere, n, column, normal,
                first_offset, second_offset)
                + impulse.y * contact_jacobian(sphere, n, column, tangent_one,
                    first_offset, second_offset)
                + impulse.z * contact_jacobian(sphere, n, column, tangent_two,
                    first_offset, second_offset);
            response += inverse[system.inverse.x + row * n + column] * jacobian;
        }
        accelerations[system.indices.x + row] += response / sphere.material.y;
    }
}

// Solve both normal impulses together after the second scalar row. This keeps
// closely spaced supporting contacts balanced even with a small sweep count.
fn resolve_capsule_face_normal_block(system: System, index: u32, second: Sphere,
    second_normal: vec3<f32>, second_offset: vec3<f32>, second_other_offset: vec3<f32>,
    second_target: f32, second_impulse: f32) -> f32 {
    if (index == 0u) { return second_impulse; }
    let first = spheres[index - 1u];
    if ((first.material.w != 43.0 && first.material.w != 46.0 && first.material.w != 49.0)
        || (first.diagnostic_first_origin.w == 0.0 && first.diagnostic_second_origin.w == 0.0)
        || dot(first.previous_normal.xyz, second_normal) < 0.9995) { return second_impulse; }
    let n = system.indices.y;
    let dt = second.material.y;
    let first_normal = first.previous_normal.xyz;
    let first_point = first.diagnostic_first.xyz;
    let first_pose = poses[first.indices.x];
    let first_com = first_pose.position.xyz + quat_rotate(first_pose.orientation, first.center_of_mass.xyz);
    let first_offset = first_point - first_com;
    let other_pose = poses[first.indices.z];
    let other_com = other_pose.position.xyz + quat_rotate(other_pose.orientation, first.other_center_of_mass.xyz);
    let first_other_offset = first.diagnostic_second.xyz - other_com;
    let capsule_center = (first.plane.xyz + first.second_axis_end.xyz) * 0.5;
    var first_prescribed = select(vec3<f32>(0.0), first.prescribed_linear.xyz
        + cross(first.prescribed_angular.xyz, first_point - capsule_center), first.material.w == 46.0);
    if (first.material.w == 49.0) {
        first_prescribed = -(first.prescribed_linear.xyz
            + cross(first.prescribed_angular.xyz, first.diagnostic_second.xyz - first.center_radius.xyz));
    }
    var incoming = dot(first_normal, first_prescribed);
    var speed = vec2<f32>(incoming, dot(second_normal, select(vec3<f32>(0.0), second.prescribed_linear.xyz
        + cross(second.prescribed_angular.xyz,
            second.diagnostic_first.xyz - capsule_center), first.material.w == 46.0)));
    if (first.material.w == 49.0) {
        speed.y = dot(second_normal, -(second.prescribed_linear.xyz
            + cross(second.prescribed_angular.xyz, second.diagnostic_second.xyz - second.center_radius.xyz)));
    }
    var k11 = 0.0;
    var k12 = 0.0;
    var k22 = 0.0;
    for (var row = 0u; row < n; row++) {
        let j1 = contact_jacobian(first, n, row, first_normal, first_offset, first_other_offset);
        let j2 = contact_jacobian(second, n, row, second_normal, second_offset, second_other_offset);
        incoming += j1 * velocities[system.indices.x + row];
        speed += vec2<f32>(j1, j2) * (velocities[system.indices.x + row]
            + dt * accelerations[system.indices.x + row]);
        var r1 = 0.0;
        var r2 = 0.0;
        for (var column = 0u; column < n; column++) {
            let entry = inverse[system.inverse.x + row * n + column];
            r1 += entry * contact_jacobian(first, n, column, first_normal, first_offset, first_other_offset);
            r2 += entry * contact_jacobian(second, n, column, second_normal, second_offset, second_other_offset);
        }
        k11 += j1 * r1;
        k12 += j1 * r2;
        k22 += j2 * r2;
    }
    let determinant = k11 * k22 - k12 * k12;
    let trace = k11 + k22;
    if (!(determinant > 1e-6 * trace * trace) || !(trace < 1e30)) { return second_impulse; }
    var first_target_velocity = -first.diagnostic_first.w / dt;
    if (first.diagnostic_first.w < 0.0) {
        first_target_velocity = contact_recovery_velocity(first.diagnostic_first.w, dt);
    }
    if (first.material.x > 0.0 && first.diagnostic_first.w <= 0.0) {
        first_target_velocity = max(first_target_velocity, -first.material.x * min(incoming, 0.0));
    }
    let old = vec2<f32>(first.impulses.x, second_impulse);
    let rhs = vec2<f32>(first_target_velocity, second_target) - speed
        + vec2<f32>(k11 * old.x + k12 * old.y, k12 * old.x + k22 * old.y);
    var solved = vec2<f32>((k22 * rhs.x - k12 * rhs.y) / determinant,
        (k11 * rhs.y - k12 * rhs.x) / determinant);
    // Enumerate the four active sets of the two-contact complementarity problem.
    if (any(solved < vec2<f32>(0.0))) {
        let first_only = max(rhs.x / k11, 0.0);
        let second_only = max(rhs.y / k22, 0.0);
        if (rhs.x >= 0.0 && k12 * first_only - rhs.y >= -1e-7) { solved = vec2<f32>(first_only, 0.0); }
        else if (rhs.y >= 0.0 && k12 * second_only - rhs.x >= -1e-7) { solved = vec2<f32>(0.0, second_only); }
        else if (all(rhs <= vec2<f32>(0.0))) { solved = vec2<f32>(0.0); }
        else { return second_impulse; }
    }
    if (!all(abs(solved) < vec2<f32>(1e30))) { return second_impulse; }
    let delta = solved - old;
    let reference = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(first_normal.z) >= 0.9);
    let tangent_one = normalize(cross(first_normal, reference));
    let tangent_two = cross(first_normal, tangent_one);
    var tangent = first.impulses.yz;
    let magnitude = length(tangent);
    let limit = first.material.z * solved.x;
    if (magnitude > limit) { tangent *= limit / magnitude; }
    // A reduced normal impulse must also reduce any already solved friction.
    apply_cached_impulse(system, first, first_normal, tangent_one, tangent_two,
        first_offset, first_other_offset, vec3<f32>(delta.x, tangent - first.impulses.yz));
    spheres[index - 1u].impulses.y = tangent.x;
    spheres[index - 1u].impulses.z = tangent.y;
    apply_cached_impulse(system, second, second_normal, vec3<f32>(0.0), vec3<f32>(0.0),
        second_offset, second_other_offset, vec3<f32>(delta.y, 0.0, 0.0));
    spheres[index - 1u].impulses.x = solved.x;
    return solved.y;
}

fn is_support_point(kind: f32) -> bool {
    return kind == 5.0 || kind == 10.0 || kind == 12.0 || kind == 13.0;
}

fn axial_ground_point(sphere: Sphere, pose: Pose) -> vec3<f32> {
    let center = pose.position.xyz + quat_rotate(pose.orientation, sphere.center_radius.xyz);
    let axis = quat_rotate(pose.orientation,
        quat_rotate(sphere.second_axis_end, vec3<f32>(0.0, 0.0, 1.0)));
    let down = -sphere.plane.xyz;
    let axial = dot(down, axis);
    let radial = down - axis * axial;
    let radial_squared = dot(radial, radial);
    let radius = sphere.other_center_radius.x;
    let half_height = sphere.other_center_radius.y;
    if (radial_squared > 1e-8) {
        let rim = radial * (radius * inverseSqrt(radial_squared));
        if (sphere.material.w == 12.0) {
            var cap_sign = select(-1.0, 1.0, axial >= 0.0);
            if (abs(axial) < 1e-6) {
                cap_sign = select(-1.0, 1.0, (sphere.indices.z & 1u) != 0u);
            }
            return center + axis * (cap_sign * half_height) + rim;
        }
        let apex = center + axis * half_height;
        let base = center - axis * half_height + rim;
        return select(base, apex, dot(down, apex) > dot(down, base));
    }
    if (sphere.material.w == 13.0 && axial >= 0.0) {
        return center + axis * half_height;
    }
    let cap = center + axis * select(-half_height, half_height,
        sphere.material.w == 12.0 && axial >= 0.0);
    let reference = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0),
        abs(axis.x) < 0.9);
    let tangent_a = normalize(cross(axis, reference));
    let tangent_b = normalize(cross(axis, tangent_a));
    let slot = sphere.indices.z;
    if (slot == 0u) { return cap + tangent_a * radius; }
    if (slot == 1u) { return cap - tangent_a * radius; }
    if (slot == 2u) { return cap + tangent_b * radius; }
    return cap - tangent_b * radius;
}

fn ground_point_candidate(sphere: Sphere) -> vec4<f32> {
    let pose = poses[sphere.indices.x];
    var center = pose.position.xyz
        + quat_rotate(pose.orientation, sphere.center_radius.xyz);
    if (sphere.material.w == 12.0 || sphere.material.w == 13.0) {
        center = axial_ground_point(sphere, pose);
    } else if (sphere.material.w == 5.0) {
        let normal = sphere.plane.xyz;
        let link_normal = quat_rotate(
            vec4<f32>(-pose.orientation.xyz, pose.orientation.w), normal);
        let local_orientation = sphere.second_axis_end;
        let local_normal = quat_rotate(
            vec4<f32>(-local_orientation.xyz, local_orientation.w), link_normal);
        let half_extents = sphere.other_center_radius.xyz;
        let first_sign = select(-1.0, 1.0, (sphere.indices.z & 1u) != 0u);
        let second_sign = select(-1.0, 1.0, (sphere.indices.z & 2u) != 0u);
        var corner = vec3<f32>(0.0);
        if (abs(local_normal.x) >= abs(local_normal.y)
            && abs(local_normal.x) >= abs(local_normal.z)) {
            corner = vec3<f32>(select(1.0, -1.0, local_normal.x >= 0.0),
                first_sign, second_sign) * half_extents;
        } else if (abs(local_normal.y) >= abs(local_normal.z)) {
            corner = vec3<f32>(first_sign,
                select(1.0, -1.0, local_normal.y >= 0.0), second_sign) * half_extents;
        } else {
            corner = vec3<f32>(first_sign, second_sign,
                select(1.0, -1.0, local_normal.z >= 0.0)) * half_extents;
        }
        let local_point = sphere.center_radius.xyz + quat_rotate(local_orientation, corner);
        center = pose.position.xyz + quat_rotate(pose.orientation, local_point);
    }
    let extent = sphere.other_center_radius.w;
    let allowed_extent = extent + select(sphere.center_radius.w, 0.0,
        sphere.material.w != 10.0);
    if (extent > 0.0 && (abs(center.x) > allowed_extent || abs(center.y) > allowed_extent)) {
        return vec4<f32>(center, -1e30);
    }
    let distance = dot(sphere.plane.xyz, center) - sphere.plane.w - sphere.center_radius.w;
    return vec4<f32>(center, max(-distance, 0.0));
}

fn same_ground_point_group(first: Sphere, second: Sphere) -> bool {
    return is_support_point(second.material.w)
        && first.indices.x == second.indices.x
        && all(first.plane == second.plane);
}

fn select_ground_points(system: System) {
    // Keep the deepest point and up to three spatially separated support points per link and plane.
    for (var row = 0u; row < system.indices.w; row++) {
        let index = system.indices.z + row;
        if (is_support_point(spheres[index].material.w)) {
            spheres[index].indices.w = 0u;
        }
    }
    for (var row = 0u; row < system.indices.w; row++) {
        let leader = spheres[system.indices.z + row];
        if (!is_support_point(leader.material.w)) { continue; }
        var seen = false;
        for (var previous = 0u; previous < row; previous++) {
            if (same_ground_point_group(leader, spheres[system.indices.z + previous])) {
                seen = true;
                break;
            }
        }
        if (seen) { continue; }
        var selected = array<u32, 4>(0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu);
        var selected_count = 0u;
        var deepest = -1e30;
        for (var candidate_row = row; candidate_row < system.indices.w; candidate_row++) {
            let candidate = spheres[system.indices.z + candidate_row];
            if (!same_ground_point_group(leader, candidate)) { continue; }
            let point = ground_point_candidate(candidate);
            if (point.w >= 0.0 && point.w > deepest) {
                deepest = point.w;
                selected[0] = candidate_row;
            }
        }
        if (selected[0] == 0xffffffffu) { continue; }
        selected_count = 1u;
        let cutoff = max(deepest - 0.02, 0.0);
        for (var slot = 1u; slot < 4u; slot++) {
            var best_row = 0xffffffffu;
            var best_spread = -1.0;
            for (var candidate_row = row; candidate_row < system.indices.w; candidate_row++) {
                let candidate = spheres[system.indices.z + candidate_row];
                if (!same_ground_point_group(leader, candidate)) { continue; }
                let point = ground_point_candidate(candidate);
                if (point.w < cutoff) { continue; }
                var already_selected = false;
                for (var chosen = 0u; chosen < selected_count; chosen++) {
                    if (selected[chosen] == candidate_row) { already_selected = true; }
                }
                if (already_selected) { continue; }
                var spread = 1e30;
                for (var chosen = 0u; chosen < selected_count; chosen++) {
                    let other = ground_point_candidate(
                        spheres[system.indices.z + selected[chosen]]);
                    let delta = point.xy - other.xy;
                    spread = min(spread, dot(delta, delta));
                }
                if (spread > best_spread) {
                    best_spread = spread;
                    best_row = candidate_row;
                }
            }
            if (best_row == 0xffffffffu || best_spread < 1e-10) { break; }
            selected[selected_count] = best_row;
            selected_count += 1u;
        }
        for (var chosen = 0u; chosen < selected_count; chosen++) {
            spheres[system.indices.z + selected[chosen]].indices.w = 1u;
        }
    }
    for (var row = 0u; row < system.indices.w; row++) {
        let index = system.indices.z + row;
        if (is_support_point(spheres[index].material.w)
            && spheres[index].indices.w == 0u) {
            spheres[index].impulses = vec4<f32>(0.0);
            spheres[index].previous_normal = vec4<f32>(0.0);
        }
    }
}

@compute @workgroup_size(1)
fn resolve_ground_contacts(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let environment = invocation.x;
    if (environment >= arrayLength(&systems)
        || atomicLoad(&state_status[environment]) != 0u) { return; }
    let system = systems[environment];
    if (inverse[system.inverse.y] != 0.0) { return; }
    let n = system.indices.y;
    let coordinate_offset = system.indices.x;
    select_ground_points(system);
    // Geometry ownership belongs to this solve, while warm-start impulses/normals
    // may be retained. Skipped and inactive rows must not publish old witnesses.
    for (var row = 0u; row < system.indices.w; row++) {
        let index = system.indices.z + row;
        spheres[index].diagnostic_first_origin = vec4<f32>(0.0);
        spheres[index].diagnostic_second_origin = vec4<f32>(0.0);
    }
    if (system.inverse.w == 0u) {
        for (var contact = 0u; contact < system.indices.w; contact++) {
            spheres[system.indices.z + contact].impulses = vec4<f32>(0.0);
        }
    }
    // Seed every contact before projecting any row, as in the CPU solver.
    let warm_start = system.inverse.w != 0u;
    let iteration_count = system.inverse.z + select(0u, 1u, warm_start);
    for (var iteration = 0u; iteration < iteration_count; iteration++) {
    let seed_only = warm_start && iteration == 0u;
    for (var contact = 0u; contact < system.indices.w; contact++) {
        let sphere_index = system.indices.z + contact;
        let sphere = spheres[sphere_index];
        if (seed_only && (sphere.material.w == 11.0 || sphere.material.w == 78.0
            || sphere.material.w == 79.0 || sphere.material.w == 80.0)) {
            continue;
        }
        if (sphere.material.w == 79.0 || sphere.material.w == 80.0) {
            let first_pose = poses[sphere.indices.x];
            var second_pose = Pose(vec4<f32>(0.0), vec4<f32>(0.0, 0.0, 0.0, 1.0));
            if (sphere.indices.z != 0xffffffffu) { second_pose = poses[sphere.indices.z]; }
            let first_point = first_pose.position.xyz
                + quat_rotate(first_pose.orientation, sphere.center_radius.xyz);
            let second_point = second_pose.position.xyz
                + quat_rotate(second_pose.orientation, sphere.other_center_radius.xyz);
            let first_offset = first_point - (first_pose.position.xyz
                + quat_rotate(first_pose.orientation, sphere.center_of_mass.xyz));
            let second_offset = second_point - (second_pose.position.xyz
                + quat_rotate(second_pose.orientation, sphere.other_center_of_mass.xyz));
            var error = first_point - second_point;
            if (sphere.material.w == 80.0) {
                let a = constraint_quat_product(first_pose.orientation, sphere.first_axis_end);
                let b = constraint_quat_product(second_pose.orientation, sphere.second_axis_end);
                var rotation = normalize(constraint_quat_product(a, vec4<f32>(-b.xyz, b.w)));
                if (rotation.w < 0.0) { rotation = -rotation; }
                let sine = length(rotation.xyz);
                error = 2.0 * rotation.xyz;
                if (sine > 1e-6) { error = rotation.xyz * (2.0 * atan2(sine, rotation.w) / sine); }
            }
            let axis_error = dot(sphere.plane.xyz, error);
            let dt = sphere.material.y;
            let target_speed = clamp(-sphere.material.x * axis_error / dt,
                -sphere.material.z, sphere.material.z);
            var prescribed_speed = 0.0;
            if (sphere.indices.z == 0xffffffffu) {
                prescribed_speed = dot(sphere.plane.xyz,
                    select(sphere.prescribed_linear.xyz, sphere.prescribed_angular.xyz,
                        sphere.material.w == 80.0));
            }
            var relative_velocity = 0.0;
            var effective = 0.0;
            for (var row = 0u; row < n; row++) {
                let j = link_constraint_jacobian(sphere, n, row, first_offset, second_offset);
                relative_velocity += j * (velocities[coordinate_offset + row]
                    + dt * accelerations[coordinate_offset + row]);
                var response = 0.0;
                for (var column = 0u; column < n; column++) {
                    response += inverse[system.inverse.x + row * n + column]
                        * link_constraint_jacobian(sphere, n, column, first_offset, second_offset);
                }
                effective += j * response;
            }
            if (!(effective < 1e30) || !(abs(target_speed) < 1e30)) {
                atomicOr(&state_status[environment], 1u); return;
            }
            if (effective <= 1e-12) {
                if (abs(axis_error) > 1e-6
                    || abs(prescribed_speed - relative_velocity) > 1e-6) {
                    atomicOr(&state_status[environment], 1u); return;
                }
                continue;
            }
            let impulse = (target_speed + prescribed_speed - relative_velocity) / effective;
            if (!(abs(impulse) < 1e30)) { atomicOr(&state_status[environment], 1u); return; }
            for (var row = 0u; row < n; row++) {
                var response = 0.0;
                for (var column = 0u; column < n; column++) {
                    response += inverse[system.inverse.x + row * n + column]
                        * link_constraint_jacobian(sphere, n, column, first_offset, second_offset);
                }
                accelerations[coordinate_offset + row] += response * impulse / dt;
            }
            continue;
        }
        if (sphere.material.w == 78.0) {
            let follower = sphere.indices.x;
            let source = sphere.indices.y;
            let dt = sphere.material.y;
            let derivative = sphere.plane.z;
            var relative_velocity = velocities[coordinate_offset + follower]
                + dt * accelerations[coordinate_offset + follower];
            var effective = inverse[system.inverse.x + follower * n + follower];
            if (source != 0xffffffffu) {
                relative_velocity -= derivative * (velocities[coordinate_offset + source]
                    + dt * accelerations[coordinate_offset + source]);
                effective += derivative * derivative * inverse[system.inverse.x + source * n + source]
                    - derivative * (inverse[system.inverse.x + follower * n + source]
                        + inverse[system.inverse.x + source * n + follower]);
            }
            if (!(effective > 1e-8) || !(effective < 1e30)) { continue; }
            let impulse = (sphere.plane.y - relative_velocity) / effective;
            if (!(abs(impulse) < 1e30)) {
                atomicOr(&state_status[environment], 1u);
                return;
            }
            for (var row = 0u; row < n; row++) {
                var response = inverse[system.inverse.x + row * n + follower];
                if (source != 0xffffffffu) {
                    response -= derivative * inverse[system.inverse.x + row * n + source];
                }
                accelerations[coordinate_offset + row] += response * impulse / dt;
            }
            continue;
        }
        if (sphere.material.w == 11.0) {
            let coordinate = sphere.indices.x;
            let dt = sphere.material.y;
            let diagonal = inverse[system.inverse.x + coordinate * n + coordinate];
            if (!(diagonal > 1e-8) || !(diagonal < 1e30)) { continue; }
            let current = select(sphere.impulses.x, 0.0,
                iteration == select(0u, 1u, warm_start));
            let predicted = velocities[coordinate_offset + coordinate]
                + dt * accelerations[coordinate_offset + coordinate];
            let bounded = clamp(current - predicted / diagonal,
                -sphere.material.x, sphere.material.x);
            let delta = bounded - current;
            if (!(abs(delta) < 1e30)) { continue; }
            for (var row = 0u; row < n; row++) {
                accelerations[coordinate_offset + row] +=
                    inverse[system.inverse.x + row * n + coordinate] * delta / dt;
            }
            spheres[sphere_index].impulses = vec4<f32>(bounded, 0.0, 0.0, 0.0);
            continue;
        }
        if (is_support_point(sphere.material.w)
            && sphere.indices.w == 0u) { continue; }
        // Stationary polyline segments are valid zero-radius capsule rows.
        if (!is_support_point(sphere.material.w)
            && sphere.center_radius.w <= 0.0
            && sphere.material.w != 24.0 && sphere.material.w != 25.0
            && sphere.material.w != 28.0 && sphere.material.w != 29.0
            && (sphere.material.w != 46.0 && sphere.material.w != 81.0)) { continue; }
        var pose = Pose(vec4<f32>(0.0), vec4<f32>(0.0, 0.0, 0.0, 1.0));
        if (sphere.material.w != 48.0 && sphere.material.w != 49.0 && sphere.material.w != 83.0
            && sphere.material.w != 60.0 && sphere.material.w != 61.0
            && sphere.material.w != 62.0 && sphere.material.w != 63.0
            && sphere.material.w != 64.0 && sphere.material.w != 65.0
            && (sphere.material.w < 66.0 || sphere.material.w > 71.0)) {
            pose = poses[sphere.indices.x];
        }
        var normal = sphere.plane.xyz;
        let center = pose.position.xyz
            + quat_rotate(pose.orientation, sphere.center_radius.xyz);
        let center_of_mass = pose.position.xyz
            + quat_rotate(pose.orientation, sphere.center_of_mass.xyz);
        var contact_offset = center - normal * sphere.center_radius.w - center_of_mass;
        var second_offset = vec3<f32>(0.0);
        var distance = dot(normal, center) - sphere.plane.w - sphere.center_radius.w;
        var contact_enabled = true;
        if ((sphere.material.w == 0.0 || sphere.material.w == 10.0)
            && sphere.other_center_radius.w > 0.0) {
            let extent = sphere.other_center_radius.w + sphere.center_radius.w;
            contact_enabled = abs(center.x) <= extent && abs(center.y) <= extent;
        }
        if (sphere.material.w == 12.0 || sphere.material.w == 13.0) {
            let world_point = axial_ground_point(sphere, pose);
            contact_offset = world_point - center_of_mass;
            distance = dot(normal, world_point) - sphere.plane.w;
            if (sphere.other_center_radius.w > 0.0) {
                let extent = sphere.other_center_radius.w;
                contact_enabled = abs(world_point.x) <= extent
                    && abs(world_point.y) <= extent;
            }
        } else if (sphere.material.w == 5.0) {
            let local_orientation = sphere.second_axis_end;
            let link_normal = quat_rotate(
                vec4<f32>(-pose.orientation.xyz, pose.orientation.w), normal);
            let local_normal = quat_rotate(
                vec4<f32>(-local_orientation.xyz, local_orientation.w), link_normal);
            let half_extents = sphere.other_center_radius.xyz;
            let first_sign = select(-1.0, 1.0, (sphere.indices.z & 1u) != 0u);
            let second_sign = select(-1.0, 1.0, (sphere.indices.z & 2u) != 0u);
            var corner = vec3<f32>(0.0);
            if (abs(local_normal.x) >= abs(local_normal.y)
                && abs(local_normal.x) >= abs(local_normal.z)) {
                corner = vec3<f32>(select(1.0, -1.0, local_normal.x >= 0.0),
                    first_sign, second_sign) * half_extents;
            } else if (abs(local_normal.y) >= abs(local_normal.z)) {
                corner = vec3<f32>(first_sign,
                    select(1.0, -1.0, local_normal.y >= 0.0), second_sign) * half_extents;
            } else {
                corner = vec3<f32>(first_sign, second_sign,
                    select(1.0, -1.0, local_normal.z >= 0.0)) * half_extents;
            }
            let local_point = sphere.center_radius.xyz + quat_rotate(local_orientation, corner);
            let world_point = pose.position.xyz + quat_rotate(pose.orientation, local_point);
            contact_offset = world_point - center_of_mass;
            distance = dot(normal, world_point) - sphere.plane.w;
            if (sphere.other_center_radius.w > 0.0) {
                let extent = sphere.other_center_radius.w;
                contact_enabled = abs(world_point.x) <= extent
                    && abs(world_point.y) <= extent;
            }
        } else if (sphere.material.w == 16.0) {
            let local_orientation = sphere.second_axis_end;
            let link_local = quat_rotate(
                vec4<f32>(-pose.orientation.xyz, pose.orientation.w),
                sphere.plane.xyz - center);
            let local = quat_rotate(
                vec4<f32>(-local_orientation.xyz, local_orientation.w), link_local);
            let half_extents = sphere.other_center_radius.xyz;
            var closest = clamp(local, -half_extents, half_extents);
            let delta = local - closest;
            let surface_distance = length(delta);
            var local_normal = vec3<f32>(0.0);
            if (surface_distance > 1e-7) {
                local_normal = delta / surface_distance;
                distance = surface_distance - sphere.center_radius.w;
            } else {
                let gaps = half_extents - abs(local);
                var axis = 0u;
                if (gaps.y < gaps.x) { axis = 1u; }
                if (gaps.z < gaps[axis]) { axis = 2u; }
                let side = select(-1.0, 1.0, local[axis] >= 0.0);
                if (axis == 0u) {
                    closest.x = side * half_extents.x;
                    local_normal = vec3<f32>(side, 0.0, 0.0);
                } else if (axis == 1u) {
                    closest.y = side * half_extents.y;
                    local_normal = vec3<f32>(0.0, side, 0.0);
                } else {
                    closest.z = side * half_extents.z;
                    local_normal = vec3<f32>(0.0, 0.0, side);
                }
                distance = -sphere.center_radius.w - gaps[axis];
            }
            normal = quat_rotate(pose.orientation,
                quat_rotate(local_orientation, local_normal));
            let box_point = center + quat_rotate(pose.orientation,
                quat_rotate(local_orientation, closest));
            contact_offset = box_point - center_of_mass;
        } else if (sphere.material.w == 20.0) {
            let box_orientation = sphere.second_axis_end;
            let local = quat_rotate(
                vec4<f32>(-box_orientation.xyz, box_orientation.w),
                center - sphere.plane.xyz);
            let half_extents = sphere.other_center_radius.xyz;
            let delta = local - clamp(local, -half_extents, half_extents);
            let surface_distance = length(delta);
            var local_normal = vec3<f32>(0.0);
            var inside = false;
            if (surface_distance > 1e-7) {
                local_normal = -delta / surface_distance;
                distance = surface_distance - sphere.center_radius.w;
            } else {
                let gaps = half_extents - abs(local);
                var axis = 0u;
                if (gaps.y < gaps.x) { axis = 1u; }
                if (gaps.z < gaps[axis]) { axis = 2u; }
                let side = select(-1.0, 1.0, local[axis] >= 0.0);
                if (axis == 0u) {
                    local_normal = vec3<f32>(-side, 0.0, 0.0);
                } else if (axis == 1u) {
                    local_normal = vec3<f32>(0.0, -side, 0.0);
                } else {
                    local_normal = vec3<f32>(0.0, 0.0, -side);
                }
                distance = -sphere.center_radius.w - gaps[axis];
                inside = true;
            }
            normal = quat_rotate(box_orientation, local_normal);
            let first_sign = select(1.0, -1.0, inside);
            contact_offset = center + first_sign * normal * sphere.center_radius.w
                - center_of_mass;
        } else if (sphere.material.w == 6.0 || sphere.material.w == 7.0
            || sphere.material.w == 9.0 || sphere.material.w == 21.0
            || sphere.material.w == 22.0 || sphere.material.w == 24.0
            || sphere.material.w == 25.0) {
            let static_box = sphere.material.w == 21.0 || sphere.material.w == 22.0;
            let static_capsule = sphere.material.w == 24.0 || sphere.material.w == 25.0;
            let local_orientation = sphere.second_axis_end;
            let capsule_center = select(center, sphere.center_radius.xyz, static_capsule);
            var box_center = sphere.plane.xyz;
            var link_local = capsule_center - box_center;
            if (static_capsule) {
                box_center = pose.position.xyz
                    + quat_rotate(pose.orientation, sphere.plane.xyz);
                link_local = quat_rotate(
                    vec4<f32>(-pose.orientation.xyz, pose.orientation.w),
                    capsule_center - box_center);
            } else if (!static_box) {
                let second_pose = poses[sphere.indices.z];
                box_center = second_pose.position.xyz
                    + quat_rotate(second_pose.orientation, sphere.plane.xyz);
                link_local = quat_rotate(
                    vec4<f32>(-second_pose.orientation.xyz, second_pose.orientation.w),
                    capsule_center - box_center);
            }
            var local = quat_rotate(
                vec4<f32>(-local_orientation.xyz, local_orientation.w), link_local);
            let half_extents = sphere.other_center_radius.xyz;
            var contact_center = capsule_center;
            var local_axis = vec3<f32>(0.0);
            var world_axis = vec3<f32>(0.0);
            let local_start = local;
            if (sphere.material.w == 7.0 || sphere.material.w == 9.0
                || static_box || static_capsule) {
                var endpoint = sphere.first_axis_end.xyz;
                if (!static_capsule) {
                    endpoint = pose.position.xyz
                        + quat_rotate(pose.orientation, sphere.first_axis_end.xyz);
                }
                var endpoint_link_local = endpoint - box_center;
                if (static_capsule) {
                    endpoint_link_local = quat_rotate(
                        vec4<f32>(-pose.orientation.xyz, pose.orientation.w),
                        endpoint - box_center);
                } else if (!static_box) {
                    let second_pose = poses[sphere.indices.z];
                    endpoint_link_local = quat_rotate(
                        vec4<f32>(-second_pose.orientation.xyz, second_pose.orientation.w),
                        endpoint - box_center);
                }
                let local_end = quat_rotate(
                    vec4<f32>(-local_orientation.xyz, local_orientation.w),
                    endpoint_link_local);
                local_axis = local_end - local;
                world_axis = endpoint - capsule_center;
                var lower = 0.0;
                var upper = 1.0;
                var fraction = 0.5;
                for (var search = 0u; search < 24u; search++) {
                    fraction = 0.5 * (lower + upper);
                    let point = local + local_axis * fraction;
                    let residual = point - clamp(point, -half_extents, half_extents);
                    let derivative = dot(residual, local_axis);
                    if (derivative == 0.0) { break; }
                    if (derivative < 0.0) { lower = fraction; }
                    else { upper = fraction; }
                }
                local += local_axis * fraction;
                contact_center += world_axis * fraction;
            }
            var closest = clamp(local, -half_extents, half_extents);
            let delta = local - closest;
            let surface_distance = length(delta);
            var local_normal = vec3<f32>(0.0);
            var inside = false;
            if (surface_distance > 1e-7) {
                local_normal = -delta / surface_distance;
                distance = surface_distance - sphere.center_radius.w;
            } else {
                let gaps = half_extents - abs(local);
                var axis = 0u;
                if (gaps.y < gaps.x) { axis = 1u; }
                if (gaps.z < gaps[axis]) { axis = 2u; }
                let side = select(-1.0, 1.0, local[axis] >= 0.0);
                if (axis == 0u) {
                    local_normal = vec3<f32>(-side, 0.0, 0.0);
                    closest.x = side * half_extents.x;
                } else if (axis == 1u) {
                    local_normal = vec3<f32>(0.0, -side, 0.0);
                    closest.y = side * half_extents.y;
                } else {
                    local_normal = vec3<f32>(0.0, 0.0, -side);
                    closest.z = side * half_extents.z;
                }
                distance = -sphere.center_radius.w - gaps[axis];
                inside = true;
            }
            if (sphere.material.w == 7.0 || sphere.material.w == 9.0
                || static_box || static_capsule) {
                var side_contact = !inside && surface_distance > 1e-7;
                let axis_length = length(local_axis);
                var face_axis = 0u;
                if (abs(local_normal.y) > abs(local_normal.x)) { face_axis = 1u; }
                if (abs(local_normal.z) > abs(local_normal[face_axis])) { face_axis = 2u; }
                side_contact = side_contact && abs(local_normal[face_axis]) >= 0.999
                    && axis_length > 1e-6
                    && abs(dot(local_axis, local_normal)) <= 0.05 * axis_length;
                var low = 0.0;
                var high = 1.0;
                if (side_contact) {
                    for (var axis = 0u; axis < 3u; axis++) {
                        if (axis == face_axis) { continue; }
                        if (abs(local_axis[axis]) < 1e-7) {
                            if (abs(local_start[axis]) > half_extents[axis]) {
                                side_contact = false;
                            }
                        } else {
                            let first = (-half_extents[axis] - local_start[axis])
                                / local_axis[axis];
                            let second = (half_extents[axis] - local_start[axis])
                                / local_axis[axis];
                            low = max(low, min(first, second));
                            high = min(high, max(first, second));
                        }
                    }
                    side_contact = side_contact && (high - low) * axis_length
                        >= max(0.25 * sphere.center_radius.w, 1e-3);
                }
                if (side_contact) {
                    let fraction = select(low, high,
                        sphere.material.w == 9.0 || sphere.material.w == 22.0
                            || sphere.material.w == 25.0);
                    let selected = local_start + local_axis * fraction;
                    let selected_closest = clamp(selected, -half_extents, half_extents);
                    let selected_delta = selected - selected_closest;
                    let selected_length = length(selected_delta);
                    side_contact = selected_length > 1e-7
                        && dot(-selected_delta / max(selected_length, 1e-7), local_normal) > 0.98;
                    if (side_contact) {
                        contact_center = capsule_center + world_axis * fraction;
                        closest = selected_closest;
                        local_normal = -selected_delta / selected_length;
                        distance = selected_length - sphere.center_radius.w;
                    }
                }
                if ((sphere.material.w == 9.0 || sphere.material.w == 22.0
                    || sphere.material.w == 25.0)
                    && !side_contact) {
                    contact_enabled = false;
                }
            }
            normal = quat_rotate(local_orientation, local_normal);
            if (static_capsule) {
                normal = quat_rotate(pose.orientation, normal);
            } else if (!static_box) {
                let second_pose = poses[sphere.indices.z];
                normal = quat_rotate(second_pose.orientation, normal);
            }
            let first_sign = select(1.0, -1.0, inside);
            contact_offset = contact_center + first_sign * normal * sphere.center_radius.w
                - center_of_mass;
            if (static_capsule) {
                let box_point = box_center + quat_rotate(pose.orientation,
                    quat_rotate(local_orientation, closest));
                contact_offset = box_point - center_of_mass;
            } else if (!static_box) {
                let second_pose = poses[sphere.indices.z];
                let box_point = box_center + quat_rotate(second_pose.orientation,
                    quat_rotate(local_orientation, closest));
                let second_center_of_mass = second_pose.position.xyz
                    + quat_rotate(second_pose.orientation, sphere.other_center_of_mass.xyz);
                second_offset = box_point - second_center_of_mass;
            }
        } else if (sphere.material.w == 8.0 || sphere.material.w == 23.0) {
            let static_box = sphere.material.w == 23.0;
            var second_pose = Pose(vec4<f32>(0.0, 0.0, 0.0, 0.0),
                vec4<f32>(0.0, 0.0, 0.0, 1.0));
            if (!static_box) { second_pose = poses[sphere.indices.z]; }
            let geometry = box_box_contact_geometry(sphere, pose, second_pose,
                u32(sphere.other_center_radius.w));
            contact_enabled = geometry.enabled;
            normal = geometry.normal;
            distance = geometry.distance;
            contact_offset = geometry.point - center_of_mass;
            if (!static_box) {
                let second_center_of_mass = second_pose.position.xyz
                    + quat_rotate(second_pose.orientation, sphere.other_center_of_mass.xyz);
                second_offset = geometry.point - second_center_of_mass;
            }
        } else if (sphere.material.w == 26.0 || sphere.material.w == 27.0) {
            let geometry = static_axial_sphere_geometry(sphere, pose, sphere.plane.xyz);
            normal = geometry.normal;
            distance = geometry.distance;
            contact_offset = geometry.point - center_of_mass;
        } else if (sphere.material.w == 60.0 || sphere.material.w == 61.0) {
            let second_pose = poses[sphere.indices.z];
            let sphere_center = second_pose.position.xyz
                + quat_rotate(second_pose.orientation, sphere.plane.xyz);
            let geometry = static_axial_sphere_geometry(sphere, pose, sphere_center);
            normal = geometry.normal;
            distance = geometry.distance;
            let second_center_of_mass = second_pose.position.xyz
                + quat_rotate(second_pose.orientation, sphere.other_center_of_mass.xyz);
            second_offset = geometry.point - second_center_of_mass;
        } else if (sphere.material.w == 28.0 || sphere.material.w == 29.0) {
            let geometry = static_axial_capsule_geometry(sphere, pose,
                sphere.plane.xyz, sphere.first_axis_end.xyz);
            normal = geometry.normal;
            distance = geometry.distance;
            contact_offset = geometry.point - center_of_mass;
        } else if (sphere.material.w == 62.0 || sphere.material.w == 63.0) {
            let second_pose = poses[sphere.indices.z];
            let start = second_pose.position.xyz
                + quat_rotate(second_pose.orientation, sphere.plane.xyz);
            let end = second_pose.position.xyz
                + quat_rotate(second_pose.orientation, sphere.first_axis_end.xyz);
            let geometry = static_axial_capsule_geometry(sphere, pose, start, end);
            normal = geometry.normal;
            distance = geometry.distance;
            let second_center_of_mass = second_pose.position.xyz
                + quat_rotate(second_pose.orientation, sphere.other_center_of_mass.xyz);
            second_offset = geometry.point - second_center_of_mass;
        } else if (sphere.material.w == 36.0 || sphere.material.w == 37.0) {
            let second_pose = poses[sphere.indices.z];
            let start = second_pose.position.xyz
                + quat_rotate(second_pose.orientation, sphere.plane.xyz);
            let end = second_pose.position.xyz
                + quat_rotate(second_pose.orientation, sphere.first_axis_end.xyz);
            let geometry = static_axial_capsule_geometry(sphere, pose, start, end);
            normal = geometry.normal;
            distance = geometry.distance;
            contact_offset = geometry.point - center_of_mass;
            let second_center_of_mass = second_pose.position.xyz
                + quat_rotate(second_pose.orientation, sphere.other_center_of_mass.xyz);
            second_offset = geometry.point - second_center_of_mass;
        } else if ((sphere.material.w >= 38.0 && sphere.material.w <= 41.0)
            || (sphere.material.w >= 66.0 && sphere.material.w <= 69.0)) {
            let second_pose = poses[sphere.indices.z];
            let geometry = axial_pair_geometry(sphere, pose, second_pose);
            contact_enabled = geometry.enabled;
            normal = geometry.normal;
            distance = geometry.distance;
            contact_offset = geometry.point - center_of_mass;
            let second_center_of_mass = second_pose.position.xyz
                + quat_rotate(second_pose.orientation, sphere.other_center_of_mass.xyz);
            second_offset = geometry.point - second_center_of_mass;
        } else if (sphere.material.w == 44.0 || sphere.material.w == 47.0) {
            var second_pose = Pose(vec4<f32>(0.0), vec4<f32>(0.0, 0.0, 0.0, 1.0));
            if (sphere.material.w == 44.0) { second_pose = poses[sphere.indices.z]; }
            let geometry = contact_hull_pair_geometry(sphere, pose,
                second_pose, u32(sphere.other_center_radius.w));
            contact_enabled = geometry.enabled;
            normal = geometry.normal;
            distance = geometry.distance;
            contact_offset = geometry.point - center_of_mass;
            if (sphere.material.w == 44.0) {
                let second_center_of_mass = second_pose.position.xyz
                    + quat_rotate(second_pose.orientation, sphere.other_center_of_mass.xyz);
                second_offset = geometry.point - second_center_of_mass;
            }
        } else if (sphere.material.w == 42.0 || sphere.material.w == 43.0 || sphere.material.w == 82.0
            || sphere.material.w == 45.0 || (sphere.material.w == 46.0 || sphere.material.w == 81.0)
            || sphere.material.w == 48.0 || sphere.material.w == 49.0 || sphere.material.w == 83.0) {
            var second_pose = Pose(vec4<f32>(0.0), vec4<f32>(0.0, 0.0, 0.0, 1.0));
            if (sphere.material.w != 45.0 && (sphere.material.w != 46.0 && sphere.material.w != 81.0)) {
                second_pose = poses[sphere.indices.z];
            }
            let geometry = convex_rounded_geometry(sphere, pose, second_pose);
            contact_enabled = geometry.enabled;
            normal = geometry.normal;
            distance = geometry.distance;
            contact_offset = geometry.point - center_of_mass;
            if (sphere.material.w != 45.0 && (sphere.material.w != 46.0 && sphere.material.w != 81.0)) {
                let second_center_of_mass = second_pose.position.xyz
                    + quat_rotate(second_pose.orientation, sphere.other_center_of_mass.xyz);
                second_offset = geometry.point - second_center_of_mass;
            }
        } else if (sphere.material.w == 50.0 || sphere.material.w == 51.0
            || sphere.material.w == 58.0 || sphere.material.w == 59.0
            || sphere.material.w == 70.0 || sphere.material.w == 71.0) {
            let moving_hull = sphere.material.w == 58.0 || sphere.material.w == 59.0
                || sphere.material.w == 70.0 || sphere.material.w == 71.0;
            var second_pose = Pose(vec4<f32>(0.0), vec4<f32>(0.0, 0.0, 0.0, 1.0));
            if (moving_hull) { second_pose = poses[sphere.indices.z]; }
            let geometry = axial_hull_geometry(sphere, pose, second_pose);
            contact_enabled = geometry.enabled;
            normal = geometry.normal;
            distance = geometry.distance;
            contact_offset = geometry.point - center_of_mass;
            if (moving_hull) {
                let second_center_of_mass = second_pose.position.xyz
                    + quat_rotate(second_pose.orientation, sphere.other_center_of_mass.xyz);
                second_offset = geometry.point - second_center_of_mass;
            }
        } else if (sphere.material.w == 52.0 || sphere.material.w == 72.0) {
            if (iteration == 0u) {
                var geometry = BoxContactGeometry(false,
                    vec3<f32>(1.0, 0.0, 0.0), center, 1.0);
                if (sphere.material.w == 52.0) {
                    geometry = scene_mesh_sphere_geometry(sphere, pose);
                } else {
                    geometry = scene_polyline_sphere_geometry(sphere, pose);
                }
                contact_enabled = geometry.enabled;
                normal = geometry.normal;
                distance = geometry.distance;
                contact_offset = geometry.point - center_of_mass;
                spheres[sphere_index].other_center_of_mass = vec4<f32>(
                    geometry.normal, select(1.0, geometry.distance, geometry.enabled));
                spheres[sphere_index].second_axis_end = vec4<f32>(
                    geometry.point, sphere.second_axis_end.w);
            } else {
                contact_enabled = sphere.previous_normal.w != 0.0;
                normal = sphere.other_center_of_mass.xyz;
                distance = sphere.other_center_of_mass.w;
                contact_offset = sphere.second_axis_end.xyz - center_of_mass;
            }
        } else if (sphere.material.w == 53.0 || sphere.material.w == 73.0) {
            if (iteration == 0u) {
                var geometry = BoxContactGeometry(false,
                    vec3<f32>(1.0, 0.0, 0.0), center, 1.0);
                if (sphere.material.w == 53.0) {
                    geometry = scene_mesh_capsule_geometry(sphere, pose);
                } else {
                    geometry = scene_polyline_capsule_geometry(sphere, pose);
                }
                contact_enabled = geometry.enabled;
                normal = geometry.normal;
                distance = geometry.distance;
                contact_offset = geometry.point - center_of_mass;
                spheres[sphere_index].other_center_of_mass = vec4<f32>(
                    geometry.point, select(1.0, geometry.distance, geometry.enabled));
            } else {
                contact_enabled = sphere.previous_normal.w != 0.0;
                normal = sphere.previous_normal.xyz;
                distance = sphere.other_center_of_mass.w;
                contact_offset = sphere.other_center_of_mass.xyz - center_of_mass;
            }
        } else if (sphere.material.w == 54.0 || sphere.material.w == 74.0) {
            if (iteration == 0u) {
                var geometry = BoxContactGeometry(false,
                    vec3<f32>(1.0, 0.0, 0.0), center, 1.0);
                if (sphere.material.w == 54.0) {
                    geometry = scene_mesh_box_geometry(sphere, pose);
                } else {
                    geometry = scene_polyline_box_geometry(sphere, pose);
                }
                contact_enabled = geometry.enabled;
                normal = geometry.normal;
                distance = geometry.distance;
                contact_offset = geometry.point - center_of_mass;
                // other_center_of_mass.w stores the box's z half extent; do not overwrite it.
                spheres[sphere_index].diagnostic_first = vec4<f32>(
                    geometry.point, select(1.0, geometry.distance, geometry.enabled));
            } else {
                contact_enabled = sphere.previous_normal.w != 0.0;
                normal = sphere.previous_normal.xyz;
                distance = sphere.diagnostic_first.w;
                contact_offset = sphere.diagnostic_first.xyz - center_of_mass;
            }
        } else if (sphere.material.w == 55.0 || sphere.material.w == 56.0
            || sphere.material.w == 75.0 || sphere.material.w == 76.0) {
            if (iteration == 0u) {
                var geometry = BoxContactGeometry(false,
                    vec3<f32>(1.0, 0.0, 0.0), center, 1.0);
                if (sphere.material.w == 55.0 || sphere.material.w == 56.0) {
                    geometry = scene_mesh_axial_geometry(sphere, pose);
                } else {
                    geometry = scene_polyline_axial_geometry(sphere, pose);
                }
                contact_enabled = geometry.enabled;
                normal = geometry.normal;
                distance = geometry.distance;
                contact_offset = geometry.point - center_of_mass;
                spheres[sphere_index].other_center_of_mass = vec4<f32>(
                    geometry.point, select(1.0, geometry.distance, geometry.enabled));
            } else {
                contact_enabled = sphere.previous_normal.w != 0.0;
                normal = sphere.previous_normal.xyz;
                distance = sphere.other_center_of_mass.w;
                contact_offset = sphere.other_center_of_mass.xyz - center_of_mass;
            }
        } else if (sphere.material.w == 57.0 || sphere.material.w == 77.0) {
            if (iteration == 0u && sphere.plane.w == 0.0) {
                // Contact rows are contiguous and processed in slot order.
                var manifold: SceneMeshConvexManifold;
                if (sphere.material.w == 57.0) {
                    manifold = scene_mesh_convex_manifold(sphere, pose);
                } else {
                    manifold = scene_polyline_convex_manifold(sphere, pose);
                }
                cache_scene_mesh_convex_contact(sphere_index, manifold.first, true);
                cache_scene_mesh_convex_contact(sphere_index + 1u, manifold.second, false);
                cache_scene_mesh_convex_contact(sphere_index + 2u, manifold.third, false);
                cache_scene_mesh_convex_contact(sphere_index + 3u, manifold.fourth, false);
            }
            let cached = spheres[sphere_index];
            contact_enabled = cached.previous_normal.w != 0.0;
            normal = select(cached.other_center_radius.xyz,
                cached.previous_normal.xyz, sphere.plane.w == 0.0);
            distance = cached.other_center_of_mass.w;
            contact_offset = cached.other_center_of_mass.xyz - center_of_mass;
        } else if (sphere.material.w == 30.0 || sphere.material.w == 31.0
            || sphere.material.w == 34.0 || sphere.material.w == 35.0
            || sphere.material.w == 64.0 || sphere.material.w == 65.0) {
            var box_pose = Pose(vec4<f32>(0.0), vec4<f32>(0.0, 0.0, 0.0, 1.0));
            if (sphere.material.w == 34.0 || sphere.material.w == 35.0
                || sphere.material.w == 64.0 || sphere.material.w == 65.0) {
                box_pose = poses[sphere.indices.z];
            }
            let geometry = static_axial_box_geometry(sphere, pose, box_pose);
            contact_enabled = geometry.enabled;
            normal = geometry.normal;
            distance = geometry.distance;
            contact_offset = geometry.point - center_of_mass;
            if (sphere.material.w == 34.0 || sphere.material.w == 35.0
                || sphere.material.w == 64.0 || sphere.material.w == 65.0) {
                let second_center_of_mass = box_pose.position.xyz
                    + quat_rotate(box_pose.orientation, sphere.other_center_of_mass.xyz);
                second_offset = geometry.point - second_center_of_mass;
            }
        } else if (sphere.material.w == 32.0 || sphere.material.w == 33.0) {
            let second_pose = poses[sphere.indices.z];
            let sphere_center = second_pose.position.xyz
                + quat_rotate(second_pose.orientation, sphere.plane.xyz);
            let geometry = static_axial_sphere_geometry(sphere, pose, sphere_center);
            normal = geometry.normal;
            distance = geometry.distance;
            contact_offset = geometry.point - center_of_mass;
            let second_center_of_mass = second_pose.position.xyz
                + quat_rotate(second_pose.orientation, sphere.other_center_of_mass.xyz);
            second_offset = geometry.point - second_center_of_mass;
        } else if (sphere.material.w == 1.0 || sphere.material.w == 14.0) {
            var second_center = sphere.other_center_radius.xyz;
            if (sphere.material.w == 1.0) {
                let second_pose = poses[sphere.indices.z];
                second_center = second_pose.position.xyz
                    + quat_rotate(second_pose.orientation, sphere.other_center_radius.xyz);
            }
            let separation = second_center - center;
            let separation_length = length(separation);
            normal = vec3<f32>(1.0, 0.0, 0.0);
            if (separation_length > 1e-7) { normal = separation / separation_length; }
            distance = separation_length - sphere.center_radius.w
                - sphere.other_center_radius.w;
            contact_offset = center + normal * sphere.center_radius.w - center_of_mass;
            if (sphere.material.w == 1.0) {
                let second_pose = poses[sphere.indices.z];
                let second_center_of_mass = second_pose.position.xyz
                    + quat_rotate(second_pose.orientation, sphere.other_center_of_mass.xyz);
                second_offset = second_center - normal * sphere.other_center_radius.w
                    - second_center_of_mass;
            }
        } else if (sphere.material.w == 17.0) {
            let axis = sphere.other_center_radius.xyz - sphere.plane.xyz;
            let axis_squared = dot(axis, axis);
            var fraction = 0.0;
            if (axis_squared > 1e-12) {
                fraction = clamp(dot(center - sphere.plane.xyz, axis) / axis_squared,
                    0.0, 1.0);
            }
            let nearest = sphere.plane.xyz + axis * fraction;
            let separation = nearest - center;
            let separation_length = length(separation);
            normal = vec3<f32>(1.0, 0.0, 0.0);
            if (separation_length > 1e-7) { normal = separation / separation_length; }
            distance = separation_length - sphere.center_radius.w - sphere.plane.w;
            contact_offset = center + normal * sphere.center_radius.w - center_of_mass;
        } else if (sphere.material.w == 2.0 || sphere.material.w == 15.0) {
            let endpoint_b = pose.position.xyz
                + quat_rotate(pose.orientation,
                    select(sphere.other_center_radius.xyz, sphere.first_axis_end.xyz,
                        sphere.material.w == 15.0));
            var sphere_center = sphere.other_center_radius.xyz;
            if (sphere.material.w == 2.0) {
                let second_pose = poses[sphere.indices.z];
                sphere_center = second_pose.position.xyz
                    + quat_rotate(second_pose.orientation, sphere.plane.xyz);
            }
            let axis = endpoint_b - center;
            let axis_squared = dot(axis, axis);
            var fraction = 0.0;
            if (axis_squared > 1e-12) {
                fraction = clamp(dot(sphere_center - center, axis) / axis_squared, 0.0, 1.0);
            }
            let nearest = center + fraction * axis;
            let separation = sphere_center - nearest;
            let separation_length = length(separation);
            normal = vec3<f32>(1.0, 0.0, 0.0);
            if (separation_length > 1e-7) { normal = separation / separation_length; }
            let second_radius = select(sphere.plane.w, sphere.other_center_radius.w,
                sphere.material.w == 15.0);
            distance = separation_length - sphere.center_radius.w - second_radius;
            contact_offset = nearest + normal * sphere.center_radius.w - center_of_mass;
            if (sphere.material.w == 2.0) {
                let second_pose = poses[sphere.indices.z];
                let second_center_of_mass = second_pose.position.xyz
                    + quat_rotate(second_pose.orientation, sphere.other_center_of_mass.xyz);
                second_offset = sphere_center - normal * second_radius - second_center_of_mass;
            }
        } else if (sphere.material.w == 3.0 || sphere.material.w == 4.0
            || sphere.material.w == 18.0 || sphere.material.w == 19.0) {
            let first_end = pose.position.xyz
                + quat_rotate(pose.orientation, sphere.other_center_radius.xyz);
            var second_start = sphere.plane.xyz;
            var second_end = sphere.second_axis_end.xyz;
            if (sphere.material.w == 3.0 || sphere.material.w == 4.0) {
                let second_pose = poses[sphere.indices.z];
                second_start = second_pose.position.xyz
                    + quat_rotate(second_pose.orientation, sphere.plane.xyz);
                second_end = second_pose.position.xyz
                    + quat_rotate(second_pose.orientation, sphere.second_axis_end.xyz);
            }
            let first_axis = first_end - center;
            let second_axis = second_end - second_start;
            let offset = center - second_start;
            let aa = dot(first_axis, first_axis);
            let bb = dot(first_axis, second_axis);
            let cc = dot(second_axis, second_axis);
            let dd = dot(first_axis, offset);
            let ee = dot(second_axis, offset);
            let denominator = aa * cc - bb * bb;
            var s = 0.0;
            if (aa > 1e-12 && denominator > 1e-12 * aa * cc) {
                s = clamp((bb * ee - cc * dd) / denominator, 0.0, 1.0);
            }
            var t = 0.0;
            if (cc > 1e-12) { t = clamp((bb * s + ee) / cc, 0.0, 1.0); }
            if (aa > 1e-12) { s = clamp((bb * t - dd) / aa, 0.0, 1.0); }
            if (cc > 1e-12) { t = clamp((bb * s + ee) / cc, 0.0, 1.0); }
            var first_nearest = center + first_axis * s;
            var second_nearest = second_start + second_axis * t;
            let first_length = sqrt(aa);
            let second_length = sqrt(cc);
            var side_contact = false;
            if (first_length > 1e-6 && second_length > 1e-6) {
                let first_direction = first_axis / first_length;
                let second_direction = second_axis / second_length;
                let initial_separation = second_nearest - first_nearest;
                let initial_length = length(initial_separation);
                var transverse = true;
                if (initial_length > 1e-7) {
                    transverse = abs(dot(first_direction, initial_separation / initial_length)) <= 0.05;
                }
                let first_projection = dot(second_start - center, first_direction);
                let second_projection = dot(second_end - center, first_direction);
                let low = max(0.0, min(first_projection, second_projection));
                let high = min(first_length, max(first_projection, second_projection));
                side_contact = abs(dot(first_direction, second_direction)) >= 0.999
                    && transverse
                    && high - low >= max(0.25 * min(sphere.center_radius.w, sphere.plane.w), 1e-3);
                if (side_contact) {
                    let first_distance = select(low, high,
                        sphere.material.w == 4.0 || sphere.material.w == 19.0);
                    first_nearest = center + first_direction * first_distance;
                    let second_fraction = clamp(dot(first_nearest - second_start, second_axis) / cc,
                        0.0, 1.0);
                    second_nearest = second_start + second_axis * second_fraction;
                }
            }
            if ((sphere.material.w == 4.0 || sphere.material.w == 19.0)
                && !side_contact) { contact_enabled = false; }
            let separation = second_nearest - first_nearest;
            let separation_length = length(separation);
            normal = vec3<f32>(1.0, 0.0, 0.0);
            if (separation_length > 1e-7) { normal = separation / separation_length; }
            distance = separation_length - sphere.center_radius.w - sphere.plane.w;
            contact_offset = first_nearest + normal * sphere.center_radius.w - center_of_mass;
            if (sphere.material.w == 3.0 || sphere.material.w == 4.0) {
                let second_pose = poses[sphere.indices.z];
                let second_center_of_mass = second_pose.position.xyz
                    + quat_rotate(second_pose.orientation, sphere.other_center_of_mass.xyz);
                second_offset = second_nearest - normal * sphere.plane.w - second_center_of_mass;
            }
        }
        if (!contact_enabled) {
            spheres[sphere_index].impulses = vec4<f32>(0.0);
            spheres[sphere_index].previous_normal = vec4<f32>(0.0);
            spheres[sphere_index].diagnostic_first_origin = vec4<f32>(0.0);
            spheres[sphere_index].diagnostic_second_origin = vec4<f32>(0.0);
            continue;
        }
        let owner_mode = contact_owner_mode(sphere.material.w);
        if (owner_mode == 3u) {
            // A shared application point prevents penetration from adding a friction couple.
            let other_pose = poses[sphere.indices.z];
            let other_com = other_pose.position.xyz + quat_rotate(other_pose.orientation, sphere.other_center_of_mass.xyz);
            let common_point = 0.5 * (center_of_mass + contact_offset + other_com + second_offset);
            contact_offset = common_point - center_of_mass;
            second_offset = common_point - other_com;
        }
        let first_sign = select(-1.0, 1.0, owner_mode == 0u);
        spheres[sphere_index].diagnostic_first = vec4<f32>(center_of_mass + contact_offset, distance);
        spheres[sphere_index].diagnostic_first_origin = vec4<f32>(pose.position.xyz, select(first_sign, 0.0, owner_mode == 2u));
        spheres[sphere_index].diagnostic_second = vec4<f32>(0.0);
        spheres[sphere_index].diagnostic_second_origin = vec4<f32>(0.0);
        if (owner_mode == 2u || owner_mode == 3u) {
            let other_pose = poses[sphere.indices.z];
            let other_com = other_pose.position.xyz + quat_rotate(other_pose.orientation, sphere.other_center_of_mass.xyz);
            spheres[sphere_index].diagnostic_second = vec4<f32>(other_com + second_offset, distance);
            spheres[sphere_index].diagnostic_second_origin = vec4<f32>(other_pose.position.xyz, 1.0);
        }
        var reference = vec3<f32>(0.0, 0.0, 1.0);
        if (abs(normal.z) >= 0.9) { reference = vec3<f32>(1.0, 0.0, 0.0); }
        let tangent_one = normalize(cross(normal, reference));
        let tangent_two = cross(normal, tangent_one);
        let dt = sphere.material.y;
        var prescribed_velocity = vec3<f32>(0.0);
        if (sphere.material.w == 14.0 || sphere.material.w == 15.0 || sphere.material.w == 16.0
            || sphere.material.w == 20.0 || sphere.material.w == 21.0 || sphere.material.w == 22.0
            || sphere.material.w == 17.0 || sphere.material.w == 18.0 || sphere.material.w == 19.0 || sphere.material.w == 24.0 || sphere.material.w == 25.0 || sphere.material.w == 28.0 || sphere.material.w == 29.0 || (sphere.material.w == 46.0 || sphere.material.w == 81.0)
            || sphere.material.w == 23.0 || sphere.material.w == 30.0 || sphere.material.w == 31.0 || sphere.material.w == 47.0 || sphere.material.w == 26.0 || sphere.material.w == 27.0 || sphere.material.w == 45.0 || sphere.material.w == 50.0 || sphere.material.w == 51.0) {
            let point = center_of_mass + contact_offset;
            var external_center = sphere.other_center_radius.xyz;
            if (sphere.material.w == 16.0 || sphere.material.w == 20.0 || sphere.material.w == 21.0 || sphere.material.w == 22.0
            || sphere.material.w == 23.0 || sphere.material.w == 30.0 || sphere.material.w == 31.0 || sphere.material.w == 47.0 || sphere.material.w == 26.0
                || sphere.material.w == 27.0 || sphere.material.w == 45.0 || sphere.material.w == 50.0 || sphere.material.w == 51.0) {
                external_center = sphere.plane.xyz;
            }
            if (sphere.material.w == 17.0) { external_center = (sphere.plane.xyz + sphere.other_center_radius.xyz) * 0.5; }
            if (sphere.material.w == 18.0 || sphere.material.w == 19.0 || (sphere.material.w == 46.0 || sphere.material.w == 81.0)) { external_center = (sphere.plane.xyz + sphere.second_axis_end.xyz) * 0.5; }
            if (sphere.material.w == 24.0 || sphere.material.w == 25.0) { external_center = (sphere.center_radius.xyz + sphere.first_axis_end.xyz) * 0.5; }
            if (sphere.material.w == 28.0 || sphere.material.w == 29.0) { external_center = (sphere.plane.xyz + sphere.first_axis_end.xyz) * 0.5; }
            prescribed_velocity = sphere.prescribed_linear.xyz
                + cross(sphere.prescribed_angular.xyz, point - external_center);
            // Capsule/box geometry orders the prescribed capsule first.
            if (sphere.material.w == 24.0 || sphere.material.w == 25.0) {
                prescribed_velocity = -prescribed_velocity;
            }
        }
        if ((sphere.material.w >= 60.0 && sphere.material.w <= 71.0)
            || sphere.material.w == 48.0 || sphere.material.w == 49.0 || sphere.material.w == 83.0) {
            // Static-first axial and convex rows store the point on the second owner.
            let other_pose = poses[sphere.indices.z];
            let other_com = other_pose.position.xyz
                + quat_rotate(other_pose.orientation, sphere.other_center_of_mass.xyz);
            let point = other_com + second_offset;
            prescribed_velocity = -(sphere.prescribed_linear.xyz
                + cross(sphere.prescribed_angular.xyz, point - sphere.center_radius.xyz));
        }
        if ((sphere.material.w >= 52.0 && sphere.material.w <= 57.0)
            || (sphere.material.w >= 72.0 && sphere.material.w <= 77.0)) {
            let point = center_of_mass + contact_offset;
            let translated_second = (sphere.material.w >= 53.0 && sphere.material.w <= 56.0)
                || (sphere.material.w >= 73.0 && sphere.material.w <= 76.0);
            let geometry_center = select(sphere.plane.xyz, sphere.second_axis_end.xyz, translated_second);
            prescribed_velocity = sphere.prescribed_linear.xyz
                + cross(sphere.prescribed_angular.xyz, point - geometry_center);
        }
        let prescribed_normal = dot(normal, prescribed_velocity);
        var normal_velocity = prescribed_normal;
        var incoming_velocity = prescribed_normal;
        for (var column = 0u; column < n; column++) {
            let jacobian = contact_jacobian(sphere, n, column, normal, contact_offset, second_offset);
            incoming_velocity += jacobian * velocities[coordinate_offset + column];
            normal_velocity += jacobian * (velocities[coordinate_offset + column]
                + dt * accelerations[coordinate_offset + column]);
        }
        var previous_impulses = sphere.impulses.xyz;
        if (iteration == 0u) {
            if (system.inverse.w != 0u
                && dot(sphere.previous_normal.xyz, normal) > 0.999
                && (distance < 0.0 || distance + dt * normal_velocity < 0.0)
                && all(abs(previous_impulses) < vec3<f32>(1e30))) {
                previous_impulses *= contact_policy.z;
                apply_cached_impulse(system, sphere, normal, tangent_one, tangent_two,
                    contact_offset, second_offset, previous_impulses);
                normal_velocity = prescribed_normal;
                for (var column = 0u; column < n; column++) {
                    let jacobian = contact_jacobian(sphere, n, column, normal,
                        contact_offset, second_offset);
                    normal_velocity += jacobian * (velocities[coordinate_offset + column]
                        + dt * accelerations[coordinate_offset + column]);
                }
            } else {
                previous_impulses = vec3<f32>(0.0);
            }
            spheres[sphere_index].impulses = vec4<f32>(previous_impulses, 0.0);
            // Cache eligibility independently: approaching contacts may have positive distance.
            spheres[sphere_index].previous_normal = vec4<f32>(normal, 1.0);
        }
        if (seed_only) { continue; }
        if (distance >= 0.0 && distance + dt * normal_velocity >= 0.0
            && previous_impulses.x == 0.0) { continue; }
        var effective = 0.0;
        for (var row = 0u; row < n; row++) {
            let jacobian_row = contact_jacobian(sphere, n, row, normal, contact_offset, second_offset);
            var response = 0.0;
            for (var column = 0u; column < n; column++) {
                response += inverse[system.inverse.x + row * n + column]
                    * contact_jacobian(sphere, n, column, normal, contact_offset, second_offset);
            }
            effective += jacobian_row * response;
        }
        if (!(effective > 1e-8) || !(effective < 1e30)) { continue; }
        var correction_distance = distance;
        if ((sphere.material.w >= 52.0 && sphere.material.w <= 57.0)
            || (sphere.material.w >= 72.0 && sphere.material.w <= 77.0)) {
            // Avoid converting sub-micrometer roundoff into a support velocity error.
            // Keep the measured distance for diagnostics and contact approach tests.
            correction_distance = sign(distance) * max(abs(distance) - 1e-6, 0.0);
        }
        var target_velocity = -correction_distance / dt;
        if (distance < 0.0) {
            target_velocity = contact_recovery_velocity(correction_distance, dt);
        }
        if (sphere.material.x > 0.0 && distance <= 0.0) {
            target_velocity = max(target_velocity,
                -sphere.material.x * min(incoming_velocity, 0.0));
        }
        var normal_impulse = max(previous_impulses.x
            + (target_velocity - normal_velocity) / effective, 0.0);
        let impulse = normal_impulse - previous_impulses.x;
        if (!(abs(impulse) < 1e30)) { continue; }
        for (var row = 0u; row < n; row++) {
            var response = 0.0;
            for (var column = 0u; column < n; column++) {
                response += inverse[system.inverse.x + row * n + column]
                    * contact_jacobian(sphere, n, column, normal, contact_offset, second_offset);
            }
            accelerations[coordinate_offset + row] += response * impulse / dt;
        }
        if (sphere.material.w == 81.0 || sphere.material.w == 82.0 || sphere.material.w == 83.0) {
            normal_impulse = resolve_capsule_face_normal_block(system, sphere_index, spheres[sphere_index],
                normal, contact_offset, second_offset, target_velocity, normal_impulse);
        }
        spheres[sphere_index].impulses = vec4<f32>(normal_impulse,
            previous_impulses.y, previous_impulses.z, 0.0);
        if (sphere.material.z <= 0.0) { continue; }
        var tangent_velocity = vec2<f32>(dot(tangent_one, prescribed_velocity),
            dot(tangent_two, prescribed_velocity));
        var k11 = 0.0;
        var k12 = 0.0;
        var k22 = 0.0;
        for (var row = 0u; row < n; row++) {
            let j1 = contact_jacobian(sphere, n, row, tangent_one, contact_offset, second_offset);
            let j2 = contact_jacobian(sphere, n, row, tangent_two, contact_offset, second_offset);
            let predicted = velocities[coordinate_offset + row]
                + dt * accelerations[coordinate_offset + row];
            tangent_velocity += vec2<f32>(j1, j2) * predicted;
            var response_one = 0.0;
            var response_two = 0.0;
            for (var column = 0u; column < n; column++) {
                let mass_inverse = inverse[system.inverse.x + row * n + column];
                response_one += mass_inverse
                    * contact_jacobian(sphere, n, column, tangent_one, contact_offset, second_offset);
                response_two += mass_inverse
                    * contact_jacobian(sphere, n, column, tangent_two, contact_offset, second_offset);
            }
            k11 += j1 * response_one;
            k12 += j1 * response_two;
            k22 += j2 * response_two;
        }
        let determinant = k11 * k22 - k12 * k12;
        let trace = k11 + k22;
        if (!(trace > 1e-8) || !(trace < 1e30)) { continue; }
        var tangent_delta = vec2<f32>(0.0);
        if (determinant > 1e-8 * trace * trace) {
            tangent_delta = vec2<f32>(
                (-k22 * tangent_velocity.x + k12 * tangent_velocity.y) / determinant,
                (k12 * tangent_velocity.x - k11 * tangent_velocity.y) / determinant,
            );
        } else {
            // The attainable tangent space may be one-dimensional.
            tangent_delta = -vec2<f32>(
                k11 * tangent_velocity.x + k12 * tangent_velocity.y,
                k12 * tangent_velocity.x + k22 * tangent_velocity.y,
            ) / (trace * trace);
        }
        var tangent_impulse = previous_impulses.yz + tangent_delta;
        let friction_limit = sphere.material.z * normal_impulse;
        let tangent_magnitude = length(tangent_impulse);
        if (!(tangent_magnitude < 1e30)) { continue; }
        if (tangent_magnitude > friction_limit) {
            tangent_impulse *= friction_limit / tangent_magnitude;
        }
        tangent_delta = tangent_impulse - previous_impulses.yz;
        for (var row = 0u; row < n; row++) {
            var response_one = 0.0;
            var response_two = 0.0;
            for (var column = 0u; column < n; column++) {
                let mass_inverse = inverse[system.inverse.x + row * n + column];
                response_one += mass_inverse
                    * contact_jacobian(sphere, n, column, tangent_one, contact_offset, second_offset);
                response_two += mass_inverse
                    * contact_jacobian(sphere, n, column, tangent_two, contact_offset, second_offset);
            }
            accelerations[coordinate_offset + row] +=
                (response_one * tangent_delta.x + response_two * tangent_delta.y) / dt;
        }
        spheres[sphere_index].impulses = vec4<f32>(normal_impulse, tangent_impulse.x,
            tangent_impulse.y, 0.0);
    }
    }
}
