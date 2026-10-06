struct RigidState {
    position_inverse_mass: vec4<f32>,
    orientation: vec4<f32>,
    linear_velocity: vec4<f32>,
    angular_velocity: vec4<f32>,
    inverse_inertia_sleep: vec4<f32>,
}

struct Shape {
    kind: vec4<u32>,
    feature_counts: vec4<u32>,
    dimensions: vec4<f32>,
}

struct Pair { a: u32, b: u32, }

struct Contact {
    point: vec4<f32>,
    normal: vec4<f32>,
    depth_hit: vec4<f32>,
}

struct Params {
    body_count: u32,
    pair_count: u32,
    ground_half_extent: f32,
    ground_enabled: u32,
    ground_memberships: u32,
    ground_filter: u32,
    padding: vec2<u32>,
}

struct CollisionGroups {
    memberships: u32,
    filter_mask: u32,
}

struct Box {
    center: vec3<f32>,
    axis_x: vec3<f32>,
    axis_y: vec3<f32>,
    axis_z: vec3<f32>,
    half_extents: vec3<f32>,
}

struct Capsule {
    start: vec3<f32>,
    end: vec3<f32>,
    radius: f32,
}

struct AxisResult {
    separated: bool,
    depth: f32,
    normal: vec3<f32>,
}

struct FacePolygon {
    points: array<vec3<f32>, 32>,
    count: u32,
}

struct PolyFace {
    polygon: FacePolygon,
    normal: vec3<f32>,
    alignment: f32,
    valid: bool,
}

// FXC cannot write a function-local struct array through a dynamic index.
fn set_face_point(polygon: ptr<function, FacePolygon>, index: u32, point: vec3<f32>) {
    switch index {
        case 0u: { (*polygon).points[0u] = point; }
        case 1u: { (*polygon).points[1u] = point; }
        case 2u: { (*polygon).points[2u] = point; }
        case 3u: { (*polygon).points[3u] = point; }
        case 4u: { (*polygon).points[4u] = point; }
        case 5u: { (*polygon).points[5u] = point; }
        case 6u: { (*polygon).points[6u] = point; }
        case 7u: { (*polygon).points[7u] = point; }
        case 8u: { (*polygon).points[8u] = point; }
        case 9u: { (*polygon).points[9u] = point; }
        case 10u: { (*polygon).points[10u] = point; }
        case 11u: { (*polygon).points[11u] = point; }
        case 12u: { (*polygon).points[12u] = point; }
        case 13u: { (*polygon).points[13u] = point; }
        case 14u: { (*polygon).points[14u] = point; }
        case 15u: { (*polygon).points[15u] = point; }
        case 16u: { (*polygon).points[16u] = point; }
        case 17u: { (*polygon).points[17u] = point; }
        case 18u: { (*polygon).points[18u] = point; }
        case 19u: { (*polygon).points[19u] = point; }
        case 20u: { (*polygon).points[20u] = point; }
        case 21u: { (*polygon).points[21u] = point; }
        case 22u: { (*polygon).points[22u] = point; }
        case 23u: { (*polygon).points[23u] = point; }
        case 24u: { (*polygon).points[24u] = point; }
        case 25u: { (*polygon).points[25u] = point; }
        case 26u: { (*polygon).points[26u] = point; }
        case 27u: { (*polygon).points[27u] = point; }
        case 28u: { (*polygon).points[28u] = point; }
        case 29u: { (*polygon).points[29u] = point; }
        case 30u: { (*polygon).points[30u] = point; }
        case 31u: { (*polygon).points[31u] = point; }
        default: {}
    }
}

@group(0) @binding(0) var<storage, read> states: array<RigidState>;
@group(0) @binding(1) var<storage, read> shapes: array<Shape>;
@group(0) @binding(2) var<storage, read> pairs: array<Pair>;
@group(0) @binding(3) var<storage, read_write> pair_contacts: array<Contact>;
@group(0) @binding(4) var<storage, read_write> ground_contacts: array<Contact>;
@group(0) @binding(5) var<uniform> params: Params;
@group(0) @binding(6) var<storage, read> convex_vertices: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> convex_edges: array<vec4<u32>>;
@group(0) @binding(8) var<storage, read> collision_groups: array<CollisionGroups>;

override MESH_HULL_MODE: u32 = 0u;

fn allows(a: CollisionGroups, b: CollisionGroups) -> bool {
    return (a.memberships & b.filter_mask) != 0u && (b.memberships & a.filter_mask) != 0u;
}

fn miss() -> Contact {
    return Contact(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
}

fn hit(point: vec3<f32>, normal: vec3<f32>, depth: f32) -> Contact {
    return Contact(
        vec4<f32>(point, 0.0),
        vec4<f32>(normal, 0.0),
        vec4<f32>(max(depth, 0.0), 1.0, 0.0, 0.0),
    );
}

fn rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

fn box_from(index: u32) -> Box {
    let state = states[index];
    return Box(
        state.position_inverse_mass.xyz,
        rotate(state.orientation, vec3<f32>(1.0, 0.0, 0.0)),
        rotate(state.orientation, vec3<f32>(0.0, 1.0, 0.0)),
        rotate(state.orientation, vec3<f32>(0.0, 0.0, 1.0)),
        shapes[index].dimensions.xyz,
    );
}

struct Simplex {
    a: vec3<f32>,
    b: vec3<f32>,
    c: vec3<f32>,
    count: u32,
    direction: vec3<f32>,
    inside: bool,
}

fn capsule_from(index: u32) -> Capsule {
    let state = states[index];
    let axis = rotate(state.orientation, vec3<f32>(0.0, 0.0, 1.0));
    let offset = axis * shapes[index].dimensions.y;
    let center = state.position_inverse_mass.xyz;
    return Capsule(center - offset, center + offset, shapes[index].dimensions.x);
}

fn segment_nearest(a: vec3<f32>, b: vec3<f32>, point: vec3<f32>) -> vec3<f32> {
    let direction = b - a;
    let length_squared = dot(direction, direction);
    if (length_squared < 1e-12) { return a; }
    return a + direction * clamp(dot(point - a, direction) / length_squared, 0.0, 1.0);
}

fn rounded_contact(a: vec3<f32>, radius_a: f32, b: vec3<f32>, radius_b: f32) -> Contact {
    let delta = b - a;
    let distance = length(delta);
    let depth = radius_a + radius_b - distance;
    if (depth < 0.0) { return miss(); }
    var normal = vec3<f32>(1.0, 0.0, 0.0);
    if (distance > 1e-7) { normal = delta / distance; }
    return hit((a + normal * radius_a + b - normal * radius_b) * 0.5, normal, depth);
}

fn sphere_capsule(center: vec3<f32>, radius: f32, capsule: Capsule) -> Contact {
    let core = segment_nearest(capsule.start, capsule.end, center);
    var contact = rounded_contact(core, capsule.radius, center, radius);
    contact.normal = -contact.normal;
    return contact;
}

fn capsule_capsule(a: Capsule, b: Capsule) -> Contact {
    let u = a.end - a.start;
    let v = b.end - b.start;
    let w = a.start - b.start;
    let aa = dot(u, u);
    let bb = dot(u, v);
    let cc = dot(v, v);
    let dd = dot(u, w);
    let ee = dot(v, w);
    let denominator = aa * cc - bb * bb;
    var s = 0.0;
    if (aa > 1e-12 && denominator > 1e-12) {
        s = clamp((bb * ee - cc * dd) / denominator, 0.0, 1.0);
    }
    var t = 0.0;
    if (cc > 1e-12) { t = clamp((bb * s + ee) / cc, 0.0, 1.0); }
    if (aa > 1e-12) { s = clamp((bb * t - dd) / aa, 0.0, 1.0); }
    if (cc > 1e-12) { t = clamp((bb * s + ee) / cc, 0.0, 1.0); }
    return rounded_contact(a.start + u * s, a.radius, b.start + v * t, b.radius);
}

fn linear_side_contacts(index: u32, a: Capsule, b: Capsule, contact: Contact) {
    pair_contacts[index] = contact;
    if (contact.depth_hit.y == 0.0) { return; }
    let segment_a = a.end - a.start;
    let segment_b = b.end - b.start;
    let length_a = length(segment_a);
    let length_b = length(segment_b);
    if (length_a < 1e-6 || length_b < 1e-6) { return; }
    let axis = segment_a / length_a;
    if (abs(dot(axis, segment_b / length_b)) < 0.999 ||
        abs(dot(axis, contact.normal.xyz)) > 0.05) { return; }
    let b_start = dot(b.start - a.start, axis);
    let b_end = dot(b.end - a.start, axis);
    let low = max(0.0, min(b_start, b_end));
    let high = min(length_a, max(b_start, b_end));
    if (high - low < max(0.25 * min(a.radius, b.radius), 1e-3)) { return; }

    let first_core = a.start + axis * low;
    let second_core = a.start + axis * high;
    let first = rounded_contact(first_core, a.radius,
        segment_nearest(b.start, b.end, first_core), b.radius);
    let second = rounded_contact(second_core, a.radius,
        segment_nearest(b.start, b.end, second_core), b.radius);
    if (first.depth_hit.y == 0.0 || second.depth_hit.y == 0.0 ||
        dot(first.normal.xyz, contact.normal.xyz) < 0.98 ||
        dot(second.normal.xyz, contact.normal.xyz) < 0.98 ||
        distance(first.point.xyz, second.point.xyz) < 1e-3) {
        return;
    }
    pair_contacts[index] = first;
    pair_contacts[params.pair_count + index * 3u] = second;
}

fn capsule_capsule_side_contacts(index: u32, a: Capsule, b: Capsule) {
    linear_side_contacts(index, a, b, capsule_capsule(a, b));
}

fn sphere_sphere(a_index: u32, b_index: u32) -> Contact {
    let a = states[a_index].position_inverse_mass.xyz;
    let b = states[b_index].position_inverse_mass.xyz;
    let radius_a = shapes[a_index].dimensions.x;
    let radius_b = shapes[b_index].dimensions.x;
    let delta = b - a;
    let distance = length(delta);
    let depth = radius_a + radius_b - distance;
    if (depth < 0.0) { return miss(); }
    var normal = vec3<f32>(1.0, 0.0, 0.0);
    if (distance > 1e-7) { normal = delta / distance; }
    let point_a = a + normal * radius_a;
    let point_b = b - normal * radius_b;
    return hit((point_a + point_b) * 0.5, normal, depth);
}

fn sphere_box(center: vec3<f32>, radius: f32, shape: Box) -> Contact {
    let offset = center - shape.center;
    let local = vec3<f32>(
        dot(offset, shape.axis_x),
        dot(offset, shape.axis_y),
        dot(offset, shape.axis_z),
    );
    var closest = clamp(local, -shape.half_extents, shape.half_extents);
    let delta = local - closest;
    let distance = length(delta);
    if (distance > radius) { return miss(); }
    var normal_local: vec3<f32>;
    var depth: f32;
    if (distance > 1e-7) {
        normal_local = delta / distance;
        depth = radius - distance;
    } else {
        let gaps = shape.half_extents - abs(local);
        var axis = 0u;
        if (gaps.y < gaps.x) { axis = 1u; }
        if (gaps.z < gaps[axis]) { axis = 2u; }
        let side = select(-1.0, 1.0, local[axis] >= 0.0);
        if (axis == 0u) {
            normal_local = vec3<f32>(side, 0.0, 0.0);
            closest.x = side * shape.half_extents.x;
        } else if (axis == 1u) {
            normal_local = vec3<f32>(0.0, side, 0.0);
            closest.y = side * shape.half_extents.y;
        } else {
            normal_local = vec3<f32>(0.0, 0.0, side);
            closest.z = side * shape.half_extents.z;
        }
        depth = radius + gaps[axis];
    }
    let normal = normal_local.x * shape.axis_x
        + normal_local.y * shape.axis_y
        + normal_local.z * shape.axis_z;
    let point = shape.center + closest.x * shape.axis_x
        + closest.y * shape.axis_y + closest.z * shape.axis_z;
    return hit(point, normal, depth);
}

fn box_local(point: vec3<f32>, shape: Box) -> vec3<f32> {
    let offset = point - shape.center;
    return vec3<f32>(dot(offset, shape.axis_x), dot(offset, shape.axis_y),
        dot(offset, shape.axis_z));
}

fn box_distance_squared(local: vec3<f32>, half_extents: vec3<f32>) -> f32 {
    let outside = max(abs(local) - half_extents, vec3<f32>(0.0));
    return dot(outside, outside);
}

fn capsule_box(capsule: Capsule, shape: Box) -> Contact {
    let start = box_local(capsule.start, shape);
    let direction = box_local(capsule.end, shape) - start;
    var low = 0.0;
    var high = 1.0;
    // Distance to an AABB is convex along a segment. Ternary search finds its minimum.
    for (var iteration = 0u; iteration < 24u; iteration++) {
        let left = low + (high - low) / 3.0;
        let right = high - (high - low) / 3.0;
        let left_distance = box_distance_squared(start + direction * left, shape.half_extents);
        let right_distance = box_distance_squared(start + direction * right, shape.half_extents);
        if (left_distance <= right_distance) {
            high = right;
        } else {
            low = left;
        }
    }
    let core = capsule.start + (capsule.end - capsule.start) * ((low + high) * 0.5);
    return sphere_box(core, capsule.radius, shape);
}

fn linear_box_face_contacts(index: u32, capsule: Capsule, shape: Box,
    capsule_first: bool, contact: Contact) {
    var oriented = contact;
    if (capsule_first) { oriented.normal = -oriented.normal; }
    pair_contacts[index] = oriented;
    if (contact.depth_hit.y == 0.0) { return; }

    let normal = contact.normal.xyz;
    let axes = array<vec3<f32>, 3>(shape.axis_x, shape.axis_y, shape.axis_z);
    var face_axis = 0u;
    var alignment = 0.0;
    for (var axis = 0u; axis < 3u; axis++) {
        let candidate = abs(dot(normal, axes[axis]));
        if (candidate > alignment) {
            alignment = candidate;
            face_axis = axis;
        }
    }
    let segment = capsule.end - capsule.start;
    let segment_length = length(segment);
    if (alignment < 0.999 || segment_length < 1e-6 ||
        abs(dot(segment, normal)) > 0.05 * segment_length) { return; }

    let local_start = box_local(capsule.start, shape);
    let local_end = box_local(capsule.end, shape);
    let local_direction = local_end - local_start;
    var low = 0.0;
    var high = 1.0;
    for (var axis = 0u; axis < 3u; axis++) {
        if (axis == face_axis) { continue; }
        if (abs(local_direction[axis]) < 1e-7) {
            if (abs(local_start[axis]) > shape.half_extents[axis]) { return; }
        } else {
            let first = (-shape.half_extents[axis] - local_start[axis]) /
                local_direction[axis];
            let second = (shape.half_extents[axis] - local_start[axis]) /
                local_direction[axis];
            low = max(low, min(first, second));
            high = min(high, max(first, second));
        }
    }
    if ((high - low) * segment_length < max(0.25 * capsule.radius, 1e-3)) {
        return;
    }
    var first_contact = sphere_box(capsule.start + segment * low,
        capsule.radius, shape);
    var second_contact = sphere_box(capsule.start + segment * high,
        capsule.radius, shape);
    if (first_contact.depth_hit.y == 0.0 || second_contact.depth_hit.y == 0.0 ||
        dot(first_contact.normal.xyz, normal) < 0.98 ||
        dot(second_contact.normal.xyz, normal) < 0.98 ||
        distance(first_contact.point.xyz, second_contact.point.xyz) < 1e-3) {
        return;
    }
    if (capsule_first) {
        first_contact.normal = -first_contact.normal;
        second_contact.normal = -second_contact.normal;
    }
    pair_contacts[index] = first_contact;
    pair_contacts[params.pair_count + index * 3u] = second_contact;
}

fn capsule_box_face_contacts(index: u32, capsule: Capsule, shape: Box,
    capsule_first: bool) {
    linear_box_face_contacts(index, capsule, shape, capsule_first,
        capsule_box(capsule, shape));
}

fn projected_radius(shape: Box, axis: vec3<f32>) -> f32 {
    return shape.half_extents.x * abs(dot(shape.axis_x, axis))
        + shape.half_extents.y * abs(dot(shape.axis_y, axis))
        + shape.half_extents.z * abs(dot(shape.axis_z, axis));
}

fn consider_axis(
    a: Box, b: Box, delta: vec3<f32>, candidate: vec3<f32>, best: AxisResult
) -> AxisResult {
    let length_squared = dot(candidate, candidate);
    if (length_squared < 1e-10 || best.separated) { return best; }
    let axis = candidate * inverseSqrt(length_squared);
    let separation = dot(delta, axis);
    let overlap = projected_radius(a, axis) + projected_radius(b, axis) - abs(separation);
    if (overlap < 0.0) { return AxisResult(true, 0.0, vec3<f32>(0.0)); }
    if (overlap < best.depth) {
        return AxisResult(false, overlap, axis * select(-1.0, 1.0, separation >= 0.0));
    }
    return best;
}

fn support(shape: Box, direction: vec3<f32>) -> vec3<f32> {
    let sx = select(-1.0, 1.0, dot(shape.axis_x, direction) >= 0.0);
    let sy = select(-1.0, 1.0, dot(shape.axis_y, direction) >= 0.0);
    let sz = select(-1.0, 1.0, dot(shape.axis_z, direction) >= 0.0);
    return shape.center
        + sx * shape.half_extents.x * shape.axis_x
        + sy * shape.half_extents.y * shape.axis_y
        + sz * shape.half_extents.z * shape.axis_z;
}

fn box_box(a: Box, b: Box) -> Contact {
    let delta = b.center - a.center;
    var result = AxisResult(false, 1e30, vec3<f32>(1.0, 0.0, 0.0));
    result = consider_axis(a, b, delta, a.axis_x, result);
    result = consider_axis(a, b, delta, a.axis_y, result);
    result = consider_axis(a, b, delta, a.axis_z, result);
    result = consider_axis(a, b, delta, b.axis_x, result);
    result = consider_axis(a, b, delta, b.axis_y, result);
    result = consider_axis(a, b, delta, b.axis_z, result);
    result = consider_axis(a, b, delta, cross(a.axis_x, b.axis_x), result);
    result = consider_axis(a, b, delta, cross(a.axis_x, b.axis_y), result);
    result = consider_axis(a, b, delta, cross(a.axis_x, b.axis_z), result);
    result = consider_axis(a, b, delta, cross(a.axis_y, b.axis_x), result);
    result = consider_axis(a, b, delta, cross(a.axis_y, b.axis_y), result);
    result = consider_axis(a, b, delta, cross(a.axis_y, b.axis_z), result);
    result = consider_axis(a, b, delta, cross(a.axis_z, b.axis_x), result);
    result = consider_axis(a, b, delta, cross(a.axis_z, b.axis_y), result);
    result = consider_axis(a, b, delta, cross(a.axis_z, b.axis_z), result);
    if (result.separated) { return miss(); }
    let witness_a = support(a, result.normal);
    let witness_b = support(b, -result.normal);
    let center_midpoint = (a.center + b.center) * 0.5;
    let normal_midpoint = (witness_a + witness_b) * 0.5;
    let point = center_midpoint + result.normal
        * dot(normal_midpoint - center_midpoint, result.normal);
    return hit(point, result.normal, result.depth);
}

fn clip_face_polygon(input: FacePolygon, center: vec3<f32>,
                     axis: vec3<f32>, limit: f32) -> FacePolygon {
    var output: FacePolygon;
    if (input.count == 0u) { return output; }
    var overflow = false;
    var previous = input.points[input.count - 1u];
    var previous_distance = dot(previous - center, axis) - limit;
    for (var i = 0u; i < input.count; i++) {
        let current = input.points[i];
        let distance = dot(current - center, axis) - limit;
        let previous_inside = previous_distance <= 0.0;
        let current_inside = distance <= 0.0;
        if (previous_inside != current_inside && output.count < 32u) {
            let fraction = previous_distance / (previous_distance - distance);
            set_face_point(&output, output.count, previous + (current - previous) * fraction);
            output.count++;
        } else if (previous_inside != current_inside) {
            overflow = true;
        }
        if (current_inside && output.count < 32u) {
            set_face_point(&output, output.count, current);
            output.count++;
        } else if (current_inside) {
            overflow = true;
        }
        previous = current;
        previous_distance = distance;
    }
    if (overflow) { output.count = 0u; }
    return output;
}

fn box_box_face_manifold(index: u32, a: Box, b: Box, primary: Contact) {
    let normal = primary.normal.xyz;
    let a_axes = array<vec3<f32>, 3>(a.axis_x, a.axis_y, a.axis_z);
    let b_axes = array<vec3<f32>, 3>(b.axis_x, b.axis_y, b.axis_z);
    var best_alignment = 0.0;
    var reference_axis = 0u;
    var reference_is_b = false;
    for (var axis = 0u; axis < 3u; axis++) {
        let a_alignment = abs(dot(normal, a_axes[axis]));
        if (a_alignment > best_alignment) {
            best_alignment = a_alignment;
            reference_axis = axis;
            reference_is_b = false;
        }
        let b_alignment = abs(dot(normal, b_axes[axis]));
        if (b_alignment > best_alignment + 1e-6) {
            best_alignment = b_alignment;
            reference_axis = axis;
            reference_is_b = true;
        }
    }
    if (best_alignment < 0.999) { return; }
    var reference = a;
    var incident = b;
    var reference_normal = normal;
    if (reference_is_b) {
        reference = b;
        incident = a;
        reference_normal = -normal;
    }
    let reference_axes = array<vec3<f32>, 3>(
        reference.axis_x, reference.axis_y, reference.axis_z);
    let incident_axes = array<vec3<f32>, 3>(
        incident.axis_x, incident.axis_y, incident.axis_z);
    let face_sign = select(-1.0, 1.0,
        dot(reference_normal, reference_axes[reference_axis]) >= 0.0);
    let face_center = reference.center + face_sign
        * reference.half_extents[reference_axis]
        * reference_axes[reference_axis];
    let tangent_u = (reference_axis + 1u) % 3u;
    let tangent_v = (reference_axis + 2u) % 3u;
    var incident_axis = 0u;
    var incident_alignment = 0.0;
    for (var axis = 0u; axis < 3u; axis++) {
        let alignment = abs(dot(reference_normal, incident_axes[axis]));
        if (alignment > incident_alignment) {
            incident_alignment = alignment;
            incident_axis = axis;
        }
    }
    let incident_sign = select(1.0, -1.0,
        dot(reference_normal, incident_axes[incident_axis]) >= 0.0);
    let incident_center = incident.center + incident_sign
        * incident.half_extents[incident_axis]
        * incident_axes[incident_axis];
    let incident_u = (incident_axis + 1u) % 3u;
    let incident_v = (incident_axis + 2u) % 3u;
    var polygon: FacePolygon;
    polygon.count = 4u;
    for (var corner = 0u; corner < 4u; corner++) {
        let sign_u = select(-1.0, 1.0, corner == 1u || corner == 2u);
        let sign_v = select(-1.0, 1.0, corner >= 2u);
        set_face_point(&polygon, corner, incident_center
            + sign_u * incident.half_extents[incident_u] * incident_axes[incident_u]
            + sign_v * incident.half_extents[incident_v] * incident_axes[incident_v]);
    }
    polygon = clip_face_polygon(polygon, face_center,
        reference_axes[tangent_u], reference.half_extents[tangent_u]);
    polygon = clip_face_polygon(polygon, face_center,
        -reference_axes[tangent_u], reference.half_extents[tangent_u]);
    polygon = clip_face_polygon(polygon, face_center,
        reference_axes[tangent_v], reference.half_extents[tangent_v]);
    polygon = clip_face_polygon(polygon, face_center,
        -reference_axes[tangent_v], reference.half_extents[tangent_v]);
    var valid: FacePolygon;
    let duplicate_tolerance = 1e-6 * max(max(reference.half_extents.x,
        reference.half_extents.y), reference.half_extents.z);
    for (var candidate = 0u; candidate < polygon.count; candidate++) {
        let depth = dot(face_center - polygon.points[candidate], reference_normal);
        if (depth >= -1e-5) {
            var duplicate = false;
            for (var known = 0u; known < valid.count; known++) {
                let delta = valid.points[known] - polygon.points[candidate];
                duplicate = duplicate || dot(delta, delta) <=
                    duplicate_tolerance * duplicate_tolerance;
            }
            if (!duplicate) {
                set_face_point(&valid, valid.count, polygon.points[candidate]);
                valid.count++;
            }
        }
    }
    if (valid.count == 0u) { return; }
    let count = min(valid.count, 4u);
    for (var point = 0u; point < count; point++) {
        let selected = valid.points[(point * valid.count) / count];
        let depth = clamp(dot(face_center - selected, reference_normal),
            0.0, primary.depth_hit.x);
        let contact = hit(selected + reference_normal * (depth * 0.5),
            normal, depth);
        if (point == 0u) {
            pair_contacts[index] = contact;
        } else {
            pair_contacts[params.pair_count + index * 3u + point - 1u] = contact;
        }
    }
}

fn convex_world_vertex(index: u32, absolute: u32) -> vec3<f32> {
    let state = states[index];
    return state.position_inverse_mass.xyz + rotate(state.orientation,
        convex_vertices[absolute].xyz);
}

fn poly_face(index: u32, direction: vec3<f32>) -> PolyFace {
    var result: PolyFace;
    let shape = shapes[index];
    if (shape.kind.x == 1u) {
        let box = box_from(index);
        let axes = array<vec3<f32>, 3>(box.axis_x, box.axis_y, box.axis_z);
        var best_axis = 0u;
        for (var axis = 0u; axis < 3u; axis++) {
            let alignment = abs(dot(axes[axis], direction));
            if (alignment > result.alignment) {
                result.alignment = alignment;
                best_axis = axis;
            }
        }
        let sign = select(-1.0, 1.0, dot(axes[best_axis], direction) >= 0.0);
        result.normal = sign * axes[best_axis];
        let center = box.center + result.normal * box.half_extents[best_axis];
        let tangent_u = (best_axis + 1u) % 3u;
        let tangent_v = (best_axis + 2u) % 3u;
        var polygon: FacePolygon;
        polygon.count = 4u;
        for (var corner = 0u; corner < 4u; corner++) {
            let sign_u = select(-1.0, 1.0, corner == 1u || corner == 2u);
            let sign_v = select(-1.0, 1.0, corner >= 2u);
            set_face_point(&polygon, corner, center
                + sign_u * box.half_extents[tangent_u] * axes[tangent_u]
                + sign_v * box.half_extents[tangent_v] * axes[tangent_v]);
        }
        result.polygon = polygon;
        result.valid = true;
        return result;
    }
    if (shape.kind.x != 5u || shape.feature_counts.x == 0u) { return result; }
    for (var face = 0u; face < shape.feature_counts.x; face++) {
        let normal = rotate(states[index].orientation,
            convex_vertices[shape.kind.w + face].xyz);
        let alignment = dot(normal, direction);
        if (alignment > result.alignment) {
            result.alignment = alignment;
            result.normal = normal;
        }
    }
    if (result.alignment <= 0.0) { return result; }
    let center = states[index].position_inverse_mass.xyz;
    var plane = -1e30;
    for (var vertex = 0u; vertex < shape.kind.z; vertex++) {
        let point = convex_world_vertex(index, shape.kind.y + vertex);
        plane = max(plane, dot(point - center, result.normal));
    }
    let tolerance = max(shape.dimensions.x * 1e-5, 1e-6);
    var first = 0xffffffffu;
    var face_edges = 0u;
    for (var edge = 0u; edge < shape.feature_counts.z; edge++) {
        let pair = convex_edges[shape.feature_counts.y + edge];
        let a = convex_world_vertex(index, pair.x);
        let b = convex_world_vertex(index, pair.y);
        if (dot(a - center, result.normal) >= plane - tolerance &&
            dot(b - center, result.normal) >= plane - tolerance) {
            first = min(first, min(pair.x, pair.y));
            face_edges++;
        }
    }
    if (face_edges < 3u || face_edges > 32u) { return result; }
    var polygon: FacePolygon;
    var previous = 0xffffffffu;
    var current = first;
    for (var step = 0u; step < 32u; step++) {
        set_face_point(&polygon, step, convex_world_vertex(index, current));
        polygon.count++;
        var next = 0xffffffffu;
        for (var edge = 0u; edge < shape.feature_counts.z; edge++) {
            let pair = convex_edges[shape.feature_counts.y + edge];
            var neighbor = 0xffffffffu;
            if (pair.x == current) { neighbor = pair.y; }
            if (pair.y == current) { neighbor = pair.x; }
            if (neighbor == 0xffffffffu || neighbor == previous) { continue; }
            let point = convex_world_vertex(index, neighbor);
            if (dot(point - center, result.normal) >= plane - tolerance) {
                next = min(next, neighbor);
            }
        }
        if (next == first) {
            result.valid = polygon.count == face_edges;
            result.polygon = polygon;
            return result;
        }
        if (next == 0xffffffffu) { return result; }
        previous = current;
        current = next;
    }
    return result;
}

fn generic_face_manifold(index: u32, a: u32, b: u32, primary: Contact) {
    let normal = primary.normal.xyz;
    let face_a = poly_face(a, normal);
    let face_b = poly_face(b, -normal);
    var reference = face_a;
    var incident = poly_face(b, -face_a.normal);
    if (face_b.alignment > face_a.alignment + 1e-6) {
        reference = face_b;
        incident = poly_face(a, -face_b.normal);
    }
    if (!reference.valid || !incident.valid || reference.alignment < 0.995) {
        return;
    }
    let face_axis = generic_axis(a, b, reference.normal,
        AxisResult(false, 1e30, vec3<f32>(0.0)));
    if (face_axis.separated ||
        abs(face_axis.depth - primary.depth_hit.x) >
            max(1e-4, primary.depth_hit.x * 1e-3)) {
        return;
    }
    var polygon = incident.polygon;
    var face_center = vec3<f32>(0.0);
    for (var vertex = 0u; vertex < reference.polygon.count; vertex++) {
        face_center += reference.polygon.points[vertex];
    }
    face_center /= f32(reference.polygon.count);
    for (var edge = 0u; edge < reference.polygon.count; edge++) {
        let start = reference.polygon.points[edge];
        let end = reference.polygon.points[(edge + 1u) % reference.polygon.count];
        var side = cross(reference.normal, end - start);
        let length_squared = dot(side, side);
        if (length_squared < 1e-16) { return; }
        side *= inverseSqrt(length_squared);
        if (dot(face_center - start, side) > 0.0) { side = -side; }
        polygon = clip_face_polygon(polygon, start, side, 0.0);
        if (polygon.count == 0u) { return; }
    }
    var valid: FacePolygon;
    let tolerance = max(max(shapes[a].dimensions.x, shapes[b].dimensions.x) * 1e-5, 1e-5);
    for (var candidate = 0u; candidate < polygon.count; candidate++) {
        let point = polygon.points[candidate];
        let depth = dot(reference.polygon.points[0] - point, reference.normal);
        if (depth < -tolerance) { continue; }
        var duplicate = false;
        for (var known = 0u; known < valid.count; known++) {
            let delta = point - valid.points[known];
            duplicate = duplicate || dot(delta, delta) <= tolerance * tolerance;
        }
        if (!duplicate) {
            set_face_point(&valid, valid.count, point);
            valid.count++;
        }
    }
    var selected: array<vec3<f32>, 4>;
    var count = 0u;
    for (var slot = 0u; slot < 4u; slot++) {
        var best = vec3<f32>(0.0);
        var best_score = -1.0;
        for (var candidate = 0u; candidate < valid.count; candidate++) {
            let point = valid.points[candidate];
            if (slot == 0u) {
                if (best_score < 0.0 || point.x < best.x ||
                    (point.x == best.x && point.y < best.y)) {
                    best = point;
                    best_score = 0.0;
                }
                continue;
            }
            var nearest = 1e30;
            for (var known = 0u; known < count; known++) {
                let delta = point - selected[known];
                nearest = min(nearest, dot(delta, delta));
            }
            if (nearest > best_score) {
                best = point;
                best_score = nearest;
            }
        }
        if (best_score < 0.0 || (slot > 0u && best_score <= tolerance * tolerance)) {
            break;
        }
        selected[count] = best;
        let depth = clamp(dot(reference.polygon.points[0] - best, reference.normal),
            0.0, primary.depth_hit.x);
        let contact = hit(best + reference.normal * (depth * 0.5), normal, depth);
        if (count == 0u) {
            pair_contacts[index] = contact;
        } else {
            pair_contacts[params.pair_count + index * 3u + count - 1u] = contact;
        }
        count++;
    }
}

fn primitive_support(index: u32, direction: vec3<f32>) -> vec3<f32> {
    return primitive_support_relative(index, direction, vec3<f32>(0.0));
}

fn primitive_support_relative(index: u32, direction: vec3<f32>, origin: vec3<f32>) -> vec3<f32> {
    return primitive_support_center(index, direction, states[index].position_inverse_mass.xyz-origin);
}

fn primitive_support_center(index: u32, direction: vec3<f32>, center: vec3<f32>) -> vec3<f32> {
    let state = states[index];
    let shape = shapes[index];
    let kind = shape.kind.x;
    let length_squared = dot(direction, direction);
    let unit = select(vec3<f32>(1.0, 0.0, 0.0),
        direction * inverseSqrt(max(length_squared, 1e-20)), length_squared > 1e-20);
    if (kind == 0u) {
        return center + unit * shape.dimensions.x;
    }
    if (kind == 1u) {
        var box = box_from(index);
        box.center = center;
        return support(box, direction);
    }
    if (kind == 5u) {
        var best = center + rotate(state.orientation,
            convex_vertices[shape.kind.y].xyz);
        var best_projection = dot(best, direction);
        for (var i = 1u; i < shape.kind.z; i++) {
            let candidate = center + rotate(state.orientation,
                convex_vertices[shape.kind.y + i].xyz);
            let projection = dot(candidate, direction);
            if (projection > best_projection) {
                best = candidate;
                best_projection = projection;
            }
        }
        return best;
    }
    let axis = normalize(rotate(state.orientation, vec3<f32>(0.0, 0.0, 1.0)));
    let axial = dot(direction, axis);
    let half_length = shape.dimensions.y;
    let radius = shape.dimensions.x;
    if (kind == 2u) {
        return center + axis * select(-half_length, half_length, axial >= 0.0)
            + unit * radius;
    }
    let radial = direction - axis * axial;
    let radial_squared = dot(radial, radial);
    let rim = select(vec3<f32>(0.0),
        radial * (radius * inverseSqrt(max(radial_squared, 1e-20))),
        radial_squared > 1e-12 * max(length_squared, 1e-20));
    if (kind == 3u) {
        return center + axis * select(-half_length, half_length, axial >= 0.0) + rim;
    }
    let apex = center + axis * half_length;
    let base = center - axis * half_length + rim;
    return select(base, apex, dot(apex, direction) > dot(base, direction));
}

fn support_pair(a: u32, b: u32, direction: vec3<f32>) -> vec3<f32> {
    return primitive_support(a, direction) - primitive_support(b, -direction);
}

fn line_direction(a: vec3<f32>, b: vec3<f32>, toward: vec3<f32>) -> vec3<f32> {
    let edge = b - a;
    let perpendicular = cross(cross(edge, toward), edge);
    if (dot(perpendicular, perpendicular) > 1e-16) { return perpendicular; }
    let axis = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0),
        abs(edge.x) < 0.9 * length(edge));
    return cross(edge, axis);
}

fn expand_simplex(simplex: Simplex, point: vec3<f32>) -> Simplex {
    let a = point;
    let b = simplex.a;
    let c = simplex.b;
    let d = simplex.c;
    let toward = -a;
    if (simplex.count == 1u) {
        if (dot(b - a, toward) > 0.0) {
            return Simplex(a, b, c, 2u, line_direction(a, b, toward), false);
        }
        return Simplex(a, b, c, 1u, toward, false);
    }
    if (simplex.count == 2u) {
        let ab = b - a;
        let ac = c - a;
        let face = cross(ab, ac);
        if (dot(cross(face, ac), toward) > 0.0) {
            if (dot(ac, toward) > 0.0) {
                return Simplex(a, c, b, 2u, line_direction(a, c, toward), false);
            }
            return Simplex(a, b, c, 2u, line_direction(a, b, toward), false);
        }
        if (dot(cross(ab, face), toward) > 0.0) {
            return Simplex(a, b, c, 2u, line_direction(a, b, toward), false);
        }
        if (dot(face, toward) > 0.0) {
            return Simplex(a, b, c, 3u, face, false);
        }
        return Simplex(a, c, b, 3u, -face, false);
    }
    var face = cross(b - a, c - a);
    if (dot(face, d - a) > 0.0) { face = -face; }
    if (dot(face, toward) > 0.0) { return Simplex(a, b, c, 3u, face, false); }
    face = cross(c - a, d - a);
    if (dot(face, b - a) > 0.0) { face = -face; }
    if (dot(face, toward) > 0.0) { return Simplex(a, c, d, 3u, face, false); }
    face = cross(d - a, b - a);
    if (dot(face, c - a) > 0.0) { face = -face; }
    if (dot(face, toward) > 0.0) { return Simplex(a, d, b, 3u, face, false); }
    return Simplex(a, b, c, 4u, toward, true);
}

fn primitive_intersects(a: u32, b: u32) -> bool {
    var direction = states[b].position_inverse_mass.xyz - states[a].position_inverse_mass.xyz;
    if (dot(direction, direction) < 1e-16) { direction = vec3<f32>(1.0, 0.0, 0.0); }
    let first = support_pair(a, b, direction);
    var simplex = Simplex(first, vec3<f32>(0.0), vec3<f32>(0.0), 1u, -first, false);
    for (var iteration = 0u; iteration < 32u; iteration++) {
        if (dot(simplex.direction, simplex.direction) < 1e-16) { return true; }
        let point = support_pair(a, b, simplex.direction);
        let projection = dot(point, simplex.direction);
        if (projection < -1e-6) { return false; }
        if (dot(point - simplex.a, point - simplex.a) < 1e-14 ||
            (simplex.count > 1u && dot(point - simplex.b, point - simplex.b) < 1e-14) ||
            (simplex.count > 2u && dot(point - simplex.c, point - simplex.c) < 1e-14)) {
            return abs(projection) < 1e-6;
        }
        simplex = expand_simplex(simplex, point);
        if (simplex.inside) { return true; }
    }
    return false;
}

fn generic_axis(a: u32, b: u32, candidate: vec3<f32>, best: AxisResult) -> AxisResult {
    let length_squared = dot(candidate, candidate);
    if (length_squared < 1e-10 || best.separated) { return best; }
    let axis = candidate * inverseSqrt(length_squared);
    let min_a = dot(primitive_support(a, -axis), axis);
    let max_a = dot(primitive_support(a, axis), axis);
    let min_b = dot(primitive_support(b, -axis), axis);
    let max_b = dot(primitive_support(b, axis), axis);
    let forward = max_a - min_b;
    let backward = max_b - min_a;
    let depth = min(forward, backward);
    if (depth < -1e-5) { return AxisResult(true, 0.0, vec3<f32>(0.0)); }
    if (depth < best.depth) {
        return AxisResult(false, max(depth, 0.0),
            axis * select(-1.0, 1.0, forward < backward));
    }
    return best;
}

fn convex_edge_vector(index: u32, edge_index: u32) -> vec3<f32> {
    let edge = convex_edges[shapes[index].feature_counts.y + edge_index];
    let local = convex_vertices[edge.y].xyz - convex_vertices[edge.x].xyz;
    return rotate(states[index].orientation, local);
}

fn generic_convex_contact(a: u32, b: u32) -> Contact {
    let polyhedral = (shapes[a].kind.x == 1u || shapes[a].kind.x == 5u)
        && (shapes[b].kind.x == 1u || shapes[b].kind.x == 5u);
    if (!polyhedral && !primitive_intersects(a, b)) { return miss(); }
    let delta = states[b].position_inverse_mass.xyz - states[a].position_inverse_mass.xyz;
    var result = AxisResult(false, 1e30, vec3<f32>(1.0, 0.0, 0.0));
    result = generic_axis(a, b, delta, result);
    let q_a = states[a].orientation;
    let q_b = states[b].orientation;
    if (shapes[a].kind.x == 5u) {
        for (var i = 0u; i < shapes[a].feature_counts.x; i++) {
            result = generic_axis(a, b, rotate(q_a,
                convex_vertices[shapes[a].kind.w + i].xyz), result);
        }
    }
    if (shapes[b].kind.x == 5u) {
        for (var i = 0u; i < shapes[b].feature_counts.x; i++) {
            result = generic_axis(a, b, rotate(q_b,
                convex_vertices[shapes[b].kind.w + i].xyz), result);
        }
    }
    let axes_a = array<vec3<f32>, 3>(
        rotate(q_a, vec3<f32>(1.0, 0.0, 0.0)),
        rotate(q_a, vec3<f32>(0.0, 1.0, 0.0)),
        rotate(q_a, vec3<f32>(0.0, 0.0, 1.0)));
    let axes_b = array<vec3<f32>, 3>(
        rotate(q_b, vec3<f32>(1.0, 0.0, 0.0)),
        rotate(q_b, vec3<f32>(0.0, 1.0, 0.0)),
        rotate(q_b, vec3<f32>(0.0, 0.0, 1.0)));
    for (var i = 0u; i < 3u; i++) {
        result = generic_axis(a, b, axes_a[i], result);
        result = generic_axis(a, b, axes_b[i], result);
        for (var j = 0u; j < 3u; j++) {
            result = generic_axis(a, b, cross(axes_a[i], axes_b[j]), result);
        }
    }
    if (shapes[a].kind.x == 5u) {
        for (var i = 0u; i < shapes[a].feature_counts.z; i++) {
            let edge_a = convex_edge_vector(a, i);
            if (shapes[b].kind.x == 5u) {
                for (var j = 0u; j < shapes[b].feature_counts.z; j++) {
                    result = generic_axis(a, b,
                        cross(edge_a, convex_edge_vector(b, j)), result);
                }
            } else if (shapes[b].kind.x == 1u) {
                for (var j = 0u; j < 3u; j++) {
                    result = generic_axis(a, b, cross(edge_a, axes_b[j]), result);
                }
            }
        }
    } else if (shapes[a].kind.x == 1u && shapes[b].kind.x == 5u) {
        for (var i = 0u; i < 3u; i++) {
            for (var j = 0u; j < shapes[b].feature_counts.z; j++) {
                result = generic_axis(a, b,
                    cross(axes_a[i], convex_edge_vector(b, j)), result);
            }
        }
    }
    if (result.separated || result.depth > 1e20) { return miss(); }
    let witness_a = primitive_support(a, result.normal);
    let witness_b = primitive_support(b, -result.normal);
    let midpoint = (states[a].position_inverse_mass.xyz +
        states[b].position_inverse_mass.xyz) * 0.5;
    let witness_midpoint = (witness_a + witness_b) * 0.5;
    let point = midpoint + result.normal * dot(witness_midpoint - midpoint, result.normal);
    return hit(point, result.normal, result.depth);
}

// Refine generic SAT axes using the capsule segment and hull features.
fn convex_capsule_contact(hull: u32, capsule_index: u32, hull_first: bool) -> Contact {
    let primary = generic_convex_contact(hull, capsule_index);
    if (primary.depth_hit.y == 0.0) { return primary; }
    let shape = shapes[hull];
    let capsule = capsule_from(capsule_index);
    let segment = capsule.end - capsule.start;
    var best = AxisResult(false, primary.depth_hit.x, primary.normal.xyz);
    for (var vertex = 0u; vertex < shape.kind.z; vertex++) {
        let point = convex_world_vertex(hull, shape.kind.y + vertex);
        best = generic_axis(hull, capsule_index,
            segment_nearest(capsule.start, capsule.end, point) - point, best);
        best = generic_axis(hull, capsule_index, capsule.start - point, best);
        best = generic_axis(hull, capsule_index, capsule.end - point, best);
    }
    for (var edge = 0u; edge < shape.feature_counts.z; edge++) {
        let pair = convex_edges[shape.feature_counts.y + edge];
        let start = convex_world_vertex(hull, pair.x);
        let end = convex_world_vertex(hull, pair.y);
        let witness = closest_segments(capsule.start, capsule.end, start, end);
        best = generic_axis(hull, capsule_index, witness.segment - witness.triangle, best);
        best = generic_axis(hull, capsule_index, cross(segment, end - start), best);
    }
    if (best.separated || best.depth > 1e20) { return miss(); }
    var hull_witness = primitive_support(hull, best.normal);
    let plane = dot(hull_witness, best.normal);
    var axis_witness = segment_nearest(capsule.start, capsule.end, hull_witness);
    var nearest_squared = dot(axis_witness - hull_witness, axis_witness - hull_witness);
    let tolerance = max(1e-6, shape.dimensions.x * 1e-5);
    // Keep the witness on the supporting feature rather than the body midpoint.
    for (var vertex = 0u; vertex < shape.kind.z; vertex++) {
        let point = convex_world_vertex(hull, shape.kind.y + vertex);
        if (dot(point, best.normal) < plane - tolerance) { continue; }
        let rounded = segment_nearest(capsule.start, capsule.end, point);
        let squared = dot(rounded - point, rounded - point);
        if (squared < nearest_squared) { hull_witness = point; axis_witness = rounded; nearest_squared = squared; }
    }
    for (var edge = 0u; edge < shape.feature_counts.z; edge++) {
        let pair = convex_edges[shape.feature_counts.y + edge];
        let start = convex_world_vertex(hull, pair.x);
        let end = convex_world_vertex(hull, pair.y);
        if (dot(start, best.normal) < plane - tolerance || dot(end, best.normal) < plane - tolerance) { continue; }
        let witness = closest_segments(capsule.start, capsule.end, start, end);
        let squared = dot(witness.segment - witness.triangle, witness.segment - witness.triangle);
        if (squared < nearest_squared) { hull_witness = witness.triangle; axis_witness = witness.segment; nearest_squared = squared; }
    }
    let rounded_axis = primitive_support(capsule_index, -best.normal) + best.normal * capsule.radius;
    let projected = rounded_axis + best.normal * (plane - dot(rounded_axis, best.normal));
    var inside = true;
    for (var face = 0u; face < shape.feature_counts.x; face++) {
        let axis = rotate(states[hull].orientation, convex_vertices[shape.kind.w + face].xyz);
        let height = dot(primitive_support(hull, axis), axis);
        if (dot(projected, axis) > height + tolerance) { inside = false; }
    }
    let projected_squared = dot(rounded_axis - projected, rounded_axis - projected);
    if (inside && projected_squared < nearest_squared) { hull_witness = projected; axis_witness = rounded_axis; }
    let point = (hull_witness + axis_witness - best.normal * capsule.radius) * 0.5;
    return hit(point, select(-best.normal, best.normal, hull_first), best.depth);
}

// Clip a face-parallel capsule axis against the supporting convex face.
fn capsule_convex_face_contacts(index: u32, hull: u32, capsule_index: u32,
    hull_first: bool, primary: Contact) {
    if (primary.depth_hit.y == 0.0) { return; }
    var normal = select(-primary.normal.xyz, primary.normal.xyz, hull_first);
    let shape = shapes[hull];
    let capsule = capsule_from(capsule_index);
    let segment = capsule.end - capsule.start;
    let squared = dot(segment, segment);
    if (squared < 1e-12 || abs(dot(segment, normal)) > 0.05 * sqrt(squared)) { return; }
    var aligned = false;
    var alignment = 0.9995;
    let candidate_normal = normal;
    for (var face = 0u; face < shape.feature_counts.x; face++) {
        let axis = rotate(states[hull].orientation, convex_vertices[shape.kind.w + face].xyz);
        let score = dot(axis, candidate_normal);
        if (score >= alignment) { aligned = true; alignment = score; normal = axis; }
    }
    if (!aligned) { return; }
    var plane = -1e30;
    for (var vertex = 0u; vertex < shape.kind.z; vertex++) {
        plane = max(plane, dot(convex_world_vertex(hull, shape.kind.y + vertex), normal));
    }
    let projected = capsule.start + normal * (plane - dot(capsule.start, normal));
    let tangent_segment = segment - normal * dot(segment, normal);
    var low = 0.0;
    var high = 1.0;
    for (var face = 0u; face < shape.feature_counts.x; face++) {
        let axis = rotate(states[hull].orientation, convex_vertices[shape.kind.w + face].xyz);
        var height = -1e30;
        for (var vertex = 0u; vertex < shape.kind.z; vertex++) {
            height = max(height, dot(convex_world_vertex(hull, shape.kind.y + vertex), axis));
        }
        let delta = dot(projected, axis) - height;
        let slope = dot(tangent_segment, axis);
        if (abs(slope) < 1e-8) {
            if (delta > 1e-6) { return; }
        } else if (slope > 0.0) { high = min(high, -delta / slope); }
        else { low = max(low, -delta / slope); }
    }
    if (high < low) { return; }
    var contacts: array<Contact, 2>;
    for (var point = 0u; point < 2u; point++) {
        let fraction = select(low, high, point == 1u);
        let axis_point = capsule.start + segment * fraction;
        let depth = plane + capsule.radius - dot(axis_point, normal);
        if (depth < 0.0) { return; }
        let hull_point = axis_point + normal * (plane - dot(axis_point, normal));
        let rounded_point = axis_point - normal * capsule.radius;
        contacts[point] = hit((hull_point + rounded_point) * 0.5, select(-normal, normal, hull_first), depth);
    }
    pair_contacts[index] = contacts[0];
    let span = tangent_segment * (high - low);
    if (dot(span, span) > max(1e-10, shape.dimensions.x * shape.dimensions.x * 1e-8)) {
        pair_contacts[params.pair_count + index * 3u] = contacts[1];
    }
}

fn convex_sphere_contact(hull: u32, sphere: u32, sphere_first: bool) -> Contact {
    let shape = shapes[hull];
    let state = states[hull];
    let sphere_center = states[sphere].position_inverse_mass.xyz;
    var result = AxisResult(false, 1e30, vec3<f32>(1.0, 0.0, 0.0));
    for (var i = 0u; i < shape.feature_counts.x; i++) {
        result = generic_axis(hull, sphere, rotate(state.orientation,
            convex_vertices[shape.kind.w + i].xyz), result);
    }
    for (var i = 0u; i < shape.kind.z; i++) {
        let vertex = state.position_inverse_mass.xyz + rotate(state.orientation,
            convex_vertices[shape.kind.y + i].xyz);
        result = generic_axis(hull, sphere, sphere_center - vertex, result);
    }
    for (var i = 0u; i < shape.feature_counts.z; i++) {
        let pair = convex_edges[shape.feature_counts.y + i];
        let vertex = state.position_inverse_mass.xyz + rotate(state.orientation,
            convex_vertices[pair.x].xyz);
        let edge = convex_edge_vector(hull, i);
        let length_squared = dot(edge, edge);
        if (length_squared > 1e-16) {
            let along = clamp(dot(sphere_center - vertex, edge) / length_squared,
                0.0, 1.0);
            let closest = vertex + edge * along;
            result = generic_axis(hull, sphere, sphere_center - closest, result);
        }
    }
    if (result.separated || result.depth > 1e20) { return miss(); }
    let witness_hull = primitive_support(hull, result.normal);
    let witness_sphere = primitive_support(sphere, -result.normal);
    let midpoint = (state.position_inverse_mass.xyz + sphere_center) * 0.5;
    let witness_midpoint = (witness_hull + witness_sphere) * 0.5;
    let point = midpoint + result.normal * dot(witness_midpoint - midpoint, result.normal);
    let normal = select(result.normal, -result.normal, sphere_first);
    return hit(point, normal, result.depth);
}

fn analytic_sphere_contact(shape_index: u32, sphere_index: u32,
    sphere_first: bool) -> Contact {
    let shape = shapes[shape_index];
    let axis = rotate(states[shape_index].orientation, vec3<f32>(0.0, 0.0, 1.0));
    let center = states[shape_index].position_inverse_mass.xyz;
    let sphere_center = states[sphere_index].position_inverse_mass.xyz;
    let sphere_radius = shapes[sphere_index].dimensions.x;
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
    let half_length = shape.dimensions.y;
    let radius = shape.dimensions.x;
    var signed_distance: f32;
    var outward: vec3<f32>;
    if (shape.kind.x == 3u) {
        let radial_distance = radial - radius;
        let cap_distance = abs(height) - half_length;
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
        let base = vec2<f32>(clamp(radial, 0.0, radius), -half_length);
        let side_start = vec2<f32>(radius, -half_length);
        let side_direction = vec2<f32>(-radius, 2.0 * half_length);
        let side_t = clamp(dot(point - side_start, side_direction)
            / dot(side_direction, side_direction), 0.0, 1.0);
        let side = side_start + side_direction * side_t;
        let base_distance_squared = dot(point - base, point - base);
        let side_distance_squared = dot(point - side, point - side);
        let use_base = base_distance_squared <= side_distance_squared;
        let closest = select(side, base, use_base);
        let offset = point - closest;
        let distance = length(offset);
        let edge_epsilon = max(half_length, radius) * 1e-6;
        let inside = height >= -half_length - edge_epsilon
            && height <= half_length + edge_epsilon
            && radial <= radius * (half_length - height) / (2.0 * half_length) + edge_epsilon;
        let base_normal = vec2<f32>(0.0, -1.0);
        let side_normal = normalize(vec2<f32>(2.0 * half_length, radius));
        let near_corner = base_distance_squared <= edge_epsilon * edge_epsilon
            && side_distance_squared <= edge_epsilon * edge_epsilon;
        var feature_normal = select(side_normal, base_normal, use_base);
        if (near_corner) { feature_normal = normalize(base_normal + side_normal); }
        let normal_2d = select(offset / max(distance, 1e-12),
            feature_normal, inside || distance <= edge_epsilon);
        outward = radial_direction * normal_2d.x + axis * normal_2d.y;
        signed_distance = select(distance, -distance, inside);
    }
    if (signed_distance > sphere_radius + 1e-6) { return miss(); }
    let surface = sphere_center - outward * signed_distance;
    let sphere_witness = sphere_center - outward * sphere_radius;
    let normal = select(outward, -outward, sphere_first);
    return hit((surface + sphere_witness) * 0.5, normal,
        sphere_radius - signed_distance);
}

fn closest_triangle(point: vec3<f32>, a: vec3<f32>, b: vec3<f32>,
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

struct MeshNode {
    lower: vec3<f32>,
    upper: vec3<f32>,
    triangle: u32,
    escape: u32,
}

fn mesh_node(mesh: Shape, ordinal: u32) -> MeshNode {
    let first = mesh.feature_counts.y + ordinal * 3u;
    let lower_bits = convex_edges[first];
    let upper_bits = convex_edges[first + 1u];
    let link = convex_edges[first + 2u];
    return MeshNode(
        vec3<f32>(bitcast<f32>(lower_bits.x), bitcast<f32>(lower_bits.y),
            bitcast<f32>(lower_bits.z)),
        vec3<f32>(bitcast<f32>(upper_bits.x), bitcast<f32>(upper_bits.y),
            bitcast<f32>(upper_bits.z)),
        link.x, link.y);
}

fn polyline_witness_contact(state: RigidState, axis_point: vec3<f32>,
    closest: vec3<f32>, edge: vec3<f32>, axis: vec3<f32>, radius: f32,
    round_first: bool) -> Contact {
    let offset = axis_point - closest;
    let distance = length(offset);
    if (distance > radius) { return miss(); }
    var normal = normalize(cross(edge, select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(normalize(edge).z) > 0.9)));
    let crossed_axes = cross(edge, axis);
    if (dot(crossed_axes, crossed_axes) > 1e-12) {
        normal = normalize(crossed_axes);
    }
    if (distance > 1e-7) { normal = offset / distance; }
    let world_normal = rotate(state.orientation, normal);
    let world_closest = state.position_inverse_mass.xyz + rotate(state.orientation, closest);
    var contact = hit((world_closest + state.position_inverse_mass.xyz + rotate(state.orientation, axis_point) - world_normal * radius) * 0.5, world_normal, max(0.0, radius - distance));
    if (round_first) { contact.normal = -contact.normal; }
    return contact;
}

fn polyline_round_contacts(index: u32, line_index: u32, round_index: u32, round_first: bool) {
    let line = shapes[line_index];
    let state = states[line_index];
    let center = states[round_index].position_inverse_mass.xyz;
    let radius = shapes[round_index].dimensions.x;
    let inverse = vec4<f32>(-state.orientation.xyz, state.orientation.w);
    var start = center;
    var end = center;
    if (shapes[round_index].kind.x == 2u) {
        let capsule = capsule_from(round_index);
        start = capsule.start;
        end = capsule.end;
    }
    let local_start = rotate(inverse, start - state.position_inverse_mass.xyz);
    let local_end = rotate(inverse, end - state.position_inverse_mass.xyz);
    var best = miss();
    var endpoints_best = array<Contact, 4>(miss(), miss(), miss(), miss());
    var cursor = 0u;
    while (cursor < line.feature_counts.z) {
        let node = mesh_node(line, cursor);
        if (any(max(local_start, local_end) < node.lower - vec3<f32>(radius)) || any(min(local_start, local_end) > node.upper + vec3<f32>(radius))) {
            cursor = node.escape; continue;
        }
        cursor++;
        if (node.triangle == 0xffffffffu) { continue; }
        let segment = convex_edges[node.triangle];
        let a = convex_vertices[segment.x].xyz;
        let b = convex_vertices[segment.y].xyz;
        let edge = b - a;
        let witness = closest_segments(local_start, local_end, a, b);
        let closest = witness.triangle;
        let contact = polyline_witness_contact(state, witness.segment, closest,
            edge, local_end - local_start, radius, round_first);
        if (contact.depth_hit.y == 0.0) { continue; }
        if (shapes[round_index].kind.x == 2u) {
            let endpoints = array<vec3<f32>, 4>(local_start, local_end, a, b);
            for (var endpoint = 0u; endpoint < 4u; endpoint++) {
                var axis_point = endpoints[endpoint];
                var line_point = segment_nearest(a, b, axis_point);
                if (endpoint >= 2u) {
                    line_point = endpoints[endpoint];
                    axis_point = segment_nearest(local_start, local_end, line_point);
                }
                let candidate = polyline_witness_contact(state, axis_point, line_point,
                    edge, local_end - local_start, radius, round_first);
                if (candidate.depth_hit.y != 0.0 &&
                    (endpoints_best[endpoint].depth_hit.y == 0.0 ||
                     candidate.depth_hit.x > endpoints_best[endpoint].depth_hit.x)) {
                    endpoints_best[endpoint] = candidate;
                }
            }
        }
        if (best.depth_hit.y == 0.0 || contact.depth_hit.x > best.depth_hit.x) { best = contact; }
    }
    pair_contacts[index] = best;
    if (best.depth_hit.y == 0.0) { return; }
    let tolerance_squared = max(1e-8, dot(end - start, end - start) * 1e-4);
    var extra = array<Contact, 3>(miss(), miss(), miss());
    var count = 0u;
    for (var endpoint = 0u; endpoint < 4u; endpoint++) {
        let candidate = endpoints_best[endpoint];
        var distinct = mesh_capsule_distinct(best, candidate, tolerance_squared);
        for (var previous = 0u; previous < count; previous++) {
            distinct = distinct && mesh_capsule_distinct(extra[previous], candidate, tolerance_squared);
        }
        if (distinct && count < 3u) {
            extra[count] = candidate;
            pair_contacts[params.pair_count + index * 3u + count] = candidate;
            count++;
        }
    }
}

fn mesh_sphere_contact(mesh_index: u32, sphere_index: u32,
    sphere_first: bool) -> Contact {
    let mesh = shapes[mesh_index];
    let state = states[mesh_index];
    let sphere_center = states[sphere_index].position_inverse_mass.xyz;
    let radius = shapes[sphere_index].dimensions.x;
    let inverse_orientation = vec4<f32>(-state.orientation.xyz, state.orientation.w);
    let local_center = rotate(inverse_orientation,
        sphere_center - state.position_inverse_mass.xyz);
    var best = miss();
    var cursor = 0u;
    while (cursor < mesh.feature_counts.z) {
        let node = mesh_node(mesh, cursor);
        if (any(local_center < node.lower - vec3<f32>(radius)) ||
            any(local_center > node.upper + vec3<f32>(radius))) {
            cursor = node.escape;
            continue;
        }
        cursor++;
        if (node.triangle == 0xffffffffu) { continue; }
        let triangle = convex_edges[node.triangle];
        let a = state.position_inverse_mass.xyz +
            rotate(state.orientation, convex_vertices[triangle.x].xyz);
        let b = state.position_inverse_mass.xyz +
            rotate(state.orientation, convex_vertices[triangle.y].xyz);
        let c = state.position_inverse_mass.xyz +
            rotate(state.orientation, convex_vertices[triangle.z].xyz);
        let face = cross(b - a, c - a);
        if (dot(face, face) < 1e-20) { continue; }
        let closest = closest_triangle(sphere_center, a, b, c);
        let offset = sphere_center - closest;
        let distance_squared = dot(offset, offset);
        if (distance_squared > radius * radius) { continue; }
        let distance = sqrt(distance_squared);
        var normal = normalize(face);
        if (distance > 1e-7) {
            normal = offset / distance;
        } else if (dot(sphere_center - a, normal) < 0.0) {
            normal = -normal;
        }
        let sphere_witness = sphere_center - normal * radius;
        let depth = max(0.0, radius - distance);
        var contact = hit((closest + sphere_witness) * 0.5, normal, depth);
        if (sphere_first) { contact.normal = -contact.normal; }
        if (best.depth_hit.y == 0.0 || depth > best.depth_hit.x) { best = contact; }
    }
    return best;
}

struct SegmentTriangleWitness {
    segment: vec3<f32>,
    triangle: vec3<f32>,
}

fn closest_segments(start: vec3<f32>, end: vec3<f32>,
    edge_start: vec3<f32>, edge_end: vec3<f32>) -> SegmentTriangleWitness {
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
        return SegmentTriangleWitness(start, edge_start + v * t);
    }
    if (cc < 1e-12) {
        let s = clamp(-dd / aa, 0.0, 1.0);
        return SegmentTriangleWitness(start + u * s, edge_start);
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
    return SegmentTriangleWitness(start + u * s, edge_start + v * t);
}

// Thin surfaces have zero penetration and a non-unique normal at intersection.
fn thin_surface_contacts(index: u32, first: u32, second: u32, second_first: bool, append: bool) {
    let shape_a = shapes[first];
    let shape_b = shapes[second];
    let state_a = states[first];
    let state_b = states[second];
    let inverse_b = vec4<f32>(-state_b.orientation.xyz, state_b.orientation.w);
    var count = 0u;
    if (append) {
        if (pair_contacts[index].depth_hit.y != 0.0) {
            count = 1u;
            for (var extra = 0u; extra < 3u; extra++) {
                if (pair_contacts[params.pair_count + index * 3u + extra].depth_hit.y != 0.0) { count++; }
            }
        }
    } else { pair_contacts[index] = miss(); }
    for (var outer = 0u; outer < shape_a.feature_counts.z; outer++) {
        let leaf = mesh_node(shape_a, outer);
        if (leaf.triangle == 0xffffffffu) { continue; }
        let indices_a = convex_edges[leaf.triangle];
        let edge_count = select(1u, 3u, shape_a.kind.x == 6u);
        for (var edge = 0u; edge < edge_count; edge++) {
            let a = state_a.position_inverse_mass.xyz + rotate(state_a.orientation, convex_vertices[indices_a[edge]].xyz);
            let b = state_a.position_inverse_mass.xyz + rotate(state_a.orientation, convex_vertices[indices_a[(edge + 1u) % 3u]].xyz);
            if (dot(b - a, b - a) < 1e-20) { continue; }
            let local_a = rotate(inverse_b, a - state_b.position_inverse_mass.xyz);
            let local_b = rotate(inverse_b, b - state_b.position_inverse_mass.xyz);
            let lower = min(local_a, local_b) - vec3<f32>(1e-6);
            let upper = max(local_a, local_b) + vec3<f32>(1e-6);
            var cursor = 0u;
            while (cursor < shape_b.feature_counts.z) {
                let node = mesh_node(shape_b, cursor);
                if (any(upper < node.lower) || any(lower > node.upper)) { cursor = node.escape; continue; }
                cursor++;
                if (node.triangle == 0xffffffffu) { continue; }
                let indices_b = convex_edges[node.triangle];
                let c = state_b.position_inverse_mass.xyz + rotate(state_b.orientation, convex_vertices[indices_b.x].xyz);
                let d = state_b.position_inverse_mass.xyz + rotate(state_b.orientation, convex_vertices[indices_b.y].xyz);
                let e = state_b.position_inverse_mass.xyz + rotate(state_b.orientation, convex_vertices[indices_b.z].xyz);
                let face = cross(d - c, e - c);
                let triangle_mode = shape_b.kind.x == 6u;
                if (triangle_mode && dot(face, face) < 1e-20) { continue; }
                var witness: SegmentTriangleWitness;
                if (triangle_mode) { witness = closest_segment_triangle(a, b, c, d, e, face); }
                else { witness = closest_segments(a, b, c, d); }
                if (dot(witness.segment - witness.triangle, witness.segment - witness.triangle) > 1e-12) { continue; }
                let direction = normalize(b - a);
                let other_direction = normalize(d - c);
                var normal = cross(direction, other_direction);
                if (triangle_mode) { normal = face; }
                let parallel = !triangle_mode && dot(normal, normal) < 1e-10;
                if (parallel) {
                    let reference = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(direction.x) > 0.9);
                    normal = cross(direction, reference);
                }
                normal = normalize(normal);
                if (dot(normal, state_b.position_inverse_mass.xyz - state_a.position_inverse_mass.xyz) < 0.0) { normal = -normal; }
                var start = (witness.segment + witness.triangle) * 0.5;
                var end = start;
                if (triangle_mode && abs(dot(direction, normal)) < 1e-6 && abs(dot(a - c, normal)) < 1e-6) {
                    let interval = clip_segment_triangle(a, b, c, d, e, normalize(face), 0.0);
                    if (interval.x > interval.y) { continue; }
                    start = mix(a, b, interval.x);
                    end = mix(a, b, interval.y);
                }
                if (second_first) { normal = -normal; }
                // The reverse edge scan completes the same intersection polygon.
                if (append && count > 0u) { normal = pair_contacts[index].normal.xyz; }
                if (parallel) {
                    let length_a = length(b - a);
                    let t_c = dot(c - a, direction);
                    let t_d = dot(d - a, direction);
                    let low = max(0.0, min(t_c, t_d));
                    let high = min(length_a, max(t_c, t_d));
                    if (low > high + 1e-6) { continue; }
                    start = a + direction * low;
                    end = a + direction * max(low, high);
                }
                for (var endpoint = 0u; endpoint < 2u; endpoint++) {
                    let point = select(start, end, endpoint == 1u);
                    var distinct = true;
                    for (var previous = 0u; previous < count; previous++) {
                        let slot = select(params.pair_count + index * 3u + previous - 1u, index, previous == 0u);
                        let old = pair_contacts[slot];
                        let delta = old.point.xyz - point;
                        distinct = distinct && dot(delta, delta) > 1e-10;
                    }
                    if (count > 0u && dot(pair_contacts[index].normal.xyz, normal) < 0.99) { continue; }
                    if (distinct && count < 4u) {
                        let slot = select(params.pair_count + index * 3u + count - 1u, index, count == 0u);
                        pair_contacts[slot] = hit(point, normal, 0.0);
                        count++;
                    }
                }
            }
        }
    }
}

fn closest_segment_triangle(start: vec3<f32>, end: vec3<f32>,
    a: vec3<f32>, b: vec3<f32>, c: vec3<f32>,
    face: vec3<f32>) -> SegmentTriangleWitness {
    let closest_start = closest_triangle(start, a, b, c);
    var best = SegmentTriangleWitness(start, closest_start);
    var delta = best.segment - best.triangle;
    var best_distance = dot(delta, delta);
    let closest_end = closest_triangle(end, a, b, c);
    delta = end - closest_end;
    let end_distance = dot(delta, delta);
    if (end_distance < best_distance) {
        best = SegmentTriangleWitness(end, closest_end);
        best_distance = end_distance;
    }
    let ab = closest_segments(start, end, a, b);
    delta = ab.segment - ab.triangle;
    let ab_distance = dot(delta, delta);
    if (ab_distance < best_distance) {
        best = ab;
        best_distance = ab_distance;
    }
    let bc = closest_segments(start, end, b, c);
    delta = bc.segment - bc.triangle;
    let bc_distance = dot(delta, delta);
    if (bc_distance < best_distance) {
        best = bc;
        best_distance = bc_distance;
    }
    let ca = closest_segments(start, end, c, a);
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
                return SegmentTriangleWitness(point, point);
            }
        }
    }
    return best;
}

fn mesh_capsule_witness_contact(segment_point: vec3<f32>,
    triangle_point: vec3<f32>, face: vec3<f32>, face_point: vec3<f32>,
    capsule_center: vec3<f32>, radius: f32, capsule_first: bool) -> Contact {
    let offset = segment_point - triangle_point;
    let distance_squared = dot(offset, offset);
    if (distance_squared > radius * radius) { return miss(); }
    let distance = sqrt(distance_squared);
    var normal = normalize(face);
    if (distance > 1e-7) {
        normal = offset / distance;
    } else if (dot(capsule_center - face_point, normal) < 0.0) {
        normal = -normal;
    }
    let capsule_witness = segment_point - normal * radius;
    var contact = hit((triangle_point + capsule_witness) * 0.5,
        normal, radius - distance);
    if (capsule_first) { contact.normal = -contact.normal; }
    return contact;
}

fn mesh_capsule_distinct(primary: Contact, candidate: Contact,
    tolerance_squared: f32) -> bool {
    let delta = primary.point.xyz - candidate.point.xyz;
    return candidate.depth_hit.y != 0.0 &&
        dot(primary.normal.xyz, candidate.normal.xyz) > 0.99 &&
        dot(delta, delta) > tolerance_squared;
}

fn mesh_capsule_contacts(index: u32, mesh_index: u32, capsule_index: u32,
    capsule_first: bool) {
    let mesh = shapes[mesh_index];
    let state = states[mesh_index];
    let capsule = capsule_from(capsule_index);
    let capsule_center = (capsule.start + capsule.end) * 0.5;
    let inverse_orientation = vec4<f32>(-state.orientation.xyz, state.orientation.w);
    let local_start = rotate(inverse_orientation,
        capsule.start - state.position_inverse_mass.xyz);
    let local_end = rotate(inverse_orientation,
        capsule.end - state.position_inverse_mass.xyz);
    var best = miss();
    var start_contact = miss();
    var end_contact = miss();
    var cursor = 0u;
    while (cursor < mesh.feature_counts.z) {
        let node = mesh_node(mesh, cursor);
        if (any(max(local_start, local_end) < node.lower - vec3<f32>(capsule.radius)) ||
            any(min(local_start, local_end) > node.upper + vec3<f32>(capsule.radius))) {
            cursor = node.escape;
            continue;
        }
        cursor++;
        if (node.triangle == 0xffffffffu) { continue; }
        let triangle = convex_edges[node.triangle];
        let a = state.position_inverse_mass.xyz +
            rotate(state.orientation, convex_vertices[triangle.x].xyz);
        let b = state.position_inverse_mass.xyz +
            rotate(state.orientation, convex_vertices[triangle.y].xyz);
        let c = state.position_inverse_mass.xyz +
            rotate(state.orientation, convex_vertices[triangle.z].xyz);
        let face = cross(b - a, c - a);
        if (dot(face, face) < 1e-20) { continue; }
        let witnesses = closest_segment_triangle(capsule.start, capsule.end,
            a, b, c, face);
        let contact = mesh_capsule_witness_contact(witnesses.segment,
            witnesses.triangle, face, a, capsule_center, capsule.radius,
            capsule_first);
        if (contact.depth_hit.y != 0.0 &&
            (best.depth_hit.y == 0.0 || contact.depth_hit.x > best.depth_hit.x)) {
            best = contact;
        }
        let start_point = closest_triangle(capsule.start, a, b, c);
        let start_candidate = mesh_capsule_witness_contact(capsule.start,
            start_point, face, a, capsule_center, capsule.radius,
            capsule_first);
        if (start_candidate.depth_hit.y != 0.0 &&
            (start_contact.depth_hit.y == 0.0 ||
                start_candidate.depth_hit.x > start_contact.depth_hit.x)) {
            start_contact = start_candidate;
        }
        let end_point = closest_triangle(capsule.end, a, b, c);
        let end_candidate = mesh_capsule_witness_contact(capsule.end,
            end_point, face, a, capsule_center, capsule.radius,
            capsule_first);
        if (end_candidate.depth_hit.y != 0.0 &&
            (end_contact.depth_hit.y == 0.0 ||
                end_candidate.depth_hit.x > end_contact.depth_hit.x)) {
            end_contact = end_candidate;
        }
    }
    pair_contacts[index] = best;
    if (best.depth_hit.y == 0.0) { return; }
    let tolerance_squared = max(1e-8,
        dot(capsule.end - capsule.start, capsule.end - capsule.start) * 1e-4);
    var count = 0u;
    if (mesh_capsule_distinct(best, start_contact, tolerance_squared)) {
        pair_contacts[params.pair_count + index * 3u] = start_contact;
        count++;
    }
    if (mesh_capsule_distinct(best, end_contact, tolerance_squared) &&
        (count == 0u || mesh_capsule_distinct(start_contact,
            end_contact, tolerance_squared))) {
        pair_contacts[params.pair_count + index * 3u + count] = end_contact;
    }
}

fn triangle_analytic_support(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>,
    normal: vec3<f32>, other_index: u32, direction: vec3<f32>) -> vec3<f32> {
    var vertex = a;
    if (dot(b, direction) > dot(vertex, direction)) { vertex = b; }
    if (dot(c, direction) > dot(vertex, direction)) { vertex = c; }
    let thickness = select(0.0, select(-1e-5, 1e-5, dot(normal, direction) >= 0.0), dot(normal, normal) > 0.0);
    return vertex + normal * thickness - primitive_support(other_index, -direction);
}

fn triangle_analytic_intersects(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>,
    normal: vec3<f32>, other_index: u32) -> bool {
    var direction = states[other_index].position_inverse_mass.xyz - (a + b + c) / 3.0;
    if (dot(direction, direction) < 1e-16) {
        direction = select(vec3<f32>(1.0, 0.0, 0.0), normal, dot(normal, normal) > 0.0);
    }
    let first = triangle_analytic_support(a, b, c, normal, other_index, direction);
    var simplex = Simplex(first, vec3<f32>(0.0), vec3<f32>(0.0), 1u, -first, false);
    for (var iteration = 0u; iteration < 32u; iteration++) {
        if (dot(simplex.direction, simplex.direction) < 1e-16) { return true; }
        let point = triangle_analytic_support(a, b, c, normal, other_index,
            simplex.direction);
        let projection = dot(point, simplex.direction);
        if (projection < -1e-6) { return false; }
        if (dot(point - simplex.a, point - simplex.a) < 1e-14 ||
            (simplex.count > 1u && dot(point - simplex.b, point - simplex.b) < 1e-14) ||
            (simplex.count > 2u && dot(point - simplex.c, point - simplex.c) < 1e-14)) {
            return abs(projection) < 1e-6;
        }
        simplex = expand_simplex(simplex, point);
        if (simplex.inside) { return true; }
    }
    return false;
}

fn triangle_analytic_axis(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>,
    other_index: u32, candidate: vec3<f32>, best: AxisResult) -> AxisResult {
    let length_squared = dot(candidate, candidate);
    if (length_squared < 1e-12 || best.separated) { return best; }
    let axis = candidate * inverseSqrt(length_squared);
    let minimum = min(dot(a, axis), min(dot(b, axis), dot(c, axis)));
    let maximum = max(dot(a, axis), max(dot(b, axis), dot(c, axis)));
    let other_minimum = dot(primitive_support(other_index, -axis), axis);
    let other_maximum = dot(primitive_support(other_index, axis), axis);
    let forward = maximum - other_minimum;
    let backward = other_maximum - minimum;
    let depth = min(forward, backward);
    if (depth < -1e-5) { return AxisResult(true, 0.0, vec3<f32>(0.0)); }
    if (depth < best.depth) {
        return AxisResult(false, max(depth, 0.0),
            axis * select(-1.0, 1.0, forward < backward));
    }
    return best;
}

fn triangle_analytic_contact(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>,
    other_index: u32) -> Contact {
    let face = cross(b - a, c - a);
    if (dot(face, face) < 1e-20) { return miss(); }
    let normal = normalize(face);
    if (!triangle_analytic_intersects(a, b, c, normal, other_index)) {
        return miss();
    }
    let center = states[other_index].position_inverse_mass.xyz;
    let axis = rotate(states[other_index].orientation, vec3<f32>(0.0, 0.0, 1.0));
    let edges = array<vec3<f32>, 3>(b - a, c - b, a - c);
    var best = AxisResult(false, 1e30, normal);
    best = triangle_analytic_axis(a, b, c, other_index, normal, best);
    if (abs(dot(axis, normal)) < 0.99) {
        best = triangle_analytic_axis(a, b, c, other_index, axis, best);
        best = triangle_analytic_axis(a, b, c, other_index,
            center - closest_triangle(center, a, b, c), best);
        for (var edge = 0u; edge < 3u; edge++) {
            best = triangle_analytic_axis(a, b, c, other_index,
                cross(edges[edge], axis), best);
        }
    }
    if (best.separated || best.depth > 1e20) { return miss(); }
    let triangle_point = closest_triangle(center, a, b, c);
    let analytic_point = primitive_support(other_index, -best.normal);
    let point = (triangle_point + analytic_point) * 0.5;
    return hit(point, best.normal, best.depth);
}

// A repeated endpoint and zero normal describe a segment with no thickness.
fn segment_analytic_contact(a: vec3<f32>, b: vec3<f32>, other_index: u32) -> Contact {
    if (!triangle_analytic_intersects(a, b, b, vec3<f32>(0.0), other_index)) { return miss(); }
    let state = states[other_index];
    let center = state.position_inverse_mass.xyz;
    let axis = rotate(state.orientation, vec3<f32>(0.0, 0.0, 1.0));
    let edge = b - a;
    let nearest = segment_nearest(a, b, center);
    let offset = center - nearest;
    var best = AxisResult(false, 1e30, axis);
    best = triangle_analytic_axis(a, b, b, other_index, axis, best);
    best = triangle_analytic_axis(a, b, b, other_index, offset, best);
    best = triangle_analytic_axis(a, b, b, other_index, cross(edge, axis), best);
    if (shapes[other_index].kind.x == 4u) {
        let radials = array<vec3<f32>, 2>(offset - axis * dot(offset, axis), cross(edge, axis));
        for (var candidate = 0u; candidate < 2u; candidate++) {
            if (dot(radials[candidate], radials[candidate]) > 1e-12) {
                let radial = normalize(radials[candidate]);
                let slope = shapes[other_index].dimensions.x / (2.0 * shapes[other_index].dimensions.y);
                best = triangle_analytic_axis(a, b, b, other_index, radial + axis * slope, best);
                best = triangle_analytic_axis(a, b, b, other_index, radial - axis * slope, best);
            }
        }
    }
    if (best.separated || best.depth > 1e20) { return miss(); }
    let analytic_point = primitive_support(other_index, -best.normal);
    var line_point = segment_nearest(a, b, analytic_point);
    let axial_projection = dot(edge, best.normal);
    if (abs(axial_projection) > length(edge) * 1e-7) {
        line_point = select(a, b, axial_projection > 0.0);
    }
    // Project onto the selected support plane instead of using a remote cone rim.
    return hit(line_point - best.normal * best.depth * 0.5, best.normal, best.depth);
}

fn mesh_analytic_contact(mesh_index: u32, other_index: u32,
    other_first: bool) -> Contact {
    let mesh = shapes[mesh_index];
    let state = states[mesh_index];
    let center = state.position_inverse_mass.xyz;
    var lower = vec3<f32>(0.0);
    var upper = vec3<f32>(0.0);
    for (var component = 0u; component < 3u; component++) {
        var local_axis = vec3<f32>(0.0);
        local_axis[component] = 1.0;
        let world_axis = rotate(state.orientation, local_axis);
        lower[component] = dot(primitive_support(other_index, -world_axis) - center,
            world_axis);
        upper[component] = dot(primitive_support(other_index, world_axis) - center,
            world_axis);
    }
    var best = miss();
    var cursor = 0u;
    while (cursor < mesh.feature_counts.z) {
        let node = mesh_node(mesh, cursor);
        if (any(upper < node.lower) || any(lower > node.upper)) {
            cursor = node.escape;
            continue;
        }
        cursor++;
        if (node.triangle == 0xffffffffu) { continue; }
        let triangle = convex_edges[node.triangle];
        let a = center + rotate(state.orientation, convex_vertices[triangle.x].xyz);
        let b = center + rotate(state.orientation, convex_vertices[triangle.y].xyz);
        let c = center + rotate(state.orientation, convex_vertices[triangle.z].xyz);
        var contact: Contact;
        if (MESH_HULL_MODE == 2u) { contact = segment_analytic_contact(a, b, other_index); }
        else { contact = triangle_analytic_contact(a, b, c, other_index); }
        if (contact.depth_hit.y != 0.0 &&
            (best.depth_hit.y == 0.0 || contact.depth_hit.x > best.depth_hit.x)) {
            best = contact;
        }
    }
    if (other_first) { best.normal = -best.normal; }
    return best;
}

// Clip a parallel contact segment to the analytic solid's axial and radial bounds.
fn analytic_line_interval(a: vec3<f32>, b: vec3<f32>, other: u32) -> vec2<f32> {
    let state = states[other];
    let inverse = vec4<f32>(-state.orientation.xyz, state.orientation.w);
    let start = rotate(inverse, a - state.position_inverse_mass.xyz);
    let delta = rotate(inverse, b - a);
    let shape = shapes[other];
    let h = shape.dimensions.y;
    var low = 0.0;
    var high = 1.0;
    if (abs(delta.z) < 1e-7) {
        if (abs(start.z) > h + 1e-5) { return vec2<f32>(1.0, 0.0); }
    } else {
        let first = (-h - start.z) / delta.z;
        let second = (h - start.z) / delta.z;
        low = max(low, min(first, second));
        high = min(high, max(first, second));
    }
    var radius = shape.dimensions.x;
    if (shape.kind.x == 4u) { radius *= clamp((h - start.z) / (2.0 * h), 0.0, 1.0); }
    let aa = dot(delta.xy, delta.xy);
    let bb = 2.0 * dot(start.xy, delta.xy);
    let cc = dot(start.xy, start.xy) - radius * radius;
    if (aa < 1e-10) {
        if (cc > 1e-5) { return vec2<f32>(1.0, 0.0); }
    } else {
        let discriminant = bb * bb - 4.0 * aa * cc;
        if (discriminant < -1e-6 * max(1.0, aa * radius * radius)) {
            return vec2<f32>(1.0, 0.0);
        }
        let root = sqrt(max(0.0, discriminant));
        low = max(low, (-bb - root) / (2.0 * aa));
        high = min(high, (-bb + root) / (2.0 * aa));
    }
    return vec2<f32>(low, high);
}

fn polyline_analytic_contacts(index: u32, line: u32, other: u32, other_first: bool) {
    var primary = mesh_analytic_contact(line, other, false);
    if (primary.depth_hit.y == 0.0) { pair_contacts[index] = primary; return; }
    let axis = rotate(states[other].orientation, vec3<f32>(0.0, 0.0, 1.0));
    let alignment = dot(axis, primary.normal.xyz);
    let cap = abs(alignment) >= 0.999 && (shapes[other].kind.x == 3u || alignment >= 0.999);
    let side = shapes[other].kind.x == 3u && abs(alignment) <= 0.001;
    var tangent = vec3<f32>(0.0);
    var low = 1e30;
    var high = -1e30;
    var low_point = vec3<f32>(0.0);
    var high_point = vec3<f32>(0.0);
    let shape = shapes[line];
    let state = states[line];
    if (cap || side) {
        var lower = vec3<f32>(0.0);
        var upper = vec3<f32>(0.0);
        for (var component = 0u; component < 3u; component++) {
            var local_axis = vec3<f32>(0.0);
            local_axis[component] = 1.0;
            let world_axis = rotate(state.orientation, local_axis);
            lower[component] = dot(primitive_support(other, -world_axis) - state.position_inverse_mass.xyz, world_axis);
            upper[component] = dot(primitive_support(other, world_axis) - state.position_inverse_mass.xyz, world_axis);
        }
        var cursor = 0u;
        while (cursor < shape.feature_counts.z) {
            let node = mesh_node(shape, cursor);
            if (any(upper < node.lower) || any(lower > node.upper)) {
                cursor = node.escape;
                continue;
            }
            cursor++;
            if (node.triangle == 0xffffffffu) { continue; }
            let edge_indices = convex_edges[node.triangle];
            let a = state.position_inverse_mass.xyz + rotate(state.orientation, convex_vertices[edge_indices.x].xyz);
            let b = state.position_inverse_mass.xyz + rotate(state.orientation, convex_vertices[edge_indices.y].xyz);
            let direction = normalize(b - a);
            if (cap && abs(dot(direction, axis)) > 1e-6) { continue; }
            if (side && abs(dot(direction, axis)) < 0.9999) { continue; }
            let candidate = segment_analytic_contact(a, b, other);
            if (candidate.depth_hit.y == 0.0 || dot(candidate.normal, primary.normal) < 0.99 ||
                abs(candidate.depth_hit.x - primary.depth_hit.x) > max(1e-4, primary.depth_hit.x * 0.05)) { continue; }
            let interval = analytic_line_interval(a, b, other);
            if (interval.x > interval.y) { continue; }
            if (low > high) { tangent = direction; }
            if (abs(dot(direction, tangent)) < 0.9999) { continue; }
            for (var endpoint = 0u; endpoint < 2u; endpoint++) {
                let point = mix(a, b, select(interval.x, interval.y, endpoint == 1u));
                let score = dot(point, tangent);
                if (score < low) { low = score; low_point = point; }
                if (score > high) { high = score; high_point = point; }
            }
        }
    }
    if (low <= high) {
        primary.point = vec4<f32>(low_point - primary.normal.xyz * primary.depth_hit.x * 0.5, 0.0);
        if (high - low > 1e-4) {
            var extra = hit(high_point - primary.normal.xyz * primary.depth_hit.x * 0.5,
                primary.normal.xyz, primary.depth_hit.x);
            if (other_first) { extra.normal = -extra.normal; }
            pair_contacts[params.pair_count + index * 3u] = extra;
        }
    }
    if (other_first) { primary.normal = -primary.normal; }
    pair_contacts[index] = primary;
}

fn clip_segment_triangle(start: vec3<f32>, end: vec3<f32>,
    a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, normal: vec3<f32>,
    tolerance: f32) -> vec2<f32> {
    let vertices = array<vec3<f32>, 3>(a, b, c);
    var low = 0.0;
    var high = 1.0;
    for (var edge = 0u; edge < 3u; edge++) {
        let first = vertices[edge];
        let direction = vertices[(edge + 1u) % 3u] - first;
        let boundary = tolerance * length(direction);
        let start_side = dot(cross(direction, start - first), normal);
        let rate = dot(cross(direction, end - start), normal);
        if (abs(rate) < 1e-12) {
            if (start_side < -boundary) { return vec2<f32>(1.0, 0.0); }
        } else if (rate > 0.0) {
            low = max(low, (-boundary - start_side) / rate);
        } else {
            high = min(high, (boundary - start_side) / rate);
        }
        if (low > high) { return vec2<f32>(1.0, 0.0); }
    }
    return vec2<f32>(low, high);
}

fn mesh_analytic_contacts(index: u32, mesh_index: u32, other_index: u32,
    other_first: bool) {
    var primary = mesh_analytic_contact(mesh_index, other_index, false);
    if (primary.depth_hit.y == 0.0) {
        pair_contacts[index] = primary;
        return;
    }
    let shape = shapes[other_index];
    let other_state = states[other_index];
    let axis = normalize(rotate(other_state.orientation, vec3<f32>(0.0, 0.0, 1.0)));
    let normal = primary.normal.xyz;
    let alignment = dot(axis, normal);
    let cap_mode = abs(alignment) >= 0.999 &&
        (shape.kind.x == 3u || alignment >= 0.999);
    let side_mode = shape.kind.x == 3u && abs(alignment) <= 0.05;
    if (!cap_mode && !side_mode) {
        if (other_first) { primary.normal = -primary.normal; }
        pair_contacts[index] = primary;
        return;
    }

    let mesh = shapes[mesh_index];
    let mesh_state = states[mesh_index];
    let center = mesh_state.position_inverse_mass.xyz;
    var lower = vec3<f32>(0.0);
    var upper = vec3<f32>(0.0);
    for (var component = 0u; component < 3u; component++) {
        var local_axis = vec3<f32>(0.0);
        local_axis[component] = 1.0;
        let world_axis = rotate(mesh_state.orientation, local_axis);
        lower[component] = dot(primitive_support(other_index, -world_axis) - center,
            world_axis);
        upper[component] = dot(primitive_support(other_index, world_axis) - center,
            world_axis);
    }
    let tolerance = max(1e-5, shape.dimensions.x * 1e-4);
    if (side_mode) {
        let radial = normalize(normal - axis * alignment);
        let side_start = other_state.position_inverse_mass.xyz
            - axis * shape.dimensions.y - radial * shape.dimensions.x;
        let side_end = other_state.position_inverse_mass.xyz
            + axis * shape.dimensions.y - radial * shape.dimensions.x;
        var low_t = 1e30;
        var high_t = -1e30;
        var low_point = vec3<f32>(0.0);
        var high_point = vec3<f32>(0.0);
        var low_depth = 0.0;
        var high_depth = 0.0;
        var cursor = 0u;
        while (cursor < mesh.feature_counts.z) {
            let node = mesh_node(mesh, cursor);
            if (any(upper < node.lower) || any(lower > node.upper)) {
                cursor = node.escape;
                continue;
            }
            cursor++;
            if (node.triangle == 0xffffffffu) { continue; }
            let triangle = convex_edges[node.triangle];
            let a = center + rotate(mesh_state.orientation, convex_vertices[triangle.x].xyz);
            let b = center + rotate(mesh_state.orientation, convex_vertices[triangle.y].xyz);
            let c = center + rotate(mesh_state.orientation, convex_vertices[triangle.z].xyz);
            let face = cross(b - a, c - a);
            if (dot(face, face) < 1e-20) { continue; }
            let face_normal = normalize(face);
            if (abs(dot(face_normal, normal)) < 0.999) { continue; }
            let projected_start = side_start - face_normal * dot(side_start - a, face_normal);
            let projected_end = side_end - face_normal * dot(side_end - a, face_normal);
            let interval = clip_segment_triangle(projected_start, projected_end,
                a, b, c, face_normal, tolerance);
            if (interval.x > interval.y) { continue; }
            for (var endpoint = 0u; endpoint < 2u; endpoint++) {
                let t = select(interval.x, interval.y, endpoint == 1u);
                let shape_point = mix(side_start, side_end, t);
                let mesh_point = mix(projected_start, projected_end, t);
                let penetration = dot(mesh_point - shape_point, normal);
                if (penetration < -tolerance ||
                    penetration > primary.depth_hit.x + max(tolerance,
                        primary.depth_hit.x * 0.05)) { continue; }
                let point = (shape_point + mesh_point) * 0.5;
                if (t < low_t) {
                    low_t = t;
                    low_point = point;
                    low_depth = max(0.0, penetration);
                }
                if (t > high_t) {
                    high_t = t;
                    high_point = point;
                    high_depth = max(0.0, penetration);
                }
            }
        }
        let span = (high_t - low_t) * (2.0 * shape.dimensions.y);
        if (span > max(0.25 * shape.dimensions.x, 1e-3)) {
            let oriented_normal = select(normal, -normal, other_first);
            pair_contacts[index] = hit(low_point, oriented_normal, low_depth);
            pair_contacts[params.pair_count + index * 3u] =
                hit(high_point, oriented_normal, high_depth);
        } else {
            if (other_first) { primary.normal = -primary.normal; }
            pair_contacts[index] = primary;
        }
        return;
    }
    let face_sign = select(-1.0, 1.0, alignment > 0.0);
    let cap_center = other_state.position_inverse_mass.xyz
        - axis * shape.dimensions.y * face_sign;
    let rim_x = normalize(rotate(other_state.orientation, vec3<f32>(1.0, 0.0, 0.0)));
    let rim_y = normalize(rotate(other_state.orientation, vec3<f32>(0.0, 1.0, 0.0)));
    let samples = array<vec3<f32>, 4>(
        cap_center + rim_x * shape.dimensions.x,
        cap_center - rim_x * shape.dimensions.x,
        cap_center + rim_y * shape.dimensions.x,
        cap_center - rim_y * shape.dimensions.x);
    var points: array<vec3<f32>, 4>;
    var depths: array<f32, 4>;
    var count = 0u;
    var cursor = 0u;
    while (cursor < mesh.feature_counts.z && count < 4u) {
        let node = mesh_node(mesh, cursor);
        if (any(upper < node.lower) || any(lower > node.upper)) {
            cursor = node.escape;
            continue;
        }
        cursor++;
        if (node.triangle == 0xffffffffu) { continue; }
        let triangle = convex_edges[node.triangle];
        let a = center + rotate(mesh_state.orientation, convex_vertices[triangle.x].xyz);
        let b = center + rotate(mesh_state.orientation, convex_vertices[triangle.y].xyz);
        let c = center + rotate(mesh_state.orientation, convex_vertices[triangle.z].xyz);
        let face = cross(b - a, c - a);
        if (dot(face, face) < 1e-20) { continue; }
        let face_normal = normalize(face);
        if (abs(dot(face_normal, normal)) < 0.999) { continue; }
        for (var slot = 0u; slot < 4u && count < 4u; slot++) {
            let shape_point = samples[slot];
            let mesh_point = shape_point - face_normal * dot(shape_point - a, face_normal);
            let penetration = dot(mesh_point - shape_point, normal);
            if (penetration < -tolerance ||
                penetration > primary.depth_hit.x + max(tolerance, primary.depth_hit.x * 0.05)) {
                continue;
            }
            let nearest = closest_triangle(mesh_point, a, b, c);
            let delta = nearest - mesh_point;
            if (dot(delta, delta) > tolerance * tolerance) { continue; }
            let point = (shape_point + mesh_point) * 0.5;
            var duplicate = false;
            for (var known = 0u; known < count; known++) {
                let separation = points[known] - point;
                duplicate = duplicate || dot(separation, separation) < tolerance * tolerance;
            }
            if (duplicate) { continue; }
            points[count] = point;
            depths[count] = max(0.0, penetration);
            count++;
        }
    }
    if (count < 2u) {
        if (other_first) { primary.normal = -primary.normal; }
        pair_contacts[index] = primary;
        return;
    }
    var oriented_normal = normal;
    if (other_first) { oriented_normal = -oriented_normal; }
    for (var slot = 0u; slot < count; slot++) {
        let contact = hit(points[slot], oriented_normal, depths[slot]);
        if (slot == 0u) {
            pair_contacts[index] = contact;
        } else {
            pair_contacts[params.pair_count + index * 3u + slot - 1u] = contact;
        }
    }
}

fn consider_triangle_box_axis(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>,
    shape: Box, candidate: vec3<f32>, best: AxisResult) -> AxisResult {
    let length_squared = dot(candidate, candidate);
    if (length_squared < 1e-10 || best.separated) { return best; }
    let axis = candidate * inverseSqrt(length_squared);
    let pa = dot(a, axis);
    let pb = dot(b, axis);
    let pc = dot(c, axis);
    let triangle_min = min(pa, min(pb, pc));
    let triangle_max = max(pa, max(pb, pc));
    let box_center = dot(shape.center, axis);
    let radius = projected_radius(shape, axis);
    let positive_depth = triangle_max - (box_center - radius);
    let negative_depth = box_center + radius - triangle_min;
    if (positive_depth < 0.0 || negative_depth < 0.0) {
        return AxisResult(true, 0.0, vec3<f32>(0.0));
    }
    let depth = min(positive_depth, negative_depth);
    if (depth < best.depth) {
        return AxisResult(false, depth,
            axis * select(-1.0, 1.0, positive_depth <= negative_depth));
    }
    return best;
}

struct MeshPolyTriangle {
    contact: Contact,
    polygon: FacePolygon,
}

// Exact segment/OBB SAT followed by slab clipping; no artificial thickness.
fn segment_box_contact(a: vec3<f32>, b: vec3<f32>, shape: Box) -> MeshPolyTriangle {
    var result: MeshPolyTriangle;
    let edge = b - a;
    let axes = array<vec3<f32>, 3>(shape.axis_x, shape.axis_y, shape.axis_z);
    var best = AxisResult(false, 1e30, vec3<f32>(0.0, 0.0, 1.0));
    for (var axis = 0u; axis < 3u; axis++) {
        // Repeating an endpoint reuses the interval projection helper.
        best = consider_triangle_box_axis(a, b, b, shape, axes[axis], best);
        best = consider_triangle_box_axis(a, b, b, shape, cross(edge, axes[axis]), best);
    }
    if (best.separated) { return result; }
    let local_start = box_local(a, shape);
    let direction = box_local(b, shape) - local_start;
    var lower = 0.0;
    var upper = 1.0;
    for (var axis = 0u; axis < 3u; axis++) {
        if (direction[axis] == 0.0) {
            if (abs(local_start[axis]) > shape.half_extents[axis]) { return result; }
        } else {
            let first = (-shape.half_extents[axis] - local_start[axis]) / direction[axis];
            let second = (shape.half_extents[axis] - local_start[axis]) / direction[axis];
            lower = max(lower, min(first, second));
            upper = min(upper, max(first, second));
        }
    }
    if (lower > upper) { return result; }
    let first_point = a + edge * lower;
    let second_point = a + edge * upper;
    result.contact = hit((first_point + second_point) * 0.5, best.normal, best.depth);
    var polygon: FacePolygon;
    polygon.count = 2u;
    set_face_point(&polygon, 0u, first_point);
    set_face_point(&polygon, 1u, second_point);
    result.polygon = polygon;
    return result;
}

fn triangle_box_contact(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>,
    shape: Box) -> MeshPolyTriangle {
    var result: MeshPolyTriangle;
    let ab = b - a;
    let bc = c - b;
    let ca = a - c;
    let face = cross(ab, c - a);
    if (dot(face, face) < 1e-20) { return result; }
    let box_axes = array<vec3<f32>, 3>(shape.axis_x, shape.axis_y, shape.axis_z);
    let edges = array<vec3<f32>, 3>(ab, bc, ca);
    var best = AxisResult(false, 1e30, vec3<f32>(0.0, 0.0, 1.0));
    best = consider_triangle_box_axis(a, b, c, shape, face, best);
    for (var axis = 0u; axis < 3u; axis++) {
        best = consider_triangle_box_axis(a, b, c, shape, box_axes[axis], best);
        for (var edge = 0u; edge < 3u; edge++) {
            best = consider_triangle_box_axis(a, b, c, shape,
                cross(box_axes[axis], edges[edge]), best);
        }
    }
    if (best.separated) { return result; }

    var polygon: FacePolygon;
    polygon.count = 3u;
    set_face_point(&polygon, 0u, a);
    set_face_point(&polygon, 1u, b);
    set_face_point(&polygon, 2u, c);
    for (var axis = 0u; axis < 3u; axis++) {
        polygon = clip_face_polygon(polygon, shape.center, box_axes[axis],
            shape.half_extents[axis]);
        polygon = clip_face_polygon(polygon, shape.center, -box_axes[axis],
            shape.half_extents[axis]);
    }
    if (polygon.count == 0u) { return result; }
    var point = vec3<f32>(0.0);
    for (var vertex = 0u; vertex < polygon.count; vertex++) {
        point += polygon.points[vertex];
    }
    result.contact = hit(point / f32(polygon.count), best.normal, best.depth);
    result.polygon = polygon;
    return result;
}

fn hull_projection(hull: u32, axis: vec3<f32>) -> vec2<f32> {
    let shape = shapes[hull];
    var lower = 1e30;
    var upper = -1e30;
    for (var vertex = 0u; vertex < shape.kind.z; vertex++) {
        let position = convex_world_vertex(hull, shape.kind.y + vertex);
        let projection = dot(position, axis);
        lower = min(lower, projection);
        upper = max(upper, projection);
    }
    return vec2<f32>(lower, upper);
}

fn consider_triangle_hull_axis(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>,
    hull: u32, candidate: vec3<f32>, best: AxisResult) -> AxisResult {
    let length_squared = dot(candidate, candidate);
    if (length_squared < 1e-10 || best.separated) { return best; }
    let axis = candidate * inverseSqrt(length_squared);
    let pa = dot(a, axis);
    let pb = dot(b, axis);
    let pc = dot(c, axis);
    let triangle_min = min(pa, min(pb, pc));
    let triangle_max = max(pa, max(pb, pc));
    let hull_range = hull_projection(hull, axis);
    let hull_min = hull_range.x;
    let hull_max = hull_range.y;
    let positive_depth = triangle_max - hull_min;
    let negative_depth = hull_max - triangle_min;
    if (positive_depth < -1e-5 || negative_depth < -1e-5) {
        return AxisResult(true, 0.0, vec3<f32>(0.0));
    }
    let depth = max(0.0, min(positive_depth, negative_depth));
    if (depth < best.depth) {
        return AxisResult(false, depth,
            axis * select(-1.0, 1.0, positive_depth <= negative_depth));
    }
    return best;
}

fn segment_hull_contact(a: vec3<f32>, b: vec3<f32>, hull: u32) -> MeshPolyTriangle {
    var result: MeshPolyTriangle;
    let shape = shapes[hull];
    let state = states[hull];
    let edge = b - a;
    var best = AxisResult(false, 1e30, vec3<f32>(0.0, 0.0, 1.0));
    var lower = 0.0;
    var upper = 1.0;
    for (var feature = 0u; feature < shape.feature_counts.x; feature++) {
        let normal = rotate(state.orientation, convex_vertices[shape.kind.w + feature].xyz);
        best = consider_triangle_hull_axis(a, b, b, hull, normal, best);
        let limit = hull_projection(hull, normal).y;
        let start = dot(a, normal);
        let direction = dot(edge, normal);
        if (abs(direction) < 1e-12) {
            if (start > limit + 1e-5) { return result; }
        } else {
            let crossing = (limit - start) / direction;
            if (direction > 0.0) { upper = min(upper, crossing); }
            else { lower = max(lower, crossing); }
        }
    }
    for (var feature = 0u; feature < shape.feature_counts.z; feature++) {
        best = consider_triangle_hull_axis(a, b, b, hull,
            cross(edge, convex_edge_vector(hull, feature)), best);
    }
    if (best.separated || lower > upper) { return result; }
    let first = a + edge * lower;
    let second = a + edge * upper;
    result.contact = hit((first + second) * 0.5, best.normal, best.depth);
    var polygon: FacePolygon;
    polygon.count = 2u;
    set_face_point(&polygon, 0u, first);
    set_face_point(&polygon, 1u, second);
    result.polygon = polygon;
    return result;
}

fn triangle_hull_contact(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>,
    hull: u32) -> MeshPolyTriangle {
    var result: MeshPolyTriangle;
    let face = cross(b - a, c - a);
    if (dot(face, face) < 1e-20) { return result; }
    let shape = shapes[hull];
    let state = states[hull];
    let edges = array<vec3<f32>, 3>(b - a, c - b, a - c);
    var best = AxisResult(false, 1e30, vec3<f32>(0.0, 0.0, 1.0));
    best = consider_triangle_hull_axis(a, b, c, hull, face, best);
    for (var feature = 0u; feature < shape.feature_counts.x; feature++) {
        let normal = rotate(state.orientation,
            convex_vertices[shape.kind.w + feature].xyz);
        best = consider_triangle_hull_axis(a, b, c, hull, normal, best);
    }
    for (var feature = 0u; feature < shape.feature_counts.z; feature++) {
        let edge = convex_edge_vector(hull, feature);
        for (var triangle_edge = 0u; triangle_edge < 3u; triangle_edge++) {
            best = consider_triangle_hull_axis(a, b, c, hull,
                cross(edges[triangle_edge], edge), best);
        }
    }
    if (best.separated) { return result; }

    var polygon: FacePolygon;
    polygon.count = 3u;
    set_face_point(&polygon, 0u, a);
    set_face_point(&polygon, 1u, b);
    set_face_point(&polygon, 2u, c);
    for (var feature = 0u; feature < shape.feature_counts.x; feature++) {
        let normal = rotate(state.orientation,
            convex_vertices[shape.kind.w + feature].xyz);
        let limit = hull_projection(hull, normal).y;
        polygon = clip_face_polygon(polygon, vec3<f32>(0.0), normal, limit);
        if (polygon.count == 0u) { return result; }
    }
    var point = vec3<f32>(0.0);
    for (var vertex = 0u; vertex < polygon.count; vertex++) {
        point += polygon.points[vertex];
    }
    result.contact = hit(point / f32(polygon.count), best.normal, best.depth);
    result.polygon = polygon;
    return result;
}

fn mesh_poly_contacts(index: u32, mesh_index: u32, other_index: u32,
    other_first: bool) {
    let mesh = shapes[mesh_index];
    let state = states[mesh_index];
    let other_shape = shapes[other_index];
    let shape = box_from(other_index);
    let inverse_orientation = vec4<f32>(-state.orientation.xyz, state.orientation.w);
    let local_center = rotate(inverse_orientation,
        shape.center - state.position_inverse_mass.xyz);
    var local_lower = vec3<f32>(1e30);
    var local_upper = vec3<f32>(-1e30);
    var extent = other_shape.dimensions.x;
    if (MESH_HULL_MODE == 0u) {
        let local_axis_x = rotate(inverse_orientation, shape.axis_x);
        let local_axis_y = rotate(inverse_orientation, shape.axis_y);
        let local_axis_z = rotate(inverse_orientation, shape.axis_z);
        let reach = abs(local_axis_x) * shape.half_extents.x
            + abs(local_axis_y) * shape.half_extents.y
            + abs(local_axis_z) * shape.half_extents.z;
        local_lower = local_center - reach;
        local_upper = local_center + reach;
        extent = max(max(shape.half_extents.x, shape.half_extents.y),
            shape.half_extents.z);
    } else {
        for (var vertex = 0u; vertex < other_shape.kind.z; vertex++) {
            let world = convex_world_vertex(other_index, other_shape.kind.y + vertex);
            let local = rotate(inverse_orientation,
                world - state.position_inverse_mass.xyz);
            local_lower = min(local_lower, local);
            local_upper = max(local_upper, local);
        }
    }
    var best = miss();
    var selected: FacePolygon;
    var scores = vec4<f32>(-1e30);
    var tangent_u = vec3<f32>(1.0, 0.0, 0.0);
    var tangent_v = vec3<f32>(0.0, 1.0, 0.0);
    var cursor = 0u;
    while (cursor < mesh.feature_counts.z) {
        let node = mesh_node(mesh, cursor);
        if (any(local_upper < node.lower) || any(local_lower > node.upper)) {
            cursor = node.escape;
            continue;
        }
        cursor++;
        if (node.triangle == 0xffffffffu) { continue; }
        let triangle = convex_edges[node.triangle];
        let a = state.position_inverse_mass.xyz +
            rotate(state.orientation, convex_vertices[triangle.x].xyz);
        let b = state.position_inverse_mass.xyz +
            rotate(state.orientation, convex_vertices[triangle.y].xyz);
        let c = state.position_inverse_mass.xyz +
            rotate(state.orientation, convex_vertices[triangle.z].xyz);
        var candidate: MeshPolyTriangle;
        if (MESH_HULL_MODE == 0u) {
            if (mesh.kind.x == 7u) {
                candidate = segment_box_contact(a, b, shape);
            } else {
                candidate = triangle_box_contact(a, b, c, shape);
            }
        } else {
            if (mesh.kind.x == 7u) {
                candidate = segment_hull_contact(a, b, other_index);
            } else {
                candidate = triangle_hull_contact(a, b, c, other_index);
            }
        }
        let contact = candidate.contact;
        if (contact.depth_hit.y == 0.0) { continue; }
        let tolerance = max(1e-4, max(best.depth_hit.x, contact.depth_hit.x) * 0.05);
        let aligned = dot(contact.normal.xyz, best.normal.xyz) > 0.99;
        if (best.depth_hit.y == 0.0 || contact.depth_hit.x > best.depth_hit.x + tolerance ||
            (!aligned && contact.depth_hit.x > best.depth_hit.x)) {
            best = contact;
            selected.count = 0u;
            scores = vec4<f32>(-1e30);
            let normal = contact.normal.xyz;
            let seed = select(vec3<f32>(1.0, 0.0, 0.0),
                vec3<f32>(0.0, 1.0, 0.0), abs(normal.x) > 0.9);
            tangent_u = normalize(cross(normal, seed));
            tangent_v = cross(normal, tangent_u);
        } else if (contact.depth_hit.x > best.depth_hit.x) {
            best = contact;
        }
        if (abs(contact.depth_hit.x - best.depth_hit.x) > tolerance ||
            dot(contact.normal.xyz, best.normal.xyz) <= 0.99) { continue; }
        for (var vertex = 0u; vertex < candidate.polygon.count; vertex++) {
            let point = candidate.polygon.points[vertex];
            let offset = point - states[other_index].position_inverse_mass.xyz;
            let u = dot(offset, tangent_u);
            let v = dot(offset, tangent_v);
            if (u + v > scores.x) {
                scores.x = u + v;
                set_face_point(&selected, 0u, point);
            }
            if (u - v > scores.y) {
                scores.y = u - v;
                set_face_point(&selected, 1u, point);
            }
            if (-u + v > scores.z) {
                scores.z = -u + v;
                set_face_point(&selected, 2u, point);
            }
            if (-u - v > scores.w) {
                scores.w = -u - v;
                set_face_point(&selected, 3u, point);
            }
            selected.count = 4u;
        }
    }
    if (best.depth_hit.y == 0.0) {
        pair_contacts[index] = miss();
        return;
    }
    var count = 0u;
    let duplicate_tolerance_squared = max(1e-10, extent * extent * 1e-8);
    for (var slot = 0u; slot < selected.count; slot++) {
        let point = selected.points[slot];
        var duplicate = false;
        for (var previous = 0u; previous < slot; previous++) {
            let delta = selected.points[previous] - point;
            duplicate = duplicate || dot(delta, delta) < duplicate_tolerance_squared;
        }
        if (duplicate) { continue; }
        var normal = best.normal.xyz;
        if (other_first) { normal = -normal; }
        let contact = hit(point, normal, best.depth_hit.x);
        if (count == 0u) {
            pair_contacts[index] = contact;
        } else {
            pair_contacts[params.pair_count + index * 3u + count - 1u] = contact;
        }
        count++;
    }
    if (count == 0u) {
        if (other_first) { best.normal = -best.normal; }
        pair_contacts[index] = best;
    }
}

@compute @workgroup_size(64)
fn pair_contacts_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= params.pair_count) { return; }
    let pair = pairs[index];
    if (MESH_HULL_MODE == 3u) {
        if (!allows(collision_groups[pair.a], collision_groups[pair.b])) { return; }
        let kind_a = shapes[pair.a].kind.x;
        let kind_b = shapes[pair.b].kind.x;
        if (kind_a == 6u && kind_b == 6u) {
            thin_surface_contacts(index, pair.a, pair.b, false, false);
            thin_surface_contacts(index, pair.b, pair.a, true, true);
        } else if (kind_a == 7u && kind_b == 7u) {
            thin_surface_contacts(index, pair.a, pair.b, false, false);
        } else if (kind_a == 7u && kind_b == 6u) {
            thin_surface_contacts(index, pair.a, pair.b, false, false);
        } else if (kind_a == 6u && kind_b == 7u) {
            thin_surface_contacts(index, pair.b, pair.a, true, false);
        }
        return;
    }
    if (MESH_HULL_MODE == 2u) {
        if (!allows(collision_groups[pair.a], collision_groups[pair.b])) { return; }
        let kind_a = shapes[pair.a].kind.x;
        let kind_b = shapes[pair.b].kind.x;
        if (kind_a == 7u && (kind_b == 3u || kind_b == 4u)) {
            polyline_analytic_contacts(index, pair.a, pair.b, false);
        } else if ((kind_a == 3u || kind_a == 4u) && kind_b == 7u) {
            polyline_analytic_contacts(index, pair.b, pair.a, true);
        }
        return;
    }
    if (MESH_HULL_MODE == 1u) {
        if (!allows(collision_groups[pair.a], collision_groups[pair.b])) { return; }
        let kind_a = shapes[pair.a].kind.x;
        let kind_b = shapes[pair.b].kind.x;
        if (kind_a == 5u && (kind_b == 6u || kind_b == 7u)) {
            mesh_poly_contacts(index, pair.b, pair.a, true);
        } else if ((kind_a == 6u || kind_a == 7u) && kind_b == 5u) {
            mesh_poly_contacts(index, pair.a, pair.b, false);
        }
        return;
    }
    for (var point = 0u; point < 3u; point++) {
        pair_contacts[params.pair_count + index * 3u + point] = miss();
    }
    if (!allows(collision_groups[pair.a], collision_groups[pair.b])) {
        pair_contacts[index] = miss();
        return;
    }
    let kind_a = shapes[pair.a].kind.x;
    let kind_b = shapes[pair.b].kind.x;
    if (kind_a == 6u && kind_b == 6u) {
        pair_contacts[index] = miss();
    } else if (kind_a == 7u && (kind_b == 0u || kind_b == 2u)) {
        polyline_round_contacts(index, pair.a, pair.b, false);
    } else if ((kind_a == 0u || kind_a == 2u) && kind_b == 7u) {
        polyline_round_contacts(index, pair.b, pair.a, true);
    } else if (kind_a == 7u && kind_b == 1u) {
        mesh_poly_contacts(index, pair.a, pair.b, false);
    } else if (kind_a == 1u && kind_b == 7u) {
        mesh_poly_contacts(index, pair.b, pair.a, true);
    } else if (kind_a == 7u || kind_b == 7u) {
        pair_contacts[index] = miss();
    } else if (kind_a == 0u && kind_b == 0u) {
        pair_contacts[index] = sphere_sphere(pair.a, pair.b);
    } else if (kind_a == 0u && kind_b == 1u) {
        var contact = sphere_box(
            states[pair.a].position_inverse_mass.xyz,
            shapes[pair.a].dimensions.x,
            box_from(pair.b),
        );
        contact.normal = -contact.normal;
        pair_contacts[index] = contact;
    } else if (kind_a == 1u && kind_b == 0u) {
        pair_contacts[index] = sphere_box(
            states[pair.b].position_inverse_mass.xyz,
            shapes[pair.b].dimensions.x,
            box_from(pair.a),
        );
    } else if (kind_a == 1u && kind_b == 1u) {
        let a = box_from(pair.a);
        let b = box_from(pair.b);
        let contact = box_box(a, b);
        pair_contacts[index] = contact;
        if (contact.depth_hit.y != 0.0) {
            box_box_face_manifold(index, a, b, contact);
        }
    } else if (kind_a == 0u && kind_b == 2u) {
        pair_contacts[index] = sphere_capsule(
            states[pair.a].position_inverse_mass.xyz,
            shapes[pair.a].dimensions.x,
            capsule_from(pair.b),
        );
    } else if (kind_a == 2u && kind_b == 0u) {
        var contact = sphere_capsule(
            states[pair.b].position_inverse_mass.xyz,
            shapes[pair.b].dimensions.x,
            capsule_from(pair.a),
        );
        contact.normal = -contact.normal;
        pair_contacts[index] = contact;
    } else if (kind_a == 2u && kind_b == 2u) {
        capsule_capsule_side_contacts(index, capsule_from(pair.a), capsule_from(pair.b));
    } else if (kind_a == 2u && kind_b == 1u) {
        capsule_box_face_contacts(index, capsule_from(pair.a), box_from(pair.b), true);
    } else if (kind_a == 1u && kind_b == 2u) {
        capsule_box_face_contacts(index, capsule_from(pair.b), box_from(pair.a), false);
    } else if (kind_a == 0u && kind_b == 5u) {
        pair_contacts[index] = convex_sphere_contact(pair.b, pair.a, true);
    } else if (kind_a == 5u && kind_b == 0u) {
        pair_contacts[index] = convex_sphere_contact(pair.a, pair.b, false);
    } else if (kind_a == 0u && kind_b == 6u) {
        pair_contacts[index] = mesh_sphere_contact(pair.b, pair.a, true);
    } else if (kind_a == 6u && kind_b == 0u) {
        pair_contacts[index] = mesh_sphere_contact(pair.a, pair.b, false);
    } else if (kind_a == 2u && kind_b == 6u) {
        mesh_capsule_contacts(index, pair.b, pair.a, true);
    } else if (kind_a == 6u && kind_b == 2u) {
        mesh_capsule_contacts(index, pair.a, pair.b, false);
    } else if ((kind_a == 3u || kind_a == 4u) && kind_b == 6u) {
        mesh_analytic_contacts(index, pair.b, pair.a, true);
    } else if (kind_a == 6u && (kind_b == 3u || kind_b == 4u)) {
        mesh_analytic_contacts(index, pair.a, pair.b, false);
    } else if (kind_a == 1u && kind_b == 6u) {
        mesh_poly_contacts(index, pair.b, pair.a, true);
    } else if (kind_a == 6u && kind_b == 1u) {
        mesh_poly_contacts(index, pair.a, pair.b, false);
    } else if (kind_a == 6u || kind_b == 6u) {
        pair_contacts[index] = miss();
    } else if (kind_a == 0u && (kind_b == 3u || kind_b == 4u)) {
        pair_contacts[index] = analytic_sphere_contact(pair.b, pair.a, true);
    } else if (kind_b == 0u && (kind_a == 3u || kind_a == 4u)) {
        pair_contacts[index] = analytic_sphere_contact(pair.a, pair.b, false);
    } else {
        var contact = miss();
        if (kind_a == 5u && kind_b == 2u) {
            contact = convex_capsule_contact(pair.a, pair.b, true);
        } else if (kind_a == 2u && kind_b == 5u) {
            contact = convex_capsule_contact(pair.b, pair.a, false);
        } else {
            contact = generic_convex_contact(pair.a, pair.b);
        }
        pair_contacts[index] = contact;
        if (kind_a == 5u && kind_b == 2u) {
            capsule_convex_face_contacts(index, pair.a, pair.b, true, contact);
        } else if (kind_a == 2u && kind_b == 5u) {
            capsule_convex_face_contacts(index, pair.b, pair.a, false, contact);
        } else if (kind_a == 3u && kind_b == 1u) {
            var box_contact = contact;
            box_contact.normal = -box_contact.normal;
            linear_box_face_contacts(index, capsule_from(pair.a), box_from(pair.b),
                true, box_contact);
        } else if (kind_a == 1u && kind_b == 3u) {
            linear_box_face_contacts(index, capsule_from(pair.b), box_from(pair.a),
                false, contact);
        } else if ((kind_a == 2u || kind_a == 3u) &&
                   (kind_b == 2u || kind_b == 3u)) {
            linear_side_contacts(index, capsule_from(pair.a), capsule_from(pair.b),
                contact);
        } else if (contact.depth_hit.y != 0.0 &&
            (kind_a == 5u || kind_b == 5u) &&
            (kind_a == 1u || kind_a == 5u) &&
            (kind_b == 1u || kind_b == 5u)) {
            generic_face_manifold(index, pair.a, pair.b, contact);
        }
    }
}

fn bottom_support(shape: Box) -> vec3<f32> {
    var point = shape.center;
    let axes = array<vec3<f32>, 3>(shape.axis_x, shape.axis_y, shape.axis_z);
    for (var axis = 0u; axis < 3u; axis++) {
        let downward = -axes[axis].z;
        var side = 0.0;
        if (downward > 1e-6) { side = 1.0; }
        if (downward < -1e-6) { side = -1.0; }
        point += side * shape.half_extents[axis] * axes[axis];
    }
    return point;
}

fn ground_hit(point: vec3<f32>, normal: vec3<f32>, depth: f32) -> Contact {
    var contact = hit(point, normal, depth);
    if (bitcast<f32>(params.padding.y) > 0.0) { contact.depth_hit.x = depth; }
    return contact;
}

fn box_ground(index: u32, half: f32) {
    let shape = box_from(index);
    let reach = vec3<f32>(
        projected_radius(shape, vec3<f32>(1.0, 0.0, 0.0)),
        projected_radius(shape, vec3<f32>(0.0, 1.0, 0.0)),
        projected_radius(shape, vec3<f32>(0.0, 0.0, 1.0)),
    );
    if (abs(shape.center.x) > half + reach.x ||
        abs(shape.center.y) > half + reach.y ||
        shape.center.z - reach.z > bitcast<f32>(params.padding.y)) {
        ground_contacts[index] = miss();
        return;
    }
    var lowest = 1e30;
    for (var vertex = 0u; vertex < 8u; vertex++) {
        let sx = select(-1.0, 1.0, (vertex & 1u) != 0u);
        let sy = select(-1.0, 1.0, (vertex & 2u) != 0u);
        let sz = select(-1.0, 1.0, (vertex & 4u) != 0u);
        let point = shape.center + sx * shape.half_extents.x * shape.axis_x
            + sy * shape.half_extents.y * shape.axis_y
            + sz * shape.half_extents.z * shape.axis_z;
        if (abs(point.x) <= half && abs(point.y) <= half) {
            lowest = min(lowest, point.z);
        }
    }
    let tolerance = max(max(shape.half_extents.x, shape.half_extents.y),
        shape.half_extents.z) * 1e-5;
    var count = 0u;
    if (lowest <= bitcast<f32>(params.padding.y)) {
        for (var vertex = 0u; vertex < 8u; vertex++) {
            let sx = select(-1.0, 1.0, (vertex & 1u) != 0u);
            let sy = select(-1.0, 1.0, (vertex & 2u) != 0u);
            let sz = select(-1.0, 1.0, (vertex & 4u) != 0u);
            let point = shape.center + sx * shape.half_extents.x * shape.axis_x
                + sy * shape.half_extents.y * shape.axis_y
                + sz * shape.half_extents.z * shape.axis_z;
            if (abs(point.x) <= half && abs(point.y) <= half &&
                point.z <= lowest + tolerance && count < 4u) {
                let contact = ground_hit(vec3<f32>(point.xy, point.z * 0.5),
                    vec3<f32>(0.0, 0.0, 1.0), -point.z);
                if (count == 0u) {
                    ground_contacts[index] = contact;
                } else {
                    ground_contacts[params.body_count + index * 3u + count - 1u] = contact;
                }
                count++;
            }
        }
    }
    if (count == 0u) {
        let bottom = bottom_support(shape);
        if (abs(bottom.x) <= half && abs(bottom.y) <= half && bottom.z <= bitcast<f32>(params.padding.y)) {
            ground_contacts[index] = ground_hit(vec3<f32>(bottom.xy, bottom.z * 0.5),
                vec3<f32>(0.0, 0.0, 1.0), -bottom.z);
        } else {
            ground_contacts[index] = miss();
        }
    }
}

fn ground_vertex_index(shape: Shape, ordinal: u32) -> u32 {
    if (shape.kind.x == 7u) {
        let segment = convex_edges[shape.kind.w + ordinal / 2u];
        return select(segment.x, segment.y, (ordinal % 2u) != 0u);
    }
    if (shape.kind.x == 6u) {
        let triangle = convex_edges[shape.kind.w + ordinal / 3u];
        let corner = ordinal % 3u;
        if (corner == 0u) { return triangle.x; }
        if (corner == 1u) { return triangle.y; }
        return triangle.z;
    }
    return shape.kind.y + ordinal;
}

fn convex_ground(index: u32, half: f32) {
    let shape = shapes[index];
    let state = states[index];
    let center = state.position_inverse_mass.xyz;
    let radius = shape.dimensions.x;
    let vertex_count = select(select(shape.kind.z, shape.feature_counts.x * 3u,
        shape.kind.x == 6u), shape.feature_counts.x * 2u, shape.kind.x == 7u);
    if (abs(center.x) > half + radius || abs(center.y) > half + radius ||
        center.z - radius > bitcast<f32>(params.padding.y)) {
        ground_contacts[index] = miss();
        return;
    }
    var lowest = 1e30;
    for (var vertex = 0u; vertex < vertex_count; vertex++) {
        let point = center + rotate(state.orientation,
            convex_vertices[ground_vertex_index(shape, vertex)].xyz);
        if (abs(point.x) <= half && abs(point.y) <= half) {
            lowest = min(lowest, point.z);
        }
    }
    if (lowest > bitcast<f32>(params.padding.y)) {
        ground_contacts[index] = miss();
        return;
    }
    let tolerance = max(radius * 1e-5, 1e-6);
    var selected: array<vec3<f32>, 4>;
    var count = 0u;
    for (var slot = 0u; slot < 4u; slot++) {
        var best = vec3<f32>(0.0);
        var best_score = -1.0;
        for (var vertex = 0u; vertex < vertex_count; vertex++) {
            let point = center + rotate(state.orientation,
                convex_vertices[ground_vertex_index(shape, vertex)].xyz);
            if (abs(point.x) > half || abs(point.y) > half ||
                point.z > lowest + tolerance) { continue; }
            if (slot == 0u) {
                if (best_score < 0.0 || point.x < best.x ||
                    (point.x == best.x && point.y < best.y)) {
                    best = point;
                    best_score = 0.0;
                }
                continue;
            }
            var min_distance = 1e30;
            for (var known = 0u; known < count; known++) {
                let delta = point.xy - selected[known].xy;
                min_distance = min(min_distance, dot(delta, delta));
            }
            if (min_distance > best_score) {
                best = point;
                best_score = min_distance;
            }
        }
        if (best_score < 0.0 || (slot > 0u && best_score <= tolerance * tolerance)) {
            break;
        }
        selected[count] = best;
        let contact = ground_hit(vec3<f32>(best.xy, best.z * 0.5),
            vec3<f32>(0.0, 0.0, 1.0), -best.z);
        if (count == 0u) {
            ground_contacts[index] = contact;
        } else {
            ground_contacts[params.body_count + index * 3u + count - 1u] = contact;
        }
        count++;
    }
    if (count == 0u) { ground_contacts[index] = miss(); }
}

@compute @workgroup_size(64)
fn ground_contacts_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= params.body_count) { return; }
    for (var point = 0u; point < 3u; point++) {
        ground_contacts[params.body_count + index * 3u + point] = miss();
    }
    let ground = CollisionGroups(params.ground_memberships, params.ground_filter);
    if (!allows(collision_groups[index], ground)) {
        ground_contacts[index] = miss();
        return;
    }
    if (params.ground_enabled == 0u) {
        ground_contacts[index] = miss();
        return;
    }
    let center = states[index].position_inverse_mass.xyz;
    let half = params.ground_half_extent;
    if (shapes[index].kind.x == 0u) {
        let radius = shapes[index].dimensions.x;
        if (abs(center.x) > half + radius || abs(center.y) > half + radius
            || center.z - radius > bitcast<f32>(params.padding.y)) {
            ground_contacts[index] = miss();
            return;
        }
        ground_contacts[index] = ground_hit(
            vec3<f32>(center.xy, (center.z - radius) * 0.5),
            vec3<f32>(0.0, 0.0, 1.0),
            radius - center.z,
        );
        return;
    }
    if (shapes[index].kind.x == 2u) {
        let capsule = capsule_from(index);
        let bottom = select(capsule.start, capsule.end, capsule.end.z < capsule.start.z);
        let radius = capsule.radius;
        if (abs(bottom.x) > half + radius || abs(bottom.y) > half + radius
            || bottom.z - radius > bitcast<f32>(params.padding.y)) {
            ground_contacts[index] = miss();
            return;
        }
        ground_contacts[index] = ground_hit(
            vec3<f32>(bottom.xy, (bottom.z - radius) * 0.5),
            vec3<f32>(0.0, 0.0, 1.0),
            radius - bottom.z,
        );
        return;
    }
    if (shapes[index].kind.x == 5u || shapes[index].kind.x == 6u || shapes[index].kind.x == 7u) {
        convex_ground(index, half);
        return;
    }
    if (shapes[index].kind.x >= 3u) {
        let bottom = primitive_support(index, vec3<f32>(0.0, 0.0, -1.0));
        let radius = shapes[index].dimensions.x;
        let reach = shapes[index].dimensions.y + radius;
        if (abs(center.x) > half + reach || abs(center.y) > half + reach
            || bottom.z > bitcast<f32>(params.padding.y)) {
            ground_contacts[index] = miss();
            return;
        }
        ground_contacts[index] = ground_hit(
            vec3<f32>(bottom.xy, bottom.z * 0.5),
            vec3<f32>(0.0, 0.0, 1.0),
            -bottom.z,
        );
        return;
    }
    box_ground(index, half);
}
