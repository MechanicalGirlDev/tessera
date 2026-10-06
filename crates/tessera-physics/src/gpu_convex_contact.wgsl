struct Hull {
    ranges: vec4<u32>,
    center: vec4<f32>,
    axis_kind: vec4<f32>,
    params: vec4<f32>,
    edge_range: vec4<u32>,
}

struct Contact {
    point: vec4<f32>,
    normal: vec4<f32>,
    depth_hit: vec4<f32>,
}

struct Simplex {
    a: vec3<f32>,
    b: vec3<f32>,
    c: vec3<f32>,
    count: u32,
    direction: vec3<f32>,
    inside: bool,
}

@group(0) @binding(0) var<storage, read> hulls: array<Hull>;
@group(0) @binding(1) var<storage, read> vertices: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> normals: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> pairs: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read_write> contacts: array<Contact>;
@group(0) @binding(5) var<storage, read> edges: array<vec4<f32>>;

fn farthest(hull: Hull, direction: vec3<f32>) -> vec3<f32> {
    if (hull.axis_kind.w == 1.0) {
        let axis = hull.axis_kind.xyz;
        let axial = dot(direction, axis);
        let radial = direction - axis * axial;
        let radial_length_squared = dot(radial, radial);
        let rim = select(
            vec3<f32>(0.0),
            radial * (hull.params.y * inverseSqrt(max(radial_length_squared, 1e-20))),
            radial_length_squared > 1e-20,
        );
        return hull.center.xyz
            + axis * select(-hull.params.x, hull.params.x, axial >= 0.0)
            + rim;
    }
    if (hull.axis_kind.w == 2.0) {
        let axis = hull.axis_kind.xyz;
        let axial = dot(direction, axis);
        let radial = direction - axis * axial;
        let radial_length_squared = dot(radial, radial);
        let apex = hull.center.xyz + axis * hull.params.x;
        if (radial_length_squared <= 1e-20) {
            return select(hull.center.xyz - axis * hull.params.x, apex, axial >= 0.0);
        }
        let base = hull.center.xyz - axis * hull.params.x
            + radial * (hull.params.y * inverseSqrt(radial_length_squared));
        return select(base, apex, dot(apex, direction) > dot(base, direction));
    }
    var best = vertices[hull.ranges.x].xyz;
    var best_projection = dot(best, direction);
    for (var i = 1u; i < hull.ranges.y; i++) {
        let candidate = vertices[hull.ranges.x + i].xyz;
        let projection = dot(candidate, direction);
        if (projection > best_projection) {
            best_projection = projection;
            best = candidate;
        }
    }
    if (hull.center.w > 0.0) {
        let length_squared = dot(direction, direction);
        let unit = select(vec3<f32>(1.0, 0.0, 0.0), direction * inverseSqrt(max(length_squared, 1e-20)), length_squared > 1e-20);
        best += unit * hull.center.w;
    }
    return best;
}

fn support(a: Hull, b: Hull, direction: vec3<f32>) -> vec3<f32> {
    return farthest(a, direction) - farthest(b, -direction);
}

fn line_direction(a: vec3<f32>, b: vec3<f32>, toward: vec3<f32>) -> vec3<f32> {
    let edge = b - a;
    let perpendicular = cross(cross(edge, toward), edge);
    if (dot(perpendicular, perpendicular) > 1e-16) {
        return perpendicular;
    }
    let axis = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(edge.x) < 0.9 * length(edge));
    return cross(edge, axis);
}

fn expand(simplex: Simplex, point: vec3<f32>) -> Simplex {
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
    if (dot(face, toward) > 0.0) {
        return Simplex(a, b, c, 3u, face, false);
    }
    face = cross(c - a, d - a);
    if (dot(face, b - a) > 0.0) { face = -face; }
    if (dot(face, toward) > 0.0) {
        return Simplex(a, c, d, 3u, face, false);
    }
    face = cross(d - a, b - a);
    if (dot(face, c - a) > 0.0) { face = -face; }
    if (dot(face, toward) > 0.0) {
        return Simplex(a, d, b, 3u, face, false);
    }
    return Simplex(a, b, c, 4u, toward, true);
}

fn intersects(a: Hull, b: Hull) -> bool {
    var direction = a.center.xyz - b.center.xyz;
    if (dot(direction, direction) < 1e-16) { direction = vec3<f32>(1.0, 0.0, 0.0); }
    let first = support(a, b, direction);
    var simplex = Simplex(first, vec3<f32>(0.0), vec3<f32>(0.0), 1u, -first, false);
    for (var iteration = 0u; iteration < 32u; iteration++) {
        if (dot(simplex.direction, simplex.direction) < 1e-16) { return true; }
        let point = support(a, b, simplex.direction);
        let projection = dot(point, simplex.direction);
        if (projection < -1e-6) { return false; }
        if (dot(point - simplex.a, point - simplex.a) < 1e-14 ||
            (simplex.count > 1u && dot(point - simplex.b, point - simplex.b) < 1e-14) ||
            (simplex.count > 2u && dot(point - simplex.c, point - simplex.c) < 1e-14)) {
            return abs(projection) < 1e-6;
        }
        simplex = expand(simplex, point);
        if (simplex.inside) { return true; }
    }
    return false;
}

fn axis_depth(a: Hull, b: Hull, candidate: vec3<f32>, current: vec4<f32>) -> vec4<f32> {
    let axis = normalize(candidate);
    let min_a = dot(farthest(a, -axis), axis);
    let max_a = dot(farthest(a, axis), axis);
    let min_b = dot(farthest(b, -axis), axis);
    let max_b = dot(farthest(b, axis), axis);
    let forward = max_a - min_b;
    let backward = max_b - min_a;
    let depth = min(forward, backward);
    if (depth < current.w) {
        return vec4<f32>(axis * select(-1.0, 1.0, forward < backward), depth);
    }
    return current;
}

struct SegmentPoints {
    a: vec3<f32>,
    b: vec3<f32>,
}

fn closest_segment_points(a: Hull, b: Hull) -> SegmentPoints {
    let a0 = vertices[a.ranges.x].xyz;
    let a1 = vertices[a.ranges.x + select(0u, 1u, a.ranges.y > 1u)].xyz;
    let b0 = vertices[b.ranges.x].xyz;
    let b1 = vertices[b.ranges.x + select(0u, 1u, b.ranges.y > 1u)].xyz;
    let da = a1 - a0;
    let db = b1 - b0;
    let offset = a0 - b0;
    let aa = dot(da, da);
    let bb = dot(db, db);
    let ab = dot(da, db);
    let ar = dot(da, offset);
    let br = dot(db, offset);
    let denominator = aa * bb - ab * ab;
    var s = 0.0;
    if (denominator > 1e-20) {
        s = clamp((ab * br - ar * bb) / denominator, 0.0, 1.0);
    }
    var t = 0.0;
    if (bb > 1e-20) {
        t = clamp((ab * s + br) / bb, 0.0, 1.0);
    }
    if (aa > 1e-20) {
        s = clamp((ab * t - ar) / aa, 0.0, 1.0);
    }
    if (bb > 1e-20) {
        t = clamp((ab * s + br) / bb, 0.0, 1.0);
    }
    return SegmentPoints(a0 + da * s, b0 + db * t);
}

fn rounded_segment_contact(a: Hull, b: Hull) -> Contact {
    let points = closest_segment_points(a, b);
    let delta = points.b - points.a;
    let distance = length(delta);
    let radii = a.center.w + b.center.w;
    if (distance > radii + 1e-6) {
        return Contact(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0, 0.0, 3.0, 0.0));
    }
    var normal = vec3<f32>(1.0, 0.0, 0.0);
    if (distance > 1e-12) {
        normal = delta / distance;
    } else {
        let centers = b.center.xyz - a.center.xyz;
        let center_distance = length(centers);
        if (center_distance > 1e-12) {
            normal = centers / center_distance;
        }
    }
    let witness_a = points.a + normal * a.center.w;
    let witness_b = points.b - normal * b.center.w;
    return Contact(
        vec4<f32>((witness_a + witness_b) * 0.5, 0.0),
        vec4<f32>(normal, 0.0),
        vec4<f32>(max(radii - distance, 0.0), 1.0, 3.0, 0.0),
    );
}

fn analytic_sphere_contact(shape: Hull, sphere: Hull, sphere_first: bool) -> Contact {
    let axis = shape.axis_kind.xyz;
    let relative = sphere.center.xyz - shape.center.xyz;
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
    let half_height = shape.params.x;
    let radius = shape.params.y;
    var signed_distance: f32;
    var outward: vec3<f32>;
    if (shape.axis_kind.w == 1.0) {
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
        let delta = point - closest;
        let distance = length(delta);
        let edge_epsilon = max(half_height, radius) * 1e-6;
        let inside = height >= -half_height - edge_epsilon
            && height <= half_height + edge_epsilon
            && radial <= radius * (half_height - height) / (2.0 * half_height) + edge_epsilon;
        let base_normal = vec2<f32>(0.0, -1.0);
        let side_normal = normalize(vec2<f32>(2.0 * half_height, radius));
        let near_corner = base_distance_squared <= edge_epsilon * edge_epsilon
            && side_distance_squared <= edge_epsilon * edge_epsilon;
        var feature_normal = select(side_normal, base_normal, use_base);
        if (near_corner) { feature_normal = normalize(base_normal + side_normal); }
        let normal_2d = select(delta / max(distance, 1e-12),
            feature_normal, inside || distance <= edge_epsilon);
        outward = radial_direction * normal_2d.x + axis * normal_2d.y;
        signed_distance = select(distance, -distance, inside);
    }
    if (signed_distance > sphere.center.w + 1e-6) {
        return Contact(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0, 0.0, 4.0, 0.0));
    }
    let surface = sphere.center.xyz - outward * signed_distance;
    let sphere_witness = sphere.center.xyz - outward * sphere.center.w;
    let pair_normal = select(outward, -outward, sphere_first);
    return Contact(
        vec4<f32>((surface + sphere_witness) * 0.5, 0.0),
        vec4<f32>(pair_normal, 0.0),
        vec4<f32>(max(sphere.center.w - signed_distance, 0.0), 1.0, 4.0, 0.0),
    );
}

fn mesh_sphere_contact(mesh: Hull, sphere: Hull, sphere_first: bool) -> Contact {
    let offset = mesh.ranges.x;
    let a = (vertices[offset].xyz + vertices[offset + 3u].xyz) * 0.5;
    let b = (vertices[offset + 1u].xyz + vertices[offset + 4u].xyz) * 0.5;
    let c = (vertices[offset + 2u].xyz + vertices[offset + 5u].xyz) * 0.5;
    let ab = b - a;
    let bc = c - b;
    let ca = a - c;
    let normal = normalize(cross(ab, c - a));
    let center = sphere.center.xyz;
    let signed_distance = dot(center - a, normal);
    let projected = center - normal * signed_distance;
    let scale = max(1.0, max(dot(ab, ab), max(dot(bc, bc), dot(ca, ca))));
    let face = !(dot(normal, cross(ab, projected - a)) < -scale * 1e-6 ||
        dot(normal, cross(bc, projected - b)) < -scale * 1e-6 ||
        dot(normal, cross(ca, projected - c)) < -scale * 1e-6);
    // Keep in sync with mesh::TRIANGLE_HALF_THICKNESS.
    let half_thickness = 1e-4;
    var surface: vec3<f32>;
    var outward: vec3<f32>;
    var depth: f32;
    if (face) {
        outward = normal * select(-1.0, 1.0, signed_distance >= 0.0);
        surface = projected + outward * half_thickness;
        depth = sphere.center.w + half_thickness - abs(signed_distance);
    } else {
        var closest = a;
        var best_distance = 1e30;
        for (var i = 0u; i < 3u; i++) {
            var start: vec3<f32>;
            var end: vec3<f32>;
            if (i == 0u) { start = a; end = b; }
            else if (i == 1u) { start = b; end = c; }
            else { start = c; end = a; }
            let edge = end - start;
            let t = clamp(dot(projected - start, edge) / dot(edge, edge), 0.0, 1.0);
            let candidate = start + edge * t;
            let distance_squared = dot(candidate - projected, candidate - projected);
            if (distance_squared < best_distance) {
                best_distance = distance_squared;
                closest = candidate;
            }
        }
        surface = closest + normal * clamp(signed_distance, -half_thickness, half_thickness);
        let delta = center - surface;
        let distance = length(delta);
        outward = select(normal * select(-1.0, 1.0, signed_distance >= 0.0),
            delta / max(distance, 1e-12), distance > 1e-12);
        depth = sphere.center.w - distance;
    }
    if (depth < 0.0) {
        return Contact(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0, 0.0, select(2.0, 1.0, face), 0.0));
    }
    let sphere_witness = center - outward * sphere.center.w;
    let point = (surface + sphere_witness) * 0.5;
    let pair_normal = select(outward, -outward, sphere_first);
    return Contact(
        vec4<f32>(point, 0.0),
        vec4<f32>(pair_normal, 0.0),
        vec4<f32>(max(depth, 0.0), 1.0, select(2.0, 1.0, face), 0.0),
    );
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&pairs)) { return; }
    let pair = pairs[id.x];
    let a = hulls[pair.x];
    let b = hulls[pair.y];
    let miss = Contact(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
    if (a.params.z > 0.5 && b.ranges.y == 1u && b.center.w > 0.0) {
        contacts[id.x] = mesh_sphere_contact(a, b, false);
        return;
    }
    if (b.params.z > 0.5 && a.ranges.y == 1u && a.center.w > 0.0) {
        contacts[id.x] = mesh_sphere_contact(b, a, true);
        return;
    }
    if (a.axis_kind.w > 0.0 && b.axis_kind.w == 0.0
        && b.ranges.y == 1u && b.center.w > 0.0) {
        contacts[id.x] = analytic_sphere_contact(a, b, false);
        return;
    }
    if (b.axis_kind.w > 0.0 && a.axis_kind.w == 0.0
        && a.ranges.y == 1u && a.center.w > 0.0) {
        contacts[id.x] = analytic_sphere_contact(b, a, true);
        return;
    }
    if (a.axis_kind.w == 0.0 && b.axis_kind.w == 0.0
        && a.ranges.y <= 2u && b.ranges.y <= 2u
        && (a.center.w > 0.0 || b.center.w > 0.0)) {
        contacts[id.x] = rounded_segment_contact(a, b);
        return;
    }
    if (!intersects(a, b)) {
        contacts[id.x] = miss;
        return;
    }
    var best = vec4<f32>(1.0, 0.0, 0.0, 1e30);
    var feature = 0.0;
    var selected_a = 0u;
    var selected_b = 0u;
    for (var i = 0u; i < a.ranges.w; i++) {
        let next = axis_depth(a, b, normals[a.ranges.z + i].xyz, best);
        if (next.w < best.w) {
            feature = 6.0;
            selected_a = i;
        }
        best = next;
    }
    for (var i = 0u; i < b.ranges.w; i++) {
        let next = axis_depth(a, b, normals[b.ranges.z + i].xyz, best);
        if (next.w < best.w) {
            feature = 7.0;
            selected_a = i;
        }
        best = next;
    }
    // Keep the bounded edge work in sync with convex_edge_manifold on the CPU.
    if (a.axis_kind.w == 0.0 && b.axis_kind.w == 0.0
        && a.center.w == 0.0 && b.center.w == 0.0
        && a.edge_range.y > 0u && b.edge_range.y > 0u
        && a.edge_range.y <= 160u / b.edge_range.y
        && a.ranges.y + b.ranges.y <= 160u / (a.edge_range.y * b.edge_range.y)) {
        for (var i = 0u; i < a.edge_range.y; i++) {
            let direction_a = edges[a.edge_range.x + i].xyz;
            for (var j = 0u; j < b.edge_range.y; j++) {
                let candidate = cross(direction_a, edges[b.edge_range.x + j].xyz);
                if (dot(candidate, candidate) > 1e-10) {
                    let next = axis_depth(a, b, candidate, best);
                    if (next.w < best.w) {
                        if (next.w + 1e-6 < best.w) {
                            feature = 5.0;
                        } else if (feature == 5.0) {
                            feature = 0.0;
                        }
                        if (feature == 5.0) {
                            selected_a = i;
                            selected_b = j;
                        }
                    }
                    best = next;
                }
            }
        }
    }
    if (b.center.w > 0.0) {
        for (var i = 0u; i < a.ranges.y; i++) {
            let candidate = b.center.xyz - vertices[a.ranges.x + i].xyz;
            if (dot(candidate, candidate) > 1e-16) {
                let next = axis_depth(a, b, candidate, best);
                if (next.w + 1e-6 < best.w) { feature = 0.0; }
                best = next;
            }
        }
    }
    if (a.center.w > 0.0) {
        for (var i = 0u; i < b.ranges.y; i++) {
            let candidate = a.center.xyz - vertices[b.ranges.x + i].xyz;
            if (dot(candidate, candidate) > 1e-16) {
                let next = axis_depth(a, b, candidate, best);
                if (next.w + 1e-6 < best.w) { feature = 0.0; }
                best = next;
            }
        }
    }
    let center_axis = b.center.xyz - a.center.xyz;
    if (dot(center_axis, center_axis) > 1e-16) {
        let next = axis_depth(a, b, center_axis, best);
        if (next.w + 1e-6 < best.w) {
            feature = 0.0;
        }
        best = next;
    }
    let has_analytic = a.axis_kind.w > 0.0 || b.axis_kind.w > 0.0;
    if ((!has_analytic && best.w < -1e-5) || best.w > 1e20) {
        contacts[id.x] = miss;
        return;
    }
    let witness_a = farthest(a, best.xyz);
    let witness_b = farthest(b, -best.xyz);
    let midpoint = (a.center.xyz + b.center.xyz) * 0.5;
    let witness_midpoint = (witness_a + witness_b) * 0.5;
    let point = midpoint + best.xyz * dot(witness_midpoint - midpoint, best.xyz);
    contacts[id.x] = Contact(
        vec4<f32>(point, f32(selected_a)),
        vec4<f32>(best.xyz, f32(selected_b)),
        vec4<f32>(max(best.w, 0.0), 1.0, feature, 0.0),
    );
}
