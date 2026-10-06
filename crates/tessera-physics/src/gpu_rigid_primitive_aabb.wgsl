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

struct Aabb {
    lower: vec4<f32>,
    upper: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> states: array<RigidState>;
@group(0) @binding(1) var<storage, read> shapes: array<Shape>;
@group(0) @binding(2) var<storage, read> convex_vertices: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> bounds: array<Aabb>;

fn rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= arrayLength(&bounds)) { return; }
    let state = states[index];
    let shape = shapes[index];
    let center = state.position_inverse_mass.xyz;
    let q = state.orientation;
    var lower = center;
    var upper = center;

    if (shape.kind.x == 0u) {
        let extent = vec3<f32>(shape.dimensions.x);
        lower -= extent;
        upper += extent;
    } else if (shape.kind.x == 1u) {
        let x = rotate(q, vec3<f32>(1.0, 0.0, 0.0));
        let y = rotate(q, vec3<f32>(0.0, 1.0, 0.0));
        let z = rotate(q, vec3<f32>(0.0, 0.0, 1.0));
        let extent = abs(x) * shape.dimensions.x
            + abs(y) * shape.dimensions.y
            + abs(z) * shape.dimensions.z;
        lower -= extent;
        upper += extent;
    } else if (shape.kind.x == 2u) {
        let axis = rotate(q, vec3<f32>(0.0, 0.0, 1.0));
        let extent = abs(axis) * shape.dimensions.y + vec3<f32>(shape.dimensions.x);
        lower -= extent;
        upper += extent;
    } else if (shape.kind.x == 3u || shape.kind.x == 4u) {
        let axis = rotate(q, vec3<f32>(0.0, 0.0, 1.0));
        let radial = sqrt(max(vec3<f32>(0.0), vec3<f32>(1.0) - axis * axis))
            * shape.dimensions.x;
        let half_axis = axis * shape.dimensions.y;
        if (shape.kind.x == 3u) {
            let extent = abs(half_axis) + radial;
            lower -= extent;
            upper += extent;
        } else {
            let base = center - half_axis;
            let apex = center + half_axis;
            lower = min(base - radial, apex);
            upper = max(base + radial, apex);
        }
    } else if (shape.kind.x == 5u || shape.kind.x == 6u || shape.kind.x == 7u) {
        let first = shape.kind.y;
        let vertex_count = shape.kind.z;
        let initial = center + rotate(q, convex_vertices[first].xyz);
        lower = initial;
        upper = initial;
        for (var vertex = 1u; vertex < vertex_count; vertex++) {
            let point = center + rotate(q, convex_vertices[first + vertex].xyz);
            lower = min(lower, point);
            upper = max(upper, point);
        }
    }

    let margin = 1e-5 * max(1.0, length(upper - lower));
    bounds[index] = Aabb(
        vec4<f32>(lower - vec3<f32>(margin), 0.0),
        vec4<f32>(upper + vec3<f32>(margin), 0.0),
    );
}
