struct Pose {
    position: vec4<f32>,
    orientation: vec4<f32>,
};

struct Shape {
    index_kind: vec4<u32>,
    a: vec4<f32>,
    b: vec4<f32>,
    orientation: vec4<f32>,
};

struct Aabb {
    lower: vec4<f32>,
    upper: vec4<f32>,
};

struct SweepMeta {
    indices: vec4<u32>,
    center_radius: vec4<f32>,
    center_of_mass_dt: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> poses: array<Pose>;
@group(0) @binding(1) var<storage, read> shapes: array<Shape>;
@group(0) @binding(2) var<storage, read_write> bounds: array<Aabb>;
@group(0) @binding(3) var<storage, read> velocities: array<f32>;
@group(0) @binding(4) var<storage, read> link_terms: array<f32>;
@group(0) @binding(5) var<storage, read> sweep_metadata: array<SweepMeta>;
@group(0) @binding(6) var<storage, read> accelerations: array<f32>;

fn rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

fn multiply(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz),
        a.w * b.w - dot(a.xyz, b.xyz));
}

fn world_bounds(shape: Shape, pose: Pose) -> Aabb {
    var lo: vec3<f32>;
    var hi: vec3<f32>;
    if (shape.index_kind.y == 0u) {
        let center = pose.position.xyz + rotate(pose.orientation, shape.a.xyz);
        lo = center - vec3<f32>(shape.a.w);
        hi = center + vec3<f32>(shape.a.w);
    } else if (shape.index_kind.y == 1u) {
        let a = pose.position.xyz + rotate(pose.orientation, shape.a.xyz);
        let b = pose.position.xyz + rotate(pose.orientation, shape.b.xyz);
        lo = min(a, b) - vec3<f32>(shape.a.w);
        hi = max(a, b) + vec3<f32>(shape.a.w);
    } else {
        let center = pose.position.xyz + rotate(pose.orientation, shape.a.xyz);
        let q = multiply(pose.orientation, shape.orientation);
        let x = rotate(q, vec3<f32>(1.0, 0.0, 0.0));
        let y = rotate(q, vec3<f32>(0.0, 1.0, 0.0));
        let z = rotate(q, vec3<f32>(0.0, 0.0, 1.0));
        let extent = abs(x) * shape.b.x + abs(y) * shape.b.y + abs(z) * shape.b.z;
        lo = center - extent;
        hi = center + extent;
    }
    // Outward padding keeps rounded bounds conservative for the narrow phase.
    let magnitude = max(abs(lo), abs(hi));
    let scale = max(1.0, max(magnitude.x, max(magnitude.y, magnitude.z)));
    let padding = vec3<f32>(1e-6 * scale);
    return Aabb(vec4<f32>(lo - padding, 0.0), vec4<f32>(hi + padding, 0.0));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let id = invocation.x;
    if (id >= arrayLength(&shapes)) { return; }
    let shape = shapes[id];
    bounds[id] = world_bounds(shape, poses[shape.index_kind.x]);
}

@compute @workgroup_size(64)
fn swept(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let id = invocation.x;
    if (id >= arrayLength(&shapes)) { return; }
    let shape = shapes[id];
    let pose = poses[shape.index_kind.x];
    let metadata = sweep_metadata[id];
    let current = world_bounds(shape, pose);
    let n = metadata.indices.z;
    let base = metadata.indices.x + 10u;
    let coordinates = metadata.indices.y;
    var linear = vec3<f32>(0.0);
    var angular = vec3<f32>(0.0);
    var linear_acceleration = vec3<f32>(0.0);
    var angular_acceleration = vec3<f32>(0.0);
    for (var column = 0u; column < n; column++) {
        let rate = velocities[coordinates + column];
        let acceleration = accelerations[coordinates + column];
        let linear_column = vec3<f32>(
            link_terms[base + column],
            link_terms[base + n + column],
            link_terms[base + 2u * n + column]);
        let angular_column = vec3<f32>(
            link_terms[base + 3u * n + column],
            link_terms[base + 4u * n + column],
            link_terms[base + 5u * n + column]);
        linear += linear_column * rate;
        angular += angular_column * rate;
        linear_acceleration += linear_column * acceleration;
        angular_acceleration += angular_column * acceleration;
    }
    if (!all(abs(linear) < vec3<f32>(1e30))
        || !all(abs(angular) < vec3<f32>(1e30))
        || !all(abs(linear_acceleration) < vec3<f32>(1e30))
        || !all(abs(angular_acceleration) < vec3<f32>(1e30))) {
        bounds[id] = current;
        return;
    }
    let center = pose.position.xyz
        + rotate(pose.orientation, metadata.center_radius.xyz);
    let center_of_mass = pose.position.xyz
        + rotate(pose.orientation, metadata.center_of_mass_dt.xyz);
    let offset = center - center_of_mass;
    let center_velocity = linear + cross(angular, offset);
    let center_acceleration = linear_acceleration + cross(angular_acceleration, offset);
    let dt = metadata.center_of_mass_dt.w;
    let displacement = center_velocity * dt + 0.5 * center_acceleration * dt * dt;
    // Rotation can move the shape and its center away from the linear forecast.
    let angular_padding = (length(angular) * dt
        + 0.5 * length(angular_acceleration) * dt * dt)
        * (metadata.center_radius.w + length(offset));
    let acceleration_padding = 0.5 * length(center_acceleration) * dt * dt;
    let padding = vec3<f32>(angular_padding + acceleration_padding);
    let lo = min(current.lower.xyz, current.lower.xyz + displacement)
        - padding;
    let hi = max(current.upper.xyz, current.upper.xyz + displacement)
        + padding;
    if (!all(abs(lo) < vec3<f32>(1e30))
        || !all(abs(hi) < vec3<f32>(1e30))) {
        bounds[id] = current;
        return;
    }
    bounds[id] = Aabb(vec4<f32>(lo, 0.0), vec4<f32>(hi, 0.0));
}
