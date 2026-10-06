struct RigidState {
    position_inverse_mass: vec4<f32>,
    orientation: vec4<f32>,
    linear_velocity: vec4<f32>,
    angular_velocity: vec4<f32>,
    inverse_inertia_sleep: vec4<f32>,
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

struct Mapping {
    body: vec4<u32>,
    local_center: vec4<f32>,
    local_orientation: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> rigid_states: array<RigidState>;
@group(0) @binding(1) var<storage, read> mappings: array<Mapping>;
@group(0) @binding(2) var<storage, read_write> obstacles: array<Obstacle>;

fn rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

fn quaternion_multiply(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz),
        a.w * b.w - dot(a.xyz, b.xyz),
    );
}

@compute @workgroup_size(64)
fn update(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= arrayLength(&mappings) { return; }
    let mapping = mappings[id.x];
    if mapping.body.x == 0xffffffffu { return; }
    let state = rigid_states[mapping.body.x];
    let offset = rotate(state.orientation, mapping.local_center.xyz);
    var obstacle = obstacles[id.x];
    obstacle.center_radius = vec4<f32>(
        state.position_inverse_mass.xyz + offset,
        obstacle.center_radius.w,
    );
    obstacle.orientation = quaternion_multiply(state.orientation, mapping.local_orientation);
    obstacle.linear_velocity_friction = vec4<f32>(
        state.linear_velocity.xyz + cross(state.angular_velocity.xyz, offset),
        obstacle.linear_velocity_friction.w,
    );
    obstacle.angular_velocity = vec4<f32>(state.angular_velocity.xyz, obstacle.angular_velocity.w);
    obstacles[id.x] = obstacle;
}
