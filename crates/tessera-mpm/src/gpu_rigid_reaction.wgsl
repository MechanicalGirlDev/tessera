struct RigidState {
    position_inverse_mass: vec4<f32>,
    orientation: vec4<f32>,
    linear_velocity: vec4<f32>,
    angular_velocity: vec4<f32>,
    inverse_inertia_sleep: vec4<f32>,
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

struct ReactionParams {
    particle_count: u32,
    obstacle_count: u32,
    integrate_pose: u32,
    dt: f32,
    gravity: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> transfers: array<Transfer>;
@group(0) @binding(1) var<storage, read> mappings: array<Mapping>;
@group(0) @binding(2) var<storage, read> obstacles: array<Obstacle>;
@group(0) @binding(3) var<storage, read_write> rigid_states: array<RigidState>;
@group(0) @binding(4) var<uniform> params: ReactionParams;

fn rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

fn quat_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz),
                     a.w * b.w - dot(a.xyz, b.xyz));
}

fn quat_from_rotation(rotation: vec3<f32>) -> vec4<f32> {
    let angle = length(rotation);
    if angle < 1e-5 {
        return normalize(vec4<f32>(0.5 * rotation, 1.0));
    }
    let half_angle = 0.5 * angle;
    return vec4<f32>(rotation * (sin(half_angle) / angle), cos(half_angle));
}

@compute @workgroup_size(64)
fn apply(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let body_index = invocation.x;
    if body_index >= arrayLength(&rigid_states) {
        return;
    }
    var state = rigid_states[body_index];
    let inverse_mass = state.position_inverse_mass.w;
    if inverse_mass <= 0.0 {
        return;
    }
    var linear_impulse = vec3<f32>(0.0);
    var angular_impulse = vec3<f32>(0.0);
    var mapped = false;
    for (var obstacle_index = 0u; obstacle_index < params.obstacle_count; obstacle_index++) {
        if mappings[obstacle_index].body.x != body_index {
            continue;
        }
        mapped = true;
        let reaction = transfers[params.particle_count + obstacle_index];
        let impulse = reaction.position.xyz;
        linear_impulse += impulse;
        angular_impulse += reaction.velocity.xyz
            + cross(obstacles[obstacle_index].center_radius.xyz - state.position_inverse_mass.xyz, impulse);
    }
    if !all(abs(linear_impulse) < vec3<f32>(1e30))
        || !all(abs(angular_impulse) < vec3<f32>(1e30)) {
        return;
    }
    if !mapped {
        return;
    }
    if dot(linear_impulse, linear_impulse) == 0.0
        && dot(angular_impulse, angular_impulse) == 0.0
        && params.integrate_pose == 0u {
        return;
    }
    let conjugate = vec4<f32>(-state.orientation.xyz, state.orientation.w);
    let torque_body = rotate(conjugate, angular_impulse);
    let angular_delta = rotate(
        state.orientation,
        torque_body * state.inverse_inertia_sleep.xyz
    );
    state.linear_velocity = vec4<f32>(
        state.linear_velocity.xyz + linear_impulse * inverse_mass,
        state.linear_velocity.w
    );
    state.angular_velocity = vec4<f32>(
        state.angular_velocity.xyz + angular_delta,
        state.angular_velocity.w
    );
    state.inverse_inertia_sleep.w = 0.0;
    if params.integrate_pose != 0u {
        state.linear_velocity = vec4<f32>(
            state.linear_velocity.xyz + params.gravity.xyz * params.dt,
            state.linear_velocity.w
        );
    }
    if params.gravity.w > 0.0 {
        let speed = length(state.linear_velocity.xyz);
        if speed > params.gravity.w {
            state.linear_velocity = vec4<f32>(
                state.linear_velocity.xyz * (params.gravity.w / speed),
                state.linear_velocity.w
            );
        }
    }
    if params.integrate_pose != 0u {
        state.position_inverse_mass = vec4<f32>(
            state.position_inverse_mass.xyz + state.linear_velocity.xyz * params.dt,
            inverse_mass
        );
        state.orientation = normalize(quat_mul(
            quat_from_rotation(state.angular_velocity.xyz * params.dt),
            state.orientation
        ));
    }
    rigid_states[body_index] = state;
}
