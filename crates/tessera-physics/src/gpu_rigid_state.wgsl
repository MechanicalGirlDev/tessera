struct RigidState {
    position_inverse_mass: vec4<f32>,
    orientation: vec4<f32>,
    linear_velocity: vec4<f32>,
    angular_velocity: vec4<f32>,
    inverse_inertia_sleep: vec4<f32>,
}

struct RigidForces {
    force: vec4<f32>,
    torque: vec4<f32>,
}

struct StepParams {
    dt_gravity: vec4<f32>,
    count: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
}

@group(0) @binding(0) var<storage, read_write> states: array<RigidState>;
@group(0) @binding(1) var<storage, read_write> forces: array<RigidForces>;
@group(0) @binding(2) var<storage, read> params: StepParams;
@group(0) @binding(3) var<storage, read_write> frame_forces: array<RigidForces>;

struct KinematicTranslation {
    origin_elapsed: vec4<f32>,
    expected_position: vec4<f32>,
    linear_velocity: vec4<f32>,
}
@group(0) @binding(4) var<storage, read_write> kinematic_translation: array<KinematicTranslation>;

fn quat_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz),
                     a.w * b.w - dot(a.xyz, b.xyz));
}

fn quat_from_rotation(rotation: vec3<f32>) -> vec4<f32> {
    let angle = length(rotation);
    if (angle < 1e-5) {
        return normalize(vec4<f32>(0.5 * rotation, 1.0));
    }
    let half_angle = 0.5 * angle;
    return vec4<f32>(rotation * (sin(half_angle) / angle), cos(half_angle));
}

fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

fn integrate_velocity(input: RigidState, applied: RigidForces) -> RigidState {
    var state = input;
    let inverse_mass = state.position_inverse_mass.w;
    if (inverse_mass == 0.0) { return state; }
    if (state.inverse_inertia_sleep.w != 0.0) {
        if (dot(applied.force.xyz, applied.force.xyz) == 0.0 &&
            dot(applied.torque.xyz, applied.torque.xyz) == 0.0) { return state; }
        state.inverse_inertia_sleep = vec4<f32>(state.inverse_inertia_sleep.xyz, 0.0);
    }
    let dt = params.dt_gravity.x;
    state.linear_velocity = vec4<f32>(
        state.linear_velocity.xyz +
            (params.dt_gravity.yzw + applied.force.xyz * inverse_mass) * dt,
        0.0,
    );
    let max_speed = bitcast<f32>(params._padding0);
    if max_speed > 0.0 {
        let speed = length(state.linear_velocity.xyz);
        if speed > max_speed {
            state.linear_velocity = vec4<f32>(state.linear_velocity.xyz * (max_speed / speed), 0.0);
        }
    }

    let q = state.orientation;
    let conjugate = vec4<f32>(-q.xyz, q.w);
    let omega_body = quat_rotate(conjugate, state.angular_velocity.xyz);
    let torque_body = quat_rotate(conjugate, applied.torque.xyz);
    let inverse_inertia = state.inverse_inertia_sleep.xyz;
    let inertia_omega = vec3<f32>(
        select(0.0, omega_body.x / inverse_inertia.x, inverse_inertia.x > 0.0),
        select(0.0, omega_body.y / inverse_inertia.y, inverse_inertia.y > 0.0),
        select(0.0, omega_body.z / inverse_inertia.z, inverse_inertia.z > 0.0),
    );
    let gyro = cross(omega_body, inertia_omega);
    let omega_next = omega_body + (torque_body - gyro) * inverse_inertia * dt;
    let omega_world = quat_rotate(q, omega_next);
    state.angular_velocity = vec4<f32>(omega_world, 0.0);

    return state;
}
fn integrate_position(input: RigidState, index: u32) -> RigidState {
    var state = input;
    if ((state.position_inverse_mass.w == 0.0 && state.linear_velocity.w != 1.0)
        || state.inverse_inertia_sleep.w != 0.0) {
        kinematic_translation[index].origin_elapsed.w = 0.0;
        return state;
    }
    let dt = params.dt_gravity.x;
    if (state.position_inverse_mass.w == 0.0) {
        var interval = kinematic_translation[index];
        if (interval.origin_elapsed.w == 0.0
            || any(interval.expected_position.xyz != state.position_inverse_mass.xyz)
            || any(interval.linear_velocity.xyz != state.linear_velocity.xyz)) {
            interval.origin_elapsed = vec4<f32>(state.position_inverse_mass.xyz, 0.0);
            interval.linear_velocity = state.linear_velocity;
        }
        interval.origin_elapsed.w += dt;
        state.position_inverse_mass = vec4<f32>(interval.origin_elapsed.xyz
            + interval.linear_velocity.xyz * interval.origin_elapsed.w, 0.0);
        interval.expected_position = state.position_inverse_mass;
        kinematic_translation[index] = interval;
    } else {
        kinematic_translation[index].origin_elapsed.w = 0.0;
        state.position_inverse_mass = vec4<f32>(state.position_inverse_mass.xyz + state.linear_velocity.xyz * dt,
            state.position_inverse_mass.w);
    }
    state.orientation = normalize(quat_mul(quat_from_rotation(state.angular_velocity.xyz * dt), state.orientation));
    return state;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.count) { return; }
    let applied = forces[id.x];
    forces[id.x] = RigidForces(vec4<f32>(0.0), vec4<f32>(0.0));
    states[id.x] = integrate_position(integrate_velocity(states[id.x], applied), id.x);
}
@compute @workgroup_size(64)
fn capture_forces(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.count) { return; }
    frame_forces[id.x] = forces[id.x];
    forces[id.x] = RigidForces(vec4<f32>(0.0), vec4<f32>(0.0));
}
@compute @workgroup_size(64)
fn velocity(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.count) { return; }
    states[id.x] = integrate_velocity(states[id.x], frame_forces[id.x]);
}
@compute @workgroup_size(64)
fn position(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.count) { return; }
    states[id.x] = integrate_position(states[id.x], id.x);
}
@compute @workgroup_size(64)
fn clamp_linear_speed(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.count { return; }
    var state = states[id.x];
    if state.position_inverse_mass.w == 0.0 { return; }
    let max_speed = bitcast<f32>(params._padding0);
    let speed = length(state.linear_velocity.xyz);
    if speed > max_speed {
        state.linear_velocity = vec4<f32>(state.linear_velocity.xyz * (max_speed / speed), 0.0);
        states[id.x] = state;
    }
}
