struct RigidState {
    position_inverse_mass: vec4<f32>,
    orientation: vec4<f32>,
    linear_velocity: vec4<f32>,
    angular_velocity: vec4<f32>,
    inverse_inertia_sleep: vec4<f32>,
}

struct Contact {
    point: vec4<f32>,
    normal: vec4<f32>,
    depth_hit: vec4<f32>,
}
struct Pair { a: u32, b: u32, }

struct SleepParams {
    thresholds: vec4<f32>,
    counts: vec4<u32>,
}

struct SleepState {
    previous_position_idle: vec4<f32>,
    flags: vec4<u32>,
}

@group(0) @binding(0) var<storage, read_write> states: array<RigidState>;
@group(0) @binding(1) var<storage, read> pair_contacts: array<Contact>;
@group(0) @binding(2) var<storage, read> ground_contacts: array<Contact>;
@group(0) @binding(3) var<storage, read> body_pair_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> body_pair_indices: array<u32>;
@group(0) @binding(5) var<storage, read_write> sleep_states: array<SleepState>;
@group(0) @binding(6) var<uniform> params: SleepParams;
@group(0) @binding(7) var<storage, read> pairs: array<Pair>;
@group(0) @binding(8) var<storage, read> moving_kinematic: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let body = id.x;
    if (body >= params.counts.x) { return; }
    var state = states[body];
    var sleep_state = sleep_states[body];
    if (state.position_inverse_mass.w == 0.0) {
        sleep_states[body] = SleepState(vec4<f32>(0.0), vec4<u32>(0u));
        return;
    }
    let delta_position = state.position_inverse_mass.xyz - sleep_state.previous_position_idle.xyz;
    let linear_displacement_sq = dot(delta_position, delta_position);
    let displacement_threshold_sq = params.thresholds.y * params.thresholds.x * params.thresholds.x;
    let previous_valid = sleep_state.flags.x != 0u;
    sleep_state.previous_position_idle = vec4<f32>(
        state.position_inverse_mass.xyz, sleep_state.previous_position_idle.w);
    sleep_state.flags.x = 1u;
    if (params.counts.z == 0u) {
        sleep_state.previous_position_idle.w = 0.0;
        sleep_states[body] = sleep_state;
        state.inverse_inertia_sleep.w = 0.0;
        states[body] = state;
        return;
    }
    var supported = false;
    let stride = params.counts.w;
    if (params.counts.y != 0u) {
        for (var row = 0u; row < stride; row++) {
            var contact_index = body;
            if (row > 0u) { contact_index = params.counts.x + body * (stride - 1u) + row - 1u; }
            supported = supported || ground_contacts[contact_index].depth_hit.y != 0.0;
        }
    }
    var prescribed_contact = moving_kinematic[body] != 0u;
    for (var slot = body_pair_offsets[body]; slot < body_pair_offsets[body + 1u]; slot++) {
        let pair_index = body_pair_indices[slot];
        var hit = false;
        for (var row = 0u; row < stride; row++) {
            var contact_index = pair_index;
            if (row > 0u) { contact_index = arrayLength(&pairs) + pair_index * (stride - 1u) + row - 1u; }
            hit = hit || pair_contacts[contact_index].depth_hit.y != 0.0;
        }
        if (hit) {
            supported = true;
            let pair = pairs[pair_index];
            let other = select(pair.a, pair.b, pair.a == body);
            prescribed_contact = prescribed_contact || moving_kinematic[other] != 0u;
        }
    }
    let linear_speed_sq = dot(state.linear_velocity.xyz, state.linear_velocity.xyz);
    let angular_speed_sq = dot(state.angular_velocity.xyz, state.angular_velocity.xyz);
    let slow_linear = linear_speed_sq <= params.thresholds.y ||
        (previous_valid && linear_displacement_sq <= displacement_threshold_sq);
    if (prescribed_contact || !supported || !slow_linear ||
        angular_speed_sq > params.thresholds.z) {
        sleep_state.previous_position_idle.w = 0.0;
        sleep_states[body] = sleep_state;
        state.inverse_inertia_sleep.w = 0.0;
        states[body] = state;
        return;
    }
    if (state.inverse_inertia_sleep.w != 0.0) {
        sleep_states[body] = sleep_state;
        return;
    }
    let idle = sleep_state.previous_position_idle.w + params.thresholds.x;
    if (idle >= params.thresholds.w) {
        sleep_state.previous_position_idle.w = 0.0;
        state.linear_velocity = vec4<f32>(0.0);
        state.angular_velocity = vec4<f32>(0.0);
        state.inverse_inertia_sleep.w = 1.0;
        states[body] = state;
    } else {
        sleep_state.previous_position_idle.w = idle;
    }
    sleep_states[body] = sleep_state;
}
