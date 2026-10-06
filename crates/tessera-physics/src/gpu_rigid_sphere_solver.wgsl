struct RigidState {
    position_inverse_mass: vec4<f32>,
    orientation: vec4<f32>,
    linear_velocity: vec4<f32>,
    angular_velocity: vec4<f32>,
    inverse_inertia_sleep: vec4<f32>,
}

struct Pair { a: u32, b: u32, }
struct Contact {
    point: vec4<f32>,
    normal: vec4<f32>,
    depth_hit: vec4<f32>,
}
struct Params {
    values: vec4<f32>,
    counts: vec4<u32>,
    flags: vec4<u32>,
    temporal: vec4<f32>,
    static_temporal: vec4<f32>,
}
struct ImpulseState {
    applied: vec4<f32>,
    bounce: vec4<f32>,
    anchor_a: vec4<f32>,
    anchor_b: vec4<f32>,
}
struct IslandRange {
    body_offset: u32,
    body_count: u32,
    pair_offset: u32,
    pair_count: u32,
}
struct Material {
    coefficients: vec4<f32>,
    rules_enabled: vec4<u32>,
}

const WAKE_IMPACT_SPEED: f32 = 0.5;

@group(0) @binding(0) var<storage, read_write> states: array<RigidState>;
@group(0) @binding(1) var<storage, read> pairs: array<Pair>;
@group(0) @binding(2) var<storage, read> pair_contacts: array<Contact>;
@group(0) @binding(3) var<storage, read> ground_contacts: array<Contact>;
@group(0) @binding(4) var<storage, read_write> impulses: array<ImpulseState>;
@group(0) @binding(5) var<uniform> params: Params;
@group(0) @binding(6) var<storage, read> island_ranges: array<IslandRange>;
@group(0) @binding(7) var<storage, read> island_indices: array<u32>;
@group(0) @binding(8) var<storage, read> materials: array<Material>;
fn ground_contact(body: u32, point: u32) -> Contact {
    if (point == 0u) { return ground_contacts[body]; }
    return ground_contacts[params.flags.z + body * 3u + point - 1u];
}

fn pair_contact(pair: u32, point: u32) -> Contact {
    if (point == 0u) { return pair_contacts[pair]; }
    return pair_contacts[params.counts.y + pair * 3u + point - 1u];
}

fn pair_slot(pair: u32, point: u32) -> u32 {
    return pair * params.flags.w + point;
}

fn ground_slot(body: u32, point: u32) -> u32 {
    return params.counts.y * params.flags.w + body * params.flags.y + point;
}

fn material_for(index: u32) -> Material {
    let stored = materials[index];
    if (stored.rules_enabled.z != 0u) { return stored; }
    return Material(
        vec4<f32>(params.values.y, params.values.z, 0.0, 0.0),
        vec4<u32>(0u),
    );
}

fn combine_coefficient(left: f32, right: f32, rule: u32) -> f32 {
    if (rule == 0u) { return (left + right) * 0.5; }
    if (rule == 1u) { return min(left, right); }
    if (rule == 2u) { return left * right; }
    return max(left, right);
}

fn contact_material(a_index: u32, b_index: u32, ground: bool) -> vec2<f32> {
    var a = material_for(a_index);
    if (ground) { a = material_for(arrayLength(&materials) - 1u); }
    let b = material_for(b_index);
    return vec2<f32>(
        combine_coefficient(a.coefficients.x, b.coefficients.x,
                            max(a.rules_enabled.x, b.rules_enabled.x)),
        combine_coefficient(a.coefficients.y, b.coefficients.y,
                            max(a.rules_enabled.y, b.rules_enabled.y)),
    );
}

fn rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

fn local_anchor(state: RigidState, point: vec3<f32>) -> vec3<f32> {
    let q = state.orientation;
    return rotate(vec4<f32>(-q.xyz, q.w), point - state.position_inverse_mass.xyz);
}

fn anchor_changed(current: vec3<f32>, cached: vec3<f32>) -> bool {
    let tolerance = max(1e-3, 0.25 * max(length(current), length(cached)));
    let drift = current - cached;
    return dot(drift, drift) > tolerance * tolerance;
}

fn inverse_inertia_mul(state: RigidState, value: vec3<f32>) -> vec3<f32> {
    let q = state.orientation;
    let body = rotate(vec4<f32>(-q.xyz, q.w), value);
    return rotate(q, body * state.inverse_inertia_sleep.xyz);
}

fn effective_mass(state: RigidState, arm: vec3<f32>, direction: vec3<f32>) -> f32 {
    let angular = cross(arm, direction);
    return state.position_inverse_mass.w + dot(angular, inverse_inertia_mul(state, angular));
}

fn apply_impulse(state: RigidState, arm: vec3<f32>, impulse: vec3<f32>) -> RigidState {
    if (state.position_inverse_mass.w == 0.0) { return state; }
    var result = state;
    result.linear_velocity = vec4<f32>(
        state.linear_velocity.xyz + impulse * state.position_inverse_mass.w, 0.0);
    result.angular_velocity = vec4<f32>(
        state.angular_velocity.xyz + inverse_inertia_mul(state, cross(arm, impulse)), 0.0);
    if (dot(impulse, impulse) > 1e-20) {
        result.inverse_inertia_sleep = vec4<f32>(result.inverse_inertia_sleep.xyz, 0.0);
    }
    return result;
}

fn contact_velocity(state: RigidState, arm: vec3<f32>) -> vec3<f32> {
    return state.linear_velocity.xyz + cross(state.angular_velocity.xyz, arm);
}

fn clear_impulse(slot: u32) {
    let old = impulses[slot];
    var anchor_a = vec4<f32>(0.0);
    var anchor_b = vec4<f32>(0.0);
    var bounce = vec4<f32>(0.0);
    if (params.temporal.w != 0.0) { anchor_a = old.anchor_a; anchor_b = old.anchor_b; }
    if (params.temporal.w != 0.0 && old.anchor_b.w == 1.0) { bounce = old.bounce; }
    impulses[slot] = ImpulseState(vec4<f32>(0.0), bounce, anchor_a, anchor_b);
}
fn contact_arm(state: RigidState, contact: Contact, anchor: vec4<f32>) -> vec3<f32> {
    if (params.temporal.w != 0.0 && anchor.w == 1.0) { return rotate(state.orientation, anchor.xyz); }
    return contact.point.xyz - state.position_inverse_mass.xyz;
}
fn capture_temporal_one(a_index: u32, b_index: u32, contact: Contact, slot: u32, ground: bool) {
    if (contact.depth_hit.y == 0.0) {
        impulses[slot] = ImpulseState(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
        return;
    }
    var old = impulses[slot];
    var anchor_a = contact.point.xyz;
    if (!ground) { anchor_a = local_anchor(states[a_index], contact.point.xyz); }
    let anchor_b = local_anchor(states[b_index], contact.point.xyz);
    if (old.anchor_b.w == 1.0 && (anchor_changed(anchor_b, old.anchor_b.xyz) ||
        (!ground && anchor_changed(anchor_a, old.anchor_a.xyz)) || dot(old.bounce.yzw, contact.normal.xyz) < 0.95)) {
        old.applied = vec4<f32>(0.0); old.bounce = vec4<f32>(0.0);
    }
    var relative = contact_velocity(states[b_index], contact.point.xyz - states[b_index].position_inverse_mass.xyz);
    if (!ground) { relative -= contact_velocity(states[a_index], contact.point.xyz - states[a_index].position_inverse_mass.xyz); }
    let vn = dot(relative, contact.normal.xyz);
    var bounce = 0.0;
    // Speculative rows must retain the incoming target until they reach contact.
    if (vn < -WAKE_IMPACT_SPEED) {
        bounce = -contact_material(a_index, b_index, ground).y * vn;
    }
    impulses[slot] = ImpulseState(old.applied, vec4<f32>(bounce, contact.normal.xyz), vec4<f32>(anchor_a, 1.0), vec4<f32>(anchor_b, 1.0));
}

fn warm_one(a_index: u32, b_index: u32, contact: Contact, slot: u32, ground: bool) {
    let cached = impulses[slot];
    if (contact.depth_hit.y == 0.0 || cached.applied.w <= 0.0 ||
        (params.temporal.w == 0.0 && cached.bounce.x > 0.0) || dot(cached.bounce.yzw, contact.normal.xyz) < 0.95) {
        clear_impulse(slot);
        return;
    }
    var a = states[a_index];
    var b = states[b_index];
    let anchor_b = local_anchor(b, contact.point.xyz);
    let anchor_a = select(vec3<f32>(0.0), local_anchor(a, contact.point.xyz), !ground);
    if (params.temporal.w == 0.0 && (anchor_changed(anchor_b, cached.anchor_b.xyz) ||
        (!ground && anchor_changed(anchor_a, cached.anchor_a.xyz)))) {
        clear_impulse(slot);
        return;
    }
    if (b.inverse_inertia_sleep.w != 0.0 ||
        (!ground && a.position_inverse_mass.w > 0.0 && a.inverse_inertia_sleep.w != 0.0)) {
        clear_impulse(slot);
        return;
    }
    var relative = contact_velocity(b, contact_arm(b, contact, cached.anchor_b));
    if (!ground) {
        relative -= contact_velocity(a, contact_arm(a, contact, cached.anchor_a));
    }
    if (params.temporal.w == 0.0 && dot(relative, relative) > WAKE_IMPACT_SPEED * WAKE_IMPACT_SPEED) {
        clear_impulse(slot);
        return;
    }
    let normal = contact.normal.xyz;
    let tangent = cached.applied.xyz - normal * dot(cached.applied.xyz, normal);
    let impulse = normal * cached.applied.w + tangent;
    if (!ground) {
        a = apply_impulse(a, contact_arm(a, contact, cached.anchor_a), -impulse);
        states[a_index] = a;
    }
    b = apply_impulse(b, contact_arm(b, contact, cached.anchor_b), impulse);
    states[b_index] = b;
    var stored_a = vec4<f32>(anchor_a, 0.0);
    var stored_b = vec4<f32>(anchor_b, 0.0);
    if (params.temporal.w != 0.0 && cached.anchor_b.w == 1.0) { stored_a = cached.anchor_a; stored_b = cached.anchor_b; }
    var bounce = vec4<f32>(0.0, normal);
    if (params.temporal.w != 0.0 && cached.anchor_b.w == 1.0) { bounce = cached.bounce; }
    impulses[slot] = ImpulseState(vec4<f32>(tangent, cached.applied.w), bounce, stored_a, stored_b);
}

fn solve_one(a_index: u32, b_index: u32, contact: Contact, slot: u32,
             ground: bool, iteration: u32) {
    if (contact.depth_hit.y == 0.0) { return; }
    var a = states[a_index];
    var b = states[b_index];
    let awake_a = !ground && a.position_inverse_mass.w > 0.0 &&
        a.inverse_inertia_sleep.w == 0.0;
    let awake_b = b.position_inverse_mass.w > 0.0 && b.inverse_inertia_sleep.w == 0.0;
    let prescribed_a = !ground && a.linear_velocity.w == 1.0;
    let prescribed_b = b.linear_velocity.w == 1.0;
    if (!awake_a && !awake_b && !prescribed_a && !prescribed_b) { return; }
    let coefficients = contact_material(a_index, b_index, ground);
    let normal = contact.normal.xyz;
    let arm_b = contact_arm(b, contact, impulses[slot].anchor_b);
    var arm_a = vec3<f32>(0.0);
    var velocity_a = vec3<f32>(0.0);
    if (!ground) {
        arm_a = contact_arm(a, contact, impulses[slot].anchor_a);
        velocity_a = contact_velocity(a, arm_a);
    }
    let relative = contact_velocity(b, arm_b) - velocity_a;
    let vn = dot(relative, normal);
    let tangent_relative = relative - normal * vn;
    // Resting support impulses must not wake a sleeping body every substep.
    let prescribed_impact = (prescribed_a || prescribed_b) &&
        (vn < -1e-6 || dot(tangent_relative, tangent_relative) > 1e-12);
    let impact = prescribed_impact || vn < -WAKE_IMPACT_SPEED ||
        dot(tangent_relative, tangent_relative) > WAKE_IMPACT_SPEED * WAKE_IMPACT_SPEED;
    let solve_a = !ground && a.position_inverse_mass.w > 0.0 &&
        (awake_a || impact);
    let solve_b = b.position_inverse_mass.w > 0.0 && (awake_b || impact);
    if (!solve_a && !solve_b) { return; }
    var normal_mass = 0.0;
    if (solve_a) { normal_mass += effective_mass(a, arm_a, normal); }
    if (solve_b) { normal_mass += effective_mass(b, arm_b, normal); }
    if (normal_mass <= 1e-12) { return; }
    let bias = params.values.w * max(contact.depth_hit.x - 1e-4, 0.0) / params.values.x;
    let previous = impulses[slot];
    var bounce = previous.bounce.x;
    if (iteration == 0u && vn < -WAKE_IMPACT_SPEED && params.temporal.w >= 0.0 &&
        (params.temporal.w == 0.0 || previous.anchor_b.w != 1.0)) {
        bounce = -coefficients.y * vn;
    }
    var desired_speed = max(bias, bounce);
    var mass_scale = 1.0;
    var impulse_scale = 0.0;
    if (params.temporal.w != 0.0) {
        let softness = select(params.temporal, params.static_temporal,
            ground || a.position_inverse_mass.w == 0.0 || b.position_inverse_mass.w == 0.0);
        // Only penetrating bias rows are compliant. Relaxation and speculative
        // stopping rows enforce the velocity constraint without softness.
        if (softness.w > 0.0 && contact.depth_hit.x >= 0.0) {
            mass_scale = softness.x;
            impulse_scale = softness.y;
        }
        if (contact.depth_hit.x < 0.0) {
            desired_speed = contact.depth_hit.x / params.values.x;
        } else {
            desired_speed = bounce;
            if (softness.w > 0.0) {
                desired_speed = max(desired_speed, min(softness.w,
                    softness.z * max(contact.depth_hit.x - 1e-4, 0.0)));
            }
        }
    }
    let old = previous.applied;
    var normal_change = -impulse_scale * old.w;
    // An underflowed spring coefficient must not multiply an infinite separation target.
    if (mass_scale > 0.0) { normal_change += mass_scale * (desired_speed - vn) / normal_mass; }
    let next_normal = max(0.0, old.w + normal_change);
    let normal_delta = normal * (next_normal - old.w);
    if (solve_a) { a = apply_impulse(a, arm_a, -normal_delta); }
    if (solve_b) { b = apply_impulse(b, arm_b, normal_delta); }

    var tangent_impulse = old.xyz;
    var tangent_velocity = contact_velocity(b, arm_b);
    if (!ground) { tangent_velocity -= contact_velocity(a, arm_a); }
    tangent_velocity -= normal * dot(tangent_velocity, normal);
    let speed = length(tangent_velocity);
    if (params.temporal.w <= 0.0) {
        if (speed > 1e-7 && coefficients.x > 0.0 && next_normal > 0.0) {
            let tangent = tangent_velocity / speed;
            var tangent_mass = 0.0;
            if (solve_b) { tangent_mass += effective_mass(b, arm_b, tangent); }
            if (solve_a) { tangent_mass += effective_mass(a, arm_a, tangent); }
            if (tangent_mass > 1e-12) { tangent_impulse -= tangent_velocity / tangent_mass; }
        }
        // Support may decrease even at zero tangent speed. Always project the
        // accumulated impulse and apply its release to both bodies.
        let max_tangent = coefficients.x * next_normal;
        let magnitude = length(tangent_impulse);
        if (magnitude > max_tangent) { tangent_impulse *= max_tangent / magnitude; }
        let tangent_delta = tangent_impulse - old.xyz;
        if (solve_a) { a = apply_impulse(a, arm_a, -tangent_delta); }
        if (solve_b) { b = apply_impulse(b, arm_b, tangent_delta); }
    }
    var stored_a = vec4<f32>(select(vec3<f32>(0.0), local_anchor(a, contact.point.xyz), !ground), 0.0);
    var stored_b = vec4<f32>(local_anchor(b, contact.point.xyz), 0.0);
    if (params.temporal.w != 0.0 && previous.anchor_b.w == 1.0) { stored_a = previous.anchor_a; stored_b = previous.anchor_b; }
    impulses[slot] = ImpulseState(
        vec4<f32>(tangent_impulse, next_normal), vec4<f32>(bounce, normal),
        stored_a, stored_b);
    if (!ground) { states[a_index] = a; }
    states[b_index] = b;
}

fn warm_island(island: IslandRange) {
    let ground_enabled = params.counts.z != 0u;
    if (params.flags.x != 0u) {
        if (ground_enabled) {
            for (var i = 0u; i < island.body_count; i++) {
                let body = island_indices[island.body_offset + i];
                for (var point = 0u; point < params.flags.y; point++) {
                    warm_one(body, body, ground_contact(body, point),
                        ground_slot(body, point), true);
                }
            }
        }
        for (var i = 0u; i < island.pair_count; i++) {
            let pair_index = island_indices[island.pair_offset + i];
            let pair = pairs[pair_index];
            for (var point = 0u; point < params.flags.w; point++) {
                warm_one(pair.a, pair.b, pair_contact(pair_index, point),
                    pair_slot(pair_index, point), false);
            }
        }
    } else {
        for (var i = 0u; i < island.pair_count; i++) {
            let pair_index = island_indices[island.pair_offset + i];
            for (var point = 0u; point < params.flags.w; point++) {
                clear_impulse(pair_slot(pair_index, point));
            }
        }
        if (ground_enabled) {
            for (var i = 0u; i < island.body_count; i++) {
                let body = island_indices[island.body_offset + i];
                for (var point = 0u; point < params.flags.y; point++) {
                    clear_impulse(ground_slot(body, point));
                }
            }
        }
    }
}

fn solve_island(island: IslandRange, iteration: u32) {
    if (params.counts.z != 0u) {
        for (var i = 0u; i < island.body_count; i++) {
            let body = island_indices[island.body_offset + i];
            for (var point = 0u; point < params.flags.y; point++) {
                solve_one(body, body, ground_contact(body, point),
                    ground_slot(body, point), true, iteration);
            }
        }
    }
    for (var i = 0u; i < island.pair_count; i++) {
        let pair_index = island_indices[island.pair_offset + i];
        let pair = pairs[pair_index];
        for (var point = 0u; point < params.flags.w; point++) {
            solve_one(pair.a, pair.b, pair_contact(pair_index, point),
                pair_slot(pair_index, point), false, iteration);
        }
    }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.counts.x) { return; }
    let island = island_ranges[id.x];
    warm_island(island);
    for (var iteration = 0u; iteration < params.counts.w; iteration++) {
        solve_island(island, iteration);
    }
}

@compute @workgroup_size(64)
fn coupled_warm(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.counts.x) { return; }
    warm_island(island_ranges[id.x]);
}

@compute @workgroup_size(64)
fn coupled_first(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.counts.x) { return; }
    solve_island(island_ranges[id.x], 0u);
}

@compute @workgroup_size(64)
fn coupled_next(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.counts.x) { return; }
    solve_island(island_ranges[id.x], 1u);
}

@compute @workgroup_size(64)
fn temporal_bias(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.counts.x) { return; }
    let island = island_ranges[id.x];
    warm_island(island);
    for (var iteration = 0u; iteration < params.counts.w; iteration++) { solve_island(island, iteration); }
}
@compute @workgroup_size(64)
fn temporal_relax(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.counts.x) { return; }
    let island = island_ranges[id.x];
    for (var iteration = 0u; iteration < params.counts.w; iteration++) { solve_island(island, iteration + 1u); }
}

@compute @workgroup_size(64)
fn temporal_capture(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.counts.x) { return; }
    let island = island_ranges[id.x];
    if (params.counts.z != 0u) {
        for (var i = 0u; i < island.body_count; i++) {
            let body = island_indices[island.body_offset + i];
            for (var point = 0u; point < params.flags.y; point++) {
                capture_temporal_one(body, body, ground_contact(body, point), ground_slot(body, point), true);
            }
        }
    }
    for (var i = 0u; i < island.pair_count; i++) {
        let index = island_indices[island.pair_offset + i];
        let pair = pairs[index];
        for (var point = 0u; point < params.flags.w; point++) {
            capture_temporal_one(pair.a, pair.b, pair_contact(index, point), pair_slot(index, point), false);
        }
    }
}
