struct Mapping {
    drive: u32,
    action: u32,
    component: u32,
    delay: u32,
};
struct Pending {
    value: f32,
    remaining: u32,
    queued: u32,
    padding: u32,
};
struct Drive {
    indices: vec4<u32>,
    orientation_target: vec4<f32>,
    velocity: vec4<f32>,
    stiffness: vec4<f32>,
    damping: vec4<f32>,
    cap: vec4<f32>,
};
struct Destinations { rows: array<u32, 8>, };
@group(0) @binding(0) var<storage, read_write> mappings: array<Mapping>;
@group(0) @binding(1) var<storage, read_write> pending: array<Pending>;
@group(0) @binding(2) var<storage, read> actions: array<f32>;
@group(0) @binding(3) var<storage, read_write> drives: array<Drive>;
@group(0) @binding(4) var<storage, read> delays: array<u32>;
@group(0) @binding(5) var<uniform> control_layout: vec4<u32>;
@group(0) @binding(6) var<storage, read> destinations: array<Destinations>;

// q and -q select the same shortest representative, including exactly pi.
fn canonical(q: vec4<f32>) -> vec4<f32> {
    var result = normalize(q);
    let negative_axis = result.x < 0.0 || (result.x == 0.0 &&
        (result.y < 0.0 || (result.y == 0.0 && result.z < 0.0)));
    if (result.w < 0.0 || (result.w == 0.0 && negative_axis)) { result = -result; }
    return result;
}
fn logarithm(q: vec4<f32>) -> vec3<f32> {
    let v = canonical(q);
    let sine = length(v.xyz);
    if (sine < 1e-6) { return 2.0 * v.xyz; }
    return v.xyz * (2.0 * atan2(sine, v.w) / sine);
}
fn exponential(v: vec3<f32>) -> vec4<f32> {
    let angle = length(v);
    if (angle < 1e-6) { return canonical(vec4<f32>(0.5 * v, 1.0)); }
    return canonical(vec4<f32>(v * (sin(0.5 * angle) / angle), cos(0.5 * angle)));
}
@compute @workgroup_size(64)
fn latch_targets(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&mappings)) { return; }
    let mapping = mappings[id.x];
    pending[id.x] = Pending(actions[mapping.action], mapping.delay, 1u, 0u);
}
@compute @workgroup_size(64)
fn update_delays(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&mappings)) { return; }
    let delay = delays[mappings[id.x].action % control_layout.x];
    mappings[id.x].delay = delay;
    if (pending[id.x].queued != 0u) { pending[id.x].remaining = delay; }
}
@compute @workgroup_size(64)
fn apply_targets(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&destinations)) { return; }
    var drive = drives[id.x];
    var rotation = logarithm(drive.orientation_target);
    var changed_position = false;
    var changed_velocity = false;
    for (var component = 0u; component < 6u; component += 1u) {
        let row = destinations[id.x].rows[component];
        if (row == 0xffffffffu) { continue; }
        if (pending[row].queued == 0u) { continue; }
        if (pending[row].remaining != 0u) {
            pending[row].remaining -= 1u;
            continue;
        }
        if (component < 3u) {
            rotation[component] = pending[row].value;
            changed_position = true;
        } else {
            drive.velocity[component - 3u] = pending[row].value;
            changed_velocity = true;
        }
        pending[row].queued = 0u;
    }
    if (changed_position) { drive.orientation_target = exponential(rotation); }
    if (changed_position || changed_velocity) { drives[id.x] = drive; }
}
