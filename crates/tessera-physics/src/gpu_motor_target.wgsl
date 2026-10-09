struct Mapping {
    coordinate: u32,
    action: u32,
    mode: u32,
    delay: u32,
};
struct Pending {
    action_value: f32,
    remaining: u32,
    queued: u32,
    padding: u32,
};
struct JointForce {
    passive: vec4<f32>,
    nonlinear: vec4<f32>,
    motor_targets: vec4<f32>,
    motor_gains: vec4<f32>,
    implicit: vec4<f32>,
};
@group(0) @binding(0) var<storage, read_write> mappings: array<Mapping>;
@group(0) @binding(1) var<storage, read_write> pending: array<Pending>;
@group(0) @binding(2) var<storage, read> actions: array<f32>;
@group(0) @binding(3) var<storage, read_write> parameters: array<JointForce>;
@group(0) @binding(4) var<storage, read> delays: array<u32>;
@group(0) @binding(5) var<uniform> control_layout: vec4<u32>;

@compute @workgroup_size(64)
fn update_delays(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&mappings)) { return; }
    let delay = delays[mappings[id.x].action % control_layout.x];
    mappings[id.x].delay = delay;
    if (pending[id.x].queued != 0u) {
        pending[id.x].remaining = delay;
    }
}

@compute @workgroup_size(64)
fn latch_targets(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&mappings)) { return; }
    let mapping = mappings[id.x];
    pending[id.x] = Pending(actions[mapping.action], mapping.delay, 1u, 0u);
}

@compute @workgroup_size(64)
fn apply_targets(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&mappings)) { return; }
    if (pending[id.x].queued == 0u) { return; }
    if (pending[id.x].remaining != 0u) {
        pending[id.x].remaining -= 1u;
        return;
    }
    let mapping = mappings[id.x];
    if (mapping.mode == 0u) {
        parameters[mapping.coordinate].motor_targets.x = pending[id.x].action_value;
    } else {
        parameters[mapping.coordinate].motor_targets.y = pending[id.x].action_value;
    }
    pending[id.x].queued = 0u;
}
