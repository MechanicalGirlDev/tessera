// Per-link parameters: dt, idle time threshold, enabled, positive mass.
@group(0) @binding(0) var<storage, read> parameters: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> parents: array<u32>;
@group(0) @binding(2) var<storage, read> contact_flags: array<u32>;
@group(0) @binding(3) var<storage, read> wake: array<u32>;
@group(0) @binding(4) var<storage, read_write> idle: array<f32>;
@group(0) @binding(5) var<storage, read_write> component_idle: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> component_flags: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> sleeping: array<u32>;
@group(0) @binding(8) var<storage, read> environments: array<u32>;
@group(0) @binding(9) var<storage, read> state_status: array<u32>;
@group(0) @binding(10) var<storage, read> mass_status: array<u32>;
@group(0) @binding(11) var<storage, read_write> guarded_wake: array<u32>;
struct IdlePolicy { contact_mask: vec4<u32> };
@group(0) @binding(12) var<uniform> policy: IdlePolicy;

@compute @workgroup_size(64)
fn guard_sources(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= arrayLength(&environments) { return; }
    let environment = environments[id.x];
    if state_status[environment] != 0u || mass_status[environment] != 0u {
        guarded_wake[id.x] = 1u;
    }
}

fn root(link: u32) -> u32 {
    var result = link;
    loop {
        let parent = parents[result];
        if parent == result { break; }
        result = parent;
    }
    return result;
}

@compute @workgroup_size(64)
fn initialize(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= arrayLength(&component_idle) { return; }
    atomicStore(&component_idle[id.x], 0x7f800000u);
    atomicStore(&component_flags[id.x], 0u);
}

@compute @workgroup_size(64)
fn reduce(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= arrayLength(&parameters) { return; }
    let p = parameters[id.x];
    if p.w == 0.0 { return; } // Massless helpers do not veto physical bodies.
    let component = root(id.x);
    let candidate = min(idle[id.x] + p.x, 3.402823466e38);
    atomicMin(&component_idle[component], bitcast<u32>(candidate));
    var flags = 4u; // At least one physical body belongs to this component.
    if p.z == 0.0 || wake[id.x] != 0u { flags |= 1u; }
    if (contact_flags[id.x] & policy.contact_mask.x) != 0u { flags |= 2u; }
    atomicOr(&component_flags[component], flags);
}

// Decide readiness against the shared component elapsed time, not each link's
// previous history. This also handles components that merged during this step.
@compute @workgroup_size(64)
fn reduce_ready(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= arrayLength(&parameters) { return; }
    if parameters[id.x].w == 0.0 { return; }
    let component = root(id.x);
    let elapsed = bitcast<f32>(atomicLoad(&component_idle[component]));
    if elapsed < parameters[id.x].y {
        atomicOr(&component_flags[component], 8u);
    }
}

@compute @workgroup_size(64)
fn finish(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= arrayLength(&parameters) { return; }
    let component = root(id.x);
    let flags = atomicLoad(&component_flags[component]);
    let quiet_contact = (flags & 7u) == 6u;
    let elapsed = bitcast<f32>(atomicLoad(&component_idle[component]));
    idle[id.x] = select(0.0, elapsed, quiet_contact);
    // Helpers inherit the physical component decision. Their disabled settings
    // or arbitrary local waiting times must not cause partial coordinate freezes.
    let helper = parameters[id.x].w == 0.0;
    let local_ready = helper || (parameters[id.x].z != 0.0 && elapsed >= parameters[id.x].y);
    sleeping[id.x] = select(0u, 1u, quiet_contact && flags == 6u && local_ready);
}
