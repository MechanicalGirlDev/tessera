struct StepParams {
    timestep: vec4<f32>,
};

@group(0) @binding(0) var<storage, read_write> positions: array<f32>;
@group(0) @binding(1) var<storage, read_write> velocities: array<f32>;
@group(0) @binding(2) var<storage, read> accelerations: array<f32>;
@group(0) @binding(3) var<storage, read> mass_status: array<u32>;
@group(0) @binding(4) var<storage, read> system_indices: array<u32>;
@group(0) @binding(5) var<storage, read_write> state_status: array<atomic<u32>>;
@group(0) @binding(6) var<uniform> params: StepParams;

@group(0) @binding(7) var<storage, read> position_integration: array<u32>;

@compute @workgroup_size(64)
fn advance(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    if (index >= arrayLength(&positions)) { return; }
    let system = system_indices[index];
    if (mass_status[system] != 0u || atomicLoad(&state_status[system]) != 0u) { return; }
    let dt = params.timestep.x;
    let velocity = velocities[index] + accelerations[index] * dt;
    var position = positions[index];
    if (position_integration[index] != 0u) { position += velocity * dt; }
    if (!(abs(velocity) < 1e30) || !(abs(position) < 1e30)) {
        atomicOr(&state_status[system], 1u);
        return;
    }
    velocities[index] = velocity;
    positions[index] = position;
}
