struct JointLimit {
    lower: f32,
    upper: f32,
    max_speed: f32,
    flags: f32,
};

struct StepParams {
    timestep: vec4<f32>,
};

@group(0) @binding(0) var<storage, read_write> positions: array<f32>;
@group(0) @binding(1) var<storage, read_write> velocities: array<f32>;
@group(0) @binding(2) var<storage, read> accelerations: array<f32>;
@group(0) @binding(3) var<storage, read> mass_status: array<u32>;
@group(0) @binding(4) var<storage, read> owners: array<u32>;
@group(0) @binding(5) var<storage, read_write> state_status: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read> limits: array<JointLimit>;
@group(0) @binding(7) var<uniform> params: StepParams;

@group(0) @binding(8) var<storage, read> position_integration: array<u32>;

@compute @workgroup_size(64)
fn advance_limited(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let coordinate = invocation.x;
    if (coordinate >= arrayLength(&positions)) { return; }
    let system = owners[coordinate];
    if (mass_status[system] != 0u || atomicLoad(&state_status[system]) != 0u) { return; }
    let dt = params.timestep.x;
    let limit = limits[coordinate];
    var velocity = velocities[coordinate] + accelerations[coordinate] * dt;
    if (!(abs(velocity) < 1e30)) {
        atomicOr(&state_status[system], 1u);
        return;
    }
    if (limit.flags >= 2.0) {
        velocity = clamp(velocity, -limit.max_speed, limit.max_speed);
    }
    let integrate_position = position_integration[coordinate] != 0u;
    var position = positions[coordinate];
    if (integrate_position) { position += velocity * dt; }
    if (!(abs(position) < 1e30)) {
        atomicOr(&state_status[system], 1u);
        return;
    }
    if (integrate_position && (limit.flags == 1.0 || limit.flags == 3.0)) {
        let bounded = clamp(position, limit.lower, limit.upper);
        if (bounded != position) {
            velocity = 0.0;
        }
        position = bounded;
    }
    velocities[coordinate] = velocity;
    positions[coordinate] = position;
}
