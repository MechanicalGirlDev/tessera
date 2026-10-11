struct JointLimit {
    lower: f32,
    upper: f32,
    max_speed: f32,
    flags: f32,
    compliance: vec4<f32>,
};
struct StepParams { timestep: vec4<f32> };
@group(0) @binding(0) var<storage, read> positions: array<f32>;
@group(0) @binding(1) var<storage, read> velocities: array<f32>;
@group(0) @binding(2) var<storage, read_write> accelerations: array<f32>;
@group(0) @binding(3) var<storage, read> inverse: array<f32>;
@group(0) @binding(4) var<storage, read> systems: array<vec4<u32>>;
@group(0) @binding(5) var<storage, read_write> state_status: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> limits: array<JointLimit>;
@group(0) @binding(7) var<uniform> params: StepParams;
@group(0) @binding(8) var<storage, read> position_integration: array<u32>;

@compute @workgroup_size(1)
fn solve_stops(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let environment = invocation.x;
    if (environment >= arrayLength(&systems)
        || atomicLoad(&state_status[environment]) != 0u) { return; }
    let system = systems[environment];
    if (inverse[system.w] != 0.0) {
        atomicOr(&state_status[environment], 1u); return;
    }
    let dt = params.timestep.x;
    for (var slot = 0u; slot < system.y; slot++) {
        limits[system.x + slot].compliance.z = 0.0;
    }
    for (var iteration = 0u; iteration < 8u; iteration++) {
        for (var slot = 0u; slot < system.y; slot++) {
            let coordinate = system.x + slot;
            let limit = limits[coordinate];
            if (limit.compliance.y == 0.0 || limit.compliance.w != 0.0
                || position_integration[coordinate] == 0u) { continue; }
            let q = positions[coordinate];
            let speed = velocities[coordinate] + dt * accelerations[coordinate];
            if (!(abs(q) < 1e30) || !(abs(speed) < 1e30)) {
                atomicOr(&state_status[environment], 1u); return;
            }
            let lower = q < limit.lower || (q <= limit.upper && q + dt * speed < limit.lower);
            let upper = q > limit.upper || (q >= limit.lower && q + dt * speed > limit.upper);
            // The impulse sign identifies the currently active unilateral row.
            var direction = select(-1.0, 1.0, lower);
            if (!lower && !upper) { direction = sign(limit.compliance.z); }
            if (direction == 0.0) { continue; }
            let gap = select(limit.upper - q, q - limit.lower, direction > 0.0);
            let target_speed = select(-gap / dt, -limit.compliance.x * gap / dt, gap < 0.0);
            let effective = inverse[system.z + slot * system.y + slot];
            let gamma = limit.compliance.y / dt;
            if (!(effective > 0.0) || !(effective < 1e30) || !(gamma < 1e30)) {
                atomicOr(&state_status[environment], 1u); return;
            }
            let old = limit.compliance.z;
            let candidate = old + (direction * target_speed - speed - gamma * old) / (effective + gamma);
            let solved = direction * max(direction * candidate, 0.0);
            let delta = solved - old;
            if (!(abs(solved) < 1e30)) {
                atomicOr(&state_status[environment], 1u); return;
            }
            for (var row = 0u; row < system.y; row++) {
                let index = system.x + row;
                accelerations[index] += inverse[system.z + row * system.y + slot] * delta / dt;
                if (!(abs(accelerations[index]) < 1e30)) {
                    atomicOr(&state_status[environment], 1u); return;
                }
            }
            limits[coordinate].compliance.z = solved;
        }
    }
}
