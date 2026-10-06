struct RigidState {
    position_inverse_mass: vec4<f32>,
    orientation: vec4<f32>,
    linear_velocity: vec4<f32>,
    angular_velocity: vec4<f32>,
    inverse_inertia_sleep: vec4<f32>,
}
@group(0) @binding(0) var<storage, read> states: array<RigidState>;
@group(0) @binding(1) var<storage, read_write> moving: array<u32>;

// Capture before sleep writes any body state, avoiding cross-invocation races.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&moving)) { return; }
    let state = states[id.x];
    let prescribed_motion = state.position_inverse_mass.w == 0.0 && state.linear_velocity.w == 1.0 &&
        (dot(state.linear_velocity.xyz, state.linear_velocity.xyz) > 1e-12 ||
         dot(state.angular_velocity.xyz, state.angular_velocity.xyz) > 1e-12);
    moving[id.x] = select(0u, 1u, prescribed_motion);
}
