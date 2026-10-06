struct RigidState {
    position_inverse_mass: vec4<f32>,
    orientation: vec4<f32>,
    linear_velocity: vec4<f32>,
    angular_velocity: vec4<f32>,
    inverse_inertia_sleep: vec4<f32>,
}

struct Aabb {
    lower: vec4<f32>,
    upper: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> states: array<RigidState>;
@group(0) @binding(1) var<storage, read> radii: array<f32>;
@group(0) @binding(2) var<storage, read_write> bounds: array<Aabb>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= arrayLength(&bounds)) { return; }
    let center = states[index].position_inverse_mass.xyz;
    let extent = vec3<f32>(radii[index]);
    bounds[index] = Aabb(
        vec4<f32>(center - extent, 0.0),
        vec4<f32>(center + extent, 0.0),
    );
}
