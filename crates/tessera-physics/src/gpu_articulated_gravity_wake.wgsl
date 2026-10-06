@group(0) @binding(0) var<storage, read> gravity: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> previous: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> changed: array<u32>;
@group(0) @binding(3) var<storage, read_write> status: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> mass_status: array<u32>;
struct Link { indices: vec4<u32>, com: vec4<f32>, thresholds: vec4<f32> };
@group(0) @binding(5) var<storage, read> links: array<Link>;
@group(0) @binding(6) var<storage, read_write> requests: array<atomic<u32>>;

@compute @workgroup_size(64)
fn detect(@builtin(global_invocation_id) id: vec3<u32>) {
    let environment = id.x;
    if environment >= arrayLength(&gravity) { return; }
    changed[environment] = 0u;
    if atomicLoad(&status[environment]) != 0u { return; }
    if mass_status[environment] != 0u ||
        !all(abs(gravity[environment].xyz) <= vec3<f32>(3.402823466e+38)) {
        atomicOr(&status[environment], 1u);
        return;
    }
    let current_bits = bitcast<vec3<u32>>(gravity[environment].xyz);
    let previous_bits = bitcast<vec3<u32>>(previous[environment].xyz);
    let current = select(current_bits, vec3<u32>(0u),
        (current_bits & vec3<u32>(0x7fffffffu)) == vec3<u32>(0u));
    let old = select(previous_bits, vec3<u32>(0u),
        (previous_bits & vec3<u32>(0x7fffffffu)) == vec3<u32>(0u));
    changed[environment] = select(0u, 1u, any(current != old));
    previous[environment] = gravity[environment];
}

@compute @workgroup_size(64)
fn broadcast(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= arrayLength(&links) { return; }
    let link = links[id.x];
    if link.com.w != 0.0 && changed[link.indices.x] != 0u {
        atomicOr(&requests[id.x], 1u);
    }
}
