// Same load ABI as gpu_articulated_force.wgsl. Gravity scale is not an external load.
struct LinkLoad { force_scale: vec4<f32>, torque: vec4<f32> };
@group(0) @binding(0) var<storage, read> loads: array<LinkLoad>;
@group(0) @binding(1) var<storage, read> environments: array<u32>;
@group(0) @binding(2) var<storage, read_write> status: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> requests: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> mass_status: array<u32>;

@compute @workgroup_size(64)
fn request_load_wake(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= arrayLength(&environments) { return; }
    let environment = environments[id.x];
    if atomicLoad(&status[environment]) != 0u { return; }
    if mass_status[environment] != 0u {
        atomicOr(&status[environment], 1u);
        return;
    }
    let load = loads[id.x];
    if !all(abs(load.force_scale.xyz) <= vec3<f32>(3.402823466e+38)) ||
        !all(abs(load.torque.xyz) <= vec3<f32>(3.402823466e+38)) {
        atomicOr(&status[environment], 1u);
        return;
    }
    // Inspect represented values, preserving subnormal requests even on devices
    // that flush floating-point comparisons to zero. Signed zero is not a load.
    let force = bitcast<vec3<u32>>(load.force_scale.xyz) & vec3<u32>(0x7fffffffu);
    let torque = bitcast<vec3<u32>>(load.torque.xyz) & vec3<u32>(0x7fffffffu);
    if any(force != vec3<u32>(0u)) || any(torque != vec3<u32>(0u)) {
        atomicOr(&requests[id.x], 1u);
    }
}
