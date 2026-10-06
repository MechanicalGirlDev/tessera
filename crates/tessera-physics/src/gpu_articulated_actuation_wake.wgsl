// Same metadata and Jacobian ABI as the predicted-motion wake pass.
struct MotionLink {
    indices: vec4<u32>,
    center_of_mass: vec4<f32>,
    thresholds: vec4<f32>,
};
@group(0) @binding(0) var<storage, read> links: array<MotionLink>;
@group(0) @binding(1) var<storage, read> terms: array<f32>;
@group(0) @binding(2) var<storage, read> efforts: array<f32>;
@group(0) @binding(3) var<storage, read_write> status: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> requests: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read> mass_status: array<u32>;

@compute @workgroup_size(64)
fn request_actuation_wake(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= arrayLength(&links) { return; }
    let link = links[id.x];
    let environment = link.indices.x;
    if atomicLoad(&status[environment]) != 0u { return; }
    if mass_status[environment] != 0u {
        atomicOr(&status[environment], 1u);
        return;
    }
    if link.center_of_mass.w == 0.0 { return; }
    var actuated = false;
    for (var column = 0u; column < link.indices.z; column += 1u) {
        let effort = efforts[link.indices.y + column];
        if !(abs(effort) <= 3.402823466e+38) {
            atomicOr(&status[environment], 1u);
            return;
        }
        if (bitcast<u32>(effort) & 0x7fffffffu) == 0u { continue; }
        for (var axis = 0u; axis < 6u; axis += 1u) {
            let jacobian = terms[link.indices.w + 10u + axis * link.indices.z + column];
            if !(abs(jacobian) <= 3.402823466e+38) {
                atomicOr(&status[environment], 1u);
                return;
            }
            if (bitcast<u32>(jacobian) & 0x7fffffffu) != 0u { actuated = true; }
        }
    }
    if actuated { atomicOr(&requests[id.x], 1u); }
}
