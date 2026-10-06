struct Drive {
    indices: vec4<u32>,
    orientation_target: vec4<f32>,
    velocity: vec4<f32>,
    stiffness: vec4<f32>,
    damping: vec4<f32>,
    cap: vec4<f32>,
}
@group(0) @binding(0) var<storage, read> drives: array<Drive>;
@group(0) @binding(1) var<storage, read> orientations: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> velocities: array<f32>;
@group(0) @binding(3) var<storage, read_write> status: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> mass_status: array<u32>;
@group(0) @binding(5) var<storage, read_write> forces: array<f32>;

fn product(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w*b.w-dot(a.xyz,b.xyz));
}
@compute @workgroup_size(64)
fn assemble_spherical_drives(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    if (index >= arrayLength(&drives)) { return; }
    let drive = drives[index];
    let environment = drive.indices.y;
    if (atomicLoad(&status[environment]) != 0u) { return; }
    if (mass_status[environment] != 0u) { atomicOr(&status[environment], 1u); return; }
    if (drive.indices.w == 0u) { return; }
    let current = orientations[drive.indices.z];
    var error = product(drive.orientation_target, vec4<f32>(-current.xyz, current.w));
    let squared = dot(error,error);
    if (!(squared > 1e-20) || !(squared < 1e30)) { atomicOr(&status[environment], 1u); return; }
    error *= inverseSqrt(squared);
    if (error.w < 0.0) { error = -error; }
    let sine = length(error.xyz);
    var logarithm = error.xyz * 2.0;
    if (sine > 1e-4) {
        var half_angle = atan2(sine, error.w);
        if (sine < 0.5 * error.w) {
            // The alternating atan series avoids backend intrinsic error for small rotations.
            let ratio = sine / error.w;
            let t = ratio * ratio;
            var polynomial = -1.0 / 19.0;
            polynomial = 1.0 / 17.0 + t * polynomial;
            polynomial = -1.0 / 15.0 + t * polynomial;
            polynomial = 1.0 / 13.0 + t * polynomial;
            polynomial = -1.0 / 11.0 + t * polynomial;
            polynomial = 1.0 / 9.0 + t * polynomial;
            polynomial = -1.0 / 7.0 + t * polynomial;
            polynomial = 1.0 / 5.0 + t * polynomial;
            polynomial = -1.0 / 3.0 + t * polynomial;
            polynomial = 1.0 + t * polynomial;
            half_angle = ratio * polynomial;
        }
        logarithm = error.xyz * (2.0 * half_angle / sine);
    }
    let offset = drive.indices.x;
    let omega = vec3<f32>(velocities[offset], velocities[offset+1u], velocities[offset+2u]);
    let torque = drive.stiffness.xyz * logarithm + drive.damping.xyz * (drive.velocity.xyz-omega);
    if (!all(abs(torque) < vec3<f32>(1e30))) { atomicOr(&status[environment], 1u); return; }
    let capped = clamp(torque, -drive.cap.xyz, drive.cap.xyz);
    let output = vec3<f32>(forces[offset],forces[offset+1u],forces[offset+2u]) + capped;
    if (!all(abs(output) < vec3<f32>(1e30))) { atomicOr(&status[environment], 1u); return; }
    forces[offset] = output.x;
    forces[offset+1u] = output.y;
    forces[offset+2u] = output.z;
}
