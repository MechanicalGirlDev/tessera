struct Layout { indices: vec4<u32>, }
struct Step { timestep: vec4<f32>, }
@group(0) @binding(0) var<storage, read_write> orientations: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> velocities: array<f32>;
@group(0) @binding(2) var<storage, read> layouts: array<Layout>;
@group(0) @binding(3) var<storage, read_write> status: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> step: Step;
@group(0) @binding(5) var<storage, read> mass_status: array<u32>;

fn product(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz),
        a.w * b.w - dot(a.xyz, b.xyz));
}

@compute @workgroup_size(64)
fn advance_spherical(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let joint = invocation.x;
    if (joint >= arrayLength(&layouts)) { return; }
    let descriptor = layouts[joint].indices;
    if (atomicLoad(&status[descriptor.y]) != 0u) { return; }
    if (mass_status[descriptor.y] != 0u) { atomicOr(&status[descriptor.y], 1u); return; }
    let offset = descriptor.x;
    // Angular velocity is expressed in the parent-side joint frame, not Euler rates.
    let angular_step = vec3<f32>(velocities[offset], velocities[offset + 1u],
        velocities[offset + 2u]) * step.timestep.x;
    let squared = dot(angular_step, angular_step);
    if (!(squared < 1e30)) { atomicOr(&status[descriptor.y], 1u); return; }
    var scale = 0.0;
    var cosine = 1.0;
    if (squared > 0.01) {
        let angle = sqrt(squared);
        scale = sin(0.5 * angle) / angle;
        cosine = cos(0.5 * angle);
    } else {
        // The fourth-order series avoids small-angle GPU transcendental error.
        scale = 0.5 - squared / 48.0 + squared * squared / 3840.0;
        cosine = 1.0 - squared / 8.0 + squared * squared / 384.0;
    }
    let value = product(vec4<f32>(angular_step * scale, cosine), orientations[joint]);
    let norm_squared = dot(value, value);
    if (!(norm_squared > 1e-20) || !(norm_squared < 1e30)) {
        atomicOr(&status[descriptor.y], 1u); return;
    }
    orientations[joint] = value * inverseSqrt(norm_squared);
}
