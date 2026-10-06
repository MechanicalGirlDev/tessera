struct Layout { indices: vec4<u32>, }
@group(0) @binding(0) var<storage, read> orientations: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> velocities: array<f32>;
@group(0) @binding(2) var<storage, read> layouts: array<Layout>;
@group(0) @binding(3) var<storage, read_write> plus: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> minus: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> status: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read> mass_status: array<u32>;

fn product(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz),
        a.w * b.w - dot(a.xyz, b.xyz));
}

@compute @workgroup_size(64)
fn shift_spherical(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let joint = invocation.x;
    if (joint >= arrayLength(&layouts)) { return; }
    let descriptor = layouts[joint].indices;
    if (atomicLoad(&status[descriptor.y]) != 0u) { return; }
    if (mass_status[descriptor.y] != 0u) { atomicOr(&status[descriptor.y], 1u); return; }
    let offset = descriptor.x;
    let angular_step = vec3<f32>(velocities[offset], velocities[offset + 1u], velocities[offset + 2u]) * 0.001;
    let squared = dot(angular_step, angular_step);
    if (!(squared < 1e30)) { atomicOr(&status[descriptor.y], 1u); return; }
    var scale = 0.0;
    var cosine = 1.0;
    if (squared > 0.01) {
        let angle = sqrt(squared);
        scale = sin(0.5 * angle) / angle;
        cosine = cos(0.5 * angle);
    } else {
        scale = 0.5 - squared / 48.0 + squared * squared / 3840.0;
        cosine = 1.0 - squared / 8.0 + squared * squared / 384.0;
    }
    let initial = orientations[descriptor.z];
    let positive = product(vec4<f32>(angular_step * scale, cosine), initial);
    let negative = product(vec4<f32>(-angular_step * scale, cosine), initial);
    let positive_norm = dot(positive, positive);
    let negative_norm = dot(negative, negative);
    if (!(positive_norm > 1e-20) || !(positive_norm < 1e30)
        || !(negative_norm > 1e-20) || !(negative_norm < 1e30)) {
        atomicOr(&status[descriptor.y], 1u); return;
    }
    plus[descriptor.z] = positive * inverseSqrt(positive_norm);
    minus[descriptor.z] = negative * inverseSqrt(negative_norm);
}
