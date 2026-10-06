struct Pose {
    position: vec4<f32>,
    orientation: vec4<f32>,
};
struct Layout { indices: vec4<u32>, };
struct Step { timestep: vec4<f32>, };
@group(0) @binding(0) var<storage, read_write> roots: array<Pose>;
@group(0) @binding(1) var<storage, read> velocities: array<f32>;
@group(0) @binding(2) var<storage, read> layouts: array<Layout>;
@group(0) @binding(3) var<storage, read_write> status: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> step: Step;

fn quat_product(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz),
        a.w * b.w - dot(a.xyz, b.xyz));
}

@compute @workgroup_size(64)
fn advance_roots(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let environment = invocation.x;
    if (environment >= arrayLength(&layouts)) { return; }
    let descriptor = layouts[environment].indices;
    if (descriptor.z == 0u || atomicLoad(&status[descriptor.y]) != 0u) { return; }
    let offset = descriptor.x;
    let dt = step.timestep.x;
    let linear = vec3<f32>(velocities[offset], velocities[offset + 1u], velocities[offset + 2u]);
    let angular = vec3<f32>(velocities[offset + 3u], velocities[offset + 4u], velocities[offset + 5u]);
    let position = roots[environment].position.xyz + linear * dt;
    let rotation_step = angular * dt;
    let squared = dot(rotation_step, rotation_step);
    if (!all(abs(position) < vec3<f32>(1e30)) || !(squared < 1e30)) {
        atomicOr(&status[descriptor.y], 1u); return;
    }
    var scale = 0.5 - squared / 48.0;
    var cosine = 1.0 - squared / 8.0;
    if (squared > 1e-12) {
        let angle = sqrt(squared);
        scale = sin(0.5 * angle) / angle;
        cosine = cos(0.5 * angle);
    }
    let delta = vec4<f32>(rotation_step * scale, cosine);
    let orientation = quat_product(delta, roots[environment].orientation);
    let norm_squared = dot(orientation, orientation);
    if (!(norm_squared > 1e-20) || !(norm_squared < 1e30)) {
        atomicOr(&status[descriptor.y], 1u); return;
    }
    roots[environment] = Pose(vec4<f32>(position, 0.0), orientation * inverseSqrt(norm_squared));
}
