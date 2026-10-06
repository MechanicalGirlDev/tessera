struct Pose { position: vec4<f32>, orientation: vec4<f32>, };
@group(0) @binding(0) var<storage, read> roots: array<Pose>;
@group(0) @binding(1) var<storage, read> velocities: array<f32>;
@group(0) @binding(2) var<storage, read> descriptors: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> plus: array<Pose>;
@group(0) @binding(4) var<storage, read_write> minus: array<Pose>;
@group(0) @binding(5) var<storage, read_write> status: array<atomic<u32>>;

fn product(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz));
}

@compute @workgroup_size(64)
fn shift_roots(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let environment = invocation.x;
    if (environment >= arrayLength(&descriptors)) { return; }
    let descriptor = descriptors[environment];
    if (atomicLoad(&status[descriptor.z]) != 0u) { return; }
    let root = roots[environment];
    if (descriptor.y == 0u) {
        plus[environment] = root; minus[environment] = root; return;
    }
    let offset = descriptor.x;
    let translation = 0.001 * vec3<f32>(velocities[offset], velocities[offset+1u], velocities[offset+2u]);
    let rotation = 0.001 * vec3<f32>(velocities[offset+3u], velocities[offset+4u], velocities[offset+5u]);
    let squared = dot(rotation, rotation);
    if (!(squared < 1e30) || !all(abs(root.position.xyz + translation) < vec3<f32>(1e30))
        || !all(abs(root.position.xyz - translation) < vec3<f32>(1e30))) {
        atomicOr(&status[descriptor.z], 1u); return;
    }
    var scale = 0.5 - squared / 48.0;
    var cosine = 1.0 - squared / 8.0;
    if (squared > 1e-12) {
        let angle = sqrt(squared);
        scale = sin(0.5 * angle) / angle; cosine = cos(0.5 * angle);
    }
    let forward = product(vec4<f32>(rotation * scale, cosine), root.orientation);
    let backward = product(vec4<f32>(-rotation * scale, cosine), root.orientation);
    let norms = vec2<f32>(dot(forward, forward), dot(backward, backward));
    if (!all(norms > vec2<f32>(1e-20)) || !all(norms < vec2<f32>(1e30))) {
        atomicOr(&status[descriptor.z], 1u); return;
    }
    plus[environment] = Pose(vec4<f32>(root.position.xyz + translation, 0.0), forward * inverseSqrt(norms.x));
    minus[environment] = Pose(vec4<f32>(root.position.xyz - translation, 0.0), backward * inverseSqrt(norms.y));
}
