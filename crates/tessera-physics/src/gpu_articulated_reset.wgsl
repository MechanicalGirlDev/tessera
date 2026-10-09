struct Request {
    destination: vec4<u32>,
    source: vec4<u32>,
    counts: vec4<u32>,
    translation: vec4<f32>,
    velocity0: vec4<f32>,
    velocity1: vec4<f32>,
}
struct Pose {
    position: vec4<f32>,
    orientation: vec4<f32>,
}
@group(0) @binding(0) var<storage, read> templates: array<u32>;
@group(0) @binding(1) var<storage, read> requests: array<Request>;
@group(0) @binding(2) var<storage, read_write> positions: array<f32>;
@group(0) @binding(3) var<storage, read_write> velocities: array<f32>;
@group(0) @binding(4) var<storage, read_write> roots: array<Pose>;
@group(0) @binding(5) var<storage, read_write> orientations: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> state_status: array<u32>;
@group(0) @binding(7) var<storage, read_write> mass_status: array<u32>;

fn vector(offset: u32) -> vec4<f32> {
    return bitcast<vec4<f32>>(vec4<u32>(
        templates[offset], templates[offset+1u], templates[offset+2u], templates[offset+3u]));
}

@compute @workgroup_size(64)
fn reset(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= arrayLength(&requests) { return; }
    let r = requests[id.x];
    for (var i = 0u; i < r.destination.w; i++) {
        positions[r.destination.x+i] = bitcast<f32>(templates[r.source.x+i]);
        var velocity = bitcast<f32>(templates[r.source.y+i]);
        if r.counts.y != 0u && i < 6u {
            if i < 4u { velocity = r.velocity0[i]; }
            else { velocity = r.velocity1[i-4u]; }
        }
        velocities[r.destination.x+i] = velocity;
    }
    let env = r.destination.y;
    roots[env].position = vector(r.source.z) + r.translation;
    roots[env].orientation = vector(r.source.z+4u);
    for (var i = 0u; i < r.counts.x; i++) {
        orientations[r.destination.z+i] = vector(r.source.w+i*4u);
    }
    state_status[env] = templates[r.counts.z];
    if !all(abs(roots[env].position.xyz) < vec3<f32>(1e30)) {
        state_status[env] = state_status[env] | 1u;
    }
    mass_status[env] = templates[r.counts.w];
}
