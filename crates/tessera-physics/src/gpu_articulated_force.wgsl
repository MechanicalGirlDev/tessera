struct Meta {
    matrix_offset: u32,
    link_offset: u32,
    link_count: u32,
    vector_offset: u32,
    dimension: u32,
    inverse_offset: u32,
    _pad1: u32,
    _pad2: u32,
};

struct LinkLoad {
    force_scale: vec4<f32>,
    torque: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> systems: array<Meta>;
@group(0) @binding(1) var<storage, read> links: array<f32>;
@group(0) @binding(2) var<storage, read> base_forces: array<f32>;
@group(0) @binding(3) var<storage, read> gravities: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> vectors: array<f32>;
@group(0) @binding(5) var<storage, read> link_loads: array<LinkLoad>;

@compute @workgroup_size(64)
fn assemble_forces(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let system = workgroup.x;
    if (system >= arrayLength(&systems)) { return; }
    let descriptor = systems[system];
    let n = descriptor.dimension;
    let coordinate_offset = descriptor.vector_offset / 2u;
    let link_stride = 10u + 6u * n;
    let gravity = gravities[system].xyz;
    for (var coordinate = lane; coordinate < n; coordinate += 64u) {
        var force = base_forces[coordinate_offset + coordinate];
        for (var index = 0u; index < descriptor.link_count; index++) {
            let link_base = descriptor.link_offset + index * link_stride;
            let mass = links[link_base];
            let load = link_loads[descriptor._pad1 + index];
            let applied = gravity * (mass * load.force_scale.w) + load.force_scale.xyz;
            let linear = link_base + 10u;
            let angular = linear + 3u * n;
            for (var row = 0u; row < 3u; row++) {
                force += links[linear + row * n + coordinate] * applied[row];
                force += links[angular + row * n + coordinate] * load.torque[row];
            }
        }
        vectors[descriptor.vector_offset + n + coordinate] = force;
    }
}
