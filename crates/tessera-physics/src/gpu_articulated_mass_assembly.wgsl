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

@group(0) @binding(0) var<storage, read> systems: array<Meta>;
@group(0) @binding(1) var<storage, read> links: array<f32>;
@group(0) @binding(2) var<storage, read> vectors: array<f32>;
@group(0) @binding(3) var<storage, read_write> augmented: array<f32>;

@compute @workgroup_size(64)
fn assemble(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let system = workgroup.x;
    if (system >= arrayLength(&systems)) {
        return;
    }
    let descriptor = systems[system];
    let n = descriptor.dimension;
    let link_stride = 10u + 6u * n;
    let row_stride = n + 1u;
    for (var element = lane; element < n * row_stride; element = element + 64u) {
        let row = element / row_stride;
        let col = element % row_stride;
        let output = descriptor.matrix_offset + element;
        if (col == n) {
            augmented[output] = vectors[descriptor.vector_offset + n + row];
            continue;
        }
        var value = 0.0;
        if (row == col) {
            value = vectors[descriptor.vector_offset + row];
        }
        for (var index = 0u; index < descriptor.link_count; index = index + 1u) {
            let base = descriptor.link_offset + index * link_stride;
            let mass = links[base];
            if (mass == 0.0) { continue; }
            let linear = base + 10u;
            let angular = linear + 3u * n;
            for (var axis = 0u; axis < 3u; axis = axis + 1u) {
                value = value + mass
                    * links[linear + axis * n + row]
                    * links[linear + axis * n + col];
                for (var component = 0u; component < 3u; component = component + 1u) {
                    value = value
                        + links[angular + axis * n + row]
                        * links[base + 1u + axis * 3u + component]
                        * links[angular + component * n + col];
                }
            }
        }
        augmented[output] = value;
    }
}
