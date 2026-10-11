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
@group(0) @binding(4) var<storage, read> prescribed: array<u32>;

var<workgroup> prescribed_diagonal: f32;

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
    storageBarrier();
    // Match the placeholder diagonal to the dynamic block's scale so excluded
    // massless coordinates do not make a well-conditioned dynamic block singular.
    if (lane == 0u) {
        var scale = 0.0;
        for (var row = 0u; row < n; row++) {
            if (prescribed[descriptor.vector_offset / 2u + row] == 0u) {
                scale = max(scale, abs(augmented[descriptor.matrix_offset + row * row_stride + row]));
            }
        }
        prescribed_diagonal = select(1.0, scale, scale > 0.0);
    }
    workgroupBarrier();
    for (var element = lane; element < n * row_stride; element += 64u) {
        let row = element / row_stride;
        let col = element % row_stride;
        let offset = descriptor.vector_offset / 2u;
        let row_prescribed = prescribed[offset + row] != 0u;
        if (col == n) {
            if (row_prescribed) { augmented[descriptor.matrix_offset + element] = 0.0; }
        } else if (row_prescribed || prescribed[offset + col] != 0u) {
            // Solve M_DD a_D = f_D with a_K = 0, not (M^-1 f)_D.
            augmented[descriptor.matrix_offset + element] =
                select(0.0, prescribed_diagonal, row == col);
        }
    }
}
