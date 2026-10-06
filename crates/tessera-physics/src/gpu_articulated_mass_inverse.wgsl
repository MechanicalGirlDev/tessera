struct Meta {
    matrix_offset: u32,
    link_offset: u32,
    link_count: u32,
    vector_offset: u32,
    dimension: u32,
    inverse_offset: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> systems: array<Meta>;
@group(0) @binding(1) var<storage, read_write> augmented: array<f32>;
@group(0) @binding(2) var<storage, read_write> inverse: array<f32>;

@compute @workgroup_size(64)
fn invert(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let system = invocation.x;
    if (system >= arrayLength(&systems)) {
        return;
    }
    let descriptor = systems[system];
    let n = descriptor.dimension;
    let matrix_stride = n + 1u;
    let output = descriptor.inverse_offset;
    let status = output + n * n;
    inverse[status] = 2.0;
    var scale = 0.0;
    for (var row = 0u; row < n; row = row + 1u) {
        for (var col = 0u; col < n; col = col + 1u) {
            let matrix_value = augmented[descriptor.matrix_offset + row * matrix_stride + col];
            scale = max(scale, abs(matrix_value));
            inverse[output + row * n + col] = select(0.0, 1.0, row == col);
        }
    }
    if (!(scale > 0.0) || !(scale < 1.0e30)) {
        inverse[status] = 1.0;
        return;
    }
    for (var pivot = 0u; pivot < n; pivot = pivot + 1u) {
        var selected = pivot;
        var magnitude = abs(augmented[descriptor.matrix_offset + pivot * matrix_stride + pivot]);
        for (var row = pivot + 1u; row < n; row = row + 1u) {
            let candidate = abs(augmented[descriptor.matrix_offset + row * matrix_stride + pivot]);
            if (candidate > magnitude) {
                magnitude = candidate;
                selected = row;
            }
        }
        if (!(magnitude > scale * 1.0e-7)) {
            inverse[status] = 1.0;
            return;
        }
        if (selected != pivot) {
            for (var col = 0u; col < n; col = col + 1u) {
                let a = descriptor.matrix_offset + pivot * matrix_stride + col;
                let b = descriptor.matrix_offset + selected * matrix_stride + col;
                let swapped = augmented[a];
                augmented[a] = augmented[b];
                augmented[b] = swapped;
                let inverse_a = output + pivot * n + col;
                let inverse_b = output + selected * n + col;
                let inverse_swapped = inverse[inverse_a];
                inverse[inverse_a] = inverse[inverse_b];
                inverse[inverse_b] = inverse_swapped;
            }
        }
        let diagonal = augmented[descriptor.matrix_offset + pivot * matrix_stride + pivot];
        for (var col = 0u; col < n; col = col + 1u) {
            let matrix_index = descriptor.matrix_offset + pivot * matrix_stride + col;
            augmented[matrix_index] = augmented[matrix_index] / diagonal;
            let inverse_index = output + pivot * n + col;
            inverse[inverse_index] = inverse[inverse_index] / diagonal;
        }
        for (var row = 0u; row < n; row = row + 1u) {
            if (row == pivot) {
                continue;
            }
            let factor = augmented[descriptor.matrix_offset + row * matrix_stride + pivot];
            for (var col = 0u; col < n; col = col + 1u) {
                let matrix_index = descriptor.matrix_offset + row * matrix_stride + col;
                let pivot_index = descriptor.matrix_offset + pivot * matrix_stride + col;
                augmented[matrix_index] = augmented[matrix_index] - factor * augmented[pivot_index];
                let inverse_index = output + row * n + col;
                inverse[inverse_index] = inverse[inverse_index]
                    - factor * inverse[output + pivot * n + col];
            }
        }
    }
    for (var element = 0u; element < n * n; element = element + 1u) {
        if (!(abs(inverse[output + element]) < 1.0e30)) {
            return;
        }
    }
    inverse[status] = 0.0;
}
