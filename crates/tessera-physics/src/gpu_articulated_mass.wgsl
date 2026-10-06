struct Meta {
    matrix_offset: u32,
    solution_offset: u32,
    dimension: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read_write> augmented: array<f32>;
@group(0) @binding(1) var<storage, read> systems: array<Meta>;
@group(0) @binding(2) var<storage, read_write> solution: array<f32>;
@group(0) @binding(3) var<storage, read_write> status: array<u32>;

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let system = invocation.x;
    if (system >= arrayLength(&systems)) {
        return;
    }
    let descriptor = systems[system];
    let n = descriptor.dimension;
    let stride = n + 1u;
    var scale = 0.0;
    for (var row = 0u; row < n; row = row + 1u) {
        for (var col = 0u; col < n; col = col + 1u) {
            scale = max(scale, abs(augmented[descriptor.matrix_offset + row * stride + col]));
        }
    }
    if (!(scale > 0.0) || !(scale < 1.0e30)) {
        status[system] = 1u;
        return;
    }
    for (var pivot = 0u; pivot < n; pivot = pivot + 1u) {
        var selected = pivot;
        var magnitude = abs(augmented[descriptor.matrix_offset + pivot * stride + pivot]);
        for (var row = pivot + 1u; row < n; row = row + 1u) {
            let candidate = abs(augmented[descriptor.matrix_offset + row * stride + pivot]);
            if (candidate > magnitude) {
                magnitude = candidate;
                selected = row;
            }
        }
        if (!(magnitude > scale * 1.0e-7)) {
            status[system] = 1u;
            return;
        }
        if (selected != pivot) {
            for (var col = pivot; col <= n; col = col + 1u) {
                let a = descriptor.matrix_offset + pivot * stride + col;
                let b = descriptor.matrix_offset + selected * stride + col;
                let swapped = augmented[a];
                augmented[a] = augmented[b];
                augmented[b] = swapped;
            }
        }
        let diagonal = augmented[descriptor.matrix_offset + pivot * stride + pivot];
        for (var row = pivot + 1u; row < n; row = row + 1u) {
            let base = descriptor.matrix_offset + row * stride;
            let factor = augmented[base + pivot] / diagonal;
            augmented[base + pivot] = 0.0;
            for (var col = pivot + 1u; col <= n; col = col + 1u) {
                augmented[base + col] = augmented[base + col]
                    - factor * augmented[descriptor.matrix_offset + pivot * stride + col];
            }
        }
    }
    for (var index = n; index > 0u; index = index - 1u) {
        let row = index - 1u;
        let base = descriptor.matrix_offset + row * stride;
        var value = augmented[base + n];
        for (var col = row + 1u; col < n; col = col + 1u) {
            value = value - augmented[base + col] * solution[descriptor.solution_offset + col];
        }
        value = value / augmented[base + row];
        if (!(abs(value) < 1.0e30)) {
            status[system] = 2u;
            return;
        }
        solution[descriptor.solution_offset + row] = value;
    }
    status[system] = 0u;
}
