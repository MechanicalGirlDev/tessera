struct RowMeta {
    inverse_offset: u32,
    jacobian_offset: u32,
    dimension: u32,
    status_offset: u32,
};

@group(0) @binding(0) var<storage, read> inverse_mass: array<f32>;
@group(0) @binding(1) var<storage, read> jacobians: array<f32>;
@group(0) @binding(2) var<storage, read> rows: array<RowMeta>;
@group(0) @binding(3) var<storage, read_write> responses: array<f32>;
@group(0) @binding(4) var<storage, read_write> effective: array<f32>;
@group(0) @binding(5) var<storage, read_write> status: array<f32>;

@compute @workgroup_size(64)
fn calculate(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let row = invocation.x;
    if (row >= arrayLength(&rows)) {
        return;
    }
    let descriptor = rows[row];
    let source_status = inverse_mass[descriptor.status_offset];
    status[row] = source_status;
    if (source_status != 0.0) {
        effective[row] = 0.0;
        for (var coordinate = 0u; coordinate < descriptor.dimension; coordinate++) {
            responses[descriptor.jacobian_offset + coordinate] = 0.0;
        }
        return;
    }
    var diagonal = 0.0;
    for (var coordinate = 0u; coordinate < descriptor.dimension; coordinate++) {
        var response = 0.0;
        for (var column = 0u; column < descriptor.dimension; column++) {
            response += inverse_mass[descriptor.inverse_offset + coordinate * descriptor.dimension + column]
                * jacobians[descriptor.jacobian_offset + column];
        }
        responses[descriptor.jacobian_offset + coordinate] = response;
        diagonal += jacobians[descriptor.jacobian_offset + coordinate] * response;
    }
    effective[row] = diagonal;
    if (!(abs(diagonal) < 1.0e30)) {
        status[row] = 2.0;
    }
}
