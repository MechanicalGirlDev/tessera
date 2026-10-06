@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> coordinates: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn expand(@builtin(global_invocation_id) id: vec3<u32>) {
    let offset = coordinates[0];
    let n = coordinates[1];
    let width = coordinates[2];
    let index = id.x;
    if (index > width * width) { return; }
    if (index == width * width) {
        output[index] = source[offset + n * n];
        return;
    }
    let row = coordinates[4 + index / width];
    let column = coordinates[4 + index % width];
    output[index] = 0.0;
    if (row != 0xffffffffu && column != 0xffffffffu) {
        output[index] = source[offset + row * n + column];
    }
}
