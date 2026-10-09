// Layout offsets come from the native packed contact type, in 32-bit words.
@group(0) @binding(0) var<storage, read> contacts: array<u32>;
@group(0) @binding(1) var<storage, read> selections: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> source_status: array<u32>;
@group(0) @binding(4) var<uniform> row_layout: vec4<u32>;
@group(0) @binding(5) var<storage, read> mass_status: array<u32>;

@compute @workgroup_size(64)
fn observe(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= arrayLength(&selections) { return; }
    let selection = selections[id.x];
    if source_status[selection.w] != 0u || mass_status[selection.w] != 0u {
        output[id.x] = bitcast<f32>(0x7fc00000u);
        return;
    }
    var total = 0.0;
    for (var row = selection.x; row < selection.x + selection.y; row++) {
        let base = row * row_layout.x;
        let first_sign = bitcast<f32>(contacts[base + row_layout.z]);
        let second_sign = bitcast<f32>(contacts[base + row_layout.w]);
        let first = contacts[base] == selection.z && first_sign != 0.0;
        let second = contacts[base + 2u] == selection.z && second_sign != 0.0;
        if first || second {
            total += max(bitcast<f32>(contacts[base + row_layout.y]), 0.0);
        }
    }
    output[id.x] = total;
}
