struct Pair {
    a: u32,
    b: u32,
};

struct Shape {
    index_kind: vec4<u32>,
    a: vec4<f32>,
    b: vec4<f32>,
    orientation: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> input_pairs: array<Pair>;
@group(0) @binding(1) var<storage, read> input_counter: array<u32>;
@group(0) @binding(2) var<storage, read> shapes: array<Shape>;
@group(0) @binding(3) var<storage, read> exclusions: array<Pair>;
@group(0) @binding(4) var<storage, read_write> output_pairs: array<Pair>;
@group(0) @binding(5) var<storage, read_write> output_counter: array<atomic<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let id = invocation.x;
    if (id >= input_counter[0]) { return; }
    let pair = input_pairs[id];
    let first_link = shapes[pair.a].index_kind.x;
    let second_link = shapes[pair.b].index_kind.x;
    if (first_link == second_link) { return; }
    for (var excluded = 0u; excluded < arrayLength(&exclusions); excluded++) {
        let edge = exclusions[excluded];
        if ((edge.a == first_link && edge.b == second_link)
            || (edge.a == second_link && edge.b == first_link)) { return; }
    }
    let slot = atomicAdd(&output_counter[0], 1u);
    if (slot < arrayLength(&output_pairs)) {
        output_pairs[slot] = pair;
    } else {
        atomicOr(&output_counter[1], 1u);
    }
}
