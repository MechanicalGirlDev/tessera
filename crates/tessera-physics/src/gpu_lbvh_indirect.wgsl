struct Params {
    workgroup_size: u32,
    pair_capacity: u32,
    padding: vec2<u32>,
}

@group(0) @binding(0) var<storage, read> counter: array<u32>;
@group(0) @binding(1) var<storage, read_write> dispatch_args: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(1)
fn main() {
    let valid = counter[1] == 0u && counter[0] <= params.pair_capacity;
    let count = select(0u, counter[0], valid);
    dispatch_args[0] = (count + params.workgroup_size - 1u) / params.workgroup_size;
    dispatch_args[1] = 1u;
    dispatch_args[2] = 1u;
}
