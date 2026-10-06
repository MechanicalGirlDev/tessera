struct Aabb {
    lower: vec4<f32>,
    upper: vec4<f32>,
}

struct Pair {
    a: u32,
    b: u32,
}

@group(0) @binding(0) var<storage, read> bounds: array<Aabb>;
@group(0) @binding(1) var<storage, read> pairs: array<Pair>;
@group(0) @binding(2) var<storage, read_write> overlaps: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= arrayLength(&pairs)) {
        return;
    }
    let pair = pairs[index];
    let a = bounds[pair.a];
    let b = bounds[pair.b];
    let hit = all(a.lower.xyz <= b.upper.xyz) && all(b.lower.xyz <= a.upper.xyz);
    overlaps[index] = select(0u, 1u, hit);
}
