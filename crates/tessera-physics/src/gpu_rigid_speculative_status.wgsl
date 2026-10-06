struct Contact { point: vec4<f32>, normal: vec4<f32>, depth_hit: vec4<f32> }
@group(0) @binding(0) var<storage, read> contacts: array<Contact>;
@group(0) @binding(1) var<storage, read_write> status: atomic<u32>;
@group(0) @binding(2) var<uniform> counts: vec4<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= counts.x) { return; }
    if (contacts[id.x].depth_hit.z != 0.0) { atomicOr(&status,1u); }
}
