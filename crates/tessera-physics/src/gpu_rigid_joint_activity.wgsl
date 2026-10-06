struct Joint {
    bodies: vec4<u32>,
    anchor_a: vec4<f32>,
    anchor_b: vec4<f32>,
    frame_a: vec4<f32>,
    frame_b: vec4<f32>,
    drive: vec4<f32>,
    servo: vec4<f32>,
}
struct Island { offset: u32, count: u32, padding_a: u32, padding_b: u32, }
@group(0) @binding(0) var<storage, read> joints: array<Joint>;
@group(0) @binding(1) var<storage, read> islands: array<Island>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> moving: array<u32>;

// Joint islands have disjoint body ownership. One invocation handles each island.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&islands)) { return; }
    let island = islands[id.x];
    var prescribed = false;
    for (var local = 0u; local < island.count; local++) {
        let joint = joints[indices[island.offset + local]];
        prescribed = prescribed || moving[joint.bodies.x] != 0u || moving[joint.bodies.y] != 0u;
    }
    if (!prescribed) { return; }
    for (var local = 0u; local < island.count; local++) {
        let joint = joints[indices[island.offset + local]];
        moving[joint.bodies.x] = 2u;
        moving[joint.bodies.y] = 2u;
    }
}
