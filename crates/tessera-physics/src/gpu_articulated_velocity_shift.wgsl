@group(0) @binding(0) var<storage, read> positions: array<f32>;
@group(0) @binding(1) var<storage, read> velocities: array<f32>;
@group(0) @binding(2) var<storage, read_write> plus: array<f32>;
@group(0) @binding(3) var<storage, read_write> minus: array<f32>;
@group(0) @binding(4) var<storage, read> owners: array<u32>;
@group(0) @binding(5) var<storage, read_write> status: array<atomic<u32>>;

@compute @workgroup_size(64)
fn shift_coordinates(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let coordinate = invocation.x;
    if (coordinate >= arrayLength(&positions)) { return; }
    if (atomicLoad(&status[owners[coordinate]]) != 0u) { return; }
    let delta = 0.001 * velocities[coordinate];
    plus[coordinate] = positions[coordinate] + delta;
    minus[coordinate] = positions[coordinate] - delta;
}
