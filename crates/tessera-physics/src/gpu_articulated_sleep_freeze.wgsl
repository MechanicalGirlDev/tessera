// Coordinate, environment, first owner, owner count.
@group(0) @binding(0) var<storage, read> coordinates: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> owners: array<u32>;
@group(0) @binding(2) var<storage, read> sleeping: array<u32>;
@group(0) @binding(3) var<storage, read_write> velocities: array<f32>;
@group(0) @binding(4) var<storage, read_write> accelerations: array<f32>;
@group(0) @binding(5) var<storage, read> mass_status: array<u32>;
@group(0) @binding(6) var<storage, read_write> status: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> frozen: array<u32>;

@compute @workgroup_size(64)
fn freeze(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= arrayLength(&coordinates) { return; }
    let coordinate = coordinates[id.x];
    frozen[coordinate.x] = 0u;
    if mass_status[coordinate.y] != 0u || atomicLoad(&status[coordinate.y]) != 0u { return; }
    if !(abs(velocities[coordinate.x]) <= 3.402823466e+38) ||
        !(abs(accelerations[coordinate.x]) <= 3.402823466e+38) {
        atomicOr(&status[coordinate.y], 1u);
        return;
    }
    var all_sleeping = coordinate.w != 0u;
    for (var i = 0u; i < coordinate.w; i += 1u) {
        let flag = sleeping[owners[coordinate.z + i]];
        if flag > 1u {
            atomicOr(&status[coordinate.y], 1u);
            return;
        }
        if flag == 0u { all_sleeping = false; }
    }
    if all_sleeping {
        velocities[coordinate.x] = 0.0;
        accelerations[coordinate.x] = 0.0;
        frozen[coordinate.x] = 1u;
    }
}
