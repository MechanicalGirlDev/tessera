struct Config { counts: vec4<u32>, }
struct GridNode {
    mass: atomic<u32>,
    momentum_x: atomic<u32>,
    momentum_y: atomic<u32>,
    momentum_z: atomic<u32>,
}
@group(0) @binding(0) var<storage, read> particle_data: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> candidates: array<vec4<i32>>;
@group(0) @binding(2) var<storage, read_write> owners: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> particle_nodes: array<u32>;
@group(0) @binding(4) var<storage, read_write> node_coords: array<vec4<i32>>;
@group(0) @binding(5) var<uniform> config: Config;
@group(0) @binding(6) var<storage, read_write> compact_ids: array<u32>;
@group(0) @binding(7) var<storage, read_write> dispatch_args: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> grid: array<GridNode>;

@compute @workgroup_size(64)
fn generate(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= config.counts.x * 27u { return; }
    let particle = id.x / 27u;
    let local = id.x % 27u;
    let base = bitcast<vec3<i32>>(particle_data[particle * config.counts.y + config.counts.z].xyz);
    let color = bitcast<u32>(particle_data[particle * config.counts.y + 3u].z);
    let offset = vec3<i32>(i32(local / 9u), i32((local / 3u) % 3u), i32(local % 3u));
    candidates[id.x] = vec4<i32>(base + offset, bitcast<i32>(color));
    particle_nodes[id.x] = 0xffffffffu;
}

fn hash(cell: vec4<i32>, manual_color: u32) -> u32 {
    let bits = bitcast<vec4<u32>>(cell);
    var value = (bits.x * 73856093u) ^ (bits.y * 19349663u)
        ^ (bits.z * 83492791u) ^ (bits.w * 2654435761u)
        ^ (manual_color * 1597334677u);
    value ^= value >> 16u;
    value *= 2246822519u;
    return value ^ (value >> 13u);
}

@compute @workgroup_size(64)
fn insert(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= config.counts.x * 27u { return; }
    let particle = id.x / 27u;
    if bitcast<f32>(particle_data[particle * config.counts.y].w) <= 0.0 { return; }
    let key = candidates[id.x];
    let manual_color = particle_data[particle * config.counts.y + 3u].w;
    let mask = config.counts.w - 1u;
    var bucket = hash(key, manual_color) & mask;
    // Owners refer only to immutable candidates written by the previous dispatch.
    // No thread waits for another thread to publish a coordinate or release a lock.
    for (var attempt = 0u; attempt < config.counts.w * 2u; attempt++) {
        let claimed = atomicCompareExchangeWeak(&owners[bucket], 0u, id.x + 1u);
        if claimed.exchanged {
            particle_nodes[id.x] = bucket;
            return;
        }
        if claimed.old_value == 0u { continue; }
        let owner_particle = (claimed.old_value - 1u) / 27u;
        let owner_manual_color = particle_data[owner_particle * config.counts.y + 3u].w;
        if all(candidates[claimed.old_value - 1u] == key) && owner_manual_color == manual_color {
            particle_nodes[id.x] = bucket;
            return;
        }
        bucket = (bucket + 1u) & mask;
    }
    // The sentinel propagates to the G2P failure flag instead of aliasing a node.
}

@compute @workgroup_size(64)
fn resolve(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= config.counts.w { return; }
    let owner = atomicLoad(&owners[id.x]);
    compact_ids[id.x] = 0xffffffffu;
    if owner != 0u {
        let compact = atomicAdd(&dispatch_args[3], 1u);
        if compact < arrayLength(&node_coords) {
            compact_ids[id.x] = compact;
            node_coords[compact] = candidates[owner - 1u];
            // Each compact node has exactly one owner and is cleared before P2G.
            // Unused buffer capacity needs no initialization on this substep.
            atomicStore(&grid[compact].mass, 0u);
            atomicStore(&grid[compact].momentum_x, 0u);
            atomicStore(&grid[compact].momentum_y, 0u);
            atomicStore(&grid[compact].momentum_z, 0u);
        }
    }
}

@compute @workgroup_size(64)
fn remap(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= config.counts.x * 27u { return; }
    let bucket = particle_nodes[id.x];
    if bucket < config.counts.w { particle_nodes[id.x] = compact_ids[bucket]; }
}

@compute @workgroup_size(64)
fn indirect(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x != 0u { return; }
    let count = min(atomicLoad(&dispatch_args[3]), arrayLength(&node_coords));
    atomicStore(&dispatch_args[0], (count + 63u) / 64u);
    atomicStore(&dispatch_args[1], 1u);
    atomicStore(&dispatch_args[2], 1u);
}
