struct Config { counts: vec4<u32>, }
struct Snapshot { position_mass: vec4<f32>, velocity_volume: vec4<f32>, }
@group(0) @binding(0) var<storage, read> particles: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> snapshots: array<Snapshot>;
@group(0) @binding(2) var<uniform> config: Config;
@compute @workgroup_size(64)
fn snapshot(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= config.counts.x { return; }
    let base = id.x * config.counts.y;
    snapshots[id.x] = Snapshot(particles[base], particles[base + 1u]);
}
