struct Bounds { minimum: vec4<f32>, maximum: vec4<f32> }
@group(0) @binding(0) var<storage, read_write> bounds: array<Bounds>;
@group(0) @binding(1) var<uniform> margin: vec4<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&bounds)) { return; }
    let delta = vec3<f32>(margin.x);
    bounds[id.x].minimum = vec4<f32>(bounds[id.x].minimum.xyz - delta, bounds[id.x].minimum.w);
    bounds[id.x].maximum = vec4<f32>(bounds[id.x].maximum.xyz + delta, bounds[id.x].maximum.w);
}
