struct Actuator {
    header: vec4<u32>,
    gains: vec4<f32>,
    bias: vec4<f32>,
    clamps: vec4<f32>,
}
@group(0) @binding(0) var<storage, read> rows: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> actuators: array<Actuator>;
@group(0) @binding(2) var<storage, read> controls: array<f32>;
@group(0) @binding(3) var<storage, read> positions: array<f32>;
@group(0) @binding(4) var<storage, read> velocities: array<f32>;
@group(0) @binding(5) var<storage, read_write> forces: array<f32>;

@compute @workgroup_size(64)
fn add_efforts(@builtin(global_invocation_id) invocation: vec3<u32>) {
    if invocation.x >= arrayLength(&rows) { return; }
    let row = rows[invocation.x];
    let q = positions[row.x];
    let v = velocities[row.x];
    var sum = 0.0;
    for (var i = 0u; i < row.z; i += 1u) {
        let a = actuators[row.y + i];
        var u = controls[a.header.y];
        if a.header.z != 0u { u = clamp(u, a.clamps.x, a.clamps.y); }
        var effort = 0.0;
        switch a.header.x {
            case 1u: { effort = u * a.gains.x; }
            case 2u: { effort = a.gains.x * (u - q) - a.gains.y * v; }
            case 3u: { effort = a.gains.x * (u - v); }
            case 4u: { effort = -a.gains.x * abs(u) * v; }
            case 5u: { effort = a.gains.x * u + a.bias.x + a.bias.y * q + a.bias.z * v; }
            case 6u: { effort = a.gains.x * u * a.gains.y; }
            default: {}
        }
        if a.header.w != 0u { effort = clamp(effort, a.clamps.z, a.clamps.w); }
        sum += effort;
    }
    forces[row.x] += sum;
}
