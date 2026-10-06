struct Sphere {
    indices: vec4<u32>,
    center_radius: vec4<f32>,
    center_of_mass: vec4<f32>,
    plane: vec4<f32>,
    material: vec4<f32>,
    other_center_radius: vec4<f32>,
    other_center_of_mass: vec4<f32>,
    second_axis_end: vec4<f32>,
    first_axis_end: vec4<f32>,
    impulses: vec4<f32>,
    previous_normal: vec4<f32>,
    diagnostic_first: vec4<f32>,
    diagnostic_second: vec4<f32>,
    diagnostic_first_origin: vec4<f32>,
    diagnostic_second_origin: vec4<f32>,
    prescribed_linear: vec4<f32>,
    prescribed_angular: vec4<f32>,
};

@group(0) @binding(0) var<storage, read_write> rows: array<Sphere>;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read_write> status: array<atomic<u32>>;

@group(0) @binding(3) var<storage, read> indices: array<u32>;

@compute @workgroup_size(64)
fn prepare_couplings(@builtin(global_invocation_id) invocation: vec3<u32>) {
    if (invocation.x >= arrayLength(&indices)) { return; }
    let index = indices[invocation.x];
    let row = rows[index];
    if (row.material.w != 78.0 || atomicLoad(&status[row.indices.w]) != 0u) { return; }
    let offset = row.indices.z;
    var x = 0.0;
    if (row.indices.y != 0xffffffffu) {
        x = positions[offset + row.indices.y] - row.center_of_mass.y;
    }
    let c = row.center_radius;
    let a4 = row.plane.x;
    let value = (((a4 * x + c.w) * x + c.z) * x + c.y) * x + c.x;
    let derivative = ((4.0 * a4 * x + 3.0 * c.w) * x + 2.0 * c.z) * x + c.y;
    let error = positions[offset + row.indices.x] - row.center_of_mass.x - value;
    let target_speed = -row.material.x * error / row.material.y;
    if (!(abs(value) < 1e30) || !(abs(derivative) < 1e30) || !(abs(target_speed) < 1e30)) {
        atomicOr(&status[row.indices.w], 1u);
        return;
    }
    rows[index].plane = vec4<f32>(a4,
        clamp(target_speed, -row.material.z, row.material.z), derivative, 0.0);
}
