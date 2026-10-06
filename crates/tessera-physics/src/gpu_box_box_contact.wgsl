struct Box {
    center: vec4<f32>,
    axis_x: vec4<f32>,
    axis_y: vec4<f32>,
    axis_z: vec4<f32>,
    half_extents: vec4<f32>,
}

struct Contact {
    point: vec4<f32>,
    normal: vec4<f32>,
    depth_hit: vec4<f32>,
}

struct AxisResult {
    separated: bool,
    depth: f32,
    normal: vec3<f32>,
}

@group(0) @binding(0) var<storage, read> boxes: array<Box>;
@group(0) @binding(1) var<storage, read_write> contacts: array<Contact>;

fn projected_radius(shape: Box, axis: vec3<f32>) -> f32 {
    return shape.half_extents.x * abs(dot(shape.axis_x.xyz, axis))
        + shape.half_extents.y * abs(dot(shape.axis_y.xyz, axis))
        + shape.half_extents.z * abs(dot(shape.axis_z.xyz, axis));
}

fn consider_axis(
    a: Box, b: Box, delta: vec3<f32>, candidate: vec3<f32>, best: AxisResult
) -> AxisResult {
    let length_squared = dot(candidate, candidate);
    if (length_squared < 1e-10 || best.separated) {
        return best;
    }
    let axis = candidate * inverseSqrt(length_squared);
    let separation = dot(delta, axis);
    let overlap = projected_radius(a, axis) + projected_radius(b, axis) - abs(separation);
    if (overlap < 0.0) {
        return AxisResult(true, 0.0, vec3<f32>(0.0));
    }
    if (overlap < best.depth) {
        return AxisResult(false, overlap, axis * select(-1.0, 1.0, separation >= 0.0));
    }
    return best;
}

fn support(shape: Box, direction: vec3<f32>) -> vec3<f32> {
    let sx = select(-1.0, 1.0, dot(shape.axis_x.xyz, direction) >= 0.0);
    let sy = select(-1.0, 1.0, dot(shape.axis_y.xyz, direction) >= 0.0);
    let sz = select(-1.0, 1.0, dot(shape.axis_z.xyz, direction) >= 0.0);
    return shape.center.xyz
        + sx * shape.half_extents.x * shape.axis_x.xyz
        + sy * shape.half_extents.y * shape.axis_y.xyz
        + sz * shape.half_extents.z * shape.axis_z.xyz;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let count = arrayLength(&boxes);
    let pair = id.x;
    if (pair >= count * count) {
        return;
    }
    let i = pair / count;
    let j = pair % count;
    if (j <= i) {
        contacts[pair] = Contact(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
        return;
    }
    let a = boxes[i];
    let b = boxes[j];
    let delta = b.center.xyz - a.center.xyz;
    var result = AxisResult(false, 1e30, vec3<f32>(1.0, 0.0, 0.0));
    result = consider_axis(a, b, delta, a.axis_x.xyz, result);
    result = consider_axis(a, b, delta, a.axis_y.xyz, result);
    result = consider_axis(a, b, delta, a.axis_z.xyz, result);
    result = consider_axis(a, b, delta, b.axis_x.xyz, result);
    result = consider_axis(a, b, delta, b.axis_y.xyz, result);
    result = consider_axis(a, b, delta, b.axis_z.xyz, result);
    result = consider_axis(a, b, delta, cross(a.axis_x.xyz, b.axis_x.xyz), result);
    result = consider_axis(a, b, delta, cross(a.axis_x.xyz, b.axis_y.xyz), result);
    result = consider_axis(a, b, delta, cross(a.axis_x.xyz, b.axis_z.xyz), result);
    result = consider_axis(a, b, delta, cross(a.axis_y.xyz, b.axis_x.xyz), result);
    result = consider_axis(a, b, delta, cross(a.axis_y.xyz, b.axis_y.xyz), result);
    result = consider_axis(a, b, delta, cross(a.axis_y.xyz, b.axis_z.xyz), result);
    result = consider_axis(a, b, delta, cross(a.axis_z.xyz, b.axis_x.xyz), result);
    result = consider_axis(a, b, delta, cross(a.axis_z.xyz, b.axis_y.xyz), result);
    result = consider_axis(a, b, delta, cross(a.axis_z.xyz, b.axis_z.xyz), result);
    if (result.separated) {
        contacts[pair] = Contact(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
        return;
    }
    let witness_a = support(a, result.normal);
    let witness_b = support(b, -result.normal);
    let center_midpoint = (a.center.xyz + b.center.xyz) * 0.5;
    let normal_midpoint = (witness_a + witness_b) * 0.5;
    let point = center_midpoint + result.normal
        * dot(normal_midpoint - center_midpoint, result.normal);
    contacts[pair] = Contact(
        vec4<f32>(point, 0.0),
        vec4<f32>(result.normal, 0.0),
        vec4<f32>(max(result.depth, 0.0), 1.0, 0.0, 0.0),
    );
}
