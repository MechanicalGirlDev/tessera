struct Sphere {
    center_radius: vec4<f32>,
}

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

@group(0) @binding(0) var<storage, read> spheres: array<Sphere>;
@group(0) @binding(1) var<storage, read> boxes: array<Box>;
@group(0) @binding(2) var<storage, read_write> contacts: array<Contact>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let count = arrayLength(&boxes);
    let pair = id.x;
    if (pair >= arrayLength(&spheres) * count) {
        return;
    }
    let sphere = spheres[pair / count].center_radius;
    let box = boxes[pair % count];
    let offset = sphere.xyz - box.center.xyz;
    let local = vec3<f32>(
        dot(offset, box.axis_x.xyz),
        dot(offset, box.axis_y.xyz),
        dot(offset, box.axis_z.xyz),
    );
    var closest = clamp(local, -box.half_extents.xyz, box.half_extents.xyz);
    let delta = local - closest;
    let distance = length(delta);
    if (distance > sphere.w) {
        contacts[pair] = Contact(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
        return;
    }
    var normal_local: vec3<f32>;
    var depth: f32;
    if (distance > 1e-7) {
        normal_local = delta / distance;
        depth = max(sphere.w - distance, 0.0);
    } else {
        let gaps = box.half_extents.xyz - abs(local);
        var axis: u32 = 0u;
        if (gaps.y < gaps.x) { axis = 1u; }
        if (gaps.z < gaps[axis]) { axis = 2u; }
        let side = select(-1.0, 1.0, local[axis] >= 0.0);
        if (axis == 0u) {
            normal_local = vec3<f32>(side, 0.0, 0.0);
            closest.x = side * box.half_extents.x;
        } else if (axis == 1u) {
            normal_local = vec3<f32>(0.0, side, 0.0);
            closest.y = side * box.half_extents.y;
        } else {
            normal_local = vec3<f32>(0.0, 0.0, side);
            closest.z = side * box.half_extents.z;
        }
        depth = sphere.w + gaps[axis];
    }
    let normal = normal_local.x * box.axis_x.xyz
        + normal_local.y * box.axis_y.xyz
        + normal_local.z * box.axis_z.xyz;
    let point = box.center.xyz + closest.x * box.axis_x.xyz
        + closest.y * box.axis_y.xyz + closest.z * box.axis_z.xyz;
    contacts[pair] = Contact(
        vec4<f32>(point, 0.0),
        vec4<f32>(normal, 0.0),
        vec4<f32>(depth, 1.0, 0.0, 0.0),
    );
}
