struct Sphere {
    center_radius: vec4<f32>,
}

struct Pair {
    a: u32,
    b: u32,
}

struct Contact {
    point: vec4<f32>,
    normal: vec4<f32>,
    depth_hit: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> spheres: array<Sphere>;
@group(0) @binding(1) var<storage, read> pairs: array<Pair>;
@group(0) @binding(2) var<storage, read_write> contacts: array<Contact>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= arrayLength(&pairs)) {
        return;
    }
    let pair = pairs[index];
    let a = spheres[pair.a].center_radius;
    let b = spheres[pair.b].center_radius;
    let delta = b.xyz - a.xyz;
    let distance = length(delta);
    let overlap = a.w + b.w - distance;
    if (overlap < 0.0) {
        contacts[index] = Contact(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
        return;
    }
    var normal = vec3<f32>(1.0, 0.0, 0.0);
    if (distance > 1e-7) {
        normal = delta / distance;
    }
    let point_a = a.xyz + normal * a.w;
    let point_b = b.xyz - normal * b.w;
    contacts[index] = Contact(
        vec4<f32>((point_a + point_b) * 0.5, 0.0),
        vec4<f32>(normal, 0.0),
        vec4<f32>(overlap, 1.0, 0.0, 0.0)
    );
}
