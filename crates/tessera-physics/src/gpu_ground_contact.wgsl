struct Sphere {
    center_radius: vec4<f32>,
}

struct Contact {
    point: vec4<f32>,
    normal: vec4<f32>,
    depth_hit: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> spheres: array<Sphere>;
@group(0) @binding(1) var<storage, read> ground: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> contacts: array<Contact>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= arrayLength(&spheres)) {
        return;
    }
    let sphere = spheres[index].center_radius;
    let half = ground[0].x;
    let hit = abs(sphere.x) <= half + sphere.w
        && abs(sphere.y) <= half + sphere.w
        && sphere.z - sphere.w <= 0.0;
    if (!hit) {
        contacts[index] = Contact(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
        return;
    }
    contacts[index] = Contact(
        vec4<f32>(sphere.xy, (sphere.z - sphere.w) * 0.5, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0),
        vec4<f32>(max(sphere.w - sphere.z, 0.0), 1.0, 0.0, 0.0)
    );
}
