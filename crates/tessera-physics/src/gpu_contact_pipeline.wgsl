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
@group(0) @binding(1) var<storage, read_write> pairs: array<Pair>;
@group(0) @binding(2) var<storage, read_write> overlaps: array<u32>;
@group(0) @binding(3) var<storage, read_write> contacts: array<Contact>;

fn row_start(row: u32, count: u32) -> u32 {
    return row * (2u * count - row - 1u) / 2u;
}

@compute @workgroup_size(64)
fn generate_pairs(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= arrayLength(&pairs)) {
        return;
    }
    let count = arrayLength(&spheres);
    var low = 0u;
    var high = count - 1u;
    while (low < high) {
        let middle = (low + high + 1u) / 2u;
        if (row_start(middle, count) <= index) {
            low = middle;
        } else {
            high = middle - 1u;
        }
    }
    pairs[index] = Pair(low, low + 1u + index - row_start(low, count));
}

@compute @workgroup_size(64)
fn broad_phase(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= arrayLength(&pairs)) {
        return;
    }
    let pair = pairs[index];
    let a = spheres[pair.a].center_radius;
    let b = spheres[pair.b].center_radius;
    let lower_a = a.xyz - vec3<f32>(a.w);
    let upper_a = a.xyz + vec3<f32>(a.w);
    let lower_b = b.xyz - vec3<f32>(b.w);
    let upper_b = b.xyz + vec3<f32>(b.w);
    overlaps[index] = select(0u, 1u, all(lower_a <= upper_b) && all(lower_b <= upper_a));
}

@compute @workgroup_size(64)
fn narrow_phase(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= arrayLength(&pairs)) {
        return;
    }
    if (overlaps[index] == 0u) {
        contacts[index] = Contact(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
        return;
    }
    let pair = pairs[index];
    let a = spheres[pair.a].center_radius;
    let b = spheres[pair.b].center_radius;
    let delta = b.xyz - a.xyz;
    let distance = length(delta);
    let depth = a.w + b.w - distance;
    if (depth < 0.0) {
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
        vec4<f32>(depth, 1.0, 0.0, 0.0)
    );
}
