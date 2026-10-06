struct RigidState {
    position_inverse_mass: vec4<f32>,
    orientation: vec4<f32>,
    linear_velocity: vec4<f32>,
    angular_velocity: vec4<f32>,
    inverse_inertia_sleep: vec4<f32>,
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

struct Params {
    body_count: u32,
    pair_count: u32,
    ground_half_extent: f32,
    ground_enabled: u32,
    ground_memberships: u32,
    ground_filter: u32,
    padding: vec2<u32>,
}

struct CollisionGroups {
    memberships: u32,
    filter_mask: u32,
}

@group(0) @binding(0) var<storage, read> states: array<RigidState>;
@group(0) @binding(1) var<storage, read> radii: array<f32>;
@group(0) @binding(2) var<storage, read> pairs: array<Pair>;
@group(0) @binding(3) var<storage, read_write> pair_contacts: array<Contact>;
@group(0) @binding(4) var<storage, read_write> ground_contacts: array<Contact>;
@group(0) @binding(5) var<uniform> params: Params;
@group(0) @binding(6) var<storage, read> collision_groups: array<CollisionGroups>;

fn allows(a: CollisionGroups, b: CollisionGroups) -> bool {
    return (a.memberships & b.filter_mask) != 0u && (b.memberships & a.filter_mask) != 0u;
}

@compute @workgroup_size(64)
fn pair_contacts_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= params.pair_count) { return; }
    let pair = pairs[index];
    if (!allows(collision_groups[pair.a], collision_groups[pair.b])) {
        pair_contacts[index] = Contact(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
        return;
    }
    let a = states[pair.a].position_inverse_mass.xyz;
    let b = states[pair.b].position_inverse_mass.xyz;
    let radius_a = radii[pair.a];
    let radius_b = radii[pair.b];
    let delta = b - a;
    let distance = length(delta);
    let depth = radius_a + radius_b - distance;
    if (depth < -bitcast<f32>(params.padding.x)) {
        pair_contacts[index] = Contact(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
        return;
    }
    var normal = vec3<f32>(1.0, 0.0, 0.0);
    if (distance > 1e-7) { normal = delta / distance; }
    let point_a = a + normal * radius_a;
    let point_b = b - normal * radius_b;
    pair_contacts[index] = Contact(
        vec4<f32>((point_a + point_b) * 0.5, 0.0),
        vec4<f32>(normal, 0.0),
        vec4<f32>(depth, 1.0, 0.0, 0.0),
    );
}

@compute @workgroup_size(64)
fn ground_contacts_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= params.body_count) { return; }
    let ground = CollisionGroups(params.ground_memberships, params.ground_filter);
    if (!allows(collision_groups[index], ground)) {
        ground_contacts[index] = Contact(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
        return;
    }
    let center = states[index].position_inverse_mass.xyz;
    let radius = radii[index];
    let half = params.ground_half_extent;
    let hit = params.ground_enabled != 0u
        && abs(center.x) <= half + radius
        && abs(center.y) <= half + radius
        && center.z - radius <= bitcast<f32>(params.padding.y);
    if (!hit) {
        ground_contacts[index] = Contact(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
        return;
    }
    ground_contacts[index] = Contact(
        vec4<f32>(center.xy, (center.z - radius) * 0.5, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0),
        vec4<f32>(radius - center.z, 1.0, 0.0, 0.0),
    );
}
