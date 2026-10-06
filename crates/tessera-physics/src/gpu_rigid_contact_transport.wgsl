struct State {
    position_inverse_mass: vec4<f32>, orientation: vec4<f32>,
    linear_velocity: vec4<f32>, angular_velocity: vec4<f32>, inverse_inertia_sleep: vec4<f32>,
}
struct Pair { a: u32, b: u32, }
struct Contact { point: vec4<f32>, normal: vec4<f32>, depth_hit: vec4<f32>, }
struct Anchor { local_a_depth: vec4<f32>, local_b_active: vec4<f32>, normal: vec4<f32>, }
struct Params { counts: vec4<u32>, totals: vec4<u32>, }
struct Address { a: u32, b: u32, slot: u32, ground: bool, }
@group(0) @binding(0) var<storage, read> states: array<State>;
@group(0) @binding(1) var<storage, read> pairs: array<Pair>;
@group(0) @binding(2) var<storage, read_write> pair_contacts: array<Contact>;
@group(0) @binding(3) var<storage, read_write> ground_contacts: array<Contact>;
@group(0) @binding(4) var<storage, read_write> anchors: array<Anchor>;
@group(0) @binding(5) var<uniform> params: Params;
fn rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}
fn local(state: State, point: vec3<f32>) -> vec3<f32> {
    return rotate(vec4<f32>(-state.orientation.xyz, state.orientation.w), point - state.position_inverse_mass.xyz);
}
fn world(state: State, point: vec3<f32>) -> vec3<f32> {
    return state.position_inverse_mass.xyz + rotate(state.orientation, point);
}
fn address(index: u32) -> Address {
    let pair_slots = params.counts.x * params.counts.z;
    if (index < pair_slots) {
        var pair_index = index;
        if (index >= params.counts.x) { pair_index = (index - params.counts.x) / (params.counts.z - 1u); }
        let pair = pairs[pair_index];
        return Address(pair.a, pair.b, index, false);
    }
    let slot = index - pair_slots;
    var body = slot;
    if (slot >= params.counts.y) { body = (slot - params.counts.y) / (params.counts.w - 1u); }
    return Address(0u, body, slot, true);
}
fn read_contact(entry: Address) -> Contact {
    if (entry.ground) { return ground_contacts[entry.slot]; }
    return pair_contacts[entry.slot];
}
fn write_contact(entry: Address, contact: Contact) {
    if (entry.ground) { ground_contacts[entry.slot] = contact; }
    else { pair_contacts[entry.slot] = contact; }
}
@compute @workgroup_size(64)
fn capture(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.totals.x) { return; }
    let entry = address(id.x);
    let contact = read_contact(entry);
    if (contact.depth_hit.y == 0.0) {
        anchors[id.x] = Anchor(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
        return;
    }
    var point_a = contact.point.xyz;
    if (!entry.ground) { point_a = local(states[entry.a], point_a); }
    anchors[id.x] = Anchor(vec4<f32>(point_a, contact.depth_hit.x),
        vec4<f32>(local(states[entry.b], contact.point.xyz), 1.0), vec4<f32>(contact.normal.xyz, 0.0));
}
@compute @workgroup_size(64)
fn refresh(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.totals.x) { return; }
    let entry = address(id.x);
    let anchor = anchors[id.x];
    if (anchor.local_b_active.w == 0.0) {
        write_contact(entry, Contact(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0)));
        return;
    }
    var point_a = anchor.local_a_depth.xyz;
    if (!entry.ground) { point_a = world(states[entry.a], point_a); }
    let point_b = world(states[entry.b], anchor.local_b_active.xyz);
    let separation = dot(point_b - point_a, anchor.normal.xyz) - anchor.local_a_depth.w;
    write_contact(entry, Contact(vec4<f32>((point_a + point_b) * 0.5, 0.0), anchor.normal,
        vec4<f32>(-separation, 1.0, 0.0, 0.0)));
}
