// Same contact-row ABI as gpu_articulated_ground_contact.wgsl.
struct ContactRow {
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

@group(0) @binding(0) var<storage, read> rows: array<ContactRow>;
// Row index, environment index, inclusive/exclusive global link bounds.
@group(0) @binding(1) var<storage, read> owners: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> status: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> activity: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> parents: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read> mobility_roots: array<u32>;
@group(0) @binding(6) var<storage, read_write> wake_requests: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> component_wake: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> link_wake: array<u32>;
@group(0) @binding(9) var<storage, read_write> previous_geometry: array<u32>;
struct LossParams { enabled: vec4<u32> };
@group(0) @binding(10) var<uniform> loss_params: LossParams;
@group(0) @binding(11) var<storage, read> gravities: array<vec4<f32>>;

@compute @workgroup_size(64)
fn track_geometric_contact(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= arrayLength(&activity) { return; }
    let touching = (atomicLoad(&activity[id.x]) & 2u) != 0u;
    if loss_params.enabled.x != 0u && previous_geometry[id.x] != 0u && !touching {
        atomicOr(&wake_requests[id.x], 1u);
    }
    previous_geometry[id.x] = select(0u, 1u, touching);
}

@compute @workgroup_size(64)
fn initialize_components(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < arrayLength(&parents) { atomicStore(&parents[id.x], mobility_roots[id.x]); }
}

fn root(link: u32) -> u32 {
    var result = link;
    loop {
        let parent = atomicLoad(&parents[result]);
        if parent == result { break; }
        result = parent;
    }
    return result;
}

@compute @workgroup_size(64)
fn reduce_wake_requests(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= arrayLength(&parents) { return; }
    let request = atomicExchange(&wake_requests[id.x], 0u);
    if request != 0u { atomicOr(&component_wake[root(id.x)], 1u); }
}

@compute @workgroup_size(64)
fn broadcast_wake_requests(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= arrayLength(&parents) { return; }
    link_wake[id.x] = atomicLoad(&component_wake[root(id.x)]);
}

fn connect(first: u32, second: u32) {
    // Parents only decrease, preventing cycles even across workgroups.
    loop {
        let a = root(first);
        let b = root(second);
        if a == b { return; }
        let high = max(a, b);
        let low = min(a, b);
        let previous = atomicMin(&parents[high], low);
        if previous == high { return; }
        // Another union changed this root; retry so neither component is lost.
    }
}

fn finite4(value: vec4<f32>) -> bool {
    return all(abs(value) <= vec4<f32>(3.402823466e+38));
}

// A speculative supporting impulse can leave a sub-ULP positive gap after
// integrating f32 state. Preserve contact only within coordinate roundoff;
// positive gaps without a supporting normal impulse remain separated.
fn touching_at_precision(point: vec4<f32>, origin: vec4<f32>, normal_impulse: f32) -> bool {
    if origin.w == 0.0 { return false; }
    if point.w <= 0.0 { return true; }
    let coordinate_scale = max(abs(point.xyz), abs(origin.xyz));
    let scale = max(1e-6, max(max(coordinate_scale.x, coordinate_scale.y), coordinate_scale.z));
    return normal_impulse > 0.0 && point.w <= scale * 4.76837158203125e-7;
}

@compute @workgroup_size(64)
fn collect_activity(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= arrayLength(&owners) { return; }
    let owner = owners[id.x];
    if atomicLoad(&status[owner.y]) != 0u { return; }
    let row = rows[owner.x];
    if row.material.w == 79.0 || row.material.w == 80.0 {
        if row.indices.x < owner.z || row.indices.x >= owner.w ||
            (row.indices.z != 0xffffffffu &&
                (row.indices.z < owner.z || row.indices.z >= owner.w)) {
            atomicOr(&status[owner.y], 1u);
            return;
        }
        if row.indices.z != 0xffffffffu {
            connect(row.indices.x, row.indices.z);
        } else {
            let anchor_flags = 16u | select(0u, 32u, row.material.w == 80.0);
            atomicOr(&activity[row.indices.x], anchor_flags);
            if (dot(row.prescribed_linear.xyz, row.prescribed_linear.xyz) > 1e-12
                || dot(row.prescribed_angular.xyz, row.prescribed_angular.xyz) > 1e-12) {
                atomicOr(&wake_requests[row.indices.x], 1u);
            }
        }
        return;
    }
    let has_impulse = any(row.impulses.xyz != vec3<f32>(0.0));
    // Bilateral and scalar equality/friction rows carry no contact owner signs.
    if row.diagnostic_first_origin.w == 0.0 && row.diagnostic_second_origin.w == 0.0 {
        return;
    }
    let normal = row.previous_normal.xyz;
    if !finite4(vec4<f32>(row.impulses.xyz, 0.0)) ||
        !finite4(vec4<f32>(normal, 0.0)) || abs(dot(normal, normal) - 1.0) > 0.0001 {
        atomicOr(&status[owner.y], 1u);
        return;
    }
    let links = array<u32, 2>(row.indices.x, row.indices.z);
    let origins = array<vec4<f32>, 2>(row.diagnostic_first_origin, row.diagnostic_second_origin);
    let points = array<vec4<f32>, 2>(row.diagnostic_first, row.diagnostic_second);
    // Validate both owners before publishing either one.
    for (var i = 0u; i < 2u; i += 1u) {
        if origins[i].w == 0.0 { continue; }
        if !finite4(origins[i]) || !finite4(points[i]) || abs(origins[i].w) != 1.0 ||
            links[i] < owner.z || links[i] >= owner.w {
            atomicOr(&status[owner.y], 1u);
            return;
        }
    }
    let touching = touching_at_precision(points[0], origins[0], row.impulses.x) ||
        touching_at_precision(points[1], origins[1], row.impulses.x);
    let external_static_contact = touching &&
        ((origins[0].w == 0.0) != (origins[1].w == 0.0));
    // Prescribed external motion must keep its entire mobility/contact component
    // awake even when the transmitted velocity remains below sleep thresholds.
    if touching && (row.material.w == 14.0 || row.material.w == 15.0 || row.material.w == 16.0
        || row.material.w == 20.0 || row.material.w == 21.0 || row.material.w == 22.0
        || (row.material.w >= 52.0 && row.material.w <= 57.0)
        || (row.material.w >= 72.0 && row.material.w <= 77.0)
        || (row.material.w >= 60.0 && row.material.w <= 71.0)
        || row.material.w == 48.0 || row.material.w == 49.0 || row.material.w == 83.0
        || row.material.w == 17.0 || row.material.w == 18.0 || row.material.w == 19.0 || row.material.w == 24.0 || row.material.w == 25.0 || row.material.w == 28.0 || row.material.w == 29.0 || (row.material.w == 46.0 || row.material.w == 81.0)
        || row.material.w == 23.0 || row.material.w == 30.0 || row.material.w == 31.0 || row.material.w == 47.0 || row.material.w == 26.0 || row.material.w == 27.0 || row.material.w == 45.0 || row.material.w == 50.0 || row.material.w == 51.0) &&
        (dot(row.prescribed_linear.xyz, row.prescribed_linear.xyz) > 1e-12 ||
         dot(row.prescribed_angular.xyz, row.prescribed_angular.xyz) > 1e-12) {
        let motion_owner = select(row.indices.x, row.indices.z,
            (row.material.w >= 60.0 && row.material.w <= 71.0)
            || row.material.w == 48.0 || row.material.w == 49.0 || row.material.w == 83.0);
        atomicOr(&wake_requests[motion_owner], 1u);
    }
    var gravity_support = false;
    if external_static_contact {
        let gravity = gravities[owner.y].xyz;
        if !finite4(vec4<f32>(gravity, 0.0)) {
            atomicOr(&status[owner.y], 1u);
            return;
        }
        let scale = max(max(abs(gravity.x), abs(gravity.y)), abs(gravity.z));
        let sign = select(origins[1].w, origins[0].w, origins[0].w != 0.0);
        if scale > 0.0 {
            // Avoid a reciprocal in the subnormal range for very large gravity.
            var scaled = gravity;
            if scale > 1e20 { scaled *= 1e-20; }
            if scale < 1e-20 { scaled *= 1e20; }
            let magnitude = max(max(abs(scaled.x), abs(scaled.y)), abs(scaled.z));
            if magnitude > 0.0 {
                gravity_support = dot(normal * sign, scaled / magnitude) < -0.000001;
            }
        }
    }
    let flags = select(0u, 1u, has_impulse) | select(0u, 2u, touching) |
        select(0u, 4u, external_static_contact) | select(0u, 8u, gravity_support);
    if flags == 0u { return; }
    for (var i = 0u; i < 2u; i += 1u) {
        if origins[i].w != 0.0 { atomicOr(&activity[links[i]], flags); }
    }
    if origins[0].w != 0.0 && origins[1].w != 0.0 { connect(links[0], links[1]); }
}
