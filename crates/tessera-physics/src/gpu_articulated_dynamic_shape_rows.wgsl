struct Pair {
    a: u32,
    b: u32,
};

struct DynamicShape {
    indices: vec4<u32>,
    placement: vec4<u32>,
    center_radius: vec4<f32>,
    center_of_mass: vec4<f32>,
    material: vec4<f32>,
    axis_end: vec4<f32>,
    orientation: vec4<f32>,
    counts: vec4<u32>,
};

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

@group(0) @binding(0) var<storage, read> pairs: array<Pair>;
@group(0) @binding(1) var<storage, read> counter: array<u32>;
@group(0) @binding(2) var<storage, read> shapes: array<DynamicShape>;
@group(0) @binding(3) var<storage, read_write> rows: array<ContactRow>;
@group(0) @binding(4) var<storage, read> pair_slots: array<u32>;
@group(0) @binding(5) var<storage, read_write> active_flags: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read> row_indices: array<u32>;

fn combine_coefficient(left: f32, right: f32, left_rule: u32, right_rule: u32) -> f32 {
    let rule = max(left_rule, right_rule);
    if (rule == 1u) { return (left + right) * 0.5; }
    if (rule == 2u) { return min(left, right); }
    if (rule == 3u) { return left * right; }
    return max(left, right);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
    if (invocation.x >= counter[0]) { return; }
    let pair = pairs[invocation.x];
    let first_index = min(pair.a, pair.b);
    let second_index = max(pair.a, pair.b);
    let first = shapes[first_index];
    let second = shapes[second_index];
    if (first.placement.x != second.placement.x
        || first.placement.z != second.placement.z) { return; }
    let count = first.placement.x;
    let a = first.placement.y;
    let b = second.placement.y;
    if (a >= b || b >= count) { return; }
    let pair_index = a * (2u * count - a - 1u) / 2u + b - a - 1u;
    let slot = pair_slots[first.counts.w + pair_index];
    if (slot == 0xffffffffu) { return; }
    let row_index = first.placement.z + slot;
    let prior = rows[row_index];
    let active_before = prior.center_radius.w > 0.0;
    let impulses = select(vec4<f32>(0.0), prior.impulses, active_before);
    let previous_normal = select(vec4<f32>(0.0), prior.previous_normal, active_before);
    var restitution = max(first.material.x, second.material.x);
    var friction = sqrt(first.material.z) * sqrt(second.material.z);
    if (first.counts.z != 0u && second.counts.z != 0u) {
        restitution = combine_coefficient(first.material.x, second.material.x,
            first.counts.z, second.counts.z);
    }
    if (first.counts.y != 0u && second.counts.y != 0u) {
        friction = combine_coefficient(first.material.z, second.material.z,
            first.counts.y, second.counts.y);
    }
    if (first.indices.z == 2u) {
        for (var point = 0u; point < 4u; point++) {
            let index = row_index + point;
            let previous = rows[index];
            let enabled_before = previous.center_radius.w > 0.0;
            rows[index] = ContactRow(
                vec4<u32>(first.indices.x, first.indices.y,
                    second.indices.x, second.indices.y),
                vec4<f32>(first.center_radius.xyz, first.axis_end.x),
                vec4<f32>(first.center_of_mass.xyz, first.axis_end.z),
                vec4<f32>(second.center_radius.xyz, first.axis_end.y),
                vec4<f32>(restitution, first.material.y, friction, 8.0),
                vec4<f32>(second.axis_end.xyz, f32(point)),
                second.center_of_mass,
                second.orientation,
                first.orientation,
                select(vec4<f32>(0.0), previous.impulses, enabled_before),
                select(vec4<f32>(0.0), previous.previous_normal, enabled_before),
                vec4<f32>(0.0),
                vec4<f32>(0.0),
                vec4<f32>(0.0),
                vec4<f32>(0.0),
                vec4<f32>(0.0),
                vec4<f32>(0.0),
            );
            atomicStore(&active_flags[first.placement.w + slot + point], 1u);
        }
        return;
    } else if (second.indices.z == 2u) {
        let mode = select(6.0, 7.0, first.indices.z == 1u);
        let first_axis_end = select(vec4<f32>(0.0), first.axis_end,
            first.indices.z == 1u);
        let row = ContactRow(
            vec4<u32>(first.indices.x, first.indices.y, second.indices.x, second.indices.y),
            first.center_radius,
            first.center_of_mass,
            second.center_radius,
            vec4<f32>(restitution, first.material.y, friction, mode),
            second.axis_end,
            second.center_of_mass,
            second.orientation,
            first_axis_end,
            impulses,
            previous_normal,
            vec4<f32>(0.0),
            vec4<f32>(0.0),
            vec4<f32>(0.0),
            vec4<f32>(0.0),
            vec4<f32>(0.0),
            vec4<f32>(0.0),
        );
        rows[row_index] = row;
        if (first.indices.z == 1u) {
            let prior_side = rows[row_index + 1u];
            let side_active_before = prior_side.center_radius.w > 0.0;
            var side = row;
            side.material.w = 9.0;
            side.impulses = select(vec4<f32>(0.0), prior_side.impulses, side_active_before);
            side.previous_normal = select(vec4<f32>(0.0), prior_side.previous_normal,
                side_active_before);
            rows[row_index + 1u] = side;
            atomicStore(&active_flags[first.placement.w + slot + 1u], 1u);
        }
    } else if (first.indices.z == 1u) {
        let row = ContactRow(
            vec4<u32>(first.indices.x, first.indices.y, second.indices.x, second.indices.y),
            first.center_radius,
            first.center_of_mass,
            second.center_radius,
            vec4<f32>(restitution, first.material.y, friction, 3.0),
            first.axis_end,
            second.center_of_mass,
            second.axis_end,
            vec4<f32>(0.0),
            impulses,
            previous_normal,
            vec4<f32>(0.0),
            vec4<f32>(0.0),
            vec4<f32>(0.0),
            vec4<f32>(0.0),
            vec4<f32>(0.0),
            vec4<f32>(0.0),
        );
        rows[row_index] = row;
        let prior_side = rows[row_index + 1u];
        let side_active_before = prior_side.center_radius.w > 0.0;
        var side = row;
        side.material.w = 4.0;
        side.impulses = select(vec4<f32>(0.0), prior_side.impulses, side_active_before);
        side.previous_normal = select(vec4<f32>(0.0), prior_side.previous_normal,
            side_active_before);
        rows[row_index + 1u] = side;
        atomicStore(&active_flags[first.placement.w + slot], 1u);
        atomicStore(&active_flags[first.placement.w + slot + 1u], 1u);
        return;
    } else if (second.indices.z == 1u) {
        rows[row_index] = ContactRow(
            vec4<u32>(second.indices.x, second.indices.y, first.indices.x, first.indices.y),
            second.center_radius,
            second.center_of_mass,
            first.center_radius,
            vec4<f32>(restitution, first.material.y, friction, 2.0),
            second.axis_end,
            first.center_of_mass,
            vec4<f32>(0.0),
            vec4<f32>(0.0),
            impulses,
            previous_normal,
            vec4<f32>(0.0),
            vec4<f32>(0.0),
            vec4<f32>(0.0),
            vec4<f32>(0.0),
            vec4<f32>(0.0),
            vec4<f32>(0.0),
        );
    } else {
        rows[row_index] = ContactRow(
            vec4<u32>(first.indices.x, first.indices.y, second.indices.x, second.indices.y),
            first.center_radius,
            first.center_of_mass,
            vec4<f32>(0.0),
            vec4<f32>(restitution, first.material.y, friction, 1.0),
            second.center_radius,
            second.center_of_mass,
            vec4<f32>(0.0),
            vec4<f32>(0.0),
            impulses,
            previous_normal,
            vec4<f32>(0.0),
            vec4<f32>(0.0),
            vec4<f32>(0.0),
            vec4<f32>(0.0),
            vec4<f32>(0.0),
            vec4<f32>(0.0),
        );
    }
    atomicStore(&active_flags[first.placement.w + slot], 1u);
}

@compute @workgroup_size(64)
fn clear_inactive(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let id = invocation.x;
    if (id >= arrayLength(&row_indices) || atomicLoad(&active_flags[id]) != 0u) { return; }
    let row_index = row_indices[id];
    var row = rows[row_index];
    row.center_radius.w = 0.0;
    row.impulses = vec4<f32>(0.0);
    row.previous_normal = vec4<f32>(0.0);
    row.diagnostic_first_origin = vec4<f32>(0.0);
    row.diagnostic_second_origin = vec4<f32>(0.0);
    rows[row_index] = row;
}
