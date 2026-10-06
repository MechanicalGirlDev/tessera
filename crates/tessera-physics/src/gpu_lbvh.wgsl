struct Aabb {
    lower: vec4<f32>,
    upper: vec4<f32>,
}

struct Domain {
    lower: vec4<f32>,
    upper: vec4<f32>,
}

struct MortonEntry {
    key: u32,
    collider: u32,
}

struct Node {
    lower: vec4<f32>,
    upper: vec4<f32>,
    left: u32,
    right: u32,
    parent: u32,
    range_first: u32,
    range_last: u32,
    _padding: u32,
}

struct Pair {
    a: u32,
    b: u32,
}

struct Params {
    count: u32,
    pair_capacity: u32,
    radix_shift: u32,
    filter_groups: u32,
}

@group(0) @binding(0) var<storage, read> bounds: array<Aabb>;
@group(0) @binding(1) var<storage, read_write> domain: Domain;
@group(0) @binding(2) var<storage, read_write> morton_in: array<MortonEntry>;
@group(0) @binding(3) var<storage, read_write> morton_out: array<MortonEntry>;
@group(0) @binding(4) var<storage, read_write> tree: array<Node>;
@group(0) @binding(5) var<storage, read_write> pairs: array<Pair>;
@group(0) @binding(6) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(7) var<uniform> params: Params;
// Radix offsets are replaced with environment IDs or packed collision masks during traversal.
@group(0) @binding(8) var<storage, read_write> radix_offsets: array<u32>;

var<workgroup> domain_mins: array<vec3<f32>, 128>;
var<workgroup> domain_maxs: array<vec3<f32>, 128>;
var<workgroup> radix_counts: array<atomic<u32>, 16>;
var<workgroup> radix_totals: array<u32, 16>;
var<workgroup> radix_digits: array<u32, 256>;

fn overlaps(lower_a: vec3<f32>, upper_a: vec3<f32>, lower_b: vec3<f32>, upper_b: vec3<f32>) -> bool {
    return all(lower_a <= upper_b) && all(lower_b <= upper_a);
}

fn pair_allowed(a: u32, b: u32) -> bool {
    if (params.filter_groups == 0u) { return true; }
    if (params.filter_groups == 1u) { return radix_offsets[a] == radix_offsets[b]; }
    let a_offset = a * 4u;
    let b_offset = b * 4u;
    return radix_offsets[a_offset] == radix_offsets[b_offset] &&
        (radix_offsets[a_offset + 1u] & radix_offsets[b_offset + 2u]) != 0u &&
        (radix_offsets[b_offset + 1u] & radix_offsets[a_offset + 2u]) != 0u;
}

fn expand_morton_bits(input: u32) -> u32 {
    var value = input & 0x000003ffu;
    value = (value | (value << 16u)) & 0x030000ffu;
    value = (value | (value << 8u)) & 0x0300f00fu;
    value = (value | (value << 4u)) & 0x030c30c3u;
    value = (value | (value << 2u)) & 0x09249249u;
    return value;
}

fn morton_3d(point: vec3<f32>) -> u32 {
    let quantized = vec3<u32>(clamp(point, vec3<f32>(0.0), vec3<f32>(0.999999)) * 1024.0);
    return expand_morton_bits(quantized.x) * 4u
        + expand_morton_bits(quantized.y) * 2u
        + expand_morton_bits(quantized.z);
}

@compute @workgroup_size(128)
fn compute_domain(
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let lane = local_id.x;
    var lower = vec3<f32>(3.402823466e+38);
    var upper = vec3<f32>(-3.402823466e+38);
    var index = lane;
    while (index < params.count) {
        let center = (bounds[index].lower.xyz + bounds[index].upper.xyz) * 0.5;
        lower = min(lower, center);
        upper = max(upper, center);
        index += 128u;
    }
    domain_mins[lane] = lower;
    domain_maxs[lane] = upper;
    workgroupBarrier();

    var stride = 64u;
    while (stride > 0u) {
        if (lane < stride) {
            domain_mins[lane] = min(domain_mins[lane], domain_mins[lane + stride]);
            domain_maxs[lane] = max(domain_maxs[lane], domain_maxs[lane + stride]);
        }
        workgroupBarrier();
        stride /= 2u;
    }
    if (lane == 0u) {
        domain.lower = vec4<f32>(domain_mins[0], 0.0);
        domain.upper = vec4<f32>(domain_maxs[0], 0.0);
    }
}

@compute @workgroup_size(64)
fn compute_morton(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= params.count) {
        return;
    }
    let center = (bounds[index].lower.xyz + bounds[index].upper.xyz) * 0.5;
    let extent = domain.upper.xyz - domain.lower.xyz;
    var normalized = vec3<f32>(0.5);
    if (extent.x > 1e-9) {
        normalized.x = (center.x - domain.lower.x) / extent.x;
    }
    if (extent.y > 1e-9) {
        normalized.y = (center.y - domain.lower.y) / extent.y;
    }
    if (extent.z > 1e-9) {
        normalized.z = (center.z - domain.lower.z) / extent.z;
    }
    morton_in[index] = MortonEntry(morton_3d(normalized), index);
}

@compute @workgroup_size(256)
fn radix_histogram(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let lane = local_id.x;
    if (lane < 16u) {
        atomicStore(&radix_counts[lane], 0u);
    }
    workgroupBarrier();
    let index = group_id.x * 256u + lane;
    if (index < params.count) {
        let digit = (morton_in[index].key >> params.radix_shift) & 15u;
        atomicAdd(&radix_counts[digit], 1u);
    }
    workgroupBarrier();
    if (lane < 16u) {
        let group_count = (params.count + 255u) / 256u;
        radix_offsets[lane * group_count + group_id.x] = atomicLoad(&radix_counts[lane]);
    }
}

@compute @workgroup_size(16)
fn radix_scan(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let digit = local_id.x;
    let group_count = (params.count + 255u) / 256u;
    var total = 0u;
    for (var group = 0u; group < group_count; group += 1u) {
        total += radix_offsets[digit * group_count + group];
    }
    radix_totals[digit] = total;
    workgroupBarrier();
    var offset = 0u;
    for (var previous = 0u; previous < digit; previous += 1u) {
        offset += radix_totals[previous];
    }
    for (var group = 0u; group < group_count; group += 1u) {
        let index = digit * group_count + group;
        let count = radix_offsets[index];
        radix_offsets[index] = offset;
        offset += count;
    }
}

@compute @workgroup_size(256)
fn radix_scatter(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let lane = local_id.x;
    let index = group_id.x * 256u + lane;
    var digit = 16u;
    if (index < params.count) {
        digit = (morton_in[index].key >> params.radix_shift) & 15u;
    }
    radix_digits[lane] = digit;
    workgroupBarrier();
    if (index < params.count) {
        var local_rank = 0u;
        for (var previous = 0u; previous < lane; previous += 1u) {
            if (radix_digits[previous] == digit) {
                local_rank += 1u;
            }
        }
        let group_count = (params.count + 255u) / 256u;
        let offset = radix_offsets[digit * group_count + group_id.x];
        morton_out[offset + local_rank] = morton_in[index];
    }
}

fn prefix_length(index: i32, other: i32) -> i32 {
    if (other < 0 || other >= i32(params.count)) {
        return -1;
    }
    let key = morton_in[u32(index)].key;
    let other_key = morton_in[u32(other)].key;
    if (key == other_key) {
        return 32 + i32(countLeadingZeros(u32(index) ^ u32(other)));
    }
    return i32(countLeadingZeros(key ^ other_key));
}

@compute @workgroup_size(64)
fn build_tree(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index + 1u >= params.count) {
        return;
    }
    let i = i32(index);
    let adjacent_delta = prefix_length(i, i + 1) - prefix_length(i, i - 1);
    let direction = select(-1, 1, adjacent_delta > 0);
    let delta_min = prefix_length(i, i - direction);
    var range_max = 2;
    for (var iteration = 0u; iteration < 32u; iteration += 1u) {
        if (prefix_length(i, i + range_max * direction) <= delta_min) {
            break;
        }
        range_max *= 2;
    }
    var range_length = 0;
    var step = range_max / 2;
    for (var iteration = 0u; iteration < 32u; iteration += 1u) {
        if (step < 1) {
            break;
        }
        if (prefix_length(i, i + (range_length + step) * direction) > delta_min) {
            range_length += step;
        }
        step /= 2;
    }
    let other_end = i + range_length * direction;
    let node_delta = prefix_length(i, other_end);
    var split_offset = 0;
    step = (range_length + 1) / 2;
    for (var iteration = 0u; iteration < 32u; iteration += 1u) {
        if (step < 1) {
            break;
        }
        if (prefix_length(i, i + (split_offset + step) * direction) > node_delta) {
            split_offset += step;
        }
        step = (step + 1) / 2;
    }
    let split = i + split_offset * direction + min(direction, 0);
    let first_leaf = i32(params.count - 1u);
    let left = select(split, first_leaf + split, min(i, other_end) == split);
    let right = select(split + 1, first_leaf + split + 1, max(i, other_end) == split + 1);
    tree[index].left = u32(left);
    tree[index].right = u32(right);
    tree[index].range_first = u32(min(i, other_end));
    tree[index].range_last = u32(max(i, other_end));
    tree[u32(left)].parent = index;
    tree[u32(right)].parent = index;
    if (index == 0u) {
        tree[0].parent = 0xffffffffu;
    }
}

@compute @workgroup_size(64)
fn refit_leaves(@builtin(global_invocation_id) id: vec3<u32>) {
    let sorted_index = id.x;
    if (sorted_index >= params.count) {
        return;
    }
    let first_leaf = params.count - 1u;
    let leaf_id = first_leaf + sorted_index;
    let collider = morton_in[sorted_index].collider;
    tree[leaf_id].lower = vec4<f32>(bounds[collider].lower.xyz, 0.0);
    tree[leaf_id].upper = vec4<f32>(bounds[collider].upper.xyz, 1.0);
    tree[leaf_id].left = collider;
}

// Each internal node owns a contiguous Morton range. Recomputing that range
// independently is portable across WebGPU backends and parallel across nodes.
@compute @workgroup_size(64)
fn refit_internal(@builtin(global_invocation_id) id: vec3<u32>) {
    let node = id.x;
    if (node + 1u >= params.count) {
        return;
    }
    let first = tree[node].range_first;
    let last = tree[node].range_last;
    var lower = vec3<f32>(3.402823466e+38);
    var upper = vec3<f32>(-3.402823466e+38);
    for (var sorted_index = first; sorted_index <= last; sorted_index += 1u) {
        let collider = morton_in[sorted_index].collider;
        lower = min(lower, bounds[collider].lower.xyz);
        upper = max(upper, bounds[collider].upper.xyz);
    }
    tree[node].lower = vec4<f32>(lower, 0.0);
    tree[node].upper = vec4<f32>(upper, 0.0);
}

@compute @workgroup_size(64)
fn find_pairs(@builtin(global_invocation_id) id: vec3<u32>) {
    let sorted_index = id.x;
    if (sorted_index >= params.count) {
        return;
    }
    let first_leaf = params.count - 1u;
    let leaf_id = first_leaf + sorted_index;
    let collider = tree[leaf_id].left;
    let query_lower = tree[leaf_id].lower.xyz;
    let query_upper = tree[leaf_id].upper.xyz;
    var current = 0u;
    var previous = 0xffffffffu;
    let max_steps = 4u * params.count;

    for (var step = 0u; step < max_steps; step += 1u) {
        let parent = tree[current].parent;
        var next = 0xffffffffu;
        if (previous == parent) {
            let hit = overlaps(query_lower, query_upper, tree[current].lower.xyz, tree[current].upper.xyz);
            if (!hit) {
                next = parent;
            } else if (current >= first_leaf) {
                let other = tree[current].left;
                if (collider < other && pair_allowed(collider, other)) {
                    let output = atomicAdd(&counters[0], 1u);
                    if (output < params.pair_capacity) {
                        pairs[output] = Pair(collider, other);
                    } else {
                        atomicStore(&counters[1], 1u);
                    }
                }
                next = parent;
            } else {
                next = tree[current].left;
            }
        } else if (current < first_leaf && previous == tree[current].left) {
            next = tree[current].right;
        } else {
            next = parent;
        }
        if (current == 0u && next == 0xffffffffu) {
            break;
        }
        previous = current;
        current = next;
    }
}
