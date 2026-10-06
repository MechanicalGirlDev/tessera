@group(0) @binding(9) var<storage, read> candidate_counter: array<u32>;
@group(0) @binding(10) var<storage, read_write> solve_status: array<u32>;
var<workgroup> colored_island_meta: array<u32, 3>;

fn candidate_pair_slot(pair: Pair, point: u32) -> u32 {
    if (params.flags.x == 0u) { return pair_slot(0u, point); }
    let a = min(pair.a, pair.b);
    let b = max(pair.a, pair.b);
    let row = a * (2u * params.flags.z - a - 1u) / 2u;
    return (row + b - a - 1u) * params.flags.w + point;
}

fn zero_impulse() -> ImpulseState {
    return ImpulseState(vec4<f32>(0.0), vec4<f32>(0.0),
                        vec4<f32>(0.0), vec4<f32>(0.0));
}

// The status buffer holds roots, per-root counts/starts/cursors, then compact indices.
fn pair_count_offset() -> u32 { return 1u + params.flags.z; }
fn pair_start_offset() -> u32 { return 1u + 2u * params.flags.z; }
fn pair_cursor_offset() -> u32 { return 1u + 3u * params.flags.z; }
fn body_count_offset() -> u32 { return 1u + 4u * params.flags.z; }
fn body_start_offset() -> u32 { return 1u + 5u * params.flags.z; }
fn body_cursor_offset() -> u32 { return 1u + 6u * params.flags.z; }
fn pair_indices_offset() -> u32 { return 1u + 7u * params.flags.z; }
fn body_indices_offset() -> u32 { return pair_indices_offset() + params.counts.y; }
fn color_indices_offset() -> u32 { return body_indices_offset() + params.flags.z; }

fn island_pair_index(index: u32, root: u32, islanded: bool) -> u32 {
    if (!islanded) { return index; }
    return solve_status[pair_indices_offset() +
        solve_status[pair_start_offset() + root] + index];
}

fn island_body_index(index: u32, root: u32, islanded: bool) -> u32 {
    if (!islanded) { return index; }
    return solve_status[body_indices_offset() +
        solve_status[body_start_offset() + root] + index];
}

fn candidate_root(body: u32) -> u32 {
    var root = body;
    loop {
        let parent = solve_status[1u + root];
        if (parent == root) { break; }
        root = parent;
    }
    return root;
}

@compute @workgroup_size(64)
fn candidate_label(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x != 0u) { return; }
    let pair_count = candidate_counter[0];
    let body_count = params.flags.z;
    if (candidate_counter[1] != 0u || pair_count > params.counts.y) {
        solve_status[0] = 1u;
        return;
    }
    solve_status[0] = 0u;
    for (var body = 0u; body < body_count; body++) {
        solve_status[1u + body] = body;
        solve_status[pair_count_offset() + body] = 0u;
        solve_status[body_count_offset() + body] = 0u;
    }
    for (var index = 0u; index < pair_count; index++) {
        let pair = pairs[index];
        if (pair.a >= body_count || pair.b >= body_count || pair.a == pair.b) {
            solve_status[0] = 1u;
            return;
        }
        let a = candidate_root(pair.a);
        let b = candidate_root(pair.b);
        if (a != b) {
            solve_status[1u + max(a, b)] = min(a, b);
        }
    }
    for (var body = 0u; body < body_count; body++) {
        solve_status[1u + body] = candidate_root(body);
    }
    for (var index = 0u; index < pair_count; index++) {
        let root = solve_status[1u + pairs[index].a];
        solve_status[pair_count_offset() + root] += 1u;
    }
    if (params.counts.z != 0u) {
        for (var body = 0u; body < body_count; body++) {
            let root = solve_status[1u + body];
            solve_status[body_count_offset() + root] += 1u;
        }
    }
    let budget = 20000u / params.counts.w;
    for (var body = 0u; body < body_count; body++) {
        if (solve_status[1u + body] == body &&
            solve_status[pair_count_offset() + body] * params.flags.w +
            solve_status[body_count_offset() + body] * params.flags.y > budget) {
            solve_status[0] = 1u;
            return;
        }
    }
    var next_pair = 0u;
    var next_body = 0u;
    for (var body = 0u; body < body_count; body++) {
        solve_status[pair_start_offset() + body] = next_pair;
        solve_status[body_start_offset() + body] = next_body;
        next_pair += solve_status[pair_count_offset() + body];
        next_body += solve_status[body_count_offset() + body];
        solve_status[pair_cursor_offset() + body] = 0u;
        solve_status[body_cursor_offset() + body] = 0u;
    }
    for (var index = 0u; index < pair_count; index++) {
        let root = solve_status[1u + pairs[index].a];
        let slot = solve_status[pair_start_offset() + root] +
            solve_status[pair_cursor_offset() + root];
        solve_status[pair_indices_offset() + slot] = index;
        solve_status[pair_cursor_offset() + root] += 1u;
    }
    if (params.counts.z != 0u) {
        for (var body = 0u; body < body_count; body++) {
            let root = solve_status[1u + body];
            let slot = solve_status[body_start_offset() + root] +
                solve_status[body_cursor_offset() + root];
            solve_status[body_indices_offset() + slot] = body;
            solve_status[body_cursor_offset() + root] += 1u;
        }
    }
    // The cursors are free after compaction. Reuse them for per-body color masks
    // and per-island color counts. A dense island falls back to serial solving.
    for (var body = 0u; body < body_count; body++) {
        solve_status[pair_cursor_offset() + body] = 0u;
        solve_status[body_cursor_offset() + body] = 0u;
    }
    for (var index = 0u; index < pair_count; index++) {
        let pair = pairs[index];
        let root = solve_status[1u + pair.a];
        if (solve_status[body_cursor_offset() + root] == 0xffffffffu) { continue; }
        let used = solve_status[pair_cursor_offset() + pair.a] |
                   solve_status[pair_cursor_offset() + pair.b];
        var color = 0u;
        loop {
            if (color == 32u) { break; }
            if ((used & (1u << color)) == 0u) { break; }
            color += 1u;
        }
        if (color == 32u) {
            solve_status[body_cursor_offset() + root] = 0xffffffffu;
            continue;
        }
        solve_status[color_indices_offset() + index] = color;
        solve_status[pair_cursor_offset() + pair.a] |= 1u << color;
        solve_status[pair_cursor_offset() + pair.b] |= 1u << color;
        solve_status[body_cursor_offset() + root] =
            max(solve_status[body_cursor_offset() + root], color + 1u);
    }
}

fn candidate_warm(pair_count: u32, root: u32, islanded: bool) {
    var local_pair_count = pair_count;
    var local_body_count = params.flags.z;
    if (islanded) {
        local_pair_count = solve_status[pair_count_offset() + root];
        local_body_count = solve_status[body_count_offset() + root];
    }
    for (var local_pair = 0u; local_pair < local_pair_count; local_pair++) {
        let pair_index = island_pair_index(local_pair, root, islanded);
        let pair = pairs[pair_index];
        for (var point = 0u; point < params.flags.w; point++) {
            let slot = select(pair_slot(pair_index, point),
                              candidate_pair_slot(pair, point), params.flags.x != 0u);
            if (params.flags.x == 0u || params.counts.x == 1u ||
                impulses[slot].anchor_a.w != f32(params.counts.x - 1u)) {
                impulses[slot] = zero_impulse();
            } else {
                warm_one(pair.a, pair.b, pair_contact(pair_index, point), slot, false);
            }
        }
    }
    if (params.counts.z != 0u) {
        for (var local_body = 0u; local_body < local_body_count; local_body++) {
            let body = island_body_index(local_body, root, islanded);
            for (var point = 0u; point < params.flags.y; point++) {
                let slot = ground_slot(body, point);
                if (params.flags.x == 0u || params.counts.x == 1u) {
                    impulses[slot] = zero_impulse();
                } else {
                    warm_one(body, body, ground_contact(body, point), slot, true);
                }
            }
        }
    }
}

fn candidate_solve(pair_count: u32, root: u32, islanded: bool) {
    candidate_warm(pair_count, root, islanded);
    var local_pair_count = pair_count;
    var local_body_count = params.flags.z;
    if (islanded) {
        local_pair_count = solve_status[pair_count_offset() + root];
        local_body_count = solve_status[body_count_offset() + root];
    }
    for (var iteration = 0u; iteration < params.counts.w; iteration++) {
        if (params.counts.z != 0u) {
            for (var local_body = 0u; local_body < local_body_count; local_body++) {
                let body = island_body_index(local_body, root, islanded);
                for (var point = 0u; point < params.flags.y; point++) {
                    solve_one(body, body, ground_contact(body, point),
                        ground_slot(body, point), true, iteration);
                }
            }
        }
        for (var local_pair = 0u; local_pair < local_pair_count; local_pair++) {
            let pair_index = island_pair_index(local_pair, root, islanded);
            let pair = pairs[pair_index];
            for (var point = 0u; point < params.flags.w; point++) {
                solve_one(pair.a, pair.b, pair_contact(pair_index, point),
                    select(pair_slot(pair_index, point), candidate_pair_slot(pair, point),
                           params.flags.x != 0u), false, iteration);
            }
        }
    }
    if (params.flags.x != 0u) {
        for (var local_pair = 0u; local_pair < local_pair_count; local_pair++) {
            let pair_index = island_pair_index(local_pair, root, islanded);
            let pair = pairs[pair_index];
            for (var point = 0u; point < params.flags.w; point++) {
                let slot = candidate_pair_slot(pair, point);
                impulses[slot].anchor_a.w = f32(params.counts.x);
            }
        }
    }
}

@compute @workgroup_size(64)
fn candidate_main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x != 0u) { return; }
    let pair_count = candidate_counter[0];
    let ground_slots = select(0u, params.flags.z * params.flags.y, params.counts.z != 0u);
    let budget = 20000u / params.counts.w;
    if (candidate_counter[1] != 0u || pair_count > params.counts.y ||
        ground_slots > budget) {
        solve_status[0] = 1u;
        return;
    }
    let max_pairs = (budget - ground_slots) / params.flags.w;
    if (pair_count > max_pairs) {
        solve_status[0] = 1u;
        return;
    }
    solve_status[0] = 0u;
    candidate_solve(pair_count, 0u, false);
}

@compute @workgroup_size(64)
fn candidate_islands(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.flags.z || solve_status[0] != 0u ||
        solve_status[1u + id.x] != id.x ||
        solve_status[body_cursor_offset() + id.x] != 0xffffffffu) { return; }
    candidate_solve(candidate_counter[0], id.x, true);
}

@compute @workgroup_size(64)
fn candidate_island_warm(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.flags.z || solve_status[0] != 0u ||
        solve_status[1u + id.x] != id.x ||
        solve_status[body_cursor_offset() + id.x] == 0xffffffffu) { return; }
    candidate_warm(candidate_counter[0], id.x, true);
}

@compute @workgroup_size(64)
fn candidate_colored(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let root = group.x;
    if (root >= params.flags.z) { return; }
    if (lane == 0u) {
        let island_enabled = solve_status[0] == 0u && solve_status[1u + root] == root &&
            solve_status[body_cursor_offset() + root] != 0xffffffffu;
        colored_island_meta[0] = select(0u, solve_status[body_cursor_offset() + root], island_enabled);
        colored_island_meta[1] = select(0u, solve_status[pair_count_offset() + root], island_enabled);
        colored_island_meta[2] = select(0u, solve_status[body_count_offset() + root], island_enabled);
    }
    let color_count = workgroupUniformLoad(&colored_island_meta[0]);
    let pair_count = workgroupUniformLoad(&colored_island_meta[1]);
    let ground_count = workgroupUniformLoad(&colored_island_meta[2]);
    for (var iteration = 0u; iteration < params.counts.w; iteration++) {
        if (params.counts.z != 0u) {
            for (var tile = 0u; tile < (ground_count + 63u) / 64u; tile++) {
                let local_body = tile * 64u + lane;
                for (var point = 0u; point < params.flags.y; point++) {
                    if (local_body < ground_count) {
                        let body = island_body_index(local_body, root, true);
                        solve_one(body, body, ground_contact(body, point),
                            ground_slot(body, point), true, iteration);
                    }
                }
            }
        }
        storageBarrier();
        for (var color = 0u; color < color_count; color++) {
            for (var tile = 0u; tile < (pair_count + 63u) / 64u; tile++) {
                let local_pair = tile * 64u + lane;
                for (var point = 0u; point < params.flags.w; point++) {
                    if (local_pair < pair_count) {
                        let pair_index = island_pair_index(local_pair, root, true);
                        if (solve_status[color_indices_offset() + pair_index] == color) {
                            let pair = pairs[pair_index];
                            solve_one(pair.a, pair.b, pair_contact(pair_index, point),
                                select(pair_slot(pair_index, point), candidate_pair_slot(pair, point),
                                       params.flags.x != 0u), false, iteration);
                        }
                    }
                }
            }
            storageBarrier();
        }
    }
    if (params.flags.x != 0u) {
        for (var local_pair = lane; local_pair < pair_count; local_pair += 64u) {
            let pair_index = island_pair_index(local_pair, root, true);
            let pair = pairs[pair_index];
            for (var point = 0u; point < params.flags.w; point++) {
                impulses[candidate_pair_slot(pair, point)].anchor_a.w = f32(params.counts.x);
            }
        }
    }
}
