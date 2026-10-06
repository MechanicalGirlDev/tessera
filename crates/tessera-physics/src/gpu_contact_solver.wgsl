struct Settings {
    dimensions: vec4<u32>,
};

struct ContactData {
    effective: vec4<f32>,
    targets: vec4<f32>,
    offsets: vec4<u32>,
};

struct IslandRange {
    start_count: vec4<u32>,
    color_start_count: vec4<u32>,
};

@group(0) @binding(0) var<storage, read_write> velocity: array<f32>;
@group(0) @binding(1) var<storage, read> jacobians: array<f32>;
@group(0) @binding(2) var<storage, read> responses: array<f32>;
@group(0) @binding(3) var<storage, read> contacts: array<ContactData>;
@group(0) @binding(4) var<storage, read_write> impulses: array<vec4<f32>>;
@group(0) @binding(5) var<uniform> settings: Settings;
@group(0) @binding(6) var<storage, read> island_ranges: array<IslandRange>;
@group(0) @binding(7) var<storage, read> island_contacts: array<u32>;
@group(0) @binding(8) var<storage, read> color_ranges: array<vec2<u32>>;

fn row_dot(offset: u32, velocity_offset: u32, width: u32) -> f32 {
    var value = 0.0;
    for (var coordinate = 0u; coordinate < width; coordinate++) {
        let coefficient = jacobians[offset + coordinate];
        if coefficient != 0.0 {
            value += coefficient * velocity[velocity_offset + coordinate];
        }
    }
    return value;
}

fn apply_response(offset: u32, velocity_offset: u32, width: u32, impulse: f32) {
    for (var coordinate = 0u; coordinate < width; coordinate++) {
        let response = responses[offset + coordinate];
        if response != 0.0 {
            velocity[velocity_offset + coordinate] += response * impulse;
        }
    }
}

@compute @workgroup_size(64)
fn solve(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    if group.x >= settings.dimensions.w {
        return;
    }
    let island = island_ranges[group.x];
    let velocity_offset = island.start_count.z;
    let width = island.start_count.w;
    // First settle support, then iterate the coupled restitution targets.
    for (var phase = 0u; phase < island.color_start_count.z; phase++) {
    for (var sweep = 0u; sweep < settings.dimensions.z; sweep++) {
        for (var color_index = 0u; color_index < island.color_start_count.y; color_index++) {
            let color = color_ranges[island.color_start_count.x + color_index];
            for (var local = lane; local < color.y; local += 64u) {
                let contact = island_contacts[color.x + local];
                let data = contacts[contact];
                if (!(data.effective.x > 1.0e-12)) {
                    continue;
                }
                let base = data.offsets.x;
                var impulse = impulses[contact];
                if data.targets.w != 0.0 {
                    let delta = (data.targets.x - row_dot(base, velocity_offset, width)) / data.effective.x;
                    let next = clamp(impulse.x + delta, -data.targets.z, data.targets.z);
                    apply_response(base, velocity_offset, width, next - impulse.x);
                    impulses[contact] = vec4<f32>(next, 0.0, 0.0, 0.0);
                    continue;
                }
                let target_speed = select(data.targets.x, max(data.targets.x, data.targets.y), phase == 1u);
                let delta = (target_speed - row_dot(base, velocity_offset, width)) / data.effective.x;
                let next_normal = max(impulse.x + delta, 0.0);
                apply_response(base, velocity_offset, width, next_normal - impulse.x);
                impulse.x = next_normal;

                var next_x = impulse.y;
                var next_y = impulse.z;
                let tangent_x = row_dot(base + width, velocity_offset, width);
                let tangent_y = row_dot(base + 2u * width, velocity_offset, width);
                let a = data.effective.y;
                let c = data.effective.z;
                let b = data.effective.w;
                let determinant = a * c - b * b;
                if a > 1e-12 && c > 1e-12 && determinant > 1e-6 * a * c {
                    next_x -= (c * tangent_x - b * tangent_y) / determinant;
                    next_y -= (a * tangent_y - b * tangent_x) / determinant;
                } else {
                    if a > 1e-12 {
                        next_x -= tangent_x / a;
                    }
                    if c > 1e-12 {
                        next_y -= tangent_y / c;
                    }
                }
                let radius = data.targets.z * next_normal;
                let tangent_length = length(vec2<f32>(next_x, next_y));
                if tangent_length > radius && tangent_length > 0.0 {
                    let scale = radius / tangent_length;
                    next_x *= scale;
                    next_y *= scale;
                }
                apply_response(base + width, velocity_offset, width, next_x - impulse.y);
                apply_response(base + 2u * width, velocity_offset, width, next_y - impulse.z);
                impulses[contact] = vec4<f32>(impulse.x, next_x, next_y, 0.0);
            }
            storageBarrier();
        }
    }
    }
}
