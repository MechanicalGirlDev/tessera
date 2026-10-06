struct Meta {
    matrix_offset: u32,
    link_offset: u32,
    link_count: u32,
    vector_offset: u32,
    dimension: u32,
    inverse_offset: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> systems: array<Meta>;
@group(0) @binding(1) var<storage, read> current_links: array<f32>;
@group(0) @binding(2) var<storage, read> plus_links: array<f32>;
@group(0) @binding(3) var<storage, read> minus_links: array<f32>;
@group(0) @binding(4) var<storage, read> velocities: array<f32>;
@group(0) @binding(5) var<storage, read_write> base_forces: array<f32>;
@group(0) @binding(6) var<storage, read_write> status: array<atomic<u32>>;

fn inertia_times(base: u32, angular: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(vec3<f32>(current_links[base + 1u], current_links[base + 2u],
            current_links[base + 3u]), angular),
        dot(vec3<f32>(current_links[base + 4u], current_links[base + 5u],
            current_links[base + 6u]), angular),
        dot(vec3<f32>(current_links[base + 7u], current_links[base + 8u],
            current_links[base + 9u]), angular),
    );
}

@compute @workgroup_size(64)
fn subtract_velocity_bias(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let system = workgroup.x;
    if (system >= arrayLength(&systems) || atomicLoad(&status[system]) != 0u) { return; }
    let descriptor = systems[system];
    let n = descriptor.dimension;
    let stride = 10u + 6u * n;
    let coordinate_offset = descriptor.vector_offset / 2u;
    for (var coordinate = lane; coordinate < n; coordinate += 64u) {
        var bias = 0.0;
        for (var link = 0u; link < descriptor.link_count; link++) {
            let base = descriptor.link_offset + link * stride;
            let mass = current_links[base];
            if (mass == 0.0) { continue; }
            let linear = base + 10u;
            let angular = linear + 3u * n;
            var drift_linear = vec3<f32>(0.0);
            var drift_angular = vec3<f32>(0.0);
            var omega = vec3<f32>(0.0);
            for (var column = 0u; column < n; column++) {
                let velocity = velocities[coordinate_offset + column];
                let plus_linear = vec3<f32>(plus_links[linear + column],
                    plus_links[linear + n + column], plus_links[linear + 2u * n + column]);
                let minus_linear = vec3<f32>(minus_links[linear + column],
                    minus_links[linear + n + column], minus_links[linear + 2u * n + column]);
                let plus_angular = vec3<f32>(plus_links[angular + column],
                    plus_links[angular + n + column], plus_links[angular + 2u * n + column]);
                let minus_angular = vec3<f32>(minus_links[angular + column],
                    minus_links[angular + n + column], minus_links[angular + 2u * n + column]);
                let current_angular = vec3<f32>(current_links[angular + column],
                    current_links[angular + n + column],
                    current_links[angular + 2u * n + column]);
                drift_linear += (plus_linear - minus_linear) * velocity;
                drift_angular += (plus_angular - minus_angular) * velocity;
                omega += current_angular * velocity;
            }
            drift_linear *= 500.0;
            drift_angular *= 500.0;
            let angular_force = inertia_times(base, drift_angular)
                + cross(omega, inertia_times(base, omega));
            let column_linear = vec3<f32>(current_links[linear + coordinate],
                current_links[linear + n + coordinate],
                current_links[linear + 2u * n + coordinate]);
            let column_angular = vec3<f32>(current_links[angular + coordinate],
                current_links[angular + n + coordinate],
                current_links[angular + 2u * n + coordinate]);
            bias += mass * dot(column_linear, drift_linear)
                + dot(column_angular, angular_force);
        }
        base_forces[coordinate_offset + coordinate] -= bias;
    }
}
