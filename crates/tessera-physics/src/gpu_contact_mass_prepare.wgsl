struct Settings {
    dimensions: vec4<u32>,
};

struct ContactData {
    effective: vec4<f32>,
    targets: vec4<f32>,
    offsets: vec4<u32>,
};

@group(0) @binding(0) var<storage, read_write> velocity: array<f32>;
@group(0) @binding(1) var<storage, read> jacobians: array<f32>;
@group(0) @binding(2) var<storage, read> responses: array<f32>;
@group(0) @binding(3) var<storage, read> effective_rows: array<f32>;
@group(0) @binding(4) var<storage, read> row_status: array<f32>;
@group(0) @binding(5) var<storage, read_write> contacts: array<ContactData>;
@group(0) @binding(6) var<storage, read_write> impulses: array<vec4<f32>>;
@group(0) @binding(7) var<uniform> settings: Settings;

@compute @workgroup_size(1)
fn prepare() {
    let width = settings.dimensions.x;
    let count = settings.dimensions.y;
    for (var contact = 0u; contact < count; contact++) {
        let row = contact * 3u;
        let base = row * width;
        var data = contacts[contact];
        let normal = effective_rows[row];
        let tangent_x = effective_rows[row + 1u];
        let tangent_y = effective_rows[row + 2u];
        if (row_status[row] != 0.0 || row_status[row + 1u] != 0.0
            || row_status[row + 2u] != 0.0 || !(normal > 1.0e-12)) {
            data.effective = vec4<f32>(0.0);
            contacts[contact] = data;
            impulses[contact] = vec4<f32>(0.0);
            continue;
        }
        var coupling = 0.0;
        for (var coordinate = 0u; coordinate < width; coordinate++) {
            coupling += jacobians[base + width + coordinate]
                * responses[base + 2u * width + coordinate];
        }
        data.effective = vec4<f32>(normal, tangent_x, tangent_y, coupling);
        contacts[contact] = data;

        var seed = impulses[contact];
        if (data.targets.w != 0.0) {
            seed.x = clamp(seed.x, -data.targets.z, data.targets.z);
            seed.y = 0.0;
            seed.z = 0.0;
        } else {
            seed.x = max(seed.x, 0.0);
            let radius = data.targets.z * seed.x;
            let tangent_length = length(seed.yz);
            if (tangent_length > radius && tangent_length > 0.0) {
                let scale = radius / tangent_length;
                seed.y *= scale;
                seed.z *= scale;
            }
            if (tangent_x <= 1.0e-12) {
                seed.y = 0.0;
            }
            if (tangent_y <= 1.0e-12) {
                seed.z = 0.0;
            }
        }
        impulses[contact] = seed;
        for (var coordinate = 0u; coordinate < width; coordinate++) {
            velocity[coordinate] += responses[base + coordinate] * seed.x
                + responses[base + width + coordinate] * seed.y
                + responses[base + 2u * width + coordinate] * seed.z;
        }
    }
}
