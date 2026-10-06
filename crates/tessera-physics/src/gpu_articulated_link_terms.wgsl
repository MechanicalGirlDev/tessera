struct Pose {
    position: vec4<f32>,
    orientation: vec4<f32>,
};

struct Joint {
    indices: vec4<u32>,
    origin_position: vec4<f32>,
    origin_orientation: vec4<f32>,
    axis_scale: vec4<f32>,
    offset: vec4<f32>,
    spherical: vec4<u32>,
};

struct Environment {
    indices: vec4<u32>,
    counts: vec4<u32>,
};

struct LinkMeta {
    indices: vec4<u32>,
    center_of_mass: vec4<f32>,
    inertia_rows: mat3x4<f32>,
    mass: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> metadata: array<LinkMeta>;
@group(0) @binding(1) var<storage, read> poses: array<Pose>;
@group(0) @binding(2) var<storage, read> joints: array<Joint>;
@group(0) @binding(3) var<storage, read> environments: array<Environment>;
@group(0) @binding(4) var<storage, read> coordinates: array<f32>;
@group(0) @binding(5) var<storage, read_write> status: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> output: array<f32>;

fn quat_multiply(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz),
        a.w * b.w - dot(a.xyz, b.xyz),
    );
}

fn quat_rotate(q: vec4<f32>, value: vec3<f32>) -> vec3<f32> {
    let doubled = 2.0 * cross(q.xyz, value);
    return value + q.w * doubled + cross(q.xyz, doubled);
}

fn axis_rotation(axis: vec3<f32>, angle: f32) -> vec4<f32> {
    let half = 0.5 * angle;
    return vec4<f32>(axis * sin(half), cos(half));
}

fn write_column(base: u32, n: u32, slot: u32, linear: vec3<f32>, angular: vec3<f32>) {
    for (var row = 0u; row < 3u; row++) {
        output[base + 10u + row * n + slot] += linear[row];
        output[base + 10u + 3u * n + row * n + slot] += angular[row];
    }
}

@compute @workgroup_size(64)
fn build_link_terms(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let link_index = invocation.x;
    if (link_index >= arrayLength(&metadata)) { return; }
    let link = metadata[link_index];
    let base = link.indices.w;
    if (base == 0xffffffffu || atomicLoad(&status[link.indices.x]) != 0u) { return; }
    let environment = environments[link.indices.x];
    let n = environment.counts.y;
    let pose = poses[link_index];
    let point = pose.position.xyz + quat_rotate(pose.orientation, link.center_of_mass.xyz);
    output[base] = link.mass.x;
    var basis: array<vec3<f32>, 3>;
    basis[0] = quat_rotate(pose.orientation, vec3<f32>(1.0, 0.0, 0.0));
    basis[1] = quat_rotate(pose.orientation, vec3<f32>(0.0, 1.0, 0.0));
    basis[2] = quat_rotate(pose.orientation, vec3<f32>(0.0, 0.0, 1.0));
    for (var row = 0u; row < 3u; row++) {
        for (var col = 0u; col < 3u; col++) {
            var inertia = 0.0;
            for (var i = 0u; i < 3u; i++) {
                for (var j = 0u; j < 3u; j++) {
                    inertia += basis[i][row] * link.inertia_rows[i][j] * basis[j][col];
                }
            }
            output[base + 1u + row * 3u + col] = inertia;
        }
    }
    for (var element = 0u; element < 6u * n; element++) {
        output[base + 10u + element] = 0.0;
    }
    if (environment.counts.z == 6u) {
        let root = poses[environment.indices.x + environment.indices.w];
        let lever = point - root.position.xyz;
        for (var axis_index = 0u; axis_index < 3u; axis_index++) {
            var axis = vec3<f32>(0.0);
            axis[axis_index] = 1.0;
            write_column(base, n, axis_index, axis, vec3<f32>(0.0));
            write_column(base, n, axis_index + 3u, cross(axis, lever), axis);
        }
    }
    var current = link_index;
    for (var depth = 0u; depth < environment.counts.x; depth++) {
        let incoming = metadata[current].indices.z;
        if (incoming == 0xffffffffu) { break; }
        let joint = joints[incoming];
        let parent_index = environment.indices.x + joint.indices.y;
        let parent = poses[parent_index];
        let origin_orientation = quat_multiply(parent.orientation, joint.origin_orientation);
        let origin_position = parent.position.xyz
            + quat_rotate(parent.orientation, joint.origin_position.xyz);
        let slot = joint.indices.w;
        if (joint.indices.x == 1u || joint.indices.x == 2u) {
            let axis = quat_rotate(origin_orientation, joint.axis_scale.xyz) * joint.axis_scale.w;
            if (joint.indices.x == 1u) {
                write_column(base, n, slot, cross(axis, point - origin_position), axis);
            } else {
                write_column(base, n, slot, axis, vec3<f32>(0.0));
            }
        } else if (joint.indices.x == 3u) {
            let qx = coordinates[environment.indices.z + slot];
            let qy = coordinates[environment.indices.z + slot + 1u];
            let x = axis_rotation(vec3<f32>(1.0, 0.0, 0.0), qx);
            let y = axis_rotation(vec3<f32>(0.0, 1.0, 0.0), qy);
            let axes = array<vec3<f32>, 3>(
                quat_rotate(origin_orientation, vec3<f32>(1.0, 0.0, 0.0)),
                quat_rotate(quat_multiply(origin_orientation, x), vec3<f32>(0.0, 1.0, 0.0)),
                quat_rotate(quat_multiply(origin_orientation, quat_multiply(x, y)),
                    vec3<f32>(0.0, 0.0, 1.0)),
            );
            for (var axis_index = 0u; axis_index < 3u; axis_index++) {
                let axis = axes[axis_index];
                write_column(base, n, slot + axis_index, cross(axis, point - origin_position), axis);
            }
        } else if (joint.indices.x == 4u) {
            let axes = array<vec3<f32>, 3>(
                quat_rotate(origin_orientation, vec3<f32>(1.0, 0.0, 0.0)),
                quat_rotate(origin_orientation, vec3<f32>(0.0, 1.0, 0.0)),
                quat_rotate(origin_orientation, vec3<f32>(0.0, 0.0, 1.0)),
            );
            for (var axis_index = 0u; axis_index < 3u; axis_index++) {
                let axis = axes[axis_index];
                write_column(base, n, slot + axis_index, cross(axis, point - origin_position), axis);
            }
        }
        current = parent_index;
    }
}
