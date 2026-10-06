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

@group(0) @binding(0) var<storage, read> coordinates: array<f32>;
@group(0) @binding(1) var<storage, read> joints: array<Joint>;
@group(0) @binding(2) var<storage, read> environments: array<Environment>;
@group(0) @binding(3) var<storage, read> roots: array<Pose>;
@group(0) @binding(4) var<storage, read_write> links: array<Pose>;
@group(0) @binding(5) var<storage, read_write> state_status: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read> spherical_orientations: array<vec4<f32>>;

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
    let half = angle * 0.5;
    return vec4<f32>(axis * sin(half), cos(half));
}

fn compose(parent: Pose, local: Pose) -> Pose {
    return Pose(
        vec4<f32>(parent.position.xyz + quat_rotate(parent.orientation, local.position.xyz), 0.0),
        normalize(quat_multiply(parent.orientation, local.orientation)),
    );
}

@compute @workgroup_size(64)
fn forward_kinematics(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let environment_index = invocation.x;
    if (environment_index >= arrayLength(&environments)) { return; }
    if (atomicLoad(&state_status[environment_index]) != 0u) { return; }
    let environment = environments[environment_index];
    let link_offset = environment.indices.x;
    let joint_offset = environment.indices.y;
    let coordinate_offset = environment.indices.z;
    let root_link = environment.indices.w;
    links[link_offset + root_link] = roots[environment_index];
    for (var edge = 0u; edge < environment.counts.x; edge++) {
        let joint = joints[joint_offset + edge];
        let parent = links[link_offset + joint.indices.y];
        let origin = Pose(joint.origin_position, joint.origin_orientation);
        var child = compose(parent, origin);
        let kind = joint.indices.x;
        if (kind == 1u || kind == 2u) {
            let q = coordinates[coordinate_offset + joint.indices.w]
                * joint.axis_scale.w + joint.offset.x;
            if (kind == 1u) {
                child.orientation = normalize(quat_multiply(
                    child.orientation,
                    axis_rotation(joint.axis_scale.xyz, q),
                ));
            } else {
                child.position = vec4<f32>(
                    child.position.xyz + quat_rotate(child.orientation, joint.axis_scale.xyz * q),
                    0.0,
                );
            }
        } else if (kind == 3u) {
            let x = axis_rotation(vec3<f32>(1.0, 0.0, 0.0),
                coordinates[coordinate_offset + joint.indices.w]);
            let y = axis_rotation(vec3<f32>(0.0, 1.0, 0.0),
                coordinates[coordinate_offset + joint.indices.w + 1u]);
            let z = axis_rotation(vec3<f32>(0.0, 0.0, 1.0),
                coordinates[coordinate_offset + joint.indices.w + 2u]);
            child.orientation = normalize(quat_multiply(child.orientation,
                quat_multiply(quat_multiply(x, y), z)));
        } else if (kind == 4u) {
            child.orientation = normalize(quat_multiply(child.orientation,
                spherical_orientations[joint.spherical.x]));
        }
        links[link_offset + joint.indices.z] = child;
    }
}
