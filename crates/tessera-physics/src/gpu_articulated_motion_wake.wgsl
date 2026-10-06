struct MotionLink {
    indices: vec4<u32>, // environment, coordinate offset, dimension, link-term offset
    center_of_mass: vec4<f32>, // local COM, positive-mass flag
    thresholds: vec4<f32>, // origin linear speed, angular speed, timestep, enabled
};
struct Pose { position: vec4<f32>, orientation: vec4<f32> };
@group(0) @binding(0) var<storage, read> links: array<MotionLink>;
@group(0) @binding(1) var<storage, read> terms: array<f32>;
@group(0) @binding(2) var<storage, read> poses: array<Pose>;
@group(0) @binding(3) var<storage, read> velocities: array<f32>;
@group(0) @binding(4) var<storage, read> accelerations: array<f32>;
@group(0) @binding(5) var<storage, read_write> status: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> requests: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read> mass_status: array<u32>;

fn rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}
fn speed(v: vec3<f32>) -> f32 {
    let scale = max(max(abs(v.x), abs(v.y)), abs(v.z));
    if scale == 0.0 { return 0.0; }
    let scaled = v / scale;
    return scale * sqrt(dot(scaled, scaled));
}
@compute @workgroup_size(64)
fn request_motion_wake(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= arrayLength(&links) { return; }
    let link = links[id.x];
    let env = link.indices.x;
    if atomicLoad(&status[env]) != 0u { return; }
    if mass_status[env] != 0u {
        atomicOr(&status[env], 1u);
        return;
    }
    if link.center_of_mass.w == 0.0 || link.thresholds.w == 0.0 { return; }
    let n = link.indices.z;
    let linear_base = link.indices.w + 10u;
    let angular_base = linear_base + 3u * n;
    var linear = vec3<f32>(0.0);
    var angular = vec3<f32>(0.0);
    for (var column = 0u; column < n; column += 1u) {
        let coordinate = link.indices.y + column;
        let v = velocities[coordinate] + accelerations[coordinate] * link.thresholds.z;
        linear += vec3<f32>(terms[linear_base + column],
            terms[linear_base + n + column], terms[linear_base + 2u * n + column]) * v;
        angular += vec3<f32>(terms[angular_base + column],
            terms[angular_base + n + column], terms[angular_base + 2u * n + column]) * v;
    }
    let offset = rotate(poses[id.x].orientation, link.center_of_mass.xyz);
    linear -= cross(angular, offset);
    if !all(abs(linear) < vec3<f32>(1e30)) || !all(abs(angular) < vec3<f32>(1e30)) {
        atomicOr(&status[env], 1u);
        return;
    }
    if speed(linear) > link.thresholds.x || speed(angular) > link.thresholds.y {
        atomicOr(&requests[id.x], 1u);
    }
}
