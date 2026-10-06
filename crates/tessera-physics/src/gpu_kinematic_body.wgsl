struct Body {
    origin: vec4<f32>,
    orientation: vec4<f32>,
    linear: vec4<f32>,
    angular: vec4<f32>,
}
struct Settings { timestep: f32, count: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<storage, read_write> bodies: array<Body>;
@group(0) @binding(1) var<storage, read_write> status: array<u32>;
@group(0) @binding(2) var<uniform> settings: Settings;

struct KinematicTranslation {
    origin_elapsed: vec4<f32>,
    expected_position: vec4<f32>,
    linear_velocity: vec4<f32>,
}
@group(0) @binding(3) var<storage, read_write> translation: array<KinematicTranslation>;

fn finite3(value: vec3<f32>) -> bool {
    return all(abs(value) <= vec3<f32>(3.402823e38));
}

@compute @workgroup_size(64)
fn integrate(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if index >= settings.count || status[index] != 0u { return; }
    let previous = bodies[index];
    var interval = translation[index];
    if interval.origin_elapsed.w == 0.0
        || any(interval.expected_position.xyz != previous.origin.xyz)
        || any(interval.linear_velocity.xyz != previous.linear.xyz) {
        interval.origin_elapsed = vec4<f32>(previous.origin.xyz, 0.0);
        interval.linear_velocity = previous.linear;
    }
    interval.origin_elapsed.w += settings.timestep;
    let origin = interval.origin_elapsed.xyz + interval.linear_velocity.xyz * interval.origin_elapsed.w;
    let rotation = previous.angular.xyz * settings.timestep;
    let angle = length(rotation);
    if !finite3(origin) || !finite3(rotation) || !(angle <= 3.402823e38) {
        status[index] = 1u;
        return;
    }
    var delta = vec4<f32>(rotation * 0.5, 1.0);
    if angle >= 1e-6 {
        delta = vec4<f32>(rotation * (sin(angle * 0.5) / angle), cos(angle * 0.5));
    }
    let q = previous.orientation;
    let orientation = normalize(vec4<f32>(
        delta.w * q.xyz + q.w * delta.xyz + cross(delta.xyz, q.xyz),
        delta.w * q.w - dot(delta.xyz, q.xyz)
    ));
    if !all(abs(orientation) <= vec4<f32>(1.0)) {
        status[index] = 1u;
        return;
    }
    bodies[index].origin = vec4<f32>(origin, 0.0);
    bodies[index].orientation = orientation;
    interval.expected_position = vec4<f32>(origin, 0.0);
    translation[index] = interval;
}
