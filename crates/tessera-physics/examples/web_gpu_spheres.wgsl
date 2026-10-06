struct Body {
    position_inverse_mass: vec4<f32>,
    orientation: vec4<f32>,
    linear_velocity: vec4<f32>,
    angular_velocity: vec4<f32>,
    inverse_inertia_sleep: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> bodies: array<Body>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) circle_position: vec2<f32>,
    @location(1) color: vec3<f32>,
}

@vertex
fn vertex_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) body_index: u32,
) -> VertexOutput {
    let corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
    );
    let point = corners[vertex_index];
    if body_index == 3u {
        var floor: VertexOutput;
        floor.clip_position = vec4<f32>(point.x * 0.92, -0.55 + point.y * 0.025, 0.0, 1.0);
        floor.circle_position = vec2<f32>(0.0, 0.0);
        floor.color = vec3<f32>(0.16, 0.23, 0.34);
        return floor;
    }
    let position = bodies[body_index].position_inverse_mass.xyz;
    let projected = vec2<f32>(
        position.x * 0.22 + position.y * 0.08,
        -0.55 + position.z * 0.21 - position.y * 0.05,
    );
    let palette = array<vec3<f32>, 3>(
        vec3<f32>(0.25, 0.75, 0.95),
        vec3<f32>(0.98, 0.64, 0.33),
        vec3<f32>(0.72, 0.48, 0.98),
    );
    var output: VertexOutput;
    output.clip_position = vec4<f32>(projected + point * vec2<f32>(0.13, 0.20), 0.0, 1.0);
    output.circle_position = point;
    output.color = palette[body_index % 3u];
    return output;
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let radius_squared = dot(input.circle_position, input.circle_position);
    if radius_squared > 1.0 {
        discard;
    }
    let lighting = 0.48 + 0.52 * sqrt(1.0 - radius_squared);
    return vec4<f32>(input.color * lighting, 1.0);
}
