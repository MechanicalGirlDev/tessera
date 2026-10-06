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
    @location(0) local: vec2<f32>,
    @location(1) color: vec3<f32>,
    @location(2) capsule: vec2<f32>,
    @location(3) @interpolate(flat) kind: u32,
}

fn rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

fn project_direction(v: vec3<f32>) -> vec2<f32> {
    return vec2<f32>(v.x * 0.22 + v.y * 0.08, v.z * 0.20 - v.y * 0.05);
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
    var output: VertexOutput;
    output.local = point;
    output.capsule = vec2<f32>(0.0, 0.0);
    output.kind = 0u;
    if body_index == 4u {
        output.clip_position = vec4<f32>(point.x * 0.92, -0.68 + point.y * 0.025, 0.0, 1.0);
        output.color = vec3<f32>(0.16, 0.23, 0.34);
        output.kind = 3u;
        return output;
    }
    if body_index == 5u {
        let anchor = project_direction(bodies[0].position_inverse_mass.xyz)
            + vec2<f32>(0.0, -0.68);
        let center = project_direction(bodies[1].position_inverse_mass.xyz)
            + vec2<f32>(0.0, -0.68);
        let axis = center - anchor;
        let direction = normalize(axis);
        let tangent = vec2<f32>(direction.y, -direction.x);
        output.clip_position = vec4<f32>((anchor + center) * 0.5
            + axis * point.y * 0.5 + tangent * point.x * 0.004, 0.0, 1.0);
        output.color = vec3<f32>(0.32, 0.48, 0.61);
        output.kind = 3u;
        return output;
    }

    let body = bodies[body_index];
    let position = body.position_inverse_mass.xyz;
    let center = project_direction(position) + vec2<f32>(0.0, -0.68);
    let palette = array<vec3<f32>, 4>(
        vec3<f32>(0.82, 0.88, 0.94),
        vec3<f32>(0.27, 0.79, 0.95),
        vec3<f32>(0.98, 0.64, 0.33),
        vec3<f32>(0.72, 0.48, 0.98),
    );
    output.color = palette[body_index];

    if body_index == 1u {
        let axis = project_direction(rotate(body.orientation, vec3<f32>(0.0, 0.0, 1.0)));
        let axis_length = max(length(axis) * 0.7, 0.0001);
        let direction = normalize(axis);
        let tangent = vec2<f32>(direction.y, -direction.x);
        let radius = 0.065;
        let local = vec2<f32>(point.x * radius, point.y * (axis_length + radius));
        output.clip_position = vec4<f32>(center + tangent * local.x + direction * local.y, 0.0, 1.0);
        output.local = local;
        output.capsule = vec2<f32>(radius, axis_length);
        output.kind = 2u;
    } else if body_index == 2u {
        let axis_x = project_direction(rotate(body.orientation, vec3<f32>(1.0, 0.0, 0.0)));
        let axis_z = project_direction(rotate(body.orientation, vec3<f32>(0.0, 0.0, 1.0)));
        output.clip_position = vec4<f32>(center + axis_x * point.x * 0.55
            + axis_z * point.y * 0.45, 0.0, 1.0);
        output.kind = 1u;
    } else {
        let radius = select(0.04, 0.085, body_index == 3u);
        output.clip_position = vec4<f32>(center + point * radius, 0.0, 1.0);
    }
    return output;
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    if input.kind == 3u {
        return vec4<f32>(input.color, 1.0);
    }
    if input.kind == 2u {
        let nearest = clamp(input.local.y, -input.capsule.y, input.capsule.y);
        if length(input.local - vec2<f32>(0.0, nearest)) > input.capsule.x {
            discard;
        }
        return vec4<f32>(input.color * (0.75 + 0.25 * input.local.x / input.capsule.x), 1.0);
    }
    if input.kind == 1u {
        return vec4<f32>(input.color * (0.75 + 0.2 * input.local.x), 1.0);
    }
    let radius_squared = dot(input.local, input.local);
    if radius_squared > 1.0 {
        discard;
    }
    let lighting = 0.48 + 0.52 * sqrt(1.0 - radius_squared);
    return vec4<f32>(input.color * lighting, 1.0);
}
