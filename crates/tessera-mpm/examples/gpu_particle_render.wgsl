struct Particle { position_mass: vec4<f32>, velocity_volume: vec4<f32>, }
@group(0) @binding(0) var<storage, read> particles: array<Particle>;
struct Vertex { @builtin(position) position: vec4<f32>, @location(0) local: vec2<f32>, @location(1) color: vec3<f32>, }
@vertex
fn vertex(@builtin(vertex_index) vertex_id: u32, @builtin(instance_index) particle_id: u32) -> Vertex {
    let corners = array<vec2<f32>, 6>(vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
        vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0));
    let particle = particles[particle_id];
    let p = particle.position_mass.xyz;
    let corner = corners[vertex_id];
    var out: Vertex;
    // Oblique orthographic camera: all three world coordinates affect projection.
    let projected = vec2<f32>(p.x - 0.4 * p.y, p.z + 0.3 * p.y);
    out.position = vec4<f32>(projected + corner * 0.025, 0.5 - p.y * 0.1, 1.0);
    if particle.position_mass.w == 0.0 { out.position = vec4<f32>(2.0, 2.0, 0.0, 1.0); }
    out.local = corner;
    out.color = vec3<f32>(0.2, 0.6, 0.9) + vec3<f32>(0.3, 0.1, 0.0) * min(length(particle.velocity_volume.xyz), 1.0);
    return out;
}
@fragment
fn fragment(in: Vertex) -> @location(0) vec4<f32> {
    if dot(in.local, in.local) > 1.0 { discard; }
    return vec4<f32>(in.color * (0.6 + 0.4 * sqrt(max(0.0, 1.0 - dot(in.local, in.local)))), 1.0);
}
