struct RigidState {
    position_inverse_mass: vec4<f32>,
    orientation: vec4<f32>,
    linear_velocity: vec4<f32>,
    angular_velocity: vec4<f32>,
    inverse_inertia_sleep: vec4<f32>,
}
struct Motion {
    header: vec4<u32>,
    linear: vec4<f32>,
    angular: vec4<f32>,
}
@group(0) @binding(0) var<storage, read_write> states: array<RigidState>;
@group(0) @binding(1) var<storage, read> motion: Motion;

struct KinematicTranslation {
    origin_elapsed: vec4<f32>,
    expected_position: vec4<f32>,
    linear_velocity: vec4<f32>,
}
@group(0) @binding(2) var<storage, read_write> kinematic_translation: array<KinematicTranslation>;

@compute @workgroup_size(1)
fn main() {
    let index = motion.header.x;
    // Test the current resident mass, including prior topology transfers.
    if (states[index].position_inverse_mass.w != 0.0) { return; }
    states[index].linear_velocity = motion.linear;
    states[index].angular_velocity = motion.angular;
    kinematic_translation[index].origin_elapsed.w = 0.0;
}
