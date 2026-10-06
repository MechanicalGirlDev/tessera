struct JointForce {
    passive: vec4<f32>,
    nonlinear: vec4<f32>,
    motor_targets: vec4<f32>,
    motor_gains: vec4<f32>,
    implicit: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> parameters: array<JointForce>;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read> velocities: array<f32>;
@group(0) @binding(3) var<storage, read> owners: array<u32>;
@group(0) @binding(4) var<storage, read_write> status: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> base_forces: array<f32>;
@group(0) @binding(6) var<storage, read_write> vectors: array<f32>;

@compute @workgroup_size(64)
fn assemble_joint_forces(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let coordinate = invocation.x;
    if (coordinate >= arrayLength(&parameters)) { return; }
    if (atomicLoad(&status[owners[coordinate]]) != 0u) { return; }
    let input = parameters[coordinate];
    let position = positions[coordinate];
    let velocity = velocities[coordinate];
    let displacement = position - input.passive.z;
    let spring = input.passive.x * (-displacement)
        - input.nonlinear.x * displacement * displacement
        - input.nonlinear.y * displacement * displacement * displacement;
    let damping = -input.passive.y * velocity
        - input.nonlinear.z * velocity * abs(velocity)
        - input.nonlinear.w * velocity * velocity * velocity;
    var motor = 0.0;
    if (input.motor_targets.w != 0.0) {
        let position_error = select(0.0, input.motor_targets.x - position,
            input.motor_gains.z != 0.0);
        motor = input.motor_gains.x * position_error
            + input.motor_gains.y * (input.motor_targets.y - velocity);
        motor = clamp(motor, -input.motor_targets.z, input.motor_targets.z);
    }
    var force = input.passive.w + spring + damping + motor;
    if (input.implicit.w != 0.0) {
        let spring_tangent = input.passive.x
            + 2.0 * input.nonlinear.x * displacement
            + 3.0 * input.nonlinear.y * displacement * displacement;
        let damping_tangent = input.passive.y
            + 2.0 * input.nonlinear.z * abs(velocity)
            + 3.0 * input.nonlinear.w * velocity * velocity;
        let dt = input.implicit.z;
        let vector_offset = bitcast<u32>(input.implicit.y);
        vectors[vector_offset] = input.implicit.x
            + dt * damping_tangent + dt * dt * spring_tangent;
        force -= dt * spring_tangent * velocity;
    }
    base_forces[coordinate] = force;
}
