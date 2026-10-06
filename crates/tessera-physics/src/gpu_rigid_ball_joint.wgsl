struct RigidState {
    position_inverse_mass: vec4<f32>,
    orientation: vec4<f32>,
    linear_velocity: vec4<f32>,
    angular_velocity: vec4<f32>,
    inverse_inertia_sleep: vec4<f32>,
}

struct BallJoint {
    bodies: vec4<u32>,
    anchor_a: vec4<f32>,
    anchor_b: vec4<f32>,
    frame_a: vec4<f32>,
    frame_b: vec4<f32>,
    drive: vec4<f32>,
    servo: vec4<f32>,
}

struct IslandRange {
    offset: u32,
    count: u32,
    padding_a: u32,
    padding_b: u32,
}

struct Params {
    dt: f32,
    bias: f32,
    iterations: u32,
    islands: u32,
    bodies: u32,
    padding_a: u32,
    padding_b: u32,
    padding_c: u32,
    temporal: vec4<f32>,
    correction_limits: vec4<f32>,
}

struct VelocitySnapshot {
    linear: vec4<f32>,
    angular: vec4<f32>,
}

struct AngleState {
    angle: f32,
    wrapped: f32,
    initialized: f32,
    drive_target: f32,
}

@group(0) @binding(0) var<storage, read_write> states: array<RigidState>;
@group(0) @binding(1) var<storage, read> joints: array<BallJoint>;
@group(0) @binding(2) var<storage, read> island_ranges: array<IslandRange>;
@group(0) @binding(3) var<storage, read> island_indices: array<u32>;
@group(0) @binding(4) var<storage, read_write> impulses: array<vec4<f32>>;
@group(0) @binding(5) var<uniform> params: Params;
@group(0) @binding(6) var<storage, read_write> angular_impulses: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> velocities_before_solve: array<VelocitySnapshot>;
@group(0) @binding(8) var<storage, read_write> angles: array<AngleState>;

fn quat_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz),
                     a.w * b.w - dot(a.xyz, b.xyz));
}

fn quat_conjugate(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.xyz, q.w);
}

fn rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

fn inverse_inertia_mul(state: RigidState, value: vec3<f32>) -> vec3<f32> {
    let q = state.orientation;
    let body = rotate(vec4<f32>(-q.xyz, q.w), value);
    return rotate(q, body * state.inverse_inertia_sleep.xyz);
}

fn effective_mass(state: RigidState, arm: vec3<f32>, direction: vec3<f32>) -> f32 {
    let angular = cross(arm, direction);
    return state.position_inverse_mass.w + dot(angular, inverse_inertia_mul(state, angular));
}

fn apply_impulse(state: RigidState, arm: vec3<f32>, impulse: vec3<f32>) -> RigidState {
    if (state.position_inverse_mass.w == 0.0) { return state; }
    var result = state;
    result.linear_velocity = vec4<f32>(
        state.linear_velocity.xyz + impulse * state.position_inverse_mass.w, 0.0);
    result.angular_velocity = vec4<f32>(
        state.angular_velocity.xyz + inverse_inertia_mul(state, cross(arm, impulse)), 0.0);
    if (dot(impulse, impulse) > 1e-20) {
        result.inverse_inertia_sleep.w = 0.0;
    }
    return result;
}

fn apply_angular_impulse(state: RigidState, impulse: vec3<f32>) -> RigidState {
    if (state.position_inverse_mass.w == 0.0) { return state; }
    var result = state;
    result.angular_velocity = vec4<f32>(
        state.angular_velocity.xyz + inverse_inertia_mul(state, impulse), 0.0);
    if (dot(impulse, impulse) > 1e-20) {
        result.inverse_inertia_sleep.w = 0.0;
    }
    return result;
}

fn contact_velocity(state: RigidState, arm: vec3<f32>) -> vec3<f32> {
    return state.linear_velocity.xyz + cross(state.angular_velocity.xyz, arm);
}

fn anchor_arm(state: RigidState, local_anchor: vec3<f32>) -> vec3<f32> {
    return rotate(state.orientation, local_anchor);
}

fn axis(index: u32) -> vec3<f32> {
    if (index == 0u) { return vec3<f32>(1.0, 0.0, 0.0); }
    if (index == 1u) { return vec3<f32>(0.0, 1.0, 0.0); }
    return vec3<f32>(0.0, 0.0, 1.0);
}

fn reference_tangent(local_axis: vec3<f32>) -> vec3<f32> {
    let pivot = select(vec3<f32>(1.0, 0.0, 0.0),
                       vec3<f32>(0.0, 1.0, 0.0), abs(local_axis.x) > 0.7);
    return normalize(cross(pivot, local_axis));
}

fn revolute_angle(joint: BallJoint, a: RigidState, b: RigidState) -> f32 {
    let axis_a = normalize(rotate(a.orientation, joint.frame_a.xyz));
    let tangent_a = rotate(a.orientation, reference_tangent(joint.frame_a.xyz));
    let tangent_b = rotate(b.orientation, reference_tangent(joint.frame_b.xyz));
    return atan2(dot(axis_a, cross(tangent_a, tangent_b)), dot(tangent_a, tangent_b));
}

fn quat_from_rotation(rotation: vec3<f32>) -> vec4<f32> {
    let angle = length(rotation);
    if (angle < 1e-5) {
        return normalize(vec4<f32>(0.5 * rotation, 1.0));
    }
    let half_angle = 0.5 * angle;
    return vec4<f32>(rotation * (sin(half_angle) / angle), cos(half_angle));
}

fn update_angle(index: u32, predict_turns: bool) {
    if (index >= arrayLength(&joints)) { return; }
    let joint = joints[index];
    if (joint.bodies.z != 2u) { return; }
    let wrapped = revolute_angle(joint, states[joint.bodies.x], states[joint.bodies.y]);
    let previous = angles[index];
    var continuous = wrapped;
    if (previous.initialized > 0.5) {
        var predicted = 0.0;
        if (predict_turns) {
            let a = states[joint.bodies.x];
            let b = states[joint.bodies.y];
            let axis = normalize(rotate(a.orientation, joint.frame_a.xyz));
            predicted = dot(axis, b.angular_velocity.xyz - a.angular_velocity.xyz) * params.dt;
        }
        let raw_delta = wrapped - previous.wrapped;
        let turns = round((predicted - raw_delta) / 6.283185307179586);
        let delta = raw_delta + turns * 6.283185307179586;
        continuous = previous.angle + delta;
    }
    angles[index] = AngleState(continuous, wrapped, 1.0, previous.drive_target);
}

@compute @workgroup_size(64)
fn capture_temporal_drives(@builtin(global_invocation_id) id: vec3<u32>) {
    let index=id.x;
    if (index>=arrayLength(&joints)) { return; }
    let joint=joints[index];
    if ((joint.bodies.w & 4u)==0u) { angles[index].drive_target=0.0; return; }
    var coordinate=angles[index].angle;
    if (joint.bodies.z==3u) {
        let a=states[joint.bodies.x]; let b=states[joint.bodies.y];
        let axis=normalize(rotate(quat_mul(a.orientation,joint.frame_a),vec3<f32>(0.0,0.0,1.0)));
        coordinate=dot((b.position_inverse_mass.xyz-a.position_inverse_mass.xyz)
            +anchor_arm(b,joint.anchor_b.xyz)-anchor_arm(a,joint.anchor_a.xyz),axis);
    }
    angles[index].drive_target=joint.servo.y*(joint.servo.x-coordinate)+joint.servo.z*joint.drive.x;
}

@compute @workgroup_size(64)
fn update_angles(@builtin(global_invocation_id) id: vec3<u32>) {
    update_angle(id.x, true);
}

@compute @workgroup_size(64)
fn sample_angles(@builtin(global_invocation_id) id: vec3<u32>) {
    update_angle(id.x, false);
}

@compute @workgroup_size(64)
fn capture_velocity(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.bodies) { return; }
    let state = states[id.x];
    velocities_before_solve[id.x] = VelocitySnapshot(state.linear_velocity, state.angular_velocity);
}

@compute @workgroup_size(64)
fn correct_pose(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.bodies) { return; }
    var state = states[id.x];
    if (state.position_inverse_mass.w == 0.0) { return; }
    let previous = velocities_before_solve[id.x];
    state.position_inverse_mass = vec4<f32>(
        state.position_inverse_mass.xyz +
        (state.linear_velocity.xyz - previous.linear.xyz) * params.dt,
        state.position_inverse_mass.w);
    let angular_delta = state.angular_velocity.xyz - previous.angular.xyz;
    state.orientation = normalize(quat_mul(
        quat_from_rotation(angular_delta * params.dt), state.orientation));
    states[id.x] = state;
}

fn warm_joint(index: u32) {
    let joint = joints[index];
    var a = states[joint.bodies.x];
    var b = states[joint.bodies.y];
    var impulse = impulses[index].xyz;
    if (joint.bodies.z == 3u) {
        let slide_axis = normalize(rotate(quat_mul(a.orientation, joint.frame_a), vec3<f32>(0.0, 0.0, 1.0)));
        impulse -= slide_axis * dot(impulse, slide_axis);
    }
    if (dot(impulse, impulse) > 1e-20) {
        a = apply_impulse(a, anchor_arm(a, joint.anchor_a.xyz), -impulse);
        b = apply_impulse(b, anchor_arm(b, joint.anchor_b.xyz), impulse);
    }
    if (joint.bodies.z != 0u) {
        var angular = angular_impulses[index].xyz;
        if (joint.bodies.z == 2u) {
            let hinge_axis = rotate(a.orientation, joint.frame_a.xyz);
            angular -= hinge_axis * dot(angular, hinge_axis);
        }
        if (dot(angular, angular) > 1e-20) {
            a = apply_angular_impulse(a, -angular);
            b = apply_angular_impulse(b, angular);
        }
    }
    if ((joint.bodies.w & 5u) != 0u) {
        let motor_impulse = angular_impulses[index].w;
        if (joint.bodies.z == 2u) {
            let hinge_axis = normalize(rotate(a.orientation, joint.frame_a.xyz));
            a = apply_angular_impulse(a, -hinge_axis * motor_impulse);
            b = apply_angular_impulse(b, hinge_axis * motor_impulse);
        } else if (joint.bodies.z == 3u) {
            let slide_axis = normalize(rotate(quat_mul(a.orientation, joint.frame_a), vec3<f32>(0.0, 0.0, 1.0)));
            a = apply_impulse(a, anchor_arm(a, joint.anchor_a.xyz), -slide_axis * motor_impulse);
            b = apply_impulse(b, anchor_arm(b, joint.anchor_b.xyz), slide_axis * motor_impulse);
        }
    }
    if ((joint.bodies.w & 2u) != 0u) {
        let limit_impulse = impulses[index].w;
        if (joint.bodies.z == 2u) {
            let hinge_axis = normalize(rotate(a.orientation, joint.frame_a.xyz));
            a = apply_angular_impulse(a, -hinge_axis * limit_impulse);
            b = apply_angular_impulse(b, hinge_axis * limit_impulse);
        } else if (joint.bodies.z == 3u) {
            let slide_axis = normalize(rotate(quat_mul(a.orientation, joint.frame_a), vec3<f32>(0.0, 0.0, 1.0)));
            a = apply_impulse(a, anchor_arm(a, joint.anchor_a.xyz), -slide_axis * limit_impulse);
            b = apply_impulse(b, anchor_arm(b, joint.anchor_b.xyz), slide_axis * limit_impulse);
        }
    }
    states[joint.bodies.x] = a;
    states[joint.bodies.y] = b;
}

fn bilateral_delta(relative: f32, error: f32, mass: f32, accumulated: f32, angular: bool) -> f32 {
    if (params.temporal.w == 0.0) { return -(relative+params.bias*error/params.dt)/mass; }
    if (params.temporal.w<0.0) { return -relative/mass; }
    var correction=0.0;
    if (params.temporal.w>0.0) {
        let maximum=select(params.correction_limits.x,params.correction_limits.y,angular);
        correction=clamp(params.temporal.z*error,-maximum,maximum);
    }
    return -params.temporal.x*(relative+correction)/mass-params.temporal.y*accumulated;
}

fn limit_target(violation: f32, angular: bool) -> f32 {
    if (params.temporal.w==0.0) { return params.bias*violation/params.dt; }
    if (violation<=0.0) { return violation/params.dt; }
    if (params.temporal.w<0.0) { return 0.0; }
    return min(params.temporal.z*violation,select(params.correction_limits.x,params.correction_limits.y,angular));
}
fn limit_delta(relative: f32, desired: f32, mass: f32, accumulated: f32, speculative: bool) -> f32 {
    if (params.temporal.w<=0.0 || speculative) { return (desired-relative)/mass; }
    return params.temporal.x*(desired-relative)/mass-params.temporal.y*accumulated;
}

fn solve_joint(index: u32) {
    let joint = joints[index];
    var a = states[joint.bodies.x];
    var b = states[joint.bodies.y];
    var accumulated = impulses[index].xyz;
    let arm_a = anchor_arm(a, joint.anchor_a.xyz);
    let arm_b = anchor_arm(b, joint.anchor_b.xyz);
    let error = b.position_inverse_mass.xyz + arm_b - a.position_inverse_mass.xyz - arm_a;
    var slide_axis = vec3<f32>(0.0, 0.0, 1.0);
    var slide_tangent = vec3<f32>(1.0, 0.0, 0.0);
    var slide_bitangent = vec3<f32>(0.0, 1.0, 0.0);
    var linear_components = 3u;
    if (joint.bodies.z == 3u) {
        slide_axis = normalize(rotate(quat_mul(a.orientation, joint.frame_a), vec3<f32>(0.0, 0.0, 1.0)));
        let pivot = select(vec3<f32>(1.0, 0.0, 0.0),
                           vec3<f32>(0.0, 1.0, 0.0), abs(slide_axis.x) > 0.7);
        slide_tangent = normalize(cross(pivot, slide_axis));
        slide_bitangent = cross(slide_axis, slide_tangent);
        accumulated -= slide_axis * dot(accumulated, slide_axis);
        linear_components = 2u;
    }
    if ((joint.bodies.w & 5u) != 0u) {
        var drive_axis = slide_axis;
        if (joint.bodies.z == 2u) {
            drive_axis = normalize(rotate(a.orientation, joint.frame_a.xyz));
        }
        var motor_accumulated = angular_impulses[index].w;
        var mass = 0.0;
        var relative = 0.0;
        if (joint.bodies.z == 2u) {
            mass = dot(drive_axis, inverse_inertia_mul(a, drive_axis)) +
                   dot(drive_axis, inverse_inertia_mul(b, drive_axis));
            relative = dot(b.angular_velocity.xyz - a.angular_velocity.xyz, drive_axis);
        } else {
            mass = effective_mass(a, arm_a, drive_axis) + effective_mass(b, arm_b, drive_axis);
            relative = dot(contact_velocity(b, arm_b) - contact_velocity(a, arm_a), drive_axis);
        }
        if (mass > 1e-12) {
            let maximum = joint.drive.y * params.dt;
            var next = clamp(motor_accumulated + (joint.drive.x - relative) / mass, -maximum, maximum);
            if ((joint.bodies.w & 4u) != 0u) {
                var coordinate = dot(error, slide_axis);
                if (joint.bodies.z == 2u) { coordinate = angles[index].angle; }
                if (params.temporal.w!=0.0) {
                    let residual=params.dt*(angles[index].drive_target-joint.servo.z*relative)-motor_accumulated;
                    let increment=residual/(1.0+params.dt*joint.servo.z*mass);
                    next=clamp(motor_accumulated+increment,-maximum,maximum);
                } else {
                    let commanded_force = joint.servo.y * (joint.servo.x - coordinate) +
                                          joint.servo.z * (joint.drive.x - relative);
                    next = clamp(commanded_force * params.dt, -maximum, maximum);
                }
            }
            let delta = next - motor_accumulated;
            if (joint.bodies.z == 2u) {
                a = apply_angular_impulse(a, -drive_axis * delta);
                b = apply_angular_impulse(b, drive_axis * delta);
            } else {
                a = apply_impulse(a, arm_a, -drive_axis * delta);
                b = apply_impulse(b, arm_b, drive_axis * delta);
            }
            motor_accumulated = next;
        }
        angular_impulses[index].w = motor_accumulated;
    }
    var limit_accumulated = impulses[index].w;
    if ((joint.bodies.w & 2u) != 0u) {
        var displacement = dot(error, slide_axis);
        var limit_axis = slide_axis;
        var mass = effective_mass(a, arm_a, limit_axis) + effective_mass(b, arm_b, limit_axis);
        var relative = dot(contact_velocity(b, arm_b) - contact_velocity(a, arm_a), limit_axis);
        if (joint.bodies.z == 2u) {
            limit_axis = normalize(rotate(a.orientation, joint.frame_a.xyz));
            displacement = angles[index].angle;
            mass = dot(limit_axis, inverse_inertia_mul(a, limit_axis)) +
                   dot(limit_axis, inverse_inertia_mul(b, limit_axis));
            relative = dot(b.angular_velocity.xyz - a.angular_velocity.xyz, limit_axis);
        }
        if (mass > 1e-12) {
            let predicted = displacement + relative * params.dt;
            var next = 0.0;
            if (displacement < joint.drive.z || predicted < joint.drive.z) {
                let previous=max(0.0,limit_accumulated);
                let desired=limit_target(joint.drive.z-displacement,joint.bodies.z==2u);
                next=max(0.0,previous+limit_delta(relative,desired,mass,previous,displacement>=joint.drive.z));
            } else if (displacement > joint.drive.w || predicted > joint.drive.w) {
                let previous=min(0.0,limit_accumulated);
                let desired=-limit_target(displacement-joint.drive.w,joint.bodies.z==2u);
                next=min(0.0,previous+limit_delta(relative,desired,mass,previous,displacement<=joint.drive.w));
            }
            let delta = next - limit_accumulated;
            if (joint.bodies.z == 2u) {
                a = apply_angular_impulse(a, -limit_axis * delta);
                b = apply_angular_impulse(b, limit_axis * delta);
            } else {
                a = apply_impulse(a, arm_a, -limit_axis * delta);
                b = apply_impulse(b, arm_b, limit_axis * delta);
            }
            limit_accumulated = next;
        }
    }
    for (var component = 0u; component < linear_components; component += 1u) {
        var direction = axis(component);
        if (joint.bodies.z == 3u) {
            direction = select(slide_tangent, slide_bitangent, component == 1u);
        }
        let mass = effective_mass(a, arm_a, direction) +
                   effective_mass(b, arm_b, direction);
        if (mass <= 1e-12) { continue; }
        let relative = contact_velocity(b, arm_b) - contact_velocity(a, arm_a);
        let delta = direction * bilateral_delta(dot(relative,direction),dot(error,direction),
            mass,dot(accumulated,direction),false);
        a = apply_impulse(a, arm_a, -delta);
        b = apply_impulse(b, arm_b, delta);
        accumulated += delta;
    }
    if (joint.bodies.z == 1u || joint.bodies.z == 3u) {
        let frame_a = quat_mul(a.orientation, joint.frame_a);
        let frame_b = quat_mul(b.orientation, joint.frame_b);
        let difference = quat_mul(frame_b, quat_conjugate(frame_a));
        let shortest = select(-1.0, 1.0, difference.w >= 0.0);
        let rotation_error = 2.0 * shortest * difference.xyz;
        var angular_accumulated = angular_impulses[index].xyz;
        for (var component = 0u; component < 3u; component += 1u) {
            let direction = axis(component);
            let mass = dot(direction, inverse_inertia_mul(a, direction)) +
                       dot(direction, inverse_inertia_mul(b, direction));
            if (mass <= 1e-12) { continue; }
            let relative = b.angular_velocity.xyz - a.angular_velocity.xyz;
            let delta = direction * bilateral_delta(dot(relative,direction),dot(rotation_error,direction),
                mass,dot(angular_accumulated,direction),true);
            a = apply_angular_impulse(a, -delta);
            b = apply_angular_impulse(b, delta);
            angular_accumulated += delta;
        }
        angular_impulses[index] = vec4<f32>(angular_accumulated, angular_impulses[index].w);
    } else if (joint.bodies.z == 2u) {
        let axis_a = normalize(rotate(a.orientation, joint.frame_a.xyz));
        let axis_b = normalize(rotate(b.orientation, joint.frame_b.xyz));
        let pivot = select(vec3<f32>(1.0, 0.0, 0.0),
                           vec3<f32>(0.0, 1.0, 0.0), abs(axis_a.x) > 0.7);
        let tangent = normalize(cross(pivot, axis_a));
        let bitangent = cross(axis_a, tangent);
        var rotation_error = cross(axis_a, axis_b);
        if (dot(axis_a, axis_b) < -0.99 &&
            dot(rotation_error, rotation_error) < 1e-4) {
            rotation_error = 3.14159265 * tangent;
        }
        var angular_accumulated = angular_impulses[index].xyz;
        angular_accumulated -= axis_a * dot(angular_accumulated, axis_a);
        for (var component = 0u; component < 2u; component += 1u) {
            let direction = select(tangent, bitangent, component == 1u);
            let mass = dot(direction, inverse_inertia_mul(a, direction)) +
                       dot(direction, inverse_inertia_mul(b, direction));
            if (mass <= 1e-12) { continue; }
            let relative = b.angular_velocity.xyz - a.angular_velocity.xyz;
            let delta = direction * bilateral_delta(dot(relative,direction),dot(rotation_error,direction),
                mass,dot(angular_accumulated,direction),true);
            a = apply_angular_impulse(a, -delta);
            b = apply_angular_impulse(b, delta);
            angular_accumulated += delta;
        }
        angular_impulses[index] = vec4<f32>(angular_accumulated, angular_impulses[index].w);
    }
    states[joint.bodies.x] = a;
    states[joint.bodies.y] = b;
    impulses[index] = vec4<f32>(accumulated, limit_accumulated);
}

fn warm_island(island: IslandRange) {
    for (var local = 0u; local < island.count; local += 1u) {
        warm_joint(island_indices[island.offset + local]);
    }
}

fn solve_island(island: IslandRange) {
    for (var local = 0u; local < island.count; local += 1u) {
        solve_joint(island_indices[island.offset + local]);
    }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.islands) { return; }
    let island = island_ranges[id.x];
    warm_island(island);
    for (var iteration = 0u; iteration < params.iterations; iteration += 1u) {
        solve_island(island);
    }
}

@compute @workgroup_size(64)
fn coupled_warm(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.islands) { return; }
    warm_island(island_ranges[id.x]);
}

@compute @workgroup_size(64)
fn coupled_iteration(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.islands) { return; }
    solve_island(island_ranges[id.x]);
}
