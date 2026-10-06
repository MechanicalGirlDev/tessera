"""Rotate one GPU-resident body about a hinge on a static body."""

import math

from tessera3d import (
    GpuAxisMotor,
    GpuAxisServo,
    GpuPrimitiveBatch,
    GpuPrimitiveBody,
    GpuPrimitiveShape,
    GpuPrimitiveWorld,
    GpuRevoluteJoint,
    GpuRevoluteLimit,
    Quaternion,
    Vec3,
)


def vec(x: float, y: float, z: float) -> Vec3:
    return Vec3(x=x, y=y, z=z)


def sphere(x: float, mass: float) -> GpuPrimitiveBody:
    return GpuPrimitiveBody(
        shape=GpuPrimitiveShape.SPHERE(radius=0.1),
        center=vec(x, 0.0, 3.0),
        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        velocity=vec(0.0, 0.0, 0.0),
        angular_velocity=vec(0.0, 0.0, 0.0),
        mass=mass,
        principal_inertia=vec(0.004, 0.004, 0.004) if mass else vec(0.0, 0.0, 0.0),
    )


world = GpuPrimitiveWorld([sphere(0.0, 0.0), sphere(1.0, 1.0)], vec(0.0, 0.0, 0.0), 10.0)
world.set_revolute_joints(
    [
        GpuRevoluteJoint(
            body_a=0,
            body_b=1,
            local_anchor_a=vec(0.0, 0.0, 0.0),
            local_anchor_b=vec(-1.0, 0.0, 0.0),
            local_axis_a=vec(0.0, 0.0, 1.0),
            local_axis_b=vec(0.0, 0.0, 1.0),
        )
    ]
)
world.set_revolute_motor(0, GpuAxisMotor(target_velocity=1.0, max_force=4.0))
for _ in range(120):
    world.step(0.005)
state = world.readback()[1]
assert state.orientation.z > 0.05, state
limit = GpuRevoluteLimit(min=0.0, max=0.6)
servo = GpuAxisServo(
    position_target=0.2, velocity_target=0.0,
    stiffness=30.0, damping=8.0, max_force=4.0,
)
world.set_revolute_limit(0, limit)
world.set_revolute_servo(0, servo)
for _ in range(200):
    world.step(0.005)
state = world.readback()[1]
angle = 2.0 * math.atan2(state.orientation.z, state.orientation.w)
assert abs(angle - 0.2) < 0.05, state
assert abs(world.readback_revolute_angle(0) - 0.2) < 0.05
print(state)

environment = [sphere(0.0, 0.0), sphere(1.0, 1.0)]
batch = GpuPrimitiveBatch([environment, environment], vec(0.0, 0.0, 0.0), 10.0)
batch.set_revolute_joints_environment(0, world.revolute_joints())
batch.set_revolute_servo_environment(0, 0, servo)
batch.set_revolute_limit_environment(0, 0, limit)
batch.set_revolute_joints_environment(1, world.revolute_joints())
batch.step_substeps(0.005, 200)
driven = batch.readback_environment(0)[1]
angle = 2.0 * math.atan2(driven.orientation.z, driven.orientation.w)
assert abs(angle - 0.2) < 0.05, driven
assert abs(batch.readback_revolute_angle_environment(0, 0) - 0.2) < 0.05
assert abs(batch.readback_revolute_angle_environment(1, 0)) < 1e-5
assert abs(batch.readback_environment(1)[1].orientation.z) < 1e-5
