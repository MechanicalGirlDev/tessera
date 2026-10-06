"""Slide one GPU-resident body along a static body's local Z axis."""

from tessera3d import (
    GpuAxisMotor,
    GpuAxisServo,
    GpuPrimitiveBatch,
    GpuPrimitiveBody,
    GpuPrimitiveShape,
    GpuPrimitiveWorld,
    GpuPrismaticJoint,
    GpuPrismaticLimit,
    Quaternion,
    Vec3,
)


def vec(x: float, y: float, z: float) -> Vec3:
    return Vec3(x=x, y=y, z=z)


identity = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)


def sphere(z: float, mass: float) -> GpuPrimitiveBody:
    return GpuPrimitiveBody(
        shape=GpuPrimitiveShape.SPHERE(radius=0.1),
        center=vec(0.0, 0.0, z),
        orientation=identity,
        velocity=vec(0.0, 0.0, 0.0),
        angular_velocity=vec(0.0, 0.0, 0.0),
        mass=mass,
        principal_inertia=vec(0.004, 0.004, 0.004) if mass else vec(0.0, 0.0, 0.0),
    )


world = GpuPrimitiveWorld([sphere(3.0, 0.0), sphere(4.0, 1.0)], vec(0.0, 0.0, 0.0), 10.0)
world.set_prismatic_joints(
    [
        GpuPrismaticJoint(
            body_a=0,
            body_b=1,
            local_anchor_a=vec(0.0, 0.0, 0.0),
            local_anchor_b=vec(0.0, 0.0, 0.0),
            local_rotation_a=identity,
            local_rotation_b=identity,
        )
    ]
)
world.set_prismatic_motor(0, GpuAxisMotor(target_velocity=2.0, max_force=4.0))
world.set_prismatic_limit(0, GpuPrismaticLimit(min=1.0, max=1.2))
for _ in range(100):
    world.step(0.005)

state = world.readback()[1]
assert 4.1 < state.center.z < 4.25, state
assert abs(state.center.x) < 0.05 and abs(state.center.y) < 0.05, state
servo = GpuAxisServo(
    position_target=1.1, velocity_target=0.0,
    stiffness=20.0, damping=8.0, max_force=5.0,
)
world.set_prismatic_servo(0, servo)
for _ in range(200):
    world.step(0.005)
state = world.readback()[1]
assert abs(state.center.z - 4.1) < 0.04, state
print(state)

environment = [sphere(3.0, 0.0), sphere(4.0, 1.0)]
batch = GpuPrimitiveBatch([environment, environment], vec(0.0, 0.0, 0.0), 10.0)
batch.set_prismatic_joints_environment(0, world.prismatic_joints())
batch.set_prismatic_motor_environment(0, 0, GpuAxisMotor(target_velocity=2.0, max_force=4.0))
batch.set_prismatic_limit_environment(0, 0, GpuPrismaticLimit(min=1.0, max=1.2))
batch.set_prismatic_joints_environment(1, world.prismatic_joints())
batch.step_substeps(0.005, 100)
batch.set_prismatic_servo_environment(0, 0, servo)
batch.step_substeps(0.005, 200)
assert abs(batch.readback_environment(0)[1].center.z - 4.1) < 0.04
assert abs(batch.readback_environment(1)[1].center.z - 4.0) < 1e-5
