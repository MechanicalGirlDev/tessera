"""Advance a GPU-resident two-body point joint without per-step readback."""

from tessera3d import (
    GpuBallJoint,
    GpuPrimitiveBody,
    GpuPrimitiveShape,
    GpuPrimitiveWorld,
    Quaternion,
    Vec3,
)


def vec(x: float, y: float, z: float) -> Vec3:
    return Vec3(x=x, y=y, z=z)


def sphere(x: float, mass: float) -> GpuPrimitiveBody:
    return GpuPrimitiveBody(
        shape=GpuPrimitiveShape.SPHERE(radius=0.1),
        center=vec(x, 0.0, 2.0),
        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        velocity=vec(0.0, 0.0, 0.0),
        angular_velocity=vec(0.0, 0.0, 0.0),
        mass=mass,
        principal_inertia=vec(0.004, 0.004, 0.004) if mass else vec(0.0, 0.0, 0.0),
    )


world = GpuPrimitiveWorld([sphere(0.0, 0.0), sphere(1.0, 1.0)], vec(0.0, 0.0, 0.0), 10.0)
world.set_ball_joints(
    [
        GpuBallJoint(
            body_a=0,
            body_b=1,
            local_anchor_a=vec(1.0, 0.0, 0.0),
            local_anchor_b=vec(0.0, 0.0, 0.0),
        )
    ]
)
for _ in range(60):
    world.write_wrench(1, vec(0.0, 10.0, 0.0), vec(0.0, 0.0, 0.0))
    world.step(0.01)
print(world.readback()[1])
