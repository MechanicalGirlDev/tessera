"""Lock two GPU-resident rigid frames through a fixed joint."""

from tessera3d import (
    GpuFixedJoint,
    GpuPrimitiveBody,
    GpuPrimitiveShape,
    GpuPrimitiveWorld,
    Quaternion,
    Vec3,
)


def vec(x: float, y: float, z: float) -> Vec3:
    return Vec3(x=x, y=y, z=z)


identity = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)


def body(x: float, mass: float) -> GpuPrimitiveBody:
    return GpuPrimitiveBody(
        shape=GpuPrimitiveShape.BOX(half_extents=vec(0.2, 0.2, 0.2)),
        center=vec(x, 0.0, 2.0),
        orientation=identity,
        velocity=vec(0.0, 0.0, 0.0),
        angular_velocity=vec(0.0, 0.0, 0.0),
        mass=mass,
        principal_inertia=vec(0.03, 0.03, 0.03) if mass else vec(0.0, 0.0, 0.0),
    )


world = GpuPrimitiveWorld([body(0.0, 0.0), body(1.0, 1.0)], vec(0.0, 0.0, 0.0), 10.0)
world.set_fixed_joints(
    [
        GpuFixedJoint(
            body_a=0,
            body_b=1,
            local_anchor_a=vec(1.0, 0.0, 0.0),
            local_anchor_b=vec(0.0, 0.0, 0.0),
            local_rotation_a=identity,
            local_rotation_b=identity,
        )
    ]
)
for _ in range(60):
    world.write_wrench(1, vec(0.0, 10.0, 0.0), vec(0.0, 0.0, 5.0))
    world.step(0.005)
print(world.readback()[1])
