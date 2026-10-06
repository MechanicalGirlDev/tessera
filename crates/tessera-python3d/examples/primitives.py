"""Run a GPU-resident scene with analytic and convex colliders."""

from tessera3d import (
    GpuPrimitiveBody,
    GpuPrimitiveShape,
    GpuPrimitiveWorld,
    Quaternion,
    Vec3,
)


def vec(x: float, y: float, z: float) -> Vec3:
    return Vec3(x=x, y=y, z=z)


orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
bodies = [
    GpuPrimitiveBody(
        shape=GpuPrimitiveShape.BOX(half_extents=vec(1.0, 1.0, 0.5)),
        center=vec(0.0, 0.0, 3.0),
        orientation=orientation,
        velocity=vec(0.0, 0.0, 0.0),
        angular_velocity=vec(0.0, 0.0, 0.0),
        mass=0.0,
        principal_inertia=vec(0.0, 0.0, 0.0),
    ),
    GpuPrimitiveBody(
        shape=GpuPrimitiveShape.SPHERE(radius=0.5),
        center=vec(1.4, 0.0, 3.0),
        orientation=orientation,
        velocity=vec(-1.0, 0.0, 0.0),
        angular_velocity=vec(0.0, 0.0, 0.0),
        mass=1.0,
        principal_inertia=vec(0.1, 0.1, 0.1),
    ),
    GpuPrimitiveBody(
        shape=GpuPrimitiveShape.CAPSULE(radius=0.25, half_length=0.5),
        center=vec(5.0, 0.0, 1.0),
        orientation=orientation,
        velocity=vec(0.0, 0.0, 0.0),
        angular_velocity=vec(0.0, 0.0, 0.0),
        mass=1.0,
        principal_inertia=vec(0.2, 0.2, 0.1),
    ),
    GpuPrimitiveBody(
        shape=GpuPrimitiveShape.CYLINDER(radius=0.25, half_length=0.5),
        center=vec(8.0, 0.0, 1.0),
        orientation=orientation,
        velocity=vec(0.0, 0.0, 0.0),
        angular_velocity=vec(0.0, 0.0, 0.0),
        mass=1.0,
        principal_inertia=vec(0.2, 0.2, 0.1),
    ),
    GpuPrimitiveBody(
        shape=GpuPrimitiveShape.CONE(radius=0.5, half_length=0.5),
        center=vec(11.0, 0.0, 1.0),
        orientation=orientation,
        velocity=vec(0.0, 0.0, 0.0),
        angular_velocity=vec(0.0, 0.0, 0.0),
        mass=1.0,
        principal_inertia=vec(0.2, 0.2, 0.1),
    ),
    GpuPrimitiveBody(
        shape=GpuPrimitiveShape.CONVEX(
            vertices=[
                vec(-0.5, -0.5, -0.5), vec(0.5, -0.5, -0.5),
                vec(-0.5, 0.5, -0.5), vec(0.5, 0.5, -0.5),
                vec(-0.5, -0.5, 0.5), vec(0.5, -0.5, 0.5),
                vec(-0.5, 0.5, 0.5), vec(0.5, 0.5, 0.5),
            ]
        ),
        center=vec(14.0, 0.0, 1.0),
        orientation=orientation,
        velocity=vec(0.0, 0.0, 0.0),
        angular_velocity=vec(0.0, 0.0, 0.0),
        mass=1.0,
        principal_inertia=vec(0.2, 0.2, 0.2),
    ),
]

world = GpuPrimitiveWorld(bodies, vec(0.0, 0.0, -9.81), 10.0)
world.step(1.0 / 120.0)
for contact in world.readback_contacts():
    print(contact.body_a, contact.body_b, contact.depth)
for index, state in enumerate(world.readback()):
    print(index, state.center, state.velocity)

added_index = world.add_body(bodies[1])
removed = world.remove_body(added_index)
print(added_index, removed.shape)
world.reset()
