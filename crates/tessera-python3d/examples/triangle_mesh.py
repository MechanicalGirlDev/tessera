"""Resolve a sphere against a two-sided GPU-resident triangle mesh."""

from tessera3d import (
    GpuPrimitiveBody,
    GpuPrimitiveShape,
    GpuPrimitiveWorld,
    GpuTriangle,
    Quaternion,
    Vec3,
)


def vec(x: float, y: float, z: float) -> Vec3:
    return Vec3(x=x, y=y, z=z)


orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
mesh = GpuPrimitiveBody(
    shape=GpuPrimitiveShape.TRIANGLE_MESH(
        vertices=[
            vec(-1.0, -1.0, 0.0),
            vec(1.0, -1.0, 0.0),
            vec(1.0, 1.0, 0.0),
            vec(-1.0, 1.0, 0.0),
        ],
        triangles=[GpuTriangle(a=0, b=1, c=2), GpuTriangle(a=0, b=2, c=3)],
    ),
    center=vec(0.0, 0.0, 2.0),
    orientation=orientation,
    velocity=vec(0.0, 0.0, 0.0),
    angular_velocity=vec(0.0, 0.0, 0.0),
    mass=0.0,
    principal_inertia=vec(0.0, 0.0, 0.0),
)
sphere = GpuPrimitiveBody(
    shape=GpuPrimitiveShape.SPHERE(radius=0.5),
    center=vec(0.0, 0.0, 2.25),
    orientation=orientation,
    velocity=vec(0.0, 0.0, 0.0),
    angular_velocity=vec(0.0, 0.0, 0.0),
    mass=1.0,
    principal_inertia=vec(0.1, 0.1, 0.1),
)
capsule = GpuPrimitiveBody(
    shape=GpuPrimitiveShape.CAPSULE(radius=0.25, half_length=0.3),
    center=vec(0.0, -1.1, 2.0),
    orientation=orientation,
    velocity=vec(0.0, 0.0, 0.0),
    angular_velocity=vec(0.0, 0.0, 0.0),
    mass=1.0,
    principal_inertia=vec(0.1, 0.1, 0.1),
)
world = GpuPrimitiveWorld([mesh, sphere, capsule], vec(0.0, 0.0, 0.0), 10.0)
world.step(1.0 / 120.0)
for contact in world.readback_contacts():
    print(contact.body_a, contact.body_b, contact.depth)
