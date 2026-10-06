"""Resolve sphere, capsule, and convex hull contacts against a GPU polyline."""

from tessera3d import (
    GpuPrimitiveBody,
    GpuPrimitiveShape,
    GpuPrimitiveWorld,
    GpuSegment,
    Quaternion,
    Vec3,
)


def vec(x: float, y: float, z: float) -> Vec3:
    return Vec3(x=x, y=y, z=z)


orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
line = GpuPrimitiveBody(
    shape=GpuPrimitiveShape.POLYLINE(
        vertices=[
            vec(-1.0, -1.0, 0.0),
            vec(1.0, -1.0, 0.0),
            vec(1.0, 1.0, 0.0),
            vec(-1.0, 1.0, 0.0),
        ],
        segments=[GpuSegment(a=0, b=1), GpuSegment(a=1, b=2), GpuSegment(a=2, b=3)],
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
    center=vec(0.0, -1.0, 2.25),
    orientation=orientation,
    velocity=vec(0.0, 0.0, 0.0),
    angular_velocity=vec(0.0, 0.0, 0.0),
    mass=1.0,
    principal_inertia=vec(0.1, 0.1, 0.1),
)
capsule = GpuPrimitiveBody(
    shape=GpuPrimitiveShape.CAPSULE(radius=0.5, half_length=0.3),
    center=vec(1.0, 0.0, 2.15),
    orientation=Quaternion(x=2.0**-0.5, y=0.0, z=0.0, w=2.0**-0.5),
    velocity=vec(0.0, 0.0, 0.0),
    angular_velocity=vec(0.0, 0.0, 0.0),
    mass=1.0,
    principal_inertia=vec(0.1, 0.1, 0.1),
)
hull = GpuPrimitiveBody(
    shape=GpuPrimitiveShape.CONVEX(
        vertices=[vec(x, y, z) for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)],
    ),
    center=vec(0.0, 1.0, 2.4),
    orientation=orientation,
    velocity=vec(0.0, 0.0, 0.0),
    angular_velocity=vec(0.0, 0.0, 0.0),
    mass=1.0,
    principal_inertia=vec(0.1, 0.1, 0.1),
)
world = GpuPrimitiveWorld([line, sphere, capsule, hull], vec(0.0, 0.0, 0.0), 10.0)
world.step(1.0 / 120.0)
contacts = world.readback_contacts()
assert len(contacts) == 5
assert all(contact.body_a == 0 for contact in contacts)
assert {contact.body_b for contact in contacts} == {1, 2, 3}
expected_depth = {1: 0.25, 2: 0.35, 3: 0.1}
assert all(abs(contact.depth - expected_depth[contact.body_b]) < 1e-3 for contact in contacts)
capsule_contacts = [contact for contact in contacts if contact.body_b == 2]
assert abs(capsule_contacts[0].point.y - capsule_contacts[1].point.y) > 0.5
hull_contacts = [contact for contact in contacts if contact.body_b == 3]
assert abs(hull_contacts[0].point.x - hull_contacts[1].point.x) > 0.99
print("GPU polyline sphere/capsule/hull depths:", [contact.depth for contact in contacts])
