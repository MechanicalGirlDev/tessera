"""Check GPU capsule/convex face and corner contacts through the Python API."""

from math import cos, sin, sqrt
from tessera3d import GpuPrimitiveBody, GpuPrimitiveShape, GpuPrimitiveWorld, Quaternion, Vec3


def vec(x=0.0, y=0.0, z=0.0):
    return Vec3(x=x, y=y, z=z)


def body(shape, center, pitch=0.0):
    return GpuPrimitiveBody(
        shape=shape, center=center,
        orientation=Quaternion(x=0.0, y=sin(pitch / 2), z=0.0, w=cos(pitch / 2)),
        velocity=vec(), angular_velocity=vec(), mass=0.0, principal_inertia=vec(),
    )


hull_shape = GpuPrimitiveShape.CONVEX(vertices=[
    vec(x, y, z) for x in [-0.5, 0.5] for y in [-0.5, 0.5] for z in [-0.5, 0.5]
])
cases = [(False, False, 0.0), (False, True, 0.0),
         (False, False, 0.01), (False, True, 0.01),
         (True, False, 0.0), (True, True, 0.0)]
bodies = []
for group, (corner, reversed_order, pitch) in enumerate(cases):
    origin_x = group * 10.0
    hull = body(hull_shape, vec(origin_x, 0.0, 3.0), pitch)
    capsule = body(
        GpuPrimitiveShape.CAPSULE(radius=0.5, half_length=0.1 if corner else 2.0),
        vec(origin_x + (0.8 if corner else 0.9), 0.8 if corner else 0.0, 3.7 if corner else 3.0),
    )
    bodies.extend([capsule, hull] if reversed_order else [hull, capsule])

world = GpuPrimitiveWorld(bodies, vec(), 100.0)
world.step(0.001)
contacts = world.readback_contacts()
assert len(contacts) == 10, contacts
for group, (corner, reversed_order, pitch) in enumerate(cases):
    pair = [contact for contact in contacts if contact.body_a == group * 2 and contact.body_b == group * 2 + 1]
    assert len(pair) == (1 if corner else 2), (group, pair)
    sign = -1 if reversed_order else 1
    if corner:
        distance = sqrt(0.3**2 * 2 + 0.1**2)
        contact = pair[0]
        assert abs(contact.depth - (0.5 - distance)) < 1e-4
        for actual, delta, vertex in zip(
            [contact.point.x, contact.point.y, contact.point.z],
            [0.3, 0.3, 0.1], [group * 10.0 + 0.5, 0.5, 3.5],
        ):
            expected = vertex + delta / distance * (distance - 0.5) * 0.5
            assert abs(actual - expected) < 1e-4, contact
        for actual, delta in zip([contact.normal.x, contact.normal.y, contact.normal.z], [0.3, 0.3, 0.1]):
            assert abs(actual * sign - delta / distance) < 1e-4
    else:
        assert abs(pair[0].point.z - pair[1].point.z) > 0.9
        for contact in pair:
            assert abs(contact.normal.x * sign - cos(pitch)) < 1e-4
            assert abs(contact.normal.z * sign + sin(pitch)) < 1e-4
            if pitch == 0.0:
                assert abs(contact.depth - 0.1) < 1e-4
                assert abs(abs(contact.point.z - 3.0) - 0.5) < 1e-4
print("primitive capsule convex clipped face, tilted face and corner Python API passed")
