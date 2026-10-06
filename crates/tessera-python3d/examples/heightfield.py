"""Check elevated GPU terrain through the generated Python binding."""

from tessera3d import (
    GpuPrimitiveBatch, GpuPrimitiveBody, GpuPrimitiveShape, GpuPrimitiveWorld,
    Quaternion, Vec3,
)


def vec(x, y, z):
    return Vec3(x=x, y=y, z=z)


def body(shape, x, z):
    return GpuPrimitiveBody(
        shape=shape, center=vec(x, 0.25, z),
        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        velocity=vec(0.0, 0.0, 0.0), angular_velocity=vec(0.0, 0.0, 0.0),
        mass=0.0, principal_inertia=vec(0.0, 0.0, 0.0),
    )


terrain = body(GpuPrimitiveShape.HEIGHTFIELD(
    rows=2, columns=4, heights=[0.0, 0.0, 1.0, 1.0] * 2,
    scale=vec(6.0, 4.0, 1.0),
), 0.0, 3.0)
sphere = GpuPrimitiveShape.SPHERE(radius=0.5)
bodies = [terrain, body(sphere, -2.0, 3.25), body(sphere, 2.0, 4.25)]


def check(contacts):
    pairs = [contact for contact in contacts if contact.body_b is not None]
    assert {contact.body_b for contact in pairs} == {1, 2}
    for contact in pairs:
        assert contact.body_a == 0
        assert abs(contact.depth - 0.25) < 1e-3
        assert abs(contact.normal.z - 1.0) < 1e-3


world = GpuPrimitiveWorld(bodies, vec(0.0, 0.0, 0.0), 10.0)
world.step(0.01)
check(world.readback_contacts())
assert isinstance(world.shape(0), GpuPrimitiveShape.HEIGHTFIELD)
world.reset()
batch = GpuPrimitiveBatch([bodies, bodies], vec(0.0, 0.0, 0.0), 10.0)
batch.step(0.01)
for environment in (0, 1):
    check(batch.readback_contacts_environment(environment))
batch.reset_environment(1, bodies)
print('heightfield terrain: single world and batch passed')
