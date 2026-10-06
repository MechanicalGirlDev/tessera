"""Check GPU contact impulse diagnostics through the generated Python binding."""

from tessera3d import (
    GpuContactImpulse,
    GpuContactImpulseReadback,
    GpuPrimitiveBatch,
    GpuPrimitiveBody,
    GpuPrimitiveShape,
    GpuPrimitiveWorld,
    GpuSphereBatch,
    GpuSphereWorld,
    Quaternion,
    SphereInput,
    TesseraError,
    Vec3,
)


def vec(x, y, z):
    return Vec3(x=x, y=y, z=z)


def check(history, velocity):
    assert isinstance(history, GpuContactImpulseReadback)
    assert abs(history.dt - 0.01) < 1e-8
    assert len(history.contacts) == 1
    contact = history.contacts[0]
    assert isinstance(contact, GpuContactImpulse)
    assert contact.body_a is None and contact.body_b == 0
    assert contact.point_index == 0 and contact.normal_impulse > 0
    for axis, initial in (('x', 0.5), ('y', 0.0), ('z', -1.0)):
        impulse = getattr(contact.impulse_on_body_b, axis)
        assert abs(impulse - (getattr(velocity, axis) - initial)) < 1e-4
        combined = (
            getattr(contact.normal, axis) * contact.normal_impulse
            + getattr(contact.tangent_impulse, axis)
        )
        assert abs(impulse - combined) < 1e-6


ball = SphereInput(
    center=vec(0.0, 0.0, 0.5), velocity=vec(0.5, 0.0, -1.0),
    radius=0.5, mass=1.0,
)
air = SphereInput(
    center=vec(0.0, 0.0, 3.0), velocity=vec(0.0, 0.0, 0.0),
    radius=0.5, mass=1.0,
)
world = GpuSphereWorld([ball], vec(0.0, 0.0, 0.0), 10.0)
assert world.readback_contact_impulses().dt is None
world.step(0.01)
check(world.readback_contact_impulses(), world.readback()[0].velocity)

batch = GpuSphereBatch([[ball], [air]], vec(0.0, 0.0, 0.0), 10.0)
batch.step(0.01)
check(batch.readback_contact_impulses_environment(0), batch.readback_environment(0)[0].velocity)
assert not batch.readback_contact_impulses_environment(1).contacts
try:
    batch.readback_contact_impulses_environment(2)
except TesseraError:
    pass
else:
    raise AssertionError('invalid environment must fail')
batch.reset_environment(0, [ball])
assert not batch.readback_contact_impulses_environment(0).contacts


def primitive(z):
    return GpuPrimitiveBody(
        shape=GpuPrimitiveShape.SPHERE(radius=0.5), center=vec(0.0, 0.0, z),
        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        velocity=vec(0.5, 0.0, -1.0), angular_velocity=vec(0.0, 0.0, 0.0),
        mass=1.0, principal_inertia=vec(0.1, 0.1, 0.1),
    )


world = GpuPrimitiveWorld([primitive(0.5)], vec(0.0, 0.0, 0.0), 10.0)
world.step(0.01)
check(world.readback_contact_impulses(), world.readback()[0].velocity)
world.reset()
assert world.readback_contact_impulses().dt is None
batch = GpuPrimitiveBatch([[primitive(0.5)], [primitive(3.0)]], vec(0.0, 0.0, 0.0), 10.0)
batch.step(0.01)
check(batch.readback_contact_impulses_environment(0), batch.readback_environment(0)[0].velocity)
assert not batch.readback_contact_impulses_environment(1).contacts
print('contact impulse diagnostics: all four GPU APIs passed')
