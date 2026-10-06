"""Validate prescribed motion through all four generated GPU Python APIs."""

from tessera3d import (
    GpuKinematicMotion,
    GpuPrimitiveBatch,
    GpuPrimitiveBody,
    GpuPrimitiveShape,
    GpuPrimitiveWorld,
    GpuSphereBatch,
    GpuSphereWorld,
    Quaternion,
    SphereInput,
    Vec3,
)


def vec(x=0.0, y=0.0, z=0.0):
    return Vec3(x=x, y=y, z=z)


def primitive(x, mass):
    return GpuPrimitiveBody(
        shape=GpuPrimitiveShape.SPHERE(radius=0.5),
        center=vec(x, 0.0, 3.0),
        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        velocity=vec(),
        angular_velocity=vec(),
        mass=mass,
        principal_inertia=vec(0.1, 0.1, 0.1) if mass else vec(),
    )


def sphere(x, mass):
    return SphereInput(center=vec(x, 0.0, 3.0), velocity=vec(), radius=0.5, mass=mass)


def check_api(factory, make_body, batch):
    bodies = [make_body(0.0, 0.0), make_body(10.0, 1.0)]
    world = factory([bodies, bodies] if batch else bodies, vec(), 20.0)

    def set_motion(index, motion):
        if batch:
            world.set_kinematic_motion(1, index, motion)
        else:
            world.set_kinematic_motion(index, motion)

    def read():
        return world.readback_environment(1) if batch else world.readback()

    initial = read()
    try:
        set_motion(0, GpuKinematicMotion(linear=vec(1e300), angular=vec()))
    except Exception:
        pass
    else:
        raise AssertionError("f32 overflow must be rejected")
    assert read()[0].velocity.x == 0.0
    try:
        set_motion(2, None)
    except Exception:
        pass
    else:
        raise AssertionError("out-of-range body must be rejected")
    set_motion(0, GpuKinematicMotion(linear=vec(0.5), angular=vec(0.0, 0.0, 0.25)))
    set_motion(1, GpuKinematicMotion(linear=vec(100.0), angular=vec(100.0)))
    assert read()[0].center.x == initial[0].center.x
    assert read()[1].velocity.x == 0.0
    world.step(0.1)
    moving = read()
    assert abs(moving[0].center.x - 0.05) < 1e-5
    assert abs(moving[0].angular_velocity.z - 0.25) < 1e-5
    assert moving[1].center.x == initial[1].center.x
    if batch:
        unchanged = world.readback_environment(0)
        assert unchanged[0].center.x == 0.0
        assert unchanged[0].velocity.x == 0.0
    set_motion(0, None)
    world.step(0.1)
    stopped = read()[0]
    assert stopped.center.x == moving[0].center.x
    assert stopped.velocity.x == 0.0
    assert stopped.angular_velocity.z == 0.0
    print(f"{factory.__name__}: prescribed motion, stop, validation and isolation passed", flush=True)


for api, body, batched in [
    (GpuSphereWorld, sphere, False),
    (GpuSphereBatch, sphere, True),
    (GpuPrimitiveWorld, primitive, False),
    (GpuPrimitiveBatch, primitive, True),
]:
    check_api(api, body, batched)
