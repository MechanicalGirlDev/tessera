"""Run independent mixed-primitive environments in one GPU batch."""

from tessera3d import (
    GpuPrimitiveBatch,
    GpuPrimitiveBody,
    GpuPrimitiveShape,
    Quaternion,
    Vec3,
)


def vec(x: float, y: float, z: float) -> Vec3:
    return Vec3(x=x, y=y, z=z)


def body(shape: GpuPrimitiveShape, x: float, mass: float) -> GpuPrimitiveBody:
    inertia = vec(0.2, 0.2, 0.2) if mass > 0.0 else vec(0.0, 0.0, 0.0)
    return GpuPrimitiveBody(
        shape=shape,
        center=vec(x, 0.0, 3.0),
        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        velocity=vec(0.0, 0.0, 0.0),
        angular_velocity=vec(0.0, 0.0, 0.0),
        mass=mass,
        principal_inertia=inertia,
    )


box_environment = [
    body(GpuPrimitiveShape.BOX(half_extents=vec(1.0, 1.0, 1.0)), 0.0, 0.0),
    body(GpuPrimitiveShape.SPHERE(radius=0.5), 1.4, 1.0),
]
capsule_environment = [
    body(GpuPrimitiveShape.CAPSULE(radius=0.5, half_length=0.5), 0.0, 0.0),
    body(GpuPrimitiveShape.SPHERE(radius=0.5), 0.9, 1.0),
]
batch = GpuPrimitiveBatch(
    [box_environment, capsule_environment], vec(0.0, 0.0, 0.0), 10.0
)
batch.step(0.01)
print(batch.readback_contacts_environment(0))
print(batch.readback_contacts_environment(1))
batch.step_substeps(1.0 / 120.0, 9)
print(batch.readback_environment(0))
print(batch.readback_environment(1))

batch.reset_environment(0, box_environment)
