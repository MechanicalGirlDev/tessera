"""Run temporal frames through an installed Tessera wheel."""
from tessera3d import (
    GpuSphereBatch, GpuSphereWorld, SphereInput, TesseraError, Vec3,
    GpuPrimitiveBatch, GpuPrimitiveBody, GpuPrimitiveShape, GpuPrimitiveWorld, Quaternion,
    GpuFixedJoint,
    default_gpu_temporal_settings,
)

def vec(x, y, z):
    return Vec3(x=x, y=y, z=z)

def main():
    settings = default_gpu_temporal_settings()
    settings.friction = 0.0
    settings.iterations = 4
    settings.speculative_margin = 0.1
    ball = SphereInput(center=vec(0.0, 0.0, 0.55), velocity=vec(0.0, 0.0, -1.0), radius=0.5, mass=1.0)
    world = GpuSphereWorld([ball], vec(0.0, 0.0, 0.0), 10.0)
    world.step_temporal(0.1, 4, settings)
    assert world.readback()[0].center.z > 0.49
    batch = GpuSphereBatch([[ball], [ball]], vec(0.0, 0.0, 0.0), 10.0)
    batch.step_temporal(0.1, 4, settings)
    assert all(batch.readback_environment(i)[0].center.z > 0.49 for i in range(2))
    box = GpuPrimitiveBody(
        shape=GpuPrimitiveShape.BOX(half_extents=vec(0.5, 0.5, 0.5)),
        center=vec(0.0, 0.0, 0.55),
        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        velocity=vec(0.0, 0.0, -1.0), angular_velocity=vec(0.0, 0.0, 0.0),
        mass=1.0, principal_inertia=vec(1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0),
    )
    settings.ground_only = True
    primitive = GpuPrimitiveWorld([box], vec(0.0, 0.0, 0.0), 10.0)
    primitive.step_temporal(0.1, 4, settings)
    assert primitive.readback()[0].center.z > 0.49
    primitive_batch = GpuPrimitiveBatch([[box], [box]], vec(0.0, 0.0, 0.0), 10.0)
    primitive_batch.step_temporal(0.1, 4, settings)
    assert all(primitive_batch.readback_environment(i)[0].center.z > 0.49 for i in range(2))
    linked = GpuSphereWorld([
        SphereInput(center=vec(-1.0, 0.0, 0.55), velocity=vec(0.0, 0.0, -1.0), radius=0.5, mass=1.0),
        SphereInput(center=vec(1.0, 0.0, 0.55), velocity=vec(0.0, 0.0, -1.0), radius=0.5, mass=1.0),
    ], vec(0.0, 0.0, -9.81), 10.0)
    identity = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    linked.set_fixed_joints([GpuFixedJoint(
        body_a=0, body_b=1, local_anchor_a=vec(1.0, 0.0, 0.0),
        local_anchor_b=vec(-1.0, 0.0, 0.0), local_rotation_a=identity, local_rotation_b=identity,
    )])
    settings.iterations = 8
    linked.write_force(0, vec(2.0, 0.0, 0.0))
    linked.step_temporal(0.1, 8, settings)
    for _ in range(12):
        linked.step_temporal(1.0 / 60.0, 4, settings)
        states = linked.readback()
        assert all(state.center.z > 0.49 for state in states)
        assert abs(states[1].center.x - states[0].center.x - 2.0) < 0.01
        assert abs((states[0].velocity.x + states[1].velocity.x) * 0.5 - 0.1) < 2e-5
    free = GpuSphereWorld([SphereInput(center=vec(0.0, 0.0, 5.0), velocity=vec(0.0, 0.0, 0.0), radius=0.5, mass=1.0)], vec(0.0, 0.0, 0.0), 10.0)
    free.write_force(0, vec(2.0, 0.0, 0.0))
    invalid = default_gpu_temporal_settings()
    invalid.normal_frequency = 0.0
    try:
        free.step_temporal(0.1, 4, invalid)
    except TesseraError:
        pass
    else:
        raise AssertionError("invalid temporal frequency was accepted")
    free.step_temporal(0.1, 4, None)
    first = free.readback()[0]
    assert abs(first.velocity.x - 0.2) < 1e-6
    assert abs(first.center.x - 0.0125) < 1e-6
    free.step_temporal(0.1, 4, None)
    second = free.readback()[0]
    assert abs(second.velocity.x - 0.2) < 1e-6
    assert abs(second.center.x - 0.0325) < 1e-6
    print("temporal wheel smoke passed: ground, independent batches, fixed joint support, force lifetime, invalid settings")

if __name__ == "__main__":
    main()
