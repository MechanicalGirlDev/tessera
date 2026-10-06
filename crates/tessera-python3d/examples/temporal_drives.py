"""Verify temporal drives and environment isolation through an installed wheel."""
from tessera3d import (
    GpuAxisMotor, GpuAxisServo, GpuPrimitiveBatch, GpuPrimitiveBody,
    GpuPrimitiveShape, GpuPrismaticJoint, GpuPrismaticLimit,
    Quaternion, Vec3, default_gpu_temporal_settings,
)


def vec(x, y, z):
    return Vec3(x=x, y=y, z=z)


def main():
    identity = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

    def sphere(z, mass):
        return GpuPrimitiveBody(
            shape=GpuPrimitiveShape.SPHERE(radius=0.1), center=vec(0.0, 0.0, z),
            orientation=identity, velocity=vec(0.0, 0.0, 0.0),
            angular_velocity=vec(0.0, 0.0, 0.0), mass=mass,
            principal_inertia=vec(mass, mass, mass),
        )

    initial = [sphere(3.0, 0.0), sphere(4.0, 1.0)]
    batch = GpuPrimitiveBatch([initial, initial], vec(0.0, 0.0, 0.0), 10.0)
    joint = GpuPrismaticJoint(
        body_a=0, body_b=1, local_anchor_a=vec(0.0, 0.0, 0.0),
        local_anchor_b=vec(0.0, 0.0, 0.0), local_rotation_a=identity, local_rotation_b=identity,
    )
    for environment in range(2):
        batch.set_prismatic_joints_environment(environment, [joint])
    settings = default_gpu_temporal_settings()
    settings.iterations = 8
    settings.friction = 0.0
    batch.set_prismatic_motor_environment(0, 0, GpuAxisMotor(target_velocity=5.0, max_force=2.0))
    batch.step_temporal(0.1, 4, settings)
    actual = batch.readback_environment(0)[1]
    assert abs(actual.velocity.z - 0.2) < 2e-5
    assert abs(actual.center.z - 4.0125) < 2e-5
    batch.reset_environment(0, initial)
    batch.set_prismatic_servo_environment(0, 0, GpuAxisServo(
        position_target=1.1, velocity_target=0.0, stiffness=100.0, damping=400.0, max_force=100.0,
    ))
    coordinate, velocity = 1.0, 0.0
    for _ in range(4):
        velocity = (velocity + 0.025 * 100.0 * (1.1 - coordinate)) / (1.0 + 0.025 * 400.0)
        coordinate += 0.025 * velocity
    batch.step_temporal(0.1, 4, settings)
    actual = batch.readback_environment(0)[1]
    assert abs(actual.velocity.z - velocity) < 2e-5
    assert abs(actual.center.z - 3.0 - coordinate) < 2e-5
    batch.reset_environment(0, initial)
    batch.set_prismatic_motor_environment(0, 0, GpuAxisMotor(target_velocity=5.0, max_force=2.0))
    batch.set_prismatic_limit_environment(0, 0, GpuPrismaticLimit(min=1.0, max=1.1))
    for _ in range(120):
        batch.step_temporal(1.0 / 120.0, 4, settings)
        actual = batch.readback_environment(0)[1]
        assert 3.997 <= actual.center.z <= 4.103
    assert abs(actual.center.z - 4.1) < 0.003
    untouched = batch.readback_environment(1)[1]
    assert abs(untouched.center.z - 4.0) < 1e-5
    assert abs(untouched.velocity.z) < 1e-5
    print("temporal drive wheel smoke passed: motor budget, implicit servo, soft limit, batch isolation")


if __name__ == "__main__":
    main()
