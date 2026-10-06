"""Check independent joint coefficients through all four GPU wheel interfaces."""
import math
from tessera3d import (
    GpuBallJoint, GpuPrimitiveBatch, GpuPrimitiveBody, GpuPrimitiveShape,
    GpuPrimitiveWorld, GpuSphereBatch, GpuSphereWorld, Quaternion, SphereInput,
    TesseraError, Vec3, default_gpu_temporal_settings,
    default_gpu_temporal_joint_settings,
)


def vec(x, y, z):
    return Vec3(x=x, y=y, z=z)


def main():
    contacts = default_gpu_temporal_settings()
    contacts.normal_frequency = 1.0
    contacts.iterations = 2
    contacts.ground_only = True
    contacts.speculative_margin = 0.01
    joints = default_gpu_temporal_joint_settings()
    joints.frequency = 20.0
    joints.iterations = 1
    joints.max_linear_correction_speed = 0.2
    joints.max_angular_correction_speed = 0.3
    h = 0.01
    omega = 2.0 * math.pi * joints.frequency
    a2 = h * omega * (2.0 * joints.damping_ratio + h * omega)
    mass_scale = a2 / (1.0 + a2)
    expected = 1.1 - h * mass_scale * joints.max_linear_correction_speed
    zero = vec(0.0, 0.0, 0.0)
    identity = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    spheres = [SphereInput(center=vec(x, 0.0, 5.0), velocity=zero, radius=0.1, mass=mass)
               for x, mass in [(0.0, 0.0), (1.1, 1.0)]]
    primitives = [GpuPrimitiveBody(shape=GpuPrimitiveShape.SPHERE(radius=0.1),
        center=body.center, orientation=identity, velocity=zero, angular_velocity=zero,
        mass=body.mass, principal_inertia=zero if body.mass == 0.0 else vec(1.0, 1.0, 1.0))
        for body in spheres]
    joint = GpuBallJoint(body_a=0, body_b=1,
        local_anchor_a=vec(1.0, 0.0, 0.0), local_anchor_b=zero)
    worlds = [GpuSphereWorld(spheres, zero, 10.0), GpuPrimitiveWorld(primitives, zero, 10.0)]
    batches = [GpuSphereBatch([spheres, spheres], zero, 10.0),
               GpuPrimitiveBatch([primitives, primitives], zero, 10.0)]
    for world in worlds:
        world.set_ball_joints([joint])
    for batch in batches:
        batch.set_ball_joints_environment(0, [joint])
    for world in worlds + batches:
        world.step_temporal_with_joints(h, 1, contacts, joints)
        states = world.readback() if world in worlds else world.readback_environment(0)
        assert abs(states[1].center.x - expected) < 2e-6, states[1]
        assert abs(states[1].velocity.x) < 2e-6, states[1]
        if world in batches:
            assert abs(world.readback_environment(1)[1].center.x - 1.1) < 2e-6
        bad = default_gpu_temporal_joint_settings()
        bad.frequency = 0.0
        try:
            world.step_temporal_with_joints(h, 1, contacts, bad)
        except TesseraError:
            pass
        else:
            raise AssertionError("invalid joint frequency was accepted")
        after = world.readback() if world in worlds else world.readback_environment(0)
        assert abs(after[1].center.x - states[1].center.x) < 1e-7
    print("independent temporal joint settings passed: four interfaces, analytic clamp, idle environment, preflight")


if __name__ == "__main__":
    main()
