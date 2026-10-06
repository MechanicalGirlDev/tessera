"""Synchronize live GPU rigid colliders into one-way GPU MPM obstacles."""

from tessera3d import (
    GpuPrimitiveBody,
    GpuPrimitiveShape,
    GpuPrimitiveWorld,
    GpuSphereWorld,
    MpmMaterial,
    MpmParticleInput,
    MpmWorld,
    Quaternion,
    SphereInput,
    Vec3,
)


def vec(x, y, z):
    return Vec3(x=x, y=y, z=z)


zero = vec(0.0, 0.0, 0.0)
center = vec(0.5, 0.5, 0.5)
velocity = vec(0.1, 0.0, 0.0)
identity = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
particle = MpmParticleInput(
    position=vec(0.63, 0.5, 0.5),
    velocity=zero,
    radius=0.03,
    density=1000.0,
    material=MpmMaterial.ELASTIC(young_modulus=1000.0, poisson_ratio=0.2),
    enabled=True,
    fixed=False,
    damping=0.0,
)


def mpm_world():
    return MpmWorld(
        particles=[particle],
        gravity=zero,
        cell_width=0.1,
        max_substep=0.001,
        bounds_min=vec(0.0, 0.0, 0.0),
        bounds_max=vec(1.0, 1.0, 1.0),
    )


sphere = GpuSphereWorld(
    bodies=[SphereInput(center=center, velocity=velocity, radius=0.1, mass=1.0)],
    gravity=zero,
    ground_half_extent=10.0,
)
box = GpuPrimitiveWorld(
    bodies=[
        GpuPrimitiveBody(
            shape=GpuPrimitiveShape.BOX(half_extents=vec(0.1, 0.1, 0.1)),
            center=center,
            orientation=identity,
            velocity=velocity,
            angular_velocity=zero,
            mass=1.0,
            principal_inertia=vec(1.0, 1.0, 1.0),
        )
    ],
    gravity=zero,
    ground_half_extent=10.0,
)
for rigid, step, submit in (
    (sphere, "step_gpu_fixed_with_sphere_world", "submit_gpu_fixed_with_sphere_world"),
    (box, "step_gpu_fixed_with_primitive_world", "submit_gpu_fixed_with_primitive_world"),
):
    mpm = mpm_world()
    getattr(mpm, step)(rigid, 0.0001, 1)
    assert mpm.obstacle_count() == 2
    rigid.step(0.01)
    getattr(mpm, step)(rigid, 0.0001, 1)
    assert mpm.obstacle_count() == 2
    rigid.step(0.01)
    getattr(mpm, step)(rigid, 0.0001, 1)
    state = mpm.particles()[0]
    assert all(abs(getattr(state.position, axis)) < 1e6 for axis in ("x", "y", "z"))
    assert state.velocity.x > 0.05

    before = mpm.substeps()
    for _ in range(2):
        rigid.step(0.01)
        getattr(mpm, submit)(rigid, 0.0001, 1)
    assert mpm.pending_gpu_substeps() == 2
    assert mpm.substeps() == before
    try:
        mpm.step_gpu_fixed(0.0001, 1)
    except Exception as error:
        assert "synchronize" in str(error)
    else:
        raise AssertionError("pending GPU steps were discarded")
    assert mpm.pending_gpu_substeps() == 2
    mpm.synchronize_gpu()
    assert mpm.pending_gpu_substeps() == 0
    assert mpm.substeps() == before + 2
    assert mpm.particles()[0].velocity.x > 0.05

print("GPU resident rigid to MPM wheel coupling passed: sphere, primitive, queued frames")
