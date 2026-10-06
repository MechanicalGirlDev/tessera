"""Verify GPU MPM reactions reach sphere and box rigid bodies through Python."""

from tessera3d import (
    GpuPrimitiveBody,
    GpuPrimitiveShape,
    GpuPrimitiveWorld,
    GpuSphereWorld,
    MpmBoundary,
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
identity = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)


def make_mpm():
    particle = MpmParticleInput(
        position=vec(2.02, 1.5, 1.5),
        velocity=vec(-1.0, 0.0, 0.0),
        radius=0.04,
        density=1000.0,
        material=MpmMaterial.ELASTIC(young_modulus=1000.0, poisson_ratio=0.2),
        enabled=True,
        fixed=False,
        damping=0.0,
    )
    return MpmWorld(
        particles=[particle],
        gravity=zero,
        cell_width=0.1,
        max_substep=0.001,
        bounds_min=vec(0.0, 0.0, 0.0),
        bounds_max=vec(3.0, 3.0, 3.0),
    )


def check(rigid, suffix):
    mpm = make_mpm()
    submit = getattr(mpm, f"submit_gpu_two_way_with_{suffix}_world")
    step = getattr(mpm, f"step_gpu_two_way_with_{suffix}_world")
    submit(rigid, 0.001)
    assert mpm.pending_gpu_substeps() == 1
    assert mpm.substeps() == 0
    assert rigid.readback()[0].velocity.x < -0.01
    mpm.synchronize_gpu()
    assert mpm.pending_gpu_substeps() == 0
    assert mpm.substeps() == 1
    assert mpm.particles()[0].velocity.x > -1.0
    mpm.set_obstacle_boundaries([MpmBoundary.STICK, MpmBoundary.NON_REFLECTING])
    assert mpm.obstacle_boundaries() == [MpmBoundary.STICK, MpmBoundary.NON_REFLECTING]
    rigid.step(0.001)
    step(rigid, 0.0005)
    assert mpm.substeps() == 2
    assert mpm.obstacle_boundaries() == [MpmBoundary.STICK, MpmBoundary.NON_REFLECTING]
    submit(rigid, 0.0005)
    rigid.step(0.001)
    submit(rigid, 0.0005)
    assert mpm.pending_gpu_substeps() == 2
    assert mpm.substeps() == 2
    mpm.synchronize_gpu()
    assert mpm.substeps() == 4

    owned = make_mpm()
    before = rigid.readback()[0].center.x
    getattr(owned, f"submit_gpu_owned_with_{suffix}_world")(rigid, 0.001)
    assert owned.pending_gpu_substeps() == 1
    assert rigid.readback()[0].center.x < before
    owned.synchronize_gpu()
    getattr(owned, f"step_gpu_owned_with_{suffix}_world")(rigid, 0.001)
    assert owned.substeps() == 2


sphere = GpuSphereWorld(
    bodies=[SphereInput(center=vec(1.0, 1.5, 1.5), velocity=zero, radius=1.0, mass=1.0)],
    gravity=zero,
    ground_half_extent=0.1,
)
check(sphere, "sphere")

box = GpuPrimitiveWorld(
    bodies=[
        GpuPrimitiveBody(
            shape=GpuPrimitiveShape.BOX(half_extents=vec(1.0, 1.0, 1.0)),
            center=vec(1.0, 1.5, 1.5),
            orientation=identity,
            velocity=zero,
            angular_velocity=zero,
            mass=1.0,
            principal_inertia=vec(1.0, 1.0, 1.0),
        )
    ],
    gravity=zero,
    ground_half_extent=0.1,
)
check(box, "primitive")

print("GPU MPM two-way Python coupling passed: sphere and box")
