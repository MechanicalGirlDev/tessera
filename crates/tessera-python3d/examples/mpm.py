"""Simulate a material point pushed by a prescribed rigid obstacle."""

from tessera3d import (
    MpmMaterial,
    MpmObstacleInput,
    MpmObstacleShape,
    MpmParticleInput,
    MpmWorld,
    Quaternion,
    TesseraError,
    Vec3,
)


def vec(x: float, y: float, z: float) -> Vec3:
    return Vec3(x=x, y=y, z=z)


particle = MpmParticleInput(
    position=vec(0.62, 0.5, 0.5),
    velocity=vec(0.0, 0.0, 0.0),
    radius=0.03,
    density=1000.0,
    material=MpmMaterial.ELASTIC(young_modulus=1000.0, poisson_ratio=0.2),
    enabled=True,
    fixed=False,
    damping=0.0,
)
obstacle = MpmObstacleInput(
    shape=MpmObstacleShape.SPHERE(radius=0.1),
    center=vec(0.5, 0.5, 0.5),
    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
    linear_velocity=vec(1.0, 0.0, 0.0),
    angular_velocity=vec(0.0, 0.0, 0.0),
    friction=0.5,
)


def make_world() -> MpmWorld:
    world = MpmWorld(
        particles=[particle],
        gravity=vec(0.0, 0.0, 0.0),
        cell_width=0.1,
        max_substep=0.001,
        bounds_min=vec(0.0, 0.0, 0.0),
        bounds_max=vec(1.0, 1.0, 1.0),
    )
    world.set_obstacles([obstacle])
    return world


cpu = make_world()
cpu.step(0.0001)
print("CPU", cpu.particles()[0].position, cpu.particles()[0].velocity)

gpu = make_world()
try:
    gpu.step_gpu_fixed(0.0001, 1)
    print("GPU", gpu.particles()[0].position, gpu.particles()[0].velocity)
except TesseraError as error:
    print("GPU unavailable:", error)

chunk = cpu.add_particles([particle])
print("new chunk", chunk, "particle count", len(cpu.particles()))
cpu.remove_chunk(chunk)
