"""Verify independent MPM grid transfer regions in CPU and WebGPU steps."""

from tessera3d import MpmMaterial, MpmParticleInput, MpmWorld, Vec3


def vec(x, y, z):
    return Vec3(x=x, y=y, z=z)


def particle(vx):
    return MpmParticleInput(
        position=vec(0.5, 0.5, 0.5),
        velocity=vec(vx, 0.0, 0.0),
        radius=0.04,
        density=1000.0,
        material=MpmMaterial.ELASTIC(young_modulus=1000.0, poisson_ratio=0.2),
        enabled=True,
        fixed=False,
        damping=0.0,
    )


def world():
    return MpmWorld(
        particles=[particle(1.0), particle(-1.0)],
        gravity=vec(0.0, 0.0, 0.0),
        cell_width=0.1,
        max_substep=0.001,
        bounds_min=vec(0.0, 0.0, 0.0),
        bounds_max=vec(1.0, 1.0, 1.0),
    )


cpu = world()
cpu.set_particle_transfer_color(1, 1)
cpu.step(0.001)
assert abs(cpu.particles()[0].velocity.x - 1.0) < 1e-8
assert abs(cpu.particles()[1].velocity.x + 1.0) < 1e-8

gpu = world()
gpu.set_particle_transfer_color(1, 1)
gpu.step_gpu_fixed(0.001, 1)
assert [state.transfer_color for state in gpu.particles()] == [0, 1]
assert abs(gpu.particles()[0].velocity.x - cpu.particles()[0].velocity.x) < 1e-4
assert abs(gpu.particles()[1].velocity.x - cpu.particles()[1].velocity.x) < 1e-4
gpu.set_particle_transfer_color(1, 0)
gpu.step_gpu_fixed(0.001, 1)
assert [state.transfer_color for state in gpu.particles()] == [0, 0]
assert abs(gpu.particles()[0].velocity.x) < 0.1
assert abs(gpu.particles()[1].velocity.x) < 0.1

mixed = world()
mixed.step(0.001)
assert abs(mixed.particles()[0].velocity.x) < 1e-8
assert abs(mixed.particles()[1].velocity.x) < 1e-8

print("MPM transfer colors passed: CPU, WebGPU resident, and shared-grid control")
