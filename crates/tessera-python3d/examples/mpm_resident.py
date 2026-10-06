"""Compare persistent fixed GPU steps with the CPU MPM reference."""

from tessera3d import MpmMaterial, MpmParticleInput, MpmWorld, TesseraError, Vec3


def vec(x, y, z):
    return Vec3(x=x, y=y, z=z)


particle = MpmParticleInput(
    position=vec(0.5, 0.5, 0.5), velocity=vec(0.0, 0.0, 0.0),
    radius=0.03, density=1000.0,
    material=MpmMaterial.FLUID(
        bulk_modulus=2000.0, gamma=7.0, viscosity=0.1, tensile_stiffness=0.0,
    ),
    enabled=True, fixed=False, damping=0.0,
)


def world():
    return MpmWorld(
        particles=[particle], gravity=vec(0.0, 0.0, -1.0),
        cell_width=0.1, max_substep=0.001,
        bounds_min=vec(0.0, 0.0, 0.0), bounds_max=vec(1.0, 1.0, 1.0),
    )


cpu, gpu = world(), world()
for step in range(20):
    if step == 4:
        cpu.set_particle_force(0, vec(0.1, 0.0, 0.0))
        gpu.set_particle_force(0, vec(0.1, 0.0, 0.0))
    if step == 8:
        cpu.step(0.0001)
        gpu.step(0.0001)
    if step == 12:
        for simulation in (cpu, gpu):
            chunk = simulation.add_particles([particle])
            assert simulation.remove_chunk(chunk) == 1
    dt = 0.00005 if step >= 16 else 0.0001
    cpu.step(dt)
    gpu.step_gpu_fixed(dt, 1)
    assert cpu.substeps() == gpu.substeps()
    expected, actual = cpu.particles()[0], gpu.particles()[0]
    for axis in ('x', 'y', 'z'):
        assert abs(getattr(expected.position, axis) - getattr(actual.position, axis)) < 1e-5
        assert abs(getattr(expected.velocity, axis) - getattr(actual.velocity, axis)) < 1e-4
before = gpu.substeps()
try:
    gpu.step_gpu_fixed(0.0001, 0)
except TesseraError:
    pass
else:
    raise AssertionError('zero substeps must fail')
assert gpu.substeps() == before
gpu.step_gpu_fixed(0.00005, 1)
print('persistent MPM GPU steps and state edits passed')

# The ordinary transfer path also constructs sparse topology on GPU.
distant = MpmParticleInput(
    position=vec(10.5, 0.5, 0.5), velocity=vec(0.0, 0.0, 0.0),
    radius=0.03, density=1000.0, material=particle.material,
    enabled=True, fixed=False, damping=0.0,
)
simulations = [MpmWorld(
    particles=[particle, distant], gravity=vec(0.0, 0.0, -1.0),
    cell_width=0.1, max_substep=0.001, bounds_min=None, bounds_max=None,
) for _ in range(2)]
cpu, gpu = simulations
for _ in range(3):
    cpu.step(0.0001)
    gpu.step_gpu(0.0001)
    for expected, actual in zip(cpu.particles(), gpu.particles()):
        for axis in ('x', 'y', 'z'):
            assert abs(getattr(expected.position, axis) - getattr(actual.position, axis)) < 1e-5
            assert abs(getattr(expected.velocity, axis) - getattr(actual.velocity, axis)) < 1e-4
print('ordinary sparse MPM GPU transfers passed')

# Unbounded fixed steps retain the session while updating one-shot forces.
for _ in range(8):
    cpu.set_particle_force(0, vec(0.1, 0.0, 0.0))
    gpu.set_particle_force(0, vec(0.1, 0.0, 0.0))
    for _ in range(4):
        cpu.step(0.0001)
    gpu.step_gpu_fixed(0.0001, 4)
    assert cpu.substeps() == gpu.substeps()
    for expected, actual in zip(cpu.particles(), gpu.particles()):
        for axis in ('x', 'y', 'z'):
            assert abs(getattr(expected.position, axis) - getattr(actual.position, axis)) < 1e-4
            assert abs(getattr(expected.velocity, axis) - getattr(actual.velocity, axis)) < 2e-3
print('unbounded fixed MPM GPU steps with force updates passed')
