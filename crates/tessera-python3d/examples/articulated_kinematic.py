"""Validate prescribed scene motion through generated articulated Python APIs."""

from tessera3d import (
    ArticulatedBatch, ArticulatedWorld, BatchModel, LinkPose, ModelFormat,
    Quaternion, Vec3,
)

XML = '''<mujoco><option gravity="0 0 0"/><worldbody>
<body name="root"><inertial mass="1" diaginertia="1 1 1"/>
<body name="x"><joint name="x" type="slide" axis="1 0 0"/>
<inertial mass="1" diaginertia="1 1 1"/></body></body>
</worldbody></mujoco>'''


def vec(x=0.0, y=0.0, z=0.0):
    return Vec3(x=x, y=y, z=z)


def check(batched, device_mass=False):
    model = BatchModel(format=ModelFormat.MJCF, xml=XML, floating_base=False, meshes=[])
    world = ArticulatedBatch([model, model]) if batched else ArticulatedWorld.from_mjcf(XML)
    pose = LinkPose(position=vec(20, 0, 10), orientation=Quaternion(x=0, y=0, z=0, w=1))
    if batched:
        world.add_scene_sphere(0, pose, 0.1, 0.0)
        world.add_scene_sphere(1, pose, 0.1, 0.0)
        world.add_scene_sphere(1, pose, 0.1, 1.0)
    else:
        world.add_scene_sphere(pose, 0.1, 0.0)
        world.add_scene_sphere(pose, 0.1, 1.0)

    def motion(body, linear, angular):
        args = (1, body, linear, angular) if batched else (body, linear, angular)
        world.set_scene_body_kinematic_motion(*args)

    def read():
        return world.scene_body_state(1, 0) if batched else world.scene_body_state(0)

    def step():
        advance = world.step_gpu_device_mass if device_mass else world.step
        advance(0.01, [[0.0], [0.0]] if batched else [0.0])

    for body, linear, angular in [(0, vec(1), None), (0, vec(float('nan')), vec()), (1, vec(1), vec()), (99, None, None)]:
        try:
            motion(body, linear, angular)
        except Exception:
            pass
        else:
            raise AssertionError("invalid motion was accepted")
    assert read().linear_velocity.x == 0
    assert not read().kinematic
    motion(0, vec(), vec())
    assert read().kinematic
    motion(0, vec(0.5), vec(0, 0, 0.25))
    assert read().kinematic
    step()
    assert abs(read().pose.position.x - 20.005) < 1e-9
    assert abs(read().angular_velocity.z - 0.25) < 1e-9
    if batched:
        assert world.scene_body_state(0, 0).pose.position.x == 20
    motion(0, None, None)
    assert not read().kinematic
    step()
    assert abs(read().pose.position.x - 20.005) < 1e-9
    assert read().linear_velocity.x == 0
    print(f"articulated {'batch' if batched else 'world'} ({'device mass' if device_mass else 'CPU'}): motion, stop, validation and isolation passed", flush=True)


check(False)
check(True)
check(False, True)
check(True, True)
