"""Verify prescribed cylinder/cone angular contact through generated Python APIs."""
from tessera3d import ArticulatedWorld, LinkPose, Quaternion, SceneBodyInput, SceneColliderInput, SceneShape, Vec3

XML = '''<mujoco><option gravity="0 0 0"/><worldbody>
<body name="root" pos="0.3 0 0.5"><inertial mass="1" diaginertia="1 1 1"/>
<body name="slider"><joint name="z" type="slide" axis="0 0 1"/>
<inertial mass="1" diaginertia="1 1 1"/><geom type="sphere" size="0.5"/>
</body></body></worldbody></mujoco>'''


def vec(x=0.0, y=0.0, z=0.0):
    return Vec3(x=x, y=y, z=z)


for cone in [False, True]:
    world = ArticulatedWorld.from_mjcf(XML)
    # Body origin, axial center and application point have distinct X coordinates.
    body = SceneBodyInput(
        pose=LinkPose(position=vec(x=-0.2, z=-0.5), orientation=Quaternion(x=0, y=0, z=0, w=1)),
        linear_velocity=vec(), angular_velocity=vec(), mass=0.0,
        inertia_tensor=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], force=vec(),
        colliders=[SceneColliderInput(
            frame=LinkPose(position=vec(x=0.2), orientation=Quaternion(x=1 if cone else 0, y=0, z=0, w=0 if cone else 1)),
            shape=(SceneShape.CONE if cone else SceneShape.CYLINDER)(half_height=0.5, radius=0.5),
        )],
    )
    index = world.add_scene_body(body)
    world.set_scene_body_kinematic_motion(index, vec(), vec(y=-0.4))
    world.step_gpu_resident(0.001, [0.0], 1)
    # omega x (contact - origin) gives +0.2 m/s; center speed alone is +0.08.
    assert abs(world.velocities()[0] - 0.2) < 5e-4, world.velocities()
    state = world.scene_body_state(index)
    assert state.kinematic
    assert abs(state.pose.orientation.y + 0.0002) < 1e-6
    assert abs(state.pose.position.x + 0.2) < 1e-6
    assert abs(state.pose.position.z + 0.5) < 1e-6
print("prescribed axial angular contact Python API passed")
