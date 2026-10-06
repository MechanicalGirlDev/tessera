"""Validate generated Python APIs for reusable resident articulated dynamics."""

from tessera3d import ArticulatedBatch, ArticulatedWorld, BatchModel, ConvexMeshPart, LinkPose, Material, ModelFormat, Quaternion, SceneBodyInput, SceneColliderInput, SceneShape, Vec3
from math import cos, sin, pi

XML = '''<mujoco><option gravity="0 0 0"/><worldbody>
<body name="root" pos="0 0 0.5"><inertial mass="1" diaginertia="1 1 1"/>
<body name="slider"><joint name="z" type="slide" axis="0 0 1"/>
<inertial mass="1" diaginertia="1 1 1"/><geom type="sphere" size="0.5"/>
</body></body></worldbody></mujoco>'''


def vec(x=0.0, y=0.0, z=0.0):
    return Vec3(x=x, y=y, z=z)


world = ArticulatedWorld.from_mjcf(XML)
pose = LinkPose(position=vec(z=-0.5), orientation=Quaternion(x=0, y=0, z=0, w=1))
body = world.add_scene_sphere(pose, 0.5, 0.0)
world.set_scene_body_kinematic_motion(body, vec(z=0.2), vec(y=0.2))

for timestep, steps in [(0.0, 1), (float('nan'), 1), (0.001, 0)]:
    try:
        world.step_gpu_resident(timestep, [0.0], steps)
    except Exception:
        pass
    else:
        raise AssertionError("invalid resident step was accepted")

world.step_gpu_resident(0.001, [0.0], 5)
world.step_gpu_resident(0.001, [0.0], 5)
state = world.scene_body_state(body)
assert state.kinematic
assert abs(state.pose.position.z + 0.498) < 1e-6
assert abs(state.pose.orientation.y - 0.001) < 1e-6
assert abs(state.linear_velocity.z - 0.2) < 1e-12

original_material = world.scene_collider_material(body, 0)
world.set_scene_collider_material(body, 0, Material(
    friction=original_material.friction + 0.1,
    restitution=original_material.restitution,
    friction_rule=original_material.friction_rule,
    restitution_rule=original_material.restitution_rule,
))
try:
    world.step_gpu_resident(0.001, [0.0], 1)
except Exception as error:
    assert "configuration changed" in str(error), str(error)
else:
    raise AssertionError("stale resident material was accepted")
assert world.scene_body_state(body).pose.position.z == state.pose.position.z
world.set_scene_collider_material(body, 0, original_material)

world.set_scene_body_kinematic_motion(body, None, None)
try:
    world.step_gpu_resident(0.001, [0.0], 1)
except Exception:
    pass
else:
    raise AssertionError("stale prescribed motion was accepted")
assert world.scene_body_state(body).pose.position.z == state.pose.position.z

world.reset_gpu_resident()
world.step_gpu_resident(0.001, [0.0], 1)
stopped = world.scene_body_state(body)
assert not stopped.kinematic
assert stopped.pose.position.z == state.pose.position.z
print("resident articulated Python API passed")

XML_TWO = '''<mujoco><option gravity="0 0 0"/><worldbody>
<body name="root" pos="0 0 0.5"><inertial mass="1" diaginertia="1 1 1"/>
<body name="slider"><joint name="z" type="slide" axis="0 0 1"/>
<inertial mass="1" diaginertia="1 1 1"/>
<body name="lateral"><joint name="x" type="slide" axis="1 0 0"/>
<inertial mass="1" diaginertia="1 1 1"/><geom type="sphere" size="0.5"/>
</body></body></body></worldbody></mujoco>'''

batch = ArticulatedBatch([
    BatchModel(format=ModelFormat.MJCF, xml=xml, floating_base=False, meshes=[])
    for xml in [XML, XML_TWO]
])
for environment in [0, 1]:
    batch.add_scene_sphere(environment, pose, 0.5, 0.0)
batch.set_scene_body_kinematic_motion(1, 0, vec(z=0.2), vec(y=0.2))
try:
    batch.step_gpu_resident(0.001, [[0.0], [0.0]], 1)
except Exception:
    pass
else:
    raise AssertionError("invalid mixed-DOF effort input was accepted")
assert batch.scene_body_state(1, 0).pose.position.z == -0.5

efforts = [[0.0], [0.0, 0.0]]
batch.step_gpu_resident(0.001, efforts, 5)
batch.step_gpu_resident(0.001, efforts, 5)
moving = batch.scene_body_state(1, 0)
assert abs(moving.pose.position.z + 0.498) < 1e-6
assert abs(moving.pose.orientation.y - 0.001) < 1e-6
assert batch.scene_body_state(0, 0).pose.position.z == -0.5

original_material = batch.scene_collider_material(1, 0, 0)
batch.set_scene_collider_material(1, 0, 0, Material(
    friction=original_material.friction + 0.1,
    restitution=original_material.restitution,
    friction_rule=original_material.friction_rule,
    restitution_rule=original_material.restitution_rule,
))
try:
    batch.step_gpu_resident(0.001, efforts, 1)
except Exception as error:
    assert "configuration changed" in str(error), str(error)
else:
    raise AssertionError("stale resident material was accepted")
assert batch.scene_body_state(1, 0).pose.position.z == moving.pose.position.z
batch.set_scene_collider_material(1, 0, 0, original_material)

batch.set_scene_body_kinematic_motion(1, 0, None, None)
try:
    batch.step_gpu_resident(0.001, efforts, 1)
except Exception:
    pass
else:
    raise AssertionError("stale batch motion was accepted")
batch.reset_gpu_resident()
batch.step_gpu_resident(0.001, efforts, 1)
assert batch.scene_body_state(1, 0).pose.position.z == moving.pose.position.z
assert batch.scene_body_state(0, 0).pose.position.z == -0.5
print("mixed-DOF resident articulated batch Python API passed")

# Positive-mass bodies resting on the ground exercise automatic scene sleep.
resting = LinkPose(position=vec(x=10.0, z=0.5), orientation=Quaternion(x=0, y=0, z=0, w=1))
sleeper = ArticulatedWorld.from_mjcf(XML.replace('gravity="0 0 0"', 'gravity="0 0 -9.81"'))
body = sleeper.add_scene_sphere(resting, 0.5, 1.0)
try:
    sleeper.enable_gpu_resident_scene_sleep()
except Exception as error:
    assert "initialize" in str(error), str(error)
else:
    raise AssertionError("uninitialized sleep configuration was accepted")
sleeper.step_gpu_resident(0.001, [0.0], 1)
sleeper.enable_gpu_resident_scene_sleep()
sleeper.step_gpu_resident(0.001, [0.0], 600)
assert sleeper.scene_body_is_sleeping(body)
sleeper.set_scene_body_force(body, vec(x=100.0))
sleeper.step_gpu_resident(0.001, [0.0], 1)
assert not sleeper.scene_body_is_sleeping(body)
assert sleeper.scene_body_state(body).linear_velocity.x > 0.0

sleep_batch = ArticulatedBatch([
    BatchModel(format=ModelFormat.MJCF, xml=xml.replace('gravity="0 0 0"', 'gravity="0 0 -9.81"'), floating_base=False, meshes=[])
    for xml in [XML, XML_TWO]
])
for environment in [0, 1]:
    sleep_batch.add_scene_sphere(environment, resting, 0.5, 1.0)
sleep_batch.step_gpu_resident(0.001, efforts, 1)
sleep_batch.enable_gpu_resident_scene_sleep()
sleep_batch.step_gpu_resident(0.001, efforts, 600)
assert all(sleep_batch.scene_body_is_sleeping(environment, 0) for environment in [0, 1])
sleep_batch.set_scene_body_force(1, 0, vec(x=100.0))
sleep_batch.step_gpu_resident(0.001, efforts, 1)
assert sleep_batch.scene_body_is_sleeping(0, 0)
assert not sleep_batch.scene_body_is_sleeping(1, 0)
assert sleep_batch.scene_body_state(1, 0).linear_velocity.x > 0.0
print("resident articulated scene sleep Python APIs passed")

# Explicit static pose uploads preserve the resident generalized state.
static_pose = LinkPose(position=vec(z=-2.0), orientation=Quaternion(x=0, y=0, z=0, w=1))
raised_pose = LinkPose(position=vec(z=0.1), orientation=Quaternion(x=0, y=0, z=0, w=1))
static_world = ArticulatedWorld.from_mjcf(XML)
body = static_world.add_scene_sphere(static_pose, 0.5, 0.0)
static_world.step_gpu_resident(0.001, [0.0], 1)
static_world.set_scene_body_pose(body, raised_pose)
try:
    static_world.step_gpu_resident(0.001, [0.0], 1)
except Exception as error:
    assert "configuration changed" in str(error), str(error)
else:
    raise AssertionError("unsynchronized static pose was accepted")
static_world.update_gpu_resident_static_scene_poses()
static_world.step_gpu_resident(0.001, [0.0], 1)
assert static_world.velocities()[0] > 0.0

static_batch = ArticulatedBatch([
    BatchModel(format=ModelFormat.MJCF, xml=xml, floating_base=False, meshes=[])
    for xml in [XML, XML_TWO]
])
for environment in [0, 1]:
    static_batch.add_scene_sphere(environment, static_pose, 0.5, 0.0)
static_batch.step_gpu_resident(0.001, efforts, 1)
unchanged = static_batch.positions(0)
static_batch.set_scene_body_pose(1, 0, raised_pose)
static_batch.update_gpu_resident_static_scene_poses()
static_batch.step_gpu_resident(0.001, efforts, 1)
assert static_batch.velocities(1)[0] > 0.0
assert static_batch.positions(0) == unchanged
print("resident static scene pose Python APIs passed")

# A robot without collision geometry generates no external contact pairs.
XML_NO_CONTACT = XML.replace('<geom type="sphere" size="0.5"/>', '')
world = ArticulatedWorld.from_mjcf(XML_NO_CONTACT)
body = world.add_scene_sphere(pose, 0.5, 0.0)
world.set_scene_body_kinematic_motion(body, vec(z=0.2), vec(y=0.2))
world.step_gpu_resident(0.001, [0.0], 100)
world.step_gpu_resident(0.001, [0.0], 10)
state = world.scene_body_state(body)
assert abs(state.pose.position.z - (pose.position.z + 0.022)) < 1e-5
assert abs(state.pose.orientation.y - 0.011) < 1e-5

batch = ArticulatedBatch([
    BatchModel(format=ModelFormat.MJCF, xml=xml, floating_base=False, meshes=[])
    for xml in [XML_NO_CONTACT, XML_TWO]
])
for environment in [0, 1]:
    batch.add_scene_sphere(environment, pose, 0.5, 0.0)
    batch.set_scene_body_kinematic_motion(environment, 0, vec(z=0.2), vec(y=0.2))
batch.step_gpu_resident(0.001, [[0.0], [0.0, 0.0]], 100)
batch.step_gpu_resident(0.001, [[0.0], [0.0, 0.0]], 10)
for environment in [0, 1]:
    state = batch.scene_body_state(environment, 0)
    assert abs(state.pose.position.z - (pose.position.z + 0.022)) < 1e-5
    assert abs(state.pose.orientation.y - 0.011) < 1e-5
print("resident unpaired kinematic Python API passed")


def box_input():
    identity = LinkPose(position=vec(), orientation=Quaternion(x=0, y=0, z=0, w=1))
    return SceneBodyInput(
        pose=LinkPose(position=vec(z=-0.5), orientation=Quaternion(x=0, y=0, z=0, w=1)),
        linear_velocity=vec(), angular_velocity=vec(), mass=0.0,
        inertia_tensor=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        force=vec(), colliders=[SceneColliderInput(frame=identity, shape=SceneShape.BOX(half_extents=vec(x=0.5, y=0.5, z=0.5)))],
    )


world = ArticulatedWorld.from_mjcf(XML)
body = world.add_scene_body(box_input())
world.set_scene_body_kinematic_motion(body, vec(z=0.2), vec(y=0.2))
world.step_gpu_resident(0.001, [0.0], 100)
world.step_gpu_resident(0.001, [0.0], 10)
state = world.scene_body_state(body)
assert abs(state.pose.position.z + 0.478) < 1e-5
assert abs(state.pose.orientation.y - 0.011) < 1e-5
assert world.velocities()[0] > 0.1

batch = ArticulatedBatch([
    BatchModel(format=ModelFormat.MJCF, xml=xml, floating_base=False, meshes=[])
    for xml in [XML, XML_TWO, XML_NO_CONTACT]
])
for environment in [0, 1, 2]:
    body = batch.add_scene_body(environment, box_input())
    batch.set_scene_body_kinematic_motion(environment, body, vec(z=0.2), vec(y=0.2))
batch.step_gpu_resident(0.001, [[0.0], [0.0, 0.0], [0.0]], 100)
batch.step_gpu_resident(0.001, [[0.0], [0.0, 0.0], [0.0]], 10)
for environment in [0, 1, 2]:
    state = batch.scene_body_state(environment, 0)
    assert abs(state.pose.position.z + 0.478) < 1e-5
    assert abs(state.pose.orientation.y - 0.011) < 1e-5
assert batch.velocities(0)[0] > 0.1
assert batch.velocities(1)[0] > 0.1
assert batch.velocities(2)[0] == 0.0
print("resident kinematic box Python APIs passed")


def convex_input():
    body = box_input()
    body.mass = 1.0
    body.pose.position.z = 0.5
    directions = [vec(x=1), vec(y=1), vec(z=1)]
    geometry = ConvexMeshPart(
        vertices=[vec(x, y, z) for x in [-0.5, 0.5] for y in [-0.5, 0.5] for z in [-0.5, 0.5]],
        face_normals=directions + [vec(x=-1), vec(y=-1), vec(z=-1)],
        edge_directions=directions,
    )
    body.colliders[0].shape = SceneShape.CONVEX(geometry=geometry)
    return body


# Convex scene bodies share the articulated solve even without robot colliders.
world = ArticulatedWorld.from_mjcf(XML_NO_CONTACT)
box = world.add_scene_body(box_input())
convex = world.add_scene_body(convex_input())
world.set_scene_body_kinematic_motion(box, vec(z=0.2), vec())
world.step_gpu_resident(0.001, [0.0], 100)
assert abs(world.scene_body_state(convex).linear_velocity.z - 0.2) < 5e-4
assert abs(world.scene_body_state(box).pose.position.z + 0.48) < 1e-5
static = world.add_scene_sphere(LinkPose(position=vec(x=10), orientation=Quaternion(x=0, y=0, z=0, w=1)), 0.5, 0.0)
# Adding geometry requires a new resident session; subsequent pose updates do not.
world.reset_gpu_resident()
world.step_gpu_resident(0.001, [0.0], 1)
world.set_scene_body_pose(static, LinkPose(position=vec(x=11), orientation=Quaternion(x=0, y=0, z=0, w=1)))
world.update_gpu_resident_static_scene_poses()
world.step_gpu_resident(0.001, [0.0], 9)
assert abs(world.scene_body_state(convex).linear_velocity.z - 0.2) < 5e-4
assert abs(world.scene_body_state(box).pose.position.z + 0.478) < 1e-5
print("resident convex scene Python API passed")


# Compound bodies use sphere-orbit host synchronization while moving box rows too.
compound = box_input()
compound.colliders.append(SceneColliderInput(
    frame=LinkPose(position=vec(x=3), orientation=Quaternion(x=0, y=0, z=0, w=1)),
    shape=SceneShape.SPHERE(radius=0.25),
))
world = ArticulatedWorld.from_mjcf(XML)
body = world.add_scene_body(compound)
world.set_scene_body_kinematic_motion(body, vec(z=0.2), vec(y=0.2))
world.step_gpu_resident(0.001, [0.0], 100)
world.step_gpu_resident(0.001, [0.0], 10)
state = world.scene_body_state(body)
assert state.collider_count == 2
assert abs(state.pose.position.z + 0.478) < 1e-5
assert abs(state.pose.orientation.y - 0.011) < 1e-5
assert world.velocities()[0] > 0.1
print("resident compound prescribed body Python API passed")

batch = ArticulatedBatch([
    BatchModel(format=ModelFormat.MJCF, xml=xml, floating_base=False, meshes=[])
    for xml in [XML_NO_CONTACT, XML, XML_TWO]
])
for environment in [0, 1, 2]:
    batch.add_scene_body(environment, box_input() if environment == 0 else compound)
    batch.set_scene_body_kinematic_motion(environment, 0, vec(z=0.2), vec())
convex = batch.add_scene_body(0, convex_input())
efforts = [[0.0], [0.0], [0.0, 0.0]]
batch.step_gpu_resident(0.001, efforts, 100)
batch.step_gpu_resident(0.001, efforts, 10)
assert abs(batch.scene_body_state(0, convex).linear_velocity.z - 0.2) < 5e-4
for environment in [0, 1, 2]:
    assert abs(batch.scene_body_state(environment, 0).pose.position.z + 0.478) < 1e-5
for environment in [1, 2]:
    assert abs(batch.velocities(environment)[0] - 0.2) < 5e-4
print("resident convex and compound mixed batch Python API passed")


def capsule_input():
    body = box_input()
    body.colliders[0] = SceneColliderInput(
        frame=LinkPose(position=vec(), orientation=Quaternion(x=0, y=2**-0.5, z=0, w=2**-0.5)),
        shape=SceneShape.CAPSULE(half_height=0.2, radius=0.5),
    )
    return body


world = ArticulatedWorld.from_mjcf(XML)
body = world.add_scene_body(capsule_input())
world.set_scene_body_kinematic_motion(body, vec(z=0.2), vec(y=0.2))
world.step_gpu_resident(0.001, [0.0], 100)
world.step_gpu_resident(0.001, [0.0], 10)
state = world.scene_body_state(body)
assert abs(state.pose.position.z + 0.478) < 1e-5
assert abs(state.pose.orientation.y - 0.011) < 1e-5
assert world.velocities()[0] > 0.1

batch = ArticulatedBatch([
    BatchModel(format=ModelFormat.MJCF, xml=xml, floating_base=False, meshes=[])
    for xml in [XML_NO_CONTACT, XML, XML_TWO]
])
for environment in [0, 1, 2]:
    capsule = capsule_input()
    if environment == 0:
        # Center the cap below the free convex body to isolate translation.
        capsule.colliders[0].frame = LinkPose(position=vec(z=-0.2), orientation=Quaternion(x=0, y=0, z=0, w=1))
    batch.add_scene_body(environment, capsule)
    batch.set_scene_body_kinematic_motion(environment, 0, vec(z=0.2), vec())
convex = batch.add_scene_body(0, convex_input())
static = batch.add_scene_sphere(0, LinkPose(position=vec(x=10), orientation=Quaternion(x=0, y=0, z=0, w=1)), 0.5, 0.0)
batch.step_gpu_resident(0.001, efforts, 100)
saved_capsule_pose = batch.scene_body_state(0, 0).pose
batch.set_scene_body_pose(0, 0, LinkPose(position=vec(z=saved_capsule_pose.position.z + 0.1), orientation=saved_capsule_pose.orientation))
try:
    batch.update_gpu_resident_static_scene_poses()
except Exception as error:
    assert "configuration changed" in str(error), str(error)
else:
    raise AssertionError("an externally edited prescribed capsule pose was accepted")
batch.set_scene_body_pose(0, 0, saved_capsule_pose)
batch.set_scene_body_pose(0, static, LinkPose(position=vec(x=11), orientation=Quaternion(x=0, y=0, z=0, w=1)))
batch.update_gpu_resident_static_scene_poses()
batch.step_gpu_resident(0.001, efforts, 10)
assert abs(batch.scene_body_state(0, convex).linear_velocity.z - 0.2) < 5e-4
for environment in [0, 1, 2]:
    assert abs(batch.scene_body_state(environment, 0).pose.position.z + 0.478) < 1e-5
for environment in [1, 2]:
    assert abs(batch.velocities(environment)[0] - 0.2) < 5e-4
print("resident kinematic capsule single and mixed batch Python APIs passed")

# Face contacts balance both normal impulses, including clipped and rotated faces.
for length, yaw in [(0.2, 0.0), (2.0, 0.0), (2.0, pi / 4)]:
    world = ArticulatedWorld.from_mjcf(XML_NO_CONTACT)
    capsule = capsule_input()
    capsule.colliders[0].shape = SceneShape.CAPSULE(half_height=length, radius=0.5)
    body = world.add_scene_body(capsule)
    hull = convex_input()
    hull.pose.orientation = Quaternion(x=0, y=0, z=sin(yaw / 2), w=cos(yaw / 2))
    convex = world.add_scene_body(hull)
    world.set_scene_body_kinematic_motion(body, vec(z=0.2), vec())
    world.step_gpu_resident(0.001, [0.0], 100)
    state = world.scene_body_state(convex)
    assert abs(state.linear_velocity.z - 0.2) < 5e-4
    angular_speed = (state.angular_velocity.x**2 + state.angular_velocity.y**2 + state.angular_velocity.z**2)**0.5
    assert angular_speed < 5e-4, (length, yaw, state.angular_velocity)
print("resident clipped capsule convex manifold Python API passed")


# Dynamic capsule contacts conserve momentum and static hull uploads update both rows.
batch = ArticulatedBatch([
    BatchModel(format=ModelFormat.MJCF, xml=XML_NO_CONTACT, floating_base=False, meshes=[])
    for _ in range(2)
])
for environment, hull_mass in enumerate([0.0, 1.0]):
    capsule = capsule_input()
    capsule.mass = 1.0
    capsule.pose.position.z = 9.5
    capsule.linear_velocity = vec(z=0.2)
    capsule.colliders[0].shape = SceneShape.CAPSULE(half_height=2.0, radius=0.5)
    batch.add_scene_body(environment, capsule)
    hull = convex_input()
    hull.mass = hull_mass
    hull.pose.position.z = 10.5
    batch.add_scene_body(environment, hull)
batch.step_gpu_resident(0.001, [[0.0], [0.0]], 100)
for environment, expected in enumerate([0.0, 0.1]):
    for body in [0, 1]:
        state = batch.scene_body_state(environment, body)
        assert abs(state.linear_velocity.z - expected) < 5e-4
        assert sum(v*v for v in [state.angular_velocity.x, state.angular_velocity.y, state.angular_velocity.z]) < (5e-4)**2
pose = batch.scene_body_state(0, 1).pose
pose.position.z -= 0.001
batch.set_scene_body_pose(0, 1, pose)
batch.update_gpu_resident_static_scene_poses()
batch.step_gpu_resident(0.001, [[0.0], [0.0]], 1)
state = batch.scene_body_state(0, 0)
assert state.linear_velocity.z < -0.1
assert sum(v*v for v in [state.angular_velocity.x, state.angular_velocity.y, state.angular_velocity.z]) < (5e-4)**2
for body in [0, 1]:
    assert abs(batch.scene_body_state(1, body).linear_velocity.z - 0.1) < 5e-4
print("resident dynamic capsule convex momentum and static pose Python API passed")


# A slightly tilted hull still clips its capsule manifold against the actual face.
world = ArticulatedWorld.from_mjcf(XML_NO_CONTACT)
capsule = capsule_input()
capsule.pose.position.z = 9.55
capsule.colliders[0].shape = SceneShape.CAPSULE(half_height=2.0, radius=0.5)
world.add_scene_body(capsule)
hull = convex_input()
hull.pose.position.z = 10.5
hull.pose.orientation = Quaternion(x=0, y=sin(0.005), z=0, w=cos(0.005))
world.add_scene_body(hull)
world.step_gpu_resident(0.001, [0.0], 1)
state = world.scene_body_state(1)
assert abs(state.linear_velocity.z - 2 * cos(0.01)) < 5e-4, state.linear_velocity
assert abs(state.linear_velocity.x - 2 * sin(0.01)) < 5e-4, state.linear_velocity
assert sum(v*v for v in [state.angular_velocity.x, state.angular_velocity.y, state.angular_velocity.z]) < (5e-4)**2, state.angular_velocity
print("resident tilted capsule convex Python API passed")


# Prescribed cylinders and cones use the same reusable World and batch APIs.
for shape in [SceneShape.CYLINDER, SceneShape.CONE]:
    def axial_input():
        body = box_input()
        body.colliders[0] = SceneColliderInput(
            frame=LinkPose(position=vec(), orientation=Quaternion(x=0, y=0, z=0, w=1)),
            shape=shape(half_height=0.5, radius=0.5),
        )
        return body

    world = ArticulatedWorld.from_mjcf(XML)
    body = world.add_scene_body(axial_input())
    world.set_scene_body_kinematic_motion(body, vec(z=0.2), vec())
    world.step_gpu_resident(0.001, [0.0], 100)
    assert abs(world.velocities()[0] - 0.2) < 5e-4
    assert abs(world.scene_body_state(body).pose.position.z + 0.48) < 1e-5
    world.set_scene_body_kinematic_motion(body, None, None)
    try:
        world.step_gpu_resident(0.001, [0.0], 1)
    except Exception:
        pass
    else:
        raise AssertionError("stale prescribed axial motion was accepted")
    world.reset_gpu_resident()
    world.step_gpu_resident(0.001, [0.0], 10)
    assert abs(world.scene_body_state(body).pose.position.z + 0.48) < 1e-5

    batch = ArticulatedBatch([
        BatchModel(format=ModelFormat.MJCF, xml=xml, floating_base=False, meshes=[])
        for xml in [XML, XML_TWO, XML_NO_CONTACT]
    ])
    for environment in range(3):
        batch.add_scene_body(environment, axial_input())
        batch.set_scene_body_kinematic_motion(
            environment, 0, vec(z=0.0 if environment == 1 else 0.2),
            vec(y=0.2 if environment == 2 else 0.0),
        )
    static = batch.add_scene_sphere(0, LinkPose(position=vec(x=10), orientation=Quaternion(x=0, y=0, z=0, w=1)), 0.5, 0.0)
    batch.step_gpu_resident(0.001, [[0.0], [0.0, 0.0], [0.0]], 100)
    batch.set_scene_body_pose(0, static, LinkPose(position=vec(x=11), orientation=Quaternion(x=0, y=0, z=0, w=1)))
    batch.update_gpu_resident_static_scene_poses()
    batch.step_gpu_resident(0.001, [[0.0], [0.0, 0.0], [0.0]], 10)
    assert abs(batch.velocities(0)[0] - 0.2) < 5e-4
    assert batch.velocities(1)[0] == 0.0
    for environment in [0, 2]:
        assert abs(batch.scene_body_state(environment, 0).pose.position.z + 0.478) < 1e-5
    assert batch.scene_body_state(1, 0).pose.position.z == -0.5
    assert abs(batch.scene_body_state(2, 0).pose.orientation.y - sin(0.011)) < 1e-5
print("resident kinematic cylinder and cone Python APIs passed")
