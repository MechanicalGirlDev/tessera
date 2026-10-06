"""Check moving mesh, polyline, and heightfield through resident Python APIs."""
from math import cos, sin, sqrt, tan
from tessera3d import (
    ArticulatedBatch, ArticulatedWorld, BatchModel, GpuSegment, GpuTriangle,
    LinkPose, Material, ModelFormat, Quaternion, SceneBodyInput, SceneColliderInput,
    SceneShape, Vec3,
)


def vec(x=0.0, y=0.0, z=0.0):
    return Vec3(x=x, y=y, z=z)


def pose(z=0.0):
    return LinkPose(position=vec(z=z), orientation=Quaternion(x=0, y=0, z=0, w=1))


XML = """<mujoco><option gravity="0 0 0"/><worldbody>
<body name="root" pos="0 0 1"><inertial mass="1" diaginertia="1 1 1"/>
<body name="slider"><joint type="slide" axis="0 0 1"/>
<inertial mass="1" diaginertia="1 1 1"/>
<geom type="sphere" pos="0 0 0.5" size="0.5" friction="0 0 0"/>
</body></body></worldbody></mujoco>"""


def shapes():
    return [
        SceneShape.TRIANGLE_MESH(
            vertices=[vec(-5, -5), vec(5, -5), vec(0, 5)],
            triangles=[GpuTriangle(a=0, b=1, c=2)],
        ),
        SceneShape.POLYLINE(
            vertices=[vec(-1), vec(1)], segments=[GpuSegment(a=0, b=1)],
        ),
        SceneShape.HEIGHT_FIELD(rows=2, columns=2, heights=[0.0]*4, scale=vec(10, 10, 1)),
    ]


def body(shape):
    return SceneBodyInput(
        pose=pose(1.0), linear_velocity=vec(), angular_velocity=vec(), mass=0.0,
        inertia_tensor=[1., 0., 0., 0., 1., 0., 0., 0., 1.], force=vec(),
        colliders=[SceneColliderInput(frame=pose(), shape=shape)],
    )


batch = ArticulatedBatch([
    BatchModel(format=ModelFormat.MJCF, xml=XML, floating_base=False, meshes=[])
    for _ in range(3)
])
for environment, shape in enumerate(shapes()):
    index = batch.add_scene_body(environment, body(shape))
    batch.set_scene_body_kinematic_motion(environment, index, vec(z=0.2), vec())
static_indices = []
for environment, shape in enumerate(shapes()):
    stationary = body(shape)
    stationary.pose.position.x = 100.0
    static_indices.append(batch.add_scene_body(environment, stationary))
for iteration in range(2):
    batch.step_gpu_resident(0.001, [[0.0]]*3, 50)
    if iteration == 0:
        for environment, index in enumerate(static_indices):
            updated = pose(1.0)
            updated.position.x = 101.0
            batch.set_scene_body_pose(environment, index, updated)
        batch.update_gpu_resident_static_scene_poses()
for environment in range(3):
    assert abs(batch.positions(environment)[0]-0.02) < 1e-5
    assert abs(batch.velocities(environment)[0]-0.2) < 5e-5
    assert abs(batch.scene_body_state(environment, 0).pose.position.z-1.02) < 1e-5
before = [batch.positions(environment) for environment in range(3)]
batch.set_scene_body_kinematic_motion(2, 0, vec(z=0.3), vec())
try:
    batch.step_gpu_resident(0.001, [[0.0]]*3, 1)
except Exception:
    pass
else:
    raise AssertionError("stale indexed body motion was accepted")
assert [batch.positions(environment) for environment in range(3)] == before
print("resident indexed geometry Python batch passed")

for shape in shapes():
    world = ArticulatedWorld.from_mjcf(XML)
    index = world.add_scene_body(body(shape))
    world.set_scene_body_kinematic_motion(index, vec(z=0.2), vec())
    stationary = body(shape)
    stationary.pose.position.x = 100.0
    stationary_index = world.add_scene_body(stationary)
    for iteration in range(2):
        world.step_gpu_resident(0.001, [0.0], 50)
        if iteration == 0:
            updated = pose(1.0)
            updated.position.x = 101.0
            world.set_scene_body_pose(stationary_index, updated)
            world.update_gpu_resident_static_scene_poses()
    assert abs(world.positions()[0]-0.02) < 1e-5
    assert abs(world.velocities()[0]-0.2) < 5e-5
    assert abs(world.scene_body_state(index).pose.position.z-1.02) < 1e-5
print("resident indexed geometry Python World passed")


# Separate body and collider origins exercise the geometry orbit.

ROTATING_XML = XML.replace('pos="0 0 0.5"', 'pos="1 0 0.5"')
rotating = ArticulatedBatch([
    BatchModel(format=ModelFormat.MJCF, xml=ROTATING_XML, floating_base=False, meshes=[])
    for _ in range(3)
])
for environment, shape in enumerate(shapes()):
    initial = body(shape)
    initial.pose.position.x = -0.5
    initial.colliders[0].frame.position.x = 0.5
    index = rotating.add_scene_body(environment, initial)
    previous = rotating.scene_collider_material(environment, index, 0)
    rotating.set_scene_collider_material(environment, index, 0, Material(
        friction=0., restitution=0., friction_rule=previous.friction_rule,
        restitution_rule=previous.restitution_rule,
    ))
    rotating.set_scene_body_kinematic_motion(environment, index, vec(), vec(y=-0.2))
rotating_static = []
for environment, shape in enumerate(shapes()):
    stationary = body(shape)
    stationary.pose.position.x = 100.0
    rotating_static.append(rotating.add_scene_body(environment, stationary))
rotating.step_gpu_resident(0.001, [[0.0]]*3, 1)
for environment in range(3):
    assert abs(rotating.velocities(environment)[0]-0.3) < 1e-5
for steps in [49, 50]:
    rotating.step_gpu_resident(0.001, [[0.0]]*3, steps)
    if steps == 49:
        for environment, index in enumerate(rotating_static):
            updated = pose(1.0)
            updated.position.x = 101.0
            rotating.set_scene_body_pose(environment, index, updated)
        rotating.update_gpu_resident_static_scene_poses()
for environment in range(3):
    if environment == 1:
        horizontal = 1.5*(1-cos(0.02))
        expected = 1.5*sin(0.02)+sqrt(0.25-horizontal*horizontal)-0.5
    else:
        expected = 0.5/cos(0.02)+1.5*tan(0.02)-0.5
    assert abs(rotating.positions(environment)[0]-expected) < 1e-5
    state = rotating.scene_body_state(environment, 0)
    assert abs(state.pose.position.x+0.5) < 1e-6
    assert abs(state.pose.position.z-1.0) < 1e-6
    assert abs(state.pose.orientation.y+sin(0.01)) < 1e-5
print("resident indexed geometry Python rotation and collider offset passed")


# Box/line contact keeps both support endpoints and avoids a sampled face normal.
BOX_XML = XML.replace(
    '<geom type="sphere" pos="0 0 0.5" size="0.5" friction="0 0 0"/>',
    '<geom type="box" pos="1 0 0.25" size="0.25 0.25 0.25" friction="0 0 0"/>',
)
box_shapes = shapes()
box_shapes[1] = SceneShape.POLYLINE(
    vertices=[vec(-5), vec(5)], segments=[GpuSegment(a=0, b=1)],
)
boxes = ArticulatedBatch([
    BatchModel(format=ModelFormat.MJCF, xml=BOX_XML, floating_base=False, meshes=[])
    for _ in range(3)
])
box_static = []
for environment, shape in enumerate(box_shapes):
    initial = body(shape)
    initial.pose.position.x = -0.5
    initial.colliders[0].frame.position.x = 0.5
    index = boxes.add_scene_body(environment, initial)
    previous = boxes.scene_collider_material(environment, index, 0)
    boxes.set_scene_collider_material(environment, index, 0, Material(
        friction=0., restitution=0., friction_rule=previous.friction_rule,
        restitution_rule=previous.restitution_rule,
    ))
    boxes.set_scene_body_kinematic_motion(environment, index, vec(), vec(y=-0.2))
    stationary = body(shape)
    stationary.pose.position.x = 100.0
    box_static.append(boxes.add_scene_body(environment, stationary))
boxes.step_gpu_resident(0.001, [[0.0]]*3, 1)
for environment in range(3):
    assert abs(boxes.velocities(environment)[0]-0.35) < 1e-5
for steps in [49, 50]:
    boxes.step_gpu_resident(0.001, [[0.0]]*3, steps)
    if steps == 49:
        for environment, index in enumerate(box_static):
            updated = pose(1.0)
            updated.position.x = 101.0
            boxes.set_scene_body_pose(environment, index, updated)
        boxes.update_gpu_resident_static_scene_poses()
for environment in range(3):
    expected = 1.75*tan(0.02)
    expected_velocity = 0.35/cos(0.02)**2
    assert abs(boxes.positions(environment)[0]-expected) < 1e-5
    assert abs(boxes.velocities(environment)[0]-expected_velocity) < 5e-5
print("resident indexed geometry Python rotating box contact passed")
