"""Check prescribed point/fixed anchors through the generated Python APIs."""
from tessera3d import ArticulatedBatch, ArticulatedWorld, BatchModel, LinkPose, LinkSceneFixedConstraint, LinkScenePointConstraint, ModelFormat, Quaternion, SceneBodyInput, Vec3


def vec(x=0., y=0., z=0.):
    return Vec3(x=x,y=y,z=z)


def pose(z=0.):
    return LinkPose(position=vec(z=z),orientation=Quaternion(x=0,y=0,z=0,w=1))


def body():
    return SceneBodyInput(pose=pose(z=10.),linear_velocity=vec(),angular_velocity=vec(),mass=0.,
        inertia_tensor=[1.,0.,0.,0.,1.,0.,0.,0.,1.],force=vec(),colliders=[])


XML = '<mujoco><option gravity="0 0 0"/><worldbody><body name="root" pos="0 0 10"><inertial mass="1" diaginertia="1 1 1"/><body name="slider"><joint type="slide" axis="1 0 0"/><inertial mass="1" diaginertia="1 1 1"/></body></body></worldbody></mujoco>'
batch = ArticulatedBatch([BatchModel(format=ModelFormat.MJCF,xml=XML,floating_base=False,meshes=[]) for _ in range(3)])
for environment, speed in enumerate([.2,0.,-.1]):
    index = batch.add_scene_body(environment,body())
    batch.set_scene_body_kinematic_motion(environment,index,vec(x=speed),vec())
    batch.set_link_scene_point_constraints(environment,[LinkScenePointConstraint(link=1,link_point=vec(),body=index,body_point=vec())])
static_indices = []
for environment in range(3):
    initial = pose(z=10.)
    initial.position.x = 100.
    static_indices.append(batch.add_scene_sphere(environment,initial,.5,0.))
for iteration in range(2):
    batch.step_gpu_resident(.001,[[0.],[0.],[0.]],50)
    if iteration == 0:
        updated = pose(z=10.)
        updated.position.x = 101.
        batch.set_scene_body_pose(0,static_indices[0],updated)
        batch.update_gpu_resident_static_scene_poses()
for environment, speed in enumerate([.2,0.,-.1]):
    assert abs(batch.positions(environment)[0]-.1*speed)<1e-5
    assert abs(batch.velocities(environment)[0]-speed)<1e-5
    assert abs(batch.scene_body_state(environment,0).pose.position.x-.1*speed)<1e-5
batch.set_scene_body_kinematic_motion(1,0,vec(x=.3),vec())
try:
    batch.step_gpu_resident(.001,[[0.],[0.],[0.]],1)
except Exception:
    pass
else:
    raise AssertionError("stale anchor motion was accepted")
for environment, speed in enumerate([.2,0.,-.1]):
    assert abs(batch.positions(environment)[0]-.1*speed)<1e-5
print("resident moving point anchor Python batch passed")

ROTATING_XML = XML.replace('type="slide" axis="1 0 0"','type="hinge" axis="0 0 1"')
for fixed in [True,False]:
    world = ArticulatedWorld.from_mjcf(ROTATING_XML)
    index = world.add_scene_body(body())
    world.set_scene_body_kinematic_motion(index,vec(),vec(z=.2))
    if fixed:
        world.set_link_scene_fixed_constraints([LinkSceneFixedConstraint(link=1,link_frame=pose(),body=index,body_frame=pose())])
    else:
        world.set_link_scene_point_constraints([LinkScenePointConstraint(link=1,link_point=vec(x=1),body=index,body_point=vec(x=1))])
    initial = pose(z=10.)
    initial.position.x = 100.
    stationary = world.add_scene_sphere(initial,.5,0.)
    for iteration in range(2):
        world.step_gpu_resident(.001,[0.],50)
        if iteration == 0:
            updated = pose(z=10.)
            updated.position.x = 101.
            world.set_scene_body_pose(stationary,updated)
            world.update_gpu_resident_static_scene_poses()
    assert abs(world.positions()[0]-.02)<1e-5
    assert abs(world.velocities()[0]-.2)<5e-5
    assert abs(world.scene_body_state(index).pose.orientation.z-.01)<1e-5
    world.set_scene_body_kinematic_motion(index,None,None)
    world.reset_gpu_resident()
    world.step_gpu_resident(.001,[0.],10)
    assert abs(world.positions()[0]-.02)<1e-5
    assert abs(world.velocities()[0])<5e-5
print("resident fixed and offset point anchor Python World passed")
