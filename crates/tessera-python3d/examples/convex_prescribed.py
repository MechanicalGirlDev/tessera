"""Check prescribed convex World and batch motion using generated bindings."""
from tessera3d import ArticulatedBatch, ArticulatedWorld, BatchModel, ConvexMeshPart, LinkPose, Material, ModelFormat, Quaternion, SceneBodyInput, SceneColliderInput, SceneShape, Vec3


def vec(x=0.0, y=0.0, z=0.0):
    return Vec3(x=x, y=y, z=z)


def pose(z=0.0, x=0.0):
    return LinkPose(position=vec(x=x, z=z), orientation=Quaternion(x=0, y=0, z=0, w=1))


XML = '''<mujoco><option gravity="0 0 0"/><worldbody>
<body name="root" pos="0 0 0.5"><inertial mass="1" diaginertia="1 1 1"/>
<body name="slider"><joint name="z" type="slide" axis="0 0 1"/>
<inertial mass="1" diaginertia="1 1 1"/><geom type="sphere" size="0.5"/>
</body></body></worldbody></mujoco>'''
CAPSULE_XML = XML.replace('type="sphere" size="0.5"', 'type="capsule" size="0.5 0.2" euler="0 90 0"')
EMPTY_XML = XML.replace('<geom type="sphere" size="0.5"/>', '')


def convex_body():
    directions = [vec(x=1), vec(y=1), vec(z=1)]
    hull = ConvexMeshPart(
        vertices=[vec(x,y,z) for x in [-0.5,0.5] for y in [-0.5,0.5] for z in [-0.5,0.5]],
        face_normals=directions+[vec(x=-1),vec(y=-1),vec(z=-1)], edge_directions=directions,
    )
    return SceneBodyInput(pose=pose(z=-0.5), linear_velocity=vec(), angular_velocity=vec(),
        mass=0.0, inertia_tensor=[1.,0.,0.,0.,1.,0.,0.,0.,1.], force=vec(),
        colliders=[SceneColliderInput(frame=pose(),shape=SceneShape.CONVEX(geometry=hull))])


for xml in [XML, CAPSULE_XML]:
    world = ArticulatedWorld.from_mjcf(xml)
    body = world.add_scene_body(convex_body())
    world.set_scene_body_kinematic_motion(body, vec(z=0.2), vec())
    world.step_gpu_resident(0.001, [0.0], 100)
    assert abs(world.velocities()[0]-0.2)<5e-4, world.velocities()
    assert abs(world.scene_body_state(body).pose.position.z+0.48)<1e-5
    world.set_scene_body_kinematic_motion(body, None, None)
    try:
        world.step_gpu_resident(0.001,[0.0],1)
    except Exception:
        pass
    else:
        raise AssertionError("stale prescribed convex motion was accepted")
    world.reset_gpu_resident()
    world.step_gpu_resident(0.001,[0.0],10)
    assert abs(world.scene_body_state(body).pose.position.z+0.48)<1e-5

batch = ArticulatedBatch([BatchModel(format=ModelFormat.MJCF,xml=xml,floating_base=False,meshes=[]) for xml in [XML,CAPSULE_XML,EMPTY_XML]])
for environment in range(3):
    batch.add_scene_body(environment,convex_body())
    batch.set_scene_body_kinematic_motion(environment,0,vec(z=0.0 if environment==1 else 0.2),vec(y=0.2 if environment==2 else 0.0))
static = batch.add_scene_sphere(0,pose(x=10),0.5,0.0)
efforts = [[0.0],[0.0],[0.0]]
batch.step_gpu_resident(0.001,efforts,100)
batch.set_scene_body_pose(0,static,pose(x=11))
batch.update_gpu_resident_static_scene_poses()
batch.step_gpu_resident(0.001,efforts,10)
assert abs(batch.velocities(0)[0]-0.2)<5e-4
assert batch.velocities(1)[0]==0.0
for environment in [0,2]:
    assert abs(batch.scene_body_state(environment,0).pose.position.z+0.478)<1e-5
assert abs(batch.scene_body_state(2,0).pose.orientation.y-0.011)<1e-5
print("prescribed convex World and batch Python APIs passed")


# Boxes and convex hulls share polyhedron rows. Exercise both insertion orders.
mixed = ArticulatedBatch([BatchModel(format=ModelFormat.MJCF,xml=EMPTY_XML,floating_base=False,meshes=[]) for _ in range(2)])
owners = []
for environment in range(2):
    ids = {}
    for is_box in ([True,False] if environment==0 else [False,True]):
        body = convex_body()
        body.pose.position.x = -2.0 if is_box else 2.0
        if is_box:
            body.colliders[0].shape = SceneShape.BOX(half_extents=vec(0.5,0.5,0.5))
        index = mixed.add_scene_body(environment,body)
        mixed.set_scene_body_kinematic_motion(environment,index,vec(z=0.2 if is_box else 0.35),vec())
        ids["box" if is_box else "convex"] = index
    for x,key in [(-2.0,"left"),(2.0,"right")]:
        body = convex_body()
        body.mass = 1.0
        body.pose = pose(x=x,z=0.5)
        ids[key] = mixed.add_scene_body(environment,body)
    ids["static"] = mixed.add_scene_sphere(environment,pose(x=10),0.5,0.0)
    owners.append(ids)
mixed.step_gpu_resident(0.001,[[0.0],[0.0]],100)
for environment,ids in enumerate(owners):
    for key,speed in [("left",0.2),("right",0.35)]:
        state = mixed.scene_body_state(environment,ids[key])
        assert abs(state.linear_velocity.z-speed)<5e-4,(environment,key,state)
        assert sum(v*v for v in [state.angular_velocity.x,state.angular_velocity.y,state.angular_velocity.z])<(5e-4)**2,(environment,key,state)
    mixed.set_scene_body_pose(environment,ids["static"],pose(x=11))
mixed.update_gpu_resident_static_scene_poses()
mixed.step_gpu_resident(0.001,[[0.0],[0.0]],10)
for environment,ids in enumerate(owners):
    for key,speed in [("left",0.2),("right",0.35)]:
        assert abs(mixed.scene_body_state(environment,ids[key]).linear_velocity.z-speed)<5e-4
    for key,speed in [("box",0.2),("convex",0.35)]:
        assert abs(mixed.scene_body_state(environment,ids[key]).pose.position.z-(-0.5+0.11*speed))<1e-5
print("mixed prescribed box and convex Python API passed")


# A freely rotating capsule needs the coupled two-point normal solve.
free_capsules = ArticulatedBatch([BatchModel(format=ModelFormat.MJCF,xml=EMPTY_XML,floating_base=False,meshes=[]) for _ in range(2)])
for environment,length in enumerate([0.2,2.0]):
    hull = free_capsules.add_scene_body(environment,convex_body())
    free_capsules.set_scene_body_kinematic_motion(environment,hull,vec(z=0.2),vec())
    capsule = convex_body()
    capsule.mass = 1.0
    capsule.pose = pose(z=0.5)
    capsule.colliders[0] = SceneColliderInput(
        frame=LinkPose(position=vec(),orientation=Quaternion(x=0,y=2**-0.5,z=0,w=2**-0.5)),
        shape=SceneShape.CAPSULE(half_height=length,radius=0.5),
    )
    free_capsules.add_scene_body(environment,capsule)
reports = free_capsules.step_gpu_resident_with_contacts(0.001,[[0.0],[0.0]],1)
for environment in range(2):
    samples = reports[environment][1].samples
    assert len(samples)==2,(environment,samples)
    assert abs(sum(sample.impulse.z for sample in samples)-0.2)<5e-4
    for sample in samples:
        assert abs(sample.impulse.z-0.1)<5e-4,(environment,sample)
        assert sample.normal.z>0.99999
        assert abs(sample.force.z*0.001-sample.impulse.z)<1e-12
for environment in range(2):
    state=free_capsules.scene_body_state(environment,1)
    assert abs(state.linear_velocity.z-0.2)<5e-4,(environment,state)
    assert sum(v*v for v in [state.angular_velocity.x,state.angular_velocity.y,state.angular_velocity.z])<(5e-4)**2,(environment,state)
free_capsules.step_gpu_resident(0.001,[[0.0],[0.0]],99)
for environment in range(2):
    state=free_capsules.scene_body_state(environment,1)
    assert abs(state.linear_velocity.z-0.2)<5e-4,(environment,state)
    assert sum(v*v for v in [state.angular_velocity.x,state.angular_velocity.y,state.angular_velocity.z])<(5e-4)**2,(environment,state)
    assert abs(free_capsules.scene_body_state(environment,0).pose.position.z+0.48)<1e-5
print("prescribed convex freely rotating capsule Python API passed")


# COM is outside the clipped face. Only the nearer endpoint can carry compression.
world = ArticulatedWorld.from_mjcf(EMPTY_XML)
hull = world.add_scene_body(convex_body())
world.set_scene_body_kinematic_motion(hull,vec(z=0.2),vec())
capsule = convex_body()
capsule.mass = 1.0
capsule.pose = pose(x=0.8,z=0.5)
capsule.colliders[0] = SceneColliderInput(
    frame=LinkPose(position=vec(x=-0.8),orientation=Quaternion(x=0,y=2**-0.5,z=0,w=2**-0.5)),
    shape=SceneShape.CAPSULE(half_height=2.0,radius=0.5),
)
body = world.add_scene_body(capsule)
for index in [hull,body]:
    previous = world.scene_collider_material(index,0)
    world.set_scene_collider_material(index,0,Material(friction=0.0,restitution=0.0,friction_rule=previous.friction_rule,restitution_rule=previous.restitution_rule))
reports=world.step_gpu_resident_with_contacts(0.001,[0.0],1)
state=world.scene_body_state(body)
impulse=0.2/(1.0+0.3**2)
samples=reports[body].samples
active=[sample for sample in samples if sample.impulse.z>1e-6]
assert len(active)==1,samples
assert abs(active[0].position.x-0.5)<1e-5,active
assert abs(active[0].impulse.z-impulse)<5e-4,active
assert abs(state.linear_velocity.z-impulse)<5e-4,state
assert abs(state.angular_velocity.y-0.3*impulse)<5e-4,state
assert abs(state.angular_velocity.x)<5e-4 and abs(state.angular_velocity.z)<5e-4,state
assert abs(state.linear_velocity.x)<5e-4 and abs(state.linear_velocity.y)<5e-4,state
print("prescribed convex capsule single active endpoint Python API passed")
