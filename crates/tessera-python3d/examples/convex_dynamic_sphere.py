"""Verify off-axis convex/sphere impulse and conserved momentum in both orders."""
from tessera3d import ArticulatedBatch, BatchModel, ConvexMeshPart, LinkPose, Material, ModelFormat, Quaternion, SceneBodyInput, SceneColliderInput, SceneShape, Vec3


def vec(x=0.0, y=0.0, z=0.0):
    return Vec3(x=x, y=y, z=z)


def pose(x=0.0, z=0.0):
    return LinkPose(position=vec(x=x,z=z),orientation=Quaternion(x=0,y=0,z=0,w=1))


XML = '<mujoco><option gravity="0 0 0"/><worldbody><body name="root"><inertial mass="1" diaginertia="1 1 1"/></body></worldbody></mujoco>'
batch = ArticulatedBatch([BatchModel(format=ModelFormat.MJCF,xml=XML,floating_base=False,meshes=[]) for _ in range(2)])
axes = [vec(x=1),vec(y=1),vec(z=1)]
hull = ConvexMeshPart(vertices=[vec(x,y,z) for x in [-.5,.5] for y in [-.5,.5] for z in [-.5,.5]],face_normals=axes+[vec(x=-1),vec(y=-1),vec(z=-1)],edge_directions=axes)
indices = []
for environment in range(2):
    origin, mass, inertia = [(0.,1.,1.),(-.2,2.,.7)][environment]
    body = SceneBodyInput(pose=pose(x=origin,z=1.5),linear_velocity=vec(z=.2),angular_velocity=vec(),mass=mass,
        inertia_tensor=[inertia,0.,0.,0.,inertia,0.,0.,0.,inertia],force=vec(),
        colliders=[SceneColliderInput(frame=pose(x=-origin),shape=SceneShape.CONVEX(geometry=hull))])
    if environment == 0:
        convex = batch.add_scene_body(environment,body)
        sphere = batch.add_scene_sphere(environment,pose(x=.3,z=2.5),.5,1.)
    else:
        sphere = batch.add_scene_sphere(environment,pose(x=.3,z=2.5),.5,1.)
        convex = batch.add_scene_body(environment,body)
    indices.append((convex,sphere))
    for index in [convex,sphere]:
        previous = batch.scene_collider_material(environment,index,0)
        batch.set_scene_collider_material(environment,index,0,Material(friction=0.,restitution=0.,friction_rule=previous.friction_rule,restitution_rule=previous.restitution_rule))
reports = batch.step_gpu_resident_with_contacts(.001,[[],[]],1)
# The sphere witness passes through its center; the hull lever uses its body origin.
for environment, (convex,sphere) in enumerate(indices):
    origin, mass, inertia = [(0.,1.,1.),(-.2,2.,.7)][environment]
    lever = .3-origin
    impulse = .2/(1./mass+1.+lever**2/inertia)
    a = batch.scene_body_state(environment,convex)
    b = batch.scene_body_state(environment,sphere)
    assert abs(a.linear_velocity.z-(.2-impulse/mass))<5e-4, a
    assert abs(a.angular_velocity.y-lever*impulse/inertia)<5e-4, a
    assert abs(b.linear_velocity.z-impulse)<5e-4, b
    assert abs(b.angular_velocity.x)+abs(b.angular_velocity.y)+abs(b.angular_velocity.z)<5e-4, b
    assert abs(mass*a.linear_velocity.z+b.linear_velocity.z-mass*.2)<1e-5
    # Angular momentum about the initial world origin, before pose integration.
    assert abs(inertia*a.angular_velocity.y+b.inertia_tensor[4]*b.angular_velocity.y-origin*mass*a.linear_velocity.z-.3*b.linear_velocity.z+origin*mass*.2)<1e-5
    for index, sign in [(convex,-1),(sphere,1)]:
        samples = [sample for sample in reports[environment][index].samples if abs(sample.impulse.z)>1e-6]
        assert len(samples)==1, samples
        assert abs(samples[0].position.x-.3)<1e-5, samples
        assert abs(samples[0].impulse.z-sign*impulse)<5e-4, samples
print("dynamic convex sphere off-axis impulse and momentum Python API passed")
