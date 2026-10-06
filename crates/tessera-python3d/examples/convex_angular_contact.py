"""Check prescribed convex angular contact and impulses in distinct environments."""
from tessera3d import ArticulatedBatch, BatchModel, ConvexMeshPart, LinkPose, Material, ModelFormat, Quaternion, SceneBodyInput, SceneColliderInput, SceneShape, Vec3


def vec(x=0.0, y=0.0, z=0.0):
    return Vec3(x=x, y=y, z=z)


def pose(x=0.0, z=0.0):
    return LinkPose(position=vec(x=x, z=z), orientation=Quaternion(x=0, y=0, z=0, w=1))


XML = '<mujoco><option gravity="0 0 0"/><worldbody><body name="root"><inertial mass="1" diaginertia="1 1 1"/></body></worldbody></mujoco>'
batch = ArticulatedBatch([BatchModel(format=ModelFormat.MJCF, xml=XML, floating_base=False, meshes=[]) for _ in range(2)])
directions = [vec(x=1), vec(y=1), vec(z=1)]
hull = ConvexMeshPart(vertices=[vec(x,y,z) for x in [-.5,.5] for y in [-.5,.5] for z in [-.5,.5]], face_normals=directions+[vec(x=-1),vec(y=-1),vec(z=-1)], edge_directions=directions)
for environment, (origin, omega) in enumerate([(-.2, -.4), (.4, .4)]):
    body = SceneBodyInput(pose=pose(x=origin,z=-.5), linear_velocity=vec(), angular_velocity=vec(), mass=0.0,
        inertia_tensor=[1.,0.,0.,0.,1.,0.,0.,0.,1.], force=vec(),
        colliders=[SceneColliderInput(frame=pose(x=-origin),shape=SceneShape.CONVEX(geometry=hull))])
    index = batch.add_scene_body(environment,body)
    batch.set_scene_body_kinematic_motion(environment,index,vec(),vec(y=omega))
    batch.add_scene_sphere(environment,pose(x=.3,z=.5),.5,1.)
    for body_index in [0,1]:
        previous = batch.scene_collider_material(environment,body_index,0)
        batch.set_scene_collider_material(environment,body_index,0,Material(friction=0.,restitution=0.,friction_rule=previous.friction_rule,restitution_rule=previous.restitution_rule))
reports = batch.step_gpu_resident_with_contacts(.001,[[],[]],1)
for environment, expected in enumerate([.2,.04]):
    state = batch.scene_body_state(environment,1)
    assert abs(state.linear_velocity.z-expected)<5e-4, state
    assert abs(state.angular_velocity.x)+abs(state.angular_velocity.y)+abs(state.angular_velocity.z)<5e-4, state
    samples = [sample for sample in reports[environment][1].samples if sample.impulse.z>1e-6]
    assert len(samples)==1, samples
    assert abs(samples[0].position.x-.3)<1e-5, samples
    assert abs(samples[0].impulse.z-expected)<5e-4, samples
    assert samples[0].normal.z>.99999, samples
print("prescribed convex angular contact and impulses Python API passed")
