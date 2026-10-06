"""Exercise the generated UniFFI Python binding against the built Rust library."""

from tessera3d import (
    ArticulatedBatch,
    ArticulatedWorld,
    BatchModel,
    CombineRule,
    ConvexMeshPart,
    GpuSphereBatch,
    GpuSphereWorld,
    JointMotor,
    LinkColliderKind,
    Material,
    MeshAsset,
    ModelFormat,
    SphereInput,
    SphereWorld,
    TesseraError,
    Vec3,
)


def vec(x: float, y: float, z: float) -> Vec3:
    return Vec3(x=x, y=y, z=z)


def main() -> None:
    material = Material(
        friction=0.7,
        restitution=0.3,
        friction_rule=CombineRule.MAX,
        restitution_rule=CombineRule.MIN,
    )
    ball = SphereInput(
        center=vec(0.0, 0.0, 0.4),
        velocity=vec(0.0, 0.0, -1.0),
        radius=0.5,
        mass=1.0,
    )
    cpu = SphereWorld([ball], vec(0.0, 0.0, 0.0), 10.0, 1.0, 0.0, 0.001, 12)
    cpu.step(0.001)
    assert cpu.contact_force(0).z > 0.0
    assert len(cpu.states()) == 1
    cpu.set_body_material(0, material)
    cpu.set_ground_material(material)
    assert cpu.body_material(0).friction == 0.7
    assert cpu.ground_material().restitution == 0.3

    urdf = """<robot name="arm"><link name="base"/><link name="tip">
      <inertial><mass value="1"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial>
      </link><joint name="slide" type="prismatic"><parent link="base"/><child link="tip"/>
      <axis xyz="1 0 0"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint></robot>"""
    robot = ArticulatedWorld.from_urdf(urdf, False)
    robot.set_joint_motor(
        0,
        JointMotor(
            position_target=0.4,
            velocity_target=0.0,
            stiffness=10.0,
            damping=1.0,
            max_force=5.0,
        ),
    )
    assert robot.joint_motor(0).position_target == 0.4
    robot.set_link_external_wrench(1, vec(1.0, 0.0, 0.0), vec(0.0, 0.0, 0.1))
    robot.set_link_gravity_scale(1, 0.5)
    assert robot.link_load(1).force.x == 1.0
    assert robot.link_load(1).gravity_scale == 0.5
    robot.set_positions([0.2])
    assert robot.joint_ranges()[0].name == "slide"
    robot.step(0.001, [0.0])
    robot.step_gpu(0.001, [0.0])
    assert len(robot.generalized_acceleration_gpu([0.0])) == 1
    assert len(robot.link_poses()) == 2

    batch_model = BatchModel(format=ModelFormat.URDF, xml=urdf, floating_base=False, meshes=[])
    articulated_batch = ArticulatedBatch([batch_model, batch_model])
    assert articulated_batch.len() == 2
    assert articulated_batch.environment_info(0).joint_ranges[0].name == "slide"
    articulated_batch.set_positions(0, [0.2])
    articulated_batch.publish_reset_template(0)
    articulated_batch.set_positions(0, [0.5])
    articulated_batch.set_positions(1, [-0.4])
    articulated_batch.reset_environment(0)
    assert articulated_batch.positions(0) == [0.2]
    assert articulated_batch.positions(1) == [-0.4]
    articulated_batch.set_joint_motor(
        0,
        0,
        JointMotor(
            position_target=0.3,
            velocity_target=0.0,
            stiffness=10.0,
            damping=1.0,
            max_force=5.0,
        ),
    )
    assert articulated_batch.joint_motor(0, 0).position_target == 0.3
    articulated_batch.set_link_external_wrench(
        0, 1, vec(1.0, 0.0, 0.0), vec(0.0, 0.0, 0.1)
    )
    articulated_batch.set_link_gravity_scale(0, 1, 0.5)
    assert articulated_batch.link_load(0, 1).force.x == 1.0
    assert articulated_batch.link_load(0, 1).gravity_scale == 0.5
    articulated_batch.publish_reset_template(0)
    articulated_batch.set_link_gravity_scale(0, 1, 1.0)
    articulated_batch.reset_environment(0)
    assert articulated_batch.link_load(0, 1).gravity_scale == 0.5
    articulated_batch.step_gpu(0.001, [[0.0], [0.0]])
    assert [len(acc) for acc in articulated_batch.generalized_accelerations_gpu([[0.0], [0.0]])] == [1, 1]
    assert len(articulated_batch.link_poses(1)) == 2

    diagonal = 3.0 ** -0.5
    mesh = MeshAsset(
        uri="mesh.obj",
        parts=[
            ConvexMeshPart(
                vertices=[vec(0, 0, 0), vec(1, 0, 0), vec(0, 1, 0), vec(0, 0, 1)],
                face_normals=[
                    vec(-1, 0, 0),
                    vec(0, -1, 0),
                    vec(0, 0, -1),
                    vec(diagonal, diagonal, diagonal),
                ],
                edge_directions=[vec(1, 0, 0), vec(0, 1, 0), vec(0, 0, 1)],
            )
        ],
    )
    mesh_urdf = """<robot name="mesh"><link name="base">
      <inertial><mass value="1"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
      <collision><geometry><mesh filename="mesh.obj" scale="2 3 4"/></geometry></collision>
    </link></robot>"""
    mesh_robot = ArticulatedWorld.from_urdf_with_meshes(mesh_urdf, False, [mesh])
    mesh_robot.set_link_material(LinkColliderKind.CONVEX, 0, material)
    assert mesh_robot.link_material(LinkColliderKind.CONVEX, 0).friction == 0.7
    mesh_robot.step(0.001, [])

    free_ball = SphereInput(
        center=vec(0.0, 0.0, 3.0),
        velocity=vec(0.0, 0.0, 0.0),
        radius=0.5,
        mass=1.0,
    )
    gpu = GpuSphereWorld([free_ball], vec(0.0, 0.0, 0.0), 10.0)
    gpu.set_body_material(0, material)
    gpu.set_ground_material(material)
    gpu.write_wrench(0, vec(0.0, 0.0, 0.0), vec(0.0, 0.0, 1.0))
    gpu.step(0.01)
    assert gpu.readback()[0].angular_velocity.z > 0.0
    assert gpu.readback()[0].orientation.z > 0.0
    added = SphereInput(
        center=vec(5.0, 0.0, 3.0),
        velocity=vec(0.0, 0.0, 0.0),
        radius=0.25,
        mass=1.0,
    )
    assert gpu.add_body(added) == 1
    assert gpu.len() == 2
    assert gpu.radius(1) == 0.25
    removed = gpu.remove_body(0)
    assert removed.radius == 0.5
    assert removed.state.angular_velocity.z > 0.0
    assert gpu.len() == 1
    assert gpu.readback()[0].center.x == 5.0

    batch = GpuSphereBatch([[ball], [ball]], vec(0.0, 0.0, 0.0), 10.0)
    batch.set_body_material(1, 0, material)
    batch.set_ground_material(material)
    batch.step(0.001)
    batch.write_force(1, 0, vec(0.0, 0.0, 10.0))
    batch.step_substeps(0.001, 2)
    second_before = batch.readback_environment(1)[0].center.z
    moved = SphereInput(
        center=vec(0.0, 0.0, 3.0),
        velocity=vec(0.0, 0.0, 0.0),
        radius=0.5,
        mass=1.0,
    )
    batch.reset_environment(0, [moved])
    assert batch.readback_environment(0)[0].center.z > 2.0
    assert batch.readback_environment(1)[0].center.z == second_before
    try:
        batch.reset_environment(
            0,
            [SphereInput(center=moved.center, velocity=moved.velocity, radius=0.6, mass=1.0)],
        )
    except TesseraError:
        pass
    else:
        raise AssertionError("reset accepted a changed collision radius")
    batch.reset_all([[ball], [ball]])
    print("tessera3d UniFFI CPU, URDF mesh, material, motor, dynamic GPU sphere, and batch smoke passed")


if __name__ == "__main__":
    main()
