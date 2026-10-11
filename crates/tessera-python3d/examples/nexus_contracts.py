"""Exercise imported controls, complete state and selected resident scene resets."""

from tessera3d import (
    ArticulatedBatch,
    ArticulatedWorld,
    BatchModel,
    GpuAxisServo,
    GpuTriangle,
    IndexedMeshPart,
    JointMotor,
    LinkPose,
    MeshAsset,
    MjcfActuatorDynamics,
    ModelFormat,
    MotorModel,
    Quaternion,
    UrdfOptions,
    Vec3,
)


def vec(x=0.0, y=0.0, z=0.0):
    return Vec3(x=x, y=y, z=z)


def main():
    # Added fields have generated defaults, preserving existing constructors.
    motor = JointMotor(
        position_target=0.5, velocity_target=0.0,
        stiffness=1.0, damping=1.0, max_force=float("inf"),
    )
    assert motor.model is None
    servo = GpuAxisServo(
        position_target=0.5, velocity_target=0.0,
        stiffness=1.0, damping=1.0, max_force=float("inf"),
    )
    assert servo.model is None

    xml = """<mujoco><option gravity="0 0 0"/><worldbody>
      <body><joint name="slide" type="slide" axis="1 0 0"/>
      <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/></body>
      </worldbody><actuator><motor joint="slide"/>
      <intvelocity name="inactive" joint="slide"/></actuator></mujoco>"""
    model = BatchModel(format=ModelFormat.MJCF, xml=xml, floating_base=False, meshes=[])
    batch = ArticulatedBatch([model, model])
    metadata = batch.actuator_info(0)
    assert isinstance(metadata[1].dynamics, MjcfActuatorDynamics.UNSUPPORTED)
    assert metadata[1].source_kind == "intvelocity"
    assert metadata[1].unsupported_reason is not None
    batch.set_controls(0, [1.0, 100.0])
    batch.set_controls(1, [-1.0, 100.0])
    batch.step_controls_gpu_resident(0.001, 2)
    template = batch.snapshot_gpu_resident()
    saved = batch.simulation_state(0)
    batch.set_controls(0, [2.0, 200.0])
    batch.step_controls_gpu_resident(0.001, 3)
    untouched = batch.simulation_state(1)
    batch.restore_gpu_resident(template, [0])
    assert batch.simulation_state(0) == saved
    assert batch.simulation_state(1) == untouched
    assert batch.controls(0) == [1.0, 100.0]
    batch.step_controls_gpu_resident(0.001, 1)

    robot = ArticulatedWorld.from_mjcf(xml)
    state = robot.simulation_state()
    state.configuration.positions = [0.25]
    state.joint_velocities = [0.5]
    robot.set_simulation_state(state)
    snapshot = robot.snapshot()
    robot.step(0.001, [0.0])
    robot.restore_snapshot(snapshot)
    assert robot.simulation_state() == state
    robot.set_joint_motor(0, motor)
    assert robot.joint_motor(0).model == MotorModel.FORCE_BASED
    assert robot.joint_motor(0).max_force == float("inf")

    identity = LinkPose(
        position=vec(),
        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    options = UrdfOptions(
        scale=2.0, shift=identity, root_pose=identity, floating_base=False,
        apply_imported_mass_props=True,
        create_colliders_from_collision_shapes=True,
        create_colliders_from_visual_shapes=False,
        enable_joint_collisions=True, convex_hull=False,
    )
    asset = MeshAsset(
        uri="surface.obj", parts=[],
        indexed_parts=[IndexedMeshPart(
            vertices=[vec(), vec(x=1.0), vec(y=1.0)],
            triangles=[GpuTriangle(a=0, b=1, c=2)],
        )],
    )
    surface = ArticulatedWorld.from_urdf_with_options(
        '<robot name="indexed"><link name="base"><collision><geometry>'
        '<mesh filename="surface.obj"/></geometry></collision></link></robot>',
        options, [asset],
    )
    assert surface.link_index("base") == 0
    print("Nexus physics contract examples passed")


if __name__ == "__main__":
    main()
