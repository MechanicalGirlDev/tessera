"""Read a link contact wrench from CPU, GPU, and batched articulated worlds."""

from tessera3d import ArticulatedBatch, ArticulatedWorld, BatchModel, ModelFormat


URDF = """<robot name="wrench">
  <link name="base">
    <inertial>
      <mass value="1"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
    <collision>
      <origin xyz="0.25 0 0.5"/>
      <geometry><sphere radius="0.5"/></geometry>
    </collision>
  </link>
</robot>"""


def main() -> None:
    cpu = ArticulatedWorld.from_urdf(URDF, True)
    gpu = ArticulatedWorld.from_urdf(URDF, True)
    batch = ArticulatedBatch(
        [BatchModel(format=ModelFormat.URDF, xml=URDF, floating_base=True, meshes=[])]
    )

    cpu.step(0.01, [])
    gpu.step_gpu(0.01, [])
    batch.step_gpu(0.01, [[]])

    cpu_wrench = cpu.link_contact_wrench(0)
    gpu_wrench = gpu.link_contact_wrench(0)
    batch_wrench = batch.link_contact_wrench(0, 0)
    assert cpu_wrench.force.z > 0.0 and cpu_wrench.torque.y < 0.0
    assert abs(gpu_wrench.torque.y - cpu_wrench.torque.y) < 1e-3
    assert abs(batch_wrench.torque.y - gpu_wrench.torque.y) < 1e-6
    print("CPU:", cpu_wrench)
    print("GPU:", gpu_wrench)
    print("Batch:", batch_wrench)


if __name__ == "__main__":
    main()
