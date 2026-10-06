"""Read kinematic link acceleration and ideal IMU values."""

from tessera3d import ArticulatedWorld, Vec3


URDF = """<robot name="imu">
  <link name="base"/>
  <link name="tip">
    <inertial><mass value="1"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>
  <joint name="slide" type="prismatic">
    <parent link="base"/><child link="tip"/>
    <axis xyz="1 0 0"/>
    <limit lower="-2" upper="2" effort="10" velocity="10"/>
  </joint>
</robot>"""


def main() -> None:
    world = ArticulatedWorld.from_urdf(URDF, False)
    sensor_point = Vec3(x=0.0, y=0.0, z=0.0)
    generalized_acceleration = [2.0]
    acceleration = world.link_point_acceleration(1, sensor_point, generalized_acceleration)
    imu = world.link_imu(1, sensor_point, generalized_acceleration)
    assert abs(acceleration.linear.x - 2.0) < 1e-9
    assert abs(imu.specific_force.x - 2.0) < 1e-9
    assert abs(imu.specific_force.z - 9.81) < 1e-9
    print("World acceleration:", acceleration)
    print("Link-frame IMU:", imu)


if __name__ == "__main__":
    main()
