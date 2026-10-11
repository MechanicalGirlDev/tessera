//! Complete simulation state and imported-controller snapshots at the FFI boundary.

use super::*;
use tessera_physics::articulated_world::{
    ArticulatedSimulationState as CoreSimulationState, ArticulatedWorldSnapshot,
};

/// Complete joint/quaternion configuration and simulation velocities.
#[derive(Clone, Debug, uniffi::Record)]
pub struct SimulationState {
    /// Root pose, reduced coordinates, and explicit spherical orientations.
    pub configuration: IkState,
    /// Reduced joint velocities; floating-base velocities are separate.
    pub joint_velocities: Vec<f64>,
    /// Floating-root world linear/angular velocity.
    pub root_twist: LinkTwist,
}

impl SimulationState {
    fn native(self) -> Result<CoreSimulationState, TesseraError> {
        Ok(CoreSimulationState {
            configuration: self.configuration.native()?,
            joint_velocities: self.joint_velocities,
            root_linear_velocity: Vector3::new(
                self.root_twist.linear.x,
                self.root_twist.linear.y,
                self.root_twist.linear.z,
            ),
            root_angular_velocity: Vector3::new(
                self.root_twist.angular.x,
                self.root_twist.angular.y,
                self.root_twist.angular.z,
            ),
        })
    }
}

impl From<CoreSimulationState> for SimulationState {
    fn from(state: CoreSimulationState) -> Self {
        Self {
            configuration: state.configuration.into(),
            joint_velocities: state.joint_velocities,
            root_twist: LinkTwist {
                linear: state.root_linear_velocity.into(),
                angular: state.root_angular_velocity.into(),
            },
        }
    }
}

/// Immutable native simulation snapshot including held imported actuator controls.
#[derive(Debug, uniffi::Object)]
pub struct SimulationSnapshot {
    world: ArticulatedWorldSnapshot,
    controls: Vec<f64>,
    joint_names: Vec<String>,
    link_names: Vec<String>,
}

#[uniffi::export]
impl ArticulatedWorld {
    /// Complete state, including floating-root twist and explicit quaternions.
    pub fn simulation_state(&self) -> Result<SimulationState, TesseraError> {
        Ok(locked(&self.inner)?.world.simulation_state().into())
    }

    /// Atomically replace complete state; subsequent GPU sessions rebuild from it.
    pub fn set_simulation_state(&self, state: SimulationState) -> Result<(), TesseraError> {
        let native = state.native()?;
        let mut inner = locked(&self.inner)?;
        inner.world.set_simulation_state(&native).map_err(failed)?;
        inner.resident = None;
        Ok(())
    }

    /// Teleport the full robot and zero all velocities, as in Nexus robot-qpos reset.
    pub fn set_configuration(&self, configuration: IkState) -> Result<(), TesseraError> {
        let configuration = configuration.native()?;
        let mut inner = locked(&self.inner)?;
        let mut state = inner.world.simulation_state();
        state.configuration = configuration;
        state.joint_velocities.fill(0.0);
        state.root_linear_velocity = Vector3::zeros();
        state.root_angular_velocity = Vector3::zeros();
        inner.world.set_simulation_state(&state).map_err(failed)?;
        inner.resident = None;
        Ok(())
    }

    /// Capture native mutable state and held actuator-order controls.
    pub fn snapshot(&self) -> Result<Arc<SimulationSnapshot>, TesseraError> {
        let inner = locked(&self.inner)?;
        Ok(Arc::new(SimulationSnapshot {
            world: inner.world.snapshot(),
            controls: inner.actuators.controls().to_vec(),
            joint_names: self
                .joint_ranges
                .iter()
                .map(|joint| joint.name.clone())
                .collect(),
            link_names: self.link_names.clone(),
        }))
    }

    /// Restore a compatible snapshot; failed validation leaves state/controls intact.
    pub fn restore_snapshot(&self, snapshot: Arc<SimulationSnapshot>) -> Result<(), TesseraError> {
        if snapshot.joint_names
            != self
                .joint_ranges
                .iter()
                .map(|joint| joint.name.clone())
                .collect::<Vec<_>>()
            || snapshot.link_names != self.link_names
        {
            return Err(failed("snapshot model layout does not match this robot"));
        }
        let mut inner = locked(&self.inner)?;
        let mut actuators = inner.actuators.clone();
        actuators.set_controls(&snapshot.controls).map_err(failed)?;
        inner
            .world
            .restore_snapshot(&snapshot.world)
            .map_err(failed)?;
        inner.actuators = actuators;
        inner.resident = None;
        Ok(())
    }

    /// Change finite gravity while retaining the current solved simulation state.
    pub fn set_gravity(&self, gravity: Vec3) -> Result<(), TesseraError> {
        let mut inner = locked(&self.inner)?;
        inner
            .world
            .set_gravity(Vector3::new(gravity.x, gravity.y, gravity.z))
            .map_err(failed)?;
        inner.resident = None;
        Ok(())
    }
}

#[uniffi::export]
impl ArticulatedBatch {
    /// Complete state of one environment, with root twist and explicit quaternions.
    pub fn simulation_state(&self, index: u32) -> Result<SimulationState, TesseraError> {
        let id = self.id(index)?;
        let inner = locked(&self.inner)?;
        let world = inner
            .batch
            .environment(id)
            .ok_or_else(|| failed("environment not found"))?;
        Ok(world.simulation_state().into())
    }

    /// Atomically replace one environment's complete simulation state.
    pub fn set_simulation_state(
        &self,
        index: u32,
        state: SimulationState,
    ) -> Result<(), TesseraError> {
        let id = self.id(index)?;
        let native = state.native()?;
        let mut inner = locked(&self.inner)?;
        let world = inner
            .batch
            .environment_mut(id)
            .ok_or_else(|| failed("environment not found"))?;
        world.set_simulation_state(&native).map_err(failed)?;
        inner.resident = None;
        Ok(())
    }

    /// Teleport one complete robot and zero its joint/root velocities.
    pub fn set_configuration(
        &self,
        index: u32,
        configuration: IkState,
    ) -> Result<(), TesseraError> {
        let id = self.id(index)?;
        let configuration = configuration.native()?;
        let mut inner = locked(&self.inner)?;
        let world = inner
            .batch
            .environment_mut(id)
            .ok_or_else(|| failed("environment not found"))?;
        let mut state = world.simulation_state();
        state.configuration = configuration;
        state.joint_velocities.fill(0.0);
        state.root_linear_velocity = Vector3::zeros();
        state.root_angular_velocity = Vector3::zeros();
        world.set_simulation_state(&state).map_err(failed)?;
        inner.resident = None;
        Ok(())
    }

    /// Capture one environment's native state and held imported controls.
    pub fn snapshot(&self, index: u32) -> Result<Arc<SimulationSnapshot>, TesseraError> {
        let id = self.id(index)?;
        let inner = locked(&self.inner)?;
        let world = inner
            .batch
            .environment(id)
            .ok_or_else(|| failed("environment not found"))?;
        let info = &self.info[index as usize];
        Ok(Arc::new(SimulationSnapshot {
            world: world.snapshot(),
            controls: inner.actuators[index as usize].controls().to_vec(),
            joint_names: info
                .joint_ranges
                .iter()
                .map(|joint| joint.name.clone())
                .collect(),
            link_names: info.link_names.clone(),
        }))
    }

    /// Restore one compatible environment without changing unselected environments.
    pub fn restore_snapshot(
        &self,
        index: u32,
        snapshot: Arc<SimulationSnapshot>,
    ) -> Result<(), TesseraError> {
        let id = self.id(index)?;
        let info = &self.info[index as usize];
        if snapshot.joint_names
            != info
                .joint_ranges
                .iter()
                .map(|joint| joint.name.clone())
                .collect::<Vec<_>>()
            || snapshot.link_names != info.link_names
        {
            return Err(failed("snapshot model layout does not match this robot"));
        }
        let mut inner = locked(&self.inner)?;
        let mut actuators = inner.actuators[index as usize].clone();
        actuators.set_controls(&snapshot.controls).map_err(failed)?;
        inner
            .batch
            .environment_mut(id)
            .ok_or_else(|| failed("environment not found"))?
            .restore_snapshot(&snapshot.world)
            .map_err(failed)?;
        inner.actuators[index as usize] = actuators;
        inner.resident = None;
        Ok(())
    }

    /// Set finite world gravity for one environment.
    pub fn set_gravity(&self, index: u32, gravity: Vec3) -> Result<(), TesseraError> {
        let id = self.id(index)?;
        let mut inner = locked(&self.inner)?;
        inner
            .batch
            .environment_mut(id)
            .ok_or_else(|| failed("environment not found"))?
            .set_gravity(gravity.nalgebra())
            .map_err(failed)?;
        inner.resident = None;
        Ok(())
    }

    /// Held imported actuator-order controls for one environment.
    pub fn controls(&self, index: u32) -> Result<Vec<f64>, TesseraError> {
        let _id = self.id(index)?;
        Ok(locked(&self.inner)?.actuators[index as usize]
            .controls()
            .to_vec())
    }

    /// Atomically replace held controls for one environment.
    pub fn set_controls(&self, index: u32, controls: Vec<f64>) -> Result<(), TesseraError> {
        let _id = self.id(index)?;
        locked(&self.inner)?.actuators[index as usize]
            .set_controls(&controls)
            .map_err(failed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROBOT: &str = r#"<robot name="state">
      <link name="base"><inertial><mass value="1"/>
        <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
      </inertial></link>
      <link name="tip"><inertial><mass value="2"/>
        <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
      </inertial></link>
      <joint name="slide" type="prismatic"><parent link="base"/><child link="tip"/>
        <axis xyz="1 0 0"/><limit lower="-2" upper="2" effort="10" velocity="10"/>
      </joint></robot>"#;

    #[test]
    fn simulation_state_applies_root_pose_twist_and_joint_state() -> Result<(), TesseraError> {
        let robot = ArticulatedWorld::from_urdf(ROBOT.into(), true)?;
        let mut state = robot.simulation_state()?;
        state.configuration.root_pose.position = Vec3 {
            x: 1.0,
            y: 2.0,
            z: 3.0,
        };
        state.configuration.positions[0] = 0.4;
        state.joint_velocities[0] = 0.3;
        state.root_twist.linear = Vec3 {
            x: 0.2,
            y: -0.4,
            z: 0.1,
        };
        state.root_twist.angular = Vec3 {
            x: 0.0,
            y: 0.5,
            z: 0.0,
        };

        robot.set_simulation_state(state)?;

        let actual = robot.simulation_state()?;
        assert_eq!(actual.configuration.root_pose.position.x, 1.0);
        assert_eq!(actual.configuration.root_pose.position.y, 2.0);
        assert_eq!(actual.configuration.root_pose.position.z, 3.0);
        assert_eq!(actual.configuration.positions, vec![0.4]);
        assert_eq!(actual.joint_velocities, vec![0.3]);
        assert_eq!(actual.root_twist.linear.y, -0.4);
        assert_eq!(actual.root_twist.angular.y, 0.5);
        Ok(())
    }

    #[test]
    fn simulation_state_invalid_velocity_does_not_partially_teleport() -> Result<(), TesseraError> {
        let robot = ArticulatedWorld::from_urdf(ROBOT.into(), true)?;
        let before = robot.simulation_state()?;
        let mut invalid = before.clone();
        invalid.configuration.root_pose.position.x = 7.0;
        invalid.configuration.positions[0] = 0.5;
        invalid.joint_velocities[0] = f64::NAN;

        assert!(robot.set_simulation_state(invalid).is_err());

        let actual = robot.simulation_state()?;
        assert_eq!(
            actual.configuration.positions,
            before.configuration.positions
        );
        assert_eq!(actual.joint_velocities, before.joint_velocities);
        assert_eq!(
            actual.configuration.root_pose.position.x,
            before.configuration.root_pose.position.x,
        );
        Ok(())
    }

    #[test]
    fn simulation_state_snapshot_restores_after_a_teleport() -> Result<(), TesseraError> {
        let robot = ArticulatedWorld::from_urdf(ROBOT.into(), true)?;
        let mut initial = robot.simulation_state()?;
        initial.joint_velocities[0] = 0.3;
        initial.root_twist.linear.x = 0.5;
        robot.set_simulation_state(initial)?;
        let snapshot = robot.snapshot()?;
        let mut moved = robot.simulation_state()?.configuration;
        moved.root_pose.position.x = 3.0;
        moved.positions[0] = -0.5;
        robot.set_configuration(moved)?;
        assert_eq!(robot.simulation_state()?.joint_velocities, vec![0.0]);
        assert_eq!(robot.simulation_state()?.root_twist.linear.x, 0.0);

        robot.restore_snapshot(snapshot)?;

        let actual = robot.simulation_state()?;
        assert_eq!(actual.configuration.positions, vec![0.0]);
        assert_eq!(actual.configuration.root_pose.position.x, 0.0);
        assert_eq!(actual.joint_velocities, vec![0.3]);
        assert_eq!(actual.root_twist.linear.x, 0.5);
        Ok(())
    }

    #[test]
    fn simulation_state_preserves_explicit_spherical_quaternions() -> Result<(), TesseraError> {
        let robot = ArticulatedWorld::from_mjcf(
            r#"<mujoco><worldbody><body name="sphere">
              <joint name="ball" type="ball"/>
              <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
              </body></worldbody></mujoco>"#
                .into(),
        )?;
        let mut state = robot.simulation_state()?;
        state.configuration.orientations = Some(vec![Some(Quaternion {
            x: 0.0,
            y: 0.0,
            z: 1.0,
            w: 0.0,
        })]);
        state.joint_velocities = vec![0.2, -0.3, 0.4];

        robot.set_simulation_state(state)?;

        let actual = robot.simulation_state()?;
        assert_eq!(actual.joint_velocities, vec![0.2, -0.3, 0.4]);
        let orientations = actual
            .configuration
            .orientations
            .ok_or_else(|| failed("missing orientations"))?;
        let rotation = orientations[0].ok_or_else(|| failed("missing ball orientation"))?;
        assert_eq!(rotation.z.abs(), 1.0);
        assert_eq!(rotation.w, 0.0);
        Ok(())
    }

    #[test]
    fn simulation_state_snapshot_restores_held_actuator_controls() -> Result<(), TesseraError> {
        let robot = ArticulatedWorld::from_mjcf(
            r#"<mujoco><worldbody><body name="tip">
              <joint name="hinge"/><inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
              </body></worldbody><actuator><motor name="drive" joint="hinge"/></actuator>
              </mujoco>"#
                .into(),
        )?;
        robot.set_controls(vec![0.4])?;
        let snapshot = robot.snapshot()?;
        robot.set_controls(vec![-0.7])?;

        robot.restore_snapshot(snapshot)?;

        assert_eq!(robot.controls()?, vec![0.4]);
        Ok(())
    }
}
