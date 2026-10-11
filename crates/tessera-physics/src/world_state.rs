//! Complete simulation state application through the native snapshot validator.

use super::*;
use crate::inverse_kinematics::IkState;

/// Joint, explicit quaternion, and floating-base state for one native world.
#[derive(Debug, Clone)]
pub struct ArticulatedSimulationState {
    /// Root world pose, joint coordinates, and optional spherical orientations.
    pub configuration: IkState,
    /// Reduced joint velocities, excluding the floating-root twist.
    pub joint_velocities: Vec<f64>,
    /// Floating-root world linear velocity in metres per second.
    pub root_linear_velocity: Vector3<f64>,
    /// Floating-root world angular velocity in radians per second.
    pub root_angular_velocity: Vector3<f64>,
}

impl ArticulatedWorld {
    /// Capture complete simulation coordinates and velocities without stepping.
    pub fn simulation_state(&self) -> ArticulatedSimulationState {
        ArticulatedSimulationState {
            configuration: IkState {
                root_pose: self.root_pose,
                positions: self.positions.as_slice().to_vec(),
                orientations: self.spherical_orientations.clone(),
            },
            joint_velocities: self.velocities.as_slice().to_vec(),
            root_linear_velocity: self.base_linear_velocity,
            root_angular_velocity: self.base_angular_velocity,
        }
    }

    /// Atomically apply root/joint/quaternion state and discard old contact history.
    ///
    /// Fixed roots cannot have a nonzero twist. All dimensions, finite values,
    /// limits, and quaternion layouts pass the same validator as native reset.
    pub fn set_simulation_state(
        &mut self,
        state: &ArticulatedSimulationState,
    ) -> Result<(), ArticulatedWorldError> {
        if !self.floating
            && (state.root_linear_velocity != Vector3::zeros()
                || state.root_angular_velocity != Vector3::zeros())
        {
            return Err(ArticulatedWorldError::InvalidInput);
        }
        if state.configuration.positions.len() != self.articulation.dof()
            || state.joint_velocities.len() != self.articulation.dof()
            || state
                .configuration
                .positions
                .iter()
                .chain(state.joint_velocities.iter())
                .chain(state.root_linear_velocity.iter())
                .chain(state.root_angular_velocity.iter())
                .chain(state.configuration.root_pose.translation.vector.iter())
                .chain(state.configuration.root_pose.rotation.coords.iter())
                .any(|value| !value.is_finite())
        {
            return Err(ArticulatedWorldError::InvalidInput);
        }
        let mut snapshot = self.snapshot();
        snapshot.root_pose = state.configuration.root_pose;
        snapshot.positions = DVector::from_vec(state.configuration.positions.clone());
        snapshot.velocities = DVector::from_vec(state.joint_velocities.clone());
        snapshot.spherical_orientations = state.configuration.orientations.clone();
        snapshot.base_linear_velocity = state.root_linear_velocity;
        snapshot.base_angular_velocity = state.root_angular_velocity;
        self.restore_snapshot(&snapshot)?;
        Ok(())
    }

    /// Set finite world gravity for subsequent native CPU/GPU steps.
    pub fn set_gravity(&mut self, gravity: Vector3<f64>) -> Result<(), ArticulatedWorldError> {
        if gravity.iter().any(|value| !value.is_finite()) {
            return Err(ArticulatedWorldError::InvalidInput);
        }
        self.params.gravity = [gravity.x, gravity.y, gravity.z];
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::urdf::{UrdfLoadOptions, load_urdf_str};

    const ROBOT: &str = r#"<robot name="physical">
      <link name="base"><inertial><mass value="1"/>
        <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
      </inertial></link>
      <link name="tip"><inertial><origin xyz="0.1 0.2 0.3"/><mass value="2"/>
        <inertia ixx="2" ixy="0" ixz="0" iyy="3" iyz="0" izz="4"/>
      </inertial></link>
      <joint name="slide" type="prismatic"><parent link="base"/><child link="tip"/>
        <origin xyz="0 0 0.5"/><axis xyz="1 0 0"/>
        <limit lower="-2" upper="2" effort="10" velocity="10"/>
      </joint></robot>"#;

    #[test]
    fn prescribed_selection_tracks_public_floating_mode_and_snapshots()
    -> Result<(), Box<dyn core::error::Error>> {
        let mut world = load_urdf_str(ROBOT, UrdfLoadOptions::default())?.world;
        world.set_gravity(Vector3::zeros())?;
        world.set_prescribed_coordinates(&[true])?;
        world.floating = true;
        assert_eq!(
            world.prescribed_coordinates(),
            &[false, false, false, false, false, false, true]
        );
        world.set_prescribed_coordinates(&[true; 7])?;
        world.base_linear_velocity.x = 0.2;
        world.base_force.x = 1000.0;
        world.velocities[0] = 0.4;
        world.step(0.001, &[1000.0])?;
        assert!((world.base_linear_velocity.x - 0.2).abs() < 1e-12);
        assert!((world.velocities[0] - 0.4).abs() < 1e-12);
        let floating = world.snapshot();
        world.set_prescribed_coordinates(&[false; 7])?;
        world.restore_snapshot(&floating)?;
        assert_eq!(world.prescribed_coordinates(), &[true; 7]);

        world.floating = false;
        assert_eq!(world.prescribed_coordinates(), &[true]);
        world.set_prescribed_coordinates(&[false])?;
        world.step(0.001, &[1.0])?;
        assert!(world.velocities[0] > 0.4);
        let fixed = world.snapshot();
        world.set_prescribed_coordinates(&[true])?;
        world.restore_snapshot(&fixed)?;
        assert_eq!(world.prescribed_coordinates(), &[false]);
        world.floating = true;
        assert_eq!(
            world.prescribed_coordinates(),
            &[true, true, true, true, true, true, false]
        );
        Ok(())
    }

    #[test]
    fn physical_snapshot_restores_inertials_joint_frames_limits_and_loads()
    -> Result<(), Box<dyn core::error::Error>> {
        let mut world = load_urdf_str(ROBOT, UrdfLoadOptions::default())?.world;
        world.set_gravity(Vector3::zeros())?;
        world.link_loads[1] = LinkExternalLoad {
            force: Vector3::new(3.0, 0.0, 0.0),
            torque: Vector3::zeros(),
            gravity_scale: 0.4,
        };
        let snapshot = world.snapshot();
        let changed = ROBOT
            .replace("mass value=\"2\"", "mass value=\"20\"")
            .replace("xyz=\"0.1 0.2 0.3\"", "xyz=\"1 2 3\"")
            .replace("xyz=\"0 0 0.5\"", "xyz=\"0 0 5\"")
            .replace("upper=\"2\"", "upper=\"20\"");
        world.articulation = load_urdf_str(&changed, UrdfLoadOptions::default())?
            .world
            .articulation;
        world.set_gravity(Vector3::new(0.0, 0.0, -2.0))?;
        world.link_loads[1] = LinkExternalLoad::default();

        world.restore_snapshot(&snapshot)?;

        let link = world
            .articulation
            .link(1)
            .ok_or(ArticulatedWorldError::InvalidInput)?;
        assert_eq!(link.mass, 2.0);
        assert_eq!(link.center_of_mass, Vector3::new(0.1, 0.2, 0.3));
        assert_eq!(link.inertia[(0, 0)], 2.0);
        assert_eq!(link.inertia[(1, 1)], 3.0);
        assert_eq!(
            world.articulation.joint_specs()[0]
                .origin
                .translation
                .vector
                .z,
            0.5
        );
        assert_eq!(world.articulation.joint_limit(0), Some((-2.0, 2.0)));
        assert_eq!(world.params.gravity, [0.0; 3]);
        assert_eq!(world.link_loads[1].force.x, 3.0);
        assert_eq!(world.link_loads[1].gravity_scale, 0.4);
        Ok(())
    }
}
