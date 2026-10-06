//! Spherical-joint drives in the parent-side joint tangent frame.

use nalgebra::{UnitQuaternion, Vector3};

/// Quaternion position and angular-velocity drive with per-axis effort limits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SphericalJointDrive {
    /// Desired child orientation relative to the parent-side joint frame.
    pub orientation_target: UnitQuaternion<f64>,
    /// Desired relative angular velocity in the parent-side joint frame.
    pub velocity_target: Vector3<f64>,
    /// Nonnegative proportional gains along the joint frame axes.
    pub stiffness: Vector3<f64>,
    /// Nonnegative derivative gains along the joint frame axes.
    pub damping: Vector3<f64>,
    /// Nonnegative component-wise torque caps in the joint frame.
    pub max_torque: Vector3<f64>,
}

/// Invalid spherical drive, state, or orientation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("invalid spherical joint drive or state")]
pub struct SphericalJointDriveError;

impl SphericalJointDrive {
    /// Compute capped torque using the shortest quaternion logarithm.
    /// The left orientation error and angular velocities share the joint frame.
    /// At exactly pi the logarithm axis is ambiguous; no continuity through that
    /// antipodal boundary is promised. Intrinsic Euler angles are never used.
    pub fn torque(
        &self,
        orientation: UnitQuaternion<f64>,
        angular_velocity: Vector3<f64>,
    ) -> Result<Vector3<f64>, SphericalJointDriveError> {
        self.validate()?;
        validate_orientation(&orientation)?;
        if !angular_velocity.iter().all(|v| v.is_finite()) {
            return Err(SphericalJointDriveError);
        }
        let error = (self.orientation_target * orientation.inverse()).scaled_axis();
        let torque = self.stiffness.component_mul(&error)
            + self
                .damping
                .component_mul(&(self.velocity_target - angular_velocity));
        if !torque.iter().all(|v| v.is_finite()) {
            return Err(SphericalJointDriveError);
        }
        Ok(Vector3::from_fn(|i, _| {
            torque[i].clamp(-self.max_torque[i], self.max_torque[i])
        }))
    }

    /// Validate parameters before a resident GPU upload or CPU force evaluation.
    pub fn validate(&self) -> Result<(), SphericalJointDriveError> {
        validate_orientation(&self.orientation_target)?;
        if !self.velocity_target.iter().all(|v| v.is_finite())
            || [&self.stiffness, &self.damping, &self.max_torque]
                .iter()
                .any(|values| values.iter().any(|v| !v.is_finite() || *v < 0.0))
        {
            return Err(SphericalJointDriveError);
        }
        Ok(())
    }
}

fn validate_orientation(q: &UnitQuaternion<f64>) -> Result<(), SphericalJointDriveError> {
    if !q.quaternion().coords.iter().all(|v| v.is_finite())
        || (q.quaternion().norm_squared() - 1.0).abs() > 1e-8
    {
        return Err(SphericalJointDriveError);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Quaternion;

    #[test]
    fn driven_tangent_dynamics_crosses_gimbal_lock_and_converges() {
        use crate::articulation::{Articulation, JointKind, JointSpec, LinkSpec};
        use nalgebra::{DVector, Isometry3, Matrix3};
        let art = Articulation::new(
            vec![
                LinkSpec {
                    mass: 0.0,
                    center_of_mass: Vector3::zeros(),
                    inertia: Matrix3::zeros(),
                },
                LinkSpec {
                    mass: 1.0,
                    center_of_mass: Vector3::zeros(),
                    inertia: Matrix3::identity() * 0.5,
                },
            ],
            vec![JointSpec {
                parent: 0,
                child: 1,
                kind: JointKind::Spherical,
                origin: Isometry3::identity(),
                axis: Vector3::z(),
                limits: None,
            }],
            0,
        )
        .unwrap();
        let drive = SphericalJointDrive {
            orientation_target: UnitQuaternion::from_scaled_axis(Vector3::y() * 1.8),
            velocity_target: Vector3::zeros(),
            stiffness: Vector3::repeat(10.0),
            damping: Vector3::repeat(4.0),
            max_torque: Vector3::repeat(10.0),
        };
        let mut orientation = UnitQuaternion::from_scaled_axis(Vector3::y() * 1.4);
        let mut velocity = Vector3::zeros();
        let mut previous_energy = 0.8;
        let mut crossed = false;
        for _ in 0..2000 {
            let dynamics = art
                .generalized_dynamics_with_spherical_orientations(
                    Isometry3::identity(),
                    &[0.0; 3],
                    &[Some(orientation)],
                    &DVector::from_column_slice(velocity.as_slice()),
                    false,
                    Vector3::zeros(),
                )
                .unwrap();
            let torque = drive.torque(orientation, velocity).unwrap();
            let acceleration = dynamics
                .mass
                .lu()
                .solve(&(DVector::from_column_slice(torque.as_slice()) - dynamics.velocity_bias))
                .unwrap();
            velocity += Vector3::from_column_slice(acceleration.as_slice()) * 0.001;
            orientation = UnitQuaternion::from_scaled_axis(velocity * 0.001) * orientation;
            crossed |= orientation.scaled_axis().y > core::f64::consts::FRAC_PI_2;
            let error = (drive.orientation_target * orientation.inverse()).angle();
            let energy = 0.25 * velocity.norm_squared() + 5.0 * error * error;
            assert!(energy <= previous_energy + 1e-9);
            previous_energy = energy;
        }
        assert!(crossed);
        assert!((drive.orientation_target * orientation.inverse()).angle() < 1e-3);
        assert!(velocity.norm() < 1e-3);
    }

    #[test]
    fn left_tangent_error_is_regular_at_euler_gimbal_lock_and_sign_invariant() {
        let current = UnitQuaternion::from_euler_angles(0.3, core::f64::consts::FRAC_PI_2, -0.5);
        let error = Vector3::new(0.2, -0.3, 0.1);
        let velocity = Vector3::new(0.4, -0.2, 0.7);
        let drive = SphericalJointDrive {
            orientation_target: UnitQuaternion::from_scaled_axis(error) * current,
            velocity_target: Vector3::new(-0.2, 0.1, 0.3),
            stiffness: Vector3::new(10.0, 20.0, 30.0),
            damping: Vector3::new(2.0, 3.0, 4.0),
            max_torque: Vector3::new(100.0, 4.0, 100.0),
        };
        let result = drive.torque(current, velocity).unwrap();
        assert!((result - Vector3::new(0.8, -4.0, 1.4)).norm() < 1e-12);
        let negative_current = UnitQuaternion::new_unchecked(-current.into_inner());
        assert!((drive.torque(negative_current, velocity).unwrap() - result).norm() < 1e-12);
        let mut negative_target = drive;
        negative_target.orientation_target =
            UnitQuaternion::new_unchecked(-drive.orientation_target.into_inner());
        assert!((negative_target.torque(current, velocity).unwrap() - result).norm() < 1e-12);
        let mut invalid = drive;
        invalid.orientation_target =
            UnitQuaternion::new_unchecked(Quaternion::new(2.0, 0.0, 0.0, 0.0));
        assert!(invalid.validate().is_err());
        invalid = drive;
        invalid.stiffness.x = -1.0;
        assert!(invalid.validate().is_err());
        invalid = drive;
        invalid.max_torque.z = f64::INFINITY;
        assert!(invalid.validate().is_err());
        assert!(
            drive
                .torque(current, Vector3::new(f64::NAN, 0.0, 0.0))
                .is_err()
        );
    }
}
