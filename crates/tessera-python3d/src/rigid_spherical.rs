//! Spherical rotation-vector controls for independent GPU rigid owners.
//! Register as a child module of primitive_batch to share its private owner lock.

use crate::*;
use tessera_physics::gpu_rigid_ball_joint::GpuRigidSphericalLimit;

/// Bounds in radians on one shortest quaternion rotation-vector component.
#[derive(Clone, Debug, uniffi::Record)]
pub struct GpuSphericalLimit {
    /// Lower relative rotation-vector bound.
    pub min: f64,
    /// Upper relative rotation-vector bound.
    pub max: f64,
}

impl From<GpuSphericalLimit> for GpuRigidSphericalLimit {
    fn from(value: GpuSphericalLimit) -> Self {
        Self {
            min: value.min as f32,
            max: value.max as f32,
        }
    }
}

impl From<GpuRigidSphericalLimit> for GpuSphericalLimit {
    fn from(value: GpuRigidSphericalLimit) -> Self {
        Self {
            min: f64::from(value.min),
            max: f64::from(value.max),
        }
    }
}

#[uniffi::export]
impl GpuSphereWorld {
    /// Number of spherical ball joints.
    pub fn spherical_joint_count(&self) -> Result<u32, TesseraError> {
        u32::try_from(locked(&self.world)?.spherical_joint_count()).map_err(failed)
    }

    /// Number of spherical rotation-vector axes.
    pub fn spherical_axis_count(&self) -> Result<u32, TesseraError> {
        u32::try_from(locked(&self.world)?.spherical_axis_count()).map_err(failed)
    }

    /// Read a spherical motor; ball-joint insertion index and body-A-local X/Y/Z axis 0/1/2.
    pub fn spherical_motor(
        &self,
        index: u32,
        axis: u32,
    ) -> Result<Option<GpuAxisMotor>, TesseraError> {
        locked(&self.world)?
            .spherical_motor(index as usize, axis as usize)
            .map(|value| value.map(Into::into))
            .map_err(failed)
    }

    /// Set or clear a spherical motor; invalid indices and parameters leave settings unchanged.
    pub fn set_spherical_motor(
        &self,
        index: u32,
        axis: u32,
        motor: Option<GpuAxisMotor>,
    ) -> Result<(), TesseraError> {
        locked(&self.world)?
            .set_spherical_motor(index as usize, axis as usize, motor.map(Into::into))
            .map_err(failed)
    }

    /// Read a spherical servo; ball-joint insertion index and body-A-local X/Y/Z axis 0/1/2.
    pub fn spherical_servo(
        &self,
        index: u32,
        axis: u32,
    ) -> Result<Option<GpuAxisServo>, TesseraError> {
        locked(&self.world)?
            .spherical_servo(index as usize, axis as usize)
            .map(|value| value.map(Into::into))
            .map_err(failed)
    }

    /// Set or clear a spherical servo; invalid indices and parameters leave settings unchanged.
    pub fn set_spherical_servo(
        &self,
        index: u32,
        axis: u32,
        servo: Option<GpuAxisServo>,
    ) -> Result<(), TesseraError> {
        locked(&self.world)?
            .set_spherical_servo(index as usize, axis as usize, servo.map(Into::into))
            .map_err(failed)
    }

    /// Read a spherical limit; ball-joint insertion index and body-A-local X/Y/Z axis 0/1/2.
    pub fn spherical_limit(
        &self,
        index: u32,
        axis: u32,
    ) -> Result<Option<GpuSphericalLimit>, TesseraError> {
        locked(&self.world)?
            .spherical_limit(index as usize, axis as usize)
            .map(|value| value.map(Into::into))
            .map_err(failed)
    }

    /// Set or clear a spherical limit; invalid indices and parameters leave settings unchanged.
    pub fn set_spherical_limit(
        &self,
        index: u32,
        axis: u32,
        limit: Option<GpuSphericalLimit>,
    ) -> Result<(), TesseraError> {
        locked(&self.world)?
            .set_spherical_limit(index as usize, axis as usize, limit.map(Into::into))
            .map_err(failed)
    }
}

#[uniffi::export]
impl GpuPrimitiveWorld {
    /// Number of spherical ball joints.
    pub fn spherical_joint_count(&self) -> Result<u32, TesseraError> {
        u32::try_from(locked(&self.inner)?.world.spherical_joint_count()).map_err(failed)
    }

    /// Number of spherical rotation-vector axes.
    pub fn spherical_axis_count(&self) -> Result<u32, TesseraError> {
        u32::try_from(locked(&self.inner)?.world.spherical_axis_count()).map_err(failed)
    }

    /// Read a spherical motor; ball-joint insertion index and body-A-local X/Y/Z axis 0/1/2.
    pub fn spherical_motor(
        &self,
        index: u32,
        axis: u32,
    ) -> Result<Option<GpuAxisMotor>, TesseraError> {
        locked(&self.inner)?
            .world
            .spherical_motor(index as usize, axis as usize)
            .map(|value| value.map(Into::into))
            .map_err(failed)
    }

    /// Set or clear a spherical motor; invalid indices and parameters leave settings unchanged.
    pub fn set_spherical_motor(
        &self,
        index: u32,
        axis: u32,
        motor: Option<GpuAxisMotor>,
    ) -> Result<(), TesseraError> {
        locked(&self.inner)?
            .world
            .set_spherical_motor(index as usize, axis as usize, motor.map(Into::into))
            .map_err(failed)
    }

    /// Read a spherical servo; ball-joint insertion index and body-A-local X/Y/Z axis 0/1/2.
    pub fn spherical_servo(
        &self,
        index: u32,
        axis: u32,
    ) -> Result<Option<GpuAxisServo>, TesseraError> {
        locked(&self.inner)?
            .world
            .spherical_servo(index as usize, axis as usize)
            .map(|value| value.map(Into::into))
            .map_err(failed)
    }

    /// Set or clear a spherical servo; invalid indices and parameters leave settings unchanged.
    pub fn set_spherical_servo(
        &self,
        index: u32,
        axis: u32,
        servo: Option<GpuAxisServo>,
    ) -> Result<(), TesseraError> {
        locked(&self.inner)?
            .world
            .set_spherical_servo(index as usize, axis as usize, servo.map(Into::into))
            .map_err(failed)
    }

    /// Read a spherical limit; ball-joint insertion index and body-A-local X/Y/Z axis 0/1/2.
    pub fn spherical_limit(
        &self,
        index: u32,
        axis: u32,
    ) -> Result<Option<GpuSphericalLimit>, TesseraError> {
        locked(&self.inner)?
            .world
            .spherical_limit(index as usize, axis as usize)
            .map(|value| value.map(Into::into))
            .map_err(failed)
    }

    /// Set or clear a spherical limit; invalid indices and parameters leave settings unchanged.
    pub fn set_spherical_limit(
        &self,
        index: u32,
        axis: u32,
        limit: Option<GpuSphericalLimit>,
    ) -> Result<(), TesseraError> {
        locked(&self.inner)?
            .world
            .set_spherical_limit(index as usize, axis as usize, limit.map(Into::into))
            .map_err(failed)
    }
}

#[uniffi::export]
impl GpuSphereBatch {
    /// Number of spherical ball joints in one environment.
    pub fn spherical_joint_count_environment(&self, environment: u32) -> Result<u32, TesseraError> {
        u32::try_from(
            locked(&self.batch)?
                .spherical_joint_count_environment(environment as usize)
                .map_err(failed)?,
        )
        .map_err(failed)
    }

    /// Number of spherical rotation-vector axes in one environment.
    pub fn spherical_axis_count_environment(&self, environment: u32) -> Result<u32, TesseraError> {
        u32::try_from(
            locked(&self.batch)?
                .spherical_axis_count_environment(environment as usize)
                .map_err(failed)?,
        )
        .map_err(failed)
    }

    /// Packed world index for an environment-local ball-joint index.
    pub fn spherical_joint_index_environment(
        &self,
        environment: u32,
        index: u32,
    ) -> Result<u32, TesseraError> {
        u32::try_from(
            locked(&self.batch)?
                .spherical_joint_index_environment(environment as usize, index as usize)
                .map_err(failed)?,
        )
        .map_err(failed)
    }

    /// Read a spherical motor; ball-joint insertion index and body-A-local X/Y/Z axis 0/1/2.
    pub fn spherical_motor_environment(
        &self,
        environment: u32,
        index: u32,
        axis: u32,
    ) -> Result<Option<GpuAxisMotor>, TesseraError> {
        locked(&self.batch)?
            .spherical_motor_environment(environment as usize, index as usize, axis as usize)
            .map(|value| value.map(Into::into))
            .map_err(failed)
    }

    /// Set or clear a spherical motor; invalid indices and parameters leave settings unchanged.
    pub fn set_spherical_motor_environment(
        &self,
        environment: u32,
        index: u32,
        axis: u32,
        motor: Option<GpuAxisMotor>,
    ) -> Result<(), TesseraError> {
        locked(&self.batch)?
            .set_spherical_motor_environment(
                environment as usize,
                index as usize,
                axis as usize,
                motor.map(Into::into),
            )
            .map_err(failed)
    }

    /// Read a spherical servo; ball-joint insertion index and body-A-local X/Y/Z axis 0/1/2.
    pub fn spherical_servo_environment(
        &self,
        environment: u32,
        index: u32,
        axis: u32,
    ) -> Result<Option<GpuAxisServo>, TesseraError> {
        locked(&self.batch)?
            .spherical_servo_environment(environment as usize, index as usize, axis as usize)
            .map(|value| value.map(Into::into))
            .map_err(failed)
    }

    /// Set or clear a spherical servo; invalid indices and parameters leave settings unchanged.
    pub fn set_spherical_servo_environment(
        &self,
        environment: u32,
        index: u32,
        axis: u32,
        servo: Option<GpuAxisServo>,
    ) -> Result<(), TesseraError> {
        locked(&self.batch)?
            .set_spherical_servo_environment(
                environment as usize,
                index as usize,
                axis as usize,
                servo.map(Into::into),
            )
            .map_err(failed)
    }

    /// Read a spherical limit; ball-joint insertion index and body-A-local X/Y/Z axis 0/1/2.
    pub fn spherical_limit_environment(
        &self,
        environment: u32,
        index: u32,
        axis: u32,
    ) -> Result<Option<GpuSphericalLimit>, TesseraError> {
        locked(&self.batch)?
            .spherical_limit_environment(environment as usize, index as usize, axis as usize)
            .map(|value| value.map(Into::into))
            .map_err(failed)
    }

    /// Set or clear a spherical limit; invalid indices and parameters leave settings unchanged.
    pub fn set_spherical_limit_environment(
        &self,
        environment: u32,
        index: u32,
        axis: u32,
        limit: Option<GpuSphericalLimit>,
    ) -> Result<(), TesseraError> {
        locked(&self.batch)?
            .set_spherical_limit_environment(
                environment as usize,
                index as usize,
                axis as usize,
                limit.map(Into::into),
            )
            .map_err(failed)
    }
}

#[uniffi::export]
impl GpuPrimitiveBatch {
    /// Number of spherical ball joints in one environment.
    pub fn spherical_joint_count_environment(&self, environment: u32) -> Result<u32, TesseraError> {
        u32::try_from(
            locked(&self.batch)?
                .spherical_joint_count_environment(environment as usize)
                .map_err(failed)?,
        )
        .map_err(failed)
    }

    /// Number of spherical rotation-vector axes in one environment.
    pub fn spherical_axis_count_environment(&self, environment: u32) -> Result<u32, TesseraError> {
        u32::try_from(
            locked(&self.batch)?
                .spherical_axis_count_environment(environment as usize)
                .map_err(failed)?,
        )
        .map_err(failed)
    }

    /// Packed world index for an environment-local ball-joint index.
    pub fn spherical_joint_index_environment(
        &self,
        environment: u32,
        index: u32,
    ) -> Result<u32, TesseraError> {
        u32::try_from(
            locked(&self.batch)?
                .spherical_joint_index_environment(environment as usize, index as usize)
                .map_err(failed)?,
        )
        .map_err(failed)
    }

    /// Read a spherical motor; ball-joint insertion index and body-A-local X/Y/Z axis 0/1/2.
    pub fn spherical_motor_environment(
        &self,
        environment: u32,
        index: u32,
        axis: u32,
    ) -> Result<Option<GpuAxisMotor>, TesseraError> {
        locked(&self.batch)?
            .spherical_motor_environment(environment as usize, index as usize, axis as usize)
            .map(|value| value.map(Into::into))
            .map_err(failed)
    }

    /// Set or clear a spherical motor; invalid indices and parameters leave settings unchanged.
    pub fn set_spherical_motor_environment(
        &self,
        environment: u32,
        index: u32,
        axis: u32,
        motor: Option<GpuAxisMotor>,
    ) -> Result<(), TesseraError> {
        locked(&self.batch)?
            .set_spherical_motor_environment(
                environment as usize,
                index as usize,
                axis as usize,
                motor.map(Into::into),
            )
            .map_err(failed)
    }

    /// Read a spherical servo; ball-joint insertion index and body-A-local X/Y/Z axis 0/1/2.
    pub fn spherical_servo_environment(
        &self,
        environment: u32,
        index: u32,
        axis: u32,
    ) -> Result<Option<GpuAxisServo>, TesseraError> {
        locked(&self.batch)?
            .spherical_servo_environment(environment as usize, index as usize, axis as usize)
            .map(|value| value.map(Into::into))
            .map_err(failed)
    }

    /// Set or clear a spherical servo; invalid indices and parameters leave settings unchanged.
    pub fn set_spherical_servo_environment(
        &self,
        environment: u32,
        index: u32,
        axis: u32,
        servo: Option<GpuAxisServo>,
    ) -> Result<(), TesseraError> {
        locked(&self.batch)?
            .set_spherical_servo_environment(
                environment as usize,
                index as usize,
                axis as usize,
                servo.map(Into::into),
            )
            .map_err(failed)
    }

    /// Read a spherical limit; ball-joint insertion index and body-A-local X/Y/Z axis 0/1/2.
    pub fn spherical_limit_environment(
        &self,
        environment: u32,
        index: u32,
        axis: u32,
    ) -> Result<Option<GpuSphericalLimit>, TesseraError> {
        locked(&self.batch)?
            .spherical_limit_environment(environment as usize, index as usize, axis as usize)
            .map(|value| value.map(Into::into))
            .map_err(failed)
    }

    /// Set or clear a spherical limit; invalid indices and parameters leave settings unchanged.
    pub fn set_spherical_limit_environment(
        &self,
        environment: u32,
        index: u32,
        axis: u32,
        limit: Option<GpuSphericalLimit>,
    ) -> Result<(), TesseraError> {
        locked(&self.batch)?
            .set_spherical_limit_environment(
                environment as usize,
                index as usize,
                axis as usize,
                limit.map(Into::into),
            )
            .map_err(failed)
    }
}
