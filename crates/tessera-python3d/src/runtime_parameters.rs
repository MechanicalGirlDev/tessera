//! Runtime rigid-world settings; updates preserve resident poses and velocities.
use super::GpuPrimitiveBatch;
use crate::*;
use tessera_physics::gpu_rigid_sphere_solver::GpuRigidSphereSolveParams;

/// Default contact material, stabilization and iteration count.
#[derive(Clone, Copy, Debug, uniffi::Record)]
pub struct GpuSolveParameters {
    /// Coulomb coefficient for colliders without a material override.
    pub friction: f32,
    /// Restitution for colliders without a material override.
    pub restitution: f32,
    /// Fraction of penetration corrected per substep.
    pub bias_factor: f32,
    /// Sequential impulse sweeps per substep.
    pub iterations: u32,
}
impl From<GpuSolveParameters> for GpuRigidSphereSolveParams {
    fn from(value: GpuSolveParameters) -> Self {
        Self {
            friction: value.friction,
            restitution: value.restitution,
            bias_factor: value.bias_factor,
            iterations: value.iterations,
        }
    }
}
impl From<GpuRigidSphereSolveParams> for GpuSolveParameters {
    fn from(value: GpuRigidSphereSolveParams) -> Self {
        Self {
            friction: value.friction,
            restitution: value.restitution,
            bias_factor: value.bias_factor,
            iterations: value.iterations,
        }
    }
}
fn gpu_gravity(gravity: Vec3) -> Result<[f32; 3], TesseraError> {
    let values = [gravity.x, gravity.y, gravity.z];
    let converted = values.map(|value| value as f32);
    if values.iter().zip(&converted).any(|(&original, &value)| {
        !original.is_finite() || !value.is_finite() || (original != 0.0 && value == 0.0)
    }) {
        return Err(failed("gravity cannot be represented in resident f32"));
    }
    Ok(converted)
}
#[uniffi::export]
impl SphereWorld {
    /// Current world gravitational acceleration.
    pub fn gravity(&self) -> Result<Vec3, TesseraError> {
        let [x, y, z] = locked(&self.world)?.params().gravity;
        Ok(Vec3 { x, y, z })
    }
    /// Change gravity without replacing body state.
    pub fn set_gravity(&self, gravity: Vec3) -> Result<(), TesseraError> {
        locked(&self.world)?
            .set_gravity([gravity.x, gravity.y, gravity.z])
            .map_err(failed)
    }
}

#[uniffi::export]
impl GpuSphereWorld {
    /// Shared world gravity in metres per second squared.
    pub fn gravity(&self) -> Result<Vec3, TesseraError> {
        let [x, y, z] = locked(&self.world)?.config().gravity;
        Ok(Vec3 {
            x: f64::from(x),
            y: f64::from(y),
            z: f64::from(z),
        })
    }
    /// Preserve poses and velocities and wake bodies when gravity changes.
    pub fn set_gravity(&self, gravity: Vec3) -> Result<(), TesseraError> {
        let gravity = gpu_gravity(gravity)?;
        locked(&self.world)?.set_gravity(gravity).map_err(failed)
    }
    /// Current default contact solve coefficients.
    pub fn solve_parameters(&self) -> Result<GpuSolveParameters, TesseraError> {
        Ok(locked(&self.world)?.config().solve.into())
    }
    /// Change default contact coefficients; collider overrides stay assigned.
    pub fn set_solve_parameters(&self, parameters: GpuSolveParameters) -> Result<(), TesseraError> {
        locked(&self.world)?
            .set_solve_params(parameters.into())
            .map_err(failed)
    }
}

#[uniffi::export]
impl GpuPrimitiveWorld {
    /// Shared world gravity in metres per second squared.
    pub fn gravity(&self) -> Result<Vec3, TesseraError> {
        let [x, y, z] = locked(&self.inner)?.world.config().gravity;
        Ok(Vec3 {
            x: f64::from(x),
            y: f64::from(y),
            z: f64::from(z),
        })
    }
    /// Preserve poses and velocities and wake bodies when gravity changes.
    pub fn set_gravity(&self, gravity: Vec3) -> Result<(), TesseraError> {
        let gravity = gpu_gravity(gravity)?;
        locked(&self.inner)?
            .world
            .set_gravity(gravity)
            .map_err(failed)
    }
    /// Current default contact solve coefficients.
    pub fn solve_parameters(&self) -> Result<GpuSolveParameters, TesseraError> {
        Ok(locked(&self.inner)?.world.config().solve.into())
    }
    /// Change default contact coefficients; collider overrides stay assigned.
    pub fn set_solve_parameters(&self, parameters: GpuSolveParameters) -> Result<(), TesseraError> {
        locked(&self.inner)?
            .world
            .set_solve_params(parameters.into())
            .map_err(failed)
    }
}

#[uniffi::export]
impl GpuSphereBatch {
    /// Shared world gravity in metres per second squared.
    pub fn gravity(&self) -> Result<Vec3, TesseraError> {
        let [x, y, z] = locked(&self.batch)?.world().config().gravity;
        Ok(Vec3 {
            x: f64::from(x),
            y: f64::from(y),
            z: f64::from(z),
        })
    }
    /// Preserve poses and velocities and wake bodies when gravity changes.
    pub fn set_gravity(&self, gravity: Vec3) -> Result<(), TesseraError> {
        let gravity = gpu_gravity(gravity)?;
        locked(&self.batch)?
            .world_mut()
            .set_gravity(gravity)
            .map_err(failed)
    }
    /// Current default contact solve coefficients.
    pub fn solve_parameters(&self) -> Result<GpuSolveParameters, TesseraError> {
        Ok(locked(&self.batch)?.world().config().solve.into())
    }
    /// Change default contact coefficients; collider overrides stay assigned.
    pub fn set_solve_parameters(&self, parameters: GpuSolveParameters) -> Result<(), TesseraError> {
        locked(&self.batch)?
            .world_mut()
            .set_solve_params(parameters.into())
            .map_err(failed)
    }
}

#[uniffi::export]
impl GpuPrimitiveBatch {
    /// Shared world gravity in metres per second squared.
    pub fn gravity(&self) -> Result<Vec3, TesseraError> {
        let [x, y, z] = locked(&self.batch)?.world().config().gravity;
        Ok(Vec3 {
            x: f64::from(x),
            y: f64::from(y),
            z: f64::from(z),
        })
    }
    /// Preserve poses and velocities and wake bodies when gravity changes.
    pub fn set_gravity(&self, gravity: Vec3) -> Result<(), TesseraError> {
        let gravity = gpu_gravity(gravity)?;
        locked(&self.batch)?
            .world_mut()
            .set_gravity(gravity)
            .map_err(failed)
    }
    /// Current default contact solve coefficients.
    pub fn solve_parameters(&self) -> Result<GpuSolveParameters, TesseraError> {
        Ok(locked(&self.batch)?.world().config().solve.into())
    }
    /// Change default contact coefficients; collider overrides stay assigned.
    pub fn set_solve_parameters(&self, parameters: GpuSolveParameters) -> Result<(), TesseraError> {
        locked(&self.batch)?
            .world_mut()
            .set_solve_params(parameters.into())
            .map_err(failed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vector(x: f64, y: f64, z: f64) -> Vec3 {
        Vec3 { x, y, z }
    }
    fn sphere() -> SphereInput {
        SphereInput {
            center: vector(0.0, 0.0, 3.0),
            velocity: vector(0.0, 0.0, 0.0),
            radius: 0.1,
            mass: 1.0,
        }
    }
    fn primitive() -> GpuPrimitiveBody {
        GpuPrimitiveBody {
            shape: GpuPrimitiveShape::Sphere { radius: 0.1 },
            center: vector(0.0, 0.0, 3.0),
            orientation: Quaternion {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 1.0,
            },
            velocity: vector(0.0, 0.0, 0.0),
            angular_velocity: vector(0.0, 0.0, 0.0),
            mass: 1.0,
            principal_inertia: vector(1.0, 1.0, 1.0),
        }
    }

    #[test]
    fn runtime_cpu_gravity_preserves_state_wakes_and_rejects_invalid_updates()
    -> Result<(), TesseraError> {
        let world = SphereWorld::new(
            vec![sphere()],
            vector(0.0, 0.0, 0.0),
            10.0,
            0.0,
            0.0,
            0.001,
            1,
        )?;
        locked(&world.world)?.sleep_body(0).map_err(failed)?;
        world.set_gravity(vector(2.0, 0.0, 0.0))?;
        assert_eq!(locked(&world.world)?.body_is_sleeping(0), Some(false));
        assert_eq!(world.states()?[0].center.z, 3.0);
        assert!(world.set_gravity(vector(f64::NAN, 0.0, 0.0)).is_err());
        assert_eq!(world.gravity()?.x, 2.0);
        world.step(0.001)?;
        assert!((world.states()?[0].velocity.x - 0.002).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn runtime_gpu_settings_preserve_state_and_drive_single_and_batch_worlds()
    -> Result<(), TesseraError> {
        let sphere_world = GpuSphereWorld::new(vec![sphere()], vector(0.0, 0.0, 0.0), 10.0)?;
        let primitive_world =
            GpuPrimitiveWorld::new(vec![primitive()], vector(0.0, 0.0, 0.0), 10.0)?;
        let sphere_batch =
            GpuSphereBatch::new(vec![vec![sphere()]; 2], vector(0.0, 0.0, 0.0), 10.0)?;
        let primitive_batch =
            GpuPrimitiveBatch::new(vec![vec![primitive()]; 2], vector(0.0, 0.0, 0.0), 10.0)?;
        let parameters = GpuSolveParameters {
            friction: 0.5,
            restitution: 0.1,
            bias_factor: 0.3,
            iterations: 3,
        };
        sphere_world.set_gravity(vector(2.0, 0.0, 0.0))?;
        primitive_world.set_gravity(vector(2.0, 0.0, 0.0))?;
        sphere_batch.set_gravity(vector(2.0, 0.0, 0.0))?;
        primitive_batch.set_gravity(vector(2.0, 0.0, 0.0))?;
        sphere_world.set_solve_parameters(parameters)?;
        primitive_world.set_solve_parameters(parameters)?;
        sphere_batch.set_solve_parameters(parameters)?;
        primitive_batch.set_solve_parameters(parameters)?;
        assert_eq!(sphere_world.readback()?[0].center.z, 3.0);
        assert_eq!(primitive_world.readback()?[0].center.z, 3.0);
        assert_eq!(sphere_batch.gravity()?.x, 2.0);
        assert_eq!(primitive_batch.solve_parameters()?.iterations, 3);
        assert!(
            sphere_world
                .set_gravity(vector(f64::MAX, 0.0, 0.0))
                .is_err()
        );
        assert!(
            primitive_batch
                .set_solve_parameters(GpuSolveParameters {
                    iterations: 0,
                    ..parameters
                })
                .is_err()
        );
        assert_eq!(primitive_batch.solve_parameters()?.iterations, 3);
        let _pairs = sphere_world.step(0.001)?;
        let _pairs = primitive_world.step(0.001)?;
        let _pairs = sphere_batch.step(0.001)?;
        let _pairs = primitive_batch.step(0.001)?;
        assert!((sphere_world.readback()?[0].velocity.x - 0.002).abs() < 1e-6);
        assert!((primitive_world.readback()?[0].velocity.x - 0.002).abs() < 1e-6);
        for environment in 0..2 {
            assert!(
                (sphere_batch.readback_environment(environment)?[0]
                    .velocity
                    .x
                    - 0.002)
                    .abs()
                    < 1e-6
            );
            assert!(
                (primitive_batch.readback_environment(environment)?[0]
                    .velocity
                    .x
                    - 0.002)
                    .abs()
                    < 1e-6
            );
        }
        Ok(())
    }
}
