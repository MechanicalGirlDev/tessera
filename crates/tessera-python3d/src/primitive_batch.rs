//! Python-facing independent mixed-primitive GPU environments.

use super::*;
use tessera_physics::gpu_rigid_sphere_world::{
    GpuRigidPrimitiveBatch as CoreGpuPrimitiveBatch, GpuRigidPrimitiveEnvironment,
};

/// Mixed-primitive environments packed into one GPU state and solver dispatch.
#[derive(Debug, uniffi::Object)]
pub struct GpuPrimitiveBatch {
    batch: Mutex<CoreGpuPrimitiveBatch>,
    shapes: Mutex<Vec<Vec<GpuPrimitiveShape>>>,
}

#[uniffi::export]
impl GpuPrimitiveBatch {
    /// Set a shared linear speed limit for all environments.
    pub fn set_max_linear_speed(&self, max_speed: Option<f32>) -> Result<(), TesseraError> {
        locked(&self.batch)?
            .world_mut()
            .set_max_linear_speed(max_speed)
            .map_err(failed)
    }

    /// Current shared linear speed limit, or None when disabled.
    pub fn max_linear_speed(&self) -> Result<Option<f32>, TesseraError> {
        Ok(locked(&self.batch)?.world().max_linear_speed())
    }

    /// Read contact impulse history with environment-local body IDs.
    pub fn readback_contact_impulses_environment(
        &self,
        environment: u32,
    ) -> Result<GpuContactImpulseReadback, TesseraError> {
        Ok(locked(&self.batch)?
            .readback_contact_impulses_environment(environment as usize)
            .map_err(failed)?
            .into())
    }

    /// Construct independent environments with shared gravity and finite ground.
    #[uniffi::constructor]
    pub fn new(
        environments: Vec<Vec<GpuPrimitiveBody>>,
        gravity: Vec3,
        ground_half_extent: f32,
    ) -> Result<Arc<Self>, TesseraError> {
        let shapes = environments
            .iter()
            .map(|bodies| bodies.iter().map(|body| body.shape.clone()).collect())
            .collect();
        let groups = environments
            .into_iter()
            .map(convert_bodies)
            .collect::<Result<Vec<_>, _>>()?;
        let config = gpu_config(gravity, ground_half_extent)?;
        let gpu = GpuContactDevice::new().map_err(failed)?;
        let environments = groups
            .iter()
            .map(|(states, shapes)| GpuRigidPrimitiveEnvironment { states, shapes })
            .collect::<Vec<_>>();
        let batch =
            CoreGpuPrimitiveBatch::new_primitives(gpu.device(), gpu.queue(), &environments, config)
                .map_err(failed)?;
        Ok(Arc::new(Self {
            batch: Mutex::new(batch),
            shapes: Mutex::new(shapes),
        }))
    }

    /// Number of independent environments.
    pub fn len(&self) -> Result<u32, TesseraError> {
        u32::try_from(locked(&self.batch)?.len()).map_err(failed)
    }

    /// Whether no environments were provided.
    pub fn is_empty(&self) -> Result<bool, TesseraError> {
        Ok(locked(&self.batch)?.is_empty())
    }

    /// Append one independent mixed-primitive environment.
    pub fn add_environment(&self, bodies: Vec<GpuPrimitiveBody>) -> Result<u32, TesseraError> {
        let stored_shapes = bodies.iter().map(|body| body.shape.clone()).collect();
        let (states, colliders): (Vec<_>, Vec<_>) = bodies
            .into_iter()
            .map(gpu_primitive)
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .unzip();
        let mut batch = locked(&self.batch)?;
        let mut shapes = locked(&self.shapes)?;
        let index = batch
            .append_environment_primitives(&states, &colliders)
            .map_err(failed)?;
        shapes.push(stored_shapes);
        u32::try_from(index).map_err(failed)
    }

    /// Remove an environment and return its last body states and shapes.
    pub fn remove_environment(
        &self,
        environment: u32,
    ) -> Result<Vec<RemovedGpuPrimitive>, TesseraError> {
        let mut batch = locked(&self.batch)?;
        let mut shapes = locked(&self.shapes)?;
        let states = batch
            .remove_environment_primitives(environment as usize)
            .map_err(failed)?;
        let removed_shapes = shapes.remove(environment as usize);
        Ok(states
            .into_iter()
            .zip(removed_shapes)
            .map(|((state, _), shape)| RemovedGpuPrimitive {
                state: state.into(),
                shape,
            })
            .collect())
    }

    /// Discard one environment without reading body states back.
    pub fn discard_environment(&self, environment: u32) -> Result<(), TesseraError> {
        let mut batch = locked(&self.batch)?;
        let mut shapes = locked(&self.shapes)?;
        batch
            .discard_environment_primitives(environment as usize)
            .map_err(failed)?;
        let _removed = shapes.remove(environment as usize);
        Ok(())
    }

    /// Append a primitive to one environment and return its local body index.
    pub fn add_body(&self, environment: u32, body: GpuPrimitiveBody) -> Result<u32, TesseraError> {
        let shape = body.shape.clone();
        let (state, collider) = gpu_primitive(body)?;
        let mut batch = locked(&self.batch)?;
        let mut shapes = locked(&self.shapes)?;
        let index = batch
            .append_primitive_environment(environment as usize, state, collider)
            .map_err(failed)?;
        shapes
            .get_mut(environment as usize)
            .ok_or_else(|| failed("environment index out of range"))?
            .push(shape);
        u32::try_from(index).map_err(failed)
    }

    /// Remove an environment-local primitive and return its last state and shape.
    pub fn remove_body(
        &self,
        environment: u32,
        body: u32,
    ) -> Result<RemovedGpuPrimitive, TesseraError> {
        let mut batch = locked(&self.batch)?;
        let mut shapes = locked(&self.shapes)?;
        let (state, _collider) = batch
            .remove_primitive_environment(environment as usize, body as usize)
            .map_err(failed)?;
        let shape = shapes
            .get_mut(environment as usize)
            .ok_or_else(|| failed("environment index out of range"))?
            .remove(body as usize);
        Ok(RemovedGpuPrimitive {
            state: state.into(),
            shape,
        })
    }

    /// Discard an environment-local primitive without reading its GPU state back.
    pub fn discard_body(&self, environment: u32, body: u32) -> Result<(), TesseraError> {
        let mut batch = locked(&self.batch)?;
        let mut shapes = locked(&self.shapes)?;
        batch
            .discard_primitive_environment(environment as usize, body as usize)
            .map_err(failed)?;
        let _removed = shapes[environment as usize].remove(body as usize);
        Ok(())
    }

    /// Replace one environment's point-to-point joints using local body IDs.
    pub fn set_ball_joints_environment(
        &self,
        environment: u32,
        joints: Vec<GpuBallJoint>,
    ) -> Result<(), TesseraError> {
        let joints = joints.into_iter().map(Into::into).collect::<Vec<_>>();
        locked(&self.batch)?
            .set_ball_joints_environment(environment as usize, &joints)
            .map_err(failed)
    }

    /// One environment's point-to-point joints using local body IDs.
    pub fn ball_joints_environment(
        &self,
        environment: u32,
    ) -> Result<Vec<GpuBallJoint>, TesseraError> {
        Ok(locked(&self.batch)?
            .ball_joints_environment(environment as usize)
            .map_err(failed)?
            .into_iter()
            .map(Into::into)
            .collect())
    }

    /// Replace one environment's fixed joints using local body IDs.
    pub fn set_fixed_joints_environment(
        &self,
        environment: u32,
        joints: Vec<GpuFixedJoint>,
    ) -> Result<(), TesseraError> {
        let joints = joints.into_iter().map(Into::into).collect::<Vec<_>>();
        locked(&self.batch)?
            .set_fixed_joints_environment(environment as usize, &joints)
            .map_err(failed)
    }

    /// One environment's fixed joints using local body IDs.
    pub fn fixed_joints_environment(
        &self,
        environment: u32,
    ) -> Result<Vec<GpuFixedJoint>, TesseraError> {
        Ok(locked(&self.batch)?
            .fixed_joints_environment(environment as usize)
            .map_err(failed)?
            .into_iter()
            .map(Into::into)
            .collect())
    }

    /// Replace one environment's revolute joints using local body IDs.
    pub fn set_revolute_joints_environment(
        &self,
        environment: u32,
        joints: Vec<GpuRevoluteJoint>,
    ) -> Result<(), TesseraError> {
        let joints = joints.into_iter().map(Into::into).collect::<Vec<_>>();
        locked(&self.batch)?
            .set_revolute_joints_environment(environment as usize, &joints)
            .map_err(failed)
    }

    /// One environment's revolute joints using local body IDs.
    pub fn revolute_joints_environment(
        &self,
        environment: u32,
    ) -> Result<Vec<GpuRevoluteJoint>, TesseraError> {
        Ok(locked(&self.batch)?
            .revolute_joints_environment(environment as usize)
            .map_err(failed)?
            .into_iter()
            .map(Into::into)
            .collect())
    }

    /// Replace one environment's slider joints using local body IDs.
    pub fn set_prismatic_joints_environment(
        &self,
        environment: u32,
        joints: Vec<GpuPrismaticJoint>,
    ) -> Result<(), TesseraError> {
        let joints = joints.into_iter().map(Into::into).collect::<Vec<_>>();
        locked(&self.batch)?
            .set_prismatic_joints_environment(environment as usize, &joints)
            .map_err(failed)
    }

    /// One environment's slider joints with local body IDs.
    pub fn prismatic_joints_environment(
        &self,
        environment: u32,
    ) -> Result<Vec<GpuPrismaticJoint>, TesseraError> {
        Ok(locked(&self.batch)?
            .prismatic_joints_environment(environment as usize)
            .map_err(failed)?
            .into_iter()
            .map(Into::into)
            .collect())
    }

    /// Set one environment-local hinge's velocity motor.
    pub fn set_revolute_motor_environment(
        &self,
        environment: u32,
        index: u32,
        motor: Option<GpuAxisMotor>,
    ) -> Result<(), TesseraError> {
        locked(&self.batch)?
            .set_revolute_motor_environment(
                environment as usize,
                index as usize,
                motor.map(Into::into),
            )
            .map_err(failed)
    }

    /// Read one environment-local hinge's velocity motor.
    pub fn revolute_motor_environment(
        &self,
        environment: u32,
        index: u32,
    ) -> Result<Option<GpuAxisMotor>, TesseraError> {
        Ok(locked(&self.batch)?
            .revolute_motor_environment(environment as usize, index as usize)
            .map_err(failed)?
            .map(Into::into))
    }

    /// Set one environment-local slider's velocity motor.
    pub fn set_prismatic_motor_environment(
        &self,
        environment: u32,
        index: u32,
        motor: Option<GpuAxisMotor>,
    ) -> Result<(), TesseraError> {
        locked(&self.batch)?
            .set_prismatic_motor_environment(
                environment as usize,
                index as usize,
                motor.map(Into::into),
            )
            .map_err(failed)
    }

    /// Read one environment-local slider's velocity motor.
    pub fn prismatic_motor_environment(
        &self,
        environment: u32,
        index: u32,
    ) -> Result<Option<GpuAxisMotor>, TesseraError> {
        Ok(locked(&self.batch)?
            .prismatic_motor_environment(environment as usize, index as usize)
            .map_err(failed)?
            .map(Into::into))
    }

    /// Set one environment-local slider's displacement limits.
    pub fn set_prismatic_limit_environment(
        &self,
        environment: u32,
        index: u32,
        limit: Option<GpuPrismaticLimit>,
    ) -> Result<(), TesseraError> {
        locked(&self.batch)?
            .set_prismatic_limit_environment(
                environment as usize,
                index as usize,
                limit.map(Into::into),
            )
            .map_err(failed)
    }

    /// Read one environment-local slider's displacement limits.
    pub fn prismatic_limit_environment(
        &self,
        environment: u32,
        index: u32,
    ) -> Result<Option<GpuPrismaticLimit>, TesseraError> {
        Ok(locked(&self.batch)?
            .prismatic_limit_environment(environment as usize, index as usize)
            .map_err(failed)?
            .map(Into::into))
    }

    /// Set one environment-local hinge's position servo.
    pub fn set_revolute_servo_environment(
        &self,
        environment: u32,
        index: u32,
        servo: Option<GpuAxisServo>,
    ) -> Result<(), TesseraError> {
        locked(&self.batch)?
            .set_revolute_servo_environment(
                environment as usize,
                index as usize,
                servo.map(Into::into),
            )
            .map_err(failed)
    }

    /// Read one environment-local hinge's position servo.
    pub fn revolute_servo_environment(
        &self,
        environment: u32,
        index: u32,
    ) -> Result<Option<GpuAxisServo>, TesseraError> {
        Ok(locked(&self.batch)?
            .revolute_servo_environment(environment as usize, index as usize)
            .map_err(failed)?
            .map(Into::into))
    }

    /// Set one environment-local hinge's wrapped angle limits.
    pub fn set_revolute_limit_environment(
        &self,
        environment: u32,
        index: u32,
        limit: Option<GpuRevoluteLimit>,
    ) -> Result<(), TesseraError> {
        locked(&self.batch)?
            .set_revolute_limit_environment(
                environment as usize,
                index as usize,
                limit.map(Into::into),
            )
            .map_err(failed)
    }

    /// Read one environment-local hinge's angle limits.
    pub fn revolute_limit_environment(
        &self,
        environment: u32,
        index: u32,
    ) -> Result<Option<GpuRevoluteLimit>, TesseraError> {
        Ok(locked(&self.batch)?
            .revolute_limit_environment(environment as usize, index as usize)
            .map_err(failed)?
            .map(Into::into))
    }

    /// Read one environment-local hinge's continuous angle.
    pub fn readback_revolute_angle_environment(
        &self,
        environment: u32,
        index: u32,
    ) -> Result<f32, TesseraError> {
        locked(&self.batch)?
            .readback_revolute_angle_environment(environment as usize, index as usize)
            .map_err(failed)
    }

    /// Set one environment-local slider's position servo.
    pub fn set_prismatic_servo_environment(
        &self,
        environment: u32,
        index: u32,
        servo: Option<GpuAxisServo>,
    ) -> Result<(), TesseraError> {
        locked(&self.batch)?
            .set_prismatic_servo_environment(
                environment as usize,
                index as usize,
                servo.map(Into::into),
            )
            .map_err(failed)
    }

    /// Read one environment-local slider's position servo.
    pub fn prismatic_servo_environment(
        &self,
        environment: u32,
        index: u32,
    ) -> Result<Option<GpuAxisServo>, TesseraError> {
        Ok(locked(&self.batch)?
            .prismatic_servo_environment(environment as usize, index as usize)
            .map_err(failed)?
            .map(Into::into))
    }

    /// Advance every environment by one substep.
    pub fn step(&self, dt: f32) -> Result<u32, TesseraError> {
        let count = locked(&self.batch)?.step(dt).map_err(failed)?;
        u32::try_from(count).map_err(failed)
    }

    /// Advance every environment through several substeps.
    pub fn step_substeps(&self, dt: f32, count: u32) -> Result<u32, TesseraError> {
        let candidates = locked(&self.batch)?
            .step_substeps(dt, count)
            .map_err(failed)?;
        u32::try_from(candidates).map_err(failed)
    }

    /// Advance every environment through one temporal frame with one force capture.
    /// `None` selects defaults; joint settings follow contact settings by default.
    pub fn step_temporal(
        &self,
        frame_dt: f32,
        substeps: u32,
        settings: Option<GpuTemporalSettings>,
    ) -> Result<u32, TesseraError> {
        let mut batch = locked(&self.batch)?;
        temporal_step_binding(batch.world_mut(), frame_dt, substeps, settings, None)
    }

    /// Advance sphere-only environments using their shared linear speed cap.
    pub fn step_temporal_speed_bounded(
        &self,
        frame_dt: f32,
        substeps: u32,
        settings: Option<GpuTemporalSettings>,
        joint_settings: Option<GpuTemporalJointSettings>,
    ) -> Result<u32, TesseraError> {
        let mut batch = locked(&self.batch)?;
        temporal_step_speed_bounded_binding(
            batch.world_mut(),
            frame_dt,
            substeps,
            settings,
            joint_settings,
        )
    }

    /// Advance a temporal frame with independent contact and joint coefficients.
    pub fn step_temporal_with_joints(
        &self,
        frame_dt: f32,
        substeps: u32,
        settings: Option<GpuTemporalSettings>,
        joint_settings: GpuTemporalJointSettings,
    ) -> Result<u32, TesseraError> {
        let mut batch = locked(&self.batch)?;
        temporal_step_binding(
            batch.world_mut(),
            frame_dt,
            substeps,
            settings,
            Some(joint_settings),
        )
    }

    /// Set environment-local prescribed motion; None restores static behavior.
    /// Dynamic bodies ignore the command. Nonfinite velocities are rejected.
    pub fn set_kinematic_motion(
        &self,
        environment: u32,
        body: u32,
        motion: Option<GpuKinematicMotion>,
    ) -> Result<(), TesseraError> {
        locked(&self.batch)?
            .set_kinematic_motion(
                environment as usize,
                body as usize,
                motion.map(GpuKinematicMotion::gpu),
            )
            .map_err(failed)
    }

    /// Apply world-space force and torque to one body in one environment.
    pub fn write_wrench(
        &self,
        environment: u32,
        body: u32,
        force: Vec3,
        torque: Vec3,
    ) -> Result<(), TesseraError> {
        let [x, y, z] = force.gpu();
        let [tx, ty, tz] = torque.gpu();
        locked(&self.batch)?
            .write_forces(
                environment as usize,
                body as usize,
                GpuRigidBodyForces {
                    force: [x, y, z, 0.0],
                    torque: [tx, ty, tz, 0.0],
                },
            )
            .map_err(failed)
    }

    /// Override one body's contact material.
    pub fn set_body_material(
        &self,
        environment: u32,
        body: u32,
        material: Material,
    ) -> Result<(), TesseraError> {
        let mut batch = locked(&self.batch)?;
        let index = body_index(&batch, environment, body)?;
        batch
            .world_mut()
            .set_body_material(index, material.into())
            .map_err(failed)
    }

    /// Restore one body's configured default material.
    pub fn clear_body_material(&self, environment: u32, body: u32) -> Result<(), TesseraError> {
        let mut batch = locked(&self.batch)?;
        let index = body_index(&batch, environment, body)?;
        batch.world_mut().clear_body_material(index).map_err(failed)
    }

    /// Set reciprocal collision masks for one environment-local primitive.
    pub fn set_body_collision_groups(
        &self,
        environment: u32,
        body: u32,
        groups: GpuCollisionGroups,
    ) -> Result<(), TesseraError> {
        locked(&self.batch)?
            .set_body_collision_groups_environment(
                environment as usize,
                body as usize,
                groups.into(),
            )
            .map_err(failed)
    }

    /// Read reciprocal collision masks for one environment-local primitive.
    pub fn body_collision_groups(
        &self,
        environment: u32,
        body: u32,
    ) -> Result<GpuCollisionGroups, TesseraError> {
        locked(&self.batch)?
            .body_collision_groups_environment(environment as usize, body as usize)
            .map(Into::into)
            .ok_or_else(|| failed("body index out of range"))
    }

    /// Override the finite ground material shared by all environments.
    pub fn set_ground_material(&self, material: Material) -> Result<(), TesseraError> {
        locked(&self.batch)?
            .world_mut()
            .set_ground_material(material.into())
            .map_err(failed)
    }

    /// Restore the finite ground's configured default material.
    pub fn clear_ground_material(&self) -> Result<(), TesseraError> {
        locked(&self.batch)?
            .world_mut()
            .clear_ground_material()
            .map_err(failed)
    }

    /// Set reciprocal collision masks for the shared finite ground.
    pub fn set_ground_collision_groups(
        &self,
        groups: GpuCollisionGroups,
    ) -> Result<(), TesseraError> {
        locked(&self.batch)?
            .world_mut()
            .set_ground_collision_groups(groups.into())
            .map_err(failed)
    }

    /// Cast rays in one environment; hit and exclusion IDs are environment-local.
    pub fn cast_rays_environment(
        &self,
        environment: u32,
        rays: Vec<GpuRay>,
    ) -> Result<Vec<Option<GpuRayHit>>, TesseraError> {
        let hits = locked(&self.batch)?
            .cast_rays_environment(environment as usize, &gpu_rays(rays))
            .map_err(failed)?;
        Ok(gpu_ray_hits(hits))
    }

    /// Project points in one environment; hit and exclusion IDs are environment-local.
    pub fn project_points_environment(
        &self,
        environment: u32,
        points: Vec<GpuPointQuery>,
    ) -> Result<Vec<Option<GpuPointHit>>, TesseraError> {
        let hits = locked(&self.batch)?
            .project_points_environment(environment as usize, &gpu_points(points))
            .map_err(failed)?;
        Ok(gpu_point_hits(hits))
    }

    /// Evaluate rays and points in one environment with one GPU scene tree.
    pub fn query_scene_environment(
        &self,
        environment: u32,
        rays: Vec<GpuRay>,
        points: Vec<GpuPointQuery>,
    ) -> Result<GpuSceneQueryHits, TesseraError> {
        let hits = locked(&self.batch)?
            .query_scene_environment(environment as usize, &gpu_rays(rays), &gpu_points(points))
            .map_err(failed)?;
        Ok(gpu_scene_query_hits(hits))
    }

    /// Evaluate all environments against one current-state GPU scene snapshot.
    pub fn query_scene_environments(
        &self,
        rays: Vec<Vec<GpuRay>>,
        points: Vec<Vec<GpuPointQuery>>,
    ) -> Result<Vec<GpuSceneQueryHits>, TesseraError> {
        let batch = locked(&self.batch)?;
        gpu_scene_query_hits_environments(&batch, rays, points)
    }

    /// Read one environment without transferring other environments' states.
    pub fn readback_environment(
        &self,
        environment: u32,
    ) -> Result<Vec<GpuPrimitiveState>, TesseraError> {
        Ok(locked(&self.batch)?
            .readback_environment(environment as usize)
            .map_err(failed)?
            .into_iter()
            .map(Into::into)
            .collect())
    }

    /// Read active contacts with body indices local to one environment.
    ///
    /// The GPU transfers all batch contacts before selecting this environment.
    pub fn readback_contacts_environment(
        &self,
        environment: u32,
    ) -> Result<Vec<GpuPrimitiveContact>, TesseraError> {
        let readback = locked(&self.batch)?
            .readback_contacts_environment(environment as usize)
            .map_err(failed)?;
        primitive_contacts(readback)
    }

    /// Reset one environment while retaining its shapes and every other state.
    pub fn reset_environment(
        &self,
        environment: u32,
        bodies: Vec<GpuPrimitiveBody>,
    ) -> Result<(), TesseraError> {
        let (states, shapes) = convert_bodies(bodies)?;
        let mut batch = locked(&self.batch)?;
        let range = batch
            .environment_range(environment as usize)
            .ok_or_else(|| failed("environment index out of range"))?;
        if shapes.len() != range.len()
            || shapes.iter().enumerate().any(|(index, shape)| {
                batch.world().shape(range.start + index).as_ref() != Some(shape)
            })
        {
            return Err(failed("environment body count or primitive shape changed"));
        }
        batch
            .reset_environment(environment as usize, &states)
            .map_err(failed)
    }

    /// Reset every environment, including after a failed GPU step.
    pub fn reset_all(&self, environments: Vec<Vec<GpuPrimitiveBody>>) -> Result<(), TesseraError> {
        let groups = environments
            .into_iter()
            .map(convert_bodies)
            .collect::<Result<Vec<_>, _>>()?;
        let mut batch = locked(&self.batch)?;
        if groups.len() != batch.len() {
            return Err(failed("environment count changed"));
        }
        let mut states = Vec::new();
        for (environment, (next_states, shapes)) in groups.into_iter().enumerate() {
            let range = batch
                .environment_range(environment)
                .ok_or_else(|| failed("environment index out of range"))?;
            if shapes.len() != range.len()
                || shapes.iter().enumerate().any(|(index, shape)| {
                    batch.world().shape(range.start + index).as_ref() != Some(shape)
                })
            {
                return Err(failed("environment body count or primitive shape changed"));
            }
            states.extend(next_states);
        }
        batch.reset_all(&states).map_err(failed)
    }
}

fn convert_bodies(
    bodies: Vec<GpuPrimitiveBody>,
) -> Result<(Vec<GpuRigidBodyState>, Vec<GpuRigidShape>), TesseraError> {
    let (states, shapes) = bodies
        .into_iter()
        .map(gpu_primitive)
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .unzip();
    Ok((states, shapes))
}

fn body_index(
    batch: &CoreGpuPrimitiveBatch,
    environment: u32,
    body: u32,
) -> Result<usize, TesseraError> {
    let range = batch
        .environment_range(environment as usize)
        .ok_or_else(|| failed("environment index out of range"))?;
    if body as usize >= range.len() {
        return Err(failed("body index out of range"));
    }
    Ok(range.start + body as usize)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn v(x: f64, y: f64, z: f64) -> Vec3 {
        Vec3 { x, y, z }
    }

    fn body(shape: GpuPrimitiveShape, x: f64) -> GpuPrimitiveBody {
        GpuPrimitiveBody {
            shape,
            center: v(x, 0.0, 3.0),
            orientation: Quaternion {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 1.0,
            },
            velocity: v(0.0, 0.0, 0.0),
            angular_velocity: v(0.0, 0.0, 0.0),
            mass: 1.0,
            principal_inertia: v(0.2, 0.2, 0.2),
        }
    }

    #[test]
    fn primitive_batch_resolves_triangle_mesh_per_environment() {
        let Ok(_device) = GpuContactDevice::new() else {
            return;
        };
        let mesh_shape = GpuPrimitiveShape::TriangleMesh {
            vertices: vec![v(-1.0, -1.0, 0.0), v(1.0, -1.0, 0.0), v(0.0, 1.0, 0.0)],
            triangles: vec![GpuTriangle { a: 0, b: 1, c: 2 }],
        };
        let mut mesh = body(mesh_shape, 0.0);
        mesh.mass = 0.0;
        mesh.principal_inertia = v(0.0, 0.0, 0.0);
        let mut near = body(GpuPrimitiveShape::Sphere { radius: 0.5 }, 0.0);
        near.center.z = 3.25;
        let mut far = near.clone();
        far.center.z = 5.0;
        let batch = GpuPrimitiveBatch::new(
            vec![vec![mesh.clone(), near], vec![mesh, far]],
            v(0.0, 0.0, 0.0),
            10.0,
        )
        .unwrap();
        batch.set_max_linear_speed(Some(2.0)).unwrap();
        assert_eq!(batch.max_linear_speed().unwrap(), Some(2.0));
        let _pairs = batch.step(0.01).unwrap();
        let contacts = batch.readback_contacts_environment(0).unwrap();
        assert_eq!(contacts.len(), 1);
        assert_eq!(contacts[0].body_b, Some(1));
        assert!((contacts[0].depth - 0.25).abs() < 1e-3);
        assert!(batch.readback_contacts_environment(1).unwrap().is_empty());
    }

    #[test]
    fn primitive_batch_exposes_environment_reset_and_shape_validation() {
        let Ok(_device) = GpuContactDevice::new() else {
            return;
        };
        let first = vec![
            body(
                GpuPrimitiveShape::Box {
                    half_extents: v(1.0, 1.0, 1.0),
                },
                0.0,
            ),
            body(GpuPrimitiveShape::Sphere { radius: 0.5 }, 1.4),
        ];
        let second = vec![
            body(
                GpuPrimitiveShape::Capsule {
                    radius: 0.5,
                    half_length: 0.5,
                },
                0.0,
            ),
            body(GpuPrimitiveShape::Sphere { radius: 0.5 }, 0.9),
        ];
        let batch =
            GpuPrimitiveBatch::new(vec![first.clone(), second.clone()], v(0.0, 0.0, 0.0), 10.0)
                .unwrap();
        assert_eq!(batch.len().unwrap(), 2);
        let joint = GpuBallJoint {
            body_a: 0,
            body_b: 1,
            local_anchor_a: v(0.5, 0.0, 0.0),
            local_anchor_b: v(-0.5, 0.0, 0.0),
        };
        batch
            .set_ball_joints_environment(0, vec![joint.clone()])
            .unwrap();
        batch
            .set_ball_joints_environment(1, vec![joint.clone()])
            .unwrap();
        assert_eq!(batch.ball_joints_environment(0).unwrap().len(), 1);
        assert_eq!(
            batch.ball_joints_environment(1).unwrap()[0].body_b,
            joint.body_b
        );
        assert_eq!(
            batch
                .add_body(0, body(GpuPrimitiveShape::Sphere { radius: 0.5 }, 100.0))
                .unwrap(),
            2
        );
        assert_eq!(batch.ball_joints_environment(1).unwrap()[0].body_b, 1);
        let _removed = batch.remove_body(0, 2).unwrap();
        assert_eq!(batch.step(0.01).unwrap(), 2);
        for environment in 0..2 {
            let contacts = batch.readback_contacts_environment(environment).unwrap();
            assert_eq!(contacts.len(), 1);
            assert_eq!(contacts[0].body_a, 0);
            assert_eq!(contacts[0].body_b, Some(1));
            assert!(contacts[0].depth > 0.0);
        }
        let untouched = batch.readback_environment(1).unwrap();
        batch.reset_environment(0, first.clone()).unwrap();
        assert!((batch.readback_environment(0).unwrap()[1].center.x - 1.4).abs() < 1e-5);
        assert_eq!(
            batch.readback_environment(1).unwrap()[0].center.x,
            untouched[0].center.x
        );
        let mut changed = first.clone();
        changed[0].shape = GpuPrimitiveShape::Sphere { radius: 1.0 };
        assert!(batch.reset_environment(0, changed).is_err());
        assert!(batch.reset_all(vec![first, second]).is_ok());
        let fixed = GpuFixedJoint {
            body_a: 0,
            body_b: 1,
            local_anchor_a: v(0.5, 0.0, 0.0),
            local_anchor_b: v(-0.5, 0.0, 0.0),
            local_rotation_a: Quaternion {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 1.0,
            },
            local_rotation_b: Quaternion {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 1.0,
            },
        };
        batch
            .set_fixed_joints_environment(0, vec![fixed.clone()])
            .unwrap();
        batch.set_fixed_joints_environment(1, vec![fixed]).unwrap();
        assert_eq!(batch.fixed_joints_environment(0).unwrap().len(), 1);
        assert_eq!(batch.fixed_joints_environment(1).unwrap()[0].body_b, 1);
        let hinge = GpuRevoluteJoint {
            body_a: 0,
            body_b: 1,
            local_anchor_a: v(0.5, 0.0, 0.0),
            local_anchor_b: v(-0.5, 0.0, 0.0),
            local_axis_a: v(0.0, 0.0, 1.0),
            local_axis_b: v(0.0, 0.0, 1.0),
        };
        batch
            .set_revolute_joints_environment(0, vec![hinge.clone()])
            .unwrap();
        batch
            .set_revolute_joints_environment(1, vec![hinge])
            .unwrap();
        assert_eq!(batch.revolute_joints_environment(0).unwrap().len(), 1);
        assert_eq!(batch.revolute_joints_environment(1).unwrap()[0].body_b, 1);
    }

    #[test]
    fn primitive_batch_exposes_environment_local_topology_edits() {
        let Ok(_device) = GpuContactDevice::new() else {
            return;
        };
        let first = body(GpuPrimitiveShape::Sphere { radius: 0.5 }, 0.0);
        let second = body(GpuPrimitiveShape::Sphere { radius: 0.5 }, 4.0);
        let batch = GpuPrimitiveBatch::new(
            vec![vec![first.clone()], vec![second.clone()]],
            v(0.0, 0.0, 0.0),
            10.0,
        )
        .unwrap();
        let added = body(
            GpuPrimitiveShape::Box {
                half_extents: v(0.5, 0.5, 0.5),
            },
            8.0,
        );
        assert_eq!(batch.add_body(0, added.clone()).unwrap(), 1);
        assert_eq!(batch.readback_environment(0).unwrap().len(), 2);
        assert_eq!(batch.readback_environment(1).unwrap()[0].center.x, 4.0);
        batch
            .reset_environment(0, vec![first.clone(), added])
            .unwrap();
        let removed = batch.remove_body(0, 1).unwrap();
        assert_eq!(removed.state.center.x, 8.0);
        assert!(matches!(removed.shape, GpuPrimitiveShape::Box { .. }));
        batch.reset_environment(0, vec![first.clone()]).unwrap();
        batch.reset_all(vec![vec![first], vec![second]]).unwrap();
    }

    #[test]
    fn primitive_batch_discard_keeps_python_shape_cache_aligned() {
        let Ok(_device) = GpuContactDevice::new() else {
            return;
        };
        let first = body(GpuPrimitiveShape::Sphere { radius: 0.5 }, 0.0);
        let second = body(
            GpuPrimitiveShape::Box {
                half_extents: v(0.5, 0.5, 0.5),
            },
            4.0,
        );
        let batch = GpuPrimitiveBatch::new(
            vec![vec![first.clone()], vec![second.clone(), first.clone()]],
            v(0.0, 0.0, 0.0),
            10.0,
        )
        .unwrap();
        batch.discard_environment(0).unwrap();
        batch.discard_body(0, 0).unwrap();
        assert_eq!(batch.len().unwrap(), 1);
        assert_eq!(batch.readback_environment(0).unwrap()[0].center.x, 0.0);
        let removed = batch.remove_body(0, 0).unwrap();
        assert!(matches!(removed.shape, GpuPrimitiveShape::Sphere { .. }));
        assert!(!batch.is_empty().unwrap());
        assert_eq!(batch.add_environment(vec![second]).unwrap(), 1);
        let query = GpuPointQuery {
            point: v(2.0, 0.0, 3.0),
            max_distance: 10.0,
            groups: GpuCollisionGroups {
                memberships: u32::MAX,
                filter: u32::MAX,
            },
            excluded_body: None,
            body_range: None,
            solid: false,
        };
        let hits = batch
            .query_scene_environments(vec![vec![], vec![]], vec![vec![query.clone()], vec![query]])
            .unwrap();
        assert_eq!(hits[0].points[0].as_ref().unwrap().body, None);
        assert_eq!(hits[1].points[0].as_ref().unwrap().body, Some(0));
    }
}
