//! Independent rigid-sphere reference world using Tessera contact impulses.
//!
//! This is a small-scene correctness model for the future articulated backend.

use nalgebra::{DMatrix, DVector, Vector3};

use crate::contact_reference::{
    ContactConstraint, ContactProblem, ContactSolveError, SolveParams, solve_contacts,
};
#[cfg(feature = "gpu-contact")]
use crate::gpu_broad_phase::GpuPair;
#[cfg(feature = "gpu-contact")]
use crate::gpu_contact_pipeline::{ContactPipelineError, GpuContactDevice, GpuContactPipeline};
#[cfg(feature = "gpu-contact")]
use crate::gpu_contact_solver::{GpuContactSolveError, GpuContactSolver};
#[cfg(feature = "gpu-contact")]
use crate::gpu_ground_contact::GpuGroundContacts;
#[cfg(feature = "gpu-contact")]
use crate::gpu_sphere_contact::{GpuSphere, GpuSphereContact};
use crate::material::{ColliderMaterial, CombinedMaterial};
use crate::sleep::{SleepSettings, SleepState};

/// A sphere with linear motion and no angular state.
#[derive(Debug, Clone)]
pub struct SphereBody {
    /// Sphere centre in world coordinates, metres.
    pub center: Vector3<f64>,
    /// Linear velocity in world coordinates, metres per second.
    pub velocity: Vector3<f64>,
    /// Sphere radius, metres.
    pub radius: f64,
    /// Body mass, kilograms. Zero means static.
    pub mass: f64,
}

/// Parameters for a finite square ground plane and fixed-step integration.
#[derive(Debug, Clone, Copy)]
pub struct SphereWorldParams {
    /// Gravity in world coordinates, metres per second squared.
    pub gravity: [f64; 3],
    /// Half-extent of the ground in X and Y, metres.
    pub ground_half_extent: f64,
    /// Coulomb friction for every contact.
    pub friction: f64,
    /// Non-negative normal coefficient of restitution for every contact.
    pub restitution: f64,
    /// Maximum integration substep in seconds.
    pub max_substep: f64,
    /// Contact solver iterations per substep.
    pub solver_iterations: usize,
    /// Automatic sleep and wake thresholds for dynamic bodies.
    pub sleep: SleepSettings,
}

impl Default for SphereWorldParams {
    fn default() -> Self {
        Self {
            gravity: [0.0, 0.0, -9.81],
            ground_half_extent: 50.0,
            friction: 1.0,
            restitution: 0.0,
            max_substep: 0.001,
            solver_iterations: 12,
            sleep: SleepSettings::default(),
        }
    }
}

/// Invalid body/world inputs or a contact solve failure.
#[derive(Debug, thiserror::Error)]
pub enum SphereWorldError {
    /// A world parameter, body state, or step size is invalid.
    #[error("invalid sphere world input")]
    InvalidInput,
    /// The number of bodies exceeds the generalized-coordinate capacity.
    #[error("sphere world capacity exceeded")]
    Capacity,
    /// Contact impulse solve failed.
    #[error("sphere world contact solve failed: {0}")]
    Contact(#[from] ContactSolveError),
    /// GPU detection failed or the world cannot be represented in GPU `f32`.
    #[cfg(feature = "gpu-contact")]
    #[error("sphere world GPU detection failed: {0}")]
    Gpu(#[from] ContactPipelineError),
    /// GPU contact impulse solve failed.
    #[cfg(feature = "gpu-contact")]
    #[error("sphere world GPU contact solve failed: {0}")]
    GpuSolve(#[from] GpuContactSolveError),
}

/// Deterministic CPU reference world with sphere-sphere and sphere-ground contact.
#[derive(Debug)]
pub struct SphereWorld {
    /// Bodies in stable insertion order.
    pub bodies: Vec<SphereBody>,
    params: SphereWorldParams,
    body_materials: Vec<ColliderMaterial>,
    ground_material: ColliderMaterial,
    max_linear_speed: Option<f64>,
    sleep_states: Vec<SleepState>,
    contact_forces: Vec<Vector3<f64>>,
}

impl SphereWorld {
    /// Immutable integration parameters used by this world.
    pub fn params(&self) -> &SphereWorldParams {
        &self.params
    }

    /// Construct a world after validating every parameter and body.
    pub fn new(
        bodies: Vec<SphereBody>,
        params: SphereWorldParams,
    ) -> Result<Self, SphereWorldError> {
        if bodies.len() > usize::MAX / 3
            || params.gravity.iter().any(|x| !x.is_finite())
            || !params.ground_half_extent.is_finite()
            || params.ground_half_extent <= 0.0
            || !params.friction.is_finite()
            || params.friction < 0.0
            || !params.restitution.is_finite()
            || params.restitution < 0.0
            || !params.max_substep.is_finite()
            || params.max_substep <= 0.0
            || params.solver_iterations == 0
            || !params.sleep.is_valid()
            || bodies.iter().any(|body| {
                body.center
                    .iter()
                    .chain(body.velocity.iter())
                    .any(|x| !x.is_finite())
                    || !body.radius.is_finite()
                    || body.radius <= 0.0
                    || !body.mass.is_finite()
                    || body.mass < 0.0
            })
        {
            return Err(SphereWorldError::InvalidInput);
        }
        let contact_forces = vec![Vector3::zeros(); bodies.len()];
        let default_material = ColliderMaterial::new(params.friction, params.restitution);
        Ok(Self {
            body_materials: vec![default_material; bodies.len()],
            ground_material: default_material,
            max_linear_speed: None,
            sleep_states: vec![SleepState::default(); bodies.len()],
            bodies,
            params,
            contact_forces,
        })
    }

    /// Optional world-space speed limit for dynamic bodies, in metres per second.
    pub fn max_linear_speed(&self) -> Option<f64> {
        self.max_linear_speed
    }

    /// Limit speeds before contact detection and after contact impulses.
    pub fn set_max_linear_speed(&mut self, limit: Option<f64>) -> Result<(), SphereWorldError> {
        if limit.is_some_and(|value| !value.is_finite() || value <= 0.0) {
            return Err(SphereWorldError::InvalidInput);
        }
        self.max_linear_speed = limit;
        Ok(())
    }

    fn limit_velocity(&self, velocity: Vector3<f64>) -> Vector3<f64> {
        if let Some(limit) = self.max_linear_speed {
            let speed = velocity.norm();
            if speed > limit {
                return velocity * (limit / speed);
            }
        }
        velocity
    }

    /// Material assigned to a body, or the world's default for a body inserted
    /// directly through the public body vector.
    pub fn body_material(&self, index: usize) -> Option<ColliderMaterial> {
        (index < self.bodies.len()).then(|| self.material_for_body(index))
    }

    /// Assign a contact material to one body.
    pub fn set_body_material(
        &mut self,
        index: usize,
        material: ColliderMaterial,
    ) -> Result<(), SphereWorldError> {
        if index >= self.bodies.len() || !material.is_valid() {
            return Err(SphereWorldError::InvalidInput);
        }
        let default_material = self.default_material();
        self.body_materials
            .resize(self.bodies.len(), default_material);
        self.body_materials[index] = material;
        Ok(())
    }

    /// Contact material of the finite ground plane.
    pub fn ground_material(&self) -> ColliderMaterial {
        self.ground_material
    }

    /// Assign the finite ground plane's contact material.
    pub fn set_ground_material(
        &mut self,
        material: ColliderMaterial,
    ) -> Result<(), SphereWorldError> {
        if !material.is_valid() {
            return Err(SphereWorldError::InvalidInput);
        }
        self.ground_material = material;
        Ok(())
    }

    /// Whether one dynamic body is currently sleeping.
    pub fn body_is_sleeping(&self, index: usize) -> Option<bool> {
        let body = self.bodies.get(index)?;
        Some(
            body.mass > 0.0
                && self
                    .sleep_states
                    .get(index)
                    .is_some_and(|state| state.sleeping),
        )
    }

    /// Wake one dynamic body and reset its idle timer.
    pub fn wake_body(&mut self, index: usize) -> Result<(), SphereWorldError> {
        if self.bodies.get(index).is_none_or(|body| body.mass <= 0.0) {
            return Err(SphereWorldError::InvalidInput);
        }
        self.ensure_sleep_states();
        self.sleep_states[index].wake();
        Ok(())
    }

    /// Put one dynamic body to sleep immediately and clear its velocity.
    pub fn sleep_body(&mut self, index: usize) -> Result<(), SphereWorldError> {
        if self.bodies.get(index).is_none_or(|body| body.mass <= 0.0) {
            return Err(SphereWorldError::InvalidInput);
        }
        self.ensure_sleep_states();
        self.sleep_states[index].sleep();
        self.bodies[index].velocity = Vector3::zeros();
        Ok(())
    }

    /// Net contact force from the most recent substep, in world coordinates.
    pub fn body_contact_force(&self, index: usize) -> Option<[f64; 3]> {
        self.contact_forces
            .get(index)
            .map(|force| [force.x, force.y, force.z])
    }

    fn default_material(&self) -> ColliderMaterial {
        ColliderMaterial::new(self.params.friction, self.params.restitution)
    }

    fn material_for_body(&self, index: usize) -> ColliderMaterial {
        self.body_materials
            .get(index)
            .copied()
            .unwrap_or_else(|| self.default_material())
    }

    fn ensure_sleep_states(&mut self) {
        self.sleep_states
            .resize(self.bodies.len(), SleepState::default());
    }

    /// Advance by `dt` seconds, subdividing at the configured maximum step.
    pub fn step(&mut self, dt: f64) -> Result<(), SphereWorldError> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(SphereWorldError::InvalidInput);
        }
        let count = (dt / self.params.max_substep).ceil();
        if count > 100_000.0 {
            return Err(SphereWorldError::Capacity);
        }
        let steps = count as usize;
        let substep = dt / steps as f64;
        for _ in 0..steps {
            self.step_once(
                substep,
                None,
                None,
                #[cfg(feature = "gpu-contact")]
                None,
            )?;
        }
        Ok(())
    }

    /// Use GPU pair generation and collision detection, then the CPU reference
    /// impulse solver and integrator. This path synchronizes once per substep.
    #[cfg(feature = "gpu-contact")]
    pub fn step_gpu(
        &mut self,
        dt: f64,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipeline: &GpuContactPipeline,
        ground_pipeline: &GpuGroundContacts,
    ) -> Result<(), SphereWorldError> {
        self.step_gpu_impl(dt, device, queue, pipeline, ground_pipeline, None)
    }

    #[cfg(feature = "gpu-contact")]
    fn step_gpu_impl(
        &mut self,
        dt: f64,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipeline: &GpuContactPipeline,
        ground_pipeline: &GpuGroundContacts,
        solver: Option<&GpuContactSolver>,
    ) -> Result<(), SphereWorldError> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(SphereWorldError::InvalidInput);
        }
        let count = (dt / self.params.max_substep).ceil();
        if count > 100_000.0 {
            return Err(SphereWorldError::Capacity);
        }
        let steps = count as usize;
        let substep = dt / steps as f64;
        for _ in 0..steps {
            let spheres = self
                .bodies
                .iter()
                .map(|body| {
                    GpuSphere::new(
                        [
                            body.center.x as f32,
                            body.center.y as f32,
                            body.center.z as f32,
                        ],
                        body.radius as f32,
                    )
                    .map_err(|_| SphereWorldError::InvalidInput)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let contacts = pipeline
                .dispatch_all_pairs(device, queue, &spheres)?
                .readback(device, queue)?;
            let ground_contacts = ground_pipeline.detect(
                device,
                queue,
                &spheres,
                self.params.ground_half_extent as f32,
            )?;
            self.step_once(
                substep,
                Some(&contacts),
                Some(&ground_contacts),
                solver.map(|solver| (device, queue, solver)),
            )?;
        }
        Ok(())
    }

    /// Step with an owned Tessera GPU contact context.
    #[cfg(feature = "gpu-contact")]
    pub fn step_gpu_with(
        &mut self,
        dt: f64,
        context: &GpuContactDevice,
    ) -> Result<(), SphereWorldError> {
        self.step_gpu_impl(
            dt,
            context.device(),
            context.queue(),
            context.pipeline(),
            context.ground(),
            Some(context.solver()),
        )
    }

    fn step_once(
        &mut self,
        dt: f64,
        #[cfg(feature = "gpu-contact")] gpu_contacts: Option<&[(GpuPair, GpuSphereContact)]>,
        #[cfg(not(feature = "gpu-contact"))] _gpu_contacts: Option<&()>,
        #[cfg(feature = "gpu-contact")] gpu_ground_contacts: Option<&[GpuSphereContact]>,
        #[cfg(not(feature = "gpu-contact"))] _gpu_ground_contacts: Option<&()>,
        #[cfg(feature = "gpu-contact")] gpu_solver: Option<(
            &wgpu::Device,
            &wgpu::Queue,
            &GpuContactSolver,
        )>,
    ) -> Result<(), SphereWorldError> {
        let body_count = self.bodies.len();
        self.ensure_sleep_states();
        for (index, body) in self.bodies.iter().enumerate() {
            if !self.params.sleep.enabled
                || (self.sleep_states[index].sleeping
                    && body.velocity.norm() > self.params.sleep.linear_velocity_threshold)
            {
                self.sleep_states[index].wake();
            }
        }
        let dimension = body_count
            .checked_mul(3)
            .ok_or(SphereWorldError::Capacity)?;
        let mut inverse_mass = DMatrix::zeros(dimension, dimension);
        let mut velocity = DVector::zeros(dimension);
        for (index, body) in self.bodies.iter().enumerate() {
            let offset = index * 3;
            if body.mass > 0.0 {
                for axis in 0..3 {
                    inverse_mass[(offset + axis, offset + axis)] = 1.0 / body.mass;
                    if !self.sleep_states[index].sleeping {
                        velocity[offset + axis] =
                            body.velocity[axis] + self.params.gravity[axis] * dt;
                    }
                }
            }
        }
        if self.max_linear_speed.is_some() {
            for index in 0..body_count {
                if self.bodies[index].mass > 0.0 {
                    let offset = index * 3;
                    let limited = self.limit_velocity(Vector3::new(
                        velocity[offset],
                        velocity[offset + 1],
                        velocity[offset + 2],
                    ));
                    velocity[offset] = limited.x;
                    velocity[offset + 1] = limited.y;
                    velocity[offset + 2] = limited.z;
                }
            }
        }
        let mut contacts = Vec::new();
        let mut body_contacting = vec![false; body_count];
        #[cfg(feature = "gpu-contact")]
        let mut pair_index = 0;
        for (index, body) in self.bodies.iter().enumerate() {
            let ground_depth = (body.center.x.abs()
                <= self.params.ground_half_extent + body.radius
                && body.center.y.abs() <= self.params.ground_half_extent + body.radius
                && body.center.z - body.radius <= 0.0)
                .then_some((body.radius - body.center.z).max(0.0));
            #[cfg(feature = "gpu-contact")]
            let ground_depth = gpu_ground_contacts.map_or(ground_depth, |result| {
                result[index]
                    .is_contact()
                    .then_some(f64::from(result[index].depth_hit[0]))
            });
            if let Some(penetration) = ground_depth.filter(|_| body.mass > 0.0) {
                body_contacting[index] = true;
                let material = self.ground_material.combine(self.material_for_body(index));
                let mut normal = DVector::zeros(dimension);
                let mut tangent_x = DVector::zeros(dimension);
                let mut tangent_y = DVector::zeros(dimension);
                normal[index * 3 + 2] = 1.0;
                tangent_x[index * 3] = 1.0;
                tangent_y[index * 3 + 1] = 1.0;
                contacts.push(ContactConstraint {
                    scalar: None,
                    normal,
                    tangents: [tangent_x, tangent_y],
                    penetration,
                    friction: material.friction,
                    restitution: material.restitution,
                });
            }
            for other_index in index + 1..body_count {
                let other = &self.bodies[other_index];
                #[cfg(feature = "gpu-contact")]
                if let Some(gpu_contacts) = gpu_contacts {
                    let (pair, contact) = &gpu_contacts[pair_index];
                    pair_index += 1;
                    debug_assert_eq!((pair.a as usize, pair.b as usize), (index, other_index));
                    if contact.is_contact() && (body.mass > 0.0 || other.mass > 0.0) {
                        body_contacting[index] = true;
                        body_contacting[other_index] = true;
                        contacts.push(Self::pair_constraint(
                            dimension,
                            index,
                            other_index,
                            Vector3::new(
                                f64::from(contact.normal[0]),
                                f64::from(contact.normal[1]),
                                f64::from(contact.normal[2]),
                            ),
                            f64::from(contact.depth_hit[0]),
                            self.material_for_body(index)
                                .combine(self.material_for_body(other_index)),
                        ));
                    }
                    continue;
                }
                if other.mass == 0.0 && body.mass == 0.0 {
                    continue;
                }
                let delta = other.center - body.center;
                let distance = delta.norm();
                let radius_sum = body.radius + other.radius;
                if distance > radius_sum {
                    continue;
                }
                body_contacting[index] = true;
                body_contacting[other_index] = true;
                let direction = if distance > 1e-12 {
                    delta / distance
                } else {
                    Vector3::x()
                };
                contacts.push(Self::pair_constraint(
                    dimension,
                    index,
                    other_index,
                    direction,
                    (radius_sum - distance).max(0.0),
                    self.material_for_body(index)
                        .combine(self.material_for_body(other_index)),
                ));
            }
        }
        let problem = ContactProblem {
            inverse_mass,
            velocity,
            contacts,
        };
        let solve_params = SolveParams {
            dt,
            iterations: self.params.solver_iterations,
            position_gain: 0.2,
            max_correction_speed: 2.0,
        };
        #[cfg(feature = "gpu-contact")]
        let solved = if let Some((device, queue, solver)) = gpu_solver {
            solver.solve(device, queue, &problem, solve_params, None)?
        } else {
            solve_contacts(&problem, solve_params, None)?
        };
        #[cfg(not(feature = "gpu-contact"))]
        let solved = solve_contacts(&problem, solve_params, None)?;
        self.contact_forces.fill(Vector3::zeros());
        for (contact, impulse) in problem.contacts.iter().zip(&solved.impulses) {
            let generalized_impulse = &contact.normal * impulse.normal
                + &contact.tangents[0] * impulse.tangents[0]
                + &contact.tangents[1] * impulse.tangents[1];
            for (index, force) in self.contact_forces.iter_mut().enumerate() {
                for axis in 0..3 {
                    force[axis] += generalized_impulse[index * 3 + axis] / dt;
                }
            }
        }
        for (index, &has_contact) in body_contacting.iter().enumerate() {
            if self.bodies[index].mass <= 0.0 {
                continue;
            }
            let solved_velocity = self.limit_velocity(Vector3::new(
                solved.velocity[index * 3],
                solved.velocity[index * 3 + 1],
                solved.velocity[index * 3 + 2],
            ));
            self.sleep_states[index].update(
                self.params.sleep,
                dt,
                solved_velocity.norm(),
                0.0,
                has_contact,
            );
            if self.sleep_states[index].sleeping {
                self.bodies[index].velocity = Vector3::zeros();
            } else {
                self.bodies[index].velocity = solved_velocity;
                self.bodies[index].center += solved_velocity * dt;
            }
        }
        Ok(())
    }

    fn pair_constraint(
        dimension: usize,
        a: usize,
        b: usize,
        normal: Vector3<f64>,
        penetration: f64,
        material: CombinedMaterial,
    ) -> ContactConstraint {
        let tangent_one = if normal.z.abs() < 0.9 {
            normal.cross(&Vector3::z()).normalize()
        } else {
            normal.cross(&Vector3::x()).normalize()
        };
        let tangent_two = normal.cross(&tangent_one);
        let row = |axis: Vector3<f64>| {
            let mut result = DVector::zeros(dimension);
            for component in 0..3 {
                result[a * 3 + component] = -axis[component];
                result[b * 3 + component] = axis[component];
            }
            result
        };
        ContactConstraint {
            scalar: None,
            normal: row(normal),
            tangents: [row(tangent_one), row(tangent_two)],
            penetration,
            friction: material.friction,
            restitution: material.restitution,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::CoefficientCombineRule;

    fn sphere(z: f64) -> SphereBody {
        SphereBody {
            center: Vector3::new(0.0, 0.0, z),
            velocity: Vector3::zeros(),
            radius: 0.5,
            mass: 1.0,
        }
    }

    #[test]
    fn dropped_sphere_settles_above_ground() {
        let mut world = SphereWorld::new(vec![sphere(2.0)], SphereWorldParams::default()).unwrap();
        for _ in 0..240 {
            world.step(1.0 / 240.0).unwrap();
        }
        assert!((world.bodies[0].center.z - 0.5).abs() < 0.02);
        assert!(world.bodies[0].velocity.z.abs() < 0.05);
        assert!(world.body_contact_force(0).unwrap()[2] > 0.0);
        for _ in 0..120 {
            world.step(1.0 / 240.0).unwrap();
        }
        assert_eq!(world.body_is_sleeping(0), Some(true));
    }

    #[test]
    fn sleeping_body_wakes_when_its_public_velocity_changes() {
        let mut world = SphereWorld::new(
            vec![sphere(0.5)],
            SphereWorldParams {
                gravity: [0.0; 3],
                sleep: SleepSettings {
                    time_threshold: 0.02,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap();
        world.step(0.03).unwrap();
        assert_eq!(world.body_is_sleeping(0), Some(true));
        world.bodies[0].velocity.z = 1.0;
        world.step(0.01).unwrap();
        assert_eq!(world.body_is_sleeping(0), Some(false));
        assert!(world.bodies[0].center.z > 0.5);
    }

    #[test]
    fn collision_impulse_wakes_a_sleeping_body() {
        let mut sleeper = sphere(10.0);
        let mut impactor = sphere(10.0);
        impactor.center.x = 0.9;
        impactor.velocity.x = -2.0;
        let mut world = SphereWorld::new(
            vec![sleeper.clone(), impactor],
            SphereWorldParams {
                gravity: [0.0; 3],
                ..Default::default()
            },
        )
        .unwrap();
        world.sleep_body(0).unwrap();
        world.step(0.01).unwrap();
        assert_eq!(world.body_is_sleeping(0), Some(false));
        assert!(world.bodies[0].velocity.x < 0.0);

        sleeper.mass = 0.0;
        let mut static_world =
            SphereWorld::new(vec![sleeper], SphereWorldParams::default()).unwrap();
        assert!(static_world.sleep_body(0).is_err());
    }

    #[test]
    fn stacked_spheres_do_not_interpenetrate() {
        let mut world =
            SphereWorld::new(vec![sphere(0.5), sphere(1.5)], SphereWorldParams::default()).unwrap();
        for _ in 0..240 {
            world.step(1.0 / 240.0).unwrap();
        }
        assert!((world.bodies[0].center.z - 0.5).abs() < 0.03);
        assert!((world.bodies[1].center.z - 1.5).abs() < 0.04);
        assert!(world.bodies[1].center.z - world.bodies[0].center.z >= 0.98);
    }

    #[test]
    fn dynamic_sphere_separates_from_static_sphere() {
        let mut anchor = sphere(2.0);
        anchor.mass = 0.0;
        let mut moving = sphere(2.0);
        moving.center.x = 0.8;
        let mut world =
            SphereWorld::new(vec![anchor, moving], SphereWorldParams::default()).unwrap();
        world.step(0.01).unwrap();
        assert_eq!(world.bodies[0].center.x, 0.0);
        assert!(world.bodies[1].center.x > 0.8);
    }

    #[test]
    fn linear_speed_limit_applies_before_integration_and_after_contact() {
        let params = SphereWorldParams {
            gravity: [0.0; 3],
            max_substep: 0.01,
            ..Default::default()
        };
        let mut moving = sphere(10.0);
        moving.velocity = Vector3::new(3.0, 4.0, 0.0);
        let mut free = SphereWorld::new(vec![moving], params).unwrap();
        assert!(free.set_max_linear_speed(Some(f64::NAN)).is_err());
        assert!(free.set_max_linear_speed(Some(0.0)).is_err());
        assert_eq!(free.max_linear_speed(), None);
        free.set_max_linear_speed(Some(2.0)).unwrap();
        free.step(0.01).unwrap();
        assert!((free.bodies[0].velocity - Vector3::new(1.2, 1.6, 0.0)).norm() < 1e-12);
        assert!((free.bodies[0].center - Vector3::new(0.012, 0.016, 10.0)).norm() < 1e-12);

        let mut anchor = sphere(10.0);
        anchor.mass = 0.0;
        let mut impactor = sphere(10.0);
        impactor.center.x = 0.8;
        let mut contact = SphereWorld::new(vec![anchor, impactor], params).unwrap();
        contact.set_max_linear_speed(Some(0.5)).unwrap();
        contact.step(0.01).unwrap();
        assert!((contact.bodies[1].velocity.norm() - 0.5).abs() < 1e-12);
        assert_eq!(contact.bodies[0].velocity, Vector3::zeros());
        contact.set_max_linear_speed(None).unwrap();
        assert_eq!(contact.max_linear_speed(), None);
    }

    #[test]
    fn restitution_bounces_a_sphere_from_the_ground() {
        let mut body = sphere(0.5);
        body.velocity.z = -2.0;
        let mut world = SphereWorld::new(
            vec![body],
            SphereWorldParams {
                gravity: [0.0; 3],
                max_substep: 0.001,
                ..Default::default()
            },
        )
        .unwrap();
        world
            .set_body_material(
                0,
                ColliderMaterial {
                    restitution: 0.75,
                    restitution_combine_rule: CoefficientCombineRule::Max,
                    ..ColliderMaterial::default()
                },
            )
            .unwrap();
        world.step(0.001).unwrap();
        assert!((world.bodies[0].velocity.z - 1.5).abs() < 1e-12);
    }

    #[test]
    fn rejects_invalid_or_unknown_material_assignment() {
        let mut world = SphereWorld::new(vec![sphere(0.5)], SphereWorldParams::default()).unwrap();
        assert!(
            world
                .set_body_material(1, ColliderMaterial::default())
                .is_err()
        );
        assert!(
            world
                .set_ground_material(ColliderMaterial::new(0.5, -0.1))
                .is_err()
        );
        assert_eq!(
            world.body_material(0),
            Some(ColliderMaterial::new(1.0, 0.0))
        );
    }

    #[cfg(feature = "gpu-contact")]
    #[tokio::test]
    async fn gpu_collision_matches_cpu_reference_step() {
        let instance = wgpu::Instance::default();
        let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
        else {
            return;
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let mut second = sphere(0.5);
        second.center.x = 0.9;
        let params = SphereWorldParams {
            max_substep: 0.01,
            ..SphereWorldParams::default()
        };
        let mut cpu = SphereWorld::new(vec![sphere(0.5), second.clone()], params).unwrap();
        let mut gpu = SphereWorld::new(vec![sphere(0.5), second], params).unwrap();
        cpu.step(0.01).unwrap();
        gpu.step_gpu(
            0.01,
            &device,
            &queue,
            &GpuContactPipeline::new(&device),
            &GpuGroundContacts::new(&device),
        )
        .unwrap();
        for (cpu_body, gpu_body) in cpu.bodies.iter().zip(&gpu.bodies) {
            assert!((cpu_body.center - gpu_body.center).norm() < 1e-5);
            assert!((cpu_body.velocity - gpu_body.velocity).norm() < 1e-5);
        }
    }

    #[cfg(feature = "gpu-contact")]
    #[test]
    fn owned_gpu_context_solves_impulses_and_matches_cpu_world() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut first = sphere(0.49);
        first.velocity = Vector3::new(0.5, 0.0, -1.0);
        let mut second = sphere(0.51);
        second.center.x = 0.9;
        second.velocity.x = -0.5;
        let params = SphereWorldParams {
            max_substep: 0.01,
            ..SphereWorldParams::default()
        };
        let mut cpu = SphereWorld::new(vec![first.clone(), second.clone()], params).unwrap();
        let mut gpu = SphereWorld::new(vec![first, second], params).unwrap();
        cpu.set_max_linear_speed(Some(0.8)).unwrap();
        gpu.set_max_linear_speed(Some(0.8)).unwrap();
        for _ in 0..4 {
            cpu.step(0.01).unwrap();
            gpu.step_gpu_with(0.01, &context).unwrap();
        }
        for (cpu_body, gpu_body) in cpu.bodies.iter().zip(&gpu.bodies) {
            assert!((cpu_body.center - gpu_body.center).norm() < 2e-4);
            assert!((cpu_body.velocity - gpu_body.velocity).norm() < 2e-4);
        }
    }
}
