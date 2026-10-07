//! Deterministic sparse-grid MLS-MPM reference pipeline.

use std::collections::BTreeMap;

use nalgebra::{Matrix3, Vector3};

use crate::material::{MaterialModel, PlasticState};
use crate::obstacle::RigidObstacle;

/// Axis-aligned particle boundary with separating, frictionless walls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorldBounds {
    /// Lower world-space corner.
    pub min: Vector3<f64>,
    /// Upper world-space corner.
    pub max: Vector3<f64>,
}

impl WorldBounds {
    fn is_valid(self) -> bool {
        self.min.iter().all(|value| value.is_finite())
            && self.max.iter().all(|value| value.is_finite())
            && (self.max - self.min).iter().all(|value| *value > 0.0)
    }
}

/// Global MPM integration parameters.
#[derive(Debug, Clone)]
pub struct MpmParams {
    /// Sparse-grid cell width in metres.
    pub cell_width: f64,
    /// World-space gravitational acceleration.
    pub gravity: Vector3<f64>,
    /// Upper bound for one explicit substep.
    pub max_substep: f64,
    /// Optional axis-aligned collision boundary.
    pub bounds: Option<WorldBounds>,
}

impl Default for MpmParams {
    fn default() -> Self {
        Self {
            cell_width: 0.1,
            gravity: Vector3::new(0.0, 0.0, -9.81),
            max_substep: 1.0 / 1_000.0,
            bounds: None,
        }
    }
}

/// One material point and its persistent APIC state.
#[derive(Debug, Clone)]
pub struct MpmParticle {
    /// World-space center position.
    pub position: Vector3<f64>,
    /// World-space velocity.
    pub velocity: Vector3<f64>,
    /// Deformation gradient.
    pub deformation: Matrix3<f64>,
    /// APIC affine velocity field.
    pub affine: Matrix3<f64>,
    /// Additional world-space force accumulated until the next step.
    pub force: Vector3<f64>,
    /// Positive particle mass.
    pub mass: f64,
    /// Positive initial particle volume.
    pub rest_volume: f64,
    /// Initial sampling radius used to derive volume and render size.
    pub radius: f64,
    /// Constitutive model.
    pub material: MaterialModel,
    /// Persistent plastic variables.
    pub plastic: PlasticState,
    /// Velocity damping coefficient in inverse seconds.
    pub damping: f64,
    /// Render/selection group with no effect on dynamics.
    pub group_id: u32,
    /// Particle-grid transfer region. Different regions use independent grid nodes.
    pub transfer_color: u16,
    /// Whether the particle is simulated.
    pub enabled: bool,
    /// Whether transfers preserve this particle's pose and zero its velocity.
    pub fixed: bool,
    /// Stable identifier of the insertion batch containing this particle.
    chunk_id: ParticleChunkId,
}

/// Identifier of an insertion batch that can be removed as one unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ParticleChunkId(u64);

impl ParticleChunkId {
    /// Initial particles passed to [`MpmWorld::new`] belong to this chunk.
    pub const INITIAL: Self = Self(0);

    /// Stable numeric identifier.
    pub fn get(self) -> u64 {
        self.0
    }
}

impl MpmParticle {
    /// Create a cubic material point matching Nexus' `(2r)^3` initial volume.
    pub fn new(position: Vector3<f64>, radius: f64, density: f64, material: MaterialModel) -> Self {
        let rest_volume = (2.0 * radius).powi(3);
        Self {
            position,
            velocity: Vector3::zeros(),
            deformation: Matrix3::identity(),
            affine: Matrix3::zeros(),
            force: Vector3::zeros(),
            mass: density * rest_volume,
            rest_volume,
            radius,
            material,
            plastic: PlasticState {
                hardening: if matches!(material, MaterialModel::SandNeoHookean { .. }) {
                    1.0
                } else {
                    0.0
                },
                ..PlasticState::default()
            },
            damping: 0.0,
            group_id: 0,
            transfer_color: 0,
            enabled: true,
            fixed: false,
            chunk_id: ParticleChunkId::INITIAL,
        }
    }

    /// Initial material density in kilograms per cubic metre.
    pub fn density(&self) -> f64 {
        self.mass / self.rest_volume
    }

    /// Insertion batch containing this particle.
    pub fn chunk_id(&self) -> ParticleChunkId {
        self.chunk_id
    }
}

/// Invalid input or non-finite simulation state.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum MpmError {
    /// Parameters or particles are invalid.
    #[error("invalid MPM input")]
    InvalidInput,
    /// The requested step would require an excessive number of substeps.
    #[error("MPM timestep is too small for the requested duration")]
    ExcessiveSubsteps,
    /// Integration produced a non-finite state.
    #[error("MPM integration produced a non-finite state")]
    NonFiniteState,
    /// A requested particle chunk does not exist.
    #[error("MPM particle chunk was not found")]
    UnknownChunk,
    /// Particle insertion or chunk identifiers exceeded supported capacity.
    #[error("MPM particle capacity exceeded")]
    ParticleCapacity,
}

#[derive(Debug, Clone, Copy, Default)]
struct GridNode {
    mass: f64,
    momentum: Vector3<f64>,
    velocity: Vector3<f64>,
}

/// Momentum transferred from MPM to one obstacle during a step.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ObstacleReaction {
    /// Linear impulse in world coordinates.
    pub linear: Vector3<f64>,
    /// Angular impulse about the obstacle center in world coordinates.
    pub angular: Vector3<f64>,
}

impl ObstacleReaction {
    fn record(&mut self, center: Vector3<f64>, point: Vector3<f64>, impulse: Vector3<f64>) {
        self.linear += impulse;
        self.angular += (point - center).cross(&impulse);
    }
}

/// Three-dimensional sparse-grid MPM world.
#[derive(Debug, Clone)]
pub struct MpmWorld {
    /// Particle state in stable insertion order.
    pub particles: Vec<MpmParticle>,
    /// Integration parameters.
    pub params: MpmParams,
    /// Prescribed rigid obstacles that push particles without receiving impulses.
    pub obstacles: Vec<RigidObstacle>,
    /// Number of explicit substeps completed since construction.
    pub substeps: u64,
    next_chunk_id: u64,
}

impl MpmWorld {
    /// Validate particles and construct a deterministic CPU reference world.
    pub fn new(mut particles: Vec<MpmParticle>, params: MpmParams) -> Result<Self, MpmError> {
        if !params.cell_width.is_finite()
            || params.cell_width <= 0.0
            || params.gravity.iter().any(|value| !value.is_finite())
            || !params.max_substep.is_finite()
            || params.max_substep <= 0.0
            || params.bounds.is_some_and(|bounds| !bounds.is_valid())
            || particles.iter().any(|particle| !valid_particle(particle))
        {
            return Err(MpmError::InvalidInput);
        }
        for particle in &mut particles {
            particle.chunk_id = ParticleChunkId::INITIAL;
        }
        Ok(Self {
            particles,
            params,
            obstacles: Vec::new(),
            substeps: 0,
            next_chunk_id: 1,
        })
    }

    /// Append a validated batch and return a stable handle for later removal.
    pub fn add_particles(
        &mut self,
        mut particles: Vec<MpmParticle>,
    ) -> Result<ParticleChunkId, MpmError> {
        if particles.is_empty() || particles.iter().any(|particle| !valid_particle(particle)) {
            return Err(MpmError::InvalidInput);
        }
        let next = self
            .next_chunk_id
            .checked_add(1)
            .ok_or(MpmError::ParticleCapacity)?;
        self.particles
            .try_reserve(particles.len())
            .map_err(|_| MpmError::ParticleCapacity)?;
        let chunk = ParticleChunkId(self.next_chunk_id);
        for particle in &mut particles {
            particle.chunk_id = chunk;
        }
        self.particles.append(&mut particles);
        self.next_chunk_id = next;
        Ok(chunk)
    }

    /// Remove every particle in a chunk, preserving other particles' order.
    pub fn remove_chunk(&mut self, chunk: ParticleChunkId) -> Result<usize, MpmError> {
        let before = self.particles.len();
        self.particles.retain(|particle| particle.chunk_id != chunk);
        let removed = before - self.particles.len();
        if removed == 0 {
            Err(MpmError::UnknownChunk)
        } else {
            Ok(removed)
        }
    }

    /// Replace the one-way rigid obstacles used by the next substep.
    pub fn set_obstacles(&mut self, obstacles: Vec<RigidObstacle>) -> Result<(), MpmError> {
        if obstacles.iter().any(|obstacle| !obstacle.is_valid()) {
            return Err(MpmError::InvalidInput);
        }
        self.obstacles = obstacles;
        Ok(())
    }

    /// Conservative CFL bound across all enabled particles.
    pub fn stable_timestep(&self) -> f64 {
        self.particles
            .iter()
            .filter(|particle| particle.enabled && !particle.fixed)
            .map(|particle| {
                particle.material.timestep_bound_with_deformation(
                    particle.density(),
                    particle.velocity,
                    self.params.cell_width,
                    particle.deformation.determinant(),
                )
            })
            .fold(self.params.max_substep, f64::min)
            .min(self.params.max_substep)
    }

    /// Advance by `dt` using CFL-limited explicit MLS-MPM substeps.
    pub fn step(&mut self, dt: f64) -> Result<(), MpmError> {
        self.advance(dt, None)
    }

    /// Advance MPM and return the reaction impulse on each obstacle.
    ///
    /// Reactions include both grid-node boundary projection and final particle
    /// collision projection. The returned vector follows obstacle order.
    pub fn step_with_obstacle_reactions(
        &mut self,
        dt: f64,
    ) -> Result<Vec<ObstacleReaction>, MpmError> {
        let mut reactions = vec![ObstacleReaction::default(); self.obstacles.len()];
        self.advance(dt, Some(&mut reactions))?;
        if reactions.iter().any(|reaction| {
            reaction
                .linear
                .iter()
                .chain(reaction.angular.iter())
                .any(|component| !component.is_finite())
        }) {
            return Err(MpmError::NonFiniteState);
        }
        Ok(reactions)
    }

    fn advance(
        &mut self,
        dt: f64,
        mut reactions: Option<&mut [ObstacleReaction]>,
    ) -> Result<(), MpmError> {
        if !dt.is_finite()
            || dt <= 0.0
            || self.obstacles.iter().any(|obstacle| !obstacle.is_valid())
        {
            return Err(MpmError::InvalidInput);
        }
        let mut remaining = dt;
        let mut count = 0usize;
        while remaining > 0.0 {
            let stable = self.stable_timestep();
            if !stable.is_finite() || stable <= 1e-12 {
                return Err(MpmError::ExcessiveSubsteps);
            }
            let substep = remaining.min(stable);
            self.substep(substep, reactions.as_deref_mut())?;
            remaining = (remaining - substep).max(0.0);
            count += 1;
            if count > 100_000 {
                return Err(MpmError::ExcessiveSubsteps);
            }
        }
        Ok(())
    }

    fn substep(
        &mut self,
        dt: f64,
        mut reactions: Option<&mut [ObstacleReaction]>,
    ) -> Result<(), MpmError> {
        let h = self.params.cell_width;
        let inv_h = h.recip();
        let inv_d = 4.0 * inv_h * inv_h;
        let transfer_keys = self
            .particles
            .iter()
            .map(|particle| transfer_key(particle, &self.obstacles))
            .collect::<Vec<_>>();
        let mut grid = BTreeMap::<([i32; 3], u64), GridNode>::new();

        for (particle, transfer_key) in self.particles.iter().zip(&transfer_keys) {
            if !particle.enabled {
                continue;
            }
            let grid_pos = particle.position * inv_h;
            let base = base_cell(grid_pos);
            let fx = grid_pos - int_vector(base);
            let weights = quadratic_weights(fx);
            let stress = particle.material.kirchhoff_stress(
                particle.deformation,
                particle.affine,
                particle.plastic,
            );
            let force_affine = stress * (-dt * particle.rest_volume * inv_d);
            let momentum_affine = force_affine + particle.affine * particle.mass;
            for_each_neighbor(base, fx, &weights, h, |node_id, weight, dpos| {
                let node = grid.entry((node_id, *transfer_key)).or_default();
                node.mass += weight * particle.mass;
                node.momentum +=
                    weight * (particle.mass * particle.velocity + momentum_affine * dpos);
            });
        }

        for ((id, _), node) in &mut grid {
            if node.mass <= 0.0 {
                continue;
            }
            node.velocity = node.momentum / node.mass + self.params.gravity * dt;
            if let Some(bounds) = self.params.bounds {
                apply_grid_boundary(*id, h, bounds, &mut node.velocity);
            }
            apply_grid_obstacles(
                int_vector(*id) * h,
                h,
                node.mass,
                &self.obstacles,
                &mut node.velocity,
                reactions.as_deref_mut(),
            );
        }

        for (particle, transfer_key) in self.particles.iter_mut().zip(&transfer_keys) {
            if !particle.enabled {
                particle.force = Vector3::zeros();
                continue;
            }
            if particle.fixed {
                particle.velocity = Vector3::zeros();
                particle.affine = Matrix3::zeros();
                particle.force = Vector3::zeros();
                continue;
            }
            let grid_pos = particle.position * inv_h;
            let base = base_cell(grid_pos);
            let fx = grid_pos - int_vector(base);
            let weights = quadratic_weights(fx);
            let mut velocity = Vector3::zeros();
            let mut affine = Matrix3::zeros();
            for_each_neighbor(base, fx, &weights, h, |node_id, weight, dpos| {
                let node_velocity = grid
                    .get(&(node_id, *transfer_key))
                    .map_or_else(Vector3::zeros, |node| node.velocity);
                velocity += node_velocity * weight;
                affine += node_velocity * dpos.transpose() * (weight * inv_d);
            });
            velocity += particle.force / particle.mass * dt;
            velocity *= (-particle.damping * dt).exp();
            particle.velocity = velocity;
            particle.affine = affine;
            particle.position += velocity * dt;
            particle.deformation = (Matrix3::identity() + affine * dt) * particle.deformation;
            (particle.deformation, particle.plastic) = particle
                .material
                .project_deformation(particle.deformation, particle.plastic);
            particle.force = Vector3::zeros();
            if let Some(bounds) = self.params.bounds {
                apply_particle_boundary(bounds, particle);
            }
            apply_particle_obstacles_with_reactions(
                &self.obstacles,
                particle,
                reactions.as_deref_mut(),
            );
            if !finite_particle_state(particle) {
                return Err(MpmError::NonFiniteState);
            }
        }
        self.substeps = self.substeps.saturating_add(1);
        Ok(())
    }
}

fn transfer_key(particle: &MpmParticle, obstacles: &[RigidObstacle]) -> u64 {
    let mut closest = [None::<(f64, bool)>; 32];
    for obstacle in obstacles {
        let Some(group) = obstacle.cpic_group else {
            continue;
        };
        if let Some(side) = obstacle.cpic_side(particle.position) {
            let entry = &mut closest[usize::from(group)];
            if entry.is_none_or(|previous| side.0 < previous.0) {
                *entry = Some(side);
            }
        }
    }
    let mut mask = 0u32;
    for (group, side) in closest.iter().enumerate() {
        if side.is_some_and(|(_, positive)| positive) {
            mask |= 1 << group;
        }
    }
    (u64::from(particle.transfer_color) << 32) | u64::from(mask)
}

fn valid_particle(particle: &MpmParticle) -> bool {
    finite_particle_state(particle)
        && particle.mass > 0.0
        && particle.rest_volume > 0.0
        && particle.radius > 0.0
        && particle.material.is_valid()
        && particle.damping >= 0.0
        && particle.plastic.plastic_det.is_finite()
        && particle.plastic.plastic_det > 0.0
        && particle.plastic.hardening.is_finite()
        && particle.plastic.log_volume_gain.is_finite()
}

pub(crate) fn finite_particle_state(particle: &MpmParticle) -> bool {
    particle.position.iter().all(|value| value.is_finite())
        && particle.velocity.iter().all(|value| value.is_finite())
        && particle.deformation.iter().all(|value| value.is_finite())
        && particle.affine.iter().all(|value| value.is_finite())
        && particle.force.iter().all(|value| value.is_finite())
        && particle.mass.is_finite()
        && particle.rest_volume.is_finite()
        && particle.radius.is_finite()
        && particle.damping.is_finite()
}

fn base_cell(grid_position: Vector3<f64>) -> [i32; 3] {
    [
        (grid_position.x - 0.5).floor() as i32,
        (grid_position.y - 0.5).floor() as i32,
        (grid_position.z - 0.5).floor() as i32,
    ]
}

fn int_vector(value: [i32; 3]) -> Vector3<f64> {
    Vector3::new(value[0] as f64, value[1] as f64, value[2] as f64)
}

fn quadratic_weights(fx: Vector3<f64>) -> [[f64; 3]; 3] {
    let axis = |value: f64| {
        [
            0.5 * (1.5 - value).powi(2),
            0.75 - (value - 1.0).powi(2),
            0.5 * (value - 0.5).powi(2),
        ]
    };
    [axis(fx.x), axis(fx.y), axis(fx.z)]
}

fn for_each_neighbor(
    base: [i32; 3],
    fx: Vector3<f64>,
    weights: &[[f64; 3]; 3],
    cell_width: f64,
    mut visit: impl FnMut([i32; 3], f64, Vector3<f64>),
) {
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                let offset = Vector3::new(i as f64, j as f64, k as f64);
                visit(
                    [base[0] + i, base[1] + j, base[2] + k],
                    weights[0][i as usize] * weights[1][j as usize] * weights[2][k as usize],
                    (offset - fx) * cell_width,
                );
            }
        }
    }
}

fn apply_grid_boundary(
    id: [i32; 3],
    cell_width: f64,
    bounds: WorldBounds,
    velocity: &mut Vector3<f64>,
) {
    let position = int_vector(id) * cell_width;
    for axis in 0..3 {
        if (position[axis] <= bounds.min[axis] + cell_width && velocity[axis] < 0.0)
            || (position[axis] >= bounds.max[axis] - cell_width && velocity[axis] > 0.0)
        {
            velocity[axis] = 0.0;
        }
    }
}

fn apply_grid_obstacles(
    point: Vector3<f64>,
    cell_width: f64,
    node_mass: f64,
    obstacles: &[RigidObstacle],
    velocity: &mut Vector3<f64>,
    mut reactions: Option<&mut [ObstacleReaction]>,
) {
    for (index, obstacle) in obstacles.iter().enumerate() {
        if !obstacle.may_contact(point, cell_width) {
            continue;
        }
        let (distance, normal) = obstacle.surface(point);
        if distance <= cell_width * (1.0 + 1e-5) {
            let before = *velocity;
            *velocity = obstacle.contact_velocity(point, *velocity, normal);
            if let Some(reactions) = reactions.as_deref_mut() {
                let surface_point = point - normal * distance;
                reactions[index].record(
                    obstacle.center,
                    surface_point,
                    (before - *velocity) * node_mass,
                );
            }
        }
    }
}

fn apply_particle_obstacles_with_reactions(
    obstacles: &[RigidObstacle],
    particle: &mut MpmParticle,
    mut reactions: Option<&mut [ObstacleReaction]>,
) {
    for (index, obstacle) in obstacles.iter().enumerate() {
        if !obstacle.may_contact(particle.position, particle.radius) {
            continue;
        }
        let (distance, normal) = obstacle.surface(particle.position);
        if distance < particle.radius {
            particle.position += normal * (particle.radius - distance);
            let before = particle.velocity;
            particle.velocity =
                obstacle.contact_velocity(particle.position, particle.velocity, normal);
            if let Some(reactions) = reactions.as_deref_mut() {
                let point = particle.position - normal * particle.radius;
                reactions[index].record(
                    obstacle.center,
                    point,
                    (before - particle.velocity) * particle.mass,
                );
            }
        }
    }
}

pub(crate) fn apply_particle_boundary(bounds: WorldBounds, particle: &mut MpmParticle) {
    for axis in 0..3 {
        let lower = bounds.min[axis] + particle.radius;
        let upper = bounds.max[axis] - particle.radius;
        if particle.position[axis] < lower {
            particle.position[axis] = lower;
            particle.velocity[axis] = particle.velocity[axis].max(0.0);
        } else if particle.position[axis] > upper {
            particle.position[axis] = upper;
            particle.velocity[axis] = particle.velocity[axis].min(0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn particle(position: Vector3<f64>) -> MpmParticle {
        MpmParticle::new(
            position,
            0.04,
            1_000.0,
            MaterialModel::elastic(1_000.0, 0.2),
        )
    }

    #[test]
    fn one_particle_free_fall_matches_gravity() {
        let mut world = MpmWorld::new(
            vec![particle(Vector3::new(0.5, 0.5, 0.5))],
            MpmParams {
                max_substep: 0.001,
                ..MpmParams::default()
            },
        )
        .unwrap();
        world.step(0.001).unwrap();
        assert!((world.particles[0].velocity.z + 0.00981).abs() < 1e-10);
        assert!((world.particles[0].position.z - (0.5 - 0.00000981)).abs() < 1e-10);
    }

    #[test]
    fn apic_transfer_preserves_uniform_velocity_without_forces() {
        let mut a = particle(Vector3::new(0.45, 0.5, 0.55));
        let mut b = particle(Vector3::new(0.55, 0.5, 0.45));
        a.velocity = Vector3::new(1.0, -2.0, 0.5);
        b.velocity = a.velocity;
        let mut world = MpmWorld::new(
            vec![a, b],
            MpmParams {
                gravity: Vector3::zeros(),
                max_substep: 0.001,
                ..MpmParams::default()
            },
        )
        .unwrap();
        world.step(0.001).unwrap();
        for particle in &world.particles {
            assert!((particle.velocity - Vector3::new(1.0, -2.0, 0.5)).norm() < 1e-10);
        }
    }

    #[test]
    fn transfer_colors_keep_opposed_particles_on_separate_grid_nodes() {
        let mut left = particle(Vector3::new(0.5, 0.5, 0.5));
        left.velocity.x = 1.0;
        let mut right = left.clone();
        right.velocity.x = -1.0;
        right.transfer_color = 1;
        let params = MpmParams {
            gravity: Vector3::zeros(),
            max_substep: 0.001,
            ..MpmParams::default()
        };
        let mut separated =
            MpmWorld::new(vec![left.clone(), right.clone()], params.clone()).unwrap();
        separated.step(0.001).unwrap();
        assert!((separated.particles[0].velocity.x - 1.0).abs() < 1e-10);
        assert!((separated.particles[1].velocity.x + 1.0).abs() < 1e-10);

        right.transfer_color = 0;
        let mut mixed = MpmWorld::new(vec![left, right], params).unwrap();
        mixed.step(0.001).unwrap();
        assert!(mixed.particles[0].velocity.x.abs() < 1e-10);
        assert!(mixed.particles[1].velocity.x.abs() < 1e-10);
    }

    #[test]
    fn cpic_triangle_assigns_opposite_sides_to_separate_grid_regions() {
        let mut left = MpmParticle::new(
            Vector3::new(0.48, 0.5, 0.5),
            0.005,
            1_000.0,
            MaterialModel::elastic(1_000.0, 0.2),
        );
        left.velocity.x = -1.0;
        let mut right = left.clone();
        right.position.x = 0.52;
        right.velocity.x = 1.0;
        let mut sheet = RigidObstacle::triangle_prism(
            Vector3::repeat(0.5),
            [
                Vector3::new(0.0, -1.0, -1.0),
                Vector3::new(0.0, 1.0, -1.0),
                Vector3::new(0.0, 0.0, 1.0),
            ],
            0.001,
            nalgebra::UnitQuaternion::identity(),
        );
        sheet.cpic_group = Some(0);
        sheet.boundary = crate::ObstacleBoundary::NonReflecting;
        let params = MpmParams {
            gravity: Vector3::zeros(),
            ..MpmParams::default()
        };
        let mut separated =
            MpmWorld::new(vec![left.clone(), right.clone()], params.clone()).unwrap();
        separated.set_obstacles(vec![sheet.clone()]).unwrap();
        separated.step(0.001).unwrap();
        assert!(
            separated.particles[0].velocity.x < -0.9,
            "{:?}",
            separated
                .particles
                .iter()
                .map(|p| p.velocity.x)
                .collect::<Vec<_>>()
        );
        assert!(separated.particles[1].velocity.x > 0.9);

        sheet.cpic_group = None;
        let mut mixed = MpmWorld::new(vec![left, right], params).unwrap();
        mixed.set_obstacles(vec![sheet]).unwrap();
        mixed.step(0.001).unwrap();
        assert!(mixed.particles[0].velocity.x.abs() < 0.5);
        assert!(mixed.particles[1].velocity.x.abs() < 0.5);
    }

    #[test]
    fn obstacle_reaction_opposes_particle_impact() {
        let mut impactor = particle(Vector3::new(1.02, 0.0, 0.0));
        impactor.velocity.x = -1.0;
        let initial_momentum = impactor.velocity * impactor.mass;
        let mut world = MpmWorld::new(
            vec![impactor],
            MpmParams {
                gravity: Vector3::zeros(),
                ..MpmParams::default()
            },
        )
        .unwrap();
        world
            .set_obstacles(vec![RigidObstacle::sphere(Vector3::zeros(), 1.0)])
            .unwrap();
        let reactions = world.step_with_obstacle_reactions(0.001).unwrap();
        assert_eq!(reactions.len(), 1);
        assert!(reactions[0].linear.x < 0.0);
        assert!(reactions[0].linear.y.abs() < 1e-10);
        assert!(reactions[0].angular.norm() < 1e-10);
        assert!(world.particles[0].velocity.x > -1.0);
        let final_momentum = world.particles[0].velocity * world.particles[0].mass;
        assert!((initial_momentum - final_momentum - reactions[0].linear).norm() < 1e-10);
    }

    #[test]
    fn fixed_particle_does_not_move() {
        let mut fixed = particle(Vector3::new(0.5, 0.5, 0.5));
        fixed.fixed = true;
        fixed.velocity = Vector3::new(1.0, 2.0, 3.0);
        let mut world = MpmWorld::new(vec![fixed], MpmParams::default()).unwrap();
        world.step(0.01).unwrap();
        assert_eq!(world.particles[0].position, Vector3::new(0.5, 0.5, 0.5));
        assert_eq!(world.particles[0].velocity, Vector3::zeros());
    }

    #[test]
    fn world_boundary_prevents_particle_escape() {
        let mut falling = particle(Vector3::new(0.5, 0.5, 0.05));
        falling.velocity.z = -10.0;
        let mut world = MpmWorld::new(
            vec![falling],
            MpmParams {
                gravity: Vector3::zeros(),
                max_substep: 0.001,
                bounds: Some(WorldBounds {
                    min: Vector3::zeros(),
                    max: Vector3::repeat(1.0),
                }),
                ..MpmParams::default()
            },
        )
        .unwrap();
        world.step(0.01).unwrap();
        assert!(world.particles[0].position.z >= 0.04);
        assert!(world.particles[0].velocity.z >= 0.0);
    }

    #[test]
    fn all_materials_complete_finite_steps() {
        let materials = [
            MaterialModel::elastic(1_000.0, 0.2),
            MaterialModel::neo_hookean(1_000.0, 0.2),
            MaterialModel::sand(1_000.0, 0.2, 35.0f64.to_radians(), 0.0),
            MaterialModel::fluid(2_000.0, 7.0, 0.01),
            MaterialModel::snow(1_000.0, 0.2),
        ];
        let particles = materials
            .into_iter()
            .enumerate()
            .map(|(index, material)| {
                MpmParticle::new(
                    Vector3::new(0.3 + index as f64 * 0.1, 0.5, 0.5),
                    0.03,
                    1_000.0,
                    material,
                )
            })
            .collect();
        let mut world = MpmWorld::new(particles, MpmParams::default()).unwrap();
        world.step(0.002).unwrap();
        assert!(world.particles.iter().all(finite_particle_state));
    }

    #[test]
    fn cfl_bound_tightens_for_stiffer_material() {
        let soft = MpmParticle::new(
            Vector3::zeros(),
            0.04,
            1_000.0,
            MaterialModel::elastic(1_000.0, 0.2),
        );
        let stiff = MpmParticle::new(
            Vector3::zeros(),
            0.04,
            1_000.0,
            MaterialModel::elastic(100_000.0, 0.2),
        );
        let params = MpmParams {
            max_substep: 1.0,
            ..MpmParams::default()
        };
        let soft_dt = MpmWorld::new(vec![soft], params.clone())
            .unwrap()
            .stable_timestep();
        let stiff_dt = MpmWorld::new(vec![stiff], params)
            .unwrap()
            .stable_timestep();
        assert!(stiff_dt < soft_dt);
    }

    #[test]
    fn rejects_invalid_particle_and_world_parameters() {
        let mut invalid = particle(Vector3::zeros());
        invalid.mass = 0.0;
        assert_eq!(
            MpmWorld::new(vec![invalid], MpmParams::default()).unwrap_err(),
            MpmError::InvalidInput
        );
        assert_eq!(
            MpmWorld::new(
                Vec::new(),
                MpmParams {
                    cell_width: 0.0,
                    ..MpmParams::default()
                }
            )
            .unwrap_err(),
            MpmError::InvalidInput
        );
    }

    #[test]
    fn moving_sphere_pushes_particle_without_changing_obstacle() {
        let material_point = particle(Vector3::new(0.62, 0.5, 0.5));
        let params = MpmParams {
            gravity: Vector3::zeros(),
            ..MpmParams::default()
        };
        let mut world = MpmWorld::new(vec![material_point], params).unwrap();
        let mut obstacle = RigidObstacle::sphere(Vector3::new(0.5, 0.5, 0.5), 0.1);
        obstacle.linear_velocity.x = 1.0;
        world.set_obstacles(vec![obstacle.clone()]).unwrap();
        world.step(0.001).unwrap();
        assert!(world.particles[0].velocity.x > 0.0);
        assert_eq!(world.obstacles[0], obstacle);
        let (distance, _) = obstacle.surface(world.particles[0].position);
        assert!(distance >= world.particles[0].radius - 1e-12);
    }

    #[test]
    fn rejects_invalid_obstacle_without_replacing_current_set() {
        let mut world = MpmWorld::new(Vec::new(), MpmParams::default()).unwrap();
        let valid = RigidObstacle::sphere(Vector3::zeros(), 1.0);
        world.set_obstacles(vec![valid.clone()]).unwrap();
        let invalid = RigidObstacle::sphere(Vector3::zeros(), -1.0);
        assert_eq!(
            world.set_obstacles(vec![invalid]),
            Err(MpmError::InvalidInput)
        );
        assert_eq!(world.obstacles, vec![valid]);
    }

    #[test]
    fn invalid_particle_batch_does_not_append_or_consume_chunk_id() {
        let mut world = MpmWorld::new(
            vec![particle(Vector3::new(0.0, 0.0, 0.5))],
            MpmParams::default(),
        )
        .unwrap();
        let mut invalid = particle(Vector3::new(0.2, 0.0, 0.5));
        invalid.mass = 0.0;
        assert_eq!(
            world.add_particles(vec![particle(Vector3::new(0.1, 0.0, 0.5)), invalid]),
            Err(MpmError::InvalidInput)
        );
        assert_eq!(world.particles.len(), 1);
        let chunk = world
            .add_particles(vec![particle(Vector3::new(0.3, 0.0, 0.5))])
            .unwrap();
        assert_eq!(chunk.get(), 1);
        assert_eq!(world.particles[1].chunk_id(), chunk);
        assert_eq!(world.remove_chunk(ParticleChunkId::INITIAL).unwrap(), 1);
        assert_eq!(world.particles[0].chunk_id(), chunk);
    }
}
