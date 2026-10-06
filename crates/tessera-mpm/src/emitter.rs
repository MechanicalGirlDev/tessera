//! Deterministic particle patch generation for dynamic MPM scenes.

use nalgebra::Vector3;

use crate::material::MaterialModel;
use crate::sampling::{VolumeSamplingError, sample_closed_mesh_volume};
use crate::world::{MpmError, MpmParticle, MpmWorld, ParticleChunkId};

/// Invalid mesh emission settings, sampling capacity, or world insertion.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum MeshEmitterError {
    /// Mesh geometry or sampling limits are invalid.
    #[error(transparent)]
    Sampling(#[from] VolumeSamplingError),
    /// Particle properties or insertion are invalid.
    #[error(transparent)]
    Particle(#[from] MpmError),
}

/// A closed triangle mesh filled with cubic MPM material points.
#[derive(Debug, Clone)]
pub struct MeshEmitter {
    /// World-space vertices; apply any mesh transform before emission.
    pub vertices: Vec<Vector3<f64>>,
    /// Consistently oriented triangle indices.
    pub triangles: Vec<[u32; 3]>,
    /// Particle spacing; each accepted point represents `spacing^3` volume.
    pub spacing: f64,
    /// Initial material density.
    pub density: f64,
    /// Constitutive model assigned to new particles.
    pub material: MaterialModel,
    /// Initial velocity of each emitted particle.
    pub velocity: Vector3<f64>,
    /// Velocity damping in inverse seconds.
    pub damping: f64,
    /// Render and selection group assigned to new particles.
    pub group_id: u32,
    /// Maximum number of AABB lattice cells to inspect.
    pub max_candidates: usize,
    /// Maximum number of particles to create.
    pub max_particles: usize,
}

impl MeshEmitter {
    /// Construct an emitter with zero initial velocity and damping.
    pub fn new(
        vertices: Vec<Vector3<f64>>,
        triangles: Vec<[u32; 3]>,
        spacing: f64,
        density: f64,
        material: MaterialModel,
        max_candidates: usize,
        max_particles: usize,
    ) -> Self {
        Self {
            vertices,
            triangles,
            spacing,
            density,
            material,
            velocity: Vector3::zeros(),
            damping: 0.0,
            group_id: 0,
            max_candidates,
            max_particles,
        }
    }

    /// Create a removable particle batch without changing a world.
    pub fn sample(&self) -> Result<Vec<MpmParticle>, MeshEmitterError> {
        let radius = self.spacing * 0.5;
        let volume = self.spacing.powi(3);
        let mass = self.density * volume;
        if !radius.is_finite()
            || radius <= 0.0
            || !volume.is_finite()
            || volume <= 0.0
            || !mass.is_finite()
            || mass <= 0.0
            || !self.material.is_valid()
            || self.velocity.iter().any(|value| !value.is_finite())
            || !self.damping.is_finite()
            || self.damping < 0.0
        {
            return Err(MpmError::InvalidInput.into());
        }
        let positions = sample_closed_mesh_volume(
            &self.vertices,
            &self.triangles,
            self.spacing,
            self.max_candidates,
            self.max_particles,
        )?;
        if positions.is_empty() {
            return Err(VolumeSamplingError::InvalidInput.into());
        }
        let mut particles = Vec::new();
        particles
            .try_reserve_exact(positions.len())
            .map_err(|_| MpmError::ParticleCapacity)?;
        for position in positions {
            let mut particle = MpmParticle::new(position, radius, self.density, self.material);
            particle.velocity = self.velocity;
            particle.damping = self.damping;
            particle.group_id = self.group_id;
            particles.push(particle);
        }
        Ok(particles)
    }

    /// Fill a closed mesh and insert the particles as one removable chunk.
    pub fn emit(&self, world: &mut MpmWorld) -> Result<ParticleChunkId, MeshEmitterError> {
        Ok(world.add_particles(self.sample()?)?)
    }
}

/// A centered three-dimensional lattice emitted as one removable chunk.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoxEmitter {
    /// Center of the emitted patch.
    pub center: Vector3<f64>,
    /// Number of particles along each axis.
    pub counts: [u32; 3],
    /// Center-to-center spacing along each axis.
    pub spacing: Vector3<f64>,
    /// Particle sampling radius.
    pub radius: f64,
    /// Initial material density.
    pub density: f64,
    /// Constitutive model assigned to new particles.
    pub material: MaterialModel,
    /// Initial velocity of each emitted particle.
    pub velocity: Vector3<f64>,
    /// Velocity damping coefficient in inverse seconds.
    pub damping: f64,
    /// Render and selection group assigned to new particles.
    pub group_id: u32,
}

impl BoxEmitter {
    /// Construct a centered patch with zero initial velocity and damping.
    pub fn new(
        center: Vector3<f64>,
        counts: [u32; 3],
        spacing: Vector3<f64>,
        radius: f64,
        density: f64,
        material: MaterialModel,
    ) -> Self {
        Self {
            center,
            counts,
            spacing,
            radius,
            density,
            material,
            velocity: Vector3::zeros(),
            damping: 0.0,
            group_id: 0,
        }
    }

    /// Create one batch of particles without changing a world.
    pub fn sample(&self) -> Result<Vec<MpmParticle>, MpmError> {
        let count = self
            .counts
            .iter()
            .try_fold(1usize, |product, axis| product.checked_mul(*axis as usize))
            .ok_or(MpmError::ParticleCapacity)?;
        let volume = (2.0 * self.radius).powi(3);
        if count == 0
            || self.center.iter().any(|value| !value.is_finite())
            || self
                .spacing
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
            || !self.radius.is_finite()
            || self.radius <= 0.0
            || !self.density.is_finite()
            || self.density <= 0.0
            || !volume.is_finite()
            || volume <= 0.0
            || !(self.density * volume).is_finite()
            || self.density * volume <= 0.0
            || !self.material.is_valid()
            || self.velocity.iter().any(|value| !value.is_finite())
            || !self.damping.is_finite()
            || self.damping < 0.0
            || (0..3).any(|axis| {
                let half_extent = f64::from(self.counts[axis] - 1) * self.spacing[axis] * 0.5;
                !half_extent.is_finite()
                    || !(self.center[axis] - half_extent).is_finite()
                    || !(self.center[axis] + half_extent).is_finite()
            })
        {
            return Err(MpmError::InvalidInput);
        }
        let mut particles = Vec::new();
        particles
            .try_reserve_exact(count)
            .map_err(|_| MpmError::ParticleCapacity)?;
        let center_index = Vector3::new(
            (f64::from(self.counts[0]) - 1.0) * 0.5,
            (f64::from(self.counts[1]) - 1.0) * 0.5,
            (f64::from(self.counts[2]) - 1.0) * 0.5,
        );
        for x in 0..self.counts[0] {
            for y in 0..self.counts[1] {
                for z in 0..self.counts[2] {
                    let index = Vector3::new(f64::from(x), f64::from(y), f64::from(z));
                    let position =
                        self.center + (index - center_index).component_mul(&self.spacing);
                    let mut particle =
                        MpmParticle::new(position, self.radius, self.density, self.material);
                    particle.velocity = self.velocity;
                    particle.damping = self.damping;
                    particle.group_id = self.group_id;
                    particles.push(particle);
                }
            }
        }
        Ok(particles)
    }

    /// Emit one patch into a world and return its removable chunk handle.
    pub fn emit(&self, world: &mut MpmWorld) -> Result<ParticleChunkId, MpmError> {
        world.add_particles(self.sample()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::MpmParams;

    #[test]
    fn mesh_emitter_creates_mass_and_removable_world_chunk() {
        let vertices = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(1.0, 0.0, 1.0),
            Vector3::new(1.0, 1.0, 1.0),
            Vector3::new(0.0, 1.0, 1.0),
        ];
        let triangles = vec![
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ];
        let mut emitter = MeshEmitter::new(
            vertices,
            triangles,
            0.25,
            1_000.0,
            MaterialModel::elastic(1_000.0, 0.2),
            64,
            64,
        );
        emitter.velocity = Vector3::new(0.0, 0.0, 1.0);
        emitter.group_id = 7;
        let mut world = MpmWorld::new(Vec::new(), MpmParams::default()).unwrap();
        let chunk = emitter.emit(&mut world).unwrap();
        assert_eq!(world.particles.len(), 64);
        assert!(world.particles.iter().all(|particle| {
            particle.rest_volume == 0.25_f64.powi(3)
                && particle.mass == 1_000.0 * 0.25_f64.powi(3)
                && particle.radius == 0.125
                && particle.velocity == emitter.velocity
                && particle.group_id == 7
                && particle.chunk_id() == chunk
        }));
        world.step(0.001).unwrap();
        assert_eq!(world.remove_chunk(chunk).unwrap(), 64);
        assert!(world.particles.is_empty());
        emitter.max_particles = 63;
        assert_eq!(
            emitter.emit(&mut world),
            Err(MeshEmitterError::Sampling(VolumeSamplingError::Capacity))
        );
        assert!(world.particles.is_empty());
    }

    #[test]
    fn box_emitter_centers_patch_and_sets_properties() {
        let mut emitter = BoxEmitter::new(
            Vector3::new(2.0, 3.0, 4.0),
            [2, 1, 3],
            Vector3::new(0.2, 0.3, 0.4),
            0.05,
            1_000.0,
            MaterialModel::sand(1_000.0, 0.2, 0.5, 0.0),
        );
        emitter.velocity = Vector3::new(1.0, 0.0, -1.0);
        emitter.group_id = 7;
        let particles = emitter.sample().unwrap();
        assert_eq!(particles.len(), 6);
        assert_eq!(particles[0].position, Vector3::new(1.9, 3.0, 3.6));
        assert_eq!(particles[5].position, Vector3::new(2.1, 3.0, 4.4));
        assert!(particles.iter().all(|particle| {
            particle.velocity == emitter.velocity && particle.group_id == emitter.group_id
        }));
    }

    #[test]
    fn emitted_chunks_can_be_removed_oldest_first() {
        let mut world = MpmWorld::new(Vec::new(), MpmParams::default()).unwrap();
        let mut emitter = BoxEmitter::new(
            Vector3::new(0.0, 0.0, 0.5),
            [2, 1, 1],
            Vector3::repeat(0.1),
            0.04,
            1_000.0,
            MaterialModel::elastic(1_000.0, 0.2),
        );
        let first = emitter.emit(&mut world).unwrap();
        emitter.center.x = 1.0;
        let second = emitter.emit(&mut world).unwrap();
        assert_ne!(first, second);
        assert_eq!(world.particles.len(), 4);
        assert_eq!(world.remove_chunk(first).unwrap(), 2);
        assert_eq!(world.particles.len(), 2);
        assert!(
            world
                .particles
                .iter()
                .all(|particle| particle.chunk_id() == second)
        );
        world.step(0.001).unwrap();
        assert_eq!(world.remove_chunk(second).unwrap(), 2);
        assert!(world.particles.is_empty());
        assert_eq!(world.remove_chunk(first), Err(MpmError::UnknownChunk));
    }

    #[test]
    fn rejects_invalid_emission_without_mutating_world() {
        let mut world = MpmWorld::new(Vec::new(), MpmParams::default()).unwrap();
        let emitter = BoxEmitter::new(
            Vector3::zeros(),
            [0, 2, 2],
            Vector3::repeat(0.1),
            0.04,
            1_000.0,
            MaterialModel::elastic(1_000.0, 0.2),
        );
        assert_eq!(emitter.emit(&mut world), Err(MpmError::InvalidInput));
        assert!(world.particles.is_empty());
    }
}
