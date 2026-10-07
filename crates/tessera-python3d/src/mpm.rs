//! Python-facing Material Point Method world.

use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use std::sync::Mutex;

use nalgebra::{Quaternion as NaQuaternion, UnitQuaternion, Vector2};
use tessera_mpm::{
    GpuMpmResidentSession, GpuMpmTransfers, MaterialModel, MeshEmitter, MpmParams, MpmParticle,
    MpmWorld as CoreMpmWorld, ObstacleBoundary, ParticleChunkId, RigidObstacle, WorldBounds,
    sample_closed_mesh_volume,
};
use tessera_physics::gpu_contact_pipeline::GpuContactDevice;
use tessera_physics::gpu_rigid_sphere_world::GpuRigidSphereWorld as CoreGpuSphereWorld;

use crate::{
    GpuPrimitiveWorld, GpuSphereWorld, GpuTriangle, Quaternion, TesseraError, Vec3, failed, locked,
};

/// Sample particle centers inside a closed, consistently oriented 3D mesh.
/// The caller chooses mass, volume, and material when constructing MPM particles.
#[uniffi::export]
pub fn sample_mpm_closed_mesh_volume(
    vertices: Vec<Vec3>,
    triangles: Vec<GpuTriangle>,
    spacing: f64,
    max_candidates: u32,
    max_samples: u32,
) -> Result<Vec<Vec3>, TesseraError> {
    let vertices = vertices.into_iter().map(Vec3::nalgebra).collect::<Vec<_>>();
    let triangles = triangles
        .into_iter()
        .map(|face| [face.a, face.b, face.c])
        .collect::<Vec<_>>();
    Ok(sample_closed_mesh_volume(
        &vertices,
        &triangles,
        spacing,
        max_candidates as usize,
        max_samples as usize,
    )
    .map_err(failed)?
    .into_iter()
    .map(Into::into)
    .collect())
}

/// Constitutive model assigned to a material point.
#[derive(Clone, Debug, uniffi::Enum)]
pub enum MpmMaterial {
    /// Corotated linear elasticity.
    Elastic {
        /// Young's modulus in pascals.
        young_modulus: f64,
        /// Poisson ratio.
        poisson_ratio: f64,
    },
    /// Compressible Neo-Hookean elasticity.
    NeoHookean {
        /// Young's modulus in pascals.
        young_modulus: f64,
        /// Poisson ratio.
        poisson_ratio: f64,
    },
    /// Drucker-Prager granular material.
    Sand {
        /// Young's modulus in pascals.
        young_modulus: f64,
        /// Poisson ratio.
        poisson_ratio: f64,
        /// Internal friction angle in radians.
        friction_angle: f64,
        /// Tensile yield offset.
        cohesion: f64,
    },
    /// Weakly compressible fluid.
    Fluid {
        /// Bulk modulus in pascals.
        bulk_modulus: f64,
        /// Tait exponent.
        gamma: f64,
        /// Dynamic viscosity in pascal seconds.
        viscosity: f64,
        /// Tensile stiffness relative to the bulk modulus.
        tensile_stiffness: f64,
    },
    /// Snow with singular-value yield limits.
    Snow {
        /// Young's modulus in pascals.
        young_modulus: f64,
        /// Poisson ratio.
        poisson_ratio: f64,
        /// Maximum elastic compression.
        critical_compression: f64,
        /// Maximum elastic stretch.
        critical_stretch: f64,
        /// Exponential compaction hardening coefficient.
        hardening: f64,
    },
    /// Neo-Hookean sand with Nexus' hardening Drucker-Prager plasticity.
    SandNeoHookean {
        /// Young's modulus in pascals.
        young_modulus: f64,
        /// Poisson ratio.
        poisson_ratio: f64,
        /// Asymptotic friction angle in radians; use 35 degrees for Nexus defaults.
        friction_angle: f64,
        /// Tensile yield offset in logarithmic volumetric strain.
        cohesion: f64,
    },
}

impl From<MpmMaterial> for MaterialModel {
    fn from(value: MpmMaterial) -> Self {
        match value {
            MpmMaterial::Elastic {
                young_modulus,
                poisson_ratio,
            } => Self::LinearElastic {
                young_modulus,
                poisson_ratio,
            },
            MpmMaterial::NeoHookean {
                young_modulus,
                poisson_ratio,
            } => Self::NeoHookean {
                young_modulus,
                poisson_ratio,
            },
            MpmMaterial::Sand {
                young_modulus,
                poisson_ratio,
                friction_angle,
                cohesion,
            } => Self::Sand {
                young_modulus,
                poisson_ratio,
                friction_angle,
                cohesion,
            },
            MpmMaterial::Fluid {
                bulk_modulus,
                gamma,
                viscosity,
                tensile_stiffness,
            } => Self::Fluid {
                bulk_modulus,
                gamma,
                viscosity,
                tensile_stiffness,
            },
            MpmMaterial::Snow {
                young_modulus,
                poisson_ratio,
                critical_compression,
                critical_stretch,
                hardening,
            } => Self::Snow {
                young_modulus,
                poisson_ratio,
                critical_compression,
                critical_stretch,
                hardening,
            },
            MpmMaterial::SandNeoHookean {
                young_modulus,
                poisson_ratio,
                friction_angle,
                cohesion,
            } => Self::SandNeoHookean {
                young_modulus,
                poisson_ratio,
                friction_angle,
                cohesion,
            },
        }
    }
}

/// New material point with initial velocity and density.
#[derive(Clone, Debug, uniffi::Record)]
pub struct MpmParticleInput {
    /// World-space position.
    pub position: Vec3,
    /// World-space velocity.
    pub velocity: Vec3,
    /// Sampling and collision radius in metres.
    pub radius: f64,
    /// Mass density in kilograms per cubic metre.
    pub density: f64,
    /// Constitutive model.
    pub material: MpmMaterial,
    /// Whether integration is disabled.
    pub enabled: bool,
    /// Whether the particle pose is fixed.
    pub fixed: bool,
    /// Velocity damping in inverse seconds.
    pub damping: f64,
}

/// Closed mesh and particle settings for one removable MPM chunk.
#[derive(Clone, Debug, uniffi::Record)]
pub struct MpmMeshEmissionInput {
    /// World-space mesh vertices.
    pub vertices: Vec<Vec3>,
    /// Consistently oriented closed triangle faces.
    pub triangles: Vec<GpuTriangle>,
    /// Cubic particle spacing in metres.
    pub spacing: f64,
    /// Initial mass density in kilograms per cubic metre.
    pub density: f64,
    /// Constitutive model for every new particle.
    pub material: MpmMaterial,
    /// Initial world-space velocity.
    pub velocity: Vec3,
    /// Velocity damping in inverse seconds.
    pub damping: f64,
    /// Maximum AABB lattice cells to inspect.
    pub max_candidates: u32,
    /// Maximum particles to create.
    pub max_particles: u32,
}

impl From<MpmParticleInput> for MpmParticle {
    fn from(input: MpmParticleInput) -> Self {
        let mut particle = Self::new(
            input.position.nalgebra(),
            input.radius,
            input.density,
            input.material.into(),
        );
        particle.velocity = input.velocity.nalgebra();
        particle.enabled = input.enabled;
        particle.fixed = input.fixed;
        particle.damping = input.damping;
        particle
    }
}

/// Current material point state in insertion order.
#[derive(Clone, Debug, uniffi::Record)]
pub struct MpmParticleState {
    /// World-space position.
    pub position: Vec3,
    /// World-space velocity.
    pub velocity: Vec3,
    /// Particle mass in kilograms.
    pub mass: f64,
    /// Undeformed volume in cubic metres.
    pub rest_volume: f64,
    /// Sampling and collision radius.
    pub radius: f64,
    /// Column-major deformation gradient.
    pub deformation: Vec<f64>,
    /// Column-major APIC affine velocity field.
    pub affine: Vec<f64>,
    /// Plastic deformation determinant.
    pub plastic_det: f64,
    /// Accumulated plastic hardening.
    pub hardening: f64,
    /// Accumulated logarithmic volume correction.
    pub log_volume_gain: f64,
    /// Stable particle batch handle.
    pub chunk_id: u64,
    /// Whether integration is enabled.
    pub enabled: bool,
    /// Whether the particle pose is fixed.
    pub fixed: bool,
    /// Particle-grid transfer region; different colors do not share grid nodes.
    pub transfer_color: u16,
}

impl From<&MpmParticle> for MpmParticleState {
    fn from(particle: &MpmParticle) -> Self {
        Self {
            position: particle.position.into(),
            velocity: particle.velocity.into(),
            mass: particle.mass,
            rest_volume: particle.rest_volume,
            radius: particle.radius,
            deformation: particle.deformation.as_slice().to_vec(),
            affine: particle.affine.as_slice().to_vec(),
            plastic_det: particle.plastic.plastic_det,
            hardening: particle.plastic.hardening,
            log_volume_gain: particle.plastic.log_volume_gain,
            chunk_id: particle.chunk_id().get(),
            enabled: particle.enabled,
            fixed: particle.fixed,
            transfer_color: particle.transfer_color,
        }
    }
}

/// Shape of one prescribed obstacle that pushes MPM particles.
#[allow(variant_size_differences)]
#[derive(Clone, Debug, uniffi::Enum)]
pub enum MpmObstacleShape {
    /// Sphere centered on its pose.
    Sphere {
        /// Collision radius.
        radius: f64,
    },
    /// Oriented box.
    Box {
        /// Local half extents.
        half_extents: Vec3,
    },
    /// Z-axis capsule.
    Capsule {
        /// Half length of the center segment.
        half_height: f64,
        /// Rounded border radius.
        radius: f64,
    },
    /// Z-axis circular cylinder.
    Cylinder {
        /// Half height.
        half_height: f64,
        /// Circular radius.
        radius: f64,
    },
    /// Z-axis circular cone.
    Cone {
        /// Half height.
        half_height: f64,
        /// Base radius.
        radius: f64,
    },
    /// Finite one-sided local XY ground plane.
    Ground {
        /// Half length along local X.
        half_extent_x: f64,
        /// Half length along local Y.
        half_extent_y: f64,
    },
    /// Thin triangular prism with three local vertices.
    TrianglePrism {
        /// Exactly three local vertices.
        vertices: Vec<Vec3>,
        /// Half thickness along the face normal.
        half_thickness: f64,
    },
    /// Convex polyhedron from local hull vertices and outward face normals.
    Convex {
        /// Local hull vertices.
        vertices: Vec<Vec3>,
        /// Outward supporting face normals.
        face_normals: Vec<Vec3>,
    },
}

/// Prescribed obstacle pose, motion, and friction.
#[derive(Clone, Debug, uniffi::Record)]
pub struct MpmObstacleInput {
    /// Collision shape.
    pub shape: MpmObstacleShape,
    /// World-space center.
    pub center: Vec3,
    /// Local-to-world XYZW unit orientation.
    pub orientation: Quaternion,
    /// World-space linear velocity at the center.
    pub linear_velocity: Vec3,
    /// World-space angular velocity.
    pub angular_velocity: Vec3,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
}

/// Velocity response at an MPM obstacle surface.
#[derive(Clone, Copy, Debug, uniffi::Enum)]
pub enum MpmBoundary {
    /// Coulomb friction with separation along the contact normal.
    Slip,
    /// Match the obstacle surface velocity.
    Stick,
    /// Remove only incoming normal velocity.
    Separate,
    /// Project penetrated positions without changing velocity.
    NonReflecting,
}

impl From<MpmBoundary> for ObstacleBoundary {
    fn from(value: MpmBoundary) -> Self {
        match value {
            MpmBoundary::Slip => Self::Slip,
            MpmBoundary::Stick => Self::Stick,
            MpmBoundary::Separate => Self::Separate,
            MpmBoundary::NonReflecting => Self::NonReflecting,
        }
    }
}

impl From<ObstacleBoundary> for MpmBoundary {
    fn from(value: ObstacleBoundary) -> Self {
        match value {
            ObstacleBoundary::Slip => Self::Slip,
            ObstacleBoundary::Stick => Self::Stick,
            ObstacleBoundary::Separate => Self::Separate,
            ObstacleBoundary::NonReflecting => Self::NonReflecting,
        }
    }
}

fn unit_orientation(input: Quaternion) -> Result<UnitQuaternion<f64>, TesseraError> {
    let norm_squared =
        input.x * input.x + input.y * input.y + input.z * input.z + input.w * input.w;
    if !norm_squared.is_finite() || (norm_squared - 1.0).abs() > 1e-5 {
        return Err(failed("obstacle orientation must be a unit quaternion"));
    }
    Ok(UnitQuaternion::new_normalize(NaQuaternion::new(
        input.w, input.x, input.y, input.z,
    )))
}

fn obstacle(input: MpmObstacleInput) -> Result<RigidObstacle, TesseraError> {
    let center = input.center.nalgebra();
    let orientation = unit_orientation(input.orientation)?;
    let mut obstacle = match input.shape {
        MpmObstacleShape::Sphere { radius } => RigidObstacle::sphere(center, radius),
        MpmObstacleShape::Box { half_extents } => {
            RigidObstacle::cuboid(center, half_extents.nalgebra(), orientation)
        }
        MpmObstacleShape::Capsule {
            half_height,
            radius,
        } => RigidObstacle::capsule(center, half_height, radius, orientation),
        MpmObstacleShape::Cylinder {
            half_height,
            radius,
        } => RigidObstacle::cylinder(center, half_height, radius, orientation),
        MpmObstacleShape::Cone {
            half_height,
            radius,
        } => RigidObstacle::cone(center, half_height, radius, orientation),
        MpmObstacleShape::Ground {
            half_extent_x,
            half_extent_y,
        } => RigidObstacle::ground(
            center,
            Vector2::new(half_extent_x, half_extent_y),
            orientation,
        ),
        MpmObstacleShape::TrianglePrism {
            vertices,
            half_thickness,
        } => {
            let vertices: [Vec3; 3] = vertices
                .try_into()
                .map_err(|_| failed("triangle prism requires exactly three vertices"))?;
            RigidObstacle::triangle_prism(
                center,
                vertices.map(Vec3::nalgebra),
                half_thickness,
                orientation,
            )
        }
        MpmObstacleShape::Convex {
            vertices,
            face_normals,
        } => RigidObstacle::convex(
            center,
            &vertices.into_iter().map(Vec3::nalgebra).collect::<Vec<_>>(),
            &face_normals
                .into_iter()
                .map(Vec3::nalgebra)
                .collect::<Vec<_>>(),
            orientation,
        ),
    };
    obstacle.orientation = orientation;
    obstacle.linear_velocity = input.linear_velocity.nalgebra();
    obstacle.angular_velocity = input.angular_velocity.nalgebra();
    obstacle.friction = input.friction;
    if !obstacle.is_valid() {
        return Err(failed("invalid MPM obstacle"));
    }
    Ok(obstacle)
}

#[derive(Debug)]
struct MpmGpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    transfers: GpuMpmTransfers,
}

impl MpmGpu {
    fn new() -> Result<Self, TesseraError> {
        let context = GpuContactDevice::new().map_err(failed)?;
        Ok(Self::from_device(context.device(), context.queue()))
    }

    fn from_device(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self {
            device: device.clone(),
            queue: queue.clone(),
            transfers: GpuMpmTransfers::new(device),
        }
    }
}

#[derive(Debug)]
enum MpmRigidSource {
    Primitive(Arc<GpuPrimitiveWorld>),
    Sphere(Arc<GpuSphereWorld>),
}

#[derive(Clone, Copy)]
enum MpmCouplingMode {
    OneWay,
    TwoWay,
    MpmOwned,
}

impl MpmRigidSource {
    fn same(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Primitive(a), Self::Primitive(b)) => Arc::ptr_eq(a, b),
            (Self::Sphere(a), Self::Sphere(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
}

#[derive(Debug)]
struct MpmInner {
    world: CoreMpmWorld,
    chunks: BTreeMap<u64, ParticleChunkId>,
    gpu: Option<MpmGpu>,
    resident: Option<(f64, GpuMpmResidentSession<'static>)>,
    coupled: Option<(MpmRigidSource, tessera_mpm::GpuRigidMpmCoupler)>,
}

impl MpmInner {
    fn pending_steps(&self) -> u64 {
        self.resident
            .as_ref()
            .map_or(0, |(_, session)| session.pending_substeps())
    }

    fn reject_pending(&self) -> Result<(), TesseraError> {
        if self.pending_steps() > 0 {
            Err(failed(
                "synchronize pending GPU substeps before editing MPM state",
            ))
        } else {
            Ok(())
        }
    }

    fn add_particles(&mut self, particles: Vec<MpmParticle>) -> Result<u64, TesseraError> {
        self.reject_pending()?;
        let chunk = if self.coupled.is_none() {
            if let Some((_, session)) = &mut self.resident {
                let chunk = session.add_particles(particles).map_err(failed)?;
                self.world = session.world().clone();
                chunk
            } else {
                self.world.add_particles(particles).map_err(failed)?
            }
        } else {
            let chunk = self.world.add_particles(particles).map_err(failed)?;
            self.resident = None;
            self.coupled = None;
            chunk
        };
        let _previous = self.chunks.insert(chunk.get(), chunk);
        Ok(chunk.get())
    }

    fn remove_chunk(&mut self, chunk_id: u64) -> Result<u64, TesseraError> {
        self.reject_pending()?;
        let chunk = *self
            .chunks
            .get(&chunk_id)
            .ok_or_else(|| failed("unknown MPM particle chunk"))?;
        let removed = if self.coupled.is_none() {
            if let Some((_, session)) = &mut self.resident {
                let removed = session.remove_chunk(chunk).map_err(failed)?;
                self.world = session.world().clone();
                removed
            } else {
                self.world.remove_chunk(chunk).map_err(failed)?
            }
        } else {
            let removed = self.world.remove_chunk(chunk).map_err(failed)?;
            self.resident = None;
            self.coupled = None;
            removed
        };
        let _removed = self.chunks.remove(&chunk_id);
        u64::try_from(removed).map_err(failed)
    }
}

/// Three-dimensional CPU and WebGPU Material Point Method world.
#[derive(Debug, uniffi::Object)]
pub struct MpmWorld {
    inner: Mutex<MpmInner>,
}

#[uniffi::export]
impl MpmWorld {
    /// Create a world with optional bounds required for fixed GPU substeps.
    #[uniffi::constructor]
    pub fn new(
        particles: Vec<MpmParticleInput>,
        gravity: Vec3,
        cell_width: f64,
        max_substep: f64,
        bounds_min: Option<Vec3>,
        bounds_max: Option<Vec3>,
    ) -> Result<Arc<Self>, TesseraError> {
        let bounds = match (bounds_min, bounds_max) {
            (None, None) => None,
            (Some(min), Some(max)) => Some(WorldBounds {
                min: min.nalgebra(),
                max: max.nalgebra(),
            }),
            _ => return Err(failed("both MPM bounds corners must be supplied")),
        };
        let world = CoreMpmWorld::new(
            particles.into_iter().map(Into::into).collect(),
            MpmParams {
                gravity: gravity.nalgebra(),
                cell_width,
                max_substep,
                bounds,
            },
        )
        .map_err(failed)?;
        let mut chunks = BTreeMap::new();
        if !world.particles.is_empty() {
            let _previous = chunks.insert(ParticleChunkId::INITIAL.get(), ParticleChunkId::INITIAL);
        }
        Ok(Arc::new(Self {
            inner: Mutex::new(MpmInner {
                world,
                chunks,
                gpu: None,
                resident: None,
                coupled: None,
            }),
        }))
    }

    /// Advance with the deterministic CPU reference path.
    pub fn step(&self, dt: f64) -> Result<(), TesseraError> {
        let mut inner = locked(&self.inner)?;
        inner.reject_pending()?;
        inner.resident = None;
        inner.coupled = None;
        inner.world.step(dt).map_err(failed)
    }

    /// Advance with WebGPU transfers and automatic CFL substeps.
    pub fn step_gpu(&self, dt: f64) -> Result<(), TesseraError> {
        let mut inner = locked(&self.inner)?;
        inner.reject_pending()?;
        inner.resident = None;
        inner.coupled = None;
        if inner.gpu.is_none() {
            inner.gpu = Some(MpmGpu::new()?);
        }
        let MpmInner { world, gpu, .. } = &mut *inner;
        let gpu = gpu
            .as_ref()
            .ok_or_else(|| failed("GPU context unavailable"))?;
        world
            .step_with_gpu_transfers(&gpu.transfers, &gpu.device, &gpu.queue, dt)
            .map_err(failed)
    }

    /// Advance fixed GPU substeps, reusing buffers while timestep and state agree.
    /// Particle edits and forces rebuild the session; obstacle updates retain it.
    pub fn step_gpu_fixed(&self, substep_dt: f64, steps: u32) -> Result<(), TesseraError> {
        if !substep_dt.is_finite() || substep_dt <= 0.0 || steps == 0 || steps > 64 {
            return Err(failed("invalid fixed GPU substep input"));
        }
        let mut inner = locked(&self.inner)?;
        inner.reject_pending()?;
        if inner.coupled.take().is_some() {
            inner.resident = None;
        }
        if inner.gpu.is_none() {
            inner.gpu = Some(MpmGpu::new()?);
        }
        if inner
            .resident
            .as_ref()
            .is_none_or(|(dt, _)| *dt != substep_dt)
        {
            let gpu = inner
                .gpu
                .as_ref()
                .ok_or_else(|| failed("GPU context unavailable"))?;
            let session = GpuMpmResidentSession::new(
                &gpu.transfers,
                &gpu.device,
                &gpu.queue,
                inner.world.clone(),
                substep_dt,
            )
            .map_err(failed)?
            .into_owned();
            inner.resident = Some((substep_dt, session));
        }
        let result = inner
            .resident
            .as_mut()
            .ok_or_else(|| failed("GPU session unavailable"))?
            .1
            .step(steps);
        if let Err(error) = result {
            inner.resident = None;
            return Err(failed(error));
        }
        let updated = inner
            .resident
            .as_ref()
            .ok_or_else(|| failed("GPU session unavailable"))?
            .1
            .world()
            .clone();
        inner.world = updated;
        Ok(())
    }

    /// Read all current particles and their persistent APIC state.
    pub fn particles(&self) -> Result<Vec<MpmParticleState>, TesseraError> {
        Ok(locked(&self.inner)?
            .world
            .particles
            .iter()
            .map(Into::into)
            .collect())
    }

    /// Number of explicit substeps completed so far.
    pub fn substeps(&self) -> Result<u64, TesseraError> {
        Ok(locked(&self.inner)?.world.substeps)
    }

    /// Number of submitted GPU substeps not yet synchronized to CPU state.
    pub fn pending_gpu_substeps(&self) -> Result<u64, TesseraError> {
        Ok(locked(&self.inner)?
            .resident
            .as_ref()
            .map_or(0, |(_, session)| session.pending_substeps()))
    }

    /// Wait for queued GPU substeps, validate them, and refresh CPU particles.
    pub fn synchronize_gpu(&self) -> Result<(), TesseraError> {
        let mut inner = locked(&self.inner)?;
        let Some((_, session)) = &mut inner.resident else {
            return Ok(());
        };
        if let Err(error) = session.synchronize() {
            inner.resident = None;
            inner.coupled = None;
            return Err(failed(error));
        }
        inner.world = session.world().clone();
        Ok(())
    }

    /// Add a validated particle batch and return its stable chunk handle.
    pub fn add_particles(&self, particles: Vec<MpmParticleInput>) -> Result<u64, TesseraError> {
        locked(&self.inner)?.add_particles(particles.into_iter().map(Into::into).collect())
    }

    /// Fill a closed world-space mesh with particles and return the removable chunk handle.
    pub fn add_closed_mesh_particles(
        &self,
        input: MpmMeshEmissionInput,
    ) -> Result<u64, TesseraError> {
        let mut emitter = MeshEmitter::new(
            input.vertices.into_iter().map(Vec3::nalgebra).collect(),
            input
                .triangles
                .into_iter()
                .map(|face| [face.a, face.b, face.c])
                .collect(),
            input.spacing,
            input.density,
            input.material.into(),
            input.max_candidates as usize,
            input.max_particles as usize,
        );
        emitter.velocity = input.velocity.nalgebra();
        emitter.damping = input.damping;
        let particles = emitter.sample().map_err(failed)?;
        locked(&self.inner)?.add_particles(particles)
    }

    /// Remove every particle in the specified chunk.
    pub fn remove_chunk(&self, chunk_id: u64) -> Result<u64, TesseraError> {
        locked(&self.inner)?.remove_chunk(chunk_id)
    }

    /// Replace the prescribed obstacles used by the next CPU or GPU step.
    pub fn set_obstacles(&self, obstacles: Vec<MpmObstacleInput>) -> Result<(), TesseraError> {
        let obstacles = obstacles
            .into_iter()
            .map(obstacle)
            .collect::<Result<Vec<_>, _>>()?;
        self.replace_obstacles(obstacles)
    }

    /// Set boundary responses in obstacle order without replacing their shapes.
    /// Call after installing explicit obstacles or initializing rigid coupling.
    pub fn set_obstacle_boundaries(
        &self,
        boundaries: Vec<MpmBoundary>,
    ) -> Result<(), TesseraError> {
        let mut inner = locked(&self.inner)?;
        inner.reject_pending()?;
        if boundaries.len() != inner.world.obstacles.len() {
            return Err(failed("MPM boundary count must match obstacle count"));
        }
        let mut obstacles = inner.world.obstacles.clone();
        for (obstacle, boundary) in obstacles.iter_mut().zip(boundaries) {
            obstacle.boundary = boundary.into();
        }
        if let Some((_, session)) = &mut inner.resident {
            session.set_obstacles(obstacles.clone()).map_err(failed)?;
        }
        inner.world.set_obstacles(obstacles).map_err(failed)
    }

    /// Read boundary responses in current obstacle order.
    pub fn obstacle_boundaries(&self) -> Result<Vec<MpmBoundary>, TesseraError> {
        Ok(locked(&self.inner)?
            .world
            .obstacles
            .iter()
            .map(|obstacle| obstacle.boundary.into())
            .collect())
    }

    /// Assign CPIC groups to triangle prism obstacles in obstacle order.
    /// None disables automatic side coloring; groups must be below 32.
    pub fn set_obstacle_cpic_groups(&self, groups: Vec<Option<u8>>) -> Result<(), TesseraError> {
        let mut inner = locked(&self.inner)?;
        inner.reject_pending()?;
        if groups.len() != inner.world.obstacles.len() {
            return Err(failed("MPM CPIC group count must match obstacle count"));
        }
        let mut obstacles = inner.world.obstacles.clone();
        for (obstacle, group) in obstacles.iter_mut().zip(groups) {
            obstacle.cpic_group = group;
            if !obstacle.is_valid() {
                return Err(failed("invalid MPM CPIC group or obstacle shape"));
            }
        }
        if let Some((_, session)) = &mut inner.resident {
            session.set_obstacles(obstacles.clone()).map_err(failed)?;
        }
        inner.world.set_obstacles(obstacles).map_err(failed)
    }

    /// Read CPIC groups in current obstacle order.
    pub fn obstacle_cpic_groups(&self) -> Result<Vec<Option<u8>>, TesseraError> {
        Ok(locked(&self.inner)?
            .world
            .obstacles
            .iter()
            .map(|obstacle| obstacle.cpic_group)
            .collect())
    }

    /// Refresh one-way MPM obstacles from the latest GPU mixed-primitive state.
    /// Resident rigid state is read back once; call after each rigid step.
    pub fn sync_gpu_primitive_world(
        &self,
        rigid: Arc<GpuPrimitiveWorld>,
    ) -> Result<(), TesseraError> {
        let obstacles = {
            let rigid = locked(&rigid.inner)?;
            tessera_mpm::gpu_rigid_world_obstacles(&rigid.world).map_err(failed)?
        };
        self.replace_obstacles(obstacles)
    }

    /// Refresh one-way MPM obstacles from the latest GPU sphere state.
    /// Resident rigid state is read back once; call after each rigid step.
    pub fn sync_gpu_sphere_world(&self, rigid: Arc<GpuSphereWorld>) -> Result<(), TesseraError> {
        let obstacles = {
            let rigid = locked(&rigid.world)?;
            tessera_mpm::gpu_rigid_world_obstacles(&rigid).map_err(failed)?
        };
        self.replace_obstacles(obstacles)
    }

    /// Advance fixed GPU MPM substeps using live mixed-shape rigid state.
    /// Step the rigid world first. The initial call installs obstacles with one
    /// readback; subsequent calls with the same rigid world and timestep update
    /// obstacle poses directly on GPU.
    pub fn step_gpu_fixed_with_primitive_world(
        &self,
        rigid: Arc<GpuPrimitiveWorld>,
        substep_dt: f64,
        steps: u32,
    ) -> Result<(), TesseraError> {
        let source = MpmRigidSource::Primitive(Arc::clone(&rigid));
        let rigid = locked(&rigid.inner)?;
        self.step_gpu_fixed_coupled(
            &rigid.world,
            source,
            substep_dt,
            steps,
            true,
            MpmCouplingMode::OneWay,
        )
    }

    /// Advance fixed GPU MPM substeps using live sphere rigid state.
    /// The first call installs obstacles; later calls reuse device-resident
    /// mappings while the source world and timestep remain unchanged.
    pub fn step_gpu_fixed_with_sphere_world(
        &self,
        rigid: Arc<GpuSphereWorld>,
        substep_dt: f64,
        steps: u32,
    ) -> Result<(), TesseraError> {
        let source = MpmRigidSource::Sphere(Arc::clone(&rigid));
        let rigid = locked(&rigid.world)?;
        self.step_gpu_fixed_coupled(
            &rigid,
            source,
            substep_dt,
            steps,
            true,
            MpmCouplingMode::OneWay,
        )
    }

    /// Queue mixed-shape rigid obstacle updates and MPM substeps on GPU.
    /// CPU particles remain at the last synchronized state until `synchronize_gpu`.
    pub fn submit_gpu_fixed_with_primitive_world(
        &self,
        rigid: Arc<GpuPrimitiveWorld>,
        substep_dt: f64,
        steps: u32,
    ) -> Result<(), TesseraError> {
        let source = MpmRigidSource::Primitive(Arc::clone(&rigid));
        let rigid = locked(&rigid.inner)?;
        self.step_gpu_fixed_coupled(
            &rigid.world,
            source,
            substep_dt,
            steps,
            false,
            MpmCouplingMode::OneWay,
        )
    }

    /// Queue sphere rigid obstacle updates and MPM substeps on GPU.
    /// Rigid steps and submissions on the same queue preserve frame order.
    pub fn submit_gpu_fixed_with_sphere_world(
        &self,
        rigid: Arc<GpuSphereWorld>,
        substep_dt: f64,
        steps: u32,
    ) -> Result<(), TesseraError> {
        let source = MpmRigidSource::Sphere(Arc::clone(&rigid));
        let rigid = locked(&rigid.world)?;
        self.step_gpu_fixed_coupled(
            &rigid,
            source,
            substep_dt,
            steps,
            false,
            MpmCouplingMode::OneWay,
        )
    }

    /// Advance one MPM substep and apply its reaction to mixed-shape rigid GPU velocities.
    /// Integrate rigid positions separately with a rigid-world step.
    pub fn step_gpu_two_way_with_primitive_world(
        &self,
        rigid: Arc<GpuPrimitiveWorld>,
        substep_dt: f64,
    ) -> Result<(), TesseraError> {
        let source = MpmRigidSource::Primitive(Arc::clone(&rigid));
        let rigid = locked(&rigid.inner)?;
        self.step_gpu_fixed_coupled(
            &rigid.world,
            source,
            substep_dt,
            1,
            true,
            MpmCouplingMode::TwoWay,
        )
    }

    /// Advance one MPM substep and apply its reaction to sphere rigid GPU velocities.
    pub fn step_gpu_two_way_with_sphere_world(
        &self,
        rigid: Arc<GpuSphereWorld>,
        substep_dt: f64,
    ) -> Result<(), TesseraError> {
        let source = MpmRigidSource::Sphere(Arc::clone(&rigid));
        let rigid = locked(&rigid.world)?;
        self.step_gpu_fixed_coupled(&rigid, source, substep_dt, 1, true, MpmCouplingMode::TwoWay)
    }

    /// Queue one two-way MPM substep without synchronizing CPU particles.
    /// The rigid GPU velocity is updated on the same queue before this returns.
    pub fn submit_gpu_two_way_with_primitive_world(
        &self,
        rigid: Arc<GpuPrimitiveWorld>,
        substep_dt: f64,
    ) -> Result<(), TesseraError> {
        let source = MpmRigidSource::Primitive(Arc::clone(&rigid));
        let rigid = locked(&rigid.inner)?;
        self.step_gpu_fixed_coupled(
            &rigid.world,
            source,
            substep_dt,
            1,
            false,
            MpmCouplingMode::TwoWay,
        )
    }

    /// Queue one two-way MPM substep for a sphere rigid world.
    pub fn submit_gpu_two_way_with_sphere_world(
        &self,
        rigid: Arc<GpuSphereWorld>,
        substep_dt: f64,
    ) -> Result<(), TesseraError> {
        let source = MpmRigidSource::Sphere(Arc::clone(&rigid));
        let rigid = locked(&rigid.world)?;
        self.step_gpu_fixed_coupled(
            &rigid,
            source,
            substep_dt,
            1,
            false,
            MpmCouplingMode::TwoWay,
        )
    }

    /// Advance one MPM substep and integrate mixed-shape rigid GPU poses.
    /// Use this path instead of stepping the rigid world for the coupled body.
    pub fn step_gpu_owned_with_primitive_world(
        &self,
        rigid: Arc<GpuPrimitiveWorld>,
        substep_dt: f64,
    ) -> Result<(), TesseraError> {
        let source = MpmRigidSource::Primitive(Arc::clone(&rigid));
        let rigid = locked(&rigid.inner)?;
        self.step_gpu_fixed_coupled(
            &rigid.world,
            source,
            substep_dt,
            1,
            true,
            MpmCouplingMode::MpmOwned,
        )
    }

    /// Advance one MPM substep and integrate sphere rigid GPU poses.
    pub fn step_gpu_owned_with_sphere_world(
        &self,
        rigid: Arc<GpuSphereWorld>,
        substep_dt: f64,
    ) -> Result<(), TesseraError> {
        let source = MpmRigidSource::Sphere(Arc::clone(&rigid));
        let rigid = locked(&rigid.world)?;
        self.step_gpu_fixed_coupled(
            &rigid,
            source,
            substep_dt,
            1,
            true,
            MpmCouplingMode::MpmOwned,
        )
    }

    /// Queue one MPM-owned mixed-shape rigid step without particle readback.
    pub fn submit_gpu_owned_with_primitive_world(
        &self,
        rigid: Arc<GpuPrimitiveWorld>,
        substep_dt: f64,
    ) -> Result<(), TesseraError> {
        let source = MpmRigidSource::Primitive(Arc::clone(&rigid));
        let rigid = locked(&rigid.inner)?;
        self.step_gpu_fixed_coupled(
            &rigid.world,
            source,
            substep_dt,
            1,
            false,
            MpmCouplingMode::MpmOwned,
        )
    }

    /// Queue one MPM-owned sphere rigid step without particle readback.
    pub fn submit_gpu_owned_with_sphere_world(
        &self,
        rigid: Arc<GpuSphereWorld>,
        substep_dt: f64,
    ) -> Result<(), TesseraError> {
        let source = MpmRigidSource::Sphere(Arc::clone(&rigid));
        let rigid = locked(&rigid.world)?;
        self.step_gpu_fixed_coupled(
            &rigid,
            source,
            substep_dt,
            1,
            false,
            MpmCouplingMode::MpmOwned,
        )
    }

    /// Number of prescribed rigid obstacles.
    pub fn obstacle_count(&self) -> Result<u64, TesseraError> {
        u64::try_from(locked(&self.inner)?.world.obstacles.len()).map_err(failed)
    }

    /// Set a force that will be consumed by the next particle step.
    pub fn set_particle_force(&self, index: u64, force: Vec3) -> Result<(), TesseraError> {
        let mut inner = locked(&self.inner)?;
        let index = usize::try_from(index).map_err(failed)?;
        if index >= inner.world.particles.len() {
            return Err(failed("MPM particle index out of range"));
        }
        if !force.x.is_finite() || !force.y.is_finite() || !force.z.is_finite() {
            return Err(failed("MPM force must be finite"));
        }
        let force = force.nalgebra();
        if let Some((_, session)) = &mut inner.resident {
            session.set_particle_force(index, force).map_err(failed)?;
        }
        inner.world.particles[index].force = force;
        Ok(())
    }

    /// Assign a particle to an independent particle-grid transfer region.
    /// This rebuilds a resident GPU grid after pending work is synchronized.
    pub fn set_particle_transfer_color(&self, index: u64, color: u32) -> Result<(), TesseraError> {
        let mut inner = locked(&self.inner)?;
        inner.reject_pending()?;
        let index = usize::try_from(index).map_err(failed)?;
        let color = u16::try_from(color).map_err(|_| failed("MPM transfer color out of range"))?;
        let particle = inner
            .world
            .particles
            .get_mut(index)
            .ok_or_else(|| failed("MPM particle index out of range"))?;
        particle.transfer_color = color;
        inner.resident = None;
        inner.coupled = None;
        Ok(())
    }
}

impl MpmWorld {
    fn step_gpu_fixed_coupled(
        &self,
        rigid: &CoreGpuSphereWorld,
        source: MpmRigidSource,
        substep_dt: f64,
        steps: u32,
        synchronize: bool,
        mode: MpmCouplingMode,
    ) -> Result<(), TesseraError> {
        if !substep_dt.is_finite() || substep_dt <= 0.0 || steps == 0 || steps > 64 {
            return Err(failed("invalid fixed GPU substep input"));
        }
        let mut inner = locked(&self.inner)?;
        if inner.pending_steps() > 0
            && (synchronize
                || inner
                    .coupled
                    .as_ref()
                    .is_none_or(|(previous, _)| !previous.same(&source))
                || inner
                    .resident
                    .as_ref()
                    .is_none_or(|(dt, _)| *dt != substep_dt))
        {
            return Err(failed(
                "synchronize pending GPU substeps before changing the session",
            ));
        }
        if inner
            .gpu
            .as_ref()
            .is_none_or(|gpu| gpu.device != *rigid.device())
        {
            inner.resident = None;
            inner.coupled = None;
            inner.gpu = Some(MpmGpu::from_device(rigid.device(), rigid.queue()));
        }
        let same_source = inner
            .coupled
            .as_ref()
            .is_some_and(|(previous, _)| previous.same(&source));
        let same_dt = inner
            .resident
            .as_ref()
            .is_some_and(|(dt, _)| *dt == substep_dt);
        if !same_source || !same_dt {
            let previous_settings = same_source.then(|| {
                inner
                    .world
                    .obstacles
                    .iter()
                    .map(|obstacle| (obstacle.boundary, obstacle.cpic_group))
                    .collect::<Vec<_>>()
            });
            inner.coupled = None;
            let gpu = inner
                .gpu
                .as_ref()
                .ok_or_else(|| failed("GPU context unavailable"))?;
            let mut session = GpuMpmResidentSession::new(
                &gpu.transfers,
                &gpu.device,
                &gpu.queue,
                inner.world.clone(),
                substep_dt,
            )
            .map_err(failed)?
            .into_owned();
            let coupler =
                tessera_mpm::GpuRigidMpmCoupler::new(rigid, &mut session).map_err(failed)?;
            if let Some(settings) = previous_settings
                && settings.len() == session.world().obstacles.len()
            {
                let mut obstacles = session.world().obstacles.clone();
                for (obstacle, (boundary, cpic_group)) in obstacles.iter_mut().zip(settings) {
                    obstacle.boundary = boundary;
                    obstacle.cpic_group = cpic_group;
                }
                session.set_obstacles(obstacles).map_err(failed)?;
            }
            inner.world = session.world().clone();
            inner.resident = Some((substep_dt, session));
            inner.coupled = Some((source, coupler));
        }
        let result = {
            let MpmInner {
                resident, coupled, ..
            } = &mut *inner;
            let coupler = &coupled
                .as_ref()
                .ok_or_else(|| failed("GPU coupler unavailable"))?
                .1;
            let session = &mut resident
                .as_mut()
                .ok_or_else(|| failed("GPU session unavailable"))?
                .1;
            match (mode, synchronize) {
                (MpmCouplingMode::OneWay, true) => coupler.step(rigid, session, steps),
                (MpmCouplingMode::OneWay, false) => coupler.submit_steps(rigid, session, steps),
                (MpmCouplingMode::TwoWay, true) => coupler.step_two_way(rigid, session),
                (MpmCouplingMode::TwoWay, false) => coupler.submit_two_way_step(rigid, session),
                (MpmCouplingMode::MpmOwned, true) => coupler.step_mpm_owned(rigid, session),
                (MpmCouplingMode::MpmOwned, false) => coupler.submit_mpm_owned_step(rigid, session),
            }
        };
        if let Err(error) = result {
            if inner.pending_steps() == 0 {
                inner.resident = None;
                inner.coupled = None;
            }
            return Err(failed(error));
        }
        if synchronize {
            inner.world = inner
                .resident
                .as_ref()
                .ok_or_else(|| failed("GPU session unavailable"))?
                .1
                .world()
                .clone();
        }
        Ok(())
    }

    fn replace_obstacles(&self, obstacles: Vec<RigidObstacle>) -> Result<(), TesseraError> {
        let mut inner = locked(&self.inner)?;
        inner.reject_pending()?;
        if let Some((_, session)) = &mut inner.resident {
            session.set_obstacles(obstacles.clone()).map_err(failed)?;
        }
        inner.world.set_obstacles(obstacles).map_err(failed)?;
        inner.coupled = None;
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn python_sand_neo_hookean_preserves_model_parameters_and_plastic_state() {
        let material = MpmMaterial::SandNeoHookean {
            young_modulus: 4_000.0,
            poisson_ratio: 0.2,
            friction_angle: 35.0f64.to_radians(),
            cohesion: 0.02,
        };
        assert_eq!(
            MaterialModel::from(material.clone()),
            MaterialModel::sand_neo_hookean(4_000.0, 0.2, 35.0f64.to_radians(), 0.02)
        );
        let mut input = particle(0.5);
        input.material = material;
        let world = MpmWorld::new(
            vec![input.clone()],
            v(0.0, 0.0, 0.0),
            0.1,
            0.001,
            None,
            None,
        )
        .unwrap();
        assert_eq!(world.particles().unwrap()[0].hardening, 1.0);
        world.step(0.0001).unwrap();
        assert_eq!(world.particles().unwrap()[0].hardening, 1.0);
        input.material = MpmMaterial::SandNeoHookean {
            young_modulus: -1.0,
            poisson_ratio: 0.2,
            friction_angle: 0.5,
            cohesion: 0.0,
        };
        assert!(MpmWorld::new(vec![input], v(0.0, 0.0, 0.0), 0.1, 0.001, None, None).is_err());
    }

    #[test]
    fn closed_mesh_volume_samples_are_available_to_python() {
        let vertices = vec![
            v(0.0, 0.0, 0.0),
            v(1.0, 0.0, 0.0),
            v(0.0, 1.0, 0.0),
            v(0.0, 0.0, 1.0),
        ];
        let faces = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]
            .map(|[a, b, c]| GpuTriangle { a, b, c })
            .to_vec();
        let samples =
            sample_mpm_closed_mesh_volume(vertices.clone(), faces.clone(), 0.25, 64, 64).unwrap();
        assert!(
            samples
                .iter()
                .any(|point| { point.x == 0.125 && point.y == 0.125 && point.z == 0.125 })
        );
        assert!(
            samples
                .iter()
                .all(|point| point.x + point.y + point.z < 1.0)
        );
        assert!(
            sample_mpm_closed_mesh_volume(vertices, faces[..3].to_vec(), 0.25, 64, 64).is_err()
        );
    }

    #[test]
    fn closed_mesh_particles_form_a_removable_python_world_chunk() {
        let world = MpmWorld::new(vec![], v(0.0, 0.0, -9.81), 0.1, 0.001, None, None).unwrap();
        let vertices = vec![
            v(0.0, 0.0, 0.0),
            v(1.0, 0.0, 0.0),
            v(0.0, 1.0, 0.0),
            v(0.0, 0.0, 1.0),
        ];
        let faces = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]
            .map(|[a, b, c]| GpuTriangle { a, b, c })
            .to_vec();
        let input = |max_particles| MpmMeshEmissionInput {
            vertices: vertices.clone(),
            triangles: faces.clone(),
            spacing: 0.25,
            density: 1_000.0,
            material: MpmMaterial::Elastic {
                young_modulus: 1_000.0,
                poisson_ratio: 0.2,
            },
            velocity: v(0.0, 0.0, 0.0),
            damping: 0.0,
            max_candidates: 64,
            max_particles,
        };
        let chunk = world.add_closed_mesh_particles(input(64)).unwrap();
        let particles = world.particles().unwrap();
        assert!(!particles.is_empty());
        assert!(particles.iter().all(|particle| {
            particle.chunk_id == chunk && particle.rest_volume == 0.25_f64.powi(3)
        }));
        assert!(world.add_closed_mesh_particles(input(1)).is_err());
        assert_eq!(world.particles().unwrap().len(), particles.len());
        world.step(0.001).unwrap();
        assert_eq!(world.remove_chunk(chunk).unwrap(), particles.len() as u64);
        assert!(world.particles().unwrap().is_empty());
    }

    fn v(x: f64, y: f64, z: f64) -> Vec3 {
        Vec3 { x, y, z }
    }

    fn particle(x: f64) -> MpmParticleInput {
        MpmParticleInput {
            position: v(x, 0.5, 0.5),
            velocity: v(0.0, 0.0, 0.0),
            radius: 0.03,
            density: 1_000.0,
            material: MpmMaterial::Elastic {
                young_modulus: 1_000.0,
                poisson_ratio: 0.2,
            },
            enabled: true,
            fixed: false,
            damping: 0.0,
        }
    }

    #[test]
    fn python_mpm_cpic_groups_validate_shape_and_range() {
        let world = MpmWorld::new(
            vec![particle(0.48)],
            v(0.0, 0.0, 0.0),
            0.1,
            0.001,
            None,
            None,
        )
        .unwrap();
        let identity = Quaternion {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        };
        world
            .set_obstacles(vec![MpmObstacleInput {
                shape: MpmObstacleShape::TrianglePrism {
                    vertices: vec![v(0.0, -1.0, -1.0), v(0.0, 1.0, -1.0), v(0.0, 0.0, 1.0)],
                    half_thickness: 0.001,
                },
                center: v(0.5, 0.5, 0.5),
                orientation: identity,
                linear_velocity: v(0.0, 0.0, 0.0),
                angular_velocity: v(0.0, 0.0, 0.0),
                friction: 0.0,
            }])
            .unwrap();
        world.set_obstacle_cpic_groups(vec![Some(0)]).unwrap();
        assert_eq!(world.obstacle_cpic_groups().unwrap(), vec![Some(0)]);
        world.set_obstacle_cpic_groups(vec![Some(31)]).unwrap();
        assert!(world.set_obstacle_cpic_groups(vec![Some(32)]).is_err());
        assert!(world.set_obstacle_cpic_groups(vec![]).is_err());
        assert_eq!(world.obstacle_cpic_groups().unwrap(), vec![Some(31)]);
    }

    #[test]
    fn python_mpm_cpu_exposes_particles_chunks_and_obstacles() {
        let world = MpmWorld::new(
            vec![particle(0.5)],
            v(0.0, 0.0, 0.0),
            0.1,
            0.001,
            Some(v(0.0, 0.0, 0.0)),
            Some(v(1.0, 1.0, 1.0)),
        )
        .unwrap();
        let identity = Quaternion {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        };
        let obstacle = MpmObstacleInput {
            shape: MpmObstacleShape::Sphere { radius: 0.1 },
            center: v(0.5, 0.5, 0.5),
            orientation: identity,
            linear_velocity: v(1.0, 0.0, 0.0),
            angular_velocity: v(0.0, 0.0, 0.0),
            friction: 0.5,
        };
        world.set_obstacles(vec![obstacle.clone()]).unwrap();
        world.step(0.0001).unwrap();
        assert!(world.particles().unwrap()[0].velocity.x > 0.0);
        assert_eq!(world.obstacle_count().unwrap(), 1);
        let chunk = world.add_particles(vec![particle(0.7)]).unwrap();
        assert_eq!(world.particles().unwrap()[1].chunk_id, chunk);
        assert_eq!(world.remove_chunk(chunk).unwrap(), 1);
        assert_eq!(world.particles().unwrap().len(), 1);
        assert!(world.remove_chunk(chunk).is_err());
        assert_eq!(world.particles().unwrap()[0].deformation.len(), 9);
        let invalid = MpmObstacleInput {
            orientation: Quaternion { w: 0.5, ..identity },
            ..obstacle
        };
        assert!(world.set_obstacles(vec![invalid]).is_err());
        assert_eq!(world.obstacle_count().unwrap(), 1);
    }

    #[test]
    fn python_mpm_material_and_obstacle_variants_convert() {
        let materials = [
            MpmMaterial::Elastic {
                young_modulus: 1_000.0,
                poisson_ratio: 0.2,
            },
            MpmMaterial::NeoHookean {
                young_modulus: 1_000.0,
                poisson_ratio: 0.2,
            },
            MpmMaterial::Sand {
                young_modulus: 1_000.0,
                poisson_ratio: 0.2,
                friction_angle: 0.5,
                cohesion: 0.0,
            },
            MpmMaterial::Fluid {
                bulk_modulus: 2_000.0,
                gamma: 7.0,
                viscosity: 0.1,
                tensile_stiffness: 0.25,
            },
            MpmMaterial::Snow {
                young_modulus: 1_000.0,
                poisson_ratio: 0.2,
                critical_compression: 0.025,
                critical_stretch: 0.0075,
                hardening: 10.0,
            },
        ];
        for material in materials {
            let mut input = particle(0.5);
            input.material = material;
            let world =
                MpmWorld::new(vec![input], v(0.0, 0.0, 0.0), 0.1, 0.001, None, None).unwrap();
            world.step(0.0001).unwrap();
            assert_eq!(world.particles().unwrap().len(), 1);
        }

        let shapes = [
            MpmObstacleShape::Sphere { radius: 0.1 },
            MpmObstacleShape::Box {
                half_extents: v(0.1, 0.1, 0.1),
            },
            MpmObstacleShape::Capsule {
                half_height: 0.1,
                radius: 0.05,
            },
            MpmObstacleShape::Cylinder {
                half_height: 0.1,
                radius: 0.05,
            },
            MpmObstacleShape::Cone {
                half_height: 0.1,
                radius: 0.05,
            },
            MpmObstacleShape::Ground {
                half_extent_x: 1.0,
                half_extent_y: 1.0,
            },
            MpmObstacleShape::TrianglePrism {
                vertices: vec![v(0.0, 0.0, 0.0), v(0.1, 0.0, 0.0), v(0.0, 0.1, 0.0)],
                half_thickness: 0.001,
            },
            MpmObstacleShape::Convex {
                vertices: vec![
                    v(-0.1, -0.1, -0.1),
                    v(0.1, -0.1, -0.1),
                    v(-0.1, 0.1, -0.1),
                    v(0.1, 0.1, -0.1),
                    v(-0.1, -0.1, 0.1),
                    v(0.1, -0.1, 0.1),
                    v(-0.1, 0.1, 0.1),
                    v(0.1, 0.1, 0.1),
                ],
                face_normals: vec![
                    v(1.0, 0.0, 0.0),
                    v(-1.0, 0.0, 0.0),
                    v(0.0, 1.0, 0.0),
                    v(0.0, -1.0, 0.0),
                    v(0.0, 0.0, 1.0),
                    v(0.0, 0.0, -1.0),
                ],
            },
        ];
        for shape in shapes {
            let input = MpmObstacleInput {
                shape,
                center: v(0.5, 0.5, 0.5),
                orientation: Quaternion {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                    w: 1.0,
                },
                linear_velocity: v(0.0, 0.0, 0.0),
                angular_velocity: v(0.0, 0.0, 0.0),
                friction: 0.5,
            };
            assert!(obstacle(input).unwrap().is_valid());
        }
    }

    #[test]
    fn python_mpm_gpu_tracks_cpu_when_adapter_is_available() {
        if GpuContactDevice::new().is_err() {
            eprintln!("No hardware WebGPU adapter available for Python MPM test");
            return;
        }
        let make_world = || {
            MpmWorld::new(
                vec![particle(0.5)],
                v(0.0, 0.0, -1.0),
                0.1,
                0.001,
                Some(v(0.0, 0.0, 0.0)),
                Some(v(1.0, 1.0, 1.0)),
            )
            .unwrap()
        };
        let cpu = make_world();
        let gpu = make_world();
        cpu.step(0.0001).unwrap();
        gpu.step_gpu_fixed(0.0001, 1).unwrap();
        assert!(locked(&gpu.inner).unwrap().resident.is_some());
        for step in 0..20 {
            if step == 4 {
                cpu.set_particle_force(0, v(0.1, 0.0, 0.0)).unwrap();
                gpu.set_particle_force(0, v(0.1, 0.0, 0.0)).unwrap();
                assert!(locked(&gpu.inner).unwrap().resident.is_some());
            }
            if step == 8 {
                cpu.step(0.0001).unwrap();
                gpu.step(0.0001).unwrap();
                assert!(locked(&gpu.inner).unwrap().resident.is_none());
            }
            if step == 12 {
                let cpu_chunk = cpu.add_particles(vec![particle(0.7)]).unwrap();
                let gpu_chunk = gpu.add_particles(vec![particle(0.7)]).unwrap();
                assert!(locked(&gpu.inner).unwrap().resident.is_some());
                assert_eq!(cpu.remove_chunk(cpu_chunk).unwrap(), 1);
                assert_eq!(gpu.remove_chunk(gpu_chunk).unwrap(), 1);
                assert!(locked(&gpu.inner).unwrap().resident.is_some());
            }
            let dt = if step >= 16 { 0.00005 } else { 0.0001 };
            cpu.step(dt).unwrap();
            gpu.step_gpu_fixed(dt, 1).unwrap();
            assert_eq!(locked(&gpu.inner).unwrap().resident.as_ref().unwrap().0, dt);
            assert_eq!(gpu.substeps().unwrap(), cpu.substeps().unwrap());
        }
        let previous = gpu.particles().unwrap()[0].position.z;
        assert!(gpu.step_gpu_fixed(0.0001, 0).is_err());
        assert_eq!(gpu.particles().unwrap()[0].position.z, previous);
        let expected = cpu.particles().unwrap();
        let actual = gpu.particles().unwrap();
        assert!((expected[0].position.z - actual[0].position.z).abs() < 1e-5);
        assert!((expected[0].velocity.z - actual[0].velocity.z).abs() < 1e-4);
        cpu.step(0.0001).unwrap();
        gpu.step_gpu(0.0001).unwrap();
        let expected = cpu.particles().unwrap();
        let actual = gpu.particles().unwrap();
        assert!((expected[0].position.z - actual[0].position.z).abs() < 1e-5);
        assert!((expected[0].velocity.z - actual[0].velocity.z).abs() < 1e-4);
        assert_eq!(gpu.substeps().unwrap(), cpu.substeps().unwrap());
        assert!((expected[0].velocity.x - actual[0].velocity.x).abs() < 1e-4);
        assert!(locked(&gpu.inner).unwrap().resident.is_none());
    }

    #[test]
    fn closed_mesh_chunk_edits_preserve_uncoupled_gpu_session() {
        if GpuContactDevice::new().is_err() {
            eprintln!("No WebGPU adapter available for closed-mesh MPM session test");
            return;
        }
        let make_world = || {
            MpmWorld::new(
                vec![particle(0.25)],
                v(0.0, 0.0, -1.0),
                0.1,
                0.001,
                Some(v(0.0, 0.0, 0.0)),
                Some(v(2.0, 2.0, 2.0)),
            )
            .unwrap()
        };
        let cpu = make_world();
        let gpu = make_world();
        cpu.step(0.0001).unwrap();
        gpu.step_gpu_fixed(0.0001, 1).unwrap();
        let input = || MpmMeshEmissionInput {
            vertices: vec![
                v(0.5, 0.5, 0.5),
                v(1.5, 0.5, 0.5),
                v(0.5, 1.5, 0.5),
                v(0.5, 0.5, 1.5),
            ],
            triangles: vec![
                GpuTriangle { a: 0, b: 2, c: 1 },
                GpuTriangle { a: 0, b: 1, c: 3 },
                GpuTriangle { a: 0, b: 3, c: 2 },
                GpuTriangle { a: 1, b: 2, c: 3 },
            ],
            spacing: 0.25,
            density: 1_000.0,
            material: MpmMaterial::Elastic {
                young_modulus: 1_000.0,
                poisson_ratio: 0.2,
            },
            velocity: v(0.0, 0.0, 0.0),
            damping: 0.0,
            max_candidates: 64,
            max_particles: 64,
        };
        let cpu_chunk = cpu.add_closed_mesh_particles(input()).unwrap();
        let gpu_chunk = gpu.add_closed_mesh_particles(input()).unwrap();
        assert!(locked(&gpu.inner).unwrap().resident.is_some());
        assert_eq!(
            cpu.particles().unwrap().len(),
            gpu.particles().unwrap().len()
        );
        let mut over_limit = input();
        over_limit.max_particles = 1;
        assert!(gpu.add_closed_mesh_particles(over_limit).is_err());
        assert!(locked(&gpu.inner).unwrap().resident.is_some());
        assert_eq!(
            cpu.particles().unwrap().len(),
            gpu.particles().unwrap().len()
        );
        cpu.step(0.0001).unwrap();
        gpu.step_gpu_fixed(0.0001, 1).unwrap();
        for (expected, actual) in cpu
            .particles()
            .unwrap()
            .iter()
            .zip(gpu.particles().unwrap())
        {
            assert!((expected.position.x - actual.position.x).abs() < 1e-4);
            assert!((expected.position.y - actual.position.y).abs() < 1e-4);
            assert!((expected.position.z - actual.position.z).abs() < 1e-4);
            assert!((expected.velocity.z - actual.velocity.z).abs() < 1e-4);
        }
        assert_eq!(
            cpu.remove_chunk(cpu_chunk).unwrap(),
            gpu.remove_chunk(gpu_chunk).unwrap()
        );
        assert!(locked(&gpu.inner).unwrap().resident.is_some());
        cpu.step(0.0001).unwrap();
        gpu.step_gpu_fixed(0.0001, 1).unwrap();
        assert_eq!(cpu.particles().unwrap().len(), 1);
        assert_eq!(gpu.particles().unwrap().len(), 1);
    }
    #[test]
    fn python_mpm_unbounded_fixed_steps_track_cpu() {
        if GpuContactDevice::new().is_err() {
            return;
        }
        let make_world = || {
            MpmWorld::new(
                vec![particle(-10.5), particle(10.5)],
                v(0.0, 0.0, -1.0),
                0.1,
                0.001,
                None,
                None,
            )
            .unwrap()
        };
        let cpu = make_world();
        let gpu = make_world();
        for _ in 0..4 {
            for _ in 0..8 {
                cpu.step(0.0001).unwrap();
            }
            gpu.step_gpu_fixed(0.0001, 8).unwrap();
            assert!(locked(&gpu.inner).unwrap().resident.is_some());
            assert_eq!(cpu.substeps().unwrap(), gpu.substeps().unwrap());
            for (actual, expected) in gpu
                .particles()
                .unwrap()
                .iter()
                .zip(cpu.particles().unwrap())
            {
                assert!((actual.position.x - expected.position.x).abs() < 1e-4);
                assert!((actual.position.z - expected.position.z).abs() < 1e-4);
                assert!((actual.velocity.z - expected.velocity.z).abs() < 2e-3);
            }
        }
    }
}
