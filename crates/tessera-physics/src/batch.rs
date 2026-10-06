//! Multiple independent articulated environments with snapshot-based reset.
//!
//! The public layout follows the batched-environment model used by GPU physics
//! engines. Contact detection prepares each environment, while compatible
//! contact problems share one packed GPU impulse dispatch per substep.

#[cfg(feature = "gpu-contact")]
use std::collections::BTreeMap;
#[cfg(feature = "gpu-contact")]
use std::sync::Arc;

use crate::articulated_world::{ArticulatedWorld, ArticulatedWorldError, ArticulatedWorldSnapshot};
#[cfg(feature = "gpu-contact")]
use crate::contact_reference::ContactSolution;
#[cfg(feature = "gpu-contact")]
use crate::gpu_articulated_mass::GpuArticulatedMassError;
#[cfg(feature = "gpu-contact")]
use crate::gpu_articulated_mass_assembly::GpuArticulatedMassAssemblyBatch;
#[cfg(feature = "gpu-contact")]
use crate::gpu_contact_pipeline::GpuContactDevice;
#[cfg(feature = "gpu-contact")]
use crate::gpu_contact_solver::{
    GpuContactSolveError, GpuContactSolveRequest, GpuDeviceMassContactRequest,
};

/// Stable index of one environment in an [`ArticulatedBatch`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EnvironmentId(usize);

impl EnvironmentId {
    /// Zero-based environment index.
    pub fn index(self) -> usize {
        self.0
    }
}

/// Failure while addressing or advancing a batched environment.
#[derive(Debug, thiserror::Error)]
pub enum BatchError {
    /// Torque input count did not match the environment count.
    #[error("expected torques for {expected} environments, got {actual}")]
    EnvironmentCount {
        /// Number of environments currently in the batch.
        expected: usize,
        /// Number of torque slices supplied by the caller.
        actual: usize,
    },
    /// An environment handle is outside this batch.
    #[error("environment {0} does not exist")]
    UnknownEnvironment(usize),
    /// One environment failed to advance or restore.
    #[error("environment {environment} failed: {source}")]
    Environment {
        /// Zero-based environment that failed.
        environment: usize,
        /// Underlying world error.
        #[source]
        source: ArticulatedWorldError,
    },
    /// A packed GPU solve failed for one or more environments.
    #[cfg(feature = "gpu-contact")]
    #[error("packed GPU contact solve failed for environments {environments:?}: {source}")]
    GpuSolve {
        /// Environment indices in the failed dispatch.
        environments: Vec<usize>,
        /// Underlying solver error.
        #[source]
        source: GpuContactSolveError,
    },
    /// A batched reduced-coordinate GPU mass solve failed.
    #[cfg(feature = "gpu-contact")]
    #[error("batched GPU articulated mass solve failed: {0}")]
    GpuMass(#[from] GpuArticulatedMassError),
}

/// Independent articulated simulations sharing the same stepping interface.
#[derive(Debug, Default)]
pub struct ArticulatedBatch {
    worlds: Vec<ArticulatedWorld>,
    reset_templates: Vec<ArticulatedWorldSnapshot>,
}

/// Reusable GPU mass solver for a sequence of articulated batch states.
///
/// It keeps the packed GPU buffers and pipelines when environment dimensions
/// and mass-bearing link counts stay the same, and rebuilds them otherwise.
#[cfg(feature = "gpu-contact")]
#[derive(Debug)]
pub struct ArticulatedMassSession<'a> {
    context: &'a GpuContactDevice,
    solver: Option<GpuArticulatedMassAssemblyBatch>,
}

/// Reusable packed GPU mass buffers for repeated articulated contact steps.
///
/// The session is tied to one GPU device. A change in active environment
/// dimensions or mass-bearing link counts rebuilds its packed buffers.
#[cfg(feature = "gpu-contact")]
#[derive(Debug)]
pub struct ArticulatedDeviceMassStepSession<'a> {
    context: &'a GpuContactDevice,
    cache: ArticulatedDeviceMassStepCache,
}

/// Owned GPU mass buffers that can be retained beside a batched world.
///
/// A cache built on another GPU device is discarded before stepping. The
/// packed buffers are also rebuilt when the active mass layout changes.
#[cfg(feature = "gpu-contact")]
#[derive(Debug, Default)]
pub struct ArticulatedDeviceMassStepCache {
    device_identity: Option<Arc<()>>,
    mass_batch: Option<GpuArticulatedMassAssemblyBatch>,
}

#[cfg(feature = "gpu-contact")]
impl ArticulatedDeviceMassStepCache {
    /// Create an empty cache with no device or environment layout.
    pub fn new() -> Self {
        Self::default()
    }

    /// Advance a batch and keep compatible GPU buffers for the next call.
    pub fn step(
        &mut self,
        batch: &mut ArticulatedBatch,
        dt: f64,
        torques: &[&[f64]],
        context: &GpuContactDevice,
    ) -> Result<(), BatchError> {
        if self
            .device_identity
            .as_ref()
            .is_some_and(|identity| !Arc::ptr_eq(identity, context.identity()))
        {
            self.mass_batch = None;
        }
        self.device_identity = Some(context.identity().clone());
        batch.step_gpu_device_mass_with_cache(dt, torques, context, &mut self.mass_batch)
    }
}

#[cfg(feature = "gpu-contact")]
impl<'a> ArticulatedDeviceMassStepSession<'a> {
    /// Bind reusable contact mass buffers to one GPU device.
    pub fn new(context: &'a GpuContactDevice) -> Self {
        Self {
            context,
            cache: ArticulatedDeviceMassStepCache::new(),
        }
    }

    /// Advance all environments, retaining compatible mass buffers for reuse.
    pub fn step(
        &mut self,
        batch: &mut ArticulatedBatch,
        dt: f64,
        torques: &[&[f64]],
    ) -> Result<(), BatchError> {
        self.cache.step(batch, dt, torques, self.context)
    }
}

/// Accelerations and inverse mass matrices in environment order.
#[cfg(feature = "gpu-contact")]
#[derive(Debug)]
pub struct ArticulatedMassSolution {
    /// Generalized acceleration for each environment.
    pub accelerations: Vec<nalgebra::DVector<f64>>,
    /// Generalized inverse mass matrix for each environment.
    pub inverse_masses: Vec<nalgebra::DMatrix<f64>>,
}

#[cfg(feature = "gpu-contact")]
impl<'a> ArticulatedMassSession<'a> {
    /// Bind a reusable session to one GPU device.
    pub fn new(context: &'a GpuContactDevice) -> Self {
        Self {
            context,
            solver: None,
        }
    }

    /// Solve current generalized accelerations without advancing state.
    pub fn solve(
        &mut self,
        batch: &ArticulatedBatch,
        torques: &[&[f64]],
    ) -> Result<Vec<nalgebra::DVector<f64>>, BatchError> {
        Ok(self.solve_internal(batch, torques, false)?.accelerations)
    }

    /// Solve accelerations and inverse mass matrices for contact coupling.
    pub fn solve_with_inverse(
        &mut self,
        batch: &ArticulatedBatch,
        torques: &[&[f64]],
    ) -> Result<ArticulatedMassSolution, BatchError> {
        self.solve_internal(batch, torques, true)
    }

    fn solve_internal(
        &mut self,
        batch: &ArticulatedBatch,
        torques: &[&[f64]],
        with_inverse: bool,
    ) -> Result<ArticulatedMassSolution, BatchError> {
        if torques.len() != batch.worlds.len() {
            return Err(BatchError::EnvironmentCount {
                expected: batch.worlds.len(),
                actual: torques.len(),
            });
        }
        let mut systems = Vec::new();
        let mut indices = Vec::new();
        let mut results = batch
            .worlds
            .iter()
            .map(|world| {
                nalgebra::DVector::zeros(
                    world.articulation.dof() + if world.floating { 6 } else { 0 },
                )
            })
            .collect::<Vec<_>>();
        let mut inverses = if with_inverse {
            batch
                .worlds
                .iter()
                .map(|world| {
                    let n = world.articulation.dof() + if world.floating { 6 } else { 0 };
                    nalgebra::DMatrix::zeros(n, n)
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        for (index, (world, torques)) in batch.worlds.iter().zip(torques).enumerate() {
            let system = world
                .generalized_mass_assembly_system(torques)
                .map_err(|source| BatchError::Environment {
                    environment: index,
                    source,
                })?;
            if !system.force.is_empty() {
                indices.push(index);
                systems.push(system);
            }
        }
        if !systems.is_empty() {
            if let Some(solver) = self
                .solver
                .as_ref()
                .filter(|solver| solver.has_layout(&systems))
            {
                solver.update(&systems)?;
            } else {
                self.solver = Some(GpuArticulatedMassAssemblyBatch::new(
                    self.context.device(),
                    self.context.queue(),
                    &systems,
                )?);
            }
            let solver = self
                .solver
                .as_ref()
                .ok_or(GpuArticulatedMassError::InvalidInput)?;
            if with_inverse {
                solver.submit_with_inverse();
            } else {
                solver.submit();
            }
            let solved = solver.readback()?;
            for (index, value) in indices.iter().copied().zip(solved) {
                results[index] = value;
            }
            if with_inverse {
                for (index, value) in indices.into_iter().zip(solver.readback_inverse()?) {
                    inverses[index] = value;
                }
            }
        }
        Ok(ArticulatedMassSolution {
            accelerations: results,
            inverse_masses: inverses,
        })
    }
}

impl ArticulatedBatch {
    /// Create an empty batch.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of environments in insertion order.
    pub fn len(&self) -> usize {
        self.worlds.len()
    }

    /// Whether this batch contains no environments.
    pub fn is_empty(&self) -> bool {
        self.worlds.is_empty()
    }

    /// Borrow all environments in stable insertion order.
    pub fn environments(&self) -> &[ArticulatedWorld] {
        &self.worlds
    }

    /// Mutably borrow all environments in stable insertion order.
    pub fn environments_mut(&mut self) -> &mut [ArticulatedWorld] {
        &mut self.worlds
    }

    /// Insert an environment and publish its current state as the reset template.
    pub fn add_environment(&mut self, world: ArticulatedWorld) -> EnvironmentId {
        let id = EnvironmentId(self.worlds.len());
        self.reset_templates.push(world.snapshot());
        self.worlds.push(world);
        id
    }

    /// Borrow one environment.
    pub fn environment(&self, id: EnvironmentId) -> Option<&ArticulatedWorld> {
        self.worlds.get(id.0)
    }

    /// Mutably borrow one environment.
    pub fn environment_mut(&mut self, id: EnvironmentId) -> Option<&mut ArticulatedWorld> {
        self.worlds.get_mut(id.0)
    }

    /// Solve current generalized accelerations for every environment on GPU.
    ///
    /// Link Jacobians and forces are prepared from the current articulated
    /// states. GPU passes assemble mass matrices and solve packed environments.
    /// This readback API does not advance states or resolve contacts.
    #[cfg(feature = "gpu-contact")]
    pub fn generalized_accelerations_gpu(
        &self,
        torques: &[&[f64]],
        context: &GpuContactDevice,
    ) -> Result<Vec<nalgebra::DVector<f64>>, BatchError> {
        ArticulatedMassSession::new(context).solve(self, torques)
    }

    /// Replace the reset template of one environment with its current state.
    pub fn publish_reset_template(&mut self, id: EnvironmentId) -> Result<(), BatchError> {
        let world = self
            .worlds
            .get(id.0)
            .ok_or(BatchError::UnknownEnvironment(id.0))?;
        self.reset_templates[id.0] = world.snapshot();
        Ok(())
    }

    /// Restore one environment from its published template.
    pub fn reset_environment(&mut self, id: EnvironmentId) -> Result<(), BatchError> {
        let snapshot = self
            .reset_templates
            .get(id.0)
            .ok_or(BatchError::UnknownEnvironment(id.0))?
            .clone();
        self.worlds[id.0]
            .restore_snapshot(&snapshot)
            .map_err(|source| BatchError::Environment {
                environment: id.0,
                source,
            })
    }

    /// Advance every environment with CPU contact detection.
    pub fn step(&mut self, dt: f64, torques: &[&[f64]]) -> Result<(), BatchError> {
        if torques.len() != self.worlds.len() {
            return Err(BatchError::EnvironmentCount {
                expected: self.worlds.len(),
                actual: torques.len(),
            });
        }
        for (environment, (world, torques)) in self.worlds.iter_mut().zip(torques).enumerate() {
            world
                .step(dt, torques)
                .map_err(|source| BatchError::Environment {
                    environment,
                    source,
                })?;
        }
        Ok(())
    }

    /// Advance environments with packed GPU contact impulse dispatches.
    ///
    /// Environments with equal substep counts and solver iterations share one
    /// dispatch for each substep. Collision detection still runs per world.
    #[cfg(feature = "gpu-contact")]
    pub fn step_gpu(
        &mut self,
        dt: f64,
        torques: &[&[f64]],
        context: &GpuContactDevice,
    ) -> Result<(), BatchError> {
        if torques.len() != self.worlds.len() {
            return Err(BatchError::EnvironmentCount {
                expected: self.worlds.len(),
                actual: torques.len(),
            });
        }
        let counts = self
            .worlds
            .iter()
            .zip(torques)
            .enumerate()
            .map(|(environment, (world, torques))| {
                world
                    .validated_substeps(dt, torques)
                    .map_err(|source| BatchError::Environment {
                        environment,
                        source,
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let max_count = counts.iter().copied().max().unwrap_or(0);
        for substep in 0..max_count {
            let mut groups = BTreeMap::new();
            for (environment, ((world, torques), &count)) in
                self.worlds.iter_mut().zip(torques).zip(&counts).enumerate()
            {
                if substep >= count {
                    continue;
                }
                let pending = world
                    .prepare_contact_step(dt / count as f64, torques, Some(context))
                    .map_err(|source| BatchError::Environment {
                        environment,
                        source,
                    })?;
                groups
                    .entry((count, pending.solve_params.iterations))
                    .or_insert_with(Vec::new)
                    .push((environment, pending));
            }
            for group in groups.into_values() {
                let requests = group
                    .iter()
                    .map(|(_, pending)| GpuContactSolveRequest {
                        problem: &pending.problem,
                        warm_start: Some(&pending.warm_start),
                    })
                    .collect::<Vec<_>>();
                let solutions = context
                    .solver()
                    .solve_packed(
                        context.device(),
                        context.queue(),
                        &requests,
                        group[0].1.solve_params,
                    )
                    .map_err(|source| BatchError::GpuSolve {
                        environments: group.iter().map(|(index, _)| *index).collect(),
                        source,
                    })?;
                for ((environment, pending), solved) in group.into_iter().zip(solutions) {
                    self.worlds[environment]
                        .finish_contact_step(pending, solved)
                        .map_err(|source| BatchError::Environment {
                            environment,
                            source,
                        })?;
                }
            }
        }
        Ok(())
    }

    /// Advance independent environments with packed GPU mass inversion.
    ///
    /// Each substep assembles all active mass matrices in one GPU batch and
    /// encodes their contact solves in one command submission. CPU contact
    /// preparation still reads each inverse matrix once per substep.
    #[cfg(feature = "gpu-contact")]
    pub fn step_gpu_device_mass(
        &mut self,
        dt: f64,
        torques: &[&[f64]],
        context: &GpuContactDevice,
    ) -> Result<(), BatchError> {
        ArticulatedDeviceMassStepCache::new().step(self, dt, torques, context)
    }

    #[cfg(feature = "gpu-contact")]
    fn step_gpu_device_mass_with_cache(
        &mut self,
        dt: f64,
        torques: &[&[f64]],
        context: &GpuContactDevice,
        mass_batch: &mut Option<GpuArticulatedMassAssemblyBatch>,
    ) -> Result<(), BatchError> {
        if torques.len() != self.worlds.len() {
            return Err(BatchError::EnvironmentCount {
                expected: self.worlds.len(),
                actual: torques.len(),
            });
        }
        let counts = self
            .worlds
            .iter()
            .zip(torques)
            .enumerate()
            .map(|(environment, (world, torques))| {
                world
                    .validated_substeps(dt, torques)
                    .map_err(|source| BatchError::Environment {
                        environment,
                        source,
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        for substep in 0..counts.iter().copied().max().unwrap_or(0) {
            let mut systems = Vec::new();
            let mut indices = Vec::new();
            for environment in 0..self.worlds.len() {
                let count = counts[environment];
                if substep >= count {
                    continue;
                }
                let system = self.worlds[environment]
                    .generalized_contact_mass_assembly_system(
                        dt / count as f64,
                        torques[environment],
                    )
                    .map_err(|source| BatchError::Environment {
                        environment,
                        source,
                    })?;
                if system.force.is_empty() {
                    self.worlds[environment]
                        .step_gpu(dt / count as f64, torques[environment], context)
                        .map_err(|source| BatchError::Environment {
                            environment,
                            source,
                        })?;
                } else {
                    indices.push(environment);
                    systems.push(system);
                }
            }
            if systems.is_empty() {
                continue;
            }
            if let Some(batch) = mass_batch
                .as_ref()
                .filter(|batch| batch.has_layout(&systems))
            {
                batch.update(&systems)?;
            } else {
                *mass_batch = Some(GpuArticulatedMassAssemblyBatch::new(
                    context.device(),
                    context.queue(),
                    &systems,
                )?);
            }
            let mass = mass_batch
                .as_ref()
                .ok_or(GpuArticulatedMassError::InvalidInput)?;
            mass.submit_with_inverse();
            let inverses = mass.readback_inverse()?;
            let inverse_buffer = mass
                .inverse_buffer()
                .ok_or(GpuArticulatedMassError::InvalidInput)?;
            let mut encoder = context.device().create_command_encoder(&Default::default());
            let mut prepared = Vec::with_capacity(indices.len());
            for (mass_index, ((&environment, inverse), inverse_range)) in indices
                .iter()
                .zip(&inverses)
                .zip(mass.inverse_ranges())
                .enumerate()
            {
                let pending = self.worlds[environment]
                    .prepare_device_mass_contact_step(
                        dt / counts[environment] as f64,
                        torques[environment],
                        inverse,
                        context,
                    )
                    .map_err(|source| BatchError::Environment {
                        environment,
                        source,
                    })?;
                let coordinates = self.worlds[environment].contact_mass_coordinate_map();
                let expanded = if coordinates.iter().any(Option::is_none) {
                    Some(mass.encode_expanded_inverse(&mut encoder, mass_index, &coordinates)?)
                } else {
                    None
                };
                let output = context
                    .solver()
                    .encode_device_inverse_mass(
                        context.device(),
                        &mut encoder,
                        GpuDeviceMassContactRequest {
                            inverse_mass: expanded.as_ref().unwrap_or(inverse_buffer),
                            inverse_range: if expanded.is_some() {
                                0..coordinates.len() * coordinates.len() + 1
                            } else {
                                inverse_range.clone()
                            },
                            velocity: &pending.problem.velocity,
                            contacts: &pending.problem.contacts,
                            params: pending.solve_params,
                            warm_start: Some(&pending.warm_start),
                        },
                    )
                    .map_err(|source| BatchError::GpuSolve {
                        environments: vec![environment],
                        source,
                    })?;
                prepared.push((environment, pending, output));
            }
            let readbacks = prepared
                .iter()
                .filter_map(|(_, _, output)| {
                    output
                        .as_ref()
                        .map(|output| output.encode_readback(context.device(), &mut encoder))
                })
                .collect::<Vec<_>>();
            let solved = if readbacks.is_empty() {
                Vec::new()
            } else {
                let _ = context.queue().submit(Some(encoder.finish()));
                crate::gpu_contact_solver::GpuContactReadback::finish_many(
                    readbacks,
                    context.device(),
                )
                .map_err(|source| BatchError::GpuSolve {
                    environments: indices.clone(),
                    source,
                })?
            };
            let mut solved = solved.into_iter();
            for (environment, pending, output) in prepared {
                let solution = if output.is_some() {
                    solved.next().ok_or(GpuArticulatedMassError::InvalidInput)?
                } else {
                    ContactSolution {
                        velocity: pending.problem.velocity.clone(),
                        impulses: Vec::new(),
                    }
                };
                self.worlds[environment]
                    .finish_contact_step(pending, solution)
                    .map_err(|source| BatchError::Environment {
                        environment,
                        source,
                    })?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "gpu-contact")]
    use nalgebra::DMatrix;
    use nalgebra::{Isometry3, Matrix3, Vector3};

    use super::*;
    #[cfg(feature = "gpu-contact")]
    use crate::articulated_world::JointMotor;
    use crate::articulated_world::{ArticulatedWorldParams, LinkSphere, SceneBody, SceneCollider};
    use crate::articulation::{Articulation, LinkSpec};
    #[cfg(feature = "gpu-contact")]
    use crate::articulation::{JointKind, JointSpec};

    fn falling_world(x: f64) -> ArticulatedWorld {
        falling_world_with(x, 0.01, ArticulatedWorldParams::default().solver_iterations)
    }

    fn falling_world_with(x: f64, max_substep: f64, solver_iterations: usize) -> ArticulatedWorld {
        let articulation = Articulation::new(
            vec![LinkSpec {
                mass: 1.0,
                center_of_mass: Vector3::zeros(),
                inertia: Matrix3::identity(),
            }],
            vec![],
            0,
        )
        .unwrap();
        ArticulatedWorld::new_floating(
            articulation,
            Isometry3::translation(x, 0.0, 2.0),
            vec![LinkSphere {
                link: 0,
                center: Vector3::zeros(),
                radius: 0.25,
            }],
            ArticulatedWorldParams {
                max_substep,
                solver_iterations,
                ..Default::default()
            },
        )
        .unwrap()
    }

    #[cfg(feature = "gpu-contact")]
    fn motor_world(x: f64, target: f64) -> ArticulatedWorld {
        let articulation = Articulation::new(
            vec![
                LinkSpec {
                    mass: 1.0,
                    center_of_mass: Vector3::zeros(),
                    inertia: Matrix3::identity() * 0.1,
                },
                LinkSpec {
                    mass: 1.0,
                    center_of_mass: Vector3::new(1.0, 0.0, 0.0),
                    inertia: Matrix3::identity() * 0.1,
                },
            ],
            vec![JointSpec {
                parent: 0,
                child: 1,
                origin: Isometry3::identity(),
                kind: JointKind::Revolute,
                axis: Vector3::y(),
                limits: None,
            }],
            0,
        )
        .unwrap();
        let mut world = ArticulatedWorld::new(
            articulation,
            Isometry3::translation(x, 0.0, 0.2),
            vec![LinkSphere {
                link: 1,
                center: Vector3::new(1.0, 0.0, 0.0),
                radius: 0.2,
            }],
            ArticulatedWorldParams {
                max_substep: 0.01,
                ..Default::default()
            },
        )
        .unwrap();
        world
            .set_joint_motor(
                0,
                Some(JointMotor {
                    position_target: Some(target),
                    velocity_target: 0.0,
                    stiffness: 3.0,
                    damping: 0.2,
                    max_force: 1.0,
                }),
            )
            .unwrap();
        world
    }

    #[cfg(feature = "gpu-contact")]
    #[test]
    fn gpu_mass_batch_solves_live_articulated_systems() {
        let mut batch = ArticulatedBatch::new();
        let first = batch.add_environment(falling_world(-1.0));
        let second = batch.add_environment(motor_world(0.0, 0.4));
        batch
            .environment_mut(first)
            .unwrap()
            .set_link_external_wrench(0, Vector3::new(2.0, 0.0, 0.0), Vector3::z())
            .unwrap();
        batch
            .environment_mut(second)
            .unwrap()
            .set_link_gravity_scale(1, 0.5)
            .unwrap();
        batch.environment_mut(second).unwrap().velocities[0] = 0.3;
        batch
            .environment_mut(second)
            .unwrap()
            .set_joint_passive(
                0,
                crate::articulated_world::JointPassive {
                    stiffness: 1.2,
                    damping: 0.8,
                    rest_position: 0.3,
                },
            )
            .unwrap();
        batch
            .environment_mut(second)
            .unwrap()
            .articulation
            .set_joint_armature(0, &[0.7])
            .unwrap();
        let torques: &[&[f64]] = &[&[], &[0.2]];
        let expected = batch
            .worlds
            .iter()
            .zip(torques)
            .map(|(world, torque)| {
                let system = world.generalized_mass_system(torque).unwrap();
                system.mass.lu().solve(&system.force).unwrap()
            })
            .collect::<Vec<_>>();
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let actual = batch
                .generalized_accelerations_gpu(torques, &context)
                .unwrap();
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(&expected) {
                assert!((actual - expected).norm() < 2e-4, "{backend:?}");
            }
            let assembled = batch
                .worlds
                .iter()
                .zip(torques)
                .map(|(world, torque)| world.generalized_mass_assembly_system(torque).unwrap())
                .collect::<Vec<_>>();
            let inverse_batch =
                GpuArticulatedMassAssemblyBatch::new(context.device(), context.queue(), &assembled)
                    .unwrap();
            inverse_batch.submit_with_inverse();
            let inverses = inverse_batch.readback_inverse().unwrap();
            for ((world, torque), inverse) in batch.worlds.iter().zip(torques).zip(&inverses) {
                let mass = world.generalized_mass_system(torque).unwrap().mass;
                let identity = DMatrix::identity(mass.nrows(), mass.ncols());
                assert!((&mass * inverse - identity).norm() < 2e-4, "{backend:?}");
            }
            let mut session = ArticulatedMassSession::new(&context);
            let first = session.solve(&batch, torques).unwrap();
            for (actual, expected) in first.iter().zip(&expected) {
                assert!((actual - expected).norm() < 2e-4, "{backend:?}");
            }
            let mass_solution = session.solve_with_inverse(&batch, torques).unwrap();
            for ((world, torque), (acceleration, inverse)) in batch.worlds.iter().zip(torques).zip(
                mass_solution
                    .accelerations
                    .iter()
                    .zip(&mass_solution.inverse_masses),
            ) {
                let system = world.generalized_mass_system(torque).unwrap();
                let identity = DMatrix::identity(system.mass.nrows(), system.mass.ncols());
                assert!(
                    (&system.mass * acceleration - &system.force).norm() < 2e-4,
                    "{backend:?}"
                );
                assert!(
                    (&system.mass * inverse - identity).norm() < 2e-4,
                    "{backend:?}"
                );
            }
            batch.environment_mut(second).unwrap().velocities[0] = -0.7;
            let next_torques: &[&[f64]] = &[&[], &[-0.4]];
            let next = session.solve(&batch, next_torques).unwrap();
            for ((world, torque), actual) in batch.worlds.iter().zip(next_torques).zip(&next) {
                let system = world.generalized_mass_system(torque).unwrap();
                let expected = system.mass.lu().solve(&system.force).unwrap();
                assert!((actual - expected).norm() < 2e-4, "{backend:?}");
            }
            let third = batch.add_environment(falling_world(1.0));
            assert_eq!(third.index(), 2);
            let expanded_torques: &[&[f64]] = &[&[], &[-0.4], &[]];
            let expanded = session.solve(&batch, expanded_torques).unwrap();
            assert_eq!(expanded.len(), 3);
            for ((world, torque), actual) in
                batch.worlds.iter().zip(expanded_torques).zip(&expanded)
            {
                let system = world.generalized_mass_system(torque).unwrap();
                let expected = system.mass.lu().solve(&system.force).unwrap();
                assert!((actual - expected).norm() < 2e-4, "{backend:?}");
            }
            assert!(batch.worlds.pop().is_some());
            assert!(batch.reset_templates.pop().is_some());
            batch.environment_mut(second).unwrap().velocities[0] = 0.3;
        }
        assert!(tested > 0);
    }

    #[test]
    fn environments_step_independently_and_reset_individually() {
        let mut batch = ArticulatedBatch::new();
        let first = batch.add_environment(falling_world(-1.0));
        let second = batch.add_environment(falling_world(2.0));

        batch.step(0.1, &[&[], &[]]).unwrap();
        assert!(
            batch
                .environment(first)
                .unwrap()
                .root_pose
                .translation
                .vector
                .z
                < 2.0
        );
        assert!(
            batch
                .environment(second)
                .unwrap()
                .root_pose
                .translation
                .vector
                .z
                < 2.0
        );

        batch.reset_environment(first).unwrap();
        assert_eq!(
            batch
                .environment(first)
                .unwrap()
                .root_pose
                .translation
                .vector,
            Vector3::new(-1.0, 0.0, 2.0)
        );
        assert!(
            batch
                .environment(second)
                .unwrap()
                .root_pose
                .translation
                .vector
                .z
                < 2.0
        );
    }

    #[test]
    fn publishing_template_captures_runtime_state_and_scene_topology() {
        let mut batch = ArticulatedBatch::new();
        let id = batch.add_environment(falling_world(0.0));
        let scene = SceneBody::new(
            Isometry3::translation(0.0, 0.0, 3.0),
            0.0,
            Matrix3::zeros(),
            vec![SceneCollider::Sphere {
                center: Vector3::zeros(),
                radius: 0.5,
            }],
        )
        .unwrap();
        let _ = batch.environment_mut(id).unwrap().add_scene_body(scene);
        batch
            .environment_mut(id)
            .unwrap()
            .root_pose
            .translation
            .vector
            .x = 4.0;
        batch.publish_reset_template(id).unwrap();
        assert!(batch.environment_mut(id).unwrap().remove_scene_body(0));
        batch
            .environment_mut(id)
            .unwrap()
            .root_pose
            .translation
            .vector
            .x = 9.0;
        batch.reset_environment(id).unwrap();
        assert_eq!(
            batch
                .environment(id)
                .unwrap()
                .root_pose
                .translation
                .vector
                .x,
            4.0
        );
        assert_eq!(batch.environment(id).unwrap().scene_bodies.len(), 1);
    }

    #[test]
    fn torque_count_must_match_environment_count() {
        let mut batch = ArticulatedBatch::new();
        let _ = batch.add_environment(falling_world(0.0));
        assert!(matches!(
            batch.step(0.01, &[]),
            Err(BatchError::EnvironmentCount {
                expected: 1,
                actual: 0
            })
        ));
    }

    #[cfg(feature = "gpu-contact")]
    #[test]
    fn packed_gpu_batch_matches_individual_world_steps_with_ground_contact() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let build_worlds = || {
            [-1.0, 1.0, 0.0]
                .into_iter()
                .enumerate()
                .map(|(index, x)| {
                    let mut world = if index == 2 {
                        falling_world_with(x, 0.02, 16)
                    } else {
                        falling_world(x)
                    };
                    world.root_pose.translation.vector.z = 0.24 + index as f64 * 0.005;
                    world.base_linear_velocity.z = -0.5;
                    world
                })
                .collect::<Vec<_>>()
        };
        let mut individual = build_worlds();
        let mut batch = ArticulatedBatch::new();
        for world in build_worlds() {
            let _ = batch.add_environment(world);
        }
        for _ in 0..4 {
            batch.step_gpu(0.02, &[&[], &[], &[]], &context).unwrap();
            for world in &mut individual {
                world.step_gpu(0.02, &[], &context).unwrap();
            }
        }
        for (index, expected) in individual.iter().enumerate() {
            let actual = batch.environment(EnvironmentId(index)).unwrap();
            assert!(
                (actual.root_pose.translation.vector - expected.root_pose.translation.vector)
                    .norm()
                    < 1e-5
            );
            assert!((actual.base_linear_velocity - expected.base_linear_velocity).norm() < 1e-5);
            assert!((actual.base_angular_velocity - expected.base_angular_velocity).norm() < 1e-5);
            assert!((actual.contact_forces[0] - expected.contact_forces[0]).norm() < 1e-4);
        }
        assert!(
            individual
                .iter()
                .any(|world| world.contact_forces[0].z > 0.0)
        );
    }

    #[cfg(feature = "gpu-contact")]
    #[test]
    fn packed_device_mass_matches_individual_worlds_with_mixed_dofs_and_substeps() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let build_worlds = || {
            let mut falling = falling_world(-2.0);
            falling.root_pose.translation.vector.z = 0.24;
            falling.base_linear_velocity.z = -0.5;
            let mut motor = motor_world(0.0, 0.4);
            motor.velocities[0] = 0.3;
            let mut scene = falling_world_with(2.0, 0.02, 12);
            scene.root_pose.translation.vector.z = 0.24;
            let prescribed = SceneBody::new(
                Isometry3::translation(10.0, 0.0, 5.0),
                0.0,
                Matrix3::identity(),
                vec![SceneCollider::Sphere {
                    center: Vector3::zeros(),
                    radius: 0.25,
                }],
            )
            .unwrap();
            let first = scene.add_scene_body(prescribed.clone());
            scene
                .set_scene_body_kinematic_motion(
                    first,
                    Some((Vector3::x() * 0.005, Vector3::zeros())),
                )
                .unwrap();
            let body = SceneBody::new(
                Isometry3::translation(3.0, 0.0, 0.23),
                1.0,
                Matrix3::identity() * 0.1,
                vec![SceneCollider::Sphere {
                    center: Vector3::zeros(),
                    radius: 0.25,
                }],
            )
            .unwrap();
            let _ = scene.add_scene_body(body);
            let mut prescribed = prescribed;
            prescribed.pose.translation.vector.x = 12.0;
            let last = scene.add_scene_body(prescribed);
            scene
                .set_scene_body_kinematic_motion(
                    last,
                    Some((Vector3::zeros(), Vector3::z() * 0.005)),
                )
                .unwrap();
            vec![falling, motor, scene]
        };
        let mut individual = build_worlds();
        let mut batch = ArticulatedBatch::new();
        for world in build_worlds() {
            let _ = batch.add_environment(world);
        }
        let torques: &[&[f64]] = &[&[], &[0.2], &[]];
        for _ in 0..2 {
            batch.step_gpu_device_mass(0.02, torques, &context).unwrap();
            for (world, torque) in individual.iter_mut().zip(torques) {
                world.step_gpu_device_mass(0.02, torque, &context).unwrap();
            }
        }
        for (index, expected) in individual.iter().enumerate() {
            let actual = batch.environment(EnvironmentId(index)).unwrap();
            assert!(
                (actual.root_pose.translation.vector - expected.root_pose.translation.vector)
                    .norm()
                    < 2e-5
            );
            assert!((&actual.positions - &expected.positions).norm() < 2e-5);
            assert!((&actual.velocities - &expected.velocities).norm() < 2e-5);
            assert!((actual.base_linear_velocity - expected.base_linear_velocity).norm() < 2e-5);
            for (actual_body, expected_body) in
                actual.scene_bodies.iter().zip(&expected.scene_bodies)
            {
                assert!(
                    (actual_body.pose.translation.vector - expected_body.pose.translation.vector)
                        .norm()
                        < 2e-5
                );
                assert!(
                    (actual_body.linear_velocity - expected_body.linear_velocity).norm() < 2e-5
                );
            }
        }
        assert!(
            individual[0].contact_forces[0].z > 0.0
                && individual[2].scene_bodies[1].linear_velocity.z > 0.0
        );
    }

    #[cfg(feature = "gpu-contact")]
    #[test]
    fn device_mass_step_session_reuses_buffers_until_environment_layout_changes() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut batch = ArticulatedBatch::new();
        let mut falling = falling_world(0.0);
        falling.root_pose.translation.vector.z = 0.24;
        falling.base_linear_velocity.z = -0.5;
        let first = batch.add_environment(falling);
        let _ = batch.add_environment(motor_world(2.0, 0.1));
        let mut session = ArticulatedDeviceMassStepSession::new(&context);
        session.step(&mut batch, 0.001, &[&[], &[0.0]]).unwrap();
        let first_buffer = session
            .cache
            .mass_batch
            .as_ref()
            .and_then(GpuArticulatedMassAssemblyBatch::inverse_buffer)
            .unwrap()
            .clone();
        session.step(&mut batch, 0.001, &[&[], &[0.0]]).unwrap();
        let reused = session
            .cache
            .mass_batch
            .as_ref()
            .and_then(GpuArticulatedMassAssemblyBatch::inverse_buffer)
            .unwrap();
        assert_eq!(&first_buffer, reused);
        assert_ne!(
            batch
                .environment(first)
                .unwrap()
                .root_pose
                .translation
                .vector
                .z,
            0.24
        );

        let _ = batch.add_environment(falling_world(4.0));
        session
            .step(&mut batch, 0.001, &[&[], &[0.0], &[]])
            .unwrap();
        let rebuilt = session
            .cache
            .mass_batch
            .as_ref()
            .and_then(GpuArticulatedMassAssemblyBatch::inverse_buffer)
            .unwrap();
        assert_ne!(&first_buffer, rebuilt);
    }

    #[cfg(feature = "gpu-contact")]
    #[test]
    fn device_mass_step_cache_rebuilds_for_a_different_gpu_device() {
        let (Ok(first_device), Ok(second_device)) =
            (GpuContactDevice::new(), GpuContactDevice::new())
        else {
            return;
        };
        assert!(!Arc::ptr_eq(
            first_device.identity(),
            second_device.identity()
        ));
        let mut batch = ArticulatedBatch::new();
        let _ = batch.add_environment(falling_world(0.0));
        let mut cache = ArticulatedDeviceMassStepCache::new();
        cache
            .step(&mut batch, 0.001, &[&[]], &first_device)
            .unwrap();
        assert!(cache.mass_batch.is_some());
        cache
            .step(&mut batch, 0.001, &[&[]], &second_device)
            .unwrap();
        assert!(cache.mass_batch.is_some());
        assert!(Arc::ptr_eq(
            cache.device_identity.as_ref().unwrap(),
            second_device.identity()
        ));
    }

    #[cfg(feature = "gpu-contact")]
    #[test]
    fn packed_gpu_batch_preserves_independent_joint_motors() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let mut individual = [motor_world(-1.0, 0.2), motor_world(1.0, -0.2)];
        let mut batch = ArticulatedBatch::new();
        let first = batch.add_environment(motor_world(-1.0, 0.2));
        let second = batch.add_environment(motor_world(1.0, -0.2));
        for _ in 0..3 {
            batch.step_gpu(0.02, &[&[0.0], &[0.0]], &context).unwrap();
            for world in &mut individual {
                world.step_gpu(0.02, &[0.0], &context).unwrap();
            }
        }
        for (id, expected) in [(first, &individual[0]), (second, &individual[1])] {
            let actual = batch.environment(id).unwrap();
            assert!((&actual.positions - &expected.positions).norm() < 1e-5);
            assert!((&actual.velocities - &expected.velocities).norm() < 1e-5);
            assert!((actual.contact_forces[1] - expected.contact_forces[1]).norm() < 1e-4);
        }
        assert_ne!(
            batch.environment(first).unwrap().joint_motor(0),
            batch.environment(second).unwrap().joint_motor(0)
        );
        batch
            .environment_mut(first)
            .unwrap()
            .set_joint_motor(0, None)
            .unwrap();
        batch.reset_environment(first).unwrap();
        assert_eq!(
            batch
                .environment(first)
                .unwrap()
                .joint_motor(0)
                .unwrap()
                .position_target,
            Some(0.2)
        );
    }
}
