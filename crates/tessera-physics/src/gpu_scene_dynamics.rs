//! Reusable resident robot/scene dynamics with synchronized host world state.

use crate::articulated_world::{
    ArticulatedWorld, ArticulatedWorldError, ResidentSceneConfiguration, SceneContactDiagnostics,
    WorldSceneDynamics,
};
use crate::articulation::{ArticulationError, SceneGeneralizedState};
use crate::gpu_articulated_dynamics::{
    GpuArticulatedDynamicsBatch, GpuArticulatedDynamicsError, GpuArticulatedDynamicsInput,
};
use crate::gpu_articulated_ground_contact::GpuArticulatedExternalSphereBodies;
use crate::gpu_contact_pipeline::GpuContactDevice;
use crate::gpu_kinematic_body::{GpuKinematicBody, GpuKinematicBodyBatch, GpuKinematicBodyError};

/// Failure to construct, advance, or synchronize resident scene dynamics.
#[derive(Debug, thiserror::Error)]
pub enum GpuSceneDynamicsError {
    /// Unsupported or invalid host world state.
    #[error(transparent)]
    World(#[from] ArticulatedWorldError),
    /// Invalid composed articulation or quaternion layout.
    #[error(transparent)]
    Articulation(#[from] ArticulationError),
    /// GPU construction, execution, or readback failed.
    #[error(transparent)]
    Dynamics(#[from] GpuArticulatedDynamicsError),
    /// Independent prescribed body integration failed.
    #[error(transparent)]
    Kinematic(#[from] GpuKinematicBodyError),
    /// External state or persistent layout changed; rebuild the session.
    #[error("world state or resident configuration changed; rebuild the session")]
    WorldChanged,
    /// A previous GPU step failed; state cannot be reused safely.
    #[error("resident session is faulted; rebuild the session")]
    Faulted,
    /// At least one fixed-timestep step must be requested.
    #[error("step count must be positive")]
    InvalidStepCount,
    /// Environment or effort counts do not match the resident layout.
    #[error("resident environment or effort count does not match")]
    EnvironmentCount,
    /// The single-environment batch did not return its environment.
    #[error("resident environment output is missing")]
    MissingOutput,
}

#[derive(Debug)]
struct UnpairedBodies {
    batch: Option<GpuKinematicBodyBatch>,
    expected: Vec<Vec<(usize, GpuKinematicBody)>>,
}

impl UnpairedBodies {
    fn collect(
        world: &ArticulatedWorld,
        mapping: &Option<GpuArticulatedExternalSphereBodies>,
    ) -> Vec<(usize, GpuKinematicBody)> {
        world
            .prescribed_gpu_bodies()
            .into_iter()
            .filter(|(slot, _)| {
                !mapping.as_ref().is_some_and(|mapping| {
                    mapping
                        .spheres
                        .iter()
                        .chain(&mapping.capsules)
                        .chain(&mapping.boxes)
                        .chain(&mapping.axial)
                        .chain(&mapping.convex)
                        .flatten()
                        .any(|input| input.user_data == Some(*slot))
                })
            })
            .collect()
    }

    fn new(
        worlds: &[&ArticulatedWorld],
        mappings: &[Option<GpuArticulatedExternalSphereBodies>],
        context: &GpuContactDevice,
        timestep: f64,
    ) -> Result<Self, GpuSceneDynamicsError> {
        let expected: Vec<_> = worlds
            .iter()
            .zip(mappings)
            .map(|(world, mapping)| Self::collect(world, mapping))
            .collect();
        let states: Vec<Vec<_>> = expected
            .iter()
            .map(|bodies| bodies.iter().map(|(_, body)| body.clone()).collect())
            .collect();
        let batch = if states.iter().all(Vec::is_empty) {
            None
        } else {
            Some(GpuKinematicBodyBatch::new(
                context.device(),
                context.queue(),
                &states,
                timestep,
            )?)
        };
        Ok(Self { batch, expected })
    }

    fn matches(
        &self,
        index: usize,
        world: &ArticulatedWorld,
        mapping: &Option<GpuArticulatedExternalSphereBodies>,
    ) -> bool {
        Self::collect(world, mapping) == self.expected[index]
    }

    fn advance(
        &self,
        count: usize,
    ) -> Result<Vec<Vec<(usize, GpuKinematicBody)>>, GpuSceneDynamicsError> {
        let Some(batch) = &self.batch else {
            return Ok(self.expected.clone());
        };
        batch.submit_steps(count)?;
        let output = batch.readback()?;
        if output.len() != self.expected.len() {
            return Err(GpuSceneDynamicsError::MissingOutput);
        }
        output
            .into_iter()
            .zip(&self.expected)
            .map(|(output, expected)| {
                if output.len() != expected.len() {
                    return Err(GpuSceneDynamicsError::MissingOutput);
                }
                Ok(output
                    .into_iter()
                    .zip(expected)
                    .map(|(body, (slot, _))| (*slot, body))
                    .collect())
            })
            .collect()
    }
}

fn read_external_orbits(
    batch: &GpuArticulatedDynamicsBatch,
    mappings: &[Option<GpuArticulatedExternalSphereBodies>],
) -> Result<
    crate::gpu_articulated_ground_contact::GpuArticulatedExternalSphereOrbits,
    GpuSceneDynamicsError,
> {
    if mappings.iter().flatten().any(|m| {
        !m.spheres.is_empty()
            || !m.capsules.is_empty()
            || !m.boxes.is_empty()
            || !m.axial.is_empty()
            || !m.convex.is_empty()
    }) {
        return Ok(batch.readback_external_sphere_orbits()?);
    }
    let empty = vec![vec![]; mappings.len()];
    Ok(
        crate::gpu_articulated_ground_contact::GpuArticulatedExternalSphereOrbits {
            spheres: empty.clone(),
            capsules: empty.clone(),
            boxes: empty.clone(),
            axial: empty.clone(),
            convex: empty,
        },
    )
}

fn matches_source(
    current: &WorldSceneDynamics,
    source: &WorldSceneDynamics,
    expected: &SceneGeneralizedState,
) -> bool {
    current.scene == source.scene
        && current.scene_slots == source.scene_slots
        && current.point_constraints == source.point_constraints
        && current.fixed_constraints == source.fixed_constraints
        && current.joint_friction == source.joint_friction
        && current.joint_couplings == source.joint_couplings
        && current.gravity == source.gravity
        && current.joint_velocity_limit == source.joint_velocity_limit
        && current.state.positions == expected.positions
        && current.state.velocities == expected.velocities
        && current.state.orientations == expected.orientations
}

fn upload_static_scene_poses(
    batch: &mut GpuArticulatedDynamicsBatch,
    inputs: &[GpuArticulatedDynamicsInput<'_>],
) -> Result<(), GpuArticulatedDynamicsError> {
    if !batch.has_contact_rows() {
        batch.initialize_external_sphere_bodies(inputs)?;
        batch.initialize_external_box_bodies(inputs)?;
        batch.initialize_external_capsule_bodies(inputs)?;
        batch.initialize_external_axial_bodies(inputs)?;
        return batch.initialize_external_convex_bodies(inputs);
    }
    macro_rules! update {
        ($method:ident, $input:ident, $values:expr) => {
            let _changed = batch.$method(
                &inputs
                    .iter()
                    .map(|$input| $values.collect::<Vec<_>>())
                    .collect::<Vec<_>>(),
            )?;
        };
    }
    update!(
        update_static_sphere_centers,
        i,
        i.static_sphere_pairs.iter().map(|p| p.static_center)
    );
    update!(
        update_static_capsule_sphere_centers,
        i,
        i.static_capsule_sphere_pairs
            .iter()
            .map(|p| p.static_center)
    );
    update!(
        update_static_box_sphere_centers,
        i,
        i.static_box_sphere_pairs.iter().map(|p| p.static_center)
    );
    update!(
        update_static_axial_sphere_centers,
        i,
        i.static_axial_sphere_pairs
            .iter()
            .filter(|p| !p.axial_is_static)
            .map(|p| p.static_center)
    );
    update!(
        update_static_convex_sphere_centers,
        i,
        i.static_convex_sphere_pairs.iter().map(|p| p.static_center)
    );
    update!(
        update_static_sphere_box_poses,
        i,
        i.static_sphere_box_pairs.iter().map(|p| p.static_pose)
    );
    update!(
        update_static_capsule_box_poses,
        i,
        i.static_capsule_box_pairs.iter().map(|p| p.static_pose)
    );
    update!(
        update_static_box_box_poses,
        i,
        i.static_box_pairs.iter().map(|p| p.static_pose)
    );
    update!(
        update_static_axial_box_poses,
        i,
        i.static_axial_box_pairs
            .iter()
            .filter(|p| !p.axial_is_static)
            .map(|p| p.static_pose)
    );
    update!(
        update_static_convex_pair_poses,
        i,
        i.static_convex_pairs.iter().map(|p| p.second_world_pose)
    );
    update!(
        update_static_axial_convex_poses,
        i,
        i.static_axial_convex_pairs.iter().map(|p| p.static_pose)
    );
    update!(
        update_scene_convex_rounded_poses,
        i,
        i.scene_convex_sphere_pairs
            .iter()
            .map(|p| p.convex_world_pose)
            .chain(
                i.scene_convex_capsule_pairs
                    .iter()
                    .map(|p| p.convex_world_pose)
            )
    );
    let mut indexed_poses: Vec<Vec<_>> = inputs
        .iter()
        .map(|input| {
            input
                .scene_mesh_sphere_pairs
                .iter()
                .map(|p| p.mesh_world_pose)
                .chain(
                    input
                        .scene_polyline_sphere_pairs
                        .iter()
                        .map(|p| p.polyline_world_pose),
                )
                .chain(
                    input
                        .scene_mesh_capsule_pairs
                        .iter()
                        .map(|p| p.mesh_world_pose),
                )
                .chain(
                    input
                        .scene_polyline_capsule_pairs
                        .iter()
                        .map(|p| p.polyline_world_pose),
                )
                .chain(input.scene_mesh_box_pairs.iter().map(|p| p.mesh_world_pose))
                .chain(
                    input
                        .scene_polyline_box_pairs
                        .iter()
                        .map(|p| p.polyline_world_pose),
                )
                .chain(
                    input
                        .scene_mesh_axial_pairs
                        .iter()
                        .map(|p| p.mesh_world_pose),
                )
                .chain(
                    input
                        .scene_polyline_axial_pairs
                        .iter()
                        .map(|p| p.polyline_world_pose),
                )
                .chain(
                    input
                        .scene_mesh_convex_pairs
                        .iter()
                        .map(|p| p.mesh_world_pose),
                )
                .chain(
                    input
                        .scene_polyline_convex_pairs
                        .iter()
                        .map(|p| p.polyline_world_pose),
                )
                .collect()
        })
        .collect();
    if indexed_poses.iter().any(|poses| !poses.is_empty()) {
        let resident = batch.readback_prescribed_indexed_poses()?;
        for ((poses, current), input) in indexed_poses.iter_mut().zip(&resident).zip(inputs) {
            if let Some(bodies) = &input.external_indexed_bodies {
                for ((pose, current_pose), body) in poses.iter_mut().zip(current).zip(bodies) {
                    if body.is_some() {
                        *pose = *current_pose;
                    }
                }
            }
        }
        batch.update_indexed_geometry_poses(&indexed_poses)?;
    }
    let frames: Vec<Vec<_>> = inputs
        .iter()
        .map(|input| {
            input
                .link_point_constraints
                .iter()
                .filter(|c| c.link_b.is_none())
                .map(|c| nalgebra::Isometry3::translation(c.point_b[0], c.point_b[1], c.point_b[2]))
                .chain(
                    input
                        .link_fixed_constraints
                        .iter()
                        .filter(|c| c.link_b.is_none())
                        .map(|c| c.frame_b),
                )
                .collect()
        })
        .collect();
    if frames.iter().any(|values| !values.is_empty()) {
        batch.update_external_constraint_frames(&frames)?;
        batch.initialize_external_constraint_bodies(inputs)?;
    }
    batch.initialize_external_sphere_bodies(inputs)?;
    batch.initialize_external_box_bodies(inputs)?;
    batch.initialize_external_capsule_bodies(inputs)?;
    batch.initialize_external_axial_bodies(inputs)?;
    batch.initialize_external_convex_bodies(inputs)?;
    Ok(())
}

/// Packed resident dynamics for independent worlds with potentially mixed DOFs.
/// Geometry, materials, and solver configuration remain fixed until rebuilt.
/// All host states and efforts are checked before submitting the packed step.
#[derive(Debug)]
pub struct GpuSceneDynamicsBatch {
    batch: GpuArticulatedDynamicsBatch,
    sources: Vec<WorldSceneDynamics>,
    configurations: Vec<ResidentSceneConfiguration>,
    expected_states: Vec<SceneGeneralizedState>,
    bodies: Vec<Option<GpuArticulatedExternalSphereBodies>>,
    expected_bodies: Vec<Option<GpuArticulatedExternalSphereBodies>>,
    unpaired: UnpairedBodies,
    scene_sleep_enabled: bool,
    faulted: bool,
}

impl GpuSceneDynamicsBatch {
    /// Construct one device batch for all worlds in slice order.
    pub fn new(
        worlds: &[ArticulatedWorld],
        context: &GpuContactDevice,
        timestep: f64,
    ) -> Result<Self, GpuSceneDynamicsError> {
        if worlds.is_empty() {
            return Err(GpuSceneDynamicsError::EnvironmentCount);
        }
        let sources = worlds
            .iter()
            .map(ArticulatedWorld::compose_scene_dynamics)
            .collect::<Result<Vec<_>, _>>()?;
        let inputs = worlds
            .iter()
            .zip(&sources)
            .map(|(world, source)| {
                world.gpu_scene_dynamics_input(source, &vec![0.0; world.articulation.dof()])
            })
            .collect::<Result<Vec<_>, _>>()?;
        let bodies: Vec<_> = inputs
            .iter()
            .map(|input| input.external_sphere_bodies.clone())
            .collect();
        let spherical = sources
            .iter()
            .map(|source| source.scene.gpu_spherical_state(&source.state))
            .collect::<Result<Vec<_>, _>>()?;
        let batch = if spherical.iter().all(Vec::is_empty) {
            GpuArticulatedDynamicsBatch::new(context, &inputs, timestep)?
        } else {
            let drives: Vec<_> = spherical
                .iter()
                .map(|joints| vec![None; joints.len()])
                .collect();
            GpuArticulatedDynamicsBatch::new_with_spherical_state(
                context,
                &inputs,
                timestep,
                &vec![false; worlds.len()],
                &spherical,
                &drives,
            )?
        };
        let unpaired = UnpairedBodies::new(
            &worlds.iter().collect::<Vec<_>>(),
            &bodies,
            context,
            timestep,
        )?;
        Ok(Self {
            batch,
            unpaired,
            expected_states: sources.iter().map(|source| source.state.clone()).collect(),
            sources,
            configurations: worlds
                .iter()
                .map(ArticulatedWorld::resident_scene_configuration)
                .collect(),
            expected_bodies: bodies.clone(),
            bodies,
            scene_sleep_enabled: false,
            faulted: false,
        })
    }

    /// Update supported static scene geometry poses in every packed environment.
    /// Validate all host layouts before uploads; upload errors fault the batch.
    pub fn update_static_scene_poses(
        &mut self,
        worlds: &[ArticulatedWorld],
    ) -> Result<(), GpuSceneDynamicsError> {
        if self.faulted {
            return Err(GpuSceneDynamicsError::Faulted);
        }
        if worlds.len() != self.sources.len() {
            return Err(GpuSceneDynamicsError::EnvironmentCount);
        }
        let current = worlds
            .iter()
            .map(ArticulatedWorld::compose_scene_dynamics)
            .collect::<Result<Vec<_>, _>>()?;
        if worlds
            .iter()
            .zip(&self.configurations)
            .any(|(world, config)| !config.matches_static_pose_update(world))
            || current.iter().enumerate().any(|(index, source)| {
                !matches_source(source, &self.sources[index], &self.expected_states[index])
            })
        {
            return Err(GpuSceneDynamicsError::WorldChanged);
        }
        let inputs = worlds
            .iter()
            .zip(&current)
            .map(|(world, source)| {
                world.gpu_scene_dynamics_input(source, &vec![0.0; world.articulation.dof()])
            })
            .collect::<Result<Vec<_>, _>>()?;
        if worlds
            .iter()
            .enumerate()
            .any(|(index, world)| !self.unpaired.matches(index, world, &self.bodies[index]))
            || inputs
                .iter()
                .zip(&self.expected_bodies)
                .any(|(input, expected)| input.external_sphere_bodies != *expected)
        {
            return Err(GpuSceneDynamicsError::WorldChanged);
        }
        self.faulted = true;
        upload_static_scene_poses(&mut self.batch, &inputs)?;
        self.configurations = worlds
            .iter()
            .map(ArticulatedWorld::resident_scene_configuration)
            .collect();
        self.faulted = false;
        Ok(())
    }

    /// Enable automatic sleeping for scene bodies, keeping robot coordinates awake.
    /// Settings are captured from each world and require reconstruction after edits.
    pub fn enable_scene_sleep(
        &mut self,
        worlds: &[ArticulatedWorld],
    ) -> Result<(), GpuSceneDynamicsError> {
        if self.faulted {
            return Err(GpuSceneDynamicsError::Faulted);
        }
        if worlds.len() != self.sources.len() {
            return Err(GpuSceneDynamicsError::EnvironmentCount);
        }
        if worlds
            .iter()
            .zip(&self.configurations)
            .any(|(world, config)| !config.matches_world(world))
        {
            return Err(GpuSceneDynamicsError::WorldChanged);
        }
        let settings = worlds
            .iter()
            .zip(&self.sources)
            .map(|(world, source)| {
                source.scene.contact_sleep_settings(
                    None,
                    &vec![Some(world.params().sleep); source.scene.bodies.len()],
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        // A failed device reconfiguration must not leave a reusable partial setup.
        self.faulted = true;
        self.batch.enable_contact_activity()?;
        self.batch.enable_contact_link_sleep(&settings)?;
        self.scene_sleep_enabled = true;
        self.faulted = false;
        Ok(())
    }

    /// Advance all environments in one packed batch and synchronize host worlds.
    /// GPU or synchronization failures fault the whole session. Geometry edits
    /// require rebuilding. Efforts remain constant over the requested steps.
    pub fn step(
        &mut self,
        worlds: &mut [ArticulatedWorld],
        efforts: &[&[f64]],
        count: usize,
    ) -> Result<Vec<SceneContactDiagnostics>, GpuSceneDynamicsError> {
        if self.faulted {
            return Err(GpuSceneDynamicsError::Faulted);
        }
        if count == 0 {
            return Err(GpuSceneDynamicsError::InvalidStepCount);
        }
        if worlds.len() != self.sources.len() || efforts.len() != worlds.len() {
            return Err(GpuSceneDynamicsError::EnvironmentCount);
        }
        if worlds
            .iter()
            .zip(&self.configurations)
            .any(|(world, config)| !config.matches_world(world))
        {
            return Err(GpuSceneDynamicsError::WorldChanged);
        }
        let current = worlds
            .iter()
            .map(ArticulatedWorld::compose_scene_dynamics)
            .collect::<Result<Vec<_>, _>>()?;
        for (index, source) in current.iter().enumerate() {
            if !matches_source(source, &self.sources[index], &self.expected_states[index]) {
                return Err(GpuSceneDynamicsError::WorldChanged);
            }
        }
        let inputs = worlds
            .iter()
            .zip(&current)
            .zip(efforts)
            .map(|((world, source), efforts)| world.gpu_scene_dynamics_input(source, efforts))
            .collect::<Result<Vec<_>, _>>()?;
        if worlds
            .iter()
            .enumerate()
            .any(|(index, world)| !self.unpaired.matches(index, world, &self.bodies[index]))
            || inputs
                .iter()
                .zip(&self.expected_bodies)
                .any(|(input, expected)| input.external_sphere_bodies != *expected)
        {
            return Err(GpuSceneDynamicsError::WorldChanged);
        }
        // Device configuration uploads can succeed only partially.
        self.faulted = true;
        self.batch.update_joints(
            &inputs
                .iter()
                .map(|input| input.joints.clone())
                .collect::<Vec<_>>(),
        )?;
        self.batch.update_link_loads(
            &inputs
                .iter()
                .map(|input| input.link_loads.clone())
                .collect::<Vec<_>>(),
        )?;
        self.batch.submit_steps(count)?;
        let output = self.batch.readback_output()?;
        if output.len() != worlds.len() {
            return Err(GpuSceneDynamicsError::MissingOutput);
        }
        let prescribed = self.unpaired.advance(count)?;
        let orbits = if self.bodies.iter().any(Option::is_some) {
            Some(read_external_orbits(&self.batch, &self.bodies)?)
        } else {
            None
        };
        let sleeping = if self.scene_sleep_enabled {
            Some(self.batch.readback_sleeping_coordinates()?)
        } else {
            None
        };
        if sleeping
            .as_ref()
            .is_some_and(|flags| flags.len() != worlds.len())
        {
            return Err(GpuSceneDynamicsError::MissingOutput);
        }
        let mut diagnostics = Vec::with_capacity(worlds.len());
        for (index, (world, output)) in worlds.iter_mut().zip(&output).enumerate() {
            diagnostics.push(if let Some(bodies) = &self.bodies[index] {
                world.apply_scene_gpu_output_with_prescribed_bodies(
                    &self.sources[index],
                    output,
                    bodies,
                    orbits
                        .as_ref()
                        .ok_or(GpuSceneDynamicsError::MissingOutput)?,
                    index,
                    &prescribed[index],
                )?
            } else {
                world.apply_scene_gpu_output(&self.sources[index], output)?
            });
        }
        if let Some(sleeping) = sleeping {
            for ((world, source), coordinates) in
                worlds.iter_mut().zip(&self.sources).zip(&sleeping)
            {
                world.apply_scene_sleep_diagnostics(source, coordinates)?;
            }
        }
        let updated = worlds
            .iter()
            .map(ArticulatedWorld::compose_scene_dynamics)
            .collect::<Result<Vec<_>, _>>()?;
        self.expected_bodies = worlds
            .iter()
            .zip(&updated)
            .zip(efforts)
            .map(|((world, source), efforts)| {
                world
                    .gpu_scene_dynamics_input(source, efforts)
                    .map(|input| input.external_sphere_bodies)
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.unpaired.expected = worlds
            .iter()
            .zip(&self.bodies)
            .map(|(world, mapping)| UnpairedBodies::collect(world, mapping))
            .collect();
        for (source, current) in self.sources.iter_mut().zip(&updated) {
            source
                .point_constraints
                .clone_from(&current.point_constraints);
            source
                .fixed_constraints
                .clone_from(&current.fixed_constraints);
            source
                .external_constraint_bodies
                .clone_from(&current.external_constraint_bodies);
        }
        self.expected_states = updated.into_iter().map(|source| source.state).collect();
        self.faulted = false;
        Ok(diagnostics)
    }
}

/// Owns one resident robot/scene batch and its world-body correspondence.
///
/// Geometry, materials, solver settings, and prescribed velocities are fixed
/// for a session; rebuild after changing them. Generalized efforts and link
/// loads are refreshed on each call. Host state and kinematic poses are checked
/// before submission to prevent silently overwriting external state edits.
#[derive(Debug)]
pub struct GpuSceneDynamics {
    batch: GpuArticulatedDynamicsBatch,
    source: WorldSceneDynamics,
    configuration: ResidentSceneConfiguration,
    expected_state: SceneGeneralizedState,
    bodies: Option<GpuArticulatedExternalSphereBodies>,
    expected_bodies: Option<GpuArticulatedExternalSphereBodies>,
    unpaired: UnpairedBodies,
    scene_sleep_enabled: bool,
    faulted: bool,
}

impl GpuSceneDynamics {
    /// Build a fixed inertial-root batch containing the robot and scene bodies.
    /// Floating scene bodies and spherical robot joints use quaternion state.
    pub fn new(
        world: &ArticulatedWorld,
        context: &GpuContactDevice,
        timestep: f64,
    ) -> Result<Self, GpuSceneDynamicsError> {
        let source = world.compose_scene_dynamics()?;
        let input =
            world.gpu_scene_dynamics_input(&source, &vec![0.0; world.articulation.dof()])?;
        let bodies = input.external_sphere_bodies.clone();
        let spherical = source.scene.gpu_spherical_state(&source.state)?;
        let batch = if spherical.is_empty() {
            GpuArticulatedDynamicsBatch::new(context, &[input], timestep)?
        } else {
            let drives = vec![None; spherical.len()];
            GpuArticulatedDynamicsBatch::new_with_spherical_state(
                context,
                &[input],
                timestep,
                &[false],
                &[spherical],
                &[drives],
            )?
        };
        let unpaired =
            UnpairedBodies::new(&[world], core::slice::from_ref(&bodies), context, timestep)?;
        Ok(Self {
            batch,
            unpaired,
            expected_state: source.state.clone(),
            source,
            configuration: world.resident_scene_configuration(),
            expected_bodies: bodies.clone(),
            bodies,
            scene_sleep_enabled: false,
            faulted: false,
        })
    }

    /// Upload changed supported static scene geometry poses without rebuilding.
    /// Other configuration, generalized state, and prescribed body state must
    /// match. Partial upload errors fault the session.
    pub fn update_static_scene_poses(
        &mut self,
        world: &ArticulatedWorld,
    ) -> Result<(), GpuSceneDynamicsError> {
        if self.faulted {
            return Err(GpuSceneDynamicsError::Faulted);
        }
        let current = world.compose_scene_dynamics()?;
        if !self.configuration.matches_static_pose_update(world)
            || !matches_source(&current, &self.source, &self.expected_state)
        {
            return Err(GpuSceneDynamicsError::WorldChanged);
        }
        let input =
            world.gpu_scene_dynamics_input(&current, &vec![0.0; world.articulation.dof()])?;
        if input.external_sphere_bodies != self.expected_bodies
            || !self.unpaired.matches(0, world, &self.bodies)
        {
            return Err(GpuSceneDynamicsError::WorldChanged);
        }
        self.faulted = true;
        upload_static_scene_poses(&mut self.batch, &[input])?;
        self.configuration = world.resident_scene_configuration();
        self.faulted = false;
        Ok(())
    }

    /// Enable automatic scene-body sleeping and synchronize its host diagnostics.
    /// Robot coordinates remain awake. Configuration errors fault the session.
    pub fn enable_scene_sleep(
        &mut self,
        world: &ArticulatedWorld,
    ) -> Result<(), GpuSceneDynamicsError> {
        if self.faulted {
            return Err(GpuSceneDynamicsError::Faulted);
        }
        if !self.configuration.matches_world(world) {
            return Err(GpuSceneDynamicsError::WorldChanged);
        }
        let settings = self.source.scene.contact_sleep_settings(
            None,
            &vec![Some(world.params().sleep); self.source.scene.bodies.len()],
        )?;
        self.faulted = true;
        self.batch.enable_contact_activity()?;
        self.batch.enable_contact_link_sleep(&[settings])?;
        self.scene_sleep_enabled = true;
        self.faulted = false;
        Ok(())
    }

    /// Download coordinate sleep diagnostics without exposing mutable buffers.
    pub fn readback_sleeping_coordinates(&self) -> Result<Vec<Vec<bool>>, GpuSceneDynamicsError> {
        if self.faulted {
            return Err(GpuSceneDynamicsError::Faulted);
        }
        Ok(self.batch.readback_sleeping_coordinates()?)
    }

    /// Download per-link contact idle candidates for diagnostics.
    pub fn readback_contact_sleep_candidates(
        &self,
    ) -> Result<Vec<Vec<bool>>, GpuSceneDynamicsError> {
        if self.faulted {
            return Err(GpuSceneDynamicsError::Faulted);
        }
        Ok(self.batch.readback_contact_sleep_candidates()?)
    }

    /// Override per-link idle settings while retaining the resident sleep topology.
    pub fn update_scene_sleep_link_settings(
        &mut self,
        settings: &[Option<crate::sleep::SleepSettings>],
    ) -> Result<(), GpuSceneDynamicsError> {
        if self.faulted {
            return Err(GpuSceneDynamicsError::Faulted);
        }
        self.faulted = true;
        self.batch
            .update_contact_link_idle_settings(&[settings.to_vec()])?;
        self.faulted = false;
        Ok(())
    }

    /// Advance with constant efforts for count steps, then synchronize the world.
    /// Geometry and prescribed motion changes require a new session.
    /// A GPU or synchronization error permanently faults this session.
    pub fn step(
        &mut self,
        world: &mut ArticulatedWorld,
        efforts: &[f64],
        count: usize,
    ) -> Result<SceneContactDiagnostics, GpuSceneDynamicsError> {
        if self.faulted {
            return Err(GpuSceneDynamicsError::Faulted);
        }
        if count == 0 {
            return Err(GpuSceneDynamicsError::InvalidStepCount);
        }
        let current = world.compose_scene_dynamics()?;
        if !self.configuration.matches_world(world)
            || !matches_source(&current, &self.source, &self.expected_state)
        {
            return Err(GpuSceneDynamicsError::WorldChanged);
        }
        let input = world.gpu_scene_dynamics_input(&current, efforts)?;
        if input.external_sphere_bodies != self.expected_bodies
            || !self.unpaired.matches(0, world, &self.bodies)
        {
            return Err(GpuSceneDynamicsError::WorldChanged);
        }
        // Fault before the first upload, including failures before submission.
        self.faulted = true;
        self.batch.update_joints(&[input.joints])?;
        self.batch.update_link_loads(&[input.link_loads])?;
        self.batch.submit_steps(count)?;
        let output = self
            .batch
            .readback_output()?
            .into_iter()
            .next()
            .ok_or(GpuSceneDynamicsError::MissingOutput)?;
        let sleeping = if self.scene_sleep_enabled {
            Some(
                self.batch
                    .readback_sleeping_coordinates()?
                    .into_iter()
                    .next()
                    .ok_or(GpuSceneDynamicsError::MissingOutput)?,
            )
        } else {
            None
        };
        let prescribed = self.unpaired.advance(count)?;
        let diagnostics = if let Some(bodies) = &self.bodies {
            let orbits = read_external_orbits(&self.batch, core::slice::from_ref(&self.bodies))?;
            world.apply_scene_gpu_output_with_prescribed_bodies(
                &self.source,
                &output,
                bodies,
                &orbits,
                0,
                &prescribed[0],
            )?
        } else {
            world.apply_scene_gpu_output(&self.source, &output)?
        };
        if let Some(sleeping) = sleeping {
            world.apply_scene_sleep_diagnostics(&self.source, &sleeping)?;
        }
        let updated = world.compose_scene_dynamics()?;
        self.expected_bodies = world
            .gpu_scene_dynamics_input(&updated, efforts)?
            .external_sphere_bodies;
        self.unpaired.expected = vec![UnpairedBodies::collect(world, &self.bodies)];
        self.source.point_constraints = updated.point_constraints;
        self.source.fixed_constraints = updated.fixed_constraints;
        self.source.external_constraint_bodies = updated.external_constraint_bodies;
        self.expected_state = updated.state;
        self.faulted = false;
        Ok(diagnostics)
    }
}
