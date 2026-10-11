//! Complete selected scene resets, including contact-bound and unpaired prescribed bodies.

use super::*;
use crate::articulated_world::ArticulatedWorldSnapshot;
use crate::gpu_articulated_dynamics::GpuArticulatedDynamicsResetTemplates;
use crate::gpu_articulated_ground_contact::scene_reset::SceneContactResetTemplates;
use crate::gpu_articulated_state::reset::GpuArticulatedResetSelection;
use crate::gpu_kinematic_body::GpuKinematicResetTemplates;

/// Immutable templates belonging to one resident scene owner.
///
/// Restore selects original environment slots; geometry topology is fixed.
/// GPU state, physical parameters, held controls and prescribed integration
/// anchors are copied on the device without reading back unselected environments.
#[derive(Debug)]
pub struct GpuSceneResetTemplates {
    dynamics: GpuArticulatedDynamicsResetTemplates,
    contacts: Option<SceneContactResetTemplates>,
    kinematic: Option<GpuKinematicResetTemplates>,
    worlds: Vec<ArticulatedWorldSnapshot>,
    sources: Vec<WorldSceneDynamics>,
    configurations: Vec<ResidentSceneConfiguration>,
    bodies: Vec<Option<GpuArticulatedExternalSphereBodies>>,
    unpaired: Vec<Vec<(usize, GpuKinematicBody)>>,
}

fn capture(
    batch: &GpuArticulatedDynamicsBatch,
    unpaired: &UnpairedBodies,
    worlds: &[&ArticulatedWorld],
    sources: &[WorldSceneDynamics],
    configurations: &[ResidentSceneConfiguration],
    expected_states: &[SceneGeneralizedState],
    expected_bodies: &[Option<GpuArticulatedExternalSphereBodies>],
) -> Result<GpuSceneResetTemplates, GpuSceneDynamicsError> {
    if worlds.len() != sources.len() {
        return Err(GpuSceneDynamicsError::EnvironmentCount);
    }
    let current = worlds
        .iter()
        .map(|world| world.compose_scene_dynamics())
        .collect::<Result<Vec<_>, _>>()?;
    for (index, world) in worlds.iter().enumerate() {
        if !configurations[index].matches_world(world)
            || !matches_source(&current[index], &sources[index], &expected_states[index])
            || !unpaired.matches(index, world, &expected_bodies[index])
            || world
                .gpu_scene_dynamics_input(&current[index], &vec![0.0; world.articulation.dof()])?
                .external_sphere_bodies
                != expected_bodies[index]
        {
            return Err(GpuSceneDynamicsError::WorldChanged);
        }
    }
    let mut encoder = batch
        .state()
        .device()
        .create_command_encoder(&Default::default());
    let dynamics = batch.snapshot_reset_templates(&mut encoder)?;
    let contacts = batch.snapshot_scene_contacts(&mut encoder);
    let kinematic = unpaired
        .batch
        .as_ref()
        .map(|bodies| bodies.snapshot_reset_templates(&mut encoder));
    let _submission = batch.state().queue().submit([encoder.finish()]);
    Ok(GpuSceneResetTemplates {
        dynamics,
        contacts,
        kinematic,
        worlds: worlds.iter().map(|world| world.snapshot()).collect(),
        sources: current,
        configurations: configurations.to_vec(),
        bodies: expected_bodies.to_vec(),
        unpaired: unpaired.expected.clone(),
    })
}

fn validate(
    batch: &GpuArticulatedDynamicsBatch,
    unpaired: &UnpairedBodies,
    worlds: &[&ArticulatedWorld],
    templates: &GpuSceneResetTemplates,
    environments: &[usize],
) -> Result<(), GpuSceneDynamicsError> {
    if worlds.len() != templates.worlds.len()
        || !batch.matches_reset_templates(&templates.dynamics)
        || !batch.matches_scene_contacts(&templates.contacts)
        || match (&unpaired.batch, &templates.kinematic) {
            (Some(bodies), Some(template)) => !bodies.matches_reset_templates(template),
            (None, None) => false,
            _ => true,
        }
    {
        return Err(GpuSceneDynamicsError::WorldChanged);
    }
    for (selection, &environment) in environments.iter().enumerate() {
        let world = worlds
            .get(environment)
            .ok_or(GpuSceneDynamicsError::EnvironmentCount)?;
        if environments[..selection].contains(&environment) {
            return Err(GpuSceneDynamicsError::EnvironmentCount);
        }
        if !templates.configurations[environment].matches_static_pose_update(world) {
            return Err(GpuSceneDynamicsError::WorldChanged);
        }
        world.validate_snapshot(&templates.worlds[environment])?;
    }
    Ok(())
}

fn encode(
    batch: &mut GpuArticulatedDynamicsBatch,
    unpaired: &UnpairedBodies,
    encoder: &mut wgpu::CommandEncoder,
    templates: &GpuSceneResetTemplates,
    environments: &[usize],
) -> Result<(), GpuSceneDynamicsError> {
    let selections = environments
        .iter()
        .map(|&environment| GpuArticulatedResetSelection {
            environment,
            template: environment,
            root_translation: None,
            root_velocity: None,
        })
        .collect::<Vec<_>>();
    batch.encode_reset_envs_from_templates(encoder, &templates.dynamics, &selections)?;
    batch.encode_reset_scene_contacts(encoder, &templates.contacts, environments);
    if let (Some(bodies), Some(template)) = (&unpaired.batch, &templates.kinematic) {
        bodies.encode_reset_environments(encoder, template, environments);
    }
    Ok(())
}

impl GpuSceneDynamicsBatch {
    /// Capture a synchronized full-scene template without GPU readback.
    pub fn snapshot_reset_templates(
        &self,
        worlds: &[ArticulatedWorld],
    ) -> Result<GpuSceneResetTemplates, GpuSceneDynamicsError> {
        if self.faulted {
            return Err(GpuSceneDynamicsError::Faulted);
        }
        capture(
            &self.batch,
            &self.unpaired,
            &worlds.iter().collect::<Vec<_>>(),
            &self.sources,
            &self.configurations,
            &self.expected_states,
            &self.expected_bodies,
        )
    }

    /// Restore only the listed original environment slots and their host worlds.
    ///
    /// Duplicate or out-of-range indices and templates from another owner are
    /// rejected before recording mutations. Empty selection is a no-op.
    pub fn reset_envs_from_templates(
        &mut self,
        worlds: &mut [ArticulatedWorld],
        templates: &GpuSceneResetTemplates,
        environments: &[usize],
    ) -> Result<(), GpuSceneDynamicsError> {
        if self.faulted {
            return Err(GpuSceneDynamicsError::Faulted);
        }
        validate(
            &self.batch,
            &self.unpaired,
            &worlds.iter().collect::<Vec<_>>(),
            templates,
            environments,
        )?;
        if environments.is_empty() {
            return Ok(());
        }
        let mut encoder = self
            .batch
            .state()
            .device()
            .create_command_encoder(&Default::default());
        encode(
            &mut self.batch,
            &self.unpaired,
            &mut encoder,
            templates,
            environments,
        )?;
        self.faulted = true;
        let _submission = self.batch.state().queue().submit([encoder.finish()]);
        for &environment in environments {
            worlds[environment].restore_snapshot(&templates.worlds[environment])?;
            self.sources[environment].clone_from(&templates.sources[environment]);
            self.expected_states[environment].clone_from(&templates.sources[environment].state);
            self.configurations[environment].clone_from(&templates.configurations[environment]);
            self.bodies[environment].clone_from(&templates.bodies[environment]);
            self.expected_bodies[environment].clone_from(&templates.bodies[environment]);
            self.unpaired.expected[environment].clone_from(&templates.unpaired[environment]);
        }
        self.faulted = false;
        Ok(())
    }
}

impl GpuSceneDynamics {
    /// Capture the entire synchronized scene on the GPU.
    pub fn snapshot_reset_templates(
        &self,
        world: &ArticulatedWorld,
    ) -> Result<GpuSceneResetTemplates, GpuSceneDynamicsError> {
        if self.faulted {
            return Err(GpuSceneDynamicsError::Faulted);
        }
        capture(
            &self.batch,
            &self.unpaired,
            &[world],
            core::slice::from_ref(&self.source),
            core::slice::from_ref(&self.configuration),
            core::slice::from_ref(&self.expected_state),
            core::slice::from_ref(&self.expected_bodies),
        )
    }

    /// Restore the robot, dynamic scene bodies and all prescribed/static bodies.
    pub fn reset_from_templates(
        &mut self,
        world: &mut ArticulatedWorld,
        templates: &GpuSceneResetTemplates,
    ) -> Result<(), GpuSceneDynamicsError> {
        if self.faulted {
            return Err(GpuSceneDynamicsError::Faulted);
        }
        validate(&self.batch, &self.unpaired, &[world], templates, &[0])?;
        let mut encoder = self
            .batch
            .state()
            .device()
            .create_command_encoder(&Default::default());
        encode(
            &mut self.batch,
            &self.unpaired,
            &mut encoder,
            templates,
            &[0],
        )?;
        self.faulted = true;
        let _submission = self.batch.state().queue().submit([encoder.finish()]);
        world.restore_snapshot(&templates.worlds[0])?;
        self.source.clone_from(&templates.sources[0]);
        self.expected_state.clone_from(&templates.sources[0].state);
        self.configuration.clone_from(&templates.configurations[0]);
        self.bodies.clone_from(&templates.bodies[0]);
        self.expected_bodies.clone_from(&templates.bodies[0]);
        self.unpaired.expected.clone_from(&templates.unpaired);
        self.faulted = false;
        Ok(())
    }
}
