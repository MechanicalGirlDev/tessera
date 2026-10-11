//! Owner-bound full-scene templates retained on the GPU.

use super::*;
use tessera_physics::gpu_scene_dynamics::GpuSceneResetTemplates;

/// Immutable snapshot of one resident owner's full scenes and held controls.
///
/// Geometry topology and owner identity must remain unchanged. Templates restore
/// original environment slots without reading back the rest of the GPU batch.
#[derive(Debug, uniffi::Object)]
pub struct GpuSimulationSnapshot {
    scenes: GpuSceneResetTemplates,
    controls: Vec<Vec<f64>>,
}

#[uniffi::export]
impl ArticulatedWorld {
    /// Capture after a resident GPU step has synchronized the host world.
    pub fn snapshot_gpu_resident(&self) -> Result<Arc<GpuSimulationSnapshot>, TesseraError> {
        let inner = locked(&self.inner)?;
        let (_, session) = inner.resident.as_ref().ok_or_else(|| {
            failed("initialize resident GPU stepping before capturing a template")
        })?;
        Ok(Arc::new(GpuSimulationSnapshot {
            scenes: session
                .snapshot_reset_templates(&inner.world)
                .map_err(failed)?,
            controls: vec![inner.actuators.controls().to_vec()],
        }))
    }

    /// Restore robot, scene bodies, prescribed motion and imported held controls.
    pub fn restore_gpu_resident(
        &self,
        snapshot: Arc<GpuSimulationSnapshot>,
    ) -> Result<(), TesseraError> {
        if snapshot.controls.len() != 1 {
            return Err(failed("GPU template environment count does not match"));
        }
        let mut inner = locked(&self.inner)?;
        let mut restored = inner.actuators.clone();
        restored
            .set_controls(&snapshot.controls[0])
            .map_err(failed)?;
        let ArticulatedInner {
            world,
            resident,
            actuators,
            ..
        } = &mut *inner;
        let (_, session) = resident
            .as_mut()
            .ok_or_else(|| failed("resident GPU owner no longer exists"))?;
        session
            .reset_from_templates(world, &snapshot.scenes)
            .map_err(failed)?;
        *actuators = restored;
        Ok(())
    }
}

#[uniffi::export]
impl ArticulatedBatch {
    /// Capture all synchronized environments into one immutable GPU template.
    pub fn snapshot_gpu_resident(&self) -> Result<Arc<GpuSimulationSnapshot>, TesseraError> {
        let inner = locked(&self.inner)?;
        let (_, session) = inner.resident.as_ref().ok_or_else(|| {
            failed("initialize resident GPU stepping before capturing a template")
        })?;
        Ok(Arc::new(GpuSimulationSnapshot {
            scenes: session
                .snapshot_reset_templates(inner.batch.environments())
                .map_err(failed)?,
            controls: inner
                .actuators
                .iter()
                .map(|actuator| actuator.controls().to_vec())
                .collect(),
        }))
    }

    /// Restore only listed original slots, preserving all unselected environments.
    pub fn restore_gpu_resident(
        &self,
        snapshot: Arc<GpuSimulationSnapshot>,
        environments: Vec<u32>,
    ) -> Result<(), TesseraError> {
        let mut inner = locked(&self.inner)?;
        if snapshot.controls.len() != inner.actuators.len() {
            return Err(failed("GPU template environment count does not match"));
        }
        let mut restored = inner.actuators.clone();
        let selections = environments
            .into_iter()
            .map(|index| {
                let index = usize::try_from(index).map_err(failed)?;
                let actuator = restored
                    .get_mut(index)
                    .ok_or_else(|| failed("environment index out of range"))?;
                actuator
                    .set_controls(&snapshot.controls[index])
                    .map_err(failed)?;
                Ok(index)
            })
            .collect::<Result<Vec<_>, TesseraError>>()?;
        let ArticulatedBatchInner {
            batch,
            resident,
            actuators,
            ..
        } = &mut *inner;
        let (_, session) = resident
            .as_mut()
            .ok_or_else(|| failed("resident GPU owner no longer exists"))?;
        session
            .reset_envs_from_templates(batch.environments_mut(), &snapshot.scenes, &selections)
            .map_err(failed)?;
        *actuators = restored;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const MODEL: &str = r#"<mujoco><option gravity="0 0 0"/><worldbody>
      <body><joint name="slide" type="slide" axis="1 0 0"/>
      <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/></body>
      </worldbody><actuator><motor joint="slide"/></actuator></mujoco>"#;

    #[test]
    fn gpu_simulation_reset_restores_selected_state_and_held_controls() -> Result<(), TesseraError>
    {
        let model = BatchModel {
            format: ModelFormat::Mjcf,
            xml: MODEL.into(),
            floating_base: false,
            meshes: vec![],
            urdf_options: None,
        };
        let batch = ArticulatedBatch::new(vec![model.clone(), model])?;
        batch.set_controls(0, vec![1.0])?;
        batch.set_controls(1, vec![-1.0])?;
        batch.step_controls_gpu_resident(0.001, 1)?;
        let template = batch.snapshot_gpu_resident()?;
        let saved = batch.simulation_state(0)?.configuration.positions;
        batch.set_controls(0, vec![2.0])?;
        batch.set_controls(1, vec![-2.0])?;
        batch.step_controls_gpu_resident(0.001, 4)?;
        let untouched = batch.simulation_state(1)?.configuration.positions;
        assert!(
            batch
                .restore_gpu_resident(template.clone(), vec![0, 0])
                .is_err()
        );
        batch.restore_gpu_resident(template, vec![0])?;
        assert_eq!(batch.simulation_state(0)?.configuration.positions, saved);
        assert_eq!(
            batch.simulation_state(1)?.configuration.positions,
            untouched
        );
        assert_eq!(batch.controls(0)?, vec![1.0]);
        assert_eq!(batch.controls(1)?, vec![-2.0]);
        batch.step_controls_gpu_resident(0.001, 1)?;
        assert!(batch.simulation_state(0)?.configuration.positions[0] > saved[0]);

        let robot = ArticulatedWorld::from_mjcf(MODEL.into())?;
        robot.set_controls(vec![1.0])?;
        robot.step_controls_gpu_resident(0.001, 1)?;
        let template = robot.snapshot_gpu_resident()?;
        let saved = robot.positions()?;
        robot.set_controls(vec![2.0])?;
        robot.step_controls_gpu_resident(0.001, 3)?;
        robot.restore_gpu_resident(template)?;
        assert_eq!(robot.positions()?, saved);
        assert_eq!(robot.controls()?, vec![1.0]);
        robot.step_controls_gpu_resident(0.001, 1)?;
        Ok(())
    }
}
