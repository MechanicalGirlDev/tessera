//! Held imported controls with feedback evaluated in native resident microsteps.

use super::*;

pub(super) fn metadata(actuators: &MjcfActuators) -> Result<Vec<MjcfActuatorInfo>, TesseraError> {
    actuators
        .entries()
        .iter()
        .map(|a| {
            let dynamics = match a.kind {
                CoreMjcfActuatorKind::Unsupported => MjcfActuatorDynamics::Unsupported,
                CoreMjcfActuatorKind::Motor => MjcfActuatorDynamics::Motor,
                CoreMjcfActuatorKind::Position { kp, kv } => {
                    MjcfActuatorDynamics::Position { kp, kv }
                }
                CoreMjcfActuatorKind::Velocity { kv } => MjcfActuatorDynamics::Velocity { kv },
                CoreMjcfActuatorKind::Damper { gain } => MjcfActuatorDynamics::Damper { gain },
                CoreMjcfActuatorKind::General { gain, bias, affine } => {
                    MjcfActuatorDynamics::General {
                        gain,
                        bias: bias.to_vec(),
                        affine,
                    }
                }
            };
            Ok(MjcfActuatorInfo {
                name: a.name.clone(),
                joint: a.joint.clone(),
                coordinate: a
                    .coordinate
                    .map(u32::try_from)
                    .transpose()
                    .map_err(failed)?,
                dynamics,
                source_kind: a.source_kind.clone(),
                attributes: a
                    .attributes
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect(),
                unsupported_reason: a.unsupported_reason.clone(),
                gear: a.gear,
                control_range: a
                    .control_range
                    .map_or_else(Vec::new, |range| range.to_vec()),
                force_range: a.force_range.map_or_else(Vec::new, |range| range.to_vec()),
            })
        })
        .collect()
}

#[uniffi::export]
impl ArticulatedWorld {
    /// Advance resident native dynamics using current imported actuator controls.
    ///
    /// Feedback reads device q/dq at every microstep; no host torque conversion is
    /// performed. Consecutive calls with the same timestep reuse resident state.
    pub fn step_controls_gpu_resident(
        &self,
        timestep: f64,
        steps: u32,
    ) -> Result<(), TesseraError> {
        if !timestep.is_finite() || timestep <= 0.0 || steps == 0 {
            return Err(failed("resident timestep and step count must be positive"));
        }
        let mut inner = locked(&self.inner)?;
        if inner.gpu.is_none() {
            inner.gpu = Some(GpuContactDevice::new().map_err(failed)?);
        }
        let ArticulatedInner {
            world,
            actuators,
            gpu,
            resident,
            resident_controls,
        } = &mut *inner;
        if !*resident_controls || !resident.as_ref().is_some_and(|(dt, _)| *dt == timestep) {
            let context = gpu.as_ref().ok_or_else(|| failed("GPU unavailable"))?;
            let mut session = GpuSceneDynamics::new(world, context, timestep).map_err(failed)?;
            session.enable_mjcf_actuators(actuators).map_err(failed)?;
            *resident = Some((timestep, session));
            *resident_controls = true;
        }
        let (_, session) = resident
            .as_mut()
            .ok_or_else(|| failed("resident GPU unavailable"))?;
        session
            .update_mjcf_controls(actuators.controls())
            .map_err(failed)?;
        let _diagnostics = session
            .step(world, &vec![0.0; world.articulation.dof()], steps as usize)
            .map_err(failed)?;
        Ok(())
    }
}

#[uniffi::export]
impl ArticulatedBatch {
    /// Imported actuator definitions, including unsupported entries, in XML order.
    pub fn actuator_info(&self, index: u32) -> Result<Vec<MjcfActuatorInfo>, TesseraError> {
        let _id = self.id(index)?;
        metadata(&locked(&self.inner)?.actuators[index as usize])
    }

    /// Imported actuator names in declaration order for one environment.
    pub fn actuator_names(&self, index: u32) -> Result<Vec<String>, TesseraError> {
        let _id = self.id(index)?;
        Ok(locked(&self.inner)?.actuators[index as usize]
            .entries()
            .iter()
            .map(|entry| entry.name.clone())
            .collect())
    }

    /// Native CPU feedback using held actuator controls in every environment.
    pub fn step_controls(&self, dt: f64) -> Result<(), TesseraError> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(failed("timestep must be positive and finite"));
        }
        let mut inner = locked(&self.inner)?;
        let ArticulatedBatchInner {
            batch,
            actuators,
            resident,
            ..
        } = &mut *inner;
        for (world, controls) in batch.environments().iter().zip(actuators.iter()) {
            let _efforts = controls.efforts(world).map_err(failed)?;
        }
        *resident = None;
        for (world, controls) in batch.environments_mut().iter_mut().zip(actuators.iter()) {
            controls.step(world, dt).map_err(failed)?;
        }
        Ok(())
    }

    /// Advance a heterogeneous resident batch using imported controls on the GPU.
    pub fn step_controls_gpu_resident(
        &self,
        timestep: f64,
        steps: u32,
    ) -> Result<(), TesseraError> {
        if !timestep.is_finite() || timestep <= 0.0 || steps == 0 {
            return Err(failed("resident timestep and step count must be positive"));
        }
        let mut inner = locked(&self.inner)?;
        if inner.gpu.is_none() {
            inner.gpu = Some(GpuContactDevice::new().map_err(failed)?);
        }
        let ArticulatedBatchInner {
            batch,
            actuators,
            gpu,
            resident,
            resident_controls,
            ..
        } = &mut *inner;
        if !*resident_controls || !resident.as_ref().is_some_and(|(dt, _)| *dt == timestep) {
            let context = gpu.as_ref().ok_or_else(|| failed("GPU unavailable"))?;
            let mut session = GpuSceneDynamicsBatch::new(batch.environments(), context, timestep)
                .map_err(failed)?;
            session.enable_mjcf_actuators(actuators).map_err(failed)?;
            *resident = Some((timestep, session));
            *resident_controls = true;
        }
        let (_, session) = resident
            .as_mut()
            .ok_or_else(|| failed("resident GPU unavailable"))?;
        for (index, controls) in actuators.iter().enumerate() {
            session
                .update_mjcf_controls(index, controls.controls())
                .map_err(failed)?;
        }
        let efforts = batch
            .environments()
            .iter()
            .map(|world| vec![0.0; world.articulation.dof()])
            .collect::<Vec<_>>();
        let refs = efforts.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let _diagnostics = session
            .step(batch.environments_mut(), &refs, steps as usize)
            .map_err(failed)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SERVO: &str = r#"<mujoco><option gravity="0 0 0"/>
      <worldbody><body name="tip"><joint name="hinge"/>
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
      </body></worldbody>
      <actuator><position name="servo" joint="hinge" kp="10" kv="1"/></actuator>
      </mujoco>"#;

    #[test]
    fn actuator_feedback_resident_microsteps_match_native_cpu_feedback() -> Result<(), TesseraError>
    {
        let gpu = ArticulatedWorld::from_mjcf(SERVO.into())?;
        let cpu = ArticulatedWorld::from_mjcf(SERVO.into())?;
        gpu.set_controls(vec![1.0])?;
        cpu.set_controls(vec![1.0])?;
        for _ in 0..50 {
            cpu.step_controls(0.001)?;
        }

        gpu.step_controls_gpu_resident(0.001, 50)?;

        assert!((gpu.positions()?[0] - cpu.positions()?[0]).abs() < 1e-5);
        assert!((gpu.velocities()?[0] - cpu.velocities()?[0]).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn actuator_feedback_batch_keeps_distinct_controls_and_reuses_resident_state()
    -> Result<(), TesseraError> {
        let model = BatchModel {
            format: ModelFormat::Mjcf,
            xml: SERVO.into(),
            floating_base: false,
            meshes: Vec::new(),
            urdf_options: None,
        };
        let batch = ArticulatedBatch::new(vec![model.clone(), model])?;
        batch.set_controls(0, vec![1.0])?;
        batch.set_controls(1, vec![-1.0])?;
        batch.step_controls_gpu_resident(0.001, 25)?;
        let before = batch.simulation_state(0)?.configuration.positions[0];
        assert!(before > 0.0);
        assert!(batch.simulation_state(1)?.configuration.positions[0] < 0.0);
        batch.set_controls(0, vec![0.5])?;

        batch.step_controls_gpu_resident(0.001, 25)?;

        assert!(batch.simulation_state(0)?.configuration.positions[0] > before);
        assert_eq!(batch.controls(0)?, vec![0.5]);
        assert_eq!(batch.controls(1)?, vec![-1.0]);
        Ok(())
    }

    #[test]
    fn actuator_feedback_preserved_unsupported_entry_stays_undriven() -> Result<(), TesseraError> {
        let xml = SERVO.replace(
            r#"<position name="servo" joint="hinge" kp="10" kv="1"/>"#,
            r#"<intvelocity name="inactive" joint="hinge"/>"#,
        );
        let robot = ArticulatedWorld::from_mjcf(xml)?;
        robot.set_controls(vec![2.0])?;
        let info = robot.actuator_info()?;
        assert!(matches!(
            info[0].dynamics,
            MjcfActuatorDynamics::Unsupported
        ));
        assert!(info[0].unsupported_reason.is_some());

        robot.step_controls_gpu_resident(0.001, 20)?;

        assert_eq!(robot.positions()?, vec![0.0]);
        assert_eq!(robot.velocities()?, vec![0.0]);
        Ok(())
    }
}
