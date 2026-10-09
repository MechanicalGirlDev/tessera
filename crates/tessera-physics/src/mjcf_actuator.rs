use super::*;

/// Stateless actuator dynamics supported by the reference MJCF converter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MjcfActuatorKind {
    /// Constant generalized force multiplied by gear.
    Motor,
    /// Scalar coordinate servo; gear does not rescale the reference servo.
    Position {
        /// Position gain.
        kp: f64,
        /// Velocity damping.
        kv: f64,
    },
    /// Scalar velocity servo.
    Velocity {
        /// Velocity gain.
        kv: f64,
    },
    /// Zero-velocity damper, scaled by the absolute control.
    Damper {
        /// Damping gain.
        gain: f64,
    },
    /// Fixed-gain general actuator, optionally with restoring affine bias.
    General {
        /// Control gain.
        gain: f64,
        /// Constant, position and velocity bias coefficients.
        bias: [f64; 3],
        /// Whether the reference uses the restoring affine servo branch.
        affine: bool,
    },
}

/// One actuator resolved to a scalar native coordinate.
#[derive(Debug, Clone, PartialEq)]
pub struct MjcfActuatorInfo {
    /// Explicit or generated actuator name.
    pub name: String,
    /// Referenced MJCF joint name.
    pub joint: String,
    /// Generalized coordinate, not actuator-control index.
    pub coordinate: usize,
    /// Supported dynamics.
    pub kind: MjcfActuatorKind,
    /// Scalar motor transmission gear.
    pub gear: f64,
    /// Enabled control clamp.
    pub control_range: Option<[f64; 2]>,
    /// Enabled generalized-force clamp, applied after motor gear.
    pub force_range: Option<[f64; 2]>,
}

/// Imported actuator metadata and persistent controls in declaration order.
#[derive(Debug, Clone, Default)]
pub struct MjcfActuators {
    entries: Vec<MjcfActuatorInfo>,
    controls: Vec<f64>,
}

impl MjcfActuators {
    /// Stable XML actuator order.
    pub fn entries(&self) -> &[MjcfActuatorInfo] {
        &self.entries
    }

    /// Last accepted controls, before per-actuator clamping.
    pub fn controls(&self) -> &[f64] {
        &self.controls
    }

    /// Replace controls atomically; rejects incorrect lengths and nonfinite values.
    pub fn set_controls(&mut self, controls: &[f64]) -> Result<(), MjcfLoadError> {
        if controls.len() != self.entries.len() || controls.iter().any(|v| !v.is_finite()) {
            return Err(MjcfLoadError::Invalid(
                "controls must be finite and match the actuator count".into(),
            ));
        }
        self.controls.copy_from_slice(controls);
        Ok(())
    }

    /// Resolve held controls into coordinate-order efforts, summing shared joints.
    pub fn efforts(&self, world: &ArticulatedWorld) -> Result<Vec<f64>, MjcfLoadError> {
        let mut efforts = vec![0.0; world.positions.len()];
        for (a, &control) in self.entries.iter().zip(&self.controls) {
            let u = a
                .control_range
                .map_or(control, |r| control.clamp(r[0], r[1]));
            let q = *world
                .positions
                .as_slice()
                .get(a.coordinate)
                .ok_or_else(|| {
                    MjcfLoadError::Invalid(
                        "actuator coordinate is not present in this world".into(),
                    )
                })?;
            let v = *world
                .velocities
                .as_slice()
                .get(a.coordinate)
                .ok_or_else(|| {
                    MjcfLoadError::Invalid(
                        "actuator coordinate is not present in this world".into(),
                    )
                })?;
            let force = match a.kind {
                MjcfActuatorKind::Motor => u * a.gear,
                MjcfActuatorKind::Position { kp, kv } => kp * (u - q) - kv * v,
                MjcfActuatorKind::Velocity { kv } => kv * (u - v),
                MjcfActuatorKind::Damper { gain } => -gain * u.abs() * v,
                MjcfActuatorKind::General { gain, bias, affine } => {
                    if affine {
                        gain * u + bias[0] + bias[1] * q + bias[2] * v
                    } else {
                        gain * u * a.gear
                    }
                }
            };
            if !force.is_finite() {
                return Err(MjcfLoadError::Invalid("nonfinite actuator force".into()));
            }
            efforts[a.coordinate] += a.force_range.map_or(force, |r| force.clamp(r[0], r[1]));
        }
        if efforts.iter().any(|v| !v.is_finite()) {
            return Err(MjcfLoadError::Invalid(
                "nonfinite summed actuator force".into(),
            ));
        }
        Ok(efforts)
    }

    /// Advance CPU dynamics, reevaluating held controls at each native substep.
    pub fn step(&self, world: &mut ArticulatedWorld, dt: f64) -> Result<(), MjcfLoadError> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(ArticulatedWorldError::InvalidInput.into());
        }
        let count = (dt / world.params().max_substep).ceil().max(1.0) as usize;
        let substep = dt / count as f64;
        for _ in 0..count {
            world.step(substep, &self.efforts(world)?)?;
        }
        Ok(())
    }
}

impl LoadedMjcf {
    /// Replace held controls in actuator declaration order.
    pub fn set_controls(&mut self, controls: &[f64]) -> Result<(), MjcfLoadError> {
        self.actuators.set_controls(controls)
    }

    /// Advance the imported world using held actuator controls.
    pub fn step(&mut self, dt: f64) -> Result<(), MjcfLoadError> {
        self.actuators.step(&mut self.world, dt)
    }
}

#[path = "mjcf_actuator_import.rs"]
mod parsing;
pub(super) use parsing::import;

#[cfg(test)]
#[path = "mjcf_actuator_tests.rs"]
mod tests;
