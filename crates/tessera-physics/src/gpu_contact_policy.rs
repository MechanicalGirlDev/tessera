//! Configurable recovery and warmstarting for resident articulated contacts.

use crate::gpu_articulated_ground_contact::GpuArticulatedGroundContactError;

/// Native contact parameters shared by the environments of a resident batch.
///
/// These configure the native impulse solver, not Nexus-specific TGS frequency
/// or damping formulas. Updating a policy does not clear existing contact history.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuContactPolicy {
    /// Fraction of penetration recovered per substep, in `[0, 1]`.
    pub position_gain: f32,
    /// Maximum recovery separation speed, in metres per second.
    pub max_correction_speed: f32,
    /// Cached-impulse multiplier in `[0, 1]`, used when warmstarting is enabled.
    pub warm_start_coefficient: f32,
    /// Penetration ignored by position recovery, in metres.
    pub allowed_penetration: f32,
}

impl Default for GpuContactPolicy {
    fn default() -> Self {
        Self {
            position_gain: 0.2,
            max_correction_speed: 2.0,
            warm_start_coefficient: 0.8,
            allowed_penetration: 0.0,
        }
    }
}

impl GpuContactPolicy {
    pub(crate) fn packed(self) -> Result<[f32; 4], GpuArticulatedGroundContactError> {
        let values = [
            self.position_gain,
            self.max_correction_speed,
            self.warm_start_coefficient,
            self.allowed_penetration,
        ];
        if values
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
            || self.position_gain > 1.0
            || self.warm_start_coefficient > 1.0
        {
            return Err(GpuArticulatedGroundContactError::InvalidInput);
        }
        Ok(values)
    }
}
