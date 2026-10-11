//! Configurable recovery and warmstarting for resident articulated contacts.

/// Native constraint error reduction and force mixing.
///
/// An impulse row solves `(J M^-1 J^T + cfm / dt) delta_lambda =
/// target_velocity - J v - cfm / dt * lambda`. Positive CFM permits a
/// load-dependent displacement; ERP alone only changes position recovery.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuConstraintCompliance {
    /// Error reduction per substep, in `[0, 1]`.
    pub erp: f32,
    /// Force mixing, nonnegative. Zero selects a hard row.
    pub cfm: f32,
}

impl Default for GpuConstraintCompliance {
    fn default() -> Self {
        Self { erp: 0.2, cfm: 0.0 }
    }
}

impl GpuConstraintCompliance {
    /// Validate both parameters without mutating any solver state.
    pub fn is_valid(self) -> bool {
        self.erp.is_finite()
            && (0.0..=1.0).contains(&self.erp)
            && self.cfm.is_finite()
            && self.cfm >= 0.0
    }
}

/// Native contact parameters shared by the environments of a resident batch.
///
/// These configure the native impulse solver, not Nexus-specific TGS frequency
/// or damping formulas. Updating a policy clears incompatible impulse history.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuContactPolicy {
    /// Legacy penetration recovery gain, in `[0, 1]`.
    /// Hard normal rows use `position_gain * normal.erp / 0.2`, so default
    /// normal ERP preserves this gain. Soft normal rows use `normal.erp` directly.
    /// Set this to `0.2` to use each normal family's ERP exactly.
    pub position_gain: f32,
    /// Maximum recovery separation speed, in metres per second.
    pub max_correction_speed: f32,
    /// Cached-impulse multiplier in `[0, 1]`, used when warmstarting is enabled.
    pub warm_start_coefficient: f32,
    /// Penetration ignored by position recovery, in metres.
    pub allowed_penetration: f32,
    /// Compliance for dynamic and self-contact normal rows.
    /// Recovery uses the hard/soft gain scaling documented on `position_gain`.
    pub dynamic_normal: GpuConstraintCompliance,
    /// Compliance for ground, fixed, and prescribed obstacle normal rows.
    pub static_normal: GpuConstraintCompliance,
    /// Compliance for point loop-closing rows.
    pub loop_constraint: GpuConstraintCompliance,
    /// Compliance for polynomial joint coupling rows.
    pub coupling_constraint: GpuConstraintCompliance,
    /// Compliance for fixed loop-closing rows (linear and angular).
    pub fixed_loop_constraint: GpuConstraintCompliance,
}

impl Default for GpuContactPolicy {
    fn default() -> Self {
        Self {
            position_gain: 0.2,
            max_correction_speed: 2.0,
            warm_start_coefficient: 0.8,
            allowed_penetration: 0.0,
            dynamic_normal: GpuConstraintCompliance::default(),
            static_normal: GpuConstraintCompliance::default(),
            loop_constraint: GpuConstraintCompliance::default(),
            coupling_constraint: GpuConstraintCompliance::default(),
            fixed_loop_constraint: GpuConstraintCompliance::default(),
        }
    }
}

impl GpuContactPolicy {
    pub(crate) fn packed(self) -> Option<[f32; 24]> {
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
            || !self.dynamic_normal.is_valid()
            || !self.static_normal.is_valid()
            || !self.loop_constraint.is_valid()
            || !self.coupling_constraint.is_valid()
            || !self.fixed_loop_constraint.is_valid()
        {
            return None;
        }
        Some([
            values[0],
            values[1],
            values[2],
            values[3],
            self.dynamic_normal.erp,
            self.dynamic_normal.cfm,
            0.0,
            0.0,
            self.static_normal.erp,
            self.static_normal.cfm,
            0.0,
            0.0,
            self.loop_constraint.erp,
            self.loop_constraint.cfm,
            0.0,
            0.0,
            self.coupling_constraint.erp,
            self.coupling_constraint.cfm,
            0.0,
            0.0,
            self.fixed_loop_constraint.erp,
            self.fixed_loop_constraint.cfm,
            0.0,
            0.0,
        ])
    }
}
