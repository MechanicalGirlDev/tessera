//! Shared CPU and resident articulated constraint configuration.

use super::*;
use tessera_physics::prescribed_dof::{
    ArticulatedConstraintPolicy as CoreConstraintPolicy, ConstraintCompliance as CoreCompliance,
};

/// Native ERP/CFM row compliance.
#[derive(Clone, Copy, Debug, uniffi::Record)]
pub struct ConstraintCompliance {
    /// Position error reduction per substep, between zero and one.
    pub erp: f64,
    /// Nonnegative force mixing; zero selects a hard constraint.
    pub cfm: f64,
}

impl From<ConstraintCompliance> for CoreCompliance {
    fn from(value: ConstraintCompliance) -> Self {
        Self {
            erp: value.erp,
            cfm: value.cfm,
        }
    }
}

impl From<CoreCompliance> for ConstraintCompliance {
    fn from(value: CoreCompliance) -> Self {
        Self {
            erp: value.erp,
            cfm: value.cfm,
        }
    }
}

/// Independent dynamic/fixed contact and coupling/loop compliance.
#[derive(Clone, Copy, Debug, uniffi::Record)]
pub struct ConstraintPolicy {
    /// Contact between dynamically responding bodies.
    pub dynamic_contact: ConstraintCompliance,
    /// Ground, fixed-link, and prescribed-body contact.
    pub fixed_contact: ConstraintCompliance,
    /// Linear and polynomial coordinate equalities.
    pub coupling: ConstraintCompliance,
    /// Point-loop constraints.
    pub point_loop: ConstraintCompliance,
    /// Fixed-frame linear/angular loop constraints.
    pub fixed_loop: ConstraintCompliance,
}

impl From<ConstraintPolicy> for CoreConstraintPolicy {
    fn from(value: ConstraintPolicy) -> Self {
        Self {
            dynamic_contact: value.dynamic_contact.into(),
            fixed_contact: value.fixed_contact.into(),
            coupling: value.coupling.into(),
            point_loop: value.point_loop.into(),
            fixed_loop: value.fixed_loop.into(),
        }
    }
}

impl From<CoreConstraintPolicy> for ConstraintPolicy {
    fn from(value: CoreConstraintPolicy) -> Self {
        Self {
            dynamic_contact: value.dynamic_contact.into(),
            fixed_contact: value.fixed_contact.into(),
            coupling: value.coupling.into(),
            point_loop: value.point_loop.into(),
            fixed_loop: value.fixed_loop.into(),
        }
    }
}

#[uniffi::export]
impl ArticulatedWorld {
    /// Constant-velocity prescribed slots, including a floating-root prefix.
    pub fn prescribed_coordinates(&self) -> Result<Vec<bool>, TesseraError> {
        Ok(locked(&self.inner)?.world.prescribed_coordinates().to_vec())
    }

    /// Select coordinates whose velocities are retained under forces and impulses.
    pub fn set_prescribed_coordinates(&self, selected: Vec<bool>) -> Result<(), TesseraError> {
        let mut inner = locked(&self.inner)?;
        inner
            .world
            .set_prescribed_coordinates(&selected)
            .map_err(failed)?;
        inner.resident = None;
        Ok(())
    }

    /// Current native constraint row policy.
    pub fn constraint_policy(&self) -> Result<ConstraintPolicy, TesseraError> {
        Ok(locked(&self.inner)?.world.constraint_policy().into())
    }

    /// Atomically replace independent contact and loop compliance.
    pub fn set_constraint_policy(&self, policy: ConstraintPolicy) -> Result<(), TesseraError> {
        let mut inner = locked(&self.inner)?;
        inner
            .world
            .set_constraint_policy(policy.into())
            .map_err(failed)?;
        inner.resident = None;
        Ok(())
    }

    /// Stop compliance in reduced joint order, excluding root slots.
    pub fn joint_limit_compliance(&self) -> Result<Vec<ConstraintCompliance>, TesseraError> {
        Ok(locked(&self.inner)?
            .world
            .joint_limit_compliance()
            .iter()
            .copied()
            .map(Into::into)
            .collect())
    }

    /// Atomically configure scalar stops; unbounded slots require zero CFM.
    pub fn set_joint_limit_compliance(
        &self,
        values: Vec<ConstraintCompliance>,
    ) -> Result<(), TesseraError> {
        let native = values.into_iter().map(Into::into).collect::<Vec<_>>();
        let mut inner = locked(&self.inner)?;
        inner
            .world
            .set_joint_limit_compliance(&native)
            .map_err(failed)?;
        inner.resident = None;
        Ok(())
    }

    /// Whether joint-adjacent links may contact, preserving explicit exclusions.
    pub fn joint_contacts_enabled(&self) -> Result<bool, TesseraError> {
        Ok(locked(&self.inner)?.world.joint_contacts_enabled())
    }

    /// Enable joint-adjacent contacts and rebuild resident pair topology.
    pub fn set_joint_contacts_enabled(&self, enabled: bool) -> Result<(), TesseraError> {
        let mut inner = locked(&self.inner)?;
        inner.world.set_joint_contacts_enabled(enabled);
        inner.resident = None;
        Ok(())
    }
}

#[uniffi::export]
impl ArticulatedBatch {
    /// Prescribed root/joint slots in one environment.
    pub fn prescribed_coordinates(&self, index: u32) -> Result<Vec<bool>, TesseraError> {
        let id = self.id(index)?;
        Ok(locked(&self.inner)?
            .batch
            .environment(id)
            .ok_or_else(|| failed("environment not found"))?
            .prescribed_coordinates()
            .to_vec())
    }

    /// Atomically configure prescribed motion in one environment.
    pub fn set_prescribed_coordinates(
        &self,
        index: u32,
        selected: Vec<bool>,
    ) -> Result<(), TesseraError> {
        let id = self.id(index)?;
        let mut inner = locked(&self.inner)?;
        inner
            .batch
            .environment_mut(id)
            .ok_or_else(|| failed("environment not found"))?
            .set_prescribed_coordinates(&selected)
            .map_err(failed)?;
        inner.resident = None;
        Ok(())
    }

    /// Configure the global contact/loop policy across all environments.
    pub fn set_constraint_policy(&self, policy: ConstraintPolicy) -> Result<(), TesseraError> {
        let native: CoreConstraintPolicy = policy.into();
        if !native.is_valid() {
            return Err(failed("invalid constraint compliance"));
        }
        let mut inner = locked(&self.inner)?;
        for &id in &self.ids {
            inner
                .batch
                .environment_mut(id)
                .ok_or_else(|| failed("environment not found"))?
                .set_constraint_policy(native)
                .map_err(failed)?;
        }
        inner.resident = None;
        Ok(())
    }

    /// Stop compliance for one environment's reduced joints.
    pub fn joint_limit_compliance(
        &self,
        index: u32,
    ) -> Result<Vec<ConstraintCompliance>, TesseraError> {
        let id = self.id(index)?;
        Ok(locked(&self.inner)?
            .batch
            .environment(id)
            .ok_or_else(|| failed("environment not found"))?
            .joint_limit_compliance()
            .iter()
            .copied()
            .map(Into::into)
            .collect())
    }

    /// Atomically configure one environment's scalar stops.
    pub fn set_joint_limit_compliance(
        &self,
        index: u32,
        values: Vec<ConstraintCompliance>,
    ) -> Result<(), TesseraError> {
        let id = self.id(index)?;
        let native = values.into_iter().map(Into::into).collect::<Vec<_>>();
        let mut inner = locked(&self.inner)?;
        inner
            .batch
            .environment_mut(id)
            .ok_or_else(|| failed("environment not found"))?
            .set_joint_limit_compliance(&native)
            .map_err(failed)?;
        inner.resident = None;
        Ok(())
    }

    /// Enable joint-adjacent contacts for one environment.
    pub fn set_joint_contacts_enabled(
        &self,
        index: u32,
        enabled: bool,
    ) -> Result<(), TesseraError> {
        let id = self.id(index)?;
        let mut inner = locked(&self.inner)?;
        inner
            .batch
            .environment_mut(id)
            .ok_or_else(|| failed("environment not found"))?
            .set_joint_contacts_enabled(enabled);
        inner.resident = None;
        Ok(())
    }
}
