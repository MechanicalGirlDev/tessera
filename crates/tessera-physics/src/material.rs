//! Contact material coefficients and deterministic combination rules.

/// Rule used to combine one coefficient from two colliders.
///
/// When colliders request different rules, the rule with the higher priority
/// is selected in the order `Average < Min < Multiply < Max`. This matches the
/// contact material contract exposed by Nexus and Rapier.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum CoefficientCombineRule {
    /// Arithmetic mean of both coefficients.
    #[default]
    Average = 0,
    /// Smaller coefficient.
    Min = 1,
    /// Product of both coefficients.
    Multiply = 2,
    /// Larger coefficient.
    Max = 3,
}

impl CoefficientCombineRule {
    /// Combine two coefficients with this rule.
    pub fn combine(self, left: f64, right: f64) -> f64 {
        match self {
            Self::Average => (left + right) * 0.5,
            Self::Min => left.min(right),
            Self::Multiply => left * right,
            Self::Max => left.max(right),
        }
    }

    fn dominant(left: Self, right: Self) -> Self {
        if left as u8 >= right as u8 {
            left
        } else {
            right
        }
    }
}

/// Friction and restitution owned by one collider.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColliderMaterial {
    /// Coulomb sliding-friction coefficient.
    pub friction: f64,
    /// Non-negative normal coefficient of restitution.
    pub restitution: f64,
    /// Rule requested for friction contacts.
    pub friction_combine_rule: CoefficientCombineRule,
    /// Rule requested for restitution contacts.
    pub restitution_combine_rule: CoefficientCombineRule,
}

impl Default for ColliderMaterial {
    fn default() -> Self {
        Self {
            friction: 0.5,
            restitution: 0.0,
            friction_combine_rule: CoefficientCombineRule::Average,
            restitution_combine_rule: CoefficientCombineRule::Average,
        }
    }
}

impl ColliderMaterial {
    /// Construct a material using average combination rules.
    pub fn new(friction: f64, restitution: f64) -> Self {
        Self {
            friction,
            restitution,
            ..Self::default()
        }
    }

    /// Whether both coefficients can be consumed by the contact solver.
    pub fn is_valid(self) -> bool {
        self.friction.is_finite()
            && self.friction >= 0.0
            && self.restitution.is_finite()
            && self.restitution >= 0.0
    }

    /// Resolve this material against another collider's material.
    pub fn combine(self, other: Self) -> CombinedMaterial {
        let friction_rule = CoefficientCombineRule::dominant(
            self.friction_combine_rule,
            other.friction_combine_rule,
        );
        let restitution_rule = CoefficientCombineRule::dominant(
            self.restitution_combine_rule,
            other.restitution_combine_rule,
        );
        CombinedMaterial {
            friction: friction_rule.combine(self.friction, other.friction),
            restitution: restitution_rule.combine(self.restitution, other.restitution),
        }
    }
}

/// Effective coefficients for one contact pair.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CombinedMaterial {
    /// Effective Coulomb friction.
    pub friction: f64,
    /// Effective restitution.
    pub restitution: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rules_compute_each_supported_combination() {
        assert_eq!(CoefficientCombineRule::Average.combine(0.2, 0.8), 0.5);
        assert_eq!(CoefficientCombineRule::Min.combine(0.2, 0.8), 0.2);
        assert!((CoefficientCombineRule::Multiply.combine(0.2, 0.8) - 0.16).abs() < 1e-12);
        assert_eq!(CoefficientCombineRule::Max.combine(0.2, 0.8), 0.8);
    }

    #[test]
    fn higher_priority_rule_wins_for_each_coefficient() {
        let left = ColliderMaterial {
            friction: 0.2,
            restitution: 0.3,
            friction_combine_rule: CoefficientCombineRule::Min,
            restitution_combine_rule: CoefficientCombineRule::Max,
        };
        let right = ColliderMaterial {
            friction: 0.8,
            restitution: 0.7,
            friction_combine_rule: CoefficientCombineRule::Multiply,
            restitution_combine_rule: CoefficientCombineRule::Average,
        };
        let combined = left.combine(right);
        assert!((combined.friction - 0.16).abs() < 1e-12);
        assert_eq!(combined.restitution, 0.7);
    }

    #[test]
    fn validation_rejects_nonfinite_and_negative_coefficients() {
        assert!(ColliderMaterial::default().is_valid());
        assert!(ColliderMaterial::new(0.5, 1.1).is_valid());
        assert!(!ColliderMaterial::new(-0.1, 0.0).is_valid());
        assert!(!ColliderMaterial::new(0.5, -0.1).is_valid());
        assert!(!ColliderMaterial::new(f64::NAN, 0.0).is_valid());
    }
}
