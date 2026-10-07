//! Constitutive and plasticity models used by the MPM reference pipeline.

use nalgebra::{Matrix3, Vector3};

/// Persistent per-particle plastic state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlasticState {
    /// Determinant of the plastic deformation gradient.
    pub plastic_det: f64,
    /// Accumulated Drucker-Prager plastic strain.
    pub hardening: f64,
    /// Accumulated logarithmic volume correction.
    pub log_volume_gain: f64,
}

impl Default for PlasticState {
    fn default() -> Self {
        Self {
            plastic_det: 1.0,
            hardening: 0.0,
            log_volume_gain: 0.0,
        }
    }
}

/// Three-dimensional MPM constitutive model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MaterialModel {
    /// Corotated linear elasticity.
    LinearElastic {
        /// Young's modulus in pascals.
        young_modulus: f64,
        /// Poisson ratio.
        poisson_ratio: f64,
    },
    /// Compressible Neo-Hookean elasticity.
    NeoHookean {
        /// Young's modulus in pascals.
        young_modulus: f64,
        /// Poisson ratio.
        poisson_ratio: f64,
    },
    /// Drucker-Prager granular material.
    Sand {
        /// Young's modulus in pascals.
        young_modulus: f64,
        /// Poisson ratio.
        poisson_ratio: f64,
        /// Internal friction angle in radians.
        friction_angle: f64,
        /// Tensile yield offset in logarithmic volumetric strain.
        cohesion: f64,
    },
    /// Weakly compressible Newtonian fluid using a Tait equation of state.
    Fluid {
        /// Bulk modulus in pascals.
        bulk_modulus: f64,
        /// Tait exponent.
        gamma: f64,
        /// Dynamic viscosity in pascal seconds.
        viscosity: f64,
        /// Tensile stiffness relative to the bulk modulus.
        tensile_stiffness: f64,
    },
    /// Snow with singular-value yield limits and compaction hardening.
    Snow {
        /// Young's modulus in pascals.
        young_modulus: f64,
        /// Poisson ratio.
        poisson_ratio: f64,
        /// Maximum elastic compression before plastic flow.
        critical_compression: f64,
        /// Maximum elastic stretch before plastic flow.
        critical_stretch: f64,
        /// Exponential compaction-hardening coefficient.
        hardening: f64,
    },
    /// Neo-Hookean sand with Nexus' hardening Drucker-Prager return mapping.
    SandNeoHookean {
        /// Young's modulus in pascals.
        young_modulus: f64,
        /// Poisson ratio.
        poisson_ratio: f64,
        /// Asymptotic friction angle (`ha`) in radians; Nexus uses 35 degrees.
        /// The current angle is `ha + (9 degrees * q - 10 degrees) * exp(-0.2 * q)`.
        friction_angle: f64,
        /// Tensile yield offset in logarithmic volumetric strain.
        cohesion: f64,
    },
}

impl Default for MaterialModel {
    fn default() -> Self {
        Self::LinearElastic {
            young_modulus: 1_000.0,
            poisson_ratio: 0.2,
        }
    }
}

impl MaterialModel {
    /// Construct a corotated linear elastic material.
    pub fn elastic(young_modulus: f64, poisson_ratio: f64) -> Self {
        Self::LinearElastic {
            young_modulus,
            poisson_ratio,
        }
    }

    /// Construct a compressible Neo-Hookean material.
    pub fn neo_hookean(young_modulus: f64, poisson_ratio: f64) -> Self {
        Self::NeoHookean {
            young_modulus,
            poisson_ratio,
        }
    }

    /// Construct dry or cohesive sand.
    pub fn sand(
        young_modulus: f64,
        poisson_ratio: f64,
        friction_angle: f64,
        cohesion: f64,
    ) -> Self {
        Self::Sand {
            young_modulus,
            poisson_ratio,
            friction_angle,
            cohesion,
        }
    }

    /// Construct Neo-Hookean sand with Nexus' Drucker-Prager hardening law.
    /// Use a 35-degree friction angle and zero cohesion for Nexus' dry sand defaults.
    pub fn sand_neo_hookean(
        young_modulus: f64,
        poisson_ratio: f64,
        friction_angle: f64,
        cohesion: f64,
    ) -> Self {
        Self::SandNeoHookean {
            young_modulus,
            poisson_ratio,
            friction_angle,
            cohesion,
        }
    }

    /// Construct a weakly compressible fluid.
    pub fn fluid(bulk_modulus: f64, gamma: f64, viscosity: f64) -> Self {
        Self::Fluid {
            bulk_modulus,
            gamma,
            viscosity,
            tensile_stiffness: 0.25,
        }
    }

    /// Construct snow with the Stomakhin yield box defaults.
    pub fn snow(young_modulus: f64, poisson_ratio: f64) -> Self {
        Self::Snow {
            young_modulus,
            poisson_ratio,
            critical_compression: 2.5e-2,
            critical_stretch: 7.5e-3,
            hardening: 10.0,
        }
    }

    /// Validate finite physical parameters.
    pub fn is_valid(self) -> bool {
        match self {
            Self::LinearElastic {
                young_modulus,
                poisson_ratio,
            }
            | Self::NeoHookean {
                young_modulus,
                poisson_ratio,
            } => valid_elasticity(young_modulus, poisson_ratio),
            Self::Sand {
                young_modulus,
                poisson_ratio,
                friction_angle,
                cohesion,
            }
            | Self::SandNeoHookean {
                young_modulus,
                poisson_ratio,
                friction_angle,
                cohesion,
            } => {
                valid_elasticity(young_modulus, poisson_ratio)
                    && friction_angle.is_finite()
                    && (0.0..core::f64::consts::FRAC_PI_2).contains(&friction_angle)
                    && cohesion.is_finite()
                    && cohesion >= 0.0
            }
            Self::Fluid {
                bulk_modulus,
                gamma,
                viscosity,
                tensile_stiffness,
            } => {
                bulk_modulus.is_finite()
                    && bulk_modulus > 0.0
                    && gamma.is_finite()
                    && gamma > 0.0
                    && viscosity.is_finite()
                    && viscosity >= 0.0
                    && tensile_stiffness.is_finite()
                    && tensile_stiffness >= 0.0
            }
            Self::Snow {
                young_modulus,
                poisson_ratio,
                critical_compression,
                critical_stretch,
                hardening,
            } => {
                valid_elasticity(young_modulus, poisson_ratio)
                    && critical_compression.is_finite()
                    && (0.0..1.0).contains(&critical_compression)
                    && critical_stretch.is_finite()
                    && critical_stretch >= 0.0
                    && hardening.is_finite()
                    && hardening >= 0.0
            }
        }
    }

    /// Kirchhoff stress for the current elastic deformation and velocity gradient.
    pub fn kirchhoff_stress(
        self,
        deformation: Matrix3<f64>,
        velocity_gradient: Matrix3<f64>,
        plastic: PlasticState,
    ) -> Matrix3<f64> {
        match self {
            Self::LinearElastic {
                young_modulus,
                poisson_ratio,
            } => corotated_stress(deformation, young_modulus, poisson_ratio, 1.0),
            Self::NeoHookean {
                young_modulus,
                poisson_ratio,
            }
            | Self::SandNeoHookean {
                young_modulus,
                poisson_ratio,
                ..
            } => neo_hookean_stress(deformation, young_modulus, poisson_ratio, 1.0),
            Self::Sand {
                young_modulus,
                poisson_ratio,
                ..
            } => corotated_stress(deformation, young_modulus, poisson_ratio, 1.0),
            Self::Fluid {
                bulk_modulus,
                gamma,
                viscosity,
                tensile_stiffness,
            } => fluid_stress(
                deformation,
                velocity_gradient,
                bulk_modulus,
                gamma,
                viscosity,
                tensile_stiffness,
            ),
            Self::Snow {
                young_modulus,
                poisson_ratio,
                hardening,
                ..
            } => {
                let factor = (hardening * (1.0 - plastic.plastic_det))
                    .exp()
                    .clamp(0.01, 100.0);
                corotated_stress(deformation, young_modulus, poisson_ratio, factor)
            }
        }
    }

    /// Apply the material's return mapping after deformation integration.
    pub fn project_deformation(
        self,
        deformation: Matrix3<f64>,
        mut plastic: PlasticState,
    ) -> (Matrix3<f64>, PlasticState) {
        match self {
            Self::Fluid { .. } => {
                let j = deformation.determinant().max(1e-9);
                (Matrix3::identity() * j.cbrt(), plastic)
            }
            Self::Snow {
                critical_compression,
                critical_stretch,
                ..
            } => {
                let Some((u, singular, v_t)) = svd_parts(deformation) else {
                    return (deformation, plastic);
                };
                let clamped = singular
                    .map(|value| value.clamp(1.0 - critical_compression, 1.0 + critical_stretch));
                let old_det = singular.iter().product::<f64>();
                let new_det = clamped.iter().product::<f64>().max(1e-12);
                plastic.plastic_det = (plastic.plastic_det * old_det / new_det).clamp(0.1, 4.0);
                (u * Matrix3::from_diagonal(&clamped) * v_t, plastic)
            }
            Self::Sand {
                young_modulus,
                poisson_ratio,
                friction_angle,
                cohesion,
            } => project_sand(
                deformation,
                plastic,
                young_modulus,
                poisson_ratio,
                friction_angle,
                cohesion,
            ),
            Self::SandNeoHookean {
                young_modulus,
                poisson_ratio,
                friction_angle,
                cohesion,
            } => {
                let (lambda, mu) = lame(young_modulus, poisson_ratio);
                if lambda == 0.0 {
                    return (deformation, plastic);
                }
                let Some((u, singular, v_t)) = svd_parts(deformation) else {
                    return (deformation, plastic);
                };
                let strain = singular.map(f64::ln) + Vector3::repeat(plastic.log_volume_gain / 3.0);
                let trace = strain.sum();
                let deviatoric = strain - Vector3::repeat(trace / 3.0);
                let norm = deviatoric.norm();
                let shifted_trace = trace - cohesion;
                let q = plastic.hardening;
                let angle = friction_angle
                    + (9.0f64.to_radians() * q - 10.0f64.to_radians()) * (-0.2 * q).exp();
                let sine = angle.sin();
                let alpha = (2.0 / 3.0f64).sqrt() * (2.0 * sine) / (3.0 - sine);
                // A rotation/isotropic stretch must select the same apex branch
                // after f32 GPU upload; exact zero is not stable under SVD roundoff.
                let isotropic = norm <= 4.0 * f64::from(f32::EPSILON);
                let (projected_log, increment) = if shifted_trace > 0.0 || isotropic {
                    (Vector3::repeat(cohesion / 3.0), strain.norm())
                } else {
                    let gamma =
                        norm + (3.0 * lambda + 2.0 * mu) / (2.0 * mu) * shifted_trace * alpha;
                    if gamma <= 0.0 {
                        return (deformation, plastic);
                    }
                    (strain - deviatoric * (gamma / norm), gamma)
                };
                let projected = projected_log.map(f64::exp);
                let old_det = singular.iter().product::<f64>();
                let new_det = projected.iter().product::<f64>();
                plastic.plastic_det *= old_det / new_det;
                plastic.log_volume_gain += old_det.ln() - new_det.ln();
                plastic.hardening += increment;
                (u * Matrix3::from_diagonal(&projected) * v_t, plastic)
            }
            Self::LinearElastic { .. } | Self::NeoHookean { .. } => (deformation, plastic),
        }
    }

    /// Conservative CFL timestep bound for one particle.
    pub fn timestep_bound(self, density: f64, velocity: Vector3<f64>, cell_width: f64) -> f64 {
        self.timestep_bound_with_deformation(density, velocity, cell_width, 1.0)
    }

    /// CFL bound including elastic volume for Nexus-compatible Neo-Hookean sand.
    /// Existing materials retain their conservative reference-density bound.
    pub fn timestep_bound_with_deformation(
        self,
        density: f64,
        velocity: Vector3<f64>,
        cell_width: f64,
        deformation_det: f64,
    ) -> f64 {
        let wave_speed = match self {
            Self::SandNeoHookean {
                young_modulus,
                poisson_ratio,
                ..
            } => {
                let (lambda, mu) = lame(young_modulus, poisson_ratio);
                let density = density / deformation_det.max(1e-6);
                let wave_speed = ((lambda + 2.0 * mu) / density).sqrt();
                return 0.5 * cell_width / wave_speed.max(velocity.norm());
            }
            Self::Fluid {
                bulk_modulus,
                gamma,
                ..
            } => (bulk_modulus * gamma / density.max(1e-12)).sqrt(),
            Self::LinearElastic {
                young_modulus,
                poisson_ratio,
            }
            | Self::NeoHookean {
                young_modulus,
                poisson_ratio,
            }
            | Self::Sand {
                young_modulus,
                poisson_ratio,
                ..
            }
            | Self::Snow {
                young_modulus,
                poisson_ratio,
                ..
            } => {
                let (lambda, mu) = lame(young_modulus, poisson_ratio);
                ((lambda + 2.0 * mu) / density.max(1e-12)).sqrt()
            }
        };
        0.4 * cell_width / (wave_speed + velocity.norm()).max(1e-12)
    }
}

fn valid_elasticity(young: f64, poisson: f64) -> bool {
    young.is_finite() && young > 0.0 && poisson.is_finite() && (-1.0..0.5).contains(&poisson)
}

fn lame(young: f64, poisson: f64) -> (f64, f64) {
    (
        young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson)),
        young / (2.0 * (1.0 + poisson)),
    )
}

fn corotated_stress(
    deformation: Matrix3<f64>,
    young: f64,
    poisson: f64,
    factor: f64,
) -> Matrix3<f64> {
    let (lambda, mu) = lame(young, poisson);
    let Some((u, _, v_t)) = svd_parts(deformation) else {
        return Matrix3::zeros();
    };
    let rotation = u * v_t;
    let j = deformation.determinant();
    let mut stress = (deformation - rotation) * deformation.transpose() * (2.0 * mu * factor);
    stress += Matrix3::identity() * (lambda * factor * (j - 1.0) * j);
    stress
}

fn neo_hookean_stress(
    deformation: Matrix3<f64>,
    young: f64,
    poisson: f64,
    factor: f64,
) -> Matrix3<f64> {
    let (lambda, mu) = lame(young, poisson);
    let j = deformation.determinant().max(1e-10);
    deformation * deformation.transpose() * (mu * factor)
        + Matrix3::identity() * (factor * (lambda * j.ln() - mu))
}

fn fluid_stress(
    deformation: Matrix3<f64>,
    velocity_gradient: Matrix3<f64>,
    bulk: f64,
    gamma: f64,
    viscosity: f64,
    tensile: f64,
) -> Matrix3<f64> {
    let j = deformation.determinant().max(1e-6);
    let pressure = if j <= 1.0 {
        bulk * (j.powf(-gamma) - 1.0)
    } else {
        -bulk * tensile * (j - 1.0)
    };
    let strain_rate = (velocity_gradient + velocity_gradient.transpose()) * 0.5;
    let deviatoric = strain_rate - Matrix3::identity() * (strain_rate.trace() / 3.0);
    deviatoric * (2.0 * viscosity * j) - Matrix3::identity() * (pressure * j)
}

fn project_sand(
    deformation: Matrix3<f64>,
    mut plastic: PlasticState,
    young: f64,
    poisson: f64,
    friction_angle: f64,
    cohesion: f64,
) -> (Matrix3<f64>, PlasticState) {
    let Some((u, singular, v_t)) = svd_parts(deformation) else {
        return (deformation, plastic);
    };
    let strain = singular.map(|value| value.max(1e-12).ln())
        + Vector3::repeat(plastic.log_volume_gain / 3.0);
    let trace = strain.sum();
    let deviatoric = strain - Vector3::repeat(trace / 3.0);
    let norm = deviatoric.norm();
    let shifted_trace = trace - cohesion;
    let sine = friction_angle.sin();
    let alpha = (2.0 / 3.0f64).sqrt() * (2.0 * sine) / (3.0 - sine);
    let (lambda, mu) = lame(young, poisson);

    let projected_log = if shifted_trace > 0.0 || norm <= 1e-12 {
        Vector3::repeat(cohesion / 3.0)
    } else {
        let gamma = norm + (3.0 * lambda + 2.0 * mu) / (2.0 * mu) * shifted_trace * alpha;
        if gamma <= 0.0 {
            return (deformation, plastic);
        }
        plastic.hardening += gamma;
        strain - deviatoric * (gamma / norm)
    };
    let projected = projected_log.map(f64::exp);
    let old_det = singular.iter().product::<f64>().max(1e-12);
    let new_det = projected.iter().product::<f64>().max(1e-12);
    plastic.plastic_det = (plastic.plastic_det * old_det / new_det).clamp(0.01, 100.0);
    plastic.log_volume_gain += old_det.ln() - new_det.ln();
    (u * Matrix3::from_diagonal(&projected) * v_t, plastic)
}

fn svd_parts(matrix: Matrix3<f64>) -> Option<(Matrix3<f64>, Vector3<f64>, Matrix3<f64>)> {
    let svd = matrix.svd(true, true);
    Some((svd.u?, svd.singular_values, svd.v_t?))
}

#[cfg(test)]
#[path = "sand_neo_hookean_tests.rs"]
mod sand_neo_hookean_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fluid_pressure_resists_compression_and_softly_resists_tension() {
        let model = MaterialModel::fluid(1_000.0, 7.0, 0.01);
        let compressed = model.kirchhoff_stress(
            Matrix3::identity() * 0.9,
            Matrix3::zeros(),
            PlasticState::default(),
        );
        let expanded = model.kirchhoff_stress(
            Matrix3::identity() * 1.1,
            Matrix3::zeros(),
            PlasticState::default(),
        );
        assert!(compressed.trace() < 0.0);
        assert!(expanded.trace() > 0.0);
        assert!(compressed.trace().abs() > expanded.trace().abs());
    }

    #[test]
    fn snow_projection_clamps_elastic_singular_values() {
        let model = MaterialModel::snow(1_000.0, 0.2);
        let deformation = Matrix3::from_diagonal(&Vector3::new(0.5, 1.0, 1.5));
        let (projected, state) = model.project_deformation(deformation, PlasticState::default());
        let singular = projected.svd(false, false).singular_values;
        assert!(singular.iter().all(|value| *value >= 0.975 - 1e-12));
        assert!(singular.iter().all(|value| *value <= 1.0075 + 1e-12));
        assert_ne!(state.plastic_det, 1.0);
    }

    #[test]
    fn sand_projection_removes_unsupported_tension() {
        let model = MaterialModel::sand(5_000.0, 0.2, 35.0f64.to_radians(), 0.0);
        let deformation = Matrix3::identity() * 1.2;
        let (projected, state) = model.project_deformation(deformation, PlasticState::default());
        assert!((projected - Matrix3::identity()).norm() < 1e-10);
        assert!(state.plastic_det > 1.0);
    }

    #[test]
    fn invalid_material_parameters_are_rejected_by_validation() {
        assert!(!MaterialModel::elastic(-1.0, 0.2).is_valid());
        assert!(!MaterialModel::fluid(1.0, 0.0, 0.0).is_valid());
        assert!(MaterialModel::neo_hookean(1_000.0, 0.49).is_valid());
    }
}
