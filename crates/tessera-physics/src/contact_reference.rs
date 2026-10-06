//! CPU reference for projected normal and Coulomb contact impulses.
//!
//! The caller supplies Jacobian rows and an inverse generalized mass matrix.
//! This keeps the contact law independent of a particular robot topology and
//! provides a numerical reference for a future GPU implementation.

use nalgebra::{DMatrix, DVector};

/// Generalized-velocity Jacobian rows for a contact or bounded scalar constraint.
#[derive(Debug, Clone)]
pub struct ContactConstraint {
    /// Bilateral scalar row parameters. None selects a unilateral contact.
    pub scalar: Option<ScalarConstraint>,
    /// Relative separating velocity is `normal.dot(velocity)`.
    pub normal: DVector<f64>,
    /// Two tangent rows in the contact frame.
    pub tangents: [DVector<f64>; 2],
    /// Positive overlap depth in metres.
    pub penetration: f64,
    /// Coulomb sliding-friction coefficient.
    pub friction: f64,
    /// Non-negative normal coefficient of restitution.
    pub restitution: f64,
}

/// Target speed and optional symmetric impulse bound for a bilateral row.
#[derive(Debug, Clone, Copy)]
pub struct ScalarConstraint {
    /// Desired Jacobian velocity after solving.
    pub target_speed: f64,
    /// Symmetric impulse bound; None leaves the row unconstrained.
    pub impulse_limit: Option<f64>,
}

impl ContactConstraint {
    /// A scalar row with a signed impulse bounded by dry friction.
    pub fn bounded_axis(normal: DVector<f64>, max_impulse: f64) -> Self {
        let width = normal.len();
        Self {
            scalar: Some(ScalarConstraint {
                target_speed: 0.0,
                impulse_limit: Some(max_impulse),
            }),
            normal,
            tangents: [DVector::zeros(width), DVector::zeros(width)],
            penetration: 0.0,
            friction: 0.0,
            restitution: 0.0,
        }
    }

    /// A bilateral scalar row with no impulse limit.
    pub fn bilateral_axis(normal: DVector<f64>, target_speed: f64) -> Self {
        let mut row = Self::bounded_axis(normal, 0.0);
        row.scalar = Some(ScalarConstraint {
            target_speed,
            impulse_limit: None,
        });
        row
    }
}

/// Inputs to one velocity-level contact solve.
#[derive(Debug, Clone)]
pub struct ContactProblem {
    /// Inverse generalized mass matrix in the Jacobian's coordinate order.
    pub inverse_mass: DMatrix<f64>,
    /// Unconstrained generalized velocity after external forces.
    pub velocity: DVector<f64>,
    /// Contacts in the order used by any warm-start impulses.
    pub contacts: Vec<ContactConstraint>,
}

/// Accumulated contact impulse, in normal/tangent frame order.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ContactImpulse {
    /// Accumulated normal impulse.
    pub normal: f64,
    /// Accumulated impulse along the two contact tangents.
    pub tangents: [f64; 2],
}

/// Fixed-step solver parameters.
#[derive(Debug, Clone, Copy)]
pub struct SolveParams {
    /// Time step in seconds.
    pub dt: f64,
    /// Number of sequential contact sweeps.
    pub iterations: usize,
    /// Fraction of penetration recovered per step, in [0, 1].
    pub position_gain: f64,
    /// Upper bound on the separating velocity from penetration correction.
    pub max_correction_speed: f64,
}

/// Generalized velocity and contact impulses after solving.
#[derive(Debug, Clone)]
pub struct ContactSolution {
    /// Corrected generalized velocity.
    pub velocity: DVector<f64>,
    /// Impulses in the same order as the input contacts.
    pub impulses: Vec<ContactImpulse>,
}

/// Invalid contact problem or solver parameters.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ContactSolveError {
    /// Matrices, vectors, or warm-start data have different dimensions.
    #[error("contact solve input has inconsistent dimensions")]
    Dimensions,
    /// A parameter is non-finite or violates its allowed range.
    #[error("contact solve input contains non-finite or invalid parameters")]
    InvalidInput,
    /// A contact has no positive effective inverse mass.
    #[error("contact Jacobian has zero or negative effective inverse mass")]
    SingularContact,
}

struct PreparedContact {
    response: [DVector<f64>; 3],
    effective_inverse_mass: [f64; 3],
    tangent_coupling: f64,
    tangent_active: [bool; 2],
    correction_speed: f64,
    restitution_speed: f64,
}

/// Solve contact impulses with projected Gauss-Seidel iterations.
///
/// A warm start must correspond to the same contact order and frame as the
/// current problem. The caller is responsible for manifold identity tracking.
/// Coupled tangent rows use a two-by-two effective mass solve before projecting
/// the result onto the Coulomb friction disk.
pub fn solve_contacts(
    problem: &ContactProblem,
    params: SolveParams,
    warm_start: Option<&[ContactImpulse]>,
) -> Result<ContactSolution, ContactSolveError> {
    let n = problem.velocity.len();
    if problem.inverse_mass.nrows() != n
        || problem.inverse_mass.ncols() != n
        || warm_start.is_some_and(|impulses| impulses.len() != problem.contacts.len())
    {
        return Err(ContactSolveError::Dimensions);
    }
    if !params.dt.is_finite()
        || params.dt <= 0.0
        || params.iterations == 0
        || !params.position_gain.is_finite()
        || !(0.0..=1.0).contains(&params.position_gain)
        || !params.max_correction_speed.is_finite()
        || params.max_correction_speed < 0.0
        || !problem.velocity.iter().all(|x| x.is_finite())
        || !problem.inverse_mass.iter().all(|x| x.is_finite())
    {
        return Err(ContactSolveError::InvalidInput);
    }

    let mut prepared = Vec::with_capacity(problem.contacts.len());
    for contact in &problem.contacts {
        if contact.normal.len() != n || contact.tangents.iter().any(|row| row.len() != n) {
            return Err(ContactSolveError::Dimensions);
        }
        if !contact.penetration.is_finite()
            || contact.penetration < 0.0
            || !contact.friction.is_finite()
            || contact.friction < 0.0
            || !contact.restitution.is_finite()
            || contact.restitution < 0.0
            || !contact.normal.iter().all(|x| x.is_finite())
            || !contact.tangents.iter().flatten().all(|x| x.is_finite())
            || contact.scalar.is_some_and(|scalar| {
                !scalar.target_speed.is_finite()
                    || scalar
                        .impulse_limit
                        .is_some_and(|bound| !bound.is_finite() || bound < 0.0)
                    || contact.penetration != 0.0
                    || contact.friction != 0.0
                    || contact.restitution != 0.0
                    || contact.tangents.iter().flatten().any(|value| *value != 0.0)
            })
        {
            return Err(ContactSolveError::InvalidInput);
        }
        let rows = [&contact.normal, &contact.tangents[0], &contact.tangents[1]];
        let response = rows.map(|row| &problem.inverse_mass * row);
        let effective_inverse_mass = core::array::from_fn(|i| rows[i].dot(&response[i]));
        let tangent_coupling = rows[1].dot(&response[2]);
        if effective_inverse_mass.iter().any(|x| !x.is_finite())
            || !tangent_coupling.is_finite()
            || effective_inverse_mass[0] <= 1e-12
        {
            return Err(ContactSolveError::SingularContact);
        }
        let incoming_speed = contact.normal.dot(&problem.velocity);
        prepared.push(PreparedContact {
            response,
            effective_inverse_mass,
            tangent_coupling,
            tangent_active: [
                effective_inverse_mass[1] > 1e-12,
                effective_inverse_mass[2] > 1e-12,
            ],
            correction_speed: (params.position_gain * contact.penetration / params.dt)
                .min(params.max_correction_speed),
            restitution_speed: if incoming_speed < -1e-3 {
                -contact.restitution * incoming_speed
            } else {
                0.0
            },
        });
    }

    let mut velocity = problem.velocity.clone();
    let mut impulses = vec![ContactImpulse::default(); problem.contacts.len()];
    if let Some(warm_start) = warm_start {
        for (i, (&seed, contact)) in warm_start.iter().zip(&problem.contacts).enumerate() {
            if !seed.normal.is_finite() || seed.tangents.iter().any(|x| !x.is_finite()) {
                return Err(ContactSolveError::InvalidInput);
            }
            if let Some(scalar) = contact.scalar {
                let normal = scalar
                    .impulse_limit
                    .map_or(seed.normal, |bound| seed.normal.clamp(-bound, bound));
                impulses[i].normal = normal;
                velocity += &prepared[i].response[0] * normal;
                continue;
            }
            let normal = seed.normal.max(0.0);
            let mut tangents = project_friction(seed.tangents, contact.friction * normal);
            for (axis, value) in tangents.iter_mut().enumerate() {
                if !prepared[i].tangent_active[axis] {
                    *value = 0.0;
                }
            }
            impulses[i] = ContactImpulse { normal, tangents };
            velocity += &prepared[i].response[0] * normal;
            velocity += &prepared[i].response[1] * tangents[0];
            velocity += &prepared[i].response[2] * tangents[1];
        }
    }

    // Settle support and friction first, then solve the coupled impact targets.
    // Both phases sweep every contact so a bounce cannot invalidate a later row.
    let phases = if prepared
        .iter()
        .any(|p| p.restitution_speed > p.correction_speed)
    {
        2
    } else {
        1
    };
    for phase in 0..phases {
        for _ in 0..params.iterations {
            for (i, contact) in problem.contacts.iter().enumerate() {
                let p = &prepared[i];
                let impulse = &mut impulses[i];
                if let Some(scalar) = contact.scalar {
                    let delta = (scalar.target_speed - contact.normal.dot(&velocity))
                        / p.effective_inverse_mass[0];
                    let next = scalar
                        .impulse_limit
                        .map_or(impulse.normal + delta, |bound| {
                            (impulse.normal + delta).clamp(-bound, bound)
                        });
                    velocity += &p.response[0] * (next - impulse.normal);
                    impulse.normal = next;
                    continue;
                }
                let target_speed = if phase == 0 {
                    p.correction_speed
                } else {
                    p.correction_speed.max(p.restitution_speed)
                };
                let delta =
                    (target_speed - contact.normal.dot(&velocity)) / p.effective_inverse_mass[0];
                let next_normal = (impulse.normal + delta).max(0.0);
                velocity += &p.response[0] * (next_normal - impulse.normal);
                impulse.normal = next_normal;

                let mut next_tangents = impulse.tangents;
                let tangent_speed = contact.tangents.each_ref().map(|row| row.dot(&velocity));
                let a = p.effective_inverse_mass[1];
                let c = p.effective_inverse_mass[2];
                let b = p.tangent_coupling;
                let determinant = a * c - b * b;
                if p.tangent_active.iter().all(|active| *active)
                    && determinant.is_finite()
                    && determinant > 1e-6 * a * c
                {
                    next_tangents[0] -= (c * tangent_speed[0] - b * tangent_speed[1]) / determinant;
                    next_tangents[1] -= (a * tangent_speed[1] - b * tangent_speed[0]) / determinant;
                } else {
                    for axis in 0..2 {
                        if p.tangent_active[axis] {
                            next_tangents[axis] -=
                                tangent_speed[axis] / p.effective_inverse_mass[axis + 1];
                        }
                    }
                }
                next_tangents = project_friction(next_tangents, contact.friction * impulse.normal);
                for (axis, value) in next_tangents.iter().enumerate() {
                    velocity += &p.response[axis + 1] * (*value - impulse.tangents[axis]);
                }
                impulse.tangents = next_tangents;
            }
        }
    }

    if !velocity.iter().all(|x| x.is_finite()) {
        return Err(ContactSolveError::InvalidInput);
    }
    Ok(ContactSolution { velocity, impulses })
}

fn project_friction(mut tangent: [f64; 2], radius: f64) -> [f64; 2] {
    let length = tangent[0].hypot(tangent[1]);
    if length > radius && length > 0.0 {
        let scale = radius / length;
        tangent[0] *= scale;
        tangent[1] *= scale;
    }
    tangent
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> SolveParams {
        SolveParams {
            dt: 0.01,
            iterations: 10,
            position_gain: 0.0,
            max_correction_speed: 1.0,
        }
    }

    fn floor_problem(velocity: [f64; 3], friction: f64) -> ContactProblem {
        ContactProblem {
            inverse_mass: DMatrix::identity(3, 3),
            velocity: DVector::from_vec(velocity.to_vec()),
            contacts: vec![ContactConstraint {
                scalar: None,
                normal: DVector::from_vec(vec![0.0, 0.0, 1.0]),
                tangents: [
                    DVector::from_vec(vec![1.0, 0.0, 0.0]),
                    DVector::from_vec(vec![0.0, 1.0, 0.0]),
                ],
                penetration: 0.0,
                friction,
                restitution: 0.0,
            }],
        }
    }

    #[test]
    fn bounded_axis_sticks_then_slips_in_both_directions() {
        let mut problem = ContactProblem {
            inverse_mass: DMatrix::identity(1, 1),
            velocity: DVector::from_element(1, 0.5),
            contacts: vec![ContactConstraint::bounded_axis(
                DVector::from_element(1, 1.0),
                0.2,
            )],
        };
        let solved = solve_contacts(&problem, params(), None).unwrap();
        assert!((solved.velocity[0] - 0.3).abs() < 1e-12);
        assert!((solved.impulses[0].normal + 0.2).abs() < 1e-12);
        problem.velocity[0] = -0.1;
        let solved = solve_contacts(&problem, params(), None).unwrap();
        assert!(solved.velocity[0].abs() < 1e-12);
        assert!((solved.impulses[0].normal - 0.1).abs() < 1e-12);
        assert!(
            solve_contacts(
                &problem,
                params(),
                Some(&[ContactImpulse {
                    normal: -2.0,
                    tangents: [1.0, 0.0]
                }])
            )
            .is_ok()
        );
    }

    #[test]
    fn bilateral_axis_conserves_mass_weighted_velocity() {
        let problem = ContactProblem {
            inverse_mass: DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 2.0])),
            velocity: DVector::from_vec(vec![1.0, 0.0]),
            contacts: vec![ContactConstraint::bilateral_axis(
                DVector::from_vec(vec![-2.0, 1.0]),
                0.0,
            )],
        };
        let solved = solve_contacts(&problem, params(), None).unwrap();
        assert!((solved.velocity[0] - 1.0 / 3.0).abs() < 1e-12);
        assert!((solved.velocity[1] - 2.0 / 3.0).abs() < 1e-12);
        assert!((solved.impulses[0].normal - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn floor_stops_downward_motion_and_limits_sliding_friction() {
        let solution =
            solve_contacts(&floor_problem([1.0, 0.0, -1.0], 0.5), params(), None).unwrap();
        assert!((solution.velocity[2]).abs() < 1e-12);
        assert!((solution.velocity[0] - 0.5).abs() < 1e-12);
        assert!((solution.impulses[0].normal - 1.0).abs() < 1e-12);
        assert!((solution.impulses[0].tangents[0] + 0.5).abs() < 1e-12);
    }

    #[test]
    fn coupled_tangent_mass_cancels_sliding_in_one_iteration() {
        let diagonal = 1.0 / 2.0f64.sqrt();
        let problem = ContactProblem {
            inverse_mass: DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 4.0, 1.0])),
            velocity: DVector::from_vec(vec![1.0, 0.0, -1.0]),
            contacts: vec![ContactConstraint {
                scalar: None,
                normal: DVector::from_vec(vec![0.0, 0.0, 1.0]),
                tangents: [
                    DVector::from_vec(vec![diagonal, diagonal, 0.0]),
                    DVector::from_vec(vec![-diagonal, diagonal, 0.0]),
                ],
                penetration: 0.0,
                friction: 10.0,
                restitution: 0.0,
            }],
        };
        let mut settings = params();
        settings.iterations = 1;
        let solution = solve_contacts(&problem, settings, None).unwrap();
        assert!(solution.velocity.norm() < 1e-12);
        assert!((solution.impulses[0].tangents[0] + diagonal).abs() < 1e-12);
        assert!((solution.impulses[0].tangents[1] - diagonal).abs() < 1e-12);
    }

    #[test]
    fn restitution_reflects_the_incoming_normal_speed() {
        let mut problem = floor_problem([0.0, 0.0, -2.0], 0.0);
        problem.contacts[0].restitution = 0.8;
        let solution = solve_contacts(&problem, params(), None).unwrap();
        assert!((solution.velocity[2] - 1.6).abs() < 1e-12);
        assert!((solution.impulses[0].normal - 3.6).abs() < 1e-12);
    }

    #[test]
    fn superelastic_restitution_can_increase_separation_speed() {
        let mut problem = floor_problem([0.0, 0.0, -2.0], 0.0);
        problem.contacts[0].restitution = 1.25;
        let solution = solve_contacts(&problem, params(), None).unwrap();
        assert!((solution.velocity[2] - 2.5).abs() < 1e-12);
    }

    #[test]
    fn final_restitution_survives_a_coupled_support_contact() {
        let normal = DVector::from_vec(vec![-0.5, 3.0f64.sqrt() * 0.5]);
        let problem = ContactProblem {
            inverse_mass: DMatrix::identity(2, 2),
            velocity: DVector::from_vec(vec![-1.0, -1.0]),
            contacts: vec![
                ContactConstraint {
                    scalar: None,
                    normal: DVector::from_vec(vec![1.0, 0.0]),
                    tangents: [DVector::zeros(2), DVector::zeros(2)],
                    penetration: 0.0,
                    friction: 0.0,
                    restitution: 1.0,
                },
                ContactConstraint {
                    scalar: None,
                    normal,
                    tangents: [DVector::zeros(2), DVector::zeros(2)],
                    penetration: 0.0,
                    friction: 0.0,
                    restitution: 0.0,
                },
            ],
        };
        let mut settings = params();
        settings.iterations = 16;
        let solved = solve_contacts(&problem, settings, None).unwrap();
        assert!((solved.velocity[0] - 1.0).abs() < 1e-9);
        assert!(problem.contacts[1].normal.dot(&solved.velocity) >= -1e-9);
        assert!(solved.impulses[0].normal > 1.0);

        settings.iterations = 1;
        let short_solve = solve_contacts(&problem, settings, None).unwrap();
        assert!(short_solve.velocity[0] > 0.6);
        assert!(problem.contacts[1].normal.dot(&short_solve.velocity) >= -1e-12);
    }

    #[test]
    fn separating_contact_applies_no_impulse() {
        let solution =
            solve_contacts(&floor_problem([0.0, 0.0, 1.0], 0.5), params(), None).unwrap();
        assert_eq!(solution.impulses[0], ContactImpulse::default());
        assert_eq!(solution.velocity[2], 1.0);
    }

    #[test]
    fn warm_start_is_projected_and_does_not_double_count() {
        let seed = ContactImpulse {
            normal: 1.0,
            tangents: [2.0, 0.0],
        };
        let solution = solve_contacts(
            &floor_problem([0.0, 0.0, -1.0], 0.5),
            params(),
            Some(&[seed]),
        )
        .unwrap();
        assert!((solution.impulses[0].normal - 1.0).abs() < 1e-12);
        assert!(solution.impulses[0].tangents[0].abs() < 1e-12);
        assert!(solution.velocity[2].abs() < 1e-12);
    }

    #[test]
    fn penetration_correction_is_capped() {
        let mut problem = floor_problem([0.0, 0.0, 0.0], 0.0);
        problem.contacts[0].penetration = 0.1;
        let mut settings = params();
        settings.position_gain = 0.2;
        let solution = solve_contacts(&problem, settings, None).unwrap();
        assert!((solution.velocity[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn contact_impulse_respects_both_bodies_mass() {
        let problem = ContactProblem {
            inverse_mass: DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 0.5])),
            velocity: DVector::from_vec(vec![1.0, -1.0]),
            contacts: vec![ContactConstraint {
                scalar: None,
                normal: DVector::from_vec(vec![-1.0, 1.0]),
                tangents: [
                    DVector::from_vec(vec![1.0, 0.0]),
                    DVector::from_vec(vec![0.0, 1.0]),
                ],
                penetration: 0.0,
                friction: 0.0,
                restitution: 0.0,
            }],
        };
        let solution = solve_contacts(&problem, params(), None).unwrap();
        assert!((solution.impulses[0].normal - 4.0 / 3.0).abs() < 1e-12);
        assert!((solution.velocity[0] + 1.0 / 3.0).abs() < 1e-12);
        assert!((solution.velocity[1] + 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn rejects_invalid_dimensions_and_singular_contact() {
        let mut problem = floor_problem([0.0, 0.0, -1.0], 0.0);
        problem.contacts[0].normal = DVector::zeros(2);
        assert_eq!(
            solve_contacts(&problem, params(), None).unwrap_err(),
            ContactSolveError::Dimensions
        );
        problem.contacts[0].normal = DVector::zeros(3);
        assert_eq!(
            solve_contacts(&problem, params(), None).unwrap_err(),
            ContactSolveError::SingularContact
        );
    }

    #[test]
    fn one_dof_contact_accepts_zero_tangent_jacobians() {
        let problem = ContactProblem {
            inverse_mass: DMatrix::identity(1, 1),
            velocity: DVector::from_vec(vec![-1.0]),
            contacts: vec![ContactConstraint {
                scalar: None,
                normal: DVector::from_vec(vec![1.0]),
                tangents: [DVector::zeros(1), DVector::zeros(1)],
                penetration: 0.0,
                friction: 0.5,
                restitution: 0.0,
            }],
        };
        let solved = solve_contacts(&problem, params(), None).unwrap();
        assert!(solved.velocity[0].abs() < 1e-12);
        assert_eq!(solved.impulses[0].tangents, [0.0; 2]);
    }
}
