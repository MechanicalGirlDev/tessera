//! Pure CPU damped-least-squares robot kinematics in Tessera coordinate order.

use nalgebra::{DMatrix, DVector, Isometry3, Point3, UnitQuaternion, Vector3};

use crate::articulated_world::JointPolynomialCoupling;
use crate::articulation::{Articulation, ArticulationError, ArticulationPose};

/// Robot configuration, independent of simulation velocities and contact caches.
#[derive(Debug, Clone)]
pub struct IkState {
    /// World root pose, updated in world tangent coordinates for floating robots.
    pub root_pose: Isometry3<f64>,
    /// Reduced joint coordinates. Quaternion spherical slots are workspace.
    pub positions: Vec<f64>,
    /// Optional per-edge quaternion state; otherwise intrinsic XYZ coordinates.
    pub orientations: Option<Vec<Option<UnitQuaternion<f64>>>>,
}

/// World-frame target for one link-local point and link orientation.
#[derive(Debug, Clone)]
pub struct IkTarget {
    /// Stable articulation link index.
    pub link: usize,
    /// Desired point position and link orientation.
    pub pose: Isometry3<f64>,
    /// Point expressed in the link frame.
    pub local_point: Vector3<f64>,
    /// World linear X/Y/Z, then angular X/Y/Z components to constrain.
    pub constrained_axes: [bool; 6],
}

/// Damped-least-squares options.
#[derive(Debug, Clone)]
pub struct IkConfig {
    /// Maximum number of displacement updates.
    pub max_iterations: usize,
    /// Positive damping; the normal matrix receives its square.
    pub damping: f64,
    /// Nonnegative constrained linear error tolerance, metres.
    pub position_tolerance: f64,
    /// Nonnegative constrained angular error tolerance, radians.
    pub rotation_tolerance: f64,
    /// Nonnegative scalar equality error tolerance.
    pub coupling_tolerance: f64,
    /// Movable generalized slots, or all slots. Floating slots precede joints.
    pub dofs: Option<Vec<usize>>,
}

impl Default for IkConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            damping: 0.05,
            position_tolerance: 1e-4,
            rotation_tolerance: 1e-3,
            coupling_tolerance: 1e-6,
            dofs: None,
        }
    }
}

/// Final state and measured convergence, including unreachable targets.
#[derive(Debug, Clone)]
pub struct IkResult {
    /// Solved configuration, without changing the input state.
    pub state: IkState,
    /// Whether all enabled pose axes and scalar equalities met their tolerances.
    pub converged: bool,
    /// Number of displacement updates actually applied.
    pub iterations: usize,
    /// World linear then angular errors; disabled axes are exactly zero.
    pub residual: [f64; 6],
    /// Maximum absolute scalar equality error.
    pub coupling_residual: f64,
}

/// Invalid IK input or a failed numerical solve.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum IkError {
    /// Invalid damping, tolerance, iteration count, or movable slots.
    #[error("invalid inverse kinematics configuration")]
    InvalidConfig,
    /// Invalid target link, point, or pose.
    #[error("invalid inverse kinematics target")]
    InvalidTarget,
    /// Invalid robot coordinates or spherical state.
    #[error("invalid inverse kinematics state")]
    InvalidState,
    /// Scalar coupling is invalid or refers to spherical workspace.
    #[error("invalid inverse kinematics scalar coupling")]
    InvalidCoupling,
    /// The normal solve or displacement became nonfinite.
    #[error("inverse kinematics numerical solve failed")]
    NumericalFailure,
}

fn valid_pose(pose: &Isometry3<f64>) -> bool {
    pose.translation
        .vector
        .iter()
        .chain(pose.rotation.coords.iter())
        .all(|v| v.is_finite())
        && (pose.rotation.norm_squared() - 1.0).abs() <= 1e-8
}

/// Evaluate all links without writing simulation state.
pub fn forward_kinematics(
    articulation: &Articulation,
    state: &IkState,
) -> Result<ArticulationPose, IkError> {
    if !valid_pose(&state.root_pose) {
        return Err(IkError::InvalidState);
    }
    match &state.orientations {
        Some(orientations) => articulation.pose_with_spherical_orientations(
            state.root_pose,
            &state.positions,
            orientations,
        ),
        None => articulation.pose(state.root_pose, &state.positions),
    }
    .map_err(|_: ArticulationError| IkError::InvalidState)
}

/// Solve one pose target with clamped reduced limits and scalar equality rows.
///
/// Mimic columns and limits come directly from the articulation. Floating roots
/// use world-linear/world-angular slots followed by reduced joint slots. Explicit
/// spherical quaternions are left-multiplied by joint-frame tangent rotations;
/// legacy spherical coordinates instead use their intrinsic XYZ Jacobians.
/// The point position and its Jacobian use the current link orientation, so
/// unconstrained rotation cannot change the meaning of a position-only target.
pub fn inverse_kinematics(
    articulation: &Articulation,
    initial: &IkState,
    floating: bool,
    target: &IkTarget,
    config: &IkConfig,
    couplings: &[JointPolynomialCoupling],
) -> Result<IkResult, IkError> {
    let prefix = if floating { 6 } else { 0 };
    let n = prefix + articulation.dof();
    if config.max_iterations == 0
        || !config.damping.is_finite()
        || config.damping <= 0.0
        || !config.damping.powi(2).is_finite()
        || config.damping.powi(2) == 0.0
        || [
            config.position_tolerance,
            config.rotation_tolerance,
            config.coupling_tolerance,
        ]
        .iter()
        .any(|v| !v.is_finite() || *v < 0.0)
    {
        return Err(IkError::InvalidConfig);
    }
    let movable = config.dofs.clone().unwrap_or_else(|| (0..n).collect());
    let mut selected = vec![false; n];
    for &slot in &movable {
        if slot >= n || selected[slot] {
            return Err(IkError::InvalidConfig);
        }
        selected[slot] = true;
    }
    if target.link >= articulation.link_count()
        || !valid_pose(&target.pose)
        || target.local_point.iter().any(|v| !v.is_finite())
    {
        return Err(IkError::InvalidTarget);
    }
    let mut state = initial.clone();
    let _ = forward_kinematics(articulation, &state)?;
    let mut spherical_slots = vec![false; articulation.dof()];
    for edge in 0..articulation.link_count() - 1 {
        if let Some(range) = articulation.joint_coordinate_range(edge)
            && range.len() == 3
        {
            spherical_slots[range].fill(true);
        }
    }
    for coupling in couplings {
        let valid_slot = |slot: usize| slot < spherical_slots.len() && !spherical_slots[slot];
        if !valid_slot(coupling.follower)
            || coupling
                .source
                .is_some_and(|slot| !valid_slot(slot) || slot == coupling.follower)
            || coupling.coefficients.iter().any(|v| !v.is_finite())
            || !coupling.follower_reference.is_finite()
            || !coupling.source_reference.is_finite()
        {
            return Err(IkError::InvalidCoupling);
        }
    }
    // Clamp only selected coordinates; frozen input coordinates must be feasible.
    for (slot, q) in state.positions.iter_mut().enumerate() {
        if let Some((lo, hi)) = articulation.joint_limit(slot) {
            if selected[prefix + slot] {
                *q = q.clamp(lo, hi);
            } else if *q < lo || *q > hi {
                return Err(IkError::InvalidState);
            }
        }
    }
    for iterations in 0..=config.max_iterations {
        let pose = forward_kinematics(articulation, &state)?;
        let actual = &pose.links[target.link];
        let actual_point = actual.transform_point(&Point3::from(target.local_point));
        let linear = target.pose.translation.vector - actual_point.coords;
        let angular = (target.pose.rotation * actual.rotation.inverse()).scaled_axis();
        let mut residual = [
            linear.x, linear.y, linear.z, angular.x, angular.y, angular.z,
        ];
        for (value, enabled) in residual.iter_mut().zip(target.constrained_axes) {
            if !enabled {
                *value = 0.0;
            }
        }
        let (jl, ja) = articulation
            .generalized_point_jacobians(&pose, target.link, target.local_point, floating)
            .map_err(|_| IkError::InvalidState)?;
        let mut jacobian = DMatrix::zeros(6 + couplings.len(), movable.len());
        let mut error = DVector::zeros(6 + couplings.len());
        for row in 0..6 {
            error[row] = residual[row];
            if target.constrained_axes[row] {
                for (column, &slot) in movable.iter().enumerate() {
                    jacobian[(row, column)] = if row < 3 {
                        jl[(row, slot)]
                    } else {
                        ja[(row - 3, slot)]
                    };
                }
            }
        }
        let mut coupling_residual: f64 = 0.0;
        for (index, coupling) in couplings.iter().enumerate() {
            let x = coupling.source.map_or(0.0, |slot| {
                state.positions[slot] - coupling.source_reference
            });
            let c = coupling.coefficients;
            let value = c[0] + x * (c[1] + x * (c[2] + x * (c[3] + x * c[4])));
            let derivative = c[1] + x * (2.0 * c[2] + x * (3.0 * c[3] + x * 4.0 * c[4]));
            let row = 6 + index;
            error[row] = value + coupling.follower_reference - state.positions[coupling.follower];
            coupling_residual = coupling_residual.max(error[row].abs());
            for (column, &slot) in movable.iter().enumerate() {
                jacobian[(row, column)] = if slot == prefix + coupling.follower {
                    1.0
                } else if coupling
                    .source
                    .is_some_and(|source| slot == prefix + source)
                {
                    -derivative
                } else {
                    0.0
                };
            }
        }
        if error.iter().chain(jacobian.iter()).any(|v| !v.is_finite()) {
            return Err(IkError::NumericalFailure);
        }
        let converged = Vector3::from_column_slice(&residual[..3]).norm()
            <= config.position_tolerance
            && Vector3::from_column_slice(&residual[3..]).norm() <= config.rotation_tolerance
            && coupling_residual <= config.coupling_tolerance;
        if converged || iterations == config.max_iterations || movable.is_empty() {
            return Ok(IkResult {
                state,
                converged,
                iterations,
                residual,
                coupling_residual,
            });
        }
        let mut normal = jacobian.transpose() * &jacobian;
        for slot in 0..movable.len() {
            normal[(slot, slot)] += config.damping.powi(2);
        }
        let delta = normal
            .cholesky()
            .ok_or(IkError::NumericalFailure)?
            .solve(&(jacobian.transpose() * error));
        if delta.iter().any(|v| !v.is_finite()) {
            return Err(IkError::NumericalFailure);
        }
        let mut displacement = DVector::zeros(n);
        for (&slot, value) in movable.iter().zip(delta.iter()) {
            displacement[slot] = *value;
        }
        if floating {
            state.root_pose.translation.vector +=
                Vector3::from_column_slice(&displacement.as_slice()[..3]);
            state.root_pose.rotation = UnitQuaternion::from_scaled_axis(
                Vector3::from_column_slice(&displacement.as_slice()[3..6]),
            ) * state.root_pose.rotation;
        }
        for slot in 0..articulation.dof() {
            if state.orientations.is_some() && spherical_slots[slot] {
                continue;
            }
            state.positions[slot] += displacement[prefix + slot];
            if let Some((lo, hi)) = articulation.joint_limit(slot) {
                state.positions[slot] = state.positions[slot].clamp(lo, hi);
            }
        }
        if let Some(orientations) = &mut state.orientations {
            for (edge, orientation) in orientations.iter_mut().enumerate() {
                if let Some(rotation) = orientation {
                    let range = articulation
                        .joint_coordinate_range(edge)
                        .ok_or(IkError::InvalidState)?;
                    let shift = Vector3::from_column_slice(
                        &displacement.as_slice()[prefix + range.start..prefix + range.end],
                    );
                    *rotation = UnitQuaternion::from_scaled_axis(shift) * *rotation;
                }
            }
        }
    }
    unreachable!("bounded IK iteration always returns its final residual")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::articulation::{JointKind, JointSpec, LinkSpec};
    use nalgebra::Matrix3;

    fn robot(kind: JointKind, limits: Option<(f64, f64)>) -> Articulation {
        let link = LinkSpec {
            mass: 1.0,
            center_of_mass: Vector3::zeros(),
            inertia: Matrix3::identity(),
        };
        Articulation::new(
            vec![link.clone(), link],
            vec![JointSpec {
                parent: 0,
                child: 1,
                origin: Isometry3::identity(),
                kind,
                axis: Vector3::x(),
                limits,
            }],
            0,
        )
        .unwrap()
    }

    fn state(n: usize) -> IkState {
        IkState {
            root_pose: Isometry3::identity(),
            positions: vec![0.0; n],
            orientations: None,
        }
    }

    fn target(pose: Isometry3<f64>) -> IkTarget {
        IkTarget {
            link: 1,
            pose,
            local_point: Vector3::zeros(),
            constrained_axes: [true; 6],
        }
    }

    #[test]
    fn reachable_pose_and_input_nonmutation() {
        let model = robot(JointKind::Prismatic, Some((-1.0, 1.0)));
        let initial = state(1);
        let result = inverse_kinematics(
            &model,
            &initial,
            false,
            &target(Isometry3::translation(0.7, 0.0, 0.0)),
            &IkConfig::default(),
            &[],
        )
        .unwrap();
        assert!(result.converged && result.iterations > 0);
        assert!((result.state.positions[0] - 0.7).abs() < 1e-4);
        assert_eq!(initial.positions, [0.0]);
        assert!(
            forward_kinematics(&model, &result.state).unwrap().links[1]
                .translation
                .x
                > 0.69
        );
    }

    #[test]
    fn per_axis_partial_target_and_limits() {
        let model = robot(JointKind::Prismatic, Some((-0.5, 0.5)));
        let mut goal = target(Isometry3::translation(0.3, 8.0, -9.0));
        goal.constrained_axes = [true, false, false, false, false, false];
        let result =
            inverse_kinematics(&model, &state(1), false, &goal, &IkConfig::default(), &[]).unwrap();
        assert!(result.converged);
        assert!((result.state.positions[0] - 0.3).abs() < 1e-4);
        assert_eq!(&result.residual[1..], &[0.0; 5]);
        goal.pose.translation.x = 2.0;
        let result =
            inverse_kinematics(&model, &state(1), false, &goal, &IkConfig::default(), &[]).unwrap();
        assert!(!result.converged);
        assert_eq!(result.iterations, 100);
        assert_eq!(result.state.positions, [0.5]);
        assert!((result.residual[0] - 1.5).abs() < 1e-12);
    }

    #[test]
    fn invalid_inputs_and_frozen_dofs() {
        let model = robot(JointKind::Prismatic, None);
        let mut goal = target(Isometry3::translation(0.4, 0.0, 0.0));
        goal.link = 2;
        assert_eq!(
            inverse_kinematics(&model, &state(1), false, &goal, &IkConfig::default(), &[])
                .unwrap_err(),
            IkError::InvalidTarget
        );
        goal.link = 1;
        let config = IkConfig {
            damping: f64::NAN,
            ..IkConfig::default()
        };
        assert_eq!(
            inverse_kinematics(&model, &state(1), false, &goal, &config, &[]).unwrap_err(),
            IkError::InvalidConfig
        );
        let config = IkConfig {
            dofs: Some(vec![1]),
            ..IkConfig::default()
        };
        assert_eq!(
            inverse_kinematics(&model, &state(1), false, &goal, &config, &[]).unwrap_err(),
            IkError::InvalidConfig
        );
        let config = IkConfig {
            dofs: Some(vec![]),
            ..IkConfig::default()
        };
        let result = inverse_kinematics(&model, &state(1), false, &goal, &config, &[]).unwrap();
        assert!(!result.converged);
        assert_eq!(result.iterations, 0);
        assert_eq!(result.state.positions, [0.0]);
        assert_eq!(
            forward_kinematics(&model, &state(0)).unwrap_err(),
            IkError::InvalidState
        );
    }

    #[test]
    fn floating_and_spherical_coordinate_order() {
        let model = robot(JointKind::Spherical, None);
        let mut initial = state(3);
        initial.orientations = Some(vec![Some(UnitQuaternion::identity())]);
        let goal = target(
            Isometry3::translation(0.4, -0.3, 0.2)
                * Isometry3::rotation(Vector3::new(0.3, -0.4, 0.2)),
        );
        let config = IkConfig {
            dofs: Some(vec![0, 1, 2, 6, 7, 8]),
            ..IkConfig::default()
        };
        let result = inverse_kinematics(&model, &initial, true, &goal, &config, &[]).unwrap();
        assert!(result.converged);
        assert!(result.state.root_pose.translation.vector.norm() > 0.5);
        assert_eq!(result.state.root_pose.rotation, UnitQuaternion::identity());
        assert_eq!(result.state.positions, [0.0; 3]);
        let pose = forward_kinematics(&model, &result.state).unwrap();
        assert!(pose.links[1].rotation.angle_to(&goal.pose.rotation) < 1e-3);
        // Legacy intrinsic XYZ uses a different Jacobian and additive update.
        let legacy = inverse_kinematics(&model, &state(3), true, &goal, &config, &[]).unwrap();
        assert!(legacy.converged);
        assert!(legacy.state.positions.iter().any(|q| q.abs() > 0.1));
    }

    #[test]
    fn mimic_limits_and_polynomial_coupling() {
        let mut model = robot(JointKind::Prismatic, Some((-1.0, 1.0)));
        let link = model.link(0).unwrap().clone();
        model = Articulation::new(
            vec![link.clone(), link.clone(), link],
            vec![
                JointSpec {
                    parent: 0,
                    child: 1,
                    origin: Isometry3::identity(),
                    kind: JointKind::Prismatic,
                    axis: Vector3::x(),
                    limits: Some((-1.0, 1.0)),
                },
                JointSpec {
                    parent: 1,
                    child: 2,
                    origin: Isometry3::identity(),
                    kind: JointKind::Prismatic,
                    axis: Vector3::x(),
                    limits: Some((-0.4, 0.4)),
                },
            ],
            0,
        )
        .unwrap();
        model.set_mimics(&[(1, 0, 2.0, 0.0)]).unwrap();
        let mut goal = target(Isometry3::translation(0.3, 0.0, 0.0));
        goal.link = 2;
        let result =
            inverse_kinematics(&model, &state(1), false, &goal, &IkConfig::default(), &[]).unwrap();
        assert!(result.converged);
        assert!((result.state.positions[0] - 0.1).abs() < 1e-4);
        goal.pose.translation.x = 2.0;
        let result =
            inverse_kinematics(&model, &state(1), false, &goal, &IkConfig::default(), &[]).unwrap();
        assert!(!result.converged);
        assert_eq!(result.state.positions, [0.2]);
        model.set_mimics(&[]).unwrap();
        goal.pose.translation.x = 0.3;
        let coupling = JointPolynomialCoupling {
            follower: 1,
            source: Some(0),
            coefficients: [0.0, 0.0, 1.0, 0.0, 0.0],
            follower_reference: 0.0,
            source_reference: 0.0,
        };
        let result = inverse_kinematics(
            &model,
            &state(2),
            false,
            &goal,
            &IkConfig::default(),
            &[coupling],
        )
        .unwrap();
        assert!(result.converged);
        assert!(result.coupling_residual <= 1e-6);
        assert!((result.state.positions[1] - result.state.positions[0].powi(2)).abs() < 1e-6);
        assert!(result.state.positions[0] > 0.2);
    }

    #[test]
    fn position_only_target_tracks_a_rotating_link_local_point() {
        let model = robot(JointKind::Revolute, Some((-2.0, 2.0)));
        let initial = state(1);
        let mut goal = target(Isometry3::translation(0.0, 0.0, 1.0));
        goal.local_point = Vector3::y();
        goal.constrained_axes = [false, true, true, false, false, false];

        let result =
            inverse_kinematics(&model, &initial, false, &goal, &IkConfig::default(), &[]).unwrap();
        assert!(result.converged);
        let pose = forward_kinematics(&model, &result.state).unwrap();
        let point = pose.links[1].transform_point(&Point3::from(goal.local_point));
        assert!((point.coords - goal.pose.translation.vector).norm() < 1e-4);
        assert!((result.state.positions[0] - core::f64::consts::FRAC_PI_2).abs() < 1e-4);
        assert_eq!(initial.positions, [0.0]);
    }

    #[test]
    fn local_point_and_rotated_world_axis() {
        let model = robot(JointKind::Prismatic, None);
        let mut initial = state(1);
        initial.root_pose.rotation =
            UnitQuaternion::from_axis_angle(&Vector3::z_axis(), core::f64::consts::FRAC_PI_2);
        let mut goal = target(initial.root_pose);
        goal.pose.translation.vector = Vector3::new(-0.2, 0.6, 0.0);
        goal.local_point = Vector3::new(0.1, 0.2, 0.0);
        goal.constrained_axes = [false, true, false, false, false, false];
        let result =
            inverse_kinematics(&model, &initial, false, &goal, &IkConfig::default(), &[]).unwrap();
        assert!(result.converged);
        assert!((result.state.positions[0] - 0.5).abs() < 1e-4);
    }
}
