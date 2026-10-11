//! Reference elimination for constant-velocity prescribed generalized coordinates.

use nalgebra::{DMatrix, DVector};

/// Error reduction and force mixing for a native impulse constraint.
///
/// Each row solves `(J M^-1 J^T + cfm / dt) delta =
/// target - J v - cfm / dt * accumulated_impulse`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConstraintCompliance {
    /// Position error reduction per substep, in `[0, 1]`.
    pub erp: f64,
    /// Nonnegative force mixing; zero selects a hard constraint.
    pub cfm: f64,
}

impl Default for ConstraintCompliance {
    fn default() -> Self {
        Self { erp: 0.2, cfm: 0.0 }
    }
}

impl ConstraintCompliance {
    /// Whether both parameters have physically meaningful finite values.
    pub fn is_valid(self) -> bool {
        self.erp.is_finite()
            && (0.0..=1.0).contains(&self.erp)
            && self.cfm.is_finite()
            && self.cfm >= 0.0
    }

    #[cfg(feature = "gpu-contact")]
    pub(crate) fn gpu(self) -> Option<crate::gpu_contact_policy::GpuConstraintCompliance> {
        let value = crate::gpu_contact_policy::GpuConstraintCompliance {
            erp: self.erp as f32,
            cfm: self.cfm as f32,
        };
        (self.is_valid()
            && value.is_valid()
            && (self.cfm == 0.0 || value.cfm > 0.0)
            && (self.erp == 0.0 || value.erp > 0.0))
            .then_some(value)
    }
}

/// Independent native contact and loop constraint policies shared by both owners.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ArticulatedConstraintPolicy {
    /// Normal rows between two dynamically responding bodies.
    pub dynamic_contact: ConstraintCompliance,
    /// Normal rows against ground, fixed links, or prescribed bodies.
    pub fixed_contact: ConstraintCompliance,
    /// Linear and polynomial generalized-coordinate couplings.
    pub coupling: ConstraintCompliance,
    /// Point loop constraints, including external anchors.
    pub point_loop: ConstraintCompliance,
    /// Linear and angular rows of fixed-frame loop constraints.
    pub fixed_loop: ConstraintCompliance,
}

impl ArticulatedConstraintPolicy {
    /// Validate every row family before changing an owner.
    pub fn is_valid(self) -> bool {
        [
            self.dynamic_contact,
            self.fixed_contact,
            self.coupling,
            self.point_loop,
            self.fixed_loop,
        ]
        .iter()
        .all(|value| value.is_valid())
    }
}

/// Add one independent impulse coordinate per soft row to the reference solve.
///
/// Its inverse mass is gamma and its Jacobian coefficient is one. Consequently
/// its velocity is exactly gamma * accumulated_impulse, including warm starts.
/// Eliminating these coordinates gives the CFM equation above without changing
/// the existing Coulomb projection or coupling between physical coordinates.
pub(crate) fn add_compliant_rows(
    problem: &mut crate::contact_reference::ContactProblem,
    cfm: &[f64],
    normal_offsets: &[f64],
    dt: f64,
) -> Result<(), crate::contact_reference::ContactSolveError> {
    use crate::contact_reference::ContactSolveError;
    if cfm.len() != problem.contacts.len()
        || normal_offsets.len() != cfm.len()
        || normal_offsets.iter().any(|value| !value.is_finite())
        || !dt.is_finite()
        || dt <= 0.0
        || cfm
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0 || !(value / dt).is_finite())
    {
        return Err(ContactSolveError::InvalidInput);
    }
    let physical = problem.velocity.len();
    let size = physical + cfm.iter().filter(|&&value| value > 0.0).count();
    if size == physical {
        return Ok(());
    }
    problem.inverse_mass.resize_mut(size, size, 0.0);
    problem.velocity.resize_vertically_mut(size, 0.0);
    let mut coordinate = physical;
    for ((row, &value), &offset) in problem.contacts.iter_mut().zip(cfm).zip(normal_offsets) {
        row.normal.resize_vertically_mut(size, 0.0);
        for tangent in &mut row.tangents {
            tangent.resize_vertically_mut(size, 0.0);
        }
        if value > 0.0 {
            row.normal[coordinate] = 1.0;
            problem.inverse_mass[(coordinate, coordinate)] = value / dt;
            problem.velocity[coordinate] = offset;
            coordinate += 1;
        }
    }
    Ok(())
}

/// Invalid layout or a failed dynamic principal-block solve.
#[derive(Debug, thiserror::Error)]
pub enum PrescribedDofError {
    /// Matrix, force, and selection dimensions differ or contain nonfinite values.
    #[error("invalid prescribed coordinate mass system")]
    InvalidInput,
    /// The remaining dynamic principal mass block is singular.
    #[error("singular prescribed coordinate dynamic mass block")]
    Singular,
    /// The dynamic solve produced a nonfinite result.
    #[error("nonfinite prescribed coordinate mass response")]
    NonFinite,
}

/// Solve the dynamic principal mass block and embed its inverse in the full layout.
///
/// `force` must include gravity, effort, and velocity bias computed using *all*
/// generalized velocities, including prescribed scalar/tangent/root motion.
/// Prescribed velocities are neither zeroed nor changed here. Their acceleration
/// and impulse response are exactly zero. The owner must also exclude prescribed
/// slots from scalar clamps, drive constraints, and sleep freezing.
///
/// Returns `(acceleration, inverse_response)` in the original coordinate order.
/// A wholly prescribed system is valid even with zero physical inertia. Invalid
/// dimensions/nonfinite inputs fail before any solve; singular dynamic inertia
/// reports `Singular`.
pub fn solve_prescribed_mass(
    mass: &DMatrix<f64>,
    force: &DVector<f64>,
    prescribed: &[bool],
) -> Result<(DVector<f64>, DMatrix<f64>), PrescribedDofError> {
    let n = force.len();
    if n == 0
        || mass.shape() != (n, n)
        || prescribed.len() != n
        || mass
            .iter()
            .chain(force.iter())
            .any(|value| !value.is_finite())
    {
        return Err(PrescribedDofError::InvalidInput);
    }
    let dynamic = prescribed
        .iter()
        .enumerate()
        .filter_map(|(index, &selected)| (!selected).then_some(index))
        .collect::<Vec<_>>();
    let mut acceleration = DVector::zeros(n);
    let mut response = DMatrix::zeros(n, n);
    if dynamic.is_empty() {
        return Ok((acceleration, response));
    }
    let reduced = DMatrix::from_fn(dynamic.len(), dynamic.len(), |row, col| {
        mass[(dynamic[row], dynamic[col])]
    });
    let rhs = DVector::from_iterator(dynamic.len(), dynamic.iter().map(|&row| force[row]));
    let factor = reduced.lu();
    let solved = factor.solve(&rhs).ok_or(PrescribedDofError::Singular)?;
    let inverse = factor.try_inverse().ok_or(PrescribedDofError::Singular)?;
    if solved
        .iter()
        .chain(inverse.iter())
        .any(|value| !value.is_finite())
    {
        return Err(PrescribedDofError::NonFinite);
    }
    for (row, &coordinate) in dynamic.iter().enumerate() {
        acceleration[coordinate] = solved[row];
        for (col, &other) in dynamic.iter().enumerate() {
            response[(coordinate, other)] = inverse[(row, col)];
        }
    }
    Ok((acceleration, response))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prescribed_cpu_eliminates_coupled_inertia_before_solving() {
        let mass = DMatrix::from_row_slice(2, 2, &[3.0, 1.0, 1.0, 1.0]);
        let force = DVector::from_vec(vec![1000.0, 2.0]);
        let (acceleration, inverse) = solve_prescribed_mass(&mass, &force, &[true, false]).unwrap();
        assert_eq!(acceleration, DVector::from_vec(vec![0.0, 2.0]));
        assert_eq!(
            inverse,
            DMatrix::from_diagonal(&DVector::from_vec(vec![0.0, 1.0]))
        );
        let (acceleration, inverse) =
            solve_prescribed_mass(&mass, &force, &[false, false]).unwrap();
        assert!((&mass * acceleration - &force).norm() < 1e-10);
        assert!((&mass * inverse - DMatrix::identity(2, 2)).norm() < 1e-10);
        let (acceleration, inverse) =
            solve_prescribed_mass(&DMatrix::zeros(2, 2), &force, &[true, true]).unwrap();
        assert_eq!(acceleration.norm(), 0.0);
        assert_eq!(inverse.norm(), 0.0);
        assert!(matches!(
            solve_prescribed_mass(&DMatrix::zeros(2, 2), &force, &[true, false]),
            Err(PrescribedDofError::Singular)
        ));
        assert!(solve_prescribed_mass(&mass, &force, &[true]).is_err());
        assert!(
            solve_prescribed_mass(&mass, &DVector::from_element(2, f64::NAN), &[false; 2]).is_err()
        );
    }
}

#[cfg(all(test, feature = "gpu-contact"))]
mod gpu_tests {
    use super::*;
    use crate::articulation::{Articulation, JointKind, JointSpec, LinkSpec};
    use crate::gpu_articulated_force::GpuArticulatedForceBatch;
    use crate::gpu_articulated_ground_contact::{
        GpuArticulatedGroundContactBatch, GpuArticulatedGroundSphere,
    };
    use crate::gpu_articulated_link_terms::GpuArticulatedLinkTermsBatch;
    use crate::gpu_articulated_mass_assembly::{
        GpuArticulatedMassAssemblyBatch, GpuMassAssemblySystem, GpuMassLink,
    };
    use crate::gpu_articulated_pose::GpuArticulatedPoseBatch;
    use crate::gpu_articulated_root::GpuArticulatedRootBatch;
    use crate::gpu_articulated_spherical::{GpuArticulatedSphericalBatch, GpuSphericalJointState};
    use crate::gpu_articulated_state::{GpuGeneralizedState, GpuGeneralizedStateBatch};
    use crate::gpu_articulated_velocity_bias::GpuArticulatedVelocityBiasBatch;
    use crate::gpu_contact_mass_response::{
        GpuContactMassResponseBatch, GpuContactMassResponseSystem,
    };
    use crate::gpu_contact_pipeline::GpuContactDevice;
    use nalgebra::{Isometry3, Matrix3, UnitQuaternion, Vector3};

    fn systems(articulation: &Articulation, dimensions: &[usize]) -> Vec<GpuMassAssemblySystem> {
        dimensions
            .iter()
            .map(|&n| GpuMassAssemblySystem {
                links: (0..articulation.link_count())
                    .map(|index| {
                        let link = articulation.link(index).unwrap();
                        GpuMassLink {
                            mass: link.mass,
                            inertia_world: link.inertia,
                            linear_jacobian: DMatrix::zeros(3, n),
                            angular_jacobian: DMatrix::zeros(3, n),
                        }
                    })
                    .collect(),
                armature: DVector::zeros(n),
                force: DVector::zeros(n),
            })
            .collect()
    }

    #[test]
    fn prescribed_gpu_mass_and_contact_response_use_dynamic_principal_block() {
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            eprintln!("prescribed coupled inertia and impulse response: {backend:?}");
            let scales = [1.0, 1.0, 1e12, 1e-12, 0.0];
            let inputs = scales.map(|scale| GpuMassAssemblySystem {
                links: vec![GpuMassLink {
                    mass: scale,
                    inertia_world: Matrix3::zeros(),
                    linear_jacobian: DMatrix::from_row_slice(3, 2, &[1.0, 1.0, 1.0, 0.0, 1.0, 0.0]),
                    angular_jacobian: DMatrix::zeros(3, 2),
                }],
                armature: DVector::zeros(2),
                force: DVector::from_vec(vec![1000.0 * scale, 2.0 * scale]),
            });
            let mut mass =
                GpuArticulatedMassAssemblyBatch::new(context.device(), context.queue(), &inputs)
                    .unwrap();
            let masks = vec![
                vec![true, false],
                vec![false, false],
                vec![true, false],
                vec![true, false],
                vec![true, true],
            ];
            mass.set_prescribed_coordinates(&masks).unwrap();
            assert!(mass.set_prescribed_coordinates(&[vec![true; 2]]).is_err());
            let mut invalid = masks.clone();
            assert_eq!(invalid[2].pop(), Some(false));
            assert!(mass.set_prescribed_coordinates(&invalid).is_err());
            let response = GpuContactMassResponseBatch::new(
                context.device(),
                mass.inverse_buffer_or_init(),
                &mass
                    .inverse_ranges()
                    .iter()
                    .map(|range| GpuContactMassResponseSystem {
                        inverse_range: range.clone(),
                        jacobians: DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 1.0, 0.0]),
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            let mut encoder = context.device().create_command_encoder(&Default::default());
            mass.encode_with_inverse(&mut encoder);
            response.encode(&mut encoder);
            let _ = context.queue().submit(Some(encoder.finish()));
            let acceleration = mass.readback().unwrap();
            let inverse = mass.readback_inverse().unwrap();
            let responses = response
                .readback(context.device(), context.queue())
                .unwrap();
            for index in [0, 2, 3] {
                assert_eq!(acceleration[index][0], 0.0);
                assert!((acceleration[index][1] - 2.0).abs() < 1e-5);
                assert_eq!(inverse[index].row(0).norm(), 0.0);
                assert_eq!(inverse[index].column(0).norm(), 0.0);
                assert!((inverse[index][(1, 1)] * scales[index] - 1.0).abs() < 1e-5);
                assert_eq!(responses[index].responses[(0, 0)], 0.0);
                assert!((responses[index].effective[0] * scales[index] - 1.0).abs() < 1e-5);
                assert_eq!(responses[index].effective[1], 0.0);
            }
            let full = DMatrix::from_row_slice(2, 2, &[3.0, 1.0, 1.0, 1.0]);
            assert!((&full * &acceleration[1] - &inputs[1].force).norm() < 1e-4);
            assert!((&full * &inverse[1] - DMatrix::identity(2, 2)).norm() < 1e-6);
            assert_eq!(acceleration[4].norm(), 0.0);
            assert_eq!(inverse[4].norm(), 0.0);

            // Replacing selection changes already-bound contact response, while
            // mass value updates retain the selected generalized layout.
            let mut changed = masks.clone();
            changed[0] = vec![false, true];
            mass.set_prescribed_coordinates(&changed).unwrap();
            mass.update(&inputs).unwrap();
            let mut encoder = context.device().create_command_encoder(&Default::default());
            mass.encode_with_inverse(&mut encoder);
            response.encode(&mut encoder);
            let _ = context.queue().submit(Some(encoder.finish()));
            let changed = response
                .readback(context.device(), context.queue())
                .unwrap();
            assert!((changed[0].responses[(0, 0)] - 1.0 / 3.0).abs() < 1e-6);
            assert_eq!(changed[0].responses[(0, 1)], 0.0);
            assert_eq!(changed[1].responses, responses[1].responses);
        }
        assert!(tested > 0, "required GPU backend unavailable");
    }

    #[test]
    fn prescribed_gpu_scalar_motion_survives_real_contact_and_effort() {
        let link = |mass| LinkSpec {
            mass,
            center_of_mass: Vector3::zeros(),
            inertia: Matrix3::identity(),
        };
        let articulation = Articulation::new(
            vec![link(0.0), link(2.0), link(1.0)],
            (0..2)
                .map(|parent| JointSpec {
                    parent,
                    child: parent + 1,
                    kind: JointKind::Prismatic,
                    origin: Isometry3::identity(),
                    axis: Vector3::z(),
                    limits: None,
                })
                .collect(),
            0,
        )
        .unwrap();
        let initial = GpuGeneralizedState {
            positions: DVector::from_vec(vec![0.2, -0.2]),
            velocities: DVector::from_vec(vec![-1.0, 0.0]),
        };
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            eprintln!("prescribed scalar real contact integration: {backend:?}");
            let mut inputs = systems(&articulation, &[2, 2]);
            for system in &mut inputs {
                system.force = DVector::from_vec(vec![1000.0, -3.0]);
            }
            let mut mass =
                GpuArticulatedMassAssemblyBatch::new(context.device(), context.queue(), &inputs)
                    .unwrap();
            mass.set_prescribed_coordinates(&[vec![true, false], vec![false, false]])
                .unwrap();
            let state = GpuGeneralizedStateBatch::from_assembly_batch(
                &mass,
                &[initial.clone(), initial.clone()],
            )
            .unwrap();
            let arts = [&articulation; 2];
            let poses =
                GpuArticulatedPoseBatch::new(&state, &arts, &[Isometry3::identity(); 2]).unwrap();
            let terms = GpuArticulatedLinkTermsBatch::new(&poses, &mass, &arts).unwrap();
            let contact = GpuArticulatedGroundContactBatch::new(
                &state,
                &poses,
                &mass,
                &arts,
                &[
                    vec![GpuArticulatedGroundSphere {
                        link: 2,
                        local_center: Vector3::zeros(),
                        radius: 0.0,
                        plane_normal: Vector3::z(),
                        plane_offset: 0.0,
                        plane_xy_half_extent: None,
                        restitution: 0.0,
                        friction: 0.0,
                    }],
                    vec![],
                ],
                0.01,
            )
            .unwrap();
            let mut encoder = context.device().create_command_encoder(&Default::default());
            poses.encode(&mut encoder);
            terms.encode(&mut encoder);
            mass.encode_with_inverse(&mut encoder);
            contact.encode(&mut encoder);
            state.encode_step(&mut encoder, 0.01).unwrap();
            let _ = context.queue().submit(Some(encoder.finish()));
            let actual = state.readback().unwrap();
            let acceleration = mass.readback().unwrap();
            assert_eq!(acceleration[0][0], 0.0);
            assert_eq!(actual[0].velocities[0], -1.0);
            assert!((actual[0].positions[0] - 0.19).abs() < 1e-6);
            // Contact cancels the *total* closing velocity, including the
            // prescribed parent's motion, using only the dynamic child response.
            assert!((actual[0].velocities[1] - 1.0).abs() < 1e-5);
            assert!((actual[0].positions[1] + 0.19).abs() < 1e-6);
            assert!(acceleration[0][1] > 99.0);
            let unrestricted = DMatrix::from_row_slice(2, 2, &[3.0, 1.0, 1.0, 1.0])
                .lu()
                .solve(&inputs[1].force)
                .unwrap();
            assert!((&acceleration[1] - &unrestricted).norm() < 1e-4);
            let expected_velocity = &initial.velocities + unrestricted * 0.01;
            assert!((&actual[1].velocities - &expected_velocity).norm() < 1e-5);
            assert!(
                (&actual[1].positions - (&initial.positions + expected_velocity * 0.01)).norm()
                    < 1e-6
            );
        }
        assert!(tested > 0, "required GPU backend unavailable");
    }

    #[test]
    fn prescribed_gpu_tangent_and_root_velocities_feed_bias_and_integrate() {
        let link = |mass| LinkSpec {
            mass,
            center_of_mass: Vector3::new(0.35, 0.12, -0.08),
            inertia: Matrix3::from_diagonal(&Vector3::new(0.2, 0.3, 0.5)),
        };
        let articulation = Articulation::new(
            vec![link(2.0), link(1.8)],
            vec![JointSpec {
                parent: 0,
                child: 1,
                kind: JointKind::Spherical,
                origin: Isometry3::translation(0.2, 0.3, -0.1),
                axis: Vector3::z(),
                limits: None,
            }],
            0,
        )
        .unwrap();
        let orientation = UnitQuaternion::from_euler_angles(0.3, 1.5, -0.4);
        let root = Isometry3::translation(0.4, -0.2, 0.3)
            * Isometry3::rotation(Vector3::new(0.1, -0.2, 0.3));
        let initial = [
            GpuGeneralizedState {
                positions: DVector::from_element(3, 123.0),
                velocities: DVector::from_vec(vec![0.8, -0.5, 0.6]),
            },
            GpuGeneralizedState {
                positions: DVector::from_element(9, 123.0),
                velocities: DVector::from_vec(vec![0.2, -0.3, 0.4, -0.7, 0.9, 0.3, 0.8, -0.5, 0.6]),
            },
        ];
        let masks = [
            vec![true, false, true],
            vec![true, true, true, true, true, true, true, false, true],
        ];
        let gravity = Vector3::new(0.0, 0.0, -9.81);
        let dt = 0.01;
        let efforts = [
            DVector::from_vec(vec![80.0, 0.2, -90.0]),
            DVector::from_fn(9, |i, _| if i == 7 { 0.2 } else { 100.0 }),
        ];
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            eprintln!("prescribed tangent/root bias and integration: {backend:?}");
            let mut inputs = systems(&articulation, &[3, 9]);
            for input in &mut inputs {
                input.armature.fill(0.2);
            }
            let mut mass =
                GpuArticulatedMassAssemblyBatch::new(context.device(), context.queue(), &inputs)
                    .unwrap();
            mass.set_prescribed_coordinates(&masks).unwrap();
            let mut state = GpuGeneralizedStateBatch::from_assembly_batch(&mass, &initial).unwrap();
            state
                .set_position_integration(&[vec![false; 3], vec![false; 9]])
                .unwrap();
            let spherical = GpuArticulatedSphericalBatch::new(
                &state,
                &[
                    vec![GpuSphericalJointState {
                        velocity_slot: 0,
                        orientation,
                    }],
                    vec![GpuSphericalJointState {
                        velocity_slot: 6,
                        orientation,
                    }],
                ],
                dt,
            )
            .unwrap();
            let arts = [&articulation; 2];
            let flags = [false, true];
            let poses = GpuArticulatedPoseBatch::new_with_spherical_state(
                &state, &arts, &[root; 2], &flags, &spherical,
            )
            .unwrap();
            let terms = GpuArticulatedLinkTermsBatch::new(&poses, &mass, &arts).unwrap();
            let forces = GpuArticulatedForceBatch::new(&mass, &efforts, &[gravity; 2]).unwrap();
            let bias = GpuArticulatedVelocityBiasBatch::new_with_spherical_state(
                &state, &mass, &forces, &arts, &poses, &spherical,
            )
            .unwrap();
            let root_integration = GpuArticulatedRootBatch::new(&poses, &state, dt).unwrap();
            let mut encoder = context.device().create_command_encoder(&Default::default());
            poses.encode(&mut encoder);
            terms.encode(&mut encoder);
            bias.encode(&mut encoder);
            forces.encode(&mut encoder);
            mass.encode_with_inverse(&mut encoder);
            state.encode_step(&mut encoder, dt).unwrap();
            spherical.encode(&mut encoder);
            root_integration.encode(&mut encoder);
            poses.encode(&mut encoder);
            let _ = context.queue().submit(Some(encoder.finish()));
            let actual = state.readback().unwrap();
            let acceleration = mass.readback().unwrap();
            let actual_poses = poses.readback().unwrap();
            let orientations = spherical.readback().unwrap();
            for environment in 0..2 {
                let initial = &initial[environment];
                let n = initial.velocities.len();
                let start = if flags[environment] { 6 } else { 0 };
                let dynamics = articulation
                    .generalized_dynamics_with_spherical_orientations(
                        root,
                        &[123.0; 3],
                        &[Some(orientation)],
                        &initial.velocities,
                        flags[environment],
                        gravity,
                    )
                    .unwrap();
                let mut zeroed = initial.velocities.clone();
                for (i, &prescribed) in masks[environment].iter().enumerate() {
                    if prescribed {
                        zeroed[i] = 0.0;
                    }
                }
                let without_motion = articulation
                    .generalized_dynamics_with_spherical_orientations(
                        root,
                        &[123.0; 3],
                        &[Some(orientation)],
                        &zeroed,
                        flags[environment],
                        gravity,
                    )
                    .unwrap();
                assert!(
                    (dynamics.velocity_bias[start + 1] - without_motion.velocity_bias[start + 1])
                        .abs()
                        > 0.01
                );
                let (expected, _) = solve_prescribed_mass(
                    &(dynamics.mass + DMatrix::identity(n, n) * 0.2),
                    &(&efforts[environment] + dynamics.gravity_force - dynamics.velocity_bias),
                    &masks[environment],
                )
                .unwrap();
                assert!((&acceleration[environment] - &expected).norm() < 0.003);
                let velocity = &initial.velocities + expected * dt;
                assert!((&actual[environment].velocities - &velocity).norm() < 4e-5);
                assert_eq!(actual[environment].positions, initial.positions);
                for (i, &prescribed) in masks[environment].iter().enumerate() {
                    if prescribed {
                        assert_eq!(acceleration[environment][i], 0.0);
                    }
                }
                let expected_orientation = UnitQuaternion::from_scaled_axis(
                    Vector3::new(velocity[start], velocity[start + 1], velocity[start + 2]) * dt,
                ) * orientation;
                assert!(
                    (orientations[environment][0].inverse() * expected_orientation).angle() < 2e-5
                );
                let mut expected_root = root;
                if flags[environment] {
                    expected_root.translation.vector +=
                        Vector3::new(velocity[0], velocity[1], velocity[2]) * dt;
                    expected_root.rotation = UnitQuaternion::from_scaled_axis(
                        Vector3::new(velocity[3], velocity[4], velocity[5]) * dt,
                    ) * root.rotation;
                }
                assert!(
                    (actual_poses[environment][0].translation.vector
                        - expected_root.translation.vector)
                        .norm()
                        < 2e-6
                );
                assert!(
                    (actual_poses[environment][0].rotation.inverse() * expected_root.rotation)
                        .angle()
                        < 2e-5
                );
            }
        }
        assert!(tested > 0, "required GPU backend unavailable");
    }
}
