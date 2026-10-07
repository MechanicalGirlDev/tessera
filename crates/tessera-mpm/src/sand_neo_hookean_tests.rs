use super::*;
use crate::{MpmParams, MpmParticle, MpmWorld};
use nalgebra::UnitQuaternion;

fn model() -> MaterialModel {
    MaterialModel::sand_neo_hookean(4_000.0, 0.2, 35.0f64.to_radians(), 0.0)
}

fn state() -> PlasticState {
    PlasticState {
        hardening: 1.0,
        ..PlasticState::default()
    }
}

#[test]
fn sand_neo_hookean_stress_vanishes_for_identity_and_rotation() {
    let rotation = UnitQuaternion::from_euler_angles(0.3, -0.4, 0.7)
        .to_rotation_matrix()
        .into_inner();
    for deformation in [Matrix3::identity(), rotation] {
        let stress = model().kirchhoff_stress(deformation, Matrix3::zeros(), state());
        assert!(stress.norm() < 1e-10);
    }
}

#[test]
fn cohesive_rotation_projection_is_stable_after_f32_upload() {
    // Given: an exact rotation and its actual GPU input representation.
    let rotation = UnitQuaternion::from_euler_angles(0.2, -0.4, 0.7)
        .to_rotation_matrix()
        .into_inner();
    let uploaded = rotation.map(|value| f64::from(value as f32));
    let material = MaterialModel::sand_neo_hookean(4_000.0, 0.2, 35.0f64.to_radians(), 0.02);

    // When: the same return mapping processes each representation.
    let (exact, exact_state) = material.project_deformation(rotation, state());
    let (rounded, rounded_state) = material.project_deformation(uploaded, state());

    // Then: roundoff cannot select different plastic branches.
    assert!((exact - rounded).norm() < 2e-6);
    assert!((exact_state.plastic_det - rounded_state.plastic_det).abs() < 2e-6);
    assert!((exact_state.log_volume_gain - rounded_state.log_volume_gain).abs() < 2e-6);
}

#[test]
fn sand_neo_hookean_stress_matches_reference_under_deformation_and_rotation() {
    let deformation = Matrix3::new(0.9, 0.2, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.1);
    let rotation = UnitQuaternion::from_euler_angles(0.2, -0.6, 0.3)
        .to_rotation_matrix()
        .into_inner();
    // Independent Lamé values for E=4000, nu=0.2.
    let expected = deformation * deformation.transpose() * (5_000.0 / 3.0)
        + Matrix3::identity() * ((10_000.0 / 9.0) * 0.792f64.ln() - 5_000.0 / 3.0);
    let stress = model().kirchhoff_stress(deformation, Matrix3::zeros(), state());
    let rotated = model().kirchhoff_stress(rotation * deformation, Matrix3::zeros(), state());
    assert!((stress - expected).norm() < 1e-10);
    assert!((rotated - rotation * expected * rotation.transpose()).norm() < 1e-10);
    let linear = MaterialModel::sand(4_000.0, 0.2, 35.0f64.to_radians(), 0.0).kirchhoff_stress(
        deformation,
        Matrix3::zeros(),
        state(),
    );
    assert!((stress - linear).norm() > 100.0);
}

#[test]
fn sand_neo_hookean_tensile_apex_updates_all_reference_plastic_state() {
    let model = MaterialModel::sand_neo_hookean(4_000.0, 0.2, 35.0f64.to_radians(), 0.06);
    let initial = PlasticState {
        plastic_det: 2.0,
        hardening: 1.5,
        log_volume_gain: 0.03,
    };
    let (projected, plastic) = model.project_deformation(Matrix3::identity() * 1.2, initial);
    assert!((projected - Matrix3::identity() * 0.02f64.exp()).norm() < 1e-12);
    assert!((plastic.plastic_det - 2.0 * 1.2f64.powi(3) / 0.06f64.exp()).abs() < 1e-12);
    assert!((plastic.log_volume_gain - (0.03 + 3.0 * 1.2f64.ln() - 0.06)).abs() < 1e-12);
    let expected_q = 1.5 + 3.0f64.sqrt() * (1.2f64.ln() + 0.01);
    assert!((plastic.hardening - expected_q).abs() < 1e-12);
}

#[test]
fn sand_neo_hookean_shear_return_uses_accumulated_hardening() {
    let strain = Vector3::new(0.15, -0.25, -0.1);
    let deformation = Matrix3::from_diagonal(&strain.map(f64::exp));
    for q in [1.0, 4.0] {
        let initial = PlasticState {
            hardening: q,
            ..state()
        };
        let (projected, plastic) = model().project_deformation(deformation, initial);
        let trace = -0.2;
        let deviatoric = strain - Vector3::repeat(trace / 3.0);
        let angle = 35.0f64.to_radians()
            + (9.0f64.to_radians() * q - 10.0f64.to_radians()) * (-0.2 * q).exp();
        let alpha = (2.0 / 3.0f64).sqrt() * 2.0 * angle.sin() / (3.0 - angle.sin());
        let gamma = deviatoric.norm() + 2.0 * trace * alpha;
        let expected = (strain - deviatoric * (gamma / deviatoric.norm())).map(f64::exp);
        assert!((projected - Matrix3::from_diagonal(&expected)).norm() < 1e-12);
        assert!((plastic.hardening - q - gamma).abs() < 1e-12);
        assert!((plastic.plastic_det - 1.0).abs() < 1e-12);
        assert!(plastic.log_volume_gain.abs() < 1e-12);
    }
}

#[test]
fn sand_neo_hookean_elastic_region_and_disabled_plasticity_preserve_state() {
    let deformation = Matrix3::from_diagonal(&Vector3::new(0.8, 0.79, 0.81));
    assert_eq!(
        model().project_deformation(deformation, state()),
        (deformation, state())
    );
    let disabled = MaterialModel::sand_neo_hookean(4_000.0, 0.0, 0.5, 0.0);
    let tension = Matrix3::identity() * 1.2;
    assert_eq!(
        disabled.project_deformation(tension, state()),
        (tension, state())
    );
}

#[test]
fn sand_neo_hookean_validation_rejects_nonphysical_parameters() {
    for (young, poisson, angle, cohesion) in [
        (-1.0, 0.2, 0.5, 0.0),
        (f64::NAN, 0.2, 0.5, 0.0),
        (1.0, 0.5, 0.5, 0.0),
        (1.0, 0.2, -0.1, 0.0),
        (1.0, 0.2, core::f64::consts::FRAC_PI_2, 0.0),
        (1.0, 0.2, f64::NAN, 0.0),
        (1.0, 0.2, 0.5, -0.1),
        (1.0, 0.2, 0.5, f64::INFINITY),
    ] {
        assert!(!MaterialModel::sand_neo_hookean(young, poisson, angle, cohesion).is_valid());
    }
    assert!(model().is_valid());
}

#[test]
fn sand_neo_hookean_particle_initialization_and_cfl_match_reference() {
    let mut particle = MpmParticle::new(Vector3::repeat(0.5), 0.04, 1_000.0, model());
    assert_eq!(particle.plastic, state());
    particle.deformation = Matrix3::identity() * 2.0;
    let params = MpmParams {
        cell_width: 0.1,
        max_substep: 1.0,
        ..MpmParams::default()
    };
    let expected_wave_speed = ((40_000.0f64 / 9.0) / (1_000.0 / 8.0)).sqrt();
    let world = MpmWorld::new(vec![particle], params).unwrap();
    assert!((world.stable_timestep() - 0.05 / expected_wave_speed).abs() < 1e-12);
    let moving =
        model().timestep_bound_with_deformation(1_000.0, Vector3::new(10.0, 0.0, 0.0), 0.1, 8.0);
    assert!((moving - 0.005).abs() < 1e-12);
}
