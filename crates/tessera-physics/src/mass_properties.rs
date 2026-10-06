//! Analytic solid primitive mass properties in shape-local coordinates.

use nalgebra::{Matrix3, Vector3};

/// Mass properties about the center of mass, separate from the shape origin.
#[derive(Debug, Clone, PartialEq)]
pub struct MassProperties {
    /// Geometric volume independent of the supplied mass.
    pub volume: f64,
    /// Local center of mass relative to the shape frame.
    pub center_of_mass: Vector3<f64>,
    /// Inertia tensor about the center of mass in the shape frame.
    pub inertia: Matrix3<f64>,
}

/// Invalid dimensions, mass, or unrepresentable computed properties.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error("invalid solid primitive mass properties")]
pub struct MassPropertiesError;

impl MassProperties {
    /// A solid Z-aligned cone with base at `-half_height` and apex at
    /// `+half_height`. Zero mass is accepted for stationary shapes.
    pub fn solid_cone(
        radius: f64,
        half_height: f64,
        mass: f64,
    ) -> Result<Self, MassPropertiesError> {
        if !radius.is_finite()
            || radius <= 0.0
            || !half_height.is_finite()
            || half_height <= 0.0
            || !mass.is_finite()
            || mass < 0.0
        {
            return Err(MassPropertiesError);
        }
        let volume = core::f64::consts::PI * radius * radius * (2.0 * half_height / 3.0);
        // Ixx = Iyy = 3m r^2/20 + 3m H^2/80 about the centroid.
        let radial = (0.15 * mass) * radius * radius;
        let transverse = radial + (0.15 * mass) * half_height * half_height;
        let axial = (0.3 * mass) * radius * radius;
        if !volume.is_finite()
            || volume <= 0.0
            || !transverse.is_finite()
            || !axial.is_finite()
            || (mass > 0.0 && (transverse <= 0.0 || axial <= 0.0))
        {
            return Err(MassPropertiesError);
        }
        Ok(Self {
            volume,
            center_of_mass: Vector3::new(0.0, 0.0, -0.5 * half_height),
            inertia: Matrix3::from_diagonal(&Vector3::new(transverse, transverse, axial)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cone_inertia_is_about_centroid_not_geometric_center() {
        let properties = MassProperties::solid_cone(2.0, 2.0, 10.0).unwrap();
        assert_eq!(properties.center_of_mass, Vector3::new(0.0, 0.0, -1.0));
        assert_eq!(properties.inertia, Matrix3::identity() * 12.0);
        assert!((properties.volume - 16.0 * core::f64::consts::PI / 3.0).abs() < 1e-12);
        assert_eq!(
            MassProperties::solid_cone(2.0, 2.0, 0.0).unwrap().inertia,
            Matrix3::zeros()
        );
    }

    #[test]
    fn cone_rejects_nonfinite_overflow_and_degenerate_properties() {
        for (r, h, m) in [
            (0.0, 1.0, 1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, -1.0),
            (f64::NAN, 1.0, 1.0),
            (1.0, f64::INFINITY, 1.0),
            (1.0, 1.0, f64::INFINITY),
            (f64::MAX, 1.0, 1.0),
            (1e-300, 1e-300, 1.0),
        ] {
            assert!(MassProperties::solid_cone(r, h, m).is_err());
        }
    }
}
