//! Shared floating/spherical resident reset fixture.

use super::*;

#[derive(Debug)]
pub(super) struct Fixture {
    pub(super) context: GpuContactDevice,
    pub(super) mass: GpuArticulatedMassBatch,
    pub(super) state: GpuGeneralizedStateBatch,
    pub(super) spherical: GpuArticulatedSphericalBatch,
    pub(super) poses: GpuArticulatedPoseBatch,
    pub(super) roots: GpuArticulatedRootBatch,
    pub(super) initial: [GpuGeneralizedState; 2],
    pub(super) joints: [Vec<GpuSphericalJointState>; 2],
    pub(super) root_poses: [Isometry3<f64>; 2],
}

pub(super) fn build(backend: wgpu::Backends) -> Option<Fixture> {
    let link = LinkSpec {
        mass: 1.0,
        center_of_mass: Vector3::zeros(),
        inertia: Matrix3::identity(),
    };
    let art = Articulation::new(
        vec![link.clone(), link],
        vec![JointSpec {
            parent: 0,
            child: 1,
            kind: JointKind::Spherical,
            axis: Vector3::z(),
            origin: Isometry3::identity(),
            limits: None,
        }],
        0,
    )
    .unwrap();
    let initial = [
        GpuGeneralizedState {
            positions: DVector::from_element(9, 0.25),
            velocities: DVector::from_element(9, 0.5),
        },
        GpuGeneralizedState {
            positions: DVector::from_element(9, -0.5),
            velocities: DVector::from_element(9, -0.25),
        },
    ];
    let joints = [
        vec![GpuSphericalJointState {
            velocity_slot: 6,
            orientation: UnitQuaternion::from_scaled_axis(Vector3::new(0.1, 0.2, 0.3)),
        }],
        vec![GpuSphericalJointState {
            velocity_slot: 6,
            orientation: UnitQuaternion::from_scaled_axis(Vector3::new(-0.3, 0.2, -0.1)),
        }],
    ];
    let root_poses = [
        Isometry3::translation(1.0, 2.0, 3.0),
        Isometry3::translation(-3.0, -2.0, -1.0),
    ];
    let systems = initial
        .iter()
        .map(|_| GpuArticulatedMassSystem {
            mass: DMatrix::identity(9, 9),
            force: DVector::zeros(9),
        })
        .collect::<Vec<_>>();
    let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
        return None;
    };
    let mass = GpuArticulatedMassBatch::new(context.device(), context.queue(), &systems).unwrap();
    let mut state = GpuGeneralizedStateBatch::from_mass_batch(&mass, &initial).unwrap();
    state
        .set_position_integration(&[vec![false; 9], vec![false; 9]])
        .unwrap();
    let spherical = GpuArticulatedSphericalBatch::new(&state, &joints, 0.125).unwrap();
    let poses = GpuArticulatedPoseBatch::new_with_spherical_state(
        &state,
        &[&art, &art],
        &root_poses,
        &[true, true],
        &spherical,
    )
    .unwrap();
    let roots = GpuArticulatedRootBatch::new(&poses, &state, 0.125).unwrap();
    Some(Fixture {
        context,
        mass,
        state,
        spherical,
        poses,
        roots,
        initial,
        joints,
        root_poses,
    })
}
