//! Device-resident root pose integration from world-frame generalized twists.

use crate::gpu_articulated_pose::GpuArticulatedPoseBatch;
use crate::gpu_articulated_state::GpuGeneralizedStateBatch;
use core::mem::size_of;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RootLayout {
    indices: [u32; 4],
}

/// Invalid root layout, timestep, or exhausted GPU resources.
#[derive(Debug, thiserror::Error)]
pub enum GpuArticulatedRootError {
    /// Root flags, source state layout, or timestep is invalid.
    #[error("invalid articulated root integration input")]
    InvalidInput,
    /// Root metadata exceeds the device capacity.
    #[error("articulated root integration exceeds GPU capacity")]
    Capacity,
}

/// Integrates root poses after generalized velocities have been updated.
///
/// Each floating environment reads its six leading velocity slots as a
/// world-frame linear/angular twist. Rotation uses a left quaternion exponential
/// update. Encode forward kinematics afterwards to refresh link poses.
/// This pass integrates poses; force, inertia, and contact responses must be
/// supplied by the generalized dynamics solve.
#[derive(Debug)]
pub struct GpuArticulatedRootBatch {
    device: wgpu::Device,
    pipeline: wgpu::ComputePipeline,
    roots: wgpu::Buffer,
    velocities: wgpu::Buffer,
    layouts: wgpu::Buffer,
    status: wgpu::Buffer,
    timestep: wgpu::Buffer,
    count: usize,
}

impl GpuArticulatedRootBatch {
    /// Bind root poses to matching fixed/floating generalized state layouts.
    pub fn new(
        poses: &GpuArticulatedPoseBatch,
        state: &GpuGeneralizedStateBatch,
        timestep: f64,
    ) -> Result<Self, GpuArticulatedRootError> {
        let dt = timestep as f32;
        if !timestep.is_finite()
            || !dt.is_finite()
            || dt <= 0.0
            || poses.floating_roots().len() != state.ranges().len()
            || poses.coordinate_buffer() != state.position_buffer()
        {
            return Err(GpuArticulatedRootError::InvalidInput);
        }
        let packed = state
            .ranges()
            .iter()
            .zip(poses.floating_roots())
            .enumerate()
            .map(|(environment, (range, &floating))| {
                if floating && range.len() < 6 {
                    return Err(GpuArticulatedRootError::InvalidInput);
                }
                Ok(RootLayout {
                    indices: [
                        u32::try_from(range.start)
                            .map_err(|_| GpuArticulatedRootError::Capacity)?,
                        u32::try_from(environment)
                            .map_err(|_| GpuArticulatedRootError::Capacity)?,
                        u32::from(floating),
                        0,
                    ],
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let device = state.device();
        let limits = device.limits();
        let bytes = packed
            .len()
            .checked_mul(size_of::<RootLayout>())
            .ok_or(GpuArticulatedRootError::Capacity)? as u64;
        if packed.is_empty()
            || bytes > limits.max_buffer_size
            || bytes > u64::from(limits.max_storage_buffer_binding_size)
            || packed.len().div_ceil(64) > limits.max_compute_workgroups_per_dimension as usize
            || limits.max_storage_buffers_per_shader_stage < 4
        {
            return Err(GpuArticulatedRootError::Capacity);
        }
        let layouts = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera floating root layouts"),
            contents: bytemuck::cast_slice(&packed),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let timestep = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera floating root timestep"),
            contents: bytemuck::cast_slice(&[dt, 0.0f32, 0.0, 0.0]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera floating root integration"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_articulated_root.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tessera floating root integration"),
            layout: None,
            module: &shader,
            entry_point: Some("advance_roots"),
            compilation_options: Default::default(),
            cache: None,
        });
        Ok(Self {
            device: device.clone(),
            pipeline,
            roots: poses.root_pose_buffer().clone(),
            velocities: state.velocity_buffer().clone(),
            layouts,
            status: state.status_buffer().clone(),
            timestep,
            count: packed.len(),
        })
    }

    /// Encode a root update without copying poses or velocities to the host.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let buffers = [
            &self.roots,
            &self.velocities,
            &self.layouts,
            &self.status,
            &self.timestep,
        ];
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera floating root bindings"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &buffers
                .iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Tessera floating root integration"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(self.count.div_ceil(64) as u32, 1, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::articulation::{Articulation, JointKind, JointSpec, LinkSpec};
    use crate::gpu_articulated_force::GpuArticulatedForceBatch;
    use crate::gpu_articulated_link_terms::GpuArticulatedLinkTermsBatch;
    use crate::gpu_articulated_mass::{
        GpuArticulatedMassBatch, GpuArticulatedMassSystem, read_buffer,
    };
    use crate::gpu_articulated_mass_assembly::{
        GpuArticulatedMassAssemblyBatch, GpuMassAssemblySystem, GpuMassLink,
    };
    use crate::gpu_articulated_state::GpuGeneralizedState;
    use crate::gpu_contact_pipeline::GpuContactDevice;
    use nalgebra::{DMatrix, DVector, Isometry3, Matrix3, Translation3, UnitQuaternion, Vector3};

    fn reference(
        art: &Articulation,
        root: Isometry3<f64>,
        q: &[f64],
        floating: bool,
    ) -> GpuMassAssemblySystem {
        let pose = art.pose(root, q).unwrap();
        let n = art.dof() + if floating { 6 } else { 0 };
        GpuMassAssemblySystem {
            links: (0..art.link_count())
                .map(|index| {
                    let link = art.link(index).unwrap();
                    let (linear_jacobian, angular_jacobian) = art
                        .generalized_point_jacobians(&pose, index, link.center_of_mass, floating)
                        .unwrap();
                    let r = pose.links[index].rotation.to_rotation_matrix();
                    GpuMassLink {
                        mass: link.mass,
                        inertia_world: r.matrix() * link.inertia * r.matrix().transpose(),
                        linear_jacobian,
                        angular_jacobian,
                    }
                })
                .collect(),
            armature: DVector::from_element(n, 0.2),
            force: DVector::from_fn(n, |i, _| 0.1 * (i + 1) as f64),
        }
    }

    #[test]
    fn resident_floating_roots_and_mass_terms_match_cpu() {
        let link = LinkSpec {
            mass: 2.0,
            center_of_mass: Vector3::new(0.2, -0.1, 0.3),
            inertia: Matrix3::from_diagonal(&Vector3::new(0.4, 0.5, 0.7)),
        };
        // Nonzero root index verifies that base columns use the actual root link.
        let joint = Articulation::new(
            vec![link.clone(), link.clone()],
            vec![JointSpec {
                parent: 1,
                child: 0,
                kind: JointKind::Revolute,
                axis: Vector3::z(),
                origin: Isometry3::translation(0.8, 0.1, -0.2),
                limits: None,
            }],
            1,
        )
        .unwrap();
        let pure = Articulation::new(vec![link], vec![], 0).unwrap();
        let roots = [
            Isometry3::translation(1.0, -2.0, 0.5)
                * Isometry3::rotation(Vector3::new(0.3, -0.2, 0.1)),
            Isometry3::translation(-1.0, 0.5, 2.0),
            Isometry3::identity(),
            Isometry3::rotation(Vector3::new(0.0, 0.3, 0.0)),
        ];
        let flags = [true, false, true, true];
        let arts = [&joint, &joint, &pure, &pure];
        let initial = vec![
            GpuGeneralizedState {
                positions: DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2]),
                velocities: DVector::from_vec(vec![0.1, -0.2, 0.3, 0.2, -0.3, 0.4, 0.05]),
            },
            GpuGeneralizedState {
                positions: DVector::from_vec(vec![-0.1]),
                velocities: DVector::from_vec(vec![0.05]),
            },
            GpuGeneralizedState {
                positions: DVector::zeros(6),
                velocities: DVector::from_vec(vec![0.1, 0.0, -0.2, 0.0, 0.0, 1e-4]),
            },
            GpuGeneralizedState {
                positions: DVector::zeros(6),
                velocities: DVector::from_vec(vec![0.0, 0.0, 0.0, 1000.0, 0.0, 0.0]),
            },
        ];
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            eprintln!("floating root integration and mass terms: {backend:?}");
            // Identity systems prescribe twists; this is not a floating dynamics solve.
            let systems = initial
                .iter()
                .map(|s| GpuArticulatedMassSystem {
                    mass: DMatrix::identity(s.positions.len(), s.positions.len()),
                    force: DVector::zeros(s.positions.len()),
                })
                .collect::<Vec<_>>();
            let direct =
                GpuArticulatedMassBatch::new(context.device(), context.queue(), &systems).unwrap();
            let state = GpuGeneralizedStateBatch::from_mass_batch(&direct, &initial).unwrap();
            assert!(GpuArticulatedPoseBatch::new(&state, &arts, &roots).is_err());
            assert!(
                GpuArticulatedPoseBatch::new_with_floating_roots(&state, &arts, &roots, &[true])
                    .is_err()
            );
            let poses =
                GpuArticulatedPoseBatch::new_with_floating_roots(&state, &arts, &roots, &flags)
                    .unwrap();
            assert!(GpuArticulatedRootBatch::new(&poses, &state, 0.0).is_err());
            let other_state = GpuGeneralizedStateBatch::from_mass_batch(&direct, &initial).unwrap();
            assert!(GpuArticulatedRootBatch::new(&poses, &other_state, 0.002).is_err());
            let advance = GpuArticulatedRootBatch::new(&poses, &state, 0.002).unwrap();
            let refs = arts
                .iter()
                .enumerate()
                .map(|(i, a)| {
                    reference(
                        a,
                        roots[i],
                        if i == 0 {
                            &[0.2]
                        } else if i == 1 {
                            &[-0.1]
                        } else {
                            &[]
                        },
                        flags[i],
                    )
                })
                .collect::<Vec<_>>();
            let assembly =
                GpuArticulatedMassAssemblyBatch::new(context.device(), context.queue(), &refs)
                    .unwrap();
            let terms = GpuArticulatedLinkTermsBatch::new(&poses, &assembly, &arts).unwrap();
            let gravity = Vector3::new(0.1, -0.2, -9.81);
            let forces = GpuArticulatedForceBatch::new(
                &assembly,
                &refs.iter().map(|r| r.force.clone()).collect::<Vec<_>>(),
                &[gravity; 4],
            )
            .unwrap();
            let mut encoder = context.device().create_command_encoder(&Default::default());
            for _ in 0..10 {
                direct.encode(&mut encoder);
                state.encode_step(&mut encoder, 0.002).unwrap();
                advance.encode(&mut encoder);
                poses.encode(&mut encoder);
            }
            terms.encode(&mut encoder);
            forces.encode(&mut encoder);
            assembly.encode(&mut encoder);
            let _ = context.queue().submit(Some(encoder.finish()));
            let actual = poses.readback().unwrap();
            let bytes = read_buffer(
                context.device(),
                context.queue(),
                assembly.link_terms_buffer(),
            )
            .unwrap();
            let floats = bytemuck::cast_slice::<u8, f32>(&bytes);
            let acceleration = assembly.readback().unwrap();
            let mut offset = 0;
            for i in 0..arts.len() {
                let root = if flags[i] {
                    let v = &initial[i].velocities;
                    Isometry3::from_parts(
                        Translation3::from(
                            roots[i].translation.vector + Vector3::new(v[0], v[1], v[2]) * 0.02,
                        ),
                        UnitQuaternion::from_scaled_axis(Vector3::new(v[3], v[4], v[5]) * 0.02)
                            * roots[i].rotation,
                    )
                } else {
                    roots[i]
                };
                let q = if i == 0 {
                    vec![0.201]
                } else if i == 1 {
                    vec![-0.099]
                } else {
                    vec![]
                };
                let cpu = arts[i].pose(root, &q).unwrap();
                for (gpu, expected) in actual[i].iter().zip(&cpu.links) {
                    assert!((gpu.translation.vector - expected.translation.vector).norm() < 2e-5);
                    assert!((gpu.rotation.inverse() * expected.rotation).angle() < 5e-5);
                }
                let expected = reference(arts[i], root, &q, flags[i]);
                let n = expected.force.len();
                let mut mass = DMatrix::identity(n, n) * 0.2;
                let mut rhs = expected.force.clone();
                for link in &expected.links {
                    rhs += link.linear_jacobian.transpose() * (link.mass * gravity);
                    let values = core::iter::once(link.mass)
                        .chain((0..3).flat_map(|r| (0..3).map(move |c| link.inertia_world[(r, c)])))
                        .chain(
                            (0..3).flat_map(|r| (0..n).map(move |c| link.linear_jacobian[(r, c)])),
                        )
                        .chain(
                            (0..3).flat_map(|r| (0..n).map(move |c| link.angular_jacobian[(r, c)])),
                        );
                    for (j, value) in values.enumerate() {
                        assert!(
                            (f64::from(floats[offset + j]) - value).abs() < 5e-5,
                            "env {i}, scalar {j}"
                        );
                    }
                    offset += 10 + 6 * n;
                    mass += link.mass * link.linear_jacobian.transpose() * &link.linear_jacobian
                        + link.angular_jacobian.transpose()
                            * link.inertia_world
                            * &link.angular_jacobian;
                }
                assert!((&acceleration[i] - mass.lu().solve(&rhs).unwrap()).norm() < 1e-4);
            }
            assert!((actual[2][0].rotation.scaled_axis().z - 2e-6).abs() < 1e-8);

            // Overflow faults the environment before either root pose field is committed.
            let mut bad = initial.clone();
            bad[0].velocities[3] = 1e29;
            state.reset(&bad).unwrap();
            let before =
                read_buffer(context.device(), context.queue(), poses.root_pose_buffer()).unwrap();
            let mut encoder = context.device().create_command_encoder(&Default::default());
            advance.encode(&mut encoder);
            poses.encode(&mut encoder);
            let _ = context.queue().submit(Some(encoder.finish()));
            assert!(poses.readback().is_err());
            assert!(state.readback().is_err());
            let after =
                read_buffer(context.device(), context.queue(), poses.root_pose_buffer()).unwrap();
            assert_eq!(&before[..32], &after[..32]);
            state.reset(&initial).unwrap();
            let mut encoder = context.device().create_command_encoder(&Default::default());
            advance.encode(&mut encoder);
            poses.encode(&mut encoder);
            let _ = context.queue().submit(Some(encoder.finish()));
            assert!(poses.readback().is_ok());

            // A batch with no joints at all must still bind a valid topology buffer.
            let direct =
                GpuArticulatedMassBatch::new(context.device(), context.queue(), &systems[2..3])
                    .unwrap();
            let state = GpuGeneralizedStateBatch::from_mass_batch(&direct, &initial[2..3]).unwrap();
            let poses = GpuArticulatedPoseBatch::new_with_floating_roots(
                &state,
                &[&pure],
                &roots[2..3],
                &[true],
            )
            .unwrap();
            let advance = GpuArticulatedRootBatch::new(&poses, &state, 0.002).unwrap();
            let mut encoder = context.device().create_command_encoder(&Default::default());
            advance.encode(&mut encoder);
            poses.encode(&mut encoder);
            let _ = context.queue().submit(Some(encoder.finish()));
            assert!((poses.readback().unwrap()[0][0].rotation.scaled_axis().z - 2e-7).abs() < 1e-9);
        }
        assert!(tested > 0, "no GPU backend available");
    }
}
