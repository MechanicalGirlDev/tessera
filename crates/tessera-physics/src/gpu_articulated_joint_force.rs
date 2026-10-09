//! Device-resident generalized drives for articulated steps.

use core::mem::size_of;

use nalgebra::DVector;
use wgpu::util::DeviceExt;

use crate::articulated_world::{JointMotor, JointNonlinearPassive, JointPassive};
use crate::gpu_articulated_force::GpuArticulatedForceBatch;
use crate::gpu_articulated_state::GpuGeneralizedStateBatch;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PackedJointForce {
    passive: [f32; 4],
    nonlinear: [f32; 4],
    motor_targets: [f32; 4],
    motor_gains: [f32; 4],
    implicit: [f32; 4],
}

#[derive(Debug, Clone)]
struct ImplicitSettings {
    dt: f32,
    armature: Vec<f32>,
    vector_offsets: Vec<u32>,
}

/// One coordinate's CPU supplied baseline and device-evaluated drive terms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuJointForceInput {
    /// Applied effort before GPU velocity bias, or after caller bias if no GPU bias pass is used.
    pub base_force: f64,
    /// Uncapped linear spring and damping.
    pub passive: JointPassive,
    /// Uncapped nonlinear spring and damping.
    pub nonlinear: JointNonlinearPassive,
    /// Optional position or velocity drive with an effort cap.
    pub motor: Option<JointMotor>,
}

impl Default for GpuJointForceInput {
    fn default() -> Self {
        Self {
            base_force: 0.0,
            passive: JointPassive::default(),
            nonlinear: JointNonlinearPassive::default(),
            motor: None,
        }
    }
}

/// Invalid joint parameters or a GPU capacity limit.
#[derive(Debug, thiserror::Error)]
pub enum GpuArticulatedJointForceError {
    /// The state, force batch, or per-coordinate parameters have incompatible layouts.
    #[error("invalid articulated joint force input")]
    InvalidInput,
    /// Packed joint parameters exceed device limits.
    #[error("articulated joint forces exceed GPU capacity")]
    Capacity,
}

/// Builds base generalized forces from resident positions and velocities.
///
/// Its output replaces the base force vector of a bound `GpuArticulatedForceBatch`.
/// Encode this pass before gravity and link load projection. The spring, damping,
/// nonlinear passive, and motor formulas match the CPU force evaluation. Use
/// `new_implicit` to also write CPU-matching passive tangent corrections to the
/// mass assembly's armature and force vector before gravity projection.
/// For a floating layout, prepend six inputs containing only world-frame base
/// force/torque efforts; joint passive and motor inputs follow those six slots.
/// Root workspace positions are not joint or Euler root coordinates.
#[derive(Debug)]
pub struct GpuArticulatedJointForceBatch {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    parameters: wgpu::Buffer,
    positions: wgpu::Buffer,
    velocities: wgpu::Buffer,
    owners: wgpu::Buffer,
    status: wgpu::Buffer,
    output: wgpu::Buffer,
    vectors: wgpu::Buffer,
    implicit: Option<ImplicitSettings>,
    dimensions: Vec<usize>,
    coordinate_count: usize,
}

impl GpuArticulatedJointForceBatch {
    pub(crate) fn parameter_buffer(&self) -> &wgpu::Buffer {
        &self.parameters
    }

    /// Bind matching state and force batches with one parameter per coordinate.
    pub fn new(
        state: &GpuGeneralizedStateBatch,
        forces: &GpuArticulatedForceBatch,
        inputs: &[Vec<GpuJointForceInput>],
    ) -> Result<Self, GpuArticulatedJointForceError> {
        Self::build(state, forces, inputs, None)
    }

    /// Bind an implicit contact substep with baseline armature and positive step time.
    ///
    /// The armature is the uncorrected diagonal for each system. Each encode
    /// reconstructs the effective diagonal from this baseline, including when
    /// multiple resident steps are encoded into one command buffer.
    pub fn new_implicit(
        state: &GpuGeneralizedStateBatch,
        forces: &GpuArticulatedForceBatch,
        inputs: &[Vec<GpuJointForceInput>],
        armature: &[DVector<f64>],
        dt: f64,
    ) -> Result<Self, GpuArticulatedJointForceError> {
        let dimensions = forces.dimensions();
        if !dt.is_finite() || dt <= 0.0 || armature.len() != dimensions.len() {
            return Err(GpuArticulatedJointForceError::InvalidInput);
        }
        let dt = finite_f32(dt)?;
        let mut packed_armature = Vec::new();
        let mut vector_offsets = Vec::new();
        let mut vector_offset = 0usize;
        for (&dimension, values) in dimensions.iter().zip(armature) {
            if values.len() != dimension {
                return Err(GpuArticulatedJointForceError::InvalidInput);
            }
            for value in values.iter() {
                packed_armature.push(finite_f32(*value)?);
                vector_offsets.push(
                    u32::try_from(vector_offset)
                        .map_err(|_| GpuArticulatedJointForceError::Capacity)?,
                );
                vector_offset += 1;
            }
            vector_offset = vector_offset
                .checked_add(dimension)
                .ok_or(GpuArticulatedJointForceError::Capacity)?;
        }
        Self::build(
            state,
            forces,
            inputs,
            Some(ImplicitSettings {
                dt,
                armature: packed_armature,
                vector_offsets,
            }),
        )
    }

    fn build(
        state: &GpuGeneralizedStateBatch,
        forces: &GpuArticulatedForceBatch,
        inputs: &[Vec<GpuJointForceInput>],
        implicit: Option<ImplicitSettings>,
    ) -> Result<Self, GpuArticulatedJointForceError> {
        let dimensions = forces.dimensions().to_vec();
        let parameters = pack_inputs(&dimensions, inputs, implicit.as_ref())?;
        if state.ranges().len() != dimensions.len()
            || state
                .ranges()
                .iter()
                .zip(&dimensions)
                .any(|(range, &n)| range.len() != n)
        {
            return Err(GpuArticulatedJointForceError::InvalidInput);
        }
        let mut owners = Vec::with_capacity(parameters.len());
        for (system, range) in state.ranges().iter().enumerate() {
            owners.extend(core::iter::repeat_n(
                u32::try_from(system).map_err(|_| GpuArticulatedJointForceError::Capacity)?,
                range.len(),
            ));
        }
        let count = parameters.len();
        let device = state.device();
        let limits = device.limits();
        let param_bytes = count
            .checked_mul(size_of::<PackedJointForce>())
            .ok_or(GpuArticulatedJointForceError::Capacity)? as u64;
        let owner_bytes = count
            .checked_mul(size_of::<u32>())
            .ok_or(GpuArticulatedJointForceError::Capacity)? as u64;
        if owner_bytes != forces.base_force_buffer().size()
            || [param_bytes, owner_bytes].into_iter().any(|bytes| {
                bytes > limits.max_buffer_size
                    || bytes > u64::from(limits.max_storage_buffer_binding_size)
            })
            || count.div_ceil(64) > limits.max_compute_workgroups_per_dimension as usize
            || limits.max_storage_buffers_per_shader_stage < 7
        {
            return Err(GpuArticulatedJointForceError::Capacity);
        }
        let parameters = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera articulated joint force parameters"),
            contents: bytemuck::cast_slice(&parameters),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let owners = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera articulated joint force owners"),
            contents: bytemuck::cast_slice(&owners),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera articulated joint force evaluation"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("gpu_articulated_joint_force.wgsl").into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tessera articulated joint force evaluation"),
            layout: None,
            module: &module,
            entry_point: Some("assemble_joint_forces"),
            compilation_options: Default::default(),
            cache: None,
        });
        Ok(Self {
            device: device.clone(),
            queue: state.queue().clone(),
            pipeline,
            parameters,
            positions: state.position_buffer().clone(),
            velocities: state.velocity_buffer().clone(),
            owners,
            status: state.status_buffer().clone(),
            output: forces.base_force_buffer().clone(),
            vectors: forces.vectors_buffer().clone(),
            implicit,
            dimensions,
            coordinate_count: count,
        })
    }

    /// Replace all joint parameters without reallocating the GPU buffers.
    pub fn update_inputs(
        &self,
        inputs: &[Vec<GpuJointForceInput>],
    ) -> Result<(), GpuArticulatedJointForceError> {
        let packed = pack_inputs(&self.dimensions, inputs, self.implicit.as_ref())?;
        self.queue
            .write_buffer(&self.parameters, 0, bytemuck::cast_slice(&packed));
        Ok(())
    }

    /// Encode force evaluation before link gravity and mass assembly.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let buffers = [
            &self.parameters,
            &self.positions,
            &self.velocities,
            &self.owners,
            &self.status,
            &self.output,
            &self.vectors,
        ];
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera articulated joint force bindings"),
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
            label: Some("Tessera articulated joint force evaluation"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(self.coordinate_count.div_ceil(64) as u32, 1, 1);
    }
}

fn pack_inputs(
    dimensions: &[usize],
    inputs: &[Vec<GpuJointForceInput>],
    implicit: Option<&ImplicitSettings>,
) -> Result<Vec<PackedJointForce>, GpuArticulatedJointForceError> {
    if dimensions.is_empty() || dimensions.len() != inputs.len() {
        return Err(GpuArticulatedJointForceError::InvalidInput);
    }
    let mut packed = Vec::new();
    for (&dimension, environment) in dimensions.iter().zip(inputs) {
        if dimension != environment.len() {
            return Err(GpuArticulatedJointForceError::InvalidInput);
        }
        for input in environment {
            let passive = input.passive;
            let nonlinear = input.nonlinear;
            if passive.stiffness < 0.0 || passive.damping < 0.0 {
                return Err(GpuArticulatedJointForceError::InvalidInput);
            }
            let (motor_targets, motor_gains) = if let Some(motor) = input.motor {
                if motor.stiffness < 0.0 || motor.damping < 0.0 || motor.max_force < 0.0 {
                    return Err(GpuArticulatedJointForceError::InvalidInput);
                }
                (
                    [
                        finite_f32(motor.position_target.unwrap_or(0.0))?,
                        finite_f32(motor.velocity_target)?,
                        finite_f32(motor.max_force)?,
                        1.0,
                    ],
                    [
                        finite_f32(motor.stiffness)?,
                        finite_f32(motor.damping)?,
                        if motor.position_target.is_some() {
                            1.0
                        } else {
                            0.0
                        },
                        0.0,
                    ],
                )
            } else {
                ([0.0; 4], [0.0; 4])
            };
            packed.push(PackedJointForce {
                passive: [
                    finite_f32(passive.stiffness)?,
                    finite_f32(passive.damping)?,
                    finite_f32(passive.rest_position)?,
                    finite_f32(input.base_force)?,
                ],
                nonlinear: [
                    finite_f32(nonlinear.spring_quadratic)?,
                    finite_f32(nonlinear.spring_cubic)?,
                    finite_f32(nonlinear.damping_quadratic)?,
                    finite_f32(nonlinear.damping_cubic)?,
                ],
                motor_targets,
                motor_gains,
                implicit: implicit.map_or([0.0; 4], |settings| {
                    let coordinate = packed.len();
                    [
                        settings.armature[coordinate],
                        f32::from_bits(settings.vector_offsets[coordinate]),
                        settings.dt,
                        1.0,
                    ]
                }),
            });
        }
    }
    Ok(packed)
}

fn finite_f32(value: f64) -> Result<f32, GpuArticulatedJointForceError> {
    let narrowed = value as f32;
    if !value.is_finite() || !narrowed.is_finite() || (value != 0.0 && narrowed == 0.0) {
        return Err(GpuArticulatedJointForceError::InvalidInput);
    }
    Ok(narrowed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_articulated_mass_assembly::GpuMassAssemblySystem;
    use crate::gpu_articulated_state::GpuGeneralizedState;
    use crate::gpu_contact_pipeline::GpuContactDevice;
    use nalgebra::DVector;

    fn expected(input: GpuJointForceInput, q: f64, v: f64) -> f64 {
        let displacement = q - input.passive.rest_position;
        let spring = -input.passive.stiffness * displacement
            - input.nonlinear.spring_quadratic * displacement.powi(2)
            - input.nonlinear.spring_cubic * displacement.powi(3);
        let damping = -input.passive.damping * v
            - input.nonlinear.damping_quadratic * v * v.abs()
            - input.nonlinear.damping_cubic * v.powi(3);
        let motor = input.motor.map_or(0.0, |motor| {
            (motor.stiffness * motor.position_target.map_or(0.0, |target| target - q)
                + motor.damping * (motor.velocity_target - v))
                .clamp(-motor.max_force, motor.max_force)
        });
        input.base_force + spring + damping + motor
    }

    #[test]
    fn gpu_joint_passive_nonlinear_and_capped_motor_match_cpu() {
        let systems = [
            GpuMassAssemblySystem {
                links: Vec::new(),
                armature: DVector::from_element(1, 2.0),
                force: DVector::zeros(1),
            },
            GpuMassAssemblySystem {
                links: Vec::new(),
                armature: DVector::from_column_slice(&[1.5, 3.0]),
                force: DVector::zeros(2),
            },
        ];
        let states = [
            GpuGeneralizedState {
                positions: DVector::from_element(1, 1.0),
                velocities: DVector::from_element(1, -0.5),
            },
            GpuGeneralizedState {
                positions: DVector::from_column_slice(&[0.2, -0.4]),
                velocities: DVector::from_column_slice(&[0.1, 0.3]),
            },
        ];
        let inputs = [
            vec![GpuJointForceInput {
                base_force: 1.0,
                passive: JointPassive {
                    stiffness: 4.0,
                    damping: 2.0,
                    rest_position: 0.25,
                },
                nonlinear: JointNonlinearPassive {
                    spring_quadratic: 1.0,
                    spring_cubic: -0.5,
                    damping_quadratic: 0.3,
                    damping_cubic: 0.1,
                },
                motor: Some(JointMotor {
                    position_target: Some(0.5),
                    velocity_target: 0.2,
                    stiffness: 10.0,
                    damping: 3.0,
                    max_force: 2.0,
                }),
            }],
            vec![
                GpuJointForceInput {
                    base_force: -0.75,
                    motor: Some(JointMotor {
                        position_target: None,
                        velocity_target: -1.0,
                        stiffness: 20.0,
                        damping: 2.0,
                        max_force: 1.0,
                    }),
                    ..GpuJointForceInput::default()
                },
                GpuJointForceInput::default(),
            ],
        ];
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let mass = crate::gpu_articulated_mass_assembly::GpuArticulatedMassAssemblyBatch::new(
                context.device(),
                context.queue(),
                &systems,
            )
            .unwrap();
            let state = GpuGeneralizedStateBatch::from_assembly_batch(&mass, &states).unwrap();
            let forces = GpuArticulatedForceBatch::new(
                &mass,
                &[DVector::zeros(1), DVector::zeros(2)],
                &[nalgebra::Vector3::zeros(); 2],
            )
            .unwrap();
            let joint = GpuArticulatedJointForceBatch::new(&state, &forces, &inputs).unwrap();
            let mut encoder = context.device().create_command_encoder(&Default::default());
            joint.encode(&mut encoder);
            forces.encode(&mut encoder);
            mass.encode(&mut encoder);
            let _ = context.queue().submit(Some(encoder.finish()));
            let result = mass.readback().unwrap();
            for system in 0..2 {
                for coordinate in 0..systems[system].force.len() {
                    let q = states[system].positions[coordinate];
                    let v = states[system].velocities[coordinate];
                    let force = expected(inputs[system][coordinate], q, v);
                    let acceleration = force / systems[system].armature[coordinate];
                    assert!((result[system][coordinate] - acceleration).abs() < 2e-5);
                }
            }
            assert!(
                joint
                    .update_inputs(&[
                        vec![GpuJointForceInput {
                            base_force: f64::NAN,
                            ..inputs[0][0]
                        }],
                        inputs[1].clone(),
                    ])
                    .is_err()
            );
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn two_resident_joint_drive_steps_use_updated_state() {
        let system = GpuMassAssemblySystem {
            links: Vec::new(),
            armature: DVector::from_element(1, 2.0),
            force: DVector::zeros(1),
        };
        let initial = GpuGeneralizedState {
            positions: DVector::from_element(1, 1.0),
            velocities: DVector::zeros(1),
        };
        let input = GpuJointForceInput {
            base_force: 0.0,
            passive: JointPassive {
                stiffness: 10.0,
                damping: 1.0,
                rest_position: 0.0,
            },
            motor: Some(JointMotor {
                position_target: Some(0.5),
                velocity_target: 0.0,
                stiffness: 4.0,
                damping: 2.0,
                max_force: 3.0,
            }),
            ..GpuJointForceInput::default()
        };
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let mass = crate::gpu_articulated_mass_assembly::GpuArticulatedMassAssemblyBatch::new(
                context.device(),
                context.queue(),
                core::slice::from_ref(&system),
            )
            .unwrap();
            let state = GpuGeneralizedStateBatch::from_assembly_batch(
                &mass,
                core::slice::from_ref(&initial),
            )
            .unwrap();
            let forces = GpuArticulatedForceBatch::new(
                &mass,
                &[DVector::zeros(1)],
                &[nalgebra::Vector3::zeros()],
            )
            .unwrap();
            let joint =
                GpuArticulatedJointForceBatch::new(&state, &forces, &[vec![input]]).unwrap();
            let mut encoder = context.device().create_command_encoder(&Default::default());
            for _ in 0..2 {
                joint.encode(&mut encoder);
                forces.encode(&mut encoder);
                mass.encode(&mut encoder);
                state.encode_step(&mut encoder, 0.1).unwrap();
            }
            let _ = context.queue().submit(Some(encoder.finish()));
            let actual = state.readback().unwrap();
            assert!((actual[0].velocities[0] + 1.068).abs() < 2e-5);
            assert!((actual[0].positions[0] - 0.8332).abs() < 2e-5);
        }
        assert!(tested > 0, "no GPU backend available");
    }

    #[test]
    fn gpu_implicit_passive_tangents_match_repeated_cpu_substeps() {
        let systems = [
            GpuMassAssemblySystem {
                links: Vec::new(),
                armature: DVector::from_column_slice(&[2.0, 3.0]),
                force: DVector::zeros(2),
            },
            GpuMassAssemblySystem {
                links: Vec::new(),
                armature: DVector::from_element(1, 1.5),
                force: DVector::zeros(1),
            },
        ];
        let states = [
            GpuGeneralizedState {
                positions: DVector::from_column_slice(&[0.6, -0.3]),
                velocities: DVector::from_column_slice(&[0.4, -0.2]),
            },
            GpuGeneralizedState {
                positions: DVector::from_element(1, 0.25),
                velocities: DVector::from_element(1, -0.5),
            },
        ];
        let inputs = [
            vec![
                GpuJointForceInput {
                    base_force: 0.7,
                    passive: JointPassive {
                        stiffness: 5.0,
                        damping: 0.8,
                        rest_position: 0.1,
                    },
                    nonlinear: JointNonlinearPassive {
                        spring_quadratic: -7.0,
                        spring_cubic: 0.5,
                        damping_quadratic: -0.2,
                        damping_cubic: 0.1,
                    },
                    motor: Some(JointMotor {
                        position_target: Some(0.2),
                        velocity_target: 0.0,
                        stiffness: 2.0,
                        damping: 0.5,
                        max_force: 1.0,
                    }),
                },
                GpuJointForceInput {
                    passive: JointPassive {
                        stiffness: 9.0,
                        damping: 1.0,
                        rest_position: 0.2,
                    },
                    ..GpuJointForceInput::default()
                },
            ],
            vec![GpuJointForceInput {
                passive: JointPassive {
                    stiffness: 2.0,
                    damping: 0.4,
                    rest_position: -0.1,
                },
                nonlinear: JointNonlinearPassive {
                    spring_quadratic: 0.3,
                    spring_cubic: -0.1,
                    damping_quadratic: 0.2,
                    damping_cubic: -0.05,
                },
                ..GpuJointForceInput::default()
            }],
        ];
        let dt = 0.03;
        let mut expected_states = states.clone();
        for _ in 0..3 {
            for system in 0..systems.len() {
                for (coordinate, &input) in inputs[system].iter().enumerate() {
                    let q = expected_states[system].positions[coordinate];
                    let v = expected_states[system].velocities[coordinate];
                    let displacement = q - input.passive.rest_position;
                    let spring_tangent = input.passive.stiffness
                        + 2.0 * input.nonlinear.spring_quadratic * displacement
                        + 3.0 * input.nonlinear.spring_cubic * displacement.powi(2);
                    let damping_tangent = input.passive.damping
                        + 2.0 * input.nonlinear.damping_quadratic * v.abs()
                        + 3.0 * input.nonlinear.damping_cubic * v.powi(2);
                    let mass = systems[system].armature[coordinate]
                        + dt * damping_tangent
                        + dt * dt * spring_tangent;
                    let force = expected(input, q, v) - dt * spring_tangent * v;
                    let new_v = v + dt * force / mass;
                    expected_states[system].velocities[coordinate] = new_v;
                    expected_states[system].positions[coordinate] = q + dt * new_v;
                }
            }
        }
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let mass = crate::gpu_articulated_mass_assembly::GpuArticulatedMassAssemblyBatch::new(
                context.device(),
                context.queue(),
                &systems,
            )
            .unwrap();
            let state = GpuGeneralizedStateBatch::from_assembly_batch(&mass, &states).unwrap();
            let forces = GpuArticulatedForceBatch::new(
                &mass,
                &[DVector::zeros(2), DVector::zeros(1)],
                &[nalgebra::Vector3::zeros(); 2],
            )
            .unwrap();
            let joint = GpuArticulatedJointForceBatch::new_implicit(
                &state,
                &forces,
                &inputs,
                &systems
                    .iter()
                    .map(|system| system.armature.clone())
                    .collect::<Vec<_>>(),
                dt,
            )
            .unwrap();
            let mut encoder = context.device().create_command_encoder(&Default::default());
            for _ in 0..3 {
                joint.encode(&mut encoder);
                forces.encode(&mut encoder);
                mass.encode(&mut encoder);
                state.encode_step(&mut encoder, dt).unwrap();
            }
            let _ = context.queue().submit(Some(encoder.finish()));
            let actual = state.readback().unwrap();
            for (actual, expected) in actual.iter().zip(&expected_states) {
                for (a, e) in actual.positions.iter().zip(expected.positions.iter()) {
                    assert!((a - e).abs() < 3e-5, "position: {a} vs {e}");
                }
                for (a, e) in actual.velocities.iter().zip(expected.velocities.iter()) {
                    assert!((a - e).abs() < 3e-5, "velocity: {a} vs {e}");
                }
            }
        }
        assert!(tested > 0, "no GPU backend available");
    }
}
