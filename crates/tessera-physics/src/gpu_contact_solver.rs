//! WebGPU reference implementation of the generalized contact impulse solve.
//!
//! Independent contact islands run in separate workgroups. Contacts within one
//! island use dependency-preserving color waves.

use core::time::Duration;
use core::{mem::size_of, ops::Range};
use std::collections::BTreeMap;
use std::sync::mpsc;

use nalgebra::{DMatrix, DVector};
use wgpu::util::DeviceExt;

use crate::contact_reference::{
    ContactConstraint, ContactImpulse, ContactProblem, ContactSolution, ContactSolveError,
    SolveParams,
};
use crate::gpu_contact_mass_response::{
    GpuContactMassResponseBatch, GpuContactMassResponseError, GpuContactMassResponseSystem,
};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuContactData {
    effective: [f32; 4],
    targets: [f32; 4],
    offsets: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuSettings {
    dimensions: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuIslandRange {
    start_count: [u32; 4],
    color_start_count: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuColorRange {
    start_count: [u32; 2],
}

#[derive(Default)]
struct PreparedGpuProblem {
    velocity: Vec<f32>,
    jacobians: Vec<f32>,
    responses: Vec<f32>,
    contacts: Vec<GpuContactData>,
    seeds: Vec<[f32; 4]>,
    islands: Vec<GpuIslandRange>,
    colors: Vec<GpuColorRange>,
    island_contacts: Vec<u32>,
}

/// An independent generalized contact problem in a packed GPU solve.
#[derive(Debug, Clone, Copy)]
pub struct GpuContactSolveRequest<'a> {
    /// Generalized velocity, inverse mass, and contact constraints.
    pub problem: &'a ContactProblem,
    /// Optional impulses in the same contact order and frame.
    pub warm_start: Option<&'a [ContactImpulse]>,
}

/// One contact solve driven by an inverse mass matrix already on the GPU.
#[derive(Debug, Clone)]
pub struct GpuDeviceMassContactRequest<'a> {
    /// Device-resident row-major inverse matrix and trailing status value.
    pub inverse_mass: &'a wgpu::Buffer,
    /// Element range of the inverse matrix and status within its buffer.
    pub inverse_range: Range<usize>,
    /// Generalized velocity before contact impulses.
    pub velocity: &'a DVector<f64>,
    /// Contact and bilateral constraint rows in warm-start order.
    pub contacts: &'a [ContactConstraint],
    /// Contact projection and timestep settings.
    pub params: SolveParams,
    /// Optional impulses from the same contact frames.
    pub warm_start: Option<&'a [ContactImpulse]>,
}

/// Invalid contact input, device capacity, or GPU completion failure.
#[derive(Debug, thiserror::Error)]
pub enum GpuContactSolveError {
    /// The reference contact contract rejected the input.
    #[error(transparent)]
    Contact(#[from] ContactSolveError),
    /// A finite f64 input cannot be represented in the f32 shader.
    #[error("contact input is outside the WebGPU f32 range")]
    Precision,
    /// Buffers or indices exceed the selected device's limits.
    #[error("contact problem exceeds GPU capacity")]
    Capacity,
    /// Device-resident inverse matrix or response failed validation.
    #[error("GPU contact inverse mass is singular or non-finite")]
    DeviceMass,
    /// The GPU mass response preparation failed before submission.
    #[error(transparent)]
    MassResponse(#[from] GpuContactMassResponseError),
    /// GPU execution or result transfer failed.
    #[error("GPU contact solve readback failed: {0}")]
    Readback(String),
}

/// GPU buffers containing one encoded contact solve's result.
#[derive(Debug)]
pub struct GpuContactBufferOutput {
    velocity: wgpu::Buffer,
    impulses: wgpu::Buffer,
    velocity_count: usize,
    contact_count: usize,
    _inputs: Vec<wgpu::Buffer>,
    _bind_group: wgpu::BindGroup,
    _mass_response: Option<GpuContactMassResponseBatch>,
    _preparation_bind_group: Option<wgpu::BindGroup>,
}

impl GpuContactBufferOutput {
    /// Storage buffer containing generalized f32 velocity after the solve.
    pub fn velocity(&self) -> &wgpu::Buffer {
        &self.velocity
    }

    /// Storage buffer containing one `vec4<f32>` impulse per contact.
    pub fn impulses(&self) -> &wgpu::Buffer {
        &self.impulses
    }

    /// Number of meaningful f32 elements in the velocity buffer.
    pub fn velocity_count(&self) -> usize {
        self.velocity_count
    }

    /// Number of meaningful impulse elements in the impulse buffer.
    pub fn contact_count(&self) -> usize {
        self.contact_count
    }

    /// Encode copies for a later CPU readback in the same command encoder.
    pub fn encode_readback(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) -> GpuContactReadback {
        let velocity_bytes = self.velocity_count as u64 * size_of::<f32>() as u64;
        let impulse_bytes = self.contact_count as u64 * size_of::<[f32; 4]>() as u64;
        let mass_bytes = self._mass_response.as_ref().map_or(0, |_| {
            self.contact_count as u64 * 3 * size_of::<f32>() as u64
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera contact solve readback"),
            size: velocity_bytes + impulse_bytes + mass_bytes * 2,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&self.velocity, 0, &staging, 0, velocity_bytes);
        encoder.copy_buffer_to_buffer(&self.impulses, 0, &staging, velocity_bytes, impulse_bytes);
        if let Some(mass) = &self._mass_response {
            encoder.copy_buffer_to_buffer(
                mass.status_buffer(),
                0,
                &staging,
                velocity_bytes + impulse_bytes,
                mass_bytes,
            );
            encoder.copy_buffer_to_buffer(
                mass.effective_buffer(),
                0,
                &staging,
                velocity_bytes + impulse_bytes + mass_bytes,
                mass_bytes,
            );
        }
        GpuContactReadback {
            staging,
            velocity_bytes,
            impulse_bytes,
            mass_bytes,
        }
    }
}

/// Staging output whose encoder must be submitted before [`Self::finish`].
#[derive(Debug)]
pub struct GpuContactReadback {
    staging: wgpu::Buffer,
    velocity_bytes: u64,
    impulse_bytes: u64,
    mass_bytes: u64,
}

impl GpuContactReadback {
    /// Wait for the submitted GPU work and recover generalized velocity and impulses.
    pub fn finish(self, device: &wgpu::Device) -> Result<ContactSolution, GpuContactSolveError> {
        Ok(Self::finish_many(vec![self], device)?.remove(0))
    }

    /// Read several outputs after one command submission and one device poll.
    pub fn finish_many(
        readbacks: Vec<Self>,
        device: &wgpu::Device,
    ) -> Result<Vec<ContactSolution>, GpuContactSolveError> {
        if readbacks.is_empty() {
            return Ok(Vec::new());
        }
        let receivers = readbacks
            .iter()
            .map(|readback| {
                let (sender, receiver) = mpsc::channel();
                readback
                    .staging
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |result| {
                        let _ = sender.send(result);
                    });
                receiver
            })
            .collect::<Vec<_>>();
        let _status = device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(5)),
            })
            .map_err(|error| GpuContactSolveError::Readback(error.to_string()))?;
        readbacks
            .into_iter()
            .zip(receivers)
            .map(|(readback, receiver)| {
                receiver
                    .recv_timeout(Duration::from_secs(5))
                    .map_err(|error| GpuContactSolveError::Readback(error.to_string()))?
                    .map_err(|error| GpuContactSolveError::Readback(error.to_string()))?;
                readback.decode()
            })
            .collect()
    }

    fn decode(self) -> Result<ContactSolution, GpuContactSolveError> {
        let view = self.staging.slice(..).get_mapped_range();
        let impulse_end = (self.velocity_bytes + self.impulse_bytes) as usize;
        let velocity = bytemuck::cast_slice::<u8, f32>(&view[..self.velocity_bytes as usize])
            .iter()
            .map(|value| f64::from(*value))
            .collect::<Vec<_>>();
        let impulses = view[self.velocity_bytes as usize..impulse_end]
            .chunks_exact(size_of::<[f32; 4]>())
            .map(|bytes| {
                let values: [f32; 4] = bytemuck::pod_read_unaligned(bytes);
                ContactImpulse {
                    normal: f64::from(values[0]),
                    tangents: [f64::from(values[1]), f64::from(values[2])],
                }
            })
            .collect::<Vec<_>>();
        let invalid_mass = if self.mass_bytes > 0 {
            let status_end = impulse_end + self.mass_bytes as usize;
            let statuses = bytemuck::cast_slice::<u8, f32>(&view[impulse_end..status_end]);
            let effective = bytemuck::cast_slice::<u8, f32>(&view[status_end..]);
            statuses.iter().any(|status| *status != 0.0)
                || effective
                    .chunks_exact(3)
                    .any(|rows| !rows[0].is_finite() || rows[0] <= 1e-12)
        } else {
            false
        };
        drop(view);
        self.staging.unmap();
        if invalid_mass {
            return Err(GpuContactSolveError::DeviceMass);
        }
        if !velocity.iter().all(|value| value.is_finite())
            || impulses.iter().any(|impulse| {
                !impulse.normal.is_finite()
                    || impulse.tangents.iter().any(|value| !value.is_finite())
            })
        {
            return Err(ContactSolveError::InvalidInput.into());
        }
        Ok(ContactSolution {
            velocity: DVector::from_vec(velocity),
            impulses,
        })
    }
}

/// Reusable WebGPU contact solver for the same generalized problem as the CPU oracle.
#[derive(Debug)]
pub struct GpuContactSolver {
    pipeline: wgpu::ComputePipeline,
    mass_prepare_pipeline: wgpu::ComputePipeline,
}

impl GpuContactSolver {
    /// Compile the vendor-neutral WGSL contact kernel.
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera generalized contact solver"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_contact_solver.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Tessera generalized contact solver layout"),
            entries: &[0u32, 1, 2, 3, 4, 5, 6, 7, 8].map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 5 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: binding == 1
                                || binding == 2
                                || binding == 3
                                || binding == 6
                                || binding == 7
                                || binding == 8,
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Tessera generalized contact solver pipeline layout"),
            bind_group_layouts: &[&layout],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tessera generalized contact solver pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("solve"),
            compilation_options: Default::default(),
            cache: None,
        });
        let prepare_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera device mass contact preparation"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_contact_mass_prepare.wgsl").into()),
        });
        let mass_prepare_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Tessera device mass contact preparation"),
                layout: None,
                module: &prepare_shader,
                entry_point: Some("prepare"),
                compilation_options: Default::default(),
                cache: None,
            });
        Self {
            pipeline,
            mass_prepare_pipeline,
        }
    }

    /// Solve a generalized contact problem and read the resulting impulses.
    ///
    /// Input packing and output readback are synchronous. Separate workgroups
    /// solve disjoint generalized-coordinate islands. Dependency-preserving
    /// color waves let disjoint contacts in an island run concurrently.
    pub fn solve(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        problem: &ContactProblem,
        params: SolveParams,
        warm_start: Option<&[ContactImpulse]>,
    ) -> Result<ContactSolution, GpuContactSolveError> {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera contact solve encoder"),
        });
        let Some(output) = self.encode(device, &mut encoder, problem, params, warm_start)? else {
            return Ok(ContactSolution {
                velocity: problem.velocity.clone(),
                impulses: Vec::new(),
            });
        };
        let readback = output.encode_readback(device, &mut encoder);
        let _submission = queue.submit(Some(encoder.finish()));
        readback.finish(device)
    }

    /// Solve independent local contact problems in one GPU dispatch.
    ///
    /// Each problem keeps its own generalized-coordinate width and contact
    /// order. The packed Jacobians contain only local rows, avoiding a dense
    /// block-diagonal matrix as the number of environments grows.
    pub fn solve_packed(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        requests: &[GpuContactSolveRequest<'_>],
        params: SolveParams,
    ) -> Result<Vec<ContactSolution>, GpuContactSolveError> {
        let mut combined = PreparedGpuProblem::default();
        let mut ranges = Vec::with_capacity(requests.len());
        for request in requests {
            let Some(mut local) =
                prepare_problem(device, request.problem, params, request.warm_start)?
            else {
                ranges.push(None);
                continue;
            };
            let velocity_start = combined.velocity.len();
            let contact_start = combined.contacts.len();
            let row_offset = u32::try_from(combined.jacobians.len())
                .map_err(|_| GpuContactSolveError::Capacity)?;
            let velocity_offset =
                u32::try_from(velocity_start).map_err(|_| GpuContactSolveError::Capacity)?;
            let contact_offset =
                u32::try_from(contact_start).map_err(|_| GpuContactSolveError::Capacity)?;
            let island_offset = u32::try_from(combined.island_contacts.len())
                .map_err(|_| GpuContactSolveError::Capacity)?;
            let color_offset =
                u32::try_from(combined.colors.len()).map_err(|_| GpuContactSolveError::Capacity)?;
            for contact in &mut local.contacts {
                contact.offsets[0] = contact.offsets[0]
                    .checked_add(row_offset)
                    .ok_or(GpuContactSolveError::Capacity)?;
            }
            for island in &mut local.islands {
                island.start_count[0] = island.start_count[0]
                    .checked_add(island_offset)
                    .ok_or(GpuContactSolveError::Capacity)?;
                island.start_count[2] = velocity_offset;
                island.color_start_count[0] = island.color_start_count[0]
                    .checked_add(color_offset)
                    .ok_or(GpuContactSolveError::Capacity)?;
            }
            for color in &mut local.colors {
                color.start_count[0] = color.start_count[0]
                    .checked_add(island_offset)
                    .ok_or(GpuContactSolveError::Capacity)?;
            }
            for index in &mut local.island_contacts {
                *index = index
                    .checked_add(contact_offset)
                    .ok_or(GpuContactSolveError::Capacity)?;
            }
            ranges.push(Some((
                velocity_start,
                local.velocity.len(),
                contact_start,
                local.contacts.len(),
            )));
            combined.velocity.extend(local.velocity);
            combined.jacobians.extend(local.jacobians);
            combined.responses.extend(local.responses);
            combined.contacts.extend(local.contacts);
            combined.seeds.extend(local.seeds);
            combined.islands.extend(local.islands);
            combined.colors.extend(local.colors);
            combined.island_contacts.extend(local.island_contacts);
            validate_prepared_capacity(device, &combined)?;
        }
        if combined.contacts.is_empty() {
            return Ok(requests
                .iter()
                .map(|request| ContactSolution {
                    velocity: request.problem.velocity.clone(),
                    impulses: Vec::new(),
                })
                .collect());
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera packed contact solve encoder"),
        });
        let output = self.encode_prepared(device, &mut encoder, combined, params)?;
        let readback = output.encode_readback(device, &mut encoder);
        let _submission = queue.submit(Some(encoder.finish()));
        let solved = readback.finish(device)?;
        Ok(requests
            .iter()
            .zip(ranges)
            .map(|(request, range)| {
                if let Some((velocity_start, width, contact_start, count)) = range {
                    ContactSolution {
                        velocity: DVector::from_iterator(
                            width,
                            solved.velocity.as_slice()[velocity_start..velocity_start + width]
                                .iter()
                                .copied(),
                        ),
                        impulses: solved.impulses[contact_start..contact_start + count].to_vec(),
                    }
                } else {
                    ContactSolution {
                        velocity: request.problem.velocity.clone(),
                        impulses: Vec::new(),
                    }
                }
            })
            .collect())
    }

    /// Encode a contact solve without submitting or reading back its GPU result.
    ///
    /// Returns `None` when there are no contacts; the unconstrained input
    /// velocity is then already the final solution. The caller can append more
    /// compute work or [`GpuContactBufferOutput::encode_readback`] before submit.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        problem: &ContactProblem,
        params: SolveParams,
        warm_start: Option<&[ContactImpulse]>,
    ) -> Result<Option<GpuContactBufferOutput>, GpuContactSolveError> {
        let Some(prepared) = prepare_problem(device, problem, params, warm_start)? else {
            return Ok(None);
        };
        Ok(Some(
            self.encode_prepared(device, encoder, prepared, params)?,
        ))
    }

    /// Encode contact impulses using a device-resident inverse mass matrix.
    ///
    /// The inverse range contains one row-major `width × width` matrix and a
    /// trailing GPU status value. This method encodes mass responses, contact
    /// coefficients, warm-start projection, and the impulse solve in order.
    /// The caller may encode articulated inverse-mass assembly earlier in the
    /// same command encoder. No inverse matrix is transferred to the CPU.
    pub fn encode_device_inverse_mass(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        request: GpuDeviceMassContactRequest<'_>,
    ) -> Result<Option<GpuContactBufferOutput>, GpuContactSolveError> {
        let GpuDeviceMassContactRequest {
            inverse_mass,
            inverse_range,
            velocity,
            contacts,
            params,
            warm_start,
        } = request;
        let width = velocity.len();
        let count = contacts.len();
        if width == 0
            || warm_start.is_some_and(|seeds| seeds.len() != count)
            || contacts.iter().any(|contact| {
                contact.normal.len() != width
                    || contact.tangents.iter().any(|row| row.len() != width)
            })
        {
            return Err(ContactSolveError::Dimensions.into());
        }
        if !params.dt.is_finite()
            || params.dt <= 0.0
            || params.iterations == 0
            || !params.position_gain.is_finite()
            || !(0.0..=1.0).contains(&params.position_gain)
            || !params.max_correction_speed.is_finite()
            || params.max_correction_speed < 0.0
            || velocity.iter().any(|value| !value.is_finite())
        {
            return Err(ContactSolveError::InvalidInput.into());
        }
        if count == 0 {
            return Ok(None);
        }
        let width_gpu = u32::try_from(width).map_err(|_| GpuContactSolveError::Capacity)?;
        let count_gpu = u32::try_from(count).map_err(|_| GpuContactSolveError::Capacity)?;
        let iterations =
            u32::try_from(params.iterations).map_err(|_| GpuContactSolveError::Capacity)?;
        let row_values = width
            .checked_mul(count)
            .and_then(|value| value.checked_mul(3))
            .ok_or(GpuContactSolveError::Capacity)?;
        let phases = if contacts.iter().any(|contact| {
            let incoming = contact.normal.dot(velocity);
            let correction = (params.position_gain * contact.penetration / params.dt)
                .min(params.max_correction_speed);
            contact.scalar.is_none()
                && incoming < -1e-3
                && -contact.restitution * incoming > correction
        }) {
            2u32
        } else {
            1u32
        };
        let work = width
            .checked_mul(3)
            .and_then(|value| value.checked_mul(count))
            .and_then(|value| value.checked_mul(phases as usize))
            .and_then(|value| value.checked_mul(params.iterations))
            .ok_or(GpuContactSolveError::Capacity)?;
        if work > 2_000_000 || row_values > u32::MAX as usize {
            return Err(GpuContactSolveError::Capacity);
        }
        let max_storage = u64::from(device.limits().max_storage_buffer_binding_size);
        let velocity_bytes = checked_contact_bytes(width, size_of::<f32>())?;
        let impulse_bytes = checked_contact_bytes(count, size_of::<[f32; 4]>())?;
        let status_bytes = checked_contact_bytes(count, 3 * size_of::<f32>())?;
        let storage_sizes = [
            velocity_bytes,
            checked_contact_bytes(count, size_of::<GpuContactData>())?,
            impulse_bytes,
            checked_contact_bytes(count, size_of::<GpuColorRange>())?,
            checked_contact_bytes(count, size_of::<u32>())?,
        ];
        if storage_sizes
            .iter()
            .any(|size| *size > max_storage || *size > device.limits().max_buffer_size)
            || velocity_bytes
                .checked_add(impulse_bytes)
                .and_then(|size| size.checked_add(status_bytes.checked_mul(2)?))
                .is_none_or(|size| size > device.limits().max_buffer_size)
        {
            return Err(GpuContactSolveError::Capacity);
        }
        let mut rows = Vec::with_capacity(row_values);
        let mut contact_data = Vec::with_capacity(count);
        let mut seeds = Vec::with_capacity(count);
        for (index, contact) in contacts.iter().enumerate() {
            if !contact.penetration.is_finite()
                || contact.penetration < 0.0
                || !contact.friction.is_finite()
                || contact.friction < 0.0
                || !contact.restitution.is_finite()
                || contact.restitution < 0.0
                || !contact.normal.iter().all(|value| value.is_finite())
                || !contact
                    .tangents
                    .iter()
                    .flatten()
                    .all(|value| value.is_finite())
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
                return Err(ContactSolveError::InvalidInput.into());
            }
            for row in [&contact.normal, &contact.tangents[0], &contact.tangents[1]] {
                rows.extend(row.iter().copied());
            }
            let incoming = contact.normal.dot(velocity);
            let correction = (params.position_gain * contact.penetration / params.dt)
                .min(params.max_correction_speed);
            let restitution = if incoming < -1e-3 {
                -contact.restitution * incoming
            } else {
                0.0
            };
            contact_data.push(GpuContactData {
                effective: [0.0; 4],
                targets: [
                    to_f32(
                        contact
                            .scalar
                            .map_or(correction, |scalar| scalar.target_speed),
                    )?,
                    to_f32(restitution)?,
                    to_f32(contact.scalar.map_or(contact.friction, |scalar| {
                        scalar.impulse_limit.unwrap_or(f32::MAX as f64)
                    }))?,
                    if contact.scalar.is_some() { 1.0 } else { 0.0 },
                ],
                offsets: [
                    u32::try_from(
                        index
                            .checked_mul(3)
                            .and_then(|value| value.checked_mul(width))
                            .ok_or(GpuContactSolveError::Capacity)?,
                    )
                    .map_err(|_| GpuContactSolveError::Capacity)?,
                    0,
                    0,
                    0,
                ],
            });
            let seed = warm_start.map_or(ContactImpulse::default(), |values| values[index]);
            if !seed.normal.is_finite() || seed.tangents.iter().any(|value| !value.is_finite()) {
                return Err(ContactSolveError::InvalidInput.into());
            }
            seeds.push([
                to_f32(seed.normal)?,
                to_f32(seed.tangents[0])?,
                to_f32(seed.tangents[1])?,
                0.0,
            ]);
        }
        let mass = GpuContactMassResponseBatch::new(
            device,
            inverse_mass,
            &[GpuContactMassResponseSystem {
                inverse_range,
                jacobians: DMatrix::from_row_slice(count * 3, width, &rows),
            }],
        )?;
        let velocity_values = velocity
            .iter()
            .map(|value| to_f32(*value))
            .collect::<Result<Vec<_>, _>>()?;
        let velocity_buffer = storage_input(
            device,
            "Tessera device-mass contact velocity",
            &velocity_values,
            wgpu::BufferUsages::COPY_SRC,
        );
        let contact_buffer = storage_input(
            device,
            "Tessera device-mass contact data",
            &contact_data,
            wgpu::BufferUsages::empty(),
        );
        let impulse_buffer = storage_input(
            device,
            "Tessera device-mass contact impulses",
            &seeds,
            wgpu::BufferUsages::COPY_SRC,
        );
        let settings = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera device-mass contact settings"),
            contents: bytemuck::bytes_of(&GpuSettings {
                dimensions: [width_gpu, count_gpu, iterations, 1],
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let ranges = storage_input(
            device,
            "Tessera device-mass contact island",
            &[GpuIslandRange {
                start_count: [0, count_gpu, 0, width_gpu],
                color_start_count: [0, count_gpu, phases, 0],
            }],
            wgpu::BufferUsages::empty(),
        );
        let colors = (0..count_gpu)
            .map(|index| GpuColorRange {
                start_count: [index, 1],
            })
            .collect::<Vec<_>>();
        let color_buffer = storage_input(
            device,
            "Tessera device-mass contact colors",
            &colors,
            wgpu::BufferUsages::empty(),
        );
        let indices = (0..count_gpu).collect::<Vec<_>>();
        let index_buffer = storage_input(
            device,
            "Tessera device-mass contact indices",
            &indices,
            wgpu::BufferUsages::empty(),
        );
        let prep_buffers = [
            &velocity_buffer,
            mass.jacobians_buffer(),
            mass.responses_buffer(),
            mass.effective_buffer(),
            mass.status_buffer(),
            &contact_buffer,
            &impulse_buffer,
            &settings,
        ];
        let preparation_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera device-mass contact preparation buffers"),
            layout: &self.mass_prepare_pipeline.get_bind_group_layout(0),
            entries: &prep_buffers
                .iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        mass.encode(encoder);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Tessera device-mass contact preparation"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.mass_prepare_pipeline);
            pass.set_bind_group(0, &preparation_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        let solve_buffers = [
            &velocity_buffer,
            mass.jacobians_buffer(),
            mass.responses_buffer(),
            &contact_buffer,
            &impulse_buffer,
            &settings,
            &ranges,
            &index_buffer,
            &color_buffer,
        ];
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera device-mass contact solve buffers"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &solve_buffers
                .iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Tessera device-mass contact solve"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        Ok(Some(GpuContactBufferOutput {
            velocity: velocity_buffer,
            impulses: impulse_buffer,
            velocity_count: width,
            contact_count: count,
            _inputs: vec![contact_buffer, settings, ranges, color_buffer, index_buffer],
            _bind_group: bind_group,
            _mass_response: Some(mass),
            _preparation_bind_group: Some(preparation_bind_group),
        }))
    }

    fn encode_prepared(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        prepared: PreparedGpuProblem,
        params: SolveParams,
    ) -> Result<GpuContactBufferOutput, GpuContactSolveError> {
        validate_prepared_capacity(device, &prepared)?;
        let width = prepared.velocity.len();
        let count = prepared.contacts.len();
        let width_gpu = u32::try_from(width).map_err(|_| GpuContactSolveError::Capacity)?;
        let count_gpu = u32::try_from(count).map_err(|_| GpuContactSolveError::Capacity)?;
        let iterations =
            u32::try_from(params.iterations).map_err(|_| GpuContactSolveError::Capacity)?;
        let island_count =
            u32::try_from(prepared.islands.len()).map_err(|_| GpuContactSolveError::Capacity)?;
        let velocity_buffer = storage_input(
            device,
            "Tessera contact velocity",
            &prepared.velocity,
            wgpu::BufferUsages::COPY_SRC,
        );
        let jacobian_buffer = storage_input(
            device,
            "Tessera contact Jacobians",
            &prepared.jacobians,
            wgpu::BufferUsages::empty(),
        );
        let response_buffer = storage_input(
            device,
            "Tessera contact responses",
            &prepared.responses,
            wgpu::BufferUsages::empty(),
        );
        let contact_buffer = storage_input(
            device,
            "Tessera contact parameters",
            &prepared.contacts,
            wgpu::BufferUsages::empty(),
        );
        let impulse_buffer = storage_input(
            device,
            "Tessera contact impulses",
            &prepared.seeds,
            wgpu::BufferUsages::COPY_SRC,
        );
        let island_range_buffer = storage_input(
            device,
            "Tessera contact island ranges",
            &prepared.islands,
            wgpu::BufferUsages::empty(),
        );
        let island_contact_buffer = storage_input(
            device,
            "Tessera contact island indices",
            &prepared.island_contacts,
            wgpu::BufferUsages::empty(),
        );
        let color_range_buffer = storage_input(
            device,
            "Tessera contact color ranges",
            &prepared.colors,
            wgpu::BufferUsages::empty(),
        );
        let settings_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera contact solve settings"),
            contents: bytemuck::bytes_of(&GpuSettings {
                dimensions: [width_gpu, count_gpu, iterations, island_count],
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let layout = self.pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera contact solve buffers"),
            layout: &layout,
            entries: [
                &velocity_buffer,
                &jacobian_buffer,
                &response_buffer,
                &contact_buffer,
                &impulse_buffer,
                &settings_buffer,
                &island_range_buffer,
                &island_contact_buffer,
                &color_range_buffer,
            ]
            .into_iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>()
            .as_slice(),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Tessera contact solve pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(island_count, 1, 1);
        }
        Ok(GpuContactBufferOutput {
            velocity: velocity_buffer,
            impulses: impulse_buffer,
            velocity_count: width,
            contact_count: count,
            _inputs: vec![
                jacobian_buffer,
                response_buffer,
                contact_buffer,
                settings_buffer,
                island_range_buffer,
                island_contact_buffer,
                color_range_buffer,
            ],
            _bind_group: bind_group,
            _mass_response: None,
            _preparation_bind_group: None,
        })
    }
}

fn prepare_problem(
    device: &wgpu::Device,
    problem: &ContactProblem,
    params: SolveParams,
    warm_start: Option<&[ContactImpulse]>,
) -> Result<Option<PreparedGpuProblem>, GpuContactSolveError> {
    let width = problem.velocity.len();
    let count = problem.contacts.len();
    if problem.inverse_mass.nrows() != width
        || problem.inverse_mass.ncols() != width
        || warm_start.is_some_and(|seed| seed.len() != count)
        || problem.contacts.iter().any(|contact| {
            contact.normal.len() != width || contact.tangents.iter().any(|row| row.len() != width)
        })
    {
        return Err(ContactSolveError::Dimensions.into());
    }
    if !params.dt.is_finite()
        || params.dt <= 0.0
        || params.iterations == 0
        || !params.position_gain.is_finite()
        || !(0.0..=1.0).contains(&params.position_gain)
        || !params.max_correction_speed.is_finite()
        || params.max_correction_speed < 0.0
        || !problem.velocity.iter().all(|value| value.is_finite())
        || !problem.inverse_mass.iter().all(|value| value.is_finite())
    {
        return Err(ContactSolveError::InvalidInput.into());
    }
    if count == 0 {
        return Ok(None);
    }

    let _width_gpu = u32::try_from(width).map_err(|_| GpuContactSolveError::Capacity)?;
    let _count_gpu = u32::try_from(count).map_err(|_| GpuContactSolveError::Capacity)?;
    let _iterations =
        u32::try_from(params.iterations).map_err(|_| GpuContactSolveError::Capacity)?;
    let row_values = width
        .checked_mul(count)
        .and_then(|size| size.checked_mul(3))
        .ok_or(GpuContactSolveError::Capacity)?;
    let row_bytes = u64::try_from(row_values)
        .ok()
        .and_then(|values| values.checked_mul(size_of::<f32>() as u64))
        .ok_or(GpuContactSolveError::Capacity)?;
    let velocity_bytes = u64::try_from(width)
        .ok()
        .and_then(|values| values.checked_mul(size_of::<f32>() as u64))
        .ok_or(GpuContactSolveError::Capacity)?;
    let contact_bytes = u64::try_from(count)
        .ok()
        .and_then(|values| values.checked_mul(size_of::<GpuContactData>() as u64))
        .ok_or(GpuContactSolveError::Capacity)?;
    let impulse_bytes = u64::try_from(count)
        .ok()
        .and_then(|values| values.checked_mul(size_of::<[f32; 4]>() as u64))
        .ok_or(GpuContactSolveError::Capacity)?;
    let max_storage = u64::from(device.limits().max_storage_buffer_binding_size);
    if row_values > u32::MAX as usize
        || [velocity_bytes, row_bytes, contact_bytes, impulse_bytes]
            .iter()
            .any(|size| *size > max_storage || *size > device.limits().max_buffer_size)
        || velocity_bytes
            .checked_add(impulse_bytes)
            .is_none_or(|size| size > device.limits().max_buffer_size)
    {
        return Err(GpuContactSolveError::Capacity);
    }
    let mut jacobians = Vec::with_capacity(row_values);
    let mut responses = Vec::with_capacity(row_values);
    let mut contact_data = Vec::with_capacity(count);
    let mut seeds = vec![[0.0f32; 4]; count];
    let mut initial_velocity = problem.velocity.clone();
    for (index, contact) in problem.contacts.iter().enumerate() {
        if !contact.penetration.is_finite()
            || contact.penetration < 0.0
            || !contact.friction.is_finite()
            || contact.friction < 0.0
            || !contact.restitution.is_finite()
            || contact.restitution < 0.0
            || !contact.normal.iter().all(|value| value.is_finite())
            || !contact
                .tangents
                .iter()
                .flatten()
                .all(|value| value.is_finite())
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
            return Err(ContactSolveError::InvalidInput.into());
        }
        let rows = [&contact.normal, &contact.tangents[0], &contact.tangents[1]];
        let mut row_responses = Vec::with_capacity(3);
        let mut effective = [0.0f64; 3];
        for (axis, row) in rows.iter().enumerate() {
            let response = mass_response(&problem.inverse_mass, row);
            effective[axis] = row.dot(&response);
            for &value in row.iter() {
                jacobians.push(to_f32(value)?);
            }
            for &value in response.iter() {
                responses.push(to_f32(value)?);
            }
            row_responses.push(response);
        }
        if effective.iter().any(|value| !value.is_finite()) || effective[0] <= 1e-12 {
            return Err(ContactSolveError::SingularContact.into());
        }
        let tangent_coupling = rows[1].dot(&row_responses[2]);
        if !tangent_coupling.is_finite() {
            return Err(ContactSolveError::InvalidInput.into());
        }
        let incoming = contact.normal.dot(&problem.velocity);
        let correction = (params.position_gain * contact.penetration / params.dt)
            .min(params.max_correction_speed);
        let restitution = if incoming < -1e-3 {
            -contact.restitution * incoming
        } else {
            0.0
        };
        contact_data.push(GpuContactData {
            effective: [
                to_f32(effective[0])?,
                to_f32(effective[1])?,
                to_f32(effective[2])?,
                to_f32(tangent_coupling)?,
            ],
            targets: [
                to_f32(
                    contact
                        .scalar
                        .map_or(correction, |scalar| scalar.target_speed),
                )?,
                to_f32(restitution)?,
                to_f32(contact.scalar.map_or(contact.friction, |scalar| {
                    scalar.impulse_limit.unwrap_or(f32::MAX as f64)
                }))?,
                if contact.scalar.is_some() { 1.0 } else { 0.0 },
            ],
            offsets: [
                u32::try_from(index * 3 * width).map_err(|_| GpuContactSolveError::Capacity)?,
                0,
                0,
                0,
            ],
        });
        if let Some(seed) = warm_start.map(|values| values[index]) {
            if !seed.normal.is_finite() || seed.tangents.iter().any(|value| !value.is_finite()) {
                return Err(ContactSolveError::InvalidInput.into());
            }
            if let Some(scalar) = contact.scalar {
                let normal = scalar
                    .impulse_limit
                    .map_or(seed.normal, |bound| seed.normal.clamp(-bound, bound));
                seeds[index][0] = to_f32(normal)?;
                initial_velocity += &row_responses[0] * normal;
                continue;
            }
            let normal = seed.normal.max(0.0);
            let mut tangent = seed.tangents;
            let length = tangent[0].hypot(tangent[1]);
            let radius = contact.friction * normal;
            if length > radius && length > 0.0 {
                tangent = tangent.map(|value| value * radius / length);
            }
            for axis in 0..2 {
                if effective[axis + 1] <= 1e-12 {
                    tangent[axis] = 0.0;
                }
            }
            seeds[index] = [
                to_f32(normal)?,
                to_f32(tangent[0])?,
                to_f32(tangent[1])?,
                0.0,
            ];
            initial_velocity += &row_responses[0] * normal;
            initial_velocity += &row_responses[1] * tangent[0];
            initial_velocity += &row_responses[2] * tangent[1];
        }
    }

    let (mut island_ranges, colors, island_contacts) =
        partition_islands(width, count, &jacobians, &responses);
    for island in &mut island_ranges {
        let start = island.start_count[0] as usize;
        let count = island.start_count[1] as usize;
        island.color_start_count[2] =
            if island_contacts[start..start + count].iter().any(|&index| {
                let contact = &contact_data[index as usize];
                contact.targets[3] == 0.0 && contact.targets[1] > contact.targets[0]
            }) {
                2
            } else {
                1
            };
    }
    let island_count =
        u32::try_from(island_ranges.len()).map_err(|_| GpuContactSolveError::Capacity)?;
    let max_island_work = island_ranges
        .iter()
        .map(|range| range.start_count[1] as usize * range.color_start_count[2] as usize)
        .max()
        .unwrap_or(0);
    let max_work = width
        .checked_mul(3)
        .and_then(|value| value.checked_mul(max_island_work))
        .and_then(|value| value.checked_mul(params.iterations))
        .ok_or(GpuContactSolveError::Capacity)?;
    if max_work > 2_000_000 || island_count > device.limits().max_compute_workgroups_per_dimension {
        return Err(GpuContactSolveError::Capacity);
    }

    let input_velocity = initial_velocity
        .iter()
        .map(|value| to_f32(*value))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Some(PreparedGpuProblem {
        velocity: input_velocity,
        jacobians,
        responses,
        contacts: contact_data,
        seeds,
        islands: island_ranges,
        colors,
        island_contacts,
    }))
}

fn validate_prepared_capacity(
    device: &wgpu::Device,
    prepared: &PreparedGpuProblem,
) -> Result<(), GpuContactSolveError> {
    let limits = device.limits();
    let max_storage = u64::from(limits.max_storage_buffer_binding_size);
    let lengths = [
        (prepared.velocity.len(), size_of::<f32>()),
        (prepared.jacobians.len(), size_of::<f32>()),
        (prepared.responses.len(), size_of::<f32>()),
        (prepared.contacts.len(), size_of::<GpuContactData>()),
        (prepared.seeds.len(), size_of::<[f32; 4]>()),
        (prepared.islands.len(), size_of::<GpuIslandRange>()),
        (prepared.colors.len(), size_of::<GpuColorRange>()),
        (prepared.island_contacts.len(), size_of::<u32>()),
    ];
    for (len, element_size) in lengths {
        let bytes = u64::try_from(len)
            .ok()
            .and_then(|len| len.checked_mul(element_size as u64))
            .ok_or(GpuContactSolveError::Capacity)?;
        if bytes > max_storage || bytes > limits.max_buffer_size {
            return Err(GpuContactSolveError::Capacity);
        }
    }
    if prepared.islands.len() > limits.max_compute_workgroups_per_dimension as usize
        || prepared.jacobians.len() > u32::MAX as usize
        || prepared.colors.len() > u32::MAX as usize
        || prepared.island_contacts.len() > u32::MAX as usize
    {
        return Err(GpuContactSolveError::Capacity);
    }
    let readback = (prepared.velocity.len() as u64)
        .checked_mul(size_of::<f32>() as u64)
        .and_then(|velocity| {
            (prepared.seeds.len() as u64)
                .checked_mul(size_of::<[f32; 4]>() as u64)
                .and_then(|impulses| velocity.checked_add(impulses))
        })
        .ok_or(GpuContactSolveError::Capacity)?;
    if readback > limits.max_buffer_size {
        return Err(GpuContactSolveError::Capacity);
    }
    Ok(())
}

fn partition_islands(
    width: usize,
    count: usize,
    jacobians: &[f32],
    responses: &[f32],
) -> (Vec<GpuIslandRange>, Vec<GpuColorRange>, Vec<u32>) {
    let mut parents = (0..count).collect::<Vec<_>>();
    let mut first_contact = vec![None; width];
    for contact in 0..count {
        for (coordinate, first) in first_contact.iter_mut().enumerate() {
            let touches = contact_touches(contact, coordinate, width, jacobians, responses);
            if !touches {
                continue;
            }
            if let Some(previous) = *first {
                let a = find_root(&mut parents, contact);
                let b = find_root(&mut parents, previous);
                parents[a.max(b)] = a.min(b);
            } else {
                *first = Some(contact);
            }
        }
    }
    let mut grouped = BTreeMap::<usize, Vec<u32>>::new();
    for contact in 0..count {
        let root = find_root(&mut parents, contact);
        grouped.entry(root).or_default().push(contact as u32);
    }
    let mut ranges = Vec::with_capacity(grouped.len());
    let mut colors = Vec::new();
    let mut indices = Vec::with_capacity(count);
    for contacts in grouped.into_values() {
        let mut previous_color = vec![None::<usize>; width];
        let mut waves = Vec::<Vec<u32>>::new();
        for contact in &contacts {
            let mut color = 0;
            for (coordinate, previous) in previous_color.iter().enumerate() {
                if contact_touches(*contact as usize, coordinate, width, jacobians, responses)
                    && let Some(previous) = *previous
                {
                    color = color.max(previous + 1);
                }
            }
            if color == waves.len() {
                waves.push(Vec::new());
            }
            waves[color].push(*contact);
            for (coordinate, previous) in previous_color.iter_mut().enumerate() {
                if contact_touches(*contact as usize, coordinate, width, jacobians, responses) {
                    *previous = Some(color);
                }
            }
        }
        ranges.push(GpuIslandRange {
            start_count: [indices.len() as u32, contacts.len() as u32, 0, width as u32],
            color_start_count: [colors.len() as u32, waves.len() as u32, 0, 0],
        });
        for wave in waves {
            colors.push(GpuColorRange {
                start_count: [indices.len() as u32, wave.len() as u32],
            });
            indices.extend(wave);
        }
    }
    (ranges, colors, indices)
}

fn contact_touches(
    contact: usize,
    coordinate: usize,
    width: usize,
    jacobians: &[f32],
    responses: &[f32],
) -> bool {
    (0..3).any(|axis| {
        let index = (contact * 3 + axis) * width + coordinate;
        jacobians[index] != 0.0 || responses[index] != 0.0
    })
}

fn find_root(parents: &mut [usize], index: usize) -> usize {
    let mut root = index;
    while parents[root] != root {
        root = parents[root];
    }
    let mut node = index;
    while parents[node] != root {
        let next = parents[node];
        parents[node] = root;
        node = next;
    }
    root
}

fn mass_response(inverse_mass: &DMatrix<f64>, row: &DVector<f64>) -> DVector<f64> {
    let mut response = DVector::zeros(row.len());
    for (column, &coefficient) in row.iter().enumerate() {
        if coefficient == 0.0 {
            continue;
        }
        for coordinate in 0..row.len() {
            response[coordinate] += inverse_mass[(coordinate, column)] * coefficient;
        }
    }
    response
}

fn to_f32(value: f64) -> Result<f32, GpuContactSolveError> {
    let converted = value as f32;
    if converted.is_finite() {
        Ok(converted)
    } else {
        Err(GpuContactSolveError::Precision)
    }
}

fn checked_contact_bytes(count: usize, stride: usize) -> Result<u64, GpuContactSolveError> {
    count
        .checked_mul(stride)
        .and_then(|value| u64::try_from(value).ok())
        .ok_or(GpuContactSolveError::Capacity)
}

fn storage_input<T: bytemuck::Pod>(
    device: &wgpu::Device,
    label: &'static str,
    data: &[T],
    extra_usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE | extra_usage,
    })
}

#[cfg(test)]
mod tests {
    use nalgebra::{DMatrix, DVector, Matrix3};

    use super::*;
    use crate::contact_reference::{ContactConstraint, solve_contacts};
    use crate::gpu_articulated_mass::read_buffer;
    use crate::gpu_articulated_mass_assembly::{
        GpuArticulatedMassAssemblyBatch, GpuMassAssemblySystem, GpuMassLink,
    };
    use crate::gpu_contact_pipeline::GpuContactDevice;

    fn params() -> SolveParams {
        SolveParams {
            dt: 0.01,
            iterations: 12,
            position_gain: 0.2,
            max_correction_speed: 2.0,
        }
    }

    fn problem() -> ContactProblem {
        ContactProblem {
            inverse_mass: DMatrix::from_diagonal(&DVector::from_vec(vec![
                1.0, 1.0, 1.0, 0.5, 0.5, 0.5,
            ])),
            velocity: DVector::from_vec(vec![1.0, 0.0, -2.0, -0.5, 0.0, 0.0]),
            contacts: vec![
                ContactConstraint {
                    scalar: None,
                    normal: DVector::from_vec(vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
                    tangents: [
                        DVector::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                        DVector::from_vec(vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
                    ],
                    penetration: 0.01,
                    friction: 0.5,
                    restitution: 0.2,
                },
                ContactConstraint {
                    scalar: None,
                    normal: DVector::from_vec(vec![-1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
                    tangents: [
                        DVector::from_vec(vec![0.0, -1.0, 0.0, 0.0, 1.0, 0.0]),
                        DVector::from_vec(vec![0.0, 0.0, -1.0, 0.0, 0.0, 1.0]),
                    ],
                    penetration: 0.0,
                    friction: 0.3,
                    restitution: 0.0,
                },
            ],
        }
    }

    #[tokio::test]
    async fn gpu_bounded_axis_matches_cpu_with_contact_coupling() {
        let instance = wgpu::Instance::default();
        let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
        else {
            return;
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let solver = GpuContactSolver::new(&device);
        let mut problem = problem();
        problem.contacts.push(ContactConstraint::bounded_axis(
            DVector::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            0.3,
        ));
        let cpu = solve_contacts(&problem, params(), None).unwrap();
        let gpu = solver
            .solve(&device, &queue, &problem, params(), None)
            .unwrap();
        for (actual, expected) in gpu.velocity.iter().zip(cpu.velocity.iter()) {
            assert!((actual - expected).abs() < 1e-4);
        }
        assert!((gpu.impulses[2].normal - cpu.impulses[2].normal).abs() < 1e-4);
        assert!(gpu.impulses[2].normal.abs() <= 0.30001);
    }

    #[tokio::test]
    async fn gpu_bilateral_axis_matches_cpu() {
        let instance = wgpu::Instance::default();
        let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
        else {
            return;
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let solver = GpuContactSolver::new(&device);
        let mut problem = ContactProblem {
            inverse_mass: DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 2.0])),
            velocity: DVector::from_vec(vec![-1.0, 0.5]),
            contacts: vec![
                ContactConstraint {
                    scalar: None,
                    normal: DVector::from_vec(vec![1.0, 0.0]),
                    tangents: [DVector::zeros(2), DVector::zeros(2)],
                    penetration: 0.0,
                    friction: 0.0,
                    restitution: 0.0,
                },
                ContactConstraint::bilateral_axis(DVector::from_vec(vec![-2.0, 1.0]), 0.0),
            ],
        };
        for target in [0.0, -0.2] {
            problem.contacts[1] =
                ContactConstraint::bilateral_axis(DVector::from_vec(vec![-2.0, 1.0]), target);
            let cpu = solve_contacts(&problem, params(), None).unwrap();
            let gpu = solver
                .solve(&device, &queue, &problem, params(), None)
                .unwrap();
            assert!((gpu.velocity[0] - cpu.velocity[0]).abs() < 1e-6);
            assert!((gpu.velocity[1] - cpu.velocity[1]).abs() < 1e-6);
            for (actual, expected) in gpu.impulses.iter().zip(cpu.impulses.iter()) {
                assert!((actual.normal - expected.normal).abs() < 1e-6);
            }
        }
    }

    #[tokio::test]
    async fn gpu_generalized_impulses_match_cpu_with_and_without_warm_start() {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let solver = GpuContactSolver::new(&device);
        let problem = problem();
        for warm_start in [
            None,
            Some(vec![
                ContactImpulse {
                    normal: 0.6,
                    tangents: [0.5, 0.0],
                },
                ContactImpulse {
                    normal: 0.3,
                    tangents: [0.0, -0.2],
                },
            ]),
        ] {
            let seed = warm_start.as_deref();
            let cpu = solve_contacts(&problem, params(), seed).unwrap();
            let gpu = solver
                .solve(&device, &queue, &problem, params(), seed)
                .unwrap();
            for (actual, expected) in gpu.velocity.iter().zip(cpu.velocity.iter()) {
                assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
            }
            for (actual, expected) in gpu.impulses.iter().zip(cpu.impulses.iter()) {
                assert!((actual.normal - expected.normal).abs() < 1e-5);
                for (actual, expected) in actual.tangents.iter().zip(expected.tangents.iter()) {
                    assert!((actual - expected).abs() < 1e-5);
                }
            }
        }
    }

    #[tokio::test]
    async fn gpu_final_restitution_survives_a_coupled_support_contact() {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
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
                    normal: DVector::from_vec(vec![-0.5, 3.0f64.sqrt() * 0.5]),
                    tangents: [DVector::zeros(2), DVector::zeros(2)],
                    penetration: 0.0,
                    friction: 0.0,
                    restitution: 0.0,
                },
            ],
        };
        let mut settings = params();
        settings.iterations = 16;
        let expected = solve_contacts(&problem, settings, None).unwrap();
        let actual = GpuContactSolver::new(&device)
            .solve(&device, &queue, &problem, settings, None)
            .unwrap();
        assert!((actual.velocity[0] - 1.0).abs() < 1e-5);
        assert!(problem.contacts[1].normal.dot(&actual.velocity) >= -1e-5);
        for (a, b) in actual.velocity.iter().zip(expected.velocity.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} != {b}");
        }

        settings.iterations = 1;
        let short_solve = GpuContactSolver::new(&device)
            .solve(&device, &queue, &problem, settings, None)
            .unwrap();
        assert!(short_solve.velocity[0] > 0.6);
        assert!(problem.contacts[1].normal.dot(&short_solve.velocity) >= -1e-5);
    }

    #[tokio::test]
    async fn gpu_coupled_tangent_mass_matches_cpu_in_one_iteration() {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let diagonal = 1.0 / 2.0f64.sqrt();
        let mut problem = ContactProblem {
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
        settings.position_gain = 0.0;
        let solver = GpuContactSolver::new(&device);
        for friction in [10.0, 0.25] {
            problem.contacts[0].friction = friction;
            let expected = solve_contacts(&problem, settings, None).unwrap();
            let actual = solver
                .solve(&device, &queue, &problem, settings, None)
                .unwrap();
            for (a, b) in actual.velocity.iter().zip(expected.velocity.iter()) {
                assert!((a - b).abs() < 1e-5, "{a} != {b}");
            }
            for (a, b) in actual.impulses[0]
                .tangents
                .iter()
                .zip(expected.impulses[0].tangents)
            {
                assert!((a - b).abs() < 1e-5, "{a} != {b}");
            }
            assert!(
                actual.impulses[0].tangents[0].hypot(actual.impulses[0].tangents[1])
                    <= friction * actual.impulses[0].normal + 1e-6
            );
        }
    }

    #[tokio::test]
    async fn two_contact_solves_share_one_command_submission() {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let solver = GpuContactSolver::new(&device);
        let first = problem();
        let mut second = problem();
        second.velocity[2] = -4.0;
        second.contacts[0].friction = 0.8;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tessera two contact solves"),
        });
        let first_gpu = solver
            .encode(&device, &mut encoder, &first, params(), None)
            .unwrap()
            .unwrap();
        let second_gpu = solver
            .encode(&device, &mut encoder, &second, params(), None)
            .unwrap()
            .unwrap();
        assert_eq!(first_gpu.velocity_count(), 6);
        assert_eq!(first_gpu.contact_count(), 2);
        assert_eq!(first_gpu.velocity().size(), 6 * size_of::<f32>() as u64);
        assert_eq!(
            first_gpu.impulses().size(),
            2 * size_of::<[f32; 4]>() as u64
        );
        let first_readback = first_gpu.encode_readback(&device, &mut encoder);
        let second_readback = second_gpu.encode_readback(&device, &mut encoder);
        let _submission = queue.submit(Some(encoder.finish()));
        let outputs =
            GpuContactReadback::finish_many(vec![first_readback, second_readback], &device)
                .unwrap();
        for (actual, input) in outputs.into_iter().zip([&first, &second]) {
            let reference = solve_contacts(input, params(), None).unwrap();
            for (actual, expected) in actual.velocity.iter().zip(reference.velocity.iter()) {
                assert!((actual - expected).abs() < 1e-5);
            }
        }
    }

    #[tokio::test]
    async fn packed_local_problems_match_cpu_with_mixed_widths_and_empty_world() {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let solver = GpuContactSolver::new(&device);
        let first = problem();
        let one_dof = ContactProblem {
            inverse_mass: DMatrix::from_element(1, 1, 0.5),
            velocity: DVector::from_element(1, -2.0),
            contacts: vec![ContactConstraint {
                scalar: None,
                normal: DVector::from_element(1, 1.0),
                tangents: [DVector::zeros(1), DVector::zeros(1)],
                penetration: 0.02,
                friction: 0.4,
                restitution: 0.0,
            }],
        };
        let empty = ContactProblem {
            inverse_mass: DMatrix::identity(3, 3),
            velocity: DVector::from_vec(vec![0.25, -0.5, 1.0]),
            contacts: Vec::new(),
        };
        let mut last = problem();
        last.velocity[2] = -3.0;
        last.contacts[0].friction = 0.8;
        let seeds = [
            ContactImpulse {
                normal: 0.6,
                tangents: [0.5, 0.0],
            },
            ContactImpulse {
                normal: 0.3,
                tangents: [0.0, -0.2],
            },
        ];
        let requests = [
            GpuContactSolveRequest {
                problem: &first,
                warm_start: Some(&seeds),
            },
            GpuContactSolveRequest {
                problem: &one_dof,
                warm_start: None,
            },
            GpuContactSolveRequest {
                problem: &empty,
                warm_start: None,
            },
            GpuContactSolveRequest {
                problem: &last,
                warm_start: None,
            },
        ];
        let packed = solver
            .solve_packed(&device, &queue, &requests, params())
            .unwrap();
        for (actual, request) in packed.iter().zip(requests) {
            let expected = solve_contacts(request.problem, params(), request.warm_start).unwrap();
            assert_eq!(actual.velocity.len(), expected.velocity.len());
            assert_eq!(actual.impulses.len(), expected.impulses.len());
            for (a, b) in actual.velocity.iter().zip(expected.velocity.iter()) {
                assert!((a - b).abs() < 1e-5, "{a} != {b}");
            }
            for (a, b) in actual.impulses.iter().zip(expected.impulses.iter()) {
                assert!((a.normal - b.normal).abs() < 1e-5);
                for (a, b) in a.tangents.iter().zip(b.tangents.iter()) {
                    assert!((a - b).abs() < 1e-5);
                }
            }
        }
        assert!(
            solver
                .solve_packed(&device, &queue, &[], params())
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            solver
                .solve_packed(&device, &queue, &requests[2..3], params())
                .unwrap()[0]
                .velocity,
            empty.velocity
        );
    }

    #[tokio::test]
    async fn independent_islands_run_in_separate_workgroups() {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let solver = GpuContactSolver::new(&device);
        let mut independent = ContactProblem {
            inverse_mass: DMatrix::identity(6, 6),
            velocity: DVector::from_vec(vec![1.0, 0.0, -2.0, -0.5, 0.0, -1.0]),
            contacts: vec![
                ContactConstraint {
                    scalar: None,
                    normal: DVector::from_vec(vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
                    tangents: [
                        DVector::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                        DVector::from_vec(vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
                    ],
                    penetration: 0.01,
                    friction: 0.5,
                    restitution: 0.2,
                },
                ContactConstraint {
                    scalar: None,
                    normal: DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
                    tangents: [
                        DVector::from_vec(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
                        DVector::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
                    ],
                    penetration: 0.0,
                    friction: 0.3,
                    restitution: 0.0,
                },
            ],
        };
        let mut jacobians = Vec::new();
        let mut responses = Vec::new();
        for contact in &independent.contacts {
            for row in [&contact.normal, &contact.tangents[0], &contact.tangents[1]] {
                jacobians.extend(row.iter().map(|value| *value as f32));
                responses.extend(row.iter().map(|value| *value as f32));
            }
        }
        let (ranges, _, indices) = partition_islands(6, 2, &jacobians, &responses);
        assert_eq!(ranges.len(), 2);
        assert_eq!(indices, vec![0, 1]);
        let prepared = prepare_problem(&device, &independent, params(), None)
            .unwrap()
            .unwrap();
        assert_eq!(prepared.islands[0].color_start_count[2], 2);
        assert_eq!(prepared.islands[1].color_start_count[2], 1);
        let cpu = solve_contacts(&independent, params(), None).unwrap();
        let gpu = solver
            .solve(&device, &queue, &independent, params(), None)
            .unwrap();
        for (actual, expected) in gpu.velocity.iter().zip(cpu.velocity.iter()) {
            assert!((actual - expected).abs() < 1e-5);
        }
        independent.inverse_mass[(2, 5)] = 0.1;
        independent.inverse_mass[(5, 2)] = 0.1;
        let mut coupled_responses = Vec::new();
        for contact in &independent.contacts {
            for row in [&contact.normal, &contact.tangents[0], &contact.tangents[1]] {
                coupled_responses.extend(
                    (&independent.inverse_mass * row)
                        .iter()
                        .map(|value| *value as f32),
                );
            }
        }
        let (ranges, _, _) = partition_islands(6, 2, &jacobians, &coupled_responses);
        assert_eq!(ranges.len(), 1);
        let cpu = solve_contacts(&independent, params(), None).unwrap();
        let gpu = solver
            .solve(&device, &queue, &independent, params(), None)
            .unwrap();
        for (actual, expected) in gpu.velocity.iter().zip(cpu.velocity.iter()) {
            assert!((actual - expected).abs() < 1e-5);
        }
    }

    #[tokio::test]
    async fn contact_colors_parallelize_disjoint_branches_without_reordering_dependencies() {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let normal = |a: usize, b: usize| {
            let mut row = DVector::zeros(4);
            row[a] = 1.0;
            row[b] = 1.0;
            row
        };
        let problem = ContactProblem {
            inverse_mass: DMatrix::identity(4, 4),
            velocity: DVector::from_vec(vec![-2.0, -1.0, -1.5, -0.5]),
            contacts: [(0, 1), (0, 2), (1, 3)]
                .map(|(a, b)| ContactConstraint {
                    scalar: None,
                    normal: normal(a, b),
                    tangents: [DVector::zeros(4), DVector::zeros(4)],
                    penetration: 0.01,
                    friction: 0.0,
                    restitution: 0.0,
                })
                .to_vec(),
        };
        let jacobians = problem
            .contacts
            .iter()
            .flat_map(|contact| {
                [&contact.normal, &contact.tangents[0], &contact.tangents[1]]
                    .into_iter()
                    .flat_map(|row| row.iter().map(|value| *value as f32))
            })
            .collect::<Vec<_>>();
        let (islands, colors, indices) = partition_islands(4, 3, &jacobians, &jacobians);
        assert_eq!(islands.len(), 1);
        assert_eq!(islands[0].color_start_count[1], 2);
        assert_eq!(colors[0].start_count, [0, 1]);
        assert_eq!(colors[1].start_count, [1, 2]);
        assert_eq!(indices, vec![0, 1, 2]);
        let chained = [(0, 1), (1, 2), (2, 3)]
            .into_iter()
            .flat_map(|(a, b)| {
                normal(a, b)
                    .iter()
                    .map(|value| *value as f32)
                    .chain(core::iter::repeat_n(0.0, 8))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let (_, chain_colors, chain_indices) = partition_islands(4, 3, &chained, &chained);
        assert_eq!(chain_colors.len(), 3);
        assert_eq!(chain_indices, vec![0, 1, 2]);
        let reference = solve_contacts(&problem, params(), None).unwrap();
        let solver = GpuContactSolver::new(&device);
        let first = solver
            .solve(&device, &queue, &problem, params(), None)
            .unwrap();
        let second = solver
            .solve(&device, &queue, &problem, params(), None)
            .unwrap();
        assert_eq!(first.velocity, second.velocity);
        assert_eq!(first.impulses, second.impulses);
        for (actual, expected) in first.velocity.iter().zip(reference.velocity.iter()) {
            assert!((actual - expected).abs() < 1e-5);
        }
        for (actual, expected) in first.impulses.iter().zip(reference.impulses.iter()) {
            assert!((actual.normal - expected.normal).abs() < 1e-5);
        }
    }

    #[tokio::test]
    async fn color_wave_larger_than_workgroup_preserves_independent_contacts() {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let width = 66;
        let constraint = |normal: DVector<f64>| ContactConstraint {
            scalar: None,
            normal,
            tangents: [DVector::zeros(width), DVector::zeros(width)],
            penetration: 0.0,
            friction: 0.0,
            restitution: 0.0,
        };
        let mut contacts = vec![constraint(DVector::from_element(width, 1.0))];
        for coordinate in 1..width {
            let mut normal = DVector::zeros(width);
            normal[coordinate] = 1.0;
            contacts.push(constraint(normal));
        }
        let velocity = DVector::from_iterator(
            width,
            (0..width).map(|index| if index % 2 == 0 { -2.0 } else { 0.0 }),
        );
        let problem = ContactProblem {
            inverse_mass: DMatrix::identity(width, width),
            velocity,
            contacts,
        };
        let prepared = prepare_problem(&device, &problem, params(), None)
            .unwrap()
            .unwrap();
        assert_eq!(prepared.islands.len(), 1);
        assert_eq!(prepared.colors.len(), 2);
        assert_eq!(prepared.colors[1].start_count, [1, 65]);
        let reference = solve_contacts(&problem, params(), None).unwrap();
        let actual = GpuContactSolver::new(&device)
            .solve(&device, &queue, &problem, params(), None)
            .unwrap();
        for (actual, expected) in actual.velocity.iter().zip(reference.velocity.iter()) {
            assert!((actual - expected).abs() < 1e-4, "{actual} != {expected}");
        }
    }

    #[tokio::test]
    async fn many_independent_islands_are_deterministic() {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let solver = GpuContactSolver::new(&device);
        let width = 64;
        let contacts = (0..width)
            .map(|coordinate| {
                let mut normal = DVector::zeros(width);
                normal[coordinate] = 1.0;
                ContactConstraint {
                    scalar: None,
                    normal,
                    tangents: [DVector::zeros(width), DVector::zeros(width)],
                    penetration: 0.0,
                    friction: 0.0,
                    restitution: 0.0,
                }
            })
            .collect::<Vec<_>>();
        let problem = ContactProblem {
            inverse_mass: DMatrix::identity(width, width),
            velocity: DVector::from_element(width, -1.0),
            contacts,
        };
        let jacobians = problem
            .contacts
            .iter()
            .flat_map(|contact| {
                [&contact.normal, &contact.tangents[0], &contact.tangents[1]]
                    .into_iter()
                    .flat_map(|row| row.iter().map(|value| *value as f32))
            })
            .collect::<Vec<_>>();
        let (ranges, _, _) = partition_islands(width, width, &jacobians, &jacobians);
        assert_eq!(ranges.len(), width);
        let reference = solve_contacts(&problem, params(), None).unwrap();
        let first = solver
            .solve(&device, &queue, &problem, params(), None)
            .unwrap();
        for _ in 0..3 {
            let again = solver
                .solve(&device, &queue, &problem, params(), None)
                .unwrap();
            assert_eq!(first.velocity, again.velocity);
            assert_eq!(first.impulses, again.impulses);
        }
        for (actual, expected) in first.velocity.iter().zip(reference.velocity.iter()) {
            assert!((actual - expected).abs() < 1e-5);
        }
    }

    #[cfg(windows)]
    #[tokio::test]
    async fn dx12_solves_generalized_contacts() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::DX12,
            ..Default::default()
        });
        let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
        else {
            eprintln!("DX12 adapter unavailable; solver test skipped");
            return;
        };
        eprintln!("DX12 solver adapter: {}", adapter.get_info().name);
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let input = problem();
        let reference = solve_contacts(&input, params(), None).unwrap();
        let solver = GpuContactSolver::new(&device);
        let actual = solver
            .solve(&device, &queue, &input, params(), None)
            .unwrap();
        for (actual, expected) in actual.velocity.iter().zip(reference.velocity.iter()) {
            assert!((actual - expected).abs() < 1e-5);
        }
        let requests = [
            GpuContactSolveRequest {
                problem: &input,
                warm_start: None,
            },
            GpuContactSolveRequest {
                problem: &input,
                warm_start: None,
            },
        ];
        let packed = solver
            .solve_packed(&device, &queue, &requests, params())
            .unwrap();
        for solution in packed {
            for (actual, expected) in solution.velocity.iter().zip(reference.velocity.iter()) {
                assert!((actual - expected).abs() < 1e-5);
            }
        }
    }

    #[tokio::test]
    async fn gpu_solver_validates_and_handles_contact_free_world() {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let solver = GpuContactSolver::new(&device);
        let empty = ContactProblem {
            inverse_mass: DMatrix::identity(1, 1),
            velocity: DVector::from_element(1, 3.0),
            contacts: Vec::new(),
        };
        let result = solver
            .solve(&device, &queue, &empty, params(), None)
            .unwrap();
        assert_eq!(result.velocity[0], 3.0);
        let mut invalid = problem();
        invalid.contacts[0].normal = DVector::zeros(1);
        assert!(matches!(
            solver.solve(&device, &queue, &invalid, params(), None),
            Err(GpuContactSolveError::Contact(ContactSolveError::Dimensions))
        ));
    }

    #[test]
    fn device_inverse_mass_solves_coupled_contact_and_warm_start_without_matrix_readback() {
        let system = GpuMassAssemblySystem {
            links: vec![GpuMassLink {
                mass: 2.0,
                inertia_world: Matrix3::from_diagonal_element(0.5),
                linear_jacobian: DMatrix::from_row_slice(3, 2, &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
                angular_jacobian: DMatrix::from_row_slice(3, 2, &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0]),
            }],
            armature: DVector::from_vec(vec![0.25, 0.5]),
            force: DVector::from_vec(vec![1.0, 2.0]),
        };
        let problem = ContactProblem {
            inverse_mass: DMatrix::from_row_slice(2, 2, &[2.75, 0.5, 0.5, 3.0])
                .try_inverse()
                .unwrap(),
            velocity: DVector::from_vec(vec![-1.0, 0.3]),
            contacts: vec![
                ContactConstraint {
                    scalar: None,
                    normal: DVector::from_vec(vec![1.0, 0.0]),
                    tangents: [DVector::from_vec(vec![0.0, 1.0]), DVector::zeros(2)],
                    penetration: 0.01,
                    friction: 0.5,
                    restitution: 0.2,
                },
                ContactConstraint::bounded_axis(DVector::from_vec(vec![-1.0, 1.0]), 0.3),
            ],
        };
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let mass = GpuArticulatedMassAssemblyBatch::new(
                context.device(),
                context.queue(),
                core::slice::from_ref(&system),
            )
            .unwrap();
            let solver = GpuContactSolver::new(context.device());
            for warm_start in [
                None,
                Some(vec![
                    ContactImpulse {
                        normal: 0.6,
                        tangents: [0.5, 0.0],
                    },
                    ContactImpulse {
                        normal: -0.5,
                        tangents: [0.0, 0.0],
                    },
                ]),
            ] {
                let mut encoder = context.device().create_command_encoder(&Default::default());
                mass.encode_with_inverse(&mut encoder);
                let output = solver
                    .encode_device_inverse_mass(
                        context.device(),
                        &mut encoder,
                        GpuDeviceMassContactRequest {
                            inverse_mass: mass.inverse_buffer().unwrap(),
                            inverse_range: mass.inverse_ranges()[0].clone(),
                            velocity: &problem.velocity,
                            contacts: &problem.contacts,
                            params: params(),
                            warm_start: warm_start.as_deref(),
                        },
                    )
                    .unwrap()
                    .unwrap();
                let readback = output.encode_readback(context.device(), &mut encoder);
                let _ = context.queue().submit(Some(encoder.finish()));
                let gpu = readback.finish(context.device()).unwrap();
                let cpu = solve_contacts(&problem, params(), warm_start.as_deref()).unwrap();
                for (actual, expected) in gpu.velocity.iter().zip(cpu.velocity.iter()) {
                    assert!(
                        (actual - expected).abs() < 1e-4,
                        "{backend:?}: {actual} != {expected}"
                    );
                }
                for (actual, expected) in gpu.impulses.iter().zip(&cpu.impulses) {
                    assert!(
                        (actual.normal - expected.normal).abs() < 1e-4,
                        "{backend:?}"
                    );
                    for (actual, expected) in actual.tangents.iter().zip(expected.tangents.iter()) {
                        assert!((actual - expected).abs() < 1e-4, "{backend:?}");
                    }
                }
            }
        }
        assert!(tested > 0);
    }

    #[test]
    fn device_inverse_mass_reports_invalid_matrix_and_singular_contact() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let solver = GpuContactSolver::new(context.device());
        let velocity = DVector::from_vec(vec![-1.0, 0.0]);
        for (values, normal) in [
            (
                [0.0f32, 0.0, 0.0, 0.0, 1.0],
                DVector::from_vec(vec![1.0, 0.0]),
            ),
            ([1.0f32, 0.0, 0.0, 1.0, 0.0], DVector::zeros(2)),
        ] {
            let inverse = context
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&values),
                    usage: wgpu::BufferUsages::STORAGE,
                });
            let contact = ContactConstraint {
                scalar: None,
                normal,
                tangents: [DVector::zeros(2), DVector::zeros(2)],
                penetration: 0.0,
                friction: 0.0,
                restitution: 0.0,
            };
            let mut encoder = context.device().create_command_encoder(&Default::default());
            let output = solver
                .encode_device_inverse_mass(
                    context.device(),
                    &mut encoder,
                    GpuDeviceMassContactRequest {
                        inverse_mass: &inverse,
                        inverse_range: 0..5,
                        velocity: &velocity,
                        contacts: &[contact],
                        params: params(),
                        warm_start: None,
                    },
                )
                .unwrap()
                .unwrap();
            let readback = output.encode_readback(context.device(), &mut encoder);
            let _ = context.queue().submit(Some(encoder.finish()));
            assert!(matches!(
                readback.finish(context.device()),
                Err(GpuContactSolveError::DeviceMass)
            ));
            let velocity_bytes =
                read_buffer(context.device(), context.queue(), output.velocity()).unwrap();
            let impulse_bytes =
                read_buffer(context.device(), context.queue(), output.impulses()).unwrap();
            assert_eq!(
                bytemuck::cast_slice::<u8, f32>(&velocity_bytes),
                &[-1.0, 0.0]
            );
            assert_eq!(bytemuck::cast_slice::<u8, f32>(&impulse_bytes), &[0.0; 4]);
        }
    }
}
