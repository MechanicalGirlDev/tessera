//! Batched GPU solves for reduced-coordinate multibody mass matrices.

use core::ops::Range;
use core::time::Duration;
use std::sync::mpsc;

use nalgebra::{DMatrix, DVector};
use wgpu::util::DeviceExt;

const MAX_DIMENSION: usize = 128;

/// One generalized mass equation, `mass * acceleration = force`.
#[derive(Debug, Clone)]
pub struct GpuArticulatedMassSystem {
    /// Symmetric generalized mass matrix in generalized-coordinate order.
    pub mass: DMatrix<f64>,
    /// Applied force after gravity, motor, and velocity-bias terms.
    pub force: DVector<f64>,
}

/// Invalid input, GPU capacity, or failed matrix solve.
#[derive(Debug, thiserror::Error)]
pub enum GpuArticulatedMassError {
    /// Non-finite input, a dimension mismatch, or an empty batch.
    #[error("invalid articulated mass system")]
    InvalidInput,
    /// A dimension or packed buffer exceeds the supported GPU capacity.
    #[error("articulated mass solve exceeds GPU capacity")]
    Capacity,
    /// A mass matrix is singular or too ill-conditioned for f32 elimination.
    #[error("articulated mass matrix {0} is singular or ill-conditioned")]
    Singular(usize),
    /// A result overflowed or became non-finite.
    #[error("articulated mass solution {0} is non-finite")]
    NonFinite(usize),
    /// GPU readback failed.
    #[error("articulated mass GPU readback failed: {0}")]
    Readback(String),
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuMassMeta {
    matrix_offset: u32,
    solution_offset: u32,
    dimension: u32,
    _pad: u32,
}

/// GPU-resident, independently solvable generalized mass equations.
///
/// Matrices and forces are converted from f64 to f32 at construction. `encode`
/// writes acceleration vectors to `solution_buffer` without CPU readback, so
/// later GPU passes can consume them on the same queue. Each encoded solve
/// modifies its augmented matrix. Call [`Self::update`] before solving another
/// frame with the same number of systems and dimensions.
#[derive(Debug)]
pub struct GpuArticulatedMassBatch {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    augmented: wgpu::Buffer,
    metadata: wgpu::Buffer,
    solution: wgpu::Buffer,
    status: wgpu::Buffer,
    ranges: Vec<Range<usize>>,
}

impl GpuArticulatedMassBatch {
    /// Upload independent mass equations using the caller's wgpu device.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        systems: &[GpuArticulatedMassSystem],
    ) -> Result<Self, GpuArticulatedMassError> {
        if systems.is_empty() {
            return Err(GpuArticulatedMassError::InvalidInput);
        }
        let mut data_capacity = 0usize;
        let mut solution_capacity = 0usize;
        for system in systems {
            let n = system.force.len();
            if n == 0 || system.mass.nrows() != n || system.mass.ncols() != n {
                return Err(GpuArticulatedMassError::InvalidInput);
            }
            if n > MAX_DIMENSION {
                return Err(GpuArticulatedMassError::Capacity);
            }
            data_capacity = data_capacity
                .checked_add(
                    n.checked_mul(n + 1)
                        .ok_or(GpuArticulatedMassError::Capacity)?,
                )
                .ok_or(GpuArticulatedMassError::Capacity)?;
            solution_capacity = solution_capacity
                .checked_add(n)
                .ok_or(GpuArticulatedMassError::Capacity)?;
        }
        let limits = device.limits();
        if u32::try_from(data_capacity).is_err()
            || u32::try_from(solution_capacity).is_err()
            || [
                bytes_for_f32(data_capacity)?,
                bytes_for_f32(solution_capacity)?,
                u64::try_from(systems.len())
                    .ok()
                    .and_then(|count| count.checked_mul(size_of::<GpuMassMeta>() as u64))
                    .ok_or(GpuArticulatedMassError::Capacity)?,
                u64::try_from(systems.len())
                    .ok()
                    .and_then(|count| count.checked_mul(size_of::<u32>() as u64))
                    .ok_or(GpuArticulatedMassError::Capacity)?,
            ]
            .iter()
            .any(|&bytes| {
                bytes > limits.max_buffer_size
                    || bytes > u64::from(limits.max_storage_buffer_binding_size)
            })
            || systems.len().div_ceil(64) > limits.max_compute_workgroups_per_dimension as usize
            || limits.max_storage_buffers_per_shader_stage < 4
        {
            return Err(GpuArticulatedMassError::Capacity);
        }
        let mut data = Vec::<f32>::with_capacity(data_capacity);
        let mut metadata = Vec::with_capacity(systems.len());
        let mut ranges = Vec::with_capacity(systems.len());
        let mut solutions = 0usize;
        for system in systems {
            let n = system.force.len();
            let matrix_offset =
                u32::try_from(data.len()).map_err(|_| GpuArticulatedMassError::Capacity)?;
            let solution_offset =
                u32::try_from(solutions).map_err(|_| GpuArticulatedMassError::Capacity)?;
            let dimension = u32::try_from(n).map_err(|_| GpuArticulatedMassError::Capacity)?;
            for row in 0..n {
                for col in 0..n {
                    data.push(finite_f32(system.mass[(row, col)])?);
                }
                data.push(finite_f32(system.force[row])?);
            }
            let end = solutions
                .checked_add(n)
                .ok_or(GpuArticulatedMassError::Capacity)?;
            ranges.push(solutions..end);
            solutions = end;
            metadata.push(GpuMassMeta {
                matrix_offset,
                solution_offset,
                dimension,
                _pad: 0,
            });
        }
        let data_bytes = bytes_for_f32(data.len())?;
        let solution_bytes = bytes_for_f32(solutions)?;
        let metadata_bytes = u64::try_from(metadata.len())
            .ok()
            .and_then(|count| count.checked_mul(size_of::<GpuMassMeta>() as u64))
            .ok_or(GpuArticulatedMassError::Capacity)?;
        let status_bytes = u64::try_from(systems.len())
            .ok()
            .and_then(|count| count.checked_mul(size_of::<u32>() as u64))
            .ok_or(GpuArticulatedMassError::Capacity)?;
        debug_assert_eq!(data_bytes, bytes_for_f32(data_capacity)?);
        debug_assert_eq!(solution_bytes, bytes_for_f32(solution_capacity)?);
        debug_assert_eq!(
            metadata_bytes,
            systems.len() as u64 * size_of::<GpuMassMeta>() as u64
        );
        debug_assert_eq!(status_bytes, systems.len() as u64 * size_of::<u32>() as u64);
        let augmented = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera articulated augmented mass matrices"),
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let metadata_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera articulated mass metadata"),
            contents: bytemuck::cast_slice(&metadata),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let solution = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera articulated generalized acceleration"),
            size: solution_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let status = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera articulated mass solve status"),
            size: status_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera articulated mass solve"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_articulated_mass.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tessera articulated mass solve"),
            layout: None,
            module: &module,
            entry_point: Some("solve"),
            compilation_options: Default::default(),
            cache: None,
        });
        Ok(Self {
            device: device.clone(),
            queue: queue.clone(),
            pipeline,
            augmented,
            metadata: metadata_buffer,
            solution,
            status,
            ranges,
        })
    }

    /// Replace equations without rebuilding the pipeline or GPU buffers.
    ///
    /// The number and dimensions of systems must match construction. All
    /// values are validated before any buffer is changed. Submit an encoded
    /// solve before updating the equations for the next frame.
    pub fn update(
        &self,
        systems: &[GpuArticulatedMassSystem],
    ) -> Result<(), GpuArticulatedMassError> {
        if systems.len() != self.ranges.len() {
            return Err(GpuArticulatedMassError::InvalidInput);
        }
        let mut data = Vec::with_capacity(self.augmented.size() as usize / size_of::<f32>());
        for (system, range) in systems.iter().zip(&self.ranges) {
            let n = range.len();
            if system.force.len() != n || system.mass.nrows() != n || system.mass.ncols() != n {
                return Err(GpuArticulatedMassError::InvalidInput);
            }
            for row in 0..n {
                for col in 0..n {
                    data.push(finite_f32(system.mass[(row, col)])?);
                }
                data.push(finite_f32(system.force[row])?);
            }
        }
        debug_assert_eq!(
            data.len() * size_of::<f32>(),
            self.augmented.size() as usize
        );
        self.queue
            .write_buffer(&self.augmented, 0, bytemuck::cast_slice(&data));
        Ok(())
    }

    /// Whether the systems fit the existing packed buffer layout.
    pub fn has_layout(&self, systems: &[GpuArticulatedMassSystem]) -> bool {
        systems.len() == self.ranges.len()
            && systems
                .iter()
                .zip(&self.ranges)
                .all(|(system, range)| system.force.len() == range.len())
    }

    /// Encode all mass solves in parallel, leaving results on the GPU.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera articulated mass solve bindings"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                &self.augmented,
                &self.metadata,
                &self.solution,
                &self.status,
            ]
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>(),
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Tessera articulated mass solve"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(self.ranges.len().div_ceil(64) as u32, 1, 1);
    }

    /// GPU solution buffer in concatenated system order.
    pub fn solution_buffer(&self) -> &wgpu::Buffer {
        &self.solution
    }

    /// Packed solution ranges in the same order as the input systems.
    pub fn solution_ranges(&self) -> &[Range<usize>] {
        &self.ranges
    }

    pub(crate) fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub(crate) fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Packed augmented matrix buffer, writable by an earlier GPU pass.
    pub(crate) fn augmented_buffer(&self) -> &wgpu::Buffer {
        &self.augmented
    }

    /// GPU per-system status buffer, where zero means success.
    pub fn status_buffer(&self) -> &wgpu::Buffer {
        &self.status
    }

    /// Submit the solve on its owning queue.
    pub fn submit(&self) {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.encode(&mut encoder);
        let _ = self.queue.submit(Some(encoder.finish()));
    }

    /// Read submitted results and report singular or non-finite systems.
    pub fn readback(&self) -> Result<Vec<DVector<f64>>, GpuArticulatedMassError> {
        let values = read_buffer(&self.device, &self.queue, &self.solution)?;
        let flags = read_buffer(&self.device, &self.queue, &self.status)?;
        for (index, flag) in flags.chunks_exact(4).enumerate() {
            match u32::from_le_bytes(
                flag.try_into()
                    .map_err(|_| GpuArticulatedMassError::Capacity)?,
            ) {
                0 => {}
                1 => return Err(GpuArticulatedMassError::Singular(index)),
                _ => return Err(GpuArticulatedMassError::NonFinite(index)),
            }
        }
        self.ranges
            .iter()
            .map(|range| {
                let solution = range
                    .clone()
                    .map(|index| {
                        let offset = index * 4;
                        let bytes: [u8; 4] = values[offset..offset + 4]
                            .try_into()
                            .map_err(|_| GpuArticulatedMassError::Capacity)?;
                        Ok(f64::from(f32::from_le_bytes(bytes)))
                    })
                    .collect::<Result<Vec<_>, GpuArticulatedMassError>>()?;
                Ok(DVector::from_vec(solution))
            })
            .collect()
    }
}

fn finite_f32(value: f64) -> Result<f32, GpuArticulatedMassError> {
    let converted = value as f32;
    if !value.is_finite() || !converted.is_finite() || (converted == 0.0 && value != 0.0) {
        return Err(GpuArticulatedMassError::InvalidInput);
    }
    Ok(converted)
}

fn bytes_for_f32(count: usize) -> Result<u64, GpuArticulatedMassError> {
    u64::try_from(count)
        .ok()
        .and_then(|count| count.checked_mul(size_of::<f32>() as u64))
        .ok_or(GpuArticulatedMassError::Capacity)
}

pub(crate) fn read_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
) -> Result<Vec<u8>, GpuArticulatedMassError> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Tessera articulated mass readback"),
        size: source.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, source.size());
    let submission = queue.submit(Some(encoder.finish()));
    let (sender, receiver) = mpsc::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
    let _ = device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(Duration::from_secs(30)),
        })
        .map_err(|error| GpuArticulatedMassError::Readback(error.to_string()))?;
    receiver
        .recv_timeout(Duration::from_secs(30))
        .map_err(|error| GpuArticulatedMassError::Readback(error.to_string()))?
        .map_err(|error| GpuArticulatedMassError::Readback(error.to_string()))?;
    let view = staging.slice(..).get_mapped_range();
    let result = view.to_vec();
    drop(view);
    staging.unmap();
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_contact_pipeline::GpuContactDevice;

    #[test]
    fn batched_mass_solve_matches_cpu_on_available_gpu_backends() {
        let systems = vec![
            GpuArticulatedMassSystem {
                mass: DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 3.0]),
                force: DVector::from_vec(vec![2.0, -1.0]),
            },
            GpuArticulatedMassSystem {
                mass: DMatrix::from_row_slice(
                    6,
                    6,
                    &[
                        3.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 4.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.2, 5.0,
                        0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 2.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.2, 2.5, 0.1,
                        0.0, 0.0, 0.0, 0.0, 0.1, 3.5,
                    ],
                ),
                force: DVector::from_vec(vec![1.0, 2.0, -1.0, 0.5, 0.25, 3.0]),
            },
        ];
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let batch =
                GpuArticulatedMassBatch::new(context.device(), context.queue(), &systems).unwrap();
            assert!(
                batch
                    .solution_buffer()
                    .usage()
                    .contains(wgpu::BufferUsages::STORAGE)
            );
            batch.submit();
            let results = batch.readback().unwrap();
            for (system, actual) in systems.iter().zip(&results) {
                let expected = system.mass.clone().lu().solve(&system.force).unwrap();
                assert!((actual - expected).norm() < 2e-5, "{backend:?}");
            }

            let mut next = systems.clone();
            next[0].mass[(0, 0)] = 5.0;
            next[0].force[1] = 3.0;
            next[1].force[4] = -2.0;
            batch.update(&next).unwrap();
            batch.submit();
            for (system, actual) in next.iter().zip(batch.readback().unwrap()) {
                let expected = system.mass.clone().lu().solve(&system.force).unwrap();
                assert!((actual - expected).norm() < 2e-5, "{backend:?}");
            }
            let before_invalid = batch.readback().unwrap();
            let mut invalid = next.clone();
            invalid[1].force[0] = f64::NAN;
            assert!(matches!(
                batch.update(&invalid),
                Err(GpuArticulatedMassError::InvalidInput)
            ));
            assert!(matches!(
                batch.update(&next[..1]),
                Err(GpuArticulatedMassError::InvalidInput)
            ));
            assert_eq!(batch.readback().unwrap(), before_invalid);

            let singular = GpuArticulatedMassBatch::new(
                context.device(),
                context.queue(),
                &[GpuArticulatedMassSystem {
                    mass: DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 4.0]),
                    force: DVector::from_vec(vec![1.0, 1.0]),
                }],
            )
            .unwrap();
            singular.submit();
            assert!(matches!(
                singular.readback(),
                Err(GpuArticulatedMassError::Singular(0))
            ));
            singular.update(&next[..1]).unwrap();
            singular.submit();
            let recovered = singular.readback().unwrap();
            let expected = next[0].mass.clone().lu().solve(&next[0].force).unwrap();
            assert!((&recovered[0] - expected).norm() < 2e-5, "{backend:?}");
        }
        assert!(tested > 0);
    }
}
