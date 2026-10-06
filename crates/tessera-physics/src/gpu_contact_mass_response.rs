//! GPU mass responses for reduced-coordinate contact Jacobian rows.

use core::{mem::size_of, ops::Range};

use nalgebra::DMatrix;
use wgpu::util::DeviceExt;

use crate::gpu_articulated_mass::{GpuArticulatedMassError, read_buffer};

/// Contact rows for one device-resident inverse mass matrix.
#[derive(Debug, Clone)]
pub struct GpuContactMassResponseSystem {
    /// Element range containing a row-major inverse matrix and trailing status.
    pub inverse_range: Range<usize>,
    /// One contact Jacobian row per matrix row, in generalized-coordinate order.
    pub jacobians: DMatrix<f64>,
}

/// Input or GPU result failure while calculating contact mass responses.
#[derive(Debug, thiserror::Error)]
pub enum GpuContactMassResponseError {
    /// Matrix layout, row count, or Jacobian value is invalid.
    #[error("invalid contact mass response input")]
    InvalidInput,
    /// Packed data exceeds WebGPU buffer or dispatch limits.
    #[error("contact mass response exceeds GPU capacity")]
    Capacity,
    /// An inverse matrix failed its GPU factorization.
    #[error("inverse mass matrix {0} is singular or ill-conditioned")]
    Singular(usize),
    /// An inverse matrix or response produced a non-finite result.
    #[error("contact mass response {0} is non-finite")]
    NonFinite(usize),
    /// Result transfer failed.
    #[error(transparent)]
    Readback(#[from] GpuArticulatedMassError),
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RowMeta {
    inverse_offset: u32,
    jacobian_offset: u32,
    dimension: u32,
    status_offset: u32,
}

/// CPU diagnostic result for one articulated system.
#[derive(Debug, Clone)]
pub struct GpuContactMassResponseResult {
    /// Rows of `M^-1 J^T`, in the same order as the input Jacobian rows.
    pub responses: DMatrix<f64>,
    /// Diagonal effective masses `J M^-1 J^T` for each row.
    pub effective: Vec<f64>,
}

/// Reusable GPU transform from contact Jacobians to impulse velocity responses.
///
/// Encode this after `GpuArticulatedMassAssemblyBatch::encode_with_inverse` in the
/// same command encoder. The inverse matrix, responses, and effective masses
/// remain device-resident for later contact passes.
#[derive(Debug)]
pub struct GpuContactMassResponseBatch {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    jacobians: wgpu::Buffer,
    _metadata: wgpu::Buffer,
    responses: wgpu::Buffer,
    effective: wgpu::Buffer,
    status: wgpu::Buffer,
    row_ranges: Vec<Range<usize>>,
    element_ranges: Vec<Range<usize>>,
    dimensions: Vec<usize>,
    row_count: u32,
}

impl GpuContactMassResponseBatch {
    /// Bind packed contact rows to an already allocated inverse mass buffer.
    pub fn new(
        device: &wgpu::Device,
        inverse_mass: &wgpu::Buffer,
        systems: &[GpuContactMassResponseSystem],
    ) -> Result<Self, GpuContactMassResponseError> {
        if systems.is_empty() || !inverse_mass.usage().contains(wgpu::BufferUsages::STORAGE) {
            return Err(GpuContactMassResponseError::InvalidInput);
        }
        if inverse_mass.size() > u64::from(device.limits().max_storage_buffer_binding_size) {
            return Err(GpuContactMassResponseError::Capacity);
        }
        let mut metadata = Vec::<RowMeta>::new();
        let mut jacobians = Vec::<f32>::new();
        let mut row_ranges = Vec::with_capacity(systems.len());
        let mut element_ranges = Vec::with_capacity(systems.len());
        let mut dimensions = Vec::with_capacity(systems.len());
        for system in systems {
            let n = system.jacobians.ncols();
            let count = system.jacobians.nrows();
            if n == 0
                || count == 0
                || system.inverse_range.start >= system.inverse_range.end
                || n.checked_mul(n).and_then(|size| size.checked_add(1))
                    != Some(system.inverse_range.end - system.inverse_range.start)
                || system.jacobians.iter().any(|value| !value.is_finite())
            {
                return Err(GpuContactMassResponseError::InvalidInput);
            }
            let matrix_end_bytes = system
                .inverse_range
                .end
                .checked_mul(size_of::<f32>())
                .ok_or(GpuContactMassResponseError::Capacity)?;
            if matrix_end_bytes as u64 > inverse_mass.size() {
                return Err(GpuContactMassResponseError::InvalidInput);
            }
            let row_start = metadata.len();
            let element_start = jacobians.len();
            for row in 0..count {
                metadata.push(RowMeta {
                    inverse_offset: checked_u32(system.inverse_range.start)?,
                    jacobian_offset: checked_u32(jacobians.len())?,
                    dimension: checked_u32(n)?,
                    status_offset: checked_u32(system.inverse_range.end - 1)?,
                });
                for column in 0..n {
                    let value = system.jacobians[(row, column)] as f32;
                    if !value.is_finite() {
                        return Err(GpuContactMassResponseError::InvalidInput);
                    }
                    jacobians.push(value);
                }
            }
            row_ranges.push(row_start..metadata.len());
            element_ranges.push(element_start..jacobians.len());
            dimensions.push(n);
        }
        let row_count = checked_u32(metadata.len())?;
        if metadata.len().div_ceil(64)
            > device.limits().max_compute_workgroups_per_dimension as usize
        {
            return Err(GpuContactMassResponseError::Capacity);
        }
        let max_storage = u64::from(device.limits().max_storage_buffer_binding_size);
        for bytes in [
            checked_bytes(metadata.len(), size_of::<RowMeta>())?,
            checked_bytes(jacobians.len(), size_of::<f32>())?,
            checked_bytes(metadata.len(), size_of::<f32>())?,
        ] {
            if bytes > max_storage || bytes > device.limits().max_buffer_size {
                return Err(GpuContactMassResponseError::Capacity);
            }
        }
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera contact mass response"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_contact_mass_response.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tessera contact mass response"),
            layout: None,
            module: &module,
            entry_point: Some("calculate"),
            compilation_options: Default::default(),
            cache: None,
        });
        let jacobian_buffer = storage_input(device, "Tessera contact Jacobian rows", &jacobians);
        let metadata_buffer = storage_input(device, "Tessera contact mass row metadata", &metadata);
        let response_buffer = storage_output(
            device,
            "Tessera contact mass responses",
            checked_bytes(jacobians.len(), 4)?,
        );
        let effective_buffer = storage_output(
            device,
            "Tessera contact effective masses",
            checked_bytes(metadata.len(), 4)?,
        );
        let status_buffer = storage_output(
            device,
            "Tessera contact mass status",
            checked_bytes(metadata.len(), 4)?,
        );
        let buffers = [
            inverse_mass,
            &jacobian_buffer,
            &metadata_buffer,
            &response_buffer,
            &effective_buffer,
            &status_buffer,
        ];
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera contact mass response buffers"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &buffers
                .iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        Ok(Self {
            pipeline,
            bind_group,
            jacobians: jacobian_buffer,
            _metadata: metadata_buffer,
            responses: response_buffer,
            effective: effective_buffer,
            status: status_buffer,
            row_ranges,
            element_ranges,
            dimensions,
            row_count,
        })
    }

    /// Write all response rows and effective masses without CPU readback.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Tessera contact mass response"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.row_count.div_ceil(64), 1, 1);
    }

    /// Update Jacobian values while preserving row counts, dimensions, and inverse ranges.
    pub fn update_jacobians(
        &self,
        queue: &wgpu::Queue,
        jacobians: &[DMatrix<f64>],
    ) -> Result<(), GpuContactMassResponseError> {
        if jacobians.len() != self.row_ranges.len() {
            return Err(GpuContactMassResponseError::InvalidInput);
        }
        let mut values = Vec::<f32>::with_capacity(self.jacobians.size() as usize / 4);
        for ((rows, &dimension), matrix) in
            self.row_ranges.iter().zip(&self.dimensions).zip(jacobians)
        {
            if matrix.shape() != (rows.len(), dimension) {
                return Err(GpuContactMassResponseError::InvalidInput);
            }
            for row in 0..matrix.nrows() {
                for column in 0..matrix.ncols() {
                    let input = matrix[(row, column)];
                    let value = input as f32;
                    if !input.is_finite() || !value.is_finite() {
                        return Err(GpuContactMassResponseError::InvalidInput);
                    }
                    values.push(value);
                }
            }
        }
        queue.write_buffer(&self.jacobians, 0, bytemuck::cast_slice(&values));
        Ok(())
    }

    /// Device-resident `M^-1 J^T` values in packed row-major order.
    pub fn responses_buffer(&self) -> &wgpu::Buffer {
        &self.responses
    }

    /// Device-resident contact Jacobian rows in the same packed order.
    pub fn jacobians_buffer(&self) -> &wgpu::Buffer {
        &self.jacobians
    }

    /// Device-resident diagonal `J M^-1 J^T` values in packed row order.
    pub fn effective_buffer(&self) -> &wgpu::Buffer {
        &self.effective
    }

    /// Device-resident row status values; zero means success.
    pub fn status_buffer(&self) -> &wgpu::Buffer {
        &self.status
    }

    /// Element ranges corresponding to each input system's response rows.
    pub fn element_ranges(&self) -> &[Range<usize>] {
        &self.element_ranges
    }

    /// Row ranges corresponding to each input system's effective masses.
    pub fn row_ranges(&self) -> &[Range<usize>] {
        &self.row_ranges
    }

    /// Read back and validate results after the encoded work is submitted.
    pub fn readback(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<GpuContactMassResponseResult>, GpuContactMassResponseError> {
        let response_bytes = read_buffer(device, queue, &self.responses)?;
        let effective_bytes = read_buffer(device, queue, &self.effective)?;
        let status_bytes = read_buffer(device, queue, &self.status)?;
        let response_values = bytemuck::cast_slice::<u8, f32>(&response_bytes);
        let effective_values = bytemuck::cast_slice::<u8, f32>(&effective_bytes);
        let status_values = bytemuck::cast_slice::<u8, f32>(&status_bytes);
        self.row_ranges
            .iter()
            .zip(&self.element_ranges)
            .zip(&self.dimensions)
            .enumerate()
            .map(|(system_index, ((rows, elements), &dimension))| {
                for &status in &status_values[rows.clone()] {
                    if status == 1.0 {
                        return Err(GpuContactMassResponseError::Singular(system_index));
                    }
                    if status != 0.0 {
                        return Err(GpuContactMassResponseError::NonFinite(system_index));
                    }
                }
                let values = response_values[elements.clone()]
                    .iter()
                    .map(|&v| f64::from(v))
                    .collect::<Vec<_>>();
                let effective = effective_values[rows.clone()]
                    .iter()
                    .map(|&v| f64::from(v))
                    .collect::<Vec<_>>();
                if values
                    .iter()
                    .chain(&effective)
                    .any(|value| !value.is_finite())
                {
                    return Err(GpuContactMassResponseError::NonFinite(system_index));
                }
                Ok(GpuContactMassResponseResult {
                    responses: DMatrix::from_row_slice(rows.len(), dimension, &values),
                    effective,
                })
            })
            .collect()
    }
}

fn checked_u32(value: usize) -> Result<u32, GpuContactMassResponseError> {
    u32::try_from(value).map_err(|_| GpuContactMassResponseError::Capacity)
}

fn checked_bytes(count: usize, element_size: usize) -> Result<u64, GpuContactMassResponseError> {
    count
        .checked_mul(element_size)
        .and_then(|value| u64::try_from(value).ok())
        .ok_or(GpuContactMassResponseError::Capacity)
}

fn storage_input<T: bytemuck::Pod>(
    device: &wgpu::Device,
    label: &str,
    values: &[T],
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(values),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    })
}

fn storage_output(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DVector, Matrix3};

    use crate::gpu_articulated_mass_assembly::{
        GpuArticulatedMassAssemblyBatch, GpuMassAssemblySystem, GpuMassLink,
    };
    use crate::gpu_contact_pipeline::GpuContactDevice;

    #[test]
    fn device_inverse_feeds_contact_rows_without_intermediate_readback() {
        let systems = [
            GpuMassAssemblySystem {
                links: vec![GpuMassLink {
                    mass: 2.0,
                    inertia_world: Matrix3::from_diagonal_element(0.5),
                    linear_jacobian: DMatrix::from_row_slice(3, 2, &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
                    angular_jacobian: DMatrix::from_row_slice(
                        3,
                        2,
                        &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                    ),
                }],
                armature: DVector::from_vec(vec![0.25, 0.5]),
                force: DVector::from_vec(vec![1.0, 2.0]),
            },
            GpuMassAssemblySystem {
                links: vec![GpuMassLink {
                    mass: 3.0,
                    inertia_world: Matrix3::identity(),
                    linear_jacobian: DMatrix::from_row_slice(3, 1, &[1.0, 0.0, 0.0]),
                    angular_jacobian: DMatrix::zeros(3, 1),
                }],
                armature: DVector::from_element(1, 0.25),
                force: DVector::from_element(1, 1.0),
            },
        ];
        let rows = [
            DMatrix::from_row_slice(3, 2, &[1.0, 0.0, 0.0, 1.0, 1.0, -1.0]),
            DMatrix::from_row_slice(1, 1, &[2.0]),
        ];
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let mass =
                GpuArticulatedMassAssemblyBatch::new(context.device(), context.queue(), &systems)
                    .unwrap();
            let response_systems = mass
                .inverse_ranges()
                .iter()
                .cloned()
                .zip(&rows)
                .map(|(inverse_range, jacobians)| GpuContactMassResponseSystem {
                    inverse_range,
                    jacobians: jacobians.clone(),
                })
                .collect::<Vec<_>>();
            let mut encoder = context.device().create_command_encoder(&Default::default());
            mass.encode_with_inverse(&mut encoder);
            let response = GpuContactMassResponseBatch::new(
                context.device(),
                mass.inverse_buffer().unwrap(),
                &response_systems,
            )
            .unwrap();
            assert!(
                response
                    .responses_buffer()
                    .usage()
                    .contains(wgpu::BufferUsages::STORAGE)
            );
            assert_eq!(response.row_ranges(), &[0..3, 3..4]);
            assert_eq!(response.element_ranges(), &[0..6, 6..7]);
            response.encode(&mut encoder);
            let _ = context.queue().submit(Some(encoder.finish()));
            let actual = response
                .readback(context.device(), context.queue())
                .unwrap();
            let matrices = [
                DMatrix::from_row_slice(2, 2, &[2.75, 0.5, 0.5, 3.0]),
                DMatrix::from_row_slice(1, 1, &[3.25]),
            ];
            for ((result, jacobian), matrix) in actual.iter().zip(&rows).zip(&matrices) {
                let inverse = matrix.clone().try_inverse().unwrap();
                for row in 0..jacobian.nrows() {
                    let j = jacobian.row(row).transpose();
                    let expected = &inverse * &j;
                    for column in 0..jacobian.ncols() {
                        assert!(
                            (result.responses[(row, column)] - expected[column]).abs() < 2e-5,
                            "{backend:?}"
                        );
                    }
                    assert!(
                        (result.effective[row] - j.dot(&expected)).abs() < 2e-5,
                        "{backend:?}"
                    );
                }
            }
            let updated_rows = [
                DMatrix::from_row_slice(3, 2, &[0.0, 1.0, 1.0, 1.0, -2.0, 0.0]),
                DMatrix::from_row_slice(1, 1, &[-1.0]),
            ];
            response
                .update_jacobians(context.queue(), &updated_rows)
                .unwrap();
            let mut encoder = context.device().create_command_encoder(&Default::default());
            response.encode(&mut encoder);
            let _ = context.queue().submit(Some(encoder.finish()));
            let updated = response
                .readback(context.device(), context.queue())
                .unwrap();
            for ((result, jacobian), matrix) in updated.iter().zip(&updated_rows).zip(&matrices) {
                let inverse = matrix.clone().try_inverse().unwrap();
                for row in 0..jacobian.nrows() {
                    let j = jacobian.row(row).transpose();
                    let expected = &inverse * &j;
                    for column in 0..jacobian.ncols() {
                        assert!(
                            (result.responses[(row, column)] - expected[column]).abs() < 2e-5,
                            "{backend:?}"
                        );
                    }
                    assert!(
                        (result.effective[row] - j.dot(&expected)).abs() < 2e-5,
                        "{backend:?}"
                    );
                }
            }
        }
        assert!(tested > 0);
    }

    #[test]
    fn rejects_mismatched_inverse_layout_and_nonfinite_rows() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let inverse = context.device().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        for system in [
            GpuContactMassResponseSystem {
                inverse_range: 0..3,
                jacobians: DMatrix::from_row_slice(1, 2, &[1.0, 0.0]),
            },
            GpuContactMassResponseSystem {
                inverse_range: 0..2,
                jacobians: DMatrix::from_row_slice(1, 1, &[f64::NAN]),
            },
        ] {
            assert!(matches!(
                GpuContactMassResponseBatch::new(context.device(), &inverse, &[system]),
                Err(GpuContactMassResponseError::InvalidInput)
            ));
        }
    }

    #[test]
    fn propagates_inverse_failure_status() {
        let Ok(context) = GpuContactDevice::new() else {
            return;
        };
        let inverse = context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&[0.0f32, 1.0]),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let system = GpuContactMassResponseSystem {
            inverse_range: 0..2,
            jacobians: DMatrix::from_row_slice(1, 1, &[1.0]),
        };
        let response =
            GpuContactMassResponseBatch::new(context.device(), &inverse, &[system]).unwrap();
        let mut encoder = context.device().create_command_encoder(&Default::default());
        response.encode(&mut encoder);
        let _ = context.queue().submit(Some(encoder.finish()));
        assert!(matches!(
            response.readback(context.device(), context.queue()),
            Err(GpuContactMassResponseError::Singular(0))
        ));
    }
}
