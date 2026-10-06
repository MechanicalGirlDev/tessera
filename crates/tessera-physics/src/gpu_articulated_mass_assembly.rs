//! GPU construction of reduced-coordinate mass equations from link Jacobians.

use core::ops::Range;
use std::sync::OnceLock;

use nalgebra::{DMatrix, DVector, Matrix3};
use wgpu::util::DeviceExt;

use crate::gpu_articulated_mass::{
    GpuArticulatedMassBatch, GpuArticulatedMassError, GpuArticulatedMassSystem, read_buffer,
};

/// One link's contribution to the generalized mass matrix.
#[derive(Debug, Clone)]
pub struct GpuMassLink {
    /// Link mass in kilograms; zero is allowed for a massless force frame.
    pub mass: f64,
    /// Center-of-mass inertia rotated into world coordinates.
    pub inertia_world: Matrix3<f64>,
    /// World linear velocity Jacobian, with three rows.
    pub linear_jacobian: DMatrix<f64>,
    /// World angular velocity Jacobian, with three rows.
    pub angular_jacobian: DMatrix<f64>,
}

/// One articulated equation assembled on GPU before its mass solve.
#[derive(Debug, Clone)]
pub struct GpuMassAssemblySystem {
    /// Link terms in stable order, optionally including massless force frames.
    pub links: Vec<GpuMassLink>,
    /// Diagonal mass terms, including reflected inertia and implicit passive corrections.
    pub armature: DVector<f64>,
    /// Gravity, motor, and external forces minus velocity bias.
    pub force: DVector<f64>,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AssemblyMeta {
    matrix_offset: u32,
    link_offset: u32,
    link_count: u32,
    vector_offset: u32,
    dimension: u32,
    inverse_offset: u32,
    _pad: [u32; 2],
}

/// Assembles link Jacobian products and solves the resulting equations on GPU.
///
/// Each environment owns one workgroup during assembly. The augmented matrix
/// stays on the device between assembly and the existing LU solve pass.
#[derive(Debug)]
pub struct GpuArticulatedMassAssemblyBatch {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    metadata: wgpu::Buffer,
    links: wgpu::Buffer,
    vectors: wgpu::Buffer,
    inverse_state: OnceLock<GpuInverseState>,
    expansion_pipeline: OnceLock<wgpu::ComputePipeline>,
    inverse_bytes: u64,
    solver: GpuArticulatedMassBatch,
    dimensions: Vec<usize>,
    link_counts: Vec<usize>,
    inverse_ranges: Vec<Range<usize>>,
}

#[derive(Debug)]
struct GpuInverseState {
    pipeline: wgpu::ComputePipeline,
    scratch: wgpu::Buffer,
    output: wgpu::Buffer,
}

impl GpuArticulatedMassAssemblyBatch {
    /// Upload link mass terms and allocate packed GPU matrix and solution buffers.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        systems: &[GpuMassAssemblySystem],
    ) -> Result<Self, GpuArticulatedMassError> {
        if systems.is_empty() {
            return Err(GpuArticulatedMassError::InvalidInput);
        }
        let mut metadata = Vec::with_capacity(systems.len());
        let mut links = Vec::<f32>::new();
        let mut vectors = Vec::<f32>::new();
        let mut dummy = Vec::with_capacity(systems.len());
        let mut dimensions = Vec::with_capacity(systems.len());
        let mut link_counts = Vec::with_capacity(systems.len());
        let mut inverse_ranges = Vec::with_capacity(systems.len());
        let mut matrix_offset = 0usize;
        let mut inverse_offset = 0usize;
        let mut mass_link_offset = 0usize;
        for system in systems {
            let n = system.force.len();
            if n == 0 || system.armature.len() != n {
                return Err(GpuArticulatedMassError::InvalidInput);
            }
            if n > 128 {
                return Err(GpuArticulatedMassError::Capacity);
            }
            let link_offset = links.len();
            let vector_offset = vectors.len();
            for link in &system.links {
                if link.mass < 0.0
                    || !link.mass.is_finite()
                    || link.linear_jacobian.shape() != (3, n)
                    || link.angular_jacobian.shape() != (3, n)
                {
                    return Err(GpuArticulatedMassError::InvalidInput);
                }
                links.push(finite_f32(link.mass)?);
                for row in 0..3 {
                    for col in 0..3 {
                        links.push(finite_f32(link.inertia_world[(row, col)])?);
                    }
                }
                for jacobian in [&link.linear_jacobian, &link.angular_jacobian] {
                    for row in 0..3 {
                        for col in 0..n {
                            links.push(finite_f32(jacobian[(row, col)])?);
                        }
                    }
                }
            }
            for value in system.armature.iter().chain(system.force.iter()) {
                vectors.push(finite_f32(*value)?);
            }
            metadata.push(AssemblyMeta {
                matrix_offset: checked_u32(matrix_offset)?,
                link_offset: checked_u32(link_offset)?,
                link_count: checked_u32(system.links.len())?,
                vector_offset: checked_u32(vector_offset)?,
                dimension: checked_u32(n)?,
                inverse_offset: checked_u32(inverse_offset)?,
                _pad: [checked_u32(mass_link_offset)?, 0],
            });
            mass_link_offset = mass_link_offset
                .checked_add(system.links.len())
                .ok_or(GpuArticulatedMassError::Capacity)?;
            matrix_offset = matrix_offset
                .checked_add(
                    n.checked_mul(n + 1)
                        .ok_or(GpuArticulatedMassError::Capacity)?,
                )
                .ok_or(GpuArticulatedMassError::Capacity)?;
            let inverse_end = inverse_offset
                .checked_add(
                    n.checked_mul(n)
                        .and_then(|count| count.checked_add(1))
                        .ok_or(GpuArticulatedMassError::Capacity)?,
                )
                .ok_or(GpuArticulatedMassError::Capacity)?;
            inverse_ranges.push(inverse_offset..inverse_end);
            inverse_offset = inverse_end;
            dummy.push(GpuArticulatedMassSystem {
                mass: DMatrix::zeros(n, n),
                force: DVector::zeros(n),
            });
            dimensions.push(n);
            link_counts.push(system.links.len());
        }
        if links.is_empty() {
            links.push(0.0);
        }
        let _ = checked_u32(inverse_offset)?;
        let inverse_bytes = checked_bytes(inverse_offset, size_of::<f32>())?;
        let limits = device.limits();
        for size in [
            checked_bytes(metadata.len(), size_of::<AssemblyMeta>())?,
            checked_bytes(links.len(), size_of::<f32>())?,
            checked_bytes(vectors.len(), size_of::<f32>())?,
            inverse_bytes,
        ] {
            if size > limits.max_buffer_size
                || size > u64::from(limits.max_storage_buffer_binding_size)
            {
                return Err(GpuArticulatedMassError::Capacity);
            }
        }
        if systems.len() > limits.max_compute_workgroups_per_dimension as usize
            || limits.max_storage_buffers_per_shader_stage < 4
        {
            return Err(GpuArticulatedMassError::Capacity);
        }
        let solver = GpuArticulatedMassBatch::new(device, queue, &dummy)?;
        let metadata = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera articulated mass assembly metadata"),
            contents: bytemuck::cast_slice(&metadata),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let links = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera articulated link mass terms"),
            contents: bytemuck::cast_slice(&links),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let vectors = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera articulated armature and force"),
            contents: bytemuck::cast_slice(&vectors),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera articulated mass assembly"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("gpu_articulated_mass_assembly.wgsl").into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tessera articulated mass assembly"),
            layout: None,
            module: &module,
            entry_point: Some("assemble"),
            compilation_options: Default::default(),
            cache: None,
        });
        Ok(Self {
            device: device.clone(),
            queue: queue.clone(),
            pipeline,
            metadata,
            links,
            vectors,
            inverse_state: OnceLock::new(),
            expansion_pipeline: OnceLock::new(),
            inverse_bytes,
            solver,
            dimensions,
            link_counts,
            inverse_ranges,
        })
    }

    /// Expand one inverse matrix into contact coordinates without CPU matrix upload.
    /// Absent coordinates have exactly zero inverse mass, for prescribed motion.
    /// Encode `encode_with_inverse` before this pass, or submit its work earlier
    /// on this queue. The source matrix status is preserved in the output.
    pub fn encode_expanded_inverse(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        environment: usize,
        coordinates: &[Option<usize>],
    ) -> Result<wgpu::Buffer, GpuArticulatedMassError> {
        let dimension = *self
            .dimensions
            .get(environment)
            .ok_or(GpuArticulatedMassError::InvalidInput)?;
        if coordinates.is_empty() || coordinates.iter().flatten().any(|&axis| axis >= dimension) {
            return Err(GpuArticulatedMassError::InvalidInput);
        }
        let count = coordinates
            .len()
            .checked_mul(coordinates.len())
            .and_then(|count| count.checked_add(1))
            .ok_or(GpuArticulatedMassError::InvalidInput)?;
        let bytes = checked_bytes(count, size_of::<f32>())?;
        let groups = checked_u32(count)?.div_ceil(64);
        if bytes > self.device.limits().max_storage_buffer_binding_size as u64
            || groups > self.device.limits().max_compute_workgroups_per_dimension
        {
            return Err(GpuArticulatedMassError::InvalidInput);
        }
        let mut mapping = vec![
            checked_u32(self.inverse_ranges[environment].start)?,
            checked_u32(dimension)?,
            checked_u32(coordinates.len())?,
            0,
        ];
        for coordinate in coordinates {
            mapping.push(match coordinate {
                Some(axis) => checked_u32(*axis)?,
                None => u32::MAX,
            });
        }
        let mapping = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tessera prescribed contact coordinate map"),
                contents: bytemuck::cast_slice(&mapping),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let output = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera expanded contact inverse mass"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let pipeline = self.expansion_pipeline.get_or_init(|| {
            let module = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Tessera contact inverse expansion"),
                    source: wgpu::ShaderSource::Wgsl(
                        include_str!("gpu_articulated_inverse_expand.wgsl").into(),
                    ),
                });
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Tessera contact inverse expansion"),
                    layout: None,
                    module: &module,
                    entry_point: Some("expand"),
                    compilation_options: Default::default(),
                    cache: None,
                })
        });
        let bindings = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera contact inverse expansion bindings"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.inverse_state().output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: mapping.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bindings, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
        drop(pass);
        Ok(output)
    }

    fn inverse_state(&self) -> &GpuInverseState {
        self.inverse_state.get_or_init(|| {
            let scratch = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Tessera articulated inverse mass scratch"),
                size: self.solver.augmented_buffer().size(),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let output = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Tessera articulated inverse mass matrices"),
                size: self.inverse_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let module = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Tessera articulated inverse mass"),
                    source: wgpu::ShaderSource::Wgsl(
                        include_str!("gpu_articulated_mass_inverse.wgsl").into(),
                    ),
                });
            let pipeline = self
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Tessera articulated inverse mass"),
                    layout: None,
                    module: &module,
                    entry_point: Some("invert"),
                    compilation_options: Default::default(),
                    cache: None,
                });
            GpuInverseState {
                pipeline,
                scratch,
                output,
            }
        })
    }

    /// Whether all environments retain their generalized and link dimensions.
    pub fn has_layout(&self, systems: &[GpuMassAssemblySystem]) -> bool {
        systems.len() == self.dimensions.len()
            && systems
                .iter()
                .zip(self.dimensions.iter().zip(&self.link_counts))
                .all(|(system, (&dimension, &links))| {
                    system.force.len() == dimension && system.links.len() == links
                })
    }

    pub(crate) fn dimensions(&self) -> &[usize] {
        &self.dimensions
    }
    pub(crate) fn link_counts(&self) -> &[usize] {
        &self.link_counts
    }
    pub(crate) fn link_terms_buffer(&self) -> &wgpu::Buffer {
        &self.links
    }
    pub(crate) fn metadata_buffer(&self) -> &wgpu::Buffer {
        &self.metadata
    }
    pub(crate) fn vectors_buffer(&self) -> &wgpu::Buffer {
        &self.vectors
    }

    /// Replace link terms and forces without reallocating GPU buffers.
    pub fn update(&self, systems: &[GpuMassAssemblySystem]) -> Result<(), GpuArticulatedMassError> {
        if !self.has_layout(systems) {
            return Err(GpuArticulatedMassError::InvalidInput);
        }
        let mut links = Vec::<f32>::with_capacity(self.links.size() as usize / 4);
        let mut vectors = Vec::<f32>::with_capacity(self.vectors.size() as usize / 4);
        for (system, &n) in systems.iter().zip(&self.dimensions) {
            if system.armature.len() != n {
                return Err(GpuArticulatedMassError::InvalidInput);
            }
            for link in &system.links {
                if link.mass < 0.0
                    || !link.mass.is_finite()
                    || link.linear_jacobian.shape() != (3, n)
                    || link.angular_jacobian.shape() != (3, n)
                {
                    return Err(GpuArticulatedMassError::InvalidInput);
                }
                links.push(finite_f32(link.mass)?);
                for row in 0..3 {
                    for col in 0..3 {
                        links.push(finite_f32(link.inertia_world[(row, col)])?);
                    }
                }
                for jacobian in [&link.linear_jacobian, &link.angular_jacobian] {
                    for row in 0..3 {
                        for col in 0..n {
                            links.push(finite_f32(jacobian[(row, col)])?);
                        }
                    }
                }
            }
            for value in system.armature.iter().chain(system.force.iter()) {
                vectors.push(finite_f32(*value)?);
            }
        }
        if links.is_empty() {
            links.push(0.0);
        }
        if links.len() * 4 != self.links.size() as usize
            || vectors.len() * 4 != self.vectors.size() as usize
        {
            return Err(GpuArticulatedMassError::InvalidInput);
        }
        self.queue
            .write_buffer(&self.links, 0, bytemuck::cast_slice(&links));
        self.queue
            .write_buffer(&self.vectors, 0, bytemuck::cast_slice(&vectors));
        Ok(())
    }

    /// Encode matrix assembly followed by LU solve into one command encoder.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        self.encode_assembly(encoder);
        self.solver.encode(encoder);
    }

    /// Encode acceleration and inverse mass matrices for contact coupling.
    ///
    /// The assembled matrix is copied on GPU before LU mutates its input.
    pub fn encode_with_inverse(&self, encoder: &mut wgpu::CommandEncoder) {
        let inverse = self.inverse_state();
        self.encode_assembly(encoder);
        encoder.copy_buffer_to_buffer(
            self.solver.augmented_buffer(),
            0,
            &inverse.scratch,
            0,
            inverse.scratch.size(),
        );
        self.solver.encode(encoder);
        let buffers = [&self.metadata, &inverse.scratch, &inverse.output];
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera articulated inverse mass bindings"),
            layout: &inverse.pipeline.get_bind_group_layout(0),
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
            label: Some("Tessera articulated inverse mass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&inverse.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(self.dimensions.len().div_ceil(64) as u32, 1, 1);
    }

    fn encode_assembly(&self, encoder: &mut wgpu::CommandEncoder) {
        let buffers = [
            &self.metadata,
            &self.links,
            &self.vectors,
            self.solver.augmented_buffer(),
        ];
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera articulated mass assembly bindings"),
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
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Tessera articulated mass assembly"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(self.dimensions.len() as u32, 1, 1);
        }
    }

    /// Submit matrix assembly and solve on the owning queue.
    pub fn submit(&self) {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.encode(&mut encoder);
        let _ = self.queue.submit(Some(encoder.finish()));
    }

    /// Submit acceleration and inverse-mass calculations on the owning queue.
    pub fn submit_with_inverse(&self) {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.encode_with_inverse(&mut encoder);
        let _ = self.queue.submit(Some(encoder.finish()));
    }

    /// Read submitted accelerations, including singular-system status.
    pub fn readback(&self) -> Result<Vec<DVector<f64>>, GpuArticulatedMassError> {
        self.solver.readback()
    }

    /// Read inverse mass matrices from the last inverse-enabled submission.
    pub fn readback_inverse(&self) -> Result<Vec<DMatrix<f64>>, GpuArticulatedMassError> {
        let inverse = self
            .inverse_state
            .get()
            .ok_or(GpuArticulatedMassError::InvalidInput)?;
        let bytes = read_buffer(&self.device, &self.queue, &inverse.output)?;
        self.inverse_ranges
            .iter()
            .zip(&self.dimensions)
            .enumerate()
            .map(|(index, (range, &n))| {
                let status_offset = (range.end - 1) * size_of::<f32>();
                let status = f32::from_le_bytes(
                    bytes[status_offset..status_offset + 4]
                        .try_into()
                        .map_err(|_| GpuArticulatedMassError::Capacity)?,
                );
                if status == 1.0 {
                    return Err(GpuArticulatedMassError::Singular(index));
                }
                if status != 0.0 {
                    return Err(GpuArticulatedMassError::NonFinite(index));
                }
                let mut values = Vec::with_capacity(n * n);
                for element in range.start..range.end - 1 {
                    let offset = element * size_of::<f32>();
                    let value = f32::from_le_bytes(
                        bytes[offset..offset + 4]
                            .try_into()
                            .map_err(|_| GpuArticulatedMassError::Capacity)?,
                    );
                    if !value.is_finite() {
                        return Err(GpuArticulatedMassError::NonFinite(index));
                    }
                    values.push(f64::from(value));
                }
                Ok(DMatrix::from_row_slice(n, n, &values))
            })
            .collect()
    }

    /// Packed row-major inverse matrices after the inverse pass is first encoded.
    /// Each matrix has one trailing status f32.
    pub fn inverse_buffer(&self) -> Option<&wgpu::Buffer> {
        self.inverse_state.get().map(|state| &state.output)
    }

    pub(crate) fn inverse_buffer_or_init(&self) -> &wgpu::Buffer {
        &self.inverse_state().output
    }

    /// Element ranges for each matrix and its trailing status value.
    pub fn inverse_ranges(&self) -> &[Range<usize>] {
        &self.inverse_ranges
    }

    /// Device-resident acceleration vector in concatenated environment order.
    pub fn solution_buffer(&self) -> &wgpu::Buffer {
        self.solver.solution_buffer()
    }

    /// Packed acceleration ranges in the same order as the input systems.
    pub fn solution_ranges(&self) -> &[Range<usize>] {
        self.solver.solution_ranges()
    }

    /// GPU per-system mass solve status, where zero means success.
    pub fn status_buffer(&self) -> &wgpu::Buffer {
        self.solver.status_buffer()
    }

    pub(crate) fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub(crate) fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }
}

fn finite_f32(value: f64) -> Result<f32, GpuArticulatedMassError> {
    let narrowed = value as f32;
    if value.is_finite() && narrowed.is_finite() {
        Ok(narrowed)
    } else {
        Err(GpuArticulatedMassError::InvalidInput)
    }
}

fn checked_u32(value: usize) -> Result<u32, GpuArticulatedMassError> {
    u32::try_from(value).map_err(|_| GpuArticulatedMassError::Capacity)
}

fn checked_bytes(count: usize, element_size: usize) -> Result<u64, GpuArticulatedMassError> {
    count
        .checked_mul(element_size)
        .and_then(|size| u64::try_from(size).ok())
        .ok_or(GpuArticulatedMassError::Capacity)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_contact_pipeline::GpuContactDevice;

    #[test]
    fn link_jacobians_assemble_and_solve_multiple_dimensions() {
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
                    linear_jacobian: DMatrix::from_row_slice(3, 1, &[2.0, 0.0, 0.0]),
                    angular_jacobian: DMatrix::zeros(3, 1),
                }],
                armature: DVector::zeros(1),
                force: DVector::from_element(1, 6.0),
            },
            GpuMassAssemblySystem {
                links: Vec::new(),
                armature: DVector::from_element(1, 4.0),
                force: DVector::from_element(1, 2.0),
            },
        ];
        let expected_mass = DMatrix::from_row_slice(2, 2, &[2.75, 0.5, 0.5, 3.0]);
        let expected = expected_mass.clone().lu().solve(&systems[0].force).unwrap();
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let batch =
                GpuArticulatedMassAssemblyBatch::new(context.device(), context.queue(), &systems)
                    .unwrap();
            assert!(batch.inverse_buffer().is_none());
            assert!(matches!(
                batch.readback_inverse(),
                Err(GpuArticulatedMassError::InvalidInput)
            ));
            batch.submit_with_inverse();
            assert!(batch.inverse_buffer().is_some());
            let solved = batch.readback().unwrap();
            assert!((&solved[0] - &expected).norm() < 1e-5, "{backend:?}");
            assert!((solved[1][0] - 0.5).abs() < 1e-6, "{backend:?}");
            assert!((solved[2][0] - 0.5).abs() < 1e-6, "{backend:?}");
            let inverse = batch.readback_inverse().unwrap();
            assert!(
                (&expected_mass * &inverse[0] - DMatrix::identity(2, 2)).norm() < 1e-5,
                "{backend:?}"
            );
            assert!((inverse[1][(0, 0)] - 1.0 / 12.0).abs() < 1e-6);
            assert!((inverse[2][(0, 0)] - 0.25).abs() < 1e-6);
            let mut changed = systems.clone();
            changed[1].force[0] = 12.0;
            batch.update(&changed).unwrap();
            batch.submit_with_inverse();
            assert!((batch.readback().unwrap()[1][0] - 1.0).abs() < 1e-6);
            changed[0].links[0].mass = f64::NAN;
            assert!(matches!(
                batch.update(&changed),
                Err(GpuArticulatedMassError::InvalidInput)
            ));
            batch.submit_with_inverse();
            assert!((batch.readback().unwrap()[1][0] - 1.0).abs() < 1e-6);
            assert!((batch.readback_inverse().unwrap()[1][(0, 0)] - 1.0 / 12.0).abs() < 1e-6);

            let singular = GpuArticulatedMassAssemblyBatch::new(
                context.device(),
                context.queue(),
                &[GpuMassAssemblySystem {
                    links: Vec::new(),
                    armature: DVector::zeros(1),
                    force: DVector::from_element(1, 1.0),
                }],
            )
            .unwrap();
            singular.submit_with_inverse();
            assert!(matches!(
                singular.readback_inverse(),
                Err(GpuArticulatedMassError::Singular(0))
            ));
        }
        assert!(tested > 0);
    }
}
