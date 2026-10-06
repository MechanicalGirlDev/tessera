//! Prescribed rigid-body motion independent of contact pair allocation.

use core::ops::Range;
use nalgebra::{Isometry3, Quaternion, Translation3, UnitQuaternion, Vector3};
use wgpu::util::DeviceExt;

/// World-space prescribed pose and constant velocities for one body.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuKinematicBody {
    /// Body origin and orientation in world coordinates.
    pub pose: Isometry3<f64>,
    /// Velocity of the body origin in world coordinates.
    pub linear_velocity: Vector3<f64>,
    /// Angular velocity in world coordinates.
    pub angular_velocity: Vector3<f64>,
}

/// Invalid input, device capacity, failed integration, or readback failure.
#[derive(Debug, thiserror::Error)]
pub enum GpuKinematicBodyError {
    /// Empty batch, invalid timestep, non-finite values, or layout mismatch.
    #[error("invalid prescribed body input")]
    InvalidInput,
    /// The packed bodies exceed the selected device's limits.
    #[error("prescribed body batch exceeds GPU capacity")]
    Capacity,
    /// A body overflowed; its previous pose remains stored until reset.
    #[error("prescribed body {0} integration failed")]
    Faulted(usize),
    /// GPU readback failed.
    #[error("prescribed body readback failed: {0}")]
    Readback(#[from] crate::gpu_articulated_mass::GpuArticulatedMassError),
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PackedBody {
    origin: [f32; 4],
    orientation: [f32; 4],
    linear: [f32; 4],
    angular: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct PackedKinematicTranslation {
    origin_elapsed: [f32; 4],
    expected_position: [f32; 4],
    linear_velocity: [f32; 4],
}

fn pack(body: &GpuKinematicBody) -> Result<PackedBody, GpuKinematicBodyError> {
    fn vector(values: impl Iterator<Item = f64>) -> Result<[f32; 4], GpuKinematicBodyError> {
        let mut result = [0.0; 4];
        for (index, value) in values.enumerate() {
            let converted = value as f32;
            if !value.is_finite() || !converted.is_finite() || (converted == 0.0 && value != 0.0) {
                return Err(GpuKinematicBodyError::InvalidInput);
            }
            result[index] = converted;
        }
        Ok(result)
    }
    let orientation = vector(body.pose.rotation.coords.iter().copied())?;
    let norm = orientation.iter().map(|v| v * v).sum::<f32>();
    if (norm - 1.0).abs() > 1e-4 {
        return Err(GpuKinematicBodyError::InvalidInput);
    }
    Ok(PackedBody {
        origin: vector(body.pose.translation.vector.iter().copied())?,
        orientation,
        linear: vector(body.linear_velocity.iter().copied())?,
        angular: vector(body.angular_velocity.iter().copied())?,
    })
}

/// Packed prescribed bodies with persistent GPU poses and per-body fault flags.
/// Empty individual environments are supported. At least one body is required.
/// Integration never requires colliders or contact rows. Encode this pass in
/// submission order with downstream collision passes on the owning queue.
#[derive(Debug)]
pub struct GpuKinematicBodyBatch {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bindings: wgpu::BindGroup,
    bodies: wgpu::Buffer,
    translation: wgpu::Buffer,
    status: wgpu::Buffer,
    ranges: Vec<Range<usize>>,
    count: u32,
}

impl GpuKinematicBodyBatch {
    /// Upload bodies in environment order with a fixed integration timestep.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        environments: &[Vec<GpuKinematicBody>],
        timestep: f64,
    ) -> Result<Self, GpuKinematicBodyError> {
        let dt = timestep as f32;
        if environments.is_empty() || !timestep.is_finite() || !dt.is_finite() || dt <= 0.0 {
            return Err(GpuKinematicBodyError::InvalidInput);
        }
        let mut ranges = Vec::with_capacity(environments.len());
        let mut total = 0usize;
        for environment in environments {
            let end = total
                .checked_add(environment.len())
                .ok_or(GpuKinematicBodyError::Capacity)?;
            ranges.push(total..end);
            total = end;
        }
        if total == 0 {
            return Err(GpuKinematicBodyError::InvalidInput);
        }
        let count = u32::try_from(total).map_err(|_| GpuKinematicBodyError::Capacity)?;
        let bytes = u64::from(count) * size_of::<PackedBody>() as u64;
        let limits = device.limits();
        if bytes > limits.max_buffer_size
            || bytes > u64::from(limits.max_storage_buffer_binding_size)
            || count.div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || limits.max_storage_buffers_per_shader_stage < 3
        {
            return Err(GpuKinematicBodyError::Capacity);
        }
        let packed = environments
            .iter()
            .flatten()
            .map(pack)
            .collect::<Result<Vec<_>, _>>()?;
        let bodies = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera prescribed body state"),
            contents: bytemuck::cast_slice(&packed),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let status = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera prescribed body status"),
            contents: bytemuck::cast_slice(&vec![0u32; total]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let translation = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera prescribed translation intervals"),
            size: u64::from(count) * size_of::<PackedKinematicTranslation>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let settings = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera prescribed body timestep"),
            contents: bytemuck::cast_slice(&[dt.to_bits(), count, 0u32, 0u32]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera prescribed body integration"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gpu_kinematic_body.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tessera prescribed body integration"),
            layout: None,
            module: &module,
            entry_point: Some("integrate"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera prescribed body bindings"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[&bodies, &status, &settings, &translation]
                .iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        Ok(Self {
            device: device.clone(),
            queue: queue.clone(),
            pipeline,
            bindings,
            bodies,
            translation,
            status,
            ranges,
            count,
        })
    }

    /// Encode one timestep. Faulted bodies remain frozen until an explicit reset.
    pub fn encode_step(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Tessera prescribed body integration"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bindings, &[]);
        pass.dispatch_workgroups(self.count.div_ceil(64), 1, 1);
    }

    /// Submit repeated timesteps without intermediate CPU readback.
    pub fn submit_steps(&self, count: usize) -> Result<(), GpuKinematicBodyError> {
        if count == 0 {
            return Err(GpuKinematicBodyError::InvalidInput);
        }
        let mut encoder = self.device.create_command_encoder(&Default::default());
        for _ in 0..count {
            self.encode_step(&mut encoder);
        }
        let _submission = self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    /// Packed state buffer, containing four vec4 values per body in input order.
    pub fn state_buffer(&self) -> &wgpu::Buffer {
        &self.bodies
    }

    /// Stable body ranges for each environment, including empty environments.
    pub fn body_ranges(&self) -> &[Range<usize>] {
        &self.ranges
    }

    /// Replace all poses and velocities and clear faults after validating all inputs.
    /// Submit any previously encoded commands before resetting.
    pub fn reset(
        &self,
        environments: &[Vec<GpuKinematicBody>],
    ) -> Result<(), GpuKinematicBodyError> {
        if environments.len() != self.ranges.len()
            || environments
                .iter()
                .zip(&self.ranges)
                .any(|(e, r)| e.len() != r.len())
        {
            return Err(GpuKinematicBodyError::InvalidInput);
        }
        let packed = environments
            .iter()
            .flatten()
            .map(pack)
            .collect::<Result<Vec<_>, _>>()?;
        self.queue
            .write_buffer(&self.bodies, 0, bytemuck::cast_slice(&packed));
        self.queue.write_buffer(
            &self.status,
            0,
            bytemuck::cast_slice(&vec![0u32; self.count as usize]),
        );
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.clear_buffer(&self.translation, 0, None);
        let _ = self.queue.submit([encoder.finish()]);
        Ok(())
    }

    /// Download poses and velocities, rejecting any fault before returning states.
    pub fn readback(&self) -> Result<Vec<Vec<GpuKinematicBody>>, GpuKinematicBodyError> {
        use crate::gpu_articulated_mass::read_buffer;
        let status = read_buffer(&self.device, &self.queue, &self.status)?;
        for (index, value) in status.chunks_exact(4).enumerate() {
            if u32::from_ne_bytes(
                value
                    .try_into()
                    .map_err(|_| GpuKinematicBodyError::InvalidInput)?,
            ) != 0
            {
                return Err(GpuKinematicBodyError::Faulted(index));
            }
        }
        let bytes = read_buffer(&self.device, &self.queue, &self.bodies)?;
        let packed = bytes
            .chunks_exact(size_of::<PackedBody>())
            .map(bytemuck::pod_read_unaligned::<PackedBody>)
            .collect::<Vec<_>>();
        self.ranges
            .iter()
            .map(|range| {
                packed[range.clone()]
                    .iter()
                    .enumerate()
                    .map(|(offset, body)| {
                        if !body
                            .origin
                            .iter()
                            .chain(&body.orientation)
                            .chain(&body.linear)
                            .chain(&body.angular)
                            .all(|v| v.is_finite())
                        {
                            return Err(GpuKinematicBodyError::Faulted(range.start + offset));
                        }
                        let q = body.orientation.map(f64::from);
                        let rotation = Quaternion::new(q[3], q[0], q[1], q[2]);
                        if (rotation.norm_squared() - 1.0).abs() > 1e-3 {
                            return Err(GpuKinematicBodyError::Faulted(range.start + offset));
                        }
                        let vector = |v: [f32; 4]| {
                            Vector3::new(f64::from(v[0]), f64::from(v[1]), f64::from(v[2]))
                        };
                        Ok(GpuKinematicBody {
                            pose: Isometry3::from_parts(
                                Translation3::from(vector(body.origin)),
                                UnitQuaternion::new_normalize(rotation),
                            ),
                            linear_velocity: vector(body.linear),
                            angular_velocity: vector(body.angular),
                        })
                    })
                    .collect()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_contact_pipeline::GpuContactDevice;

    #[test]
    fn independent_prescribed_small_displacements_and_reset_preserve_precision() {
        let small = GpuKinematicBody {
            pose: Isometry3::translation(1.0, 2.0, -3.0),
            linear_velocity: Vector3::new(0.2, -0.3, 0.1),
            angular_velocity: Vector3::zeros(),
        };
        let mut large = small.clone();
        large.pose = Isometry3::translation(10000.0, -10000.0, 1.0);
        let input = vec![vec![small.clone(), large.clone()], vec![]];
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(gpu) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            eprintln!("independent translation interval backend: {backend:?}");
            let batch =
                GpuKinematicBodyBatch::new(gpu.device(), gpu.queue(), &input, 0.001).unwrap();
            batch.submit_steps(100).unwrap();
            let output = batch.readback().unwrap();
            for (index, initial) in [small.clone(), large.clone()].iter().enumerate() {
                for axis in 0..3 {
                    let expected = (initial.pose.translation.vector[axis]
                        + (initial.linear_velocity[axis] as f32) as f64 * 0.1)
                        as f32 as f64;
                    let actual = output[0][index].pose.translation.vector[axis];
                    if index == 0 {
                        assert!((actual - expected).abs() < 1e-6);
                    } else {
                        assert_eq!(actual, expected);
                    }
                }
            }
            assert!(output[1].is_empty());
            let mut reset = output.clone();
            reset[0][0].linear_velocity = Vector3::y() * 0.2;
            reset[0][1].linear_velocity = Vector3::zeros();
            batch.reset(&reset).unwrap();
            batch.submit_steps(100).unwrap();
            let advanced = batch.readback().unwrap();
            assert_eq!(
                advanced[0][0].pose.translation.x,
                reset[0][0].pose.translation.x
            );
            assert_eq!(
                advanced[0][0].pose.translation.z,
                reset[0][0].pose.translation.z
            );
            assert!(
                (advanced[0][0].pose.translation.y - reset[0][0].pose.translation.y - 0.02).abs()
                    < 1e-6
            );
            assert_eq!(advanced[0][1].pose, reset[0][1].pose);
        }
        assert!(tested > 0);
    }

    #[test]
    fn independent_prescribed_bodies_integrate_and_fault_on_available_backends() {
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(gpu) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            eprintln!("prescribed body backend: {:?}", gpu.adapter_info());
            let body = GpuKinematicBody {
                pose: Isometry3::from_parts(
                    Translation3::new(1.0, 2.0, 3.0),
                    UnitQuaternion::from_euler_angles(0.2, -0.3, 0.4),
                ),
                linear_velocity: Vector3::new(0.2, -0.1, 0.3),
                angular_velocity: Vector3::new(0.3, -0.2, 0.1),
            };
            let mut second = body.clone();
            second.linear_velocity = -body.linear_velocity;
            second.angular_velocity = Vector3::zeros();
            let input = vec![
                vec![],
                vec![body.clone()],
                vec![second.clone(), body.clone()],
            ];
            let batch =
                GpuKinematicBodyBatch::new(gpu.device(), gpu.queue(), &input, 0.001).unwrap();
            assert_eq!(batch.body_ranges(), &[0..0, 0..1, 1..3]);
            batch.submit_steps(100).unwrap();
            let output = batch.readback().unwrap();
            assert!(output[0].is_empty());
            for (actual, initial) in output.iter().flatten().zip(input.iter().flatten()) {
                let position = initial.pose.translation.vector + initial.linear_velocity * 0.1;
                let rotation = UnitQuaternion::from_scaled_axis(initial.angular_velocity * 0.1)
                    * initial.pose.rotation;
                assert!((actual.pose.translation.vector - position).norm() < 3e-5);
                assert!(actual.pose.rotation.angle_to(&rotation) < 3e-5);
            }
            let mut invalid = input.clone();
            invalid[2][1].linear_velocity.x = f64::INFINITY;
            assert!(batch.reset(&invalid).is_err());
            assert_eq!(batch.readback().unwrap()[1][0].pose, output[1][0].pose);
            batch.reset(&input).unwrap();
            batch.submit_steps(1).unwrap();
            assert!((batch.readback().unwrap()[1][0].pose.translation.x - 1.0002).abs() < 1e-6);
            let mut overflow = body.clone();
            overflow.pose.translation.x = 3e38;
            overflow.linear_velocity.x = 3e38;
            let faulty =
                GpuKinematicBodyBatch::new(gpu.device(), gpu.queue(), &[vec![overflow]], 1.0)
                    .unwrap();
            faulty.submit_steps(2).unwrap();
            assert!(matches!(
                faulty.readback(),
                Err(GpuKinematicBodyError::Faulted(0))
            ));
            faulty.reset(&[vec![body]]).unwrap();
            faulty.submit_steps(1).unwrap();
            assert!(faulty.readback().is_ok());
        }
        assert!(
            tested > 0,
            "no hardware backend available for prescribed body regression"
        );
    }
}
