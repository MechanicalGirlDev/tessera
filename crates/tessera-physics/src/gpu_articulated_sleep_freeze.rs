//! GPU freezing of sleeping generalized coordinates before integration.

use core::ops::Range;
use wgpu::util::DeviceExt;

use crate::gpu_articulated_mass_assembly::GpuArticulatedMassAssemblyBatch;
use crate::gpu_articulated_state::GpuGeneralizedStateBatch;

/// Invalid ownership, incompatible buffers, or unsupported GPU capacity.
#[derive(Debug, thiserror::Error)]
pub enum GpuSleepFreezeError {
    /// Layouts or local link owners are incompatible.
    #[error("invalid articulated sleep freeze input")]
    InvalidInput,
    /// Packed metadata exceeds the owning device's limits.
    #[error("articulated sleep freeze exceeds GPU capacity")]
    Capacity,
}

/// Stops both velocity and acceleration without changing coordinate positions.
///
/// Encode after contact, wake propagation and sleep evaluation, before scalar,
/// spherical and floating-root integration. The caller must provide complete
/// kinematic owners and group dynamically coupled coordinates consistently.
/// Candidate flags alone are insufficient unless support and all wake sources
/// have been evaluated. This pass does not implement that policy.
#[derive(Debug)]
pub struct GpuArticulatedSleepFreezeBatch {
    device: wgpu::Device,
    pipeline: wgpu::ComputePipeline,
    buffers: [wgpu::Buffer; 8],
    count: u32,
}

impl GpuArticulatedSleepFreezeBatch {
    /// Bind matching resident state, mass solution and per-link sleeping flags.
    /// Owners are local link indices, in environment/generalized-coordinate order.
    /// Empty owners leave a coordinate active. Invalid input uploads nothing.
    pub fn new(
        state: &GpuGeneralizedStateBatch,
        mass: &GpuArticulatedMassAssemblyBatch,
        sleeping: &wgpu::Buffer,
        link_ranges: &[Range<usize>],
        owners: &[Vec<Vec<usize>>],
    ) -> Result<Self, GpuSleepFreezeError> {
        if mass
            .dimensions()
            .iter()
            .copied()
            .ne(state.ranges().iter().map(Range::len))
        {
            return Err(GpuSleepFreezeError::InvalidInput);
        }
        let (coordinates, packed_owners) = pack_owners(state.ranges(), link_ranges, owners)?;
        if sleeping.size() != link_ranges.last().map_or(0, |r| r.end) as u64 * 4
            || sleeping.size() == 0
            || !sleeping.usage().contains(wgpu::BufferUsages::STORAGE)
            || mass.solution_buffer().size() != state.velocity_buffer().size()
        {
            return Err(GpuSleepFreezeError::InvalidInput);
        }
        Self::build(
            state.device(),
            &coordinates,
            &packed_owners,
            [
                sleeping,
                state.velocity_buffer(),
                mass.solution_buffer(),
                state.mass_status_buffer(),
                state.status_buffer(),
            ],
        )
    }

    fn build(
        device: &wgpu::Device,
        coordinates: &[[u32; 4]],
        owners: &[u32],
        inputs: [&wgpu::Buffer; 5],
    ) -> Result<Self, GpuSleepFreezeError> {
        let count = u32::try_from(coordinates.len()).map_err(|_| GpuSleepFreezeError::Capacity)?;
        let owner_data = if owners.is_empty() {
            &[0u32][..]
        } else {
            owners
        };
        let sizes = [
            size_of_val(coordinates) as u64,
            size_of_val(owner_data) as u64,
            u64::from(count) * 4,
        ];
        let limits = device.limits();
        if count == 0
            || limits.max_storage_buffers_per_shader_stage < 8
            || count.div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || sizes.iter().any(|&size| {
                size > limits.max_buffer_size
                    || size > u64::from(limits.max_storage_buffer_binding_size)
            })
        {
            return Err(GpuSleepFreezeError::Capacity);
        }
        let buffer = |label, bytes: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera articulated sleep freeze"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("gpu_articulated_sleep_freeze.wgsl").into(),
            ),
        });
        Ok(Self {
            device: device.clone(),
            count,
            pipeline: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Tessera articulated sleep freeze"),
                layout: None,
                module: &shader,
                entry_point: Some("freeze"),
                compilation_options: Default::default(),
                cache: None,
            }),
            buffers: [
                buffer(
                    "Tessera sleep coordinate owners",
                    bytemuck::cast_slice(coordinates),
                ),
                buffer(
                    "Tessera sleep link owners",
                    bytemuck::cast_slice(owner_data),
                ),
                inputs[0].clone(),
                inputs[1].clone(),
                inputs[2].clone(),
                inputs[3].clone(),
                inputs[4].clone(),
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Tessera frozen coordinate flags"),
                    size: u64::from(count) * 4,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
            ],
        })
    }

    /// Encode freezing; no CPU readback or intermediate submission is required.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera sleep freeze bindings"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &self
                .buffers
                .iter()
                .enumerate()
                .map(|(i, buffer)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Tessera articulated sleep freeze"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(self.count.div_ceil(64), 1, 1);
    }

    /// Clear stopped-coordinate diagnostics after a reset or topology change.
    /// This retains ownership and leaves positions, velocities and accelerations untouched.
    pub fn encode_clear_frozen(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.buffers[7], 0, None);
    }

    /// Device-resident flags in generalized-coordinate order, from the last pass.
    pub fn frozen_buffer(&self) -> &wgpu::Buffer {
        &self.buffers[7]
    }
}

fn pack_owners(
    states: &[Range<usize>],
    links: &[Range<usize>],
    owners: &[Vec<Vec<usize>>],
) -> Result<(Vec<[u32; 4]>, Vec<u32>), GpuSleepFreezeError> {
    if states.is_empty() || states.len() != links.len() || states.len() != owners.len() {
        return Err(GpuSleepFreezeError::InvalidInput);
    }
    let u32_index = |value| u32::try_from(value).map_err(|_| GpuSleepFreezeError::Capacity);
    let mut coordinates = Vec::new();
    let mut packed = Vec::new();
    let mut next_link = 0;
    for (environment, ((state, link), owner)) in states.iter().zip(links).zip(owners).enumerate() {
        if state.start != coordinates.len()
            || state.is_empty()
            || link.start != next_link
            || link.is_empty()
            || owner.len() != state.len()
        {
            return Err(GpuSleepFreezeError::InvalidInput);
        }
        next_link = link.end;
        for (coordinate, owner) in state.clone().zip(owner) {
            let start = packed.len();
            for (i, &local) in owner.iter().enumerate() {
                if local >= link.len() || owner[..i].contains(&local) {
                    return Err(GpuSleepFreezeError::InvalidInput);
                }
                packed.push(u32_index(link.start + local)?);
            }
            coordinates.push([
                u32_index(coordinate)?,
                u32_index(environment)?,
                u32_index(start)?,
                u32_index(owner.len())?,
            ]);
        }
    }
    Ok((coordinates, packed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_contact_pipeline::GpuContactDevice;

    #[test]
    fn freeze_owners_validate_local_indices_duplicates_and_offsets() {
        let states = [0..2, 2..4];
        let links = [0..3, 3..4];
        let owners = vec![vec![vec![0, 1], vec![1, 2]], vec![vec![0], vec![]]];
        let (coordinates, packed) = pack_owners(&states, &links, &owners).unwrap();
        assert_eq!(
            coordinates,
            [[0, 0, 0, 2], [1, 0, 2, 2], [2, 1, 4, 1], [3, 1, 5, 0]]
        );
        assert_eq!(packed, [0, 1, 1, 2, 3]);
        let mut invalid = owners.clone();
        invalid[1][0] = vec![1];
        assert!(pack_owners(&states, &links, &invalid).is_err());
        invalid[1][0] = vec![0, 0];
        assert!(pack_owners(&states, &links, &invalid).is_err());
        assert!(pack_owners(&states, &[0..3, 2..4], &owners).is_err());
        assert!(pack_owners(&[0..2, 1..3], &links, &owners).is_err());
        assert!(pack_owners(&states, &links, &owners[..1]).is_err());
    }

    #[test]
    fn freeze_stops_velocity_and_acceleration_only_for_all_sleeping_owners() {
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            let device = context.device();
            let queue = context.queue();
            let buffer = |label, bytes: &[u8]| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytes,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                })
            };
            let sleeping = buffer("freeze test sleep", bytemuck::cast_slice(&[1u32, 1, 0, 1]));
            let velocities = buffer(
                "freeze test velocity",
                bytemuck::cast_slice(&[1.0f32, 2.0, 3.0, 4.0]),
            );
            let accelerations = buffer(
                "freeze test acceleration",
                bytemuck::cast_slice(&[10.0f32, 20.0, 30.0, 40.0]),
            );
            let mass_status = buffer("freeze test mass status", bytemuck::cast_slice(&[0u32; 2]));
            let status = buffer("freeze test status", bytemuck::cast_slice(&[0u32; 2]));
            let coordinates = [[0, 0, 0, 2], [1, 0, 2, 2], [2, 1, 4, 1], [3, 1, 5, 0]];
            let pass = GpuArticulatedSleepFreezeBatch::build(
                device,
                &coordinates,
                &[0, 1, 1, 2, 3],
                [
                    &sleeping,
                    &velocities,
                    &accelerations,
                    &mass_status,
                    &status,
                ],
            )
            .unwrap();
            let step = || {
                let mut encoder = device.create_command_encoder(&Default::default());
                pass.encode(&mut encoder);
                let _ = queue.submit(Some(encoder.finish()));
            };
            let read = |buffer| {
                crate::gpu_articulated_mass::read_buffer(device, queue, buffer)
                    .unwrap()
                    .chunks_exact(4)
                    .map(|v| u32::from_ne_bytes(v.try_into().unwrap()))
                    .collect::<Vec<_>>()
            };
            let read_float = |buffer| {
                read(buffer)
                    .into_iter()
                    .map(f32::from_bits)
                    .collect::<Vec<_>>()
            };
            step();
            assert_eq!(read_float(&velocities), [0.0, 2.0, 0.0, 4.0]);
            assert_eq!(read_float(&accelerations), [0.0, 20.0, 0.0, 40.0]);
            assert_eq!(read(pass.frozen_buffer()), [1, 0, 1, 0]);
            let mut encoder = device.create_command_encoder(&Default::default());
            pass.encode_clear_frozen(&mut encoder);
            let _ = queue.submit(Some(encoder.finish()));
            assert_eq!(read(pass.frozen_buffer()), [0; 4]);
            assert_eq!(read_float(&velocities), [0.0, 2.0, 0.0, 4.0]);
            assert_eq!(read_float(&accelerations), [0.0, 20.0, 0.0, 40.0]);
            step();
            assert_eq!(read(pass.frozen_buffer()), [1, 0, 1, 0]);
            queue.write_buffer(
                &velocities,
                0,
                bytemuck::cast_slice(&[1.0f32, 2.0, 3.0, 4.0]),
            );
            queue.write_buffer(
                &accelerations,
                0,
                bytemuck::cast_slice(&[10.0f32, 20.0, 30.0, 40.0]),
            );
            queue.write_buffer(&sleeping, 0, bytemuck::cast_slice(&[0u32; 4]));
            step();
            assert_eq!(read_float(&velocities), [1.0, 2.0, 3.0, 4.0]);
            assert_eq!(read_float(&accelerations), [10.0, 20.0, 30.0, 40.0]);
            assert_eq!(read(pass.frozen_buffer()), [0; 4]);
            queue.write_buffer(&sleeping, 0, bytemuck::cast_slice(&[1u32; 4]));
            queue.write_buffer(&mass_status, 0, bytemuck::cast_slice(&[1u32, 0]));
            queue.write_buffer(&status, 0, bytemuck::cast_slice(&[0u32, 8]));
            step();
            assert_eq!(read_float(&velocities), [1.0, 2.0, 3.0, 4.0]);
            assert_eq!(read_float(&accelerations), [10.0, 20.0, 30.0, 40.0]);
            assert_eq!(read(pass.frozen_buffer()), [0; 4]);
            queue.write_buffer(&mass_status, 0, bytemuck::cast_slice(&[0u32; 2]));
            queue.write_buffer(&status, 0, bytemuck::cast_slice(&[0u32; 2]));
            queue.write_buffer(&sleeping, 0, bytemuck::cast_slice(&[1u32, 2, 1, 1]));
            step();
            assert_eq!(read(&status), [1, 0]);
            assert_eq!(read_float(&velocities), [1.0, 2.0, 0.0, 4.0]);
            assert_eq!(read(pass.frozen_buffer()), [0, 0, 1, 0]);
            queue.write_buffer(&status, 0, bytemuck::cast_slice(&[0u32; 2]));
            queue.write_buffer(&sleeping, 0, bytemuck::cast_slice(&[1u32; 4]));
            queue.write_buffer(
                &velocities,
                0,
                bytemuck::cast_slice(&[f32::NAN, 2.0, 3.0, 4.0]),
            );
            step();
            assert_eq!(read(&status), [1, 0]);
            assert!(read_float(&velocities)[0].is_nan());
            assert_eq!(read(pass.frozen_buffer())[0], 0);
            queue.write_buffer(&status, 0, bytemuck::cast_slice(&[0u32; 2]));
            queue.write_buffer(
                &velocities,
                0,
                bytemuck::cast_slice(&[1.0f32, 2.0, 3.0, 4.0]),
            );
            queue.write_buffer(
                &accelerations,
                0,
                bytemuck::cast_slice(&[f32::INFINITY, 20.0, 30.0, 40.0]),
            );
            step();
            assert_eq!(read(&status), [1, 0]);
            assert!(read_float(&accelerations)[0].is_infinite());
            assert_eq!(read(pass.frozen_buffer())[0], 0);
            eprintln!("sleep coordinate freeze passed on {backend:?}");
        }
        assert!(tested > 0, "no GPU backend available");
    }
}
