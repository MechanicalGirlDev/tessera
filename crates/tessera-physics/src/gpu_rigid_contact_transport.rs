//! GPU local-anchor capture and pose-based contact refresh for temporal solvers.
//!
//! This is a building block for TGS, not a complete temporal solver. The initial
//! normal stays fixed for the frame. Signed depth may become negative (separation).

use crate::gpu_rigid_sphere_contact::GpuRigidSphereContacts;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    counts: [u32; 4],
    totals: [u32; 4],
}

/// Contact-anchor storage or dispatch would exceed device limits.
#[derive(Debug, thiserror::Error)]
#[error("temporal contact anchor capacity exceeds GPU limits")]
pub struct GpuRigidContactTransportError;

/// Contact points captured in body-local frames and refreshed from live GPU poses.
///
/// Bind this object to a fixed contact topology. Recreate it when body buffers,
/// pairs, or manifold strides change. Encode capture after narrow phase, then
/// refresh after pose integration, without regenerating the original manifold.
/// Inactive captured slots stay inactive until another capture. Active slots
/// retain signed depth, including separation, for a later speculative solver.
/// This operation does not apply impulses or discover new collisions.
#[derive(Debug)]
pub struct GpuRigidContactTransport {
    capture: wgpu::ComputePipeline,
    refresh: wgpu::ComputePipeline,
    inputs: wgpu::BindGroup,
    groups: u32,
}
impl GpuRigidContactTransport {
    /// Allocate anchor storage and bind a resident contact set on the same device.
    pub fn new(
        device: &wgpu::Device,
        contacts: &GpuRigidSphereContacts,
    ) -> Result<Self, GpuRigidContactTransportError> {
        let pair_count =
            u32::try_from(contacts.pairs().len()).map_err(|_| GpuRigidContactTransportError)?;
        let body_count =
            u32::try_from(contacts.body_count()).map_err(|_| GpuRigidContactTransportError)?;
        let pair_stride = contacts.pair_contact_stride();
        let ground_stride = contacts.ground_contact_stride();
        let pair_slots = pair_count
            .checked_mul(pair_stride)
            .ok_or(GpuRigidContactTransportError)?;
        let ground_slots = if contacts.ground_contact_buffer().is_some() {
            body_count
                .checked_mul(ground_stride)
                .ok_or(GpuRigidContactTransportError)?
        } else {
            0
        };
        let total = pair_slots
            .checked_add(ground_slots)
            .ok_or(GpuRigidContactTransportError)?;
        Self::new_bound(
            device,
            [
                contacts.state_buffer(),
                contacts.pair_buffer(),
                contacts.pair_contact_buffer(),
                contacts.ground_buffer_raw(),
            ],
            Params {
                counts: [pair_count, body_count, pair_stride, ground_stride],
                totals: [total, 0, 0, 0],
            },
        )
    }

    fn new_bound(
        device: &wgpu::Device,
        bound: [&wgpu::Buffer; 4],
        values: Params,
    ) -> Result<Self, GpuRigidContactTransportError> {
        let total = values.totals[0];
        let groups = total.div_ceil(64);
        let bytes = u64::from(total.max(1)) * 48;
        let limits = device.limits();
        if groups > limits.max_compute_workgroups_per_dimension
            || bytes > limits.max_buffer_size
            || bytes > u64::from(limits.max_storage_buffer_binding_size)
        {
            return Err(GpuRigidContactTransportError);
        }
        let anchors = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tessera temporal contact anchors"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tessera temporal contact counts"),
            contents: bytemuck::bytes_of(&values),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let entries: Vec<_> = (0..6)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 5 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: binding < 2,
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Tessera temporal contact layout"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Tessera temporal contact pipeline layout"),
            bind_group_layouts: &[&layout],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tessera temporal contact transport"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("gpu_rigid_contact_transport.wgsl").into(),
            ),
        });
        let pipeline = |entry_point| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Tessera temporal contact pass"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let buffers = [bound[0], bound[1], bound[2], bound[3], &anchors, &params];
        let bindings: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let inputs = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tessera temporal contact inputs"),
            layout: &layout,
            entries: &bindings,
        });
        Ok(Self {
            capture: pipeline("capture"),
            refresh: pipeline("refresh"),
            inputs,
            groups,
        })
    }
    /// Capture the current narrow-phase points and their initial signed depth.
    pub fn encode_capture(&self, encoder: &mut wgpu::CommandEncoder) {
        self.encode(encoder, &self.capture);
    }
    /// Refresh points and signed depth from captured anchors and current poses.
    ///
    /// The normal stays fixed; positive depth means penetration, negative depth
    /// means separation. This preserves active rows for a speculative solve.
    pub fn encode_refresh(&self, encoder: &mut wgpu::CommandEncoder) {
        self.encode(encoder, &self.refresh);
    }
    fn encode(&self, encoder: &mut wgpu::CommandEncoder, pipeline: &wgpu::ComputePipeline) {
        if self.groups == 0 {
            return;
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Tessera temporal contact anchor pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.inputs, &[]);
        pass.dispatch_workgroups(self.groups, 1, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_broad_phase::GpuPair;
    use crate::gpu_contact_pipeline::GpuContactDevice;
    use crate::gpu_rigid_state::{GpuRigidBodyState, GpuRigidStateSession};
    use crate::gpu_sphere_contact::GpuSphereContact;

    #[test]
    fn manifold_slots_follow_both_pair_and_ground_body_poses() {
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            eprintln!("anchor manifold backend: {backend:?}");
            let states: Vec<_> = (0..3)
                .map(|i| GpuRigidBodyState {
                    position_inverse_mass: [i as f32 * 2.0, 0.0, 1.0, 0.0],
                    orientation: [0.0, 0.0, 0.0, 1.0],
                    linear_velocity: [0.0; 4],
                    angular_velocity: [0.0; 4],
                    inverse_inertia_sleep: [0.0; 4],
                })
                .collect();
            let state =
                GpuRigidStateSession::new(context.device(), context.queue(), &states).unwrap();
            let pair_owners = [0usize, 1, 0, 0, 0, 1, 1, 1];
            let ground_owners = [0usize, 1, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2];
            let pair_points: Vec<_> = pair_owners
                .iter()
                .enumerate()
                .map(|(i, owner)| GpuSphereContact {
                    point: [*owner as f32 * 2.0 + 1.0, i as f32 * 0.1, 1.0, 0.0],
                    normal: [1.0, 0.0, 0.0, 0.0],
                    depth_hit: [0.5, if i == 3 || i == 7 { 0.0 } else { 1.0 }, 0.0, 0.0],
                })
                .collect();
            let ground_points: Vec<_> = ground_owners
                .iter()
                .enumerate()
                .map(|(i, owner)| GpuSphereContact {
                    point: [*owner as f32 * 2.0, i as f32 * 0.1, 0.0, 0.0],
                    normal: [0.0, 0.0, 1.0, 0.0],
                    depth_hit: [0.0, if i == 5 || i == 10 { 0.0 } else { 1.0 }, 0.0, 0.0],
                })
                .collect();
            let buffer = |bytes: &[u8]| {
                context
                    .device()
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Tessera anchor test fixture"),
                        contents: bytes,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    })
            };
            let pairs = buffer(bytemuck::cast_slice(&[
                GpuPair { a: 0, b: 1 },
                GpuPair { a: 1, b: 2 },
            ]));
            let pair_contacts = buffer(bytemuck::cast_slice(&pair_points));
            let ground_contacts = buffer(bytemuck::cast_slice(&ground_points));
            let transport = GpuRigidContactTransport::new_bound(
                context.device(),
                [
                    state.state_buffer(),
                    &pairs,
                    &pair_contacts,
                    &ground_contacts,
                ],
                Params {
                    counts: [2, 3, 4, 4],
                    totals: [20, 0, 0, 0],
                },
            )
            .unwrap();
            let mut capture = context.device().create_command_encoder(&Default::default());
            transport.encode_capture(&mut capture);
            let _ = context.queue().submit(Some(capture.finish()));
            for (index, original) in states.iter().enumerate() {
                let mut moved = *original;
                let delta = 0.1 * (index + 1) as f32;
                moved.position_inverse_mass[0] += delta;
                moved.position_inverse_mass[2] += delta;
                state.write_body(context.queue(), index, moved).unwrap();
            }
            let staging = context.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("Tessera anchor fixture readback"),
                size: 20 * 48,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut refresh = context.device().create_command_encoder(&Default::default());
            transport.encode_refresh(&mut refresh);
            refresh.copy_buffer_to_buffer(&pair_contacts, 0, &staging, 0, 8 * 48);
            refresh.copy_buffer_to_buffer(&ground_contacts, 0, &staging, 8 * 48, 12 * 48);
            let _ = context.queue().submit(Some(refresh.finish()));
            let (sender, receiver) = std::sync::mpsc::channel();
            staging
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    let _ = sender.send(result);
                });
            let _status = context
                .device()
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(core::time::Duration::from_secs(5)),
                })
                .unwrap();
            receiver
                .recv_timeout(core::time::Duration::from_secs(5))
                .unwrap()
                .unwrap();
            let mapped = staging.slice(..).get_mapped_range();
            let actual: Vec<GpuSphereContact> = mapped
                .chunks_exact(48)
                .map(bytemuck::pod_read_unaligned)
                .collect();
            for (index, original) in pair_points.iter().chain(&ground_points).enumerate() {
                let point = actual[index];
                assert_eq!(point.is_contact(), original.is_contact());
                if !original.is_contact() {
                    continue;
                }
                let (shift, depth) = if index < 8 {
                    (0.15 + pair_owners[index] as f32 * 0.1, 0.4)
                } else {
                    let delta = 0.1 * (ground_owners[index - 8] + 1) as f32;
                    (delta * 0.5, -delta)
                };
                assert!(
                    (point.point[0] - original.point[0] - shift).abs() < 1e-5,
                    "{index}: {point:?}"
                );
                assert!((point.point[2] - original.point[2] - shift).abs() < 1e-5);
                assert!((point.depth_hit[0] - depth).abs() < 1e-5);
                assert_eq!(point.normal, original.normal);
            }
            drop(mapped);
            staging.unmap();
        }
        assert!(tested > 0, "no GPU backend available");
    }
}
