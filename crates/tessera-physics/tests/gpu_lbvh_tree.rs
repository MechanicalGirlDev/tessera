//! Independent resident LBVH trees without candidate-pair generation.
#![cfg(feature = "gpu-contact")]
use tessera_physics::{
    gpu_broad_phase::GpuAabb, gpu_contact_pipeline::GpuContactDevice, gpu_lbvh::GpuLbvh,
};
use wgpu::util::DeviceExt;
#[test]
fn tree_builds_keep_bounds_and_leaf_ids_independent() -> Result<(), Box<dyn core::error::Error>> {
    let mut tested = 0;
    for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
        let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
            continue;
        };
        tested += 1;
        let builder = GpuLbvh::new(context.device());
        let bounds = (0..67)
            .map(|i| {
                let x = ((i * 23) % 67) as f32;
                GpuAabb::new([x, -2.0, -1.0], [x + 0.5, 2.0, 1.0])
            })
            .collect::<Result<Vec<_>, _>>()?;
        let shifted = bounds
            .iter()
            .map(|b| GpuAabb {
                lower: [b.lower[0] + 100.0, -2.0, -1.0, 0.0],
                upper: [b.upper[0] + 100.0, 2.0, 1.0, 0.0],
            })
            .collect::<Vec<_>>();
        let upload = |data: &[GpuAabb]| {
            context
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(data),
                    usage: wgpu::BufferUsages::STORAGE,
                })
        };
        let mut encoder = context.device().create_command_encoder(&Default::default());
        let first =
            builder.encode_tree_resident(context.device(), &mut encoder, &upload(&bounds), 67)?;
        let second =
            builder.encode_tree_resident(context.device(), &mut encoder, &upload(&shifted), 67)?;
        let bytes = first.nodes.size();
        assert_eq!(bytes, 133 * 64);
        let staging = context.device().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes * 2,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&first.nodes, 0, &staging, 0, bytes);
        encoder.copy_buffer_to_buffer(&second.nodes, 0, &staging, bytes, bytes);
        let _ = context.queue().submit(Some(encoder.finish()));
        let (sender, receiver) = std::sync::mpsc::channel();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender.send(r);
        });
        let _ = context.device().poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(core::time::Duration::from_secs(30)),
        })?;
        receiver.recv_timeout(core::time::Duration::from_secs(30))??;
        let view = staging.slice(..).get_mapped_range();
        for (ordinal, source) in [&bounds, &shifted].into_iter().enumerate() {
            let data = &view[ordinal * bytes as usize..(ordinal + 1) * bytes as usize];
            let word = |node: usize, field: usize| {
                let start = node * 64 + field * 4;
                u32::from_le_bytes([
                    data[start],
                    data[start + 1],
                    data[start + 2],
                    data[start + 3],
                ])
            };
            assert_eq!(word(0, 10), u32::MAX);
            let mut leaves = [false; 67];
            for node in 66..133 {
                let body = word(node, 8) as usize;
                assert!(body < 67);
                assert!(!leaves[body]);
                leaves[body] = true;
                assert_eq!(f32::from_bits(word(node, 7)), 1.0);
                for axis in 0..3 {
                    assert_eq!(f32::from_bits(word(node, axis)), source[body].lower[axis]);
                    assert_eq!(
                        f32::from_bits(word(node, axis + 4)),
                        source[body].upper[axis]
                    );
                }
            }
            for node in 0..66 {
                let left = word(node, 8) as usize;
                let right = word(node, 9) as usize;
                assert!(left < 133 && right < 133 && left != right);
                assert_eq!(word(left, 10), node as u32);
                assert_eq!(word(right, 10), node as u32);
                for axis in 0..3 {
                    assert_eq!(
                        f32::from_bits(word(node, axis)),
                        f32::from_bits(word(left, axis)).min(f32::from_bits(word(right, axis)))
                    );
                    assert_eq!(
                        f32::from_bits(word(node, axis + 4)),
                        f32::from_bits(word(left, axis + 4))
                            .max(f32::from_bits(word(right, axis + 4)))
                    );
                }
            }
            assert!(leaves.iter().all(|seen| *seen));
            assert_eq!(f32::from_bits(word(0, 0)), ordinal as f32 * 100.0);
            assert_eq!(f32::from_bits(word(0, 4)), 66.5 + ordinal as f32 * 100.0);
        }
        drop(view);
        staging.unmap();
        let mut encoder = context.device().create_command_encoder(&Default::default());
        assert!(
            builder
                .encode_tree_resident(context.device(), &mut encoder, &upload(&bounds), 1)
                .is_err()
        );
    }
    assert!(tested > 0);
    Ok(())
}
