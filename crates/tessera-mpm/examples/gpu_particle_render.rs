//! Offscreen MPM rendering from a GPU snapshot, without particle readback.
use core::{error::Error, time::Duration};
use nalgebra::Vector3;
use std::{path::PathBuf, sync::mpsc};
use tessera_mpm::{
    GpuMpmParticleSnapshot, GpuMpmResidentSession, GpuMpmTransfers, MaterialModel, MpmParams,
    MpmParticle, MpmWorld, WorldBounds,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let directory = PathBuf::from(
        args.next()
            .ok_or("usage: gpu_particle_render <output-directory> [vulkan|dx12]")?,
    );
    let backend = match args.next().as_deref().and_then(|s| s.to_str()) {
        None | Some("vulkan") => wgpu::Backends::VULKAN,
        Some("dx12") => wgpu::Backends::DX12,
        _ => return Err("backend must be vulkan or dx12".into()),
    };
    if args.next().is_some() {
        return Err("unexpected argument".into());
    }
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: backend,
        ..Default::default()
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await?;
    println!("Render adapter: {:?}", adapter.get_info());
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await?;
    let mut particles = Vec::new();
    for x in 0..3 {
        for y in 0..3 {
            for z in 0..3 {
                let mut particle = MpmParticle::new(
                    Vector3::new(
                        -0.35 + f64::from(x) * 0.08,
                        -0.1 + f64::from(y) * 0.08,
                        0.3 + f64::from(z) * 0.08,
                    ),
                    0.02,
                    1_000.0,
                    MaterialModel::fluid(2_000.0, 7.0, 0.1),
                );
                particle.velocity.x = 0.8;
                particles.push(particle);
            }
        }
    }
    let world = MpmWorld::new(
        particles,
        MpmParams {
            gravity: Vector3::new(0.0, 0.0, -9.81),
            cell_width: 0.1,
            bounds: Some(WorldBounds {
                min: Vector3::repeat(-1.0),
                max: Vector3::repeat(1.0),
            }),
            ..MpmParams::default()
        },
    )?;
    let transfers = GpuMpmTransfers::new(&device);
    let mut session = GpuMpmResidentSession::new(&transfers, &device, &queue, world, 0.001)?;
    let snapshot = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("MPM render snapshots"),
        size: 27 * size_of::<GpuMpmParticleSnapshot>() as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("MPM particle renderer"),
        source: wgpu::ShaderSource::Wgsl(include_str!("gpu_particle_render.wgsl").into()),
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("MPM particle pipeline"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vertex"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fragment"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });
    let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("MPM render bindings"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: snapshot.as_entire_binding(),
        }],
    });
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("MPM offscreen frame"),
        size: wgpu::Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("frame readback"),
        size: 256 * 256 * 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    std::fs::create_dir_all(&directory)?;
    let mut frames = Vec::new();
    for frame in 0..2 {
        if frame == 1 {
            session.submit_steps(64)?;
            session.submit_steps(64)?;
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let count = session.encode_particle_snapshot(&mut encoder, &snapshot)?;
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("MPM snapshot render"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bindings, &[]);
            pass.draw(0..6, 0..count);
        }
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(1024),
                    rows_per_image: Some(256),
                },
            },
            texture.size(),
        );
        let _submission = queue.submit(Some(encoder.finish()));
        // Rendering was submitted before any particle readback or CPU wait.
        session.synchronize_async().await?;
        let (sender, receiver) = mpsc::channel();
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        let _status = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(10)),
        })?;
        receiver.recv_timeout(Duration::from_secs(10))??;
        let data = readback.slice(..).get_mapped_range();
        let rgb = data
            .chunks_exact(4)
            .flat_map(|p| p[..3].iter().copied())
            .collect::<Vec<_>>();
        drop(data);
        readback.unmap();
        if rgb
            .chunks_exact(3)
            .filter(|p| p.iter().any(|&v| v != 0))
            .count()
            < 64
        {
            return Err("rendered frame is empty".into());
        }
        let mut ppm = b"P6\n256 256\n255\n".to_vec();
        ppm.extend_from_slice(&rgb);
        std::fs::write(directory.join(format!("frame-{frame}.ppm")), ppm)?;
        frames.push(rgb);
    }
    let changed = frames[0]
        .chunks_exact(3)
        .zip(frames[1].chunks_exact(3))
        .filter(|(a, b)| a != b)
        .count();
    if changed < 64 {
        return Err("MPM render did not change after simulation".into());
    }
    println!(
        "GPU snapshot render verified: {changed} changed pixels; output {}",
        directory.display()
    );
    Ok(())
}
