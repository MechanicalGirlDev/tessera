//! Browser WebGPU MPM simulation with asynchronous synchronization and GPU snapshot rendering.
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    eprintln!("Build for wasm32-unknown-unknown; see examples/web_gpu_particles/README.md");
}
#[cfg(target_arch = "wasm32")]
fn main() {}
#[cfg(target_arch = "wasm32")]
mod browser {
    use core::error::Error;
    use nalgebra::Vector3;
    use tessera_mpm::{
        GpuMpmParticleSnapshot, GpuMpmResidentSession, GpuMpmTransfers, MaterialModel, MpmParams,
        MpmParticle, MpmWorld, WorldBounds,
    };
    use wasm_bindgen::{JsCast, prelude::*};
    struct Demo {
        device: wgpu::Device,
        queue: wgpu::Queue,
        surface: wgpu::Surface<'static>,
        pipeline: wgpu::RenderPipeline,
        bindings: wgpu::BindGroup,
        snapshot: wgpu::Buffer,
        session: GpuMpmResidentSession<'static>,
        frames: u32,
    }
    fn js_error(error: impl core::fmt::Display) -> JsValue {
        JsValue::from_str(&error.to_string())
    }
    fn report(error: JsValue) {
        web_sys::console::error_1(&error);
        if let Some(element) = web_sys::window()
            .and_then(|w| w.document())
            .and_then(|d| d.get_element_by_id("status"))
        {
            element.set_text_content(Some(&format!("Error: {error:?}")));
        }
    }
    impl Demo {
        async fn new() -> Result<Self, JsValue> {
            let document = web_sys::window()
                .and_then(|w| w.document())
                .ok_or_else(|| js_error("document unavailable"))?;
            let canvas = document
                .get_element_by_id("scene")
                .ok_or_else(|| js_error("canvas unavailable"))?
                .dyn_into::<web_sys::HtmlCanvasElement>()?;
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: wgpu::Backends::BROWSER_WEBGPU,
                ..Default::default()
            });
            let surface = instance
                .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
                .map_err(js_error)?;
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    compatible_surface: Some(&surface),
                    ..Default::default()
                })
                .await
                .map_err(js_error)?;
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor::default())
                .await
                .map_err(js_error)?;
            let surface_config = surface
                .get_default_config(&adapter, canvas.width(), canvas.height())
                .ok_or_else(|| js_error("surface unsupported"))?;
            surface.configure(&device, &surface_config);
            let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
            let resources = Self::resources(&device, &queue, &surface_config);
            if let Some(error) = scope.pop().await {
                return Err(js_error(error));
            }
            let (session, snapshot, pipeline, bindings) = resources.map_err(js_error)?;
            Ok(Self {
                device,
                queue,
                surface,
                pipeline,
                bindings,
                snapshot,
                session,
                frames: 0,
            })
        }
        fn resources(
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            surface_config: &wgpu::SurfaceConfiguration,
        ) -> Result<
            (
                GpuMpmResidentSession<'static>,
                wgpu::Buffer,
                wgpu::RenderPipeline,
                wgpu::BindGroup,
            ),
            Box<dyn Error>,
        > {
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
            let transfers = GpuMpmTransfers::new(device);
            let session =
                GpuMpmResidentSession::new(&transfers, device, queue, world, 0.001)?.into_owned();
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
                        format: surface_config.format,
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

            Ok((session, snapshot, pipeline, bindings))
        }
        async fn frame(&mut self) -> Result<(), JsValue> {
            let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
            self.session.submit_steps(8).map_err(js_error)?;
            let texture = self.surface.get_current_texture().map_err(js_error)?;
            let view = texture
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            let count = self
                .session
                .encode_particle_snapshot(&mut encoder, &self.snapshot)
                .map_err(js_error)?;
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("WebGPU MPM particles"),
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
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.bindings, &[]);
                pass.draw(0..6, 0..count);
            }
            let _submission = self.queue.submit(Some(encoder.finish()));
            texture.present();
            if let Some(error) = scope.pop().await {
                return Err(js_error(error));
            }
            self.session.synchronize_async().await.map_err(js_error)?;
            self.frames += 1;
            if let Some(element) = web_sys::window()
                .and_then(|w| w.document())
                .and_then(|d| d.get_element_by_id("status"))
            {
                let p = &self.session.world().particles[0];
                element.set_text_content(Some(&format!(
                    "frame={} substeps={} x={:.4} z={:.4}",
                    self.frames,
                    self.session.world().substeps,
                    p.position.x,
                    p.position.z
                )));
            }
            Ok(())
        }
    }
    fn schedule(mut demo: Demo) -> Result<(), JsValue> {
        let window = web_sys::window().ok_or_else(|| js_error("window unavailable"))?;
        let callback = Closure::once_into_js(move |_: f64| {
            wasm_bindgen_futures::spawn_local(async move {
                match demo.frame().await {
                    Ok(()) => {
                        if let Err(error) = schedule(demo) {
                            report(error);
                        }
                    }
                    Err(error) => report(error),
                }
            });
        });
        let _request = window.request_animation_frame(callback.unchecked_ref())?;
        Ok(())
    }
    #[wasm_bindgen(start)]
    pub fn start() {
        wasm_bindgen_futures::spawn_local(async {
            match Demo::new().await {
                Ok(demo) => {
                    if let Err(error) = schedule(demo) {
                        report(error);
                    }
                }
                Err(error) => report(error),
            }
        });
    }
}
