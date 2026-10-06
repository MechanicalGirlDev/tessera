//! Browser WebGPU demo that renders Tessera's GPU-resident rigid states directly.

#[cfg(target_arch = "wasm32")]
extern crate alloc;

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    eprintln!(
        "Build this example for wasm32-unknown-unknown and open examples/web_gpu_spheres.html"
    );
}

#[cfg(target_arch = "wasm32")]
mod browser {
    use alloc::rc::Rc;
    use core::cell::RefCell;

    use tessera_physics::gpu_rigid_sphere_world::{GpuRigidSphereWorld, GpuRigidSphereWorldConfig};
    use tessera_physics::gpu_rigid_state::GpuRigidBodyState;
    use wasm_bindgen::JsCast;
    use wasm_bindgen::prelude::*;

    struct Demo {
        device: wgpu::Device,
        queue: wgpu::Queue,
        surface: wgpu::Surface<'static>,
        surface_config: wgpu::SurfaceConfiguration,
        pipeline: wgpu::RenderPipeline,
        state_bind_group: wgpu::BindGroup,
        world: GpuRigidSphereWorld,
        initial_states: [GpuRigidBodyState; 3],
        frames: u32,
    }

    fn sphere(x: f32, z: f32, velocity_x: f32) -> GpuRigidBodyState {
        GpuRigidBodyState {
            position_inverse_mass: [x, 0.0, z, 1.0],
            orientation: [0.0, 0.0, 0.0, 1.0],
            linear_velocity: [velocity_x, 0.0, 0.0, 0.0],
            angular_velocity: [0.0; 4],
            inverse_inertia_sleep: [1.0, 1.0, 1.0, 0.0],
        }
    }

    fn js_error(error: impl core::fmt::Display) -> JsValue {
        JsValue::from_str(&error.to_string())
    }

    impl Demo {
        async fn new() -> Result<Self, JsValue> {
            let window = web_sys::window().ok_or_else(|| js_error("window unavailable"))?;
            let document = window
                .document()
                .ok_or_else(|| js_error("document unavailable"))?;
            let canvas = document
                .get_element_by_id("scene")
                .ok_or_else(|| js_error("canvas #scene unavailable"))?
                .dyn_into::<web_sys::HtmlCanvasElement>()?;
            let instance = wgpu::Instance::default();
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
                .ok_or_else(|| js_error("WebGPU surface unsupported"))?;
            surface.configure(&device, &surface_config);

            let initial_states = [
                sphere(-2.0, 2.7, 1.4),
                sphere(2.0, 2.7, -1.4),
                sphere(0.0, 5.0, 0.0),
            ];
            let world = GpuRigidSphereWorld::new(
                &device,
                &queue,
                &initial_states,
                &[0.65; 3],
                GpuRigidSphereWorldConfig {
                    gravity: [0.0, 0.0, -9.81],
                    ground_half_extent: Some(20.0),
                    ..Default::default()
                },
            )
            .map_err(js_error)?;
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Tessera browser sphere renderer"),
                source: wgpu::ShaderSource::Wgsl(include_str!("web_gpu_spheres.wgsl").into()),
            });
            let state_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Tessera browser rigid states"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
            let state_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Tessera browser rigid state binding"),
                layout: &state_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: world.state_buffer().as_entire_binding(),
                }],
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Tessera browser renderer layout"),
                bind_group_layouts: &[&state_layout],
                immediate_size: 0,
            });
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Tessera browser spheres"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vertex_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fragment_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surface_config.format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });
            Ok(Self {
                device,
                queue,
                surface,
                surface_config,
                pipeline,
                state_bind_group,
                world,
                initial_states,
                frames: 0,
            })
        }

        fn frame(&mut self) -> Result<(), JsValue> {
            if self.frames == 360 {
                self.world.reset(&self.initial_states).map_err(js_error)?;
                self.frames = 0;
            }
            let _candidates = self.world.step_substeps(1.0 / 120.0, 2).map_err(js_error)?;
            self.frames += 1;
            let texture = match self.surface.get_current_texture() {
                Ok(texture) => texture,
                Err(_) => {
                    self.surface.configure(&self.device, &self.surface_config);
                    self.surface.get_current_texture().map_err(js_error)?
                }
            };
            let view = texture
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Tessera browser frame"),
                });
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Tessera browser spheres"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.025,
                                g: 0.04,
                                b: 0.08,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.state_bind_group, &[]);
                pass.draw(0..6, 3..4);
                pass.draw(0..6, 0..self.world.len() as u32);
            }
            let _submission = self.queue.submit(Some(encoder.finish()));
            texture.present();
            Ok(())
        }
    }

    fn schedule(demo: Rc<RefCell<Demo>>) -> Result<(), JsValue> {
        let window = web_sys::window().ok_or_else(|| js_error("window unavailable"))?;
        let callback = Closure::once_into_js(move |_: f64| {
            if let Err(error) = demo
                .borrow_mut()
                .frame()
                .and_then(|()| schedule(demo.clone()))
            {
                web_sys::console::error_1(&error);
            }
        });
        let _request = window.request_animation_frame(callback.unchecked_ref())?;
        Ok(())
    }

    #[wasm_bindgen(start)]
    pub fn start() {
        wasm_bindgen_futures::spawn_local(async {
            match Demo::new().await {
                Ok(demo) => {
                    if let Err(error) = schedule(Rc::new(RefCell::new(demo))) {
                        web_sys::console::error_1(&error);
                    }
                }
                Err(error) => web_sys::console::error_1(&error),
            }
        });
    }
}

#[cfg(target_arch = "wasm32")]
fn main() {}
