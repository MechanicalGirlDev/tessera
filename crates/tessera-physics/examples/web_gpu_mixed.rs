//! Browser WebGPU demo of mixed rigid shapes and a ball joint.

#[cfg(target_arch = "wasm32")]
extern crate alloc;

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    eprintln!(
        "Build this example for wasm32-unknown-unknown and open examples/web_gpu_mixed/index.html"
    );
}

#[cfg(target_arch = "wasm32")]
mod browser {
    use alloc::rc::Rc;
    use core::cell::RefCell;

    use tessera_physics::gpu_rigid_ball_joint::GpuRigidBallJoint;
    use tessera_physics::gpu_rigid_shape::GpuRigidShape;
    use tessera_physics::gpu_rigid_sphere_world::{
        GpuRigidPrimitiveWorld, GpuRigidSphereWorldConfig,
    };
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
        world: GpuRigidPrimitiveWorld,
        initial_states: [GpuRigidBodyState; 4],
        frames: u32,
    }

    fn body(x: f32, z: f32, inverse_mass: f32) -> GpuRigidBodyState {
        GpuRigidBodyState {
            position_inverse_mass: [x, 0.0, z, inverse_mass],
            orientation: [0.0, 0.0, 0.0, 1.0],
            linear_velocity: [0.0; 4],
            angular_velocity: [0.0; 4],
            inverse_inertia_sleep: [inverse_mass, inverse_mass, inverse_mass, 0.0],
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
                body(-1.6, 4.8, 0.0),
                body(-0.2, 4.8, 1.0),
                body(1.4, 3.5, 1.0),
                body(2.2, 5.2, 1.0),
            ];
            let shapes = [
                GpuRigidShape::Sphere { radius: 0.18 },
                GpuRigidShape::Capsule {
                    radius: 0.3,
                    half_length: 0.7,
                },
                GpuRigidShape::Box {
                    half_extents: [0.55, 0.35, 0.45],
                },
                GpuRigidShape::Sphere { radius: 0.4 },
            ];
            let mut world = GpuRigidPrimitiveWorld::new_primitives(
                &device,
                &queue,
                &initial_states,
                &shapes,
                GpuRigidSphereWorldConfig {
                    gravity: [0.0, 0.0, -9.81],
                    ground_half_extent: Some(20.0),
                    ..Default::default()
                },
            )
            .map_err(js_error)?;
            world
                .set_ball_joints(&[GpuRigidBallJoint {
                    body_a: 0,
                    body_b: 1,
                    local_anchor_a: [0.0; 3],
                    local_anchor_b: [-1.4, 0.0, 0.0],
                }])
                .map_err(js_error)?;
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Tessera browser mixed renderer"),
                source: wgpu::ShaderSource::Wgsl(include_str!("web_gpu_mixed.wgsl").into()),
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
                label: Some("Tessera browser mixed bodies"),
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
            if self.frames == 240 {
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
                    label: Some("Tessera browser mixed scene"),
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
                pass.draw(0..6, 4..6);
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
