# Rendering MPM GPU snapshots

`gpu_particle_render` renders 27 fluid particles offscreen with a 3D oblique projection.
The vertex shader reads the GPU snapshot's 32-byte records as a STORAGE buffer
and draws each particle as a 6-vertex circular billboard. Color varies with velocity.

Run it from the repository root. Pass the output directory as an argument.

```powershell
cargo run -p tessera-mpm --features gpu-mpm --example gpu_particle_render -- target/mpm-render-vulkan vulkan
cargo run -p tessera-mpm --features gpu-mpm --example gpu_particle_render -- target/mpm-render-dx12 dx12
```

It writes `frame-0.ppm` for the initial state and `frame-1.ppm` after 128 substeps.
Files with the same names in the output directory are overwritten. The PPMs are P6, 256×256 pixels.
The run fails on an empty image, a change of fewer than 64 pixels, or a GPU error.

The snapshot compute pass and the draw are recorded in the same command encoder. Particles
aren't read back to the CPU for drawing. Saving the image reads the texture back.
The simulation submits `submit_steps(64)` asynchronously twice. After the draw submit,
`synchronize_async().await` checks for GPU errors and updates the CPU world.
The existing `session.step` is still available as a synchronous API.
This example doesn't implement an interactive viewer, a perspective camera, or depth testing.

For the browser version, see [web_gpu_particles](web_gpu_particles/README.md).
