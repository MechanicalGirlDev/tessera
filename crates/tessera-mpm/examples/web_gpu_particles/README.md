# WebGPU MPM demo

This demo simulates 27 fluid particles on WebGPU, and the vertex shader reads the GPU snapshot directly.
Each frame submits 8 substeps. After the canvas draw is submitted,
`synchronize_async().await` checks for errors and syncs the CPU world.
Particles are drawn as circular billboards with a 3D oblique projection. There's no perspective or depth testing.

Run these commands from the repository root.

```powershell
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.129 --locked
cargo build -p tessera-mpm --features gpu-mpm --example web_gpu_particles --target wasm32-unknown-unknown --release
wasm-bindgen --target web --out-dir crates/tessera-mpm/examples/web_gpu_particles/pkg target/wasm32-unknown-unknown/release/examples/web_gpu_particles.wasm
python -m http.server 8769 --bind 127.0.0.1 --directory crates/tessera-mpm/examples/web_gpu_particles
```

Open `http://127.0.0.1:8769/` in a WebGPU-capable browser. The page shows the frame and substep counts
and the coordinates of the first particle. Reloading resets to the initial state. A GPU validation error
or a bad simulation result stops the demo and is reported in the status area and the console.

Verified on Chromium on Windows: simulation, rendering, an image change between two points in time, and zero warnings.
Other operating systems, browsers, GPU vendors, large-scale performance, and long-run stability haven't been verified.
