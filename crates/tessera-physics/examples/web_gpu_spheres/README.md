# Tessera WebGPU 3D example

This example solves sphere collisions and ground contact with WebGPU compute in the browser, then draws
by reading the same GPU state buffer from the vertex shader. Rigid body state isn't read back to the CPU during a frame.

Run these commands from the repository root.

```powershell
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.129 --locked
cargo build -p tessera-physics --features gpu-contact --example web_gpu_spheres --target wasm32-unknown-unknown --release
wasm-bindgen --target web --out-dir crates/tessera-physics/examples/web_gpu_spheres/pkg target/wasm32-unknown-unknown/release/examples/web_gpu_spheres.wasm
python -m http.server 8000 --directory crates/tessera-physics/examples/web_gpu_spheres
```

Open `http://localhost:8000` in a WebGPU-capable browser. `pkg/` is generated output and isn't committed to Git.
This example shows the GPU-resident path for a small sphere world. Updating LBVH candidates for large worlds
currently needs a CPU readback, so the browser doesn't use that path.
