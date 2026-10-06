# WebGPU mixed shapes and joints demo

This demo runs sphere, box, capsule, and finite-ground contacts plus a ball joint on WebGPU in the browser.
The render shader reads rigid body positions and orientations directly from the GPU state buffer. The shapes
on screen are a simple projection, not mesh renderings of the physics shapes themselves.

Run these commands from the repository root.

```powershell
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.129 --locked
cargo build -p tessera-physics --features gpu-contact --example web_gpu_mixed --target wasm32-unknown-unknown --release
wasm-bindgen --target web --out-dir crates/tessera-physics/examples/web_gpu_mixed/pkg target/wasm32-unknown-unknown/release/examples/web_gpu_mixed.wasm
python -m http.server 8765 --directory crates/tessera-physics/examples/web_gpu_mixed
```

Open `http://localhost:8765/` in a WebGPU-capable browser. This example uses the small-scale all-pairs GPU path.
Large-scale LBVH candidates currently go through a CPU readback.
