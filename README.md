# Tessera

Vendor-neutral 3D rigid-body and Material Point Method physics.

## Crates

- `tessera-physics`: CPU rigid-body and articulated dynamics, URDF/MJCF loading,
  contact solvers, and optional WebGPU compute (`gpu-contact`).
- `tessera-mpm`: CPU and WebGPU Material Point Method simulation, including
  optional rigid-body coupling.
- `tessera-python3d`: UniFFI bindings and a platform-specific `tessera3d` Python wheel.

## Reiny 0.7 integration

All three crates remain independent of Reiny: `tessera-physics` owns rigid-body
state and solvers, `tessera-mpm` owns particle simulation and coupling, and
`tessera-python3d` exposes those APIs through UniFFI. Do not add deployment or
transport dependencies to the solver or Python-binding layers.

A Reiny host adapter owns `main.yaml`, compiled command/state message types,
named `Cloudy::input`/`output` ports and deployment/module provenance. Initialize
the world and any required GPU adapter before `Cloudy::ready()`, and propagate
initialization failures instead of reporting a usable simulation. On
`Cloudy::shutdown()`, finish the host's work and release its simulation/GPU
resources. Solver library calls and Python calls are not managed Reiny modules
by themselves.

The software-GPU opt-in below is for explicit compute validation; Reiny
integration does not change the application's hardware-adapter policy.

## Development

Rust 1.97 is pinned, matching the other Reiny ecosystem repositories.
Run these commands from this repository root:

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```

GPU computation normally requires a hardware adapter. CI explicitly enables
`TESSERA_ALLOW_SOFTWARE_ADAPTER=1` to run the real Vulkan kernels on Mesa.
This does not enable software adapters by default in applications.

Native examples live in `crates/tessera-physics/examples` and
`crates/tessera-mpm/examples`; those directories also contain WebGPU browser
examples. Python bindings and executable examples are documented in
[`crates/tessera-python3d/README.md`](crates/tessera-python3d/README.md).

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
