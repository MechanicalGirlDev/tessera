# Tessera

Vendor-neutral 3D rigid-body and Material Point Method physics, extracted from
[HumanoidSystem](https://github.com/MechanicalGirlDev/humanoid-system).
Tessera has no HumanoidSystem, robot configuration, or Reiny dependency.

## Crates

- `tessera-physics`: CPU rigid-body and articulated dynamics, URDF/MJCF loading,
  contact solvers, and optional WebGPU compute (`gpu-contact`).
- `tessera-mpm`: CPU and WebGPU Material Point Method simulation, including
  optional rigid-body coupling.
- `tessera-python3d`: UniFFI bindings and a platform-specific `tessera3d` Python wheel.

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
