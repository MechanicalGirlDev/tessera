# Nexus physics contracts in Tessera

Reference: [dimforge/nexus at ca2cc91](https://github.com/dimforge/nexus/tree/ca2cc91).
The comparison uses the Nexus implementation and its Rapier 0.36.0 contracts.
Tessera retains native Rust CPU and WebGPU solvers; it does not call Nexus,
Rapier or a Python physics package at runtime.

## Implemented contracts

| Contract | Native implementation | Python surface / regression evidence |
| --- | --- | --- |
| Articulated models and joint identity | `urdf.rs`, `mjcf.rs`, `articulation.rs`; fixed, revolute, prismatic, spherical and expanded planar/floating components, mimic maps, armature and imported physical properties | `UrdfOptions`, `JointInfo`, `joints`, `joint_components`, `link_index`, batch environment metadata; `urdf_policy`, `loader_metadata` |
| Selected collision/visual URDF geometry | Loader scale/shift, imported-mass policy, adjacency collision policy, convex parts or indexed surfaces | `from_urdf_with_options`, `BatchModel.urdf_options`, `MeshAsset.indexed_parts`; `urdf_indexed_mesh` |
| Native collider families | Spheres, capsules, boxes, cylinders, cones, convex parts, indexed meshes, heightfields and polylines in the owners that expose them | GPU primitive shapes and articulated scene shapes; existing collider, manifold, terrain and scene tests |
| Materials, filtering and sleep | Native contact coefficients, collision masks/exclusions, contact impulses and per-body sleep | Single/batch material and filtering APIs, contact impulse readback, native sleep regressions |
| Spherical contact witnesses | CPU and direct/resident GPU normal impulses preserve spherical support tangents instead of interpolating hull-center tangents, avoiding artificial off-axis sphere torque | CPU/direct GPU sphere-hull geometry tests and the composed indexed-mesh scene regression |
| Force and acceleration motor gains | `JointMotor`, `SphericalJointDrive`, `GpuRigidAxisServo`; acceleration gains use the projected diagonal of the coupled inverse inertia | `MotorModel.FORCE_BASED` / `ACCELERATION_BASED`; `native_drive_contract`, `gpu_drive_owner`, `gpu_rigid_spherical_drives` |
| Physical motor caps | Zero, finite nonnegative, or positive-infinite force/torque caps; invalid caps rejected | `JointMotor.max_force`, `GpuAxisServo.max_force`, GPU axis motors; native drive regressions |
| Spherical impulse axis controls | Body-A-local shortest quaternion rotation-vector axes, velocity motors, position servos and limits; settings survive owner rebuilding | Single/batch `spherical_*` APIs and `GpuSphericalLimit`; `gpu_rigid_spherical_owners` |
| Resident spherical action scatter | GPU-produced position/velocity action buffers, partial axis updates, independent delays and selected cancellation | Native `GpuSphericalMotorTargetMapping` and dynamics-owner encode methods; standalone scatter and `gpu_drive_owner` tests |
| Compliant articulated constraints | Separate dynamic/fixed contact, point/fixed loop and coupling policies; compliant joint stops on CPU and GPU | `ConstraintCompliance`, `ConstraintPolicy`, limit setters; `articulated_compliance`, `articulated_owner_contract` |
| Prescribed generalized coordinates | Selected floating-root/scalar/tangent coordinates retain prescribed velocity; the dynamic principal mass system excludes them from force and impulse response. Public floating-mode changes preserve independent root/joint selections and snapshots | `set_prescribed_coordinates`; native mixed dynamic/prescribed owner, floating-mode snapshot and scene regressions |
| Complete simulation state | Joint configuration/velocity, root pose/twist and explicit spherical orientations; atomic validation and velocity-zeroing teleport | `SimulationState`, `set_simulation_state`, `set_configuration`, CPU snapshots; `simulation_state` |
| Physical resident reset templates | Mass, inertia, COM, joint frames/axes/limits, armature, gravity, persistent loads, motors, prescribed selection and held actuator controls | Native dynamics templates; selected owner reset regressions |
| Full scene selected resets | Dynamic bodies, contact-bound prescribed bodies, collider-free prescribed bodies and static poses; GPU integration anchors restored, selected contact/sleep/action history cleared | Native `GpuSceneResetTemplates`; `snapshot_gpu_resident` / `restore_gpu_resident` on Python single and batch articulated owners; native scene and Python GPU reset regressions |
| Imported actuator feedback | Motor, position, velocity, damper and supported general force laws evaluated using current q/dq every GPU microstep; signed gear and asymmetric clipping, deterministic additive targets | `controls`, `set_controls`, `step_controls`, `step_controls_gpu_resident`, single/batch metadata; native `gpu_mjcf_actuator` and Python `actuator_feedback` |
| Unsupported actuator metadata | Original element kind, resolved attributes, declaration order, optional joint coordinate and diagnostic retained; unsupported entries apply no force | `MjcfActuatorDynamics.UNSUPPORTED`, `actuator_info`, `actuator_names`; native import and Python feedback regressions |
| Runtime global settings | Gravity changes preserve body state; GPU rigid contact coefficients remain adjustable without rebuilding bodies | `gravity` / `set_gravity`, `GpuSolveParameters`; `runtime_parameters` |
| MPM material and boundary behavior | Native elastic/fluid/snow/sand models, linear sand Drucker-Prager hardening and bilateral free-slip relative normal projection on CPU and GPU | MPM material and obstacle records; CPU, direct GPU and resident regressions |
| Runtime MPM settings | Gravity, bounds, cell width, timestep limit, frame subdivision and CPIC enablement; pending GPU changes rejected atomically | `MpmWorld.set_params`, `set_gravity`, `set_substeps`, `set_use_cpic`; `runtime_mpm_*` |

The existing native APIs also provide heterogeneous batches, external wrenches,
ray/point/shape queries, contact observations, link twists, IMU kinematics and
rigid/MPM synchronization and coupling. Their owner-specific support and
step/readback semantics remain documented in the crate READMEs.

## Semantics and limits

- CPU state is f64; resident kernels use f32. GPU-representability validation is
  separate from CPU input validity. Matching contracts does not imply identical
  trajectories, convergence, warm-start behavior or timestep accuracy with Rapier.
- Scalar and tangent spherical acceleration drives project the inverse of the
  coupled native mass system. Their caps are physical force/torque caps, not
  acceleration caps. Positive infinity explicitly disables a cap.
- Resident spherical position actions replace components of `Log(target)` in the
  parent joint frame, then store `Exp(vector)`. This is a shortest rotation vector,
  not three Euler angles. Partial updates retain the other current logarithm
  components; rotations outside the principal ball wrap through the quaternion.
- Indexed articulated meshes retain topology and holes. CPU and resident contact
  preparation expand each triangle into its native thin support prism
  (`TRIANGLE_HALF_THICKNESS = 1e-4` metres). This is not a whole-mesh convex hull or
  a moving-mesh GPU BVH. Very large articulated meshes incur per-triangle contact
  preparation cost. URI resolution and file decoding belong to the caller.
- Full-scene templates belong to the resident owner that captured them. Scene
  reset selects original environment slots with fixed geometry topology; it does
  not transplant a scene into an unrelated owner. Low-level articulated
  templates additionally support compatible source/destination environments and
  explicit floating-root offsets.
- A packed articulated contact solve has one shared contact/loop compliance
  policy. Constructing a batch with differing policies reports an error.
- Imported unsupported actuator definitions remain inspectable but undriven.
  Retention is not implementation of activation dynamics, tendon/site actuation
  or arbitrary user-defined actuator callbacks.
- GPU template capture requires a synchronized resident owner. CPU edits that
  invalidate that owner require new templates after rebuilding it.

## Verification

Repository gates:

```sh
cargo fmt --all --check
cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
cargo test --locked --workspace --all-features
```

GPU tests execute actual wgpu kernels. CI explicitly opts into Mesa software
compute; applications still require a hardware adapter unless they opt in.
`crates/tessera-python3d/examples/nexus_contracts.py` exercises the generated
UniFFI API, including old record-constructor defaults, complete state, imported
unsupported metadata, selected resident reset and indexed URDF assets.

On disk-constrained Windows hosts, the same full suite can be built with
`CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 CARGO_INCREMENTAL=0`.
These settings remove debug symbols and incremental caches; they do not disable
tests, debug assertions or GPU execution.

The final Windows workspace run completed with exit code 0: 729 native physics
unit tests, 102 MPM unit tests, 54 Python binding unit tests, all integration
targets and doc-tests passed. Formatting and the strict workspace Clippy gate
also passed. The generated UniFFI `nexus_contracts.py` and `basic.py` examples
were exercised successfully.

The full run used the debug-symbol settings above, `CARGO_BUILD_JOBS=4` and
`RUST_TEST_THREADS=1`, with
`cargo test --locked --workspace --all-features --no-fail-fast -- --nocapture`.
An earlier parallel run reported one Vulkan driver-loading failure during
adapter creation; the same moving-anchor regression passed alone and in the
final serial workspace run. The serial setting changes test scheduling, not
the solver, assertions or GPU backends.
