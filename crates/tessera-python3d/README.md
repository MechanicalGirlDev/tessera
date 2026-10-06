# tessera3d

A UniFFI module for driving Tessera's 3D physics worlds from Python. It maps the Rust physics model to UniFFI records and objects, and the wheel bundles the generated Python code with the shared library. `SphereWorld` runs on the CPU in f64. `GpuSphereWorld`, `GpuPrimitiveWorld`, `GpuSphereBatch`, and `GpuPrimitiveBatch` keep rigid body state on wgpu. `ArticulatedWorld` can be built from URDF / MJCF strings. `MpmWorld` provides the material point method on the CPU and WebGPU.

## Building the wheel

Run the following in `crates/tessera-python3d`. `setup.py` builds `tessera-python3d` from this repository's Cargo workspace and generates the UniFFI Python code from the same library. The wheel lands in `dist`.

```powershell
uv run --no-project --with build --with setuptools --with wheel python -m build --wheel --no-isolation
python -m pip install (Get-ChildItem dist/tessera3d-*.whl | Select-Object -First 1 -ExpandProperty FullName)
```

Set `TESSERA_WHEEL_PROFILE=debug` at build time for a shorter development build. The default is release. The output targets the OS and CPU architecture it was built on, and the GPU paths need a hardware adapter that wgpu can select. Run `cargo test -p tessera-python3d` and `cargo clippy -p tessera-python3d --all-targets -- -D warnings` from the repository root.

## Usage

Zero-mass GPU bodies can follow a prescribed velocity through `set_kinematic_motion`.
A single world takes `(body_index, GpuKinematicMotion(linear=Vec3(...), angular=Vec3(...)))`,
and a batch takes `(environment_index, body_index, motion)`. `motion=None` zeroes the
velocity and turns the body static again. No CPU readback of the pose is needed, and the command
is ignored for dynamic bodies. Non-finite velocities, f32 overflow, and out-of-range indices are errors.
See [examples/kinematic.py](examples/kinematic.py) for usage and verification.

GPU terrain is passed as `GpuPrimitiveShape.HEIGHTFIELD(rows, columns, heights, scale)`.
GPU polylines are passed as `GpuPrimitiveShape.POLYLINE(vertices=[Vec3(...), ...], segments=[GpuSegment(a=0, b=1), ...])`.
They're treated as zero-thickness segments. GPU-resident contact currently supports spheres, capsules, boxes, convex hulls, and the finite ground.
Contacts with a cylinder end cap, a cylinder side parallel to the axis, or a cone base generate two points from the ends of the overlap interval. Other cylinder and cone contacts use one point. The Rust GPU path has been verified for axis-aligned endpoints, touching boundaries with shared rotation, and 2 seconds of single-body support on a split segment. Arbitrary relative orientations, general contact accuracy on curved surfaces, and stacking still need verification.
Polyline pairs handle segment crossings, collinear overlaps, and endpoint contacts at depth 0, and pass up to 4 deduplicated points. Between zero-thickness segments the normal isn't unique, so crossings use the cross product and parallel segments use a deterministic perpendicular direction. Dynamic collision response between segments hasn't been verified.
Polylines against triangle meshes are also treated as zero-thickness intersections. They generate the point where a segment pierces a triangle, or the endpoints of a coplanar overlap, and pass up to 4 points at depth 0. Dynamic collision response and continuous collision detection haven't been verified.
Triangle mesh pairs test each mesh's edges against the other mesh's triangles and handle penetration, coplanar overlap, and vertex contact at depth 0. This isn't the same as computing penetration depth for closed meshes as solids, and dynamic response hasn't been verified.
Capsule contacts build a manifold by adding up to 3 endpoint candidates whose normals line up with the primary contact.
Box contacts use a separating axis test plus segment clipping, and build the manifold from the endpoints of the contact interval.
Convex hull contacts likewise use a separating axis test over face normals and edges, with clipping against the face half-spaces.
Segment indices must be in range, and the two endpoints must be at different positions.
See [examples/polyline.py](examples/polyline.py) for usage.
`heights` are row-major heights with rows along y and columns along x. `scale.x/y` are the full widths,
and `scale.z` is the height multiplier. The terrain becomes a z-up triangle mesh centered on the body origin's x/y,
and uses the GPU BVH and mesh contact. Cell diagonals and face orientation match the CPU
`HeightFieldGeometry`. It's an error if converting to f32 overflows a coordinate or if a triangle
is degenerate. [examples/heightfield.py](examples/heightfield.py) checks contact against
terrain of different heights and batch independence.

GPU contact impulses are available through `readback_contact_impulses()` on a single world and
`readback_contact_impulses_environment(index)` on a batch.
`GpuContactImpulseReadback.dt` is the duration of the last solver substep, and `contacts` are the manifold points
with a positive normal impulse. `body_a` is `None` for the ground, and `body_b` is the body
that receives the positive impulse. Batch indices are per-environment. Each contact returns `normal_impulse`,
the world-space `normal` and `tangent_impulse`, and the combined `impulse_on_body_b`.
These are solver history, not frame totals. Support history may linger while bodies sleep,
so they aren't independent force sensor readings. After a cache clear, `dt` is `None`.
See [examples/contact_impulses.py](examples/contact_impulses.py) for a runnable verification example.

For MPM particles, elastic / sand / fluid / snow materials, moving obstacles, and adding or removing particles in chunks,
see [examples/mpm.py](examples/mpm.py). `step(dt)` runs on the CPU, `step_gpu(dt)` is
the WebGPU path with automatic CFL substeps, and `step_gpu_fixed(substep_dt, steps)` is the GPU path with
the world's fixed substep. Each GPU call syncs particle state back to the CPU.
`sample_mpm_closed_mesh_volume` returns interior points of a closed triangle mesh.
`MpmWorld.add_closed_mesh_particles(MpmMeshEmissionInput(...))` adds those same points as MPM particles with volume `spacing³` and mass
`density × spacing³`, and returns a chunk ID for later removal.
The mesh must be closed, non-self-intersecting, and consistently oriented. `max_candidates` caps the number
of AABB grid candidates. Once added, the particles work with both CPU and GPU steps.
Consecutive `step_gpu_fixed` calls with the same timestep reuse the resident session's particle/grid buffers.
Adding regular or closed-mesh particles and removing chunks rebuilds the particle buffers inside the
GPU resident session and keeps the same session, as long as there are no unsynchronized steps and
rigid coupling isn't in use. A CPU step, a regular GPU step, editing particles during coupling, or changing the timestep
rebuilds the session when needed. Obstacle and force updates keep the particle and grid buffers.
`set_particle_force` writes to the GPU buffer a force that's consumed only by the next substep.
With a resident session active, forces that f32 can't represent are rejected and CPU/GPU state is left untouched.
If GPU execution fails, the session is dropped and the last synchronized CPU state is kept.
Each fixed-substep call runs 1 to 64 substeps. See
[examples/mpm_resident.py](examples/mpm_resident.py) for a verification example.
`MpmWorld.sync_gpu_sphere_world(rigid)` and
`MpmWorld.sync_gpu_primitive_world(rigid)` update one-way MPM obstacles from the latest state, shapes, and friction
of a GPU rigid body world. Call them again after advancing the rigid bodies and before advancing MPM.
They support spheres, boxes, capsules, cylinders, cones, convex shapes, polylines, triangle meshes, and the finite ground.
Polyline segments and mesh triangles are expanded into individual obstacles. Syncing reads the rigid body
state back to the CPU once. There's no reaction from MPM back onto the rigid bodies. A wheel
example is [examples/mpm_gpu_rigid_sync.py](examples/mpm_gpu_rigid_sync.py).
Fixed substeps also run without boundaries, rebuilding the GPU sparse grid every substep.
Coordinates are limited to the f32 range and node indices to the i32 range. Results outside those ranges aren't applied.
The regular `step_gpu` also uses a GPU hash topology for sparse particle sets.
If the hash buffer or dispatch capacity limits are exceeded, it falls back to the CPU compact topology.
The fixed path rebuilds the hash grid every substep, so it follows particles as they move between grid cells.

```python
from tessera3d import SphereInput, SphereWorld, GpuSphereWorld, GpuSphereBatch, Vec3

ball = SphereInput(
    center=Vec3(x=0.0, y=0.0, z=0.4),
    velocity=Vec3(x=0.0, y=0.0, z=-1.0),
    radius=0.5,
    mass=1.0,
)
gravity = Vec3(x=0.0, y=0.0, z=0.0)

cpu = SphereWorld([ball], gravity, 10.0, 1.0, 0.0, 0.001, 12)
cpu.step(0.001)
print(cpu.states(), cpu.contact_force(0))

gpu = GpuSphereWorld([ball], gravity, 10.0)
gpu.step_substeps(1.0 / 120.0, 10)
print(gpu.readback())

batch = GpuSphereBatch([[ball], [ball]], gravity, 10.0)
batch.step(0.001)
print(batch.readback_environment(1))
```

Load URDF / MJCF with `ArticulatedWorld.from_urdf(xml, floating_base)` and `ArticulatedWorld.from_mjcf(xml)`, then use `positions()` / `set_positions()`, `step()` / `step_gpu()`, and `link_poses()`. A complete runnable set is in [examples/basic.py](examples/basic.py).

`ArticulatedWorld.link_contact_wrench(link)` returns the contact force and torque on an articulated link from the last substep. `ArticulatedBatch.link_contact_wrench(environment, link)` returns the same `LinkContactWrench`. Force and torque are in world coordinates, and torque is taken about the link origin.
`link_twists()` returns the world-frame linear and angular velocity at each link origin, in link order.
Query any point on a link with `link_point_twist(link, local_point)`. On a batch, pass
the environment index first. Floating-root and mimic joint velocities are included, and the values are computed from the current
joint state.
`link_point_acceleration(link, local_point, generalized_acceleration)` takes the current pose and velocity plus
a given generalized acceleration, and returns the point's world-frame linear acceleration and the link's angular acceleration.
For a floating root, the first 6 components are the world-frame translational and angular acceleration, followed by the independent joint components.
`link_imu` takes the same inputs and returns an ideal accelerometer reading in the link frame (specific force, with gravity subtracted),
plus angular velocity and angular acceleration. On a batch, pass the environment index first.
Computing a generalized acceleration that includes contact impulses, and modeling sensor noise and bias, are up to the caller.
See [examples/articulated_imu.py](examples/articulated_imu.py) for an example.
CPU, GPU, and per-environment batch examples are in [examples/articulated_contact_wrench.py](examples/articulated_contact_wrench.py).

Batches expose per-environment force, reset, and readback, plus a reset of all environments. A reset can't change the body count or sphere radii.

`GpuSphereWorld.add_body(body)` and `remove_body(index)` add and remove spheres at runtime. The remaining bodies' state is copied between GPU buffers, and only the removed body's state is read back so it can be returned. The contact buffers are rebuilt too, so this costs more than a regular step. Removal returns the deleted sphere's state and radius, and every later body index shifts down by one. Pending forces, warm-start impulses, and sleep timers are discarded when the configuration changes. `GpuSphereBatch` / `GpuPrimitiveBatch` can also add and remove bodies within an environment through `add_body(environment, body)` and `remove_body(environment, index)`, keeping the environment count. When the configuration changes, the remaining joints keep their body indices and their motor, servo, limit, and rotation angle history. Joints connected to a removed body are detached.

`GpuPrimitiveWorld` handles mixed scenes of spheres, oriented boxes, Z-axis capsules / cylinders / cones, convex shapes, and triangle meshes. Give each `GpuPrimitiveBody` an XYZW orientation, velocity, mass, and principal inertia, then use `step`, force/torque, material, `reset`, and state and contact readback. `readback_contacts()` returns only real contacts, and `body_b` is `None` for contacts with the finite ground. `add_body` / `remove_body` copy the remaining bodies' state on the GPU and rebuild the buffers, keeping state and materials. Removal reads back only the returned body's state. `reset` restores the initial state for the current shape configuration. Face contacts between boxes, between convex shapes, and between a box and a convex shape, as well as box and convex contacts with the finite ground, solve up to 4 points, and `readback_contacts()` returns each point separately. Edge-edge contacts, and pairs where the face boundary can't be determined, use one point. Face-parallel contacts between a convex shape and a capsule clip the axis against the convex face half-spaces and generate up to 2 points. Rounded ends and contacts that aren't parallel to a face use the existing single point. Convex-sphere pairs evaluate candidate axes from support faces, vertices, and edges, and convex vs box / convex pairs use face and edge SAT. Other mixed pairs without meshes use an approximate depth from GJK and candidate axes.
`GpuPrimitiveBatch` takes environments of mixed shapes as a nested list and keeps contacts separate between environments. It supports per-environment force/torque, materials, partial state readback, contact point readback, individual resets that keep the shapes, and a reset of all environments. Body indices in contact points are per-environment. When contact points are fetched, the all-pairs path transfers the target environment's slots, and the GPU-resident candidate path picks the target environment out of all candidates. Gravity and the finite ground are shared by all environments. See [examples/primitive_batch.py](examples/primitive_batch.py) for an example.

GPU-resident worlds and batches provide mutual masks through `GpuCollisionGroups(memberships, filter)`. A pair of bodies only makes contact when `memberships & the other's filter` is non-zero in both directions. Change and inspect individual bodies with `set_body_collision_groups` / `body_collision_groups`. Batches use per-environment indices as `(environment, body, groups)`. `set_ground_collision_groups` changes the shared mask of the finite ground. All bits are enabled by default. Worlds with few bodies update the candidate list, while large worlds apply the mask during the GPU LBVH traversal. The settings survive resets and body additions and removals in a single world.

```python
from tessera3d import GpuCollisionGroups

gpu.set_body_collision_groups(0, GpuCollisionGroups(memberships=0b01, filter=0b01))
gpu.set_body_collision_groups(1, GpuCollisionGroups(memberships=0b10, filter=0b10))
# Body 0 and body 1 no longer generate a contact pair.
```
GPU-resident `GpuSphereWorld` / `GpuPrimitiveWorld` support point constraints with `GpuBallJoint`, position and orientation constraints with `GpuFixedJoint`, single-axis rotation with `GpuRevoluteJoint`, and single-axis translation with `GpuPrismaticJoint`. When the four kinds share bodies, they're solved in the same GPU island. The prismatic slide direction is the Z axis of each body's local quaternion frame. The frame origin is used for the off-axis position constraint. On `GpuSphereBatch` / `GpuPrimitiveBatch`, pass per-environment body indices to the matching `set_*_joints_environment`. Bodies can be added and removed while joints are set. Joints connected to a removed body are detached, and the remaining joints keep their settings and rotation angle history. Replacing joints or resetting discards the warm start. `GpuAxisMotor` sets the axis velocity and maximum torque / force for revolute and prismatic joints. `GpuPrismaticLimit` limits the slide displacement between the two joint frames. `GpuAxisServo` sets position and velocity targets plus stiffness, damping, and a force limit for both joint types. `GpuRevoluteLimit` limits a continuous angle that can span several turns. Read the angle with `readback_revolute_angle(index)`, or `readback_revolute_angle_environment(environment, index)` on a batch. Topology rebuilds of the same joint keep the angle history, while a reset or an explicit body state overwrite reinitializes the matching history. Orientation is integrated with the rotation exponential map, and the number of turns is estimated from angular velocity, so rotations beyond π or 2π in one substep are still tracked. Large collision impulses and off-axis rotation make that estimate ambiguous, so verify with a small timestep. Contact and joint warm starts are applied once each, then contacts and joints are iterated alternately in that order inside the same GPU command encoder. Constraint islands are built separately for contacts and joints, and contact points aren't regenerated during iteration. For single-world examples, see [ball](examples/ball_joint.py), [fixed](examples/fixed_joint.py), [revolute](examples/revolute_joint.py), and [prismatic](examples/prismatic_joint.py).
The Python shape constructors are `GpuPrimitiveShape.SPHERE(radius=...)`, `.BOX(half_extents=Vec3(...))`, `.CAPSULE(radius=..., half_length=...)`, `.CYLINDER(radius=..., half_length=...)`, `.CONE(radius=..., half_length=...)`, and `.CONVEX(vertices=[Vec3(...), ...])`. Convex shapes need at least 4 finite, non-coplanar vertices. See [examples/primitives.py](examples/primitives.py) for an example. [examples/capsule_convex.py](examples/capsule_convex.py) verifies capsule / convex contacts on flat faces, slopes, and corners in both body orders. Triangle meshes are passed as `GpuPrimitiveShape.TRIANGLE_MESH(vertices=[Vec3(...), ...], triangles=[GpuTriangle(a=0, b=1, c=2), ...])`. Vertices must be finite, and triangle indices must be in range and non-degenerate. A BVH is built at creation time, and sphere and capsule contacts narrow down candidate triangles on the GPU. GPU-resident contact against meshes supports spheres, capsules, and the finite ground. Contact with other shapes isn't implemented. See [examples/triangle_mesh.py](examples/triangle_mesh.py) for usage.

Materials are set with `Material` and `CombineRule`. `SphereWorld` can set and get materials for bodies and the ground. `ArticulatedWorld` can set and get them for link colliders, scene colliders, and the ground. GPU spheres and batches can set and clear body and ground materials. A batch's ground material is shared by all environments. Joint position and velocity drives are set by passing a `JointMotor` to `set_joint_motor(slot, motor)`, and `None` clears them. The core's input validation applies unchanged.

Reflected inertia on joint axes is set and read with `set_joint_armature(edge, values)` / `joint_armature(edge)`. `edge` follows the joint order of `joint_ranges()`. Revolute and prismatic joints take one value, ball joints take three in X/Y/Z axis order, and fixed joints take an empty list. On a batch, pass the environment index as the first argument. MJCF `<joint armature="...">` and `default` classes are also loaded, and the values feed into the CPU joint step and the GPU mass matrix solve.

Passive joint springs and viscous damping are set by passing `JointPassive(stiffness, damping, rest_position)` to `set_joint_passive(slot, value)`. They're added to the generalized force independently of the motor force limit, and saved in snapshots and batch reset templates. Dry friction is set with `set_joint_friction(slot, force_or_torque)`. It clamps the signed joint impulse in the same iterations as contacts, so it handles both sticking and sliding. Read the setting back with `joint_friction(slot)`. MJCF scalar `stiffness`, `damping`, `springref`, and `frictionloss` with default classes, and URDF `<dynamics damping="..." friction="..."/>`, are loaded. MJCF `springdamper="timeconst dampratio"` derives spring and damping coefficients from the joint inertia at the reference pose plus armature, and takes precedence over directly specified coefficients. For URDF mimic joints, damping is added to the shared degree of freedom scaled by the square of the multiplier, and dry friction by its absolute value. Batch APIs take the environment index as the first argument.

MJCF `stiffness="a b c"` and `damping="a b c"` on hinge / slide joints are loaded as the spring force `-(a x + b x² + c x³)` (with `x = q - springref`) and the damping force `-(a v + b v|v| + c v³)`. The quadratic and cubic coefficients can be set on single worlds and batches with `set_joint_nonlinear_passive(slot, value)` using `JointNonlinearPassive`, and are saved in snapshots. The force's tangent with respect to position and velocity is evaluated every substep and folded into the joint's effective mass. On joints with `springdamper`, the directly specified nonlinear coefficients are also overridden.

MJCF `ref` on hinge / slide joints is loaded as the initial joint value. The loader compensates the joint origin, so adding `ref` doesn't change the initial link poses. `positions()`, joint limits, `springref`, and joint equalities use this absolute joint value. Hinge `ref` is converted from degrees or radians according to the MJCF `compiler angle`.

To couple two independent degrees of freedom, use `set_joint_couplings([JointCoupling(source, follower, multiplier, offset)])`. Reactions are distributed to both axes in the same iterations as contact and friction so that `follower = multiplier × source + offset` holds. Read the settings back with `joint_couplings()`. URDF mimic joints are converted to a shared degree of freedom at load time, so this API is for coupling independent joint axes. MJCF `<equality><joint joint1="..." joint2="..." polycoef="a0 a1 a2 a3 a4"/>` is loaded as a quartic relative to the initial pose. Omitting `joint2` pins `joint1` in place, and `active="false"` is skipped. Setting and reading `JointPolynomialCoupling` works on both single worlds and batches. MJCF equality `solref` / `solimp` and tendons aren't supported.

For closed-loop ball constraints that make points on two links of a multibody coincide, use `set_link_point_constraints([LinkPointConstraint(link_a, point_a, link_b, point_b)])`. Points are in each link's local coordinates. When `link_b=None`, `point_b` is a fixed point in world coordinates. The three axes of position error and point velocity are computed with the generalized Jacobian and fed into the same CPU/GPU iterations as contacts and joint axis couplings. They can be set and read on single worlds and per-environment batches, and are saved in snapshots and reset templates.

For fixed closed-loop constraints that match both position and orientation, use `set_link_fixed_constraints([LinkFixedConstraint(link_a, frame_a, link_b, frame_b)])`. Frames are given as `LinkPose(position=Vec3(...), orientation=Quaternion(x, y, z, w))`. When `link_b=None`, `frame_b` is a fixed frame in world coordinates. Three axes of position error and three axes of rotation error go into the same CPU/GPU bilateral impulse iterations as contacts. MJCF `<equality><connect>` and `<weld>` generate the matching link constraints from body or site references. Body-form `connect` uses `anchor`. Body-form `weld` uses `anchor` in body2 coordinates and either the given `relpose` or the relative frame from the initial pose. The `site1` / `site2` form matches the local points and frames of both sites, and sites directly under worldbody are treated as fixed anchors. `weld torquescale="0"` is treated as a point constraint, and the default `1` as a fixed-frame constraint. Other `torquescale` values and `solref` / `solimp` aren't supported, and the loader returns an error if they're given.

To connect an independent scene body to an articulated link, add a rigid body with `add_scene_body(SceneBodyInput(...))`, then use `set_link_scene_point_constraints([LinkScenePointConstraint(link, link_point, body, body_point)])` or `set_link_scene_fixed_constraints([LinkSceneFixedConstraint(link, link_frame, body, body_frame)])`. `SceneBodyInput` takes a world pose, initial linear and angular velocity, mass, the inertia tensor about the body origin (9 values, row-major), a persistent force, and a list of `SceneColliderInput(frame, shape)`. An empty list creates a body with no collision shape. `SceneShape` can be a sphere, box, capsule, cylinder, cone, convex shape, triangle mesh, heightfield, or polyline, and one body can carry several colliders. In the generated Python you'd write, for example, `SceneShape.BOX(half_extents=Vec3(...))`. For a plain sphere, `add_scene_sphere(pose, radius, mass)` also works. Constraint points and frames are in each object's local coordinates. `scene_body_state(body)` returns the pose, velocity, mass, inertia, persistent force, and collider count. `set_scene_body_pose` moves the body and clears the contact cache, and `set_scene_body_velocity` updates a dynamic body's velocity. `set_scene_body_force(body, force)` sets a persistent force in world coordinates. `remove_scene_body(body)` detaches the constraints that reference it and shifts later body indices down. Batch APIs take the environment index as the first argument.

External collision meshes are passed through `ArticulatedWorld.from_urdf_with_meshes(xml, floating_base, assets)` and `ArticulatedWorld.from_mjcf_with_meshes(xml, assets)`. Each `MeshAsset` URI must match the reference string in the document, and each `ConvexMeshPart` supplies the already-decomposed convex vertices, outward unit face normals, and unit edge directions. The loader applies non-uniform scale to vertices, normals, and edge directions. Reading files and convex decomposition of arbitrary meshes are up to the caller.

`ArticulatedBatch` builds URDF / MJCF environments from a list of `BatchModel`. You can change each environment's coordinates, velocities, motors, and the materials of link colliders, scene colliders, and the ground, and reinitialize environments individually with `publish_reset_template(index)` and `reset_environment(index)`. `step(dt, torques)` is the CPU path, and `step_gpu(dt, torques)` dispatches contact impulses for compatible environments to the GPU together. `step_gpu_device_mass(dt, torques)` inverts the mass matrices of environments with different degree-of-freedom counts together on the GPU, and submits the contact iterations in the same GPU submission. Consecutive calls with the same environment layout reuse the GPU mass buffers. A single `ArticulatedWorld` has APIs with the same names. `torques` is a nested list in environment order, and each inner list's length must match that environment's degrees of freedom. Collision detection and joint dynamics setup still run per environment on the CPU, and the inverse mass matrix is read back every substep to prepare contacts.

`set_implicit_coriolis(enabled)` on articulated worlds and batches toggles whether Coriolis and centrifugal forces are evaluated iteratively at the midpoint velocity. On a batch, pass the environment index as the first argument. It's enabled by default. When disabled, the bias evaluated at the current velocity is used for that substep. The setting is included in snapshots and environment reset templates.

`ArticulatedWorld.set_self_contacts_enabled(enabled)` toggles self-contact between non-adjacent links. On `ArticulatedBatch`, pass the environment index as the first argument. It's enabled by default, and contacts within the same link or between adjacent links are always excluded. The setting is saved in snapshots and per-environment reset templates, and doesn't affect contacts with scene bodies.

The CPU `SphereWorld` and the GPU sphere and mixed-shape worlds / batches can cap the linear speed of dynamic bodies in m/s with `set_max_linear_speed(Some(value))`. `None` removes the cap. The CPU accepts an f64 limit and the GPU an f32 limit. On a batch, the cap is shared by all environments. Speed is clamped during integration and after contact and joint impulses, and static body velocities aren't changed.

Integration and dynamics setup for articulated worlds run on the CPU on the Rust side, and `step_gpu` uses the GPU for contact handling. The GPU-resident mixed primitive path is independent of articulated worlds.

## GPU temporal contact step

`GpuSphereWorld` / `GpuPrimitiveWorld` / `GpuSphereBatch` / `GpuPrimitiveBatch` expose
`step_temporal(frame_dt, substeps, settings)`. `frame_dt` is the duration of the whole frame, which differs from
the substep duration of `step_substeps`. `substeps` ranges from 1 to 1024.
`settings=None` uses the Rust defaults. To change coefficients, get a
`GpuTemporalSettings` from `default_gpu_temporal_settings()`.

```python
from tessera3d import default_gpu_temporal_settings

settings = default_gpu_temporal_settings()
settings.normal_frequency = 30.0
settings.damping_ratio = 1.0
settings.static_normal_frequency = 60.0
settings.static_damping_ratio = 1.0
settings.max_corrective_velocity = 3.0
settings.iterations = 4
settings.speculative_margin = 0.1
world.step_temporal(1.0 / 60.0, 4, settings)
```

External forces and torques are captured once at the start of the frame and applied to every substep.
Contacts are detected at the start of the frame, and each substep updates the gap from per-body local anchors.
`speculative_margin=0` detects only real contacts. A positive value also solves separated contacts
whose initial gap is at most the margin. `ground_only=True` limits the speculative test to the finite ground.
Pair tests that use a margin report distance computation failures through a 4-byte GPU status.
Paths with many bodies read the frame-start candidate pairs back to the CPU.
Sphere-only worlds / batches can call
`step_temporal_speed_bounded(frame_dt, substeps, settings, joint_settings)` after `set_max_linear_speed(Some(speed))`.
The pair margin is derived from the maximum relative travel of the two bodies, or the explicit margin is used if it's larger.
`joint_settings` may be `None`. Mixed shapes, a missing speed cap, and
`ground_only=True` are rejected. There's no CCD guarantee for sweeps of rotating shapes or curved trajectories.

Ball, fixed, revolute, and prismatic joints are solved alternately with contacts.
Joint frequency, damping, linear correction speed, and iteration count come from the contact settings,
and the angular correction speed defaults to 3 rad/s. Motor / servo force limits apply to the impulse
over the whole substep. The servo spring term is captured at the start of the substep, and velocity damping is solved implicitly.
Limits clamp the approach speed based on the gap, and when crossed, apply a soft correction using frequency / damping.
New collisions in the middle of a frame aren't re-detected, and there's no CCD.
[examples/temporal.py](examples/temporal.py) verifies ground support for sphere / box in single worlds and batches,
the frame lifetime of external forces, and rejection of invalid coefficients.
`examples/temporal_drives.py` checks the motor force budget, implicit servo,
soft limits, and state preservation of undriven environments in a multi-environment batch.

### Setting separate TGS coefficients for contacts and joints

All four GPU world / batch types provide `step_temporal_with_joints(frame_dt, substeps, settings, joint_settings)`.
Passing `None` for `settings` uses the default contact settings.
Get `joint_settings` from `default_gpu_temporal_joint_settings()`.

```python
from tessera3d import default_gpu_temporal_settings, default_gpu_temporal_joint_settings

contacts = default_gpu_temporal_settings()
contacts.iterations = 8
joints = default_gpu_temporal_joint_settings()
joints.frequency = 20.0
joints.damping_ratio = 1.0
joints.max_linear_correction_speed = 0.2  # m/s
joints.max_angular_correction_speed = 0.3  # rad/s
joints.iterations = 4
world.step_temporal_with_joints(1.0 / 60.0, 4, contacts, joints)
```

When contacts and joints have different iteration counts, each substep's alternating iterations run each up to its own count.
`ground_only` is also available. All coefficients are validated before the force capture and GPU submission.
These joint settings apply to passive constraints and limits. Motor / servo force budgets and
servo gains use the values set on each joint.
The existing `step_temporal` keeps picking joint coefficients from the contact settings.

[examples/temporal_joint_settings.py](examples/temporal_joint_settings.py) runs all four APIs and checks
independent coefficients, analytic values, preservation of untouched environments, and up-front rejection of invalid input.
## GPU raycast

A single GPU world exposes `cast_rays(rays)`, and a GPU batch exposes `cast_rays_environment(environment, rays)`.
They run against the latest GPU physics state and shapes, transfer no body state, and read back only the ray results.
These are synchronous APIs. To keep results on the GPU and pass them to a later pass, use the Rust prepared query API.

```python
from tessera3d import GpuRay, GpuCollisionGroups, Vec3

ray = GpuRay(
    origin=Vec3(x=0.0, y=0.0, z=5.0),
    direction=Vec3(x=0.0, y=0.0, z=-1.0),
    max_t=10.0,
    groups=GpuCollisionGroups(memberships=0xffffffff, filter=0xffffffff),
    excluded_body=None,
    body_range=None,
    solid=True,
)
hit = world.cast_rays([ray])[0]
if hit is not None:
    print(hit.body, hit.point, hit.normal, hit.toi, hit.feature)
```

Ray directions don't have to be unit length, and the hit point is `origin + direction * toi`.
A miss is `None`, and a ground hit has `body` set to `None`. Inside a volume with `solid=True`, you get
`toi=0`, `inside_solid=True`, and a zero normal. Polyline normals aren't unique either, so they're 0.
`GpuRayBodyRange(start, end)` selects a half-open range. Group tests against colliders are bidirectional.
In a batch, exclusions, ranges, and hit body IDs are per-environment, and other environments at the same coordinates aren't searched.
Each query uses the current buffers, so it works after a step, after adding or removing bodies, and after changing groups.
Valid queries and rejected invalid input don't change pending forces or step history.

[examples/raycast.py](examples/raycast.py) runs all four GPU world / batch types and checks the nearest hit,
filters, ground, solid, environment separation, positions after a step, and preservation of pending forces.
The query pipeline is reused per contact set and carries over across contact buffer rebuilds.
With 16 or fewer bodies, the given range is scanned linearly, and meshes / polylines use their internal BVH.

### GPU point projection

`project_points(queries)` on `GpuSphereWorld` / `GpuPrimitiveWorld` and
`project_points_environment(environment, queries)` on `GpuSphereBatch` / `GpuPrimitiveBatch`
find nearest points. A `GpuPointQuery` takes a world-space `point`, a finite
`max_distance`, `groups`, `excluded_body`, `body_range`, and `solid`.
Ranges share `GpuRayBodyRange(start, end)`, and batch indices are per-environment.

Results are `GpuPointHit | None` in input order. A hit exposes `point`, `normal`, `distance`, `is_inside`,
`feature`, and `body`. The ground has `body=None`, and a miss is `None` as a whole.
Interior points get distance zero with `solid=True`, or are projected onto the boundary with `False`.
Meshes / polylines are treated as thin surfaces and lines, with no inside/outside volume test.
`normal` is a unit direction from the boundary toward the input point for exterior points, and from the input point toward the nearest boundary for interior points.
It's zero when the input point and the boundary point coincide. For thin surfaces and lines it faces the input point's side.

Calls synchronously read back only the results, with no body state readback.
See `examples/point_projection.py` for a working example. Large scenes use the LBVH search described below.
Convex shapes compare the projection onto support faces against the nearest points on hull edges. Performance evaluation for large scenes
isn't finished.
### Searching large scenes

The synchronous `cast_rays` / `project_points` and the matching batch APIs build AABBs and an LBVH from the current GPU state
when the shared world has 17 or more objects. With 16 or fewer, they search linearly. There's no CPU readback
of body state or candidate pairs. The search tree is rebuilt on every call, even after objects move or are added,
so stale bounds are never reused. Searching doesn't consume pending external forces.

To run rays and points against the same state together, use
`query_scene(rays, points)` on a single world, or
`query_scene_environment(environment, rays, points)` on a batch. The results in
`GpuSceneQueryHits.rays` / `.points` are in input order, and misses are `None`.
Even with 17 or more bodies, the GPU scene tree is built once and both queries run
in a single submission. Batch body IDs, exclusions, and ranges are per-environment.

`examples/scene_queries.py` runs the four interfaces with 32 objects. In Rust,
`encode_ray_queries` / `encode_point_queries` add bounds / tree / query passes after earlier state
updates in the same encoder, and leave the results on the GPU.
`encode_scene_queries` shares one scene tree between rays and points.
The explicit `prepare_*` APIs use either a linear search or a tree supplied by the caller,
and in the latter case the caller manages bounds updates and pass ordering.

# Direct coupling between GPU rigid bodies and MPM

`MpmWorld.step_gpu_fixed_with_sphere_world(rigid, substep_dt, steps)` and
`MpmWorld.step_gpu_fixed_with_primitive_world(rigid, substep_dt, steps)` transfer the current rigid body state to the MPM obstacles
on the same GPU device, then advance the particles. `step` the rigid world first.
The first call reads the rigid body state back to the CPU once to build the obstacle shapes.
When you continue with the same rigid world and timestep, later obstacle positions, orientations, and velocities are updated
on the GPU. Particle state is still read back to the CPU on every MPM step.

After changing shapes, masses, friction, the ground, or the body count, the coupling has to be rebuilt.
See [mpm_gpu_rigid_sync.py](examples/mpm_gpu_rigid_sync.py) for an example.

To submit several rigid frames without reading particles back to the CPU, call
`submit_gpu_fixed_with_sphere_world` or `submit_gpu_fixed_with_primitive_world`
after the rigid `step`. Advancing the rigid bodies in between and submitting again keeps the order on the same queue.
`pending_gpu_substeps()` returns the number of unsynchronized steps. `particles()` and `substeps()` return
the last synchronized state, and `synchronize_gpu()` checks the GPU results and applies them to the CPU.
Changing the timestep or the rigid world, or editing MPM state, while unsynchronized steps are pending is an error.

To feed particle reactions back into rigid bodies, call `submit_gpu_two_way_with_sphere_world` or
`submit_gpu_two_way_with_primitive_world`. Each call submits one MPM
substep and updates rigid body linear and angular velocity on the same GPU queue.
`step_gpu_two_way_with_sphere_world` and `step_gpu_two_way_with_primitive_world`
also synchronize particle state. To integrate rigid body position and orientation, call that world's `step` separately.
Up to 16 obstacles are supported, and each element of an expanded polyline or triangle mesh counts toward that.
See [mpm_gpu_two_way.py](examples/mpm_gpu_two_way.py) for an example.
Once obstacles are placed, pass `MpmBoundary` values in obstacle order to `set_obstacle_boundaries`
to choose `SLIP`, `STICK`, `SEPARATE`, or `NON_REFLECTING`.
Check the current values with `obstacle_boundaries()`. Changes are rejected while unsynchronized GPU steps are pending.
Boundary settings carry over when coupling with the same rigid world is rebuilt because the timestep changed.

To also advance rigid body position and orientation inside the MPM substep, use `submit_gpu_owned_with_sphere_world`
or `submit_gpu_owned_with_primitive_world`. `step_gpu_owned_with_*` also
synchronizes particle state. Don't layer a separate rigid `step` on rigid bodies that use this path.
The MPM-owned path applies rigid body gravity and the MPM reaction, but it doesn't handle rigid-rigid contact constraints, joints,
external forces accumulated in the rigid world, or sleep settings.
If `set_max_linear_speed` is set on the rigid world, the cap also applies to the velocity after the MPM reaction and gravity are added.

To separate grid transfers between particle groups, use `set_particle_transfer_color(index, color)`.
Particles of different colors don't share mass or momentum even at the same coordinates. Before changing it, pending GPU steps
must be synchronized, and the resident grid is rebuilt. Check manual settings through
`transfer_color` in `particles()`. [mpm_transfer_colors.py](examples/mpm_transfer_colors.py) compares
the separation results on the CPU and WebGPU. For thin `TRIANGLE_PRISM` obstacles,
set CPIC groups in obstacle order, for example `set_obstacle_cpic_groups([0])`.
`None` disables it, and groups range from 0 to 31. Check them with `obstacle_cpic_groups()`.
Particle positions are projected onto the triangle's plane, and if they fall inside the triangle, grid transfers on the two sides are separated.
This is a side test against a finite triangle. A full CPIC distance field and temporal side tracking aren't implemented.

# Per-link external forces on articulated bodies

`ArticulatedWorld` and `ArticulatedBatch` can set, through `set_link_external_wrench`, a world-frame force acting
at each link's center of mass and a torque about that center of mass.
`set_link_gravity_scale` changes gravity for that link only. The settings persist, so to clear them,
set the force and torque back to zero and the gravity scale back to 1. Read the current values with `link_load`.
Batch reset templates include these settings too. Articulated dynamics currently run on the CPU,
and `step_gpu` passes the results to the GPU contact computation. See [basic.py](examples/basic.py) for usage.

# Kinematic scene bodies in multibody worlds

Zero-mass scene bodies move with the linear and angular velocity given to
`ArticulatedWorld.set_scene_body_kinematic_motion(body, linear, angular)`. On `ArticulatedBatch`, pass the environment index first.
Passing `None` for both velocities stops the body and makes it static again. Passing `None` for only one, non-finite velocities,
out-of-range indices, and bodies with positive mass are errors. The velocity is transferred to contact partners,
but the body's own velocity isn't changed by contact impulses or gravity.
Read the current mode from `scene_body_state(...).kinematic`. It's `True` when prescribed motion is enabled, even with both
velocities at zero, and `False` after returning to static with `None`.

The CPU `step`, the regular `step_gpu`, and `step_gpu_device_mass` are supported.
The device-mass path adds zero rows and columns for kinematic coordinates to the inverse mass matrix on the GPU.
The resident configuration of `ArticulatedWorld.step_gpu_resident` supports kinematic bodies with sphere, box, capsule, cylinder, cone, and convex colliders.
Bodies with no contact pairs are also integrated, through a GPU pose buffer that's independent of the contact rows.
Kinematic boxes and capsules can contact link shapes of type sphere, capsule, box, cylinder, cone, and convex.
Kinematic cylinders and cones generate reserved pairs for the same contact partners. Vulkan / DX12 World/batch tests have confirmed velocity transfer to all partners through translation, and continued operation after static pose updates. Stopping against spheres, environment separation, and Python execution are also verified. Verification of contacts involving rotation and of long-run stability is ongoing.
Face contacts between a convex shape and a capsule, when both are dynamic, when one is static, and when a world capsule follows prescribed motion, clip the capsule axis to the convex face and generate up to 2 points,
then solve the normal reactions of both points together. Rounded ends, and contacts where the clipped axis collapses to a single point, use one point.
Other kinematic shapes and moving-anchor constraints are still unsupported and are rejected at build time.
[articulated_kinematic.py](examples/articulated_kinematic.py) uses the generated Python bindings to run
movement, stopping, argument validation, and environment separation for single worlds and batches.

# Resident GPU sessions for multibody worlds

`ArticulatedWorld.step_gpu_resident(timestep, torques, steps)` keeps GPU batches for the robot and scene,
runs fixed-timestep steps, and synchronizes the world state before returning.
Consecutive calls with the same timestep reuse the batch. A different timestep rebuilds it from the current host state.
The origins and orientations of kinematic spheres, boxes, capsules, cylinders, cones, and convex shapes are also reflected in `scene_body_state`.

After changing geometry, materials, solver settings, state, or prescribed motion,
call `reset_gpu_resident()` before resuming.
Externally edited state or kinematic pose/velocity produces an error instead of being overwritten with stale device state.
If shapes, materials, solver settings, or static body poses differ from the saved configuration, that's also an error before GPU submission.
If only the poses of static spheres, boxes, or convex shapes changed, `update_gpu_resident_static_scene_poses()` applies them to the existing session. It works on both single worlds and batches. Shapes, materials, generalized state, and kinematic body state must match the saved configuration. Rebuild for other static shapes.
Switching to a CPU, regular GPU, or device-mass step discards the resident session.
A session whose GPU execution failed can't be reused and needs a reset.
After a resident step has initialized the session, `enable_gpu_resident_scene_sleep()` enables automatic sleep for scene bodies.
Read the synchronized state with `scene_body_is_sleeping(body)` on a single world, or `scene_body_is_sleeping(environment, body)` on a batch.
Robot coordinates aren't put to sleep. If the session is discarded by a reset, a timestep change, or a switch to a regular step, enable sleep again after it's reinitialized.
`ArticulatedBatch.step_gpu_resident(timestep, torques, steps)` packs environments with different DOF counts into one GPU batch. `torques` is a nested list in environment order. After editing settings or state, rebuild with the batch's `reset_gpu_resident()`. Connecting this to the HS backend is planned for later.

[articulated_resident.py](examples/articulated_resident.py) checks consecutive steps with kinematic spheres, boxes, capsules, cylinders, and cones,
argument validation, errors after prescribed motion changes, rebuilding after a reset, scene sleep, and static pose updates.
It also runs velocity transfer to dynamic convex scene bodies, pose sync for a compound body with both a sphere and a box,
and a batch with different DOF counts that mixes all of these.

# GPU mass matrix solve for multibody worlds

`ArticulatedWorld.generalized_acceleration_gpu(torques)` and
`ArticulatedBatch.generalized_accelerations_gpu(torques_per_environment)` compute link poses, Jacobians, gravity, motors,
link external forces, and velocity-dependent terms on the CPU from the current joint state, then assemble the mass matrix on the GPU
from each link's mass, inertia, and Jacobian and solve for the generalized acceleration.
Batches combine environments with different degrees of freedom into one GPU matrix assembly and solve. This call
doesn't advance state or solve contacts. Results come from an f32 GPU computation and are read back to the CPU.
In Rust, `GpuArticulatedMassAssemblyBatch::solution_buffer()` can feed a later GPU pass
without a readback. Up to 128 generalized degrees of freedom are currently accepted, and singular or ill-conditioned matrices
are errors. This doesn't mean the full articulated dynamics of `step_gpu` are now GPU-resident.

In Rust, `submit_with_inverse()` produces the inverse mass matrix in a GPU buffer in addition to the accelerations.
`inverse_buffer()` and `inverse_ranges()` are the entry points for handing it to a later contact constraint pass,
and the former returns `None` until the inverse pass is first encoded.
`readback_inverse()` can be used to compare against CPU reference values. A regular
`submit()` that doesn't request the inverse does no extra work. In Rust, `ArticulatedMassSession::solve_with_inverse()`
returns accelerations and inverse mass matrices for several environments as an `ArticulatedMassSolution`, and reuses the GPU buffers on the next frame with the same shape.
`ArticulatedWorld.step_gpu` doesn't use this inverse yet,
and still prepares the contact problem on the CPU.

[axial_angular_contact.py](examples/axial_angular_contact.py) verifies velocity transfer at the contact point from angular velocity, using cylinder / cone contacts with a sphere where the rotation origin, shape center, and contact point are offset.

[convex_prescribed.py](examples/convex_prescribed.py) verifies contacts between a convex shape with prescribed motion and spheres / capsules, session rebuilds on stop, and state sync for a batch that includes a resting environment and an environment with no contacts. It also checks, across two environments that add the box and convex shape in swapped order, that different velocities are transferred to a dynamic convex shape and that stepping continues after static pose updates.

`convex_prescribed.py` also uses a short free capsule and a free capsule longer than the convex face, and checks over 1 step and 100 steps that velocity transfers from the prescribed convex shape and that symmetric contact doesn't produce spurious angular velocity.

The same example also uses an asymmetric contact with the capsule's center of mass moved outside the support region, and compares translational and angular velocity against the frictionless analytic solution for a single active endpoint.

`ArticulatedWorld.step_gpu_resident_with_contacts` and `ArticulatedBatch.step_gpu_resident_with_contacts` return, for the final substep, the force and torque on each scene body plus the position, normal, force, impulse, and distance of each contact point. Order follows the original scene body order, and batches nest the results per environment. Positions and torques are referenced to the pose before integration, and the impulse is the contact force times the timestep. These aren't accumulated over multiple steps. Samples for static collision-only bodies are empty.

[convex_angular_contact.py](examples/convex_angular_contact.py) checks contact point velocity and impulse from a convex face onto a free sphere, in two environments where the rotation origin and shape center are offset. It also verifies that a normal force through the sphere's center of mass doesn't produce spurious angular velocity.

[convex_dynamic_sphere.py](examples/convex_dynamic_sphere.py) checks off-center contact between a free convex shape and a sphere in two environments. It varies the add order, mass, inertia, and collider offset, and compares both bodies' impulses, translational and angular velocities, and linear and angular momentum against the frictionless analytic solution.

[moving_anchors.py](examples/moving_anchors.py) uses kinematic scene bodies as anchors for point / fixed constraints on the resident GPU path. It checks translation across 3 environments, rotation of fixed / offset point constraints in a single World, a static geometry hot update partway through, rejection of stale motion, and session rebuilds after stopping. Environments with moving anchors can still hot update the pose of unrelated static bodies. Changing the pose of an anchor body itself from outside requires a session rebuild.

[moving_indexed_geometry.py](examples/moving_indexed_geometry.py) is a resident GPU example where a mesh / polyline / heightfield moving at a prescribed velocity pushes a sphere. It checks consecutive steps in a 3-environment batch and a single World, body pose sync, and rejection of external velocity edits. It also compares contact velocity and position against analytic values in a rotating batch where the body origin and collider origin are offset. It confirms that the pose of a separate static indexed geometry can be updated during translation and rotation while the session continues. Contacts between a box and a rotating mesh / polyline / heightfield are also compared against analytic values for initial velocity, final position, and final velocity.
