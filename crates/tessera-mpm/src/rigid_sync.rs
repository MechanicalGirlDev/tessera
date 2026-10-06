//! Synchronization and CPU reaction transfer between Tessera rigid worlds and MPM.

use nalgebra::{DVector, Isometry3, Point3, UnitQuaternion, Vector2, Vector3};
use tessera_physics::articulated_world::{ArticulatedWorld, SceneBody, SceneCollider};
use tessera_physics::articulation::ArticulationPose;
#[cfg(feature = "gpu-rigid-coupling")]
use tessera_physics::gpu_rigid_shape::{GpuRigidShape, convex_face_normals};
#[cfg(feature = "gpu-rigid-coupling")]
use tessera_physics::gpu_rigid_sphere_world::GpuRigidSphereWorld;
use tessera_physics::mesh::{PolylineGeometry, TRIANGLE_HALF_THICKNESS, TriangleMeshGeometry};
use tessera_physics::sphere_world::SphereWorld;

use crate::obstacle::RigidObstacle;
use crate::world::{MpmError, MpmWorld, ObstacleReaction};
#[cfg(feature = "gpu-mpm")]
use crate::{GpuMpmError, GpuMpmResidentSession};

/// A rigid world cannot be represented as the current MPM obstacle set.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum RigidSyncError {
    /// A rigid pose, velocity, or collider property is invalid.
    #[error("invalid rigid world state")]
    InvalidRigidState,
    /// Reading the current resident rigid states failed.
    #[error("GPU rigid state readback failed: {0}")]
    GpuRigidReadback(String),
    /// A scene collider has no MPM obstacle representation yet.
    #[error("unsupported scene collider at body {body_index}, collider {collider_index}")]
    UnsupportedSceneCollider {
        /// Scene body index.
        body_index: usize,
        /// Collider index within the body.
        collider_index: usize,
    },
    /// An articulated collider has no MPM obstacle representation yet.
    #[error("unsupported {kind} collider at index {index}")]
    UnsupportedLinkCollider {
        /// Collider kind.
        kind: &'static str,
        /// Index in the corresponding shape collection.
        index: usize,
    },
    /// Expanded mesh obstacles exceeded available memory.
    #[error("rigid mesh obstacle capacity exceeded")]
    Capacity,
    /// The MPM world rejected the replacement obstacle set.
    #[error(transparent)]
    Mpm(#[from] MpmError),
    /// A coupled step must fit within one MPM CFL-limited substep.
    #[error("coupled MPM timestep exceeds one stable substep")]
    CouplingStepTooLarge,
}

/// Convert a GPU-resident rigid scene into one-way MPM obstacles.
///
/// All eight resident collider kinds are supported. Mesh triangles and polyline
/// segments become individual thin obstacles. This explicitly reads rigid
/// body states back to the CPU; call after the rigid step and before MPM step.
#[cfg(feature = "gpu-rigid-coupling")]
pub fn gpu_rigid_world_obstacles(
    world: &GpuRigidSphereWorld,
) -> Result<Vec<RigidObstacle>, RigidSyncError> {
    use nalgebra::Quaternion;

    let states = world
        .readback()
        .map_err(|error| RigidSyncError::GpuRigidReadback(error.to_string()))?;
    if states.len() != world.len() {
        return Err(RigidSyncError::InvalidRigidState);
    }
    let mut obstacles = Vec::new();
    let vec3 = |value: [f32; 3]| value.map(f64::from).into();
    for (index, state) in states.iter().enumerate() {
        if !state.is_valid() {
            return Err(RigidSyncError::InvalidRigidState);
        }
        let center = vec3([
            state.position_inverse_mass[0],
            state.position_inverse_mass[1],
            state.position_inverse_mass[2],
        ]);
        let linear = vec3([
            state.linear_velocity[0],
            state.linear_velocity[1],
            state.linear_velocity[2],
        ]);
        let angular = vec3([
            state.angular_velocity[0],
            state.angular_velocity[1],
            state.angular_velocity[2],
        ]);
        let [x, y, z, w] = state.orientation;
        let orientation = UnitQuaternion::new_normalize(Quaternion::new(
            f64::from(w),
            f64::from(x),
            f64::from(y),
            f64::from(z),
        ));
        let friction = world
            .body_material_override(index)
            .ok_or(RigidSyncError::InvalidRigidState)?
            .map_or(f64::from(world.config().solve.friction), |material| {
                material.friction
            });
        let shape = world
            .shape(index)
            .ok_or(RigidSyncError::InvalidRigidState)?;
        let extra = match &shape {
            GpuRigidShape::Polyline { segments, .. } => segments.len(),
            GpuRigidShape::TriangleMesh { triangles, .. } => triangles.len(),
            _ => 1,
        };
        obstacles
            .try_reserve(extra)
            .map_err(|_| RigidSyncError::Capacity)?;
        let mut append = |mut obstacle: RigidObstacle| -> Result<(), RigidSyncError> {
            obstacle.angular_velocity = angular;
            obstacle.linear_velocity = linear + angular.cross(&(obstacle.center - center));
            obstacle.friction = friction;
            if !obstacle.is_valid() {
                return Err(RigidSyncError::InvalidRigidState);
            }
            obstacles.push(obstacle);
            Ok(())
        };
        match shape {
            GpuRigidShape::Sphere { radius } => {
                let mut sphere = RigidObstacle::sphere(center, f64::from(radius));
                sphere.orientation = orientation;
                append(sphere)?;
            }
            GpuRigidShape::Box { half_extents } => {
                append(RigidObstacle::cuboid(
                    center,
                    vec3(half_extents),
                    orientation,
                ))?;
            }
            GpuRigidShape::Capsule {
                radius,
                half_length,
            } => append(RigidObstacle::capsule(
                center,
                f64::from(half_length),
                f64::from(radius),
                orientation,
            ))?,
            GpuRigidShape::Cylinder {
                radius,
                half_length,
            } => append(RigidObstacle::cylinder(
                center,
                f64::from(half_length),
                f64::from(radius),
                orientation,
            ))?,
            GpuRigidShape::Cone {
                radius,
                half_length,
            } => append(RigidObstacle::cone(
                center,
                f64::from(half_length),
                f64::from(radius),
                orientation,
            ))?,
            GpuRigidShape::Convex { vertices } => {
                let normals = convex_face_normals(&vertices);
                let vertices = vertices.into_iter().map(vec3).collect::<Vec<_>>();
                let normals = normals.into_iter().map(vec3).collect::<Vec<_>>();
                append(RigidObstacle::convex(
                    center,
                    &vertices,
                    &normals,
                    orientation,
                ))?;
            }
            GpuRigidShape::Polyline { vertices, segments } => {
                for [a, b] in segments {
                    let a = vertices
                        .get(a as usize)
                        .ok_or(RigidSyncError::InvalidRigidState)?;
                    let b = vertices
                        .get(b as usize)
                        .ok_or(RigidSyncError::InvalidRigidState)?;
                    let a = vec3(*a);
                    let b = vec3(*b);
                    let delta = b - a;
                    let length = delta.norm();
                    if !length.is_finite() || length <= 0.0 {
                        return Err(RigidSyncError::InvalidRigidState);
                    }
                    let axis = UnitQuaternion::rotation_between(&Vector3::z(), &(delta / length))
                        .unwrap_or_else(|| {
                            UnitQuaternion::from_axis_angle(
                                &Vector3::x_axis(),
                                core::f64::consts::PI,
                            )
                        });
                    let local_center = (a + b) / 2.0;
                    append(RigidObstacle::capsule(
                        center + orientation * local_center,
                        length / 2.0,
                        0.0,
                        orientation * axis,
                    ))?;
                }
            }
            GpuRigidShape::TriangleMesh {
                vertices,
                triangles,
            } => {
                for [a, b, c] in triangles {
                    let triangle = [a, b, c].map(|id| vertices.get(id as usize).copied().map(vec3));
                    let [Some(a), Some(b), Some(c)] = triangle else {
                        return Err(RigidSyncError::InvalidRigidState);
                    };
                    let centroid = (a + b + c) / 3.0;
                    append(RigidObstacle::triangle_prism(
                        center + orientation * centroid,
                        [a - centroid, b - centroid, c - centroid],
                        TRIANGLE_HALF_THICKNESS,
                        orientation,
                    ))?;
                }
            }
        }
    }
    if let Some(half_extent) = world.config().ground_half_extent {
        let mut ground = RigidObstacle::ground(
            Vector3::zeros(),
            Vector2::repeat(f64::from(half_extent)),
            UnitQuaternion::identity(),
        );
        ground.friction = world
            .ground_material_override()
            .map_or(f64::from(world.config().solve.friction), |material| {
                material.friction
            });
        if !ground.is_valid() {
            return Err(RigidSyncError::InvalidRigidState);
        }
        obstacles.push(ground);
    }
    Ok(obstacles)
}

/// A rigid world could not be synchronized with a resident GPU MPM session.
#[cfg(feature = "gpu-mpm")]
#[derive(Debug, thiserror::Error)]
pub enum ResidentRigidSyncError {
    /// The rigid world could not be converted to MPM obstacles.
    #[error(transparent)]
    Rigid(#[from] RigidSyncError),
    /// The GPU session rejected the replacement obstacle set.
    #[error(transparent)]
    Gpu(#[from] GpuMpmError),
}

/// Convert all sphere bodies and the finite ground plane to one-way obstacles.
/// Body order is stable, with the ground plane last.
pub fn sphere_world_obstacles(world: &SphereWorld) -> Result<Vec<RigidObstacle>, RigidSyncError> {
    let mut obstacles = Vec::with_capacity(world.bodies.len() + 1);
    for (index, body) in world.bodies.iter().enumerate() {
        let mut obstacle = RigidObstacle::sphere(body.center, body.radius);
        obstacle.linear_velocity = body.velocity;
        obstacle.friction = world
            .body_material(index)
            .ok_or(RigidSyncError::InvalidRigidState)?
            .friction;
        if !obstacle.is_valid() {
            return Err(RigidSyncError::InvalidRigidState);
        }
        obstacles.push(obstacle);
    }
    let extent = world.params().ground_half_extent;
    let mut ground = RigidObstacle::ground(
        Vector3::zeros(),
        Vector2::repeat(extent),
        UnitQuaternion::identity(),
    );
    ground.friction = world.ground_material().friction;
    if !ground.is_valid() {
        return Err(RigidSyncError::InvalidRigidState);
    }
    obstacles.push(ground);
    Ok(obstacles)
}

fn apply_sphere_reactions(
    world: &mut SphereWorld,
    reactions: &[ObstacleReaction],
) -> Result<(), RigidSyncError> {
    if reactions.len() != world.bodies.len() + 1 {
        return Err(RigidSyncError::InvalidRigidState);
    }
    let velocities = world
        .bodies
        .iter()
        .enumerate()
        .map(|(index, body)| {
            let delta = if body.mass > 0.0 {
                reactions[index].linear / body.mass
            } else {
                Vector3::zeros()
            };
            let velocity = body.velocity + delta;
            velocity
                .iter()
                .all(|component| component.is_finite())
                .then_some(velocity)
                .ok_or(RigidSyncError::InvalidRigidState)
        })
        .collect::<Result<Vec<_>, _>>()?;
    for (index, body) in world.bodies.iter_mut().enumerate() {
        if body.mass > 0.0 {
            body.velocity = velocities[index];
        }
    }
    for (index, reaction) in reactions.iter().take(world.bodies.len()).enumerate() {
        if world.bodies[index].mass > 0.0 && reaction.linear.norm_squared() > 0.0 {
            world
                .wake_body(index)
                .map_err(|_| RigidSyncError::InvalidRigidState)?;
        }
    }
    Ok(())
}

fn link_motion(
    world: &ArticulatedWorld,
    pose: &ArticulationPose,
    generalized_velocity: &DVector<f64>,
    link: usize,
    local_center: Vector3<f64>,
) -> Result<(Vector3<f64>, Vector3<f64>), RigidSyncError> {
    let (linear_jacobian, angular_jacobian) = world
        .articulation
        .generalized_point_jacobians(pose, link, local_center, world.floating)
        .map_err(|_| RigidSyncError::InvalidRigidState)?;
    let linear = linear_jacobian * generalized_velocity;
    let angular = angular_jacobian * generalized_velocity;
    Ok((
        Vector3::new(linear[0], linear[1], linear[2]),
        Vector3::new(angular[0], angular[1], angular[2]),
    ))
}

fn append_mesh_obstacles(
    obstacles: &mut Vec<RigidObstacle>,
    body: &SceneBody,
    frame: Isometry3<f64>,
    geometry: &TriangleMeshGeometry,
    friction: f64,
) -> Result<(), RigidSyncError> {
    obstacles
        .try_reserve(geometry.triangles().len())
        .map_err(|_| RigidSyncError::Capacity)?;
    for triangle in geometry.triangles() {
        let vertices = triangle.map(|id| geometry.vertices()[id as usize]);
        let centroid = (vertices[0] + vertices[1] + vertices[2]) / 3.0;
        let center = frame.transform_point(&Point3::from(centroid)).coords;
        let mut obstacle = RigidObstacle::triangle_prism(
            center,
            vertices.map(|vertex| vertex - centroid),
            TRIANGLE_HALF_THICKNESS,
            frame.rotation,
        );
        obstacle.linear_velocity = body.linear_velocity
            + body
                .angular_velocity
                .cross(&(obstacle.center - body.pose.translation.vector));
        obstacle.angular_velocity = body.angular_velocity;
        obstacle.friction = friction;
        if !obstacle.is_valid() {
            return Err(RigidSyncError::InvalidRigidState);
        }
        obstacles.push(obstacle);
    }
    Ok(())
}

fn append_polyline_obstacles(
    obstacles: &mut Vec<RigidObstacle>,
    body: &SceneBody,
    frame: Isometry3<f64>,
    geometry: &PolylineGeometry,
    friction: f64,
) -> Result<(), RigidSyncError> {
    obstacles
        .try_reserve(geometry.segments().len())
        .map_err(|_| RigidSyncError::Capacity)?;
    for index in 0..geometry.segments().len() {
        let [a, b] = geometry
            .segment_vertices(index)
            .ok_or(RigidSyncError::InvalidRigidState)?;
        let delta = b - a;
        let length = delta.norm();
        if !length.is_finite() || length <= 0.0 {
            return Err(RigidSyncError::InvalidRigidState);
        }
        let direction = delta / length;
        let segment_rotation = UnitQuaternion::rotation_between(&Vector3::z(), &direction)
            .unwrap_or_else(|| {
                UnitQuaternion::from_axis_angle(&Vector3::x_axis(), core::f64::consts::PI)
            });
        let center = frame.transform_point(&Point3::from((a + b) / 2.0)).coords;
        let mut obstacle =
            RigidObstacle::capsule(center, length / 2.0, 0.0, frame.rotation * segment_rotation);
        obstacle.linear_velocity = body.linear_velocity
            + body
                .angular_velocity
                .cross(&(center - body.pose.translation.vector));
        obstacle.angular_velocity = body.angular_velocity;
        obstacle.friction = friction;
        if !obstacle.is_valid() {
            return Err(RigidSyncError::InvalidRigidState);
        }
        obstacles.push(obstacle);
    }
    Ok(())
}

/// Convert the supported articulated and free-body colliders into MPM obstacles.
///
/// Link spheres, boxes, cylinders, and convex hulls, plus scene spheres, boxes,
/// capsules, cylinders, cones, convex hulls, polylines, triangle meshes, and
/// heightfields are supported. Unsupported future volume shapes produce an
/// error so that an incomplete obstacle set is never installed.
/// Ground-only contact points have no volume. The finite ground plane is last.
pub fn articulated_world_obstacles(
    world: &ArticulatedWorld,
) -> Result<Vec<RigidObstacle>, RigidSyncError> {
    let pose = world
        .articulation
        .pose(world.root_pose, world.positions.as_slice())
        .map_err(|_| RigidSyncError::InvalidRigidState)?;
    let generalized_velocity = DVector::from_iterator(
        world.velocities.len() + if world.floating { 6 } else { 0 },
        world
            .floating
            .then_some(
                world
                    .base_linear_velocity
                    .iter()
                    .chain(world.base_angular_velocity.iter())
                    .copied(),
            )
            .into_iter()
            .flatten()
            .chain(world.velocities.iter().copied()),
    );
    if generalized_velocity.iter().any(|value| !value.is_finite()) {
        return Err(RigidSyncError::InvalidRigidState);
    }

    let mut obstacles = Vec::with_capacity(
        world.colliders.len()
            + world.boxes.len()
            + world.cylinders.len()
            + world.scene_bodies.len()
            + 1,
    );
    for (index, sphere) in world.colliders.iter().enumerate() {
        let link_pose = pose
            .links
            .get(sphere.link)
            .ok_or(RigidSyncError::InvalidRigidState)?;
        let center = link_pose
            .transform_point(&Point3::from(sphere.center))
            .coords;
        let (linear, angular) = link_motion(
            world,
            &pose,
            &generalized_velocity,
            sphere.link,
            sphere.center,
        )?;
        let mut obstacle = RigidObstacle::sphere(center, sphere.radius);
        obstacle.linear_velocity = linear;
        obstacle.angular_velocity = angular;
        obstacle.friction = world
            .link_sphere_material(index)
            .ok_or(RigidSyncError::InvalidRigidState)?
            .friction;
        if !obstacle.is_valid() {
            return Err(RigidSyncError::InvalidRigidState);
        }
        obstacles.push(obstacle);
    }
    for (index, shape) in world.boxes.iter().enumerate() {
        let link_pose = pose
            .links
            .get(shape.link)
            .ok_or(RigidSyncError::InvalidRigidState)?;
        let frame = link_pose * shape.origin;
        let (linear, angular) = link_motion(
            world,
            &pose,
            &generalized_velocity,
            shape.link,
            shape.origin.translation.vector,
        )?;
        let mut obstacle =
            RigidObstacle::cuboid(frame.translation.vector, shape.half_extents, frame.rotation);
        obstacle.linear_velocity = linear;
        obstacle.angular_velocity = angular;
        obstacle.friction = world
            .link_box_material(index)
            .ok_or(RigidSyncError::InvalidRigidState)?
            .friction;
        if !obstacle.is_valid() {
            return Err(RigidSyncError::InvalidRigidState);
        }
        obstacles.push(obstacle);
    }
    for (index, shape) in world.cylinders.iter().enumerate() {
        let link_pose = pose
            .links
            .get(shape.link)
            .ok_or(RigidSyncError::InvalidRigidState)?;
        let frame = link_pose * shape.origin;
        let (linear, angular) = link_motion(
            world,
            &pose,
            &generalized_velocity,
            shape.link,
            shape.origin.translation.vector,
        )?;
        let mut obstacle = RigidObstacle::cylinder(
            frame.translation.vector,
            shape.half_height,
            shape.radius,
            frame.rotation,
        );
        obstacle.linear_velocity = linear;
        obstacle.angular_velocity = angular;
        obstacle.friction = world
            .link_cylinder_material(index)
            .ok_or(RigidSyncError::InvalidRigidState)?
            .friction;
        if !obstacle.is_valid() {
            return Err(RigidSyncError::InvalidRigidState);
        }
        obstacles.push(obstacle);
    }
    for (index, shape) in world.convex_shapes.iter().enumerate() {
        let link_pose = pose
            .links
            .get(shape.link)
            .ok_or(RigidSyncError::InvalidRigidState)?;
        let frame = link_pose * shape.origin;
        let (linear, angular) = link_motion(
            world,
            &pose,
            &generalized_velocity,
            shape.link,
            shape.origin.translation.vector,
        )?;
        let mut obstacle = RigidObstacle::convex(
            frame.translation.vector,
            &shape.geometry.vertices,
            &shape.geometry.face_normals,
            frame.rotation,
        );
        obstacle.linear_velocity = linear;
        obstacle.angular_velocity = angular;
        obstacle.friction = world
            .link_convex_material(index)
            .ok_or(RigidSyncError::InvalidRigidState)?
            .friction;
        if !obstacle.is_valid() {
            return Err(RigidSyncError::InvalidRigidState);
        }
        obstacles.push(obstacle);
    }

    for (body_index, body) in world.scene_bodies.iter().enumerate() {
        for (collider_index, collider) in body.colliders.iter().enumerate() {
            let friction = world
                .scene_collider_material(body_index, collider_index)
                .ok_or(RigidSyncError::InvalidRigidState)?
                .friction;
            let mesh = match collider {
                SceneCollider::TriangleMesh { origin, geometry } => {
                    Some((body.pose * origin, geometry))
                }
                SceneCollider::HeightField { origin, geometry } => {
                    Some((body.pose * origin, geometry.mesh()))
                }
                _ => None,
            };
            if let Some((frame, geometry)) = mesh {
                append_mesh_obstacles(&mut obstacles, body, frame, geometry, friction)?;
                continue;
            }
            if let SceneCollider::Polyline { origin, geometry } = collider {
                append_polyline_obstacles(
                    &mut obstacles,
                    body,
                    body.pose * origin,
                    geometry,
                    friction,
                )?;
                continue;
            }
            let mut obstacle = match collider {
                SceneCollider::Sphere { center, radius } => {
                    let center = body.pose.transform_point(&Point3::from(*center)).coords;
                    RigidObstacle::sphere(center, *radius)
                }
                SceneCollider::Box {
                    origin,
                    half_extents,
                } => {
                    let frame = body.pose * origin;
                    RigidObstacle::cuboid(frame.translation.vector, *half_extents, frame.rotation)
                }
                SceneCollider::Capsule {
                    origin,
                    half_height,
                    radius,
                } => {
                    let frame = body.pose * origin;
                    RigidObstacle::capsule(
                        frame.translation.vector,
                        *half_height,
                        *radius,
                        frame.rotation,
                    )
                }
                SceneCollider::Cylinder {
                    origin,
                    half_height,
                    radius,
                } => {
                    let frame = body.pose * origin;
                    RigidObstacle::cylinder(
                        frame.translation.vector,
                        *half_height,
                        *radius,
                        frame.rotation,
                    )
                }
                SceneCollider::Cone {
                    origin,
                    half_height,
                    radius,
                } => {
                    let frame = body.pose * origin;
                    RigidObstacle::cone(
                        frame.translation.vector,
                        *half_height,
                        *radius,
                        frame.rotation,
                    )
                }
                SceneCollider::Convex { origin, geometry } => {
                    let frame = body.pose * origin;
                    RigidObstacle::convex(
                        frame.translation.vector,
                        &geometry.vertices,
                        &geometry.face_normals,
                        frame.rotation,
                    )
                }
                _ => {
                    return Err(RigidSyncError::UnsupportedSceneCollider {
                        body_index,
                        collider_index,
                    });
                }
            };
            obstacle.linear_velocity = body.linear_velocity
                + body
                    .angular_velocity
                    .cross(&(obstacle.center - body.pose.translation.vector));
            obstacle.angular_velocity = body.angular_velocity;
            obstacle.friction = friction;
            if !obstacle.is_valid() {
                return Err(RigidSyncError::InvalidRigidState);
            }
            obstacles.push(obstacle);
        }
    }
    let mut ground = RigidObstacle::ground(
        Vector3::zeros(),
        Vector2::repeat(world.params().ground_half_extent),
        UnitQuaternion::identity(),
    );
    ground.friction = world.ground_material().friction;
    if !ground.is_valid() {
        return Err(RigidSyncError::InvalidRigidState);
    }
    obstacles.push(ground);
    Ok(obstacles)
}

impl MpmWorld {
    /// Refresh obstacles from current GPU rigid state after a rigid step.
    #[cfg(feature = "gpu-rigid-coupling")]
    pub fn sync_gpu_rigid_world(
        &mut self,
        world: &GpuRigidSphereWorld,
    ) -> Result<(), RigidSyncError> {
        self.set_obstacles(gpu_rigid_world_obstacles(world)?)?;
        Ok(())
    }

    /// Replace obstacles from the current sphere bodies, preserving the old set
    /// if conversion fails. Call again after changing the rigid world state.
    pub fn sync_sphere_world(&mut self, world: &SphereWorld) -> Result<(), RigidSyncError> {
        self.set_obstacles(sphere_world_obstacles(world)?)?;
        Ok(())
    }

    /// Transfer one MPM substep's particle reaction to dynamic sphere bodies.
    ///
    /// Call after advancing the rigid world to the same frame. Sphere bodies
    /// receive linear impulse immediately; their updated velocity participates
    /// in the next rigid step. The sphere model has no angular state, so angular
    /// reaction is returned for diagnostics but cannot rotate a sphere body.
    pub fn step_sphere_two_way(
        &mut self,
        world: &mut SphereWorld,
        dt: f64,
    ) -> Result<Vec<ObstacleReaction>, RigidSyncError> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(RigidSyncError::Mpm(MpmError::InvalidInput));
        }
        if dt > self.stable_timestep() {
            return Err(RigidSyncError::CouplingStepTooLarge);
        }
        self.sync_sphere_world(world)?;
        let reactions = self.step_with_obstacle_reactions(dt)?;
        apply_sphere_reactions(world, &reactions)?;
        Ok(reactions)
    }

    /// Transfer GPU MPM obstacle reactions to dynamic CPU sphere bodies.
    ///
    /// The MPM contact and reaction reduction run on the device. Particle and
    /// reaction readback is required before updating the CPU sphere velocities.
    #[cfg(feature = "gpu-mpm")]
    pub fn step_sphere_two_way_gpu(
        &mut self,
        world: &mut SphereWorld,
        gpu: &crate::GpuMpmTransfers,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        dt: f64,
    ) -> Result<Vec<ObstacleReaction>, ResidentRigidSyncError> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(RigidSyncError::Mpm(MpmError::InvalidInput).into());
        }
        if dt > self.stable_timestep() {
            return Err(RigidSyncError::CouplingStepTooLarge.into());
        }
        self.sync_sphere_world(world)?;
        let reactions = self.step_with_gpu_obstacle_reactions(gpu, device, queue, dt)?;
        apply_sphere_reactions(world, &reactions)?;
        Ok(reactions)
    }

    /// Replace obstacles from current articulated and scene collider state.
    /// Call again after advancing the rigid world.
    pub fn sync_articulated_world(
        &mut self,
        world: &ArticulatedWorld,
    ) -> Result<(), RigidSyncError> {
        self.set_obstacles(articulated_world_obstacles(world)?)?;
        Ok(())
    }
}

#[cfg(feature = "gpu-mpm")]
impl GpuMpmResidentSession<'_> {
    /// Refresh obstacles from current GPU rigid state before the MPM step.
    /// Rigid state is read back once; conversion or upload errors preserve the old set.
    #[cfg(feature = "gpu-rigid-coupling")]
    pub fn sync_gpu_rigid_world(
        &mut self,
        world: &GpuRigidSphereWorld,
    ) -> Result<(), ResidentRigidSyncError> {
        self.set_obstacles(gpu_rigid_world_obstacles(world)?)?;
        Ok(())
    }

    /// Refresh obstacles from the current sphere world before the next GPU step.
    /// Conversion and upload failures preserve the previous obstacle set.
    pub fn sync_sphere_world(&mut self, world: &SphereWorld) -> Result<(), ResidentRigidSyncError> {
        self.set_obstacles(sphere_world_obstacles(world)?)?;
        Ok(())
    }

    /// Refresh articulated and scene colliders before the next GPU step.
    /// Conversion and upload failures preserve the previous obstacle set.
    pub fn sync_articulated_world(
        &mut self,
        world: &ArticulatedWorld,
    ) -> Result<(), ResidentRigidSyncError> {
        self.set_obstacles(articulated_world_obstacles(world)?)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{Isometry3, Matrix3, Translation3, UnitQuaternion};
    use tessera_physics::articulated_world::{
        ArticulatedWorldParams, LinkBox, LinkConvex, LinkCylinder, LinkSphere, SceneBody,
    };
    use tessera_physics::articulation::{Articulation, JointKind, JointSpec, LinkSpec};
    use tessera_physics::convex::ConvexGeometry;
    use tessera_physics::material::ColliderMaterial;
    use tessera_physics::mesh::{HeightFieldGeometry, PolylineGeometry, TriangleMeshGeometry};
    use tessera_physics::sphere_world::{SphereBody, SphereWorldParams};

    use super::*;
    use crate::material::MaterialModel;
    use crate::world::MpmParams;

    #[cfg(feature = "gpu-rigid-coupling")]
    #[test]
    fn gpu_rigid_scene_syncs_all_eight_shapes_and_live_state() {
        use tessera_physics::gpu_contact_pipeline::GpuContactDevice;
        use tessera_physics::gpu_rigid_sphere_world::{
            GpuRigidPrimitiveWorld, GpuRigidSphereWorldConfig,
        };
        use tessera_physics::gpu_rigid_state::GpuRigidBodyState;

        fn check(context: &GpuContactDevice) {
            let cube = [-0.1, 0.1]
                .into_iter()
                .flat_map(|x| {
                    [-0.1, 0.1]
                        .into_iter()
                        .flat_map(move |y| [-0.1, 0.1].into_iter().map(move |z| [x, y, z]))
                })
                .collect();
            let shapes = vec![
                GpuRigidShape::Sphere { radius: 0.1 },
                GpuRigidShape::Box {
                    half_extents: [0.1; 3],
                },
                GpuRigidShape::Capsule {
                    radius: 0.1,
                    half_length: 0.1,
                },
                GpuRigidShape::Cylinder {
                    radius: 0.1,
                    half_length: 0.1,
                },
                GpuRigidShape::Cone {
                    radius: 0.1,
                    half_length: 0.1,
                },
                GpuRigidShape::Convex { vertices: cube },
                GpuRigidShape::Polyline {
                    vertices: vec![[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.2, 0.2, 0.0]],
                    segments: vec![[0, 1], [1, 2]],
                },
                GpuRigidShape::TriangleMesh {
                    vertices: vec![[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0]],
                    triangles: vec![[0, 1, 2]],
                },
            ];
            let states = (0..shapes.len())
                .map(|index| GpuRigidBodyState {
                    position_inverse_mass: [index as f32, 0.5, 0.5, 1.0],
                    orientation: [0.0, 0.0, 0.0, 1.0],
                    linear_velocity: [1.0, 0.0, 0.0, 0.0],
                    angular_velocity: [0.0, 0.0, 2.0, 0.0],
                    inverse_inertia_sleep: [1.0, 1.0, 1.0, 0.0],
                })
                .collect::<Vec<_>>();
            let mut rigid = GpuRigidPrimitiveWorld::new_primitives(
                context.device(),
                context.queue(),
                &states,
                &shapes,
                GpuRigidSphereWorldConfig {
                    gravity: [0.0; 3],
                    ground_half_extent: Some(10.0),
                    ..Default::default()
                },
            )
            .unwrap();
            rigid
                .set_body_material(0, ColliderMaterial::new(0.7, 0.0))
                .unwrap();
            let obstacles = gpu_rigid_world_obstacles(&rigid).unwrap();
            assert_eq!(obstacles.len(), 10);
            assert!(matches!(
                obstacles[0].shape,
                crate::ObstacleShape::Sphere { .. }
            ));
            assert!(matches!(
                obstacles[1].shape,
                crate::ObstacleShape::Box { .. }
            ));
            assert!(matches!(
                obstacles[2].shape,
                crate::ObstacleShape::Capsule { .. }
            ));
            assert!(matches!(
                obstacles[3].shape,
                crate::ObstacleShape::Cylinder { .. }
            ));
            assert!(matches!(
                obstacles[4].shape,
                crate::ObstacleShape::Cone { .. }
            ));
            assert!(matches!(
                obstacles[5].shape,
                crate::ObstacleShape::Convex { .. }
            ));
            assert!(matches!(
                obstacles[6].shape,
                crate::ObstacleShape::Capsule { .. }
            ));
            assert!(matches!(
                obstacles[7].shape,
                crate::ObstacleShape::Capsule { .. }
            ));
            assert!(matches!(
                obstacles[8].shape,
                crate::ObstacleShape::TrianglePrism { .. }
            ));
            assert!(matches!(
                obstacles[9].shape,
                crate::ObstacleShape::Ground { .. }
            ));
            assert!((obstacles[0].friction - 0.7).abs() < 1e-6);
            assert!((obstacles[7].linear_velocity.x - 0.8).abs() < 1e-6);
            let mut moved = states[0];
            moved.position_inverse_mass[0] = 0.25;
            rigid.write_body(0, moved).unwrap();
            let mut mpm = MpmWorld::new(Vec::new(), MpmParams::default()).unwrap();
            mpm.sync_gpu_rigid_world(&rigid).unwrap();
            assert!((mpm.obstacles[0].center.x - 0.25).abs() < 1e-6);

            let particle = crate::world::MpmParticle::new(
                Vector3::new(0.45, 0.5, 0.5),
                0.03,
                1_000.0,
                MaterialModel::elastic(1_000.0, 0.2),
            );
            let world = MpmWorld::new(vec![particle], MpmParams::default()).unwrap();
            let transfers = crate::GpuMpmTransfers::new(context.device());
            let mut session = GpuMpmResidentSession::new(
                &transfers,
                context.device(),
                context.queue(),
                world,
                0.0001,
            )
            .unwrap();
            session.sync_gpu_rigid_world(&rigid).unwrap();
            assert_eq!(session.world().obstacles.len(), 10);
            session.step(1).unwrap();
            assert!(
                session.world().particles[0]
                    .position
                    .iter()
                    .all(|x| x.is_finite())
            );
        }
        let mut tested = 0;
        for backend in [wgpu::Backends::VULKAN, wgpu::Backends::DX12] {
            let Ok(context) = GpuContactDevice::new_with_backends(backend) else {
                continue;
            };
            tested += 1;
            check(&context);
        }
        assert!(tested > 0);
    }

    fn root_articulation() -> Articulation {
        Articulation::new(
            vec![LinkSpec {
                mass: 1.0,
                center_of_mass: Vector3::zeros(),
                inertia: Matrix3::identity(),
            }],
            vec![],
            0,
        )
        .unwrap()
    }

    fn tetrahedron() -> ConvexGeometry {
        ConvexGeometry::new(
            vec![
                Vector3::zeros(),
                Vector3::x() * 0.2,
                Vector3::y() * 0.2,
                Vector3::z() * 0.2,
            ],
            vec![
                -Vector3::x(),
                -Vector3::y(),
                -Vector3::z(),
                Vector3::repeat(1.0).normalize(),
            ],
            vec![Vector3::x(), Vector3::y(), Vector3::z()],
        )
        .unwrap()
    }

    #[test]
    fn link_and_scene_convex_sync_preserves_pose_motion_and_material() {
        let mut rigid = ArticulatedWorld::new(
            root_articulation(),
            Isometry3::identity(),
            vec![],
            ArticulatedWorldParams::default(),
        )
        .unwrap();
        rigid
            .set_convex_shapes(vec![LinkConvex {
                link: 0,
                origin: Isometry3::translation(0.3, 0.4, 0.5),
                geometry: tetrahedron(),
            }])
            .unwrap();
        rigid
            .set_link_convex_material(0, ColliderMaterial::new(0.4, 0.0))
            .unwrap();
        let mut scene = SceneBody::new(
            Isometry3::translation(0.1, 0.2, 0.3),
            0.0,
            Matrix3::zeros(),
            vec![SceneCollider::Convex {
                origin: Isometry3::translation(0.2, 0.2, 0.2),
                geometry: tetrahedron(),
            }],
        )
        .unwrap();
        scene.linear_velocity = Vector3::x();
        scene.angular_velocity = Vector3::z() * 2.0;
        let body_index = rigid.add_scene_body(scene);
        rigid
            .set_scene_collider_material(body_index, 0, ColliderMaterial::new(0.7, 0.0))
            .unwrap();
        let obstacles = articulated_world_obstacles(&rigid).unwrap();
        assert_eq!(obstacles.len(), 3);
        assert!(matches!(
            obstacles[0].shape,
            crate::ObstacleShape::Convex { .. }
        ));
        assert!((obstacles[0].center - Vector3::new(0.3, 0.4, 0.5)).norm() < 1e-12);
        assert_eq!(obstacles[0].friction, 0.4);
        assert!(matches!(
            obstacles[1].shape,
            crate::ObstacleShape::Convex { .. }
        ));
        assert!((obstacles[1].center - Vector3::new(0.3, 0.4, 0.5)).norm() < 1e-12);
        assert!((obstacles[1].linear_velocity - Vector3::new(0.6, 0.4, 0.0)).norm() < 1e-12);
        assert_eq!(obstacles[1].friction, 0.7);
    }

    #[test]
    fn sphere_world_sync_tracks_motion_and_contact_material() {
        let mut rigid = SphereWorld::new(
            vec![SphereBody {
                center: Vector3::new(0.5, 0.5, 0.5),
                velocity: Vector3::x(),
                radius: 0.1,
                mass: 1.0,
            }],
            SphereWorldParams::default(),
        )
        .unwrap();
        rigid
            .set_body_material(0, ColliderMaterial::new(0.7, 0.0))
            .unwrap();
        let particle = crate::world::MpmParticle::new(
            Vector3::new(0.62, 0.5, 0.5),
            0.03,
            1000.0,
            MaterialModel::LinearElastic {
                young_modulus: 1000.0,
                poisson_ratio: 0.2,
            },
        );
        let mut mpm = MpmWorld::new(
            vec![particle],
            MpmParams {
                gravity: Vector3::zeros(),
                ..MpmParams::default()
            },
        )
        .unwrap();
        mpm.sync_sphere_world(&rigid).unwrap();
        assert_eq!(mpm.obstacles.len(), 2);
        assert!(matches!(
            mpm.obstacles[1].shape,
            crate::ObstacleShape::Ground { .. }
        ));
        assert_eq!(mpm.obstacles[0].linear_velocity, Vector3::x());
        assert_eq!(mpm.obstacles[0].friction, 0.7);
        mpm.step(0.001).unwrap();
        assert!(mpm.particles[0].velocity.x > 0.0);
        assert_eq!(rigid.bodies[0].velocity, Vector3::x());

        rigid.bodies[0].center.x += 0.2;
        rigid.bodies[0].velocity.x = 2.0;
        mpm.sync_sphere_world(&rigid).unwrap();
        assert!((mpm.obstacles[0].center.x - 0.7).abs() < 1e-12);
        assert_eq!(mpm.obstacles[0].linear_velocity.x, 2.0);
    }

    #[test]
    fn sphere_two_way_step_applies_particle_reaction_and_rejects_large_step() {
        let mut rigid = SphereWorld::new(
            vec![SphereBody {
                center: Vector3::new(0.5, 0.5, 0.5),
                velocity: Vector3::zeros(),
                radius: 0.1,
                mass: 1.0,
            }],
            SphereWorldParams::default(),
        )
        .unwrap();
        let mut particle = crate::world::MpmParticle::new(
            Vector3::new(0.62, 0.5, 0.5),
            0.03,
            1000.0,
            MaterialModel::elastic(1000.0, 0.2),
        );
        particle.velocity.x = -1.0;
        let initial_momentum = particle.velocity * particle.mass;
        let mut mpm = MpmWorld::new(
            vec![particle],
            MpmParams {
                gravity: Vector3::zeros(),
                ..MpmParams::default()
            },
        )
        .unwrap();
        assert_eq!(
            mpm.step_sphere_two_way(&mut rigid, 0.01).unwrap_err(),
            RigidSyncError::CouplingStepTooLarge
        );
        assert_eq!(mpm.substeps, 0);
        let reactions = mpm.step_sphere_two_way(&mut rigid, 0.001).unwrap();
        assert_eq!(reactions.len(), 2);
        assert!(reactions[0].linear.x < 0.0);
        assert!((rigid.bodies[0].velocity - reactions[0].linear).norm() < 1e-10);
        assert_eq!(reactions[1], ObstacleReaction::default());
        assert_eq!(mpm.substeps, 1);
        let final_momentum = rigid.bodies[0].velocity * rigid.bodies[0].mass
            + mpm.particles[0].velocity * mpm.particles[0].mass;
        assert!((final_momentum - initial_momentum).norm() < 1e-10);
    }

    #[cfg(feature = "gpu-mpm")]
    #[tokio::test]
    async fn gpu_sphere_two_way_matches_cpu_reaction() {
        let instance = wgpu::Instance::default();
        let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
        else {
            return;
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let body = SphereBody {
            center: Vector3::new(0.5, 0.5, 0.5),
            velocity: Vector3::zeros(),
            radius: 0.1,
            mass: 1.0,
        };
        let mut cpu_rigid =
            SphereWorld::new(vec![body.clone()], SphereWorldParams::default()).unwrap();
        let mut gpu_rigid = SphereWorld::new(vec![body], SphereWorldParams::default()).unwrap();
        let mut particle = crate::world::MpmParticle::new(
            Vector3::new(0.62, 0.5, 0.5),
            0.03,
            1000.0,
            MaterialModel::elastic(1000.0, 0.2),
        );
        particle.velocity.x = -1.0;
        let params = MpmParams {
            gravity: Vector3::zeros(),
            ..MpmParams::default()
        };
        let mut cpu_mpm = MpmWorld::new(vec![particle.clone()], params.clone()).unwrap();
        let mut gpu_mpm = MpmWorld::new(vec![particle], params).unwrap();
        let expected = cpu_mpm.step_sphere_two_way(&mut cpu_rigid, 0.001).unwrap();
        let actual = gpu_mpm
            .step_sphere_two_way_gpu(
                &mut gpu_rigid,
                &crate::GpuMpmTransfers::new(&device),
                &device,
                &queue,
                0.001,
            )
            .unwrap();
        assert!((actual[0].linear - expected[0].linear).norm() < 1e-4);
        assert!((gpu_rigid.bodies[0].velocity - cpu_rigid.bodies[0].velocity).norm() < 1e-4);
        assert!((gpu_mpm.particles[0].velocity - cpu_mpm.particles[0].velocity).norm() < 1e-4);
    }

    #[test]
    fn sphere_world_sync_includes_finite_ground_material() {
        let mut rigid = SphereWorld::new(
            vec![],
            SphereWorldParams {
                ground_half_extent: 1.25,
                ..SphereWorldParams::default()
            },
        )
        .unwrap();
        rigid
            .set_ground_material(ColliderMaterial::new(0.9, 0.0))
            .unwrap();
        let mut mpm = MpmWorld::new(vec![], MpmParams::default()).unwrap();
        mpm.sync_sphere_world(&rigid).unwrap();
        assert_eq!(mpm.obstacles.len(), 1);
        assert_eq!(mpm.obstacles[0].friction, 0.9);
        assert_eq!(
            mpm.obstacles[0].shape,
            crate::ObstacleShape::Ground {
                half_extents: Vector2::repeat(1.25),
            }
        );
    }

    #[cfg(feature = "gpu-mpm")]
    #[tokio::test]
    async fn synced_rigid_sphere_matches_cpu_on_gpu_transfer_path() {
        let instance = wgpu::Instance::default();
        let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
        else {
            eprintln!("No WebGPU adapter available for rigid MPM sync test");
            return;
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let rigid = SphereWorld::new(
            vec![SphereBody {
                center: Vector3::repeat(0.5),
                velocity: Vector3::x(),
                radius: 0.1,
                mass: 1.0,
            }],
            SphereWorldParams::default(),
        )
        .unwrap();
        let particle = crate::world::MpmParticle::new(
            Vector3::new(0.62, 0.5, 0.5),
            0.03,
            1000.0,
            MaterialModel::elastic(1000.0, 0.2),
        );
        let params = MpmParams {
            gravity: Vector3::zeros(),
            ..MpmParams::default()
        };
        let mut cpu = MpmWorld::new(vec![particle.clone()], params.clone()).unwrap();
        let mut gpu = MpmWorld::new(vec![particle], params).unwrap();
        cpu.sync_sphere_world(&rigid).unwrap();
        gpu.sync_sphere_world(&rigid).unwrap();
        cpu.step(0.002).unwrap();
        gpu.step_with_gpu_transfers(
            &crate::gpu::GpuMpmTransfers::new(&device),
            &device,
            &queue,
            0.002,
        )
        .unwrap();
        assert!((cpu.particles[0].position - gpu.particles[0].position).norm() < 1e-5);
        assert!((cpu.particles[0].velocity - gpu.particles[0].velocity).norm() < 1e-4);
        assert_eq!(cpu.obstacles, gpu.obstacles);
    }

    #[cfg(feature = "gpu-mpm")]
    #[tokio::test]
    async fn resident_session_tracks_sphere_and_articulated_world_motion() {
        let instance = wgpu::Instance::default();
        let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
        else {
            eprintln!("No WebGPU adapter available for resident rigid sync test");
            return;
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let particle = crate::world::MpmParticle::new(
            Vector3::new(0.62, 0.5, 0.5),
            0.03,
            1_000.0,
            MaterialModel::elastic(1_000.0, 0.2),
        );
        let params = MpmParams {
            gravity: Vector3::zeros(),
            bounds: Some(crate::WorldBounds {
                min: Vector3::zeros(),
                max: Vector3::repeat(1.0),
            }),
            ..MpmParams::default()
        };
        let world = MpmWorld::new(vec![particle], params).unwrap();
        let transfers = crate::GpuMpmTransfers::new(&device);
        let mut session =
            GpuMpmResidentSession::new(&transfers, &device, &queue, world.clone(), 0.0001).unwrap();
        let mut reference = world;
        let mut rigid = SphereWorld::new(
            vec![SphereBody {
                center: Vector3::repeat(0.5),
                velocity: Vector3::x(),
                radius: 0.1,
                mass: 1.0,
            }],
            SphereWorldParams::default(),
        )
        .unwrap();

        for stage in 0..2 {
            if stage == 1 {
                rigid.bodies[0].center.x += 0.02;
                rigid.bodies[0].velocity.x = 2.0;
            }
            session.sync_sphere_world(&rigid).unwrap();
            reference.sync_sphere_world(&rigid).unwrap();
            assert_eq!(session.world().obstacles, reference.obstacles);
            session.step(1).unwrap();
            reference
                .step_gpu_resident_fixed_substeps(&transfers, &device, &queue, 0.0001, 1)
                .unwrap();
            assert!(
                (session.world().particles[0].position - reference.particles[0].position).norm()
                    < 1e-6
            );
            assert!(
                (session.world().particles[0].velocity - reference.particles[0].velocity).norm()
                    < 1e-6
            );
        }
        assert!(session.world().particles[0].velocity.x > 0.0);
        let previous = session.world().obstacles.clone();
        rigid.bodies[0].radius = f64::NAN;
        assert!(matches!(
            session.sync_sphere_world(&rigid),
            Err(ResidentRigidSyncError::Rigid(
                RigidSyncError::InvalidRigidState
            ))
        ));
        assert_eq!(session.world().obstacles, previous);
        rigid.bodies[0].radius = 1e100;
        assert!(matches!(
            session.sync_sphere_world(&rigid),
            Err(ResidentRigidSyncError::Gpu(GpuMpmError::InvalidInput))
        ));
        assert_eq!(session.world().obstacles, previous);

        let mut articulated = ArticulatedWorld::new(
            root_articulation(),
            Isometry3::identity(),
            vec![],
            ArticulatedWorldParams::default(),
        )
        .unwrap();
        let mut scene = SceneBody::new(
            Isometry3::translation(0.5, 0.5, 0.5),
            0.0,
            Matrix3::zeros(),
            vec![SceneCollider::Sphere {
                center: Vector3::zeros(),
                radius: 0.1,
            }],
        )
        .unwrap();
        scene.linear_velocity = Vector3::x();
        let scene_index = articulated.add_scene_body(scene);
        for stage in 0..2 {
            if stage == 1 {
                articulated.scene_bodies[scene_index]
                    .pose
                    .translation
                    .vector
                    .x += 0.02;
                articulated.scene_bodies[scene_index].linear_velocity.x = 2.0;
            }
            session.sync_articulated_world(&articulated).unwrap();
            reference.sync_articulated_world(&articulated).unwrap();
            assert_eq!(session.world().obstacles, reference.obstacles);
            session.step(1).unwrap();
            reference
                .step_gpu_resident_fixed_substeps(&transfers, &device, &queue, 0.0001, 1)
                .unwrap();
            assert!(
                (session.world().particles[0].position - reference.particles[0].position).norm()
                    < 1e-6
            );
            assert!(
                (session.world().particles[0].velocity - reference.particles[0].velocity).norm()
                    < 1e-6
            );
        }
    }

    #[test]
    fn scene_box_sync_uses_local_pose_and_body_origin_velocity() {
        let mut rigid = ArticulatedWorld::new(
            root_articulation(),
            Isometry3::identity(),
            vec![],
            ArticulatedWorldParams::default(),
        )
        .unwrap();
        let rotation =
            UnitQuaternion::from_axis_angle(&Vector3::z_axis(), core::f64::consts::FRAC_PI_2);
        let mut scene = SceneBody::new(
            Isometry3::from_parts(Translation3::new(1.0, 2.0, 3.0), rotation),
            0.0,
            Matrix3::zeros(),
            vec![SceneCollider::Box {
                origin: Isometry3::translation(0.4, 0.0, 0.0),
                half_extents: Vector3::new(0.2, 0.3, 0.4),
            }],
        )
        .unwrap();
        scene.linear_velocity = Vector3::x();
        scene.angular_velocity = Vector3::z() * 2.0;
        let body_index = rigid.add_scene_body(scene);
        rigid
            .set_scene_collider_material(body_index, 0, ColliderMaterial::new(0.6, 0.0))
            .unwrap();
        let obstacles = articulated_world_obstacles(&rigid).unwrap();
        assert_eq!(obstacles.len(), 2);
        assert!((obstacles[0].center - Vector3::new(1.0, 2.4, 3.0)).norm() < 1e-12);
        assert!((obstacles[0].linear_velocity - Vector3::new(0.2, 0.0, 0.0)).norm() < 1e-12);
        assert_eq!(obstacles[0].angular_velocity, Vector3::z() * 2.0);
        assert_eq!(obstacles[0].friction, 0.6);
        assert!((obstacles[0].orientation * Vector3::x() - Vector3::y()).norm() < 1e-12);
    }

    #[test]
    fn link_colliders_follow_revolute_pose_and_velocity() {
        let tree = Articulation::new(
            vec![
                LinkSpec {
                    mass: 0.0,
                    center_of_mass: Vector3::zeros(),
                    inertia: Matrix3::zeros(),
                },
                LinkSpec {
                    mass: 1.0,
                    center_of_mass: Vector3::zeros(),
                    inertia: Matrix3::identity(),
                },
            ],
            vec![JointSpec {
                parent: 0,
                child: 1,
                origin: Isometry3::identity(),
                kind: JointKind::Revolute,
                axis: Vector3::z(),
                limits: None,
            }],
            0,
        )
        .unwrap();
        let mut rigid = ArticulatedWorld::new(
            tree,
            Isometry3::identity(),
            vec![LinkSphere {
                link: 1,
                center: Vector3::x(),
                radius: 0.1,
            }],
            ArticulatedWorldParams::default(),
        )
        .unwrap();
        rigid
            .set_boxes(vec![LinkBox {
                link: 1,
                origin: Isometry3::translation(0.5, 0.0, 0.0),
                half_extents: Vector3::repeat(0.1),
            }])
            .unwrap();
        rigid.positions[0] = core::f64::consts::FRAC_PI_2;
        rigid.velocities[0] = 2.0;
        let obstacles = articulated_world_obstacles(&rigid).unwrap();
        assert_eq!(obstacles.len(), 3);
        assert!((obstacles[0].center - Vector3::y()).norm() < 1e-12);
        assert!((obstacles[0].linear_velocity + Vector3::x() * 2.0).norm() < 1e-12);
        assert_eq!(obstacles[0].angular_velocity, Vector3::z() * 2.0);
        assert!((obstacles[1].center - Vector3::y() * 0.5).norm() < 1e-12);
        assert!((obstacles[1].linear_velocity + Vector3::x()).norm() < 1e-12);
    }

    #[test]
    fn floating_root_rotation_contributes_to_link_center_velocity() {
        let mut rigid = ArticulatedWorld::new_floating(
            root_articulation(),
            Isometry3::identity(),
            vec![LinkSphere {
                link: 0,
                center: Vector3::x(),
                radius: 0.1,
            }],
            ArticulatedWorldParams::default(),
        )
        .unwrap();
        rigid.base_linear_velocity = Vector3::y();
        rigid.base_angular_velocity = Vector3::z() * 2.0;
        let obstacles = articulated_world_obstacles(&rigid).unwrap();
        assert_eq!(obstacles[0].center, Vector3::x());
        assert!((obstacles[0].linear_velocity - Vector3::y() * 3.0).norm() < 1e-12);
        assert_eq!(obstacles[0].angular_velocity, Vector3::z() * 2.0);
    }

    #[test]
    fn scene_primitives_and_link_cylinder_sync_without_geometry_loss() {
        let mut rigid = ArticulatedWorld::new_floating(
            root_articulation(),
            Isometry3::identity(),
            vec![],
            ArticulatedWorldParams::default(),
        )
        .unwrap();
        rigid
            .set_cylinders(vec![LinkCylinder {
                link: 0,
                origin: Isometry3::translation(0.5, 0.0, 0.0),
                half_height: 0.3,
                radius: 0.2,
            }])
            .unwrap();
        rigid
            .set_link_cylinder_material(0, ColliderMaterial::new(0.8, 0.0))
            .unwrap();
        rigid.base_linear_velocity = Vector3::y();
        rigid.base_angular_velocity = Vector3::z() * 2.0;
        let mut scene = SceneBody::new(
            Isometry3::identity(),
            0.0,
            Matrix3::zeros(),
            vec![
                SceneCollider::Capsule {
                    origin: Isometry3::translation(1.0, 0.0, 0.0),
                    half_height: 0.4,
                    radius: 0.1,
                },
                SceneCollider::Cylinder {
                    origin: Isometry3::translation(2.0, 0.0, 0.0),
                    half_height: 0.5,
                    radius: 0.2,
                },
                SceneCollider::Cone {
                    origin: Isometry3::translation(3.0, 0.0, 0.0),
                    half_height: 0.6,
                    radius: 0.3,
                },
            ],
        )
        .unwrap();
        scene.angular_velocity = Vector3::z();
        let _body_index = rigid.add_scene_body(scene);
        let obstacles = articulated_world_obstacles(&rigid).unwrap();
        assert_eq!(obstacles.len(), 5);
        assert_eq!(
            obstacles[0].shape,
            crate::ObstacleShape::Cylinder {
                half_height: 0.3,
                radius: 0.2,
            }
        );
        assert_eq!(obstacles[0].friction, 0.8);
        assert!((obstacles[0].linear_velocity - Vector3::y() * 2.0).norm() < 1e-12);
        assert!(matches!(
            obstacles[1].shape,
            crate::ObstacleShape::Capsule { .. }
        ));
        assert!(matches!(
            obstacles[2].shape,
            crate::ObstacleShape::Cylinder { .. }
        ));
        assert!(matches!(
            obstacles[3].shape,
            crate::ObstacleShape::Cone { .. }
        ));
        assert!((obstacles[3].linear_velocity - Vector3::y() * 3.0).norm() < 1e-12);
    }

    #[test]
    fn scene_mesh_and_heightfield_expand_into_moving_triangle_prisms() {
        let mut rigid = ArticulatedWorld::new(
            root_articulation(),
            Isometry3::identity(),
            vec![],
            ArticulatedWorldParams::default(),
        )
        .unwrap();
        let mesh = TriangleMeshGeometry::new(
            vec![
                Vector3::new(0.4, 0.4, 0.5),
                Vector3::new(0.7, 0.4, 0.5),
                Vector3::new(0.4, 0.7, 0.5),
            ],
            vec![[0, 1, 2]],
        )
        .unwrap();
        let heightfield =
            HeightFieldGeometry::new(2, 2, vec![0.5; 4], Vector3::new(1.0, 1.0, 1.0)).unwrap();
        let mut scene = SceneBody::new(
            Isometry3::identity(),
            0.0,
            Matrix3::zeros(),
            vec![
                SceneCollider::TriangleMesh {
                    origin: Isometry3::identity(),
                    geometry: mesh,
                },
                SceneCollider::HeightField {
                    origin: Isometry3::identity(),
                    geometry: heightfield,
                },
            ],
        )
        .unwrap();
        scene.linear_velocity = Vector3::x();
        scene.angular_velocity = Vector3::z() * 2.0;
        let body_index = rigid.add_scene_body(scene);
        rigid
            .set_scene_collider_material(body_index, 0, ColliderMaterial::new(0.7, 0.0))
            .unwrap();
        let obstacles = articulated_world_obstacles(&rigid).unwrap();
        assert_eq!(obstacles.len(), 4);
        assert!(matches!(
            obstacles[0].shape,
            crate::ObstacleShape::TrianglePrism { .. }
        ));
        assert!((obstacles[0].center - Vector3::new(0.5, 0.5, 0.5)).norm() < 1e-12);
        assert!((obstacles[0].linear_velocity - Vector3::y()).norm() < 1e-12);
        assert_eq!(obstacles[0].friction, 0.7);
        assert!(
            obstacles[1..3].iter().all(|obstacle| matches!(
                obstacle.shape,
                crate::ObstacleShape::TrianglePrism { .. }
            ))
        );
        assert!(matches!(
            obstacles[3].shape,
            crate::ObstacleShape::Ground { .. }
        ));
    }

    #[test]
    fn scene_polyline_segments_preserve_zero_radius_and_motion() {
        let mut rigid = ArticulatedWorld::new(
            root_articulation(),
            Isometry3::identity(),
            vec![],
            ArticulatedWorldParams::default(),
        )
        .unwrap();
        let mut scene = SceneBody::new(
            Isometry3::identity(),
            0.0,
            Matrix3::zeros(),
            vec![SceneCollider::Polyline {
                origin: Isometry3::identity(),
                geometry: PolylineGeometry::new(
                    vec![
                        Vector3::new(0.3, 0.5, 0.5),
                        Vector3::new(0.7, 0.5, 0.5),
                        Vector3::new(0.7, 0.7, 0.5),
                    ],
                    vec![[0, 1], [1, 2]],
                )
                .unwrap(),
            }],
        )
        .unwrap();
        scene.linear_velocity = Vector3::x();
        scene.angular_velocity = Vector3::z() * 2.0;
        let body_index = rigid.add_scene_body(scene);
        rigid
            .set_scene_collider_material(body_index, 0, ColliderMaterial::new(0.6, 0.0))
            .unwrap();
        let obstacles = articulated_world_obstacles(&rigid).unwrap();
        assert_eq!(obstacles.len(), 3);
        assert!(matches!(
            obstacles[0].shape,
            crate::ObstacleShape::Capsule { radius: 0.0, .. }
        ));
        assert!((obstacles[0].surface(Vector3::new(0.5, 0.5, 0.53)).0 - 0.03).abs() < 1e-12);
        assert!((obstacles[0].linear_velocity - Vector3::y()).norm() < 1e-12);
        assert_eq!(obstacles[0].friction, 0.6);
        assert!((obstacles[1].surface(Vector3::new(0.7, 0.6, 0.53)).0 - 0.03).abs() < 1e-12);
    }

    #[cfg(feature = "gpu-mpm")]
    #[tokio::test]
    async fn synced_triangle_mesh_shared_edge_blocks_particles_on_gpu_transfer_path() {
        let instance = wgpu::Instance::default();
        let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
        else {
            eprintln!("No WebGPU adapter available for MPM mesh coupling test");
            return;
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        let mut rigid = ArticulatedWorld::new(
            root_articulation(),
            Isometry3::identity(),
            vec![],
            ArticulatedWorldParams::default(),
        )
        .unwrap();
        let scene = SceneBody::new(
            Isometry3::identity(),
            0.0,
            Matrix3::zeros(),
            vec![SceneCollider::TriangleMesh {
                origin: Isometry3::identity(),
                geometry: TriangleMeshGeometry::new(
                    vec![
                        Vector3::new(0.35, 0.35, 0.5),
                        Vector3::new(0.65, 0.35, 0.5),
                        Vector3::new(0.65, 0.65, 0.5),
                        Vector3::new(0.35, 0.65, 0.5),
                    ],
                    vec![[0, 1, 2], [0, 2, 3]],
                )
                .unwrap(),
            }],
        )
        .unwrap();
        let _body_index = rigid.add_scene_body(scene);
        let mut particle = crate::world::MpmParticle::new(
            Vector3::new(0.5, 0.5, 0.54),
            0.03,
            1_000.0,
            MaterialModel::elastic(1_000.0, 0.2),
        );
        particle.velocity.z = -10.0;
        let params = MpmParams {
            gravity: Vector3::zeros(),
            ..MpmParams::default()
        };
        let mut cpu = MpmWorld::new(vec![particle.clone()], params.clone()).unwrap();
        let mut gpu = MpmWorld::new(vec![particle], params).unwrap();
        cpu.sync_articulated_world(&rigid).unwrap();
        gpu.sync_articulated_world(&rigid).unwrap();
        assert_eq!(cpu.obstacles.len(), 3);
        cpu.step(0.002).unwrap();
        gpu.step_with_gpu_transfers(
            &crate::gpu::GpuMpmTransfers::new(&device),
            &device,
            &queue,
            0.002,
        )
        .unwrap();
        assert!(cpu.particles[0].position.z >= 0.5301 - 1e-7);
        assert!((cpu.particles[0].position - gpu.particles[0].position).norm() < 1e-5);
        assert!((cpu.particles[0].velocity - gpu.particles[0].velocity).norm() < 1e-4);
    }

    #[test]
    fn invalid_scene_shape_keeps_previous_obstacles() {
        let mut rigid = ArticulatedWorld::new(
            root_articulation(),
            Isometry3::identity(),
            vec![],
            ArticulatedWorldParams::default(),
        )
        .unwrap();
        let mut scene = SceneBody::new(
            Isometry3::identity(),
            0.0,
            Matrix3::zeros(),
            vec![
                SceneCollider::Sphere {
                    center: Vector3::zeros(),
                    radius: 1.0,
                },
                SceneCollider::Convex {
                    origin: Isometry3::identity(),
                    geometry: ConvexGeometry::new(
                        vec![Vector3::zeros(), Vector3::x(), Vector3::y(), Vector3::z()],
                        vec![
                            -Vector3::x(),
                            -Vector3::y(),
                            -Vector3::z(),
                            Vector3::repeat(1.0).normalize(),
                        ],
                        vec![Vector3::x(), Vector3::y(), Vector3::z()],
                    )
                    .unwrap(),
                },
            ],
        )
        .unwrap();
        if let SceneCollider::Convex { geometry, .. } = &mut scene.colliders[1] {
            geometry.face_normals[0].x = f64::NAN;
        }
        let _index = rigid.add_scene_body(scene);
        let previous = RigidObstacle::sphere(Vector3::new(4.0, 0.0, 0.0), 0.25);
        let mut mpm = MpmWorld::new(vec![], MpmParams::default()).unwrap();
        mpm.set_obstacles(vec![previous.clone()]).unwrap();
        assert_eq!(
            mpm.sync_articulated_world(&rigid),
            Err(RigidSyncError::InvalidRigidState)
        );
        assert_eq!(mpm.obstacles, vec![previous]);
    }
}
