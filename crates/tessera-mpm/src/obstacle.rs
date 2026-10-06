//! One-way rigid obstacles for material points and grid velocities.

use std::sync::Arc;

use nalgebra::{UnitQuaternion, Vector3};

/// Collision shape of a rigid obstacle that pushes MPM particles.
#[allow(variant_size_differences)]
#[derive(Debug, Clone, PartialEq)]
pub enum ObstacleShape {
    /// Sphere centered at the obstacle pose.
    Sphere {
        /// Positive sphere radius.
        radius: f64,
    },
    /// Oriented box centered at the obstacle pose.
    Box {
        /// Positive half extents along the obstacle's local axes.
        half_extents: Vector3<f64>,
    },
    /// Z-axis capsule in the obstacle's local frame.
    Capsule {
        /// Nonnegative half length of the central segment.
        half_height: f64,
        /// Nonnegative spherical border radius; zero represents a line segment.
        radius: f64,
    },
    /// Z-axis circular cylinder in the obstacle's local frame.
    Cylinder {
        /// Positive half height.
        half_height: f64,
        /// Positive circular radius.
        radius: f64,
    },
    /// Z-axis circular cone with its base at local -Z and apex at +Z.
    Cone {
        /// Positive half height.
        half_height: f64,
        /// Positive base radius.
        radius: f64,
    },
    /// Finite, one-sided local XY ground plane with positive local Z normal.
    Ground {
        /// Positive half extents of the rectangle in local X and Y.
        half_extents: nalgebra::Vector2<f64>,
    },
    /// A thin triangular prism with two parallel triangular faces.
    TrianglePrism {
        /// Three non-collinear vertices in the obstacle's local frame.
        vertices: [Vector3<f64>; 3],
        /// Positive half thickness along the triangle normal.
        half_thickness: f64,
    },
    /// A convex polyhedron represented by outward supporting face planes.
    Convex {
        /// Local planes stored as [normal_x, normal_y, normal_z, offset].
        planes: Arc<[[f64; 4]]>,
        /// Conservative sphere radius around the local origin.
        bound_radius: f64,
    },
}

/// Velocity response at a rigid MPM boundary.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ObstacleBoundary {
    /// Remove incoming normal velocity and apply Coulomb friction.
    #[default]
    Slip,
    /// Match the moving obstacle's surface velocity while in contact.
    Stick,
    /// Remove incoming normal velocity while preserving tangential slip.
    Separate,
    /// Project penetrated positions without changing particle or grid velocity.
    NonReflecting,
}

/// A prescribed rigid pose and velocity used for one-way MPM contact.
#[derive(Debug, Clone, PartialEq)]
pub struct RigidObstacle {
    /// Collision shape.
    pub shape: ObstacleShape,
    /// World-space center.
    pub center: Vector3<f64>,
    /// Local-to-world orientation.
    pub orientation: UnitQuaternion<f64>,
    /// World-space linear velocity of the center.
    pub linear_velocity: Vector3<f64>,
    /// World-space angular velocity in radians per second.
    pub angular_velocity: Vector3<f64>,
    /// Nonnegative Coulomb friction coefficient.
    pub friction: f64,
    /// Velocity response applied at this obstacle's surface.
    pub boundary: ObstacleBoundary,
    /// CPIC transfer group for a thin triangle; triangles in one group share a side bit.
    pub cpic_group: Option<u8>,
}

impl RigidObstacle {
    /// Construct a stationary sphere obstacle.
    pub fn sphere(center: Vector3<f64>, radius: f64) -> Self {
        Self {
            shape: ObstacleShape::Sphere { radius },
            center,
            orientation: UnitQuaternion::identity(),
            linear_velocity: Vector3::zeros(),
            angular_velocity: Vector3::zeros(),
            friction: 0.0,
            boundary: ObstacleBoundary::default(),
            cpic_group: None,
        }
    }

    /// Construct a stationary oriented box obstacle.
    pub fn cuboid(
        center: Vector3<f64>,
        half_extents: Vector3<f64>,
        orientation: UnitQuaternion<f64>,
    ) -> Self {
        Self {
            shape: ObstacleShape::Box { half_extents },
            center,
            orientation,
            linear_velocity: Vector3::zeros(),
            angular_velocity: Vector3::zeros(),
            friction: 0.0,
            boundary: ObstacleBoundary::default(),
            cpic_group: None,
        }
    }

    /// Construct a stationary z-axis capsule obstacle.
    pub fn capsule(
        center: Vector3<f64>,
        half_height: f64,
        radius: f64,
        orientation: UnitQuaternion<f64>,
    ) -> Self {
        Self::oriented_primitive(
            center,
            orientation,
            ObstacleShape::Capsule {
                half_height,
                radius,
            },
        )
    }

    /// Construct a stationary z-axis circular cylinder obstacle.
    pub fn cylinder(
        center: Vector3<f64>,
        half_height: f64,
        radius: f64,
        orientation: UnitQuaternion<f64>,
    ) -> Self {
        Self::oriented_primitive(
            center,
            orientation,
            ObstacleShape::Cylinder {
                half_height,
                radius,
            },
        )
    }

    /// Construct a stationary z-axis circular cone obstacle.
    pub fn cone(
        center: Vector3<f64>,
        half_height: f64,
        radius: f64,
        orientation: UnitQuaternion<f64>,
    ) -> Self {
        Self::oriented_primitive(
            center,
            orientation,
            ObstacleShape::Cone {
                half_height,
                radius,
            },
        )
    }

    /// Construct a stationary finite ground plane centered on the supplied pose.
    pub fn ground(
        center: Vector3<f64>,
        half_extents: nalgebra::Vector2<f64>,
        orientation: UnitQuaternion<f64>,
    ) -> Self {
        Self::oriented_primitive(center, orientation, ObstacleShape::Ground { half_extents })
    }

    /// Construct a stationary thin triangle-prism obstacle.
    pub fn triangle_prism(
        center: Vector3<f64>,
        vertices: [Vector3<f64>; 3],
        half_thickness: f64,
        orientation: UnitQuaternion<f64>,
    ) -> Self {
        Self::oriented_primitive(
            center,
            orientation,
            ObstacleShape::TrianglePrism {
                vertices,
                half_thickness,
            },
        )
    }

    /// Construct a convex obstacle from hull vertices and outward face normals.
    pub fn convex(
        center: Vector3<f64>,
        vertices: &[Vector3<f64>],
        face_normals: &[Vector3<f64>],
        orientation: UnitQuaternion<f64>,
    ) -> Self {
        let planes = face_normals
            .iter()
            .map(|normal| {
                let offset = vertices
                    .iter()
                    .map(|vertex| normal.dot(vertex))
                    .fold(f64::NEG_INFINITY, f64::max);
                [normal.x, normal.y, normal.z, offset]
            })
            .collect::<Vec<_>>()
            .into();
        let bound_radius = if vertices.len() >= 4
            && vertices
                .iter()
                .all(|vertex| vertex.iter().all(|value| value.is_finite()))
        {
            vertices.iter().map(Vector3::norm).fold(0.0f64, f64::max)
        } else {
            f64::NAN
        };
        Self::oriented_primitive(
            center,
            orientation,
            ObstacleShape::Convex {
                planes,
                bound_radius,
            },
        )
    }

    fn oriented_primitive(
        center: Vector3<f64>,
        orientation: UnitQuaternion<f64>,
        shape: ObstacleShape,
    ) -> Self {
        Self {
            shape,
            center,
            orientation,
            linear_velocity: Vector3::zeros(),
            angular_velocity: Vector3::zeros(),
            friction: 0.0,
            boundary: ObstacleBoundary::default(),
            cpic_group: None,
        }
    }

    /// Check that all collision and motion parameters are finite and valid.
    pub fn is_valid(&self) -> bool {
        let shape_valid = match &self.shape {
            ObstacleShape::Sphere { radius } => radius.is_finite() && *radius > 0.0,
            ObstacleShape::Box { half_extents } => half_extents
                .iter()
                .all(|value| value.is_finite() && *value > 0.0),
            ObstacleShape::Capsule {
                half_height,
                radius,
            } => {
                half_height.is_finite()
                    && *half_height >= 0.0
                    && radius.is_finite()
                    && *radius >= 0.0
                    && (*half_height > 0.0 || *radius > 0.0)
            }
            ObstacleShape::Cylinder {
                half_height,
                radius,
            }
            | ObstacleShape::Cone {
                half_height,
                radius,
            } => {
                half_height.is_finite() && *half_height > 0.0 && radius.is_finite() && *radius > 0.0
            }
            ObstacleShape::Ground { half_extents } => half_extents
                .iter()
                .all(|value| value.is_finite() && *value > 0.0),
            ObstacleShape::TrianglePrism {
                vertices,
                half_thickness,
            } => {
                let area_squared = (vertices[1] - vertices[0])
                    .cross(&(vertices[2] - vertices[0]))
                    .norm_squared();
                half_thickness.is_finite()
                    && *half_thickness > 0.0
                    && vertices
                        .iter()
                        .all(|vertex| vertex.iter().all(|value| value.is_finite()))
                    && area_squared.is_finite()
                    && area_squared > 1e-24
            }
            ObstacleShape::Convex {
                planes,
                bound_radius,
            } => {
                planes.len() >= 4
                    && bound_radius.is_finite()
                    && *bound_radius > 0.0
                    && planes.iter().all(|plane| {
                        plane.iter().all(|value| value.is_finite())
                            && (plane[0]
                                .mul_add(plane[0], plane[1].mul_add(plane[1], plane[2] * plane[2]))
                                - 1.0)
                                .abs()
                                <= 1e-4
                    })
            }
        };
        shape_valid
            && self.cpic_group.is_none_or(|group| {
                group < 32 && matches!(&self.shape, ObstacleShape::TrianglePrism { .. })
            })
            && self.center.iter().all(|value| value.is_finite())
            && self
                .orientation
                .quaternion()
                .coords
                .iter()
                .all(|value| value.is_finite())
            && self.linear_velocity.iter().all(|value| value.is_finite())
            && self.angular_velocity.iter().all(|value| value.is_finite())
            && self.friction.is_finite()
            && self.friction >= 0.0
    }

    /// Signed side of a projected point on a CPIC triangle, if it covers the point.
    pub(crate) fn cpic_side(&self, point: Vector3<f64>) -> Option<(f64, bool)> {
        let ObstacleShape::TrianglePrism { vertices, .. } = &self.shape else {
            return None;
        };
        let local = self
            .orientation
            .inverse_transform_vector(&(point - self.center));
        let a = vertices[0];
        let edge_b = vertices[1] - a;
        let edge_c = vertices[2] - a;
        let normal = edge_b.cross(&edge_c).normalize();
        let signed = (local - a).dot(&normal);
        let projected = local - normal * signed - a;
        let d00 = edge_b.dot(&edge_b);
        let d01 = edge_b.dot(&edge_c);
        let d11 = edge_c.dot(&edge_c);
        let d20 = projected.dot(&edge_b);
        let d21 = projected.dot(&edge_c);
        let denominator = d00 * d11 - d01 * d01;
        let v = (d11 * d20 - d01 * d21) / denominator;
        let w = (d00 * d21 - d01 * d20) / denominator;
        if v >= -1e-8 && w >= -1e-8 && v + w <= 1.0 + 1e-8 {
            Some((signed.abs(), signed >= 0.0))
        } else {
            None
        }
    }

    /// Signed distance and outward normal at a world-space point.
    pub fn surface(&self, point: Vector3<f64>) -> (f64, Vector3<f64>) {
        let relative = point - self.center;
        match &self.shape {
            ObstacleShape::Sphere { radius } => {
                let norm = relative.norm();
                let normal = if norm > 1e-12 {
                    relative / norm
                } else {
                    Vector3::z()
                };
                (norm - radius, normal)
            }
            ObstacleShape::Box { half_extents } => {
                let local = self.orientation.inverse_transform_vector(&relative);
                let closest =
                    local.zip_map(half_extents, |value, extent| value.clamp(-extent, extent));
                let outside = local - closest;
                let norm = outside.norm();
                if norm > 1e-12 {
                    (norm, self.orientation.transform_vector(&(outside / norm)))
                } else {
                    let distance = half_extents - local.map(f64::abs);
                    let axis = if distance.x <= distance.y && distance.x <= distance.z {
                        0
                    } else if distance.y <= distance.z {
                        1
                    } else {
                        2
                    };
                    let mut normal = Vector3::zeros();
                    normal[axis] = if local[axis] < 0.0 { -1.0 } else { 1.0 };
                    (-distance[axis], self.orientation.transform_vector(&normal))
                }
            }
            ObstacleShape::Capsule {
                half_height,
                radius,
            } => {
                let local = self.orientation.inverse_transform_vector(&relative);
                let closest_z = local.z.clamp(-half_height, *half_height);
                let delta = local - Vector3::new(0.0, 0.0, closest_z);
                let norm = delta.norm();
                let axis_epsilon = half_height.max(*radius) * 1e-6;
                let normal = if norm > axis_epsilon {
                    delta / norm
                } else {
                    Vector3::x()
                };
                (norm - radius, self.orientation.transform_vector(&normal))
            }
            ObstacleShape::Cylinder {
                half_height,
                radius,
            } => {
                let local = self.orientation.inverse_transform_vector(&relative);
                let (distance, normal) = cylinder_surface(local, *half_height, *radius);
                (distance, self.orientation.transform_vector(&normal))
            }
            ObstacleShape::Cone {
                half_height,
                radius,
            } => {
                let local = self.orientation.inverse_transform_vector(&relative);
                let (distance, normal) = cone_surface(local, *half_height, *radius);
                (distance, self.orientation.transform_vector(&normal))
            }
            ObstacleShape::Ground { half_extents } => {
                let local = self.orientation.inverse_transform_vector(&relative);
                let outside_x = local.x - local.x.clamp(-half_extents.x, half_extents.x);
                let outside_y = local.y - local.y.clamp(-half_extents.y, half_extents.y);
                let outside = Vector3::new(outside_x, outside_y, local.z);
                let horizontal_distance = outside_x.hypot(outside_y);
                if horizontal_distance <= 1e-12 {
                    (local.z, self.orientation.transform_vector(&Vector3::z()))
                } else {
                    let distance = outside.norm();
                    (
                        distance,
                        self.orientation.transform_vector(&(outside / distance)),
                    )
                }
            }
            ObstacleShape::TrianglePrism {
                vertices,
                half_thickness,
            } => {
                let local = self.orientation.inverse_transform_vector(&relative);
                let (distance, normal) = triangle_prism_surface(local, *vertices, *half_thickness);
                (distance, self.orientation.transform_vector(&normal))
            }
            ObstacleShape::Convex {
                planes,
                bound_radius,
            } => {
                let local = self.orientation.inverse_transform_vector(&relative);
                let (distance, normal) = convex_surface(local, planes, *bound_radius);
                (distance, self.orientation.transform_vector(&normal))
            }
        }
    }

    /// Conservative sphere rejection for finite capsules and mesh primitives.
    pub(crate) fn may_contact(&self, point: Vector3<f64>, margin: f64) -> bool {
        match &self.shape {
            ObstacleShape::Capsule {
                half_height,
                radius,
            } => {
                let bound = half_height + radius + margin;
                (point - self.center).norm_squared() <= bound * bound
            }
            ObstacleShape::TrianglePrism {
                vertices,
                half_thickness,
            } => {
                let radius = vertices.iter().map(Vector3::norm).fold(0.0f64, f64::max)
                    + half_thickness
                    + margin;
                (point - self.center).norm_squared() <= radius * radius
            }
            ObstacleShape::Convex { bound_radius, .. } => {
                let bound = bound_radius + margin;
                (point - self.center).norm_squared() <= bound * bound
            }
            _ => true,
        }
    }

    /// World-space obstacle velocity at a point.
    pub fn point_velocity(&self, point: Vector3<f64>) -> Vector3<f64> {
        self.linear_velocity + self.angular_velocity.cross(&(point - self.center))
    }

    /// Apply the configured boundary response to a contacting velocity.
    pub fn contact_velocity(
        &self,
        point: Vector3<f64>,
        velocity: Vector3<f64>,
        normal: Vector3<f64>,
    ) -> Vector3<f64> {
        let surface_velocity = self.point_velocity(point);
        if self.boundary == ObstacleBoundary::NonReflecting {
            return velocity;
        }
        if self.boundary == ObstacleBoundary::Stick {
            return surface_velocity;
        }
        let relative = velocity - surface_velocity;
        let normal_speed = relative.dot(&normal);
        if normal_speed >= 0.0 {
            return velocity;
        }
        let tangent = relative - normal * normal_speed;
        if self.boundary == ObstacleBoundary::Separate {
            return surface_velocity + tangent;
        }
        let tangent_speed = tangent.norm();
        let friction_scale = if tangent_speed > 1e-12 {
            (1.0 - self.friction * -normal_speed / tangent_speed).max(0.0)
        } else {
            0.0
        };
        surface_velocity + tangent * friction_scale
    }
}

fn cylinder_surface(local: Vector3<f64>, half_height: f64, radius: f64) -> (f64, Vector3<f64>) {
    let radial = local.xy().norm();
    let radial_normal = if radial > 1e-12 {
        Vector3::new(local.x / radial, local.y / radial, 0.0)
    } else {
        Vector3::x()
    };
    let radial_distance = radial - radius;
    let cap_distance = local.z.abs() - half_height;
    let edge_epsilon = half_height.max(radius) * 1e-6;
    if radial_distance.abs() <= edge_epsilon && cap_distance.abs() <= edge_epsilon {
        let cap_normal = if local.z < 0.0 {
            -Vector3::z()
        } else {
            Vector3::z()
        };
        return (
            radial_distance.max(cap_distance),
            (radial_normal + cap_normal).normalize(),
        );
    }
    let radial_outside = radial_distance.max(0.0);
    let cap_outside = cap_distance.max(0.0);
    let outside = radial_normal * radial_outside
        + Vector3::z()
            * (if local.z < 0.0 {
                -cap_outside
            } else {
                cap_outside
            });
    let outside_length = outside.norm();
    if outside_length > 1e-12 {
        (outside_length, outside / outside_length)
    } else if radial_distance >= cap_distance {
        (radial_distance, radial_normal)
    } else {
        (
            cap_distance,
            if local.z < 0.0 {
                -Vector3::z()
            } else {
                Vector3::z()
            },
        )
    }
}

fn cone_surface(local: Vector3<f64>, half_height: f64, radius: f64) -> (f64, Vector3<f64>) {
    let radial = local.xy().norm();
    let radial_normal = if radial > 1e-12 {
        Vector3::new(local.x / radial, local.y / radial, 0.0)
    } else {
        Vector3::x()
    };
    let point = nalgebra::Vector2::new(radial, local.z);
    let base = nalgebra::Vector2::new(radial.clamp(0.0, radius), -half_height);
    let side_start = nalgebra::Vector2::new(radius, -half_height);
    let side_direction = nalgebra::Vector2::new(-radius, 2.0 * half_height);
    let side_t =
        ((point - side_start).dot(&side_direction) / side_direction.norm_squared()).clamp(0.0, 1.0);
    let side = side_start + side_direction * side_t;
    let edge_epsilon = half_height.max(radius) * 1e-6;
    let base_distance_squared = (point - base).norm_squared();
    let side_distance_squared = (point - side).norm_squared();
    let use_base = base_distance_squared <= side_distance_squared;
    let closest = if use_base { base } else { side };
    let delta = point - closest;
    let norm = delta.norm();
    let inside = local.z >= -half_height - edge_epsilon
        && local.z <= half_height + edge_epsilon
        && radial <= radius * (half_height - local.z) / (2.0 * half_height) + edge_epsilon;
    let base_normal = nalgebra::Vector2::new(0.0, -1.0);
    let side_normal = nalgebra::Vector2::new(2.0 * half_height, radius).normalize();
    let near_corner = base_distance_squared <= edge_epsilon * edge_epsilon
        && side_distance_squared <= edge_epsilon * edge_epsilon;
    let feature_normal = if near_corner {
        (base_normal + side_normal).normalize()
    } else if use_base {
        base_normal
    } else {
        side_normal
    };
    let normal_2d = if inside || norm <= edge_epsilon {
        feature_normal
    } else {
        delta / norm
    };
    let normal = radial_normal * normal_2d.x + Vector3::z() * normal_2d.y;
    (if inside { -norm } else { norm }, normal)
}

fn triangle_prism_surface(
    point: Vector3<f64>,
    vertices: [Vector3<f64>; 3],
    half_thickness: f64,
) -> (f64, Vector3<f64>) {
    let [a, b, c] = vertices;
    let normal = (b - a).cross(&(c - a)).normalize();
    let offset = normal * half_thickness;
    let prism = [
        a + offset,
        b + offset,
        c + offset,
        a - offset,
        b - offset,
        c - offset,
    ];
    let faces = [
        [0usize, 1, 2],
        [3, 5, 4],
        [3, 4, 1],
        [3, 1, 0],
        [4, 5, 2],
        [4, 2, 1],
        [5, 3, 0],
        [5, 0, 2],
    ];
    let edge_scale = (b - a)
        .norm()
        .max((c - b).norm())
        .max((a - c).norm())
        .max(half_thickness);
    let epsilon = edge_scale * 1e-6;
    let mut inside = true;
    let mut closest = Vector3::zeros();
    let mut best_distance_squared = f64::INFINITY;
    let mut best_normal = normal;
    let mut tied_normals = Vector3::zeros();
    for [ia, ib, ic] in faces {
        let face_normal = (prism[ib] - prism[ia])
            .cross(&(prism[ic] - prism[ia]))
            .normalize();
        inside &= (point - prism[ia]).dot(&face_normal) <= epsilon;
        let candidate = closest_point_triangle(point, prism[ia], prism[ib], prism[ic]);
        let distance_squared = (point - candidate).norm_squared();
        if distance_squared + epsilon * epsilon < best_distance_squared {
            best_distance_squared = distance_squared;
            closest = candidate;
            best_normal = face_normal;
            tied_normals = face_normal;
        } else if (distance_squared - best_distance_squared).abs() <= epsilon * epsilon {
            tied_normals += face_normal;
        }
    }
    let distance = best_distance_squared.sqrt();
    let feature_normal = tied_normals.try_normalize(1e-12).unwrap_or(best_normal);
    let outward = if inside || distance <= epsilon {
        feature_normal
    } else {
        (point - closest) / distance
    };
    (if inside { -distance } else { distance }, outward)
}

fn convex_surface(
    point: Vector3<f64>,
    planes: &[[f64; 4]],
    bound_radius: f64,
) -> (f64, Vector3<f64>) {
    let mut maximum = f64::NEG_INFINITY;
    let mut nearest_normal = Vector3::z();
    for plane in planes {
        let normal = Vector3::new(plane[0], plane[1], plane[2]);
        let distance = normal.dot(&point) - plane[3];
        if distance > maximum {
            maximum = distance;
            nearest_normal = normal;
        }
    }
    if maximum <= 0.0 {
        return (maximum, nearest_normal);
    }

    let mut projected = point;
    let mut stack_corrections = [Vector3::zeros(); 32];
    let mut heap_corrections = Vec::new();
    let corrections: &mut [Vector3<f64>] = if planes.len() <= stack_corrections.len() {
        &mut stack_corrections[..planes.len()]
    } else {
        heap_corrections.resize(planes.len(), Vector3::zeros());
        &mut heap_corrections
    };
    let tolerance = bound_radius.max(1.0) * 1e-10;
    for _ in 0..128 {
        let previous = projected;
        for (plane, correction) in planes.iter().zip(corrections.iter_mut()) {
            let normal = Vector3::new(plane[0], plane[1], plane[2]);
            let shifted = projected + *correction;
            let violation = (normal.dot(&shifted) - plane[3]).max(0.0);
            projected = shifted - normal * violation;
            *correction = shifted - projected;
        }
        if (projected - previous).norm() <= tolerance {
            break;
        }
    }
    let offset = point - projected;
    let distance = offset.norm();
    if distance > tolerance {
        (distance, offset / distance)
    } else {
        (maximum, nearest_normal)
    }
}

fn closest_point_triangle(
    point: Vector3<f64>,
    a: Vector3<f64>,
    b: Vector3<f64>,
    c: Vector3<f64>,
) -> Vector3<f64> {
    let ab = b - a;
    let ac = c - a;
    let ap = point - a;
    let d1 = ab.dot(&ap);
    let d2 = ac.dot(&ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = point - b;
    let d3 = ab.dot(&bp);
    let d4 = ac.dot(&bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return a + ab * (d1 / (d1 - d3));
    }
    let cp = point - c;
    let d5 = ab.dot(&cp);
    let d6 = ac.dot(&cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return a + ac * (d2 / (d2 - d6));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    let reciprocal = 1.0 / (va + vb + vc);
    a + ab * (vb * reciprocal) + ac * (vc * reciprocal)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotated_box_reports_world_normal_and_signed_distance() {
        let box_shape = RigidObstacle::cuboid(
            Vector3::new(1.0, 2.0, 3.0),
            Vector3::new(0.2, 0.4, 0.6),
            UnitQuaternion::from_axis_angle(&Vector3::z_axis(), core::f64::consts::FRAC_PI_2),
        );
        let (distance, normal) = box_shape.surface(Vector3::new(1.5, 2.0, 3.0));
        assert!((distance - 0.1).abs() < 1e-12);
        assert!((normal - Vector3::x()).norm() < 1e-12);
        let (inside, _) = box_shape.surface(box_shape.center);
        assert!((inside + 0.2).abs() < 1e-12);
    }

    #[test]
    fn moving_surface_pushes_and_friction_reduces_sliding() {
        let mut sphere = RigidObstacle::sphere(Vector3::zeros(), 1.0);
        sphere.linear_velocity = Vector3::new(1.0, 0.0, 0.0);
        sphere.friction = 0.5;
        let velocity = sphere.contact_velocity(
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 2.0, 0.0),
            Vector3::x(),
        );
        assert!((velocity - Vector3::new(1.0, 1.5, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn boundary_modes_apply_distinct_contact_velocities() {
        let mut sphere = RigidObstacle::sphere(Vector3::zeros(), 1.0);
        sphere.linear_velocity = Vector3::new(1.0, 0.0, 0.0);
        sphere.friction = 0.5;
        let point = Vector3::x();
        let incoming = Vector3::new(0.0, 2.0, 0.0);
        let normal = Vector3::x();
        let cases = [
            (ObstacleBoundary::Slip, Vector3::new(1.0, 1.5, 0.0)),
            (ObstacleBoundary::Stick, Vector3::new(1.0, 0.0, 0.0)),
            (ObstacleBoundary::Separate, Vector3::new(1.0, 2.0, 0.0)),
            (ObstacleBoundary::NonReflecting, incoming),
        ];
        for (boundary, expected) in cases {
            sphere.boundary = boundary;
            let actual = sphere.contact_velocity(point, incoming, normal);
            assert!(
                (actual - expected).norm() < 1e-12,
                "{boundary:?}: {actual:?}"
            );
        }
    }

    #[test]
    fn capsule_cylinder_and_cone_report_signed_surfaces() {
        let capsule =
            RigidObstacle::capsule(Vector3::zeros(), 0.5, 0.2, UnitQuaternion::identity());
        let (distance, normal) = capsule.surface(Vector3::new(0.3, 0.0, 0.0));
        assert!((distance - 0.1).abs() < 1e-12);
        assert!((normal - Vector3::x()).norm() < 1e-12);
        let (distance, normal) = capsule.surface(Vector3::new(0.0, 0.0, 0.8));
        assert!((distance - 0.1).abs() < 1e-12);
        assert!((normal - Vector3::z()).norm() < 1e-12);
        assert!((capsule.surface(Vector3::zeros()).0 + 0.2).abs() < 1e-12);
        let line = RigidObstacle::capsule(Vector3::zeros(), 0.5, 0.0, UnitQuaternion::identity());
        assert!(line.is_valid());
        assert!(line.may_contact(Vector3::new(0.0, 0.0, 0.54), 0.04));
        assert!(!line.may_contact(Vector3::new(0.0, 0.0, 0.55), 0.04));

        let cylinder =
            RigidObstacle::cylinder(Vector3::zeros(), 0.5, 0.2, UnitQuaternion::identity());
        let (distance, normal) = cylinder.surface(Vector3::new(0.3, 0.0, 0.7));
        assert!((distance - 0.1_f64.hypot(0.2)).abs() < 1e-12);
        assert!((normal - Vector3::new(0.1, 0.0, 0.2).normalize()).norm() < 1e-12);
        assert!((cylinder.surface(Vector3::zeros()).0 + 0.2).abs() < 1e-12);

        let cone = RigidObstacle::cone(Vector3::zeros(), 0.5, 0.4, UnitQuaternion::identity());
        let (above, above_normal) = cone.surface(Vector3::new(0.0, 0.0, 0.7));
        assert!((above - 0.2).abs() < 1e-12);
        assert!((above_normal - Vector3::z()).norm() < 1e-12);
        let (below, below_normal) = cone.surface(Vector3::new(0.0, 0.0, -0.7));
        assert!((below - 0.2).abs() < 1e-12);
        assert!((below_normal + Vector3::z()).norm() < 1e-12);
        let (inside, side_normal) = cone.surface(Vector3::zeros());
        assert!(inside < -0.1);
        assert!(side_normal.x > 0.0 && side_normal.z > 0.0);
        assert!(cone.is_valid());
    }

    #[test]
    fn rotated_capsule_and_invalid_primitives() {
        let orientation =
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), core::f64::consts::FRAC_PI_2);
        let capsule = RigidObstacle::capsule(Vector3::zeros(), 0.5, 0.2, orientation);
        let (distance, normal) = capsule.surface(Vector3::new(0.8, 0.0, 0.0));
        assert!((distance - 0.1).abs() < 1e-12);
        assert!((normal - Vector3::x()).norm() < 1e-12);
        assert!(
            !RigidObstacle::cylinder(Vector3::zeros(), 0.0, 0.2, UnitQuaternion::identity(),)
                .is_valid()
        );
        assert!(
            !RigidObstacle::cone(Vector3::zeros(), 0.5, -0.2, UnitQuaternion::identity(),)
                .is_valid()
        );
    }

    #[test]
    fn finite_ground_has_one_sided_interior_and_finite_edges() {
        let ground = RigidObstacle::ground(
            Vector3::zeros(),
            nalgebra::Vector2::repeat(1.0),
            UnitQuaternion::identity(),
        );
        assert_eq!(
            ground.surface(Vector3::new(0.0, 0.0, 0.2)),
            (0.2, Vector3::z())
        );
        assert_eq!(
            ground.surface(Vector3::new(0.0, 0.0, -0.2)),
            (-0.2, Vector3::z())
        );
        let (outside, normal) = ground.surface(Vector3::new(1.3, 0.0, 0.0));
        assert!((outside - 0.3).abs() < 1e-12);
        assert!((normal - Vector3::x()).norm() < 1e-12);
        let (below_edge, normal) = ground.surface(Vector3::new(1.3, 0.0, -0.4));
        assert!((below_edge - 0.5).abs() < 1e-12);
        assert!((normal - Vector3::new(0.6, 0.0, -0.8)).norm() < 1e-12);
        assert!(ground.is_valid());
    }

    #[test]
    fn triangle_prism_exposes_caps_sides_and_interior() {
        let vertices = [
            Vector3::new(-0.2, -0.2, 0.0),
            Vector3::new(0.2, -0.2, 0.0),
            Vector3::new(0.0, 0.2, 0.0),
        ];
        let prism = RigidObstacle::triangle_prism(
            Vector3::zeros(),
            vertices,
            0.01,
            UnitQuaternion::identity(),
        );
        assert!(prism.is_valid());
        let (above, above_normal) = prism.surface(Vector3::new(0.0, -0.05, 0.05));
        assert!((above - 0.04).abs() < 1e-12);
        assert!((above_normal - Vector3::z()).norm() < 1e-12);
        let (below, below_normal) = prism.surface(Vector3::new(0.0, -0.05, -0.05));
        assert!((below - 0.04).abs() < 1e-12);
        assert!((below_normal + Vector3::z()).norm() < 1e-12);
        let (inside, _) = prism.surface(Vector3::new(0.0, -0.05, 0.0));
        assert!((inside + 0.01).abs() < 1e-12);
        let (outside, side_normal) = prism.surface(Vector3::new(0.0, -0.3, 0.0));
        assert!((outside - 0.1).abs() < 1e-12);
        assert!((side_normal + Vector3::y()).norm() < 1e-12);
    }

    #[test]
    fn convex_planes_report_face_edge_vertex_and_interior() {
        let convex = RigidObstacle::convex(
            Vector3::zeros(),
            &[Vector3::zeros(), Vector3::x(), Vector3::y(), Vector3::z()],
            &[
                -Vector3::x(),
                -Vector3::y(),
                -Vector3::z(),
                Vector3::repeat(1.0).normalize(),
            ],
            UnitQuaternion::identity(),
        );
        assert!(convex.is_valid());
        let (inside, inside_normal) = convex.surface(Vector3::repeat(0.1));
        assert!((inside + 0.1).abs() < 1e-12);
        assert!((inside_normal + Vector3::x()).norm() < 1e-12);
        let (face, face_normal) = convex.surface(Vector3::new(-0.2, 0.1, 0.1));
        assert!((face - 0.2).abs() < 1e-9);
        assert!((face_normal + Vector3::x()).norm() < 1e-9);
        let (edge, edge_normal) = convex.surface(Vector3::new(-0.2, -0.3, 0.1));
        assert!((edge - 0.2_f64.hypot(0.3)).abs() < 1e-8);
        assert!((edge_normal - Vector3::new(-0.2, -0.3, 0.0).normalize()).norm() < 1e-8);
        let (vertex, _) = convex.surface(Vector3::new(1.2, 0.1, 0.1));
        assert!((vertex - 0.06_f64.sqrt()).abs() < 1e-7);
        let invalid = RigidObstacle::convex(
            Vector3::zeros(),
            &[Vector3::new(f64::NAN, 0.0, 0.0)],
            &[-Vector3::x(); 4],
            UnitQuaternion::identity(),
        );
        assert!(!invalid.is_valid());
    }

    #[test]
    fn convex_plane_distance_matches_analytic_box() {
        let half = Vector3::new(0.2, 0.3, 0.4);
        let vertices = [-1.0, 1.0]
            .into_iter()
            .flat_map(|x| {
                [-1.0, 1.0].into_iter().flat_map(move |y| {
                    [-1.0, 1.0]
                        .into_iter()
                        .map(move |z| Vector3::new(x * half.x, y * half.y, z * half.z))
                })
            })
            .collect::<Vec<_>>();
        let convex = RigidObstacle::convex(
            Vector3::zeros(),
            &vertices,
            &[
                Vector3::x(),
                -Vector3::x(),
                Vector3::y(),
                -Vector3::y(),
                Vector3::z(),
                -Vector3::z(),
            ],
            UnitQuaternion::identity(),
        );
        let cuboid = RigidObstacle::cuboid(Vector3::zeros(), half, UnitQuaternion::identity());
        for x in [-0.6, -0.2, -0.1, 0.0, 0.1, 0.2, 0.6] {
            for y in [-0.7, -0.3, 0.0, 0.3, 0.7] {
                for z in [-0.8, -0.4, 0.0, 0.4, 0.8] {
                    let point = Vector3::new(x, y, z);
                    let actual = convex.surface(point).0;
                    let expected = cuboid.surface(point).0;
                    assert!(
                        (actual - expected).abs() < 1e-7,
                        "point={point:?}, actual={actual}, expected={expected}"
                    );
                }
            }
        }
    }
}
