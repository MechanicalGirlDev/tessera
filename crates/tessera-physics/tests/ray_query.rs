//! Analytic reference checks for volume and thin-surface ray casts.
#![allow(clippy::unwrap_used)]
use nalgebra::{Isometry3, Point3, Vector3};
use tessera_physics::{
    convex::ConvexGeometry,
    mesh::{HeightFieldGeometry, PolylineGeometry, TriangleMeshGeometry},
    ray_query::{Ray, RayShape, cast_ray},
};
fn ray(o: [f64; 3], d: [f64; 3]) -> Ray {
    Ray {
        origin: Vector3::from(o),
        direction: Vector3::from(d),
    }
}
#[test]
fn distant_sphere_tangency_does_not_lose_the_perpendicular_offset() {
    let pose = Isometry3::identity();
    assert!(
        cast_ray(
            ray([1e8, 0.5001, 0.0], [-2.0, 0.0, 0.0]),
            &pose,
            RayShape::Sphere(0.5),
            5e7,
            false
        )
        .unwrap()
        .is_none()
    );
    let hit = cast_ray(
        ray([1e8, 0.5, 0.0], [-2.0, 0.0, 0.0]),
        &pose,
        RayShape::Sphere(0.5),
        5e7,
        false,
    )
    .unwrap()
    .unwrap();
    assert_eq!(hit.toi, 5e7);
    assert_eq!(hit.point, Vector3::new(0.0, 0.5, 0.0));
    assert_eq!(hit.normal, Vector3::y());
}
#[test]
fn segment_bvh_preserves_candidates_within_intersection_roundoff() {
    let line = PolylineGeometry::new(
        vec![Vector3::new(-1.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0)],
        vec![[0, 1]],
    )
    .unwrap();
    let hit = cast_ray(
        ray([0.0, 32.0 * f64::EPSILON, 1.0], [0.0, 0.0, -1.0]),
        &Isometry3::identity(),
        RayShape::Polyline(&line),
        1.0,
        false,
    )
    .unwrap()
    .unwrap();
    assert_eq!(hit.toi, 1.0);
    assert_eq!(hit.feature, 0);
    assert!(
        cast_ray(
            ray([0.0, 1e-6, 1.0], [0.0, 0.0, -1.0]),
            &Isometry3::identity(),
            RayShape::Polyline(&line),
            1.0,
            false
        )
        .unwrap()
        .is_none()
    );
}
#[test]
fn volume_entry_exit_limits_and_pose_match_analytic_boundaries() {
    let shapes = [
        (RayShape::Sphere(1.0), 1.0),
        (RayShape::Box(Vector3::repeat(0.5)), 0.5),
        (
            RayShape::Capsule {
                radius: 1.0,
                half_length: 1.0,
            },
            1.0,
        ),
        (
            RayShape::Cylinder {
                radius: 1.0,
                half_length: 1.0,
            },
            1.0,
        ),
        (
            RayShape::Cone {
                radius: 1.0,
                half_length: 1.0,
            },
            0.5,
        ),
    ];
    for (shape, radius) in shapes {
        for pose in [
            Isometry3::identity(),
            Isometry3::new(Vector3::new(10.0, -4.0, 3.0), Vector3::new(0.0, 0.0, 1.1)),
        ] {
            let r = Ray {
                origin: pose.transform_point(&Point3::new(2.0, 0.0, 0.0)).coords,
                direction: pose.transform_vector(&Vector3::new(-2.0, 0.0, 0.0)),
            };
            let expected = (2.0 - radius) * 0.5;
            let hit = cast_ray(r, &pose, shape, 2.0, false).unwrap().unwrap();
            assert!((hit.toi - expected).abs() < 1e-12, "{shape:?}: {hit:?}");
            assert!((hit.point - (r.origin + r.direction * expected)).norm() < 1e-12);
            assert!((hit.normal.norm() - 1.0).abs() < 1e-12);
            assert!(
                cast_ray(r, &pose, shape, expected - 1e-6, false)
                    .unwrap()
                    .is_none()
            );
            let inside = Ray {
                origin: pose.translation.vector,
                direction: r.direction,
            };
            assert_eq!(
                cast_ray(inside, &pose, shape, 0.0, true)
                    .unwrap()
                    .unwrap()
                    .toi,
                0.0
            );
            let exit = cast_ray(inside, &pose, shape, 2.0, false).unwrap().unwrap();
            assert!((exit.toi - radius * 0.5).abs() < 1e-12);
            assert!(exit.normal.dot(&inside.direction) > 0.0);
        }
    }
}
#[test]
fn caps_tangency_and_linear_cone_equation_are_exact() {
    let pose = Isometry3::identity();
    for shape in [
        RayShape::Cylinder {
            radius: 1.0,
            half_length: 1.0,
        },
        RayShape::Cone {
            radius: 1.0,
            half_length: 1.0,
        },
    ] {
        for (origin, direction, normal) in [
            ([0.0, 0.0, 2.0], [0.0, 0.0, -1.0], Vector3::z()),
            ([0.0, 0.0, -2.0], [0.0, 0.0, 1.0], -Vector3::z()),
        ] {
            let hit = cast_ray(ray(origin, direction), &pose, shape, 2.0, false)
                .unwrap()
                .unwrap();
            assert!((hit.toi - 1.0).abs() < 1e-12);
            assert!((hit.normal - normal).norm() < 1e-12);
        }
    }
    let tangent = cast_ray(
        ray([1.0, 1.0, 0.0], [0.0, -1.0, 0.0]),
        &pose,
        RayShape::Sphere(1.0),
        1.0,
        false,
    )
    .unwrap()
    .unwrap();
    assert!((tangent.toi - 1.0).abs() < 1e-12);
    let cone = cast_ray(
        ray([1.0, 1.0, 0.0], [0.0, -1.0, 0.0]),
        &pose,
        RayShape::Cone {
            radius: 1.0,
            half_length: 1.0,
        },
        3.0,
        false,
    )
    .unwrap();
    assert!(cone.is_none());
    let cone = cast_ray(
        ray([1.0, 0.0, 1.0], [-0.5, 0.0, -1.0]),
        &pose,
        RayShape::Cone {
            radius: 1.0,
            half_length: 1.0,
        },
        3.0,
        false,
    )
    .unwrap()
    .unwrap();
    assert!((cone.toi - 1.0).abs() < 1e-12);
}
#[test]
fn convex_mesh_heightfield_and_polyline_use_real_features() {
    let pose = Isometry3::identity();
    let vertices = (0..8)
        .map(|i| {
            Vector3::new(
                if i & 1 == 0 { -0.5 } else { 0.5 },
                if i & 2 == 0 { -0.5 } else { 0.5 },
                if i & 4 == 0 { -0.5 } else { 0.5 },
            )
        })
        .collect();
    let hull = ConvexGeometry::new(
        vertices,
        vec![
            Vector3::x(),
            -Vector3::x(),
            Vector3::y(),
            -Vector3::y(),
            Vector3::z(),
            -Vector3::z(),
        ],
        vec![Vector3::x(), Vector3::y(), Vector3::z()],
    )
    .unwrap();
    let hit = cast_ray(
        ray([0.0, 0.0, 2.0], [0.0, 0.0, -1.0]),
        &pose,
        RayShape::Convex(&hull),
        2.0,
        false,
    )
    .unwrap()
    .unwrap();
    assert!((hit.toi - 1.5).abs() < 1e-12);
    let mesh = TriangleMeshGeometry::new(
        vec![
            Vector3::new(-1.0, -1.0, 0.0),
            Vector3::new(1.0, -1.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        ],
        vec![[0, 1, 2]],
    )
    .unwrap();
    for sign in [-1.0, 1.0] {
        let r = ray([0.0, 0.0, sign], [0.0, 0.0, -sign]);
        let hit = cast_ray(r, &pose, RayShape::TriangleMesh(&mesh), 1.0, true)
            .unwrap()
            .unwrap();
        assert!((hit.toi - 1.0).abs() < 1e-12);
        assert_eq!(hit.feature, 0);
        assert!(hit.normal.dot(&r.direction) < 0.0);
    }
    let line = PolylineGeometry::new(vec![-Vector3::x(), Vector3::x()], vec![[0, 1]]).unwrap();
    for r in [
        ray([0.0, 0.0, 1.0], [0.0, 0.0, -1.0]),
        ray([2.0, 0.0, 0.0], [-1.0, 0.0, 0.0]),
    ] {
        let hit = cast_ray(r, &pose, RayShape::Polyline(&line), 1.0, false)
            .unwrap()
            .unwrap();
        assert!((hit.toi - 1.0).abs() < 1e-12);
        assert_eq!(hit.normal, Vector3::zeros());
    }
    assert!(
        cast_ray(
            ray([0.0, 0.01, 1.0], [0.0, 0.0, -1.0]),
            &pose,
            RayShape::Polyline(&line),
            2.0,
            false
        )
        .unwrap()
        .is_none()
    );
    let field = HeightFieldGeometry::new(2, 2, vec![0.0; 4], Vector3::new(2.0, 2.0, 1.0)).unwrap();
    let hit = cast_ray(
        ray([0.0, 0.0, 1.0], [0.0, 0.0, -1.0]),
        &pose,
        RayShape::Heightfield(&field),
        1.0,
        false,
    )
    .unwrap()
    .unwrap();
    assert!((hit.toi - 1.0).abs() < 1e-12);
}
#[test]
fn invalid_and_parallel_queries_do_not_create_hits() {
    let pose = Isometry3::identity();
    for r in [
        ray([0.0; 3], [0.0; 3]),
        ray([f64::NAN, 0.0, 0.0], [1.0, 0.0, 0.0]),
    ] {
        assert!(cast_ray(r, &pose, RayShape::Sphere(1.0), 2.0, true).is_err());
    }
    assert!(
        cast_ray(
            ray([2.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            &pose,
            RayShape::Box(Vector3::repeat(0.5)),
            3.0,
            false
        )
        .unwrap()
        .is_none()
    );
    assert!(
        cast_ray(
            ray([2.0, 0.0, 0.0], [-1.0, 0.0, 0.0]),
            &pose,
            RayShape::Sphere(-1.0),
            3.0,
            false
        )
        .is_err()
    );
}

#[test]
fn nearly_parallel_segment_retains_its_actual_intersection() {
    let line = PolylineGeometry::new(
        vec![Vector3::new(-1.0, 0.0, 0.0), Vector3::new(1.0, 1e-10, 0.0)],
        vec![[0, 1]],
    )
    .unwrap();
    let r = ray([2.0, 0.0, 0.0], [-1.0, 0.0, 0.0]);
    let pose = Isometry3::identity();
    assert!(
        cast_ray(r, &pose, RayShape::Polyline(&line), 2.0, false)
            .unwrap()
            .is_none()
    );
    let hit = cast_ray(r, &pose, RayShape::Polyline(&line), 4.0, false)
        .unwrap()
        .unwrap();
    assert!((hit.toi - 3.0).abs() < 1e-12);
}
