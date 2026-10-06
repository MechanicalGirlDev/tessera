//! Independent analytic projection checks across all shapes and transforms.
#![allow(clippy::unwrap_used)]
use nalgebra::{Isometry3, Point3, Vector3};
use tessera_physics::{
    convex::ConvexGeometry,
    mesh::{HeightFieldGeometry, PolylineGeometry, TriangleMeshGeometry},
    point_query::project_point,
    ray_query::RayShape,
};
fn v(x: f64, y: f64, z: f64) -> Vector3<f64> {
    Vector3::new(x, y, z)
}
#[test]
fn scene_nearest_distance_filters_ties_and_invalid_inputs() {
    use tessera_physics::point_query::{PointQueryBody, project_point_scene};
    let bodies = [
        PointQueryBody {
            pose: Isometry3::translation(10.0, 0.0, 0.0),
            shape: RayShape::Sphere(1.0),
            groups: 1,
        },
        PointQueryBody {
            pose: Isometry3::translation(3.0, 0.0, 0.0),
            shape: RayShape::Sphere(1.0),
            groups: 2,
        },
        PointQueryBody {
            pose: Isometry3::translation(-3.0, 0.0, 0.0),
            shape: RayShape::Sphere(1.0),
            groups: 2,
        },
    ];
    let query = |bound, groups, excluded| {
        project_point_scene(Vector3::zeros(), &bodies, bound, groups, excluded, true)
    };
    let hit = query(2.0, 3, None).unwrap().unwrap();
    assert_eq!(hit.body, 1);
    assert_eq!(hit.projection.point, v(2.0, 0.0, 0.0));
    assert_eq!(hit.projection.distance, 2.0);
    assert_eq!(query(2.0, 3, Some(1)).unwrap().unwrap().body, 2);
    assert_eq!(query(f64::INFINITY, 1, None).unwrap().unwrap().body, 0);
    assert!(query(1.999, 3, None).unwrap().is_none());
    assert!(query(f64::INFINITY, 0, None).unwrap().is_none());
    assert!(query(-1.0, 3, None).is_err());
    assert!(query(f64::NAN, 3, None).is_err());
    assert!(query(2.0, 3, Some(3)).is_err());
    assert!(project_point_scene(v(f64::INFINITY, 0.0, 0.0), &[], 0.0, 0, None, true).is_err());
    let overlap = [
        PointQueryBody {
            pose: Isometry3::identity(),
            shape: RayShape::Sphere(2.0),
            groups: 1,
        },
        PointQueryBody {
            pose: Isometry3::identity(),
            shape: RayShape::Sphere(1.0),
            groups: 1,
        },
    ];
    assert_eq!(
        project_point_scene(Vector3::zeros(), &overlap, 0.0, 1, None, true)
            .unwrap()
            .unwrap()
            .body,
        0
    );
    assert_eq!(
        project_point_scene(Vector3::zeros(), &overlap, 2.0, 1, None, false)
            .unwrap()
            .unwrap()
            .body,
        1
    );
    let invalid = [
        overlap[0],
        PointQueryBody {
            shape: RayShape::Sphere(-1.0),
            ..overlap[1]
        },
    ];
    assert!(project_point_scene(Vector3::zeros(), &invalid, 2.0, 1, None, true).is_err());
    assert!(project_point_scene(Vector3::zeros(), &invalid, 2.0, 1, Some(1), true).is_ok());
}
#[test]
fn primitive_boundaries_inside_modes_and_pose_are_exact() {
    let sphere = RayShape::Sphere(1.0);
    let cube = RayShape::Box(v(1.0, 2.0, 3.0));
    let capsule = RayShape::Capsule {
        radius: 1.0,
        half_length: 1.0,
    };
    let cylinder = RayShape::Cylinder {
        radius: 1.0,
        half_length: 1.0,
    };
    let cone = RayShape::Cone {
        radius: 1.0,
        half_length: 1.0,
    };
    let cases = [
        (sphere, v(2.0, 0.0, 0.0), v(1.0, 0.0, 0.0), false),
        (sphere, v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0), true),
        (cube, v(4.0, 4.0, 4.0), v(1.0, 2.0, 3.0), false),
        (cube, v(0.9, 0.0, 0.0), v(1.0, 0.0, 0.0), true),
        (capsule, v(0.0, 0.0, 3.0), v(0.0, 0.0, 2.0), false),
        (capsule, v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0), true),
        (cylinder, v(2.0, 0.0, 2.0), v(1.0, 0.0, 1.0), false),
        (cylinder, v(0.0, 0.0, 0.9), v(0.0, 0.0, 1.0), true),
        (cone, v(2.0, 0.0, 0.0), v(0.8, 0.0, -0.6), false),
        (cone, v(0.0, 0.0, 0.0), v(0.4, 0.0, 0.2), true),
        (cone, v(0.0, 0.0, 2.0), v(0.0, 0.0, 1.0), false),
        (cone, v(0.0, 0.0, -2.0), v(0.0, 0.0, -1.0), false),
    ];
    for pose in [
        Isometry3::identity(),
        Isometry3::new(v(10.0, -4.0, 3.0), v(0.3, -0.7, 0.2)),
    ] {
        for (shape, p, q, inside) in cases {
            let world = pose.transform_point(&Point3::from(p)).coords;
            for solid in [false, true] {
                let result = project_point(world, &pose, shape, solid).unwrap();
                let expected = if solid && inside {
                    world
                } else {
                    pose.transform_point(&Point3::from(q)).coords
                };
                assert!(
                    (result.point - expected).norm() < 1e-12,
                    "{shape:?}: solid={solid}: {result:?}: expected={expected:?}"
                );
                assert_eq!(result.is_inside, inside);
                assert!((result.distance - (expected - world).norm()).abs() < 1e-12);
                let direction = if inside {
                    expected - world
                } else {
                    world - expected
                };
                let expected_normal = if solid && inside {
                    let boundary = pose.transform_point(&Point3::from(q)).coords;
                    (boundary - world).normalize()
                } else if direction.norm() > 0.0 {
                    direction.normalize()
                } else {
                    Vector3::zeros()
                };
                assert!(
                    (result.normal - expected_normal).norm() < 1e-12,
                    "{shape:?}: solid={solid}: {result:?}"
                );
            }
        }
    }
}
#[test]
fn convex_face_centres_and_interior_vertices_do_not_change_boundary() {
    let mut vertices = (0..8)
        .map(|i| {
            v(
                if i & 1 == 0 { -1.0 } else { 1.0 },
                if i & 2 == 0 { -1.0 } else { 1.0 },
                if i & 4 == 0 { -1.0 } else { 1.0 },
            )
        })
        .collect::<Vec<_>>();
    vertices.extend([
        v(1.0, 0.0, 0.0),
        v(-1.0, 0.0, 0.0),
        v(0.0, 0.0, 0.0),
        v(1.0, 1.0, 1.0),
    ]);
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
    for (p, q, inside) in [
        (v(2.0, 3.0, 4.0), v(1.0, 1.0, 1.0), false),
        (v(0.8, 0.1, 0.2), v(1.0, 0.1, 0.2), true),
    ] {
        let result =
            project_point(p, &Isometry3::identity(), RayShape::Convex(&hull), false).unwrap();
        assert!((result.point - q).norm() < 1e-12, "{result:?}");
        assert_eq!(result.is_inside, inside);
    }
}
#[test]
fn bvh_mesh_and_segment_features_select_the_actual_nearest_surface() {
    let mesh = TriangleMeshGeometry::new(
        vec![
            v(50.0, 0.0, 0.0),
            v(51.0, 0.0, 0.0),
            v(50.0, 1.0, 0.0),
            v(-1.0, -1.0, 0.0),
            v(1.0, -1.0, 0.0),
            v(0.0, 1.0, 0.0),
            v(-1.0, -1.0, 2.0),
            v(1.0, -1.0, 2.0),
            v(0.0, 1.0, 2.0),
        ],
        vec![[0, 1, 2], [3, 4, 5], [6, 7, 8]],
    )
    .unwrap();
    for (p, q, feature) in [
        (v(0.0, 0.0, 1.8), v(0.0, 0.0, 2.0), 2),
        (v(0.0, 0.0, 0.2), v(0.0, 0.0, 0.0), 1),
        (v(2.0, -1.0, 0.5), v(1.0, -1.0, 0.0), 1),
    ] {
        let result = project_point(
            p,
            &Isometry3::identity(),
            RayShape::TriangleMesh(&mesh),
            true,
        )
        .unwrap();
        assert!((result.point - q).norm() < 1e-12, "{result:?}");
        assert_eq!(result.feature, feature);
        assert!(!result.is_inside);
    }
    let line = PolylineGeometry::new(
        vec![
            v(50.0, 0.0, 0.0),
            v(51.0, 0.0, 0.0),
            v(-1.0, 0.0, 0.0),
            v(1.0, 0.0, 0.0),
        ],
        vec![[0, 1], [2, 3]],
    )
    .unwrap();
    let result = project_point(
        v(2.0, 1.0, 0.0),
        &Isometry3::identity(),
        RayShape::Polyline(&line),
        true,
    )
    .unwrap();
    assert_eq!(result.point, v(1.0, 0.0, 0.0));
    assert_eq!(result.feature, 1);
    assert!(!result.is_inside);
    let field = HeightFieldGeometry::new(2, 2, vec![0.0; 4], v(2.0, 2.0, 1.0)).unwrap();
    let result = project_point(
        v(2.0, 0.0, 0.5),
        &Isometry3::identity(),
        RayShape::Heightfield(&field),
        true,
    )
    .unwrap();
    assert_eq!(result.point, v(1.0, 0.0, 0.0));
    assert!(!result.is_inside);
}
#[test]
fn tiny_sphere_keeps_direction_and_does_not_classify_an_exterior_point_as_inside() {
    let result = project_point(
        v(0.0, 2e-200, 0.0),
        &Isometry3::identity(),
        RayShape::Sphere(1e-200),
        true,
    )
    .unwrap();
    assert_eq!(result.point, v(0.0, 1e-200, 0.0));
    assert_eq!(result.distance, 1e-200);
    assert!(!result.is_inside);
}
#[test]
fn invalid_points_and_dimensions_are_rejected() {
    let pose = Isometry3::identity();
    assert!(project_point(v(f64::NAN, 0.0, 0.0), &pose, RayShape::Sphere(1.0), false).is_err());
    for shape in [
        RayShape::Sphere(0.0),
        RayShape::Box(v(0.0, 1.0, 1.0)),
        RayShape::Cone {
            radius: 1.0,
            half_length: 0.0,
        },
        RayShape::Capsule {
            radius: 1.0,
            half_length: -1.0,
        },
    ] {
        assert!(project_point(Vector3::zeros(), &pose, shape, false).is_err());
    }
}
