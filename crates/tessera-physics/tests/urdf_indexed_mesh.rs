//! Indexed URDF surfaces keep their gaps and transfer impulses to their links.

use nalgebra::{Isometry3, Vector3};
use tessera_physics::articulated_world::{ArticulatedWorld, ArticulatedWorldParams};
use tessera_physics::convex::ConvexGeometry;
use tessera_physics::mesh::{ConvexMeshResolver, TriangleMeshGeometry};
use tessera_physics::urdf::{UrdfLoadError, UrdfLoadOptions, load_urdf_str_with_mesh_resolver};

type TestResult<T = ()> = Result<T, Box<dyn core::error::Error>>;

const DT: f64 = 0.001;

#[derive(Default)]
struct IndexedResolver {
    calls: Vec<(String, [f64; 3])>,
}

fn walls(
    scale: [f64; 3],
) -> Result<TriangleMeshGeometry, tessera_physics::mesh::MeshGeometryError> {
    let scale = Vector3::from(scale);
    let vertices = [-1.0, 1.0]
        .into_iter()
        .flat_map(|x| {
            [
                Vector3::new(x, -1.0, -1.0),
                Vector3::new(x, 1.0, -1.0),
                Vector3::new(x, 1.0, 1.0),
                Vector3::new(x, -1.0, 1.0),
            ]
        })
        .map(|vertex| vertex.component_mul(&scale))
        .collect();
    TriangleMeshGeometry::new(vertices, vec![[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
}

impl ConvexMeshResolver for IndexedResolver {
    fn resolve(&mut self, _: &str, _: [f64; 3]) -> Result<Vec<ConvexGeometry>, String> {
        panic!("indexed URDF loading must not request a convex hull");
    }

    fn resolve_indexed(
        &mut self,
        filename: &str,
        scale: [f64; 3],
    ) -> Result<Vec<TriangleMeshGeometry>, String> {
        self.calls.push((filename.into(), scale));
        Ok(vec![walls(scale).map_err(|error| error.to_string())?])
    }
}

fn model(sphere_x: f64) -> TestResult<ArticulatedWorld> {
    let xml = format!(
        r#"<robot name="indexed">
        <link name="base"/>
        <link name="mesh">
          <inertial><mass value="1"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
          <collision><geometry><mesh filename="walls.obj"/></geometry></collision>
        </link>
        <link name="sphere">
          <inertial><mass value="1"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
          <collision><geometry><sphere radius="0.1"/></geometry></collision>
        </link>
        <joint name="mesh_slide" type="prismatic"><parent link="base"/><child link="mesh"/>
          <origin xyz="0 0 2"/><axis xyz="1 0 0"/><limit lower="-10" upper="10" effort="100" velocity="100"/></joint>
        <joint name="sphere_slide" type="prismatic"><parent link="base"/><child link="sphere"/>
          <origin xyz="{sphere_x} 0.25 2.15"/><axis xyz="1 0 0"/><limit lower="-10" upper="10" effort="100" velocity="100"/></joint>
        </robot>"#
    );
    Ok(load_urdf_str_with_mesh_resolver(
        &xml,
        UrdfLoadOptions {
            convex_hull: false,
            world: ArticulatedWorldParams {
                gravity: [0.0; 3],
                max_substep: DT,
                friction: 0.0,
                ..Default::default()
            },
            ..Default::default()
        },
        &mut IndexedResolver::default(),
    )?
    .world)
}

#[test]
fn indexed_collision_and_visual_selection_retains_scaled_topology_and_origins() -> TestResult {
    let xml = r#"<robot name="selection"><link name="base">
      <collision><origin xyz="1 2 3" rpy="0 0 0.5"/><geometry><mesh filename="collision.obj" scale="-2 3 4"/></geometry></collision>
      <visual><origin xyz="4 5 6"/><geometry><mesh filename="visual.obj"/></geometry></visual>
    </link></robot>"#;
    for collision in [false, true] {
        for visual in [false, true] {
            let mut resolver = IndexedResolver::default();
            let loaded = load_urdf_str_with_mesh_resolver(
                xml,
                UrdfLoadOptions {
                    convex_hull: false,
                    scale: 2.0,
                    create_colliders_from_collision_shapes: collision,
                    create_colliders_from_visual_shapes: visual,
                    ..Default::default()
                },
                &mut resolver,
            )?;
            assert!(loaded.world.convex_shapes.is_empty());
            let meshes = loaded.world.triangle_meshes();
            assert_eq!(meshes.len(), usize::from(collision) + usize::from(visual));
            let mut expected = Vec::new();
            if collision {
                expected.push(("collision.obj".to_owned(), [-4.0, 6.0, 8.0]));
                assert_eq!(meshes[0].geometry, walls([-4.0, 6.0, 8.0])?);
                assert_eq!(
                    meshes[0].origin.translation.vector,
                    Vector3::new(2.0, 4.0, 6.0)
                );
                assert!((meshes[0].origin.rotation.angle() - 0.5).abs() < 1e-12);
            }
            if visual {
                expected.push(("visual.obj".to_owned(), [2.0; 3]));
                let mesh = meshes.last().ok_or("missing selected visual mesh")?;
                assert_eq!(mesh.geometry, walls([2.0; 3])?);
                assert_eq!(mesh.origin, Isometry3::translation(8.0, 10.0, 12.0));
            }
            assert_eq!(resolver.calls, expected);
        }
    }
    Ok(())
}

#[test]
fn indexed_mesh_contact_preserves_gap_and_moves_both_native_owners() -> TestResult {
    for x in [0.0, 0.95] {
        let mut world = model(x)?;
        assert_eq!(world.triangle_meshes()[0].geometry.triangles().len(), 4);
        world.step(DT, &[0.0; 2])?;
        if x == 0.0 {
            assert!(world.velocities.norm() < 1e-12);
            assert!(world.contact_samples.iter().all(Vec::is_empty));
        } else {
            assert!(world.velocities[0] > 0.01);
            assert!(world.velocities[1] < -0.01);
            assert!((world.velocities[0] + world.velocities[1]).abs() < 1e-10);
            assert!(world.contact_forces[1].x > 0.0);
            assert!(world.contact_forces[2].x < 0.0);
        }
    }
    Ok(())
}

#[test]
fn indexed_mesh_link_validation_is_atomic() -> TestResult {
    let mut world = model(0.95)?;
    let original = world.triangle_meshes().to_vec();
    let mut invalid = original.clone();
    invalid[0].link = world.articulation.link_count();
    assert!(world.set_triangle_meshes(invalid).is_err());
    assert_eq!(world.triangle_meshes(), original);
    let mut invalid = original.clone();
    invalid[0].origin.translation.x = f64::NAN;
    assert!(world.set_triangle_meshes(invalid).is_err());
    assert_eq!(world.triangle_meshes(), original);
    Ok(())
}

#[test]
fn indexed_resolver_errors_keep_filename_and_do_not_request_convex_parts() -> TestResult {
    let xml = r#"<robot name="mesh"><link name="base"><collision><geometry><mesh filename="missing.obj"/></geometry></collision></link></robot>"#;
    let mut resolver = |_: &str, _: [f64; 3]| -> Result<Vec<ConvexGeometry>, String> {
        panic!("convex resolution is not an indexed fallback")
    };
    assert!(matches!(
        load_urdf_str_with_mesh_resolver(
            xml,
            UrdfLoadOptions { convex_hull: false, ..Default::default() },
            &mut resolver,
        ),
        Err(UrdfLoadError::Mesh { filename, .. }) if filename == "missing.obj"
    ));
    Ok(())
}

#[cfg(feature = "gpu-contact")]
#[test]
fn indexed_link_mesh_contacts_match_cpu_in_resident_and_host_gpu_owners() -> TestResult {
    use tessera_physics::gpu_articulated_dynamics::GpuArticulatedDynamicsBatch;
    use tessera_physics::gpu_contact_pipeline::GpuContactDevice;

    let context = GpuContactDevice::new_with_backends(wgpu::Backends::VULKAN)?;
    for x in [0.0, 0.95] {
        let mut cpu = model(x)?;
        let mut host_gpu = model(x)?;
        let resident_world = model(x)?;
        let input = resident_world.gpu_dynamics_input(&[0.0; 2])?;
        assert_eq!(input.convex_sphere_pairs.len(), 4);
        assert!(
            input
                .convex_sphere_pairs
                .iter()
                .all(|pair| { pair.convex_link == 1 && pair.sphere_link == 2 })
        );
        let resident = GpuArticulatedDynamicsBatch::new(&context, &[input], DT)?;
        resident.submit_steps(1)?;
        let actual = resident.readback()?;
        cpu.step(DT, &[0.0; 2])?;
        host_gpu.step_gpu(DT, &[0.0; 2], &context)?;
        for slot in 0..2 {
            assert!((actual[0].velocities[slot] - cpu.velocities[slot]).abs() < 1e-3);
            assert!((host_gpu.velocities[slot] - cpu.velocities[slot]).abs() < 1e-3);
        }
        if x == 0.0 {
            assert!(actual[0].velocities.iter().all(|value| value.abs() < 1e-12));
        } else {
            assert!(actual[0].velocities[0] > 0.01);
            assert!(actual[0].velocities[1] < -0.01);
        }
    }
    Ok(())
}

#[cfg(feature = "gpu-contact")]
#[test]
fn indexed_mesh_keeps_its_link_in_composed_resident_scene_and_invalidates_geometry() -> TestResult {
    use nalgebra::Matrix3;
    use tessera_physics::articulated_world::{SceneBody, SceneCollider};
    use tessera_physics::gpu_contact_pipeline::GpuContactDevice;
    use tessera_physics::gpu_scene_dynamics::{GpuSceneDynamics, GpuSceneDynamicsError};

    let context = GpuContactDevice::new_with_backends(wgpu::Backends::VULKAN)?;
    for x in [0.0, 0.95] {
        let make_scene = || -> TestResult<ArticulatedWorld> {
            let mut world = model(x)?;
            world.colliders.clear();
            let _body = world.add_scene_body(SceneBody::new(
                Isometry3::translation(x, 0.25, 2.15),
                1.0,
                Matrix3::identity(),
                vec![SceneCollider::Sphere {
                    center: Vector3::zeros(),
                    radius: 0.1,
                }],
            )?);
            Ok(world)
        };
        let mut cpu = make_scene()?;
        let mut gpu = make_scene()?;
        let mut owner = GpuSceneDynamics::new(&gpu, &context, DT)?;
        cpu.step(DT, &[0.0; 2])?;
        let _diagnostics = owner.step(&mut gpu, &[0.0; 2], 1)?;
        assert!(
            (gpu.velocities[0] - cpu.velocities[0]).abs() < 1e-3,
            "sphere_x={x}: robot GPU {:?}, CPU {:?}; body GPU {:?}, CPU {:?}; angular GPU {:?}, CPU {:?}; samples GPU {:?}, CPU {:?}",
            gpu.velocities,
            cpu.velocities,
            gpu.scene_bodies[0].linear_velocity,
            cpu.scene_bodies[0].linear_velocity,
            gpu.scene_bodies[0].angular_velocity,
            cpu.scene_bodies[0].angular_velocity,
            gpu.contact_samples,
            cpu.contact_samples,
        );
        assert!(
            (gpu.scene_bodies[0].linear_velocity.x - cpu.scene_bodies[0].linear_velocity.x).abs()
                < 1e-3
        );
        if x == 0.0 {
            assert!(gpu.velocities.norm() < 1e-12);
            assert!(gpu.scene_bodies[0].linear_velocity.norm() < 1e-12);
        } else {
            assert!(gpu.velocities[0] > 0.01);
            assert!(gpu.scene_bodies[0].linear_velocity.x < -0.01);
        }
        let mut changed = gpu.triangle_meshes().to_vec();
        changed[0].origin.translation.x += 0.1;
        gpu.set_triangle_meshes(changed)?;
        assert!(matches!(
            owner.step(&mut gpu, &[0.0; 2], 1),
            Err(GpuSceneDynamicsError::WorldChanged)
        ));
    }
    Ok(())
}
