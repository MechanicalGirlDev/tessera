"""Automatic resident scene indexes through all four installed-wheel interfaces."""
from tessera3d import (GpuCollisionGroups, GpuPointQuery, GpuRay, GpuSphereWorld,
    GpuSphereBatch, GpuPrimitiveWorld, GpuPrimitiveBatch, GpuPrimitiveBody,
    GpuPrimitiveShape, SphereInput, Quaternion, TesseraError, Vec3)

def vec(x, y, z):
    return Vec3(x=x, y=y, z=z)

def main():
    zero = vec(0.0, 0.0, 0.0)
    identity = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    spheres = [SphereInput(center=vec(i*4.0, 0.0, 3.0), velocity=vec(1.0 if i == 0 else 0.0, 0.0, 0.0),
        radius=0.5, mass=1.0 if i == 0 else 0.0) for i in range(32)]
    primitives = [GpuPrimitiveBody(shape=GpuPrimitiveShape.BOX(half_extents=vec(0.5, 0.5, 0.5)),
        center=s.center, orientation=identity, velocity=s.velocity, angular_velocity=zero,
        mass=s.mass, principal_inertia=vec(1.0/6.0, 1.0/6.0, 1.0/6.0) if s.mass else zero) for s in spheres]
    worlds = [GpuSphereWorld(spheres, zero, 10.0), GpuPrimitiveWorld(primitives, zero, 10.0)]
    batches = [GpuSphereBatch([spheres[:16], spheres[:16]], zero, 10.0),
        GpuPrimitiveBatch([primitives[:16], primitives[:16]], zero, 10.0)]
    groups = GpuCollisionGroups(memberships=0xffffffff, filter=0xffffffff)
    ray = GpuRay(origin=vec(2.0, 0.0, 3.0), direction=vec(-2.0, 0.0, 0.0), max_t=1.0,
        groups=groups, excluded_body=None, body_range=None, solid=False)
    point = GpuPointQuery(point=ray.origin, max_distance=2.0, groups=groups,
        excluded_body=None, body_range=None, solid=False)
    invalid = GpuPointQuery(point=ray.origin, max_distance=-1.0, groups=groups,
        excluded_body=None, body_range=None, solid=False)
    for world in worlds + batches:
        batched = world in batches
        cast = (lambda q: world.cast_rays_environment(1, q)) if batched else world.cast_rays
        project = (lambda q: world.project_points_environment(1, q)) if batched else world.project_points
        query_scene = (lambda r, p: world.query_scene_environment(1, r, p)) if batched else world.query_scene
        combined = query_scene([ray], [point])
        assert combined.rays[0].body == 0 and abs(combined.rays[0].toi-0.75) < 1e-5
        assert combined.points[0].body == 0 and abs(combined.points[0].distance-1.5) < 1e-5
        hit = cast([ray])[0]
        assert hit.body == 0 and abs(hit.toi-0.75) < 1e-5 and abs(hit.normal.x-1.0) < 1e-5
        hit = project([point])[0]
        assert hit.body == 0 and abs(hit.distance-1.5) < 1e-5 and abs(hit.point.x-0.5) < 1e-5
        if batched:
            world.write_wrench(1, 0, vec(10.0, 0.0, 0.0), zero)
        else:
            world.write_wrench(0, vec(10.0, 0.0, 0.0), zero)
        try:
            project([invalid])
        except TesseraError:
            pass
        else:
            raise AssertionError("invalid distance was accepted")
        assert abs(cast([ray])[0].toi-0.75) < 1e-5
        world.step(0.1)
        combined = query_scene([ray], [point])
        assert abs(combined.rays[0].toi-0.65) < 1e-5
        assert abs(combined.points[0].distance-1.3) < 1e-5
        assert abs(cast([ray])[0].toi-0.65) < 1e-5
        assert abs(project([point])[0].distance-1.3) < 1e-5
        if batched:
            assert abs(world.cast_rays_environment(0, [ray])[0].toi-0.7) < 1e-5
            assert abs(world.project_points_environment(0, [point])[0].distance-1.4) < 1e-5
    print("GPU scene-query wheel passed: four interfaces, combined rays/points, 32 bodies, current-state indexes, local IDs, forces, environment isolation")

if __name__ == "__main__":
    main()
