"""Point-query smoke test through an installed wheel, without body-state readback."""
from tessera3d import (
    GpuCollisionGroups, GpuPrimitiveBatch, GpuPrimitiveBody, GpuPrimitiveShape,
    GpuPrimitiveWorld, GpuPointQuery, GpuRayBodyRange, GpuSphereBatch, GpuSphereWorld,
    Quaternion, SphereInput, TesseraError, Vec3,
)


def vec(x, y, z):
    return Vec3(x=x, y=y, z=z)


def ray(origin=None, max_t=10.0, excluded=None, bounds=None, solid=False, groups=None):
    return GpuPointQuery(point=origin or vec(6.0, 0.0, 3.0),
        max_distance=max_t, groups=groups or GpuCollisionGroups(memberships=0xffffffff, filter=0xffffffff),
        excluded_body=excluded, body_range=bounds, solid=solid)


def main():
    zero = vec(0.0, 0.0, 0.0)
    identity = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    spheres = [SphereInput(center=vec(0.0, 0.0, 3.0), velocity=zero, radius=0.5, mass=0.0),
        SphereInput(center=vec(4.0, 0.0, 3.0), velocity=vec(1.0, 0.0, 0.0), radius=0.5, mass=1.0)]
    primitives = [GpuPrimitiveBody(shape=GpuPrimitiveShape.SPHERE(radius=0.5), center=spheres[0].center,
        orientation=identity, velocity=zero, angular_velocity=zero, mass=0.0, principal_inertia=zero),
        GpuPrimitiveBody(shape=GpuPrimitiveShape.BOX(half_extents=vec(0.5, 0.5, 0.5)), center=spheres[1].center,
        orientation=identity, velocity=spheres[1].velocity, angular_velocity=zero, mass=1.0, principal_inertia=vec(1.0/6.0, 1.0/6.0, 1.0/6.0))]
    worlds = [GpuSphereWorld(spheres, zero, 10.0), GpuPrimitiveWorld(primitives, zero, 10.0)]
    batches = [GpuSphereBatch([spheres, spheres], zero, 10.0), GpuPrimitiveBatch([primitives, primitives], zero, 10.0)]
    inputs = [ray(), ray(max_t=0.5), ray(excluded=1), ray(bounds=GpuRayBodyRange(start=0, end=1)),
        ray(origin=vec(0.0, 0.0, 3.0), solid=True),
        ray(origin=vec(0.0, 2.0, 1.0)),
        ray(groups=GpuCollisionGroups(memberships=0, filter=0))]
    for world in worlds + batches:
        batched = world in batches
        cast = (lambda rays: world.project_points_environment(1, rays)) if batched else world.project_points
        assert cast([]) == []
        hits = cast(inputs)
        assert hits[0].body == 1 and abs(hits[0].distance - 1.5) < 1e-5
        assert abs(hits[0].point.x - 4.5) < 1e-5
        assert abs(hits[0].normal.x - 1.0) < 1e-5
        assert hits[1] is None and hits[6] is None
        assert hits[2].body is None and abs(hits[2].distance - 3.0) < 1e-5
        assert hits[3].body is None and abs(hits[3].distance - 3.0) < 1e-5
        assert hits[4].is_inside and hits[4].distance == 0.0
        assert abs(hits[4].normal.x - 1.0) < 1e-5
        assert hits[5].body is None and abs(hits[5].distance - 1.0) < 1e-5
        assert abs(hits[5].normal.z - 1.0) < 1e-5
        if batched:
            world.write_wrench(1, 1, vec(10.0, 0.0, 0.0), zero)
        else:
            world.write_wrench(1, vec(10.0, 0.0, 0.0), zero)
        for invalid in [ray(max_t=-1.0), ray(excluded=2), ray(bounds=GpuRayBodyRange(start=0, end=3))]:
            try:
                cast([invalid])
            except TesseraError:
                pass
            else:
                raise AssertionError("invalid point query was accepted")
        # Neither failed nor successful queries consume the pending wrench.
        assert abs(cast([ray()])[0].distance - 1.5) < 1e-5
        world.step(0.1)
        assert abs(cast([ray()])[0].distance - 1.3) < 1e-5
        if batched:
            other = world.project_points_environment(0, [ray()])[0]
            assert other.body == 1 and abs(other.distance - 1.4) < 1e-5
    print("GPU point projection wheel passed: four interfaces, nearest/filter/range/inside/ground, local IDs, post-step poses, pending forces")


if __name__ == "__main__":
    main()
