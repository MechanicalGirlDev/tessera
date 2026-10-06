// Same contact-row ABI as gpu_articulated_ground_contact.wgsl.
struct ContactRow {
    indices: vec4<u32>,
    center_radius: vec4<f32>,
    center_of_mass: vec4<f32>,
    plane: vec4<f32>,
    material: vec4<f32>,
    other_center_radius: vec4<f32>,
    other_center_of_mass: vec4<f32>,
    second_axis_end: vec4<f32>,
    first_axis_end: vec4<f32>,
    impulses: vec4<f32>,
    previous_normal: vec4<f32>,
    diagnostic_first: vec4<f32>,
    diagnostic_second: vec4<f32>,
    diagnostic_first_origin: vec4<f32>,
    diagnostic_second_origin: vec4<f32>,
    prescribed_linear: vec4<f32>,
    prescribed_angular: vec4<f32>,
};
@group(0) @binding(0) var<storage, read_write> rows: array<ContactRow>;
struct System { indices: vec4<u32>, inverse: vec4<u32> };
@group(0) @binding(1) var<storage, read> systems: array<System>;
@group(0) @binding(2) var<storage, read_write> status: array<atomic<u32>>;
struct SphereOrbit { origin: vec4<f32>, linear: vec4<f32>, orientation: vec4<f32>, translation_anchor: vec4<f32> };
@group(0) @binding(3) var<storage, read_write> orbits: array<SphereOrbit>;

fn rotate_offset(offset: vec3<f32>, rotation: vec3<f32>) -> vec3<f32> {
    let angle = length(rotation);
    if (angle < 1e-6) {
        return offset + cross(rotation, offset) + 0.5 * cross(rotation, cross(rotation, offset));
    }
    let axis = rotation / angle;
    let sine = sin(angle);
    let cosine = cos(angle);
    return offset * cosine + cross(axis, offset) * sine + axis * dot(axis, offset) * (1.0 - cosine);
}

fn integrate_orientation(previous: vec4<f32>, rotation: vec3<f32>) -> vec4<f32> {
    let angle = length(rotation);
    var delta = vec4<f32>(rotation * 0.5, 1.0);
    if (angle >= 1e-6) {
        delta = vec4<f32>(rotation * (sin(angle * 0.5) / angle), cos(angle * 0.5));
    }
    // World angular velocity left-multiplies the current body orientation.
    return normalize(vec4<f32>(delta.w * previous.xyz + previous.w * delta.xyz
        + cross(delta.xyz, previous.xyz), delta.w * previous.w - dot(delta.xyz, previous.xyz)));
}

@compute @workgroup_size(64)
fn integrate(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&systems) || atomicLoad(&status[id.x]) != 0u) { return; }
    let system = systems[id.x];
    for (var index = system.indices.z; index < system.indices.z + system.indices.w; index++) {
        let row = rows[index];
        if (row.material.w == 79.0 || row.material.w == 80.0) {
            var orbit = orbits[index];
            if (orbit.origin.w == 0.0) { continue; }
            let rotation = row.prescribed_angular.xyz * row.material.y;
            let offset = rotate_offset(row.other_center_radius.xyz - orbit.origin.xyz, rotation);
            orbit.origin = vec4<f32>(orbit.origin.xyz + orbit.linear.xyz * row.material.y, 1.0);
            orbit.orientation = integrate_orientation(orbit.orientation, rotation);
            let anchor = orbit.origin.xyz + offset;
            let orientation = integrate_orientation(row.second_axis_end, rotation);
            let linear = orbit.linear.xyz + cross(row.prescribed_angular.xyz, offset);
            if (!all(abs(anchor) <= vec3<f32>(3.402823466e+38))
                || !all(abs(orbit.origin.xyz) <= vec3<f32>(3.402823466e+38))
                || !all(abs(orientation) <= vec4<f32>(3.402823466e+38))
                || !all(abs(linear) <= vec3<f32>(3.402823466e+38))) {
                atomicOr(&status[id.x], 1u); return;
            }
            orbits[index] = orbit;
            rows[index].other_center_radius = vec4<f32>(anchor, row.other_center_radius.w);
            rows[index].second_axis_end = orientation;
            rows[index].prescribed_linear = vec4<f32>(linear, row.prescribed_linear.w);
            continue;
        }
        if ((row.material.w >= 52.0 && row.material.w <= 57.0)
            || (row.material.w >= 72.0 && row.material.w <= 77.0)) {
            var orbit = orbits[index];
            if (orbit.origin.w == 0.0) { continue; }
            let translated_second = (row.material.w >= 53.0 && row.material.w <= 56.0)
                || (row.material.w >= 73.0 && row.material.w <= 76.0);
            let convex = row.material.w == 57.0 || row.material.w == 77.0;
            let previous_center = select(row.plane.xyz, row.second_axis_end.xyz, translated_second);
            let previous_rotation = select(row.first_axis_end, row.second_axis_end, convex);
            let rotation = row.prescribed_angular.xyz * row.material.y;
            let offset = rotate_offset(previous_center - orbit.origin.xyz, rotation);
            // Evaluate prescribed translation from its start, avoiding repeated world-position additions.
            if (orbit.translation_anchor.w == 0.0) {
                orbit.translation_anchor = vec4<f32>(orbit.origin.xyz, 0.0);
            }
            orbit.translation_anchor.w += row.material.y;
            orbit.origin = vec4<f32>(orbit.translation_anchor.xyz
                + orbit.linear.xyz * orbit.translation_anchor.w, 1.0);
            orbit.orientation = integrate_orientation(orbit.orientation, rotation);
            let origin = orbit.origin.xyz + offset;
            let orientation = integrate_orientation(previous_rotation, rotation);
            let linear = orbit.linear.xyz + cross(row.prescribed_angular.xyz, offset);
            if (!all(abs(origin) <= vec3<f32>(3.402823466e+38))
                || !all(abs(orbit.origin.xyz) <= vec3<f32>(3.402823466e+38))
                || !all(abs(orientation) <= vec4<f32>(3.402823466e+38))
                || !all(abs(linear) <= vec3<f32>(3.402823466e+38))) {
                atomicOr(&status[id.x], 1u); return;
            }
            orbits[index] = orbit;
            if (translated_second) {
                rows[index].second_axis_end = vec4<f32>(origin, row.second_axis_end.w);
            } else { rows[index].plane = vec4<f32>(origin, row.plane.w); }
            if (convex) { rows[index].second_axis_end = orientation; }
            else { rows[index].first_axis_end = orientation; }
            rows[index].prescribed_linear = vec4<f32>(linear, row.prescribed_linear.w);
            continue;
        }
        let prescribed_box = row.material.w == 20.0 || row.material.w == 21.0
            || row.material.w == 22.0 || row.material.w == 23.0 || row.material.w == 30.0 || row.material.w == 31.0 || row.material.w == 47.0 || row.material.w == 50.0 || row.material.w == 51.0;
        let prescribed_capsule = row.material.w == 17.0 || row.material.w == 18.0 || row.material.w == 19.0
            || row.material.w == 24.0 || row.material.w == 25.0 || row.material.w == 28.0 || row.material.w == 29.0 || (row.material.w == 46.0 || row.material.w == 81.0);
        if (prescribed_capsule) {
            var orbit = orbits[index];
            if (orbit.origin.w == 0.0) { continue; }
            var a = row.plane.xyz;
            var b = row.second_axis_end.xyz;
            if (row.material.w == 17.0) { b = row.other_center_radius.xyz; }
            if (row.material.w == 24.0 || row.material.w == 25.0) { a = row.center_radius.xyz; b = row.first_axis_end.xyz; }
            if (row.material.w == 28.0 || row.material.w == 29.0) { b = row.first_axis_end.xyz; }
            let rotation = row.prescribed_angular.xyz * row.material.y;
            let offset_a = rotate_offset(a - orbit.origin.xyz, rotation);
            let offset_b = rotate_offset(b - orbit.origin.xyz, rotation);
            orbit.origin = vec4<f32>(orbit.origin.xyz + orbit.linear.xyz * row.material.y, 1.0);
            orbit.orientation = integrate_orientation(orbit.orientation, rotation);
            a = orbit.origin.xyz + offset_a;
            b = orbit.origin.xyz + offset_b;
            let linear = orbit.linear.xyz + cross(row.prescribed_angular.xyz, (offset_a + offset_b) * 0.5);
            if (!all(abs(a) <= vec3<f32>(3.402823466e+38)) || !all(abs(b) <= vec3<f32>(3.402823466e+38))
                || !all(abs(orbit.origin.xyz) <= vec3<f32>(3.402823466e+38))
                || !all(abs(orbit.orientation) <= vec4<f32>(3.402823466e+38))
                || !all(abs(linear) <= vec3<f32>(3.402823466e+38))) {
                atomicOr(&status[id.x], 1u); return;
            }
            orbits[index] = orbit;
            rows[index].prescribed_linear = vec4<f32>(linear, row.prescribed_linear.w);
            if (row.material.w == 24.0 || row.material.w == 25.0) {
                rows[index].center_radius = vec4<f32>(a, row.center_radius.w);
                rows[index].first_axis_end = vec4<f32>(b, row.first_axis_end.w);
            } else {
                rows[index].plane = vec4<f32>(a, row.plane.w);
                if (row.material.w == 17.0) { rows[index].other_center_radius = vec4<f32>(b, row.other_center_radius.w); }
                else if (row.material.w == 28.0 || row.material.w == 29.0) { rows[index].first_axis_end = vec4<f32>(b, row.first_axis_end.w); }
                else { rows[index].second_axis_end = vec4<f32>(b, row.second_axis_end.w); }
            }
            continue;
        }
        if (row.material.w == 48.0 || row.material.w == 49.0 || row.material.w == 83.0) {
            var orbit = orbits[index];
            if (orbit.origin.w == 0.0) { continue; }
            let rotation = row.prescribed_angular.xyz * row.material.y;
            let offset = rotate_offset(row.center_radius.xyz - orbit.origin.xyz, rotation);
            orbit.origin = vec4<f32>(orbit.origin.xyz + orbit.linear.xyz * row.material.y, 1.0);
            orbit.orientation = integrate_orientation(orbit.orientation, rotation);
            let center = orbit.origin.xyz + offset;
            let linear = orbit.linear.xyz + cross(row.prescribed_angular.xyz, offset);
            let previous = row.first_axis_end;
            let orientation = integrate_orientation(previous, rotation);
            if (!all(abs(center) <= vec3<f32>(3.402823466e+38))
                || !all(abs(linear) <= vec3<f32>(3.402823466e+38))
                || !all(abs(orbit.origin.xyz) <= vec3<f32>(3.402823466e+38))
                || !all(abs(orbit.orientation) <= vec4<f32>(3.402823466e+38))
                || !all(abs(orientation) <= vec4<f32>(3.402823466e+38))) {
                atomicOr(&status[id.x], 1u); return;
            }
            orbits[index] = orbit;
            rows[index].center_radius = vec4<f32>(center, row.center_radius.w);
            rows[index].prescribed_linear = vec4<f32>(linear, row.prescribed_linear.w);
            rows[index].first_axis_end = orientation;
            continue;
        }
        if (row.material.w >= 60.0 && row.material.w <= 71.0) {
            var orbit = orbits[index];
            if (orbit.origin.w == 0.0) { continue; }
            let rotation = row.prescribed_angular.xyz * row.material.y;
            let offset = rotate_offset(row.center_radius.xyz - orbit.origin.xyz, rotation);
            orbit.origin = vec4<f32>(orbit.origin.xyz + orbit.linear.xyz * row.material.y, 1.0);
            orbit.orientation = integrate_orientation(orbit.orientation, rotation);
            let center = orbit.origin.xyz + offset;
            let linear = orbit.linear.xyz + cross(row.prescribed_angular.xyz, offset);
            let previous = select(row.first_axis_end, row.second_axis_end, row.material.w < 66.0);
            let orientation = integrate_orientation(previous, rotation);
            if (!all(abs(center) <= vec3<f32>(3.402823466e+38))
                || !all(abs(linear) <= vec3<f32>(3.402823466e+38))
                || !all(abs(orbit.origin.xyz) <= vec3<f32>(3.402823466e+38))
                || !all(abs(orbit.orientation) <= vec4<f32>(3.402823466e+38))
                || !all(abs(orientation) <= vec4<f32>(3.402823466e+38))) {
                atomicOr(&status[id.x], 1u); return;
            }
            orbits[index] = orbit;
            rows[index].center_radius = vec4<f32>(center, row.center_radius.w);
            rows[index].prescribed_linear = vec4<f32>(linear, row.prescribed_linear.w);
            if (row.material.w < 66.0) { rows[index].second_axis_end = orientation; }
            else { rows[index].first_axis_end = orientation; }
            continue;
        }
        if (prescribed_box && orbits[index].origin.w == 0.0) { continue; }
        if (!prescribed_box && row.material.w != 14.0 && row.material.w != 15.0 && row.material.w != 16.0
            && row.material.w != 26.0 && row.material.w != 27.0 && row.material.w != 45.0) { continue; }
        var previous_center = row.other_center_radius.xyz;
        let plane_center = prescribed_box || row.material.w == 16.0 || row.material.w == 26.0
            || row.material.w == 27.0 || row.material.w == 45.0;
        if (plane_center) { previous_center = row.plane.xyz; }
        var center = previous_center + row.prescribed_linear.xyz * row.material.y;
        var orbit = orbits[index];
        var linear = row.prescribed_linear.xyz;
        var box_orientation = row.second_axis_end;
        if (orbit.origin.w != 0.0) {
            let offset = rotate_offset(previous_center - orbit.origin.xyz, row.prescribed_angular.xyz * row.material.y);
            orbit.origin = vec4<f32>(orbit.origin.xyz + orbit.linear.xyz * row.material.y, 1.0);
            center = orbit.origin.xyz + offset;
            linear = orbit.linear.xyz + cross(row.prescribed_angular.xyz, offset);
            orbit.orientation = integrate_orientation(orbit.orientation, row.prescribed_angular.xyz * row.material.y);
            if (prescribed_box) {
                box_orientation = integrate_orientation(row.second_axis_end, row.prescribed_angular.xyz * row.material.y);
            }
        }
        if (!all(abs(center) <= vec3<f32>(3.402823466e+38))
            || !all(abs(orbit.origin.xyz) <= vec3<f32>(3.402823466e+38))
            || !all(abs(linear) <= vec3<f32>(3.402823466e+38))
            || !all(abs(orbit.orientation) <= vec4<f32>(3.402823466e+38))
            || (prescribed_box && !all(abs(box_orientation) <= vec4<f32>(3.402823466e+38)))) {
            atomicOr(&status[id.x], 1u);
            return;
        }
        orbits[index] = orbit;
        if (prescribed_box) { rows[index].second_axis_end = box_orientation; }
        rows[index].prescribed_linear = vec4<f32>(linear, row.prescribed_linear.w);
        if (plane_center) {
            rows[index].plane = vec4<f32>(center, row.plane.w);
        } else {
            rows[index].other_center_radius = vec4<f32>(center, row.other_center_radius.w);
        }
    }
}
