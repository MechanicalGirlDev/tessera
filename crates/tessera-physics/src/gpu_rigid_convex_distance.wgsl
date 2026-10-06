// GJK distance with paired support witnesses for separated convex shapes.
struct DistanceWitness { a: vec3<f32>, b: vec3<f32>, delta: vec3<f32> }
struct DistanceSimplex {
    a: DistanceWitness, b: DistanceWitness, c: DistanceWitness,
    count: u32, closest: DistanceWitness,
}
struct DistanceGeometry { index: u32, surface: bool, a: vec3<f32>, b: vec3<f32>, c: vec3<f32> }
fn convex_geometry(index: u32) -> DistanceGeometry {
    return DistanceGeometry(index,false,vec3<f32>(0.0),vec3<f32>(0.0),vec3<f32>(0.0));
}
fn surface_geometry(index: u32, triangle: u32) -> DistanceGeometry {
    let state=states[index]; let vertices=convex_edges[triangle];
    return DistanceGeometry(index,true,
        rotate(state.orientation,convex_vertices[vertices.x].xyz),
        rotate(state.orientation,convex_vertices[vertices.y].xyz),
        rotate(state.orientation,convex_vertices[vertices.z].xyz));
}
fn geometry_support(g: DistanceGeometry, direction: vec3<f32>, origin: vec3<f32>, offset: vec3<f32>) -> vec3<f32> {
    let center=states[g.index].position_inverse_mass.xyz-origin;
    if (!g.surface) { return primitive_support_center(g.index,direction,center-offset); }
    var best=g.a;
    if (dot(g.b,direction)>dot(best,direction)) { best=g.b; }
    if (dot(g.c,direction)>dot(best,direction)) { best=g.c; }
    return center+(best-offset);
}
fn geometry_center(g: DistanceGeometry, origin: vec3<f32>, offset: vec3<f32>) -> vec3<f32> {
    let center=states[g.index].position_inverse_mass.xyz-origin;
    if (g.surface) { return center+((g.a-offset)+(g.b-offset)+(g.c-offset))/3.0; }
    return center-offset;
}
fn distance_support(a: DistanceGeometry, b: DistanceGeometry, direction: vec3<f32>, origin: vec3<f32>, offset: vec3<f32>) -> DistanceWitness {
    let point_a=geometry_support(a,direction,origin,offset);
    let point_b=geometry_support(b,-direction,origin,offset);
    return DistanceWitness(point_a,point_b,point_a-point_b);
}
fn witness_delta(w: DistanceWitness) -> vec3<f32> { return w.delta; }
fn witness_mix(a: DistanceWitness, b: DistanceWitness, t: f32) -> DistanceWitness {
    return DistanceWitness(mix(a.a,b.a,t), mix(a.b,b.b,t), mix(a.delta,b.delta,t));
}
fn distance_vertex(w: DistanceWitness) -> DistanceSimplex {
    return DistanceSimplex(w,w,w,1u,w);
}
fn distance_choose(best: DistanceSimplex, candidate: DistanceSimplex) -> DistanceSimplex {
    if (dot(witness_delta(candidate.closest), witness_delta(candidate.closest)) <
        dot(witness_delta(best.closest), witness_delta(best.closest))) { return candidate; }
    return best;
}
fn distance_edge(a: DistanceWitness, b: DistanceWitness) -> DistanceSimplex {
    let start = witness_delta(a);
    let edge = witness_delta(b) - start;
    let length_squared = dot(edge,edge);
    if (length_squared <= 1e-20) { return distance_vertex(a); }
    let t = clamp(-dot(start,edge)/length_squared,0.0,1.0);
    if (t <= 0.0) { return distance_vertex(a); }
    if (t >= 1.0) { return distance_vertex(b); }
    return DistanceSimplex(a,b,b,2u,witness_mix(a,b,t));
}
fn distance_triangle(a: DistanceWitness, b: DistanceWitness, c: DistanceWitness) -> DistanceSimplex {
    var best = distance_choose(distance_edge(a,b), distance_edge(a,c));
    best = distance_choose(best, distance_edge(b,c));
    let p = witness_delta(a);
    let ab = witness_delta(b)-p;
    let ac = witness_delta(c)-p;
    let aa = dot(ab,ab); let cc = dot(ac,ac); let bb = dot(ab,ac);
    let determinant = aa*cc-bb*bb;
    if (determinant <= max(aa*cc*1e-7,1e-24)) { return best; }
    let d = closest_triangle(vec3<f32>(0.0),p,p+ab,p+ac)-p;
    let v = clamp((dot(d,ab)*cc-dot(d,ac)*bb)/determinant,0.0,1.0);
    let w = clamp((dot(d,ac)*aa-dot(d,ab)*bb)/determinant,0.0,1.0-v);
    let u = 1.0-v-w;
    // Interpolate the Minkowski delta directly; subtracting mixed witnesses loses small gaps.
    let closest = DistanceWitness(a.a*u+b.a*v+c.a*w, a.b*u+b.b*v+c.b*w,
        a.delta*u+b.delta*v+c.delta*w);
    return distance_choose(best,DistanceSimplex(a,b,c,3u,closest));
}
fn distance_reduce(simplex: DistanceSimplex, next: DistanceWitness) -> DistanceSimplex {
    var best = distance_choose(distance_vertex(next),distance_edge(next,simplex.a));
    if (simplex.count >= 2u) {
        best = distance_choose(best,distance_triangle(next,simplex.a,simplex.b));
    }
    if (simplex.count >= 3u) {
        best = distance_choose(best,distance_triangle(next,simplex.a,simplex.c));
        best = distance_choose(best,distance_triangle(next,simplex.b,simplex.c));
        best = distance_choose(best,distance_triangle(simplex.a,simplex.b,simplex.c));
    }
    return best;
}
fn geometry_distance_contact(a: DistanceGeometry, b: DistanceGeometry, margin: f32) -> Contact {
    // Keep simplex witnesses near the pair instead of subtracting world-space mixtures.
    let origin=states[a.index].position_inverse_mass.xyz;
    let offset=select(vec3<f32>(0.0),a.a,a.surface);
    var direction = geometry_center(b,origin,offset)-geometry_center(a,origin,offset);
    if (dot(direction,direction) <= 1e-20) { direction=vec3<f32>(1.0,0.0,0.0); }
    var simplex = distance_vertex(distance_support(a,b,direction,origin,offset));
    var separated = false;
    var converged = false;
    var last_gap=0.0; var last_distance_squared=0.0;
    for (var iteration=0u; iteration<48u; iteration++) {
        let closest = witness_delta(simplex.closest);
        let distance_squared = dot(closest,closest);
        if (distance_squared <= 1e-16) { return miss(); }
        let next = distance_support(a,b,-closest,origin,offset);
        let projection = dot(closest,witness_delta(next));
        separated = projection > 0.0;
        last_gap=distance_squared-projection; last_distance_squared=distance_squared;
        // A separating support plane beyond the margin certifies an ordinary miss.
        if (separated && projection / sqrt(distance_squared) > margin) { return miss(); }
        // Support switching amplifies sub-ULP closest-point residuals by face extent.
        // Include f32 roundoff at the support scale instead of requiring exact cancellation.
        let distance=sqrt(distance_squared);
        let support_scale=max(distance,length(witness_delta(next)));
        let roundoff=16.0*1.1920929e-7*support_scale*distance;
        if (distance_squared-projection <= 1e-6*max(distance_squared,1e-6)+roundoff) {
            converged = true;
            break;
        }
        simplex = distance_reduce(simplex,next);
    }
    if (!separated || !converged) {
        return Contact(vec4<f32>(witness_delta(simplex.closest),0.0),
            vec4<f32>(last_gap,last_distance_squared,select(0.0,1.0,separated),0.0),vec4<f32>(0.0,0.0,1.0,0.0));
    }
    let delta = witness_delta(simplex.closest);
    let distance = length(delta);
    if (distance > margin || distance <= 1e-8) { return miss(); }
    return Contact(vec4<f32>(origin+offset+(simplex.closest.a+simplex.closest.b)*0.5,0.0),
        vec4<f32>(-delta/distance,0.0),vec4<f32>(-distance,1.0,0.0,0.0));
}
// Exact feature enumeration avoids iterative distance failures for near-parallel boxes.
fn distance_box_vertex(shape: Box, bits: u32) -> vec3<f32> {
    return shape.center + shape.axis_x * shape.half_extents.x * select(-1.0,1.0,(bits & 1u)!=0u)
        + shape.axis_y * shape.half_extents.y * select(-1.0,1.0,(bits & 2u)!=0u)
        + shape.axis_z * shape.half_extents.z * select(-1.0,1.0,(bits & 4u)!=0u);
}
fn distance_box_projection(shape: Box, point: vec3<f32>) -> vec3<f32> {
    let delta=point-shape.center;
    let local=clamp(vec3<f32>(dot(delta,shape.axis_x),dot(delta,shape.axis_y),dot(delta,shape.axis_z)),
        -shape.half_extents,shape.half_extents);
    return shape.center+shape.axis_x*local.x+shape.axis_y*local.y+shape.axis_z*local.z;
}
fn distance_box_edge(shape: Box, edge: u32) -> SegmentTriangleWitness {
    let axis=edge/4u;
    let other=edge%4u;
    let bit=1u<<axis;
    let first=(axis+1u)%3u;
    let second=(axis+2u)%3u;
    let bits=((other & 1u)<<first)+(((other>>1u)&1u)<<second);
    return SegmentTriangleWitness(distance_box_vertex(shape,bits),distance_box_vertex(shape,bits+bit));
}
fn box_distance_contact(a_index: u32, b_index: u32, margin: f32) -> Contact {
    var a=box_from(a_index); var b=box_from(b_index);
    let origin=a.center;
    b.center-=origin; a.center=vec3<f32>(0.0);
    let first=distance_box_vertex(a,0u);
    var best=SegmentTriangleWitness(first,distance_box_projection(b,first));
    for (var vertex=0u; vertex<8u; vertex++) {
        let pa=distance_box_vertex(a,vertex);
        best=nearer_surface_witness(best,SegmentTriangleWitness(pa,distance_box_projection(b,pa)));
        let pb=distance_box_vertex(b,vertex);
        best=nearer_surface_witness(best,SegmentTriangleWitness(distance_box_projection(a,pb),pb));
    }
    for (var edge_a=0u; edge_a<12u; edge_a++) {
        let ea=distance_box_edge(a,edge_a);
        for (var edge_b=0u; edge_b<12u; edge_b++) {
            let eb=distance_box_edge(b,edge_b);
            best=nearer_surface_witness(best,closest_segments(ea.segment,ea.triangle,eb.segment,eb.triangle));
        }
    }
    let delta=best.triangle-best.segment;
    let distance=length(delta);
    if (distance<=1e-8 || distance>margin) { return miss(); }
    return Contact(vec4<f32>(origin+(best.segment+best.triangle)*0.5,0.0),
        vec4<f32>(delta/distance,0.0),vec4<f32>(-distance,1.0,0.0,0.0));
}

@compute @workgroup_size(64)
fn convex_speculative_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index=id.x;
    if (index >= params.pair_count || pair_contacts[index].depth_hit.y != 0.0) { return; }
    let pair=pairs[index];
    if (!allows(collision_groups[pair.a],collision_groups[pair.b])) { return; }
    let margin=bitcast<f32>(params.padding.x);
    let surface_a=shapes[pair.a].kind.x > 5u;
    let surface_b=shapes[pair.b].kind.x > 5u;
    var contact=miss();
    if (surface_a && surface_b) { contact=surface_pair_distance(pair.a,pair.b,margin); }
    else if (surface_a) { contact=surface_convex_distance(pair.a,pair.b,margin); }
    else if (surface_b) { contact=surface_convex_distance(pair.b,pair.a,margin); contact.normal=-contact.normal; }
    else if (shapes[pair.a].kind.x==1u && shapes[pair.b].kind.x==1u) { contact=box_distance_contact(pair.a,pair.b,margin); }
    else { contact=geometry_distance_contact(convex_geometry(pair.a),convex_geometry(pair.b),margin); }
    pair_contacts[index]=contact;
    if (contact.depth_hit.y != 0.0 && (shapes[pair.a].kind.x == 1u || shapes[pair.a].kind.x == 5u) &&
        (shapes[pair.b].kind.x == 1u || shapes[pair.b].kind.x == 5u)) {
        speculative_face_manifold(index,pair.a,pair.b,contact);
    }
}

fn speculative_face_manifold(index: u32, a: u32, b: u32, primary: Contact) {
    let normal = primary.normal.xyz;
    let face_a = poly_face(a, normal);
    let face_b = poly_face(b, -normal);
    var reference = face_a;
    var incident = poly_face(b, -face_a.normal);
    if (face_b.alignment > face_a.alignment + 1e-6) {
        reference = face_b;
        incident = poly_face(a, -face_b.normal);
    }
    if (!reference.valid || !incident.valid || reference.alignment < 0.995) {
        return;
    }
    let face_depth = dot(primitive_support(a,normal)-primitive_support(b,-normal),normal);
    if (abs(face_depth-primary.depth_hit.x) > max(1e-4,abs(primary.depth_hit.x)*1e-3)) { return; }
    var polygon = incident.polygon;
    var face_center = vec3<f32>(0.0);
    for (var vertex = 0u; vertex < reference.polygon.count; vertex++) {
        face_center += reference.polygon.points[vertex];
    }
    face_center /= f32(reference.polygon.count);
    for (var edge = 0u; edge < reference.polygon.count; edge++) {
        let start = reference.polygon.points[edge];
        let end = reference.polygon.points[(edge + 1u) % reference.polygon.count];
        var side = cross(reference.normal, end - start);
        let length_squared = dot(side, side);
        if (length_squared < 1e-16) { return; }
        side *= inverseSqrt(length_squared);
        if (dot(face_center - start, side) > 0.0) { side = -side; }
        polygon = clip_face_polygon(polygon, start, side, 0.0);
        if (polygon.count == 0u) { return; }
    }
    var valid: FacePolygon;
    let tolerance = max(max(shapes[a].dimensions.x, shapes[b].dimensions.x) * 1e-5, 1e-5);
    for (var candidate = 0u; candidate < polygon.count; candidate++) {
        let point = polygon.points[candidate];
        let depth = dot(reference.polygon.points[0] - point, reference.normal);
        if (depth < -bitcast<f32>(params.padding.x)-tolerance) { continue; }
        var duplicate = false;
        for (var known = 0u; known < valid.count; known++) {
            let delta = point - valid.points[known];
            duplicate = duplicate || dot(delta, delta) <= tolerance * tolerance;
        }
        if (!duplicate) {
            set_face_point(&valid, valid.count, point);
            valid.count++;
        }
    }
    var selected: FacePolygon;
    var count = 0u;
    for (var slot = 0u; slot < 4u; slot++) {
        var best = vec3<f32>(0.0);
        var best_score = -1.0;
        for (var candidate = 0u; candidate < valid.count; candidate++) {
            let point = valid.points[candidate];
            if (slot == 0u) {
                if (best_score < 0.0 || point.x < best.x ||
                    (point.x == best.x && point.y < best.y)) {
                    best = point;
                    best_score = 0.0;
                }
                continue;
            }
            var nearest = 1e30;
            for (var known = 0u; known < count; known++) {
                let delta = point - selected.points[known];
                nearest = min(nearest, dot(delta, delta));
            }
            if (nearest > best_score) {
                best = point;
                best_score = nearest;
            }
        }
        if (best_score < 0.0 || (slot > 0u && best_score <= tolerance * tolerance)) {
            break;
        }
        set_face_point(&selected, count, best);
        let depth = dot(reference.polygon.points[0] - best, reference.normal);
        let contact = Contact(vec4<f32>(best+reference.normal*(depth*0.5),0.0),
            vec4<f32>(normal,0.0),vec4<f32>(depth,1.0,0.0,0.0));
        if (count == 0u) {
            pair_contacts[index] = contact;
        } else {
            pair_contacts[params.pair_count + index * 3u + count - 1u] = contact;
        }
        count++;
    }
}

fn distance_select(best: Contact, candidate: Contact) -> Contact {
    if (best.depth_hit.z != 0.0) { return best; }
    if (candidate.depth_hit.z != 0.0) { return candidate; }
    if (candidate.depth_hit.y != 0.0 && (best.depth_hit.y == 0.0 || candidate.depth_hit.x > best.depth_hit.x)) { return candidate; }
    return best;
}
struct DistanceBounds { lower: vec3<f32>, upper: vec3<f32> }
fn geometry_local_bounds(g: DistanceGeometry, surface: u32) -> DistanceBounds {
    let state=states[surface];
    let inverse=vec4<f32>(-state.orientation.xyz,state.orientation.w);
    let x=rotate(state.orientation,vec3<f32>(1.0,0.0,0.0));
    let y=rotate(state.orientation,vec3<f32>(0.0,1.0,0.0));
    let z=rotate(state.orientation,vec3<f32>(0.0,0.0,1.0));
    let origin=state.position_inverse_mass.xyz;
    // Scalar component writes avoid FXC forcing enclosing BVH loops to unroll.
    let lower=vec3<f32>(
        rotate(inverse,geometry_support(g,-x,origin,vec3<f32>(0.0))).x,
        rotate(inverse,geometry_support(g,-y,origin,vec3<f32>(0.0))).y,
        rotate(inverse,geometry_support(g,-z,origin,vec3<f32>(0.0))).z);
    let upper=vec3<f32>(
        rotate(inverse,geometry_support(g,x,origin,vec3<f32>(0.0))).x,
        rotate(inverse,geometry_support(g,y,origin,vec3<f32>(0.0))).y,
        rotate(inverse,geometry_support(g,z,origin,vec3<f32>(0.0))).z);
    return DistanceBounds(lower,upper);
}
fn bounds_reject(bounds: DistanceBounds, node: MeshNode, margin: f32) -> bool {
    return any(bounds.upper < node.lower-vec3<f32>(margin)) || any(bounds.lower > node.upper+vec3<f32>(margin));
}
fn nearer_surface_witness(best: SegmentTriangleWitness, candidate: SegmentTriangleWitness) -> SegmentTriangleWitness {
    let delta=best.segment-best.triangle;
    let next=candidate.segment-candidate.triangle;
    if (dot(next,next)<dot(delta,delta)) { return candidate; }
    return best;
}
fn surface_distance_contact(a: DistanceGeometry, b: DistanceGeometry, margin: f32) -> Contact {
    let origin=states[a.index].position_inverse_mass.xyz;
    let offset=a.a;
    let shift=states[b.index].position_inverse_mass.xyz-origin;
    let aa=vec3<f32>(0.0); let ab=a.b-offset; let ac=a.c-offset;
    let ba=shift+(b.a-offset); let bb=shift+(b.b-offset); let bc=shift+(b.c-offset);
    let face_b=cross(bb-ba,bc-ba);
    var best=closest_segment_triangle(aa,ab,ba,bb,bc,face_b);
    best=nearer_surface_witness(best,closest_segment_triangle(ab,ac,ba,bb,bc,face_b));
    best=nearer_surface_witness(best,closest_segment_triangle(ac,aa,ba,bb,bc,face_b));
    let face_a=cross(ab-aa,ac-aa);
    var reverse=closest_segment_triangle(ba,bb,aa,ab,ac,face_a);
    reverse=nearer_surface_witness(reverse,closest_segment_triangle(bb,bc,aa,ab,ac,face_a));
    reverse=nearer_surface_witness(reverse,closest_segment_triangle(bc,ba,aa,ab,ac,face_a));
    best=nearer_surface_witness(best,SegmentTriangleWitness(reverse.triangle,reverse.segment));
    let delta=best.triangle-best.segment;
    let squared=dot(delta,delta);
    if (!(squared>=0.0) || squared>3.402823e38) {
        return Contact(vec4<f32>(0.0),vec4<f32>(0.0),vec4<f32>(0.0,0.0,1.0,0.0));
    }
    let distance=sqrt(squared);
    if (distance<=1e-8 || distance>margin) { return miss(); }
    return Contact(vec4<f32>(origin+offset+(best.segment+best.triangle)*0.5,0.0),
        vec4<f32>(delta/distance,0.0),vec4<f32>(-distance,1.0,0.0,0.0));
}
fn surface_convex_distance(surface: u32, convex: u32, margin: f32) -> Contact {
    let geometry=convex_geometry(convex);
    let bounds=geometry_local_bounds(geometry,surface);
    let shape=shapes[surface];
    var best=miss(); var cursor=0u;
    while (cursor<shape.feature_counts.z) {
        let node=mesh_node(shape,cursor);
        if (bounds_reject(bounds,node,margin)) { cursor=node.escape; continue; }
        cursor++;
        if (node.triangle == 0xffffffffu) { continue; }
        let leaf=surface_geometry(surface,node.triangle);
        if (shapes[convex].kind.x==0u || shapes[convex].kind.x==2u) {
            best=distance_select(best,surface_round_distance(leaf,convex,margin));
        } else {
            best=distance_select(best,geometry_distance_contact(leaf,geometry,margin));
        }
    }
    return best;
}
fn surface_round_distance(leaf: DistanceGeometry, index: u32, margin: f32) -> Contact {
    let origin=states[leaf.index].position_inverse_mass.xyz;
    let center=(states[index].position_inverse_mass.xyz-origin)-leaf.a;
    var half_axis=vec3<f32>(0.0);
    if (shapes[index].kind.x==2u) {
        half_axis=rotate(states[index].orientation,vec3<f32>(0.0,0.0,shapes[index].dimensions.y));
    }
    let a=vec3<f32>(0.0); let b=leaf.b-leaf.a; let c=leaf.c-leaf.a;
    let closest=closest_segment_triangle(center-half_axis,center+half_axis,a,b,c,cross(b,c));
    let delta=closest.segment-closest.triangle;
    let squared=dot(delta,delta);
    if (!(squared>=0.0) || squared>3.402823e38) {
        return Contact(vec4<f32>(0.0),vec4<f32>(0.0),vec4<f32>(0.0,0.0,1.0,0.0));
    }
    let distance=sqrt(squared);
    let gap=distance-shapes[index].dimensions.x;
    if (gap<=0.0 || gap>margin || distance<=1e-8) { return miss(); }
    let normal=delta/distance;
    return Contact(vec4<f32>(origin+leaf.a+closest.triangle+normal*(gap*0.5),0.0),
        vec4<f32>(normal,0.0),vec4<f32>(-gap,1.0,0.0,0.0));
}
fn surface_pair_distance(a: u32, b: u32, margin: f32) -> Contact {
    var best=miss(); var first=0u;
    while (first<shapes[a].feature_counts.z) {
        let leaf=mesh_node(shapes[a],first); first++;
        if (leaf.triangle == 0xffffffffu) { continue; }
        let geometry=surface_geometry(a,leaf.triangle);
        let bounds=geometry_local_bounds(geometry,b);
        var cursor=0u;
        while (cursor<shapes[b].feature_counts.z) {
            let node=mesh_node(shapes[b],cursor);
            if (bounds_reject(bounds,node,margin)) { cursor=node.escape; continue; }
            cursor++;
            if (node.triangle == 0xffffffffu) { continue; }
            best=distance_select(best,surface_distance_contact(geometry,surface_geometry(b,node.triangle),margin));
        }
    }
    return best;
}
