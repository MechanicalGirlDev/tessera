struct State { position_inverse_mass:vec4<f32>,orientation:vec4<f32>,linear_velocity:vec4<f32>,angular_velocity:vec4<f32>,inverse_inertia_sleep:vec4<f32> }
struct Shape {kind:vec4<u32>,features:vec4<u32>,dimensions:vec4<f32>}
struct Groups {memberships:u32,mask:u32}
struct Ray {origin_max:vec4<f32>,direction:vec4<f32>,mask:vec4<u32>,range:vec4<u32>}
struct Hit {point_toi:vec4<f32>,normal:vec4<f32>,ids:vec4<u32>}
struct Params {counts:vec4<u32>,ground_groups:vec4<u32>}
struct LocalHit {normal_t:vec4<f32>,flags:vec4<u32>}
struct Interval {enter:f32,exit:f32,ne:vec3<f32>,nx:vec3<f32>,valid:bool,inside:bool}
@group(0) @binding(0) var<storage,read> states:array<State>;
@group(0) @binding(1) var<storage,read> shapes:array<Shape>;
@group(0) @binding(2) var<storage,read> vertices:array<vec4<f32>>;
@group(0) @binding(3) var<storage,read> edges:array<vec4<u32>>;
@group(0) @binding(4) var<storage,read> groups:array<Groups>;
@group(0) @binding(5) var<storage,read> rays:array<Ray>;
@group(0) @binding(6) var<storage,read_write> hits:array<Hit>;
@group(0) @binding(7) var<uniform> params:Params;
struct Node { lower:vec4<f32>,upper:vec4<f32>,left:u32,right:u32,parent:u32,first:u32,last:u32,padding:u32 }
@group(0) @binding(8) var<storage,read> scene:array<Node>;
const FAR:f32=3.402823e38;
fn rotate(q:vec4<f32>,v:vec3<f32>)->vec3<f32>{let t=2.0*cross(q.xyz,v);return v+q.w*t+cross(q.xyz,t);}
fn empty()->LocalHit{return LocalHit(vec4<f32>(0.0,0.0,0.0,FAR),vec4<u32>(0u));}
fn accept(old:LocalHit,t:f32,n:vec3<f32>,feature:u32,maximum:f32)->LocalHit {
    if (!(t>=0.0 && t<=maximum)) {return old;}
    if (old.flags.y==0u || t<old.normal_t.w || (t==old.normal_t.w && feature<old.flags.x)) {
        return LocalHit(vec4<f32>(n,t),vec4<u32>(feature,1u,0u,0u));
    }
    return old;
}
fn quadratic(aa:f32,bb:f32,cc:f32)->vec2<f32> {
    let scale=max(abs(aa),max(abs(bb),abs(cc)));
    if (!(scale>0.0 && scale<=FAR)) {return vec2<f32>(-FAR);}
    let a=aa/scale;let b=bb/scale;let c=cc/scale;
    if (a==0.0) {if(b==0.0){return vec2<f32>(-FAR);}return vec2<f32>(-c/b);}
    let det=b*b-4.0*a*c;
    if(det < -32.0*1.1920929e-7*(b*b+abs(4.0*a*c))){return vec2<f32>(-FAR);}
    let root=sqrt(max(det,0.0));let q=-0.5*(b+select(-root,root,b>=0.0));
    if(q==0.0){return vec2<f32>(-b/(2.0*a));}return vec2<f32>(q/a,c/q);
}
fn sphere_hits(old:LocalHit,o:vec3<f32>,d:vec3<f32>,center:vec3<f32>,r:f32,maximum:f32,clip:i32)->LocalHit {
    // Closest approach avoids discriminant cancellation for distant tangencies.
    let v=o-center;let dd=dot(d,d);let closest=-dot(v,d)/dd;
    let perpendicular=v+d*closest;let squared=dot(perpendicular,perpendicular);
    let gap=r*r-squared;let tolerance=32.0*1.1920929e-7*(r*r+squared);
    if(gap < -tolerance){return old;}
    let span=sqrt(max(gap,0.0)/dd);let ts=vec2<f32>(closest-span,closest+span);var best=old;
    for(var i=0u;i<2u;i++){let t=ts[i];let p=o+d*t;
        if(clip==0 || (clip>0 && p.z>=center.z) || (clip<0 && p.z<=center.z)){
            best=accept(best,t,(p-center)/r,0u,maximum);
        }
    }return best;
}
fn clip_plane(old:Interval,o:vec3<f32>,d:vec3<f32>,n:vec3<f32>,offset:f32)->Interval {
    var next=old;let distance=dot(n,o)-offset;let speed=dot(n,d);
    next.inside=next.inside && distance<=0.0;
    if(speed==0.0){next.valid=next.valid && distance<=0.0;return next;}
    let t=-distance/speed;
    if(speed<0.0 && t>next.enter){next.enter=t;next.ne=n;}
    if(speed>0.0 && t<next.exit){next.exit=t;next.nx=n;}
    next.valid=next.valid && next.enter<=next.exit;return next;
}
fn bounds_hit(o:vec3<f32>,d:vec3<f32>,lower:vec3<f32>,upper:vec3<f32>,maximum:f32)->bool {
    var near=0.0;var far=maximum;
    for(var axis=0u;axis<3u;axis++){
        if(d[axis]==0.0){if(o[axis]<lower[axis] || o[axis]>upper[axis]){return false;}}
        else{let a=(lower[axis]-o[axis])/d[axis];let b=(upper[axis]-o[axis])/d[axis];near=max(near,min(a,b));far=min(far,max(a,b));}
    }return near<=far;
}
fn surface_hits(shape:Shape,o:vec3<f32>,d:vec3<f32>,maximum:f32)->LocalHit {
    var best=empty();var cursor=0u;
    while(cursor<shape.features.z){
        let base=shape.features.y+cursor*3u;let lower=bitcast<vec4<f32>>(edges[base]).xyz;
        let upper=bitcast<vec4<f32>>(edges[base+1u]).xyz;let link=edges[base+2u];
        let limit=min(maximum,best.normal_t.w);
        // Segment intersection accepts a numerical residual, so its BVH must
        // conservatively retain those same candidates after pose roundoff.
        var margin=0.0;
        if(shape.kind.x==7u){margin=64.0*1.1920929e-7*(1.0+length(o)+2.0*length(max(abs(lower),abs(upper))));}
        if(!bounds_hit(o,d,lower-vec3<f32>(margin),upper+vec3<f32>(margin),limit)){cursor=link.y;continue;}
        cursor++;
        if(link.x==0xffffffffu){continue;}
        let feature=edges[link.x];let a=vertices[feature.x].xyz;let b=vertices[feature.y].xyz;
        if(shape.kind.x==7u){
            let e=b-a;let w=o-a;let product=cross(d,e);let denominator=dot(product,product);
            let tolerance=64.0*1.1920929e-7*(1.0+length(o)+length(a)+length(b));
            if(denominator>0.0){let t=dot(cross(-w,e),product)/denominator;let u=dot(cross(-w,d),product)/denominator;
                if(u>=0.0 && u<=1.0 && length(o+d*t-a-e*u)<=tolerance){best=accept(best,t,vec3<f32>(0.0),link.x-shape.kind.w,maximum);}
            }else if(length(cross(w,d))<=tolerance*length(d)){
                let t0=dot(a-o,d)/dot(d,d);let t1=dot(b-o,d)/dot(d,d);
                if(max(t0,t1)>=0.0){best=accept(best,max(0.0,min(t0,t1)),vec3<f32>(0.0),link.x-shape.kind.w,maximum);}
            }
        }else{
            let c=vertices[feature.z].xyz;let e1=b-a;let e2=c-a;let p=cross(d,e2);let det=dot(e1,p);
            if(abs(det)<=16.0*1.1920929e-7*length(e1)*length(e2)*length(d)){continue;}
            let delta=o-a;let u=dot(delta,p)/det;let q=cross(delta,e1);let v=dot(d,q)/det;
            if(u>=0.0 && v>=0.0 && u+v<=1.0){var n=normalize(cross(e1,e2));if(dot(n,d)>0.0){n=-n;}
                best=accept(best,dot(e2,q)/det,n,link.x-shape.kind.w,maximum);
            }
        }
    }return best;
}
fn local_cast(shape:Shape,o:vec3<f32>,d:vec3<f32>,maximum:f32,solid:bool)->LocalHit {
    var best=empty();var inside=false;let r=shape.dimensions.x;let h=shape.dimensions.y;
    if(shape.kind.x==0u){inside=dot(o,o)<=r*r;best=sphere_hits(best,o,d,vec3<f32>(0.0),r,maximum,0);}
    else if(shape.kind.x==1u || shape.kind.x==5u){
        var interval=Interval(-FAR,FAR,vec3<f32>(0.0),vec3<f32>(0.0),true,true);
        if(shape.kind.x==1u){
            interval=clip_plane(interval,o,d,vec3<f32>(1.0,0.0,0.0),shape.dimensions.x);
            interval=clip_plane(interval,o,d,vec3<f32>(-1.0,0.0,0.0),shape.dimensions.x);
            interval=clip_plane(interval,o,d,vec3<f32>(0.0,1.0,0.0),shape.dimensions.y);
            interval=clip_plane(interval,o,d,vec3<f32>(0.0,-1.0,0.0),shape.dimensions.y);
            interval=clip_plane(interval,o,d,vec3<f32>(0.0,0.0,1.0),shape.dimensions.z);
            interval=clip_plane(interval,o,d,vec3<f32>(0.0,0.0,-1.0),shape.dimensions.z);
        }else{
            for(var face=0u;face<shape.features.x;face++){
                let n=vertices[shape.kind.w+face].xyz;var offset=-FAR;
                for(var vertex=0u;vertex<shape.kind.z;vertex++){offset=max(offset,dot(n,vertices[shape.kind.y+vertex].xyz));}
                interval=clip_plane(interval,o,d,n,offset);
            }
        }
        if(interval.valid){inside=interval.inside;best=accept(best,interval.enter,interval.ne,0u,maximum);best=accept(best,interval.exit,interval.nx,0u,maximum);}
    }else if(shape.kind.x<=4u){
        let cone=shape.kind.x==4u;let capsule=shape.kind.x==2u;
        if(capsule){let offset=o-vec3<f32>(0.0,0.0,clamp(o.z,-h,h));inside=dot(offset,offset)<=r*r;
            best=sphere_hits(best,o,d,vec3<f32>(0.0,0.0,h),r,maximum,1);
            best=sphere_hits(best,o,d,vec3<f32>(0.0,0.0,-h),r,maximum,-1);
        }else{let local_r=select(r,r*(h-o.z)/(2.0*h),cone);inside=abs(o.z)<=h && dot(o.xy,o.xy)<=local_r*local_r;}
        let k2=select(0.0,(r/(2.0*h))*(r/(2.0*h)),cone);
        let ts=quadratic(dot(d.xy,d.xy)-k2*d.z*d.z,2.0*(dot(o.xy,d.xy)+k2*(h-o.z)*d.z),dot(o.xy,o.xy)-select(r*r,k2*(h-o.z)*(h-o.z),cone));
        for(var i=0u;i<2u;i++){let p=o+d*ts[i];if(abs(p.z)<=h){var n=vec3<f32>(p.xy,k2*(h-p.z));if(dot(n,n)>0.0){n=normalize(n);}else{n=vec3<f32>(0.0,0.0,1.0);}best=accept(best,ts[i],n,0u,maximum);}}
        if(!capsule && d.z!=0.0){
            let bottom=(-h-o.z)/d.z;let p=o+d*bottom;
            if(dot(p.xy,p.xy)<=r*r){best=accept(best,bottom,vec3<f32>(0.0,0.0,-1.0),0u,maximum);}
            let top=(h-o.z)/d.z;let q=o+d*top;let rr=select(r*r,0.0,cone);
            if(dot(q.xy,q.xy)<=rr){best=accept(best,top,vec3<f32>(0.0,0.0,1.0),0u,maximum);}
        }
    }else{best=surface_hits(shape,o,d,maximum);}
    if(solid && inside){return LocalHit(vec4<f32>(0.0),vec4<u32>(0u,1u,1u,0u));}return best;
}
fn cast_body(body:u32,ray:Ray,old:Hit)->Hit {
    var best=old;let o=ray.origin_max.xyz;let d=ray.direction.xyz;let maximum=ray.origin_max.w;
        if(body<ray.range.x || body>=ray.range.y || body==ray.mask.z || (ray.mask.x & groups[body].mask)==0u || (groups[body].memberships & ray.mask.y)==0u){return best;}
        let state=states[body];let inverse=vec4<f32>(-state.orientation.xyz,state.orientation.w);
        let local=local_cast(shapes[body],rotate(inverse,o-state.position_inverse_mass.xyz),rotate(inverse,d),min(maximum,best.point_toi.w),ray.mask.w!=0u);
        if(local.flags.y!=0u && (best.ids.z==0u || local.normal_t.w<best.point_toi.w || (local.normal_t.w==best.point_toi.w && body<best.ids.x))){
            best=Hit(vec4<f32>(o+d*local.normal_t.w,local.normal_t.w),vec4<f32>(rotate(state.orientation,local.normal_t.xyz),0.0),vec4<u32>(body,local.flags.x,1u,local.flags.z));
        }
    return best;
}
fn tree_cast(ray:Ray,old:Hit)->Hit {
    var best=old;var current=0u;var previous=0xffffffffu;
    let first_leaf=params.counts.x-1u;
    while(current!=0xffffffffu){
        let node=scene[current];var next=node.parent;
        if(previous==node.parent){
            let tolerance=64.0*1.1920929e-7*(1.0+length(node.lower.xyz)+length(node.upper.xyz));
            if(bounds_hit(ray.origin_max.xyz,ray.direction.xyz,node.lower.xyz-vec3<f32>(tolerance),node.upper.xyz+vec3<f32>(tolerance),best.point_toi.w)){
                if(current>=first_leaf){best=cast_body(node.left,ray,best);}else{next=node.left;}
            }
        }else if(current<first_leaf && previous==node.left){next=node.right;}
        previous=current;current=next;
    }return best;
}
@compute @workgroup_size(64)
fn ray_main(@builtin(global_invocation_id) id:vec3<u32>){
    if(id.x>=params.counts.y){return;}
    let ray=rays[id.x];let o=ray.origin_max.xyz;let d=ray.direction.xyz;let maximum=ray.origin_max.w;
    var best=Hit(vec4<f32>(0.0,0.0,0.0,maximum),vec4<f32>(0.0),vec4<u32>(0xfffffffeu,0u,0u,0u));
    if(params.ground_groups.z!=0u){best=tree_cast(ray,best);}
    else{for(var body=ray.range.x;body<ray.range.y;body++){best=cast_body(body,ray,best);}}
    if(params.counts.z!=0u && d.z!=0.0 && (ray.mask.x & params.ground_groups.y)!=0u && (params.ground_groups.x & ray.mask.y)!=0u){
        let t=-o.z/d.z;let p=o+d*t;let extent=bitcast<f32>(params.counts.w);
        if(t>=0.0 && t<=maximum && abs(p.x)<=extent && abs(p.y)<=extent && (best.ids.z==0u || t<best.point_toi.w)){
            best=Hit(vec4<f32>(p,t),vec4<f32>(0.0,0.0,select(-1.0,1.0,d.z<0.0),0.0),vec4<u32>(0xffffffffu,0u,1u,0u));
        }
    }hits[id.x]=best;
}
