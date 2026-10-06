struct State { position_inverse_mass:vec4<f32>,orientation:vec4<f32>,linear_velocity:vec4<f32>,angular_velocity:vec4<f32>,inverse_inertia_sleep:vec4<f32> }
struct Shape {kind:vec4<u32>,features:vec4<u32>,dimensions:vec4<f32>}
struct Groups {memberships:u32,mask:u32}
struct PointQuery {origin_max:vec4<f32>,direction:vec4<f32>,mask:vec4<u32>,range:vec4<u32>}
struct Hit {point_toi:vec4<f32>,normal:vec4<f32>,ids:vec4<u32>}
struct Params {counts:vec4<u32>,ground_groups:vec4<u32>}
struct LocalHit {normal_t:vec4<f32>,flags:vec4<u32>}
struct Interval {enter:f32,exit:f32,ne:vec3<f32>,nx:vec3<f32>,valid:bool,inside:bool}
@group(0) @binding(0) var<storage,read> states:array<State>;
@group(0) @binding(1) var<storage,read> shapes:array<Shape>;
@group(0) @binding(2) var<storage,read> vertices:array<vec4<f32>>;
@group(0) @binding(3) var<storage,read> edges:array<vec4<u32>>;
@group(0) @binding(4) var<storage,read> groups:array<Groups>;
@group(0) @binding(5) var<storage,read> queries:array<PointQuery>;
@group(0) @binding(6) var<storage,read_write> hits:array<Hit>;
@group(0) @binding(7) var<uniform> params:Params;
struct Node { lower:vec4<f32>,upper:vec4<f32>,left:u32,right:u32,parent:u32,first:u32,last:u32,padding:u32 }
@group(0) @binding(8) var<storage,read> scene:array<Node>;
const FAR:f32=3.402823e38;
fn rotate(q:vec4<f32>,v:vec3<f32>)->vec3<f32>{let t=2.0*cross(q.xyz,v);return v+q.w*t+cross(q.xyz,t);}

struct Projection {point:vec3<f32>,inside:bool,feature:u32}
fn segment(p:vec3<f32>,a:vec3<f32>,b:vec3<f32>)->vec3<f32>{
 let e=b-a;let n=dot(e,e);if(n==0.0){return a;}return a+e*clamp(dot(p-a,e)/n,0.0,1.0);
}
fn nearer(p:vec3<f32>,a:vec3<f32>,b:vec3<f32>)->vec3<f32>{if(length(b-p)<length(a-p)){return b;}return a;}
fn triangle(p:vec3<f32>,a:vec3<f32>,b:vec3<f32>,c:vec3<f32>)->vec3<f32>{
 let e=b-a;let f=c-a;let n=cross(e,f);let squared=dot(n,n);
 if(squared>0.0){let q=p-n*dot(p-a,n)/squared;let u=dot(cross(q-a,f),n)/squared;let v=dot(cross(e,q-a),n)/squared;
 if(u>=0.0 && v>=0.0 && u+v<=1.0){return q;}}
 return nearer(p,nearer(p,segment(p,a,b),segment(p,a,c)),segment(p,b,c));
}
fn radial(p:vec2<f32>)->vec2<f32>{let n=length(p);if(n==0.0){return vec2<f32>(1.0,0.0);}return p/n;}
fn surface(shape:Shape,p:vec3<f32>)->Projection{
 var best=Projection(vec3<f32>(FAR),false,0u);var distance=FAR;var cursor=0u;
 while(cursor<shape.features.z){
 let base=shape.features.y+cursor*3u;let lo=bitcast<vec4<f32>>(edges[base]).xyz;let hi=bitcast<vec4<f32>>(edges[base+1u]).xyz;let link=edges[base+2u];
 if(length(p-clamp(p,lo,hi))>distance){cursor=link.y;continue;}cursor++;
 if(link.x==0xffffffffu){continue;}
 let feature=edges[link.x];let a=vertices[feature.x].xyz;let b=vertices[feature.y].xyz;var q=segment(p,a,b);
 if(shape.kind.x!=7u){q=triangle(p,a,b,vertices[feature.z].xyz);}
 let d=length(q-p);let index=link.x-shape.kind.w;
 if(d<distance || (d==distance && index<best.feature)){best=Projection(q,false,index);distance=d;}
 }return best;
}
fn hull(shape:Shape,p:vec3<f32>)->Projection{
 var best=Projection(vec3<f32>(FAR),true,0u);var distance=FAR;
 let tolerance=32.0*1.1920929e-7*(1.0+shape.dimensions.x);
 // A closest boundary point lies in a face interior or on a hull edge.
 for(var face=0u;face<shape.features.x;face++){
 let face_plane=vertices[shape.kind.w+face];let n=face_plane.xyz;let offset=face_plane.w;
 let separation=dot(p,n)-offset;best.inside=best.inside && separation<=0.0;
 let q=p-n*separation;var accepted=true;
 for(var other=0u;other<shape.features.x;other++){
 let plane=vertices[shape.kind.w+other];
 if(dot(q,plane.xyz)>plane.w+tolerance){accepted=false;break;}
 }
 if(accepted){let d=length(q-p);if(d<distance){best.point=q;distance=d;}}
 }
 for(var edge=0u;edge<shape.features.z;edge++){
 let ends=edges[shape.features.y+edge];let q=segment(p,vertices[ends.x].xyz,vertices[ends.y].xyz);
 let d=length(q-p);if(d<distance){best.point=q;distance=d;}
 }return best;
}
fn project(shape:Shape,p:vec3<f32>)->Projection{
 let r=shape.dimensions.x;let h=shape.dimensions.y;var q=p;var inside=false;
 if(shape.kind.x==0u){let n=length(p);inside=n<=r;q=vec3<f32>(r,0.0,0.0);if(n>0.0){q=p*(r/n);}}
 else if(shape.kind.x==1u){let half=shape.dimensions.xyz;inside=all(abs(p)<=half);q=clamp(p,-half,half);
 if(inside){let gap=half-abs(p);var axis=0u;if(gap.y<gap.x){axis=1u;}if(gap.z<gap[axis]){axis=2u;}if(axis==0u){q.x=select(-half.x,half.x,p.x>=0.0);}else if(axis==1u){q.y=select(-half.y,half.y,p.y>=0.0);}else{q.z=select(-half.z,half.z,p.z>=0.0);}}}
 else if(shape.kind.x==2u){let center=vec3<f32>(0.0,0.0,clamp(p.z,-h,h));let delta=p-center;let n=length(delta);inside=n<=r;q=center+vec3<f32>(r,0.0,0.0);if(n>0.0){q=center+delta*(r/n);}}
 else if(shape.kind.x==3u){let direction=radial(p.xy);let disk=p.xy*min(1.0,r/max(length(p.xy),1e-30));q=vec3<f32>(direction*r,clamp(p.z,-h,h));
 q=nearer(p,q,vec3<f32>(disk,-h));q=nearer(p,q,vec3<f32>(disk,h));inside=length(p.xy)<=r && abs(p.z)<=h;}
 else if(shape.kind.x==4u){let direction=radial(p.xy);let meridian=vec3<f32>(length(p.xy),p.z,0.0);let side=segment(meridian,vec3<f32>(r,-h,0.0),vec3<f32>(0.0,h,0.0));
 q=vec3<f32>(direction*side.x,side.y);let disk=p.xy*min(1.0,r/max(length(p.xy),1e-30));q=nearer(p,q,vec3<f32>(disk,-h));inside=abs(p.z)<=h && length(p.xy)<=r*(h-p.z)/(2.0*h);}
 else if(shape.kind.x==5u){let result=hull(shape,p);q=result.point;inside=result.inside;}
 else{return surface(shape,p);}
 return Projection(q,inside,0u);
}
// Conservative origin-centred bounds, including translated local hull geometry.
fn radius_bound(shape:Shape)->f32 {
 if(shape.kind.x==1u){return length(shape.dimensions.xyz);}
 if(shape.kind.x==2u){return shape.dimensions.x+shape.dimensions.y;}
 if(shape.kind.x==3u || shape.kind.x==4u){return length(shape.dimensions.xy);}
 return shape.dimensions.x;
}
fn project_body(body:u32,query:PointQuery,old:Hit)->Hit {
 var best=old;let p=query.origin_max.xyz;let maximum=query.origin_max.w;
 if(body<query.range.x || body>=query.range.y || body==query.mask.z || (query.mask.x & groups[body].mask)==0u || (groups[body].memberships & query.mask.y)==0u){return best;}
 let state=states[body];let inverse=vec4<f32>(-state.orientation.xyz,state.orientation.w);let local=rotate(inverse,p-state.position_inverse_mass.xyz);
 let shape=shapes[body];let radius=radius_bound(shape);let limit=min(maximum,best.point_toi.w);
 let roundoff=64.0*1.1920929e-7*(1.0+radius+length(local));
 if(length(local)-radius>limit+roundoff){return best;}
 let projection=project(shape,local);
 let solid_inside=query.mask.w!=0u && projection.inside;
 let distance=select(length(projection.point-local),0.0,solid_inside);
 if(distance<=maximum && (best.ids.z==0u || distance<best.point_toi.w || (distance==best.point_toi.w && body<best.ids.x))){
 let boundary=rotate(state.orientation,projection.point)+state.position_inverse_mass.xyz;
 let world=select(boundary,p,solid_inside);
 let displacement=select(p-boundary,boundary-p,projection.inside);
 let span=length(displacement);var normal=vec3<f32>(0.0);
 if(span>0.0){normal=displacement/span;}
 best=Hit(vec4<f32>(world,distance),vec4<f32>(normal,0.0),vec4<u32>(body,projection.feature,1u,u32(projection.inside)));}
 return best;
}
fn tree_project(query:PointQuery,old:Hit)->Hit {
 var best=old;var current=0u;var previous=0xffffffffu;
 let first_leaf=params.counts.x-1u;let p=query.origin_max.xyz;
 while(current!=0xffffffffu){
 let node=scene[current];var next=node.parent;
 if(previous==node.parent){
 let distance=length(p-clamp(p,node.lower.xyz,node.upper.xyz));
 let tolerance=64.0*1.1920929e-7*(1.0+length(p)+length(node.lower.xyz)+length(node.upper.xyz));
 if(distance<=best.point_toi.w+tolerance){
 if(current>=first_leaf){best=project_body(node.left,query,best);}else{next=node.left;}
 }
 }else if(current<first_leaf && previous==node.left){next=node.right;}
 previous=current;current=next;
 }return best;
}
@compute @workgroup_size(64)
fn point_main(@builtin(global_invocation_id) id:vec3<u32>){
 if(id.x>=params.counts.y){return;}let query=queries[id.x];let p=query.origin_max.xyz;let maximum=query.origin_max.w;
 var best=Hit(vec4<f32>(0.0,0.0,0.0,maximum),vec4<f32>(0.0),vec4<u32>(0xfffffffeu,0u,0u,0u));
 if(params.ground_groups.z!=0u){best=tree_project(query,best);}
 else{for(var body=query.range.x;body<query.range.y;body++){best=project_body(body,query,best);}}
 if(params.counts.z!=0u && (query.mask.x & params.ground_groups.y)!=0u && (params.ground_groups.x & query.mask.y)!=0u){
 let extent=bitcast<f32>(params.counts.w);let q=vec3<f32>(clamp(p.xy,vec2<f32>(-extent),vec2<f32>(extent)),0.0);let distance=length(q-p);
 if(distance<=maximum && (best.ids.z==0u || distance<best.point_toi.w)){
 var normal=vec3<f32>(0.0);if(distance>0.0){normal=(p-q)/distance;}
 best=Hit(vec4<f32>(q,distance),vec4<f32>(normal,0.0),vec4<u32>(0xffffffffu,0u,1u,0u));}
 }hits[id.x]=best;
}
