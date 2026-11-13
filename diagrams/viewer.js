const VS = `attribute vec3 p; attribute vec3 n; uniform mat4 M, VP; uniform mat3 NM; varying vec3 vn; varying vec3 wp;
  void main(){ vec4 w = M*vec4(p,1.0); wp = w.xyz; vn = NM*n; gl_Position = VP*w; }`;
const FS = `precision mediump float; varying vec3 vn; varying vec3 wp; uniform vec4 col; uniform int mode; uniform vec3 L;
  void main(){ if(mode==1){ gl_FragColor = vec4(0.0,0.0,0.0,0.28); return; }
    vec3 c = col.rgb;
    if(mode==2){ gl_FragColor = col; return; }
    float d = max(dot(normalize(vn), -L), 0.0); gl_FragColor = vec4(c*(0.42+0.58*d), col.a); }`;
function norm(v){ const l=Math.hypot(...v)||1; return v.map(x=>x/l); }
function cross(a,b){ return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }
function dot(a,b){ return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
function sphereLike(rx, ry, rz, zoff, half){ // half: 0 full, 1 upper (z>=0), -1 lower
  const P=[], Q=[], nu=28, nv=14;
  const th = (j)=> half===1 ? (j/nv)*Math.PI/2 : half===-1 ? Math.PI/2 + (j/nv)*Math.PI/2 : (j/nv)*Math.PI;
  const pt = (i,j)=>{ const t=th(j), f=2*Math.PI*i/nu; const x=Math.sin(t)*Math.cos(f), y=Math.sin(t)*Math.sin(f), z=Math.cos(t);
    return [[rx*x, ry*y, rz*z+zoff], norm([x/rx, y/ry, z/rz])]; };
  for(let j=0;j<nv;j++) for(let i=0;i<nu;i++){ const a=pt(i,j), b=pt(i+1,j), c=pt(i+1,j+1), d=pt(i,j+1);
    for(const q of [a,b,c,a,c,d]){ P.push(...q[0]); Q.push(...q[1]); } }
  return [P,Q];
}
function cylinder(r, h){ const P=[],Q=[],nu=36;
  for(let i=0;i<nu;i++){ const f0=2*Math.PI*i/nu, f1=2*Math.PI*(i+1)/nu, c0=Math.cos(f0), s0=Math.sin(f0), c1=Math.cos(f1), s1=Math.sin(f1);
    const q=[[r*c0,r*s0,-h],[r*c1,r*s1,-h],[r*c1,r*s1,h],[r*c0,r*s0,h]], nn=[[c0,s0,0],[c1,s1,0],[c1,s1,0],[c0,s0,0]];
    for(const k of [0,1,2,0,2,3]){ P.push(...q[k]); Q.push(...nn[k]); }
    for(const z of [h,-h]){ const tri=[[0,0,z],[r*c0,r*s0,z],[r*c1,r*s1,z]]; for(const v of (z>0?tri:[tri[0],tri[2],tri[1]])){ P.push(...v); Q.push(0,0,Math.sign(z)); } } }
  return [P,Q]; }
function capsule(r, h){ const [P1,N1]=cylinder(r,h); const [P2,N2]=sphereLike(r,r,r,h,1); const [P3,N3]=sphereLike(r,r,r,-h,-1);
  return [P1.concat(P2,P3), N1.concat(N2,N3)]; }
function box(a,b,c){ const P=[],Q=[]; const F=[[[1,0,0],[0,1,0],[0,0,1]],[[-1,0,0],[0,0,1],[0,1,0]],[[0,1,0],[0,0,1],[1,0,0]],[[0,-1,0],[1,0,0],[0,0,1]],[[0,0,1],[1,0,0],[0,1,0]],[[0,0,-1],[0,1,0],[1,0,0]]];
  const s=[a,b,c]; for(const [n,u,v] of F){ const cpt=(su,sv)=>[0,1,2].map(k=>n[k]*s[k]+su*u[k]*s[k]+sv*v[k]*s[k]);
    const q=[cpt(-1,-1),cpt(1,-1),cpt(1,1),cpt(-1,1)]; for(const k of [0,1,2,0,2,3]){ P.push(...q[k]); Q.push(...n); } }
  return [P,Q]; }
function geomMesh(g){ const s=g.size; switch(g.type){
  case 'box': return box(s[0],s[1],s[2]); case 'cylinder': return cylinder(s[0],s[1]); case 'capsule': return capsule(s[0],s[1]);
  case 'ellipsoid': return sphereLike(s[0],s[1],s[2],0,0); default: return sphereLike(s[0],s[0],s[0],0,0); } }
function geomRadius(g){ const s=g.size; switch(g.type){ case 'box': return Math.hypot(s[0],s[1],s[2]); case 'cylinder': return Math.hypot(s[0],s[1]);
  case 'capsule': return s[0]+s[1]; case 'ellipsoid': return Math.max(s[0],s[1],s[2]); default: return s[0]; } }
// matrices (column-major 4x4); R is a row-major 3x3
function mat(R,t){ return [R[0],R[3],R[6],0, R[1],R[4],R[7],0, R[2],R[5],R[8],0, t[0],t[1],t[2],1]; }
function mul(a,b){ const o=new Array(16).fill(0); for(let i=0;i<4;i++) for(let j=0;j<4;j++) for(let k=0;k<4;k++) o[j*4+i]+=a[k*4+i]*b[j*4+k]; return o; }
function persp(f,asp,n,fr){ const t=1/Math.tan(f/2); return [t/asp,0,0,0, 0,t,0,0, 0,0,(fr+n)/(n-fr),-1, 0,0,2*fr*n/(n-fr),0]; }
function lookAt(e,c,u){ const z=norm([e[0]-c[0],e[1]-c[1],e[2]-c[2]]), x=norm(cross(u,z)), y=cross(z,x);
  return [x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0, -dot(x,e),-dot(y,e),-dot(z,e),1]; }
const LIGHT = norm([-0.35,-0.45,-1.0]), FOV = 0.75;
const SHADOW = [1,0,0,0, 0,1,0,0, -LIGHT[0]/LIGHT[2],-LIGHT[1]/LIGHT[2],0,0, 0,0,0.0003,1];
function makeRenderer(cv){
  const gl = cv.getContext('webgl', {antialias: true}); if(!gl) return null;
  const sh = (t, src) => { const o = gl.createShader(t); gl.shaderSource(o, src); gl.compileShader(o); return o; };
  const prog = gl.createProgram(); gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FS)); gl.linkProgram(prog); gl.useProgram(prog);
  const U = n => gl.getUniformLocation(prog, n), A = n => gl.getAttribLocation(prog, n);
  const buf = arr => { const b=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,b); gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(arr),gl.STATIC_DRAW); return b; };
  const MESH = {}; for(const name of ['mug','spoon']) MESH[name] = BODIES[name].map(g=>{ const [P,Q]=geomMesh(g); return {p:buf(P), n:buf(Q), cnt:P.length/3, g}; });
  const r = 0.6, tableMesh = {p:buf([-r,-r,0, r,-r,0, r,r,0, -r,-r,0, r,r,0, -r,r,0]), n:buf([0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1]), cnt:6};
  const bind = m => { gl.bindBuffer(gl.ARRAY_BUFFER,m.p); gl.enableVertexAttribArray(A('p')); gl.vertexAttribPointer(A('p'),3,gl.FLOAT,false,0,0);
    gl.bindBuffer(gl.ARRAY_BUFFER,m.n); gl.enableVertexAttribArray(A('n')); gl.vertexAttribPointer(A('n'),3,gl.FLOAT,false,0,0); };
  return function(x, view, wpx, hpx){
    if(cv.width!==wpx||cv.height!==hpx){ cv.width=wpx; cv.height=hpx; }
    gl.viewport(0,0,wpx,hpx); gl.clearColor(0,0,0,0); gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT); gl.enable(gl.DEPTH_TEST);
    const tg=view.target, e=[tg[0]+view.dist*Math.cos(view.el)*Math.cos(view.az), tg[1]+view.dist*Math.cos(view.el)*Math.sin(view.az), tg[2]+view.dist*Math.sin(view.el)];
    const VP = mul(persp(FOV, wpx/hpx, 0.005, 5), lookAt(e, tg, [0,0,1]));
    gl.uniformMatrix4fv(U('VP'), false, VP); gl.uniform3fv(U('L'), LIGHT);
    if(!x.outlineMask){
    gl.uniform1i(U('mode'),2); gl.uniform4fv(U('col'), getComputedStyle(document.body).colorScheme==='dark'?[.10,.11,.12,1]:[.95,.953,.957,1]); gl.uniformMatrix4fv(U('M'),false,[1,0,0,0,0,1,0,0,0,0,1,0,tg[0],tg[1],0,1]);
    gl.uniformMatrix3fv(U('NM'),false,[1,0,0,0,1,0,0,0,1]); bind(tableMesh); gl.drawArrays(gl.TRIANGLES,0,tableMesh.cnt);
    }
    const parts=[];
    ['mug','spoon'].forEach((name,b)=>{ if(x.outlineBody!==undefined&&b!==x.outlineBody)return; const P=x.pose[b], Rb=P.slice(0,9), tb=P.slice(9);
      MESH[name].forEach(m=>{ parts.push({m, M: mul(mat(Rb, tb), mat(m.g.R, m.g.pos)), col: name==='mug' ? [0.36,0.42,0.48,1] : [0.75,0.39,0.17,1]}); }); });
    gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA); gl.depthMask(false); gl.uniform1i(U('mode'),1);
    if(!x.outlineMask) for(const q of parts){ gl.uniformMatrix4fv(U('M'),false,mul(SHADOW,q.M)); bind(q.m); gl.drawArrays(gl.TRIANGLES,0,q.m.cnt); }
    gl.depthMask(true); gl.disable(gl.BLEND); gl.uniform1i(U('mode'),0);
    for(const q of parts){ const M=q.M; gl.uniformMatrix4fv(U('M'),false,M); gl.uniformMatrix3fv(U('NM'),false,[M[0],M[1],M[2],M[4],M[5],M[6],M[8],M[9],M[10]]);
      gl.uniform4fv(U('col'),q.col); bind(q.m); gl.drawArrays(gl.TRIANGLES,0,q.m.cnt); }
  };
}
// framing of one scene: centre and radius of a sphere that contains both bodies
function sceneBounds(x){
  const cs=[]; ['mug','spoon'].forEach((name,b)=>{ const P=x.pose[b], R=P.slice(0,9), t=P.slice(9);
    BODIES[name].forEach(g=>{ const q=g.pos; cs.push([[R[0]*q[0]+R[1]*q[1]+R[2]*q[2]+t[0], R[3]*q[0]+R[4]*q[1]+R[5]*q[2]+t[1], R[6]*q[0]+R[7]*q[1]+R[8]*q[2]+t[2]], geomRadius(g)]); }); });
  const lo=[1e9,1e9,1e9], hi=[-1e9,-1e9,-1e9];
  for(const [c,r] of cs) for(let k=0;k<3;k++){ lo[k]=Math.min(lo[k],c[k]-r); hi[k]=Math.max(hi[k],c[k]+r); }
  const ctr=[(lo[0]+hi[0])/2,(lo[1]+hi[1])/2,Math.max(0,(lo[2]+hi[2])/2)]; let rad=0;
  for(const [c,r] of cs) rad=Math.max(rad, Math.hypot(c[0]-ctr[0],c[1]-ctr[1],c[2]-ctr[2])+r);
  const tm=x.pose[0].slice(9), ts=x.pose[1].slice(9), d=Math.hypot(ts[0]-tm[0], ts[1]-tm[1]);
  return {ctr, rad, az: (d>1e-6 ? Math.atan2(ts[1]-tm[1], ts[0]-tm[0]) : 0) - Math.PI/2};   // camera to the side of the mug-spoon line
}

// Shared viewer for explanatory figures; no optimizer or physical transitions.
class Q13DiagramView {
  constructor(host, options={}) {
    this.canvas=host.querySelector('canvas.scene');this.overlay=host.querySelector('canvas.overlay');
    this.ctx=this.overlay.getContext('2d');this.render=makeRenderer(this.canvas);this.host=host;
    this.view={az:-.9,el:.65,dist:.65,target:[.045,-.025,.045],...options};
    this.pose=null;this.decorate=null;this.queued=false;
    let drag=null;this.canvas.tabIndex=0;
    this.canvas.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY];this.canvas.setPointerCapture(e.pointerId);});
    this.canvas.addEventListener('pointermove',e=>{if(!drag)return;this.view.az-=(e.clientX-drag[0])*.009;this.view.el=Math.max(.12,Math.min(1.5,this.view.el+(e.clientY-drag[1])*.007));drag=[e.clientX,e.clientY];this.draw();});
    for(const event of ['pointerup','pointercancel'])this.canvas.addEventListener(event,()=>{drag=null;});
    this.canvas.addEventListener('keydown',e=>{const a={ArrowLeft:[-.12,0],ArrowRight:[.12,0],ArrowUp:[0,.1],ArrowDown:[0,-.1]}[e.key];if(a){e.preventDefault();this.view.az+=a[0];this.view.el=Math.max(.12,Math.min(1.5,this.view.el+a[1]));this.draw();}});
    new ResizeObserver(()=>this.draw()).observe(host);
    matchMedia('(prefers-color-scheme:dark)').addEventListener('change',()=>this.draw());
  }
  set(pose,decorate){this.pose=pose;this.decorate=decorate;this.draw();}
  draw(){if(!this.pose||this.queued)return;this.queued=true;requestAnimationFrame(()=>this.paint());}
  paint(){
    this.queued=false;const box=this.host.getBoundingClientRect(),w=box.width,h=box.height,dpr=Math.min(2,devicePixelRatio||1);
    this.w=w;this.h=h;const v={...this.view,dist:this.view.dist*Math.max(1,1.05/(w/h))};
    if(!this.render){this.host.setAttribute('data-error','This diagram requires WebGL.');return;}
    this.render({pose:this.pose},v,Math.round(w*dpr),Math.round(h*dpr));
    this.overlay.width=Math.round(w*dpr);this.overlay.height=Math.round(h*dpr);this.ctx.setTransform(dpr,0,0,dpr,0,0);this.ctx.clearRect(0,0,w,h);
    const t=v.target,e=[t[0]+v.dist*Math.cos(v.el)*Math.cos(v.az),t[1]+v.dist*Math.cos(v.el)*Math.sin(v.az),t[2]+v.dist*Math.sin(v.el)];
    this.VP=mul(persp(FOV,w/h,.005,5),lookAt(e,t,[0,0,1]));
    this.ink=getComputedStyle(document.body).color;this.bg=getComputedStyle(document.body).backgroundColor;
    if(this.decorate)this.decorate(this);
  }
  project(p){const q=[...p,1],z=[0,1,2,3].map(i=>q.reduce((s,x,k)=>s+this.VP[k*4+i]*x,0));return[(z[0]/z[3]+1)*this.w/2,(1-z[1]/z[3])*this.h/2];}
  line(a,b,color=this.ink,dash=[],width=1.5){const p=this.project(a),q=this.project(b),c=this.ctx;c.strokeStyle=color;c.lineWidth=width;c.setLineDash(dash);c.beginPath();c.moveTo(...p);c.lineTo(...q);c.stroke();c.setLineDash([]);return[p,q];}
  arrow(a,b,color=this.ink,dash=[]){const[p,q]=this.line(a,b,color,dash),d=[q[0]-p[0],q[1]-p[1]],l=Math.hypot(...d);if(l<7)return;const u=d.map(x=>x/l),c=this.ctx;c.strokeStyle=color;c.beginPath();c.moveTo(q[0]-7*u[0]+3*u[1],q[1]-7*u[1]-3*u[0]);c.lineTo(...q);c.lineTo(q[0]-7*u[0]-3*u[1],q[1]-7*u[1]+3*u[0]);c.stroke();}
  dot(p,color=this.ink,hollow=false,size=3.3){const q=this.project(p),c=this.ctx;c.beginPath();c.arc(...q,size,0,2*Math.PI);c.fillStyle=hollow?this.bg:color;c.fill();c.lineWidth=1.4;c.strokeStyle=color;c.stroke();}
  label(p,text,dx=7,dy=-7,color=this.ink){const q=this.project(p),c=this.ctx;c.font='13px system-ui, sans-serif';const width=c.measureText(text).width;const x=Math.max(4,Math.min(this.w-width-4,q[0]+dx)),y=Math.max(16,Math.min(this.h-8,q[1]+dy));c.fillStyle=this.bg;c.fillRect(x-2,y-13,width+4,17);c.fillStyle=color;c.fillText(text,x,y);}
  initialOutlines(pose,colors){
    // Reference geometry is cached until the release scene, camera or size changes.
    // Render each initial body without table/shadows; its alpha silhouette gives
    // the actual projected outline, including the handle opening.
    const dpr=Math.min(2,devicePixelRatio||1),w=this.canvas.width,h=this.canvas.height;
    const key=JSON.stringify([pose,this.view,w,h,colors]);
    if(this.outlineKey!==key){
      if(!this.outlineRenderer){this.outlineMask=document.createElement('canvas');this.outlineRenderer=makeRenderer(this.outlineMask);}
      if(!this.outlineRenderer)return;
      if(!this.outlineImage){this.outlineImage=document.createElement('canvas');this.outlineScratch=document.createElement('canvas');this.outlineEdge=document.createElement('canvas');}
      const mask=this.outlineScratch,edge=this.outlineEdge;
      for(const cv of [this.outlineImage,mask,edge])if(cv.width!==w||cv.height!==h){cv.width=w;cv.height=h;}
      const result=this.outlineImage.getContext('2d');result.clearRect(0,0,w,h);
      const m=mask.getContext('2d'),e=edge.getContext('2d'),offset=Math.max(1,Math.round(1.2*dpr));
      const camera={...this.view,dist:this.view.dist*Math.max(1,1.05/(this.w/this.h))};
      for(let b=0;b<2;b++){
        this.outlineRenderer({pose,outlineMask:true,outlineBody:b},camera,w,h);
        m.globalCompositeOperation='source-over';m.clearRect(0,0,w,h);m.drawImage(this.outlineMask,0,0);
        m.globalCompositeOperation='source-in';m.fillStyle=colors[b];m.fillRect(0,0,w,h);m.globalCompositeOperation='source-over';
        e.globalCompositeOperation='source-over';e.clearRect(0,0,w,h);
        for(const [dx,dy]of[[-offset,0],[offset,0],[0,-offset],[0,offset],[-offset,-offset],[offset,offset],[-offset,offset],[offset,-offset]])e.drawImage(mask,dx,dy);
        e.globalCompositeOperation='destination-out';e.drawImage(mask,0,0);e.globalCompositeOperation='source-over';
        result.drawImage(edge,0,0);
      }
      this.outlineKey=key;
    }
    this.ctx.save();this.ctx.globalAlpha=.72;this.ctx.drawImage(this.outlineImage,0,0,this.w,this.h);this.ctx.restore();
  }
  fit(poses){const bs=poses.map(p=>sceneBounds({pose:p})),lo=[0,1,2].map(k=>Math.min(...bs.map(b=>b.ctr[k]-b.rad))),hi=[0,1,2].map(k=>Math.max(...bs.map(b=>b.ctr[k]+b.rad)));this.view.target=lo.map((x,k)=>(x+hi[k])/2);const r=Math.max(...bs.map(b=>Math.hypot(...b.ctr.map((x,k)=>x-this.view.target[k]))+b.rad));this.view.dist=Math.max(.3,1.05*r/Math.sin(FOV/2));}
}
window.Q13DiagramView=Q13DiagramView;
window.Q13Geometry={mat,mul,world:(p,pose)=>[0,1,2].map(i=>pose[3*i]*p[0]+pose[3*i+1]*p[1]+pose[3*i+2]*p[2]+pose[9+i])};
