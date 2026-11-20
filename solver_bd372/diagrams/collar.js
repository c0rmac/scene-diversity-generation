(() => {
 const $=s=>document.querySelector(s), D=COLLAR_DATA, bin=atob(D.poses), F=new Float32Array(new Uint8Array([...bin].map(c=>c.charCodeAt(0))).buffer);
 const G=SceneGeometry, view=new SceneDiagramView($('#physical-view'),{dist:.6,target:[.06,0,.05],az:-.95,el:.55});
 const nT=D.meta.tilts.length,nP=D.meta.pitches.length,nD=D.meta.dirs.length,nS=D.meta.sizes.length,nO=D.meta.orders.length;
 const idx=(it,ip,id,is,io)=>((((it*nP+ip)*nD+id)*nS+is)*nO+io);
 const mm=v=>(1000*v).toFixed(0)+' mm', f2=v=>v.toFixed(2);
 const cv=$('#collar-canvas'), css=v=>getComputedStyle(document.body).getPropertyValue(v).trim();
 function poseAt(k){const o=24*k;return [Array.from(F.slice(o,o+12)),Array.from(F.slice(o+12,o+24))];}
 function com(b,P){return G.world(DIAGRAM_COM[b],P);}
 function draw2d(g,xi,w){const dpr=Math.min(2,devicePixelRatio||1),W=cv.clientWidth,H=cv.clientHeight;cv.width=Math.round(W*dpr);cv.height=Math.round(H*dpr);const c=cv.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);c.clearRect(0,0,W,H);
  const ink=css('--text-primary'),mut=css('--text-secondary'),grid=css('--grid'),s1=css('--series-1'),s2=css('--series-2');c.font='12px system-ui,sans-serif';c.lineWidth=1.2;
  // left: the sphere of directions in the mug frame, seen along the tilt axis (mug y)
  const cx=W*0.24,cy=H*0.52,R=Math.min(W*0.14,H*0.34);c.strokeStyle=grid;c.beginPath();c.arc(cx,cy,R,0,2*Math.PI);c.stroke();c.beginPath();c.moveTo(cx-R,cy);c.lineTo(cx+R,cy);c.stroke();
  c.fillStyle=mut;c.textAlign='center';c.fillText('u on S² (mug frame)',cx,cy-R-22);c.fillText('upside down',cx,cy-R-6);c.fillText('upright',cx,cy+R+16);c.textAlign='right';c.fillText('on its side',cx-R-4,cy+4);
  const u=g.u,px=cx+R*u[0],py=cy-R*u[2];c.strokeStyle=s1;c.lineWidth=2;c.beginPath();c.moveTo(cx,cy);c.lineTo(px,py);c.stroke();c.fillStyle=s1;c.beginPath();c.arc(px,py,4.5,0,2*Math.PI);c.fill();
  c.fillStyle=ink;c.fillText('u',px+7,py-5);
  // right: the offset plane seen from above, axes e1 (right) and e2 (up), mm
  const ox=W*0.68,oy=H*0.52,ext=Math.max(...g.K.map(Math.abs),60)+Math.max(...g.spoon.map(Math.abs)),sc=Math.min(W*0.29,H*0.44)/ext;
  const P=(x,y)=>[ox+sc*x,oy-sc*y];const poly=(pts,fill,stroke,dash)=>{c.beginPath();for(let k=0;k<pts.length;k+=2){const q=P(pts[k],pts[k+1]);k?c.lineTo(q[0],q[1]):c.moveTo(q[0],q[1]);}c.closePath();if(fill){c.fillStyle=fill;c.fill();}if(stroke){c.strokeStyle=stroke;c.setLineDash(dash||[]);c.lineWidth=1.4;c.stroke();c.setLineDash([]);}};
  c.strokeStyle=grid;c.beginPath();c.moveTo(ox-sc*200,oy);c.lineTo(ox+sc*200,oy);c.moveTo(ox,oy-sc*200);c.lineTo(ox,oy+sc*200);c.stroke();
  poly(g.mug,'rgba(120,120,120,0.35)',mut);poly(g.K,null,ink,[5,4]);
  const sp=g.spoon.map((v,k)=>v+(k%2?w[1]:w[0]));poly(sp,'rgba(235,104,52,0.18)',s2);
  c.strokeStyle=ink;c.lineWidth=1.6;c.beginPath();const a=P(0,0),b=P(w[0],w[1]);c.moveTo(a[0],a[1]);c.lineTo(b[0],b[1]);c.stroke();c.fillStyle=ink;c.beginPath();c.arc(b[0],b[1],3.5,0,2*Math.PI);c.fill();
  c.fillStyle=mut;c.textAlign='center';c.fillText('offset plane, seen from above',ox,14);c.textAlign='left';c.fillText('e₁ →',ox+sc*ext*0.82,oy+14);c.fillText('↑ e₂',ox+4,oy-sc*ext*0.86);
 }
 function update(){const it=+$('#tilt').value,ip=+$('#pitch').value,id=+$('#dir').value,is=+$('#size').value,io=+$('#order').value,k=idx(it,ip,id,is,io),P=poseAt(k),g=D.geo[it*nP+ip];
  $('#tilt-v').textContent=D.meta.tilts[it]+'°';$('#pitch-v').textContent=D.meta.pitches[ip]+'°';$('#dir-v').textContent=D.meta.dirs[id]+'°';$('#size-v').textContent=D.meta.sizes[is].toFixed(2);
  const r=D.meta.sizes[is],ph=D.meta.dirs[id]*Math.PI/180,Hh=D.H[k],w=[1000*Hh*r*Math.cos(ph),1000*Hh*r*Math.sin(ph)];
  $('#uout').textContent='('+g.u.map(f2).join(', ')+')';$('#xiout').textContent=r.toFixed(2)+' (cos '+D.meta.dirs[id]+'° e₁ + sin '+D.meta.dirs[id]+'° e₂)';$('#xin').textContent=r.toFixed(2);$('#hout').textContent=(1000*Hh).toFixed(0)+' mm';
  $('#orderout').textContent=io?'p < 0: the spoon is lowered onto the table first, then the mug onto spoon and table':'p ≥ 0: the mug is lowered onto the table first, then the spoon onto mug and table';
  const cm=com(0,P[0]),cs=com(1,P[1]);$('#hm').textContent=mm(cm[2]);$('#hs').textContent=mm(cs[2]);$('#woff').textContent=(Math.hypot(cs[0]-cm[0],cs[1]-cm[1])*1000).toFixed(0)+' mm';
  view.set(P,v=>{v.dot(cm,v.ink,true);v.dot(cs,v.ink,true);v.arrow(cm,[cs[0],cs[1],cm[2]],v.ink,[5,4]);v.label(cm,'mug COM',8,-8);v.label(cs,'spoon COM',8,-8);});
  draw2d(g,null,w);}
 for(const id of ['tilt','pitch','dir','size','order'])$('#'+id).addEventListener('input',update);
 new ResizeObserver(update).observe(cv); matchMedia('(prefers-color-scheme:dark)').addEventListener('change',update); update();
})();
