(() => {
 const G=window.Q13Geometry,$=s=>document.querySelector(s),bodyNames=['mug','spoon'];
 const world=G.world,pi=Math.PI;
 const rx=a=>[1,0,0,0,Math.cos(a),-Math.sin(a),0,Math.sin(a),Math.cos(a)];
 const rz=a=>[Math.cos(a),-Math.sin(a),0,Math.sin(a),Math.cos(a),0,0,0,1];
 const mm=(a,b)=>[0,1,2,3,4,5,6,7,8].map(k=>[0,1,2].reduce((s,j)=>s+a[Math.floor(k/3)*3+j]*b[j*3+k%3],0));
 const mv=(r,p)=>[0,1,2].map(i=>r[3*i]*p[0]+r[3*i+1]*p[1]+r[3*i+2]*p[2]);
 function height(b,R){let low=Infinity;for(const g of BODIES[bodyNames[b]]){const q=mm(R,g.R),v=q.slice(6),s=g.size;let e;
  if(g.type==='box')e=s.reduce((z,x,i)=>z+x*Math.abs(v[i]),0);
  else if(g.type==='ellipsoid')e=Math.hypot(...s.map((x,i)=>x*v[i]));
  else if(g.type==='capsule')e=s[0]+s[1]*Math.abs(v[2]);
  else if(g.type==='cylinder')e=s[0]*Math.hypot(v[0],v[1])+s[1]*Math.abs(v[2]);
  else e=s[0];low=Math.min(low,mv(R,g.pos)[2]-e);}return-low;}
 const physical=new Q13DiagramView($('#physical-view'),{dist:.52,target:[.065,0,.04],az:-.95,el:.7});
 const polar=new Q13DiagramView($('#polar-view'),{dist:.52,target:[.065,0,.025],az:-.85,el:1.18});
 const controls=['radius','direction','relative','tilt','order','selected-body'];
 function update(){
  const rad=Number($('#radius').value)/1000,phi=Number($('#direction').value)*pi/180,delta=Number($('#relative').value)*pi/180,tilt=Number($('#tilt').value)*pi/180,b=Number($('#selected-body').value);
  const Rm=rx(tilt),Q=mm(rz(delta),rx(pi/2)),Rs=mm(rz(phi),Q),R=[Rm,Rs],v=[rad*Math.cos(phi),rad*Math.sin(phi),0];
  const poses=R.map((r,k)=>{const q=mv(r,DIAGRAM_COM[k]),c=k?v:[0,0,0];return[...r,c[0]-q[0],c[1]-q[1],height(k,r)];});
  const p=b?[0,0,.168]:[.081,0,.05],o=poses[b].slice(9),y=world(p,poses[b]),cp=world(DIAGRAM_COM[b],poses[b]);
  physical.set(poses,view=>{
   const origin=[0,0,0];
   for(let k=0;k<3;k++){const a=[0,0,0];a[k]=.035;view.arrow(origin,a,view.ink);}
   view.label([0,0,.035],'world axes',-64,-6);
   view.arrow(origin,o,'#646b76');view.label(o.map(x=>x*.5),'t_b',-18,-10,'#646b76');
   view.arrow(o,y,'#2a78d6');view.label(o.map((x,i)=>(x+y[i])/2),'R_b p',14,1,'#2a78d6');
   view.arrow(origin,y,'#d86b30');view.label(y.map(x=>x*.48),'y_b',-25,22,'#d86b30');view.label(y,'material point',5,16);
   for(let k=0;k<3;k++){const e=[0,0,0];e[k]=.037;const a=mv(R[b],e).map((x,i)=>x+o[i]);view.arrow(o,a,'#2a78d6');if(k===1)view.label(a,'R_b',5,-6,'#2a78d6');}
   view.dot(o,view.ink,true);view.label(o,'body origin',7,-26);view.dot(cp,'#a25195');view.label(cp,'COM',7,24,'#a25195');view.dot(y,'#d86b30',false,4);
  });
  polar.set(poses,view=>{
   const c0=[0,0,0],c1=[...v];view.arrow(c0,c1,'#2a78d6',[5,3]);view.label(v.map(x=>x*.6),`rL = ${(rad*1000).toFixed(0)} mm`,5,16);
   view.line(c0,[.065,0,0],view.ink,[3,3]);
   const arc=.062;for(let j=0;j<48;j++){const a=phi*j/48,z=phi*(j+1)/48;view.line([arc*Math.cos(a),arc*Math.sin(a),.001],[arc*Math.cos(z),arc*Math.sin(z),.001],'#d86b30',[],2);}
   view.label([arc*Math.cos(phi/2),arc*Math.sin(phi/2),.002],`φ = ${(phi*180/pi).toFixed(0)}°`,-45,-24,'#d86b30');
   for(let k=0;k<2;k++){const ax=[0,0,0];ax[k]=.04;const z=mv(rz(phi),ax).map((x,i)=>x+c1[i]);view.arrow(c1,z,'#7c5baa');if(k===1)view.label(z,'direction frame',-40,-16,'#7c5baa');}
   view.dot(c0);view.dot(c1,'#2a78d6');view.label(c1,'spoon COM',9,36);view.label([0,0,0],'mug COM',-82,20);
   for(let k=0;k<2;k++){const c=world(DIAGRAM_COM[k],poses[k]);view.line([c[0],c[1],0],c,view.ink,[2,4]);}
  });
  $('#radius-value').textContent=`r = ${(rad/.1).toFixed(2)} · ${(rad*1000).toFixed(0)} mm`;
  $('#direction-value').textContent=`${(phi*180/pi).toFixed(0)}° = ${phi.toFixed(2)} rad`;
  $('#relative-value').textContent=`${(delta*180/pi).toFixed(0)}°`;$('#tilt-value').textContent=`${(tilt*180/pi).toFixed(0)}°`;
  $('#placement-order').textContent=$('#order').value==='1'?'ξ = +1: mug, then spoon':'ξ = −1: spoon, then mug';
  const fmt=a=>'('+a.map(x=>(1000*x).toFixed(1)).join(', ')+') mm';
  $('#point-local').textContent=fmt(p);$('#translation').textContent=fmt(o);$('#point-world').textContent=fmt(y);
  $('#body-name').textContent=bodyNames[b];$('#heights').textContent=poses.map((x,k)=>`${bodyNames[k]} ${(1000*(world(DIAGRAM_COM[k],x)[2])).toFixed(1)} mm`).join(' · ');
  $('#polar-equation').textContent=`v = (${(1000*v[0]).toFixed(1)}, ${(1000*v[1]).toFixed(1)}) mm`;
 }
 for(const id of controls)$('#'+id).addEventListener('input',update);
 update();
})();
