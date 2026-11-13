(() => {
 const $=s=>document.querySelector(s),view=new Q13DiagramView($('#release-view'),{dist:.72});
 let data,scene=0,time=0,playing=false,last=null,speed=.25,request=0;
 const setPlay=x=>{playing=x;$('#play').textContent=x?'Pause':'Play';$('#play').setAttribute('aria-pressed',String(x));};
 function show(){if(!data)return;const tr=data.traces[scene],k=Math.min(500,Math.max(0,Math.floor(time/.002+1e-8))),f=tr.frames[k];
  view.set(f.pose,v=>{
   const colors=['#2a78d6','#d86b30'];
   v.initialOutlines(tr.frames[0].pose,colors);
   for(let b=0;b<2;b++){const p=f.excursionLines[b],a=p.slice(0,3),z=p.slice(3);if(f.X[b]>.001){v.arrow(a,z,colors[b],[5,3]);v.dot(a,colors[b],true);v.dot(z,colors[b]);v.label(z,`X_${b?'spoon':'mug'} = ${f.X[b].toFixed(1)} mm`,8,b?20:-10,colors[b]);}}
   const contact=f.contact,depth=f.delta/1000;
   if(depth>.000001&&!(k===0&&tr.delta0>.001)){const a=contact.slice(0,3).map((x,j)=>x-contact[3+j]*depth/2),z=contact.slice(0,3).map((x,j)=>x+contact[3+j]*depth/2);v.line(a,z,'#c43c42',[],4);v.dot(contact.slice(0,3),'#c43c42',false,4);v.label(contact.slice(0,3),'peak penetration',9,-12,'#c43c42');}
   if(k===0&&tr.delta0>.001){const p=tr.initialContact,d=tr.delta0/1000,a=p.slice(0,3).map((x,j)=>x-p[3+j]*d/2),z=p.slice(0,3).map((x,j)=>x+p[3+j]*d/2);v.line(a,z,'#a22d78',[],4);v.dot(p.slice(0,3),'#a22d78',false,5);v.label(p.slice(0,3),'starting overlap δ₀',9,-15,'#a22d78');}
   const c=v.ctx,x=14,y=22,w=Math.min(240,v.w-28);c.fillStyle=v.bg;c.fillRect(x-5,y-17,w,104);c.font='12px system-ui, sans-serif';c.fillStyle=v.ink;c.fillText('Penetration depths · enlarged',x,y);
   [[tr.delta0,'δ₀(s), fixed','#a22d78'],[f.delta,'δ(s; τ), peak','#c43c42']].forEach(([d,name,col],i)=>{const yy=y+25+i*35,L=Math.min(w-12,Math.max(0,d*30));c.fillStyle=v.ink;c.fillText(`${name}: ${d.toFixed(3)} mm`,x,yy);c.strokeStyle=col;c.lineWidth=2;c.beginPath();c.moveTo(x,yy+8);c.lineTo(x+L,yy+8);c.moveTo(x,yy+4);c.lineTo(x,yy+12);c.moveTo(x+L,yy+4);c.lineTo(x+L,yy+12);c.stroke();});
  });$('#time-value').textContent=`τ = ${(k*.002).toFixed(3)} s`;$('#scrub').value=String(k);
  $('#xm').textContent=f.X[0].toFixed(3)+' mm';$('#xs').textContent=f.X[1].toFixed(3)+' mm';$('#delta').textContent=f.delta.toFixed(3)+' mm';$('#delta0').textContent=tr.delta0.toFixed(3)+' mm';
  $('#scene-id').textContent=tr.deliberateOverlap?'Deliberately overlapping input · outside the touching-scene map':`Recorded scene ${tr.id} of 18 · starting particle ${tr.row}`;
 }
 function loop(t){if(!playing){request=0;last=null;return;}if(last!==null)time=Math.min(1,time+(t-last)/1000*speed);last=t;show();if(time>=1){setPlay(false);request=0;last=null;}else request=requestAnimationFrame(loop);}
 function start(){if(time>=1)time=0;setPlay(true);last=null;if(!request)request=requestAnimationFrame(loop);}
 function select(index,autoplay){if(!data)return;setPlay(false);if(request)cancelAnimationFrame(request);request=0;last=null;scene=index;time=0;const f=data.traces[scene].frames;view.fit(f.map(x=>x.pose));show();if(autoplay&&!matchMedia('(prefers-reduced-motion:reduce)').matches)start();}
 function choose(){if(!data)return;const a=new Uint32Array(1),n=18,limit=Math.floor(4294967296/n)*n;do{crypto.getRandomValues(a);}while(a[0]>=limit);select(a[0]%n,true);}
 $('#random-scene').addEventListener('click',choose);
 $('#initial-overlap').addEventListener('click',()=>{if(data)select(data.traces.findIndex(x=>x.deliberateOverlap),false);});
 $('#play').addEventListener('click',()=>{if(playing)setPlay(false);else start();});
 $('#speed').addEventListener('input',e=>{speed=Number(e.target.value)/100;$('#speed-value').textContent=speed.toFixed(2)+'×';});
 $('#scrub').addEventListener('input',e=>{setPlay(false);time=Number(e.target.value)*.002;show();});
 fetch('release-data.json').then(r=>{if(!r.ok)throw Error('Unable to load recorded releases');return r.json();}).then(d=>{data=d;$('#random-scene').disabled=false;$('#initial-overlap').disabled=false;$('#play').disabled=false;$('#loading').hidden=true;view.fit(d.traces[0].frames.map(x=>x.pose));show();}).catch(e=>{$('#loading').textContent=e.message;$('#loading').setAttribute('role','alert');});
})();
