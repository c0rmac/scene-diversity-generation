// How each direction of the coordinates moves the material (solver page, Section 9). Geometry from aniso-data.js.
(() => {
  const $ = s => document.querySelector(s), A = AX;
  const view = new SceneDiagramView($('#view'), { dist: .5, el: .42, az: -1.05 });
  const selScene = $('#scene'), selDir = $('#dir'), selMode = $('#mode'), btn = $('#play'), chk = $('#arrows'), info = $('#info'), lengths = $('#lengths');
  const STEP = 0.15, TARGET_MM = 5, PERIOD = 2400;
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  let sc = null, k = 6, playing = !reduced, t0 = performance.now();
  ANISO.scenes.forEach((s, i) => selScene.add(new Option(s.label, i)));
  const lamText = x => x < 0.1 ? x.toFixed(3) : x.toFixed(2);
  const dirLabel = d => d.kind === 'eigen' ? `eigen-direction ${d.name.split(' ').pop()}, λ = ${lamText(d.lam)}` : d.name;
  function stepOf(d) { const per = d.length_L * A.L * 1000; return selMode.value === 'coord' ? STEP : TARGET_MM / per; }
  function fillDirs() {
    selDir.innerHTML = '';
    const g1 = document.createElement('optgroup'), g2 = document.createElement('optgroup');
    g1.label = 'coordinates'; g2.label = 'eigen-directions of G̃, soft to stiff';
    sc.dirs.forEach((d, i) => (d.kind === 'eigen' ? g2 : g1).append(new Option(dirLabel(d), i)));
    selDir.append(g1, g2); selDir.value = String(k);
  }
  function fillLengths() {
    const max = Math.max(...sc.dirs.map(d => d.length_L)) * A.L * 1000, s1 = A.css('--series-1');
    const row = (d, i) => `<button type="button" data-k="${i}" class="${i === k ? 'on' : ''}" aria-pressed="${i === k}"><span>${d.kind === 'eigen' ? 'eigen ' + d.name.split(' ').pop() + ' · λ ' + lamText(d.lam) : d.name.replace('spoon turn about its ', 'spoon turn, ').replace('tilt of the mug along', 'mug tilt,')}</span>` +
      `<span class="bar"><i style="width:${(100 * d.length_L * A.L * 1000 / max).toFixed(1)}%;background:${s1}"></i></span><span class="val">${(d.length_L * A.L * 1000).toFixed(1)}</span></button>`;
    lengths.innerHTML = '<h4>Material length of a unit step, mm</h4><p class="note">unit step: 1 rad, or 1 L = 0.1 m for the offset</p>' +
      '<h4>coordinates</h4>' + sc.dirs.map((d, i) => d.kind === 'eigen' ? '' : row(d, i)).join('') + '<h4>eigen-directions of G̃</h4>' + sc.dirs.map((d, i) => d.kind === 'eigen' ? row(d, i) : '').join('');
    lengths.querySelectorAll('button').forEach(b => b.addEventListener('click', () => { k = +b.dataset.k; selDir.value = String(k); update(); }));
  }
  function describe(d, s) {
    const per = d.length_L * A.L * 1000, rms = d.rms_mm.map(x => x * s), unit = d.kind === 'eigen' ? '' : (d.unit === 'L' ? ' L' : ' rad');
    const size = `A step of ${s.toFixed(s < 0.1 ? 3 : 2)}${unit} moves the mug's surface points by ${rms[0].toFixed(1)} mm and the spoon's by ${rms[1].toFixed(1)} mm (root mean square); its material length is ${(per * s).toFixed(1)} mm.`;
    if (d.kind !== 'eigen') return `<b>${d.name[0].toUpperCase() + d.name.slice(1)}.</b> One unit of this coordinate has a material length of ${per.toFixed(1)} mm. ${size}`;
    return `<b>Eigen-direction ${d.name.split(' ').pop()}</b>, λ = ${lamText(d.lam)}. Per unit step the mug turns ${d.mug_turn.toFixed(2)} rad and the spoon ${d.spoon_turn.toFixed(2)} rad, ${Math.round(100 * d.shaft_share)} % of it about its own shaft, and the spoon's centre of mass moves ${d.rel_com_L.toFixed(2)} L relative to the mug's. ${size}`;
  }
  function update() {
    const d = sc.dirs[k], s = stepOf(d);
    info.innerHTML = describe(d, s);
    lengths.querySelectorAll('button').forEach(b => { const on = +b.dataset.k === k; b.classList.toggle('on', on); b.setAttribute('aria-pressed', on); });
    draw();
  }
  function frame(now) {
    const d = sc.dirs[k], s = stepOf(d), ph = playing ? Math.sin(2 * Math.PI * ((now - t0) % PERIOD) / PERIOD) : 1;
    const tw = A.twist(sc, d.v.map(x => x * s)), twNow = tw.map(x => x * ph), pose = A.moved(sc, twNow);
    const s1 = A.css('--series-1'), s2 = A.css('--series-2');
    view.set(pose, v => {
      v.initialOutlines(sc.pose, [s1, s2]);
      if (!chk.checked) return;
      const eye = A.eyeDir(v), tg = v.view.target, items = [];
      [0, 1].forEach(b => {
        const P = sc.pose[b], R = P.slice(0, 9), c = sc.com[b], vb = tw.slice(6 * b, 6 * b + 3).map(x => x * A.L), E = A.expso3(tw.slice(6 * b + 3, 6 * b + 6));
        ANISO.markers[b].forEach(q => { const r = A.matv(R, q), p = A.add(c, r), p2 = A.add(A.add(c, vb), A.matv(E, r)); items.push({ b, p, p2, z: A.dot3([p[0] - tg[0], p[1] - tg[1], p[2] - tg[2]], eye) }); });
      });
      const zs = items.map(x => x.z), lo = Math.min(...zs), hi = Math.max(...zs);
      items.sort((a, b) => a.z - b.z).forEach(x => A.arrow(v, x.p, x.p2, x.b ? s2 : s1, 0.3 + 0.7 * (x.z - lo) / Math.max(hi - lo, 1e-9)));
    });
  }
  function draw() { frame(performance.now()); }
  function loop(now) { if (playing) { frame(now); requestAnimationFrame(loop); } }
  function setScene() {
    sc = ANISO.scenes[+selScene.value]; fillDirs(); fillLengths();
    view.fit([sc.pose]); view.view.dist *= 0.9; update();
  }
  selScene.addEventListener('change', setScene);
  selDir.addEventListener('change', () => { k = +selDir.value; update(); });
  selMode.addEventListener('change', update);
  chk.addEventListener('change', draw);
  btn.addEventListener('click', () => { playing = !playing; btn.textContent = playing ? 'Pause' : 'Play'; btn.setAttribute('aria-pressed', playing); if (playing) { t0 = performance.now(); requestAnimationFrame(loop); } else draw(); });
  if (!playing) { btn.textContent = 'Play'; btn.setAttribute('aria-pressed', 'false'); }
  setScene(); if (playing) requestAnimationFrame(loop);
})();
