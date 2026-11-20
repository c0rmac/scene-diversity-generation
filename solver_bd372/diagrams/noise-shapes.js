// Isotropic and shaped noise, in the coordinates and in the material (solver page, Section 9). Geometry from aniso-data.js.
(() => {
  const $ = s => document.querySelector(s), A = AX, N = 300, AXES = [0, 3, 6], DWELL = 1100;
  const view = new SceneDiagramView($('#view'), { dist: .5, el: .42, az: -1.05 });
  const selScene = $('#scene'), selNoise = $('#noise'), selUnits = $('#units'), amp = $('#amp'), ampV = $('#amp-v'), btn = $('#play'), next = $('#next');
  const cv = $('#cloud'), drawinfo = $('#drawinfo'), stats = $('#stats');
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  let sc = null, draws = null, j = 0, playing = !reduced, timer = 0, cam = { az: 0.75, el: 0.42 };
  ANISO.scenes.forEach((s, i) => selScene.add(new Option(s.label, i)));
  const rnd = A.normals(20261118), Z = Array.from({ length: N }, () => Array.from({ length: 7 }, rnd));
  const lamText = x => x < 0.1 ? x.toFixed(3) : x.toFixed(2), a = () => 0.01 * (+amp.value);
  function compute() {
    const C = A.cholesky(sc.Gt), V = sc.V, mmPer = k => Math.sqrt(sc.lam[k] * ANISO.cbar) * A.L * 1000, shaft = [sc.pose[1][2], sc.pose[1][5], sc.pose[1][8]];
    const out = { iso: [], shaped: [] };
    for (const kind of ['iso', 'shaped']) for (const z of Z) {
      const eta = (kind === 'iso' ? z : A.solveCT(C, z)).map(x => x * a()), c = AXES.map(k => V.reduce((s, row, i) => s + row[k] * eta[i], 0));
      const tw = A.twist(sc, eta), rms = A.bodyRms(sc, tw);
      out[kind].push({ eta, tw, coord: c, mat: c.map((x, i) => x * mmPer(AXES[i])), len: Math.sqrt(Math.max(A.quad(sc.G, eta), 0)) * A.L * 1000, rms,
                       roll: A.dot3(tw.slice(9, 12), shaft) * 180 / Math.PI, tilt: Math.hypot(tw[3], tw[4]) * 180 / Math.PI });
    }
    const rmsOf = (arr, f) => Math.sqrt(arr.reduce((s, x) => s + f(x) ** 2, 0) / arr.length);
    out.summary = ['iso', 'shaped'].map(k => ({ len: rmsOf(out[k], x => x.len), roll: rmsOf(out[k], x => x.roll), tilt: rmsOf(out[k], x => x.tilt), axes: [0, 1, 2].map(i => rmsOf(out[k], x => x.mat[i])) }));
    return out;
  }
  function drawCloud() {
    const dpr = Math.min(2, devicePixelRatio || 1), W = cv.clientWidth, H = cv.clientHeight; if (!W || !H) return;
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); const c = cv.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, W, H);
    const units = selUnits.value, key = units === 'coord' ? 'coord' : 'mat', ink = A.css('--text-primary'), mut = A.css('--text-secondary'), grid = A.css('--grid');
    const col = { iso: A.css('--series-1'), shaped: A.css('--series-2') }, shown = selNoise.value;
    const r0 = units === 'coord' ? a() : a() * Math.sqrt(ANISO.cbar) * A.L * 1000;           // isotropic per-coordinate spread, or shaped per-direction displacement
    let ext = r0 * 2.2; for (const k of ['iso', 'shaped']) for (const d of draws[k]) for (const x of d[key]) ext = Math.max(ext, Math.abs(x));
    const sc2 = 0.42 * Math.min(W, H) / ext, ca = Math.cos(cam.az), sa = Math.sin(cam.az), ce = Math.cos(cam.el), se = Math.sin(cam.el);
    // axes x: soft, y: middle, z: stiff (vertical on screen)
    const P = p => { const x1 = ca * p[0] - sa * p[1], y1 = sa * p[0] + ca * p[1]; return [W / 2 + sc2 * x1, H / 2 - sc2 * (ce * p[2] - se * y1), ce * y1 + se * p[2]]; };
    c.lineWidth = 1; c.strokeStyle = grid;
    for (const plane of [[0, 1], [0, 2], [1, 2]]) { c.beginPath(); for (let i = 0; i <= 72; i++) { const t = 2 * Math.PI * i / 72, p = [0, 0, 0]; p[plane[0]] = r0 * Math.cos(t); p[plane[1]] = r0 * Math.sin(t); const q = P(p); i ? c.lineTo(q[0], q[1]) : c.moveTo(q[0], q[1]); } c.stroke(); }
    const names = AXES.map((k, i) => (['soft', 'middle', 'stiff'][i]) + ' · λ' + String.fromCharCode(0x2081 + k) + ' = ' + lamText(sc.lam[k]));
    c.font = '12px system-ui,sans-serif';
    for (let i = 0; i < 3; i++) { const e = [0, 0, 0]; e[i] = ext * 1.02; const p = P(e.map(x => -x)), q = P(e); c.strokeStyle = mut; c.beginPath(); c.moveTo(p[0], p[1]); c.lineTo(q[0], q[1]); c.stroke(); c.fillStyle = mut; c.fillText(names[i], Math.min(W - c.measureText(names[i]).width - 4, q[0] + 4), Math.max(14, q[1] - 4)); }
    const pts = [];
    for (const k of ['iso', 'shaped']) draws[k].forEach((d, i) => { const q = P(d[key]); pts.push({ k, i, q }); });
    const zs = pts.map(p => p.q[2]), lo = Math.min(...zs), hi = Math.max(...zs);
    pts.sort((p, q) => p.q[2] - q.q[2]).forEach(p => { const main = p.k === shown; c.globalAlpha = (main ? 0.45 : 0.18) + (main ? 0.5 : 0.2) * (p.q[2] - lo) / Math.max(hi - lo, 1e-9); c.fillStyle = col[p.k]; c.beginPath(); c.arc(p.q[0], p.q[1], main ? 2.6 : 2, 0, 2 * Math.PI); c.fill(); });
    c.globalAlpha = 1; const cur = P(draws[shown][j][key]); c.strokeStyle = ink; c.lineWidth = 1.6; c.beginPath(); c.arc(cur[0], cur[1], 6.5, 0, 2 * Math.PI); c.stroke();
    c.fillStyle = mut; c.fillText(units === 'coord' ? 'units of the coordinates (rad, or L for the offset)' : 'material displacement along each direction, mm', 6, H - 8);
  }
  function drawScene() {
    const d = draws[selNoise.value][j], s1 = A.css('--series-1'), s2 = A.css('--series-2');
    view.set(A.moved(sc, d.tw), v => v.initialOutlines(sc.pose, [s1, s2]));
    drawinfo.textContent = `draw ${j + 1} of ${N} · material length ${d.len.toFixed(1)} mm (mug ${d.rms[0].toFixed(1)}, spoon ${d.rms[1].toFixed(1)}) · spoon roll about its shaft ${d.roll.toFixed(0)}° · mug tilt ${d.tilt.toFixed(1)}°`;
  }
  function render() { drawCloud(); drawScene(); }
  function writeStats() {
    const [i, s] = draws.summary, f = x => x.toFixed(1);
    stats.innerHTML = `<b>Over the 300 draws at a = ${a().toFixed(2)}</b>: the root-mean-square material length is ${f(i.len)} mm for the isotropic noise and ${f(s.len)} mm for the shaped noise. ` +
      `Along the soft, middle and stiff directions the isotropic noise moves the material by ${i.axes.map(f).join(', ')} mm, the shaped noise by ${s.axes.map(f).join(', ')} mm. ` +
      `The shaped noise rolls the spoon about its shaft by ${f(s.roll)}° (isotropic ${f(i.roll)}°), which moves little material, and tilts the mug by ${f(s.tilt)}° (isotropic ${f(i.tilt)}°).`;
  }
  function setAll() { sc = ANISO.scenes[+selScene.value]; ampV.textContent = a().toFixed(2); draws = compute(); j = Math.min(j, N - 1); view.fit([sc.pose]); view.view.dist *= 0.95; writeStats(); render(); }
  function tick() { j = (j + 1) % N; render(); }
  function schedule() { clearInterval(timer); if (playing) timer = setInterval(tick, DWELL); }
  let drag = null;
  cv.addEventListener('pointerdown', e => { drag = [e.clientX, e.clientY]; cv.setPointerCapture(e.pointerId); });
  cv.addEventListener('pointermove', e => { if (!drag) return; cam.az -= (e.clientX - drag[0]) * 0.01; cam.el = Math.max(-1.4, Math.min(1.4, cam.el + (e.clientY - drag[1]) * 0.01)); drag = [e.clientX, e.clientY]; drawCloud(); });
  for (const ev of ['pointerup', 'pointercancel']) cv.addEventListener(ev, () => { drag = null; });
  cv.addEventListener('keydown', e => { const m = { ArrowLeft: [-.12, 0], ArrowRight: [.12, 0], ArrowUp: [0, .1], ArrowDown: [0, -.1] }[e.key]; if (m) { e.preventDefault(); cam.az += m[0]; cam.el = Math.max(-1.4, Math.min(1.4, cam.el + m[1])); drawCloud(); } });
  new ResizeObserver(() => drawCloud()).observe(cv);
  matchMedia('(prefers-color-scheme:dark)').addEventListener('change', render);
  selScene.addEventListener('change', setAll); amp.addEventListener('input', setAll);
  selNoise.addEventListener('change', render); selUnits.addEventListener('change', drawCloud);
  btn.addEventListener('click', () => { playing = !playing; btn.textContent = playing ? 'Pause' : 'Play'; btn.setAttribute('aria-pressed', playing); schedule(); });
  next.addEventListener('click', () => { playing = false; btn.textContent = 'Play'; btn.setAttribute('aria-pressed', 'false'); schedule(); tick(); });
  if (!playing) { btn.textContent = 'Play'; btn.setAttribute('aria-pressed', 'false'); }
  setAll(); schedule();
})();
