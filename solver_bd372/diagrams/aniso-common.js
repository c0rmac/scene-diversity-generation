// Shared helpers of the anisotropic-diffusion figures of the solver page (18 Nov 2026). Geometry only: the scenes, twists and
// metrics come from aniso-data.js (aniso_figures_v1.py); nothing here runs the solver or a physics simulation.
const AX = (() => {
  const L = ANISO.L;
  const matv = (R, v) => [R[0] * v[0] + R[1] * v[1] + R[2] * v[2], R[3] * v[0] + R[4] * v[1] + R[5] * v[2], R[6] * v[0] + R[7] * v[1] + R[8] * v[2]];
  const crossv = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  function expso3(w) {
    const th = Math.hypot(w[0], w[1], w[2]); if (th < 1e-12) return [1, 0, 0, 0, 1, 0, 0, 0, 1];
    const k = w.map(x => x / th), c = Math.cos(th), s = Math.sin(th), C = 1 - c;
    return [c + k[0] * k[0] * C, k[0] * k[1] * C - k[2] * s, k[0] * k[2] * C + k[1] * s,
            k[1] * k[0] * C + k[2] * s, c + k[1] * k[1] * C, k[1] * k[2] * C - k[0] * s,
            k[2] * k[0] * C - k[1] * s, k[2] * k[1] * C + k[0] * s, c + k[2] * k[2] * C];
  }
  function mul3(A, B) { const o = new Array(9).fill(0); for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) o[3 * i + j] += A[3 * i + k] * B[3 * k + j]; return o; }
  // The world twist (v_mug, w_mug, v_spoon, w_spoon) of a step y of the seven coordinates, gauge removed; v in units of L.
  function twist(sc, y) { const T = sc.twists, out = new Array(12).fill(0); for (let r = 0; r < 12; r++) for (let j = 0; j < 7; j++) out[r] += T[r][j] * y[j]; return out; }
  // The scene moved by a twist: each body turned about its own centre of mass and its centre moved (lengths in metres).
  function moved(sc, tw) {
    return [0, 1].map(b => {
      const P = sc.pose[b], R = P.slice(0, 9), t = P.slice(9), c = sc.com[b];
      const v = tw.slice(6 * b, 6 * b + 3).map(x => x * L), w = tw.slice(6 * b + 3, 6 * b + 6), E = expso3(w);
      const Ect = matv(E, [c[0] - t[0], c[1] - t[1], c[2] - t[2]]);
      return mul3(E, R).concat([c[0] + v[0] - Ect[0], c[1] + v[1] - Ect[1], c[2] + v[2] - Ect[2]]);
    });
  }
  // World positions of the material markers and, for a twist, their first-order displacement v + w x r (metres).
  function markers(sc, tw) {
    return [0, 1].map(b => {
      const R = sc.pose[b].slice(0, 9), c = sc.com[b], v = tw ? tw.slice(6 * b, 6 * b + 3).map(x => x * L) : null, w = tw ? tw.slice(6 * b + 3, 6 * b + 6) : null;
      return ANISO.markers[b].map(q => { const r = matv(R, q); return { p: add(c, r), d: tw ? add(v, crossv(w, r)) : null }; });
    });
  }
  // Root-mean-square displacement of each body's surface samples under a twist, in mm (exact, from the moments).
  function bodyRms(sc, tw) { return [0, 1].map(b => { const x = tw.slice(6 * b, 6 * b + 6), K = sc.K[b]; let s = 0; for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++) s += x[i] * K[i][j] * x[j]; return Math.sqrt(Math.max(s, 0)) * L * 1000; }); }
  const quad = (A, x) => { let s = 0; for (let i = 0; i < x.length; i++) for (let j = 0; j < x.length; j++) s += x[i] * A[i][j] * x[j]; return s; };
  // Seeded standard normals (mulberry32 and Box-Muller), so that every reader sees the same draws.
  function normals(seed) {
    let a = seed >>> 0;
    const u = () => { a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    return () => { let x = 0; while (x === 0) x = u(); return Math.sqrt(-2 * Math.log(x)) * Math.cos(2 * Math.PI * u()); };
  }
  function cholesky(A) {
    const n = A.length, C = A.map(r => r.map(() => 0));
    for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) { let s = A[i][j]; for (let k = 0; k < j; k++) s -= C[i][k] * C[j][k]; C[i][j] = i === j ? Math.sqrt(s) : s / C[j][j]; }
    return C;
  }
  // x = C^-T z: the shaped increment, whose covariance is (C C^T)^-1 = Gt^-1 when z is standard normal.
  function solveCT(C, z) { const n = z.length, x = new Array(n).fill(0); for (let i = n - 1; i >= 0; i--) { let s = z[i]; for (let k = i + 1; k < n; k++) s -= C[k][i] * x[k]; x[i] = s / C[i][i]; } return x; }
  const css = v => getComputedStyle(document.body).getPropertyValue(v).trim();
  // An arrow on the overlay of a SceneDiagramView, with a halo in the page colour so that it reads on top of the bodies.
  function arrow(view, a, b, color, alpha) {
    const p = view.project(a), q = view.project(b), c = view.ctx, d = [q[0] - p[0], q[1] - p[1]], l = Math.hypot(d[0], d[1]);
    c.save(); c.globalAlpha = alpha; c.lineCap = 'round';
    for (const [col, wd] of [[view.bg, 3.2], [color, 1.4]]) {
      c.strokeStyle = col; c.lineWidth = wd; c.beginPath(); c.moveTo(p[0], p[1]); c.lineTo(q[0], q[1]);
      if (l >= 6) { const u = [d[0] / l, d[1] / l], h = Math.min(6, 0.45 * l); c.moveTo(q[0] - h * u[0] + 0.5 * h * u[1], q[1] - h * u[1] - 0.5 * h * u[0]); c.lineTo(q[0], q[1]); c.lineTo(q[0] - h * u[0] - 0.5 * h * u[1], q[1] - h * u[1] + 0.5 * h * u[0]); }
      c.stroke();
    }
    if (l < 2) { c.fillStyle = color; c.beginPath(); c.arc(p[0], p[1], 1.6, 0, 2 * Math.PI); c.fill(); }
    c.restore();
  }
  // Unit vector from the view's target towards its eye: larger values of dot(p - target, eye) are nearer to the reader.
  const eyeDir = view => [Math.cos(view.view.el) * Math.cos(view.view.az), Math.cos(view.view.el) * Math.sin(view.view.az), Math.sin(view.view.el)];
  const fmt = (x, d = 1) => (Math.abs(x) < 0.05 && d === 1 ? x.toFixed(2) : x.toFixed(d));
  return { L, matv, crossv, add, dot3, expso3, mul3, twist, moved, markers, bodyRms, quad, normals, cholesky, solveCT, css, arrow, eyeDir, fmt };
})();
