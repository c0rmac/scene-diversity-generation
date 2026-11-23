"""The contact-pair search space: scenes of a mug (body A, index 0) and a spoon (body B, index 1) on a table in which the two bodies
always touch and never overlap.

STATE (one designated pair, k = 1): yA and yB, points of space in the frames of the mug and of the spoon that single out a place on
each surface (the nearest surface sample); theta, the turn of the spoon about the common normal; u, a unit vector, the direction of
gravity as seen from the mug. Arrays: yA, yB (n, k, 3), theta (n,), u (n, 3).

MAP state -> scene, every step explicit (page Section 4):
 1. contact point and normal on each body: the nearest sample p (q) and the normal blended from the m nearest samples with weights
    exp(-(d_k - d_1) / delta); on a face it is the face normal, round a sharp edge it sweeps between the two faces' normals.
 2. relative pose of the spoon in the mug's frame: R = Q(nu_A, theta) R0(nu_B, -nu_A), t = p - R q (the normals oppose and the
    chosen points coincide). For k >= 2 designated pairs: the rotation and translation that best fit all pairs (closed form, SVD).
 3. push-off: the spoon is moved along nu_A by the smallest distance s* >= 0 at which the two ball unions, each inflated by half the
    clearance, do not overlap (exact interval sweep). s* equals the clearance when the designated contact is realised.
 4. gravity and table: the mug is turned so that u points down, the spoon follows rigidly, the pair is raised so that its lowest point
    is at the clearance above the table, and the mug's centre of mass is put over the origin.
The scene never overlaps in the ball-union model, the bodies are exactly the clearance apart at their closest, and the pair stands at
the clearance above the table.

STARTING LAW (law='point', the default; page Section 5): the mug's contact place is uniform over its surface (y_A uniform by volume
in the layer within the skin radius outside the mug), and given it, the spoon's place and the turn are uniform among those that
realise the contact there (candidates are drawn until one fits, at most TRIES). u is uniform on the sphere.

REGIONS (Section 5.1): restricted(maskA, maskB) restricts the contact places to subsets of the samples (pocket_mask, predicate_mask,
or any boolean array); every state of the restricted chart decodes to a place of the region.

STABLE MUG (Sections 5.2, 5.3): stable_mug(postures) keeps the mug in one of its own resting postures, standing on the table by itself;
u is decoded to the nearest allowed posture; a spoon that would reach below the table is raised straight up until clear. The
starting law then draws the posture together with the spoon's place and turn. stable_mug('upright') keeps the mug upright.

This is a standalone copy of the research code's pair_chart_v1.py and pair_chart_v2.py (23 Nov 2026), with the same arithmetic; the
only change is where the bodies come from (data/bodies.npz instead of the research formulation)."""
import copy, os
import numpy as np
from scipy.spatial import cKDTree
from .rotations import EZ, unit, rot_axis, minrot, section
from .bodies import load_bodies
from .lift import Lifter
from .postures import resting_postures


class PairChart:
    def __init__(self, bodies=None, k=1, H=None, delta=None, m=48, threads=None, clear=None, data_dir=None):
        """bodies: [mug, spoon] (default: load_bodies(data_dir)); k: number of designated pairs; H: thickness of the layer in which
        y is sampled (default the skin radius); delta: the normal blending length (default a third of the skin radius); m: samples
        blended; clear: the clearance (default (sqrt 2 - 1) x the 99 % quantile of the clouds' nearest-sample spacing)."""
        self.bodies = bodies or load_bodies(data_dir); self.data_dir = data_dir; self.k = int(k); self.threads = int(threads or os.cpu_count() or 1); self.m = int(m); bd = self.bodies
        self.P = [np.ascontiguousarray(b.p, float) for b in bd]; self.Nn = [np.ascontiguousarray(b.n, float) for b in bd]; self.tree = [cKDTree(p) for p in self.P]
        self.spacing99 = max(float(np.quantile(tr.query(p, k=2, workers=self.threads)[0][:, 1], 0.99)) for tr, p in zip(self.tree, self.P))
        self.clear = (np.sqrt(2.0) - 1.0) * self.spacing99 if clear is None else float(clear)
        self.skin = float(bd[0].rmax); self.H = self.skin if H is None else float(H); self.delta = self.skin / 3.0 if delta is None else float(delta)
        self.lam = float(bd[1].rms); self.com = [np.asarray(b.com, float) for b in bd]
        self.lifter = Lifter(bd[1], bd[0], inflate=0.5 * self.clear)      # the spoon moves; balls inflated by half the clearance each
        self.box = [(p.min(0) - self.H, p.max(0) + self.H) for p in self.P]; self.adm = self.skin; self.region = [None, None]; self.stable = None; self._pockets = None; self._build_region()

    def _build_region(self):
        self.rtree, self.ridx = [], []
        for b in (0, 1):
            m = self.region[b]
            if m is None: self.rtree.append(self.tree[b]); self.ridx.append(None)
            else:
                m = np.asarray(m, bool); assert m.shape == (len(self.P[b]),) and m.any(), 'region mask'; idx = np.flatnonzero(m); self.region[b] = m; self.rtree.append(cKDTree(self.P[b][idx])); self.ridx.append(idx)

    def restricted(self, maskA=None, maskB=None):
        """the same search space with the contact places restricted to regions of the two surfaces (one boolean per surface sample,
        or None for the whole surface). Every state of the restricted chart decodes to a place of the region."""
        c = copy.copy(self); c.region = [maskA, maskB]; c._build_region(); return c

    def declaration(self):
        return dict(k=self.k, region_samples=[None if m is None else int(m.sum()) for m in self.region], clearance_m=self.clear, spacing_q99_m=self.spacing99, shell_H_m=self.H, admissible_within_m=self.adm,
                    blend_delta_m=self.delta, blend_neighbours=self.m, normal_weight_lam_m=self.lam, samples=[len(p) for p in self.P],
                    stable_mug=None if self.stable is None else dict(which=self.stable['which'], labels=self.stable['label'], g=self.stable['g'].tolist(), com_height_mm=self.stable['com_height_mm']))

    # ---------------------------------------------------------------- step 1
    def contact(self, b, y):
        """y (..., 3) in body b's frame -> sample index, surface point p, blended unit normal nu"""
        sh = y.shape[:-1]; Y = y.reshape(-1, 3); m = min(self.m, self.rtree[b].n); d, j = self.rtree[b].query(Y, k=m, workers=self.threads)
        if m == 1: d = d[:, None]; j = j[:, None]
        if self.ridx[b] is not None: j = self.ridx[b][j]
        w = np.exp(-(d - d[:, :1]) / self.delta)
        nu = (w[:, :, None] * self.Nn[b][j]).sum(1); nn = np.linalg.norm(nu, axis=1); bad = nn < 1e-6; nu = nu / np.maximum(nn, 1e-300)[:, None]; nu[bad] = self.Nn[b][j[bad, 0]]
        return j[:, 0].reshape(sh), self.P[b][j[:, 0]].reshape(sh + (3,)), nu.reshape(sh + (3,))

    # ---------------------------------------------------------------- step 2
    def register(self, pA, nA, pB, nB, theta):
        """relative pose (R, t) of the spoon in the mug's frame from contact points and normals (n, k, 3)"""
        N, k = pA.shape[:2]
        if k == 1:
            R = rot_axis(nA[:, 0], theta) @ minrot(nB[:, 0], -nA[:, 0]); t = pA[:, 0] - np.einsum('nij,nj->ni', R, pB[:, 0]); return R, t
        pm = pA.mean(1); qm = pB.mean(1); Hm = np.einsum('nki,nkj->nij', pB - qm[:, None], pA - pm[:, None]) + self.lam ** 2 * np.einsum('nki,nkj->nij', nB, -nA)
        U, S, Vt = np.linalg.svd(Hm); V = np.swapaxes(Vt, 1, 2); Ut = np.swapaxes(U, 1, 2); dt = np.linalg.det(V @ Ut); D = np.zeros((N, 3, 3)); D[:, 0, 0] = 1; D[:, 1, 1] = 1; D[:, 2, 2] = dt
        R = V @ D @ Ut; return R, pm - np.einsum('nij,nj->ni', R, qm)

    # ---------------------------------------------------------------- step 3
    def backoff(self, R, t, d, max_iter=200):
        """smallest s >= 0 with the spoon moved by s d free of the mug (mug frame); returns s, status (2: iteration limit)"""
        Q = minrot(d, np.broadcast_to(EZ, d.shape).copy()); cB = np.einsum('nij,j->ni', R, self.com[1]) + t
        L = self.lifter.lift(Q @ R, np.einsum('nij,nj->ni', Q, cB), Q, np.einsum('nij,j->ni', Q, self.com[0]), use_other=True, use_table=False, threads=self.threads, max_iter=max_iter)
        return L['t'], L['status']

    def _lowest(self, b, R):
        """lowest point of body b along world z relative to its origin"""
        return self.bodies[b].lowest(R)

    # ---------------------------------------------------------------- the map
    def relative(self, yA, yB, theta=None):
        """steps 1 to 3: the spoon's pose (R, t) in the mug's frame and the intermediate quantities"""
        jA, pA, nA = self.contact(0, yA); jB, pB, nB = self.contact(1, yB); N = len(yA)
        R, t = self.register(pA, nA, pB, nB, np.zeros(N) if theta is None else theta); d = nA.sum(1); nd = np.linalg.norm(d, axis=1); d = np.where((nd > 1e-6)[:, None], d / np.maximum(nd, 1e-300)[:, None], nA[:, 0])
        s, st = self.backoff(R, t, d); t = t + s[:, None] * d
        gap = np.linalg.norm(pA - (np.einsum('nij,nkj->nki', R, pB) + t[:, None]), axis=2)      # designated pairs after the push-off
        return R, t, dict(backoff=s, status=st, pair_dist=gap, jA=jA, jB=jB, d=d, nA=nA, nB=nB, pA=pA, pB=pB)

    def realised(self, pA, nA, pB, nB, theta):
        """k = 1: is the designated contact realised (push-off within the skin radius)? (contact points and normals (n, 1, 3))"""
        R, t = self.register(pA, nA, pB, nB, theta); d = nA[:, 0]; s, st = self.backoff(R, t, d, max_iter=1); ok = (st == 0) | (s <= self.adm); k = np.flatnonzero(ok & (st != 0))
        if len(k):
            s2, _ = self.backoff(R[k], t[k], d[k]); ok[k] = s2 <= self.adm
        return ok

    def realize(self, yA, yB, theta, u):
        """state -> scene: R (n, 2, 3, 3), t (n, 2, 3) with x_world = R_b x_body + t_b (b = 0 mug, 1 spoon), and info"""
        if self.stable is not None: return self._realize_stable(yA, yB, theta, u)
        Rr, tr, info = self.relative(yA, yB, theta); u = unit(np.asarray(u, float)); RA = section(u); RB = RA @ Rr; tB0 = np.einsum('nij,nj->ni', RA, tr)
        lo = np.minimum(self._lowest(0, RA), self._lowest(1, RB) + tB0[:, 2]); cw = np.einsum('nij,j->ni', RA, self.com[0])
        tA = np.stack([-cw[:, 0], -cw[:, 1], -lo + self.clear], 1); R = np.stack([RA, RB], 1); t = np.stack([tA, tB0 + tA], 1); info.update(Rrel=Rr, trel=tr); return R, t, info

    # ---------------------------------------------------------------- the stable mug
    def stable_mug(self, postures='classes'):
        """the same search space with the mug kept in one of its own resting postures on the table: 'classes' (upright, upside down,
        on its side either way), 'upright', 'all', or a list of library labels (see postures.resting_postures)"""
        assert self.k == 1, 'the stable-mug option is built for one designated pair'
        c = copy.copy(self); lib = resting_postures(0, postures); S = section(lib['g']); c.stable = dict(lib, S=S, zA=-self._lowest(0, S) + self.clear); return c

    def posture(self, u):
        """index of the allowed posture whose downward direction is nearest to u (n, 3)"""
        return np.argmax(unit(np.asarray(u, float)) @ self.stable['g'].T, axis=1)

    def _realize_stable(self, yA, yB, theta, u):
        S = self.stable; Rr, tr, info = self.relative(yA, yB, theta); m = self.posture(u); RA = S['S'][m]; RB = RA @ Rr; n = len(m)
        cw = np.einsum('nij,j->ni', RA, self.com[0]); tA = np.stack([-cw[:, 0], -cw[:, 1], S['zA'][m]], 1); tB = np.einsum('nij,nj->ni', RA, tr) + tA
        h = np.zeros(n); hs = np.zeros(n, np.int32); k = np.flatnonzero(self._lowest(1, RB) + tB[:, 2] < self.clear - 1e-9)      # the spoon reaches below the table: raise it
        if len(k):
            dz = np.array([0.0, 0.0, 0.5 * self.clear]); cA = cw[k] + tA[k] - dz; cB = np.einsum('nij,j->ni', RB[k], self.com[1]) + tB[k] - dz
            L = self.lifter.lift(RB[k], cB, RA[k], cA, use_other=True, use_table=True, threads=self.threads); h[k] = L['t']; hs[k] = L['status']
        tB[:, 2] += h; trel = tr + h[:, None] * RA[:, 2, :]; pd = np.linalg.norm(info['pA'] - (np.einsum('nij,nkj->nki', Rr, info['pB']) + trel[:, None]), axis=2)
        info.update(Rrel=Rr, trel=trel, posture=m, table_raise=h, raise_status=hs, pair_dist_before_raise=info['pair_dist'], pair_dist=pd, admissible=(info['backoff'] <= self.adm) & (h == 0))
        return np.stack([RA, RB], 1), np.stack([tA, tB], 1), info

    def realised_stable(self, pA, nA, pB, nB, theta, m):
        """is the combination admissible with the mug in posture m (n,): the designated contact realised and the spoon above the table?"""
        S = self.stable; R, t = self.register(pA, nA, pB, nB, theta); d = nA[:, 0]; RA = S['S'][m]; c = self.clear
        z0 = self._lowest(1, RA @ R) + np.einsum('nj,nj->n', RA[:, 2, :], t) + S['zA'][m]; dz = np.einsum('nj,nj->n', RA[:, 2, :], d)      # the spoon's lowest point at push-off s is z0 + s dz
        ok = np.maximum(z0, z0 + self.adm * dz) >= c - 1e-9; k = np.flatnonzero(ok)
        if len(k):
            s, st = self.backoff(R[k], t[k], d[k], max_iter=1); kk = np.flatnonzero((s <= self.adm) & (st != 0))
            if len(kk): s[kk] = self.backoff(R[k[kk]], t[k[kk]], d[k[kk]])[0]
            ok[k] = (s <= self.adm) & (z0[k] + s * dz[k] >= c - 1e-9)
        return ok

    # ---------------------------------------------------------------- starting law
    def sample_shell(self, b, n, rng, mask=None):
        """n points uniform by volume in {0 < distance to the surface <= H, outside the body}; with a mask (one boolean per surface
        sample) only the part of the layer whose nearest sample is in the mask: a region of the surface"""
        lo, hi = self.box[b]; out = np.empty((0, 3)); mask = self.region[b] if mask is None else mask
        if mask is not None:
            mask = np.asarray(mask, bool); assert mask.shape == (len(self.P[b]),) and mask.any(), 'region mask'; lo, hi = self.P[b][mask].min(0) - self.H, self.P[b][mask].max(0) + self.H
        while len(out) < n:
            Y = rng.uniform(lo, hi, size=(max(4 * n, 200000), 3)); d, j = self.tree[b].query(Y, k=1, distance_upper_bound=self.H, workers=self.threads); ok = np.isfinite(d)
            ok[ok] &= ((Y[ok] - self.P[b][j[ok]]) * self.Nn[b][j[ok]]).sum(1) > 0
            if mask is not None: ok[ok] &= mask[j[ok]]
            out = np.concatenate([out, Y[ok]])
        return out[:n]

    def pocket_mask(self, b, pocket=1):
        """samples of body b that face its pocket number POCKET (for the mug, 1 is the inside of the cup): the point one skin radius
        outside the sample lies in the pocket"""
        if self._pockets is None:
            from .pockets import PocketMap
            self._pockets = PocketMap(self.data_dir)
        return self._pockets.lookup(b, self.P[b] + self.skin * self.Nn[b]) == int(pocket)

    def predicate_mask(self, b, fn):
        """samples of body b for which fn(points, normals) (arrays in the body's frame) is true: any region"""
        return np.asarray(fn(self.P[b], self.Nn[b]), bool)

    def sample_start(self, N, seed, admissible=True, law='point', angles=24, region=None, lead=0, tries=4096):
        """the starting law: N states (dict of arrays yA, yB, theta, u, tries, ...) for SEED. law='point' (default, k = 1): the
        leading body's contact place (lead 0: the mug) uniform over its surface or region, the other body's place and the turn uniform
        among those that realise the contact there. Older laws for the record: 'pairs', 'triples' (see the page's Section 5).
        admissible=False: every draw is kept."""
        if region is not None: return self.restricted(*region).sample_start(N, seed, admissible, law, angles, None, lead, tries)
        rng = np.random.default_rng(np.random.SeedSequence([seed, 1121, self.k])); acc = None; drawn = 0
        if admissible and law == 'point' and self.k == 1: return self._point_led(N, rng, None, int(lead), int(tries))
        while True:
            st = self._draw(max(N, 2000), rng, region); drawn += len(st['u']); st['free_angles'] = np.full(len(st['u']), np.nan)
            if admissible and law == 'pairs' and self.k == 1:
                M = len(st['u']); _, pA, nA = self.contact(0, st['yA']); _, pB, nB = self.contact(1, st['yB']); T = int(angles); th = (2 * np.pi * (np.arange(T)[None, :] + rng.random((M, 1))) / T)
                rp = lambda a: np.repeat(a, T, axis=0); ok = self.realised(rp(pA), rp(nA), rp(pB), rp(nB), th.ravel()).reshape(M, T); nf = ok.sum(1); keep = nf > 0
                pick = np.argmax(np.cumsum(ok, 1) > (rng.random(M) * np.maximum(nf, 1))[:, None], axis=1); st['theta'] = th[np.arange(M), pick]; st['free_angles'] = nf / T; st = {kk: v[keep] for kk, v in st.items()}
            elif admissible:
                _, _, info = self.relative(st['yA'], st['yB'], st['theta']); keep = (info['pair_dist'] <= self.adm).all(1); st = {kk: v[keep] for kk, v in st.items()}
            acc = st if acc is None else {kk: np.concatenate([acc[kk], st[kk]]) for kk in st}
            if len(acc['u']) >= N: break
        out = {kk: v[:N] for kk, v in acc.items()}; out['acceptance'] = len(acc['u']) / drawn; return out

    def _point_led(self, N, rng, region, lead, tries):
        if self.stable is not None: return self._point_led_stable(N, rng, lead, tries)
        masks = (None, None); other = 1 - lead; acc = None; drawn = 0; dropped = []
        while True:
            M = max(N, 2000); yL = self.sample_shell(lead, M, rng, masks[lead]); _, pL, nL = self.contact(lead, yL[:, None, :]); drawn += M
            yO = np.full((M, 3), np.nan); th = np.full(M, np.nan); used = np.zeros(M, np.int64); cost = np.zeros(M, np.int64); pend = np.arange(M); b = 8
            while len(pend) and used[pend[0]] < tries:
                m = len(pend); yc = self.sample_shell(other, m * b, rng, masks[other]); tc = rng.uniform(0, 2 * np.pi, m * b); _, pc, nc = self.contact(other, yc[:, None, :]); rp = lambda a: np.repeat(a[pend], b, axis=0)
                ok = (self.realised(rp(pL), rp(nL), pc, nc, tc) if lead == 0 else self.realised(pc, nc, rp(pL), rp(nL), tc)).reshape(m, b); hit = ok.any(1); first = ok.argmax(1)
                idx = pend[hit]; sel = np.flatnonzero(hit) * b + first[hit]; yO[idx] = yc[sel]; th[idx] = tc[sel]; cost[idx] = used[idx] + first[hit] + 1; used[pend] += b; pend = pend[~hit]; b = min(4 * b, 128)
            found = np.isfinite(th); dropped.append(yL[~found]); yA, yB = (yL, yO) if lead == 0 else (yO, yL); g = rng.standard_normal((M, 3))
            st = dict(yA=yA[found][:, None, :], yB=yB[found][:, None, :], theta=th[found], u=unit(g)[found], start_mismatch=np.zeros((int(found.sum()), 1)), free_angles=np.full(int(found.sum()), np.nan), tries=cost[found])
            acc = st if acc is None else {kk: np.concatenate([acc[kk], st[kk]]) for kk in st}
            if len(acc['u']) >= N: break
        out = {kk: v[:N] for kk, v in acc.items()}; out['acceptance'] = len(acc['u']) / drawn; out['dropped_leading'] = np.concatenate(dropped); return out

    def _point_led_stable(self, N, rng, lead, tries):
        assert lead == 0, 'with the stable mug the mug leads'
        S = self.stable; G = S['g']; Sz = S['S'][:, 2, :]; acc = None; drawn = 0; dropped, dropped_table = [], []
        while True:
            M = max(N, 2000); yL = self.sample_shell(0, M, rng, None); _, pL, nL = self.contact(0, yL[:, None, :]); drawn += M
            # postures in which this place's contact point can be above the table at some admissible push-off (necessary condition)
            feas = (pL[:, 0] @ Sz.T + S['zA'][None, :] + np.maximum(0.0, self.adm * (nL[:, 0] @ Sz.T))) >= self.clear - 1e-9; can = feas.any(1)
            yO = np.full((M, 3), np.nan); th = np.full(M, np.nan); pm = np.full(M, -1, np.int64); used = np.zeros(M, np.int64); cost = np.zeros(M, np.int64); pend = np.flatnonzero(can); b = 8
            while len(pend) and used[pend[0]] < tries:
                m_ = len(pend); yc = self.sample_shell(1, m_ * b, rng, None); tc = rng.uniform(0, 2 * np.pi, m_ * b); _, pc, nc = self.contact(1, yc[:, None, :]); rp = lambda a: np.repeat(a[pend], b, axis=0)
                cum = np.cumsum(rp(feas), 1); mc = np.argmax(cum > (rng.random(m_ * b) * cum[:, -1])[:, None], axis=1)      # posture uniform over the feasible ones
                ok = self.realised_stable(rp(pL), rp(nL), pc, nc, tc, mc).reshape(m_, b); hit = ok.any(1); first = ok.argmax(1)
                idx = pend[hit]; sel = np.flatnonzero(hit) * b + first[hit]; yO[idx] = yc[sel]; th[idx] = tc[sel]; pm[idx] = mc[sel]; cost[idx] = used[idx] + first[hit] + 1; used[pend] += b; pend = pend[~hit]; b = min(4 * b, 128)
            found = np.isfinite(th); dropped.append(yL[can & ~found]); dropped_table.append(yL[~can]); nf = int(found.sum())
            st = dict(yA=yL[found][:, None, :], yB=yO[found][:, None, :], theta=th[found], u=G[pm[found]], posture=pm[found], start_mismatch=np.zeros((nf, 1)), free_angles=np.full(nf, np.nan), tries=cost[found])
            acc = st if acc is None else {kk: np.concatenate([acc[kk], st[kk]]) for kk in st}
            if len(acc['u']) >= N: break
        out = {kk: v[:N] for kk, v in acc.items()}; out['acceptance'] = len(acc['u']) / drawn; out['dropped_leading'] = np.concatenate(dropped); out['dropped_table'] = np.concatenate(dropped_table); return out

    def _draw(self, N, rng, region=None):
        k = self.k; mA, mB = None, None
        yA = self.sample_shell(0, N * k, rng, mA).reshape(N, k, 3); yB = np.empty((N, k, 3)); yB[:, 0] = self.sample_shell(1, N, rng, mB); mism = np.zeros((N, k))
        if k > 1:
            jA, pA, _ = self.contact(0, yA); PB = self.P[1]; NB = self.Nn[1]; _, q0, _ = self.contact(1, yB[:, 0]); Q = [q0]
            for i in range(1, k):
                yi = np.empty((N, 3)); qi = np.empty((N, 3)); r = rng.random(N)
                for a in range(N):      # samples of B whose distances to the points already drawn match those on A
                    e = np.zeros(len(PB))
                    for c in range(i): e = np.maximum(e, np.abs(np.linalg.norm(PB - Q[c][a], axis=1) - np.linalg.norm(pA[a, i] - pA[a, c])))
                    cand = np.flatnonzero(e <= self.H); jj = cand[int(r[a] * len(cand))] if len(cand) else int(np.argmin(e)); mism[a, i] = e[jj]; qi[a] = PB[jj]; yi[a] = PB[jj] + 0.5 * self.H * NB[jj]
                yB[:, i] = yi; Q.append(qi)
        g = rng.standard_normal((N, 3)); u = unit(g); theta = rng.uniform(0, 2 * np.pi, N)
        return dict(yA=yA, yB=yB, theta=theta, u=u, start_mismatch=mism)
