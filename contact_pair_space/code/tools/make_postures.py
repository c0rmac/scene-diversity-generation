"""Recomputes the resting-posture library (data/postures.json) from the bodies' primitives, exactly and deterministically.

A body alone on the table with the direction g of its own frame pointing down has its centre of mass at h(g) = sigma(g) - c . g
(sigma the support function of its primitives, closed form per primitive; contact_pair.primitives.Height). It rests where h has a
strict local minimum.
  lattice      h on a Fibonacci lattice of M directions (equal area); each point's descent to its lowest neighbour among its k
               nearest gives the lattice minima and their basins.
  merge tree   sublevel sets of h on the lattice graph: edges weighted max(h_p, h_q) in ascending order (minimum spanning tree),
               union-find; when two basins meet at height w the one with the higher minimum dies with persistence w - h(min): the
               least rise of the centre of mass that rolls it into the other.
  coarse       persistence-based clustering (ToMATo, Chazal et al. 2013): a basin merges into its neighbour iff its persistence is
               below tau, tau the middle of the largest gap in the sorted persistences.
  fine         every lattice minimum refined by Nelder-Mead in a tangent chart; kept if it is a strict local minimum on a ring at
               the rest-tolerance angle (1 mm over the body's largest radius) and has not drifted more than 2 degrees; minima closer
               than that angle are merged.
This is the research code's exact_v1.py and exact_v3.py (19 Nov 2026) with the primitives read from data/primitives.json.
usage: python tools/make_postures.py [OUT.json] [--M 400000] [--k 10]   (the mug takes about a minute, the spoon longer)"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from contact_pair.primitives import load_primitives, Height, body_points

NAMES = ('mug', 'spoon')


def fibonacci(M):
    i = np.arange(M) + 0.5; z = 1 - 2 * i / M; r = np.sqrt(1 - z * z); ph = i * np.pi * (3 - np.sqrt(5)); return np.column_stack([r * np.cos(ph), r * np.sin(ph), z])


def chart(g0):
    a = np.array([1.0, 0, 0]) if abs(g0[0]) < 0.9 else np.array([0, 1.0, 0]); e1 = np.cross(g0, a); e1 /= np.linalg.norm(e1); e2 = np.cross(g0, e1)
    return lambda x: (lambda v: v / np.linalg.norm(v))(g0 + x[0] * e1 + x[1] * e2)


def refine(h, g0):
    f = chart(g0); obj = lambda x: float(h(f(x))[0])
    r = minimize(obj, np.zeros(2), method='Nelder-Mead', options=dict(xatol=1e-11, fatol=1e-14, maxiter=4000, initial_simplex=np.array([[0, 0], [5e-3, 0], [0, 5e-3]])))
    return f(r.x), float(r.fun)


def strict_min(h, g, val, delta=1e-3, n=72):
    f = chart(g); th = np.linspace(0, 2 * np.pi, n, endpoint=False); ring = np.array([f(delta * np.array([np.cos(a), np.sin(a)])) for a in th])
    return bool((h(ring) > val + 1e-12).all()), float((h(ring) - val).min())


class UF:
    def __init__(self, n, h): self.p = np.arange(n); self.m = np.arange(n); self.h = h

    def find(self, a):
        r = a
        while self.p[r] != r: r = self.p[r]
        while self.p[a] != r: self.p[a], a = r, self.p[a]
        return r


def study(name, body, M, k, floor):
    tic = time.time(); H = Height(body['geoms'], body['com']); G = fibonacci(M); hv = H(G); _, nb = cKDTree(G).query(G, k + 1); nb = nb[:, 1:]
    j = np.where(hv[nb].min(1) < hv, nb[np.arange(M), hv[nb].argmin(1)], np.arange(M))
    for _ in range(400):
        j2 = j[j]
        if np.array_equal(j2, j): break
        j = j2
    lmins = np.unique(j)
    rows = np.repeat(np.arange(M), k); cols = nb.ravel(); w = np.maximum(hv[rows], hv[cols]) - hv.min() + 1e-12
    T = minimum_spanning_tree(coo_matrix((w, (rows, cols)), shape=(M, M)).tocsr()).tocoo(); order = np.argsort(T.data, kind='stable')
    E = np.column_stack([T.row[order], T.col[order]]); Wt = T.data[order] + hv.min() - 1e-12; uf = UF(M, hv); pers = {}
    for (a, b), wt in zip(E, Wt):
        ra, rb = uf.find(a), uf.find(b)
        if ra == rb: continue
        ma, mb = uf.m[ra], uf.m[rb]; young, old = (ra, rb) if hv[ma] > hv[mb] else (rb, ra); pers[int(uf.m[young])] = float(wt - hv[uf.m[young]]); uf.p[young] = old
    P = np.array(sorted(pers.values())); gaps = np.diff(np.concatenate([[0.0], P])); g_i = int(np.argmax(gaps)); tau = float(0.5 * (np.concatenate([[0.0], P])[g_i] + np.concatenate([[0.0], P])[g_i + 1]))
    uf2 = UF(M, hv)
    for (a, b), wt in zip(E, Wt):
        ra, rb = uf2.find(a), uf2.find(b)
        if ra == rb: continue
        ma, mb = uf2.m[ra], uf2.m[rb]; young, old = (ra, rb) if hv[ma] > hv[mb] else (rb, ra)
        if wt - hv[uf2.m[young]] < tau: uf2.p[young] = old
    root = np.array([uf2.find(i) for i in range(M)]); roots = np.unique(root); coarse_lat = np.searchsorted(roots, root) + 1
    mins, vals, src = [], [], []
    for c in lmins:
        g, v = refine(H, G[c])
        if np.arccos(np.clip(g @ G[c], -1, 1)) > np.radians(2.0) or not strict_min(H, g, v, delta=floor)[0]: continue
        hit = [i for i, m in enumerate(mins) if np.arccos(np.clip(g @ m, -1, 1)) < floor]
        if hit:
            if v < vals[hit[0]]: mins[hit[0]], vals[hit[0]] = g, v
            src[hit[0]].append(int(c))
        else: mins.append(g); vals.append(v); src.append([int(c)])
    mins = np.array(mins); vals = np.array(vals); share = np.zeros(len(mins)); cls = []
    _, near_lat = cKDTree(G).query(mins) if len(mins) else (None, np.zeros(0, int))
    for i, s in enumerate(src):
        for c in s: share[i] += (j == c).sum() / M
    for i in np.argsort(-share):
        c0 = int(near_lat[i]); strict, rise = strict_min(H, mins[i], vals[i], delta=floor)
        cls.append(dict(label=int(i + 1), coarse=int(coarse_lat[c0]), share=float(share[i]), rep_g=mins[i].tolist(), com_height_mm=float(1e3 * vals[i]),
                        persistence_mm=(float(1e3 * min(pers.get(c, np.inf) for c in src[i])) if any(c in pers for c in src[i]) else None), strict_at_floor=strict, ring_rise_at_floor_um=float(1e6 * rise), lattice_minima=len(src[i])))
    coarse_info = []
    for r_ in range(1, len(roots) + 1):
        m_ = coarse_lat == r_; lo = int(np.flatnonzero(m_)[np.argmin(hv[m_])])
        coarse_info.append(dict(coarse=r_, basin_share=float(m_.mean()), lowest_g=G[lo].tolist(), lowest_com_height_mm=float(1e3 * hv[lo]), fine=[c['label'] for c in cls if c['coarse'] == r_]))
    return dict(body=name, M=M, k=k, lattice_minima=int(len(lmins)), fine_postures=int(len(mins)), tau_mm=1e3 * tau, persistence_sorted_mm=[float(1e3 * x) for x in P], coarse=coarse_info, classes=cls,
                cut=dict(floor_deg=float(np.degrees(floor)), tau_mm=1e3 * tau, fine_classes=int(len(mins)), coarse_classes=int(len(roots))), seconds=time.time() - tic)


def library(M=400000, k=10, bodies=NAMES):
    prim = load_primitives(); rmax = {nm: float(np.linalg.norm(body_points(prim[nm]) - prim[nm]['com'], axis=1).max()) for nm in NAMES}; floor = {nm: 1e-3 / rmax[nm] for nm in NAMES}
    res = dict(rule=__doc__, r_max_m=[rmax[nm] for nm in NAMES], floor_deg=[float(np.degrees(floor[nm])) for nm in NAMES])
    for nm in bodies: res[nm] = study(nm, prim[nm], M, k, floor[nm])
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('out', nargs='?', default='postures_recomputed.json'); ap.add_argument('--M', type=int, default=400000); ap.add_argument('--k', type=int, default=10); a = ap.parse_args()
    res = library(a.M, a.k); Path(a.out).write_text(json.dumps(res) + '\n')
    for nm in NAMES: z = res[nm]; print('%s: %d lattice minima -> %d resting postures in %d classes (tau %.3f mm), %.0f s' % (nm, z['lattice_minima'], z['fine_postures'], z['cut']['coarse_classes'], z['tau_mm'], z['seconds']))
    print('written', a.out)
