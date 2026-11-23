"""The bodies' collision primitives (boxes, capsules, cylinders, ellipsoids, spheres) from data/primitives.json, with the exact
support function h(g) used for resting postures and the inside tests and surface samplings used for pockets. Only the tools that
recompute the posture library and the pocket grids, and the tests, use this module; the map itself works on the point clouds."""
import json
from pathlib import Path
import numpy as np

DATA = Path(__file__).resolve().parent.parent / 'data'


def load_primitives(data_dir=None):
    """{'mug': dict(com, geoms=[(type, size, pos, R)]), 'spoon': ...} in each body's frame (metres)"""
    raw = json.loads((Path(data_dir or DATA) / 'primitives.json').read_text())
    return {nm: dict(com=np.asarray(raw[nm]['com'], float), geoms=[(g['type'], list(g['size']), np.asarray(g['pos'], float), np.asarray(g['R'], float)) for g in raw[nm]['geoms']]) for nm in ('mug', 'spoon')}


def dirs14():
    u = [s * e for e in np.eye(3) for s in (-1, 1)]; c = 1 / np.sqrt(3); u += [np.array([i, j, k]) * c for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]
    return np.array(u)


def dirs26():
    u = [np.array([i, j, k], float) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)]
    return np.array([v / np.linalg.norm(v) for v in u])


def geom_points(gtype, s):
    """a few points of each primitive (corners, rims, 14 or 26 directions): used for the body's largest radius"""
    if gtype == 'box': return np.array([[i * s[0], j * s[1], k * s[2]] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
    if gtype == 'cylinder': a = 2 * np.pi * np.arange(32) / 32; return np.array([[s[0] * np.cos(t), s[0] * np.sin(t), e * s[1]] for e in (-1, 1) for t in a])
    if gtype == 'capsule': return np.array([[s[0] * u[0], s[0] * u[1], e * s[1] + s[0] * u[2]] for e in (-1, 1) for u in dirs14()])
    if gtype == 'ellipsoid': return dirs26() * np.asarray(s[:3])
    if gtype == 'sphere': return dirs26() * s[0]
    raise ValueError(gtype)


def body_points(body):
    return np.vstack([geom_points(t, s) @ R.T + p for t, s, p, R in body['geoms']])


class Height:
    """h(g) = sigma(g) - c . g: the height of the centre of mass above the table when the direction g (body frame) points straight
    down; sigma is the support function of the primitives (closed form per primitive). Vectorised over g (n, 3)."""
    def __init__(self, geoms, com):
        self.g = [(t, np.asarray(s, float), np.asarray(p, float), np.asarray(R, float)) for t, s, p, R in geoms]; self.c = np.asarray(com, float)

    def __call__(self, g):
        g = np.atleast_2d(g); out = np.full(len(g), -np.inf)
        for t, s, p, R in self.g:
            cg = g @ p; Rg = g @ R
            if t == 'box': v = cg + (np.abs(Rg) * s[None, :]).sum(1)
            elif t == 'capsule': v = cg + s[1] * np.abs(Rg[:, 2]) + s[0]
            elif t == 'cylinder': v = cg + s[1] * np.abs(Rg[:, 2]) + s[0] * np.sqrt(np.maximum(1 - Rg[:, 2] ** 2, 0))
            elif t == 'ellipsoid': v = cg + np.sqrt(((Rg * s[None, :]) ** 2).sum(1))
            elif t == 'sphere': v = cg + s[0]
            else: raise ValueError(t)
            out = np.maximum(out, v)
        return out - g @ self.c


def sphere_dirs(step_deg):
    th = np.radians(np.arange(0, 180 + 1e-9, step_deg)); ph = np.radians(np.arange(0, 360, step_deg)); T, P = np.meshgrid(th, ph, indexing='ij')
    return np.unique(np.round(np.column_stack([(np.sin(T) * np.cos(P)).ravel(), (np.sin(T) * np.sin(P)).ravel(), np.cos(T).ravel()]), 12), axis=0)


def surface_points(geoms, step_deg=4.0):
    """a fixed sampling of the primitives' surfaces, dense enough for their convex hull"""
    U = sphere_dirs(step_deg); out = []
    for t, s, p, R in geoms:
        s = np.asarray(s, float); p = np.asarray(p, float); R = np.asarray(R, float); a = R[:, 2]
        if t == 'box': out.append(np.array([p + R @ (np.array([i, j, k]) * s) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]))
        elif t == 'capsule': out.append(np.vstack([p + e * s[1] * a + s[0] * U for e in (-1, 1)]))
        elif t == 'cylinder':
            ph = np.radians(np.arange(0, 360, step_deg)); ring = np.column_stack([np.cos(ph), np.sin(ph), np.zeros_like(ph)]) @ R.T; out.append(np.vstack([p + e * s[1] * a + s[0] * ring for e in (-1, 1)]))
        elif t == 'ellipsoid': out.append(p + (U * s[None, :]) @ R.T)
        elif t == 'sphere': out.append(p + s[0] * U)
        else: raise ValueError(t)
    return np.vstack(out)


def inside_body(X, geoms):
    """which points X (m, 3) lie inside some primitive"""
    ins = np.zeros(len(X), bool)
    for t, s, p, R in geoms:
        s = np.asarray(s, float); p = np.asarray(p, float); R = np.asarray(R, float); L = (X - p) @ R
        if t == 'box': ins |= (np.abs(L) <= s[None, :]).all(1)
        elif t == 'capsule': z = np.clip(L[:, 2], -s[1], s[1]); ins |= (L[:, 0] ** 2 + L[:, 1] ** 2 + (L[:, 2] - z) ** 2) <= s[0] ** 2
        elif t == 'cylinder': ins |= (np.abs(L[:, 2]) <= s[1]) & (L[:, 0] ** 2 + L[:, 1] ** 2 <= s[0] ** 2)
        elif t == 'ellipsoid': ins |= ((L / s[None, :]) ** 2).sum(1) <= 1
        elif t == 'sphere': ins |= (L ** 2).sum(1) <= s[0] ** 2
    return ins


def dense_surface(geoms, spacing=1e-3):
    """a fixed surface sampling of the body about SPACING apart (stored with the pocket grids)"""
    pts = []
    for t, s, p, R in geoms:
        s = np.asarray(s, float); p = np.asarray(p, float); R = np.asarray(R, float)
        if t == 'box':
            g = [np.arange(-si, si + 1e-12, spacing) if 2 * si > spacing else np.array([0.0]) for si in s]; X, Y, Z = np.meshgrid(*g, indexing='ij'); L = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
            L = L[(np.isclose(np.abs(L), s[None, :], atol=spacing / 2)).any(1)] if len(L) else L
        elif t in ('capsule', 'cylinder'):
            n_ph = max(8, int(np.ceil(2 * np.pi * s[0] / spacing))); ph = np.linspace(0, 2 * np.pi, n_ph, endpoint=False); zz = np.arange(-s[1], s[1] + 1e-12, spacing)
            Zg, Pg = np.meshgrid(zz, ph, indexing='ij'); L = np.column_stack([s[0] * np.cos(Pg).ravel(), s[0] * np.sin(Pg).ravel(), Zg.ravel()])
            if t == 'capsule':
                U = sphere_dirs(max(2.0, np.degrees(spacing / s[0]))); L = np.vstack([L, s[1] * np.array([0, 0, 1.0]) + s[0] * U[U[:, 2] >= 0], -s[1] * np.array([0, 0, 1.0]) + s[0] * U[U[:, 2] <= 0]])
        elif t == 'ellipsoid': U = sphere_dirs(max(1.0, np.degrees(spacing / s.max()))); L = U * s[None, :]
        elif t == 'sphere': U = sphere_dirs(max(1.0, np.degrees(spacing / s[0]))); L = s[0] * U
        pts.append(p + L @ R.T)
    return np.unique(np.round(np.vstack(pts), 9), axis=0)
