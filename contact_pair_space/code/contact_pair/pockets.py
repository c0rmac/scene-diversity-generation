"""Pockets of a body: the connected parts of the space inside its convex hull but outside the body (Section 5.1 of the page). For the
mug, pocket 1 is the inside of the cup and pocket 2 the space around the handle. Regions of the search space are made from them: a
surface sample faces a pocket when the point one skin radius outside it lies in that pocket.

The grids are computed once from the collision primitives on a 1 mm voxel grid (compute_pockets; tools/make_pockets.py writes
data/pockets.npz) and used through PocketMap, which opens each raw grid with a 3 mm cube (removing slivers thinner than the voxel
grid can resolve), labels its face-connected parts and numbers them by decreasing volume."""
from pathlib import Path
import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, Delaunay
from .primitives import surface_points, inside_body, dense_surface

DATA = Path(__file__).resolve().parent.parent / 'data'
NAMES = ('mug', 'spoon'); VOX = 1e-3


class PocketMap:
    def __init__(self, data_dir=None):
        pk = np.load(Path(data_dir or DATA) / 'pockets.npz'); self.vox = float(pk['voxel']); self.lab, self.origin = [], []
        for nm in NAMES:
            raw = pk[f'{nm}_label']; opened = ndimage.binary_opening(raw > 0, structure=np.ones((3, 3, 3), bool)); lab, n = ndimage.label(opened, structure=ndimage.generate_binary_structure(3, 1))
            vol = np.bincount(lab.ravel(), minlength=n + 1); keep = np.flatnonzero(vol[1:] > 0) + 1; order = keep[np.argsort(-vol[keep])]; remap = np.zeros(n + 1, np.int16); remap[order] = np.arange(1, len(order) + 1)
            self.lab.append(remap[lab]); self.origin.append(pk[f'{nm}_origin'])

    def lookup(self, b, X):
        """pocket number (0: none) of points X (..., 3) in body b's frame"""
        idx = np.floor((X - self.origin[b]) / self.vox).astype(np.int64); L = self.lab[b]; ok = ((idx >= 0) & (idx < np.array(L.shape))).all(-1)
        out = np.zeros(X.shape[:-1], np.int16); out[ok] = L[idx[ok][:, 0], idx[ok][:, 1], idx[ok][:, 2]]; return out


def compute_pockets(geoms):
    """raw pocket grid of one body from its primitives: conv(B) minus B on a VOX grid, face-connected parts numbered by decreasing
    volume. Returns (labels int16 grid, origin (3,))"""
    S = surface_points(geoms); hull = ConvexHull(S); lo = S.min(0) - 2 * VOX; hi = S.max(0) + 2 * VOX
    shape = np.ceil((hi - lo) / VOX).astype(int); idx = np.indices(shape).reshape(3, -1).T; X = lo + (idx + 0.5) * VOX
    D = Delaunay(S[hull.vertices]); in_hull = np.zeros(len(X), bool)
    for a in range(0, len(X), 200000): in_hull[a:a + 200000] = D.find_simplex(X[a:a + 200000]) >= 0
    empty = (in_hull & ~inside_body(X, geoms)).reshape(shape); lab, n = ndimage.label(empty, structure=ndimage.generate_binary_structure(3, 1))
    vol = np.bincount(lab.ravel(), minlength=n + 1)[1:] * VOX ** 3; order = np.argsort(-vol); remap = np.zeros(n + 1, np.int32); remap[1 + order] = np.arange(1, n + 1)
    return remap[lab].astype(np.int16), lo


def compute_pocket_file(primitives):
    """the arrays of data/pockets.npz from the primitives (load_primitives())"""
    arr = dict(voxel=VOX)
    for nm in NAMES:
        lab, lo = compute_pockets(primitives[nm]['geoms']); arr[f'{nm}_label'] = lab; arr[f'{nm}_origin'] = lo; arr[f'{nm}_surface'] = dense_surface(primitives[nm]['geoms']); arr[f'{nm}_com'] = np.asarray(primitives[nm]['com'], float)
    return arr
