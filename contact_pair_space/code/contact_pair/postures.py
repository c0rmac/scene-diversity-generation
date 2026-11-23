"""Resting postures of a body alone on the table (Section 5.2 of the page). With the direction g of its own frame pointing straight
down, the body's centre of mass stands at h(g) = sigma(g) - c . g above the table (sigma the support function); it rests where h has
a strict local minimum, and minima are grouped into classes by the rise of the centre of mass needed to roll from one into another.
The library (data/postures.json) is computed exactly from the primitives by tools/make_postures.py; resting_postures picks from it."""
import hashlib, json
from pathlib import Path
import numpy as np
from .rotations import unit

LIB = Path(__file__).resolve().parent.parent / 'data' / 'postures.json'
NAMES = ('mug', 'spoon')


def resting_postures(b=0, which='classes', lib_path=None):
    """resting postures of body b (0 mug, 1 spoon):
      'classes' (default) the deepest posture of each class: for the mug upright, upside down, on its side with the handle down
                either way; 'all' every strict minimum; 'upright' the one posture with the body's own z axis up;
      a list of labels: those postures of the library.
    Returns dict(g (K, 3) downward directions in the body frame, label, coarse, com_height_mm, persistence_mm, library_sha256, which)."""
    path = Path(lib_path or LIB); raw = path.read_bytes(); cl = json.loads(raw)[NAMES[b]]['classes']
    if isinstance(which, str) and which == 'classes':
        pick = []
        for c in sorted(set(x['coarse'] for x in cl)):
            grp = [x for x in cl if x['coarse'] == c]; pick.append(min(grp, key=lambda x: (x['com_height_mm'], x['label'])))
    elif isinstance(which, str) and which == 'all': pick = sorted(cl, key=lambda x: x['label'])
    elif isinstance(which, str) and which == 'upright':
        x = min(cl, key=lambda x: (x['rep_g'][2], x['label'])); assert x['rep_g'][2] < -0.99, 'no resting posture with the body z axis up'; pick = [x]
    else:
        want = [int(w) for w in which]; by = {x['label']: x for x in cl}; assert all(w in by for w in want), 'unknown posture label'; pick = [by[w] for w in want]
    return dict(g=unit(np.array([x['rep_g'] for x in pick], float)), label=[int(x['label']) for x in pick], coarse=[int(x['coarse']) for x in pick], com_height_mm=[float(x['com_height_mm']) for x in pick],
                persistence_mm=[float(x['persistence_mm']) for x in pick], library_sha256=hashlib.sha256(raw).hexdigest(), which=(which if isinstance(which, str) else [int(w) for w in which]))


def height_from_balls(body, G, chunk=64):
    """h(g) computed from the body's inscribed balls instead of its primitives (the clouds lie up to the clearance inside the
    primitives, so the two agree to a fraction of a millimetre): max_k (c_k . g + r_k) - com . g for directions G (m, 3)"""
    c, r = np.asarray(body.p - body.r[:, None] * body.n), np.asarray(body.r); out = np.empty(len(G))
    for a in range(0, len(G), chunk): out[a:a + chunk] = (G[a:a + chunk] @ c.T + r[None, :]).max(1)
    return out - G @ body.com
