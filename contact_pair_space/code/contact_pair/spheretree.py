"""Binary bounding-sphere hierarchy of a ball set, used by the exact sweep to skip pairs of balls that cannot overlap. A node sphere
contains all of its balls; leaves hold at most LEAF balls. The tree changes only the speed of the sweep, never its result."""
import numpy as np


class SphereTree:
    def __init__(self, c, r, leaf=8):
        self.c = np.asarray(c, float); self.r = np.asarray(r, float); perm = np.arange(len(self.c)); nodes = []

        def make(lo, hi):
            idx = perm[lo:hi]; pc = self.c[idx]; mid = 0.5 * (pc.min(0) + pc.max(0)); rr = float(np.max(np.linalg.norm(pc - mid, axis=1) + self.r[idx]))
            nodes.append([mid, rr, -1, -1, lo, hi - lo]); return len(nodes) - 1

        todo = [make(0, len(perm))]
        while todo:
            k = todo.pop(); _, _, _, _, lo, n = nodes[k]
            if n <= leaf: continue
            idx = perm[lo:lo + n]; pc = self.c[idx]; ax = int(np.argmax(pc.max(0) - pc.min(0))); o = np.argsort(pc[:, ax], kind='stable'); perm[lo:lo + n] = idx[o]
            h = n // 2; a = make(lo, lo + h); b = make(lo + h, lo + n); nodes[k][2] = a; nodes[k][3] = b; todo += [a, b]
        self.perm = perm; self.cen = np.array([x[0] for x in nodes]); self.rad = np.array([x[1] for x in nodes])
        self.left = np.array([x[2] for x in nodes]); self.right = np.array([x[3] for x in nodes]); self.lo = np.array([x[4] for x in nodes]); self.cnt = np.array([x[5] for x in nodes])
