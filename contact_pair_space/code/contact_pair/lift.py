"""The exact sweep: the smallest t >= 0 such that a moving ball union, raised by t along the world vertical, overlaps neither another
ball union nor (optionally) the table z >= 0. The chart uses it for the push-off along the mug's normal (Section 4.3 of the page) by
turning that normal to the vertical first, and for raising the spoon above the table (stable mug).

Every pair of balls that overlaps forbids an open interval of t; the iteration t <- max{upper end over the pairs that overlap at t}
starts at t = 0 and stops when nothing overlaps, so it returns the smallest t >= 0 outside the union of the intervals: an exact
answer, not a convergence to a tolerance. Status: 0 nothing overlapped at t = 0; 1 raised; 2 iteration limit (t a lower bound).

Two implementations with the same arithmetic: the native one (native/vertical_lift.cpp, branch and bound over sphere trees, threads;
build it with build_native.sh) and lift_reference below (NumPy, one state at a time, for checking and for small jobs)."""
import ctypes, os, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from .spheretree import SphereTree

_HERE = Path(__file__).resolve().parent / '_native'
LIB = _HERE / ('libvertical_lift.dylib' if sys.platform == 'darwin' else 'libvertical_lift.so')
_D = ctypes.POINTER(ctypes.c_double); _I = ctypes.POINTER(ctypes.c_int32); _L = ctypes.POINTER(ctypes.c_int64)
f64 = lambda a: np.ascontiguousarray(a, dtype=np.float64)
i32 = lambda a: np.ascontiguousarray(a, dtype=np.int32)


def native_available():
    return LIB.exists()


def _tree(T, body, inflate):
    """sphere-tree arrays relative to the body's centre of mass, every radius increased by INFLATE"""
    com = np.asarray(body.com, float)
    return dict(cen=f64(T.cen - com), rad=f64(T.rad + inflate), left=i32(T.left), right=i32(T.right), lo=i32(T.lo), cnt=i32(T.cnt),
                bc=f64(body.all_c[T.perm] - com), br=f64(body.all_r[T.perm] + inflate))


class Lifter:
    """moving body and other body (Body objects); poses are (R, c) with c the world position of the body's centre of mass.
    inflate: every ball radius is increased by this amount (the chart uses half the clearance, so the unions end the clearance apart)."""
    def __init__(self, moving, other, inflate=0.0, leaf=8):
        self.inflate = float(inflate); self.A = _tree(SphereTree(moving.all_c, moving.all_r, leaf), moving, self.inflate); self.B = _tree(SphereTree(other.all_c, other.all_r, leaf), other, self.inflate)
        self._f = None
        if native_available():
            lib = ctypes.CDLL(str(LIB)); f = lib.vl_lift; f.restype = ctypes.c_int
            f.argtypes = ([_D, _D, _I, _I, _I, _I, _D, _D] * 2 + [ctypes.c_int, ctypes.c_int, ctypes.c_int, _D, _D, _D, _D, ctypes.c_double, ctypes.c_int, ctypes.c_int] + [_D, _I, _I, _L])
            self._lib, self._f = lib, f

    def lift(self, Ra, ca, Rb, cb, use_other=True, use_table=True, tol=1e-9, max_iter=200, threads=None):
        """n poses at once: Ra, Rb (n, 3, 3), ca, cb (n, 3). Ra and Rb must be rotations: the sphere-tree bounds hold only under
        rotations, and another matrix gives wrong answers. Returns dict(t, status, iters, visited)."""
        if self._f is None:
            raise RuntimeError('the native sweep is not built: run ./build_native.sh (or use lift_reference for a few states)')
        Ra = f64(Ra); ca = f64(ca); Rb = f64(Rb); cb = f64(cb); n = len(ca); threads = int(threads or os.cpu_count() or 1)
        out = dict(t=np.empty(n), status=np.empty(n, np.int32), iters=np.empty(n, np.int32), visited=np.empty(n, np.int64)); p = lambda a, t: a.ctypes.data_as(t); args = []
        for T in (self.A, self.B):
            args += [p(T['cen'], _D), p(T['rad'], _D), p(T['left'], _I), p(T['right'], _I), p(T['lo'], _I), p(T['cnt'], _I), p(T['bc'], _D), p(T['br'], _D)]
        self._f(*args, int(use_other), int(use_table), n, p(Ra, _D), p(ca, _D), p(Rb, _D), p(cb, _D), float(tol), int(max_iter), threads, p(out['t'], _D), p(out['status'], _I), p(out['iters'], _I), p(out['visited'], _L))
        return out

    def lift_reference(self, Ra, ca, Rb, cb, use_other=True, use_table=True, tol=1e-9, max_iter=200):
        """the same answer with NumPy and a k-d tree, one pose at a time (no sphere tree): an independent check of the native code"""
        n = len(ca); out = dict(t=np.zeros(n), status=np.zeros(n, np.int32), iters=np.zeros(n, np.int32))
        A, B = self.A, self.B
        for k in range(n):
            P = A['bc'] @ np.asarray(Ra[k]).T + ca[k]; Q = B['bc'] @ np.asarray(Rb[k]).T + cb[k]; rA, rB = A['br'], B['br']
            I, J = np.zeros(0, np.int64), np.zeros(0, np.int64)
            if use_other:
                # Pairs within reach in the horizontal plane, built in blocks of moving balls to bound memory. Two kinds of pair are
                # dropped because they never change the answer: horizontal distance at least the sum of radii (they never overlap),
                # and upper end at most 0 (they could only forbid t < 0).
                tree = cKDTree(Q[:, :2]); rmax = rB.max(); Is, Js = [], []
                for a in range(0, len(P), 2048):
                    b = min(a + 2048, len(P)); nb = tree.query_ball_point(P[a:b, :2], r=rA[a:b] + rmax); cnt = np.fromiter(map(len, nb), np.int64, len(nb))
                    i = a + np.repeat(np.arange(b - a), cnt); j = np.fromiter((j for x in nb for j in x), np.int64, int(cnt.sum()))
                    h2 = (P[i, 0] - Q[j, 0]) ** 2 + (P[i, 1] - Q[j, 1]) ** 2; s = rA[i] + rB[j]
                    keep = (h2 < s * s) & ((Q[j, 2] - P[i, 2]) + np.sqrt(np.maximum(0.0, s * s - h2)) > 0.0); Is.append(i[keep]); Js.append(j[keep])
                I, J = np.concatenate(Is), np.concatenate(Js)
            ex, ey = P[I, 0] - Q[J, 0], P[I, 1] - Q[J, 1]; h2 = ex * ex + ey * ey; s = rA[I] + rB[J]; up = (Q[J, 2] - P[I, 2]) + np.sqrt(np.maximum(0.0, s * s - h2))
            t, it, status = 0.0, 0, 0
            while True:
                best = -np.inf
                if use_table:
                    m = P[:, 2] + t - rA < -tol
                    if m.any(): best = max(best, float((rA[m] - P[m, 2]).max()))
                if len(I):
                    ez = P[I, 2] + t - Q[J, 2]; m = np.sqrt(h2 + ez * ez) - s < -tol
                    if m.any(): best = max(best, float(up[m].max()))
                if not (best > t + 1e-12): break
                t = best; status = 1; it += 1
                if it >= max_iter: status = 2; break
            out['t'][k], out['status'][k], out['iters'][k] = t, status, it
        return out
