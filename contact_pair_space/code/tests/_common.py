"""shared set-up of the tests: the package on the path, one chart for all tests, fixtures, and the switch for the slow tests
(set CONTACT_PAIR_SLOW=1 to run them)."""
import os, sys, unittest
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from contact_pair import PairChart, native_available

FIX = ROOT / 'tests' / 'fixtures'; SLOW = os.environ.get('CONTACT_PAIR_SLOW') == '1'
needs_native = unittest.skipUnless(native_available(), 'the native sweep is not built: run ./build_native.sh')
slow = unittest.skipUnless(SLOW, 'slow test: set CONTACT_PAIR_SLOW=1')
_CHART = None


def chart():
    global _CHART
    if _CHART is None: _CHART = PairChart()
    return _CHART


def fixture(name):
    return np.load(FIX / (name + '.npz'))


def min_gap(ch, Rrel, trel, chunk=20000):
    """independent of the sweep: the smallest distance minus radii between the two ball unions (spoon placed by Rrel, trel in the
    mug's frame), over all pairs of balls within reach; also the pair that attains it"""
    from scipy.spatial import cKDTree
    A, B = ch.bodies; tree = cKDTree(A.all_c); cB = B.all_c @ Rrel.T + trel; best = np.inf
    for a in range(0, len(cB), chunk):
        d, j = tree.query(cB[a:a + chunk], k=8, distance_upper_bound=0.01)
        ok = np.isfinite(d)
        if ok.any():
            jj = np.where(ok, j, 0); g = np.where(ok, d - B.all_r[a:a + chunk, None] - A.all_r[jj], np.inf); best = min(best, float(g.min()))
    return best
