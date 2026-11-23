"""the tools that compute the data from the primitives reproduce the stored data (slow: CONTACT_PAIR_SLOW=1)"""
import json, sys, unittest
import numpy as np
from _common import *


class Tools(unittest.TestCase):
    @slow
    def test_pockets_reproduced(self):
        from contact_pair.primitives import load_primitives
        from contact_pair.pockets import compute_pocket_file
        new = compute_pocket_file(load_primitives()); old = np.load(ROOT / 'data' / 'pockets.npz')
        for k in old.files: self.assertTrue(np.array_equal(np.asarray(new[k]), old[k]), k)

    @slow
    def test_postures_reproduced(self):
        sys.path.insert(0, str(ROOT / 'tools')); import make_postures
        new = make_postures.library(M=400000, k=10, bodies=('mug',)); old = json.loads((ROOT / 'data' / 'postures.json').read_text())
        a, b = new['mug']['classes'], old['mug']['classes']; self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            self.assertEqual((x['label'], x['coarse']), (y['label'], y['coarse'])); self.assertAlmostEqual(x['com_height_mm'], y['com_height_mm'], places=9); self.assertLess(np.abs(np.subtract(x['rep_g'], y['rep_g'])).max(), 1e-9)


if __name__ == '__main__':
    unittest.main()
