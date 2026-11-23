"""the starting law: sample_start(300, seed 11) reproduces the research code's states exactly, and every state it returns is
admissible. The free mug runs by default (about 20 s); inside the cup, stable and upright are slow (CONTACT_PAIR_SLOW=1)."""
import unittest
import numpy as np
from _common import *


@needs_native
class StartingLaw(unittest.TestCase):
    def check(self, c, name):
        z = fixture(name); st = c.sample_start(300, 11)
        for k in ('yA', 'yB', 'theta', 'u', 'tries'): self.assertTrue(np.array_equal(st[k], z[k]), k)
        self.assertEqual(st['acceptance'], float(z['acceptance']))
        R, t, info = c.realize(st['yA'], st['yB'], st['theta'], st['u']); self.assertTrue(np.array_equal(R, z['R'])); self.assertLess(np.abs(t - z['t']).max(), 1e-12)
        self.assertTrue((info['backoff'] <= c.adm).all()); return st, info

    def test_free_mug(self):
        self.check(chart(), 'law_free')

    @slow
    def test_inside_the_cup(self):
        c = chart().restricted(chart().pocket_mask(0, 1)); st, info = self.check(c, 'law_cup'); self.assertTrue(c.region[0][info['jA']].all())

    @slow
    def test_stable_mug(self):
        self.check(chart().stable_mug('classes'), 'law_stable')

    @slow
    def test_upright_mug(self):
        self.check(chart().stable_mug('upright'), 'law_upright')


if __name__ == '__main__':
    unittest.main()
