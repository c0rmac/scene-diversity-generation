"""the stable and upright mug: the research scenes are reproduced, the mug stands in a resting posture, the spoon is above the table"""
import unittest
import numpy as np
from _common import *


@needs_native
class StableMug(unittest.TestCase):
    def check(self, which, name):
        c = chart().stable_mug(which); z = fixture(name); R, t, info = c.realize(z['yA'], z['yB'], z['theta'], z['u'])
        self.assertTrue(np.array_equal(R, z['R'])); self.assertLess(np.abs(t - z['t']).max(), 1e-12); self.assertTrue(np.array_equal(info['posture'], z['posture']))
        self.assertTrue(np.array_equal(R[:, 0], c.stable['S'][info['posture']]))                                       # the mug is in an allowed posture
        self.assertLess(np.abs(c._lowest(0, R[:, 0]) + t[:, 0, 2] - c.clear).max(), 1e-12)                            # standing on the table by itself
        self.assertTrue((c._lowest(1, R[:, 1]) + t[:, 1, 2] >= c.clear - 1e-9).all())                                 # the spoon above the table
        self.assertTrue(info['admissible'].all())
        return c

    def test_any_stable_posture(self):
        c = self.check('classes', 'map_stable'); self.assertEqual(len(c.stable['label']), 4)

    def test_upright(self):
        c = self.check('upright', 'map_upright'); self.assertEqual(c.stable['label'], [24]); self.assertLess(c.stable['g'][0, 2], -0.99)


if __name__ == '__main__':
    unittest.main()
