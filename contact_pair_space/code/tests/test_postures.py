"""resting postures: the library's heights are the support-function heights of the primitives, each posture is a strict local
minimum, and the clouds give the same heights to a fraction of a millimetre"""
import unittest
import numpy as np
from _common import *
from contact_pair.primitives import load_primitives, Height
from contact_pair.postures import resting_postures, height_from_balls


class Postures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prim = load_primitives(); cls.lib = resting_postures(0, 'all')

    def test_library_heights(self):
        H = Height(self.prim['mug']['geoms'], self.prim['mug']['com']); self.assertLess(np.abs(1e3 * H(self.lib['g']) - np.array(self.lib['com_height_mm'])).max(), 1e-6)

    def test_strict_local_minima(self):
        H = Height(self.prim['mug']['geoms'], self.prim['mug']['com'])
        for g in self.lib['g']:
            a = np.array([1.0, 0, 0]) if abs(g[0]) < 0.9 else np.array([0, 1.0, 0]); e1 = np.cross(g, a); e1 /= np.linalg.norm(e1); e2 = np.cross(g, e1); th = np.linspace(0, 2 * np.pi, 72, endpoint=False)
            ring = g + 0.01 * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2); ring /= np.linalg.norm(ring, axis=1, keepdims=True); self.assertTrue((H(ring) > H(g)[0]).all())

    def test_clouds_agree(self):
        h = 1e3 * height_from_balls(chart().bodies[0], self.lib['g']); self.assertLess(np.abs(h - np.array(self.lib['com_height_mm'])).max(), 0.05)

    def test_four_classes_and_upright(self):
        c = resting_postures(0, 'classes'); self.assertEqual(len(c['label']), 4); self.assertEqual(resting_postures(0, 'upright')['label'], [24])


if __name__ == '__main__':
    unittest.main()
