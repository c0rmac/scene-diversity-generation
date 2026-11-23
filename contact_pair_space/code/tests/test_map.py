"""the map state -> scene: it reproduces the research code's scenes, and its scenes have the properties the page states"""
import unittest
import numpy as np
from _common import *


@needs_native
class Map(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ch = chart(); cls.z = fixture('map_free'); z = cls.z; cls.R, cls.t, cls.info = cls.ch.realize(z['yA'], z['yB'], z['theta'], z['u'])

    def test_reproduces_research_scenes(self):
        z = self.z; self.assertTrue(np.array_equal(self.R, z['R'])); self.assertLess(np.abs(self.t - z['t']).max(), 1e-12)
        self.assertLess(np.abs(self.info['backoff'] - z['backoff']).max(), 1e-12); self.assertTrue(np.array_equal(self.info['jA'], z['jA'])); self.assertTrue(np.array_equal(self.info['jB'], z['jB']))

    def test_normals_opposed(self):
        i = self.info; Rn = np.einsum('nij,nj->ni', i['Rrel'], i['nB'][:, 0]); self.assertLess(np.abs(Rn + i['nA'][:, 0]).max(), 1e-9)

    def test_designated_contact_realised(self):
        self.assertTrue((self.info['pair_dist'][:, 0] <= self.ch.adm).all())

    def test_pair_on_the_table_at_the_clearance(self):
        ch, R, t = self.ch, self.R, self.t; low = np.minimum(ch._lowest(0, R[:, 0]) + t[:, 0, 2], ch._lowest(1, R[:, 1]) + t[:, 1, 2]); self.assertLess(np.abs(low - ch.clear).max(), 1e-12)

    def test_mug_centre_of_mass_over_the_origin(self):
        cw = np.einsum('nij,j->ni', self.R[:, 0], self.ch.com[0]) + self.t[:, 0]; self.assertLess(np.abs(cw[:, :2]).max(), 1e-12)

    def test_no_overlap_and_touching_by_brute_force(self):
        for k in range(10):
            g = min_gap(self.ch, self.info['Rrel'][k], self.info['trel'][k]); self.assertGreaterEqual(g, self.ch.clear - 1e-9); self.assertLessEqual(g, self.ch.clear + 1e-9)


if __name__ == '__main__':
    unittest.main()
