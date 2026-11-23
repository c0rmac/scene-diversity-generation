"""regions: the pockets, the inside of the cup, and that a restricted chart decodes every state to a place of its region"""
import unittest
import numpy as np
from _common import *


class Regions(unittest.TestCase):
    def test_cup_mask_equals_research(self):
        m = chart().pocket_mask(0, 1); self.assertTrue(np.array_equal(m, fixture('pocket1_mask_mug')['mask'])); self.assertAlmostEqual(float(m.mean()), 0.4048, places=3)

    def test_restricted_chart_decodes_into_the_region(self):
        ch = chart(); m = ch.pocket_mask(0, 1); c = ch.restricted(m); rng = np.random.default_rng(2); Y = rng.uniform(ch.box[0][0], ch.box[0][1], size=(5000, 3))
        j, _, _ = c.contact(0, Y); self.assertTrue(m[j].all())

    def test_layer_samples_stay_in_the_region(self):
        ch = chart(); m = ch.predicate_mask(0, lambda p, n: n[:, 2] > 0.9); c = ch.restricted(m); Y = c.sample_shell(0, 3000, np.random.default_rng(4))
        j, _, _ = ch.contact(0, Y); self.assertTrue(m[j].all())      # every point of the restricted layer is nearest to a sample of the region


if __name__ == '__main__':
    unittest.main()
