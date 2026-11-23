"""the exact sweep: the native library against the NumPy reference, the table, and what the designated pair alone forbids"""
import unittest
import numpy as np
from _common import *
from contact_pair.rotations import minrot, section, unit, EZ


@needs_native
class Sweep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ch = chart(); z = fixture('map_free'); ch = cls.ch; k = np.arange(12)
        _, pA, nA = ch.contact(0, z['yA'][k]); _, pB, nB = ch.contact(1, z['yB'][k]); th = z['theta'][k] + np.where(k % 2, 2.4, 0.0)      # half at the sampled turn, half turned
        cls.R, cls.t = ch.register(pA, nA, pB, nB, th); cls.d = nA[:, 0]

    def test_native_equals_reference(self):
        ch, R, t, d = self.ch, self.R, self.t, self.d; Q = minrot(d, np.broadcast_to(EZ, d.shape).copy()); cB = np.einsum('nij,j->ni', R, ch.com[1]) + t
        args = (Q @ R, np.einsum('nij,nj->ni', Q, cB), Q, np.einsum('nij,j->ni', Q, ch.com[0]))
        a = ch.lifter.lift(*args, use_other=True, use_table=False); b = ch.lifter.lift_reference(*args, use_other=True, use_table=False)
        self.assertLess(np.abs(a['t'] - b['t']).max(), 1e-12); self.assertTrue(np.array_equal(a['status'], b['status'])); self.assertTrue((a['t'] > 5e-3).any(), 'some placements should be blocked')

    def test_table_only(self):
        ch = self.ch; B = ch.bodies[1]; rng = np.random.default_rng(1); R = section(unit(rng.standard_normal((20, 3)))); c = np.zeros((20, 3)); c[:, 2] = 0.01
        L = ch.lifter.lift(R, c, np.repeat(np.eye(3)[None], 20, 0), np.zeros((20, 3)), use_other=False, use_table=True)
        low = np.min(((B.all_c - B.com) @ R.transpose(0, 2, 1))[..., 2] - (B.all_r + 0.5 * ch.clear)[None, :], axis=1) + 0.01      # lowest point of the inflated union
        self.assertLess(np.abs(L['t'] - np.maximum(0, -low)).max(), 1e-12)

    def test_realised_pushoff_is_the_clearance(self):
        s, _ = self.ch.backoff(self.R[::2], self.t[::2], self.d[::2]); self.assertTrue((s >= self.ch.clear - 1e-12).all()); self.assertTrue((s <= self.ch.adm).all())


if __name__ == '__main__':
    unittest.main()
