"""the rotations of the map: the smallest rotation taking a onto b, the gravity turn S(u), turns about an axis"""
import unittest
import numpy as np
from _common import *
from contact_pair.rotations import minrot, section, rot_axis, unit


def err(R, a, b):
    """how far R a is from b, and how far R is from a rotation (largest entry of R^T R - I, largest |det R - 1|)"""
    return (np.abs(np.einsum('nij,nj->ni', R, a) - b).max(1), np.abs(np.einsum('nji,njk->nik', R, R) - np.eye(3)).max((1, 2)), np.abs(np.linalg.det(R) - 1))


def near_opposite(a, ang, rng):
    """unit vectors at angle ANG (n,) from -a"""
    p = unit(np.cross(a, rng.standard_normal(a.shape))); return unit(-a * np.cos(ang)[:, None] + p * np.sin(ang)[:, None])


class Rotations(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(3); self.a = unit(self.rng.standard_normal((2000, 3))); self.b = unit(self.rng.standard_normal((2000, 3)))

    def test_minrot_generic(self):
        for e in err(minrot(self.a, self.b), self.a, self.b): self.assertLess(e.max(), 1e-12)

    def test_minrot_opposite(self):
        """a = -b: a half turn about an axis orthogonal to a"""
        for e in err(minrot(self.a, -self.a), self.a, -self.a): self.assertLess(e.max(), 1e-14)

    def test_minrot_nearly_opposite(self):
        """within 1e-6 rad of opposite the half turn is used, so R a misses b by at most the angle. Further out the formula
        I + [v]x + [v]x^2 / (1 + c) divides by 1 + c = 1 + a.b: its error grows like the rounding error over 1 + c (largest near
        1.5e-6 rad, about 1e-3). This is the research code's arithmetic, kept so that scenes are reproduced bit for bit; in its
        40,000 stored scenes the worst rotation is orthogonal to 9.4e-9."""
        ang = np.exp(self.rng.uniform(np.log(1e-12), np.log(1e-6), len(self.a))); b = near_opposite(self.a, ang, self.rng)
        e = err(minrot(self.a, b), self.a, b); self.assertTrue((e[0] <= ang + 1e-15).all()); self.assertLess(max(e[1].max(), e[2].max()), 1e-14)
        ang = np.exp(self.rng.uniform(np.log(1.5e-6), np.log(1.0), len(self.a))); b = near_opposite(self.a, ang, self.rng); c1 = 1.0 + (self.a * b).sum(1)
        for e in err(minrot(self.a, b), self.a, b): self.assertTrue((e <= 1e-14 / c1).all())

    def test_section_turns_gravity_down(self):
        S = section(self.a); self.assertLess(np.abs(np.einsum('nij,nj->ni', S, self.a) - [0, 0, -1]).max(), 1e-12)

    def test_turn_keeps_its_axis(self):
        th = np.linspace(0, 2 * np.pi, len(self.a)); Q = rot_axis(self.a, th); self.assertLess(np.abs(np.einsum('nij,nj->ni', Q, self.a) - self.a).max(), 1e-12)


if __name__ == '__main__':
    unittest.main()
