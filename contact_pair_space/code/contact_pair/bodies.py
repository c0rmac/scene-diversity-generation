"""The two bodies as the search space sees them: a cloud of surface samples with outward unit normals, a ball inscribed at each sample,
further balls filling the interior, the centre of mass, and the balls on the convex hull used for lowest points. Body 0 is the mug
(body A), body 1 the spoon (body B). Lengths in metres, in each body's own frame."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np

DATA = Path(__file__).resolve().parent.parent / 'data'
NAMES = ('mug', 'spoon')


@dataclass
class Body:
    name: str
    p: np.ndarray        # (n, 3) surface samples
    n: np.ndarray        # (n, 3) outward unit normals at the samples
    r: np.ndarray        # (n,) radius of the ball inscribed at each sample (at most rmax)
    all_c: np.ndarray    # (N, 3) centres of all balls: the inscribed balls and the interior balls
    all_r: np.ndarray    # (N,) their radii; the union of these balls is the solid body used by every overlap test
    cand_c: np.ndarray   # (K, 3) balls that can be lowest in some orientation (on the convex hull of the union)
    cand_r: np.ndarray   # (K,)
    com: np.ndarray      # (3,) centre of mass
    rmax: float          # largest inscribed radius (the skin radius, 1.5 mm)
    rms: float           # root-mean-square distance of the samples from the centre of mass

    def lowest(self, R, chunk=1000):
        """height of the body's lowest point above its origin, for rotations R (m, 3, 3): min_k (e_z . R c_k - r_k)"""
        lo = np.empty(len(R))
        for a in range(0, len(R), chunk):
            lo[a:a + chunk] = np.min(R[a:a + chunk, 2, :] @ self.cand_c.T - self.cand_r[None, :], axis=1)
        return lo


def load_bodies(data_dir=None):
    """the mug and the spoon from data/bodies.npz"""
    z = np.load(Path(data_dir or DATA) / 'bodies.npz')
    return [Body(name=nm, p=z[f'{nm}_p'], n=z[f'{nm}_n'], r=z[f'{nm}_r'], all_c=z[f'{nm}_all_c'], all_r=z[f'{nm}_all_r'], cand_c=z[f'{nm}_cand_c'], cand_r=z[f'{nm}_cand_r'],
                 com=z[f'{nm}_com'], rmax=float(z[f'{nm}_rmax']), rms=float(z[f'{nm}_rms'])) for nm in NAMES]
