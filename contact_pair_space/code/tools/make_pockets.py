"""Recomputes data/pockets.npz from the bodies' primitives (data/primitives.json): for each body, conv(B) minus B on a 1 mm voxel
grid, its face-connected parts numbered by decreasing volume, and a fixed 1 mm surface sampling. Deterministic (no random numbers).
usage: python tools/make_pockets.py [OUT.npz]   (default: writes pockets_recomputed.npz in the current folder)"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from contact_pair.primitives import load_primitives
from contact_pair.pockets import compute_pocket_file

if __name__ == '__main__':
    out = Path(sys.argv[1] if len(sys.argv) > 1 else 'pockets_recomputed.npz'); arr = compute_pocket_file(load_primitives()); np.savez(out, **arr)
    for nm in ('mug', 'spoon'):
        lab = arr[f'{nm}_label']; vol = np.bincount(lab.ravel())[1:]; print('%s: grid %s, %d pockets, largest %s cm3' % (nm, lab.shape, len(vol), np.round(vol[:3] * 1e-3, 3).tolist()))
    print('written', out)
