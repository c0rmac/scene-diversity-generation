"""Quick start: sample scenes of the contact-pair search space, map a state by hand, and save the scenes.
usage: python examples/quickstart.py [N]     (default N = 500; 75 s on an 18-core laptop)"""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from contact_pair import PairChart

N = int(sys.argv[1]) if len(sys.argv) > 1 else 500
tic = time.time(); chart = PairChart()        # loads data/bodies.npz and builds the search trees (about 1 s)
print('chart ready in %.1f s: clearance %.2f mm, skin radius %.1f mm' % (time.time() - tic, 1e3 * chart.clear, 1e3 * chart.skin))

# 1. The starting law: every place of the mug equally likely to be the contact; the spoon's place and the turn uniform among those
#    that fit there; gravity uniform. Each state maps to a scene in which the bodies touch at the chosen points.
tic = time.time(); st = chart.sample_start(N, seed=1)
R, t, info = chart.realize(st['yA'], st['yB'], st['theta'], st['u'])
print('%d states in %.1f s (%.0f candidates per state on average); designated contact realised in all: %s'
      % (N, time.time() - tic, st['tries'].mean(), bool((info['pair_dist'][:, 0] <= chart.adm).all())))

# 2. A region: the mug's contact place inside the cup only (pocket 1 of the mug).
cup = chart.restricted(chart.pocket_mask(0, 1))
tic = time.time(); st_cup = cup.sample_start(N, seed=1); R_cup, t_cup, _ = cup.realize(st_cup['yA'], st_cup['yB'], st_cup['theta'], st_cup['u'])
print('inside the cup: %d states in %.1f s' % (N, time.time() - tic))

# 3. The mug kept upright on the table, contact inside the cup.
up = cup.stable_mug('upright')
tic = time.time(); st_up = up.sample_start(N, seed=1); R_up, t_up, info_up = up.realize(st_up['yA'], st_up['yB'], st_up['theta'], st_up['u'])
print('upright mug, inside the cup: %d states in %.1f s; the mug has one pose: %s' % (N, time.time() - tic, bool(np.allclose(R_up[:, 0], R_up[0, 0]))))

# 4. One state by hand: a point just above the rim of the mug, a point just outside the end of the spoon's bowl, the spoon turned by
#    90 degrees about the common normal, gravity along the mug's -z (upright). States are arrays: yA, yB (n, 1, 3), theta (n,), u (n, 3).
yA = np.array([[[0.045, 0.0, 0.1012]]]); yB = np.array([[[0.0, 0.0, -0.0235]]]); theta = np.array([np.pi / 2]); u = np.array([[0.0, 0.0, -1.0]])
R1, t1, i1 = chart.realize(yA, yB, theta, u)
print('hand-made state: contact on the mug at %s mm, push-off %.2f mm (%s)' % (np.round(1e3 * i1['pA'][0, 0], 1).tolist(), 1e3 * i1['backoff'][0], 'realised' if i1['backoff'][0] <= chart.adm else 'blocked: the spoon touches elsewhere'))

# A scene is a rotation and a translation per body: the body's point x is at R[i, b] @ x + t[i, b] in the world (b = 0 mug, 1 spoon).
mug, spoon = chart.bodies; world_spoon = spoon.p @ R[0, 1].T + t[0, 1]
print('scene 0: the spoon\'s lowest sample is %.2f mm above the table' % (1e3 * world_spoon[:, 2].min()))

np.savez('scenes.npz', R=R, t=t, R_cup=R_cup, t_cup=t_cup, R_upright=R_up, t_upright=t_up)
print('saved scenes.npz')
