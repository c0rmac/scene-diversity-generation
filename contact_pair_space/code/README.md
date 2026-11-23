# Contact-pair search space: code

This folder holds everything needed to build the search space of touching scenes described on the page
[A search space of touching scenes: contact pairs of a mug and a spoon](https://c0rmac.github.io/scene-diversity-generation/contact_pair_space/).

In this space, every state is a scene of a mug and a spoon on a table in which the two bodies touch and do not overlap. A state is:
- a point near each body's surface, which picks the place of contact;
- one angle, the turn of the spoon about the common normal;
- the direction of gravity.

The map from a state to a scene is an explicit computation on the two bodies' point clouds; nothing is solved iteratively. The
folder also contains the sampling law that makes every place of the mug equally likely to be the contact, the regions (for
example the inside of the cup), and the options that keep the mug resting on the table or upright.

The code reproduces the research code that made the page bit for bit: rotations and push-offs are identical, and positions agree
to 1e-16 m. The tests check this against stored research outputs.

## Quick start

Requirements: Python 3.10 or later with NumPy and SciPy, and a C++17 compiler (`c++`, `clang++` or `g++`).

```bash
pip install -r requirements.txt
```

```bash
./build_native.sh
```

```bash
python examples/quickstart.py
```

```bash
python -m unittest discover -s tests -v
```

- `build_native.sh` compiles the exact sweep in about a second.
- `quickstart.py` samples scenes and saves them to `scenes.npz`: 75 s on an 18-core laptop.
- The test command runs the default tests: 37 s on the same laptop.

In Python:

```python
from contact_pair import PairChart

chart = PairChart()                                    # the mug and the spoon from data/; about 1 s
st = chart.sample_start(1000, seed=1)                  # the starting law: 1000 states
R, t, info = chart.realize(st['yA'], st['yB'], st['theta'], st['u'])
# R[i, b] (3x3) and t[i, b] (3,) place body b (0 mug, 1 spoon) of scene i: a body point x is at R[i, b] @ x + t[i, b].

cup = chart.restricted(chart.pocket_mask(0, 1))        # contact places inside the cup only
upright = cup.stable_mug('upright')                    # ... with the mug standing upright on the table
resting = chart.stable_mug()                           # the mug in any of its four resting postures
```

Units are metres. The table is the plane z = 0, and z points up.

## What the files are for

| File | What it is for |
|---|---|
| `contact_pair/chart.py` | **The search space.** `PairChart`: the state → scene map (`realize`), its steps (`contact`, `register`, `backoff`), the starting law (`sample_start`), regions (`restricted`, `pocket_mask`, `predicate_mask`) and the stable or upright mug (`stable_mug`). Read its docstring first. |
| `contact_pair/lift.py` | The exact sweep: the smallest distance that removes every overlap between two ball unions (and the table). Used for the push-off along the mug's normal and for lifting the spoon above the table. Wraps the native library, and has a NumPy reference `lift_reference` used to check it. |
| `native/vertical_lift.cpp` | Source of the native sweep: branch and bound over sphere trees, multithreaded. `build_native.sh` compiles it into `contact_pair/_native/`. |
| `contact_pair/spheretree.py` | Bounding-sphere trees of the ball sets, which the native sweep traverses. They change only the speed, never the result. |
| `contact_pair/bodies.py` | `Body` and `load_bodies`: each body's surface samples, normals and balls from `data/bodies.npz`, and its lowest point in any orientation. |
| `contact_pair/rotations.py` | Small rotation helpers: the smallest rotation taking one vector onto another, the gravity turn S(u), and turns about an axis. |
| `contact_pair/pockets.py` | Pockets (the space inside a body's convex hull but outside the body; for the mug, the inside of the cup and the space by the handle). `PocketMap` looks them up; `compute_pockets` computes them from the primitives. |
| `contact_pair/postures.py` | Resting postures of the mug alone on the table, from the library in `data/postures.json`; the centre-of-mass height h(g) from the clouds. |
| `contact_pair/primitives.py` | The bodies' collision primitives: the exact support-function height h(g), and inside tests and surface samplings for the pocket tool. |
| `tools/make_pockets.py` | Recomputes `data/pockets.npz` from the primitives. |
| `tools/make_postures.py` | Recomputes `data/postures.json` (the resting-posture library) from the primitives. |
| `examples/quickstart.py` | A short, commented tour: sampling, a region, the upright mug, one state by hand, saving scenes. |
| `tests/` | The tests (see below), with `tests/fixtures/`: stored outputs of the research code that the tests must reproduce. |
| `data/bodies.npz` | The two bodies as point clouds: samples, normals, inscribed and interior balls, centres of mass. This is the input the construction starts from. |
| `data/primitives.json` | The bodies' collision primitives and centres of mass (boxes, capsules, a cylinder, an ellipsoid). |
| `data/pockets.npz`, `data/postures.json` | Pocket grids and resting postures, computed from the primitives by the two tools. |
| `data/PROVENANCE.json` | sha256 of every data file and of the research source files they come from. |
| `build_native.sh`, `requirements.txt` | Build script for the native sweep, and the Python dependencies. |

## The search space in brief

**State.** One designated pair: `yA` and `yB` are points of space in the frames of the mug and of the spoon (arrays `(n, 1, 3)`);
each singles out the nearest surface sample. `theta` (`(n,)`) is the turn about the common normal, and `u` (`(n, 3)`) the direction
of gravity as seen from the mug.

**Map** (`realize`; Section 4 of the page).
1. The contact point and normal on each body. The normal is blended from the 48 nearest samples with weights exp(−(d_k − d_1)/δ),
   δ = 0.5 mm, so that round a sharp edge it turns smoothly from one face's normal to the other's.
2. The spoon's pose relative to the mug: R = Q(ν_A, θ) R₀(ν_B, −ν_A) and t = p − R q. The normals are opposed and the chosen
   points coincide.
3. The push-off. The spoon moves along ν_A by the smallest distance s* at which the ball unions are at least the clearance c apart.
   s* = c exactly when the designated contact is realised; otherwise the spoon ends up touching the mug somewhere else.
4. Gravity and the table. The mug is turned so that u points down, the pair is set on the table with its lowest point at height c,
   and the mug's centre of mass is placed over the origin.

**Clearance** c = (√2 − 1) × the 99 % quantile of the clouds' sample spacing, 0.37 mm here. It covers the corner tips that the
inscribed balls leave uncovered at sharp edges.

**Starting law** (`sample_start`; Section 5). The mug's place of contact is uniform over its surface (or region). Given that place,
the spoon's place and the turn are uniform among those that fit: candidates are drawn until one realises the contact, up to 4,096
tries, and a place with none is given up. Gravity is uniform on the sphere. The law is deterministic for a given seed.

**Regions** (Section 5.1). `chart.restricted(maskA, maskB)` restricts the places to subsets of the samples, and every state of the
restricted chart decodes to a place in the region. Masks come from:
- `pocket_mask(0, 1)`: the inside of the cup;
- `predicate_mask(b, fn)`: any rule on positions and normals;
- any boolean array, one value per sample.

**Stable and upright mug** (Sections 5.2 and 5.3). `chart.stable_mug(postures)` keeps the mug in one of its own resting postures,
standing on the table by itself:
- `'classes'`: upright, upside down, or on its side either way;
- `'upright'`: upright only;
- `'all'`: every resting posture, including the shallow facets;
- a list of library labels.

The gravity coordinate is decoded to the nearest allowed posture. A spoon that would reach below the table is lifted straight up.
The starting law then draws the posture together with the spoon's place and turn.

## Tests

```bash
python -m unittest discover -s tests -v
```

```bash
CONTACT_PAIR_SLOW=1 python -m unittest discover -s tests -v
```

The first command runs the default tests: 37 s, with peak memory 0.6 GB. The second also runs the slow ones: about two minutes,
peak 2.2 GB. Both were measured on an 18-core laptop.

| Test file | What it checks |
|---|---|
| `test_rotations.py` | R₀(a, b) takes a onto b and is a rotation: to 1e-12 for general vectors, exactly for opposite ones, and within the stated bound for nearly opposite ones (see Notes). S(u) turns u straight down; a turn keeps its axis. |
| `test_lift.py` | The native sweep equals the NumPy reference (to 1e-12 m) at realised and blocked placements. A table-only lift equals the exact lowest point. A realised push-off is never below the clearance. |
| `test_map.py` | 200 states reproduce the research scenes, push-offs and contact samples. The normals are opposed and the designated contact is realised. The pair stands at the clearance above the table with the mug's centre of mass over the origin. A brute-force check over all ball pairs finds the two unions exactly the clearance apart. |
| `test_stable.py` | The same for the stable and the upright mug. The mug is exactly in an allowed posture and stands on the table by itself, and the spoon is above the table. |
| `test_sampling.py` | `sample_start(300, seed 11)` reproduces the research code's states, numbers of tries and scenes exactly, and all states are admissible. Default: the free mug. Slow: inside the cup, stable mug, upright mug. |
| `test_regions.py` | The cup mask equals the research mask (40.5 % of the mug's samples). A restricted chart decodes every point to its region. |
| `test_postures.py` | The library heights equal the primitives' support-function heights, each posture is a strict local minimum, and the clouds give the same heights within 0.05 mm. |
| `test_tools.py` | Slow: `make_pockets` reproduces `data/pockets.npz` exactly, and `make_postures` reproduces the mug's resting postures. |

Tests that need the native sweep are skipped, with a message, until `./build_native.sh` has run.

## Where the data comes from

The bodies are given as point clouds, as on the page ("What is given"). `data/bodies.npz` holds them as the research code used
them:
- 131,072 surface samples of the mug and 32,768 of the spoon, uniform by area;
- each sample's outward normal and inscribed-ball radius (at most 1.5 mm);
- the interior balls;
- the centres of mass of the MuJoCo model.

They were generated from the bodies' collision primitives by a separate geometry pipeline of the research code, which is not part
of this folder.

The pocket grids and the resting-posture library are computed from the primitives in `data/primitives.json` by the two tools
here; the tests check that the tools reproduce them. `data/PROVENANCE.json` lists the sha256 of every file and of the research
sources it was taken from. The stored research outputs in `tests/fixtures/` come from the same research code.

## Notes

- Only two bodies and one designated contact pair are supported in practice. `k >= 2` designated pairs are implemented (the pose
  that fits all pairs best, by SVD), but two independently chosen pairs almost never fit together (page Section 7).
- The map is sensitive where the spoon is thin. Moving the contact places by 0.05 mm moves the scene by a median of 0.37 mm, and by
  more than 20 mm in 0.5 % of cases.
- A sampled scene touches but is not at rest. Releasing it in a physics simulator (not part of this folder) shows where it settles.
- `lift_reference` is exact but slow, one state at a time; use it to check the native sweep, not to sample.
- R₀(a, b) loses accuracy when a and b are nearly opposite. Within 1e-6 rad of opposite a half turn is used, which misses b by at
  most that angle. Between 1e-6 and about 1e-4 rad the Rodrigues formula divides by 1 + a·b, and its rounding error reaches about 1e-3
  near 1.5e-6 rad. The code keeps the research arithmetic so that scenes are reproduced bit for bit. In the 40,000 scenes the
  research code stored, the worst rotation is orthogonal to 9.4e-9, which distorts a body by at most a few nanometres.
- Threads: by default the chart uses every core (`PairChart(threads=...)` to change it). Results do not depend on the number of
  threads.
