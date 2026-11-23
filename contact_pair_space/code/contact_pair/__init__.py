"""contact_pair: a search space of touching scenes for a mug and a spoon. See README.md.

    from contact_pair import PairChart
    chart = PairChart()                                   # the bodies from data/, the exact sweep from contact_pair/_native/
    st = chart.sample_start(1000, seed=1)                 # the starting law
    R, t, info = chart.realize(st['yA'], st['yB'], st['theta'], st['u'])   # states -> scenes
"""
from .chart import PairChart
from .bodies import Body, load_bodies
from .lift import Lifter, native_available
from .postures import resting_postures, height_from_balls
from .pockets import PocketMap
from .rotations import unit, minrot, section, rot_axis

__all__ = ['PairChart', 'Body', 'load_bodies', 'Lifter', 'native_available', 'resting_postures', 'height_from_balls', 'PocketMap', 'unit', 'minrot', 'section', 'rot_axis']
__version__ = '1.0.0'
