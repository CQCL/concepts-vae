# Copyright 2019 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# python2 python3
"""Constants for shapes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import enum
import numpy as np
from spriteworld import shapes
from spriteworld import factor_distributions as distribs


# A selection of simple shapes
SHAPES = {
    'triangle': shapes.polygon(num_sides=3, theta_0=np.pi/2),
    'square': shapes.polygon(num_sides=4, theta_0=np.pi/4),
    'pentagon': shapes.polygon(num_sides=5, theta_0=np.pi/2),
    'hexagon': shapes.polygon(num_sides=6),
    'octagon': shapes.polygon(num_sides=8),
    'circle': shapes.polygon(num_sides=30),
    'star_4': shapes.star(num_sides=4, theta_0=np.pi/4),
    'star_5': shapes.star(num_sides=5, theta_0=np.pi + np.pi/10),
    'star_6': shapes.star(num_sides=6),
    'spoke_4': shapes.spokes(num_sides=4, theta_0=np.pi/4),
    'spoke_5': shapes.spokes(num_sides=5, theta_0=np.pi + np.pi/10),
    'spoke_6': shapes.spokes(num_sides=6),
}

COLOURS = {
    'red': distribs.Mixture([distribs.Continuous('c0', 0.95, 1.), distribs.Continuous('c0',0.,0.05)]),
    'blue': distribs.Continuous('c0', 0.53, 0.70),
    'green': distribs.Continuous('c0', 0.27, 0.37),
    'yellow': distribs.Continuous('c0', 0.1, 0.2)
}

SIZE = {
    'small':  distribs.Continuous('scale', 0.1, 0.2),
    'medium':  distribs.Continuous('scale', 0.2, 0.3),
    'large':  distribs.Continuous('scale', 0.3, 0.4),
}

SHAPE = {
    'triangle': distribs.Discrete('shape', ['triangle']),
    'square': distribs.Discrete('shape', ['square']),
    'pentagon': distribs.Discrete('shape', ['pentagon']),
    'hexagon': distribs.Discrete('shape', ['hexagon']),
    'octagon': distribs.Discrete('shape', ['octagon']),
    'circle': distribs.Discrete('shape', ['circle']),
    'star_4': distribs.Discrete('shape', ['star_4']),
    'star_5': distribs.Discrete('shape', ['star_5']),
    'star_6': distribs.Discrete('shape', ['star_6']),
    'spoke_4': distribs.Discrete('shape', ['spoke_4']),
    'spoke_5': distribs.Discrete('shape', ['spoke_5']),
    'spoke_6': distribs.Discrete('shape', ['spoke_6']),
}

POSITION = {
    'top': distribs.Product([distribs.Continuous('y', 0.66, 0.8), distribs.Discrete('x', [0.5])]),
    'centre': distribs.Product([distribs.Continuous('y', 0.33, 0.66), distribs.Discrete('x', [0.5])]),
    'bottom': distribs.Product([distribs.Continuous('y', 0.2, 0.33), distribs.Discrete('x', [0.5])]),
}


class ShapeType(enum.IntEnum):
    """Enumerate SHAPES, useful for a state description of the environment."""
    triangle = 1
    square = 2
    pentagon = 3
    hexagon = 4
    octagon = 5
    circle = 6
    star_4 = 7
    star_5 = 8
    star_6 = 9
    spoke_4 = 10
    spoke_5 = 11
    spoke_6 = 12
