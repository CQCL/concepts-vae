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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from spriteworld import factor_distributions as distribs
from spriteworld import sprite_generators
from spriteworld import constants

# Task Parameters
NUM_SPRITES_PER_CLUSTER = 1


def get_sprite(colour, size, shape, position):
    sprite_factors = distribs.Product([
        constants.COLOURS[colour],
        constants.SIZE[size],
        constants.SHAPE[shape],
        constants.POSITION[position],
        distribs.Continuous('c1', 0.3, 1.), #saturation
        distribs.Continuous('c2', 0.9, 1.), #brightness
    ])

    sprite_gen_per_cluster = [
        sprite_generators.generate_sprites(
            sprite_factors, num_sprites=NUM_SPRITES_PER_CLUSTER)
    ]
    # Concat clusters into single scene to generate.
    sprite_gen = sprite_generators.chain_generators(*sprite_gen_per_cluster)
    # Randomize sprite ordering to eliminate any task information from occlusions
    sprite_gen = sprite_generators.shuffle(sprite_gen)
    sprite = sprite_gen()

    return sprite
