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

import os
from spriteworld import factor_distributions as distribs
from spriteworld import sprite_generators

# Task Parameters
NUM_SPRITES_PER_CLUSTER = 1
MAX_EPISODE_LENGTH = 1

# Define possible clusters (here using Hue as selection attribute)
CLUSTERS_DISTS = {
    # 'red': distribs.Gaussian('c0', 0.95, 0.01667),
    # 'blue': distribs.Gaussian('c0', 0.6, 0.01667),
    # 'green': distribs.Gaussian('c0', 0.32, 0.01667),
    # 'yellow': distribs.Gaussian('c0', 0.15, 0.01667),
    'red': distribs.Continuous('c0', 0.9, 1.),
    'blue': distribs.Continuous('c0', 0.55, 0.65),
    'green': distribs.Continuous('c0', 0.27, 0.37),
    'yellow': distribs.Continuous('c0', 0.1, 0.2),
}

# Define train/test generalization splits
MODES = {
    'train': ('red',),
    'test': (),
}


def get_sprite(mode='train'):

  # Select clusters to use, and their c0 factor distribution.
  c0_clusters = [CLUSTERS_DISTS[cluster] for cluster in MODES[mode]]

  other_factors = distribs.Product([
      # distribs.Continuous('x', 0.1, 0.9),
      # distribs.Continuous('y', 0.1, 0.9),
      distribs.Discrete('x', [0.5]),
      distribs.Discrete('y', [0.5]),
      distribs.Discrete('shape', ['circle']),
      distribs.Discrete('scale', [0.3]),
      distribs.Continuous('c1', 0.3, 1.),
      distribs.Continuous('c2', 0.9, 1.),
      # distribs.Discrete('c1', [0.6]),
      # distribs.Discrete('c2', [0.9]),
  ])

  # Generate the sprites to be used in this task, by combining Hue with the
  # other factors.
  sprite_factors = [
      distribs.Product((other_factors, c0)) for c0 in c0_clusters
  ]
  # Convert to sprites, generating the appropriate number per cluster.
  sprite_gen_per_cluster = [
      sprite_generators.generate_sprites(
          factors, num_sprites=NUM_SPRITES_PER_CLUSTER)
      for factors in sprite_factors
  ]
  # Concat clusters into single scene to generate.
  sprite_gen = sprite_generators.chain_generators(*sprite_gen_per_cluster)
  # Randomize sprite ordering to eliminate any task information from occlusions
  sprite_gen = sprite_generators.shuffle(sprite_gen)
  sprite = sprite_gen()

  return sprite
