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
"""Action spaces for Spriteworld.

This file contains action space classes compatible with Spriteworld.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dm_env import specs
import numpy as np


class SelectMove(object):
  """Select-Move action space.

  This action space takes in a continuous vector of length 4 with each component
  in [0, 1]. This can be intuited as representing two consecutive clicks:
    [first_x, first_y, second_x, second_y].

  These two clicks are then processed to generate a position and a motion:
    * Position = [first_x, first_y]
    * Motion = scale * [second_x - 0.5, second_y - 0.5]

  If the Position, viewed as a point in the arena, lies inside of a sprite, that
  sprite will be moved by Motion, which is a scaled version of the second click
  relative to the center of the arena. If the Position does not lie inside of a
  sprite then no sprite will move. So to move a sprite you have to click on it
  and click on the direction you want it to move, like a touch screen.

  There is an optional control cost proportional to the norm of the motion.
  """

  def __init__(self, scale=1.0, motion_cost=0.0, noise_scale=None):
    """Constructor.

    Args:
      scale: Multiplier by which the motion is scaled down. Should be in [0.0,
        1.0].
      motion_cost: Factor by which motion incurs cost.
      noise_scale: Optional stddev of the noise. If scalar, applied to all
        action space components. If vector, must have same shape as action.
    """
    self._scale = scale
    self._motion_cost = motion_cost
    self._noise_scale = noise_scale
    self._action_spec = specs.BoundedArray(
        shape=(4,), dtype=np.float32, minimum=0.0, maximum=1.0)

  def get_motion(self, action):
    delta_pos = (action[2:] - 0.5) * self._scale
    return delta_pos

  def apply_noise_to_action(self, action):
    if self._noise_scale:
      noise = np.random.normal(
          loc=0.0, scale=self._noise_scale, size=action.shape)
      return action + noise
    else:
      return action

  def get_sprite_from_position(self, position, sprites):
    for sprite in sprites[::-1]:
      if sprite.contains_point(position):
        return sprite
    return None

  def step(self, action, sprites, keep_in_frame):
    """Take an action and move the sprites.

    Args:
      action: Numpy array of shape (4,) in [0, 1]. First two components are the
        position selection, second two are the motion selection.
      sprites: Iterable of sprite.Sprite() instances. If a sprite is moved by
        the action, its position is updated.
      keep_in_frame: Bool. Whether to force sprites to stay in the frame by
        clipping their centers of mass to be in [0, 1].

    Returns:
      Scalar cost of taking this action.
    """
    noised_action = self.apply_noise_to_action(action)
    position = noised_action[:2]
    motion = self.get_motion(noised_action)
    clicked_sprite = self.get_sprite_from_position(position, sprites)
    if clicked_sprite is not None:
      clicked_sprite.move(motion, keep_in_frame=keep_in_frame)

    return -self._motion_cost * np.linalg.norm(motion)

  def sample(self):
    """Sample an action uniformly randomly."""
    return np.random.uniform(0., 1., size=(4,))

  def action_spec(self):
    return self._action_spec

