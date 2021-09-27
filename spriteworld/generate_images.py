from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from PIL import Image
import os

from spriteworld import environment
from spriteworld import renderers


def gen_images(env_config, render_size, task_hsv_colors, anti_aliasing, save_dir, file_name, num_images=10):
  """Start a Demo UI given an env_config."""
  env_config['renderers'] = {
      'image':
          renderers.PILRenderer(
              image_size=(render_size, render_size),
              color_to_rgb=renderers.color_maps.hsv_to_rgb
              if task_hsv_colors else None,
              anti_aliasing=anti_aliasing),
      'success':
          renderers.Success()
  }
  env = environment.Environment(**env_config)

  for i in range(num_images):
    timestep = env.reset()
    im = Image.fromarray(timestep.observation['image'])
    im.save(os.path.join(save_dir, file_name + '_' + str(i) + '.png'))