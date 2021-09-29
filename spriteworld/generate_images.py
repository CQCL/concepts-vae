from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from PIL import Image
import os

from spriteworld import renderers
from spriteworld.concepts import get_sprite


def gen_images(render_size, task_hsv_colors, anti_aliasing, save_dir, file_name, num_images=10):
  renderer = renderers.PILRenderer(
              image_size=(render_size, render_size),
              color_to_rgb=renderers.color_maps.hsv_to_rgb
              if task_hsv_colors else None,
              anti_aliasing=anti_aliasing)

  for i in range(num_images):
    sprite = get_sprite()
    image = renderer.render(sprite)
    im = Image.fromarray(image)
    im.save(os.path.join(save_dir, file_name + '_' + str(i) + '.png'))