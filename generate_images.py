from __future__ import absolute_import

import os
from PIL import Image
from spriteworld import renderers
from spriteworld.concepts import get_sprite

config = {
    'task_hsv_colors': True,
    'render_size': 256,
    'anti_aliasing': 10,
    'num_images': 10,
    'file_name': 'red_circle'
}


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


def main():
    gen_images(config['render_size'], config['task_hsv_colors'],
                        config['anti_aliasing'], os.path.join('images', config['file_name']), 
                        config['file_name'], config['num_images'])


if __name__ == '__main__':
    main()
