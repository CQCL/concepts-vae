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


def gen_images(colour, size, shape, position):
    renderer = renderers.PILRenderer(
                image_size=(config['render_size'], config['render_size']),
                color_to_rgb=renderers.color_maps.hsv_to_rgb
                if config['task_hsv_colors'] else None,
                anti_aliasing=config['anti_aliasing'])

    for i in range(config['num_images']):
        sprite = get_sprite(colour, size, shape, position)
        image = renderer.render(sprite)
        im = Image.fromarray(image)
        im.save(os.path.join('images', config['file_name'], config['file_name'] + '_' + str(i) + '.png'))


if __name__ == '__main__':
    gen_images('blue', 'small', 'triangle', 'top')
