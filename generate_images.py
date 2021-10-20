from __future__ import absolute_import

import os
from PIL import Image
import random
from spriteworld import renderers
from spriteworld.concepts import get_sprite
from itertools import product

config = {
    'task_hsv_colors': True,
    'render_size': 64,
    'anti_aliasing': 10,
    'num_images': 3000,
    'folder_name': 'basic'
}


def gen_images(parameters):
    renderer = renderers.PILRenderer(
                image_size=(config['render_size'], config['render_size']),
                color_to_rgb=renderers.color_maps.hsv_to_rgb
                if config['task_hsv_colors'] else None,
                anti_aliasing=config['anti_aliasing'])

    for i in range(config['num_images']):
        colour, size, shape, position = random.choice(parameters)
        sprite = get_sprite(colour, size, shape, position)
        image = renderer.render(sprite)
        im = Image.fromarray(image)
        im.save(os.path.join('images', config['folder_name'], 
            str(i) + '_' + colour + '_' + size + '_' + shape + '_' + position + '.png'))
        # im.save(os.path.join('images', config['folder_name'], 
        #     str(i) + '_' + colour + '_' + shape + '_' + str(sprite[0].c0) + '_' + str(sprite[0].c1) + '_' + str(sprite[0].c2) + '.png'))


if __name__ == '__main__':

    gen_images(list
        (product(
        ['red', 'green', 'blue'],
        ['small', 'medium', 'large'],
        ['circle', 'square', 'triangle'],
        ['top', 'centre', 'bottom']
    )))
