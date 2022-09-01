from __future__ import absolute_import

import os
from PIL import Image
import random
from spriteworld import constants
from spriteworld import factor_distributions as distribs
from spriteworld import renderers
from spriteworld.concepts import get_sprite
from itertools import product

config = {
    'task_hsv_colors': True,
    'render_size': 64,
    'anti_aliasing': 10,
    'num_images': 100,
    'folder_name': 'twake_test'
}


def gen_images_twake(parameters, non_correlation_threshold=0.3):
    renderer = renderers.PILRenderer(
                image_size=(config['render_size'], config['render_size']),
                color_to_rgb=renderers.color_maps.hsv_to_rgb
                if config['task_hsv_colors'] else None,
                anti_aliasing=config['anti_aliasing'])

    for i in range(config['num_images']):
        colour, shape = random.choice(parameters)
        if random.choice([True, False]):
            correlated = True
            file_name = os.path.join('images', config['folder_name'], 
            str(i) + '_' + colour + '_' + shape + '_twake.png')
        else:
            correlated = False
            file_name = os.path.join('images', config['folder_name'], 
            str(i) + '_' + colour + '_' + shape + '_not_twake.png')
        size, position = get_size_position(correlated, non_correlation_threshold)
        sprite_factors = distribs.Product([
            constants.COLOURS[colour],
            distribs.Discrete('scale', [size]), #size
            constants.SHAPE[shape],
            distribs.Product([distribs.Discrete('y', [position]), distribs.Discrete('x', [0.5])]), #position
            distribs.Continuous('c1', 0.5, 1.), #saturation
            distribs.Continuous('c2', 0.9, 1.), #brightness
        ])
        sprite = get_sprite(sprite_factors=sprite_factors)
        image = renderer.render(sprite)
        im = Image.fromarray(image)
        im.save(file_name)

def get_size_position(correlated=True, non_correlation_threshold=0.2):
    size_range = [0.1, 0.3]
    position_range = [0.2, 0.74]
    if correlated:
        size_rand = random.uniform(0.0, 1.0)
        pos_rand = size_rand
    else:
        pos_rand = random.uniform(0.0, 1.0)
        size_rand = random.choice([
            random.uniform(0.0, pos_rand - non_correlation_threshold),
            random.uniform(pos_rand + non_correlation_threshold, 1.0),
        ])
    size = size_range[0] + size_rand * (size_range[1] - size_range[0])
    position = position_range[0] + pos_rand * (position_range[1] - position_range[0])

    return size,position


if __name__ == '__main__':

    gen_images_twake(list
        (product(
        ['red', 'green', 'blue'],
        ['circle', 'square', 'triangle'],
    )))
