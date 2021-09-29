"""Start demo GUI for Spriteworld task configs.

To play a task, run this on the task config:
```bash
python run_demo.py --config=$path_to_task_config$
```

Be aware that this demo overrides the action space and renderer for ease of
playing, so those will be different from what are specified in the task config.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import importlib
from absl import app
from absl import flags
from spriteworld import generate_images

FLAGS = flags.FLAGS
flags.DEFINE_string('config', 'spriteworld.configs.concepts',
                    'Module name of task config to use.')
flags.DEFINE_boolean('task_hsv_colors', True,
                     'Whether the task config uses HSV as color factors.')
flags.DEFINE_integer('render_size', 256,
                     'Height and width of the output image.')
flags.DEFINE_integer('anti_aliasing', 10, 'Renderer anti-aliasing factor.')
flags.DEFINE_integer('num_images', 100, 'Number of images to generate.')
flags.DEFINE_string('file_name', 'red_circle', 'Name of generated images')


def main(_):
    generate_images.gen_images(FLAGS.render_size, FLAGS.task_hsv_colors,
                        FLAGS.anti_aliasing, os.path.join('images', FLAGS.file_name), 
                        FLAGS.file_name, FLAGS.num_images)


if __name__ == '__main__':
    app.run(main)
