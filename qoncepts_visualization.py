import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import matplotlib as plt
from matplotlib import animation
import numpy as np
import qutip
from scipy.spatial.transform import Rotation as R
import tensorflow as tf

from quantum.utils import load_saved_model
from vae.data_generator import ImageGenerator
import vae.encoding_dictionary as enc

# function that rotates a vector
def rotate(vec, rotation_vector):
    r = R.from_rotvec(rotation_vector)
    return r.apply(vec)

def rotate_zero_state(rotation_vector=np.array([0, 0, 0])):
    state = np.array([0, 0, 1])
    state = rotate(state, [rotation_vector[0], 0, 0])
    state = rotate(state, [0, rotation_vector[1], 0])
    state = rotate(state, [0, 0, rotation_vector[2]])
    return state

def get_colour_list(n, name='plasma'):
    cmap = plt.cm.get_cmap(name, n)
    colour_list = []
    for i in range(n):
        colour_list.append(
            '#%02x%02x%02x' % tuple([int(c) for c in (np.array(cmap(i))*255)[:-1]])
        )
    return colour_list

if __name__ == '__main__':
    IMAGE_DIR = 'images/basic_train'
    qoncepts = load_saved_model('saved_models/qoncepts_October_10_18_36', image_dir=IMAGE_DIR)
    data_it = ImageGenerator(IMAGE_DIR, batch_size=1, encode_labels=False)

    image_vectors = [{conc: [] for conc in enc.enc_dict[d]} for d in enc.concept_domains]

    for data in data_it:
        params = tf.squeeze(qoncepts.encoder_cnn(data[0]))
        params = tf.split(params, 4)
        for i, p in enumerate(params):
            vector = rotate_zero_state(np.array(p))
            image_vectors[i][data[1][0][i]].append(vector)

    for i, domain in enumerate(enc.concept_domains):
        num_concepts = len(enc.enc_dict[domain])-1
        b = qutip.Bloch()
        b.point_marker = ['o']
        b.point_size = [5]
        b.view = [0, 20]
        colour_list = get_colour_list(num_concepts)
        b.point_color = colour_list
        b.vector_color = colour_list
        for j in range(num_concepts):
            concept_name = enc.dec_dict[domain][j]
            points = np.array(image_vectors[i][concept_name]).T
            if len(points) == 0:
                continue
            b.add_points(points)
            vector = qoncepts.concept_params.concept_params[i][j]
            vector = -1 * vector
            vec = rotate_zero_state(vector[::-1])
            b.add_vectors(vec)
            b.add_annotation(vec, concept_name)
        b.show()
        def animate(i):
            b.axes.view_init(azim=i, elev=20)
            return b.axes
        ani = animation.FuncAnimation(b.axes.figure, animate, np.linspace(0, 360, 360),
                                      blit=False, repeat=False)
        ani.save('images/bloch/' + domain + '_bloch.mp4', fps=60)
