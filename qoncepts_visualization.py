import numpy as np
import qutip
from scipy.spatial.transform import Rotation as R
import tensorflow as tf

from quantum.model import Qoncepts
from vae.data_generator import get_tf_dataset
import vae.encoding_dictionary as enc


IMAGE_DIR = 'images/basic_test'

dataset_tf, image_input_shape = get_tf_dataset(IMAGE_DIR, 1, return_image_shape=True)
params = {
    'image_input_shape': image_input_shape,
    'num_domains': 4,
    'num_qubits_per_domain': 1,
    'mixed_states': False,
    'num_encoder_pqc_layers': 1,
    'num_concept_pqc_layers': 1,
    # NN setup
    'num_layers': 4,    # number of convolutional layers
    'kernel_size': 4,   # the size of the sliding window in CNN
    'num_strides': 2,   # the size of the step for which the sliding window is moved in CNN
}

# function that rotates a vector
def rotate(vec, rotation_vector):
    r = R.from_rotvec(rotation_vector)
    return r.apply(vec)

def rotate_zero_state(rotation_vector=np.array([0, 0, 0])):
    return rotate(np.array([0, 0, 1]), rotation_vector)


qoncepts = Qoncepts(params)
qoncepts.compile()
sample_input = list(dataset_tf.take(1).as_numpy_iterator())[0]
qoncepts(sample_input)
qoncepts.load_weights('saved_models/qoncepts_April_14_02_42.h5')



for i, domain in enumerate(enc.concept_domains):
    b = qutip.Bloch()
    for j in range(len(enc.enc_dict[domain])-1):
        rotation_vector = qoncepts.concept_pqcs.pqc_params[i][j]
        vec = rotate_zero_state(rotation_vector)
        b.add_vectors(vec)
        b.add_annotation(vec, enc.dec_dict[domain][j])
    b.save(name='images/bloch/' + domain + '_bloch.png')

