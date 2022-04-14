import numpy as np
import qutip
from scipy.spatial.transform import Rotation as R
import tensorflow as tf

from quantum.model import load_saved_model
from vae.data_generator import get_tf_dataset
import vae.encoding_dictionary as enc


# function that rotates a vector
def rotate(vec, rotation_vector):
    r = R.from_rotvec(rotation_vector)
    return r.apply(vec)

def rotate_zero_state(rotation_vector=np.array([0, 0, 0])):
    return rotate(np.array([0, 0, 1]), rotation_vector)


qoncepts = load_saved_model('saved_models/qoncepts_April_14_16_24')

for i, domain in enumerate(enc.concept_domains):
    b = qutip.Bloch()
    for j in range(len(enc.enc_dict[domain])-1):
        rotation_vector = qoncepts.concept_pqcs.pqc_params[i][j]
        vec = rotate_zero_state(rotation_vector)
        b.add_vectors(vec)
        b.add_annotation(vec, enc.dec_dict[domain][j])
    b.save(name='images/bloch/' + domain + '_bloch.png')
