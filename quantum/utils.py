import cirq
import json

from functools import reduce
from vae.data_generator import get_tf_dataset


def load_saved_model(file_name, image_dir='images/basic_train'):
    """
    Loads a saved model from the file `file_name`.
    """
    with open(file_name + '_params.json', 'r') as f:
        params = json.load(f)
    dataset_tf = get_tf_dataset(image_dir, 1, return_image_shape=False)
    from quantum.model import Qoncepts
    qoncepts = Qoncepts(params)
    qoncepts.compile()
    sample_input = list(dataset_tf.take(1).as_numpy_iterator())[0]
    qoncepts(sample_input)
    qoncepts.load_weights(file_name + '.h5')
    return qoncepts

def load_learned_concept(file_name, image_dir='images/basic_train', **kwargs):
    """
    Loads a learned concepts from the file `file_name`.
    """
    dataset_tf = get_tf_dataset(image_dir, 1, return_image_shape=False)
    from quantum.concept_learner import ConceptLearner
    learned_concept = ConceptLearner(**kwargs)
    learned_concept.compile()
    sample_input = list(dataset_tf.take(1).as_numpy_iterator())[0]
    learned_concept(sample_input[0])
    learned_concept.load_weights(file_name + '.h5')
    return learned_concept

def create_zeros_measurement_operator(qubits):
    I_plus_Z_by_2 = []
    for q in qubits:
        I = cirq.PauliString(cirq.I(q), coefficient=0.5)
        Z = cirq.PauliString(cirq.Z(q), coefficient=0.5)
        I_plus_Z_by_2.append(cirq.PauliSum.from_pauli_strings([I, Z]))
    I_plus_Z_by_2_tensored = reduce(cirq.mul, I_plus_Z_by_2)
    return I_plus_Z_by_2_tensored
