import cirq
import itertools
import json

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
    all_combinations = list(itertools.product('IZ', repeat=len(qubits)))
    non_zero_coeff = 2 / len(all_combinations)
    zero_coeff = non_zero_coeff - 1
    pauli_ops = [
        cirq.DensePauliString(combination, coefficient=non_zero_coeff).on(*qubits)
        for combination in all_combinations[1:]
    ]
    pauli_ops.append(
        cirq.DensePauliString(all_combinations[0], coefficient=zero_coeff).on(*qubits)
    )
    measurement_operator = cirq.PauliSum.from_pauli_strings(pauli_ops)
    return measurement_operator
