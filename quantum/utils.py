import json

from quantum.concept_learner import ConceptLearner
from quantum.model import Qoncepts
from vae.data_generator import get_tf_dataset


def load_saved_model(file_name, image_dir='images/basic_train'):
    """
    Loads a saved model from the file `file_name`.
    """
    with open(file_name + '_params.json', 'r') as f:
        params = json.load(f)
    dataset_tf = get_tf_dataset(image_dir, 1, return_image_shape=False)
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
    learned_concept = ConceptLearner(**kwargs)
    learned_concept.compile()
    sample_input = list(dataset_tf.take(1).as_numpy_iterator())[0]
    learned_concept(sample_input[0])
    learned_concept.load_weights(file_name + '.h5')
    return learned_concept

