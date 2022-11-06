import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from quantum.utils import load_learned_concept, load_saved_model

from vae.data_generator import ImageGenerator
from vae.utils import encode_or_decode


# configuring tensorflow
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


def learned_qoncepts_test(image, model):
    """
    Tests if the image is an example of the learned concept.
    """
    image = np.expand_dims(image, axis=0)
    expectation = model.call(image)
    true_expectation = np.ones_like(expectation)
    false_expectation = np.zeros_like(expectation)
    if np.linalg.norm(true_expectation - expectation) < np.linalg.norm(false_expectation - expectation):
        return True
    else:
        return False


IMAGE_DIR = 'images/twake_test'
QONCEPTS_MODEL='saved_models/qoncepts_April_15_01_13'
CONCEPT_DOMAINS = [1, 3] # 0 for colour, 2 for shape
NUM_IMAGES = 300 # number of images to classify
NUM_CONCEPT_PQC_LAYERS = 2
MIXED = False

data_it = ImageGenerator(IMAGE_DIR, batch_size=1, encode_labels=False)
learned_qoncept_file = 'saved_models/learned_concept_decoder_twake_September_21_21_29'
qoncepts = load_saved_model(QONCEPTS_MODEL, image_dir=IMAGE_DIR)
learned_qoncept = load_learned_concept(
    learned_qoncept_file,
    qoncepts=qoncepts,
    concept_domains=CONCEPT_DOMAINS,
    num_concept_pqc_layers=NUM_CONCEPT_PQC_LAYERS,
    mixed=MIXED,
    image_dir=IMAGE_DIR,
)

def concept_truth(labels):
    # if 'red' in labels or 'square' in labels:
    #     classification = True
    # elif 'blue' in labels and 'circle' in labels:
    #     classification = True
    #this if for twake concept
    if 'not' in labels:
        classification = False
    else:
        classification = True
    return classification

qoncepts_prediction_labels = []
truth_labels = []
for i in range(NUM_IMAGES):
    print("Classifying image " + str(i+1) + " of " + str(NUM_IMAGES), end='\r')
    truth_labels.append(concept_truth(data_it[i][1][0]))
    qoncepts_prediction_labels.append(learned_qoncepts_test(data_it[i][0][0], learned_qoncept))

print(classification_report(truth_labels, qoncepts_prediction_labels))
print(confusion_matrix(truth_labels, qoncepts_prediction_labels))
