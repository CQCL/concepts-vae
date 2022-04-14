import itertools
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
from quantum.model import Qoncepts, load_saved_model

from vae.data_generator import ImageGenerator, get_tf_dataset
from vae.utils import encode_or_decode
import vae.encoding_dictionary as enc


# configuring tensorflow
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

CONCEPT_NAMES = [['blue', 'red', 'green'],
                 ['small', 'medium', 'large'],
                 ['circle', 'square', 'triangle'],
                 ['top', 'centre', 'bottom']]

IMAGE_DIR = 'images/basic_test'


def qoncepts_classifier(image, model, concept_names=CONCEPT_NAMES):
    """
    Classifies the image using the trained qoncepts model.
    """
    concept_combinations = list(itertools.product(*concept_names))
    # for each concept combination, encode the image and get the loss
    losses = {}
    image = np.expand_dims(image, axis=0)
    for concept_combination in concept_combinations:
        label = np.expand_dims(encode_or_decode(concept_combination), axis=0)
        label = np.array(label, dtype=np.float32)
        loss = model.compute_loss((image, label))
        losses[concept_combination] = loss
    # sort the dictionary by the loss
    losses = sorted(losses.items(), key=lambda x: x[1])
    return list(losses[0][0])


def print_results(result, title, concept_names=CONCEPT_NAMES):
    print('\n' + title + '\n')
    for i in range(len(concept_names)):
        print(enc.concept_domains[i])
        print(result[i])


data_it = ImageGenerator(IMAGE_DIR, batch_size=1)

qoncepts = load_saved_model('saved_models/qoncepts_April_14_16_24')

num_images = 100 # number of images to classify
qoncepts_prediction_labels = []
truth_labels = []
for i in range(num_images):
    print("Classifying image " + str(i) + " of " + str(num_images), end='\r')
    truth_labels.append(encode_or_decode(data_it[i][1][0]))
    qoncepts_prediction_labels.append(qoncepts_classifier(data_it[i][0][0], qoncepts))
print('\n')
qoncepts_prediction_labels = np.array(qoncepts_prediction_labels).T
truth_labels = np.array(truth_labels).T

# create classification report for each concept domain
qoncepts_classification_reports = []
for i in range(len(CONCEPT_NAMES)):
    qoncepts_classification_reports.append(classification_report(truth_labels[i], qoncepts_prediction_labels[i]))

# create confusion matrix
qoncepts_confusion_matrix = []
for i in range(len(CONCEPT_NAMES)):
    qoncepts_confusion_matrix.append(confusion_matrix(truth_labels[i], qoncepts_prediction_labels[i], labels=CONCEPT_NAMES[i]))

print_results(qoncepts_classification_reports, 'Qoncepts Classification Report')
print_results(qoncepts_confusion_matrix, 'Qoncepts Confusion Matrix')
