import itertools
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix

from vae import encoding_dictionary as enc
from vae.model import Sampling

from vae.data_generator import ImageGenerator
from vae.utils import encode_or_decode, get_concept_gaussians, save_image

# configuring tensorflow
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


vae = keras.models.load_model('saved_models/vae_weights_December_14_11:29')
data_it = ImageGenerator('images/basic', batch_size=1)


CONCEPT_NAMES = [['blue', 'red', 'green'],
                 ['small', 'medium', 'large'],
                 ['circle', 'square', 'triangle'],
                 ['top', 'centre', 'bottom']]


# classification using decoder
def decoder_classifier(image, model, concept_names=CONCEPT_NAMES, 
                           num_samples=10, save_images=False, return_prediction_list=False):
    if save_images:
        save_image('images/classify/', 'image_to_classify', [image])
    sampling = Sampling()
    if model.get_config()['model_type'] == 'conceptual':
        means, log_vars = get_concept_gaussians(concept_names, model)
        concept_gaussians_dict = create_dict(concept_names, means, log_vars)
    concept_combinations = list(itertools.product(*concept_names))

    # for all concept combinations, reconstruct an image using concept gaussian and get the reconstruction loss
    reconstruction_losses = {}
    for concept_combination in concept_combinations:
        if model.get_config()['model_type'] == 'conceptual':
            concept_gaussians = np.array([concept_gaussians_dict[concept] for concept in concept_combination])
            # add unit normal gaussians to make the length of the gaussians equal to the number of latent dimensions
            unit_gaussians = np.array([(0, 0)] * (model.get_config()['latent_dim'] - len(concept_gaussians)))
            gaussians = np.concatenate([concept_gaussians, unit_gaussians])
        else:
            # if model type is conditional, then gausians are just unit normal gaussians
            gaussians =  np.array([(0, 0)] * (model.get_config()['latent_dim']))

        # stack gaussians, concept labels, and image num_samples times
        gaussians = np.stack([gaussians.T] * num_samples, axis=1)
        labels = np.stack([encode_or_decode(concept_combination)] * num_samples, axis=0)
        images = np.stack([image] * num_samples, axis=0)

        # set dtype to float32
        gaussians = np.array(gaussians, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)
        images = np.array(images, dtype=np.float32)

        gaussians_sample = sampling(gaussians)
        reconstruction_loss = model.compute_reconstruction_loss((images, labels), gaussians_sample)
        reconstruction_losses[concept_combination] = reconstruction_loss

    # sort the dictionary by the reconstruction loss
    sorted_reconstruction_losses = sorted(reconstruction_losses.items(), key=lambda x: x[1])
    if return_prediction_list:
        return sorted_reconstruction_losses
    else:
        return list(sorted_reconstruction_losses[0][0])


def create_dict(keys, *values):
    # flatten lists
    keys = list(itertools.chain.from_iterable(keys))
    values = [list(itertools.chain.from_iterable(value)) for value in values]
    # zip values
    values = list(zip(*values))
    # create a dictionary from two lists
    dictionary = dict(zip(keys, values))
    return dictionary


def encoder_classifier(image, model, concept_names=CONCEPT_NAMES, 
                           num_samples=10, save_images=False, return_prediction_list=False):
    if save_images:
        save_image('images/classify/', 'image_to_classify', [image])
    concept_combinations = list(itertools.product(*concept_names))
    # for each concept combination, encode the image and get the loss
    total_losses = {}
    reconstruction_losses = {}
    kl_losses = {}
    for concept_combination in concept_combinations:
        labels = np.stack([encode_or_decode(concept_combination)] * num_samples, axis=0)
        images = np.stack([image] * num_samples, axis=0)
        labels = np.array(labels, dtype=np.float32)
        images = np.array(images, dtype=np.float32)
        total_loss, reconstruction_loss, kl_loss = model.compute_loss((images, labels))
        total_losses[concept_combination] = total_loss
        reconstruction_losses[concept_combination] = reconstruction_loss
        kl_losses[concept_combination] = kl_loss
    
    # sort the dictionary by the loss
    total_losses = sorted(total_losses.items(), key=lambda x: x[1])
    reconstruction_losses = sorted(reconstruction_losses.items(), key=lambda x: x[1])
    kl_losses = sorted(kl_losses.items(), key=lambda x: x[1])
    if return_prediction_list:
        return total_losses, reconstruction_losses, kl_losses
    else:
        return list(total_losses[0][0])


def encoder_mean_distance_classifier(image, model, concept_names=CONCEPT_NAMES, save_images=False):
    if save_images:
        save_image('images/classify/', 'image_to_classify', [image])

    if model.get_config()['use_labels_in_encoder']:
        raise ValueError('Encoder mean distance classifier is not compatible with models that input labels in the encoder')
    
    num_concept_domains = len(concept_names)
    empty_labels = tf.zeros((1, num_concept_domains))  # create empty labels because encoder input type still requires labels
    images = np.array([image], dtype=np.float32)
    z_mean, _, _ = model.encoder((images, empty_labels))
    z_mean = z_mean.numpy()[0]

    # for each concept domain, get the concept with mean closest to z_mean
    result = []
    for i, concepts in enumerate(concept_names):
        encoding_dict = enc.enc_dict[enc.concept_domains[i]]
        concepts_enc = np.array([encoding_dict[concept] for concept in concepts])
        concept_means = model.concept_gaussians.mean.numpy()[i][concepts_enc]
        distances = [np.abs(concept_mean - z_mean[i]) for concept_mean in concept_means]
        result.append(concepts[np.argmin(distances)])
    return result
    

def print_results(result, title, concept_names=CONCEPT_NAMES):
    print('\n' + title + '\n')
    for i in range(len(concept_names)):
        print(enc.concept_domains[i])
        print(result[i])


num_samples = 50 # number of samples to use for classification
num_images = 200 # number of images to classify
encoder_prediction_labels = []
decoder_prediction_labels = []
mean_distance_prediction_labels = []
truth_labels = []
for i in range(num_images):
    print("Classifying image " + str(i) + " of " + str(num_images), end='\r')
    truth_labels.append(encode_or_decode(data_it[i][1][0]))
    encoder_prediction_labels.append(encoder_classifier(data_it[i][0][0], vae, num_samples=num_samples))
    decoder_prediction_labels.append(decoder_classifier(data_it[i][0][0], vae, num_samples=num_samples))
    mean_distance_prediction_labels.append(encoder_mean_distance_classifier(data_it[i][0][0], vae))
print('\n')
encoder_prediction_labels = np.array(encoder_prediction_labels).T
decoder_prediction_labels = np.array(decoder_prediction_labels).T
mean_distance_prediction_labels = np.array(mean_distance_prediction_labels).T
truth_labels = np.array(truth_labels).T

# create classification report for each concept domain
encoder_classification_reports = []
decoder_classification_reports = []
mean_distance_classification_reports = []
for i in range(len(CONCEPT_NAMES)):
    encoder_classification_reports.append(classification_report(truth_labels[i], encoder_prediction_labels[i]))
    decoder_classification_reports.append(classification_report(truth_labels[i], decoder_prediction_labels[i]))
    mean_distance_classification_reports.append(classification_report(truth_labels[i], mean_distance_prediction_labels[i]))

# create confusion matrix
encoder_confusion_matrix = []
decoder_confusion_matrix = []
mean_distance_confusion_matrix = []
for i in range(len(CONCEPT_NAMES)):
    encoder_confusion_matrix.append(confusion_matrix(truth_labels[i], encoder_prediction_labels[i], labels=CONCEPT_NAMES[i]))
    decoder_confusion_matrix.append(confusion_matrix(truth_labels[i], decoder_prediction_labels[i], labels=CONCEPT_NAMES[i]))
    mean_distance_confusion_matrix.append(confusion_matrix(truth_labels[i], mean_distance_prediction_labels[i], labels=CONCEPT_NAMES[i]))

print_results(encoder_classification_reports, 'Encoder Classification Report')
print_results(decoder_classification_reports, 'Decoder Classification Report')
print_results(mean_distance_classification_reports, 'Mean Distance Classification Report')
print_results(encoder_confusion_matrix, 'Encoder Confusion Matrix')
print_results(decoder_confusion_matrix, 'Decoder Confusion Matrix')
print_results(mean_distance_confusion_matrix, 'Mean Distance Confusion Matrix')
