import itertools
import numpy as np
import tensorflow as tf

from vae.model import Sampling
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
from tensorflow import keras

from vae import encoding_dictionary as enc
from vae.data_generator import ImageGenerator
from vae.utils import encode_or_decode, get_concept_gaussians, save_image

vae = keras.models.load_model('saved_models/vae_weights_December_13_17:23')
data_it = ImageGenerator('images/basic', batch_size=1)


CONCEPT_NAMES = [['blue', 'red', 'green'],
                 ['small', 'medium', 'large'],
                 ['circle', 'square', 'triangle'],
                 ['top', 'centre', 'bottom']]

NUM_LATENT_DIM = 6

# classification using decoder
def classify_using_decoder(image, model, concept_names=CONCEPT_NAMES, num_samples=10, save_images=False):
    if save_images:
        save_image('images/classify/', 'image_to_classify', [image])
    sampling = Sampling()
    means, log_vars = get_concept_gaussians(concept_names, model)
    concept_gaussians_dict = create_dict(concept_names, means, log_vars)
    concept_combinations = list(itertools.product(*concept_names))

    # for all concept combinations, reconstruct an image using concept gaussian and get the reconstruction loss
    reconstruction_losses = {}
    for concept_combination in concept_combinations:
        gaussians = [concept_gaussians_dict[concept] for concept in concept_combination]
        # add unit normal gaussians to make the length of the gaussians equal to the number of latent dimensions
        unit_gaussians = np.array([(0, 1)] * (NUM_LATENT_DIM - len(gaussians)))
        gaussians = np.concatenate([np.array(gaussians), unit_gaussians])

        # stack gaussians, concept labels, and image num_samples times
        gaussians = np.stack([gaussians] * num_samples, axis=0)
        gaussians = gaussians.reshape((2, num_samples, NUM_LATENT_DIM))
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
    return sorted_reconstruction_losses


def create_dict(keys, *values):
    # flatten lists
    keys = list(itertools.chain.from_iterable(keys))
    values = [list(itertools.chain.from_iterable(value)) for value in values]
    # zip values
    values = list(zip(*values))
    # create a dictionary from two lists
    dictionary = dict(zip(keys, values))
    return dictionary


def classify_using_encoder(image, model, concept_names=CONCEPT_NAMES, num_samples=10, save_images=False):
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
        total_loss, reconstruction_loss, kl_loss = model.compute_loss(((images, labels),))
        total_losses[concept_combination] = total_loss
        reconstruction_losses[concept_combination] = reconstruction_loss
        kl_losses[concept_combination] = kl_loss
    
    # sort the dictionary by the loss
    total_losses = sorted(total_losses.items(), key=lambda x: x[1])
    reconstruction_losses = sorted(reconstruction_losses.items(), key=lambda x: x[1])
    kl_losses = sorted(kl_losses.items(), key=lambda x: x[1])

    return total_losses, reconstruction_losses, kl_losses


success = 0
print('truth\t\t\t\t\t prediction')
itr = len(data_it)
# itr = 100
for i in range(itr):
    total_losses, reconstruction_losses, kl_losses = classify_using_encoder(data_it[i][0][0][0], vae, num_samples=20)
    # reconstruction_losses = classify_using_decoder(data_it[i][0][0][0], vae, num_samples=20)
    prediction_label = list(total_losses[0][0])
    true_label = encode_or_decode(data_it[i][0][1][0])
    print(true_label, prediction_label)
    if true_label == prediction_label:
        success += 1
print('accuracy: ' + str(success / itr * 100) + '%')