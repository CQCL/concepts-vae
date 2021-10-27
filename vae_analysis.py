from itertools import product
import numpy as np
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
from tensorflow import keras

from vae.data_generator import ImageGenerator
from vae import utils




vae = keras.models.load_model('saved_models/vae_weights')


utils.save_reconstructed_images_with_data(vae, data_it, num_images=100, folder_name='images/reconstructed/', file_name='reconstructed')
utils.save_vae_clusters(vae, data_it, 6, 'images/clusters/cluster')

concept1 = ['red','large','square','bottom']
utils.generate_images_from_concept(vae, concept1, 20, 'images/concept_images/')
