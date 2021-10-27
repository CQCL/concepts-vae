from itertools import product
import numpy as np
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
from tensorflow import keras

from vae.data_generator import ImageGenerator
from vae import utils


vae = keras.models.load_model('saved_models/vae_weights_October_26_19:26')
data_it = ImageGenerator('images/basic', batch_size=1000)

latent_space = vae.encoder.predict(data_it)

#change min and max of the plots based on the appropriate ranges for the latent space
dim_min = [-9, -4, -1, -5, -4, -4]
dim_max = [6, 2, 5, 8, 4, 4]

for i in range(6):
    utils.plot_latent_space(vae, [latent_space[2][33]], i, dim_min[i], dim_max[i], 
        num_images=30, figsize=15, file_name='images/latent_dimensions/img_33')


utils.save_reconstructed_images_with_data(vae, data_it, num_images=100, folder_name='images/reconstructed/', file_name='reconstructed')
utils.save_vae_clusters(vae, data_it, 6, 'images/clusters/cluster')

concept1 = ['red','large','square','bottom']
utils.generate_images_from_concept(vae, concept1, 20, 'images/concept_images/')
