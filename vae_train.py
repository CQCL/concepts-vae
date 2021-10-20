import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"  

import datetime
from itertools import product
import numpy as np
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
from tensorflow import keras

from vae.model import VAE
from vae.data_generator import ImageGenerator
from vae.utils import save_vae_clusters, save_reconstructed_images, generate_images_from_concept


IMAGE_DIR='images/various/'
BATCH_SIZE=16


data_it = ImageGenerator(IMAGE_DIR, BATCH_SIZE)

img_width = data_it[0][0][0].shape[1]
img_height = data_it[0][0][0].shape[2]
num_channels = 3  #3 for rgb
image_input_shape = (img_height, img_width, num_channels)


params = {
    'num_strides': 2,
    'kernel_size': 4,
    'latent_dim': 6,
    'pool_size':(2,2),
    'num_channels': num_channels,
    'input_shape': [image_input_shape, (4,)],
    'use_labels_in_encoder': False,
}

vae = VAE(params)
# vae.compile(optimizer=keras.optimizers.Adam(), run_eagerly=True)
vae.compile(optimizer=keras.optimizers.Adam())

tbCallBack = keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=0, write_graph=True, write_images=True, update_freq='batch' )

vae.fit(data_it, epochs=200, steps_per_epoch=len(data_it), callbacks=[tbCallBack])

vae.save('vae_weights')


# save_reconstructed_images(vae, data_it, num_images=100, folder_name='images/reconstructed/', file_name='reconstructed')
save_vae_clusters(vae, data_it, params['latent_dim'], 'images/clusters/cluster')

# concept1 = ['red','large','square','bottom']

# generate_images_from_concept(vae, concept1, 20, 'images/concept_images/')
