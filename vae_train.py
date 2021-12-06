import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"  

from datetime import datetime
import tensorflow as tf
from vae.callbacks import GaussianPlotCallback, ImageSaveCallback
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
from tensorflow import keras

from vae.model import VAE
from vae.data_generator import ImageGenerator


IMAGE_DIR='images/basic/'
BATCH_SIZE=16
NUM_EPOCHS=200


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
    'model_type': 'conceptual',
    # 'model_type': 'conditional',
    'use_labels_in_encoder': True,
    'if_regularize_unit_normal': True,
    'beta': 1
}

vae = VAE(params)
# vae.compile(optimizer=keras.optimizers.Adam(), run_eagerly=True)
vae.compile(optimizer=keras.optimizers.Adam())

tbCallBack = keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=0, write_graph=True, write_images=True, update_freq='batch' )
imgCallback = ImageSaveCallback(data_it[0], 'images/training/')
gaussCallback = GaussianPlotCallback('images/training/')

vae.fit(data_it, epochs=NUM_EPOCHS, steps_per_epoch=len(data_it), callbacks=[tbCallBack, gaussCallback])

gaussCallback.save_video_from_images('gaus_vid_regularized')

save_location = os.path.join('saved_models', 'vae_weights_' + datetime.utcnow().strftime("%B_%d_%H:%M"))
vae.save(save_location)
