import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"  
os.environ["TF_GPU_THREAD_MODE"]="gpu_private"

from datetime import datetime
import tensorflow as tf
from vae.callbacks import GaussianPlotCallback, ImageSaveCallback
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
from tensorflow import keras

from vae.model import VAE
from vae.data_generator import get_tf_dataset
from vae import encoding_dictionary as enc


IMAGE_DIR='images/basic/'
BATCH_SIZE=32
NUM_EPOCHS=200

dataset_tf, image_input_shape = get_tf_dataset(IMAGE_DIR, BATCH_SIZE, return_image_shape=True)

params = {
    'num_strides': 2,
    'kernel_size': 4,
    'latent_dim': 6,
    'pool_size': (2,2),
    'num_channels': image_input_shape[2],
    'input_shape': [image_input_shape, (len(enc.concept_domains),)],
    'model_type': 'conceptual',
    # 'model_type': 'conditional',
    'use_labels_in_encoder': True,
    'gaussians_mean_init': (-1., 1.),
    'gaussians_log_var_init': (0.7, 0.),
    'if_regularize_unit_normal': False,
    'unit_normal_regularization_factor': 0.1, # set to 0 if you don't want to regularize
    'beta': 1
}

vae = VAE(params)
# vae.compile(optimizer=keras.optimizers.Adam(), run_eagerly=True)
vae.compile(optimizer=keras.optimizers.Adam())

# call the model to build it
sample_input = list(dataset_tf.take(1).as_numpy_iterator())[0]
vae.predict(sample_input)

tbCallBack = keras.callbacks.TensorBoard(log_dir='logs', 
                                         histogram_freq=0,
                                         write_graph=True,
                                         write_images=True,
                                         update_freq='batch',
                                        #  profile_batch=(600,650)
                                         )
imgCallback = ImageSaveCallback(sample_input[0][0], 'images/training/')
gaussCallback = GaussianPlotCallback('images/training/')
# add/remove callbacks if you want
callbacks = [tbCallBack]

vae.fit(dataset_tf, epochs=NUM_EPOCHS, callbacks=callbacks)

if gaussCallback in callbacks:
    gaussCallback.save_video_from_images('gaus_vid')

save_location = os.path.join('saved_models', 'vae_weights_' + datetime.utcnow().strftime("%B_%d_%H:%M"))
vae.save(save_location)
