import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"   # if we do NOT want to use GPU (hides the GPU)
os.environ["TF_GPU_THREAD_MODE"]="gpu_private"  # when using GPU; allocates a separate thread on GPU for optimised performance

from datetime import datetime  # for adding date and time stamps to names
import tensorflow as tf
from tensorflow import keras

from vae import encoding_dictionary as enc
from vae.callbacks import (GaussianPlotCallback,  # for saving images; for visualising Gaussians
                           ImageSaveCallback)
from vae.data_generator import get_tf_dataset   # imports the optimised data generator function
from vae.model import VAE   

# configuring tensorflow
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


IMAGE_DIR='images/any_one/'   # location of dataset images
BATCH_SIZE=32
NUM_EPOCHS=200

# prepare dataset
dataset_tf, image_input_shape = get_tf_dataset(IMAGE_DIR, BATCH_SIZE, return_image_shape=True)

valid_concepts = {
    'colour' : ['blue', 'green', 'red'],
    'size' : ['small', 'medium', 'large'],
    'shape' : ['triangle', 'square', 'circle'],
    'position' : ['top', 'centre', 'bottom'],
}
# encode the values of the concepts using encoding dictionary
valid_concepts_encoded = {}
for domain, concepts in valid_concepts.items():
    valid_concepts_encoded[domain] = [enc.enc_dict[domain][concept] for concept in concepts]

params = {
    'model_type': 'conceptual', # choose either 'conceptual' or 'conditional'
    'beta': 1,  # factor for beta-VAE; total_loss = reconstruction_loss + beta * KL_loss
    'latent_dim': 6,    # number of dimensions in the latent space
    'num_channels': image_input_shape[2],   # in our case 3 because images are in RGB [would be 1 for black and white]
    'input_shape': [image_input_shape, (len(enc.concept_domains),)],    # shape of an instance in the dataset that is passed to VAE
                                                                        # our case: [(64,64,3),(4,)]

    # NN setup
    'kernel_size': 4,   # the size of the sliding window in CNN
    'num_strides': 2,   # the size of the step for which the sliding window is moved in CNN
#    'pool_size': (2,2), # not used atm
    'dense_dropout': 0.5, # dropout rate for dense layers; range [0,1]; set to 0 for no dropout for dense layers
    'convolutional_dropout': 0.1, # dropout rate for dense layers; range [0,1]; set to 0 for no dropout for convolutional layers

    # extra parameters for conceptual VAE
    'use_labels_in_encoder': False,  # whether we are passing labels in encoder
    'gaussians_mean_init': (-1., 1.),   # initialisation interval for means of Gaussians
    'gaussians_log_var_init': (-7, 0.),    # initialisation interval for log var of Gaussians
    'unit_normal_regularization_factor': 0, # regularisation factor for concept Gaussians; set to 0 if you don't want to regularize Gaussians
    
    # extra parameters for ANY label of conceptual VAE
    'valid_concepts': valid_concepts_encoded,  # dictionary of valid concepts for each domain
    'num_samples_for_any_kl': 10000,  # number of samples for calculating KL divergence for ANY label
}


vae = VAE(params)   # builds (initialises?) vae with above parameters
# vae.compile(optimizer=keras.optimizers.Adam(), run_eagerly=True)  # to run step-by-step
vae.compile(optimizer=keras.optimizers.Adam())

# next lines need better explanation 
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


# saving weights for the current trained model with a time stamp
save_location = os.path.join('saved_models', 'vae_weights_' + datetime.utcnow().strftime("%B_%d_%H_%M"))
vae.save(save_location)
