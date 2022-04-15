import os

# set to "0" or "1", to use GPU0 or GPU1; set to "-1" to use CPU
# it is better to make only one GPU visible because tensorflow
# allocates memory on both GPUs even if you only use one of them
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["TF_GPU_THREAD_MODE"]="gpu_private"  # when using GPU; allocates a separate thread on GPU for optimised performance

from datetime import datetime  # for adding date and time stamps to names

import tensorflow as tf
from tensorflow import keras

from quantum.model import Qoncepts
from vae.data_generator import get_tf_dataset  # imports the data generator function

# configuring tensorflow
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


IMAGE_DIR='images/basic_train/'   # location of dataset images
BATCH_SIZE=32
NUM_EPOCHS=200

# prepare dataset
dataset_tf, image_input_shape = get_tf_dataset(IMAGE_DIR, BATCH_SIZE, return_image_shape=True)

params = {
    'input_shape': [image_input_shape, (4,)], # 4 labels
    'num_domains': 4,
    'num_qubits_per_domain': 1,
    'mixed_states': False,
    'num_encoder_pqc_layers': 1,
    'num_concept_pqc_layers': 1,
    'num_cnn_layers': 4,    # number of convolutional layers
    'kernel_size': 4,   # the size of the sliding window in CNN
    'num_strides': 2,   # the size of the step for which the sliding window is moved in CNN
}

qoncepts = Qoncepts(params)
# qoncepts.compile(optimizer=tf.keras.optimizers.Adam(), run_eagerly=True)  # to run step-by-step
qoncepts.compile(optimizer=tf.keras.optimizers.Adam())

tbCallBack = keras.callbacks.TensorBoard(log_dir='logs/{}'.format(datetime.now().strftime("%B_%d_%H_%M")), 
                                         histogram_freq=0,
                                         write_graph=True,
                                         write_images=True,
                                         update_freq='batch'
                                         )
callbacks = [tbCallBack]

qoncepts.fit(dataset_tf, epochs=NUM_EPOCHS, callbacks=callbacks)

# saving weights for the current trained model with a time stamp
file_name = os.path.join(
    'saved_models',
    'qoncepts_' + datetime.utcnow().strftime("%B_%d_%H_%M")
)
qoncepts.save_model(file_name)
