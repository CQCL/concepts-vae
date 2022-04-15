import os

from quantum.concept_learner import ConceptLearner
from quantum.utils import load_saved_model

# set to "0" or "1", to use GPU0 or GPU1; set to "-1" to use CPU
# it is better to make only one GPU visible because tensorflow
# allocates memory on both GPUs even if you only use one of them
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["TF_GPU_THREAD_MODE"]="gpu_private"  # when using GPU; allocates a separate thread on GPU for optimised performance

from datetime import datetime  # for adding date and time stamps to names
import tensorflow as tf
from tensorflow import keras

from quantum.model import Qoncepts
from vae.data_generator import create_data_generator_with_classification_condition, get_tf_dataset_from_generator

# configuring tensorflow
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


IMAGE_DIR='images/basic_train/'   # location of dataset images
BATCH_SIZE=32
NUM_EPOCHS=200
QONCEPTS_MODEL='saved_models/qoncepts_April_15_01_13'

def condition(labels):
    if 'red' in labels and 'square' in labels:
        classification = 1
    elif 'blue' in labels and 'circle' in labels:
        classification = 1
    else:
        classification = -1
    return classification

data_gen, output_signature, num_images = create_data_generator_with_classification_condition(IMAGE_DIR, condition)
dataset_tf = get_tf_dataset_from_generator(data_gen, output_signature, num_images, BATCH_SIZE)

qoncepts = load_saved_model(QONCEPTS_MODEL)
concept_domains = [0,1]

concept_learner = ConceptLearner(qoncepts, concept_domains, num_concept_pqc_layers=None, mixed=False)

# concept_learner.compile(optimizer=tf.keras.optimizers.Adam(), run_eagerly=True)  # to run step-by-step
concept_learner.compile(optimizer=tf.keras.optimizers.Adam())

concept_learner.fit(dataset_tf, epochs=NUM_EPOCHS)

# saving weights for the current trained model with a time stamp
file_name = os.path.join(
    'saved_models',
    'learned_concept_' + datetime.utcnow().strftime("%B_%d_%H_%M")
)
concept_learner.save_model(file_name)
