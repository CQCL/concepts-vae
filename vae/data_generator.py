from tensorflow import keras
import tensorflow as tf
import numpy as np
import PIL
import random
import os
from vae import encoding_dictionary as enc
from vae.utils import encode_or_decode

class ImageGenerator(keras.utils.Sequence) :
  
    def __init__(self, image_dir, batch_size) :
        self.batch_size = batch_size
        self.image_dir = image_dir
        self.image_files = []
        for root_path, _, files in os.walk(image_dir):
            for f in files:
                self.image_files.append(os.path.join(root_path, f))
        random.shuffle(self.image_files)
      
      
    def __len__(self) :
        return (np.ceil(len(self.image_files) / float(self.batch_size))).astype(np.int)
    
    
    def __getitem__(self, idx) :
        batch_x = self.image_files[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_images =  np.array([np.asarray(
            PIL.Image.open(file_path), dtype=np.float64) / 255.0 for file_path in batch_x])
        batch_labels = np.zeros((len(batch_x), len(enc.concept_domains)), dtype=float)
        for i, file_path in enumerate(batch_x):
            file_name = os.path.splitext(os.path.split(file_path)[1])[0]
            keywords = file_name.split('_')
            for j, concept in enumerate(enc.concept_domains):
                batch_labels[i][j] = enc.enc_dict[concept][keywords[j+1]]
        return [batch_images, batch_labels]

def get_tf_dataset_from_generator(data_generator, output_signature, num_images, batch_size=16):
    dataset_tf = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=output_signature
    )
    # shuffle, batch and optimize the data
    dataset_tf = dataset_tf.shuffle(buffer_size=num_images)
    dataset_tf = dataset_tf.batch(batch_size)
    dataset_tf = dataset_tf.cache()
    dataset_tf = dataset_tf.prefetch(tf.data.AUTOTUNE)
    return dataset_tf


def get_tf_dataset(image_dir, batch_size=16, return_image_shape=False, include_labels=True):
    # create a generator for the training data
    image_files = []
    for root_path, _, files in os.walk(image_dir):
        for f in files:
            image_files.append(os.path.join(root_path, f))

    img_data = np.asarray(PIL.Image.open(image_files[0]), dtype=np.float32)
    img_height = img_data.shape[0]
    img_width = img_data.shape[1]
    num_channels = img_data.shape[2]
    image_shape = (img_height, img_width, num_channels)

    def data_generator():
        for file_path in image_files:
            img_data = np.asarray(PIL.Image.open(file_path), dtype=np.float32) / 255.0
            file_name = os.path.splitext(os.path.split(file_path)[1])[0]
            keywords = file_name.split('_')
            labels = encode_or_decode(keywords[1:])
            if include_labels:
                yield tf.convert_to_tensor(img_data), tf.convert_to_tensor(labels)
            else:
                yield tf.convert_to_tensor(img_data)

    if include_labels:
            output_signature=(
                tf.TensorSpec(shape=image_shape, dtype=tf.float32),
                tf.TensorSpec(shape=(len(enc.concept_domains),), dtype=tf.float32)
            )
    else:
            output_signature=(tf.TensorSpec(shape=image_shape, dtype=tf.float32))

    dataset_tf = get_tf_dataset_from_generator(
        data_generator,
        output_signature=output_signature,
        num_images=len(image_files),
        batch_size=batch_size
        )
    if return_image_shape:
        return dataset_tf, image_shape
    return dataset_tf

