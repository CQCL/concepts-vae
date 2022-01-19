import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod, abstractproperty
from tensorflow import keras
from tensorflow.keras import layers

import vae.encoding_dictionary as enc


class AbstractVAE(ABC):
    """Abstract class from which all VAEs should inherit."""

    def example(self, input1):
        """Description."""

    @abstractmethod
    def ex_abstract_method(self, input1):
        """Description."""

    @abstractproperty
    def ex_abstract_property(self):
        """Description."""


class Sampling(layers.Layer):
    """
    Uses [z_mean, z_log_var] to sample z, the vector encoding an image.
    """

    @tf.function(jit_compile=True)
    def call(self, inputs):
        """
        inputs: has the shape [z_mean, z_log_var], where each of those is a tensor (as everything in tensorflow)
        """
        z_mean = inputs[0]
        z_log_var = inputs[1]
        batch = tf.shape(z_mean)[0]     # number of elements in a batch
        dim = tf.shape(z_mean)[1]     # number of latent dimensions
        epsilon = keras.backend.random_normal(shape=(batch, dim))   # sampling from unit normal
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon   # scale the sample (epsilon) from unit normal to actual mean and log_var


class ConceptGaussians(layers.Layer):
    def __init__(self, mean_init=(-1., 1.), log_var_init=(0.7, 0.0), **kwargs):
        super(ConceptGaussians, self).__init__(**kwargs)
        self.mean_init = mean_init
        self.log_var_init = log_var_init

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        # the max number of different values for different domains
        max_concepts = max([len(enc.enc_dict[concept]) for concept in enc.concept_domains]) 
        self.mean = self.add_weight(name='kernel',
                                      shape=(len(enc.concept_domains), max_concepts),
                                      initializer=keras.initializers.RandomUniform(minval=self.mean_init[0], 
                                                                                   maxval=self.mean_init[1]),
                                      trainable=True)
        self.log_var = self.add_weight(name='kernel',
                                      shape=(len(enc.concept_domains), max_concepts),
                                      initializer=keras.initializers.RandomUniform(minval=self.log_var_init[0], 
                                                                                   maxval=self.log_var_init[1]),
                                      trainable=True)
        super(ConceptGaussians, self).build(input_shape)

    @tf.function(jit_compile=True)  # for faster training, just in time compilation 
    def call(self, labels, **kwargs):
        """
        labels: list of labels corresponding to an image?
        """
        labels = tf.cast(labels, tf.int32)  # casting from float to int
        # gather_nd gathers slices from self.mean into a Tensor with shape specified by indices  
        # (i.e., the needed indices corresponding to current concepts based on labels)
        # (cf., https://www.tensorflow.org/api_docs/python/tf/gather_nd)
        indices = tf.reshape(tf.transpose(labels), (tf.shape(labels)[1], tf.shape(labels)[0],1))    
        means = tf.transpose(tf.gather_nd(self.mean, indices, batch_dims=1))
        log_vars = tf.transpose(tf.gather_nd(self.log_var, indices, batch_dims=1))
        return means, log_vars
        

class VAE(keras.Model):
    def __init__(self, params, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.params = params
        self.encoder, conv_shape = self.encoder_model()
        self.decoder = self.decoder_model(conv_shape)
        self.encoder.summary()
        self.decoder.summary()
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        if  self.params['model_type'] == 'conceptual':
            self.kl_loss_function = self.kl_conceptual_fun
            self.concept_gaussians = ConceptGaussians(self.params['gaussians_mean_init'], 
                                                      self.params['gaussians_log_var_init'])
        elif self.params['model_type'] == 'conditional':
            self.kl_loss_function = self.kl_conditional_fun
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.mse = tf.keras.losses.MeanSquaredError(reduction='none')
    
    def get_config(self):
        # returns parameters with which vae was instanciated
        return self.params

    def encoder_model(self):
        encoder_image_inputs = keras.Input(shape=self.params['input_shape'][0])
        encoder_label_inputs = keras.Input(shape=self.params['input_shape'][1])
        x = layers.Conv2D(64, self.params['kernel_size'], activation="relu", strides=self.params['num_strides'], padding="same")(encoder_image_inputs)
        x = layers.Dropout(self.params['convolutional_dropout'])(x)
        x = layers.Conv2D(64, self.params['kernel_size'], activation="relu", strides=self.params['num_strides'], padding="same")(x)
        x = layers.Dropout(self.params['convolutional_dropout'])(x)
        x = layers.Conv2D(64, self.params['kernel_size'], activation="relu", strides=self.params['num_strides'], padding="same")(x)
        x = layers.Dropout(self.params['convolutional_dropout'])(x)
        x = layers.Conv2D(64, self.params['kernel_size'], activation="relu", strides=self.params['num_strides'], padding="same")(x)
        x = layers.Dropout(self.params['convolutional_dropout'])(x)
        conv_shape = tf.keras.backend.int_shape(x) # Shape of convonlutional layer to be provided to decoder
        x = layers.Flatten()(x)
        if self.params['model_type'] == 'conceptual' and self.params['use_labels_in_encoder']:
            x = layers.Concatenate()([encoder_label_inputs, x])
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(self.params['dense_dropout'])(x)
        z_mean = layers.Dense(self.params['latent_dim'], name="z_mean")(x)
        z_mean = layers.Dropout(self.params['dense_dropout'])(z_mean)
        z_log_var = layers.Dense(self.params['latent_dim'], name="z_log_var")(x)
        z_log_var = layers.Dropout(self.params['dense_dropout'])(z_log_var)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model([encoder_image_inputs, encoder_label_inputs], [z_mean, z_log_var, z], name="encoder")
        return encoder, conv_shape

    def decoder_model(self, conv_shape):
        latent_inputs = keras.Input(shape=(self.params['latent_dim'],))
        label_inputs = keras.Input(shape=self.params['input_shape'][1])
        if self.params['model_type'] == 'conditional':
            inputs = layers.Concatenate()([latent_inputs, label_inputs])
        else:
            inputs = latent_inputs
        x = layers.Dense(256, activation="relu")(inputs)
        x = layers.Dropout(self.params['dense_dropout'])(x)
        x = layers.Dense(conv_shape[1]*conv_shape[2]*conv_shape[3], activation="relu")(x)
        x = layers.Dropout(self.params['dense_dropout'])(x)
        x = layers.Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(x)
        x = layers.Conv2DTranspose(64, self.params['kernel_size'], activation="relu", strides=self.params['num_strides'], padding="same")(x)
        x = layers.Dropout(self.params['convolutional_dropout'])(x)
        x = layers.Conv2DTranspose(64, self.params['kernel_size'], activation="relu", strides=self.params['num_strides'], padding="same")(x)
        x = layers.Dropout(self.params['convolutional_dropout'])(x)
        x = layers.Conv2DTranspose(64, self.params['kernel_size'], activation="relu", strides=self.params['num_strides'], padding="same")(x)
        x = layers.Dropout(self.params['convolutional_dropout'])(x)
        x = layers.Conv2DTranspose(64, self.params['kernel_size'], activation="relu", strides=self.params['num_strides'], padding="same")(x)
        x = layers.Dropout(self.params['convolutional_dropout'])(x)
        decoder_outputs = layers.Conv2DTranspose(self.params['num_channels'], self.params['kernel_size'], activation="relu", padding="same")(x)
        decoder = keras.Model([latent_inputs, label_inputs], decoder_outputs, name="decoder")
        return decoder

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    @tf.function(jit_compile=True)
    def call(self, inputs):
        _, _, z = self.encoder(inputs)
        reconstruction = self.decoder([z, inputs[1]])
        return reconstruction

    @tf.function
    def train_step(self, images_and_labels):
        with tf.GradientTape() as tape:
            total_loss, reconstruction_loss, kl_loss = self.compute_loss(images_and_labels)
        grads = tape.gradient(total_loss, self.trainable_weights)

        self.optimizer.apply_gradients((grad, weights) 
            for (grad, weights) in zip(grads, self.trainable_weights)
            if grad is not None)

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    @tf.function(jit_compile=True)
    def compute_loss(self, images_and_labels):
        z_mean, z_log_var, z = self.encoder(images_and_labels)
        reconstruction_loss = self.compute_reconstruction_loss(images_and_labels, z)
        kl_loss = self.kl_loss_function(images_and_labels, z_mean, z_log_var)
        total_loss = reconstruction_loss + self.params['beta'] * kl_loss
        return total_loss, reconstruction_loss, kl_loss

    @tf.function(jit_compile=True)
    def compute_reconstruction_loss(self, images_and_labels, z):
        reconstruction = self.decoder([z, images_and_labels[1]])
        reconstruction_loss = tf.reduce_mean(tf.reduce_sum(
            self.mse(images_and_labels[0], reconstruction), axis=(1,2)
        ))
        return reconstruction_loss

    @tf.function(jit_compile=True)
    def kl_loss_normal(self, z_mean, z_log_var):
        return -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))

    @tf.function(jit_compile=True)
    def kl_loss_general(self, mean_0, log_var_0, mean_1, log_var_1):
        return -0.5 * (1 + log_var_0 - log_var_1 - tf.square(mean_1-mean_0)/tf.exp(log_var_1) - tf.exp(log_var_0)/tf.exp(log_var_1))

    @tf.function(jit_compile=True)
    def kl_conceptual_fun(self, images_and_labels, z_mean, z_log_var):
        concept_mean, concept_log_var = self.concept_gaussians(images_and_labels[1])
        kl_loss = 0
        for i in range(z_mean.shape[1]):
            if i < len(enc.concept_domains):
                kl_loss = kl_loss + self.kl_loss_general(z_mean[:,i], z_log_var[:,i], concept_mean[:,i], concept_log_var[:,i])
                kl_loss = kl_loss + self.params['unit_normal_regularization_factor'] * \
                                    self.kl_loss_normal(z_mean[:,i], z_log_var[:,i])
            else:
                kl_loss = kl_loss + self.kl_loss_normal(z_mean[:,i], z_log_var[:,i])
        kl_loss = tf.reduce_mean(kl_loss)
        return kl_loss

    @tf.function(jit_compile=True)
    def kl_conditional_fun(self, images_and_labels, z_mean, z_log_var):
        kl_loss = self.kl_loss_normal(z_mean, z_log_var)
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        return kl_loss
