import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import vae.encoding_dictionary as enc


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class ConceptGaussians(layers.Layer):

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        max_concepts = max([len(enc.enc_dict[concept]) for concept in enc.concept_domains])
        mean_initializer = tf.keras.initializers.RandomUniform(minval=-10., maxval=10.)
        log_var_initializer = tf.keras.initializers.RandomUniform(minval=-7., maxval=0.5)
        self.mean = self.add_weight(name='kernel',
                                      shape=(input_shape[1], max_concepts),
                                      initializer=mean_initializer,
                                      trainable=True)
        self.log_var = self.add_weight(name='kernel',
                                      shape=(input_shape[1], max_concepts),
                                      initializer=log_var_initializer,
                                      trainable=True)
        super(ConceptGaussians, self).build(input_shape)

    def call(self, labels, **kwargs):
        required_means = np.zeros(labels.shape, dtype=np.float32)
        required_log_vars = np.zeros(labels.shape, dtype=np.float32)
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                required_means[i][j] = self.mean[j][int(labels[i][j])]
                required_log_vars[i][j] = self.log_var[j][int(labels[i][j])]
        return tf.convert_to_tensor(required_means), tf.convert_to_tensor(required_log_vars)


class VAE(keras.Model):
    def __init__(self, params, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.params = params
        self.encoder, conv_shape = self.encoder_model()
        self.decoder = self.decoder_model(conv_shape)
        self.concept_gaussians = ConceptGaussians()
        self.encoder.summary()
        self.decoder.summary()
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        # self.reconstruction_loss_function = tf.keras.losses.binary_crossentropy
        self.reconstruction_loss_function = tf.keras.losses.MeanSquaredError(reduction='none')

    def encoder_model(self):
        encoder_inputs = keras.Input(shape=self.params['input_shape'])
        x = layers.Conv2D(64, self.params['kernel_size'], activation="relu", strides=self.params['num_strides'], padding="same")(encoder_inputs)
        # x = layers.MaxPooling2D(pool_size=self.params['pool_size'])(x)
        x = layers.Conv2D(64, self.params['kernel_size'], activation="relu", strides=self.params['num_strides'], padding="same")(x)
        x = layers.Conv2D(64, self.params['kernel_size'], activation="relu", strides=self.params['num_strides'], padding="same")(x)
        x = layers.Conv2D(64, self.params['kernel_size'], activation="relu", strides=self.params['num_strides'], padding="same")(x)
        # x = layers.MaxPooling2D(pool_size=self.params['pool_size'])(x)
        conv_shape = tf.keras.backend.int_shape(x) #Shape of conv to be provided to decoder
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation="relu")(x)
        z_mean = layers.Dense(self.params['latent_dim'], name="z_mean")(x)
        z_log_var = layers.Dense(self.params['latent_dim'], name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        return encoder, conv_shape

    def decoder_model(self, conv_shape):
        latent_inputs = keras.Input(shape=(self.params['latent_dim'],))
        x = layers.Dense(256, activation="relu")(latent_inputs)
        x = layers.Dense(conv_shape[1]*conv_shape[2]*conv_shape[3], activation="relu")(x)
        x = layers.Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(x)
        x = layers.Conv2DTranspose(64, self.params['kernel_size'], activation="relu", strides=self.params['num_strides'], padding="same")(x)
        x = layers.Conv2DTranspose(64, self.params['kernel_size'], activation="relu", strides=self.params['num_strides'], padding="same")(x)
        x = layers.Conv2DTranspose(64, self.params['kernel_size'], activation="relu", strides=self.params['num_strides'], padding="same")(x)
        x = layers.Conv2DTranspose(64, self.params['kernel_size'], activation="relu", strides=self.params['num_strides'], padding="same")(x)
        # x = layers.UpSampling2D(size=self.params['pool_size'])(x)
        # x = layers.Conv2DTranspose(64, self.params['kernel_size'], activation="relu", strides=self.params['num_strides'], padding="same")(x)
        # x = layers.UpSampling2D(size=self.params['pool_size'])(x)
        decoder_outputs = layers.Conv2DTranspose(self.params['num_channels'], self.params['kernel_size'], activation="relu", padding="same")(x)
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        return decoder

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs):
        _, _, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction

    def train_step(self, data):
        images, labels = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(images)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    self.reconstruction_loss_function(images, reconstruction), axis=(1,2)
                )
            )
            concept_mean, concept_log_var = self.concept_gaussians(labels)
            kl_loss = 0
            for i in range(z_mean.shape[1]):
                if i < concept_mean.shape[1]:
                    kl_loss = kl_loss + self.kl_loss_general(z_mean[:,i], z_log_var[:,i], concept_mean[:,i], concept_log_var[:,i])
                else:
                    kl_loss = kl_loss + self.kl_loss_normal(z_mean[:,i], z_log_var[:,i])
            kl_loss = tf.reduce_mean(kl_loss)
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


    def kl_loss_normal(self, z_mean, z_log_var):
        return -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))

    def kl_loss_general(self, mean_0, log_var_0, mean_1, log_var_1):
        return -0.5 * (1 + log_var_0 - log_var_1 - tf.square(mean_1-mean_0)/tf.exp(log_var_1) - tf.exp(log_var_0)/tf.exp(log_var_1))
