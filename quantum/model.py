import tensorflow as tf
from tensorflow import keras
import tensorflow_quantum as tfq

import cirq
import sympy
import vae.encoding_dictionary as enc


class Qoncepts(keras.Model):
    def __init__(self, params, **kwargs):
        super(Qoncepts, self).__init__(**kwargs)
        self.params = params
        self.model = self.define_model()
        self.model.summary()
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.concept_pqcs = ConceptPQCs()
        self.mse = keras.losses.MeanSquaredError(reduction='none')

    def get_config(self):
        # returns parameters with which the model was instanciated
        return self.params

    def define_model(self):
        image_input = keras.Input(shape=self.params['image_input_shape'])
        concept_PQCs_params = keras.Input(shape=(self.params['num_domains']*3))
        # create CNN layers
        encoder_cnn = image_input
        for _ in range(self.params['num_layers']):
            encoder_cnn = keras.layers.Conv2D(64, self.params['kernel_size'], activation="relu", 
                strides=self.params['num_strides'], padding="same")(encoder_cnn)
        encoder_cnn = keras.layers.Flatten()(encoder_cnn)
        encoder_cnn = keras.layers.Dense(256, activation="relu")(encoder_cnn)
        encoder_cnn = keras.layers.Dense(self.params['num_domains']*3, activation="relu")(encoder_cnn)
        # create pqc layers
        # one qubit for each domain
        qubits = [cirq.GridQubit(i, 0) for i in range(self.params['num_domains'])]
        # Parameters that the classical NN will feed values into. Three parameters for each domain (qubit).
        control_params = sympy.symbols(['x{0:03d}'.format(i) for i in range(self.params['num_domains']*3)])
        pqc = cirq.Circuit([
            (
                cirq.rx(control_params[i*3])(qubits[i]), 
                cirq.ry(control_params[i*3+1])(qubits[i]),
                cirq.rz(control_params[i*3+2])(qubits[i])
            )
            for i in range(self.params['num_domains'])
        ])
        concept_params = sympy.symbols(['y{0:03d}'.format(i) for i in range(self.params['num_domains']*3)])
        concept_pqc = cirq.Circuit([
            (
                cirq.rz(concept_params[i*3])(qubits[i]), 
                cirq.ry(concept_params[i*3+1])(qubits[i]),
                cirq.rx(concept_params[i*3+2])(qubits[i])
            ) for i in range(self.params['num_domains'])
        ])
        pqc.append(concept_pqc)
        measurement_operators = [cirq.Z(qubits[i]) for i in range(self.params['num_domains'])]
        # TFQ layer for classically controlled circuits.
        controlled_pqc = tfq.layers.ControlledPQC(
            pqc,
            operators=measurement_operators
        )
        print(pqc)
        # input 0 state to the pqc
        circuits_input =  tfq.convert_to_tensor([cirq.Circuit()])
        # repeat the input for the number of samples in batch
        circuits_input = tf.repeat(circuits_input, tf.shape(image_input)[0], axis=0)
        pqc_all_params = tf.concat([encoder_cnn, concept_PQCs_params], axis=1)
        expectation = controlled_pqc([circuits_input, pqc_all_params]) 
        # The full Keras model is built from our layers.
        model = keras.Model(inputs=([image_input, concept_PQCs_params]), outputs=expectation)
        return model

    @property
    def metrics(self):
        return [
            self.loss_tracker,
        ]

    @tf.function
    def call(self, images_and_labels):
        concept_PQCs_params = self.concept_pqcs(images_and_labels[1])
        #flatten the tensor
        concept_PQCs_params = tf.reshape(concept_PQCs_params, (-1, self.params['num_domains']*3))
        return self.model([images_and_labels[0], concept_PQCs_params])

    # @tf.function
    def train_step(self, images_and_labels):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(images_and_labels)
        grads = tape.gradient(loss, self.trainable_weights)

        self.optimizer.apply_gradients((grad, weights)
            for (grad, weights) in zip(grads, self.trainable_weights)
            if grad is not None)

        self.loss_tracker.update_state(loss)
        return {
            "loss": self.loss_tracker.result(),
        }

    # @tf.function
    def compute_loss(self, images_and_labels):
        expectation = self.call(images_and_labels)
        loss = tf.reduce_mean(tf.reduce_sum(tf.math.square(expectation), axis=1))
        return loss


class ConceptPQCs(keras.layers.Layer):
    def build(self, input_shape):
        # the max number of different values for different domains
        max_concepts = max([len(enc.enc_dict[concept]) 
            for concept in enc.concept_domains]) - 1 # -1 because of the 'ANY' concept
                                                     #TODO: fix this hacky way of dealing with ANY
        # Create a trainable weight variable for this layer.
        self.pqc_params = self.add_weight(
            name="pqc_params",
            shape=(len(enc.concept_domains), max_concepts, 3),
            trainable=True
        )
        super(ConceptPQCs, self).build(input_shape)

    @tf.function(jit_compile=True)
    def call(self, labels, **kwargs):
        labels = tf.cast(labels, tf.int32)  # casting from float to int
        # gather_nd gathers slices from self.mean into a Tensor with shape specified by indices
        # (i.e., the needed indices corresponding to current concepts based on labels)
        # (cf., https://www.tensorflow.org/api_docs/python/tf/gather_nd)
        indices = tf.reshape(tf.transpose(labels), (tf.shape(labels)[1], tf.shape(labels)[0],1))
        pqc_params = tf.transpose(tf.gather_nd(self.pqc_params, indices, batch_dims=1), perm=[1,0,2])
        return pqc_params