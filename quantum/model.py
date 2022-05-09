import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow_probability import distributions as tfd
import tensorflow_quantum as tfq

import cirq
import sympy
from quantum.circuit_creation import entangling_layer, one_qubit_rotation
import vae.encoding_dictionary as enc


class Qoncepts(keras.Model):
    def __init__(self, params, **kwargs):
        super(Qoncepts, self).__init__(**kwargs)
        self.params = params
        self.all_qubits, self.qubits = self.initialize_qubits()
        self.model = self.define_model()
        self.model.summary()
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.all_labels = tf.expand_dims(tf.repeat(
            tf.expand_dims(tf.range(self.concept_params.max_concepts, dtype=tf.float32), axis=0), 
            self.params['num_domains'], 
            axis=0
        ), axis=0)

    def initialize_qubits(self):
        if self.params['mixed_states']:
            num_total_qubits = 2 * self.params['num_domains'] * self.params['num_qubits_per_domain']
            all_qubits = [cirq.GridQubit(i, 0) for i in range(num_total_qubits)]
            qubits = all_qubits[::2] # half normal qubits, half mixture qubits
        else:
            num_total_qubits = self.params['num_domains'] * self.params['num_qubits_per_domain']
            all_qubits = [cirq.GridQubit(i, 0) for i in range(num_total_qubits)]
            qubits = all_qubits
        return all_qubits, qubits

    def define_model(self):
        image_input = keras.Input(shape=self.params['input_shape'][0])
        label_input = keras.Input(shape=self.params['input_shape'][1])
        self.concept_pqc, self.measurement_operators, num_concept_symbols = self.define_concept_pqc()
        self.encoder_pqc, self.num_encoder_symbols = self.define_encoder_pqc()
        self.encoder_cnn = self.define_encoder_cnn(self.num_encoder_symbols)
        self.concept_params = ConceptParameters(num_concept_symbols // self.params['num_domains'])
        con_params = self.concept_params(label_input)
        con_params = tf.reshape(con_params, (-1, con_params.shape[1]*con_params.shape[2]))  #flatten the tensor
        pqc = self.encoder_pqc + self.concept_pqc
        controlled_pqc = tfq.layers.ControlledPQC(
            pqc,
            operators=self.measurement_operators
        )
        # input 0 state to the pqc
        circuits_input =  tfq.convert_to_tensor([cirq.Circuit()])
        # repeat the input for the number of samples in batch
        circuits_input = tf.repeat(circuits_input, tf.shape(image_input)[0], axis=0)
        encoder_cnn_output = self.encoder_cnn(image_input)
        expectation = controlled_pqc([circuits_input, tf.concat([encoder_cnn_output, con_params], axis=1)])
        model = keras.Model(inputs=([image_input, label_input]), outputs=expectation)
        return model

    def define_concepts(self):
        circuits_input = keras.Input(shape=(),dtype=tf.string)
        pqc, measurement_operators, num_concept_symbols = self.define_concept_pqc()
        concept_PQCs_params = keras.Input(shape=(num_concept_symbols,))
        controlled_pqc = tfq.layers.ControlledPQC(
            pqc,
            operators=measurement_operators
        )
        concepts_expectation = controlled_pqc([circuits_input, concept_PQCs_params])
        concepts = keras.Model(
            inputs=[circuits_input, concept_PQCs_params],
            outputs=concepts_expectation,
            name='concepts'
        )
        return concepts, num_concept_symbols

    def define_encoder_pqc(self):
        # Parameters that the classical NN will feed values into
        num_encoder_symbols = 3 * len(self.qubits) * self.params['num_encoder_pqc_layers']
        control_params = sympy.symbols(['x{0:03d}'.format(i) for i in range(num_encoder_symbols)])
        # Define circuit
        pqc = cirq.Circuit()
        for layer in range(self.params['num_encoder_pqc_layers']):
            for i, qubit in enumerate(self.qubits):
                offset = (i * self.params['num_encoder_pqc_layers'] + layer) * 3
                pqc += one_qubit_rotation(qubit, control_params[offset:offset+3])
            for i in range(self.params['num_domains']):
                offset = i*self.params['num_qubits_per_domain']
                pqc += entangling_layer(self.qubits[offset:offset+self.params['num_qubits_per_domain']])
        return pqc, num_encoder_symbols

    def define_encoder_cnn(self, num_outputs):
        image_input = keras.Input(shape=self.params['input_shape'][0])
        encoder_cnn = image_input
        for _ in range(self.params['num_cnn_layers']):
            encoder_cnn = keras.layers.Conv2D(
                64,
                self.params['kernel_size'],
                activation="relu", 
                strides=self.params['num_strides'],
                padding="same"
            )(encoder_cnn)
        encoder_cnn = keras.layers.Flatten()(encoder_cnn)
        encoder_cnn = keras.layers.Dense(256, activation="relu")(encoder_cnn)
        encoder_cnn = keras.layers.Dense(num_outputs, activation="relu")(encoder_cnn)
        encoder_cnn_model = keras.Model(inputs=image_input, outputs=encoder_cnn)
        return encoder_cnn_model

    def define_concept_pqc(self):
        pqc = cirq.Circuit()
        num_concept_symbols = 3 * len(self.all_qubits) * self.params['num_concept_pqc_layers']
        concept_params = sympy.symbols(['y{0:03d}'.format(i) for i in range(num_concept_symbols)])
        for layer in range(self.params['num_concept_pqc_layers']):
            for i, qubit in enumerate(self.all_qubits):
                offset = (i * self.params['num_concept_pqc_layers'] + layer) * 3
                pqc += one_qubit_rotation(qubit, concept_params[offset:offset+3])
            for i in range(self.params['num_domains']):
                if self.params['mixed_states']:
                    start = 2 * i * self.params['num_qubits_per_domain']
                    end = start + 2 * self.params['num_qubits_per_domain']
                else:
                    start = i * self.params['num_qubits_per_domain']
                    end = start + self.params['num_qubits_per_domain']
                pqc += entangling_layer(self.all_qubits[start:end])
        measurement_operators = [cirq.Z(self.qubits[i]) for i in range(len(self.qubits))]
        return pqc, measurement_operators, num_concept_symbols

    @property
    def metrics(self):
        return [
            self.loss_tracker,
        ]

    @tf.function
    def call(self, images_and_labels):
        return self.model(images_and_labels)

    @tf.function
    def train_step(self, images_and_labels):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(images_and_labels)
        grads = tape.gradient(loss, self.trainable_weights)
        tf.print(self.concept_params.trainable_weights)

        self.optimizer.apply_gradients((grad, weights)
            for (grad, weights) in zip(grads, self.trainable_weights)
            if grad is not None)

        self.loss_tracker.update_state(loss)
        return {
            "loss": self.loss_tracker.result(),
        }

    @tf.function
    def compute_loss(self, images_and_labels):
        # positive samples
        pos_expectation = self.call(images_and_labels)
        loss = tf.reduce_mean(tf.reduce_sum(tf.math.square(1 - pos_expectation), axis=1))
        # negative samples
        all_labels = tf.repeat(self.all_labels, tf.shape(images_and_labels[1])[0], axis=0)
        cur_labels = tf.repeat(
            tf.expand_dims(images_and_labels[1], axis=2),
            self.concept_params.max_concepts,
            axis=2
        )
        dist = tfd.Categorical(probs=tf.cast(all_labels != cur_labels, tf.float32))
        samples = dist.sample()
        neg_expectation = self.call([images_and_labels[0], samples])
        loss += tf.reduce_mean(tf.reduce_sum(tf.math.square(-1 - neg_expectation), axis=1))
        return loss
    
    def get_config(self):
        # returns parameters with which the model was instanciated
        return self.params

    def save_model(self, file_name):
        self.save_weights(file_name + '.h5')
        with open(file_name + '_params.json', 'w') as f:
            json.dump(self.params, f)


class ConceptParameters(keras.layers.Layer):
    def __init__(self, params_per_concept, **kwargs):
        super(ConceptParameters, self).__init__(**kwargs)
        self.params_per_concept = params_per_concept
        # the max number of different values for different domains
        self.max_concepts = max([len(enc.enc_dict[concept]) 
            for concept in enc.concept_domains]) - 1 # -1 because of the 'ANY' concept
                                                     #TODO: fix this hacky way of dealing with ANY

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.concept_params = self.add_weight(
            name="concept_params",
            shape=(len(enc.concept_domains), self.max_concepts, self.params_per_concept),
            trainable=True,
            initializer=tf.keras.initializers.RandomUniform(0, 2*np.pi)
        )
        super(ConceptParameters, self).build(input_shape)

    @tf.function(jit_compile=True)
    def call(self, labels, **kwargs):
        labels = tf.cast(labels, tf.int32)  # casting from float to int
        # gather_nd gathers slices from self.mean into a Tensor with shape specified by indices
        # (i.e., the needed indices corresponding to current concepts based on labels)
        # (cf., https://www.tensorflow.org/api_docs/python/tf/gather_nd)
        indices = tf.reshape(tf.transpose(labels), (tf.shape(labels)[1], tf.shape(labels)[0],1))
        concept_params = tf.transpose(tf.gather_nd(self.concept_params, indices, batch_dims=1), perm=[1,0,2])
        return concept_params

