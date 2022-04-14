import json
import tensorflow as tf
from tensorflow import keras
from tensorflow_probability import distributions as tfd
import tensorflow_quantum as tfq

import cirq
import sympy
import vae.encoding_dictionary as enc
from vae.data_generator import get_tf_dataset


def one_qubit_rotation(qubit, symbols):
    """
    Returns Cirq gates that apply a rotation of the bloch sphere about the X,
    Y and Z axis, specified by the values in `symbols`.
    """
    return [cirq.rx(symbols[0])(qubit),
            cirq.ry(symbols[1])(qubit),
            cirq.rz(symbols[2])(qubit)]

def entangling_layer(qubits):
    """
    Returns a layer of CZ entangling gates on `qubits` (arranged in a circular topology).
    """
    if len(qubits)==1:
        return cirq.Circuit()
    cz_ops = [cirq.CZ(q0, q1) for q0, q1 in zip(qubits, qubits[1:])]
    cz_ops += ([cirq.CZ(qubits[0], qubits[-1])] if len(qubits) != 2 else [])
    return cz_ops

def load_saved_model(file_name, image_dir='images/basic_train'):
    """
    Loads a saved model from the file `file_name`.
    """
    with open(file_name + '_params.json', 'r') as f:
        params = json.load(f)
    dataset_tf = get_tf_dataset(image_dir, 1, return_image_shape=False)
    qoncepts = Qoncepts(params)
    qoncepts.compile()
    sample_input = list(dataset_tf.take(1).as_numpy_iterator())[0]
    qoncepts(sample_input)
    qoncepts.load_weights(file_name + '.h5')
    return qoncepts

class Qoncepts(keras.Model):
    def __init__(self, params, **kwargs):
        super(Qoncepts, self).__init__(**kwargs)
        self.params = params
        self.model = self.define_model()
        self.model.summary()
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mse = keras.losses.MeanSquaredError(reduction='none')
        self.all_labels = tf.expand_dims(tf.repeat(
            tf.expand_dims(tf.range(self.concept_pqcs.max_concepts, dtype=tf.float32), axis=0), 
            self.params['num_domains'], 
            axis=0
        ), axis=0)

    def get_config(self):
        # returns parameters with which the model was instanciated
        return self.params

    def save_model(self, file_name):
        self.save_weights(file_name + '.h5')
        with open(file_name + '_params.json', 'w') as f:
            json.dump(self.params, f)

    def define_model(self):
        # create pqc layers
        controlled_pqc, num_encoder_symbols, num_concept_symbols = self.define_pqc()
        self.concept_pqcs = ConceptPQCs(num_concept_symbols // self.params['num_domains'])
        concept_PQCs_params = keras.Input(shape=(num_concept_symbols,))
        image_input = keras.Input(shape=self.params['image_input_shape'])
        # create CNN layers
        encoder_cnn = self.define_cnn(image_input, num_encoder_symbols)
        # input 0 state to the pqc
        circuits_input =  tfq.convert_to_tensor([cirq.Circuit()])
        # repeat the input for the number of samples in batch
        circuits_input = tf.repeat(circuits_input, tf.shape(image_input)[0], axis=0)
        pqc_all_params = tf.concat([encoder_cnn, concept_PQCs_params], axis=1)
        expectation = controlled_pqc([circuits_input, pqc_all_params]) 
        # The full Keras model is built from our layers.
        model = keras.Model(inputs=([image_input, concept_PQCs_params]), outputs=expectation)
        return model

    def define_pqc(self):
        if self.params['mixed_states']:
            num_total_qubits = 2 * self.params['num_domains'] * self.params['num_qubits_per_domain']
            all_qubits = [cirq.GridQubit(i, 0) for i in range(num_total_qubits)]
            qubits = all_qubits[::2] # half normal qubits, half mixture qubits
        else:
            num_total_qubits = self.params['num_domains'] * self.params['num_qubits_per_domain']
            all_qubits = [cirq.GridQubit(i, 0) for i in range(num_total_qubits)]
            qubits = all_qubits
        # Parameters that the classical NN will feed values into
        num_encoder_symbols = 3 * len(qubits) * self.params['num_encoder_pqc_layers']
        control_params = sympy.symbols(['x{0:03d}'.format(i) for i in range(num_encoder_symbols)])
        # Define circuit
        pqc = cirq.Circuit()
        for layer in range(self.params['num_encoder_pqc_layers']):
            for i, q in enumerate(qubits):
                offset = (i * self.params['num_encoder_pqc_layers'] + layer) * 3
                pqc += one_qubit_rotation(q, control_params[offset:offset+3])
            for i in range(self.params['num_domains']):
                offset = i*self.params['num_qubits_per_domain']
                pqc += entangling_layer(qubits[offset:offset+self.params['num_qubits_per_domain']])
        num_concept_symbols = 3 * len(all_qubits) * self.params['num_concept_pqc_layers']
        concept_params = sympy.symbols(['y{0:03d}'.format(i) for i in range(num_concept_symbols)])
        for layer in range(self.params['num_concept_pqc_layers']):
            for i, q in enumerate(all_qubits):
                offset = (i * self.params['num_concept_pqc_layers'] + layer) * 3
                pqc += one_qubit_rotation(q, concept_params[offset:offset+3])
            for i in range(self.params['num_domains']):
                if self.params['mixed_states']:
                    start = 2 * i * self.params['num_qubits_per_domain']
                    end = start + 2 * self.params['num_qubits_per_domain']
                else:
                    start = i * self.params['num_qubits_per_domain']
                    end = start + self.params['num_qubits_per_domain']
                pqc += entangling_layer(all_qubits[start:end])
        measurement_operators = [cirq.Z(qubits[i]) for i in range(len(qubits))]
        # TFQ layer for classically controlled circuits.
        controlled_pqc = tfq.layers.ControlledPQC(
            pqc,
            operators=measurement_operators
        )
        print(pqc)
        return controlled_pqc, num_encoder_symbols, num_concept_symbols

    def define_cnn(self, image_input, num_outputs):
        encoder_cnn = image_input
        for _ in range(self.params['num_layers']):
            encoder_cnn = keras.layers.Conv2D(64, self.params['kernel_size'], activation="relu", 
                strides=self.params['num_strides'], padding="same")(encoder_cnn)
        encoder_cnn = keras.layers.Flatten()(encoder_cnn)
        encoder_cnn = keras.layers.Dense(256, activation="relu")(encoder_cnn)
        encoder_cnn = keras.layers.Dense(num_outputs, activation="relu")(encoder_cnn)
        return encoder_cnn

    @property
    def metrics(self):
        return [
            self.loss_tracker,
        ]

    @tf.function
    def call(self, images_and_labels):
        concept_PQCs_params = self.concept_pqcs(images_and_labels[1])
        #flatten the tensor
        concept_PQCs_params = tf.reshape(
            concept_PQCs_params, 
            (-1, concept_PQCs_params.shape[1]*concept_PQCs_params.shape[2])
        )
        return self.model([images_and_labels[0], concept_PQCs_params])

    @tf.function
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

    @tf.function
    def compute_loss(self, images_and_labels):
        # positive samples
        pos_expectation = self.call(images_and_labels)
        loss = tf.reduce_mean(tf.reduce_sum(tf.math.square(1 - pos_expectation), axis=1))
        # negative samples
        all_labels = tf.repeat(self.all_labels, tf.shape(images_and_labels[1])[0], axis=0)
        cur_labels = tf.repeat(
            tf.expand_dims(images_and_labels[1], axis=2),
            self.concept_pqcs.max_concepts,
            axis=2
        )
        dist = tfd.Categorical(probs=tf.cast(all_labels != cur_labels, tf.float32))
        samples = dist.sample()
        neg_expectation = self.call([images_and_labels[0], samples])
        loss += tf.reduce_mean(tf.reduce_sum(tf.math.square(-1 - neg_expectation), axis=1))
        return loss


class ConceptPQCs(keras.layers.Layer):
    def __init__(self, params_per_concept, **kwargs):
        super(ConceptPQCs, self).__init__(**kwargs)
        self.params_per_concept = params_per_concept
        # the max number of different values for different domains
        self.max_concepts = max([len(enc.enc_dict[concept]) 
            for concept in enc.concept_domains]) - 1 # -1 because of the 'ANY' concept
                                                     #TODO: fix this hacky way of dealing with ANY

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.pqc_params = self.add_weight(
            name="pqc_params",
            shape=(len(enc.concept_domains), self.max_concepts, self.params_per_concept),
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