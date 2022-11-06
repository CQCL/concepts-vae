import cirq
import numpy as np
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq
from tensorflow import keras

from quantum.circuit_creation import entangling_layer, one_qubit_rotation, one_qubit_rotation_rev
from quantum.utils import create_zeros_measurement_operator


class ConceptLearner(keras.Model):
    def __init__(self, qoncepts, concept_domains, num_concept_pqc_layers=None, mixed=False, **kwargs):
        super().__init__(**kwargs)
        self.qoncepts = qoncepts
        self.concept_domains = concept_domains
        if num_concept_pqc_layers is None:
            self.num_concept_pqc_layers = self.qoncepts.params['num_concept_pqc_layers']
        else:
            self.num_concept_pqc_layers = num_concept_pqc_layers
        self.mixed = mixed
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.concept_pqc, self.measurement_operator, self.concept_params = self.define_concept_pqc()
        self.concept_learner_model = self.define_concept_learner_model()
        self.concept_learner_model.summary()
        

    def define_concept_pqc(self):
        """
        Defines the concept pqc layers.
        """
        qubits_per_domain = self.qoncepts.params['num_qubits_per_domain']
        num_concept_symbols = 3 * qubits_per_domain * len(self.concept_domains) * self.num_concept_pqc_layers
        if self.mixed:
            num_concept_symbols = num_concept_symbols * 2
        concept_params = sympy.symbols(['y{0:03d}'.format(i) for i in range(num_concept_symbols)])
        concept_qubits = []
        for domain in self.concept_domains:
            offset = domain * qubits_per_domain
            qubits = self.qoncepts.qubits[offset:offset + qubits_per_domain]
            concept_qubits.extend(qubits)
        if self.mixed:
            mixture_qubits = [cirq.GridQubit(len(self.qoncepts.qubits) + i, 0)\
                for i in range(len(concept_qubits))]
        else:
            mixture_qubits = []
        all_concept_qubits = concept_qubits.copy()
        all_concept_qubits.extend(mixture_qubits)
        pqc = cirq.Circuit()
        for layer in range(self.num_concept_pqc_layers):
            for i, qubit in enumerate(all_concept_qubits):
                offset = (i * self.num_concept_pqc_layers + layer) * 3
                pqc += one_qubit_rotation_rev(qubit, concept_params[offset:offset+3])
            pqc += entangling_layer(all_concept_qubits)
        if self.mixed:
            measurement_operator = create_zeros_measurement_operator(mixture_qubits)
        else:
            measurement_operator = create_zeros_measurement_operator(concept_qubits)
        concept_params_weights = self.add_weight(
            name="concept_params",
            shape=(1, num_concept_symbols),
            trainable=True,
            initializer=keras.initializers.RandomUniform(minval=0., maxval=2 * np.pi)
        )
        return pqc, measurement_operator, concept_params_weights

    def define_concept_learner_model(self):
        """
        Defines the concept learner model.
        """
        image_input = keras.Input(shape=self.qoncepts.params['input_shape'][0])
        self.qoncepts.trainable = False
        for layer in self.qoncepts.encoder_cnn.layers:
            layer.trainable = False
        pqc = self.qoncepts.encoder_pqc + self.concept_pqc
        print(pqc)
        controlled_pqc = tfq.layers.ControlledPQC(
            pqc,
            operators=self.measurement_operator
        )
        # input 0 state to the pqc
        circuits_input =  tfq.convert_to_tensor([cirq.Circuit()])
        # repeat the input for the number of samples in batch
        circuits_input = tf.repeat(circuits_input, tf.shape(image_input)[0], axis=0)
        concept_params_batch = tf.repeat(self.concept_params, tf.shape(image_input)[0], axis=0)
        encoder_cnn_output = self.qoncepts.encoder_cnn(image_input)
        control_params = tf.concat([encoder_cnn_output, concept_params_batch], axis=1)
        expectation = controlled_pqc([circuits_input, control_params])
        # scale the expectation by a learned parameter
        self.scale_factor = self.add_weight(name="expectation_scaling_factor",
            shape=(1,),
            initializer=tf.keras.initializers.Ones(),
            trainable=True
        )
        scaled_expectation = self.scale_factor * expectation
        model = keras.Model(inputs=([image_input]), outputs=scaled_expectation)
        return model

    @property
    def metrics(self):
        return [
            self.loss_tracker,
        ]

    @tf.function
    def call(self, images):
        return self.concept_learner_model(images)

    @tf.function
    def train_step(self, images_and_classifications):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(images_and_classifications)
        grads = tape.gradient(loss, self.trainable_weights)

        self.optimizer.apply_gradients((grad, weights)
            for (grad, weights) in zip(grads, self.trainable_weights)
            if grad is not None)

        self.loss_tracker.update_state(loss)
        return {
            "loss": self.loss_tracker.result(),
        }

    @tf.function
    def compute_loss(self, images_and_true_classifications):
        images, true_classifications = images_and_true_classifications
        expectation = tf.squeeze(self.call(images))
        loss = tf.reduce_mean(tf.math.square(true_classifications - expectation))
        return loss

    def save_model(self, file_name):
        self.save_weights(file_name + '.h5')
