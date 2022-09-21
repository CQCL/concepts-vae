import tensorflow as tf

from quantum.model import Qoncepts


class Qoncepts_hierarchy(Qoncepts):
    @tf.function
    def compute_loss(self, images_and_labels):
        pos_images  = images_and_labels[0]
        neg_images  = images_and_labels[1]
        labels = images_and_labels[2]
        pos_expectation = self.call([pos_images, labels])
        neg_expectation = self.call([neg_images, labels])
        loss = tf.reduce_sum(tf.math.square(1 - pos_expectation), axis=1)
        loss = loss + tf.reduce_sum(tf.math.square(0 - neg_expectation), axis=1)
        return tf.reduce_mean(loss)
