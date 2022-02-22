
import os
import subprocess

import numpy as np
import scipy.stats as stats
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, silhouette_score

from vae import encoding_dictionary as enc
from vae import utils
from vae.data_generator import ImageGenerator


class ImageSaveCallback(tf.keras.callbacks.Callback):
    def __init__(self, image_and_label, folder_name, **kwargs):
        super(ImageSaveCallback, self).__init__(**kwargs)
        self.image_and_label = image_and_label
        self.epoch_number = 0
        self.folder_name = folder_name
        self.img_frames = []
        utils.save_image(self.folder_name, 'original_image', 
            np.array([self.image_and_label[0][0]]), timestamp_in_name=False)

    def on_epoch_end(self, epoch, logs=None):
        reconstructed = self.model.call(self.image_and_label)
        file_name = 'reconstructed_epoch_' + str(epoch)
        utils.save_image(self.folder_name, file_name, 
            np.array([reconstructed[0]]), timestamp_in_name=False)


class GaussianPlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, folder_name, **kwargs):
        super(GaussianPlotCallback, self).__init__(**kwargs)
        self.epoch_number = 0
        self.folder_name = folder_name
        self.concept_names = [['blue', 'red', 'green'],
                              ['small', 'medium', 'large'],
                              ['circle', 'square', 'triangle'],
                              ['top', 'centre', 'bottom']]
        self.concept_encoding = utils.encode_or_decode(self.concept_names)
    
    def plot_gaussians(self, file_name, concept_names, epoch, means, log_vars):
        sigmas = tf.sqrt(tf.exp(log_vars))
        plt.gca().set_ylim([0, 2])
        x = np.linspace(-3, 3, 100)
        for i in range(len(means)):
            plt.plot(x, stats.norm.pdf(x, means[i], sigmas[i]), label='concept ' + concept_names[i])
        plt.legend()
        plt.title('Epoch ' + str(epoch))
        plt.savefig(file_name)
        plt.close()

    def on_epoch_end(self, epoch, logs=None):
        means, log_vars = utils.get_concept_gaussians(self.concept_encoding, self.model)
        for i in range(4):
            file_name = os.path.join(self.folder_name, 'gaussian_epoch_' + str(epoch) + '_dim_' + str(i) + '.png')
            self.plot_gaussians(file_name, self.concept_names[i], epoch, means[i], log_vars[i])
        
    def save_video_from_images(self, video_name):
        for i in range(4):
            subprocess.run(['ffmpeg', 
                            '-framerate',
                            '1',
                            '-i',
                            os.path.join(self.folder_name, 'gaussian_epoch_' + '%01d' + '_dim_' + str(i) + '.png'),
                            '-pix_fmt',
                            'yuv420p',
                            os.path.join(self.folder_name, video_name + '_dim_' + str(i) + '.mp4')
            ])
        # ffmpeg -framerate 1 -i images/training/gaussian_epoch_%01d_dim_0.png -pix_fmt yuv420p images/training/a_video.mp4




class ClassificationCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_folder_name, **kwargs):
        super(ClassificationCallback, self).__init__(**kwargs)
        self.data_it = ImageGenerator(val_folder_name, batch_size=1)
        self.num_images = len(self.data_it)
        self.concept_names = [['blue', 'red', 'green'],
                              ['small', 'medium', 'large'],
                              ['circle', 'square', 'triangle'],
                              ['top', 'centre', 'bottom']]


    def on_epoch_end(self, epoch, logs=None):
        predictions = []
        truth_labels = []
        for i in range(self.num_images):
            tf.print("Classifying image " + str(i+1) + " of " + str(self.num_images), end='\r')
            image_and_label = self.data_it[i]
            truth_labels.append(utils.encode_or_decode(self.data_it[i][1][0]))
            num_concept_domains = len(self.concept_names)
            z_mean, _, _ = self.model.encoder(image_and_label)
            z_mean = z_mean.numpy()[0]

            # for each concept domain, get the concept with mean closest to z_mean
            result = []
            for i, concepts in enumerate(self.concept_names):
                encoding_dict = enc.enc_dict[enc.concept_domains[i]]
                concepts_enc = np.array([encoding_dict[concept] for concept in concepts])
                concept_means = self.model.concept_gaussians.mean.numpy()[i][concepts_enc]
                distances = [np.abs(concept_mean - z_mean[i]) for concept_mean in concept_means]
                result.append(concepts[np.argmin(distances)])
            predictions.append(result)
        tf.print('')
        truth_labels = np.array(truth_labels)
        predictions = np.array(predictions)
        accuracy_list = np.zeros(num_concept_domains)
        for i in range(num_concept_domains):
            accuracy_list[i] = accuracy_score(truth_labels[:,i], predictions[:,i])
            tf.summary.scalar(enc.concept_domains[i] + ' accuracy', data=accuracy_list[i], step=epoch)
        tf.summary.scalar('average accuracy', data=np.mean(accuracy_list), step=epoch)


class ClusterQualityCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_folder_name, **kwargs):
        super(ClusterQualityCallback, self).__init__(**kwargs)
        self.num_images = len(os.listdir(val_folder_name))
        self.data_it = ImageGenerator(val_folder_name, batch_size=self.num_images)
        self.data_it = self.data_it[0]

    def calculate_dimensions_for_domains(self):
        domain_weights = self.model.concept_gaussians.domain_weights
        probs = tf.nn.softmax(tf.reduce_sum(domain_weights, axis=2), axis=1)
        available_dimensions = list(range(len(enc.concept_domains)))
        dimensions_for_domains = []
        for i in range(len(enc.concept_domains)):
            dim_with_highest_prob = tf.argmax(tf.gather(probs, available_dimensions)[:,i])
            chosen_dimension = available_dimensions[dim_with_highest_prob]
            dimensions_for_domains.append(chosen_dimension)
            available_dimensions.remove(chosen_dimension)
        return dimensions_for_domains

    def on_epoch_end(self, epoch, logs=None):
        z_mean, _, _ = self.model.encoder(self.data_it)
        dimensions_for_domains = self.calculate_dimensions_for_domains()
        cluster_quality = []
        for domain, dimension in enumerate(dimensions_for_domains):
            means = z_mean[:, dimension]
            labels = self.data_it[1][:, dimension]
            cluster_quality.append(silhouette_score(means[:, tf.newaxis], labels))
        for i, quality in enumerate(cluster_quality):
            tf.summary.scalar(enc.concept_domains[i] + ' cluster quality', data=cluster_quality[i], step=epoch)
        tf.summary.scalar('average cluster quality', data=np.mean(cluster_quality), step=epoch)

