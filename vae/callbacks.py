
import os
import subprocess

import numpy as np
import scipy.stats as stats
import tensorflow as tf
from matplotlib import pyplot as plt

from vae import encoding_dictionary as enc
from vae import utils


class ImageSaveCallback(tf.keras.callbacks.Callback):
    def __init__(self, image_and_label, folder_name, **kwargs):
        super(ImageSaveCallback, self).__init__(**kwargs)
        self.image_and_label = image_and_label[0]
        self.epoch_number = 0
        self.folder_name = folder_name
        self.img_frames = []
        utils.save_image(self.folder_name, 'original_image', 
            np.array([self.image_and_label[0][0]]), timestamp_in_name=False)

    def on_epoch_begin(self, epoch, logs=None):
        reconstructed = self.model.call(self.image_and_label)
        file_name = 'reconstructed_epoch_' + str(epoch)
        utils.save_image(self.folder_name, file_name, 
            np.array([reconstructed[0]]), timestamp_in_name=False)


class GaussianPlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, image_and_label, folder_name, **kwargs):
        super(GaussianPlotCallback, self).__init__(**kwargs)
        self.image_and_label = image_and_label[0]
        self.epoch_number = 0
        self.folder_name = folder_name
        self.concept_names = [['blue', 'red', 'green'],
                              ['small', 'medium', 'large'],
                              ['circle', 'square', 'triangle'],
                              ['top', 'centre', 'bottom']]
        self.concept_encoding = []
        for i in range(len(self.concept_names)):
            encoding = enc.enc_dict[enc.concept_domains[i]]
            self.concept_encoding.append([encoding[concept] for concept in self.concept_names[i]])
    
    def get_concept_gaussians(self):
        all_means = []
        all_log_vars = []
        for i in range(4):
            con_enc = self.concept_encoding[i]
            means = []
            log_vars = []
            for j in range(len(con_enc)):
                label_array = np.zeros((1,4))
                label_array[0][i] = con_enc[j]
                m, lv = self.model.concept_gaussians(label_array)
                means.append(m[0][i])
                log_vars.append(lv[0][i])
            all_means.append(means)
            all_log_vars.append(log_vars)
        return all_means, all_log_vars

    
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

    def on_epoch_begin(self, epoch, logs=None):
        means, log_vars = self.get_concept_gaussians()
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


