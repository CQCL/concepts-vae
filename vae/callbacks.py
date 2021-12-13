
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


