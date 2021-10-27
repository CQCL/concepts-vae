from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
import PIL
import random
import tensorflow as tf
import vae.encoding_dictionary as enc
from vae.model import Sampling


def get_cmap(n, name='rainbow'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def save_vae_clusters(vae, data, latent_dim, file_name='clusters'):
    # display a 2D plot of the digit classes in the latent space
    for i in range(latent_dim):
        fig = plt.figure(figsize=(12, 10))
        axs = fig.subplots(2,2).flatten()
        for k, plot_label in enumerate(enc.concept_domains):
            z_mean_list=[]
            z_var_list=[]
            label_list=[]
            for j in range(len(data)):
                z_mean, z_var, _ = vae.encoder.predict(data[j][0])
                z_mean_list.append(z_mean[:, i])
                z_var_list.append(z_var[:, i])
                label_list.append(data[j][0][1][:,k])
            z_mean_list = np.hstack(np.array(z_mean_list,dtype=object).flatten())
            z_var_list = np.hstack(np.array(z_var_list,dtype=object).flatten())
            label_list = np.hstack(np.array(label_list,dtype=object).flatten())
            unique_labels = np.unique(label_list)
            color=get_cmap(len(unique_labels))

            for j, label in enumerate(unique_labels):
                idx = np.where(label_list==label)[0]
                axs[k].scatter(z_mean_list[idx], z_var_list[idx], color=color(j), label=enc.dec_dict[plot_label][label], s=1.25, alpha=0.2)
            axs[k].legend()
            axs[k].title.set_text(plot_label)
            axs[k].set_xlabel('z' + str(i) + '-mean')
            axs[k].set_ylabel('z' + str(i) + '-var')
            # plt.savefig(file_name + '_latent_dim' + str(i) + '_label_dim' + str(k) + '.png')
        plt.savefig(file_name + '_latent_dim' + str(i) + '.png')


def plot_latent_space(vae, latent_space, plot_dim, dim_min, dim_max, num_images=30, image_size=64, figsize=15, file_name='image'):
    latent_space =  np.ndarray.copy(np.array(latent_space))
    figure = np.zeros((image_size, image_size * num_images, 3))
    dim_value = np.linspace(dim_min, dim_max, num_images)

    for i, dv in enumerate(dim_value):
        latent_space[0][plot_dim] = dv
        decoded_image = vae.decoder.predict(tf.convert_to_tensor(latent_space, dtype=np.float))[0]
        figure[
                :, i * image_size : (i + 1) * image_size, :
            ] = decoded_image

    plt.figure(figsize=(figsize, figsize/num_images + 1))
    start_range = image_size // 2
    end_range = num_images * image_size + start_range
    pixel_range = np.arange(start_range, end_range, image_size)
    sample_range_x = np.round(dim_value, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks([])
    plt.xlabel('latent dimension ' + str(plot_dim))
    plt.imshow(figure)
    plt.savefig(file_name + '_latent_dim' + str(plot_dim) + '.png')


def save_images(folder_name, file_name, img):
    img *= 255
    for j in range(len(img)):
        im = PIL.Image.fromarray(np.uint8(img[j]))
        im.save(os.path.join(folder_name, datetime.utcnow().strftime("%B_%d_%H%M%S%f_") + file_name + '.png'))


def generate_images_from_gaussians(vae, means, log_vars):
    ''' 
        vae = VAE
        means = array of means of Gaussians (of the dimension of the latent space)
        log_vars = array of log_vars of Gaussians (of the dimension of the latent space)

        returns: image directly from the decoder
    '''
    input = tf.convert_to_tensor(np.array([means,log_vars], dtype=np.float))
    z = Sampling()(input)
    img = vae.decoder.predict(z)
    return img


def generate_images_from_concept(vae, concept, num_images = 1, folder_name='images/concept_images/'):
    ''' 
        vae = VAE
        concept = list of strings
        num_images: how many images we want to generate
        returns: nothing
        saves image
    '''
    means = []
    log_vars = []
    for i, domain in enumerate(enc.concept_domains):
        concept_number = enc.enc_dict[domain][concept[i]]
        means.append(vae.concept_gaussians.mean[i][concept_number])
        log_vars.append(vae.concept_gaussians.log_var[i][concept_number])
    means = np.array(means)
    log_vars = np.array(log_vars)
    extra_dimensions = vae.params['latent_dim'] - len(concept)
    means = np.concatenate((means, np.zeros(extra_dimensions)))
    log_vars = np.concatenate((log_vars, np.ones(extra_dimensions)))
    means = np.tile(means,(num_images,1))
    log_vars = np.tile(log_vars,(num_images,1))
    images = generate_images_from_gaussians(vae, means, log_vars)
    save_images(folder_name, '_'.join(concept), images)


def generate_images_from_multiple_concepts(vae, concept_list, num_images=10, folder_name='images/reconstructed/'):
    for _ in range(num_images):
        colour, size, shape, position = random.choice(concept_list)
        generate_images_from_concept(vae, [colour, size, shape, position], 1, folder_name)


def save_reconstructed_images_with_data(vae, data, num_images=10, folder_name='images/reconstructed/', file_name='reconstructed'):
    image_num = 1
    for i in range(num_images):
        _, _, z = vae.encoder.predict(data[i][0])
        img = vae.decoder.predict(z)
        img *= 255
        for j in range(len(img)):
            if image_num > num_images:
                return
            im = PIL.Image.fromarray(np.uint8(img[j]))
            im.save(os.path.join(folder_name, file_name + str(image_num) + '.png'))
            image_num = image_num + 1
