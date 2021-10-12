import os
import matplotlib.pyplot as plt
import numpy as np
import PIL
import vae.encoding_dictionary as enc


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
                label_list.append(data[j][1][:,k])
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

def save_reconstructed_images(vae, data, num_images=10, folder_name='images/reconstructed/', file_name='reconstructed'):
    image_num = 1
    for i in range(len(data)):
        _, _, z = vae.encoder.predict(data[i][0])
        img = vae.decoder.predict(z)
        img *= 255
        for j in range(len(img)):
            if image_num > num_images:
                return
            im = PIL.Image.fromarray(np.uint8(img[j]))
            im.save(os.path.join(folder_name, file_name + str(image_num) + '.png'))
            image_num = image_num + 1

    # for i in range(10):
    #     input = tf.convert_to_tensor(np.array([[[0,0]], [[1,1]]], dtype=np.float))
    #     z = Sampling()(input)
    #     print(z)
    #     img = vae.decoder.predict(z)

    #     img *= 255
    #     im = PIL.Image.fromarray(np.uint8(img[0]))
    #     im.save('reconstructed' + str(i) + '.png')