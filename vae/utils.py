import os
import matplotlib.pyplot as plt
import numpy as np
import PIL


def get_cmap(n, name='rainbow'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def save_vae_clusters(vae, data, file_name='clusters'):
    # display a 2D plot of the digit classes in the latent space
    for i in range(4):
        fig = plt.figure(figsize=(12, 10))
        ax1 = fig.add_subplot(111)
        z_mean_list=[]
        z_var_list=[]
        label_list=[]
        for j in range(len(data)):
            z_mean, z_var, _ = vae.encoder.predict(data[j][0])
            z_mean_list.append(z_mean[:, i])
            z_var_list.append(z_var[:, i])
            label_list.append(data[j][1][:,i])
        z_mean_list = np.array(z_mean_list).flatten()
        z_var_list = np.array(z_var_list).flatten()
        label_list = np.array(label_list).flatten()
        unique_labels = np.unique(label_list)
        color=get_cmap(len(unique_labels))
        
        for j, label in enumerate(unique_labels):
            idx = np.where(label_list==label)[0]
            ax1.scatter(z_mean_list[idx], z_var_list[idx], color=color(j), label=label)
        ax1.legend()
        plt.xlabel('z' + str(i) + '-mean')
        plt.ylabel('z' + str(i) + '-var')
        plt.savefig(file_name+ str(i) + '.png')

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