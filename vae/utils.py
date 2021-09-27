import os
import matplotlib.pyplot as plt
import numpy as np
import PIL


def save_vae_clusters(vae, data, labels, file_name='clusters.png'):
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = vae.encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(file_name)

def save_reconstructed_images(vae, data, num_images=10, folder_name='images/reconstructed/', file_name='reconstructed'):
    image_num = 1
    for i in range(len(data)):
        _, _, z = vae.encoder.predict(data[i])
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