from tensorflow import keras
import numpy as np
import PIL
import random
import os
from vae import encoding_dictionary as enc

class ImageGenerator(keras.utils.Sequence) :
  
    def __init__(self, image_dir, batch_size) :
        self.batch_size = batch_size
        self.image_dir = image_dir
        self.image_files = []
        for root_path, _, files in os.walk(image_dir):
            for f in files:
                self.image_files.append(os.path.join(root_path, f))
        random.shuffle(self.image_files)
      
      
    def __len__(self) :
        return (np.ceil(len(self.image_files) / float(self.batch_size))).astype(np.int)
    
    
    def __getitem__(self, idx) :
        batch_x = self.image_files[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_images =  np.array([np.asarray(
            PIL.Image.open(file_path), dtype=np.float64) / 255.0 for file_path in batch_x])
        batch_labels = []
        for file_path in batch_x:
            file_name = os.path.splitext(os.path.split(file_path)[1])[0]
            keywords = file_name.split('_')
            batch_labels.append([
                enc.enc_dict['colour'][keywords[1]],
                enc.enc_dict['size'][keywords[2]],
                enc.enc_dict['shape'][keywords[3]],
                enc.enc_dict['position'][keywords[4]],
            ])
        batch_labels = np.array(batch_labels, dtype=float)
        return batch_images, batch_labels
