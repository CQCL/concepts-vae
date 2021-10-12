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
        batch_labels = np.zeros((len(batch_x), len(enc.concept_domains)), dtype=float)
        for i, file_path in enumerate(batch_x):
            file_name = os.path.splitext(os.path.split(file_path)[1])[0]
            keywords = file_name.split('_')
            for j, concept in enumerate(enc.concept_domains):
                batch_labels[i][j] = enc.enc_dict[concept][keywords[j+1]]
        return batch_images, batch_labels
