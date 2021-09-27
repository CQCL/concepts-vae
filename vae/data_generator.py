from tensorflow import keras
import numpy as np
import PIL
import random
import os

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
        return np.array([
                np.asarray(PIL.Image.open(file_path), dtype=np.float64) / 255.0
                    for file_path in batch_x])
