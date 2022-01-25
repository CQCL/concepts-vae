import os
import random
import shutil

image_dir = 'images/basic'
train_dir  = image_dir + '_train'
test_dir   = image_dir + '_test'
val_dir    = image_dir + '_val'

test = 0.1
val = 0.1


image_files = []
for root_path, _, files in os.walk(image_dir):
    for f in files:
        image_files.append(os.path.join(root_path, f))

# split files into train, val, and test
train_files = []
val_files = []
test_files = []
for file_path in image_files:
    # split into train, val, and test
    random_num = random.random()
    if random_num < test:
        test_files.append(file_path)
    elif random_num - test < val:
        val_files.append(file_path)
    else:
        train_files.append(file_path)

os.makedirs(train_dir)
os.makedirs(val_dir)
os.makedirs(test_dir)

for f in train_files:
    shutil.copy2(f, train_dir)
for f in val_files:
    shutil.copy2(f, val_dir)
for f in test_files:
    shutil.copy2(f, test_dir)