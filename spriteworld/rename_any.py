import os
import random

image_dir = 'images/any_three'
num_any = 3

image_files = []
for root_path, _, files in os.walk(image_dir):
    for f in files:
        image_files.append(os.path.join(root_path, f))

for file_path in image_files:
    file_name = os.path.splitext(os.path.split(file_path)[1])[0]
    keywords = file_name.split('_')
    # sample num_any random concepts
    keywords_for_any = random.sample(keywords[1:], num_any)
    # replace keywords with 'any' if they are in the list
    for i, keyword in enumerate(keywords):
        if keyword in keywords_for_any:
            keywords[i] = 'any'
    # create new file name
    new_file_name = '_'.join(keywords) + '.png'
    # rename file
    os.rename(file_path, os.path.join(image_dir, new_file_name))
    