import os
import random
import time
from operator import itemgetter

import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

start = time.time()
data_generator = ImageDataGenerator(
    rotation_range=180,
    width_shift_range=0.075,
    height_shift_range=0.075,
    zoom_range=0.05,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='constant',
    cval=255)

path = "./Owoce"
folders = []

for r, d, f in os.walk(path):
    for folder in d:
        folder_path = os.path.join(r, folder)
        images_count = len([name for name in os.listdir(folder_path)])
        folders.append((folder_path, images_count))

biggest_set_count = max(folders, key=itemgetter(1))[1]
print(biggest_set_count)
# Augmentacja danych
for path, count in folders:
    print(path, count)
    if "_aug" not in path:
        i = 0
        aug_path = path + "_aug"
        total_count = count
        if not os.path.exists(aug_path):
            os.mkdir(aug_path)
        else:
            total_count += len([name for name in os.listdir(aug_path)])
        while i < (biggest_set_count - total_count):
            random_image_path = path + "/" + random.choice(os.listdir(path))
            img = load_img(random_image_path)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)
            for batch in data_generator.flow(x, save_to_dir=aug_path, save_format='jpeg'):
                break
            i += 1

# Przeniesienie z folderu "xxx_aug" do podstawowego
# for path, count in folders:
#     i = 0
#     aug_path = path + "_aug"
#     if "_aug" not in path and len([name for name in os.listdir(aug_path)]) > 0:
#         print(path, count)
#         while i < (biggest_set_count - count):
#             if len([name for name in os.listdir(aug_path)]) <= 0:
#                 break
#             random_image_path = "/" + random.choice(os.listdir(aug_path))
#             os.rename(aug_path + random_image_path, path + random_image_path)
#             i += 1

# Usunięcie nadwyżki plików
# for path, count in folders:
#     i = 0
#     print(path, count)
#     while i < (count - 7354):
#         random_image_path = random.choice(os.listdir(path))
#         if random_image_path.startswith('_'):
#             os.remove(path + "/" + random_image_path)
#             i += 1
end = time.time()
print("It all took: " + str(end - start) + " sec")
