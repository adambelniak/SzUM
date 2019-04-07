import numpy as np
import os
from keras.preprocessing import image
import json
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
train_path = "Owoce v.1.0"
normalize_data_path = "Owoce_normalized"

input_dim = (100, 100)


def load_data():
    x_data = []
    y_data = []
    x_path = []
    num_to_fruit_name = {}
    for i, folder in enumerate(os.listdir(train_path)):
        if folder == '.DS_Store':
            continue
        num_to_fruit_name[i] = folder
        categories = os.path.join(train_path, folder)
        for image_name in os.listdir(categories):
            img_path = os.path.join(categories, image_name)
            img = image.load_img(img_path, target_size=input_dim)
            x = image.img_to_array(img)
            x_data.append(x)
            y_data.append(i)
            x_path.append(os.path.join(folder, image_name))
    fruit_code_file = json.dumps(num_to_fruit_name, ensure_ascii=False)
    f = open('fruit_code.json', 'w')
    f.write(fruit_code_file)
    return np.array(x_data, dtype='float32'), np.array(y_data), num_to_fruit_name, x_path


def plot_and_normalize_data(images):
    return images / 255


def plot_data_distribution(images):
    sns.distplot(images.flatten(), kde=False)
    plt.show()


def save_normalize_images(images, x_path, fruit_code_file):
    os.mkdir(normalize_data_path)
    for key in fruit_code_file.keys():
        os.mkdir(os.path.join(normalize_data_path, fruit_code_file[key]))
    for (data, path) in zip(images, x_path):
        image.save_img(os.path.join(normalize_data_path, path), image.array_to_img(data))

if __name__ == '__main__':
    x_data, y_data, fruit_code_fil, x_path = load_data()
    # plot_data_distribution(x_data)
    normalize_x_data = plot_and_normalize_data(x_data)
    # plot_data_distribution(normalize_x_data[:100])
    save_normalize_images(normalize_x_data, x_path, fruit_code_fil)


