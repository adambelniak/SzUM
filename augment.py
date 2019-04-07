import glob
import random
import skimage as sk
import numpy
from skimage import util

from PIL import Image


def rotate_image(path, angle):
    img = Image.open(path)
    im2 = img.convert('RGBA')
    rot = im2.rotate(angle, expand=1)
    # change size back to 100px x 100px
    rot = rot.crop(box=(rot.size[0] / 2 - img.size[0] / 2,
                        rot.size[1] / 2 - img.size[1] / 2,
                        rot.size[0] / 2 + img.size[0] / 2,
                        rot.size[1] / 2 + img.size[1] / 2))
    fff = Image.new('RGBA', rot.size, (255,) * 4)
    out = Image.composite(rot, fff, rot)
    out.convert(img.mode).save(path[:-4] + "aug" + path[-4:])


def mirror_image(path):
    image_obj = Image.open(path)
    rotated_image = image_obj.transpose(Image.FLIP_LEFT_RIGHT)
    rotated_image.save(path[:-4] + "aug" + path[-4:])


def random_noise(path):
    img = Image.open(path)
    noise = numpy.random.randint(5, size=(164, 278, 4), dtype='int64')
    np_img = numpy.array(img)
    for i in range(100):
        for j in range(100):
            for k in range(3):
                if np_img[i][j][k] != 255:
                    np_img[i][j][k] += noise[i][j][k]
    # solution below makes too much noise
    # np_img = sk.util.random_noise(np_img)
    img = Image.fromarray(np_img)
    img.save(path[:-4] + "augnoise" + path[-4:])


counter = 0
augmentation_percentage = 10
files = [f for f in glob.glob("**/*.jpg", recursive=True)]

for f in files:
    if counter % augmentation_percentage == 0:
        function_to_apply = random.randint(0, 2)
        if function_to_apply == 0:
            if random.randint(0, 1) == 0:
                rotate_image(f, random.randint(10, 76))
            else:
                rotate_image(f, -random.randint(10, 76))
        if function_to_apply == 1:
            mirror_image(f)
        if function_to_apply == 2:
            random_noise(f)

    counter += 1
