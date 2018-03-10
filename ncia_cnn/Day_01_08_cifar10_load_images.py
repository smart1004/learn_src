# Day_01_08_cifar10_load_images.py
import os
import numpy as np
from sklearn import preprocessing
from PIL import Image


def load_image(filename):
    img = Image.open(filename)
    img.load()
    return np.asarray(img)


def read_file(filename):
    img = load_image(filename)

    splits = os.path.split(filename)
    name_only = splits[-1]
    label, other = name_only.split('_')

    return label, img


def read_dataset(folder):
    filenames = os.listdir(folder)
    # print(*filenames, sep='\n')

    labels, images = [], np.array([])  # np.empty([0], dtype=np.uint8)
    for item in sorted(filenames):
        path = os.path.join(folder, item)

        label, array = read_file(path)

        labels.append(label)
        images = np.append(images, array)

    # print(images.shape)     # (92160,)
    # print(*labels, sep='\n')

    labels = preprocessing.LabelBinarizer().fit_transform(labels)
    images = images.reshape(-1, 32 * 32 * 3)
    # print(images.shape)     # (30, 3072)
    # print(labels)

    return labels, images


if __name__ == '__main__':
    read_dataset('cifar10_mini')
