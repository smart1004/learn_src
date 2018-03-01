# Day_01_07_cifar10_save_images.py
import os
import numpy as np
import pickle
from PIL import Image
from collections import defaultdict


def show_info():
    result = unpickle('cifar10/data_batch_1')
    print(result)
    print(result.keys())
    # dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
    print('-' * 50)

    print(result[b'batch_label'])
    print(result[b'labels'])
    print(result[b'data'])
    print(result[b'filenames'])
    print('-' * 50)

    print(unpickle('cifar10/batches.meta'))
    # {b'num_cases_per_batch': 10000,
    #  b'label_names': [b'airplane', b'automobile', b'bird', b'cat', b'deer', b'dog', b'frog', b'horse', b'ship', b'truck'],
    #  b'num_vis': 3072}


def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict


def save_image_to_file(filename, array):
    im = Image.fromarray(array)
    im.save(filename)


def make_mini_dataset(folder, targets, size, filename):
    if not os.path.isdir(folder):
        os.mkdir(folder)

    result = unpickle(filename)

    labels = result[b'labels']
    images = result[b'data']
    filenames = result[b'filenames']

    dic = defaultdict(list)
    for label, _, image in zip(labels, filenames, images):
        dic[label].append(image)

    print(dic.keys())
    # dict_keys([6, 9, 4, 1, 2, 7, 8, 3, 5, 0])

    for k, v in dic.items():
        np.random.shuffle(v)

    names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
             'dog', 'frog', 'horse', 'ship', 'truck']

    for k, v in dic.items():
        if not k in targets:
            continue

        parent = os.path.join(folder, names[k])

        for i in range(size):
            image_1 = v[i]
            image_2 = image_1.reshape(3, 32, 32)    # depth, height, width
            image_3 = image_2.transpose(1, 2, 0)    # height, width, depth

            filename = parent + '_{:03}.jpg'.format(i+1)
            save_image_to_file(filename, image_3)


# show_info()
make_mini_dataset('cifar10_mini',
                  [0, 1, 2], 10,
                  'cifar10/data_batch_1')
