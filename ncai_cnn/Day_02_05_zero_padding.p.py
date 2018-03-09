# Day_02_05_zero_padding.p
import tensorflow as tf
import numpy as np


def padding_mnist():
    # SAME
    for stride in range(1, 6):
        #      이미지,필터
        output = (28 - 5 + 4) // stride + 1
        print('{} : {}'.format(stride, output))
    print('-' * 50)

    # VALID
    for stride in range(1, 6):
        #      이미지,필터
        output = (28 - 5) // stride + 1
        print('{} : {}'.format(stride, output))


def padding_alexnet(kernel, width):
    x = np.zeros([32, width, width, 3], dtype=np.float32)
    w = tf.Variable(tf.zeros([kernel, kernel, 3, 96]))

    for stride in range(1, 10):
        conv = tf.nn.conv2d(x, w,
                            [1, stride, stride, 1],
                            padding='SAME')
        output = (width - kernel + (kernel-1)) // stride + 1

        print('{} : {}, {}'.format(stride, conv.get_shape(), output))

    for stride in range(1, 10):
        conv = tf.nn.conv2d(x, w,
                            [1, stride, stride, 1],
                            padding='VALID')
        output = (width - kernel) // stride + 1

        print('{} : {}, {}'.format(stride, conv.get_shape(), output))


# padding_mnist()
# padding_alexnet(11, 224)
padding_alexnet(11, 227)
