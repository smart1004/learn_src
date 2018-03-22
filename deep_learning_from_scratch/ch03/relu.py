# coding: utf-8
import numpy as np
import matplotlib.pylab as plt


def relu(x):
    # np.maximum: Element-wise maximum of array elements.
    return np.maximum(0, x)

def test_relu():
    x = np.arange(-5.0, 5.0, 0.1)
    y = relu(x)
    plt.plot(x, y)
    plt.ylim(-1.0, 5.5)
    plt.show()

def test_relu2():
    for x in np.arange(-5.0, 5.0, 0.1):
        y = relu(x)
        print('{:.2f} : {:.2f}'.format(x, y))

# test_relu2()
test_relu()
