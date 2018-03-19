# coding: utf-8
import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def test_sigmoid():
    X = np.arange(-5.0, 5.0, 0.1)
    Y = sigmoid(X)
    plt.plot(X, Y)
    plt.ylim(-0.1, 1.1)
    plt.show()

def test_sigmoid2():
    for x in np.arange(-5.0, 5.0, 0.1):
        y = sigmoid(x)
        print('{:.2f} : {:.2f}'.format(x, y))

#------------
def exp1(x):
    return np.exp(-x)
def test_exp2():
    for x in np.arange(-5.0, 5.0, 0.1):
        y = exp1(x)
        print('{:.2f} : {:.2f}'.format(x, y))
#------------
# test_exp2()
test_sigmoid()
# print(sigmoid(1))
# print('np.exp(-2)', np.exp(-2))
# print('np.exp(-1)', np.exp(-1))
# print('np.exp(-0)', np.exp(-0))
# print('np.exp(0)', np.exp(0))
# print('np.exp(1)', np.exp(1))
# print('np.exp(2)', np.exp(2))
