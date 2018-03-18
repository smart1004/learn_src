# Day_02_01_sigmoid.py
import math
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + math.e ** -z)

def sigmoid_plot():
    print(math.e)       # 2.718281828459045
    print('-' * 50)

    print(sigmoid(-1))  # 0.2689414213699951
    print(sigmoid(0))   # 0.5
    print(sigmoid(1))   # 0.7310585786300049

    for i in range(-100, 100):
        i /= 10  # 원본  i = i / 10
        s = sigmoid(i)
        plt.plot(i, s, 'ro')
    plt.show()
# sigmoid_plot()


def log_A():
    return 'A'

def log_B():
    return 'B'

def test_sigmoid():
    y = 1
    if y == 1:
        print(log_A(), log_A() * y)
    else:
        print(log_B(), log_B() * (1 - y))

    print(log_A() * y + log_B() * (1 - y))

    y = 0
    if y == 1:
        print(log_A(), log_A() * y)
    else:
        print(log_B(), log_B() * (1 - y))

    print(log_A() * y + log_B() * (1 - y))
# test_sigmoid()

def numpy_1():
    import numpy as np
    a = np.arange(3)  # [0 1 2]
    b = np.arange(3).reshape(3, 1)

    print(a)  # [0 1 2]
    print(b)
    '''   np.arange(3).reshape(3, 1)
    [[0]
     [1]
     [2]]     '''
    print(a.shape, b.shape)  # (3,) (3, 1)
    c = a - b
    print(c)
    '''   c = a - b
    [[ 0  1  2]
     [-1  0  1]
     [-2 -1  0]]    '''

numpy_1()