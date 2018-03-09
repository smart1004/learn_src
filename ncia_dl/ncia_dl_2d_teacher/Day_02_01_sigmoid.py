# Day_02_01_sigmoid.py
import math
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + math.e ** -z)


def sigmoid_plot():
    print(math.e)
    print('-' * 50)

    print(sigmoid(-1))
    print(sigmoid(0))
    print(sigmoid(1))

    for i in range(-100, 100):
        i /= 10
        s = sigmoid(i)
        plt.plot(i, s, 'ro')

    plt.show()


def log_A():
    return 'A'


def log_B():
    return 'B'


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


import numpy as np

a = np.arange(3)
b = np.arange(3).reshape(3, 1)
print(a)
print(b)
print(a.shape, b.shape)

c = a - b
print(c)
