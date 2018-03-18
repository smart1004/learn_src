# Day_02_04_stochastic.py
# @@@@ 추가 학습 필요
# stochastic 확률(론)적인
import numpy as np
import matplotlib.pyplot as plt
import random


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gradient_descent(x, y):
    m, n = x.shape              # 100, 3
    w = np.zeros([n, 1])        # (3, 1)
    lr = 0.01

    for _ in range(m):
        z = np.dot(x, w)        # (100, 1) = (100, 3) x (3, 1)
        h = sigmoid(z)          # (100, 1)
        e = h - y               # (100, 1) = (100, 1) - (100, 1)
        g = np.dot(x.T, e)      # (3, 1) = (3, 100) x (100, 1)
        w -= lr * g
    return w.reshape(-1)        # (3,)

def gradient_descent_stoc(x, y):
    m, n = x.shape              # 100, 3
    w = np.zeros(n)             # (3,)
    lr = 0.01

    for i in range(m * 10):
        p = i % m
        z = np.sum(x[p] * w)    # scalar = sum((3,) * (3,))
        h = sigmoid(z)          # scalar
        e = h - y[p]            # scalar
        g = e * x[p]            # (3,) -= scalar * (3,)
        w -= lr * g
    return w


def gradient_descent_stoc_random(x, y):
    m, n = x.shape              # 100, 3
    w = np.zeros(n)             # (3,)
    lr = 0.01

    for i in range(m * 10):
        p = random.randrange(m)
        z = np.sum(x[p] * w)    # scalar = sum((3,) * (3,))
        h = sigmoid(z)          # scalar
        e = h - y[p]            # scalar
        g = e * x[p]            # (3,) -= scalar * (3,)
        w -= lr * g
    return w


def minibatch(x, y):
    m, n = x.shape              # 100, 3
    w = np.zeros([n, 1])        # (3, 1)
    lr = 0.01
    epochs = 10
    batch_size = 5

    for _ in range(epochs):
        count = m // batch_size
        for k in range(count):
            s = k * batch_size
            f = s + batch_size

            z = np.dot(x[s:f], w)   # (5, 1) = (5, 3) x (3, 1)
            h = sigmoid(z)          # (5, 1)
            e = h - y[s:f]          # (5, 1) = (5, 1) - (5, 1)
            g = np.dot(x[s:f].T, e) # (3, 1) = (3, 5) x (5, 1)
            w -= lr * g
    return w.reshape(-1)            # (3,)


def minibatch_random(x, y):
    m, n = x.shape              # 100, 3
    w = np.zeros([n, 1])        # (3, 1)
    lr = 0.01
    epochs = 10
    batch_size = 5

    for _ in range(epochs):
        count = m // batch_size
        for k in range(count):
            s = k * batch_size
            f = s + batch_size

            z = np.dot(x[s:f], w)   # (5, 1) = (5, 3) x (3, 1)
            h = sigmoid(z)          # (5, 1)
            e = h - y[s:f]          # (5, 1) = (5, 1) - (5, 1)
            g = np.dot(x[s:f].T, e) # (3, 1) = (3, 5) x (5, 1)
            w -= lr * g

        t = np.random.randint(1000)
        np.random.seed(t)
        np.random.shuffle(x)
        np.random.seed(t)
        np.random.shuffle(y)

    return w.reshape(-1)            # (3,)


def decision_boundary(w, c):
    # h(x) = w1 * x1 + w2 * x2 + b
    # 0 = w1 * x1 + w2 * x2 + b
    # 0 = w1 * x + w2 * y + b
    # -(w1 * x + b) = w2 * y
    # -(w1 * x + b) / w2 = y
    # y = -(w1 * x + b) / w2

    b, w1, w2 = w[0], w[1], w[2]

    y1 = -(w1 * -4 + b) / w2
    y2 = -(w1 *  4 + b) / w2

    plt.plot([-4, 4], [y1, y2], c)


action = np.loadtxt('Data/action.txt', delimiter=',')
print(action.shape)
# print(action)

x = action[:, :-1]
y = action[:, -1:]
print(x.shape, y.shape)

# w = gradient_descent(x, y)
# decision_boundary(w, 'r')

# for item in action:
#     print(item)

for _, x1, x2, label in action:
    print(x1, x2, label)

    # plt.plot(x1, x2, ['ro', 'go'][int(label)])
    plt.plot(x1, x2, 'ro' if label else 'go')

decision_boundary(gradient_descent(x, y), 'r')
decision_boundary(gradient_descent_stoc(x, y), 'g')
decision_boundary(gradient_descent_stoc_random(x, y), 'b')
decision_boundary(minibatch(x, y), 'm')
decision_boundary(minibatch_random(x, y), 'c')

# plt.xlim(-3, 15)
plt.show()



