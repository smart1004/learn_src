# Day_01_01_cost.py
import matplotlib.pyplot as plt
import numpy as np

# ctrl + /
# ctrl + shift + f10
# shift + f10


def cost(x, y, w):
    c = 0
    for i in range(len(x)):
        hx = w * x[i]
        e = (hx - y[i]) ** 2
        c += e
    return c / len(x)


def gradient_descent(x, y, w):
    c = 0
    for i in range(len(x)):
        hx = w * x[i]
        e = (hx - y[i]) * x[i]
        c += e
    return c / len(x)


def use_cost():
    #     1    0
    # y = wx + b
    x = [1, 2, 3]
    y = [1, 2, 3]

    print(cost(x, y, -1))
    print(cost(x, y, 0))
    print(cost(x, y, 1))
    print(cost(x, y, 2))
    print(cost(x, y, 3))
    print('-' * 50)

    for i in np.arange(-3, 5, 0.1):
        print('{:.2f} : {:.2f}'.format(i, cost(x, y, i)))

        plt.plot(i, cost(x, y, i), 'ro')

    plt.show()

# 미분 : 기울기, 순간변화량
#       x축으로 1만큼 변할 때의 y 변화량

# y = 3         0=1, 0=2, 0=3
# y = x         1=1, 2=2, 3=3
# y = 2x        2=1, 4=2, 6=3
# y = (x+1)     2=1, 3=2, 4=3
# y = x^2       1=1, 4=2, 9=3       x^2 -> 2x
# y = (x+1)^2                           -> 2(x+1)
# y = (x+z)
# y = xz

x = [1, 2, 3]
y = [1, 2, 3]

w = 10
for i in range(100):
    c = cost(x, y, w)
    g = gradient_descent(x, y, w)
    w -= 0.1 * g

    # early stopping (공짜 점심)
    if c < 1e-15:
        break

    # print(i, w)
    print(i, c)

print('w :', w)

# 문제
# w를 1로 만드는 2가지 방법을 찾아보세요.
