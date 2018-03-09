# Day_03_03_slim_concat.py
import numpy as np


def usage_1():
    t1 = [[1, 2, 3],
          [4, 5, 6]]
    t2 = [[7, 8, 9],
          [10, 11, 12]]

    v1 = np.concatenate([t1, t2], axis=0)
    v2 = np.concatenate([t1, t2], axis=1)
    print(v1)
    print(v2)
    print(v1.shape)
    print(v2.shape)


def usage_2():
    # 4차원 : [건물, 층, 행, 열]
    # [3, 층, 행, 열]
    a = np.arange(21, 29).reshape(-1, 1, 1, 2)
    b = np.arange(51, 59).reshape(-1, 1, 1, 2)

    print(a)
    print(b)
    print('-' * 50)

    print(a.shape)
    print(b.shape)
    print('-' * 50)

    #  0 1 2 3
    # [][][][]

    v0 = np.concatenate([a, b], axis=0)
    v1 = np.concatenate([a, b], axis=1)
    v2 = np.concatenate([a, b], axis=2)
    v3 = np.concatenate([a, b], axis=3)
    print('axis 0:', v0.shape)
    print(v0)
    print('-' * 50)

    print('axis 1:', v1.shape)
    print(v1)
    print('-' * 50)

    print('axis 2:', v2.shape)
    print(v2)
    print('-' * 50)

    print('axis 3:', v3.shape)
    print(v3)
    print('-' * 50)


usage_1()
usage_2()
