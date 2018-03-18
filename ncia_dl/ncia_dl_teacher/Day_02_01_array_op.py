# Day_02_01_sigmoid.py
import math
import matplotlib.pyplot as plt

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