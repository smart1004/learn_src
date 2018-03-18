# Day_02_05_softmax.py
import math
import tensorflow as tf
import numpy as np


def softmax():

    a = math.e ** 2.0
    b = math.e ** 1.0
    c = math.e ** 0.1

    base = a + b + c
    print('math.e : ', math.e)  # 2.718281828459045
    print(a / base)  # 0.6590011388859678
    print(b / base)  # 0.2424329707047139
    print(c / base)  # 0.09856589040931818

# softmax()

def softmax_1():
    xxy = np.loadtxt('Data/softmax.txt', dtype=np.float32)
    # print(xxy)  # [1. 6. 6. 1. 0. 0.]

    x = xxy[:, :3]  # @@@ 배열 슬라이싱 0~2열까지
    y = xxy[:, 3:]  # 3 ~ 5열
    print(x.shape, y.shape) # (8, 3) (8, 3)

    w = tf.Variable(tf.zeros([3, 3]))
    # shape(8, 3) <-- (8, 3) *  (3, 3)
    z =      tf.matmul(x,        w)
    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=z, labels=y)
    cost = tf.reduce_mean(cost_i)  # shape = () scalar

    train = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

    # --------------------------- #

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i, sess.run(cost))

    sess.close()
# softmax_1()

# 문제
# 3시간 공부하고 7번 출석한 경우와
# 8시간 공부하고 2번 출석한 경우의 학점을 예측해주세요.
def softmax_2():
    xxy = np.loadtxt('Data/softmax.txt', dtype=np.float32)
    print(xxy)

    xx = xxy[:, :3]
    y = xxy[:, 3:]
    # print(x.shape, y.shape) #(8, 3) (8, 3)

    x = tf.placeholder(tf.float32)
    w = tf.Variable(tf.zeros([3, 3]))
    # shape(8, 3) <-- (8, 3) *  (3, 3)
    z = tf.matmul(    x,         w)
    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=z, labels=y)
    cost = tf.reduce_mean(cost_i)  # shape = () scalar

    train = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

    # --------------------------- #

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train, {x: xx})
        print(i, sess.run(cost, {x: xx}))

    # --------------------------- #

    hypothesis = tf.nn.softmax(z)  # hypothesis:  shape(?, 3)

    # 3시간 공부하고 7번 출석한 경우와
    # 8시간 공부하고 2번 출석한 경우의 학점을 예측해주세요.
    y_hat = sess.run(hypothesis,
                     {x: ????????????????????  })
    print(y_hat)

    y_arg = np.argmax(y_hat, axis=1)
    print(y_arg) # [1 0]
    # @@@ 아래 비즈코드에 대한 디코딩 방법, 좋은 방법이다
    grades = np.array(['A', 'B', 'C'])
    print(grades[y_arg]) # ['B' 'A']

    sess.close()
# softmax_2()

# 문제
# 행렬 곱셈에서 w와 x의 위치를 바꾸세요.
def softmax_3_1():
    xxy = np.loadtxt('Data/softmax.txt', dtype=np.float32, unpack=True)
    ''' unpack=True
    print(xxy)
    [[1. 1. 1. 1. 1. 1. 1. 1.]
     [2. 3. 3. 5. 7. 2. 6. 7.]
     [1. 2. 4. 5. 5. 5. 6. 7.]
     [0. 0. 0. 0. 0. 0. 1. 1.]
     [0. 0. 0. 1. 1. 1. 0. 0.]
     [1. 1. 1. 0. 0. 0. 0. 0.]]    
    '''

    xx = xxy[:3]  # <-- xxy[:, :3]
    y  = xxy[3:]   # <-- xxy[:, 3:]

    print(xx.shape, y.shape) #

    x = tf.placeholder(tf.float32)
    w = tf.Variable(tf.zeros([????????? ]))

    #
    z = tf.matmul(w,       x)

    #
    hypothesis = tf.nn.softmax(z, axis=0)

    #
    cross_entropy = y * -tf.log(hypothesis)

    #
    cost_i = tf.reduce_sum(cross_entropy, axis=0)
    # cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=z, labels=y)

    # scalar <--  shape = ()
    cost = tf.reduce_mean(cost_i)

    train = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

    # --------------------------- #

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # for i in range(10):
    #     sess.run(train, {x: xx})
    #     print(i, sess.run(cost, {x: xx}))

    # --------------------------- #

    np.set_printoptions(linewidth=1000)

    zz = sess.run(z, {x: xx})
    print(zz)
    print(zz.shape)
    print('-' * 50)

    hh = sess.run(hypothesis, {x: xx})
    print(hh)
    print(hh.shape)
    print('-' * 50)

    ee = sess.run(cross_entropy, {x: xx})
    print(ee)
    print(ee.shape)
    print('-' * 50)

    ii = sess.run(cost_i, {x: xx})
    print(ii)
    print(ii.shape)
    print('-' * 50)

    cc = sess.run(cost, {x: xx})
    print(cc)
    print(cc.shape)
    print('-' * 50)

    # --------------------------- #

    # y_hat = sess.run(hypothesis,
    #                  {x: [[1., 1.],
    #                       [3., 8.],
    #                       [7., 2.]]})
    # print(y_hat)
    #
    # y_arg = np.argmax(y_hat, axis=0)
    # print(y_arg)
    #
    # grades = np.array(['A', 'B', 'C'])
    # print(grades[y_arg])

    sess.close()


    print('\n\n\n\n\n\n')

# softmax_3_1()