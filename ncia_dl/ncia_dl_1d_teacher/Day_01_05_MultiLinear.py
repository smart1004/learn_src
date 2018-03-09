# Day_01_05_MultiLinear.py
import tensorflow as tf


def multi_1():
    x1 = [1, 0, 3, 0, 5]
    x2 = [0, 2, 0, 4, 0]
    y = [1, 2, 3, 4, 5]

    w1 = tf.Variable(tf.random_uniform([1], -1, 1))
    w2 = tf.Variable(tf.random_uniform([1], -1, 1))
    b = tf.Variable(tf.random_uniform([1], -1, 1))

    hypothesis = w1 * x1 + w2 * x2 + b
    cost = tf.reduce_mean((hypothesis - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i, sess.run(cost))

    sess.close()


# 문제
# x1과 x2를 하나의 변수로 합쳐주세요. (tf.matmul)
def multi_2():
    x = [[1., 0., 3., 0., 5.],
         [0., 2., 0., 4., 0.]]
    y = [1, 2, 3, 4, 5]

    w = tf.Variable(tf.random_uniform([1, 2], -1, 1))
    b = tf.Variable(tf.random_uniform([1], -1, 1))

    # hypothesis = w[0] * x[0] + w[1] * x[1] + b

    # (1, 5) = (1, 2) x (2, 5)
    hypothesis = tf.matmul(w, x) + b
    cost = tf.reduce_mean((hypothesis - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i, sess.run(cost))

    sess.close()


# 문제
# bias를 없애보세요.
def multi_3():
    x = [[1., 0., 3., 0., 5.],
         [0., 2., 0., 4., 0.],
         [1., 1., 1., 1., 1.]]
    y = [1, 2, 3, 4, 5]

    w = tf.Variable(tf.random_uniform([1, 3], -1, 1))

    # hypothesis = w[0] * x[0] + w[1] * x[1] + w[2] * x[?]
    # hypothesis = w[0] * x[0] + w[1] * x[1] +  b   * x[?]

    # (1, 5) = (1, 3) x (3, 5)
    hypothesis = tf.matmul(w, x)
    cost = tf.reduce_mean((hypothesis - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i, sess.run(cost))

    sess.close()


# 문제
# placeholder 버전으로 수정해서
# (5, 7)일 때와 (5, 12)일 때의 값을 예측해보세요.
xx = [[1., 0., 3., 0., 5.],
      [0., 2., 0., 4., 0.],
      [1., 1., 1., 1., 1.]]
y = [1, 2, 3, 4, 5]

x = tf.placeholder(tf.float32)
w = tf.Variable(tf.random_uniform([1, 3], -1, 1))

hypothesis = tf.matmul(w, x)
cost = tf.reduce_mean((hypothesis - y) ** 2)

optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(10):
    sess.run(train, {x: xx})
    print(i, sess.run(cost, {x: xx}))

# print(sess.run(hypothesis, {x: [5, 7]}))
# print(sess.run(hypothesis, {x: [5, 7, 1]}))
print(sess.run(hypothesis, {x: xx}))
print(sess.run(hypothesis, {x: [[1., 0., 3., 0., 5.],
                                [0., 2., 0., 4., 0.],
                                [1., 1., 1., 1., 1.]]}))
print(sess.run(hypothesis, {x: [[5., 5.],
                                [7., 12.],
                                [1., 1.]]}))

sess.close()





