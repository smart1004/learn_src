# Day_01_04_LinearRegression_cars.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def loadtxt():
    cars = np.loadtxt('Data/cars.csv', delimiter=',')
    print(type(cars))
    print(cars.shape, cars.dtype)
    print(cars)

    cars = np.loadtxt('Data/cars.csv', delimiter=',',
                      unpack=True, dtype=np.float32)
    print(type(cars))
    print(cars.shape, cars.dtype)
    print(cars)


# 문제
# cars.csv 파일로 학습한 다음에
# 속도가 10, 15일 때의 제동거리를 예측해보세요.
# cars = np.loadtxt('Data/cars.csv', delimiter=',',
#                   unpack=True, dtype=np.float32)
#
# xx = cars[0]
# y = cars[1]

cars = np.loadtxt('Data/cars.csv', delimiter=',',
                  dtype=np.float32)

xx = cars[:, 0]
y = cars[:, 1]

x = tf.placeholder(tf.float32)

w = tf.Variable(tf.random_uniform([1], -1, 1))
b = tf.Variable(tf.random_uniform([1], -1, 1))

hypothesis = w * x + b
cost = tf.reduce_mean((hypothesis - y) ** 2)

optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(10):
    sess.run(train, feed_dict={x: xx})
    print(i, sess.run(cost, {x: xx}))

print(sess.run(hypothesis, {x: [10, 15]}))

y1 = sess.run(hypothesis, {x: 0})
y2 = sess.run(hypothesis, {x: 30})
print(y2, y2[0])

sess.close()

plt.plot(xx, y, 'ro')
plt.plot([0, 30], [0, y2])
plt.plot([0, 30], [y1, y2])
plt.show()
