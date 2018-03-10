# Day_03_01_softmax_iris.py
import numpy as np
import tensorflow as tf
from sklearn import model_selection     # scipy

# 문제
# iris_softmax.csv 파일을 소프트맥스 알고리즘에 적용해보세요.
# 120개로 학습하고 30개에 대해 예측해주세요.
def get_iris_1():
    iris = np.loadtxt('Data/iris_softmax.csv',
                      delimiter=',')
    # print(iris)

    x = iris[:, :-3]
    y = iris[:, -3:]

    x_train = np.vstack([x[:40], x[50:90], x[100:140]])
    y_train = np.vstack([y[:40], y[50:90], y[100:140]])
    # print(y_train.shape)

    x_test = np.vstack([x[40:50], x[90:100], x[140:]])
    y_test = np.vstack([y[40:50], y[90:100], y[140:]])

    return x_train, x_test, y_train, y_test


def get_iris_2():
    iris = np.loadtxt('Data/iris_softmax.csv', delimiter=',')

    np.random.shuffle(iris)

    x_train = iris[:120, :-3]
    y_train = iris[:120, -3:]

    x_test = iris[120:, :-3]
    y_test = iris[120:, -3:]

    return x_train, x_test, y_train, y_test


def get_iris_3():
    iris = np.loadtxt('Data/iris_softmax.csv', delimiter=',')

    # return model_selection.train_test_split(iris[:, :-3], iris[:, -3:])
    # return model_selection.train_test_split(iris[:, :-3],
    #                                         iris[:, -3:],
    #                                         train_size=120,
    #                                         test_size=30)
    return model_selection.train_test_split(iris[:, :-3],
                                            iris[:, -3:],
                                            train_size=0.7,
                                            test_size=0.3)


x_train, x_test, y_train, y_test = get_iris_3()

x = tf.placeholder(tf.float32)
w = tf.Variable(tf.zeros([5, 3]))

# (120, 3) = (120, 5) x (5, 3)
z = tf.matmul(x, w)
hypothesis = tf.nn.softmax(z)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=z, labels=y_train)
cost = tf.reduce_mean(cost_i)

train = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

# --------------------------- #

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(10):
    sess.run(train, {x: x_train})
    print(i, sess.run(cost, {x: x_train}))

# --------------------------- #

pred = sess.run(hypothesis, {x: x_test})
print(pred.shape)
print(pred[:3])
print(y_test[:3])
print('-' * 50)

pred_arg = np.argmax(pred, axis=1)
test_arg = np.argmax(y_test, axis=1)
print(pred_arg)
print(test_arg)
print('-' * 50)

equals = (pred_arg == test_arg)
print(equals)
print('accuracy :', np.mean(equals))

sess.close()






print('\n\n\n\n\n\n\n\n')