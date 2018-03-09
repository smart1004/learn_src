# Day_02_03_Logistic_iris.py
import csv
import tensorflow as tf
import numpy as np


def get_iris(sp_true, sp_false):
    f = open('Data/iris.csv')

    # skip header.
    f.readline()

    rows = []
    for row in csv.reader(f):
        # print(row)

        if row[-1] != sp_true and row[-1] != sp_false:
            continue

        item = [1.]
        item += [float(i) for i in row[1:-1]]
        item.append(int(row[-1] == sp_true))
        # item.append(1 if row[-1] == sp_true else 0)
        # item += [1 if row[-1] == sp_true else 0]    # extend()

        rows.append(item)

    f.close()

    return rows


# 문제
# 70개의 데이터로 학습한 다음에 30개의 데이터에 대해 예측해보세요.
rows = get_iris('setosa', 'versicolor')
# print(*rows, sep='\n')

iris = np.array(rows)
print(iris.shape)

features = iris[:, :-1]
targets = iris[:, -1:]
print(features.shape, targets.shape)

# xx = features[:70]        # 데이터 편중
# y = targets[:70]
# xx = features[:35] + features[-35:]   # 벡터 연산. 크기가 35개.
# y = targets[:35] + targets[-35:]
xx = np.vstack([features[:35], features[-35:]])
y = np.vstack([targets[:35], targets[-35:]])
print(xx.shape, y.shape)

x = tf.placeholder(tf.float32)
w = tf.Variable(tf.random_uniform([5, 1], -1, 1))

# (100, 1) = (100, 5) x (5, 1)
z = tf.matmul(x, w)
hypothesis = tf.nn.sigmoid(z)

cost = tf.reduce_mean(y * -tf.log(hypothesis) +
                      (1 - y) * -tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(100):
    sess.run(train, {x: xx})
    print(i, sess.run(cost, {x: xx}))

# --------------------------------- #

# xx = features[70:]        # 데이터 편중.
# y = targets[70:]
xx = features[35:-35]
y = targets[35:-35]
y = y.reshape(-1)

y_hat = sess.run(hypothesis, {x: xx})
y_hat = y_hat.reshape(-1)
print(y_hat)
print(y)

# y_hat_bool = (y_hat >= 0.5)
y_hat_bool = np.float32(y_hat >= 0.5)
print(y_hat_bool)
print('-' * 50)

# predict
print(y_hat_bool == y)
print(np.mean(y_hat_bool == y))

sess.close()







print('\n\n\n\n\n\n')