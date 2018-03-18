# Day_03_02_minibatch.py
import numpy as np
import tensorflow as tf
from sklearn import model_selection     # scipy

iris = np.loadtxt('Data/iris_softmax.csv', delimiter=',')

data = model_selection.train_test_split(iris[:, :-3], iris[:, -3:])
x_train, x_test, y_train, y_test = data

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
w = tf.Variable(tf.zeros([5, 3]))

# (120, 3) = (120, 5) x (5, 3)
z = tf.matmul(x, w)
hypothesis = tf.nn.softmax(z)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=z, labels=y)
cost = tf.reduce_mean(cost_i)

train = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

# --------------------------- #

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 문제
# 미니배치 방식으로 수정해주세요.
train_set = np.hstack([x_train, y_train])

epochs, batch_size = 15, 5
loops = len(x_train) // batch_size
for i in range(epochs):
    total = 0
    for j in range(loops):
        s = j * batch_size
        f = s + batch_size

        # sess.run(train, {x: x_train[s:f], y: y_train[s:f]})
        # total += sess.run(cost, {x: x_train[s:f], y: y_train[s:f]})

        _, loss = sess.run([train, cost], {x: x_train[s:f], y: y_train[s:f]})
        total += loss

    print(i, total / loops)

    # np.random.shuffle(train_set)
    # x_train = train_set[:, :-3]
    # y_train = train_set[:, -3:]

    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]

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

# print('\n\n\n\n\n\n\n\n')