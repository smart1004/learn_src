# Day_03_03_ensenble.py
import numpy as np
import tensorflow as tf
from sklearn import model_selection


def get_iris():
    iris = np.loadtxt('Data/iris_softmax.csv', delimiter=',')
    return model_selection.train_test_split(iris[:, :-3], iris[:, -3:])


class Simple:
    def __init__(self, x_train, x_test, y_train, y_test):
        x = tf.placeholder(tf.float32)
        y = tf.placeholder(tf.float32)

        n_feature = x_train.shape[-1]
        n_class = y_train.shape[-1]

        w = tf.Variable(tf.zeros([n_feature, n_class]))

        z = tf.matmul(x, w)
        hypothesis = tf.nn.softmax(z)

        cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=z, labels=y)
        cost = tf.reduce_mean(cost_i)

        train = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

        # --------------------------- #

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        epochs, batch_size = 15, 5
        loops = len(x_train) // batch_size
        for i in range(epochs):
            total = 0
            for j in range(loops):
                s = j * batch_size
                f = s + batch_size

                _, loss = sess.run([train, cost], {x: x_train[s:f], y: y_train[s:f]})
                total += loss

            indices = np.arange(len(x_train))
            np.random.shuffle(indices)
            x_train = x_train[indices]
            y_train = y_train[indices]

        # --------------------------- #

        self.pred = sess.run(hypothesis, {x: x_test})
        self.y_test = y_test

        sess.close()

    def show_accuracy(self):
        pred_arg = np.argmax(self.pred, axis=1)
        test_arg = np.argmax(self.y_test, axis=1)

        equals = (pred_arg == test_arg)
        print('accuracy :', np.mean(equals))


class Ensenble:
    def __init__(self, count, x_train, x_test, y_train, y_test):
        self.models = [Simple(x_train, x_test, y_train, y_test) for _ in range(count)]
        self.y_test = y_test

    def show_accuracy(self):
        total = np.zeros_like(self.y_test)
        for m in self.models:
            m.show_accuracy()
            total += m.pred
        print('-' * 50)

        pred_arg = np.argmax(total, axis=1)
        test_arg = np.argmax(self.y_test, axis=1)

        equals = (pred_arg == test_arg)
        print('accuracy :', np.mean(equals))


x_train, x_test, y_train, y_test = get_iris()

simple = Simple(x_train, x_test, y_train, y_test)
simple.show_accuracy()
print('-' * 50)

ensenble = Ensenble(7, x_train, x_test, y_train, y_test)
ensenble.show_accuracy()
