# Day_01_01_softmax_iris.py
# ctrl + shift + f10
# alt + 1, alt + 4

from sklearn import datasets, model_selection, preprocessing
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def softmax_1():
    iris = datasets.load_iris()
    print(iris)
    print(iris.keys())
    # dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names'])
    print(iris['data'])
    print(iris['target'])

    onehot = preprocessing.LabelBinarizer().fit_transform(iris['target'])
    # print(onehot)

    data = model_selection.train_test_split(iris['data'],
                                            onehot)
    print(type(data), len(data))
    # <class 'list'> 4

    x_train, x_test, y_train, y_test = data
    print(x_train.shape, x_test.shape)
    print(y_train.shape, y_test.shape)
    # (112, 4) (38, 4)
    # (112, 3) (38, 3)

    x = tf.placeholder(tf.float32)

    w = tf.Variable(tf.zeros([4, 3]))
    b = tf.Variable(tf.zeros([3]))

    # (?, ?) = (112, 4) x (4, 3)
    z = tf.matmul(x, w) + b
    hypothesis = tf.nn.softmax(z)
    cost_i = tf.nn.softmax_cross_entropy_with_logits(labels=y_train,
                                                     logits=z)
    cost = tf.reduce_mean(cost_i)
    train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    sess = tf.Session(config=config)   # sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train, feed_dict={x: x_train})

        if i % 20 == 0:
            print(i, sess.run(cost, feed_dict={x: x_train}))

    prediction = tf.equal(tf.argmax(hypothesis, 1),
                          tf.argmax(y_test, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: x_test}))

    sess.close()


def softmax_2():
    iris = datasets.load_iris()

    data = model_selection.train_test_split(iris['data'],
                                            iris.target)
    x_train, x_test, y_train, y_test = data
    print(x_train.shape, x_test.shape)
    print(y_train.shape, y_test.shape)
    # (112, 4) (38, 4)
    # (112,) (38,)

    x = tf.placeholder(tf.float32)

    w = tf.Variable(tf.zeros([4, 3]))
    b = tf.Variable(tf.zeros([3]))

    # (?, ?) = (112, 4) x (4, 3)
    z = tf.matmul(x, w) + b
    hypothesis = tf.nn.softmax(z)
    cost_i = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y_train, logits=z)
    cost = tf.reduce_mean(cost_i)
    train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        sess.run(train, feed_dict={x: x_train})

        if i % 20 == 0:
            print(i, sess.run(cost, feed_dict={x: x_train}))

    y_hat = sess.run(hypothesis,
                     feed_dict={x: x_test})
    # print(y_hat)

    # 에러.
    # prediction = tf.equal(hypothesis, y_test)
    prediction = tf.equal(tf.argmax(hypothesis, 1), y_test)
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: x_test}))

    sess.close()


softmax_1()
# softmax_2()
