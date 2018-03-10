# Day_03_04_save_model.py
import tensorflow as tf


def save_model():
    xx = [1, 2, 3]
    y = [1, 2, 3]

    x = tf.placeholder(tf.float32)

    w = tf.Variable(tf.zeros([1]))
    b = tf.Variable(tf.zeros([1]))

    hypothesis = w * x + b
    cost = tf.reduce_mean((hypothesis - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    for i in range(10):
        sess.run(train, feed_dict={x: xx})
        saver.save(sess=sess, save_path='model/second',
                   global_step=i)

    # saver.save(sess=sess, save_path='model/first')
    sess.close()


def restore_model():
    x = tf.placeholder(tf.float32)

    w = tf.Variable(tf.zeros([1]))
    b = tf.Variable(tf.zeros([1]))

    hypothesis = w * x + b

    sess = tf.Session()

    latest = tf.train.latest_checkpoint('model')
    print(latest)

    saver = tf.train.Saver()
    saver.restore(sess=sess, save_path=latest)

    print(sess.run(hypothesis, {x: [5, 7]}))
    # [4.653379 6.399143]

    sess.close()


def restore_model_adv():
    w = tf.Variable([0.])
    b = tf.Variable([0.])

    hypothesis = w * [5, 7] + b

    sess = tf.Session()

    latest = tf.train.latest_checkpoint('model')
    print(latest)

    saver = tf.train.Saver()
    saver.restore(sess=sess, save_path=latest)

    print(sess.run(hypothesis))
    # [4.653379 6.399143]

    sess.close()


# save_model()
# restore_model()
restore_model_adv()




