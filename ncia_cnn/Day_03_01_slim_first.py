# Day_03_01_slim_first.py
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
import matplotlib.pyplot as plt


def regression_model(inputs, is_training=True):
    with tf.variable_scope('deep', 'deep_reg', [inputs]):
        net = slim.fully_connected(inputs, 32)
        net = slim.dropout(net, 0.8, is_training=is_training)
        net = slim.fully_connected(net, 16)

        pred = slim.fully_connected(net, 1, activation_fn=None)
        return pred


def produce_batch(batch_size, noise=0.3):
    xs = np.random.random(size=[batch_size, 1]) * 10
    ys = np.sin(xs) + 5 + np.random.normal(size=[batch_size, 1],
                                           scale=noise)
    return np.float32(xs), np.float32(ys)


def save_model(ckpt_dir):
    inputs, targets = produce_batch(200)

    with tf.Graph().as_default():
        preds = regression_model(inputs, is_training=True)
        optimizer = tf.train.AdamOptimizer(0.005)

        loss = tf.losses.mean_squared_error(labels=targets,
                                            predictions=preds)
        train_op = slim.learning.create_train_op(loss, optimizer)

        final_loss = slim.learning.train(train_op,
                                         logdir=ckpt_dir,
                                         number_of_steps=1000)
        print('loss :', final_loss)


def restore_model(ckpt_dir):
    inputs, targets = produce_batch(200)

    with tf.Graph().as_default():
        preds = regression_model(inputs, is_training=False)

        latest = tf.train.latest_checkpoint(ckpt_dir)
        print('latest :', latest)

        sess = tf.Session()

        saver = tf.train.Saver()
        saver.restore(sess, latest)

        y_hat = sess.run(preds)

    plt.plot(inputs, targets, 'ro')
    plt.plot(inputs, y_hat, 'go')
    plt.show()


ckpt_dir = 'model/regression'
# save_model(ckpt_dir)
restore_model(ckpt_dir)

