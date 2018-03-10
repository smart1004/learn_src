# Day_03_07_gan.py
# gan : Generative Adversarial Nets
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import os

mnist = input_data.read_data_sets('mnist', one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
z = tf.placeholder(tf.float32, [None, 128])

G_w1 = tf.Variable(tf.random_normal([128, 256], stddev=0.01))
G_b1 = tf.Variable(tf.zeros([256]))
G_w2 = tf.Variable(tf.random_normal([256, 784], stddev=0.01))
G_b2 = tf.Variable(tf.zeros([784]))

D_w1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
D_b1 = tf.Variable(tf.zeros([256]))
D_w2 = tf.Variable(tf.random_normal([256, 1], stddev=0.01))
D_b2 = tf.Variable(tf.zeros([1]))


def generator(noises):
    hidden = tf.nn.relu(tf.matmul(noises, G_w1) + G_b1)
    output = tf.nn.sigmoid(tf.matmul(hidden, G_w2) + G_b2)
    return output


def discriminator(inputs):
    hidden = tf.nn.relu(tf.matmul(inputs, D_w1) + D_b1)
    output = tf.nn.sigmoid(tf.matmul(hidden, D_w2) + D_b2)
    return output


def get_noises(batch_size, noises):
    return np.float32(np.random.normal(size=[batch_size, noises]))


def make_image(samples, epoch):
    count = samples.shape[0]
    fig, ax = plt.subplots(1, count, figsize=[count, 1])

    for i in range(count):
        ax[i].set_axis_off()
        ax[i].imshow(np.reshape(samples[i], [28, 28]))

    if not os.path.isdir('samples'):
        os.mkdir('samples')

    plt.savefig('samples/{:03}.png'.format(epoch))
    plt.close(fig)


D = discriminator(x)
G = discriminator(generator(z))

loss_D = tf.reduce_mean(tf.log(D) + tf.log(1 - G))
loss_G = tf.reduce_mean(tf.log(G))

opt_D = tf.train.AdamOptimizer(0.0002)
opt_G = tf.train.AdamOptimizer(0.0002)

train_D = opt_D.minimize(-loss_D, var_list=[D_w1, D_b1, D_w2, D_b2])
train_G = opt_G.minimize(-loss_G, var_list=[G_w1, G_b1, G_w2, G_b2])

# ---------------------------- #

sess = tf.Session()
sess.run(tf.global_variables_initializer())

epochs, batch_size = 30, 128
loops = mnist.train.num_examples // batch_size

for i in range(epochs):
    total_D, total_G = 0, 0
    for _ in range(loops):
        xx, _ = mnist.train.next_batch(batch_size)
        noises = get_noises(batch_size, 128)

        _, cost_D = sess.run([train_D, loss_D], feed_dict={x: xx, z: noises})
        _, cost_G = sess.run([train_G, loss_G], feed_dict={z: noises})

        total_D += cost_D
        total_G += cost_G

    print('{:2} : {}, {}'.format(i, total_D / loops, total_G / loops))

    # ---------------------------------- #

    if i % 1 == 0:
        noises = get_noises(10, 128)
        samples = sess.run(generator(noises))

        make_image(samples, i)
