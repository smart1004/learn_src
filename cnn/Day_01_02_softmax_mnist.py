# Day_01_02_softmax_mnist.py
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


def mnist_1():
      mnist = input_data.read_data_sets('mnist')

      print(mnist.train.images.shape,
            mnist.train.labels.shape)
      print(mnist.test.images.shape,
            mnist.test.labels.shape)
      print(type(mnist.train.labels))     # <class 'numpy.ndarray'>
      print(mnist.train.labels.dtype)     # uint8

      target_train = np.int32(mnist.train.labels)
      print(target_train.dtype)

      # 문제
      # mnist 데이터셋을 소프트맥스로 학습시키고 결과도 예측해보세요.
      x = tf.placeholder(tf.float32)

      w = tf.Variable(tf.zeros([784, 10]))
      b = tf.Variable(tf.zeros([10]))

      # (55000, 10) = (55000, 784) x (784, 10)
      z = tf.matmul(x, w) + b
      hypothesis = tf.nn.softmax(z)
      cost_i = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target_train, logits=z)
      cost = tf.reduce_mean(cost_i)
      train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

      sess = tf.Session()
      sess.run(tf.global_variables_initializer())

      for i in range(1000):
            sess.run(train, feed_dict={x: mnist.train.images})

            if i % 20 == 0:
                  print(i, sess.run(cost, feed_dict={x: mnist.train.images}))

      target_test = np.int32(mnist.test.labels)

      prediction = tf.equal(tf.argmax(hypothesis, 1), target_test)
      accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
      print(sess.run(accuracy,
                     feed_dict={x: mnist.test.images}))

      sess.close()


# mnist = input_data.read_data_sets('mnist', one_hot=True)
mnist = input_data.read_data_sets('mnist')

print(mnist.train.images.shape,
      mnist.train.labels.shape)
print(mnist.test.images.shape,
      mnist.test.labels.shape)
print(type(mnist.train.labels))     # <class 'numpy.ndarray'>
print(mnist.train.labels.dtype)     # uint8

# target_train = np.int32(mnist.train.labels)
# print(target_train.dtype)

# 문제
# mnist 데이터셋을 소프트맥스로 학습시키고 결과도 예측해보세요.
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.int32)

w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# (100, 10) = (100, 784) x (784, 10)
z = tf.matmul(x, w) + b
hypothesis = tf.nn.softmax(z)
cost_i = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=y, logits=z)
cost = tf.reduce_mean(cost_i)
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

epochs, batch_size = 10, 100
count = mnist.train.num_examples // batch_size
print('count :', count)
for i in range(epochs):
    for _ in range(count):
        x_batch, y_batch = mnist.train.next_batch(batch_size)
        # y_batch = np.asarray(y_batch, dtype=np.int32)
        y_batch = np.int32(y_batch)
        sess.run(train, feed_dict={x: x_batch,
                                   y: y_batch})

target_test = np.int32(mnist.test.labels)

prediction = tf.equal(tf.argmax(hypothesis, 1), target_test)
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print(sess.run(accuracy,
               feed_dict={x: mnist.test.images,
                          y: target_test}))

sess.close()

