# Day_02_02_lenet5.py
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('mnist', one_hot=True)

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# ------------------------------------------------------------ #
# 변수 생성

# [문제 1] 변수 생성에 들어가는 숫자를 채우세요.
w1 = tf.Variable(tf.truncated_normal([5, 5, 1, 6], stddev=0.1))
b1 = tf.Variable(tf.zeros([6]))

w2 = tf.Variable(tf.truncated_normal([5, 5, 6, 16], stddev=0.1))
b2 = tf.Variable(tf.ones([16]))

w3 = tf.Variable(tf.truncated_normal([400, 120], stddev=0.1))
b3 = tf.Variable(tf.ones([120]))

w4 = tf.Variable(tf.truncated_normal([120, 84], stddev=0.1))
b4 = tf.Variable(tf.ones([84]))

w5 = tf.Variable(tf.truncated_normal([84, 10], stddev=0.1))
b5 = tf.Variable(tf.ones([10]))

# ------------------------------------------------------------ #
# 모델 구성

# [문제 2] padding을 'SAME'과 'VALID' 중에서 하나로 채우세요
l1_conv = tf.nn.conv2d(x, w1, [1, 1, 1, 1], padding='SAME')
l1_actv = tf.sigmoid(l1_conv + b1)
l1_pool = tf.nn.avg_pool(l1_actv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

l2_conv = tf.nn.conv2d(l1_pool, w2, [1, 1, 1, 1], padding='VALID')
l2_actv = tf.sigmoid(l2_conv + b2)
l2_pool = tf.nn.avg_pool(l2_actv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

# [문제 3] 컨볼루션에서 FC로 넘어가는 아래 숫자를 채우세요.
fc_layer = tf.reshape(l2_pool, [-1, 400])

l3_fccd = tf.matmul(fc_layer, w3) + b3
l3_actv = tf.nn.sigmoid(l3_fccd)

l4_fccd = tf.matmul(l3_actv, w4) + b4
l4_actv = tf.nn.sigmoid(l4_fccd)

l5_fccd = tf.matmul(l4_actv, w5) + b5
hypothesis = tf.nn.softmax(l5_fccd)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=l5_fccd, labels=y)
cost = tf.reduce_mean(cost_i)

optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(cost)

# ------------------------------------------------------------ #

sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 128
for step in range(10001):
    xx, yy = mnist.train.next_batch(batch_size)

    # [문제 4] 4차원 배열에 들어가는 숫자를 채우세요.
    xx = xx.reshape(-1, 28, 28, 1)

    sess.run(train, feed_dict={x: xx, y: yy})

    if step % 100 == 0:
        c = sess.run(cost, feed_dict={x: xx, y: yy})
        print('{:-4} : {}'.format(step, c))

# ------------------------------------------------------------ #

# [문제 5] mnist 데이터셋에 포함된 test 셋에 대해서 정확도를 알려주세요.
x_images = mnist.test.images.reshape(-1, 28, 28, 1)
prediction = tf.equal(tf.argmax(hypothesis, 1),
                      tf.argmax(mnist.test.labels, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print(sess.run(accuracy,
               feed_dict={x: x_images}))

sess.close()
