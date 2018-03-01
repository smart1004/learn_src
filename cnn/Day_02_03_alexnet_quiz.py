# Day_02_03_alexnet_quiz.py
import numpy as np
import tensorflow as tf
import tflearn.datasets.oxflower17 as oxflower17

# def load_data(dirname="17flowers_onehot", resize_pics=(224, 224), shuffle=True, one_hot=False):
# 데이터셋 크기 : (1360, 224, 224, 3), (1360, 17)
features, labels = oxflower17.load_data(one_hot=True)

x_train, x_test = features[:1000], features[1000:]
y_train, y_test = labels[:1000], labels[1000:]

num_labels = 17
batch_size = 32

# [문제 1] shape을 채우세요.
# 일단 여기서 None 대신 batch_size로 대신한다.
x = tf.placeholder(tf.float32, shape=(batch_size, 0, 0, 0))
y = tf.placeholder(tf.float32, shape=(batch_size, 0))

# ------------------------------------------------------------ #
# 변수 생성

# [문제 2] 아래 숫자들을 채우세요.
w1 = tf.Variable(tf.truncated_normal([0, 0, 0, 0], stddev=0.1))
b1 = tf.Variable(tf.zeros([0]))

w2 = tf.Variable(tf.truncated_normal([0, 0, 0, 0], stddev=0.1))
b2 = tf.Variable(tf.ones([0]))

w3 = tf.Variable(tf.truncated_normal([0, 0, 0, 0], stddev=0.1))
b3 = tf.Variable(tf.zeros([0]))

w4 = tf.Variable(tf.truncated_normal([0, 0, 0, 0], stddev=0.1))
b4 = tf.Variable(tf.ones([0]))

w5 = tf.Variable(tf.truncated_normal([0, 0, 0, 0], stddev=0.1))
b5 = tf.Variable(tf.zeros([0]))

w6 = tf.Variable(tf.truncated_normal([0, 0], stddev=0.1))
b6 = tf.Variable(tf.ones([0]))

w7 = tf.Variable(tf.truncated_normal([0, 0], stddev=0.1))
b7 = tf.Variable(tf.ones([0]))

w8 = tf.Variable(tf.truncated_normal([0, 0], stddev=0.1))
b8 = tf.Variable(tf.ones([0]))

# ------------------------------------------------------------ #
# 모델 구성

l1_conv = tf.nn.conv2d(x, w1, [0, 0, 0, 0], padding='')
l1_relu = tf.nn.relu(l1_conv + b1)
l1_pool = tf.nn.max_pool(l1_relu, [0, 0, 0, 0], [0, 0, 0, 0], padding='')
l1_norm = tf.nn.local_response_normalization(l1_pool)

l2_conv = tf.nn.conv2d(l1_pool, w2, [0, 0, 0, 0], padding='')
l2_relu = tf.nn.relu(l2_conv + b2)
l2_pool = tf.nn.max_pool(l2_relu, [0, 0, 0, 0], [0, 0, 0, 0], padding='')
l2_norm = tf.nn.local_response_normalization(l2_pool)

l3_conv = tf.nn.conv2d(l2_pool, w3, [0, 0, 0, 0], padding='')
l3_relu = tf.nn.relu(l3_conv + b3)

l4_conv = tf.nn.conv2d(l3_relu, w4, [0, 0, 0, 0], padding='')
l4_relu = tf.nn.relu(l4_conv + b4)

l5_conv = tf.nn.conv2d(l4_relu, w5, [0, 0, 0, 0], padding='')
l5_relu = tf.nn.relu(l5_conv + b5)
l5_pool = tf.nn.max_pool(l5_relu, [0, 0, 0, 0], [0, 0, 0, 0], padding='')

# [문제 3] 아래 내용을 채우세요.
fc_layer = tf.reshape(l5_pool, [0, 0])

l6_fccd = tf.matmul(fc_layer, w6) + b6
l6_relu = tf.nn.tanh(l6_fccd)
l6_drop = tf.nn.dropout(l6_relu, 0.5)

l7_fccd = tf.matmul(l6_drop, w7) + b7
l7_relu = tf.nn.tanh(l7_fccd)
l7_drop = tf.nn.dropout(l7_relu, 0.5)

hypothesis = tf.matmul(l7_drop, w8) + b8

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=y)
cost = tf.reduce_mean(cost_i)

# ------------------------------------------------------------ #

optimizer = tf.train.AdamOptimizer(0.0001)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

indices = np.arange(len(x_train))
epoches = 15
for i in range(epoches):
    total = 0
    count = len(x_train) // batch_size
    for j in range(count):
        n1 = j * batch_size
        n2 = n1 + batch_size

        xx = x_train[n1:n2]
        yy = y_train[n1:n2]

        c, _ = sess.run([cost, train], feed_dict={x: xx, y: yy})
        total += c
        # print(i, count, c)

    print('{} : {}'.format(i, total / count))

    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]

# ------------------------------------------------------------ #

# [문제 4] 테스트셋에 대해서 정확도를 알려주세요.

sess.close()