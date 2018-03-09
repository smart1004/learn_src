# Day_03_06_tflearn_alexnet.py

# 문제
# flowers 데이터셋에 대해서
# alexnet 모델로 구성해보세요.

import tflearn
import tflearn.datasets.oxflower17 as oxflower17

net = tflearn.input_data(shape=[None, 224, 224, 3])

net = tflearn.conv_2d(net, 96, 11, 4, activation='relu')
net = tflearn.local_response_normalization(net)
net = tflearn.max_pool_2d(net, 3, strides=2)

net = tflearn.conv_2d(net, 256, 5, activation='relu')
net = tflearn.local_response_normalization(net)
net = tflearn.max_pool_2d(net, 3, strides=2)

net = tflearn.conv_2d(net, 384, 3, activation='relu')

net = tflearn.conv_2d(net, 384, 3, activation='relu')

net = tflearn.conv_2d(net, 256, 3, activation='relu')
net = tflearn.max_pool_2d(net, 3, strides=2)

net = tflearn.fully_connected(net, 4096, activation='tanh')
net = tflearn.dropout(net, 0.5)

net = tflearn.fully_connected(net, 4096, activation='tanh')
net = tflearn.dropout(net, 0.5)

net = tflearn.fully_connected(net, 17, activation='softmax')
net = tflearn.regression(net)

# ---------------------------- #

x, y = oxflower17.load_data(one_hot=True,
                            dirname='17flowers')
x, y = x[:64], y[:64]
print(y.shape)          # (64, 17)

model = tflearn.DNN(net)
model.fit(x, y, n_epoch=1, validation_set=0.1,
          shuffle=True, batch_size=64, show_metric=True)

