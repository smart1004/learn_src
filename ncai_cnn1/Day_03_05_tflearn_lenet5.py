# Day_03_05_tflearn_lenet5.py
import tflearn
from tensorflow.examples.tutorials.mnist import input_data

net = tflearn.input_data([None, 28, 28, 1])

net = tflearn.conv_2d(net, nb_filter=6, filter_size=5, strides=1,
                      padding='VALID', activation='sigmoid')
net = tflearn.avg_pool_2d(net, kernel_size=2, strides=2, padding='SAME')

net = tflearn.conv_2d(net, 6, 5, 1, 'VALID', 'sigmoid')
net = tflearn.avg_pool_2d(net, 2, 2, 'SAME')

net = tflearn.fully_connected(net, n_units=120, activation='sigmoid')
net = tflearn.fully_connected(net, n_units=84, activation='sigmoid')
net = tflearn.fully_connected(net, n_units=10, activation='softmax')

net = tflearn.regression(net, learning_rate=0.001, optimizer='sgd')

# ---------------------------- #

mnist = input_data.read_data_sets('mnist', one_hot=True)

x = mnist.train.images.reshape(-1, 28, 28, 1)
y = mnist.train.labels

x, y = x[:256], y[:256]

model = tflearn.DNN(net)
model.fit(x, y, n_epoch=15, validation_set=0.2, shuffle=True,
          batch_size=128, show_metric=True)

# ------------------------ #

x = mnist.test.images.reshape(-1, 28, 28, 1)
y = mnist.test.labels

x, y = x[:256], y[:256]
print('acc :', model.evaluate(x, y))
