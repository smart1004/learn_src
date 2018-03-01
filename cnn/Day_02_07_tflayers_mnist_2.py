# Day_02_07_tflayers_mnist_2.py
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def make_model(features, is_training):
    input_layer = tf.reshape(features['x'],
                             [-1, 28, 28, 1])

    conv1 = tf.layers.conv2d(inputs=input_layer,
                             filters=32,
                             kernel_size=[3, 3],
                             activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                    pool_size=[2, 2],
                                    strides=[2, 2])

    conv2 = tf.layers.conv2d(pool1, 64, [3, 3], activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(conv2, [2, 2], [2, 2])

    flats = tf.layers.flatten(pool2)

    fc1 = tf.layers.dense(flats, 256, activation=tf.nn.relu)
    drops = tf.layers.dropout(fc1, 0.5, is_training)

    hypothesis = tf.layers.dense(drops, 10)
    return hypothesis


def cnn_model_fn(features, labels, mode):
    logits = make_model(features, mode == tf.estimator.ModeKeys.TRAIN)

    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                      logits=logits)
        optimizer = tf.train.GradientDescentOptimizer(0.001)
        train_op = optimizer.minimize(loss=loss,
                             global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        ops = {'accuracy': tf.metrics.accuracy(labels=labels,
                     predictions=tf.argmax(input=logits, axis=1)),
               'my value:': tf.metrics.mean(logits)}
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                      logits=logits)
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          eval_metric_ops=ops)

    predictions = {'classes': tf.argmax(logits, axis=1),
                   'prob:': tf.nn.softmax(logits)}
    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions)


mnist = tf.contrib.learn.datasets.load_dataset('mnist')

x_train, y_train = mnist.train.images, np.int32(mnist.train.labels)
x_test , y_test  = mnist.test .images, np.int32(mnist.test .labels)

clf = tf.estimator.Estimator(model_fn=cnn_model_fn,
                             model_dir='model/tflearn_mnist')

input_fn = tf.estimator.inputs.numpy_input_fn(x={'x': x_train},
                                              y=y_train,
                                              batch_size=100,
                                              num_epochs=None,
                                              shuffle=True)
# clf.train(input_fn=input_fn, steps=1)
# clf.train(input_fn=input_fn, max_steps=5)
# clf.train(input_fn=input_fn, steps=1, max_steps=5)    # error.
# clf.train(input_fn=input_fn)

input_fn = tf.estimator.inputs.numpy_input_fn(x={'x': x_test},
                                              y=y_test,
                                              num_epochs=2,
                                              shuffle=True)
print(clf.evaluate(input_fn))
# {'accuracy': 0.0411, 'loss': 2.3056645, 'global_step': 12}
# {'accuracy': 0.0411, 'loss': 2.3055782, 'my value:': -0.009881937, 'global_step': 12}

input_fn = tf.estimator.inputs.numpy_input_fn(x={'x': x_test},
                                              shuffle=True)
preds = clf.predict(input_fn)
print(preds)
# <generator object Estimator.predict at 0x00000000100761A8>

for i, v in enumerate(preds):
    print(i, v)
    if i >= 3:
        break
# 9999 {'classes': 7,
#       'prob:': array([0.09371038, 0.09363862, 0.09563068, 0.08678868, 0.11049802,
#        0.11340575, 0.08978088, 0.11836119, 0.10365175, 0.09453403],
#       dtype=float32)}

preds = clf.predict(input_fn, predict_keys='classes')
for i, v in enumerate(preds):
    print(i, v)
    if i >= 3:
        break

