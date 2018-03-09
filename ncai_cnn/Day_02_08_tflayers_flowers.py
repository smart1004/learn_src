# Day_02_08_tflayers_flowers.py
import tflearn.datasets.oxflower17 as oxflower17
import tensorflow as tf
from sklearn import model_selection

tf.logging.set_verbosity(tf.logging.INFO)


def make_model(features, is_training):
    input_layer = tf.reshape(features['x'],
                             [-1, 224, 224, 3])

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

    hypothesis = tf.layers.dense(drops, 17)
    return hypothesis


def cnn_model_fn(features, labels, mode):
    logits = make_model(features, mode == tf.estimator.ModeKeys.TRAIN)

    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels,
                                               logits=logits)
        optimizer = tf.train.GradientDescentOptimizer(0.001)
        train_op = optimizer.minimize(loss=loss,
                             global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        # 틀린 코드
        # ops = {'accuracy': tf.metrics.accuracy(labels=labels,
        #                                        predictions=logits)}
        # 맞는 코드
        ops = {'accuracy': tf.metrics.accuracy(
            labels=tf.argmax(labels, 1),
            predictions=tf.argmax(logits, 1))}
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels,
                                               logits=logits)
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          eval_metric_ops=ops)

    predictions = {'classes': tf.argmax(logits, axis=1),
                   'prob:': tf.nn.softmax(logits)}
    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions)


features, labels = oxflower17.load_data(one_hot=True)

data = model_selection.train_test_split(features, labels)
x_train, x_test, y_train, y_test = data
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)
# (1020, 224, 224, 3) (340, 224, 224, 3)
# (1020, 17) (340, 17)

# 문제
# flowers 데이터셋에 대해 동작하도록 수정하세요.

clf = tf.estimator.Estimator(model_fn=cnn_model_fn,
                             model_dir='model/tflearn_flower')

input_fn = tf.estimator.inputs.numpy_input_fn(x={'x': x_train},
                                              y=y_train,
                                              batch_size=100,
                                              num_epochs=None,
                                              shuffle=True)
# clf.train(input_fn=input_fn, steps=1)

input_fn = tf.estimator.inputs.numpy_input_fn(x={'x': x_test},
                                              y=y_test,
                                              num_epochs=2,
                                              shuffle=True)
print(clf.evaluate(input_fn))

input_fn = tf.estimator.inputs.numpy_input_fn(x={'x': x_test},
                                              shuffle=True)
preds = clf.predict(input_fn)

for i, v in enumerate(preds):
    print(i, v)
