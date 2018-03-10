# Day_02_09_tflayers_vgg.py
# Day_02_08_tflayers_flowers.py
import tflearn.datasets.oxflower17 as oxflower17
import tensorflow as tf
from sklearn import model_selection

tf.logging.set_verbosity(tf.logging.INFO)


def make_model(features, is_training):
    input_layer = tf.reshape(features['x'],
                             [-1, 224, 224, 3])

    # 퀴즈
    # vgg 네트워크로 수정해보세요.

    conv1_1 = tf.layers.conv2d(input_layer, 64, 3,
                               activation=tf.nn.relu, padding='same')
    conv1_2 = tf.layers.conv2d(conv1_1, 64, 3,
                               activation=tf.nn.relu, padding='same')
    pool1 = tf.layers.max_pooling2d(conv1_2, 2, 2)

    conv2_1 = tf.layers.conv2d(pool1, 128, 3,
                               activation=tf.nn.relu, padding='same')
    conv2_2 = tf.layers.conv2d(conv2_1, 128, 3,
                               activation=tf.nn.relu, padding='same')
    pool2 = tf.layers.max_pooling2d(conv2_2, 2, 2)

    conv3_1 = tf.layers.conv2d(pool2, 256, 3,
                               activation=tf.nn.relu, padding='same')
    conv3_2 = tf.layers.conv2d(conv3_1, 256, 3,
                               activation=tf.nn.relu, padding='same')
    conv3_3 = tf.layers.conv2d(conv3_2, 256, 3,
                               activation=tf.nn.relu, padding='same')
    pool3 = tf.layers.max_pooling2d(conv3_3, 2, 2)

    conv4_1 = tf.layers.conv2d(pool3, 512, 3,
                               activation=tf.nn.relu, padding='same')
    conv4_2 = tf.layers.conv2d(conv4_1, 512, 3,
                               activation=tf.nn.relu, padding='same')
    conv4_3 = tf.layers.conv2d(conv4_2, 512, 3,
                               activation=tf.nn.relu, padding='same')
    pool4 = tf.layers.max_pooling2d(conv4_3, 2, 2)

    conv5_1 = tf.layers.conv2d(pool4, 512, 3,
                               activation=tf.nn.relu, padding='same')
    conv5_2 = tf.layers.conv2d(conv5_1, 512, 3,
                               activation=tf.nn.relu, padding='same')
    conv5_3 = tf.layers.conv2d(conv5_2, 512, 3,
                               activation=tf.nn.relu, padding='same')
    pool5 = tf.layers.max_pooling2d(conv5_3, 2, 2)

    flats = tf.layers.flatten(pool5)

    fc1 = tf.layers.dense(flats, 4096, activation=tf.nn.relu)
    drop1 = tf.layers.dropout(fc1, 0.5, is_training)

    fc2 = tf.layers.dense(drop1, 4096, activation=tf.nn.relu)
    drop2 = tf.layers.dropout(fc2, 0.5, is_training)

    hypothesis = tf.layers.dense(drop2, 17)
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


# features, labels = oxflower17.load_data(one_hot=True,
#                                         dirname='17flowers_onehot')
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
                             model_dir='model/tflearn_vgg')

input_fn = tf.estimator.inputs.numpy_input_fn(x={'x': x_train},
                                              y=y_train,
                                              batch_size=100,
                                              num_epochs=None,
                                              shuffle=True)
clf.train(input_fn=input_fn, steps=1)

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
