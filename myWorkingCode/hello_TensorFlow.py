import tensorflow as tf
hello = tf.constant('TensorFlow, Hello world!')

sess = tf.Session()

print(sess.run(hello))
