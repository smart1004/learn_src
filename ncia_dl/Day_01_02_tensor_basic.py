# Day_01_02_tensor_basic.py
import tensorflow as tf


def basic_1():
    a = tf.constant(3)
    b = tf.Variable(5)
    add = tf.add(a, b)
    print(a) #Tensor("Const:0", shape=(), dtype=int32)
    print(b)
    print(add)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # b.initializer.run()

    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(add))

    sess.close()

# basic_1()

def basic_2():
    # 문제
    # 변수 a의 이름을 aa로 고친 다음에 발생하는 에러를 모두 해결하세요.
    aa = tf.placeholder(tf.int32)
    b = tf.placeholder(tf.int32)
    add = aa + b
    print(add)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    feed = {aa: 3, b: 5}
    print(sess.run(add, feed))

    sess.close()


# 문제
# 구구단의 특정 단을 출력하는 함수를 placeholder 버전으로 구현하세요.
def nine_nine(dan):
    left = tf.placeholder(tf.int32)
    right = tf.placeholder(tf.int32)
    multiply = tf.multiply(left, right)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1, 10):
        # print('{} x {} = {:2}'.format(dan, i, dan*i))

        result = sess.run(multiply, {left: dan, right: i})
        print('{} x {} = {:2}'.format(dan, i, result))

    sess.close()
# nine_nine(7)

def nine_nine_adv(dan):
    # left = tf.placeholder(tf.int32) placeholder를 사용할 필요가 없어서 제거함. dan을 바로 사용
    rite = tf.placeholder(tf.int32)    
    multiply = tf.multiply(dan, rite)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1, 10):
        # print('{} x {} = {:2}'.format(dan, i, dan*i))

        result = sess.run(multiply, {rite: i})
        print('{} x {} = {:2}'.format(dan, i, result))

    sess.close()

nine_nine_adv(7)
