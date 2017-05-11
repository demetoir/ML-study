import tensorflow as tf
import random


class TestSet:
    def __init__(self, data, label):
        self.data = data
        self.label = label


shape1D = [3]
shape2D = [3, 3]

# def getTestset(shape):
#     return [[random.randint(1, 10) for i in range(shape[1])] for i in range(shape[0])]

get1DTestset = lambda shape: [random.randint(1, 10) for i in range(shape[0])]
get2DTestset = lambda shape: [[random.randint(1, 10) for i in range(shape[1])] for j in range(shape[0])]


def test():
    shape = shape2D
    a = tf.placeholder(dtype=tf.float64, shape=shape)
    b = tf.placeholder(dtype=tf.float64, shape=shape)
    c = tf.placeholder(dtype=tf.float64, shape=shape)

    x = tf.multiply(a, c)
    y = tf.add(a, b)
    z = tf.subtract(b, c)

    sess = tf.Session()
    input_dict = {a: get1DTestset(shape),
                  b: get1DTestset(shape),
                  c: get1DTestset(shape)}

    fetches = [x, y, z]
    print(sess.run(fetches=x, feed_dict=input_dict))
    print(sess.run(fetches=y, feed_dict=input_dict))
    print(sess.run(fetches=z, feed_dict=input_dict))

    print(sess.run(fetches=fetches, feed_dict=input_dict))
    sess.close()

    pass


# 100 ->10*10
if __name__ == '__main__':
    # tf.constant
    constant = tf.constant(value=6.6, dtype=tf.float64, name="constant", shape=[1])

    # tf,placeholder
    placeholder = tf.placeholder(dtype=tf.float64, shape=[3, 4], name="name")

    # tf.Variable
    Variable = tf.Variable(initial_value=0, dtype=tf.float64, expected_shape=[None, 10])
    # init_op = tf.initialize_all_variables

    # tf.shape, tf.reshape
    testset = get2DTestset(shape2D)
    print(testset)
    print(tf.shape(testset))

    a = tf.placeholder(dtype=tf.float64, shape=[9], name='a')
    print(a)

    b = tf.reshape(a, [3, 3])

    print(b)
    print(tf.shape(b))
    pass
