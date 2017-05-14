import tensorflow as tf
import random
import numpy as np

get1DList = lambda shape: [random.randint(1, 10) for _ in range(shape[0])]
get2DList = lambda shape: [[random.randint(1, 10) for _ in range(shape[1])] for _i in range(shape[0])]
printVal = lambda val: print(val.__str__(), "=", val)


# if shape = [3,4] returns [[2, 3, 3, 3], [7, 3, 4, 10], [8, 5, 8, 3]]
def getShapedList(shape):
    def _getShapedList(shape, index):
        if index == len(shape) - 1:
            return [random.randint(1, 10) for i in range(shape[index])]
        return [_getShapedList(shape, index + 1) for i in range(shape[index])]

    return _getShapedList(shape, 0)


def ex_constant():
    def test_constant(value, dtype, shape, name):
        print("value =", value, "dtype =", dtype, "shape =", shape, "name =", name)
        constant = tf.constant(value=value, dtype=dtype, shape=shape, name=name)
        print(constant)

    # each element in exampleList -> [value, dtype, shape, name]
    # name seems must not include " "(space)
    exampleList = [
        [6., tf.float64, [0], "example_1"],
        [6, tf.float16, [1], "example_2"],
        [[1, 2, 3], tf.float64, [3], "example_3"],
        [get2DList([3, 4]), tf.int64, [3, 4], "example_4"],
    ]
    print("example of constant")
    for value, dtype, shape, name in exampleList:
        test_constant(value, dtype, shape, name)
        print()
    print()
    return


def ex_shape():
    def test_shape(shape, value):
        print("shape = ", shape, "value =", value)
        print("tf.shape() : ", tf.shape(value))

    # element of ex_shape_list [shape value]
    ex_shape_list = [
        [0, 5],
        [1, [4]],
        [5, [1, 2, 3, 4, 5]],
        [[1, 5], [1, 2, 3, 4, 5]],
        [[5, 1], [[1], [2], [3], [4], [5]]],
        [[3, 4], [[1, 2, 3, 4], [1, 2, 3, 4], [2, 3, 4, 5], [4, 3, 2, 1]]],
        [[1, 2, 3], [[[1, 2, 3], [3, 2, 1]]]],
    ]
    print("example of shape")
    for shape, value in ex_shape_list:
        test_shape(shape, value)
        print()
    print()
    return


def ex_placeholder():
    def test_placeholder(dtype, shape, name):
        print("dtype =", dtype, "shape =", shape, "name =", name)
        placeholder = tf.placeholder(dtype=dtype, shape=shape, name=name)
        print(placeholder)
        return

    # a = tf.placeholder(dtype=tf.float64, shape=[9], name='a')

    # element of example [dtype, shape, name]
    exampleList = [
        [tf.float64, 0, "example_1"],
        [tf.float64, 1, "example_2"],
        [tf.float64, 2, "example_3"],
        [tf.float64, 3, "example_4"],
        [tf.float64, [1, 2], "example_5"],
        [tf.float64, [2, 4], "example_6"],
    ]
    print("example of placeholder")
    for dtype, shape, name in exampleList:
        test_placeholder(dtype, shape, name)
        print()
    print()
    return


def ex_Variable():
    def test_Variable(initial_value, name, dtype):
        print("initial_valuem =", initial_value, "name =", name, "dtype =", dtype)
        Variable = tf.Variable(initial_value=initial_value, name=name, dtype=dtype)
        print(Variable)
        return

    # [initial_value, name, dtype]
    exampleList = [
        [1, "exmaple_1", tf.float64],
        [[1], "exmaple_2", tf.float64],
        [[1, 2], "exmaple_3", tf.float64],
        [[[1, 2, 3], [4, 5, 6]], "exmaple_4", tf.float64],
        [[[1], [2], [3]], "exmaple_5", tf.float64],
        [[[[1, 2], [3, 4], [5, 6]], [[1, 2], [3, 4], [5, 6]]], "exmaple_6", tf.float64],
    ]
    print("example of Variable")
    for initial_value, name, dtype in exampleList:
        test_Variable(initial_value, name, dtype)
        print()
    print()

    # # Create a variable with a random value.
    # weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),
    #                       name="weights")
    # # Create another variable with the same value as 'weights'.
    # w2 = tf.Variable(weights.initialized_value(), name="w2")
    # # Create another variable with twice the value of 'weights'
    # w_twice = tf.Variable(weights.initialized_value() * 2.0, name="w_twice")

    return


def ex_session():
    x = tf.placeholder(dtype=tf.float64, shape=[2, 2])
    y = tf.placeholder(dtype=tf.float64, shape=[2, 2])
    output = tf.mul(x, y, "output")
    data = {x: [[1, 1], [2, 2]], y: [[2, 2], [3, 3]]}

    # session example
    sess = tf.Session()
    a = sess.run(output, feed_dict=data)
    print(a)
    sess.close()

    # session example using 'with'
    # must use "with tf.device("/cpu:0"):"
    with tf.Session() as sess:
        with tf.device("/gpu:0"):
            a = sess.run(output, feed_dict=data)
            print(a)


# not done yet
def ex_tensor_board():
    # http: // pythonkim.tistory.com / 38
    # https: // www.youtube.com / watch?v = lmrWZPFYjHM
    summary = tf.merge_all_summaries()
    #
    # with tf.Session() as sess:
    #     with tf.device("/gpu:0"):
    #         summary_writer = tf.train.SummaryWriter(FlAGS.train_dir, sess.graph)
    return


def ex_misc():
    # dtype=tf.float64, shape=[2, 3] returns [[0., 0., 0.],[0., 0., 0.]]
    zeros = tf.zeros(dtype=tf.float64, shape=[2, 3])
    reduce_mean = tf.reduce_mean([[1, 2], [3, 4]])
    reduce_sum = tf.reduce_sum([[1, 2], [3, 4]])
    arg_max = tf.arg_max([[1, 2, 3, 4], [8, 7, 6, 5]], 0)
    arg_min = tf.arg_min([[1, 2, 3, 4], [8, 7, 6, 5]], 1)
    linspace = tf.linspace(0, 10, 2)
    range = tf.range(0, 10, 2)
    random_shuffle = tf.random_shuffle([[1, 2, 3], [5, 6, 7]])

    with tf.Session() as sess:
        print(sess.run(zeros))
        print(sess.run(reduce_mean))
        print(sess.run(reduce_sum))
        print(sess.run(arg_max))
        print(sess.run(arg_min))
        print(sess.run(linspace))
        print(sess.run(range))
        print(sess.run(random_shuffle))

    return


# not done yet
def training_MNIST():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # why shape need to be [none, 28*28]?
    # let try just [28*28]
    input_p = tf.placeholder(dtype=tf.float64, shape=[None, 28 * 28], name="input")

    init_op = tf.initialize_all_variables()

    with tf.Session() as sess:
        with tf.device("/gpu:0"):
            sess.run(init_op)
            pass

    return


def training_simple_neural_one_by_one():
    # logical and operation
    data_X1 = [0., 1., 0., 1.]
    data_X2 = [0., 0., 1., 1.]
    data_y = [0, 1, 1, 1]

    X1 = tf.placeholder(tf.float32, None, name="X1")
    X2 = tf.placeholder(tf.float32, None, name="X2")
    y = tf.placeholder(tf.float32, None, name="y")

    dataList = []
    for a, b, c in list(zip(data_X1, data_X2, data_y)):
        dataList += [{X1: a, X2: b, y: c}]

    init_value = tf.random_uniform(shape=[1], name="init_value")
    W1 = tf.Variable(init_value, name="W1")
    W2 = tf.Variable(init_value, name="W2")
    bias = tf.constant(-5.0)

    h = tf.sigmoid(X1 * W1 + X2 * W2 + bias)

    cost = tf.reduce_mean(y - h)
    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

    output = tf.cast(h > 0.5, tf.float32, name="output")

    acc = tf.reduce_mean(tf.cast(tf.equal(y, output), dtype=tf.float32), name="acc")
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        with tf.device("/gpu:0"):
            sess.run(init_op)
            for i in range(1000):
                for data in dataList:
                    sess.run(train_op, feed_dict=data)

                if i % 100 == 0:
                    for data in dataList:
                        # print(output, sess.run(output, feed_dict=data))
                        print(acc, sess.run(acc, feed_dict=data))

    return


def training_simple_neural_tensor_flow_tic():
    tf.set_random_seed(777)
    # logical and operation
    data_X = [
        [0., 0.],
        [1., 0.],
        [0., 1.],
        [1., 1.]
    ]
    data_y = [
        [0.],
        [1.],
        [1.],
        [1.]
    ]

    X = tf.placeholder(tf.float32, [None, 2], name="X")
    y = tf.placeholder(tf.float32, [None, 1], name="y")

    W = tf.Variable(tf.random_normal([2, 1]), name='weight')
    bias = tf.Variable(tf.random_normal([1]), name='bias')

    h = tf.sigmoid(tf.matmul(X, W) + bias, name="h")

    # 여기가 비용함수 계산하는 방식 알아두기
    cost = tf.reduce_mean((y - h) ** 2)
    # cost = -tf.reduce_mean(y * tf.log(h) + (1 - y) * tf.log(1 - h))

    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

    output = tf.cast(h > 0.5, tf.float32, name="output")
    acc = tf.reduce_mean(tf.cast(tf.equal(output, y), dtype=tf.float32), name="acc")

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)

        for step in range(10001):
            sess.run(train_op, feed_dict={X: data_X, y: data_y})
            if step % 100 == 0:
                print(step,
                      sess.run(cost, feed_dict={X: data_X, y: data_y}),
                      sess.run(W))

        # Accuracy report
        hy, o, a = sess.run([h, output, acc],
                            feed_dict={X: data_X, y: data_y})
    print("\nHypothesis: ", hy, "\nCorrect: ", o, "\nAccuracy: ", a)

    return


def training_simple_neural_network():
    # xor opearation
    x_data = [
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1]
    ]
    y_data = [
        [0],
        [1],
        [1],
        [0]
    ]

    # input layer
    X = tf.placeholder(tf.float32, [None, 2], name="X")
    y = tf.placeholder(tf.float32, [None, 1], name="Y")

    # layer 1
    layer1len = 10
    W1 = tf.Variable(tf.random_normal([2, layer1len]), name="W1")
    B1 = tf.Variable(tf.random_normal([layer1len]), name="B1")
    layer1 = tf.sigmoid(tf.matmul(X, W1) + B1, name="layer1")

    # layer 2
    layer2len = 20
    W2 = tf.Variable(tf.random_normal([layer1len, layer2len]), name="W2")
    B2 = tf.Variable(tf.random_normal([layer2len]), name="B2")
    layer2 = tf.sigmoid(tf.matmul(layer1, W2) + B2, name="layer2")

    # layer 3
    layer3len = 30
    W3 = tf.Variable(tf.random_normal([layer2len, layer3len]), name="W3")
    B3 = tf.Variable(tf.random_normal([layer3len]), name="B3")
    layer3 = tf.sigmoid(tf.matmul(layer2, W3) + B3, name="layer3")

    # output layer
    outputLayerLen = 1
    W4 = tf.Variable(tf.random_normal([layer3len, outputLayerLen]), name="W4")
    B4 = tf.Variable(tf.random_normal([outputLayerLen]), name="B4")
    h = tf.sigmoid(tf.matmul(layer3, W4) + B4, name="h")

    # cost function
    cost = -tf.reduce_mean(y * tf.log(h) + (1 - y) * tf.log(1 - h), name="cost")

    # train operation
    learning_rate = 0.01
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    output = tf.cast(h > 0.5, tf.float32, name="output")

    acc = tf.reduce_mean(tf.cast(tf.equal(y, output), dtype=tf.float32), name="acc")

    data = {X: x_data, y: y_data}
    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        with tf.device("/gpu:0"):
            sess.run(init_op)

            for i in range(10001):
                sess.run(train_op, feed_dict=data)

                if i % 100 == 0:
                    print(i, "acc =", sess.run(acc, data), "cost =", sess.run(cost, data))
                    # print(sess.run(h, data))
                    # print(sess.run(y, data))
                    # print(sess.run(output, data))
                    # print()

            print(sess.run(acc, data))
            print(sess.run(cost, data))
            print(sess.run(output, data))
            print(sess.run(y, data))

    return


def training_softmax_regression():
    # http: // pythonkim.tistory.com / 21

    print("train softmax regression")

    # #x0 x1 x2 y[A   B   C]
    # 1   2   1   0   0   1     # C
    # 1   3   2   0   0   1
    # 1   3   4   0   0   1
    # 1   5   5   0   1   0     # B
    # 1   7   5   0   1   0
    # 1   2   5   0   1   0
    # 1   6   6   1   0   0     # A
    # 1   7   7   1   0   0
    # 출처: http: // pythonkim.tistory.com / 21[파이쿵]
    fileName = "softmax_regression_train_set.txt"
    xy = np.loadtxt(fileName, unpack=True, dtype='float32')

    # xy는 6x8. xy[:3]은 3x8. 행렬 곱셈을 하기 위해 미리 transpose.
    x_data = np.transpose(xy[:3])
    y_data = np.transpose(xy[3:])

    print('x_data :', x_data.shape)  # x_data : (8, 3)
    print('y_data :', y_data.shape)  # y_data : (8, 3)

    X = tf.placeholder(tf.float32, [None, 3], name="X")
    Y = tf.placeholder(tf.float32, [None, 3], name="Y")

    # W = tf.Variable(tf.random_uniform([3, 3]), name="W")
    W = tf.Variable(tf.zeros([3, 3]))

    # softmax function
    h = tf.nn.softmax(tf.matmul(X, W), name="h")

    # cost function
    cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(h), reduction_indices=1), name="cost")
    # cost = tf.reduce_mean(tf.reduce_sum(Y - h, reduction_indices=1))

    learning_rate = 0.01
    train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, name="train")

    # final predict output
    output = tf.arg_max(h, 1, name="output")

    # acc = tf.reduce_mean(tf.cast(tf.equal(output, y), dtype=tf.float32), name="acc")
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y, 1), output), dtype=tf.float32), name="acc")

    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        with tf.device("/gpu:0"):

            sess.run(init_op)
            data = {X: x_data, Y: y_data}

            # train step
            for i in range(5000):
                sess.run(train, feed_dict=data)
                if i % 100 == 0:
                    print(i, "acc =", sess.run(acc, data))
                    # print(sess.run(cost, data))
                    # print(sess.run(W, data))
                    # print(sess.run(tf.argmax(Y, 1), data))
                    # print(sess.run(output, data))
                    print()

            # 한번에 여러 개 판단 가능
            data = {X: [[1, 11, 7], [1, 3, 4], [1, 1, 0]]}
            print(sess.run(output, data))
            # d :  ...  [0 1 2]

    return


if __name__ == '__main__':
    # ex_shape()
    # ex_constant()
    # ex_placeholder()
    # ex_Variable()
    # ex_session()
    # ex_misc()


    # traning_simple_neural_onebyone()
    # training_simple_neural_tensorflowtic()
    # training_softmax_regression()
    training_simple_neural_network()

    # 정리하기
    # https: // www.youtube.com / watch?v = 6CCXyfvubvY
    # https: // www.slideshare.net / dahlmoon / 20160623 - 63318427


    pass
