# 텐서플로우 첫걸음 2장 선형 회귀
# 지도학습

# matplotlib/..../font_manager.py 수정
# http://matplotlib.1069221.n5.nabble.com/Error-on-import-matplotlib-pyplot-on-Anaconda3-for-Windows-10-Home-64-bit-PC-td46477.html


# matplotlib pyplot_tutorial
# http://matplotlib.org/users/pyplot_tutorial.html


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# y = W * x + b 인 데이터 셋 생성
def make_test_set(W, b):
    num_point = 1000
    vectors_set = []
    for i in range(num_point):
        x1 = np.random.normal(1, 1)
        y1 = x1 * W + b + np.random.normal(1, 1)
        vectors_set.append([x1, y1])

    x_data = [v[0] for v in vectors_set]
    y_data = [v[1] for v in vectors_set]
    return x_data, y_data


# 데이터셋의 좌표를 그린다
def show_test_set(x_data, y_data):
    dot_size = 1
    plt.scatter(x_data, y_data, s=dot_size)
    plt.legend()
    plt.show()


# 데이터셋과 train 비교를 좌표로 그린다
def show_train_data(x_data, y_data, W, b):
    plt.plot(x_data, y_data, 'ro')
    plt.plot(x_data, W * x_data + b)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


# 선형 회귀를 경사 하강법으로 시행함
def linear_regression(x_data, y_data):
    TRAIN_CONSTANT = 0.1
    TRAIN_SIZE = 100

    W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    b = tf.Variable(tf.zeros([1]))
    y = W * x_data + b

    #거리기반의 비용함수 생성
    loss = tf.reduce_mean(tf.square(y - y_data))

    # 최적화함수를 경사하강법으로 생성
    optimizer = tf.train.GradientDescentOptimizer(TRAIN_CONSTANT)
    train = optimizer.minimize(loss)

    init = tf.initialize_all_variables()

    sess = tf.Session()
    s_run = sess.run
    s_run(init)

    for step in range(TRAIN_SIZE):
        s_run(train)
        print(s_run(W), s_run(b))

    ret_W, ret_b = s_run(W), s_run(b)
    sess.close()

    return ret_W, ret_b


CONSTANT_W = 10
CONSTANT_B = 3

if __name__ == "__main__":
    x_data, y_data = make_test_set(CONSTANT_W, CONSTANT_B)
    show_test_set(x_data, y_data)
    W, b = linear_regression(x_data, y_data)
    show_train_data(x_data, y_data, W, b)
