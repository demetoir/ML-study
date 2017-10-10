# 텐서플로우 첫걸음 4장 단일계층 신경망
#
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

if __name__ == "__main__":
    tf.convert_to_tensor(mnist.train.images).get_shape()

    # 가중치 텐서와 바이어스 텐서를 만듭니다.
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # 훈련 이미지 데이터를 넣을 플레이스홀더와 소프트맥스 텐서를 만듭니다.
    x = tf.placeholder("float", [None, 784])
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # 실제 레이블을 담기위한 텐서와 교차 엔트로피 방식을 이용하는 그래디언트 디센트 방식을 선택합니다.
    y_ = tf.placeholder("float", [None, 10])
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.02).minimize(cross_entropy)

    startTime = time.time()
    # 변수를 초기화하고 세션을 시작합니다.
    sess = tf.Session(config=tf.ConfigProto())
    sess.run(tf.global_variables_initializer())

    # 1000의 반복을 수행하고 결과를 출력합니다. 최종 정확도는 91% 정도 입니다.
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # if i % 100 == 0:
        #   print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

        print(i, sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

    print("total time :", time.time() - startTime)



















