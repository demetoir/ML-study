# simple_neural_network 예제에서 MNIST 데이터를 이미 다운 받았으므로 다시 다운 받지 않습니다.
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# x, y_ 플레이스홀더를 지정하고 x 를 28x28x1 크기로 차원을 변경합니다.
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

x_image = tf.reshape(x, [-1, 28, 28, 1])
print("x_image=", x_image)


# 가중치를 표준편차 0.1을 갖는 난수로 초기화하는 함수와 바이어스를 0.1로 초기화하는 함수를 정의합니다.
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# stride는 1로 하고 패딩은 0으로 하는 콘볼루션 레이어를 만드는 함수와 2x2 맥스 풀링 레이어를 위한 함수를 정의합니다.
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 첫번째 콘볼루션 레이어를 만들기 위해 가중치와 바이어스 텐서를 만들고 활성화함수는 렐루 함수를 사용했습니다.
# 그리고 콘볼루션 레이어 뒤에 맥스 풀링 레이어를 추가했습니다.
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# SAME 패딩이므로 콘볼루션으로는 차원이 변경되지 않고 풀링 단계에서 스트라이드에 따라 차원이 반으로 줄어든다.
print(x_image.get_shape())
print(h_conv1.get_shape())
h_pool1.get_shape()

# 두번째 콘볼루션 레이어와 풀링 레이어를 만듭니다.
# 첫번째 콘볼루션의 필터가 32개라 두번째 콘볼루션의 컬러 채널이 32개가 되는 것과 같은 효과가 있습니다.
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# SAME 패딩이므로 콘볼루션으로는 차원이 변경되지 않고 풀링 단계에서 스트라이드에 따라 차원이 반으로 줄어든다.
print(h_conv2.get_shape())
h_pool2.get_shape()

# 마지막 소프트맥스 레이어에 연결하기 위해 완전연결 레이어를 추가합니다.
#  이전 콘볼루션의 레이어의 결과 텐서를 다시 1차원 텐서로 변환하여 렐루 활성화 함수에 전달합니다.
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 드롭아웃되지 않을 확률 값을 저장할 플레이스홀더를 만들고 드롭아웃 레이어를 추가합니다.
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 마지막으로 소프트맥스 레이어를 추가합니다.
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 크로스엔트로피와 최적화알고리즘, 평가를 위한 연산을 정의합니다.
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 세션을 시작하고 변수를 초기화 합니다.
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 20,000번 반복을 수행합니다.
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    #print("step %d, training accuracy %g" % (i, train_accuracy))
    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# 최종 정확도를 출력합니다.
print("test accuracy %g" % sess.run(
    accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))