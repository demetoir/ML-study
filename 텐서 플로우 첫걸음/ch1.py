## 텐서플로우 첫걸음 1장

import tensorflow as tf

a = tf.placeholder(dtype="float")
b = tf.placeholder(dtype="float")
c = tf.placeholder(dtype="float")

# x = tf.mul(a, c)
x = tf.multiply(a, c)
y = tf.add(a, b)
# z = tf.sub(b,c)
z = tf.subtract(b, c)

sess = tf.Session()
input_dict = {a: 3, b: 4, c: 5}

fetches = [x, y, z]
print(sess.run(fetches=fetches, feed_dict=input_dict))

sess.close()