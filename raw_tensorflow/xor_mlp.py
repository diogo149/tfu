import numpy as np
import tensorflow as tf
import tf_utils as tfu

data = {"x": np.array([[0, 0],
                       [0, 1],
                       [1, 0],
                       [1, 1]],
                      dtype="float32"),
        "y": np.array([0, 1, 1, 0], dtype="float32")}

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 2])
y_ = tf.placeholder(tf.float32, shape=[None])

h = x

with tf.variable_scope("mlp",
                       initializer=tf.random_uniform_initializer(-0.05, 0.05)):
    h = tfu.affine("fc", h, 32)
    h = tf.nn.relu(h)
    h = tfu.affine("logit", h, 1)

y = tf.nn.sigmoid(tf.squeeze(h, squeeze_dims=[1]))

cross_entropy = tf.reduce_mean(tfu.binary_cross_entropy(y, y_))
accuracy = tf.reduce_mean(tfu.binary_accuracy(y, y_))

train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

sess.run(tf.initialize_all_variables())

for i in range(1, 100 + 1):
    acc, _ = sess.run([accuracy, train_step],
                      feed_dict={x: data["x"], y_: data["y"]})
    if i % 10 == 0:
        print acc
