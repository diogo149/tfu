import numpy as np
import tensorflow as tf
import tf_utils as tfu
from du.tasks.sequence_tasks import add_task_minibatch

# hyperparameters
NUM_HIDDEN = 32
NUM_EPOCHS = 100
NUM_BATCHES = 20
BATCH_SIZE = 64
LENGTH = 70

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, LENGTH, 2])
y_ = tf.placeholder(tf.float32, shape=[None, 1])


with tf.variable_scope("model",
                       initializer=tf.random_normal_initializer(stddev=0.2)):
    h = x
    h = tf.transpose(h, perm=(1, 0, 2))
    init_state = {"c": tf.zeros(shape=(1, NUM_HIDDEN)),
                  "h": tf.zeros(shape=(1, NUM_HIDDEN))}
    outputs = tfu.rnn_reduce("rnn",
                             tfu.lstm_step,
                             [h],
                             init_state)
    h = outputs[-1]["h"]
    h = tfu.affine("final_dense", h, num_units=1)
    y = h

mse = tf.reduce_mean(tf.square(y - y_))

train_step = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(mse)

# create validation task
v = add_task_minibatch(batch_size=BATCH_SIZE * 25,
                       min_length=LENGTH,
                       max_length=LENGTH,
                       dtype="float32")

sess.run(tf.initialize_all_variables())

# train network
try:
    for i in xrange(1, NUM_EPOCHS + 1):
        costs = []
        for j in xrange(NUM_BATCHES):
            m = add_task_minibatch(batch_size=BATCH_SIZE,
                                   min_length=LENGTH,
                                   max_length=LENGTH,
                                   dtype="float32")
            cost, _ = sess.run([mse, train_step],
                               feed_dict={x: m["x"], y_: m["y"]})
            costs.append(cost)
        valid_cost, = sess.run([mse], feed_dict={x: v["x"], y_: v["y"]})
        print "Epoch: %d\tTrain: %.5g\tValid: %.2f" % (i, np.mean(costs), valid_cost)
except KeyboardInterrupt:
    pass
