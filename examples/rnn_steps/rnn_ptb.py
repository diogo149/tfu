import numpy as np
import tensorflow as tf
import tfu
import du
import du.tasks.nlp_tasks as nlp_tasks

# hyperparameters
NUM_HIDDEN = 512
NUM_EPOCHS = 100
BATCH_SIZE = 128
LENGTH = 128


x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, LENGTH, 50])
y_ = tf.placeholder(tf.float32, shape=[BATCH_SIZE, LENGTH, 50])
x_test = tf.placeholder(tf.float32, shape=[1, 1, 50])
y_test = tf.placeholder(tf.float32, shape=[1, 1, 50])
step = tfu.SimpleRNNStep(NUM_HIDDEN)

with tf.variable_scope("rnn_state_%d" % BATCH_SIZE) as vs:
    rnn_state = step.variable_state(batch_size=BATCH_SIZE,
                                    trainable=False)
with tf.variable_scope("rnn_state_1") as vs:
    test_state = step.variable_state(batch_size=1,
                                     trainable=False)

tfu.add_hook(tfu.inits.set_weight_init(tfu.inits.orthogonal))


def model(x, y_, rnn_state):
    with tf.variable_scope("model"):
        h = x
        h = tf.transpose(h, perm=(1, 0, 2))
        outputs = step.apply_layer("rnn", [h], initial_state=rnn_state)
        h = tf.pack(outputs)
        h = tfu.affine("final_dense", h, num_units=50)
        h = tf.transpose(h, perm=(1, 0, 2))
        y = h

    cross_entropy = tf.reduce_mean(
        tfu.softmax_cross_entropy_with_logits(
            pred_logits=tf.reshape(y, (-1, 50)),
            target=tf.reshape(y_, (-1, 50))))

    update_rnn_state_op = step.update_state_op(rnn_state, outputs[-1])
    return cross_entropy, update_rnn_state_op

with tf.variable_scope("model"):
    cross_entropy, update_rnn_state_op = model(x, y_, rnn_state)

with tf.variable_scope("model", reuse=True):
    cross_entropy_test, update_rnn_state_op_test = model(
        x_test, y_test, test_state)

train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

reset_rnn_state_op = step.reset_state_op(rnn_state)


def train_fn(sess, train_m):
    train_cost, _, _ = sess.run([cross_entropy,
                                 train_step,
                                 update_rnn_state_op],
                                feed_dict={x: train_m['x'],
                                           y_: train_m['y']})
    return [train_cost]


def valid_fn(sess, valid_m):
    valid_cost, _ = sess.run([cross_entropy,
                              update_rnn_state_op],
                             feed_dict={x: valid_m['x'],
                                        y_: valid_m['y']})
    return [valid_cost]


def reset_fn(sess):
    sess.run(reset_rnn_state_op)


def test_fn(sess, test_m):
    test_cost, _ = sess.run([cross_entropy_test,
                             update_rnn_state_op_test],
                            feed_dict={x_test: test_m['x'],
                                       y_test: test_m['y']})
    return [test_cost]


sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

datamaps = nlp_tasks.penn_treebank_char("int32")
datamaps = map(lambda dm: nlp_tasks.one_hot_x(dm, 50, "float32"), datamaps)
datamaps = map(nlp_tasks.add_unsupervised_sequential_y, datamaps)
datamaps = map(nlp_tasks.truncate_to_batch_size,
               datamaps,
               [BATCH_SIZE * LENGTH, BATCH_SIZE * LENGTH, 1])
datamaps = map(nlp_tasks.batch_and_split,
               datamaps,
               [BATCH_SIZE, BATCH_SIZE, 1],
               [LENGTH, LENGTH, 1])
train, valid, test = datamaps

best_valid_cost = 1000


try:
    for i in xrange(1, NUM_EPOCHS + 1):
        with du.timer("epoch"):
            reset_fn(sess)
            train_costs = []
            for train_m in train:
                train_cost, = train_fn(sess, train_m)
                train_costs.append(train_cost)
            reset_fn(sess)
            valid_costs = []
            for valid_m in valid:
                valid_cost, = valid_fn(sess, valid_m)
                valid_costs.append(valid_cost)
            train_cost = np.mean(train_costs) / np.log(2)
            valid_cost = np.mean(valid_costs) / np.log(2)
            log = (i,
                   train_cost,
                   valid_cost)
            print "Epoch: %d\tTrain: %.5g\tValid: %.2f" % log
except KeyboardInterrupt:
    pass
with du.timer("test"):
    test_costs = []
    for test_m in test:
        test_costs.append(test_fn(sess, test_m)[0])
    print "Test: %.5g" % (np.mean(test_costs) / np.log(2))
