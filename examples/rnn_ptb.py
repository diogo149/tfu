import numpy as np
import tensorflow as tf
import tfu
import du
import du.tasks.nlp_tasks as nlp_tasks

# hyperparameters
NUM_HIDDEN = 32
NUM_EPOCHS = 100
BATCH_SIZE = 128
LENGTH = 128

x = tf.placeholder(tf.float32, shape=[None, LENGTH, 50])
y_ = tf.placeholder(tf.float32, shape=[None, LENGTH, 50])

with tf.variable_scope("model",
                       initializer=tf.random_uniform_initializer(-0.05, 0.05)):
    h = x
    h = tf.transpose(h, perm=(1, 0, 2))
    rnn_state = tf.get_variable('rnn_state',
                                shape=[1, NUM_HIDDEN],
                                validate_shape=False,
                                trainable=False)
    init_state = tf.identity(rnn_state)
    init_state.set_shape([None, NUM_HIDDEN])
    tfu.add_hook(tfu.inits.set_weight_init(tfu.inits.orthogonal))
    outputs = tfu.rnn_reduce("rnn",
                             tfu.simple_rnn_step,
                             [h],
                             init_state)
    h = tf.pack(outputs)
    h = tfu.affine("final_dense", h, num_units=50)
    h = tf.transpose(h, perm=(1, 0, 2))
    y = h

cross_entropy = tf.reduce_mean(
    tfu.softmax_cross_entropy_with_logits(
        pred_logits=tf.reshape(y, (-1, 50)),
        target=tf.reshape(y_, (-1, 50))))


train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

reset_rnn_state_op = tf.assign(rnn_state,
                               tf.zeros([1, NUM_HIDDEN]),
                               validate_shape=False)

with tf.control_dependencies([cross_entropy]):
    update_rnn_state_op = tf.assign(rnn_state,
                                    outputs[-1],
                                    validate_shape=False)

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


def train_fn(train_m):
    train_cost, _, _ = sess.run([cross_entropy,
                                 train_step,
                                 update_rnn_state_op],
                                feed_dict={x: train_m['x'],
                                           y_: train_m['y']})
    return [train_cost]


def valid_fn(valid_m):
    valid_cost, _ = sess.run([cross_entropy,
                              update_rnn_state_op],
                             feed_dict={x: valid_m['x'],
                                        y_: valid_m['y']})
    return [valid_cost]


try:
    for i in xrange(1, NUM_EPOCHS + 1):
        sess.run(reset_rnn_state_op)
        train_costs = []
        for train_m in train:
            train_cost, = train_fn(train_m)
            train_costs.append(train_cost)
        sess.run(reset_rnn_state_op)
        valid_costs = []
        for valid_m in valid:
            valid_cost, = valid_fn(valid_m)
            valid_costs.append(valid_cost)
        train_cost = np.mean(train_costs) / np.log(2)
        valid_cost = np.mean(valid_costs) / np.log(2)
        log = (i,
               train_cost,
               valid_cost)
        print "Epoch: %d\tTrain: %.5g\tValid: %.2f" % log

except KeyboardInterrupt:
    pass
sess.run(reset_rnn_state_op)
test_costs = []
for test_m in test:
    test_costs.append(valid_fn(test_m)[0])
print "Test: %.5g" % (np.mean(test_costs) / np.log(2))
