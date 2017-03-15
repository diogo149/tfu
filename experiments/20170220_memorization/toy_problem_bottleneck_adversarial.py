import du

import numpy as np
import tensorflow as tf
import tfu
import tfu.sandbox.batch_normalization as bn

true_dims = 8
fake_dims = 8 - true_dims
output_dim = 32
train_size = 1024
test_size = 1024
true_hidden_size = 1
true_n_hidden = 1
n_hidden = 2
hidden_size = 64
train_keep_prob = .5
print_frequency = 1
num_epochs = 500
l2_weight = 0
l1_weight = 0
mlp_learning_rate = 3e-3

_epoch_idx = tf.Variable(0.)
progress = 1 - (num_epochs - _epoch_idx) / (num_epochs - 1)

probe_learning_rate = 1e-3
probe_initial_steps = 20
probe_update_steps = 5
adv_learning_rate = mlp_learning_rate * 1. * (1 - progress)
adv_learning_rate = mlp_learning_rate * .2
adv_learning_rate = 0
adv_update_steps = 1
probe_keep_prob = 0.5
probe_hiddens = 0
probe_hidden_size = 64
probe_dim = 1024

np.random.seed(149)
dims = true_dims + train_size
x_size = (train_size + test_size, true_dims)
if 1:
    tmp = np.random.uniform(low=-1, high=1, size=x_size).astype("float32")
if 0:
    tmp = np.random.randint(2, size=x_size).astype("float32")
assert train_size == test_size
x_train = np.concatenate([tmp[:train_size], np.eye(train_size)], axis=1)
x_test = np.concatenate([tmp[-test_size:], np.eye(test_size)], axis=1)
# make a random neural network
h = np.concatenate((x_train, x_test))
h = h[:, :true_dims]
for _ in range(true_n_hidden):
    h = h.dot(np.random.randn(h.shape[1], true_hidden_size))
    h += np.random.randn(true_hidden_size)
    h = np.maximum(h, h * 0.3)
h = h.dot(np.random.randn(h.shape[1], output_dim))
y_true = (h > h.mean()).astype("float32")
y_train = y_true[:train_size]
y_test = y_true[train_size:]

with tf.device('/cpu:0'):
    x = tf.placeholder(tf.float32, shape=[None, dims])
    y = tf.placeholder(tf.float32, shape=[None, output_dim])

    tfu.add_hook(tfu.inits.set_weight_init(tfu.inits.msr_normal))

    def mlp(deterministic):
        feats = []
        h = x
        for i in range(1, n_hidden + 1):
            name = "fc%d" % i
            with tfu.variable_scope(name):
                h = tfu.affine(h, hidden_size)
                h = tf.nn.relu(h)
                feats.append((name, h))
                if not deterministic:
                    h = tf.nn.dropout(h, keep_prob=train_keep_prob)
        with tfu.variable_scope("logit"):
            h = tfu.affine(h, output_dim)
            feats.append(("logit", h))
        raw_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=h,
                                                                    labels=y)
        cross_entropy = tf.reduce_mean(raw_cross_entropy)
        accuracy = tf.reduce_mean(tfu.binary_accuracy(h, y))
        return feats, raw_cross_entropy, cross_entropy, accuracy

    with tfu.variable_scope("mlp"):
        _, _, train_ce, _ = mlp(False)
    with tfu.variable_scope("mlp", reuse=True):
        feats, raw_cross_entropy, cross_entropy, accuracy = mlp(True)
        last_hidden = feats[-2][1]

    train_cost = train_ce
    weights = tfu.find_variables(weight=True)
    if l2_weight != 0:
        for w in weights:
            train_cost += l2_weight * tf.reduce_sum(tf.square(w))
    if l1_weight != 0:
        for w in weights:
            train_cost += l1_weight * tf.reduce_sum(abs(w))

    def probe(h, deterministic):
        for i in range(1, probe_hiddens + 1):
            with tfu.variable_scope("probe%d" % i):
                h = tfu.affine(h, probe_hidden_size)
                # h = tf.nn.relu(h)
                # leaky relu
                h = tf.maximum(h, h * 0.1)
                if not deterministic:
                    h = tf.nn.dropout(h, keep_prob=probe_keep_prob)
        h = tfu.linear(h, probe_dim)
        return h

    with tfu.variable_scope("probe"):
        probe_deterministic = probe(last_hidden, deterministic=True)
    with tfu.variable_scope("probe", reuse=True):
        probe_input = tf.placeholder(tf.float32,
                                     shape=tfu.get_shape_values(last_hidden))
        probe_stochastic = probe(probe_input, deterministic=False)

    if 0:
        probe_target = tf.constant(
            np.random.randint(2, size=(train_size, probe_dim)).astype("float32"))
        probe_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=probe_stochastic,
                                                                            labels=probe_target))
        if 0:
            probe_reward = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=probe_deterministic,
                                                                                  labels=probe_target))
        if 0:
            probe_reward = -tf.reduce_mean(tf.square(probe_deterministic))

        if 1:
            probe_reward = -tf.reduce_mean(tf.abs(probe_deterministic))

    if 1:
        assert probe_dim == train_size
        probe_target = tf.range(train_size, dtype=tf.int64)
        probe_cost = tf.reduce_mean(tfu.softmax_cross_entropy_with_logits(probe_stochastic,
                                                                          probe_target))
        probe_reward = tf.reduce_mean(tfu.softmax_cross_entropy_with_logits(probe_deterministic,
                                                                            probe_target))

    params = tfu.find_variables(variable_scope="mlp")
    with tf.name_scope("mlp_opt"):
        optimizer = tf.train.RMSPropOptimizer(learning_rate=mlp_learning_rate)
        grads = tf.gradients(train_cost, params)
        grads_and_vars = list(zip(grads, params))
        mlp_train_step = optimizer.apply_gradients(
            grads_and_vars=grads_and_vars)

    params = tfu.find_variables(variable_scope="probe")
    with tf.name_scope("probe_opt"):
        optimizer = tf.train.RMSPropOptimizer(learning_rate=probe_learning_rate)
        grads = tf.gradients(probe_cost, params)
        grads_and_vars = list(zip(grads, params))
        probe_train_step = optimizer.apply_gradients(
            grads_and_vars=grads_and_vars)

    params = tfu.find_variables(variable_scope="mlp")
    with tf.name_scope("adv_opt"):
        optimizer = tf.train.RMSPropOptimizer(learning_rate=adv_learning_rate)
        grads = tf.gradients(-probe_reward, params)
        grads_and_vars = list(zip(grads, params))
        adv_train_step = optimizer.apply_gradients(
            grads_and_vars=grads_and_vars)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch_idx in range(1, num_epochs + 1):
            sess.run(_epoch_idx.assign_add(1.))

            if adv_learning_rate != 0:
                # adversarial section
                # ====
                # get training features
                adv_input, = sess.run([last_hidden],
                                      feed_dict={x: x_train,
                                                 y: y_train})

                for _ in range(probe_update_steps
                               if epoch_idx != 1
                               else probe_initial_steps):
                    sess.run([probe_train_step, probe_cost],
                             feed_dict={probe_input: adv_input})

                for _ in range(adv_update_steps):
                    sess.run([adv_train_step, probe_reward],
                             feed_dict={x: x_train})
                # ====

            sess.run([mlp_train_step],
                     feed_dict={x: x_train,
                                y: y_train})

            train_cost, train_acc = sess.run([cross_entropy, accuracy],
                                             feed_dict={x: x_train,
                                                        y: y_train})
            test_cost, test_acc = sess.run([cross_entropy, accuracy],
                                           feed_dict={x: x_test,
                                                      y: y_test})

            if (epoch_idx % print_frequency) == 0:
                print "Epoch %03d tr-cost: %.4g tr-acc: %.4g ts-cost: %.4g ts-acc: %.4g" % (
                    epoch_idx,
                    train_cost,
                    train_acc,
                    test_cost,
                    test_acc,
                )
