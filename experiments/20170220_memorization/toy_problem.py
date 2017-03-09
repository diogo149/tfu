import du

import numpy as np
import tensorflow as tf
import tfu
import tfu.sandbox.batch_normalization as bn

true_dims = 8
fake_dims = 8 - true_dims
train_size = 1280
test_size = 1280
true_n_hidden = 0
true_hidden_size = 4
n_hidden = 2
hidden_size = 128
train_keep_prob = 0.5
use_bn = False
print_frequency = 10
num_epochs = 500
l2_weight = 1e-4
l1_weight = 1e-4
output = "binary"
mlp_learning_rate = 1e-3

_epoch_idx = tf.Variable(0.)
progress = 1 - (num_epochs - _epoch_idx) / (num_epochs - 1)

np.random.seed(149)
dims = true_dims + fake_dims
x_train = np.random.uniform(low=-1,
                            high=1,
                            size=(train_size, dims)).astype("float32")
x_test = np.random.uniform(low=-1,
                           high=1,
                           size=(test_size, dims)).astype("float32")

# make a random neural network
h = np.concatenate((x_train, x_test))
h = h[:, :true_dims]
for _ in range(true_n_hidden):
    h = h.dot(np.random.randn(h.shape[1], true_hidden_size))
    h += np.random.randn(true_hidden_size)
    h = np.maximum(h, h * 0.3)
h = h.dot(np.random.randn(h.shape[1], 1))
if output == "binary":
    y_true = (h > h.mean()).astype("float32")
elif output == "real":
    y_true = ((h - h.mean()) / h.std()).astype("float32")
y_train = y_true[:train_size]
y_test = y_true[train_size:]

with tf.device('/cpu:0'):
    x = tf.placeholder(tf.float32, shape=[None, dims])
    y = tf.placeholder(tf.float32, shape=[None, 1])

    tfu.add_hook(tfu.inits.set_weight_init(tfu.inits.msr_normal))

    def mlp(deterministic):
        feats = []
        h = x
        for i in range(1, n_hidden + 1):
            name = "fc%d" % i
            with tfu.variable_scope(name):
                h = tfu.affine(h, hidden_size)
                if use_bn:
                    h = bn.ema_batch_normalization(h,
                                                   use_batch_stats=not deterministic)
                h = tf.nn.relu(h)
                feats.append((name, h))
                if not deterministic:
                    h = tf.nn.dropout(h, keep_prob=train_keep_prob)
        with tfu.variable_scope("logit"):
            h = tfu.affine(h, 1)
            if use_bn:
                h = bn.ema_batch_normalization(h,
                                               use_batch_stats=not deterministic)
            feats.append(("logit", h))
        if output == "binary":
            raw_cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=h,
                                                               labels=y)
        elif output == "real":
            raw_cost = tf.square(h - y)
        cost = tf.reduce_mean(raw_cost)
        return feats, raw_cost, cost

    with tfu.variable_scope("mlp"):
        _, _, train_cost = mlp(False)
    with tfu.variable_scope("mlp", reuse=True):
        feats, raw_cost, cost = mlp(True)
        last_hidden = feats[-2][1]

    weights = tfu.variables(weight=True)
    if l2_weight != 0:
        for w in weights:
            train_cost += l2_weight * tf.reduce_sum(tf.square(w))
    if l1_weight != 0:
        for w in weights:
            train_cost += l1_weight * tf.reduce_sum(abs(w))

    train_step = tf.train.RMSPropOptimizer(
        learning_rate=mlp_learning_rate,
    ).minimize(train_cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch_idx in range(1, num_epochs + 1):
            sess.run(_epoch_idx.assign_add(1.))

            sess.run([train_step],
                     feed_dict={x: x_train,
                                y: y_train})

            if (epoch_idx % print_frequency) == 0:
                train_cost,  = sess.run([cost],
                                        feed_dict={x: x_train,
                                                   y: y_train})
                test_cost, = sess.run([cost],
                                      feed_dict={x: x_test,
                                                 y: y_test})

                print "Epoch %03d tr-cost: %.4g ts-cost: %.4g" % (
                    epoch_idx,
                    train_cost,
                    test_cost,
                )
