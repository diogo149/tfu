import du

import numpy as np
import tensorflow as tf
import tfu
import tfu.sandbox.batch_normalization as bn

true_dims = 8
train_size = 1024
test_size = 1024
true_hidden_size = 4
true_n_hidden = 1
n_hidden = 2
hidden_size = 512
train_keep_prob = .5
print_frequency = 1
num_epochs = 500
l2_weight = 0
l1_weight = 0

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
h = h.dot(np.random.randn(h.shape[1], 1))
y_true = (h > h.mean()).astype("float32")
y_train = y_true[:train_size]
y_test = y_true[train_size:]

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
            h = tf.nn.relu(h)
            feats.append((name, h))
            if not deterministic:
                h = tf.nn.dropout(h, keep_prob=train_keep_prob)
    with tfu.variable_scope("logit"):
        h = tfu.affine(h, 1)
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


def mlp_train_fn():
    params = tfu.find_variables(variable_scope="mlp")
    with tf.name_scope("mlp_opt"):
        optimizer = tf.train.GradientDescentOptimizer(1e-1)
        grads = tf.gradients(train_cost, params)
        grads_and_vars = list(zip(grads, params))
        mlp_train_step = optimizer.apply_gradients(
            grads_and_vars=grads_and_vars)

    def inner(count):
        costs = []
        for _ in range(count):
            _, cost = sess.run([mlp_train_step, train_cost],
                               feed_dict={x: x_train,
                                          y: y_train})
            costs.append(cost)
        return np.mean(costs)

    return inner


def probe_train_fn():
    params = tfu.find_variables(variable_scope="probe")
    with tf.name_scope("probe_opt"):
        optimizer = tf.train.GradientDescentOptimizer(1e-1)
        grads = tf.gradients(probe_cost, params)
        grads_and_vars = list(zip(grads, params))
        probe_train_step = optimizer.apply_gradients(
            grads_and_vars=grads_and_vars)

    adv_input, = sess.run([last_hidden],
                          feed_dict={x: x_train,
                                     y: y_train})

    def inner(count):
        costs = []
        for _ in range(count):
            _, cost = sess.run([probe_train_step, probe_cost],
                               feed_dict={probe_input: adv_input})
            costs.append(cost)
        return np.mean(costs)

    return inner


def probe_debug():
    res, = sess.run([probe_deterministic],
                    feed_dict={x: x_train})

    print "probe_acc: %.4g" % (np.argmax(res, axis=1) == np.arange(train_size)).mean()


def adv_train_fn():
    params = tfu.find_variables(variable_scope="mlp")
    with tf.name_scope("adv_opt"):
        optimizer = tf.train.GradientDescentOptimizer(1e-1)
        grads = tf.gradients(-probe_reward, params)
        grads_and_vars = list(zip(grads, params))
        adv_train_step = optimizer.apply_gradients(
            grads_and_vars=grads_and_vars)

    def inner(count):
        costs = []
        for _ in range(count):
            _, cost = sess.run([adv_train_step, -probe_reward],
                               feed_dict={x: x_train})
            costs.append(cost)
        return np.mean(costs)

    return inner


def print_status():
    train_cost, train_acc = sess.run([cross_entropy, accuracy],
                                     feed_dict={x: x_train,
                                                y: y_train})
    test_cost, test_acc = sess.run([cross_entropy, accuracy],
                                   feed_dict={x: x_test,
                                              y: y_test})
    print "tr-cost: %.4g tr-acc: %.4g ts-cost: %.4g ts-acc: %.4g" % (
        train_cost,
        train_acc,
        test_cost,
        test_acc,
    )

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

print_status()
if 0:
    fn = mlp_train_fn()
    fn(1000)
    print_status()
    fn = probe_train_fn()
    fn(30000)
    probe_debug()
    fn = adv_train_fn()
    print_status()
    fn(25)
    print_status()
    probe_debug()
