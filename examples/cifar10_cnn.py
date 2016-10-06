import numpy as np
import tensorflow as tf
import tfu
import du

train, valid, test = du.tasks.image_tasks.cifar10(x_dtype="float32",
                                                  y_dtype="int64")
for dataset in [train, valid, test]:
    dataset["x"] = dataset["x"].transpose(0, 2, 3, 1)

sess = tf.Session()

x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y_ = tf.placeholder(tf.int64, shape=[None])

h = x

tfu.add_hook(tfu.hooks.default_kwargs_dsl(kwargs={"filter_size": (3, 3)},
                                          key="conv2d"))
tfu.add_hook(tfu.hooks.default_kwargs_dsl(kwargs={"ksize": (3, 3),
                                                  "strides": (2, 2)},
                                          key="max_pool2d"))
tfu.add_hook(tfu.inits.set_weight_init(tfu.inits.orthogonal))

with tf.variable_scope("mlp"):
    h = tfu.conv2d(h, num_filters=96, name="conv1")
    h = tf.nn.relu(h)
    h = tfu.conv2d(h, num_filters=96, name="conv2")
    h = tf.nn.relu(h)
    h = tfu.max_pool2d(h)
    h = tfu.conv2d(h, num_filters=192, name="conv3")
    h = tf.nn.relu(h)
    h = tfu.conv2d(h, num_filters=192, name="conv4")
    h = tf.nn.relu(h)
    h = tfu.conv2d(h, num_filters=192, name="conv5")
    h = tf.nn.relu(h)
    h = tfu.max_pool2d(h)
    h = tfu.conv2d(h, num_filters=192, name="conv6")
    h = tf.nn.relu(h)
    h = tfu.conv2d(h, num_filters=192, filter_size=(1, 1), name="conv7")
    h = tf.nn.relu(h)
    h = tfu.conv2d(h, num_filters=10, filter_size=(1, 1), name="conv8")
    h = tfu.global_avg_pool2d(h)

cross_entropy = tf.reduce_mean(tfu.softmax_cross_entropy_with_logits(h, y_))

train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

accuracy = tf.reduce_mean(tfu.categorical_accuracy(h, y_))

sess.run(tf.initialize_all_variables())


def to_minibatches(dataset, batch_size):
    epoch_size = len(dataset.values()[0])
    while True:
        res = {k: [] for k in dataset}
        for _ in range(batch_size):
            idx = np.random.randint(epoch_size)
            for k, v in dataset.items():
                res[k].append(v[idx])
        res = {k: np.array(v) for k, v in res.items()}
        yield res


def split_input(dataset, batch_size):
    epoch_size = len(dataset.values()[0])
    split = []
    for idx in range(int(np.ceil(float(epoch_size) / batch_size))):
        tmp = {}
        for k, v in dataset.items():
            tmp[k] = v[batch_size * idx: batch_size * (idx + 1)]
        split.append(tmp)
    return split


train_gen = to_minibatches(train, 256)
for _ in range(200):
    with du.timer("epoch"):
        for i in range(196):
            batch = train_gen.next()
            sess.run(train_step,
                     feed_dict={x: batch["x"], y_: batch["y"]})
        accuracies = []
        for m in split_input(valid, 256):
            accuracies.append(sess.run(accuracy,
                                       feed_dict={x: m["x"], y_: m["y"]}))
        print(np.mean(accuracies))
