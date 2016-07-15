import numpy as np
import tensorflow as tf
import tf_utils as tfu
import du

DATA_FORMAT = "NHWC"
# DATA_FORMAT = "NCHW"  # TODO test

train, valid, test = du.tasks.image_tasks.mnist("float32")
for dataset in [train, valid, test]:
    dataset["x"] = dataset["x"].astype("float32")
    dataset["y"] = dataset["y"].astype("int64")

    if DATA_FORMAT == "NHWC":
        dataset["x"] = dataset["x"].reshape((-1, 28, 28, 1))

init_shape = {
    "NHWC": [None, 28, 28, 1],
    "NCHW": [None, 1, 28, 28],
}[DATA_FORMAT]

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=init_shape)
y_ = tf.placeholder(tf.int64, shape=[None])

h = x

with tf.variable_scope("mlp",
                       initializer=tf.random_uniform_initializer(-0.05, 0.05)):
    h = tfu.conv2d("conv1",
                   h,
                   num_filters=16,
                   filter_size=(5, 5),
                   # strides=(2, 2),
                   data_format=DATA_FORMAT)
    h = tf.nn.relu(h)
    h = tfu.max_pool(h, (2, 2), data_format=DATA_FORMAT)
    h = tfu.conv2d("conv2",
                   h,
                   num_filters=32,
                   filter_size=(5, 5),
                   # strides=(2, 2),
                   data_format=DATA_FORMAT)
    h = tf.nn.relu(h)
    h = tfu.max_pool(h, (2, 2), data_format=DATA_FORMAT)
    h = tfu.flatten(h, 2)
    h = tfu.affine("fc1", h, 256)
    h = tf.nn.relu(h)
    h = tfu.affine("fc2", h, 256)
    h = tf.nn.relu(h)
    h = tfu.affine("logit", h, 10)

cross_entropy = tf.reduce_mean(tfu.softmax_cross_entropy_with_logits(h, y_))

# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)


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


accuracy = tf.reduce_mean(tfu.categorical_accuracy(h, y_))

sess.run(tf.initialize_all_variables())

train_gen = to_minibatches(train, 50)
for _ in range(10):
    for i in range(1000):
        batch = train_gen.next()
        train_step.run(feed_dict={x: batch["x"], y_: batch["y"]})
    print(accuracy.eval(feed_dict={x: valid["x"], y_: valid["y"]}))
