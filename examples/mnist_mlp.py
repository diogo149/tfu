import numpy as np
import tensorflow as tf
import tfu
import du

train, valid, test = du.tasks.image_tasks.mnist("float32")
for dataset in [train, valid, test]:
    dataset["x"] = dataset["x"].astype("float32")
    dataset["y"] = dataset["y"].astype("int64")

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 1, 28, 28])
y_ = tf.placeholder(tf.int64, shape=[None])

keep_prob = tf.placeholder(tf.float32)

h = tfu.flatten(x, 2)

with tf.variable_scope("mlp"):
    with tfu.temporary_hook(tfu.inits.set_weight_init(tfu.inits.xavier_normal)):
        h = tfu.affine(h, 512, "fc1")
        h = tf.nn.relu(h)
        h = tf.nn.dropout(h, keep_prob=keep_prob)
        h = tfu.affine(h, 512, "fc2")
        h = tf.nn.relu(h)
        h = tf.nn.dropout(h, keep_prob=keep_prob)
        h = tfu.affine(h, 10, "logit")

cross_entropy = tf.reduce_mean(tfu.softmax_cross_entropy_with_logits(h, y_))

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

sess.run(tf.global_variables_initializer())

train_gen = to_minibatches(train, 500)
for _ in range(25):
    with du.timer("epoch"):
        for i in range(100):
            batch = train_gen.next()
            train_step.run(feed_dict={x: batch["x"],
                                      y_: batch["y"],
                                      keep_prob: 0.5})
        print(accuracy.eval(feed_dict={x: valid["x"],
                                       y_: valid["y"],
                                       keep_prob: 1.0}))
