import numpy as np
import tensorflow as tf
import tf_utils as tfu
import du

train, valid, test = du.tasks.image_tasks.mnist("float32")
for dataset in [train, valid, test]:
    dataset["x"] = dataset["x"].reshape(len(dataset["x"]), -1)
    dataset["y"] = dataset["y"].astype("int64")

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.int64, shape=[None])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = tf.reduce_mean(tfu.categorical_cross_entropy(y, y_))

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


accuracy = tf.reduce_mean(tfu.categorical_accuracy(y, y_))

sess.run(tf.initialize_all_variables())

train_gen = to_minibatches(train, 50)
for _ in range(10):
    for i in range(1000):
        batch = train_gen.next()
        train_step.run(feed_dict={x: batch["x"], y_: batch["y"]})
    print(accuracy.eval(feed_dict={x: valid["x"], y_: valid["y"]}))
