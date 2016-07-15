import numpy as np
import tensorflow as tf
import tf_utils as tfu
import du

train, valid, test = du.tasks.image_tasks.mnist("float32")
for dataset in [train, valid, test]:
    dataset["x"] = dataset["x"].astype("float32")
    dataset["y"] = dataset["y"].astype("int64")

x = tf.placeholder(tf.float32, shape=[None, 1, 28, 28])
y_ = tf.placeholder(tf.int64, shape=[None])

keep_prob = tf.placeholder(tf.float32)

h = tfu.flatten(x, 2)

with tf.variable_scope("mlp",
                       initializer=tf.random_uniform_initializer(-0.05, 0.05)):
    h = tfu.affine("fc1", h, 256)
    h = tf.nn.relu(h)
    h = tf.nn.dropout(h, keep_prob=keep_prob)
    h = tfu.affine("fc2", h, 256)
    h = tf.nn.relu(h)
    h = tf.nn.dropout(h, keep_prob=keep_prob)
    h = tfu.affine("logit", h, 10)

cross_entropy = tf.reduce_mean(tfu.softmax_cross_entropy_with_logits(h, y_))


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


with tf.name_scope('adam'):
    # Gradient Descent
    optimizer = tf.train.AdamOptimizer()
    # Op to calculate every variable gradient
    grads = tf.gradients(cross_entropy, tf.trainable_variables())
    grads = list(zip(grads, tf.trainable_variables()))
    # Op to update all variables according to their gradient
    train_step = optimizer.apply_gradients(grads_and_vars=grads)

tf.scalar_summary("cost", cross_entropy)
tf.scalar_summary("accuracy", accuracy)
for var in tf.trainable_variables():
    tf.histogram_summary(var.name, var)
for grad, var in grads:
    tf.histogram_summary(var.name + '/gradient', grad)
merged_summary_op = tf.merge_all_summaries()


with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(tf.initialize_all_variables())

    summary_writer = tf.train.SummaryWriter("./tensorflow_logs",
                                            graph=tf.get_default_graph())

    train_gen = to_minibatches(train, 50)
    for epoch_idx in range(10):
        for batch_idx in range(1000):
            batch = train_gen.next()
            _, summary = sess.run([train_step, merged_summary_op],
                                  feed_dict={x: batch["x"],
                                             y_: batch["y"],
                                             keep_prob: 0.5})
            summary_writer.add_summary(summary, epoch_idx * 1000 + batch_idx)
        valid_acc, = sess.run([accuracy],
                              feed_dict={x: valid["x"],
                                         y_: valid["y"],
                                         keep_prob: 1.0})
        print(valid_acc)

    print "run 'tensorboard --logdir=./tensorflow_logs'"
    print "then open http://0.0.0.0:6006/"
