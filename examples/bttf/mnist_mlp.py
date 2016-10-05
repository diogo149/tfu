import numpy as np
import tensorflow as tf
import tfu
from tfu.sandbox import bttf
import du


def bachelor_normalization(name, x, beta=0.95, epsilon=1e-4):
    # TODO try different beta for mu and mu2
    with tf.variable_scope(name):
        curr_indices = filter(lambda x: x != 1,
                              range(tfu.ndim(x)))
        mix_indices = filter(lambda x: x != 0,
                             curr_indices)

        # shift by mean
        current_mean = tf.reduce_mean(x,
                                      reduction_indices=curr_indices,
                                      keep_dims=True)
        mix_mean = tf.reduce_mean(x,
                                  reduction_indices=mix_indices,
                                  keep_dims=True)
        bttf_mu = bttf.bttf_mean(current_mean=current_mean)
        mu = beta * bttf_mu + (1 - beta) * mix_mean

        # scale by x^2
        # TODO should we scale by x^2 or (x-mu)^2
        if 1:
            x2 = tf.square(x)
        else:
            x2 = tf.square(x - mu)
        current_mean2 = tf.reduce_mean(x2,
                                       reduction_indices=curr_indices,
                                       keep_dims=True)
        mix_mean2 = tf.reduce_mean(x2,
                                   reduction_indices=mix_indices,
                                   keep_dims=True)
        bttf_mu2 = bttf.bttf_mean(current_mean=current_mean2,
                                  mean_initializer=1,
                                  name="bttf_mean2")
        mu2 = beta * bttf_mu2 + (1 - beta) * mix_mean2

        scale = tfu.learned_scaling(tf.inv(tf.sqrt(mu2 + epsilon)),
                                    axis=1)
        res = (x - mu) * scale
        res = tfu.add_bias(res, axis=1)
        return res


train, valid, test = du.tasks.image_tasks.mnist("float32")
for dataset in [train, valid, test]:
    dataset["x"] = dataset["x"].astype("float32")
    dataset["y"] = dataset["y"].astype("int64")

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 1, 28, 28])
y_ = tf.placeholder(tf.int64, shape=[None])

h = tfu.flatten(x, 2)

tfu.add_hook(tfu.inits.set_weight_init(tfu.inits.orthogonal))

with tf.variable_scope("mlp"):
    h = tfu.linear(h, 512, "fc1")
    h = bachelor_normalization("bn1", h)
    h = tf.nn.relu(h)
    h = tfu.linear(h, 512, "fc2")
    h = bachelor_normalization("bn2", h)
    h = tf.nn.relu(h)
    h = tfu.affine(h, 10, name="logit")

cross_entropy = tf.reduce_mean(tfu.softmax_cross_entropy_with_logits(h, y_))

# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

bttf_update_op = bttf.bttf_mean_updates(cross_entropy, alpha1=0.75)


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

train_gen = to_minibatches(train, 500)
for _ in range(25):
    with du.timer("epoch"):
        for i in range(100):
            batch = train_gen.next()
            sess.run([train_step, bttf_update_op],
                     feed_dict={x: batch["x"],
                                y_: batch["y"]})
        print(accuracy.eval(feed_dict={x: valid["x"],
                                       y_: valid["y"]}))
