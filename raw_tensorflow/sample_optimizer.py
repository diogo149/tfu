import tensorflow as tf
import tf_utils as tfu


def sgd(cost, parameters=None, learning_rate=0.01):
    if parameters is None:
        parameters = tf.trainable_variables()

    grads = tf.gradients(cost, parameters)

    all_updates = []
    for grad, param in zip(grads, parameters):
        assigned = tf.assign_sub(param, learning_rate * grad)
        all_updates.append(assigned)

    update_op = tf.group(*all_updates)
    return update_op


def adam(cost,
         parameters=None,
         learning_rate=1e-3,
         beta1=0.9,
         beta2=0.999,
         epsilon=1e-8):
    if parameters is None:
        parameters = tf.trainable_variables()

    grads = tf.gradients(cost, parameters)
    all_updates = []
    zero_init = tf.constant_initializer(0.)
    with tf.variable_scope("adam"):
        t_prev = tf.get_variable("t",
                                 shape=(),
                                 initializer=zero_init)
        t = tf.assign_add(t_prev, 1)
        all_updates.append(t)

        for grad, param in zip(grads, parameters):
            with tf.variable_scope(param.name.replace(":", "_")):
                param_shape = tfu.get_shape_values(param)
                m_prev = tf.get_variable("m",
                                         shape=param_shape,
                                         initializer=zero_init)
                v_prev = tf.get_variable("v",
                                         shape=param_shape,
                                         initializer=zero_init)
                m = tf.assign(m_prev,
                              m_prev * beta1 + grad * (1 - beta1))
                v = tf.assign(v_prev,
                              v_prev * beta2 + tf.square(grad) * (1 - beta2))

                numerator = learning_rate * m / (1 - tf.pow(beta1, t))
                denominator = tf.sqrt(v / (1 - tf.pow(beta2, t))) + epsilon
                assigned = tf.assign_sub(param, numerator / denominator)
                all_updates += [m, v, assigned]

    update_op = tf.group(*all_updates)
    return update_op


if __name__ == "__main__":
    import numpy as np
    import tensorflow as tf
    import tf_utils as tfu
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

    if 0:  # sgd
        if 1:  # custom
            train_step = sgd(cross_entropy)
        else:  # builtin
            optim = tf.train.GradientDescentOptimizer(learning_rate=0.01)
            train_step = optim.minimize(cross_entropy)
    else:  # adam
        if 0:  # custom
            train_step = adam(cross_entropy)
        else:  # builtin
            optim = tf.train.AdamOptimizer()
            train_step = optim.minimize(cross_entropy)

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
        with du.simple_timer("epoch"):
            for i in range(1000):
                batch = train_gen.next()
                train_step.run(feed_dict={x: batch["x"],
                                          y_: batch["y"],
                                          keep_prob: 0.5})
        print(accuracy.eval(feed_dict={x: valid["x"],
                                       y_: valid["y"],
                                       keep_prob: 1.0}))
