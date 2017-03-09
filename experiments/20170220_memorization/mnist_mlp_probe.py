import os
import du

trial_name = os.path.basename(__file__)[:-3]
with du.trial.run_trial(trial_name=trial_name) as trial:

    import numpy as np
    import tensorflow as tf
    import tfu

    batch_size = 500
    num_epochs = 500
    train_size = 100

    train, test = du.tasks.image_tasks.mnist(x_dtype="float32",
                                             y_dtype="int64",
                                             include_valid_split=False)

    train = du.tasks.tasks_utils.split_datamap(train, train_size)[0]

    if 1:
        train["probe_target"] = np.arange(train_size, dtype="int64")

    train_batches = du.tasks.tasks_utils.split_datamap(train, batch_size)
    test_batches = [test]

    train_summaries = []
    test_summaries = []
    probe_summaries = []

    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    y = tf.placeholder(tf.int64, shape=[None])

    if 1:
        probe_target = tf.placeholder(tf.int64, shape=[None])

    keep_prob = tf.placeholder(tf.float32)

    h = tfu.flatten(x, 2)

    tfu.add_hook(tfu.inits.set_weight_init(tfu.inits.xavier_normal))

    with tfu.variable_scope("mlp"):
        h = tfu.affine(h, 512, name="fc1")
        fc1 = h = tf.nn.relu(h)
        h = tf.nn.dropout(h, keep_prob=keep_prob)
        h = tfu.affine(h, 512, name="fc2")
        fc2 = h = tf.nn.relu(h)
        h = tf.nn.dropout(h, keep_prob=keep_prob)
        logit = h = tfu.affine(h, 10, name="logit")

    cross_entropy = tf.reduce_mean(tfu.softmax_cross_entropy_with_logits(h, y))
    accuracy = tf.reduce_mean(tfu.categorical_accuracy(h, y))

    params = tfu.variables(variable_scope="mlp")
    with tf.name_scope("adam"):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        # Op to calculate every variable gradient
        grads = tf.gradients(cross_entropy, params)
        grads_and_vars = list(zip(grads, params))
        # Op to update all variables according to their gradient
        train_step = optimizer.apply_gradients(grads_and_vars=grads_and_vars)

    with tfu.variable_scope("probe"):
        probe_cost = 0
        for name, feats in [("x", tfu.flatten(x, 2)),
                            ("fc1", fc1),
                            ("fc2", fc2),
                            ("logit", logit)]:
            with tfu.variable_scope(name):
                h = tfu.linear(feats, train_size)
            cost = tf.reduce_mean(
                tfu.softmax_cross_entropy_with_logits(h, probe_target))
            probe_cost += cost
            probe_summaries.append(tf.summary.scalar("probe_%s" % name, cost))

    params = tfu.variables(variable_scope="probe")
    assert len(params) == 4
    with tf.name_scope("probe_adam"):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        # Op to calculate every variable gradient
        grads = tf.gradients(probe_cost, params)
        grads_and_vars = list(zip(grads, params))
        # Op to update all variables according to their gradient
        probe_train_step = optimizer.apply_gradients(
            grads_and_vars=grads_and_vars)

    cost_summary = tf.summary.scalar("cost", cross_entropy)
    train_summaries.append(cost_summary)
    test_summaries.append(cost_summary)

    accuracy_summary = tf.summary.scalar("accuracy", accuracy)
    train_summaries.append(accuracy_summary)
    test_summaries.append(accuracy_summary)

    train_summary_op = tf.summary.merge(train_summaries)
    test_summary_op = tf.summary.merge(test_summaries)
    probe_summary_op = tf.summary.merge(probe_summaries)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        train_writer = tf.summary.FileWriter(trial.file_path("train_summary"))
        test_writer = tf.summary.FileWriter(trial.file_path("test_summary"))

        batch_idx = -1
        for epoch_idx in range(num_epochs):
            print epoch_idx
            with du.timer("train"):
                for train_batch in train_batches:
                    batch_idx += 1
                    _, summary = sess.run([train_step, train_summary_op],
                                          feed_dict={x: train_batch["x"],
                                                     y: train_batch["y"],
                                                     keep_prob: 0.5})
                    train_writer.add_summary(summary, batch_idx)

            with du.timer("test"):
                summary, = sess.run([test_summary_op],
                                    feed_dict={x: test["x"],
                                               y: test["y"],
                                               keep_prob: 1.0})
                test_writer.add_summary(summary, batch_idx)

            with du.timer("probe"):
                for train_batch in train_batches:
                    sess.run([probe_train_step],
                             feed_dict={x: train_batch["x"],
                                        probe_target: train_batch["probe_target"],
                                        keep_prob: 1.})

                summary, = sess.run([probe_summary_op],
                                    feed_dict={x: train["x"],
                                               probe_target: train_batch["probe_target"],
                                               keep_prob: 1.0})
                train_writer.add_summary(summary, batch_idx)
