import os
import du

trial_name = os.path.basename(__file__)[:-3]
with du.trial.run_trial(trial_name=trial_name) as trial:

    import tensorflow as tf
    import tfu

    batch_size = 500
    num_epochs = 500
    train_size = 100
    learning_rate = 0.0003
    hidden_size = 512

    train, test = du.tasks.image_tasks.mnist(x_dtype="float32",
                                             y_dtype="int64",
                                             include_valid_split=False)

    _, train = du.tasks.tasks_utils.train_test_split_datamap(
        train, test_size=train_size, stratify="y")

    train_batches = du.tasks.tasks_utils.split_datamap(train, batch_size)
    test_batches = [test]

    train_summaries = []
    test_summaries = []

    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    y = tf.placeholder(tf.int64, shape=[None])

    keep_prob = tf.placeholder(tf.float32)

    tfu.add_hook(tfu.inits.set_weight_init(tfu.inits.xavier_normal))

    with tfu.variable_scope("net"):
        h = x
        with tfu.variable_scope("conv1"):
            h = tfu.conv2d(h, num_filters=32, filter_size=(3, 3))
            h = tfu.add_bias(h)
            h = tf.nn.relu(h)
        with tfu.variable_scope("conv2"):
            h = tfu.conv2d(h, num_filters=32, filter_size=(3, 3))
            h = tfu.add_bias(h)
            h = tf.nn.relu(h)
        h = tfu.max_pool2d(h, (2, 2))
        with tfu.variable_scope("conv3"):
            h = tfu.conv2d(h, num_filters=32, filter_size=(3, 3))
            h = tfu.add_bias(h)
            h = tf.nn.relu(h)
        with tfu.variable_scope("conv4"):
            h = tfu.conv2d(h, num_filters=32, filter_size=(3, 3))
            h = tfu.add_bias(h)
            h = tf.nn.relu(h)
        h = tfu.max_pool2d(h, (2, 2))
        h = tfu.flatten(h, 2)
        h = tf.nn.dropout(h, keep_prob=keep_prob)
        with tfu.variable_scope("fc1"):
            h = tfu.affine(h, hidden_size)
        fc1 = h = tf.nn.relu(h)
        h = tf.nn.dropout(h, keep_prob=keep_prob)
        logit = h = tfu.affine(h, 10, name="logit")

    cross_entropy = tf.reduce_mean(tfu.softmax_cross_entropy_with_logits(h, y))
    accuracy = tf.reduce_mean(tfu.categorical_accuracy(h, y))

    params = tfu.variables(variable_scope="net")
    with tf.name_scope("opt"):
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        # Op to calculate every variable gradient
        grads = tf.gradients(cross_entropy, params)
        grads_and_vars = list(zip(grads, params))
        # Op to update all variables according to their gradient
        net_train_step = optimizer.apply_gradients(
            grads_and_vars=grads_and_vars)

    cost_summary = tf.summary.scalar("cost", cross_entropy)
    train_summaries.append(cost_summary)
    test_summaries.append(cost_summary)

    accuracy_summary = tf.summary.scalar("accuracy", accuracy)
    train_summaries.append(accuracy_summary)
    test_summaries.append(accuracy_summary)

    train_summary_op = tf.summary.merge(train_summaries)
    test_summary_op = tf.summary.merge(test_summaries)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        train_writer = tf.summary.FileWriter(trial.file_path("train_summary"))
        test_writer = tf.summary.FileWriter(trial.file_path("test_summary"))

        batch_idx = -1
        for epoch_idx in range(num_epochs):
            print trial.trial_name, trial.iteration_num, epoch_idx
            with du.timer("train"):
                for train_batch in train_batches:
                    batch_idx += 1
                    _, summary = sess.run([net_train_step,
                                           train_summary_op],
                                          feed_dict={x: train_batch["x"],
                                                     y: train_batch["y"],
                                                     keep_prob: 1.0})
                    train_writer.add_summary(summary, batch_idx)

            with du.timer("test"):
                summary, = sess.run([test_summary_op],
                                    feed_dict={x: test["x"],
                                               y: test["y"],
                                               keep_prob: 1.0})
                test_writer.add_summary(summary, batch_idx)

print trial.file_path("")
