import os
import du

trial_name = os.path.basename(__file__)[:-3]
with du.trial.run_trial(trial_name=trial_name) as trial:

    import numpy as np
    import scipy.stats
    import tensorflow as tf
    import tfu
    import tfu.sandbox.batch_normalization as bn
    import sklearn.metrics
    import sklearn.linear_model
    import sklearn.svm
    import sklearn.ensemble

    batch_size = 500
    num_epochs = 25
    train_size = 60000
    probe_size = 100
    hidden_size = 512
    train_keep_prob = 0.5
    use_bn = False

    train, test = du.tasks.image_tasks.mnist(x_dtype="float32",
                                             y_dtype="int64",
                                             include_valid_split=False)

    if train_size < 60000:
        _, train = du.tasks.tasks_utils.train_test_split_datamap(
            train, test_size=train_size, stratify="y")

    train_batches = du.tasks.tasks_utils.split_datamap(train, batch_size)

    _, train_probe = du.tasks.tasks_utils.train_test_split_datamap(
        train, test_size=probe_size, stratify="y")
    _, test_probe = du.tasks.tasks_utils.train_test_split_datamap(
        test, test_size=probe_size, stratify="y")

    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    y = tf.placeholder(tf.int64, shape=[None])
    keep_prob = tf.placeholder(tf.float32)

    tfu.add_hook(tfu.inits.set_weight_init(tfu.inits.xavier_normal))

    def mlp(deterministic):
        h = tfu.flatten(x, 2)
        with tfu.variable_scope("fc1"):
            h = tfu.affine(h, hidden_size)
            if use_bn:
                h = bn.ema_batch_normalization(
                    h, use_batch_stats=not deterministic)
            fc1 = h = tf.nn.relu(h)
        with tfu.variable_scope("fc2"):
            if not deterministic:
                h = tf.nn.dropout(h, keep_prob=keep_prob)
            h = tfu.affine(h, hidden_size, name="fc2")
            if use_bn:
                h = bn.ema_batch_normalization(
                    h, use_batch_stats=not deterministic)
            fc2 = h = tf.nn.relu(h)
        with tfu.variable_scope("logit"):
            if not deterministic:
                h = tf.nn.dropout(h, keep_prob=keep_prob)
            h = tfu.affine(h, 10, name="logit")
            if use_bn:
                h = bn.ema_batch_normalization(
                    h, use_batch_stats=not deterministic)
            logit = h
        raw_cross_entropy = tfu.softmax_cross_entropy_with_logits(logit, y)
        cross_entropy = tf.reduce_mean(raw_cross_entropy)
        accuracy = tf.reduce_mean(tfu.categorical_accuracy(logit, y))
        return fc1, fc2, logit, raw_cross_entropy, cross_entropy, accuracy

    with tfu.variable_scope("mlp"):
        _, _, _, _, train_ce, _ = mlp(False)
    with tfu.variable_scope("mlp", reuse=True):
        fc1, fc2, logit, raw_cross_entropy, cross_entropy, accuracy = mlp(True)

    train_step = tf.train.AdamOptimizer().minimize(train_ce)

    summary_op = tf.summary.merge([
        tf.summary.scalar("cost", cross_entropy),
        tf.summary.scalar("accuracy", accuracy),
    ])

    epochs = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        train_writer = tf.summary.FileWriter(trial.file_path("train_summary"))
        test_writer = tf.summary.FileWriter(trial.file_path("test_summary"))

        batch_idx = -1
        for epoch_idx in range(num_epochs):
            print trial.trial_name, trial.iteration_num, epoch_idx
            epoch_data = {}
            with du.timer("train"):
                for train_batch in train_batches:
                    batch_idx += 1
                    sess.run([train_step],
                             feed_dict={x: train_batch["x"],
                                        y: train_batch["y"],
                                        keep_prob: train_keep_prob})

            with du.timer("train_eval"):
                summary, cost, acc = sess.run([summary_op, cross_entropy, accuracy],
                                              feed_dict={x: train["x"],
                                                         y: train["y"]})
                train_writer.add_summary(summary, batch_idx)

            epoch_data["batch_idx"] = batch_idx
            epoch_data["train_cost"] = cost
            epoch_data["train_acc"] = acc

            with du.timer("test"):
                summary, cost, acc = sess.run([summary_op, cross_entropy, accuracy],
                                              feed_dict={x: test["x"],
                                                         y: test["y"]})
                test_writer.add_summary(summary, batch_idx)

            epoch_data["test_cost"] = cost
            epoch_data["test_acc"] = acc

            with du.timer("probe_eval"):
                train_ce, train_fc1, train_fc2, train_logit = sess.run(
                    [raw_cross_entropy, fc1, fc2, logit],
                    feed_dict={x: train_probe["x"],
                               y: train_probe["y"]})
                epoch_data["train_fc1"] = train_fc1
                epoch_data["train_fc2"] = train_fc2
                epoch_data["train_logit"] = train_logit

                test_ce, test_fc1, test_fc2, test_logit = sess.run(
                    [raw_cross_entropy, fc1, fc2, logit],
                    feed_dict={x: test_probe["x"],
                               y: train_probe["y"]})
                epoch_data["test_fc1"] = test_fc1
                epoch_data["test_fc2"] = test_fc2
                epoch_data["test_logit"] = test_logit

            with du.timer("probe_compute"):
                train_probe_data = {}
                test_probe_data = {}
                for name in ["fc1", "fc2", "logit"]:
                    train_feat = epoch_data["train_" + name]
                    test_feat = epoch_data["test_" + name]

                    feats = np.concatenate([train_feat, test_feat])
                    targets = np.arange(len(feats))

                    if 1:
                        clf = sklearn.linear_model.LogisticRegression(
                        )
                    if 0:
                        clf = sklearn.ensemble.RandomForestClassifier(
                            min_samples_leaf=2
                        )
                    if 0:
                        clf = sklearn.svm.SVC(
                            kernel="rbf",
                            probability=True,
                        )
                    clf.fit(feats, targets)
                    nll = -np.log(np.diag(clf.predict_proba(feats)))
                    train_nll = nll[:len(train_feat)]
                    test_nll = nll[-len(test_feat):]

                    train_probe_data["probe_cost_" + name] = train_nll.mean()
                    test_probe_data["probe_cost_" + name] = test_nll.mean()
                    train_pearsonr = np.corrcoef(train_nll, train_ce)[0, 1]
                    test_pearsonr = np.corrcoef(test_nll, test_ce)[0, 1]
                    train_spearmanr = scipy.stats.spearmanr(
                        train_nll, train_ce)[0]
                    test_spearmanr = scipy.stats.spearmanr(test_nll, test_ce)[0]
                    train_probe_data[
                        "cls_probe_pearsonr_" + name] = train_pearsonr
                    test_probe_data[
                        "cls_probe_pearsonr_" + name] = test_pearsonr
                    train_probe_data[
                        "cls_probe_spearmanr_" + name] = train_spearmanr
                    test_probe_data[
                        "cls_probe_spearmanr_" + name] = test_spearmanr

                for k, v in train_probe_data.items():
                    epoch_data["train_" + k] = v
                for k, v in test_probe_data.items():
                    epoch_data["test_" + k] = v

                train_writer.add_summary(tfu.dict_to_scalar_summary(train_probe_data),
                                         batch_idx)
                test_writer.add_summary(tfu.dict_to_scalar_summary(test_probe_data),
                                        batch_idx)

            epochs.append(epoch_data)
    train_writer.flush()
    test_writer.flush()
    if 1:
        import matplotlib.pyplot as plt
        plt.plot(train_nll, np.log(train_ce + 1e-8), "or")
        plt.plot(test_nll, np.log(test_ce + 1e-8), "ob")
        plt.savefig(trial.file_path("scatter.png"))
        plt.close()
        plt.plot(train_nll, du.numpy_utils.to_ranking(train_ce), "or")
        plt.plot(test_nll, du.numpy_utils.to_ranking(test_ce), "ob")
        plt.savefig(trial.file_path("scatter_rank.png"))
        plt.close()
