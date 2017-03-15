import os
import du

trial_name = os.path.basename(__file__)[:-3]
with du.trial.run_trial(trial_name=trial_name) as trial:

    NUM_EPOCHS = 25
    BATCH_SIZE = 500

    import numpy as np
    import tensorflow as tf
    import tfu
    import du.sandbox.summary

    train, valid, test = du.tasks.image_tasks.mnist(x_dtype="float32",
                                                    y_dtype="int64")

    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    y = tf.placeholder(tf.int64, shape=[None])

    tfu.add_hook(tfu.hooks.reuse_variables(variable_scope="valid",
                                           replace=[("valid", "train")]))
    tfu.add_hook(tfu.inits.set_weight_init(tfu.inits.msr_normal))
    tfu.add_hook(tfu.inits.scale_weight_inits(np.sqrt(2)))
    tfu.counter.make_default_counter(expected_count=NUM_EPOCHS - 1)

    def model(deterministic):
        h = tfu.flatten(x, 2)
        with tfu.variable_scope("fc1"):
            h = tfu.affine(h, 512)
            h = tf.nn.relu(h)
            if not deterministic:
                h = tf.nn.dropout(h, keep_prob=0.5)
        with tfu.variable_scope("fc2"):
            h = tfu.affine(h, 512)
            h = tf.nn.relu(h)
            if not deterministic:
                h = tf.nn.dropout(h, keep_prob=0.5)
        h = tfu.affine(h, 10, "logit")

        cross_entropy = tf.reduce_mean(
            tfu.softmax_cross_entropy_with_logits(h, y))
        accuracy = tf.reduce_mean(tfu.categorical_accuracy(h, y))

        tfu.summary.scalar("cost", cross_entropy)
        tfu.summary.scalar("accuracy", accuracy)

        return dict(
            cross_entropy=cross_entropy,
            accuracy=accuracy,
        )

    summary = du.sandbox.summary.Summary()
    summary.add_recipe("trial_info", trial)
    # FIXME
    # summary.add("_iter", how="last")
    # summary.add("_time", how="last")
    # summary.add_recipe("s_per_iter")
    # summary.add_recipe("x min+iter", "valid_cost", format="%.4g")
    # summary.add_recipe("x max+iter", "valid_accuracy", format="%.4g")
    # summary.add_recipe("add_finals",
    #                    ["train_cost", "valid_cost", "valid_accuracy"],
    #                    format="%.4g")

    file_writer = tf.summary.FileWriter(trial.file_path("summary"))

    train_summary = tfu.SummaryAccumulator()
    train_summary.add_file_writer(file_writer)

    valid_summary = tfu.SummaryAccumulator()
    valid_summary.add_file_writer(file_writer)

    updates = tfu.UpdatesAccumulator()

    with tfu.variable_scope("train"), train_summary, updates:
        train_out = model(False)
        tfu.updates.adam(train_out["cross_entropy"])

    with tfu.variable_scope("valid"), valid_summary:
        valid_out = model(True)

    # enable XLA
    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    sess = tf.InteractiveSession(config=config)

    tfu.counter.set_session(sess)
    sess.run(tf.global_variables_initializer())

    train_fn = tfu.tf_fn(sess=sess,
                         inputs={"x": x,
                                 "y": y},
                         outputs=train_out,
                         ops=[updates, train_summary],
                         name="train_fn")
    train_fn = tfu.wrap.split_input(train_fn, split_size=BATCH_SIZE)
    train_fn = tfu.wrap.shuffle_input(train_fn)

    valid_fn = tfu.tf_fn(sess=sess,
                         inputs={"x": x,
                                 "y": y},
                         outputs=valid_out,
                         ops=[valid_summary],
                         name="valid_fn")
    valid_fn = tfu.wrap.split_input(valid_fn, split_size=BATCH_SIZE)
    # FIXME summaries
    # FIXME summary printers

    for _ in range(NUM_EPOCHS):
        with du.timer("epoch"):
            train_res = train_fn(train)
            valid_res = valid_fn(valid)
            print "train", train_res
            print "valid", valid_res
            tfu.counter.step()
