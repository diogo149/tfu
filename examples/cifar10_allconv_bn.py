import os
import du

trial_name = os.path.basename(__file__)[:-3]
with du.trial.run_trial(trial_name=trial_name) as trial:

    BATCH_SIZE = 128
    NUM_EPOCHS = 82

    import numpy as np
    import tensorflow as tf
    import tfu
    import tfu.sandbox.batch_normalization as bn

    datamaps = du.tasks.image_tasks.cifar10(x_dtype="float32",
                                            y_dtype="int64",
                                            include_valid_split=False)
    datamaps = du.tasks.image_tasks.subtract_per_pixel_mean(datamaps)
    train, valid = datamaps

    x = tf.placeholder(tf.float32, [None, 32, 32, 3], name="x")
    y = tf.placeholder(tf.int64, [None], name="y")

    tfu.add_hook(tfu.hooks.reuse_variables(variable_scope="valid",
                                           replace=[("valid", "train")]))
    tfu.add_hook(tfu.inits.set_weight_init(tfu.inits.msr_normal))
    tfu.add_hook(tfu.inits.scale_weight_inits(scale=np.sqrt(2)))
    tfu.add_hook(tfu.hooks.default_kwargs_dsl(kwargs={"filter_size": (3, 3)},
                                              key="conv2d"))

    epoch_size = len(train["x"]) * 2  # multiply by 2 for flips
    batches_per_epoch = np.ceil(epoch_size / float(BATCH_SIZE))
    expected_count = NUM_EPOCHS * batches_per_epoch
    tfu.counter.make_default_counter(expected_count=expected_count)

    def model():
        norm = bn.ema_batch_normalization

        h = x
        with tfu.variable_scope("conv1"):
            h = tfu.conv2d(h, num_filters=96)
            h = norm(h)
            h = tf.nn.relu(h)
        with tfu.variable_scope("conv2"):
            h = tfu.conv2d(h, num_filters=96)
            h = norm(h)
            h = tf.nn.relu(h)
        with tfu.variable_scope("conv3"):
            h = tfu.conv2d(h, num_filters=96, strides=(2, 2))
            h = norm(h)
            h = tf.nn.relu(h)
        with tfu.variable_scope("conv4"):
            h = tfu.conv2d(h, num_filters=192)
            h = norm(h)
            h = tf.nn.relu(h)
        with tfu.variable_scope("conv5"):
            h = tfu.conv2d(h, num_filters=192)
            h = norm(h)
            h = tf.nn.relu(h)
        with tfu.variable_scope("conv6"):
            h = tfu.conv2d(h, num_filters=192, strides=(2, 2))
            h = norm(h)
            h = tf.nn.relu(h)
        with tfu.variable_scope("conv7"):
            h = tfu.conv2d(h, num_filters=192)
            h = norm(h)
            h = tf.nn.relu(h)
        with tfu.variable_scope("conv8"):
            h = tfu.conv2d(h, num_filters=192, filter_size=(1, 1))
            h = norm(h)
            h = tf.nn.relu(h)
        with tfu.variable_scope("conv9"):
            h = tfu.conv2d(h, num_filters=10, filter_size=(1, 1))
            h = norm(h)

        h = tfu.global_avg_pool2d(h)
        cross_entropy = tf.reduce_mean(
            tfu.softmax_cross_entropy_with_logits(h, y))
        accuracy = tf.reduce_mean(tfu.categorical_accuracy(h, y))

        l2 = tfu.costs.l2()
        cost = cross_entropy + l2 * 2e-4

        tfu.summary.scalar("cost", cost)
        tfu.summary.scalar("cross_entropy", cross_entropy)
        tfu.summary.scalar("accuracy", accuracy)

        return dict(
            cross_entropy=cross_entropy,
            accuracy=accuracy,
            cost=cost,
        )

    file_writer = tf.summary.FileWriter(trial.file_path("summary"))

    train_summary = tfu.SummaryAccumulator()
    train_summary.add_file_writer(file_writer)

    valid_summary = tfu.SummaryAccumulator()
    valid_summary.add_file_writer(file_writer)

    updates = tfu.UpdatesAccumulator()

    with tfu.variable_scope("train"), train_summary, updates:
        train_out = model()
        learning_rate = tfu.counter.discrete_scale_schedule(
            0.1,
            scale=0.1,
            thresholds=[0.5, 0.75])
        tfu.summary.scalar("learning_rate", learning_rate)
        tfu.updates.nesterov_momentum(train_out["cost"],
                                      learning_rate=learning_rate)

    with tfu.variable_scope("valid", deterministic=True), valid_summary:
        valid_out = model()

    # enable XLA
    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    sess = tf.InteractiveSession(config=config)

    tfu.counter.set_session(sess)
    sess.run(tf.global_variables_initializer())

    summary_printer = tfu.SummaryPrinter()
    summary_printer.add_recipe("trial_info", trial)
    summary_printer.add_recipe("progress")
    summary_printer.add_recipe("iter")
    summary_printer.add_recipe("time")
    summary_printer.add_recipe("s_per_iter")
    summary_printer.add_recipe("x min+iter",
                               "valid_epoch/cost",
                               format="%.4g")
    summary_printer.add_recipe("x max+iter",
                               "valid_epoch/accuracy",
                               format="%.4g")
    summary_printer.add_recipe("add_finals",
                               ["train_epoch/cost",
                                "train_epoch/accuracy",
                                "valid_epoch/cost",
                                "valid_epoch/accuracy"],
                               format="%.4g")
    train_summary.add_summary_printer(summary_printer)
    valid_summary.add_summary_printer(summary_printer)

    train_fn = tfu.tf_fn(sess=sess,
                         inputs={"x": x,
                                 "y": y},
                         outputs=train_out,
                         ops=[updates, train_summary, tfu.counter.step_op()])
    train_fn = tfu.wrap.output_nan_guard(train_fn)
    train_fn = tfu.wrap.split_input(train_fn, split_size=BATCH_SIZE)
    train_fn = tfu.wrap.format_output_keys(train_fn, "train_epoch/%s")
    train_fn = tfu.wrap.update_summary_printer(train_fn, summary_printer)

    valid_fn = tfu.tf_fn(sess=sess,
                         inputs={"x": x,
                                 "y": y},
                         outputs=valid_out,
                         ops=[valid_summary])
    valid_fn = tfu.wrap.split_input(valid_fn, split_size=500)
    valid_fn = tfu.wrap.format_output_keys(valid_fn, "valid_epoch/%s")
    valid_fn = tfu.wrap.update_summary_printer(valid_fn, summary_printer)

    train_gen = du.tasks.image_tasks.gen_standard_cifar10_augmentation(train)

    while tfu.counter.get_count_value() < expected_count:
        with du.timer("epoch"):
            train_epoch = train_gen.next()
            train_res = train_fn(train_epoch)
            valid_res = valid_fn(valid)
            print(summary_printer.to_org_list())
    if 0:
        tfu.serialization.dump_variables(trial.file_path("final_variables"))
