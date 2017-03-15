"""
from "Wide Residual Networks"
http://arxiv.org/abs/1605.07146
"""

import os
import du

BATCH_SIZE = 128
NUM_EPOCHS = 200

trial_name = os.path.basename(__file__)[:-3]
with du.trial.run_trial(trial_name=trial_name) as trial:
    import time
    import pprint
    import numpy as np
    import tensorflow as tf
    import tfu
    import tfu.sandbox.batch_normalization as bn
    # FIXME
    # import thu.sandbox.expected_iterations as ei
    # import du.sandbox.monitor_ui
    # from du.sandbox.summary import Summary

    # FIXME fix these functions
    datamaps = du.tasks.image_tasks.cifar10(x_dtype="float32",
                                            y_dtype="int64",
                                            include_valid_split=False)
    datamaps = du.tasks.image_tasks.subtract_per_pixel_mean(datamaps)
    train, valid = datamaps

    x = tf.placeholder(tf.float32, [None, 3, 32, 32], name="x")
    y = tf.placeholder(tf.int64, [None], name="y")

    tfu.add_hook(tfu.inits.set_weight_init(tfu.inits.msr_normal))
    tfu.add_hook(tfu.inits.scale_weight_inits(scale=np.sqrt(2)))
    tfu.add_hook(tfu.hooks.default_kwargs_dsl(kwargs={"filter_size": (3, 3)},
                                              key="conv2d"))

    # FIXME
    # acc = thu.UpdatesAccumulator(train=True)

    num_images = float(len(train["x"]) * 2)  # x2 for data augmentation
    batches_per_epoch = np.ceil(num_images / BATCH_SIZE)
    expected_iterations = NUM_EPOCHS * batches_per_epoch
    # FIXME
    # with acc:
    #     ei.set_expected_iterations(expected_iterations)

    def model(is_train):
        if 1:
            print "WRN-40-4"
            depth = 40
            widen_factor = 4

        if 0:
            print "WRN-16-8"
            depth = 16
            widen_factor = 8

        if 0:
            print "WRN-28-10"
            depth = 28
            widen_factor = 10

        keep_prob = 1

        assert (depth - 4) % 6 == 0
        blocks_per_group = (depth - 4) // 6
        num_filters = 16 * widen_factor
        # NOTE: do not change, blocks_per_group calculation assumes this is 3
        num_groups = 3

        if not is_train:
            keep_prob = 1

        # FIXME use is_train for ema_batch_normalization

        def shortcut(h, prev):
            h_shape = tfu.get_shape_values(h)
            prev_shape = tfu.get_shape_values(prev)
            if h_shape == prev_shape:
                return h + prev
            else:
                new_filters = h_shape[3]
                strides = (prev_shape[1] / h_shape[1],
                           prev_shape[2] / h_shape[2])
                return h + tfu.conv2d(prev,
                                      num_filters=new_filters,
                                      strides=strides,
                                      name="projection")

        h = x
        # FIXME fix dims for cifar-10?
        h = tfu.dimshuffle(x, [0, 2, 3, 1])
        h = tfu.conv2d(h, num_filters=16, name="conv_initial")
        h = bn.ema_batch_normalization(h, name="bn_initial")
        prev_act = prev = h = tf.nn.relu(h)
        for group_idx in range(1, num_groups + 1):
            for block_idx in range(blocks_per_group):
                with tfu.variable_scope("block_%d_%d" %
                                        (group_idx, block_idx)):
                    increase_dim = (group_idx != 1) and (block_idx == 0)
                    h = tfu.conv2d(h,
                                   num_filters=num_filters,
                                   strides=(2 if increase_dim else 1,) * 2,
                                   name="conv1")
                    h = bn.ema_batch_normalization(h, name="bn1")
                    h = tf.nn.relu(h)
                    if keep_prob != 1:
                        h = tf.nn.dropout(h, keep_prob)
                    h = tfu.conv2d(h,
                                   num_filters=num_filters,
                                   name="conv2")
                    if increase_dim:
                        # share bn + relu of preactivation
                        # for projection
                        prev = prev_act
                    prev = h = shortcut(h, prev)
                    h = bn.ema_batch_normalization(h, name="bn2")
                    prev_act = h = tf.nn.relu(h)
            num_filters *= 2

        h = tf.reduce_mean(h, axis=(2, 3))
        h = tfu.affine(h, num_units=10, name="logit")
        cross_entropy = tf.reduce_mean(
            tfu.softmax_cross_entropy_with_logits(h, y))
        accuracy = tf.reduce_mean(tfu.categorical_accuracy(h, y))

        l2 = tfu.costs.l2()
        cost = cross_entropy + l2 * 2e-4

        return dict(
            cross_entropy=cross_entropy,
            accuracy=accuracy,
            cost=cost,
        )

    with tfu.temporary_hooks([
            tfu.hooks.default_kwargs_dsl({"use_batch_stats": True},
                                         key="ema_batch_normalization"),
    ]):
        # FIXME
        # with acc:
        with tfu.variable_scope("model"):
            train_outputs = model(is_train=True)
        # FIXME
        # learning_rate = ei.discrete_scale_schedule(
        #     0.1,
        #     scale=0.2,
        #     thresholds=[0.3, 0.6, 0.8])
        learning_rate = tf.Variable(0.1, trainable=False)
        update_lr_op = tf.assign(learning_rate, learning_rate * 0.2)
        # thu.updates.nesterov_momentum(train_outputs["cost"],
        #                               learning_rate=learning_rate)
        if 0:
            train_op = tf.train.MomentumOptimizer(
                learning_rate=learning_rate,
                momentum=0.9,
                use_nesterov=True,
            ).minimize(train_outputs["cost"])
        else:
            with tfu.variable_scope("optimizer"):
                updates = tfu.updates.nesterov_momentum(
                    loss_or_grads=train_outputs["cost"],
                    learning_rate=learning_rate
                )
            train_op = tf.group(*[tf.assign(k, v) for k, v in updates.items()])

    with tfu.temporary_hooks([
            # FIXME
            # thu.hooks.reuse_shared(),
            tfu.hooks.default_kwargs_dsl({"use_batch_stats": False},
                                         key="ema_batch_normalization"),
    ]):
        with tfu.variable_scope("model", reuse=True):
            valid_outputs = model(is_train=False)

    if 1:
        config = tf.ConfigProto()
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        sess = tf.InteractiveSession(config=config)
        sess.run(tf.global_variables_initializer())
        train_gen = du.tasks.image_tasks.gen_standard_cifar10_augmentation(
            train)
        for epoch_idx in range(1, NUM_EPOCHS + 1):
            with du.timer("EPOCH"):
                train_epoch = train_gen.next()
                train_costs = []
                for batch in du.tasks.tasks_utils.split_datamap(train_epoch, BATCH_SIZE):
                    _, cost = sess.run([train_op, train_outputs["cost"]],
                                       feed_dict={x: batch["x"],
                                                  y: batch["y"]})
                    train_costs.append(cost)
                valid_costs = []
                valid_accs = []
                for batch in du.tasks.tasks_utils.split_datamap(valid, 500):
                    cost, acc = sess.run([valid_outputs["cost"], valid_outputs["accuracy"]],
                                         feed_dict={x: batch["x"],
                                                    y: batch["y"]})
                    valid_costs.append(cost)
                    valid_accs.append(acc)
                print "Epoch %03d trn: %.4g vld: %.4g acc: %.4g" % (
                    epoch_idx,
                    np.mean(train_costs),
                    np.mean(valid_costs),
                    np.mean(valid_accs),
                )
                if epoch_idx in {60, 120, 160}:
                    sess.run(update_lr_op)

    # FIXME
    if 0:
        # ################################ functions ###########################
        # train function
        with du.timer("train_fn"):
            train_fn = thu.dict_function(inputs={"x": x, "y": y},
                                         outputs=train_outputs,
                                         updates=acc.updates)
            train_fn = thu.wrap.output_nan_guard(train_fn)
            train_fn = thu.wrap.split_input(train_fn,
                                            split_size=BATCH_SIZE,
                                            keys=["x", "y"])
            train_fn = thu.wrap.time_call(train_fn)
            train_fn = thu.wrap.format_output_keys(train_fn, "train_%s")

        # valid function
        with du.timer("valid_fn"):
            valid_fn = thu.dict_function(inputs={"x": x, "y": y},
                                         outputs=valid_outputs)
            valid_fn = thu.wrap.split_input(valid_fn,
                                            split_size=500,
                                            keys=["x", "y"])
            valid_fn = thu.wrap.time_call(valid_fn)
            valid_fn = thu.wrap.format_output_keys(valid_fn, "valid_%s")

        # ################################# training ###########################

        summary = Summary()
        summary.add_recipe("trial_info", trial)
        summary.add("_iter", how="last")
        summary.add("_time", how="last")
        summary.add_recipe("s_per_iter")
        summary.add_recipe("x min+iter", "valid_cost", format="%.4g")
        summary.add_recipe("x max+iter", "valid_accuracy", format="%.4g")
        summary.add_recipe("add_finals",
                           ["train_cost", "valid_cost", "valid_accuracy"],
                           format="%.4g")

        result_writer = du.sandbox.monitor_ui.ResultWriter(
            dirname=trial.file_path("monitor_ui"),
            default_settings_file="monitor_ui_settings.json")

        print("Starting training...")
        start_time = time.time()
        train_gen = du.tasks.image_tasks.gen_standard_cifar10_augmentation(
            train)
        for epoch_idx in range(1, NUM_EPOCHS + 1):
            res = du.AttrDict(_iter=epoch_idx)
            train_epoch = train_gen.next()
            train_out = train_fn(train_epoch)
            valid_out = valid_fn(valid)
            res.update(train_out)
            res.update(valid_out)
            res._time = time.time() - start_time
            result_writer.write(res)
            trial.store("train_log", res, silent=True)
            summary.update(res)
            pprint.pprint(res)
            print(summary.to_org_list())

        thu.dump_shared(trial.file_path("best_model"))
