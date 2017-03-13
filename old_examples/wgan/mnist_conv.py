"""
based on https://gist.github.com/f0k/f3190ebba6c53887d598d03119ca2066
"""

import os
import du

trial_name = os.path.basename(__file__)[:-3]
with du.trial.run_trial(trial_name=trial_name) as trial:

    import numpy as np
    import tensorflow as tf
    import tfu
    import tfu.sandbox.batch_normalization as bn

    BATCH_SIZE = 64
    Z_DIM = 100
    LEARNING_RATE = 5e-5
    CLIP = 0.01
    NUM_EPOCHS = 1000
    EPOCH_SIZE = 100

    train, _ = du.tasks.image_tasks.mnist(
        x_dtype="float32",
        y_dtype="int64",
        include_valid_split=False)

    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    sess = tf.InteractiveSession(config=config)

    tfu.add_hook(tfu.inits.set_weight_init(tfu.inits.xavier_normal))

    def generator_function(z):
        l = z
        with tfu.variable_scope("fc1"):
            l = tfu.affine(l, num_units=1024)
            l = bn.simple_batch_normalization(l)
            l = tf.nn.relu(l)
        with tfu.variable_scope("fc2"):
            l = tfu.affine(l, num_units=128 * 7 * 7)
            l = bn.simple_batch_normalization(l)
            l = tf.nn.relu(l)
        l = tf.reshape(l, [-1, 7, 7, 128])
        with tfu.variable_scope("deconv1"):
            l = tfu.conv2d_transpose(l,
                                     num_filters=64,
                                     filter_size=(5, 5),
                                     strides=(2, 2))
            l = bn.simple_batch_normalization(l)
            l = tf.nn.relu(l)
        with tfu.variable_scope("deconv2"):
            l = tfu.conv2d_transpose(l,
                                     num_filters=1,
                                     filter_size=(5, 5),
                                     strides=(2, 2))
            l = bn.simple_batch_normalization(l)
            l = tf.nn.sigmoid(l)
        return l

    def critic_function(imgs):
        l = imgs
        with tfu.variable_scope("conv1"):
            l = tfu.conv2d(l,
                           num_filters=64,
                           filter_size=(5, 5),
                           strides=(2, 2))
            l = bn.simple_batch_normalization(l)
            l = tfu.leaky_relu(l, 0.2)
        with tfu.variable_scope("conv2"):
            l = tfu.conv2d(l,
                           num_filters=128,
                           filter_size=(5, 5),
                           strides=(2, 2))
            l = bn.simple_batch_normalization(l)
            l = tfu.leaky_relu(l, 0.2)
        l = tfu.flatten(l, 2)
        with tfu.variable_scope("fc1"):
            l = tfu.affine(l, num_units=1024)
            l = bn.simple_batch_normalization(l)
            l = tfu.leaky_relu(l, 0.2)
        with tfu.variable_scope("fc2"):
            l = tfu.linear(l, num_units=1)
        return l

    real = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 28, 28, 1])
    z = tf.random_uniform([BATCH_SIZE, Z_DIM])
    with tf.variable_scope("generator"):
        fake = generator_function(z)
    with tf.variable_scope("critic"):
        real_score = tf.reduce_mean(critic_function(real))
    with tf.variable_scope("critic", reuse=True):
        fake_score = tf.reduce_mean(critic_function(fake))

    critic_score = real_score - fake_score

    lr = tfu.get_variable("eta", shape=[], initial_value=LEARNING_RATE)
    new_lr = tf.placeholder(tf.float32)
    update_lr_op = tf.assign(lr, new_lr)

    generator_params = tfu.variables(variable_scope="generator")
    critic_params = tfu.variables(variable_scope="critic")

    generator_train_op = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(
        loss=-fake_score,
        var_list=generator_params,
    )
    critic_train_op = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(
        loss=-critic_score,
        var_list=critic_params,
    )
    critic_clip_op = tf.group(*[tf.assign(p, tf.clip_by_value(p, -CLIP, CLIP))
                                for p in critic_params])

    sess.run(tf.global_variables_initializer())
    batch_gen = du.tasks.tasks_utils.random_sample_datamap(train, BATCH_SIZE)
    generator_updates = 0
    for epoch_idx in range(NUM_EPOCHS):
        with du.timer("EPOCH"):
            critic_scores = []
            generator_scores = []

            for _ in range(EPOCH_SIZE):
                if (generator_updates < 25) or (generator_updates % 500 == 0):
                    critic_runs = 100
                else:
                    critic_runs = 5
                for _ in range(critic_runs):
                    s, _ = sess.run([critic_score, critic_train_op],
                                    feed_dict={real: next(batch_gen)["x"]})
                    sess.run([critic_clip_op])
                    critic_scores.append(s)

                s, _ = sess.run([fake_score, generator_train_op])
                generator_scores.append(s)
                generator_updates += 1

            # FIXME log with tensorboard
            print "Epoch %05d critic: %.4g generator: %.4g" % (
                epoch_idx,
                np.mean(critic_scores),
                np.mean(generator_scores),
            )

            # FIXME plotting
            import matplotlib.pyplot as plt
            samples = sess.run(fake)[:42]
            plt.imsave(trial.file_path('wgan_mnist_samples.png'),
                       (samples.reshape(6, 7, 28, 28)
                        .transpose(0, 2, 1, 3)
                        .reshape(6 * 28, 7 * 28)),
                       cmap='gray')

            # FIXME schedule
            if epoch_idx >= NUM_EPOCHS // 2:
                progress = float(epoch_idx) / NUM_EPOCHS
                sess.run([update_lr_op],
                         feed_dict={new_lr: LEARNING_RATE * 2 * (1 - progress)})
            # FIXME log learning rate
