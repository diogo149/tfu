from collections import OrderedDict
import tensorflow as tf
import tfu


@tfu.hooked
def simple_batch_normalization(x, epsilon=1e-4, name=None):
    with tfu.variable_scope(name):
        with tfu.variable_scope("batch_normalization"):
            num_units = tfu.get_shape_values(x)[-1]
            beta = tfu.get_variable(name="beta",
                                    shape=[num_units],
                                    dtype=x.dtype,
                                    trainable=True,
                                    bias=True,
                                    bn_beta=True)
            gamma = tfu.get_variable(name="gamma",
                                     shape=[num_units],
                                     dtype=x.dtype,
                                     initial_value=1.0,
                                     trainable=True,
                                     bn_gamma=True)
            mean, variance = tf.nn.moments(
                x=x,
                axes=list(range(tfu.ndim(x) - 1)),
                keep_dims=True)
            return tf.nn.batch_normalization(x=x,
                                             mean=mean,
                                             variance=variance,
                                             offset=beta,
                                             scale=gamma,
                                             variance_epsilon=epsilon)


@tfu.hooked
def ema_batch_normalization(x,
                            use_batch_stats=True,
                            alpha=0.1,
                            epsilon=1e-4,
                            name=None):
    if tfu.current_metadata().get("deterministic"):
        use_batch_stats = False

    with tfu.variable_scope(name):
        with tfu.variable_scope("batch_normalization"):
            num_units = tfu.get_shape_values(x)[-1]
            beta = tfu.get_variable(name="beta",
                                    shape=[num_units],
                                    dtype=x.dtype,
                                    trainable=True,
                                    bias=True,
                                    bn_beta=True)
            gamma = tfu.get_variable(name="gamma",
                                     shape=[num_units],
                                     dtype=x.dtype,
                                     initial_value=1.0,
                                     trainable=True,
                                     bn_gamma=True)

            mean, variance = tf.nn.moments(
                x=x,
                axes=list(range(tfu.ndim(x) - 1)),
                keep_dims=True)
            inv_std = tf.reciprocal(tf.sqrt(variance + epsilon))
            mu = tfu.get_variable(name="mu",
                                  shape=tfu.get_shape_values(mean),
                                  dtype=x.dtype,
                                  trainable=False)
            inv_sigma = tfu.get_variable(name="inv_sigma",
                                         shape=tfu.get_shape_values(inv_std),
                                         dtype=x.dtype,
                                         initial_value=1.0,
                                         trainable=False)

            if use_batch_stats:
                updates = OrderedDict()
                updates[mu] = (1 - alpha) * mu + alpha * mean
                updates[inv_sigma] = (1 - alpha) * inv_sigma + alpha * inv_std
                tfu.register_updates(updates,
                                     train=True,
                                     bn=True,
                                     # TODO should this be required
                                     required=True)

                return (x - mean) * (inv_std * gamma) + beta
            else:
                return (x - mu) * (inv_sigma * gamma) + beta
