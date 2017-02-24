import tensorflow as tf
import tfu


@tfu.hooked
def simple_batch_normalization(tensor, epsilon=1e-4, name=None):
    with tfu.variable_scope(name):
        with tfu.variable_scope("batch_normalization"):
            num_units = tfu.get_shape_values(tensor)[1]
            beta = tfu.get_variable("beta",
                                    shape=[num_units],
                                    dtype=tensor.dtype,
                                    trainable=True,
                                    bias=True,
                                    bn_beta=True)
            gamma = tfu.get_variable("gamma",
                                     shape=[num_units],
                                     dtype=tensor.dtype,
                                     initial_value=1.0,
                                     trainable=True,
                                     bn_gamma=True)
            mean, variance = tf.nn.moments(
                x=tensor,
                axes=[dim for dim in range(tfu.ndim(tensor))
                      if dim != 1],
                keep_dims=True)
            return tf.nn.batch_normalization(x=tensor,
                                             mean=mean,
                                             variance=variance,
                                             offset=beta,
                                             scale=gamma,
                                             variance_epsilon=epsilon)


@tfu.hooked
def ema_batch_normalization(tensor,
                            use_batch_stats,
                            alpha=0.1,
                            epsilon=1e-4,
                            name=None):
    with tfu.variable_scope(name):
        with tfu.variable_scope("batch_normalization"):
            num_units = tfu.get_shape_values(tensor)[1]
            beta = tfu.get_variable("beta",
                                    shape=[num_units],
                                    dtype=tensor.dtype,
                                    trainable=True,
                                    bias=True,
                                    bn_beta=True)
            gamma = tfu.get_variable("gamma",
                                     shape=[num_units],
                                     dtype=tensor.dtype,
                                     initial_value=1.0,
                                     trainable=True,
                                     bn_gamma=True)

            pattern = ["x"] * tfu.ndim(tensor)
            pattern[1] = 0
            gamma = tfu.dimshuffle(gamma, pattern)
            beta = tfu.dimshuffle(beta, pattern)

            mean, variance = tf.nn.moments(
                x=tensor,
                axes=[dim for dim in range(tfu.ndim(tensor))
                      if dim != 1],
                keep_dims=True)
            inv_std = tf.reciprocal(tf.sqrt(variance + epsilon))
            mu = tfu.get_variable("mu",
                                  shape=tfu.get_shape_values(mean),
                                  dtype=tensor.dtype,
                                  trainable=False)
            inv_sigma = tfu.get_variable("inv_sigma",
                                         shape=tfu.get_shape_values(inv_std),
                                         dtype=tensor.dtype,
                                         initial_value=1.0,
                                         trainable=False)

            if use_batch_stats:
                update = tf.group(
                    tf.assign(mu, (1 - alpha) * mu + alpha * mean),
                    tf.assign(inv_sigma, (1 - alpha) *
                              inv_sigma + alpha * inv_std),
                )
                with tf.control_dependencies([update]):
                    return (tensor - mean) * (inv_std * gamma) + beta
            else:
                return (tensor - mu) * (inv_sigma * gamma) + beta
