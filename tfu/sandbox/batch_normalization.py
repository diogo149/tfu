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
