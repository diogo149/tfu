from collections import OrderedDict
import tensorflow as tf
import tfu


def _get_bn_stats(x, epsilon):
    mean, variance = tf.nn.moments(
        x=x,
        axes=list(range(tfu.ndim(x) - 1)),
        keep_dims=True)
    inv_std = tf.reciprocal(tf.sqrt(variance + epsilon))
    return mean, inv_std


@tfu.hooked
def simple_batch_normalization(x, epsilon=1e-4, name=None):
    with tfu.variable_scope(name):
        with tfu.variable_scope("batch_normalization"):
            mean, inv_std = _get_bn_stats(x, epsilon)
            return tfu.add_bias((x - mean) * tfu.learned_scaling(inv_std))


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
            if use_batch_stats:
                mean, inv_std = _get_bn_stats(x, epsilon)

                updates_metadata = dict(train=True,
                                        bn=True,
                                        # TODO should this be required
                                        required=True)
                tfu.ema(mean,
                        alpha=alpha,
                        name="mu",
                        **updates_metadata)
                tfu.ema(inv_std,
                        alpha=alpha,
                        name="inv_sigma",
                        **updates_metadata)
            else:
                mean = tfu.get_ema("mu")
                inv_std = tfu.get_ema("inv_sigma")

            return tfu.add_bias((x - mean) * tfu.learned_scaling(inv_std))
