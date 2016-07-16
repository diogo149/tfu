import numpy as np
import tensorflow as tf

from . import base


def scale_inits(scale, **filter_dsl_kwargs):
    if scale == "relu":
        scale = float(np.sqrt(2))

    def inner(hs):
        if "initializer" in hs.kwargs:
            old_initializer = hs.kwargs["initializer"]

            def inner(shape, dtype):
                res = old_initializer(shape, dtype)
                return res * scale

            hs.kwargs["initializer"] = inner
        return hs()

    return base.filter_dsl(inner,
                           key="get_variable",
                           **filter_dsl_kwargs)

# ############################### weight inits ###############################


def set_weight_init(weight_init, **filter_dsl_kwargs):
    def inner(hs):
        if "in_axes" in hs.kwargs and "out_axes" in hs.kwargs:
            hs.kwargs["initializer"] = weight_init(**hs.kwargs)
        return hs()

    return base.filter_dsl(inner,
                           key="get_variable",
                           **filter_dsl_kwargs)


def xavier_magnitude(shape, in_axes, out_axes):
    shape = np.array(shape)
    other_axes_size = np.prod([s
                               for dim, s in enumerate(shape)
                               if not ((dim in in_axes) or
                                       (dim in out_axes))])
    in_axes_size = np.prod(shape[in_axes])
    out_axes_size = np.prod(shape[out_axes])

    return float(np.sqrt(2.0 / ((in_axes_size + out_axes_size) *
                                other_axes_size)))


def xavier_normal(shape, in_axes, out_axes, **_):
    std = xavier_magnitude(shape, in_axes, out_axes)
    return tf.random_normal_initializer(stddev=std)


def xavier_uniform(shape, in_axes, out_axes, **_):
    magnitude = float(np.sqrt(3)) * xavier_magnitude(shape, in_axes, out_axes)
    return tf.random_uniform_initializer(minval=-magnitude,
                                         maxval=magnitude)


# ############################### other inits ###############################


def set_forget_bias_init(init_value=0.):
    def inner(hs):
        hs.kwargs["initializer"] = tf.constant_initializer(init_value)
        return hs()
    return base.filter_dsl(inner,
                           key="get_variable",
                           variable_scope=["lstm", "forget_bias"])
