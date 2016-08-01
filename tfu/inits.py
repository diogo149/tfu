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


def msr_magnitude(shape, in_axes, out_axes):
    """
    http://arxiv.org/abs/1502.01852

    NOTE: also called He init
    """
    # consider all non-out_axes as in_axes
    in_axes_size = np.prod([s
                            for dim, s in enumerate(shape)
                            if dim not in out_axes])
    # NOTE: this is actually sqrt(2) in the paper, but a gain of sqrt(2)
    # is recommended for ReLUs
    return float(np.sqrt(1.0 / in_axes_size))


def msr_normal(shape, in_axes, out_axes, **_):
    std = msr_magnitude(shape, in_axes, out_axes)
    return tf.random_normal_initializer(stddev=std)


def msr_uniform(shape, in_axes, out_axes, **_):
    magnitude = float(np.sqrt(3)) * msr_magnitude(shape, in_axes, out_axes)
    return tf.random_uniform_initializer(minval=-magnitude,
                                         maxval=magnitude)


def orthogonal(shape, in_axes, out_axes, **_):
    """
    http://arxiv.org/abs/1312.6120

    implementation from Sander Dieleman
    """
    assert len(shape) >= 2

    def inner(shape, dtype):
        # consider all non-out_axes as in_axes
        tmp_out_shape = []
        tmp_in_shape = []
        in_axes_size = 1
        out_axes_size = 1
        tmp_order = list(out_axes)
        for dim, s in enumerate(shape):
            if dim in out_axes:
                out_axes_size *= s
                tmp_out_shape.append(s)
            else:
                in_axes_size *= s
                tmp_in_shape.append(s)
                tmp_order.append(dim)
        # calculate matrix shape
        flat_shape = (out_axes_size, in_axes_size)
        # make orthogonal matrix
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        # reshape to temporary shape
        q = q.reshape(tmp_out_shape + tmp_in_shape)
        # transpose out axes from the beginning
        transpose_axes = [None] * len(shape)
        for idx, dim in enumerate(tmp_order):
            transpose_axes[dim] = idx
        return np.transpose(q, axes=transpose_axes)

    return inner


# ############################### other inits ###############################


def set_forget_bias_init(init_value=0.):
    def inner(hs):
        hs.kwargs["initializer"] = tf.constant_initializer(init_value)
        return hs()
    return base.filter_dsl(inner,
                           key="get_variable",
                           variable_scope=["lstm", "forget", "bias"])
