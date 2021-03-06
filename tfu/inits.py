import numpy as np
import tensorflow as tf

from . import base


def scale_inits(scale, **filter_dsl_kwargs):
    """
    NOTE: this scales all inits, not just weights
    """
    if scale == "relu":
        scale = float(np.sqrt(2))

    def scale_inits_inner(hs):
        res = hs()
        return res * scale

    return base.filter_dsl(scale_inits_inner,
                           key="get_initial_value",
                           **filter_dsl_kwargs)

# ############################### weight inits ###############################


def scale_weight_inits(scale, **filter_dsl_kwargs):
    if scale == "relu":
        scale = float(np.sqrt(2))

    def scale_inits_inner(hs):
        metadata = hs.args[0]
        res = hs()
        if metadata.get("weight"):
            return res * scale
        else:
            return res

    return base.filter_dsl(scale_inits_inner,
                           key="get_initial_value",
                           **filter_dsl_kwargs)


def set_weight_init(weight_init, **filter_dsl_kwargs):
    def set_weight_init_inner(hs):
        metadata = hs.args[0]
        if metadata.get("weight"):
            return weight_init(metadata)
        else:
            return hs()

    return base.filter_dsl(set_weight_init_inner,
                           key="get_initial_value",
                           **filter_dsl_kwargs)


def random_normal(loc=0.0, scale=1.0):
    def random_normal_inner(metadata):
        shape = metadata["shape"]
        return np.random.normal(loc=loc, scale=scale, size=shape)

    return random_normal_inner


def random_uniform(low=0.0, high=1.0):
    def random_uniform_inner(metadata):
        shape = metadata["shape"]
        return np.random.uniform(low=low, high=high, size=shape)

    return random_uniform_inner


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


def xavier_normal(metadata):
    shape = metadata["shape"]
    std = xavier_magnitude(shape,
                           metadata["in_axes"],
                           metadata["out_axes"])
    return np.random.normal(scale=std, size=shape)


def xavier_uniform(metadata):
    shape = metadata["shape"]
    magnitude = float(np.sqrt(3)) * xavier_magnitude(shape,
                                                     metadata["in_axes"],
                                                     metadata["out_axes"])
    return np.random.uniform(low=-magnitude, high=magnitude, size=shape)


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


def msr_normal(metadata):
    shape = metadata["shape"]
    std = msr_magnitude(shape,
                        metadata["in_axes"],
                        metadata["out_axes"])
    return np.random.normal(scale=std, size=shape)


def msr_uniform(metadata):
    shape = metadata["shape"]
    magnitude = float(np.sqrt(3)) * msr_magnitude(shape,
                                                  metadata["in_axes"],
                                                  metadata["out_axes"])
    return np.random.uniform(low=-magnitude, high=magnitude, size=shape)


def orthogonal(metadata):
    """
    http://arxiv.org/abs/1312.6120

    implementation from Sander Dieleman
    """
    shape = metadata["shape"]
    out_axes = metadata["out_axes"]

    assert len(shape) >= 2

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


# ############################### other inits ###############################


def set_lstm_forget_bias_init(initial_value=0.):
    def set_lstm_forget_bias_init_inner(hs):
        res = hs()
        return res * 0 + initial_value

    return base.filter_dsl(set_lstm_forget_bias_init_inner,
                           key="get_initial_value",
                           variable_scope=["lstm", "forget", "bias"])
