import tensorflow as tf

from . import utils
from . import base


@base.hooked
def linear(tensor, num_units, name=None):
    with base.variable_scope(name):
        with base.variable_scope("linear"):
            num_inputs = utils.get_shape_values(tensor)[-1]
            W = base.get_variable(name="W",
                                  shape=(num_inputs, num_units),
                                  dtype=tensor.dtype,
                                  trainable=True,
                                  weight=True,
                                  in_axes=[0],
                                  out_axes=[1])
            return utils.dot(tensor, W)


@base.hooked
def add_bias(tensor, axis=-1, name=None):
    with base.variable_scope(name):
        with base.variable_scope("add_bias"):
            # TODO allow for multiple axes
            num_units = utils.get_shape_values(tensor)[axis]
            b = base.get_variable(name="b",
                                  shape=(num_units,),
                                  dtype=tensor.dtype,
                                  trainable=True,
                                  bias=True)
            pattern = ["x"] * utils.ndim(tensor)
            pattern[axis] = 0
            return tensor + utils.dimshuffle(b, pattern)


@base.hooked
def learned_scaling(tensor, axis=-1, exponential_scale=False, name=None):
    with base.variable_scope(name):
        with base.variable_scope("learned_scaling"):
            # TODO allow for multiple axes
            num_units = utils.get_shape_values(tensor)[axis]
            if exponential_scale:
                log_scale = base.get_variable(name="log_scale",
                                              shape=(num_units,),
                                              dtype=tensor.dtype,
                                              trainable=True)
                scale = tf.exp(log_scale)
            else:
                scale = base.get_variable(name="scale",
                                          shape=(num_units,),
                                          dtype=tensor.dtype,
                                          initial_value=1.0,
                                          trainable=True)
            pattern = ["x"] * utils.ndim(tensor)
            pattern[axis] = 0
            return tensor * utils.dimshuffle(scale, pattern)


@base.hooked
def affine(tensor, num_units, name=None):
    with base.variable_scope(name):
        with base.variable_scope("affine"):
            return add_bias(linear(tensor, num_units))


def split_axis(tensor, axis, sizes):
    """
    splits a tensor along a given axis according to the given sizes
    """
    results = []
    begin = [0] * utils.ndim(tensor)
    size = [-1] * utils.ndim(tensor)
    for s in sizes:
        size[axis] = s
        res = tf.slice(tensor, begin, size)
        results.append(res)
        begin[axis] += s
    return results


@base.hooked
def multi_linear(names, tensor, num_units, split_output=True):
    with base.variable_scope("multi_linear"):
        if isinstance(num_units, int):
            num_units = [num_units] * len(names)
        assert len(num_units) == len(names)
        num_inputs = utils.get_shape_values(tensor)[-1]
        Ws = []
        for name, n in zip(names, num_units):
            with base.variable_scope(name):
                W = base.get_variable(name="W",
                                      shape=(num_inputs, n),
                                      dtype=tensor.dtype,
                                      trainable=True,
                                      weight=True,
                                      in_axes=[0],
                                      out_axes=[1])
                Ws.append(W)
        W_concat = tf.concat(values=Ws, axis=1)
        combined = utils.dot(tensor, W_concat)
        if split_output:
            return split_axis(combined, axis=-1, sizes=num_units)
        else:
            return combined


@base.hooked
def multi_affine(names, tensor, num_units):
    results = multi_linear(names, tensor, num_units)
    new_results = []
    for name, result in zip(names, results):
        with base.variable_scope(name):
            new_results.append(add_bias(result))
    return new_results


@base.hooked
def conv2d(tensor,
           num_filters,
           filter_size,
           strides=(1, 1),
           padding="SAME",
           name=None):
    assert isinstance(filter_size, tuple)
    assert utils.ndim(tensor) == 4
    with base.variable_scope(name):
        with base.variable_scope("conv2d"):
            strides = (1,) + strides + (1,)
            num_channels = utils.get_shape_values(tensor)[3]
            filter_shape = filter_size + (num_channels, num_filters)
            W = base.get_variable(name="W",
                                  shape=filter_shape,
                                  dtype=tensor.dtype,
                                  trainable=True,
                                  weight=True,
                                  in_axes=[2],
                                  out_axes=[3])
            return tf.nn.conv2d(input=tensor,
                                filter=W,
                                strides=strides,
                                padding=padding,
                                name=name)


@base.hooked
def conv2d_transpose(tensor,
                     num_filters,
                     filter_size,
                     strides=(1, 1),
                     padding="SAME",
                     name=None):
    assert isinstance(filter_size, tuple)
    assert utils.ndim(tensor) == 4
    with base.variable_scope(name):
        with base.variable_scope("conv2d"):
            strides = (1,) + strides + (1,)
            num_channels = utils.get_shape_values(tensor)[3]
            filter_shape = filter_size + (num_channels, num_filters)
            W = base.get_variable(name="W",
                                  shape=filter_shape,
                                  dtype=tensor.dtype,
                                  trainable=True,
                                  weight=True,
                                  in_axes=[2],
                                  out_axes=[3])
            output_shape = utils.get_shape_symbolic(tensor)
            for idx, stride in enumerate(strides):
                if stride != 1:
                    output_shape[idx] *= stride
            return tf.nn.conv2d_transpose(value=tensor,
                                          filter=W,
                                          output_shape=output_shape,
                                          strides=strides,
                                          padding=padding,
                                          name=name)


@base.hooked
def max_pool2d(tensor,
               ksize,
               strides=None,
               padding="SAME",
               name=None):
    if strides is None:
        strides = ksize
    assert len(strides) == len(ksize) == 2
    ksize = (1,) + ksize + (1,)
    strides = (1,) + strides + (1,)
    return tf.nn.max_pool(value=tensor,
                          ksize=ksize,
                          strides=strides,
                          padding=padding,
                          name=name)


@base.hooked
def avg_pool2d(tensor,
               ksize,
               strides=None,
               padding="SAME",
               name=None):
    if strides is None:
        strides = ksize
    assert len(strides) == len(ksize) == 2
    ksize = (1,) + ksize + (1,)
    strides = (1,) + strides + (1,)
    return tf.nn.avg_pool(value=tensor,
                          ksize=ksize,
                          strides=strides,
                          padding=padding,
                          name=name)


@base.hooked
def global_avg_pool2d(tensor, name=None):
    ksize = [1] + utils.get_shape_symbolic(tensor)[1:3] + [1]
    res = tf.nn.avg_pool(value=tensor,
                         ksize=ksize,
                         strides=ksize,
                         padding="VALID",
                         name=name)
    return tf.squeeze(res, axis=[1, 2])


@base.hooked
def rnn_reduce(rnn_fn,
               tensors,
               initial_state,
               name=None):
    # use set to make sure all tensors have same number of steps
    num_steps, = set(map(lambda x: utils.get_shape_values(x)[0], tensors))
    state = initial_state
    outputs = []
    with base.variable_scope(name):
        for time_step in range(num_steps):
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()
            state = rnn_fn([utils.tensor_index(tensor, time_step)
                            for tensor in tensors],
                           state)
            outputs.append(state)
    return outputs


@base.hooked
def binary_cross_entropy(pred, target):
    return -(target * tf.log(pred) + (1 - target) * tf.log(1 - pred))


@base.hooked
def categorical_cross_entropy(pred, target, axis=1):
    if target.dtype == tf.int64:
        num_targets = utils.get_shape_values(pred)[axis]
        target = tf.one_hot(indices=target,
                            depth=num_targets,
                            axis=axis,
                            dtype=pred.dtype)
    return -tf.reduce_sum(target * tf.log(pred),
                          axis=[axis])


@base.hooked
def softmax_cross_entropy_with_logits(pred_logits, target, axis=1):
    if utils.ndim(pred_logits) == 2 and axis == 1:
        if target.dtype in {tf.int32, tf.int64}:
            return tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=pred_logits,
                labels=target)
        else:
            return tf.nn.softmax_cross_entropy_with_logits(
                logits=pred_logits,
                labels=target)
    else:
        # TODO could reshape and use built-in ops
        # TODO make softmax that takes in axis
        assert False
        pred = tf.nn.softmax(pred_logits)
        return categorical_cross_entropy(pred=pred, target=target)


@base.hooked
def binary_accuracy(pred, target):
    return tf.cast(tf.equal(target,
                            tf.cast(pred > 0.5, pred.dtype)),
                   pred.dtype)


@base.hooked
def categorical_accuracy(pred, target, axis=1):
    if target.dtype != tf.int64:
        # NOTE: assuming that it is one-hot encoded if not int64
        target = tf.argmax(target, axis=axis)
    class_preds = tf.argmax(pred, axis=axis)
    return tf.cast(tf.equal(class_preds, target),
                   pred.dtype)


@base.hooked
def leaky_relu(x, leak=0, name="leaky_relu"):
    """
    from https://github.com/tensorflow/tensorflow/issues/4079
    """
    with base.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)
