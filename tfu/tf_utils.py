import functools
import six
import numpy as np
import tensorflow as tf

from . import base

# ############################## smart reducing ##############################


def smart_reduce(op, iterable):
    iterable = list(iterable)
    if len(iterable) == 1:
        return iterable[0]
    else:
        return functools.reduce(op, iterable[1:], iterable[0])


def smart_add(x, y):
    """
    0-aware add, to prevent computation graph from getting very large
    """
    if x == 0:
        return y
    elif y == 0:
        return x
    else:
        return x + y


def smart_mul(x, y):
    """
    0- and 1- aware multiply, to prevent computation graph from getting very
    large
    """
    if x == 0 or y == 0:
        return 0
    elif x == 1:
        return y
    elif y == 1:
        return x
    else:
        return x * y

smart_sum = functools.partial(smart_reduce, smart_add)
smart_product = functools.partial(smart_reduce, smart_mul)

# ############################# misc tensorflow #############################


def is_tensor(o):
    return isinstance(o, tf.Tensor)


def is_variable(o):
    return isinstance(o, tf.Variable)


def is_symbolic(o):
    return (is_tensor(o) or
            is_variable(o) or
            isinstance(o, tf.IndexedSlices))


def smart_reshape(tensor, shape, name=None):
    """
    similar to tf.reshape, but is better about handling output shape

    eg.
    shape of expression below will be (None, None)
    tf.reshape(tf.constant(np.random.rand(2, 3, 4)), (tf.constant(2), 12))

    shape of expression below will be (2, 12)
    smart_reshape(tf.constant(np.random.rand(2, 3, 4)), (tf.constant(2), 12))
    """
    assert isinstance(shape, list)
    num_symbolic = sum([is_symbolic(dim) for dim in shape])
    if num_symbolic > 1:
        # too much laziness - nothing we can do in this case
        new_shape = shape
    elif num_symbolic == 1:
        # replace symbolic dim with -1, so that shape can be inferred
        # ---
        # why doesn't tensorflow shape inference work in this case?
        new_shape = [-1 if is_symbolic(dim) else dim for dim in shape]
    else:
        # in this case, we can try to do things like replace -1 with correct
        # shape
        # TODO
        new_shape = shape
    return tf.reshape(tensor, new_shape, name)


def ndim(tensor):
    """
    returns number of dimensions for tensor
    """
    return len(tensor.get_shape())


def get_shape_values(tensor):
    """
    returns shape of a tensor as a list of ints or None
    """
    return tensor.get_shape().as_list()


def get_shape_symbolic(tensor):
    """
    returns shape of a tensor as a list of ints or symbolic scalars
    """
    res = get_shape_values(tensor)
    symbolic_shape = tf.shape(tensor)
    for idx, value in enumerate(res):
        if value is None:
            res[idx] = symbolic_shape[idx]
    return res


def get_tensors(graph=None):
    """
    returns all tensors in a graph
    """
    if graph is None:
        graph = tf.get_default_graph()
    tensors = set()
    for op in graph.get_operations():
        tensors.update(op.inputs)
        tensors.update(op.outputs)
    return tensors


def get_by_name(name, collection=None):
    """
    name:
    if a string, look for a substring
    if a list, matches if it is a subsequence of a name split by "/"
    """
    # TODO do we want this to work on regexes instead of exact string matches
    if collection is None:
        collection = tf.GraphKeys.VARIABLES
    coll = tf.get_collection(collection)
    if isinstance(name, six.string_types):
        return [var for var in coll if name in var.name]
    elif isinstance(name, list):
        res = []
        for var in coll:
            name_list = var.name.split("/")
            for subname in name:
                if subname not in name_list:
                    break
                name_list = name_list[name_list.index(subname) + 1:]
            else:
                res.append(var)
        return res
    else:
        raise ValueError("wrong name type: %s" % name)


def tensor_to_variable(tensor):
    """
    converts a read tensor to its corresponding variable
    """
    assert tensor.name.endswith("/read:0")
    var, = get_by_name(tensor.name[:-len("/read:0")])
    assert is_variable(var)
    return var


def dimshuffle(tensor, pattern):
    """
    similar to theano's dimshuffle
    """
    result = tensor

    squeeze_dims = []
    for dim in range(ndim(tensor)):
        if dim not in pattern:
            squeeze_dims.append(dim)
    if squeeze_dims:
        # remove extra dims
        result = tf.squeeze(result,
                            squeeze_dims=squeeze_dims)
        # transform pattern to not have extra dims
        non_x = [dim for dim in pattern if dim != "x"]
        non_x = list(sorted(non_x))
        new_pattern = []
        for dim in pattern:
            if dim == "x":
                new_pattern.append(dim)
            else:
                # map the old index to a new index
                new_pattern.append(non_x.index(dim))
        pattern = new_pattern

    perm = [dim for dim in pattern if dim != "x"]
    result = tf.transpose(result, perm=perm)

    x_dims = [idx
              for idx, dim in enumerate(pattern)
              if dim == "x"]
    for dim in x_dims:
        result = tf.expand_dims(result, dim=dim)

    return result


def tensor_index(tensor, *idxs):
    # allow mutation
    idxs = list(idxs)
    # pad to the required number of dimensions
    idxs += [slice(None)] * (ndim(tensor) - len(idxs))
    # TODO put more stuff here
    # see https://github.com/tensorflow/tensorflow/issues/206
    return tensor[tuple(idxs)]


def flatten(tensor, outdim=1):
    assert outdim >= 1
    shape = get_shape_symbolic(tensor)
    remaining_shape = smart_product(shape[outdim - 1:])
    return smart_reshape(tensor, shape[:outdim - 1] + [remaining_shape])


def initialize_uninitialized_variables(session):
    var_list = tf.all_variables()
    init_list = []
    for var in var_list:
        if not session.run(tf.is_variable_initialized(var)):
            init_list.append(var.initializer)
    session.run(tf.group(*init_list))


def sequentially_initialize_all_variables(session):
    var_list = tf.all_variables()
    for var in var_list:
        session.run(var.initializer)


def list_reduce_mean(tensors):
    assert len(tensors) > 0
    if len(tensors) == 1:
        return tensors[0]
    else:
        # TODO benchmark different indicies to do this over
        packed = tf.pack(tensors, axis=0)
        return tf.reduce_mean(packed, reduction_indices=[0])


def dot(a, b, **kwargs):
    """
    wrapper around tf.matmul to take non-matrix tensors
    """
    assert ndim(a) >= 1
    assert ndim(b) == 2  # TODO implement this
    last_shape = get_shape_symbolic(a)[-1]
    tmp_a = tf.reshape(a, [-1, last_shape])
    res = tf.matmul(tmp_a, b)
    res_shape = (get_shape_symbolic(a)[:-1] +
                 [get_shape_symbolic(b)[1]])
    return smart_reshape(res, res_shape)


@base.hooked
def linear(name, tensor, num_units):
    with tf.variable_scope(name):
        num_inputs = get_shape_values(tensor)[-1]
        W = base.get_variable(name="W",
                              shape=(num_inputs, num_units),
                              dtype=tensor.dtype,
                              collections=[tf.GraphKeys.VARIABLES,
                                           tf.GraphKeys.WEIGHTS],
                              in_axes=[0],
                              out_axes=[1])
        return dot(tensor, W)


@base.hooked
def add_bias(name, tensor, axis=-1):
    with tf.variable_scope(name):
        # TODO allow for multiple axes
        num_units = get_shape_values(tensor)[axis]
        b = base.get_variable(name="b",
                              shape=(num_units,),
                              dtype=tensor.dtype,
                              collections=[tf.GraphKeys.VARIABLES,
                                           tf.GraphKeys.BIASES])
        if axis == -1:
            return tensor + b
        else:
            pattern = ["x"] * ndim(tensor)
            pattern[axis] = 0
            return tensor + dimshuffle(b, pattern)



@base.hooked
def affine(name, tensor, num_units):
    with tf.variable_scope(name):
        return add_bias("bias", linear("linear", tensor, num_units))


@base.hooked
def multi_linear(names, tensor, num_units):
    with tf.variable_scope("multi_linear"):
        if isinstance(num_units, int):
            num_units = [num_units] * len(names)
        assert len(num_units) == len(names)
        num_inputs = get_shape_values(tensor)[-1]
        Ws = []
        for name, n in zip(names, num_units):
            with tf.variable_scope(name):
                W = base.get_variable(name="W",
                                      shape=(num_inputs, n),
                                      dtype=tensor.dtype,
                                      collections=[tf.GraphKeys.VARIABLES,
                                                   tf.GraphKeys.WEIGHTS],
                                      in_axes=[0],
                                      out_axes=[1])
                Ws.append(W)
        W_concat = tf.concat(concat_dim=1, values=Ws)
        combined = dot(tensor, W_concat)
        results = []
        begin = [0] * ndim(combined)
        size = [-1] * ndim(combined)
        for n in num_units:
            size[-1] = n
            res = tf.slice(combined, begin, size)
            results.append(res)
            begin[-1] += n
        return results


@base.hooked
def multi_affine(names, tensor, num_units):
    results = multi_linear(names, tensor, num_units)
    new_results = []
    for name, result in zip(names, results):
        with tf.variable_scope(name):
            new_results.append(add_bias("bias", result))
    return new_results


@base.hooked
def conv2d(name,
           tensor,
           num_filters,
           filter_size,
           strides=(1, 1),
           padding="SAME",
           data_format="NHWC"):
    assert isinstance(filter_size, tuple)
    assert ndim(tensor) == 4
    with tf.variable_scope(name):
        if data_format == "NHWC":
            strides = (1,) + strides + (1,)
            num_channels = get_shape_values(tensor)[3]
            filter_shape = filter_size + (num_channels, num_filters)
            W = base.get_variable(name="W",
                                  shape=filter_shape,
                                  dtype=tensor.dtype,
                                  collections=[tf.GraphKeys.VARIABLES,
                                               tf.GraphKeys.WEIGHTS],
                                  in_axes=[2],
                                  out_axes=[3])
        elif data_format == "NCHW":
            # TODO are these right?
            strides = (1, 1) + strides
            num_channels = get_shape_values(tensor)[1]
            filter_shape = filter_size + (num_channels, num_filters)
            # filter_shape = (num_channels,) + filter_size + (num_filters, )
            W = base.get_variable(name="W",
                                  shape=filter_shape,
                                  dtype=tensor.dtype,
                                  collections=[tf.GraphKeys.VARIABLES,
                                               tf.GraphKeys.WEIGHTS],
                                  in_axes=[2],
                                  out_axes=[3])
        else:
            raise ValueError

        return tf.nn.conv2d(input=tensor,
                            filter=W,
                            strides=strides,
                            padding=padding,
                            data_format=data_format,
                            name=name)


@base.hooked
def max_pool2d(tensor,
               ksize,
               strides=None,
               padding="SAME",
               data_format="NHWC",
               name=None):
    if strides is None:
        strides = ksize
    assert len(strides) == len(ksize) == 2
    assert data_format == "NHWC"
    ksize = (1,) + ksize + (1,)
    strides = (1,) + strides + (1,)
    return tf.nn.max_pool(value=tensor,
                          ksize=ksize,
                          strides=strides,
                          padding=padding,
                          data_format=data_format,
                          name=name)


@base.hooked
def avg_pool2d(tensor,
               ksize,
               strides=None,
               padding="SAME",
               data_format="NHWC",
               name=None):
    if strides is None:
        strides = ksize
    assert len(strides) == len(ksize) == 2
    assert data_format == "NHWC"
    ksize = (1,) + ksize + (1,)
    strides = (1,) + strides + (1,)
    return tf.nn.avg_pool(value=tensor,
                          ksize=ksize,
                          strides=strides,
                          padding=padding,
                          data_format=data_format,
                          name=name)


@base.hooked
def global_avg_pool2d(tensor, data_format="NHWC", name=None):
    assert data_format == "NHWC"
    ksize = [1] + get_shape_symbolic(tensor)[1:3] + [1]
    res = tf.nn.avg_pool(value=tensor,
                         ksize=ksize,
                         strides=ksize,
                         padding="VALID",
                         data_format=data_format,
                         name=name)
    return tf.squeeze(res, squeeze_dims=[1, 2])


@base.hooked
def batch_normalization(name, tensor, epsilon=1e-4):
    with tf.variable_scope(name):
        num_units = get_shape_values(tensor)[1]
        beta = base.get_variable("beta",
                                 shape=[num_units],
                                 dtype=tensor.dtype,
                                 initializer=tf.constant_initializer(0.0),
                                 collections=[tf.GraphKeys.VARIABLES,
                                              tf.GraphKeys.BIASES,
                                              "bn_beta"])
        gamma = base.get_variable("gamma",
                                  shape=[num_units],
                                  dtype=tensor.dtype,
                                  initializer=tf.constant_initializer(1.0),
                                  collections=[tf.GraphKeys.VARIABLES,
                                               "bn_gamma"])
        mean, variance = tf.nn.moments(x=tensor,
                                       axes=[dim for dim in range(ndim(tensor))
                                             if dim != 1],
                                       keep_dims=True)
        return tf.nn.batch_normalization(x=tensor,
                                         mean=mean,
                                         variance=variance,
                                         offset=beta,
                                         scale=gamma,
                                         variance_epsilon=epsilon)


@base.hooked
def rnn_reduce(name,
               rnn_fn,
               tensors,
               initial_state):
    # use set to make sure all tensors have same number of steps
    num_steps, = set(map(lambda x: get_shape_values(x)[0], tensors))
    state = initial_state
    outputs = []
    with tf.variable_scope(name):
        for time_step in range(num_steps):
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()
            state = rnn_fn([tensor_index(tensor, time_step)
                            for tensor in tensors],
                           state)
            outputs.append(state)
    return outputs


@base.hooked
def simple_rnn_step(tensors, state):
    # TODO have different fn to also precompute input
    with tf.variable_scope("simple_rnn"):
        x, = tensors
        h = state
        assert is_symbolic(x)
        assert is_symbolic(h)
        num_units = get_shape_values(h)[-1]
        logit = add_bias("bias",
                         linear("x_to_h", x, num_units) +
                         linear("h_to_h", h, num_units))

        @base.hooked
        def nonlinearity(logit):
            return tf.tanh(logit)

        return nonlinearity(logit=logit)


@base.hooked
def lstm_step(tensors, state):
    # TODO group linear operations for more efficiency
    with tf.variable_scope("lstm"):
        x, = tensors
        h = state["h"]
        c = state["c"]
        assert is_symbolic(x)
        assert is_symbolic(h)
        assert is_symbolic(c)
        num_units = get_shape_values(h)[-1]
        assert get_shape_values(c)[-1] == num_units
        forget_logit = add_bias("forget_bias",
                                linear("forget_x", x, num_units) +
                                linear("forget_h", h, num_units))
        input_logit = add_bias("input_bias",
                               linear("input_x", x, num_units) +
                               linear("input_h", h, num_units))
        output_logit = add_bias("output_bias",
                                linear("output_x", x, num_units) +
                                linear("output_h", h, num_units))
        update_logit = add_bias("update_bias",
                                linear("update_x", x, num_units) +
                                linear("update_h", h, num_units))
        f = tf.nn.sigmoid(forget_logit)
        i = tf.nn.sigmoid(input_logit)
        o = tf.nn.sigmoid(output_logit)
        u = tf.tanh(update_logit)
        new_c = f * c + i * u
        new_h = tf.tanh(new_c) * o
        return {"h": new_h, "c": new_c}


@base.hooked
def simple_rnn_layer(name, tensor, state):
    with tf.variable_scope(name):
        x = tensor
        h = state
        assert is_symbolic(x)
        assert is_symbolic(h)
        num_units = get_shape_values(h)[-1]
        z = affine("x_to_h", x, num_units=num_units)

        @base.hooked
        def nonlinearity(logit):
            return tf.tanh(logit)

        def _step(tensors, state):
            z_, = tensors
            h_ = state
            logit = z_ + linear("h_to_h", h_, num_units=num_units)
            return nonlinearity(logit=logit)

        outputs = rnn_reduce("rnn", _step, [z], h)
        return outputs


@base.hooked
def lstm_layer(name, tensor, state):
    with tf.variable_scope(name):
        x = tensor
        h = state["h"]
        c = state["c"]
        assert is_symbolic(x)
        assert is_symbolic(h)
        assert is_symbolic(c)
        num_units = get_shape_values(h)[-1]
        assert get_shape_values(c)[-1] == num_units

        # broadcast initial state to have batch size
        batch_size = get_shape_symbolic(x)[1]
        zero_state = tf.zeros((batch_size, num_units), dtype=x.dtype)
        state = {"h": zero_state + h, "c": zero_state + c}

        def _step(tensors, state_):
            x_, = tensors
            h_ = state_["h"]
            c_ = state_["c"]
            results = multi_affine(["h_to_forgetgate",
                                    "h_to_cellgate",
                                    "h_to_outputgate",
                                    "h_to_inputgate"],
                                   tf.concat(concat_dim=1, values=[x_, h_]),
                                   num_units=num_units)

            f = tf.nn.sigmoid(results[0])
            u = tf.tanh(results[1])
            o = tf.nn.sigmoid(results[2])
            i = tf.nn.sigmoid(results[3])
            new_c = f * c_ + i * u
            new_h = tf.tanh(new_c) * o
            return {"h": new_h, "c": new_c}

        outputs = rnn_reduce("rnn", _step, [x], state)
        return outputs


@base.hooked
def binary_cross_entropy(pred, target):
    return -(target * tf.log(pred) + (1 - target) * tf.log(1 - pred))


@base.hooked
def categorical_cross_entropy(pred, target, axis=1):
    if target.dtype == tf.int64:
        num_targets = get_shape_values(pred)[axis]
        target = tf.one_hot(indices=target,
                            depth=num_targets,
                            axis=axis,
                            dtype=pred.dtype)
    return -tf.reduce_sum(target * tf.log(pred),
                          reduction_indices=[axis])


@base.hooked
def softmax_cross_entropy_with_logits(pred_logits, target, axis=1):
    if ndim(pred_logits) == 2 and axis == 1:
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
        target = tf.argmax(target, dimension=axis)
    class_preds = tf.argmax(pred, dimension=axis)
    return tf.cast(tf.equal(class_preds, target),
                   pred.dtype)
