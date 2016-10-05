import functools
import six
import numpy as np
import tensorflow as tf

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