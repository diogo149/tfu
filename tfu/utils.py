import functools
import io
import numbers
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

# ############################### misc python ###############################


def is_number(x):
    return isinstance(x, numbers.Number)


def is_integral(x):
    return isinstance(x, numbers.Integral)


def identity(x):
    return x


# ################################ misc numpy ################################


def is_ndarray(x):
    return isinstance(x, np.ndarray)


# ############################### misc datamap ###############################


def datamap_merge(datamaps, scalar_merge="mean"):
    """
    concatenates datamaps along axis 0, and merges scalars using a given method
    """
    if scalar_merge == "mean":
        scalar_merge = np.mean
    elif scalar_merge == "identity":
        scalar_merge = identity
    res = {}
    for key in datamaps[0].keys():  # assumes at least 1 datamap
        outputs = [r[key] for r in datamaps]
        if is_ndarray(outputs[0]) and outputs[0].shape:
            res[key] = np.concatenate(outputs)
        else:
            res[key] = scalar_merge(outputs)
    return res


# ############################# misc tensorflow #############################

def full_variable_name(name, variable_scope=None):
    if variable_scope is None:
        variable_scope = tf.get_variable_scope()
    vs_name = variable_scope.name
    if vs_name == "":
        full_name = name + ":0"
    elif vs_name.endswith("/"):
        full_name = vs_name + name + ":0"
    else:
        full_name = vs_name + "/" + name + ":0"
    return full_name


def is_tensor(o):
    return isinstance(o, tf.Tensor)


def is_variable(o):
    return isinstance(o, tf.Variable)


def is_symbolic(o):
    return (is_tensor(o) or
            is_variable(o) or
            isinstance(o, tf.IndexedSlices))


def smart_reshape(x, shape, name=None):
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
    return tf.reshape(x, new_shape, name)


def ndim(x):
    """
    returns number of dimensions for tensor
    """
    return len(x.get_shape())


def get_shape_values(x):
    """
    returns shape of a tensor as a list of ints or None
    """
    return x.get_shape().as_list()


def get_shape_symbolic(x):
    """
    returns shape of a tensor as a list of ints or symbolic scalars
    """
    res = get_shape_values(x)
    symbolic_shape = tf.shape(x)
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
        collection = tf.GraphKeys.GLOBAL_VARIABLES
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


def tensor_to_variable(x):
    """
    converts a read tensor to its corresponding variable
    """
    assert x.name.endswith("/read:0")
    var, = get_by_name(x.name[:-len("/read:0")])
    assert is_variable(var)
    return var


def dimshuffle(x, pattern):
    """
    similar to theano's dimshuffle
    """
    result = x
    # normalize pattern type
    assert isinstance(pattern, (list, tuple))
    pattern = tuple(pattern)

    # optimization: if pattern is all "x"'s followed by all the axes
    # in order, simply return the original tensor
    if (set(pattern[:-ndim(x)]) == {"x"} and
            pattern[-ndim(x):] == tuple(range(ndim(x)))):
        return x

    squeeze_dims = []
    for dim in range(ndim(x)):
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


def tensor_index(x, *idxs):
    # allow mutation
    idxs = list(idxs)
    # pad to the required number of dimensions
    idxs += [slice(None)] * (ndim(x) - len(idxs))
    # TODO put more stuff here
    # see https://github.com/tensorflow/tensorflow/issues/206
    return x[tuple(idxs)]


def flatten(x, outdim=1):
    assert outdim >= 1
    shape = get_shape_symbolic(x)
    remaining_shape = smart_product(shape[outdim - 1:])
    return smart_reshape(x, shape[:outdim - 1] + [remaining_shape])


def initialize_uninitialized_variables(session):
    var_list = tf.all_variables()
    init_list = []
    for var in var_list:
        if not session.run(tf.is_variable_initialized(var)):
            init_list.append(var.initializer)
    session.run(tf.group(*init_list))


def sequential_global_variables_initializer(session):
    var_list = tf.all_variables()
    for var in var_list:
        session.run(var.initializer)


def list_reduce_mean(xs):
    assert len(xs) > 0
    if len(xs) == 1:
        return xs[0]
    else:
        # TODO benchmark different indicies to do this over
        packed = tf.pack(xs, axis=0)
        return tf.reduce_mean(packed, axis=[0])


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


def swapaxes(x, axes):
    ax1, ax2 = axes
    perm = list(range(ndim(x)))
    if perm[ax1] == perm[ax2]:
        # don't perform transpose if axes are the same
        return x
    perm[ax1], perm[ax2] = perm[ax2], perm[ax1]
    return tf.transpose(x, perm=perm)


def _axis_apply(old_axis, new_axis, fn, x):
    """
    util for applying a function on one axis to other axes
    """
    axes = [old_axis, new_axis]
    tmp = swapaxes(x, axes)
    res = fn(tmp)
    return swapaxes(res, axes)


def sort(x, axis=-1, increasing=True):
    assert increasing  # TODO

    def sort_inner(x):
        return tf.nn.top_k(x, k=get_shape_symbolic(x)[-1]).values

    return _axis_apply(old_axis=-1,
                       new_axis=axis,
                       fn=sort_inner,
                       x=x)


def sort_by(x, vector, axis=-1, increasing=True):
    """
    sorts along a given axis, according to an index
    """
    assert increasing  # TODO
    vec_len, = get_shape_symbolic(vector)
    indices = tf.nn.top_k(vector, k=vec_len).indices

    def sort_by_inner(x):
        return tf.gather(x, indices)

    return _axis_apply(old_axis=0,
                       new_axis=axis,
                       fn=sort_by_inner,
                       x=x)

# ############################## summary utils ##############################


def dict_to_scalar_summary(x):
    values = []
    for k, v in sorted(x.items()):
        values.append(tf.summary.Summary.Value(tag=k,
                                               simple_value=v))
    return tf.summary.Summary(value=values)


def scalar_summary_to_dict(summ):
    res = {}
    assert isinstance(summ, tf.summary.Summary)
    for value in summ.value:
        assert isinstance(value, tf.summary.Summary.Value)
        if hasattr(value, "simple_value") and value.histo.ByteSize() == 0:
            res[value.tag] = value.simple_value
    return res

# ############################## plotting utils ##############################


def pyplot_to_png_bytes():
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf.getvalue()


def png_bytes_to_image_summary(name, png_bytes=None):
    """
    returns a placeholder as well as the summary if png_bytes is None
    """
    if png_bytes is None:
        _bytes = tf.placeholder(tf.string)
    else:
        _bytes = png_bytes

    # convert png to TF image
    img = tf.image.decode_png(_bytes, channels=4)
    # add batch dimension
    imgs = tf.expand_dims(img, 0)
    # create summary
    summary_op = tf.summary.image(name, imgs)

    if png_bytes is None:
        return _bytes, summary_op
    else:
        return summary_op
