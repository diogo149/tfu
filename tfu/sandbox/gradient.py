import tensorflow as tf


def ignore_additional_arg(tensor, arg):
    """
    used to combine additional unused information into the tf graph
    """
    arg = tf.convert_to_tensor(arg)
    if arg.dtype == tensor.dtype:
        # tf.cast is a no-op if the tensor has the correct type,
        # so add an additional op here so that get_ignored_arg
        # can unwrap a consistent number of ops
        casted = tf.identity(arg)
    else:
        casted = tf.cast(arg, tensor.dtype)
    return tensor + 0 * tf.reduce_sum(casted)


def get_ignored_arg(op):
    """
    used to retrieve additional unused information from the tf graph
    (stored with ignore_additional_arg)
    """
    orig = op.inputs[0]
    mul = op.inputs[1]
    reduce_sum = mul.op.inputs[1]
    cast = reduce_sum.op.inputs[0]
    arg = cast.op.inputs[0]
    return orig, arg


def ignore_additional_args(tensor, args):
    res = tensor
    # TODO this could be made more efficient by reordering based on shape
    # (if the tf optimizer doesn't handle this case)
    for arg in args:
        res = ignore_additional_arg(res, arg)
    return res


def get_ignored_args(op, num_args):
    args = []
    orig = None
    res_op = op
    for i in range(num_args):
        if i != 0:
            res_op = orig.op
        orig, arg = get_ignored_arg(res_op)
        args.append(arg)
    return orig, list(reversed(args))


@tf.RegisterGradient("GradientReversal")
def _gradient_reversal_grad(unused_op, grad):
    return [tf.neg(grad)]


def gradient_reversal(tensor, name=None):
    g = tf.get_default_graph()
    with g.gradient_override_map({"Identity": "GradientReversal"}):
        return tf.identity(tensor, name=name)


@tf.RegisterGradient("ElementwiseGradientClip")
def _elementwise_gradient_clip_grad(op, grad):
    # ignore the identity op
    op_with_args = op.inputs[0].op
    orig, args = get_ignored_args(op_with_args, 2)
    clip_value_min, clip_value_max = args
    return [tf.clip_by_value(grad,
                             tf.cast(clip_value_min, grad.dtype),
                             tf.cast(clip_value_max, grad.dtype))]


def elementwise_gradient_clip(tensor,
                              clip_value_min,
                              clip_value_max,
                              name=None):
    res = ignore_additional_args(tensor, [clip_value_min,
                                          clip_value_max])
    g = tf.get_default_graph()
    with g.gradient_override_map({"Identity": "ElementwiseGradientClip"}):
        return tf.identity(res, name=name)
