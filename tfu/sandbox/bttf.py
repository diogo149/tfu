import collections
import tensorflow as tf
import tfu


def _backprop_to_the_future_mean_impl(current_mean,
                                      rolling_mean,
                                      rolling_grad):
    return rolling_mean


@tf.RegisterGradient("BackpropToTheFutureMean")
def _backprop_to_the_future_mean_grad(op, grad):
    current_mean, rolling_mean, rolling_grad = op.inputs
    return (rolling_grad,
            # tf.zeros_like(rolling_mean),
            None,
            None,
            )


def backprop_to_the_future_mean(current_mean,
                                rolling_mean=None,
                                rolling_grad=None,
                                mean_initializer=0,
                                name="bttf_mean"):
    with tfu.variable_scope(name):
        if rolling_mean is None:
            rolling_mean = tfu.get_variable(
                name="rolling_mean",
                shape=current_mean.get_shape(),
                dtype=current_mean.dtype,
                initial_value=mean_initializer,
                trainable=False,
            )
        if rolling_grad is None:
            rolling_grad = tfu.get_variable(
                name="rolling_grad",
                shape=current_mean.get_shape(),
                dtype=current_mean.dtype,
                trainable=False,
            )

        g = tf.get_default_graph()
        with g.gradient_override_map({"PyFuncStateless": "BackpropToTheFutureMean"}):
            out, = tf.py_func(func=_backprop_to_the_future_mean_impl,
                              inp=[current_mean,
                                   rolling_mean,
                                   rolling_grad],
                              Tout=[current_mean.dtype],
                              stateful=False,
                              name=None)
            out.set_shape(current_mean.get_shape())

        tf.add_to_collection("bttf_current_mean", current_mean)
        tf.add_to_collection("bttf_rolling_mean", rolling_mean)
        tf.add_to_collection("bttf_rolling_grad", rolling_grad)
        tf.add_to_collection("bttf_mean", out)
        return out


def raw_backprop_to_the_future_mean_updates(loss, alpha1, alpha2=None):
    if alpha2 is None:
        alpha2 = alpha1

    results = tf.get_collection("bttf_mean")
    rm_to_cm = collections.defaultdict(list)
    rg_to_res = collections.defaultdict(list)
    for res in results:
        current_mean, rolling_mean, rolling_grad = res.op.inputs
        rm = tfu.tensor_to_variable(rolling_mean)
        rg = tfu.tensor_to_variable(rolling_grad)
        rm_to_cm[rm].append(current_mean)
        rg_to_res[rg].append(res)

    updates = []

    for rm, cms in rm_to_cm.items():
        avg_cm = tfu.list_reduce_mean(cms)
        updates.append((rm, alpha1 * rm + (1 - alpha1) * avg_cm))

    res_to_grad = dict(zip(results, tf.gradients(loss, results)))
    for rg, ress in rg_to_res.items():
        grads = [res_to_grad[res] for res in ress]
        avg_grad = tfu.list_reduce_mean(grads)
        updates.append((rg, alpha2 * rg + (1 - alpha2) * avg_grad))

    return updates


def backprop_to_the_future_mean_updates(loss, alpha1, alpha2=None):
    raw_updates = raw_backprop_to_the_future_mean_updates(loss, alpha1, alpha2)
    updates = []
    with tf.control_dependencies([loss]):
        for var, update in raw_updates:
            updates.append(var.assign(update))
    return tf.group(*updates)

# shortcuts
bttf_mean = backprop_to_the_future_mean
bttf_mean_updates = backprop_to_the_future_mean_updates
