from collections import OrderedDict
import tensorflow as tf

from . import utils
from . import base


# ############################# transformations #############################


def apply_momentum(updates, params=None, momentum=0.9):
    """
    returns a modified update dictionary including momentum
    """
    if params is None:
        params = updates.keys()
    updates = OrderedDict(updates)

    with base.variable_scope("momentum"):
        for param in params:
            with base.variable_scope(param.name):
                velocity = base.get_variable(
                    name="velocity",
                    shape=utils.get_shape_values(param),
                    update_state=True)
                x = momentum * velocity + updates[param]
                updates[velocity] = x - param
                updates[param] = x

    return updates


def apply_nesterov_momentum(updates, params=None, momentum=0.9):
    """
    returns a modified update dictionary including nesterov momentum
    """
    if params is None:
        params = updates.keys()
    updates = OrderedDict(updates)

    with base.variable_scope("nesterov_momentum"):
        for param in params:
            with base.variable_scope(param.name):
                velocity = base.get_variable(
                    name="velocity",
                    shape=utils.get_shape_values(param),
                    update_state=True)
                x = momentum * velocity + updates[param] - param
                updates[velocity] = x
                updates[param] = momentum * x + updates[param]

    return updates


# ################################### sgd ###################################


def sgd(loss_or_grads,
        params=None,
        learning_rate=0.1,
        accumulate_updates=True):
    if params is None:
        params = base.variables(trainable=True)
    if not isinstance(loss_or_grads, (list, tuple)):
        all_grads = tf.gradients(loss_or_grads, params)
    else:
        all_grads = loss_or_grads

    updates = OrderedDict()
    for param, g_t in zip(params, all_grads):
        updates[param] = param - learning_rate * g_t

    if accumulate_updates:
        base.accumulate_update_dict(updates,
                                    train=True,
                                    sgd=True)
    return updates


def momentum(loss_or_grads,
             params=None,
             learning_rate=0.1,
             momentum=0.9,
             accumulate_updates=True):
    sgd_updates = sgd(loss_or_grads=loss_or_grads,
                      params=params,
                      learning_rate=learning_rate,
                      accumulate_updates=False)
    updates = apply_momentum(sgd_updates, momentum=momentum)
    if accumulate_updates:
        base.accumulate_update_dict(updates,
                                    train=True,
                                    sgd=True)
    return updates


def nesterov_momentum(loss_or_grads,
                      params=None,
                      learning_rate=0.1,
                      momentum=0.9,
                      accumulate_updates=True):
    sgd_updates = sgd(loss_or_grads=loss_or_grads,
                      params=params,
                      learning_rate=learning_rate,
                      accumulate_updates=False)
    updates = apply_nesterov_momentum(sgd_updates, momentum=momentum)
    if accumulate_updates:
        base.accumulate_update_dict(updates,
                                    train=True,
                                    sgd=True)
    return updates


def adam(loss_or_grads,
         params=None,
         learning_rate=1e-3,
         beta1=0.9,
         beta2=0.999,
         epsilon=1e-8,
         accumulate_updates=True):
    """
    from http://arxiv.org/abs/1412.6980
    """
    if params is None:
        params = base.variables(trainable=True)
    if not isinstance(loss_or_grads, (list, tuple)):
        all_grads = tf.gradients(loss_or_grads, params)
    else:
        all_grads = loss_or_grads

    with base.variable_scope("adam"):
        t_prev = base.get_variable(shape=(),
                                   name="t")
        updates = OrderedDict()

        t = t_prev + 1
        a_t = learning_rate * tf.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

        for param, g_t in zip(params, all_grads):
            with base.variable_scope(param.name):
                shape = utils.get_shape_values(param)
                m_prev = base.get_variable(shape=shape,
                                           name="m",
                                           update_state=True)
                v_prev = base.get_variable(shape=shape,
                                           name="v",
                                           update_state=True)

                m_t = beta1 * m_prev + (1 - beta1) * g_t
                v_t = beta2 * v_prev + (1 - beta2) * g_t**2
                step = a_t * m_t / (tf.sqrt(v_t) + epsilon)

                updates[m_prev] = m_t
                updates[v_prev] = v_t
                updates[param] = param - step

        updates[t_prev] = t
        if accumulate_updates:
            base.accumulate_update_dict(updates,
                                        train=True,
                                        sgd=True)
        return updates
