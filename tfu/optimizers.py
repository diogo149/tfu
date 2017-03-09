import tensorflow as tf

from . import utils
from . import base


def sgd(cost, parameters=None, learning_rate=0.01):
    if parameters is None:
        parameters = tf.trainable_variables()

    grads = tf.gradients(cost, parameters)

    all_updates = []
    for grad, param in zip(grads, parameters):
        assigned = tf.assign_sub(param, learning_rate * grad)
        all_updates.append(assigned)

    # FIXME should we be returning an op instead of updates
    update_op = tf.group(*all_updates)
    return update_op


def adam(cost,
         parameters=None,
         learning_rate=1e-3,
         beta1=0.9,
         beta2=0.999,
         epsilon=1e-8):
    if parameters is None:
        parameters = tf.trainable_variables()

    grads = tf.gradients(cost, parameters)
    all_updates = []
    zero_init = tf.constant_initializer(0.)
    with tf.variable_scope("adam"):
        t_prev = base.get_variable("t",
                                   shape=(),
                                   initializer=zero_init)
        t = tf.assign_add(t_prev, 1)
        all_updates.append(t)

        for grad, param in zip(grads, parameters):
            with tf.variable_scope(param.name.replace(":", "_")):
                param_shape = utils.get_shape_values(param)
                m_prev = base.get_variable("m",
                                           shape=param_shape,
                                           initializer=zero_init)
                v_prev = base.get_variable("v",
                                           shape=param_shape,
                                           initializer=zero_init)
                m = tf.assign(m_prev,
                              m_prev * beta1 + grad * (1 - beta1))
                v = tf.assign(v_prev,
                              v_prev * beta2 + tf.square(grad) * (1 - beta2))

                numerator = learning_rate * m / (1 - tf.pow(beta1, t))
                denominator = tf.sqrt(v / (1 - tf.pow(beta2, t))) + epsilon
                assigned = tf.assign_sub(param, numerator / denominator)
                all_updates += [m, v, assigned]

    # FIXME should we be returning an op instead of updates
    update_op = tf.group(*all_updates)
    return update_op
