import tensorflow as tf

from . import utils
from . import base


def default_kwargs(kwargs):
    def inner(hs):
        new_kwargs = dict(kwargs)
        new_kwargs.update(hs.kwargs)
        hs.kwargs = new_kwargs
        return hs()

    return inner


def default_kwargs_dsl(kwargs, key):
    return base.filter_dsl(default_kwargs(kwargs), key=key)


def override_kwargs(kwargs):
    def inner(hs):
        new_kwargs = dict(hs.kwargs)
        new_kwargs.update(kwargs)
        hs.kwargs = new_kwargs
        return hs()

    return inner


def override_kwargs_dsl(kwargs, key):
    return base.filter_dsl(override_kwargs(kwargs), key=key)


def auto_initialize_variables(session):
    """
    NOTE: this will only initialize tfu variables, not external ones
    (eg. Adam state)
    """

    def hook(hs):
        res = hs()
        if (utils.is_variable(res) and
                not session.run(tf.is_variable_initialized(res))):
            session.run(res.initializer)
        return res

    return base.filter_dsl(hook, key="get_variable")
