import re
import tensorflow as tf

from . import utils
from . import base


def default_kwargs(kwargs):
    def default_kwargs_inner(hs):
        new_kwargs = dict(kwargs)
        new_kwargs.update(hs.kwargs)
        hs.kwargs = new_kwargs
        return hs()

    return default_kwargs_inner


def default_kwargs_dsl(kwargs, key):
    return base.filter_dsl(default_kwargs(kwargs), key=key)


def override_kwargs(kwargs):
    def override_kwargs_inner(hs):
        new_kwargs = dict(hs.kwargs)
        new_kwargs.update(kwargs)
        hs.kwargs = new_kwargs
        return hs()

    return override_kwargs_inner


def override_kwargs_dsl(kwargs, key):
    return base.filter_dsl(override_kwargs(kwargs), key=key)


def replace_fn_dsl(new_fn, key):
    """
    replaces one function with another
    """

    def replace_fn_dsl_inner(hs):
        hs.fn = new_fn
        return hs()

    return base.filter_dsl(hook=replace_fn_dsl_inner,
                           key=key)


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


def reuse_variables(variable_scope=None,
                    replace=(),
                    create_if_nonexistent=False):
    """
    NOTE: this should probably be the inner-most hook for get_variable
    """

    def reuse_variables_inner(hs):
        name = hs.kwargs["name"]
        full_name = utils.full_variable_name(name)
        # perform replacement
        for r in replace:
            if isinstance(r, tuple):
                # re.sub DSL
                full_name = re.sub(r[0], r[1], full_name)
            else:
                # otherwise, it's an arbitrary function
                full_name = r(full_name)

        if full_name in base.default_graph_state().variables:
            var = base.default_graph_state().variables[full_name]
            # safefy checks
            if hs.kwargs.get("shape"):
                shape = hs.kwargs["shape"]
                assert utils.get_shape_values(var) == list(shape)
            if hs.kwargs.get("dtype"):
                dtype = hs.kwargs["dtype"]
                assert var.dtype == dtype
            return var
        else:
            if create_if_nonexistent:
                return hs()
            else:
                raise ValueError("variable with name %s doesn't exist" %
                                 full_name)

    return base.filter_dsl(reuse_variables_inner,
                           key="get_variable",
                           variable_scope=variable_scope)
