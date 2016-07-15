import functools
import contextlib
import six
import tensorflow as tf

# ############################ base functionality ############################


# global handlers state
HANDLERS = []


class HookedState(object):
    """
    container for the state of hooked function calls
    """

    def __init__(self, key, fn, args, kwargs):
        self.key = key
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        # make a copy of handlers
        # ---
        # rationale: this allows a handler to edit the list of handlers,
        # thus allowing handlers to be composed of other handlers
        self.handlers = list(HANDLERS)

    def __call__(self):
        if self.handlers:
            handler = self.handlers.pop()
            return handler(self)
        else:
            return self.fn(*self.args, **self.kwargs)

    def __repr__(self):
        return "HookedState({})".format(self.__dict__)


def hooked(fn):
    """
    decorator for hooked functions
    """

    @functools.wraps(fn)
    def inner(*args, **kwargs):
        hs = HookedState(
            key=fn.func_name,
            fn=fn,
            args=args,
            kwargs=kwargs
        )
        return hs()

    return inner


def add_handler(fn):
    # TODO does it ever make sense to have a handler twice?
    assert fn not in HANDLERS
    HANDLERS.append(fn)


def remove_handler(fn):
    while fn in HANDLERS:
        HANDLERS.remove(fn)


@contextlib.contextmanager
def temporary_handler(fn, key=None):
    try:
        add_handler(fn)
        yield
    finally:
        remove_handler(fn)

# ############################### get_variable ###############################


VARIABLE_METADATA = {}


@hooked
def get_variable(name,
                 shape=None,
                 dtype=tf.float32,
                 initializer=None,
                 regularizer=None,
                 trainable=True,
                 collections=None,
                 caching_device=None,
                 partitioner=None,
                 validate_shape=True,
                 **metadata):
    """
    wrapper around tf.get_variable that takes in additional keyword arguments
    as metadata and stores that metadata
    """
    var = tf.get_variable(name=name,
                          shape=shape,
                          dtype=dtype,
                          initializer=initializer,
                          regularizer=regularizer,
                          trainable=trainable,
                          collections=collections,
                          caching_device=caching_device,
                          partitioner=partitioner,
                          validate_shape=validate_shape)
    metadata.update(dict(
        name=name,
        shape=shape,
        dtype=dtype,
        initializer=initializer,
        regularizer=regularizer,
        trainable=trainable,
        collections=collections,
        caching_device=caching_device,
        partitioner=partitioner,
        validate_shape=validate_shape,
    ))
    VARIABLE_METADATA[var] = metadata
    return var

# ################################# filters #################################


def wrap_filter(handler, filter_fn):
    """
    takes in a handler and only applies it when the given filter function
    returns a truthy value
    """
    def inner(hs):
        if filter_fn(hs):
            return handler(hs)
        else:
            return hs()

    return inner


def filter_dsl(handler,
               key=None,
               variable_scope=None,
               filter_fn=None):
    """
    takes in a handler and only applies it when the hooked state matches a
    DSL

    key:
    if a string, matches for an exact key match
    if a list, matches if the key is in the list

    variable_scope:
    if a string, matches if it is a subset of the current variable scope's name
    if a list, matches if it is a subsequence of the current variable scope's
    name split by "/"
    """
    # TODO do we want it to work on regexes instead of exact string matches

    def inner(hs):
        if key is not None:
            if isinstance(key, six.string_types):
                if key != hs.key:
                    return hs()
            elif isinstance(key, list):
                if hs.key not in key:
                    return hs()
            else:

                raise ValueError("wrong key type: %s" % key)

        if variable_scope is not None:
            vs_name = tf.get_variable_scope().name
            if isinstance(variable_scope, six.string_types):
                if variable_scope not in vs_name:
                    return hs()
            elif isinstance(variable_scope, list):
                vs_list = vs_name.split("/")
                for subset in variable_scope:
                    if subset not in vs_list:
                        return hs()
                    vs_list = vs_list[vs_list.index(subset) + 1:]
            else:
                raise ValueError("wrong variable_scope type: %s" %
                                 variable_scope)

        if filter_fn is not None:
            if not filter_fn(hs):
                return hs()

        return handler(hs)

    return inner

# ################################# handlers #################################


def default_kwargs_handler(kwargs):
    def inner(hs):
        new_kwargs = dict(kwargs)
        new_kwargs.update(hs.kwargs)
        hs.kwargs = new_kwargs
        return hs()

    return inner


def default_kwargs_dsl(kwargs, key):
    return filter_dsl(default_kwargs_handler(kwargs), key=key)


def override_kwargs_handler(kwargs):
    def inner(hs):
        new_kwargs = dict(hs.kwargs)
        new_kwargs.update(kwargs)
        hs.kwargs = new_kwargs
        return hs()

    return inner


def override_kwargs_dsl(kwargs, key):
    return filter_dsl(override_kwargs_handler(kwargs), key=key)


def set_forget_bias_init(init_value=0.):
    def inner(hs):
        hs.kwargs["initializer"] = tf.constant_initializer(init_value)
        return hs()
    return filter_dsl(inner,
                      key="get_variable",
                      variable_scope=["lstm", "forget_bias"])

# TODO weight normalization handler
# TODO initialization handler (have in/out axes metadata)
# TODO test forget gate bias init handler


if __name__ == "__main__":
    print "Running main"

    def double_args_handler(hs):
        hs.args = [arg * 2 for arg in hs.args]
        return hs()

    def double_output_handler(hs):
        return 2 * hs()

    @hooked
    def foo(a):
        return a + 42

    assert 45 == foo(3)

    with temporary_handler(double_args_handler):
        assert 48 == foo(3)

    with temporary_handler(double_output_handler):
        assert 90 == foo(3)

    with temporary_handler(default_kwargs_dsl(kwargs={"a": 2},
                                              key="foo")):
        assert 44 == foo()
        assert 45 == foo(a=3)

    with temporary_handler(override_kwargs_dsl(kwargs={"a": 2},
                                               key="foo")):
        assert 44 == foo()
        assert 44 == foo(a=3)
