import collections
import functools
import contextlib
import six
import tensorflow as tf

# ################################ base hooks ################################


# global hooks state
HOOKS = []


class HookedState(object):
    """
    container for the state of hooked function calls
    """

    def __init__(self, key, fn, args, kwargs):
        self.key = key
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        # make a copy of hooks
        # ---
        # rationale: this allows a hook to edit the list of hooks,
        # thus allowing hooks to be composed of other hooks
        self.hooks = list(HOOKS)

    def __call__(self):
        if self.hooks:
            hook = self.hooks.pop()
            return hook(self)
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


def add_hook(fn, location="outer"):
    # TODO does it ever make sense to have a hook twice?
    assert fn not in HOOKS
    if location == "outer":
        HOOKS.append(fn)
    elif location == "inner":
        HOOKS.insert(0, fn)
    else:
        raise ValueError("Invalid hook location: %s" % location)


def remove_hook(fn):
    while fn in HOOKS:
        HOOKS.remove(fn)


@contextlib.contextmanager
def temporary_hook(fn, location="outer"):
    try:
        add_hook(fn, location=location)
        yield
    finally:
        remove_hook(fn)

# ############################### get_variable ###############################


VARIABLE_METADATA = collections.defaultdict(dict)


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
    if initializer is None:
        initializer = tf.constant_initializer(0.)
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
    VARIABLE_METADATA.update(metadata)
    return var

# ################################# filters #################################


def wrap_filter(hook, filter_fn):
    """
    takes in a hook and only applies it when the given filter function
    returns a truthy value
    """
    def inner(hs):
        if filter_fn(hs):
            return hook(hs)
        else:
            return hs()

    return inner


def filter_dsl(hook,
               key=None,
               variable_scope=None,
               filter_fn=None):
    """
    takes in a hook and only applies it when the hooked state matches a
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

        return hook(hs)

    return inner
