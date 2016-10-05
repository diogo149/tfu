import collections
import functools
import contextlib
import six
import tensorflow as tf

# ############################### state class ###############################


class GraphState(object):

    def __init__(self, graph=None):
        if graph is None:
            graph = tf.Graph()
        self.graph = graph
        self.variable_metadata = collections.defaultdict(dict)
        self.current_metadata = {}
        self.hooks = []
        self.accumulators = []
        self.misc = {}

DEFAULT_GRAPH_STATE = [GraphState(graph=tf.get_default_graph())]


def default_graph_state():
    return DEFAULT_GRAPH_STATE[0]

# ################################ base hooks ############################


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
        self.hooks = list(default_graph_state().hooks)

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
    assert fn not in default_graph_state().hooks
    if location == "outer":
        default_graph_state().hooks.append(fn)
    elif location == "inner":
        default_graph_state().hooks.insert(0, fn)
    else:
        raise ValueError("Invalid hook location: %s" % location)


def remove_hook(fn):
    while fn in default_graph_state().hooks:
        default_graph_state().hooks.remove(fn)


@contextlib.contextmanager
def temporary_hook(fn, location="outer"):
    try:
        add_hook(fn, location=location)
        yield
    finally:
        remove_hook(fn)

# ################## modified variable_scope / get_variable ##################


@contextlib.contextmanager
def variable_scope(name_or_scope=None,
                   default_name=None,
                   reuse=None,
                   **metadata):
    """
    wrapped version of variable_scope; differences:
    - less arguments available
    - allows specifying optional metadata
    """
    with tf.variable_scope(name_or_scope=name_or_scope,
                           default_name=default_name,
                           reuse=reuse):
        prev_metadata = default_graph_state().current_metadata
        new_metadata = dict(prev_metadata)  # make a copy
        new_metadata.update(metadata)
        try:
            default_graph_state().current_metadata = new_metadata
            yield
        finally:
            default_graph_state().current_metadata = prev_metadata


@hooked
def get_variable(name,
                 shape=None,
                 dtype=tf.float32,
                 initializer=tf.constant_initializer(0.),
                 **metadata):
    """
    wrapped version of tf.get_variable; differences:
    - less arguments available
    - allows specifying optional metadata
    - allows hooking of initializer
    - default to zero init
    """
    # make a copy of currrent metadata
    new_metadata = dict(default_graph_state().current_metadata)
    # overwrite with given metadata
    new_metadata.update(metadata)
    # also including these fields in the metadata because they may be
    # useful
    # e.g. for initialization
    new_metadata.update(dict(
        name=name,
        shape=shape,
        dtype=dtype,
    ))

    @hooked
    def get_initializer(metadata):
        return initializer

    new_initializer = get_initializer(new_metadata)

    var = tf.get_variable(name=name,
                          shape=shape,
                          dtype=dtype,
                          initializer=new_initializer)
    default_graph_state().variable_metadata[var] = new_metadata
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
                # TODO allow searching by metadata
                raise ValueError("wrong variable_scope type: %s" %
                                 variable_scope)

        if filter_fn is not None:
            if not filter_fn(hs):
                return hs()

        return hook(hs)

    return inner
