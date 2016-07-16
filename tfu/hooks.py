from . import base


def default_kwargs(kwargs):
    def inner(hs):
        new_kwargs = dict(kwargs)
        new_kwargs.update(hs.kwargs)
        hs.kwargs = new_kwargs
        return hs()

    return inner


def default_kwargs(kwargs, key):
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
