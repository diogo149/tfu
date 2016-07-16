import numpy as np
import tensorflow as tf
import nose.tools as nt
import tfu


def test_temporary_hook():
    def double_args_hook(hs):
        hs.args = [arg * 2 for arg in hs.args]
        return hs()

    def double_output_hook(hs):
        return 2 * hs()

    @tfu.hooked
    def foo(a):
        return a + 42

    assert 45 == foo(3)

    with tfu.temporary_hook(double_args_hook):
        assert 48 == foo(3)

    with tfu.temporary_hook(double_output_hook):
        assert 90 == foo(3)

    with tfu.temporary_hook(tfu.hooks.default_kwargs_dsl(kwargs={"a": 2},
                                                         key="foo")):
        assert 44 == foo()
        assert 45 == foo(a=3)

    with tfu.temporary_hook(tfu.hooks.override_kwargs_dsl(kwargs={"a": 2},
                                                          key="foo")):
        assert 44 == foo()
        assert 44 == foo(a=3)
