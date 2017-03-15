"""
A way of storing a summary of the state as an experiment progresses

modified from du.sandbox.summary

eg.
>>> s = SummaryPrinter()
>>> s.add("foo", how="max")
>>> s.update({"foo": 1})
>>> s.update({"foo": 3})
>>> s.update({"foo": 2})
>>> print s.to_org_list()
- foo: 3
"""
import socket
import du


class SummaryPrinter(object):

    RECIPES = {}

    HOW_FUNCS = dict(
        last=lambda x_new, x_old: x_new,
        min=lambda x_new, x_old: (x_new
                                  if x_old is None
                                  else min(x_new, x_old)),
        max=lambda x_new, x_old: (x_new
                                  if x_old is None
                                  else max(x_new, x_old)),
    )

    def __init__(self):
        self.fields = {}
        self.field_order = []

    def add(self, field_name, **kwargs):
        """
        possible kwargs:
        - how: how the final value is found
          eg. "min" / "max" / "last" (default) / some custom function
        - format: how the value is displayed when printed
        - value: an initial value
        - in_keys: which keys for the `how` function to take
          default: (field_name,)
        - prev_keys: which keys of the summary the `how` function should take
          default: (field_name,)
        - on_change: function which gets called with the current summary object
          when updated
        """
        params = du.AttrDict(
            how="last",
            format="%s",
            value=None,
            in_keys=(field_name,),
            prev_keys=(field_name,),
            on_change=None,
        )
        params.update(kwargs)
        if params.how in self.HOW_FUNCS:
            params.how = self.HOW_FUNCS[params.how]
        self.fields[field_name] = params
        self.field_order.append(field_name)

    def update(self, result_dict):
        for f in self.field_order:
            params = self.fields[f]
            # make sure result has all of the needed keys
            if not all(k in result_dict for k in params.in_keys):
                continue
            in_args = [result_dict[k] for k in params.in_keys]
            prev_args = [self.fields[k].value for k in params.prev_keys]
            result = params.how(*(in_args + prev_args))
            old_result = params.value
            if old_result != result:
                # store result
                params.value = result
                # optionally have callback
                if params.on_change is not None:
                    params.on_change(old_result, result)

    def value(self, field_name):
        return self.fields[field_name].value

    def to_org_list(self, spaces=0):
        out_strs = []
        for f in self.field_order:
            params = self.fields[f]
            if params.value is not None:
                out_strs += ["".join([" " * spaces,
                                      "- ",
                                      f,
                                      ": ",
                                      params.format % params.value])]
        return "\n".join(out_strs)

    def to_value_dict(self):
        return {f: self.fields[f].value for f in self.field_order}

    def add_recipe(self, recipe_name, *args, **kwargs):
        SummaryPrinter.RECIPES[recipe_name](self, *args, **kwargs)

    @staticmethod
    def save_recipe(name=None):
        """
        context manager to create new recipes - reusable summary snippets

        eg.
        >>> @SummaryPrinter.save_recipe()
        >>> def add_constant_42(summary):
        >>>    summary.add("foo", value=42)
        >>>
        >>> summary.add_recipe("add_constant_42")
        """

        def decorator(func):
            if name is None:
                name_ = func.func_name
            else:
                name_ = name
            assert name_ not in SummaryPrinter.RECIPES
            SummaryPrinter.RECIPES[name_] = func
            return func

        return decorator

# -------
# recipes
# -------


@SummaryPrinter.save_recipe()
def s_per_iter(summary):
    summary.add(field_name="s_per_iter",
                in_keys=(),
                how=(lambda t, i: float(t) / i),
                prev_keys=("time", "iter"),
                value=0.0,
                # round to 4 significant figures
                format="%.4g")


@SummaryPrinter.save_recipe()
def ms_per_obs(summary, obs_per_iter):
    summary.add(field_name="ms_per_obs",
                in_keys=(),
                how=(lambda spi: spi * 1000.0 / obs_per_iter),
                prev_keys=("s_per_iter",),
                value=0.0,
                # round to 4 significant figures
                format="%.4g")


@SummaryPrinter.save_recipe()
def in_key_on_field_change(summary,
                           field_name,
                           in_key,
                           new_field_name=None,
                           **kwargs):
    prev_value = []

    def update(key_value, next_value, last_key_value):
        if len(prev_value) == 0:
            prev_value.append(next_value)
            return key_value
        elif prev_value[0] != next_value:
            prev_value[0] = next_value
            return key_value
        else:
            return last_key_value

    if new_field_name is None:
        new_field_name = "%s %s" % (field_name, in_key)
    summary.add(field_name=new_field_name,
                how=update,
                in_keys=[in_key],
                prev_keys=[field_name, new_field_name],
                **kwargs)


@SummaryPrinter.save_recipe()
def prev_key_on_field_change(summary,
                             field_name,
                             prev_key,
                             new_field_name=None,
                             **kwargs):
    prev_value = []

    def update(key_value, next_value, last_key_value):
        if len(prev_value) == 0:
            prev_value.append(next_value)
            return key_value
        elif prev_value[0] != next_value:
            prev_value[0] = next_value
            return key_value
        else:
            return last_key_value

    if new_field_name is None:
        new_field_name = "%s %s" % (field_name, prev_key)
    summary.add(field_name=new_field_name,
                how=update,
                in_keys=[],
                prev_keys=[prev_key, field_name, new_field_name],
                **kwargs)


@SummaryPrinter.save_recipe()
def field_iter(summary, field_name):
    summary.add_recipe("prev_key_on_field_change",
                       field_name=field_name,
                       prev_key="iter",
                       new_field_name="%s iter" % field_name,
                       value=-1,
                       format="%d")


@SummaryPrinter.save_recipe("x max+iter")
def x_max_and_iter(summary, in_key, **kwargs):
    field_name = "%s max" % in_key
    summary.add(field_name=field_name,
                how="max",
                in_keys=[in_key],
                **kwargs)
    summary.add_recipe("field_iter", field_name)


@SummaryPrinter.save_recipe("x min+iter")
def x_min_and_iter(summary, in_key, **kwargs):
    field_name = "%s min" % in_key
    summary.add(field_name=field_name,
                how="min",
                in_keys=[in_key],
                **kwargs)
    summary.add_recipe("field_iter", field_name)


@SummaryPrinter.save_recipe("x last+iter")
def x_last_and_iter(summary, in_key, **kwargs):
    field_name = "%s last" % in_key
    summary.add(field_name=field_name,
                how="last",
                in_keys=[in_key],
                **kwargs)
    summary.add_recipe("field_iter", field_name)


@SummaryPrinter.save_recipe()
def before_x_minutes(summary, field_name, minutes, **kwargs):
    def inner(time, new_value, old_value):
        if time < minutes * 60:
            return new_value
        else:
            return old_value

    this_field_name = "%s before %s minutes" % (field_name, minutes)
    summary.add(field_name=this_field_name,
                how=inner,
                in_keys=["time"],
                prev_keys=[field_name, this_field_name],
                **kwargs)


@SummaryPrinter.save_recipe()
def trial_info(summary, trial=None):
    summary.add("trial", value="%s:%s:%d" % (socket.gethostname(),
                                             trial.trial_name,
                                             trial.iteration_num))


@SummaryPrinter.save_recipe()
def add_finals(summary, field_names, **kwargs):
    for field_name in field_names:
        summary.add("final_%s" % field_name,
                    in_keys=[field_name],
                    how="last",
                    **kwargs)


@SummaryPrinter.save_recipe()
def time(summary):
    import time
    start_time = time.time()
    summary.add(field_name="time",
                in_keys=(),
                how=lambda: time.time() - start_time,
                prev_keys=(),
                value=0.0,
                format="%f")


@SummaryPrinter.save_recipe()
def iter(summary):
    from . import counter
    initial_count = counter.get_count_value()
    summary.add(field_name="iter",
                in_keys=(),
                how=lambda: counter.get_count_value() - initial_count,
                prev_keys=(),
                value=0.0,
                format="%d")
