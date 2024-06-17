from contextlib import contextmanager
from functools import partial


class ConfigManager(dict):
    pass


class _Default:
    pass


default_var = _Default()


def create_config(default=None):
    if default is None:
        default = {}
    _config = ConfigManager(default)

    def set_(name, var):
        """
        set a configuration.
        """
        if name in _config:
            _config[name] = var
        else:
            raise Exception("No configuration named {} found.".format(name))

    def get_(name, default=default_var):
        """
        get a configuration.
        """
        if name in _config:
            return _config[name]
        if isinstance(default, _Default):
            raise Exception("No configuration named {} found.".format(name))
        return default

    def regist_(name, var=None):
        """
        regist a configuration.
        """
        if name in _config:
            raise Exception(
                "Configuration named {} already exists.".format(name)
            )
        if var is None:

            def regist(f):
                _config[name] = f
                return f

            return regist
        _config[name] = var
        return var

    return set_, get_, regist_


set_config, get_config, regist_config = create_config(
    {}
)


@contextmanager
def temp_config(name, var):
    tmp = get_config(name)
    set_config(name, var)
    yield var
    set_config(name, tmp)