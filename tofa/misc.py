from io import StringIO
from pprint import pprint

from addict import Dict


class AttribDict(Dict):
    def __missing__(self, key):
        raise KeyError(f"key {key} not found in AttribDict")


def getattr_nested(obj, name, default=None):
    for attr in name.split("."):
        if hasattr(obj, attr):
            obj = getattr(obj, attr)
        else:
            return default


def first(iterable):
    return next(iter(iterable))


def pop(dict, name, default=None):
    if name in dict:
        return dict.pop(name)
    return default


def pretty_str(*vals):
    str_stream = StringIO()
    for val in vals:
        pprint(val, stream=str_stream)
    return str_stream.getvalue()
