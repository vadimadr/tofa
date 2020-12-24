import hashlib
import json

from tofa.filesystem import as_path
from tofa.io import _json_serializer, load_pickle, save_pickle


def object_hash_hex(obj, hash_method=hashlib.md5):
    obj_serialize = json.dumps(obj, default=_json_serializer)
    digest = hash_method(obj_serialize.encode())
    return digest.hexdigest()


def cached(file=None):
    """Cache returned value of a wrapped function to disk. Next call with the same
    arguments will result in loading the saved values."""

    def decorator(fn):
        nonlocal file
        if file is None:
            file = "{}.cache".format(fn.__name__)

        def wrapped(*args, **kwargs):
            data = {"args": None, "kwargs": None, "ret": None}
            args_hex = object_hash_hex((args, kwargs))[-8:]
            file_hex = as_path("{!s}.{}".format(file, args_hex))
            if file_hex.exists():
                data = load_pickle(file_hex)

            if data["args"] != args or data["kwargs"] != kwargs:
                data["args"] = args
                data["kwargs"] = kwargs
                data["ret"] = fn(*args, **kwargs)

                save_pickle(data, file_hex)
            return data["ret"]

        return wrapped

    return decorator
