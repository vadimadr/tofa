import argparse
import os
import os.path as osp
import re
import sys
from contextlib import contextmanager
from importlib import import_module
from typing import Iterable
from warnings import warn

from addict import Dict

from tofa.filesystem import existing_path
from tofa.io import load


class BoolFlagAction(argparse.Action):
    """Action that stores bool flag depending on whether --option or --no-option is passed"""

    def __init__(self, option_strings, dest, default=False, required=False, help=None):
        option_strings = option_strings + [
            s.replace("--", "--no-") for s in option_strings
        ]
        super().__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=0,
            default=default,
            required=required,
            help=help,
        )

    def __call__(self, parser, namespace, values, option_string=None):
        val = not option_string.startswith("--no")
        setattr(namespace, self.dest, val)


# pylint: disable=protected-access
class ConfigDict(Dict):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls, *args, **kwargs)
        object.__setattr__(instance, "__frozen", False)
        return instance

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def freeze(self):
        object.__setattr__(self, "__frozen", True)
        for nested in self.values():
            if isinstance(nested, ConfigDict):
                nested.freeze()

    def unfreeze(self):
        object.__setattr__(self, "__frozen", False)
        for nested in self.values():
            if isinstance(nested, ConfigDict):
                nested.unfreeze()

    def __setitem__(self, key, value):
        if getattr(self, "__frozen"):
            raise RuntimeError("ConfigDict is frozen. Can not update its items")
        super().__setitem__(key, value)

    def __setattr__(self, key, value):
        if self.frozen:
            raise RuntimeError("ConfigDict is frozen. Can not update its items")
        super().__setattr__(key, value)

    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super(ConfigDict, self).__getattr__(name)
        except KeyError:
            ex = AttributeError(
                "'{}' object has no attribute '{}'".format(
                    self.__class__.__name__, name
                )
            )
        except Exception as e:
            ex = e
        else:
            return value
        raise ex

    def __contains__(self, item):
        current = self
        for part in item.split("."):
            if not super(ConfigDict, current).__contains__(part):
                return False
            current = current[part]
        return True

    @property
    def frozen(self):
        return getattr(self, "__frozen")


# pylint: disable=protected-access
def add_args(parser, cfg, prefix="", option_strings=None):
    for k, v in cfg.items():
        if not prefix and k == "required":
            continue
        k = k.replace("_", "-")
        if "--" + k in option_strings:
            continue

        if not v:
            continue

        if isinstance(v, str):
            parser.add_argument("--" + prefix + k)
        elif isinstance(v, bool):
            parser.add_argument("--" + prefix + k, action=BoolFlagAction)
        elif isinstance(v, int):
            parser.add_argument("--" + prefix + k, type=int)
        elif isinstance(v, float):
            parser.add_argument("--" + prefix + k, type=float)
        elif isinstance(v, dict):
            add_args(parser, v, prefix + k + ".", option_strings)
        elif isinstance(v, Iterable):
            parser.add_argument("--" + prefix + k, type=type(v[0]), nargs="+")
        else:
            warn("can not parse key {} of type {}".format(prefix + k, type(v)))

    return parser


def validate_config(config_dict, prefix=""):
    for k, v in config_dict.items():
        if v is required and not (prefix == "" and k == "required"):
            raise ValueError(
                "Config option {prefix}{k} is required but not provided".format(
                    prefix=prefix, k=k
                )
            )
        if isinstance(v, ConfigDict):
            validate_config(v, prefix=f"{prefix}{k}.")


def _read_env_file(env_file_name):
    if not os.path.exists(str(env_file_name)):
        return {}
    with open(str(env_file_name)) as f:
        content = f.read()

    rex_line = re.compile(r"(?:export\s+)?([A-Za-z_0-9]+)\s*=\s*(.*)")
    rex_val = re.compile(r"\"(.*)\"|'(.*)'")
    env = {}

    for line in content.splitlines():
        m = rex_line.match(line)
        if m:
            key = m.group(1)
            value = m.group(2)
            m2 = rex_val.match(value)
            if m2:
                value = m2.group(1) if m2.group(1) is not None else m2.group(2)
            if key not in env:
                env[key] = value
    return env


class Config(object):
    @classmethod
    def from_file(cls, filename):
        filename = existing_path(filename).as_posix()
        if filename.endswith(".py"):
            module_name = osp.basename(filename)[:-3]
            if "." in module_name:
                raise ValueError("Dots are not allowed in config file path.")
            config_dir = osp.dirname(filename)
            sys.path.insert(0, config_dir)
            mod = import_module(module_name)
            sys.path.pop(0)
            cfg_dict = {
                name: value
                for name, value in mod.__dict__.items()
                if not name.startswith("_") or name == "required"
            }
        else:
            cfg_dict = load(filename)
        return cls(cfg_dict, filename=filename)

    @staticmethod
    def auto_argparser(description=None):
        """Generate argparser from config file automatically (experimental)
        """
        partial_parser = ConfigArgumentParser(description=description)
        partial_parser.add_argument("config", help="config file path")
        cfg_file = partial_parser.parse_known_args()[0].config
        cfg = Config.from_file(cfg_file)
        parser = ConfigArgumentParser(description=description)
        parser.add_argument("config", help="config file path")
        add_args(parser, cfg)
        return parser, cfg

    def __init__(self, cfg_dict=None, filename=None):
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError(
                "cfg_dict must be a dict, but got {}".format(type(cfg_dict))
            )

        super(Config, self).__setattr__("_cfg_dict", ConfigDict(cfg_dict))
        super(Config, self).__setattr__("_filename", filename)
        if filename:
            with open(filename, "r") as f:
                super(Config, self).__setattr__("_text", f.read())
        else:
            super(Config, self).__setattr__("_text", "")

    def generate_argparser_options(self, argparser):
        options_strings = set()
        for action in argparser._actions:
            options_strings.update(action.option_strings)

        add_args(argparser, self, option_strings=options_strings)

    def update_from_config(self, config):
        self._cfg_dict.update(config._cfg_dict)
        return self

    def update_from_args(self, args, argparser=None):
        if argparser is not None:
            if isinstance(argparser, ConfigArgumentParser):
                default_args = {
                    arg for arg in vars(args) if arg not in argparser.seen_actions
                }
            else:
                # this will fail if we explicitly provide default argument in CLI
                known_args = argparser.parse_known_args()
                default_args = {k for k, v in vars(args).items() if known_args[k] == v}
        else:
            default_args = {k for k, v in vars(args).items() if v is None}

        for key, value in vars(args).items():
            if (
                key not in default_args
                or key not in self._cfg_dict
                and value is not None
            ):
                self[key] = value

    def update_from_env(
        self, prefix="PIPELINE", read_env_file=True, env_file_name=".env"
    ):
        if read_env_file:
            env = _read_env_file(env_file_name)
            if prefix:
                env = {f"{prefix}_{k}": v for k, v in env.items()}
            for k, v in env.items():
                os.environ.setdefault(k, v)
        for k, v in os.environ.items():
            if not k.startswith(prefix):
                continue
            k = k[len(prefix) + 1 :].lower()
            parts = k.split("__")
            self[".".join(parts)] = v

    def validate(self):
        validate_config(self._cfg_dict)

    @property
    def dict(self):
        return self._cfg_dict

    def __len__(self):
        return len(self._cfg_dict)

    def __getattr__(self, name):
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name):
        return self._cfg_dict.__getitem__(name)

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setattr__(name, value)

    def __setitem__(self, name, value):
        if "." in name:
            value_nested = {}
            value_current = value_nested
            for part in name.split(".")[:-1]:
                value_current[part] = {}
                value_current = value_current[part]
            value_current[name.split(".")[-1]] = value

            value_nested = ConfigDict(value_nested)
            self._cfg_dict.update(value_nested)
            return

        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setitem__(name, value)

    def __iter__(self):
        return iter(self._cfg_dict)

    def freeze(self):
        self._cfg_dict.freeze()

    def unfreeze(self):
        self._cfg_dict.unfreeze()


class _ArgumentGroup(argparse._ArgumentGroup):
    def _add_action(self, action):
        super()._add_action(_ActionWrapper(action))


class _ActionWrapper(argparse.Action):
    def __init__(self, action):
        self._action = action
        super().__init__(
            action.option_strings,
            action.dest,
            nargs=action.nargs,
            const=action.const,
            default=action.default,
            type=action.type,
            choices=action.choices,
            required=action.required,
            help=action.help,
            metavar=action.metavar,
        )
        self._action = action

    def __getattr__(self, item):
        return getattr(self._action, item)

    def __call__(self, parser, namespace, values, option_string=None):
        parser.seen_actions.add(self._action.dest)
        return self._action(parser, namespace, values, option_string)


class _ActionContainer(argparse._ActionsContainer):
    def add_argument_group(self, *args, **kwargs):
        group = _ArgumentGroup(self, *args, **kwargs)
        self._action_groups.append(group)
        return group


class ConfigArgumentParser(_ActionContainer, argparse.ArgumentParser):
    """ArgumentParser that saves which arguments are provided"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)
        self.seen_actions = set()

    def parse_known_args(self, args=None, namespace=None):
        self.seen_actions.clear()
        return super().parse_known_args(args, namespace)


class ConfigOptionPlaceholder:
    pass


required = ConfigOptionPlaceholder()


@contextmanager
def mutate_config(config):
    is_frozen = config.frozen
    config.unfreeze()
    yield
    if is_frozen:
        config.freeze()
