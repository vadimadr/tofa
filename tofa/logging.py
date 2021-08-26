from logging import (
    CRITICAL,
    DEBUG,
    ERROR,
    FATAL,
    FileHandler,
    Formatter,
    INFO,
    StreamHandler,
    WARNING,
    getLogger,
    root,
)
import sys

import numpy as np

_all_loggers = {}


def make_logger(name=root.name, level="DEBUG", colored=True, **kwargs):
    """Create logger and perform basic configuration."""
    if name in _all_loggers:
        return _all_loggers[name]

    logger = _all_loggers[name] = getLogger(name)

    if colored:
        import coloredlogs

        coloredlogs.install(logger=logger, level=level)

        # do not provide basic stream config because coloredlogs already does this.
        kwargs["stream"] = None

    basic_config(logger, level=level, **kwargs)

    return logger


def basic_config(
    logger=root.name,
    level="DEBUG",
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    file_name=None,
    file_mode="a",
    stream=sys.stderr,
    datefmt="%Y-%m-%d %H:%M:%S",
    style="%",
    propagate=True,
):
    """Basic logging configuration with explicit arguments"""
    if isinstance(logger, str):
        logger = getLogger(logger)

    logger.propagate = propagate
    logger.setLevel(level)

    if sys.version_info >= (3, 8):
        formatter = Formatter(format, datefmt, style=style, validate=True)
    else:
        formatter = Formatter(format, datefmt, style=style)

    if file_name:
        file_handler = FileHandler(file_name, file_mode)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if stream is not None:
        stream_handler = StreamHandler(stream)
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(level)

    return logger


def level_to_number(loglevel):
    levels = {
        "DEBUG": DEBUG,
        "INFO": INFO,
        "WARNING": WARNING,
        "ERROR": ERROR,
        "FATAL": FATAL,
        "CRITICAL": CRITICAL,
    }
    return levels.get(loglevel)
