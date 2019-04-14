"""Minimalist caching/memoisation implementation backed by on-disk pickle files.

Provides functions for transparently storing expensive-to-compute values to disk and
reading them back when they are next used.
"""

import functools
import logging
import os
import pathlib
import pickle
from typing import Any, Callable
import warnings

logger = logging.getLogger(__name__)

#: Name of the environment variable to load cache path from.
DIR_ENV_VAR = "OITG_CACHE_DIR"


def get_cache_dir() -> pathlib.Path:
    """Return the OITG-wide directory to use for cache files.

    The default path can be overwritten by setting the :data:`DIR_ENV_VAR` environment
    variable.
    """
    path = pathlib.Path(os.getenv(DIR_ENV_VAR, "/tmp/oitg"))
    path.mkdir(exist_ok=True)
    return path


def _get_cache_path(key: str) -> pathlib.Path:
    return get_cache_dir() / "{}_cache.dat".format(key)


def read_or_create_pickle_cache(key: str, compute_value: Callable[[], Any]) -> Any:
    """Attempt to read the cached value for the given key; if it does not exist,
    compute it and save it for future calls.

    Assumes `compute_value()` is pure (i.e. returns the same value every time); no
    no support for cache invalidation is provided beyond manually clearing a key using
    :meth:`clear_pickle_cache`.

    :param key: Cache key, to be used as part of the file name. The user must
        ensure that names are unique across client code.
    :param compute_value: Function to invoke for actually computing the value if it has
        not been cached yet.

    :return: The computed or loaded value.
    """
    path = _get_cache_path(key)
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except BaseException:
        logger.debug("Recomputing value for key '%s'", key)
        result = compute_value()
        try:
            with path.open("wb") as f:
                pickle.dump(result, f)
        except OSError:
            warnings.warn(
                "Failed to write cache path to '{}'. "
                "Consider setting {} to a user-writable directory.".format(
                    path, DIR_ENV_VAR), RuntimeWarning)
        return result


def clear_pickle_cache(key: str) -> None:
    """Remove the cache file for the given key, if it exists."""
    path = _get_cache_path(key)
    try:
        path.unlink()
    except OSError as err:
        logger.debug("Failed to remove cache path '%s', ignoring: %s", path, err)


def cache_to_pickle_file(key):
    """Wrap a parameterless function to cache its results to a pickle file and reload it
    from there on subsequent invocations.

    See :meth:`read_or_create_pickle_cache`.
    """

    def decorator(function):
        @functools.wraps(function)
        def wrapper():
            return read_or_create_pickle_cache(key, function)

        return wrapper

    return decorator
