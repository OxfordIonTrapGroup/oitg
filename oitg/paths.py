"""
Well-known file system paths within the Oxford Ion Trap Quantum Computing group's
computer/network setup.
"""

import os
from datetime import date
from typing import Optional


def _get_user() -> str:
    try:
        return os.environ["OITG_USER"]
    except KeyError:
        raise Exception("No user supplied, and no OITG_USER environment key")


def shared_area_path() -> str:
    r"""Return the standard path to the shared area on the current platform.

    For Windows, the standard mount point is ``Z:\``, and for Unix-like systems
    (Linux/macOS) it is ``~/steaneShared``; this can be overwritten using the
    ``OITG_SHARED_AREA`` environment variable.
    """

    try:
        return os.environ["OITG_SHARED_AREA"]
    except KeyError:
        pass

    if os.name == "nt":  # Windows
        return "Z:\\"
    if os.name == "unix" or os.name == "posix":  # Linux / OSX / ...
        return os.path.expanduser("~/steaneShared/")
    raise Exception("Unknown OS")


def analysis_root_path(user: Optional[str] = None) -> str:
    """Return the path to the given users analysis directory on the shared area
    (``<shared_area>/Users/<user>/analysis``).

    :param user: The name of the shared-area user directory to target. If ``None``,
        defaults to the environment variable ``OITG_USER``.
    """
    if user is None:
        user = _get_user()
    return os.path.join(shared_area_path(), "Users", user, "analysis")


def todays_analysis_path(day: Optional[str] = None, user: Optional[str] = None) -> str:
    """Return the path to the analysis directory for the given day, defaulting to today.

    The analysis directory is intended to be used as working space for analysing data
    while it is taken, so that the code can easily be found again later if the data or
    conclusions reached are reexamined.

    If the directory does not exist, it is created.

    :param day: The date to use, in ISO format (``yyyy-mm-dd``), or ``None`` for today.
    :param user: The name of the shared-area user directory to target; see
        :func:`.analysis_root_path`.
    """
    if day is None:
        day = date.today().isoformat()
    if user is None:
        user = _get_user()
    path = os.path.join(analysis_root_path(user=user), day)

    if not os.access(path, os.R_OK):
        # If the dir does not exist, create it
        os.mkdir(path)

    return path


def artiq_results_path(experiment: Optional[str] = None) -> str:
    """Return the path to an experiment's ARTIQ results directory.

    The standard results path is ``<shared_area>/artiqResults/<experiment>``.

    :param experiment: The name of the experimental setup, as per the corresponding
        subdirectory of the shared area results directory. If ``None``, the environment
        variable ``OITG_EXPERIMENT`` is used.
    """

    path = os.path.join(shared_area_path(), "artiqResults")

    if experiment is None:
        try:
            experiment = os.environ["OITG_EXPERIMENT"]
        except KeyError:
            raise Exception(
                "No experiment supplied, and no OITG_EXPERIMENT environment key")

    return os.path.join(path, experiment)
