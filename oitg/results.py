from collections.abc import Iterable
from datetime import date
from glob import glob
import h5py
import json
import shutil
import os
import re
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
from . import rid_index
from .paths import artiq_results_path


def _iterify(x):
    """Ensure the passed argument is iterable, that is, if it is a single element, wrap
    it into a list.

    A string is treated as a single element, not an iterable of characters.
    """
    if x is None:
        return None
    if not isinstance(x, Iterable) or isinstance(x, str):
        return [x]
    return x


def load_hdf5_file(filename: str) -> Dict[str, Any]:
    """Load an ARTIQ results file.

    :returns: A dictionary containing the logical contents of the HDF5 file, including:

     * ``"start_time"``: the Unix timestamp when the experiment was built
     * ``"expid"``: experiment description, including submission arguments
     * ``"datasets"``: dictionary containing all set datasets
    """
    with h5py.File(filename, "r") as f:
        r = {}

        # `expid` is actually serialised as PYON on the ARTIQ side, but as it is within
        # the JSON subset (as of ARTIQ 5, anyway), we can avoid the library dependency
        # in data analysis environments.
        r["expid"] = json.loads(f["expid"][()])

        for k in ["artiq_version", "start_time"]:
            r[k] = f[k][()]

        ds = {}
        r["datasets"] = ds
        for k in f["datasets"]:
            ds[k] = f["datasets"][k][()]

        ar = {}
        r["archive"] = ar
        for k in f["archive"]:
            ar[k] = f["archive"][k][()]

        return r


def load_result(day: Union[None, str, List[str]] = None,
                hour: Union[None, int, List[int]] = None,
                rid: Union[None, int, List[int]] = None,
                class_name: Union[None, str, List[str]] = None,
                experiment: Optional[str] = None,
                root_path: Optional[str] = None,
                local_path: Optional[str] = None) -> Dict[str, Any]:
    """Find and load an HDF5 results file from an ARTIQ master results directory.

    The results file is described by a rid and a day (provided date string, defaults to
    today). See :func:`find_results` for a full description of the arguments.

    :param local_path: If specified, searches first in this directory before falling
        back to ``root_path``, and copying the files from there.
        This automatically creates a copy of the accessed files in ``local_path``.
        If ``local_path`` is on the local drive, using this will speed up future calls
        to ``load_result`` as it circumvents the shared drive, which does not even have
        to be mounted then.

    :return: A dictionary containing the contents of the file; see
        :func:`load_hdf5_file`.
    """
    def _find_results(path):
        rs = find_results(day, hour, rid, class_name, experiment, path)
        if len(rs) == 0:
            raise IOError("No results file found")
        if len(rs) > 1:
            raise IOError("More than one matching results file found")
        return rs

    if local_path is None:
        rs = _find_results(root_path)
    else:
        # Try to locate the file in ``local_path``.
        try:
            rs = _find_results(local_path)
        except IOError:
            # If the file cannot be found, fall back to ``root_path`` and copy
            # it for future use.
            rs = _find_results(root_path)
            og_file_path = next(iter(rs.values())).path
            relpath = os.path.relpath(og_file_path, start=root_path)
            local_file_path = os.path.join(local_path, relpath)
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            shutil.copy2(og_file_path, local_file_path)

    try:
        return load_hdf5_file(next(iter(rs.values())).path)
    except Exception:
        raise IOError("Failure parsing results file")


Result = NamedTuple('Result', [('path', str), ('day', str), ('hour', int),
                               ('cls', str)])


def parse_result_path(path: str) -> Tuple[int, Result]:
    head, file_name = os.path.split(path)
    head, hour_str = os.path.split(head)
    _, this_day = os.path.split(head)
    rid_part, class_part = file_name.split('-')
    return int(rid_part), Result(path=path,
                                 cls=class_part.split('.')[0],
                                 day=this_day,
                                 hour=int(hour_str))


def find_results(day: Union[None, str, List[str]] = None,
                 hour: Union[None, int, List[int]] = None,
                 rid: Union[None, int, List[int]] = None,
                 class_name: Union[None, str, List[str]] = None,
                 experiment: Optional[str] = None,
                 root_path: Optional[str] = None) -> Dict[int, Result]:
    """Find all ARTIQ result files matching the given filters.

    To implement this, the file system in the given ``root_path`` (or the standard root
    path for the given experiment as per :func:`.artiq_results_path`) is searched for
    HDF5 files matching the standard ARTIQ folder/file name structure.

    :param day: Acquisition date of results to match, or list of dates, in ISO format
        (yyyy-mm-dd). If ``None``, defaults to the current date (today); if ``"auto"``
        and ``rid`` is given, the RID index is used to determine the location.
    :param hour: Hour or list of hours when the experiment was built. If ``None``,
        includes all hours.
    :param rid: An experiment run id or list of run ids to match. If ``None``, includes
        all rids.
    :param class_name: The class name of the experiment to match, or a list of names. If
        ``None``, includes all classes.
    :param experiment: The experiment name, used for determining the results path if
        ``root_path`` is not given. See :func:`oitg.paths.artiq_results_path`.
    :param root_path: The ARTIQ results directory to search. If not given, defaults to
        the :func:`oitg.paths.artiq_results_path`. An IOError is raised if the path does
        not exist.

    :return: A dictionary of results, indexed by rid. The values are named tuples
        ``(path, day, hour, cls)``.
    """

    if root_path is None:
        root_path = artiq_results_path(experiment=experiment)

    if not os.path.exists(root_path):
        raise IOError(f"Result path '{root_path}' does not exist. Shared drive not " +
                      "mounted? Wrong experiment name?")

    # Form list of day strings to search over
    if day is None:
        day = date.today().isoformat()

    days = _iterify(day)
    rids = _iterify(rid)
    paths = []
    if list(days) == ["auto"]:
        if not rids:
            raise ValueError("Without day specified, rid(s) must be given")
        try:
            index = rid_index.read_index(root_path)
        except FileNotFoundError:
            raise IOError(
                "To resolve results with only RID specified, the result path for " +
                f"the target experiment ({experiment}) needs to be indexed. (Within " +
                "the Oxford ion trap quantum computing group, this is done via " +
                "nightly systemd jobs on 10.255.6.4.)")
        for rid in rids:
            # Discover location
            paths.append(rid_index.resolve_rid(root_path, index, rid))
    else:
        # Collect all the data files on these days
        for day in days:
            day_path = os.path.join(root_path, day)
            # To increase speed on a slow filesystem (such as an SMB mount) we could
            # only list directories with appropriate hours.
            paths.extend([
                y for x in os.walk(day_path) for y in glob(os.path.join(x[0], "*.h5"))
            ])

    results = {}
    hours = _iterify(hour)
    class_names = _iterify(class_name)
    for path in paths:
        this_rid, result = parse_result_path(path)

        # If any extra filters are given, skip if no match.
        if class_name is not None and result.cls not in class_names:
            continue
        if rid is not None and this_rid not in rids:
            continue
        if hours is not None and result.hour not in hours:
            continue

        results[this_rid] = result
    return results


def normalise_experiment_magic(exp):
    if exp in ("alice", "bob"):
        return "lab1_" + exp
    return exp


def parse_magic(rid):
    match = re.match("^([a-zA-Z0-9_]+_)?([0-9]{1,9})$", rid)
    if not match:
        return None
    exp, rid = match.groups()
    spec = {"rid": int(rid)}
    if exp:
        # Strip trailing _.
        exp = exp[:-1]
        # Don't treat "rid_1234" as an experiment name, as it shows up as the default
        # ndscan source name if no source id dataset is set.
        if exp != "rid":
            spec["experiment"] = normalise_experiment_magic(exp)
    return spec


def find_by_magic(rid_or_path):
    params = parse_magic(rid_or_path)
    if params:
        return find_results(day="auto", **params)
    # FIXME: This only really returns the expected result for files that are part of the
    # date/hour-ordered tree (as opposed to e.g. copies made in a user's analysis
    # folder, and passed as full path).
    rid, info = parse_result_path(rid_or_path)
    return {rid: info}


def load_by_magic(rid_or_path):
    params = parse_magic(rid_or_path)
    if params:
        return load_result(**params)
    return load_hdf5_file(rid_or_path)
