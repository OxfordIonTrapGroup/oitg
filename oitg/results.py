import json
import os
from glob import glob
import h5py
from datetime import date
from collections import Iterable
from typing import Any, Dict, List, NamedTuple, Optional, Union

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

        return r


def load_result(day: Union[None, str, List[str]] = None,
                hour: Union[None, int, List[int]] = None,
                rid: Union[None, int, List[int]] = None,
                class_name: Union[None, str, List[str]] = None,
                experiment: Optional[str] = None,
                root_path: Optional[str] = None) -> Dict[str, Any]:
    """Find and load an HDF5 results file from an ARTIQ master results directory.

    The results file is described by a rid and a day (provided date string, defaults to
    today). See :func:`find_results` for a full description of the arguments.

    :return: A dictionary containing the contents of the file; see
        :func:`load_hdf5_file`.
    """
    rs = find_results(day=day,
                      rid=rid,
                      hour=hour,
                      class_name=class_name,
                      experiment=experiment,
                      root_path=root_path)
    if len(rs) == 0:
        raise IOError("No results file found")
    if len(rs) > 1:
        raise IOError("More than one matching results file found")

    try:
        return load_hdf5_file(rs[rid].path)
    except Exception:
        raise IOError("Failure parsing results file")


Result = NamedTuple('Result', [('path', str), ('day', str), ('hour', int),
                               ('cls', str)])


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
        (yyyy-mm-dd). If ``None``, defaults to the current date (today).
    :param hour: Hour or list of hours when the experiment was built. If ``None``,
        includes all hours.
    :param rid: An experiment run id or list of run ids to match. If ``None``, includes
        all rids.
    :param class_name: The class name of the experiment to match, or a list of names. If
        ``None``, includes all classes.
    :param experiment: The experiment name, used for determining the results path if
        ``root_path`` is not given. See :func:`oitg.paths.artiq_results_path`.
    :param root_path: The ARTIQ results directory to search. If not given, defaults to
        the :func:`oitg.paths.artiq_results_path`.

    :return: A dictionary of results, indexed by rid. The values are named tuples
        ``(path, day, hour, cls)``.
    """

    if root_path is None:
        root_path = artiq_results_path(experiment=experiment)

    # Form list of day strings to search over
    if day is None:
        day = date.today().isoformat()
    days = _iterify(day)

    # Collect all the data files on these days
    paths = []
    for day in days:
        day_path = os.path.join(root_path, day)
        # To increase speed on a slow filesystem (such as an SMB mount) we could only
        # list directories with appropriate hours.
        paths.extend(
            [y for x in os.walk(day_path) for y in glob(os.path.join(x[0], "*.h5"))])

    results = {}
    rids = _iterify(rid)
    hours = _iterify(hour)
    class_names = _iterify(class_name)
    for path in paths:
        head, file_name = os.path.split(path)
        head, hour_str = os.path.split(head)
        _, this_day = os.path.split(head)
        rid_part, class_part = file_name.split('-')
        this_rid = int(rid_part)
        this_class = class_part.split('.')[0]
        this_hour = int(hour_str)

        # Filter on class name
        if class_name is not None and this_class not in class_names:
            continue

        # Filter on rid
        if rid is not None and this_rid not in rids:
            continue

        # Filter on hours
        if hours is not None and this_hour not in hours:
            continue

        results[this_rid] = Result(path=path,
                                   cls=this_class,
                                   day=this_day,
                                   hour=this_hour)
    return results
