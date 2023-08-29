from collections.abc import Iterable
from datetime import date
from glob import glob
import h5py
import json
import shutil
import os
import re
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
from sipyco import pyon
import numpy as np
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
    match = re.match("^([a-zA-Z]+_)?([0-9]{1,9})$", rid)
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


def load_ndscan(
    day: Union[None, str, List[str]] = None,
    hour: Union[None, int, List[int]] = None,
    rid: Union[None, int, List[int]] = None,
    class_name: Union[None, str, List[str]] = None,
    experiment: Optional[str] = None,
    root_path: Optional[str] = None,
    return_results: bool = False,
) -> Tuple[
    Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], Optional[Dict[str, Any]]
]:
    """
    Unpacks the results from an N-dimensional ndscan experiment to make scan data
    and axes more accessible. Returns sorted results and axes.

    :return: A tuple containing the following:
        - scan_results: a dictionary containing dictionaries of scan data for each
            results channel, mapped to by the name of the results channel. Each scan
            dictionary contains entries:

                - data: numpy N-dimensional array (or N+M dimensional for results
                    channels with M-dimensional lists) containing data sorted according
                    to the sorted scan axes.
                - data_raw: numpy array containing the raw scan results.
                - spec: results spec.

        - scan_axes: a list of dictionaries containing a dictionary of axes data for
            each scanned param. The axes are ordered with the innermost axis first.
            Each axis dictionary contains entries:

                - data: numpy array containing the sorted axis data.
                - data_raw: numpy array containing the raw scanned axis data.
                - description: The param description provided in the experiment
                    (if any).
                - path: Path to the scanned param.
                - spec: Param spec.
                - ax_idx: The index of the axis in the N-dimensional scan, with 0 being
                    the innermost axis being scanned.

        - args: A dictionary containing the arguments submitted to the experiment.

        - raw_results: the raw output of load_result().
    """
    # TODO: add analyses and annotations.
    raw_results = load_result(
        day=day,
        hour=hour,
        rid=rid,
        class_name=class_name,
        experiment=experiment,
        root_path=root_path,
    )
    d = raw_results["datasets"]
    a = raw_results["expid"]["arguments"]
    base_key = f"ndscan.rid_{rid}."

    axs = json.loads(d[base_key + "axes"])
    if axs == []:
        scan_axes = []
        points_key = "point."
    else:
        scan_axes = [
            {
                "data_raw": d[base_key + f"points.axis_{i}"],
                "description": ax["param"].get("description", None),
                "path": ax["path"],
                "spec": ax["param"]["spec"],
                "ax_idx": i,
            }
            for i, ax in enumerate(axs)
        ]
        points_key = "points.channel_"

    ndscan_results_channel_spec = json.loads(d[base_key + "channels"])
    scan_results = {}
    for chan, spec in ndscan_results_channel_spec.items():
        try:
            scan_results[chan] = {
                "data_raw": d[base_key + points_key + chan],
                "spec": spec,
            }
        except KeyError:
            print(f"Results channel {chan} not found.")

    scan_results, scan_axes = sort_data(scan_results, scan_axes)

    args = {}
    for key, arg in a.items():
        if key == "ndscan_params":
            ndscan_params = pyon.decode(arg)
            for fqn, overrides in ndscan_params["overrides"].items():
                for override in overrides:
                    schem = ndscan_params["schemata"][fqn]
                    value = override["value"]
                    description = schem["description"]
                    path = override["path"]
                    try:
                        args[description] = {
                            "value": value,
                            "fqn": fqn,
                            "path": path,
                            "unit": schem.get("unit", ""),
                            "scale": schem["spec"]["scale"],
                            "ndscan": True,
                        }
                    except KeyError:
                        print(fqn)

            args["scan"] = ndscan_params["scan"]

        else:
            args[key] = {"value": arg, "ndscan": False}
    args["completed"] = d[base_key + "completed"]

    if return_results:
        return scan_results, scan_axes, args, raw_results
    else:
        return scan_results, scan_axes, args


def sort_data(
    scan_results: Dict[str, Any], scan_axes: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Sort the results of an N-dimensional scan. Takes in dictionaries with
    entries 'data_raw' and adds an entry 'data' with a sorted scan axis, or
    a sorted N-dimensional array of results values that match the axes. If a
    result value is missing (due to eg an unfinished refined scan), entries
    are left as np.nan.

    Returns the (mutated) input scan_results and scan_axes dictionaries. If
    the scan data can't be sorted, sets 'data' entry to None.
    """
    # Sort the axis data into 1-D arrays.
    for axis in scan_axes:
        axis["data"] = np.unique(axis["data_raw"])
    axes_lengths = [np.size(ax["data"]) for ax in scan_axes]
    num_points = len(scan_axes[0]["data_raw"])

    # Find the coordinates of each point in the raw result data according to the
    # sorted axes.
    coords = []
    for point_num in range(num_points):
        _coords = []
        for ax in scan_axes:
            idcs = np.nonzero(ax["data"] == ax["data_raw"][point_num])
            _coords.append(idcs[0][0])
        coords.append(tuple(np.flip(_coords)))

    # Create N-dimensional arrays that store the result data, according to
    # the obtained coordinates. If a coordinate is missing (due to eg an
    # unfinished refined scan) leaves entry as nan.
    for key, dat_dict in scan_results.items():
        try:
            dat = dat_dict["data_raw"]
        except IndexError:
            print(f"Key 'data_raw' missing in dictionary for {key}")
        try:
            # Take into account results channels that are arrays.
            data_shape = np.shape(dat)
            _axes = tuple(
                np.concatenate((np.flip(axes_lengths), data_shape[1:])).astype(int)
            )
            _dat_sorted = np.zeros(_axes) + np.nan
            for point_number, d in enumerate(dat):
                _dat_sorted[coords[point_number]] = d
            scan_results[key]["data"] = _dat_sorted
        except Exception:
            print(f"Couldn't sort results channel {key}. Setting 'data' entry to None")
            scan_results[key]["data"] = None

    return scan_results, scan_axes
