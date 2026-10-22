import logging
import lzma
import os
import pickle
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Dict

import numpy

logger = logging.getLogger(__name__)

INDEX_FILE = "rid_index.pickle.xz"

# Relative path of a results file below the results root, as produced by the ARTIQ
# master: <ISO date>/<hour>/<RID>-<experiment class>.h5. The RID is zero-padded to at
# least 9 digits (as per ``"{:09}".format(rid)``), and the class name is a Python
# identifier, which may contain non-ASCII letters (hence \w).
_REL_PATH_RE = re.compile(
    r"^(?P<day>\d{4}-\d{2}-\d{2})/(?P<hour>\d{2})/(?P<rid>\d{9,})-(?P<class>\w+)\.h5$"
)


def iterate_hour_dirs(root_path, day, warn_on_unexpected=False):
    hour_regex = re.compile(r"^[0-2][0-9]$")
    day_dir = Path(root_path) / day
    if not day_dir.is_dir():
        # No results from this day; skip.
        return
    for entry in os.listdir(day_dir):
        path = f"{day}/{entry}"
        if re.match(hour_regex, entry):
            yield path
        elif warn_on_unexpected:
            logger.warning("Ignoring unexpected directory '%s'", path)


def add_rids_from_path(root_path, path, index: Dict[int, str]):
    for name in os.listdir(Path(root_path) / path):
        rel_path = f"{path}/{name}"
        match = _REL_PATH_RE.match(rel_path)
        if not match:
            logger.warning("Ignoring file '%s'", rel_path)
            continue
        rid = int(match["rid"])
        if previous_path := index.get(rid):
            if previous_path != rel_path:
                logger.warning(
                    "RID %s encountered more than once; '%s' will be ignored",
                    rid,
                    previous_path,
                )
        index[rid] = rel_path


def index_day(root_path, day, index):
    for rel_path in iterate_hour_dirs(root_path, day):
        add_rids_from_path(root_path, rel_path, index)


def create_initial_index(root_path):
    date_regex = re.compile(r"^\d{4}-([0][1-9]|1[0-2])-([0-2][1-9]|[1-3]0|3[01])$")
    entries = os.listdir(root_path)
    dates = list(filter(lambda a: re.match(date_regex, a), entries))
    dates.sort()
    index = {}
    for day in dates:
        index_day(root_path, day, index)
    return index


def read_index(root_path):
    with lzma.open(Path(root_path) / INDEX_FILE, "rb") as in_file:
        return pickle.load(in_file)


def write_index_from_dict(root_path, index: Dict[int, str]):
    # Pickle the dictionary keys/values separately and xz-compress the result for size.
    # A typical size comparison during development (lab1_alice, early 2021):
    #  - Just pickle dict: 908 kB
    #  - This option: 604 kB
    with lzma.open(Path(root_path) / INDEX_FILE, "wb") as out_file:
        rids = numpy.array(list(index.keys()))
        order = numpy.argsort(rids)
        paths = list(index.values())
        pickle.dump((rids[order], [paths[i] for i in order]), out_file)


def dates_between(first, last):
    step = timedelta(days=1)
    current = first
    while current <= last:
        yield current
        current += step


def update_index(root_path):
    try:
        rids, paths = read_index(root_path)
    except FileNotFoundError:
        print(
            f"Index ({Path(root_path) / INDEX_FILE}) not found, "
            + "creating it from scratch. This might take a while..."
        )
        write_index_from_dict(root_path, create_initial_index(root_path))
        print("...done.")
        return

    # Add days between the last entry (which might still be incomplete) and the current
    # date.
    last_day, _, _ = paths[-1].split("/")
    index = dict(zip(rids, paths))
    for day in dates_between(date.fromisoformat(last_day), date.today()):
        index_day(root_path, day.isoformat(), index)

    write_index_from_dict(root_path, index)


def find_h5_in_folder(path, rid):
    matches = list(Path(path).glob(f"{rid:09}-*.h5"))
    if not matches:
        return None
    if len(matches) > 2:
        raise IOError(f"Multiple files for the same RID, {rid}, found in '{path}'")
    return matches[0]


def resolve_rid(root_path, index_pairs, rid):
    rids, paths = index_pairs
    idx = numpy.searchsorted(rids, rid)
    if idx < len(rids) and rids[idx] == rid:
        result = Path(root_path) / paths[idx]
        if not result.exists():
            raise FileNotFoundError(
                f"HDF5 file for RID {rid} not found in expected path '{result}'"
            )
        return result

    # RID isn't in index, but could be more recent than the last update. Since we update
    # daily, check everything from the last update (by proxy, the one of the last rid
    # recorded, as we don't mind being slightly pessimistic) to today.
    last_day, _, _ = paths[-1].split("/")
    for day in dates_between(date.fromisoformat(last_day), date.today()):
        day_str = day.isoformat()
        for path in iterate_hour_dirs(root_path, day_str):
            result = find_h5_in_folder(Path(root_path) / path, rid)
            if result:
                return result

    raise FileNotFoundError(f"RID {rid} not found in root path '{root_path}'")
