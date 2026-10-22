import csv
import logging
import lzma
import os
import pickle
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict

import numpy

logger = logging.getLogger(__name__)

INDEX_FILE = "rid_index.pickle.xz"
CSV_INDEX_FILE = "rid_index.csv.xz"
CSV_INDEX_FORMAT = "oitg-rid-index 1"

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


def _replace_atomically(path: Path, write: Callable[[Path], None]) -> None:
    """Call ``write()`` with a temporary file name next to ``path`` and then move the
    result into place, such that readers never observe a partially written file.
    """
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        write(tmp_path)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def write_pickle_index(root_path, index: Dict[int, str]) -> None:
    """Write the pickled index (:data:`INDEX_FILE`) used by :func:`read_index` and
    :func:`resolve_rid`.

    This is the legacy format, see :meth:`write_csv_index` for the newer CSV-based
    format.

    :param index: Map from RID to the path of the results file relative to
        ``root_path``.
    """
    # Pickle the dictionary keys/values separately and xz-compress the result for size.
    # A typical size comparison during development (lab1_alice, early 2021):
    #  - Just pickle dict: 908 kB
    #  - This option: 604 kB
    rids = numpy.array(list(index.keys()))
    order = numpy.argsort(rids)
    paths = list(index.values())

    def write(path):
        with lzma.open(path, "wb") as out_file:
            pickle.dump((rids[order], [paths[i] for i in order]), out_file)

    _replace_atomically(Path(root_path) / INDEX_FILE, write)


def write_csv_index(root_path, index: Dict[int, str]) -> None:
    """Write the index as an xz-compressed CSV table (:data:`CSV_INDEX_FILE`).

    The file looks like this::

        # oitg-rid-index 1
        # experiment=lab1_alice generated=2026-10-22T06:00:12Z
        rid,day,hour,class
        3,2017-10-05,23,MonitorFluo
        ...

    Rows are sorted by RID. ``day`` (ISO date) and ``hour`` (two digits) give the
    results directory the file is stored in, ``class`` is the experiment class name, so
    the path relative to the results root is ``f"{day}/{hour}/{rid:09}-{class}.h5"``.

    Note that ``artiq_master`` names the results directories according to the time zone
    configured on its machine (typically local time), and that for any single
    experiment, the times of submission, ``run()`` start, and completion may differ
    significantly (e.g. straddling an hour/day boundary).

    No field needs CSV escaping (class names are identifiers), so the file can also be
    parsed by just splitting lines on commas, ignoring those starting with ``#``.

    :param index: Map from RID to the path of the results file relative to
        ``root_path``. Entries not matching the standard ARTIQ results layout are
        skipped with a warning.
    """
    rows = []
    for rid in sorted(index):
        rel_path = index[rid]
        match = _REL_PATH_RE.match(rel_path)
        if not match or int(match["rid"]) != rid:
            logger.warning(
                "Not including RID %s in CSV index, unexpected path '%s'", rid, rel_path
            )
            continue
        rows.append((int(rid), match["day"], match["hour"], match["class"]))

    experiment = Path(os.path.abspath(root_path)).name
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def write(path):
        # The default xz preset is as good as it gets for this data (lab1_alice with
        # 818k entries, 2026-10: 1.60 MB at preset 6 vs. 1.61 MB at 9 | EXTREME), at a
        # fraction of the time and memory, so no need to be fancy.
        with lzma.open(path, "wt", encoding="utf-8", newline="") as out_file:
            out_file.write(f"# {CSV_INDEX_FORMAT}\n")
            out_file.write(f"# experiment={experiment} generated={generated}\n")
            writer = csv.writer(out_file, lineterminator="\n")
            writer.writerow(["rid", "day", "hour", "class"])
            writer.writerows(rows)

    _replace_atomically(Path(root_path) / CSV_INDEX_FILE, write)


def write_index_from_dict(root_path, index: Dict[int, str]) -> None:
    """Write both the pickled index and the CSV index from the given RID -> relative
    path map; see :func:`write_pickle_index` and :func:`write_csv_index`.
    """
    write_pickle_index(root_path, index)
    write_csv_index(root_path, index)


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
