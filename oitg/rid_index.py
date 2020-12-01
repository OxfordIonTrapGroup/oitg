from bisect import bisect
from datetime import date, timedelta
import logging
import lzma
import os
from pathlib import Path
import pickle
import re

logger = logging.getLogger(__name__)

INDEX_FILE = "rid_index.pickle.xz"


def get_last_rid(path):
    rid_match = re.compile(r"(\d{9})-[A-Za-z][A-Za-z0-9_]*.h5$")
    rids = []
    for name in os.listdir(path):
        match = rid_match.match(name)
        if match:
            rids.append(int(match.group(1)))
    return max(rids, default=None)


def list_hours(root_path, date):
    hour_regex = re.compile(r"^[0-2][0-9]$")
    try:
        entries = os.listdir(Path(root_path) / date)
    except FileNotFoundError:
        logger.debug("Ignoring %s, as no such directory in %s exists", date, root_path)
        return []
    return list(filter(lambda a: re.match(hour_regex, a), entries))


def index_day(root_path, date):
    bounds = []
    for hour in list_hours(root_path, date):
        path = date + "/" + hour
        bound = get_last_rid(root_path / path)
        if bound is not None:
            bounds.append((bound, path))
    return bounds


def create_initial_index(root_path):
    date_regex = re.compile(r"^\d{4}-([0][1-9]|1[0-2])-([0-2][1-9]|[1-3]0|3[01])$")
    entries = os.listdir(root_path)
    dates = list(filter(lambda a: re.match(date_regex, a), entries))
    dates.sort()
    last_rid = 0
    bounds = []
    for day in dates:
        new = index_day(root_path, day)
        # This should just be bounds.extend(new), but Alice in lab one somehow ended up
        # with duplicate RIDs. The entire lookup system relies on RIDs being monotonous,
        # so we'll have to live without being able to access some slice of results in
        # Alice by RID only, but at least things don't break that way.
        for rid, path in new:
            if rid <= last_rid:
                continue
            bounds.append((rid, path))
            last_rid = rid
    return bounds


def read_index(root_path):
    with lzma.open(Path(root_path) / INDEX_FILE, "rb") as in_file:
        return pickle.load(in_file)


def ensure_sorted_by_rid(index):
    for i in range(len(index) - 1):
        if index[i + 1][0] <= index[i][0]:
            logger.warn("RID order inconsistency detected between %s and %s", index[i],
                        index[i + 1])
            raise ValueError("Index not sorted")


def write_index(root_path, index):
    ensure_sorted_by_rid(index)
    with lzma.open(Path(root_path) / INDEX_FILE, "wb") as out_file:
        pickle.dump(index, out_file)


def dates_between(first, last):
    step = timedelta(days=1)
    current = first
    while current <= last:
        yield current
        current += step


def update_index(root_path):
    try:
        index = read_index(root_path)
        ensure_sorted_by_rid(index)
    except FileNotFoundError:
        print(f"Index ({Path(root_path) / INDEX_FILE}) not found, " +
              "creating it from scratch. This might take a while...")
        write_index(root_path, create_initial_index())
        print("...done.")
        return

    # The last entry might be incomplete.
    _, last_path = index[-1]
    index = index[:-1]

    last_day, _ = last_path.split("/")
    for day in dates_between(date.fromisoformat(last_day), date.today()):
        date_str = day.isoformat()
        new_bounds = index_day(root_path, date_str)

        if date_str == last_day:
            # Skip already populated directories. (We could skip those in index_day()
            # as well, but the performance gain would be marginal.)
            start_idx = 0
            while start_idx < len(new_bounds):
                if new_bounds[start_idx][1] >= last_path:
                    break
                start_idx += 1
            new_bounds = new_bounds[start_idx:]

        index.extend(new_bounds)

    write_index(root_path, index)


def find_h5_in_folder(path, rid):
    matches = list(Path(path).glob(f"{rid:09}-*.h5"))
    if not matches:
        return None
    if len(matches) > 2:
        raise IOError(f"Multiple files for the same RID, {rid}, found in '{path}'")
    return matches[0]


def resolve_rid(root_path, index, rid):
    # Find upper bound from index (bisect doesn't support keys, so we rely on tuple
    # comparison semantics, where a 1-tuple comes before any 2-tuple with the same first
    # element).
    idx = bisect(index, (rid, ))
    if idx < len(index):
        path = Path(root_path) / index[idx][1]
        result = find_h5_in_folder(path, rid)
        if not result:
            raise FileNotFoundError(
                f"HDF5 file for RID {rid} not found in expected path '{path}'")

    # RID is more recent than last index entry, search.
    _, last_path = index[-1]

    # Search remaining hours on same day. (This small optimisation is worth it for the
    # common case of looking at data from the same day over slow network connections.)
    last_day, last_hour = last_path.split("/")
    for hour in list_hours(root_path, last_day):
        if int(hour) >= int(last_hour):
            result = find_h5_in_folder(Path(root_path) / last_day / hour, rid)
            if result:
                return result

    # Search any other days.
    for day in dates_between(
            date.fromisoformat(last_day) + timedelta(days=1), date.today()):
        day_str = day.isoformat()
        for hour in list_hours(root_path, day_str):
            result = find_h5_in_folder(Path(root_path) / day_str / hour, rid)
            if result:
                return result

    raise FileNotFoundError(f"RID {rid} not found in root path '{root_path}'")
