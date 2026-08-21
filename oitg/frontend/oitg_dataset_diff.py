#!/usr/bin/env python3
"""Query InfluxDB for ARTIQ dataset changes within a given time window.

For each key that changed, prints the last value written before the window
start, as well as every write that occurred within the window (or the
before -> after difference if --compact is given).

This relies on ARTIQ's artiq_influxdb tool forwarding dataset writes to an
InfluxDB server, which is queried over the network.

The START / END points are interpreted as local time and can be given as
ISO-8601-style dates, e.g.:
    2024-01-15T08:00:00
    "2024-01-15 08:00:00"
    2024-01-15
"""

import argparse
import sys
from datetime import datetime, timezone

from influxdb import InfluxDBClient

from oitg.paths import OitgEnvError, default_experiment

# Field names used by ARTIQ when writing datasets to InfluxDB.
# Each write sets exactly one of these; the others are absent/null.
FIELD_TYPES = ["float", "int", "bool", "string"]

MEASUREMENT_DEFAULT = "artiq"
DATASET_TAG = "dataset"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_time(s: str) -> str:
    """Parse a human-friendly local time string into an RFC-3339 UTC string."""
    for fmt in (
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d",
    ):
        try:
            dt = datetime.strptime(s, fmt).astimezone(timezone.utc)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(
        f"Cannot parse time {s!r}. Expected format: YYYY-MM-DDTHH:MM:SS or YYYY-MM-DD"
    )


def _extract_value(point: dict, field: str):
    """Return the field value from a result point, or None if absent."""
    return point.get(field)


# ---------------------------------------------------------------------------
# InfluxDB queries
# ---------------------------------------------------------------------------


def query_window(
    client: InfluxDBClient,
    db: str,
    measurement: str,
    start: str,
    end: str,
) -> dict[str, list[tuple[int, object]]]:
    """Return {key: [(time_ns, value), ...]} for all writes in [start, end)."""
    result_map: dict[str, list[tuple[int, object]]] = {}

    for field in FIELD_TYPES:
        q = (
            f'SELECT "{field}" FROM "{measurement}" '
            f"WHERE time >= '{start}' AND time < '{end}' "
            f'GROUP BY "{DATASET_TAG}"'
        )
        try:
            result = client.query(q, database=db, epoch="ns")
        except Exception as exc:
            print(
                f"Warning: window query failed for field {field!r}: {exc}",
                file=sys.stderr,
            )
            continue

        for (_, tags), points in result.items():
            key = tags[DATASET_TAG]
            for p in points:
                val = _extract_value(p, field)
                if val is not None:
                    result_map.setdefault(key, []).append((p["time"], val))

    # Sort each key's writes by timestamp (integer nanoseconds since the epoch,
    # as requested via epoch="ns"), as they are merged from one query per type.
    for key in result_map:
        result_map[key].sort(key=lambda t: t[0])

    return result_map


def get_last_before(
    client: InfluxDBClient,
    db: str,
    measurement: str,
    key: str,
    start: str,
) -> object | None:
    """Return the most recent value of *key* strictly before *start*, or None."""
    for field in FIELD_TYPES:
        q = (
            f'SELECT last("{field}") FROM "{measurement}" '
            f"WHERE (\"{DATASET_TAG}\" = '{key}') AND time < '{start}'"
        )
        try:
            result = client.query(q, database=db)
            points = list(result.get_points())
            if points:
                # InfluxDB 1.x names the aggregate column "last"
                val = points[0].get("last")
                if val is not None:
                    return val
        except Exception:
            continue

    return None


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def fmt_value(v: object) -> str:
    if isinstance(v, float):
        # Avoid unnecessary trailing zeros while preserving precision
        return f"{v:.10g}"
    return repr(v)


def fmt_time(time_ns: int) -> str:
    """Format an epoch-nanosecond timestamp as a local time string."""
    return datetime.fromtimestamp(time_ns / 1e9).strftime("%Y-%m-%d %H:%M:%S")


def print_key_changes_verbose(
    key: str,
    prev: object | None,
    updates: list[tuple[int, object]],
) -> None:
    print(f"{key}:")
    if prev is None:
        print("  (no value before window)")
    else:
        print(f"  {'(before)':>19}  {fmt_value(prev)}")
    for t, v in updates:
        print(f"  {fmt_time(t):>19}  {fmt_value(v)}")
    print()


def print_key_changes_compact(
    key: str,
    prev: object | None,
    updates: list[tuple[int, object]],
) -> None:
    before = fmt_value(prev) if prev is not None else "(none)"
    after = fmt_value(updates[-1][1])
    print(f"{key}: {before} -> {after}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("start", metavar="START", help="Window start time")
    parser.add_argument("end", metavar="END", help="Window end time")
    parser.add_argument(
        "--host", default="10.179.20.72", help="InfluxDB host (default: 10.179.20.72)"
    )
    parser.add_argument(
        "--port", type=int, default=8086, help="InfluxDB port (default: 8086)"
    )
    parser.add_argument(
        "--db",
        default=None,
        metavar="DB",
        help="InfluxDB database (default: experiment name from the "
        "OITG_EXPERIMENT environment variable)",
    )
    parser.add_argument(
        "--measurement",
        default=MEASUREMENT_DEFAULT,
        metavar="MEAS",
        help=f"InfluxDB measurement (default: {MEASUREMENT_DEFAULT})",
    )
    parser.add_argument("--user", default="", help="InfluxDB username")
    parser.add_argument("--password", default="", help="InfluxDB password")
    parser.add_argument(
        "-c",
        "--compact",
        action="store_true",
        help="One line per dataset showing only before -> after",
    )
    parser.add_argument(
        "-i",
        "--ignore-prefix",
        metavar="PREFIX",
        action="append",
        default=[],
        dest="ignore_prefixes",
        help="Ignore datasets whose key starts with PREFIX "
        "(may be specified multiple times)",
    )

    args = parser.parse_args()

    if args.db is None:
        try:
            args.db = default_experiment()
        except OitgEnvError:
            parser.error("no --db given, and OITG_EXPERIMENT not set")

    start = parse_time(args.start)
    end = parse_time(args.end)

    if start >= end:
        parser.error("START must be earlier than END")

    client = InfluxDBClient(
        host=args.host,
        port=args.port,
        username=args.user,
        password=args.password,
    )

    print(
        f"Querying {args.host}:{args.port}/{args.db} from {start} to {end} …",
        file=sys.stderr,
    )

    window_data = query_window(client, args.db, args.measurement, start, end)

    if not window_data:
        print("No writes found in the specified time window.")
        return

    print("---", file=sys.stderr)
    n_changed = 0
    for key in sorted(window_data):
        if any(key.startswith(p) for p in args.ignore_prefixes):
            continue
        updates = window_data[key]
        prev = get_last_before(client, args.db, args.measurement, key, start)

        # Skip keys where every write in the window matches the pre-window
        # value; a key with no prior value (newly appeared) always counts as
        # changed.
        if prev is not None and all(v == prev for _, v in updates):
            continue

        if args.compact:
            print_key_changes_compact(key, prev, updates)
        else:
            print_key_changes_verbose(key, prev, updates)
        n_changed += 1
    print("---", file=sys.stderr)

    if n_changed == 0:
        print("No actual value changes found in the specified time window.")
    else:
        print(f"{n_changed} key(s) changed.", file=sys.stderr)


if __name__ == "__main__":
    main()
