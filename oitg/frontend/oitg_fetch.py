import argparse
import logging
from pathlib import Path
from oitg.results import find_by_magic
import shutil

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch ARTIQ results file from shared area")
    parser.add_argument("-o",
                        "--output-path",
                        default=None,
                        type=str,
                        metavar="filename",
                        help="Output file name (defaults to current directory)")
    # TODO: Also allow specifying day/hour/experiment/… via command line args?
    parser.add_argument(
        "rid", help="RID of experiment, or 'magic' RID string (e.g. alice_12345)")
    args = parser.parse_args()

    resolved = find_by_magic(args.rid)
    if len(resolved) == 0:
        raise FileNotFoundError("No results file found")
    if len(resolved) > 1:
        raise FileNotFoundError("More than one matching results file found")
    rid, info = next(iter(resolved.items()))
    filename = Path(info.path).parts[-1]

    if args.output_path:
        output_path = Path(args.output_path)
        if output_path.is_dir():
            output_path = output_path / filename
    else:
        output_path = Path.cwd() / filename

    logger.info("Copying RID %s from %s", rid, info.path)
    shutil.copy(info.path, output_path)


if __name__ == "__main__":
    main()
