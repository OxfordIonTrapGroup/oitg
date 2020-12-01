import argparse
from pathlib import Path
from ..rid_index import update_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", help="Results root directory to index")
    args = parser.parse_args()

    path = Path(args.results_dir)
    if not (path.exists() and path.is_dir):
        raise ValueError(f"Invalid root dir: {path}")
    update_index(path)


if __name__ == "__main__":
    main()
