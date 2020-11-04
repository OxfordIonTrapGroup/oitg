import argparse
import pprint as pp
import sipyco.pyon as pyon
import oitg.results


def ndscan_params(day, rid, exp="comet"):
    """Return the parameters used for a specific experimental RID which used ndscan

    :param day: The date to use, in ISO format (``yyyy-mm-dd``).
    :param rid: The run ID (RID) of the experiemtn of interest.
    :param exp: The name of the experimental setup, as per the corresponding
        subdirectory of the shared area results directory.
    """
    data = oitg.results.load_result(day, rid=rid, experiment=exp, root_path=None)
    params = pyon.decode(data["expid"]["arguments"]["ndscan_params"])
    return params["scan"], params["overrides"]


def main():
    parser = argparse.ArgumentParser(description='Get RID params')
    parser.add_argument('date', type=str, help='RID date in "yyyy-mm-dd" format')
    parser.add_argument('rid', type=int, help='experiment RID')
    parser.add_argument('-e', '--exp', type=str, help='experiment e.g. "comet"')
    args = parser.parse_args()

    if args.exp is not None:
        scan, overides = ndscan_params(args.date, args.rid, args.exp)
    else:
        scan, overides = ndscan_params(args.date, args.rid)

    pp.pprint(scan)
    pp.pprint(overides)


if __name__ == '__main__':
    main()
