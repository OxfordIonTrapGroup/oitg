from math import floor, log10, isnan
from numpy import isinf


def uncertainty_to_string(x, err, precision=1):
    """Returns a string representing nominal value x with uncertainty err.
    Precision is the number of significant digits in uncertainty

    Returns the shortest string representation of `x +- err` either as
        x.xx(ee)e+xx
    or as
        xxx.xx(ee)

    Based on http://stackoverflow.com/questions/6671053/python-pretty-print-errorbars"""

    if isnan(x) or isnan(err):
        return "NaN"

    if isinf(x) or isinf(err):
        return "inf"

    # Chuck away sign of err
    err = abs(err)

    # An error of 0 is not meaningful
    assert (err > 0)

    # base 10 exponents
    err_exp = int(floor(log10(err)))

    # If x is 0 set the x_exp to be the same as err_exp, meaning that the
    # result is formatted as 0(err)
    try:
        x_exp = int(floor(log10(abs(x))))
    except ValueError:
        x_exp = err_exp

    # Or if |x| < err, do the same
    if abs(x) < err:
        x_exp = err_exp

    # uncertainty
    un_exp = err_exp - precision + 1
    un_int = round(err * 10**(-un_exp))

    # nominal value
    no_exp = un_exp
    no_int = round(x * 10**(-no_exp))

    # format - nom(unc)exp
    fieldw = x_exp - no_exp
    fmt = '%%.%df' % fieldw

    result1 = (fmt + "(%.0f)e%d") % (no_int * 10**(-fieldw), un_int, x_exp)

    # format - nom(unc)
    fieldw = max(0, -no_exp)
    fmt = "%%.%df" % fieldw
    result2 = (fmt + "(%.0f)") % (no_int * 10**no_exp, un_int * 10**max(0, un_exp))

    # return shortest representation
    if len(result2) <= len(result1):
        return result2
    else:
        return result1


if __name__ == "__main__":
    xs = [
        0, 12.34567, -0.123456, 0.001234560000, -0.0000123456,
        float('nan'), 0,
        float('inf'), 10
    ]
    xes = [
        1e-4, 0.00123, 0.000123, 0.000000012345, 0.0000001234, 1,
        float('nan'),
        float('inf'), 100
    ]
    precs = [1, 2, 3, 4, 1, 1, 1, 1, 1]

    for (x, xe_, prec) in zip(xs, xes, precs):
        print("{} +- {} --> {}".format(x, xe_, uncertainty_to_string(x, xe_, prec)))
