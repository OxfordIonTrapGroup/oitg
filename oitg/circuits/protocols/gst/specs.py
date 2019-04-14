r"""Predefined :class:`GSTSpec`\ s corresponding to commonly used gate sets."""

import numpy as np
from itertools import chain
from ...gate import Gate
from .generate import GSTSpec


def make_1q_xz_pi_2_spec():
    """Return a single-qubit gate set using π/2 x- and z-rotations, corresponding to
    pyGSTi's `std1Q_XZ` model.
    """
    x = Gate("rx", (np.pi / 2, ), (0, ))
    z = Gate("rz", (np.pi / 2, ), (0, ))

    prep = [(), (x, ), (x, z), (x, x), (x, x, x), (x, z, x, x)]
    meas = [seq[::-1] for seq in prep]
    germs = [(x, ), (z, ), (z, x, x), (z, z, x)]

    return GSTSpec(prep, meas, germs, "std1Q_XZ")


def make_2q_xy_pi_2_cphase_spec():
    """Return a two-qubit gate set using a CPHASE (CZ) gate and local π/2 x- and
    y-rotations, corresponding to pyGSTi's `std1Q_XZ` model.
    """
    xi = (Gate("rx", (np.pi / 2, ), (0, )), )
    ix = (Gate("rx", (np.pi / 2, ), (1, )), )
    yi = (Gate("ry", (np.pi / 2, ), (0, )), )
    iy = (Gate("ry", (np.pi / 2, ), (1, )), )
    cz = (Gate("cz", (), (0, 1)), )

    prep = [(), (ix, ), (iy, ), (ix, ix), (xi, ), (xi, ix), (xi, iy), (xi, ix, ix),
            (yi, ), (yi, ix), (yi, iy), (yi, ix, ix), (xi, xi), (xi, xi, ix),
            (xi, xi, iy), (xi, xi, ix, ix)]

    meas = [(), (ix, ), (iy, ), (ix, ix), (xi, ), (yi, ), (xi, xi), (xi, ix), (xi, iy),
            (yi, ix), (yi, iy)]

    germs = [
        (xi, ), (yi, ), (ix, ), (iy, ), (cz, ), (xi, yi), (ix, iy), (iy, yi), (ix, xi),
        (ix, yi), (iy, xi), (xi, cz), (yi, cz), (ix, cz), (iy, cz), (xi, xi, yi),
        (ix, ix, iy), (ix, iy, cz), (xi, yi, yi), (ix, iy, iy), (iy, xi, xi),
        (iy, xi, yi), (ix, xi, iy), (ix, yi, xi), (ix, yi, iy), (ix, iy, yi),
        (ix, iy, xi), (iy, yi, xi), (xi, cz, cz), (ix, xi, cz), (ix, cz, cz),
        (yi, cz, cz), (iy, xi, cz), (iy, yi, cz), (iy, cz, cz), (ix, yi, cz),
        (cz, ix, xi, xi), (yi, ix, xi, iy), (ix, iy, xi, yi), (ix, ix, ix, iy),
        (xi, yi, yi, yi), (yi, yi, iy, yi), (yi, ix, ix, ix), (xi, yi, ix, ix),
        (cz, ix, cz, iy), (ix, xi, yi, cz), (iy, yi, xi, xi, iy), (xi, xi, iy, yi, iy),
        (iy, ix, xi, ix, xi), (yi, iy, yi, ix, ix), (iy, xi, ix, iy, yi),
        (iy, iy, xi, yi, xi), (ix, yi, ix, ix, cz), (xi, ix, iy, xi, iy, yi),
        (xi, iy, ix, yi, ix, ix), (cz, ix, yi, cz, iy, xi), (xi, xi, yi, xi, yi, yi),
        (ix, ix, iy, ix, iy, iy), (yi, xi, ix, iy, xi, ix), (yi, xi, ix, xi, ix, iy),
        (xi, ix, iy, iy, xi, yi), (ix, iy, iy, ix, xi, xi), (yi, iy, xi, iy, iy, iy),
        (yi, yi, yi, iy, yi, ix), (iy, iy, xi, iy, ix, iy),
        (iy, ix, yi, yi, ix, xi, iy), (yi, xi, iy, xi, ix, xi, yi, iy),
        (ix, ix, yi, xi, iy, xi, iy, yi)
    ]

    def flatten(gs):
        return [tuple(chain.from_iterable(g)) for g in gs]

    return GSTSpec(flatten(prep), flatten(meas), flatten(germs), "std2Q_XYCPHASE")
