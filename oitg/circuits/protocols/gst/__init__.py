"""Implements circuit generation for Gate Set Tomography.

Gate Set Tomography is a method for self-consistent process tomography of a given set of
gates. It does not require any prior knowledge about the gates used to prepare and
measure in various bases, and is robust against state preparation and measurement
errors.

The protocol was developed by and around Robin Blume-Kohout at Sandia; see e.g.
[BGN+13]_ and [Blum15]_ for more details, and [KWS+14]_ and [BGN+17]_ for experimental
demonstrations.

This module only contains the (small amount of) code necessary to generate a list of
gate sequences for a GST experiment. While analysis is relatively straightforward, there
is a comprehensive and well-tested open-source implementation already available in the
form of the [pyGSTi]_ package.

.. rubric:: References
.. [BGN+13] Blume-Kohout, R. et al. Robust, self-consistent, closed-form tomography of
   quantum logic gates on a trapped ion qubit. arxiv:1310.4492 (2013).
.. [Blum15] Blume-Kohout, R. et al. Report: Turbocharging Quantum Tomography.
   (Sandia National Laboratories, 2015).
.. [KWS+14] Kim, D. et al. Microwave-driven coherent operation of a semiconductor
   quantum dot charge qubit. Nature Nanotechnology 10, 243–247 (2015).
.. [BGN+17] Blume-Kohout, R. et al. Demonstration of qubit operations below a rigorous
   fault tolerance threshold with gate set tomography. Nature Communications 8, (2017).
.. [pyGSTi] A python implementation of Gate Set Tomography. http://www.pygsti.info/
"""

from typing import List
from ...gate import GateSequence


class GSTSpec:
    """Specifies a model for Gate Set Tomography.

    :param prep_fiducials: The list of preparation fiducials to use.
    :param meas_fiducials: The list of measurement fiducials to use.
    :param germs: The list of gate sequence germs to use.
    :param pygsti_name: The name of the equivalent pyGSTi standard model construction,
        if any.
    """
    def __init__(self,
                 prep_fiducials: List[GateSequence],
                 meas_fiducials: List[GateSequence],
                 germs: List[GateSequence],
                 pygsti_name: str = ""):
        self.prep_fiducials = prep_fiducials
        self.meas_fiducials = meas_fiducials
        self.germs = germs
        self.pygsti_name = pygsti_name
