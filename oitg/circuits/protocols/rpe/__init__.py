r"""Contains a generic implementation of robust phase estimation for single-qubit gate
rotation angles.

Robust phase estimation formalises the intuitive strategy of iteratively using more and
more repetitions of a given pulse to increase the calibration precision without
"skipping a fringe".

Sequences of power-of-two length are used to iteratively subdivide the interval of
possible phases; theoretically achieving :math:`\mathrm{O}(1 / N)` scaling (where
:math:`N` is the sequence length). By also including sequences with an extra π/2 pulse
at the end, the protocol is robust against SPAM errors.

A general description of the protocol is given in [KLY15]_. We follow the same procedure
described by the Sandia group in [RKLM17]_. (A variant of the protocol can be used to
extract the angle between the rotation axis of two gates; this is not implemented here.)

.. rubric:: References

.. [KLY15] Kimmel, S., Low, G. H. & Yoder, T. J. Robust calibration of a universal
    single-qubit gate set via robust phase estimation. Physical Review A 062315, 1–13
    (2015).

.. [RKLM17] Rudinger, K., Kimmel, S., Lobser, D. & Maunz, P. Experimental Demonstration
    of a Cheap and Accurate Phase Estimation. Physical Review Letters 118, 1–6 (2017).
"""
