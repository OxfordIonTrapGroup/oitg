"""Functionality for building and running experiments that correspond to a quantum
circuit on a number of qubits, and for analysing and visualising the results.

Like much of ``oitg``, most of this code will be used from within ARTIQ experiments, or
to analyse results produced by them. However, the code here should *not* directly depend
on ARTIQ; such artefacts should be placed in ``oxart.circuits`` instead.
"""
