"""A few helpers for manipulating quantum states/channels as NumPy arrays."""

import numpy as np


def mat2vec(matrix):
    """Return the given matrix in vector form.

    As per NumPy shape conventions, this is done by stacking rows one after another. The
    convention chosen is irrelevant, though, as long as it is used consistently.

    See :func:`vec2mat`.
    """
    return matrix.reshape(-1)


def vec2mat(vector):
    """Transforms a matrix-as-vector back into a matrix.

    See :func:`mat2vec`.
    """
    float_dim = np.sqrt(vector.shape[0])
    dim = int(float_dim)
    assert float_dim == dim
    return vector.reshape(dim, dim)


def projector(ket):
    r"""Return the projector :math:`\left|\psi\right>\left<\psi\right|` for the given
    ket :math:`\left|\psi\right>` as a dense matrix.
    """
    return np.outer(ket, np.conjugate(ket))


def _sqrt_dim(a):
    float_dim = np.sqrt(a.shape[0])
    dim = int(float_dim)
    assert dim == float_dim
    return dim


def choi2liou(choi):
    """Convert a superoperator in Choi representation to Liouville representation.

    With our normalisation convention, this is *almost* an involution up to the
    different normalisation factors.

    See :func:`liou2choi`.
    """
    dim = _sqrt_dim(choi)
    return dim * np.reshape(choi, (dim, dim, dim, dim)).swapaxes(0, 3).reshape(
        (dim**2, dim**2))


def liou2choi(liou):
    """Convert a superoperator in Liouville representation to Choi representation.

    See :func:`choi2liou`.
    """
    return choi2liou(liou) / liou.shape[0]


def avg_gate_fidelity(liou, target_unitary):
    r"""Compute the average gate fidelity of the given superoperator to the target
    unitary.

    :param liou: The superoperator, in Liouville form (i.e
        :math:`\overline{U} \otimes U` in the ideal case, where :math:`U` is the given
        target unitary).
    :param target_unitary: The unitary matrix of the target gate `liou` is supposed to
        implement.
    """
    target_liou = np.kron(np.conj(target_unitary), target_unitary)
    dim = _sqrt_dim(liou)
    return (np.real(np.trace(liou @ np.conjugate(target_liou).T)) + dim) / (dim**2 +
                                                                            dim)
