"""Implements different methods for reconstructing an estimate for a quantum channel
(process) from tomography measurements.

Currently, linear inversion tomography and two maximum-likelihood techniques are
supported.

All implementations assume that the processes in question have the same input and output
dimensions, i.e. are endomorphisms on the space of bounded operators (density matrices)
on some qubit Hilbert space.

.. rubric:: References
.. [FH01] Fiurášek, J. & Hradil, Z.
    Maximum-likelihood estimation of quantum processes.
    Physical Review A 63, (2001).
.. [RHKL07] Reháček, J., Hradil, Z., Knill, E. & Lvovsky, A. I.
    Diluted maximum-likelihood algorithm for quantum tomography.
    Physical Review A 75, 1–5 (2007).
.. [AL12] Anis, A. & Lvovsky, A. I.
    Maximum-likelihood coherent-state quantum process tomography.
    New J. Phys. 14, 105021 (2012).
.. [KBLG18] Knee, G. C., Bolduc, E., Leach, J. & Gauger, E. M.
    Quantum process tomography via completely positive and trace-preserving projection.
    Physical Review A 98, (2018).
"""

import numpy as np
import warnings
from typing import Dict, List, Optional
from ...gate import *
from ...to_matrix import gate_sequence_matrix
from .tools import *


def _find_first_index(needle, haystack):
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            return i
    raise ValueError


def guess_prepare_target_measure_split(
    all_sequences: List[GateSequence]
) -> Tuple[GateSequence, List[Tuple[GateSequence, GateSequence]]]:
    """For a given list of gate sequences making up a tomography experiment, guesses
    which is the target sequence to be analysed, and the preparation/measuemrent
    fiducial sequences used.

    :param all_sequences: A list of all the gate sequence for which data was acquired.
    :return: A tuple ``(target_seq, [(prep_seq, meas_seq)])`` of the guess for the
        target sequence and, for each input sequence, the respective state preparation
        sequence before/measurement sequence after the tomography target.
    """
    # FIXME: This is needlessly generic, yet fails if the target is more than one gate
    # or it also appears as part of the prepare/measure fiducials…
    possible_targets = set(all_sequences[0])
    for seq in all_sequences[1:]:
        possible_targets &= set(seq)
    if len(possible_targets) > 1:
        raise ValueError
    target_seq = tuple(possible_targets)

    target_start_idxs = [_find_first_index(target_seq, s) for s in all_sequences]

    return target_seq, [(s[:t], s[t + len(target_seq):])
                        for t, s in zip(target_start_idxs, all_sequences)]


def auto_prepare_data(
    outcomes: Dict[GateSequence, np.ndarray]
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    """Given a results dictionary, guess the target and fiducial sequences and extract
    the tomography input data.

    See :func:`prepare_data` for details.

    :param outcomes: A dictionary mapping sequences run to the number each outcome was
        observed.
    :return: A tuple of prepared states, measured states, and the respective number of
        observations each (see :func:`prepare_data`).
    """
    seqs = list(outcomes.keys())
    _, fiducial_pairs = guess_prepare_target_measure_split(seqs)
    return prepare_data({f: outcomes[s] for f, s in zip(fiducial_pairs, seqs)})


def prepare_data(
    outcomes: Dict[Tuple[GateSequence, GateSequence], np.ndarray],
    initial_state: Optional[List[np.ndarray]] = None,
    readout_projectors: Optional[List[np.ndarray]] = None
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    r"""Given a dictionary of observed measurement outcomes indexed by pairs of the
    state preparation and measurement gate sequences used, compute the prepared/measured
    states and respective number of operations.

    Assumes the initial state (i.e. before the preparation sequences are run) is
    :math:`\left|00\ldots0\right>`), and outcomes are given in canonical order (i.e.
    corresponding to projection onto :math:`\left|0\ldots00\right>,
    \left|0\ldots01\right>, \ldots, \left|1\ldots11\right>`.)

    :param outcomes: A dictionary mapping tuples ``(prepare, measure)`` of state
        preparation/measurement gate sequences to an array giving the number each
        outcome was observed.

    :return: A tuple of prepared states, measured states, and the respective number of
        observations each. Rows in the retuned observation count array correspond to the
        different prepared states, and columns to the measured state. Measured states
        corresponding to the different outcomes are enumerated explicitly, so for a
        :math:`n`-qubit system and :math:`k` measurement fiducials, there will be
        :math:`k\ 2^n` measured states/columns in the observation count matrix. All
        states are given as density matrices/projectors.
    """
    fiducial_pairs = list(outcomes.keys())

    prep_sequences = list(set(f[0] for f in fiducial_pairs))
    prep_indices = [prep_sequences.index(f[0]) for f in fiducial_pairs]

    meas_sequences = list(set(f[1] for f in fiducial_pairs))
    meas_indices = [meas_sequences.index(f[1]) for f in fiducial_pairs]

    num_qubits = max(
        max(collect_operands(s), default=0) for f in fiducial_pairs for s in f) + 1

    def basis_ket(k):
        psi = np.zeros(2**num_qubits, dtype=np.complex128)
        psi[k] = 1
        return psi

    ###

    if initial_state is None:
        initial_state = projector(basis_ket(0))
    prep_unitaries = [gate_sequence_matrix(s, num_qubits) for s in prep_sequences]
    prep_projectors = [u @ initial_state @ u.T.conj() for u in prep_unitaries]

    ###

    meas_unitaries = [gate_sequence_matrix(s, num_qubits) for s in meas_sequences]
    if readout_projectors is None:
        readout_projectors = [projector(basis_ket(k)) for k in range(2**num_qubits)]
    meas_projectors = [
        u.T.conj() @ readout_projector @ u for u in meas_unitaries
        for readout_projector in readout_projectors
    ]

    ###

    observations = np.full((len(prep_projectors), len(meas_projectors)), -1)
    for prep_idx, meas_idx, fids in zip(prep_indices, meas_indices, fiducial_pairs):
        meas_base_idx = 2**num_qubits * meas_idx
        for i, counts in enumerate(outcomes[fids]):
            observations[prep_idx, meas_base_idx + i] = counts
    if np.sum(observations == -1) != 0:
        raise NotImplementedError(
            "Currently assuming all prepare/measure combinations are present")
    return prep_projectors, meas_projectors, observations


def build_choi_predictor(prep_operators: Iterable[np.ndarray],
                         meas_operators: Iterable[np.ndarray]) -> np.ndarray:
    r"""Given a list of prepared and measured states as state vectors, return a matrix
    that predicts observed probabilities when applied to the Choi matrix of a process.

    In other words, for given prepared states :math:`\rho_i` and measurement operators
    :math:`P_j`, the returned matrix :math:`M` computes the outcomes
    :math:`p_{ij} = \operatorname{tr}\left(\mathcal{E}(\rho_i) P_j \right)` by applying
    it to the Choi matrix :math:`C_\mathcal{E}`, i.e. :math:`p = M\ C_\mathcal{E}`.

    This is the same convention for the order of entries in the outcome matrix
    :math:`p` as used by :func:`prepare_data`.

    :param prep_operators: The prepared states, given as density matrices. Typically,
        the states are pure states :math:`\left(\left|\psi_i\right>\right)_i`, and
        :math:`\rho_i = \left|\psi_i\right>\left<\psi_i\right|`.
    :param meas_operators: The measurement operators. Typically, the measurements are
        ideal projections onto states :math:`\left(\left|\phi_j\right>\right)_j`, and
        :math:`P_j = \left|\phi_j\right>\left<\phi_j\right|`.
    """
    return np.vstack([
        mat2vec(np.kron(prep, meas.T)) for prep in prep_operators
        for meas in meas_operators
    ])


def invert_choi_predictor(choi_predictor: np.ndarray,
                          observations: np.ndarray) -> np.ndarray:
    r"""Obtain an estimate for the process :math:`\mathcal{E}` by applying the inverse
    of the given Choi predictor to a matrix of experimentally measured outcomes.

    If the set of measurements is over-complete, the least-squares estimate will be
    computed (which is sometimes stated by using the Moore-Penrose pseudo-inverse).

    :param choi_predictor: The Choi predictor matrix, as computed by
        :func:`build_choi_predictor`.
    :param observations: The number of observations per prepared/measured state; see
        :func:`prepare_data`. Will be normalised row-by-row as necessary.

    :return: The linear inversion tomography estimate for the process, as a Choi matrix
        :math:`C_{\mathcal{E}}`. Note that in the presence of sampling noise or
        experimental imperfections, the returned superoperator will not necessarily be
        physical, i.e. neither completely positive nor trace-preserving.
    """
    # For consistent normalisation, infer the dimension of the underlying state Hilbert
    # space and the number of measurement bases from the given predictor/observation
    # matrix.
    pure_state_dimension = np.sqrt(np.sqrt(choi_predictor.shape[1]))
    if pure_state_dimension != int(pure_state_dimension):
        raise ValueError("Choi predictor not of right shape for CPTP involution")
    num_measurement_bases, rem = divmod(observations.shape[1], pure_state_dimension)
    if rem:
        raise ValueError("Number of observation matrix columns not consistent with "
                         "dim(pure_state) measurements per basis")

    normalised_observations = observations.astype(np.float64)
    shots_per_basis = np.sum(observations, axis=1) / num_measurement_bases
    for i in range(normalised_observations.shape[0]):
        normalised_observations[i, :] /= shots_per_basis[i]

    solution, residuals, rank, singular_vals = \
        np.linalg.lstsq(choi_predictor, mat2vec(normalised_observations), rcond=None)
    if rank != solution.size:
        raise ValueError("Predictor matrix was rank-deficient; "
                         "check that input/measurement state sets are complete")
    return vec2mat(solution) / pure_state_dimension


def linear_inversion_tomography(prep_projectors: List[np.ndarray],
                                meas_operators: List[np.ndarray],
                                observations: np.ndarray) -> np.ndarray:
    """Calculate the linear inversion estimate of the quantum process that has produced
    the given observations.

    :param prep_projectors: A list of the states prepared in the tomography experiment.
    :param meas_operators: A list of the measurement operators (projectors) in the
        tomography experiment.
    :param observations: A matrix giving the number of times each outcome was observed
        in the experiment; see :func:`prepare_data`.

    :return: The linear inversion tomography estimate for the process as a Choi matrix;
        see :func:`invert_choi_predictor`.
    """
    predictor = build_choi_predictor(prep_projectors, meas_operators)
    return invert_choi_predictor(predictor, observations)


def negative_log_likelihood(choi_predictor: np.ndarray, observation_vec: np.ndarray,
                            choi: np.ndarray) -> float:
    """Return the negative log-likelihood for the given outcomes to be observed (with
    experiments as described by the Choi predictor) as a function of the given
    superoperator in Choi representation.

    See :func:`negative_log_likelihood_gradient` for calculating the gradient.
    """

    # See e.g. [KBLG18] eq. 3/appendix A.
    probability_vec = np.real(choi_predictor @ mat2vec(choi))

    # Fudge predictions away from 0 to avoid stalling as per [KBLG18] appendix D.
    mask_small = probability_vec < 1e-16
    if np.any(mask_small):
        warnings.warn("{} very small probabilities encountered".format(
            np.sum(mask_small)))
    probability_vec[mask_small] = 1e-16
    probability_vec /= np.sum(probability_vec)

    return -observation_vec.T @ np.log(probability_vec)


def negative_log_likelihood_gradient(choi_predictor: np.ndarray,
                                     observation_vec: np.ndarray,
                                     choi: np.ndarray) -> np.ndarray:
    r"""Calculate the derivative of the log-likelihood
    :math:`\mathcal{L}(C_\mathcal{E})` around the given Choi matrix
    :math:`C_\mathcal{E}`.

    See :func:`negative_log_likelihood` for computing the value at a given point.

    :return: A matrix giving the gradient :math:`\nabla \mathcal{L}(C_\mathcal{E}) =
        \frac{\partial\mathcal{L}(C_\mathcal{E})}{\partial C_\mathcal{E}}`, in the usual
        element-wise matrix calculus sense.
    """

    probability_vec = np.real(choi_predictor @ mat2vec(choi))

    # Fudge predictions away from 0 to avoid stalling as suggested in
    # [KBLG18] appendix D.
    mask_small = probability_vec < 1e-16
    if np.any(mask_small):
        warnings.warn("{} very small probabilities encountered".format(
            np.sum(mask_small)))
    probability_vec[mask_small] = 1e-16
    probability_vec /= np.sum(probability_vec)

    # See e.g. [AL12] eq. 13 and [KBLG18] eq. 6.
    # [KBLG18] suggests the equivalent of `-choi_predictor.conj().T @ vec2mat(...)` in
    # appendix A but that doesn't work out in terms of dimensions.
    return -vec2mat(choi_predictor.conj().T @ (observation_vec / probability_vec))


def _ptrace(rho, d):
    # Calculates partial trace keeping the first of two systems of dimension d. This is
    # unoptimised, horrible, not to be imitated, etc.
    result = np.zeros((d, d), dtype=np.complex128)
    for i in range(d):
        e = np.zeros((1, d))
        e[0, i] = 1
        proj = np.kron(np.eye(d), e)
        result += proj @ rho @ proj.T
    return result


def diluted_mle_tomography(choi_predictor: np.ndarray,
                           observations: np.ndarray,
                           rel_tol: float = 1e-10,
                           iteration_limit: int = 10000) -> np.ndarray:
    """Calculate the tomography estimate of the quantum process that has produced
    the given observations using a diluted fixed-point iteration method.

    By definition, the maximum-likelihood estimate for the superoperator maximises the
    likelihood function. The extremal condition can be stated in a form amenable to
    fixed-point iteration as shown in [FH01]_. The trace-preserving constraint is
    expressed via Lagrange multipliers ([FH01]_ eq. 16), and complete positivity is
    enforced at each step by explicitly forcing the iterates to be Hermitian.

    The particular formulation used here is described in [AL12]_ section 2.1. The
    dilution parameter is chosen to ensure progress at each step as suggested in
    [RHKL07]_ (but without any of the possible further optimisations for maximising the
    likelihood increase discussed there).

    :param choi_predictor: The Choi predictor matrix, see :func:`build_choi_predictor`.
    :param observations: The number of observations per prepared/measured state; see
        :func:`prepare_data`. Will be normalised row-by-row as necessary.
    :param rel_tol: Stopping criterion, given as relative change of the negative
        log-likelihood.
    :param iteration_limit: Maximum number of iterations; an error is thrown if it is
        reached before the stopping criterion is fulfilled.

    :return: The CPTP tomography estimate for the process, given in the Choi
        representation.
    """
    import scipy.linalg

    float_pure_state_dimension = np.sqrt(np.sqrt(choi_predictor.shape[1]))
    pure_state_dimension = int(float_pure_state_dimension)
    if pure_state_dimension != float_pure_state_dimension:
        raise ValueError("Choi predictor not of right shape for CPTP endomorphism")

    # Normalise so that sum of all probabilities is 1.0, and applying the predictor to a
    # Choi matrix with normalisation such that entries are in [-1, 1] yields that.
    number_of_basis_combinations = choi_predictor.shape[0] / pure_state_dimension
    choi_predictor = choi_predictor / number_of_basis_combinations
    observation_vec = mat2vec(observations) / np.sum(observations)

    choi = np.eye(pure_state_dimension**2, dtype=np.complex128) / pure_state_dimension

    old_nll = np.inf
    for iter_idx in range(iteration_limit):
        grad = negative_log_likelihood_gradient(choi_predictor, observation_vec, choi)

        # Decrease dilution parameter `eps` until likelihood starts to increase.
        eps = 1.0
        new_choi = None
        while True:
            diluted_grad = eps * grad + (1 - eps) * np.eye(pure_state_dimension**2)

            # Calculate λ, the TP Lagrange multiplier matrix, see e.g. [AL12] eq. 17.
            lambda_ = scipy.linalg.sqrtm(
                _ptrace(diluted_grad @ choi @ diluted_grad, pure_state_dimension))

            # Build inverse of λ ⊗ id as used in the iteration step (e.g. [AL12]
            # eq. 16).
            # For large systems, we would probably want to write the multiplications
            # with the inverse in terms of linalg.solve instead, but just calculating
            # the inverse avoids tensor product index gymnastics for now.
            lambda_inv = np.kron(np.linalg.inv(np.complex128(lambda_)), np.eye(pure_state_dimension))

            new_choi = lambda_inv @ diluted_grad @ choi @ diluted_grad @ lambda_inv
            if negative_log_likelihood(choi_predictor, observation_vec,
                                       new_choi) < old_nll:
                break
            eps /= 2
            if eps < 1e-16:
                raise ValueError("Did not converge (dilution limit reached, but "
                                 "likelihood still not decreasing)")

        # Ensure complete positivity, i.e. Hermitian Choi matrix.
        choi = (new_choi + new_choi.T.conj()) / 2

        # If the relative change in likelihood is smaller than the tolerance specified,
        # we are done.
        new_nll = negative_log_likelihood(choi_predictor, observation_vec, choi)
        if (old_nll - new_nll) / new_nll < rel_tol:
            return choi / pure_state_dimension
        old_nll = new_nll
    raise ValueError("Did not converge (iteration limit reached)")


def project_into_cp(choi: np.ndarray) -> np.ndarray:
    """For the given superoperator (in Choi representation), return the nearest
    completely-positive one.

    See [KBLG18]_ eq. 8.
    """
    eigvals, eigvecs = np.linalg.eigh((choi + choi.conj().T) / 2)
    eigvals[eigvals < 0.0] = 0.0
    return eigvecs @ np.diag(eigvals) @ eigvecs.conj().T


class TPProjector:
    r"""Projects superoperators into the trace-preserving subspace.

    (This is a class to allow some ancillary matrices to be re-used across different
    :meth:`project` invocations.)

    :param pure_state_dimension: The dimension :math:`d` of the Hilbert space the
        density operators of which the superoperators to project act on; i.e.
        :math:`d = 2^n` for :math:`n` qubits, and the Choi matrices are
        :math:`d^2 \times d^2` in size.
    """
    def __init__(self, pure_state_dimension):
        self.dim = pure_state_dimension

        m = np.zeros((pure_state_dimension**2, pure_state_dimension**4))
        for i in range(pure_state_dimension):
            e = np.zeros(pure_state_dimension)
            e[i] = 1.0
            b = np.kron(np.eye(pure_state_dimension), e.T)
            m += np.kron(b, b)
        self.mdagger_m = m.conj().T @ m
        self.mdagger_id = m.conj().T @ mat2vec(np.eye(pure_state_dimension))

    def project(self, choi: np.ndarray) -> np.ndarray:
        """Return the result of orthogonally projecting the given Choi operator into the
        trace-preserving subspace.

        See [KBLG18]_ eq. 12.
        """
        return choi + vec2mat(self.mdagger_id -
                              self.mdagger_m @ mat2vec(choi)) / self.dim


def project_into_cptp(choi: np.ndarray,
                      tp_projector: TPProjector,
                      tol: float = 1e-4,
                      iteration_limit: int = 10000) -> np.ndarray:
    """Project the given Choi matrix onto the closest superoperator that is both
    completely positive and trace-preserving.

    This implements Algorithm 1 from [KBLG18]_.

    If the iteration did not converge after the given number of steps, an exception is
    raised.
    """
    old_cp_step = np.zeros_like(choi)
    old_tp_step = np.zeros_like(choi)
    old_after_cp = np.zeros_like(choi)
    old_choi = choi

    for i in range(iteration_limit):
        before_cp = old_choi + old_cp_step
        after_cp = project_into_cp(before_cp)
        cp_step = before_cp - after_cp

        before_tp = after_cp + old_tp_step
        choi = tp_projector.project(before_tp)
        tp_step = before_tp - choi

        if (np.linalg.norm(old_cp_step - cp_step)**2 +
                np.linalg.norm(old_tp_step - tp_step)**2 +
                np.abs(2 * mat2vec(old_cp_step).conj().T @ mat2vec(choi - old_choi)) +
                np.abs(2 *
                       mat2vec(old_tp_step).conj().T @ mat2vec(after_cp - old_after_cp))
                < tol):
            return choi
        old_cp_step = cp_step
        old_tp_step = tp_step
        old_after_cp = after_cp
        old_choi = choi
    raise ValueError("Did not converge")


def pgdb_mle_tomography(choi_predictor: np.ndarray,
                        observations: np.ndarray) -> np.ndarray:
    """Calculate the tomography estimate of the quantum process that has produced
    the given observations using a projected gradient descent algorithm with
    backtracking, as proposed by [KBLG18]_.

    :param choi_predictor: The Choi predictor matrix, see :func:`build_choi_predictor`.
    :param observations: The number of observations per prepared/measured state; see
        :func:`prepare_data`. Will be normalised row-by-row as necessary.

    :return: The CPTP tomography estimate for the process, given in the Choi
        representation.
    """
    float_pure_state_dimension = np.sqrt(np.sqrt(choi_predictor.shape[1]))
    pure_state_dimension = int(float_pure_state_dimension)
    if pure_state_dimension != float_pure_state_dimension:
        raise ValueError("Choi predictor not of right shape for CPTP endomorphism")

    tp_projector = TPProjector(pure_state_dimension)

    # Note: Different normalisation used here to match statement of algorithm in
    # [KBLG18]; divisor would be pure_state_dimension**2 with our usual convention.
    choi = np.eye(pure_state_dimension**2, dtype=np.complex128) / pure_state_dimension

    mu = 1.5 / pure_state_dimension**2
    gamma = 0.3

    # Normalise so that sum of all probabilities is 1.0, and applying the predictor to a
    # Choi matrix with normalisation such that entries are in [-1, 1] yields that.
    number_of_basis_combinations = choi_predictor.shape[0] / pure_state_dimension
    choi_predictor = choi_predictor / number_of_basis_combinations
    observation_vec = mat2vec(observations) / np.sum(observations)

    old_nll = negative_log_likelihood(choi_predictor, observation_vec, choi)
    while True:
        nll_gradient = negative_log_likelihood_gradient(choi_predictor, observation_vec,
                                                        choi)

        choi_step = project_into_cptp(choi - nll_gradient / mu, tp_projector) - choi

        alpha = 1.0
        change = gamma * mat2vec(choi_step).conj().T @ mat2vec(nll_gradient)

        while True:
            nll = negative_log_likelihood(choi_predictor, observation_vec,
                                          choi + alpha * choi_step)
            if nll <= old_nll + change or alpha < 1e-10:
                break
            alpha /= 2
            change /= 2

        choi += alpha * choi_step
        nll = negative_log_likelihood(choi_predictor, observation_vec, choi)
        if old_nll - nll < 1e-10:
            break
        old_nll = nll

    # Convert back to our usual normalisation convention.
    return choi / pure_state_dimension
