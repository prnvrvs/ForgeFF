"""Tests for active-learning algorithms."""

import numpy as np

from forgeff.grade.maxvol import MaxVol, _maxvol, _mlip


def test_maxvol() -> None:
    """Test if MaxVol algorithms give the same result.

    Since the MaxVol algorithm is a greedy algorithm and does not necessarily
    give the same result as the exhaustive search, but for simple systems they
    may agree.
    """
    rng = np.random.default_rng(42)
    matrix = rng.random((20, 10))
    result_ref = MaxVol(algorithm="exhaust").run(matrix)
    result = MaxVol(algorithm="maxvol", init_method="random", rng=rng).run(matrix)

    np.testing.assert_array_equal(result.indices, result_ref.indices)


def test_maxvol_handles_zero_iterations() -> None:
    matrix = np.eye(3)
    indices = np.array([0, 1, 2])

    result = _maxvol(matrix, indices, maxiter=0)

    assert result.nit == 0
    np.testing.assert_array_equal(result.indices, indices)


def test_mlip_handles_zero_iterations() -> None:
    matrix = np.eye(3)

    result = _mlip(matrix, maxiter=0)

    assert result.nit == 0
    assert result.success is False
