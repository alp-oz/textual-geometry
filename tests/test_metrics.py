import numpy as np
import pytest

from textgeom.metrics import common_direction, dtw, frechet, move_directions, procrustes, rc

A = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], float)
B = np.array([[0, 0], [0, 0], [0, 0], [1, 0], [2, 0], [3, 0]], float)
C = A + [0, 1]
E = np.array([[0, 0], [1, 0], [2, 5], [3, 0]], float)


def test_proposal_example_frechet():
    assert frechet(A, B) == pytest.approx(0)
    assert frechet(A, C) == pytest.approx(1)
    assert frechet(A, E) == pytest.approx(5)


def test_proposal_example_dtw():
    assert dtw(A, B) == pytest.approx(0)
    assert dtw(A, C) == pytest.approx(4)
    assert dtw(A, E) == pytest.approx(5)


def test_proposal_example_procrustes():
    assert procrustes(A, C) == pytest.approx(0, abs=1e-6)
    assert procrustes(A, E) > 0.1


def test_procrustes_invariances():
    rng = np.random.default_rng(0)
    P = rng.normal(size=(30, 5))
    Q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    assert procrustes(P, 3 * P @ Q + 7) == pytest.approx(0, abs=1e-6)


def test_rc_gaussian():
    X = np.random.default_rng(0).normal(size=(400, 50))
    assert rc(X) == pytest.approx(1 / np.sqrt(2 * 50), rel=0.25)


def test_common_direction():
    rng = np.random.default_rng(0)
    same = np.cumsum(np.tile([1.0, 0, 0], (50, 3, 1)) + 0.01 * rng.normal(size=(50, 3, 3)), axis=1)
    assert common_direction(move_directions(same, 1)) > 0.99
    rand = rng.normal(size=(400, 3, 40)).cumsum(axis=1)
    assert common_direction(move_directions(rand, 1)) < 0.2
