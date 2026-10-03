import numpy as np
import pytest

from textgeom import chunk_tokens, mst_power_sum, phd


def embedded_sphere(d, n, ambient=50, seed=0):
    """n points on a d-dim subspace (uniform cube) rotated into R^ambient."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(ambient, d)))
    return rng.uniform(size=(n, d)) @ Q.T


def test_mst_line():
    X = np.arange(5, dtype=float)[:, None]
    assert mst_power_sum(X) == pytest.approx(4.0)


@pytest.mark.parametrize("d", [2, 4, 6])
def test_phd_recovers_dimension(d):
    est = phd(embedded_sphere(d, 600), seed=1)
    assert abs(est - d) < 0.25 * d + 0.5


def test_phd_orders_dimensions():
    assert phd(embedded_sphere(3, 500)) < phd(embedded_sphere(8, 500))


def test_too_few_points():
    with pytest.raises(ValueError):
        phd(np.zeros((5, 3)))


def test_chunking_drops_short_tail():
    assert [len(c) for c in chunk_tokens(list(range(600)), 256, 128)] == [256, 256]


def test_duplicate_points_do_not_inflate_mst():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 5))
    dup = np.concatenate([X, X[:10], X[:10]])      # 20 exact duplicates
    assert mst_power_sum(dup) == pytest.approx(mst_power_sum(X), rel=1e-3)
