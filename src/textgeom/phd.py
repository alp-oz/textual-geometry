"""Persistent-homology (MST) intrinsic dimension, after Tulchinskii et al. 2023.

E_alpha(X) = sum of |e|^alpha over edges of the Euclidean MST of X
           ~ C * n^((d - alpha)/d)   (Steele 1988)
so with alpha = 1, slope k of log E vs log n gives d = 1 / (1 - k).
"""
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform


def mst_power_sum(X: np.ndarray, alpha: float = 1.0) -> float:
    """Sum of MST edge lengths ** alpha (= 0-dim persistence energy)."""
    D = squareform(pdist(X))
    # scipy treats 0 as "no edge"; identical points (e.g. a repeated token) must stay connected at a negligible length
    D[(D == 0) & ~np.eye(len(D), dtype=bool)] = 1e-6  # scipy also drops values within ~1e-8 of 0
    T = minimum_spanning_tree(D)
    return float(np.sum(T.data ** alpha))


def phd(
    X: np.ndarray,
    alpha: float = 1.0,
    min_n: int = 10,
    n_sizes: int = 8,
    n_draws: int = 7,
    n_fits: int = 3,
    seed: int = 0,
) -> float:
    """Estimate intrinsic dimension of point cloud X (n_points, dim)."""
    rng = np.random.default_rng(seed)
    n = len(X)
    if n < 2 * min_n:
        raise ValueError(f"need at least {2 * min_n} points, got {n}")
    sizes = np.unique(np.linspace(min_n, n, n_sizes).astype(int))
    dims = []
    for _ in range(n_fits):
        logE = []
        for m in sizes:
            vals = []
            for _ in range(n_draws):
                idx = rng.choice(n, size=m, replace=False)
                vals.append(mst_power_sum(X[idx], alpha))
            logE.append(np.log(np.mean(vals)))
        slope = np.polyfit(np.log(sizes), logE, 1)[0]
        # E ~ n^((d-alpha)/d)  =>  slope = 1 - alpha/d  =>  d = alpha/(1-slope)
        if slope < 1:
            dims.append(alpha / (1.0 - slope))
    return float(np.mean(dims)) if dims else float("nan")
