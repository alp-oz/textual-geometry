"""Geometry of a text's trajectory: states x_1..x_T of a causal LM after each token."""
import numpy as np

from .phd import phd


def trajectory_features(X: np.ndarray, prefixes=(32, 64, 96, 128)) -> dict:
    """X: (T, q) states, one per token, in reading order."""
    d = np.diff(X, axis=0)
    step = np.linalg.norm(d, axis=1)
    U = d / step[:, None]
    f = {
        "mean_step": float(step.mean()),
        "straightness": float(np.linalg.norm(X[-1] - X[0]) / step.sum()),  # 1 = straight line
        "turn_cos": float((U[1:] * U[:-1]).sum(1).mean()),                 # 1 = keeps going, 0 = random turns
        "spread": float(np.linalg.norm(X - X.mean(0), axis=1).mean() / step.mean()),
    }
    for k in prefixes:
        if k <= len(X):
            f[f"dim_{k}"] = phd(X[:k], min_n=8, n_sizes=6, n_draws=5, n_fits=2, seed=0)
    return f
