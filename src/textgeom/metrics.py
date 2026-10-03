"""Measurements from the embedding-geometry proposal (RC, move directions, path distances)."""
import numpy as np
from numba import njit
from scipy.spatial.distance import pdist


def rc(X: np.ndarray) -> float:
    """Relative contrast s/m of pairwise distances. ~1/sqrt(2d) for Gaussian points in d dims."""
    d = pdist(X)
    return float(d.std() / d.mean())


def move_directions(paths: np.ndarray, k: int) -> np.ndarray:
    """Unit moves u_{i,k} = delta/|delta| for paths of shape (n, T, q)."""
    d = paths[:, k] - paths[:, k - 1]
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def common_direction(U: np.ndarray) -> float:
    """R_k = |mean of unit moves|; ~1/sqrt(n) when directions are random."""
    return float(np.linalg.norm(U.mean(axis=0)))


def centered_cosines(U: np.ndarray) -> np.ndarray:
    """Cosines between moves after removing the common direction."""
    V = U - U.mean(axis=0)
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    return V @ V.T


def cost_matrix(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    P = P.astype(np.float64)
    Q = Q.astype(np.float64)
    d2 = (P**2).sum(1)[:, None] + (Q**2).sum(1)[None, :] - 2 * P @ Q.T
    return np.sqrt(np.maximum(d2, 0.0))


@njit(cache=True)
def _frechet(C):
    n, m = C.shape
    F = np.empty((n, m))
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                F[i, j] = C[0, 0]
            elif i == 0:
                F[i, j] = max(F[0, j - 1], C[0, j])
            elif j == 0:
                F[i, j] = max(F[i - 1, 0], C[i, 0])
            else:
                F[i, j] = max(min(F[i - 1, j], F[i, j - 1], F[i - 1, j - 1]), C[i, j])
    return F[n - 1, m - 1]


@njit(cache=True)
def _dtw(C):
    n, m = C.shape
    D = np.empty((n, m))
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                D[i, j] = C[0, 0]
            elif i == 0:
                D[i, j] = D[0, j - 1] + C[0, j]
            elif j == 0:
                D[i, j] = D[i - 1, 0] + C[i, 0]
            else:
                D[i, j] = min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1]) + C[i, j]
    return D[n - 1, m - 1]


def frechet(P, Q) -> float:
    """Discrete Frechet distance (shortest leash)."""
    return float(_frechet(cost_matrix(P, Q)))


def dtw(P, Q) -> float:
    """Dynamic time warping: sum of leash lengths along the best walk."""
    return float(_dtw(cost_matrix(P, Q)))


def procrustes_prep(P: np.ndarray) -> np.ndarray:
    """Center, scale to unit norm; return W = U S so nuclear norm of P^T Q = that of W_P^T W_Q."""
    P = P.astype(np.float64)
    P = P - P.mean(axis=0)
    P = P / np.linalg.norm(P)
    U, S, _ = np.linalg.svd(P, full_matrices=False)
    return U * S


def procrustes_from_prep(Wp: np.ndarray, Wq: np.ndarray) -> float:
    nuc = np.linalg.svd(Wp.T @ Wq, compute_uv=False).sum()
    return float(np.sqrt(max(2.0 - 2.0 * nuc, 0.0)))


def procrustes(P, Q) -> float:
    """Shape distance after centering, unit scaling and the best rotation/reflection."""
    return procrustes_from_prep(procrustes_prep(P), procrustes_prep(Q))
