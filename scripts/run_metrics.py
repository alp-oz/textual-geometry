"""Apply the proposal's measurements to texts. A "patient" is one text unit (poem/sonnet/window),
truncated to L tokens; its path is the sequence of its L token embeddings (xlm-roberta-base).

    python scripts/run_metrics.py --out metrics_results.json
"""
import argparse
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np

from textgeom import chunk_tokens, load_text, phd
from textgeom.corpus import split_baudelaire, split_delimited, split_sonnets
from textgeom.metrics import (centered_cosines, common_direction, cost_matrix, _dtw, _frechet,
                              move_directions, procrustes_from_prep, procrustes_prep, rc)

SYMBOLIC = {
    "baudelaire": ["L'ALBATROS", "ELEVATION", "LA VIE ANTERIEURE", "LE CYGNE", "LES CHATS", "LE VOYAGE", "LA BEAUTE"],
    "mallarme": ["Brise marine", "L'Azur", "Le Cygne", "Hérodiade", "Sonnet", "Ses purs ongles", "Sainte", "Don du poème"],
}


def windows(path, n, rng, L, start_char=0):
    """n random text windows from a long file (each long enough for >= L tokens)."""
    t = load_text(path)[start_char:]
    starts = rng.choice(len(t) - 1200, size=n * 3, replace=False)
    return [(f"{Path(path).stem}@{s}", t[s:s + 1200]) for s in sorted(starts)]


def build_groups(data, rng, L, n_windows):
    g = {}
    g["shakespeare_sonnets"] = split_sonnets(f"{data}/shakespeare/sonnets.txt")
    g["shakespeare_plays"] = windows(f"{data}/shakespeare/complete_works.txt", n_windows, rng, L, start_char=250_000)
    g["baudelaire"] = split_baudelaire(f"{data}/baudelaire/fleurs_du_mal.txt")
    mp = Path(f"{data}/mallarme/poesies.txt")
    if mp.exists() and mp.stat().st_size > 0:
        g["mallarme"] = split_delimited(mp)
    g["beckett_fr"] = windows(f"{data}/beckett/linnommable_fr.txt", n_windows, rng, L, start_char=3000)
    g["ramanujan_proof_ch5"] = windows(f"{data}/ramanujan/andrews_ch5_hrr_expansion.txt", n_windows, rng, L)
    other = [f"{data}/ramanujan/andrews_ch1_elementary.txt", f"{data}/ramanujan/andrews_ch13_combinatorics.txt"]
    g["math_other"] = sum((windows(p, n_windows // 2, rng, L) for p in other), [])
    return g


def embed_groups(groups, L, cache):
    if Path(cache).exists():
        z = np.load(cache, allow_pickle=True)
        return z["emb"].item(), z["toks"].item(), z["labels"].item()
    from textgeom.embed import Embedder
    emb = Embedder("xlm-roberta-base")
    E, T, Lab = {}, {}, {}
    for name, units in groups.items():
        E[name], T[name], Lab[name] = [], [], []
        for label, text in units:
            ids = emb.token_ids(text)
            if len(ids) < L:
                continue
            ids = ids[:L]
            E[name].append(emb.embed_ids(ids).astype(np.float32)); T[name].append(ids); Lab[name].append(label)
        E[name] = np.stack(E[name]); T[name] = np.array(T[name])
        print(f"{name}: {len(E[name])} patients", flush=True)
    np.savez(cache, emb=np.array(E, dtype=object), toks=np.array(T, dtype=object), labels=np.array(Lab, dtype=object))
    return E, T, Lab


def dim(X, seed=0):
    return phd(X, min_n=10, n_sizes=8, n_draws=7, n_fits=3, seed=seed)


def shuffled(X, rng):
    return np.stack([rng.permutation(X[:, j]) for j in range(X.shape[1])], axis=1)


def pairwise(paths, fn):
    n = len(paths); D = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        D[i, j] = D[j, i] = fn(paths[i], paths[j])
    return D


def group_means(D, labels, names):
    out = {}
    for a in names:
        for b in names:
            ia, ib = np.where(labels == a)[0], np.where(labels == b)[0]
            block = D[np.ix_(ia, ib)]
            out[f"{a}|{b}"] = float(block[~np.eye(len(ia), dtype=bool)].mean() if a == b else block.mean())
    return out


def perm_test(D, labels, n_perm, rng):
    """Within-group minus between-group mean distance; labels shuffled for the null."""
    iu = np.triu_indices(len(labels), 1)
    def stat(lab):
        same = (lab[:, None] == lab[None, :])[iu]
        return D[iu][same].mean() - D[iu][~same].mean()
    obs = stat(labels)
    null = np.array([stat(rng.permutation(labels)) for _ in range(n_perm)])
    return float(obs), float((null <= obs).mean()), float(null.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--L", type=int, default=96)
    ap.add_argument("--n-max", type=int, default=80)
    ap.add_argument("--cache", default="/tmp/embeddings_L96.npz")
    ap.add_argument("--out", default="metrics_results.json")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    L = args.L

    groups = build_groups(args.data, rng, L, n_windows=args.n_max)
    E, T, Lab = embed_groups(groups, L, args.cache)
    names = [n for n in E if len(E[n]) >= 20]
    # equal-size random subsets
    sel = {n: np.sort(rng.choice(len(E[n]), size=min(len(E[n]), args.n_max), replace=False)) for n in names}
    res = {"L": L, "n": {n: int(len(sel[n])) for n in names}}
    print("patients per group:", res["n"])

    # --- Q3, Q4, Q1: per position k, point set = embeddings of all patients of a group after k tokens
    ks = [4, 16, 48, L - 1]
    res["geometry"] = {}
    for n in names:
        P = E[n][sel[n]]
        rows = []
        for k in ks:
            X = P[:, k]
            d, dsh = dim(X), dim(shuffled(X, rng))
            if np.isnan(dsh):  # slope >= 1: shuffled cloud is too high-dimensional to estimate at this N
                dsh = float('inf')
            r = rc(X)
            U = move_directions(P, k)
            rows.append(dict(k=k, d=d, d_shuffled=dsh, RC=r, RC_random_same_d=1 / np.sqrt(2 * d),
                             R=common_direction(U), R_random=1 / np.sqrt(len(P))))
        res["geometry"][n] = rows
        print(f"\n{n} (N={len(P)})")
        for r in rows:
            print("  k={k:3d} d={d:5.1f} d_shuf={d_shuffled:5.1f} RC={RC:.3f} (random {RC_random_same_d:.3f}) "
                  "R={R:.3f} (random {R_random:.3f})".format(**r))

    # --- Q2: same token => same move?  pooled over groups, common direction removed per position
    toks = np.concatenate([T[n][sel[n]] for n in names]); P_all = np.concatenate([E[n][sel[n]] for n in names])
    def q2(token_matrix):
        same, diff = [], []
        for k in range(2, L):
            C = centered_cosines(move_directions(P_all, k))
            t = token_matrix[:, k]
            eq = (t[:, None] == t[None, :]); iu = np.triu_indices(len(t), 1)
            same.append(C[iu][eq[iu]]); diff.append(C[iu][~eq[iu]])
        return np.concatenate(same), np.concatenate(diff)
    s, d_ = q2(toks)
    obs = s.mean() - d_.mean()
    null = []
    for _ in range(20):
        perm = np.stack([rng.permutation(toks[:, k]) for k in range(L)], axis=1)
        ss, dd = q2(perm); null.append(ss.mean() - dd.mean())
    res["q2"] = dict(n_same=int(len(s)), mean_cos_same=float(s.mean()), mean_cos_diff=float(d_.mean()),
                     diff=float(obs), null_mean=float(np.mean(null)), null_sd=float(np.std(null)))
    print("\nQ2 same-token vs different-token centered cosine:", res["q2"])

    # --- Q5 analogue: do patients of one group have closer paths than patients of different groups?
    paths = [p for n in names for p in E[n][sel[n]]]
    labels = np.array([n for n in names for _ in sel[n]])
    prep = [procrustes_prep(p) for p in paths]
    Cm = {}
    fns = {
        "frechet": lambda i, j: float(_frechet(cost_matrix(paths[i], paths[j]))),
        "dtw": lambda i, j: float(_dtw(cost_matrix(paths[i], paths[j]))),
        "procrustes": lambda i, j: procrustes_from_prep(prep[i], prep[j]),
    }
    res["paths"] = {}
    idx = list(range(len(paths)))
    for name, fn in fns.items():
        D = pairwise(idx, fn)
        obs, p, nm = perm_test(D, labels, 2000, rng)
        res["paths"][name] = dict(within_minus_between=obs, p_perm=p, null_mean=nm, group_means=group_means(D, labels, names))
        print(f"\n{name}: within-group minus between-group mean distance = {obs:+.3f} (null {nm:+.3f}), perm p={p:.4f}")
        for a in names:
            print("  ", a.ljust(22), " ".join(f"{res['paths'][name]['group_means'][f'{a}|{b}']:8.2f}" for b in names))
        res["paths"][name]["D"] = D.tolist() if len(idx) <= 400 else None
        if name == "dtw":
            Dd = D
    res["labels"] = labels.tolist(); res["names"] = names

    # --- symbolic poems: how central are they in their own group (mean DTW to the rest of the group)?
    res["symbolic"] = {}
    for n, keys in SYMBOLIC.items():
        if n not in names:
            continue
        gi = np.where(labels == n)[0]
        cent = np.array([Dd[i, gi[gi != i]].mean() for i in gi])
        lab = [Lab[n][j] for j in sel[n]]
        for i, l in zip(range(len(gi)), lab):
            if any(k.lower() in l.lower() for k in keys):
                pct = float((cent < cent[i]).mean())
                res["symbolic"][f"{n}:{l}"] = dict(mean_dtw_to_group=float(cent[i]), percentile_among_group=pct)
                print(f"  symbolic {n}:{l}: mean DTW to group {cent[i]:.1f}, centrality percentile {pct:.2f} (0=most central)")
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
