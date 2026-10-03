"""Stricter 'which work is this passage from?' test.
Every work is split by position in the text. The guesser learns from passages of one half of a work and must guess
passages of the OTHER half (both directions); passages within 256 tokens of the split are dropped, so no test passage
overlaps a learning passage. Poe's tales are split by tale (tales 0-3 against tales 4-6), so the learner never sees the
tale it is tested on. Compared with the earlier leave-one-out test, where neighbours from the same stretch of text helped.
    python scripts/honest_test.py SCRATCH_DIR > examples/honest_test.txt"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from textgeom.metrics import cost_matrix, _dtw

S = sys.argv[1]
GAP = 256
TR = {"Qwen": "Qwen contextual", "XLM-R": "XLM-R contextual"}
HR = "Ramanujan 1918 (Hardy-Ramanujan paper)"
FAM = {
    "French literature": ["Baudelaire poems", "Mallarmé poems", "Beckett L'Innommable", "Poe tales (French, Baudelaire)", "Rimbaud poems"],
    "English literature": ["Shakespeare sonnets", "Shakespeare plays", "Poe tales (English)", "Joyce Ulysses"],
    "Proofs": [HR, "Andrews ch.5 (modern retelling of the same proof)", "Andrews ch.1 (simple modern proof)",
               "Euclid (ancient)", "Hilbert (1899)", "Dedekind (1888)", "Your paper"],
}
FEATS = {
    "surprise": ["surprise_qwen"],
    "step length (Qwen)": ["step|Qwen contextual"],
    "dimension (Qwen)": ["dim|Qwen contextual"],
    "dimension (XLM-R)": ["dim|XLM-R contextual"],
    "different words": ["n_distinct_qwen"],
    "all measures together": [f"{m}|{t}" for t in TR.values() for m in ("dim", "spread", "step")] + ["surprise_qwen", "n_distinct_qwen"],
}

lit = pd.concat([pd.read_csv(f"{S}/{d}/tracks.csv") for d in ("tracks", "new")], ignore_index=True)
pro = pd.read_csv(f"{S}/proofs_raw/proofs.csv")
Z = {d: np.load(f"{S}/{d}/states.npz") for d in ("tracks", "new", "proofs_raw")}


def states(tr, g):
    for d in ("proofs_raw", "new", "tracks"):
        if f"{tr}|{g}" in Z[d].files:
            return Z[d][f"{tr}|{g}"]


def halves(sub):
    """0 = first half of the work, 1 = second half, -1 = dropped (too close to the split)."""
    h = np.full(len(sub), -1)
    for g in sub.group.unique():
        m = (sub.group == g).values
        lab = sub.label[m]
        if g.startswith("Poe tales"):
            tale = lab.str.extract(r"tale(\d+)@")[0].astype(int).values
            h[m] = (tale >= 4).astype(int)
        else:
            pos = lab.str.split("@").str[-1].astype(float).values
            med = np.median(pos)
            h[m] = np.where(pos < med - GAP, 0, np.where(pos > med + GAP, 1, -1))
    return h


def centroid(Ftr, ytr, Fte):
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-12
    Ftr, Fte = (Ftr - mu) / sd, (Fte - mu) / sd
    cents = {c: Ftr[ytr == c].mean(0) for c in np.unique(ytr)}
    return np.array([min(cents, key=lambda c: np.linalg.norm(f - cents[c])) for f in Fte])


def loo_centroid(F, y):
    F = (F - F.mean(0)) / (F.std(0) + 1e-12)
    out = []
    for i in range(len(y)):
        k = np.arange(len(y)) != i
        c = {a: F[k & (y == a)].mean(0) for a in np.unique(y)}
        out.append(min(c, key=lambda a: np.linalg.norm(F[i] - c[a])))
    return np.array(out)


rows = []
for fam, gs in FAM.items():
    sub = (pro if fam == "Proofs" else lit)
    sub = sub[sub.group.isin(gs)].reset_index(drop=True)
    h = halves(sub)
    keep = h >= 0
    sub, h = sub[keep].reset_index(drop=True), h[keep]
    y = sub.group.values
    # states in the same row order as sub (windows are stored per group in csv order)
    csv = (pro if fam == "Proofs" else lit)
    csv = csv[csv.group.isin(gs)].reset_index(drop=True)
    idx_in_group = csv.groupby("group").cumcount().values
    kept_idx = np.where(keep)[0]
    paths = {nm: [states(tr, csv.group[i])[idx_in_group[i]].astype(np.float64) for i in kept_idx] for nm, tr in TR.items()}
    k = len(gs)
    counts = ", ".join(g.split(" (")[0] + " " + str(int((y == g).sum())) for g in gs)
    print(f"\n{fam}: {len(y)} passages kept ({counts}); chance {1 / k:.0%}", flush=True)
    for nm, cols in FEATS.items():
        F = sub[cols].values.astype(float)
        correct = loo = 0
        for d in (0, 1):
            tr_, te_ = h == d, h == 1 - d
            correct += int((centroid(F[tr_], y[tr_], F[te_]) == y[te_]).sum())
        loo = int((loo_centroid(F, y) == y).sum())
        rows.append((fam, nm, correct, len(y), loo, k))
    for nm in TR:
        P = paths[nm]
        correct = 0
        for d in (0, 1):
            tr_ = np.where(h == d)[0]
            for i in np.where(h == 1 - d)[0]:
                dist = [_dtw(cost_matrix(P[i], P[j])) for j in tr_]
                correct += int(y[tr_[int(np.argmin(dist))]] == y[i])
        n = len(y)
        D = np.full((n, n), np.inf)
        for i in range(n):
            for j in range(i + 1, n):
                D[i, j] = D[j, i] = _dtw(cost_matrix(P[i], P[j]))
        loo = int((y[D.argmin(1)] == y).sum())
        rows.append((fam, f"path distance ({nm})", correct, n, loo, k))

res = pd.DataFrame(rows, columns=["family", "measure", "right", "n", "right_loo", "k"])
res["acc"] = res.right / res.n
res["acc_loo"] = res.right_loo / res.n
res["chance"] = 1 / res.k
res["p"] = [binomtest(r, n, 1 / k, alternative="greater").pvalue for r, n, k in zip(res.right, res.n, res.k)]
ci = [binomtest(int(r), int(n)).proportion_ci(confidence_level=0.95, method="wilson") for r, n in zip(res.right, res.n)]
res["lo"], res["hi"] = [c.low for c in ci], [c.high for c in ci]
# Holm correction over all tests
order = res.p.sort_values().index
m = len(res)
holm = {}
run = 0.0
for rank, i in enumerate(order):
    run = max(run, min(1.0, res.p[i] * (m - rank)))
    holm[i] = run
res["p_holm"] = pd.Series(holm)
res.to_csv(f"{S}/honest_test.csv", index=False)

print("\nSTRICT TEST: learn on one half of each work, guess passages of the other half (both directions)")
print("acc = share guessed right; 95% range is the uncertainty from the small number of passages; p = against random guessing,")
print("corrected for the %d tests run (Holm). 'old' = the earlier leave-one-out accuracy on the same passages.\n" % m)
for fam in FAM:
    r = res[res.family == fam]
    print(f"{fam}  (random guessing {r.chance.iloc[0]:.0%}, {r.n.iloc[0]} passages tested)")
    print(f"  {'measure':26s}{'strict':>8s}{'95% range':>14s}{'old':>7s}{'p (corrected)':>16s}")
    for _, x in r.sort_values("acc", ascending=False).iterrows():
        print(f"  {x.measure:26s}{x.acc:8.0%}{f'{x.lo:.0%}-{x.hi:.0%}':>14s}{x.acc_loo:7.0%}{x.p_holm:16.3g}")
    print()
