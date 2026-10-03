"""Analyse the matched-window run. Comparisons are made WITHIN families only."""
import sys
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from textgeom.metrics import cost_matrix, _dtw, _frechet, procrustes_prep, procrustes_from_prep

d = sys.argv[1]
df = pd.read_csv(f"{d}/windows.csv")
Z = np.load(f"{d}/states.npz")
FAM = {
    "A. Same stories, two languages": ["Poe tales (English)", "Poe tales (French, Baudelaire)"],
    "B. French literature": ["Baudelaire poems", "Mallarmé poems", "Beckett L'Innommable", "Poe tales (French, Baudelaire)"],
    "C. English literature": ["Shakespeare sonnets", "Shakespeare plays", "Poe tales (English)"],
    "D. English math": ["Ramanujan proof (Andrews ch.5)", "Math ch.1 (elementary)", "Math ch.13 (combinatorics)"],
}
rng = np.random.default_rng(0)

def ci(x):
    b = [rng.choice(x, len(x)).mean() for _ in range(1000)]
    return np.percentile(b, [2.5, 97.5])

print("EVERY GROUP: 60 windows x 128 tokens\n")
print(f"{'group':34s} {'n':>3s} {'dimension':>16s} {'step length':>16s} {'spread':>15s} {'surprise':>15s}")
for g in df.group.unique():
    s = df[df.group == g]
    f = lambda c: f"{s[c].mean():6.1f} ±{(ci(s[c].values)[1]-ci(s[c].values)[0])/2:4.1f}" if c != "surprise" else f"{s[c].mean():6.2f} ±{(ci(s[c].values)[1]-ci(s[c].values)[0])/2:4.2f}"
    print(f"{g:34s} {len(s):3d} {f('dim'):>16s} {f('mean_step'):>16s} {f('spread'):>15s} {f('surprise'):>15s}")

print("\nWITHIN-FAMILY TESTS (Mann-Whitney p; effect = difference of group means)")
for fam, gs in FAM.items():
    print("\n" + fam)
    for a, b in combinations(gs, 2):
        A, B = df[df.group == a], df[df.group == b]
        out = []
        for c, nm in [("dim", "dimension"), ("mean_step", "step"), ("surprise", "surprise")]:
            p = mannwhitneyu(A[c], B[c]).pvalue
            out.append(f"{nm} {A[c].mean()-B[c].mean():+.2f} (p={p:.2g})")
        print(f"  {a} vs {b}: " + "; ".join(out))

print("\nCENTRE DISTANCES (average of all token points of a group) - same family only")
cen = {g: Z[g].astype(np.float64).reshape(-1, Z[g].shape[-1]).mean(0) for g in Z.files}
for fam, gs in FAM.items():
    print("\n" + fam)
    for a, b in combinations(gs, 2):
        print(f"  {a} <-> {b}: {np.linalg.norm(cen[a]-cen[b]):.0f}")

print("\nPATH DISTANCES between windows (DTW, Frechet, Procrustes); mean within the same group vs mean to windows of the other group(s) of the family")
def mats(gs):
    P = [w.astype(np.float64) for g in gs for w in Z[g]]
    lab = np.array([g for g in gs for _ in Z[g]])
    prep = [procrustes_prep(p) for p in P]
    n = len(P); out = {k: np.zeros((n, n)) for k in ("dtw", "frechet", "procrustes")}
    for i, j in combinations(range(n), 2):
        C = cost_matrix(P[i], P[j])
        out["dtw"][i, j] = out["dtw"][j, i] = _dtw(C)
        out["frechet"][i, j] = out["frechet"][j, i] = _frechet(C)
        out["procrustes"][i, j] = out["procrustes"][j, i] = procrustes_from_prep(prep[i], prep[j])
    return lab, out
for fam, gs in FAM.items():
    if fam.startswith("A"):
        continue
    lab, M = mats(gs)
    print("\n" + fam)
    for k, D in M.items():
        iu = np.triu_indices(len(lab), 1); same = (lab[:, None] == lab[None, :])[iu]
        obs = D[iu][same].mean() - D[iu][~same].mean()
        null = []
        for _ in range(1000):
            pl = rng.permutation(lab); s2 = (pl[:, None] == pl[None, :])[iu]; null.append(D[iu][s2].mean() - D[iu][~s2].mean())
        p = (np.array(null) <= obs).mean()
        print(f"  {k:10s} within - between = {obs:+.3g}  (shuffle p = {p:.3f}; negative = same-work windows are closer)")

# language control: do translated windows of the same passage lie closer than mismatched passages?
en, fr = Z["Poe tales (English)"].astype(np.float64), Z["Poe tales (French, Baudelaire)"].astype(np.float64)
pair = df[df.group == "Poe tales (English)"].pair.values
dt = lambda a, b: _dtw(cost_matrix(a, b))
same = np.array([dt(en[i], fr[i]) for i in range(len(en))])
diff = np.array([dt(en[i], fr[j]) for i in range(len(en)) for j in range(len(fr)) if i != j])
print(f"\nLANGUAGE CONTROL: DTW between a window and the translation of the SAME passage: {same.mean():.0f}; "
      f"to translations of OTHER passages: {diff.mean():.0f}  (Mann-Whitney p={mannwhitneyu(same, diff).pvalue:.2g})")
