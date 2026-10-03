"""Hide which work a window came from; how often can a measure guess it? (leave-one-out, inside each family)."""
import sys
from collections import Counter
import numpy as np
import pandas as pd
from textgeom.metrics import cost_matrix, _dtw

D0 = sys.argv[1]
df = pd.read_csv(f"{D0}/tracks.csv"); Z = np.load(f"{D0}/states.npz")
FAM = {
    "French literature": ["Baudelaire poems", "Mallarmé poems", "Beckett L'Innommable", "Poe tales (French, Baudelaire)"],
    "English literature": ["Shakespeare sonnets", "Shakespeare plays", "Poe tales (English)"],
    "English math": ["Ramanujan proof (Andrews ch.5)", "Math ch.1 (elementary)", "Math ch.13 (combinatorics)"],
}
TR = {"Qwen (reads left to right)": "Qwen contextual", "XLM-R (reads whole chunk)": "XLM-R contextual",
      "Qwen input table (no model)": "Qwen table", "XLM-R input table (no model)": "XLM-R table"}

def centroid_guess(F, y):
    F = (F - F.mean(0)) / (F.std(0) + 1e-12); pred = []
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        cents = {c: F[keep & (y == c)].mean(0) for c in np.unique(y)}
        pred.append(min(cents, key=lambda c: np.linalg.norm(F[i] - cents[c])))
    return np.array(pred)

def acc(pred, y): return float((pred == y).mean())

res = {}   # (what, family) -> accuracy
recall = {}
for fam, gs in FAM.items():
    sub = df[df.group.isin(gs)].reset_index(drop=True); y = sub.group.values
    for nm, tr in TR.items():
        for ms, cols in {"dimension": [f"dim|{tr}"], "spread": [f"spread|{tr}"], "step length": [f"step|{tr}"],
                         "all three together": [f"dim|{tr}", f"spread|{tr}", f"step|{tr}"]}.items():
            res[(nm, ms, fam)] = acc(centroid_guess(sub[cols].values, y), y)
    res[("(no model needed)", "count of different words", fam)] = acc(centroid_guess(sub[["n_distinct_qwen"]].values, y), y)
    res[("Qwen", "surprise (how predictable)", fam)] = acc(centroid_guess(sub[["surprise_qwen"]].values, y), y)
    # distance between whole paths (DTW), nearest neighbour
    for nm, tr in TR.items():
        P = [w.astype(np.float64) for g in gs for w in Z[f"{tr}|{g}"]]; n = len(P)
        Dm = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n): Dm[i, j] = Dm[j, i] = _dtw(cost_matrix(P[i], P[j]))
        np.fill_diagonal(Dm, np.inf); lab = np.array([g for g in gs for _ in Z[f"{tr}|{g}"]])
        res[(nm, "path distance (DTW)", fam)] = acc(lab[Dm.argmin(1)], lab)
    # joint: everything from both models together -> per-work recall and confusions
    cols = [f"{m}|{t}" for t in ("Qwen contextual", "XLM-R contextual") for m in ("dim", "spread", "step")] + ["surprise_qwen", "n_distinct_qwen"]
    pred = centroid_guess(sub[cols].values, y); recall[fam] = {g: float((pred[y == g] == g).mean()) for g in gs}
    recall[fam]["_all"] = acc(pred, y); recall[fam]["_conf"] = Counter((a, b) for a, b in zip(y, pred) if a != b).most_common(3)

chance = {f: 1 / len(g) for f, g in FAM.items()}
print("CHANCE LEVEL (random guessing):", {f: f"{c:.0%}" for f, c in chance.items()})
fams = list(FAM)
def table(title, keys):
    print(f"\n{title}\n{'':44s}" + "".join(f"{f:>20s}" for f in fams) + f"{'average':>10s}")
    for (a, b) in keys:
        v = [res[(a, b, f)] for f in fams]; print(f"{a + ' - ' + b:44s}" + "".join(f"{x:20.0%}" for x in v) + f"{np.mean(v):10.0%}")
print("\n1) WHICH MODEL? (all three measures together: dimension + spread + step length)")
table("", [(nm, "all three together") for nm in TR])
print("\n2) WHICH MEASURE?")
for ms in ("dimension", "spread", "step length", "path distance (DTW)"):
    table(ms, [(nm, ms) for nm in TR])
table("measures that need no model reading", [("(no model needed)", "count of different words"), ("Qwen", "surprise (how predictable)")])
print("\n3) WHICH WORKS ARE EASIEST TO RECOGNISE? (all measures from both models together)")
for fam in fams:
    r = recall[fam]; print(f"\n{fam}: {r['_all']:.0%} correct overall (chance {chance[fam]:.0%})")
    for g, v in r.items():
        if not g.startswith("_"): print(f"   {g:34s} recognised {v:.0%} of the time")
    print("   most common mix-ups (real work -> guessed work):", [(a.split(' (')[0], b.split(' (')[0], c) for (a, b), c in r["_conf"]])
