"""Proof family (7 works incl. 'Your paper'), raw and prose-only versions.   python scripts/analyze_proofs.py DIR [DIR ...]"""
import sys
from collections import Counter
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from textgeom.metrics import cost_matrix, _dtw

TR = {"Qwen": "Qwen contextual", "XLM-R": "XLM-R contextual", "Qwen table": "Qwen table", "XLM-R table": "XLM-R table"}
HR, YOU = "Ramanujan 1918 (Hardy-Ramanujan paper)", "Your paper"
rng = np.random.default_rng(0)

def centroid_guess(F, y):
    F = (F - F.mean(0)) / (F.std(0) + 1e-12); pred = []
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        c = {k: F[keep & (y == k)].mean(0) for k in np.unique(y)}
        pred.append(min(c, key=lambda k: np.linalg.norm(F[i] - c[k])))
    return np.array(pred)
acc = lambda p, y: float((p == y).mean())

for D in sys.argv[1:]:
    df = pd.read_csv(f"{D}/proofs.csv"); Z = np.load(f"{D}/states.npz"); gs = list(df.group.unique()); y = df.group.values
    n = df.groupby("group").size().iloc[0]
    print("=" * 110, f"\nVERSION: {D.split('/')[-1]}   ({n} passages of 256 tokens per work, {len(gs)} works, chance = {1/len(gs):.0%})\n")
    print(f"{'':56s}{'different':>10s}{'surprise':>9s}{'dimension':>20s}{'step length':>22s}\n{'':56s}{'words':>10s}{'(Qwen)':>9s}{'Qwen':>10s}{'XLM-R':>10s}{'Qwen':>11s}{'XLM-R':>11s}")
    for g in gs:
        s = df[df.group == g]
        print(f"{g:56s}{s.n_distinct_qwen.mean():10.0f}{s.surprise_qwen.mean():9.2f}{s['dim|Qwen contextual'].mean():10.1f}{s['dim|XLM-R contextual'].mean():10.1f}{s['step|Qwen contextual'].mean():11.0f}{s['step|XLM-R contextual'].mean():11.2f}")
    # B. guess the work
    res = {}
    for nm, tr in TR.items():
        for ms, cs in {"dimension": [f"dim|{tr}"], "spread": [f"spread|{tr}"], "step length": [f"step|{tr}"], "all three": [f"dim|{tr}", f"spread|{tr}", f"step|{tr}"]}.items():
            res[(nm, ms)] = acc(centroid_guess(df[cs].values, y), y)
        P = [w.astype(np.float64) for g in gs for w in Z[f"{tr}|{g}"]]; lab = np.array([g for g in gs for _ in Z[f"{tr}|{g}"]]); m = len(P)
        Dm = np.full((m, m), np.inf)
        for i, j in combinations(range(m), 2): Dm[i, j] = Dm[j, i] = _dtw(cost_matrix(P[i], P[j]))
        res[(nm, "path distance (DTW)")] = acc(lab[Dm.argmin(1)], lab)
        res[(nm, "_Dm")] = (Dm, lab)
    res[("-", "count of different words")] = acc(centroid_guess(df[["n_distinct_qwen"]].values, y), y)
    res[("Qwen", "surprise")] = acc(centroid_guess(df[["surprise_qwen"]].values, y), y)
    print("\nHOW OFTEN CAN WE GUESS THE WORK FROM A HIDDEN PASSAGE?")
    for ms in ("all three", "dimension", "spread", "step length", "path distance (DTW)"):
        print(f"  {ms:22s}" + "".join(f"  {nm}: {res[(nm, ms)]:4.0%}" for nm in TR))
    print(f"  {'different words':22s}  {res[('-', 'count of different words')]:4.0%}      surprise (Qwen): {res[('Qwen', 'surprise')]:4.0%}")
    allc = [f"{m}|{t}" for t in ("Qwen contextual", "XLM-R contextual") for m in ("dim", "spread", "step")] + ["surprise_qwen", "n_distinct_qwen"]
    pred = centroid_guess(df[allc].values, y)
    print(f"\nWHICH WORKS ARE RECOGNISED (all measures together: {acc(pred, y):.0%} right overall)")
    for g in gs: print(f"   {g:56s} {(pred[y == g] == g).mean():4.0%}")
    print("   most common mix-ups:", [(a.split(' (')[0], b.split(' (')[0], c) for (a, b), c in Counter((a, b) for a, b in zip(y, pred) if a != b).most_common(4)])
    # D. Ramanujan 1918 vs the others
    print("\nRAMANUJAN 1918 vs EACH OTHER PROOF (difference of means; Mann-Whitney p):")
    for g in gs:
        if g == HR: continue
        a, b = df[df.group == HR], df[df.group == g]; out = []
        for c, nm in [("dim|Qwen contextual", "dim Qwen"), ("dim|XLM-R contextual", "dim XLM-R"), ("step|Qwen contextual", "step"), ("surprise_qwen", "surprise"), ("n_distinct_qwen", "diff. words")]:
            out.append(f"{nm} {a[c].mean() - b[c].mean():+.2f} (p={mannwhitneyu(a[c], b[c]).pvalue:.2g})")
        print(f"   vs {g.split(' (')[0]:22s}" + "; ".join(out))
    # E. where does 'Your paper' fall? mean DTW from your passages to each other work (Qwen and XLM-R)
    print("\nWHICH WORKS IS 'YOUR PAPER' CLOSEST TO? (mean path distance DTW from your passages to the passages of each work; smaller = closer)")
    for nm in ("Qwen", "XLM-R"):
        Dm, lab = res[(nm, "_Dm")]; mine = np.where(lab == YOU)[0]
        d = {g: float(np.mean([Dm[i, j] for i in mine for j in np.where(lab == g)[0]])) for g in gs if g != YOU}
        print(f"   {nm:6s}: " + " < ".join(f"{g.split(' (')[0]} {v:,.0f}" for g, v in sorted(d.items(), key=lambda kv: kv[1])))
