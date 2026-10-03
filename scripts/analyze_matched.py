"""Compare the two models on identical windows (matched.csv)."""
import sys
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

df = pd.read_csv(sys.argv[1])
FAM = {
    "A. Same stories, two languages": ["Poe tales (English)", "Poe tales (French, Baudelaire)"],
    "B. French literature": ["Baudelaire poems", "Mallarmé poems", "Beckett L'Innommable", "Poe tales (French, Baudelaire)"],
    "C. English literature": ["Shakespeare sonnets", "Shakespeare plays", "Poe tales (English)"],
    "D. English math": ["Ramanujan proof (Andrews ch.5)", "Math ch.1 (elementary)", "Math ch.13 (combinatorics)"],
}
rng = np.random.default_rng(0)
def ci(x): return np.percentile([rng.choice(x, len(x)).mean() for _ in range(1000)], [2.5, 97.5])

print(f"N per group = {df.groupby('group').size().min()}..{df.groupby('group').size().max()}; every window has the same number of points under both models\n")
print(f"{'group':34s} {'dim Qwen (running state)':>26s} {'dim XLM-R (whole chunk)':>26s} {'within-group sd Qwen':>21s} {'sd XLM-R':>9s}")
for g in df.group.unique():
    s = df[df.group == g]
    q, x = s.dim_qwen.values, s.dim_xlmr.values
    print(f"{g:34s} {q.mean():8.1f} ±{(ci(q)[1]-ci(q)[0])/2:4.1f}{'':>12s} {x.mean():8.1f} ±{(ci(x)[1]-ci(x)[0])/2:4.1f}{'':>11s} {q.std():10.1f} {x.std():14.1f}")

print("\nHOW WELL DOES EACH MODEL SEPARATE THE GROUPS INSIDE A FAMILY?  (share of variance explained by group; 0 = none, 1 = all)")
for fam, gs in FAM.items():
    sub = df[df.group.isin(gs)]
    out = []
    for c in ("dim_qwen", "dim_xlmr"):
        tot = ((sub[c] - sub[c].mean()) ** 2).sum()
        bet = sum(len(s) * (s[c].mean() - sub[c].mean()) ** 2 for _, s in sub.groupby("group"))
        out.append(f"{c}: {bet / tot:.2f}")
    print(f"  {fam:34s} " + "   ".join(out))

print("\nDO THE TWO MODELS AGREE?")
rho = spearmanr(df.dim_qwen, df.dim_xlmr)
print(f"  window by window (all windows): rank correlation {rho.correlation:+.2f} (p={rho.pvalue:.2g})")
for fam, gs in FAM.items():
    sub = df[df.group.isin(gs)]
    if len(gs) >= 3:
        m = sub.groupby("group")[["dim_qwen", "dim_xlmr"]].mean()
        print(f"  {fam}: ordering of group means  Qwen {list(m.dim_qwen.sort_values(ascending=False).index)}\n{'':32s}XLM-R {list(m.dim_xlmr.sort_values(ascending=False).index)}")
print("\nPAIRWISE GROUP DIFFERENCES INSIDE FAMILIES (difference of means; Mann-Whitney p)")
for fam, gs in FAM.items():
    for a, b in combinations(gs, 2):
        A, B = df[df.group == a], df[df.group == b]
        qa = f"Qwen {A.dim_qwen.mean()-B.dim_qwen.mean():+.2f} (p={mannwhitneyu(A.dim_qwen, B.dim_qwen).pvalue:.2g})"
        xa = f"XLM-R {A.dim_xlmr.mean()-B.dim_xlmr.mean():+.2f} (p={mannwhitneyu(A.dim_xlmr, B.dim_xlmr).pvalue:.2g})"
        print(f"  {a} vs {b}: {qa}; {xa}")
