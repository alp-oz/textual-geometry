"""Analyse the fixed-vector run (static.csv): groups, families, and the distinct-token confound."""
import sys
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

df = pd.read_csv(sys.argv[1])
FAM = {
    "A. Same stories, two languages": ["Poe tales (English)", "Poe tales (French, Baudelaire)"],
    "B. French literature": ["Baudelaire poems", "Mallarmé poems", "Beckett L'Innommable", "Poe tales (French, Baudelaire)"],
    "C. English literature": ["Shakespeare sonnets", "Shakespeare plays", "Poe tales (English)"],
    "D. English math": ["Ramanujan proof (Andrews ch.5)", "Math ch.1 (elementary)", "Math ch.13 (combinatorics)"],
}
rng = np.random.default_rng(0)
def hw(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    b = [rng.choice(x, len(x)).mean() for _ in range(500)]; return (np.percentile(b, 97.5) - np.percentile(b, 2.5)) / 2
def eta(sub, c):
    sub = sub.dropna(subset=[c]); tot = ((sub[c] - sub[c].mean()) ** 2).sum()
    return sum(len(s) * (s[c].mean() - sub[c].mean()) ** 2 for _, s in sub.groupby("group")) / tot

for table, T in df.groupby("table"):
    print("=" * 100); print(f"FIXED VECTORS FROM THE {table.upper()}  (every window = exactly 256 tokens, no context)\n")
    print(f"{'group':34s} {'n':>3s} {'distinct tokens':>16s} {'dimension (all 256)':>20s} {'dimension (distinct only)':>26s} {'spread':>14s}")
    for g in df.group.unique():
        s = T[T.group == g]
        if len(s) == 0: continue
        print(f"{g:34s} {len(s):3d} {s.n_distinct.mean():10.0f} ±{hw(s.n_distinct):3.0f} {s.dim_occ.mean():9.1f} ±{hw(s.dim_occ):4.1f} {s.dim_distinct.mean():14.1f} ±{hw(s.dim_distinct):4.1f} {s.spread.mean():8.3f} ±{hw(s.spread):.3f}")
    print("\nSHARE OF VARIANCE EXPLAINED BY GROUP INSIDE EACH FAMILY (0 = none, 1 = all)")
    print(f"  {'family':34s} {'dim(all 256)':>13s} {'dim(distinct)':>14s} {'spread':>8s} {'distinct tokens':>16s}")
    for fam, gs in FAM.items():
        sub = T[T.group.isin(gs)]
        print(f"  {fam:34s} {eta(sub,'dim_occ'):13.2f} {eta(sub,'dim_distinct'):14.2f} {eta(sub,'spread'):8.2f} {eta(sub,'n_distinct'):16.2f}")
    r = spearmanr(T.dim_occ, T.n_distinct, nan_policy="omit")
    print(f"\n  CONFOUND CHECK: rank correlation between a window's dimension (all 256) and its number of distinct tokens: {r.correlation:+.2f}")
    print("\nPAIRWISE INSIDE FAMILIES (difference of mean dimension(all 256); Mann-Whitney p)")
    for fam, gs in FAM.items():
        for a, b in combinations(gs, 2):
            A, B = T[T.group == a], T[T.group == b]
            if len(A) and len(B):
                print(f"  {a} vs {b}: {A.dim_occ.mean()-B.dim_occ.mean():+.2f} (p={mannwhitneyu(A.dim_occ, B.dim_occ).pvalue:.2g})")
