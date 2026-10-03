"""Analyse lone.csv from run_lone.py: table vs lone-token output vs contextual vs shuffled-contextual."""
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

df = pd.read_csv(sys.argv[1])
diag = df[df.group == "__diagnostic__"]
df = df[df.group != "__diagnostic__"]
KINDS = ["table", "lone", "lone_bare", "context", "shuffled"]
FAM = {
    "A. Same stories, two languages": ["Poe tales (English)", "Poe tales (French, Baudelaire)"],
    "B. French literature": ["Baudelaire poems", "Mallarmé poems", "Poe tales (French, Baudelaire)"],
    "C. English literature": ["Shakespeare sonnets", "Shakespeare plays", "Poe tales (English)"],
}


def eta(sub, c):
    sub = sub.dropna(subset=[c]); tot = ((sub[c] - sub[c].mean()) ** 2).sum()
    return sum(len(s) * (s[c].mean() - sub[c].mean()) ** 2 for _, s in sub.groupby("group")) / tot


print("COLLAPSE CHECK for lone-token vectors (all distinct tokens seen in the windows)")
print("  mean_cos = average cosine between two different tokens' vectors (1 = all alike);")
print("  partic_ratio = effective number of directions the vectors use")
print(diag[["model", "label", "n_distinct_tokens", "mean_norm", "sd_norm", "mean_cos", "partic_ratio"]].round(3).to_string(index=False))

for model, T in df.groupby("model", sort=False):
    print("\n" + "=" * 110)
    print(f"{model}   (dimension, mean ± sd over windows; {T.groupby('group').size().min()}+ windows per group, 128 points each)\n")
    print(f"{'group':34s}" + "".join(f"{k:>14s}" for k in KINDS) + f"{'distinct tok':>14s}")
    for g in T.group.unique():
        s = T[T.group == g]
        print(f"{g:34s}" + "".join(f"{s['dim_' + k].mean():8.1f} ±{s['dim_' + k].std():4.1f}" for k in KINDS) + f"{s.n_distinct_tokens.mean():14.0f}")
    print("\n  same, dimension of DISTINCT tokens only (removes duplicate points)")
    for g in T.group.unique():
        s = T[T.group == g]
        print(f"  {g:34s}" + "".join(f"{s['dimdistinct_' + k].mean():8.1f}      " for k in KINDS))
    print("\nSHARE OF VARIANCE EXPLAINED BY GROUP INSIDE EACH FAMILY (0 = none, 1 = all)")
    print(f"  {'family':34s}" + "".join(f"{k:>11s}" for k in KINDS))
    for fam, gs in FAM.items():
        sub = T[T.group.isin(gs)]
        print(f"  {fam:34s}" + "".join(f"{eta(sub, 'dim_' + k):11.2f}" for k in KINDS))
    print("\nDO THE VERSIONS AGREE? window-by-window rank correlation of dimension")
    print(f"  {'':12s}" + "".join(f"{k:>11s}" for k in KINDS))
    for a in KINDS:
        print(f"  {a:12s}" + "".join(f"{spearmanr(T['dim_' + a], T['dim_' + b]).correlation:+11.2f}" for b in KINDS))
    r = spearmanr(T.dim_table, T.n_distinct_tokens).correlation
    r2 = spearmanr(T.dim_context, T.n_distinct_tokens).correlation
    print(f"\n  confound: corr(dimension, number of distinct tokens)   table {r:+.2f}   context {r2:+.2f}")
    print("\nORDER OF GROUP MEANS (high -> low)")
    for k in KINDS:
        m = T.groupby("group")["dim_" + k].mean().sort_values(ascending=False)
        print(f"  {k:10s}", " > ".join(f"{g.split(' (')[0]} {v:.1f}" for g, v in m.items()))
    print("\nSHUFFLE EFFECT on contextual dimension (context minus shuffled, per group)")
    for g in T.group.unique():
        s = T[T.group == g]
        print(f"  {g:34s} {(s.dim_context - s.dim_shuffled).mean():+.2f}")
