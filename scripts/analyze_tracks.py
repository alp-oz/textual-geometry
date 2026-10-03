"""All metrics on all tracks, plus the shuffle control.   python scripts/analyze_tracks.py TRACKS_DIR TRACKS_SHUF_DIR"""
import sys
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from textgeom.metrics import cost_matrix, _dtw, _frechet, procrustes_prep, procrustes_from_prep

D0, D1 = sys.argv[1], sys.argv[2]
A = pd.read_csv(f"{D0}/tracks.csv"); B = pd.read_csv(f"{D1}/tracks.csv")
TR = ["Qwen contextual", "XLM-R contextual", "Qwen first-token", "XLM-R first-token", "Qwen table", "XLM-R table"]
FAM = {
    "A. Same stories, 2 languages": ["Poe tales (English)", "Poe tales (French, Baudelaire)"],
    "B. French literature": ["Baudelaire poems", "Mallarmé poems", "Beckett L'Innommable", "Poe tales (French, Baudelaire)"],
    "C. English literature": ["Shakespeare sonnets", "Shakespeare plays", "Poe tales (English)"],
    "D. English math": ["Ramanujan proof (Andrews ch.5)", "Math ch.1 (elementary)", "Math ch.13 (combinatorics)"],
}
short = lambda t: t.replace(" contextual", " ctx").replace(" first-token", " 1st").replace(" table", " tbl")
rng = np.random.default_rng(0)

def eta(sub, c):
    tot = ((sub[c] - sub[c].mean()) ** 2).sum()
    return sum(len(s) * (s[c].mean() - sub[c].mean()) ** 2 for _, s in sub.groupby("group")) / tot

print("=" * 110, "\n1. GROUP MEANS PER TRACK (original text, order kept)")
for m, nm in [("dim", "Steele dimension (order-blind)"), ("spread", "spread = mean distance to the window's centre (order-blind)"), ("step", "step length = mean distance between consecutive tokens (ORDER-AWARE)")]:
    print(f"\n{nm}\n{'group':34s}" + "".join(f"{short(t):>10s}" for t in TR))
    for g in A.group.unique():
        s = A[A.group == g]; print(f"{g:34s}" + "".join(f"{s[f'{m}|{t}'].mean():10.3g}" for t in TR))
print("\nnumber of distinct tokens per window (Qwen tokens / XLM-R tokens):")
for g in A.group.unique():
    s = A[A.group == g]; print(f"  {g:34s} {s.n_distinct_qwen.mean():6.0f} / {s.n_distinct_xlmr.mean():6.0f}")
print("\nsurprise (Qwen only, nats/token):", {g: round(s.surprise_qwen.mean(), 2) for g, s in A.groupby("group", sort=False)})

print("\n", "=" * 110, "\n2. HOW WELL DOES EACH METRIC SEPARATE THE WORKS INSIDE A FAMILY?  (share of variance explained by group; 0 none, 1 all)")
for m in ("dim", "spread", "step"):
    print(f"\n{m}\n{'family':34s}" + "".join(f"{short(t):>10s}" for t in TR))
    for fam, gs in FAM.items():
        sub = A[A.group.isin(gs)]; print(f"{fam:34s}" + "".join(f"{eta(sub, f'{m}|{t}'):10.2f}" for t in TR))
print("\ndistinct-token count alone:", {fam: round(eta(A[A.group.isin(gs)], 'n_distinct_qwen'), 2) for fam, gs in FAM.items()})

print("\n", "=" * 110, "\n3. SHUFFLE CONTROL: words shuffled inside each window; mean change (shuffled - original) over paired windows, Wilcoxon p")
M = A.merge(B, on=["group", "label"], suffixes=("", "_s")); print(f"({len(M)} paired windows)")
print(f"\n{'metric':10s}" + "".join(f"{short(t):>22s}" for t in TR))
for m in ("dim", "spread", "step"):
    out = []
    for t in TR:
        d = (M[f"{m}|{t}_s"] - M[f"{m}|{t}"]).values
        try: p = wilcoxon(d).pvalue
        except ValueError: p = 1.0
        out.append(f"{d.mean():+10.3g} (p={p:.2g})")
    print(f"{m:10s}" + "".join(f"{o:>22s}" for o in out))

print("\n", "=" * 110, "\n4. PATH DISTANCES (order-aware): are windows of the same work closer than windows of other works of the family?")
print("   relative gap = (mean distance to other works - mean distance within the work) / mean distance to other works; permutation p")
S0 = np.load(f"{D0}/states.npz"); S1 = np.load(f"{D1}/states.npz")
def family_paths(S, track, gs):
    P = [w.astype(np.float64) for g in gs for w in S[f"{track}|{g}"]]
    return P, np.array([g for g in gs for _ in S[f"{track}|{g}"]])
def matrices(P, procrustes=True):
    n = len(P); out = {"DTW": np.zeros((n, n)), "Frechet": np.zeros((n, n)), "Procrustes": np.zeros((n, n))}
    prep = [procrustes_prep(p) for p in P] if procrustes else None
    for i, j in combinations(range(n), 2):
        C = cost_matrix(P[i], P[j]); out["DTW"][i, j] = out["DTW"][j, i] = _dtw(C); out["Frechet"][i, j] = out["Frechet"][j, i] = _frechet(C)
        if procrustes: out["Procrustes"][i, j] = out["Procrustes"][j, i] = procrustes_from_prep(prep[i], prep[j])
    return out
def gap(Dm, lab, perms=300):
    iu = np.triu_indices(len(lab), 1)
    def stat(l):
        s = (l[:, None] == l[None, :])[iu]; return (Dm[iu][~s].mean() - Dm[iu][s].mean()) / Dm[iu][~s].mean()
    obs = stat(lab); null = [stat(rng.permutation(lab)) for _ in range(perms)]
    return obs, (np.array(null) >= obs).mean()
for fam in ("B. French literature", "C. English literature", "D. English math"):
    gs = FAM[fam]; print(f"\n{fam}\n{'track':22s}{'metric':12s}{'original order':>22s}{'words shuffled':>22s}")
    for t in TR:
        P0, lab = family_paths(S0, t, gs); P1, lab1 = family_paths(S1, t, gs)
        m0, m1 = matrices(P0), matrices(P1, procrustes=False)
        for k in ("DTW", "Frechet", "Procrustes"):
            o, p = gap(m0[k], lab)
            s = "" if k == "Procrustes" else "%+.3f (p=%.3f)" % gap(m1[k], lab1)
            print(f"{t:22s}{k:12s}{'%+.3f (p=%.3f)' % (o, p):>22s}{s:>22s}")
