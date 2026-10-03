"""Final comparison on the agreed families (no first-token track).
   python scripts/analyze_final.py OLD_TRACKS_DIR NEW_DIR NEW_SHUF_DIR [OLD_SHUF_DIR]"""
import sys
from collections import Counter
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon
from textgeom.metrics import cost_matrix, _dtw

OLD, NEW, NEWS = sys.argv[1], sys.argv[2], sys.argv[3]
OLDS = sys.argv[4] if len(sys.argv) > 4 else None
EU2, EU2S = (sys.argv[5], sys.argv[6]) if len(sys.argv) > 6 else (None, None)   # re-run of Euclid on the cleaned text overrides the old Euclid rows
TR = {"Qwen": "Qwen contextual", "XLM-R": "XLM-R contextual", "Qwen table": "Qwen table", "XLM-R table": "XLM-R table"}
HR, A5, A1, EU = "Ramanujan 1918 (Hardy-Ramanujan paper)", "Ramanujan proof (Andrews ch.5)", "Math ch.1 (elementary)", "Euclid (ancient proofs)"
FAM = {
    "French literature": ["Baudelaire poems", "Mallarmé poems", "Beckett L'Innommable", "Poe tales (French, Baudelaire)", "Rimbaud poems"],
    "English literature": ["Shakespeare sonnets", "Shakespeare plays", "Poe tales (English)", "Joyce Ulysses"],
    "Proofs (math)": [HR, A5, A1, EU],
}
NICE = {A5: "Andrews ch.5 (modern retelling of the same proof)", A1: "Andrews ch.1 (simple modern proof)", HR: "Ramanujan 1918 (original)", EU: "Euclid (ancient)"}
cols = ["group", "label", "n_distinct_qwen", "surprise_qwen"] + [f"{m}|{t}" for t in TR.values() for m in ("dim", "spread", "step")]
df = pd.concat([pd.read_csv(f"{OLD}/tracks.csv")[cols], pd.read_csv(f"{NEW}/tracks.csv")[cols]], ignore_index=True)
if EU2:
    df = pd.concat([df[df.group != EU], pd.read_csv(f"{EU2}/tracks.csv")[cols]], ignore_index=True)
Z0, Z1 = np.load(f"{OLD}/states.npz"), np.load(f"{NEW}/states.npz")
Z2 = np.load(f"{EU2}/states.npz") if EU2 else None
def getst(tr, g):
    k = f"{tr}|{g}"
    if Z2 is not None and k in Z2.files: return Z2[k]
    return Z1[k] if k in Z1.files else Z0[k]
rng = np.random.default_rng(0)

print("=" * 100, "\nA. THE GROUPS (22 passages of 256 tokens each). Means per work:")
print("   'different words' = how many different tokens a passage uses; 'surprise' = how unpredictable the text is for Qwen")
for fam, gs in FAM.items():
    print(f"\n{fam}\n{'':52s}{'different':>10s}{'surprise':>9s}{'dimension':>20s}{'step length':>22s}\n{'':52s}{'words':>10s}{'(Qwen)':>9s}{'Qwen':>10s}{'XLM-R':>10s}{'Qwen':>11s}{'XLM-R':>11s}")
    for g in gs:
        s = df[df.group == g]
        print(f"{NICE.get(g, g):52s}{s.n_distinct_qwen.mean():10.0f}{s.surprise_qwen.mean():9.2f}{s['dim|Qwen contextual'].mean():10.1f}{s['dim|XLM-R contextual'].mean():10.1f}{s['step|Qwen contextual'].mean():11.0f}{s['step|XLM-R contextual'].mean():11.2f}")

def centroid_guess(F, y):
    F = (F - F.mean(0)) / (F.std(0) + 1e-12); pred = []
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        c = {k: F[keep & (y == k)].mean(0) for k in np.unique(y)}
        pred.append(min(c, key=lambda k: np.linalg.norm(F[i] - c[k])))
    return np.array(pred)
acc = lambda p, y: float((p == y).mean())

print("\n", "=" * 100, "\nB. HIDE WHICH WORK A PASSAGE COMES FROM; HOW OFTEN CAN WE GUESS IT?  (chance = 1 / number of works)")
res = {}; recall = {}
for fam, gs in FAM.items():
    sub = df[df.group.isin(gs)].reset_index(drop=True); y = sub.group.values
    for nm, tr in TR.items():
        for ms, cs in {"dimension": [f"dim|{tr}"], "spread": [f"spread|{tr}"], "step length": [f"step|{tr}"],
                       "all three": [f"dim|{tr}", f"spread|{tr}", f"step|{tr}"]}.items():
            res[(nm, ms, fam)] = acc(centroid_guess(sub[cs].values, y), y)
        P = [w.astype(np.float64) for g in gs for w in getst(tr, g)]; lab = np.array([g for g in gs for _ in getst(tr, g)]); n = len(P)
        Dm = np.full((n, n), np.inf)
        for i, j in combinations(range(n), 2): Dm[i, j] = Dm[j, i] = _dtw(cost_matrix(P[i], P[j]))
        res[(nm, "path distance (DTW)", fam)] = acc(lab[Dm.argmin(1)], lab)
    res[("-", "count of different words", fam)] = acc(centroid_guess(sub[["n_distinct_qwen"]].values, y), y)
    res[("Qwen", "surprise", fam)] = acc(centroid_guess(sub[["surprise_qwen"]].values, y), y)
    allc = [f"{m}|{t}" for t in ("Qwen contextual", "XLM-R contextual") for m in ("dim", "spread", "step")] + ["surprise_qwen", "n_distinct_qwen"]
    pred = centroid_guess(sub[allc].values, y); recall[fam] = ({g: float((pred[y == g] == g).mean()) for g in gs}, acc(pred, y), Counter((a, b) for a, b in zip(y, pred) if a != b).most_common(3))
fams = list(FAM)
print("chance:", {f: f"{1/len(FAM[f]):.0%}" for f in fams})
for title, keys in [("WHICH MODEL (dimension + spread + step length together)", [(nm, "all three") for nm in TR]),
                    ("WHICH MEASURE - dimension", [(nm, "dimension") for nm in TR]), ("spread", [(nm, "spread") for nm in TR]),
                    ("step length", [(nm, "step length") for nm in TR]), ("path distance between passages (DTW)", [(nm, "path distance (DTW)") for nm in TR]),
                    ("no model reading needed / Qwen's surprise", [("-", "count of different words"), ("Qwen", "surprise")])]:
    print(f"\n{title}\n{'':36s}" + "".join(f"{f:>20s}" for f in fams))
    for k in keys: print(f"{k[0] + ' - ' + k[1]:36s}" + "".join(f"{res[(k[0], k[1], f)]:20.0%}" for f in fams))

print("\n", "=" * 100, "\nC. WHICH WORKS ARE EASIEST TO RECOGNISE (all measures of both models together)")
for fam in fams:
    r, a, conf = recall[fam]; print(f"\n{fam}: {a:.0%} right overall (chance {1/len(FAM[fam]):.0%})")
    for g, v in r.items(): print(f"   {NICE.get(g, g):52s} recognised {v:.0%}")
    print("   most common mix-ups:", [(x.split(' (')[0], y.split(' (')[0], c) for (x, y), c in conf])

print("\n", "=" * 100, "\nD. RAMANUJAN 1918 vs THE OTHER PROOFS (difference of means, Mann-Whitney p)")
for g in (A5, A1, EU):
    a, b = df[df.group == HR], df[df.group == g]; out = []
    for c, nm in [("dim|Qwen contextual", "dimension Qwen"), ("dim|XLM-R contextual", "dimension XLM-R"), ("step|Qwen contextual", "step Qwen"), ("surprise_qwen", "surprise"), ("n_distinct_qwen", "different words")]:
        out.append(f"{nm} {a[c].mean() - b[c].mean():+.2f} (p={mannwhitneyu(a[c], b[c]).pvalue:.2g})")
    print(f"  vs {NICE[g]}:\n      " + "; ".join(out))

print("\n", "=" * 100, "\nE. SHUFFLE CONTROL (words shuffled inside each passage), mean change over paired passages; Wilcoxon p")
sh = [pd.read_csv(f"{NEWS}/tracks.csv")[cols]]
if OLDS: sh.append(pd.read_csv(f"{OLDS}/tracks.csv")[cols])
S = pd.concat(sh, ignore_index=True)
if EU2S: S = pd.concat([S[S.group != EU], pd.read_csv(f"{EU2S}/tracks.csv")[cols]], ignore_index=True); M = df.merge(S, on=["group", "label"], suffixes=("", "_s")); M = M[~M.group.isin(["Math ch.13 (combinatorics)"])]
print(f"({len(M)} paired passages)\n{'':8s}" + "".join(f"{k:>24s}" for k in TR))
for m in ("dim", "step"):
    o = []
    for k, tr in TR.items():
        d = (M[f"{m}|{tr}_s"] - M[f"{m}|{tr}"]).values
        try: p = wilcoxon(d).pvalue
        except ValueError: p = 1.0
        o.append(f"{d.mean():+.3g} (p={p:.2g})")
    print(f"{m:8s}" + "".join(f"{x:>24s}" for x in o))
