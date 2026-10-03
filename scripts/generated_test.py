"""Human or machine? Passages written by Claude (data/generated) against the real works they imitate.
A. Human against Claude, one pair at a time: learn on one half of each text (by position), guess passages of the other half,
   both directions, passages within 256 tokens of the split dropped (same strict rule as honest_test.py). Chance = 50%.
B. Which work does Claude's text resemble? A guesser trained on all passages of the real works of the family labels each Claude passage.
C. Means of the simple measures, human against Claude.
    python scripts/generated_test.py SCRATCH_DIR > examples/generated_test.txt     (SCRATCH_DIR holds tracks/, proofs_raw/, generated/)
Layout check: add the word `layout` as a second argument to use generated_lay/ (texts given the same layout as the real ones,
see scripts/layout_clean.py); only part A is run, and the real Euclid is the whitespace-normalised one from the same run."""
import sys
from collections import Counter
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from textgeom.metrics import cost_matrix, _dtw

S = sys.argv[1]
LAYOUT = len(sys.argv) > 2 and sys.argv[2] == "layout"
GAP = 256
TR = {"Qwen": "Qwen contextual", "XLM-R": "XLM-R contextual"}
PAIRS = [  # (name, real group, source dir of the real group, Claude group, family for part B)
    ("Shakespeare's sonnets", "Shakespeare sonnets", "tracks", "Sonnets written by Claude", "English literature"),
    ("Baudelaire", "Baudelaire poems", "tracks", "Baudelaire-style poems written by Claude", "French literature"),
    ("Euclid", "Euclid (ancient)", "proofs_raw", "Euclid-style proofs written by Claude", "Proofs"),
]
FAM = {
    "English literature": ("tracks", ["Shakespeare sonnets", "Shakespeare plays", "Poe tales (English)"]),
    "French literature": ("tracks", ["Baudelaire poems", "Mallarmé poems", "Beckett L'Innommable", "Poe tales (French, Baudelaire)"]),
    "Proofs": ("proofs_raw", ["Ramanujan 1918 (Hardy-Ramanujan paper)", "Andrews ch.5 (modern retelling of the same proof)",
                              "Andrews ch.1 (simple modern proof)", "Euclid (ancient)", "Hilbert (1899)", "Dedekind (1888)", "Your paper"]),
}
FEATS = {
    "surprise": ["surprise_qwen"], "step length (Qwen)": ["step|Qwen contextual"], "dimension (Qwen)": ["dim|Qwen contextual"],
    "dimension (XLM-R)": ["dim|XLM-R contextual"], "different words": ["n_distinct_qwen"],
    "all measures together": [f"{m}|{t}" for t in TR.values() for m in ("dim", "spread", "step")] + ["surprise_qwen", "n_distinct_qwen"],
}
GEN_DIR = "generated_lay" if LAYOUT else "generated"
if LAYOUT:
    PAIRS[2] = ("Euclid", "Euclid (ancient), whitespace normalised", "generated", "Euclid-style proofs written by Claude", "Proofs")
CSV = {"tracks": "tracks.csv", "proofs_raw": "proofs.csv", "generated": "proofs.csv"}
DIR = {"tracks": "tracks", "proofs_raw": "proofs_raw", "generated": GEN_DIR}
DF = {d: pd.read_csv(f"{S}/{DIR[d]}/{CSV[d]}") for d in CSV}
Z = {d: np.load(f"{S}/{DIR[d]}/states.npz") for d in CSV}


def window(d, g):
    """rows of group g in source d, with their passages' states for both readers, in csv order."""
    df = DF[d][DF[d].group == g].reset_index(drop=True)
    st = {nm: Z[d][f"{tr}|{g}"].astype(np.float64) for nm, tr in TR.items()}
    return df, st


def halves(label):
    pos = label.str.split("@").str[-1].astype(float).values
    med = np.median(pos)
    return np.where(pos < med - GAP, 0, np.where(pos > med + GAP, 1, -1))


def centroid(Ftr, ytr, Fte):
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-12
    Ftr, Fte = (Ftr - mu) / sd, (Fte - mu) / sd
    c = {k: Ftr[ytr == k].mean(0) for k in np.unique(ytr)}
    return np.array([min(c, key=lambda k: np.linalg.norm(f - c[k])) for f in Fte])


rows, part_c = [], []
print("A. HUMAN AGAINST CLAUDE (learn on one half, guess the other half, both directions; random guessing 50%)")
for name, rg, rd, gg, fam in PAIRS:
    R, Rs = window(rd, rg)
    G, Gs = window("generated", gg)
    df = pd.concat([R.assign(src="human"), G.assign(src="Claude")], ignore_index=True)
    h = np.concatenate([halves(R.label), halves(G.label)])
    keep = h >= 0
    y = df.src.values
    for fn, cols in FEATS.items():
        F = df[cols].values.astype(float)
        ok = 0
        for d in (0, 1):
            tr_, te_ = keep & (h == d), keep & (h == 1 - d)
            ok += int((centroid(F[tr_], y[tr_], F[te_]) == y[te_]).sum())
        rows.append((name, fn, ok, int(keep.sum())))
    for nm in TR:
        P = list(Rs[nm]) + list(Gs[nm])
        ok = 0
        for d in (0, 1):
            tr_ = np.where(keep & (h == d))[0]
            for i in np.where(keep & (h == 1 - d))[0]:
                dist = [_dtw(cost_matrix(P[i], P[j])) for j in tr_]
                ok += int(y[tr_[int(np.argmin(dist))]] == y[i])
        rows.append((name, f"path distance ({nm})", ok, int(keep.sum())))
    for fn, col in {"different words": "n_distinct_qwen", "surprise (Qwen)": "surprise_qwen", "step length (Qwen)": "step|Qwen contextual",
                    "dimension (Qwen)": "dim|Qwen contextual", "dimension (XLM-R)": "dim|XLM-R contextual"}.items():
        part_c.append((name, fn, R[col].mean(), G[col].mean()))
res = pd.DataFrame(rows, columns=["pair", "measure", "right", "n"])
res["acc"] = res.right / res.n
res["p"] = [binomtest(r, n, 0.5, alternative="greater").pvalue for r, n in zip(res.right, res.n)]
ci = [binomtest(int(r), int(n)).proportion_ci(confidence_level=0.95, method="wilson") for r, n in zip(res.right, res.n)]
res["lo"], res["hi"] = [c.low for c in ci], [c.high for c in ci]
m, run, holm = len(res), 0.0, {}
for rank, i in enumerate(res.p.sort_values().index):
    run = max(run, min(1.0, res.p[i] * (m - rank))); holm[i] = run
res["p_holm"] = pd.Series(holm)
for name, *_ in PAIRS:
    r = res[res.pair == name]
    print(f"\n{name}: {r.n.iloc[0]} passages tested")
    print(f"  {'measure':26s}{'right':>8s}{'95% range':>14s}{'p (corrected)':>16s}")
    for _, x in r.sort_values("acc", ascending=False).iterrows():
        print(f"  {x.measure:26s}{x.acc:8.0%}{f'{x.lo:.0%}-{x.hi:.0%}':>14s}{x.p_holm:16.3g}")
print(f"\n(p-values corrected for the {m} tests of part A)")

if not LAYOUT:
    print("\nB. WHICH WORK DOES CLAUDE'S TEXT RESEMBLE? (guesser trained on every passage of the real works of the family)")
    for name, rg, rd, gg, fam in PAIRS:
        d, gs = FAM[fam]
        sub = DF[d][DF[d].group.isin(gs)].reset_index(drop=True)
        G, Gs = window("generated", gg)
        y = sub.group.values
        out = {}
        cols = FEATS["all measures together"]
        out["all measures together"] = centroid(sub[cols].values.astype(float), y, G[cols].values.astype(float))
        for nm, tr in TR.items():
            P = [Z[d][f"{tr}|{g}"][i].astype(np.float64) for g in gs for i in range(len(DF[d][DF[d].group == g]))]
            lab = np.array([g for g in gs for _ in range(len(DF[d][DF[d].group == g]))])
            out[f"path distance ({nm})"] = np.array([lab[int(np.argmin([_dtw(cost_matrix(q.astype(np.float64), p)) for p in P]))] for q in Gs[nm]])
        print(f"\n{gg}  (real works in the family: {', '.join(g.split(' (')[0] for g in gs)})")
        for k, pred in out.items():
            c = Counter(pred)
            print(f"  {k:26s} -> " + ", ".join(f"{g.split(' (')[0]} {n}" for g, n in c.most_common()))


print("\nC. MEANS (human | Claude)")
print(f"  {'':24s}{'measure':22s}{'human':>9s}{'Claude':>9s}")
for name, fn, a, b in part_c:
    print(f"  {name:24s}{fn:22s}{a:9.2f}{b:9.2f}")
res.to_csv(f"{S}/{'layout_check' if LAYOUT else 'generated_test'}.csv", index=False)
