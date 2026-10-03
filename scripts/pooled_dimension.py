"""Dimension of the POOLED cloud of a whole work (all 22 passages x 256 points = 5,632 points), at several numbers of points
so that the curve shows where the estimate levels off.  Two contextual readers (Qwen, XLM-R).
    python scripts/pooled_dimension.py SCRATCH_DIR OUT.csv"""
import os
import sys
import numpy as np
import pandas as pd
from textgeom import phd

S, OUT = sys.argv[1], sys.argv[2]
old, new, eu, pr = (np.load(f"{S}/{d}/states.npz") for d in ("tracks", "new", "eu", "proofs_raw"))
def get(tr, g, proofs=False):
    for z in ((pr,) if proofs else (eu, new, old)):
        if f"{tr}|{g}" in z.files: return z[f"{tr}|{g}"]
FAM = {"French literature": ["Baudelaire poems", "Mallarmé poems", "Beckett L'Innommable", "Poe tales (French, Baudelaire)", "Rimbaud poems"],
       "English literature": ["Shakespeare sonnets", "Shakespeare plays", "Poe tales (English)", "Joyce Ulysses"],
       "Proofs": ["Ramanujan 1918 (Hardy-Ramanujan paper)", "Andrews ch.5 (modern retelling of the same proof)", "Andrews ch.1 (simple modern proof)",
                  "Euclid (ancient)", "Hilbert (1899)", "Dedekind (1888)", "Your paper"]}
SIZES = [250, 500, 1000, 2000, 3000]
rows = pd.read_csv(OUT).to_dict('records') if os.path.exists(OUT) else []   # resume after an interrupted run
done = {(r['work'], r['model']) for r in rows}
for fam, works in FAM.items():
    for g in works:
        for tr in ("Qwen contextual", "XLM-R contextual"):
            if (g, tr.split()[0]) in done: continue
            P = get(tr, g, proofs=(fam == "Proofs")); X = P.reshape(-1, P.shape[-1]).astype(np.float64)
            for n, seed in [(s, 0) for s in SIZES] + [(1000, 1)]:
                idx = np.random.default_rng(seed).choice(len(X), n, replace=False)
                d = phd(X[idx], min_n=20, n_sizes=6, n_draws=3, n_fits=2, seed=seed)
                rows.append(dict(family=fam, work=g, model=tr.split()[0], n_points=n, replicate=seed, dim=d))
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(fam, g, tr, "done", flush=True)
