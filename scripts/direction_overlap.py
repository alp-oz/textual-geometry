"""Do two works spread along the SAME directions?  For each work: pool the points of its passages, find the 10 directions of largest spread
(principal directions), and measure how much the 10-direction sets of two works overlap (1 = identical set, 0 = unrelated; random is about 10/dimension).
Ceiling = overlap between two random halves of the SAME work's passages.   python scripts/direction_overlap.py SCRATCH_DIR"""
import sys
import numpy as np

S = sys.argv[1]; K = 10
old, new, eu, pr = (np.load(f"{S}/{d}/states.npz") for d in ("tracks", "new", "eu", "proofs_raw"))
def get(tr, g):
    for z in (eu, new, old):
        if f"{tr}|{g}" in z.files: return z[f"{tr}|{g}"].astype(np.float32)
FAM = {"French literature": ["Baudelaire poems", "Mallarmé poems", "Beckett L'Innommable", "Poe tales (French, Baudelaire)", "Rimbaud poems"],
       "English literature": ["Shakespeare sonnets", "Shakespeare plays", "Poe tales (English)", "Joyce Ulysses"]}
PROOFS = ["Ramanujan 1918 (Hardy-Ramanujan paper)", "Andrews ch.5 (modern retelling of the same proof)", "Andrews ch.1 (simple modern proof)", "Euclid (ancient)", "Hilbert (1899)", "Dedekind (1888)", "Your paper"]
def dirs(P):
    X = P.reshape(-1, P.shape[-1]); X = X - X.mean(0)
    w, v = np.linalg.eigh(np.cov(X.T)); return v[:, ::-1][:, :K]
ov = lambda A, B: float((np.linalg.norm(A.T @ B) ** 2) / K)
rng = np.random.default_rng(0)
def run(title, names, getter, tr):
    U = {g: dirs(getter(tr, g)) for g in names}; ceil = {}
    for g in names:
        P = getter(tr, g); idx = rng.permutation(len(P)); h = len(P) // 2
        ceil[g] = ov(dirs(P[idx[:h]]), dirs(P[idx[h:2 * h]]))
    print(f"\n{title} - {tr}   (ceiling = same work, two halves: {np.mean(list(ceil.values())):.2f}; random = {K / U[names[0]].shape[0]:.2f})")
    short = lambda g: g.split(' (')[0][:16]
    print(f"{'':18s}" + "".join(f"{short(g):>17s}" for g in names))
    for a in names:
        print(f"{short(a):18s}" + "".join(f"{ov(U[a], U[b]):17.2f}" for b in names))
for tr in ("Qwen contextual", "XLM-R contextual"):
    for fam, gs in FAM.items(): run(fam, gs, get, tr)
    run("Proofs", PROOFS, lambda t, g: pr[f"{t}|{g}"].astype(np.float32), tr)
