"""Same matched windows as run_fair.py (identical selection, checked against windows.csv), embedded with
XLM-RoBERTa-base (bidirectional, the Tulchinskii-style recipe) -> one dimension per window.

    python scripts/run_fair_xlmr.py --fair-dir DIR
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from run_fair import sources, poe  # noqa: E402
from textgeom import phd  # noqa: E402


def choose(a, tok, rng, L, N):
    """Identical to the selection in run_fair.py."""
    W = {}
    for g, text in sources(a.data).items():
        ids = tok(text, add_special_tokens=False)["input_ids"]
        starts = np.arange(0, len(ids) - L + 1, L)
        pick = np.sort(rng.choice(starts, size=min(N, len(starts)), replace=False))
        W[g] = [(f"{g}@{s}", ids[s:s + L], -1) for s in pick]
    en, fr = poe(a.data)
    ids_en = [tok(t, add_special_tokens=False)["input_ids"] for t in en]
    ids_fr = [tok(t, add_special_tokens=False)["input_ids"] for t in fr]
    wts = np.array([len(x) for x in ids_en], float); wts /= wts.sum()
    W["Poe tales (English)"], W["Poe tales (French, Baudelaire)"] = [], []
    for k in range(N):
        t = rng.choice(len(en), p=wts); f = rng.uniform()
        for g, ids in (("Poe tales (English)", ids_en), ("Poe tales (French, Baudelaire)", ids_fr)):
            s = int(f * (len(ids[t]) - L))
            W[g].append((f"tale{t}@{f:.3f}", ids[t][s:s + L], k))
    return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--fair-dir", required=True)
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from textgeom.embed import Embedder
    qtok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    W = choose(a, qtok, np.random.default_rng(0), a.L, a.n)
    ref = pd.read_csv(f"{a.fair_dir}/windows.csv")
    mine = [(g, w[0]) for g, ws in W.items() for w in ws]
    assert mine == list(zip(ref.group, ref.label)), "window selection differs from the main run"
    print("window selection identical to the main run:", len(mine), "windows", flush=True)
    emb = Embedder("xlm-roberta-base")
    rows = []
    for g, ws in W.items():
        for label, ids, pair in ws:
            text = qtok.decode(ids)
            xi = emb.token_ids(text)[:256]
            X = emb.embed_ids(xi)
            rows.append(dict(group=g, label=label, n_pts=len(xi),
                             dim_xlmr=phd(X, min_n=8, n_sizes=6, n_draws=7, n_fits=3, seed=0)))
        print(g, "done", flush=True)
    pd.DataFrame(rows).to_csv(f"{a.fair_dir}/windows_xlmr.csv", index=False)


if __name__ == "__main__":
    main()
