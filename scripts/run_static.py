"""Fixed-vector version: NO contextual model. A token is mapped to its fixed vector in the model's input
embedding table (same vector wherever the token occurs); a window = exactly 256 tokens of a fixed text;
we measure the dispersion of those points. Two different tables (XLM-R, Qwen) are used as a check.

    python scripts/run_static.py --out-dir DIR [--n 30]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from run_fair import sources, poe  # noqa: E402
from textgeom import phd  # noqa: E402

L = 256


def windows(tok, S, en, fr, rng, n):
    W = {}
    for g, text in S.items():
        ids = tok(text, add_special_tokens=False)["input_ids"]
        starts = np.arange(0, len(ids) - L + 1, L)
        pick = np.sort(rng.choice(starts, size=min(n, len(starts)), replace=False))
        W[g] = [(f"{g}@{s}", ids[s:s + L], -1) for s in pick]
    ie = [tok(t, add_special_tokens=False)["input_ids"] for t in en]
    iff = [tok(t, add_special_tokens=False)["input_ids"] for t in fr]
    wts = np.array([len(x) for x in ie], float); wts /= wts.sum()
    W["Poe tales (English)"], W["Poe tales (French, Baudelaire)"] = [], []
    for k in range(n):
        t = rng.choice(len(en), p=wts); f = rng.uniform()
        for g, ids in (("Poe tales (English)", ie), ("Poe tales (French, Baudelaire)", iff)):
            s = int(f * (len(ids[t]) - L))
            W[g].append((f"tale{t}@{f:.3f}", ids[t][s:s + L], k))
    return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    from transformers import AutoModel, AutoTokenizer
    S = sources(a.data)
    en, fr = poe(a.data)
    est = dict(min_n=8, n_sizes=6, n_draws=7, n_fits=3, seed=0)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    f = out / "static.csv"
    rows = pd.read_csv(f).to_dict("records") if f.exists() else []
    done = {(r["table"], r["group"]) for r in rows}
    for name, hub, getter in [("XLM-R table", "xlm-roberta-base", lambda m: m.embeddings.word_embeddings.weight),
                              ("Qwen table", "Qwen/Qwen2.5-0.5B", lambda m: m.get_input_embeddings().weight)]:
        tok = AutoTokenizer.from_pretrained(hub)
        table = getter(AutoModel.from_pretrained(hub)).detach().float().numpy()
        W = windows(tok, S, en, fr, np.random.default_rng(0), a.n)
        for g, ws in W.items():
            if (name, g) in done:
                continue
            for label, ids, pair in ws:
                X = table[np.array(ids)].astype(np.float64)
                u = np.unique(ids)
                rows.append(dict(table=name, group=g, label=label, pair=pair, n_distinct=len(u),
                                 dim_occ=phd(X, **est), dim_distinct=phd(table[u].astype(np.float64), **est) if len(u) >= 40 else np.nan,
                                 spread=np.linalg.norm(X - X.mean(0), axis=1).mean()))
            pd.DataFrame(rows).to_csv(f, index=False)
            print(name, g, len(ws), "windows", flush=True)


if __name__ == "__main__":
    main()
