"""Same pipeline as run_tracks.py (windows of exactly 256 points, same estimator) for the NEW groups, without the
first-token track (dropped: see README). Tracks: Qwen contextual, XLM-R contextual, Qwen input table, XLM-R input table.

    python scripts/run_new_groups.py --out-dir DIR [--shuffle]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from textgeom import load_text, phd
from textgeom.corpus import split_delimited

P, SPAN = 256, 321


def sources(D):
    return {
        "Rimbaud poems": "\n\n".join(t for _, t in split_delimited(f"{D}/rimbaud/poesies.txt")),
        "Joyce Ulysses": load_text(f"{D}/joyce/ulysses.txt"),
        "Ramanujan 1918 (Hardy-Ramanujan paper)": load_text(f"{D}/ramanujan/hardy_ramanujan_1918.txt"),
        "Euclid (ancient proofs)": load_text(f"{D}/euclid/elements_casey_proofs_only.txt"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--n", type=int, default=22)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--only", default=None, help="run just this group (name)")
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from textgeom.embed import Embedder
    qtok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    qm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.float32).eval()
    xe = Embedder("xlm-roberta-base")
    q_table = qm.get_input_embeddings().weight.detach().float().numpy()
    x_table = xe.model.embeddings.word_embeddings.weight.detach().float().numpy()
    rng, shuf = np.random.default_rng(0), np.random.default_rng(1)
    est = dict(min_n=8, n_sizes=6, n_draws=7, n_fits=3, seed=0)
    rows, states = [], {}
    for g, text in sources(a.data).items():
        if a.only and g != a.only:
            continue
        ids = qtok(text, add_special_tokens=False)["input_ids"]
        starts = np.arange(0, len(ids) - SPAN + 1, SPAN)
        order = rng.permutation(len(starts))
        kept = 0
        for k in order:
            if kept >= a.n:
                break
            s = int(starts[k]); label = f"{g}@{s}"
            span = qtok.decode(ids[s:s + SPAN])
            if len(xe.token_ids(span)) < P:
                continue
            if a.shuffle:
                span = " ".join(shuf.permutation(span.split()))
            qi, xi = qtok(span, add_special_tokens=False)["input_ids"], xe.token_ids(span)
            if len(qi) < P + 1 or len(xi) < P:
                continue
            with torch.no_grad():
                o = qm(input_ids=torch.tensor([qi[:P + 1]]), output_hidden_states=True)
            lp = torch.log_softmax(o.logits[0, :-1], -1).gather(1, torch.tensor(qi[1:P + 1])[:, None])[:, 0]
            qt, xt = np.array(qi[1:P + 1]), np.array(xi[:P])
            pts = {"Qwen contextual": o.hidden_states[-1][0].numpy()[1:], "XLM-R contextual": xe.embed_ids(list(xt)),
                   "Qwen table": q_table[qt], "XLM-R table": x_table[xt]}
            row = dict(group=g, label=label, pair=-1, n_distinct_qwen=len(set(qt.tolist())), n_distinct_xlmr=len(set(xt.tolist())),
                       surprise_qwen=float(-lp.mean()))
            for tr, X in pts.items():
                X = X.astype(np.float64)
                row[f"dim|{tr}"] = phd(X, **est)
                row[f"spread|{tr}"] = np.linalg.norm(X - X.mean(0), axis=1).mean()
                row[f"step|{tr}"] = np.linalg.norm(np.diff(X, axis=0), axis=1).mean()
                states.setdefault(f"{tr}|{g}", []).append(X.astype(np.float16))
            rows.append(row); kept += 1
        print(f"{g}: {kept} windows (of {len(starts)} possible)", flush=True)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "tracks.csv", index=False)
    np.savez_compressed(out / "states.npz", **{k: np.stack(v) for k, v in states.items()})


if __name__ == "__main__":
    main()
