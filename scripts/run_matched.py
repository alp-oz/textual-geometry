"""Both models on identical terms. Every window = a span of text; each model embeds it and keeps EXACTLY
128 points (Qwen: states of tokens 1..128 of a causal read; XLM-R: its first 128 tokens, contextual
embeddings). Same estimator (Steele/MST dimension), same windows, same N per group.

    python scripts/run_matched.py --out-dir DIR [--n 40]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from run_fair import sources, poe  # noqa: E402
from textgeom import phd  # noqa: E402

P = 128            # points per window, for every model and every group
SPAN = 193         # Qwen tokens per text span (1 sink token + 128 points + margin so XLM-R also gets >= 128)


def spans(a, qtok, rng, n):
    W = {}
    for g, text in sources(a.data).items():
        ids = qtok(text, add_special_tokens=False)["input_ids"]
        starts = np.arange(0, len(ids) - SPAN + 1, SPAN)
        order = rng.permutation(len(starts))
        W[g] = [(f"{g}@{starts[i]}", ids[starts[i]:starts[i] + SPAN], -1) for i in order[: n + 10]]
    en, fr = poe(a.data)
    ids_en = [qtok(t, add_special_tokens=False)["input_ids"] for t in en]
    ids_fr = [qtok(t, add_special_tokens=False)["input_ids"] for t in fr]
    wts = np.array([len(x) for x in ids_en], float); wts /= wts.sum()
    W["Poe tales (English)"], W["Poe tales (French, Baudelaire)"] = [], []
    for k in range(n + 10):
        t = rng.choice(len(en), p=wts); f = rng.uniform()
        for g, ids in (("Poe tales (English)", ids_en), ("Poe tales (French, Baudelaire)", ids_fr)):
            s = int(f * (len(ids[t]) - SPAN))
            W[g].append((f"tale{t}@{f:.3f}", ids[t][s:s + SPAN], k))
    return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--points", type=int, default=128, help="points per window, identical for both models")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    global P, SPAN
    P, SPAN = a.points, a.points + 65
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from textgeom.embed import Embedder
    qtok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    qm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.float32).eval()
    xe = Embedder("xlm-roberta-base")
    W = spans(a, qtok, np.random.default_rng(0), a.n)
    est = dict(min_n=8, n_sizes=6, n_draws=7, n_fits=3, seed=0)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    f = out / "matched.csv"
    rows = pd.read_csv(f).to_dict("records") if f.exists() else []     # resume: skip groups already done
    done = {r["group"] for r in rows}
    for g, ws in W.items():
        if g in done:
            continue
        kept = 0
        for label, ids, pair in ws:
            if kept >= a.n:
                break
            xi = xe.token_ids(qtok.decode(ids))
            if len(xi) < P:                       # XLM-R would have fewer than 128 points: skip this span for both
                continue
            with torch.no_grad():
                o = qm(input_ids=torch.tensor([ids[:P + 1]]), output_hidden_states=True)
            H = o.hidden_states[-1][0].numpy()[1:]                          # exactly P points
            lp = torch.log_softmax(o.logits[0, :-1], -1).gather(1, torch.tensor(ids[1:P + 1])[:, None])[:, 0]
            X = xe.embed_ids(xi[:P])                                        # exactly P points
            rows.append(dict(group=g, label=label, pair=pair,
                             dim_qwen=phd(H, **est), dim_xlmr=phd(X, **est),
                             step_qwen=np.linalg.norm(np.diff(H, axis=0), axis=1).mean(), surprise_qwen=float(-lp.mean()),
                             spread_qwen=np.linalg.norm(H - H.mean(0), axis=1).mean(),
                             spread_xlmr=np.linalg.norm(X - X.mean(0), axis=1).mean()))
            kept += 1
        print(f"{g}: {kept} windows", flush=True)
        pd.DataFrame(rows).to_csv(f, index=False)                       # save after every group


if __name__ == "__main__":
    main()
