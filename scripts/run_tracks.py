"""Three tracks on IDENTICAL windows (exactly P=256 points each), every metric on every track.

 Track 1  CONTEXTUAL, running  : Qwen2.5-0.5B read left to right; the state after each token.
 Track 2  CONTEXTUAL, whole    : XLM-RoBERTa-base reads the whole chunk; one vector per token.
 Track 3  FIRST-TOKEN          : each token is fed ALONE as the whole input to the model; the output vector is the
                                 token's fixed vector (same wherever the token occurs). Laid out in text order.
          (reference) TABLE    : the input lookup-table vector (before any layer).
 Order-blind metrics : Steele dimension, spread, number of distinct tokens.
 Order-aware metrics : step length (+ path distances computed later in analyze_tracks.py), surprise (Qwen only).
 --shuffle : words of every window are shuffled before anything is computed (control).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import run_matched  # noqa: E402
from textgeom import phd  # noqa: E402

TRACKS = ["Qwen contextual", "XLM-R contextual", "Qwen first-token", "XLM-R first-token", "Qwen table", "XLM-R table"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--n", type=int, default=22)
    ap.add_argument("--points", type=int, default=256)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--check-labels", default=None, help="csv whose labels the window selection must reproduce")
    a = ap.parse_args()
    P = a.points
    run_matched.SPAN = P + 65
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from textgeom.embed import Embedder
    qtok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    qm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.float32).eval()
    xe = Embedder("xlm-roberta-base")
    q_table = qm.get_input_embeddings().weight.detach().float().numpy()
    x_table = xe.model.embeddings.word_embeddings.weight.detach().float().numpy()
    first = {"Qwen": {}, "XLM-R": {}}

    def first_vecs(kind, toks):
        cache = first[kind]
        need = sorted({int(t) for t in toks} - cache.keys())
        for i in range(0, len(need), 256):
            b = need[i:i + 256]
            with torch.no_grad():
                if kind == "Qwen":     # the token alone is the whole input
                    h = qm.model(input_ids=torch.tensor(b)[:, None]).last_hidden_state[:, 0].numpy()
                else:                  # <s> token </s>; take the token's own output
                    ids = torch.tensor([[xe.tok.cls_token_id, t, xe.tok.sep_token_id] for t in b])
                    h = xe.model(input_ids=ids).last_hidden_state[:, 1].numpy()
            for t, v in zip(b, h):
                cache[t] = v
        return np.stack([cache[int(t)] for t in toks])

    W = run_matched.spans(a, qtok, np.random.default_rng(0), a.n)
    est = dict(min_n=8, n_sizes=6, n_draws=7, n_fits=3, seed=0)
    shuf = np.random.default_rng(1)
    rows, states, labels_seen = [], {}, []
    for g, ws in W.items():
        kept = 0
        for label, ids, pair in ws:
            if kept >= a.n:
                break
            text = qtok.decode(ids)
            if len(xe.token_ids(text)) < P:          # same skip rule as the matched run (decided on the unshuffled text)
                continue
            if a.shuffle:
                text = " ".join(shuf.permutation(text.split()))
            qi = qtok(text, add_special_tokens=False)["input_ids"]
            xi = xe.token_ids(text)
            labels_seen.append((g, label))
            if len(qi) < P + 1 or len(xi) < P:
                continue
            with torch.no_grad():
                o = qm(input_ids=torch.tensor([qi[:P + 1]]), output_hidden_states=True)
            lp = torch.log_softmax(o.logits[0, :-1], -1).gather(1, torch.tensor(qi[1:P + 1])[:, None])[:, 0]
            qt, xt = np.array(qi[1:P + 1]), np.array(xi[:P])
            pts = {"Qwen contextual": o.hidden_states[-1][0].numpy()[1:],
                   "XLM-R contextual": xe.embed_ids(list(xt)),
                   "Qwen first-token": first_vecs("Qwen", qt), "XLM-R first-token": first_vecs("XLM-R", xt),
                   "Qwen table": q_table[qt], "XLM-R table": x_table[xt]}
            row = dict(group=g, label=label, pair=pair, n_distinct_qwen=len(set(qt.tolist())), n_distinct_xlmr=len(set(xt.tolist())),
                       surprise_qwen=float(-lp.mean()))
            for tr, X in pts.items():
                X = X.astype(np.float64)
                row[f"dim|{tr}"] = phd(X, **est)
                row[f"spread|{tr}"] = np.linalg.norm(X - X.mean(0), axis=1).mean()
                row[f"step|{tr}"] = np.linalg.norm(np.diff(X, axis=0), axis=1).mean()
                states.setdefault(f"{tr}|{g}", []).append(X.astype(np.float16))
            rows.append(row)
            kept += 1
        print(f"{g}: {kept} windows", flush=True)
    if a.check_labels:
        ref = pd.read_csv(a.check_labels)
        mine = [(g, l) for g, l in labels_seen]
        assert mine[: len(ref)] == list(zip(ref.group, ref.label)) or a.shuffle, "window selection differs from the reference run"
        print("window selection matches the reference run" if not a.shuffle else "shuffle run: selection rule identical")
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "tracks.csv", index=False)
    np.savez_compressed(out / "states.npz", **{k: np.stack(v) for k, v in states.items()})
    # diagnostics of the first-token vectors (are they informative or all alike?)
    for kind, cache in first.items():
        M = np.stack(list(cache.values())); Mn = M / np.linalg.norm(M, axis=1, keepdims=True)
        idx = np.random.default_rng(0).choice(len(M), min(500, len(M)), replace=False)
        cos = (Mn[idx] @ Mn[idx].T)[np.triu_indices(len(idx), 1)].mean()
        print(f"first-token vectors, {kind}: {len(M)} distinct tokens, mean pairwise cosine {cos:.3f}, mean norm {np.linalg.norm(M, axis=1).mean():.1f}")


if __name__ == "__main__":
    main()
