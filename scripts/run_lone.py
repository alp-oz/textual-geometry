"""Baselines without context vs contextual vectors, same text spans, same number of points.

For every window (the same text span, exactly P tokens per model) and for each model (XLM-R, Qwen):
  table     fixed vector from the input lookup table (before any layer)
  lone      model output when the token is the whole input (all layers, no context)
  context   model output on the whole window (XLM-R: bidirectional; Qwen: causal read)
  shuffled  as context, but the window's tokens are randomly permuted (order control)
Dimension = Steele/MST estimator (src/textgeom/phd.py). Also stores collapse diagnostics for the lone vectors.

    PYTHONPATH=src python scripts/run_lone.py --out-dir DIR [--n 22] [--points 128]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from run_fair import poe  # noqa: E402
from textgeom import load_text, phd  # noqa: E402
from textgeom.corpus import split_baudelaire, split_delimited, split_sonnets  # noqa: E402

EST = dict(min_n=8, n_sizes=6, n_draws=7, n_fits=3, seed=0)


def sources(D):
    cw = load_text(f"{D}/shakespeare/complete_works.txt")
    en, fr = poe(D)
    return {
        "Shakespeare sonnets": "\n\n".join(t for _, t in split_sonnets(f"{D}/shakespeare/sonnets.txt")),
        "Shakespeare plays": cw[250000:],
        "Baudelaire poems": "\n\n".join(t for _, t in split_baudelaire(f"{D}/baudelaire/fleurs_du_mal.txt")),
        "Mallarmé poems": "\n\n".join(t for _, t in split_delimited(f"{D}/mallarme/poesies.txt")),
        "Poe tales (English)": "\n\n".join(en),
        "Poe tales (French, Baudelaire)": "\n\n".join(fr),
    }


def spans(text, rng, n, chars=1500):
    """n random, non-overlapping-ish character spans starting at a word boundary."""
    out, tries = [], 0
    while len(out) < n and tries < 10000:
        tries += 1
        s = int(rng.integers(0, len(text) - chars))
        j = text.find(" ", s)
        if j == -1 or j + chars > len(text):
            continue
        if all(abs(j - k) > chars for k in out):
            out.append(j)
    return [text[j:j + chars] for j in sorted(out)]


class Model:
    def __init__(self, name):
        import torch
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
        self.t, self.name = torch, name
        if name == "XLM-R":
            self.tok = AutoTokenizer.from_pretrained("xlm-roberta-base")
            self.m = AutoModel.from_pretrained("xlm-roberta-base").eval()
            self.table = self.m.embeddings.word_embeddings.weight.detach().numpy()
            self.pre, self.post = [self.tok.cls_token_id], [self.tok.sep_token_id]
        else:
            self.tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
            self.m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.float32).eval()
            self.table = self.m.get_input_embeddings().weight.detach().numpy()
            self.pre, self.post = [], []
        self.cache = {}       # token id -> lone output vector (with the model's usual start/end tokens)
        self.cache_bare = {}  # token id -> lone output vector with no special tokens at all

    def hidden(self, seqs):
        t = self.t
        with t.no_grad():
            ids = t.tensor(seqs)
            if self.name == "XLM-R":
                return self.m(input_ids=ids).last_hidden_state.numpy()
            return self.m(input_ids=ids, output_hidden_states=True).hidden_states[-1].numpy()

    def lone(self, ids, bare=False):
        cache = self.cache_bare if bare else self.cache
        need = sorted({int(i) for i in ids} - cache.keys())
        pre, post = ([], []) if bare else (self.pre, self.post)
        for k in range(0, len(need), 256):
            batch = need[k:k + 256]
            H = self.hidden([pre + [i] + post for i in batch])[:, len(pre)]
            cache.update({i: h for i, h in zip(batch, H)})
        return np.stack([cache[int(i)] for i in ids])

    def context(self, ids):
        """Contextual vectors for exactly the tokens in ids."""
        H = self.hidden([self.pre + list(ids) + self.post])[0]
        return H[len(self.pre):len(self.pre) + len(ids)]


def dims(X):
    u = np.unique(X, axis=0)
    return phd(X.astype(np.float64), **EST), (phd(u.astype(np.float64), **EST) if len(u) >= 2 * EST["min_n"] else np.nan), len(u)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--n", type=int, default=22)
    ap.add_argument("--points", type=int, default=128)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--models", default="XLM-R,Qwen", help="comma-separated; each model is saved to its own csv")
    a = ap.parse_args()
    P = a.points
    S = sources(a.data)
    rng = np.random.default_rng(0)
    texts = {g: spans(t, rng, a.n + 8) for g, t in S.items()}
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for name in a.models.split(","):
        rows = []
        M = Model(name)
        # Qwen: first token of a causal read is the attention sink; drop it from every cloud as the matched run did
        skip = 1 if name == "Qwen" else 0
        for g, ts in texts.items():
            kept = 0
            for t in ts:
                if kept >= a.n:
                    break
                ids = M.tok(t, add_special_tokens=False)["input_ids"]
                if len(ids) < P + skip:
                    continue
                ids = ids[:P + skip]
                pts = ids[skip:]
                shuf = list(np.random.default_rng(kept).permutation(ids))   # same tokens, random order
                clouds = {
                    "table": M.table[np.array(pts)],
                    "lone": M.lone(pts),
                    "lone_bare": M.lone(pts, bare=True),
                    "context": M.context(ids)[skip:],
                    "shuffled": M.context(shuf)[skip:],
                }
                row = dict(model=name, group=g, label=f"{g}#{kept}", n_distinct_tokens=len(set(pts)))
                for k, X in clouds.items():
                    d, dd, nu = dims(X)
                    row.update({f"dim_{k}": d, f"dimdistinct_{k}": dd,
                                f"spread_{k}": float(np.linalg.norm(X - X.mean(0), axis=1).mean())})
                rows.append(row)
                kept += 1
            print(name, g, kept, "windows", flush=True)
            pd.DataFrame(rows).to_csv(out / f"lone_{name}.csv", index=False)
        # collapse diagnostics over every distinct token seen: do lone vectors all look alike?
        for kind, cache in (("lone", M.cache), ("lone_bare", M.cache_bare)):
            V = np.stack(list(cache.values()))
            Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
            idx = np.random.default_rng(0).choice(len(V), min(1500, len(V)), replace=False)
            C = Vn[idx] @ Vn[idx].T
            sv = np.linalg.svd(V - V.mean(0), compute_uv=False) ** 2
            rows.append(dict(model=name, group="__diagnostic__", label=kind, n_distinct_tokens=len(V),
                             mean_norm=float(np.linalg.norm(V, axis=1).mean()),
                             sd_norm=float(np.linalg.norm(V, axis=1).std()),
                             mean_cos=float(C[np.triu_indices(len(idx), 1)].mean()),
                             partic_ratio=float(sv.sum() ** 2 / (sv ** 2).sum())))
        pd.DataFrame(rows).to_csv(out / f"lone_{name}.csv", index=False)


if __name__ == "__main__":
    main()
