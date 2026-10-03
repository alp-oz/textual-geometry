"""Proof family: every text of a directory, passages of exactly 256 tokens, same estimator and tracks as the other runs.
   python scripts/run_proofs.py --dir data/proofs_raw --n 22 --out-dir OUT      (use --dir data/prose for the prose-only version)
The number of passages per work is min(--n, shortest text's number of passages - 1) so that every work has the same number."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from textgeom import phd

P, SPAN = 256, 321
NAMES = {"hr1918_marker": "Ramanujan 1918 (Hardy-Ramanujan paper)", "user_paper": "Your paper",
         "andrews_ch5": "Andrews ch.5 (modern retelling of the same proof)", "andrews_ch1": "Andrews ch.1 (simple modern proof)",
         "euclid": "Euclid (ancient)", "hilbert": "Hilbert (1899)", "dedekind": "Dedekind (1888)"}


NAMES_GENERATED = {"sonnets_claude": "Sonnets written by Claude", "baudelaire_claude": "Baudelaire-style poems written by Claude",
                   "euclid_claude": "Euclid-style proofs written by Claude",
                   "euclid_human": "Euclid (ancient), whitespace normalised"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--n", type=int, default=22)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--generated", action="store_true", help="read the texts written by Claude (data/generated) instead of the proofs")
    a = ap.parse_args()
    NAMES_USED = NAMES_GENERATED if a.generated else NAMES
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from textgeom.embed import Embedder
    qtok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    qm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.float32).eval()
    xe = Embedder("xlm-roberta-base")
    q_table = qm.get_input_embeddings().weight.detach().float().numpy()
    x_table = xe.model.embeddings.word_embeddings.weight.detach().float().numpy()
    texts = {NAMES_USED[p.stem]: p.read_text(encoding="utf-8") for p in sorted(Path(a.dir).glob("*.txt")) if p.stem in NAMES_USED}
    ids = {g: qtok(t, add_special_tokens=False)["input_ids"] for g, t in texts.items()}
    avail = {g: len(range(0, len(i) - SPAN + 1, SPAN)) for g, i in ids.items()}
    n = min(a.n, min(avail.values()) - 1)
    print("passages available per work:", avail, "-> using", n, "per work", flush=True)
    rng = np.random.default_rng(0)
    est = dict(min_n=8, n_sizes=6, n_draws=7, n_fits=3, seed=0)
    rows, states = [], {}
    for g, tok_ids in ids.items():
        starts = np.arange(0, len(tok_ids) - SPAN + 1, SPAN)
        kept = 0
        for k in rng.permutation(len(starts)):
            if kept >= n:
                break
            s = int(starts[k]); qi = tok_ids[s:s + SPAN]
            xi = xe.token_ids(qtok.decode(qi))
            if len(xi) < P:
                continue
            with torch.no_grad():
                o = qm(input_ids=torch.tensor([qi[:P + 1]]), output_hidden_states=True)
            lp = torch.log_softmax(o.logits[0, :-1], -1).gather(1, torch.tensor(qi[1:P + 1])[:, None])[:, 0]
            qt, xt = np.array(qi[1:P + 1]), np.array(xi[:P])
            pts = {"Qwen contextual": o.hidden_states[-1][0].numpy()[1:], "XLM-R contextual": xe.embed_ids(list(xt)),
                   "Qwen table": q_table[qt], "XLM-R table": x_table[xt]}
            row = dict(group=g, label=f"{g}@{s}", n_distinct_qwen=len(set(qt.tolist())), surprise_qwen=float(-lp.mean()))
            for tr, X in pts.items():
                X = X.astype(np.float64)
                row[f"dim|{tr}"] = phd(X, **est); row[f"spread|{tr}"] = np.linalg.norm(X - X.mean(0), axis=1).mean()
                row[f"step|{tr}"] = np.linalg.norm(np.diff(X, axis=0), axis=1).mean()
                states.setdefault(f"{tr}|{g}", []).append(X.astype(np.float16))
            rows.append(row); kept += 1
        print(f"{g}: {kept} passages", flush=True)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "proofs.csv", index=False)
    np.savez_compressed(out / "states.npz", **{k: np.stack(v) for k, v in states.items()})


if __name__ == "__main__":
    main()
