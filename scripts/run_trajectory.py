"""Token-by-token trajectories from a causal LM; per-text geometry features.

    python scripts/run_trajectory.py --out trajectory_results.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from run_metrics import build_groups  # noqa: E402
from textgeom.trajectory import trajectory_features  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--n-max", type=int, default=80)
    ap.add_argument("--out", default="trajectory_results.csv")
    ap.add_argument("--shuffle", action="store_true", help="control: shuffle the words within each text")
    ap.add_argument("--save-traj", default=None, help="npz path for the raw trajectories")
    a = ap.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model, torch_dtype=torch.float32).eval()

    rng = np.random.default_rng(0)
    rng_shuf = np.random.default_rng(1)  # separate stream: same texts are selected with or without --shuffle
    groups = build_groups(a.data, rng, a.L, n_windows=a.n_max)
    rows, raw = [], {}
    for g, units in groups.items():
        order = rng.permutation(len(units))
        kept = 0
        for i in order:
            label, text = units[i]
            if a.shuffle:
                text = " ".join(rng_shuf.permutation(text.split()))
            ids = tok(text, add_special_tokens=False)["input_ids"]
            if len(ids) < a.L + 1:
                continue
            with torch.no_grad():
                h = model(input_ids=torch.tensor([ids[: a.L + 1]])).last_hidden_state[0].numpy()
            X = h[1:]  # drop token 0: attention sink with an outlier norm
            rows.append(dict(group=g, label=label, **trajectory_features(X)))
            raw[f"{g}|{label}"] = X.astype(np.float16)
            kept += 1
            if kept >= a.n_max:
                break
        print(f"{g}: {kept} texts", flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    if a.save_traj:
        np.savez_compressed(a.save_traj, **raw)


if __name__ == "__main__":
    main()
