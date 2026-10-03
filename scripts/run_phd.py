"""Compute PHD per chunk for each text file under data/<author>/*.txt.

    python scripts/run_phd.py --data data --out results.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from textgeom import chunk_tokens, load_text, phd
from textgeom.embed import Embedder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--model", default="xlm-roberta-base")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--max-chunks", type=int, default=100, help="per file")
    args = ap.parse_args()

    emb = Embedder(args.model)
    rows = []
    for path in sorted(Path(args.data).glob("*/*.txt")):
        author = path.parent.name
        ids = emb.token_ids(load_text(path))
        chunks = chunk_tokens(ids, args.chunk, args.chunk // 2)
        if len(chunks) > args.max_chunks:  # sample evenly across the whole file
            keep = np.linspace(0, len(chunks) - 1, args.max_chunks).astype(int)
            chunks = [chunks[k] for k in keep]
        for i, c in enumerate(chunks):
            d = phd(emb.embed_ids(c))
            rows.append((author, path.name, i, len(c), d))
            if i % 20 == 0:
                print(f"  {path.name} chunk {i}/{len(chunks)} phd={d:.2f}", flush=True)
        vals = [r[4] for r in rows if r[0] == author and r[1] == path.name]
        if vals:
            print(f"{author}/{path.name}: n={len(vals)} mean PHD={np.nanmean(vals):.2f} sd={np.nanstd(vals):.2f}")

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["author", "file", "chunk", "n_tokens", "phd"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
