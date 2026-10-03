"""Where do fixed texts sit in the embedding space? Every token of every text -> one point.
Each text is fed to a causal LM whole, from its first word (so every token sees its full history).
"""
import argparse
import re
from pathlib import Path

import numpy as np

from textgeom import load_text
from textgeom.corpus import split_baudelaire, split_delimited, split_sonnets


def passage(path, start_marker=None):
    t = load_text(path)
    if start_marker and start_marker in t:
        t = t[t.find(start_marker):]
    return t


def build(data):
    g = {}
    g["Shakespeare sonnets"] = split_sonnets(f"{data}/shakespeare/sonnets.txt")
    g["Baudelaire poems"] = split_baudelaire(f"{data}/baudelaire/fleurs_du_mal.txt")
    g["Mallarmé poems"] = split_delimited(f"{data}/mallarme/poesies.txt")
    cw = load_text(f"{data}/shakespeare/complete_works.txt")
    i = cw.find("Elsinore. A platform before the Castle")
    g["Shakespeare Hamlet (opening)"] = [("Hamlet Act 1", cw[i:i + 9000])]
    g["Beckett L'Innommable (opening)"] = [("Innommable", passage(f"{data}/beckett/linnommable_fr.txt")[3000:12000])]
    g["Ramanujan proof (Andrews ch.5)"] = [("ch5", passage(f"{data}/ramanujan/andrews_ch5_hrr_expansion.txt")[:9000])]
    g["Other math (Andrews ch.1)"] = [("ch1", passage(f"{data}/ramanujan/andrews_ch1_elementary.txt")[:9000])]
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model, dtype=torch.float32).eval()
    pts, grp, txt, pos = [], [], [], []
    names = []
    for g, units in build(a.data).items():
        n_tok = 0
        for ti, (label, text) in enumerate(units):
            ids = tok(text, add_special_tokens=False)["input_ids"][: a.max_tokens]
            if len(ids) < 20:
                continue
            with torch.no_grad():
                h = model(input_ids=torch.tensor([ids])).last_hidden_state[0].numpy()[1:]  # drop token 0 (sink)
            pts.append(h.astype(np.float32)); n_tok += len(h)
            grp += [g] * len(h); txt += [f"{g}|{label}"] * len(h); pos += list(range(1, len(ids)))
        print(f"{g}: {n_tok} tokens", flush=True)
    np.savez_compressed(a.out, pts=np.concatenate(pts), grp=np.array(grp), txt=np.array(txt), pos=np.array(pos))


if __name__ == "__main__":
    main()
