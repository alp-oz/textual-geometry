"""Fair, matched comparison. Every group: N windows of exactly L tokens, sampled over the whole work,
each window read by the causal LM from its own first token. Per window we store the token states and
the model's surprise. Poe tales (EN and Baudelaire's FR) use the same (tale, relative position) windows.

    python scripts/run_fair.py --out-dir DIR [--n 80] [--L 128]
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from textgeom import load_text, phd
from textgeom.corpus import split_baudelaire, split_delimited, split_sonnets

HEADERS = [r"^\s*\d+\s+.*Chap\.\s*\d+(\.\d+)?\s*$", r"^\s*\d+\.\d+\s+.*\s\d+\s*$", r"^\s*\d+\s*$",
           r"^.*Cambridge Books Online.*$", r"^\s*Sec\.\s*\d+.*$"]


def clean_andrews(t):
    return "\n".join(l for l in t.split("\n") if not any(re.match(h, l) for h in HEADERS))


def sources(D):
    cw = load_text(f"{D}/shakespeare/complete_works.txt")
    beck = load_text(f"{D}/beckett/linnommable_fr.txt")
    i = beck.find("Où maintenant")
    beck = beck[i:] if i != -1 else beck[10000:]
    S = {
        "Shakespeare sonnets": "\n\n".join(t for _, t in split_sonnets(f"{D}/shakespeare/sonnets.txt")),
        "Shakespeare plays": cw[250000:],
        "Baudelaire poems": "\n\n".join(t for _, t in split_baudelaire(f"{D}/baudelaire/fleurs_du_mal.txt")),
        "Mallarmé poems": "\n\n".join(t for _, t in split_delimited(f"{D}/mallarme/poesies.txt")),
        "Beckett L'Innommable": beck,
        "Ramanujan proof (Andrews ch.5)": clean_andrews(load_text(f"{D}/ramanujan/andrews_ch5_hrr_expansion.txt")),
        "Math ch.1 (elementary)": clean_andrews(load_text(f"{D}/ramanujan/andrews_ch1_elementary.txt")),
        "Math ch.13 (combinatorics)": clean_andrews(load_text(f"{D}/ramanujan/andrews_ch13_combinatorics.txt")),
    }
    return S


def poe(D):
    f = lambda lang: [b.split("\n", 1)[1] for b in Path(f"{D}/poe_{lang}/tales.txt").read_text(encoding="utf-8").split("\n=====\n")]
    return f("en"), f("fr")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shuffle", action="store_true", help="control: shuffle words inside each window's text")
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    rng = np.random.default_rng(0)
    L, N = a.L, a.n

    # ---- choose windows (token id lists) per group
    W = {}
    for g, text in sources(a.data).items():
        ids = tok(text, add_special_tokens=False)["input_ids"]
        starts = np.arange(0, len(ids) - L + 1, L)                    # non-overlapping windows over the whole work
        pick = np.sort(rng.choice(starts, size=min(N, len(starts)), replace=False))
        W[g] = [(f"{g}@{s}", ids[s:s + L], -1) for s in pick]
        print(f"{g}: {len(ids)} tokens -> {len(starts)} possible windows, using {len(pick)}", flush=True)
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
    print("Poe: tale sizes (EN tokens)", [len(x) for x in ids_en], "(FR tokens)", [len(x) for x in ids_fr])

    # ---- run the model, one window at a time
    rows, states = [], {}
    shuf = np.random.default_rng(1)
    for g, wins in W.items():
        S = []
        for label, ids, pair in wins:
            if a.shuffle:
                words = tok.decode(ids).split()
                ids = tok(" ".join(shuf.permutation(words)), add_special_tokens=False)["input_ids"][:L]
                if len(ids) < L:
                    continue
            with torch.no_grad():
                o = model(input_ids=torch.tensor([ids]), output_hidden_states=True)
            H = o.hidden_states[-1][0].numpy()[1:]                    # drop token 0: attention-sink outlier
            lp = torch.log_softmax(o.logits[0, :-1], -1).gather(1, torch.tensor(ids[1:])[:, None])[:, 0]
            step = np.linalg.norm(np.diff(H, axis=0), axis=1)
            rows.append(dict(group=g, label=label, pair=pair, dim=phd(H, min_n=8, n_sizes=6, n_draws=7, n_fits=3, seed=0),
                             mean_step=step.mean(), spread=np.linalg.norm(H - H.mean(0), axis=1).mean(),
                             surprise=float(-lp.mean())))
            S.append(H.astype(np.float16))
        states[g] = np.stack(S)
        print(f"{g}: done ({len(S)} windows)", flush=True)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "windows.csv", index=False)
    np.savez_compressed(out / "states.npz", **{g: v for g, v in states.items()})


if __name__ == "__main__":
    main()
