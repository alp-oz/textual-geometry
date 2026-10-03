"""Real numbers for the README: one line of Sonnet-like text under every kind of vector."""
import numpy as np
import torch
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from transformers import AutoModelForCausalLM, AutoTokenizer
from textgeom.embed import Embedder

TEXT = " Thou art more lovely and more temperate"      # leading space: every word is tokenised the same way wherever it sits
SHUF = " lovely Thou temperate more art and more"          # same words, different order
qtok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
qm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.float32).eval()
xe = Embedder("xlm-roberta-base")
qt_tab = qm.get_input_embeddings().weight.detach().numpy()
xt_tab = xe.model.embeddings.word_embeddings.weight.detach().numpy()

def tracks(text):
    qi, xi = qtok(text, add_special_tokens=False)["input_ids"], xe.token_ids(text)
    with torch.no_grad():
        qc = qm.model(input_ids=torch.tensor([qi])).last_hidden_state[0].numpy()
        qf = np.stack([qm.model(input_ids=torch.tensor([[t]])).last_hidden_state[0, 0].numpy() for t in qi])
        xf = np.stack([xe.embed_ids([t])[0] for t in xi])
    return ({"Qwen contextual": qc, "XLM-R contextual": xe.embed_ids(xi), "Qwen first-token": qf, "XLM-R first-token": xf,
             "Qwen table": qt_tab[qi], "XLM-R table": xt_tab[xi]},
            [qtok.decode([t]) for t in qi], [xe.tok.convert_ids_to_tokens(t) for t in xi])

E1 = lambda X: minimum_spanning_tree(squareform(pdist(X))).sum()
path = lambda X: np.linalg.norm(np.diff(X, axis=0), axis=1).sum()
T, qw, xw = tracks(TEXT)
print("TEXT:", TEXT); print("Qwen tokens  :", qw); print("XLM-R tokens :", xw)
print("\n1) THE SAME WORD AT TWO POSITIONS (positions found by searching for the token)")
for t, X in T.items():
    words = qw if t.startswith("Qwen") else xw
    i, j = [k for k, w in enumerate(words) if w.replace("▁", "").strip() == "more"]
    print(f"   {t:20s} distance between the two 'more' vectors: {np.linalg.norm(X[i] - X[j]):10.2f}   (typical distance between neighbouring words: {np.linalg.norm(np.diff(X, axis=0), axis=1).mean():.2f})")
print("\n2) FIRST THREE NUMBERS OF THE VECTOR OF ' more' (first occurrence), Qwen:")
for t in ("Qwen table", "Qwen first-token", "Qwen contextual"):
    k = [k for k, w in enumerate(qw) if w.strip() == "more"][0]
    print(f"   {t:20s} {np.round(T[t][k][:3], 3)}  length {np.linalg.norm(T[t][k]):.2f}")
S, _, _ = tracks(SHUF)
print(f"\n3) SAME WORDS, TWO ORDERS:\n   A = '{TEXT}'\n   B = '{SHUF}'")
print(f"   {'track':20s} {'E1 (Steele: total MST length)':>34s} {'path length (order-aware)':>30s}")
for t in T:
    print(f"   {t:20s} A {E1(T[t]):9.2f}  B {E1(S[t]):9.2f}   {'same' if abs(E1(T[t]) - E1(S[t])) < 1e-3 * E1(T[t]) else 'different':>9s}   A {path(T[t]):9.2f}  B {path(S[t]):9.2f}")
