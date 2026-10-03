"""Worked example: what the trajectory / Steele measures see on 4 lines of Sonnet 18."""
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import squareform, pdist
from transformers import AutoModel, AutoTokenizer

TEXT = ("Shall I compare thee to a summer's day?\nThou art more lovely and more temperate:\n"
        "Rough winds do shake the darling buds of May,\nAnd summer's lease hath all too short a date;")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model = AutoModel.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.float32).eval()
ids = tok(TEXT, add_special_tokens=False)["input_ids"]
with torch.no_grad():
    H = model(input_ids=torch.tensor([ids])).last_hidden_state[0].numpy()
words = [tok.decode([i]) for i in ids]
X, words = H[1:], words[1:]            # drop token 0 (attention-sink outlier)
n = len(X)
D = squareform(pdist(X)); T = minimum_spanning_tree(D).tocoo()
step = np.linalg.norm(np.diff(X, axis=0), axis=1)
U = np.diff(X, axis=0) / step[:, None]
turn = (U[1:] * U[:-1]).sum(1)

print(f"{n} tokens (token 0 dropped):", " | ".join(repr(w) for w in words))
print("\nstep lengths |x_k - x_{k-1}| (state moved by token k):")
print(" ".join(f"{w.strip() or '_'}:{s:.0f}" for w, s in zip(words[1:], step)))
print(f"\nmean step {step.mean():.0f}, mean turn cosine {turn.mean():+.2f}, "
      f"straightness {np.linalg.norm(X[-1]-X[0])/step.sum():.3f}")
print(f"\nMST over the {n} points: {n-1} edges, total length E1 = {T.data.sum():.0f}")
for i, j, d in sorted(zip(T.row, T.col, T.data), key=lambda e: e[2])[:8]:
    print(f"  shortest edges: {words[i].strip()!r}(pos {i}) -- {words[j].strip()!r}(pos {j}) length {d:.0f}")
print("  longest edge:", max(T.data).round(0))

rng = np.random.default_rng(0)
print("\nSteele step: MST total length E1 of random subsets of the points")
sizes = [8, 12, 16, 24, n]
logE = []
for m in sizes:
    vals = []
    for _ in range(30 if m < n else 1):
        idx = rng.choice(n, m, replace=False) if m < n else np.arange(n)
        vals.append(minimum_spanning_tree(squareform(pdist(X[idx]))).sum())
    logE.append(np.log(np.mean(vals))); print(f"  n={m:3d} points -> E1 = {np.mean(vals):7.0f}")
s = np.polyfit(np.log(sizes), logE, 1)[0]
print(f"slope of log E1 vs log n = {s:.3f}  ->  d = 1/(1-slope) = {1/(1-s):.1f}  (only {n} points: rough)")

# 2-D picture: PCA of the path, MST edges in grey
Xc = X - X.mean(0); _, _, Vt = np.linalg.svd(Xc, full_matrices=False); P = Xc @ Vt[:2].T
fig, ax = plt.subplots(1, 2, figsize=(15, 6.5))
for a, title in zip(ax, ["Trajectory: arrows = order the tokens were read", "Steele's measure: minimum spanning tree on the same points"]):
    a.scatter(P[:, 0], P[:, 1], s=18, c=np.arange(n), cmap="viridis", zorder=3)
    for k, w in enumerate(words): a.annotate(w.strip() or "_", P[k], fontsize=7, xytext=(3, 3), textcoords="offset points")
    a.set_title(title); a.set_xticks([]); a.set_yticks([])
for k in range(n - 1): ax[0].annotate("", P[k + 1], P[k], arrowprops=dict(arrowstyle="->", color="#999", lw=0.7))
for i, j in zip(T.row, T.col): ax[1].plot(P[[i, j], 0], P[[i, j], 1], color="#d62728", lw=1)
fig.suptitle("Sonnet 18, first 4 lines, shown in the 2 largest directions of a 896-dim space", fontsize=11)
fig.tight_layout(); fig.savefig("examples/sonnet18_example.png", dpi=140)
