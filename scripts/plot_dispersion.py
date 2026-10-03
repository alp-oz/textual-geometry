"""Plot where fixed texts sit in embedding space (input: run_dispersion.py output)."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

z = np.load(sys.argv[1], allow_pickle=True)
out = sys.argv[2] if len(sys.argv) > 2 else "examples/dispersion.png"
X, G, T = z["pts"].astype(np.float64), z["grp"], z["txt"]
groups = list(dict.fromkeys(G))
rng = np.random.default_rng(0)
idx = {g: np.where(G == g)[0] for g in groups}

# PCA fitted on an equal number of tokens per group
n = min(len(v) for v in idx.values())
fit = np.concatenate([rng.choice(v, n, replace=False) for v in idx.values()])
mu = X[fit].mean(0)
_, S, Vt = np.linalg.svd(X[fit] - mu, full_matrices=False)
P = (X - mu) @ Vt[:2].T
ev = (S**2 / (S**2).sum())[:2]

print(f"PCA fitted on {n} tokens per group; PC1 {ev[0]:.0%}, PC2 {ev[1]:.0%} of variance")
tot = ((X[fit] - mu) ** 2).sum()
gm = {g: X[idx[g]].mean(0) for g in groups}
bet = sum(n * ((gm[g] - mu) ** 2).sum() for g in groups)
print(f"share of all variance explained by which group a token belongs to (full space): {bet / tot:.0%}")
print(f"\n{'group':34s} {'tokens':>6s} {'texts':>5s} {'spread':>7s} {'dist of centre to overall centre':>33s} {'mean PC1':>9s}")
for g in groups:
    v = idx[g]
    spread = np.linalg.norm(X[v] - gm[g], axis=1).mean()
    print(f"{g:34s} {len(v):6d} {len(set(T[v])):5d} {spread:7.0f} {np.linalg.norm(gm[g] - mu):33.0f} {P[v, 0].mean():9.1f}")
print("\ndistance between group centres (full space):")
for a in groups:
    print(f"{a[:28]:28s}", " ".join(f"{np.linalg.norm(gm[a] - gm[b]):5.0f}" for b in groups))

# ---- figure: small multiples
GRAY, BLUE, INK, MUTED = "#c3c2b7", "#2a78d6", "#0b0b0b", "#52514e"
xl, yl = np.percentile(P[:, 0], [0.5, 99.5]), np.percentile(P[:, 1], [0.5, 99.5])
bg = np.concatenate([rng.choice(v, min(len(v), 1500), replace=False) for v in idx.values()])
fig, axes = plt.subplots(2, 4, figsize=(16, 7.6), sharex=True, sharey=True)
for ax, g in zip(axes.flat, groups):
    ax.scatter(P[bg, 0], P[bg, 1], s=2, c=GRAY, alpha=0.35, linewidths=0, rasterized=True)
    v = rng.choice(idx[g], min(len(idx[g]), 3000), replace=False)
    ax.scatter(P[v, 0], P[v, 1], s=3, c=BLUE, alpha=0.6, linewidths=0, rasterized=True)
    ax.set_title(f"{g}\n{len(idx[g])} tokens, {len(set(T[idx[g]]))} text(s)", fontsize=9, color=INK)
ax = axes.flat[len(groups)]
offs = [(6, 6), (-8, 12), (6, -14), (6, -2), (6, -14), (6, 6), (-30, 8)]  # staggered so labels do not collide
for g, o in zip(groups, offs):
    c = P[idx[g]].mean(0)
    ax.scatter(*c, s=60, c=BLUE, zorder=3, edgecolors="white", linewidths=1.5)
    ax.annotate(g.split(" (")[0], c, xytext=o, textcoords="offset points", fontsize=8, color=INK)
ax.set_title("Centre of each group", fontsize=9, color=INK)
for ax in axes.flat:
    ax.set_xlim(xl); ax.set_ylim(yl); ax.set_xticks([]); ax.set_yticks([])
    for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.suptitle("Where the tokens of fixed texts sit in the model's space (2 largest directions of 896; each dot = one token's point)",
             fontsize=11, color=INK)
fig.text(0.5, 0.01, f"x = direction 1 ({ev[0]:.0%} of variance), y = direction 2 ({ev[1]:.0%}); gray = tokens of all texts, blue = this group",
         ha="center", fontsize=9, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 0.95)); fig.savefig(out, dpi=140)
