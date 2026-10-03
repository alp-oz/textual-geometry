"""Compare PHD distributions from results.csv and plot them."""
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "results.csv")
df["group"] = df.author + "/" + df.file.str.replace(".txt", "", regex=False)
print(df.groupby("group").phd.agg(["count", "mean", "std", "median"]).round(2))

def cmp(a, b):
    x, y = df[df.group == a].phd, df[df.group == b].phd
    p = mannwhitneyu(x, y).pvalue
    print(f"{a} vs {b}: mean diff {x.mean() - y.mean():+.2f}, Mann-Whitney p={p:.2g}")

cmp("shakespeare/complete_works", "beckett/linnommable_fr")
for g in df[df.author == "ramanujan"].group.unique():
    if "ch5" not in g:
        cmp("ramanujan/andrews_ch5_hrr_expansion", g)

order = list(df.groupby("group").phd.mean().sort_values().index)
fig, ax = plt.subplots(figsize=(8, 4))
ax.boxplot([df[df.group == g].phd for g in order], vert=False, tick_labels=order)
ax.set_xlabel("PHD per 256-token chunk (XLM-RoBERTa-base)")
fig.tight_layout(); fig.savefig("phd_comparison.png", dpi=150)
