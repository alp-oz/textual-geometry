"""Cut Poe's tales out of the Gutenberg 'Works of Edgar Allan Poe' volumes (English originals)
and keep only the tales that also exist in Baudelaire's French translation (data/poe_fr/tales.txt)."""
import re
import subprocess
import sys
from pathlib import Path

cache = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp")
VOLS = {2147: ["THE GOLD-BUG", "THE MURDERS IN THE RUE MORGUE"],
        2148: ["THE FALL OF THE HOUSE OF USHER", "THE PIT AND THE PENDULUM", "THE PURLOINED LETTER"],
        2149: ["LIGEIA", "MORELLA"]}
ORDER = ["THE FALL OF THE HOUSE OF USHER", "LIGEIA", "MORELLA", "THE PIT AND THE PENDULUM",
         "THE GOLD-BUG", "THE MURDERS IN THE RUE MORGUE", "THE PURLOINED LETTER"]
TITLE_LINE = re.compile(r"\n\n\n+\s*([A-ZÀ-ÖØ-Þ][A-ZÀ-ÖØ-Þ0-9 ,.'’()*\-\[\]]{5,})\s*\n\n")
out = {}
for vid, titles in VOLS.items():
    f = cache / f"poe{vid}.txt"
    if not f.exists():
        subprocess.run(["curl", "-sS", "-L", "-m", "120", "-o", str(f), f"https://www.gutenberg.org/ebooks/{vid}.txt.utf-8"])
    t = f.read_text(encoding="utf-8", errors="ignore").replace("\r", "")
    for title in titles:
        starts = [m.end() for m in re.finditer(r"^\s*" + re.escape(title) + r"\s*$", t, flags=re.M)]
        s = starts[-1]                      # the first hits are in the table of contents
        nxt = TITLE_LINE.search(t, s + 300)
        e = nxt.start() if nxt else len(t)
        body = re.sub(r"\n{3,}", "\n\n", t[s:e]).strip()
        out[title] = re.sub(r"[ \t]*\n[ \t]*", "\n", body)
fr = {b.split("\n", 1)[0]: b for b in Path("data/poe_fr/tales.txt").read_text(encoding="utf-8").split("\n=====\n")}
fr_names = {"THE FALL OF THE HOUSE OF USHER": "La_Chute_de_la_maison_Usher", "LIGEIA": "Ligeia", "MORELLA": "Morella",
            "THE PIT AND THE PENDULUM": "Le_Puits_et_le_pendule", "THE GOLD-BUG": "Le_Scarabée_d’or",
            "THE MURDERS IN THE RUE MORGUE": "Double_assassinat_dans_la_rue_Morgue", "THE PURLOINED LETTER": "La_Lettre_volée"}
en_blocks, fr_blocks = [], []
print(f"{'tale':34s} {'EN words':>8s} {'FR words':>8s} {'FR/EN':>6s}")
for ti in ORDER:
    f = fr[fr_names[ti]].split("\n", 1)[1]
    print(f"{ti:34s} {len(out[ti].split()):8d} {len(f.split()):8d} {len(f.split())/len(out[ti].split()):6.2f}")
    en_blocks.append(f"{ti}\n{out[ti]}"); fr_blocks.append(f"{fr_names[ti]}\n{f}")
Path("data/poe_en/tales.txt").write_text("\n=====\n".join(en_blocks), encoding="utf-8")
Path("data/poe_fr/tales.txt").write_text("\n=====\n".join(fr_blocks), encoding="utf-8")
