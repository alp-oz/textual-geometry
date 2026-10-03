"""Layout check, step 1: give Claude's texts the same layout as the real ones, so that layout cannot give them away.
Poems: headings (roman numeral and title) removed, blank lines inside a poem removed, poems separated by one blank line
(this is how the real sonnets and Baudelaire were prepared). Euclid-style proofs and the real Euclid: all whitespace
collapsed to single spaces (the real text is wrapped at 80 columns, Claude's is not).
    python scripts/layout_clean.py OUT_DIR"""
import re
import sys
from pathlib import Path

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
G = Path("data/generated")


def poems(path, n_head):
    blocks = re.split(r"\n\n(?=[IVXLC]+\n)", Path(path).read_text(encoding="utf-8").strip())
    res = []
    for b in blocks:
        lines = [l.strip() for l in b.strip().split("\n")]
        assert re.fullmatch(r"[IVXLC]+", lines[0]), lines[0]
        res.append("\n".join(l for l in lines[n_head:] if l))
    return "\n\n".join(res)


(out / "sonnets_claude.txt").write_text(poems(G / "sonnets_claude.txt", 1), encoding="utf-8")
(out / "baudelaire_claude.txt").write_text(poems(G / "baudelaire_claude.txt", 2), encoding="utf-8")
flat = lambda p: re.sub(r"\s+", " ", Path(p).read_text(encoding="utf-8")).strip()
(out / "euclid_claude.txt").write_text(flat(G / "euclid_claude.txt"), encoding="utf-8")
(out / "euclid_human.txt").write_text(flat("data/proofs_raw/euclid.txt"), encoding="utf-8")
for f in sorted(out.glob("*.txt")):
    print(f.name, len(f.read_text(encoding="utf-8").split()), "words")
