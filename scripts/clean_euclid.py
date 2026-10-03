"""Euclid (Casey's edition) -> propositions and their proofs only.
Drops the exercises, the observations/scholia/analyses (commentary) and everything before the first proposition."""
import re
import sys

src, dst = sys.argv[1], sys.argv[2]
paras = [p.strip() for p in re.split(r"\n\s*\n", open(src, encoding="utf-8").read()) if p.strip()]
keep, state, dropped = [], "keep", {"exercises": 0, "commentary": 0}
for p in paras:
    if p.startswith("PROP."):
        state = "keep"
    elif p.startswith(("Exercises", "EXERCISES", "Questions for Examination")):
        state = "exercises"
    elif p.startswith(("Obs", "Scholium", "Analysis", "Note", "Remark")):
        state = "commentary"
    elif p.startswith(("Dem.", "Sol.", "Cor.", "Def.", "BOOK")):
        state = "keep"
    if state == "keep":
        keep.append(p)
    else:
        dropped[state] += len(p.split())
text = "\n\n".join(keep)
open(dst, "w", encoding="utf-8").write(text)
print(f"kept {len(text.split())} of {len(' '.join(paras).split())} words; dropped {dropped}")
