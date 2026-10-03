"""LaTeX source -> prose: math removed, formatting commands unwrapped, footnotes dropped.
    python scripts/latex_to_prose.py IN.tex OUT.txt"""
import re
import sys

t = open(sys.argv[1], encoding="utf-8", errors="ignore").read()
t = t[t.find("\\begin{document}"):] if "\\begin{document}" in t else t
t = re.sub(r"(?<!\\)%.*", "", t)                                            # comments
t = re.sub(r"\\begin\{(equation|align|eqnarray|gather|multline|displaymath|math)\*?\}.*?\\end\{\1\*?\}", " ", t, flags=re.S)
t = re.sub(r"\$\$.*?\$\$", " ", t, flags=re.S)
t = re.sub(r"\\\[.*?\\\]", " ", t, flags=re.S)
t = re.sub(r"\\\(.*?\\\)", " ", t, flags=re.S)
t = re.sub(r"(?<!\\)\$.*?(?<!\\)\$", " ", t, flags=re.S)                   # inline math
t = re.sub(r"\\footnote\{(?:[^{}]|\{[^{}]*\})*\}", " ", t)
t = re.sub(r"\\(begin|end)\{(figure|table|tabular|array|center|minipage|tikzpicture)\}.*?\\end\{\2\}", " ", t, flags=re.S)
for _ in range(3):                                                           # unwrap \cmd{text}
    t = re.sub(r"\\(?:emph|textit|textbf|textsc|textrm|text|mbox|section\*?|subsection\*?|subsubsection\*?|chapter\*?|paragraph|title|author)\{([^{}]*)\}", r" \1 ", t)
t = re.sub(r"\\(begin|end)\{[^}]*\}(\[[^\]]*\])?", " ", t)
t = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", t)                          # remaining commands
t = re.sub(r"\\.", " ", t)
t = re.sub(r"[{}~]", " ", t).replace("``", '"').replace("''", '"')
t = re.sub(r"[ \t]+", " ", t)
t = re.sub(r"\n\s*\n+", "\n\n", t).strip()
open(sys.argv[2], "w", encoding="utf-8").write(t)
print(f"{len(t.split())} words ->", sys.argv[2])
