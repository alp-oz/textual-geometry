"""Text clean-ups for the proof family: markdown (Marker output), LaTeX source, and a common prose-only filter."""
import re


def clean_marker_md(md: str) -> str:
    t = md
    cut = re.search(r"\nTABLE I\b", t)
    t = t[:cut.start()] if cut else t                       # numerical tables at the end
    t = re.sub(r"!\[\]\([^)]*\)", " ", t)
    t = "\n".join(l for l in t.split("\n") if not l.lstrip().startswith("|"))
    t = re.sub(r"<sup>.*?</sup>", " ", t, flags=re.S)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\\([*_])", r"\1", t)
    t = re.sub(r"[*#_]+", " ", t)
    t = re.sub(r"[ \t]+", " ", t)
    return re.sub(r"\n\s*\n+", "\n\n", t).strip()


def latex_to_prose(t: str) -> str:
    t = t[t.find("\\begin{document}"):] if "\\begin{document}" in t else t
    t = re.sub(r"(?<!\\)%.*", "", t)
    t = re.sub(r"\\begin\{(equation|align|eqnarray|gather|multline|displaymath|math)\*?\}.*?\\end\{\1\*?\}", " ", t, flags=re.S)
    t = re.sub(r"\$\$.*?\$\$", " ", t, flags=re.S)
    t = re.sub(r"\\\[.*?\\\]", " ", t, flags=re.S)
    t = re.sub(r"\\\(.*?\\\)", " ", t, flags=re.S)
    t = re.sub(r"(?<!\\)\$.*?(?<!\\)\$", " ", t, flags=re.S)
    t = re.sub(r"\\footnote\{(?:[^{}]|\{[^{}]*\})*\}", " ", t)
    t = re.sub(r"\\(begin|end)\{(figure|table|tabular|array|center|minipage|tikzpicture)\}.*?\\end\{\2\}", " ", t, flags=re.S)
    for _ in range(3):
        t = re.sub(r"\\(?:emph|textit|textbf|textsc|textrm|text|mbox|section\*?|subsection\*?|subsubsection\*?|chapter\*?|paragraph|title|author)\{([^{}]*)\}", r" \1 ", t)
    t = re.sub(r"\\(begin|end)\{[^}]*\}(\[[^\]]*\])?", " ", t)
    t = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", t)
    t = re.sub(r"\\.", " ", t)
    t = re.sub(r"[{}~]", " ", t).replace("``", '"').replace("''", '"')
    t = re.sub(r"[ \t]+", " ", t)
    return re.sub(r"\n\s*\n+", "\n\n", t).strip()


_WORD = re.compile(r"^[\"'(]*[A-Za-z][A-Za-z'’-]*[.,;:!?)\"']*$")


def prose_only(text: str, min_words: int = 4) -> str:
    """Same rule for every text: keep ordinary words, drop numbers, symbols, point labels (AB, ABC),
    single letters except 'a'/'A'/'I', and lines with fewer than `min_words` ordinary words left."""
    out = []
    for line in text.split("\n"):
        toks = line.split()
        if sum(tok.strip(".,;:()").isdigit() for tok in toks) >= 3:      # reference lists, tables of numbers
            continue
        keep = []
        for tok in toks:
            core = tok.strip("\"'()[].,;:!?")
            if not _WORD.match(tok):
                continue
            if core.isupper() and len(core) >= 2 and core not in ("I", "II", "III", "IV", "THE", "AND", "OF", "TO", "IN", "IS", "A"):
                continue                                                  # labels like AB, ABC (but not headings made of real words)
            if len(core) == 1 and core not in ("a", "A", "I"):
                continue
            keep.append(tok)
        if len(keep) >= min_words:
            out.append(" ".join(keep))
    return "\n".join(out)
