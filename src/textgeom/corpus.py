"""Split source files into units ("patients"): one poem / sonnet / chunk each."""
import re
from pathlib import Path

from .chunking import load_text

TITLE = r"[A-ZÉÈÀÂÊÎÔÛÇËÏÜ0-9'’ ,.;:!?\-]{3,}"


def split_baudelaire(path):
    t = load_text(path).replace("\r", "")
    # skip the editor's preface
    t = t[t.find("SPLEEN ET IDÉAL"):] if "SPLEEN ET IDÉAL" in t else t
    parts = re.split(rf"\n\n+(?=  {TITLE}\n\n)", t)
    out = []
    for p in parts:
        lines = [l.strip() for l in p.strip().split("\n")]
        title = lines[0]
        body = " ".join(l for l in lines[1:] if l)
        # section headings (e.g. "SPLEEN ET IDÉAL") come with the next poem's title
        while body.startswith(tuple(["  "])) or re.fullmatch(TITLE, body.split("  ")[0] or ""):
            break
        if len(body.split()) >= 30:
            out.append((title, "\n".join(l for l in lines[1:] if l)))
    return out


def split_sonnets(path):
    t = load_text(path).replace("\r", "")
    parts = re.split(r"\n\n(?=[IVXLC]+\n\n)", t)
    out = []
    for p in parts:
        lines = [l.strip() for l in p.strip().split("\n")]
        if re.fullmatch(r"[IVXLC]+", lines[0]) and len(lines) > 8:
            out.append((f"Sonnet {lines[0]}", "\n".join(l for l in lines[1:] if l)))
    return out


def split_delimited(path, sep="\n=====\n"):
    out = []
    for p in Path(path).read_text(encoding="utf-8").split(sep):
        title, _, body = p.partition("\n")
        out.append((title.strip(), body.strip()))
    return out
