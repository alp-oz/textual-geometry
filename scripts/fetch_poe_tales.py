"""Poe's tales in English and in Baudelaire's French translation (same content, two languages).
Both from Wikisource page HTML -> data/poe_en/tales.txt, data/poe_fr/tales.txt ('=====' between tales)."""
import re
import subprocess
import time
import urllib.parse
from html import unescape

UA = "textual-geometry-research/0.1 (research project; github.com/alp-oz/textual-geometry)"
PAIRS = [  # (English page, French page)
    ("The_Fall_of_the_House_of_Usher", "La_Chute_de_la_maison_Usher"),
    ("The_Tell-Tale_Heart", "Le_Cœur_révélateur"),
    ("The_Black_Cat_(Poe)", "Le_Chat_noir"),
    ("The_Masque_of_the_Red_Death", "Le_Masque_de_la_mort_rouge"),
    ("Ligeia", "Ligeia"),
    ("Berenice", "Bérénice_(Poe)"),
    ("Morella", "Morella"),
    ("The_Pit_and_the_Pendulum", "Le_Puits_et_le_pendule"),
    ("The_Gold-Bug", "Le_Scarabée_d’or"),
    ("The_Murders_in_the_Rue_Morgue", "Double_assassinat_dans_la_rue_Morgue"),
    ("The_Purloined_Letter", "La_Lettre_volée"),
]


def get(lang, title):
    url = f"https://{lang}.wikisource.org/wiki/" + urllib.parse.quote(title, safe="/_()-")
    for wait in (3, 15, 45):
        h = subprocess.run(["curl", "-sS", "-L", "-m", "90", "-A", UA, url], capture_output=True, text=True).stdout
        if "<html" in h[:2000].lower():
            return h
        time.sleep(wait)
    raise RuntimeError("cannot fetch " + url)


def body(html):
    m = re.search(r'<div class="[^"]*mw-parser-output[^"]*"[^>]*>(.*?)<div class="printfooter"', html, flags=re.S) or \
        re.search(r'<div class="[^"]*mw-parser-output[^"]*"[^>]*>(.*)', html, flags=re.S)
    h = m.group(1)
    h = re.sub(r"<(script|style|table|sup)[^>]*>.*?</\1>", " ", h, flags=re.S)
    paras = re.findall(r"<p[^>]*>(.*?)</p>", h, flags=re.S)
    out = []
    for p in paras:
        t = unescape(re.sub(r"<[^>]+>", "", p)).replace("\xa0", " ").strip()
        if len(t.split()) >= 4:
            out.append(re.sub(r"\s+", " ", t))
    return "\n".join(out)


texts = {"en": [], "fr": []}
for en, fr in PAIRS:
    time.sleep(1.5)
    e = body(get("en", en)); time.sleep(1.5); f = body(get("fr", fr))
    print(f"{en:36s} EN {len(e.split()):6d} words | FR {len(f.split()):6d} words | ratio {len(f.split())/max(len(e.split()),1):.2f}", flush=True)
    texts["en"].append(f"{en}\n{e}"); texts["fr"].append(f"{fr}\n{f}")
for lang in ("en", "fr"):
    open(f"data/poe_{lang}/tales.txt", "w", encoding="utf-8").write("\n=====\n".join(texts[lang]))
