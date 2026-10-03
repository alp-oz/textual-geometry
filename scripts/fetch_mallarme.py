"""Fetch Mallarme's Poesies (1914 ed.) poem by poem from French Wikisource -> data/mallarme/poesies.txt
(one poem per block, blocks separated by a line '=====', first line of a block = title).
Reads the normal page HTML (the API is rate-limited for shared IPs)."""
import re
import subprocess
import time
import urllib.parse
from html import unescape
from html.parser import HTMLParser

UA = "textual-geometry-research/0.1 (research project; github.com/alp-oz/textual-geometry)"
MAIN = "Poésies (Mallarmé, 1914, 8e éd.)"


def get(title):
    url = "https://fr.wikisource.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"), safe="/_(),.")
    for wait in (5, 20, 60):
        r = subprocess.run(["curl", "-sS", "-L", "-m", "60", "-A", UA, url], capture_output=True, text=True)
        if "<html" in r.stdout[:1500].lower() or r.stdout.lstrip().lower().startswith("<!doctype"):
            return r.stdout
        time.sleep(wait)
    raise RuntimeError("cannot fetch " + title)


class Poem(HTMLParser):
    """Collect text inside <div class="poem">."""

    def __init__(self):
        super().__init__()
        self.depth = 0
        self.out = []
        self.blocks = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "div" and "poem" in a.get("class", "").split():
            self.depth, self.out = 1, []
            return
        if self.depth:
            if tag == "div":
                self.depth += 1
            if tag in ("br", "p"):
                self.out.append("\n")

    def handle_endtag(self, tag):
        if self.depth and tag == "div":
            self.depth -= 1
            if self.depth == 0:
                self.blocks.append("".join(self.out))
        if self.depth and tag == "p":
            self.out.append("\n")

    def handle_data(self, d):
        if self.depth:
            self.out.append(d)


index = get(MAIN)
links = []
for h in re.findall(r'href="(?:https://fr\.wikisource\.org)?/wiki/([^"#]+)"', index):
    t = urllib.parse.unquote(h).replace("_", " ")
    if t.startswith(MAIN + "/") and t not in links:
        links.append(t)
print(len(links), "subpages", flush=True)

poems = []
for t in links:
    time.sleep(1.5)
    p = Poem()
    p.feed(get(t))
    txt = "\n".join(re.sub(r"\n{3,}", "\n\n", unescape(b)).strip() for b in p.blocks).strip()
    if len(txt.split()) > 12:
        poems.append(f"{t.split('/')[-1]}\n{txt}")
        print(t.split("/")[-1], len(txt.split()), flush=True)
open("data/mallarme/poesies.txt", "w", encoding="utf-8").write("\n=====\n".join(poems))
print(len(poems), "poems;", sum(len(p.split()) for p in poems), "words")
