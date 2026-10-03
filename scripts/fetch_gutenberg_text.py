"""Fetch a Project Gutenberg book as plain text; fall back to extracting the EPUB when no .txt exists.
    python scripts/fetch_gutenberg_text.py ID OUT.txt"""
import re
import subprocess
import sys
import zipfile
from html import unescape

gid, out = sys.argv[1], sys.argv[2]
tmp = f"/tmp/pg{gid}"
r = subprocess.run(["curl", "-sS", "-L", "-m", "120", "-o", tmp + ".txt", "-w", "%{http_code}", f"https://www.gutenberg.org/ebooks/{gid}.txt.utf-8"], capture_output=True, text=True)
text = None
if r.stdout.strip() == "200":
    text = open(tmp + ".txt", encoding="utf-8", errors="ignore").read()
else:
    subprocess.run(["curl", "-sS", "-L", "-m", "120", "-o", tmp + ".epub", f"https://www.gutenberg.org/ebooks/{gid}.epub.noimages"])
    z = zipfile.ZipFile(tmp + ".epub"); parts = []
    for n in z.namelist():
        if n.endswith((".xhtml", ".html", ".htm")):
            h = z.read(n).decode("utf-8", "ignore")
            h = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", h, flags=re.S)
            h = re.sub(r"</(p|div|h\d|li|tr)>", "\n", h); h = re.sub(r"<br\s*/?>", "\n", h)
            t = unescape(re.sub(r"<[^>]+>", "", h)); parts.append(re.sub(r"\n\s*\n+", "\n\n", re.sub(r"[ \t]+", " ", t)).strip())
    text = "\n\n".join(parts)
open(out, "w", encoding="utf-8").write(text)
m = re.search(r"Title:\s*(.+)", text)
print(gid, "via", "txt" if r.stdout.strip() == "200" else "epub", "|", (m.group(1).strip() if m else "?")[:80], "|", len(text.split()), "words")
