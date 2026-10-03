# textual-geometry

> **Start here: [`REPORT.md`](REPORT.md)** — the results in plain language. This README explains the method in detail.

How are fixed literary and mathematical texts laid out in the embedding space of language models?
Each text is cut into windows of the same number of tokens, every token becomes a point (a vector of several
hundred numbers), and we measure the geometry of that cloud of points and of the path the points trace through
the text. Texts: Shakespeare (sonnets, plays), Baudelaire, Mallarmé, Beckett (French), Poe's tales (English and
Baudelaire's French translation of the same tales), and mathematics (the Hardy–Ramanujan–Rademacher proof and two
other chapters of Andrews' *The Theory of Partitions*).

## The three tracks: three ways to turn a token into a point

All numbers below are real, from `scripts/readme_examples.py` (output in `examples/readme_example_output.txt`).
The example text is ` Thou art more lovely and more temperate`, in which the word **more** occurs twice.

### Track 3 – fixed vector, no context (the input table)
Every token has one vector that is the same wherever the token occurs: the row of the model's **input lookup table**
(before any layer). The two occurrences of "more" are at distance **0.00**. A text is then a bag of points, one per word
type, repeated whenever the word recurs. Word order and word sense play no role in where the points are; order
enters only if a measure uses the *sequence* of the points (a path in text order). This track needs no model reading
the text and is kept as the baseline.

> **Remark – the first-token embedding was tried and is not pursued further.** The idea: feed each token alone as the
> whole input to the model and take the output (a fixed vector in the model's output space). Two versions were tried.
> *XLM-R*: the vectors collapse (mean pairwise cosine 0.986 between different tokens, 1.000 without the start/end markers),
> so it is useless. *Qwen*: the vectors do not collapse (cosine 0.50) but all have almost the same length (about 243, the
> "first position" effect); it carried some signal beyond the input table in English literature (step length: 0.46 of the
> variance explained vs 0.24 for the table; path-distance gap 0.044 vs 0.015) but never beat the contextual Qwen track.
> Because it never improved on the other tracks and complicates the comparison, it is not used in later tests; the early
> results (`examples/tracks_results.txt`) still contain it for the record. The full 6-group comparison of lone-token
> vectors, input table and contextual vectors is in `examples/lone_results.txt` (code: `scripts/run_lone.py`, `scripts/analyze_lone.py`).

### Track 1 – contextual, running through the text (Qwen2.5-0.5B)
The model reads the text left to right. After each token it has a state (a vector) that summarises everything read so far
and is what it uses to guess the next word. The state at a token depends only on the tokens up to it. The two occurrences
of "more" are at distance **214.7**, as large as the distance between neighbouring words (190.8). Each point is
"what the model has understood so far". The sequence of states is a trajectory.

### Track 2 – contextual, whole chunk (XLM-RoBERTa-base)
The model reads the whole chunk at once, and each token's vector depends on the words on both sides of it. The two
occurrences of "more" are at distance **1.66**, against 3.61 between neighbouring words: context moves a word less here
than in Track 1. This is the recipe of Tulchinskii et al. (NeurIPS 2023).

| the word "more", twice | input table | Track 1 Qwen | Track 2 XLM-R |
|---|---|---|---|
| distance between its two points | 0.00 | 214.7 | 1.66 |

First numbers of the Qwen vector of " more": input table `[0.017, -0.012, 0.001]` (length 0.41), contextual
`[-2.818, -1.723, 3.099]` (length 286.9).
The same words in another order (` lovely Thou temperate more art and more`): the set of points of the fixed tracks is
identical (Steele's total tree length E1 is unchanged: 4.07 and 4.07 for the Qwen input table), while
the path length changes (4.32 → 4.37). For the contextual tracks the set of points itself changes (Qwen E1:
1090.3 → 910.2), because the model rereads the reordered text.

## Metrics
Order-blind (they see only the set of points):
* **Steele dimension** (`src/textgeom/phd.py`): grow a random sample of the points, record how the length of the shortest
  tree connecting them grows with the sample size; the growth rate gives the number of independent directions the
  points spread along (a line is 1, a sheet is 2). Reliable only with a few hundred points per window.
* **spread**: average distance of the points to their centre.
* **distinct tokens**: how many different tokens a window contains (lexical variety). With fixed vectors the dimension
  is almost entirely this number (rank correlation 0.97).
* **centre distance** between two works.

Order-aware (they use the sequence of points):
* **step length**: average distance between consecutive points.
* **path distances** between two windows (`src/textgeom/metrics.py`): Fréchet (shortest leash), DTW (sum of the leash
  lengths), Procrustes (shape after shifting, scaling, rotating). Tested on the worked example of the proposal.
* **surprise** (Qwen only): how wrong the model's next-word guess is, on average.

Controls: the **word-shuffle** (shuffle the words of every window, recompute everything) separates what depends on word
order from what depends on vocabulary; the **language control** (the same Poe tales in English and in French).

## Design that makes the comparison fair
* Every group has the same number of windows and every window exactly 256 tokens (points), cut from random places of the
  whole work and read from its first token. Texts are cleaned of page headers and front matter.
* Windows are compared **inside families** only: French literature (Baudelaire, Mallarmé, Beckett, Poe in French),
  English literature (Shakespeare sonnets and plays, Poe in English), English mathematics, and the Poe tales in both
  languages. Across languages the numbers mix in language and tokenizer effects.

## Reproduce
```
pip install -e ".[embed,dev]" numba pandas matplotlib
pytest
python scripts/run_tracks.py --n 22 --out-dir out/tracks          # tracks, all metrics
python scripts/run_tracks.py --n 22 --shuffle --out-dir out/tracks_shuf
python scripts/analyze_tracks.py out/tracks out/tracks_shuf
```
## Data
Texts go in `data/<author>/*.txt`. **Included in the repository** (public domain): Shakespeare, Baudelaire, Mallarmé, Rimbaud,
Poe's tales in English and in Baudelaire's French translation (fetched by the `scripts/fetch_*.py` scripts).
**Not included** (copyright, or private; git-ignored, so supply your own copy to rerun those parts): Beckett *L'Innommable*,
Andrews' *The Theory of Partitions*, Joyce *Ulysses*, Euclid (Casey), Hilbert, Dedekind, the 1918 Hardy–Ramanujan paper
(`data/proofs_raw/`, `data/prose/`) and the author's own paper (`data/user/`). The per-passage numbers for all works are in `examples/`.

Licence: MIT (code). The texts keep their own status.
Earlier experiments (unmatched, Qwen-only, XLM-R-only, static table) are in `RESULTS.md` and `examples/`.
