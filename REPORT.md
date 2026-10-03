# What the geometry of a text tells us — results in plain language

## The question
When a language model reads a text, each piece of the text (a word or part of a word) becomes a point in a large space.
Do the points of different works — Shakespeare, Baudelaire, Beckett, Ramanujan's proof — lie in recognisably different
places or paths? And does Ramanujan's 1918 proof look different from other proofs?

## How we measured (everything explained once)
* **A passage** = 256 consecutive pieces of text (a few hundred words) cut from a work. Every work has the same number
  of passages, and each passage is read by the model from its own first word.
* **Three ways to turn text into points.**
  *Qwen* reads left to right, like a person reading, and each point is what it has understood so far.
  *XLM-R* reads the whole passage at once. The *input table* gives each word a fixed point and does no reading at all.
* **Five measures.**
  *Different words*: how many different pieces a passage uses (vocabulary richness).
  *Surprise*: how hard the text is for the model to predict (higher = less predictable).
  *Step length*: how far the point jumps from one word to the next.
  *Dimension*: how many independent directions the cloud of points spreads along (a line is 1, a sheet is 2).
  *Path distance*: how different the whole sequence of points of two passages is.
* **The test.** Hide which work a passage came from and try to guess it from a measure. Random guessing would be right
  14–25% of the time (depending on how many works), so anything well above that is real signal.
* **Fair rules.** Same number of passages and points for every work; works are compared only inside their family
  (French literature, English literature, proofs); the same stories are used in two languages to measure the effect of language.

## Result 1 — Only the whole path tells works apart well
How often a passage's work is guessed correctly:

| Measure | French literature (5 works) | English literature (4 works) | Proofs (7 works) |
|---|---|---|---|
| **Path distance** | **57%** | **85%** | **76%** |
| Surprise | 52% | 57% | 27% |
| Step length | 32% | 44% | 30% |
| Dimension | 25% | 50% | 29% |
| Different words | 32% | 30% | 21% |
| *Random guessing* | *20%* | *25%* | *14%* |

Comparing whole paths is by far the best fingerprint. Dimension, our starting point, is one of the weakest.

## Result 2 — Dimension mostly reflects vocabulary, and depends on how many points you use
* With no reading at all (input table), a passage's dimension is almost the same as its number of different words (correlation 0.97).
* The estimate **does not grow with more points; it falls and levels off**. Same passages, XLM-R, sonnets: 25.7 (64 points), 20.2 (128), 15.2 (256).
  Qwen is steadier (11.4 → 10.8 between 128 and 256 points). So only compare dimensions at the same number of points, as we did.

## Result 3 — Word order shows up in step length, not in dimension
Shuffling the words of each passage and measuring again: the steps shrink by about a third for Qwen (−35%) and
by 1.5% even for the input table. The dimension of Qwen and XLM-R rises a little (+1.6). So word order matters, and step length
and path distance are the measures that notice it.

## Result 4 — Language moves everything
The same Poe stories in English and in Baudelaire's French translation sit about 180 apart in the model's space, against 23–68 between
different works *within* one language. A passage is not closer to its own translation than to other passages (p = 0.36). So texts must be
compared within a language.

## Result 5 — Which literary works are easiest to recognise
* French: Baudelaire, Mallarmé and Beckett 68% each; Rimbaud and Poe in French only 27% (Rimbaud is mistaken for Baudelaire, French Poe for Beckett).
* English: Poe 86%, Shakespeare's sonnets 82%, plays 68%, Joyce 50% (mistaken for Shakespeare's plays).

## Result 6 — Ramanujan's 1918 proof against other proofs
| Work | Different words | Surprise | Dimension (XLM-R) | Step length (Qwen) | Recognised |
|---|---|---|---|---|---|
| **Ramanujan 1918** | 140 | **3.46** | **12.4** | **165** | 59% |
| Andrews ch.5 (modern retelling of the same proof) | 135 | 3.21 | 11.6 | 184 | 9% |
| Andrews ch.1 (simple modern proof) | 126 | 2.46 | 9.6 | 194 | 32% |
| Dedekind (1888) | 118 | 2.79 | 10.3 | 183 | 32% |
| Hilbert (1899) | 123 | 2.50 | 10.4 | 186 | 14% |
| Euclid (ancient) | 108 | 2.19 | 7.6 | 195 | 59% |
| Your paper | 143 | 3.12 | 12.1 | 179 | 91% |

* Ramanujan's paper is at the rich, hard-to-predict end: against Euclid, Hilbert, Dedekind and Andrews ch.1 it is less predictable
  (by 0.7–1.3), has more different words (+15 to +32), takes smaller steps and has a higher XLM-R dimension (+2 to +4.8); all differences are clear (p < 0.03).
* A modern retelling of the same proof (Andrews ch.5) is **indistinguishable** from the original: the measures see the content, not the era of the writing.
* Euclid sits at the opposite end (simplest vocabulary, easiest to predict).
* Your paper has the richest vocabulary and the highest Qwen dimension; its paths are closest to Dedekind, Hilbert and Ramanujan 1918.

## Result 7 — Do works spread along the same directions?
For each work we took the 10 directions along which its points spread most, and measured how much two works' sets of directions overlap
(1 = identical, 0.01 = what random directions would give; two halves of the same work overlap 0.77–0.95). Everything overlaps a lot, because all texts
share the model's main directions, but the pattern is meaningful (Qwen):
* French poets: Baudelaire–Mallarmé **0.94** (practically the same directions), Mallarmé–Rimbaud 0.87; Beckett and Poe in French (prose) overlap less with the poets (0.68–0.76).
* Proofs: Ramanujan 1918 overlaps most with Andrews' retelling of the same proof (**0.85**) and ch.1 (0.84), least with Euclid (0.64) and your paper (0.62); Euclid is the most distant from all.
Full matrices: `examples/direction_overlap.txt`.

## All the measures we tried
| Measure | What it asks | Status |
|---|---|---|
| Different words | how rich is the vocabulary | used (weak alone) |
| Surprise | how predictable is the text for the model | used (good for literature) |
| Dimension | how many independent directions the points spread along | used (see Result 2) |
| Spread | how far points lie from their centre | used (weak) |
| Step length | how far the point jumps between consecutive words | used (notices word order) |
| Path distance (DTW) | how different two whole sequences of points are | used (best) |
| Path distance (Fréchet, Procrustes) | other ways to compare two sequences | tried, weaker than DTW |
| Centre distance between works | how far apart the averages of two works are | used |
| Direction overlap | do two works spread along the same directions | new (Result 7) |
| Zig-zag (angle between consecutive steps), straightness | does the path keep going or turn back | dropped: same for every text |
| Common direction, same-word-same-move | early checks from the proposal | early run only |
| Word shuffle, same stories in two languages | controls | used |
| Not done | how often the path returns to a place; dimension layer by layer; dimension of a whole work's cloud | possible next steps |

## What we cannot conclude
* Only 22 passages per work (10 in the prose-only version of the proofs); passages of one work resemble each other, so the guessing rates are somewhat optimistic.
* The works differ in topic (geometry, number theory, logic), which affects vocabulary.
* Some texts come from scans or conversions with garbled formulas (the 1918 paper, Andrews); texts from LaTeX have gaps where formulas were removed.
* p-values are not corrected for the many comparisons made.
* Nothing here measures quality, depth or creativity; it shows only where the model places the words.
* Tried and dropped: feeding each word alone to the model (the "first-token" idea). With XLM-R it collapses (all words almost identical); with Qwen it never beat the other methods.

## Where to look
Code in `scripts/`, per-passage numbers and full outputs in `examples/`, the running log of every experiment in `RESULTS.md`,
and `README.md` for how the three ways of reading differ, with real numbers.
