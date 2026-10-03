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

### Result 1b — The same test, made stricter
In the table above, a passage's neighbours from the same stretch of text could help the guess. The stricter version splits every work by position:
the guesser learns from passages of one half and must guess passages of the **other** half, with passages near the split dropped (Poe is split by tale,
so a tale is never both learned and tested). Same measures, corrected p-values (Holm, 24 tests):

| Measure | French (5 works) | English (4 works) | Proofs (7 works) |
|---|---|---|---|
| **Path distance (Qwen)** | **58%** (was 57%) | **80%** (was 85%) | **64%** (was 75%) |
| Path distance (XLM-R) | 53% (58%) | 63% (65%) | 45% (67%) |
| Surprise | 55% (52%) | 45% (58%) | 21% (27%), not significant |
| Step length (Qwen) | 33% (32%) | 42% (47%) | 23% (30%) |
| Dimension (XLM-R) | 35% (37%) | 33% (29%), not significant | 28% (27%) |
| Dimension (Qwen) | 25% (25%), not significant | 42% (42%) | 23% (27%) |
| Different words | 34% (32%) | 23% (21%), not significant | 18% (21%), not significant |
| *Random guessing* | *20%* | *25%* | *14%* |

(in brackets: the earlier test on the same passages). The ranking does not change: **path distance with Qwen stays best in all three families**, and dimension stays weak.
What did change: the proofs lost the most (path distance 75% to 64% with Qwen, 67% to 45% with XLM-R), so part of the earlier proof result came from
neighbouring passages resembling each other. The 95% ranges are about ±8 points (full list in `examples/honest_test.txt`). They assume passages are independent,
which is not quite true, so treat them as slightly optimistic.

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

## Result 8 — Dimension of a whole work, at 250 to 3000 points
Instead of one passage, we pooled all the points of a work (22 passages x 256 = 5,632) and measured dimension on random samples of 250, 500, 1000, 2000 and 3000 points.
Average over the 16 works:

| Points | 250 | 500 | 1000 | 2000 | 3000 |
|---|---|---|---|---|---|
| Qwen | 12.2 | 11.6 | 11.1 | 11.0 | 11.0 |
| XLM-R | 15.2 | 13.3 | 12.3 | 11.8 | 11.8 |

* **More points do not double the dimension.** It falls a little and levels off from about 2000 points. Qwen is almost flat (12.2 to 11.0).
* Two random draws of the same size differ by about 0.9 on average, so differences below about 1 are noise.
* At 3000 points the ordering is stable (XLM-R): rich literary vocabulary at the top (Mallarmé 15.2, Poe in English 15.0, Baudelaire 14.9, Joyce 14.8), plain proofs at the bottom
  (Euclid 8.5, Dedekind 8.5, Andrews ch.1 9.0, Hilbert 9.0, Poe in French 9.1). Ramanujan 1918 is in the middle (12.1), your paper 10.0.
  Qwen compresses the differences (8.8 to 12.3).
Per-work numbers: `examples/pooled_dimension.csv`.

## Result 9 — Can the measures tell a human text from one written by an AI?
We asked Claude (the AI writing this report) to write about 7,500 tokens in the style of three of the works: Shakespearean sonnets, Baudelaire-style French poems,
and Euclid-style geometry propositions (`data/generated/`). Passages were then cut and measured exactly like the real texts. The test is the strict one from Result 1b:
learn on one half of each text, guess passages of the other half, human against Claude. Random guessing is 50%.

| Measure | Sonnets | Baudelaire | Euclid |
|---|---|---|---|
| **Path distance (Qwen)** | **95%** | **100%** | **86%** |
| **Surprise** (how predictable) | 88% | 95% | 74% |
| Step length (Qwen) | 75% | 67%, not significant | 64%, not significant |
| Path distance (XLM-R) | 50% | 55% | 71%, not significant |
| Dimension (Qwen or XLM-R) | 55-60%, not significant | 55-57%, not significant | 52-55%, not significant |
| Different words | 60%, not significant | 36%, not significant | 52%, not significant |

* **Dimension and vocabulary richness cannot tell the two apart.** The AI text has the same number of different words and about the same dimension as the human text (e.g. sonnets: 160 against 163 different words; Qwen dimension 10.2 against 10.3).
* **The path through the model's space and the predictability can.** The AI text is clearly more predictable to the model (surprise 3.56 against 4.13 for sonnets, 3.00 against 3.44 for Baudelaire, 1.86 against 2.19 for Euclid), which is what one expects from fluent, generic writing.
* **In style, the AI text still sits next to the work it imitates.** Asked "which work is this from?" with only the real works to choose from, path distance (Qwen) puts all 22 Claude sonnet passages with Shakespeare's sonnets, all 22 Euclid-style passages with Euclid, and the Baudelaire-style passages with Baudelaire (12) or Mallarmé (10).
  The simple measures together are less kind: they call the sonnets "Poe" because the AI text is more predictable than any real author.
* **Cautions.** (1) One AI, one author of the fake text, written by a model that knew what was being measured. (2) Each fake text is one continuous piece with its own layout (headings, blank lines), and the real texts come from Project Gutenberg with their own layout; part of the path-distance result could come from layout and not from the writing. We did not test this. (3) Only 3 works and about 40 passages per pair.
Full output: `examples/generated_test.txt`.

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
| Pooled dimension of a whole work | dimension of all a work's points together, at 250 to 3000 points | done (Result 8) |
| Human against AI-written text | can the measures tell a human text from an AI's imitation | done (Result 9) |
| Not done | how often the path returns to a place; dimension layer by layer | possible next steps |

## What we cannot conclude
* Only 22 passages per work (10 in the prose-only version of the proofs); passages of one work resemble each other, so the guessing rates of Result 1 are optimistic. Result 1b removes the largest part of that, and its numbers are the ones to quote.
* The works differ in topic (geometry, number theory, logic), which affects vocabulary.
* Some texts come from scans or conversions with garbled formulas (the 1918 paper, Andrews); texts from LaTeX have gaps where formulas were removed.
* The p-values of Result 1b are corrected for the 24 tests run; the p-values quoted in Results 6 and 7 are not.
* Nothing here measures quality, depth or creativity; it shows only where the model places the words.
* Tried and dropped: feeding each word alone to the model (the "first-token" idea). With XLM-R it collapses (all words almost identical); with Qwen it never beat the other methods.

## Where to look
Code in `scripts/`, per-passage numbers and full outputs in `examples/`, the running log of every experiment in `RESULTS.md`,
and `README.md` for how the three ways of reading differ, with real numbers.
