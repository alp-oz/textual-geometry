# Results: proposal metrics applied to texts

Patient = one text unit (poem / sonnet / 1200-char window) cut to its first L=96 tokens; path = its 96
XLM-RoBERTa-base token embeddings. N = 80 patients per group (Mallarmé 58). Script: `scripts/run_metrics.py`,
raw output `metrics_results.json`. Groups: Shakespeare sonnets and plays, Baudelaire (Les Fleurs du mal),
Mallarmé (Poésies), Beckett (L'Innommable, FR), Ramanujan proof (Andrews ch. 5), other math (Andrews ch. 1, 13).

## Caveats first
- **Bidirectional encoder.** "Embedding after k tokens" also depends on later tokens, so paths are not causal histories.
- **Steele dimension needs many points.** Simulation (uniform data in a d-dim subspace of R^768):
  N=80 gives d=5→4.9, 10→10.2, 20→31±15, 40→122±58. So the across-patient estimates here (13–45) mean
  only "roughly 15 or more" and must not be compared across groups. The shuffled-coordinates control is not
  estimable (slope ≥ 1: ~768 effective dimensions).
- Languages differ between groups (Beckett, Baudelaire, Mallarmé French; others English).

## Findings
- **Q1 common direction:** R_k ≈ 1/√n at k=4,16,48 (≈0.11 vs 0.11): no shared "clock" direction. At the last position
  (k=95) R≈0.22–0.27: a chunk-boundary effect (next to the end token), not time.
- **Q2 same event, same move?** Same token at the same position: centred cosine 0.50 vs −0.005 for different
  tokens (shuffle null −0.0002 ± 0.0008). The embedding is largely token-driven.
- **Q4 RC:** 0.15–0.25, near or above 1/√(2d) at the (unreliable) estimated d: distances are not degenerate.
- **Q5 analogue, paths (within-group minus between-group mean distance, label-shuffle p):**
  Fréchet −0.24 (p<0.001), DTW −20.3 (p<0.001), Procrustes ≈ 0 (p=0.07). Texts of one group have closer paths than
  texts of different groups under Fréchet/DTW; path *shape* (Procrustes) does not separate groups.
  Mean DTW within group: sonnets 391, plays 396, Mallarmé 402, Baudelaire 414, Beckett 415, Ramanujan ch.5 428, other math 449.
  Baudelaire poems are closer to Mallarmé (412) than to each other (414); the two math groups are as close to each
  other (442) as to themselves (428/449). Language/register probably matters as much as author.
- **Symbolic poems** (DTW centrality percentile in own group, 0 = most central): Baudelaire L'Albatros 0.28,
  La Vie antérieure 0.16, Les Chats 0.65, Élévation 0.90; Mallarmé Brise marine 0.76, Hérodiade 0.74,
  Don du poème 0.83, Sainte 0.86, Sonnet en -yx 0.62. Mallarmé's symbolic poems sit toward the periphery of his group; no such pattern for Baudelaire.

## Trajectories (causal LM, Qwen2.5-0.5B): `scripts/run_trajectory.py`, `trajectory_results.csv`
Each text = path of 128 token states (state after reading k tokens; token 0 dropped). 80 texts per group (Mallarmé 52).
Dimension of the first k points of the path (Steele/MST): estimates at k=32 are unreliable (too few points).

| group | dim@64 | dim@96 | dim@128 | mean step | turn cos |
|---|---|---|---|---|---|
| Shakespeare sonnets | 15.1 | 11.7 | 11.6 | 202 | -0.40 |
| Mallarmé | 16.0 | 11.5 | 11.4 | 246 | -0.43 |
| Baudelaire | 14.4 | 11.3 | 11.3 | 250 | -0.43 |
| Beckett (FR) | 11.5 | 10.5 | 11.0 | 227 | -0.39 |
| Shakespeare plays | 9.8 | 9.5 | 10.7 | 215 | -0.40 |
| Ramanujan proof (Andrews ch.5) | 10.0 | 9.3 | 9.2 | 180 | -0.42 |
| Other math | 8.6 | 8.5 | 8.6 | 186 | -0.42 |

Straightness (net displacement / path length) is ~0.01 for all groups: the path is a jagged wander, not a drift.
Consecutive steps tend to reverse (cos ≈ -0.4). Math has the lowest dimension and shortest steps.
Ramanujan ch.5 vs other math: no difference (dim p=0.17); ch.5 vs Beckett: dim p=1e-7.

## Shuffle control (words shuffled within each text; same texts, `--shuffle`, `trajectory_results_shuffled.csv`)
Mean over paired texts, original -> shuffled (522 pairs):

| group | dim@128 | mean step | straightness | turn cos |
|---|---|---|---|---|
| Shakespeare sonnets | 11.6 -> 18.2 | 202 -> 105 | .010 -> .019 | -.40 -> -.44 |
| Shakespeare plays | 10.7 -> 12.2 | 215 -> 129 | .011 -> .016 | -.40 -> -.42 |
| Baudelaire | 11.3 -> 14.2 | 250 -> 178 | .009 -> .011 | -.43 -> -.42 |
| Mallarmé | 11.5 -> 13.8 | 246 -> 182 | .009 -> .012 | -.43 -> -.42 |
| Beckett (FR) | 11.0 -> 11.9 | 227 -> 154 | .010 -> .012 | -.39 -> -.41 |
| Ramanujan ch.5 | 9.2 -> 8.3 | 180 -> 135 | .012 -> .014 | -.42 -> -.42 |
| Other math | 8.6 -> 8.0 | 186 -> 140 | .011 -> .013 | -.42 -> -.41 |

- Step length depends strongly on word order (-69 on average, p<1e-80). Turn cosine does not (-0.43 -> ~-0.42): the zig-zag is a property of the model/tokens.
- Dimension rises for poetry/English when shuffled, falls slightly for math. The per-text correlation of dim@128 before/after shuffling is 0.02 and the sd of the change is 15: per-text dimension is noisy (estimator blow-ups); use group means only.
- "Math lowest" survives shuffling and widens (math ~8, poetry 12-18): vocabulary/tokens, not word order, drives that gap.

## Comparing the measures (trajectory run, original vs shuffled)
| measure | order sensitivity (paired d_z) | separates the 7 groups (epsilon^2) | literary > math (AUC) | per-text corr orig~shuffled |
|---|---|---|---|---|
| dim@128 | +0.12 | 0.23 | 0.72 | 0.02 |
| mean step | -2.75 | 0.78 | 0.93 | 0.63 |
| straightness | +0.72 | 0.21 | 0.30 (i.e. 0.70 reversed) | 0.14 |
| turn cos | -0.21 | 0.29 | 0.60 | 0.14 |

Median dim@128 (original): Mallarmé 11.4, Baudelaire 10.8, Beckett 10.8, sonnets 10.0, plays 9.3, ch.5 8.9, other math 8.6.
(shuffled): Mallarmé 13.3, sonnets 13.3, Baudelaire 12.6, plays 10.9, Beckett 10.2, ch.5 8.0, other math 7.8.
Robust conclusion: math < literature. The order among literary groups is not stable (means vs medians, original vs shuffled).
Not yet shuffle-tested: the across-patient measures (RC, Steele across patients, Frechet/DTW/Procrustes group closeness, symbolic-poem centrality).

## Fair matched run (supersedes the earlier unmatched dispersion comparison)
Design: every group = 60 windows x 128 tokens (7,680 tokens), sampled over the whole work, each window read by Qwen2.5-0.5B
from its own first token; math cleaned of page headers; Beckett cleaned of front matter; comparisons only inside families.
Poe tales: English originals (Gutenberg) vs Baudelaire's French translation (Wikisource), same tales.
Full output: `examples/fair_results.txt`; per-window numbers: `examples/fair_windows.csv`; script: `scripts/run_fair.py`, `scripts/analyze_fair.py`.

| group | dimension | step length | surprise (nats/token) |
|---|---|---|---|
| Shakespeare sonnets | 10.9 | 200 | 4.39 |
| Shakespeare plays | 9.2 | 212 | 4.19 |
| Poe tales (EN) | 10.7 | 213 | 3.58 |
| Baudelaire poems | 11.3 | 249 | 3.70 |
| Mallarmé poems | 11.0 | 242 | 4.21 |
| Beckett L'Innommable | 10.3 | 224 | 3.15 |
| Poe tales (FR, Baudelaire) | 11.5 | 231 | 3.05 |
| Ramanujan proof (Andrews ch.5) | 8.9 | 183 | 3.34 |
| Math ch.1 | 8.5 | 190 | 2.66 |
| Math ch.13 | 8.6 | 181 | 2.91 |

Surprise is not comparable across languages (French is split into more tokens per word).
Language control (same stories EN vs FR): centres 180 apart, versus 23-68 between texts inside one language.
Windows of a passage vs the translation of the same passage: DTW 42,350 vs 42,447 for other passages (p=0.36): no detectable closeness.
Caveats: windows cut from the same work are correlated, so p-values are optimistic; dimension CI is +-0.4 to +-1.2.

## Two models on identical windows (22 windows x exactly 256 points per group; `scripts/run_matched.py --points 256`)
Same text spans for Qwen2.5-0.5B (running state) and XLM-RoBERTa-base (whole-chunk embedding), same estimator.
Full output: `examples/matched256_results.txt`; per-window numbers `examples/matched256_windows.csv`.

| group | dimension, Qwen | dimension, XLM-R |
|---|---|---|
| Shakespeare sonnets | 10.3 | 14.9 |
| Shakespeare plays | 8.6 | 13.3 |
| Poe tales (EN) | 11.2 | 15.8 |
| Baudelaire | 10.7 | 16.5 |
| Mallarmé | 10.4 | 19.2 |
| Beckett | 10.5 | 11.9 |
| Poe tales (FR, Baudelaire) | 10.6 | 12.2 |
| Ramanujan proof (Andrews ch.5) | 8.7 | 11.1 |
| Math ch.1 | 8.8 | 9.4 |
| Math ch.13 | 9.0 | 9.5 |

- Window-by-window rank correlation between the two models: +0.62 (it was +0.38 with 128 points).
- Share of variance explained by group, Qwen vs XLM-R: same stories two languages 0.02 vs 0.21; French literature 0.00 vs 0.41;
  English literature 0.27 vs 0.16; English math 0.01 vs 0.15. XLM-R dimension is the more distinctive one in 3 of 4 families.
- XLM-R: Ramanujan proof above ch.1 and ch.13 (+1.7, p=0.009 / 0.017); ch.1 = ch.13. Qwen sees no difference.
- XLM-R: Mallarmé 19.2 > Baudelaire 16.5 > Poe FR 12.2 ~ Beckett 11.9. Same stories EN 15.8 vs FR 12.2 (p=0.003): language shifts the dimension.
- Caveats: 22 windows per group, per-window sd 1.5-4.5 (XLM-R), p-values not corrected for multiple comparisons.

## Fixed vectors only (no contextual model): `scripts/run_static.py`, `scripts/analyze_static.py`
Each token -> its fixed row of the model's input embedding table (XLM-R and Qwen tables). Window = exactly 256 tokens, 30 windows per group.
Full output: `examples/static_results.txt`, per-window numbers `examples/static_windows.csv`.
- A window's Steele dimension (5-13) is almost entirely its number of distinct tokens: rank correlation +0.97 (Qwen table), +0.96 (XLM-R table).
- Using only the distinct tokens the estimator returns hundreds to thousands (no thin structure: distinct token vectors are scattered nearly equidistantly).
- Both tables give the same ordering: Mallarmé > Baudelaire > English literature > Beckett ~ Poe FR > Ramanujan proof > math ch.1 ~ ch.13.
- Ramanujan proof above ch.1/ch.13 (+1.1 to +1.4, p < 0.025) is already present with fixed vectors, so it follows from vocabulary variety.
- Spread (mean distance to the window centre) differs little between works; it differs by language (French higher).
- Consequence: part or most of the XLM-R contextual dimension differences are probably lexical variety. To test: control for the number of distinct tokens in the contextual runs.

## Three tracks, every metric, shuffle control (`scripts/run_tracks.py`, `scripts/analyze_tracks.py`)
22 windows x exactly 256 points per group, identical windows on all tracks (checked); full output `examples/tracks_results.txt`,
per-window numbers `examples/tracks_windows.csv` and `examples/tracks_windows_shuffled.csv`.
Tracks: Qwen contextual (running), XLM-R contextual (whole chunk), Qwen/XLM-R first-token (token alone as the whole input), Qwen/XLM-R table (reference).

Share of the variance of the Steele dimension explained by which work the window comes from (0 none, 1 all):

| family | Qwen ctx | XLM-R ctx | Qwen 1st | XLM-R 1st | Qwen tbl | XLM-R tbl | distinct tokens alone |
|---|---|---|---|---|---|---|---|
| Same stories, 2 languages | 0.02 | 0.21 | 0.02 | 0.09 | 0.07 | 0.16 | 0.05 |
| French literature | 0.00 | 0.41 | 0.28 | 0.36 | 0.38 | 0.45 | 0.29 |
| English literature | 0.27 | 0.16 | 0.08 | 0.04 | 0.03 | 0.03 | 0.03 |
| English math | 0.01 | 0.15 | 0.07 | 0.09 | 0.07 | 0.03 | 0.04 |

- French literature: fixed vectors (no context) separate the works as well as XLM-R contextual (0.28-0.45 vs 0.41); lexical variety alone explains 0.29.
- English literature: only Qwen's running reading separates the works (0.27), fixed vectors do not (0.03-0.08): this is the one place where context adds beyond vocabulary.
- Word shuffle: contextual dimension rises (Qwen +1.62, XLM-R +1.63; p<0.001). Fixed-vector dimension is unchanged by construction (small residual changes come from tokenisation of the shuffled text).
- Step length falls under shuffling: Qwen contextual -34%, Qwen first-token -4.6%, Qwen table -1.4% (all p<1e-24): real word order has larger steps than random order, and the contextual model amplifies it.
- Path distances (relative gap within vs between works, DTW): French literature Qwen ctx +0.056, Qwen 1st +0.041, XLM-R ctx +0.026, tables +0.017-0.024; shuffling leaves the contextual gaps unchanged but roughly halves the fixed-vector gaps, so the sequence of words carries work-specific information even without a model.
  Fréchet and Procrustes gaps are small. Math chapters are not separable by any path distance.
- XLM-R first-token vectors collapse (mean pairwise cosine 0.986); treat that track as unreliable. Qwen first-token vectors do not (0.50).
- Another session on this repo independently found: XLM-R lone vectors collapse (0.986), Qwen lone 0.51, fixed-vector dimension tracks distinct tokens (+0.89/+0.94), shuffling raises the contextual dimension.

## Final comparison on the agreed families (no first-token track): `scripts/run_new_groups.py`, `scripts/analyze_final.py`
22 passages x 256 tokens per work. Families: French literature (Baudelaire, Mallarmé, Beckett, Poe FR, Rimbaud), English literature
(Shakespeare sonnets and plays, Poe EN, Joyce *Ulysses*), proofs (Hardy-Ramanujan 1918 original, Andrews ch.5 = modern retelling of the same
proof, Andrews ch.1 = simple modern proof, Euclid = Casey's edition). Full output `examples/final_results.txt`.
- Ramanujan 1918 vs Andrews ch.5 (same proof, modern text): no difference on any measure (p 0.28-0.97).
- Ramanujan 1918 vs Andrews ch.1 and vs Euclid: more surprising (+0.95 / +1.06 nats, p<1e-4), smaller steps (-20 / -24, p<0.001), more distinct tokens
  (+14.5 / +20.5), higher XLM-R dimension (+1.8 / +2.8, p<0.02); Qwen dimension no different.
- Guessing the work from a passage (chance 20-25%): path distance (DTW) is the best measure (Qwen 57/85/70% for French/English/proofs);
  dimension, spread and step length are weak (25-50%). No model wins everywhere: Qwen best on English literature (62%), XLM-R on proofs (48%, barely),
  the Qwen input table (no model reading) has the best average (50%).
- Most recognisable works: Baudelaire/Mallarmé/Beckett 68% each, Poe EN 86%, sonnets 82%, Euclid 68%; least: Rimbaud and Poe FR 27% (Rimbaud mistaken
  for Baudelaire), Andrews ch.5 9% (mistaken for the original), Joyce 50% (mistaken for Shakespeare's plays).
- Shuffle control: contextual dimension +1.6 (both models); step length -35% (Qwen), -1.5% (input table).
- Caveats: 22 passages per work; passages of one work are correlated; the 1918 text is OCR with garbled formulas; Euclid is Casey's annotated edition.

## Euclid with exercises and commentary stripped (`scripts/clean_euclid.py`)
Kept the propositions and their proofs (63,424 of 84,882 words; exercises 19,105 and commentary 2,353 words removed). Euclid group re-run (22 passages, normal and
word-shuffled) and merged into the final analysis: `examples/final_results_euclid_clean.txt`. Conclusions unchanged:
Ramanujan 1918 vs Euclid: surprise +1.07 (p<1e-6), step -26.8, different words +23, XLM-R dimension +2.5 (p=0.003); Euclid recognised 73% (was 68%).
Some numbered cases inside proofs remain, and a few exercise-like items may remain.

## Proof family with more eras, the 1918 paper (Marker conversion) and the user's paper (`scripts/run_proofs.py`, `scripts/analyze_proofs.py`)
Seven works: Ramanujan 1918 (Hardy-Ramanujan, Marker conversion, tables removed), Andrews ch.5 (modern retelling of the same proof), Andrews ch.1 (simple modern
proof), Euclid (Casey, exercises/commentary stripped), Hilbert (1899), Dedekind (1888), and the user's own paper (LaTeX, math removed).
Two versions: *raw* (22 passages per work) and *prose-only* (the same filter for every text: numbers, symbols, point labels and formula lines dropped; 10 passages per work
because the shortest texts have 11-13). Full output `examples/proofs_results.txt`; per-passage numbers `examples/proofs_raw_windows.csv`, `examples/proofs_prose_windows.csv`.
- Ramanujan 1918 vs the classical/elementary proofs (Euclid, Hilbert, Dedekind, Andrews ch.1): hardest to predict (surprise +0.7 to +1.3 nats, p<0.001), smallest steps
  (-19 to -31, p<0.01), more different words (+15 to +32, p<0.02), higher XLM-R dimension (+2.0 to +4.8, p<0.03). No difference from Andrews ch.5 (same proof) or the user's paper on
  vocabulary and XLM-R dimension (raw); in the prose-only version its XLM-R dimension is also above the user's paper (+3.2, p=0.005).
- Guessing the work (7 works, chance 14%): whole-path distance (DTW) 76% (Qwen), 78% (Qwen input table), 64% (XLM-R); dimension, spread and step length 21-30% (near chance); surprise 27%.
- Most recognisable (raw, all measures): the user's paper 91%, Euclid 59%, Ramanujan 1918 59%; prose-only: Ramanujan 1918 90%. Least: Andrews ch.5 9% (mistaken for the original), Hilbert 14%.
- The user's paper: highest Qwen dimension (11.1) and most different words (143); closest by path distance to Dedekind/Hilbert/Ramanujan 1918.
- Caveats: prose-only has 10 passages per work (noisy); passages of one work are correlated, so guessing accuracy is optimistic; the works differ in topic (geometry, number theory, logic);
  LaTeX-derived texts have gaps where the math was removed; p-values are not corrected for multiple comparisons.

## Pooled dimension of whole works (scripts/pooled_dimension.py, examples/pooled_dimension.csv)
16 works x 2 readers (Qwen, XLM-R), all 5,632 points of a work pooled, random samples of 250/500/1000/2000/3000 points, plus a second draw at 1000.
Mean over works: Qwen 12.2, 11.6, 11.1, 11.0, 11.0; XLM-R 15.2, 13.3, 12.3, 11.8, 11.8. Two draws at 1000 points differ by 0.9 on average.
Dimension falls then levels off from about 2000 points; it does not grow with more points. See REPORT.md Result 8.
