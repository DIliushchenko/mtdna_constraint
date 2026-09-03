# mtdna_constraint

Constraint and mutational-spectrum analysis of human mitochondrial DNA, built on
the Lake 2024 supplementary datasets, gnomAD and HelixMT population counts, and
MITOMAP locus coordinates.

## Pipeline

Notebooks run in order; each one reads what the previous ones wrote.

| Notebook | Purpose |
|---|---|
| `00_build_master_table` | Assemble one row per possible mtDNA substitution from all sources |
| `01_prepare_model_input` | Derive population summaries and the analysis-group column |
| `02_fit_isolation_forest` | One-class outlier model; produces the T95 neutral-domain grouping the spectrum notebooks consume |
| `02b_classify_variants` | Classification: label granularity, model and feature subset, with the one-class forest as a baseline |
| `03`-`04` | Assign spectrum groups and attach population carrier weights (12- and 192-component variants) |
| `05`-`06` | Compare mutational spectra overall and by MITOMAP functional class |
| `05b_compare_spectra_by_class_192` | The same 192-component normalisation applied to the four predicted severity classes |

Two reusable modules sit behind the classification branch:
[`scripts/features.py`](scripts/features.py) builds the feature table and the
label schemes, and [`scripts/classification.py`](scripts/classification.py)
runs model selection, the baseline comparison and the application step.

Large generated tables are not tracked in git; see `.gitignore`. Everything is
regenerable from `data/raw/` by running the notebooks in order.

## Variant classification

[`notebooks/02b_classify_variants.ipynb`](notebooks/02b_classify_variants.ipynb)
decides three things: how many ordered severity classes are separable, which
model should carry the classification of all 49,704 possible substitutions, and
which features it needs. The reusable calculation is in
[`scripts/classification.py`](scripts/classification.py); the feature table it
consumes is built by [`scripts/features.py`](scripts/features.py) and written to
`data/processed/model_features.tsv`.

### Feature panel

Six conceptual predictors, ten columns after one-hot encoding:

| Feature | Source | Role |
|---|---|---|
| `mlc_score` | Lake 2024 dataset 7 | Local constraint, per substitution |
| `phyloP100way` | UCSC, 100 vertebrates | Cross-species conservation, per position |
| `hom_rarity_soft` | gnomAD and HelixMT | A single frequency signal: rarity in the homoplasmic state |
| `cons_*` | VEP consequence | Effect class: missense, synonymous, truncating, non-coding RNA, intergenic |
| `codon_pos2_any`, `codon_pos3_any` | Gene reading frame | Codon position |

This replaces an earlier nine-feature panel that carried four separate frequency
features. Their pairwise Spearman correlation reached 0.99, so they contributed
roughly one and a half independent dimensions between them, and the full panel
scored *worse* than a five-feature subset of it. The VEP consequence now enters
directly rather than through codon position, which was only ever a proxy for it.

### Two structural caveats, measured rather than assumed

Both label criteria are partly reconstructible from the features. The dataset 8
neutral reference was assembled from haplogroup-defining variants, which are
common by construction, and from the lowest phyloP decile. A frequency feature
therefore encodes the first criterion and phyloP the second. On a cohort holding
only the haplogroup-selected neutrals, `hom_rarity_soft` alone separates benign
from pathogenic at AUC 0.98 — a number that reflects the selection rule, not
biology. Restricting the evaluation to that cohort removes the phyloP
circularity but concentrates the frequency one, so neither cohort is clean and
both are reported.

Only `mlc_score` and `hom_rarity_soft` distinguish alternative alleles at one
position; the rest are positional. For roughly six in ten unlabelled positions
all three substitutions receive an identical feature vector, so the model ranks
positions rather than alleles.

### Isolation Forest baseline

The one-class forest is kept as a baseline rather than dropped. It is refitted
on the same features and scored on the same position-grouped folds as the
supervised models, with its threshold calibrated on held-out benign rows exactly
as the T95 rule of notebook 02 calibrates on held-out neutrals. Single features
are scored on the same folds under the same rule, which is what makes the
comparison informative: on the earlier nine-feature panel the forest reached
85.7% recall on curated pathogenic variants at a 5.3% false-positive rate, while
`phyloP100way` alone reached 78.6% and the frequency features alone reached
zero. The forest is largely a phyloP detector, and its neutral reference mixes
common haplogroup variants with never-observed low-conservation positions —
two populations glued into one notion of "normal", which is why two thirds of
unlabelled substitutions fall outside its domain.

Tables are stored in `results/classification/` and figures in
`results/figures/classification/`.

The predicted `pathogenic` class means resemblance to the 91 curated pathogenic
variants on these features, not clinical pathogenicity. `hom_rarity_soft` is 1
for every substitution nobody has observed, which is seven in ten of them, so
unobserved variants drift toward that class by construction.

## Spectra of the predicted classes

[`notebooks/05b_compare_spectra_by_class_192.ipynb`](notebooks/05b_compare_spectra_by_class_192.ipynb)
applies the 192-component normalisation of notebook 05 to the four severity
classes predicted in notebook 02b, instead of the two groups defined by the T95
threshold. Strand orientation, opportunity counts, the shared denominator and
the position bootstrap are unchanged; only the grouping column, the number of
groups and the figures differ.

The grouping does not come from the main classifier. A spectrum is weighted by
population carrier counts, so a model that uses population frequency as a
predictor would assign variants to groups partly by the quantity that later
weights them, and the spectra would differ in part by construction. The T95
grouping of notebook 05 has exactly this problem, since the Isolation Forest was
fitted on four rarity features, but there it is unmeasured.

[`scripts/spectrum_grouping.py`](scripts/spectrum_grouping.py) therefore refits
the selected classifier on a frequency-free panel — local constraint,
conservation, consequence class and codon position. Class membership is then
independent of the weights. The cost is stated rather than hidden: macro F1
falls from 0.53 to 0.41 and quadratic kappa from 0.71 to 0.39, and the share of
substitutions called pathogenic-like falls from 50.2% to 23.8%, which is itself
evidence that the frequency feature was inflating that class among never
observed variants.

`scripts/mutspec192_ci.py` now accepts more than two groups. The shared
denominator spans every group passed, and differences are reported for all
ordered pairs; with two groups the output is unchanged.

### Why channels look absent

No SBS192 channel is structurally impossible here: every one of the 64
trinucleotides occurs in mtDNA between 54 and 624 times, and all 192 channels
hold candidate substitutions. A channel looks empty for two other reasons.

The first is observation. Most candidates have never been seen in a carrier —
61.7% in the benign-like class, 87.1% in the pathogenic-like class — and enter
only through the pseudocount `max(count, 1)`. Counting channels that hold at
least one genuinely observed variant gives 192, 155, 143 and 114 for the
benign-like, pathogenic-like, low-VUS and high-VUS classes. Those pseudocount
variants carry 0.3% to 9.0% of a class's weight, so a channel built solely from
them is near zero by construction rather than by biology. The per-class figures
are audited in `channel_observation_audit.tsv`.

The second is concentration inside a class rather than the choice of
denominator. In the pathogenic-like class one variant carries 48% of the class
weight and ten carry 89%, so on any axis the remaining channels are small. Each
per-class figure is drawn on its own scale, so this is a property of the data,
not of the plot.

The denominator itself is shared across all four classes: the frequencies of all
4 × 192 = 768 cells sum to one jointly, and each class keeps its share of the
combined mass (0.9028, 0.0453, 0.0053 and 0.0466). That is what makes class
totals comparable, and it is the normalisation used throughout.

Tables are stored in `results/mutation_spectra_by_class/` and figures in
`results/figures/mutation_spectra_by_class_192/`; figures named
`spectrum_per_group_*` use the within-class normalisation.

## 192-component spectrum workflow

The 192-component workflow uses directional trinucleotide categories
(`4 flanking bases × 12 substitutions × 4 flanking bases`), written as
`A[C>T]G`. It is kept separate from the native 12-component workflow so both
pipelines can coexist after merging:

1. [`notebooks/03_prepare_spectrum_groups_192.ipynb`](notebooks/03_prepare_spectrum_groups_192.ipynb)
2. [`notebooks/04_add_spectrum_weights_192.ipynb`](notebooks/04_add_spectrum_weights_192.ipynb)
3. [`notebooks/05_compare_mutational_spectra_192.ipynb`](notebooks/05_compare_mutational_spectra_192.ipynb)
4. [`notebooks/06_compare_functional_class_spectra_192.ipynb`](notebooks/06_compare_functional_class_spectra_192.ipynb)

All generated tables and figure directories from this workflow include `192`
in their names.

In both the 12- and 192-component comparisons, the final spectrum frequencies
use one denominator shared across the analyzed model groups. Frequencies sum to
one across groups and channels jointly, so each group's total retains its share
of the combined spectrum weight.

The 90% intervals in the 192-component notebooks use 4,000 paired bootstrap
replicates of whole mtDNA positions. **The resampling unit is the position, not
the variant and not the carrier**: 16,566 analysable positions are drawn with
replacement, and one draw is shared by every group and every SBS192 channel, so
all alternate alleles and group assignments at a position travel together. The
known rCRS context-opportunity table is kept fixed, and the opportunity
correction plus the shared denominator are reapplied in every replicate.

That choice explains the wide whiskers. A position is absent from a replicate
with probability (1 - 1/16566)^16566 = 0.368, so any channel resting on one
position vanishes in about 37% of replicates and its lower quantile sits at
zero, while the replicates that draw it two or three times produce the long
upper tail. Interval width therefore tracks how many positions hold a channel
up, and nothing else:

| Effective positions per channel | Median interval width, relative to the estimate |
|---|---|
| about 1 | 2.83 |
| 1.5 to 3 | 2.21 |
| 3 to 10 | 1.41 |
| 10 to 50 | 0.82 |
| over 50 | 0.53 |

Every spectrum table therefore reports `n_positions`, `top_position_share` and
`n_effective_positions`, the inverse Simpson count `1 / sum(share^2)`. Raw
position counts mislead on their own: the largest pathogenic-like channel
`T[A>G]A` draws on 70 positions but its top position carries 99.3% of the
weight, so its effective count is 1.01 and it behaves as a single-position
channel. The other 69 positions contribute only pseudocounts.

`scripts/mutspec192_ci.py` implements exactly one resampling scheme and one
denominator, with no alternative to choose between, and reports it the way
PyMutSpec expects. That library computes no intervals of its own —
`plot_mutspec192` only renders precomputed columns — so the scheme is stated
here rather than inherited, while the presentation follows it exactly: the bar
is the point estimate, the grey marker and the whisker sit on the resampling
median, and the interval runs from the 5th to the 95th percentile, hence 90%
rather than 95%. Every spectrum table also carries `MutSpec`, `MutSpec_median`,
`MutSpec_q05` and `MutSpec_q95` aliases, so it can be passed straight to
`plot_mutspec192`. A visible gap between the top of a bar and its grey marker
means the resampling distribution is skewed away from the point estimate, which
happens where a channel rests on very few positions. The earlier conditional Poisson-count alternative has been removed:
aggregated carrier counts in the millions produced much narrower intervals under
a different, count-conditional interpretation, and keeping two schemes invited
comparing figures that were not comparable.

Before opportunity normalization and plotting in notebooks 05 and 06, the 1076
positions in canonical MITOMAP loci marked `[on complement]` are oriented to
the functional strand. These loci are `MT-ND6`, `MT-TQ`, `MT-TA`, `MT-TN`,
`MT-TC`, `MT-TY`, `MT-TS1`, `MT-TE`, and `MT-TP`. Thus
`L[REF>ALT]R` becomes `comp(R)[comp(REF)>comp(ALT)]comp(L)` in the
192-component analysis, while the 12-component analysis complements `REF>ALT`.
The reference-base or trinucleotide opportunity counts and population weights
are then recomputed from the oriented position universe. Original rCRS fields
remain available in notebook memory, and in the notebook 06 annotated output,
with an `_rcrs` suffix. If differently oriented canonical loci overlap, presence
in any complement-strand locus takes precedence; this applies to positions
4329–4331 shared by `MT-TI` and `MT-TQ`.

## MITOMAP functional-class spectra

The functional-class analysis compares:

- the 13 canonical protein-coding mtDNA genes;
- the 22 tRNA genes plus `MT-RNR1` and `MT-RNR2`;
- the circular control region `MT-CR` (`16024–16569` and `1–576`).

Coordinates come from the
[MITOMAP Genome Loci](https://www.mitomap.org/foswiki/bin/view/MITOMAP/GenomeLoci)
page. The exporter converts its embedded DataTables array to a versioned TSV
and a JSON provenance record:

```bash
python scripts/export_mitomap_functional_loci_192.py \
  --expected-revision r889 \
  --expected-record-count 87 \
  --expected-tsv-sha256 7d396431a03109917a889b6a37f4d887f5cb1745e92fe9d15fb4f91b74dc8ed1
```

The exported source files are stored under `data/raw/mitomap/`. Overlapping
loci within one functional class are counted once. Other MITOMAP loci remain
available as traceability annotations but do not alter the three canonical
classes.

The main analysis normalizes each strand-oriented reference trinucleotide by
the number of analyzable positions carrying that context inside the
corresponding functional class. Genome-wide opportunity normalization and
raw-count normalization are retained as sensitivity analyses. The primary tidy output is
`results/mutation_spectra/functional_class_spectra_192_T95.tsv`; accompanying
QC tables, summaries, top contributors, annotated variants, and figures are
also named with the `192` marker.

## Interpretation report

[`reports/README_192_component_spectrum.md`](reports/README_192_component_spectrum.md)
explains why the 192-component plots look sparse and highly peaked, using
concrete channels, exact variants, opportunity counts, and concentration
statistics from the current T95 outputs.
