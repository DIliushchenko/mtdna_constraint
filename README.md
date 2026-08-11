# mtdna_constraint

## Isolation Forest characterization

[`notebooks/02b_characterize_isolation_forest.ipynb`](notebooks/02b_characterize_isolation_forest.ipynb)
characterizes the final nine-feature Isolation Forest fitted in notebook 02.
Because Isolation Forest has neither linear coefficients nor native feature
weights, the notebook reports fixed-model permutation influence,
leave-one-feature-out refits across three random seeds, full-model seed
stability, feature correlations, and local neutral-replacement perturbations.

Notebook 02 did not serialize its fitted forest, so 02b reconstructs the same
pipeline from the saved feature table and records agreement with the saved
scores in `results/model_characterization/model_reconstruction_qc.tsv`. All
other characterization outputs are stored under
`results/model_characterization/`, with figures under
`results/figures/model_characterization/`. The primary side-by-side readout is
`results/model_characterization/feature_influence_summary.tsv`; its ranks are
diagnostic orderings, not fitted coefficients or causal effects.

## Random Forest benchmark

[`notebooks/02c_compare_random_forest.ipynb`](notebooks/02c_compare_random_forest.ipynb)
compares a supervised Random Forest with the current one-class Isolation Forest.
The reusable calculation is in
[`scripts/random_forest_benchmark.py`](scripts/random_forest_benchmark.py). It
uses clean non-overlapping Dataset 8 neutral and Dataset 9 pathogenic labels,
repeated five-fold cross-validation grouped by mtDNA position, inner neutral
T95 calibration, and position-cluster bootstrap confidence intervals.

On the current nine features Random Forest performs better, but that advantage
is not robust after removing `phyloP100way`. Because the neutral reference was
partly selected using low phyloP and haplogroup criteria, this benchmark is kept
as a sensitivity analysis rather than a replacement for the downstream
neutral-domain grouping. Tables are stored in
`results/model_random_forest_benchmark/` and figures in
`results/figures/model_random_forest_benchmark/`.

## Random Forest application to unlabeled variants

[`notebooks/02d_apply_random_forest_to_unlabeled.ipynb`](notebooks/02d_apply_random_forest_to_unlabeled.ipynb)
applies repeated position-grouped Random Forest models to variants that were
not used as strict neutral or pathogenic labels. The reusable calculation is
in
[`scripts/random_forest_application.py`](scripts/random_forest_application.py).
Each of 25 models uses an independent neutral calibration subset for its own
T95-like threshold. The notebook reports both the current nine-feature panel
and a prespecified sensitivity panel without `phyloP100way`, held-out
permutation importance, model-to-model call stability, and joint
Isolation-Forest/Random-Forest classes.

The RF output is treated as supervised pathogenic similarity rather than a
clinical pathogenicity probability. On the current data, the all-feature RF
calls 89.8% of unlabeled substitutions pathogenic-like and is strongly driven
by phyloP; therefore it is retained as a second prioritization axis, not a
replacement for the neutral-domain classification. Tables are stored in
`results/model_random_forest_application/` and figures in
`results/figures/model_random_forest_application/`.

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

The 95% intervals in the 192-component notebooks use 4,000 paired bootstrap
replicates of whole mtDNA positions. All alternate alleles, group assignments,
and SBS192 contributions at a position are resampled together; the known rCRS
context-opportunity table is kept fixed, and the opportunity correction plus
shared denominator are reapplied in every replicate. The older conditional
Poisson-count implementation remains available through
`resampling_method="poisson_counts"` in `scripts/mutspec192_ci.py`, but it is
not the primary figure because millions of aggregated carrier counts produce
much narrower intervals with a different, count-conditional interpretation.

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
