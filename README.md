# mtdna_constraint

## 192-component spectrum workflow

The 192-component workflow uses full-strand trinucleotide categories
(`4 flanking bases × 12 substitutions × 4 flanking bases`), written as
`A[C>T]G`. It is kept separate from the native 12-component workflow so both
pipelines can coexist after merging:

1. [`notebooks/03_prepare_spectrum_groups_192.ipynb`](notebooks/03_prepare_spectrum_groups_192.ipynb)
2. [`notebooks/04_add_spectrum_weights_192.ipynb`](notebooks/04_add_spectrum_weights_192.ipynb)
3. [`notebooks/05_compare_mutational_spectra_192.ipynb`](notebooks/05_compare_mutational_spectra_192.ipynb)
4. [`notebooks/06_compare_functional_class_spectra_192.ipynb`](notebooks/06_compare_functional_class_spectra_192.ipynb)

All generated tables and figure directories from this workflow include `192`
in their names.

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

The main analysis normalizes each reference trinucleotide by the number of
analyzable positions carrying that context inside the corresponding functional
class. Genome-wide opportunity normalization and raw-count normalization are
retained as sensitivity analyses. The primary tidy output is
`results/mutation_spectra/functional_class_spectra_192_T95.tsv`; accompanying
QC tables, summaries, top contributors, annotated variants, and figures are
also named with the `192` marker.

## Interpretation report

[`reports/README_192_component_spectrum.md`](reports/README_192_component_spectrum.md)
explains why the 192-component plots look sparse and highly peaked, using
concrete channels, exact variants, opportunity counts, and concentration
statistics from the current T95 outputs.
