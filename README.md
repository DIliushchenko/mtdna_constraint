# mtdna_constraint

The `192_mutspec` branch extends the downstream spectrum workflow to 192
full-strand trinucleotide categories (`4 flanking bases × 12 substitutions ×
4 flanking bases`), written as `A[C>T]G`.

Run notebooks `03_prepare_spectrum_groups.ipynb`,
`04_add_spectrum_weights.ipynb`, and `05_compare_mutational_spectra.ipynb` in
order. They use the circular GRCh38 chrM reference, normalize by genomic
trinucleotide opportunities, and write separate outputs containing `_192_T95`.
