"""Focused feature table for mtDNA substitution classification.

The classification task uses six conceptual predictors, chosen because each
carries evidence the others do not:

    mlc_score          local constraint from Lake 2024, per substitution
    phyloP100way       cross-species conservation, per position
    consequence        VEP effect class, one-hot encoded
    hom_rarity_soft    a single population-frequency signal
    codon_pos2_any     second codon position
    codon_pos3_any     third codon position

This replaces the earlier nine-feature panel, which carried four separate
frequency features whose pairwise Spearman correlation reached 0.99 and which
therefore contributed roughly one and a half independent dimensions between
them. The VEP consequence enters directly rather than through codon position,
which was only ever a proxy for it.

Two caveats are structural rather than incidental, and both are measured in
notebook 02b rather than hidden here.

First, both label criteria are partly reconstructible from these features. The
dataset 8 neutral reference was assembled from haplogroup-defining variants,
which are common by construction, and from the lowest phyloP decile. So a
frequency feature partly encodes the first criterion and phyloP partly encodes
the second.

Second, only `mlc_score` and `hom_rarity_soft` distinguish alternative alleles
at one position; the rest are positional. For roughly six in ten unlabelled
positions all three substitutions therefore receive an identical feature vector.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Feature panel
# ---------------------------------------------------------------------------

FORCED_FEATURES = ["mlc_score", "phyloP100way"]

CONSEQUENCE_FEATURES = [
    "cons_missense",
    "cons_synonymous",
    "cons_truncating",
    "cons_noncoding_rna",
    "cons_intergenic",
]

FEATURE_COLUMNS = (
    FORCED_FEATURES
    + ["hom_rarity_soft"]
    + CONSEQUENCE_FEATURES
    + ["codon_pos2_any", "codon_pos3_any"]
)

# Most severe first: a compound VEP annotation on overlapping genes is reduced
# to its most severe component.
CONSEQUENCE_PRIORITY = [
    ("stop_gained", "truncating"),
    ("start_lost", "truncating"),
    ("stop_lost", "truncating"),
    ("incomplete_terminal_codon", "truncating"),
    ("missense", "missense"),
    ("stop_retained", "synonymous"),
    ("synonymous", "synonymous"),
    ("non_coding_transcript", "noncoding_rna"),
    ("intergenic", "intergenic"),
]

REVERSE_STRAND_GENES = {"MT-ND6", "ND6"}

RARITY_EPS = 1e-6
RARITY_CLIP = 6.0


# ---------------------------------------------------------------------------
# Label schemes
# ---------------------------------------------------------------------------

SCHEME_CLASSES = {
    "2class": ["benign", "pathogenic"],
    "3class": ["benign", "vus", "pathogenic"],
    "4class": ["benign", "vus_low", "vus_high", "pathogenic"],
}

D3_TO_SEVERITY = {
    "Benign & Likely Benign": "benign",
    "VUS of low clinical significance": "vus_low",
    "VUS": "vus_high",
    "VUS of high clinical significance": "vus_high",
    "Pathogenic & Likely pathogenic": "pathogenic",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_ucsc_phylop(path: Path, score_col: str = "phyloP100way") -> pd.DataFrame:
    """Read a UCSC bedGraph-like table; a [start, end) interval is position start+1."""
    df = pd.read_csv(path, sep="\t", skiprows=1)
    missing = {"chrom", "start", "end", "value"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    out = pd.DataFrame(
        {
            "position": pd.to_numeric(df["start"], errors="coerce").astype("Int64") + 1,
            score_col: pd.to_numeric(df["value"], errors="coerce"),
        }
    )
    if out["position"].duplicated().any():
        raise ValueError(f"Duplicated positions in {path}")
    return out


def simplify_consequence(raw: str) -> str:
    text = str(raw)
    for token, label in CONSEQUENCE_PRIORITY:
        if token in text:
            return label
    return "other"


def add_consequence_features(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out["consequence_class"] = out["consequence"].map(simplify_consequence)
    for label in ["missense", "synonymous", "truncating", "noncoding_rna", "intergenic"]:
        out[f"cons_{label}"] = out["consequence_class"].eq(label).astype(int)
    return out


def add_frequency_feature(data: pd.DataFrame) -> pd.DataFrame:
    """One frequency signal: how rare the variant is in the homoplasmic state.

    Homoplasmy is the informative state. A substitution severe enough to matter
    can persist heteroplasmically, where unaffected copies of the genome
    compensate, but cannot reach homoplasmy in a viable carrier. The maximum
    across gnomAD and HelixMT is used rather than a sum because the two
    databases overlap in samples.
    """
    out = data.copy()
    for column in ["gnomad_homoplasmic_af", "helix_af_hom"]:
        out[column] = pd.to_numeric(out.get(column), errors="coerce")

    out["pop_af_hom_max"] = out[["gnomad_homoplasmic_af", "helix_af_hom"]].max(axis=1)
    out["pop_af_hom_max"] = out["pop_af_hom_max"].fillna(0.0)
    out["hom_rarity_soft"] = np.clip(
        -np.log10(out["pop_af_hom_max"] + RARITY_EPS), 0.0, RARITY_CLIP
    )
    return out


def add_codon_features(data: pd.DataFrame) -> pd.DataFrame:
    """Codon position within the reading frame of the gene, not of the genome."""
    out = data.copy()
    symbol = (
        out["gene_constraint_symbol"].astype(str).str.upper().str.replace(" ", "", regex=False)
    )
    start = pd.to_numeric(out["gene_constraint_start_position"], errors="coerce")
    end = pd.to_numeric(out["gene_constraint_end_position"], errors="coerce")
    position = pd.to_numeric(out["position"], errors="coerce")

    is_coding = start.notna() & end.notna()
    is_reverse = symbol.isin(REVERSE_STRAND_GENES)

    relative = pd.Series(np.nan, index=out.index, dtype="float64")
    forward = is_coding & ~is_reverse
    reverse = is_coding & is_reverse
    relative.loc[forward] = position.loc[forward] - start.loc[forward] + 1
    relative.loc[reverse] = end.loc[reverse] - position.loc[reverse] + 1

    frame = relative % 3
    out["codon_pos1_any"] = frame.eq(1).astype(int)
    out["codon_pos2_any"] = frame.eq(2).astype(int)
    out["codon_pos3_any"] = frame.eq(0).astype(int)
    out["codon_position_simple"] = np.select(
        [frame.eq(1), frame.eq(2), frame.eq(0)],
        ["pos1", "pos2", "pos3"],
        default="noncoding",
    )
    return out


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def attach_label_sources(
    data: pd.DataFrame, dataset3: Path, dataset8: Path, dataset9: Path
) -> pd.DataFrame:
    d3 = pd.read_csv(dataset3, sep="\t")
    d3["variant_id"] = (
        "m." + d3["Position"].astype(str) + d3["Reference"] + ">" + d3["Alternate"]
    )
    # Dataset 3 ships some group names with trailing whitespace ("VUS ").
    d3 = d3[["variant_id", "Classification_group"]].rename(
        columns={"Classification_group": "d3_classification_group"}
    )
    d3["d3_classification_group"] = d3["d3_classification_group"].str.strip()

    d8 = pd.read_csv(dataset8, sep="\t").rename(
        columns={"Variant": "variant_id", "Criteria": "d8_criteria"}
    )[["variant_id", "d8_criteria"]]

    d9 = pd.read_csv(dataset9, sep="\t").rename(
        columns={"Variant": "variant_id", "Curated_status": "d9_curated_status"}
    )[["variant_id", "d9_curated_status"]]

    merged = data
    for source in (d3, d8, d9):
        if source["variant_id"].duplicated().any():
            raise ValueError("Duplicated variant_id in a label source")
        merged = merged.merge(source, on="variant_id", how="left")
    if len(merged) != len(data):
        raise ValueError("Label merge changed the row count")
    return merged


def build_label_schemes(data: pd.DataFrame) -> pd.DataFrame:
    """Derive the 2-, 3- and 4-class targets under one precedence rule.

    Precedence is dataset 9 over dataset 3 over dataset 8. Dataset 3 outranks
    dataset 8 because it records an individual clinical call on that variant,
    whereas dataset 8 membership follows from a rule applied in bulk.
    """
    severity = pd.Series(pd.NA, index=data.index, dtype="object")

    from_d8 = data["is_neutral_dataset8"].eq(1)
    severity = severity.mask(from_d8, "benign")

    d3_severity = data["d3_classification_group"].map(D3_TO_SEVERITY)
    labelled_d3 = data["d3_classification_group"].notna()
    if d3_severity.notna().sum() != labelled_d3.sum():
        unknown = sorted(
            set(data.loc[labelled_d3, "d3_classification_group"]) - set(D3_TO_SEVERITY)
        )
        raise ValueError(f"Unmapped dataset 3 groups: {unknown}")
    severity = severity.mask(d3_severity.notna(), d3_severity)

    from_d9 = data["is_pathogenic_dataset9"].eq(1)
    severity = severity.mask(from_d9, "pathogenic")

    out = data.copy()
    out["label_source"] = np.select(
        [from_d9, d3_severity.notna(), from_d8],
        ["dataset9_curated_pathogenic", "dataset3_clinical_group", "dataset8_neutral"],
        default="unlabeled",
    )
    out["label_4class"] = severity
    out["label_3class"] = severity.replace({"vus_low": "vus", "vus_high": "vus"})
    out["label_2class"] = severity.where(severity.isin(["benign", "pathogenic"]))

    # Neutral variants chosen by the lowest phyloP decile carry the circularity
    # between phyloP and the label; the haplogroup-chosen ones do not.
    out["neutral_selection_rule"] = out["d8_criteria"].fillna("not_dataset8")
    out["in_leakage_control_cohort"] = ~out["neutral_selection_rule"].eq(
        "lowest_decile_phyloP"
    )
    return out


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

OUTPUT_COLUMNS = [
    "variant_id",
    "position",
    "reference",
    "alternate",
    "consequence",
    "consequence_class",
    "codon_position_simple",
    "pop_af_hom_max",
    "is_neutral_dataset8",
    "is_disease_suspected_dataset3",
    "is_pathogenic_dataset9",
    "d3_classification_group",
    "d8_criteria",
    "d9_curated_status",
    "label_source",
    "label_2class",
    "label_3class",
    "label_4class",
    "neutral_selection_rule",
    "in_leakage_control_cohort",
    "codon_pos1_any",
]


def build_features(
    master_path: Path,
    phylop_path: Path,
    dataset3: Path,
    dataset8: Path,
    dataset9: Path,
) -> pd.DataFrame:
    master = pd.read_csv(master_path, sep="\t", low_memory=False)
    n_input = len(master)

    master["position"] = pd.to_numeric(master["position"], errors="coerce")
    master["mlc_score"] = pd.to_numeric(master["mlc_score"], errors="coerce")

    phylop = load_ucsc_phylop(phylop_path)
    work = master.merge(phylop, on="position", how="left")
    if len(work) != n_input:
        raise ValueError("phyloP merge changed the row count")

    work = add_consequence_features(work)
    work = add_frequency_feature(work)
    work = add_codon_features(work)
    work = attach_label_sources(work, dataset3, dataset8, dataset9)
    work = build_label_schemes(work)

    missing = work[FEATURE_COLUMNS].isna().sum()
    missing = missing[missing > 0]
    if len(missing):
        raise ValueError(f"Missing feature values:\n{missing}")

    columns = OUTPUT_COLUMNS + FEATURE_COLUMNS
    return work[[c for c in columns if c in work.columns]].copy()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path,
                        default=root / "data/processed/master_snv_table.tsv")
    parser.add_argument("--phylop", type=Path,
                        default=root / "data/raw/phyloP100way.tsv")
    raw = root / "data/raw/lake_2024_supplement_data"
    parser.add_argument("--dataset3", type=Path, default=raw / "supplementary_dataset_3.tsv")
    parser.add_argument("--dataset8", type=Path, default=raw / "supplementary_dataset_8.tsv")
    parser.add_argument("--dataset9", type=Path, default=raw / "supplementary_dataset_9.tsv")
    parser.add_argument("--output", type=Path,
                        default=root / "data/processed/model_features.tsv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = build_features(
        args.master, args.phylop, args.dataset3, args.dataset8, args.dataset9
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, sep="\t", index=False)

    print(f"rows: {len(features):,}   features: {len(FEATURE_COLUMNS)}")
    print("\nconsequence classes:")
    print(features["consequence_class"].value_counts().to_string())
    print("\nlabelled variants per scheme:")
    for scheme in SCHEME_CLASSES:
        counts = features[f"label_{scheme}"].value_counts()
        print(f"  {scheme}: {counts.to_dict()}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
