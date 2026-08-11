"""Cross-fitted Random Forest scoring for variants without strict labels.

This module deliberately keeps the supervised Random Forest separate from the
one-class Isolation Forest.  Random Forest models are trained only on clean
Dataset 8 neutral and non-overlapping Dataset 9 pathogenic variants.  Every
model uses an independent neutral calibration subset to define a T95-like
threshold, then scores variants that were not used as either strict class.

The resulting labels mean ``pathogenic-like under the curated supervised
task``.  They do not redefine membership in the neutral domain.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from random_forest_benchmark import (
    FEATURE_COLS,
    binary_metrics,
    make_random_forest,
    threshold_at_neutral_tail,
)


APPLICATION_PANELS = {
    "current_all_9": FEATURE_COLS,
    "without_phyloP": [
        feature for feature in FEATURE_COLS if feature != "phyloP100way"
    ],
}

FEATURE_FAMILIES = {
    "mlc": ["mlc_score"],
    "population_rarity": [
        "rarity_soft",
        "hom_rarity_soft",
        "het_rarity_soft",
        "no_homoplasmic_signal",
    ],
    "codon_position": [
        "codon_pos1_any",
        "codon_pos2_any",
        "codon_pos3_any",
    ],
    "phylogenetic_conservation": ["phyloP100way"],
}

IF_EXPANDED = "expanded_neutral_like_T95"
IF_OUT = "unlabeled_out_of_neutral_domain_T95"
RANDOM_SEED = 20260810


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "results/model/isolation_forest_scores_T95.tsv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results/model_random_forest_application",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=root / "results/figures/model_random_forest_application",
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def load_application_data(
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(path, sep="\t", low_memory=False)
    required = {
        "variant_id",
        "position",
        "reference",
        "alternate",
        "analysis_group",
        "is_neutral_dataset8",
        "is_pathogenic_dataset9",
        "is_disease_suspected_dataset3",
        "isolation_forest_outlier_score",
        "isolation_forest_above_T95",
        "spectrum_group_primary_T95_preview",
        *FEATURE_COLS,
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing application columns: {missing}")

    numeric_cols = [
        "position",
        "is_neutral_dataset8",
        "is_pathogenic_dataset9",
        "is_disease_suspected_dataset3",
        "isolation_forest_outlier_score",
        "isolation_forest_above_T95",
        *FEATURE_COLS,
    ]
    for column in numeric_cols:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data[["position", *FEATURE_COLS]].isna().any().any():
        missing_counts = data[["position", *FEATURE_COLS]].isna().sum()
        raise ValueError(
            "Missing values in application matrix:\n"
            f"{missing_counts[missing_counts.gt(0)]}"
        )
    if data["variant_id"].duplicated().any():
        raise ValueError("variant_id must be unique.")

    neutral_flag = data["is_neutral_dataset8"].eq(1)
    pathogenic_flag = data["is_pathogenic_dataset9"].eq(1)
    overlap = neutral_flag & pathogenic_flag
    clean_neutral = neutral_flag & ~pathogenic_flag
    clean_pathogenic = pathogenic_flag & ~neutral_flag
    strict = clean_neutral | clean_pathogenic

    data["rf_cohort"] = "not_in_application"
    data.loc[clean_neutral, "rf_cohort"] = "clean_known_neutral"
    data.loc[clean_pathogenic, "rf_cohort"] = "clean_confirmed_pathogenic"
    data.loc[overlap, "rf_cohort"] = "excluded_neutral_pathogenic_overlap"
    data.loc[
        data["analysis_group"].eq("unlabeled_or_other") & ~strict & ~overlap,
        "rf_cohort",
    ] = "unlabeled"
    data.loc[
        data["analysis_group"].eq("disease_suspected_posthoc")
        & ~strict
        & ~overlap,
        "rf_cohort",
    ] = "disease_suspected_posthoc"

    benchmark = data.loc[strict].copy()
    benchmark["target_pathogenic"] = clean_pathogenic.loc[
        benchmark.index
    ].astype(int)
    benchmark = benchmark.reset_index(drop=True)
    application = data[
        data["rf_cohort"].isin(
            ["unlabeled", "disease_suspected_posthoc"]
        )
    ].copy().reset_index(drop=True)

    audit = (
        data.groupby("rf_cohort", dropna=False)
        .agg(
            n_variants=("variant_id", "size"),
            n_positions=("position", "nunique"),
        )
        .reset_index()
    )
    audit = pd.concat(
        [
            pd.DataFrame(
                [{
                    "rf_cohort": "all_variants",
                    "n_variants": len(data),
                    "n_positions": data["position"].nunique(),
                }]
            ),
            audit,
        ],
        ignore_index=True,
    )
    return data, benchmark, application, audit


def _permutation_units(panel_cols: Sequence[str]) -> list[tuple[str, str, list[str]]]:
    units = [("individual", feature, [feature]) for feature in panel_cols]
    for family, family_cols in FEATURE_FAMILIES.items():
        available = [feature for feature in family_cols if feature in panel_cols]
        if available:
            units.append(("family", family, available))
    return units


def run_cross_fitted_application(
    benchmark: pd.DataFrame,
    application: pd.DataFrame,
    *,
    outer_folds: int,
    repeats: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target = benchmark["target_pathogenic"].to_numpy(int)
    groups = benchmark["position"].to_numpy()
    n_models = outer_folds * repeats
    application_parts: list[pd.DataFrame] = []
    oof_parts: list[pd.DataFrame] = []
    threshold_rows: list[dict] = []
    importance_rows: list[dict] = []

    for panel_index, (panel_name, panel_cols) in enumerate(
        APPLICATION_PANELS.items()
    ):
        benchmark_features = benchmark[panel_cols]
        application_features = application[panel_cols]
        application_score = np.empty(
            (len(application), n_models), dtype=np.float32
        )
        application_call = np.empty(
            (len(application), n_models), dtype=bool
        )
        application_margin = np.empty(
            (len(application), n_models), dtype=np.float32
        )
        model_column = 0

        for repeat in range(repeats):
            repeat_seed = random_state + repeat * 1000
            outer_cv = StratifiedGroupKFold(
                n_splits=outer_folds,
                shuffle=True,
                random_state=repeat_seed,
            )
            for fold, (outer_train_idx, test_idx) in enumerate(
                outer_cv.split(benchmark_features, target, groups), start=1
            ):
                inner_target = target[outer_train_idx]
                inner_groups = groups[outer_train_idx]
                inner_cv = StratifiedGroupKFold(
                    n_splits=4,
                    shuffle=True,
                    random_state=repeat_seed + fold * 17,
                )
                model_rel_idx, calibration_rel_idx = next(
                    inner_cv.split(
                        benchmark_features.iloc[outer_train_idx],
                        inner_target,
                        inner_groups,
                    )
                )
                model_idx = outer_train_idx[model_rel_idx]
                calibration_idx = outer_train_idx[calibration_rel_idx]
                calibration_neutral = target[calibration_idx] == 0
                if set(target[model_idx]) != {0, 1}:
                    raise ValueError("A model split lacks one strict class.")

                fold_seed = repeat_seed + fold + panel_index * 100
                model = make_random_forest(fold_seed)
                model.fit(
                    benchmark_features.iloc[model_idx], target[model_idx]
                )
                calibration_score = model.predict_proba(
                    benchmark_features.iloc[calibration_idx]
                )[:, 1]
                threshold = threshold_at_neutral_tail(
                    calibration_score, calibration_neutral
                )

                test_features = benchmark_features.iloc[test_idx]
                test_score = model.predict_proba(test_features)[:, 1]
                test_call = test_score > threshold
                oof = benchmark.loc[
                    test_idx,
                    ["variant_id", "position", "target_pathogenic"],
                ].copy()
                oof["feature_panel"] = panel_name
                oof["repeat"] = repeat + 1
                oof["fold"] = fold
                oof["rf_score"] = test_score
                oof["rf_threshold"] = threshold
                oof["rf_call"] = test_call.astype(int)
                oof_parts.append(oof)

                score = model.predict_proba(application_features)[:, 1]
                application_score[:, model_column] = score
                application_call[:, model_column] = score > threshold
                application_margin[:, model_column] = score - threshold

                threshold_rows.append({
                    "feature_panel": panel_name,
                    "repeat": repeat + 1,
                    "fold": fold,
                    "model_seed": fold_seed,
                    "n_model_train": len(model_idx),
                    "n_pathogenic_model_train": int(target[model_idx].sum()),
                    "n_neutral_calibration": int(calibration_neutral.sum()),
                    "rf_threshold": threshold,
                })

                baseline_ap = average_precision_score(
                    target[test_idx], test_score
                )
                baseline_roc = roc_auc_score(target[test_idx], test_score)
                for unit_index, (unit_type, unit_name, unit_cols) in enumerate(
                    _permutation_units(panel_cols)
                ):
                    permutation_rng = np.random.default_rng(
                        fold_seed * 1000 + unit_index
                    )
                    order = permutation_rng.permutation(len(test_features))
                    permuted = test_features.copy()
                    permuted.loc[:, unit_cols] = (
                        test_features.iloc[order][unit_cols].to_numpy()
                    )
                    permuted_score = model.predict_proba(permuted)[:, 1]
                    importance_rows.extend([
                        {
                            "feature_panel": panel_name,
                            "repeat": repeat + 1,
                            "fold": fold,
                            "importance_type": unit_type,
                            "feature_or_family": unit_name,
                            "metric": "average_precision",
                            "baseline_value": baseline_ap,
                            "permuted_value": average_precision_score(
                                target[test_idx], permuted_score
                            ),
                        },
                        {
                            "feature_panel": panel_name,
                            "repeat": repeat + 1,
                            "fold": fold,
                            "importance_type": unit_type,
                            "feature_or_family": unit_name,
                            "metric": "roc_auc",
                            "baseline_value": baseline_roc,
                            "permuted_value": roc_auc_score(
                                target[test_idx], permuted_score
                            ),
                        },
                    ])
                model_column += 1

        if model_column != n_models:
            raise AssertionError("Unexpected number of application models.")

        aggregate = application[
            [
                "variant_id",
                "position",
                "reference",
                "alternate",
                "analysis_group",
                "rf_cohort",
                "isolation_forest_outlier_score",
                "isolation_forest_above_T95",
                "spectrum_group_primary_T95_preview",
            ]
        ].copy()
        aggregate["feature_panel"] = panel_name
        aggregate["n_models"] = n_models
        aggregate["rf_score_mean"] = application_score.mean(axis=1)
        aggregate["rf_score_sd"] = application_score.std(axis=1, ddof=1)
        aggregate["rf_score_median"] = np.median(
            application_score, axis=1
        )
        aggregate["rf_score_q025"] = np.quantile(
            application_score, 0.025, axis=1
        )
        aggregate["rf_score_q975"] = np.quantile(
            application_score, 0.975, axis=1
        )
        aggregate["rf_margin_mean"] = application_margin.mean(axis=1)
        aggregate["rf_call_fraction"] = application_call.mean(axis=1)
        aggregate["rf_majority_pathogenic_like"] = (
            aggregate["rf_call_fraction"] >= 0.5
        )
        aggregate["rf_consensus_class"] = np.select(
            [
                aggregate["rf_call_fraction"] >= 0.8,
                aggregate["rf_call_fraction"] <= 0.2,
            ],
            ["pathogenic_like_high_consensus", "neutral_like_high_consensus"],
            default="uncertain",
        )
        aggregate["rf_score_percentile_within_application"] = (
            aggregate["rf_score_mean"].rank(method="average", pct=True)
        )
        application_parts.append(aggregate)

    importance_detail = pd.DataFrame(importance_rows)
    importance_detail["importance"] = (
        importance_detail["baseline_value"]
        - importance_detail["permuted_value"]
    )
    importance_summary = (
        importance_detail.groupby(
            [
                "feature_panel",
                "importance_type",
                "feature_or_family",
                "metric",
            ],
            as_index=False,
        )["importance"]
        .agg(
            mean_importance="mean",
            sd_importance="std",
            min_importance="min",
            max_importance="max",
        )
    )
    return (
        pd.concat(application_parts, ignore_index=True),
        pd.concat(oof_parts, ignore_index=True),
        pd.DataFrame(threshold_rows),
        importance_summary,
    )


def summarize_oof(oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregate = (
        oof.groupby(
            ["feature_panel", "variant_id", "position", "target_pathogenic"],
            as_index=False,
        )
        .agg(
            rf_score_mean=("rf_score", "mean"),
            rf_score_sd=("rf_score", "std"),
            rf_call_fraction=("rf_call", "mean"),
            n_oof_predictions=("repeat", "size"),
        )
    )
    rows = []
    for panel_name, panel in aggregate.groupby("feature_panel", sort=False):
        values = binary_metrics(
            panel["target_pathogenic"].to_numpy(int),
            panel["rf_score_mean"].to_numpy(float),
            panel["rf_call_fraction"].ge(0.5).to_numpy(int),
        )
        for metric, value in values.items():
            rows.append({
                "feature_panel": panel_name,
                "metric": metric,
                "value": value,
                "evaluation": "aggregated_out_of_fold",
            })
    return aggregate, pd.DataFrame(rows)


def build_hybrid_table(
    application_predictions: pd.DataFrame,
    application: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    score_columns = [
        "rf_score_mean",
        "rf_score_sd",
        "rf_score_q025",
        "rf_score_q975",
        "rf_margin_mean",
        "rf_call_fraction",
        "rf_majority_pathogenic_like",
        "rf_consensus_class",
        "rf_score_percentile_within_application",
    ]
    wide_parts = []
    for panel_name, panel in application_predictions.groupby(
        "feature_panel", sort=False
    ):
        renamed = panel[["variant_id", *score_columns]].rename(
            columns={
                column: f"{panel_name}__{column}" for column in score_columns
            }
        )
        wide_parts.append(renamed)
    wide = wide_parts[0]
    for part in wide_parts[1:]:
        wide = wide.merge(part, on="variant_id", how="inner", validate="one_to_one")

    identity_columns = [
        "variant_id",
        "position",
        "reference",
        "alternate",
        "analysis_group",
        "rf_cohort",
        "isolation_forest_outlier_score",
        "isolation_forest_above_T95",
        "spectrum_group_primary_T95_preview",
        *FEATURE_COLS,
    ]
    hybrid = application[identity_columns].merge(
        wide, on="variant_id", how="inner", validate="one_to_one"
    )
    primary_call = hybrid[
        "current_all_9__rf_majority_pathogenic_like"
    ].astype(bool)
    sensitivity_call = hybrid[
        "without_phyloP__rf_majority_pathogenic_like"
    ].astype(bool)
    hybrid["rf_panel_call_agreement"] = np.select(
        [
            primary_call & sensitivity_call,
            primary_call & ~sensitivity_call,
            ~primary_call & sensitivity_call,
        ],
        [
            "both_pathogenic_like",
            "all9_only_pathogenic_like",
            "without_phyloP_only_pathogenic_like",
        ],
        default="neither_pathogenic_like",
    )

    unlabeled = hybrid[hybrid["rf_cohort"].eq("unlabeled")].copy()
    if_group = unlabeled["spectrum_group_primary_T95_preview"]
    call = unlabeled["current_all_9__rf_majority_pathogenic_like"].astype(bool)
    unlabeled["hybrid_if_rf_class"] = np.select(
        [
            if_group.eq(IF_EXPANDED) & ~call,
            if_group.eq(IF_EXPANDED) & call,
            if_group.eq(IF_OUT) & ~call,
            if_group.eq(IF_OUT) & call,
        ],
        [
            "inside_domain__rf_neutral_like",
            "inside_domain__rf_pathogenic_like",
            "outside_domain__rf_neutral_like",
            "outside_domain__rf_pathogenic_like",
        ],
        default="unclassified",
    )
    hybrid.loc[unlabeled.index, "hybrid_if_rf_class"] = unlabeled[
        "hybrid_if_rf_class"
    ]

    crosstab = (
        unlabeled.groupby(
            [
                "spectrum_group_primary_T95_preview",
                "rf_panel_call_agreement",
                "hybrid_if_rf_class",
            ],
            dropna=False,
        )
        .size()
        .rename("n_variants")
        .reset_index()
    )
    panel_agreement = (
        hybrid.groupby(["rf_cohort", "rf_panel_call_agreement"])
        .size()
        .rename("n_variants")
        .reset_index()
    )
    return hybrid, crosstab, panel_agreement


def build_application_summary(
    application_predictions: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        application_predictions.groupby(["feature_panel", "rf_cohort"])
        .agg(
            n_variants=("variant_id", "size"),
            n_majority_pathogenic_like=(
                "rf_majority_pathogenic_like", "sum"
            ),
            n_high_consensus_pathogenic_like=(
                "rf_consensus_class",
                lambda values: int(
                    np.sum(values == "pathogenic_like_high_consensus")
                ),
            ),
            n_uncertain=(
                "rf_consensus_class",
                lambda values: int(np.sum(values == "uncertain")),
            ),
            mean_call_fraction=("rf_call_fraction", "mean"),
            median_call_fraction=("rf_call_fraction", "median"),
            mean_rf_score=("rf_score_mean", "mean"),
        )
        .reset_index()
    )
    summary["majority_pathogenic_like_fraction"] = (
        summary["n_majority_pathogenic_like"] / summary["n_variants"]
    )
    summary["high_consensus_pathogenic_like_fraction"] = (
        summary["n_high_consensus_pathogenic_like"] / summary["n_variants"]
    )
    return summary


def build_feature_profiles(
    data: pd.DataFrame,
    hybrid: pd.DataFrame,
) -> pd.DataFrame:
    frames: list[tuple[str, pd.DataFrame]] = [
        (
            "clean_known_neutral",
            data[data["rf_cohort"].eq("clean_known_neutral")],
        ),
        (
            "clean_confirmed_pathogenic",
            data[data["rf_cohort"].eq("clean_confirmed_pathogenic")],
        ),
    ]
    unlabeled = hybrid[hybrid["rf_cohort"].eq("unlabeled")]
    for label, mask in [
        (
            "unlabeled_rf_neutral_like",
            ~unlabeled["current_all_9__rf_majority_pathogenic_like"].astype(bool),
        ),
        (
            "unlabeled_rf_pathogenic_like",
            unlabeled["current_all_9__rf_majority_pathogenic_like"].astype(bool),
        ),
        (
            "unlabeled_inside_if_domain",
            unlabeled["spectrum_group_primary_T95_preview"].eq(IF_EXPANDED),
        ),
        (
            "unlabeled_outside_if_domain",
            unlabeled["spectrum_group_primary_T95_preview"].eq(IF_OUT),
        ),
    ]:
        frames.append((label, unlabeled.loc[mask]))

    rows = []
    for cohort, frame in frames:
        for feature in FEATURE_COLS:
            values = frame[feature].astype(float)
            rows.append({
                "cohort": cohort,
                "feature": feature,
                "n_variants": len(values),
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "q25": float(values.quantile(0.25)),
                "median": float(values.median()),
                "q75": float(values.quantile(0.75)),
            })
    return pd.DataFrame(rows)


def plot_permutation_importance(
    importance: pd.DataFrame,
    output_path: Path,
) -> None:
    plot_data = importance[
        importance["importance_type"].eq("individual")
        & importance["metric"].eq("average_precision")
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=False)
    for ax, panel_name in zip(axes, APPLICATION_PANELS):
        panel = (
            plot_data[plot_data["feature_panel"].eq(panel_name)]
            .sort_values("mean_importance")
        )
        ax.barh(
            panel["feature_or_family"],
            panel["mean_importance"],
            xerr=panel["sd_importance"],
            capsize=3,
        )
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(panel_name)
        ax.set_xlabel("Held-out average-precision decrease")
    fig.suptitle("Random Forest held-out permutation importance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_score_distributions(
    application_predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    unlabeled = application_predictions[
        application_predictions["rf_cohort"].eq("unlabeled")
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    bins = np.linspace(0, 1, 26)
    colors = {IF_EXPANDED: "#5DA5DA", IF_OUT: "#F17CB0"}
    labels = {IF_EXPANDED: "Inside neutral domain", IF_OUT: "Outside neutral domain"}
    for ax, panel_name in zip(axes, APPLICATION_PANELS):
        panel = unlabeled[unlabeled["feature_panel"].eq(panel_name)]
        for group_name in [IF_EXPANDED, IF_OUT]:
            values = panel.loc[
                panel["spectrum_group_primary_T95_preview"].eq(group_name),
                "rf_call_fraction",
            ]
            ax.hist(
                values,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=2,
                color=colors[group_name],
                label=labels[group_name],
            )
        ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
        ax.set_title(panel_name)
        ax.set_xlabel("Fraction of calibrated RF models calling pathogenic-like")
        ax.set_xlim(0, 1)
    axes[0].set_ylabel("Density")
    axes[0].legend()
    fig.suptitle("Supervised RF calls within current Isolation Forest groups")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_joint_if_rf(hybrid: pd.DataFrame, output_path: Path) -> None:
    unlabeled = hybrid[hybrid["rf_cohort"].eq("unlabeled")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
    for ax, panel_name in zip(axes, APPLICATION_PANELS):
        values = unlabeled[f"{panel_name}__rf_call_fraction"]
        image = ax.hexbin(
            unlabeled["isolation_forest_outlier_score"],
            values,
            gridsize=55,
            bins="log",
            mincnt=1,
            cmap="viridis",
        )
        ax.axhline(0.5, color="white", linestyle="--", linewidth=1)
        ax.set_title(panel_name)
        ax.set_xlabel("Isolation Forest outlier score")
        fig.colorbar(image, ax=ax, label="log10 count")
    axes[0].set_ylabel("RF calibrated-call fraction")
    fig.suptitle("Isolation-domain novelty and supervised pathogenic similarity")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_hybrid_counts(hybrid: pd.DataFrame, output_path: Path) -> None:
    unlabeled = hybrid[hybrid["rf_cohort"].eq("unlabeled")]
    counts = unlabeled["hybrid_if_rf_class"].value_counts()
    order = [
        "inside_domain__rf_neutral_like",
        "inside_domain__rf_pathogenic_like",
        "outside_domain__rf_neutral_like",
        "outside_domain__rf_pathogenic_like",
    ]
    labels = [
        "Inside IF domain\nRF neutral-like",
        "Inside IF domain\nRF pathogenic-like",
        "Outside IF domain\nRF neutral-like",
        "Outside IF domain\nRF pathogenic-like",
    ]
    values = counts.reindex(order, fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(
        np.arange(len(order)),
        values,
        color=["#4C78A8", "#F2CF5B", "#B279A2", "#E45756"],
    )
    ax.bar_label(bars, labels=[f"{int(value):,}" for value in values], padding=3)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Unlabeled variants")
    ax.set_title("Hybrid Isolation Forest / Random Forest classification")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_panel_agreement(hybrid: pd.DataFrame, output_path: Path) -> None:
    unlabeled = hybrid[hybrid["rf_cohort"].eq("unlabeled")]
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.hexbin(
        unlabeled["current_all_9__rf_call_fraction"],
        unlabeled["without_phyloP__rf_call_fraction"],
        gridsize=45,
        bins="log",
        mincnt=1,
        cmap="magma",
    )
    ax.axvline(0.5, color="white", linestyle="--", linewidth=1)
    ax.axhline(0.5, color="white", linestyle="--", linewidth=1)
    ax.plot([0, 1], [0, 1], color="white", linewidth=0.8, alpha=0.8)
    ax.set_xlabel("All-nine RF calibrated-call fraction")
    ax.set_ylabel("Without-phyloP RF calibrated-call fraction")
    ax.set_title("Sensitivity of unlabeled calls to phyloP removal")
    fig.colorbar(image, ax=ax, label="log10 count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_decision_summary(
    audit: pd.DataFrame,
    application_summary: pd.DataFrame,
    oof_metrics: pd.DataFrame,
    hybrid: pd.DataFrame,
    *,
    outer_folds: int,
    repeats: int,
) -> dict:
    def audit_count(cohort: str) -> int:
        return int(
            audit.loc[audit["rf_cohort"].eq(cohort), "n_variants"].iloc[0]
        )

    def metric(panel: str, name: str) -> float:
        return float(
            oof_metrics.loc[
                oof_metrics["feature_panel"].eq(panel)
                & oof_metrics["metric"].eq(name),
                "value",
            ].iloc[0]
        )

    def summary_value(panel: str, cohort: str, column: str) -> float:
        return float(
            application_summary.loc[
                application_summary["feature_panel"].eq(panel)
                & application_summary["rf_cohort"].eq(cohort),
                column,
            ].iloc[0]
        )

    unlabeled = hybrid[hybrid["rf_cohort"].eq("unlabeled")]
    hybrid_counts = {
        str(key): int(value)
        for key, value in unlabeled["hybrid_if_rf_class"].value_counts().items()
    }
    agreement_counts = {
        str(key): int(value)
        for key, value in unlabeled["rf_panel_call_agreement"].value_counts().items()
    }
    return {
        "verdict": "use_random_forest_as_a_supervised_second_axis_not_as_a_replacement_for_isolation_forest",
        "strict_training_cohort": {
            "clean_neutral": audit_count("clean_known_neutral"),
            "clean_pathogenic": audit_count("clean_confirmed_pathogenic"),
            "excluded_neutral_pathogenic_overlap": audit_count(
                "excluded_neutral_pathogenic_overlap"
            ),
        },
        "application_cohort": {
            "unlabeled": audit_count("unlabeled"),
            "disease_suspected_posthoc": audit_count(
                "disease_suspected_posthoc"
            ),
        },
        "ensemble": {
            "outer_folds": outer_folds,
            "repeats": repeats,
            "models_per_panel": outer_folds * repeats,
            "threshold": "95th percentile of independent neutral calibration scores per model",
            "majority_call": "at least 50% of calibrated models call pathogenic-like",
            "high_consensus": "at least 80% of calibrated models call pathogenic-like",
        },
        "oof_metrics": {
            panel: {
                "roc_auc": metric(panel, "roc_auc"),
                "average_precision": metric(panel, "average_precision"),
                "precision": metric(panel, "precision"),
                "sensitivity": metric(panel, "sensitivity"),
                "specificity": metric(panel, "specificity"),
            }
            for panel in APPLICATION_PANELS
        },
        "unlabeled_application": {
            panel: {
                "majority_pathogenic_like_fraction": summary_value(
                    panel,
                    "unlabeled",
                    "majority_pathogenic_like_fraction",
                ),
                "high_consensus_pathogenic_like_fraction": summary_value(
                    panel,
                    "unlabeled",
                    "high_consensus_pathogenic_like_fraction",
                ),
            }
            for panel in APPLICATION_PANELS
        },
        "hybrid_counts_current_all_9": hybrid_counts,
        "feature_panel_agreement_counts": agreement_counts,
        "interpretation": {
            "rf_positive": "similar to curated Dataset 9 pathogenic rather than proof of pathogenicity",
            "if_positive": "outside the learned neutral domain rather than proof of pathogenicity",
            "important_limit": "phyloP and population-derived features overlap with Dataset 8 curation criteria; unlabeled predictions have no external ground truth",
        },
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    data, benchmark, application, audit = load_application_data(args.input)
    predictions, oof, thresholds, importance = run_cross_fitted_application(
        benchmark,
        application,
        outer_folds=args.outer_folds,
        repeats=args.repeats,
        random_state=args.random_state,
    )
    oof_aggregate, oof_metrics = summarize_oof(oof)
    hybrid, crosstab, panel_agreement = build_hybrid_table(
        predictions, application
    )
    application_summary = build_application_summary(predictions)
    feature_profiles = build_feature_profiles(data, hybrid)
    top_candidates = (
        hybrid[hybrid["rf_cohort"].eq("unlabeled")]
        .sort_values(
            [
                "current_all_9__rf_call_fraction",
                "without_phyloP__rf_call_fraction",
                "current_all_9__rf_score_mean",
            ],
            ascending=False,
        )
        .head(200)
    )

    audit.to_csv(args.output_dir / "application_cohort_audit.tsv", sep="\t", index=False)
    predictions.to_csv(
        args.output_dir / "application_predictions_by_panel.tsv",
        sep="\t",
        index=False,
    )
    oof_aggregate.to_csv(
        args.output_dir / "application_oof_predictions.tsv",
        sep="\t",
        index=False,
    )
    oof_metrics.to_csv(
        args.output_dir / "application_oof_metrics.tsv", sep="\t", index=False
    )
    thresholds.to_csv(
        args.output_dir / "application_model_thresholds.tsv",
        sep="\t",
        index=False,
    )
    importance.to_csv(
        args.output_dir / "heldout_permutation_importance.tsv",
        sep="\t",
        index=False,
    )
    hybrid.to_csv(
        args.output_dir / "hybrid_if_rf_classification.tsv",
        sep="\t",
        index=False,
    )
    crosstab.to_csv(
        args.output_dir / "hybrid_if_rf_crosstab.tsv", sep="\t", index=False
    )
    panel_agreement.to_csv(
        args.output_dir / "rf_panel_agreement.tsv", sep="\t", index=False
    )
    application_summary.to_csv(
        args.output_dir / "application_summary.tsv", sep="\t", index=False
    )
    feature_profiles.to_csv(
        args.output_dir / "feature_profiles_by_rf_group.tsv",
        sep="\t",
        index=False,
    )
    top_candidates.to_csv(
        args.output_dir / "top_200_unlabeled_rf_candidates.tsv",
        sep="\t",
        index=False,
    )

    plot_permutation_importance(
        importance, args.figure_dir / "heldout_permutation_importance.png"
    )
    plot_score_distributions(
        predictions, args.figure_dir / "rf_calls_by_isolation_group.png"
    )
    plot_joint_if_rf(
        hybrid, args.figure_dir / "isolation_vs_random_forest.png"
    )
    plot_hybrid_counts(
        hybrid, args.figure_dir / "hybrid_class_counts.png"
    )
    plot_panel_agreement(
        hybrid, args.figure_dir / "rf_panel_agreement.png"
    )

    decision = build_decision_summary(
        audit,
        application_summary,
        oof_metrics,
        hybrid,
        outer_folds=args.outer_folds,
        repeats=args.repeats,
    )
    (args.output_dir / "decision_summary.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
