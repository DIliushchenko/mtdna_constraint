"""Position-grouped benchmark of Random Forest versus Isolation Forest.

The benchmark intentionally does not replace the production neutral-domain model.
It evaluates whether supervised Random Forest separates clean known-neutral and
confirmed-pathogenic variants better than the current one-class approach when
both receive the same nine features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


FEATURE_COLS = [
    "mlc_score",
    "rarity_soft",
    "hom_rarity_soft",
    "het_rarity_soft",
    "no_homoplasmic_signal",
    "codon_pos1_any",
    "codon_pos2_any",
    "codon_pos3_any",
    "phyloP100way",
]

# The neutral reference set was partly curated using low phyloP and common
# haplogroup variants. Sensitivity panels therefore remove features that can
# reconstruct those curation rules. The all-nine panel remains the direct
# apples-to-apples comparison with notebook 02.
FEATURE_PANELS = {
    "current_all_9": FEATURE_COLS,
    "without_phyloP": [
        feature for feature in FEATURE_COLS if feature != "phyloP100way"
    ],
    "mlc_and_codon_only": [
        "mlc_score",
        "codon_pos1_any",
        "codon_pos2_any",
        "codon_pos3_any",
    ],
    "mlc_only": ["mlc_score"],
}

MODEL_LABELS = {
    "isolation_forest": "Isolation Forest",
    "random_forest": "Random Forest",
}

METRIC_LABELS = {
    "roc_auc": "ROC AUC",
    "average_precision": "Average precision",
    "balanced_accuracy": "Balanced accuracy",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "precision": "Precision",
    "f1": "F1",
    "mcc": "MCC",
}


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
        default=root / "results/model_random_forest_benchmark",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=root / "results/figures/model_random_forest_benchmark",
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=20260810)
    return parser.parse_args()


def load_clean_benchmark(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(path, sep="\t", low_memory=False)
    required = {
        "variant_id",
        "position",
        "is_neutral_dataset8",
        "is_pathogenic_dataset9",
        *FEATURE_COLS,
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing benchmark columns: {missing}")

    for col in ["is_neutral_dataset8", "is_pathogenic_dataset9", *FEATURE_COLS]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data["position"] = pd.to_numeric(data["position"], errors="coerce")

    overlap = (
        data["is_neutral_dataset8"].eq(1)
        & data["is_pathogenic_dataset9"].eq(1)
    )
    neutral = (
        data["is_neutral_dataset8"].eq(1)
        & data["is_pathogenic_dataset9"].ne(1)
    )
    pathogenic = (
        data["is_pathogenic_dataset9"].eq(1)
        & data["is_neutral_dataset8"].ne(1)
    )

    benchmark = data.loc[neutral | pathogenic].copy()
    benchmark["target_pathogenic"] = pathogenic.loc[benchmark.index].astype(int)
    benchmark["benchmark_label"] = np.where(
        benchmark["target_pathogenic"].eq(1),
        "confirmed_pathogenic",
        "known_neutral",
    )
    benchmark = benchmark.reset_index(drop=True)

    if benchmark[["position", *FEATURE_COLS]].isna().any().any():
        missing_counts = benchmark[["position", *FEATURE_COLS]].isna().sum()
        raise ValueError(
            "Missing values in benchmark matrix:\n"
            f"{missing_counts[missing_counts.gt(0)]}"
        )
    if benchmark["variant_id"].duplicated().any():
        raise ValueError("Duplicate variant_id values in benchmark cohort.")

    audit = pd.DataFrame(
        [
            {
                "cohort": "all_input_variants",
                "n_variants": len(data),
                "n_positions": data["position"].nunique(),
            },
            {
                "cohort": "excluded_neutral_pathogenic_overlap",
                "n_variants": int(overlap.sum()),
                "n_positions": data.loc[overlap, "position"].nunique(),
            },
            {
                "cohort": "clean_known_neutral",
                "n_variants": int(neutral.sum()),
                "n_positions": data.loc[neutral, "position"].nunique(),
            },
            {
                "cohort": "clean_confirmed_pathogenic",
                "n_variants": int(pathogenic.sum()),
                "n_positions": data.loc[pathogenic, "position"].nunique(),
            },
        ]
    )
    return benchmark, audit


def make_random_forest(random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=1000,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )


def make_isolation_forest(random_state: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", RobustScaler()),
            (
                "model",
                IsolationForest(
                    n_estimators=500,
                    contamination="auto",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def score_isolation_forest(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    return -model.decision_function(features)


def threshold_at_neutral_tail(scores: np.ndarray, neutral_mask: np.ndarray) -> float:
    neutral_scores = np.asarray(scores)[np.asarray(neutral_mask, dtype=bool)]
    if len(neutral_scores) < 20:
        raise ValueError("Too few neutral calibration observations.")
    return float(np.quantile(neutral_scores, 0.95))


def binary_metrics(
    target: np.ndarray,
    score: np.ndarray,
    call: np.ndarray,
) -> dict[str, float]:
    target = np.asarray(target, dtype=int)
    call = np.asarray(call, dtype=int)
    tn, fp, fn, tp = confusion_matrix(target, call, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(target, score)),
        "average_precision": float(average_precision_score(target, score)),
        "balanced_accuracy": float(balanced_accuracy_score(target, call)),
        "sensitivity": float(recall_score(target, call, zero_division=0)),
        "specificity": float(tn / (tn + fp)),
        "precision": float(precision_score(target, call, zero_division=0)),
        "f1": float(f1_score(target, call, zero_division=0)),
        "mcc": float(matthews_corrcoef(target, call)),
    }


def run_repeated_grouped_cv(
    benchmark: pd.DataFrame,
    *,
    outer_folds: int,
    repeats: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target = benchmark["target_pathogenic"].to_numpy(int)
    groups = benchmark["position"].to_numpy()
    features = benchmark[FEATURE_COLS]
    prediction_parts: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    importance_rows: list[dict] = []

    for repeat in range(repeats):
        repeat_seed = random_state + repeat * 1000
        outer_cv = StratifiedGroupKFold(
            n_splits=outer_folds,
            shuffle=True,
            random_state=repeat_seed,
        )
        for fold, (outer_train_idx, test_idx) in enumerate(
            outer_cv.split(features, target, groups), start=1
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
                    features.iloc[outer_train_idx],
                    inner_target,
                    inner_groups,
                )
            )
            model_idx = outer_train_idx[model_rel_idx]
            calibration_idx = outer_train_idx[calibration_rel_idx]

            if set(target[model_idx]) != {0, 1}:
                raise ValueError("A model-training split lacks one class.")
            calibration_neutral = target[calibration_idx] == 0

            neutral_model_idx = model_idx[target[model_idx] == 0]
            for panel_index, (feature_panel, panel_cols) in enumerate(
                FEATURE_PANELS.items()
            ):
                fold_seed = repeat_seed + fold + panel_index * 100
                panel_features = benchmark[panel_cols]

                random_forest = make_random_forest(fold_seed)
                random_forest.fit(
                    panel_features.iloc[model_idx], target[model_idx]
                )
                rf_calibration_score = random_forest.predict_proba(
                    panel_features.iloc[calibration_idx]
                )[:, 1]
                rf_threshold = threshold_at_neutral_tail(
                    rf_calibration_score, calibration_neutral
                )
                rf_test_score = random_forest.predict_proba(
                    panel_features.iloc[test_idx]
                )[:, 1]
                rf_test_call = rf_test_score > rf_threshold

                isolation_forest = make_isolation_forest(fold_seed)
                isolation_forest.fit(panel_features.iloc[neutral_model_idx])
                if_calibration_score = score_isolation_forest(
                    isolation_forest, panel_features.iloc[calibration_idx]
                )
                if_threshold = threshold_at_neutral_tail(
                    if_calibration_score, calibration_neutral
                )
                if_test_score = score_isolation_forest(
                    isolation_forest, panel_features.iloc[test_idx]
                )
                if_test_call = if_test_score > if_threshold

                fold_predictions = benchmark.loc[
                    test_idx,
                    [
                        "variant_id",
                        "position",
                        "benchmark_label",
                        "target_pathogenic",
                    ],
                ].copy()
                fold_predictions["feature_panel"] = feature_panel
                fold_predictions["repeat"] = repeat + 1
                fold_predictions["fold"] = fold
                fold_predictions["random_forest_score"] = rf_test_score
                fold_predictions["random_forest_threshold"] = rf_threshold
                fold_predictions["random_forest_call"] = rf_test_call.astype(int)
                fold_predictions["isolation_forest_score"] = if_test_score
                fold_predictions["isolation_forest_threshold"] = if_threshold
                fold_predictions["isolation_forest_call"] = if_test_call.astype(int)
                prediction_parts.append(fold_predictions)

                for model_name, score, call, threshold in [
                    ("random_forest", rf_test_score, rf_test_call, rf_threshold),
                    ("isolation_forest", if_test_score, if_test_call, if_threshold),
                ]:
                    values = binary_metrics(target[test_idx], score, call)
                    for metric, value in values.items():
                        metric_rows.append(
                            {
                                "feature_panel": feature_panel,
                                "repeat": repeat + 1,
                                "fold": fold,
                                "model": model_name,
                                "metric": metric,
                                "value": value,
                                "threshold": threshold,
                                "n_test": len(test_idx),
                                "n_pathogenic_test": int(target[test_idx].sum()),
                            }
                        )

                for feature, value in zip(
                    panel_cols, random_forest.feature_importances_
                ):
                    importance_rows.append(
                        {
                            "feature_panel": feature_panel,
                            "repeat": repeat + 1,
                            "fold": fold,
                            "feature": feature,
                            "mdi_importance": float(value),
                        }
                    )

    predictions = pd.concat(prediction_parts, ignore_index=True)
    fold_metrics = pd.DataFrame(metric_rows)
    feature_importance = pd.DataFrame(importance_rows)
    expected_predictions = len(benchmark) * repeats * len(FEATURE_PANELS)
    if len(predictions) != expected_predictions:
        raise ValueError(
            f"Expected {expected_predictions} OOF rows, found {len(predictions)}."
        )
    return predictions, fold_metrics, feature_importance


def aggregate_oof_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    aggregate = (
        predictions.groupby(
            [
                "feature_panel",
                "variant_id",
                "position",
                "benchmark_label",
                "target_pathogenic",
            ],
            as_index=False,
        )
        .agg(
            random_forest_score=("random_forest_score", "mean"),
            random_forest_call_fraction=("random_forest_call", "mean"),
            isolation_forest_score=("isolation_forest_score", "mean"),
            isolation_forest_call_fraction=("isolation_forest_call", "mean"),
            n_oof_predictions=("repeat", "size"),
        )
    )
    aggregate["random_forest_call"] = (
        aggregate["random_forest_call_fraction"] >= 0.5
    ).astype(int)
    aggregate["isolation_forest_call"] = (
        aggregate["isolation_forest_call_fraction"] >= 0.5
    ).astype(int)
    return aggregate


def _cluster_bootstrap_single_panel(
    aggregate: pd.DataFrame,
    *,
    n_bootstrap: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target = aggregate["target_pathogenic"].to_numpy(int)
    positions = aggregate["position"].to_numpy()
    unique_positions = np.unique(positions)
    position_rows = [np.flatnonzero(positions == value) for value in unique_positions]
    model_arrays = {
        "random_forest": (
            aggregate["random_forest_score"].to_numpy(float),
            aggregate["random_forest_call"].to_numpy(int),
        ),
        "isolation_forest": (
            aggregate["isolation_forest_score"].to_numpy(float),
            aggregate["isolation_forest_call"].to_numpy(int),
        ),
    }

    point_rows: list[dict] = []
    point_by_model: dict[str, dict[str, float]] = {}
    for model_name, (score, call) in model_arrays.items():
        point_by_model[model_name] = binary_metrics(target, score, call)
        for metric, value in point_by_model[model_name].items():
            point_rows.append({"model": model_name, "metric": metric, "value": value})

    rng = np.random.default_rng(random_state)
    bootstrap_rows: list[dict] = []
    for bootstrap_index in range(n_bootstrap):
        sampled_positions = rng.integers(
            0, len(unique_positions), size=len(unique_positions)
        )
        sampled_rows = np.concatenate(
            [position_rows[index] for index in sampled_positions]
        )
        sampled_target = target[sampled_rows]
        if sampled_target.min() == sampled_target.max():
            continue
        bootstrap_values: dict[str, dict[str, float]] = {}
        for model_name, (score, call) in model_arrays.items():
            values = binary_metrics(
                sampled_target,
                score[sampled_rows],
                call[sampled_rows],
            )
            bootstrap_values[model_name] = values
            for metric, value in values.items():
                bootstrap_rows.append(
                    {
                        "bootstrap": bootstrap_index + 1,
                        "model": model_name,
                        "metric": metric,
                        "value": value,
                    }
                )
        for metric in point_by_model["random_forest"]:
            bootstrap_rows.append(
                {
                    "bootstrap": bootstrap_index + 1,
                    "model": "random_forest_minus_isolation_forest",
                    "metric": metric,
                    "value": (
                        bootstrap_values["random_forest"][metric]
                        - bootstrap_values["isolation_forest"][metric]
                    ),
                }
            )

    bootstrap = pd.DataFrame(bootstrap_rows)
    point = pd.DataFrame(point_rows)
    summary = (
        bootstrap.groupby(["model", "metric"])["value"]
        .agg(
            bootstrap_mean="mean",
            ci_lower=lambda values: values.quantile(0.025),
            ci_upper=lambda values: values.quantile(0.975),
        )
        .reset_index()
    )
    model_summary = point.merge(
        summary[summary["model"].isin(MODEL_LABELS)].drop(
            columns="bootstrap_mean"
        ),
        on=["model", "metric"],
        how="left",
        validate="one_to_one",
    )

    differences = summary[
        summary["model"].eq("random_forest_minus_isolation_forest")
    ].copy()
    point_pivot = point.pivot(index="metric", columns="model", values="value")
    differences["value"] = differences["metric"].map(
        point_pivot["random_forest"] - point_pivot["isolation_forest"]
    )
    differences["probability_difference_gt_zero"] = differences["metric"].map(
        bootstrap[
            bootstrap["model"].eq("random_forest_minus_isolation_forest")
        ]
        .groupby("metric")["value"]
        .agg(lambda values: float(np.mean(values > 0)))
    )
    differences = differences[
        [
            "model",
            "metric",
            "value",
            "bootstrap_mean",
            "ci_lower",
            "ci_upper",
            "probability_difference_gt_zero",
        ]
    ]
    return model_summary, differences, bootstrap


def cluster_bootstrap_metrics(
    aggregate: pd.DataFrame,
    *,
    n_bootstrap: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_parts: list[pd.DataFrame] = []
    difference_parts: list[pd.DataFrame] = []
    bootstrap_parts: list[pd.DataFrame] = []
    for panel_index, feature_panel in enumerate(FEATURE_PANELS):
        panel_data = aggregate[
            aggregate["feature_panel"].eq(feature_panel)
        ].copy()
        model_summary, differences, bootstrap = _cluster_bootstrap_single_panel(
            panel_data,
            n_bootstrap=n_bootstrap,
            random_state=random_state + panel_index * 10_000,
        )
        for table in [model_summary, differences, bootstrap]:
            table.insert(0, "feature_panel", feature_panel)
        model_parts.append(model_summary)
        difference_parts.append(differences)
        bootstrap_parts.append(bootstrap)
    return (
        pd.concat(model_parts, ignore_index=True),
        pd.concat(difference_parts, ignore_index=True),
        pd.concat(bootstrap_parts, ignore_index=True),
    )


def summarize_repeat_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (feature_panel, repeat), repeat_df in predictions.groupby(
        ["feature_panel", "repeat"]
    ):
        target = repeat_df["target_pathogenic"].to_numpy(int)
        for model_name in MODEL_LABELS:
            values = binary_metrics(
                target,
                repeat_df[f"{model_name}_score"].to_numpy(float),
                repeat_df[f"{model_name}_call"].to_numpy(int),
            )
            for metric, value in values.items():
                rows.append(
                    {
                        "feature_panel": feature_panel,
                        "repeat": repeat,
                        "model": model_name,
                        "metric": metric,
                        "value": value,
                    }
                )
    return pd.DataFrame(rows)


def plot_curves(aggregate: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(
        len(FEATURE_PANELS), 2, figsize=(13, 4.6 * len(FEATURE_PANELS))
    )
    for row, feature_panel in enumerate(FEATURE_PANELS):
        panel_data = aggregate[
            aggregate["feature_panel"].eq(feature_panel)
        ]
        target = panel_data["target_pathogenic"].to_numpy(int)
        for model_name, label in MODEL_LABELS.items():
            score = panel_data[f"{model_name}_score"].to_numpy(float)
            false_positive_rate, true_positive_rate, _ = roc_curve(target, score)
            precision, recall, _ = precision_recall_curve(target, score)
            axes[row, 0].plot(
                false_positive_rate,
                true_positive_rate,
                linewidth=2,
                label=f"{label}: {roc_auc_score(target, score):.3f}",
            )
            axes[row, 1].plot(
                recall,
                precision,
                linewidth=2,
                label=f"{label}: {average_precision_score(target, score):.3f}",
            )
        axes[row, 0].plot(
            [0, 1], [0, 1], linestyle="--", color="grey", linewidth=1
        )
        axes[row, 0].set(xlabel="False-positive rate", ylabel="True-positive rate")
        axes[row, 0].set_title(f"{feature_panel}: position-grouped ROC")
        axes[row, 0].legend(title="ROC AUC")
        prevalence = float(target.mean())
        axes[row, 1].axhline(
            prevalence, linestyle="--", color="grey", linewidth=1
        )
        axes[row, 1].set(xlabel="Recall", ylabel="Precision")
        axes[row, 1].set_title(
            f"{feature_panel}: precision-recall (prevalence {prevalence:.2%})"
        )
        axes[row, 1].legend(title="Average precision")
    fig.suptitle("Held-out clean labels; groups are mtDNA positions")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metric_comparison(summary: pd.DataFrame, output_path: Path) -> None:
    metrics = [
        "roc_auc",
        "average_precision",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
    ]
    x = np.arange(len(metrics))
    width = 0.36
    fig, axes = plt.subplots(2, 2, figsize=(17, 12), sharey=True)
    for ax, feature_panel in zip(axes.flat, FEATURE_PANELS):
        plot_df = summary[
            summary["feature_panel"].eq(feature_panel)
            & summary["metric"].isin(metrics)
        ]
        for index, model_name in enumerate(MODEL_LABELS):
            model_df = (
                plot_df[plot_df["model"].eq(model_name)]
                .set_index("metric")
                .loc[metrics]
            )
            position = x + (index - 0.5) * width
            y = model_df["value"].to_numpy(float)
            yerr = np.vstack(
                [
                    np.maximum(y - model_df["ci_lower"].to_numpy(float), 0),
                    np.maximum(model_df["ci_upper"].to_numpy(float) - y, 0),
                ]
            )
            ax.bar(position, y, width=width, label=MODEL_LABELS[model_name])
            ax.errorbar(
                position, y, yerr=yerr, fmt="none", color="black", capsize=3
            )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [METRIC_LABELS[value] for value in metrics], rotation=28, ha="right"
        )
        ax.set_ylim(0, 1.05)
        ax.set_title(feature_panel)
        ax.legend(fontsize=8)
    axes[0, 0].set_ylabel("Metric value (95% position-bootstrap CI)")
    axes[1, 0].set_ylabel("Metric value (95% position-bootstrap CI)")
    fig.suptitle("Random Forest versus Isolation Forest across feature panels")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(feature_importance: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    summary = (
        feature_importance.groupby(
            ["feature_panel", "feature"], as_index=False
        )["mdi_importance"]
        .agg(
            mean_importance="mean",
            sd_importance="std",
            min_importance="min",
            max_importance="max",
        )
        .sort_values(["feature_panel", "mean_importance"])
    )
    plot_data = summary[summary["feature_panel"].eq("current_all_9")]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        plot_data["feature"],
        plot_data["mean_importance"],
        xerr=plot_data["sd_importance"],
        capsize=3,
    )
    ax.set_xlabel("Mean impurity importance across outer fits")
    ax.set_title("Random Forest feature use (descriptive, correlated features)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return summary.sort_values(
        ["feature_panel", "mean_importance"], ascending=[True, False]
    )


def build_decision_summary(
    audit: pd.DataFrame,
    model_summary: pd.DataFrame,
    differences: pd.DataFrame,
    *,
    outer_folds: int,
    repeats: int,
) -> dict:
    def metric_value(
        table: pd.DataFrame,
        feature_panel: str,
        model: str,
        metric: str,
        column: str,
    ) -> float:
        return float(
            table.loc[
                table["feature_panel"].eq(feature_panel)
                & table["model"].eq(model)
                & table["metric"].eq(metric),
                column,
            ].iloc[0]
        )

    panel_readout = {}
    robust_panels = []
    for feature_panel in FEATURE_PANELS:
        roc_delta_low = metric_value(
            differences,
            feature_panel,
            "random_forest_minus_isolation_forest",
            "roc_auc",
            "ci_lower",
        )
        ap_delta_low = metric_value(
            differences,
            feature_panel,
            "random_forest_minus_isolation_forest",
            "average_precision",
            "ci_lower",
        )
        robust = roc_delta_low > 0 and ap_delta_low > 0
        if robust:
            robust_panels.append(feature_panel)
        panel_readout[feature_panel] = {
            "random_forest_roc_auc": metric_value(
                model_summary, feature_panel, "random_forest", "roc_auc", "value"
            ),
            "isolation_forest_roc_auc": metric_value(
                model_summary,
                feature_panel,
                "isolation_forest",
                "roc_auc",
                "value",
            ),
            "roc_auc_difference_ci_lower": roc_delta_low,
            "random_forest_average_precision": metric_value(
                model_summary,
                feature_panel,
                "random_forest",
                "average_precision",
                "value",
            ),
            "isolation_forest_average_precision": metric_value(
                model_summary,
                feature_panel,
                "isolation_forest",
                "average_precision",
                "value",
            ),
            "average_precision_difference_ci_lower": ap_delta_low,
            "robust_random_forest_advantage": robust,
        }

    if len(robust_panels) == len(FEATURE_PANELS):
        verdict = "random_forest_better_across_all_feature_sensitivity_panels"
    elif "current_all_9" in robust_panels:
        verdict = "random_forest_better_for_current_features_but_not_all_sensitivities"
    else:
        verdict = "no_robust_random_forest_advantage"

    cohort_counts = audit.set_index("cohort")["n_variants"].astype(int).to_dict()
    return {
        "verdict": verdict,
        "comparison_scope": "clean known-neutral versus confirmed-pathogenic variants",
        "neutral_variants": cohort_counts["clean_known_neutral"],
        "pathogenic_variants": cohort_counts["clean_confirmed_pathogenic"],
        "excluded_label_overlaps": cohort_counts[
            "excluded_neutral_pathogenic_overlap"
        ],
        "outer_folds": outer_folds,
        "repeats": repeats,
        "split_group": "mtDNA position",
        "threshold_rule": "95th percentile of inner-calibration neutral scores",
        "feature_panels": panel_readout,
        "panels_with_robust_random_forest_advantage": robust_panels,
        "curation_circularity_warning": (
            "The neutral reference includes variants selected for low phyloP and "
            "haplogroup status; all-nine performance can partly reconstruct label "
            "curation rather than independent biology."
        ),
        "important_limit": (
            "Random Forest uses pathogenic labels during training; this benchmark "
            "does not establish validity for unlabeled variants or justify replacing "
            "the neutral-domain pipeline without external validation."
        ),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    benchmark, audit = load_clean_benchmark(args.input)
    predictions, fold_metrics, feature_importance = run_repeated_grouped_cv(
        benchmark,
        outer_folds=args.outer_folds,
        repeats=args.repeats,
        random_state=args.random_state,
    )
    repeat_metrics = summarize_repeat_metrics(predictions)
    aggregate = aggregate_oof_predictions(predictions)
    model_summary, differences, bootstrap = cluster_bootstrap_metrics(
        aggregate,
        n_bootstrap=args.bootstrap,
        random_state=args.random_state + 999_999,
    )

    audit.to_csv(args.output_dir / "benchmark_cohort_audit.tsv", sep="\t", index=False)
    aggregate.to_csv(
        args.output_dir / "aggregated_oof_predictions.tsv", sep="\t", index=False
    )
    fold_metrics.to_csv(
        args.output_dir / "fold_metrics.tsv", sep="\t", index=False
    )
    repeat_metrics.to_csv(
        args.output_dir / "repeat_metrics.tsv", sep="\t", index=False
    )
    model_summary.to_csv(
        args.output_dir / "model_metric_summary.tsv", sep="\t", index=False
    )
    differences.to_csv(
        args.output_dir / "random_forest_minus_isolation_forest.tsv",
        sep="\t",
        index=False,
    )
    plot_curves(aggregate, args.figure_dir / "roc_pr_comparison.png")
    plot_metric_comparison(model_summary, args.figure_dir / "metric_comparison.png")
    importance_summary = plot_feature_importance(
        feature_importance, args.figure_dir / "random_forest_feature_importance.png"
    )
    importance_summary.to_csv(
        args.output_dir / "random_forest_feature_importance.tsv",
        sep="\t",
        index=False,
    )

    decision = build_decision_summary(
        audit,
        model_summary,
        differences,
        outer_folds=args.outer_folds,
        repeats=args.repeats,
    )
    (args.output_dir / "decision_summary.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(decision, indent=2, ensure_ascii=False))
    print("\nModel metric summary:")
    print(model_summary.to_string(index=False))
    print("\nRandom Forest minus Isolation Forest:")
    print(differences.to_string(index=False))


if __name__ == "__main__":
    main()
