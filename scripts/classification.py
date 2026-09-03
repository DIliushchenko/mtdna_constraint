"""Model and label-granularity selection for mtDNA substitution classification.

Notebook 02 fits a one-class Isolation Forest and splits variants into a neutral
domain and an out-of-neutral domain. Notebooks 02c and 02d benchmark and apply a
binary Random Forest. Neither answers two questions that decide the downstream
analysis:

1. How many ordered severity classes are actually separable in these features?
2. Which supervised model, with which subset of the nine features, should carry
   the classification of all possible mtDNA substitutions?

This module answers both with the same evaluation protocol. Three label schemes
(2, 3 and 4 ordered classes) are built from Lake 2024 supplementary datasets 3,
8 and 9 under an explicit precedence rule. A fixed zoo of multiclass models is
tuned by randomized search, scored by repeated position-grouped cross
validation, ranked by a Friedman/Nemenyi critical-difference test, confirmed by
nested cross validation, and reduced to a parsimonious feature subset by forward
selection under a one-standard-error rule.

`mlc_score` and `phyloP100way` are forced into every feature subset by request.
That makes one bias explicit rather than hidden: 3,972 of the 8,284 dataset 8
neutral variants were selected as the lowest phyloP decile, so phyloP partly
defines the neutral label instead of independently predicting it. Every headline
number is therefore recomputed on a leakage-control cohort that keeps only the
haplogroup-selected neutral variants, where that circularity is absent.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*The least populated class.*")


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

from features import (
    CONSEQUENCE_FEATURES,
    FEATURE_COLUMNS,
    FORCED_FEATURES,
    SCHEME_CLASSES,
    attach_label_sources,
    build_label_schemes,
)

# The panel lives in features.py so the notebook, the builder and this module
# cannot drift apart.
CORE_FEATURES = FEATURE_COLUMNS

FEATURE_PANELS = {
    "all": CORE_FEATURES,
    "forced_pair": FORCED_FEATURES,
    "forced_plus_consequence": FORCED_FEATURES + CONSEQUENCE_FEATURES,
    "forced_plus_frequency": FORCED_FEATURES + ["hom_rarity_soft"],
}

CLASS_COLORS = {
    "benign": "#4C78A8",
    "vus": "#F2A93B",
    "vus_low": "#9ECAE9",
    "vus_high": "#F58518",
    "pathogenic": "#C4453C",
    "unlabeled": "#BAB0AC",
}

def label_audit_table(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scheme, classes in SCHEME_CLASSES.items():
        column = f"label_{scheme}"
        counts = data[column].value_counts(dropna=False)
        for class_name in classes:
            subset = data[data[column].eq(class_name)]
            rows.append(
                {
                    "scheme": scheme,
                    "class": class_name,
                    "n_variants": int(counts.get(class_name, 0)),
                    "n_positions": int(subset["position"].nunique()),
                    "n_haplogroup_neutral": int(
                        subset["neutral_selection_rule"].eq("haplogroup_variant").sum()
                    ),
                    "n_lowest_phylop_neutral": int(
                        subset["neutral_selection_rule"].eq("lowest_decile_phyloP").sum()
                    ),
                    "median_phyloP100way": float(subset["phyloP100way"].median()),
                    "median_mlc_score": float(subset["mlc_score"].median()),
                }
            )
        rows.append(
            {
                "scheme": scheme,
                "class": "unlabeled_prediction_target",
                "n_variants": int(data[column].isna().sum()),
                "n_positions": int(data.loc[data[column].isna(), "position"].nunique()),
                "n_haplogroup_neutral": 0,
                "n_lowest_phylop_neutral": 0,
                "median_phyloP100way": float(
                    data.loc[data[column].isna(), "phyloP100way"].median()
                ),
                "median_mlc_score": float(
                    data.loc[data[column].isna(), "mlc_score"].median()
                ),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Ordinal model (Frank & Hall decomposition)
# ---------------------------------------------------------------------------


class OrdinalClassifier(BaseEstimator, ClassifierMixin):
    """Ordered-class classifier built from K-1 cumulative binary problems.

    For ordered classes 0..K-1 it fits P(y > k) for each k and differences the
    cumulative probabilities. Unlike a plain multiclass fit, this respects the
    benign < VUS < pathogenic ordering, so confusing benign with pathogenic is
    structurally harder than confusing neighbouring classes.
    """

    def __init__(self, base_estimator=None, n_estimators=300, max_depth=None,
                 min_samples_leaf=1, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def _make_base(self):
        if self.base_estimator is not None:
            return clone(self.base_estimator)
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            class_weight="balanced_subsample",
            random_state=self.random_state,
            n_jobs=1,
        )

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if self.classes_.size < 2:
            raise ValueError("OrdinalClassifier needs at least two classes")
        self.estimators_ = []
        for k in range(self.classes_.size - 1):
            target = (y > self.classes_[k]).astype(int)
            model = self._make_base()
            model.fit(X, target)
            self.estimators_.append(model)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        n_classes = self.classes_.size
        cumulative = np.zeros((X.shape[0], n_classes - 1))
        for k, model in enumerate(self.estimators_):
            positive_index = list(model.classes_).index(1)
            cumulative[:, k] = model.predict_proba(X)[:, positive_index]
        # Enforce monotone non-increasing cumulative probabilities.
        cumulative = np.minimum.accumulate(cumulative, axis=1)
        proba = np.zeros((X.shape[0], n_classes))
        proba[:, 0] = 1.0 - cumulative[:, 0]
        for k in range(1, n_classes - 1):
            proba[:, k] = cumulative[:, k - 1] - cumulative[:, k]
        proba[:, -1] = cumulative[:, -1]
        proba = np.clip(proba, 1e-12, None)
        return proba / proba.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ---------------------------------------------------------------------------
# Model zoo
# ---------------------------------------------------------------------------


@dataclass
class ModelSpec:
    name: str
    label: str
    estimator: object
    param_distributions: dict = field(default_factory=dict)
    n_iter: int = 40
    needs_scaling: bool = False


def make_model_zoo(random_state: int, budget: str = "thorough") -> dict[str, ModelSpec]:
    scale = {"fast": 0.35, "medium": 0.7, "thorough": 1.0}[budget]

    def iters(n: int) -> int:
        return max(4, int(round(n * scale)))

    specs = [
        ModelSpec(
            name="logistic_regression",
            label="Multinomial logistic regression",
            estimator=LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=random_state,
            ),
            param_distributions={
                "model__C": stats.loguniform(1e-3, 1e3),
                "model__solver": ["lbfgs", "newton-cg"],
            },
            n_iter=iters(30),
            needs_scaling=True,
        ),
        ModelSpec(
            name="random_forest",
            label="Random Forest",
            estimator=RandomForestClassifier(
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=1,
            ),
            param_distributions={
                "model__n_estimators": [300, 600, 1000],
                "model__max_depth": [None, 4, 6, 8, 12, 16],
                "model__min_samples_leaf": [1, 2, 4, 8, 16],
                "model__max_features": ["sqrt", "log2", 0.5, 0.8, None],
            },
            n_iter=iters(45),
        ),
        ModelSpec(
            name="extra_trees",
            label="Extremely randomized trees",
            estimator=ExtraTreesClassifier(
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=1,
            ),
            param_distributions={
                "model__n_estimators": [300, 600, 1000],
                "model__max_depth": [None, 6, 10, 16],
                "model__min_samples_leaf": [1, 2, 4, 8],
                "model__max_features": ["sqrt", "log2", 0.5, 0.8, None],
            },
            n_iter=iters(40),
        ),
        ModelSpec(
            name="hist_gradient_boosting",
            label="Histogram gradient boosting",
            estimator=HistGradientBoostingClassifier(
                class_weight="balanced",
                early_stopping=False,
                random_state=random_state,
            ),
            param_distributions={
                "model__learning_rate": stats.loguniform(0.01, 0.4),
                "model__max_iter": [150, 300, 500],
                "model__max_leaf_nodes": [7, 15, 31, 63],
                "model__min_samples_leaf": [5, 10, 20, 40],
                "model__l2_regularization": stats.loguniform(1e-4, 10.0),
            },
            n_iter=iters(45),
        ),
        ModelSpec(
            name="svm_rbf",
            label="Support vector machine (RBF)",
            estimator=SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=random_state,
            ),
            param_distributions={
                "model__C": stats.loguniform(1e-2, 1e3),
                "model__gamma": stats.loguniform(1e-4, 1e1),
            },
            n_iter=iters(20),
            needs_scaling=True,
        ),
        ModelSpec(
            name="k_nearest_neighbors",
            label="k-nearest neighbours",
            estimator=KNeighborsClassifier(n_jobs=1),
            param_distributions={
                "model__n_neighbors": [5, 10, 20, 40, 80],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
            n_iter=iters(20),
            needs_scaling=True,
        ),
        ModelSpec(
            name="ordinal_forest",
            label="Ordinal forest (Frank & Hall)",
            estimator=OrdinalClassifier(random_state=random_state),
            param_distributions={
                "model__n_estimators": [300, 600],
                "model__max_depth": [None, 6, 10, 16],
                "model__min_samples_leaf": [1, 2, 4, 8],
            },
            n_iter=iters(25),
        ),
        ModelSpec(
            name="prior_baseline",
            label="Stratified prior baseline",
            estimator=DummyClassifier(strategy="stratified", random_state=random_state),
            param_distributions={},
            n_iter=1,
        ),
    ]
    return {spec.name: spec for spec in specs}


def build_pipeline(spec: ModelSpec) -> Pipeline:
    steps = []
    if spec.needs_scaling:
        steps.append(("scaler", RobustScaler()))
    steps.append(("model", clone(spec.estimator)))
    return Pipeline(steps)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def scheme_cohort(
    data: pd.DataFrame, scheme: str, leakage_control: bool = False
) -> pd.DataFrame:
    column = f"label_{scheme}"
    cohort = data[data[column].notna()].copy()
    if leakage_control:
        cohort = cohort[cohort["in_leakage_control_cohort"]].copy()
    cohort["target_name"] = cohort[column]
    order = {name: idx for idx, name in enumerate(SCHEME_CLASSES[scheme])}
    cohort["target"] = cohort["target_name"].map(order).astype(int)
    return cohort.reset_index(drop=True)


def score_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray | None,
    n_classes: int,
) -> dict[str, float]:
    labels = list(range(n_classes))
    metrics = {
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "quadratic_kappa": cohen_kappa_score(
            y_true, y_pred, labels=labels, weights="quadratic"
        ),
        "macro_precision": precision_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
    }
    for class_index in labels:
        metrics[f"recall_class{class_index}"] = recall_score(
            y_true, y_pred, labels=[class_index], average="macro", zero_division=0
        )
    if proba is not None and len(np.unique(y_true)) == n_classes:
        try:
            if n_classes == 2:
                metrics["roc_auc"] = roc_auc_score(y_true, proba[:, 1])
                metrics["average_precision"] = average_precision_score(
                    y_true, proba[:, 1]
                )
            else:
                metrics["roc_auc"] = roc_auc_score(
                    y_true, proba, multi_class="ovr", average="macro"
                )
                metrics["average_precision"] = np.mean(
                    [
                        average_precision_score((y_true == k).astype(int), proba[:, k])
                        for k in labels
                    ]
                )
        except ValueError:
            metrics["roc_auc"] = np.nan
            metrics["average_precision"] = np.nan
    else:
        metrics["roc_auc"] = np.nan
        metrics["average_precision"] = np.nan
    return metrics


def repeated_grouped_cv(
    estimator: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_classes: int,
    n_splits: int,
    n_repeats: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeated position-grouped CV returning per-fold metrics and OOF probabilities."""
    fold_rows = []
    oof_frames = []
    for repeat in range(n_repeats):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state + repeat
        )
        proba_accumulator = np.zeros((len(y), n_classes))
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
            model = clone(estimator)
            model.fit(X[train_idx], y[train_idx])
            proba = model.predict_proba(X[test_idx])
            # Align columns when a fold's training split misses a rare class.
            aligned = np.zeros((len(test_idx), n_classes))
            model_classes = model.named_steps["model"].classes_
            for column, class_value in enumerate(model_classes):
                aligned[:, int(class_value)] = proba[:, column]
            proba_accumulator[test_idx] = aligned
            y_pred = np.argmax(aligned, axis=1)
            metrics = score_predictions(y[test_idx], y_pred, aligned, n_classes)
            metrics.update({"repeat": repeat, "fold": fold, "n_test": len(test_idx)})
            fold_rows.append(metrics)
        oof = pd.DataFrame(
            proba_accumulator, columns=[f"proba_class{k}" for k in range(n_classes)]
        )
        oof["repeat"] = repeat
        oof["row_index"] = np.arange(len(y))
        oof["y_true"] = y
        oof["y_pred"] = np.argmax(proba_accumulator, axis=1)
        oof_frames.append(oof)
    return pd.DataFrame(fold_rows), pd.concat(oof_frames, ignore_index=True)


def tune_model(
    spec: ModelSpec,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    random_state: int,
    scoring: str,
    n_jobs: int,
) -> tuple[Pipeline, dict]:
    pipeline = build_pipeline(spec)
    if not spec.param_distributions:
        return pipeline, {}
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=spec.param_distributions,
        n_iter=spec.n_iter,
        scoring=scoring,
        cv=splitter,
        random_state=random_state,
        n_jobs=n_jobs,
        refit=True,
        error_score=np.nan,
    )
    search.fit(X, y, groups=groups)
    return search.best_estimator_, search.best_params_


def nested_cv_estimate(
    spec: ModelSpec,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_classes: int,
    n_splits: int,
    random_state: int,
    scoring: str,
    n_jobs: int,
) -> pd.DataFrame:
    """Unbiased estimate: hyperparameters are re-searched inside every outer fold."""
    outer = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    rows = []
    for fold, (train_idx, test_idx) in enumerate(outer.split(X, y, groups)):
        best_estimator, best_params = tune_model(
            spec,
            X[train_idx],
            y[train_idx],
            groups[train_idx],
            n_splits=max(3, n_splits - 2),
            random_state=random_state + fold,
            scoring=scoring,
            n_jobs=n_jobs,
        )
        proba = best_estimator.predict_proba(X[test_idx])
        aligned = np.zeros((len(test_idx), n_classes))
        for column, class_value in enumerate(best_estimator.named_steps["model"].classes_):
            aligned[:, int(class_value)] = proba[:, column]
        metrics = score_predictions(
            y[test_idx], np.argmax(aligned, axis=1), aligned, n_classes
        )
        metrics.update({"fold": fold, "best_params": json.dumps(best_params, default=str)})
        rows.append(metrics)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistical comparison
# ---------------------------------------------------------------------------

# Studentized range statistics for the Nemenyi test at alpha = 0.05
# (Demsar 2006, Table 5), divided by sqrt(2).
NEMENYI_Q05 = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
    9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391,
}


def friedman_nemenyi(score_matrix: pd.DataFrame) -> dict:
    """Rank models across folds and return the Nemenyi critical difference.

    `score_matrix` has one row per fold and one column per model. Ranks are
    computed within each fold, so the comparison is paired and does not assume
    comparable score scales between label schemes.
    """
    clean = score_matrix.dropna(axis=0, how="any")
    n_datasets, n_models = clean.shape
    if n_datasets < 3 or n_models < 2:
        return {"available": False, "reason": "not enough folds or models"}

    ranks = clean.rank(axis=1, ascending=False)
    mean_ranks = ranks.mean(axis=0)
    statistic, p_value = stats.friedmanchisquare(*[clean[c].to_numpy() for c in clean])
    q_alpha = NEMENYI_Q05.get(n_models, NEMENYI_Q05[15])
    critical_difference = q_alpha * np.sqrt(
        n_models * (n_models + 1) / (6.0 * n_datasets)
    )
    return {
        "available": True,
        "n_folds": int(n_datasets),
        "n_models": int(n_models),
        "friedman_statistic": float(statistic),
        "friedman_p_value": float(p_value),
        "critical_difference": float(critical_difference),
        "mean_ranks": mean_ranks.sort_values().to_dict(),
    }


def bootstrap_ci_by_position(
    oof: pd.DataFrame,
    positions: np.ndarray,
    n_classes: int,
    metric: str,
    n_boot: int,
    random_state: int,
) -> tuple[float, float, float]:
    """Cluster bootstrap over mtDNA positions, matching the repo's CI convention."""
    rng = np.random.default_rng(random_state)
    averaged = oof.groupby("row_index", sort=True).agg(
        {**{f"proba_class{k}": "mean" for k in range(n_classes)}, "y_true": "first"}
    )
    proba = averaged[[f"proba_class{k}" for k in range(n_classes)]].to_numpy()
    y_true = averaged["y_true"].to_numpy().astype(int)
    y_pred = np.argmax(proba, axis=1)

    unique_positions = np.unique(positions)
    position_to_rows = {
        pos: np.flatnonzero(positions == pos) for pos in unique_positions
    }
    point = score_predictions(y_true, y_pred, proba, n_classes)[metric]

    draws = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_positions, size=unique_positions.size, replace=True)
        rows = np.concatenate([position_to_rows[pos] for pos in sampled])
        if len(np.unique(y_true[rows])) < 2:
            continue
        draws.append(
            score_predictions(y_true[rows], y_pred[rows], proba[rows], n_classes)[metric]
        )
    if not draws:
        return float(point), float("nan"), float("nan")
    lower, upper = np.percentile(draws, [2.5, 97.5])
    return float(point), float(lower), float(upper)


# ---------------------------------------------------------------------------
# Feature subset selection
# ---------------------------------------------------------------------------


def forward_feature_selection(
    estimator: Pipeline,
    data: pd.DataFrame,
    candidate_features: list[str],
    forced_features: list[str],
    n_classes: int,
    n_splits: int,
    n_repeats: int,
    random_state: int,
    metric: str,
) -> pd.DataFrame:
    """Greedy forward selection with `forced_features` always present.

    Returns one row per subset size with the mean and standard error of the
    metric, which is what the one-standard-error parsimony rule consumes.
    """
    y = data["target"].to_numpy()
    groups = data["position"].to_numpy()
    selected = list(forced_features)
    remaining = [f for f in candidate_features if f not in forced_features]
    rows = []

    def evaluate(features: list[str]) -> tuple[float, float]:
        X = data[features].to_numpy(dtype=float)
        fold_metrics, _ = repeated_grouped_cv(
            estimator, X, y, groups, n_classes, n_splits, n_repeats, random_state
        )
        values = fold_metrics[metric].to_numpy()
        return float(np.mean(values)), float(stats.sem(values))

    mean_score, sem_score = evaluate(selected)
    rows.append(
        {
            "n_features": len(selected),
            "features": ",".join(selected),
            "added_feature": "",
            f"{metric}_mean": mean_score,
            f"{metric}_sem": sem_score,
        }
    )

    while remaining:
        scored = []
        for candidate in remaining:
            trial_mean, trial_sem = evaluate(selected + [candidate])
            scored.append((trial_mean, trial_sem, candidate))
        scored.sort(reverse=True)
        best_mean, best_sem, best_feature = scored[0]
        selected.append(best_feature)
        remaining.remove(best_feature)
        rows.append(
            {
                "n_features": len(selected),
                "features": ",".join(selected),
                "added_feature": best_feature,
                f"{metric}_mean": best_mean,
                f"{metric}_sem": best_sem,
            }
        )
    return pd.DataFrame(rows)


def one_standard_error_choice(curve: pd.DataFrame, metric: str) -> dict:
    """Smallest feature subset within one standard error of the best score."""
    mean_col, sem_col = f"{metric}_mean", f"{metric}_sem"
    best_row = curve.loc[curve[mean_col].idxmax()]
    threshold = best_row[mean_col] - best_row[sem_col]
    eligible = curve[curve[mean_col] >= threshold]
    chosen = eligible.loc[eligible["n_features"].idxmin()]
    return {
        "best_n_features": int(best_row["n_features"]),
        "best_score": float(best_row[mean_col]),
        "one_se_threshold": float(threshold),
        "chosen_n_features": int(chosen["n_features"]),
        "chosen_score": float(chosen[mean_col]),
        "chosen_features": chosen["features"].split(","),
    }


# ---------------------------------------------------------------------------
# Held-out permutation importance
# ---------------------------------------------------------------------------


def grouped_permutation_importance(
    estimator: Pipeline,
    data: pd.DataFrame,
    features: list[str],
    n_classes: int,
    n_splits: int,
    n_repeats: int,
    random_state: int,
    metric: str,
) -> pd.DataFrame:
    """Permutation importance measured on held-out folds, not on training data."""
    X = data[features].to_numpy(dtype=float)
    y = data["target"].to_numpy()
    groups = data["position"].to_numpy()
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rng = np.random.default_rng(random_state)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
        model = clone(estimator)
        model.fit(X[train_idx], y[train_idx])

        def fold_score(matrix: np.ndarray) -> float:
            proba = model.predict_proba(matrix)
            aligned = np.zeros((matrix.shape[0], n_classes))
            for column, class_value in enumerate(model.named_steps["model"].classes_):
                aligned[:, int(class_value)] = proba[:, column]
            return score_predictions(
                y[test_idx], np.argmax(aligned, axis=1), aligned, n_classes
            )[metric]

        baseline = fold_score(X[test_idx])
        for feature_index, feature in enumerate(features):
            drops = []
            for _ in range(n_repeats):
                permuted = X[test_idx].copy()
                permuted[:, feature_index] = rng.permutation(permuted[:, feature_index])
                drops.append(baseline - fold_score(permuted))
            rows.append(
                {
                    "fold": fold,
                    "feature": feature,
                    "baseline": baseline,
                    "importance_mean": float(np.mean(drops)),
                    "importance_std": float(np.std(drops)),
                }
            )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("feature")["importance_mean"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "importance_mean", "std": "importance_std"})
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    return summary


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_class_inventory(audit: pd.DataFrame, path: Path) -> None:
    schemes = list(SCHEME_CLASSES)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, scheme in zip(axes, schemes):
        subset = audit[
            audit["scheme"].eq(scheme)
            & ~audit["class"].eq("unlabeled_prediction_target")
        ]
        colors = [CLASS_COLORS[c] for c in subset["class"]]
        bars = ax.bar(subset["class"], subset["n_variants"], color=colors)
        ax.set_yscale("log")
        ax.set_title(f"{scheme}  (n={int(subset['n_variants'].sum())})")
        ax.tick_params(axis="x", rotation=30)
        ax.set_ylabel("Variants (log scale)" if scheme == schemes[0] else "")
        for bar, value in zip(bars, subset["n_variants"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{int(value):,}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.suptitle("Labelled cohort size by scheme (41,280 unlabelled kept as prediction target)")
    fig.tight_layout()
    _save(fig, path)


def plot_feature_by_class(data: pd.DataFrame, path: Path) -> None:
    classes = SCHEME_CLASSES["4class"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, feature in zip(axes, FORCED_FEATURES):
        groups, colors, ticks = [], [], []
        for class_name in classes:
            values = data.loc[data["label_4class"].eq(class_name), feature].dropna()
            groups.append(values.to_numpy())
            colors.append(CLASS_COLORS[class_name])
            ticks.append(f"{class_name}\nn={len(values)}")
        unlabelled = data.loc[data["label_4class"].isna(), feature].dropna()
        groups.append(unlabelled.to_numpy())
        colors.append(CLASS_COLORS["unlabeled"])
        ticks.append(f"unlabeled\nn={len(unlabelled):,}")

        parts = ax.violinplot(groups, showmedians=True, widths=0.85)
        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color)
            body.set_alpha(0.85)
            body.set_edgecolor("#333333")
        for key in ("cmedians", "cbars", "cmins", "cmaxes"):
            if key in parts:
                parts[key].set_color("#333333")
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels(ticks, fontsize=8)
        ax.set_title(feature)
        ax.grid(axis="y", alpha=0.3)
    axes[0].axhline(
        data.loc[data["neutral_selection_rule"].eq("lowest_decile_phyloP"), "phyloP100way"].max(),
        color="#C4453C",
        linestyle="--",
        linewidth=1.2,
        label="upper edge of the phyloP-selected neutral decile",
    )
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("The two forced features across classes; the dashed line marks the label-definition boundary")
    fig.tight_layout()
    _save(fig, path)


def plot_model_comparison(fold_metrics: pd.DataFrame, metric: str, path: Path) -> None:
    schemes = list(SCHEME_CLASSES)
    models = (
        fold_metrics.groupby("model")[metric].mean().sort_values(ascending=False).index.tolist()
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, scheme in zip(axes, schemes):
        subset = fold_metrics[fold_metrics["scheme"].eq(scheme)]
        box_data, ticks = [], []
        for model in models:
            values = subset.loc[subset["model"].eq(model), metric].dropna().to_numpy()
            box_data.append(values if len(values) else np.array([np.nan]))
            ticks.append(model.replace("_", " "))
        bp = ax.boxplot(box_data, vert=False, patch_artist=True, widths=0.6)
        for patch, model in zip(bp["boxes"], models):
            patch.set_facecolor("#BAB0AC" if model == "prior_baseline" else "#4C78A8")
            patch.set_alpha(0.85)
        for median in bp["medians"]:
            median.set_color("#222222")
        ax.set_yticks(np.arange(1, len(models) + 1))
        ax.set_yticklabels(ticks, fontsize=8)
        ax.set_title(scheme)
        ax.set_xlabel(metric.replace("_", " "))
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle(f"Repeated position-grouped cross validation: {metric.replace('_', ' ')} per fold")
    fig.tight_layout()
    _save(fig, path)


def plot_critical_difference(cd_result: dict, scheme: str, path: Path) -> None:
    if not cd_result.get("available"):
        return
    ranks = pd.Series(cd_result["mean_ranks"]).sort_values()
    cd = cd_result["critical_difference"]
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(ranks) + 2.4))
    y_positions = np.arange(len(ranks))[::-1]
    ax.scatter(ranks.to_numpy(), y_positions, s=70, color="#4C78A8", zorder=3)
    for y, (name, rank) in zip(y_positions, ranks.items()):
        ax.text(rank + 0.06, y, f"{name.replace('_', ' ')} ({rank:.2f})", va="center", fontsize=9)
    best_rank = ranks.iloc[0]
    ax.axvspan(best_rank, best_rank + cd, color="#4C78A8", alpha=0.12,
               label=f"within critical difference of the best (CD = {cd:.2f})")
    ax.axvline(best_rank, color="#C4453C", linestyle="--", linewidth=1.2)
    ax.set_yticks([])
    ax.set_xlabel("Mean rank across folds (lower is better)")
    ax.set_xlim(0.5, len(ranks) + 1.6)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(
        f"Nemenyi ranking, {scheme}\nFriedman p = {cd_result['friedman_p_value']:.3g}, "
        f"{cd_result['n_folds']} folds"
    )
    fig.tight_layout()
    _save(fig, path)


def plot_confusion_matrices(
    confusions: dict[str, tuple[np.ndarray, list[str]]], path: Path
) -> None:
    fig, axes = plt.subplots(1, len(confusions), figsize=(5.2 * len(confusions), 4.6))
    if len(confusions) == 1:
        axes = [axes]
    for ax, (scheme, (matrix, classes)) in zip(axes, confusions.items()):
        normalized = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1, None)
        image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=35, ha="right", fontsize=8)
        ax.set_yticklabels(classes, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(scheme)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(
                    j, i,
                    f"{normalized[i, j]:.2f}\n({int(matrix[i, j])})",
                    ha="center", va="center", fontsize=7,
                    color="white" if normalized[i, j] > 0.5 else "#222222",
                )
        fig.colorbar(image, ax=ax, fraction=0.046, shrink=0.85)
    fig.suptitle("Row-normalised out-of-fold confusion matrices for the selected model")
    fig.tight_layout()
    _save(fig, path)


def plot_feature_curve(curve: pd.DataFrame, choice: dict, metric: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    mean = curve[f"{metric}_mean"].to_numpy()
    sem = curve[f"{metric}_sem"].to_numpy()
    n_features = curve["n_features"].to_numpy()
    ax.errorbar(n_features, mean, yerr=sem, marker="o", color="#4C78A8", capsize=3)
    ax.axhline(
        choice["one_se_threshold"], color="#C4453C", linestyle="--", linewidth=1.2,
        label=f"one standard error below the best ({choice['one_se_threshold']:.3f})",
    )
    ax.axvline(
        choice["chosen_n_features"], color="#F58518", linestyle=":", linewidth=1.6,
        label=f"parsimonious choice: {choice['chosen_n_features']} features",
    )
    for x, y, added in zip(n_features, mean, curve["added_feature"]):
        if added:
            ax.annotate(f"+{added}", (x, y), textcoords="offset points",
                        xytext=(0, -16), ha="center", fontsize=7, rotation=25)
    ax.set_xlabel("Number of features (mlc_score and phyloP100way always included)")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title("Greedy forward selection with the one-standard-error parsimony rule")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, path)


def plot_permutation_importance(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    order = summary.sort_values("importance_mean")
    colors = [
        "#C4453C" if f in FORCED_FEATURES else "#4C78A8" for f in order["feature"]
    ]
    ax.barh(order["feature"], order["importance_mean"],
            xerr=order["importance_std"], color=colors, alpha=0.9, capsize=3)
    ax.axvline(0, color="#222222", linewidth=0.9)
    ax.set_xlabel("Held-out macro F1 drop when the feature is permuted")
    ax.set_title("Permutation influence of the selected model, measured on held-out folds")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(
        handles=[
            Patch(facecolor="#C4453C", label="forced feature"),
            Patch(facecolor="#4C78A8", label="selected feature"),
        ],
        fontsize=8, loc="lower right",
    )
    fig.tight_layout()
    _save(fig, path)


def plot_roc_pr(
    oof: pd.DataFrame, classes: list[str], n_classes: int, path: Path
) -> None:
    averaged = oof.groupby("row_index", sort=True).agg(
        {**{f"proba_class{k}": "mean" for k in range(n_classes)}, "y_true": "first"}
    )
    proba = averaged[[f"proba_class{k}" for k in range(n_classes)]].to_numpy()
    y_true = averaged["y_true"].to_numpy().astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for k, class_name in enumerate(classes):
        binary = (y_true == k).astype(int)
        if binary.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(binary, proba[:, k])
        auc = roc_auc_score(binary, proba[:, k])
        axes[0].plot(fpr, tpr, color=CLASS_COLORS[class_name],
                     label=f"{class_name} (AUC {auc:.3f})")
        precision, recall, _ = precision_recall_curve(binary, proba[:, k])
        ap = average_precision_score(binary, proba[:, k])
        axes[1].plot(recall, precision, color=CLASS_COLORS[class_name],
                     label=f"{class_name} (AP {ap:.3f})")
        axes[1].axhline(binary.mean(), color=CLASS_COLORS[class_name],
                        linestyle=":", linewidth=0.9, alpha=0.6)
    axes[0].plot([0, 1], [0, 1], color="#888888", linestyle="--", linewidth=0.9)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("One-vs-rest ROC")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("One-vs-rest precision/recall (dotted lines are class prevalence)")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Out-of-fold discrimination per class for the selected model")
    fig.tight_layout()
    _save(fig, path)


def plot_coverage_accuracy(
    oof: pd.DataFrame, n_classes: int, path: Path
) -> None:
    averaged = oof.groupby("row_index", sort=True).agg(
        {**{f"proba_class{k}": "mean" for k in range(n_classes)}, "y_true": "first"}
    )
    proba = averaged[[f"proba_class{k}" for k in range(n_classes)]].to_numpy()
    y_true = averaged["y_true"].to_numpy().astype(int)
    y_pred = np.argmax(proba, axis=1)
    confidence = proba.max(axis=1)

    order = np.argsort(-confidence)
    thresholds = np.linspace(0.0, 0.99, 100)
    coverage, macro_f1, accuracy = [], [], []
    for threshold in thresholds:
        keep = confidence >= threshold
        if keep.sum() < 20 or len(np.unique(y_true[keep])) < 2:
            continue
        coverage.append(keep.mean())
        macro_f1.append(
            f1_score(y_true[keep], y_pred[keep], labels=list(range(n_classes)),
                     average="macro", zero_division=0)
        )
        accuracy.append(balanced_accuracy_score(y_true[keep], y_pred[keep]))

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(coverage, macro_f1, marker="", color="#4C78A8", label="macro F1")
    ax.plot(coverage, accuracy, marker="", color="#F58518", label="balanced accuracy")
    ax.set_xlabel("Coverage: share of variants kept after the confidence cutoff")
    ax.set_ylabel("Out-of-fold score on the kept variants")
    ax.set_title("Abstention trade-off: how much accuracy is bought by declining low-confidence calls")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, path)
    del order


def plot_decision_landscape(
    estimator: Pipeline,
    cohort: pd.DataFrame,
    features: list[str],
    classes: list[str],
    path: Path,
) -> None:
    """Decision regions in the phyloP/mlc plane with the other features held at their median."""
    if not {"phyloP100way", "mlc_score"}.issubset(features):
        return
    x_index = features.index("phyloP100way")
    y_index = features.index("mlc_score")
    X = cohort[features].to_numpy(dtype=float)
    medians = np.median(X, axis=0)

    x_grid = np.linspace(np.percentile(X[:, x_index], 0.5), np.percentile(X[:, x_index], 99.5), 320)
    y_grid = np.linspace(np.percentile(X[:, y_index], 0.5), np.percentile(X[:, y_index], 99.5), 320)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid = np.tile(medians, (xx.size, 1))
    grid[:, x_index] = xx.ravel()
    grid[:, y_index] = yy.ravel()
    predicted = estimator.predict(grid).reshape(xx.shape)

    from matplotlib.colors import ListedColormap

    cmap = ListedColormap([CLASS_COLORS[c] for c in classes])
    fig, ax = plt.subplots(figsize=(9, 6.4))
    ax.pcolormesh(xx, yy, predicted, cmap=cmap, alpha=0.22, shading="auto")
    for k, class_name in enumerate(classes):
        mask = cohort["target"].to_numpy() == k
        size = 6 if class_name == "benign" else 34
        ax.scatter(X[mask, x_index], X[mask, y_index], s=size, alpha=0.75,
                   color=CLASS_COLORS[class_name], edgecolor="#222222",
                   linewidth=0.25 if class_name == "benign" else 0.5,
                   label=f"{class_name} (n={int(mask.sum())})")
    ax.set_xlabel("phyloP100way")
    ax.set_ylabel("mlc_score")
    ax.set_title(
        "Decision regions of the selected model\n"
        "(other features fixed at their median; shading is the predicted class)"
    )
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    _save(fig, path)


def plot_genome_map(predictions: pd.DataFrame, classes: list[str], path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    bins = np.arange(0, 16570, 200)
    predictions = predictions.copy()
    predictions["bin"] = pd.cut(predictions["position"], bins=bins, labels=bins[:-1])
    share = (
        predictions.groupby(["bin", "predicted_class"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    share = share.div(share.sum(axis=1).clip(lower=1), axis=0)
    bottom = np.zeros(len(share))
    for class_name in classes:
        if class_name not in share:
            continue
        values = share[class_name].to_numpy()
        axes[0].bar(share.index.astype(int), values, bottom=bottom, width=190,
                    color=CLASS_COLORS[class_name], label=class_name)
        bottom += values
    axes[0].set_ylabel("Predicted class share")
    axes[0].set_ylim(0, 1)
    axes[0].legend(ncol=len(classes), fontsize=8, loc="upper center",
                   bbox_to_anchor=(0.5, 1.22))
    axes[0].set_title("Predicted class composition along the mitochondrial genome (200 bp bins)")

    confidence = predictions.groupby("bin", observed=True)["confidence"].mean()
    axes[1].plot(confidence.index.astype(int), confidence.to_numpy(), color="#333333", linewidth=1.1)
    axes[1].set_ylabel("Mean confidence")
    axes[1].set_xlabel("mtDNA position (rCRS)")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, path)


def plot_pca_projection(
    cohort: pd.DataFrame, features: list[str], classes: list[str], path: Path
) -> None:
    X = cohort[features].to_numpy(dtype=float)
    scaled = RobustScaler().fit_transform(X)
    coords = PCA(n_components=2, random_state=0).fit_transform(scaled)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharex=True, sharey=True)
    for ax, column, title in [
        (axes[0], "target", "True class"),
        (axes[1], "oof_predicted", "Out-of-fold predicted class"),
    ]:
        if column not in cohort:
            continue
        for k, class_name in enumerate(classes):
            mask = cohort[column].to_numpy() == k
            if mask.sum() == 0:
                continue
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       s=6 if class_name == "benign" else 30,
                       alpha=0.6, color=CLASS_COLORS[class_name],
                       label=f"{class_name} ({int(mask.sum())})")
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("PC2")
    axes[1].legend(fontsize=8, markerscale=1.5)
    fig.suptitle("Feature space of the labelled cohort, true versus predicted")
    fig.tight_layout()
    _save(fig, path)


def plot_leakage_control(comparison: pd.DataFrame, metric: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    models = comparison["model"].tolist()
    x = np.arange(len(models))
    width = 0.38
    ax.bar(x - width / 2, comparison[f"{metric}_full"], width,
           color="#4C78A8", label="full labelled cohort")
    ax.bar(x + width / 2, comparison[f"{metric}_control"], width,
           color="#F58518", label="leakage control (haplogroup-selected neutrals only)")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ") for m in models], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(
        "Does the ranking survive removing the neutrals that were defined by low phyloP?"
    )
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, path)



def per_class_metrics_by_repeat(oof: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    """Per-class precision, recall and F1 with the CV repeat as the unit of spread."""
    n_classes = len(classes)
    rows = []
    for repeat, block in oof.groupby("repeat"):
        y_true = block["y_true"].to_numpy().astype(int)
        proba = block[[f"proba_class{k}" for k in range(n_classes)]].to_numpy()
        y_pred = np.argmax(proba, axis=1)
        for k, class_name in enumerate(classes):
            rows.append(
                {
                    "repeat": int(repeat),
                    "class": class_name,
                    "support": int((y_true == k).sum()),
                    "precision": precision_score(
                        y_true, y_pred, labels=[k], average="macro", zero_division=0
                    ),
                    "recall": recall_score(
                        y_true, y_pred, labels=[k], average="macro", zero_division=0
                    ),
                    "f1": f1_score(
                        y_true, y_pred, labels=[k], average="macro", zero_division=0
                    ),
                }
            )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("class")[["precision", "recall", "f1"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["class" if a == "class" else f"{a}_{b}" for a, b in summary.columns]
    supports = detail.groupby("class")["support"].first().reset_index()
    return summary.merge(supports, on="class")


def plot_per_class_performance(
    per_class: pd.DataFrame, classes: list[str], path: Path
) -> None:
    """How well the selected model recovers each class, with support shown."""
    ordered = per_class.set_index("class").loc[classes].reset_index()
    x = np.arange(len(classes))
    width = 0.26
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    for offset, metric, color in [
        (-width, "precision", "#4C78A8"),
        (0.0, "recall", "#F58518"),
        (width, "f1", "#54A24B"),
    ]:
        ax.bar(
            x + offset, ordered[f"{metric}_mean"], width,
            yerr=ordered[f"{metric}_std"], capsize=3, color=color,
            label=metric.capitalize(),
        )
        for xi, value in zip(x + offset, ordered[f"{metric}_mean"]):
            ax.text(xi, value + 0.02, f"{value:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{c}\nn = {int(s):,}" for c, s in zip(ordered["class"], ordered["support"])]
    )
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Out-of-fold score")
    ax.set_title(
        "Per-class performance of the selected model\n"
        "(error bars are the spread across cross-validation repeats)"
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, path)


def plot_prediction_inventory(
    predictions: pd.DataFrame, classes: list[str], path: Path
) -> None:
    """How many of all possible substitutions land in each class."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    order = classes + ["uncertain"]
    colors = [CLASS_COLORS.get(c, "#BAB0AC") for c in order]

    total = predictions["called_class"].value_counts().reindex(order, fill_value=0)
    bars = axes[0].bar(order, total.to_numpy(), color=colors)
    axes[0].set_title(f"All possible substitutions (n = {len(predictions):,})")
    axes[0].set_ylabel("Substitutions")
    for bar, value in zip(bars, total.to_numpy()):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2, value,
            f"{int(value):,}\n{value / len(predictions):.1%}",
            ha="center", va="bottom", fontsize=8,
        )

    split = (
        predictions.assign(
            cohort=np.where(predictions["is_training_variant"], "labelled", "unlabelled")
        )
        .groupby(["cohort", "called_class"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=order, fill_value=0)
    )
    bottom = np.zeros(len(split))
    for class_name, color in zip(order, colors):
        values = split[class_name].to_numpy()
        axes[1].bar(split.index, values, bottom=bottom, color=color, label=class_name)
        bottom += values
    axes[1].set_title("Split by whether the variant carries a label")
    axes[1].set_ylabel("Substitutions")
    axes[1].legend(fontsize=8)

    unlabelled = predictions[~predictions["is_training_variant"]]
    confident = (
        unlabelled.groupby(["predicted_class", "is_confident"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(index=classes, fill_value=0)
    )
    for flag in [True, False]:
        if flag not in confident:
            confident[flag] = 0
    axes[2].barh(confident.index, confident[True], color="#4C78A8",
                 label="above the confidence threshold")
    axes[2].barh(confident.index, confident[False], left=confident[True],
                 color="#BAB0AC", label="below it, reported as uncertain")
    axes[2].set_title("Unlabelled variants: how much survives the threshold")
    axes[2].set_xlabel("Substitutions")
    axes[2].legend(fontsize=8)

    for ax in axes[:2]:
        ax.grid(axis="y", alpha=0.3)
    axes[2].grid(axis="x", alpha=0.3)
    fig.suptitle("Where all possible mtDNA substitutions land under the selected model")
    fig.tight_layout()
    _save(fig, path)


def plot_probability_distributions(
    predictions: pd.DataFrame, classes: list[str], path: Path
) -> None:
    """Probability of the most severe class, split by what is already known."""
    severe = classes[-1]
    column = f"proba_{severe}"
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    groups = [(c, predictions[predictions["known_class"].eq(c)][column]) for c in classes]
    groups.append(("unlabeled", predictions[predictions["known_class"].isna()][column]))
    for name, values in groups:
        if not len(values):
            continue
        axes[0].hist(
            values, bins=40, histtype="step", density=True, linewidth=1.8,
            color=CLASS_COLORS.get(name, "#BAB0AC"),
            label=f"{name} (n = {len(values):,})",
        )
    axes[0].set_xlabel(f"P({severe})")
    axes[0].set_ylabel("Density (log scale)")
    axes[0].set_yscale("log")
    axes[0].set_title(f"P({severe}) by known label")
    axes[0].legend(fontsize=8)

    confidence = predictions[[f"proba_{c}" for c in classes]].to_numpy().max(axis=1)
    is_train = predictions["is_training_variant"].to_numpy()
    axes[1].hist(confidence[is_train], bins=40, histtype="step", density=True,
                 linewidth=1.8, color="#4C78A8", label="labelled")
    axes[1].hist(confidence[~is_train], bins=40, histtype="step", density=True,
                 linewidth=1.8, color="#BAB0AC", label="unlabelled")
    axes[1].set_xlabel("Confidence (probability of the predicted class)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Confidence distribution")
    axes[1].legend(fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Score distributions of the selected model")
    fig.tight_layout()
    _save(fig, path)


def plot_calibration(oof: pd.DataFrame, classes: list[str], path: Path) -> None:
    """Do the predicted probabilities mean what they claim?"""
    n_classes = len(classes)
    averaged = oof.groupby("row_index", sort=True).agg(
        {**{f"proba_class{k}": "mean" for k in range(n_classes)}, "y_true": "first"}
    )
    y_true = averaged["y_true"].to_numpy().astype(int)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.plot([0, 1], [0, 1], color="#888888", linestyle="--", linewidth=1,
            label="perfect calibration")
    edges = np.linspace(0, 1, 11)
    for k, class_name in enumerate(classes):
        proba = averaged[f"proba_class{k}"].to_numpy()
        observed, expected = [], []
        for low, high in zip(edges[:-1], edges[1:]):
            mask = (proba >= low) & (proba < high)
            if mask.sum() < 20:
                continue
            observed.append(float((y_true[mask] == k).mean()))
            expected.append(float(proba[mask].mean()))
        if expected:
            ax.plot(expected, observed, marker="o",
                    color=CLASS_COLORS[class_name], label=class_name)
    ax.set_xlabel("Mean predicted probability in the bin")
    ax.set_ylabel("Observed frequency in the bin")
    ax.set_title(
        "Calibration of out-of-fold probabilities\n"
        "(bins holding fewer than 20 variants are dropped)"
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, path)



# ---------------------------------------------------------------------------
# Isolation Forest baseline
# ---------------------------------------------------------------------------


def isolation_forest_baseline(
    data: pd.DataFrame,
    features: list[str],
    n_splits: int,
    n_repeats: int,
    random_state: int,
    target_fpr: float = 0.05,
    supervised: Pipeline | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare the one-class model of notebook 02 against supervised models.

    The comparison runs on the benign-versus-pathogenic subtask, because that is
    the only question the one-class model can be asked: it never sees pathogenic
    labels. Within each fold the forest is fitted on part of the benign training
    rows and its threshold is calibrated on the rest, mirroring the neutral
    train/validation split and the T95 rule of notebook 02.

    Single features are scored on the same folds with the same calibration, so a
    forest that merely reproduces one feature is visible as such. The orientation
    of each feature is taken from the training rows, never from the test fold.
    """
    cohort = scheme_cohort(data, "2class")
    X = cohort[features].to_numpy(dtype=float)
    y = cohort["target"].to_numpy()
    groups = cohort["position"].to_numpy()

    rows = []
    for repeat in range(n_repeats):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state + repeat
        )
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
            if len(np.unique(y[test_idx])) < 2:
                continue
            benign_train = train_idx[y[train_idx] == 0]
            rng = np.random.default_rng(random_state + 97 * repeat + fold)
            shuffled = rng.permutation(benign_train)
            n_calibration = max(40, int(round(0.25 * len(shuffled))))
            calibration_idx, fit_idx = shuffled[:n_calibration], shuffled[n_calibration:]
            if len(fit_idx) < 50:
                continue

            def record(name, kind, test_scores, calibration_scores):
                threshold = float(np.quantile(calibration_scores, 1.0 - target_fpr))
                positive = test_scores[y[test_idx] == 1]
                negative = test_scores[y[test_idx] == 0]
                rows.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "method": name,
                        "kind": kind,
                        "roc_auc": roc_auc_score(y[test_idx], test_scores),
                        "average_precision": average_precision_score(
                            y[test_idx], test_scores
                        ),
                        "recall_at_target_fpr": float((positive > threshold).mean()),
                        "realised_fpr": float((negative > threshold).mean()),
                        "n_pathogenic": int(len(positive)),
                    }
                )

            forest = Pipeline(
                [
                    ("scaler", RobustScaler()),
                    (
                        "model",
                        IsolationForest(
                            n_estimators=1000,
                            contamination="auto",
                            random_state=random_state,
                            n_jobs=1,
                        ),
                    ),
                ]
            )
            forest.fit(X[fit_idx])
            record(
                "isolation_forest", "one_class",
                -forest.decision_function(X[test_idx]),
                -forest.decision_function(X[calibration_idx]),
            )

            for index, feature in enumerate(features):
                # Orientation from training rows only.
                sign = 1.0
                pathogenic_train = X[train_idx][y[train_idx] == 1, index]
                benign_train_values = X[train_idx][y[train_idx] == 0, index]
                if len(pathogenic_train) and np.mean(pathogenic_train) < np.mean(
                    benign_train_values
                ):
                    sign = -1.0
                record(
                    f"single: {feature}", "single_feature",
                    sign * X[test_idx][:, index],
                    sign * X[calibration_idx][:, index],
                )

            if supervised is not None:
                model = clone(supervised)
                model.fit(X[train_idx], y[train_idx])
                proba = model.predict_proba(X[test_idx])
                classes = list(model.named_steps["model"].classes_)
                column = classes.index(1)
                calibration_proba = model.predict_proba(X[calibration_idx])[:, column]
                record(
                    "supervised model", "supervised",
                    proba[:, column], calibration_proba,
                )

    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, detail
    summary = (
        detail.groupby(["method", "kind"])[
            ["roc_auc", "average_precision", "recall_at_target_fpr", "realised_fpr"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        a if b == "" else f"{a}_{b}" for a, b in summary.columns
    ]
    summary = summary.sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)
    return detail, summary


def plot_isolation_forest_baseline(
    summary: pd.DataFrame, target_fpr: float, path: Path
) -> None:
    if summary.empty:
        return
    order = summary.sort_values("recall_at_target_fpr_mean")
    colors = {
        "one_class": "#F58518",
        "supervised": "#4C78A8",
        "single_feature": "#BAB0AC",
    }
    labels = {
        "one_class": "Isolation Forest (one-class)",
        "supervised": "supervised model",
        "single_feature": "single feature",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), sharey=True)
    names = [
        n.replace("single: ", "") for n in order["method"]
    ]
    bar_colors = [colors[k] for k in order["kind"]]

    axes[0].barh(names, order["recall_at_target_fpr_mean"],
                 xerr=order["recall_at_target_fpr_std"], capsize=3,
                 color=bar_colors, alpha=.9)
    axes[0].set_xlabel(f"Recall on pathogenic at a matched {target_fpr:.0%} false-positive rate")
    axes[0].set_title("Detection at one operating point")
    axes[0].grid(axis="x", alpha=.3)

    axes[1].barh(names, order["roc_auc_mean"], xerr=order["roc_auc_std"],
                 capsize=3, color=bar_colors, alpha=.9)
    axes[1].axvline(0.5, color="#222222", linewidth=.9, linestyle="--")
    axes[1].set_xlim(0.4, 1.0)
    axes[1].set_xlabel("ROC AUC")
    axes[1].set_title("Ranking quality over all thresholds")
    axes[1].grid(axis="x", alpha=.3)

    handles = [Patch(facecolor=c, label=labels[k]) for k, c in colors.items()]
    axes[1].legend(handles=handles, fontsize=8, loc="lower right")
    fig.suptitle(
        "Does the one-class forest add anything over a single feature?\n"
        "Same position-grouped folds, same calibration rule, benign versus pathogenic"
    )
    fig.tight_layout()
    _save(fig, path)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def evaluate_all_models(
    data: pd.DataFrame,
    zoo: dict[str, ModelSpec],
    features: list[str],
    n_splits: int,
    n_repeats: int,
    tune_splits: int,
    random_state: int,
    scoring: str,
    n_jobs: int,
    leakage_control: bool = False,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict, dict]:
    """Tune then repeatedly cross-validate every model under every label scheme."""
    fold_rows = []
    oof_store: dict[tuple[str, str], pd.DataFrame] = {}
    tuned_store: dict[tuple[str, str], dict] = {}

    for scheme, classes in SCHEME_CLASSES.items():
        cohort = scheme_cohort(data, scheme, leakage_control=leakage_control)
        X = cohort[features].to_numpy(dtype=float)
        y = cohort["target"].to_numpy()
        groups = cohort["position"].to_numpy()
        n_classes = len(classes)

        for model_name, spec in zoo.items():
            if verbose:
                print(f"  [{scheme}] {model_name} ...", flush=True)
            estimator, best_params = tune_model(
                spec, X, y, groups, tune_splits, random_state, scoring, n_jobs
            )
            fold_metrics, oof = repeated_grouped_cv(
                estimator, X, y, groups, n_classes, n_splits, n_repeats, random_state
            )
            fold_metrics["scheme"] = scheme
            fold_metrics["model"] = model_name
            fold_rows.append(fold_metrics)
            oof_store[(scheme, model_name)] = oof
            tuned_store[(scheme, model_name)] = {
                "estimator": estimator,
                "best_params": best_params,
                "n_classes": n_classes,
                "cohort": cohort,
            }
    return pd.concat(fold_rows, ignore_index=True), oof_store, tuned_store


def summarise_folds(fold_metrics: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    aggregation = {metric: ["mean", "std", "count"] for metric in metrics}
    summary = fold_metrics.groupby(["scheme", "model"]).agg(aggregation)
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    for metric in metrics:
        summary[f"{metric}_sem"] = summary[f"{metric}_std"] / np.sqrt(
            summary[f"{metric}_count"].clip(lower=1)
        )
    return summary.sort_values(["scheme", "macro_f1_mean"], ascending=[True, False])


def compare_schemes(
    data: pd.DataFrame,
    oof_store: dict,
    tuned_store: dict,
    model_name: str,
    fold_metrics: pd.DataFrame,
    viability_threshold: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Put the three schemes on one comparable footing.

    Macro F1 is not comparable between a 2-class and a 4-class problem, so each
    scheme is additionally collapsed onto the same benign-versus-pathogenic
    subtask on the variants that every scheme labels. The collapsed score uses
    the probability of the most severe class as a ranking score, which is
    scale-free and therefore comparable across schemes.
    """
    rows = []
    per_class_rows = []
    for scheme, classes in SCHEME_CLASSES.items():
        key = (scheme, model_name)
        if key not in oof_store:
            continue
        n_classes = len(classes)
        cohort = tuned_store[key]["cohort"]
        oof = oof_store[key]
        averaged = oof.groupby("row_index", sort=True).agg(
            {**{f"proba_class{k}": "mean" for k in range(n_classes)}, "y_true": "first"}
        )
        proba = averaged[[f"proba_class{k}" for k in range(n_classes)]].to_numpy()
        y_true = averaged["y_true"].to_numpy().astype(int)
        y_pred = np.argmax(proba, axis=1)

        native = score_predictions(y_true, y_pred, proba, n_classes)

        # Collapse to benign versus pathogenic on the shared subset.
        true_names = np.array(classes)[y_true]
        shared = np.isin(true_names, ["benign", "pathogenic"])
        binary_true = (true_names[shared] == "pathogenic").astype(int)
        pathogenic_score = proba[shared, n_classes - 1]
        binary_pred = (np.array(classes)[y_pred[shared]] == "pathogenic").astype(int)

        # Class viability is measured as one-vs-rest MCC, not as recall against
        # the stratified baseline. Under stratified guessing that baseline's
        # recall equals the class prevalence, so the majority class sets a bar
        # near 0.97 that any deliberately balanced model fails by design, while
        # a rare class sets a bar near zero that anything clears. One-vs-rest
        # MCC is zero under chance for every class regardless of prevalence.
        baseline_rows = fold_metrics[
            fold_metrics["scheme"].eq(scheme) & fold_metrics["model"].eq("prior_baseline")
        ]
        per_class = {}
        for k in range(n_classes):
            per_class[k] = {
                "recall": native[f"recall_class{k}"],
                "ovr_mcc": matthews_corrcoef((y_true == k).astype(int), (y_pred == k).astype(int)),
                "baseline_recall": (
                    float(baseline_rows[f"recall_class{k}"].mean())
                    if len(baseline_rows) else float("nan")
                ),
                "prevalence": float((y_true == k).mean()),
            }
        for k, values in per_class.items():
            per_class_rows.append({"scheme": scheme, "class": classes[k], **values})
        weakest_class = min(per_class, key=lambda k: per_class[k]["ovr_mcc"])
        baseline_recall = per_class[weakest_class]["baseline_recall"]
        min_ovr_mcc = min(per_class[k]["ovr_mcc"] for k in per_class)
        viability = min_ovr_mcc >= viability_threshold

        rows.append(
            {
                "scheme": scheme,
                "n_classes": n_classes,
                "n_labelled": len(cohort),
                "native_macro_f1": native["macro_f1"],
                "native_balanced_accuracy": native["balanced_accuracy"],
                "native_quadratic_kappa": native["quadratic_kappa"],
                "weakest_class": classes[weakest_class],
                "weakest_class_recall": native[f"recall_class{weakest_class}"],
                "weakest_class_baseline_recall": baseline_recall,
                "weakest_class_ovr_mcc": min_ovr_mcc,
                "viability_threshold": viability_threshold,
                "all_classes_above_chance": viability,
                "shared_n": int(shared.sum()),
                "shared_roc_auc": roc_auc_score(binary_true, pathogenic_score),
                "shared_average_precision": average_precision_score(
                    binary_true, pathogenic_score
                ),
                "shared_mcc": matthews_corrcoef(binary_true, binary_pred),
                "shared_pathogenic_recall": recall_score(
                    binary_true, binary_pred, zero_division=0
                ),
                "shared_pathogenic_precision": precision_score(
                    binary_true, binary_pred, zero_division=0
                ),
            }
        )
    comparison = pd.DataFrame(rows)
    best_shared = comparison["shared_mcc"].max()
    comparison["shared_mcc_relative_drop"] = (
        best_shared - comparison["shared_mcc"]
    ) / best_shared
    return comparison, pd.DataFrame(per_class_rows)


def select_label_scheme(comparison: pd.DataFrame, tolerance: float) -> dict:
    """Pick the most detailed scheme that earns its extra classes.

    Two conditions must both hold. Every class must be recovered above its own
    chance level, otherwise the class exists in the label table but not in the
    feature space. And the extra classes must not damage the benign-versus-
    pathogenic decision that the rest of the project depends on, measured as a
    relative drop in shared MCC against the best scheme.
    """
    eligible = comparison[
        comparison["all_classes_above_chance"]
        & (comparison["shared_mcc_relative_drop"] <= tolerance)
    ]
    if eligible.empty:
        chosen = comparison.sort_values("shared_mcc", ascending=False).iloc[0]
        reason = "no scheme met both conditions; fell back to the best shared MCC"
    else:
        chosen = eligible.sort_values("n_classes", ascending=False).iloc[0]
        reason = (
            f"most detailed scheme with every class above chance and at most "
            f"{tolerance:.0%} relative loss of shared MCC"
        )
    return {
        "scheme": str(chosen["scheme"]),
        "reason": reason,
        "tolerance": tolerance,
        "shared_mcc": float(chosen["shared_mcc"]),
        "shared_mcc_relative_drop": float(chosen["shared_mcc_relative_drop"]),
        "rejected": comparison.loc[
            ~comparison["scheme"].eq(chosen["scheme"]), "scheme"
        ].tolist(),
    }


def plot_scheme_tradeoff(comparison: pd.DataFrame, tolerance: float, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    x = np.arange(len(comparison))
    labels = comparison["scheme"].tolist()

    axes[0].bar(x - 0.2, comparison["shared_mcc"], 0.4, color="#4C78A8",
                label="shared MCC (benign vs pathogenic)")
    axes[0].bar(x + 0.2, comparison["shared_average_precision"], 0.4, color="#F58518",
                label="shared average precision")
    best = comparison["shared_mcc"].max()
    axes[0].axhline(best * (1 - tolerance), color="#C4453C", linestyle="--", linewidth=1.2,
                    label=f"tolerance: {tolerance:.0%} below the best MCC")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Score on the shared subtask")
    axes[0].set_title("Cost of extra classes on the decision that matters downstream")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    threshold = float(comparison["viability_threshold"].iloc[0])
    colors = [
        "#4C78A8" if value >= threshold else "#C4453C"
        for value in comparison["weakest_class_ovr_mcc"]
    ]
    axes[1].bar(x, comparison["weakest_class_ovr_mcc"], 0.5, color=colors)
    axes[1].axhline(threshold, color="#C4453C", linestyle="--", linewidth=1.2,
                    label=f"viability threshold ({threshold:.2f})")
    axes[1].axhline(0, color="#222222", linewidth=0.9)
    for i, row in comparison.reset_index(drop=True).iterrows():
        axes[1].text(i, row["weakest_class_ovr_mcc"] + 0.015,
                     f"{row['weakest_class']}\nrecall {row['weakest_class_recall']:.2f}",
                     ha="center", fontsize=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("One-vs-rest MCC of the weakest class")
    axes[1].set_title("Is the hardest class recovered above chance?\n(MCC is zero under chance at any prevalence)")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("Choosing the number of classes: both panels must pass")
    fig.tight_layout()
    _save(fig, path)


def apply_to_all_substitutions(
    estimator: Pipeline,
    data: pd.DataFrame,
    features: list[str],
    classes: list[str],
    confidence_threshold: float,
) -> pd.DataFrame:
    X_all = data[features].to_numpy(dtype=float)
    proba = estimator.predict_proba(X_all)
    aligned = np.zeros((len(data), len(classes)))
    for column, class_value in enumerate(estimator.named_steps["model"].classes_):
        aligned[:, int(class_value)] = proba[:, column]

    predicted_index = np.argmax(aligned, axis=1)
    confidence = aligned.max(axis=1)
    scheme = f"{len(classes)}class"

    out = data[
        [
            "variant_id",
            "position",
            "reference",
            "alternate",
            "mlc_score",
            "phyloP100way",
            "label_source",
            f"label_{scheme}",
        ]
    ].copy()
    out = out.rename(columns={f"label_{scheme}": "known_class"})
    out["predicted_class"] = np.array(classes)[predicted_index]
    out["confidence"] = confidence
    out["is_confident"] = confidence >= confidence_threshold
    out["called_class"] = np.where(
        out["is_confident"], out["predicted_class"], "uncertain"
    )
    for k, class_name in enumerate(classes):
        out[f"proba_{class_name}"] = aligned[:, k]
    # Expected severity on the ordinal scale, useful for ranking within a class.
    out["expected_severity"] = aligned @ np.arange(len(classes))
    out["is_training_variant"] = out["known_class"].notna()

    for column in ["neutral_domain_class_T95", "isolation_forest_above_T95"]:
        if column in data.columns:
            out[column] = data[column].to_numpy()
    return out


def recommend_confidence_threshold(
    oof: pd.DataFrame, n_classes: int, min_coverage: float
) -> dict:
    averaged = oof.groupby("row_index", sort=True).agg(
        {**{f"proba_class{k}": "mean" for k in range(n_classes)}, "y_true": "first"}
    )
    proba = averaged[[f"proba_class{k}" for k in range(n_classes)]].to_numpy()
    y_true = averaged["y_true"].to_numpy().astype(int)
    y_pred = np.argmax(proba, axis=1)
    confidence = proba.max(axis=1)

    best = {
        "threshold": 0.0,
        "coverage": 1.0,
        "macro_f1": f1_score(
            y_true, y_pred, labels=list(range(n_classes)), average="macro", zero_division=0
        ),
    }
    for threshold in np.linspace(0.0, 0.95, 96):
        keep = confidence >= threshold
        coverage = keep.mean()
        if coverage < min_coverage or len(np.unique(y_true[keep])) < 2:
            continue
        score = f1_score(
            y_true[keep], y_pred[keep], labels=list(range(n_classes)),
            average="macro", zero_division=0,
        )
        if score > best["macro_f1"]:
            best = {"threshold": float(threshold), "coverage": float(coverage),
                    "macro_f1": float(score)}
    return best


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=root / "data/processed/model_features.tsv",
    )
    parser.add_argument(
        "--dataset3", type=Path,
        default=root / "data/raw/lake_2024_supplement_data/supplementary_dataset_3.tsv",
    )
    parser.add_argument(
        "--dataset8", type=Path,
        default=root / "data/raw/lake_2024_supplement_data/supplementary_dataset_8.tsv",
    )
    parser.add_argument(
        "--dataset9", type=Path,
        default=root / "data/raw/lake_2024_supplement_data/supplementary_dataset_9.tsv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "results/classification"
    )
    parser.add_argument(
        "--figure-dir", type=Path,
        default=root / "results/figures/classification",
    )
    parser.add_argument("--budget", choices=["fast", "medium", "thorough"],
                        default="thorough")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--tune-folds", type=int, default=5)
    parser.add_argument("--selection-repeats", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--finalists", type=int, default=3)
    parser.add_argument("--min-coverage", type=float, default=0.70)
    parser.add_argument(
        "--target-fpr", type=float, default=0.05,
        help="False-positive rate on benign variants at which the baseline "
             "methods are compared; matches the T95 rule of notebook 02.",
    )
    parser.add_argument(
        "--viability-threshold", type=float, default=0.10,
        help=(
            "Smallest one-vs-rest MCC a class must reach for its scheme to count "
            "as separable. Zero is chance level at any prevalence."
        ),
    )
    parser.add_argument(
        "--scheme-tolerance", type=float, default=0.05,
        help=(
            "Largest acceptable relative drop in shared benign-versus-pathogenic "
            "MCC that extra classes may cost. This is an explicit judgement call, "
            "not an estimate: raise it to favour granularity."
        ),
    )
    parser.add_argument("--primary-metric", default="macro_f1")
    parser.add_argument("--scoring", default="f1_macro")
    parser.add_argument("--random-state", type=int, default=20260901)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--skip-nested", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    metric = args.primary_metric

    print("Loading the feature table ...", flush=True)
    data = pd.read_csv(args.input, sep="\t", low_memory=False)
    if "label_3class" not in data.columns:
        data = attach_label_sources(data, args.dataset3, args.dataset8, args.dataset9)
        data = build_label_schemes(data)

    missing = data[CORE_FEATURES].isna().sum()
    if missing.any():
        raise ValueError(f"Missing feature values:\n{missing[missing > 0]}")

    audit = label_audit_table(data)
    audit.to_csv(args.output_dir / "label_scheme_audit.tsv", sep="\t", index=False)
    plot_class_inventory(audit, args.figure_dir / "label_scheme_inventory.png")
    plot_feature_by_class(data, args.figure_dir / "forced_features_by_class.png")

    zoo = make_model_zoo(args.random_state, args.budget)
    reported_metrics = [
        "macro_f1", "balanced_accuracy", "mcc", "quadratic_kappa",
        "macro_precision", "macro_recall", "roc_auc", "average_precision",
    ]

    print("Stage 1/7: tuning and cross-validating every model under every scheme ...", flush=True)
    fold_metrics, oof_store, tuned_store = evaluate_all_models(
        data, zoo, CORE_FEATURES, args.folds, args.repeats, args.tune_folds,
        args.random_state, args.scoring, args.n_jobs,
    )
    fold_metrics.to_csv(args.output_dir / "model_fold_metrics.tsv", sep="\t", index=False)
    summary = summarise_folds(fold_metrics, reported_metrics)
    summary.to_csv(args.output_dir / "model_scheme_summary.tsv", sep="\t", index=False)
    plot_model_comparison(fold_metrics, metric, args.figure_dir / "model_comparison.png")

    print("Stage 2/7: Friedman and Nemenyi ranking per scheme ...", flush=True)
    ranking_rows = []
    cd_results = {}
    for scheme in SCHEME_CLASSES:
        subset = fold_metrics[fold_metrics["scheme"].eq(scheme)].copy()
        subset["fold_id"] = subset["repeat"].astype(str) + "_" + subset["fold"].astype(str)
        matrix = subset.pivot_table(index="fold_id", columns="model", values=metric)
        result = friedman_nemenyi(matrix)
        cd_results[scheme] = result
        plot_critical_difference(
            result, scheme, args.figure_dir / f"critical_difference_{scheme}.png"
        )
        if result.get("available"):
            best_rank = min(result["mean_ranks"].values())
            for model_name, rank in result["mean_ranks"].items():
                ranking_rows.append(
                    {
                        "scheme": scheme,
                        "model": model_name,
                        "mean_rank": rank,
                        "critical_difference": result["critical_difference"],
                        "tied_with_best": bool(
                            rank - best_rank <= result["critical_difference"]
                        ),
                        "friedman_p_value": result["friedman_p_value"],
                    }
                )
    ranking = pd.DataFrame(ranking_rows)
    ranking.to_csv(args.output_dir / "model_ranking_nemenyi.tsv", sep="\t", index=False)

    # The selected model is the best-ranked one that is not the prior baseline,
    # chosen on the scheme where it ranks best overall.
    real_models = ranking[~ranking["model"].eq("prior_baseline")]
    overall_rank = real_models.groupby("model")["mean_rank"].mean().sort_values()
    selected_model = overall_rank.index[0]
    print(f"  selected model: {selected_model}", flush=True)

    print("Stage 3/7: comparing label granularity on a shared subtask ...", flush=True)
    scheme_comparison, scheme_per_class = compare_schemes(
        data, oof_store, tuned_store, selected_model, fold_metrics,
        args.viability_threshold,
    )
    scheme_comparison.to_csv(
        args.output_dir / "label_scheme_comparison.tsv", sep="\t", index=False
    )
    scheme_per_class.to_csv(
        args.output_dir / "label_scheme_per_class.tsv", sep="\t", index=False
    )
    plot_scheme_tradeoff(
        scheme_comparison, args.scheme_tolerance,
        args.figure_dir / "label_scheme_tradeoff.png",
    )

    scheme_choice = select_label_scheme(scheme_comparison, args.scheme_tolerance)
    selected_scheme = scheme_choice["scheme"]
    print(f"  selected label scheme: {selected_scheme} ({scheme_choice['reason']})", flush=True)

    # Confusion matrices for every scheme under the selected model, so the cost
    # of the rejected granularities is visible rather than asserted.
    scheme_confusions = {}
    for scheme, classes in SCHEME_CLASSES.items():
        key = (scheme, selected_model)
        if key not in oof_store:
            continue
        k_classes = len(classes)
        scheme_oof = oof_store[key].groupby("row_index", sort=True).agg(
            {**{f"proba_class{k}": "mean" for k in range(k_classes)}, "y_true": "first"}
        )
        y_true_scheme = scheme_oof["y_true"].to_numpy().astype(int)
        y_pred_scheme = np.argmax(
            scheme_oof[[f"proba_class{k}" for k in range(k_classes)]].to_numpy(), axis=1
        )
        scheme_confusions[scheme] = (
            confusion_matrix(y_true_scheme, y_pred_scheme, labels=list(range(k_classes))),
            classes,
        )
    plot_confusion_matrices(
        scheme_confusions, args.figure_dir / "confusion_matrix_by_scheme.png"
    )

    selected_classes = SCHEME_CLASSES[selected_scheme]
    n_classes = len(selected_classes)
    cohort = tuned_store[(selected_scheme, selected_model)]["cohort"]
    selected_estimator = tuned_store[(selected_scheme, selected_model)]["estimator"]

    print("Stage 4/7: forward feature selection under the one-standard-error rule ...", flush=True)
    curve = forward_feature_selection(
        selected_estimator, cohort, CORE_FEATURES, FORCED_FEATURES, n_classes,
        args.folds, args.selection_repeats, args.random_state, metric,
    )
    curve.to_csv(args.output_dir / "feature_selection_curve.tsv", sep="\t", index=False)
    choice = one_standard_error_choice(curve, metric)
    plot_feature_curve(curve, choice, metric, args.figure_dir / "feature_selection_curve.png")
    selected_features = choice["chosen_features"]
    print(f"  selected features ({len(selected_features)}): {selected_features}", flush=True)

    print("Stage 5/7: refit, nested check, importance and leakage control ...", flush=True)
    X_sel = cohort[selected_features].to_numpy(dtype=float)
    y_sel = cohort["target"].to_numpy()
    groups_sel = cohort["position"].to_numpy()
    final_estimator, final_params = tune_model(
        zoo[selected_model], X_sel, y_sel, groups_sel, args.tune_folds,
        args.random_state, args.scoring, args.n_jobs,
    )
    final_folds, final_oof = repeated_grouped_cv(
        final_estimator, X_sel, y_sel, groups_sel, n_classes,
        args.folds, args.repeats, args.random_state,
    )
    final_folds.to_csv(args.output_dir / "selected_model_fold_metrics.tsv", sep="\t", index=False)
    final_oof.to_csv(
        args.output_dir / "selected_model_oof_predictions.tsv", sep="\t", index=False
    )
    per_class = per_class_metrics_by_repeat(final_oof, selected_classes)
    per_class.to_csv(
        args.output_dir / "selected_model_per_class_metrics.tsv", sep="\t", index=False
    )
    plot_per_class_performance(
        per_class, selected_classes,
        args.figure_dir / "selected_model_per_class_performance.png",
    )
    plot_calibration(
        final_oof, selected_classes, args.figure_dir / "selected_model_calibration.png"
    )

    averaged_oof = final_oof.groupby("row_index", sort=True).agg(
        {**{f"proba_class{k}": "mean" for k in range(n_classes)}, "y_true": "first"}
    )
    oof_proba = averaged_oof[[f"proba_class{k}" for k in range(n_classes)]].to_numpy()
    cohort = cohort.copy()
    cohort["oof_predicted"] = np.argmax(oof_proba, axis=1)
    confusion = confusion_matrix(
        cohort["target"], cohort["oof_predicted"], labels=list(range(n_classes))
    )
    pd.DataFrame(
        confusion, index=[f"true_{c}" for c in selected_classes],
        columns=[f"pred_{c}" for c in selected_classes],
    ).to_csv(args.output_dir / "selected_model_confusion_matrix.tsv", sep="\t")

    ci_rows = []
    for ci_metric in ["macro_f1", "balanced_accuracy", "mcc", "quadratic_kappa"]:
        point, lower, upper = bootstrap_ci_by_position(
            final_oof, cohort["position"].to_numpy(), n_classes, ci_metric,
            args.bootstrap, args.random_state,
        )
        ci_rows.append(
            {"metric": ci_metric, "point_estimate": point,
             "ci_lower": lower, "ci_upper": upper, "n_bootstrap": args.bootstrap}
        )
    pd.DataFrame(ci_rows).to_csv(
        args.output_dir / "selected_model_bootstrap_ci.tsv", sep="\t", index=False
    )

    nested = pd.DataFrame()
    if not args.skip_nested:
        finalists = overall_rank.index[: args.finalists].tolist()
        nested_frames = []
        for model_name in finalists:
            estimate = nested_cv_estimate(
                zoo[model_name], X_sel, y_sel, groups_sel, n_classes, args.folds,
                args.random_state, args.scoring, args.n_jobs,
            )
            estimate["model"] = model_name
            nested_frames.append(estimate)
        nested = pd.concat(nested_frames, ignore_index=True)
        nested.to_csv(args.output_dir / "nested_cv_finalists.tsv", sep="\t", index=False)

    importance = grouped_permutation_importance(
        final_estimator, cohort, selected_features, n_classes, args.folds,
        max(3, args.selection_repeats), args.random_state, metric,
    )
    importance.to_csv(
        args.output_dir / "selected_model_permutation_importance.tsv", sep="\t", index=False
    )
    plot_permutation_importance(
        importance, args.figure_dir / "selected_model_permutation_importance.png"
    )

    control_fold_metrics, _, _ = evaluate_all_models(
        data, zoo, selected_features, args.folds, max(2, args.repeats // 2),
        args.tune_folds, args.random_state, args.scoring, args.n_jobs,
        leakage_control=True, verbose=False,
    )
    control_summary = summarise_folds(control_fold_metrics, [metric])
    full_summary = summarise_folds(
        fold_metrics[fold_metrics["scheme"].eq(selected_scheme)], [metric]
    )
    comparison = full_summary[["model", f"{metric}_mean"]].rename(
        columns={f"{metric}_mean": f"{metric}_full"}
    ).merge(
        control_summary[control_summary["scheme"].eq(selected_scheme)][
            ["model", f"{metric}_mean"]
        ].rename(columns={f"{metric}_mean": f"{metric}_control"}),
        on="model", how="inner",
    )
    comparison["delta"] = comparison[f"{metric}_control"] - comparison[f"{metric}_full"]
    comparison.to_csv(
        args.output_dir / "leakage_control_comparison.tsv", sep="\t", index=False
    )
    plot_leakage_control(comparison, metric, args.figure_dir / "leakage_control.png")

    print("Stage 6/7: Isolation Forest baseline on the same protocol ...", flush=True)
    baseline_detail, baseline_summary = isolation_forest_baseline(
        data, selected_features, args.folds, max(2, args.repeats // 2),
        args.random_state, args.target_fpr, supervised=final_estimator,
    )
    if not baseline_summary.empty:
        baseline_detail.to_csv(
            args.output_dir / "isolation_forest_baseline_folds.tsv", sep="\t", index=False
        )
        baseline_summary.to_csv(
            args.output_dir / "isolation_forest_baseline_summary.tsv", sep="\t", index=False
        )
        plot_isolation_forest_baseline(
            baseline_summary, args.target_fpr,
            args.figure_dir / "isolation_forest_baseline.png",
        )

    print("Stage 7/7: applying the selected model to all possible substitutions ...", flush=True)
    threshold = recommend_confidence_threshold(final_oof, n_classes, args.min_coverage)
    final_estimator.fit(X_sel, y_sel)
    predictions = apply_to_all_substitutions(
        final_estimator, data, selected_features, selected_classes, threshold["threshold"]
    )
    predictions.to_csv(
        args.output_dir / "all_substitution_predictions.tsv", sep="\t", index=False
    )

    unlabelled = predictions[~predictions["is_training_variant"]]
    call_counts = (
        unlabelled["called_class"].value_counts().rename_axis("called_class")
        .reset_index(name="n_variants")
    )
    call_counts["share"] = call_counts["n_variants"] / len(unlabelled)
    call_counts.to_csv(
        args.output_dir / "unlabeled_call_counts.tsv", sep="\t", index=False
    )

    if "neutral_domain_class_T95" in predictions.columns:
        crosstab = pd.crosstab(
            predictions["called_class"], predictions["neutral_domain_class_T95"]
        )
        crosstab.to_csv(args.output_dir / "agreement_with_isolation_forest.tsv", sep="\t")

    plot_confusion_matrices(
        {selected_scheme: (confusion, selected_classes)},
        args.figure_dir / "selected_model_confusion_matrix.png",
    )
    plot_roc_pr(final_oof, selected_classes, n_classes, args.figure_dir / "selected_model_roc_pr.png")
    plot_coverage_accuracy(final_oof, n_classes, args.figure_dir / "abstention_tradeoff.png")
    plot_pca_projection(cohort, selected_features, selected_classes,
                        args.figure_dir / "feature_space_projection.png")
    plot_decision_landscape(final_estimator, cohort, selected_features, selected_classes,
                            args.figure_dir / "decision_landscape.png")
    plot_prediction_inventory(
        predictions, selected_classes,
        args.figure_dir / "prediction_class_inventory.png",
    )
    plot_probability_distributions(
        predictions, selected_classes,
        args.figure_dir / "prediction_score_distributions.png",
    )
    plot_genome_map(predictions, selected_classes, args.figure_dir / "genome_prediction_map.png")

    manifest = {
        "input": str(args.input),
        "budget": args.budget,
        "selected_model": selected_model,
        "selected_scheme": selected_scheme,
        "scheme_choice": scheme_choice,
        "selected_classes": selected_classes,
        "selected_features": selected_features,
        "final_hyperparameters": {k: str(v) for k, v in final_params.items()},
        "feature_selection": choice,
        "confidence_threshold": threshold,
        "overall_mean_rank": overall_rank.to_dict(),
        "critical_difference": {k: v for k, v in cd_results.items()},
        "bootstrap_ci": ci_rows,
        "n_labelled": int(len(cohort)),
        "n_unlabelled": int(len(unlabelled)),
        "unlabelled_call_shares": call_counts.set_index("called_class")["share"].to_dict(),
        "isolation_forest_baseline": (
            baseline_summary.set_index("method")["roc_auc_mean"].to_dict()
            if not baseline_summary.empty else {}
        ),
        "nested_cv_macro_f1": (
            nested.groupby("model")["macro_f1"].mean().to_dict() if len(nested) else {}
        ),
    }
    with open(args.output_dir / "selection_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=str)

    print("Done.", flush=True)
    return {
        "data": data,
        "audit": audit,
        "fold_metrics": fold_metrics,
        "summary": summary,
        "ranking": ranking,
        "scheme_comparison": scheme_comparison,
        "scheme_per_class": scheme_per_class,
        "scheme_choice": scheme_choice,
        "curve": curve,
        "choice": choice,
        "importance": importance,
        "per_class": per_class,
        "leakage": comparison,
        "nested": nested,
        "predictions": predictions,
        "call_counts": call_counts,
        "baseline_summary": baseline_summary,
        "confusion": confusion,
        "cohort": cohort,
        "manifest": manifest,
    }


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
