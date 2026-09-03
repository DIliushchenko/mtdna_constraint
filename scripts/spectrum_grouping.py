"""Assign every substitution a severity class for the mutational-spectrum comparison.

The spectrum notebooks weight each variant by its population carrier count. A
grouping model that uses population frequency as a predictor would therefore
assign variants to groups partly by the same quantity that later weights them,
and the resulting spectra would differ in part by construction. The T95 grouping
of notebook 02 has this problem: the Isolation Forest was fitted on four rarity
features.

This module avoids it by refitting the selected classifier on a frequency-free
panel — local constraint, cross-species conservation, consequence class and
codon position. Group membership is then independent of the carrier counts used
for weighting, and a difference between class spectra cannot be an artefact of
that circularity.

The cost is real and is reported: the frequency-free model is weaker than the
full one. What it buys is that a spectral difference between classes means
something.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import FEATURE_COLUMNS, SCHEME_CLASSES  # noqa: E402

FREQUENCY_FEATURES = ["hom_rarity_soft"]
GROUPING_FEATURES = [f for f in FEATURE_COLUMNS if f not in FREQUENCY_FEATURES]


def out_of_fold_quality(
    data: pd.DataFrame,
    features: list[str],
    classes: list[str],
    scheme: str,
    n_splits: int,
    n_repeats: int,
    random_state: int,
    model_params: dict,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Position-grouped out-of-fold estimate for the frequency-free model."""
    cohort = data[data[f"label_{scheme}"].notna()].copy()
    order = {name: index for index, name in enumerate(classes)}
    y = cohort[f"label_{scheme}"].map(order).to_numpy()
    X = cohort[features].to_numpy(dtype=float)
    groups = cohort["position"].to_numpy()

    accumulated = np.zeros((len(y), len(classes)))
    for repeat in range(n_repeats):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state + repeat
        )
        for train_idx, test_idx in splitter.split(X, y, groups):
            model = RandomForestClassifier(
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=-1,
                **model_params,
            )
            model.fit(X[train_idx], y[train_idx])
            proba = model.predict_proba(X[test_idx])
            for column, class_value in enumerate(model.classes_):
                accumulated[test_idx, int(class_value)] += proba[:, column] / n_repeats

    y_pred = np.argmax(accumulated, axis=1)
    rows = []
    for index, class_name in enumerate(classes):
        rows.append(
            {
                "class": class_name,
                "support": int((y == index).sum()),
                "recall": float(
                    ((y_pred == index) & (y == index)).sum() / max((y == index).sum(), 1)
                ),
                "precision": float(
                    ((y_pred == index) & (y == index)).sum() / max((y_pred == index).sum(), 1)
                ),
            }
        )
    summary = pd.DataFrame(rows)
    summary.attrs["macro_f1"] = f1_score(
        y, y_pred, labels=list(range(len(classes))), average="macro", zero_division=0
    )
    summary.attrs["quadratic_kappa"] = cohen_kappa_score(
        y, y_pred, labels=list(range(len(classes))), weights="quadratic"
    )
    summary.attrs["confusion"] = confusion_matrix(
        y, y_pred, labels=list(range(len(classes)))
    )
    return summary, accumulated


def assign_classes(
    data: pd.DataFrame,
    features: list[str],
    classes: list[str],
    scheme: str,
    random_state: int,
    model_params: dict,
    confidence_threshold: float,
) -> pd.DataFrame:
    labelled = data[data[f"label_{scheme}"].notna()]
    order = {name: index for index, name in enumerate(classes)}
    y = labelled[f"label_{scheme}"].map(order).to_numpy()

    model = RandomForestClassifier(
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
        **model_params,
    )
    model.fit(labelled[features].to_numpy(dtype=float), y)

    proba = model.predict_proba(data[features].to_numpy(dtype=float))
    aligned = np.zeros((len(data), len(classes)))
    for column, class_value in enumerate(model.classes_):
        aligned[:, int(class_value)] = proba[:, column]

    predicted = np.argmax(aligned, axis=1)
    confidence = aligned.max(axis=1)

    out = data[["variant_id", "position", "reference", "alternate"]].copy()
    out["spectrum_class"] = np.array(classes)[predicted]
    out["spectrum_class_confidence"] = confidence
    out["spectrum_class_confident"] = np.where(
        confidence >= confidence_threshold, out["spectrum_class"], "uncertain"
    )
    out["expected_severity"] = aligned @ np.arange(len(classes))
    for index, class_name in enumerate(classes):
        out[f"spectrum_proba_{class_name}"] = aligned[:, index]
    out["known_class"] = data[f"label_{scheme}"].to_numpy()
    return out


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path,
                        default=root / "data/processed/model_features.tsv")
    parser.add_argument("--manifest", type=Path,
                        default=root / "results/classification/selection_manifest.json")
    parser.add_argument("--output", type=Path,
                        default=root / "results/classification/spectrum_class_assignment.tsv")
    parser.add_argument("--quality-output", type=Path,
                        default=root / "results/classification/spectrum_grouping_quality.tsv")
    parser.add_argument("--scheme", default="4class", choices=sorted(SCHEME_CLASSES))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--random-state", type=int, default=20260901)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.features, sep="\t", low_memory=False)
    classes = SCHEME_CLASSES[args.scheme]

    model_params = {"n_estimators": 600, "min_samples_leaf": 4, "max_features": "sqrt"}
    if args.manifest.exists():
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        stored = manifest.get("final_hyperparameters", {})
        parsed = {}
        for key, value in stored.items():
            name = key.replace("model__", "")
            if name not in {"n_estimators", "min_samples_leaf", "max_features", "max_depth"}:
                continue
            if value == "None":
                parsed[name] = None
            elif str(value).isdigit():
                parsed[name] = int(value)
            else:
                parsed[name] = value
        if parsed:
            model_params = parsed

    print(f"grouping features ({len(GROUPING_FEATURES)}): {GROUPING_FEATURES}")
    print(f"excluded to keep grouping independent of the spectrum weights: {FREQUENCY_FEATURES}")
    print(f"model params: {model_params}\n")

    quality, _ = out_of_fold_quality(
        data, GROUPING_FEATURES, classes, args.scheme,
        args.folds, args.repeats, args.random_state, model_params,
    )
    print("Out-of-fold quality of the frequency-free grouping model:")
    print(quality.round(4).to_string(index=False))
    print(f"\n  macro F1        {quality.attrs['macro_f1']:.4f}")
    print(f"  quadratic kappa {quality.attrs['quadratic_kappa']:.4f}")
    print("\n  confusion (rows true, columns predicted):")
    print(pd.DataFrame(
        quality.attrs["confusion"],
        index=[f"true_{c}" for c in classes],
        columns=[f"pred_{c}" for c in classes],
    ).to_string())

    assignment = assign_classes(
        data, GROUPING_FEATURES, classes, args.scheme,
        args.random_state, model_params, args.confidence_threshold,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    assignment.to_csv(args.output, sep="\t", index=False)
    quality_out = quality.copy()
    quality_out["macro_f1"] = quality.attrs["macro_f1"]
    quality_out["quadratic_kappa"] = quality.attrs["quadratic_kappa"]
    quality_out["scheme"] = args.scheme
    quality_out.to_csv(args.quality_output, sep="\t", index=False)

    print("\nAssigned classes across all substitutions:")
    counts = assignment["spectrum_class"].value_counts().reindex(classes, fill_value=0)
    for class_name, n in counts.items():
        print(f"  {class_name:12} {n:7,}  ({n / len(assignment):.1%})")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
