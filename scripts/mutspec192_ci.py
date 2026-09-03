"""Uncertainty estimation and reference-style plots for SBS192 spectra.

The plotting palette and substitution ordering follow the visual convention
used by PyMutSpec: reverse-complementary substitutions share one base colour.
When two spectra are compared, the first group uses a light shade and the
second group a dark shade of that same colour.

The main uncertainty model is a paired nonparametric bootstrap of whole mtDNA
positions.  All alternate alleles and both model groups at a position travel
together in a replicate.  The known rCRS context-opportunity table is held
fixed, while opportunity correction and the shared denominator across groups
are reapplied after resampling.  A conditional Poisson-count model remains
available as a sensitivity analysis.
"""

from __future__ import annotations

from math import erfc, sqrt
from itertools import combinations
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, to_rgb
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd


SBS12_REFERENCE_ORDER = (
    "C>A", "G>T", "C>G", "G>C", "C>T", "G>A",
    "T>A", "A>T", "T>C", "A>G", "T>G", "A>C",
)

SBS12_BASE_COLORS = {
    "C>A": "deepskyblue",
    "G>T": "deepskyblue",
    "C>G": "black",
    "G>C": "black",
    "C>T": "red",
    "G>A": "red",
    "T>A": "silver",
    "A>T": "silver",
    "T>C": "yellowgreen",
    "A>G": "yellowgreen",
    "T>G": "pink",
    "A>C": "pink",
}


def reference_sbs192_order() -> list[str]:
    """Return the PyMutSpec-style order of all 192 directional channels."""
    order: list[str] = []
    for substitution in SBS12_REFERENCE_ORDER:
        reference = substitution[0]
        if reference in {"C", "T"}:
            order.extend(
                f"{left}[{substitution}]{right}"
                for left in "ACGT"
                for right in "ACGT"
            )
        else:
            # Reverse-complementary blocks are ordered so paired channels
            # occupy the same within-block position as in PyMutSpec.
            order.extend(
                f"{left}[{substitution}]{right}"
                for right in "TGCA"
                for left in "TGCA"
            )
    if len(order) != 192 or len(set(order)) != 192:
        raise AssertionError("The reference SBS192 order must contain 192 channels.")
    return order


SBS192_REFERENCE_ORDER = tuple(reference_sbs192_order())


def _blend(color: str, target: tuple[float, float, float], amount: float) -> str:
    rgb = np.asarray(to_rgb(color), dtype=float)
    target_rgb = np.asarray(target, dtype=float)
    return to_hex((1.0 - amount) * rgb + amount * target_rgb)


def _light_color(color: str) -> str:
    return _blend(color, (1.0, 1.0, 1.0), 0.48)


def _dark_color(color: str) -> str:
    return _blend(color, (0.0, 0.0, 0.0), 0.14)


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    if p_values.ndim != 1:
        raise ValueError("p_values must be one-dimensional.")
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def _normal_two_sided_p(z_scores: np.ndarray) -> np.ndarray:
    return np.asarray([erfc(abs(float(z)) / sqrt(2.0)) for z in z_scores])


def _iter_strata(
    data: pd.DataFrame,
    strata_cols: Sequence[str],
):
    if not strata_cols:
        yield (), data
        return

    grouper = strata_cols[0] if len(strata_cols) == 1 else list(strata_cols)
    for key, stratum in data.groupby(grouper, sort=False, dropna=False):
        if len(strata_cols) == 1:
            key = (key,)
        yield tuple(key), stratum



def estimate_shared_spectrum_uncertainty(
    data: pd.DataFrame,
    *,
    group_col: str,
    group_order: Sequence[str],
    category_col: str = "substitution_type_192",
    category_order: Sequence[str] = SBS192_REFERENCE_ORDER,
    count_col: str = "combined_db_count_pc",
    opportunity_col: str = "ref_context_count",
    position_col: str = "position",
    strata_cols: Sequence[str] = (),
    n_simulations: int = 4000,
    confidence_level: float = 0.90,
    random_seed: int = 20260810,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate spectra and between-group differences by resampling.

    Every ordered pair in ``group_order`` is reported as
    ``comparison_group - reference_group``; with two groups that is the single
    difference ``group_order[1] - group_order[0]``. The denominator is shared
    across all groups in ``group_order``, so passing four groups renormalises
    the spectra over four rather than comparing two at a time.
    Benjamini-Hochberg correction is performed separately inside every stratum
    and every pair, across all SBS192 channels.

    Positions are the resampling unit: whole mtDNA positions are drawn with
    replacement, and one draw is shared by every group and every channel, so
    variants at the same position travel together. The denominator is shared
    across all groups, so the frequencies sum to one jointly and each group
    keeps its share of the combined mass.
    """
    if len(group_order) < 2:
        raise ValueError("At least two groups are required for a comparison.")
    if len(set(group_order)) != len(group_order):
        raise ValueError("group_order must not repeat a group.")
    if n_simulations < 200:
        raise ValueError("Use at least 200 simulations for stable intervals.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one.")

    required = {
        group_col, category_col, count_col, opportunity_col, *strata_cols,
    }
    required.add(position_col)
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing uncertainty columns: {missing}")

    category_order = tuple(category_order)
    if len(category_order) != 192 or len(set(category_order)) != 192:
        raise ValueError("category_order must contain 192 unique channels.")

    work = data[data[group_col].isin(group_order)].copy()
    if work.empty:
        raise ValueError("No rows remain after filtering to group_order.")

    count = pd.to_numeric(work[count_col], errors="raise").to_numpy(float)
    opportunity = pd.to_numeric(
        work[opportunity_col], errors="raise"
    ).to_numpy(float)
    if np.any(count < 0) or not np.isfinite(count).all():
        raise ValueError("Adjusted counts must be finite and non-negative.")
    if np.any(opportunity <= 0) or not np.isfinite(opportunity).all():
        raise ValueError("Opportunities must be finite and positive.")
    work[count_col] = count
    work[opportunity_col] = opportunity

    unknown_categories = sorted(
        set(work[category_col].dropna()) - set(category_order)
    )
    if unknown_categories:
        raise ValueError(f"Unknown SBS192 channels: {unknown_categories[:10]}")

    rng = np.random.default_rng(random_seed)
    spectrum_parts: list[pd.DataFrame] = []
    difference_parts: list[pd.DataFrame] = []
    alpha = 1.0 - confidence_level
    quantiles = [alpha / 2.0, 0.5, 1.0 - alpha / 2.0]
    group_lookup = {name: idx for idx, name in enumerate(group_order)}
    category_lookup = {name: idx for idx, name in enumerate(category_order)}
    n_categories = len(category_order)
    n_cells = len(group_order) * n_categories

    for stratum_key, stratum in _iter_strata(work, tuple(strata_cols)):
        group_index = stratum[group_col].map(group_lookup).to_numpy(int)
        category_index = stratum[category_col].map(category_lookup).to_numpy(int)
        cell_index = group_index * n_categories + category_index

        cell_lambda = np.bincount(
            cell_index,
            weights=stratum[count_col].to_numpy(float),
            minlength=n_cells,
        )
        cell_n_variants = np.bincount(cell_index, minlength=n_cells).astype(int)
        category_opportunity = stratum.groupby(
            category_col, sort=False
        )[opportunity_col]
        opportunity_nunique = category_opportunity.nunique()
        if (opportunity_nunique > 1).any():
            bad = opportunity_nunique[opportunity_nunique > 1].index.tolist()
            raise ValueError(
                "A channel contains multiple opportunity values: "
                f"{bad[:5]}"
            )
        category_scale = np.zeros(n_categories, dtype=float)
        for category, value in category_opportunity.first().items():
            category_scale[category_lookup[category]] = 1.0 / float(value)
        if np.any(category_scale <= 0):
            missing_categories = np.asarray(category_order)[category_scale <= 0]
            raise ValueError(
                "No opportunity was available for SBS192 channels: "
                f"{missing_categories[:5].tolist()}"
            )
        cell_scale = np.tile(category_scale, len(group_order))

        point_weight = cell_lambda * cell_scale
        if point_weight.sum() <= 0:
            raise ValueError(f"Non-positive shared spectrum total for {stratum_key}.")
        point_frequency = point_weight / point_weight.sum()

        simulations = np.empty((n_simulations, n_cells), dtype=float)
        if stratum[position_col].isna().any():
            raise ValueError(f"{position_col!r} contains missing values.")
        position_codes, positions = pd.factorize(
            stratum[position_col], sort=True
        )
        n_bootstrap_units = int(len(positions))
        if n_bootstrap_units < 2:
            raise ValueError(
                "Position bootstrap requires at least two unique positions."
            )
        position_contribution = np.zeros(
            (n_bootstrap_units, n_cells), dtype=float
        )
        np.add.at(
            position_contribution,
            (position_codes, cell_index),
            stratum[count_col].to_numpy(float),
        )
        # How concentrated each cell is over positions. Positions are the
        # resampling unit, so this, and not the carrier count, sets the width of
        # the interval.
        cell_position_total = position_contribution.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            position_share = np.divide(
                position_contribution,
                cell_position_total[None, :],
                out=np.zeros_like(position_contribution),
                where=cell_position_total[None, :] > 0,
            )
        simpson = (position_share ** 2).sum(axis=0)
        n_effective_positions = np.divide(
            1.0, simpson, out=np.zeros_like(simpson), where=simpson > 0
        )
        top_position_share = position_share.max(axis=0)
        n_positions_present = (position_contribution > 0).sum(axis=0)

        probabilities = np.full(
            n_bootstrap_units, 1.0 / n_bootstrap_units
        )
        batch_size = min(50, n_simulations)
        for start in range(0, n_simulations, batch_size):
            stop = min(start + batch_size, n_simulations)
            multiplicity = rng.multinomial(
                n_bootstrap_units,
                probabilities,
                size=stop - start,
            )
            sampled_count = multiplicity @ position_contribution
            sampled_weight = sampled_count * cell_scale
            if np.any(sampled_weight.sum(axis=1) <= 0):
                raise ValueError("A bootstrap replicate has zero total weight.")
            simulations[start:stop] = (
                sampled_weight / sampled_weight.sum(axis=1)[:, None]
            )

        q_low, q_median, q_high = np.quantile(
            simulations, quantiles, axis=0
        )
        stratum_values = dict(zip(strata_cols, stratum_key))

        spectrum_part = pd.DataFrame({
            group_col: np.repeat(group_order, n_categories),
            category_col: np.tile(category_order, len(group_order)),
            "n_variants": cell_n_variants,
            "weighted_frequency": point_frequency,
            "bootstrap_median": q_median,
            "ci_lower": q_low,
            "ci_upper": q_high,
            "n_bootstrap_units": n_bootstrap_units,
        })
        # Aliases under the PyMutSpec names, so a spectrum table can be handed
        # straight to plot_mutspec192. The quantiles are the ones that library
        # expects: q05 and q95 at the default confidence level of 0.90.
        spectrum_part["n_positions"] = n_positions_present
        spectrum_part["n_effective_positions"] = n_effective_positions
        spectrum_part["top_position_share"] = top_position_share
        spectrum_part["MutSpec"] = spectrum_part["weighted_frequency"]
        spectrum_part["MutSpec_median"] = spectrum_part["bootstrap_median"]
        spectrum_part["MutSpec_q05"] = spectrum_part["ci_lower"]
        spectrum_part["MutSpec_q95"] = spectrum_part["ci_upper"]
        for column, value in stratum_values.items():
            spectrum_part.insert(0, column, value)
        spectrum_parts.append(spectrum_part)

        for reference_index, comparison_index in combinations(
            range(len(group_order)), 2
        ):
            reference_slice = slice(
                reference_index * n_categories, (reference_index + 1) * n_categories
            )
            comparison_slice = slice(
                comparison_index * n_categories, (comparison_index + 1) * n_categories
            )
            point_difference = (
                point_frequency[comparison_slice] - point_frequency[reference_slice]
            )
            simulated_difference = (
                simulations[:, comparison_slice] - simulations[:, reference_slice]
            )
            diff_low, diff_median, diff_high = np.quantile(
                simulated_difference, quantiles, axis=0
            )
            diff_se = simulated_difference.std(axis=0, ddof=1)
            z_score = np.divide(
                point_difference,
                diff_se,
                out=np.zeros_like(point_difference),
                where=diff_se > 0,
            )
            centered_difference = simulated_difference - point_difference
            p_value = (
                1
                + np.sum(
                    np.abs(centered_difference)
                    >= np.abs(point_difference),
                    axis=0,
                )
            ) / (n_simulations + 1)
            p_value_method = "centered_position_bootstrap"
            q_value = _benjamini_hochberg(p_value)
            ci_excludes_zero = (diff_low > 0) | (diff_high < 0)

            difference_part = pd.DataFrame({
                "reference_group": group_order[reference_index],
                "comparison_group": group_order[comparison_index],
                category_col: category_order,
                "frequency_difference": point_difference,
                "bootstrap_median_difference": diff_median,
                "ci_lower": diff_low,
                "ci_upper": diff_high,
                "bootstrap_se": diff_se,
                "z_score": z_score,
                "p_value": p_value,
                "p_value_method": p_value_method,
                "q_value_bh": q_value,
                "ci_excludes_zero": ci_excludes_zero,
                "significant_fdr_05": ci_excludes_zero & (q_value < 0.05),
                "n_bootstrap_units": n_bootstrap_units,
            })
            for column, value in stratum_values.items():
                difference_part.insert(0, column, value)
            difference_parts.append(difference_part)

    spectrum_result = pd.concat(spectrum_parts, ignore_index=True)
    difference_result = pd.concat(difference_parts, ignore_index=True)
    for result in (spectrum_result, difference_result):
        result["confidence_level"] = confidence_level
        result["n_simulations"] = n_simulations
        result["resampling_method"] = "position_bootstrap"
        result["bootstrap_unit"] = "mtDNA_position"
        result["opportunities_resampled"] = False
        result["uncertainty_model"] = (
            "paired nonparametric bootstrap of whole mtDNA positions"
        )
    return spectrum_result, difference_result


def _plot_positions(n_categories: int = 192, block_size: int = 16, gap: float = 1.8):
    index = np.arange(n_categories, dtype=float)
    return index + np.floor(index / block_size) * gap


def _ordered_plot_frame(
    data: pd.DataFrame,
    *,
    group_name: str,
    group_col: str,
    category_col: str,
) -> pd.DataFrame:
    result = (
        data[data[group_col] == group_name]
        .set_index(category_col)
        .reindex(SBS192_REFERENCE_ORDER)
        .reset_index()
    )
    required = {"weighted_frequency", "ci_lower", "ci_upper"}
    missing = sorted(required.difference(result.columns))
    if missing or result[list(required)].isna().any().any():
        raise ValueError(
            f"Incomplete uncertainty data for {group_name}: missing={missing}"
        )
    return result


def _marked_significance_mask(
    difference: pd.DataFrame,
    *,
    category_col: str,
    top_n_significant: int | None,
) -> tuple[np.ndarray, int]:
    """Select the largest significant differences for plot annotations."""
    ordered = (
        difference.set_index(category_col)
        .reindex(SBS192_REFERENCE_ORDER)
    )
    significant = (
        ordered["significant_fdr_05"].fillna(False).to_numpy(bool)
    )
    if top_n_significant is None:
        return significant, int(significant.sum())
    if top_n_significant < 1:
        raise ValueError("top_n_significant must be positive or None.")

    marked = np.zeros_like(significant)
    candidate_indices = np.flatnonzero(significant)
    if candidate_indices.size:
        magnitude = np.abs(
            ordered["frequency_difference"].to_numpy(float)
        )
        ranked_candidates = candidate_indices[
            np.argsort(magnitude[candidate_indices])[::-1]
        ]
        marked[ranked_candidates[:top_n_significant]] = True
    return marked, min(top_n_significant, int(candidate_indices.size))


def _decorate_sbs192_axis(
    ax,
    *,
    positions: np.ndarray,
    show_xticklabels: bool,
    show_class_headers: bool,
):
    ax.grid(axis="y", alpha=0.35, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(positions[0] - 0.8, positions[-1] + 0.8)
    ax.set_xticks(positions)
    if show_xticklabels:
        ax.set_xticklabels(
            SBS192_REFERENCE_ORDER,
            rotation=90,
            ha="center",
            fontsize=4.8,
            fontfamily="monospace",
        )
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)

    for block in range(1, len(SBS12_REFERENCE_ORDER)):
        left = positions[block * 16 - 1]
        right = positions[block * 16]
        ax.axvline((left + right) / 2.0, color="#d9d9d9", linewidth=0.7)

    if show_class_headers:
        transform = ax.get_xaxis_transform()
        for block, substitution in enumerate(SBS12_REFERENCE_ORDER):
            start = positions[block * 16] - 0.42
            end = positions[(block + 1) * 16 - 1] + 0.42
            color = SBS12_BASE_COLORS[substitution]
            ax.add_patch(Rectangle(
                (start, 1.012),
                end - start,
                0.028,
                transform=transform,
                facecolor=color,
                edgecolor="none",
                clip_on=False,
            ))
            ax.text(
                (start + end) / 2.0,
                1.048,
                substitution,
                transform=transform,
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
                clip_on=False,
            )


def plot_mutspec192_with_ci(
    spectrum: pd.DataFrame,
    *,
    group_order: Sequence[str],
    group_labels: Mapping[str, str] | None = None,
    difference: pd.DataFrame | None = None,
    group_col: str = "group_name",
    category_col: str = "substitution_type_192",
    title: str = "192-component mutational spectrum",
    ylabel: str = "Shared-normalized weighted frequency",
    xlabel: str = "Functional-strand-oriented trinucleotide substitution",
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (34, 7),
    ax=None,
    show: bool = True,
    show_xticklabels: bool = True,
    show_class_headers: bool = True,
    show_legend: bool = True,
    top_n_significant: int | None = 10,
):
    """Plot one or two SBS192 spectra with asymmetric confidence intervals."""
    if len(group_order) not in {1, 2}:
        raise ValueError("One or two groups can be drawn on a single axis.")
    group_labels = dict(group_labels or {})
    created_figure = ax is None
    if created_figure:
        _, ax = plt.subplots(figsize=figsize)

    positions = _plot_positions()
    width = 0.34 if len(group_order) == 2 else 0.62
    offsets = [-width / 2.0, width / 2.0] if len(group_order) == 2 else [0.0]
    all_upper: list[np.ndarray] = []

    for group_index, (group_name, offset) in enumerate(zip(group_order, offsets)):
        plot_df = _ordered_plot_frame(
            spectrum,
            group_name=group_name,
            group_col=group_col,
            category_col=category_col,
        )
        substitutions = plot_df[category_col].str.slice(2, 5)
        base_colors = [SBS12_BASE_COLORS[value] for value in substitutions]
        if len(group_order) == 1:
            colors = base_colors
        elif group_index == 0:
            colors = [_light_color(value) for value in base_colors]
        else:
            colors = [_dark_color(value) for value in base_colors]

        y = plot_df["weighted_frequency"].to_numpy(float)
        median = plot_df["bootstrap_median"].to_numpy(float)
        lower = plot_df["ci_lower"].to_numpy(float)
        upper = plot_df["ci_upper"].to_numpy(float)
        all_upper.append(upper)
        x = positions + offset
        ax.bar(
            x,
            y,
            width=width,
            color=colors,
            edgecolor="none",
            zorder=2,
        )
        # PyMutSpec convention: the bar is the point estimate, the marker and the
        # interval sit on the resampling median, and the interval is measured
        # from that median rather than from the bar.
        ax.errorbar(
            x,
            median,
            yerr=np.vstack(
                [np.maximum(median - lower, 0), np.maximum(upper - median, 0)]
            ),
            fmt=".",
            color="gray",
            markersize=2.0,
            elinewidth=0.7,
            capsize=2,
            zorder=3,
        )

    if difference is not None and len(group_order) == 2:
        significance, n_marked = _marked_significance_mask(
            difference,
            category_col=category_col,
            top_n_significant=top_n_significant,
        )
        upper_envelope = np.maximum.reduce(all_upper)
        y_range = max(float(np.nanmax(upper_envelope)), 1e-12)
        for x, upper, is_significant in zip(positions, upper_envelope, significance):
            if is_significant:
                ax.text(
                    x,
                    upper + y_range * 0.025,
                    "*",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    color="black",
                    zorder=4,
                )
        if significance.any():
            ax.text(
                0.995,
                0.985,
                f"* top {n_marked} significant differences by |difference|",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7,
            )
            current_top = ax.get_ylim()[1]
            ax.set_ylim(
                top=max(current_top, float(np.nanmax(upper_envelope)) * 1.14)
            )

    _decorate_sbs192_axis(
        ax,
        positions=positions,
        show_xticklabels=show_xticklabels,
        show_class_headers=show_class_headers,
    )
    if show_class_headers:
        ax.set_title(title, fontsize=12, y=1.085, pad=2)
    else:
        ax.set_title(title, fontsize=12, pad=8)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel if show_xticklabels else "")

    if show_legend:
        if len(group_order) == 1:
            handles = [Patch(
                facecolor="#777777",
                label=group_labels.get(group_order[0], group_order[0]),
            )]
        else:
            handles = [
                Patch(
                    facecolor=_light_color("#666666"),
                    label=group_labels.get(group_order[0], group_order[0]),
                ),
                Patch(
                    facecolor=_dark_color("#666666"),
                    label=group_labels.get(group_order[1], group_order[1]),
                ),
            ]
        ax.legend(handles=handles, loc="upper left", frameon=False, ncol=len(handles))

    if created_figure:
        ax.figure.tight_layout()
    if output_path is not None:
        ax.figure.savefig(output_path, dpi=300, bbox_inches="tight")
    if created_figure and show:
        plt.show()
    elif created_figure and not show:
        plt.close(ax.figure)
    return ax


def plot_mutspec192_difference_with_ci(
    difference: pd.DataFrame,
    *,
    category_col: str = "substitution_type_192",
    title: str = "Difference between 192-component spectra",
    ylabel: str = "Shared-normalized frequency difference",
    xlabel: str = "Functional-strand-oriented trinucleotide substitution",
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (34, 7),
    ax=None,
    show: bool = True,
    show_xticklabels: bool = True,
    show_class_headers: bool = True,
    top_n_significant: int | None = 10,
):
    """Plot the second-minus-first group difference and its confidence interval."""
    plot_df = (
        difference.set_index(category_col)
        .reindex(SBS192_REFERENCE_ORDER)
        .reset_index()
    )
    required = {
        "frequency_difference", "bootstrap_median_difference",
        "ci_lower", "ci_upper", "significant_fdr_05",
    }
    missing = sorted(required.difference(plot_df.columns))
    if missing or plot_df[list(required - {"significant_fdr_05"})].isna().any().any():
        raise ValueError(f"Incomplete difference data: missing={missing}")

    created_figure = ax is None
    if created_figure:
        _, ax = plt.subplots(figsize=figsize)
    positions = _plot_positions()
    substitutions = plot_df[category_col].str.slice(2, 5)
    colors = [SBS12_BASE_COLORS[value] for value in substitutions]
    y = plot_df["frequency_difference"].to_numpy(float)
    lower = plot_df["ci_lower"].to_numpy(float)
    upper = plot_df["ci_upper"].to_numpy(float)
    significant, n_marked = _marked_significance_mask(
        difference,
        category_col=category_col,
        top_n_significant=top_n_significant,
    )

    ax.bar(positions, y, width=0.62, color=colors, edgecolor="none", zorder=2)
    median = plot_df["bootstrap_median_difference"].to_numpy(float)
    ax.errorbar(
        positions,
        median,
        yerr=np.vstack(
            [np.maximum(median - lower, 0), np.maximum(upper - median, 0)]
        ),
        fmt=".",
        color="gray",
        markersize=2.0,
        elinewidth=0.7,
        capsize=2,
        zorder=3,
    )
    ax.axhline(0.0, color="black", linewidth=0.8, zorder=1)
    span = max(float(np.nanmax(upper) - np.nanmin(lower)), 1e-12)
    for x, y_value, lo, hi, is_significant in zip(
        positions, y, lower, upper, significant
    ):
        if is_significant:
            marker_y = (
                hi + 0.025 * span
                if y_value >= 0
                else lo - 0.025 * span
            )
            ax.text(
                x,
                marker_y,
                "*",
                ha="center",
                va="bottom" if marker_y >= 0 else "top",
                fontsize=7.5,
                color="black",
                zorder=4,
            )

    _decorate_sbs192_axis(
        ax,
        positions=positions,
        show_xticklabels=show_xticklabels,
        show_class_headers=show_class_headers,
    )
    if show_class_headers:
        ax.set_title(title, fontsize=12, y=1.085, pad=2)
    else:
        ax.set_title(title, fontsize=12, pad=8)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel if show_xticklabels else "")
    if significant.any():
        ax.text(
            0.995,
            0.985,
            f"* top {n_marked} significant differences by |difference|",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
        )
        ax.margins(y=0.12)

    if created_figure:
        ax.figure.tight_layout()
    if output_path is not None:
        ax.figure.savefig(output_path, dpi=300, bbox_inches="tight")
    if created_figure and show:
        plt.show()
    elif created_figure and not show:
        plt.close(ax.figure)
    return ax
