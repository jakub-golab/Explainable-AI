"""Cross-validation stability of important amino acid residues across folds."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from src.interpretability import aggregate_to_residues, model_feature_importance
from src.models import _build_pipeline, _make_models, resolve_imbalance_strategy


@dataclass
class CVStabilityResult:
    fold_residues: dict[int, list[str]]
    frequency: pd.DataFrame
    core_residues: list[str]
    mean_pairwise_jaccard: float
    min_pairwise_jaccard: float
    pairwise_jaccard: pd.DataFrame
    n_folds: int
    top_n: int


def _jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def top_residues_for_fold(
    pipeline,
    feature_names: list[str],
    top_n: int,
) -> pd.DataFrame:
    """Extract top-N residues from a model fitted on one CV training fold."""
    importance = model_feature_importance(pipeline, feature_names)
    return aggregate_to_residues(importance, top_n=top_n)


def cv_residue_stability(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    receptor: str,
    use_undersampling: bool = False,
    top_n: int = 15,
    n_splits: int = 5,
    random_state: int = 42,
) -> CVStabilityResult:
    """
    Test whether the same amino acids rank highest in every CV fold.

    For each fold: train on train split (with undersampling if enabled),
    rank residues by native model importance, record top-N.
    """
    imbalance_strategy = resolve_imbalance_strategy(receptor, use_undersampling)
    models = _make_models(y, imbalance_strategy)
    if model_name not in models:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(models)}")

    base_model = models[model_name]
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    feature_names = list(X.columns)

    fold_residues: dict[int, list[str]] = {}
    fold_tables: dict[int, pd.DataFrame] = {}

    for fold_idx, (train_idx, _test_idx) in enumerate(cv.split(X, y)):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]

        pipeline = _build_pipeline(clone(base_model), imbalance_strategy)
        pipeline.fit(X_train, y_train)

        residues_df = top_residues_for_fold(pipeline, feature_names, top_n=top_n)
        fold_tables[fold_idx] = residues_df
        fold_residues[fold_idx] = residues_df["residue"].tolist()

    # Frequency: how many folds each residue appears in top-N
    residue_counts: dict[str, int] = {}
    for residues in fold_residues.values():
        for r in residues:
            residue_counts[r] = residue_counts.get(r, 0) + 1

    frequency_records = []
    for residue, count in sorted(residue_counts.items(), key=lambda x: (-x[1], x[0])):
        meta = next(
            (row for df in fold_tables.values() for _, row in df.iterrows() if row["residue"] == residue),
            None,
        )
        frequency_records.append(
            {
                "residue": residue,
                "aa": meta["aa"] if meta is not None else "?",
                "position": meta["position"] if meta is not None else 0,
                "folds_in_top_n": count,
                "fold_fraction": count / n_splits,
                "stable": count == n_splits,
            }
        )
    frequency = pd.DataFrame(frequency_records)

    sets = {fold: set(residues) for fold, residues in fold_residues.items()}
    core_residues = sorted(set.intersection(*sets.values())) if sets else []

    pairwise_rows = []
    jaccard_values = []
    for (f1, s1), (f2, s2) in itertools.combinations(sets.items(), 2):
        j = _jaccard(s1, s2)
        jaccard_values.append(j)
        pairwise_rows.append({"fold_1": f1, "fold_2": f2, "overlap": len(s1 & s2), "jaccard": j})

    pairwise_jaccard = pd.DataFrame(pairwise_rows)

    return CVStabilityResult(
        fold_residues=fold_residues,
        frequency=frequency,
        core_residues=core_residues,
        mean_pairwise_jaccard=float(np.mean(jaccard_values)) if jaccard_values else 0.0,
        min_pairwise_jaccard=float(np.min(jaccard_values)) if jaccard_values else 0.0,
        pairwise_jaccard=pairwise_jaccard,
        n_folds=n_splits,
        top_n=top_n,
    )


def stability_summary(result: CVStabilityResult) -> dict[str, Any]:
    """Human-readable summary for reporting."""
    n_stable = int(result.frequency["stable"].sum()) if not result.frequency.empty else 0
    return {
        "top_n": result.top_n,
        "n_folds": result.n_folds,
        "core_residues_in_all_folds": result.core_residues,
        "n_core_residues": len(result.core_residues),
        "mean_pairwise_jaccard": result.mean_pairwise_jaccard,
        "min_pairwise_jaccard": result.min_pairwise_jaccard,
        "n_residues_in_all_folds": n_stable,
        "hypothesis_supported": len(result.core_residues) > 0 and result.mean_pairwise_jaccard >= 0.5,
    }


def plot_cv_stability(result: CVStabilityResult, out_path: str, receptor: str, model_name: str) -> None:
    """Bar chart of residue frequency across CV folds."""
    plot_df = result.frequency.head(result.top_n + 5).copy()
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, max(5, len(plot_df) * 0.35)))
    colors = ["#2ecc71" if s else "#3498db" for s in plot_df["stable"]]
    ax.barh(plot_df["residue"], plot_df["folds_in_top_n"], color=colors)
    ax.set_xlabel(f"Folds (of {result.n_folds}) in top-{result.top_n}")
    ax.set_title(
        f"{receptor.upper()} – CV stability of top residues ({model_name})\n"
        f"green = in all folds | mean Jaccard = {result.mean_pairwise_jaccard:.2f}"
    )
    ax.set_xlim(0, result.n_folds)
    ax.axvline(result.n_folds, color="gray", linestyle="--", alpha=0.5, label="all folds")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_fold_heatmap(result: CVStabilityResult, out_path: str, receptor: str) -> None:
    """Heatmap: rows=residues, cols=folds, 1 if residue in top-N for that fold."""
    all_residues = result.frequency["residue"].tolist()
    if not all_residues:
        return

    matrix = np.zeros((len(all_residues), result.n_folds), dtype=int)
    residue_to_idx = {r: i for i, r in enumerate(all_residues)}

    for fold, residues in result.fold_residues.items():
        for r in residues:
            if r in residue_to_idx:
                matrix[residue_to_idx[r], fold] = 1

    fig, ax = plt.subplots(figsize=(8, max(5, len(all_residues) * 0.3)))
    sns.heatmap(
        matrix,
        yticklabels=all_residues,
        xticklabels=[f"Fold {i}" for i in range(result.n_folds)],
        cmap="Greens",
        cbar_kws={"label": "In top-N"},
        ax=ax,
    )
    ax.set_title(f"{receptor.upper()} – Residue presence per CV fold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
