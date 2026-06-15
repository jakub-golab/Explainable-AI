"""Simple statistical tests comparing active vs inactive interaction frequencies."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu

from src.data_loading import feature_to_residue_key, parse_feature_name


def feature_statistical_tests(
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    For each binary interaction feature, test association with activity label.

    Uses Fisher's exact test (preferred for sparse binary data) and reports
    frequency difference between active and inactive groups.
    """
    active = X[y == 1]
    inactive = X[y == 0]
    records = []

    for col in X.columns:
        a_present = int(active[col].sum())
        a_absent = len(active) - a_present
        i_present = int(inactive[col].sum())
        i_absent = len(inactive) - i_present

        table = [[a_present, a_absent], [i_present, i_absent]]
        _, p_fisher = fisher_exact(table)
        chi2, p_chi2, _, _ = chi2_contingency(table)

        freq_active = a_present / len(active) if len(active) else 0.0
        freq_inactive = i_present / len(inactive) if len(inactive) else 0.0

        records.append(
            {
                "feature": col,
                "freq_active": freq_active,
                "freq_inactive": freq_inactive,
                "freq_diff": freq_active - freq_inactive,
                "p_fisher": p_fisher,
                "p_chi2": p_chi2,
                "significant": p_fisher < alpha,
            }
        )

    df = pd.DataFrame(records).sort_values("p_fisher")
    df["p_fisher_adj"] = _benjamini_hochberg(df["p_fisher"].values)
    df["significant_adj"] = df["p_fisher_adj"] < alpha
    return df.reset_index(drop=True)


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = np.empty(n)
    for i, idx in enumerate(order):
        ranked[idx] = p_values[idx] * n / (i + 1)
    # enforce monotonicity
    for i in range(n - 2, -1, -1):
        idx = order[i]
        next_idx = order[i + 1]
        ranked[idx] = min(ranked[idx], ranked[next_idx])
    return np.minimum(ranked, 1.0)


def aggregate_stats_to_residues(stats_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Rank residues by strongest statistical signal.

    Uses minimum adjusted p-value across interaction types at each residue.
    """
    residue_records: dict[str, dict] = {}

    for _, row in stats_df.iterrows():
        residue = feature_to_residue_key(row["feature"])
        parsed = parse_feature_name(row["feature"])
        if residue not in residue_records or row["p_fisher_adj"] < residue_records[residue]["p_fisher_adj"]:
            residue_records[residue] = {
                "residue": residue,
                "aa": parsed[0] if parsed else "?",
                "position": parsed[1] if parsed else 0,
                "best_feature": row["feature"],
                "p_fisher_adj": row["p_fisher_adj"],
                "freq_diff": row["freq_diff"],
                "significant_adj": row["significant_adj"],
            }

    out = pd.DataFrame(residue_records.values()).sort_values("p_fisher_adj")
    return out.head(top_n).reset_index(drop=True)


def top_statistical_features(stats_df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Return top-N individual interaction features by adjusted Fisher p-value."""
    import math

    out = stats_df.sort_values("p_fisher_adj").head(top_n).copy()
    out["score"] = -out["p_fisher_adj"].clip(lower=1e-300).apply(math.log10)
    return out.reset_index(drop=True)


def compare_features(
    primary_top: list[str] | set[str],
    secondary_top: list[str] | set[str],
) -> dict:
    """Generic Jaccard overlap between two feature/residue sets."""
    a = set(primary_top)
    b = set(secondary_top)
    overlap = a & b
    union = a | b
    return {
        "primary_top": sorted(a),
        "secondary_top": sorted(b),
        "overlap": sorted(overlap),
        "n_overlap": len(overlap),
        "jaccard": len(overlap) / len(union) if union else 0.0,
        "primary_only": sorted(a - b),
        "secondary_only": sorted(b - a),
    }


def compare_shap_vs_statistics_features(
    shap_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    top_n: int = 15,
) -> dict:
    """Compare top interaction features from SHAP vs Fisher test."""
    shap_set = set(shap_df.head(top_n)["feature"])
    stat_set = set(stats_df.sort_values("p_fisher_adj").head(top_n)["feature"])
    return compare_features(shap_set, stat_set)


def compare_ml_vs_statistics(
    ml_residues: pd.DataFrame,
    stat_residues: pd.DataFrame,
    top_n: int = 15,
) -> dict:
    """Compare top residues from ML interpretability vs statistical analysis."""
    ml_set = set(ml_residues.head(top_n)["residue"])
    stat_set = set(stat_residues.head(top_n)["residue"])
    overlap = ml_set & stat_set
    union = ml_set | stat_set

    return {
        "ml_top": sorted(ml_set),
        "stat_top": sorted(stat_set),
        "overlap": sorted(overlap),
        "n_overlap": len(overlap),
        "jaccard": len(overlap) / len(union) if union else 0.0,
        "ml_only": sorted(ml_set - stat_set),
        "stat_only": sorted(stat_set - ml_set),
    }
