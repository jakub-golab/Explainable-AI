"""Interpretability: feature importance, SHAP, and amino-acid-level aggregation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from src.data_loading import feature_to_residue_key, parse_feature_name


def _get_classifier(pipeline) -> Any:
    """Extract the classifier step from an imblearn/sklearn pipeline."""
    if hasattr(pipeline, "named_steps"):
        return pipeline.named_steps["clf"]
    return pipeline


def model_feature_importance(
    pipeline,
    feature_names: list[str],
) -> pd.DataFrame:
    """Native feature importance from tree models or |coefficients| for linear models."""
    clf = _get_classifier(pipeline)

    if hasattr(clf, "feature_importances_"):
        values = clf.feature_importances_
        method = "native_importance"
    elif hasattr(clf, "coef_"):
        values = np.abs(clf.coef_.ravel())
        method = "abs_coefficient"
    else:
        raise ValueError(f"Model {type(clf)} has no built-in feature importance.")

    return pd.DataFrame({"feature": feature_names, "importance": values, "method": method}).sort_values(
        "importance", ascending=False
    )


def permutation_feature_importance(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Permutation importance on held-out or full data."""
    result = permutation_importance(
        pipeline,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )
    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance": result.importances_mean,
                "std": result.importances_std,
                "method": "permutation",
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def shap_feature_importance(
    pipeline,
    X: pd.DataFrame,
    max_samples: int = 500,
) -> pd.DataFrame:
    """
    Mean |SHAP| per feature.

    Uses TreeExplainer for tree models; falls back to KernelExplainer for others.
    """
    clf = _get_classifier(pipeline)
    X_sample = X.sample(min(max_samples, len(X)), random_state=42)
    scaler = pipeline.named_steps.get("scaler") if hasattr(pipeline, "named_steps") else None
    X_in = scaler.transform(X_sample) if scaler is not None else X_sample.values

    if hasattr(clf, "feature_importances_"):
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_in)
        if isinstance(shap_values, list):
            values = np.abs(shap_values[1]).mean(axis=0)
        else:
            arr = np.asarray(shap_values)
            if arr.ndim == 3:
                class_vals = arr[:, :, 1] if arr.shape[2] > 1 else arr[:, :, 0]
                values = np.abs(class_vals).mean(axis=0)
            else:
                values = np.abs(arr).mean(axis=0)
    else:
        background = shap.sample(X, min(100, len(X)), random_state=42)
        explainer = shap.KernelExplainer(clf.predict_proba, background)
        shap_values = explainer.shap_values(X_sample, nsamples=100)
        values = np.abs(shap_values[1]).mean(axis=0)

    return (
        pd.DataFrame({"feature": X.columns, "importance": values, "method": "shap"})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def aggregate_to_residues(importance_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Sum feature importances per amino-acid residue (across interaction types)."""
    residue_scores: dict[str, float] = defaultdict(float)
    residue_meta: dict[str, tuple[str, int]] = {}

    for _, row in importance_df.iterrows():
        residue = feature_to_residue_key(row["feature"])
        residue_scores[residue] += row["importance"]
        parsed = parse_feature_name(row["feature"])
        if parsed:
            residue_meta[residue] = (parsed[0], parsed[1])

    records = []
    for residue, score in residue_scores.items():
        aa, pos = residue_meta.get(residue, ("?", 0))
        records.append({"residue": residue, "aa": aa, "position": pos, "importance": score})

    out = pd.DataFrame(records).sort_values("importance", ascending=False)
    return out.head(top_n).reset_index(drop=True)


def top_features_by_method(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    top_n: int = 20,
    include_shap: bool = True,
    include_permutation: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Compute interpretability rankings.

    Primary comparison: feature importance (model-native) vs SHAP.
    """
    outputs: dict[str, pd.DataFrame] = {}

    native = model_feature_importance(pipeline, list(X.columns))
    outputs["native"] = native.head(top_n)

    if include_permutation:
        perm = permutation_feature_importance(pipeline, X, y)
        outputs["permutation"] = perm.head(top_n)

    if include_shap:
        shap_df = shap_feature_importance(pipeline, X)
        outputs["shap"] = shap_df.head(top_n)

    for method, df in list(outputs.items()):
        outputs[f"{method}_residues"] = aggregate_to_residues(df, top_n=top_n)

    return outputs


def _compare_residue_sets(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    top_n: int,
    left_key: str,
    right_key: str,
) -> dict:
    """Jaccard overlap between top-N residue rankings from two methods."""
    left_set = set(left_df.head(top_n)["residue"])
    right_set = set(right_df.head(top_n)["residue"])
    overlap = left_set & right_set
    union = left_set | right_set

    return {
        f"{left_key}_top": sorted(left_set),
        f"{right_key}_top": sorted(right_set),
        "overlap": sorted(overlap),
        "n_overlap": len(overlap),
        "jaccard": len(overlap) / len(union) if union else 0.0,
        f"{left_key}_only": sorted(left_set - right_set),
        f"{right_key}_only": sorted(right_set - left_set),
    }


def compare_fi_vs_shap(interp: dict[str, pd.DataFrame], top_n: int = 15) -> dict:
    """Compare top residues from feature importance vs SHAP (hypothesis 1)."""
    if "native_residues" not in interp or "shap_residues" not in interp:
        raise ValueError("Both native_residues and shap_residues are required.")

    return _compare_residue_sets(
        interp["native_residues"],
        interp["shap_residues"],
        top_n,
        left_key="fi",
        right_key="shap",
    )


def compare_shap_vs_permutation(interp: dict[str, pd.DataFrame], top_n: int = 15) -> dict:
    """Compare top residues from SHAP vs permutation importance."""
    if "shap_residues" not in interp or "permutation_residues" not in interp:
        raise ValueError("Both shap_residues and permutation_residues are required.")

    return _compare_residue_sets(
        interp["shap_residues"],
        interp["permutation_residues"],
        top_n,
        left_key="shap",
        right_key="permutation",
    )


def compare_method_overlap(
    residue_rankings: dict[str, pd.DataFrame],
    top_n: int = 15,
    methods: tuple[str, str] = ("native_residues", "shap_residues"),
) -> pd.DataFrame:
    """Jaccard overlap between two interpretability methods (default: FI vs SHAP)."""
    if methods[0] not in residue_rankings or methods[1] not in residue_rankings:
        available = [m for m in residue_rankings if m.endswith("_residues")]
        if len(available) < 2:
            return pd.DataFrame()
        methods = (available[0], available[1])

    sets = {m: set(residue_rankings[m].head(top_n)["residue"]) for m in methods}
    inter = len(sets[methods[0]] & sets[methods[1]])
    union = len(sets[methods[0]] | sets[methods[1]])
    jaccard = inter / union if union else 0.0

    return pd.DataFrame(
        [
            {
                "method_1": methods[0],
                "method_2": methods[1],
                "overlap": inter,
                "jaccard": jaccard,
            }
        ]
    )
